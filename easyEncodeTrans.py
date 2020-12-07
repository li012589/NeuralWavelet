import numpy as np
import argparse, json, math
import os, glob

import flow, utils, source

import torch, torchvision
from torch import nn

from encoder import rans, coder
from utils import cdfDiscreteLogitstic, cdfMixDiscreteLogistic

torch.manual_seed(42)

parser = argparse.ArgumentParser(description="")

parser.add_argument("-folder", default=None, help="Path to load the trained model")
parser.add_argument("-cuda", type=int, default=-1, help="Which device to use with -1 standing for CPU, number bigger than -1 is N.O. of GPU.")
parser.add_argument("-nbins", type=int, default=4096, help="bin number of ran")
parser.add_argument("-precision", type=int, default=24, help="precision of CDF")
parser.add_argument("-earlyStop", type=int, default=10, help="fewer batch of testing")
parser.add_argument("-best", action='store_false', help="if load the best model")
parser.add_argument('-target', type=str, default='CIFAR', choices=['CIFAR', 'ImageNet32', 'ImageNet64', 'MNIST'], metavar='DATASET', help='Dataset choice.')
parser.add_argument("-batch", type=int, default=-1, help="batch size")


args = parser.parse_args()

device = torch.device("cpu" if args.cuda < 0 else "cuda:" + str(args.cuda))

if args.folder is None:
    raise Exception("No loading")
else:
    rootFolder = args.folder
    if rootFolder[-1] != '/':
        rootFolder += '/'
    with open(rootFolder + "parameter.json", 'r') as f:
        config = json.load(f)
        locals().update(config)

        target = config['target']
        repeat = config['repeat']
        nhidden = config['nhidden']
        hchnl = config['hchnl']
        nMixing = config['nMixing']
        simplePrior = config['simplePrior']
        batch = config['batch']


target = args.target
if args.batch != -1:
    batch = args.batch

# Building the target dataset
if target == "CIFAR":
    # Define dimensions
    targetSize = [3, 32, 32]
    dimensional = 2
    channel = targetSize[0]
    blockLength = targetSize[-1]

    # Define nomaliziation and decimal
    decimal = flow.ScalingNshifting(256, -128)
    rounding = utils.roundingWidentityGradient

    # Building train & test datasets
    lambd = lambda x: (x * 255).byte().to(torch.float32).to(device)
    trainsetTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambd)])
    trainTarget = torchvision.datasets.CIFAR10(root='./data/cifar', train=True, download=True, transform=trainsetTransform)
    testTarget = torchvision.datasets.CIFAR10(root='./data/cifar', train=False, download=True, transform=trainsetTransform)
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=batch, shuffle=True)
    targetTestLoader = torch.utils.data.DataLoader(testTarget, batch_size=batch, shuffle=True)
elif args.target == "ImageNet32":
    # Define dimensions
    targetSize = [3, 32, 32]
    dimensional = 2
    channel = targetSize[0]
    blockLength = targetSize[-1]

    # Define nomaliziation and decimal
    decimal = flow.ScalingNshifting(256, -128)
    rounding = utils.roundingWidentityGradient

    # Building train & test datasets
    lambd = lambda x: (x * 255).byte().to(torch.float32).to(device)
    trainsetTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambd)])
    trainTarget = utils.ImageNet(root='./data/ImageNet32', train=True, download=True, transform=trainsetTransform)
    testTarget = utils.ImageNet(root='./data/ImageNet32', train=False, download=True, transform=trainsetTransform)
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=batch, shuffle=True)
    targetTestLoader = torch.utils.data.DataLoader(testTarget, batch_size=batch, shuffle=True)

elif args.target == "ImageNet64":
    # Define dimensions
    targetSize = [3, 64, 64]
    dimensional = 2
    channel = targetSize[0]
    blockLength = targetSize[-1]

    # Define nomaliziation and decimal
    decimal = flow.ScalingNshifting(256, -128)
    rounding = utils.roundingWidentityGradient

    # Building train & test datasets
    lambd = lambda x: (x * 255).byte().to(torch.float32).to(device)
    trainsetTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambd)])
    trainTarget = utils.ImageNet(root='./data/ImageNet64', train=True, download=True, transform=trainsetTransform, d64=True)
    testTarget = utils.ImageNet(root='./data/ImageNet64', train=False, download=True, transform=trainsetTransform, d64=True)
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=batch, shuffle=True)
    targetTestLoader = torch.utils.data.DataLoader(testTarget, batch_size=batch, shuffle=True)

elif args.target == "MNIST":
    pass
else:
    raise Exception("No such target")

# decide which model to load
if args.best:
    name = max(glob.iglob(os.path.join(rootFolder, '*.saving')), key=os.path.getctime)
else:
    name = max(glob.iglob(os.path.join(rootFolder, 'savings', '*.saving')), key=os.path.getctime)

# load the model
print("load saving at " + name)
loadedF = torch.load(name, map_location=device)

if 'easyMera' in name:
    layerList = loadedF.layerList[:(4 * repeat)]
    layerList = [layerList[no] for no in range(4 * repeat)]
elif '1to2Mera' in name:
    layerList = loadedF.layerList[:(2 * repeat)]
    layerList = [layerList[no] for no in range(2 * repeat)]
else:
    raise Exception("model not define")

dimensional = 2
channel = targetSize[0]
blockLength = targetSize[-1]

# Define nomaliziation and decimal
if 'easyMera' in name:
    decimal = flow.ScalingNshifting(256, -128)
elif '1to2Mera' in name:
    decimal = flow.ScalingNshifting(256, 0)
else:
    raise Exception("model not define")

if 'simplePrior_False' in name:
    meanNNlist = [loadedF.meanNNlist[0]]
    scaleNNlist = [loadedF.scaleNNlist[0]]
else:
    meanNNlist = None
    scaleNNlist = None

rounding = utils.roundingWidentityGradient

# Building MERA mode
if 'easyMera' in name:
    f = flow.SimpleMERA(blockLength, layerList, meanNNlist, scaleNNlist, repeat, None, nMixing, decimal=decimal, rounding=utils.roundingWidentityGradient).to(device)
elif '1to2Mera' in name:
    f = flow.OneToTwoMERA(blockLength, layerList, meanNNlist, scaleNNlist, repeat, None, nMixing, decimal=decimal, rounding=utils.roundingWidentityGradient).to(device)

tmpLine = targetSize[-1] ** 2 // 4
shapeList = []
while tmpLine != 1:
    shapeList.append([3, tmpLine, 3])
    tmpLine = tmpLine // 4

shapeList.append([3, 1, 4])


def grp2im(t):
    return t.reshape(t.shape[0], t.shape[1], int(t.shape[2] ** 0.5), int(t.shape[2] ** 0.5), 2, 2).permute([0, 1, 2, 4, 3, 5]).reshape(t.shape[0], t.shape[1], int(t.shape[2] ** 0.5) * 2, int(t.shape[2] ** 0.5) * 2)


def join(rcnZ):
    for no in reversed(range(len(rcnZ))):
        if no == len(rcnZ) - 1:
            ul = grp2im(rcnZ[no])
        else:
            rcnZpart = rcnZ[no].reshape(*ul.shape, 3)
            ul = ul.reshape(*ul.shape, 1)
            _x = torch.cat([ul, rcnZpart], -1).reshape(*ul.shape[:2], -1, 4)
            ul = grp2im(_x).contiguous()
    return ul


def encode(z):
    _mean = []
    _cdf = []
    state = [None for _ in range(z.shape[0])]
    theoryBPD = []
    count = 0
    for i in range(z.shape[0]):
        _zpart = z[i]

        mean = int(0.5 * (_zpart.max() + _zpart.min()))
        _mean.append(mean)

        cha, freq = np.unique((_zpart - mean).detach().numpy() + args.nbins // 2, return_counts=True)
        histogram = np.zeros(args.nbins)
        histogram[cha.astype(int)] = freq

        prob = histogram / np.prod(_zpart.shape)
        logp = -np.log2(prob)
        logp[logp == float('inf')] = 0
        theorySize = (logp * histogram).sum().item() #+ cha.shape[0] * 32
        theoryBPD.append(theorySize / np.prod(_zpart.shape))

        cdf = prob
        for j in range(prob.shape[0] - 1):
            cdf[j + 1] = cdf[j] + cdf[j + 1]

        sCDF = (cdf * ((1 << args.precision))).astype('int').reshape(args.nbins)
        _cdf.append(sCDF)

        symbols = (_zpart.reshape(-1) - mean + args.nbins // 2 - 1).int().numpy()
        if state[i] is None:
            s = rans.x_init
            count += 1
        else:
            s = state[i]
        for j in reversed(range(len(symbols))):
            s = coder.encoder(sCDF, symbols[j], s)
        state[i] = s

    state = [rans.flatten(term) for term in state]
    return _cdf, _mean, state, np.mean(theoryBPD)


def decode(CDF, MEAN, targetSize, state):
    rcnPart = []
    state = [rans.unflatten(term) for term in state]
    rcn = [[] for _ in range(len(state))]
    for i in range(len(state)):
        s = state[i]
        for j in range(np.prod(targetSize)):
            s, rcnSymbol = coder.decoder(CDF[i], s)
            state[i] = s
            rcn[i].append(rcnSymbol + MEAN[i] - args.nbins // 2 + 1)

    rcn = torch.tensor(rcn).reshape(-1, *targetSize)
    rcnPart.append(rcn)

    retZ = torch.cat(rcnPart, 0)
    return retZ


def cdf2int(cdf):
    return (cdf * ((1 << args.precision) - args.nbins)).int().detach() + torch.arange(args.nbins).reshape(-1, 1, 1, 1)


def testBPD(loader, earlyStop=-1):
    actualBPD = []
    theoryBPD = []
    ERR = []

    count = 0
    for samples, _ in loader:
        count += 1
        z, _ = f.inverse(samples)

        if f.meanNNlist is not None:
            baseline = [torch.round(decimal.forward_(term)) for term in f.meanList] + [torch.round(decimal.forward_(loadedF.prior.lastPrior.mean.mean(0, keepdim=True).repeat([batch, 1, 1, 1])))]
        else:
            baseline = [torch.round(decimal.forward_(loadedF.prior.priorList[0].mean.repeat([batch, 1, 1, 1]))) for _ in range(f.depth - 1)] + [torch.round(decimal.forward_(loadedF.prior.priorList[-1].mean.mean(0, keepdim=True).repeat([batch, 1, 1, 1])))]

        baseline = join(baseline)

        CDF, MEAN, state, tBPD = encode(z - baseline)

        actualBPD.append(32 / (np.prod(samples.shape[1:])) * np.mean([s.shape[0] for s in state]))
        theoryBPD.append(tBPD)

        rcnZ = decode(CDF, MEAN, targetSize, state)

        rcnZ = rcnZ + baseline

        rcnSamples, _ = f.forward(rcnZ.float())

        ERR.append(torch.abs(samples - rcnSamples).sum().item())
        if count >= earlyStop and earlyStop > 0:
            break

    actualBPD = np.array(actualBPD)
    theoryBPD = np.array(theoryBPD)
    ERR = np.array(ERR)

    print("===========================SUMMARY==================================")
    print("Actual Mean BPD:", actualBPD.mean(), "Theory Mean BPD:", theoryBPD.mean(), "Mean Error:", ERR.mean())

    return actualBPD, theoryBPD, ERR


print("Train Set:")
testBPD(targetTrainLoader, earlyStop=args.earlyStop)
print("Test Set:")
testBPD(targetTestLoader, earlyStop=args.earlyStop)

