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
        depth = config['depth']
        repeat = config['repeat']
        nhidden = config['nhidden']
        hdim = config['hdim']
        nNICE = config['nNICE']
        nMixing = config['nMixing']
        smallPrior = config['smallPrior']
        batch = config['batch']

target = args.target

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

layerList = loadedF.layerList[:(repeat + 1)]
layerList = [layerList[no] for no in range(repeat + 1)]

# Building the sub-priors
# TODO: depth less than int(math.log(blockLength, 2))
priorList = []
_length = int((blockLength * blockLength) / 4)
for n in range(int(math.log(blockLength, 2))):
    if n != (int(math.log(blockLength, 2))) - 1:
        # intermedia variable prior, 3 here means the left 3 variable
        if smallPrior:
            priorList.append(source.DiscreteLogistic([channel, 1, 3], decimal, rounding))
        else:
            priorList.append(source.DiscreteLogistic([channel, _length, 3], decimal, rounding))
    elif n == depth - 1:
        # if depth is specified, the last prior
        priorList.append(source.MixtureDiscreteLogistic([channel, _length, 4], nMixing, decimal, rounding))
        break
    else:
        # final variable prior, all 4 variable
        priorList.append(source.MixtureDiscreteLogistic([channel, _length, 4], nMixing, decimal, rounding))
    _length = int(_length / 4)

# Building the hierarchy prior
p = source.HierarchyPrior(channel, blockLength, priorList, repeat=repeat)


# Building NICE model inside MERA
assert nNICE % 2 == 0
assert depth <= int(math.log(blockLength, 2))

if depth == -1:
    depth = None

# Building MERA model
f = flow.MERA(dimensional, blockLength, layerList, repeat, depth=depth, prior=p).to(device)

shapeList = [[3, *term.shape] for term in f.prior.factorOutIList]


def encode(z):
    parts = []
    CDF = []
    theoryBPD = []
    MEAN = []
    state = [None for _ in range(z.shape[0])]
    SHAPE = []

    for no in range(int(math.log(blockLength, 2))):
        _, zpart = utils.dispatch(f.prior.factorOutIList[no], f.prior.factorOutJList[no], z)

        _CDF = []
        _MEAN = []
        SHAPE.append(zpart.shape[1:])
        for i in range(zpart.shape[0]):

            mean = int(0.5 * (zpart[i].max() + zpart[i].min()))
            _MEAN.append(mean)
            histogram = torch.histc(zpart[i] - mean, bins=args.nbins, min=-args.nbins // 2, max=args.nbins // 2)

            prob = histogram / np.prod(zpart[i].shape)
            logp = -torch.log2(prob)
            logp[logp == float('inf')] = 0
            theorySize = (logp * histogram).sum().item()
            theoryBPD.append(theorySize / np.prod(zpart[i].shape))

            cdf = prob.detach().numpy()
            for j in range(prob.shape[0] - 1):
                cdf[j + 1] = cdf[j] + cdf[j + 1]

            _CDF.append((cdf * ((1 << args.precision))).astype('int').reshape(args.nbins))

            symbols = (zpart[i].reshape(-1) - mean + args.nbins // 2 - 1).int().numpy()
            if state[i] is None:
                s = rans.x_init
            else:
                s = state[i]
            for j in reversed(range(len(symbols))):
                s = coder.encoder(_CDF[-1], symbols[j], s)
            state[i] = s

        MEAN.append(_MEAN)
        CDF.append(_CDF)
        parts.append(zpart.reshape(zpart.shape[0], -1).int().detach())
    state = [rans.flatten(term) for term in state]
    return CDF, MEAN, SHAPE, state, np.mean(theoryBPD)


def decode(CDF, MEAN, SHAPE, state):
    rcnPart = []
    state = [rans.unflatten(term) for term in state]
    for no in reversed(range(int(math.log(blockLength, 2)))):
        rcn = [[] for _ in range(len(state))]
        for i in range(len(state)):
            s = state[i]
            for j in range(np.prod(SHAPE[no])):
                s, rcnSymbol = coder.decoder(CDF[no][i], s)
                state[i] = s
                rcn[i].append(rcnSymbol + MEAN[no][i] - args.nbins // 2 + 1)

        rcn = torch.tensor(rcn).reshape(-1, *SHAPE[no])
        rcnPart.append(rcn)
    rcnPart = list(reversed(rcnPart))
    retZ = torch.zeros(batch, *targetSize).to(device).int()
    for no in range(int(math.log(blockLength, 2))):
        retZ = utils.collect(f.prior.factorOutIList[no], f.prior.factorOutJList[no], retZ, rcnPart[no].int())
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

        CDF, MEAN, SHAPE, state, tBPD = encode(z)

        actualBPD.append(32 / (np.prod(samples.shape[1:])) * np.mean([s.shape[0] for s in state]))
        theoryBPD.append(tBPD)

        rcnZ = decode(CDF, MEAN, SHAPE, state)

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

