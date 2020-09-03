import numpy as np
import argparse, json, math
import os, glob

import flow, utils, source

import torch, torchvision
from torch import nn

from encoder import rans, coder
from utils import cdfDiscreteLogitstic, cdfMixDiscreteLogistic


parser = argparse.ArgumentParser(description="")

parser.add_argument("-folder", default=None, help="Path to load the trained model")
parser.add_argument("-cuda", type=int, default=-1, help="Which device to use with -1 standing for CPU, number bigger than -1 is N.O. of GPU.")
parser.add_argument("-nbins", type=int, default=4096, help="bin number of ran")
parser.add_argument("-best", action='store_false', help="if load the best model")


args = parser.parse_args()

device = torch.device("cpu" if args.cuda < 0 else "cuda:" + str(args.cuda))

if args.folder is None:
    raise Exception("No loading")
else:
    rootFolder = args.folder
    if rootFolder[-1] != '/':
        rootFolder += '/'
    with open(rootFolder + "/parameter.json", 'r') as f:
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
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=batch, shuffle=False)
    targetTestLoader = torch.utils.data.DataLoader(testTarget, batch_size=batch, shuffle=False)
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
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=batch, shuffle=False)
    targetTestLoader = torch.utils.data.DataLoader(testTarget, batch_size=batch, shuffle=False)

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
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=batch, shuffle=False)
    targetTestLoader = torch.utils.data.DataLoader(testTarget, batch_size=batch, shuffle=False)

elif args.target == "MNIST":
    pass
else:
    raise Exception("No such target")

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

layerList = []
if depth == -1:
    depth = None
# NOTE HERE: Same wavelet at each RG scale. If want different wavelet, change (repeat + 1)
#            to depth * (repeat + 1)!
for _ in range(repeat + 1):
    maskList = []
    for n in range(nNICE):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(3 * 2 * 1), torch.ones(3 * 2 * 1)])[torch.randperm(3 * 2 * 2)].reshape(1, 3, 2, 2)
        else:
            b = 1 - b
        maskList.append(b)
    maskList = torch.cat(maskList, 0)#.to(torch.float32)
    tList = [utils.SimpleMLPreshape([3 * 2 * 1] + [hdim] * nhidden + [3 * 2 * 1], [nn.ELU()] * nhidden + [None]) for _ in range(nNICE)]
    layerList.append(flow.DiscreteNICE(maskList, tList, decimal, rounding))

# Building MERA model
f = flow.MERA(dimensional, blockLength, layerList, repeat, depth=depth, prior=p).to(device)

# decide which model to load
if args.best:
    name = max(glob.iglob(os.path.join(rootFolder, '*.saving')), key=os.path.getctime)
else:
    name = max(glob.iglob(os.path.join(rootFolder, 'savings', '*.saving')), key=os.path.getctime)

# load the model
print("load saving at " + name)
f = torch.load(name).to(device)

shapeList = [np.prod(term.shape) for term in f.prior.factorOutIList]

def divide(z):
    parts = []
    for no in range(int(math.log(blockLength, 2))):
        _, zpart = utils.dispatch(f.prior.factorOutIList[no], f.prior.factorOutJList[no], z)
        parts.append(zpart)
    return parts


CDF = []
_bins = torch.arange(-args.nbins // 2, args.nbins // 2).reshape(-1, 1, 1, 1)
for no, prior in enumerate(f.prior.priorList):
    if no != len(f.prior.priorList) - 1:
        bins = _bins - 1 + decimal.forward_(prior.mean).reshape(1, *prior.mean.shape).int()
        cdf = cdfDiscreteLogitstic(bins, prior.mean, prior.logscale, decimal=prior.decimal)
    else:
        bins = _bins - 1 + (decimal.forward_(prior.mean.permute([1, 2, 3, 0])) * prior.mixing).sum(-1).reshape(1, *prior.mean.shape[1:]).int()
        cdf = cdfMixDiscreteLogistic(bins, prior.mean, prior.logscale, prior.mixing, decimal=prior.decimal)
    CDF.append(cdf)

actualBPD = []
theoryBPD = []
ERR = []

STATES = []
for samples, _ in targetTrainLoader:
    z, _ = f.inverse(samples)
    parts = divide(z)
    state = [None for _ in range(z.shape[0])]
    for no, part in enumerate(parts):
        for i in range(z.shape[0]):
            symbols = part[i].reshape(-1).int().detach()
            cdf = CDF[no].reshape(args.nbins, -1).detach()
            for j, symbol in enumerate(symbols):
                state[i] = coder.encoder(cdf[:, j], symbol, state[i])

    import pdb
    pdb.set_trace()

    actualBPD.append(32 * sum([len(s) for s in state]))
    theoryBPD.append((-f.logProbability(samples).mean() / (np.prod(samples.shape[1:]) * np.log(2.))).detach().item())
    STATES += state
    print("Train Set Actual BPD:", actualBPD[-1], "Train Set Theory BPD:", theoryBPD[-1])
    import pdb
    pdb.set_trace()

    for no in range(len(state)):
        for j, shape in enumerate(shapeList):
            rcn = []
            cdf = CDF[j].reshape(args.nbins, -1).detach()
            for i in range(shape):
                state[no], recon_symbol = coder.decoder(cdf, state[no])
                rcn.append(recon_symbol)
            rcn = torch.tensor(rcn).reshape()
    for no, cdf in reversed(enumerate(CDF)):
        for i in range(z.shape[0]):
            cdf = cdf.reshape(-1).detach().cpu().numpy()
            recon = []
            for _ in range():
                recon.append(recon_symbol)
            recon = torch.tensor(recon)

    ERR.append()

actualBPD = np.array(actualBPD)
theoryBPD = np.array(theoryBPD)
ERR = np.array(ERR)

print("===========================SUMMARY==================================")
print("Train Set Actual BPD:", actualBPD.mean(), "Train Set Theory BPD:", theoryBPD.mean(), "Train Set Error:", ERR.mean())

BPD = []
ERR = []
for samples, _ in targetTestLoader:
    pass
BPD = np.array(BPD)
ERR = np.array(ERR)

print("Test Set BPD:", BPD.mean(), "Test Set Error:", ERR.mean())
