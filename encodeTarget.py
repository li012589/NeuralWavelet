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

_pbase = int(math.log(blockLength, 2)) - len(loadedF.prior.priorList)
plen = list(range(_pbase, int(math.log(blockLength, 2))))
baselen = list(range(_pbase))

# Building the sub-priors
# TODO: depth less than int(math.log(blockLength, 2))
priorList = []
_length = int((blockLength * blockLength) / 4)
for n in range(int(math.log(blockLength, 2))):
    if n in plen:
        priorList.append(loadedF.prior.priorList[n - _pbase])
        _length = int(_length / 4)
        continue
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


def divide(z):
    extraPart = []
    parts = []
    for no in range(int(math.log(blockLength, 2))):
        _, zpart = utils.dispatch(f.prior.factorOutIList[no], f.prior.factorOutJList[no], z)

        if no in baselen:
            extraPart.append(zpart)
            continue

        if no != int(math.log(blockLength, 2)) - 1:
            zpart = zpart - decimal.forward_(f.prior.priorList[no].mean).reshape(1, *f.prior.priorList[no].mean.shape).int() + args.nbins // 2
        else:
            zpart = zpart - (decimal.forward_(f.prior.priorList[no].mean.permute([1, 2, 3, 0])) * torch.softmax(f.prior.priorList[no].mixing, -1)).sum(-1).reshape(1, *f.prior.priorList[no].mean.shape[1:]).int() + args.nbins // 2
        parts.append(zpart.reshape(zpart.shape[0], -1).int().detach())
    return torch.cat(parts, -1).numpy(), extraPart


def join(rcnZ, extraPart):
    retZ = torch.zeros(batch, *targetSize).to(device).int()
    for no in range(int(math.log(blockLength, 2))):
        rcnZpart = rcnZ[:, :(np.prod(shapeList[no]))].reshape(rcnZ.shape[0], *shapeList[no])
        rcnZ = rcnZ[:, np.prod(shapeList[no]):]

        if no == int(math.log(blockLength, 2)) - 1:
            rcnZpart = rcnZpart + (decimal.forward_(f.prior.priorList[no].mean.permute([1, 2, 3, 0])) * torch.softmax(f.prior.priorList[no].mixing, -1)).sum(-1).reshape(1, *f.prior.priorList[no].mean.shape[1:]).int() - args.nbins // 2
        else:
            rcnZpart = rcnZpart + decimal.forward_(f.prior.priorList[no].mean).reshape(1, *f.prior.priorList[no].mean.shape).int() - args.nbins // 2

        if no in baselen:
            rcnZpart = extraPart[no]
        retZ = utils.collect(f.prior.factorOutIList[no], f.prior.factorOutJList[no], retZ, rcnZpart.int())
    return retZ


def cdf2int(cdf):
    return (cdf * ((1 << args.precision) - args.nbins)).int().detach() + torch.arange(args.nbins).reshape(-1, 1, 1, 1)


CDF = []
_bins = torch.arange(-args.nbins // 2, args.nbins // 2).reshape(-1, 1, 1, 1)
for no, prior in enumerate(f.prior.priorList):
    if no != len(f.prior.priorList) - 1:
        bins = _bins - 1 + decimal.forward_(prior.mean).reshape(1, *prior.mean.shape).int()
        cdf = cdfDiscreteLogitstic(bins, prior.mean, prior.logscale, decimal=prior.decimal)
    else:
        bins = _bins - 1 + (decimal.forward_(prior.mean.permute([1, 2, 3, 0])) * prior.mixing).sum(-1).reshape(1, *prior.mean.shape[1:]).int()
        cdf = cdfMixDiscreteLogistic(bins, prior.mean, prior.logscale, prior.mixing, decimal=prior.decimal)
    CDF.append(cdf2int(cdf).reshape(args.nbins, -1))

CDF = torch.cat(CDF, -1).numpy()


def partCoder(part, states):
    HISTos = []
    MEANs = []
    theoryBPD = []
    CDFs = []

    for i in range(part.shape[0]):
        mean = int(0.5 * (part[i].max() + part[i].min()))
        MEANs.append(mean)

        histogram = torch.histc(part[i] - mean, bins=args.nbins, min=-args.nbins // 2, max=args.nbins // 2)
        HISTos.append(histogram)

        prob = histogram / np.prod(part[i].shape)
        logp = -torch.log2(prob)
        logp[logp == float('inf')] = 0
        theorySize = (logp * histogram).sum().item()
        theoryBPD.append(theorySize / np.prod(part.shape[1:]))

        cdf = prob.detach().numpy()
        for j in range(prob.shape[0] - 1):
            cdf[j + 1] = cdf[j] + cdf[j + 1]

        # np.arange(nBins)).reshape(1, nBins)) here to avoid zero freq in ans, add extra BPD
        CDFs.append(torch.from_numpy((cdf * ((1 << args.precision))).astype('int').reshape(1, args.nbins)))

    MEANs = torch.tensor(MEANs).reshape(part.shape[0], 1)
    CDFs = torch.cat(CDFs, 0).numpy()

    print("theory BPD:", np.mean(theoryBPD))

    for i in range(part.shape[0]):
        symbols = part[i] - MEANs[i] + args.nbins // 2 - 1
        symbols = symbols.reshape(-1).int()
        state = states[i]
        for j in reversed(range(len(symbols))):
            state = coder.encoder(CDFs[i], symbols[j], state)
        state = rans.flatten(state)
        states[i] = state
    #aBPD = np.mean([32 * len(term) / np.prod(part.shape[1:]) for term in states])
    #print("actual BPD:", aBPD)

    return states, MEANs, CDFs


def partDecode(shape, MEANs, CDFs, states):
    after_states = []
    reconstruction = [[] for i in range(len(states))]
    for i in range(len(states)):
        state = rans.unflatten(states[i])
        for j in range(np.prod(shape)):
            state, recon_symbol = coder.decoder(CDFs[i], state)
            reconstruction[i].append(recon_symbol + MEANs[i] - args.nbins // 2 + 1)
        after_states.append(state)
    reconstruction = torch.tensor(reconstruction).reshape(len(states), *shape)
    return reconstruction, after_states


def testBPD(loader, earlyStop=-1):
    actualBPD = []
    theoryBPD = []
    ERR = []

    count = 0
    for samples, _ in loader:
        count += 1
        z, _ = f.inverse(samples)

        zparts, extras = divide(z)

        state = []
        for i in range(batch):
            symbols = zparts[i]
            s = rans.x_init
            for j in reversed(range(symbols.shape[-1])):
                cdf = CDF[:, j]
                s = coder.encoder(cdf, symbols[j], s)
            state.append(rans.flatten(s))

        '''
        from copy import copy, deepcopy
        backState = copy(state)
        #state = [rans.x_init for _ in range(samples.shape[0])]

        SHAPEs = []
        MEANs = []
        CDFs = []
        for extra in extras:
            SHAPEs.append(extra.shape[1:])
            state, means, cdfs = partCoder(extra, state)
            MEANs.append(means)
            CDFs.append(cdfs)

        actualBPD.append(32 / (np.prod(samples.shape[1:])) * np.mean([s.shape[0] for s in state]))
        print("actualBPD:", actualBPD)
        #theoryBPD.append((-f.logProbability(samples).mean() / (np.prod(samples.shape[1:]) * np.log(2.))).detach().item())
        RCNExtraPart = []
        for t in range(len(SHAPEs)):
            rcnExtras, state = partDecode(SHAPEs[t], MEANs[t], CDFs[t], state)
            RCNExtraPart.append(rcnExtras)

        for no in range(200):
            assert (state[no] == backState[no] and id(state[no]) != id(backState[no]))

        import pdb
        pdb.set_trace()
        '''
        #state = [rans.flatten(s) for s in state]

        rcnParts = []
        for i in range(batch):
            s = state[i]
            symbols = []
            for j in range(np.prod(targetSize)):
                cdf = CDF[:, j]
                s, rcnSymbol = coder.decoder(cdf, s)
                symbols.append(rcnSymbol)
            rcnParts.append(torch.tensor(symbols).reshape(1, -1))
        rcnParts = torch.cat(rcnParts, 0)

        import pdb
        pdb.set_trace()

        rcnZ = join(rcnParts)

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

