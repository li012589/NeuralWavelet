import numpy as np
import argparse, json, math

import torch, torchvision
from torch import nn
import matplotlib.pyplot as plt

import flow, source, train, utils


parser = argparse.ArgumentParser(description="")

group = parser.add_argument_group("Target Parameters")
group.add_argument('-target', type=str, default='CIFAR', choices=['CIFAR', 'ImageNet', 'MNIST'], metavar='DATASET', help='Dataset choice.')

group = parser.add_argument_group("Architecture Parameters")
group.add_argument("-depth", type=int, default=-1, help="depth of hierarchy structure, -1 means full depth")
group.add_argument("-repeat", type=int, default=1, help="num of disentangler layers of each RG scale")
group.add_argument("-nhidden", type=int, default=3, help="num of MLP layers inside NICE inside MERA")
group.add_argument("-hdim", type=int, default=50, help="layer dimension of MLP inside NICE inside MERA")
group.add_argument("-nNICE", type=int, default=2, help="num of NICE layers of each RG scale (even number only)")
group.add_argument("-nMixing", type=int, default=5, help="num of mixing distributions of last sub-priors")

group = parser.add_argument_group('Learning  parameters')
group.add_argument("-epoch", type=int, default=400, help="num of epoches to train")
group.add_argument("-batch", type=int, default=200, help="batch size")
group.add_argument("-savePeriod", type=int, default=10, help="save after how many steps")
group.add_argument("-lr", type=float, default=0.001, help="learning rate")

group = parser.add_argument_group("Etc")
group.add_argument("-folder", default=None, help="Path to save")
group.add_argument("-cuda", type=int, default=-1, help="Which device to use with -1 standing for CPU, number bigger than -1 is N.O. of GPU.")
group.add_argument("-load", action='store_true', help="If load or not")

args = parser.parse_args()

device = torch.device("cpu" if args.cuda < 0 else "cuda:" + str(args.cuda))

# Creating save folder
if args.folder is None:
    rootFolder = './opt/default_' + args.target + "_depth_" + str(args.depth) + "_repeat_" + str(args.repeat) + "_nhidden_" + str(args.nhidden) + "_hdim_" + str(args.hdim) + "/"
    print("No specified saving path, using", rootFolder)
else:
    rootFolder = args.folder
if rootFolder[-1] != '/':
    rootFolder += '/'
utils.createWorkSpace(rootFolder)

# Decoding parameters to mem, saving them to save folder.
if not args.load:
    target = args.target
    depth = args.depth
    repeat = args.repeat
    nhidden = args.nhidden
    hdim = args.hdim
    nNICE = args.nNICE
    nMixing = args.nMixing
    epoch = args.epoch
    batch = args.batch
    savePeriod = args.savePeriod
    lr = args.lr
    with open(rootFolder + "/parameter.json", "w") as f:
        config = {'target': target, 'depth': depth, 'repeat': repeat, 'nhidden': nhidden, 'hdim': hdim, 'nNICE': nNICE, 'nMixing': nMixing, 'epoch': epoch, 'batch': batch, 'savePeriod': savePeriod, 'lr': lr}
        json.dump(config, f)
else:
    # load saved parameters, and decoding them to mem
    with open(rootFolder + "/parameter.json", 'r') as f:
        config = json.load(f)
        locals().update(config)

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
    lambd = lambda x: (x * 255).byte().to(torch.float32)
    trainsetTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambd)])
    trainTarget = torchvision.datasets.CIFAR10(root='./data/cifar', train=True, download=True, transform=trainsetTransform)
    testTarget = torchvision.datasets.CIFAR10(root='./data/cifar', train=False, download=True, transform=trainsetTransform)
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=batch, shuffle=True)
    targetTestLoader = torch.utils.data.DataLoader(testTarget, batch_size=batch, shuffle=True)
elif args.target == "ImageNet":
    pass
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
        priorList.append(source.DiscreteLogistic([channel, _length, 3], decimal, rounding).to(device))
    else:
        # final variable prior, all 4 variable
        priorList.append(source.MixtureDiscreteLogistic([channel, _length, 4], nMixing, decimal, rounding).to(device))
    _length = int(_length / 4)

# Building the hierarchy prior
p = source.HierarchyPrior(channel, blockLength, priorList)

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


# Define plot function
def plotfn(f, train, test, LOSS, VALLOSS):
    # loss plot
    lossfig = plt.figure(figsize=(8, 5))
    lossax = lossfig.add_subplot(111)

    epoch = len(LOSS)
    lossax.plot(np.arange(epoch), np.array(LOSS), 'go-', label="loss")
    # lossax.plot(np.arange(epoch), np.array(VALLOSS), 'ro-', label="val. loss")

    lossax.set_xlim(0, epoch)
    lossax.legend()
    lossax.set_title("Loss Curve")
    plt.savefig(rootFolder + 'pic/lossCurve.png', bbox_inches="tight", pad_inches=0)
    plt.close()

    # TODO: wavelet plot, Fig. 3 in draft


# Training
f = train.forwardKLD(f, targetTrainLoader, targetTestLoader, epoch, lr, savePeriod, rootFolder, device, targetSize, plotfn=plotfn)

# Pasuse
import pdb
pdb.set_trace()
