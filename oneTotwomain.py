import numpy as np
import argparse, json, math

import torch, torchvision
from torch import nn
import matplotlib.pyplot as plt

import flow, source, train, utils
from utils import buildWaveletLayers, harrInitMethod1, harrInitMethod2, leGallInitMethod1, leGallInitMethod2


parser = argparse.ArgumentParser(description="")

group = parser.add_argument_group("Target Parameters")
group.add_argument('-target', type=str, default='CIFAR', choices=['CIFAR', 'ImageNet32', 'ImageNet64', 'MNIST'], metavar='DATASET', help='Dataset choice.')

group = parser.add_argument_group("Architecture Parameters")
group.add_argument("-repeat", type=int, default=1, help="num of disentangler layers of each RG scale")
group.add_argument("-hchnl", type=int, default=12, help="intermediate channel dimension of Conv1d inside NICE inside MERA")
group.add_argument("-nhidden", type=int, default=1, help="num of intermediate channel of Conv1d inside NICE inside MERA")
group.add_argument("-nMixing", type=int, default=5, help="num of mixing distributions of last sub-priors")

group = parser.add_argument_group('Learning  parameters')
group.add_argument('-init', type=str, default='harr', choices=['harr', 'legall'], metavar='initWavelet', help='which wavelet to init.')
group.add_argument("-epoch", type=int, default=400, help="num of epoches to train")
group.add_argument("-batch", type=int, default=200, help="batch size")
group.add_argument("-savePeriod", type=int, default=10, help="save after how many steps")
group.add_argument("-lr", type=float, default=0.001, help="learning rate")
group.add_argument("-simplePrior", action="store_true", help="if use simple version prior, no crossover")

group = parser.add_argument_group("Etc")
group.add_argument("-folder", default=None, help="Path to save")
group.add_argument("-cuda", type=int, default=-1, help="Which device to use with -1 standing for CPU, number bigger than -1 is N.O. of GPU.")
group.add_argument("-load", action='store_true', help="If load or not")

args = parser.parse_args()

device = torch.device("cpu" if args.cuda < 0 else "cuda:" + str(args.cuda))

# Creating save folder
if args.folder is None:
    rootFolder = './opt/default_1to2Mera_' + args.target + "_simplePrior_" + str(args.simplePrior) + "_initWavelet_" + args.init + "_repeat_" + str(args.repeat) + "_hchnl_" + str(args.hchnl) + "_nhidden_" + str(args.nhidden) + "_nMixing_" + str(args.nMixing) + "/"
    print("No specified saving path, using", rootFolder)
else:
    rootFolder = args.folder
if rootFolder[-1] != '/':
    rootFolder += '/'
utils.createWorkSpace(rootFolder)

# Decoding parameters to mem, saving them to save folder.
if not args.load:
    target = args.target
    init = args.init
    repeat = args.repeat
    hchnl = args.hchnl
    nhidden = args.nhidden
    nMixing = args.nMixing
    epoch = args.epoch
    batch = args.batch
    savePeriod = args.savePeriod
    simplePrior = args.simplePrior
    lr = args.lr
    with open(rootFolder + "/parameter.json", "w") as f:
        config = {'target': target, 'repeat': repeat, 'init': init, 'hchnl': hchnl, 'nhidden': nhidden, 'nMixing': nMixing, 'epoch': epoch, 'batch': batch, 'savePeriod': savePeriod, 'lr': lr, 'simplePrior': simplePrior}
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
    decimal = flow.ScalingNshifting(256, 0)
    rounding = utils.roundingWidentityGradient

    # Building train & test datasets
    lambd = lambda x: (x * 255).byte().to(torch.float32).to(device)
    trainsetTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambd)])
    trainTarget = torchvision.datasets.CIFAR10(root='./data/cifar', train=True, download=True, transform=trainsetTransform)
    testTarget = torchvision.datasets.CIFAR10(root='./data/cifar', train=False, download=True, transform=trainsetTransform)
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=batch, shuffle=True)
    targetTestLoader = torch.utils.data.DataLoader(testTarget, batch_size=batch, shuffle=False)
elif target == "ImageNet32":
    # Define dimensions
    targetSize = [3, 32, 32]
    dimensional = 2
    channel = targetSize[0]
    blockLength = targetSize[-1]

    # Define nomaliziation and decimal
    decimal = flow.ScalingNshifting(256, 0)
    rounding = utils.roundingWidentityGradient

    # Building train & test datasets
    lambd = lambda x: (x * 255).byte().to(torch.float32).to(device)
    trainsetTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambd)])
    trainTarget = utils.ImageNet(root='./data/ImageNet32', train=True, download=True, transform=trainsetTransform)
    testTarget = utils.ImageNet(root='./data/ImageNet32', train=False, download=True, transform=trainsetTransform)
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=batch, shuffle=True)
    targetTestLoader = torch.utils.data.DataLoader(testTarget, batch_size=batch, shuffle=False)

elif target == "ImageNet64":
    # Define dimensions
    targetSize = [3, 64, 64]
    dimensional = 2
    channel = targetSize[0]
    blockLength = targetSize[-1]

    # Define nomaliziation and decimal
    decimal = flow.ScalingNshifting(256, 0)
    rounding = utils.roundingWidentityGradient

    # Building train & test datasets
    lambd = lambda x: (x * 255).byte().to(torch.float32).to(device)
    trainsetTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambd)])
    trainTarget = utils.ImageNet(root='./data/ImageNet64', train=True, download=True, transform=trainsetTransform, d64=True)
    testTarget = utils.ImageNet(root='./data/ImageNet64', train=False, download=True, transform=trainsetTransform, d64=True)
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=batch, shuffle=True)
    targetTestLoader = torch.utils.data.DataLoader(testTarget, batch_size=batch, shuffle=False)

elif target == "MNIST":
    pass
else:
    raise Exception("No such target")


def buildLayers2D(shapeList):
    layers = []
    for no, chn in enumerate(shapeList[:-1]):
        if no != 0 and no != len(shapeList) - 2:
            layers.append(torch.nn.Conv2d(chn, shapeList[no + 1], 1))
        else:
            layers.append(torch.nn.Conv2d(chn, shapeList[no + 1], 3, padding=1))
        if no != len(shapeList) - 2:
            layers.append(torch.nn.ReLU(inplace=True))
    return layers


def buildLayers1D(shapeList):
    layers = []
    for no, chn in enumerate(shapeList[:-1]):
        if no != 0 and no != len(shapeList) - 2:
            layers.append(torch.nn.Conv1d(chn, shapeList[no + 1], 1))
        else:
            layers.append(torch.nn.Conv1d(chn, shapeList[no + 1], 3, padding=1))
        if no != len(shapeList) - 2:
            layers.append(torch.nn.ReLU(inplace=True))
    layers = torch.nn.Sequential(*layers)
    torch.nn.init.zeros_(layers[-1].weight)
    torch.nn.init.zeros_(layers[-1].bias)
    return layers


initMethods = []
if init == 'harr':
    initMethods.append(lambda: harrInitMethod1(targetSize[0]))
    initMethods.append(lambda: harrInitMethod2(targetSize[0]))
elif init == 'legall':
    initMethods.append(lambda: leGallInitMethod1(targetSize[0]))
    initMethods.append(lambda: leGallInitMethod2(targetSize[0]))
else:
    raise Exception("No such wavelet")

orders = [True, False]

shapeList1D = [targetSize[0]] + [hchnl] * (nhidden + 1) + [targetSize[0]]

layerList = []
for i in range(repeat):
    for j in range(2):
        layerList.append(buildWaveletLayers(initMethods[j], targetSize[0], hchnl, nhidden, orders[j]))

for i in range(repeat * (repeat - 1)):
    layerList.append(buildLayers1D(shapeList1D))

shapeList2D = [targetSize[0]] + [hchnl] * (nhidden + 1) + [targetSize[0] * 3]

if not simplePrior:
    meanNNlist = []
    scaleNNlist = []
    layers = buildLayers2D(shapeList2D)
    meanNNlist.append(torch.nn.Sequential(*layers))
    layers = buildLayers2D(shapeList2D)
    scaleNNlist.append(torch.nn.Sequential(*layers))
    torch.nn.init.zeros_(meanNNlist[-1][-1].weight)
    torch.nn.init.zeros_(meanNNlist[-1][-1].bias)
    torch.nn.init.zeros_(scaleNNlist[-1][-1].weight)
    torch.nn.init.zeros_(scaleNNlist[-1][-1].bias)
else:
    meanNNlist = None
    scaleNNlist = None

f = flow.OneToTwoMERA(blockLength, layerList, meanNNlist, scaleNNlist, repeat, None, nMixing, decimal=decimal, rounding=utils.roundingWidentityGradient).to(device)


# Define plot function
def plotfn(f, train, test, LOSS, VALLOSS):
    # loss plot
    lossfig = plt.figure(figsize=(8, 5))
    lossax = lossfig.add_subplot(111)

    epoch = len(LOSS)
    lossax.plot(np.arange(epoch), np.array(LOSS), 'go-', label="loss", markersize=2.5)
    lossax.plot(np.arange(epoch), np.array(VALLOSS), 'ro-', label="val. loss", markersize=2.5)

    lossax.set_xlim(0, epoch)
    lossax.legend()
    lossax.set_title("Loss Curve")
    plt.savefig(rootFolder + 'pic/lossCurve.png', bbox_inches="tight", pad_inches=0)
    plt.close()


# Training
f = train.forwardKLD(f, targetTrainLoader, targetTestLoader, epoch, lr, savePeriod, rootFolder, plotfn=plotfn)

# Pasuse
import pdb
pdb.set_trace()
