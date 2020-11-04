import numpy as np
import argparse, json, math

import torch, torchvision
from torch import nn
import matplotlib.pyplot as plt

import flow, source, train, utils


parser = argparse.ArgumentParser(description="")

group = parser.add_argument_group("Target Parameters")
group.add_argument('-target', type=str, default='CIFAR', choices=['CIFAR', 'ImageNet32', 'ImageNet64', 'MNIST'], metavar='DATASET', help='Dataset choice.')

group = parser.add_argument_group("Architecture Parameters")
group.add_argument("-repeat", type=int, default=1, help="num of disentangler layers of each RG scale")
group.add_argument("-hchnl", type=int, default=12, help="intermediate channel dimension of Conv2d inside NICE inside MERA")
group.add_argument("-nhidden", type=int, default=1, help="num of intermediate channel of Conv2d inside NICE inside MERA")
group.add_argument("-nMixing", type=int, default=5, help="num of mixing distributions of last sub-priors")

group = parser.add_argument_group('Learning  parameters')
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
    rootFolder = './opt/default_easyMera_' + args.target + "_simplePrior_" + str(args.simplePrior) + "_repeat_" + str(args.repeat) + "_hchnl_" + str(args.hchnl) + "_nhidden_" + str(args.nhidden) + "_nMixing_" + str(args.nMixing) + "/"
    print("No specified saving path, using", rootFolder)
else:
    rootFolder = args.folder
if rootFolder[-1] != '/':
    rootFolder += '/'
utils.createWorkSpace(rootFolder)

# Decoding parameters to mem, saving them to save folder.
if not args.load:
    target = args.target
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
        config = {'target': target, 'repeat': repeat, 'hchnl': hchnl, 'nhidden': nhidden, 'nMixing': nMixing, 'epoch': epoch, 'batch': batch, 'savePeriod': savePeriod, 'lr': lr, 'simplePrior': simplePrior}
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
    decimal = flow.ScalingNshifting(256, -128)
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
    decimal = flow.ScalingNshifting(256, -128)
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

'''
# define the way to init parameters in NN
def initMethod(weight, bias, num):
    if num == nhidden:
        torch.nn.init.zeros_(weight)
        torch.nn.init.zeros_(bias)
'''


def buildLayers(shapeList):
    layers = []
    for no, chn in enumerate(shapeList[:-1]):
        if no != 0 and no != len(shapeList) - 2:
            layers.append(torch.nn.Conv2d(chn, shapeList[no + 1], 1))
        else:
            layers.append(torch.nn.Conv2d(chn, shapeList[no + 1], 3, padding=1))
        if no != len(shapeList) - 2:
            layers.append(torch.nn.ReLU(inplace=True))
    return layers


layerList = []
shapeList = [targetSize[0] * 3] + [hchnl] * (nhidden + 1) + [targetSize[0]]
for i in range(4 * repeat):
    layers = buildLayers(shapeList)
    layerList.append(torch.nn.Sequential(*layers))
    #layerList.append(torch.nn.Sequential(torch.nn.Conv2d(9, hchnl, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(hchnl, hchnl, 1, padding=0), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(hchnl, 3, 3, padding=1)))
    torch.nn.init.zeros_(layerList[-1][-1].weight)
    torch.nn.init.zeros_(layerList[-1][-1].bias)

shapeList = [targetSize[0]] + [hchnl] * (nhidden + 1) + [targetSize[0] * 3]
if not simplePrior:
    meanNNlist = []
    scaleNNlist = []
    layers = buildLayers(shapeList)
    meanNNlist.append(torch.nn.Sequential(*layers))
    #meanNNlist.append(torch.nn.Sequential(torch.nn.Conv2d(3, hchnl, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(hchnl, hchnl, 1, padding=0), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(hchnl, 9, 3, padding=1)))
    layers = buildLayers(shapeList)
    scaleNNlist.append(torch.nn.Sequential(*layers))
    #scaleNNlist.append(torch.nn.Sequential(torch.nn.Conv2d(3, hchnl, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(hchnl, hchnl, 1, padding=0), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(hchnl, 9, 3, padding=1)))
    torch.nn.init.zeros_(meanNNlist[-1][-1].weight)
    torch.nn.init.zeros_(meanNNlist[-1][-1].bias)
    torch.nn.init.zeros_(scaleNNlist[-1][-1].weight)
    torch.nn.init.zeros_(scaleNNlist[-1][-1].bias)
else:
    meanNNlist = None
    scaleNNlist = None

# Building MERA model
f = flow.SimpleMERA(blockLength, layerList, meanNNlist, scaleNNlist, repeat, nMixing, decimal=decimal, rounding=utils.roundingWidentityGradient).to(device)

'''
from utils import getIndeices
shape = [blockLength, blockLength]
depth = int(math.log(blockLength, 2))
kernelSize = 2
indexList = []
for no in range(depth):
    indexList.append(getIndeices(shape, kernelSize, kernelSize, kernelSize * (kernelSize**no), kernelSize**no, 0))
indexIList = [item[0] for item in indexList]
indexJList = [item[1] for item in indexList]

factorOutIList = [term[:, 1:] if no != len(indexIList) - 1 else term for no, term in enumerate(indexIList)]
factorOutJList = [term[:, 1:] if no != len(indexJList) - 1 else term for no, term in enumerate(indexJList)]
'''


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

    '''
    # wavelet plot, Fig. 3 in draft

    # draw samples, same samples
    samplesNum = 10
    samples, _ = iter(train).next()
    samples = samples[:samplesNum].to(device)

    # build a shallow flow, change var _depth here wouldn't change how plot behave
    _depth = 2
    ftest = flow.MERA(dimensional, blockLength, f.layerList[:(_depth * (repeat + 1))], repeat, depth=_depth).to(device)

    # do the transformations
    z, _ = ftest.inverse(samples)

    # collect parts
    zparts = []
    for no in range(_depth):
        _, z_ = utils.dispatch(factorOutIList[no], factorOutJList[no], z)
        zparts.append(z_)

    _, zremain = utils.dispatch(ftest.indexI[-1], ftest.indexJ[-1], z)
    _linesize = np.sqrt(zremain.shape[-2]).astype(np.int)

    if repeat % 2 == 0:
        # the inner upper left part
        zremain = zremain[:, :, :, :1].reshape(*zremain.shape[:-2], _linesize, _linesize)
    else:
        # the inner low right part
        zremain = zremain[:, :, :, -1:].reshape(*zremain.shape[:-2], _linesize, _linesize)

    # define renorm fn
    def back01(tensor):
        ten = tensor.clone()
        ten = ten.view(ten.shape[0] * ten.shape[1], -1)
        ten -= ten.min(1, keepdim=True)[0]
        ten /= ten.max(1, keepdim=True)[0]
        ten = ten.view(tensor.shape)
        return ten

    # another renorm fn
    def clip(tensor, l=0, h=255):
        return torch.clamp(tensor, l, h).int()

    # yet another renorm fn
    def batchNorm(tensor):
        m = nn.BatchNorm2d(tensor.shape[1], affine=False).to(tensor)
        return m(tensor).float() + 1.0


    renormFn = lambda x: back01(batchNorm(x))

    # norm the remain
    zremain = renormFn(zremain)

    for i in range(_depth):

        # inner parts, odd repeat order: upper left, upper right, down left; even repeat order: upper right, down left, down right
        parts = []
        for no in range(3):
            part = renormFn(zparts[-(i + 1)][:, :, :, no].reshape(*zremain.shape))
            parts.append(part)

        # piece the inner up
        if repeat % 2 == 0:
            zremain = torch.cat([zremain, parts[0]], dim=-1)
            tmp = torch.cat([parts[1], parts[2]], dim=-1)
            zremain = torch.cat([zremain, tmp], dim=-2)
        else:
            tmp = torch.cat([parts[0], parts[1]], dim=-1)
            zremain = torch.cat([parts[2], zremain], dim=-1)
            zremain = torch.cat([tmp, zremain], dim=-2)


    # convert zremaoin to numpy array
    zremain = zremain.permute([0, 2, 3, 1]).detach().cpu().numpy()

    # convert samples to numpy int array
    samples = samples.to(torch.int).permute([0, 2, 3, 1]).detach().cpu().numpy()

    for no in range(samplesNum):
        waveletPlot = plt.figure(figsize=(8, 8))
        waveletAx = waveletPlot.add_subplot(111)
        waveletAx.imshow(zremain[no])
        plt.savefig(rootFolder + 'pic/waveletPlot' + str(no) + '.png', bbox_inches="tight", pad_inches=0)
        plt.close()
        originalPlot = plt.figure(figsize=(8, 8))
        originalAx = originalPlot.add_subplot(111)
        originalAx.imshow(samples[no])
        plt.savefig(rootFolder + 'pic/originalPlot' + str(no) + '.png', bbox_inches="tight", pad_inches=0)
        plt.close()
    '''

# Training
f = train.forwardKLD(f, targetTrainLoader, targetTestLoader, epoch, lr, savePeriod, rootFolder, plotfn=plotfn)

# Pasuse
import pdb
pdb.set_trace()
