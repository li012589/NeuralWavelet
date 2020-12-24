import numpy as np
import argparse, json, math
import os, glob

import flow, utils, source

import torch, torchvision
from torch import nn

from encoder import rans, coder
from utils import cdfDiscreteLogitstic, cdfMixDiscreteLogistic
from matplotlib import pyplot as plt
import matplotlib

#torch.manual_seed(42)

parser = argparse.ArgumentParser(description="")

parser.add_argument("-folder", default=None, help="Path to load the trained model")
parser.add_argument("-cuda", type=int, default=-1, help="Which device to use with -1 standing for CPU, number bigger than -1 is N.O. of GPU.")
parser.add_argument("-baseScale", type=float, default=-2.0, help="exp scaling of distribution's logscale to achieve better quality")
parser.add_argument("-exBaseScale", type=float, default=-2.0, help="exp scaling of distribution's logscale to achieve better quality for exoplot")
parser.add_argument("-best", action='store_false', help="if load the best model")
parser.add_argument("-epoch", type=int, default=-1, help="epoch to load")
parser.add_argument("-exdepth", type=int, default=2, help="num of image to expand")
parser.add_argument("-num", type=int, default=10, help="num of image to demo")
parser.add_argument("-fix", action='store_true', help="fix color shift in exoplot")
parser.add_argument('-target', type=str, default='original', choices=['original', 'CIFAR', 'ImageNet32', 'ImageNet64', 'MNIST'], metavar='DATASET', help='Dataset choice.')

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
        nMixing = config['nMixing']
        batch = config['batch']
        try:
            HUE = config['HUE']
        except:
            HUE = True


# update batch using passing parameter
batch = args.num

if args.fix:
    print("Fix color")
    from AWB2 import color_transfer

if HUE:
    lambd = lambda x: (x * 255).byte().to(torch.float32).to(device)
else:
    lambd = lambda x: utils.rgb2ycc((x * 255).byte().float(), True).to(torch.float32).to(device)

if args.target != 'original':
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
    trainsetTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambd)])
    trainTarget = torchvision.datasets.CIFAR10(root='./data/cifar', train=True, download=True, transform=trainsetTransform)
    testTarget = torchvision.datasets.CIFAR10(root='./data/cifar', train=False, download=True, transform=trainsetTransform)
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=batch, shuffle=True)
    targetTestLoader = torch.utils.data.DataLoader(testTarget, batch_size=batch, shuffle=True)
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
    trainsetTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambd)])
    trainTarget = utils.ImageNet(root='./data/ImageNet32', train=True, download=True, transform=trainsetTransform)
    testTarget = utils.ImageNet(root='./data/ImageNet32', train=False, download=True, transform=trainsetTransform)
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=batch, shuffle=True)
    targetTestLoader = torch.utils.data.DataLoader(testTarget, batch_size=batch, shuffle=True)

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
    trainsetTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambd)])
    trainTarget = utils.ImageNet(root='./data/ImageNet64', train=True, download=True, transform=trainsetTransform, d64=True)
    testTarget = utils.ImageNet(root='./data/ImageNet64', train=False, download=True, transform=trainsetTransform, d64=True)
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=batch, shuffle=True)
    targetTestLoader = torch.utils.data.DataLoader(testTarget, batch_size=batch, shuffle=True)

elif target == "MNIST":
    pass
else:
    raise Exception("No such target")
# decide which model to load
if args.best:
    name = max(glob.iglob(os.path.join(rootFolder, '*.saving')), key=os.path.getctime)
elif args.epoch == -1:
    name = max(glob.iglob(os.path.join(rootFolder, 'savings', '*.saving')), key=os.path.getctime)
else:
    name = max(glob.iglob(os.path.join(rootFolder, 'savings', 'SimpleMERA_epoch_' + str(args.epoch) + '.saving')), key=os.path.getctime)

depth = int(math.log(blockLength, 2))
exdepth = args.exdepth

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

if 'simplePrior_True' in name:
    meanNNList = None
    scaleNNlist = None
    length = targetSize[-1]
    lenList = []
    _length = int(length * length / 4)
    for n in range(depth):
        if n != depth - 1:
            lenList.append([batch, 3, _length, 3])
        _length = int(_length / 4)

elif 'simplePrior_False' in name:
    meanNNList = [loadedF.meanNNlist[0]]
    scaleNNlist = [loadedF.scaleNNlist[0]]
else:
    raise Exception('prior not defined')

meanFn = torch.nn.ModuleList(meanNNList)
scaleFn = torch.nn.ModuleList(scaleNNlist)

# Define nomaliziation and decimal
if 'easyMera' in name:
    decimal = flow.ScalingNshifting(256, -128)
elif '1to2Mera' in name:
    decimal = flow.ScalingNshifting(256, 0)
else:
    raise Exception("model not define")

rounding = utils.roundingWidentityGradient

# Building MERA mode
if 'easyMera' in name:
    f = flow.SimpleMERA(blockLength, layerList, meanNNList, scaleNNlist, repeat, None, nMixing, decimal=decimal, rounding=utils.roundingWidentityGradient).to(device)
    flist = []
    for i in range(exdepth):
        flist.append(flow.SimpleMERA(blockLength, layerList, meanNNList, scaleNNlist, repeat, depth + i + 1, nMixing, decimal=decimal, rounding=utils.roundingWidentityGradient).to(device))
elif '1to2Mera' in name:
    f = flow.OneToTwoMERA(blockLength, layerList, meanNNList, scaleNNlist, repeat, None, nMixing, decimal=decimal, rounding=utils.roundingWidentityGradient).to(device)
    fex = flow.OneToTwoMERA(blockLength, layerList, meanNNList, scaleNNlist, repeat, 2 * depth, nMixing, decimal=decimal, rounding=utils.roundingWidentityGradient).to(device)
else:
    raise Exception("model not define")


def im2grp(t):
    return t.reshape(t.shape[0], t.shape[1], t.shape[2] // 2, 2, t.shape[3] // 2, 2).permute([0, 1, 2, 4, 3, 5]).reshape(t.shape[0], t.shape[1], -1, 4)


def grp2im(t):
    return t.reshape(t.shape[0], t.shape[1], int(t.shape[2] ** 0.5), int(t.shape[2] ** 0.5), 2, 2).permute([0, 1, 2, 4, 3, 5]).reshape(t.shape[0], t.shape[1], int(t.shape[2] ** 0.5) * 2, int(t.shape[2] ** 0.5) * 2)


def divide(z):
    parts = []
    ul = z
    for no in range(int(math.log(blockLength, 2))):
        _x = im2grp(ul)
        ul = _x[:, :, :, 0].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
        zpart = _x[:, :, :, 1:].contiguous()
        parts.append(zpart.detach())
    parts.append(ul.detach())
    return parts


def join(rcnZ):
    for no in reversed(range(len(rcnZ))):
        if no == len(rcnZ) - 1:
            ul = rcnZ[no]
        else:
            rcnZpart = rcnZ[no].reshape(*ul.shape, 3)
            ul = ul.reshape(*ul.shape, 1)
            _x = torch.cat([ul, rcnZpart], -1).reshape(*ul.shape[:2], -1, 4)
            ul = grp2im(_x).contiguous()
    return ul


def reform(tensor):
    return tensor.reshape(tensor.shape[0], tensor.shape[1] // 3, 3, tensor.shape[2], tensor.shape[3]).permute([0, 1, 3, 4, 2]).contiguous().reshape(tensor.shape[0], tensor.shape[1] // 3, tensor.shape[2] * tensor.shape[3], 3)


def sampleMoreDetails(samples):
    if 'simplePrior_False' in name:
        mean = reform(f.meanNNlist[0](f.decimal.inverse_(samples))).contiguous()
        scale = reform(f.scaleNNlist[0](f.decimal.inverse_(samples))).contiguous()
        sampledDetails = utils.sampleDiscreteLogistic([*mean.shape], mean, scale + args.exBaseScale, decimal=f.decimal)
    else:
        sampledDetails = utils.sampleDiscreteLogistic([batch, 3, np.prod(samples.shape[-2:]), 3], loadedF.prior.priorList[0].mean, loadedF.prior.priorList[0].logscale + args.exBaseScale, decimal=f.decimal)
    return sampledDetails


def plotLoading(loader):
    samples, _ = next(iter(loader))
    z, _ = f.inverse(samples)

    zParts = divide(z)

    augmenZ = []
    for no in range(int(math.log(blockLength, 2))):
        tmpZ = []
        for i in range(no):
            #tmpZ.append(f.prior.priorList[i].sample(batch))
            if 'simplePrior_False' in name:
                sampledDetails = utils.sampleDiscreteLogistic([*f.meanList[i].shape], f.meanList[i], f.scaleList[i] + args.baseScale, decimal=f.decimal)
            else:
                sampledDetails = utils.sampleDiscreteLogistic(lenList[i], loadedF.prior.priorList[i].mean, loadedF.prior.priorList[i].logscale + args.baseScale, decimal=f.decimal)
            #sampledDetails = torch.zeros_like(sampledDetails)
            tmpZ.append(sampledDetails)
        tmpZ = tmpZ + zParts[no:]
        augmenZ.append(join(tmpZ))

    expSamples = samples
    plotList = []
    expZparts = zParts
    for i in range(exdepth):
        moreDetails = sampleMoreDetails(expSamples)
        expZparts = [moreDetails] + expZparts
        expZ = join(expZparts)
        expSamples, _ = flist[i].forward(expZ)
        plotList.append(expSamples)

    rcnZ = torch.cat(augmenZ, 0)

    rcnSamples, _ = f.forward(rcnZ)

    if not HUE:
        rcnSamples = utils.ycc2rgb(rcnSamples, True, True).int()

    rcnSamples = rcnSamples.detach().reshape(int(math.log(blockLength, 2)), batch, *rcnSamples.shape[1:])

    def back01(tensor):
        ten = tensor.clone()
        ten = ten.view(ten.shape[0], -1)
        ten -= ten.min(1, keepdim=True)[0]
        ten /= ten.max(1, keepdim=True)[0]
        ten = ten.view(tensor.shape)
        return ten

    def clip(tensor):
        return torch.clamp(tensor, 0, 255).int()

    def grayWorld(tensor):
        meanRGB = tensor.reshape(tensor.shape[0], 3, -1).mean(-1)
        gray = meanRGB.sum(-1, keepdim=True) / 3
        scaleRGB = gray / meanRGB
        scaledTensor = torch.round(tensor.reshape(tensor.shape[0], 3, -1) * scaleRGB.reshape(*scaleRGB.shape, 1)).reshape(tensor.shape)
        return torch.clamp(scaledTensor, 0, 255).int()


    def perfReflect(tensor, ratio=0.1):
        assert tensor.shape[0] == 1
        ilum = tensor.sum(1)
        hists, bins = np.histogram(ilum.flatten(), 766, [0, 766])
        Y = 765
        num, key = 0, 0
        while Y >= 0:  # Select threshold according to ratio
            num += hists[Y]
            if num > ilum.flatten().shape[0] * ratio:
                key = Y
                break
            Y = Y - 1

        idx, idy = np.where(ilum >= key)[1:]
        sumRGB = tensor[:, :, idx, idy].mean(-1, keepdim=True)
        maxRGB = tensor.reshape(1, 3, -1).max(-1, keepdim=True)[0]
        scaleRGB = maxRGB / sumRGB
        scaledTensor = torch.round(tensor.reshape(1, 3, -1) * scaleRGB).reshape(tensor.shape)
        return torch.clamp(scaledTensor, 0, 255).int()


    for i in range(rcnSamples.shape[1]):
        for j in range(int(math.log(blockLength, 2))):
            #im = back01(rcnSamples[j][i]).permute([1, 2, 0]).detach().numpy()
            im = clip(rcnSamples[j][i]).permute([1, 2, 0]).detach().numpy().astype('uint8')
            matplotlib.image.imsave(rootFolder + 'pic/proloadPlot_N_' + str(i) + '_P_' + str(j) + '.png', im)
            '''
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(clip(rcnSamples[j][i]).permute([1, 2, 0]).detach().numpy())
            plt.axis('off')
            plt.savefig(rootFolder + 'pic/proloadPlot_N_' + str(i) + '_P_' + str(j) + '.tiff', bbox_inches="tight", pad_inches=0)
            plt.close()
            '''

    for i in range(batch):
        for j, term in enumerate(plotList):
            if args.fix:
                im = color_transfer(rcnSamples[0, i].int().detach().permute([1, 2, 0]).numpy().astype('uint8'), (back01(term[i]) * 255).permute([1, 2, 0]).detach().numpy().astype('uint8'))
            else:
                #im = back01(term[i]).permute([1, 2, 0]).detach().numpy()
                im = clip(term[i]).permute([1, 2, 0]).detach().numpy().astype('uint8')
            #im = grayWorld(term[i:i + 1])[0].permute([1, 2, 0]).detach().numpy().astype('uint8')
            #im = perfReflect(term[i:i + 1].detach(), ratio=0.02)[0].permute([1, 2, 0]).detach().numpy().astype('uint8')
            #im = retinex_adjust(torch.clamp(term[i], 0, 255).permute([1, 2, 0]).detach().numpy().astype('uint8'))
            matplotlib.image.imsave(rootFolder + 'pic/exoloadPlot_N_' + str(i) + '_P_' + str(j) + '.png', im)
            '''
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(clip(term[i]).permute([1, 2, 0]).detach().numpy())
            plt.axis('off')
            plt.savefig(rootFolder + 'pic/exoloadPlot_N_' + str(i) + '_P_' + str(j) + '.tiff', bbox_inches="tight", pad_inches=0)
            plt.close()
            '''


plotLoading(targetTrainLoader)
#plotLoading(targetTestLoader)

