import numpy as np
import argparse, json, math
import os, glob
from PIL import Image

import flow, utils, source

import torch, torchvision
from torch import nn

from encoder import rans, coder
from utils import cdfDiscreteLogitstic, cdfMixDiscreteLogistic
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description="")

parser.add_argument("-folder", default=None, help="Path to load the trained model")
parser.add_argument("-cuda", type=int, default=-1, help="Which device to use with -1 standing for CPU, number bigger than -1 is N.O. of GPU.")
parser.add_argument("-depth", type=int, default=2, help="wavelet depth")
parser.add_argument("-best", action='store_false', help="if load the best model")
parser.add_argument("-img", default=None, help="the img path")

args = parser.parse_args()

if args.img is None:
    raise Exception("No image input")

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

IMG = Image.open(args.img)
IMG = torch.from_numpy(np.array(IMG)).permute([2, 0, 1])
IMG = IMG.reshape(1, *IMG.shape).float().to(device)

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

# Define dimensions
targetSize = IMG.shape[1:]
dimensional = 2
channel = targetSize[0]
blockLength = targetSize[-1]

# Define nomaliziation and decimal
decimal = flow.ScalingNshifting(256, -128)
rounding = utils.roundingWidentityGradient

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

if depth == -1:
    depth = None

# Building MERA model
f = flow.MERA(dimensional, blockLength, layerList, repeat, depth=args.depth, prior=p).to(device)

z, _ = f.inverse(IMG)

assert args.depth <= int(math.log(blockLength, 2))

# collect parts
zparts = []
for no in range(args.depth):
    _, z_ = utils.dispatch(p.factorOutIList[no], p.factorOutJList[no], z)
    zparts.append(z_)

_, zremain = utils.dispatch(f.indexI[-1], f.indexJ[-1], z)
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
    m = nn.BatchNorm2d(tensor.shape[1], affine=False)
    return m(tensor).float() + 1.0


renormFn = lambda x: back01(batchNorm(x))

zremain = renormFn(zremain)

for i in range(args.depth):
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

waveletPlot = plt.figure(figsize=(8, 8))
waveletAx = waveletPlot.add_subplot(111)
waveletAx.imshow(zremain[0])
plt.axis('off')
plt.savefig(rootFolder + 'pic/BigWavelet.pdf', bbox_inches="tight", pad_inches=0)
plt.close()

