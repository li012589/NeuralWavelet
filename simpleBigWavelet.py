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
        repeat = config['repeat']
        nhidden = config['nhidden']
        hchnl = config['hchnl']
        nMixing = config['nMixing']
        simplePrior = config['simplePrior']
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

if 'easyMera' in name:
    layerList = loadedF.layerList[:(4 * repeat)]
    layerList = [layerList[no] for no in range(4 * repeat)]
elif '1to2Mera' in name:
    layerList = loadedF.layerList[:(2 * repeat)]
    layerList = [layerList[no] for no in range(2 * repeat)]
else:
    raise Exception("model not define")

# Define dimensions
targetSize = IMG.shape[1:]
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

rounding = utils.roundingWidentityGradient

# Building MERA mode
if 'easyMera' in name:
    f = flow.SimpleMERA(blockLength, layerList, None, None, repeat, args.depth + 1, nMixing, decimal=decimal, rounding=utils.roundingWidentityGradient).to(device)
elif '1to2Mera' in name:
    f = flow.OneToTwoMERA(blockLength, layerList, None, None, repeat, args.depth + 1, nMixing, decimal=decimal, rounding=utils.roundingWidentityGradient).to(device)
else:
    raise Exception("model not define")

z, _ = f.inverse(IMG)

assert args.depth <= int(math.log(blockLength, 2))


def im2grp(t):
    return t.reshape(t.shape[0], t.shape[1], t.shape[2] // 2, 2, t.shape[3] // 2, 2).permute([0, 1, 2, 4, 3, 5]).reshape(t.shape[0], t.shape[1], -1, 4)


# define renorm fn
def back01(tensor):
    ten = tensor.clone().float()
    ten = ten.view(ten.shape[0] * ten.shape[1], -1)
    ten -= ten.min(1, keepdim=True)[0]
    ten /= ten.max(1, keepdim=True)[0]
    ten = ten.view(tensor.shape)
    return ten


def backMeanStd(tensor):
    mean = IMG.reshape(*IMG.shape[:2], -1).mean(-1).reshape(*IMG.shape[:2], 1, 1)
    std = IMG.reshape(*IMG.shape[:2], -1).std(-1).reshape(*IMG.shape[:2], 1, 1)
    return tensor * std.repeat([1, 1, tensor.shape[-1], tensor.shape[-1]]) + mean.repeat([1, 1, tensor.shape[-1], tensor.shape[-1]])

# another renorm fn
def clip(tensor, l=0, h=255):
    return torch.clamp(tensor, l, h).int()


# yet another renorm fn
def batchNorm(tensor, base=1.0):
    m = nn.BatchNorm2d(tensor.shape[1], affine=False)
    return m(tensor).float() + base


renormFn = lambda x: back01(batchNorm(x))

# collect parts
ul = z
UR = []
DL = []
DR = []
for _ in range(args.depth):
    _x = im2grp(ul)
    ul = _x[:, :, :, 0].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
    ur = _x[:, :, :, 1].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
    dl = _x[:, :, :, 2].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
    dr = _x[:, :, :, 3].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
    UR.append(renormFn(ur))
    DL.append(renormFn(dl))
    DR.append(renormFn(dr))

#ul = back01(backMeanStd(batchNorm(ul, 0)))
ul = renormFn(ul)
#ul = back01(clip(backMeanStd(batchNorm(ul))))

for no in reversed(range(args.depth)):

    ur = UR[no]
    dl = DL[no]
    dr = DR[no]

    upper = torch.cat([ul, ur], -1)
    down = torch.cat([dl, dr], -1)
    ul = torch.cat([upper, down], -2)

# convert zremaoin to numpy array
zremain = ul.permute([0, 2, 3, 1]).detach().cpu().numpy()

waveletPlot = plt.figure(figsize=(8, 8))
waveletAx = waveletPlot.add_subplot(111)
waveletAx.imshow(zremain[0])
plt.axis('off')
plt.savefig(rootFolder + 'pic/BigWavelet.pdf', bbox_inches="tight", pad_inches=0)
plt.close()


