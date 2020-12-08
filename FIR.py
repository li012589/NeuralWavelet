import torch, torchvision
import numpy as np

from utils import buildWaveletLayers, harrInitMethod1, harrInitMethod2, leGallInitMethod1, leGallInitMethod2
import utils
import flow
import os, glob
import argparse, json, math

from PIL import Image

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

parser = argparse.ArgumentParser(description="")

parser.add_argument("-folder", default=None, help="Path to load the trained model")
parser.add_argument("-step", type=float, default=0.001, help="step of omega")
parser.add_argument("-best", action='store_false', help="if load the best model")
parser.add_argument('-target', type=str, default='original', choices=['original', 'CIFAR', 'ImageNet32', 'ImageNet64', 'MNIST'], metavar='DATASET', help='Dataset choice.')

args = parser.parse_args()

device = torch.device("cpu")

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

batch = 1

step = args.step

# decide which model to load
if args.best:
    name = max(glob.iglob(os.path.join(rootFolder, '*.saving')), key=os.path.getctime)
else:
    name = max(glob.iglob(os.path.join(rootFolder, 'savings', '*.saving')), key=os.path.getctime)

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
    lambd = lambda x: (x * 255).byte().to(torch.float32).to(device)
    trainsetTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambd)])
    trainTarget = utils.ImageNet(root='./data/ImageNet64', train=True, download=True, transform=trainsetTransform, d64=True)
    testTarget = utils.ImageNet(root='./data/ImageNet64', train=False, download=True, transform=trainsetTransform, d64=True)
    targetTrainLoader = torch.utils.data.DataLoader(trainTarget, batch_size=batch, shuffle=True)
    targetTestLoader = torch.utils.data.DataLoader(testTarget, batch_size=batch, shuffle=True)

samples, _ = next(iter(targetTrainLoader))
shape = samples.shape[1:]
img = samples[0].reshape(1, -1).requires_grad_()

IMG = img.reshape(1, *shape)

decimal = flow.ScalingNshifting(256, 0)

targetSize = IMG.shape[1:]
depth = int(math.log(targetSize[-1], 2))

# load the model
print("load saving at " + name)
f = torch.load(name, map_location=device)

if args.target != 'original':
    if 'easyMera' in name:
        layerList = f.layerList[:(4 * repeat)]
        layerList = [layerList[no] for no in range(4 * repeat)]
    elif '1to2Mera' in name:
        layerList = f.layerList[:(2 * repeat)]
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
        meanNNlist = [f.meanNNlist[0]]
        scaleNNlist = [f.scaleNNlist[0]]
    else:
        meanNNlist = None
        scaleNNlist = None

    rounding = utils.roundingWidentityGradient

    prior = f.prior
    prior.depth = int(math.log(targetSize[-1], 2))
    if 'simplePrior_False' in name:
        pass
    else:
        prior.priorList = torch.nn.ModuleList([prior.priorList[0] for _ in range(int(math.log(targetSize[-1], 2)) - 1)] + [prior.priorList[-1]])
    # Building MERA mode
    if 'easyMera' in name:
        f = flow.SimpleMERA(blockLength, layerList, meanNNlist, scaleNNlist, repeat, None, nMixing, decimal=decimal, rounding=utils.roundingWidentityGradient).to(device)
    elif '1to2Mera' in name:
        f = flow.OneToTwoMERA(blockLength, layerList, meanNNlist, scaleNNlist, repeat, None, nMixing, decimal=decimal, rounding=utils.roundingWidentityGradient).to(device)

    f.prior = prior

z, _ = f.inverse(IMG)


def im2grp(t):
    return t.reshape(t.shape[0], t.shape[1], t.shape[2] // 2, 2, t.shape[3] // 2, 2).permute([0, 1, 2, 4, 3, 5]).reshape(t.shape[0], t.shape[1], -1, 4)


def grp2im(t):
    return t.reshape(t.shape[0], t.shape[1], int(t.shape[2] ** 0.5), int(t.shape[2] ** 0.5), 2, 2).permute([0, 1, 2, 4, 3, 5]).reshape(t.shape[0], t.shape[1], int(t.shape[2] ** 0.5) * 2, int(t.shape[2] ** 0.5) * 2)


# define renorm fn
def back01(tensor):
    ten = tensor.clone().float()
    ten = ten.view(ten.shape[0] * ten.shape[1], -1)
    ten -= ten.min(1, keepdim=True)[0]
    ten /= ten.max(1, keepdim=True)[0]
    ten = ten.view(tensor.shape)
    return ten


renormFn = lambda x: x

# collect parts
ul = z
UR = []
DL = []
DR = []
for _ in range(depth):
    _x = im2grp(ul)
    ul = _x[:, :, :, 0].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
    ur = _x[:, :, :, 1].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
    dl = _x[:, :, :, 2].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
    dr = _x[:, :, :, 3].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
    UR.append(renormFn(ur))
    DL.append(renormFn(dl))
    DR.append(renormFn(dr))

ul = renormFn(ul)

for no in reversed(range(depth)):

    ur = UR[no]
    dl = DL[no]
    dr = DR[no]

    upper = torch.cat([ul, ur], -1)
    down = torch.cat([dl, dr], -1)
    ul = torch.cat([upper, down], -2)

y = ul[:1, 0, 0, :]

grad = utils.jacobian(y, img)

H = grad[0, :targetSize[-1], :targetSize[-1]]

plt.imshow(H.detach())

deltaexp = np.exp((np.arange(targetSize[-1]) * 1j * step * np.pi))
mod = [np.exp((np.arange(targetSize[-1]) * 1j * 0 * np.pi))]
for n in range(int(1 / step)):
    _mod = mod[-1] * deltaexp
    mod.append(_mod)

mod = np.vstack(mod)

plt.figure()
for no in range(targetSize[-1] // 2):

    h = H[no, :].detach().numpy()
    ph = np.abs(mod.dot(h))

    plt.plot(ph)

plt.figure()
for no in range(targetSize[-1] // 2, targetSize[-1]):

    h = H[no, :].detach().numpy()
    ph = np.abs(mod.dot(h))

    plt.plot(ph)

plt.show()

