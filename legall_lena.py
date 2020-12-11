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
parser.add_argument("-deltaDepth", type=int, default=1, help="wavelet depth")
parser.add_argument("-best", action='store_false', help="if load the best model")
parser.add_argument("-img", default='./etc/lena512color.tiff', help="the img path")

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

# decide which model to load
if args.best:
    name = max(glob.iglob(os.path.join(rootFolder, '*.saving')), key=os.path.getctime)
else:
    name = max(glob.iglob(os.path.join(rootFolder, 'savings', '*.saving')), key=os.path.getctime)

if args.img != 'target':
    IMG = Image.open(args.img)
    IMG = torch.from_numpy(np.array(IMG)).permute([2, 0, 1])
    IMG = IMG.reshape(1, *IMG.shape).float()
else:
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
    IMG = samples[0].reshape(1, *samples.shape[1:])


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

targetSize = IMG.shape[1:]
blockLength = targetSize[-1]

# Define nomaliziation and decimal
if 'easyMera' in name:
    decimal = flow.ScalingNshifting(256, -128)
elif '1to2Mera' in name:
    decimal = flow.ScalingNshifting(256, 0)
else:
    raise Exception("model not define")

if 'simplePrior_False' in name:
    meanNNlist = [loadedF.meanNNlist[0]]
    scaleNNlist = [loadedF.scaleNNlist[0]]
else:
    meanNNlist = None
    scaleNNlist = None

rounding = utils.roundingWidentityGradient

# Building MERA mode
if 'easyMera' in name:
    f = flow.SimpleMERA(blockLength, layerList, None, None, repeat, args.deltaDepth, nMixing, decimal=decimal, rounding=utils.roundingWidentityGradient).to(device)
    ftmp = flow.SimpleMERA(blockLength, layerList, meanNNlist, scaleNNlist, repeat, int(math.log(targetSize[-1], 2)), nMixing, decimal=decimal, rounding=utils.roundingWidentityGradient).to(device)
elif '1to2Mera' in name:
    f = flow.OneToTwoMERA(blockLength, layerList, None, None, repeat, args.deltaDepth, nMixing, decimal=decimal, rounding=utils.roundingWidentityGradient).to(device)
else:
    raise Exception("model not define")


z, _ = f.inverse(IMG)

if 'simplePrior_False' in name:
    zerosCore = ftmp.inference(torch.round(decimal.forward_(loadedF.prior.lastPrior.mean[0].reshape(1, 3, 2, 2))), int(math.log(targetSize[-1], 2)) - args.deltaDepth, startDepth=1)
else:
    zerosCore = ftmp.inference(torch.round(decimal.forward_(loadedF.prior.priorList[-1].mean[0].reshape(1, 3, 2, 2))), int(math.log(targetSize[-1], 2)) - args.deltaDepth, startDepth=1)


def fftplot(img):
    f = np.fft.fft2(img)
    f = np.fft.fftshift(f)
    f = np.log(np.abs(f) + 1)

    return f

'''
fig = plt.figure(figsize=(8, 4))

ax = plt.subplot(121)
ax.imshow(fftplot(IMGnp), cmap="gray")
ax = plt.subplot(122)
ax.imshow(fftplot(znp), cmap="gray")

plt.show()
'''

z = z


def im2grp(t):
    return t.reshape(t.shape[0], t.shape[1], t.shape[2] // 2, 2, t.shape[3] // 2, 2).permute([0, 1, 2, 4, 3, 5]).reshape(t.shape[0], t.shape[1], -1, 4)


def grp2im(t):
    return t.reshape(t.shape[0], t.shape[1], int(t.shape[2] ** 0.5), int(t.shape[2] ** 0.5), 2, 2).permute([0, 1, 2, 4, 3, 5]).reshape(t.shape[0], t.shape[1], int(t.shape[2] ** 0.5) * 2, int(t.shape[2] ** 0.5) * 2)


def reform(tensor):
    return tensor.reshape(tensor.shape[0], tensor.shape[1] // 3, 3, tensor.shape[2], tensor.shape[3]).permute([0, 1, 3, 4, 2]).contiguous().reshape(tensor.shape[0], tensor.shape[1] // 3, tensor.shape[2] * tensor.shape[3], 3)


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
for _ in range(args.deltaDepth):
    _x = im2grp(ul)
    ul = _x[:, :, :, 0].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
    ur = _x[:, :, :, 1].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
    dl = _x[:, :, :, 2].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
    dr = _x[:, :, :, 3].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
    UR.append(renormFn(ur))
    DL.append(renormFn(dl))
    DR.append(renormFn(dr))

ul = renormFn(ul)

lowul = ul
highul = zerosCore


for no in reversed(range(args.deltaDepth)):
    if meanNNlist is not None:
        zeroDetails = torch.round(decimal.forward_(reform(loadedF.meanNNlist[0](decimal.inverse_(lowul))).contiguous()))
    else:
        zeroDetails = torch.round(decimal.forward_(loadedF.prior.priorList[0].mean.reshape(1, 3, 1, 3).repeat(lowul.shape[0], 1, np.prod(lowul.shape[-2:]), 3)).contiguous())
    ur = zeroDetails[:, :, :, 0].reshape(*lowul.shape, 1)
    dl = zeroDetails[:, :, :, 1].reshape(*lowul.shape, 1)
    dr = zeroDetails[:, :, :, 2].reshape(*lowul.shape, 1)
    lowul = lowul.reshape(*lowul.shape, 1)

    _x = torch.cat([lowul, ur, dl, dr], -1).reshape(*lowul.shape[:2], -1, 4)
    lowul = grp2im(_x).contiguous()

for no in reversed(range(args.deltaDepth)):
    ur = UR[no].reshape(*highul.shape, 1)
    dl = DL[no].reshape(*highul.shape, 1)
    dr = DR[no].reshape(*highul.shape, 1)
    highul = highul.reshape(*highul.shape, 1)

    _x = torch.cat([highul, ur, dl, dr], -1).reshape(*highul.shape[:2], -1, 4)
    highul = grp2im(_x).contiguous()

lowIMG, _ = f.forward(lowul)
highIMG, _ = f.forward(highul)

plt.figure()
plt.imshow(IMG.int().detach().reshape(targetSize).permute([1, 2, 0]).numpy())
plt.figure()
plt.imshow(lowIMG.int().detach().reshape(targetSize).permute([1, 2, 0]).numpy())
plt.figure()
plt.imshow(highIMG.int().detach().reshape(targetSize).permute([1, 2, 0]).numpy())

ff = fftplot(IMG.reshape(IMG.shape[1:]).permute([1, 2, 0]).detach().numpy())
lowff = fftplot(lowIMG.reshape(lowIMG.shape[1:]).permute([1, 2, 0]).detach().numpy())
highff = fftplot(highIMG.reshape(highIMG.shape[1:]).permute([1, 2, 0]).detach().numpy())


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

ff = rgb2gray(ff)
lowff = rgb2gray(lowff)
highff = rgb2gray(highff)


'''
'''
X = np.arange(0, blockLength, 1)
Y = np.arange(0, blockLength, 1)
X, Y = np.meshgrid(X, Y)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, ff, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(elev=15., azim=-75)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, lowff, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(elev=15., azim=-75)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, highff, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(elev=15., azim=-75)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
import pdb
pdb.set_trace()
'''
for no in reversed(range(args.depth)):

    ur = UR[no]
    dl = DL[no]
    dr = DR[no]

    upper = torch.cat([ul, ur], -1)
    down = torch.cat([dl, dr], -1)
    ul = torch.cat([upper, down], -2)

# convert zremaoin to numpy array
zremain = ul.permute([0, 2, 3, 1]).int().detach().cpu().numpy()

waveletPlot = plt.figure(figsize=(8, 8))
waveletAx = waveletPlot.add_subplot(111)
waveletAx.imshow(zremain[0])

plt.show()
'''
