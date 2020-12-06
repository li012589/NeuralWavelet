import torch, torchvision
import numpy as np

from utils import buildWaveletLayers, harrInitMethod1, harrInitMethod2, leGallInitMethod1, leGallInitMethod2
import utils
import flow, source
import os, glob
import argparse, json, math

from PIL import Image

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

parser = argparse.ArgumentParser(description="")

parser.add_argument("-folder", default=None, help="Path to load the trained model")
parser.add_argument("-depth", type=int, default=1, help="wavelet depth")
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
        depth = config['depth']
        repeat = config['repeat']
        nhidden = config['nhidden']
        hdim = config['hdim']
        nNICE = config['nNICE']
        nMixing = config['nMixing']
        smallPrior = config['smallPrior']
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

zeromeanList = []
for no, term in enumerate(loadedF.prior.priorList[1:]):
    if no == len(loadedF.prior.priorList) - 2:
        mean = term.mean[0]
    else:
        mean = term.mean
    zeromeanList.append(mean)

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


# Building MERA model
f = flow.MERA(dimensional, blockLength, layerList, repeat, depth=args.depth, prior=p).to(device)

ftmp = flow.MERA(dimensional, blockLength, layerList, repeat, 1, prior=p).to(device)

zzero = torch.zeros([1, 3, targetSize[-1], targetSize[-1]])

for no in reversed(range(1, int(math.log(targetSize[-1], 2)))):
    zzero = utils.collect(ftmp.prior.factorOutIList[no], ftmp.prior.factorOutJList[no], zzero, zeromeanList[no - 1])

# feature not finished: zero low coef
xzero, _ = ftmp.forward(zzero)


z, _ = f.inverse(IMG)

def fftplot(img):
    f = np.fft.fft2(img)
    f = np.fft.fftshift(f)
    f = np.log(np.abs(f) + 1)
    f = f / f.reshape(-1, 3).max(0, keepdims=True)

    return f

# define renorm fn
def back01(tensor):
    ten = tensor.clone().float()
    ten = ten.view(ten.shape[0] * ten.shape[1], -1)
    ten -= ten.min(1, keepdim=True)[0]
    ten /= ten.max(1, keepdim=True)[0]
    ten = ten.view(tensor.shape)
    return ten


renormFn = lambda x: x


'''
fig = plt.figure(figsize=(8, 4))

ax = plt.subplot(121)
ax.imshow(fftplot(IMGnp), cmap="gray")
ax = plt.subplot(122)
ax.imshow(fftplot(znp), cmap="gray")

plt.show()
'''

# collect parts
zparts = []
for no in range(args.depth):
    _, z_ = utils.dispatch(p.factorOutIList[no], p.factorOutJList[no], z)
    zparts.append(z_)

_, zremain = utils.dispatch(f.indexI[-1], f.indexJ[-1], z)
_linesize = np.sqrt(zremain.shape[-2]).astype(np.int)

'''
'''
if repeat % 2 == 0:
    # the inner upper left part
    zmain = zremain[:, :, :, :1]
    zdetail = zremain[:, :, :, 1:]
else:
    # the inner low right part
    zmain = zremain[:, :, :, -1:]
    zdetail = zremain[:, :, :, :-1]


lowmain = zmain
import pdb
pdb.set_trace()
lowdetail = torch.round(decimal.forward_(loadedF.prior.priorList[0].mean)).reshape(zdetail.shape)

highmain = torch.zeros_like(zmain)
highdetail = zdetail

if repeat % 2 == 0:
    lowzremain = torch.cat([lowmain, lowdetail], -1)
    highzremain = torch.cat([highmain, highdetail], -1)
else:
    lowzremain = torch.cat([lowdetail, lowmain], -1)
    highzremain = torch.cat([highdetail, highmain], -1)


lowz = torch.zeros_like(z)
lowz = utils.collect(f.indexI[-1], f.indexJ[-1], lowz, lowzremain)

highz = torch.zeros_like(z)
highz = utils.collect(f.indexI[-1], f.indexJ[-1], highz, highzremain)


lowIMG,_ = f.forward(lowz)
highIMG,_ = f.forward(highz)

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

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, lowff, cmap=cm.coolwarm, linewidth=0, antialiased=False)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, highff, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
import pdb
pdb.set_trace()

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