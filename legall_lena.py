import torch
import numpy as np

from utils import buildWaveletLayers, harrInitMethod1, harrInitMethod2, leGallInitMethod1, leGallInitMethod2
import utils
import flow

from PIL import Image

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


nhidden = 1
depth = 1
hchnl = 6

IMG = Image.open('./etc/lena512color.tiff')
IMG = torch.from_numpy(np.array(IMG)).permute([2, 0, 1])
IMG = IMG.reshape(1, *IMG.shape).float()

decimal = flow.ScalingNshifting(256, 0)

targetSize = IMG.shape[1:]

initMethods = []
initMethods.append(lambda: harrInitMethod1(targetSize[0]))
initMethods.append(lambda: harrInitMethod2(targetSize[0]))

orders = [True, False]

shapeList1D = [targetSize[0]] + [hchnl] * (nhidden + 1) + [targetSize[0]]

layerList = []
for j in range(2):
    layerList.append(buildWaveletLayers(initMethods[j], targetSize[0], hchnl, nhidden, orders[j]))


f = flow.OneToTwoMERA(targetSize[-1], layerList, None, None, 1, depth, 5, decimal=decimal, rounding=utils.roundingWidentityGradient)

z, _ = f.inverse(IMG)

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

lowul = ul
highul = torch.zeros_like(ul)

for no in reversed(range(depth)):
    ur = torch.zeros_like(UR[no].reshape(*lowul.shape, 1))
    dl = torch.zeros_like(DL[no].reshape(*lowul.shape, 1))
    dr = torch.zeros_like(DR[no].reshape(*lowul.shape, 1))
    lowul = lowul.reshape(*lowul.shape, 1)

    _x = torch.cat([lowul, ur, dl, dr], -1).reshape(*lowul.shape[:2], -1, 4)
    lowul = grp2im(_x).contiguous()

for no in reversed(range(depth)):
    ur = UR[no].reshape(*highul.shape, 1)
    dl = DL[no].reshape(*highul.shape, 1)
    dr = DR[no].reshape(*highul.shape, 1)
    highul = highul.reshape(*highul.shape, 1)

    _x = torch.cat([highul, ur, dl, dr], -1).reshape(*highul.shape[:2], -1, 4)
    highul = grp2im(_x).contiguous()

lowIMG,_ = f.forward(lowul)
highIMG,_ = f.forward(highul)

ff = fftplot(IMG.reshape(IMG.shape[1:]).permute([1, 2, 0]).detach().numpy())
lowff = fftplot(lowIMG.reshape(lowIMG.shape[1:]).permute([1, 2, 0]).detach().numpy())
highff = fftplot(highIMG.reshape(highIMG.shape[1:]).permute([1, 2, 0]).detach().numpy())


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

ff = rgb2gray(ff)
lowff = rgb2gray(lowff)
highff = rgb2gray(highff)

import pdb
pdb.set_trace()
'''
ffi = ff / ff.max()
plt.imshow(ffi, cmap='gray')
plt.show()
'''

X = np.arange(0, 512, 1)
Y = np.arange(0, 512, 1)
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
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


for no in reversed(range(depth)):

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
waveletAx.imshow(zremain[0].int())

plt.show()