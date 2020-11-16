import torch
from torch import nn
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def buildTransMatrix(n):
    core1 = torch.tensor([[0.5, 0.5]])
    core2 = torch.tensor([[-1, 1]])
    gap = torch.zeros(1, n)
    upper = torch.cat([core1 if i % 2 == 0 else gap for i in range(n - 1)], -1).reshape(n // 2, n)
    down = torch.cat([core2 if i % 2 == 0 else gap for i in range(n - 1)], -1).reshape(n // 2, n)
    return torch.cat([upper, down], 0)


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


def plot(tensor):
    plt.imshow(renormFn(tensor).permute([0, 2, 3, 1])[0])
    plt.show()


IMG = Image.open('./etc/lena512color.tiff')
IMG = torch.from_numpy(np.array(IMG)).permute([2, 0, 1])
IMG = IMG.reshape(1, *IMG.shape).float()

depth = 2

ul = IMG

blockSize = IMG.shape[-1]
UR = []
DL = []
DR = []
for i in range(depth):
    transMatrix = buildTransMatrix(blockSize)
    ul = torch.matmul(ul, transMatrix.t())
    blockSize //= 2

    down = ul[:, :, :, blockSize:]
    upper = ul[:, :, :, :blockSize]

    down = torch.matmul(down.transpose(-1, -2), transMatrix.t()).transpose(-1, -2)
    upper = torch.matmul(upper.transpose(-1, -2), transMatrix.t()).transpose(-1, -2)

    ul = upper[:, :, :blockSize, :]
    ur = upper[:, :, blockSize:, :]
    dl = down[:, :, :blockSize, :]
    dr = down[:, :, blockSize:, :]

    UR.append(ur)
    DL.append(dl)
    DR.append(dr)

for i in reversed(range(depth)):
    ur = UR[i]
    dl = DL[i]
    dr = DR[i]

    blockSize *= 2
    upper = torch.cat([ul, ur], -1)
    down = torch.cat([dl, dr], -1)

    ul = torch.cat([upper, down], -2)

UR = []
DL = []
DR = []
for _ in range(depth):
    blockSize //= 2
    UR.append(renormFn(ul[:, :, :blockSize, blockSize:]))
    DL.append(renormFn(ul[:, :, blockSize:, :blockSize]))
    DR.append(renormFn(ul[:, :, blockSize:, blockSize:]))
    ul = ul[:, :, :blockSize, :blockSize]

ul = renormFn(ul)

for no in reversed(range(depth)):
    ur = UR[no]
    dl = DL[no]
    dr = DR[no]

    upper = torch.cat([ul, ur], -1)
    down = torch.cat([dl, dr], -1)
    ul = torch.cat([upper, down], -2)

zremain = ul.permute([0, 2, 3, 1]).detach().cpu().numpy()

waveletPlot = plt.figure(figsize=(8, 8))
waveletAx = waveletPlot.add_subplot(111)
waveletAx.imshow(zremain[0])
plt.axis('off')
plt.savefig('./WaveletPlot.pdf', bbox_inches="tight", pad_inches=0)
plt.close()




