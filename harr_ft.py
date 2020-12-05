import torch
import numpy as np

from PIL import Image

from matplotlib import pyplot as plt


IMG = Image.open('./etc/lena512color.tiff')
IMG = torch.from_numpy(np.array(IMG)).permute([2, 0, 1])
IMG = IMG.reshape(1, *IMG.shape).float()


def buildTransMatrix(n):
    core1 = torch.tensor([[1, 1]])
    core2 = torch.tensor([[-1, 1]])
    gap = torch.zeros(1, n)
    upper = torch.cat([core1 if i % 2 == 0 else gap for i in range(n - 1)], -1).reshape(n // 2, n)
    down = torch.cat([core2 if i % 2 == 0 else gap for i in range(n - 1)], -1).reshape(n // 2, n)
    return torch.cat([upper, down], 0)

line = IMG[0, 0, 40, :]

n = line.shape[0]

transMatrix = buildTransMatrix(n)
zline = torch.matmul(line, transMatrix.t())

(torch.matmul(zline, transMatrix) / 2).allclose(line)

zlineHalf = zline.clone()
zlineHalf[256:] = 0

lowline = torch.matmul(zlineHalf, transMatrix) / 2

zlineHalfAo = zline.clone()
zlineHalfAo[:256] = 0

highline = torch.matmul(zlineHalfAo, transMatrix) / 2

def fft(x):
    f = np.fft.fft(x.detach().numpy())
    f = np.fft.fftshift(f)
    f = np.log(np.abs(f) + 1)
    return f

import pdb
pdb.set_trace()

plt.figure()
plt.plot(line)
plt.figure()
plt.plot(fft(line))
plt.figure()
plt.plot(fft(lowline))
plt.figure()
plt.plot(fft(highline))

plt.show()


