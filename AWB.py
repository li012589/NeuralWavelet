import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


image = Image.open('./opt/base/default_easyMera_ImageNet64_simplePrior_False_repeat_3_hchnl_350_nhidden_3_nMixing_5_sameDetail_True_clamp_-1_heavy_False/pic/exoloadPlot_N_9_P_0.png')
image = image.convert("RGB")
IMG = torch.from_numpy(np.array(image))
IMGtorch = IMG.permute([2, 0, 1]).reshape(1, 3, *IMG.shape[0:2]).float()

image = Image.open('./opt/base/default_easyMera_ImageNet64_simplePrior_False_repeat_3_hchnl_350_nhidden_3_nMixing_5_sameDetail_True_clamp_-1_heavy_False/pic/proloadPlot_N_9_P_0.png')
image = image.convert("RGB")
groundTruth = torch.from_numpy(np.array(image))

plot = plt.figure(figsize=(8, 8))
ax = plot.add_subplot(111)
ax.imshow(groundTruth)


def grayWorld(tensor):
    meanRGB = tensor.reshape(tensor.shape[0], 3, -1).mean(-1)
    gray = meanRGB.sum(-1, keepdim=True) / 3
    scaleRGB = gray / meanRGB
    scaledTensor = torch.round(tensor.reshape(tensor.shape[0], 3, -1) * scaleRGB.reshape(*scaleRGB.shape, 1)).reshape(tensor.shape)
    return torch.clamp(scaledTensor, 0, 255).int()


def grey_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)
    return  nimg.transpose(1, 2, 0).astype(np.uint8)

def retinex(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = nimg[1].max()
    nimg[0] = np.minimum(nimg[0]*(mu_g/float(nimg[0].max())),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/float(nimg[2].max())),255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def retinex_adjust(nimg):
    """
    from 'Combining Gray World and Retinex Theory for Automatic White Balance in Digital Photography'
    """
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    sum_r = np.sum(nimg[0])
    sum_r2 = np.sum(nimg[0]**2)
    max_r = nimg[0].max()
    max_r2 = max_r**2
    sum_g = np.sum(nimg[1])
    max_g = nimg[1].max()
    coefficient = np.linalg.solve(np.array([[sum_r2,sum_r],[max_r2,max_r]]),
                                  np.array([sum_g,max_g]))
    nimg[0] = np.minimum((nimg[0]**2)*coefficient[0] + nimg[0]*coefficient[1],255)
    sum_b = np.sum(nimg[1])
    sum_b2 = np.sum(nimg[1]**2)
    max_b = nimg[1].max()
    max_b2 = max_r**2
    coefficient = np.linalg.solve(np.array([[sum_b2,sum_b],[max_b2,max_b]]),
                                             np.array([sum_g,max_g]))
    nimg[1] = np.minimum((nimg[1]**2)*coefficient[0] + nimg[1]*coefficient[1],255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def retinex_with_adjust(nimg):
    return retinex_adjust(retinex(nimg))


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


plot = plt.figure(figsize=(8, 8))
ax = plot.add_subplot(111)
ax.imshow(IMG)


plot = plt.figure(figsize=(8, 8))
ax = plot.add_subplot(111)

IMGtorch = perfReflect(IMGtorch)[0].permute([1, 2, 0]).detach().numpy()

ax.imshow(IMGtorch)
'''
plot = plt.figure(figsize=(8, 8))
ax = plot.add_subplot(111)

ax.imshow(retinex_with_adjust(IMG.numpy()))
'''


plt.show()