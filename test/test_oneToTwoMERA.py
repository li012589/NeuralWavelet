from flowRelated import *

import os
import sys
sys.path.append(os.getcwd())

import torch, math
from torch import nn
import numpy as np
import utils
import flow
import source
import utils
from PIL import Image
from utils import harrInitMethod1, harrInitMethod2, buildWaveletLayers
from numpy.testing import assert_allclose
from matplotlib import pyplot as plt

#torch.manual_seed(42)

def test_wavelet():

    def back01(tensor):
        ten = tensor.clone().float()
        ten = ten.view(ten.shape[0] * ten.shape[1], -1)
        ten -= ten.min(1, keepdim=True)[0]
        ten /= ten.max(1, keepdim=True)[0]
        ten = ten.view(tensor.shape)
        return ten


    # yet another renorm fn
    def batchNorm(tensor, base=1.0):
        m = nn.BatchNorm2d(tensor.shape[1], affine=False)
        return m(tensor).float() + base


    renormFn = lambda x: back01(batchNorm(x))


    def im2grp(t):
        return t.reshape(t.shape[0], t.shape[1], t.shape[2] // 2, 2, t.shape[3] // 2, 2).permute([0, 1, 2, 4, 3, 5]).reshape(t.shape[0], t.shape[1], -1, 4)


    def grp2im(t):
        return t.reshape(t.shape[0], t.shape[1], int(t.shape[2] ** 0.5), int(t.shape[2] ** 0.5), 2, 2).permute([0, 1, 2, 4, 3, 5]).reshape(t.shape[0], t.shape[1], int(t.shape[2] ** 0.5) * 2, int(t.shape[2] ** 0.5) * 2)


    decimal = flow.ScalingNshifting(256, 0)

    psudoRounding = torch.nn.Identity()

    IMG = Image.open('./etc/lena512color.tiff')
    IMG = torch.from_numpy(np.array(IMG)).permute([2, 0, 1])
    IMG = IMG.reshape(1, *IMG.shape).float()

    v = IMG

    def buildTransMatrix(n):
        core = torch.tensor([[0.5, 0.5], [-1, 1]])
        gap = torch.zeros(2, n)
        return torch.cat([core if i % 2 == 0 else gap for i in range(n - 1)], -1).reshape(2, n // 2, n).permute([1, 0, 2]).reshape(n, n)

    depth = int(math.log(v.shape[-1], 2))
    up = v

    blockSize = v.shape[-1]

    UR = []
    DL = []
    DR = []

    for i in range(depth):
        transMatrix = buildTransMatrix(blockSize)
        for _ in range(2):
            up = torch.matmul(up, transMatrix.t())
            up = up.permute([0, 1, 3, 2])
        blockSize //= 2

        _x = im2grp(up)
        ul = _x[:, :, :, 0].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
        ur = _x[:, :, :, 1].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
        dl = _x[:, :, :, 2].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
        dr = _x[:, :, :, 3].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()

        UR.append(ur)
        DL.append(dl)
        DR.append(dr)

        up = ul

    ul = up

    for no in reversed(range(depth)):
        ur = UR[no].reshape(*ul.shape, 1)
        dl = DL[no].reshape(*ul.shape, 1)
        dr = DR[no].reshape(*ul.shape, 1)
        ul = ul.reshape(*ul.shape, 1)

        _x = torch.cat([ul, ur, dl, dr], -1).reshape(*ul.shape[:2], -1, 4)
        ul = grp2im(_x).contiguous()

    transV = ul

    '''
    ul = ul
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

    #ul = back01(backMeanStd(batchNorm(ul, 0)))
    ul = renormFn(ul)
    #ul = back01(clip(backMeanStd(batchNorm(ul))))

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
    waveletAx.imshow(zremain[0])
    plt.axis('off')
    plt.savefig('./testWavelet.pdf', bbox_inches="tight", pad_inches=0)
    plt.close()
    '''

    initMethods = []
    initMethods.append(lambda: harrInitMethod1(3))
    initMethods.append(lambda: harrInitMethod2(3))

    orders = [True, False]

    layerList = []
    for j in range(2):
        layerList.append(buildWaveletLayers(initMethods[j], 3, 12, 1, orders[j]))

    shapeList2D = [3] + [12] * (1 + 1) + [3 * 3]
    shapeList1D = [3] + [12] * (1 + 1) + [3]

    def buildLayers2D(shapeList):
        layers = []
        for no, chn in enumerate(shapeList[:-1]):
            if no != 0 and no != len(shapeList) - 2:
                layers.append(torch.nn.Conv2d(chn, shapeList[no + 1], 1))
            else:
                layers.append(torch.nn.Conv2d(chn, shapeList[no + 1], 3, padding=1))
            if no != len(shapeList) - 2:
                layers.append(torch.nn.ReLU(inplace=True))
        return layers

    def buildLayers1D(shapeList):
        layers = []
        for no, chn in enumerate(shapeList[:-1]):
            if no != 0 and no != len(shapeList) - 2:
                layers.append(torch.nn.Conv1d(chn, shapeList[no + 1], 1))
            else:
                layers.append(torch.nn.Conv1d(chn, shapeList[no + 1], 3, padding=1))
            if no != len(shapeList) - 2:
                layers.append(torch.nn.ReLU(inplace=True))
        layers = torch.nn.Sequential(*layers)
        torch.nn.init.zeros_(layers[-1].weight)
        torch.nn.init.zeros_(layers[-1].bias)
        return layers

    # repeat, add one more layer of NICE
    for _ in range(2):
        layerList.append(buildLayers1D(shapeList1D))

    meanNNlist = []
    scaleNNlist = []
    layers = buildLayers2D(shapeList2D)
    meanNNlist.append(torch.nn.Sequential(*layers))
    layers = buildLayers2D(shapeList2D)
    scaleNNlist.append(torch.nn.Sequential(*layers))
    torch.nn.init.zeros_(meanNNlist[-1][-1].weight)
    torch.nn.init.zeros_(meanNNlist[-1][-1].bias)
    torch.nn.init.zeros_(scaleNNlist[-1][-1].weight)
    torch.nn.init.zeros_(scaleNNlist[-1][-1].bias)

    f = flow.OneToTwoMERA(v.shape[-1], layerList, meanNNlist, scaleNNlist, repeat=2, depth=depth, nMixing=5, decimal=decimal, rounding=psudoRounding.forward)

    vpp = f.inverse(v)[0]

    assert_allclose(vpp.detach().numpy(), transV.detach().numpy())

    '''
    vpp = ul
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

    #ul = back01(backMeanStd(batchNorm(ul, 0)))
    ul = renormFn(ul)
    #ul = back01(clip(backMeanStd(batchNorm(ul))))

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
    waveletAx.imshow(zremain[0])
    plt.axis('off')
    plt.savefig('./testWavelet2.pdf', bbox_inches="tight", pad_inches=0)
    plt.close()
    '''
    # Test depth
    vp = IMG

    depth = 2
    up = vp
    blockSize = v.shape[-1]

    UR = []
    DL = []
    DR = []

    for i in range(depth):
        transMatrix = buildTransMatrix(blockSize)
        for _ in range(2):
            up = torch.matmul(up, transMatrix.t())
            up = up.permute([0, 1, 3, 2])
        blockSize //= 2

        _x = im2grp(up)
        ul = _x[:, :, :, 0].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
        ur = _x[:, :, :, 1].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
        dl = _x[:, :, :, 2].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
        dr = _x[:, :, :, 3].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()

        UR.append(ur)
        DL.append(dl)
        DR.append(dr)

        up = ul

    ul = up

    for no in reversed(range(depth)):
        ur = UR[no].reshape(*ul.shape, 1)
        dl = DL[no].reshape(*ul.shape, 1)
        dr = DR[no].reshape(*ul.shape, 1)
        ul = ul.reshape(*ul.shape, 1)

        _x = torch.cat([ul, ur, dl, dr], -1).reshape(*ul.shape[:2], -1, 4)
        ul = grp2im(_x).contiguous()

    transVp = ul

    fp = flow.OneToTwoMERA(8, layerList, meanNNlist, scaleNNlist, 2, depth, 5, decimal=decimal, rounding=psudoRounding.forward)

    vpp = fp.inverse(vp)[0]

    assert_allclose(vpp.detach().numpy(), transVp.detach().numpy())

def test_bijective():

    shapeList2D = [3] + [12] * (1 + 1) + [3 * 3]
    shapeList1D = [3] + [12] * (1 + 1) + [3]

    def buildLayers2D(shapeList):
        layers = []
        for no, chn in enumerate(shapeList[:-1]):
            if no != 0 and no != len(shapeList) - 2:
                layers.append(torch.nn.Conv2d(chn, shapeList[no + 1], 1))
            else:
                layers.append(torch.nn.Conv2d(chn, shapeList[no + 1], 3, padding=1))
            if no != len(shapeList) - 2:
                layers.append(torch.nn.ReLU(inplace=True))
        return layers

    def buildLayers1D(shapeList):
        layers = []
        for no, chn in enumerate(shapeList[:-1]):
            if no != 0 and no != len(shapeList) - 2:
                layers.append(torch.nn.Conv1d(chn, shapeList[no + 1], 1))
            else:
                layers.append(torch.nn.Conv1d(chn, shapeList[no + 1], 3, padding=1))
            if no != len(shapeList) - 2:
                layers.append(torch.nn.ReLU(inplace=True))
        layers = torch.nn.Sequential(*layers)
        #torch.nn.init.zeros_(layers[-1].weight)
        #torch.nn.init.zeros_(layers[-1].bias)
        return layers

    decimal = flow.ScalingNshifting(256, 0)

    layerList = []
    for i in range(2 * 2):
        layerList.append(buildLayers1D(shapeList1D))

    meanNNlist = []
    scaleNNlist = []
    layers = buildLayers2D(shapeList2D)
    meanNNlist.append(torch.nn.Sequential(*layers))
    layers = buildLayers2D(shapeList2D)
    scaleNNlist.append(torch.nn.Sequential(*layers))

    t = flow.OneToTwoMERA(8, layerList, meanNNlist, scaleNNlist, 2, None, 5, decimal=decimal, rounding=utils.roundingWidentityGradient)

    samples = torch.randint(0, 255, (100, 3, 8, 8)).float()

    zSamples, _ = t.inverse(samples)

    rcnSamples, _ = t.forward(zSamples)
    prob = t.logProbability(samples)

    assert_allclose(samples.detach().numpy(), rcnSamples.detach().numpy())

    # Test depth argument
    t = flow.OneToTwoMERA(8, layerList, meanNNlist, scaleNNlist, 2, 2, 5, decimal=decimal, rounding=utils.roundingWidentityGradient)

    samples = torch.randint(0, 255, (100, 3, 8, 8)).float()

    zSamples, _ = t.inverse(samples)
    rcnSamples, _ = t.forward(zSamples)
    prob = t.logProbability(samples)

    assert_allclose(samples.detach().numpy(), rcnSamples.detach().numpy())


def test_saveload():
    shapeList2D = [3] + [12] * (1 + 1) + [3 * 3]
    shapeList1D = [3] + [12] * (1 + 1) + [3]

    def buildLayers2D(shapeList):
        layers = []
        for no, chn in enumerate(shapeList[:-1]):
            if no != 0 and no != len(shapeList) - 2:
                layers.append(torch.nn.Conv2d(chn, shapeList[no + 1], 1))
            else:
                layers.append(torch.nn.Conv2d(chn, shapeList[no + 1], 3, padding=1))
            if no != len(shapeList) - 2:
                layers.append(torch.nn.ReLU(inplace=True))
        return layers

    def buildLayers1D(shapeList):
        layers = []
        for no, chn in enumerate(shapeList[:-1]):
            if no != 0 and no != len(shapeList) - 2:
                layers.append(torch.nn.Conv1d(chn, shapeList[no + 1], 1))
            else:
                layers.append(torch.nn.Conv1d(chn, shapeList[no + 1], 3, padding=1))
            if no != len(shapeList) - 2:
                layers.append(torch.nn.ReLU(inplace=True))
        layers = torch.nn.Sequential(*layers)
        #torch.nn.init.zeros_(layers[-1].weight)
        #torch.nn.init.zeros_(layers[-1].bias)
        return layers

    decimal = flow.ScalingNshifting(256, 0)

    layerList = []
    for i in range(2 * 2):
        layerList.append(buildLayers1D(shapeList1D))

    meanNNlist = []
    scaleNNlist = []
    layers = buildLayers2D(shapeList2D)
    meanNNlist.append(torch.nn.Sequential(*layers))
    layers = buildLayers2D(shapeList2D)
    scaleNNlist.append(torch.nn.Sequential(*layers))

    t = flow.OneToTwoMERA(8, layerList, meanNNlist, scaleNNlist, 2, None, 5, decimal=decimal, rounding=utils.roundingWidentityGradient)


    decimal = flow.ScalingNshifting(256, 0)

    layerList = []
    for i in range(2 * 2):
        layerList.append(buildLayers1D(shapeList1D))

    meanNNlist = []
    scaleNNlist = []
    layers = buildLayers2D(shapeList2D)
    meanNNlist.append(torch.nn.Sequential(*layers))
    layers = buildLayers2D(shapeList2D)
    scaleNNlist.append(torch.nn.Sequential(*layers))

    tt = flow.OneToTwoMERA(8, layerList, meanNNlist, scaleNNlist, 2, None, 5, decimal=decimal, rounding=utils.roundingWidentityGradient)

    samples = torch.randint(0, 255, (100, 3, 8, 8)).float()

    torch.save(t.save(), "testsaving.saving")
    tt.load(torch.load("testsaving.saving"))

    tzSamples, _ = t.inverse(samples)
    ttzSamples, _ = tt.inverse(samples)

    rcnSamples, _ = t.forward(tzSamples)
    ttrcnSamples, _ = tt.forward(ttzSamples)

    assert_allclose(tzSamples.detach().numpy(), ttzSamples.detach().numpy())
    assert_allclose(samples.detach().numpy(), rcnSamples.detach().numpy())
    assert_allclose(rcnSamples.detach().numpy(), ttrcnSamples.detach().numpy())


if __name__ == "__main__":
    test_wavelet()
    test_bijective()
    #test_saveload()