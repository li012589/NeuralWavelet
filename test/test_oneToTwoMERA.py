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
from utils import harrInitMethod1, harrInitMethod2, buildWaveletLayers
from numpy.testing import assert_allclose

#torch.manual_seed(42)

def test_wavelet():

    decimal = flow.ScalingNshifting(256, 0)

    psudoRounding = torch.nn.Identity()

    v = torch.randint(255, [100, 3, 8, 8]).float()

    def buildTransMatrix(n):
        core = torch.tensor([[0.5, 0.5], [-1, 1]])
        gap = torch.zeros(2, n)
        return torch.cat([core if i % 2 == 0 else gap for i in range(n - 1)], -1).reshape(2, n // 2, n).permute([1, 0, 2]).reshape(n, n)

    depth = int(math.log(8, 2))
    up = v
    for _ in range(2):
        blockSize = 8
        DN = []
        for i in range(depth):
            transMatrix = buildTransMatrix(blockSize)
            up = torch.matmul(up, transMatrix.t())
            blockSize //= 2
            up = up.reshape(*up.shape[:-2], up.shape[-2], blockSize, 2).transpose(-1, -2)
            DN.append(up[:, :, :, 1, :])
            up = up[:, :, :, 0, :]
        for i in reversed(range(depth)):
            up = up.reshape(up.shape[0], up.shape[1], up.shape[2], 1, up.shape[3])
            dn = DN[i].reshape(*up.shape)
            blockSize *= 2
            up = torch.cat([up, dn], -2).transpose(-1, -2).reshape(up.shape[0], up.shape[1], up.shape[2], blockSize)
        up = up.transpose(-1, -2)

    transV = up

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

    f = flow.OneToTwoMERA(8, layerList, meanNNlist, scaleNNlist, 2, None, 5, decimal=decimal, rounding=psudoRounding.forward)

    vpp = f.inverse(v)[0]

    assert_allclose(vpp.detach().numpy(), transV.detach().numpy())

    # Test depth
    vp = torch.randint(255, [100, 3, 8, 8]).float()

    depth = 2
    up = vp
    for _ in range(2):
        blockSize = 8
        DN = []
        for i in range(depth):
            transMatrix = buildTransMatrix(blockSize)
            up = torch.matmul(up, transMatrix.t())
            blockSize //= 2
            up = up.reshape(*up.shape[:-2], up.shape[-2], blockSize, 2).transpose(-1, -2)
            DN.append(up[:, :, :, 1, :])
            up = up[:, :, :, 0, :]
        for i in reversed(range(depth)):
            up = up.reshape(up.shape[0], up.shape[1], up.shape[2], 1, up.shape[3])
            dn = DN[i].reshape(*up.shape)
            blockSize *= 2
            up = torch.cat([up, dn], -2).transpose(-1, -2).reshape(up.shape[0], up.shape[1], up.shape[2], blockSize)
        up = up.transpose(-1, -2)

    transVp = up

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
    #test_bijective()
    #test_saveload()