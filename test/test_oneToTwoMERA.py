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
    for i in range(1):
        for j in range(2):
            layerList.append(buildWaveletLayers(initMethods[j], 3, 12, 1, orders[j]))

    shapeList = [3] + [12] * (1 + 1) + [3 * 3]

    def buildLayers(shapeList):
        layers = []
        for no, chn in enumerate(shapeList[:-1]):
            if no != 0 and no != len(shapeList) - 2:
                layers.append(torch.nn.Conv2d(chn, shapeList[no + 1], 1))
            else:
                layers.append(torch.nn.Conv2d(chn, shapeList[no + 1], 3, padding=1))
            if no != len(shapeList) - 2:
                layers.append(torch.nn.ReLU(inplace=True))
        return layers

    meanNNlist = []
    scaleNNlist = []
    layers = buildLayers(shapeList)
    meanNNlist.append(torch.nn.Sequential(*layers))
    layers = buildLayers(shapeList)
    scaleNNlist.append(torch.nn.Sequential(*layers))
    torch.nn.init.zeros_(meanNNlist[-1][-1].weight)
    torch.nn.init.zeros_(meanNNlist[-1][-1].bias)
    torch.nn.init.zeros_(scaleNNlist[-1][-1].weight)
    torch.nn.init.zeros_(scaleNNlist[-1][-1].bias)

    f = flow.OneToTwoMERA(8, layerList, meanNNlist, scaleNNlist, 1, 5, decimal=decimal, rounding=psudoRounding.forward)

    vpp = f.inverse(v)[0]

    assert_allclose(vpp.detach().numpy(), transV.detach().numpy())


def test_bijective():

    decimal = flow.ScalingNshifting(256, 0)

    layerList = []
    for i in range(4 * 2):
        f = torch.nn.Sequential(torch.nn.Conv2d(9, 9, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 9, 1, padding=0), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 3, 3, padding=1))
        layerList.append(f)

    meanNNlist = []
    scaleNNlist = []
    meanNNlist.append(torch.nn.Sequential(torch.nn.Conv2d(3, 9, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 9, 1, padding=0), torch.nn.ReLU(inplace=True)))
    scaleNNlist.append(torch.nn.Sequential(torch.nn.Conv2d(3, 9, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 9, 1, padding=0), torch.nn.ReLU(inplace=True)))

    t = flow.SimpleMERA(8, layerList, meanNNlist, scaleNNlist, 2, 5, decimal, utils.roundingWidentityGradient)

    samples = torch.randint(0, 255, (100, 3, 8, 8)).float()

    zSamples, _ = t.inverse(samples)
    rcnSamples, _ = t.forward(zSamples)
    prob = t.logProbability(samples)

    assert_allclose(samples.detach().numpy(), rcnSamples.detach().numpy())


def test_saveload():
    decimal = flow.ScalingNshifting(256, -128)

    layerList = []
    for i in range(4):
        f = torch.nn.Sequential(torch.nn.Conv2d(9, 9, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 9, 1, padding=0), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 3, 3, padding=1))
        layerList.append(f)

    meanNNlist = []
    scaleNNlist = []
    meanNNlist.append(torch.nn.Sequential(torch.nn.Conv2d(3, 9, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 9, 1, padding=0), torch.nn.ReLU(inplace=True)))
    scaleNNlist.append(torch.nn.Sequential(torch.nn.Conv2d(3, 9, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 9, 1, padding=0), torch.nn.ReLU(inplace=True)))

    t = flow.SimpleMERA(8, layerList, meanNNlist, scaleNNlist, 1, 5, decimal, utils.roundingWidentityGradient)

    decimal = flow.ScalingNshifting(256, -128)

    layerList = []
    for i in range(4):
        f = torch.nn.Sequential(torch.nn.Conv2d(9, 9, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 9, 1, padding=0), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 3, 3, padding=1))
        layerList.append(f)

    meanNNlist = []
    scaleNNlist = []
    meanNNlist.append(torch.nn.Sequential(torch.nn.Conv2d(3, 9, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 9, 1, padding=0), torch.nn.ReLU(inplace=True)))
    scaleNNlist.append(torch.nn.Sequential(torch.nn.Conv2d(3, 9, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 9, 1, padding=0), torch.nn.ReLU(inplace=True)))

    tt = flow.SimpleMERA(8, layerList, meanNNlist, scaleNNlist, 1, 5, decimal, utils.roundingWidentityGradient)

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