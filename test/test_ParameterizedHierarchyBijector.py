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
from numpy.testing import assert_allclose


def test_bijective():
    decimal = flow.ScalingNshifting(256, -128)

    repeat = 2
    layerList = []
    for _ in range(repeat + 1):
        maskList = []
        for n in range(4):
            if n % 2 == 0:
                b = torch.cat([torch.zeros(3 * 2 * 1), torch.ones(3 * 2 * 1)])[torch.randperm(3 * 2 * 2)].reshape(1, 3, 2, 2)
            else:
                b = 1 - b
            maskList.append(b)
        maskList = torch.cat(maskList, 0).to(torch.float32)
        tList = [utils.SimpleMLPreshape([3 * 2 * 1, 20, 20, 3 * 2 * 1], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
        f = flow.DiscreteNICE(maskList, tList, decimal, utils.roundingWidentityGradient, None)
        layerList.append(f)

    meanNNlist = []
    scaleNNlist = []
    for _ in range(int(math.log(8, 2)) - 1):
        meanNNlist.append(torch.nn.Sequential(torch.nn.Conv2d(3, 9, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 9, 1, padding=0), torch.nn.ReLU(inplace=True)))
        scaleNNlist.append(torch.nn.Sequential(torch.nn.Conv2d(3, 9, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 9, 1, padding=0), torch.nn.ReLU(inplace=True)))

    t = flow.ParameterizedMERA(2, 8, layerList, meanNNlist, scaleNNlist, 5, repeat, None, decimal, utils.roundingWidentityGradient)

    tImage = flow.MERA(2, 8, layerList, repeat, prior=None)

    samples = torch.randint(0, 255, (100, 3, 8, 8)).float()

    zSamples, _ = t.inverse(samples)
    zSamplesImage, _ = tImage.inverse(samples)
    rcnSamples, _ = t.forward(zSamples)
    rcnSamplesImage, _ = t.forward(zSamplesImage)
    prob = t.logProbability(samples)

    assert_allclose(zSamples.detach().numpy(), zSamplesImage.detach().numpy())
    assert_allclose(samples.detach().numpy(), rcnSamples.detach().numpy())
    assert_allclose(samples.detach().numpy(), rcnSamplesImage.detach().numpy())


def test_saveload():
    decimal = flow.ScalingNshifting(256, -128)

    repeat = 2
    layerList = []
    for _ in range(repeat + 1):
        maskList = []
        for n in range(4):
            if n % 2 == 0:
                b = torch.cat([torch.zeros(3 * 2 * 1), torch.ones(3 * 2 * 1)])[torch.randperm(3 * 2 * 2)].reshape(1, 3, 2, 2)
            else:
                b = 1 - b
            maskList.append(b)
        maskList = torch.cat(maskList, 0).to(torch.float32)
        tList = [utils.SimpleMLPreshape([3 * 2 * 1, 20, 20, 3 * 2 * 1], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
        f = flow.DiscreteNICE(maskList, tList, decimal, utils.roundingWidentityGradient, None)
        layerList.append(f)

    meanNNlist = []
    scaleNNlist = []
    for _ in range(int(math.log(8, 2)) - 1):
        meanNNlist.append(torch.nn.Sequential(torch.nn.Conv2d(3, 9, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 9, 1, padding=0), torch.nn.ReLU(inplace=True)))
        scaleNNlist.append(torch.nn.Sequential(torch.nn.Conv2d(3, 9, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 9, 1, padding=0), torch.nn.ReLU(inplace=True)))

    t = flow.ParameterizedMERA(2, 8, layerList, meanNNlist, scaleNNlist, 5, repeat, None, decimal, utils.roundingWidentityGradient)

    decimal = flow.ScalingNshifting(256, -128)

    repeat = 2
    layerList = []
    for _ in range(repeat + 1):
        maskList = []
        for n in range(4):
            if n % 2 == 0:
                b = torch.cat([torch.zeros(3 * 2 * 1), torch.ones(3 * 2 * 1)])[torch.randperm(3 * 2 * 2)].reshape(1, 3, 2, 2)
            else:
                b = 1 - b
            maskList.append(b)
        maskList = torch.cat(maskList, 0).to(torch.float32)
        tList = [utils.SimpleMLPreshape([3 * 2 * 1, 20, 20, 3 * 2 * 1], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
        f = flow.DiscreteNICE(maskList, tList, decimal, utils.roundingWidentityGradient, None)
        layerList.append(f)

    meanNNlist = []
    scaleNNlist = []
    for _ in range(int(math.log(8, 2)) - 1):
        meanNNlist.append(torch.nn.Sequential(torch.nn.Conv2d(3, 9, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 9, 1, padding=0), torch.nn.ReLU(inplace=True)))
        scaleNNlist.append(torch.nn.Sequential(torch.nn.Conv2d(3, 9, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(9, 9, 1, padding=0), torch.nn.ReLU(inplace=True)))

    tt = flow.ParameterizedMERA(2, 8, layerList, meanNNlist, scaleNNlist, 5, repeat, None, decimal, utils.roundingWidentityGradient)


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
    test_bijective()
    test_saveload()