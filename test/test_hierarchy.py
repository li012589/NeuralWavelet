from flowRelated import *

import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
import numpy as np
import utils
import flow
import source
import utils


def test_bijective():
    decimal = flow.ScalingNshifting(256, -128)
    p = source.DiscreteLogistic([3, 4, 4], decimal, utils.roundingWidentityGradient)

    layerList = []
    for _ in range(3):
        maskList = []
        for n in range(4):
            if n % 2 == 0:
                b = torch.cat([torch.zeros(3 * 2 * 1), torch.ones(3 * 2 * 1)])[torch.randperm(3 * 2 * 2)].reshape(1, 3, 2, 2)
            else:
                b = 1 - b
            maskList.append(b)
        maskList = torch.cat(maskList, 0).to(torch.float32)
        tList = [utils.SimpleMLPreshape([3 * 2 * 1, 20, 20, 3 * 2 * 1], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
        f = flow.DiscreteNICE(maskList, tList, decimal, utils.roundingWidentityGradient, p)
        layerList.append(f)

    t = flow.MERA(2, 4, layerList, 2, prior=p)


    bijective(t)



def test_saveload():
    decimal = flow.ScalingNshifting(256, -128)
    p = source.DiscreteLogistic([3, 4, 4], decimal, utils.roundingWidentityGradient)

    layerList = []
    for _ in range(3):
        maskList = []
        for n in range(4):
            if n % 2 == 0:
                b = torch.cat([torch.zeros(3 * 2 * 1), torch.ones(3 * 2 * 1)])[torch.randperm(3 * 2 * 2)].reshape(1, 3, 2, 2)
            else:
                b = 1 - b
            maskList.append(b)
        maskList = torch.cat(maskList, 0).to(torch.float32)
        tList = [utils.SimpleMLPreshape([3 * 2 * 1, 20, 20, 3 * 2 * 1], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
        f = flow.DiscreteNICE(maskList, tList, decimal, utils.roundingWidentityGradient, p)
        layerList.append(f)

    t = flow.MERA(2, 4, layerList, 2, prior=p)

    p = source.DiscreteLogistic([3, 4, 4], decimal, utils.roundingWidentityGradient)

    layerList = []
    for _ in range(3):
        maskList = []
        for n in range(4):
            if n % 2 == 0:
                b = torch.cat([torch.zeros(3 * 2 * 1), torch.ones(3 * 2 * 1)])[torch.randperm(3 * 2 * 2)].reshape(1, 3, 2, 2)
            else:
                b = 1 - b
            maskList.append(b)
        maskList = torch.cat(maskList, 0).to(torch.float32)
        tList = [utils.SimpleMLPreshape([3 * 2 * 1, 20, 20, 3 * 2 * 1], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
        f = flow.DiscreteNICE(maskList, tList, decimal, utils.roundingWidentityGradient, p)
        layerList.append(f)

    blankt = flow.MERA(2, 4, layerList, 2, prior=p)

    saveload(t, blankt)


if __name__ == "__main__":
    test_bijective()
    test_saveload()