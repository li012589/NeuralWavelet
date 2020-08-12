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
from copy import deepcopy


def test_bijective():
    decimal = flow.ScalingNshifting(256, -128)
    p = source.MixtureDiscreteLogistic([3, 32, 32], 5, decimal, utils.roundingWidentityGradient)

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(3 * 32 * 16), torch.ones(3 * 32 * 16)])[torch.randperm(3 * 32 * 32)].reshape(1, 3, 32, 32)
        else:
            b = 1 - b
        maskList.append(b)
    maskList = torch.cat(maskList, 0).to(torch.float32)
    tList = [utils.SimpleMLPreshape([3 * 32 * 16, 200, 500, 3 * 32 * 16], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
    f = flow.DiscreteNICE(maskList, tList, decimal, utils.roundingWidentityGradient, p)

    bijective(f)


def test_saveload():
    decimal = flow.ScalingNshifting(256, -128)
    p = source.MixtureDiscreteLogistic([3, 32, 32], 5, decimal, utils.roundingWidentityGradient)

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(3 * 32 * 16), torch.ones(3 * 32 * 16)])[torch.randperm(3 * 32 * 32)].reshape(1, 3, 32, 32)
        else:
            b = 1 - b
        maskList.append(b)
    maskList = torch.cat(maskList, 0).to(torch.float32)
    tList = [utils.SimpleMLPreshape([3 * 32 * 16, 200, 500, 3 * 32 * 16], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
    f = flow.DiscreteNICE(maskList, tList, decimal, utils.roundingWidentityGradient, p)

    p = source.MixtureDiscreteLogistic([3, 32, 32], 5, decimal, utils.roundingWidentityGradient)

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(3 * 32 * 16), torch.ones(3 * 32 * 16)])[torch.randperm(3 * 32 * 32)].reshape(1, 3, 32, 32)
        else:
            b = 1 - b
        maskList.append(b)
    maskList = torch.cat(maskList, 0).to(torch.float32)
    tList = [utils.SimpleMLPreshape([3 * 32 * 16, 200, 500, 3 * 32 * 16], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
    blankf = flow.DiscreteNICE(maskList, tList, decimal, utils.roundingWidentityGradient, p)

    saveload(f,blankf)


def test_grad():
    decimal = flow.ScalingNshifting(256, -128)
    p = source.MixtureDiscreteLogistic([3, 32, 32], 5, decimal, utils.roundingWidentityGradient)

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(3 * 32 * 16), torch.ones(3 * 32 * 16)])[torch.randperm(3 * 32 * 32)].reshape(1, 3, 32, 32)
        else:
            b = 1 - b
        maskList.append(b)
    maskList = torch.cat(maskList, 0).to(torch.float32)
    tList = [utils.SimpleMLPreshape([3 * 32 * 16, 200, 500, 3 * 32 * 16], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
    f = flow.DiscreteNICE(maskList, tList, decimal, utils.roundingWidentityGradient, p)

    fcopy = deepcopy(f)
    fcopy.rounding = torch.round

    field = p.sample(100).detach()
    cfield = deepcopy(field).requires_grad_()
    field.requires_grad_()
    xfield, _ = f.inverse(field)
    xcfield, _ = fcopy.inverse(cfield)
    L = xfield.sum()
    Lc = xcfield.sum()
    L.backward()
    Lc.backward()

    ou = [term for term in f.parameters()]
    ouc = [term for term in fcopy.parameters()]
    assert not np.all(ou[-1].grad.detach().numpy() == ouc[-1].grad.detach().numpy())


def test_integer():
    decimal = flow.ScalingNshifting(256, -128)
    p = source.MixtureDiscreteLogistic([3, 32, 32], 5, decimal, utils.roundingWidentityGradient)

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(3 * 32 * 16), torch.ones(3 * 32 * 16)])[torch.randperm(3 * 32 * 32)].reshape(1, 3, 32, 32)
        else:
            b = 1 - b
        maskList.append(b)
    maskList = torch.cat(maskList, 0).to(torch.float32)
    tList = [utils.SimpleMLPreshape([3 * 32 * 16, 200, 500, 3 * 32 * 16], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
    f = flow.DiscreteNICE(maskList, tList, decimal, utils.roundingWidentityGradient, p)

    x, _ = f.sample(100)
    assert np.all(np.equal(np.mod(x.detach().numpy(), 1), 0))

    zx, _ = f.inverse(x)
    assert np.all(np.equal(np.mod(zx.detach().numpy(), 1), 0))

    xzx, _ = f.forward(zx)
    assert np.all(np.equal(np.mod(xzx.detach().numpy(), 1), 0))


if __name__ == "__main__":
    '''
    test_bijective()
    test_saveload()
    test_integer()
    '''
    test_grad()

    
