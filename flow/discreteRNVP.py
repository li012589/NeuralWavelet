import torch
from torch import nn
from numpy.testing import assert_array_equal
from .flow import Flow

import numpy as np

class DiscreteRNVP(Flow):
    def __init__(self, maskList, tList, sList, decimal, rounding, prior=None, name="DiscreteRNVP"):
        super(DiscreteRNVP, self).__init__(prior, name)

        assert len(tList) == len(sList)
        assert len(tList) == len(maskList)

        self.maskList = nn.Parameter(maskList, requires_grad=False)

        self.tList = torch.nn.ModuleList(tList)
        self.sList = torch.nn.ModuleList(sList)
        self.decimal = decimal
        self.rounding = rounding

    def inverse(self, y, rounding=True):
        inverseLogjac = y.new_zeros(y.shape[0])
        tmplist = []
        YAs = []
        Ys = []
        for i in range(len(self.tList)):
            maskR = (1 - self.maskList[i]).bool()
            mask = self.maskList[i].bool()
            yA = torch.masked_select(y, mask).reshape(y.shape[0], -1)
            Ys.append([yA, torch.masked_select(y, maskR).reshape(y.shape[0], -1)])
            yB = self.decimal.inverse_(torch.masked_select(y, maskR).reshape(y.shape[0], -1))

            t = self.decimal.forward_(self.tList[i](yB))
            s = self.sList[i](yB)
            assert_array_equal(t.shape, s.shape)
            assert_array_equal(yB.shape, s.shape)

            tmplist.append([self.rounding(t + yA * torch.exp(s)), t + yA * s, t, s])
            yA = yA + self.rounding(t + yA * torch.exp(s))
            YAs.append(yA)
            y = y.masked_scatter(mask, yA).contiguous()
        self.tmplist = tmplist
        self.YAs = YAs
        self.Ys = Ys
        return y, inverseLogjac

    def forward(self, z, rounding=True):
        forwardLogjac = z.new_zeros(z.shape[0])
        for i in reversed(range(len(self.tList))):
            maskR = (1 - self.maskList[i]).bool()
            mask = self.maskList[i].bool()
            zA = torch.masked_select(z, mask).reshape(z.shape[0], -1)
            zzB = torch.masked_select(z, maskR).reshape(z.shape[0], -1)
            zB = self.decimal.inverse_(torch.masked_select(z, maskR).reshape(z.shape[0], -1))

            t = self.decimal.forward_(self.tList[i](zB))
            s = self.sList[i](zB)
            assert_array_equal(t.shape, s.shape)
            assert_array_equal(zB.shape, s.shape)
            zA = zA - self.rounding(t + zA * torch.exp(-s))
            z = z.masked_scatter(mask, zA).contiguous()
        return z, forwardLogjac


