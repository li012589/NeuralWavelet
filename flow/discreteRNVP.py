import torch
from torch import nn
from numpy.testing import assert_array_equal
from .flow import Flow


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
        for i in range(len(self.tList)):
            mask = self.maskList[i].bool()
            maskR = (1 - mask).bool()
            yA = torch.masked_select(y, mask).reshape(y.shape[0], -1)
            yB = self.decimal.inverse(torch.masked_select(y, maskR).reshape(y.shape[0], -1))

            t = self.decimal.forward(self.tList[i](yA))
            s = self.decimal.forward(self.sList[i](yA))
            assert_array_equal(t.shape, s.shape)
            assert_array_equal(yB.shape, s.shape)

            yB = yB + self.rounding(t + yB * s)
            y = y.masked_scatter(maskR, yB).contiguous()
        return y, inverseLogjac

    def forward(self, z, rounding=True):
        forwardLogjac = z.new_zeros(z.shape[0])
        for i in reversed(range(len(self.tList))):
            mask = self.maskList[i].bool()
            maskR = (1 - mask).bool()
            zA = torch.masked_select(z, mask).reshape(z.shape[0], -1)
            zB = self.decimal.inverse(torch.masked_select(z, maskR).reshape(z.shape[0], -1))

            t = self.deimal.forward(self.tList[i](zA))
            s = self.decimal.forward(self.sList[i](zA))
            assert_array_equal(t.shape, s.shape)
            assert_array_equal(zB.shape, s.shape)

            zB = zB - self.rounding(t + zB * s)
            z = z.masked_scatter(maskR, zB).contiguous()
        return z, forwardLogjac


