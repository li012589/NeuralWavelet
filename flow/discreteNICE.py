import torch
from torch import nn
from numpy.testing import assert_array_equal
from .flow import Flow


class DiscreteNICE(Flow):
    def __init__(self, maskList, tList, decimal, rounding, prior=None, name="DiscreteNICE"):
        super(DiscreteNICE, self).__init__(prior, name)

        assert len(tList) == len(maskList)

        self.maskList = nn.Parameter(maskList, requires_grad=False)

        self.tList = torch.nn.ModuleList(tList)

        self.decimal = decimal
        self.rounding = rounding

    def inverse(self, y):
        inverseLogjac = y.new_zeros(y.shape[0])
        for i in range(len(self.tList)):
            maskR = (1 - self.maskList[i]).bool()
            mask = self.maskList[i].bool()
            yA = torch.masked_select(y, mask).reshape(y.shape[0], -1)
            yB = self.decimal.inverse_(torch.masked_select(y, maskR).reshape(y.shape[0], -1))

            t = self.decimal.forward_(self.tList[i](yB))
            assert_array_equal(t.shape, yB.shape)

            yA = yA + self.rounding(t)
            y = y.masked_scatter(mask, yA).contiguous()
        return y, inverseLogjac

    def forward(self, z):
        forwardLogjac = z.new_zeros(z.shape[0])
        for i in reversed(range(len(self.tList))):
            maskR = (1 - self.maskList[i]).bool()
            mask = self.maskList[i].bool()
            zA = torch.masked_select(z, mask).reshape(z.shape[0], -1)
            zB = self.decimal.inverse_(torch.masked_select(z, maskR).reshape(z.shape[0], -1))

            t = self.decimal.forward_(self.tList[i](zB))
            assert_array_equal(t.shape, zB.shape)

            zA = zA - self.rounding(t)
            z = z.masked_scatter(mask, zA).contiguous()
        return z, forwardLogjac


