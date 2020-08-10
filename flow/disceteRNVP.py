import torch
from torch import nn

from .flow import Flow

class DiscreteRNVP(Flow):
    def __init__(self, maskList, tList, sList, scaling, prior=None, name="DiscreteRNVP"):
        super(DiscreteRNVP, self).__init__(prior, name)

        assert len(tList) == len(sList)
        assert len(tList) == len(maskList)

        self.maskList = nn.Parameter(maskList, requires_grad=False)
        self.maskListR = nn.Parameter(1 - maskList, requires_grad=False)
        self.scaling = nn.Parameter(scaling, requires_grad=False)

        self.tList = torch.nn.ModuleList(tList)
        self.sList = torch.nn.ModuleList(sList)

    def inverse(self, y, rounding=True):
        pass

    def forward(self, z, rounding=True):
        pass

    
