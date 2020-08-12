import torch
from torch import nn
from .flow import Flow


class ScalingNshifting(Flow):
    def __init__(self, scaling, shifting, name="ScalingNshifting"):
        super(ScalingNshifting, self).__init__(None, name)
        self.scaling = nn.Parameter(torch.tensor(scaling).float(), requires_grad=False)
        self.shifting = nn.Parameter(torch.tensor(shifting).float(), requires_grad=False)

    def inverse(self, y):
        # to decimal
        return (y + self.shifting) * (1 / self.scaling), y.new_zeros(y.shape[0])

    def inverse_(self, y):
        return (y + self.shifting) * (1 / self.scaling)

    def forward(self, z):
        return z * (self.scaling) - self.shifting, z.new_zeros(z.shape[0])

    def forward_(self, z):
        return z * (self.scaling) - self.shifting