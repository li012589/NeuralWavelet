import torch
from torch import nn
import numpy as np
from utils import log_min_exp
import torch.nn.functional as F

from .source import Source


class DiscreteLogistic(Source):
    def __init__(self, nvars, decimal, rounding, K=1.0, mean=None, logscale=None, train=True, name="disceteLogistic"):
        super(DiscreteLogistic, self).__init__(nvars, K, name)

        if mean is None:
            mean = torch.zeros(1)
        if logscale is None:
            logscale = torch.zeros(1)

        # if want each dimension is trainable, just pass high dimensional mean and logscale
        self.mean = nn.Parameter(mean, requires_grad=train)
        self.logscale = nn.Parameter(logscale, requires_grad=train)

        self.decimal = decimal
        self.rounding = rounding

    def sample(self, batchSize, K=None):
        y = torch.randn(batchSize + self.nvars).to(self.mean)
        x = torch.exp(self.logscale) * torch.log(y / (1 - y)) + self.mean
        return self.rounding(self.decimal.forward(x))

    def _energy(self, z):
        uplus = (self.decimal.inverse(z + 0.5) - self.mean) / torch.exp(self.logscale)
        uminus = (self.decimal.inverse(z - 0.5) - self.mean) / torch.exp(self.logscale)
        return log_min_exp(F.logsigmoid(uplus), F.logsigmoid(uminus))


class MixtureDiscreteLogistic(Source):
    def __init__(self, nvars, nMixing, decimal, rounding, K=1.0, mean=None, logscale=None, train=True, name="mixtureDiscreteLogistic"):
        super(MixtureDiscreteLogistic, self).__init__(nvars, K, name)
        self.nMixing = nMixing
        self.mixing = nn.Parameter(torch.ones(nMixing) / nMixing, requires_grad=False)

        if mean is None:
            mean = torch.zeros([1] * (len(nvars) + 1) + [nMixing])
        if logscale is None:
            logscale = torch.zeros([1] * (len(nvars) + 1) + [nMixing])

        self.mean = nn.Parameter(mean, requires_grad=train)
        self.logscale = nn.Parameter(logscale, requires_grad=train)

        self.decimal = decimal
        self.rounding = rounding

    def sample(self, batchSize, K=None):
        pass

    def _energy(self, z):
        z = z.view(*z.shape, 1)
        uplus = (self.decimal.inverse(z + 0.5) - self.mean) / torch.exp(self.logscale)
        uminus = (self.decimal.inverse(z - 0.5) - self.mean) / torch.exp(self.logscale)
        