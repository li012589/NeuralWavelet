import torch
from torch import nn
import numpy as np
from utils import logDiscreteLogistic, sampleDiscreteLogistic, logMixDiscreteLogistic, sampleMixDiscreteLogistic
import torch.nn.functional as F

from .source import Source


class DiscreteLogistic(Source):
    def __init__(self, nvars, decimal, rounding, K=1.0, mean=None, logscale=None, train=True, name="disceteLogistic"):
        super(DiscreteLogistic, self).__init__(nvars, K, name)

        if mean is None:
            mean = torch.zeros(nvars)
        if logscale is None:
            logscale = torch.zeros(nvars)

        # if want each dimension is trainable, just pass high dimensional mean and logscale
        self.mean = nn.Parameter(mean, requires_grad=train)
        self.logscale = nn.Parameter(logscale, requires_grad=train)

        self.decimal = decimal
        self.rounding = rounding

    def sample(self, batchSize, K=None):
        return sampleDiscreteLogistic([batchSize] + self.nvars, self.mean, self.logscale, rounding=self.rounding, decimal=self.decimal)

    def _energy(self, z):
        return -logDiscreteLogistic(z, self.mean, self.logscale, decimal=self.decimal).reshape(z.shape[0], -1).sum(-1)


class MixtureDiscreteLogistic(Source):
    def __init__(self, nvars, nMixing, decimal, rounding, K=1.0, mean=None, logscale=None, train=True, name="mixtureDiscreteLogistic"):
        super(MixtureDiscreteLogistic, self).__init__(nvars, K, name)
        self.nMixing = nMixing
        self.mixing = nn.Parameter(torch.softmax(torch.zeros(nvars + [nMixing]), dim=-1), requires_grad=train)

        if mean is None:
            mean = torch.zeros([nMixing] + nvars)
            for i in range(nMixing):
                mean[i, ...] += (i - (nMixing - 1) / 2.) / 8.
        if logscale is None:
            logscale = torch.zeros([nMixing] + nvars)

        self.mean = nn.Parameter(mean, requires_grad=train)
        self.logscale = nn.Parameter(logscale, requires_grad=train)

        self.decimal = decimal
        self.rounding = rounding

    def sample(self, batchSize, K=None):
        return sampleMixDiscreteLogistic([batchSize] + self.nvars, self.mean, self.logscale, self.mixing, rounding=self.rounding, decimal=self.decimal)

    def _energy(self, z):
        return -logMixDiscreteLogistic(z, self.mean, self.logscale, self.mixing, decimal=self.decimal).reshape(z.shape[0], -1).sum(-1)


