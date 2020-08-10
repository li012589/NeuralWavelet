import torch
import numpy as np

from .source import Source


class DiscreteLogistic(Source):
    def __init__(self, nvars, K=1.0, name="disceteLogistic"):
        super(DiscreteLogistic, self).__init__(nvars, K, name)

    def sample(self, batchSize, K=None):
        pass

    def _energy(self, z):
        pass


class MixtureDiscreteLogistic(Source):
    def __init__(self, nvars, K=1.0, name="mixtureDiscreteLogistic"):
        super(MixtureDiscreteLogistic, self).__init__(nvars, K, name)

    def sample(self, batchSize, K=None):
        pass

    def _energy(self, z):
        pass