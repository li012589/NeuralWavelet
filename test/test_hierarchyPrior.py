import torch
import numpy as np
from numpy.testing import assert_allclose
import os
import sys
sys.path.append(os.getcwd())
import source
import flow
import utils


def test_hierarchyPrior():

    class UniTestPrior(source.Source):
        def __init__(self, nvars, element, name="UniTestPrior"):
            super(UniTestPrior, self).__init__(nvars, 1.0, name)
            self.element = torch.nn.Parameter(torch.tensor(element), requires_grad=False)

        def sample(self, batchSize):
            return torch.ones([batchSize] + self.nvars).to(self.element).float() * self.element

        def _energy(self, z):
            return (torch.tensor([2])**self.element * np.prod(z.shape[2:]))

    length = 8
    channel = 3
    decimal = flow.ScalingNshifting(256, -128)
    p1 = source.DiscreteLogistic([channel, 16, 3], decimal, rounding=utils.roundingWidentityGradient)
    p2 = source.DiscreteLogistic([channel, 4, 3], decimal, rounding=utils.roundingWidentityGradient)
    p3 = source.MixtureDiscreteLogistic([channel, 1, 4], 5, decimal, rounding=utils.roundingWidentityGradient)

    P = source.HierarchyPrior(channel, length, [p1, p2, p3], repeat=2)

    x = P.sample(100)
    logp = P.logProbability(x)

    p1 = UniTestPrior([channel, 16, 3], 1)
    p2 = UniTestPrior([channel, 4, 3], 2)
    p3 = UniTestPrior([channel, 1, 4], 3)

    P = source.HierarchyPrior(channel, length, [p1, p2, p3], repeat=2)

    x = P.sample(1)
    logp = P.logProbability(x)

    target = np.array([[3, 1, 2, 1, 3, 1, 2, 1], [1, 1, 1, 1, 1, 1, 1, 1], [2, 1, 2, 1, 2, 1, 2, 1], [1, 1, 1, 1, 1, 1, 1, 1], [3, 1, 2, 1, 3, 1, 2, 1], [1, 1, 1, 1, 1, 1, 1, 1], [2, 1, 2, 1, 2, 1, 2, 1], [1, 1, 1, 1, 1, 1, 1, 1]])
    assert_allclose(x[0, 0].detach().numpy(), target)
    assert logp == -(16 * 3 * 2**1 + 4 * 3 * 2**2 + 1 * 4 * 2**3)


def test_grad():
    length = 8
    channel = 3
    decimal = flow.ScalingNshifting(256, -128)
    p1 = source.DiscreteLogistic([channel, 16, 3], decimal, rounding=utils.roundingWidentityGradient)
    p2 = source.DiscreteLogistic([channel, 4, 3], decimal, rounding=utils.roundingWidentityGradient)
    p3 = source.MixtureDiscreteLogistic([channel, 1, 4], 5, decimal, rounding=utils.roundingWidentityGradient)

    P = source.HierarchyPrior(channel, length, [p1, p2, p3], repeat=2)

    x = P.sample(100)
    logp = P.logProbability(x)
    L = logp.mean()
    L.backward()

    assert p1.mean.grad.sum().detach().item() != 0
    assert p2.mean.grad.sum().detach().item() != 0
    assert p3.mean.grad.sum().detach().item() != 0

    assert p1.logscale.grad.sum().detach().item() != 0
    assert p2.logscale.grad.sum().detach().item() != 0
    assert p3.logscale.grad.sum().detach().item() != 0

    assert p3.mixing.grad.sum().detach().item() != 0


if __name__ == "__main__":
    test_grad()
