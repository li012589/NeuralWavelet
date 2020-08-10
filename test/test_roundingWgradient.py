import torch
import numpy as np
from numpy.testing import assert_array_almost_equal,assert_array_equal
import os
import sys
sys.path.append(os.getcwd())
from utils import roundingWidentityGradient


def test_roundingWidentityGradient():

    rounding = roundingWidentityGradient()
    test = (torch.arange(0, 10).to(torch.float)) / 10
    test.requires_grad_()
    L = test.sum()
    L.backward()
    assert_array_equal(test.grad.detach().numpy(), np.ones(test.shape))


if __name__ == "__main__":
    test_roundingWidentityGradient()