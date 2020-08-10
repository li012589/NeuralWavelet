from flowRelated import *

import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
import numpy as np
import utils
import flow
import source
import utils


def test_ScalingNshifting():
    p = source.Gaussian([2, 2])
    s = float(np.random.randint(1, 11))
    t = float(np.random.randint(0, 11))
    f = flow.ScalingNshifting(s, t)
    f.prior = p

    bijective(f)

def test_ScalingNshifting_saveload():
    p = source.Gaussian([2, 2])

    s = float(np.random.randint(1, 11))
    t = float(np.random.randint(0, 11))
    f = flow.ScalingNshifting(s, t)
    f.prior = p

    pp = source.Gaussian([2, 2])
    s = float(np.random.randint(1, 11))
    t = float(np.random.randint(0, 11))
    blankf = flow.ScalingNshifting(s, t)
    blankf.prior = pp

    saveload(f, blankf)

if __name__ == "__main__":
    test_ScalingNshifting_saveload()