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

    f = flow.ScalingNshifting(0.5, 0.4)
    f.prior = p

    bijective(f)

def test_ScalingNshifting_saveload():
    p = source.Gaussian([2, 2])

    f = flow.ScalingNshifting(0.5, 0.4)
    f.prior = p

    pp = source.Gaussian([2, 2])
    blankf = flow.ScalingNshifting(15.1, 4.1)
    blankf.prior = pp

    saveload(f, blankf)

if __name__ == "__main__":
    test_ScalingNshifting_saveload()