import torch
import numpy as np


def broadcastSize(shape, args, test=True):
    args_ = []
    for term in args:
        if test:
            for i, d in enumerate(reversed(term.shape)):
                assert d == shape[-i - 1]
        termshape = [1] * (len(shape) - len(term.shape)) + list(term.shape)
        args_.append(term.view(termshape))
    return args_


def selectArgs(size, parts, args):
    batchDim = size[0]
    varDim = parts.shape[:-1]
    weights = parts.view(-1, parts.shape[-1])
    idx = torch.multinomial(weights, num_samples=batchDim, replacement=True).reshape(batchDim, -1)

    args_ = []
    for term in args:
        termshape = [size[0]] + [1] * (len(size[1:]) - len(varDim)) + list(varDim)
        term = term.view(parts.shape[-1], int(np.prod(varDim)))[idx, torch.arange(int(np.prod(varDim)))].view(termshape)
        args_.append(term)

    return args_

