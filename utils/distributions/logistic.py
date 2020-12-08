import torch
import torch.nn.functional as F
import numpy as np
from utils import logMinExp
from .etc import broadcastSize, selectArgs


def logLogistic(x, mean, logscale, testBroadcastSize=False):
    mean, logscale = broadcastSize(x.shape, [mean, logscale], testBroadcastSize)
    u = (x - mean) / torch.exp(logscale)
    return F.logsigmoid(u) + F.logsigmoid(-u) - logscale


def sampleLogistic(size, mean, logscale, testBroadcastSize=False, eps=1e-19):
    mean, logscale = broadcastSize(size, [mean, logscale], testBroadcastSize)
    y = torch.rand(size).to(mean)
    return torch.exp(logscale) * torch.log(y / (1 - y) + eps) + mean


def logDiscreteLogistic(x, mean, logscale, decimal=None, test=True, testBroadcastSize=False):
    if test:
        assert np.all(np.isfinite(x.cpu().detach().numpy()))
        assert np.all(np.equal(np.mod(x.cpu().detach().numpy(), 1), 0))
    mean, logscale = broadcastSize(x.shape, [mean, logscale], testBroadcastSize)
    if decimal is None:
        uplus = ((x + 0.5) - mean) / torch.exp(logscale)
        uminus = ((x - 0.5) - mean) / torch.exp(logscale)
    else:
        uplus = (decimal.inverse_(x + 0.5) - mean) / torch.exp(logscale)
        uminus = (decimal.inverse_(x - 0.5) - mean) / torch.exp(logscale)
    return logMinExp(F.logsigmoid(uplus), F.logsigmoid(uminus))


def sampleDiscreteLogistic(size, mean, logscale, rounding=torch.round, decimal=None, testBroadcastSize=False, eps=1e-19):
    mean, logscale = broadcastSize(size, [mean, logscale], testBroadcastSize)
    x = sampleLogistic(size, mean, logscale, testBroadcastSize, eps)
    if decimal is None:
        return rounding(x)
    else:
        return rounding(decimal.forward_(x))


def cdfDiscreteLogitstic(x, mean, logscale, decimal=None, test=True, testBroadcastSize=False):
    if test:
        assert np.all(np.isfinite(x.cpu().detach().numpy()))
        assert np.all(np.equal(np.mod(x.cpu().detach().numpy(), 1), 0))
    mean, logscale = broadcastSize(x.shape, [mean, logscale], testBroadcastSize)
    if decimal is None:
        return torch.sigmoid((x + 0.5 - mean) / torch.exp(logscale))
    else:
        return torch.sigmoid((decimal.inverse_(x + 0.5) - mean) / torch.exp(logscale))


def logMixDiscreteLogistic(x, mean, logscale, parts, decimal=None, test=True, eps=1e-19):
    '''
    x, mean, logscale are of broadcastable size,
    parts is torch.tensor with size [..., parts dim], ... are broadcastable with x
    '''
    assert mean.shape[0] == parts.shape[-1]
    assert logscale.shape[0] == parts.shape[-1]
    parts = torch.softmax(parts, dim=-1)

    #assert parts.sum() == 1 * np.prod(parts.shape[:-1])

    if test:
        assert np.all(np.isfinite(x.cpu().detach().numpy()))
        assert np.all(np.equal(np.mod(x.cpu().detach().numpy(), 1), 0))
    mean, logscale = mean.permute(torch.arange(len(mean.shape)).roll(-1).tolist()), logscale.permute(torch.arange(len(mean.shape)).roll(-1).tolist())
    x = x.view(*x.shape, 1)
    mean, logscale, parts = broadcastSize(x.shape, [mean, logscale, parts], test=False)
    if decimal is None:
        probs = torch.sigmoid((x + 0.5 - mean) / torch.exp(logscale)) - torch.sigmoid((x - 0.5 - mean) / torch.exp(logscale))
    else:
        probs = torch.sigmoid((decimal.inverse_(x + 0.5) - mean) / torch.exp(logscale)) - torch.sigmoid((decimal.inverse_(x - 0.5) - mean) / torch.exp(logscale))

    probs = torch.sum(probs * parts, dim=-1)
    return torch.log(probs + eps)


def sampleMixDiscreteLogistic(size, mean, logscale, parts, rounding=torch.round, decimal=None, test=True, testBroadcastSize=False, eps=1e-8):
    assert mean.shape[0] == parts.shape[-1]
    assert logscale.shape[0] == parts.shape[-1]
    parts = torch.softmax(parts, dim=-1)

    mean, logscale = selectArgs(size, parts, [mean, logscale])

    return sampleDiscreteLogistic(size, mean, logscale, rounding, decimal, testBroadcastSize, eps)


def cdfMixDiscreteLogistic(x, mean, logscale, parts, decimal=None, test=True):
    assert mean.shape[0] == parts.shape[-1]
    assert logscale.shape[0] == parts.shape[-1]
    parts = torch.softmax(parts, dim=-1)

    if test:
        assert np.all(np.isfinite(x.cpu().detach().numpy()))
        assert np.all(np.equal(np.mod(x.cpu().detach().numpy(), 1), 0))
    mean, logscale = mean.permute(torch.arange(len(mean.shape)).roll(-1).tolist()), logscale.permute(torch.arange(len(mean.shape)).roll(-1).tolist())
    x = x.view(*x.shape, 1)
    mean, logscale, parts = broadcastSize(x.shape, [mean, logscale, parts], test=False)

    if decimal is None:
        cdfs = torch.sigmoid((x + 0.5 - mean) / torch.exp(logscale))
    else:
        cdfs = torch.sigmoid((decimal.inverse_(x + 0.5) - mean) / torch.exp(logscale))

    cdf = torch.sum(cdfs * parts, dim=-1)
    return cdf
