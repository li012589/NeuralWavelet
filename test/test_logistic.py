import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from flow import ScalingNshifting
from utils import logLogistic, sampleLogistic, logDiscreteLogistic, sampleDiscreteLogistic, logMixDiscreteLogistic, sampleMixDiscreteLogistic, mixtureSample, mixtureLogProbability, cdfDiscreteLogitstic, cdfMixDiscreteLogistic


def test_logLogistic():
    # test compute probability of a image of size [100 ,3, 32, 32] from a single distribution
    a = logLogistic(torch.randn(100, 3, 32, 32), torch.randn(1), torch.randn(32)).detach().numpy()
    # test compute probability of a image of size [100 ,3, 32, 32] from a every distribution at per piexl
    b = logLogistic(torch.randn(100, 3, 32, 32), torch.randn([32, 32]), torch.randn([32, 32])).detach().numpy()
    # test compute probability of a image of size [100 ,3, 32, 32] from a every distribution at per piexl per channel
    c = logLogistic(torch.randn(100, 3, 32, 32), torch.randn([3, 32, 32]), torch.randn([3, 32, 32])).detach().numpy()
    # test compute probability of a image of size [100 ,3, 32, 32] from a every distribution at per piexl per channel per batch
    d = logLogistic(torch.randn(100, 3, 32, 32), torch.randn([100, 3, 32, 32]), torch.randn([100, 3, 32, 32])).detach().numpy()

    # same size and no nan
    assert not np.isnan((a + b + c + d).sum())

    field = torch.ones(100, 3, 32, 32)
    # test change of mean and logscale do makes a different
    a = logLogistic(field, torch.tensor(torch.arange(32).float().reshape(1, 32) / 10), torch.tensor(torch.arange(32).float().reshape(32, 1) / 10)).detach().numpy()

    # no element is equal, random check
    for _ in range(10):
        (a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, int(torch.randint(1, 32, [1]))] == a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, 0]).sum() == 0
        (a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, int(torch.randint(1, 32, [1]))] == a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), 0, :]).sum() == 0
    # batch channel is same, random check
    for _ in range(10):
        (a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, :] == a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, :]).sum() == 32 * 32


def test_sampleLogistic():
    # test mean
    mean = torch.tensor(torch.arange(32).float().reshape(1, 32))
    logscale = torch.tensor(torch.arange(32).float().reshape(32, 1) / 10)
    f = sampleLogistic([1000, 3, 32, 32], mean, logscale).detach().numpy()

    # test sampling no nan
    assert not np.isnan(f.sum())

    fmean = f.reshape(-1, 32).mean(0)

    # test mean
    (np.round(fmean) == np.arange(32)).sum() == 32

    #test var
    mean = torch.tensor(torch.zeros(1).float().reshape(1))
    logscale = torch.tensor(torch.arange(32).float() / 10)
    f = sampleLogistic([1000, 3, 32, 32], mean, logscale).detach().numpy()

    # test sampling no nan
    assert not np.isnan(f.sum())

    fvar = f.reshape(-1, 32)
    var = fvar.var(0)
    # the expect var of logistic is $\fra{1}{3} \pi^2 e^{2 logscale}$
    Evar = np.exp(2 * logscale.numpy()) * (np.pi ** 2) / 3

    assert_allclose(var, Evar, atol=.1, rtol=0.025)


def test_logDiscreteLogistic():
    shape = [100, 3, 32, 32]

    # test compute probability of a image of size [100 ,3, 32, 32] from a single distribution
    a = logDiscreteLogistic(torch.randint(255, shape), torch.randint(255, [1]) + torch.randn(1), torch.randn([32])).detach().numpy()
    assert (np.exp(a) > 1).sum() == 0
    # test compute probability of a image of size [100 ,3, 32, 32] from a every distribution at per piexl
    b = logDiscreteLogistic(torch.randint(255, shape), torch.randint(255, [32, 32]) + torch.randn([32, 32]), torch.randn([32, 32])).detach().numpy()
    assert (np.exp(b) > 1).sum() == 0
    # test compute probability of a image of size [100 ,3, 32, 32] from a every distribution at per piexl per channel
    c = logDiscreteLogistic(torch.randint(255, shape), torch.randint(255, [3, 32, 32]) + torch.randn([3, 32, 32]), torch.randn([3, 32, 32])).detach().numpy()
    assert (np.exp(c) > 1).sum() == 0
    # test compute probability of a image of size [100 ,3, 32, 32] from a every distribution at per piexl per channel per batch
    d = logDiscreteLogistic(torch.randint(255, shape), torch.randint(255, [100, 3, 32, 32]) + torch.randn([100, 3, 32, 32]), torch.randn([100, 3, 32, 32])).detach().numpy()
    assert (np.exp(d) > 1).sum() == 0

    # same size and no nan
    assert not np.isnan((a + b + c + d).sum())

    field = torch.ones(100, 3, 32, 32).int()
    # test change of mean and logscale do makes a different
    a = logDiscreteLogistic(field, torch.tensor(torch.arange(32).float().reshape(1, 32)), torch.tensor(torch.arange(32).float().reshape(32, 1) / 10)).detach().numpy()

    # no element is equal, random check
    for _ in range(10):
        (a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, int(torch.randint(1, 32, [1]))] == a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, 0]).sum() == 0
        (a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, int(torch.randint(1, 32, [1]))] == a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), 0, :]).sum() == 0
    # batch channel is same, random check
    for _ in range(10):
        (a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, :] == a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, :]).sum() == 32 * 32


    # test decimal
    decimal = ScalingNshifting(scaling=255.0, shifting=-128.0)

    # test compute probability of a image of size [100 ,3, 32, 32] from a single distribution
    a = logDiscreteLogistic(torch.randint(255, shape), torch.randn(1), torch.randn([32]), decimal=decimal).detach().numpy()
    assert (np.exp(a) > 1).sum() == 0
    # test compute probability of a image of size [100 ,3, 32, 32] from a every distribution at per piexl
    b = logDiscreteLogistic(torch.randint(255, shape), torch.randn([32, 32]), torch.randn([32, 32]), decimal=decimal).detach().numpy()
    assert (np.exp(b) > 1).sum() == 0
    # test compute probability of a image of size [100 ,3, 32, 32] from a every distribution at per piexl per channel
    c = logDiscreteLogistic(torch.randint(255, shape), torch.randn([3, 32, 32]), torch.randn([3, 32, 32]), decimal=decimal).detach().numpy()
    assert (np.exp(c) > 1).sum() == 0
    # test compute probability of a image of size [100 ,3, 32, 32] from a every distribution at per piexl per channel per batch
    d = logDiscreteLogistic(torch.randint(255, shape), torch.randn([100, 3, 32, 32]), torch.randn([100, 3, 32, 32]), decimal=decimal).detach().numpy()
    assert (np.exp(d) > 1).sum() == 0

    # same size and no nan
    assert not np.isnan((a + b + c + d).sum())

    field = torch.ones(100, 3, 32, 32).int()
    # test change of mean and logscale do makes a different
    a = logDiscreteLogistic(field, torch.tensor(torch.arange(32).float().reshape(1, 32)), torch.tensor(torch.arange(32).float().reshape(32, 1) / 10), decimal=decimal).detach().numpy()

    # no element is equal, random check
    for _ in range(10):
        (a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, int(torch.randint(1, 32, [1]))] == a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, 0]).sum() == 0
        (a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, int(torch.randint(1, 32, [1]))] == a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), 0, :]).sum() == 0
    # batch channel is same, random check
    for _ in range(10):
        (a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, :] == a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, :]).sum() == 32 * 32


def test_sampleDiscreteLogistic():
    # test mean
    mean = torch.tensor(torch.arange(32).float().reshape(1, 32))
    logscale = torch.tensor(torch.arange(32).float().reshape(32, 1) / 10)
    f = sampleDiscreteLogistic([1000, 3, 32, 32], mean, logscale).detach().numpy()

    # test sampling no nan and integer
    assert not np.isnan(f.sum())
    assert (f.astype(int) - f).sum() == 0

    fmean = f.reshape(-1, 32).mean(0)

    # test mean
    (np.round(fmean) == np.arange(32)).sum() == 32

    #test var
    mean = torch.tensor(torch.zeros(1).float().reshape(1))
    logscale = torch.tensor(torch.arange(32).float() / 10)
    f = sampleLogistic([1000, 3, 32, 32], mean, logscale).detach().numpy()

    # test sampling no nan
    assert not np.isnan(f.sum())

    fvar = f.reshape(-1, 32)
    var = fvar.var(0)
    # the expect var of logistic is $\fra{1}{3} \pi^2 e^{2 logscale}$
    Evar = np.exp(2 * logscale.numpy()) * (np.pi ** 2) / 3

    assert_allclose(var, Evar, atol=.1, rtol=0.025)

    # test decimal
    decimal = ScalingNshifting(scaling=255.0, shifting=-128.0)
    mean = torch.tensor(torch.arange(32).float().reshape(1, 32))
    logscale = torch.tensor(torch.arange(32).float().reshape(32, 1) / 10)
    f = sampleDiscreteLogistic([1000, 3, 32, 32], mean, logscale, decimal=decimal)

    # test sampling no nan and integer
    assert not np.isnan(f.sum())
    assert (f.int() - f).sum() == 0

    fmean = f.reshape(-1, 32).mean(0)

    # test mean
    (torch.round(decimal.inverse_(fmean)) == torch.arange(32)).float().sum() == 32

    #test var
    mean = torch.tensor(torch.zeros(1).float().reshape(1))
    logscale = torch.tensor(torch.arange(32).float() / 10)
    f = sampleDiscreteLogistic([1000, 3, 32, 32], mean, logscale, decimal=decimal)

    # test sampling no nan and integer
    assert not np.isnan(f.sum())
    assert (f.int() - f).sum() == 0

    fvar = decimal.inverse_(f.reshape(-1, 32))
    var = fvar.var(0)
    # the expect var of logistic is $\fra{1}{3} \pi^2 e^{2 logscale}$
    Evar = np.exp(2 * logscale.numpy()) * (np.pi ** 2) / 3

    assert_allclose(var, Evar, atol=.1, rtol=0.025)


def test_logMixDiscreteLogistic():
    shape = [100, 3, 32, 32]
    # test compute probability of a image of size [100 ,3, 32, 32] from a single distribution
    # here 5 is the n mixing
    a = logMixDiscreteLogistic(torch.randint(255, shape), torch.randint(255, [5, 1]) + torch.randn([5, 1]), torch.randn([5, 1]), torch.softmax(torch.randn(1, 5), -1)).detach().numpy()
    assert (np.exp(a) > 1).sum() == 0
    # test compute probability of a image of size [100 ,3, 32, 32] from a every distribution at per piexl
    b = logMixDiscreteLogistic(torch.randint(255, shape), torch.randint(255, [5, 32, 32]) + torch.randn([5, 32, 32]), torch.randn([5, 32, 32]), torch.softmax(torch.randn(32, 32, 5), -1)).detach().numpy()
    assert (np.exp(b) > 1).sum() == 0
    # test compute probability of a image of size [100 ,3, 32, 32] from a every distribution at per piexl per channel
    c = logMixDiscreteLogistic(torch.randint(255, shape), torch.randint(255, [5, 3, 32, 32]) + torch.randn([5, 3, 32, 32]), torch.randn([5, 3, 32, 32]), torch.softmax(torch.randn(3, 32, 32, 5), -1)).detach().numpy()
    assert (np.exp(c) > 1).sum() == 0
    # test compute probability of a image of size [100 ,3, 32, 32] from a every distribution at per piexl per channel per batch
    d = logMixDiscreteLogistic(torch.randint(255, shape), torch.randint(255, [5, 100, 3, 32, 32]) + torch.randn([5, 100, 3, 32, 32]), torch.randn([5, 100, 3, 32, 32]), torch.softmax(torch.randn(100, 3, 32, 32, 5), -1)).detach().numpy()
    assert (np.exp(d) > 1).sum() == 0

    # same size and no nan
    assert not np.isnan((a + b + c + d).sum())

    field = torch.ones(100, 3, 32, 32).int()
    parts = torch.softmax(torch.arange(1, 6).reshape(1, 5).float(), -1)
    monparts = torch.softmax(torch.tensor([1, 0, 0, 0, 0]).reshape(1, 5).float(), -1)
    # test change of mean and logscale do makes a different
    a = logMixDiscreteLogistic(field, torch.tensor(torch.arange(32).float().reshape(1, 32).repeat(5, 1, 1)), torch.tensor(torch.arange(32).float().reshape(32, 1).repeat(5, 1, 1) / 10), parts, eps=1e-19).detach().numpy()
    ap = logDiscreteLogistic(field, torch.tensor(torch.arange(32).float().reshape(1, 32)), torch.tensor(torch.arange(32).float().reshape(32, 1) / 10)).detach().numpy()
    app = logMixDiscreteLogistic(field, torch.tensor(torch.arange(32).float().reshape(1, 32).repeat(5, 1, 1)), torch.tensor(torch.arange(32).float().reshape(32, 1).repeat(5, 1, 1) / 10), monparts, eps=1e-19).detach().numpy()

    # test fallback to single mode
    assert_allclose(a, ap, rtol=1e-3)
    assert_allclose(a, app, rtol=1e-4)

    # no element is equal, random check
    for _ in range(10):
        (a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, int(torch.randint(1, 32, [1]))] == a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, 0]).sum() == 0
        (a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, int(torch.randint(1, 32, [1]))] == a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), 0, :]).sum() == 0
    # batch channel is same, random check
    for _ in range(10):
        (a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, :] == a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, :]).sum() == 32 * 32

    # test decimal
    decimal = ScalingNshifting(scaling=255.0, shifting=-128.0)

    # test compute probability of a image of size [100 ,3, 32, 32] from a single distribution
    a = logMixDiscreteLogistic(torch.randint(255, shape), torch.randint(255, [5, 1]) + torch.randn([5, 1]), torch.randn([5, 1]), torch.softmax(torch.randn(1, 5), -1), decimal=decimal).detach().numpy()
    assert (np.exp(a) > 1).sum() == 0
    # test compute probability of a image of size [100 ,3, 32, 32] from a every distribution at per piexl
    b = logMixDiscreteLogistic(torch.randint(255, shape), torch.randint(255, [5, 32, 32]) + torch.randn([5, 32, 32]), torch.randn([5, 32, 32]), torch.softmax(torch.randn(32, 32, 5), -1), decimal=decimal).detach().numpy()
    assert (np.exp(b) > 1).sum() == 0
    # test compute probability of a image of size [100 ,3, 32, 32] from a every distribution at per piexl per channel
    c = logMixDiscreteLogistic(torch.randint(255, shape), torch.randint(255, [5, 3, 32, 32]) + torch.randn([5, 3, 32, 32]), torch.randn([5, 3, 32, 32]), torch.softmax(torch.randn(3, 32, 32, 5), -1), decimal=decimal).detach().numpy()
    assert (np.exp(c) > 1).sum() == 0
    # test compute probability of a image of size [100 ,3, 32, 32] from a every distribution at per piexl per channel per batch
    d = logMixDiscreteLogistic(torch.randint(255, shape), torch.randint(255, [5, 100, 3, 32, 32]) + torch.randn([5, 100, 3, 32, 32]), torch.randn([5, 100, 3, 32, 32]), torch.softmax(torch.randn(100, 3, 32, 32, 5), -1), decimal=decimal).detach().numpy()
    assert (np.exp(d) > 1).sum() == 0

    # same size and no nan
    assert not np.isnan((a + b + c + d).sum())

    field = torch.ones(100, 3, 32, 32).int()
    parts = torch.softmax(torch.arange(1, 6).reshape(1, 5).float(), -1)
    # test change of mean and logscale do makes a different
    a = logMixDiscreteLogistic(field, torch.tensor(torch.arange(32).float().reshape(1, 32).repeat(5, 1, 1)), torch.tensor(torch.arange(32).float().reshape(32, 1).repeat(5, 1, 1) / 10), parts, decimal=decimal, eps=1e-19).detach().numpy()
    ap = logDiscreteLogistic(field, torch.tensor(torch.arange(32).float().reshape(1, 32)), torch.tensor(torch.arange(32).float().reshape(32, 1) / 10), decimal=decimal).detach().numpy()
    app = logMixDiscreteLogistic(field, torch.tensor(torch.arange(32).float().reshape(1, 32).repeat(5, 1, 1)), torch.tensor(torch.arange(32).float().reshape(32, 1).repeat(5, 1, 1) / 10), monparts, decimal=decimal, eps=1e-19).detach().numpy()

    # test fallback to single mode
    assert_allclose(a, ap, rtol=1e-3)
    assert_allclose(a, app, rtol=1e-4)

    # no element is equal, random check
    for _ in range(10):
        (a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, int(torch.randint(1, 32, [1]))] == a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, 0]).sum() == 0
        (a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, int(torch.randint(1, 32, [1]))] == a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), 0, :]).sum() == 0
    # batch channel is same, random check
    for _ in range(10):
        (a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, :] == a[int(torch.randint(100, [1])), int(torch.randint(3, [1])), :, :]).sum() == 32 * 32


def test_sampleMixDiscreteLogistic():

    # test mean
    mean = torch.tensor(torch.randint(5, [2, 32]).float())
    logscale = torch.tensor(torch.randint(3, [2, 32]).float() / 100)
    parts = torch.softmax(torch.randn(32, 2), -1)
    f = sampleMixDiscreteLogistic([1000, 3, 32, 32], mean, logscale, parts).detach().numpy()

    # test sampling no nan and integer
    assert not np.isnan(f.sum())
    assert (f.astype(int) - f).sum() == 0

    fmean = f.reshape(-1, 32).mean(0)
    Emean = torch.sum(mean.permute(torch.arange(len(mean.shape)).roll(-1).tolist()) * parts, dim=-1).numpy()
    assert max(abs(fmean - Emean)) < 2.0

    '''
    import pdb
    pdb.set_trace()
    # test mean
    (np.round(fmean) == np.arange(32)).sum() == 32

    #test var
    mean = torch.tensor(torch.zeros(1).float().reshape(1))
    logscale = torch.tensor(torch.arange(32).float() / 10)
    f = sampleLogistic([1000, 3, 32, 32], mean, logscale).detach().numpy()

    # test sampling no nan
    assert not np.isnan(f.sum())

    fvar = f.reshape(-1, 32)
    var = fvar.var(0)
    # the expect var of logistic is $\fra{1}{3} \pi^2 e^{2 logscale}$
    Evar = np.exp(2 * logscale.numpy()) * (np.pi ** 2) / 3

    assert_allclose(var, Evar, atol=.1, rtol=0.025)

    # test decimal
    decimal = ScalingNshifting(scaling=255.0, shifting=-128.0)
    mean = torch.tensor(torch.arange(32).float().reshape(1, 32))
    logscale = torch.tensor(torch.arange(32).float().reshape(32, 1) / 10)
    f = sampleDiscreteLogistic([1000, 3, 32, 32], mean, logscale, decimal=decimal)

    # test sampling no nan and integer
    assert not np.isnan(f.sum())
    assert (f.int() - f).sum() == 0

    fmean = f.reshape(-1, 32).mean(0)

    # test mean
    (torch.round(decimal.inverse(fmean)) == torch.arange(32)).float().sum() == 32

    #test var
    mean = torch.tensor(torch.zeros(1).float().reshape(1))
    logscale = torch.tensor(torch.arange(32).float() / 10)
    f = sampleDiscreteLogistic([1000, 3, 32, 32], mean, logscale, decimal=decimal)

    # test sampling no nan and integer
    assert not np.isnan(f.sum())
    assert (f.int() - f).sum() == 0

    fvar = decimal.inverse(f.reshape(-1, 32))
    var = fvar.var(0)
    # the expect var of logistic is $\fra{1}{3} \pi^2 e^{2 logscale}$
    Evar = np.exp(2 * logscale.numpy()) * (np.pi ** 2) / 3

    assert_allclose(var, Evar, atol=.1, rtol=0.025)
    '''


def test_cdf_Discretelogistic():
    decimal = ScalingNshifting(scaling=255.0, shifting=-128.0)
    nbins = 4096
    mean = decimal.inverse_(torch.tensor(torch.arange(32).float().reshape(2, 16)))
    logscale = decimal.inverse_(torch.tensor(torch.arange(32).float().reshape(2, 16) / 10))

    bins = torch.arange(-nbins // 2, nbins // 2).reshape(-1, 1, 1)
    bins = bins + decimal.forward_(mean.reshape(1, *mean.shape)).int() - 1
    CDF = cdfDiscreteLogitstic(bins, mean, logscale, decimal=decimal)

    assert CDF.shape == bins.shape
    assert CDF.max() <= 1.0
    assert CDF.min() >= 0.0


def test_cdf_mixtureDiscretelogistic():
    decimal = ScalingNshifting(scaling=255.0, shifting=-128.0)
    nbins = 4096
    mean = decimal.inverse_(torch.tensor(torch.randint(25, [5, 2, 32]).float()))
    logscale = torch.tensor(torch.randint(3, [5, 2, 32]).float() / 10)
    parts = torch.randn(2, 32, 5)

    bins = torch.arange(-nbins // 2, nbins // 2).reshape(-1, 1, 1)
    bins = bins + (decimal.forward_(mean.permute([1, 2, 0])) * parts).sum(-1).reshape(1, *mean.shape[1:]).int() - 1

    CDF = cdfMixDiscreteLogistic(bins, mean, logscale, parts, decimal=decimal)

    assert CDF.shape == bins.shape
    assert CDF.max() <= 1.0
    assert CDF.min() >= 0.0


if __name__ == "__main__":
    '''
    test_sampleDiscreteLogistic()
    test_logDiscreteLogistic()
    test_logLogistic()
    test_sampleLogistic()
    test_sampleDiscreteLogistic()
    test_logMixDiscreteLogistic()
    test_sampleMixDiscreteLogistic()
    test_cdf_Discretelogistic()
    '''
    test_cdf_mixtureDiscretelogistic()
