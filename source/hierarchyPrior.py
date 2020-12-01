import math
import torch
from utils import getIndeices, dispatch, collect
from .source import Source
from .discreteLogistic import DiscreteLogistic, MixtureDiscreteLogistic
import utils


class HierarchyPrior(Source):
    def __init__(self, channel, length, priorList, depth=None, repeat=1, K=1.0, name="hierarchyPiror"):
        super(HierarchyPrior, self).__init__([channel, length, length], K, name)
        kernelSize = 2
        shape = [length, length]
        if depth is None:
            depth = int(math.log(length, kernelSize))

        indexList = []
        for no in range(depth):
            indexList.append(getIndeices(shape, kernelSize, kernelSize, kernelSize * (kernelSize**no), kernelSize**no, 0))
        indexIList = [item[0] for item in indexList]
        indexJList = [item[1] for item in indexList]

        self.factorOutIList = [term[:, 1:] if no != len(indexIList) - 1 else term for no, term in enumerate(indexIList)]
        self.factorOutJList = [term[:, 1:] if no != len(indexJList) - 1 else term for no, term in enumerate(indexJList)]

        self.indexIList = indexIList
        self.indexJList = indexJList

        assert len(priorList) == len(self.factorOutIList)

        self.priorList = torch.nn.ModuleList(priorList)

        # skip check to allow smaller prior
        '''
        for no in range(len(self.priorList)):
            assert self.priorList[no].nvars == [channel] + list(self.factorOutIList[no].shape)
        '''

    def sample(self, batchSize, K=None):
        x = torch.zeros([batchSize] + self.nvars)
        for no in range(len(self.priorList)):
            x_ = self.priorList[no].sample(batchSize)
            x = collect(self.factorOutIList[no], self.factorOutJList[no], x, x_)
        return x

    def _energy(self, z):
        logp = z.new_zeros(z.shape[0])
        for no in range(len(self.priorList)):
            _, z_ = dispatch(self.factorOutIList[no], self.factorOutJList[no], z)
            logp = logp + self.priorList[no]._energy(z_)
        return logp


class ParameterizedHierarchyPrior(Source):
    def __init__(self, channel, length, prior, depth=None, repeat=1, decimal=None, rounding=None, K=1.0, name="parameterizedHierarchyPrior"):
        super(ParameterizedHierarchyPrior, self).__init__([channel, length, length], K, name)
        kernelSize = 2
        shape = [length, length]
        if depth is None:
            depth = int(math.log(length, kernelSize))

        indexList = []
        for no in range(depth):
            indexList.append(getIndeices(shape, kernelSize, kernelSize, kernelSize * (kernelSize**no), kernelSize**no, 0))
        indexIList = [item[0] for item in indexList]
        indexJList = [item[1] for item in indexList]

        self.factorOutIList = [term[:, 1:] if no != len(indexIList) - 1 else term for no, term in enumerate(indexIList)]
        self.factorOutJList = [term[:, 1:] if no != len(indexJList) - 1 else term for no, term in enumerate(indexJList)]
        self.decimal = decimal
        self.rounding = rounding
        self.lastPrior = prior

    def sample(self, batchSize, K=None):
        raise Exception("Not implemented")

    def logProbability(self, z, K=None, meanList=None, scaleList=None):
        if meanList is None or scaleList is None:
            raise Exception("no mean or scale passed")
        logp = z.new_zeros(z.shape[0])
        for no in range(len(self.factorOutIList)):
            _, z_ = dispatch(self.factorOutIList[no], self.factorOutJList[no], z)
            if no == len(self.factorOutIList) - 1:
                _logp = self.lastPrior._energy(z_)
            else:
                mean = meanList[no]
                scale = scaleList[no]
                assert mean.shape == scale.shape
                assert mean.shape == z_.shape
                _logp = -utils.logDiscreteLogistic(z_, mean, scale, self.decimal).reshape(z_.shape[0], -1).sum(-1)
            logp = logp + _logp

        return -logp


def im2grp(t):
    return t.reshape(t.shape[0], t.shape[1], t.shape[2] // 2, 2, t.shape[3] // 2, 2).permute([0, 1, 2, 4, 3, 5]).reshape(t.shape[0], t.shape[1], -1, 4)


class PassiveHierarchyPrior(Source):
    def __init__(self, length, prior, decimal=None, rounding=None, K=1.0, name="SimpleHierarchyPiror"):
        super(PassiveHierarchyPrior, self).__init__([3, length, length], K, name)
        self.depth = int(math.log(length, 2))

        self.decimal = decimal
        self.rounding = rounding
        self.lastPrior = prior

    def sample(self, batchSize, K=None):
        raise Exception("Not implemented")

    def logProbability(self, z, K=None, meanList=None, scaleList=None):
        if meanList is None or scaleList is None:
            raise Exception("no mean or scale passed")
        logp = z.new_zeros(z.shape[0])
        ul = z
        for no in range(self.depth):
            if no == self.depth - 1:
                ul = ul.reshape(*ul.shape[:2], 1, 4)
                _logp = self.lastPrior._energy(ul)
            else:
                _x = im2grp(ul)
                z_ = _x[:, :, :, 1:].contiguous()
                ul = _x[:, :, :, 0].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
                mean = meanList[no]
                scale = scaleList[no]
                assert mean.shape == scale.shape
                assert mean.shape == z_.shape
                _logp = -utils.logDiscreteLogistic(z_, mean, scale, self.decimal).reshape(z_.shape[0], -1).sum(-1)
            logp = logp + _logp
        return -logp


class SimpleHierarchyPrior(Source):
    def __init__(self, length, nMixing, decimal=None, rounding=None, clamp=None, sameDetail=True, K=1.0, name="SimpleHierarchyPiror"):
        super(SimpleHierarchyPrior, self).__init__([3, length, length], K, name)
        self.depth = int(math.log(length, 2))

        priorList = []

        if sameDetail:
            detailPrior = DiscreteLogistic([3, 1, 3], decimal, rounding)
            for no in range(self.depth):
                if no == self.depth - 1:
                    priorList.append(MixtureDiscreteLogistic([3, 1, 4], nMixing, decimal, rounding, clamp=clamp))
                else:
                    priorList.append(detailPrior)
        else:
            _length = int(length * length / 4)
            for n in range(self.depth):
                if n != self.depth - 1:
                    priorList.append(DiscreteLogistic([3, _length, 3], decimal, rounding))
                else:
                    priorList.append(MixtureDiscreteLogistic([3, _length, 4], nMixing, decimal, rounding, clamp=clamp))
                _length = int(_length / 4)

        self.priorList = torch.nn.ModuleList(priorList)

        assert len(priorList) == self.depth

    def sample(self, batchSize, K=None):
        raise Exception("Not implemented")

    def logProbability(self, z, K=None):
        logp = z.new_zeros(z.shape[0])
        ul = z
        for no in range(self.depth):
            if no == self.depth - 1:
                ul = ul.reshape(*ul.shape[:2], 1, 4)
                _logp = self.priorList[no]._energy(ul)
            else:
                _x = im2grp(ul)
                z_ = _x[:, :, :, 1:].contiguous()
                ul = _x[:, :, :, 0].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
                _logp = self.priorList[no]._energy(z_)
            logp = logp + _logp
        return -logp


