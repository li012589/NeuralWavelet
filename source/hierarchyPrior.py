import math
import torch
from utils import getIndeices, dispatch, collect
from .source import Source
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


class ParameterizedHierarchyPrior(HierarchyPrior):
    def __init__(self, channel, length, priorList, meanNNlsit, scaleNNlist, depth=None, repeat=1, K=1.0, name="parameterizedHierarchyPrior"):
        super(ParameterizedHierarchyPrior, self).__init__(channel, length, priorList, depth, repeat, K, name)
        self.meanNNlist = torch.nn.ModuleList(meanNNlsit)
        self.scaleNNlist = torch.nn.ModuleList(scaleNNlist)
        self.decimal = priorList[0].decimal
        self.batchNorms = torch.nn.ModuleList([torch.nn.BatchNorm2d(channel) for _ in range(len(self.priorList))])

        assert len(self.scaleNNlist) == len(self.priorList) - 1

    def sample(self, batchSize, K=None):
        raise Exception("Not implemented")

    def _energy(self, z):
        logp = z.new_zeros(z.shape[0])
        for no in reversed(range(len(self.priorList))):
            if no == len(self.priorList) - 1:
                _, z_ = dispatch(self.factorOutIList[no], self.factorOutJList[no], z)
                logp = logp + self.priorList[no]._energy(z_)
            else:
                _, z_pre = dispatch(self.indexIList[no + 1], self.indexJList[no + 1], z)
                _, z_ = dispatch(self.factorOutIList[no], self.factorOutJList[no], z)
                z_pre = self.batchNorms[no](z_pre)
                z_pre = z_pre.reshape(z_.shape[0], -1)
                mean = self.meanNNlist[no](z_pre).reshape(*z_.shape)
                logscale = self.scaleNNlist[no](z_pre).reshape(*z_.shape)

                _logp = -utils.logDiscreteLogistic(z_, mean, logscale, self.decimal).reshape(z_.shape[0], -1).sum(-1)
                logp = logp + _logp
        return logp

