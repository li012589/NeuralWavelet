import math
import torch
from utils import getIndeices, dispatch, collect
from .source import Source


class HierarchyPrior(Source):
    def __init__(self, channel, length, priorList, depth=None, repeat=1, K=1.0, name="hierarchyPiror"):
        super(HierarchyPrior, self).__init__([channel, length, length], K, name)
        kernelSize = 2
        shape = [length, length]
        if depth is None:
            depth = int(math.log(length, kernelSize))

        indexList = []
        for no in range(depth):
            if repeat % 2 == 0:
                indexList.append(getIndeices(shape, kernelSize, kernelSize, kernelSize * (kernelSize**no), kernelSize**no, 0))
            else:
                indexList.append(getIndeices(shape, kernelSize, kernelSize, kernelSize * (kernelSize**no), kernelSize**no, kernelSize**no))
        indexIList = [item[0] for item in indexList]
        indexJList = [item[1] for item in indexList]

        self.factorOutIList = [term[:, 1:] if no != len(indexIList) - 1 else term for no, term in enumerate(indexIList)]
        self.factorOutJList = [term[:, 1:] if no != len(indexJList) - 1 else term for no, term in enumerate(indexJList)]

        assert len(priorList) == len(self.factorOutIList)

        self.priorList = torch.nn.ModuleList(priorList)

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
