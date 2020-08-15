import torch
import numpy as np

from ..flow import Flow
from .im2col import dispatch, collect


class HierarchyBijector(Flow):
    def __init__(self, kernelShape, indexI, indexJ, layerList, prior=None, name="HierarchyBijector"):
        super(HierarchyBijector, self).__init__(prior, name)
        assert len(layerList) == len(indexI)
        assert len(layerList) == len(indexJ)

        self.depth = len(layerList)

        self.kernelShape = kernelShape
        self.layerList = torch.nn.ModuleList(layerList)
        self.indexI = indexI
        self.indexJ = indexJ

    def inverse(self, x):
        batchSize = x.shape[0]
        channelSize = x.shape[1]
        forwardLogjac = x.new_zeros(x.shape[0])
        for no in range(len(self.indexI)):
            x, x_ = dispatch(self.indexI[no], self.indexJ[no], x)
            x_, logProbability = self.layerList[no].forward(x_.permute([0, 2, 1, 3]).reshape(-1, channelSize, *self.kernelShape))
            forwardLogjac += logProbability.reshape(batchSize, -1).sum(1)
            x_ = x_.reshape(x.shape[0], -1, channelSize, np.prod(self.kernelShape)).permute([0, 2, 1, 3])
            x = collect(self.indexI[no], self.indexJ[no], x, x_)
        return x, forwardLogjac

    def forward(self, z):
        batchSize = z.shape[0]
        channelSize = z.shape[1]
        inverseLogjac = z.new_zeros(z.shape[0])
        for no in reversed(range(len(self.indexI))):
            z, z_ = dispatch(self.indexI[no], self.indexJ[no], z)
            z_, logProbability = self.layerList[no].inverse(z_.permute([0, 2, 1, 3]).reshape(-1, channelSize, *self.kernelShape))
            inverseLogjac += logProbability.reshape(batchSize, -1).sum(1)
            z_ = z_.reshape(z.shape[0], -1, channelSize, np.prod(self.kernelShape)).permute([0, 2, 1, 3])
            z = collect(self.indexI[no], self.indexJ[no], z, z_)
        return z, inverseLogjac


