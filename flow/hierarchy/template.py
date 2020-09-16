import torch
import numpy as np

from ..flow import Flow
from utils import dispatch, collect


def form(tensor):
    shape = int(tensor.shape[-2] ** 0.5)
    return tensor.reshape(tensor.shape[0], tensor.shape[1], shape, shape, 2, 2).permute([0, 1, 2, 4, 3, 5]).reshape(tensor.shape[0], tensor.shape[1], shape * 2, shape * 2)


def reform(tensor):
    return tensor.reshape(tensor.shape[0], tensor.shape[1] // 3, 3, tensor.shape[2], tensor.shape[3]).permute([0, 1, 3, 4, 2]).contiguous().reshape(tensor.shape[0], tensor.shape[1] // 3, tensor.shape[2] * tensor.shape[3], 3)


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
        inverseLogjac = x.new_zeros(x.shape[0])
        for no in range(len(self.indexI)):
            x, x_ = dispatch(self.indexI[no], self.indexJ[no], x)
            x_, logProbability = self.layerList[no].inverse(x_.permute([0, 2, 1, 3]).reshape(-1, channelSize, *self.kernelShape))
            inverseLogjac += logProbability.reshape(batchSize, -1).sum(1)
            x_ = x_.reshape(x.shape[0], -1, channelSize, np.prod(self.kernelShape)).permute([0, 2, 1, 3])
            x = collect(self.indexI[no], self.indexJ[no], x, x_)
        return x, inverseLogjac

    def forward(self, z):
        batchSize = z.shape[0]
        channelSize = z.shape[1]
        forwardLogjac = z.new_zeros(z.shape[0])
        for no in reversed(range(len(self.indexI))):
            z, z_ = dispatch(self.indexI[no], self.indexJ[no], z)
            z_, logProbability = self.layerList[no].forward(z_.permute([0, 2, 1, 3]).reshape(-1, channelSize, *self.kernelShape))
            forwardLogjac += logProbability.reshape(batchSize, -1).sum(1)
            z_ = z_.reshape(z.shape[0], -1, channelSize, np.prod(self.kernelShape)).permute([0, 2, 1, 3])
            z = collect(self.indexI[no], self.indexJ[no], z, z_)
        return z, forwardLogjac


class ParameterizedHierarchyBijector(HierarchyBijector):
    def __init__(self, kernelShape, indexI, indexJ, layerList, meanNNlist, scaleNNlist, decimal, prior=None, name="ParameterizedHierarchyBijector"):
        super(ParameterizedHierarchyBijector, self).__init__(kernelShape, indexI, indexJ, layerList, prior, name)
        self.meanNNlist = torch.nn.ModuleList(meanNNlist)
        self.scaleNNlist = torch.nn.ModuleList(scaleNNlist)
        self.decimal = decimal

    def inverse(self, x):
        batchSize = x.shape[0]
        channelSize = x.shape[1]
        inverseLogjac = x.new_zeros(x.shape[0])
        self.meanList = []
        self.scaleList = []
        for no in range(len(self.indexI)):
            x, x_ = dispatch(self.indexI[no], self.indexJ[no], x)
            if no % (self.repeat + 1) == 0 and no != 0 and no != len(self.indexI) - 1:
                tmpx_ = form(x_)
                self.meanList.append(reform(self.meanNNlist[no // (self.repeat + 1) - 1](self.decimal.inverse_(tmpx_))))
                self.scaleList.append(reform(self.scaleNNlist[no // (self.repeat + 1) - 1](self.decimal.inverse_(tmpx_))))
            x_, logProbability = self.layerList[no].inverse(x_.permute([0, 2, 1, 3]).reshape(-1, channelSize, *self.kernelShape))
            inverseLogjac += logProbability.reshape(batchSize, -1).sum(1)
            x_ = x_.reshape(x.shape[0], -1, channelSize, np.prod(self.kernelShape)).permute([0, 2, 1, 3])
            x = collect(self.indexI[no], self.indexJ[no], x, x_)
        return x, inverseLogjac

    def forward(self, z):
        batchSize = z.shape[0]
        channelSize = z.shape[1]
        forwardLogjac = z.new_zeros(z.shape[0])
        for no in reversed(range(len(self.indexI))):
            z, z_ = dispatch(self.indexI[no], self.indexJ[no], z)
            z_, logProbability = self.layerList[no].forward(z_.permute([0, 2, 1, 3]).reshape(-1, channelSize, *self.kernelShape))
            forwardLogjac += logProbability.reshape(batchSize, -1).sum(1)
            z_ = z_.reshape(z.shape[0], -1, channelSize, np.prod(self.kernelShape)).permute([0, 2, 1, 3])
            z = collect(self.indexI[no], self.indexJ[no], z, z_)
        return z, forwardLogjac

    def logProbability(self, x, K=None):
        z, logp = self.inverse(x)
        if self.prior is not None:
            return self.prior.logProbability(z, K, self.meanList, self.scaleList) + logp
        return logp


class OneToTwoHierarchyBijector(HierarchyBijector):
    def __init__(self, kernelShape, indexI, indexJ, layerList, prior=None, name="OneToTwoHierarchyBijector"):
        super(OneToTwoHierarchyBijector, self).__init__(kernelShape, indexI, indexJ, layerList, prior, name)

    def inverse(self, x):
        pass

    def forward(self, z):
        pass
