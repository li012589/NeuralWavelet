import math, torch

from .template import HierarchyBijector, ParameterizedHierarchyBijector, OneToTwoHierarchyBijector
from utils import getIndeices
from flow import Flow
import source

class MERA(HierarchyBijector):
    def __init__(self, kernelDim, length, layerList, repeat=1, depth=None, prior=None, name="MERA"):
        kernelSize = 2
        shape = [length, length]
        if depth is None:
            depth = int(math.log(length, kernelSize))

        indexList = []

        for no in range(depth):
            indexList.append(getIndeices(shape, kernelSize, kernelSize, kernelSize * (kernelSize**no), kernelSize**no, 0))
            for i in range(repeat):
                if i % 2 == 0:
                    indexList.append(getIndeices(shape, kernelSize, kernelSize, kernelSize * (kernelSize**no), kernelSize**no, kernelSize**no))
                else:
                    indexList.append(getIndeices(shape, kernelSize, kernelSize, kernelSize * (kernelSize**no), kernelSize**no, 0))

        indexIList = [item[0] for item in indexList]
        indexJList = [item[1] for item in indexList]

        # to share parameters along RG direction, pass a shorter layerList
        if len(layerList) == repeat + 1:
            layerList = layerList * depth

        assert len(layerList) == len(indexIList)

        if kernelDim == 2:
            kernelShape = [kernelSize, kernelSize]
        elif kernelDim == 1:
            kernelShape = [kernelSize * 2]

        super(MERA, self).__init__(kernelShape, indexIList, indexJList, layerList, prior, name)


class ParameterizedMERA(ParameterizedHierarchyBijector):
    def __init__(self, kernelDim, length, layerList, meanNNlist, scaleNNlist, nMixing=5, repeat=1, depth=None, decimal=None, rounding=None, name="ParameterizedMERA"):
        kernelSize = 2
        shape = [length, length]
        if depth is None or depth == -1:
            depth = int(math.log(length, kernelSize))

        indexList = []

        for no in range(depth):
            indexList.append(getIndeices(shape, kernelSize, kernelSize, kernelSize * (kernelSize**no), kernelSize**no, 0))
            for i in range(repeat):
                if i % 2 == 0:
                    indexList.append(getIndeices(shape, kernelSize, kernelSize, kernelSize * (kernelSize**no), kernelSize**no, kernelSize**no))
                else:
                    indexList.append(getIndeices(shape, kernelSize, kernelSize, kernelSize * (kernelSize**no), kernelSize**no, 0))

        indexIList = [item[0] for item in indexList]
        indexJList = [item[1] for item in indexList]

        # to share parameters along RG direction, pass a shorter layerList
        if len(layerList) == repeat + 1:
            layerList = layerList * depth

        assert len(meanNNlist) == len(scaleNNlist)

        if len(meanNNlist) == 1:
            meanNNlist = meanNNlist * (depth - 1)
            scaleNNlist = scaleNNlist * (depth - 1)

        assert len(layerList) == len(indexIList)
        assert len(meanNNlist) == depth - 1

        if kernelDim == 2:
            kernelShape = [kernelSize, kernelSize]
        elif kernelDim == 1:
            kernelShape = [kernelSize * 2]

        self.repeat = repeat

        lastPrior = source.MixtureDiscreteLogistic([3, 1, 4], nMixing, decimal, rounding)

        prior = source.ParameterizedHierarchyPrior(3, length, lastPrior, repeat=repeat, decimal=decimal, rounding=rounding)

        super(ParameterizedMERA, self).__init__(kernelShape, indexIList, indexJList, layerList, meanNNlist, scaleNNlist, decimal, prior, name)


def sig2prt(t):
    return t.reshape(t.shape[0], t.shape[1], t.shape[2], t.shape[3] // 2, 2).permute([0, 2, 1, 3, 4]).reshape(t.shape[0] * t.shape[2], t.shape[1], t.shape[3] // 2, 2)


def prt2sig(t):
    return t.reshape(t.shape[0] // (2 * t.shape[-2]), t.shape[-2] * 2, t.shape[1], t.shape[-2] * 2).permute([0, 2, 1, 3])


class OneToTwoMERA(Flow):
    def __init__(self, length, layerList, meanNNlist=None, scaleNNlist=None, repeat=1, nMixing=5, decimal=None, rounding=None, name="OneToTwoMERA"):
        depth = int(math.log(length, 2))

        if meanNNlist is None or scaleNNlist is None:
            prior = source.SimpleHierarchyPrior(length, nMixing, decimal, rounding)
        else:
            lastPrior = source.MixtureDiscreteLogistic([3, 1, 4], nMixing, decimal, rounding)
            prior = source.PassiveHierarchyPrior(length, lastPrior, decimal=decimal, rounding=rounding)
        super(OneToTwoMERA, self).__init__(prior, name)

        self.decimal = decimal
        self.rounding = rounding
        self.repeat = repeat

        layerList = layerList * depth

        self.layerList = torch.nn.ModuleList(layerList)

        if meanNNlist is not None and scaleNNlist is not None:
            meanNNlist = meanNNlist * depth
            scaleNNlist = scaleNNlist * depth

            self.meanNNlist = torch.nn.ModuleList(meanNNlist)
            self.scaleNNlist = torch.nn.ModuleList(scaleNNlist)
        else:
            self.meanNNlist = None
            self.scaleNNlist = None

    def inverse(self, x):
        depth = int(math.log(x.shape[-1], 2))
        for _ in range(2):
            up = x
            DN = []
            for no in range(depth):
                _x = sig2prt(up)
                up = _x[:, :, :, 0].contiguous()
                dn = _x[:, :, :, 1].contiguous()
                for i in range(2 * self.repeat):
                    if i % 2 == 0:
                        pass
                    else:
                        pass

    def forward(self, z):
        pass

    def logProbability(self, x, K=None):
        z, logp = self.inverse(x)
        if self.prior is not None:
            if self.meanNNlist is not None and self.scaleNNlist is not None:
                return self.prior.logProbability(z, K, self.meanList, self.scaleList) + logp
            else:
                return self.prior.logProbability(z, K) + logp
        return logp


def im2grp(t):
    return t.reshape(t.shape[0], t.shape[1], t.shape[2] // 2, 2, t.shape[3] // 2, 2).permute([0, 1, 2, 4, 3, 5]).reshape(t.shape[0], t.shape[1], -1, 4)


def grp2im(t):
    return t.reshape(t.shape[0], t.shape[1], int(t.shape[2] ** 0.5), int(t.shape[2] ** 0.5), 2, 2).permute([0, 1, 2, 4, 3, 5]).reshape(t.shape[0], t.shape[1], int(t.shape[2] ** 0.5) * 2, int(t.shape[2] ** 0.5) * 2)


def form(tensor):
    shape = int(tensor.shape[-2] ** 0.5)
    return tensor.reshape(tensor.shape[0], tensor.shape[1], shape, shape, 2, 2).permute([0, 1, 2, 4, 3, 5]).reshape(tensor.shape[0], tensor.shape[1], shape * 2, shape * 2)


def reform(tensor):
    return tensor.reshape(tensor.shape[0], tensor.shape[1] // 3, 3, tensor.shape[2], tensor.shape[3]).permute([0, 1, 3, 4, 2]).contiguous().reshape(tensor.shape[0], tensor.shape[1] // 3, tensor.shape[2] * tensor.shape[3], 3)


class SimpleMERA(Flow):
    def __init__(self, length, layerList, meanNNlist=None, scaleNNlist=None, repeat=1, nMixing=5, decimal=None, rounding=None, name="SimpleMERA"):
        depth = int(math.log(length, 2))

        if meanNNlist is None or scaleNNlist is None:
            prior = source.SimpleHierarchyPrior(length, nMixing, decimal, rounding)
        else:
            lastPrior = source.MixtureDiscreteLogistic([3, 1, 4], nMixing, decimal, rounding)
            prior = source.PassiveHierarchyPrior(length, lastPrior, decimal=decimal, rounding=rounding)
        super(SimpleMERA, self).__init__(prior, name)

        self.decimal = decimal
        self.rounding = rounding
        self.repeat = repeat

        layerList = layerList * depth

        self.layerList = torch.nn.ModuleList(layerList)

        if meanNNlist is not None and scaleNNlist is not None:
            meanNNlist = meanNNlist * depth
            scaleNNlist = scaleNNlist * depth

            self.meanNNlist = torch.nn.ModuleList(meanNNlist)
            self.scaleNNlist = torch.nn.ModuleList(scaleNNlist)
        else:
            self.meanNNlist = None
            self.scaleNNlist = None

    def inverse(self, x):
        depth = int(math.log(x.shape[-1], 2))
        self.meanList = []
        self.scaleList = []

        ul = x
        UR = []
        DL = []
        DR = []
        for no in range(depth):
            _x = im2grp(ul)
            ul = _x[:, :, :, 0].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
            ur = _x[:, :, :, 1].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
            dl = _x[:, :, :, 2].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
            dr = _x[:, :, :, 3].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
            for i in range(4 * self.repeat):
                if i % 4 == 0:
                    tmp = torch.cat([ur, dl, dr], 1)
                    tmp = self.rounding(self.layerList[no * 4 + i](self.decimal.inverse_(tmp)) * self.decimal.scaling)
                    ul = ul + tmp
                elif i % 4 == 1:
                    tmp = torch.cat([ul, dl, dr], 1)
                    tmp = self.rounding(self.layerList[no * 4 + i](self.decimal.inverse_(tmp)) * self.decimal.scaling)
                    ur = ur + tmp
                elif i % 4 == 2:
                    tmp = torch.cat([ul, ur, dr], 1)
                    tmp = self.rounding(self.layerList[no * 4 + i](self.decimal.inverse_(tmp)) * self.decimal.scaling)
                    dl = dl + tmp
                else:
                    tmp = torch.cat([ul, ur, dl], 1)
                    tmp = self.rounding(self.layerList[no * 4 + i](self.decimal.inverse_(tmp)) * self.decimal.scaling)
                    dr = dr + tmp
            if self.meanNNlist is not None and self.scaleNNlist is not None:
                self.meanList.append(reform(self.meanNNlist[no](self.decimal.inverse_(ul))).contiguous())
                self.scaleList.append(reform(self.scaleNNlist[no](self.decimal.inverse_(ul))).contiguous())

            UR.append(ur)
            DL.append(dl)
            DR.append(dr)

        for no in reversed(range(depth)):
            ur = UR[no].reshape(*ul.shape, 1)
            dl = DL[no].reshape(*ul.shape, 1)
            dr = DR[no].reshape(*ul.shape, 1)
            ul = ul.reshape(*ul.shape, 1)

            _x = torch.cat([ul, ur, dl, dr], -1).reshape(*ul.shape[:2], -1, 4)
            ul = grp2im(_x).contiguous()

        return ul, ul.new_zeros(ul.shape[0])

    def forward(self, z):
        depth = int(math.log(z.shape[-1], 2))

        ul = z
        UR = []
        DL = []
        DR = []
        for no in range(depth):
            _x = im2grp(ul)
            ul = _x[:, :, :, 0].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
            ur = _x[:, :, :, 1].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
            dl = _x[:, :, :, 2].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
            dr = _x[:, :, :, 3].reshape(*_x.shape[:2], int(_x.shape[2] ** 0.5), int(_x.shape[2] ** 0.5)).contiguous()
            UR.append(ur)
            DL.append(dl)
            DR.append(dr)

        for no in reversed(range(depth)):
            ur = UR[no]
            dl = DL[no]
            dr = DR[no]
            for i in reversed(range(4 * self.repeat)):
                if i % 4 == 0:
                    tmp = torch.cat([ur, dl, dr], 1)
                    tmp = self.rounding(self.layerList[no * 4 + i](self.decimal.inverse_(tmp)) * self.decimal.scaling)
                    ul = ul - tmp
                elif i % 4 == 1:
                    tmp = torch.cat([ul, dl, dr], 1)
                    tmp = self.rounding(self.layerList[no * 4 + i](self.decimal.inverse_(tmp)) * self.decimal.scaling)
                    ur = ur - tmp
                elif i % 4 == 2:
                    tmp = torch.cat([ul, ur, dr], 1)
                    tmp = self.rounding(self.layerList[no * 4 + i](self.decimal.inverse_(tmp)) * self.decimal.scaling)
                    dl = dl - tmp
                else:
                    tmp = torch.cat([ul, ur, dl], 1)
                    tmp = self.rounding(self.layerList[no * 4 + i](self.decimal.inverse_(tmp)) * self.decimal.scaling)
                    dr = dr - tmp

            ur = ur.reshape(*ul.shape, 1)
            dl = dl.reshape(*ul.shape, 1)
            dr = dr.reshape(*ul.shape, 1)
            ul = ul.reshape(*ul.shape, 1)

            _x = torch.cat([ul, ur, dl, dr], -1).reshape(*ul.shape[:2], -1, 4)
            ul = grp2im(_x).contiguous()

        return ul, ul.new_zeros(ul.shape[0])

    def logProbability(self, x, K=None):
        z, logp = self.inverse(x)
        if self.prior is not None:
            if self.meanNNlist is not None and self.scaleNNlist is not None:
                return self.prior.logProbability(z, K, self.meanList, self.scaleList) + logp
            else:
                return self.prior.logProbability(z, K) + logp
        return logp

