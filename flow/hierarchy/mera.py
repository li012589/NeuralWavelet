import math

from .template import HierarchyBijector, ParameterizedHierarchyBijector, OneToTwoHierarchyBijector
from utils import getIndeices
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


class OneToTwoMERA(OneToTwoHierarchyBijector):
    def __init__(self, kernelDim, length, layerList, repeat=1, depth=None, prior=None, name="OneToTwoMERA"):
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

        super(OneToTwoMERA, self).__init__(kernelShape, indexIList, indexJList, layerList, prior, name)

