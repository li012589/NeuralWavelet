import math

from .template import HierarchyBijector
from .im2col import getIndeices


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

        if len(layerList) == repeat + 1:
            layerList = layerList * depth

        assert len(layerList) == len(indexIList)

        if kernelDim == 2:
            kernelShape = [kernelSize, kernelSize]
        elif kernelDim == 1:
            kernelShape = [kernelSize * 2]

        super(MERA, self).__init__(kernelShape, indexIList, indexJList, layerList, prior, name)


