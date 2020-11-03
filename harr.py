import torch
from flow import ScalingNshifting

originalChnl = 3
lineSize = 4
batch = 2
hchnl = 10
nhidden = 2

decimal = ScalingNshifting(256, -128)

o = decimal.inverse_(torch.randint(255, [batch, 3, lineSize]))
e = decimal.inverse_(torch.randint(255, [batch, 3, lineSize]))

v = torch.cat([o, e], -1).reshape(batch, 3, 2, lineSize).permute([0, 1, 3, 2]).reshape(batch, 3, 2 * lineSize)

transMatrix = torch.tensor([[1/2, 1/2, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1/2, 1/2, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1/2, 1/2, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1/2, 1/2],
                            [-1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, -1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, -1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, -1, 1]])

trans_v = torch.matmul(v, transMatrix.t())

def initMethod1(originalChnl):
    return torch.cat([torch.ones(originalChnl, 1), torch.zeros(originalChnl, 2)], 1).reshape(originalChnl, 1, 3)

def initMethod2(originalChnl):
    return torch.cat([torch.zeros(originalChnl, 2), torch.ones(originalChnl, 1) / 2], 1).reshape(originalChnl, 1, 3)


def buildLayers(initMethod, originalChnl, hchnl, nhidden=1, right=True):
    chnlList = [originalChnl] + [hchnl] * (nhidden + 1) + [originalChnl]
    if right:
        convList = [torch.nn.ReplicationPad1d((0, 2), )]
    else:
        convList = [torch.nn.ReplicationPad1d((2, 0), )]

    for no, chnl in enumerate(chnlList[:-1]):
        if no == 0:
            layer = torch.nn.Conv1d(chnl, chnlList[no + 1], 3)
            tmp = initMethod()
            tmp1 = torch.cat([tmp, torch.zeros(originalChnl, chnl, 3)], 1).reshape(-1)[:originalChnl * chnl * 3].reshape(originalChnl, chnl, 3)
            tmp2 = -torch.cat([tmp, torch.zeros(originalChnl, chnl, 3)], 1).reshape(-1)[:originalChnl * chnl * 3].reshape(originalChnl, chnl, 3)
            layer.weight.data[:2 * originalChnl, :, :] = torch.cat([tmp1, tmp2], 0).float()
        elif no == len(chnlList) - 2:
            layer = torch.nn.Conv1d(chnl, chnlList[no + 1], 3, padding=1, padding_mode="replicate")
            layer.weight.data.zero_()
            tmp = torch.cat([torch.zeros(originalChnl, 1), torch.ones(originalChnl, 1), torch.zeros(originalChnl, 1)], 1).reshape(originalChnl, 1, 3)
            tmp1 = torch.cat([tmp, torch.zeros(originalChnl, originalChnl, 3)], 1).reshape(-1)[:originalChnl * originalChnl * 3].reshape(originalChnl, originalChnl, 3)
            tmp2 = -torch.cat([tmp, torch.zeros(originalChnl, originalChnl, 3)], 1).reshape(-1)[:originalChnl * originalChnl * 3].reshape(originalChnl, originalChnl, 3)
            layer.weight.data[:originalChnl, :2 * originalChnl, :] = torch.cat([tmp1, tmp2], 1).float()
        else:
            layer = torch.nn.Conv1d(chnl, chnlList[no + 1], 3, padding=1, padding_mode="replicate")
            tmp = torch.cat([torch.zeros(2 * originalChnl, 1), torch.ones(2 * originalChnl, 1), torch.zeros(2 * originalChnl, 1)], 1).reshape(2 * originalChnl, 1, 3)
            layer.weight.data[:2 * originalChnl, :, :] = torch.cat([tmp, torch.zeros(2 * originalChnl, chnl, 3)], 1).reshape(-1)[:2 * originalChnl * chnl * 3].reshape(2 * originalChnl, chnl, 3).float()
        layer.bias.data[:2 * originalChnl].zero_()
        convList.append(layer)
        if no != len(chnlList) - 2:
            convList.append(torch.nn.ReLU(inplace=True))

    return torch.nn.Sequential(*convList)

layer1 = buildLayers(lambda: initMethod1(originalChnl), originalChnl, hchnl, nhidden)

layer2 = buildLayers(lambda: initMethod2(originalChnl), originalChnl, hchnl, nhidden, False)

d = e - layer1(o)

s = o + layer2(d)

vpp = torch.cat([s, d], -1)

assert vpp.allclose(trans_v)

l1 = torch.nn.Conv1d(3, 6, 3)
l1.bias.data.zero_()

weight = torch.tensor([[[1, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]],
                       [[0, 0, 0],
                        [1, 0, 0],
                        [0, 0, 0]],
                       [[0, 0, 0],
                        [0, 0, 0],
                        [1, 0, 0]],
                       [[-1, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]],
                       [[0, 0, 0],
                        [-1, 0, 0],
                        [0, 0, 0]],
                       [[0, 0, 0],
                        [0, 0, 0],
                        [-1, 0, 0]]]).float()

l1.weight.data[:, :, :] = weight

o1 = torch.nn.functional.pad(o, (0, 2), "replicate")

o2 = l1(o1)

o3 = torch.relu(o2)

l2 = torch.nn.Conv1d(6, 3, 3, padding=1, padding_mode="replicate")
l2.bias.data.zero_()

weight = torch.tensor([[[0, 1, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, -1, 0],
                        [0, 0, 0],
                        [0, 0, 0]],
                       [[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, -1, 0],
                        [0, 0, 0]],
                       [[0, 0, 0],
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, -1, 0]]]).float()

l2.weight.data[:, :, :] = weight

tmpd = l2(o3)


'''

l11 = torch.nn.Conv1d(3, hchnl, 3)

l11.weight.data.zero_()

weight = torch.eye(3)
weight = torch.tensor([[[1, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]],
                       [[0, 0, 0],
                        [1, 0, 0],
                        [0, 0, 0]],
                       [[0, 0, 0],
                        [0, 0, 0],
                        [1, 0, 0]]]).float()

l11.weight.data[:3, :3, :] = weight
l11.bias.data.zero_()

l12 = torch.nn.Conv1d(hchnl, hchnl, 3, padding=1)

l12.weight.data.zero_()
weight = torch.tensor([[[0, 1, 0],
                        [0, 0, 0],
                        [0, 0, 0]],
                       [[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]],
                       [[0, 0, 0],
                        [0, 0, 0],
                        [0, 1, 0]]]).float()

l12.weight.data[:3, :3, :] = weight
l12.bias.data.zero_()

l13 = torch.nn.Conv1d(hchnl, 3, 3, padding=1)

l13.weight.data.zero_()
weight = torch.tensor([[[0, 1, 0],
                        [0, 0, 0],
                        [0, 0, 0]],
                       [[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]],
                       [[0, 0, 0],
                        [0, 0, 0],
                        [0, 1, 0]]]).float()

l13.weight.data[:3, :3, :] = weight
l13.bias.data.zero_()

import pdb
pdb.set_trace()

tmpd = l13(l12(l11(torch.nn.functional.pad(o, (0, 2), "replicate"))))

d = e - tmpd

l2 = torch.nn.Conv1d(3, 3, 3)

weight = torch.tensor([[[0, 0, 1/2],
                        [0, 0, 0],
                        [0, 0, 0]],
                       [[0, 0, 0],
                        [0, 0, 1/2],
                        [0, 0, 0]],
                       [[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 1/2]]])

l2.weight.data = weight
l2.bias.data.zero_()

tmps = l2(torch.nn.functional.pad(d, (2, 0), "replicate"))

s = o + tmps

vp = torch.cat([s, d], -1)

assert vp.allclose(trans_v)
'''

