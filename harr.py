import torch
from flow import ScalingNshifting

lineSize = 4
batch = 2
hchnl = 5

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

originalChnl = 3
chnlList = [originalChnl, hchnl, hchnl, originalChnl]
convList = [torch.nn.ReplicationPad1d((0, 2), )]

for no, chnl in enumerate(chnlList[:-1]):
    if no == 0:
        layer = torch.nn.Conv1d(chnl, chnlList[no + 1], 3)
        layer(o)
        tmp = torch.cat([torch.ones(originalChnl, 1), torch.zeros(originalChnl, 2)], 1).reshape(originalChnl, 1, 3)
        layer.weight.data[:originalChnl, :, :] = torch.cat([tmp, torch.zeros(originalChnl, chnl, 3)], 1).reshape(-1)[:originalChnl * chnl * 3].reshape(originalChnl, chnl, 3).float()
    else:
        layer = torch.nn.Conv1d(chnl, chnlList[no + 1], 3, padding=1, padding_mode="replicate")
        if no == len(chnlList) - 2:
            layer.weight.data.zero_()
        tmp = torch.cat([torch.zeros(originalChnl, 1), torch.ones(originalChnl, 1), torch.zeros(originalChnl, 1)], 1).reshape(originalChnl, 1, 3)
        layer.weight.data[:originalChnl, :, :] = torch.cat([tmp, torch.zeros(originalChnl, chnl, 3)], 1).reshape(-1)[:originalChnl * chnl * 3].reshape(originalChnl, chnl, 3).float()
    layer.bias.data[:originalChnl].zero_()
    convList.append(layer)
    if no != len(chnlList) - 2:
        convList.append(torch.nn.Hardtanh(inplace=True))

layer1 = torch.nn.Sequential(*convList)

originalChnl = 3
chnlList = [originalChnl, hchnl, hchnl, originalChnl]
convList = [torch.nn.ReplicationPad1d((2, 0), )]

for no, chnl in enumerate(chnlList[:-1]):
    if no == 0:
        layer = torch.nn.Conv1d(chnl, chnlList[no + 1], 3)
        layer(o)
        tmp = torch.cat([torch.zeros(originalChnl, 2), torch.ones(originalChnl, 1) / 2], 1).reshape(originalChnl, 1, 3)
        layer.weight.data[:originalChnl, :, :] = torch.cat([tmp, torch.zeros(originalChnl, chnl, 3)], 1).reshape(-1)[:originalChnl * chnl * 3].reshape(originalChnl, chnl, 3).float()
    else:
        layer = torch.nn.Conv1d(chnl, chnlList[no + 1], 3, padding=1, padding_mode="replicate")
        if no == len(chnlList) - 2:
            layer.weight.data.zero_()
        tmp = torch.cat([torch.zeros(originalChnl, 1), torch.ones(originalChnl, 1), torch.zeros(originalChnl, 1)], 1).reshape(originalChnl, 1, 3)
        layer.weight.data[:originalChnl, :, :] = torch.cat([tmp, torch.zeros(originalChnl, chnl, 3)], 1).reshape(-1)[:originalChnl * chnl * 3].reshape(originalChnl, chnl, 3).float()
    layer.bias.data[:originalChnl].zero_()
    convList.append(layer)
    if no != len(chnlList) - 2:
        convList.append(torch.nn.Hardtanh(inplace=True))

layer2 = torch.nn.Sequential(*convList)

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

o4 = l2(o3)

import pdb
pdb.set_trace()


'''
import pdb
pdb.set_trace()

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

