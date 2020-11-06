import torch


# 1 = right, 2 = left
def idenInitMethod1(originalChnl):
    return torch.cat([torch.ones(originalChnl, 1), torch.zeros(originalChnl, 2)], 1).reshape(originalChnl, 1, 3)


def idenInitMethod2(originalChnl):
    return torch.cat([torch.zeros(originalChnl, 2), torch.ones(originalChnl, 1)], 1).reshape(originalChnl, 1, 3)


def harrInitMethod1(originalChnl):
    return torch.cat([torch.ones(originalChnl, 1), torch.zeros(originalChnl, 2)], 1).reshape(originalChnl, 1, 3)


def harrInitMethod2(originalChnl):
    return torch.cat([torch.zeros(originalChnl, 2), torch.ones(originalChnl, 1) / 2], 1).reshape(originalChnl, 1, 3)


def leGallInitMethod1(originalChnl):
    return torch.cat([torch.ones(originalChnl, 1, 2) / 2, torch.zeros(originalChnl, 1, 1)], -1)


def leGallInitMethod2(originalChnl):
    return torch.cat([torch.zeros(originalChnl, 1, 1), torch.ones(originalChnl, 1, 2) / 4], -1)


def buildWaveletLayers(initMethod, originalChnl, hchnl, nhidden=1, right=True):
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
