import torch
from .etc import broadcastSize, selectArgs


def mixtureLogProbability(distribution, parts, eps=1e-8):
    '''
    input of size [b, c, w, h]
    args are torch.tensor[s] of size [parts, ...]
    parts is torch.tensor of size [parts]
    '''
    def logProbability(x, *args, **kwargs):
        x = x.view(*x.shape, 1)
        args = [term.permute(torch.arange(len(term.shape)).roll(-1).tolist()) for term in args]
        args = broadcastSize(x.shape, args, test=False)
        logprobs = distribution(x, *args, **kwargs)
        assert logprobs.shape[0] == x.shape[0]
        assert logprobs.shape[-1] == parts.shape[-1]
        probs = torch.sum(torch.exp(logprobs) * parts, dim=-1)
        return torch.log(probs + eps)
    return logProbability


def mixtureSample(sampleDistribution, parts):
    '''
    size is python list of sampling size [batch, a1, ....]
    args are torch.tensor[s] of size [parts, ...]
    parts is torch.tensor of size [parts]
    '''
    def sampling(size, *args, **kwargs):
        args = selectArgs(parts, args)
        return sampleDistribution(size, *args, **kwargs)
    return sampling


