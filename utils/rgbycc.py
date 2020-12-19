import torch


def rgb2ycc(samples, round=False, batch=False):
    if not batch:
        samples = samples.reshape(1, *samples.shape)
    k = torch.tensor([[0.299, 0.587, 0.114], [-0.1687, -0.3313, 0.5], [0.5, -0.4187, -0.0813]])
    b = torch.tensor([0, 128, 128])
    samples = (torch.matmul(samples.permute(0, 2, 3, 1), k.T) + b).permute(0, 3, 1, 2)
    if round:
        samples = torch.round(samples)
    if not batch:
        samples = samples.reshape(*samples.shape[1:])
    return samples


def ycc2rgb(samples, round=False, batch=False):
    if not batch:
        samples = samples.reshape(1, *samples.shape)
    k = torch.tensor([[1, 0, 1.402], [1, -0.34414, -0.71414], [1, 1.772, 0]])
    b = torch.tensor([-179.456, 135.45984, -226.816])
    samples = (torch.matmul(samples.permute(0, 2, 3, 1), k.T) + b).permute(0, 3, 1, 2)
    if round:
        samples = torch.round(samples)
    if not batch:
        samples = samples.reshape(*samples.shape[1:])
    return samples
