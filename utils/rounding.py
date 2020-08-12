import torch


class RoundingWidentityGradient(torch.autograd.Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, input):
        rounded = torch.round(input)
        return rounded

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


roundingWidentityGradient = RoundingWidentityGradient.apply