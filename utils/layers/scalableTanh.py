import math
import torch
import torch.nn as nn

class ScalableTanh(nn.Module):
    def __init__(self,input_size):
        super(ScalableTanh,self).__init__()
        self.scale = nn.Parameter(torch.randn(input_size) * 0.01, requires_grad=True)
    def forward(self,x):
        return self.scale * torch.tanh(x)

