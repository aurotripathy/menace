"""
Additional activation functions not rasdily aavilable in PyTorch
packaged here as a subclass of PyTorch nn.Module.
The activation function can now be easily used in training models

"""
import torch
import torch.nn as nn
import math

class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))

class SiLU(nn.Module):
    '''
    Also know as 'swish'
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    Reference:
        https://arxiv.org/abs/1710.05941
    '''
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, x):
        return x * torch.sigmoid(x)
