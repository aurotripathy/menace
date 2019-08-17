import torch
import torch.nn as nn
import math

class Gelu(nn.Module):
    def __init__(self):
        super(Gelu, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))
