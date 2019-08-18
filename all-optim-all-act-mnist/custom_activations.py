"""
Additional activation functions not available in PyTorch are
packaged here as a subclass of nn.Module.
The activation function can now be easily used in training models
"""

import math
import torch
import torch.nn as nn

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


class SoftExponential(nn.Module):
    '''
    Parameters:
        - alpha - trainable parameter
    References:
        https://arxiv.org/pdf/1602.01321.pdf
    '''
    def __init__(self, alpha=None):

        super(SoftExponential, self).__init__()

        # initialize alpha as trainable tensor
        if alpha is None:
            self.alpha = torch.tensor(0.0)
        else:
            self.alpha = torch.tensor(alpha)

        self.alpha.requires_grad = True # trainable, set requiresGrad to true

    def forward(self, x):

        if self.alpha == 0.0:
            return x

        if self.alpha < 0.0:
            return - torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha

        if self.alpha > 0.0:
            return (torch.exp(self.alpha * x) - 1)/ self.alpha + self.alpha
