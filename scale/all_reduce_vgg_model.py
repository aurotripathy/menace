"""
Simple test to all_reduce_sum a set of model parameters the size of the VGG-16 model (~138M floats)
    1. Paramters are loaded into each of the N=4 GPUs
    2. nccl.all_reduce is invoked on the paramters
TO get a breakdown of the VGG model size, see...
https://stackoverflow.com/questions/28232235/how-to-calculate-the-number-of-parameters-of-convolutional-neural-networks
"""

from __future__ import print_function
import torch
import torch.cuda.nccl as nccl
import torch.cuda
import time as timer
import sys

size = 138000000
rand_val = 5.0

nGPUs = torch.cuda.device_count()


def all_reduce_vgg_model_size():
    tensors = [torch.FloatTensor(size).fill_(rand_val) for i in range(nGPUs)]  # dim size, value random
    expected = torch.FloatTensor(size).zero_()
    for tensor in tensors:
        expected.add_(tensor)  # add in-place

    tensors = [tensors[i].cuda(i) for i in range(nGPUs)]  # move ith tensor into ith GPU
    
    start_time = timer.time()
    nccl.all_reduce(tensors)
    time_taken = timer.time() - start_time
    for tensor in tensors:
        assert torch.all(torch.eq(tensor.cpu(), expected))  # move to CPU and compare
    return time_taken

_ = all_reduce_vgg_model()  # throw this one away as "warm up"
reduction_time = all_reduce_vgg_model_size()

print('Python VERSION:', sys.version)
print('pyTorch VERSION:', torch.__version__)
print ('Available GPUs ', nGPUs)
print("Time taken:{:.3f} seconds".format(reduction_time))
