"""
Simple test to all_reduce_sum a set of model parameter the size of the VGG-16 model (~138M floats)
TO get a breakdown of the VGG model, see...
https://stackoverflow.com/questions/28232235/how-to-calculate-the-number-of-parameters-of-convolutional-neural-networks
"""

import torch
import torch.cuda.nccl as nccl
import torch.cuda
import time as timer

size = 138000000
rand_val = 5.0

nGPUs = 4

def all_reduce_vgg_model_size():
    tensors = [torch.FloatTensor(size).fill_(rand_val) for i in range(nGPUs)]  # dim size, value random
    expected = torch.FloatTensor(size).zero_()
    for tensor in tensors:
        expected.add_(tensor)  # add in-place

    start_time = timer.time()
    tensors = [tensors[i].cuda(i) for i in range(nGPUs)]  # move ith tensor into ith GPU 
    nccl.all_reduce(tensors)
    print("time taken:", timer.time() - start_time)

    for tensor in tensors:
        assert torch.all(torch.eq(tensor.cpu(), expected))  # move to CPU and compare


all_reduce_vgg_model_size()
