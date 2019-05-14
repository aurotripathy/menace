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
const_val = 5.0

nGPUs = torch.cuda.device_count()


def time_all_reduce_vgg_model_size(repeat=12, discard=2):
    print('repeat', repeat, 'discard', discard)
    times_per_iteration = []
    for _ in range(repeat):
        tensors = [torch.FloatTensor(size).fill_(const_val) for i in range(nGPUs)]  # dim size, value random
        expected = torch.FloatTensor(size).zero_()
        for tensor in tensors:
            expected.add_(tensor)  # add in-place on CPU

        tensors = [tensors[i].cuda(i) for i in range(nGPUs)]  # move ith tensor into ith GPU

        start_time = timer.time()
        nccl.all_reduce(tensors)
        times_per_iteration.append(timer.time() - start_time)
        for tensor in tensors:
            assert torch.all(torch.eq(tensor.cpu(), expected))  # move to CPU and compare
    times_per_iteration = times_per_iteration[discard:]  # discard first few
    print(len(times_per_iteration), times_per_iteration)
    avg_time_taken = sum(times_per_iteration)/(repeat - discard)
    return avg_time_taken

reduction_time = time_all_reduce_vgg_model_size(22, 2)

print('Python VERSION:', sys.version)
print('pyTorch VERSION:', torch.__version__)
print ('Available GPUs ', nGPUs)
print("Time taken:{:.9f} seconds".format(reduction_time))
