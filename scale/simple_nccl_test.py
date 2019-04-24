import torch
import torch.cuda.nccl as nccl
import torch.cuda

nGPUs = 4

def test_all_reduce():
    tensors = [torch.FloatTensor(10).fill_(5) for i in range(nGPUs)]  # dim 10, value 5
    expected = torch.FloatTensor(10).zero_()
    for tensor in tensors:
        expected.add_(tensor)  # add in-place

    tensors = [tensors[i].cuda(i) for i in range(nGPUs)]  # move ith tensor into ith GPU 
    nccl.all_reduce(tensors)

    for tensor in tensors:
        assert torch.all(torch.eq(tensor.cpu(), expected))  # move to CPU and compare


test_all_reduce()
