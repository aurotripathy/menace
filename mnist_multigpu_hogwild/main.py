"""
Adding multi-gpu support to mnist w/hogwild
"""
from __future__ import print_function
import argparse
import torch
import torch.multiprocessing as mp
from model import Net
from train import train, test
from shared_optim import SharedAdam


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-ids', type=int, default=-1, nargs='+',
                    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument('--amsgrad', default=True, metavar='AM',
                    help='Adam optimizer amsgrad parameter')



if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn', force=True)


    use_cuda = args.cuda and torch.cuda.is_available()
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}


    shared_model = Net()  # On cpu, shared across gpus
    shared_model.share_memory() # gradients are allocated lazily, so they are not shared here

    shared_optimizer = SharedAdam(shared_model.parameters(),
                                  lr=args.lr, amsgrad=args.amsgrad)
    shared_optimizer.share_memory()
    
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args,
                                           shared_model, shared_optimizer,
                                           dataloader_kwargs))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # Once training is complete, we can test the model
    test(args, shared_model, dataloader_kwargs)
