"""
Multi-gpu training module
"""
import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from setproctitle import setproctitle as ptitle
from model import Net


def ensure_shared_grads(model, shared_model, gpu=False):
    """ working comment --- maintains the grads on the CPU """
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        shared_param._grad = param.grad.cpu()


def train(rank, args,
          shared_model, optimizer,
          dataloader_kwargs):
    """ Per process training """
    ptitle('Training Agent: {}'.format(rank))

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    train_data_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=args.batch_size, shuffle=True, num_workers=1,
        **dataloader_kwargs)

    local_model = Net()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            local_model = local_model.cuda()
            local_model.train()
            for epoch in range(1, args.epochs + 1):
                pid = os.getpid()
                for batch_idx, (data, target) in enumerate(train_data_loader):
                    # optimizer.zero_grad()
                    local_model.zero_grad()
                    local_model.load_state_dict(shared_model.state_dict())
                    output = local_model(data.cuda())
                    loss = F.nll_loss(output, target.cuda())
                    loss.backward()
                    ensure_shared_grads(local_model, shared_model, gpu=gpu_id >= 0)
                    if batch_idx % args.log_interval == 0:
                        print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            pid, epoch, batch_idx * len(data), len(train_data_loader.dataset),
                            100. * batch_idx / len(train_data_loader), loss.item()))
                    optimizer.step()


def test(args, model, dataloader_kwargs):
    """ Testring on CPU """
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1,
        **dataloader_kwargs)

    test_epoch(model, test_loader)


def train_epoch(epoch, args, local_model, shared_model,
                gpu_id, data_loader, optimizer):
    """ Train a single epoch """
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        with torch.cuda.device(gpu_id):
            local_model.load_state_dict(shared_model.state_dict())
            output = local_model(data.to(gpu_id))
            loss = F.nll_loss(output, target.to(gpu_id))
            loss.backward()
            ensure_shared_grads(local_model, shared_model, gpu=gpu_id >= 0)
            if batch_idx % args.log_interval == 0:
                print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), loss.item()))
        optimizer.step()


def test_epoch(model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
