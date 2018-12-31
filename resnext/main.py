from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from resnext import ResNeXt
from utils import progress_bar
from pudb import set_trace
    
def train(args, model, device, train_loader, optimizer, criterion, epoch):
    print("\nEpoch: {}".format(epoch))
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        progress_bar(i, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(i + 1), 100.*correct/total, correct, total))
        
def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar(i, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(i + 1), 100.*correct/total, correct, total))

            
def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: %(default)s)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M',
                        help='SGD momentum (default: %(default)s)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: %(default)s)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before \
                        logging training status (default: %(default)s)')    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model (default: %(default)s)')
    
    args = parser.parse_args()
    print('Arguments:', args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}


    # Load and normalize the CIFAR10 training and test datasets using torchvision
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_transform_pipe = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])

    test_transform_pipe = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../cifar10_data', train=True, download=True,
                         transform=train_transform_pipe),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../cifar10_data', train=False,
                         transform=test_transform_pipe),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = ResNeXt(num_blocks_list=[3,3,3], cardinality=32, bottleneck_width=4).to(device)
    total_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total model paramters:", total_model_params)
    
    model = torch.nn.DataParallel(model).cuda()
    
    # Define a loss and optomizer function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Train and test the network
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        test(args, model, device, test_loader, criterion)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    print('Using PyTorch version: {}'.format(torch.__version__))
    main()
