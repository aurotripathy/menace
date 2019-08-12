"""
Tries every flavor of activation with every flavor of optimizer  
Network is Lenet5, dataset is mnist
"""

from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pudb import set_trace

data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)


cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width': 1200,
    'height': 600,
}

criterion = nn.CrossEntropyLoss()  # keeping this a constant and varying the activations and optims
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(epoch, net, optimizer):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.to(device)
        
        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        loss.backward()
        optimizer.step()


def test(net, optimizer):
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))


def train_and_test(epoch, net, optimizer_str, activation):
    optimizer_fn = getattr(optim, optimizer_str)
    set_trace()
    # optimizer = optimizer_fn(net.parameters(), lr=2e-3)
    optimizer = optim.Adam(net.parameters(), lr=2e-3)
    train(epoch, net, optimizer)
    test(net, optimizer)

nb_epochs = 3
def main(optimizer, activation):
    net = LeNet5(activation).cuda()
    for epoch in range(1, nb_epochs + 1):
        train_and_test(epoch, net, optimizer, activation)

activations = ['ReLU', 'Sigmoid', 'Tanh', 'ELU', 'LeakyReLU']
optimizers = ['Adam', 'SGD']
if __name__ == '__main__':
    for activation in activations:
        for optimizer in optimizers:
            print('Currently training with optimizer {} and activation {}'.format(optimizer,activation))
            main(optimizer, activation)
    
