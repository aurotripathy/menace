import torch, torchvision
from visdom import Visdom
from torch import nn, optim
from torch.autograd import Variable
import numpy as np

# Part 1 write the data loaders
T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnis_data = torchvision.datasets.MNIST('mnist_data', transform=T,
                                       download=True)
mnist_ladaloader = torch.utils.data.DataLoader(mnis_data, batch_size=128)

# Part 2 Write the neural net, a class with two functions
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.linear1 = nn.Linear(28*28, 100)
        self.linear2 = nn.Linear(100, 50)
        self.final_linear = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, images): # backward by autograd
        x = images.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.final_linear(x)
        return x

        
# Part 3 Write the training loop
model = MnistNet()
cec_loss = nn.CrossEntropyLoss()
params = model.parameters()
optimizer = optim.Adam(params=params, lr=0.001)

n_epochs = 3
n_interations = 0

vis = Visdom()
vis_window = vis.line(np.array([0]), np.array([0]))

for e in range(n_epochs):
    for i, (images, labels) in enumerate(mnist_ladaloader):
        images = Variable(images)
        labels = Variable(labels)
        output = model(images)

        model.zero_grad()
        loss = cec_loss(output, labels)
        loss.backward()

        optimizer.step()

        n_interations += 1

        vis.line([loss.item()],
                 np.array([n_interations]),
                 win=vis_window,
                 update='append')
        
# Note - before you start, do the following:
# 1. start the visdom server, python -m visdom.server &
# 2. go to the browser and type http://127.0.0.1:8097/ in the URL line

