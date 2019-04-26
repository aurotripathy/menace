"""
Attributions:
code from https://github.com/stefbraun/rnn_benchmarks (minor changes)
Arxiv paper https://arxiv.org/abs/1806.01818
Essentially, a single-layer LSTM with 
- 320 hidden units
- 100 time steps
- batch size 64
- 1D input feature size, 123
- output 10 classes

LSTMCell takes ONE input x_t at time t. 
    You need to loop in order to do one pass of backprop through time.

LSTM takes a SEQUENCE of inputs x_1,x_2,…,x_T. 
    NO need to write a loop to do one pass of backprop through time.

"""

import os
import time as timer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from support import toy_batch, default_hyperparams, print_results, check_results

def get_paramter_count():
    params = 0
    for param in list(net.parameters()):
        sizes = 1
        for el in param.size():
            sizes = sizes * el
            params += sizes
    return params
    

# Get data
bX, _, bY, classes = toy_batch()
batch_size, seq_len, inp_dims = bX.shape
hidden_units_size, learning_rate, batches = default_hyperparams()
print("Hidden units:{}, Learming Rate:{}, Batches:{}".format(hidden_units_size,
                                                             learning_rate, batches))

# PyTorch compatibility: time first, batch second
bX = np.transpose(bX, (1, 0, 2))

# Create Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTMCell(input_size=inp_dims, hidden_size=hidden_units_size, bias=True)
        self.fc = nn.Linear(hidden_units_size, classes, bias=False)

    def forward(self, x):
        max_len, batch_size, features = x.size()
        h_lstm = Variable(torch.zeros(batch_size, hidden_units_size)).cuda()
        c_lstm = Variable(torch.zeros(batch_size, hidden_units_size)).cuda()

        
        output = []
        for i in range(max_len):
            h_lstm, c_lstm = self.lstm(x[i], (h_lstm, c_lstm))
            output.append(h_lstm)
            
        h1 = torch.stack(output)
        h2 = h1[-1, :, :]
        h3 = self.fc(h2)
        return h3


net = Net()
net.cuda()

params = get_paramter_count()

# Create optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()  # loss definition

# Check for correct sizes
assert (net.fc.in_features == hidden_units_size)  # final projection input size (hidden_unit_size)
assert (net.fc.weight.cpu().data.numpy().shape == (
    classes, hidden_units_size))  # final projection output size (classes, hidden_units_size)
bXt = Variable(torch.from_numpy(bX).cuda())
torch.cuda.synchronize()
output = net(bXt)
output_numpy = output.data.cpu().numpy()
assert (output_numpy.shape == (batch_size, classes))

# Start training
batch_time = []
batch_loss = []
print("Starting the training benchmark with {} batches".format(batches))
train_start = timer.clock()
for _ in range(batches):
    torch.cuda.synchronize() # synchronize function call for precise time measurement
    batch_start = timer.clock()

    bXt = Variable(torch.from_numpy(bX).cuda())
    bYt = Variable(torch.from_numpy(bY).cuda())

    optimizer.zero_grad()
    output = net(bXt)
    loss = criterion(output, bYt.long())
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize() # synchronize function call for precise time measurement
    batch_end = timer.clock()
    batch_time.append(batch_end - batch_start)
    batch_loss.append(float(loss.data.cpu().numpy()))
train_end = timer.clock() # end of training

# Write results
print_results(batch_time)
check_results(batch_loss, batch_time, train_start, train_end)

