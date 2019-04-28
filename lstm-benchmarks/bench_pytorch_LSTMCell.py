"""
Attributions:
code loosely based on https://github.com/stefbraun/rnn_benchmarks
Arxiv paper https://arxiv.org/abs/1806.01818
Description:
Benchmarking a single-layer LSTM with:
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

import time as timer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from support import get_batch, set_hyperparams, print_results, check_results


# Get data
hidden_units_size, learning_rate, seq_len, batch_size, batches = set_hyperparams()
classes = 10
in_dim = 125
bX, bY = get_batch(shape=(batch_size, seq_len, in_dim), classes=classes)
batch_size, seq_len, inp_dims = bX.shape
print("Hidden units:{}, Learning Rate:{}, Batches:{}".format(hidden_units_size,
                                                             learning_rate, batches))

# PyTorch compatibility: time first, batch second
bX = np.transpose(bX, (1, 0, 2))

class Net(nn.Module):
    """ Create the LSTM network """
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTMCell(input_size=inp_dims, hidden_size=hidden_units_size, bias=True)
        self.fully_connect = nn.Linear(hidden_units_size, classes, bias=False)

    def forward(self, x):
        sequence_len, batch_size, features = x.size()
        h_lstm = Variable(torch.zeros(batch_size, hidden_units_size)).cuda()
        c_lstm = Variable(torch.zeros(batch_size, hidden_units_size)).cuda()

        
        output = []
        for i in range(sequence_len):
            h_lstm, c_lstm = self.lstm(x[i], (h_lstm, c_lstm))
            output.append(h_lstm)

        h1 = torch.stack(output)
        h2 = h1[-1, :, :]
        h3 = self.fully_connect(h2)
        return h3

def validate_lstm_in_out():
    assert net.fully_connect.in_features == hidden_units_size
    assert (net.fully_connect.weight.cpu().data.numpy().shape == (
        classes, hidden_units_size))  # final projection output size (classes, hidden_units_size)
    bXt = Variable(torch.from_numpy(bX).cuda())
    torch.cuda.synchronize()
    output = net(bXt)
    output_numpy = output.data.cpu().numpy()
    assert output_numpy.shape == (batch_size, classes)


def train_lstm():
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # loss definition

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
        batch_time.append(timer.clock() - batch_start)
        batch_loss.append(float(loss.data.cpu().numpy()))
    train_end = timer.clock()

    # Write results
    print_results(batch_time)
    check_results(batch_loss, batch_time, train_start, train_end)

if __name__ == '__main__':
    net = Net()
    net.cuda()
    validate_lstm_in_out()
    print('Trainable params', sum(p.numel() for p in net.parameters() if p.requires_grad))
    train_lstm()
