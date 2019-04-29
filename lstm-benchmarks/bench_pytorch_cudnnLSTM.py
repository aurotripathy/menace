# coding: utf-8
"""
Attributions:
code loosely based on https://github.com/stefbraun/rnn_benchmarks
Arxiv paper https://arxiv.org/abs/1806.01818
Description:
Benchmarking a single-layer LSTM with:
- 320 hidden units
- 100 time steps
- batch size 64
- 1D input feature size, 125
- output 10 classes

LSTMCell takes ONE input x_t at time t.
    You need to loop in order to do one pass of backprop through time.

LSTM takes a SEQUENCE of inputs x_1,x_2,…,x_T.
    NO need to write a loop to do one pass of backprop through time.
"""

import time as timer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import get_batch, set_hyperparams, print_results

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_units_size, num_layers=1,
                            bias=True, bidirectional=False)
        self.fully_connect = nn.Linear(hidden_units_size, classes, bias=False)

    def forward(self, x):
        h1, state = self.lstm(x)
        h2 = h1[-1, :, :]
        h3 = self.fully_connect(h2)
        return h3


def validate_lstm_in_out():
    """ Detect any dimension mismatch issues"""
    assert net.fully_connect.in_features == hidden_units_size
    assert (net.fully_connect.weight.cpu().data.numpy().shape == (
        classes, hidden_units_size))  # final projection output size (classes, hidden_units_size)
    t_b_X = Variable(torch.from_numpy(time_first_batch_of_X).cuda())
    torch.cuda.synchronize()
    output = net(t_b_X)
    output_numpy = output.data.cpu().numpy()
    assert output_numpy.shape == (batch_size, classes)


def train_lstm():
    """ train for a nb_batches """
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # loss definition

    batch_time = []
    batch_loss = []
    print("Starting the training benchmark for {} batches".format(nb_batches))
    train_start = timer.clock()
    for _ in range(nb_batches):
        torch.cuda.synchronize() # synchronize function call for precise time measurement
        batch_start = timer.clock()

        t_b_X = Variable(torch.from_numpy(time_first_batch_of_X).cuda())
        b_y = Variable(torch.from_numpy(batch_of_y).cuda())

        optimizer.zero_grad()
        output = net(t_b_X)
        loss = criterion(output, b_y.long())
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize() # synchronize function call for precise time measurement
        batch_time.append(timer.clock() - batch_start)
        batch_loss.append(float(loss.data.cpu().numpy()))
    train_end = timer.clock()
    print_results(batch_loss, batch_time, train_start, train_end)

if __name__ == '__main__':

    hidden_units_size, learning_rate, seq_len, batch_size, nb_batches = set_hyperparams()
    classes = 10
    in_dim = 125
    time_first_batch_of_X, batch_of_y = get_batch(shape=(seq_len, batch_size, in_dim), classes=classes)
    print("Hidden units:{}, Learning Rate:{}, LSTM time steps:{}, Batch size:{}, Batches:{}".format(hidden_units_size,
                                                                                                   learning_rate, seq_len,
                                                                                                   batch_size, nb_batches))

    net = Net()
    net.cuda()
    validate_lstm_in_out()
    print('Trainable params', sum(p.numel() for p in net.parameters() if p.requires_grad))
    train_lstm()
