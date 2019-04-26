import os
import time as timer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from support import toy_batch, default_hyperparams, write_results, print_results, check_results

def get_paramter_count():
    params = 0
    for param in list(net.parameters()):
        sizes = 1
        for el in param.size():
            sizes = sizes * el
            params += sizes
    return params
    

# Experiment_type
bench = 'pytorch_cudnnLSTM'
version = torch.__version__
experiment = '1x320-LSTM_cross-entropy'

# Get data
bX, _, bY, classes = toy_batch()
batch_size, seq_len, inp_dims = bX.shape
hidden_units_size, learning_rate, batches = default_hyperparams()
print("Hidden unit:{}, learming rate:{}, batches:{}".format(hidden_units_size, learning_rate, batches))

# PyTorch compatibility: time first, batch second
bX = np.transpose(bX, (1, 0, 2))

# Create Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=inp_dims, hidden_size=hidden_units_size, num_layers=1,
                            bias=True, bidirectional=False)
        self.fc = nn.Linear(hidden_units_size, classes, bias=False)

    def forward(self, x):
        h1, state = self.lstm(x)
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
for i in range(batches):
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
# write_results(script_name=os.path.basename(__file__), bench=bench, experiment=experiment, parameters=params,
#               run_time=batch_time, version=version, logfile="logfile")
