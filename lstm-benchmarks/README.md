#### How to run these simple LSTM benchmarks

<code>
Attributions:<br/>
code loosely based on https://github.com/stefbraun/rnn_benchmarks

Arxiv paper https://arxiv.org/abs/1806.01818

Description:

Benchmarking a single-layer LSTM with:

- 320 hidden units

- 100 time steps

- batch size 64

- 1D input features, size 125

- output 10 classes

LSTMCell takes ONE input x_t at time t.

    You need to loop time-steps in order to do one pass of 
    
    backprop through time.
    
LSTM takes a SEQUENCE of inputs x_1,x_2,…,x_T.

    NO need to loop time-steps to do one pass of backprop through time.
    
  </code>
  
