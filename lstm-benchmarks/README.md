#### How to the LSTM benchmarks

python bench_pytorch_LSTMCell.py  # to run the nn.LSTMCell based benchmark

python bench_pytorch_cudnnLSTM.py  # to run the nn.LSTM based benchmark

<code>
Attributions:<br/>
    
Code loosely based on https://github.com/stefbraun/rnn_benchmarks<br/>

Arxiv paper, https://arxiv.org/abs/1806.01818<br/>

Description:<br/>
Benchmarking a single-layer LSTM with:<br/>
- 320 hidden units<br/>
- 100 time steps<br/>
- batch size 64<br/>
- 1D input features, size 125<br/>
- output 10 classes<br/>

##### LSTMCell takes ONE input x_t at time t.<br/>

You need to loop time-steps in order to do one pass of backprop through time.
    
##### LSTM takes a SEQUENCE of inputs x_1,x_2,…,x_T.

NO need to loop time-steps to do one pass of backprop through time.
    
  </code>
  
