Runs all combinations of specified activations with specified optimizers.
Network is Lenet5, dataset is mnist.

List so far:

```
activation_strs = ['ReLU', 'Sigmoid', 'Tanh', 'ELU', 'LeakyReLU']
optimizer_strs = ['RMSprop', 'SGD', 'Adam']
```

List above is by no means complete but a start.

Note

The combination would not converge without the use of batch normalization layers (after convolution and activation)

