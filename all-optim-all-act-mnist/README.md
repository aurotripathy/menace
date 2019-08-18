Runs all combinations of specified activations with specified optimizers.
Network is Lenet-5 (like), dataset is mnist. See below for modifications.

List so far:

```python
    activation_strs = ['SoftExponential', 'Sigmoid', 'SiLU', 'GeLU', 'ReLU', 'Tanh', 'ELU', 'LeakyReLU']
    optimizer_strs = ['SGD', 'RMSprop', 'Adam']

```

List above is by no means complete but a start.

Note

The combination of SGD and Sigmoid would not converge without the use of batch normalization layers (after convolution and activation layers). So the newtwork ended up looking like this:

Convolutional
```python
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('activation1', activation_fn()),
            ('bn1', nn.BatchNorm2d(6)),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('activation3', activation_fn()),
            ('bn2', nn.BatchNorm2d(16)),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('activation5', activation_fn())
        ]))

```
Fully Connected
```python

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('activation6', activation_fn()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))
```
