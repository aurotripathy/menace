Runs all combinations of specified activations with specified optimizers.
Network is Lenet5, dataset is mnist.
```
activation_strs = ['ReLU', 'Sigmoid', 'Tanh', 'ELU', 'LeakyReLU']
optimizer_strs = ['RMSprop', 'SGD', 'Adam']
```

List above is by no means complete but a start.

The hyperparamters don't work for this case (optimizer SGD and activation Sigmoid).


```
...Training with optimizer, SGD and activation, Sigmoid
Epoch 1, Avg. Test Loss: 0.002301, Accuracy: 0.113500
Epoch 2, Avg. Test Loss: 0.002301, Accuracy: 0.113500
Epoch 3, Avg. Test Loss: 0.002301, Accuracy: 0.113500
Epoch 4, Avg. Test Loss: 0.002301, Accuracy: 0.113500
Epoch 5, Avg. Test Loss: 0.002301, Accuracy: 0.113500
Epoch 6, Avg. Test Loss: 0.002301, Accuracy: 0.113500
Epoch 7, Avg. Test Loss: 0.002301, Accuracy: 0.113500
Epoch 8, Avg. Test Loss: 0.002301, Accuracy: 0.113500
Epoch 9, Avg. Test Loss: 0.002302, Accuracy: 0.113500
Epoch 10, Avg. Test Loss: 0.002302, Accuracy: 0.113500
  ```


