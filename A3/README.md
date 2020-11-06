# Current Model Design Architecture

## Neurons class
### Attributes:
1. weights matrix
2. bias matrix
### Methods:
0. init: [GlorotNormal](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) init with seed=42
1. Eval funtion : (forward prop): prop(0.9, activation(w.x + b), 0.1, 0)
2. backward function: math equations for; updtae the weights and bias -> Loss derivative

Layer Class:
init: List of neurons
forward: take the inputs from the prev layer as a matrix and pass it to every neuron
backward: take the loss from the next layer, pass it to the neurons in the current layer and then pass the current neurons loss to the prev layer

NN Class
init: hard code neuros per layer, create the list of layers objects
NN implement batches and 
early stopping -> Store the list of the weight-bias matrices
Compare the loss values after every epoch
Count timer sort of technique

functon for binary_crossentropy (pred_y_train, true_y_train)

Links:
-------
Adam Optimizer:
https://mlfromscratch.com/optimizers-explained/
https://github.com/jiexunsee/Adam-Optimizer-from-scratch/blob/master/adamoptimizer.py


Binary_crossentropy
Link for the same is in Piazza Post: @75:
