# Current Model Design Architecture

## Neurons class
### Attributes:
1. weights matrix
2. bias matrix
### Methods:
0. init: [GlorotNormal](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) init with seed=42
1. Eval funtion (forward prop): 
  a. Call the random function and check if the output is in the range [0, 10)
    - If so then output 0
  b. If it is in the range (10, 100]
    - Then return activation(w.x + b)
2. Backward function: math equations for; updtae the weights and bias -> Loss derivative

## Layer Class:
### Attributes:
- TBD
### Methods:
0. init: Create a list of neurons with the correct configuration of activation function
1. Forward function:
  a. Take the output of the prev. layer as input
  b. Pass the input to every neuron in the current layer
  c. Collect the outputs from the neurons in the current layer
  d. Randomly select a few neurons
2. Backward function: Take the loss from the next layer, pass it to the neurons in the current layer and then pass the current neurons loss to the prev layer

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
