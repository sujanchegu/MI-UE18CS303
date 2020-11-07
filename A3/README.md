# Current Model Design Architecture

## Layer Class
### Attributes:
1. Weights matrix
2. Bias matrix
### Methods:
0. **init:** 
  - Create the weights and bias matrices with the [GlorotNormal](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) initialization with seed=42
1. **Forward function**:
  a. Take the output of the prev. layer as input
  b. Perform the matrix vector calculation: w<sup>T</sup>x + b to produce the z vector
  c. Randomly convert elements of the z vector to 0 with a probability of rate (for our implementation, rate is set to 0.1) at each step during training time, (to implement dropouts)
    - Save the list/index of neurons' whose output in the z-vector set to 0 due to dropout, so that only the required neurons are updated in backprop.
    - This parameter has to **optional** as the final layer does not have any dropouts
    - Inputs not set to 0 are scaled up by **1/(1 - rate)** such that the sum over all inputs is unchanged.
    * [Description of dropouts as per Keras](https://keras.io/api/layers/regularization_layers/dropout/)
  f. Apply the activation function to the elements in the *z vector* for each of the neurons in the layer
    - Activation will be implement as a callback function to make coding easier and increase modularity of the code
2. **Backward function**: Take the loss from the next layer, calculate the adjustment to the neurons in the current layer and then pass the current neurons' loss to the prev. layer
  - [How dropouts work with forward and backward propagation](https://stats.stackexchange.com/questions/219236/dropout-forward-prop-vs-back-prop-in-machine-learning-neural-network)
3. **ReLU function (Class function)**: This will apply the ReLU activation function to its input
4. **SoftMax function (Class Function)**: This will apply the SoftMax activation function to its input
  - This function needs the outputs of the other neurons in the layer before acitvation function output, i.e. wTx + b for the denominator term which is common to all neurons in the layer using SoftMax
  - So first the sum of the wTx + b values should be taken and then the formula can be applied to each wTx + b value in the layer. Note again that the sum of the wTx + b value is needed for the denominator of the softmax function

## NN Class
### Attributes:
  - TBD
### Methods:
  0. **init**: hard code neuros per layer, create the list of layers objects
  1. NN implement batches and 
  2. early stopping -> Store the list of the weight-bias matrices
  3. Compare the loss values after every epoch
  4. Count timer sort of technique

functon for binary_crossentropy (pred_y_train, true_y_train)

## Note
**TBD**: *To Be Decided*, which means we can proceed and add items to this *while coding* or *after discussion*. The spec. sheet has not yet defined any requirements on this topic.

## Links:
Adam Optimizer:
https://mlfromscratch.com/optimizers-explained/
https://github.com/jiexunsee/Adam-Optimizer-from-scratch/blob/master/adamoptimizer.py


Binary_crossentropy
Link for the same is in Piazza Post: @75:

Neural Networks from Scratch: https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3
