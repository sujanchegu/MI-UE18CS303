# Current Model Design Architecture

## Layer Class
<p align="center">
  <img width="1600" height="400" src="https://i.imgur.com/RG5dstB.png"><br>
  <b>Diagram showing how a layer interacts with the other layers in the Neural Netwok</b>
</p>

### Attributes:
1. **Weights matrix:**
  - The dimensions of the weight matrix are:
    - Number of rows = Number of neurons in the current layer
    - Number of cols = Number of neurons in the previous layer. For the first layer this would the number of features in the dataset fed into the model
2. **Bias matrix:**
  - The dimensions of the bias matrix are:
    - Number of rows = Number of neurons in the current layer
    - Number of cols = 1
### Methods:
1. **init:** 
  - Create the weights and bias matrices with the [GlorotNormal](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) initialization technique with seed=42
1. **Forward function**:
  1. Take the output of the prev. layer as input, this is called as ***X** vector in the image above*
  1. Perform the matrix vector calculation: **W<sup>T</sup>X + B** to produce the *Z vector*
  1. Randomly convert elements of the z vector to 0 with a probability of the variable *rate* (for our implementation, rate is set to 0.1) at each step during training time, (to implement dropouts in our neural network)
    - In every pass save the list/index of neurons' whose output in the z-vector set to 0 due to dropout
      * This may be important during the dropout stage for that forward propagation
      * This data is only needed per forward propagation
      * [Resouce describing the relationship between batch and epoch as well as when model updations occur](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/]
    - Dropouts needs to be an **optional** parameter as the final layer does not have any dropouts
    - Inputs not set to 0 are scaled up by **1/(1 - rate)** such that the sum over all inputs is unchanged.
      * [Description of dropouts as per Keras](https://keras.io/api/layers/regularization_layers/dropout/)
  1. Apply the activation function to the elements in the modified *z vector* (i.e. after dropouts)
    - Activation will be implement as a callback function for ease of coding as well as code modularity
1. **Backward function**: Take the loss from the next layer, calculate the adjustment to the neurons in the current layer and then pass the current neurons' loss to the prev. layer
    1. [How dropouts work with forward and backward propagation](https://stats.stackexchange.com/questions/219236/dropout-forward-prop-vs-back-prop-in-machine-learning-neural-network)
1. **ReLU function (Class function)**: This will apply the ReLU activation function to its input
1. **SoftMax function (Class Function)**: This will apply the SoftMax activation function to its input
  - This function needs the outputs of the other neurons in the layer before acitvation function output, i.e. wTx + b for the denominator term which is common to all neurons in the layer using SoftMax
  - So first the sum of the wTx + b values should be taken and then the formula can be applied to each wTx + b value in the layer. Note again that the sum of the wTx + b value is needed for the denominator of the softmax function

## NN Class
### Attributes:
  - TBD
### Methods:
  1. **init**: hard code neuros per layer, create the list of layers objects
  1. NN implement batches and 
  1. early stopping -> Store the list of the weight-bias matrices
  1. Compare the loss values after every epoch
  1. Count timer sort of technique

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
