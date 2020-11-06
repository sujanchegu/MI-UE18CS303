# Current Model Design Architecture

## Neurons class
### Attributes:
1. weights matrix
2. bias matrix
### Methods:
0. **init**: [GlorotNormal](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) init with seed=42
1. **Eval funtion (forward prop)**:
  a. Call the random function and check if the output is in the range [0, 10)
    - If so then output 0
  b. If it is in the range (10, 100]
    - Then return (w.x + b)
2. **Backward function**:
  - Math equations for backprop are needed here
  - Update the weights and bias -> Loss derivative
  - Return the loss values to pass to the prev. layer

## Layer Class
### Attributes:
- TBD
### Methods:
0. **init:** Create a list of neurons with the correct configuration of activation function
1. **Forward function**:
  a. Take the output of the prev. layer as input
  b. Randomly select neurons in the layer for forward propagation with a probability value of 90% (to implement dropouts)
    - Save the list/index of selected neurons so that only the required neurons are updated in backprop.
    - This will reduce the number of neurons we invoke in each layer as if we dropping the result of 0.1 of the neurons then might as well not call them for processing
    - This parameter has to **optional** as the final layer does not have any dropouts
  c. Pass the input to every neuron in the current layer
  d. Collect the outputs from the neurons in the current layer
  e. Apply the activation function to the outputs for each of the neurons in the layer
    - Activation will be implement as a callback function to make coding easier and increase modularity of the code
2. **Backward function**: Take the loss from the next layer, pass it to the neurons in the current layer and then pass the current neurons loss to the prev layer
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
