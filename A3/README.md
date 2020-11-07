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
  - Create the W and B matrices with the [GlorotNormal](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) initialization technique with seed=42
  - [Tensorflow Documentation's description of GlorotNormal](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotNormal)
  - [Resouce with code describing how GlorotNormal initilization can be implemented](https://visualstudiomagazine.com/articles/2019/09/05/neural-network-glorot.aspx)
  - [Condensed Resouce containing the formulas for the GlorotNormal Initliaization and its variants which are just different by the presence few constant factors](https://mmuratarat.github.io/2019-02-25/xavier-glorot-he-weight-init)
  - [Math only resource, which is for confirming the formula used, *skip the derivations if needed, only focus on the final formula and variables' description*](https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404)
1. **Forward function**:
    1. Take the output of the prev. layer as input, this is called as ***X<sub>Vector</sub>** vector in the image above*
    1. Perform the matrix vector calculation: **W<sup>T</sup>X<sub>Vector</sub> + B** to produce the *Z vector*
    1. Randomly convert elements of the Z vector to 0 with a probability of the variable *rate* (for our implementation, rate is set to 0.1) at each step during training time, (to implement dropouts in our neural network)
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
1. **ReLU function (Class function)**: This will apply the ReLU activation function to the vector passed to it
    1. [Great resource for learning about ReLU and its Implementation](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)
1. **SoftMax function (Class Function)**: This will apply the SoftMax activation function to its input
    - This function is only applied to the last layer of the Neural Network which has 2 neurons
    - [Great resource for learning about Softmax and its Implementation](https://machinelearningmastery.com/softmax-activation-function-with-python/)
    - [This resourcer does go a bit mathematical but there is a numpy implementation description here](https://www.python-course.eu/softmax.php)
    - [This resource has no code, and is only meant for *verification of formulas* and the *meaning of the variables* in them as well as for looking at example of *how the formula works for testing*](https://deepai.org/machine-learning-glossary-and-terms/softmax-layer)
    - So first the sum of all the elements in the modified Z vector should be taken and then the formula can be applied to each element in the layer. Note again that the sum of the **W<sup>T</sup>X<sub>Vector</sub> + B** value is needed for calculating the denominator of the softmax function
    - Quick reference for the softmax function formula and usage:
    <p align="center">
      <img width="750" height="300" src="https://i.imgur.com/lZKb266.png"><br>
      <img width="750" height="300" src="https://i.imgur.com/3bYFrju.jpg"><br>
      <b>Diagram showing the softmax function formula and usage</b>
    </p>

## NN Class
### Attributes:
  - TBD
### Methods:
1. **init:** 
    1. This will contain a list of layer objects
    1. The layer objects will be initialized here along setting with the activation function for each one
        
1. **Fit Function**
    1. Batches of input data is fed in to the model as a matrix called **X<sub>Matrix</sub>**
        1. This matrix **X<sub>Matrix</sub>** is got by stacking column vectors **X<sub>Vector</sub>** one next to the other
        1. The column vector **X<sub>Vector</sub>** contains each input row from the dataset as a column vector
        1. The dimensions of the **X<sub>Matrix</sub>** are:
            - Number of rows = Number of features in the input dataset
            - Number of cols = Number of input rows to consider per batch (*for us the batch size is no.of data-objects/rows in the dataset*)
            - *Simple example:* The output of the model (i.e. from the SoftMax Layer) will have *2* Rows and *no.of data-objects/rows in the dataset* number of Cols, where each column is the output for the corresponding input row from the dataset
        1. As the batch size is set to the total number of examples in the training dataset, what we are performing is called as Batch Gradient Descent
            1. [The steps for selecting a batch of a given size from the data set](https://stackoverflow.com/questions/13693966/neural-net-selecting-data-for-each-mini-batch)
            1. [The different batching techniques and describing what Batch Gradient Descent is](https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/)
            1. [This resource describes how Batch Gradient Descent is implemented in words](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)
    1. **Early Stopping**
        1. Store the list of the weight-bias matrices
        1. Compare the loss values after every epoch
        1. Count timer sort of technique
        
    2. **SGD**
        1. [Code for Stochastic Gradient Descent](https://adventuresinmachinelearning.com/stochastic-gradient-descent/)
            1. This will be modified for Adam

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
