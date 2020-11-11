# Current Model Design Architecture

## Layer Class
<p align="center">
  <img width="1600" height="400" src="https://i.imgur.com/RG5dstB.png"><br>
  <b>Diagram showing how a layer interacts with the other layers in the Neural Netwok</b><br>
  <b>Next to the diagram we see the description of the equation: W<sup>T</sup>X<sub>Vector</sub> + B = Z<sub>Vector</sub></b>
</p>

### Attributes:
1. **Weights matrix:**
  - The dimensions of the weight matrix are:
    - **Number of rows** = Number of neurons in the previous layer. For the first layer this would be the number of features in the dataset, which is fed into the model (for us this would be 3 as we are only using 3 features from the dataset: **Weight**, **HB**, **BP**)
    - **Number of cols** = Number of neurons in the current layer
2. **Bias matrix:**
  - The dimensions of the bias matrix are:
    - **Number of rows** = Number of neurons in the current layer
    - **Number of cols** = 1
### Methods:
1. **init:** 
  - Create the W and B matrices with the [GlorotNormal initialization technique](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) (*this is the link to the original paper. Not all of it is important to use, we mainly need to focus on pages 251 and 253*) with seed=42
  - [Tensorflow Documentation's description of GlorotNormal](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotNormal)
  - [Resouce with code describing how GlorotNormal initilization can be implemented](https://visualstudiomagazine.com/articles/2019/09/05/neural-network-glorot.aspx)
  - [Condensed Resouce containing the formulas for the GlorotNormal Initliaization and its variants *(the variations mainly have different constant factors)*](https://mmuratarat.github.io/2019-02-25/xavier-glorot-he-weight-init)
  - [Math only resource, which is for confirming the formula used, *(skip the derivations if needed, only focus on the **final formula and variables' description**)*](https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404)
1. **Set Function**:
    1. Sets the the values of the weight matrix and bias vector of the layer, to the values which are input to this function
    1. Should return the old values of the weight matrix and bias vector, before updation
1. **Get Function**
    1. Returns the values of the weight matrix and bias vector of the layer
1. **Forward function**:
    1. Take the output of the prev. layer as input, this is called as ***X<sub>Vector</sub>** vector in the image above*
    1. Perform the matrix vector calculation: **W<sup>T</sup>X<sub>Vector</sub> + B** to produce the *Z vector*
    1. Randomly convert elements of the Z vector to 0 with a probability set in the variable ***rate*** (for our implementation, ***rate*** is set to 0.1) 
        - This should be done to implement dropouts in the neural network
        - [Code resource for Implementation of Dropouts](https://wiseodd.github.io/techblog/2016/06/25/dropout/)
            * [Resouce describing the relationship between batch and epoch as well as when model updations occur](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
    - Dropouts needs to be an **optional** parameter as the final layer does not have any dropouts
    - Inputs not set to 0 are scaled up by **1/(1 - rate)** such that the sum over all inputs is unchanged.
      * [Description of dropouts as per Keras](https://keras.io/api/layers/regularization_layers/dropout/)
      1. Apply the activation function to the elements in the modified *z vector* (i.e. after dropouts)
    - The Activation step can be implemented as using *decorator function technique* (The decorator function will be one of the activation funtions described below, depending on the layer)
    - The Activation takes the output from the Forward Function it decorates and applies the activation to it
        - This will help in ease of coding as well as code modularity
1. **Backward function**: Take the loss from the next layer, calculate the adjustment to the neurons in the current layer and then pass the current neurons' loss to the prev. layer
    1. [How dropouts work with forward and backward propagation](https://stats.stackexchange.com/questions/219236/dropout-forward-prop-vs-back-prop-in-machine-learning-neural-network)
1. **ReLU function (Class function)**: This will apply the ReLU activation function to the **Z<sub>Vector</sub>** passed to it
    1. [Great resource for learning about ReLU and its Implementation](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)
1. **SoftMax function (Class Function)**: This will apply the SoftMax activation function to its **Z<sub>Vector</sub>**
    - This function is only applied to the last layer of the Neural Network which has 2 neurons in our case
    - [Great resource for learning about Softmax and its Implementation](https://machinelearningmastery.com/softmax-activation-function-with-python/)
    - [This resourcer does go a bit mathematical but there is a numpy implementation description here](https://www.python-course.eu/softmax.php)
    - [This resource has no code, and is only meant for *verification of formulas* and the *meaning of the variables* in them as well as for looking at example of *how the formula works for testing*](https://deepai.org/machine-learning-glossary-and-terms/softmax-layer)
    - So first the sum of all the elements in the **Z<sub>Vector</sub>** should be taken and then the formula can be applied to each element in the layer. 
    - **Note** that the sum of the **W<sup>T</sup>X<sub>Vector</sub> + B** value is needed for calculating the denominator of the softmax function
    - Quick references for the softmax function formula and usage:
    <p align="center">
      <img width="750" height="300" src="https://i.imgur.com/lZKb266.png"><br>
      <img width="750" height="300" src="https://i.imgur.com/3bYFrju.jpg"><br>
      <b>Diagram showing the softmax function formula and usage</b>
    </p>

## NN Class
### Attributes:
  - List of layer objects
  - ```MAX_PATIENCE_VAL```: This is the value for the maximum patience value. This is a constant and should be treated as such
  - ```BEST_LAYER_CONFIG```: This holds a list of the weights and biases matrices of each layer of the best performing model during training
    * For us this value is ```100```
### Methods:
1. **init:** 
    1. This will create the list of layer objects
    1. The layer objects will be initialized here along setting with the activation function for each one
        
1. **Fit Function**
    1. The output of a layer will be passed to the **Forward function** of the next layer as we iterate thought the list of layers
        - The output of the final layer along with the true values will both be passed to the *loss function* and the *accuracy functions*
            - They will be used for backpropagation and displaying progress of training to the user
        - The input to the first layer will be **X<sub>Matrix</sub>** created from the rows of the entire dataset, as we are performing Batch Gradient Descent, *(which is always done over the entire dataset)*
        - [RECOMMENDATION] This complete forward propagation can be implemented as a loop over the list of layer objects, with a common variable possibly called ```carry_forward``` which is initialised to the **X<sub>Matrix</sub>** created from the rows of the entire dataset:
            * This variable is fed into the layer as input
            * The corresponding ouput from the layer is assigned to back to this variable
            * This happens in every iteration 
            * **Note:** In **X<sub>Matrix</sub>**, each row of out input dataset is a column vector called **X<sub>Vector</sub>**
    1. Batches of input data are fed into the model as a matrix called **X<sub>Matrix</sub>**
        1. This matrix **X<sub>Matrix</sub>** is got by stacking column vectors **X<sub>Vector</sub>** one next to the other
        1. The column vector **X<sub>Vector</sub>** contains each input row from the dataset as a column vector
        1. The dimensions of the **X<sub>Matrix</sub>** are:
            - Number of rows = Number of features in the input dataset
            - Number of cols = Number of input rows to use per batch (*for us the batch size is no.of data-objects/rows in the dataset*)
            - *Simple example:* The output of the model (i.e. from the SoftMax Layer) will have *2* Rows and the number of columns present will be equal to the *no.of data-objects/rows in the dataset*, where each column vector **Z<sub>Vector</sub>** in the output **Z<sub>Matrix</sub>** is the output for the corresponding input row from the dataset
        1. As the batch size is set to the total number of examples in the training dataset, we are performing a technique which is called as Batch Gradient Descent
            1. [The steps for selecting a batch of a given size from the data set](https://stackoverflow.com/questions/13693966/neural-net-selecting-data-for-each-mini-batch)
            1. [The different batching techniques and describing what Batch Gradient Descent is](https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/)
            1. [This resource describes how Batch Gradient Descent is implemented in words](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)
            1. As batches of input data is fed into the model, the output of the model as well as each layer of it will be **Z<sub>Matrix</sub>** which is composed of multiple column vectors, each called **Z<sub>Vector</sub>** stacked once next to the other.
                - Each **Z<sub>Vector</sub>** is basically the output vector of the layer for a particular input **X<sub>Vector</sub>** which comes from **X<sub>Matrix</sub>** 
                - **X<sub>Matrix</sub>** is either the previous layer ouput or the input to the model itself *(if we are talking about the first layer of the model)*
                - *Simple example:* The output of the first layer (i.e. the 8 Neuron Layer in our model) will have *8* Rows (1 for each neuron) and the number of columns present will be equal to the *no.of data-objects/rows in the dataset*, where each column of **Z<sub>Matrix</sub>** is the output for the corresponding **X<sub>Vector</sub>** in the input **X<sub>Matrix</sub>**
                - In this case the dimensions of the **Z<sub>Matrix</sub>** are:
                    * Number of rows = Number of neurons in the layer whose output is **Z<sub>Matrix</sub>**, the *Simple Example* above it is 8
                    * Number of cols = Number of data objects in the batch (in our case it is the number of rows in the entire dataset itself)
    1. **Early Stopping**
        1. The **Get Function** and **Set Function** would be especially useful here
        1. This needs to be checked after every epoch
        1. Initialize the value of the counter to ```MAX_PATIENCE_VAL```
        1. Calculate the loss value of the model on the *validation set* by passing it through forward propagation
            - No backward propagation needs to be done for early stopping
        1. Compare the new loss value obtained to the lowest loss value recorded so far 
            - Initially the *lowest loss value recorded can be* ```float('inf')```
        1. If the new loss value obtained is lesser than the lowest loss value recorded so far, by a margin greater than the set threshold of ```0.001``` then
            - Reset the counter to the ```MAX_PATIENCE_VAL```
            - Save the weight and bias configuration of each layer of the current model, to the variable ```BEST_LAYER_CONFIG``` *(The previous contents of this variable can be overwritten)*
                - We'll need the variable ```BEST_LAYER_CONFIG``` in the end to extract the best model
        1. Else
            - Decrement the value of the counter variable
            - Check if the value of the counter variable is 0
                - If so then stop training immediately and reset the weights and biases matricesof the model to the best ones obtained so far
                - These best values of the weights and biases matrices would be stored in the variable ```BEST_LAYER_CONFIG```
            - Else
                - Continue another epoch of training
        1. Store the complete list of the weight-bias matrices form each layer, of the least loss NN model so far
    1. **Loss Function**
        1. This will be called on the train, validation and test sets
        1. It will take the *train set features values* (**Weight**, **HB**, **BP**) and the corresponding *true outputs* (**Result_0.0**, **Result_1.0**) as input to the function
        1. The loss function used in our model is the Binary crossentropy ( i.e. binary_crossentropy (pred_(train|valid|test), true_(train|valid|test)) )
        1. This loss function is essentially similar to entropy which we learnt from decision trees *(except we are now restricted to only 2 possible outputs, i.e. 1 and 0)*, so some code sharing is possible with the Decision Trees Assignment
        1. Piazza post: @75, which is all about Binary crossentropy, is also a very handy resource
        1. [This resource has the complete formula for Binary crossentropy](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/binary-crossentropy)
        1. [Code resource for implementing Binary crossentropy, but it is missing the averaging step which is present in the immediate previous resource](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy)
        1. [Code resource for implementation but with better integration with the forward and backward propagation](https://gombru.github.io/2018/05/23/cross_entropy_loss/)
            - This resource includes forward and backward propagation steps
            - You may have to scroll down a bit to get to the code
        1. [This resource is purely theoretical and is only for verification of the formula](https://www.machinecurve.com/index.php/2019/10/22/how-to-use-binary-categorical-crossentropy-with-keras/#binary-crossentropy-for-binary-classification)
    1. **Accuracy Function**
        1. This will be called on the train, validation and test sets
            1. It will take the *train set features values* (**Weight**, **HB**, **BP**) and the corresponding *true outputs* (**Result_0.0**, **Result_1.0**) as input to the function
        1. The binary accuracy function is essentially the accuracy which is calculated from a 2x2 confusion matrix
            - The only difference is that the output of our model will be **continuous probabilities** and **not discrete values** so we need a threshold to say whether a probability should be converted to the *discrete value* of 1 or a *discrete value* of 0
            - This very threshold is what makes *binary accuracy* work, the default threshold of *binary accuracy* is **0.5**, which we have used as well
        1. The most detailed resource for accuracy and confusion matrices alike is [the description on Wikipedia itself](https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers)
        1. [This is a very important resource for the implementation of Binary Accuracy](https://towardsdatascience.com/keras-accuracy-metrics-8572eb479ec7)
            - This resource actually talks about the threshold, which is needed when converting each output value of the model in the **Z<sub>Matrix</sub>** to a 1 or 0
                - By each output value of the model I mean each and every element in the **Z<sub>Matrix</sub>**
        1. [This resource is more Keras' implementation Related](https://neptune.ai/blog/keras-metrics)
    1. **SGD**
        1. [Code for Stochastic Gradient Descent](https://adventuresinmachinelearning.com/stochastic-gradient-descent/)
            1. This will be modified later for Adam, but this stub code is fine for now, for sanity checks to see if backprop. works in the first place or not
    1. **Monte Carlo Dropouts**
        - https://towardsdatascience.com/monte-carlo-dropout-7fd52f8b6571
            - This resouce uses numpy to implement the Monte Carlo dropout section of the model
            - This could be adapted for out model as well, because at the moment the dropouts in our model continue to happen even when testing
                - This is exactly the primary requirements of *Monte Carlo dropout* so implementing this should be easy

<p align="center">
        <img width="750" height="700" src="https://i.imgur.com/QCdLJcX.jpg"><br>
    <b>The X<sub>Matrix</sub> Matrix which is made up of many X<sub>Vector</sub></b>
    </p>

## FAQs
1. How exactly **X<sub>Matrix</sub>** look and how does it relate to **X<sub>Vector</sub>**?
    - The following image should clear it up:
![The X<sub>Matrix</sub> Matrix which is made up of many X<sub>Vector</sub>](https://i.imgur.com/QCdLJcX.jpg)
    - In the **X<sub>Matrix</sub>** matrix above, each and every column is a **X<sub>Vector</sub>**
    - An **X<sub>Vector</sub>** is essentially a single row of from the input dataset
        - Here by ***row*** I mean the row from the input dataset ***with only all the input features (or columns) from the row*** and no output or target features of the row
        - In our case there are 3 input features (or columns) per row, in our dataset: **Weight**, **HB** and **BP**
        - This means that the **number of rows** in **X<sub>Matrix</sub>** is equal to 3 + 1
        - **NOTE:** The **+ 1** is done so that we can combine the *weights matrix* and *bias vector* of the current layer together into only a single weigths matrix *(by considering the bias as a weight with input 1)*
    - The number of columns in **X<sub>Matrix</sub>** is equal to the number of rows from the input dataset which is taken for one batch of forward propagation
        - In our case as we feed in all the rows in the input dataset as a single batch, the **number of columns* in **X<sub>Matrix</sub>** is equal to the number of rows in the input dataset
    - ***Notice*** that the last row of the **X<sub>Matrix</sub>** is all ***1's***, this is done so that we can combine the *weights matrix* and *bias vector* of the current layer together into only a single weigths matrix *(by considering the bias as a weight with input 1)*

1. How does the Weights matrix look when the bias terms are also considered as a part of the Weights matrix?
    - The following image of **W<sup>T</sup><sub>Matrix</sub>** should clear it up:
![The diagram and **W<sup>T</sup><sub>Matrix</sub>** for the first layer of the Neural Network](https://i.imgur.com/JI6y8TH.jpg)
    - In the **W<sup>T</sup><sub>Matrix</sub>** matrix above, every row contains the weights and the bias of a particular neuron
        - For example in the first row, we can see the weights and biases of the first neuron in the 8 neuron layer
    - The number of rows in **W<sup>T</sup><sub>Matrix</sub>** is equal to the number of neurons in the current layer
    - The number of columns in **W<sup>T</sup><sub>Matrix</sub>** is equal to: The *number of rows in **X<sub>Matrix</sub>***
    - Always remember that there is a bias term, which as you can see above is at the last column of each row of the **W<sup>T</sup><sub>Matrix</sub>**

## Note
**TBD**: *To Be Decided*, which means we can proceed and add items to this *while coding* or *after discussion*. The spec. sheet has not yet defined any requirements on this topic.

## Links:
### Real Gold Resources:
1. https://github.com/wiseodd/hipsternet
    - This resource has literally has all the code needed to get kickstarted!
2. https://www.jeremyjordan.me/intro-to-neural-networks/
    - Very good visual, matrix-wise description of Neural Networks
3. https://towardsdatascience.com/under-the-hood-of-neural-network-forward-propagation-the-dreaded-matrix-multiplication-a5360b33426
    - Theoretical **(i.e. only words, no code)** but very thorough resource on forward propagation
    - This resource should be helpful in understanding how to interpret the matrices and how they evolve (or change) during the process of forward propagation
4. https://github.com/jrios6/Adam-vs-SGD-Numpy/blob/master/Adam%20vs%20SGD%20-%20On%20Kaggle's%20Titanic%20Dataset.ipynb
    - Good explanation and complete code for forward and backward propagation using only NumPy, it also implements Adam optimizer with the derivatives portion as well.

Adam Optimizer:
https://mlfromscratch.com/optimizers-explained/
https://github.com/jiexunsee/Adam-Optimizer-from-scratch/blob/master/adamoptimizer.py


Neural Networks from Scratch: https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3
