import numpy as np
import sys


class Layer:
    def __init__(self, _numInputs, _numNeurons, _activFunc):
        # Note, numInputs = number of neurons in the previous layer
        self.activFuncName = _activFunc
        self.prevShape = _numInputs + 1  # For bias
        self.shape = _numNeurons
        self.droprate = 0.1  # The proportion of neurons to drop at a layer
        self.seed = np.random.RandomState(42)
        sd = np.sqrt(6.0 / (self.prevShape + self.shape))
        self.weights = np.random.uniform(-sd, sd, (self.prevShape, self.shape))

        # Setting the bias values to 0 in the weights matrix
        for i in range(self.shape):
            self.weights[-1][i] = 0

    @classmethod
    def ReLU(cls, inputs):
        matrix = np.transpose(inputs)
        return np.transpose(np.array([
                                      [max(0.0, x) for x in matrix[j]]
                                      for j in range(len(matrix))
                                      ])
                            )

    @classmethod
    def ReLU_Prime(cls, inputs):
        matrix = np.transpose(inputs)
        return np.transpose(np.array([
                                      [1 if i > 0 else 0 for i in matrix[j]]
                                      for j in range(len(matrix))
                                      ])
                            )

    @classmethod
    def softmax(cls, inputs):
        # After this transpose, every output for a particular
        # input row to the model will be place row wise
        matrix = np.transpose(inputs)
        ret = []
        for row in matrix:
            exps = [np.exp(x) for x in row]
            sumexps = sum(exps)
            ret.append(np.array([exps[i] / sumexps for i in range(len(exps))]))

        # This step of transpose is needed so the every output for
        # a particular input row to the model, is now column-wise
        return np.transpose(ret)

    @classmethod
    def softmax_Prime(cls, inputs):
        '''
            d(S(Zi))/dZj = derivatives[i][j]
        '''
        matrix = np.transpose(inputs)
        ret = []
        for row in matrix:
            exps = [np.exp(x) for x in row]
            derivatives = [[exps[i]*(1-exps[i]) if i==j else exps[i] * -1 * exps[j] for j in range(len(row))] for i in range(len(row))]
            ret.append(np.array(derivatives))
        return np.transpose(ret)

    def set_params(self, _weights):  # , _biases):
        temp_weights = self.weights
        self.weights = _weights
        return temp_weights

    def get_params(self):
        return self.weights

    def drop(self):
        return np.random.binomial(1, 1 - self.droprate, size=self.shape)

    def forward(self, _input, _train=True):
        # Here we perform the matrix multiplication of W^T * X
        self.output = np.dot(self.weights.T, _input)
        if _train:
            self.activeNeurons = self.drop()
            # Performs elements wise multiplication
            self.output *= self.activeNeurons

            # Divide the output matrix by the fraction of outputs
            # kept and not dropped
            # We perform elements wise division here
            self.output /= (np.count_nonzero(self.activeNeurons) /
                            np.size(self.activeNeurons))
        return self.activationFunc(self.activFuncName, self.output)

    def activationFunc(self, _activFuncName, inputs):
        if(_activFuncName == 'ReLU'):
            return self.ReLU(inputs)
        elif(_activFuncName == 'softmax'):
            return self.softmax(inputs)
        else:
            # Exit the program on failure
            print("Wrong Activation Function Name!")
            sys.exit(1)

    def backward(self, _currentLayerDelta, _prevLayerOutputs):
        """
            Computes delta values for previous layer
        """
        # If the previous layer is a hidden layer
        prevlayer = [0 for i in range(_prevLayerOutputs.size)]
        for i in range(_prevLayerOutputs.size):
            if (_prevLayerOutputs[i] > 0):
                prevlayer[i] = 1
            else:
                prevlayer[i] = 0
        prevlayer = np.array(prevlayer)
        a = np.dot(self.weights, _currentLayerDelta)
        print(prevlayer)
        print(a)
        return np.multiply(a, prevlayer)


class NeuralNet:
    def __init__(self):
        self.hL1 = Layer(3, 8, 'ReLU')
        self.hL2 = Layer(8, 6, 'ReLU')
        self.outL = Layer(6, 2, 'softmax')
        self.layers = [self.hL1, self.hL2, self.outL]

    def fit(self, inputs, _train, _numEpochs, truthValues):
        for i in range(_numEpochs):
            output = inputs[i]
            for layer in self.layers:
                output = layer.forward(output, _train)
            epoch_loss = self.loss(output, truthValues)
            epoch_accuracy = self.accuracy(output, truthValues)
            print(f"> Epoch: {i} --> Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")

    @classmethod
    def loss(self, yHat, y):
        # We take the transpose so that the output for
        # each input row from the dataset is now row-wise in yHat
        yHatT = np.transpose(yHat)
        # yT = np.transpose(y)
        ret = []
        for rowYHAT, rowY in zip(yHatT, y):
            ret.append(
                        -1 * sum(
                                 np.array(
                                          [
                                           np.log(rowYHAT[i])
                                           if rowY[i] == 1
                                           else np.log(1 - rowYHAT[i])
                                           for i in range(len(rowYHAT))
                                           ]
                                          )
                                )
                        )
        return np.array(ret)

    def accuracy(self, yHat, y):
        yHatT = np.transpose(yHat)
        yT = np.transpose(y)
        ret = []
        for rowYHAT, rowY in zip(yHatT, yT):
            trueval = 0
            print(rowYHAT, rowY)
            for i in range(len(rowYHAT)):
                if((rowYHAT[i] >= 0.5 and rowY[i] == 1) or (rowYHAT[i] <=0.5 and rowY[i] == 0)):
                    trueval += 1
            ret.append(trueval/len(rowYHAT))
        return np.array(ret)
        


l1 = Layer(3, 5, 'ReLU')
print(l1.weights)

