import numpy as np


class Layer:
    def __init__(self, _numInputs, _numNeurons, _activFunc):
        # Note, numInputs = number of neurons in the previous layer
        self.activFuncName = _activFunc
        self.prevShape = _numInputs + 1  # For bias
        self.shape = _numNeurons
        self.droprate = 0.1
        self.seed = np.random.RandomState(42)
        sd = np.sqrt(6.0 / (self.prevShape + self.shape))
        self.weights = np.random.uniform(-sd, sd, (self.prevShape, self.shape))
        for i in range(self.shape):
            self.weights[-1][i] = 0

    @classmethod
    def ReLU(cls, inputs):
        return np.array([max(0.0, x) for x in inputs])

    @classmethod
    def ReLU_Prime(cls, inputs):
        return np.array([1 if i > 0 else 0 for i in inputs])

    @classmethod
    def softmax(cls, inputs):
        exps = [np.exp(x) for x in inputs]
        sumexps = sum(exps)
        return np.array([exps[i]/sumexps for i in range(len(exps))])

    @classmethod
    def softmax_Prime(cls, inputs):
        '''
            d(S(Zi))/dZj = derivatives[i][j]
        '''
        exps = [np.exp(x) for x in inputs]
        derivatives = [[exps[i]*(1-exps[i]) if i==j else exps[i] * -1 * exps[j] for j in range(len(inputs))] for i in range(len(inputs))]
        return np.array(derivatives)

    def set_params(self, _weights, _biases):
        temp_weights = self.weights
        self.weights = _weights
        return temp_weights

    def get_params(self):
        return self.weights

    def drop(self):
        return np.random.binomial(1, 1 - self.droprate, size=self.shape)

    def forward(self, _input, _train=False):
        self.output = np.dot(self.weights.T, _input)
        if _train:
            self.activeNeurons = drop()
            self.output *= self.activeNeurons
            self.output /= self.droprate
        return self.activationFunc(self.activFuncName, self.output)

    def activationFunc(self, _activFuncName, inputs):
        if(_activFuncName == 'ReLU'):
            return self.ReLU(inputs)
        elif(_activFuncName == 'softmax'):
            return self.softmax(inputs)
        else:
            printf("Wrong Activation Function Name")
        
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
                0

        prevlayer = np.array(prevlayer)
        a = np.dot(self.weights, _currentLayerDelta)
        print(prevlayer)
        print(a)
        return np.multiply(a, prevlayer)



class NeuralNet:
    def __init__(self):
        self.hL1 = Layer(4, 8)
        self.hL2 = Layer(8, 6)
        self.outL = Layer(6, 2)
        self.layers = [self.hL1, self.hL2, self.outL]

    def fit(self, inputs, _train, truthValues):
        output = truthValues
        for layer in self.layers:
            output = layer.forward(output, _train)
        self.loss(output, truthValues)
        self.accuracy(output, truthValues)

    def loss(self):
        pass

    def accuracy(self):
        pass


l1 = Layer(3, 5, 'ReLU')
print(l1.weights)




