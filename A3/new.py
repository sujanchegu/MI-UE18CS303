import numpy as np


class Layer:
    def __init__(self, _numInputs, _numNeurons, _activFunc):
        # Note, numInputs = number of neurons in the previous layer
        self.activFuncName = _activFunc
        self.prevShape = _numInputs + 1  # For bias
        self.shape = _numNeurons
        self.droprate = 0.1
        self.seed = np.random.RandomState(42)
        # self.biases = np.zeros((1, self.shape))
        sd = np.sqrt(6.0 / (self.prevShape + self.shape))
        self.weights = np.random.uniform(-sd,
                                         sd, (self.prevShape, self.shape))

    @classmethod
    def ReLU(cls, inputs):
        return np.array([max(0.0, x) for x in inputs])

    @classmethod
    def ReLU_Prime(cls, ):
        pass

    @classmethod
    def softmax(cls, inputs):
        exps = [np.exp(x) for x in inputs]
        sumexps = sum(exps)
        return np.array([exps[i]/sumexps for i in range(len(exps))])

    def set_params(self, _weights, _biases):
        temp_weights = self.weights
        # temp_biases = self.biases
        self.weights = _weights
        # self.biases = _biases
        # return (temp_weights, temp_biases)
        return temp_weights

    def get_params(self):
        # return (self.weights, self.biases)
        return self.weights

    def drop(self):
        return np.random.binomial(1, 1 - self.droprate, size=self.shape)

    def forward(self, _input, _train=False):
        self.output = np.dot(self.weights.T, _input)
        if _train:
            self.activeNeurons = drop()
            self.output *= self.activeNeurons
            # self.output *= np.random.binomial(1, self.droprate, size=self.shape) / (1 - self.droprate)
            self.output /= (1 - self.droprate)
        return self.activationFunc(self.activFuncName, self.output)

    def activationFunc(self, _activFuncName, inputs):
        if(_activFuncName == 'ReLU'):
            return self.ReLU(inputs)
        elif(_activFuncName == 'softmax'):
            return self.softmax(inputs)
        else:
            printf("Wrong Activation Function Name")
        

    # def backward(self, _nextLayerInputs):



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
