import numpy as np



class Layer:
    def __init__(self, _numInputs, _numNeurons):
        self.prevShape = _numInputs
        self.shape = _numNeurons
        self.droprate = 0.1
        self.seed = np.random.RandomState(42)
        self.biases = np.zeros((1, self.shape))
        sd = np.sqrt(6.0 / (self.prevShape + self.shape))
        self.weights = np.random.uniform(-sd, sd, (self.prevShape, self.shape))

    @classmethod
    def ReLU(cls, _nextInputs):
        return np.array([max(0.0,x) for x in _nextInputs])

    @classmethod
    def softmax(cls, _nextInputs):
        exps = [np.exp(x) for x in _nextInputs]
        sumexps = sum(exps)
        return np.array([exps[i]/sumexps for i in range(len(exps))])

    def set_params(self, _weights, _biases):
        temp_weights = self.weights
        temp_biases = self.biases
        self.weights = _weights
        self.biases = _biases
        return (temp_weights, temp_biases)

    def get_params(self):
        return (self.weights, self.biases)

    def forward(self, _input, _train=False):
        self.output = np.dot(self.weights.T, _input) + self.biases
        if _train:
            self.output *= np.random.binomial(1, self.droprate, size=self.shape) / (1 - self.droprate)
        #TODO return activationFunc(self.output)

    # def backward(self, _nextLayerInputs):
        

    


    


    # def _decorator(func):
    #     def wrapper(*args, **kwargs):
    #         return func(args[0])
    #     return wrapper
    
    # @_decorator
    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))


class NueralNet:
    def __init__(self):
        self.hL1 = Layer(4,8)
        self.hL2 = Layer(8,6)
        self.outL = Layer(6,2)
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
