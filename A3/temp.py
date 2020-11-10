import numpy as np


class Layer:
    def __init__(self, _numInputs, _numNeurons):
        # Note, numInputs = number of neurons in the previous layer
        self.prevShape = _numInputs + 1 # For bias
        self.shape = _numNeurons
        self.droprate = 0.1
        self.seed = np.random.RandomState(42)
        # self.biases = np.zeros((1, self.shape))
        sd = np.sqrt(6.0 / (self.prevShape + self.shape))
        self.weights = np.random.uniform(-sd,
                                         sd, (self.prevShape, self.shape))
        

    def set_params(self, _weights):
        temp_weights = self.weights
        # temp_biases = self.biases
        self.weights = _weights
        # self.biases = _biases
        # return (temp_weights, temp_biases)
        return temp_weights

    def get_params(self):
        # return (self.weights, self.biases)
        return self.weights

    def forward(self, _input, _train=False):
        # Input should include 1 as the value of x0 at the end
        # Eg. x = <x1,x2,x0>
        self.output = np.dot(self.weights.T, _input)
        if _train:
            self.output *= np.random.binomial(1, self.droprate, size=self.shape) / self.droprate
        


# l1 = Layer(5, 6)
# print(l1.weights)
# print(l1.output)
