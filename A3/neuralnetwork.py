# -*- coding: utf-8 -*-
"""NeuralNetwork.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MBod6QEwuUU4McbyAahCaYDWjkP7SA-A
"""

import numpy as np
from numpy.random import default_rng
import pandas as pd
import sys
import random
import copy

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

        """ Returns the shape of the weights matrix in that layer (object)
        """
    def shape(self):
        return self.weights.shape

    @classmethod
    def ReLU(cls, inputs):
        return np.maximum(inputs, np.zeros(inputs.shape))

        # matrix = np.transpose(inputs)
        # return np.transpose(np.array([
        #                               [max(0.0, x) for x in matrix[j]]
        #                               for j in range(len(matrix))
        #                               ])
        #                     )

    @classmethod
    def ReLU_Prime(cls, inputs):
        assert inputs.shape[1] == 65, \
            "The 65 rows of the dataset should be along the \
                columns of the input matrix"
        return np.maximum(inputs, np.zeros(inputs.shape))
        """
        We take the transpose so that the output for a particular input row
        from the dataset is now along the rows of the matrix
        """
        # matrix = np.transpose(inputs)

        """
        We take the transpose so that the output for a particular
        input row from the dataset is now along the cols of the matrix
        """
        # return np.transpose(np.array([
        #                               [1 if i > 0 else 0 for i in matrix[j]]
        #                               for j in range(len(matrix))
        #                               ])
        #                     )

    @classmethod
    def softmax(cls, inputs):
        assert inputs.shape[1] == 65, "The 65 rows of the dataset should be along the columns of the input matrix"
        # After this transpose, every output for a particular
        # input row from the dataset, will be placed row wise
        matrix = np.transpose(inputs)
        ret = []

        res = (matrix.T - np.max(matrix, axis=1)).T
        res = np.exp(res)

        ans = res.T / np.sum(res, axis=1)
        assert ans.shape == (2, 65), f"The dimensions of {ans} are: {ans.shape}"
        # This step of transpose is needed so that every output for
        # a particular input row to the model, is now column-wise
        return ans

        # The code below has been decommisioned
        # for row in matrix:
        #     exps = [np.exp(x) for x in row]
        #     sumexps = sum(exps)
        #     ret.append(np.array([exps[i] / sumexps for i in range(len(exps))]))

        # ret = np.array(ret)
        # assert type(ret) == np.ndarray

        # # This step of transpose is needed so that every output for
        # # a particular input row to the model, is now column-wise
        # return np.transpose(ret)

    @classmethod
    def softmax_Prime(cls, inputs):
        assert inputs.shape[1] == 65, \
            "The 65 rows of the dataset should be \
                along the columns of the input matrix"
        '''
            d(S(Zi))/dZj = derivatives[i][j]
        '''
        # After this transpose, every output for a particular
        # input row from the dataset, will be placed row wise
        matrix = np.transpose(inputs)
        ret = []
        for row in matrix:
            exps = np.exp(row)  # [np.exp(x) for x in row]
            sum_exps = np.sum(exps)
            derivatives = [[(exps[i]/sum_exps) * (1 - (exps[i]/sum_exps))
                            if i == j else (exps[i]/sum_exps) *
                            -1 * (exps[j]/sum_exps)
                            for j in range(len(row))] for i in range(len(row))]
            ret.append(np.array(derivatives))
        assert type(ret) == list
        return np.transpose(np.array(ret))

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
            temp1 = self.activeNeurons.copy()
            temp = self.output.shape[1]
            for i in range(temp - 1):
                self.activeNeurons = np.vstack((self.activeNeurons, temp1))
            self.activeNeurons = self.activeNeurons.transpose()
            # Performs elements wise multiplication
            self.output *= self.activeNeurons

            # Divide the output matrix by the fraction of outputs
            # kept and not dropped
            # We perform elements wise division here
            self.output = self.output/(np.count_nonzero(self.activeNeurons) /
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

# class BackPropagation:
#     @classmethod
#     def outputLayerBackpropGradient(cls, yHat, y, layer_input, W_matrix):
#         # We take the transpose so that the output for
#         # each input row from the dataset is now row-wise in yHat
#         yHatT = np.transpose(yHat)
#         # yT = np.transpose(y)
#         ret = []
#         # The dimensions must be the same
#         assert yHatT.shape == y.shape, f"Shape of yHatT = {yHatT.shape}\nShape of y = {y.shape}"

#         W_matrix_T = np.transpose(W_matrix)
#         assert W_matrix_T.shape == (2, 7), f"The actual shape of W_matrix_T is {W_matrix_T.shape}"

#         layer_input_T = np.transpose(layer_input)
#         assert layer_input_T.shape[0] == 65, f"The actual shape of layer_input_T is {layer_input_T.shape}"
        
#         weight_gradient_per_output = []

#         rowCount, colCount = W_matrix_T.shape
#         for truValInd, truVal in enumerate(y):
#             weight_gradient = []
#             for rowInd in range(rowCount):
#                 weight_gradient_row = []
#                 for colInd in range(colCount):
#                     ans = (yHatT[truValInd][rowInd] - (1 if np.argmax(truVal) == rowInd else 0)) * layer_input_T[truValInd][rowInd]
#                     ans = ans * (1 / rowCount)
#                     assert type(ans) != np.ndarray
#                     weight_gradient_row.append(ans)
#                 weight_gradient.append(weight_gradient_row)
#             weight_gradient_per_output.append(weight_gradient)
        
#         weight_gradient_per_output = np.array(weight_gradient_per_output)

#         print(weight_gradient_per_output.shape)

#         # Get the mean gradient for the output layer
#         return np.mean(weight_gradient_per_output, axis = 0)

#     @classmethod
#     def middleLayerBackpropGradient(cls, yHat, layer_input, W_matrix, outputLayer_W_matrix_wo_bias, output_layer_loss):
#         assert outputLayer_W_matrix_wo_bias.shape == output_layer_loss.shape
#         _NO_OF_NEURONS = W_matrix.shape[0]

#         for i in range(_NO_OF_NEURONS):
#             temp = np.reshape(layer_input, shape=(65, 9, 1))

#         res = np.ones(shape=W_matrix.shape) * layer_input * Layer.ReLU_Prime(yHat) * np.sum(outputLayer_W_matrix_wo_bias * next_layer_loss, axis=1)


class NeuralNet:
    _THRESHOLD = 0.5

    def __init__(self):
        self.hL1 = Layer(3, 8, 'ReLU')
        self.hL2 = Layer(8, 6, 'ReLU')
        self.outL = Layer(6, 2, 'softmax')
        self.layers = [self.hL1, self.hL2, self.outL]
        # self.layer_outputs = []

    def fit(self, inputs, _train, _numEpochs, truthValues):
        for i in range(_numEpochs):
            output = inputs
            # self.layer_outputs.append(output)
            for layer, to_train in zip(self.layers, _train):
                # np.array([1 for i in range(output.shape[1])])))
                output = np.vstack((output, np.ones(output.shape[1])))
                # self.layer_outputs.append(output)
                output = layer.forward(output, to_train)

            # self.layer_outputs.append(output)
            # self.layer_outputs.append(np.vstack((output,
            # np.ones(output.shape[1]))))

            epoch_loss = self.loss(output, truthValues)
            epoch_accuracy = self.accuracy(output, truthValues)
            epoch_loss = np.mean(epoch_loss)
            print()
            print(f"> Epoch: {i} --> Loss: \n{epoch_loss}, \
                  Accuracy: {epoch_accuracy}")
            return epoch_accuracy, epoch_loss
            # Backpropagation takes place here...
            # print(BackPropagation.outputLayerBackpropGradient(
            # self.layer_outputs[-1], truthValues, self.layer_outputs[-2],
            # self.layers[-1].weights))
            # print(BackPropagation.middleLayerBackpropGradient(
            # self.layer_outputs[-2], self.layer_outputs[-3],
            # self.layers[-2].weights[:-1, :], ))

    @classmethod
    def loss(self, yHat, y):
        # We take the transpose so that the output for
        # each input row from the dataset is now row-wise in yHat
        yHatT = np.transpose(yHat)
        # yT = np.transpose(y)
        ret = []
        # The dimensions must be the same
        assert yHatT.shape == y.shape, \
            f"Shape of yHatT = {yHatT.shape}\nShape of y = {y.shape}"

        # Going row-wise, i.e. corresponding input and output-wise
        for rowYHAT, rowY in zip(yHatT, y):  # yT):
            # They need to be of the same length as if there
            # are 2 target values then we need 2 outputs, per
            # row
            assert len(rowYHAT) == len(rowY)
            # print("rowYHAT is: ", rowYHAT)
            # print("rowY is: ", rowY)
            ret.append(
                        -1 * sum(
                                 np.array(
                                          [
                                           np.log(rowYHAT[i] + 10e-8)
                                           # 10e-8 is needed to prevent nan
                                           # values
                                           if rowY[i] > 0  # == 1
                                           else 0  # np.log(1 - rowYHAT[i])
                                           for i in range(len(rowYHAT))
                                           ]
                                          )
                                ) / len(rowY)
                        )
        # print("ret is: \n", ret)
        return np.array(ret)

    @classmethod
    def loss_prime(self, yHat, y):
        pass

    @classmethod
    def threshold_func(cls, x):
        return 0 if x <= cls._THRESHOLD else 1

    def accuracy(self, yHat, y) -> float:
        # We take the transpose so that the output for
        # each input row from the dataset is now row-wise in yHat
        yHatT = np.transpose(yHat)
        # yT = np.transpose(y)
        # ret = []
        # The dimensions must be the same
        assert yHatT.shape == y.shape

        # Count for the number of correctly classified training samples
        correctly_classified_count = 0

        # Going row-wise, i.e. corresponding input and output-wise
        for rowYHAT, rowY in zip(yHatT, y):  # yT):
            # They need to be of the same length as if there
            # are 2 target values then we need 2 outputs per
            # row
            assert len(rowYHAT) == len(rowY)
            # print(rowYHAT, rowY)

            rowYHAT_after_threshold = rowYHAT.astype(int)

            for i in range(len(rowYHAT)):
                # Apply the threshold on rowYHAT values
                rowYHAT_after_threshold[i] = NeuralNet.\
                                                threshold_func(rowYHAT[i])
            if all(rowYHAT_after_threshold == rowY):
                correctly_classified_count += 1

            #     # if True_Positive or False_Positive
            #     rowYHAT_after_threshold = [lambda x : 0
            #                                if x <= _THRESHOLD else 1]
            #     if (rowYHAT[i] >= _THRESHOLD and rowY[i] == 1) \
            #        or (rowYHAT[i] <= _THRESHOLD and rowY[i] == 0):
            #         correctly_classified_count += 1

        # The number of true values cannot be more than the number of input
        # rows
        assert correctly_classified_count <= y.shape[0]

        # ret.append(correctly_classified_count / y.shape[0])
        # return ret

        return correctly_classified_count / y.shape[0]

# l1 = Layer(3, 5, 'ReLU')
# print(l1.weights)


class Chromosome:
    neural_net_obj = NeuralNet()

    def __init__(self, layer_list) -> None:
        self.numberOfNeuronsInNetwork = 0
        self.numberOfNeuronsPerLayer = []
        self.chromosome = np.array([])

        for layer in layer_list:
            flattened_weight_matrix = np.flatten(layer.weights)
            self.chromosome = np.hstack((self.chromosome,
                                         flattened_weight_matrix))
            self.numberOfNeuronsPerLayer.append(layer.weights.shape[1])

        assert len(self.numberOfNeuronsPerLayer) == 3

        self.numberOfNeuronsInNetwork = sum(self.numberOfNeuronsPerLayer)
        assert self.numberOfNeuronsInNetwork == 16, \
               f"The number of neurons in the neural network is 16, \
                   but you got {self.numberOfNeuronsInNetwork}"

    def _rebuildChromosome(self, selected_weight_matrix, layer):
        assert 0 <= layer <= 2, \
               f"Layer variable value is not in the range [0, 2]"
        if layer == 0:
            assert self.chromosome[:32].shape == selected_weight_matrix\
                                                 .flatten().shape
            self.chromosome[:32] = selected_weight_matrix.flatten()
        elif layer == 1:
            assert self.chromosome[32:86].shape == selected_weight_matrix\
                                                   .flatten().shape
            self.chromosome[32:86] = selected_weight_matrix.flatten()
        else:
            assert self.chromosome[86:].shape == selected_weight_matrix\
                                                 .flatten().shape
            self.chromosome[86:] = selected_weight_matrix.flatten()

    def getWeights(self):
        weight_matrix_1 = np.reshape(self.chromosome[:32], shape=(4, 8))
        weight_matrix_2 = np.reshape(self.chromosome[32:86], shape=(9, 6))
        weight_matrix_3 = np.reshape(self.chromosome[86:], shape=(7, 2))

        return weight_matrix_1, weight_matrix_2, weight_matrix_3

    def mutate(self, n=2):
        # Select n non-input nodes from the chromosome
        nodes_to_mutate = np.random\
                            .randint(0, Chromosome.numberOfNeuronsInNetwork, n)

        # Find the weight matrix to which n belongs to and mutate its weights,
        # i.e. that particular column
        for neuron in nodes_to_mutate:
            layerNo = 0  # Start from the first layer
            while neuron > self.numberOfNeuronsPerLayer[layerNo]:
                neuron -= self.numberOfNeuronsPerLayer[layerNo]
                layerNo += 1
            selected_weight_matrix = self.getWeights()[layerNo]
            assert type(selected_weight_matrix) == np.array, \
                   "The type of the selected_weight_matrix is not a numpy\
                        array"

            # A column of the W matrix is the weights and bias of the neuron
            input_links_of_selected_neuron = selected_weight_matrix[:,
                                                                    neuron - 1]

            # Mutate by adding a random value from the initialization
            # probability distribution
            prevShape = selected_weight_matrix.shape[0]
            shape = selected_weight_matrix.shape[1]
            sd = np.sqrt(6.0 / (prevShape + shape))

            # The mutation happens here
            input_links_of_selected_neuron += np.random.uniform(-sd, sd,
                                                                input_links_of_selected_neuron.shape)

            # Assign the mutated incoming links back to the weights matrix
            selected_weight_matrix[:, neuron - 1] = input_links_of_selected_neuron

            # Add the mutation to the chromosome
            self._rebuildChromosome(selected_weight_matrix, layerNo)

    def evaluate(self, inputs, truthValues, _train=[True, True, False]):
        new_weights = self.getWeights()
        layerList = Chromosome.neural_net_obj.layers

        # Set the weights of the chromosome to the layer object
        for layerInd, layer in enumerate(layerList):
            # Change the weight matrix associated with the layer object
            layer.set_params(new_weights[layerInd])

        self.chromosome_accuracy, self.chromosome_loss = Chromosome\
            .neural_net_obj.fit(inputs, _train, 1, truthValues)\
            # (self, inputs, _train, _numEpochs, truthValues):

        return self.chromosome_accuracy, self.chromosome_loss

    @classmethod
    def cross_over(cls, parent_1, parent_2):
        """cross_over Generate a new Chromosome object from the 2 input
        parent chromosome objects

        :param parent_1: First parent for crossover
        :type parent_1: Chromosome class
        :param parent_2: Second parent for crossover
        :type parent_2: Chromosome class
        :return: Child object created from the 2 parent objects
        :rtype: Chromosome class
        """
        assert parent_1.numberOfNeuronsInNetwork == parent_2\
               .numberOfNeuronsInNetwork, "The number of neurons in \
                                           both the networks are \
                                           not the same!"
        # Contains the weights matrices for all the layers of the child
        # chromosome
        child_layer_list = []
        for layerInd, neuronsCount in enumerate(parent_1
                                                .numberOfNeuronsPerLayer):
            childlayer_weight_matrix = np.array([])
            for neuronInd in range(neuronsCount):
                # Randomly pick between the parent_1 and parent_2
                chosen_parent_index = np.random.randint(0, 2, 1)
                if chosen_parent_index == 0:  # parent_1 is chosen
                    # Get the weights and bias of a particular neuron in
                    # parent_1
                    incoming_links = parent_1.getWeights[layerInd]\
                                     .T[neuronInd, :]
                    childlayer_weight_matrix = np.vstack((childlayer_weight_matrix, incoming_links))
                else:  # parent_2 is chosen
                    # Get the weights and bias of a particular neuron in
                    # parent_2
                    incoming_links = parent_2.getWeights[layerInd]\
                                    .T[neuronInd, :]
                    childlayer_weight_matrix = np.vstack((childlayer_weight_matrix, incoming_links))

            assert np.transpose(childlayer_weight_matrix).shape[1] in [2, 6, 8],\
                   f"{np.transpose(childlayer_weight_matrix)} has the shape\
                       {np.transpose(childlayer_weight_matrix).shape}"
            child_layer_list.append(np.transpose(childlayer_weight_matrix)
                                    .copy())
        assert len(child_layer_list) == 3, f"The child_layer_list looks like \
                                             this: {child_layer_list}"
        return Chromosome(child_layer_list)


class GeneticAlgo:
    _CHROMOSOME_INDEX = 3
    _TIE_BREAKING_INDEX = 2
    _LOSS_INDEX = 1
    _ACCURACY_INDEX = 0
    _TRUTH_VALUES = None
    _INPUTS = None

    def __init__(self, init_population=50, inputs=None, truthValues=None):
        assert inputs is not None, "input not provided!"
        assert truthValues is not None, "truthValues not provided!"
        self.tournament_size = 0
        GeneticAlgo._INPUTS = inputs
        GeneticAlgo._TRUTH_VALUES = truthValues
        """
        Creating the initial population, for the genetic algorithm

        The format for every member of the population list is:
        (<Chromosome_Accuracy>, -1 * abs(<Chromosome_Loss>), <Tie_Breaking_Value>, <Chromosome_Object>)
        """
        # We have created the initial population, so generation-0
        self.generation = 0
        self.population = []
        self.population_count = init_population
        for _ in range(init_population):
            self.population.append((1, -np.absolute(2), _,
                                   Chromosome(NeuralNet().layers)))

        temp = []
        # Create generation-1, after evaluating the generation 0
        MonteCarloList = []
        for individual in self.population:
            node = individual[GeneticAlgo._CHROMOSOME_INDEX]
            for trials in range(10):
                MonteCarloList.append(node.evaluate(GeneticAlgo._INPUTS,
                                                    GeneticAlgo._TRUTH_VALUES,
                                                    [
                                                     random.choice([
                                                                    True,
                                                                    False])
                                                     for _ in range(2)
                                                     ] + False)
                                      )

            _accuracy, _loss = np.mean(np.array(MonteCarloList), axis=0)

            temp.append(
                        (_accuracy, -np.absolute(_loss),
                         individual[GeneticAlgo._TIE_BREAKING_INDEX],
                         node)
                        )

        self.population = copy.deepcopy(temp)

        # Sort the population based on the fitness, such that the fittest
        # setup is first
        self.population.sort(reverse=True)
        del(temp)

    def tournament_selection(self, participant_list):
        """tournament_selection Runs a tournament with the list of
        participant tuples of Chromosomes and their metrics

        :param participant_list: List of tuples of individuals
        :type participant_list: List of tuples of Chromosomes and their metrics
        :return: Tuple containing the 2 selected tuples of Chromosomes and
        their metrics
        :rtype: Tuple of tuples
        """
        assert len(participant_list) == self.tournament_size, \
            f"tournament_size {self.tournament_size} and participants_list \
                length: {len(participant_list)}, are not equal!"
        winner = None
        # Run multiple round as long as the number of participants does not
        # reduce to 2
        while len(participant_list) != 2:
            winners_of_rounds = []
            for battle_participants_ind in range(0, len(participant_list), 2):
                # If there are 2 participants in a round
                if battle_participants_ind <= len(participant_list) - 2:
                    if participant_list[battle_participants_ind] > \
                       participant_list[battle_participants_ind + 1]:
                        winner = participant_list[battle_participants_ind]
                    else:
                        winner = participant_list[battle_participants_ind + 1]

                # If there is only participant in the round then that
                # participant moves forward
                elif battle_participants_ind == len(participant_list) - 1:
                    winner = participant_list[battle_participants_ind]

                assert winner is not None, "Winner is None for some reason \
                and not a tuple of Chromosomes with metrics"
                winners_of_rounds.append(winner)

            participants_list = winners_of_rounds
            random.shuffle(participants_list)

        return participant_list

    def createNextGeneration(self, elitism_frac=0.1, tournament_size=20,
                             mutation_frac=0.1):
        self.tournament_size = tournament_size
        newPopulation = []
        # Order the parents based on the accuracy, loss and tie-breaking value
        self.population.sort(reverse=True)

        # Perform elitism, let the best parent go forward unchanged
        for i in range(round(elitism_frac * self.population_count)):
            temp = copy.deepcopy(self.population[i])
            temp[GeneticAlgo._TIE_BREAKING_INDEX] = i
            newPopulation.append(copy.deepcopy(temp))
            del(temp)

        """
        Tournament selection and Crossover have to be performed until the next
        generation's population size is not equal self.population_count
        """
        while len(newPopulation) != self.population_count:
            # Select 2 parent from the initial population, and perfrom
            # crossover, using tournament selection
            # Select participants for the tournament
            _participant_list = []
            # A participant may appear multiple times in the tournament
            _participant_indices = np.random.randint(0, self.population_count,
                                                     tournament_size)
            for _ in range(self.population_count):
                # Check if the population member is a participant
                if _ in _participant_indices:
                    # Add the participant as many times as its index showed up
                    # in the _participant_indices
                    for _freq in range(np.
                                       count_nonzero(_participant_indices == _)
                                       ):
                        _participant_list.append(self.population[_])
                        # ..[np.copy(GeneticAlgo._CHROMOSOME_INDEX)])

            assert len(_participant_list) == tournament_size,\
                f"Participant list length {len(_participant_list)} \
                    != tournament_size {tournament_size}"
            # Shuffle the participants
            random.shuffle(_participant_list)

            # Run the tournament
            _winner_1, _winner_2 = self\
                .tournament_selection(copy
                                      .deepcopy(_participant_list))

            # Perform cross-over on the 2 remaining winners at the end to
            # get the child node
            child_node = Chromosome\
                .cross_over(_winner_1[GeneticAlgo._CHROMOSOME_INDEX],
                            _winner_2[GeneticAlgo._CHROMOSOME_INDEX])

            # Find the accuracy and loss of the child node and append it to
            # the new population
            # Randomly enable or disbale dropouts in the different layers of
            # the neural network - MonteCarlo Dropout
            MonteCarloList = []
            for i in range(10):
                MonteCarloList.append(child_node
                                      .evaluate(GeneticAlgo._INPUTS,
                                                GeneticAlgo._TRUTH_VALUES,
                                                [random.choice([True, False])
                                                 for i in range(2)] + False)
                                      )

            _accuracy, _loss = np.mean(np.array(MonteCarloList), axis=0)

            newPopulation.append((_accuracy, _loss, len(newPopulation),
                                  child_node))

        assert all([len(x) == 4 for x in newPopulation]) is True, "The Length\
            of all members of newPopulation is not 4!"

        """
        Mutation has to be done after the new population has been created
        """
        # Randomly select individuals without replacement from the new
        # population to perform mutation on
        rng = default_rng()
        mutation_candidates_index = np\
            .random\
            .shuffle(rng.choice(self.population_count,
                     size=round(self.population_count * mutation_frac),
                     replace=False))

        _accuracy = None
        _loss = None
        for _ in range(self.population_count):
            if _ in mutation_candidates_index:
                newPopulation[_][GeneticAlgo._CHROMOSOME_INDEX].mutate()

                mutatedNode = newPopulation[_][GeneticAlgo._CHROMOSOME_INDEX]
                # Recalculate the individual's accuracy and loss
                MonteCarloList = []
                for i in range(10):
                    MonteCarloList.append(mutatedNode
                                          .evaluate(GeneticAlgo._INPUTS,
                                                    GeneticAlgo._TRUTH_VALUES,
                                                    [random.choice([
                                                                    True,
                                                                    False
                                                                    ])
                                                     for i in range(2)]
                                                    + False)
                                          )
                    _accuracy, _loss = np.mean(np.array(MonteCarloList),
                                               axis=0)
                assert _accuracy is not None
                assert _loss is not None
                newPopulation[_] = (_accuracy, _loss, newPopulation[_]
                                    [GeneticAlgo._TIE_BREAKING_INDEX],
                                    mutatedNode)

        self.population = copy.deepcopy(newPopulation)
        self.population.sort(reverse=True)

    def runner(self, noOfGenerations):
        for _ in noOfGenerations:
            self.createNextGeneration()
            print("The accuracy and loss of the best 5 individuals are:")
            for ind, individual in enumerate(self.population[:5]):
                print(f"Individual no. : {ind}")
                print(individual[GeneticAlgo._ACCURACY_INDEX])
                print(individual[GeneticAlgo._LOSS_INDEX])
                print("-"*80)

        # Return the best Chromosome object
        return self.population[0]


df = pd.read_csv("train_split.csv")
df.head()

numpyinput = df[['Weight', 'HB', 'BP']].to_numpy()
numpyinput = numpyinput.transpose()
# numpyinput

numpyoutput = df[['Result_0.0', 'Result_1.0']].to_numpy()

darwin = GeneticAlgo(inputs=numpyinput, truthValues=numpyoutput)
darwin.runner(10)
# numpyoutput = numpyoutput.transpose()
# numpyoutput

# network = NeuralNet()
# network.fit(numpyinput, [True, True, False], 1, numpyoutput)

# -sum([np.log(0.48958263+10e-8), 0])/2, -np.log(0.48958263+10e-8)/2

# a = [[1, 2, 3],[3, 4, 5],[6, 7, 8]]
# a = np.array(a).transpose()
# Layer.softmax_Prime(a)
