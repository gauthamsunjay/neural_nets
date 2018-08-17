import math
import random
import numpy as np


class NN(object):

    SIGMOID_THRESHOLD = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, lr, epochs):
        self.weights = []
        self.num_ouputs = num_outputs
        self.num_hidden = num_hidden

        if num_hidden > 0:
            hidden = np.random.uniform(-0.01, 0.01, size=(num_inputs + 1,
                                                          num_hidden))
            self.weights.append(hidden)

        else:
            num_hidden = num_inputs

        output = np.random.uniform(-0.01, 0.01, size=(num_hidden + 1,
                                                      num_outputs))

        self.weights.append(output)
        self.lr = lr
        self.epochs = epochs

    @staticmethod
    def sigmoid(val):
        return 1 / (1 + math.exp(-val))

    def __forward_propagate(self, record):
        sigmoid = np.vectorize(NN.sigmoid)
        activation = record
        activations = []

        for layer in self.weights:
            activation = np.insert(activation, 0, 1, axis=1)
            activations.append(activation)
            activation = sigmoid(np.dot(activation, layer))

        prediction = activation[0][0]
        return activations, prediction

    def __back_propagate(self, expected, activations, prediction):
        deltas = []
        updated_weights = []
        for i in range(len(self.weights) - 1, -1, -1):
            if i == len(self.weights) - 1:
                delta = np.array([expected - prediction], ndmin=2)
            else:
                weight_sums = np.dot(self.weights[i + 1], deltas[i - 1])
                prod = np.multiply(activations[i + 1], 1 - activations[i + 1])
                weight_sums = weight_sums.transpose()
                delta = np.multiply(weight_sums, prod)

            deltas.append(delta)

        for i, weight in enumerate(reversed(self.weights)):
            j = len(self.weights) - 1 - i
            if i == 0:
                delt = deltas[i]
            else:
                delt = deltas[i][0][1:]
                delt = np.reshape(delt, (1, delt.shape[0]))

            del_times_act = self.lr * np.dot(np.transpose(activations[j]),
                                             delt)

            updated_weights.append(np.add(weight, del_times_act))

        self.weights = list(reversed(updated_weights))

    def train(self, data):
        for epoch in range(1, self.epochs + 1):
            misclassified = correctly_classified = ce_error = 0
            random.shuffle(data)
            for record, expected in data:
                activations, prediction = self.__forward_propagate(record)
                predicted_class = 1 if prediction >= NN.SIGMOID_THRESHOLD else 0
                if expected == predicted_class:
                    correctly_classified += 1
                else:
                    misclassified += 1

                self.__back_propagate(expected, activations, prediction)

            for record, expected in data:
                activations, prediction = self.__forward_propagate(record)
                ce_error += ((-expected * math.log(prediction, math.e)) -
                             ((1 - expected) * math.log(1 - prediction, math.e)))

            print "%d\t%.8f\t%d\t%d" % (epoch, ce_error,
                                        correctly_classified, misclassified)

    def predict(self, record):
        activations, prediction = self.__forward_propagate(record)
        return prediction
