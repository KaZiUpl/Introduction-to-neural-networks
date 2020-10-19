import numpy as np


class Perceptron:
    def __init__(self, no_of_inputs, learning_rate=0.01, iterations=100):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.no_of_inputs = no_of_inputs
        self.weights = np.zeros(self.no_of_inputs + 1)

    def train(self, training_data, labels):
        for _ in range(self.iterations):
            for input, label in zip(training_data, labels):
                prediction = self.predict(input)
                self.weights[1:] += self.learning_rate * \
                    (label - prediction) * input
                self.weights[0] += self.learning_rate * (label - prediction)

    def predict(self, input):
        # self.weights[0] == -theta
        summation = np.dot(input, self.weights[1:]) + self.weights[0]
        if summation >= 0:
            activation = 1
        else:
            activation = -1
        return activation
