import numpy as np
import random


class Perceptron:
    def __init__(self, no_of_inputs, learning_rate=0.01, iterations=10000):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.no_of_inputs = no_of_inputs
        self.weights = np.random.rand(self.no_of_inputs + 1)

    def train(self, training_data, labels):
        for _ in range(self.iterations):
            for input, label in zip(training_data, labels):
                prediction = self.predict(input)
                self.weights[1:] += self.learning_rate * \
                    (label - prediction) * input
                self.weights[0] += self.learning_rate * (label - prediction)

    def trainSPLA(self, training_data, labels):
        correct_flag = False
        zippedList = list(zip(training_data, labels))

        while correct_flag == False:
            correct_flag = True
            index = random.randint(0, len(zippedList) - 1)

            input = zippedList[index][0]
            label = zippedList[index][1]

            prediction = self.predict(input)
            err = label - prediction
            if err != 0:
                correct_flag = False
                self.weights[1:] += self.learning_rate * err * input
                self.weights[0] += self.learning_rate * err
            else:
                if self.checkPredictions(training_data, labels) == False:
                    correct_flag = False

    def trainPLA(self, training_data, labels):
        alive_timer = 0
        best_weights = self.weights
        best_alive_timer = 0

        for _ in range(self.iterations):
            index = random.randint(0, len(training_data) - 1)

            input = training_data[index]
            label = labels[index]

            err = label - self.predict(input)

            if err != 0:
                # correct weights
                self.weights[1:] += self.learning_rate * err * input
                self.weights[0] += self.learning_rate * err
                alive_timer = 0
            else:
                # add alive timer and continue
                alive_timer += 1
                if alive_timer > best_alive_timer:
                    # change best weights
                    best_weights = self.weights
                    best_alive_timer = alive_timer

        self.weights = best_weights

    def trainPLARatchet(self, training_data, labels):
        alive_timer = 0
        best_weights = self.weights
        best_alive_timer = best_examples_counter = 0

        for _ in range(self.iterations):
            index = random.randint(0, len(training_data) - 1)

            input = training_data[index]
            label = labels[index]

            err = label - self.predict(input)

            if err != 0:
                self.weights[1:] += self.learning_rate * err * input
                self.weights[0] += self.learning_rate * err
                alive_timer = 0
            else:
                alive_timer += 1
                correct_classif = self.correctClassifications(
                    training_data, labels)
                if alive_timer > best_alive_timer and correct_classif > best_examples_counter:
                    best_weights = self.weights
                    best_examples_counter = correct_classif
        self.weights = best_weights

    def predict(self, input):
        # self.weights[0] == -theta
        summation = np.dot(input, self.weights[1:]) + self.weights[0]
        if summation > 0:
            return 1
        else:
            return -1

    def checkPredictions(self, training_data, labels):
        for input, label in zip(training_data, labels):
            err = label - self.predict(input)
            if err != 0:
                return False
        return True

    def correctClassifications(self, training_data, labels):
        out = 0
        for input, label in zip(training_data, labels):
            err = label - self.predict(input)
            if err == 0:
                out += 1
        return out
