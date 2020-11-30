import numpy as np
import random
import matplotlib.pyplot as plt


def fourier_transform(x):
    a = np.abs(np.fft.fft(x))
    return a / np.max(a)


class Adaline(object):
    def __init__(self, label, no_of_inputs, learning_rate=0.1, iterations=10000, biased=False):
        self.no_of_inputs = no_of_inputs
        self.learning_rate = learning_rate
        self.iterations = iterations
        # zadanie: dodanie biasu jest opcjonalne
        self.weights = np.random.randn(2*self.no_of_inputs + 1)-0.5
        self.errors = []
        self.biased = biased
        self.number = label

    def standarize_features(self, x):
        return x
        # return (x-np.mean(x))/np.std(x)

    def train(self, training_data, labels):
        preprocessed_training_data = self.standarize_features(
            training_data)
        for _ in range(self.iterations):

            e = 0
            # shuffle list
            randomize_list = list(zip(training_data, labels))
            random.shuffle(randomize_list)
            training_data, labels = zip(*randomize_list)

            for example, label in zip(training_data, labels):
                example_fft = np.concatenate([
                    example, fourier_transform(example)])
                out = self.output(example)

                self.weights[1:] += self.learning_rate * \
                    (label-out) * example_fft * \
                    self.activation_derivative(out)
                self.weights[0] += self.learning_rate * \
                    (label-out) * self.activation_derivative(out)
                e += 0.5 * (label - out)**2
            self.errors.append(e)
        plt.plot(range(len(self.errors)), self.errors, label=str(self.number))
        plt.legend()
        plt.savefig('errors.pdf')

    def activation(self, x):  # dodanie funkcji aktywacji -> zmiana pochodnej
        return 1/(1 + np.exp(-x))

    def activation_derivative(self, x):
        return self.activation(x)*(1-self.activation(x))

    def output(self, input, biased=False):
        inp = np.concatenate([input, fourier_transform(input)])
        return self.activation(np.dot(self.weights[1:], inp) + self.weights[0])
