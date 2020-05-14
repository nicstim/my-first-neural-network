"""
Состав сети:
-Два входа
-Один скрытый слой с двумя нейронами
-Слой вывода с одним нейроном
"""

import numpy as np

def sigmoid(x):
    # функция активации: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias


    def feedforward(self, input):
        total = np.dot(self.weights, input) + self.bias
        return sigmoid(total)


class NeuralNetwork:
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)


    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1


network = NeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))  # 0.7216325609518421
