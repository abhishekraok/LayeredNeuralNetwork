import unittest
import numpy as np
from layered_neural_network import LayeredNeuralNetwork


class TestLayeredNeuralNetwork(unittest.TestCase):
    def test_train_twice(self):
        input_dimension = 3
        sample_size = 5
        X = np.random.rand(sample_size, input_dimension)
        Y = np.random.randint(0, high=1, size=sample_size)
        base_label = 'test_train_'
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        for i in range(2):
            model.train(X, Y, base_label + str(i))
