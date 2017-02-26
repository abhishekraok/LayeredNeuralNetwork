import unittest
import numpy as np
from layered_neural_network import LayeredNeuralNetwork
from node_manager import NodeManager
from node import Node


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


class TestNodes(unittest.TestCase):
    pass


class TestNodeManager(unittest.TestCase):
    def test_simple_get_output_from_input(self):
        input_dimension = 2
        sample_size = 5
        node_manager = NodeManager(input_dimension)
        X = np.random.rand(sample_size, input_dimension)
        output = node_manager.get_output(X, 'input_0')
        self.assertEqual(X, output[:,0])


class TestTransformFunction(unittest.TestCase):
    pass
