import unittest
import numpy as np
from layered_neural_network import LayeredNeuralNetwork
from node_manager import NodeManager
from node import Node
import transform_function


class TestLayeredNeuralNetwork(unittest.TestCase):
    def test_train_twice(self):
        input_dimension = 3
        sample_size = 5
        X = np.random.rand(sample_size, input_dimension)
        Y = np.random.randint(0, high=2, size=sample_size)
        base_label = 'test_train_'
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        for i in range(2):
            model.fit(X, Y, base_label + str(i))

    def test_identity_learn_perfect(self):
        input_dimension = 1
        sample_size = 5
        Y = np.random.randint(0, high=2, size=sample_size)
        X = Y.reshape([-1,1])
        label = 'identity'
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        model.fit(X, Y, label)
        score = model.score(X,Y, label=label)
        self.assertEqual(1, score)


class TestNodeManager(unittest.TestCase):
    def test_simple_get_output_from_input(self):
        input_dimension = 2
        sample_size = 5
        node_manager = NodeManager(input_dimension)
        X = np.random.rand(sample_size, input_dimension)
        output = node_manager.get_output(X, 'input_0')
        self.assertTrue(np.array_equal(X[:, 0], output))

    def test_linear_transform(self):
        input_dimension = 2
        sample_size = 5
        weights = np.array([0.5, 0.5])
        bias = 0
        linear_transform = transform_function \
            .LinearTransformFunction(input_dimension=input_dimension, weights=weights, bias=bias)
        node_manager = NodeManager(input_dimension)
        first_node = Node('first', node_manager.get_input_names(),
                          linear_transform, node_manager, is_input=False)
        node_manager.add_node(first_node)
        X = np.random.rand(sample_size, input_dimension)
        output = node_manager.get_output(X, first_node.name)
        self.assertTrue(np.array_equal(X.mean(axis=1), output))
