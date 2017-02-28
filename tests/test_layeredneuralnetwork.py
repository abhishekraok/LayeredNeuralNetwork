import unittest
import numpy as np
from layeredneuralnetwork.layered_neural_network import LayeredNeuralNetwork
from layeredneuralnetwork.node_manager import NodeManager
from layeredneuralnetwork.node import Node
from layeredneuralnetwork import transform_function
from layeredneuralnetwork import utilities


class TestLayeredNeuralNetwork(unittest.TestCase):
    def test_train_twice(self):
        input_dimension = 3
        sample_size = 500
        X = np.random.rand(sample_size, input_dimension)
        Y = np.random.randint(0, high=2, size=sample_size)
        base_label = 'test_train_'
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        for i in range(2):
            model.fit(X, Y, base_label + str(i))

    def test_identity_learn_perfect(self):
        input_dimension = 1
        sample_size = 500
        Y = np.random.randint(0, high=2, size=sample_size)
        X = Y.reshape([-1, 1])
        label = 'identity'
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        model.fit(X, Y, label)
        score = model.score(X, Y, label=label)
        self.assertEqual(1, score)

    def test_get_weight_correct_dimension(self):
        input_dimension = 4
        sample_size = 500
        X = np.random.rand(sample_size, input_dimension)
        Y = np.random.randint(0, high=2, size=sample_size)
        label = 'weight_check'
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        model.fit(X, Y, label)
        weights = model.get_weights()
        self.assertTrue(weights.flatten().shape[0], input_dimension)

    def test_new_node_name(self):
        label = 'hi'
        expected_new_nod_name = 'hi' + utilities.node_version_separator + '0'
        model = LayeredNeuralNetwork(2)
        new_node_name = model.get_new_node_name(label)
        self.assertEqual(expected_new_nod_name, new_node_name)
        model.labels = [label]
        model.label_to_node_name[label] = expected_new_nod_name
        expected_new_nod_name2 = 'hi' + utilities.node_version_separator + '1'
        new_node_name2 = model.get_new_node_name(label)
        self.assertEqual(expected_new_nod_name2, new_node_name2)


class TestNodeManager(unittest.TestCase):
    def test_simple_get_output_from_input(self):
        input_dimension = 2
        sample_size = 500
        node_manager = NodeManager(input_dimension)
        X = np.random.rand(sample_size, input_dimension)
        output = node_manager.get_output(X, 'input_0')
        self.assertTrue(np.array_equal(X[:, 0], output))
