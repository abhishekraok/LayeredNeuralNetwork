import unittest
from layered_neural_network import LayeredNeuralNetwork
import school


class TestLayeredNeuralNetworkSchool(unittest.TestCase):
    def test_train_and_score(self):
        input_dimension = 9
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        score = school.teach_and(model)
        self.assertEqual(1, score)

    def test_train_or_score(self):
        input_dimension = 9
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        score = school.teach_or(model)
        self.assertEqual(1, score)

    def test_train_xor_score(self):
        input_dimension = 9
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        score = school.teach_xor(model)
        self.assertEqual(1, score)
