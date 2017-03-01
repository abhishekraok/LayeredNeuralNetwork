import unittest
from layeredneuralnetwork.layered_neural_network import LayeredNeuralNetwork
from layeredneuralnetwork.school import binary
from layeredneuralnetwork.school import frequency


class TestLayeredNeuralNetworkSchool(unittest.TestCase):
    def test_train_and_score(self):
        input_dimension = 9
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        score = binary.Binary.teach_and(model)
        self.assertEqual(1, score)

    def test_train_or_score(self):
        input_dimension = 9
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        score = binary.Binary.teach_or(model)
        self.assertEqual(1, score)

    def test_train_xor_score_fail_without_pre_requisite(self):
        input_dimension = 9
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        score = binary.Binary.teach_xor(model)
        self.assertLess(score, 0.7)

    def test_train_xor_score_succeed_with_pre_requisite(self):
        input_dimension = 2
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        and_score = binary.Binary.teach_and(model)
        self.assertEqual(1, and_score)
        or_score = binary.Binary.teach_or(model)
        self.assertEqual(1, or_score)
        xor_score = binary.Binary.teach_xor(model)
        self.assertEqual(1, xor_score)

    def test_frequency_all(self):
        input_dimension = 21
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        score, detailed_report = frequency.Frequency.teach_all_frequency(model)
        self.assertGreater(score, 0.9)
