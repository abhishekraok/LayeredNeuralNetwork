import unittest
from layered_neural_network import LayeredNeuralNetwork
import school


class TestLayeredNeuralNetworkSchool(unittest.TestCase):
    def test_train_and_score(self):
        input_dimension = 9
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        score = school.Binary.teach_and(model)
        self.assertEqual(1, score)

    def test_train_or_score(self):
        input_dimension = 9
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        score = school.Binary.teach_or(model)
        self.assertEqual(1, score)

    def test_train_xor_score_fail_without_pre_requisite(self):
        input_dimension = 9
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        score = school.Binary.teach_xor(model)
        self.assertLess(score, 0.7)

    @unittest.skip('Need to investigate further')
    def test_train_xor_score_succeed_with_pre_requisite(self):
        input_dimension = 9
        model = LayeredNeuralNetwork(input_dimension=input_dimension)
        and_score = school.Binary.teach_and(model)
        self.assertEqual(1, and_score)
        or_score = school.Binary.teach_or(model)
        self.assertEqual(1, or_score)
        xor_score = school.Binary.teach_xor(model)
        self.assertEqual(1, xor_score)
