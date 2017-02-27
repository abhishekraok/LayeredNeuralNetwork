"""
A Transfer Learning School.

Teaches classifier different simple task.
The difference from traditional task is here some tasks have pre-requisites.
e.g. Learning XOR is easier if you know AND and OR task.
After learning each task the classifier's score is calculated.
As the classifier gets better more challenging tasks can be taught.
"""
import layered_neural_network
import numpy as np


class Binary:
    @staticmethod
    def teach_and(classifier):
        """
        Teaches the AND task

        :type classifier: layered_neural_network.LayeredNeuralNetwork
        :return: F1 Score in learning this task
        """
        return Binary.teach_binary_function(classifier, 'and', lambda a, b: a and b)

    @staticmethod
    def teach_or(classifier):
        """
        Teaches the OR task

        :type classifier: layered_neural_network.LayeredNeuralNetwork
        :return: F1 Score in learning this task
        """
        return Binary.teach_binary_function(classifier, 'or', lambda a, b: a or b)

    @staticmethod
    def teach_xor(classifier):
        """
        Teaches the XOR task.

        Pre-requisite AND task, OR task.

        :type classifier: layered_neural_network.LayeredNeuralNetwork
        :return: F1 Score in learning this task
        """
        return Binary.teach_binary_function(classifier, 'xor', lambda a, b: int(a) ^ int(b))

    @staticmethod
    def teach_binary_function(classifier, label, fx):
        """
        Teaches the classifier to calculate a binary function fx

        :type classifier: layered_neural_network.LayeredNeuralNetwork
        :type label: str
        :param fx: a function that takes two input and returns a single output
        :return: F1 Score in learning this task
        """
        print('Teaching classifier task of ' + label)
        classifier_dimension = classifier.input_dimension
        if classifier_dimension < 2:
            raise ValueError('Need classifier with at least input dimension of 2')
        sample_size = 100
        X = np.random.randn(sample_size, classifier_dimension)
        X[:, :2] = np.random.randint(low=0, high=2, size=[sample_size, 2], dtype=np.int)
        Y = np.array([fx(i, j) for i, j in zip(X[:, 0], X[:, 1])], dtype=np.int)
        classifier.fit(X, Y, label=label)
        return classifier.score(X, Y, label)
