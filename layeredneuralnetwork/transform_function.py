import numpy as np


class TransformFunction:
    """
    Interface, transforms input into output based on some function like neural network node.
    """

    def __init__(self, input_dimension):
        self.input_dimension = input_dimension

    def transform(self, X):
        """
        Transforms the given input matrix into output feature vector

        :param X: a numpy array of size (samples, input_dimension)
        :type X: np.array of size (samples)
        :rtype: np.array
        """
        raise NotImplementedError('This is an interface.')
