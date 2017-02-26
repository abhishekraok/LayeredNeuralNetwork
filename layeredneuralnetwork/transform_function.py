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
        :rtype: np.array
        """
        raise NotImplementedError('This is an interface.')


class LinearTransformFunction(TransformFunction):
    def __init__(self, input_dimension, weights, bias):
        if len(weights.shape) is not 1 or weights.shape[0] is not input_dimension:
            raise ValueError('Weight shape is improper ' + str(weights.shape))
        TransformFunction.__init__(self, input_dimension)
        self.weights = weights
        self.bias = bias

    def transform(self, X):
        return np.dot(X, self.weights) + self.bias
