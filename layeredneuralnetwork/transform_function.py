import numpy as np
from sklearn.svm import LinearSVC


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
        if len(weights.shape) != 1 and len(weights.shape) != 2:
            raise ValueError('Weight dimension is improper. Needs to be 1 or 0. But is '
                             + str(weights.shape))
        if len(weights.shape) == 2:
            flattened_weights = weights.flatten()
        else:
            flattened_weights = weights
        if flattened_weights.shape[0] != input_dimension:
            raise ValueError('Weight shape is improper. Needs to be '
                             + str(input_dimension) + ' but is ' + str(weights.shape))
        TransformFunction.__init__(self, input_dimension)
        self.weights = flattened_weights
        self.bias = bias

    def transform(self, X):
        return np.dot(X, self.weights) + self.bias

    def get_weights(self):
        return np.hstack([self.weights, self.bias])


class SVCTransformFunction(TransformFunction):
    def __init__(self, input_dimension, svm):
        """
        Linear Support Vector Machine

        :type input_dimension: int
        :type svm: LinearSVC
        """
        TransformFunction.__init__(self, input_dimension)
        self.svm = svm

    def transform(self, X):
        return self.svm.decision_function(X)

    def get_weights(self):
        return np.hstack([self.svm.coef_, [self.svm.intercept_]])
