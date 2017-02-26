import numpy as np


class LayeredNeuralNetwork():
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension

    def train(self, X, Y, label):
        """
        Trains the model using data X for class named "label".
        Y is binary indicating presence/absence of class.

        :param X: numpy matrix of size (samples, features)
        :param Y: numpy array of integers with values 0,1
        :type label: str
        :rtype: None
        """
        pass

    def predict(self, X , label):
        """
        Predicts whether given X belongs to class "label".

        :param X: numpy matrix of size (samples, features)
        :type label: str
        :return: a numpy array of size (samples) containing 1,0
        :rtype: np.array
        """
        pass


    def identify(self, X):
        """
        Best guess for which class X belongs to.
        :param X: numpy matrix of size (samples, features)
        :return: guessed class name
        :rtype: str
        """
        pass