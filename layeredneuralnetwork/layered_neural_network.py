import numpy as np
import node_manager


class LayeredNeuralNetwork():
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.node_manager = node_manager.NodeManager(input_dimension)
        self.label_to_node_name = {}

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
        if not label in self.label_to_node_name:
            raise ValueError('No label named ' + label + ' in this LNN')
        node_name = self.label_to_node_name[label]
        features = self.node_manager.get_output(X, node_name)
        return np.array(features > 0, dtype=np.int)


    def identify(self, X):
        """
        Best guess for which class X belongs to.
        :param X: numpy matrix of size (samples, features)
        :return: guessed class name
        :rtype: str
        """
        max_label = self.label_to_node_name.keys()[0]
        max_strength = 0
        for label in self.label_to_node_name.keys():
            features = self.node_manager.get_output(X, label)
            feature_strength = features.mean_square()
            if feature_strength > max_strength:
                max_strength = feature_strength
                max_label = label
        return max_label
