import numpy as np
import node_manager
import node
from sklearn import svm, metrics
import transform_function

retrain_threshold_f1_score = 0.9


class LayeredNeuralNetwork():
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.node_manager = node_manager.NodeManager(input_dimension)
        self.label_to_node_name = {}
        self.labels = []

    def train(self, X, Y, label):
        """
        Trains the model using data X for class named "label".
        Y is binary indicating presence/absence of class.

        :param X: numpy matrix of size (samples, features)
        :param Y: numpy array of integers with values 0,1
        :type label: str
        :rtype: bool
        :return: whether retrained
        """
        print('Training for label ' + label)
        if label in self.labels:
            score = self.score(X, Y, label)
            if score > retrain_threshold_f1_score:
                print('Label {0} already exists with score {1}. Not retraining'.format(label, score))
                return False
            else:
                print('Label {0} exists with score {1}. Retraining'.format(label, score))

        sample_count = X.shape[0]
        input_and_features = np.zeros(shape=[sample_count, self.input_dimension + len(self.labels)])
        input_and_features[:, :self.input_dimension] = X
        input_and_features[:, self.input_dimension:] = self.activate_all(X)
        linear_svc = svm.LinearSVC(penalty='l1')
        linear_svc.fit(input_and_features, Y)
        score = linear_svc.score(X, Y)
        print('Trained new Linear SVC with score ' + str(score))
        learned_transform_function = transform_function.LinearTransformFunction(
            input_dimension=input_and_features.shape[1],
            weights=linear_svc.coef_,
            bias=linear_svc.intercept_)

        node_name = label + '_' + str(np.random.randint(low=0, high=999))
        new_node = node.Node(name=node_name, input_names=self.node_manager.get_input_names() + self.labels,
                             transform_function=learned_transform_function,
                             node_manager=self.node_manager,
                             is_input=False)
        self.node_manager.add_node(new_node)

    def predict(self, X, label):
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

    def activate_all(self, X):
        """
        Activates all the labels.

        :rtype: np.array
        :return: numpy array with dimension (samples, learned label count)
        """
        sample_count = X.shape[0]
        result = np.zeros(shape=[sample_count, self.feature_count()])
        for i, label in enumerate(self.labels):
            result[:, i] = self.node_manager.get_output(X, label)
        return result

    def identify(self, X):
        """
        Best guess for which class X belongs to.
        :param X: numpy matrix of size (samples, features)
        :return: guessed class name
        :rtype: str
        """
        features = self.activate_all(X)
        max_index = np.argmax((features ** 2).mean(axis=0))
        return self.labels[max_index]

    def feature_count(self):
        return len(self.labels)

    def score(self, X, Y, label):
        predicted_y = self.predict(X, label)
        return metrics.f1_score(Y, predicted_y)
