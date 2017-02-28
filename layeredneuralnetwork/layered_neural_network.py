import numpy as np
import node_manager
import node
from sklearn import svm, metrics
import transform_function
from classifier_interface import ClassifierInterface

retrain_threshold_f1_score = 0.9


class LayeredNeuralNetwork(ClassifierInterface):
    def __init__(self, input_dimension):
        ClassifierInterface.__init__(self, input_dimension)
        self.input_dimension = input_dimension
        self.node_manager = node_manager.NodeManager(input_dimension)
        self.label_to_node_name = {}
        self.labels = []

    def fit(self, X, Y, label):
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
        self.fit_new_node(X, Y, label)
        return True

    def fit_new_node(self, X, Y, label):
        sample_count = X.shape[0]
        input_and_features = np.zeros(shape=[sample_count, self.input_dimension + len(self.labels)])
        input_and_features[:, :self.input_dimension] = X
        input_and_features[:, self.input_dimension:] = self.activate_all(X)
        linear_svc = svm.LinearSVC(dual=False, penalty='l1')
        linear_svc.fit(input_and_features, Y)
        score = linear_svc.score(input_and_features, Y)
        print('Trained new Linear SVC with score ' + str(score))
        learned_transform_function = transform_function.LinearTransformFunction(
            input_dimension=input_and_features.shape[1],
            weights=linear_svc.coef_,
            bias=linear_svc.intercept_)
        # todo increment version rather than create random number
        node_name = label + '_' + str(np.random.randint(low=0, high=999))
        input_names = self.node_manager.get_input_names() + self.latest_node_names()
        new_node = node.Node(name=node_name,
                             input_names=input_names,
                             transform_function=learned_transform_function,
                             node_manager=self.node_manager,
                             is_input=False)
        self.node_manager.add_node(new_node)
        if label not in self.label_to_node_name:
            self.labels.append(label)
            self.label_to_node_name[label] = node_name

    def latest_node_names(self):
        return [self.label_to_node_name[i] for i in self.labels]

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
            result[:, i] = self.node_manager.get_output(X, self.label_to_node_name[label])
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
        """
        Gets the F1 score for given input for given label.
        :rtype: float
        """
        predicted_y = self.predict(X, label)
        return metrics.f1_score(Y, predicted_y)

    def get_weights(self):
        weights = np.zeros(shape=[len(self.labels), self.input_dimension + len(self.labels)])
        for i, label in enumerate(self.labels):
            label_weight = self.node_manager.get_weight(self.label_to_node_name[label])
            weights[i, :label_weight.shape[0]] = label_weight
        return weights
