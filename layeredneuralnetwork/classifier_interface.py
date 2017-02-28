class ClassifierInterface:
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension

    def fit(self, X, Y, label):
        raise NotImplementedError('This is an interface')

    def predict(self, X, label):
        """
        Predicts whether given X belongs to class "label".

        :param X: numpy matrix of size (samples, features)
        :type label: str
        :return: a numpy array of size (samples) containing 1,0
        :rtype: np.array
        """
        raise NotImplementedError('This is an interface')

    def identify(self, X):
        """
        Best guess for which class X belongs to.
        :param X: numpy matrix of size (samples, features)
        :return: guessed class name
        :rtype: str
        """
        raise NotImplementedError('This is an interface')

    def score(self, X, Y, label):
        """
        Gets the F1 score for given input for given label.
        :rtype: float
        """
        raise NotImplementedError('This is an interface')
