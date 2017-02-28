"""
Teaches frequency domanain calculations.
"""
from layeredneuralnetwork.classifier_interface import ClassifierInterface

import math
import numpy as np
from sklearn.model_selection import train_test_split


class Frequency:
    @staticmethod
    def teach_all_frequency(classifier):
        """
        Will teach to identify sin(x) and cos(x) of all frequencies.

        :type classifier:  ClassifierInterface
        """
        N = classifier.input_dimension
        sample_count = 500
        X = np.random.randn(sample_count, N)
        Y = np.random.randint(low=0, high=1, size=[sample_count])
        scores = []
        for f in range(N):
            label = 'frequency_' + str(f)
            wave = np.sin([f * i * (2 * math.pi) / N for i in range(N)])
            X[:250, :] = np.array([wave, ] * 250).transpose()
            Y[:250] = np.ones(shape=250)
            x_train, x_test, y_train, y_test = train_test_split(X, Y)
            classifier.fit(x_train, y_train, label)
            scores.append(classifier.score(x_test, y_test, label))
        return np.mean(scores)
