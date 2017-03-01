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
        :return: a tuple of mean F1 score and dictionary of label:F1 score
        """
        N = classifier.input_dimension
        positive_sample_count = 1000
        X = np.random.randn(2 * positive_sample_count, N)
        Y = np.random.randint(low=0, high=1, size=[2 * positive_sample_count])
        scores = {}
        base_labels = ['cos_', 'sin_']
        for base_label in base_labels:
            for f in range(N):
                label = base_label + str(f)
                wave = np.sin([f * i * (2 * math.pi) / N for i in range(N)])
                X[:positive_sample_count, :] = np.array([wave, ] * positive_sample_count)
                Y[:positive_sample_count] = np.ones(shape=positive_sample_count)
                x_train, x_test, y_train, y_test = train_test_split(X, Y)
                classifier.fit(x_train, y_train, label)
                scores[label] = (classifier.score(x_test, y_test, label))

        mean = np.mean(scores.values())
        print('Finished training on frequency domain mean F1 score is ' + str(mean))
        return mean, scores
