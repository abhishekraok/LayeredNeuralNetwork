# LayeredNeuralNetwork
A transfer learning Neural Network that uses features learnt from past training.

[![Build Status](https://travis-ci.org/abhishekraok/LayeredNeuralNetwork.svg?branch=master)](https://travis-ci.org/abhishekraok/LayeredNeuralNetwork)

This is implementation of Layered Neural Network (LNN) first proposed in this [MS Thesis](https://etda.libraries.psu.edu/catalog/26405)

The difference between normal classifier and this one is this is not reset after every training. 
Here every training task has a label (like class name). 
Once a mapping from input to output is learnt, this is used as a feature for the subsequent task.
![LNN structure](https://artmapstore.blob.core.windows.net/firstnodes/photos/LNN.PNG)


## API 

- **Fit(X,Y, label)** : Trains the model using data X for class named "label". 
Y is binary indicating presence/absence of class. 
X is a numpy matrix of size (samples, features), Y is a numpy array of integers with values 0,1. label is a string.
- **Predict(X, label)** : Predicts presence or absence of class "label" in X.
- **Identify(X)** : Guesses the best class for X



## To Do

- Normalize input
- Create notebooks
- __str__
- plot [classifier like scikit-learn](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)


## Done 
- naming nodes incrementally
