# LayeredNeuralNetwork
Neural Network that uses features learnt from past training.

This is implementation of Layered Neural Network first proposed in this (MS Thesis)[https://etda.libraries.psu.edu/catalog/26405]

This is a transfer learning technique where learned output from past training is used as feature for next task.









## API 

1. Train(X,Y, label) : Trains the model using data X for class named "label". Y is binary indicating presence/absence of class. X is a numpy matrix of size (samples, features), Y is a numpy array of integers with values 0,1. label is a string.
2. Predict(X, label) : Predicts presence or absence of class "label" in X.
3. Identify(X) : Guesses the best class for X
