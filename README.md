# LayeredNeuralNetwork
Neural Network that uses features learnt from past training.

This is implementation of Layered Neural Network (LNN) first proposed in this [MS Thesis](https://etda.libraries.psu.edu/catalog/26405)

This is a transfer learning technique where learned output from past training is used as feature for next task.
![LNN structure](https://artmapstore.blob.core.windows.net/firstnodes/photos/LNN.PNG)


## API 

1. Train(X,Y, label) : Trains the model using data X for class named "label". Y is binary indicating presence/absence of class. X is a numpy matrix of size (samples, features), Y is a numpy array of integers with values 0,1. label is a string.
2. Predict(X, label) : Predicts presence or absence of class "label" in X.
3. Identify(X) : Guesses the best class for X


## Classes

####Layered Node
A layered node (LNode) is the most basic unit of computation inside LNN. 
It has a single property **Input Dimension** and a single method ** get_output(X) ** Where X is a matrix of size [samples, input dimension]. 
LNode needs to know two things. How to transform the input X into features and how to transform these features into output.
