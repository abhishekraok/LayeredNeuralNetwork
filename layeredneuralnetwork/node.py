import numpy as np

try:
    from layeredneuralnetwork import node_manager
except ImportError:
    import node_manager
from layeredneuralnetwork import transform_function


class Node:
    def __init__(self, name, input_names, transform_function, node_manager, is_input):
        """
        Individual nodes in LNN

        :type name: str
        :type input_names: list of str
        :type transform_function: transform_function.TransformFunction
        :type node_manager: node_manager.NodeManager
        :type is_input: bool
        """
        self.name = name
        self.input_names = input_names
        self.transform_function = transform_function
        self.is_activated = False
        self.output = None
        self.node_manager = node_manager
        self.is_input = is_input
        self.transform_function_input_dimension = len(input_names)

    def get_output(self, X):
        """
        Activates node and transforms X.

        :param X: numpy array of dimension (samples, input_dimension)
        :type X: np.array
        :rtype: np.array
        """
        if self.is_activated:
            if self.output is None:
                raise Exception('Node ' + self.name + ' is activated but output is None')
            return self.output
        sample_size = X.shape[0]
        features = np.zeros(shape=[sample_size, self.transform_function_input_dimension])
        for column, node_name in enumerate(self.input_names):
            features[:, column] = self.node_manager.get_output(X, node_name)
        calculated_output = self.transform_function.transform(features)
        self.set_output(calculated_output)
        return self.output

    def set_output(self, output):
        self.output = output
        self.is_activated = True

    def deactivate(self):
        self.output = None
        self.is_activated = False

    @staticmethod
    def create_input_node(name, node_manager):
        return Node(name=name, input_names=[], transform_function=None,
                    node_manager=node_manager, is_input=True)
