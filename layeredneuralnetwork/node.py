import node_manager
import transform_function

class Node:
    def __init__(self, name, input_names, transform_function, node_manager):
        """
        Individual nodes in LNN

        :type name: str
        :type input_names: list of str
        :type transform_function: transform_function.TransformFunction
        :type node_manager: node_manager.NodeManager
        """
        self.name = name
        self.input_names = input_names
        self.transform_function = transform_function
        self.is_activated = False
        self.output = None
        self.node_manager = node_manager

    def get_output(self, X):
        pass
