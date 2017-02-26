import node_manager
import transform_function




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

    def get_output(self, X):
        pass

    @staticmethod
    def create_input_node(name, node_manager):
        return Node(name=name, input_names=None, transform_function=None,
                    node_manager=node_manager, is_input=True)
