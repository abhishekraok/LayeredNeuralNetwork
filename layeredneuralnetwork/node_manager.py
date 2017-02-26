import node

input_node_base_name = 'input_'


class NodeManager:
    def __init__(self, input_dimension):
        """

        :type input_dimension: int
        """
        self.node_name_to_node = {}  # type: Dict from str to node.Node
        self.input_dimension = input_dimension
        for i in range(input_dimension):
            input_node = node.Node.create_input_node(name=input_node_base_name + str(i), node_manager=self)
            self.node_name_to_node[input_node.name] = input_node

    def get_output(self, X, node_name):
        """
        Activates node with name node_name, and gets it's output.

        :param X: numpy array of size (samples, input_dimension)
        :type node_name: str
        :return: numpy array of size (samples)
        :rtype: np.array
        """
        if node_name not in self.node_name_to_node:
            raise ValueError('Node name ' + node_name + ' not found in node manager')
        return self.node_name_to_node[node_name].get_output(X)

