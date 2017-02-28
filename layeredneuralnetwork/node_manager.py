import node
import utilities

input_node_base_name = 'input_'


class NodeManager:
    def __init__(self, input_dimension):
        """

        :type input_dimension: int
        """
        self.node_name_to_node = {}  # type: Dict from str to node.Node
        self.input_dimension = input_dimension
        self.input_nodes = []  # type: List of node.Node
        for i in range(input_dimension):
            input_node = node.Node.create_input_node(name=input_node_base_name + str(i), node_manager=self)
            self.node_name_to_node[input_node.name] = input_node
            self.input_nodes.append(input_node)

    def get_output(self, X, node_name):
        """
        Activates node with name node_name, and gets it's output.

        :param X: numpy array of size (samples, input_dimension)
        :type node_name: str
        :return: numpy array of size (samples)
        :rtype: np.array
        """
        if not self.has_node_name(node_name):
            raise ValueError('Node name ' + node_name + ' not found in node manager')
        utilities.check_2d_shape(X, self.input_dimension)
        self.deactivate_all()
        self.activate_input(X)
        return self.node_name_to_node[node_name].get_output(X)

    def activate_input(self, X):
        for i, input_node in enumerate(self.input_nodes):
            input_node.set_output(X[:, i])

    def get_input_names(self):
        return [i.name for i in self.input_nodes]

    def add_node(self, node):
        """"
        :type node: node.Node
        """
        if node.name in self.node_name_to_node:
            raise ValueError('Node already exists: Name ' + node.name)
        self.node_name_to_node[node.name] = node

    def deactivate_all(self):
        for node in self.node_name_to_node.values():
            node.deactivate()

    def has_node_name(self, label):
        return label in self.node_name_to_node

    def get_weight(self, node_name):
        node = self.node_name_to_node[node_name]
        return node.transform_function.get_weights()
