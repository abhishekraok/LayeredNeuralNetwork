import node

class NodeManager:
    def __init__(self, node_name_to_node):
        if node_name_to_node:
            self.node_name_to_node = node_name_to_node
        else:
            self.node_name_to_node = {}