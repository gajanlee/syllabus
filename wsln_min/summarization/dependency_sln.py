import numpy as np
from summarization_interface import LinkType

class DependencySLN:

    def __init__(self):
        self.node_to_id = {}
        self.id_to_node = {}
        self.dep_list = []
        self.matrix = None

    def append(self, pre, post):
        self.dep_list.append((pre, post))
        for node in (pre, post):
            if node not in self.node_to_id:
                _id = len(self.node_to_id)
                self.node_to_id[node] = _id
                self.id_to_node[_id] = node
    
    def build(self):
        node_set = set()
        for pre, post in self.dep_list:
            node_set.update([pre, post])

        self.matrix = np.zeros((len(self.node_to_id), len(self.node_to_id)))
        for pre, post in self.dep_list:
            self.matrix[self.node_to_id[pre]][self.node_to_id[post]] += 1

        # norms = np.linalg.norm(self.matrix, axis = 0, keepdims=True)
        # normalized = self.matrix / norms
        
        # W = []
        norms = np.linalg.norm(self.matrix, axis = 0, keepdims=True)
        P = self.matrix / norms
        np.fill_diagonal(P, 0)

        d = 0.85; alpha = 0.9; max_iter = 1000
        G = d * P + (1 - d) * np.ones()

        for _ in range(max_iter):
            G = alpha * G + (1 - alpha) * G * G

        return 

def construct_dependency_sln(sentences, core_concepts):
    dep_sln = DependencySLN()
    for index, sentence in enumerate(sentences):
        for pre, ind, rtype, post, position in sentence.links:

            # Action link
            if rtype == LinkType.Action:
                dep_sln.append(pre, post)

            # Attribute link
            if rtype == LinkType.Attribute:
                dep_sln.append(pre, post)

            # CauseEffect link
