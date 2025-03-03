import numpy as np
from summarization_interface import LinkType

class DependencySLN:

    def __init__(self, d = 0.85, alpha = 0.95, max_iter = 100):
        '''
        d: (1-d)为概率转移矩阵中，添加全局的概率
        alpha: (1-alpha)为二阶概率转移矩阵的概率
        '''
        self.node_to_id = {}
        self.id_to_node = {}
        self.dep_list = []
        self.d = d
        self.alpha = alpha
        self.max_iter = max_iter
        
    def append(self, pre, post, rtype):
        self.dep_list.append((pre, post, rtype))
        for node in (pre, post):
            if node not in self.node_to_id:
                _id = len(self.node_to_id)
                self.node_to_id[node] = _id
                self.id_to_node[_id] = node
    
    def build(self):
        self.matrix = np.zeros((len(self.node_to_id), len(self.node_to_id)))
        for pre, post, rtype in self.dep_list:
            self.matrix[self.node_to_id[pre]][self.node_to_id[post]] += 1

        P = (self.d * self.matrix + (1 - self.d) / len(self.node_to_id))
        np.fill_diagonal(P, 0)
        P /= np.sum(P, axis=1, keepdims=True)

        last_G = P
        G = P

        for index in range(self.max_iter):
            two_stage = np.matmul(G, G)
            np.fill_diagonal(two_stage, 0)
            G = self.alpha * G + (1 - self.alpha) * two_stage
            # calculate loss
            loss = np.sum(abs(G - last_G))
            print(f'{index}:{loss:.5f}')
            last_G = G
        
        # TODO: norm G, 每行的概率和都是1
        np.fill_diagonal(G, 0)
        G /= np.sum(G, axis=1, keepdims=True)
        self.prob_transfer_matrix = G

def construct_dependency_sln(sentences, core_concepts, clusters):
    '''
    return
        - pairs: 根据规则挖掘到的dependency
    '''
    pairs = []
    for index, sentence in enumerate(sentences):
        active_node = None
        for pre, ind, rtype, post, position in sentence.links:
            if pre not in core_concepts or post not in core_concepts:
                continue
            
            # Action link
            if rtype in [LinkType.Action, LinkType.Attribute]:
                active_node = pre
                pairs.append((pre, post, rtype.value))
            elif active_node:
                pairs.append((active_node, pre, 'action'))
                pairs.append((active_node, post, 'action'))
                
            # CauseEffect link
            
    for key, sub_concepts in clusters.items():
        if key not in core_concepts: continue
        for concept in sub_concepts:
            if concept not in core_concepts: continue
            pairs.append((key, concept, 'abstract'))
            
    dep_sln = DependencySLN()
    for pre, post, rtype in pairs:
        if pre in core_concepts and post in core_concepts:
            dep_sln.append(pre, post, rtype)        
    dep_sln.build()

    return dep_sln, pairs

def test_dependency_sln():
    dep_sln = DependencySLN()
    dep_sln.append('A', 'B')
    dep_sln.append('B', 'C')
    dep_sln.append('B', 'D')
    dep_sln.append('D', 'C')
    dep_sln.append('A', 'E')
    dep_sln.append('E', 'C')
    
    dep_sln.build()
    
if __name__ == '__main__':
    test_dependency_sln()
