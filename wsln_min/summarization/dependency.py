# Construct Dependency Chain
import tqdm
import copy

# dependency chain
class DependencyNode:

    def __init__(self, content: str):
        self.content = content
        self.next = []
        self.pre = []
    
    def __hash__(self):
        return hash(self.content)
    
    def __eq__(self, query):
        return self.content == query.content

    def has_next(self, query):
        # query is a chainNode
        searched = set()
        pending = {self}
        
        while pending:
            node = pending.pop()
            searched.add(node)
            
            if node == query:
                return True
            for next in node.next:
                if next not in searched:
                    pending.add(next)
        
        return False

    def __str__(self):
        return self.content
        # string = f'{self.content}'
        # strings = []
        # for node in self.next:
        #     strings.append(
        #         f'{string} ----> {str(node)}'
        #     )
        # if not strings:
        #     return string
        # return '\n'.join(strings)
        
    def __repr__(self):
        return f'ChainNode({self.content})'


class DependencyForest:

    def __init__(self):
        self.roots = []
        self.string_to_node = {}

    def _append(self, pre_node, post_node):
        # 如果存在post_node -> ... -> pre_node的依赖链路，则不添加
        if post_node.has_next(pre_node):
            return False
        
        return True

    def append(self, pair):
        # pair: (pre, post), where pre is depended on post
        pre, post = pair
        pre_node = self.search(pre)
        post_node = self.search(post)

        pre_as_root = False

        if pre_node is not None and post_node is not None:
            pass
        elif pre_node is not None and post_node is None:
            post_node = DependencyNode(post)
        elif pre_node is None and post_node is not None:
            pre_node = DependencyNode(pre)
            pre_as_root = True
        elif pre_node is None and post_node is None:
            pre_node = DependencyNode(pre)
            post_node = DependencyNode(post)
            pre_as_root = True

        # 如果存在post_node -> ... -> pre_node的依赖链路，则不添加
        if post_node.has_next(pre_node):
            return
        
        if post_node not in pre_node.next:
            pre_node.next.append(post_node)
            post_node.pre.append(pre_node)

        # pre没有依赖，添加root节点
        if pre_as_root:
            self.roots.append(pre_node)

        # post如果是root，移除
        if post_node in self.roots:
            self.roots.remove(post_node)
        
        self.string_to_node[pre_node.content] = pre_node
        self.string_to_node[post_node.content] = post_node

    def search(self, query):
        return self.string_to_node.get(query, None)

    def node_search(self, node, query):
        # print('searching, ', node.content, query)
        if node.content == query:
            return node
        for _node in node.next:
            if (result := self.node_search(_node, query)):
                return result

    def get_node_sequence(self, from_concept, to_concept, forward=True) -> list[str]:
        '''
        forward: True, search process is from `from_node` to `to_node`
            False, which is from `to_node` backward to `from_node`
        
        to_concept: if it is none, which represents the endless iteration
        return the middle concepts from `from_node` to `to_node`
        return [] if the dependency relations does not exist
        
        TODO: 遍历依赖强度顺序添加
        notice that forward=True and forward=False may be inconsistent
        '''
        if (forward and not from_concept) or (not forward and not to_concept):
            raise Exception('get node sequence invalid')
        
        from_node = self.string_to_node[from_concept] if from_concept else None
        to_node = self.string_to_node[to_concept] if to_concept else None
        
        iterated_nodes = set()
        paths = []
        
        if forward:
            queue = [(from_node, [from_node])]
            # 找到所有dependency sequence
            while queue:
                node, path = queue.pop()
                iterated_nodes.add(node)
                
                for next in node.next:
                    if next not in iterated_nodes:
                        iterated_nodes.add(node)
                        new_path = path + [next]
                        if (not to_node and len(next.next) == 0) or (to_node and next == to_node):
                            paths.append(new_path)
                        else:
                            queue.append(
                                (next, new_path)
                            )
        
        # 从后向前
        if not forward:
            queue = [(to_node, [to_node])]
            # 找到所有dependency sequence
            while queue:
                node, path = queue.pop()
                iterated_nodes.add(node)
                
                for pre in node.pre:
                    if pre not in iterated_nodes:
                        iterated_nodes.add(node)
                        
                        new_path = [pre] + path
                        if (not from_node and len(pre.pre) == 0) or (from_node and pre == from_node):
                            paths.append(new_path)
                        else:
                            queue.append(
                                (pre, new_path)
                            )
        
        # 按照path的长度，从长到短进行排序，生成摘要的时候可以动态选择，根据句子预算（sentence budget）
        return sorted(paths, key=lambda p: len(p))

    def __str__(self):
        string = ''
        for root in self.roots:
            string += str(root) + '\n==================\n'
        return string
    
    @property
    def pairs(self):
        pending_nodes = set(copy.deepcopy(self.roots))
        result_pairs = []
        iterated_nodes = set()
        while pending_nodes:
            pre = pending_nodes.pop()
            if pre in iterated_nodes:
                continue
            iterated_nodes.add(pre)
            
            for post in pre.next:
                result_pairs.append((pre, post))
                pending_nodes.add(post)

        return result_pairs

def construct_dependency_foreset(dependency_matrix):
    dependency_forest = DependencyForest()
    for index, (pair, value) in enumerate(
        tqdm.tqdm(sorted(dependency_matrix.items(), key=lambda x: -x[1]), desc='dependency forest')
    ):
        # pair[0] 依赖 pair[1]
        dependency_forest.append(pair)
        
    return dependency_forest


######################################################################

## 打印dependency tree

######################################################################


def breath_first_iteration(root):
    searched = set()
    pending = {root}
    mapper = {}
        
    while pending:
        parent = pending.pop()
        searched.add(parent)

        mapper[parent.content] = []

        for next in parent.next:
            if next not in searched and next not in pending:
                pending.add(next)
                mapper[parent.content].append(next.content)

    # print(mapper)
    # return len(searched), print_tree(root.content, mapper, '')
    return print_tree(root.content, mapper, '')
        
def print_tree(root, mapper, string, last=True, header='') -> str:
    elbow = "└──"
    pipe = "│  "
    tee = "├──"
    blank = "   "
    string += (header + (elbow if last else tee) + root) + '\n'
    children = mapper[root]
    for i, c in enumerate(children):
        string += print_tree(c, mapper, '', header=header + (blank if last else pipe), last=i == len(children) - 1)
    return string

# len(dependency_chain.chains[0])
# print(breath_first_iteration(dependency_forest.roots[0]))