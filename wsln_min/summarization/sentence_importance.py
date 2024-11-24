
def get_importance_of_pair(pre_node, post_node, links):
    '''
    计算两个节点的顺序重要度
    路径的可达性，考虑到W-SLN在构建过程中已经考虑了依赖关系，所以只考虑出链，可达的长度
    '''
    dp = {}

    for pre_node, post_node in pairs:
        pass


def rank_sentences(sentences, dependency_pairs, n = 10):
    '''
    返回历史概念
    '''
    import copy
    concepts = set()
    for pre, post in dependency_pairs:
        concepts.update([pre, post])

    latter_concepts = {c for c in concepts}
    history_concepts = set()
    for pre, post in dependency_pairs:
        latter_concepts.remove(pre)
        latter_concepts.remove(post)

        # TODO: 选择最重要的句子
        get_importance_of_a_sentence(sentence.links, history_concepts, latter_concepts)

        history_concepts.update(pre, post)

    return history_concepts


def cluster_forest(n):
    '''相似的概念聚合到一起，主要利用到abstract link
    '''
    
def decompose_forest(n):
    '''
    要求尽可能包含更多的concepts
    返回核心概念，下属概念，和
    '''

    return core_concepts, 

def summarization_framework(sentences, history_dependency_matrix, common_dependency_forest, core_dependency_forest):
    # 分成多个路径
    core_concepts = decompose_forest(core_dependency_forest, n=4)

    rank_sentences(sentences, dependency_pairs, n = 10)

