'''
设计摘要的动态规划算法
'''
import numpy as np
import re
from summarization_interface import split_nodes
from utils import red_text

# def summarize_dp(sentences, )

def construct_model_matrix(sentences, dependency_pairs_list, group_n=300):
    '''
    按照距离

    dependency_pairs: 每个元素是一个列表[(pre, post, weight), (pre, post, weight), ...]
    '''
    all_links = []
    for index, sentence in enumerate(sentences):
        if not sentence.links: continue
        all_links += sentence.links
        # link_groups.append(sentence.links)
    count_per_group = len(all_links) // (group_n) + 1
    link_groups = []
    for index in range(0, len(all_links), count_per_group):
        link_groups.append(all_links[index: index+count_per_group])

    count = len(link_groups)

    # link group之间的概率转移矩阵，按照指数倒数概率，softmax
    transfer_prob_matrix = np.zeros((count, count))
    for i in range(count):
        for j in range(count):
        # for j in range(i + 1, count):
            if i == j:
                transfer_prob_matrix[i][j] = 0
            else:
                transfer_prob_matrix[i][j] = (count - abs(i - j)) / (count ** 2)

            # numerator = np.exp(-(j - i))
            # denominator = np.sum([np.exp(-(k - i)) for k in range(i + 1, count)])
            # transfer_prob_matrix[i][j] = numerator / denominator

    # 每个link group与"依赖对"之间的相关性
    link_dep_prob_matrix = np.zeros((count, len(dependency_pairs_list)))
    for link_i, link_group in enumerate(link_groups):
        probs = []
        for dependency_pairs in dependency_pairs_list:
            # 计算相关性
            dependency_words = []
            for pre, post, weight in dependency_pairs:
                dependency_words.extend(re.split('[ \t]', pre) + re.split('[ \t]', post))
            
            dependency_words = pre.split(' ') + post.split(' ')
            link_words = []
            for _pre, ind, rtype, _post, position in link_group:
                link_words += re.split('[ \t]', _pre) + re.split('[ \t]', _post)

            # TODO: 可以引入weight，计算杰卡德距离
            probs.append(
                sum(link_words.count(dw) + 1 for dw in dependency_words) / len(link_words)
            )

        # 归一化概率
        for dep_j in range(len(dependency_pairs_list)):
            link_dep_prob_matrix[link_i][dep_j] = probs[dep_j] / sum(probs)

    # TODO: 初始化概率，越早的dependency pair概率越大
    # 目前是均等概率
    init_probs = link_dep_prob_matrix[:, dep_j]
    # [1 / len(link_groups) for _ in range(len(link_groups))]

    return link_groups, transfer_prob_matrix, link_dep_prob_matrix, init_probs


def tranverse_max_path(link_groups, dependency_pairs_list, transfer_matrix, link_concept_matrix, init_probs, observed_dependency_sequence):
    '''
    dependency_pairs_list
    link_transfer_matrix: 代表两个语义链之间的转换概率，即选定某个语义链后，另一个语义链被选中的概率
    link_concept_matrix: 代表每个语义链内包含某个concept的概率
    observed_dependency_sequence: 观测变量的序列
    '''
    link_probs = np.zeros((len(observed_dependency_sequence), len(link_groups)))

    max_sequence = np.zeros((len(observed_dependency_sequence), len(link_groups)), dtype=np.int32)

    for time_index, dependency in enumerate(observed_dependency_sequence):
        for link_i_index in range(len(link_groups)):
            dependency_index = dependency_pairs_list.index(dependency)
            if time_index == 0:
                link_probs[time_index][link_i_index] = (
                    init_probs[link_i_index] * link_concept_matrix[link_i_index][dependency_index]
                )
                # max_sequence[time_index][link_i_index] = 0
            else:
                # 计算上一个节点的所有状态到这个节点的最优值
                probs = []
                for _link_j_index in range(len(link_groups)):
                    probs.append(
                        link_probs[time_index - 1][_link_j_index] * transfer_matrix[_link_j_index][link_i_index])

                max_index = np.argmax(probs)

                link_probs[time_index][link_i_index] = probs[max_index] * link_concept_matrix[link_i_index][dependency_index]

                max_sequence[time_index][link_i_index] = max_index

    sequence = [None for _ in range(len(observed_dependency_sequence))]
    sequence[-1] = np.argmax(link_probs[len(observed_dependency_sequence) - 1])
    
    for time_index in range(len(observed_dependency_sequence) - 2, -1, -1):
        sequence[time_index] = max_sequence[time_index + 1][sequence[time_index + 1]]

    return [link_groups[index] for index in sequence]


def test_tranverse_max_path():
    link_groups = ['盒子1', '盒子2', '盒子3']
    transfer_matrix = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5],
    ])
    map_prob_matrix = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3],
    ])
    init_probs = [0.2, 0.4, 0.4]

    sequence = tranverse_max_path(link_groups, ['红', '白'], transfer_matrix, map_prob_matrix, init_probs, ['红', '白', '红'])
    assert sequence == ['盒子3', '盒子3', '盒子3']
    print('test ok')


# test_tranverse_max_path()

def summarize_markov(sentences,
            common_dependency_forest, history_dependency_forest,
            common_dependency_matrix: dict, history_dependency_matrix,
            core_concepts, importance_mapper, total_sentence_n = 50, top_n = 5):
    
    history_pairs, linking_main_pairs = split_nodes(
        common_dependency_forest, history_dependency_forest,
        common_dependency_matrix, history_dependency_matrix,
        core_concepts, top_n=top_n
    )

    



    # background
    link_groups, transfer_prob_matrix, link_dep_prob_matrix, init_probs = construct_model_matrix(sentences, history_pairs)
    # 按照概率，采样
    # TODO: 如何采样
    observed_dependency_sequence = history_pairs
    # 类似注意力机制地展示
    res = tranverse_max_path(link_groups, history_pairs, transfer_prob_matrix, link_dep_prob_matrix, init_probs, observed_dependency_sequence)
    
    # 从抽取的link groups中选择最相关的links

    for linking_pairs, main in linking_main_pairs:
        pass
    

    # TODO: 过滤语义链
    for dep, _res in zip(history_pairs, res):
        dep1, dep2, w = dep
        print(red_text(f'{dep1}-{dep2}'))
        for pre, ind, rtype, post, _ in _res:
            if dep1 in pre or dep2 in pre or dep1 in post or dep2 in post:
                print(f'{pre}-{ind}->{post}')
        print('---------------------\n\n')
    exit()

