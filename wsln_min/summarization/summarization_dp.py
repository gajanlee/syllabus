'''
设计摘要的动态规划算法
'''
import numpy as np
import re
import math
from summarization_interface import split_nodes
from utils import red_text

# def summarize_dp(sentences, )

def construct_model_matrix(sentences, sampled_paths, importance_mapper, group_n=300):
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
            # i和j离得越近，概率越高
            if i == j:
                transfer_prob_matrix[i][j] = 0
            else:
                transfer_prob_matrix[i][j] = np.exp(-abs(i-j))
                # transfer_prob_matrix[i][j] = (count - abs(i - j)) / (count ** 2)

        transfer_prob_matrix[i] /= np.sum(transfer_prob_matrix[i])

            # numerator = np.exp(-(j - i))
            # denominator = np.sum([np.exp(-(k - i)) for k in range(i + 1, count)])
            # transfer_prob_matrix[i][j] = numerator / denominator

    # 每个link group与"依赖对"之间的相关性
    link_dep_prob_matrix = np.zeros((count, len(sampled_paths)))
    for link_i, link_group in enumerate(link_groups):
        link_nodes = {node for _pre, _, _, _post, _ in link_group for node in [_pre, _post]}
        link_words = {word for _pre, _, _, _post, _ in link_group for node in [_pre, _post] for word in re.split('[ \t]', node)} - {'to', 'with', 'of', 'in', 'at', 'on', 'from', 'for', 'as'}
        probs = []
        for path in sampled_paths:
            # 计算相关性
            # dependency_words = set()
            # for node in path:
            #     # 分隔符为\t和空格
            #     dependency_words.update(re.split('[ \t]', node))
                
            # dependency_words -= {'to', 'with', 'of', 'in', 'at', 'on', 'from', 'for', 'as'}

            # link_words = []
            # for _pre, ind, rtype, _post, position in link_group:
            #     link_words += re.split('[ \t]', _pre) + re.split('[ \t]', _post)
            
            dependency_nodes = {node for node in path}
            dependency_words = {word for node in path for word in re.split('[ \t]', node)} - {'to', 'with', 'of', 'in', 'at', 'on', 'from', 'for', 'as'}
            
            prob = sum(importance_mapper.get(node, 0) for node in dependency_nodes & link_nodes) * 5 + 1 #  / len(dependency_nodes | link_nodes)
            # prob = len(dependency_words & link_words) * 100 #  / len(dependency_nodes | link_nodes)
            probs.append(prob)

            # TODO: 可以引入weight，计算杰卡德距离
            # 计算召回率
            # probs.append(
            #     sum(link_words.count(dw) + 1 for dw in dependency_words)
            # )
            
        # 归一化概率
        for dep_j in range(len(sampled_paths)):
            if sum(probs) == 0:
                link_dep_prob_matrix[link_i][dep_j] = 1 / len(sampled_paths)
            else: 
                link_dep_prob_matrix[link_i][dep_j] = probs[dep_j] / sum(probs)
                
    # TODO: 初始化概率，越早的dependency pair概率越大
    # 目前是均等概率
    init_probs = link_dep_prob_matrix[:, dep_j]
    # [1 / len(link_groups) for _ in range(len(link_groups))]

    return link_groups, transfer_prob_matrix, link_dep_prob_matrix, init_probs


# def tranverse_path_approximate(link_groups, dependency_pairs_list, transfer_matrix, link_concept_matrix, observed_dependency, ):
#     for group in link_groups:
#     return 
    


def traverse_max_path(link_groups, dependency_pairs_list, transfer_matrix, link_concept_matrix, init_probs, observed_dependency_sequence):
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
                # 魔改：降低其他节点到max_index的概率值
                transfer_matrix[:, max_index] *= 0.3

                link_probs[time_index][link_i_index] = probs[max_index] * link_concept_matrix[link_i_index][dependency_index]

                max_sequence[time_index][link_i_index] = max_index

    sequence = [None for _ in range(len(observed_dependency_sequence))]
    sequence[-1] = np.argmax(link_probs[len(observed_dependency_sequence) - 1])
    
    for time_index in range(len(observed_dependency_sequence) - 2, -1, -1):
        sequence[time_index] = max_sequence[time_index + 1][sequence[time_index + 1]]

    return [link_groups[index] for index in sequence]


def test_traverse_max_path():
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

    sequence = traverse_max_path(link_groups, ['红', '白'], transfer_matrix, map_prob_matrix, init_probs, ['红', '白', '红'])
    assert sequence == ['盒子3', '盒子3', '盒子3']
    print('test ok')


def sample_path(dependency_sln, init_prob, path_node_count):
    path = []
    for _ in range(path_node_count):
        curr_node_idx = np.random.choice(len(dependency_sln.id_to_node), size=1, p=init_prob)[0]
        path.append(dependency_sln.id_to_node[curr_node_idx])
        # print(dependency_sln.id_to_node[curr_node_idx], init_prob[curr_node_idx])
        init_prob = dependency_sln.prob_transfer_matrix[curr_node_idx]

    return path


def summarize_markov(main_material,
                    main_dependency_sln, importance_mapper, total_sentence_n = 50, top_n = 5, rule='base'):
    
    # TODO: 每个path多少个句子？
    sampled_paths = []
    sentence_per_path = 1 if len(main_material.sentences) > 3000 else 2
    
    sentences = main_material.sentences
    
    # 基础概念优先，按照入度（列和）排序
    if rule == 'base':
        init_probs = np.sum(main_dependency_sln.prob_transfer_matrix, axis=0)
        init_probs /= np.sum(init_probs)
        
    elif rule == 'important':
        init_probs = np.zeros(len(main_dependency_sln.node_to_id))
        
        # for concept in {'resource', 'coordinate', 'node', 'storing', 'axis', 'proces', 'network', 'theorem', 'peer', 'chapter', 'rsm', 'resource space model', 'user', 'operation', 'resource space'}:
        for concept in {'rsm', 'resource space model',  'resource space', 'semantic link', 'semantic link network', 'sln'}:
            _id = main_dependency_sln.node_to_id[concept]
            
            init_probs[_id] = 1
            
        init_probs /= np.sum(init_probs)
        
    path_count = math.ceil(1.5 * total_sentence_n / sentence_per_path)
    path_node_count = 5 # 每个依赖路径有多少个节点
    # 句子数量决定扩散的强度
    # alpha = path_count / len(main_dependency_sln.node_to_id)
    alpha = 0.3
    for _ in range(path_count):
        sampled_path = sample_path(main_dependency_sln, init_probs, path_node_count)
        sampled_paths.append(sampled_path)
        # 调整下一次采样概率
        # 相关节点概率上升
        init_probs = np.matmul(init_probs, main_dependency_sln.prob_transfer_matrix)
        
        # 选中节点的概率都下降
        for node in sampled_path:
            init_probs[
                main_dependency_sln.node_to_id[node]
            ] *= alpha

        init_probs /= np.sum(init_probs)
    
    link_groups, transfer_prob_matrix, link_dep_prob_matrix, init_probs = construct_model_matrix(sentences, sampled_paths, importance_mapper)
    observed_dependency_sequence = sampled_paths
    # 类似注意力机制地展示
    res_link_groups = traverse_max_path(link_groups, sampled_paths, transfer_prob_matrix, link_dep_prob_matrix, init_probs, observed_dependency_sequence)
    
    main_material.position_sentence_mapper
    structured_summary = []
    summary_sentences = []
    selected_positions = []
    for group, path in zip(res_link_groups, sampled_paths):
        # 按照action type对
        position_to_nodes = {}
        for pre, ind, rtype, post, position in group:
            position_to_nodes[position] = position_to_nodes.get(position, set()) | {pre, post}
        
        path_summarys = []
        # 用importance mapper评价相关性
        group_sents = []
        for position, _ in sorted(position_to_nodes.items(), key=lambda p_nodes: sum(importance_mapper.get(node, 1) if node in path else 0 for node in p_nodes[1]), reverse=True):
            if position in selected_positions:
                continue
            # 每个group只选取sentence_per_path个句子
            if len(group_sents) > sentence_per_path:
                break
            group_sents.append(position)
            
            summary_sentences.append(main_material.position_sentence_mapper[position].text)
            selected_positions.append(position)
            path_summarys.append((position, main_material.position_sentence_mapper[position]))
        structured_summary.append(
            (path, path_summarys)
        )
        
        if len(selected_positions) > total_sentence_n:
            break
    
    return ' '.join(summary_sentences), structured_summary


if __name__ == '__main__':
    test_traverse_max_path()
