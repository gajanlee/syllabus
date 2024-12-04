
##########################################################################################

##############       生成摘要               ##############################################
# 1. 重要度–
# 2. 相关度

##########################################################################################

####################################################

### 句子排序

######################################################

# 对句子进行信息量排序
# 方案1: 直接用重要度排序
# 方案2: PageRank类方法

# 所有包含基础概念的句子，进行排序
# TODO: 移除所有已经选过的句子，并降权
# TODO: 写一个句子打印函数，输出加粗关键概念

import copy
from datetime import datetime
from pathlib import Path

from summarization_interface import Material
from dependency import construct_dependency_foreset, DependencyForest
from summarization_interface import (
    get_concept_importance, extract_common_forest,
    find_core_concepts, find_core_concepts_funcs,
    construct_dependency_matrix, dependency_matrix_funs,
    get_idf_value, LinkType
)
from utils import *
from tqdm import tqdm
from evaluation import compose_row, tabulate, word_limit
    

data_set_dir = Path(__file__).parent.parent.parent / 'datasets'


from enum import Enum
class NodeSource(Enum):
    """Docstring for MyEnum."""
    HISTORY = "some_value"
    COMMON = "some_other_value"

DependencyPair = ('', ('pre', 'post', 'weight'))

def split_nodes(common_dependency_forest, history_dependency_forest, core_concepts):
    '''
    按照核心概念，分割成history，以及链接到不同core_concepts的linking
    '''
    history_concepts, main_concepts = set(), set()
    
    for node in common_dependency_forest.string_to_node:
        # 优先判断，出现在历史中的，肯定是历史概念
        if node in history_dependency_forest.string_to_node:
            history_concepts.add(node)
        else:
            main_concepts.add(node)
    
    history_pairs = []
    # TODO: 最小生成树，按照权重，遍历history_forest
    for pre, post in common_dependency_forest.pairs:
        pre, post = pre.content, post.content
        if pre in history_concepts and post in history_concepts:
            weight = (get_idf_value(pre) + get_idf_value(post)) / 2
            history_pairs.append((pre, post, weight))
    
    linking_main_pairs = []
    linking_pairs_list, main_pairs_list = [], []
    
    # TODO: 聚类删除abstract类概念
    for core_concept in core_concepts[:5]:
        linking_pairs = common_dependency_forest.get_node_sequence('', to_concept=core_concept, forward=False)
        main_pairs = common_dependency_forest.get_node_sequence(core_concept, '', forward=True)
        
        # 
        # TODO: 移除相似的前序节点，和相同的后序节点
        # TODO: 已经出现过的词，不再出现
        # TODO: pairs长度根据句子数量动态调整，目前还是选择第一个序列
        # TODO-latest:把linking_pairs和main_pairs转为(pre, post, weight)的形式
        filtered_linking_pairs = []
        seq = linking_pairs[0]
        pre = seq[0].content
        for post in seq[1:]:
            post = post.content
            weight = (get_idf_value(pre) + get_idf_value(post)) / 2
            filtered_linking_pairs.append((pre, post, weight))
            pre = post
            
        filtered_main_pairs = []
        seq = main_pairs[0]
        pre = seq[0].content
        for post in seq[1:]:
            post = post.content
            weight = (get_idf_value(pre) + get_idf_value(post)) / 2
            filtered_main_pairs.append((pre, post, weight))
            pre = post
        
        linking_main_pairs.append((core_concept, filtered_linking_pairs, filtered_main_pairs))
    
    # TODO: linking和main pairs的序列长度，选择最贴近目标句子数量的
    return sorted(history_pairs, key=lambda x: -x[2]), linking_main_pairs


def summarize(sentences, common_dependency_forest, history_dependency_forest, core_concepts, n = 50):
    '''
    n: 句子数量
    '''
    # TODO: 计算，分配不同部分的权重，给定多少个句子
    
    # 按照权重/相关性顺序把dependency pair列出来，保留权重
    history_pairs, linking_main_pairs = split_nodes(common_dependency_forest, history_dependency_forest, core_concepts)
    
    sentences = copy.deepcopy(sentences)
    
    print('ranking background sentences')
    summary_sentences = []
    summary_sentences += rank_sentences(sentences, history_pairs, history_dependency_forest)
    # 去除已经选择的句子
    for sentence in summary_sentences:
        sentences.remove(sentence)
    
    for core_concept, linking_pairs, core_pairs in linking_main_pairs:
        # TODO: 用模板句子，format
        summary_sentences += [core_concept]
        
        print(f'ranking linking sentences of {red_text(core_concept)}')

        # 链接概念句子
        ranked_sentences = rank_sentences(sentences, linking_pairs, common_dependency_forest, n = 5)
        for rank_sentence in ranked_sentences:
            summary_sentences.append(rank_sentence)
            sentences.remove(rank_sentence)
        
        print(f'ranking main sentences of {red_text(core_concept)}')

        # 主要概念句子
        ranked_sentences = rank_sentences(sentences, core_pairs, common_dependency_forest, n = 5)
        for rank_sentence in ranked_sentences:
            summary_sentences.append(rank_sentence)
            sentences.remove(rank_sentence)

    return summary_sentences

def rank_sentences(sentences, dependency_pairs, dependency_forest, n = 10):
    '''
    返回历史概念
    
    dependency_pairs，也可以是dependency_forest，目标是找到的句子尽可能覆盖掉所有的dependency关系
    '''
    concepts = set()
    for pre, post, weight in dependency_pairs:
        concepts.update([pre, post])

    latter_concepts = {c for c in concepts}
    history_concepts = set()
    
    # TODO: 选择n个最重要的dependency_pairs
    filtered_dependency_pairs = sorted(dependency_pairs, key=lambda x: -x[2])[:n]
    selected_sentences = []
    
    for pre, post, weight in filtered_dependency_pairs:        
        if pre not in history_concepts:
            latter_concepts.remove(pre)
        if post not in history_concepts:
            latter_concepts.remove(post)

        # TODO: 选择最重要的句子
        importance_list = []
        for sentence in sentences:
            if sentence in selected_sentences:
                continue
            importance = get_importance_of_a_sentence(sentence, history_concepts, latter_concepts, dependency_forest)
            importance_list.append((sentence, importance))
            
        s_sentence = sorted(importance_list, key=lambda p: -p[1])[0][0]
        selected_sentences.append(s_sentence)

        history_concepts.update([pre, post])

    return selected_sentences

def get_importance_of_a_sentence(sentence, picked_concepts, latter_concepts, dependency_forest):
    '''
    返回重要度，重要的半句
    '''
    word_set = set()
    for pre, ind, rtype, post, position in sentence.links:
        # TODO: 还有很多词不在dependency_foreset中，怎么处理？目前是删掉所有不在dependency_forest中的词，应该构建一颗更完整的大树
        # word_set.update([pre, post])
        if pre in dependency_forest.string_to_node:
            word_set.add(pre)
        if post in dependency_forest.string_to_node:
            word_set.add(post)
        
        # TODO: action link特殊处理，因为ind不在string_to_nodes中，无法获取pair权重
        # if rtype == LinkType.Action:
        #     word_set.add(ind)

    score = 0
    for word in word_set:
        # 词本身的重要度（IDF*action数量），与前文的紧密程度，与后文的紧密程度
        # TODO: 权重衰减，越远的picked和latter，权重越低
        basic_importance = get_idf_value(word) # * importances[word]
        for picked in picked_concepts:
            basic_importance += get_importance_of_pair(picked, word, dependency_forest)
        
        for latter in latter_concepts:
            basic_importance += get_importance_of_pair(word, latter, dependency_forest)

        score += basic_importance

    if not word_set:
        return 0

    return score / len(word_set)


def get_importance_of_pair(pre, post, dependency_forest):
    sequences = dependency_forest.get_node_sequence(pre, post)
    
    # 最短路径的长度，分之1，作为pair的权重
    if sequences:
        return 1 / len(sorted(sequences, key=lambda s: len(s))[0])
    return 0


def load_med_rag():
    '''
    TODO: 没有拆分summary和正文
    TODO: 历史材料聚合
    '''    
    med_rag_dir = data_set_dir / '/med_rag_textbooks/'
    materials, references = [], []
    for sentence_file in tqdm((med_rag_dir / 'sentences').glob('*'), desc='loading med rag'):
        if sentence_file.name not in ['Anatomy_Gray', 'Biochemistry_Lippincott', 'First_Aid_Step1', 'First_Aid_Step2', 'Pathoma_Husain', 'Pediatrics_Nelson', 'Psichiatry_DSM-5']:
            continue

        material = Material(
            sentence_file.name,
            (med_rag_dir / f'links/{sentence_file.name}'),
            (sentence_file),
        )

        reference_sents = []
        for index, line in enumerate(sentence_file.read_text(encoding='utf-8').split('\n')):
            if not line: continue
            
            if index < 100: reference_sents.append(line)
            else: break

        reference = ' '.join(reference_sents)
        materials.append(material)
        references.append(reference)

    output_dir = med_rag_dir / 'wsln_output/'

    return materials, references, output_dir

def load_rsm():
    rsm_dir = data_set_dir / 'rsm'

    reference_sents = []
    for line in (rsm_dir / 'sentences/rsm').read_text(encoding='utf-8').split('\n'):
        if not line: continue
        text, position = line.split('\t')
        if position.startswith('preface'):
            reference_sents.append(text)

    return [Material('rsm', rsm_dir / 'links/rsm', rsm_dir / 'sentences/rsm')], ' '.join(reference_sents), rsm_dir / 'wsln_output'


def wsln_summarize(main_material, history_material):
    print('extracting core concepts')
    core_concepts = find_core_concepts(
        main_material.sentences,
        find_core_concepts_funcs['action_link_count'],
        150
    )

    print(f'core concepts: {core_concepts}')

    # core_concepts += extend_core_concepts()
    # core_concepts = compound_concepts(sentences, core_concepts)

    print('constructing dependency matrix')

    # 计算依赖值
    # TODO: 可以考虑多保留一些concepts，比实际core_concepts数量多
    main_dependency_matrix = construct_dependency_matrix(main_material.links, dependency_matrix_funs['avg_idf'], core_concepts)
    
    # TODO: 去除值为0的
    main_dependency_forest = construct_dependency_foreset(main_dependency_matrix)
    
    print('dependency forest: ')

    print('loading history data')

    history_dependency_matrix = construct_dependency_matrix(history_material.links, dependency_matrix_funs['avg_idf'], core_concepts)
    history_dependency_forest = construct_dependency_foreset(history_dependency_matrix)
    
    common_history_forest = extract_common_forest(main_dependency_forest, history_dependency_forest)
    
    # importance_mapper = get_concept_importance(common_history_forest)
    
    print('summarizing')

    # TODO: 前几个core_concepts的确定，聚类分解
    sentences = summarize(main_material.sentences, common_history_forest, history_dependency_forest, core_concepts, n = 50)

    summary_sents = []

    for index, sentence in enumerate(sentences):
        if type(sentence) is str:
            print(red_text(sentence))
        else:
            print(green_text(index), sentence.text)
            summary_sents.append(sentence.text)

    return ' '.join(summary_sents)


if __name__ == '__main__':
    materials, references, output_base_dir = load_rsm()
    # materials, references, output_base_dir = load_med_rag()
    references = [word_limit(r) for r in references]
    model_name = f"wsln_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    output_dir = output_base_dir / datetime.now().strftime('%y%m%d_%H%M%S')
    output_dir.mkdir()

    history_material = Material('foundations_of_database', data_set_dir / 'rsm/foundations_of_database/links/foundations_of_database', data_set_dir / 'rsm/foundations_of_database/sentences/foundations_of_database')

    predictions = []
    for material, reference in tqdm(zip(materials, references), total=len(materials)):
        prediction = wsln_summarize(material, history_material)
        predictions.append(prediction)
        (output_dir / material.name).write_text(prediction, encoding='utf-8')

    row, headers = compose_row(predictions, references, model_name)
    table = tabulate([row], headers = headers, tablefmt = 'fancy_grid')
    print(table)