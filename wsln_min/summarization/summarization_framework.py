
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
import random
import psutil
import os
import time
from datetime import datetime
from pathlib import Path

from summarization_interface import Material
from dependency import construct_dependency_foreset, DependencyForest
from summarization_interface import *
# (
#     get_concept_importance, extract_common_forest,
#     find_core_concepts, find_core_concepts_funcs,
#     construct_dependency_matrix, dependency_matrix_funs,
#     get_idf_value, LinkType,
# )
from utils import *
from tqdm import tqdm
from evaluation import compose_row, tabulate, word_limit
from summarization_dp import *
from dependency_sln import *
import pandas as pd

data_set_dir = Path(__file__).parent.parent.parent / 'datasets'


from enum import Enum
class NodeSource(Enum):
    """Docstring for MyEnum."""
    HISTORY = "some_value"
    COMMON = "some_other_value"

DependencyPair = ('', ('pre', 'post', 'weight'))

def summarize(sentences,
            common_dependency_forest: DependencyForest, history_dependency_forest,
            common_dependency_matrix: dict, history_dependency_matrix,
            core_concepts, importance_mapper, total_sentence_n = 30, top_n = 5):
    '''
    n: 句子数量
    '''
    # TODO: 计算，分配不同部分的权重，给定多少个句子
    # 预定义的结构：多个core concepts

    # 按照权重/相关性顺序把dependency pair列出来，保留权重
    history_pairs, linking_main_pairs = split_nodes(
        common_dependency_forest, history_dependency_forest,
        common_dependency_matrix, history_dependency_matrix,
        core_concepts, top_n=top_n
    )
    
    background_sentence_count = total_sentence_n // (top_n + 1)
    content_sentence_count = total_sentence_n - background_sentence_count
    pair_counts = []
    for i in range(top_n):
        core_concept, linking_pairs, main_pairs = linking_main_pairs[i]
        pair_counts.append((len(linking_pairs), len(main_pairs)))

    pair_sum = sum(v for pair in pair_counts for v in pair)
    sentence_counts = [[
            max(linking * content_sentence_count // pair_sum, 1),
            max(main * content_sentence_count // pair_sum, 1)
        ] for linking, main in pair_counts]
    sentence_counts[-1][-1] += content_sentence_count - sum(sum(c) for c in sentence_counts)

    sentences = copy.deepcopy(sentences)
    
    print('ranking background sentences')
    summary_sentences = []
    summary_sentences += rank_sentences(sentences, history_pairs, set(), history_dependency_forest, importance_mapper, n = background_sentence_count)
    # 去除已经选择的句子
    for sentence, _ in summary_sentences:
        sentences.remove(sentence)
    
    for (core_concept, linking_pairs, core_pairs), (linking_count, main_count) in zip(linking_main_pairs, sentence_counts):
        # TODO: 用模板句子，format
        summary_sentences += [core_concept]
        
        print(f'ranking linking sentences of {red_text(core_concept)}')

        # 链接概念句子
        ranked_sentences = rank_sentences(sentences, linking_pairs, {core_concept}, common_dependency_forest, importance_mapper, n = linking_count)
        summary_sentences += ranked_sentences
        summary_sentences += ['-------Main-----']

        for s, _ in ranked_sentences:
            sentences.remove(s)
        
        print(f'ranking main sentences of {red_text(core_concept)}')

        # 主要概念句子
        ranked_sentences = rank_sentences(sentences, core_pairs, {core_concept}, common_dependency_forest, importance_mapper, n = main_count)
        summary_sentences += ranked_sentences

        for s, _ in ranked_sentences:
            sentences.remove(s)

    return summary_sentences




def rank_sentences(sentences, dependency_pairs, current_concepts, dependency_forest, importance_mapper, n = 10):
    '''
    返回历史概念
    
    dependency_pairs，也可以是dependency_forest，目标是找到的句子尽可能覆盖掉所有的dependency关系
    current_concepts: 围绕该核心概念集合，进行句子组排序，即resource space model及其子概念
    '''
    
    # TODO: 选择n个最重要的dependency_pairs，要求考虑连贯性

    import random
    # TODO: 如果dependency pair不足的话，随机重复选取
    # filtered_dependency_pairs = sorted(dependency_pairs, key=lambda x: -x[2])[:n]
    # while len(filtered_dependency_pairs) < n:
    #     filtered_dependency_pairs.append(random.choice(filtered_dependency_pairs))
    
    selected_sentences = []; selected_text = set()
    selected_pairs = set(); pending_pairs = {(pre, post) for pre, post, weight in dependency_pairs}
    for _ in range(n):
        # TODO: 选择最重要的句子
        importance_list = []
        for sentence in sentences:
            # TODO: 存在句子内容重复的情况，在material端解决吧
            if sentence in selected_text:
                continue
            # 修改selected_pairs和pending_pairs
            importance, _selected_pairs, _pending_pairs, _hitting_pairs = get_importance_of_a_sentence(
                sentence, selected_pairs, pending_pairs, current_concepts)
            importance_list.append((sentence, importance, _selected_pairs, _pending_pairs, _hitting_pairs))

        # 按照hitting pairs的数量排序是不合理的
        # s_sentence, _, selected_pairs, pending_pairs, hitting_pairs = sorted(importance_list, key=lambda p: -len(p[-1]))[0]
        # 按照重要度排序
        s_sentence, _, selected_pairs, pending_pairs, hitting_pairs = sorted(importance_list, key=lambda p: -p[1])[0]
        selected_sentences.append((s_sentence, hitting_pairs))
        selected_text.add(s_sentence)

    return selected_sentences

def get_importance_of_a_sentence(sentence, selected_pairs, pending_pairs, current_concepts):
    '''
    TODO: 返回重要度，重要的半句
    TODO: 跨句子检查dependency pair

    Return Hitting Pairs
    '''
    selected_pairs = copy.copy(selected_pairs); pending_pairs = copy.copy(pending_pairs)
    hitting_pairs = set()
    score = 0; pairs = set(); pre_list = []
    for pre, ind, rtype, post, _ in sentence.links:
        # 如果在selected pair中出现，则只保留30%的评分
        # 如果在pending pair中出现，保留原分
        value = (get_idf_value(pre) + get_idf_value(ind) + get_idf_value(post))
        pre_list.append(pre)
        for _pre in pre_list:
            pairs.add((_pre, post))
        score += value

    for pair in pairs:
        pre, post = pair
        value = (get_idf_value(pre) + get_idf_value(ind) + get_idf_value(post))
        for concept in current_concepts:
            if concept in pre or concept in post:
                value *= 10

        if pair in pending_pairs:
            score += value * 5
            selected_pairs.add(pair)
            pending_pairs.remove(pair)
            hitting_pairs.add(pair)
        elif pair in selected_pairs:
            score += value * 3

    return score / len(sentence.words), selected_pairs, pending_pairs, hitting_pairs

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
    med_rag_dir = data_set_dir / 'med_rag_textbooks'
    materials, references = [], []
    for sentence_file in tqdm((med_rag_dir / 'sentences').glob('*'), desc='loading med rag'):
        # if sentence_file.name not in ['Anatomy_Gray', 'Biochemistry_Lippincott', 'First_Aid_Step1', 'First_Aid_Step2', 'Pathoma_Husain', 'Pediatrics_Nelson', 'Psichiatry_DSM-5']:
        #     continue

        material = Material(
            sentence_file.name,
            (med_rag_dir / f'links/{sentence_file.name}'),
            (sentence_file),
        )

        reference_sents = []
        for index, line in enumerate(sentence_file.read_text(encoding='utf-8').split('\n')):
            if not line: continue
            text, position = line.split('\t')
            if index < 100: reference_sents.append(text)
            else: break

        reference = ' '.join(reference_sents)
        materials.append(material)
        references.append(reference)

    output_dir = med_rag_dir / 'wsln_output/'

    return materials, references, output_dir

def load_khanacademy():
    '''
    TODO: 超过6000个句子截断
    '''
    materials, references = [], []
    khan_dir = data_set_dir / 'cosmopedia/khanacademy/'

    for sentence_file in (khan_dir / 'sentences').glob('*'):
        materials.append(Material(sentence_file.name, khan_dir / 'links' / sentence_file.name, sentence_file))

        reference_sents = []
        for index, line in enumerate(sentence_file.read_text(encoding='utf-8').split('\n')):
            if not line: continue
            text, position = line.split('\t')
            if index < 100: reference_sents.append(text)
            else: break

        reference = ' '.join(reference_sents)
        references.append(reference)
        
    output_dir = khan_dir / 'wsln_output'

    return materials, references, output_dir


def load_bigsurvey():
    materials, references = [], []
    bigsurvey_dir = data_set_dir / 'bigsurvey/MDS_truncated'

    for sentence_file in (bigsurvey_dir / 'sentences').glob('*'):
        materials.append(Material(sentence_file.name, bigsurvey_dir / 'links' / sentence_file.name, sentence_file))

        reference_sents = []
        for index, line in enumerate(sentence_file.read_text(encoding='utf-8').split('\n')):
            if not line: continue
            text, position = line.split('\t')
            if index < 100: reference_sents.append(text)
            else: break

        reference = ' '.join(reference_sents)
        references.append(reference)

    output_dir = bigsurvey_dir / 'wsln_output'

    return materials, references, output_dir


def load_rsm():
    rsm_dir = data_set_dir / 'rsm'

    reference_sents = []
    for line in (rsm_dir / 'sentences/rsm').read_text(encoding='utf-8').split('\n'):
        if not line: continue
        text, position = line.split('\t')
        if position.startswith('preface'):
            reference_sents.append(text)

    return [Material('rsm', rsm_dir / 'links/rsm', rsm_dir / 'sentences/rsm')], [' '.join(reference_sents)], rsm_dir / 'wsln_output'


def wsln_summarize(main_material, history_material):
    print('extracting core concepts')
    core_concepts, importance_mapper = find_core_concepts(
        main_material.sentences,
        find_core_concepts_funcs['mix'],
        15
    )

    print(f'core concepts: {core_concepts}')

    # TODO: 计算top 50和Preface的重合率
    # 组合关系
    global_concept_mapper, new_core_concepts = compound_core_concepts(main_material.sentences, core_concepts)
    print(f'{len(new_core_concepts)} new core concepts')
    core_concepts |= new_core_concepts
    
    if len(core_concepts) > 500:
        core_concepts = set(random.sample(sorted(core_concepts), 500))

    clusters, compound_concepts = cluster_core_concepts_on_PMI(main_material.sentences, core_concepts)
    
    core_concepts |= compound_concepts
    # TODO: 控制core concepts的数量，否则不好计算依存sln
    print(f'core concepts {len(core_concepts)} after compounding PMI')
    # TODO: 加入包含关系
    main_dependency_sln, dependency_pairs = construct_dependency_sln(main_material.sentences, core_concepts, clusters)
    
    # dependency_to_neo4j(dependency_pairs, history_material.sentences)
    # node_to_neo4j(main_material.sentences, history_material.sentences, core_concepts)

    # TODO
    print('loading history data')

    # history_dependency_matrix = construct_dependency_matrix(history_material.sentences, dependency_matrix_funs['avg_idf'], core_concepts)
    # history_dependency_forest = construct_dependency_foreset(history_dependency_matrix)
    
    # common_history_forest = extract_common_forest(main_dependency_forest, history_dependency_forest)
    
    print('summarizing')

    # TODO: 前几个core_concepts的确定，聚类分解
    # sentences = summarize(main_material.sentences,
    #                     common_history_forest, history_dependency_forest,
    #                     main_dependency_matrix, history_dependency_matrix,
    #                     core_concepts, importance_mapper,
    #                     total_sentence_n=15, top_n=2)

    summary_text, structured_summary = summarize_markov(main_material,
                        main_dependency_sln, importance_mapper,
                        total_sentence_n=50, top_n=2, rule='base')

    dependency_list = []

    for sampled_path, path_sentences in structured_summary:
        sampled_path_words = {word for node in sampled_path for word in re.split('[ \t]', node)} - {'to', 'with', 'of', 'in', 'at', 'on', 'from', 'for', 'as'}
        
        dependency_list += sampled_path
        
        print(green_text(sampled_path))
        
        for position, sentence in path_sentences:
            print(' '.join(
                red_text(word) if word in sampled_path_words else word for word in sentence.words
            )
            )

    return summary_text, dependency_list


def polish(text, dependency_list):
    import ollama
    
    dependencies = '->'.join(dependency_list)
# <dependencies>{dependencies}<dependencies>
    instruction = f'''You are a perfect writer. Your task is to polish the following text according to a sequence of concept dependencies:
<input text>{text}</input text>

Condition 1. Do not change the literal meaningful and key concepts;
Condition 2. The sequence of sentences can be slightly swapped for higher coherence;
Condition 3. The sentence can be slightly simplified by removing the less important words;
Condition 4. The output text should refers to the sequence of dependencies.
Only output the edited text without any warning and other information, the edited length should exceed 1250 words. The edited text: '''
    
    print('start summarizing', len(instruction))
    # 8B
    response = ollama.chat(model='llama3.1', messages=[{
        'role': 'user',
        'content': instruction,},]
    )
    print('end summarizing')
    
    return response['message']['content']

def calculate_time_cost():
    pass

if __name__ == '__main__':
    import sys
    materials, references, output_base_dir = {
        'rsm': load_rsm,
        'med_rag': load_med_rag,
        'khan': load_khanacademy,
        'bigsurvey': load_bigsurvey,
    }[sys.argv[1]]()

    # references = [word_limit(r) for r in references]

    history_material = Material('foundations_of_database', data_set_dir / 'rsm/foundations_of_database/links/foundations_of_database', data_set_dir / 'rsm/foundations_of_database/sentences/foundations_of_database')

    model_name = f"wsln_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    output_dir = output_base_dir / datetime.now().strftime('%y%m%d_%H%M%S')

    predictions = []
    polished_predictions = []
    for material, reference in tqdm(zip(materials, references), total=len(materials)):
        process = psutil.Process(os.getpid())
        start_time = time.time()
        start_memory = process.memory_info().rss / (1024 ** 2)
        
        prediction, dependency_list = wsln_summarize(material, history_material)
        
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 ** 2)
        print(f'summarizing time: {end_time - start_time:.2f}s')
        print(f'memory cost: {end_memory - start_memory:.2f} MB')
        
        prediction = word_limit(prediction, 1250)
        print(prediction)
        
        start_time = time.time()
        start_memory = process.memory_info().rss / (1024 ** 2)
        
        polished_prediction = word_limit(polish(prediction, dependency_list), 1250)
        
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 ** 2)
        print(f'polish time: {end_time - start_time:.2f}s')
        print(f'memory cost: {end_memory - start_memory:.2f} MB')
        
        predictions.append(prediction)
        polished_predictions.append(polished_prediction)
        
        if not output_dir.exists(): output_dir.mkdir()
        
        (output_dir / material.name).write_text(prediction, encoding='utf-8')
        (output_dir / f'{material.name}-polished').write_text(polished_prediction, encoding='utf-8')

    row, headers = compose_row(predictions, references, model_name)
    polished_row, headers = compose_row(polished_predictions, references, f'{model_name}-polished')
    table = tabulate([row, polished_row], headers = headers, tablefmt = 'fancy_grid')
    init_time = time.time()
    print(table)