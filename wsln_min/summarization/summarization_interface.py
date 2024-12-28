import nltk
import math
import itertools
import re
import tqdm
from pathlib import Path
from collections import Counter, namedtuple
stop_words = set(nltk.corpus.stopwords.words('english'))
import termcolor    # 命令行变色
import inflect      # 判断单词复数
import copy

def log(message: str):
    print(message)


###########################################################

## IDF值loading
# TODO: 合并单复数

###########################################################

idf_value_mapper = {}
Default_IDF_Value = 0
for line in (Path(__file__).parent.parent / 'idf.txt').read_text(encoding='utf-8').split('\n'):
    if not line: continue
    word, value = line.split('\t')
    value = float(value)
    idf_value_mapper[word] = value

    Default_IDF_Value += value

# 默认的、不在IDF词典中的词，为全部idf的平均值
Default_IDF_Value /= len(idf_value_mapper)

def get_idf_value(string: str):
    '''
    获取单词/短语的idf值

    '''
    idf_values = []
    
    for word in (words := string.split(' ')):
        if word in idf_value_mapper:
            idf_values.append(idf_value_mapper[word])

    if len(idf_values):
        return sum(idf_values) / len(idf_values)
    return Default_IDF_Value


############################################################

### 加载学习材料（Resource Space Model）的语义链

# 1. 加载数据，语义链和句子的对应关系
# 2. TODO: 打印句子的函数

######################################################

from collections import namedtuple

# sentence with links

class SentenceLinks:

    def __init__(self, text, links, words, position):
        self.text, self.links, self.words, self.position = text, links, words, position


# SentenceLinks = namedtuple('SentenceLinks', ('text', 'links', 'words', 'position'))

from enum import Enum
class LinkType(Enum):
    Action = 'action'
    Attribute = 'attribute'
    Negative = 'negative'
    Conjunctive = 'conjunctive'
    Constraint = 'constraint'
    Sequential = 'sequential'
    CauseEffect = 'cause-effect'
    EffectCause = 'effect-cause'
    Purpose = 'purpose'

Link = namedtuple('Link', ['pre', 'ind', 'rtype', 'post', 'position'])

def normalize_word(word):
    '''
    大小写统一为小写
    TODO: 单复数转为单数
    '''
    word = word.lower()

    dictionary = {
        'axis': 'axis',
        'axes': 'axis',
        'is': 'is',
        'as': 'as',
        'this': 'this',
        'calculus': 'calculus',
        'class': 'class',
        'classes': 'class',
        'thus': 'thus',
    }

    if word in dictionary:
        return dictionary[word]

    p = inflect.engine()
    singular_word = p.singular_noun(word)
    if singular_word:
        return singular_word
    return word

def _preprocess_node(node):
        '''
        Filter the meaningful words.
        TODO: 纯数字，如2004保留
        '''
        
        # TODO: 更完善地处理-，'-'删掉，但是first-normal-form，先变成空格，再判断每个词是不都是词，如果存在非单词的，就合并
        node = node.replace('-', '')
        
        # node = node.lower()
        words = []
        word_pattern = re.compile('(\d*[a-zA-Z]{3,}\d*|<b>)')
        
        for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(node)):
            if tag not in ['NN', 'NNS', 'NNP', 'NNPS', 'VBG', 'JJ', 'CD', 'VBZ', 'VBN', 'RB', 'JJS', 'JJR', 'VB', 'RBR', 'RBS', 'VBD', 'PDT', 'VBP', 'RP', 'IN']:
                continue
            # the word starts with a alphabet
            if not word_pattern.match(word) or word in stop_words or len(word) <= 2:
                continue
            words.append(normalize_word(word))

        return ' '.join(words) 

def print_sentence(s):
    '''
    TODO: 
    '''
    for word in s.words:
        if get_idf_value(word) > threshold:
            word.red


class Material:
    
    def __init__(self, name, links_file, sentences_file, sentence_limit_n = 6000):
        self.name = name
        position_links_mapper = self._load_position_links_mapper(Path(links_file))
        position_sentence_mapper = self._load_position_sentence_mapper(sentences_file, sentence_limit_n)
        
        # self.links = []
        self.position_sentence_mapper = {}
        for position, sentence_text in position_sentence_mapper.items():
            if position not in position_links_mapper:
                continue
            words = nltk.word_tokenize(sentence_text)
            self.position_sentence_mapper[position] = SentenceLinks(sentence_text, position_links_mapper[position], words,  position)
            
            # self.links += position_links_mapper[position]

    def _load_position_links_mapper(self, links_file):
        position_links_mapper = {}
        
        preprocessed_file = links_file.parent / 'preprocessed' / f'{links_file.name}'; 
        preprocessed = False
        if preprocessed_file.exists():
            preprocessed = True
            links_file = preprocessed_file
        
        preprocessed_links = []
        for line in tqdm.tqdm(links_file.read_text(encoding='utf-8').split('\n'), desc=f'load links from {links_file.name}'):
            if not line: continue
            pre, ind, rtype, post, position = line.split('\t')
            link_rtype = {
                link_type.value: link_type
                for link_type in LinkType
            }[rtype]
            
            if not preprocessed:
                # pre, ind, post = _preprocess_node(pre), _preprocess_node(ind), _preprocess_node(post)
                pre, ind, post = _preprocess_node(pre), ind, _preprocess_node(post)
                if not all([pre, post]) or pre == post: continue
            
            position_links_mapper[position] = position_links_mapper.get(position, []) + [Link(pre, ind, link_rtype, post, position)]
        
            if not preprocessed:
                preprocessed_links.append('\t'.join([pre, ind, rtype, post, position]))

        if not preprocessed:
            preprocessed_file.write_text('\n'.join(preprocessed_links), encoding='utf-8')
        
        return position_links_mapper

    def _load_position_sentence_mapper(self, file: Path, sentence_limit_n):
        position_section_mapper = {}
        for index, line in tqdm.tqdm(enumerate(file.read_text(encoding='utf-8').split('\n')), desc=f'load file {file.name}'):
            if not line: continue
            sentence_text, position = line.split('\t')
            position_section_mapper[position] = sentence_text
            if index > sentence_limit_n:
                break
        return position_section_mapper

    @property
    def sentences(self):
        return list(self.position_sentence_mapper.values())


############################################################

##### 对语义节点进行重要度评估
# 用action link的数量
# TODO: 用其他指标
# TODO: 打印top的语义节点
# TODO: 用conjunctive link进行扩展

###########################################################

def _action_link_count_f(sentences):
    '''
    TODO: 归一化counter
    TODO: 归一化idf
    '''
    action_link_counter = {}
    for sentence in sentences:
        for pre, ind, rtype, post, position in sentence.links:
            if rtype == LinkType.Action:
                action_link_counter[pre] = action_link_counter.get(pre, []) + [ind]
            
    return [k[0] for k in sorted(action_link_counter.items(), key=lambda x: -len(x[1]))], {k: len(v) for k, v in action_link_counter.items()}

def _tf_link_count_f(sentences):
    tf_link_counter = Counter()
    for sentence in sentences:
        tf_link_counter.update(sentence.words)
    return tf_link_counter.most_common(), tf_link_counter

def _mix_f(sentences):
    bucket_count = 100
    # budges = [{} for _ in range(budget_count)]
    bucket = {}
    for index, sentence in enumerate(sentences):
        position = int(100 * (index // (len(sentences) / bucket_count)) / bucket_count)

        for pre, ind, rtype, post, _ in sentence.links:
            # for node in [pre, post]:
            # 只考虑active node
            if rtype != LinkType.Action: continue
            for node in [pre]:
                bucket[node] = bucket.get(node, [0 for _ in range(bucket_count)])
                bucket[node][position] += 1

    importance_mapper = {}
    print('node count: ', len(bucket))
    for node, counts in bucket.items():
        vars = 0; mean = sum(counts) / len(counts)
        for count in counts:
            vars += (count - mean) ** 2
        vars /= 10000
        
        global_distance = 0;
        positions = [index for index, count in enumerate(counts) if count != 0]
        for (pos1, pos2) in itertools.product(positions, positions):
            if pos1 >= pos2: continue
            global_distance += abs(pos1 - pos2)
        # 如果只出现在一个桶里，那该值为0
        if len(positions) == 1:
            global_distance = 1 # 按相邻计算
        else:
            global_distance /= len(positions) * (len(positions) - 1) / 2
        global_distance /= 100

        # from utils import red_text
        # print(red_text(node))
        # print('sum counts: ', sum(counts), counts)
        # print('idf value: ', get_idf_value(node))
        # print('global distance: ', global_distance)
        # print('vars: ', vars)
        # print('score: ', sum(counts) * get_idf_value(node) * (1 / (vars + 1)) * global_distance)
        # print('===========')

        importance_mapper[node] = sum(counts) * get_idf_value(node) * (1 / (vars + 1)) * global_distance

    sorted_words = [t[0] for t in sorted(importance_mapper.items(), key=lambda x: -x[1])]

    return sorted_words, importance_mapper


find_core_concepts_funcs = {
    'action_link_count': _action_link_count_f,
    'term_frequency': _tf_link_count_f,
    'mix': _mix_f,
}

def find_core_concepts(sentences, func, n=50):
    '''
    func: 评估重要度的函数
    n: 排名前n的概念作为核心概念

    返回top n的概念，字典key: weight
    '''
    # 按照RSM-preface中的权重进行确定


    ranked_concepts, counter = func(sentences)
    return set(ranked_concepts[:n]), counter

# core_concepts = find_core_concepts(mainMaterial.sentences, _action_link_count_f, 150)

def extend_core_concepts(sentences, core_concepts: set[str]) -> set[str]:
    '''
    通过conjunctive link对core concepts进行扩展
    '''
    import time
    start_time = time.time()
    pattern = re.compile('(' + '|'.join(core_concepts) + ')')

    conj_concepts, abs_concepts = set(), set(); abs_mapper = {}
    for sentence in sentences:
        for pre, ind, rtype, post, _ in sentence.links:
            if rtype == LinkType.Conjunctive and (pre in core_concepts or post in core_concepts):
                conj_concepts.update([pre, post])
            else:
                if (abs := re.search(pattern, pre)) and pre not in core_concepts:
                    abs_concepts.add(pre)
                    abs = abs.string
                    abs_mapper[abs] = abs_mapper.get(abs, []) + [pre]

                # elif core_concept in post and core_concept != post:
                elif (abs := re.search(pattern, post)) and post not in core_concepts:
                    abs_concepts.add(post)
                    abs = abs.string
                    abs_mapper[abs] = abs_mapper.get(abs, []) + [post]

    return conj_concepts, abs_concepts, abs_mapper


def compound_core_concepts(sentences, core_concepts):
    '''
    TODO:
    1. 通过互信息判断constraint link是否应该合并
    2. 
    '''
    global_concept_mapper = {}
    new_core_concepts = set()

    for sentence in sentences:
        links = []
        local_concpet_mapper = {} # 记录本link内的变换

        # 先展示Constraint语义链，过滤出所有节点对，使得两个语义节点之间只有constraint link
        only_one_pre_to_post = {}
        for pre, ind, rtype, post, position in sentence.links:
            only_one_pre_to_post[(pre, post)] = only_one_pre_to_post.get((pre, post), []) + [ind]
        only_one_pre_to_post_set = {
            pair
            for pair, values in only_one_pre_to_post.items() if len(values) == 1
        }

        for link in sorted(sentence.links, key=lambda l: l.rtype != LinkType.Constraint):
            pre, ind, rtype, post, position = link
            # 删除合并的语义链
            if (rtype == LinkType.Constraint and
                (pre in core_concepts or post in core_concepts) and
                (pre, post) in only_one_pre_to_post_set
            ):
                fake_pre = local_concpet_mapper.get(pre, pre)
                new_concept = f'{fake_pre}\t{ind}\t{post}'
                # 如果pre 已经结合过了，A和B都是A of B,再来了一个B of C，那A, B, C都变成A of B of C 
                if pre in local_concpet_mapper:
                    # A
                    pre_pre = local_concpet_mapper[pre].split('\t')[0]
                    local_concpet_mapper[pre_pre] = new_concept

                new_core_concepts.add(new_concept)
                # new_concept = local_concpet_mapper.get(pre, pre) + f' {ind} ' + local_concpet_mapper.get(post, post) 
                local_concpet_mapper[pre] = new_concept
                local_concpet_mapper[post] = new_concept

                global_concept_mapper[pre] = global_concept_mapper.get(pre, set()) | {new_concept}
                global_concept_mapper[post] = global_concept_mapper.get(post, set()) | {new_concept}
                continue
        
        for pre, ind, rtype, post, position in sentence.links:
            if pre in local_concpet_mapper and post in local_concpet_mapper:
                continue
            pre = local_concpet_mapper.get(pre, pre)
            post = local_concpet_mapper.get(post, post)

            links.append((pre, ind, rtype, post, position))

        # new_sentence = copy.deepcopy(sentence)
        sentence.links = links
        # 语义链是否重新赋值？

    return global_concept_mapper, new_core_concepts

def cluster_core_concepts_on_word_level(sentences, core_concepts):
    '''
    根据聚类水平，找到最重要的多个概念
    基础概念只是单词，如resource, space
    TODO:
    1. 如果有抽象链接，则合并为一个节点（TODO: 修改dependencyNode的定义）
    
    TODO: 应该弄成短语级，层次的
    如果两个重复覆盖率过高，则选择更具体的词
    '''
    mapper = {}
    node_set = set()
    for sentence in sentences:
        for pre, ind, rtype, post, _ in sentence.links:
            node_set.update([pre, post])
    
    clusters = {}

    for node in sorted(node_set, key=lambda x: len(x)):
        if len(node) < 5: continue
        success = False
        for cluster_key in clusters:
            for word in node.split(' '):
                if cluster_key == word:
                    clusters[cluster_key].add(node)
                    success = True

        if not success:
            clusters[node] = {node}

    print('\n\n\n'*3)
    for key, values in sorted(clusters.items(), key=lambda x: -len(x[1])):
        if len(values) < 10:
            continue
        from utils import red_text
        print(red_text(key), values)

    return clusters


def cluster_core_concepts_on_PMI(sentences, core_concepts):
    '''
    根据聚类水平，找到最重要的多个概念，短语级别
    如果两个重复覆盖率过高，则选择更具体的词
    '''

    clusters = {}

    def get_all_sub_words(string):
        # query language\tof\tresource space model，找到节点resource space model
        nodes = [node for index, node in enumerate(string.split('\t'), 1) if index % 2 == 1]
        # query language\tof\tresource space model，找到指示词of
        indicators = [node for index, node in enumerate(string.split('\t'), 1) if index % 2 == 0]

        sub_words = []
        for node in nodes:
            words = node.split(' ')
            for phrase_len in range(1, len(words) + 1):
                for start in range(0, len(words) - phrase_len + 1):
                    sub_word = ' '.join(words[start: start + phrase_len])
                    sub_words.append(sub_word)

                    if phrase_len > 1:
                        _lst = sub_word.split(' ')
                        key = ' '.join(_lst[:-1])
                        clusters[key] = clusters.get(key, set()) | {sub_word}
                        key = ' '.join(_lst[1:])
                        clusters[key] = clusters.get(key, set()) | {sub_word}

        for node_len in range(2, len(nodes) + 1):
            for start in range(0, len(nodes) - node_len + 1):
                node = nodes[start]
                for index in range(start + 1, start + node_len):
                    node += '\t' + indicators[index - 1] + '\t'+ nodes[index]
                sub_words.append(node)

                # 映射pre节点
                # query\tof\tresource -> query\tof\tresource\tas\tspace
                _sub_node = nodes[start]
                for index in range(start + 1, start + node_len - 1):
                    _sub_node += '\t' + indicators[index - 1] + '\t'+ nodes[index]
                clusters[_sub_node] = clusters.get(_sub_node, set()) | {node}

                # 映射post节点
                # resource\tas\tspace -> query\tof\tresource\tas\tspace
                _sub_node = nodes[start + 1]
                for index in range(start + 2, start + node_len):
                    _sub_node += '\t' + indicators[index - 1] + '\t'+ nodes[index]
                    clusters[nodes[start]] 
                clusters[_sub_node] = clusters.get(_sub_node, set()) | {node}


        return sub_words

    node_counter = Counter()
    for sentence in sentences:
        for pre, ind, rtype, post, _ in sentence.links:
            node_counter.update(get_all_sub_words(pre))
            node_counter.update(get_all_sub_words(post))

            # phrases = pre.split('\t') + post.split('\t')
            # words = [word for phrase in phrases for word in phrase.split(' ')]

            # word_sum += len(words)

    def calculate_pmi(word1, word2):
        p_xy = node_counter.get(word1 + ' ' + word2) / len(node_counter)
        p_x = node_counter.get(word1) / len(node_counter)
        p_y = node_counter.get(word2) / len(node_counter)

        return math.log(p_xy / (p_x * p_y), 2)

    def get_abstract_value(string):
        nodes = [node for index, node in enumerate(string.split('\t'), 1) if index % 2 == 1]
        pmi = 0
        for node in nodes:
            words = node.split(' ')
            if len(words) == 1:
                return 0

            pmi += (calculate_pmi(words[0],' '.join(words[1:])) + calculate_pmi(' '.join(words[:-1]), words[-1])) / 2
        
        return pmi / len(nodes)

    pmi_dict = {}
    for node in sorted(node_counter, key=lambda x: len(x)):
        value = get_abstract_value(node)
        pmi_dict[node] = value

        # for phrase in node.split('\t'):
        #     # phrase和node之间存在关系
        #     get_abstract_value(phrase)

        #     # word相当于是phrase的父概念
        #     for word in get_all_sub_words(phrase):
        #         print()

        #     for word in phrase.split(' '):
        #         # word和phrase之间存在关系
        #         pass

    print('\n\n\n'*3)
    for key, values in sorted(clusters.items(), key=lambda x: -len(x[1])):
        if len(values) < 10:
            continue
        from utils import red_text
        print(red_text(key), values)

    return clusters


############################################################

##### 构建依赖链
# 1. 一个语义链内，前面的词依赖于后面的词
# 2. 一句话内，前面的词依赖于后面的词，依赖强度取决于间隔了多少个语义节点
# TODO: 融入constraint权重

###########################################################

# concept_weight_mapper = {}

# for index, (word, inds) in enumerate(sorted(action_link_counter.items(), key = lambda x: -len(x[1]) * get_idf_value(x[0]))):
#     print(word, len(inds))
#     if index <= 100:
#         core_concepts.add(word)
#         concept_weight_mapper[word] = len(action_link_counter[word]) * get_idf_value(word)

def _weight_avg_idf_f(pre, post, ind, rtype, position):
    return (get_idf_value(pre) + get_idf_value(post) + get_idf_value(ind)) / 3

dependency_matrix_funs = {
    'avg_idf': _weight_avg_idf_f,
}

def construct_dependency_matrix(sentences, weight_f, core_concepts):
    '''
    TODO: 检查代码正确性
    TODO: 计算权重的公式，重新考虑
    '''
    dependency_matrix = {}
    paragraph_links_mapper = {}

    node_span = {}
    # 同一个段落下
    for index, sentence in enumerate(sentences):
        for link in sentence.links:
            pre, ind, rtype, post, position = link
            # 同一个段落的放在一起
            *section, paragraph, _ = position.split('-')
            section = ' '.join(section)
            position = f'{section}-{paragraph}'
            paragraph_links_mapper[position] = paragraph_links_mapper.get(position, []) + [link]

            for node in [pre, post]:
                if node not in node_span: node_span[node] = [index, index]
                else: node_span[node][1] = index
                if node not in node_span: node_span[node] = [index, index]
                else: node_span[node][1] = index

    core_node_span = {node: (end - start) / index for node, (start, end) in node_span.items() if node in core_concepts}

    # # 归一化，print，值越大，越global
    # for core_concept1 in core_node_span:
    #     for core_concept2 in core_node_span:
    #         if core_node_span[core_concept1] < core_node_span[core_concept2]:
    #             print(f'local【{core_concept1}】->global【{core_concept2}】')


    for section, links in paragraph_links_mapper.items():
        pre_list = []
        for pre, ind, rtype, post, position in links:
            # 离的越近，（_pre, post）后面的加权越多，所以pre_list需要逆转
            for index, _pre in enumerate(pre_list[::-1], 1):
                if _pre in core_concepts and post in core_concepts:
                    dependency_matrix[(_pre, post)] = dependency_matrix.get((_pre, post), 0) + (1 / index) * (get_idf_value(_pre) + get_idf_value(post)) / 2

            # action link的话，后面的依赖前面的

            if rtype == LinkType.Action and pre in core_concepts and post in core_concepts:
                # action link
                dependency_matrix[(pre, post)] = dependency_matrix.get((pre, post), 0) + weight_f(pre, post, ind, rtype, position)
                # local relies on global
                local, globa = (pre, post) if core_node_span[pre] < core_node_span[post] else (post, pre)
                dependency_matrix[(local, globa)] = dependency_matrix.get((local, globa), 0) + core_node_span[globa] - core_node_span[local]

            # if pre in post:
            #     dependency_matrix[(pre, post)] = dependency_matrix.get((pre, post), 0) + weight_f(pre, post, ind, rtype, position)

            pre_list.append(pre)

    core_dependency_matrix = {}
    # 只保留核心概念的依赖关系
    # 删除反向依赖，保留权重更高的那个
    for pair, value in dependency_matrix.items():
        pre, post = pair
        if pre not in core_concepts or post not in core_concepts or pre == post:
            dependency_matrix[(pre, post)] = 0
            continue
        
        # 比较反向链权重
        if (post, pre) not in dependency_matrix:
            continue
        
        # 如果存在抽象关系，如pre为'space', post为'resource space'
        if pre in post:
            core_dependency_matrix[(pre, post)] = value + dependency_matrix.get((post, pre)) + 1
            core_dependency_matrix[(post, pre)] = 0
            continue

        if dependency_matrix[(pre, post)] >= dependency_matrix[(post, pre)]:
            core_dependency_matrix[(post, pre)] = 0
            core_dependency_matrix[(pre, post)] = value
        # TODO: 验证正确性
        # else:
        #     core_dependency_matrix[(pre, post)] = 0
        #     core_dependency_matrix[(post, pre)] = value
            
        # core_dependency_matrix[pair] = value

    return core_dependency_matrix

# main_dependency_matrix = construct_dependency_matrix(mainMaterial.links, _weight_avg_idf_f, core_concepts)

# from wsln_min.summarization.dependency import construct_dependency_foreset, DependencyForest

# # TODO: 去除值为0的
# main_dependency_forest = construct_dependency_foreset(main_dependency_matrix)

# # core_dependency_forest.roots[0].content
# # core_dependency_forest.get_node_sequence('object', 'community')


####################################################

## 加载已经学过的材料

######################################################

# # history的原文不是很重要
# history_material = Material('foundations_of_database.triplets', 'foundations_of_database.txt')

# history_dependency_matrix = construct_dependency_matrix(history_material.links, _weight_avg_idf_f, core_concepts)
# # select the top xxx concepts
# history_dependency_forest = construct_dependency_foreset(history_dependency_matrix)

####################################################

### 找到main和history公共的基础概念
# 因为一本书，有Introduction，肯定会涉及到基础概念的选取，历史材料里面的概念可能没什么用，取交集

######################################################


def extract_common_forest(main_forest, history_forest):
    '''
    在main_forest的基础上进行扩展，而不是找公共子树
    common forest可能会有更多的中间节点，即依赖链路增加
    '''
    
    pairs = []
    iterated_nodes = set()
    queue = [(root, []) for root in history_forest.roots]
    pair_sequences = []
    # 找到所有dependency sequence
    while queue:
        node, pairs = queue.pop()
        iterated_nodes.add(node)
        for next in node.next:
            if next not in iterated_nodes:
                queue.append(
                    (next, pairs + [(node, next)])
                )
            else:
                pair_sequences.append(pairs)
        if len(node.next) == 0:
            pair_sequences.append(pairs)

    filtered_pairs = []
    # 找到最短路径
    for sequence in tqdm.tqdm(pair_sequences):
        start = -1; end = -1
        for index, (pre, post) in enumerate(sequence):
            if pre.content in main_forest.string_to_node and start == -1:
                start = index
            if post.content in main_forest.string_to_node and start != -1:
                end = index + 1

        if end != -1:
            # 获取中间所有concepts
            filtered_pairs.append(sequence[start:end])
    
    # common_forest = copy.deepcopy(main_forest)
    common_forest = main_forest
    for pairs in tqdm.tqdm(filtered_pairs, desc='construct common dependency forest'):
        for pre, post in pairs:
            common_forest.append((pre.content, post.content))
    return common_forest
    

def extract_common_forest_dep(main_foreset, history_forest):
    '''
    merge the side_chain to main_chain
    找公共树，弃用
    '''
    pairs = []
    iterated_nodes = set()
    queue = [(root, []) for root in history_forest.roots]
    pair_sequences = []
    # 找到所有dependency sequence
    while queue:
        node, pairs = queue.pop()
        iterated_nodes.add(node)
        for next in node.next:
            if next not in iterated_nodes:
                queue.append(
                    (next, pairs + [(node, next)])
                )
            else:
                pair_sequences.append(pairs)
        if len(node.next) == 0:
            pair_sequences.append(pairs)

    filtered_pairs = []
    # 找到最短路径
    for sequence in tqdm.tqdm(pair_sequences):
        start = -1; end = -1
        for index, (pre, post) in enumerate(sequence):
            if pre.content in main_foreset.string_to_node and start == -1:
                start = index
            if post.content in main_foreset.string_to_node and start != -1:
                end = index + 1

        if end != -1:
            # 获取中间所有concepts
            filtered_pairs.append(sequence[start:end])

    # for filtered_pair in filtered_pairs:
    #     main_dependency_matrix.append(filtered_pair)
    from dependency import DependencyForest
    common_forest = DependencyForest()
    for pairs in tqdm.tqdm(filtered_pairs, desc='construct common dependency forest'):
        for pre, post in pairs:
            common_forest.append((pre.content, post.content))
    return common_forest

# common_history_forest = extract_common_forest(main_dependency_forest, history_dependency_forest)


####################################################

### 分解依赖链
# 1. 计算语义节点的重要度（PageRank）
# 2. 选择前n个，分别抽取出依赖链（核心概念前面的依赖项可以重复，后面的不能重复）

######################################################


def get_concept_importance(dependency_forest):
    # 计算所有节点的重要度，core concepts
    # TODO: action节点数量作为初始化
    importance_mapper = {}
    # importance init
    for concept in dependency_forest.string_to_node.keys():
        importance_mapper[concept] = 1 # or the count of action links

    # PageRank算法，重要度节点迁移
    N = len(dependency_forest.string_to_node.keys())
    # 出边更重要，作为主语（TODO：继续思考逻辑）
    beta = 0.4

    # the ending condition，迭代100次或是收敛
    for _ in range(100):
        for concept in dependency_forest.string_to_node.keys():
            incoming = [n.content for n in dependency_forest.string_to_node[concept].pre]
            outgoing = [n.content for n in dependency_forest.string_to_node[concept].next]
        
            importance_mapper[concept] *= (len(incoming) +len(outgoing)) / N
        
            importance_mapper[concept] += beta * sum(importance_mapper[p] for p in incoming) / (len(incoming) + 1)
            importance_mapper[concept] += (1 - beta) * sum(importance_mapper[p] for p in incoming) / (len(incoming) + 1)

    sorted(importance_mapper.items(), key=lambda x: -x[1])
    
    return importance_mapper

# 节点聚类
# 1. abstract link
# 2. 依赖度


def split_nodes(common_dependency_forest, history_dependency_forest, common_dependency_matrix, history_dependency_matrix, core_concepts, top_n = 5):
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
    for pre, post in common_dependency_forest.pairs:
        pre, post = pre.content, post.content

        if pre in history_concepts and post in history_concepts:
            weight = (get_idf_value(pre) + get_idf_value(post)) / 2
            history_pairs.append((pre, post, weight))
    
    # 按从高到低进行排序。TODO：没有考虑到连贯性
    history_pairs = sorted(history_pairs, key=lambda x: -x[2])
    

    # TODO: 根据依赖森林选出TOP 5的语义节点
    # 规则：在common dependency forest中，有最多前置后置依赖关系的节点
    concepts_weight_mapper = {}
    for pre, post in common_dependency_forest.pairs:
        pre, post = pre.content, post.content
        weight = (get_idf_value(pre) + get_idf_value(post)) / 2
        
        concepts_weight_mapper[pre] = concepts_weight_mapper.get(pre, 0) + weight
        concepts_weight_mapper[post] = concepts_weight_mapper.get(post, 0) + weight

    remain_concepts = {c for c in main_concepts}; top_n_concepts = []
    # pre to post
    for _ in range(top_n):
        node_dependencies = {}; concept_pair_mapping = {}
        for (pre, post), weight in common_dependency_matrix.items():
            if pre in remain_concepts and post in remain_concepts:
                node_dependencies[pre] = node_dependencies.get(pre, 0) + weight
                concept_pair_mapping[pre] = concept_pair_mapping.get(pre, set()) | {post}
                node_dependencies[post] = node_dependencies.get(post, 0) + weight
                concept_pair_mapping[post] = concept_pair_mapping.get(post, set()) | {pre}

        # top_concept = sorted(node_dependencies.items(), key=lambda x: -x[1])[0][0]
        top_concept = sorted(concept_pair_mapping.items(), key=lambda x: -len(x[1]))[0][0]
        top_n_concepts.append(top_concept)
        # 移除和top_concept有关的
        remain_concepts -= {top_concept} | concept_pair_mapping[top_concept]
        # for node in concept_pair_mapping[top_concept]:
        #     remain_concepts -= concept_pair_mapping[node]

    # top_n_concepts = [p[0] for p in sorted(concepts_weight_mapper.items(), key=lambda p: p[1], reverse=True)[:top_n]]
    
    linking_main_pairs = []
    # TODO: 聚类删除abstract类概念
    for core_concept in top_n_concepts:
        linking_pairs = common_dependency_forest.get_node_sequence('', to_concept=core_concept, forward=False)
        main_pairs = common_dependency_forest.get_node_sequence(core_concept, '', forward=True)
        
        # TODO: 移除相似的前序节点，和相同的后序节点
        # TODO: 已经出现过的词，不再出现
        # TODO: pairs长度根据句子数量动态调整，目前还是选择第一个序列
        # TODO-latest:把linking_pairs和main_pairs转为(pre, post, weight)的形式
        # TODO: seq只有一个怎么办
        # 把所有的linking_pairs合并到一起，然后排序
        # 每个序列的形式如[ChainNode('existing resource'), 'user', 'resource space model']
        filtered_linking_pairs = []
        for sequence in linking_pairs:
            pre = sequence[0].content
            for post in sequence[1:]:
                post = post.content
                weight = (get_idf_value(pre) + get_idf_value(post)) / 2
                filtered_linking_pairs.append((pre, post, weight))
                pre = post
        
        # core_concepts向后的序列
        filtered_main_pairs = []
        for sequence in main_pairs:
            pre = sequence[0].content
            for post in sequence[1:]:
                post = post.content
                weight = (get_idf_value(pre) + get_idf_value(post)) / 2
                filtered_main_pairs.append((pre, post, weight))
                pre = post
        
        linking_main_pairs.append((core_concept, filtered_linking_pairs, filtered_main_pairs))
    
    # TODO: linking和main pairs的序列长度，选择最贴近目标句子数量的
    return history_pairs, linking_main_pairs



