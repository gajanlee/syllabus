import nltk
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
for line in (Path(__file__).parent.parent / 'idf.txt').read_text().split('\n'):
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
SentenceLinks = namedtuple('SentenceLinks', ('text', 'links', 'words', 'position'))

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

Link = namedtuple('Link', ['pre', 'ind', 'rtype', 'post', 'position'])

def _preprocess_node(node):
        '''
        Filter the meaningful words.
        TODO: 纯数字，如2004
        '''
        node = node.replace('-', '')
        node = node.lower()
        words = []
        word_pattern = re.compile('(\d*[a-zA-Z]{3,}\d*|<b>)')
        
        for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(node)):
            if tag not in ['NN', 'NNS', 'VBG', 'JJ', 'CD', 'VBZ', 'VBN', 'RB', 'JJS', 'JJR', 'VB', 'RBR', 'RBS', 'VBD', 'PDT', 'VBP', 'RP', 'IN']:
                continue
            # the word starts with a alphabet
            if not word_pattern.match(word) or word in stop_words or len(word) <= 2:
                continue
            words.append(word)

        return ' '.join(words)

def print_sentence(s):
    '''
    TODO: 
    '''
    for word in s.words:
        if get_idf_value(word) > threshold:
            word.red


class Material:
    
    def __init__(self, links_file, sentences_file):
        position_links_mapper = self._load_position_links_mapper(Path(links_file))
        if sentences_file == 'rsm':
            position_sentence_mapper = self._load_rsm_coref()
        elif sentences_file == 'foundations_of_database':
            position_sentence_mapper = self._load_foundations_sentences()
        else:
            raise NotImplementedError
        
        self.links = []
        self.position_sentence_mapper = {}
        for position, sentence_text in position_sentence_mapper.items():
            if position not in position_links_mapper:
                continue
            words = nltk.word_tokenize(sentence_text)
            self.position_sentence_mapper[position] = SentenceLinks(sentence_text, position_links_mapper[position], words,  position)
            
            self.links += position_links_mapper[position]

    def _load_position_links_mapper(self, links_file):
        position_links_mapper = {}
        
        preprocessed_file = links_file.parent / f'preprocessed_{links_file.name}'; output = ''
        preprocessed = False
        if preprocessed_file.exists():
            preprocessed = True
            links_file = preprocessed_file
        
        for line in tqdm.tqdm(links_file.read_text().split('\n'), desc=f'load links from {links_file}'):
            if not line: continue
            pre, ind, rtype, post, position = line.split('\t')
            link_rtype = {
                link_type.value: link_type
                for link_type in LinkType
            }[rtype]
            
            if not preprocessed:
                pre, ind, post = _preprocess_node(pre), _preprocess_node(ind), _preprocess_node(post)
                if not all([pre, post]) or pre == post: continue
            
            position_links_mapper[position] = position_links_mapper.get(position, []) + [Link(pre, ind, link_rtype, post, position)]
        
            if not preprocessed:
                output += '\t'.join([pre, ind, rtype, post, position]) + '\n'
    
        if not preprocessed:
            preprocessed_file.write_text(output)
        
        return position_links_mapper
    
    def _load_foundations_sentences(self):
        position_sentences_mapper = {}
        for line in (Path(__file__).parent.parent / 'foundations_of_database.sentences').read_text().split('\n'):
            if not line: continue
            sentence, position = line.split('\t')
            position_sentences_mapper[position] = sentence

        return position_sentences_mapper
    
    def _load_rsm_coref(self):
        '''
        return position_section_mapper
        '''
        position_section_mapper = {}
        for path in sorted((Path(__file__).parent.parent / 'rsm_coref/').glob('*')):
            for para_index, paragraph in enumerate(tqdm.tqdm(path.read_text().split('\n'), desc=path.name), 1):
                for sent_index, sentence_text in enumerate(nltk.tokenize.sent_tokenize(paragraph), 1):
                    # sentence = Sentence(sentence_text)
                    words = nltk.word_tokenize(sentence_text)
                    position = f"{path.name.split(' ')[0]}-{para_index}-{sent_index}"
                    position_section_mapper[position] = sentence_text

        return position_section_mapper

    def _load_med_rag(self, files):
        for file in files:
            

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
    action_link_counter = {}
    for sentence in sentences:
        for pre, ind, rtype, post, position in sentence.links:
            if rtype == LinkType.Action:
                action_link_counter[pre] = action_link_counter.get(pre, []) + [ind]
            
    return [k[0] for k in sorted(action_link_counter.items(), key=lambda x: -len(x[1]))]

def _tf_link_count_f(sentences):
    tf_link_counter = Counter()
    for sentence in sentences:
        tf_link_counter.update(sentence.words)
    return tf_link_counter.most_common()

find_core_concepts_funcs = {
    'action_link_count': _action_link_count_f,
    'term_frequency': _tf_link_count_f,
}

def find_core_concepts(sentences, func, n=50):
    '''
    func: 评估重要度的函数
    n: 排名前n的概念作为核心概念
    '''
    ranked_concepts = func(sentences)
    return ranked_concepts[:n]

# core_concepts = find_core_concepts(mainMaterial.sentences, _action_link_count_f, 150)

def extend_core_concepts(sentences, core_concepts: set[str]):
    '''
    通过conjunctive link对core concepts进行扩展
    '''
    core_concepts = copy.deepcopy(core_concepts)
    for sentence in sentences:
        for pre, ind, rtype, post, _ in sentence.links:
            if rtype == LinkType.Conjunctive and (pre in core_concepts or post in core_concepts):
                core_concepts.update([pre, post])

    return core_concepts


def compound_core_concepts(sentences, core_concepts):
    '''
    TODO:
    1. 通过互信息判断constraint link是否应该合并
    2. 
    '''
    importance_mapper = {}
    
    for sentence in sentences:
        for pre, ind, rtype, post, _ in sentence.links:
            if rtype != LinkType.Constraint:
                continue
            
            if pre in core_concepts:
                pass
            
            if post in core_concepts:
                pass

            if pre == get_idf_value(pre) and post == get_idf_value(post):
                



def cluster_core_concepts(sentences, core_concepts, n):
    '''
    根据聚类水平，找到最重要的多个概念
    TODO:
    1. 如果有抽象链接，则合并为一个节点（TODO: 修改dependencyNode的定义）
    
    '''
    for core_concept in core_concepts:
        if core_concept:
            pass


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

def construct_dependency_matrix(links, weight_f, core_concepts):
    '''
    '''
    dependency_matrix = {}
    paragraph_links_mapper = {}

    # 同一个段落下
    for link in links:
        position = link.position
        section, paragraph, sentence = position.split('-')
        position = f'{section}-{paragraph}'
        paragraph_links_mapper[position] = paragraph_links_mapper.get(position, []) + [link]

    for section, links in paragraph_links_mapper.items():
        pre_list = []
        # TODO: relation type给一个系数
        for pre, ind, rtype, post, position in links:
            # 离的越近，（_pre, post）后面的加权越多，所以pre_list需要逆转
            for index, _pre in enumerate(pre_list[::-1], 1):
                dependency_matrix[(_pre, post)] = dependency_matrix.get((_pre, post), 0) + (1 / index) * (get_idf_value(_pre) + get_idf_value(post)) / 2

            # TODO: 更完善的计算
            dependency_matrix[(pre, post)] = dependency_matrix.get((pre, post), 0) + weight_f(pre, post, ind, rtype, position)
            
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
        
        if dependency_matrix[(pre, post)] >= dependency_matrix[(post, pre)]:
            dependency_matrix[(post, pre)] = 0
            core_dependency_matrix[(pre, post)] = value
        else:
            dependency_matrix[(pre, post)] = 0
            core_dependency_matrix[(post, pre)] = value
            
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
    
    common_forest = copy.deepcopy(main_forest)
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




