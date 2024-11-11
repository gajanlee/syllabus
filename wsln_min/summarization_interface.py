import nltk
import re
import tqdm
from pathlib import Path
from collections import Counter, namedtuple
stop_words = set(nltk.corpus.stopwords.words('english'))
from lexer import Sentence
import termcolor    # 命令行变色
import inflect      # 判断单词复数


###########################################################

## IDF值loading

###########################################################

# TODO: 可以考虑移除idf值最低的20%词
idf_value_mapper = {}
Default_IDF_Value = 0
for line in Path('idf.txt').read_text().split('\n'):
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

###########################################################

class LinkType:
    ActionType = 'action'
    AttributeType = 'attribution'
    ConjunctiveType = 'conjunction'
    ConstraintType = 'constraint'
    SequentialType = 'sequential'

Link = namedtuple('Link', ['pre', 'ind', 'rtype', 'post', 'position'])

def _preprocess_node(node):
        '''
        Filter the meaningful words.
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

def load_links(file):
    links = []
    for line in tqdm.tqdm(file.read_text().split('\n')):
        if not line: continue
        pre, ind, rtype, post, position = line.split('\t')
        
        pre, ind, post = _preprocess_node(pre), _preprocess_node(ind), _preprocess_node(post)
        if not all([pre, post]) or pre == post: continue
        
        links.append(Link(pre, ind, rtype, post, position))
    return links

core_concepts = set()
action_link_counter = {}
tf_link_counter = {}
main_links = load_links(Path('rsm.triplets'))
for pre, ind, rtype, post, _ in main_links:
    if rtype == ActionType:
        action_link_counter[pre] = action_link_counter.get(pre, []) + [ind]
    tf_link_counter[pre] = tf_link_counter.get(pre, []) + [ind]



############################################################

##### 对语义节点进行重要度评估
# TODO: 打印top的语义节点

###########################################################

print(len(action_link_counter))
core_concepts = set()
concept_weight_mapper = {}

for index, (word, inds) in enumerate(sorted(action_link_counter.items(), key = lambda x: -len(x[1]) * get_idf_value(x[0]))):
    print(word, len(inds))
    if index <= 100:
        core_concepts.add(word)
        concept_weight_mapper[word] = len(action_link_counter[word]) * get_idf_value(word)


############################################################

##### 构建依赖链
# 1. 一个语义链内，前面的词依赖于后面的词
# 2. 一句话内，前面的词依赖于后面的词，依赖强度取决于间隔了多少个语义节点
# TODO: 融入constraint权重

###########################################################

def get_dependency_matrix(link_list):
    dependency_matrix = {}
    position_mapper = {}

    for link in link_list:
        position = link.position
        position_mapper[position] = position_mapper.get(position, []) + [link]

    for section, links in position_mapper.items():
        pre_list = []
        # TODO: relation type给一个系数
        for pre, ind, rtype, post, position in links:
            # 离的越近，（_pre, post）后面的加权越多，所以pre_list需要逆转
            for index, _pre in enumerate(pre_list, 1):
                dependency_matrix[(_pre, post)] = dependency_matrix.get((_pre, post), 0) + (1 / index) * (get_idf_value(_pre) + get_idf_value(post)) / 2

            # TODO: 更完善的计算
            dependency_matrix[(pre, post)] = dependency_matrix.get((pre, post), 0) + (get_idf_value(pre) + get_idf_value(post) + get_idf_value(ind)) / 3
            pre_list.append(pre)

    return dependency_matrix

main_dependency_matrix = get_dependency_matrix(main_links)

# TODO: 删除反向依赖值
core_dependency_matrix = {}
for pair, value in main_dependency_matrix.items():
    if pair[0] not in core_concepts or pair[1] not in core_concepts or pair[0] == pair[1]:
        continue
    core_dependency_matrix[pair] = value

for pair, value in sorted(core_dependency_matrix.items(), key=lambda x: -x[1])[:10]:
    print(pair, value)

core_dependency_forest = construct_dependency_foreset(core_dependency_matrix)
core_dependency_forest.roots[0].content
core_dependency_forest.get_node_sequence('object', 'community')



####################################################

## 加载已经学过的材料

######################################################

history_links = load_links(Path('foundations_of_database.triplets'))
history_dependency_matrix = get_dependency_matrix(history_links)
# select the top xxx concepts
history_dependency_forest = construct_dependency_foreset(history_dependency_matrix)

####################################################

### 找到二者公共的基础概念
# 因为一本书，有Introduction，肯定会涉及到基础概念的选取，历史材料里面的概念可能没什么用，取交集

######################################################

def extract_common_forest(main_foreset, history_forest):
    '''
    merge the side_chain to main_chain
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

    print(len(filtered_pairs))
    
    for pre, post in filtered_pairs[0]:
        print(pre.content, '->', post.content)

    # for filtered_pair in filtered_pairs:
    #     main_dependency_matrix.append(filtered_pair)
    
    common_forest = DependencyForest()
    for pairs in tqdm.tqdm(filtered_pairs):
        for pre, post in pairs:
            common_forest.append((pre.content, post.content))
    return common_forest

common_history_forest = extract_common_forest(core_dependency_forest, history_dependency_forest)


####################################################

### 分解依赖链
# 1. 计算语义节点的重要度
# 2. 选择前n个，分别抽取出依赖链（核心概念前面的依赖项可以重复，后面的不能重复）

######################################################

# 计算所有节点的重要度
importances = {}

# importance init
for concept in core_dependency_forest.string_to_node.keys():
    importances[concept] = 1 # or the count of action links

# importance transfer
N = len(core_dependency_forest.string_to_node.keys())
beta = 0.4

# the ending condition:
for i in range(9):
    for concept in core_dependency_forest.string_to_node.keys():
        incoming = [n.content for n in core_dependency_forest.string_to_node[concept].pre]
        outgoing = [n.content for n in core_dependency_forest.string_to_node[concept].next]
    
        importances[concept] *= (len(incoming) +len(outgoing)) / N
    
        importances[concept] += beta * sum(importances[p] for p in incoming) / (len(incoming) + 1)
        importances[concept] += (1 - beta) * sum(importances[p] for p in incoming) / (len(incoming) + 1)

sorted(importances.items(), key=lambda x: -x[1])



##########################################################################################

##############       生成摘要               ##############################################
# 1. 
# 2. 

##########################################################################################



####################################################

### 加载数据，语义链和句子的对应关系

######################################################

# load rsm sentences
position_section_mapper = {}

for path in sorted(Path('rsm_coref/').glob('*')):
    for para_index, paragraph in enumerate(tqdm.tqdm(path.read_text().split('\n'), desc=path.name), 1):
        for sent_index, sentence_text in enumerate(nltk.tokenize.sent_tokenize(paragraph), 1):
            # sentence = Sentence(sentence_text)
            words = nltk.word_tokenize(sentence_text)
            position = f"{path.name.split(' ')[0]}-{para_index}-{sent_index}"
            position_section_mapper[position] = (
                sentence_text, words
            )


####################################################

### 加载数据，语义链和句子的对应关系

######################################################

# 对句子进行信息量排序
# 方案1: 直接用重要度排序
# 方案2: PageRank类方法

# 所有包含基础概念的句子，进行排序
# TODO: 移除所有已经选过的句子，并降权
# TODO: 输出加粗关键概念

def get_importance_of_a_sentence(sentence, words, query_concepts):
    # TODO: 用importances字典
    if len(words) == 0:
        return 0
    score = sum(get_idf_value(concept) for word in words if word not in query_concepts)
    score += sum(importances[concept] for word in words if word in query_concepts)
    
    return score / len(words)

global_selected_sentence = set()

def summarize(links, concepts: set, sentence_count = 1):
    # 过滤出所有包含指定concepts的句子子集
    # TODO: concepts变为依赖关系，目前没有考虑【依赖关系】
    filtered_sentences = []
    selected_position = set()
    for link in links:
        if link.pre not in concepts and link.post not in concepts:
            continue
        if link.position in selected_position:
            continue
        selected_position.add(link.position)
        sentence, words = position_section_mapper[link.position]
        
        # TODO: 句子不会重复选择，更好的策略
        if sentence in global_selected_sentence:
            continue
        # global_selected_sentence.add(sentence)
        
        # TODO: sentence的评价标准没有加入和query concepts之间的关联，或者说get_importance_of_a_sentence的关联度不强
        filtered_sentences.append((sentence, words, link))

    # 按照position对所有句子进行排序 (sentence, words, link)
    ranked_sentences = sorted(filtered_sentences, key=lambda item: -get_importance_of_a_sentence(item[0], item[1], concepts))
    
    result_sentences = []
    for sentence, words, link in ranked_sentences[:sentence_count]:
        result_sentences.append(sentence)
        global_selected_sentence.add(sentence)
    return result_sentences

sentences = []

# 算上roots以外，不在core concepts中的概念（sequence的结尾也是core concepts）
basic_concepts = (set(common_history_forest.string_to_node.keys()) - set(core_concepts)) | set([root.content for root in common_history_forest.roots])

# linking concepts是basic concepts和最核心core concepts之前存在依赖关系的concepts
# TODO: 排序core concepts
linking_concepts = set(common_history_forest.string_to_node.keys()) - set(basic_concepts)
# 有多少条item，就有多少个链接路
top_5_concepts = sorted(core_concepts, key=lambda c: -importances.get(c, 0))[:5]
for query_concept in top_5_concepts:
    linking_sequences = []
    for from_concept in linking_concepts:
        linking_sequence = core_dependency_forest.get_node_sequence(from_concept, query_concept, forward=False)
        linking_sequences.append(linking_sequence)

# rest_core_concepts是最最核心的概念，及之后的概念。
# 暂时不考虑分点，直接用top5作为分点
# rest_core_concepts = core_concepts - linking_concepts

content_sequences = []
for query_concept in top_5_concepts:
    sequence = core_dependency_forest.get_node_sequence(query_concept, '', forward=True)
    content_sequences.append(sequence)

# 生成摘要
summary = []
background_sentences = summarize(main_links, basic_concepts, 2)
summary += background_sentences
summary.append('='*30)

for index, (query_concept, linking_sequence, content_sequence) in enumerate(zip(top_5_concepts, linking_sequences, content_sequences), 1):
    summary.append(f'{index}. {query_concept}')
    # sequence转为concepts，逐个学习路径
    linking_concepts = set()
    for sequence in linking_sequence:
        linking_concepts.update(node.content for node in sequence)
    
    motivation_sentences = summarize(main_links, linking_concepts, 2)
    summary += motivation_sentences
    summary.append('-'*30)
    
    # sequence转为concepts
    for sequence in content_sequence:
        content_concepts = set(node.content for node in sequence)
    content_sentences = summarize(main_links, content_concepts, 2)
    summary += content_sentences
    summary.append('='*30)
