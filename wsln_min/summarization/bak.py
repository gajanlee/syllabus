





# def get_importance_of_a_sentence(sentence, words, query_concepts):
#     # TODO: 用importances字典
#     if len(words) == 0:
#         return 0
#     score = sum(get_idf_value(concept) for word in words if word not in query_concepts)
#     score += sum(importances[concept] for word in words if word in query_concepts)
    
#     return score / len(words)

# global_selected_sentence = set()

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
        sentence, words = [link.position]
        
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
