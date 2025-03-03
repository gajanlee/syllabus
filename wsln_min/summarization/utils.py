import termcolor    # 命令行变色
import json
from types import SimpleNamespace
from pathlib import Path
from collections import Counter

# TODO: 也可以作为评估指标，出现陌生概念的比例
# 绿色：已经学会的概念，在history概念集中
green_text = lambda s: termcolor.colored(s, 'green')
# 蓝色：在【已学会-核心】概念之间的概念，common
blue_text = lambda s: termcolor.colored(s, 'blue')
# 红色：完全未知的核心概念
red_text = lambda s: termcolor.colored(s, 'red')

def colored_sentence(words: list[str]):
    result = []
    
    for word in words:
        if word in core_concepts:
            word = red_text(word)
            
        if word in linking_concepts:
            word += blue_text(word)

        if word in basic_concepts:
            word += green_text(word)
            
        result.append(word)
        
    return ' '.join(result)

# for sentence in summary:
#     print(colored_sentence(nltk.tokenize.word_tokenize(sentence)))

# patterns_path = Path(__file__).parent / "res/patterns_v012.json"
# changelog_path = Path(__file__).parent / "res/changelog"

pattern_pos_class_mapper = {
    "N": ["NN", "NNS", "NNP", "NNPS", "PRP", "EX", "WP", "WDT", "CD"],
    "V": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"],
    "ADJ": ["JJ", "JJR", "JJS", "PRP$", "DT", "POS", "WP$"],
    "ADV": ["RB", "RBR", "RBS", "WRB"],
    "SYM": ["SYM"],
    "CONJ": ["CC"],
    "PREP": ["IN", "TO", "RP"],
    "OTHER": ["INTJ", "UH"],
}

pos_mapper = {
    value: key
    for key, values in pattern_pos_class_mapper.items()
    for value in values
}

BE_words = ["be", "is", "am", "are", "were", "was"]

is_noun = lambda token: token.pos in pattern_pos_class_mapper["N"]
DEFAULT_NOUN_POS = "NN"
is_real_noun = lambda token: token.pos in ["NN", "NNS", "NNP", "NNPS"]
is_verb = lambda token: token.pos in pattern_pos_class_mapper["V"]
is_adj  = lambda token: token.pos in pattern_pos_class_mapper["ADJ"]
is_adv  = lambda token: token.pos in pattern_pos_class_mapper["ADV"]
is_conj = lambda token: token.pos in pattern_pos_class_mapper["CONJ"]
is_prep = lambda token: token.pos in pattern_pos_class_mapper["PREP"]


def load_json_to_meta(json_path: str):
    """
    Convert a json file into a python object. 
    """
    return json.load(open(json_path), object_hook=lambda d: SimpleNamespace(**d))

def node_to_neo4j(main_sentences, history_sentences, core_concepts):
    main_nodes, history_nodes = Counter(), Counter()
    
    for sentence in main_sentences:
        for pre, ind, rtype, post, position in sentence.links:
            main_nodes.update([pre, post])
        
    for sentence in history_sentences:
        for pre, ind, rtype, post, position in sentence.links:
            history_nodes.update([pre, post])
        
    main_nodes = {n for n in main_nodes if main_nodes[n] > 5}
    history_nodes = {n for n in history_nodes if history_nodes[n] > 5}
        
    node_to_id = {
        node: index for index, node in enumerate(main_nodes | history_nodes, 1)
    }
    
    def get_source(node):
        if node in core_concepts:
            return 'core'
        if node in main_nodes and node in history_nodes:
            return 'common'
        if node in main_nodes:
            return 'main'
        return 'history'

    output_string = 'MATCH (n) DETACH DELETE n;\nCREATE'
    for node, index in node_to_id.items():
        output_string += f"(n{node_to_id[node]}:{get_source(node)} {{name: '{node}', isCore: {1 if node in core_concepts else 0}}}),\n"
        
    for sentence in main_sentences + history_sentences:
        for pre, ind, rtype, post, position in sentence.links:
            if pre not in node_to_id or post not in node_to_id:
                continue
            
            output_string += f"(n{node_to_id[pre]})-[:{rtype.name}]->(n{node_to_id[post]}),\n"
            # relations.append({
            #     'startId:START_ID': node_to_id[pre],
            #     'endId:END_ID': node_to_id[post],
            #     'type:TYPE': f'{rtype.name}',
            #     'indicator': ind,
            # })
            
    output_string = output_string[:-2] + ';'
    Path('figs/neo4j/nodes_relations.cypher').write_text(output_string)
    
                
    # pd.DataFrame(relations).to_csv('figs/neo4j/relationships.csv', index=False)


def dependency_to_neo4j(dep_pairs, history_sentences):
    output_string = 'MATCH (n) DETACH DELETE n;\nCREATE'
    
    dep_pairs = set(dep_pairs)
    node_set = set()
    for pre, post, rtype in dep_pairs:
        node_set.update([pre, post])
        
    history_nodes = set()
    for sentence in history_sentences:
        for pre, ind, rtype, post, _ in sentence.links:
            history_nodes.update([pre, post])
        
    node_to_id = {node: f'n{index}' for index, node in enumerate(node_set, 1)}
    for node, index in node_to_id.items():
        output_string += f"({node_to_id[node]}:{'DepNode' if node not in history_nodes else 'DepHistoryNode'} {{name: '{node}'}}),\n"
    
    for pre, post, rtype in dep_pairs:
        output_string += f"({node_to_id[pre]})-[:{rtype}_dep]->({node_to_id[post]}),\n"
    Path('figs/neo4j/dependency_relations.cypher').write_text(output_string[:-2] + ';')