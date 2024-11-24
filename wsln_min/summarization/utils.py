import termcolor    # 命令行变色
import json
from types import SimpleNamespace

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
