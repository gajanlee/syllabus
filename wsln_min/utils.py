import json
from types import SimpleNamespace

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
