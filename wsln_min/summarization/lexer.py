#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   lexer.py
@Time    :   2023/04/07 16:38:39
@Author  :   Li Jiazheng
@Contact :   lee_jiazh@163.com
@Desc    :   Input a string of a sentence, convert to an object of Sentence with dependency
'''

import string
import nltk
import stanza
import copy
from stanza.utils.conll import CoNLL
from stanza.server import ud_enhancer
from typing import List, Tuple, Union
from collections import namedtuple

from utils import DEFAULT_NOUN_POS

__all__ = ['Token', 'Word', 'Phrase', 'Sentence']

symbols = r"！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏."
symbols += string.punctuation + "\\"
symbols += "".join([chr(i) for i in range(945, 970)])
digits = "1234567890"

class Token:
    """Token is an abstract class to represent the  
    
    Description
    
    Attributes:

    """


class Word(Token):
    """Word is ...
    
    Description:
        ddd
        
    Attributes:
        text: str, 
    
    """

    def __init__(self, text: str, pos: str = "UNK", index: int = -1):
        self.text = text
        self.pos = pos
        self.index = index
    
    def __repr__(self) -> str:
        return f"Word('{self.text}', '{self.pos}', {self.index})"

    def __str__(self) -> str:
        return self.text

    def __eq__(self, other) -> bool:
        if type(other) is not Word:
            return False
        return self.text == other.text and self.pos == other.pos and self.index == other.index

    def __hash__(self):
        return hash(repr(self))


class Phrase(Token):

    def __init__(self, words: List[Word], pos: str = "UNK", index: int = -1):
        self.pos, self.index = pos, index
        self.words = []
        for word in words:
            self.append(word)

        if len(words) == 1:
            self.pos = words[0].pos
            self.index = words[0].index

    @property
    def text(self) -> str:
        return " ".join(to.text for to in self.words)
    
    @property
    def indexes(self) -> List[int]:
        return [t.index for t in self.words]

    @property
    def core_word(self) -> Word:
        if self.index == -1 or len(self) <= 1:
            return None
        for word in self.words:
            if self.index == word.index:
                return word
        raise Exception(f"can't find the core word of {self}")

    def append(self, word) ->  None:
        if type(word) is Word:
            self.words.append(word)
        elif type(word) is Phrase:
            self.words += word.words
        else:
            raise Exception(f"invalid type of {word} with {type(word)}, expected Word or Phrase")

        self.words = sorted(self.words, key=lambda t: t.index)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            return self.words[index]

        raise Exception("the index must be a int or slice")

    def __iter__(self):
        return iter(self.words)

    def __str__(self) -> str:
        return self.text

    def __repr__(self):
        return f"Phrase({self.words}, '{self.pos}', {self.index})"
    
    def __eq__(self, other) -> bool:
        if type(other) is not Phrase:
            return False
        return self.words == other.words and self.pos == other.pos and self.index == other.index
    
    def __hash__(self):
        return hash(repr(self))

NULL_PHRASE = Phrase([Word('')])

class DepTree:
    
    def __init__(self, tree=None, root=None):
        self.tree = tree if tree else {}
        self.root = root if root else None
        
    def append(self, head: Token, child: Token, deprel: str):
        self.tree[head] = self.tree.get(head, {})
        self.tree[head][child] = deprel

    def set_root(self, root):
        self.root = root

    def iterate(self, reverse=False):
        iter_sequence = []
        nodes = [self.root]
        while nodes:
            head = nodes[0]; nodes = nodes[1:]
            if head not in self.tree: continue
            for child, deprel in self.tree[head].items():
                nodes.append(child)
                iter_sequence.append((head, child, deprel))
        
        return iter_sequence if not reverse else iter_sequence[::-1]    
    
    def get_children(self, node, deprel=None):
        children = list(self.tree[node].keys()) if node in self.tree else []
        if deprel:
            return [child for child in children if self.tree[node][child] == deprel]
        return children
    
    def remove_dep(self, head, child):
        del self.tree[head][child]
        
        # move the children of the child to the head
        if child in self.tree:
            for sub_child, deprel in self.tree[child].items():
                self.tree[head][sub_child] = deprel
    
    def __iter__(self):
        yield from self.iterate()
    
    def __eq__(self, other):
        if not isinstance(other, DepTree):
            return False
        return self.root == other.root and self.tree == other.tree

    def __repr__(self) -> str:
        string = ""
        for index, (head, child, deprel) in enumerate(self.iterate(reverse=False), 1):
            string += f"{index}\t{head}--{child}-->{deprel}\n"
        return string


class Sentence:

    def __init__(self, text: str):
        """
        remove the punctuations.
        """
        # 删除一些对关系不重要的词
        block_words = {"then"}
        text = " ".join(
            [word for word in nltk.word_tokenize(text) if word not in block_words]
        )
        self.text = text
        
        self.phrase_list, self.phrase_dep_tree = self.preprocess(text)
        # self.phrase_list, self.phrase_dep_tree, self.phrase_enhanced_dep = self.merge_phrase(self.word_list, self.word_dep_tree, self.word_enhanced_dep)

        if not self.phrase_list:
            return

    def get_phrase(self, index):
        if isinstance(index, (int, slice)):
            return self.word_list[index]

        raise Exception("the index must be an integer or slice")
    
    def preprocess(self, text: str) -> Tuple[List[Word], DepTree, DepTree]:
        """
        Return the word_list and the dependency tree for the input text.
        """
        word_list, word_dep_tree, enhanced_dependency = parse_wordList_depTree_stanza(text)
        phrase_list, phrase_dep_tree = merge_phrase(word_list, word_dep_tree)
        phrase_list, phrase_dep_tree = remove_punct(phrase_list, phrase_dep_tree)
        return phrase_list, phrase_dep_tree

    @property
    def word_list(self) -> List[Word]:
        return [word for phrase in self.phrase_list for word in phrase.words]

    @property
    def words(self) -> List[str]:
        return [word.text for word in self.word_list]

    def __len__(self) -> int:
        return len(self.word_list)

    def __eq__(self, other):
        if not type(other) is Sentence:
            return False
        return self.word_list == other.word_list
    
    def __str__(self) -> str:
        return self.text

    def __repr__(self):
        return f"Sentence({self.text})"


# download method is 2 means that `DownloadMethod.REUSE_RESOURCES` in the stanza.core package 
stanza_parser = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', logging_level="ERROR", use_gpu=False, download_method=None)

def parse_wordList_depTree_stanza(text):
    # Parse the basic and enhanced++ universal dependency for word segmentation and part-of-speeches
    with ud_enhancer.UniversalEnhancer(language="en") as enhancer:
        basic_dependency = stanza_parser(text)
        depparse_enhanced = enhancer.process(basic_dependency)
        enhanced_dependency = depparse_enhanced.sentence[0].enhancedDependencies.edge

    word_list = []
    # Construct the word list with the corresponding index, where the index is counted from 0
    for index, data in enumerate(basic_dependency.to_dict()[0], 1):
        word, pos_tag = data["text"], data["xpos"]
        if data["upos"] == "X":     # X means other, which can be regarded as a nominal
            pos_tag = DEFAULT_NOUN_POS
        if data["upos"] in ["SYM", "PUNCT"]:
            pos_tag = "SYM"
        # TODO: check all the pos tags are under control
        
        word_list.append(Word(word, pos_tag, index))
        
    word_dep_tree, enhanced_dependency = _construct_dep_tree(word_list, basic_dependency, enhanced_dependency)
    
    return word_list, word_dep_tree, enhanced_dependency

def _construct_dep_tree(word_list, basic_dependency, enhanced_dependency):
    word_dep_tree = DepTree()
    
    # Construct the basic universal dependency tree
    for index, data in enumerate(basic_dependency.to_dict()[0], 1):
        child, deprel = word_list[data["id"] - 1], data["deprel"]
        if deprel == "root":
            word_dep_tree.set_root(child)
            continue
        elif deprel == "advmod" and str(child) == "over":
            deprel = "case"
        # elif deprel == "expl":
        #     deprel = "nsubj"
        # elif deprel == "appos":
        #     deprel = "conj"

        word_dep_tree.append(word_list[data["head"] - 1], child, deprel)

    # Construct the enhanced++ universal dependency tree
    return_enhanced_dep_list = []
    for dep in enhanced_dependency:
        # TODO: 是否所有的extra语义关系都要拿出来？
        # if dep.isExtra:
        dep_type = dep.dep.split(":")[0]
        # if dep.isExtra:
        if dep_type in ["nsubj", "obj"]:
            return_enhanced_dep_list.append((word_list[dep.source - 1], word_list[dep.target - 1], dep_type))

            # if dep.dep in ["nsubj", "ccomp", "csubj", "xcomp", "obj"]:
            #     head, child, deprel = dep.source, dep.target, dep.dep

            #     self.subj_deps.append((index_to_phrase[head], index_to_phrase[child], deprel))
            # if dep.dep in ["obj", "ccomp", "xcomp"]:
            #     self.obj_deps.append((index_to_phrase[head], index_to_phrase[child], deprel))

    return word_dep_tree, return_enhanced_dep_list


def merge_phrase(word_list, word_dep_tree):
    index_to_phrase = {word.index: Phrase([word], pos=word.pos, index=word.index) for word in word_list}
    for head, child, deprel in word_dep_tree.iterate(reverse=True):
        if (
            (_is_modifier(deprel) and not word_dep_tree.get_children(child)) or
            child.text in ["'s", "’s"] or
            deprel == "nummod"
        ):
            child_phrase = index_to_phrase[child.index]
            if str(child_phrase) not in ["Meanwhile", "there", "Initially"]:
                index_to_phrase[head.index].append(child_phrase)
            del index_to_phrase[child.index]
            word_dep_tree.remove_dep(head, child)

    phrase_list = [index_to_phrase[index] for index in sorted(index_to_phrase.keys())]
    
    phrase_dep_tree = DepTree(root=index_to_phrase[word_dep_tree.root.index])
    for head, child, deprel in word_dep_tree:
        phrase_dep_tree.append(
            index_to_phrase[head.index], index_to_phrase[child.index], deprel,
        )

    return phrase_list, phrase_dep_tree

def _is_modifier(deprel):
    """
    TODO: dep
    """
    dep_type = deprel.split(":")[0]
    return (
        dep_type.endswith("mod") or
        # dep_type.endswith("comp") or    # was renamed, Jones are merged by "xcomp"
        dep_type in ["det", "aux", "compound", "fixed", "flat", "appos"]
    )

def remove_punct(token_list, token_dep_tree):
    """Remove the punctuations in the token_list and the corresponding 
    dependencies in dep_tree.
    """
    token_list = [token for token in token_list]
    dep_tree = DepTree()

    for head, child, deprel in token_dep_tree:
        if deprel != "punct":
            dep_tree.append(head, child, deprel)
        else:
            token_list.remove(child)
    
    dep_tree.set_root(token_dep_tree.root)
    return token_list, dep_tree