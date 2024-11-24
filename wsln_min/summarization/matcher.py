#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   matcher.py
@Time    :   2023/04/30 15:41:45
@Author  :   Li Jiazheng
@Contact :   lee_jiazh@163.com
@Desc    :   Segment the Sentence into nominal snippets, and return the semantic links within nominal snippet.
'''

import re
from pathlib import Path
from typing import Union, Optional, List, Any, Tuple, Dict, Set
from lexer import Token, Word, Phrase, NULL_PHRASE
from utils import load_json_to_meta, is_noun, DEFAULT_NOUN_POS, pattern_pos_class_mapper, BE_words, is_verb, is_conj, is_prep, is_adv, is_adj
from collections import namedtuple

__all__ = ['Relation',
           'read_patterns', 'PatternSetHandler']

Relation = namedtuple('Relation', ['pre_entity', 'indicator', 'rtype', 'post_entity'])


def segment_noun_snippets(word_list: List[Word], padding=None) -> List[List[Word]]:
    """Cut the sentence inobj noun snippets.

    If the first word or the last word is not noun, then the first noun snippet is from 0 obj the position.
    """
    word_list = word_list.copy()
    previous_noun_index = 0
    snippets = []

    if padding:
        if not is_noun(word_list[0]):
            word_list = [padding] + word_list 
        if not is_noun(word_list[-1]):
            word_list.append(padding)

    for word_index, word in enumerate(word_list[1:-1], 1):
        if is_noun(word):
            snippets.append(word_list[previous_noun_index:word_index+1])
            previous_noun_index = word_index
    else:
        snippets.append(word_list[previous_noun_index:])

    return snippets


class PatternItem:

    def __init__(self, definition: str):
        """
        definition: it has the formal of `{TYPE}{ID}[({WORD1}|{WORD2}|...)]`
        The `{TYPE}{ID}` is called itemid
        """
        self.definition = definition
        result = re.search(r"(\w+)(\(.+\))*", definition)
        self.itemid = result.group(1)
        self.pos_type = re.search(r"([a-zA-Z]+)(\d*)", self.itemid).group(1)

        self.candidate_words = []
        if self.pos_type == "I":
            self.candidate_words = result.group(2)[1:-1].split("|")

    def match(self, words: List[Token]) -> int:
        if self.pos_type == "I":
            return self._match_indicator(words, self.candidate_words)
        elif self.pos_type == "BE":
            return self._match_indicator(words, BE_words)
        return 1 if words[0].pos in pattern_pos_class_mapper[self.pos_type] else 0

    def _match_indicator(self, words: List[Token], cand_words) -> int:
        # handle the phrase with multiple words
        word_texts = [word.text for word in words]
        for length in range(len(words), 0, -1):
            if " ".join(word_texts[:length]) in cand_words:
                return length
        return 0

    def __eq__(self, other) -> bool:
        if type(other) is not PatternItem:
            return False
        return self.definition == other.definition

    def __repr__(self):
        return f"PatternItem('{self.definition}')"


class Pattern:
    """
    A pattern is like `N1 I23(don't|do not|never) V N22`
    """

    def __init__(self, definition, *args):
        """
        args nominate the pre_entity/indicator/rtype/
        """
        if args:
            self.pre_entity_id, self.indicator_id, self.rtype, self.post_entity_id = args

        self.item_list = [
            PatternItem(item) for item in re.findall(r"(\w+\(.*\)|\w+)", definition)
        ]
    
    def __iter__(self) -> List[PatternItem]:
        return iter(self.item_list)

    def extract_relations(self, words: List[Token]) -> List[Relation]:
        """Extract relations from words according obj a given pattern

        :return indexes: The indexes of link words, which should be removed in the next extraction.
        """
        if not (all_itemid_to_word := self.match(words)):
            return []

        return [Relation(
            itemid_to_word.get(self.pre_entity_id, NULL_PHRASE),
            itemid_to_word.get(self.indicator_id, NULL_PHRASE),
            self.rtype,
            itemid_to_word.get(self.post_entity_id, NULL_PHRASE),
        ) for itemid_to_word in all_itemid_to_word]

    def match(self, words: List[Token]):
        return self._match(words, self.item_list, {})

    def _match(self, words: List[Word], item_list: List[PatternItem], itemid_to_word: Dict) -> List[Dict[str, Union[Word, List[Word]]]]:
        """
        words: a sequence of words is pended obj match with the item_list
        item_list: a list of PatterItem
        mapper: the matched word 
        """
        if len(item_list) == 0:
            return [itemid_to_word]
        if len(words) == 0:
            return []

        all_itemid_to_word = []

        while len(words) > 0:
            if (length := item_list[0].match(words)) == 0:
                words = words[1:]
                continue
            
            itemid_to_word = itemid_to_word.copy()
            itemid_to_word[item_list[0].itemid] = Phrase(words[:length]) if length > 1 else words[0]

            words = words[length:]
            all_itemid_to_word += self._match(
                words,
                item_list[1:],
                itemid_to_word
            )

        return all_itemid_to_word


def read_patterns(file_path: Path) -> List[Pattern]:
    patterns = []
    for link_def in load_json_to_meta(file_path).links:
        type = link_def.type
        for pattern_def in link_def.patterns:
            patterns.append(Pattern(
                pattern_def.pattern,
                pattern_def.from_id,
                pattern_def.indicator,
                type,
                pattern_def.to_id,
            ))

    return patterns


class PatternSetHandler:

    def __init__(self, patterns: List[Pattern], whitelist=None):
        self.patterns = patterns
        if whitelist:
            self.patterns = [
                pattern for pattern in self.patterns if pattern.rtype in whitelist
            ]

    def extract_relations(self, words: List[Word]) -> List[Relation]:
        """
        :param words: a nominal snippet that the first word and the last word is a nominal
        """
        # generate a sequential link if the input nominal snippet only contains two nouns
        if len(words) == 2 and is_noun(words[0]) and is_noun(words[1]):
            return [Relation(words[0], NULL_PHRASE, "sequential", words[1])]

        # all_itemid_to_word = [
        #     itemid_to_word
        #     for pattern in self.patterns
        #     for itemid_to_word in pattern.match(words)
        # ]

        relations = [
            relation
            for pattern in self.patterns
            for relation in pattern.extract_relations(words)
        ]

        if len(relations) == 0 and is_noun(words[0]) and is_noun(words[-1]):
            relations.append(Relation(words[0], NULL_PHRASE, "sequential", words[-1]))

        # If the indicator has been extracted, then remove the link.
        filtered_relations, indicator_set = [], set()
        for relation in relations:
            if relation.indicator not in indicator_set:
                filtered_relations.append(relation)
                indicator_set.add(relation.indicator)

        return filtered_relations