import sys
from pathlib import Path
import pandas as pd

# sys.path.append(str(Path(__file__).absolute().parent.parent.parent))

from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from pathlib import Path

# import argparse, cmd, os, itertools, json
# try:
#     import readline
# except:
#     print("windows system: readline is invalid")

from collections import Counter
from lexer import Sentence, Phrase, Word
from matcher import PatternSetHandler, read_patterns, segment_noun_snippets

ph = PatternSetHandler(
    read_patterns(Path(__file__).parent / "basic_patterns.json")
)

from string import punctuation

def normalize_node(text):
    new_text = ''
    for char in text.lower().strip():
        if char == ' ':
            new_text += '_'
        elif 'a' <= char <= 'z':
            new_text += char
    
    return new_text.strip('_ ')

def triplets_to_neo4j():
    node_statements, relation_statements = Counter(), []
    for pred, ind, rtype, post, section in [line.split('\t') for line in (Path('./') / 'triplets').read_text().split('\n')]:
        pred_id = normalize_node(pred)
        post_id = normalize_node(post)
        ind = normalize_node(ind)
        rtype = rtype if rtype != 'cause-effect' else 'ce'
        rtype = rtype if rtype != 'effect-cause' else 'ec'

        if len(pred_id) <= 3 or len(post_id) <= 3:
            continue

        node_statements.update([
            f"CREATE ({pred_id}:Node {{name: '{pred_id}'}})",
            f"CREATE ({post_id}:Node {{name: '{post_id}'}})",
        ])

        relation_statements.append(
            f'CREATE ({pred_id})-[:{rtype}_{ind}]->({post_id})'
        )

    output_str = 'MATCH(n) DETACH DELETE n\n'

    output_str += '\n'.join(
        [k for k, v in node_statements.most_common()] + relation_statements
    )

    Path('counter').write_text(str(node_statements))
    Path('neo4j.statements').write_text(output_str)

    print(f'node count: {len(node_statements)}')
    print(f'node count: {len(relation_statements)}')

def entity_to_text(entity):
    strings = []
    for word in entity:
        strings.append(f'{word.text}_{word.pos}_{word.index}')

    return ' '.join(strings)

if __name__ == '__main__':
    print(list(Path('D:\\实验室\\2024_03_05课程大纲\\数据\\syllabus\\wsln_min\\高远实验20241105').glob('*.txt')))

    for path in sorted(Path('D:\\实验室\\2024_03_05课程大纲\\数据\\syllabus\\wsln_min\\高远实验20241105').glob('*.txt')):
        tuples = []

        for index, line in enumerate(tqdm(path.read_text(encoding='utf-8').split('\n'), desc=path.name)):
            if not line: continue
            sentence = Sentence(line)
            for relation in [relation for snippet in segment_noun_snippets(sentence.phrase_list, padding=Phrase([Word('<B>', 'NN', 100)], 'NN', 100))
                for relation in ph.extract_relations(snippet)]:

                    tuples.append({
                            'index': index,
                            'pre': entity_to_text(relation.pre_entity),
                            'tag': entity_to_text(relation.indicator), 
                            'type': relation.rtype, 
                            'post': entity_to_text(relation.post_entity),
                    })

        df = pd.DataFrame(tuples)

        df.to_csv(f'D:\\实验室\\2024_03_05课程大纲\\数据\\syllabus\\wsln_min\\高远实验20241105\\output\\{path.name}.csv', encoding='utf-8', index=False)
