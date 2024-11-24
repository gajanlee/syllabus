import sys
from pathlib import Path

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

def text_to_triplets():
    triplets = []

    for path in sorted(Path('/mnt/e/project/coref_resolve/rsm_coref/').glob('*')):
        for para_index, paragraph in enumerate(tqdm(path.read_text().split('\n'), desc=path.name), 1):
            for sent_index, sentence_text in enumerate(sent_tokenize(paragraph), 1):
                sentence = Sentence(sentence_text)
                for relation in [relation for snippet in segment_noun_snippets(sentence.phrase_list, padding=Phrase([Word('<B>', 'NN', 100)], 'NN', 100))
                        for relation in ph.extract_relations(snippet)]:
                    triplets.append(
                        (relation.pre_entity.text, relation.indicator.text, relation.rtype,     relation.post_entity.text, f"{path.name.split(' ')[0]}-{para_index}-{sent_index}")
                    )

        (Path('./') / 'triplets').write_text('\n'.join(
            f'{pre}\t{ind}\t{rtype}\t{post}\t{section}' for pre, ind, rtype, post, section in triplets
        ))


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

if __name__ == '__main__':

    sentences = [
        ''
    ]

    for line in Path('foundations_of_database.txt').read_text().split('\n'):
        if not line:
            sentences.append('')
            continue

        if not ('a' < line[0] < 'z' or 'A' < line[0] < 'Z'):
            continue
        
        sentences[-1] += f'{line} '


    sentence_output_file = Path('foundations_of_database.sentences').open('a')
    
    for paragraph_index, paragraph in enumerate(tqdm([s for s in sentences if s]), 1):
        triplets = []
        # output_file = Path('foundations_of_database.triplets').open('a')
        

        for sent_index, sentence_text in enumerate(sent_tokenize(paragraph), 1):
            # sentence = Sentence(sentence_text)
            # for relation in [r for snippet in segment_noun_snippets(sentence.phrase_list, padding=Phrase([Word('<B>', 'NN', 100)], 'NN', 100))
            #         for r in ph.extract_relations(snippet)]:

            #     triplet = (
            #         (relation.pre_entity.text, relation.indicator.text, relation.rtype,     relation.post_entity.text, f"{paragraph_index}-{sent_index}")
            #     )

            #     triplets.append(
            #         '\t'.join(triplet)
            #     )
            
            sentence_output_file.write(f'{sentence_text}\tx-{paragraph_index}-{sent_index}\n')
        
    sentence_output_file.close()

        # output_file.write('\n'.join(triplets) + '\n')
        # output_file.close()
