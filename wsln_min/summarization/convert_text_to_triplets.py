import sys
import json
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
    read_patterns(Path(__file__).parent.parent / "basic_patterns.json")
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


if __name__ == '__main__':
    data_set_dir = Path(__file__).parent.parent.parent / 'datasets'
    
    med_rag_dir = data_set_dir / 'med_rag_textbooks'
    # med_rag_files = [med_rag_dir / 'chunk/Anatomy_Gray.jsonl']
    med_rag_files = [med_rag_dir / 'chunk/Biochemistry_Lippincott.jsonl']

    for file in med_rag_files:
        # summary_path = file.parent.parent / 'sentences' / f'{file.name}.summary'
        # summary_links_path = file.parent.parent / 'links' / f'{file.name}.summary.triplets'
        name = file.name[:-6]   # remove '.jsonl'
        normal_sentences_path = file.parent.parent / 'sentences' / name
        normal_links_path = file.parent.parent / 'links' / f'{name}.triplets'
        
        for para_index, line in tqdm(list(
            enumerate(file.read_text().split('\n'))), desc=file.name):

            sentences, links = [], []
            # # summary
            # if para_index <= 2:
            #     pass
            
            normal_sentences_file = normal_sentences_path.open('a')
            normal_links_file = normal_links_path.open('a')
            
            paragraph = json.loads(line)['content']
            
            for sent_index, sentence_text in enumerate(sent_tokenize(paragraph), 1):
                position = f'x-{para_index}-{sent_index}'
                sentences.append(f'{sentence_text}\t{position}')

                sentence = Sentence(sentence_text)
                for relation in [r
                    for snippet in segment_noun_snippets(
                        sentence.phrase_list,
                        padding=Phrase([Word('<B>', 'NN', 100)], 'NN', 100)
                    )
                    for r in ph.extract_relations(snippet)
                ]:
                    links.append('\t'.join([relation.pre_entity.text,
                                relation.indicator.text,
                                relation.rtype,
                                relation.post_entity.text,
                                position]))

            normal_sentences_file.write('\n'.join(sentences) + '\n')
            normal_links_file.write('\n'.join(links) + '\n')
            
            normal_sentences_file.close()
            normal_links_file.close()

