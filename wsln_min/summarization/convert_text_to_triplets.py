import sys
import multiprocessing
import json
from pathlib import Path
from multiprocessing import Process, Queue

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

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
from lexer import Sentence, Phrase, Word, Document
from matcher import PatternSetHandler, read_patterns, segment_noun_snippets

ph = PatternSetHandler(
    read_patterns(Path(__file__).parent.parent / "basic_patterns.json")
)

def text_to_triplets():
    triplets = []

    for path in sorted(Path('/mnt/e/project/coref_resolve/rsm_coref/').glob('*')):
        for para_index, paragraph in enumerate(tqdm(path.read_text(encoding='utf-8').split('\n'), desc=path.name), 1):
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

import time
def sentence_to_links_str(sentence: Sentence, position):
    '''
    返回一个字符串
    '''
    links = []

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
    
    return '\n'.join(links)


def convert_links(input_file, output_file: Path):
    sentences = []
    for line in tqdm(input_file.read_text(encoding='utf-8').split('\n'), desc=f'extracting sentences: {input_file.name}'):
        if not line: continue
        # sentence_text, position = line.split('\t')
        sentences.append(line)
    
    # 单进程
    link_strings = []
    for sentence in tqdm(sentences):
        link_string = sentence_to_triplets(sentence)
        link_strings.append(link_string)

    # pool = multiprocessing.Pool()
    # link_strings = []
    # for link_string in tqdm(pool.imap(sentence_to_triplets, sentences), total=len(sentences), desc=f'links of {input_file.name}'):
    #     link_strings.append(link_string)

    # pool.close()
    # pool.join()

    output_file.write_text('\n'.join(link_strings), encoding='utf-8')


def load_med_rag(data_set_dir):
    med_rag_dir = data_set_dir / 'med_rag_textbooks'
    data_list = []
    for file in (med_rag_dir / 'chunk').glob('*.jsonl'):
        para_list = []
        for para_index, line in tqdm(list(
            enumerate(file.read_text(encoding='utf-8').split('\n')), 1), desc=file.name):
            
            sentences, links = [], []
            paragraph = json.loads(line)['content']

            para_list.append((paragraph, f'x-{para_index}'))
        
        if len(para_list) > 6000:
            continue

        # remove *.jsonl
        data_list.append((para_list, file.name[:-6]))

    sentence_dir = (med_rag_dir / 'sentences')
    link_dir = (med_rag_dir / 'links')

    return data_list, sentence_dir, link_dir


def convert_med_rag_to_standard():
    '''
    一次性的函数，把之前的convert triplets方法改成标准的，每行的
    '''
    raise NotImplementedError

def load_para_list(file):
    para_list = []
    for para_index, line in enumerate(file.read_text(encoding='utf-8').split('\n'), 1):
        if not line: continue
        para = json.loads(line)

        para_list.append(
            (para['text'].replace('\n', ''), f'{para["position"]}-{para_index}')
        )
    return para_list


def load_khanacademy(data_set_dir):
    '''
    每行
    '''
    data_list = []

    for file in tqdm((data_set_dir / 'cosmopedia/khanacademy/inputs/').glob('*'), 'loading khanacademy ', total=130):
        data_list.append(
            (para_list, file.name)
        )

    sentence_dir = (data_set_dir / 'cosmopedia/khanacademy/sentences')
    link_dir = (data_set_dir / 'cosmopedia/khanacademy/links')
    
    return data_list, sentence_dir, link_dir

def load_bigsurvey(dataset_dir):
    data_list = []

    for file in (data_set_dir / 'bigsurvey/MDS_truncated/inputs').glob('*'):
        para_list = load_para_list(file)
        data_list.append(
            (para_list, file.name)
        )
    
    sentence_dir = (data_set_dir / 'bigsurvey/MDS_truncated/sentences')
    link_dir = (data_set_dir / 'bigsurvey/MDS_truncated/links')

    return data_list, sentence_dir, link_dir


def test_multiprocessing(data_list):

    for para_list, file_name in tqdm(data_list, desc='files'):
        pool = multiprocessing.Pool()
        for para_index, para in enumerate(tqdm(para_list, desc=file_name), 1):
            for link_string in tqdm(pool.imap(sub, para_list), total=len(para_list), desc=f'links of {file_name}'):
                pass


if __name__ == '__main__':
    '''
    拆分，先转句子，再转links
    '''

    data_set_dir = Path(__file__).parent.parent.parent / 'datasets'

    # data_list, sentence_dir, link_dir = load_med_rag(data_set_dir)
    # data_list, sentence_dir, link_dir = load_khanacademy(data_set_dir)
    data_list, sentence_dir, link_dir = load_bigsurvey(data_set_dir)
    print(f'outputing to {sentence_dir} and {link_dir}')

    if not sentence_dir.exists(): sentence_dir.mkdir()
    if not link_dir.exists(): link_dir.mkdir()

    for para_list, file_name in tqdm(data_list, desc='files'):
        sentence_path = sentence_dir / file_name
        link_path = link_dir / file_name

        if sentence_path.exists() and link_path.exists():
            continue

        for para_index, para_with_position in enumerate(tqdm(para_list, desc=file_name), 1):
            para, para_position = para_with_position
            doc = Document(para)

            sent_string = ''; link_string = ''
            for sent_index, sentence in enumerate(doc.sentences, 1):
                position = f'{para_position}-{sent_index}'
                sent_string += f'{sentence.text}\t{position}\n'
                link_string += sentence_to_links_str(sentence, position) + '\n'
            
            sentence_file = sentence_path.open('a', encoding='utf-8'); sentence_file.write(sent_string)
            link_file = link_path.open('a', encoding='utf-8'); link_file.write(link_string)
            sentence_file.close(); link_file.close()
