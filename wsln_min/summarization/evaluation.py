import nltk
import multiprocessing
import itertools
import ollama

import psutil
import os
import time

from pathlib import Path
from tabulate import tabulate
from multiprocessing import Process
from collections import namedtuple

# from pyAutoSummarizer.base import summarization
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer

from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

LANGUAGE = 'english'

from functools import partial
from tqdm import tqdm
# ROUGE 1, ROUGE 2, ROUGE L, ROUGE S, BLEU, METEOR
from rouge_score import rouge_scorer
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

DataPair = namedtuple('DataPair', ['path', 'input', 'reference'])
EvalPair = namedtuple('EvalPair', ['reference', 'prediction'])

dataset_dir = Path(__file__).parent.parent.parent / 'datasets'

def split_text_summary(file, summary_n = 200):
    normal_list, reference_list = [], []
    for index, line in enumerate(file.read_text(encoding='utf-8').split('\n'), 1):
        if not line: continue

        text, position = line.split('\t')
        
        if index <= summary_n:
            reference_list.append(text)
        else:
            normal_list.append(text)

        # if index > 6000:
        #     break
        if index > 20000:
            break

    return ' '.join(normal_list), ' '.join(reference_list)

def load_med_rag():
    '''
    长上下文，前100个句子作为金标准
    '''
    files = (dataset_dir / 'med_rag_textbooks/sentences').glob('*')
    
    data_pairs = []
    for file in tqdm(files, desc='loading med rag'):
        # if file.name not in ['Anatomy_Gray', 'Biochemistry_Lippincott', 'First_Aid_Step1', 'First_Aid_Step2', 'Pathoma_Husain', 'Pediatrics_Nelson', 'Psichiatry_DSM-5']:
            # continue

        text, summary = split_text_summary(file)
        data_pairs.append(DataPair(file, text, summary))

    output_dir = dataset_dir / 'med_rag_textbooks/output'
    return data_pairs, output_dir


def load_khanacademy():
    files = (dataset_dir / 'cosmopedia/khanacademy/sentences').glob('*')
    data_pairs = []

    for file in tqdm(files, desc='loading khamacademy'):
        text, summary = split_text_summary(file)
        data_pairs.append(DataPair(file, text, summary))

    output_dir = dataset_dir / 'cosmopedia/khanacademy/output'

    return data_pairs, output_dir

def load_bigsurvey():
    data_pairs = []

    for file in tqdm((dataset_dir / 'bigsurvey/MDS_truncated/sentences').glob('*'), desc='loading bigsurvey'):
        text, _ = split_text_summary(file, -1)
        # text = ' '.join(file.read_text(encoding='utf-8').split('\n')[:6000])
        # text = ' '.join(file.read_text(encoding='utf-8').split('\n')[:20000])
        summary = (dataset_dir / f'bigsurvey/MDS_truncated/summary/{file.name}').read_text(encoding='utf-8')

        data_pairs.append(DataPair(file, text, summary))

    return data_pairs, dataset_dir / 'bigsurvey/MDS_truncated/output'

def load_rsm():
    rsm_file = dataset_dir / 'rsm/sentences/rsm'

    reference_sents = []
    norm_sents = []
    for line in rsm_file.read_text(encoding='utf-8').split('\n'):
        if not line: continue
        text, position = line.split('\t')
        if position.startswith('preface'):
            reference_sents.append(text)
        else:
            norm_sents.append(text)

    return [DataPair(rsm_file, ' '.join(norm_sents), ' '.join(reference_sents))], dataset_dir / 'rsm/output'


def chunking_s(sentences, sentence_limit = 3000):
    '''
    按照句子数量进行切割
    '''
    chunks = []
    for index in range(0, len(sentences), sentence_limit):
        chunks.append(sentences[index: index+sentence_limit])

    return chunks


def chunking_w(sentences, word_limit = 500) -> list[str]:
    words_list = [[]]
    for sentence in sentences:
        _words = nltk.word_tokenize(sentence)
        words_list[-1].extend(_words)
        if len(words_list) > word_limit:
            words_list[-1] = words_list[-1][:word_limit]
            words_list.append([])

    return [' '.join(lst) for lst in words_list]


def chunking_token(sents, token_limit = 1000, sentence_N = 200):
    '''
    每1000个token，一个chunk
    '''
    chunks = [[]]
    token_count = 0

    # total_tokens = bart_tokenizer.tokenize(' '.join(sents))

    total_tokens = nltk.word_tokenize(' '.join(sents))

    if len(total_tokens) > sentence_N * 40 or len(sents) > (sentence_N * 1.5):
        # 不求一步到位
        min_length = max(20, int((sentence_N * 20) / (len(total_tokens) / token_limit)))
        print('total tokens is ', len(total_tokens), '; min_length is ', min_length)
    else:
        return None, None

    # token_limit = min(token_limit, len(total_tokens) // (N+1))

    for sent in sents:
        # _tokens = bart_tokenizer.tokenize(sent)
        _tokens = nltk.word_tokenize(sent)
        # 句子长度过长
        if len(_tokens) > token_limit:
            continue
        token_count += len(_tokens)
        
        if token_count > token_limit:
            chunks.append([sent])
            token_count = len(_tokens)
        else:
            chunks[-1].append(sent)

    # 既然每个chunk的输出摘要不一定是一句话，那就全保留。
    if token_count < min_length:
        return chunks[:-1], min_length
    return chunks, min_length


def _sumy_summarizing(text, SummarizerClass, N):
    chunks = chunking_s(nltk.sent_tokenize(text), 5000)
    sentence_per_chunk = min(3000 // len(chunks), 200)

    chunk_summary = ''
    for chunk in tqdm(chunks, desc='chunking summarization'):
        parser = PlaintextParser.from_string(' '.join(chunk), Tokenizer(LANGUAGE))
        stemmer = Stemmer(LANGUAGE)
        summarizer = SummarizerClass(stemmer)
        summarizer.stop_words = get_stop_words(LANGUAGE)

        for sentence in summarizer(parser.document, sentence_per_chunk):
            chunk_summary += sentence._text + ' '

    parser = PlaintextParser.from_string(chunk_summary, Tokenizer(LANGUAGE))
    summary = ''
    for sentence in summarizer(parser.document, N):
        summary += sentence._text + '\n'
    return summary


def luhn_func(text, N = 200):
    return _sumy_summarizing(text, LuhnSummarizer, N)

def textrank_func(text, N = 200):
    '''
    N: sentence count
    '''
    return _sumy_summarizing(text, TextRankSummarizer, N)


def lexrank_func(text, N = 200):
    return _sumy_summarizing(text, LexRankSummarizer, N)


def kl_func(text, N = 200):
    return _sumy_summarizing(text, KLSummarizer, N)


def lsa_func(text, N = 200):
    return _sumy_summarizing(text, LsaSummarizer, N)


# from transformers import pipeline
# bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device='cuda')

# from transformers import BartTokenizer
# bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")


def _chunking_summarization(sentences, token_limit, _chunk_summarizer, sentence_N):
    '''
    _chunk_summarizer: 输入是chunk_sents(一个chunk内的句子列表)，min_length：摘要输出的最小长度
    输出为：chunk_summ_sents（chunk对应的摘要后的分句结果）
    '''
    for _ in range(2):
        summary_sents = []
        chunks, min_length = chunking_token(sentences, token_limit, sentence_N)
        if not chunks:
            summary_sents = sentences
            break

        for chunk_sents in tqdm(chunks, desc='chunking'):
            chunk_summ_sents = _chunk_summarizer(chunk_sents, min_length)
            summary_sents += chunk_summ_sents
        sentences = summary_sents

    if not summary_sents:
        raise Exception('summarization error')

    return summary_sents


def bart_func(text, N = 200):
    # 最多迭代两次，剩下的都作为摘要，筛选前200句就行了
    sentences = nltk.sent_tokenize(text)
    def _bart_summarizer(chunk_sents, min_length):
        summ = bart_summarizer(' '.join(chunk_sents), max_length=min_length*2, min_length=min_length)
        return nltk.sent_tokenize(summ[0]['summary_text'])
    
    summary_sents = _chunking_summarization(sentences, 1000, _bart_summarizer, N)
    return '\n'.join(summary_sents)


def _ollama_summarizer(chunk_sents, min_length, model_name):
    text = ' '.join(chunk_sents)

    response = ollama.chat(model=model_name, messages=[{
        'role': 'user',
        'content': f'Summarize the text: <input text>{text}</input text> Summarize the text within 50 sentences.',},]
    )
    return nltk.sent_tokenize(response['message']['content'])


def _ollama_text_summarizer(text, min_length, model_name):
    text = text
    response = ollama.chat(model=model_name, messages=[{
        'role': 'user',
        'content': f'Summarize the text: <input text>{text}</input text> Summarize the text within 50 sentences.',},]
    )
    print(f'{model_name}: {len(text)}')
    return nltk.sent_tokenize(response['message']['content'])


def llama_3d2_1b_func(text, N = 200):
    # summary_sents = _chunking_summarization(
    #     nltk.sent_tokenize(text),
    #     1000,
    #     partial(_ollama_summarizer, model_name='llama3.2:1b'),
    #     N
    # )
    
    summary_sents = _ollama_text_summarizer(text, 0, model_name='llama3.2:1b')
    # summary_sents = _ollama_summarizer([text], 0, model_name='llama3.2:1b')
    return '\n'.join(summary_sents)

def llama_3d2_3b_func(text, N = 200):
    # summary_sents = _chunking_summarization(
    #     nltk.sent_tokenize(text),
    #     1000,
    #     partial(_ollama_summarizer, model_name='llama3.2:3b'),
    #     N
    # )
    summary_sents = _ollama_text_summarizer(text, 0, model_name='llama3.2:3b')
    # summary_sents = _ollama_summarizer([text], 0, model_name='llama3.2:1b')
    return '\n'.join(summary_sents)

def llama_3d1_8b_func(text, N = 200):
    # summary_sents = _chunking_summarization(
    #     nltk.sent_tokenize(text),
    #     1000,
    #     partial(_ollama_summarizer, model_name='llama3.1'),
    #     N
    # )
    # summary_sents = _ollama_text_summarizer(text, 0, model_name='llama3.1')
    summary_sents = _ollama_text_summarizer(text[:30000], 0, model_name='llama3.1')
    # summary_sents = _ollama_summarizer([text], 0, model_name='llama3.2:1b')
    return '\n'.join(summary_sents)

def phi3_4b_func(text, N = 200):
    '''
    phi3: 3.8b
    '''
    summary_sents = _chunking_summarization(
        nltk.sent_tokenize(text),
        1000,
        partial(_ollama_summarizer, model_name='phi3'),
        N
    )
    return '\n'.join(summary_sents)


def phi3_14b_func(text, N = 200):
    '''
    phi3: 14b
    '''
    summary_sents = _chunking_summarization(
        nltk.sent_tokenize(text),
        1000,
        partial(_ollama_summarizer, model_name='phi3:medium'),
        N
    )
    return '\n'.join(summary_sents)

def mistral_7b_func(text, N = 200):
    summary_sents = _chunking_summarization(
        nltk.sent_tokenize(text),
        1000,
        partial(_ollama_summarizer, model_name='mistral'),
        N
    )
    return '\n'.join(summary_sents)

def deepseek_r1_7b_func(text, N = 200):
    summary_sents = _chunking_summarization(
        nltk.sent_tokenize(text),
        1000,
        partial(_ollama_summarizer, model_name='deepseek-r1:7b'),
        N
    )
    return '\n'.join(summary_sents)


def iteration_summarization(model, text):
    '''
    model: short-context model, iterate
    TODO: 找一个层次式摘要的框架
    '''
    summary = model(text, 200)

    return summary


def get_model_funcs(baseline_name, sentence_n):
    model_funcs = {
        'textrank_n_50': (
            partial(summarization.summ_text_rank, iteration = 1000, D = 0.85, model = 'all-MiniLM-L6-v2'),
            partial(summarization.show_summary, n=50)
        ),
        # 'lexrank_n_50': (
        #     partial(summarization.summ_lex_rank, iteration = 1000, D = 0.85),
        #     partial(summarization.show_summary, n=50)
        # ),
        # 'lsa_n_50': (
        #     partial(summarization.summ_ext_LSA, embeddings = False, model = 'all-MiniLM-L6-v2'),
        #     partial(summarization.show_summary, n=50)
        # ),
        # 'kl_n_50': (
        #     partial(summarization.summ_ext_KL, n=3),
        #     partial(summarization.show_summary, n=50),
        # ),
        # # TODO: 对比长上下文的模型
    }
    
    output_dir = Path(r'D:\实验室\2024_03_05课程大纲\数据\bigsurvey\output')
    
    for model_name, (exec_func, summ_func) in model_funcs.items():
        for source, reference in tqdm(data_pairs, desc=model_name):
            smr = summarization(source, **parameters)
            rank = exec_func(smr)
            summary = summ_func(smr, rank)
            generated_summary = smr.show_summary(rank, n = 50)

            file = Path(output_dir / model_name).open('a')
            file.write(generated_summary + '\n')
            file.close()

    return model_funcs

# ROUGE N
from functools import partial

def calculate_scores(generated, reference):
    # metric_funcs = {
    #     'rouge-1': partial(smr.rouge_N, n = 1),
    #     'rouge-2': partial(smr.rouge_N, n = 2),
    #     'rouge-l':smr.rouge_L,
    #     'rouge-s': partial(smr.rouge_S, skip_distance = 2),
    #     'bleu': partial(smr.bleu, n = 4),
    #     'meteor': smr.meteor,
    # }
    
    score_mapper = {}
    norm = lambda score: round(score * 100, 2)
    
    for metric, score in rouge_scorer.score(reference, generated).items():        
        p, r, f = norm(score.precision), norm(score.recall), norm(score.fmeasure)
        score_mapper[metric] = [p, r, f]

    score_mapper['length-pred/ref'] = [len(nltk.word_tokenize(generated)), len(nltk.word_tokenize(reference))]

    return score_mapper

def get_avg_scores(score_mappers):
    avg_mapper = {}
    for score_mapper in score_mappers:
        for metric, scores in score_mapper.items():
            if metric.startswith('length'):
                avg_mapper[metric] = avg_mapper.get(metric, [0, 0])
            else:
                avg_mapper[metric] = avg_mapper.get(metric, [0, 0, 0])
            for index, score in enumerate(scores):
                avg_mapper[metric][index] += score

    for metric, scores in avg_mapper.items():
        for index, score in enumerate(scores):
            avg_mapper[metric][index] = score / len(score_mappers)

    return avg_mapper

def compose_row(generated_summaries, reference_summaries, method_name):
    score_mappers = [calculate_scores(g, r) for g, r in zip(generated_summaries, reference_summaries)]
    avg_mapper = get_avg_scores(score_mappers)

    values = []; headers = ['Method']
    for metric, scores in avg_mapper.items():
        if len(scores) == 1:
            value = scores[0]
        elif metric.startswith('length'):
            value = f'{scores[0]:.2f}/{scores[1]:.2f}'
        elif metric.startswith('rouge'):
            metric = f'{metric}-P/R/F'
            p, r, f1 = scores
            value = f'{p:.2f}/{r:.2f}/{f1:.2f}'
        values.append(value)
        headers.append(metric)
        
    row = [method_name, *values]
    return row, headers

def summarize_one_pair(input_text, output_path, summarizer, message):
    print(message)
    prediction = summarizer(input_text)
    output_path.write_text(prediction, encoding='utf-8')

def word_limit(text, n = 1250):
        return ' '.join(nltk.word_tokenize(text)[:n])

def perform_baseline(loader, baseline_funcs):
    '''
    N: 摘要sentence数量
    '''
    # baseline_funcs = [
    #     # partial(luhn_func, N=200),
    #     # partial(textrank_func, N=200),
    #     # partial(lexrank_func, N = 200),
    #     # partial(lsa_func, N=200),
    #     partial(bart_func, N=200),

    #     # partial(edmundson_func, N=200),
    #     # partial(kl_func, N=200),
    # ]

    data_pairs, dataset_output_dir = loader()

    for baseline_func in tqdm(baseline_funcs, desc='baselines'):
        predictions = []
        
        output_dir = dataset_output_dir / f'{baseline_func.func.__name__[:-5]}'
        if not output_dir.exists():
            output_dir.mkdir()

        for pair in tqdm(data_pairs, desc=baseline_func.func.__name__[:-5]):
            output_file = output_dir / pair.path.name
            if output_file.exists():
                continue
            summarize_one_pair(pair.input, output_file, baseline_func, pair.path.name)
        
        
        #   停止ollama模型
        # for model in ollama.ps()['models']:
        #     print(f'ollama stop {model.name}')
        #     os.system(f'ollama stop {model.name}')

    # evaluate(data_pairs, dataset_dir)

def evaluate(data_pairs, dataset_output_dir, length):

    prediction_mapper = {}
    references = [pair.reference for pair in data_pairs]

    for dir in itertools.chain(
        dataset_output_dir.glob('*'),
        (dataset_output_dir.parent / 'wsln_output').glob('*')
    ):
        if not dir or dir.name.endswith('bak') or dir.name == '.DS_Store': continue
        print(f'find model result of {dir.name}')
        for pair in data_pairs:
            prediction_mapper[dir.name] = prediction_mapper.get(dir.name, []) + [
                word_limit(
                    (dir / pair.path.name).read_text(encoding='utf-8'),
                    length
                )
            ]
    
    rows = []
    for model_name, predictions in tqdm(prediction_mapper.items(), desc='calculation rouge'):
        row, headers = compose_row(predictions, references, model_name)
        rows.append(row)

    # 按照ROUGE1-F排序
    rows = sorted(
        rows, key=lambda r: r[1].split('/')[-1], reverse=True
    )

    table = tabulate(rows, headers = headers, tablefmt = 'fancy_grid')
    print(table)


if __name__ == '__main__':
    '''
    python evaluation.py run rsm,bigsurvey bart,lsa 200 运行实验
    python evaluation.py eval rsm,bigsurvey 1250 运行评估（单词限制1250）
    '''
    
    dataset_loader = {
        'rsm': load_rsm,
        'med_rag': load_med_rag,
        'khan': load_khanacademy, 
        'bigsurvey': load_bigsurvey,
    }
    
    import sys
    dataset_names = sys.argv[2].split(',')
    if sys.argv[1] == 'eval':
        limit_length = int(sys.argv[3])
        for name in dataset_names:
            data_pairs, dataset_output_dir = dataset_loader[name]()
            print(name)
            evaluate(data_pairs, dataset_output_dir, limit_length)

        exit()
    
    baseline_names = sys.argv[3].split(',')
    sentence_N = int(sys.argv[4])

    print(dataset_names, '\n', baseline_names, '\n', sentence_N)

    baseline_func_mapper = {
        'luhn': luhn_func,
        'textrank': textrank_func,
        'lexrank': lexrank_func,
        'lsa': lsa_func,
        'bart': bart_func,
        'llama3.2_1b': llama_3d2_1b_func,
        'llama3.2_3b': llama_3d2_3b_func,
        'llama3.1_8b': llama_3d1_8b_func,
        'phi3_3.8b': phi3_4b_func,
        'phi3_14b': phi3_14b_func,
        'mistral_7b': mistral_7b_func,
        'deepseek_r1_7b': deepseek_r1_7b_func,
    }

    # for dataset, baseline in itertools.product(dataset_names, baseline_names):

    assert [dataset_loader[dataset] for dataset in dataset_names]
    assert [baseline_func_mapper[baseline] for baseline in baseline_names]

    for dataset in dataset_names:
        if dataset not in dataset_loader:
            raise Exception(f'unknown dataset {dataset}')
        
        process = psutil.Process(os.getpid())

        for baseline in baseline_names:
            start_time = time.time()
            start_memory = process.memory_info().rss / (1024 ** 2)
            
            perform_baseline(dataset_loader[dataset], [
                partial(baseline_func_mapper[baseline], N=sentence_N)
            ])
            
            end_time = time.time()
            end_memory = process.memory_info().rss / (1024 ** 2)
            
            print(f'{baseline} running time: {end_time - start_time:.2f}')
            print(f'{baseline} consumes memory: {end_memory - start_memory:.2f} MB')

        
    # perform_baseline(load_rsm)
    # perform_baseline(load_med_rag)
    # perform_baseline(load_khanacademy)
    # perform_baseline(load_bigsurvey)


    # textrank_row, headers = compose_row(
    #     generates,
    #     references,
    #     'text_rank',
    # )

    # row1, headers = compose_row([generated_summary, generated_summary], [reference_summary, reference_summary], 'text_rank')
    # row2, headers = compose_row([lexrank_g], [reference_summary], 'lexrank')
    # row3, headers = compose_row([lsa_g], [reference_summary], 'lsa')
    # row_bart, headers = compose_row([bart_g], [reference_summary], 'bart')
    # row_t5, headers = compose_row([t5_g], [reference_summary], 't5')
    # table = tabulate([row1, row2, row3, row_bart, row_t5], headers = headers, tablefmt = 'fancy_grid')
    # print(table)