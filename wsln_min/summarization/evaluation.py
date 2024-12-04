import nltk
import multiprocessing
from pathlib import Path
from tabulate import tabulate
from multiprocessing import Process
from collections import namedtuple
from pyAutoSummarizer.base import summarization
from functools import partial
from tqdm import tqdm
# ROUGE 1, ROUGE 2, ROUGE L, ROUGE S, BLEU, METEOR
from rouge_score import rouge_scorer
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

DataPair = namedtuple('DataPair', ['path', 'input', 'reference'])
EvalPair = namedtuple('EvalPair', ['reference', 'prediction'])

dataset_dir = Path(__file__).parent.parent.parent / 'datasets'

def split_text_summary(file, summary_n = 100):
    normal_list, reference_list = [], []
    for index, line in enumerate(file.read_text(encoding='utf-8').split('\n'), 1):
        if not line: continue

        text, position = line.split('\t')
        
        if index <= summary_n:
            reference_list.append(text)
        else:
            normal_list.append(text)

    return ' '.join(normal_list), ' '.join(reference_list)

def load_med_rag():
    '''
    长上下文，前100个句子作为金标准
    '''
    files = (dataset_dir / 'med_rag_textbooks/sentences').glob('*')
    
    data_pairs = []
    for file in tqdm(files, desc='loading med rag'):
        if file.name not in ['Anatomy_Gray', 'Biochemistry_Lippincott', 'First_Aid_Step1', 'First_Aid_Step2', 'Pathoma_Husain', 'Pediatrics_Nelson', 'Psichiatry_DSM-5']:
            continue

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
        text = ' '.join(file.read_text(encoding='utf-8').split('\n'))
        summary = (dataset_dir / f'bigsurvey/MDS_truncated/summary/{file.name}').read_text(encoding='utf-8')

        data_pairs.append(DataPair(file, text, summary))

    return data_pairs

def textrank_func(text, N = 200):
    '''
    N: sentence count
    '''
    parameters = { 'stop_words':        ['en'],
        'n_words':           -1,
        'n_chars':           -1,
        'lowercase':         True,
        'rmv_accents':       True,
        'rmv_special_chars': True,
        'rmv_numbers':       False,
        'rmv_custom_words':  [],
        'verbose':           False
    }

    smr = summarization(text, **parameters)
    rank = smr.summ_text_rank(iteration = 1000, D = 0.85, model = 'all-MiniLM-L6-v2')
    generated_summary = smr.show_summary(rank, n = N)
    
    return generated_summary


def lexrank_func(text, N = 200):
    parameters = { 'stop_words':        ['en'],
        'n_words':           -1,
        'n_chars':           -1,
        'lowercase':         True,
        'rmv_accents':       True,
        'rmv_special_chars': True,
        'rmv_numbers':       False,
        'rmv_custom_words':  [],
        'verbose':           False
    }
    smr = summarization(text, **parameters)
    rank = smr.summ_lex_rank(iteration = 1000, D = 0.85)
    generated_summary = smr.show_summary(rank, n = N)
    
    return generated_summary

def lsa_func(text, N = 200):
    parameters = { 'stop_words':        ['en'],
        'n_words':           -1,
        'n_chars':           -1,
        'lowercase':         True,
        'rmv_accents':       True,
        'rmv_special_chars': True,
        'rmv_numbers':       False,
        'rmv_custom_words':  [],
        'verbose':           False
    }
    smr = summarization(text, **parameters)
    rank = smr.summ_ext_LSA(embeddings = False, model = 'all-MiniLM-L6-v2')
    generated_summary = smr.show_summary(rank, n = N)

    return generated_summary

def kl_func(text, N = 200):
    parameters = { 'stop_words':        ['en'],
        'n_words':           -1,
        'n_chars':           -1,
        'lowercase':         True,
        'rmv_accents':       True,
        'rmv_special_chars': True,
        'rmv_numbers':       False,
        'rmv_custom_words':  [],
        'verbose':           False
    }
    smr = summarization(text, **parameters)
    rank = smr.summ_ext_KL(n=3)
    generated_summary = smr.show_summary(rank, n = N)
    
    return generated_summary

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
        # # 最长1024，TODO: 层次式方法
        # 'bart_len_1250': (
        #     partial(summarization.summ_ext_bart, model = 'facebook/bart-large-cnn', max_len = 250),
        #     lambda r: r,
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

    return score_mapper

def get_avg_scores(score_mappers):
    avg_mapper = {}
    for score_mapper in score_mappers:
        for metric, scores in score_mapper.items():
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
        else:
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

def perform_baseline(loader):
    '''
    '''

    baseline_funcs = [
        partial(textrank_func, N=200),
        partial(lexrank_func, N=200),
        partial(lsa_func, N=200),
        # partial(kl_func, N=200),
    ]

    data_pairs, dataset_output_dir = loader()
    # dataset_output_dir = dataset_dir / f'med_rag_textbooks/output'

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
    
    prediction_mapper = {}
    references = [word_limit(pair.reference) for pair in data_pairs]

    for dir in dataset_output_dir.glob('*'):
        if not dir or dir.name.endswith('bak'): continue
        print(f'find model result of {dir.name}')
        for pair in data_pairs:
            prediction_mapper[dir.name] = prediction_mapper.get(dir.name, []) + [
                word_limit(
                    (dir / pair.path.name).read_text(encoding='utf-8')
                )
            ]

    rows = []
    for model_name, predictions in tqdm(prediction_mapper.items(), desc='calculation rouge'):
        row, headers = compose_row(predictions, references, model_name)
        rows.append(row)

    table = tabulate(rows, headers = headers, tablefmt = 'fancy_grid')
    print(table)

if __name__ == '__main__':
    # perform_baseline(load_med_rag)
    perform_baseline(load_khanacademy)
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