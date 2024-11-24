import nltk
from pathlib import Path
from tabulate import tabulate
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

def load_med_rag(summary_n = 100):
    files = [
        dataset_dir / 'med_rag_textbooks/sentences/Anatomy_Gray'
    ]
    data_pairs = []
    for file in files:
        gold_summary = ''
        input = ''
        
        for index, line in enumerate(file.read_text().split('\n'), 1):
            text, position = line.split('\t')
            
            if index <= summary_n:
                gold_summary += text + ' '
            else:
                input += text + ' '

        data_pairs.append(DataPair(file, input, gold_summary))
    
    return data_pairs


def perform_textrank(text, N):
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
            value = f'{p}/{r}/{f1}'
        values.append(value)
        headers.append(metric)
    
    row = [method_name, *values]
    return row, headers


def main():
    data_pairs = load_med_rag()
    references = [pair.reference for pair in data_pairs]
    
    N = 100
    output_dir = dataset_dir / f'med_rag_textbooks/output/textrank_n_{N}'
    if not output_dir.exists():
        output_dir.mkdir()
        predictions = []
        for pair in data_pairs:
            prediction = perform_textrank(pair.input, N)

            predictions.append(prediction)
            (output_dir / pair.path.name).write_text(prediction)
    else:
        predictions = [pred
            for pair in data_pairs
            for pred in (output_dir / pair.path.name).read_text()
        ]

    row, headers = compose_row(predictions, references)
    table = tabulate([row], headers = headers, tablefmt = 'fancy_grid')
    print(table)

if __name__ == '__main__':
    main()


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