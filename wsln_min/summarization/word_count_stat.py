from evaluation import load_bigsurvey, load_med_rag, load_khanacademy
import nltk
from tqdm import tqdm


dataset_funcs = {
    # 'rsm': load_rsm,
    'med_rag': load_med_rag,
    'khan': load_khanacademy,
    'bigsurvey': load_bigsurvey,
}

def mean(lst):
    return sum(lst) / len(lst)

for name, func in dataset_funcs.items():
    data_pairs, output_base_dir = func()
    word_counts = []
    
    for pair in tqdm(data_pairs):
        word_count = len(nltk.word_tokenize(pair.input))
        word_counts.append(word_count)
    print(f'{name}: {max(word_counts)}, {min(word_counts)}, {mean(word_counts)}')


