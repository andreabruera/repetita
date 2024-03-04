import fasttext
import matplotlib
import multiprocessing
import numpy
import os
import pickle

from matplotlib import pyplot
from tqdm import tqdm

### folder
plot_f = os.path.join('plots', 'noun_candidates')
os.makedirs(plot_f, exist_ok=True)

### excluded words
global excluded
excluded = set()
with open(os.path.join('data', 'sentlist.txt')) as i:
    for l in i:
        for w in l.strip().split():
            excluded.add(w)

### loading word frequencies
global word_pos
with open(os.path.join('..', 'german_avocado', 'pickles', 'sdewac_word_pos_freqs.pkl'), 'rb') as i:
    word_pos = pickle.load(i)
with open(os.path.join('..', 'german_avocado', 'pickles', 'sdewac_word_freqs.pkl'), 'rb') as i:
    word_freqs = pickle.load(i)

all_inputs = [(w, f) for w, f in word_freqs.items()]
step = 100000
inputs = [all_inputs[start:start+step] for start in range(0, len(all_inputs), step) if start<len(all_inputs)]

ft_de = fasttext.load_model('../../dataset/word_vectors/de/cc.de.300.bin')

def find_candidates(ins):
    ### frequency threshold: 1000
    freq_threshold = 10
    min_len = 5
    max_len = 11
    nouns_candidates = list()
    #for w, freq in lemma_freqs.items():
    for w, freq in tqdm(ins):
        ### filtering
        if len(w) < min_len or len(w) > max_len:
            continue
        if freq < freq_threshold:
            continue
        if w in excluded:
            continue
        ### checking pos
        w_pos = sorted(word_pos[w].items(), key=lambda item : item[1], reverse=True)
        marker = False
        ### recognizing as nouns words with high relative dominance (>50%) of the noun usage
        if w_pos[0][0] == 'NN':
            marker = True
        else:
            if 'NN' in [p[0] for p in w_pos]:
                proportion = word_pos[w]['NN'] / sum([p[1] for p in w_pos])
                if proportion > 0.75:
                    marker = True
        if marker:
            if w in ft_de.words:
                nouns_candidates.append(w)
    return nouns_candidates

with multiprocessing.Pool() as i:
    lsts = i.map(find_candidates, inputs)
    i.terminate()
    i.join()

nouns_candidates = [w for v in lsts for w in v]

### plotting distributions
fig, ax = pyplot.subplots(constrained_layout=True)
ax.violinplot([word_freqs[w] for w in nouns_candidates], [0], showmeans=True, showmedians=True)
title = 'frequencies for the candidate nouns'
ax.set_title(title)
pyplot.savefig(os.path.join(plot_f, 'cand_noun_freq_distr.jpg'))
pyplot.clf()
pyplot.close()
import pdb; pdb.set_trace()
