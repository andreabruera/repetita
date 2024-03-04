import fasttext
import numpy
import os
import pickle
import scipy

from tqdm import tqdm

ft_de = fasttext.load_model('../../dataset/word_vectors/de/cc.de.300.bin')
candidates = list()
## read candidates
with open('candidates.txt') as i:
    for l in i:
        w = l.strip()
        candidates.append(w)

sims = numpy.zeros((len(candidates), len(candidates)))
for w_i, w in tqdm(enumerate(candidates)):
    for w_two_i, w_two in enumerate(candidates):
        if w_two_i < w_i:
            continue
        sim = 1 - scipy.spatial.distance.cosine(ft_de[w], ft_de[w_two])
        sims[w_i, w_two_i] = sim
        sims[w_two_i, w_i] = sim
avg = numpy.average(sims)
std = numpy.std(sims)
z_sims = (sims * avg) / std
assert z_sims.shape == sims.shape
### sorting
can
sorted_lst = sorted([(w, numpy.sum(numpy.absolute(z_sims[w_i]))) for w_i, w in enumerate(candidates)], key=lambda item: item[1])
with open('shortlist_candidates.txt', 'w') as o:
    for w, sum_w in sorted_lst:
        o.write('{}\t{}\n'.format(w, sum_w))
