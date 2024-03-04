import fasttext
import numpy
import os
import pickle
import scipy
import sklearn

from sklearn import cluster
from tqdm import tqdm

ft_de = fasttext.load_model('../../dataset/word_vectors/de/cc.de.300.bin')
candidates = list()
vecs = list()
## read candidates
with open('candidates.txt') as i:
    for l in i:
        w = l.strip()
        candidates.append(w)
        vecs.append(ft_de[w])

clstr = sklearn.cluster.SpectralClustering(n_clusters=60, random_state=3).fit(vecs)
labels = set(clstr.labels_)

with open('clustered_candidates.txt', 'w') as o:
    for l in sorted(labels):
        for w, label in zip(candidates, clstr.labels_):
            if l != label:
                continue
            o.write('{}\t{}\n'.format(w, label))
