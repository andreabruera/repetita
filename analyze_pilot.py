import fasttext
import matplotlib
import numpy
import os
import re
import scipy
import spacy

from matplotlib import pyplot
from scipy import spatial

from errors import errors

plots = 'distributions'
os.makedirs(plots, exist_ok=True)

errs = errors()
print(errs)

nlp = spacy.load('de_core_news_lg')
ft = fasttext.load_model('cc.en.300.bin')

results = list()
with open('sentences_responses_allsubs.tsv') as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = [val for val in line]
            print(header)
            continue
        assert len(line[1:]) == len(header)
        results.append(line[1:])
subjects = set([result[header.index('subject')] for result in results])
conditions = set([result[header.index('db_cond')] for result in results])
thresholds = set([result[header.index('threshold')] for result in results])

struc_res = {t : {c : dict() for c in conditions} for t in thresholds}
for r in results:
    real = r[header.index('target_sentence')].lower()
    real = re.sub('\W', ' ', real)
    real = re.sub('\s+', ' ', real)
    pred = re.sub('\W', ' ', r[header.index('response')]).lower()
    pred = re.sub('\s+', ' ', pred)
    sub = r[header.index('subject')]
    for er in sorted(errs, key=lambda item : len(item), reverse=True):
        pred = pred.replace(er, '')
        pred = pred.strip()
        if len(pred) == 0:
            pred = 'NA'
    if 'nicht verstanden' in pred or 'nichts verstanden' in pred:
        #if 'nicht' in pred or 'verstanden' in pred:
        pred = 'NA'
    try:
        struc_res[r[header.index('threshold')]][r[header.index('db_cond')]][sub].append((real, pred))
    except KeyError:
        struc_res[r[header.index('threshold')]][r[header.index('db_cond')]][sub] = [(real, pred)]

### overall sentence overlap (no lemma)
plot_results = {
                'semantic_similarity' : dict(),
                'words' : dict(),
                'lemmas' : dict(),
                'nouns' : dict(),
                'verbs' : dict(),
                'adjvs' : dict(),
                'function' : dict(),
                }

for t, t_data in struc_res.items():
    for c, c_data in t_data.items():
        print('\n')
        print(t)
        print(c)
        print('\n')
        for k in plot_results.keys():
            try:
                plot_results[k][float(c)][t] = list()
            except KeyError:
                plot_results[k][float(c)] = {t : list()}
        #sims = list()
        #res = list()
        #lemma_res = list()
        #noun_res = list()
        #verb_res = list()
        #ad_res = list()
        #func_res = list()
        for _, sub_data in c_data.items():
            for real, pred in sub_data:
                if pred == 'NA':
                    sim = 0.
                    #pass
                else:
                    real_ft = numpy.average([ft[w] for w in real.split()], axis=0)
                    pred_ft = numpy.average([ft[w] for w in pred.split()], axis=0)
                    sim = 1 - scipy.spatial.distance.cosine(real_ft, pred_ft)
                    #sims.append(sim)
                plot_results['semantic_similarity'][float(c)][t].append(sim)
                lemma_real = [w.lemma_ for w in nlp(real)]
                pos_real = [(w.text, w.pos_) for w in nlp(real)]
                lemma_pred = [w.lemma_ for w in nlp(pred)]
                acc = sum([1 for w in real.split() if w in pred.split()]) / len(real.split())
                lemma_acc = sum([1 for w in lemma_real if w in lemma_pred]) / len(lemma_real)
                #res.append(acc)
                #lemma_res.append(lemma_acc)
                plot_results['words'][float(c)][t].append(acc)
                plot_results['lemmas'][float(c)][t].append(lemma_acc)
                ### pos
                noun_l = sum([1 for w, pos in pos_real if pos=='NOUN'])
                if noun_l > 0:
                    noun_acc = sum([1 for w, pos in pos_real if w in pred and pos=='NOUN']) / noun_l
                    #noun_res.append(noun_acc)
                    plot_results['nouns'][float(c)][t].append(noun_acc)
                verb_l = sum([1 for w, pos in pos_real if pos=='VERB'])
                if verb_l > 0:
                    verb_acc = sum([1 for w, pos in pos_real if w in pred and pos=='VERB']) / verb_l
                    #verb_res.append(verb_acc)
                    plot_results['verbs'][float(c)][t].append(verb_acc)
                ad_l = sum([1 for w, pos in pos_real if pos in ['ADJ', 'ADV']])
                if ad_l > 0:
                    ad_acc = sum([1 for w, pos in pos_real if w in pred and pos in ['ADJ', 'ADV']]) / ad_l
                    #ad_res.append(ad_acc)
                    plot_results['adjvs'][float(c)][t].append(ad_acc)
                func_l = sum([1 for w, pos in pos_real if pos not in ['NOUN', 'VERB', 'ADJ', 'ADV']])
                if func_l > 0:
                    func_acc = sum([1 for w, pos in pos_real if w in pred and pos not in ['NOUN', 'VERB', 'ADJ', 'ADV']]) / func_l
                    #func_res.append(func_acc)
                    plot_results['function'][float(c)][t].append(func_acc)

conditions = sorted(plot_results['function'].keys())
thresholds = sorted(plot_results['nouns'][0].keys())
colors = ['orange', 'teal', 'hotpink', 'darkgrey', 'dodgerblue', 'gold']
colors = {t : colors[t_i] for t_i, t in enumerate(thresholds)}
if len(thresholds) % 2 == 0:
    corrections = [corr*0.2 for corr in range(-int(len(thresholds)/2), int(len(thresholds)/2))]
else:
    corrections = [corr*0.2 for corr in range(-int(len(thresholds)/2), int(len(thresholds)/2)+1)]
assert len(thresholds) == len(corrections)
for mode, mode_data in plot_results.items():
    marker = 0
    fig, ax = pyplot.subplots(
                              figsize=(20, 10),
                              constrained_layout=True,
                              )
    ax.set_ylim(bottom=-0.15, top=1.15)
    for c_i, c in enumerate(conditions):
        for t_corr, t in zip(corrections, thresholds):
            data = mode_data[c][t]
            vi = ax.violinplot(
                          data,
                          positions=[c_i+t_corr],
                          showmeans=False,
                          widths=0.35,
                          showextrema=False,
                          )
            ax.text(
                    s='avg=\n{}'.format(round(numpy.average(data), 3)),
                    x=c_i+t_corr,
                    y=-.1,
                    ha='center',
                    va='center')
            ax.scatter(
                       x = c_i+t_corr,
                       y = numpy.average(data),
                       marker = 'D',
                       c=colors[t],
                       edgecolors='white',
                       )
            ### removing right side
            for b in vi['bodies']:
                m = numpy.mean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], -numpy.inf, m)
                b.set_color(colors[t])
            ### legend dummies
            if marker < len(thresholds):
                ax.bar([0.], [0.], color=colors[t], label=t)
                marker += 1
    ax.legend(fontsize=23, loc=9, ncols=len(thresholds))
    pyplot.xticks(
                  ticks=range(len(conditions)),
                  labels=conditions,
                  fontsize=20,
                  )
    pyplot.xticks(fontsize=15)
    pyplot.title(
                 label=mode.replace('_', ' '),
                 fontsize=23,
                 )
    pyplot.savefig(
                   os.path.join(plots, '{}.jpg'.format(mode)),
                   dpi=300,
                   )
