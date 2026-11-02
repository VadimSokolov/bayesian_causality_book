"""Benchmark covariates for the sensitivity analysis of Chapter 13 (Section sens:benchmark): how strongly is each of the nine
401(k) covariates related to eligibility and to net financial assets?

Data: 1991 SIPP, n = 9,915 (data/sipp1991.dta). Treatment: e401 (401(k) eligibility). Outcome: net_tfa in thousands of dollars.
Linear: the partial R^2 of covariate j is 1 - RSS_full / RSS_reduced, where the full regression has the intercept and all nine covariates
(and eligibility, for the outcome) and the reduced regression drops covariate j.
Block: the same with the 54 term quadratic basis of Chapter 16 (nine linear terms, nine squares, 36 products of the standardized covariates),
dropping the ten terms that involve covariate j.

usage (from the repository root): python book_scripts/benchmark_401k.py
writes results/res_benchmark_401k.json
"""
import json
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement

d = pd.read_stata('data/sipp1991.dta')
xv = ['age', 'inc', 'educ', 'fsize', 'marr', 'twoearn', 'db', 'pira', 'hown']
Y = d.net_tfa.values.astype(float) / 1000.0
D = d.e401.values.astype(float)
X = d[xv].values.astype(float)
X = (X - X.mean(0)) / X.std(0)
n = len(Y)
one = np.ones((n, 1))


def rss(M, y):
    b, *_ = np.linalg.lstsq(M, y, rcond=None)
    r = y - M @ b
    return float(r @ r)


terms = [(i,) for i in range(9)] + list(combinations_with_replacement(range(9), 2))
B = np.column_stack([np.prod(X[:, list(t)], axis=1) for t in terms])
B = (B - B.mean(0)) / B.std(0)

out = {'n': int(n), 'covariates': xv, 'linear': {}, 'block': {}}
for j, name in enumerate(xv):
    keep = [i for i in range(len(xv)) if i != j]
    out['linear'][name] = dict(
        eligibility=1 - rss(np.column_stack([one, X]), D) / rss(np.column_stack([one, X[:, keep]]), D),
        assets=1 - rss(np.column_stack([one, D, X]), Y) / rss(np.column_stack([one, D, X[:, keep]]), Y))
    inblock = np.array([j in t for t in terms])
    out['block'][name] = dict(
        eligibility=1 - rss(np.column_stack([one, B]), D) / rss(np.column_stack([one, B[:, ~inblock]]), D),
        assets=1 - rss(np.column_stack([one, D, B]), Y) / rss(np.column_stack([one, D, B[:, ~inblock]]), Y),
        terms=int(inblock.sum()))
with open('results/res_benchmark_401k.json', 'w') as f:
    json.dump(out, f, indent=1)
for kind in ('linear', 'block'):
    print(kind)
    for name in xv:
        print('  %-8s eligibility %.4f  assets %.4f' % (name, out[kind][name]['eligibility'], out[kind][name]['assets']))
