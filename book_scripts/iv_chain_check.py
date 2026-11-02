"""Why the structural posterior of Chapter 12 is computed exactly: the one at a time Gibbs sampler of earlier versions of iv_card.py did not converge.

Weak instrument design of iv_card.py with one instrument (n = 2000, first stage coefficient 0.03, tau = 0.1, error correlation 0.6, seed 7),
first 20 data sets: the posterior of tau under the structural prior is computed (a) exactly (iv_exact.exact_tau_cdf), (b) by the blocked Gibbs
sampler (4 chains of 4,000 iterations, the first 1,000 discarded), (c) by the one at a time sampler as it was run for the book (1,200 iterations,
the first 300 discarded, started at the least squares fit) and (d) by the same sampler with 20,000 iterations (2,000 discarded). For each method
the script records the median over data sets of the posterior median and of the length of the 90% interval, the share of intervals covering the
truth, and for (b) to (d) the median absolute difference between the posterior median and the exact one.
Card (1995): the exact posterior, the blocked sampler (4 chains of 11,000 iterations, 1,000 discarded) and the one at a time sampler (4 chains of
6,000 iterations, 1,000 discarded, started at tau = -0.5, 0, 0.5 and 1 and at the least squares fit of the other coefficients) are compared through
their quantiles, split R-hat and effective sample size of tau, the lag one autocorrelation of the draws, and the standard deviation of the one step
change of tau from one iteration to the next in the one at a time sampler, against the exact posterior standard deviation.

usage (from the repository root): python book_scripts/iv_chain_check.py
writes results/res_iv_chains.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from iv_exact import exact_tau_cdf, cdf_quantiles, biv_book, biv_block, split_rhat_ess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
probs = [0.05, 0.5, 0.95]


def weak_sets(R, n=2000, pi=0.03, tau=0.1, rho=0.6, seed=7):
    r = np.random.default_rng(seed)
    for rep in range(R):
        Z = r.normal(size=(n, 1)); U = r.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
        X = Z[:, 0] * pi + U[:, 0]; Y = tau * X + U[:, 1]
        yield rep, Y, X, Z


def summarize(rows, exact_med=None):
    o = np.array(rows, float)
    out = dict(data_sets=int(len(o)), median_of_medians=float(np.median(o[:, 0])), length_median=float(np.median(o[:, 2])), coverage=float(o[:, 1].mean()))
    if exact_med is not None:
        out['median_abs_diff_from_exact_median'] = float(np.median(np.abs(o[:, 0] - exact_med)))
    return out


def main():
    R, tau = 20, 0.1
    rows = {k: [] for k in ('exact', 'block', 'old_1200', 'old_20000')}
    for rep, Y, X, Z in weak_sets(R):
        W = np.ones((len(Y), 1))
        tg, _, cdf = exact_tau_cdf(Y, X, Z, W)
        lo, me, hi = cdf_quantiles(tg, cdf, probs); rows['exact'].append((me, lo <= tau <= hi, hi - lo))
        d = np.concatenate([biv_block(Y, X, Z, W, iters=4000, burn=1000, seed=1000 * rep + c)[:, 0] for c in range(4)])
        lo, me, hi = np.quantile(d, probs); rows['block'].append((me, lo <= tau <= hi, hi - lo))
        d = biv_book(Y, X, Z, W, iters=1200, burn=300, seed=rep)[:, 0]
        lo, me, hi = np.quantile(d, probs); rows['old_1200'].append((me, lo <= tau <= hi, hi - lo))
        d = biv_book(Y, X, Z, W, iters=20000, burn=2000, seed=rep)[:, 0]
        lo, me, hi = np.quantile(d, probs); rows['old_20000'].append((me, lo <= tau <= hi, hi - lo))
        print(rep, flush=True)
    exact_med = np.array(rows['exact'], float)[:, 0]
    weak = {k: summarize(v, None if k == 'exact' else exact_med) for k, v in rows.items()}

    D = pd.read_csv(os.path.join(ROOT, 'data', 'card.csv')).dropna(subset=['lwage', 'educ', 'nearc4'])
    ctrl = ['exper', 'expersq', 'black', 'smsa', 'south', 'smsa66'] + [f'reg66{k}' for k in range(2, 10)]
    y = D['lwage'].values; x = D['educ'].values.astype(float); z = D['nearc4'].values.astype(float)
    W = np.column_stack([np.ones(len(D))] + [D[c].values.astype(float) for c in ctrl])
    Zc = z[:, None]
    tg, dens, cdf = exact_tau_cdf(y, x, Zc, W, scale=0.5, center=0.13)
    trap = getattr(np, 'trapezoid', None) or np.trapz
    m1 = trap(tg * dens, tg)
    card = dict(exact=dict(zip(('q05', 'q50', 'q95'), map(float, cdf_quantiles(tg, cdf, probs))), mean=float(m1), sd=float(np.sqrt(trap((tg - m1) ** 2 * dens, tg)))))
    blk = np.array([biv_block(y, x, Zc, W, iters=11000, burn=1000, seed=s)[:, 0] for s in range(1, 5)])
    rh, es = split_rhat_ess(blk)
    card['block'] = dict(zip(('q05', 'q50', 'q95'), map(float, np.quantile(blk.ravel(), probs))), rhat=rh, ess=es)
    old = np.array([biv_book(y, x, Zc, W, iters=6000, burn=1000, seed=s, tau0=t0)[:, 0] for s, t0 in zip(range(1, 5), (-0.5, 0.0, 0.5, 1.0))])
    rh, es = split_rhat_ess(old)
    lag1 = float(np.mean([np.corrcoef(c[:-1], c[1:])[0, 1] for c in old]))
    card['old'] = dict(zip(('q05', 'q50', 'q95'), map(float, np.quantile(old.ravel(), probs))), rhat=rh, ess=es, lag1_autocorrelation=lag1,
                       chain_medians=[float(np.median(c)) for c in old], sd_of_one_step_change=float(np.mean([np.std(np.diff(c)) for c in old])))
    out = dict(weak_one_first_20_data_sets=weak, card=card)
    with open(os.path.join(ROOT, 'results', 'res_iv_chains.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
