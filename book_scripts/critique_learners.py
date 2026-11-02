"""Chapter 7, review loop F17: what do forests, boosting and a stack do inside the cross fitted partially linear score?

Design. The source paper's partially linear law (critique_common.make_design), p = 400, tau = 1, X standard normal, sparse support s = 5, 15, 30, treatment
residual sd sigma_nu = 1 and 0.3 (weak overlap), n = 300 and 1,200. 1,000 replications per cell, every learner and every K on the same data set
(seeds crc32 of the cell label and the replication index, so n = 300 with sigma_nu = 1 and s = 5, 15, 30 and the weak overlap cell with s = 15 reuse the
source paper's data sets). The pooled cross fitted score with K = 5 folds (and K = 2 at n = 300), the same folds for every learner. Learners for both
nuisance regressions (D on X, Y on X), each tuned only inside its training fold:
  lasso    LassoCV (50 penalties, five fold);
  hs       horseshoe posterior mean (Gibbs, 500 sweeps, 150 burn in);
  rf       random forest, 100 trees, min leaf 3, max_features in {0.1, 0.33} of the 400 chosen by out of bag R^2;
  gb       histogram gradient boosting, depth 3, learning rate 0.05, at most 300 rounds, early stopping on a 15% validation split;
  stack    non negative least squares weights on the four fold out of fold predictions of lasso, rf (max_features 0.33) and gb in the training fold, applied
           to the tuned lasso, rf and gb fits of that fold (the horseshoe is not in the stack, for cost).
Recorded per data set, K and learner: tau_hat, the sandwich se, the two channels on the held out rows (own = mean e_m^2, cross = mean e_m e_l, with
e_m = m_hat - X alpha and e_l = l_hat - X varsigma the errors against the known truth), the alignment rho = cross / sqrt(own mean e_l^2), the bias functional
(cross - tau own)/J times sqrt(n) with J the mean squared treatment residual, and the fitting time. The aggregator reports bias, Monte Carlo sd, mean
se, coverage of tau_hat +/- 1.96 se and the rest, with Monte Carlo standard errors, and draws figs/critique_learners.pdf.

usage: PYTHONPATH=book_scripts python3 book_scripts/critique_learners.py run --n 300 --s 15 --snu 1.0 --start 0 --end 10 --out DIR
       python3 book_scripts/critique_learners.py agg DIR   (writes results/res_critique_learners.json and figs/critique_learners.pdf)
"""
import argparse, glob, json, math, os, sys, time, warnings, zlib
import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from critique_common import make_design, horseshoe_mean, THETA0
warnings.filterwarnings('ignore')
import sklearn
ALPHAS = dict(alphas=50) if tuple(int(v) for v in sklearn.__version__.split('.')[:2]) >= (1, 7) else dict(n_alphas=50)   # 50 penalties on either side of the n_alphas deprecation
LEARNERS = ['lasso', 'hs', 'rf', 'gb', 'stack']
Z95 = 1.959963984540054


def label(n, s, snu):
    if n == 300 and snu == 1.0: return 'sparse_s%d' % s
    if n == 300 and s == 15: return 'weak_overlap'
    return 'n%d_s%d_snu%g' % (n, s, snu)


def f_lasso(X, y, rs): return LassoCV(**ALPHAS, cv=5, max_iter=20000, random_state=rs).fit(X, y)
def f_rf(X, y, rs, mf=0.33): return RandomForestRegressor(100, max_features=mf, min_samples_leaf=3, n_jobs=1, random_state=rs, oob_score=True).fit(X, y)
def f_gb(X, y, rs): return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=rs).fit(X, y)


def fit_predict(Xtr, ytr, Xte, rng, T):
    """Predictions of all five learners on Xte from one training fold, and the seconds each took."""
    rs = int(rng.integers(2 ** 31)); ybar = ytr.mean(); out = {}; sec = {}
    t = time.time(); out['lasso'] = f_lasso(Xtr, ytr, rs).predict(Xte); sec['lasso'] = time.time() - t
    t = time.time(); xb = Xtr.mean(0); bh = horseshoe_mean(Xtr - xb, ytr - ybar, rng); out['hs'] = (Xte - xb) @ bh + ybar; sec['hs'] = time.time() - t
    t = time.time(); m = max((f_rf(Xtr, ytr, rs, mf) for mf in (0.1, 0.33)), key=lambda q: q.oob_score_); out['rf'] = m.predict(Xte); sec['rf'] = time.time() - t
    t = time.time(); out['gb'] = f_gb(Xtr, ytr, rs).predict(Xte); sec['gb'] = time.time() - t
    t = time.time(); oof = np.zeros((len(ytr), 3))
    for a, b in KFold(4, shuffle=True, random_state=rs % 10 ** 6).split(Xtr):
        oof[b, 0] = f_lasso(Xtr[a], ytr[a], rs).predict(Xtr[b]); oof[b, 1] = f_rf(Xtr[a], ytr[a], rs).predict(Xtr[b]); oof[b, 2] = f_gb(Xtr[a], ytr[a], rs).predict(Xtr[b])
    w = nnls(oof - ybar, ytr - ybar)[0]
    out['stack'] = ybar + np.column_stack([out['lasso'], out['rf'], out['gb']]).__sub__(ybar) @ w; sec['stack'] = time.time() - t + sec['lasso'] + sec['rf'] + sec['gb']
    return out, sec, w


def one_rep(n, s, snu, r, Ks):
    seed = zlib.crc32(('%s|%d' % (label(n, s, snu), r)).encode()) % (2 ** 31)
    data = make_design(np.random.default_rng(seed), n=n, s=s, sigma_nu=snu, dense=False); X, D, Y = data['X'], data['D'], data['Y']
    mtrue = X @ data['alpha']; ltrue = X @ data['varsigma']; rec = dict(n=n, s=s, snu=snu, rep=r, seed=seed, res={})
    for K in Ks:
        rng = np.random.default_rng(seed + K); fold = np.empty(n, int); fold[rng.permutation(n)] = np.arange(n) % K
        mh = {l: np.empty(n) for l in LEARNERS}; lh = {l: np.empty(n) for l in LEARNERS}; secs = {l: 0.0 for l in LEARNERS}; wts = []
        for k in range(K):
            te = fold == k; tr = ~te
            pm, sm, _ = fit_predict(X[tr], D[tr], X[te], rng, 'D'); pl, sl, w = fit_predict(X[tr], Y[tr], X[te], rng, 'Y'); wts.append(w)
            for l in LEARNERS: mh[l][te] = pm[l]; lh[l][te] = pl[l]; secs[l] += sm[l] + sl[l]
        rl = {}
        for l in LEARNERS:
            rD = D - mh[l]; rY = Y - lh[l]; J = np.mean(rD ** 2); th = np.mean(rD * rY) / J; psi = rD * (rY - th * rD) / J
            em = mh[l] - mtrue; el = lh[l] - ltrue; own = np.mean(em ** 2); cross = np.mean(em * el); nl2 = np.mean(el ** 2)
            rl[l] = dict(theta=float(th), se=float(math.sqrt(np.mean(psi ** 2) / n)), own=float(own), cross=float(cross), nl2=float(nl2),
                         rho=float(cross / math.sqrt(own * nl2)), rootn_b=float(math.sqrt(n) * (cross - THETA0 * own) / J), seconds=float(secs[l]))
        rl['stack_weights'] = np.mean(wts, 0).tolist(); rec['res'][str(K)] = rl
    return rec


def mc(v):
    v = np.asarray(v, float); return [float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))]


def plot(out):
    """Figure of the learner comparison. The n axis is categorical (300, 1,200), evenly spaced, so the tick labels cannot collide."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    cols = [(s, sn) for sn in (1.0, 0.3) for s in (5, 15, 30)]
    fig, ax = plt.subplots(2, 6, figsize=(12, 4.2), sharex=True)
    pos = {300: 0, 1200: 1}
    for j, (s, sn) in enumerate(cols):
        for k, l in enumerate(LEARNERS):
            ks = [(n, out.get('n%d_s%d_snu%g_K5_%s' % (n, s, sn, l))) for n in (300, 1200)]; ks = [(n, c) for n, c in ks if c]
            if not ks: continue
            xs = [pos[n] + 0.03 * (k - (len(LEARNERS) - 1) / 2) for n, c in ks]; ax[0, j].errorbar(xs, [c['cover'][0] for n, c in ks], [1.96 * c['cover'][1] for n, c in ks], marker='o', ms=3, label=l)
            ax[1, j].errorbar(xs, [c['bias'][0] for n, c in ks], [1.96 * c['bias'][1] for n, c in ks], marker='o', ms=3)
        ax[0, j].axhline(0.95, color='k', lw=0.5, ls=':'); ax[1, j].axhline(0, color='k', lw=0.5); ax[0, j].set_title('s = %d, $\\sigma_\\nu$ = %g' % (s, sn), fontsize=8)
        ax[0, j].set_ylim(0, 1); ax[1, j].set_xlim(-0.4, 1.4); ax[1, j].set_xticks([0, 1]); ax[1, j].set_xticklabels(['300', '1,200']); ax[1, j].set_xlabel('n')
    ax[0, 0].set_ylabel('coverage of 95% interval'); ax[1, 0].set_ylabel('signed bias'); ax[0, 0].legend(fontsize=6); fig.tight_layout(); fig.savefig('figs/critique_learners.pdf')


def aggregate(d):
    recs = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, '*.jsonl'))) for l in open(f)]; out = {}
    for key in sorted({(r['n'], r['s'], r['snu']) for r in recs}):
        R = [r for r in recs if (r['n'], r['s'], r['snu']) == key]
        for K in R[0]['res']:
            for l in LEARNERS:
                x = [r['res'][K][l] for r in R]; th = np.array([v['theta'] for v in x]); se = np.array([v['se'] for v in x]); cov = (np.abs(th - THETA0) <= Z95 * se).astype(float)
                c = dict(R=len(R), bias=mc(th - THETA0), mc_sd=float(th.std(ddof=1)), mean_se=mc(se), cover=mc(cov), ratio_se_sd=float(se.mean() / th.std(ddof=1)),
                         rmse=float(math.sqrt(np.mean((th - THETA0) ** 2))), rootn_b=mc([v['rootn_b'] for v in x]), rho=mc([v['rho'] for v in x]),
                         own=mc([v['own'] for v in x]), cross=mc([v['cross'] for v in x]), seconds=float(np.mean([v['seconds'] for v in x])))
                out['n%d_s%d_snu%g_K%s_%s' % (key[0], key[1], key[2], K, l)] = c
    json.dump(out, open('results/res_critique_learners.json', 'w'), indent=1)
    plot(out)
    for k, c in out.items():
        if '_K5_' in k: print(k, 'bias %+.3f cover %.3f se/sd %.2f rho %.2f rootn_b %+.2f' % (c['bias'][0], c['cover'][0], c['ratio_se_sd'], c['rho'][0], c['rootn_b'][0]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('dir', nargs='?'); ap.add_argument('--n', type=int); ap.add_argument('--s', type=int); ap.add_argument('--snu', type=float)
    ap.add_argument('--start', type=int); ap.add_argument('--end', type=int); ap.add_argument('--out'); a = ap.parse_args()
    if a.cmd == 'agg': aggregate(a.dir)
    elif a.cmd == 'plot': plot(json.load(open('results/res_critique_learners.json')))
    else:
        os.makedirs(a.out, exist_ok=True); Ks = [5, 2] if a.n == 300 else [5]
        with open(os.path.join(a.out, 'n%d_s%d_snu%g_%04d_%04d.jsonl' % (a.n, a.s, a.snu, a.start, a.end)), 'w') as f:
            for r in range(a.start, a.end):
                f.write(json.dumps(one_rep(a.n, a.s, a.snu, r, Ks)) + '\n'); f.flush(); print(a.n, a.s, a.snu, r, flush=True)
