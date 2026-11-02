"""Chapter 7, review loop F16: does a bootstrap that refits the nuisance lassos repair the cross fitted interval?

Design. The source paper's partially linear law (critique_common.make_design): n = 300, p = 400, tau = 1, X standard normal, five designs: sparse with
s = 5, 15, 30 (sigma_nu = 1), weak overlap (s = 15, sigma_nu = 0.3) and dense (every coordinate nonzero, the signal of s = 15 spread over all 400).
1,000 outer replications per design (seeds shared across the four procedures and with the source paper's first 500). On every data set, with cross
validated lasso nuisances (LassoCV, 50 penalties, five inner folds) and the pooled cross fitted score with K = 5:
  single   one cross fit; interval tau_hat +/- 1.96 se with the sandwich se;
  mult     the multiplier bootstrap of the score at that same fit (Mammen weights, 499 draws, percentile interval), no refitting;
  rsplit   20 repeated splits (the single fit is the first), median estimate, se = median of sqrt(se_s^2 + (tau_s - median)^2), Wald interval;
  pairs    pairs bootstrap, 499 resamples of the n observations with replacement; in each, both nuisance lassos are refitted inside a fresh five fold
           cross fit and tau is recomputed. The folds are assigned to the ORIGINAL observations, so all copies of one observation lie in one fold, and the
           inner cross validation of the lasso is grouped by original observation the same way. Intervals: percentile, and tau_hat +/- 1.96 sd of the draws.
Recorded per data set: estimates, sandwich se, intervals, and for the single fit the bias term Delta_bar_n of Proposition crit:cfvar, computed from the
known truth (e_m, e_l are coefficient differences because X ~ N(0, I); J = sigma_nu^2). The aggregator reports bias, RMSE, coverage, mean width, mean squared
reported error, Var(tau_hat), Var(Delta_bar), 2 Cov(Delta_bar, U), Var(U) with U = tau_hat - tau - Delta_bar, and Monte Carlo standard errors.

usage (repository root, PYTHONPATH=book_scripts): python3 book_scripts/critique_refit_bootstrap.py run --design sparse_s15 --start 0 --end 20 --out DIR
                                                  python3 book_scripts/critique_refit_bootstrap.py agg DIR  (writes results/res_critique_refit_bootstrap.json)
"""
import argparse, glob, json, math, os, sys, time, warnings
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from critique_common import DESIGNS, make_design, rep_seed, THETA0
warnings.filterwarnings('ignore')
import sklearn
ALPHAS = dict(alphas=50) if tuple(int(v) for v in sklearn.__version__.split('.')[:2]) >= (1, 7) else dict(n_alphas=50)   # 50 penalties on either side of the n_alphas deprecation
Z95 = 1.959963984540054


def lasso(X, y, rs, groups=None):
    cv = 5 if groups is None else list(GroupKFold(5).split(X, groups=groups))
    return LassoCV(**ALPHAS, cv=cv, max_iter=20000, random_state=rs).fit(X, y)


def crossfit(X, D, Y, fold, rng, groups=None, truth=None):
    """Pooled cross fitted partially linear estimate with folds given by `fold` (an integer per row)."""
    n = len(D); rD = np.empty(n); rY = np.empty(n); wb = wj = 0.0
    for k in np.unique(fold):
        te = fold == k; tr = ~te; g = None if groups is None else groups[tr]
        rs = int(rng.integers(2 ** 31)); mD = lasso(X[tr], D[tr], rs, g); mY = lasso(X[tr], Y[tr], rs, g)
        rD[te] = D[te] - mD.predict(X[te]); rY[te] = Y[te] - mY.predict(X[te])
        if truth is not None:
            em = mD.coef_ - truth['alpha']; el = mY.coef_ - truth['varsigma']; own = float(em @ em)
            w = te.sum() / n; wb += w * (float(em @ el) - THETA0 * own); wj += w * (truth['sigma_nu'] ** 2 + own)
    J = np.mean(rD ** 2); th = float(np.mean(rD * rY) / J); psi = rD * (rY - th * rD) / J
    out = dict(theta=th, se=float(math.sqrt(np.mean(psi ** 2) / n)))
    if truth is not None:
        out['delta'] = wb / wj
    return out, psi


def folds_perm(n, K, rng):
    f = np.empty(n, int); f[rng.permutation(n)] = np.arange(n) % K; return f


def one_rep(design, r, K=5, B=499, S=20):
    seed = rep_seed(design, r); data = make_design(np.random.default_rng(seed), **DESIGNS[design])
    X, D, Y = data['X'], data['D'], data['Y']; n = len(D); rec = dict(design=design, rep=r, seed=seed); t0 = time.time()
    rng = np.random.default_rng(seed + 1)
    s1, psi = crossfit(X, D, Y, folds_perm(n, K, rng), rng, truth=data); rec['single'] = dict(s1, lo=s1['theta'] - Z95 * s1['se'], hi=s1['theta'] + Z95 * s1['se'])
    # multiplier bootstrap at this fit (Mammen weights)
    a = (1 - math.sqrt(5)) / 2; b = (1 + math.sqrt(5)) / 2; pr = (math.sqrt(5) + 1) / (2 * math.sqrt(5))
    u = np.where(rng.random((B, n)) < pr, a, b); dr = s1['theta'] + u @ psi / n
    rec['mult'] = dict(theta=s1['theta'], se=float(dr.std(ddof=1)), lo=float(np.quantile(dr, .025)), hi=float(np.quantile(dr, .975)))
    # repeated split median
    th = [s1['theta']]; se = [s1['se']]; de = [s1['delta']]
    for _ in range(S - 1):
        o, _p = crossfit(X, D, Y, folds_perm(n, K, rng), rng, truth=data); th.append(o['theta']); se.append(o['se']); de.append(o['delta'])
    th, se = np.array(th), np.array(se); med = float(np.median(th)); sm = float(np.median(np.sqrt(se ** 2 + (th - med) ** 2)))
    rec['rsplit'] = dict(theta=med, se=sm, lo=med - Z95 * sm, hi=med + Z95 * sm, delta=float(np.mean(de)), sd_splits=float(th.std(ddof=1)))
    t1 = time.time(); bt = np.empty(B)
    for j in range(B):
        idx = rng.integers(0, n, n); fo = folds_perm(n, K, rng)[idx]
        bt[j] = crossfit(X[idx], D[idx], Y[idx], fo, rng, groups=idx)[0]['theta']
    sd = float(bt.std(ddof=1)); rec['pairs'] = dict(theta=s1['theta'], se=sd, lo=float(np.quantile(bt, .025)), hi=float(np.quantile(bt, .975)),
                                                      lo_n=s1['theta'] - Z95 * sd, hi_n=s1['theta'] + Z95 * sd, boot_mean=float(bt.mean()))
    rec['seconds'] = dict(nonboot=t1 - t0, boot=time.time() - t1)
    return rec


def mc(v):
    v = np.asarray(v, float); return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


def aggregate(d):
    recs = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, '*.jsonl'))) for l in open(f)]
    out = {}
    for design in DESIGNS:
        R = [r for r in recs if r['design'] == design]
        if not R: continue
        cell = {'R': len(R)}
        for m in ('single', 'mult', 'rsplit', 'pairs'):
            th = np.array([r[m]['theta'] for r in R]) - THETA0; lo = np.array([r[m]['lo'] for r in R]); hi = np.array([r[m]['hi'] for r in R])
            cov = ((lo <= THETA0) & (THETA0 <= hi)).astype(float); se2 = np.array([r[m]['se'] for r in R]) ** 2
            sq = th ** 2; rm = math.sqrt(sq.mean())
            c = dict(bias=mc(th), rmse=(rm, float(sq.std(ddof=1) / (2 * rm * math.sqrt(len(R))))), cover=mc(cov), width=mc(hi - lo),
                     mean_se2=mc(se2), var_theta=float(th.var(ddof=1)), sd_theta=float(th.std(ddof=1)), ratio_se_sd=float(math.sqrt(se2.mean()) / th.std(ddof=1)))
            if m == 'pairs':
                c['cover_normal'] = mc(((np.array([r[m]['lo_n'] for r in R]) <= THETA0) & (THETA0 <= np.array([r[m]['hi_n'] for r in R]))).astype(float))
            if m in ('single', 'rsplit'):
                dl = np.array([r[m]['delta'] for r in R]); U = th - dl
                nb = 20; blk = np.array_split(np.arange(len(R)), nb); rat = []
                for j in range(nb):
                    k = np.concatenate([blk[i] for i in range(nb) if i != j]); rat.append(2 * np.cov(dl[k], U[k])[0, 1])
                c.update(var_delta=float(dl.var(ddof=1)), cov2=float(2 * np.cov(dl, U)[0, 1]), cov2_se=float(np.std(rat, ddof=1) * math.sqrt(nb - 1)), var_U=float(U.var(ddof=1)))
            cell[m] = c
        cell['seconds_boot_mean'] = float(np.mean([r['seconds']['boot'] for r in R])); cell['seconds_nonboot_mean'] = float(np.mean([r['seconds']['nonboot'] for r in R]))
        out[design] = cell
    json.dump(out, open('results/res_critique_refit_bootstrap.json', 'w'), indent=1)
    for dsg, c in out.items():
        print(dsg, c['R'])
        for m in ('single', 'mult', 'rsplit', 'pairs'):
            x = c[m]; print('  %-7s bias %+.3f(%.3f) rmse %.3f cover %.3f(%.3f) width %.3f se/sd %.2f' % (m, x['bias'][0], x['bias'][1], x['rmse'][0], x['cover'][0], x['cover'][1], x['width'][0], x['ratio_se_sd']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('dir', nargs='?'); ap.add_argument('--design'); ap.add_argument('--start', type=int); ap.add_argument('--end', type=int)
    ap.add_argument('--out'); ap.add_argument('--B', type=int, default=499); ap.add_argument('--S', type=int, default=20); a = ap.parse_args()
    if a.cmd == 'agg': aggregate(a.dir)
    else:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, '%s_%04d_%04d.jsonl' % (a.design, a.start, a.end)), 'w') as f:
            for r in range(a.start, a.end):
                f.write(json.dumps(one_rep(a.design, r, B=a.B, S=a.S)) + '\n'); f.flush(); print(a.design, r, flush=True)
