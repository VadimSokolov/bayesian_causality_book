"""Many instruments and the horseshoe (Chapter 12, Section iv:sec:many), review loop F12 version.

Design. n = 2000, K = 20 independent standard normal instruments, tau = 0.1, structural and first stage errors standard normal with correlation 0.6.
Six cells: concentration parameter c in {1.8, 20, 135} by s in {3, 20} active instruments, first stage coefficient pi_j = sqrt(c / (n s)) on the s active
instruments and 0 on the other K - s (so c = n |pi|^2). The earlier weak design is the cell (1.8, 20) and the earlier sparse design the cell (135, 3).
1,000 data sets per cell. Methods on each data set: two stage least squares (with an intercept in the first stage); the exact grid posterior with the
diffuse normal first stage prior (flat); the structural model of iv_exact.py with a hierarchical normal first stage prior (common scale, inverse gamma
prior, as in iv_weak2.py) and with the horseshoe prior, both by the blocked Gibbs sampler, four chains of 4,000 iterations with the first 1,000 discarded; if the
largest split R-hat of tau over the four chains exceeds 1.01 the four chains are rerun with 16,000 iterations and 4,000 discarded (the count and the final R-hat are
recorded). Per cell the aggregator reports, by method: median of the posterior medians, bias and RMSE of the posterior median (2SLS: of the estimate),
coverage of the equal tailed 90% interval with its Monte Carlo standard error, the median interval length, and the RMSE of the first stage coefficients
separately over the active and over the inactive instruments (posterior means for hier and horseshoe, least squares for 2SLS and flat).

usage (repository root): python3 book_scripts/iv_horseshoe.py run --c 1.8 --s 20 --start 0 --end 50 --out DIR
                         python3 book_scripts/iv_horseshoe.py agg DIR   (writes results/res_iv_hs_r1023.json; results/res_iv_hs.json is left as it was)
"""
import argparse, glob, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import iv_exact
from iv_exact import exact_tau_cdf, cdf_quantiles, split_rhat_ess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAU, N, K = 0.1, 2000, 20
# the sampler of iv_exact.biv_block, with the draws of all K first stage coefficients returned after (tau, error correlation)
_src = open(iv_exact.__file__).read().split('def biv_block')[1].split('\ndef ')[0].replace(
    "out.append((tau, theta[0], s12 / np.sqrt(s11 * s22)))", "out.append((tau, s12 / np.sqrt(s11 * s22), *theta[:k]))")
_ns = dict(np=np); exec('import numpy as np\ndef biv_block' + _src, _ns); biv_pi = _ns['biv_block']
CELLS = [(c, s) for c in (1.8, 20.0, 135.0) for s in (3, 20)]


def gen(r, pivec, rho=0.6):
    Z = r.normal(size=(N, len(pivec))); U = r.multivariate_normal([0, 0], [[1, rho], [rho, 1]], N)
    return TAU * (Z @ pivec + U[:, 0]) + U[:, 1], Z @ pivec + U[:, 0], Z


def tsls_ols(Y, X, Z):
    Zc = np.column_stack([Z, np.ones(N)]); co = np.linalg.lstsq(Zc, X, rcond=None)[0]; xh = Zc @ co
    return np.cov(xh, Y)[0, 1] / np.cov(xh, X)[0, 1], co[:K]


def one_rep(ci, rep):
    c, s = CELLS[ci]; pivec = np.r_[np.full(s, math.sqrt(c / (N * s))), np.zeros(K - s)]
    Y, X, Z = gen(np.random.default_rng([20261023, ci, rep]), pivec); W = np.ones((N, 1)); rec = dict(cell=ci, c=c, s=s, rep=rep)
    t, pi_ols = tsls_ols(Y, X, Z); rec['tsls'] = dict(est=t, pi=pi_ols.tolist())
    tg, _, cdf = exact_tau_cdf(Y, X, Z, W); lo, me, hi = cdf_quantiles(tg, cdf, [0.05, 0.5, 0.95]); rec['flat'] = dict(med=float(me), lo=float(lo), hi=float(hi), pi=pi_ols.tolist())
    for m, kw in (('hier', dict(hier=True)), ('horseshoe', dict(horseshoe=True))):
        iters, burn, ext = 4000, 1000, 0
        while True:
            dr = np.array([biv_pi(Y, X, Z, W, iters=iters, burn=burn, seed=1000 * rep + ch, **kw) for ch in range(4)])
            rh = split_rhat_ess(dr[:, :, 0])[0]
            if rh <= 1.01 or ext == 1: break
            iters, burn, ext = 16000, 4000, 1
        tau_d = dr[:, :, 0].ravel(); lo, me, hi = np.quantile(tau_d, [0.05, 0.5, 0.95])
        rec[m] = dict(med=float(me), lo=float(lo), hi=float(hi), rhat=float(rh), extended=ext, pi=dr[:, :, 2:].reshape(-1, K).mean(0).tolist())
    return rec


def mc(v):
    v = np.asarray(v, float); return [float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))]


def aggregate(d):
    recs = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, '*.jsonl'))) for l in open(f)]; out = {}
    for ci, (c, s) in enumerate(CELLS):
        R = [r for r in recs if r['cell'] == ci]
        if not R: continue
        truth = np.r_[np.full(s, math.sqrt(c / (N * s))), np.zeros(K - s)]; cell = dict(R=len(R), c=c, s=s)
        for m in ('tsls', 'flat', 'hier', 'horseshoe'):
            est = np.array([r[m]['est'] if m == 'tsls' else r[m]['med'] for r in R]); er = est - TAU; sq = er ** 2; rm = math.sqrt(sq.mean())
            x = dict(med=float(np.median(est)), bias=mc(er), rmse=[rm, float(sq.std(ddof=1) / (2 * rm * math.sqrt(len(R))))], med_abs_err=float(np.median(np.abs(er))))
            if m != 'tsls':
                cov = np.array([r[m]['lo'] <= TAU <= r[m]['hi'] for r in R], float); x.update(cover=mc(cov), len_med=float(np.median([r[m]['hi'] - r[m]['lo'] for r in R])))
            P = np.array([r[m]['pi'] for r in R]) - truth; x['pi_rmse_active'] = float(math.sqrt(np.mean(P[:, :s] ** 2))); x['pi_rmse_inactive'] = float(math.sqrt(np.mean(P[:, s:] ** 2)))
            if m in ('hier', 'horseshoe'):
                x.update(rhat_max=float(max(r[m]['rhat'] for r in R)), share_extended=float(np.mean([r[m]['extended'] for r in R])), share_rhat_above_1_01_final=float(np.mean([r[m]['rhat'] > 1.01 for r in R])))
            cell[m] = x
        out['c%g_s%d' % (c, s)] = cell
    out['note'] = 'n=2000, K=20, tau=0.1, error correlation 0.6, 1000 data sets per cell; pi_j = sqrt(c/(n s)) on s active instruments'
    json.dump(out, open(os.path.join(ROOT, 'results', 'res_iv_hs_r1023.json'), 'w'), indent=1)
    for k, cl in out.items():
        if k == 'note': continue
        print(k, cl['R'])
        for m in ('tsls', 'flat', 'hier', 'horseshoe'):
            x = cl[m]; print('  %-9s med %.3f rmse %.3f cover %s len %s pi_rmse act %.4f inact %.4f' % (m, x['med'], x['rmse'][0], '%.3f(%.3f)' % tuple(x['cover']) if 'cover' in x else '-', '%.3f' % x['len_med'] if 'len_med' in x else '-', x['pi_rmse_active'], x['pi_rmse_inactive']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('dir', nargs='?'); ap.add_argument('--c', type=float); ap.add_argument('--s', type=int)
    ap.add_argument('--start', type=int); ap.add_argument('--end', type=int); ap.add_argument('--out'); a = ap.parse_args()
    if a.cmd == 'agg': aggregate(a.dir)
    else:
        os.makedirs(a.out, exist_ok=True); ci = CELLS.index((a.c, a.s))
        with open(os.path.join(a.out, 'cell%d_%04d_%04d.jsonl' % (ci, a.start, a.end)), 'w') as f:
            for r in range(a.start, a.end):
                f.write(json.dumps(one_rep(ci, r)) + '\n'); f.flush(); print(ci, r, flush=True)
