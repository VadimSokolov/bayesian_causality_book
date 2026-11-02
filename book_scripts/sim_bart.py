"""Chapter 11, review loop F10: a Hill style simulation of BART and BCF against least squares and cross fitted AIPW, with targeted selection.

Design. n = 500 and 2,000, p = 10 standard normal covariates, prognostic surface mu(x) = 2 sin(pi x1) + 1.5 1(x2 > 0.5) + x3^2, effect tau(x) = 1 + 0.5 x4,
Y = mu + tau T + N(0, 1). Propensity: design A (the earlier design) e(x) = expit{1.2 x1 + 0.8 1(x2 > 0.5)}; designs T1 and T3 (targeted selection)
e(x) = 0.01 + 0.98 expit[g {mu(x) - E mu(X)} / sd mu(X)] with g = 1 and 3 (E mu(X) and sd mu(X) are population values from 2 million draws). The target is the
sample average of tau(x_i) of each data set. 1,000 replications per cell (3 designs x 2 sample sizes). Methods on identical data: least squares of Y on
T and X with an HC3 interval; Hill's BART (T as a covariate, effect = mean of f(x, 1) - f(x, 0), 95% posterior interval); BART with the estimated
propensity as an extra covariate; BCF with the cross validated logistic propensity; AIPW with five fold cross fitting, random forests (200 trees, min leaf 5),
propensities clipped to [0.01, 0.99] and a Wald interval. stochtree settings of the text: 10 grow from root sweeps and 300 retained draws. On the first 50
replications of each cell a check also runs four chains of 2,000 retained draws (seeds 1 to 4) for the three tree methods and records the pooled interval and the
spread of the endpoints across chains. Saved per replication: estimate, interval, CATE RMSE for the tree methods, timing.

usage: PYTHONPATH=. python3 book_scripts/sim_bart.py run --design A --n 500 --start 0 --end 25 --out DIR
       python3 book_scripts/sim_bart.py agg DIR  (writes results/res_simbart.json and figs/simbart.pdf)
"""
import argparse, glob, json, math, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
expit = lambda z: 1 / (1 + np.exp(-z))
mu_f = lambda X: 2 * np.sin(np.pi * X[:, 0]) + 1.5 * (X[:, 1] > 0.5) + X[:, 2] ** 2
_ref = np.random.default_rng(99).normal(size=(2_000_000, 10)); _m = mu_f(_ref); MU_BAR, MU_SD = float(_m.mean()), float(_m.std()); del _ref, _m
DESIGN_ID = {'A': 0, 'T1': 1, 'T3': 3}
METHODS = ['ls', 'bart', 'bartps', 'bcf', 'aipw']


def propensity(X, d):
    if d == 'A': return expit(1.2 * X[:, 0] + 0.8 * (X[:, 1] > 0.5))
    g = DESIGN_ID[d]; return 0.01 + 0.98 * expit(g * (mu_f(X) - MU_BAR) / MU_SD)


def hc3(T, X, Y):
    Z = np.column_stack([T, np.ones(len(Y)), X]); A = np.linalg.inv(Z.T @ Z); b = A @ Z.T @ Y; u = Y - Z @ b; h = np.einsum('ij,jk,ik->i', Z, A, Z)
    V = A @ (Z.T * (u / (1 - h)) ** 2) @ Z @ A; return b[0], math.sqrt(V[0, 0])


def bart_fit(BARTModel, BCFModel, X, T, Y, ps, seed, nm, kind):
    """Returns the posterior draws of the average effect and the posterior mean of the individual effects."""
    n, p = X.shape
    if kind == 'bcf':
        b = BCFModel(); b.sample(X_train=X, Z_train=T, y_train=Y, propensity_train=ps, num_gfr=10, num_mcmc=nm, general_params={"random_seed": seed})
        tb = b.tau_hat_train; return tb.mean(0), tb.mean(1)
    Xa = np.column_stack([X, T]) if kind == 'bart' else np.column_stack([X, ps, T]); q = Xa.shape[1] - 1
    m = BARTModel(); m.sample(X_train=Xa, y_train=Y, num_gfr=10, num_mcmc=nm, general_params={"random_seed": seed})
    X1 = Xa.copy(); X1[:, q] = 1; X0 = Xa.copy(); X0[:, q] = 0; f1 = m.predict(X1); f0 = m.predict(X0)
    f1 = f1["y_hat"] if isinstance(f1, dict) else f1; f0 = f0["y_hat"] if isinstance(f0, dict) else f0
    return (f1 - f0).mean(0), (f1 - f0).mean(1)


def one_rep(d, n, r, check, main4=False):
    from stochtree import BARTModel, BCFModel
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import KFold
    rng = np.random.default_rng([20261023, DESIGN_ID[d], n, r]); p = 10
    X = rng.normal(size=(n, p)); e = propensity(X, d); T = rng.binomial(1, e).astype(float); tau = 1 + 0.5 * X[:, 3]; Y = mu_f(X) + tau * T + rng.normal(size=n); tbar = float(tau.mean())
    rec = dict(design=d, n=n, rep=r, truth=tbar, e_min=float(e.min()), e_max=float(e.max()), treated=float(T.mean())); tm = {}
    t = time.time(); est, se = hc3(T, X, Y); rec['ls'] = dict(est=est, lo=est - 1.959964 * se, hi=est + 1.959964 * se); tm['ls'] = time.time() - t
    ps = LogisticRegressionCV(cv=5, max_iter=2000).fit(X, T).predict_proba(X)[:, 1]
    for kind, name in (('bart', 'bart'), ('bartps', 'bartps'), ('bcf', 'bcf')):
        t = time.time()
        if main4:  # review loop B2: the main interval pools four chains of 2,000 retained draws (seeds 1000 (c + 1) + r), no 300 draw run and no separate check
            parts = [bart_fit(BARTModel, BCFModel, X, T, Y, ps, 1000 * (c + 1) + r, 2000, kind) for c in range(4)]
            dr = np.concatenate([q[0] for q in parts]); ci = np.mean([q[1] for q in parts], axis=0)
            e4 = np.array([(np.quantile(q[0], .025), np.quantile(q[0], .975)) for q in parts])
        else: dr, ci = bart_fit(BARTModel, BCFModel, X, T, Y, ps, r, 300, kind)
        tm[name] = time.time() - t
        rec[name] = dict(est=float(dr.mean()), lo=float(np.quantile(dr, .025)), hi=float(np.quantile(dr, .975)), cate_rmse=float(np.sqrt(np.mean((ci - tau) ** 2))))
        if main4: rec[name].update(lo_sd=float(e4[:, 0].std(ddof=1)), hi_sd=float(e4[:, 1].std(ddof=1)))
        if check:
            ends = []; pooled = []
            for c in range(4):
                dc, _ = bart_fit(BARTModel, BCFModel, X, T, Y, ps, 1000 * (c + 1) + r, 2000, kind); ends.append((np.quantile(dc, .025), np.quantile(dc, .975))); pooled.append(dc)
            pooled = np.concatenate(pooled); ends = np.array(ends)
            rec[name]['chk'] = dict(lo=float(np.quantile(pooled, .025)), hi=float(np.quantile(pooled, .975)), est=float(pooled.mean()), lo_sd=float(ends[:, 0].std(ddof=1)), hi_sd=float(ends[:, 1].std(ddof=1)))
    t = time.time(); kf = KFold(5, shuffle=True, random_state=r); m1 = np.zeros(n); m0 = np.zeros(n); eh = np.zeros(n)
    for tr, te in kf.split(X):
        for tt, arr in ((1, m1), (0, m0)):
            s = tr[T[tr] == tt]; arr[te] = RandomForestRegressor(200, min_samples_leaf=5, random_state=r).fit(X[s], Y[s]).predict(X[te])
        eh[te] = RandomForestClassifier(200, min_samples_leaf=5, random_state=r).fit(X[tr], T[tr]).predict_proba(X[te])[:, 1]
    ec = np.clip(eh, 0.01, 0.99); psi = m1 - m0 + T * (Y - m1) / ec - (1 - T) * (Y - m0) / (1 - ec); s_ = psi.std() / math.sqrt(n)
    rec['aipw'] = dict(est=float(psi.mean()), lo=float(psi.mean() - 1.959964 * s_), hi=float(psi.mean() + 1.959964 * s_)); tm['aipw'] = time.time() - t
    rec['seconds'] = tm; return rec


def mc(v):
    v = np.asarray(v, float); return [float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))]


def aggregate(dr):
    recs = [json.loads(l) for f in sorted(glob.glob(os.path.join(dr, '*.jsonl'))) for l in open(f)]; out = {}
    for d in DESIGN_ID:
        for n in (500, 2000):
            R = [r for r in recs if r['design'] == d and r['n'] == n]
            if not R: continue
            cell = dict(R=len(R), e_min_mean=float(np.mean([r['e_min'] for r in R])), e_max_mean=float(np.mean([r['e_max'] for r in R])), treated_share=float(np.mean([r['treated'] for r in R])))
            for m in METHODS:
                er = np.array([r[m]['est'] - r['truth'] for r in R]); sq = er ** 2; rm = math.sqrt(sq.mean())
                c = dict(bias=mc(er), rmse=[rm, float(sq.std(ddof=1) / (2 * rm * math.sqrt(len(R))))], seconds=float(np.mean([r['seconds'][m] for r in R])))
                if m != 'ls' or True:
                    cov = np.array([r[m]['lo'] <= r['truth'] <= r[m]['hi'] for r in R], float); c.update(cover=mc(cov), width=mc([r[m]['hi'] - r[m]['lo'] for r in R]))
                if 'lo_sd' in R[0][m]: c['endpoint_sd_across_chains'] = float(np.mean([(r[m]['lo_sd'] + r[m]['hi_sd']) / 2 for r in R]))
                if 'cate_rmse' in R[0][m]: c['cate_rmse'] = mc([r[m]['cate_rmse'] for r in R])
                cell[m] = c
            for m in ('bart', 'bartps', 'bcf'):
                C = [r for r in R if 'chk' in r[m]]
                if C:
                    cell[m]['check'] = dict(R=len(C), cover_long=mc([r[m]['chk']['lo'] <= r['truth'] <= r[m]['chk']['hi'] for r in C]), cover_short=mc([r[m]['lo'] <= r['truth'] <= r[m]['hi'] for r in C]),
                                            width_long=mc([r[m]['chk']['hi'] - r[m]['chk']['lo'] for r in C]), width_short=mc([r[m]['hi'] - r[m]['lo'] for r in C]),
                                            endpoint_sd_across_chains=float(np.mean([(r[m]['chk']['lo_sd'] + r[m]['chk']['hi_sd']) / 2 for r in C])),
                                            max_abs_endpoint_diff_long_vs_short=float(max(max(abs(r[m]['chk']['lo'] - r[m]['lo']), abs(r[m]['chk']['hi'] - r[m]['hi'])) for r in C)))
            out['%s_n%d' % (d, n)] = cell
    json.dump(out, open('results/res_simbart.json', 'w'), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 3, figsize=(8, 4.4), sharey='row')
    for i, n in enumerate((500, 2000)):
        for j, d in enumerate(DESIGN_ID):
            R = [r for r in recs if r['design'] == d and r['n'] == n]
            if R: ax[i, j].boxplot([[r[m]['est'] - r['truth'] for r in R] for m in METHODS], tick_labels=["LS", "BART", "BART+e", "BCF", "AIPW"], showfliers=False)
            ax[i, j].axhline(0, color="k", lw=0.5); ax[i, j].set_title('design %s, n = %d' % (d, n), fontsize=8)
    ax[0, 0].set_ylabel("Estimate minus sample ATE"); ax[1, 0].set_ylabel("Estimate minus sample ATE"); fig.tight_layout(); fig.savefig("figs/simbart.pdf")
    for k, c in out.items():
        print(k, c['R'], 'e range %.3f-%.3f' % (c['e_min_mean'], c['e_max_mean']))
        for m in METHODS: print('   %-7s bias %+.3f rmse %.3f cover %.3f(%.3f) width %.2f' % (m, c[m]['bias'][0], c[m]['rmse'][0], c[m]['cover'][0], c[m]['cover'][1], c[m]['width'][0]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('dir', nargs='?'); ap.add_argument('--design'); ap.add_argument('--n', type=int); ap.add_argument('--start', type=int)
    ap.add_argument('--end', type=int); ap.add_argument('--out'); ap.add_argument('--main4', action='store_true'); a = ap.parse_args()
    if a.cmd == 'agg': aggregate(a.dir)
    else:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, '%s_n%d_%04d_%04d.jsonl' % (a.design, a.n, a.start, a.end)), 'w') as f:
            for r in range(a.start, a.end):
                f.write(json.dumps(one_rep(a.design, a.n, r, check=(r < 50 and not a.main4), main4=a.main4)) + '\n'); f.flush(); print(a.design, a.n, r, flush=True)
