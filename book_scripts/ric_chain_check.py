"""Convergence check for the horseshoe Gibbs samplers behind Section sec:ricsim (Chapter 8), the 401(k) analysis (Chapter 16) and Section comp:sec:cost (Chapter 9).

Simulation designs: the data generating process of ric_sim.py (n = 200, p independent standard normal covariates, five confounders with
gamma_j = beta_j = 1, standard normal nu and eps, tau = 1) at p = 50 (20 data sets) and at p = 150 and p = 300 (10 data sets each). For each
data set the sampler of ric_sim.py (1,500 iterations, the first 500 discarded) is run as four independent chains, on the regression of Y on
(T, X) with T unpenalized (the independent horseshoe) and on the regression of Y on (T, e_hat, X) with T and the propensity covariate e_hat
unpenalized (the coupled horseshoe, e_hat the whole sample LassoCV fit of T on X, variant 4 of ric_sim.py).
401(k) design: the sampler of app401k_book.py (3,000 iterations, the first 1,000 discarded), four chains on the real data, independent
(treatment and intercept unpenalized) and coupled (a cross validated logistic regression propensity as a third unpenalized column).
For the coefficient of the treatment the script records the split R-hat (each chain cut in half, classic between and within variance form)
and the effective sample size (Geyer's initial positive sequence on the pooled split chains), and reports the share of data sets with
R-hat below 1.01.

usage (from the repository root): python book_scripts/ric_chain_check.py
writes results/res_ric_chains.json
  python book_scripts/ric_chain_check.py --weak30   the weak30 design only (10 data sets), writes results/res_ric_chains_weak30.json
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Horseshoe Gibbs sampler of Chapter 8, copied from ric_sim.py without change.
def hs_gibbs(y, X, unpen, iters=1500, burn=500, rng=None):
    n, p = X.shape
    pen = np.array([j not in unpen for j in range(p)])
    lam2 = np.ones(p); nu = np.ones(p); tau2 = 1.0; xi = 1.0; sig2 = 1.0
    XtX = X.T @ X; Xty = X.T @ y
    draws = []
    for it in range(iters):
        prec = np.where(pen, 1 / (lam2 * tau2), 1e-6)
        A = XtX + np.diag(prec)
        L = np.linalg.cholesky(A)
        mu = np.linalg.solve(A, Xty)
        b = mu + np.sqrt(sig2) * np.linalg.solve(L.T, rng.normal(size=p))
        res = y - X @ b
        bp = b[pen]
        sig2 = 1 / rng.gamma((n + pen.sum()) / 2, 1 / (res @ res / 2 + (bp ** 2 / (lam2[pen] * tau2)).sum() / 2))
        lam2[pen] = 1 / rng.gamma(1, 1 / (1 / nu[pen] + bp ** 2 / (2 * tau2 * sig2)))
        nu[pen] = 1 / rng.gamma(1, 1 / (1 + 1 / lam2[pen]))
        tau2 = 1 / rng.gamma((pen.sum() + 1) / 2, 1 / (1 / xi + (bp ** 2 / lam2[pen]).sum() / (2 * sig2)))
        xi = 1 / rng.gamma(1, 1 / (1 + 1 / tau2))
        if it >= burn:
            draws.append(b.copy())
    return np.array(draws)


# Horseshoe Gibbs sampler of the 401(k) analysis, copied from app401k_book.py without change (rng passed in).
def hs401(Z, y, unpen, rng, iters=3000, burn=1000):
    nn, q = Z.shape
    pen = np.ones(q, bool); pen[list(unpen)] = False
    ZtZ = Z.T @ Z; Zty = Z.T @ y; yty = y @ y
    lam2 = np.ones(q); nu = np.ones(q); tau2 = 1.0; xi = 1.0; s2 = np.var(y)
    dr = []
    for it in range(iters):
        prec = np.where(pen, 1 / (lam2 * tau2), 1e-8)
        A = ZtZ + np.diag(prec); L = np.linalg.cholesky(A)
        b = np.linalg.solve(A, Zty) + np.sqrt(s2) * np.linalg.solve(L.T, rng.normal(size=q))
        rss = yty - 2 * b @ Zty + b @ ZtZ @ b
        bp = b[pen]
        s2 = 1 / rng.gamma((nn + pen.sum()) / 2, 1 / (rss / 2 + (bp ** 2 / (lam2[pen] * tau2)).sum() / 2))
        lam2[pen] = 1 / rng.gamma(1, 1 / (1 / nu[pen] + bp ** 2 / (2 * tau2 * s2)))
        nu[pen] = 1 / rng.gamma(1, 1 / (1 + 1 / lam2[pen]))
        tau2 = 1 / rng.gamma((pen.sum() + 1) / 2, 1 / (1 / xi + (bp ** 2 / lam2[pen]).sum() / (2 * s2)))
        xi = 1 / rng.gamma(1, 1 / (1 + 1 / tau2))
        if it >= burn:
            dr.append(b.copy())
    return np.array(dr)


def autocov(v):
    n = len(v)
    v = v - v.mean()
    f = np.fft.rfft(v, 2 * n)
    return np.fft.irfft(f * np.conj(f))[:n] / n


def split_rhat_ess(chains):
    """chains: (m, n). Split each chain in half, then the classic R-hat and the Geyer truncated effective sample size."""
    m, n = chains.shape
    h = n // 2
    x = np.concatenate([chains[:, :h], chains[:, h:2 * h]], axis=0)
    M, N = x.shape
    W = x.var(axis=1, ddof=1).mean()
    B = N * x.mean(axis=1).var(ddof=1)
    var_plus = (N - 1) / N * W + B / N
    rhat = float(np.sqrt(var_plus / W))
    mean_acov = np.mean([autocov(x[i]) * N / (N - 1) for i in range(M)], axis=0)
    rho = 1.0 - (W - mean_acov) / var_plus
    s = 0.0
    for t in range(1, N - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        s += pair
    tau_int = 1.0 + 2.0 * s
    ess = float(M * N / max(tau_int, 1.0 / np.log10(M * N)))
    return rhat, ess


def summarize(name, rhats, esss, draws_per_chain, extra):
    rhats = np.array(rhats); esss = np.array(esss)
    out = dict(design=name, data_sets=int(len(rhats)), chains=4, draws_per_chain=int(draws_per_chain), total_draws=int(4 * draws_per_chain),
               split_rhat_max=float(rhats.max()), split_rhat_median=float(np.median(rhats)), share_rhat_below_1_01=float((rhats < 1.01).mean()),
               ess_min=float(esss.min()), ess_median=float(np.median(esss)), **extra,
               rhat_all=[float(x) for x in rhats], ess_all=[float(x) for x in esss])
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if not k.endswith("_all")}, flush=True)
    return out


def simulation(p, sets, coupled, seed):
    from sklearn.linear_model import LassoCV
    rng = np.random.default_rng(seed)
    n = 200
    rhats, esss = [], []
    t0 = time.time()
    for r in range(sets):
        X = rng.normal(size=(n, p))
        g = np.zeros(p); g[:5] = 1.0
        t = X @ g + rng.normal(size=n)
        y = t + X @ g + rng.normal(size=n)
        if coupled:
            that = LassoCV(cv=5).fit(X, t).predict(X)
            Z = np.column_stack([t, that, X]); unpen = {0, 1}
        else:
            Z = np.column_stack([t, X]); unpen = {0}
        ch = np.array([hs_gibbs(y, Z, unpen=unpen, rng=np.random.default_rng(1000 * r + c))[:, 0] for c in range(4)])
        rh, es = split_rhat_ess(ch)
        rhats.append(rh); esss.append(es)
    name = "simulation n=200, p=%d, %s horseshoe, five confounders gamma=beta=1, tau=1" % (p, "coupled" if coupled else "independent")
    return summarize(name, rhats, esss, 1000, dict(iterations=1500, burn_in=500, seed=seed, seconds=round(time.time() - t0, 1)))


def weak30(sets=10, seed=83):
    """Design weak30 of ric_sim.py (n = 200, p = 150, 30 confounders with gamma = beta = 0.3, columns 30 to 33 with beta = 1 in the outcome equation only, tau = 1):
    four chains of 1,500 sweeps (500 discarded) from the default start for the independent horseshoe and for both coupled horseshoes (propensity covariate fitted on
    the whole sample, and cross fitted on KFold(5, shuffle, random_state = data set index) as in ric_sim.py). Writes results/res_ric_chains_weak30.json."""
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import KFold
    rng = np.random.default_rng(seed)
    n, p = 200, 150
    res = {"independent": ([], []), "coupled_whole_sample": ([], []), "coupled_cross_fitted": ([], [])}
    t0 = time.time()
    for r in range(sets):
        X = rng.normal(size=(n, p))
        g = np.zeros(p); g[:30] = 0.3
        b = np.zeros(p); b[:30] = 0.3; b[30:34] = 1.0
        t = X @ g + rng.normal(size=n)
        y = t + X @ b + rng.normal(size=n)
        that = LassoCV(cv=5).fit(X, t).predict(X)
        mcf = np.zeros(n)
        for tr, te in KFold(5, shuffle=True, random_state=r).split(X):
            mcf[te] = LassoCV(cv=5).fit(X[tr], t[tr]).predict(X[te])
        for name, Z, unpen in (("independent", np.column_stack([t, X]), {0}), ("coupled_whole_sample", np.column_stack([t, that, X]), {0, 1}),
                               ("coupled_cross_fitted", np.column_stack([t, mcf, X]), {0, 1})):
            ch = np.array([hs_gibbs(y, Z, unpen=unpen, rng=np.random.default_rng(1000 * r + c))[:, 0] for c in range(4)])
            rh, es = split_rhat_ess(ch)
            res[name][0].append(rh); res[name][1].append(es)
        print("data set", r, {k: (round(v[0][-1], 4), round(v[1][-1])) for k, v in res.items()}, flush=True)
    runs = [summarize("weak30 n=200, p=150, 30 confounders 0.3, 4 outcome only 1, %s horseshoe" % k, v[0], v[1], 1000,
                      dict(iterations=1500, burn_in=500, seed=seed, seconds=round(time.time() - t0, 1))) for k, v in res.items()]
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "res_ric_chains_weak30.json"), "w") as f:
        json.dump(dict(note="split R-hat and effective sample size of the treatment coefficient, four chains per data set, design weak30", runs=runs), f, indent=1)
    print("wrote results/res_ric_chains_weak30.json")


def k401(coupled):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    t0 = time.time()
    d = pd.read_stata(os.path.join(ROOT, "data", "sipp1991.dta"))
    xv = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]
    Y = d.net_tfa.values.astype(float) / 1000.0
    D = d.e401.values.astype(float)
    X = StandardScaler().fit_transform(d[xv].values.astype(float))
    P = PolynomialFeatures(2, include_bias=False).fit_transform(X)
    _, keep = np.unique(np.round(P, 8), axis=1, return_index=True)
    P = StandardScaler().fit_transform(P[:, np.sort(keep)])
    one = np.ones(len(Y))
    if coupled:
        prop = LogisticRegressionCV(cv=5, Cs=10, max_iter=2000).fit(P, D).predict_proba(P)[:, 1]
        Z = np.column_stack([D, one, prop, P]); unpen = {0, 1, 2}
    else:
        Z = np.column_stack([D, one, P]); unpen = {0, 1}
    ch = np.array([hs401(Z, Y, unpen, np.random.default_rng(401 + c))[:, 0] for c in range(4)])
    rh, es = split_rhat_ess(ch)
    name = "401(k) data, n=9915, %s horseshoe" % ("coupled" if coupled else "independent")
    return summarize(name, [rh], [es], 2000, dict(iterations=3000, burn_in=1000, seconds=round(time.time() - t0, 1)))


def main():
    import sys
    if "--weak30" in sys.argv:
        weak30()
        return
    runs = [simulation(50, 20, False, 77), simulation(50, 20, True, 78),
            simulation(150, 10, False, 79), simulation(150, 10, True, 80),
            simulation(300, 10, False, 81), simulation(300, 10, True, 82),
            k401(False), k401(True)]
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "res_ric_chains.json"), "w") as f:
        json.dump(dict(note="split R-hat and effective sample size of the treatment coefficient, four chains per data set", runs=runs), f, indent=1)
    print("wrote results/res_ric_chains.json")


if __name__ == "__main__":
    main()
