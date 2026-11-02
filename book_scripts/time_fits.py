"""Timings quoted in Chapter 9 (Sections comp:sec:why and comp:sec:cost): how long the horseshoe Gibbs sampler takes, and how that
compares with the lasso and forest fits inside DML. One thread, wall clock, seconds.

Design A is the simulation of Section sec:ricsim: n = 200, X ~ N(0, I_50), T = X_1 + ... + X_5 + nu and Y = T + X_1 + ... + X_5 + eps
with standard normal nu and eps, so the regression of Y has p = 51 columns (T and X). 20 replications.
Design B is the 401(k) analysis of Chapter 10: n = 9,915, the nine characteristics expanded to 54 terms, plus the treatment and an
intercept, 56 columns. 5 repetitions.

Timed in design A: the Gibbs sampler of Chapter 8 (1,500 iterations), one LassoCV(cv=5) fit of Y on (T, X), and the 10 LassoCV fits of
a five fold cross fitted partialling out estimate (the lasso DML of Section sec:ricsim).
Timed in design B: the Gibbs sampler (3,000 iterations, as in app401k_book.py), one LassoCV(cv=5) fit on the 54 term basis and four
fifths of the rows, one random forest fit (200 trees, min_samples_leaf 5, 4 features per split, as in app401k_book.py) on the nine
characteristics and four fifths of the rows, and the 10 forest fits of one five fold cross fitted estimate.

usage (from the repository root): python book_scripts/time_fits.py
writes results/res_timing.json
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import platform
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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
        if it >= burn: dr.append(b.copy())
    return np.array(dr)


def clock(f):
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0


def summary(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0, min=float(x.min()), max=float(x.max()), runs=int(len(x)))


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def design_a(rng, n=200, p=50):
    X = rng.normal(size=(n, p))
    g = np.zeros(p); g[:5] = 1.0
    t = X @ g + rng.normal(size=n)
    y = t + X @ g + rng.normal(size=n)
    return X, t, y


def main():
    out = dict(cpu=cpu_model(), threads=1, python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, units="seconds, wall clock")
    # design A
    rng = np.random.default_rng(8)
    A = dict(n=200, p_columns=51, reps=20, hs_iters=1500)
    hs, lasso1, dml10 = [], [], []
    for r in range(20):
        X, t, y = design_a(rng)
        Z = np.column_stack([t, X])
        hs.append(clock(lambda: hs_gibbs(y, Z, unpen={0}, rng=rng)))
        lasso1.append(clock(lambda: LassoCV(cv=5).fit(Z, y)))

        def crossfit():
            for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
                LassoCV(cv=5).fit(X[tr], y[tr]).predict(X[te])
                LassoCV(cv=5).fit(X[tr], t[tr]).predict(X[te])
        dml10.append(clock(crossfit))
    A.update(horseshoe_gibbs=summary(hs), lasso_cv_one_fit=summary(lasso1), dml_lasso_cross_fitting_10_fits=summary(dml10))
    out["design_A_simulation_n200_p51"] = A
    print("design A", {k: (round(v["mean"], 3) if isinstance(v, dict) else v) for k, v in A.items()}, flush=True)
    # design B
    d = pd.read_stata(os.path.join(ROOT, "data", "sipp1991.dta"))
    xv = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]
    Y = d.net_tfa.values.astype(float) / 1000.0
    D = d.e401.values.astype(float)
    X = StandardScaler().fit_transform(d[xv].values.astype(float))
    P = PolynomialFeatures(2, include_bias=False).fit_transform(X)
    _, keep = np.unique(np.round(P, 8), axis=1, return_index=True)
    P = StandardScaler().fit_transform(P[:, np.sort(keep)])
    n = len(Y)
    Z1 = np.column_stack([D, np.ones(n), P])
    tr, _ = next(KFold(5, shuffle=True, random_state=1).split(X))
    B = dict(n=int(n), p_columns=int(Z1.shape[1]), reps=5, hs_iters=3000)
    hs, lasso1, rf1, rf10 = [], [], [], []
    rng = np.random.default_rng(401)
    for r in range(5):
        hs.append(clock(lambda: hs401(Z1, Y, {0, 1}, rng)))
        lasso1.append(clock(lambda: LassoCV(cv=5).fit(P[tr], Y[tr])))
        rf1.append(clock(lambda: RandomForestRegressor(200, min_samples_leaf=5, max_features=4, random_state=r).fit(X[tr], Y[tr])))

        def rf_crossfit():
            for a, b in KFold(5, shuffle=True, random_state=1).split(X):
                RandomForestRegressor(200, min_samples_leaf=5, max_features=4, random_state=r).fit(X[a], Y[a]).predict(X[b])
                RandomForestRegressor(200, min_samples_leaf=5, max_features=4, random_state=r).fit(X[a], D[a]).predict(X[b])
        rf10.append(clock(rf_crossfit))
    B.update(horseshoe_gibbs=summary(hs), lasso_cv_one_fit=summary(lasso1), random_forest_one_fit=summary(rf1), dml_forest_cross_fitting_10_fits=summary(rf10))
    out["design_B_401k_n9915_p56"] = B
    print("design B", {k: (round(v["mean"], 3) if isinstance(v, dict) else v) for k, v in B.items()}, flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "res_timing.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results/res_timing.json")


if __name__ == "__main__":
    main()
