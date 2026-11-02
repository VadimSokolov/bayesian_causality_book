"""Chapter 16 (Exercise chk401:1) and Chapter 18 (Step 5): the 401(k) horseshoe with Student t errors instead of Gaussian errors.

Data and basis as in app401k_book.py (1991 SIPP, n = 9,915; nine standardized covariates, squares and products: 54 terms; outcome in thousands of dollars).
Model: y_i = x_i' b + e_i with e_i = sigma z_i / sqrt(w_i), z_i standard normal and w_i ~ Gamma(nu/2, rate nu/2), that is Student t errors with nu = 3 degrees of freedom
and scale sigma, as a scale mixture of normals; the horseshoe on the 54 terms (half Cauchy local scales and one global scale, scaled by sigma) is the one of app401k_book.py,
and the unpenalized terms are the effect (or the five quintile effects), an intercept and the cross validated logistic propensity (the coupled prior). One Gibbs sweep adds
one step to that sampler, the draw of the weights w_i given the residuals. 3,000 sweeps, the first 1,000 discarded, one chain, as in app401k_book.py.
Saved: the effect of eligibility with one common effect (hs_coupled_t), and with five income quintile effects the quintile posteriors and the accountings over all
households, the eligible and the ineligible (quintile_t, acc_ate_t, acc_att_t, acc_atu_t), each as [mean, sd, 2.5%, 97.5%]; sigma and the share of the weights below 0.1
(the observations the t model treats as outliers); the split R-hat and effective sample size of the common effect from four chains of the same length; and the common effect for a sweep of the degrees of freedom (1, 2, 3, 5, 10, 30 and the Gaussian limit, nu = None, one chain each, seeds 4100 to 4106), which moves the target from the centre of the outcome distribution toward its mean (compare k401_quantiles.py). Each run has its own seed, so the keys written before the sweep do not change when it is added.

usage (from the repository root): python book_scripts/k401_student_t.py
writes results/res_k401_t.json
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegressionCV

warnings.filterwarnings("ignore")
t0 = time.time()
NU = 3.0
d = pd.read_stata("data/sipp1991.dta")
xv = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]
Y = d.net_tfa.values.astype(float) / 1000.0
D = d.e401.values.astype(float)
X = StandardScaler().fit_transform(d[xv].values.astype(float))
P = PolynomialFeatures(2, include_bias=False).fit_transform(X)
_, keep = np.unique(np.round(P, 8), axis=1, return_index=True)
P = StandardScaler().fit_transform(P[:, np.sort(keep)])
n, p = P.shape
one = np.ones(n)
prop = LogisticRegressionCV(cv=5, Cs=10, max_iter=2000).fit(P, D).predict_proba(P)[:, 1]


def summ(x):
    return [float(np.mean(x)), float(np.std(x)), float(np.quantile(x, .025)), float(np.quantile(x, .975))]


def hs_t(Z, y, unpen, rng, nu=NU, iters=3000, burn=1000):
    nn, q = Z.shape
    pen = np.ones(q, bool); pen[list(unpen)] = False
    lam2 = np.ones(q); nv = np.ones(q); tau2 = 1.0; xi = 1.0; s2 = np.var(y); om = np.ones(nn)
    dr, sig, low = [], [], []
    for it in range(iters):
        ZtZ = Z.T @ (Z * om[:, None]); Zty = Z.T @ (om * y)
        A = ZtZ + np.diag(np.where(pen, 1 / (lam2 * tau2), 1e-8)); L = np.linalg.cholesky(A)
        b = np.linalg.solve(A, Zty) + np.sqrt(s2) * np.linalg.solve(L.T, rng.normal(size=q))
        r = y - Z @ b
        rss = float(om @ (r ** 2)); bp = b[pen]
        s2 = 1 / rng.gamma((nn + pen.sum()) / 2, 1 / (rss / 2 + (bp ** 2 / (lam2[pen] * tau2)).sum() / 2))
        lam2[pen] = 1 / rng.gamma(1, 1 / (1 / nv[pen] + bp ** 2 / (2 * tau2 * s2)))
        nv[pen] = 1 / rng.gamma(1, 1 / (1 + 1 / lam2[pen]))
        tau2 = 1 / rng.gamma((pen.sum() + 1) / 2, 1 / (1 / xi + (bp ** 2 / lam2[pen]).sum() / (2 * s2)))
        xi = 1 / rng.gamma(1, 1 / (1 + 1 / tau2))
        if nu is not None:
            om = rng.gamma((nu + 1) / 2, 2 / (nu + r ** 2 / s2))
        if it >= burn:
            dr.append(b.copy()); sig.append(np.sqrt(s2)); low.append(float((om < 0.1).mean()))
    return np.array(dr), np.array(sig), np.array(low)


def split_rhat_ess(ch):
    ch = np.asarray(ch, float); m, nn = ch.shape; h = nn // 2
    parts = np.vstack([ch[:, :h], ch[:, h:2 * h]]); mm, hh = parts.shape
    W = parts.var(axis=1, ddof=1).mean(); B = hh * parts.mean(axis=1).var(ddof=1)
    rhat = float(np.sqrt(((hh - 1) / hh * W + B / hh) / W))
    x = ch - ch.mean(axis=1, keepdims=True); f = np.fft.rfft(x, 2 * nn, axis=1); ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :nn].mean(axis=0)
    ac = ac / ac[0]; s = 0.0
    for k in range(1, nn - 1, 2):
        pair = ac[k] + ac[k + 1]
        if pair < 0: break
        s += pair
    return rhat, float(m * nn / (1 + 2 * s))


out = {"nu": NU, "n": int(n), "p_basis": int(p), "iterations": 3000, "burn_in": 1000}
chains = []
for c in range(4):
    rng = np.random.default_rng(4010 + c)
    dr, sig, low = hs_t(np.column_stack([D, one, prop, P]), Y, {0, 1, 2}, rng)
    chains.append(dr[:, 0])
    if c == 0:
        out["hs_coupled_t"] = summ(dr[:, 0]); out["sigma"] = summ(sig); out["share_weights_below_0.1"] = float(low.mean())
    print("common effect chain", c, round(time.time() - t0, 1), flush=True)
rh, es = split_rhat_ess(np.array(chains))
out["common_effect_rhat"] = rh; out["common_effect_ess"] = es

rng = np.random.default_rng(4020)
q = pd.qcut(d.inc, 5, labels=False).values
Q = np.column_stack([(q == k) * D for k in range(5)])
dr3, _, _ = hs_t(np.column_stack([Q, one, prop, P]), Y, {0, 1, 2, 3, 4, 5, 6}, rng)
tq = dr3[:, :5]
out["quintile_t"] = [summ(tq[:, k]) for k in range(5)]
sh_all = np.array([np.mean(q == k) for k in range(5)])
sh_tr = np.array([np.mean(q[D == 1] == k) for k in range(5)])
sh_un = np.array([np.mean(q[D == 0] == k) for k in range(5)])
out["acc_ate_t"] = summ(tq @ sh_all); out["acc_att_t"] = summ(tq @ sh_tr); out["acc_atu_t"] = summ(tq @ sh_un)
sweep = []
for k, nu in enumerate([1.0, 2.0, 3.0, 5.0, 10.0, 30.0, None]):
    rng = np.random.default_rng(4100 + k)
    dr, sig, low = hs_t(np.column_stack([D, one, prop, P]), Y, {0, 1, 2}, rng, nu=nu)
    sweep.append({"nu": nu, "effect": summ(dr[:, 0]), "sigma": float(sig.mean()), "share_weights_below_0.1": float(low.mean())})
    print("nu", nu, round(time.time() - t0, 1), flush=True)
out["nu_sweep"] = sweep
out["seconds"] = round(time.time() - t0, 1)
with open("results/res_k401_t.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
