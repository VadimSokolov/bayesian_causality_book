import numpy as np, pandas as pd, json, time, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegressionCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
t0 = time.time()
rng = np.random.default_rng(401)
d = pd.read_stata("data/sipp1991.dta")
xv = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]
Y = d.net_tfa.values.astype(float) / 1000.0          # thousands of dollars
D = d.e401.values.astype(float)
X = StandardScaler().fit_transform(d[xv].values.astype(float))
P = PolynomialFeatures(2, include_bias=False).fit_transform(X)
_, keep = np.unique(np.round(P, 8), axis=1, return_index=True)
P = StandardScaler().fit_transform(P[:, np.sort(keep)])
n, p = P.shape
out = {"n": int(n), "p_basis": int(p), "treated_share": float(D.mean())}

out["naive"] = float(Y[D == 1].mean() - Y[D == 0].mean())
out["ols"] = float(LinearRegression().fit(np.column_stack([D, X]), Y).coef_[0])
out["ols54"] = float(LinearRegression().fit(np.column_stack([D, P]), Y).coef_[0])   # least squares on the 54 term basis (9.84 in Chapters 5 and 16)

def dml(Z, learner, seed):
    kf = KFold(5, shuffle=True, random_state=seed)
    ry = np.zeros(n); rd = np.zeros(n)
    for tr, te in kf.split(Z):
        if learner == "rf":
            my = RandomForestRegressor(200, min_samples_leaf=5, max_features=4, random_state=seed).fit(Z[tr], Y[tr])
            md = RandomForestRegressor(200, min_samples_leaf=5, max_features=4, random_state=seed).fit(Z[tr], D[tr])
        else:
            my = LassoCV(cv=5).fit(Z[tr], Y[tr]); md = LassoCV(cv=5).fit(Z[tr], D[tr])
        ry[te] = Y[te] - my.predict(Z[te]); rd[te] = D[te] - md.predict(Z[te])
    th = (rd @ ry) / (rd @ rd)
    psi = (ry - th * rd) * rd
    se = np.sqrt(np.mean(psi ** 2) / np.mean(rd ** 2) ** 2 / n)
    return float(th), float(se)

import os
if os.path.exists("results/res401k.json"):
    out = {**json.load(open("results/res401k.json")), **out}
if "dml_rf" not in out:
    # partition used in the book's table (random_state = 1); Chapter 12 reports the spread over partitions
    out["dml_rf"] = list(dml(X, "rf", 1)); out["dml_lasso"] = list(dml(P, "lasso", 1))
    print("dml", out["dml_rf"], out["dml_lasso"], flush=True)
def hs(Z, y, unpen, iters=3000, burn=1000):
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

def summ(x): return [float(np.mean(x)), float(np.std(x)), float(np.quantile(x, .025)), float(np.quantile(x, .975))]
one = np.ones(n)
Z1 = np.column_stack([D, one, P])
dr1 = hs(Z1, Y, {0, 1}); out["hs_indep"] = summ(dr1[:, 0]); print("hs indep", out["hs_indep"], time.time() - t0, flush=True)
prop = LogisticRegressionCV(cv=5, Cs=10, max_iter=2000).fit(P, D).predict_proba(P)[:, 1]
out["prop_range"] = [float(prop.min()), float(prop.max())]
Z2 = np.column_stack([D, one, prop, P])
dr2 = hs(Z2, Y, {0, 1, 2}); out["hs_coupled"] = summ(dr2[:, 0]); print("hs coupled", out["hs_coupled"], time.time() - t0, flush=True)

# Heterogeneity by income quintile: interactions of D with quintile dummies, unpenalized
q = pd.qcut(d.inc, 5, labels=False).values
Q = np.column_stack([(q == k) * D for k in range(5)])
Z3 = np.column_stack([Q, one, prop, P])
dr3 = hs(Z3, Y, {0, 1, 2, 3, 4, 5, 6})
tq = dr3[:, :5]
out["quintile"] = [summ(tq[:, k]) for k in range(5)]
shares_all = np.array([np.mean(q == k) for k in range(5)])
shares_tr = np.array([np.mean(q[D == 1] == k) for k in range(5)])
shares_un = np.array([np.mean(q[D == 0] == k) for k in range(5)])
out["acc_ate"] = summ(tq @ shares_all); out["acc_att"] = summ(tq @ shares_tr); out["acc_atu"] = summ(tq @ shares_un)
print("quintiles", out["quintile"], time.time() - t0, flush=True)

fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.9))
axs[0].hist(dr1[:, 0], bins=50, density=True, histtype="step", label="independent horseshoe")
axs[0].hist(dr2[:, 0], bins=50, density=True, histtype="step", label="coupled horseshoe")
axs[0].axvline(out["dml_rf"][0], color="k", ls="--", lw=0.8)
axs[0].set_xlabel("Effect of eligibility, $000"); axs[0].legend(frameon=False, fontsize=7)
m = [s[0] for s in out["quintile"]]; lo = [s[2] for s in out["quintile"]]; hi = [s[3] for s in out["quintile"]]
axs[1].errorbar(range(1, 6), m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt="o", capsize=3)
axs[1].set_xlabel("Income quintile"); axs[1].set_ylabel("Effect, $000")
fig.tight_layout(); fig.savefig("figs/k401.pdf"); plt.close()
json.dump(out, open("results/res401k.json", "w"), indent=1)
print("done", time.time() - t0)
