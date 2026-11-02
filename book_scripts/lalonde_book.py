import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from causaldata import nsw_mixtape, cps_mixtape
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegressionCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(1986)
nsw = nsw_mixtape.load_pandas().data; cps = cps_mixtape.load_pandas().data
out = {"experimental": float(nsw[nsw.treat == 1].re78.mean() - nsw[nsw.treat == 0].re78.mean())}
d = pd.concat([nsw[nsw.treat == 1], cps.assign(treat=0)], ignore_index=True)
d["u74"] = (d.re74 == 0).astype(float); d["u75"] = (d.re75 == 0).astype(float)
xv = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75", "u74", "u75"]
Y = d.re78.values.astype(float) / 1000; D = d.treat.values.astype(float); n = len(Y)
X = StandardScaler().fit_transform(d[xv].values.astype(float))
P = PolynomialFeatures(2, include_bias=False).fit_transform(X)
_, k = np.unique(np.round(P, 8), axis=1, return_index=True)
P = P[:, np.sort(k)]; P = P[:, P.std(0) > 0]; P = StandardScaler().fit_transform(P)
out["n"] = n; out["n_treated"] = int(D.sum()); out["p_basis"] = P.shape[1]
out["naive"] = float(Y[D == 1].mean() - Y[D == 0].mean())
out["ols"] = float(LinearRegression().fit(np.column_stack([D, X]), Y).coef_[0])
prop = LogisticRegressionCV(cv=5, Cs=10, max_iter=3000).fit(P, D).predict_proba(P)[:, 1]
out["prop_controls_below_001"] = float(np.mean(prop[D == 0] < 0.001)); out["prop_treated_median"] = float(np.median(prop[D == 1]))
def dml_att(Z, seed):
    kf = KFold(5, shuffle=True, random_state=seed); m0 = np.zeros(n); e = np.zeros(n)
    for tr, te in kf.split(Z):
        c = tr[D[tr] == 0]
        m0[te] = RandomForestRegressor(200, min_samples_leaf=5, random_state=seed).fit(Z[c], Y[c]).predict(Z[te])
        e[te] = LogisticRegressionCV(cv=3, Cs=5, max_iter=3000).fit(P[tr], D[tr]).predict_proba(P[te])[:, 1]
    e = np.clip(e, 1e-3, 1 - 1e-3); p1 = D.mean()
    psi = (D * (Y - m0) - (1 - D) * e / (1 - e) * (Y - m0)) / p1
    th = psi.mean(); se = psi.std() / np.sqrt(n)
    return float(th), float(se)
out["dml_att"] = dml_att(X, 1)
out["dml_att_splits"] = [dml_att(X, s)[0] for s in range(2, 7)]
def hs(Z, y, unpen, iters=3000, burn=1000):
    nn, q = Z.shape; pen = np.ones(q, bool); pen[list(unpen)] = False
    ZtZ = Z.T @ Z; Zty = Z.T @ y; yty = y @ y
    lam2 = np.ones(q); nu = np.ones(q); tau2 = 1.0; xi = 1.0; s2 = np.var(y); dr = []
    for it in range(iters):
        A = ZtZ + np.diag(np.where(pen, 1 / (lam2 * tau2), 1e-8)); L = np.linalg.cholesky(A)
        b = np.linalg.solve(A, Zty) + np.sqrt(s2) * np.linalg.solve(L.T, rng.normal(size=q))
        rss = yty - 2 * b @ Zty + b @ ZtZ @ b; bp = b[pen]
        s2 = 1 / rng.gamma((nn + pen.sum()) / 2, 1 / (rss / 2 + (bp ** 2 / (lam2[pen] * tau2)).sum() / 2))
        lam2[pen] = 1 / rng.gamma(1, 1 / (1 / nu[pen] + bp ** 2 / (2 * tau2 * s2))); nu[pen] = 1 / rng.gamma(1, 1 / (1 + 1 / lam2[pen]))
        tau2 = 1 / rng.gamma((pen.sum() + 1) / 2, 1 / (1 / xi + (bp ** 2 / lam2[pen]).sum() / (2 * s2))); xi = 1 / rng.gamma(1, 1 / (1 + 1 / tau2))
        if it >= burn: dr.append(b.copy())
    return np.array(dr)
def summ(x): return [float(np.mean(x)), float(np.std(x)), float(np.quantile(x, .025)), float(np.quantile(x, .975))]
one = np.ones(n); lp = np.log(prop / (1 - prop))
# outcome model with treatment interacting with the logit propensity, so effects can vary
Z = np.column_stack([D, D * lp, one, lp, P])
dr = hs(Z, Y, {0, 1, 2, 3})
tau_i = dr[:, [0]] + dr[:, [1]] * lp[None, :]
out["att"] = summ(tau_i[:, D == 1].mean(1)); out["ate_all"] = summ(tau_i.mean(1))
Zi = np.column_stack([D, one, P]); dri = hs(Zi, Y, {0, 1}); out["hs_indep_const"] = summ(dri[:, 0])
json.dump(out, open("results/res_lalonde.json", "w"), indent=1); print(json.dumps(out, indent=1))
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.9))
axs[0].hist(lp[D == 0], bins=60, density=True, alpha=.6, label="CPS controls"); axs[0].hist(lp[D == 1], bins=30, density=True, alpha=.6, label="NSW treated")
axs[0].set_xlabel("Estimated logit propensity"); axs[0].legend(frameon=False, fontsize=7)
axs[1].hist(tau_i[:, D == 1].mean(1), bins=40, density=True, histtype="step", label="effect on treated")
axs[1].hist(tau_i.mean(1), bins=40, density=True, histtype="step", label="average over all")
axs[1].axvline(out["experimental"] / 1000, color="k", ls="--", lw=.8); axs[1].set_xlabel("Effect on 1978 earnings, $000"); axs[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig("figs/lalonde.pdf")
