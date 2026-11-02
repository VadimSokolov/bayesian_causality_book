"""Chapter 18 (Step 5 of the workflow, Figure fig:ppc401k), item F10 of round 1 (October 2026): a posterior predictive check of the 401(k) outcome model from the two fitted posteriors, in place of the plug in check of workflow_figs.py (which is unchanged).
Run from the repository root; writes results/res_k401_ppc.json and figs/ppc401k_post.pdf.

Data and fits: 1991 SIPP, n = 9,915 households, the 54 term basis, outcome net financial assets in thousands of dollars, exactly as k401_student_t.py and app401k_book.py. Two models, each the coupled horseshoe (treatment, intercept and cross validated logistic propensity unpenalized, horseshoe on the 54 terms),
fitted by the Gibbs sampler of k401_student_t.py with 3,000 sweeps and the first 1,000 dropped: one with Gaussian errors, and one with Student t errors with 3 degrees of freedom as a scale mixture of normals. Seeds 4301 (Gaussian) and 4302 (t).
Check: from each model 1,000 posterior draws (every second kept draw), and for each draw one replicated data set at the observed covariates and eligibility: the draw's regression function plus error, Gaussian errors at the draw's sigma for the Gaussian model; for the t model, t_3 errors at the draw's scale sigma with the weights w_i redrawn from their prior Gamma(3/2, rate 3/2), not taken from the fit.
Reported: (a) observed against replicated quantiles at levels 0.01 to 0.99 (99 levels), overall and separately for eligible and ineligible households, with the median and the 95% range of the replicated quantiles and the share of levels at which the observed quantile lies outside the range, and the replicated range of the 1%, 5%, 50%, 95% and 99% quantiles;
(b) the mean of assets by eligibility within income quintile (quintiles of income as in app401k_book.py), observed against the replicated mean with its 95% range; (c) posterior predictive p values (the share of replicates whose statistic is at least the observed) for the mean, standard deviation, skewness, minimum and maximum, overall and by eligibility.
"""
import json, time, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegressionCV
warnings.filterwarnings("ignore")
t0 = time.time(); NU = 3.0
d = pd.read_stata("data/sipp1991.dta")
xv = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]
Y = d.net_tfa.values.astype(float) / 1000.0; D = d.e401.values.astype(float)
X = StandardScaler().fit_transform(d[xv].values.astype(float))
P = PolynomialFeatures(2, include_bias=False).fit_transform(X)
_, keep = np.unique(np.round(P, 8), axis=1, return_index=True)
P = StandardScaler().fit_transform(P[:, np.sort(keep)])
n, p = P.shape; one = np.ones(n)
prop = LogisticRegressionCV(cv=5, Cs=10, max_iter=2000).fit(P, D).predict_proba(P)[:, 1]
Zfull = np.column_stack([D, one, prop, P])

def hs_t(Z, y, unpen, rng, nu=None, iters=3000, burn=1000):
    nn, q = Z.shape; pen = np.ones(q, bool); pen[list(unpen)] = False
    lam2 = np.ones(q); nv = np.ones(q); tau2 = 1.0; xi = 1.0; s2 = np.var(y); om = np.ones(nn); dr, sig = [], []
    for it in range(iters):
        ZtZ = Z.T @ (Z * om[:, None]); Zty = Z.T @ (om * y)
        A = ZtZ + np.diag(np.where(pen, 1 / (lam2 * tau2), 1e-8)); L = np.linalg.cholesky(A)
        b = np.linalg.solve(A, Zty) + np.sqrt(s2) * np.linalg.solve(L.T, rng.normal(size=q))
        r = y - Z @ b; rss = float(om @ (r ** 2)); bp = b[pen]
        s2 = 1 / rng.gamma((nn + pen.sum()) / 2, 1 / (rss / 2 + (bp ** 2 / (lam2[pen] * tau2)).sum() / 2))
        lam2[pen] = 1 / rng.gamma(1, 1 / (1 / nv[pen] + bp ** 2 / (2 * tau2 * s2))); nv[pen] = 1 / rng.gamma(1, 1 / (1 + 1 / lam2[pen]))
        tau2 = 1 / rng.gamma((pen.sum() + 1) / 2, 1 / (1 / xi + (bp ** 2 / lam2[pen]).sum() / (2 * s2))); xi = 1 / rng.gamma(1, 1 / (1 + 1 / tau2))
        if nu is not None: om = rng.gamma((nu + 1) / 2, 2 / (nu + r ** 2 / s2))
        if it >= burn: dr.append(b.copy()); sig.append(np.sqrt(s2))
    return np.array(dr), np.array(sig)

qint = pd.qcut(d.inc, 5, labels=False).values
levels = np.round(np.linspace(0.01, 0.99, 99), 2)
key_lv = [0.01, 0.05, 0.5, 0.95, 0.99]
def skew(v): return float(((v - v.mean()) ** 3).mean() / v.std() ** 3)
STATS = {"mean": np.mean, "sd": np.std, "skewness": skew, "min": np.min, "max": np.max}
out = {"n": int(n), "levels": levels.tolist(), "draws": 1000, "nu": NU, "obs_mean": float(Y.mean()), "obs_sd": float(Y.std())}
fig_data = {}
for name, nu, seed in [("gaussian", None, 4301), ("student_t", NU, 4302)]:
    rng = np.random.default_rng(seed)
    dr, sig = hs_t(Zfull, Y, {0, 1, 2}, rng, nu=nu); print(name, "fitted", dr[:, 0].mean(), sig.mean(), round(time.time() - t0, 1), flush=True)
    dr = dr[::2]; sig = sig[::2]; K = len(sig)                       # 1,000 draws
    res = {"effect": [float(dr[:, 0].mean()), float(dr[:, 0].std())], "sigma_mean": float(sig.mean())}
    qrep = {"all": np.zeros((K, len(levels))), 1: np.zeros((K, len(levels))), 0: np.zeros((K, len(levels)))}
    mean_cell = np.zeros((K, 5, 2)); st = {g: {s: np.zeros(K) for s in STATS} for g in ["all", 1, 0]}
    for k in range(K):
        if nu is None: eps = sig[k] * rng.normal(size=n)
        else:
            w = rng.gamma(nu / 2, 2 / nu, size=n); eps = sig[k] * rng.normal(size=n) / np.sqrt(w)
        yr = Zfull @ dr[k] + eps
        qrep["all"][k] = np.quantile(yr, levels)
        for g in [1, 0]:
            m = D == g; qrep[g][k] = np.quantile(yr[m], levels)
            for qq in range(5): mean_cell[k, qq, g] = yr[m & (qint == qq)].mean()
        for g, m in [("all", np.ones(n, bool)), (1, D == 1), (0, D == 0)]:
            for s, f in STATS.items(): st[g][s][k] = f(yr[m])
    for g, m in [("all", np.ones(n, bool)), (1, D == 1), (0, D == 0)]:
        qo = np.quantile(Y[m], levels); lo = np.quantile(qrep[g], .025, 0); hi = np.quantile(qrep[g], .975, 0); md = np.median(qrep[g], 0)
        lab = {"all": "all", 1: "eligible", 0: "ineligible"}[g]
        res[lab] = {"n": int(m.sum()), "share_levels_outside_95": float(np.mean((qo < lo) | (qo > hi))),
                    "key": {str(l): {"obs": float(qo[list(levels).index(l)]), "rep_median": float(md[list(levels).index(l)]), "rep_lo": float(lo[list(levels).index(l)]), "rep_hi": float(hi[list(levels).index(l)])} for l in key_lv},
                    "stats": {s: {"obs": float(f(Y[m])), "rep_mean": float(st[g][s].mean()), "rep_lo": float(np.quantile(st[g][s], .025)), "rep_hi": float(np.quantile(st[g][s], .975)), "ppp": float((st[g][s] >= f(Y[m])).mean())} for s, f in STATS.items()}}
        fig_data[(name, g)] = (qo, lo, hi, md)
    cells = []
    for qq in range(5):
        for g in [1, 0]:
            m = (D == g) & (qint == qq); mm = mean_cell[:, qq, g]
            cells.append({"quintile": qq + 1, "eligible": g, "n": int(m.sum()), "obs_mean": float(Y[m].mean()), "rep_mean": float(mm.mean()), "rep_lo": float(np.quantile(mm, .025)), "rep_hi": float(np.quantile(mm, .975)),
                          "obs_in_range": bool(np.quantile(mm, .025) <= Y[m].mean() <= np.quantile(mm, .975))})
    res["mean_by_quintile_eligibility"] = cells; res["share_cells_obs_in_95"] = float(np.mean([c["obs_in_range"] for c in cells]))
    out[name] = res; fig_data[(name, "cells")] = cells
    print(name, "checked", round(time.time() - t0, 1), flush=True)
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open("results/res_k401_ppc.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.6))
for ax, name, ttl in [(axs[0], "gaussian", "Gaussian errors"), (axs[1], "student_t", "Student $t$ errors, 3 df")]:
    for g, col, lab in [(1, "C3", "eligible"), (0, "C0", "ineligible")]:
        qo, lo, hi, md = fig_data[(name, g)]; ax.fill_between(qo, lo, hi, color=col, alpha=0.3, label=lab)
    qo = fig_data[(name, "all")][0] if (name, "all") in fig_data else fig_data[(name, 1)][0]
    lim = [min(fig_data[(name, 1)][0].min(), fig_data[(name, 0)][0].min()), max(fig_data[(name, 1)][0].max(), fig_data[(name, 0)][0].max())]
    ax.plot(lim, lim, "k--", lw=0.6); ax.set_xscale("symlog"); ax.set_yscale("symlog"); ax.set_title(ttl, fontsize=8); ax.set_xlabel("Observed quantile, $000")
axs[0].set_ylabel("Replicated quantile, $000"); axs[0].legend(frameon=False, fontsize=6)
ax = axs[2]
for j, (name, col, mk) in enumerate([("gaussian", "C1", "s"), ("student_t", "C2", "^")]):
    for g, ls in [(1, "-"), (0, ":")]:
        cs = [c for c in fig_data[(name, "cells")] if c["eligible"] == g]; x = np.arange(1, 6) + (j - 0.5) * 0.18 + (0.0 if g == 1 else 0.0)
        ax.errorbar(x, [c["rep_mean"] for c in cs], yerr=[[c["rep_mean"] - c["rep_lo"] for c in cs], [c["rep_hi"] - c["rep_mean"] for c in cs]], fmt=mk, color=col, ms=2.5, lw=0.6, capsize=0, alpha=0.9 if g == 1 else 0.5)
cs1 = [c for c in fig_data[("gaussian", "cells")] if c["eligible"] == 1]; cs0 = [c for c in fig_data[("gaussian", "cells")] if c["eligible"] == 0]
ax.plot(range(1, 6), [c["obs_mean"] for c in cs1], "ko-", ms=3, lw=0.6, label="observed, eligible"); ax.plot(range(1, 6), [c["obs_mean"] for c in cs0], "k^--", ms=3, lw=0.6, mfc="none", label="observed, ineligible")
ax.set_xlabel("Income quintile"); ax.set_ylabel("Mean assets, $000"); ax.set_title("Means by cell", fontsize=8); ax.legend(frameon=False, fontsize=5.5)
fig.tight_layout(); fig.savefig("figs/ppc401k_post.pdf"); plt.close()
print("done", out["seconds"])
