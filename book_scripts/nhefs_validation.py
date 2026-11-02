"""Chapter 17 (NHEFS, quitting smoking and weight gain), item F6 of round 1 (October 2026): a posterior predictive check and a coverage simulation for the estimators of Table tab:nhefs.
Run from the repository root with PYTHONPATH=. (needs stochtree and scikit-learn). Three modes:
  python book_scripts/nhefs_validation.py ppc           writes results/res_nhefs_ppc.json and figs/nhefs_ppc.pdf
  python book_scripts/nhefs_validation.py sim TASK [R]   one task of the simulation, R replications of one cell (default 20), writes results/nhefs_val/sim_taskTTT.json; tasks 0 to 99, cell = TASK // 25, replications (TASK % 25) * R to (TASK % 25 + 1) * R
  python book_scripts/nhefs_validation.py merge         combines the tasks into results/res_nhefs_validation.json

Data and estimators are those of apps2.py (n = 1,566 smokers of the Hernan and Robins file, nine covariates, the standardized basis of the four continuous covariates, their squares and dummies; weight change 1971 to 1982 in kilograms):
least squares with the Hernan and Robins covariates (classical interval), DML with five fold cross fitting and forests of 500 trees (minimum leaf 5, propensity clipped to [0.01, 0.99], normal interval), the coupled horseshoe with effect modifiers (3,000 Gibbs sweeps, 1,000 dropped),
BART with treatment as a covariate (20 warm start sweeps, 1,000 MCMC draws) and the Bayesian causal forest with the logistic propensity (the same). The posterior interval of the Bayesian methods is the 2.5% and 97.5% quantile of the average effect over the 1,566 units.

Posterior predictive check (ppc): one fit of the horseshoe and of the Bayesian causal forest to the observed data; for each of the 1,000 posterior draws one replicated outcome vector at the observed covariates and treatment, the draw's regression function plus Gaussian error at the draw's error variance (the error variance draws are kept).
Reported by arm (quit, did not quit): the observed quantiles at levels 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99 against the median and the 95% range of the replicated quantiles, and the posterior predictive p value (the share of replicates whose statistic exceeds the observed) of the mean, standard deviation, skewness, minimum and maximum.

Coverage simulation (sim): the 1,566 covariate vectors are fixed. Treatment is redrawn as Bernoulli with the fitted logistic propensities of the real data (standardized basis, no penalty). The control mean is the covariate part of the additive least squares regression of weight change on treatment and the basis (intercept and basis coefficients as fitted). The effect of unit i is tau_i, either constant 3.5 kg ("const") or 3.5 + (h_i - mean h)/sd h with h = sin(z_age) + z_intensity^2, z the standardized age and cigarettes per day ("het", sample mean 3.5, standard deviation 1 kg).
The error is 7.3 Z ("gauss") or 7.3 t_4 / sqrt(2) ("t4"; the same standard deviation, heavier tails). The target is the sample average effect, 3.5 in all four cells. Four cells (const/gauss, const/t4, het/gauss, het/t4), 500 replications each; each replication draws treatment and errors from default_rng([20261023, cell, replication]) and fits all five estimators (the propensity of the horseshoe and the forest is refitted on the replicate treatment; sampler seeds are fixed functions of the replication).
Reported by cell and method with Monte Carlo standard errors: bias, RMSE, 95% coverage and mean interval width.
"""
import sys, os, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from causaldata import nhefs_complete
from stochtree import BARTModel, BCFModel
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold

d = nhefs_complete.load_pandas().data
T0 = d.qsmk.values.astype(float); Y0 = d.wt82_71.values.astype(float); n = len(Y0)
num = ["age", "smokeintensity", "smokeyrs", "wt71"]; cat = ["sex", "race", "education", "exercise", "active"]
D = pd.get_dummies(d[cat].astype(int).astype(str), drop_first=True).astype(float)
Xn = d[num].astype(float)
Xhr = np.column_stack([Xn.values, Xn.values ** 2, D.values]); Xraw = np.column_stack([Xn.values, d[cat].astype(float).values])
Zs = (Xhr - Xhr.mean(0)) / Xhr.std(0)

def propensity(T):
    return LogisticRegression(C=1e6, max_iter=5000).fit(Zs, T).predict_proba(Zs)[:, 1]

def hs(Z, y, unpen, rng, iters=3000, burn=1000):
    nn, q = Z.shape; pen = np.ones(q, bool); pen[list(unpen)] = False
    ZtZ = Z.T @ Z; Zty = Z.T @ y; yty = y @ y
    lam2 = np.ones(q); nu = np.ones(q); tau2 = 1.0; xi = 1.0; s2 = np.var(y); dr = []; ss = []
    for it in range(iters):
        A = ZtZ + np.diag(np.where(pen, 1 / (lam2 * tau2), 1e-8)); L = np.linalg.cholesky(A)
        b = np.linalg.solve(A, Zty) + np.sqrt(s2) * np.linalg.solve(L.T, rng.normal(size=q))
        rss = yty - 2 * b @ Zty + b @ ZtZ @ b; bp = b[pen]
        s2 = 1 / rng.gamma((nn + pen.sum()) / 2, 1 / (rss / 2 + (bp ** 2 / (lam2[pen] * tau2)).sum() / 2))
        lam2[pen] = 1 / rng.gamma(1, 1 / (1 / nu[pen] + bp ** 2 / (2 * tau2 * s2))); nu[pen] = 1 / rng.gamma(1, 1 / (1 + 1 / lam2[pen]))
        tau2 = 1 / rng.gamma((pen.sum() + 1) / 2, 1 / (1 / xi + (bp ** 2 / lam2[pen]).sum() / (2 * s2))); xi = 1 / rng.gamma(1, 1 / (1 + 1 / tau2))
        if it >= burn: dr.append(b.copy()); ss.append(s2)
    return np.array(dr), np.array(ss)

def fit_hs(Y, T, e, rng):
    inter = Zs[:, :4] * T[:, None]
    Z = np.column_stack([T, np.ones(n), e, inter, Zs]); dr, ss = hs(Z, Y, {0, 1, 2}, rng)
    tau_i = dr[:, [0]] + dr[:, 3:7] @ Zs[:, :4].T
    return tau_i.mean(1), Z, dr, ss

def fit_bart(Y, T, seed):
    Xa = np.column_stack([Xraw, T]); m = BARTModel()
    m.sample(X_train=Xa, y_train=Y, num_gfr=20, num_mcmc=1000, general_params={"random_seed": seed})
    X1 = Xa.copy(); X1[:, -1] = 1; X0 = Xa.copy(); X0[:, -1] = 0
    f1 = m.predict(X1); f0 = m.predict(X0)
    f1 = f1["y_hat"] if isinstance(f1, dict) else f1; f0 = f0["y_hat"] if isinstance(f0, dict) else f0
    return (f1 - f0).mean(0)

def fit_bcf(Y, T, e, seed):
    b = BCFModel(); b.sample(X_train=Xraw, Z_train=T, y_train=Y, propensity_train=e, num_gfr=20, num_mcmc=1000, general_params={"random_seed": seed})
    return b.tau_hat_train.mean(0), b

def fit_dml(Y, T, seed=1):
    kf = KFold(5, shuffle=True, random_state=seed); m1 = np.zeros(n); m0 = np.zeros(n); eh = np.zeros(n)
    for tr, te in kf.split(Xraw):
        for t, arr in [(1, m1), (0, m0)]:
            s = tr[T[tr] == t]; arr[te] = RandomForestRegressor(500, min_samples_leaf=5, random_state=1).fit(Xraw[s], Y[s]).predict(Xraw[te])
        eh[te] = RandomForestClassifier(500, min_samples_leaf=5, random_state=1).fit(Xraw[tr], T[tr]).predict_proba(Xraw[te])[:, 1]
    eh = np.clip(eh, 0.01, 0.99)
    psi = m1 - m0 + T * (Y - m1) / eh - (1 - T) * (Y - m0) / (1 - eh)
    se = psi.std() / np.sqrt(n); return float(psi.mean()), float(psi.mean() - 1.96 * se), float(psi.mean() + 1.96 * se)

def fit_ols(Y, T):
    X1 = np.column_stack([np.ones(n), T, Xhr]); b1 = np.linalg.lstsq(X1, Y, rcond=None)[0]; r1 = Y - X1 @ b1
    se = float(np.sqrt(r1 @ r1 / (n - X1.shape[1]) * np.linalg.inv(X1.T @ X1)[1, 1]))
    return float(b1[1]), float(b1[1] - 1.96 * se), float(b1[1] + 1.96 * se)

def ci(x): return [float(np.mean(x)), float(np.quantile(x, .025)), float(np.quantile(x, .975))]
METHODS = ["ols", "dml", "horseshoe", "bart", "bcf"]
CELLS = [("const", "gauss"), ("const", "t4"), ("het", "gauss"), ("het", "t4")]

def ppc():
    t0 = time.time(); rng = np.random.default_rng(20261024); e = propensity(T0)
    tau, Z, dr, ss = fit_hs(Y0, T0, e, rng); print("hs", tau.mean(), time.time() - t0, flush=True)
    mean_hs = dr @ Z.T                                             # draws x n
    tb, b = fit_bcf(Y0, T0, e, 3); yh = np.asarray(b.y_hat_train).T; s2b = np.asarray(b.global_var_samples)
    print("bcf", tb.mean(), yh.shape, yh.mean(), Y0.mean(), np.sqrt(s2b.mean()), time.time() - t0, flush=True)
    levels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    def skew(v): return float(((v - v.mean()) ** 3).mean() / v.std() ** 3)
    stats = {"mean": np.mean, "sd": np.std, "skewness": skew, "min": np.min, "max": np.max}
    out = {"n": int(n), "levels": levels, "resid_sd_hs": float(np.sqrt(ss.mean())), "resid_sd_bcf": float(np.sqrt(s2b.mean())), "draws": int(len(ss))}
    figdata = {}
    for name, mu, s2 in [("horseshoe", mean_hs, ss), ("bcf", yh, s2b)]:
        rep = mu + np.sqrt(s2)[:, None] * rng.normal(size=mu.shape)
        o = {}
        for arm in [1, 0]:
            msk = T0 == arm; y = Y0[msk]; r = rep[:, msk]
            qo = np.quantile(y, levels); qr = np.quantile(r, levels, axis=1)           # levels x draws
            o["quit" if arm else "noquit"] = {
                "n": int(msk.sum()), "obs_q": qo.tolist(), "rep_median": np.median(qr, 1).tolist(), "rep_lo": np.quantile(qr, .025, 1).tolist(), "rep_hi": np.quantile(qr, .975, 1).tolist(),
                "obs_outside_95": [float(l) for l, a, lo, hi in zip(levels, qo, np.quantile(qr, .025, 1), np.quantile(qr, .975, 1)) if a < lo or a > hi],
                "stats": {k: {"obs": float(f(y)), "rep_mean": float(np.mean([f(r[i]) for i in range(len(r))])), "ppp": float(np.mean([f(r[i]) >= f(y) for i in range(len(r))]))} for k, f in stats.items()}}
            figdata[(name, arm)] = (qo, np.quantile(qr, .025, 1), np.quantile(qr, .975, 1), np.median(qr, 1))
        out[name] = o
    json.dump(out, open("results/res_nhefs_ppc.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.9), sharex=True, sharey=True)
    for ax, name, ttl in [(axs[0], "horseshoe", "Horseshoe"), (axs[1], "bcf", "Bayesian causal forest")]:
        for arm, col, lab in [(1, "C3", "quit"), (0, "C0", "did not quit")]:
            qo, lo, hi, md = figdata[(name, arm)]
            ax.fill_between(qo, lo, hi, color=col, alpha=0.25); ax.plot(qo, md, color=col, lw=0.8, label=lab)
        ax.plot([-30, 40], [-30, 40], "k--", lw=0.6); ax.set_title(ttl, fontsize=9); ax.set_xlabel("Observed quantile, kg")
    axs[0].set_ylabel("Replicated quantile, kg"); axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig("figs/nhefs_ppc.pdf"); plt.close()
    print(json.dumps({k: out[k] for k in ["resid_sd_hs", "resid_sd_bcf"]}), "done", time.time() - t0)

def sim(task, R=20):
    cell = task // 25; k0 = (task % 25) * R; eff, err = CELLS[cell]
    os.makedirs("results/nhefs_val", exist_ok=True); t0 = time.time()
    e_true = propensity(T0)
    X1 = np.column_stack([np.ones(n), T0, Xhr]); bb = np.linalg.lstsq(X1, Y0, rcond=None)[0]
    mu0 = bb[0] + Xhr @ bb[2:]
    zage = (d.age.values - d.age.values.mean()) / d.age.values.std(); zint = (d.smokeintensity.values - d.smokeintensity.values.mean()) / d.smokeintensity.values.std()
    h = np.sin(zage) + zint ** 2
    tau_i = np.full(n, 3.5) if eff == "const" else 3.5 + (h - h.mean()) / h.std()
    truth = float(tau_i.mean()); res = {m: [] for m in METHODS}; reps = []
    for k in range(k0, k0 + R):
        rng = np.random.default_rng([20261023, cell, k])
        T = (rng.random(n) < e_true).astype(float)
        z = rng.normal(size=n) if err == "gauss" else rng.standard_t(4, size=n) / np.sqrt(2)
        Y = mu0 + tau_i * T + 7.3 * z; e = propensity(T)
        res["ols"].append(fit_ols(Y, T)); res["dml"].append(fit_dml(Y, T))
        res["horseshoe"].append(ci(fit_hs(Y, T, e, rng)[0])); res["bart"].append(ci(fit_bart(Y, T, 3 + k)))
        res["bcf"].append(ci(fit_bcf(Y, T, e, 3 + k)[0])); reps.append(k)
        print("task", task, "rep", k, round(time.time() - t0, 1), {m: round(res[m][-1][0], 2) for m in METHODS}, flush=True)
    json.dump({"cell": cell, "effect": eff, "error": err, "truth": truth, "reps": reps, "est": res, "seconds": time.time() - t0}, open(f"results/nhefs_val/sim_task{task:03d}.json", "w"))

def merge():
    import glob
    out = {"design": "4 cells, replications per cell as listed, truth 3.5", "cells": {}}
    for cell, (eff, err) in enumerate(CELLS):
        files = sorted(glob.glob(f"results/nhefs_val/sim_task{cell * 25:03d}.json") + [f for t in range(cell * 25 + 1, cell * 25 + 25) for f in glob.glob(f"results/nhefs_val/sim_task{t:03d}.json")])
        parts = [json.load(open(f)) for f in files]; truth = parts[0]["truth"]; c = {"effect": eff, "error": err, "truth": truth, "reps": int(sum(len(p["reps"]) for p in parts)), "methods": {}}
        for m in METHODS:
            a = np.array([r for p in parts for r in p["est"][m]]); est, lo, hi = a[:, 0], a[:, 1], a[:, 2]; R = len(est); err_ = est - truth
            cov = ((lo <= truth) & (truth <= hi)).astype(float); w = hi - lo; rmse = float(np.sqrt((err_ ** 2).mean()))
            c["methods"][m] = {"bias": float(err_.mean()), "bias_se": float(err_.std(ddof=1) / np.sqrt(R)), "rmse": rmse, "rmse_se": float(np.std(err_ ** 2, ddof=1) / np.sqrt(R) / (2 * rmse)),
                               "coverage": float(cov.mean()), "coverage_se": float(np.sqrt(cov.mean() * (1 - cov.mean()) / R)), "width": float(w.mean()), "width_se": float(w.std(ddof=1) / np.sqrt(R)), "sd_est": float(est.std(ddof=1))}
        out["cells"][f"{eff}_{err}"] = c
    json.dump(out, open("results/res_nhefs_validation.json", "w"), indent=1)
    for k, c in out["cells"].items():
        print(k, c["reps"])
        for m, v in c["methods"].items(): print("  ", m, {a: round(b, 3) for a, b in v.items()})

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ppc": ppc()
    elif mode == "sim": sim(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    else: merge()
