"""Chapter 17: NHEFS (quitting smoking and weight gain) and Thornton (incentives and learning HIV results).
Run from the repository root with PYTHONPATH=. (needs stochtree and polyagamma); writes results/res_apps2.json and figs/nhefs_ate.pdf, nhefs_het.pdf, thornton.pdf.
Every random number comes from the seeded generator rng, the seeded BART and BCF fits and the seeded forests, so a rerun reproduces the JSON.
Also saved: the classical interval of least squares, the descriptives of quitters and nonquitters, the propensity and weight diagnostics, the Thornton descriptives,
and the sizes of the groups of the heterogeneity figure (four groups of cigarettes per day and quintiles of age, non overlapping).
"""
import numpy as np, pandas as pd, json, time, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from causaldata import nhefs_complete, thornton_hiv
from stochtree import BARTModel, BCFModel
from polyagamma import random_polyagamma
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
FIG = "figs/"; R = {}; t0 = time.time()
rng = np.random.default_rng(1971)
def summ(x): return [float(np.mean(x)), float(np.std(x)), float(np.quantile(x, .025)), float(np.quantile(x, .975))]

# ---------- NHEFS: quitting smoking and weight gain ----------
d = nhefs_complete.load_pandas().data
T = d.qsmk.values.astype(float); Y = d.wt82_71.values.astype(float)
num = ["age", "smokeintensity", "smokeyrs", "wt71"]
cat = ["sex", "race", "education", "exercise", "active"]
D = pd.get_dummies(d[cat].astype(int).astype(str), drop_first=True).astype(float)
Xn = d[num].astype(float)
Xhr = np.column_stack([Xn.values, Xn.values ** 2, D.values])        # Hernan Robins specification
Xraw = np.column_stack([Xn.values, d[cat].astype(float).values])     # for trees
R["n"] = int(len(Y)); R["n_quit"] = int(T.sum())
R["naive"] = float(Y[T == 1].mean() - Y[T == 0].mean())
R["ols"] = float(LinearRegression().fit(np.column_stack([T, Xhr]), Y).coef_[0])
X1 = np.column_stack([np.ones(len(Y)), T, Xhr]); b1 = np.linalg.lstsq(X1, Y, rcond=None)[0]; r1 = Y - X1 @ b1
ols_se = float(np.sqrt(r1 @ r1 / (len(Y) - X1.shape[1]) * np.linalg.inv(X1.T @ X1)[1, 1]))
R["ols_interval"] = [float(b1[1] - 1.96 * ols_se), float(b1[1] + 1.96 * ols_se)]; R["ols_se"] = ols_se
R["desc"] = {nm: [float(d[nm][T == 1].mean()), float(d[nm][T == 0].mean())] for nm in ["age", "smokeintensity", "smokeyrs", "wt71", "wt82_71"]}
e = LogisticRegression(C=1e6, max_iter=5000).fit((Xhr - Xhr.mean(0)) / Xhr.std(0), T).predict_proba((Xhr - Xhr.mean(0)) / Xhr.std(0))[:, 1]
w = np.where(T == 1, T.mean() / e, (1 - T.mean()) / (1 - e))
R["prop_range"] = [float(e.min()), float(e.max())]; R["max_weight"] = float(w.max()); R["ess_weights"] = float(w.sum() ** 2 / (w ** 2).sum())
R["ipw"] = float(np.average(Y[T == 1], weights=w[T == 1]) - np.average(Y[T == 0], weights=w[T == 0]))
# horseshoe coupled
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
Zs = (Xhr - Xhr.mean(0)) / Xhr.std(0)
inter = Zs[:, :4] * T[:, None]                       # effect modification by the four continuous covariates
Z = np.column_stack([T, np.ones(len(Y)), e, inter, Zs])
dr = hs(Z, Y, {0, 1, 2})
tau_i = dr[:, [0]] + dr[:, 3:7] @ Zs[:, :4].T       # draws x units
R["hs_ate"] = summ(tau_i.mean(1)); R["hs_att"] = summ(tau_i[:, T == 1].mean(1)); R["hs_atu"] = summ(tau_i[:, T == 0].mean(1))
print("nhefs hs", R["hs_ate"], time.time() - t0, flush=True)
# BART (Hill) and BCF
Xa = np.column_stack([Xraw, T]); m = BARTModel()
m.sample(X_train=Xa, y_train=Y, num_gfr=20, num_mcmc=1000, general_params={"random_seed": 3})
X1 = Xa.copy(); X1[:, -1] = 1; X0 = Xa.copy(); X0[:, -1] = 0
f1 = m.predict(X1); f0 = m.predict(X0)
f1 = f1["y_hat"] if isinstance(f1, dict) else f1; f0 = f0["y_hat"] if isinstance(f0, dict) else f0
R["bart_ate"] = summ((f1 - f0).mean(0))
b = BCFModel(); b.sample(X_train=Xraw, Z_train=T, y_train=Y, propensity_train=e, num_gfr=20, num_mcmc=1000, general_params={"random_seed": 3})
tb = b.tau_hat_train
R["bcf_ate"] = summ(tb.mean(0)); R["bcf_att"] = summ(tb[T == 1].mean(0)); R["bcf_atu"] = summ(tb[T == 0].mean(0))
# DML AIPW
n = len(Y); kf = KFold(5, shuffle=True, random_state=1); m1 = np.zeros(n); m0 = np.zeros(n); eh = np.zeros(n)
for tr, te in kf.split(Xraw):
    for t, arr in [(1, m1), (0, m0)]:
        s = tr[T[tr] == t]; arr[te] = RandomForestRegressor(500, min_samples_leaf=5, random_state=1).fit(Xraw[s], Y[s]).predict(Xraw[te])
    eh[te] = RandomForestClassifier(500, min_samples_leaf=5, random_state=1).fit(Xraw[tr], T[tr]).predict_proba(Xraw[te])[:, 1]
eh = np.clip(eh, 0.01, 0.99)
psi = m1 - m0 + T * (Y - m1) / eh - (1 - T) * (Y - m0) / (1 - eh)
se = psi.std() / np.sqrt(n); R["dml_ate"] = [float(psi.mean()), float(se), float(psi.mean() - 1.96 * se), float(psi.mean() + 1.96 * se)]
print("nhefs bart bcf dml", R["bart_ate"], R["bcf_ate"], R["dml_ate"], time.time() - t0, flush=True)
# heterogeneity figure: BCF tau by smoking intensity and age
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.9))
cig = d.smokeintensity.values
q5 = pd.qcut(d.age.values, 5, labels=False)
groups = {"cig": (cig, [cig <= 10, (cig > 10) & (cig < 20), cig == 20, cig > 20], ["1 to 10", "11 to 19", "20", "21 or more"], "Cigarettes per day, 1971"),
          "age": (d.age.values, [q5 == k for k in range(5)], [f"{int(d.age.values[q5 == k].min())} to {int(d.age.values[q5 == k].max())}" for k in range(5)], "Age, 1971")}
for ax, key in [(axs[0], "cig"), (axs[1], "age")]:
    v, masks, names, lab = groups[key]; mids, mm, lo, hi, sizes = [], [], [], [], []
    for s in masks:
        dd = tb[s].mean(0)
        mids.append(np.median(v[s])); mm.append(dd.mean()); lo.append(np.quantile(dd, .025)); hi.append(np.quantile(dd, .975)); sizes.append(int(s.sum()))
    ax.errorbar(mids, mm, yerr=[np.array(mm) - lo, np.array(hi) - mm], fmt="o", capsize=3); ax.set_xlabel(lab)
    R["bcf_by_" + key] = dict(groups=names, n=sizes, median=list(map(float, mids)), mean=list(map(float, mm)), lo=list(map(float, lo)), hi=list(map(float, hi)))
axs[0].set_ylabel("Effect of quitting, kg")
fig.tight_layout(); fig.savefig(FIG + "nhefs_het.pdf"); plt.close()
# summary figure
labs = ["naive", "least squares", "IPW", "DML (AIPW)", "horseshoe", "BART", "BCF"]
vv = [[R["naive"]] * 3, [R["ols"]] + R["ols_interval"], [R["ipw"]] * 3, R["dml_ate"][0:1] + R["dml_ate"][2:], R["hs_ate"][0:1] + R["hs_ate"][2:], R["bart_ate"][0:1] + R["bart_ate"][2:], R["bcf_ate"][0:1] + R["bcf_ate"][2:]]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for i, v in enumerate(vv): ax.errorbar(v[0], i, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", capsize=3, color="C0")
ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs); ax.set_xlabel("Average effect of quitting on weight gain, kg")
fig.tight_layout(); fig.savefig(FIG + "nhefs_ate.pdf"); plt.close()

# ---------- Thornton: incentives and learning HIV results ----------
h = thornton_hiv.load_pandas().data.dropna(subset=["got", "any", "distvct", "age", "tinc"])
h = h[h.hiv2004 >= 0] if "hiv2004" in h else h
yv = h.got.values.astype(float); av = h["any"].values.astype(float); dist = h.distvct.values.astype(float); age = h.age.values.astype(float); tinc = h.tinc.values.astype(float)
R["th_n"] = int(len(yv)); R["th_raw"] = [float(yv[av == 1].mean()), float(yv[av == 0].mean())]
R["th_desc"] = dict(distinct_amounts=int(len(np.unique(tinc))), max_amount=float(tinc.max()), n_none=int((tinc == 0).sum()), share_none=float((tinc == 0).mean()))
dz = (dist - dist.mean()) / dist.std(); az = (age - age.mean()) / age.std()
def pg_logit(Zm, y, iters=3000, burn=1000, v0=10.0):
    nn, q = Zm.shape; b = np.zeros(q); kap = y - 0.5; P0 = np.eye(q) / v0; out = []
    for it in range(iters):
        om = random_polyagamma(1, Zm @ b, random_state=rng)
        V = np.linalg.inv(Zm.T @ (Zm * om[:, None]) + P0); V = (V + V.T) / 2
        b = rng.multivariate_normal(V @ (Zm.T @ kap), V)
        if it >= burn: out.append(b.copy())
    return np.array(out)
Zt = np.column_stack([np.ones(len(yv)), av, dz, az, av * dz])
B = pg_logit(Zt, yv)
expit = lambda z: 1 / (1 + np.exp(-z))
Z1 = Zt.copy(); Z1[:, 1] = 1; Z1[:, 4] = dz; Z0 = Zt.copy(); Z0[:, 1] = 0; Z0[:, 4] = 0
p1 = expit(B @ Z1.T); p0 = expit(B @ Z0.T)
rd = (p1 - p0).mean(1); m1_, m0_ = p1.mean(1), p0.mean(1)
mor = (m1_ / (1 - m1_)) / (m0_ / (1 - m0_)); cor = np.exp(B[:, 1])
R["th_rd"] = summ(rd); R["th_cond_or"] = summ(cor); R["th_marg_or"] = summ(mor)
far = dist > np.quantile(dist, 0.75); near = dist < np.quantile(dist, 0.25)
R["th_rd_near"] = summ((p1 - p0)[:, near].mean(1)); R["th_rd_far"] = summ((p1 - p0)[:, far].mean(1))
R["th_delta"] = summ(B[:, 4]); R["th_delta_ppos"] = float((B[:, 4] > 0).mean()); R["th_gap"] = summ(cor - mor)
# dose response in incentive amount among all, logistic with spline in tinc
Zd = np.column_stack([np.ones(len(yv)), av, tinc, np.maximum(tinc - 1, 0), dz, az])
Bd = pg_logit(Zd, yv)
grid = np.linspace(0, tinc.max(), 31); curves = []
for g in grid:
    Zg = Zd.copy(); Zg[:, 1] = float(g > 0); Zg[:, 2] = g; Zg[:, 3] = max(g - 1, 0)
    curves.append(expit(Bd @ Zg.T).mean(1))
C = np.array(curves)
R["th_dose"] = {"grid": grid.tolist(), "mean": C.mean(1).tolist(), "lo": np.quantile(C, .025, 1).tolist(), "hi": np.quantile(C, .975, 1).tolist()}
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.9))
axs[0].fill_between(grid, R["th_dose"]["lo"], R["th_dose"]["hi"], alpha=0.3); axs[0].plot(grid, R["th_dose"]["mean"])
axs[0].set_xlabel("Incentive amount (tinc, as coded in the data)"); axs[0].set_ylabel("P(learned HIV result)")
axs[1].hist(cor, bins=40, density=True, histtype="step", label="conditional OR"); axs[1].hist(mor, bins=40, density=True, histtype="step", label="marginal OR")
axs[1].set_xlabel("Odds ratio for any incentive"); axs[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIG + "thornton.pdf"); plt.close()
# the Thornton model without the treatment by distance term, for the gap between the conditional and marginal odds ratios (run last, so the
# draws above are unchanged)
Zn = Zt[:, :4]; Bn = pg_logit(Zn, yv)
Z1n = Zn.copy(); Z1n[:, 1] = 1; Z0n = Zn.copy(); Z0n[:, 1] = 0
m1n, m0n = expit(Bn @ Z1n.T).mean(1), expit(Bn @ Z0n.T).mean(1); morn = (m1n / (1 - m1n)) / (m0n / (1 - m0n))
R["th_nointer_cond_or"] = summ(np.exp(Bn[:, 1])); R["th_nointer_marg_or"] = summ(morn); R["th_nointer_gap"] = summ(np.exp(Bn[:, 1]) - morn)
json.dump(R, open("results/res_apps2.json", "w"), indent=1); print(json.dumps(R)[:3000]); print("done", time.time() - t0)
