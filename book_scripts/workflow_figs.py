import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(5)
d = pd.read_stata("data/sipp1991.dta")
xv = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]
Y = d.net_tfa.values / 1000; D = d.e401.values.astype(float)
X = StandardScaler().fit_transform(d[xv].values.astype(float))
P = PolynomialFeatures(2, include_bias=False).fit_transform(X)
Z = np.column_stack([D, P]); lr = LinearRegression().fit(Z, Y); fit = lr.predict(Z); s = np.std(Y - fit)
qs = np.linspace(0.01, 0.99, 99); obs = np.quantile(Y, qs)
reps = np.array([np.quantile(fit + rng.normal(0, s, len(Y)), qs) for _ in range(200)])
# Student t errors with 3 df, scaled to match the median absolute residual
res = Y - fit; sc = np.median(np.abs(res)) / 0.765
rept = np.array([np.quantile(fit + sc * rng.standard_t(3, len(Y)), qs) for _ in range(200)])
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.9))
for ax, rr, lab in [(axs[0], reps, "Gaussian errors"), (axs[1], rept, "Student t errors, 3 df")]:
    ax.fill_between(obs, np.quantile(rr, .025, 0), np.quantile(rr, .975, 0), alpha=0.3)
    ax.plot(obs, obs, "k--", lw=0.7); ax.set_xscale("symlog"); ax.set_yscale("symlog")
    ax.set_xlabel("Observed quantile, $000"); ax.set_title(lab, fontsize=9)
axs[0].set_ylabel("Replicated quantile, $000")
fig.tight_layout(); fig.savefig("figs/ppc401k.pdf"); plt.close()
R = {"resid_sd": float(s), "obs_q": [float(np.quantile(Y, q)) for q in [0.05, 0.5, 0.95, 0.99]],
     "rep_gauss_q": [float(np.median(np.quantile(fit + rng.normal(0, s, len(Y)), [q]))) for q in [0.05, 0.5, 0.95, 0.99]]}
# LaLonde Bayesian sensitivity using the BART ATT posterior (mean 1.54, sd 0.33)
scales = np.linspace(0.01, 1.5, 60); probs = []
for sc_ in scales:
    a = rng.normal(0, sc_, 200000); b = rng.normal(0, sc_, 200000)
    probs.append(np.mean(rng.normal(1.54, 0.333, 200000) - a * b > 0))
R["lalonde_sens"] = [[float(scales[i]), float(probs[i])] for i in [0, 20, 40, 59]]
fig, ax = plt.subplots(figsize=(5, 3.1)); ax.plot(scales, probs)
ax.set_xlabel("Prior scale of each confounding parameter ($000 units)"); ax.set_ylabel("Posterior P(effect > 0)")
fig.tight_layout(); fig.savefig("figs/lalonde_sens.pdf")
json.dump(R, open("results/res_workflow.json", "w"), indent=1); print(R)
