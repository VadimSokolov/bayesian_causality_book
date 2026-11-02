import numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LassoCV
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
OUT = "figs/"
R = {}
rng = np.random.default_rng(2002)
expit = lambda z: 1 / (1 + np.exp(-z))

# A. One posterior, several accountings: tau_Q for different covariate laws Q
n = 1500
age = rng.normal(0, 1, n)
t = rng.binomial(1, expit(0.8 * age))
y = rng.binomial(1, expit(-0.5 + 0.8 * t + 0.6 * age - 0.7 * t * age))
def fit(w):
    Z = np.column_stack([t, age, t * age])
    m = LogisticRegression(C=1e6, max_iter=2000).fit(Z, y, sample_weight=w)
    return m.intercept_[0], m.coef_[0]
def tauQ(a, b, xs):
    return np.mean(expit(a + b[0] + (b[1] + b[2]) * xs) - expit(a + b[1] * xs))
Qs = {"sample": age, "treated": age[t == 1], "older cohort": rng.normal(1, 1, 5000)}
draws = {k: [] for k in Qs}
for s in range(400):
    w = rng.exponential(size=n); w = w / w.mean()
    a, b = fit(w)
    for k, xs in Qs.items():
        draws[k].append(tauQ(a, b, xs))
fig, ax = plt.subplots(figsize=(5, 3.2))
# Gray scale print: one outline style and gray level per covariate law (solid black for the sample)
for k, ls, col in zip(Qs, ["-", "--", "-."], ["k", "0.3", "0.55"]):
    ax.hist(draws[k], bins=40, density=True, histtype="step", color=col, ls=ls, lw=1.5, label=k)
ax.set_xlabel("Average treatment effect, risk difference"); ax.set_ylabel("Posterior density")
ax.legend(frameon=False, handlelength=3)
fig.tight_layout(); fig.savefig(OUT + "tauQ.pdf"); plt.close()
R["tauQ"] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in draws.items()}
import sys
if "--figs-only" in sys.argv:    # redraw tauQ.pdf only (section A is the first user of the seed 2002 stream, so it reproduces the posterior exactly); results/results2.json is not written, and the means and sds are compared with the saved ones
    saved = json.load(open("results/results2.json")).get("tauQ", {})
    print("tauQ: largest difference from results/results2.json:", max(abs(a - b) for k in saved for a, b in zip(R["tauQ"][k], saved[k])) if saved else "no saved value"); sys.exit(0)

# B. Paired versus unpaired designs
icc = np.linspace(0, 0.9999, 400)   # the paired variance is zero at 1, which a log axis cannot show, so the curve stops just short of it
# variance of mean difference: unpaired with N children: 4 sigma2_tot / N ; paired with m pairs: 2 sigma2_tot (1-icc)/m
N = 20000; m = 50
var_unpaired = 4 / N
var_paired = 2 * (1 - icc) / m
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.semilogy(icc, var_paired, label="50 twin pairs")
ax.axhline(var_unpaired, color="k", ls="--", label="20,000 unpaired, ideal randomization")
ax.set_xlim(0, 1); ax.set_xlabel("Within pair correlation of growth"); ax.set_ylabel("Variance of estimate (total variance 1)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "twins.pdf"); plt.close()

# C. Bayesian sensitivity: posterior probability of positive effect under priors on confounding
tau_hat, se = 0.30, 0.05
scales = np.linspace(0.01, 1.0, 60)
probs = []
for s in scales:
    a = rng.normal(0, s, 200000); bU = rng.normal(0, s, 200000)
    tau = rng.normal(tau_hat, se, 200000) - a * bU
    probs.append(np.mean(tau > 0))
R["sens"] = dict(scales=[float(scales[i]) for i in [0, 19, 39, 59]], probs=[float(probs[i]) for i in [0, 19, 39, 59]])
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot(scales, probs)
ax.set_xlabel("Prior scale of each confounding parameter"); ax.set_ylabel("Posterior P(effect > 0)")
fig.tight_layout(); fig.savefig(OUT + "sensitivity.pdf"); plt.close()

# D. Shared local scales versus independent horseshoe, moderate confounders
def gibbs_shared(y, t, X, iters=1500, burn=500, shared=True):
    n, p = X.shape
    Zy = np.column_stack([t, X])
    lam2 = np.ones(p); nu = np.ones(p)
    lam2t = np.ones(p); nut = np.ones(p)
    ty2 = tt2 = 1.0; xy = xt = 1.0; sy2 = st2 = 1.0
    ZtZ = Zy.T @ Zy; Zty = Zy.T @ y; XtX = X.T @ X; Xtt = X.T @ t
    out = []
    for it in range(iters):
        A = ZtZ + np.diag(np.r_[1e-6, 1 / (lam2 * ty2)])
        L = np.linalg.cholesky(A)
        b = np.linalg.solve(A, Zty) + np.sqrt(sy2) * np.linalg.solve(L.T, rng.normal(size=p + 1))
        beta = b[1:]
        lt = lam2 if shared else lam2t
        B = XtX + np.diag(1 / (lt * tt2))
        Lb = np.linalg.cholesky(B)
        g = np.linalg.solve(B, Xtt) + np.sqrt(st2) * np.linalg.solve(Lb.T, rng.normal(size=p))
        ry = y - Zy @ b; rt = t - X @ g
        sy2 = 1 / rng.gamma((n + p) / 2, 1 / (ry @ ry / 2 + (beta ** 2 / (lam2 * ty2)).sum() / 2))
        st2 = 1 / rng.gamma((n + p) / 2, 1 / (rt @ rt / 2 + (g ** 2 / (lt * tt2)).sum() / 2))
        if shared:
            lam2 = 1 / rng.gamma(1.5, 1 / (1 / nu + beta ** 2 / (2 * ty2 * sy2) + g ** 2 / (2 * tt2 * st2)))
            nu = 1 / rng.gamma(1, 1 / (1 + 1 / lam2))
        else:
            lam2 = 1 / rng.gamma(1, 1 / (1 / nu + beta ** 2 / (2 * ty2 * sy2)))
            nu = 1 / rng.gamma(1, 1 / (1 + 1 / lam2))
            lam2t = 1 / rng.gamma(1, 1 / (1 / nut + g ** 2 / (2 * tt2 * st2)))
            nut = 1 / rng.gamma(1, 1 / (1 + 1 / lam2t))
        lt = lam2 if shared else lam2t
        ty2 = 1 / rng.gamma((p + 1) / 2, 1 / (1 / xy + (beta ** 2 / lam2).sum() / (2 * sy2)))
        xy = 1 / rng.gamma(1, 1 / (1 + 1 / ty2))
        tt2 = 1 / rng.gamma((p + 1) / 2, 1 / (1 / xt + (g ** 2 / lt).sum() / (2 * st2)))
        xt = 1 / rng.gamma(1, 1 / (1 + 1 / tt2))
        if it >= burn:
            out.append(b[0])
    return np.array(out)

res = []
for r in range(40):
    n, p = 200, 50
    X = rng.normal(size=(n, p))
    g = np.zeros(p); g[:5] = 0.5; g[5:10] = 1.0
    b = np.zeros(p); b[:5] = 0.5; b[10:15] = 1.0
    t = X @ g + rng.normal(size=n)
    y = 1.0 * t + X @ b + rng.normal(size=n)
    ind = gibbs_shared(y, t, X, shared=False)
    sh = gibbs_shared(y, t, X, shared=True)
    res.append([ind.mean() - 1, sh.mean() - 1,
                float(np.quantile(ind, .025) <= 1 <= np.quantile(ind, .975)),
                float(np.quantile(sh, .025) <= 1 <= np.quantile(sh, .975))])
res = np.array(res)
R["shared"] = dict(bias=res[:, :2].mean(0).tolist(), rmse=np.sqrt((res[:, :2] ** 2).mean(0)).tolist(),
                   cover=res[:, 2:].mean(0).tolist())
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.boxplot([res[:, 0], res[:, 1]], tick_labels=["independent scales", "shared scales"])
ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("Posterior mean minus truth")
fig.tight_layout(); fig.savefig(OUT + "shared.pdf"); plt.close()

json.dump(R, open("results/results2.json", "w"), indent=1)
print(json.dumps(R, indent=1))
