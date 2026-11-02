"""Weak overlap paradox simulation (Chapter 6, Table tab:weak and Figure ipw_weak).

Design: B = 1000 cells with selection probabilities p_b on a grid (0.01 to 0.99, or 0.1 to 0.9 for the MAR design), n = 100 units drawn with
replacement from the cells, R ~ Bernoulli(p_b), Y ~ Bernoulli(theta_b). Four designs with theta_b ~ U(a, b) independent of p, (a, b) in
(0.1, 0.9), (0.1, 0.4), (0.35, 0.65), (0.6, 0.9), and a fifth, MAR, with theta_b = 0.5 p_b + 0.5 U(0,1). The target is the mean of the theta_b.
Six estimators: Horvitz Thompson, Hajek, Li's hierarchical estimator, the binned Hajek estimator (Ghosh's coarsening of the propensity into ten
bins), the coupled prior, and the bin posterior (independent Beta(1,1) priors in ten bins of the selection probability, estimator
sum_j w_j (1 + S_j)/(2 + M_j) with w_j the known share of the cells in bin j, S_j and M_j the sum of Y and the count of selected units in the bin).
5,000 replications per design (200 before the first review round); the stream of the random numbers is that of the earlier script, which the
bin posterior does not touch. Fixed seed 2204. Saved: results/res_ipw.json (bias, mean squared error times 100 and their Monte Carlo standard
errors, paired standard errors of each mean squared error against the coupled prior, the mean fraction of bins with no selected unit).
"""
import numpy as np, json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(2204)
B, n, reps = 1000, 100, int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 5000
EMPTY = []
def estimators(p, theta, k=10):
    X = rng.integers(0, B, n); R = rng.binomial(1, p[X]); Y = rng.binomial(1, theta[X]); px = p[X]
    ht = np.mean(R * Y / px)
    hj = np.sum(R * Y / px) / np.sum(R / px)
    li = (np.sum(R * Y) + 1) / (np.sum(R) + 2)
    edges = np.linspace(p.min(), p.max() + 1e-9, k + 1); bins = np.digitize(px, edges) - 1
    num = den = 0.0
    for j in range(k):
        s = bins == j
        if s.sum() == 0: continue
        pt = px[s].mean(); num += np.sum(R[s] * Y[s]) / pt; den += np.sum(R[s]) / pt
    bs = num / den if den > 0 else li
    # coupled hierarchical prior: logit theta ~ a + b logit p, fitted on observed, shrink observed cells toward it
    obs = R == 1
    if obs.sum() > 5 and len(np.unique(Y[obs])) > 1:
        lr = LogisticRegression(C=1e6, max_iter=500).fit(np.log(px[obs] / (1 - px[obs])).reshape(-1, 1), Y[obs])
        pm = lr.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
    else:
        pm = np.full(B, li)
    m = 5.0; s_ = np.bincount(X[obs], weights=Y[obs], minlength=B); c_ = np.bincount(X[obs], minlength=B)
    cp = ((m * pm + s_) / (m + c_)).mean()
    ebins = np.digitize(p, edges) - 1; wj = np.bincount(ebins, minlength=k)[:k] / B       # known population shares of the bins
    Mj = np.bincount(bins[obs], minlength=k)[:k]; Sj = np.bincount(bins[obs], weights=Y[obs], minlength=k)[:k]
    bp = np.sum(wj * (1 + Sj) / (2 + Mj)); EMPTY.append(np.mean(Mj == 0))
    return [ht, hj, li, bs, cp, bp]
names = ["Horvitz Thompson", "Hajek", "Li hierarchical", "binned Hajek", "coupled prior", "bin posterior"]
def mc(E):
    """Mean squared error times 100 and bias; the Monte Carlo standard error of each mean squared error, and the paired standard error of
    each estimator's difference from the coupled prior (the estimators share the simulated data, so the paired error is the one that compares them)."""
    S = 100 * E ** 2
    return {"mse100": S.mean(0).tolist(), "bias": E.mean(0).tolist(), "mse100_se": (S.std(0, ddof=1) / np.sqrt(len(E))).tolist(),
            "diff_coupled_se": ((S - S[:, [4]]).std(0, ddof=1) / np.sqrt(len(E))).tolist(), "empty_bin_frac": float(np.mean(EMPTY)), "reps": len(E)}
out = {}
EMPTY.clear()
p = np.linspace(0.01, 0.99, B)
for (a, b) in [(0.1, 0.9), (0.1, 0.4), (0.35, 0.65), (0.6, 0.9)]:
    E = []
    for r in range(reps):
        theta = rng.uniform(a, b, B); psi = theta.mean()
        E.append(np.array(estimators(p, theta)) - psi)
    E = np.array(E); out[f"mcar_{a}_{b}"] = mc(E); EMPTY.clear()
# MAR: theta correlated with p
p2 = np.linspace(0.1, 0.9, B); E = []
for r in range(reps):
    theta = 0.5 * p2 + 0.5 * rng.uniform(0, 1, B); psi = theta.mean()
    E.append(np.array(estimators(p2, theta)) - psi)
E = np.array(E); out["mar"] = mc(E)
# theoretical bias of Li under MAR: sigma_{theta,p}/pbar
th = 0.5 * p2 + 0.25; out["li_theory_bias"] = float((np.mean(th * p2) - p2.mean() * th.mean()) / p2.mean())
import sys
if "--figs-only" in sys.argv:    # redraw ipw_weak.pdf only: results/res_ipw.json is left as it is, and the largest difference between it and this seeded rerun is printed
    saved = json.load(open("results/res_ipw.json"))
    print("largest difference from the saved res_ipw.json:", max(abs(a - b) for k in saved if isinstance(saved[k], dict) for m_ in saved[k] for a, b in zip(saved[k][m_], out[k][m_])))
else:
    json.dump(out, open("results/res_ipw.json", "w"), indent=1)
fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.0))
keys = ["mcar_0.1_0.9", "mcar_0.1_0.4", "mcar_0.35_0.65", "mcar_0.6_0.9"]; labs = ["[0.1,0.9]", "[0.1,0.4]", "[0.35,0.65]", "[0.6,0.9]"]
w = 0.13
# Gray scale print: the bars differ by fill and hatch (the coupled prior is black) and the medians of the boxes are black
fills = ["white", "0.85", "0.55", "white", "0.0", "0.7"]; hatches = ["////", "", "", "xxxx", "", "\\\\\\\\"]
for i, nm in enumerate(names):
    axs[0].bar(np.arange(4) + (i - 2.5) * w, [out[k]["mse100"][i] for k in keys], w, label=nm, color=fills[i], edgecolor="k", linewidth=0.5, hatch=hatches[i])
axs[0].set_xticks(range(4)); axs[0].set_xticklabels(labs, fontsize=7); axs[0].set_ylabel("MSE x 100"); axs[0].set_title("Selection unrelated to outcome", fontsize=9)
axs[0].legend(frameon=False, fontsize=7, handleheight=1.3)
axs[1].boxplot([E[:, i] for i in range(6)], tick_labels=["HT", "Hajek", "Li", "binned", "coupled", "bin post."], medianprops=dict(color="k", linewidth=1.8))
axs[1].axhline(0, color="k", lw=0.6); axs[1].set_title("Selection related to outcome", fontsize=9); axs[1].set_ylabel("Estimate minus truth")
fig.tight_layout(); fig.savefig("figs/ipw_weak.pdf")
print(json.dumps(out, indent=0))
