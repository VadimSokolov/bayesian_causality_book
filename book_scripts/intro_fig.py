"""The opening figure: one fitted model, many answers, for the 401(k) data.

Reads results/deep401k_draws.npz (from deep_401k.py) and writes figs/intro_onefit.pdf.
The DML intervals are those of Table tab:k401 in Chapter 16 (app401k_book.py). The bands of the network are the mean of the 200 refits
plus or minus 1.96 standard deviations, as in deep_401k.py.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PAL = ["#1f5a96", "#d1495b", "#edae49", "#00798c", "#66a182", "#8d5fa8"]
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "pdf.fonttype": 42})
D = np.load("results/deep401k_draws.npz")
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.5), gridspec_kw=dict(width_ratios=[1.1, 1.2, 1.0]))
# A: the one number DML reports, against the posterior of the same number
a = axs[0]
for y, (est, lo, hi), lab in [(2, (8.59, 6.05, 11.12), "DML, forests"), (1, (9.52, 6.89, 12.14), "DML, lasso")]:
    a.plot([lo, hi], [y, y], color=PAL[2], lw=3, solid_capstyle="round"); a.plot(est, y, "o", color="k", ms=3)
    a.text(lo, y + 0.25, lab, fontsize=7)
ate = D["ate"]; xs = np.linspace(5, 13, 200); kde = stats.gaussian_kde(ate, bw_method=0.6)
dens = kde(xs) / kde(xs).max() * 0.8
a.fill_between(xs, -0.2, -0.2 + dens, color=PAL[0], alpha=0.6, lw=0); a.text(5.1, 0.45, "posterior", fontsize=7, color=PAL[0])
a.set_yticks([]); a.set_ylim(-0.4, 2.7); a.set_xlabel("average effect, \\$1000"); a.set_title("One number", fontsize=9)
# B: the whole distributional effect from the same fit
a = axs[1]; QS = D["QS"]; q = D["qte"]; lo, hi = q.mean(0) - 1.96 * q.std(0, ddof=1), q.mean(0) + 1.96 * q.std(0, ddof=1)
a.fill_between(QS, lo, hi, color=PAL[0], alpha=0.25, lw=0); a.plot(QS, q.mean(0), color=PAL[0], lw=2)
a.axhline(0, color="grey", lw=0.5); a.set_xlabel("quantile of net financial assets"); a.set_ylabel("effect, \\$1000")
a.set_title("Every quantile", fontsize=9)
# C: who gains
a = axs[2]; G = D["inc_effects"]; m = G.mean(0); lo, hi = m - 1.96 * G.std(0, ddof=1), m + 1.96 * G.std(0, ddof=1)
cols = [PAL[3], PAL[4], PAL[2], PAL[1], PAL[5]]
a.bar(range(5), m, color=cols, alpha=0.9); a.errorbar(range(5), m, yerr=[m - lo, hi - m], fmt="none", ecolor="k", capsize=2, lw=0.8)
a.set_xticks(range(5)); a.set_xticklabels(["1", "2", "3", "4", "5"]); a.set_xlabel("income quintile")
a.set_title("Every subgroup", fontsize=9)
fig.tight_layout(); fig.savefig("figs/intro_onefit.pdf"); plt.close()
print("ok")
