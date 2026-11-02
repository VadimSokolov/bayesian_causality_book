"""Merge the 60 networks of localsbc_density.py: heat maps, medians at the typical and extreme trials, region summaries.

Reads results/localsbc_density/net_*.npz and exact.npz. For each (size, design): the mean over the ten seeds of the Wasserstein-1 error (the network curve sorted over the 200 levels, saved as ratio_sorted since r1023b) divided
by the exact posterior sd; the unsorted mean absolute quantile error (saved as ratio, an upper bound on it) is summarized in the same way beside it, per grid summary (the heat map at N = 60,000 goes to figs/open_localsbc_density.pdf); the median over seeds at the
typical trial (18, 27) and the extreme trial (1, 49) with the standard error of the mean over seeds; the average over the central region
(distance from (25, 25) at most 12.5, one sd of the Gaussian design) and the outer region; the Spearman correlation across grid summaries
between the seed averaged error and the design's mass q(s), and between the error and the distance from the centre. Writes results/res_localsbc_density.json (keys "sorted" and "unsorted", each as the earlier file, plus "gap_unsorted_over_sorted"). The heat map is drawn from the sorted metric.
"""
import json, glob
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9})
M = 50; SIZES = [15000, 60000, 240000]; DES = ["uniform", "gaussian"]
YY0, YY1 = np.meshgrid(np.arange(M + 1), np.arange(M + 1), indexing="ij"); dist = np.hypot(YY0 - 25, YY1 - 25); central = dist <= 12.5
E = np.load("results/localsbc_density/exact.npz"); q = E["qmass"]
def summarize(key):
    G = {}
    for f in glob.glob("results/localsbc_density/net_*.npz"):
        z = np.load(f); G.setdefault((int(z["N"]), DES[int(z["design"])]), []).append(z[key])
    R = {}
    for N in SIZES:
        for d in DES:
            a = np.array(G[(N, d)]); mean = a.mean(0); mass = (np.full_like(q, 1.0 / q.size) if d == "uniform" else q)
            tp, ex = a[:, 18, 27], a[:, 1, 49]
            R[f"{N}|{d}"] = dict(seeds=len(a), typical_median=float(np.median(tp)), typical_mean=float(tp.mean()), typical_se=float(tp.std(ddof=1) / np.sqrt(len(a))),
                                 extreme_median=float(np.median(ex)), extreme_mean=float(ex.mean()), extreme_se=float(ex.std(ddof=1) / np.sqrt(len(a))),
                                 central_mean=float(a[:, central].mean()), central_se=float(a[:, central].mean(1).std(ddof=1) / np.sqrt(len(a))),
                                 outer_mean=float(a[:, ~central].mean()), outer_se=float(a[:, ~central].mean(1).std(ddof=1) / np.sqrt(len(a))),
                                 overall_mean=float(a.mean()), overall_median_cell=float(np.median(mean)),
                                 corner_mean=float(a[:, (YY0 <= 5) & (YY1 >= 45)].mean()),
                                 spearman_mass=float(stats.spearmanr(mean.ravel(), mass.ravel())[0]) if d == "gaussian" else None,
                                 spearman_dist=float(stats.spearmanr(mean.ravel(), dist.ravel())[0]), expected_count_extreme=float(N * mass[1, 49]), expected_count_typical=float(N * mass[18, 27]))
        u, g = np.array(G[(N, "uniform")]).mean(0), np.array(G[(N, "gaussian")]).mean(0)
        R[f"{N}|gaussian_over_uniform"] = dict(extreme=float(g[1, 49] / u[1, 49]), typical=float(g[18, 27] / u[18, 27]), outer=float(g[~central].mean() / u[~central].mean()), central=float(g[central].mean() / u[central].mean()))
    return R, G
RS, GS = summarize("ratio_sorted"); RU, GU = summarize("ratio"); G = GS
gap = {}
for N in SIZES:
    for d in DES:
        s_, u_ = np.array(GS[(N, d)]), np.array(GU[(N, d)])
        gap[f"{N}|{d}"] = dict(typical=float(u_[:, 18, 27].mean() / s_[:, 18, 27].mean()), extreme=float(u_[:, 1, 49].mean() / s_[:, 1, 49].mean()), overall=float(u_.mean() / s_.mean()),
                               max_cell=float((u_.mean(0) / s_.mean(0)).max()), share_cells_crossing=float((u_.mean(0) > s_.mean(0) * (1 + 1e-9)).mean()))
R = dict(sorted=RS, unsorted=RU, gap_unsorted_over_sorted=gap)
json.dump(R, open("results/res_localsbc_density.json", "w"), indent=1)
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9)); vmax = max(np.array(G[(60000, d)]).mean(0).max() for d in DES)
for ax, d in zip(axs, DES):
    im = ax.imshow(np.array(G[(60000, d)]).mean(0).T, origin="lower", vmin=0, vmax=vmax, cmap="Greys"); ax.set_title(f"trained on {d} summaries", fontsize=9)
    ax.set_xlabel("$y_0$"); ax.set_ylabel("$y_1$"); ax.plot([18], [27], "o", mfc="none", mec="C3", ms=6); ax.plot([1], [49], "s", mfc="none", mec="C3", ms=6)
fig.colorbar(im, ax=axs, shrink=0.85, label="Wasserstein error / exact sd"); fig.savefig("figs/open_localsbc_density.pdf", bbox_inches="tight"); plt.close()
for m in R:
    for k, v in R[m].items(): print(m, k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
