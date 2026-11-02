"""Figure for the section "A posterior is a map from noise" of the chapter "Computing the Posterior".

Produces figs/comp_inverse.pdf: one bimodal posterior, the mixture 0.4 N(-1.5, 0.45^2) + 0.6 N(1.5, 0.6^2), shown as a distribution
function with six uniform draws carried through it, as its quantile function, and as a histogram of 5,000 draws Q(U) with the exact density.
Seeded (numpy default_rng(7)); requires numpy, scipy and matplotlib and runs in a few seconds.
Run from the repository root: python book_scripts/comp_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "pdf.fonttype": 42})
PAL = ["#1f5a96", "#d1495b", "#edae49", "#00798c", "#66a182", "#8d5fa8"]
OUT = "figs/"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

# a bimodal posterior: 0.4 N(-1.5, 0.45^2) + 0.6 N(1.5, 0.6^2)
W, M1, S1, M2, S2 = 0.4, -1.5, 0.45, 1.5, 0.6
cdf = lambda t: W * stats.norm.cdf(t, M1, S1) + (1 - W) * stats.norm.cdf(t, M2, S2)
pdf = lambda t: W * stats.norm.pdf(t, M1, S1) + (1 - W) * stats.norm.pdf(t, M2, S2)
grid = np.linspace(-4.5, 4.5, 20001); F = cdf(grid)
Q = lambda u: np.interp(u, F, grid)          # the quantile function, by numerical inversion of the distribution function

u6 = rng.uniform(size=6); t6 = Q(u6)         # six uniform draws and the posterior draws they produce
draws = Q(rng.uniform(size=5000))            # 5,000 draws Q(U)

LO, HI = -3.5, 4.0
fig = plt.figure(figsize=(6.4, 2.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.55], wspace=0.55)
ax = fig.add_subplot(gs[0]); ay = fig.add_subplot(gs[1]); az = fig.add_subplot(gs[2], sharey=ay)

tt = np.linspace(LO, HI, 400)
ax.plot(tt, cdf(tt), color=PAL[0])
for u, t in zip(u6, t6):
    ax.plot([LO, t, t], [u, u, 0], ls=":", color=PAL[1], lw=0.9)
ax.plot(t6, np.zeros(6), "o", color=PAL[1], ms=3.5, clip_on=False)
ax.set_xlim(LO, HI); ax.set_ylim(0, 1)
ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$u = F(\theta)$"); ax.set_title("Distribution function", fontsize=10)

uu = np.linspace(0.002, 0.998, 800)
ay.plot(uu, Q(uu), color=PAL[0])
ay.plot(u6, t6, "o", color=PAL[1], ms=3.5)
ay.set_xlim(0, 1); ay.set_ylim(LO, HI)
ay.set_xlabel(r"level $u$"); ay.set_ylabel(r"$\theta = Q(u)$"); ay.set_title("Quantile function", fontsize=10)

az.hist(draws, bins=60, range=(LO, HI), density=True, orientation="horizontal", color=PAL[3], alpha=0.45, edgecolor="none")
az.plot(pdf(tt), tt, color=PAL[1], lw=1.0)
az.set_xticks([]); az.tick_params(labelleft=False); az.set_title("Draws", fontsize=10)

fig.subplots_adjust(left=0.09, right=0.98, bottom=0.2, top=0.88)
fig.savefig(OUT + "comp_inverse.pdf"); plt.close()
print("written", OUT + "comp_inverse.pdf", "u6 =", np.round(u6, 3), "draws mean/sd =", round(draws.mean(), 3), round(draws.std(), 3))
