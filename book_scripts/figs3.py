import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import integrate, stats
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
b = np.linspace(-6, 6, 1201)
def hs_dens(x):
    f = lambda lam: stats.norm.pdf(x, 0, lam) * 2 / (np.pi * (1 + lam ** 2))
    return integrate.quad(f, 1e-6, np.inf, limit=200)[0]
hs = np.array([hs_dens(x) if abs(x) > 1e-3 else np.nan for x in b])
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.9))
# Gray scale print: one line style per prior (the horseshoe is solid black), so that the legend can be read without colour
axs[0].plot(b, hs, "-", color="k", lw=1.8, label="horseshoe"); axs[0].plot(b, stats.laplace.pdf(b, scale=1 / np.sqrt(2)), "--", color="0.3", lw=1.5, label="Laplace")
axs[0].plot(b, stats.norm.pdf(b), ":", color="0.3", lw=2.0, label="normal"); axs[0].plot(b, stats.cauchy.pdf(b), "-.", color="0.55", lw=1.5, label="Cauchy")
axs[0].set_ylim(0, 0.8); axs[0].set_xlabel(r"$\beta$"); axs[0].set_ylabel("Prior density"); axs[0].legend(frameon=False, fontsize=7, handlelength=3.2)
k = np.linspace(0.005, 0.995, 400)
axs[1].plot(k, stats.beta.pdf(k, 0.5, 0.5), "-", color="k", lw=1.8, label="horseshoe, Beta(1/2,1/2)")
# Laplace prior with unit variance: kappa density via simulation
lam2 = np.random.default_rng(1).exponential(1, 200000); kap = 1 / (1 + lam2)
axs[1].hist(kap, bins=60, density=True, histtype="step", color="0.45", lw=1.2, label="Laplace")
axs[1].set_ylim(0, 4); axs[1].set_xlabel(r"shrinkage coefficient $\kappa$"); axs[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig("figs/hs_prior.pdf"); plt.close()
d = np.arange(0, 6)
for a, bb, lab in [(0.95, 2, "BART default"), (0.25, 3, "BCF effect forest")]:
    pass
fig, ax = plt.subplots(figsize=(5, 3))
w = 0.38
ax.bar(d - w / 2, 0.95 * (1 + d) ** -2.0, w, label=r"$\alpha=0.95,\ \beta=2$ (BART)")
ax.bar(d + w / 2, 0.25 * (1 + d) ** -3.0, w, label=r"$\alpha=0.25,\ \beta=3$ (BCF effect)")
ax.set_xlabel("Node depth"); ax.set_ylabel("Prior probability of splitting"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figs/bart_depth.pdf")
