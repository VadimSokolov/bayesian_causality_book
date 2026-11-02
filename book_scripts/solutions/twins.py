import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
OUT = "figs/"
# B. Paired versus unpaired designs (copied from book_scripts/make_figs2.py, section B only)
icc = np.linspace(0, 0.9999, 400)
N = 20000; m = 50
var_unpaired = 4 / N
var_paired = 2 * (1 - icc) / m
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.semilogy(icc, var_paired, label="50 twin pairs")
ax.axhline(var_unpaired, color="k", ls="--", label="20,000 unpaired, ideal randomization")
ax.set_xlim(0, 1); ax.set_xlabel("Within pair correlation of growth"); ax.set_ylabel("Variance of estimate (total variance 1)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "twins.pdf"); plt.close()
