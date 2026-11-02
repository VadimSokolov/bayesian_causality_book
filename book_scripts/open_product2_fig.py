"""Problem 9: shrinkage of a null coefficient as the shared scale grows (deterministic, no sampling).

Design: beta ~ N(0, c^2 eta^2), eta ~ C+(0, 1), y ~ N(beta, 1); the plot shows E(kappa | y), kappa = 1 / (1 + c^2 eta^2), by one dimensional
quadrature over eta, for y = 0, 1, 2 and 40 values of the effective global scale c = lambda_j tau_{g,Y} on a log grid from 0.1 to 100.
No random numbers. Produces figs/open_product2.pdf and results/res_open_product2.json (the same P9 numbers as in res_open2.json).
Split out of open_sims2.py on 23 October 2026 (r1023) so that the figure can be regenerated without the simulations; the only change
to the figure is the axis label, which now reads c = lambda_j tau_{g,Y} (the horseshoe global scale, not the treatment effect).
"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats, integrate
PAL = ["#1f5a96", "#d1495b", "#edae49", "#00798c", "#66a182", "#8d5fa8"]
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "pdf.fonttype": 42, "axes.prop_cycle": plt.cycler(color=PAL)})

def shrink(y, c):
    """E[kappa | y] for beta ~ N(0, c^2 eta^2), eta ~ C+(0,1), y ~ N(beta, 1); kappa = 1/(1 + c^2 eta^2)."""
    f = lambda e: stats.norm.pdf(y, 0, np.sqrt(1 + c * c * e * e)) * 2 / (np.pi * (1 + e * e))
    num = integrate.quad(lambda e: f(e) / (1 + c * c * e * e), 0, np.inf, limit=200)[0]
    den = integrate.quad(f, 0, np.inf, limit=200)[0]
    return num / den

cs = np.logspace(-1, 2, 40)
fig, ax = plt.subplots(figsize=(5.2, 3.1))
for y, ls in [(0.0, "-"), (1.0, "--"), (2.0, ":")]:
    ax.semilogx(cs, [shrink(y, c) for c in cs], ls, label=f"$y = {y:.0f}$")
ax.set_xlabel("effective global scale $c = \\lambda_j \\tau_{\\mathrm{g},Y}$"); ax.set_ylabel("posterior shrinkage $E(\\kappa_j \\mid y)$"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figs/open_product2.pdf"); plt.close()
P9 = {f"shrink_y0_c{c}": float(shrink(0.0, c)) for c in [0.3, 1, 3, 10, 30]}
P9.update({f"shrink_y2_c{c}": float(shrink(2.0, c)) for c in [0.3, 1, 3, 10, 30]})
json.dump(P9, open("results/res_open_product2.json", "w"), indent=1)
old = json.load(open("results/res_open2.json"))["P9"]
print("max abs difference from res_open2.json P9:", max(abs(P9[k] - old[k]) for k in P9))
