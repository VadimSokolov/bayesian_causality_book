"""Figure and numbers for the cereal merger and remedy example."""
import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

R = pickle.load(open("results/cereal.pkl", "rb"))
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED, INK, INK2, MUTED, GRID = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0")
# Drawn at the printed width: the text block is 5.25 in and the figure is 5.25 in wide and is included at \linewidth, so it
# prints at scale 1.0 and the font sizes below are the printed sizes (base 9 pt, axis labels 9, tick labels 8, legend 8,
# titles 9, annotations 8). savefig.bbox is "standard" so that the PDF page is exactly the figure size.
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "axes.titlecolor": INK, "savefig.bbox": "standard"})
# gray scale choices for black and white print (misc/grayscale/audit-1.md): the merger is black and the remedy mid gray,
# GBC is solid and BLP GMM dashed, so every curve differs from the others by gray level or line style
MER, REM = INK, "#808080"
BRANDS = ["A1", "A2", "A3", "B1", "B2", "C1", "C2", "D1"]
tr, g, kr = R["truth"], R["gbc_est"], R["gmm_kr"]

# layout: (a) the brand violins over the full width, (b) and (c) side by side below, one legend under (b) and (c)
fig = plt.figure(figsize=(5.25, 5.8), layout="constrained")
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.25, 0.3])
ax = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
al = fig.add_subplot(gs[2, :]); al.axis("off")
# (a) brand level price effects of the merger, GBC posterior
b = 100 * R["brand"]
x = np.arange(8)
cols = [MUTED] * 3 + [BLUE] * 2 + [VIOLET] * 2 + [AQUA]
vp = ax[0].violinplot([b[:, j] for j in range(8)], positions=x, widths=0.7, showextrema=False)
for body, c in zip(vp["bodies"], cols):
    body.set_facecolor(c); body.set_edgecolor(c); body.set_alpha(0.75)
ax[0].scatter(x, 100 * R["brand_true"], marker="_", s=300, color=INK, lw=1.6, label="truth", zorder=4)
ax[0].axhline(5, color=RED, ls="--", lw=0.9); ax[0].text(7.4, 5.4, "SSNIP 5%", color=RED, fontsize=8, ha="right")
ax[0].set_xticks(x); ax[0].set_xticklabels(BRANDS); ax[0].set_ylabel("price change, %")
ax[0].set_title("Merger of B and C: brand level posterior"); ax[0].legend(loc="upper left")
# (b) merger versus remedy, GBC and GMM
grid = np.linspace(0, 100 * max(g[:, 1].max(), kr[:, 1].max()) * 1.02, 400)
for i, (lab, col) in enumerate([("merger", MER), ("with divestiture of C2", REM)]):
    y = gaussian_kde(100 * g[:, 1 + i])(grid)
    ax[1].fill_between(grid, y, color=col, alpha=0.2, lw=0)
    ax[1].plot(grid, y, color=col, lw=1.6, label=f"GBC, {lab}")
    yk = gaussian_kde(100 * kr[:, 1 + i])(grid)
    ax[1].plot(grid, yk, color=col, lw=1.3, ls="--", label=f"BLP GMM, {lab}")
    ax[1].axvline(100 * tr[1 + i], color=col, lw=1.1, ls=":")
ax[1].axvline(5, color=RED, ls="--", lw=0.9)
ax[1].text(5.3, 0.98, "SSNIP 5%", color=RED, fontsize=8, rotation=90, ha="left", va="top", transform=ax[1].get_xaxis_transform())
ax[1].set_yticks([]); ax[1].set_xlabel("merging parties' average\nprice change, % (dotted: truth)")
ax[1].set_title("Does the remedy fix the merger?")
# (c) decision curves, same styles as (b)
tau = np.linspace(0, 100 * g[:, 1].max(), 300)
for i, col in enumerate([MER, REM]):
    ax[2].plot(tau, [(100 * g[:, 1 + i] > t).mean() for t in tau], color=col, lw=1.6)
    ax[2].plot(tau, [(100 * kr[:, 1 + i] > t).mean() for t in tau], color=col, lw=1.3, ls="--")
ax[2].axvline(5, color=RED, ls="--", lw=0.9)
ax[2].text(5.3, 0.09, "SSNIP 5%", color=RED, fontsize=8, rotation=90, ha="left", va="bottom", transform=ax[2].get_xaxis_transform())
ax[2].set_xlabel(r"threshold $r$, %"); ax[2].set_ylabel(r"$\Pr(\Delta p/p>r\mid\mathcal{D})$")
ax[2].set_title("Inputs to the Bayes rule")
hh, ll = ax[1].get_legend_handles_labels()
al.legend(hh, ll, loc="center", ncol=2, handlelength=3, columnspacing=1.6)      # one legend for (b) and (c): the same four curves
fig.savefig("figures/fig_cereal.pdf"); plt.close(fig)

nums = {}
def f(v, k=1): return f"{v:.{k}f}"
nums["cerdpmtrue"] = f(100 * tr[1]); nums["cerdprtrue"] = f(100 * tr[2])
nums["cerdpmgbc"] = f(100 * g[:, 1].mean()); nums["cerdpmgbcsd"] = f(100 * g[:, 1].std())
nums["cerdprgbc"] = f(100 * g[:, 2].mean()); nums["cerdprgbcsd"] = f(100 * g[:, 2].std())
nums["cerdpmgmm"] = f(100 * R["gmm_point"][1]); nums["cerdprgmm"] = f(100 * R["gmm_point"][2])
nums["cerdpmgmmsd"] = f(100 * kr[:, 1].std()); nums["cerdprgmmsd"] = f(100 * kr[:, 2].std())
nums["cerdpmlogit"] = f(100 * R["logit"][1]); nums["cerdprlogit"] = f(100 * R["logit"][2])
nums["cerpmgbc"] = f((g[:, 1] > 0.05).mean(), 2); nums["cerprgbc"] = f((g[:, 2] > 0.05).mean(), 2)
nums["cerpmgmm"] = f((kr[:, 1] > 0.05).mean(), 2); nums["cerprgmm"] = f((kr[:, 2] > 0.05).mean(), 2)
nums["cercsmtrue"] = f(tr[3], 3); nums["cercsrtrue"] = f(tr[4], 3)
nums["cercsmgbc"] = f(g[:, 3].mean(), 3); nums["cercsrgbc"] = f(g[:, 4].mean(), 3)
nums["cerltrue"] = f(tr[0], 3); nums["cerlgbc"] = f(g[:, 0].mean(), 3)
sh = R["shares"]
nums["cershA"] = f(100 * sh[:3].sum()); nums["cershB"] = f(100 * sh[3:5].sum())
nums["cershC"] = f(100 * sh[5:7].sum()); nums["cershD"] = f(100 * sh[7])
nums["cerinside"] = f(100 * sh.sum())
# the remedy row of the decision discussion: the count of posterior draws above 5 percent and the central 90 percent of the draws, GBC and GMM
nums["cerprgbcnabove"] = f"{int((g[:, 2] > 0.05).sum()):,}".replace(",", "{,}")
nums["cerngbcdraws"] = f"{len(g):,}".replace(",", "{,}")
nums["cerdprgbcqlo"] = f(100 * np.quantile(g[:, 2], 0.05), 2); nums["cerdprgbcqhi"] = f(100 * np.quantile(g[:, 2], 0.95), 2)
nums["cerdprgmmqlo"] = f(100 * np.quantile(kr[:, 2], 0.05), 2); nums["cerdprgmmqhi"] = f(100 * np.quantile(kr[:, 2], 0.95), 2)
with open("results/numbers_cereal.tex", "w") as fh:
    for k, v in nums.items():
        fh.write(f"\\newcommand{{\\num{k}}}{{{v}}}\n")
print(nums)
