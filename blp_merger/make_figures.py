r"""All figures for the paper. Reads results/experiments.pkl, writes figures/*.pdf and
results/numbers.tex (macros used in the text).

Each figure is drawn at the width it prints at in the 7 x 10 in book: the text width W = 5.25 in times the width option
of its \includegraphics in book_tex/chapters/m15.tex (see newfig). The PDF page is then exactly the printed size, a 9 pt
font prints at 9 pt, and constrained layout fixes the margins (no bbox tight, which would change the page width)."""
import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, norm
from estimators import PARAMS
from blp_core import CONDUCTS

R = pickle.load(open("results/experiments.pkl", "rb"))
FIG = "figures/"
W = 5.25                      # text width in inches: 7 in paper, 1.0 in inner and 0.75 in outer margin
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
METH = [("gbc2", "GBC (this chapter)", BLUE), ("jmr", "Bayes inversion MCMC", ORANGE),
        ("gmm", "BLP GMM", AQUA), ("logit", "Logit IV", YELLOW)]
# Gray scale choices for black and white print (misc/grayscale/audit-1.md). One style per method, the same in every
# figure below: GBC is black and solid, then dark gray dashed, mid gray dash-dot and light gray solid, so that two
# series never differ by hue alone. A fill is drawn only under the GBC curve. ZO keeps the black curve on top.
G_GBC, G_JMR, G_GMM, G_LOGIT, G_FILL = "#0b0b0b", "#4d4d4d", "#8c8c8c", "#b3b3b3", "#dcdcdc"
STY = {"gbc2": dict(color=G_GBC, ls="-"), "jmr": dict(color=G_JMR, ls="--"), "gmm": dict(color=G_GMM, ls="-."),
       "logit": dict(color=G_LOGIT, ls="-")}
MARK = {"gbc2": "o", "jmr": "s", "gmm": "^", "logit": "D"}
ZO = {"gbc2": 3.6, "jmr": 3.4, "gmm": 3.2, "logit": 3.0}
CCOL = {0: G_JMR, 1: G_GMM, 2: G_GBC}      # conduct: Bertrand Nash dark gray, joint monopoly mid gray, perfect competition black
CNAME = {0: "Bertrand Nash", 1: "Joint monopoly", 2: "Perfect competition"}
BLUES = LinearSegmentedColormap.from_list("b", ["#f2f6fc", "#9cc0ee", "#2a78d6", "#123f78"])
# Printed size: base 9 pt, axis labels 9, tick labels 8, legends 8, panel titles 9, annotations 7.5 or more. Line widths
# are the printed widths (the figures are no longer shrunk by a half in the book).
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.titlesize": 9, "figure.titlesize": 9.5,
    "figure.labelsize": 9, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False, "legend.labelspacing": 0.4,
    "axes.titlecolor": INK,
    "savefig.dpi": 300,          # raster parts (imshow cells, colour bars) come out at 300 ppi at the drawn, hence printed, size
    "hatch.linewidth": 0.5})
TR = R["truth"]
TM = 3                       # featured market
PROD = ["A1", "A2", "B1", "B2", "C1", "C2"]
nums = {}


def q(x, a=5, b=95):
    return np.percentile(x, [a, b], axis=0)


def newfig(frac, h):
    """Figure at its printed size: width frac * W (frac is the width option of \\includegraphics), height h, in inches."""
    fig = plt.figure(figsize=(frac * W, h), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.04, hspace=0.04)
    return fig


# ------------------------------------------------------------------ 1 parameter posteriors
fig = newfig(1.0, 2.8)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.2])
ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
names = [("alpha", "price sensitivity\nscale " + r"$\alpha$", 0), ("sp", "income\nheterogeneity " + r"$\sigma_p$", 1),
         ("sx", "taste\ndispersion " + r"$\sigma_x$", 2)]
for a, (k, lab, i) in zip(ax, names):
    grid = np.linspace({0: .6, 1: 0, 2: .2}[i], {0: 1.35, 1: .85, 2: 1.25}[i], 400)
    for key, nm, col in METH[:3]:
        if key == "gmm":
            y = norm.pdf(grid, R["gmm"]["th2"][i], np.sqrt(R["gmm"]["cov"][i, i]))
        else:
            y = gaussian_kde(R[key]["draws"]["th2"][:, i])(grid)
        if key == "gbc2":
            a.fill_between(grid, y, color=G_FILL, lw=0)          # the only fill in a panel
        a.plot(grid, y, lw=1.5, label=nm, zorder=ZO[key], **STY[key])
    y = gaussian_kde(R["gbc2_nomicro"]["draws"]["th2"][:, i])(grid)
    a.plot(grid, y, color=G_LOGIT, lw=1.9, ls="-", label="GBC without micro moments", zorder=2.8)
    a.axvline(R["true"][k], color=INK, lw=1.1, ls=":", label="true value", zorder=4)
    a.set_xlabel(lab); a.set_yticks([]); a.xaxis.set_major_locator(plt.MaxNLocator(4))
h, l = ax[0].get_legend_handles_labels()
lax = fig.add_subplot(gs[1, :]); lax.axis("off")
lax.legend(h, l, loc="center", ncol=3, handlelength=3.0, columnspacing=1.4)
fig.suptitle("Posterior distributions of the nonlinear demand parameters", color=INK)
fig.savefig(FIG + "fig_theta_posteriors.pdf"); plt.close(fig)

# ------------------------------------------------------------------ 2 price decomposition
d = R["gbc2"]["draws"]
p = R["data"]["p"][TM]
m_mean, (m_lo, m_hi) = d["m"][:, TM].mean(0), q(d["m"][:, TM])
fig = newfig(0.85, 3.4)
gs = fig.add_gridspec(2, 1, height_ratios=[0.17, 1])
lax = fig.add_subplot(gs[0]); lax.axis("off")
ax = fig.add_subplot(gs[1])
x = np.arange(6)
ax.bar(x, p - m_mean, width=0.62, color="#9cc0ee", label="posterior mean marginal cost", zorder=2)
ax.bar(x, m_mean, bottom=p - m_mean, width=0.62, color=BLUE, label="posterior mean markup", zorder=2,
       edgecolor="white", linewidth=1.2)
ax.errorbar(x, p - m_mean + m_mean, yerr=[m_mean - m_lo, m_hi - m_mean], fmt="none", ecolor=INK,
            capsize=3, lw=1.0, zorder=3)
ax.scatter(x, p - TR["m"][TM] + 0 * x, marker="D", s=26, color=ORANGE, zorder=4,
           label="true marginal cost", edgecolor="white", linewidth=0.8)
for j in range(6):
    ax.text(j, p[j] + 0.18 + (m_hi[j] - m_mean[j]), f"L = {d['lerner'][:, TM, j].mean():.2f}",
            ha="center", fontsize=7.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([f"{n}\n(firm {n[0]})" for n in PROD])
ax.set_ylabel("price, $10,000"); ax.set_ylim(0, p.max() + 1.2)
h, l = ax.get_legend_handles_labels()
lax.legend([h[i] for i in (1, 2, 0)], [l[i] for i in (1, 2, 0)], loc="center", ncol=2, columnspacing=1.5)   # cost, markup, truth
fig.suptitle("Observed price = recovered marginal cost + markup (market %d)" % (TM + 1), color=INK, fontsize=9)
fig.savefig(FIG + "fig_price_decomposition.pdf"); plt.close(fig)

# ------------------------------------------------------------------ 3 Lerner forest plot
fig = newfig(0.85, 3.5)
gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.27])
ax = fig.add_subplot(gs[0]); lax = fig.add_subplot(gs[1]); lax.axis("off")
off = np.linspace(-0.27, 0.27, 4)
for (key, nm, col), o in zip(METH, off):
    L = R[key]["draws"]["lerner"][:, TM]
    lo, hi = q(L)
    ax.hlines(np.arange(6) + o, lo, hi, color=STY[key]["color"], lw=1.9)
    ax.scatter(L.mean(0), np.arange(6) + o, color=STY[key]["color"], s=20, zorder=3, edgecolor="white", lw=0.7,
               marker=MARK[key])
ax.scatter(TR["lerner"][TM], np.arange(6), marker="|", s=170, color=INK, lw=1.4, zorder=4)
ax.set_yticks(range(6)); ax.set_yticklabels(PROD); ax.invert_yaxis()
ax.set_xlabel(r"Lerner index $(p_j - c_j)/p_j$, 90% intervals")
# gray scale: one gray level and one marker per method; the order inside every row is fixed and the legend title says so
hs = [Line2D([], [], color=STY[k_]["color"], lw=1.9, marker=MARK[k_], markersize=5, markeredgecolor="white",
             markeredgewidth=0.7, label=nm_) for k_, nm_, _ in METH]
hs.append(Line2D([], [], color=INK, ls="", marker="|", markersize=11, markeredgewidth=1.4, label="truth"))
lax.legend(handles=hs, loc="center", ncol=3, title="top to bottom in each row", title_fontsize=8, handlelength=2.6,
           columnspacing=1.4)
ax.set_title("Product level markups, market %d" % (TM + 1))
fig.savefig(FIG + "fig_lerner_forest.pdf"); plt.close(fig)

# ------------------------------------------------------------------ 4 markup ridge: parameters weakly, markups strongly
dj = R["jmr"]["draws"]
fig = newfig(1.0, 4.8)
gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.03])
ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
cax = fig.add_subplot(gs[0, 1]); cax.grid(False)
Lm = dj["lerner"].mean((1, 2))
# gray scale: a thin dark outline instead of a white one keeps the palest draws (lowest markups) visible on white paper
sc = ax[0].scatter(dj["th2"][:, 0], dj["th2"][:, 1], c=Lm, cmap=BLUES, s=14, edgecolor="#555555", lw=0.3)
cb = fig.colorbar(sc, cax=cax); cb.set_label("industry mean Lerner index", color=INK2); cb.outline.set_linewidth(0.5)
ax[0].set_xlabel(r"$\alpha$"); ax[0].set_ylabel(r"$\sigma_p$")
ax[0].set_title(r"Posterior draws of $(\alpha,\sigma_p)$ shaded by implied markup")
Ea = dj["th2"][:, 0] * np.exp(dj["th2"][:, 1] ** 2 / 2)
ax[1].scatter(1 / Ea, Lm, color=BLUE, s=14, alpha=0.8, edgecolor="white", lw=0.3)
ax[1].set_xlabel(r"$1/E[\alpha_i]$, inverse mean price sensitivity")
ax[1].set_ylabel("mean Lerner index")
cc = np.corrcoef(1 / Ea, Lm)[0, 1]
ax[1].set_title("Markups track inverse price sensitivity (corr %.2f)" % cc)
nums["corrLa"] = f"{cc:.2f}"
fig.savefig(FIG + "fig_markup_ridge.pdf"); plt.close(fig)

# ------------------------------------------------------------------ 5 conduct
ref = R["ref"]
fig = newfig(1.0, 5.8)
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
ax = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
xa, ya = ref["S"][:, 13], ref["S"][:, 14]
xl, yl = np.percentile(xa, [1, 99]), np.percentile(ya, [1, 99])
# gray scale: one marker and one gray per conduct, nearly opaque. Perfect competition (black crosses) is drawn last so
# that its cluster at the origin stays on top of the Bertrand band.
CMARK, CSIZE, CLW = {0: "o", 1: "^", 2: "x"}, {0: 11, 1: 12, 2: 14}, {0: 0, 1: 0, 2: 0.8}
for c in [0, 1, 2]:
    k = (ref["C"] == c)
    ax[0].scatter(xa[k][:900], ya[k][:900], s=CSIZE[c], color=CCOL[c], alpha=0.8, marker=CMARK[c], lw=CLW[c], zorder=2 + c)
ax[0].scatter(R["S_obs"][13], R["S_obs"][14], marker="*", s=130, color=INK, edgecolor="white", lw=0.8, zorder=6)
ax[0].set_xlim(*xl); ax[0].set_ylim(*yl)
ax[0].set_xlabel("loading on firm level share term\n" + r"$1/(1-s_f)-1/(1-s_j)$")
ax[0].set_ylabel("loading on industry share term\n" + r"$1/s_0-1/(1-s_f)$")
ax[0].set_title("Markup regressions separate conduct")
hs = [Line2D([], [], ls="", marker=CMARK[c], color=CCOL[c], markersize=5, markeredgewidth=CLW[c], label=CNAME[c])
      for c in [0, 1, 2]]
hs.append(Line2D([], [], ls="", marker="*", color=INK, markeredgecolor="white", markersize=10, label="observed minivan data"))
ax[0].legend(handles=hs, loc="upper left")


def _med(c, lo_=-np.inf, hi_=np.inf):
    sel = (ref["C"] == c) & (xa > lo_) & (xa < hi_)
    if not sel.any():
        sel = (ref["C"] == c)
    return np.median(xa[sel]), np.median(ya[sel])


# direct labels, so that the legend is only a backup: text placed in an empty part of the panel, with a line to the cloud
for c, rng, (fx, fy) in [(2, (-np.inf, np.inf), (0.20, 0.07)), (0, (2.5, 3.5), (0.52, 0.27)), (1, (3.8, 4.6), (0.68, 0.38))]:
    ax[0].annotate(CNAME[c], xy=_med(c, *rng), xytext=(xl[0] + fx * (xl[1] - xl[0]), yl[0] + fy * (yl[1] - yl[0])),
                   ha="center", va="center", fontsize=7.5, color=INK, zorder=7,
                   bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                   arrowprops=dict(arrowstyle="-", lw=0.6, color=INK2, shrinkA=2, shrinkB=3))
pc = R["gbc2"]["draws"] and R["gbc"]["pc"]
ax[1].bar(range(3), pc, color=[CCOL[c] for c in range(3)], width=0.6, edgecolor=INK, lw=0.5)
for c in range(3):
    ax[1].text(c, pc[c] + 0.02, f"{pc[c]:.3f}", ha="center", fontsize=8, color=INK)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["Bertrand\nNash", "Joint\nmonopoly", "Perfect\ncompetition"])
ax[1].set_ylim(0, 1.12); ax[1].set_title("Posterior over conduct")
cal = R["calibration"]
M = np.zeros((3, 3))
for t, pr in zip(cal["conduct_true"], cal["conduct_pred"]):
    M[t, pr] += 1
M = M / M.sum(1, keepdims=True)
ax[2].imshow(M, cmap=BLUES, vmin=0, vmax=1); ax[2].grid(False)
for i in range(3):
    for j in range(3):
        ax[2].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if M[i, j] > .5 else INK)
lab = ["Bertrand", "Monopoly", "Perfect"]
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(lab)
ax[2].set_yticks(range(3)); ax[2].set_yticklabels(lab)
ax[2].tick_params(labelsize=7.5)             # the three cells are 41 pt wide, "Monopoly" is 38 pt at 8 pt
ax[2].set_xlabel("posterior mode"); ax[2].set_ylabel("true conduct")
ax[2].set_title("Held out classification")
nums["conductacc"] = f"{np.trace(M) / 3:.2f}"
fig.savefig(FIG + "fig_conduct.pdf"); plt.close(fig)

# ------------------------------------------------------------------ 6 diversion ratios
# full text width: a cell of the 6 by 6 maps must hold the two lines "0.12" and "(0.10)" at 7.5 pt (about 23 pt wide)
fig = newfig(1.0, 3.5)
ax = fig.subplots(1, 2, sharey=True)
Dt = TR["div"][TM]; Dg = R["gbc2"]["draws"]["div"][:, TM].mean(0)
Dl = R["logit"]["draws"]["div"][:, TM].mean(0)
vmax = max(Dt.max(), Dg.max())
for a, M_, ttl in [(ax[0], Dg, "GBC posterior mean"), (ax[1], Dl, "Logit IV")]:
    im = a.imshow(M_, cmap=BLUES, vmin=0, vmax=vmax); a.grid(False)
    for i in range(6):
        for j in range(6):
            if i != j:
                a.text(j, i, f"{M_[i, j]:.2f}\n({Dt[i, j]:.2f})", ha="center", va="center", fontsize=7.5,
                       color="white" if M_[i, j] > 0.6 * vmax else INK)
    a.set_xticks(range(6)); a.set_xticklabels(PROD); a.set_yticks(range(6)); a.set_yticklabels(PROD)
    a.set_xlabel("to product k"); a.set_title(ttl)
ax[0].set_ylabel("from product j")                  # the second map shares the y axis, so its y label and ticks are dropped
ax[1].tick_params(labelleft=False)
cbar = fig.colorbar(im, ax=list(ax), orientation="horizontal", shrink=0.6, aspect=40, pad=0.04)
cbar.set_label("diversion ratio $D_{jk}$ (truth in brackets)", color=INK2); cbar.outline.set_linewidth(0.5)
fig.savefig(FIG + "fig_diversion.pdf"); plt.close(fig)

# ------------------------------------------------------------------ 7 merger price effects
fig = newfig(1.0, 5.2)
gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1])
ax = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
off = np.linspace(-0.3, 0.3, 4)
# gray scale: each method has its own fill level and hatch in the violins, and the legend carries the same patches
VIO = {"gbc2": ("#5a5a5a", ""), "jmr": ("#e6e6e6", "////"), "gmm": ("#a6a6a6", "xxxx"), "logit": ("#ffffff", "....")}
for (key, nm, col), o in zip(METH, off):
    dp = 100 * R[key]["draws"]["dp"].mean(1)
    vp = ax[0].violinplot([dp[:, j] for j in range(6)], positions=np.arange(6) + o, widths=0.18,
                          showextrema=False)
    for b in vp["bodies"]:
        b.set_facecolor(VIO[key][0]); b.set_edgecolor("#262626"); b.set_hatch(VIO[key][1]); b.set_alpha(1)
        b.set_linewidth(0.5)
tmark = 100 * TR["dp"].mean(0)
ax[0].scatter(range(6), tmark, marker="_", s=420, color="white", lw=3.4, zorder=4)   # halo under the marks
ax[0].scatter(range(6), tmark, marker="_", s=420, color=INK, lw=1.4, zorder=5)
th = Line2D([], [], color=INK, ls="", marker="_", markersize=18, markeredgewidth=1.4, label="truth")
ax[0].axhline(5, color=RED, lw=0.9, ls="--")
ax[0].annotate("SSNIP 5%", xy=(5.45, 5), xytext=(0, 2), textcoords="offset points", color=RED, fontsize=7.5,
               ha="right", va="bottom")
ax[0].set_xticks(range(6)); ax[0].set_xticklabels(PROD)
ax[0].set_ylabel("post merger price change, %")
y0_, y1_ = ax[0].get_ylim(); ax[0].set_ylim(y0_, y1_ + 0.32 * (y1_ - y0_))      # head room for the legend
ax[0].legend(handles=[Patch(facecolor=VIO[k_][0], edgecolor="#262626", hatch=VIO[k_][1], lw=0.5, label=nm_)
                      for k_, nm_, _ in METH] + [th], loc="upper center", ncol=3, columnspacing=1.2)
ax[0].set_title("Merger of firms A and B: price effects by product")
grid = np.linspace(-0.26, -0.06, 300)
for key, nm, col in METH:
    y = gaussian_kde(R[key]["draws"]["dcs"].mean(1))(grid)
    if key == "gbc2":
        ax[1].fill_between(grid, y, color=G_FILL, lw=0)
    ax[1].plot(grid, y, lw=1.5, label=nm, zorder=ZO[key], **STY[key])
ax[1].axvline(TR["dcs"].mean(), color=INK, ls=":", lw=1.1, label="truth", zorder=4, ymax=0.62)   # stops below the legend
ax[1].set_yticks([]); ax[1].set_ylim(0, ax[1].get_ylim()[1] * 1.7)      # head room for the legend
ax[1].legend(loc="upper left", ncol=3, handlelength=3, columnspacing=1.2)
ax[1].set_xlabel("change in consumer surplus per consumer, $10,000")
ax[1].set_title("Consumer harm"); ax[1].xaxis.set_major_locator(plt.MaxNLocator(6))
fig.savefig(FIG + "fig_merger.pdf"); plt.close(fig)

# ------------------------------------------------------------------ 8 GUPPI
fig = newfig(0.8, 3.5)
axs = fig.subplots(2, 2, sharex=True, sharey=True)
G = 100 * R["gbc2"]["draws"]["guppi"].mean(1)[:, :4]
Gl = 100 * R["logit"]["draws"]["guppi"].mean(1)[:, :4]
bins = np.linspace(0, max(G.max(), Gl.max()) * 1.05, 60)
# gray scale: one panel per product (small multiples) with one light gray fill and a black dotted truth line; the product
# and its P(GUPPI > 5%) are in the panel title, so the four overlapping translucent histograms and the legend are gone
for j, ax in enumerate(axs.ravel()):
    ax.hist(G[:, j], bins=bins, color="#cfcfcf", edgecolor="#6e6e6e", lw=0.3)
    ax.axvline(100 * TR["guppi"].mean(0)[j], color=INK, lw=1.3, ls=":")
    ax.axvline(5, color=RED, lw=1.1, ls="--")
    ax.set_title(f"{PROD[j]}: P(GUPPI > 5%) = {(G[:, j] > 5).mean():.2f}", fontsize=8.5)
    ax.set_yticks([])
for ax in axs.ravel():
    ax.text(5.4, ax.get_ylim()[1] * 0.95, "5%\nscreen", color=RED, fontsize=7.5, va="top")
fig.supxlabel("GUPPI, % of price (dotted: truth)")      # one label under both columns, the panels are too narrow for two
fig.suptitle("Posterior upward pricing pressure\nfor the merging products", color=INK)
fig.savefig(FIG + "fig_guppi.pdf"); plt.close(fig)

# ------------------------------------------------------------------ 9 direct estimands vs plug in, micro moments
fig = newfig(1.0, 4.4)
gs = fig.add_gridspec(2, 2)
ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])]
lax = fig.add_subplot(gs[1, 1]); lax.axis("off")             # the fourth cell holds the legend
ests = [("industry mean Lerner index", lambda dd: dd["lerner"].mean((1, 2)), TR["lerner"].mean(), 0),
        ("merging parties price change", lambda dd: dd["dp"][:, :, :4].mean((1, 2)), TR["dp"][:, :4].mean(), 1),
        (r"$\Delta$CS per consumer", lambda dd: dd["dcs"].mean(1), TR["dcs"].mean(), 2)]
for a, (nm, f, tv, i) in zip(ax, ests):
    xs = [R["gbc2"]["est"][:, i], f(R["gbc2"]["draws"]), R["gbc2_nomicro"]["est"][:, i]]
    lo_, hi_ = min(np.percentile(x, 0.5) for x in xs), max(np.percentile(x, 99.5) for x in xs)
    g = np.linspace(lo_, hi_, 300)
    for xx, col, ls, lab in [(xs[0], BLUE, "-", "GBC, estimand learned directly"),
                             (xs[1], VIOLET, "--", "GBC, plug in through parameters"),
                             (xs[2], MAGENTA, "-.", "GBC direct, no micro moments")]:
        y = gaussian_kde(xx)(g)
        a.plot(g, y, color=col, lw=1.5, ls=ls, label=lab)
        if ls == "-":
            a.fill_between(g, y, color=col, alpha=0.15, lw=0)
    a.axvline(tv, color=INK, ls=":", lw=1.1, label="truth"); a.set_yticks([]); a.set_xlabel(nm)
    a.xaxis.set_major_locator(plt.MaxNLocator(5))
h, l = ax[0].get_legend_handles_labels()
lax.legend(h, l, loc="center", handlelength=2.6, labelspacing=0.7)
fig.suptitle("Antitrust estimands as posterior targets", color=INK)
fig.savefig(FIG + "fig_estimands.pdf"); plt.close(fig)

# ------------------------------------------------------------------ 10 calibration
fig = newfig(1.0, 4.7)
ax = fig.subplots(2, 1)
cov = cal["cover"].mean(0)
lbl = [r"$\beta_0$", r"$\beta_1$", r"$\beta_2$", r"$\alpha$", r"$\sigma_p$", r"$\sigma_x$", r"$\gamma_0$",
       r"$\gamma_1$", r"$\gamma_2$", r"$\sigma_\xi$", r"$\sigma_\omega$", r"$\rho$"]
covn = cal["cover_naive"].mean(0)
# gray scale: in sample bars white with a hatch, cross fitted bars mid gray, both with a dark outline
ax[0].bar(np.arange(12) - 0.2, covn, color="#f5f5f5", edgecolor="#333333", lw=0.6, hatch="///", width=0.38,
          label="in sample residuals")
ax[0].bar(np.arange(12) + 0.2, cov, color="#a0a0a0", edgecolor="#333333", lw=0.6, width=0.38, label="cross fitted residuals")
ax[0].axhline(0.9, color=RED, ls="--", lw=1.0, zorder=4); ax[0].set_ylim(0.5, 1.12)
ax[0].set_xticks(range(12)); ax[0].set_xticklabels(lbl)
ax[0].set_ylabel("empirical coverage"); ax[0].set_title("Coverage of nominal 90% intervals")
ax[0].legend(loc="upper center", ncol=2)
nums["covnaive"] = f"{covn.mean():.2f}"
# gray scale: the three outlines differ by line style and gray (black solid, dark gray dashed, mid gray dotted); the uniform
# reference is a thin solid light line, so that it cannot be mistaken for the dashed outline
PIT = {3: (G_GBC, "-", 1.5), 4: (G_JMR, "--", 1.5), 5: (G_GMM, ":", 1.9)}
for i in [3, 4, 5]:
    ax[1].hist(cal["q"][:, i], bins=np.linspace(0, 1, 11), histtype="step", lw=PIT[i][2], ls=PIT[i][1],
               color=PIT[i][0], label=lbl[i], density=True)
ax[1].axhline(1, color="#b0b0b0", lw=0.8, zorder=1); ax[1].set_xlabel("posterior quantile of the true value")
ax[1].set_title("Probability integral transform"); ax[1].legend(handlelength=3, loc="lower center", ncol=3)
fig.savefig(FIG + "fig_calibration.pdf"); plt.close(fig)
nums["ncal"] = str(len(cal["cover"]))
nums["covavg"] = f"{cov.mean():.2f}"
nums["covmin"] = f"{cov.min():.2f}"

# ------------------------------------------------------------------ 11 decision curve
fig = newfig(0.82, 3.65)
gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.3])
ax = fig.add_subplot(gs[0]); lax = fig.add_subplot(gs[1]); lax.axis("off")
tau = np.linspace(0, 25, 300)
curves = [("GBC, direct estimand", 100 * R["gbc2"]["est"][:, 1], "gbc2"),
          ("Bayes inversion MCMC", 100 * R["jmr"]["draws"]["dp"][:, :, :4].mean((1, 2)), "jmr"),
          ("BLP GMM, Krinsky and Robb", 100 * R["gmm"]["draws"]["dp"][:, :, :4].mean((1, 2)), "gmm"),
          ("Logit IV", 100 * R["logit"]["draws"]["dp"][:, :, :4].mean((1, 2)), "logit")]
# gray scale: one gray and one line style per method (STY), black GBC on top
for nm, x, key in curves:
    ax.plot(tau, [(x > t).mean() for t in tau], lw=1.5, label=nm, zorder=ZO[key], **STY[key])
ax.axvline(100 * TR["dp"][:, :4].mean(), color=INK, ls=":", lw=1.1, label="true effect")
ax.axvline(5, color=RED, ls="--", lw=1.0); ax.text(5.3, 0.05, "SSNIP 5%", color=RED, fontsize=7.5)
ax.set_xlabel(r"price increase threshold $r$, %"); ax.set_ylabel(r"$\Pr(\Delta p/p > r \mid$ data$)$")
lax.legend(*ax.get_legend_handles_labels(), loc="center", ncol=2, handlelength=3, columnspacing=1.4)
ax.set_title("Posterior probability of a price increase\nabove each threshold")
fig.savefig(FIG + "fig_decision.pdf"); plt.close(fig)
for t, rn in [(10, "x"), (12, "xii"), (15, "xv")]:
    nums[f"pgbc{rn}"] = f"{(R['gbc2']['est'][:, 1] > t / 100).mean():.2f}"
    nums[f"plogit{rn}"] = f"{(R['logit']['draws']['dp'][:, :, :4].mean((1, 2)) > t / 100).mean():.2f}"

# ------------------------------------------------------------------ numbers for the text and table
def fmt(x, k=3):
    return f"{x:.{k}f}"

rows = []
for key, nm, _ in [("logit", "Logit IV", 0), ("gmm", "BLP GMM", 0), ("jmr", "Bayes inversion MCMC", 0),
                   ("gbc", "GBC, amortized round", 0), ("gbc2_nomicro", "GBC, local, no micro", 0),
                   ("gbc2", "GBC, local, with micro", 0)]:
    dd = R[key]["draws"]
    th = dd["th2"]
    L = dd["lerner"].mean((1, 2)); dp = 100 * dd["dp"][:, :, :4].mean((1, 2)); dc = dd["dcs"].mean(1)
    def cell(v, k=3):
        return f"{v.mean():.{k}f} ({v.std():.{k}f})"
    s1, s2 = (cell(th[:, 1]), cell(th[:, 2])) if key != "logit" else ("0 (fixed)", "0 (fixed)")
    c0 = cell(th[:, 0])
    if key == "logit":      # point estimate and sandwich standard error (rerun of October 2026, review item F1)
        c0 = f"{R['logit']['alpha']:.3f} ({R['logit']['se']:.3f})"
    elif key == "gmm":      # point estimate and sandwich standard errors, diag of the GMM covariance
        pt, se_ = np.asarray(R["gmm"]["th2"]), np.sqrt(np.diag(R["gmm"]["cov"]))
        c0, s1, s2 = [f"{pt[i]:.3f} ({se_[i]:.3f})" for i in range(3)]
    rows.append(" & ".join([nm, c0, s1, s2, cell(L), cell(dp, 2), cell(dc)]) + r" \\")
truth_row = " & ".join(["Truth", fmt(R["true"]["alpha"]), fmt(R["true"]["sp"]), fmt(R["true"]["sx"]),
                        fmt(TR["lerner"].mean()), f"{100 * TR['dp'][:, :4].mean():.2f}", fmt(TR["dcs"].mean())]) + r" \\"
with open("results/table_main.tex", "w") as f:
    f.write("\n".join(rows) + "\n\\midrule\n" + truth_row + "\n")
e = R["gbc2"]["est"]
nums.update(dict(
    ldir=f"{e[:, 0].mean():.3f}", ldirsd=f"{e[:, 0].std():.3f}", dpdir=f"{100 * e[:, 1].mean():.1f}",
    dpdirsd=f"{100 * e[:, 1].std():.1f}", csdir=f"{e[:, 2].mean():.3f}", csdirsd=f"{e[:, 2].std():.3f}",
    ltrue=f"{TR['lerner'].mean():.3f}", dptrue=f"{100 * TR['dp'][:, :4].mean():.1f}", cstrue=f"{TR['dcs'].mean():.3f}",
    spmicro=f"{R['gbc2']['draws']['th2'][:, 1].std():.3f}", spnomicro=f"{R['gbc2_nomicro']['draws']['th2'][:, 1].std():.3f}",
    pbert=f"{R['gbc']['pc'][0]:.3f}", jmracc=f"{R['jmr']['acc']:.2f}",
    pharm=f"{(R['gbc2']['est'][:, 1] > 0.05).mean():.2f}",
    dplogit=f"{100 * R['logit']['draws']['dp'][:, :, :4].mean((1, 2)).mean():.1f}",
    insideshare=f"{R['data']['s'].sum(1).mean():.2f}", meanprice=f"{R['data']['p'].mean():.2f}"))
for j in range(4):
    nums[f"guppi{'abcd'[j]}"] = f"{(G[:, j] > 5).mean():.2f}"
with open("results/numbers.tex", "w") as f:
    for k, v in nums.items():
        f.write(f"\\newcommand{{\\num{k}}}{{{v}}}\n")
print(nums)
