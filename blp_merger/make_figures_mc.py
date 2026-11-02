"""Figures and tables for the Monte Carlo study and the conduct misspecification experiment."""
import pickle, numpy as np, matplotlib, matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mc import TRUE

import os
R = pickle.load(open("results/mc.pkl", "rb"))
R05 = pickle.load(open("results/mc_sp005.pkl", "rb")) if os.path.exists("results/mc_sp005.pkl") else None
PART = pickle.load(open("results/mc_partial.pkl", "rb")) if os.path.exists("results/mc_partial.pkl") else None
FIG = "figures/"
RED, INK, INK2, MUTED, GRID = "#e34948", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
# The figures are drawn at the printed width. The text block is 5.25 in and each figure is 5.25 in wide and is included at
# \linewidth, so it prints at scale 1.0 and the sizes below are the printed sizes: base 9 pt, axis labels 9, tick labels 8,
# legends 8, panel titles 9, annotations 8. savefig.bbox is "standard" so that the PDF page is exactly the figure size
# (a tight bounding box would change the width and with it the scale).
W_IN = 5.25
# gray scale choices for black and white print, the same as the violins of fig_merger in make_figures.py: GBC is dark gray
# and solid, BLP GMM mid gray with an 'xxxx' hatch, the truth white with a dark outline; the legends carry the same patches
FILL = {"gbc": ("#5a5a5a", ""), "gmm": ("#a6a6a6", "xxxx"), "truth": ("#ffffff", "")}
EDGE = "#262626"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "figure.titlesize": 9, "figure.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "hatch.linewidth": 0.6,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "axes.titlecolor": INK, "savefig.bbox": "standard"})


def patch(k, label):
    return Patch(facecolor=FILL[k][0], hatch=FILL[k][1], edgecolor=EDGE, linewidth=0.7, label=label)


QN = [r"$\alpha$", r"$\sigma_p$", "Lerner", r"$\Delta p/p$ (\%)", r"$\Delta$CS"]
QFIG = [r"$\alpha$", r"$\sigma_p$", "industry Lerner", "merger price effect, %", r"$\Delta$CS"]
QSHORT = [r"$\alpha$", r"$\sigma_p$", "Lerner", r"$\Delta p/p$", r"$\Delta$CS"]
SCALE = [1, 1, 1, 100, 1]
nums, stats = {}, {}


def collect(T, RR=None):
    RR = R if RR is None else RR
    reps = RR[T]["reps"]
    out = {"gmm": [], "gbc": [], "gbcnm": [], "plug": []}
    nb, nx = 0, 0
    for rr in reps:
        if not rr["gmm_ok"] or len(rr["gmm_est"]) < 20:
            continue
        sp_true = rr.get("sp", TRUE["sp"])
        tv = np.array([TRUE["alpha"], sp_true] + list(rr["truth"]))
        se = np.sqrt(np.diag(rr["gmm_cov"]))[:2]
        gpt = np.concatenate([rr["gmm_th"][:2], np.median(rr["gmm_est"], axis=0)])
        glo = np.concatenate([rr["gmm_th"][:2] - 1.645 * se, np.percentile(rr["gmm_est"], 5, axis=0)])
        ghi = np.concatenate([rr["gmm_th"][:2] + 1.645 * se, np.percentile(rr["gmm_est"], 95, axis=0)])
        b = rr["gbc"][:, [0, 1, 3, 4, 5]]
        bpt, blo, bhi = b.mean(0), np.percentile(b, 5, axis=0), np.percentile(b, 95, axis=0)
        out["gmm"].append((gpt, glo, ghi, tv)); out["gbc"].append((bpt, blo, bhi, tv))
        bn = rr["gbc_nomicro"][:, [0, 1, 3, 4, 5]]
        out["gbcnm"].append((bn.mean(0), np.percentile(bn, 5, axis=0), np.percentile(bn, 95, axis=0), tv))
        if "gmm_plug" in rr and np.all(np.isfinite(rr["gmm_plug"])):
            out["plug"].append((np.concatenate([rr["gmm_th"][:2], rr["gmm_plug"]]), glo, ghi, tv))
        nb += rr["gmm_th"][1] < 0.05
        nx += np.percentile(rr["gmm_est"][:, 1], 95) > 1.0
    res = {}
    for k, v in out.items():
        if not v:
            continue
        pt, lo, hi, tv = [np.array(x) for x in zip(*v)]
        err = pt - tv; n_ = len(err); cv = ((tv >= lo) & (tv <= hi)).mean(0)
        rm = np.sqrt((err ** 2).mean(0))
        res[k] = dict(err=err, cover=cv, bias=err.mean(0), rmse=rm, width=np.median(hi - lo, axis=0), n=n_,
                      se_bias=err.std(0, ddof=1) / np.sqrt(n_), se_rmse=(err ** 2).std(0, ddof=1) / (2 * rm * np.sqrt(n_)),
                      se_cover=np.sqrt(cv * (1 - cv) / n_))
    res["n"] = len(out["gmm"]); res["nreps"] = len(reps); res["boundary"] = nb; res["explosive"] = nx
    return res


for T in [40, 15]:
    stats[T] = collect(T)
stats05 = {T: collect(T, R05) for T in [40, 15]} if R05 is not None else None

# ------------------------------------------------------------------ errors figure
# one row per quantity, T = 40 markets on the left and T = 15 on the right; horizontal violins, so that the error axis has
# the full panel width; a header row carries the number of markets
fig, axs = plt.subplots(6, 2, figsize=(W_IN, 6.3), layout="constrained", sharey="row",
                        gridspec_kw=dict(height_ratios=[0.2, 1, 1, 1, 1, 1]))
for c, T in enumerate([40, 15]):
    st = stats[T]
    h = axs[0, c]; h.axis("off")
    h.text(0.5, 0.45, f"T = {T} markets", ha="center", va="center", fontweight="bold", color=INK, transform=h.transAxes)
    for q in range(5):
        a = axs[q + 1, c]
        data = [SCALE[q] * st["gmm"]["err"][:, q], SCALE[q] * st["gbc"]["err"][:, q]]
        vp = a.violinplot(data, positions=[0, 1], widths=0.7, showextrema=False, orientation="horizontal")
        for b, k in zip(vp["bodies"], ["gmm", "gbc"]):
            b.set_facecolor(FILL[k][0]); b.set_edgecolor(EDGE); b.set_hatch(FILL[k][1]); b.set_alpha(1); b.set_linewidth(0.7)
        for i, dd in enumerate(data):
            a.scatter(dd, np.full(len(dd), i) + np.random.default_rng(q).uniform(-0.08, 0.08, len(dd)), s=5,
                      facecolor="white", edgecolor=EDGE, linewidth=0.3, zorder=3)
        a.axvline(0, color="white", lw=2.2, zorder=3.5); a.axvline(0, color=INK, lw=0.9, ls=":", zorder=4)   # white halo: the zero line shows on the dark violin
        a.set_ylim(1.6, -0.6); a.set_yticks([0, 1]); a.set_yticklabels(["BLP GMM", "GBC"])
        a.grid(axis="y", visible=False); a.tick_params(axis="y", length=0)
        a.set_title(QFIG[q], loc="left")
fig.suptitle("Sampling distribution of errors across Monte Carlo replications", color=INK)
fig.supxlabel("estimate minus truth")
fig.savefig(FIG + "fig_mc_errors.pdf"); plt.close(fig)

# ------------------------------------------------------------------ coverage figure
# T = 40 and T = 15 side by side, one legend above both panels that also names the dashed nominal level
fig = plt.figure(figsize=(W_IN, 3.0), layout="constrained")
gs = fig.add_gridspec(2, 2, height_ratios=[0.12, 1])
al = fig.add_subplot(gs[0, :]); al.axis("off")
ax = [fig.add_subplot(gs[1, 0])]
ax.append(fig.add_subplot(gs[1, 1], sharey=ax[0]))
for a, T in zip(ax, [40, 15]):
    st = stats[T]
    x = np.arange(5)
    for k, off in [("gmm", -0.2), ("gbc", 0.2)]:
        a.bar(x + off, st[k]["cover"], width=0.38, facecolor=FILL[k][0], hatch=FILL[k][1], edgecolor=EDGE, linewidth=0.7)
        a.errorbar(x + off, st[k]["cover"], yerr=2 * st[k]["se_cover"], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2, zorder=5)   # two Monte Carlo standard errors
    a.axhline(0.9, color=RED, ls="--", lw=1.0, zorder=4, path_effects=[pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])   # dashed white halo: the line shows over the dark bars
    a.set_xticks(x); a.set_xticklabels(QSHORT); a.set_ylim(0, 1.04)
    a.set_title(f"T = {T} markets")
ax[1].tick_params(labelleft=False)
ax[0].set_ylabel("coverage of nominal 90% intervals")
al.legend(handles=[patch("gmm", "BLP GMM (Wald, Krinsky and Robb)"), patch("gbc", "GBC posterior"),
                   Line2D([], [], color=RED, ls="--", lw=1.0, label="nominal 90%")],
          loc="center", ncol=3, handlelength=2.2, columnspacing=1.4)
fig.savefig(FIG + "fig_mc_coverage.pdf"); plt.close(fig)

# ------------------------------------------------------------------ table
def table_rows(stt, ncol_label="replications"):
    rows = []
    for T in [40, 15]:
        st = stt[T]
        rows.append(r"\multicolumn{9}{l}{\textit{T = %d markets, %d replications}} \\" % (T, st["n"]))
        for q in range(5):
            g, b = st["gmm"], st["gbc"]
            sc = SCALE[q]
            rows.append(f"{QN[q]} & {sc * g['bias'][q]:.3f} & {sc * g['rmse'][q]:.3f} & {g['cover'][q]:.2f} & {sc * g['width'][q]:.3f}"
                        f" & {sc * b['bias'][q]:.3f} & {sc * b['rmse'][q]:.3f} & {b['cover'][q]:.2f} & {sc * b['width'][q]:.3f} \\\\")
        if T == 40:
            rows.append(r"\midrule")
    return rows


def table_se_rows(stt):
    """Monte Carlo standard errors of bias, RMSE and coverage, same layout as the main table (lengths: not computed)."""
    rows = []
    for T in [40, 15]:
        st = stt[T]
        rows.append(r"\multicolumn{9}{l}{\textit{T = %d markets, %d replications}} \\" % (T, st["n"]))
        for q in range(5):
            g, b = st["gmm"], st["gbc"]
            sc = SCALE[q]
            rows.append(f"{QN[q]} & {sc * g['se_bias'][q]:.3f} & {sc * g['se_rmse'][q]:.3f} & {g['se_cover'][q]:.2f} & --"
                        f" & {sc * b['se_bias'][q]:.3f} & {sc * b['se_rmse'][q]:.3f} & {b['se_cover'][q]:.2f} & -- \\\\")
        if T == 40:
            rows.append(r"\midrule")
    return rows


def table_plug_rows(stt):
    """GMM point prediction by plugging in the point estimate against the Krinsky and Robb median (bias, RMSE; the parameters coincide)."""
    rows = []
    for T in [40, 15]:
        st = stt[T]; p_, g = st["plug"], st["gmm"]
        rows.append(r"\multicolumn{5}{l}{\textit{T = %d markets, %d replications with a plug in}} \\" % (T, p_["n"]))
        for q in (2, 3, 4):
            sc = SCALE[q]
            rows.append(f"{QN[q]} & {sc * g['bias'][q]:.3f} & {sc * g['rmse'][q]:.3f} & {sc * p_['bias'][q]:.3f} & {sc * p_['rmse'][q]:.3f} \\\\")
        if T == 40:
            rows.append(r"\midrule")
    return rows


def wr(fn, rows):
    with open(fn, "w") as f:
        f.write("\n".join(rows) + "\n")


wr("results/table_mc.tex", table_rows(stats))
wr("results/table_mc_se.tex", table_se_rows(stats))
wr("results/table_mc_plug.tex", table_plug_rows(stats))
if stats05 is not None:
    wr("results/table_mc_sp005.tex", table_rows(stats05))
    wr("results/table_mc_sp005_se.tex", table_se_rows(stats05))
SUF = {40: "xl", 15: "xv"}
for T in [40, 15]:
    st = stats[T]
    nums[f"mcn{SUF[T]}"] = str(st["n"]); nums[f"mcbound{SUF[T]}"] = str(st["boundary"]); nums[f"mcexpl{SUF[T]}"] = str(st["explosive"])
    for q, k in zip([1, 3], ["sp", "dp"]):
        nums[f"mc{k}rmsegmm{SUF[T]}"] = f"{SCALE[q] * st['gmm']['rmse'][q]:.3f}"
        nums[f"mc{k}rmsegbc{SUF[T]}"] = f"{SCALE[q] * st['gbc']['rmse'][q]:.3f}"
        nums[f"mc{k}covgmm{SUF[T]}"] = f"{st['gmm']['cover'][q]:.2f}"
        nums[f"mc{k}covgbc{SUF[T]}"] = f"{st['gbc']['cover'][q]:.2f}"
        nums[f"mc{k}rmsenm{SUF[T]}"] = f"{SCALE[q] * st['gbcnm']['rmse'][q]:.3f}"
        nums[f"mc{k}covnm{SUF[T]}"] = f"{st['gbcnm']['cover'][q]:.2f}"
        nums[f"mc{k}lengbc{SUF[T]}"] = f"{SCALE[q] * st['gbc']['width'][q]:.2f}"
        nums[f"mc{k}lengmm{SUF[T]}"] = f"{SCALE[q] * st['gmm']['width'][q]:.2f}"
        nums[f"mc{k}rmseplug{SUF[T]}"] = f"{SCALE[q] * st['plug']['rmse'][q]:.3f}"
        nums[f"mc{k}biasgmm{SUF[T]}"] = f"{SCALE[q] * st['gmm']['bias'][q]:.3f}"
        nums[f"mc{k}biasplug{SUF[T]}"] = f"{SCALE[q] * st['plug']['bias'][q]:.3f}"
        nums[f"mc{k}rmsesegmm{SUF[T]}"] = f"{SCALE[q] * st['gmm']['se_rmse'][q]:.3f}"
        nums[f"mc{k}rmsesegbc{SUF[T]}"] = f"{SCALE[q] * st['gbc']['se_rmse'][q]:.3f}"
    nums[f"mccovse{SUF[T]}"] = f"{np.sqrt(0.9 * 0.1 / st['n']):.3f}"
    for q, k in zip([2], ["ler"]):
        nums[f"mc{k}covgmm{SUF[T]}"] = f"{st['gmm']['cover'][q]:.2f}"
        nums[f"mc{k}covgbc{SUF[T]}"] = f"{st['gbc']['cover'][q]:.2f}"
        nums[f"mc{k}covsegmm{SUF[T]}"] = f"{st['gmm']['se_cover'][q]:.3f}"
    if stats05 is not None:
        s5 = stats05[T]
        nums[f"mcfive{SUF[T]}n"] = str(s5["n"]); nums[f"mcfive{SUF[T]}bound"] = str(s5["boundary"])
        for q, k in zip([1, 3], ["sp", "dp"]):
            nums[f"mcfive{k}rmsegmm{SUF[T]}"] = f"{SCALE[q] * s5['gmm']['rmse'][q]:.3f}"
            nums[f"mcfive{k}rmsegbc{SUF[T]}"] = f"{SCALE[q] * s5['gbc']['rmse'][q]:.3f}"
            nums[f"mcfive{k}covgmm{SUF[T]}"] = f"{s5['gmm']['cover'][q]:.2f}"
            nums[f"mcfive{k}covgbc{SUF[T]}"] = f"{s5['gbc']['cover'][q]:.2f}"

# ------------------------------------------------------------------ misspecification
# the two quantities are stacked (each gets the full width, so the long condition labels fit), one legend above them
M = R["misspec"]
fig = plt.figure(figsize=(W_IN, 5.2), layout="constrained")
gs = fig.add_gridspec(3, 1, height_ratios=[0.12, 1, 1])
al = fig.add_subplot(gs[0]); al.axis("off")
ax = [fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
labs = {"collusion": "data from joint monopoly", "perfect": "data from perfect competition"}
for j, (q, ttl, sc, fmt_) in enumerate([(0, "industry Lerner index", 1, "{:.2f}"),
                                         (1, "merger price effect, %", 100, "{:.1f}")]):
    a = ax[j]; tops, allv, ticks = [], [0.0], []
    for i, k in enumerate(["collusion", "perfect"]):
        m = M[k]; g = m["gbc_est"]
        vals = [abs(sc * m["truth"][q]), sc * m["gmm_bertrand"][q], abs(sc * g[:, q].mean())]
        lo, hi = sc * np.percentile(g[:, q], 5), sc * np.percentile(g[:, q], 95)
        xs = i * 4 + np.arange(3)
        for x, v, kk in zip(xs, vals, ["truth", "gmm", "gbc"]):
            a.bar(x, v, width=0.75, facecolor=FILL[kk][0], hatch=FILL[kk][1], edgecolor=EDGE, linewidth=0.8)
        yerr = [[max(vals[2] - lo, 0)], [max(hi - vals[2], 0)]]
        a.errorbar(xs[2], vals[2], yerr=yerr, color="white", capsize=4.5, capthick=3, elinewidth=3, fmt="none", zorder=4)   # halo
        a.errorbar(xs[2], vals[2], yerr=yerr, color=INK, capsize=3.5, capthick=1.1, elinewidth=1.1, fmt="none", zorder=5)
        ytop = [vals[0], vals[1], max(vals[2], hi)]          # the GBC value sits above its interval
        for x, v, yt in zip(xs, vals, ytop):
            a.annotate(fmt_.format(v), (x, yt), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", color=INK)
        tops += ytop; allv += vals
        lab = labs[k]
        if j == 0:
            pc = m["gbc_pc"][["bertrand", "collusion", "perfect"].index(k)]
            lab += f"\nP(true conduct) = {pc:.2f}"
        ticks.append(lab)
    a.set_xticks([1, 5]); a.set_xticklabels(ticks); a.set_ylim(min(0, 1.2 * min(allv)), 1.2 * max(tops))
    a.set_title(ttl, loc="left"); a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
al.legend(handles=[patch("truth", "truth"), patch("gmm", "BLP GMM assuming Bertrand Nash"),
                   patch("gbc", "GBC, conduct estimated")], loc="center", ncol=3, handlelength=1.6, columnspacing=1.2)
fig.savefig(FIG + "fig_misspec.pdf"); plt.close(fig)
for k in ["collusion", "perfect"]:
    m = M[k]
    nums[f"ms{k}alpha"] = f"{m['gmm_th'][0]:.2f}"
    nums[f"ms{k}true"] = f"{abs(100 * m['truth'][1]):.1f}"
    nums[f"ms{k}gmm"] = f"{100 * m['gmm_bertrand'][1]:.1f}"
    nums[f"ms{k}gbc"] = f"{abs(100 * m['gbc_est'][:, 1].mean()):.1f}"
    nums[f"ms{k}ltrue"] = f"{m['truth'][0]:.3f}"
    nums[f"ms{k}lgmm"] = f"{m['gmm_bertrand'][0]:.3f}"
    nums[f"ms{k}lgbc"] = f"{m['gbc_est'][:, 0].mean():.3f}"
    nums[f"ms{k}pc"] = f"{m['gbc_pc'][['bertrand', 'collusion', 'perfect'].index(k)]:.3f}"
# ------------------------------------------------------------------ partial rival internalization (kappa = 0.5), stress test
if PART is not None:
    tr = np.array([x["truth"] for x in PART]); trb = np.array([x["truth_bertrand"] for x in PART])
    gm = np.array([x["gmm_bertrand"] for x in PART]); gb = np.array([x["gbc_est"].mean(0) for x in PART]); pc = np.array([x["gbc_pc"] for x in PART])
    n_ = len(PART); rows = []
    for nm, v in [("truth, kappa = 0.5", tr), ("Bertrand analyst at the true $\\theta_2$", trb), ("BLP GMM assuming Bertrand", gm), ("GBC, conduct estimated", gb)]:
        rows.append(f"{nm} & {np.nanmean(v[:, 0]):.3f} & {100 * np.nanmean(v[:, 1]):.2f} & {np.nanmean(v[:, 2]):.4f} \\\\")
    wr("results/table_mc_partial.tex", rows)
    for k, nm in [("gmm", gm), ("gbc", gb)]:
        e = nm - tr
        nums[f"mspart{k}biasdp"] = f"{100 * np.nanmean(e[:, 1]):.2f}"; nums[f"mspart{k}rmsedp"] = f"{100 * np.sqrt(np.nanmean(e[:, 1] ** 2)):.2f}"
        nums[f"mspart{k}biasler"] = f"{np.nanmean(e[:, 0]):.3f}"; nums[f"mspart{k}rmseler"] = f"{np.sqrt(np.nanmean(e[:, 0] ** 2)):.3f}"
        nums[f"mspart{k}sedp"] = f"{100 * np.nanstd(e[:, 1], ddof=1) / np.sqrt(n_):.2f}"
    nums["msparttruedp"] = f"{100 * tr[:, 1].mean():.2f}"; nums["msparttruebertdp"] = f"{100 * trb[:, 1].mean():.2f}"
    nums["msparttrueler"] = f"{tr[:, 0].mean():.3f}"; nums["mspartn"] = str(n_)
    for j, c_ in enumerate(["bert", "coll", "perf"]):
        nums[f"mspartpc{c_}"] = f"{pc[:, j].mean():.2f}"
with open("results/numbers_mc.tex", "w") as f:
    for k, v in nums.items():
        f.write(f"\\newcommand{{\\num{k}}}{{{v}}}\n")
print(nums)
print(open("results/table_mc.tex").read())
