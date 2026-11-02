import numpy as np, pandas as pd, json, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from causaldata import castle
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(2007); FIG = "figs/"; R = {}

def group_hs(Z, y, groups, iters=2500, burn=700):
    """Gibbs sampler for y = Z b + e. groups[j] = -1 means flat prior, otherwise index of a horseshoe group
    with its own global scale. Returns draws of b."""
    n, q = Z.shape; groups = np.asarray(groups); G = groups.max() + 1
    pen = groups >= 0
    ZtZ = Z.T @ Z; Zty = Z.T @ y; yty = y @ y
    lam2 = np.ones(q); nu = np.ones(q); tau2 = np.ones(G) * 0.1; xi = np.ones(G); s2 = np.var(y)
    out = []
    for it in range(iters):
        pv = np.where(pen, lam2 * tau2[np.maximum(groups, 0)], np.inf)
        A = ZtZ + np.diag(np.where(pen, 1 / pv, 1e-8)); L = np.linalg.cholesky(A)
        b = np.linalg.solve(A, Zty) + np.sqrt(s2) * np.linalg.solve(L.T, rng.normal(size=q))
        rss = yty - 2 * b @ Zty + b @ ZtZ @ b
        bp = b[pen]; tp = tau2[groups[pen]]
        s2 = 1 / rng.gamma((n + pen.sum()) / 2, 1 / (rss / 2 + (bp ** 2 / (lam2[pen] * tp)).sum() / 2))
        lam2[pen] = 1 / rng.gamma(1, 1 / (1 / nu[pen] + bp ** 2 / (2 * tp * s2)))
        nu[pen] = 1 / rng.gamma(1, 1 / (1 + 1 / lam2[pen]))
        for g in range(G):
            s = groups == g
            tau2[g] = 1 / rng.gamma((s.sum() + 1) / 2, 1 / (1 / xi[g] + (b[s] ** 2 / lam2[s]).sum() / (2 * s2)))
            xi[g] = 1 / rng.gamma(1, 1 / (1 + 1 / tau2[g]))
        if it >= burn: out.append(b.copy())
    return np.array(out)

def build(unit, year, cohort, y, leads=(-5, -4, -3, -2), trends=True):
    """Design: unit FE, year FE (flat); flat event time path theta_e; cohort by event time deviations from it (HS group 0);
    leads (HS group 1); unit specific linear trends (HS group 2). cohort = adoption year or 0 if never."""
    units = np.unique(unit); years = np.unique(year)
    cols, groups, names = [], [], []
    for u in units: cols.append((unit == u).astype(float)); groups.append(-1); names.append(("unit", u))
    for t in years[1:]: cols.append((year == t).astype(float)); groups.append(-1); names.append(("year", t))
    post = ((cohort > 0) & (year >= cohort)).astype(float)
    ev = np.where(cohort > 0, year - cohort, -999)
    evs = sorted({e for e in ev if e >= 0})
    for e in evs:                                   # flat event time path theta_e
        cols.append((ev == e).astype(float)); groups.append(-1); names.append(("path", e))
    cells = sorted({(c, e) for c, e in zip(cohort, ev) if c > 0 and e >= 0})
    for (c, e) in cells:                            # horseshoe cohort deviations from the path
        cols.append(((cohort == c) & (ev == e)).astype(float)); groups.append(0); names.append(("dev", (c, e)))
    for l in leads:
        cols.append((ev == l).astype(float)); groups.append(1); names.append(("lead", l))
    if trends:
        tc = (year - year.mean()) / year.std()
        for u in units: cols.append((unit == u) * tc); groups.append(2); names.append(("trend", u))
    Z = np.column_stack(cols); return Z, np.array(groups), names, post, cells


def att_from(dr, names, cells, coh, ev):
    pidx = {nm[1]: i for i, nm in enumerate(names) if nm[0] == "path"}
    didx = {nm[1]: i for i, nm in enumerate(names) if nm[0] == "dev"}
    w = np.array([np.sum((coh == c) & (ev == e)) for (c, e) in cells], float)
    tot = np.zeros(dr.shape[0])
    for k, (c, e) in enumerate(cells):
        tot += w[k] * (dr[:, pidx[e]] + dr[:, didx[(c, e)]])
    return tot / w.sum()

def twfe(unit, year, y, post):
    U = pd.get_dummies(unit).values.astype(float); T = pd.get_dummies(year).values[:, 1:].astype(float)
    Z = np.column_stack([post, U, T]); return np.linalg.lstsq(Z, y, rcond=None)[0][0]

def cs_att(unit, year, cohort, y):
    """Cohort by time ATT with never treated controls and base period cohort - 1, averaged over treated cells."""
    df = pd.DataFrame(dict(u=unit, t=year, c=cohort, y=y)); est = []; w = []
    for c in sorted(df.c[df.c > 0].unique()):
        base = c - 1
        for t in sorted(df.t.unique()):
            if t < c: continue
            g = df[df.c == c]; n_ = df[df.c == 0]
            d1 = g[g.t == t].set_index("u").y - g[g.t == base].set_index("u").y
            d0 = n_[n_.t == t].set_index("u").y - n_[n_.t == base].set_index("u").y
            est.append(d1.mean() - d0.mean()); w.append(len(d1))
    return float(np.average(est, weights=w))

def plot_castle_raw(y, year, coh):
    """Raw means of the log homicide rate by adoption cohort (castle_raw.pdf). Gray scale print: the series differ by line style,
    gray level and marker, so that the legend can be read without colour."""
    sty = {"never adopted": dict(color="k", ls="-", marker="o"), "adopted 2006": dict(color="0.55", ls="--", marker="s"),
           "adopted 2007 or later": dict(color="0.3", ls=":", marker="^"), "adopted 2005": dict(color="0.75", ls="-.", marker="D")}
    fig, ax = plt.subplots(figsize=(5, 3.1))
    for lab, s in [("never adopted", coh == 0), ("adopted 2006", coh == 2006), ("adopted 2007 or later", coh >= 2007), ("adopted 2005", coh == 2005)]:
        if s.sum() == 0: continue
        g = pd.Series(y[s]).groupby(year[s]).mean(); ax.plot(g.index, g.values, ms=3.5, label=lab, **sty[lab])
    ax.set_xlabel("Year"); ax.set_ylabel("Mean log homicide rate"); ax.legend(frameon=False, fontsize=7, handlelength=3.2)
    fig.tight_layout(); fig.savefig(FIG + "castle_raw.pdf"); plt.close()

import sys
if "--figs-only" in sys.argv:    # redraw castle_raw.pdf only: no simulation, no sampler, nothing is written to results/ (needs the repository root on the path for the causaldata stub)
    d = castle.load_pandas().data.copy(); first = d[d.post > 0].groupby("sid").year.min(); d["cohort"] = d.sid.map(first).fillna(0).astype(int)
    plot_castle_raw(d.l_homicide.values.astype(float), d.year.values, d.cohort.values); sys.exit(0)

# ---------------- Simulation: staggered adoption, dynamic effects, a few trend violators
res = []
for r in range(30):
    N, T = 40, 16; years = np.arange(T); unit = np.repeat(np.arange(N), T); year = np.tile(years, N)
    ccode = np.repeat(rng.choice([4, 8, 12, 0], N, p=[0.25, 0.25, 0.25, 0.25]), T)
    alpha = np.repeat(rng.normal(0, 1, N), T); gamma = np.tile(np.cumsum(rng.normal(0, 0.2, T)), N)
    ev = np.where(ccode > 0, year - ccode, -999)
    tau = np.where((ccode > 0) & (ev >= 0), 0.5 + 0.15 * ev, 0.0)        # effects grow with exposure
    viol = np.repeat(rng.random(N) < 0.1, T) * 0.15 * (year - T / 2)       # about 10 percent of units trend away
    y = alpha + gamma + tau + viol + rng.normal(0, 0.5, N * T)
    post = ((ccode > 0) & (year >= ccode)).astype(float)
    att_true = tau[post == 1].mean()
    tw = twfe(unit, year, y, post); cs = cs_att(unit, year, ccode, y)
    Z, grp, names, _, cells = build(unit, year, ccode, y)
    dr = group_hs(Z, y, grp, iters=1500, burn=500)
    att_draws = att_from(dr, names, cells, ccode, ev)
    lo, hi = np.quantile(att_draws, [0.025, 0.975])
    res.append([tw - att_true, cs - att_true, att_draws.mean() - att_true, float(lo <= att_true <= hi), hi - lo])
    print(r, np.round(res[-1], 3), flush=True)
res = np.array(res)
R["sim"] = dict(bias=res[:, :3].mean(0).tolist(), rmse=np.sqrt((res[:, :3] ** 2).mean(0)).tolist(), cover=float(res[:, 3].mean()), width=float(res[:, 4].mean()))
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.boxplot([res[:, 0], res[:, 1], res[:, 2]], tick_labels=["two way fixed effects", "cohort by time", "horseshoe DiD"])
ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("Estimate minus true ATT")
fig.tight_layout(); fig.savefig(FIG + "did_sim.pdf"); plt.close()

# ---------------- Castle doctrine
d = castle.load_pandas().data.copy()
first = d[d.post > 0].groupby("sid").year.min()
d["cohort"] = d.sid.map(first).fillna(0).astype(int)
unit = d.sid.values; year = d.year.values; y = d.l_homicide.values.astype(float); coh = d.cohort.values
postb = ((coh > 0) & (year >= coh)).astype(float)
R["castle_n_states"] = int(len(np.unique(unit))); R["castle_treated_states"] = int((first > 0).sum())
R["castle_cohorts"] = {int(k): int(v) for k, v in first.value_counts().sort_index().items()}
R["castle_twfe"] = float(twfe(unit, year, y, postb)); R["castle_cs"] = cs_att(unit, year, coh, y)
Z, grp, names, _, cells = build(unit, year, coh, y, leads=(-5, -4, -3, -2))
dr = group_hs(Z, y, grp, iters=4000, burn=1000)
ev = np.where(coh > 0, year - coh, -999)
att = att_from(dr, names, cells, coh, ev)
cell_w = np.array([np.sum((coh == c) & (ev == e)) for (c, e) in cells])
pidx = {nm[1]: i for i, nm in enumerate(names) if nm[0] == "path"}
dev_idx = [i for i, nm in enumerate(names) if nm[0] == "dev"]
R["castle_hs_att"] = [float(att.mean()), float(att.std()), float(np.quantile(att, .025)), float(np.quantile(att, .975))]
R["castle_hs_prob_pos"] = float((att > 0).mean())
# event study from draws: tbar + average dev at each event time; leads
es = {}
for e in range(0, 6):
    idx = [k for k, (c, ee) in enumerate(cells) if ee == e]
    if not idx: continue
    w = cell_w[idx]; draws = dr[:, pidx[e]] + dr[:, [dev_idx[k] for k in idx]] @ w / w.sum()
    es[e] = draws
for l in (-5, -4, -3, -2):
    j = [i for i, nm in enumerate(names) if nm == ("lead", l)][0]; es[l] = dr[:, j]
es[-1] = np.zeros(len(att))
ks = sorted(es); m = [es[k].mean() for k in ks]; lo = [np.quantile(es[k], .025) for k in ks]; hi = [np.quantile(es[k], .975) for k in ks]
R["castle_event"] = {"e": ks, "mean": list(map(float, m)), "lo": list(map(float, lo)), "hi": list(map(float, hi))}
trend_idx = [i for i, nm in enumerate(names) if nm[0] == "trend"]
R["castle_trend_absmax"] = float(np.abs(dr[:, trend_idx].mean(0)).max())
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.9))
axs[0].errorbar(ks, m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt="o", capsize=3); axs[0].axhline(0, color="k", lw=0.5); axs[0].axvline(-0.5, color="grey", ls=":")
axs[0].set_xlabel("Years since adoption"); axs[0].set_ylabel("Effect on log homicide rate")
axs[1].hist(att, bins=50, density=True, histtype="step"); axs[1].axvline(R["castle_twfe"], color="k", ls="--", lw=0.8, label="TWFE")
axs[1].axvline(R["castle_cs"], color="C3", ls=":", lw=0.9, label="cohort by time"); axs[1].legend(frameon=False, fontsize=7)
axs[1].set_xlabel("ATT, log homicide rate")
fig.tight_layout(); fig.savefig(FIG + "castle.pdf"); plt.close()
# raw trends plot
plot_castle_raw(y, year, coh)
json.dump(R, open("results/res_did.json", "w"), indent=1, default=float); print(json.dumps(R, default=float)[:2500])
