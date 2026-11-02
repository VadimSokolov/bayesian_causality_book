"""Checks behind the Chapter 15 numbers that book_scripts/did_hs.py does not write (book_tex/chapters/m13.tex).

Run from the repository root (about a minute):  export PYTHONPATH=. && python book_scripts/did_checks.py
Writes results/res_did_checks.json.  The sampler and the estimators are those of did_hs.py: the part of that file above its
simulation section is executed here, so the two scripts cannot drift apart.

sim_with_departures, sim_no_departures
    The 30 panels of did_hs.py (40 units, 16 periods, seed 2007), once as published and once with the trend of the violating
    units multiplied by zero.  The random stream is unchanged, so the panels are the same in both runs.  For two way fixed effects,
    the cohort by time estimator and the horseshoe (in that order): bias, RMSE, coverage of the 95 percent interval, mean interval
    width, the squared error gap between the cohort by time estimator and the horseshoe, and the two way fixed effects cell weights.
    gap_share_without_departures is the gap without departures over the gap with them (about two thirds in the chapter).
    bias_se is the Monte Carlo standard error of each bias over the 30 panels (the standard deviation of the 30 errors over the root of 30).
sim_single_effect
    The first version of the horseshoe estimator, which Chapter 15 reports as a design error: one flat common effect for every treated
    cell and a horseshoe on the deviations of every cohort by event time cell from it, with the leads and the unit trends as in the
    published estimator.  It is fitted to the same 30 panels as sim_with_departures (the panels are kept in memory, so the sampler's
    random stream cannot change them): bias, RMSE, coverage of the 95 percent interval and mean interval width of the effect on the treated.
castle
    The castle doctrine fit of did_hs.py (4,000 sweeps, 1,000 discarded) as published (as_published), without the unit trend block
    (no_trend_block) and with flat priors on the leads (flat_leads).  Each fit starts a fresh generator seeded 2007, so the third
    decimal can differ from numbers obtained after the simulation in one process.  Also the two way fixed effects estimate, the
    cohort by time estimate and the cell weights of the two way fixed effects coefficient in the castle panel.
"""
import numpy as np, pandas as pd, json

SRC = "book_scripts/did_hs.py"; MARK = "# ---------------- Simulation"
src = open(SRC).read(); assert src.count(MARK) == 1
exec(compile(src.split(MARK)[0], SRC, "exec"))     # imports, rng, group_hs, build, att_from, twfe, cs_att
R = {}
PANELS = []


def reseed():
    global rng
    rng = np.random.default_rng(2007)


def cell_weights(unit, year, post, coh):
    """Weights of the two way fixed effects coefficient on the treated cells (cohort:event time); they sum to one."""
    U = pd.get_dummies(unit).values.astype(float); Yd = pd.get_dummies(year).values[:, 1:].astype(float)
    X = np.column_stack([U, Yd]); dres = post - X @ np.linalg.lstsq(X, post, rcond=None)[0]
    ev = np.where(coh > 0, year - coh, -999)
    cells = sorted({(int(c), int(e)) for c, e in zip(coh, ev) if c > 0 and e >= 0})
    den = dres[post == 1].sum()
    return {f"{c}:{e}": float(dres[(coh == c) & (ev == e)].sum() / den) for c, e in cells}


def simulate(viol_scale):
    """The simulation loop of did_hs.py; viol_scale = 0 switches the trend departures off without changing the random stream."""
    reseed(); res, wmin, w4 = [], [], []
    for r in range(30):
        N, T = 40, 16; years = np.arange(T); unit = np.repeat(np.arange(N), T); year = np.tile(years, N)
        ccode = np.repeat(rng.choice([4, 8, 12, 0], N, p=[0.25, 0.25, 0.25, 0.25]), T)
        alpha = np.repeat(rng.normal(0, 1, N), T); gamma = np.tile(np.cumsum(rng.normal(0, 0.2, T)), N)
        ev = np.where(ccode > 0, year - ccode, -999)
        tau = np.where((ccode > 0) & (ev >= 0), 0.5 + 0.15 * ev, 0.0)
        viol = np.repeat(rng.random(N) < 0.1, T) * 0.15 * (year - T / 2) * viol_scale
        y = alpha + gamma + tau + viol + rng.normal(0, 0.5, N * T)
        post = ((ccode > 0) & (year >= ccode)).astype(float)
        att_true = tau[post == 1].mean()
        if viol_scale == 1.0: PANELS.append((unit, year, ccode, y, att_true, ev))
        tw = twfe(unit, year, y, post); cs = cs_att(unit, year, ccode, y)
        Z, grp, names, _, cells = build(unit, year, ccode, y)
        dr = group_hs(Z, y, grp, iters=1500, burn=500)
        draws = att_from(dr, names, cells, ccode, ev)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        res.append([tw - att_true, cs - att_true, draws.mean() - att_true, float(lo <= att_true <= hi), hi - lo, att_true])
        w = cell_weights(unit, year, post, ccode)
        wmin.append(min(w.values())); w4.append([w.get(f"4:{e}") for e in range(8, 12)])
        print(viol_scale, r, np.round(res[-1], 3), flush=True)
    res = np.array(res); rmse = np.sqrt((res[:, :3] ** 2).mean(0))
    return dict(bias=res[:, :3].mean(0).tolist(), rmse=rmse.tolist(), cover=float(res[:, 3].mean()), cover_count=int(res[:, 3].sum()),
                bias_se=(res[:, :3].std(0, ddof=1) / np.sqrt(len(res))).tolist(),
                width=float(res[:, 4].mean()), att_true=float(res[:, 5].mean()),
                sq_gap_cs_minus_hs=float(rmse[1] ** 2 - rmse[2] ** 2),
                twfe_weight_min_by_panel=wmin, twfe_weight_period4_e8_to_11_by_panel=w4)


def build_single(unit, year, cohort, y):
    """First version of the estimator: one flat common effect for every treated cell instead of the event time path."""
    Z, grp, names, post, cells = build(unit, year, cohort, y)
    keep = [i for i, nm in enumerate(names) if nm[0] != "path"]
    return np.column_stack([post, Z[:, keep]]), np.concatenate([[-1], grp[keep]]), [("tbar", None)] + [names[i] for i in keep], post, cells


def att_single(dr, names, cells, coh, ev):
    didx = {nm[1]: i for i, nm in enumerate(names) if nm[0] == "dev"}
    w = np.array([np.sum((coh == c) & (ev == e)) for (c, e) in cells], float)
    return dr[:, 0] + sum(w[k] * dr[:, didx[(c, e)]] for k, (c, e) in enumerate(cells)) / w.sum()


def simulate_single():
    """The panels of simulate(1.0) fitted with the single effect parametrization."""
    assert len(PANELS) == 30
    reseed(); res = []
    for r, (unit, year, ccode, y, att_true, ev) in enumerate(PANELS):
        Z, grp, names, _, cells = build_single(unit, year, ccode, y)
        dr = group_hs(Z, y, grp, iters=1500, burn=500)
        draws = att_single(dr, names, cells, ccode, ev)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        res.append([draws.mean() - att_true, float(lo <= att_true <= hi), hi - lo, att_true])
        print("single", r, np.round(res[-1], 3), flush=True)
    res = np.array(res)
    return dict(bias=float(res[:, 0].mean()), rmse=float(np.sqrt((res[:, 0] ** 2).mean())), bias_se=float(res[:, 0].std(ddof=1) / np.sqrt(len(res))),
                cover=float(res[:, 1].mean()), cover_count=int(res[:, 1].sum()), width=float(res[:, 2].mean()), att_true=float(res[:, 3].mean()),
                bias_over_true_effect=float(res[:, 0].mean() / res[:, 3].mean()))


def castle_data():
    d = castle.load_pandas().data.copy()
    first = d[d.post > 0].groupby("sid").year.min()
    d["cohort"] = d.sid.map(first).fillna(0).astype(int)
    return d.sid.values, d.year.values, d.l_homicide.values.astype(float), d.cohort.values


def castle_fit(variant, unit, year, y, coh):
    reseed()
    ev = np.where(coh > 0, year - coh, -999)
    Z, grp, names, _, cells = build(unit, year, coh, y, leads=(-5, -4, -3, -2), trends=(variant != "no_trend_block"))
    if variant == "flat_leads":
        grp = np.where(grp == 1, -1, np.where(grp == 2, 1, grp))       # leads flat; the trend block keeps its horseshoe
    dr = group_hs(Z, y, grp, iters=4000, burn=1000)
    att = att_from(dr, names, cells, coh, ev)
    pidx = {nm[1]: i for i, nm in enumerate(names) if nm[0] == "path"}
    dev_idx = [i for i, nm in enumerate(names) if nm[0] == "dev"]
    cell_w = np.array([np.sum((coh == c) & (ev == e)) for (c, e) in cells])
    es = {}
    for e in range(0, 6):
        idx = [k for k, (c, ee) in enumerate(cells) if ee == e]
        if not idx: continue
        w = cell_w[idx]; es[e] = dr[:, pidx[e]] + dr[:, [dev_idx[k] for k in idx]] @ w / w.sum()
    for l in (-5, -4, -3, -2):
        j = [i for i, nm in enumerate(names) if nm == ("lead", l)][0]; es[l] = dr[:, j]
    out = dict(att_mean_lo_hi=[float(att.mean()), float(np.quantile(att, .025)), float(np.quantile(att, .975))], prob_pos=float((att > 0).mean()),
               event={str(k): [float(v.mean()), float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in sorted(es.items())})
    tr = [i for i, nm in enumerate(names) if nm[0] == "trend"]
    if tr: out["trend_absmax"] = float(np.abs(dr[:, tr].mean(0)).max())
    return out


R["sim_with_departures"] = simulate(1.0)
R["sim_no_departures"] = simulate(0.0)
R["sim_single_effect"] = simulate_single()
R["gap_share_without_departures"] = R["sim_no_departures"]["sq_gap_cs_minus_hs"] / R["sim_with_departures"]["sq_gap_cs_minus_hs"]

unit, year, y, coh = castle_data(); postb = ((coh > 0) & (year >= coh)).astype(float)
R["castle"] = dict(twfe=float(twfe(unit, year, y, postb)), cohort_by_time=float(cs_att(unit, year, coh, y)))
w = cell_weights(unit, year, postb, coh)
R["castle"]["twfe_cell_weights"] = w; R["castle"]["twfe_weight_min"] = min(w.values()); R["castle"]["twfe_n_cells"] = len(w)
R["castle"]["twfe_n_positive"] = int(sum(v > 0 for v in w.values()))
for v in ("as_published", "no_trend_block", "flat_leads"):
    R["castle"][v] = castle_fit(v, unit, year, y, coh)
    print(v, np.round(R["castle"][v]["att_mean_lo_hi"], 3), round(R["castle"][v]["prob_pos"], 2), flush=True)

for k in ("sim_with_departures", "sim_no_departures"):
    s = R[k]; print(k, "bias", np.round(s["bias"], 3), "rmse", np.round(s["rmse"], 3), "covered", s["cover_count"], "of 30, width", round(s["width"], 2))
s = R["sim_single_effect"]; print("single effect: bias", round(s["bias"], 3), "of true", round(s["bias_over_true_effect"], 3), "rmse", round(s["rmse"], 3), "covered", s["cover_count"], "of 30, width", round(s["width"], 2))
print("squared error gap share without departures", round(R["gap_share_without_departures"], 2))
print("castle twfe weights: min", round(R["castle"]["twfe_weight_min"], 3), "positive", R["castle"]["twfe_n_positive"], "of", R["castle"]["twfe_n_cells"])
json.dump(R, open("results/res_did_checks.json", "w"), indent=1, default=float)
