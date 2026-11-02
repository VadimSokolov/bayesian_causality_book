"""Chapter 15 difference in differences simulation, enlarged (rerun r1023, group C, item ch15 F6).

Design.  The staggered adoption panels of did_hs.py: T = 16 periods, each unit adopts in period 4, 8 or 12 or never,
each with probability 1/4 (the adoption draw is repeated if any of the four groups is empty), unit effect alpha_i ~ N(0, 1),
common period effect a random walk gamma_t with N(0, 0.2^2) steps, noise sd 0.5, effect on a treated unit period
tau = 0.5 + 0.15 e (e = periods since adoption) plus a cohort deviation (see effect designs), and a trend departure
0.15 (t - 8) for a unit that is a violator.  The configurations cross
    N in {40, 200} units,
    departure probability in {0, 0.1, 0.3} (each unit independently a violator),
    effect design (a) common path as in the chapter, (b) common path plus one sparse cohort deviation u(4, 2) = 0.8,
                   (c) common path plus dense deviations u(g, e) = 0.3 sin(g + e) for every treated cohort by event time cell.
There are 500 panels per configuration (the Monte Carlo standard error of a 95 percent coverage near 0.95 is about 0.01).
The target is the average of tau over the treated unit periods of the panel (the effect on the treated).
Estimators, all on the same panel: two way fixed effects; the cohort by time estimator with never treated controls
(cs_att of did_hs.py); the horseshoe difference in differences estimator of did_hs.py with 1,500 sweeps, 500 discarded
(its 95 percent interval is the central posterior interval); the imputation estimator of Borusyak, Jaravel and Spiess (2024),
OLS of the untreated observations on unit and period dummies, imputation of Y(infinity) in the treated cells and the mean of the
differences over treated unit periods ('bjs'), and the same imputation with unit specific linear trends ('bjs_trend').
The imputation estimators get a conservative unit clustered standard error in the style of Borusyak, Jaravel and Spiess:
the influence weights of the estimator times the imputation residuals (untreated cells) or the effect residuals around the
cohort by event time cell mean (treated cells), summed within unit; a normal 95 percent interval follows.
Two way fixed effects and cohort by time report only the point estimate.
Random streams: the panel of (configuration, panel index) has its own generator (SeedSequence([20261023, 0, cfg, panel])),
the sampler a separate one (SeedSequence([20261023, 1, cfg, panel])).

Added in rerun r1023b, group C2 (item ch15 F1): four further configurations, indices 18 to 21 (the first 18 and their random streams are unchanged),
in which the departures go with adoption.  Design (a) effects, N in {40, 200}, 500 panels each, the same five estimators.
    'adopt0.3': a unit departs with probability 0.3 if it adopts (cohort 4, 8 or 12) and with probability 0 if it is never treated;
    'c12_0.3' : a unit departs with probability 0.3 if it is in the period 12 cohort and with probability 0 otherwise.
The departure is the same 0.15 (t - 8) trend.  The uniform draws that decide departure are drawn as in the original grid and then masked,
and the seed scheme is the same (SeedSequence([20261023, 0, cfg, panel]), configuration index 18..21).
Run on tasks k = 180..219 (4 configurations x 10 chunks of 50 panels); merge with  python book_scripts/did_sim2.py --merge-dep
which adds the keys N{N}_dep{adopt0.3|c12_0.3}_a to results/res_did_sim2.json and did_sim2_reps.json without touching the other keys.

Added in rerun r1023c4, group C4 (item ch15 F3b): configurations 22 and 23, 'adopt0.1' (a unit departs with probability 0.1 if it adopts, never otherwise;
about 7.5 percent of units depart and every one of them is treated), N in {40, 200}, design (a), 500 panels each, tasks k = 220..239 (2 configurations x 10 chunks);
merge with  python book_scripts/did_sim2.py --merge-dep  (now covers configurations 18 to 23).  The probability is parsed from the name; 'adopt0.3' and 'c12_0.3' are as before.

Run (Hopper array task k = 0..179, 50 panels each, 18 configurations x 10 chunks):  python book_scripts/did_sim2.py K [PANELS_PER_TASK] [ITERS BURN]
Writes results/did_sim2/task_K.json (replication level rows).  merge with:  python book_scripts/did_sim2.py --merge
which writes results/res_did_sim2.json (summary, paired squared error differences) and results/did_sim2_reps.json.
Columns of a replication row: att_true, then estimate minus truth for twfe, cs, hs, bjs, bjs_trend, then lo and hi of
hs, bjs, bjs_trend.
"""
import sys, os, json, itertools
import numpy as np, pandas as pd

SRC = "book_scripts/did_hs.py"; MARK = "# ---------------- Simulation"
src = open(SRC).read(); assert src.count(MARK) == 1
exec(compile(src.split(MARK)[0], SRC, "exec"))     # imports, rng, group_hs, build, att_from, twfe, cs_att

T = 16
CONFIGS = [(N, p, d) for N in (40, 200) for p in (0.0, 0.1, 0.3) for d in ("a", "b", "c")]
CONFIGS = CONFIGS + [(N, p, "a") for p in ("adopt0.3", "c12_0.3") for N in (40, 200)]     # departures correlated with adoption (r1023b)
CONFIGS = CONFIGS + [(N, "adopt0.1", "a") for N in (40, 200)]                              # sparse case, indices 22, 23 (r1023c4)
PER_CFG, CHUNKS = 500, 10
METHODS = ["twfe", "cs", "hs", "bjs", "bjs_trend"]
OUTDIR = "results/did_sim2"


def make_panel(gen, N, pdep, design):
    while True:
        cohort = gen.choice([4, 8, 12, 0], N)
        if len(set(cohort.tolist())) == 4: break
    unit = np.repeat(np.arange(N), T); year = np.tile(np.arange(T), N)
    ccode = np.repeat(cohort, T)
    alpha = np.repeat(gen.normal(0, 1, N), T); gamma = np.tile(np.cumsum(gen.normal(0, 0.2, T)), N)
    if isinstance(pdep, str):                      # departures correlated with adoption
        pr = float(pdep.split("adopt")[1]) if pdep.startswith("adopt") else float(pdep.split("_")[1])
        flag = gen.random(N) < pr
        flag = flag & (cohort > 0) if pdep.startswith("adopt") else flag & (cohort == 12)
        make_panel.flag = flag                    # kept for the diagnostics of did_dep_diag.py
    else:
        flag = gen.random(N) < pdep
    viol = np.repeat(flag, T) * 0.15 * (year - T / 2)
    noise = gen.normal(0, 0.5, N * T)
    ev = np.where(ccode > 0, year - ccode, -999)
    treated = (ccode > 0) & (ev >= 0)
    tau = np.where(treated, 0.5 + 0.15 * ev, 0.0)
    if design == "b":
        tau = tau + np.where(treated & (ccode == 4) & (ev == 2), 0.8, 0.0)
    elif design == "c":
        tau = tau + np.where(treated, 0.3 * np.sin(ccode + ev), 0.0)
    y = alpha + gamma + tau + viol + noise
    return unit, year, ccode, ev, y, treated, float(tau[treated].mean())


def bjs(unit, year, ccode, ev, y, treated, trends):
    """Imputation estimator with a clustered conservative standard error; returns (estimate, se)."""
    n = len(y); N = unit.max() + 1
    X = np.zeros((n, N + (T - 1) + (N if trends else 0)))
    idx = np.arange(n)
    X[idx, unit] = 1.0
    m = year > 0; X[idx[m], N + year[m] - 1] = 1.0
    if trends: X[idx, N + T - 1 + unit] = (year - (T - 1) / 2) / T
    X0, y0, X1, y1 = X[~treated], y[~treated], X[treated], y[treated]
    beta = np.linalg.lstsq(X0, y0, rcond=None)[0]
    eff = y1 - X1 @ beta; est = float(eff.mean())
    a = X1.mean(0); Ainv = np.linalg.pinv(X0.T @ X0, hermitian=True)
    om0 = -(X0 @ (Ainv @ a)); om1 = np.full(len(y1), 1.0 / len(y1))
    e0 = y0 - X0 @ beta
    cell = pd.Series(eff).groupby([ccode[treated], ev[treated]]).transform("mean").values
    e1 = eff - cell
    s = np.bincount(unit[~treated], weights=om0 * e0, minlength=N) + np.bincount(unit[treated], weights=om1 * e1, minlength=N)
    return est, float(np.sqrt((s ** 2).sum()))


def one_panel(cfg_id, p, iters, burn):
    global rng
    N, pdep, design = CONFIGS[cfg_id]
    gen = np.random.default_rng(np.random.SeedSequence([20261023, 0, cfg_id, p]))
    unit, year, ccode, ev, y, treated, att_true = make_panel(gen, N, pdep, design)
    post = treated.astype(float)
    tw = twfe(unit, year, y, post); cs = cs_att(unit, year, ccode, y)
    rng = np.random.default_rng(np.random.SeedSequence([20261023, 1, cfg_id, p]))
    Z, grp, names, _, cells = build(unit, year, ccode, y)
    dr = group_hs(Z, y, grp, iters=iters, burn=burn)
    draws = att_from(dr, names, cells, ccode, ev)
    hlo, hhi = np.quantile(draws, [0.025, 0.975])
    b0, s0 = bjs(unit, year, ccode, ev, y, treated, False)
    b1, s1 = bjs(unit, year, ccode, ev, y, treated, True)
    return [att_true, tw - att_true, cs - att_true, draws.mean() - att_true, b0 - att_true, b1 - att_true,
            hlo - att_true, hhi - att_true, b0 - 1.96 * s0 - att_true, b0 + 1.96 * s0 - att_true,
            b1 - 1.96 * s1 - att_true, b1 + 1.96 * s1 - att_true]


def summarize(rows):
    A = np.array(rows); err = A[:, 1:6]; n = len(A)
    out = {}
    for j, m in enumerate(METHODS):
        e = err[:, j]
        d = dict(bias=float(e.mean()), bias_se=float(e.std(ddof=1) / np.sqrt(n)), rmse=float(np.sqrt((e ** 2).mean())),
                 rmse_se=float(((e ** 2).std(ddof=1) / np.sqrt(n)) / (2 * np.sqrt((e ** 2).mean()) + 1e-12)))
        if m in ("hs", "bjs", "bjs_trend"):
            k = {"hs": 6, "bjs": 8, "bjs_trend": 10}[m]
            lo, hi = A[:, k], A[:, k + 1]; cov = (lo <= 0) & (hi >= 0)
            d.update(cover=float(cov.mean()), cover_se=float(np.sqrt(cov.mean() * (1 - cov.mean()) / n)),
                     width=float((hi - lo).mean()), width_se=float((hi - lo).std(ddof=1) / np.sqrt(n)))
        out[m] = d
    sq = err ** 2; hs = sq[:, 2]
    out["paired_sq_error_minus_hs"] = {m: dict(mean=float((sq[:, j] - hs).mean()), se=float((sq[:, j] - hs).std(ddof=1) / np.sqrt(n)))
                                       for j, m in enumerate(METHODS) if m != "hs"}
    out["att_true_mean"] = float(A[:, 0].mean()); out["n_panels"] = n
    return out


def build_single(unit, year, cohort, y):
    """First version of the estimator (the design error of Chapter 15): one flat common effect for every treated cell instead of
    the event time path; horseshoe on the cohort by event time deviations from it, leads and unit trends as in the published estimator."""
    Z, grp, names, post, cells = build(unit, year, cohort, y)
    keep = [i for i, nm in enumerate(names) if nm[0] != "path"]
    return np.column_stack([post, Z[:, keep]]), np.concatenate([[-1], grp[keep]]), [("tbar", None)] + [names[i] for i in keep], post, cells


def att_single(dr, names, cells, coh, ev):
    didx = {nm[1]: i for i, nm in enumerate(names) if nm[0] == "dev"}
    w = np.array([np.sum((coh == c) & (ev == e)) for (c, e) in cells], float)
    return dr[:, 0] + sum(w[k] * dr[:, didx[(c, e)]] for k, (c, e) in enumerate(cells)) / w.sum()


def one_single(cfg_id, p, iters, burn):
    """The same panel as one_panel (same seeds) fitted with the single effect parametrization: [att_true, error of the mean, lo, hi]."""
    global rng
    N, pdep, design = CONFIGS[cfg_id]
    gen = np.random.default_rng(np.random.SeedSequence([20261023, 0, cfg_id, p]))
    unit, year, ccode, ev, y, treated, att_true = make_panel(gen, N, pdep, design)
    rng = np.random.default_rng(np.random.SeedSequence([20261023, 2, cfg_id, p]))
    Z, grp, names, _, cells = build_single(unit, year, ccode, y)
    draws = att_single(group_hs(Z, y, grp, iters=iters, burn=burn), names, cells, ccode, ev)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return [att_true, draws.mean() - att_true, lo - att_true, hi - att_true]


if __name__ == "__main__":
    if "--single" in sys.argv:       # python did_sim2.py --single K : K = 0..19, configurations (N, 0.1, a) for N = 40, 200, 50 panels each
        k = int(sys.argv[sys.argv.index("--single") + 1]); cfg_id = {0: 3, 1: 12}[k // 10]; chunk = k % 10
        os.makedirs(OUTDIR, exist_ok=True)
        rows = [one_single(cfg_id, chunk * 50 + j, 1500, 500) for j in range(50)]
        json.dump(dict(cfg_id=cfg_id, chunk=chunk, rows=rows), open(f"{OUTDIR}/single_{k}.json", "w")); sys.exit(0)
    if "--merge-single" in sys.argv:
        out = {}
        for cfg_id in (3, 12):
            rows = []
            for k in range(10 * (cfg_id == 12), 10 * (cfg_id == 12) + 10):
                rows += json.load(open(f"{OUTDIR}/single_{k}.json"))["rows"]
            A = np.array(rows); e = A[:, 1]; cov = (A[:, 2] <= 0) & (A[:, 3] >= 0); n = len(A)
            out[f"N{CONFIGS[cfg_id][0]}_dep0.1_a"] = dict(n_panels=n, att_true_mean=float(A[:, 0].mean()), bias=float(e.mean()), bias_se=float(e.std(ddof=1) / np.sqrt(n)),
                bias_over_true=float(e.mean() / A[:, 0].mean()), rmse=float(np.sqrt((e ** 2).mean())), cover=float(cov.mean()), cover_count=int(cov.sum()),
                cover_se=float(np.sqrt(cov.mean() * (1 - cov.mean()) / n)), width=float((A[:, 3] - A[:, 2]).mean()))
        json.dump(out, open("results/res_did_sim2_single.json", "w"), indent=1); print(json.dumps(out, indent=1)); sys.exit(0)
    if "--merge-dep" in sys.argv:
        res = json.load(open("results/res_did_sim2.json")); allr = json.load(open("results/did_sim2_reps.json"))
        for cid in range(18, len(CONFIGS)):
            rows = []
            for chunk in range(CHUNKS):
                rows += json.load(open(f"{OUTDIR}/task_{cid * CHUNKS + chunk}.json"))["rows"]
            N, pdep, design = CONFIGS[cid]; key = f"N{N}_dep{pdep}_{design}"
            res[key] = dict(N=N, departure_prob=pdep, design=design, **summarize(rows)); allr[key] = rows
        json.dump(res, open("results/res_did_sim2.json", "w"), indent=1); json.dump(allr, open("results/did_sim2_reps.json", "w"))
        print(len(res), "configurations in res_did_sim2.json"); sys.exit(0)
    if "--merge" in sys.argv:
        reps = {}
        for k in range(18 * CHUNKS):
            f = f"{OUTDIR}/task_{k}.json"
            if os.path.exists(f):
                d = json.load(open(f)); reps.setdefault(d["cfg_id"], {})[d["chunk"]] = d["rows"]
        res = {}
        for cid, ch in sorted(reps.items()):
            rows = [r for c in sorted(ch) for r in ch[c]]
            N, pdep, design = CONFIGS[cid]
            res[f"N{N}_dep{pdep}_{design}"] = dict(N=N, departure_prob=pdep, design=design, **summarize(rows))
        json.dump(res, open("results/res_did_sim2.json", "w"), indent=1)
        json.dump({f"N{CONFIGS[c][0]}_dep{CONFIGS[c][1]}_{CONFIGS[c][2]}": [r for k in sorted(ch) for r in ch[k]] for c, ch in sorted(reps.items())},
                  open("results/did_sim2_reps.json", "w"))
        print(len(res), "configurations merged"); sys.exit(0)
    k = int(sys.argv[1]); per = int(sys.argv[2]) if len(sys.argv) > 2 else PER_CFG // CHUNKS
    iters, burn = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1500, 500)
    cfg_id, chunk = divmod(k, CHUNKS)
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for j in range(per):
        rows.append(one_panel(cfg_id, chunk * per + j, iters, burn))
        print(k, CONFIGS[cfg_id], j, np.round(rows[-1][:6], 3), flush=True)
    json.dump(dict(cfg_id=cfg_id, chunk=chunk, config=CONFIGS[cfg_id], rows=rows), open(f"{OUTDIR}/task_{k}.json", "w"))
