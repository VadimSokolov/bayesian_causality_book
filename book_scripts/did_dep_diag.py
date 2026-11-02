"""Diagnosis of the horseshoe's bias when the adopters depart from parallel trends (rerun r1023c4, group C4, item ch15 F3b).

Design.  The panels of did_sim2.py configurations N{40,200}_depadopt0.3_a (T = 16 periods, adoption in period 4, 8, 12 or never, unit effects N(0,1),
random walk period effects, noise sd 0.5, effects 0.5 + 0.15 e, a unit that adopts departs with probability 0.3 and then has the trend 0.15 (t - 8)).
The same panels (same seeds, SeedSequence([20261023, 0, cfg, panel])) are fitted with the horseshoe difference in differences estimator of did_hs.py
(flat unit and period effects and event time path; horseshoe groups: 0 cohort by event time deviations, 1 leads, 2 unit specific linear trends)
under variants of the prior and the sampler.  The first variant uses the sampler stream of did_sim2.py (SeedSequence([20261023, 1, cfg, panel])) and
reproduces the published horseshoe row; the others use SeedSequence([20261023, 3, cfg, panel, variant]).
Variants (all with 1,500 sweeps and 500 discarded unless stated):
  base      the horseshoe of the chapter;
  long      the same with 6,000 sweeps and 2,000 discarded;
  tau_a     global scale of the trend group fixed at tau = 0.005 (N = 200) or 0.011 (N = 40) times sigma, the value p0/(D - p0)/sqrt(n) of Piironen and Vehtari with
            p0 = 0.22 D the true number of departing units, D the number of trend coefficients, n the number of observations;
  tau_b     fixed at tau = 1 (times sigma); tau_c fixed at tau = 5;
  reg3      regularized horseshoe (Piironen and Vehtari) with slab sd c = 3 sigma on every penalized coefficient, scales updated by Metropolis steps;
  reg10     the same with c = 10 sigma;
  flat      unit trends with a flat prior (no shrinkage on the trends; leads and deviations still horseshoe);
  oracle    trends only for the units that truly depart, flat prior (the identification check: nothing is shrunk, nothing is selected).
Prior variance of a penalized coefficient is sigma^2 lambda^2 tau^2 (regularized: sigma^2 c^2 lambda^2 tau^2 / (c^2 + lambda^2 tau^2)), lambda and tau half Cauchy.
Quantities per panel and variant: error of the posterior mean of the average effect on the treated, the 95 percent central interval and its coverage;
for the base, long and tau variants also the posterior mean of the trend coefficient of every unit divided by the true coefficient 0.15 sd(t) = 0.6915
(shrinkage factor), averaged over departing adopters by cohort, over adopters that do not depart, and over never treated units, and the posterior mean of the
global scale of the trend group, of the deviations and of the leads.
Run:  python book_scripts/did_dep_diag.py N K    N = 40 or 200; K = task index; panels K*P .. K*P+P-1 with P = 20 (N = 40) or 10 (N = 200), 500 panels in all (25 and 50 tasks).
Merge: python book_scripts/did_dep_diag.py --merge
"""
import sys, os, json, time
import numpy as np
sys.argv_saved = list(sys.argv)
sys.argv = [sys.argv[0]]
import book_scripts.did_sim2 as S       # noqa: E402  (exec of did_hs.py inside; gives make_panel, build, att_from)
sys.argv = sys.argv_saved

T = S.T; OUT = "results/did_dep_diag"
SLOPE = 0.15 * np.arange(T).std()      # true trend coefficient on the standardized year used by build()


def sampler(Z, y, groups, rng, iters=1500, burn=500, fix_tau=None, c=None, flat=(), record_tau=False, mh_steps=5):
    """Gibbs sampler of did_hs.group_hs with options.  fix_tau: {group: tau}; c: slab sd (regularized horseshoe) in units of sigma; flat: groups with a flat prior."""
    n, q = Z.shape; groups = np.asarray(groups); G = groups.max() + 1
    pen = (groups >= 0) & ~np.isin(groups, list(flat))
    gi = np.maximum(groups, 0)
    ZtZ = Z.T @ Z; Zty = Z.T @ y; yty = y @ y
    lam2 = np.ones(q); nu = np.ones(q); tau2 = np.ones(G) * 0.1; xi = np.ones(G); s2 = np.var(y)
    fixed = np.zeros(G, bool)
    if fix_tau:
        for g, t in fix_tau.items(): tau2[g] = t ** 2; fixed[g] = True
    for g in flat: fixed[g] = True
    out = []; taus = []
    reg = c is not None; c2 = (c ** 2) if reg else None

    def var(l2, t2):
        return c2 * l2 * t2 / (c2 + l2 * t2) if reg else l2 * t2

    def lp_beta(b2, v):                      # log density of b | v (units of s2), up to constants
        return -0.5 * np.log(v) - b2 / (2 * s2 * v)

    for it in range(iters):
        pv = np.where(pen, var(lam2, tau2[gi]), np.inf)
        A = ZtZ + np.diag(np.where(pen, 1 / pv, 1e-8)); L = np.linalg.cholesky(A)
        b = np.linalg.solve(A, Zty) + np.sqrt(s2) * np.linalg.solve(L.T, rng.normal(size=q))
        rss = yty - 2 * b @ Zty + b @ ZtZ @ b
        bp = b[pen]; tp = tau2[gi[pen]]
        vp = var(lam2[pen], tp)
        s2 = 1 / rng.gamma((n + pen.sum()) / 2, 1 / (rss / 2 + (bp ** 2 / vp).sum() / 2))
        if not reg:
            lam2[pen] = 1 / rng.gamma(1, 1 / (1 / nu[pen] + bp ** 2 / (2 * tp * s2)))
            nu[pen] = 1 / rng.gamma(1, 1 / (1 + 1 / lam2[pen]))
            for g in range(G):
                if fixed[g]: continue
                s = groups == g
                tau2[g] = 1 / rng.gamma((s.sum() + 1) / 2, 1 / (1 / xi[g] + (b[s] ** 2 / lam2[s]).sum() / (2 * s2)))
                xi[g] = 1 / rng.gamma(1, 1 / (1 + 1 / tau2[g]))
        else:                                # Metropolis on log lambda and log tau, half Cauchy priors (density in the log scale: x^2 / (1 + x^2))
            idx = np.where(pen)[0]; b2 = b[idx] ** 2
            for _ in range(mh_steps):
                ll = 0.5 * np.log(lam2[idx]); prop = ll + rng.normal(0, 1.0, len(idx))
                l2n = np.exp(2 * prop); t2 = tau2[gi[idx]]
                cur = lp_beta(b2, var(lam2[idx], t2)) + 2 * ll - np.log1p(np.exp(2 * ll))
                new = lp_beta(b2, var(l2n, t2)) + 2 * prop - np.log1p(np.exp(2 * prop))
                acc = np.log(rng.random(len(idx))) < new - cur
                lam2[idx[acc]] = l2n[acc]
            for g in range(G):
                if fixed[g]: continue
                s = np.where(pen & (groups == g))[0]
                if len(s) == 0: continue
                b2s = b[s] ** 2
                for _ in range(mh_steps):
                    lt = 0.5 * np.log(tau2[g]); pt = lt + rng.normal(0, 0.5)
                    cur = lp_beta(b2s, var(lam2[s], tau2[g])).sum() + 2 * lt - np.log1p(np.exp(2 * lt))
                    new = lp_beta(b2s, var(lam2[s], np.exp(2 * pt))).sum() + 2 * pt - np.log1p(np.exp(2 * pt))
                    if np.log(rng.random()) < new - cur: tau2[g] = np.exp(2 * pt)
        if it >= burn:
            out.append(b.copy()); taus.append(np.sqrt(tau2.copy()))
    return np.array(out), np.array(taus)


def fit(panel, variant, cfg_id, p, pen_depart):
    unit, year, ccode, ev, y, treated, att_true = panel
    Z, grp, names, _, cells = S.build(unit, year, ccode, y)
    grp = grp.copy()
    N = unit.max() + 1
    tr_idx = [i for i, nm in enumerate(names) if nm[0] == "trend"]
    if variant == "oracle":
        keep = np.array([i for i in range(len(names)) if names[i][0] != "trend" or pen_depart[names[i][1]]])
        Z, grp = Z[:, keep], grp[keep]; names = [names[i] for i in keep]; grp[[i for i, nm in enumerate(names) if nm[0] == "trend"]] = -1
    if variant == "base":
        rng = np.random.default_rng(np.random.SeedSequence([20261023, 1, cfg_id, p])); kw = {}
    else:
        rng = np.random.default_rng(np.random.SeedSequence([20261023, 3, cfg_id, p, VARIANTS.index(variant)]))
        kw = dict(long=dict(iters=6000, burn=2000), tau_a=dict(fix_tau={2: 0.22 / 0.78 / np.sqrt(16 * N)}), tau_b=dict(fix_tau={2: 1.0}), tau_c=dict(fix_tau={2: 5.0}),
                  reg3=dict(c=3.0), reg10=dict(c=10.0), flat=dict(flat=(2,)), oracle=dict())[variant]
    dr, taus = sampler(Z, y, grp, rng, **kw)
    draws = S.att_from(dr, names, cells, ccode, ev)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    res = dict(err=float(draws.mean() - att_true), lo=float(lo - att_true), hi=float(hi - att_true))
    if variant != "oracle":
        tm = dr[:, tr_idx].mean(0) / SLOPE            # shrinkage factor of every unit
        coh = ccode[::T]; dep = pen_depart
        res["shr"] = {f"dep{c}": float(tm[dep & (coh == c)].mean()) if (dep & (coh == c)).any() else None for c in (4, 8, 12)}
        res["shr"]["dep_all"] = float(tm[dep].mean()) if dep.any() else None
        res["shr"]["nodep_adopt"] = float(tm[(~dep) & (coh > 0)].mean())
        res["shr"]["nodep_nevertreated"] = float(tm[coh == 0].mean())
        res["tau"] = [float(v) for v in taus.mean(0)]   # trend group is index 2; for 'long' etc the same
    return res


VARIANTS = ["base", "long", "tau_a", "tau_b", "tau_c", "reg3", "reg10", "flat", "oracle"]


def one(cfg_id, p):
    N, pdep, design = S.CONFIGS[cfg_id]
    gen = np.random.default_rng(np.random.SeedSequence([20261023, 0, cfg_id, p]))
    panel = S.make_panel(gen, N, pdep, design); dep = S.make_panel.flag.copy()
    return dict(panel=p, n_dep=int(dep.sum()), att_true=panel[-1], **{v: fit(panel, v, cfg_id, p, dep) for v in VARIANTS})


if __name__ == "__main__":
    if sys.argv_saved[1] == "--merge":
        res = {}
        for N in (40, 200):
            rows = []
            for k in range(500 // (20 if N == 40 else 10)):
                rows += json.load(open(f"{OUT}/diag_N{N}_{k}.json"))
            res[f"N{N}"] = rows
        json.dump(res, open("results/res_did_dep_diag.json", "w")); print({k: len(v) for k, v in res.items()}); sys.exit(0)
    N = int(sys.argv_saved[1]); k = int(sys.argv_saved[2]); P = 20 if N == 40 else 10
    cfg_id = {40: 18, 200: 19}[N]       # indices of N40_depadopt0.3_a and N200_depadopt0.3_a in CONFIGS (checked below)
    assert S.CONFIGS[cfg_id] == (N, "adopt0.3", "a"), S.CONFIGS[cfg_id]
    os.makedirs(OUT, exist_ok=True); rows = []
    for j in range(P):
        t0 = time.time(); rows.append(one(cfg_id, k * P + j))
        print(N, k, j, {v: round(rows[-1][v]["err"], 3) for v in VARIANTS}, round(time.time() - t0), "s", flush=True)
    json.dump(rows, open(f"{OUT}/diag_N{N}_{k}.json", "w"))
