"""Sensitivity of the simulated demand integrals to the number of integration draws (review item F15).

Design. The application industry of Section blp:sec:app (the minivan data: T = 40 markets, J = 6, TRUE parameters, data simulated with the
200 draws of the Design, seed 0, rng 2026, Bertrand Nash) and 50 prior predictive industries per conduct (Bertrand, collusion,
perfect competition; priors of estimators.PRIOR; rng 31415; data simulated with the 200 draws). Every quantity is recomputed with
200 draws (the current rule: nu and eta standard normal, fixed across markets), 800 and 3,200 draws. The larger sets are built
antithetically from a base set: (nu, eta), (-nu, eta), (nu, -eta), (-nu, -eta). The base set of the 800 draws is the current 200
plus 600 new standard normal pairs (rng 99), so the 800 contain the 200 and the 3,200 are the antithetic set of a base of 800.
For the application the observed shares and prices are held fixed and, at the true (alpha, sigma_p, sigma_x) and at the BLP GMM point
estimate, the contraction, markups, Lerner index, merger price effect (merging products A and B) and consumer surplus change are
recomputed with each draw set; for the prior predictive industries the same is done at the true parameters under the true conduct.
Reported: the largest change relative to 200 draws (maximum over markets and products for the application, maximum and median over
industries for the aggregates). Also: the largest change in the shares themselves at the 200 draw delta (an integration error of
the share integral), a multi start check of the equilibrium solver at the application (starting prices c + k / alpha for k in
0.25, 0.5, 1, 3, the observed price times 0.6 and 1.5), and a check for profitable unilateral deviations: for each of the three firms
in the first 8 markets, the own prices are re optimised by L-BFGS-B from six starting points with rivals held at the equilibrium, and
the largest profit gain relative to the equilibrium profit is reported (200 and 3,200 draws). Writes results/res_integration_check.json."""
import copy, json, time, numpy as np
from scipy.optimize import minimize
from blp_core import (Design, simulate_market, contraction, ind_probs, shares, solve_equilibrium, CONDUCTS)
from estimators import draw_prior, BLPGMM
from antitrust import structural, estimands, MERGING

TRUE = dict(b0=2.0, b1=1.0, b2=0.5, alpha=1.0, sp=0.4, sx=0.7, g0=2.0, g1=0.5, g2=0.3, sxi=0.5, somega=0.3, rho=0.3)
D0 = Design(seed=0)


def anti(nu, eta):
    return (np.concatenate([nu, -nu, nu, -nu]), np.concatenate([eta, eta, -eta, -eta]))


rng99 = np.random.default_rng(99)
base800 = (np.concatenate([D0.nu, rng99.normal(size=600)]), np.concatenate([D0.eta, rng99.normal(size=600)]))
SETS = {200: (D0.nu, D0.eta), 800: anti(D0.nu, D0.eta), 3200: anti(*base800)}


def design(n):
    d = Design(seed=0)
    d.nu, d.eta = SETS[n]; d.ns = len(d.nu)
    return d


DS = {n: design(n) for n in SETS}
t0 = time.time()
rng = np.random.default_rng(2026)
data = simulate_market(D0, TRUE, "bertrand", rng)
s, p = data["s"], data["p"]
out = dict(draws=[200, 800, 3200])


def quantities(d, s, p, th2, conduct):
    q = structural(d, s, p, th2, conduct)
    return q


def compare(th2, conduct, s, p, label):
    base = quantities(DS[200], s, p, th2, conduct); res = {}
    for n in (800, 3200):
        q = quantities(DS[n], s, p, th2, conduct)
        res[str(n)] = dict(
            max_abs_change_markup=float(np.abs(q["m"] - base["m"]).max()), max_abs_change_lerner=float(np.abs(q["lerner"] - base["lerner"]).max()),
            max_abs_change_dp_pts=float(100 * np.abs(q["dp"] - base["dp"]).max()),
            max_abs_change_dcs=float(np.abs(q["dcs"] - base["dcs"]).max()),
            change_mean_lerner=float(q["lerner"].mean() - base["lerner"].mean()),
            change_merger_effect_pts=float(100 * (q["dp"][:, MERGING].mean() - base["dp"][:, MERGING].mean())),
            change_mean_dcs=float(q["dcs"].mean() - base["dcs"].mean()),
            rel_change_merger_effect=float(q["dp"][:, MERGING].mean() / base["dp"][:, MERGING].mean() - 1), ok=bool(q["ok"]))
    res["base"] = dict(lerner=float(base["lerner"].mean()), merger_effect_pts=float(100 * base["dp"][:, MERGING].mean()), dcs=float(base["dcs"].mean()))
    # share integration error at the 200 draw delta
    a, sp, sx = th2
    dl = base["delta"]
    res["max_abs_share_change_at_200_delta"] = {str(n): float(np.abs(shares(dl, p, DS[n], a, sp, sx) - s).max()) for n in (800, 3200)}
    print(label, json.dumps(res)[:400], flush=True)
    return res


# ---- application, at the truth and at the GMM point estimate
th_true = (TRUE["alpha"], TRUE["sp"], TRUE["sx"])
out["application_true"] = compare(th_true, "bertrand", s, p, "app true")
g = BLPGMM(D0, s, p).fit()
out["gmm_th2"] = [float(x) for x in g.th2]
out["application_gmm_estimate"] = compare(np.abs(g.th2), "bertrand", s, p, "app gmm")

# ---- prior predictive industries
crng = np.random.default_rng(31415); inds = {}
for conduct in CONDUCTS:
    rows = []
    while len(rows) < 50:
        th = draw_prior(crng); r = simulate_market(D0, th, conduct, crng)
        if not r["ok"]:
            continue
        t2 = (th["alpha"], th["sp"], th["sx"])
        try:
            b = structural(DS[200], r["s"], r["p"], t2, conduct)
            row = []
            for n in (800, 3200):
                q = structural(DS[n], r["s"], r["p"], t2, conduct)
                row.append([q["lerner"].mean() - b["lerner"].mean(), 100 * (q["dp"][:, MERGING].mean() - b["dp"][:, MERGING].mean()),
                            q["dcs"].mean() - b["dcs"].mean(), float(b["ok"] and q["ok"])])
            if all(r_[3] for r_ in row) and np.all(np.isfinite(row)):
                rows.append(row)
        except Exception:
            continue
    A = np.array(rows)            # (50, 2 sizes, 4)
    inds[conduct] = {str(n): dict(max_abs_lerner=float(np.abs(A[:, k, 0]).max()), median_abs_lerner=float(np.median(np.abs(A[:, k, 0]))),
                                  max_abs_merger_pts=float(np.abs(A[:, k, 1]).max()), median_abs_merger_pts=float(np.median(np.abs(A[:, k, 1]))),
                                  max_abs_dcs=float(np.abs(A[:, k, 2]).max()), median_abs_dcs=float(np.median(np.abs(A[:, k, 2]))))
                     for k, n in enumerate((800, 3200))}
    print(conduct, json.dumps(inds[conduct]), flush=True)
out["prior_predictive"] = inds

# ---- equilibrium solver from several starts, and unilateral deviations, at 200 and 3200 draws
H = D0.ownership("bertrand")
dev = {}
for n in (200, 3200):
    d = DS[n]
    pstar, ok = solve_equilibrium(data["delta"], data["c"], d, TRUE["alpha"], TRUE["sp"], TRUE["sx"], H, p0=None)
    starts = [data["c"] + k / TRUE["alpha"] for k in (0.25, 0.5, 1.0, 3.0)] + [0.6 * p, 1.5 * p]
    spread, nconv = 0.0, 0
    for st in starts:
        pp, ok2 = solve_equilibrium(data["delta"], data["c"], d, TRUE["alpha"], TRUE["sp"], TRUE["sx"], H, p0=st)
        nconv += int(ok2); spread = max(spread, float(np.abs(pp - pstar).max()))
    gains = []
    firms = d.firms
    for t in range(8):
        dl, ct = data["delta"][t:t + 1], data["c"][t]
        d1 = copy.copy(d); d1.x1 = d.x1[t:t + 1]     # one market view: shares() takes x1 from the design (r1023 fix: it used market 0 for every t)
        def profit(pv, f):
            pr = pstar[t:t + 1].copy(); idx = np.where(firms == f)[0]; pr[0, idx] = pv
            sh = shares(dl, pr, d1, TRUE["alpha"], TRUE["sp"], TRUE["sx"])[0]
            return float(((pr[0, idx] - ct[idx]) * sh[idx]).sum())
        for f in range(3):
            idx = np.where(firms == f)[0]; p0 = pstar[t, idx]; base = profit(p0, f); best = base
            sts = [p0 * k for k in (0.5, 0.8, 1.2, 1.6)] + [p0 + 0.7, np.maximum(ct[idx] + 0.05, p0 - 0.7)]
            for st in sts:
                res = minimize(lambda v: -profit(v, f), st, method="L-BFGS-B", bounds=[(ct[i] , ct[i] + 20) for i in idx])
                best = max(best, -res.fun)
            gains.append((best - base) / abs(base))
    dev[str(n)] = dict(solver_converged_of_6=nconv, max_abs_price_spread_across_starts=spread, max_abs_price_vs_dgp_price=float(np.abs(pstar - p).max()),
                       max_relative_profit_gain_from_deviation=float(max(gains)), n_firm_markets=len(gains))
    print("deviation", n, json.dumps(dev[str(n)]), flush=True)
out["equilibrium_checks"] = dev
out["secs"] = time.time() - t0
json.dump(out, open("results/res_integration_check.json", "w"), indent=1)
print("done", f"{out['secs']:.0f}s")
