"""Diagnosis of the unilateral-deviation gains of review item F15 (round 1, group D2).

Design. The application industry of Section blp:sec:app (minivan data: T = 40 markets, J = 6, three two-product firms, TRUE parameters,
data simulated with Design(seed=0) and rng 2026, Bertrand Nash), with the integration rule of validate_integration.py (200 draws, or the
antithetic sets of 800, 3,200 and 12,800 draws, built from the same base as there; the 12,800 set is the antithetic set of a base of 3,200
whose first 800 are those of the 3,200 base, rng 99 then 98). For the first 8 markets and the 3 firms (24 firm and market pairs), at the
equilibrium prices p* returned by blp_core.solve_equilibrium (rivals held fixed and delta held fixed, as in the check), it reports:
(1) the analytic and finite difference gradient of the firm's profit at p* (is p* a stationary point?), (2) the eigenvalues of the 2 x 2
Hessian of the profit at p* (second order condition), (3) a dense grid search of the firm's own two prices over a wide box (markup p - c
from 0.01 to 40 on a log grid, 160 x 160 points), (4) 60 random L-BFGS-B starts plus the best grid points, with analytic gradients,
(5) all distinct stationary local maxima found, their profit, gradient norm and Hessian eigenvalues, (6) whether the best point sits
on the edge of the box, (7) the solver re-run at tolerance 1e-14 (price change and profit change). Fixed seed 7 + market index. Optional second argument: a single market index (for array jobs). Writes
candidate_r1023/diag_equilibrium_<n>[_t<t>].json for the draw count n given on the command line."""
import sys, json, time, numpy as np
from scipy.optimize import minimize
from blp_core import Design, simulate_market, ind_probs, alpha_i, price_derivs, solve_equilibrium

TRUE = dict(b0=2.0, b1=1.0, b2=0.5, alpha=1.0, sp=0.4, sx=0.7, g0=2.0, g1=0.5, g2=0.3, sxi=0.5, somega=0.3, rho=0.3)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
D0 = Design(seed=0)


def anti(nu, eta):
    return (np.concatenate([nu, -nu, nu, -nu]), np.concatenate([eta, eta, -eta, -eta]))


rng99 = np.random.default_rng(99)
base800 = (np.concatenate([D0.nu, rng99.normal(size=600)]), np.concatenate([D0.eta, rng99.normal(size=600)]))
rng98 = np.random.default_rng(98)
base3200 = (np.concatenate([base800[0], rng98.normal(size=2400)]), np.concatenate([base800[1], rng98.normal(size=2400)]))
SETS = {200: (D0.nu, D0.eta), 800: anti(D0.nu, D0.eta), 3200: anti(*base800), 12800: anti(*base3200)}
d = Design(seed=0); d.nu, d.eta = SETS[n]; d.ns = len(d.nu)
rng = np.random.default_rng(2026)
data = simulate_market(D0, TRUE, "bertrand", rng)
H = D0.ownership("bertrand")
al, sp_, sx_ = TRUE["alpha"], TRUE["sp"], TRUE["sx"]
a_i = alpha_i(al, sp_, d.nu)
firms = d.firms
TS = [int(sys.argv[2])] if len(sys.argv) > 2 else range(8)


def probs_batch(dl, x1, P):
    """P (B, J) prices -> individual probabilities (B, J, ns)."""
    V = dl[None, :, None] - a_i[None, None, :] * P[:, :, None] + sx_ * d.eta[None, None, :] * x1[None, :, None]
    m = np.maximum(V.max(axis=1), 0.0)
    e = np.exp(V - m[:, None, :])
    return e / (np.exp(-m) + e.sum(axis=1))[:, None, :]


def profit_batch(dl, x1, ct, pfix, idx, Pown, chunk=800):
    out = np.empty(len(Pown))
    for a in range(0, len(Pown), chunk):
        Pb = np.repeat(pfix[None, :], len(Pown[a:a + chunk]), axis=0); Pb[:, idx] = Pown[a:a + chunk]
        s = probs_batch(dl, x1, Pb).mean(-1)
        out[a:a + chunk] = ((Pb[:, idx] - ct[idx]) * s[:, idx]).sum(1)
    return out


def prof_grad(dl, x1, ct, pfix, idx, pown):
    pr = pfix.copy(); pr[idx] = pown
    sij = probs_batch(dl, x1, pr[None])
    s = sij.mean(-1)[0]
    Dm, _, _ = price_derivs(sij, a_i)
    Dm = Dm[0]                                  # Dm[j,k] = ds_j/dp_k
    mk = pr[idx] - ct[idx]
    val = float((mk * s[idx]).sum())
    g = s[idx] + Dm[np.ix_(idx, idx)].T @ mk    # g_k = s_k + sum_j m_j ds_j/dp_k
    return val, g


def hessian(dl, x1, ct, pfix, idx, pown, h=1e-5):
    k = len(idx); Hs = np.zeros((k, k))
    for i in range(k):
        e = np.zeros(k); e[i] = h
        Hs[:, i] = (prof_grad(dl, x1, ct, pfix, idx, pown + e)[1] - prof_grad(dl, x1, ct, pfix, idx, pown - e)[1]) / (2 * h)
    return 0.5 * (Hs + Hs.T)


t0 = time.time()
pstar, ok = solve_equilibrium(data["delta"], data["c"], d, al, sp_, sx_, H, p0=None)
pstar14, ok14 = solve_equilibrium(data["delta"], data["c"], d, al, sp_, sx_, H, p0=None, tol=1e-14, maxit=20000)
out = dict(n=n, solver_ok=bool(ok), solver_ok_1e14=bool(ok14), max_price_change_tol1e14=float(np.abs(pstar14 - pstar).max()), rows=[])
G = 160
mk_grid = np.logspace(-2, np.log10(40), G)
for t in TS:
    rs = np.random.default_rng(7 + t)
    dl, ct, x1 = data["delta"][t], data["c"][t], d.x1[t]
    for f in range(3):
        idx = np.where(firms == f)[0]; pfix = pstar[t].copy(); p0 = pstar[t, idx].copy()
        base, g0 = prof_grad(dl, x1, ct, pfix, idx, p0)
        # finite difference gradient (check of the analytic gradient)
        fd = np.array([(profit_batch(dl, x1, ct, pfix, idx, (p0 + e)[None])[0] - profit_batch(dl, x1, ct, pfix, idx, (p0 - e)[None])[0]) / 2e-6
                       for e in (np.eye(2) * 1e-6)])
        eig0 = np.linalg.eigvalsh(hessian(dl, x1, ct, pfix, idx, p0))
        p14 = pstar14[t, idx]; base14 = prof_grad(dl, x1, ct, pstar14[t], idx, p14)[0]
        # grid in markups
        M1, M2 = np.meshgrid(mk_grid, mk_grid, indexing="ij")
        Pg = np.stack([ct[idx[0]] + M1.ravel(), ct[idx[1]] + M2.ravel()], axis=1)
        pg = profit_batch(dl, x1, ct, pfix, idx, Pg).reshape(G, G)
        # grid local maxima (8 neighbours)
        pad = np.pad(pg, 1, constant_values=-np.inf); ismax = np.ones_like(pg, bool)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    ismax &= pg >= pad[1 + di:1 + di + G, 1 + dj:1 + dj + G]
        gi = np.argwhere(ismax); gv = pg[ismax]
        order = np.argsort(-gv)
        gbest = float(pg.max()); gbi = np.unravel_index(pg.argmax(), pg.shape)
        # polish from top grid maxima, from p*, and from random starts
        starts = [Pg[i * G + j] for i, j in gi[order][:8]] + [p0]
        for _ in range(60):
            starts.append(ct[idx] + np.exp(rs.uniform(np.log(0.02), np.log(30), 2)))
        pts = []
        for st in starts:
            r = minimize(lambda v: tuple(-x for x in prof_grad(dl, x1, ct, pfix, idx, v)), st, jac=True, method="L-BFGS-B",
                         bounds=[(ct[i] + 1e-6, ct[i] + 60) for i in idx], options=dict(ftol=1e-16, gtol=1e-11, maxiter=500))
            v, g = prof_grad(dl, x1, ct, pfix, idx, r.x)
            pts.append((v, r.x, float(np.linalg.norm(g))))
        pts.sort(key=lambda z: -z[0])
        distinct = []
        for v, x, gn in pts:
            if not any(np.abs(x - y).max() < 1e-4 for _, y, _ in distinct):
                distinct.append((v, x, gn))
        loc = []
        for v, x, gn in distinct:
            e = np.linalg.eigvalsh(hessian(dl, x1, ct, pfix, idx, x))
            atb = bool(np.any(x - ct[idx] > 59.9) or np.any(x - ct[idx] < 1.1e-6))
            loc.append(dict(profit=v, prices=[float(z) for z in x], markups=[float(z) for z in (x - ct[idx])], grad_norm=gn,
                            hess_eigs=[float(z) for z in e], at_bound=atb, is_pstar=bool(np.abs(x - p0).max() < 1e-4)))
        best = loc[0]
        out["rows"].append(dict(t=t, f=f, profit_pstar=base, grad_norm_pstar=float(np.linalg.norm(g0)), fd_grad_norm_pstar=float(np.linalg.norm(fd)),
                                hess_eigs_pstar=[float(z) for z in eig0], markups_pstar=[float(z) for z in (p0 - ct[idx])],
                                profit_pstar_tol1e14=float(base14), grid_best_gain=(gbest - base) / abs(base),
                                grid_best_markups=[float(mk_grid[gbi[0]]), float(mk_grid[gbi[1]])], grid_best_on_edge=bool(gbi[0] in (0, G - 1) or gbi[1] in (0, G - 1)),
                                n_grid_local_max=int(len(gv)), best_gain=(best["profit"] - base) / abs(base), best=best, n_distinct_stationary=len(loc),
                                local_maxima=loc[:6]))
        print(n, t, f, f"gain {out['rows'][-1]['best_gain']:.5f} gridgain {out['rows'][-1]['grid_best_gain']:.5f} nloc {len(loc)}", flush=True)
gains = [r["best_gain"] for r in out["rows"]]
out["max_gain"] = float(max(gains)); out["n_pairs_gain_gt_1e-6"] = int(sum(g > 1e-6 for g in gains)); out["secs"] = time.time() - t0
json.dump(out, open(f"candidate_r1023/diag_equilibrium_{n}" + (f"_t{TS[0]}" if len(sys.argv) > 2 else "") + ".json", "w"), indent=1)
print("done", out["max_gain"], f"{out['secs']:.0f}s")
