"""Corrected unilateral deviation check (review item F15, round 1, group D2).

Design. The application industry of Section blp:sec:app (T = 40 markets, J = 6, three two-product firms, TRUE parameters, Design(seed=0),
rng 2026, Bertrand Nash), integration with 200 draws and with the antithetic 3,200 draws of validate_integration.py. For the first 8
markets, every firm re-optimises its own prices (L-BFGS-B from 6 starts, plus the grid of diagnose_equilibrium.py is the stronger check)
with rivals and delta held at the equilibrium; the largest relative profit gain is reported, for (a) the pre merger Bertrand equilibrium
(3 firms, 24 pairs) and (b) the post merger equilibrium of the merger of products A and B (two firms, 4 + 2 products, 16 pairs), at the
true costs. The share function takes the product characteristic x1 from the design, so each market is passed as a one market view of
the design (the version of this check in validate_integration.py before r1023 passed market 0's x1 for every market, which created the
reported gains). Writes candidate_r1023/res_deviation_check.json."""
import copy, json, numpy as np
from scipy.optimize import minimize
from blp_core import Design, simulate_market, shares, solve_equilibrium
import validate_integration_draws as vd   # draw sets only

TRUE = dict(b0=2.0, b1=1.0, b2=0.5, alpha=1.0, sp=0.4, sx=0.7, g0=2.0, g1=0.5, g2=0.3, sxi=0.5, somega=0.3, rho=0.3)
POST = (0, 0, 0, 0, 2, 2)
D0 = Design(seed=0)
data = simulate_market(D0, TRUE, "bertrand", np.random.default_rng(2026))
res = {}
for n in (200, 3200):
    d = vd.design(n)
    out = {}
    for label, own, H in (("pre_merger", d.firms, d.ownership("bertrand")), ("post_merger", np.array(POST), d.ownership("bertrand", POST))):
        pstar, ok = solve_equilibrium(data["delta"], data["c"], d, TRUE["alpha"], TRUE["sp"], TRUE["sx"], H, p0=data["p"])
        gains = []
        for t in range(8):
            dl, ct = data["delta"][t:t + 1], data["c"][t]
            d1 = copy.copy(d); d1.x1 = d.x1[t:t + 1]
            for f in np.unique(own):
                idx = np.where(own == f)[0]

                def profit(pv):
                    pr = pstar[t:t + 1].copy(); pr[0, idx] = pv
                    sh = shares(dl, pr, d1, TRUE["alpha"], TRUE["sp"], TRUE["sx"])[0]
                    return float(((pr[0, idx] - ct[idx]) * sh[idx]).sum())
                p0 = pstar[t, idx]; base = profit(p0); best = base
                for st in [p0 * k for k in (0.5, 0.8, 1.2, 1.6)] + [p0 + 0.7, np.maximum(ct[idx] + 0.05, p0 - 0.7)]:
                    r = minimize(lambda v: -profit(v), st, method="L-BFGS-B", bounds=[(ct[i], ct[i] + 40) for i in idx])
                    best = max(best, -r.fun)
                gains.append((best - base) / abs(base))
        out[label] = dict(solver_ok=bool(ok), max_relative_profit_gain=float(max(gains)), n_firm_markets=len(gains))
    res[str(n)] = out
    print(n, json.dumps(out), flush=True)
json.dump(res, open("candidate_r1023/res_deviation_check.json", "w"), indent=1)
