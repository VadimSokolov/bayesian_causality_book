"""Antitrust quantities as functions of (theta2, conduct) and the data (Propositions 1 and 2)."""
import numpy as np
from blp_core import (contraction, ind_probs, alpha_i, markups, price_derivs,
                      solve_equilibrium, consumer_surplus)

POST_FIRMS = (0, 0, 0, 0, 2, 2)     # firms A and B merge
MERGING = [0, 1, 2, 3]


def structural(d, s, p, th2, conduct, dl0=None, merger_conduct=None):
    """Markups, costs, Lerner, diversion, GUPPI, merger price and welfare effects.

    conduct          : conduct used to recover costs from the pre merger data
    merger_conduct   : conduct after the merger (defaults to conduct)."""
    a, sp, sx = th2
    dl = contraction(s, p, d, a, sp, sx, delta0=dl0, tol=1e-11)
    ai = alpha_i(a, sp, d.nu)
    sij = ind_probs(dl, p, d, a, sp, sx)
    H = d.ownership(conduct)
    m = markups(s, sij, ai, H)
    c = p - m
    Dm, _, _ = price_derivs(sij, ai)
    own = np.diagonal(Dm, axis1=1, axis2=2)
    div = -np.transpose(Dm, (0, 2, 1)) / own[:, :, None]
    div[:, range(d.J), range(d.J)] = 0.0
    guppi = np.zeros_like(p)
    for j in MERGING:
        partners = [k for k in MERGING if d.firms[k] != d.firms[j]]
        guppi[:, j] = (div[:, j, partners] * m[:, partners]).sum(1) / p[:, j]
    mc_ = merger_conduct or conduct
    if mc_ == "bertrand":
        p1, ok = solve_equilibrium(dl, c, d, a, sp, sx, d.ownership("bertrand", POST_FIRMS), p0=p)
    elif mc_ == "collusion":
        p1, ok = solve_equilibrium(dl, c, d, a, sp, sx, d.ownership("collusion"), p0=p)
    elif mc_ == "partial":
        p1, ok = solve_equilibrium(dl, c, d, a, sp, sx, d.ownership("partial", POST_FIRMS), p0=p)
    else:
        p1, ok = c.copy(), True
    cs0 = consumer_surplus(dl, p, d, a, sp, sx)
    cs1 = consumer_surplus(dl, p1, d, a, sp, sx)
    elas = Dm * p[:, None, :] / s[:, :, None]
    return dict(m=m, c=c, lerner=m / p, div=div, guppi=guppi, dp=(p1 - p) / p, dcs=cs1 - cs0,
                elas=elas, ok=ok, delta=dl)


def estimands(r):
    """Scalar estimands: industry Lerner, merging party price change, consumer surplus change."""
    return np.array([r["lerner"].mean(), r["dp"][:, MERGING].mean(), r["dcs"].mean()])
