"""Exact population values of the front door simulation in make_figs.py (section 11, Chapter 14).

Design: U ~ Bernoulli(0.3); P(S = 1 | U) = expit(-0.5 + 2U); P(M = 1 | S) = expit(-2 + 4S);
P(C = 1 | M, U) = expit(-3 + 1.5M + 1.5U). S is smoking, M tar, C cancer, U the hidden genotype.

Reports P(S = 1), P(C = 1), the true risk difference P(C = 1 | do(S = 1)) - P(C = 1 | do(S = 0))
and the naive contrast P(C = 1 | S = 1) - P(C = 1 | S = 0). No random numbers are drawn: the values
are exact sums over U, M and S, so they are the population targets of the 100 replications in
make_figs.py. Run from the repository root: timeout 600 python3 -I book_scripts/frontdoor_dgp.py
Writes results/res_frontdoor_dgp.json.
"""
import json
import os
from math import exp


def expit(x):
    return 1.0 / (1.0 + exp(-x))


P_U = {0: 0.7, 1: 0.3}


def p_s(u):
    return expit(-0.5 + 2.0 * u)


def p_m(s):
    return expit(-2.0 + 4.0 * s)


def p_c(m, u):
    return expit(-3.0 + 1.5 * m + 1.5 * u)


def risk_given(s, u):
    """P(C = 1 | S = s, U = u): tar is drawn from P(M | S = s), C depends on M and U only."""
    return p_m(s) * p_c(1, u) + (1.0 - p_m(s)) * p_c(0, u)


def risk_do(s):
    """P(C = 1 | do(S = s)): average over the marginal law of U."""
    return sum(P_U[u] * risk_given(s, u) for u in (0, 1))


def risk_obs(s):
    """P(C = 1 | S = s): average over the law of U among people with S = s."""
    num = den = 0.0
    for u in (0, 1):
        w = P_U[u] * (p_s(u) if s == 1 else 1.0 - p_s(u))
        den += w
        num += w * risk_given(s, u)
    return num / den


p_smoke = sum(P_U[u] * p_s(u) for u in (0, 1))
p_cancer = p_smoke * risk_obs(1) + (1.0 - p_smoke) * risk_obs(0)
out = dict(
    P_S1=p_smoke,
    P_C1=p_cancer,
    risk_difference=risk_do(1) - risk_do(0),
    naive_contrast=risk_obs(1) - risk_obs(0),
)
os.makedirs('results', exist_ok=True)
with open('results/res_frontdoor_dgp.json', 'w') as f:
    json.dump(out, f, indent=2)
print({k: round(v, 4) for k, v in out.items()})
