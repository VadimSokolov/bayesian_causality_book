"""Chapters 3 and 13: the simulated milk trial of Figure fig:milk when the teachers select on something nobody recorded.

The teachers judge each child by its condition U, which is not recorded; the recorded initial size S is standard bivariate normal
with U, correlation r. The gain is G = 1 - 0.3 S - 0.3 U + 0.2 T + e with e ~ N(0, 0.5^2), and milk is given with probability
expit(-1.2 U), the rule of Figure fig:milk with U in place of S (Exercise ex:ch03:2 derives the bias).

Chapter 3: for r = 0.95, 0.8, 0.5 and 0, the coefficient of T in the least squares regression of G on T and S, averaged over 1,000
simulated trials of 2,000 children, with its Monte Carlo standard error.

Chapter 13 (Lanarkshire revisited): one simulated trial of 2,000 children at r = 0.8, analysed as Section sec:sensitivity proposes.
The adjusted estimate and its standard error; the size benchmarks, a_S, the coefficient of T in a regression of S on T, and b_S, the
coefficient of S in the regression of G on T and S; and the posterior of the effect tau = tau_hat - a b_U + se z under the priors
a ~ N(k a_S, (k a_S)^2) and b_U ~ N(k b_S, (k b_S)^2), a hidden confounder k times as strong as size on each arm with the signs of
size, for k = 1/4, 1/2 and 1, from 200,000 draws. Also the hidden score's own a, from one simulated sample of a million children.

Writes results/res_milk_unrecorded.json. Requires numpy. A few seconds.
"""
import json
import numpy as np

expit = lambda z: 1 / (1 + np.exp(-z))


def trial(rng, r, n=2000):
    U = rng.normal(size=n); S = r * U + np.sqrt(1 - r ** 2) * rng.normal(size=n)
    T = rng.binomial(1, expit(-1.2 * U)); G = 1 - 0.3 * S - 0.3 * U + 0.2 * T + rng.normal(0, 0.5, n)
    return S, T, G


def ols(y, *cols):
    """Least squares with an intercept: coefficients (intercept first) and their classical standard errors."""
    X = np.column_stack([np.ones(len(y))] + list(cols)); beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta; s2 = res @ res / (len(y) - X.shape[1])
    return beta, np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))


out = {"design": "G = 1 - 0.3 S - 0.3 U + 0.2 T + N(0, 0.5^2), P(T = 1) = expit(-1.2 U), corr(S, U) = r, 2,000 children", "truth": 0.2}
rng = np.random.default_rng(1931)
out["adjusted_mean"] = {}
for r in [0.95, 0.8, 0.5, 0.0]:
    est = np.array([ols(G, T, S)[0][1] for S, T, G in (trial(rng, r) for _ in range(1000))])
    out["adjusted_mean"][str(r)] = dict(mean=float(est.mean()), mc_se=float(est.std(ddof=1) / np.sqrt(len(est))), sd=float(est.std(ddof=1)))

rng = np.random.default_rng(1932)
S, T, G = trial(rng, 0.8)
(b0, tau_hat, b_S), (_, se, _) = ols(G, T, S)
a_S = float(ols(S, T)[0][1])
sens = {"tau_hat": float(tau_hat), "se": float(se), "a_S": a_S, "b_S": float(b_S), "ci_classical": [float(tau_hat - 1.96 * se), float(tau_hat + 1.96 * se)]}
z = np.random.default_rng(1933).normal(size=(3, 200000))
for k in [0.25, 0.5, 1.0]:
    a = k * a_S * (1 + z[0]); b = k * b_S * (1 + z[1]); tau = tau_hat - a * b + se * z[2]
    sens[str(k)] = dict(mean=float(tau.mean()), ci=np.quantile(tau, [0.025, 0.975]).tolist(), p_pos=float(np.mean(tau > 0)),
                        p_below_truth=float(np.mean(tau < 0.2)))
rng = np.random.default_rng(1934); U1 = rng.normal(size=1000000); S1 = 0.8 * U1 + 0.6 * rng.normal(size=1000000)
sens["a_U"] = float(ols(U1, rng.binomial(1, expit(-1.2 * U1)), S1)[0][1]); sens["b_U"] = -0.3     # the hidden score's own coefficients
out["sensitivity_r0.8"] = sens
json.dump(out, open("results/res_milk_unrecorded.json", "w"), indent=1)
print(json.dumps(out, indent=1))
