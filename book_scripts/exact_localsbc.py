"""Chapter 9 and Problem 6 of Chapter 21: local simulation based calibration of the exact posterior itself.

The protocol of open_localsbc.py with the trained network replaced by the exact posterior of the risk difference
(independent Beta posteriors for the two arms under uniform priors). At a typical and at an extreme trial it records
the chi squared statistic of the 20 local rank counts over 200 repeats of the M = 400 protocol, the share of repeats
that reject uniformity at 5 percent, and the share of ranks in the upper half at M = 400,000. It also checks the
"two thirds in the upper half" statement for the normal example at ybar = 1.5.

Revised 23 October 2026 (r1023): the earlier version drew the control probability a independently of tau and clipped
b = a + tau; now (a, b) come from the independent beta posteriors and tau = b - a. Writes results/res_exact_localsbc.json. numpy and scipy only, so the thread count does not matter.
"""
import json, os, numpy as np
from scipy import stats

os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(20261014)
m, L = 50, 19


def local_exact(obs, M):
    """Local rank counts of the exact posterior at the observed trial obs = (y0, y1) out of m in each arm."""
    a = rng.beta(1 + obs[0], 1 + m - obs[0], M)  # control probability from its exact posterior
    b = rng.beta(1 + obs[1], 1 + m - obs[1], M)  # treated probability from its own independent exact posterior
    # (p0, p1) are independent given the data, so tau = b - a is an exact posterior draw and no clipping is needed
    y0, y1 = rng.binomial(m, a), rng.binomial(m, b)
    d1 = rng.beta(1 + y1[:, None], 1 + m - y1[:, None], (M, L))
    d0 = rng.beta(1 + y0[:, None], 1 + m - y0[:, None], (M, L))
    ranks = ((d1 - d0) < (b - a)[:, None]).sum(1)  # rank of the generating tau = b - a
    counts = np.bincount(ranks, minlength=L + 1)
    chi = stats.chisquare(counts)
    return counts, float(chi.statistic), float(chi.pvalue), float((ranks > L / 2).mean())


res = {}
for label, obs in [("typical (18 of 50, 27 of 50)", (18, 27)), ("extreme (1 of 50, 49 of 50)", (1, 49))]:
    chis, ps = [], []; pooled = np.zeros(L + 1)
    for _ in range(200):
        cnt, c, p, _ = local_exact(obs, 400)
        chis.append(c); ps.append(p); pooled += cnt
    chis, ps = np.array(chis), np.array(ps)
    _, c_big, _, up_big = local_exact(obs, 400000)
    res[label] = dict(obs=obs, repeats=200, M=400, chi_median=float(np.median(chis)),
                      chi_5th=float(np.quantile(chis, 0.05)), chi_95th=float(np.quantile(chis, 0.95)),
                      reject_share=float((ps < 0.05).mean()), upper_half_share_M400000=up_big,
                      rank_counts_per_400=[float(x) for x in pooled / 200], upper_half_share_M400=float(pooled[10:].sum() / pooled.sum()))

# normal example of the chapter: theta ~ N(0, 1), ten observations N(theta, 1), summary ybar, observed ybar = 1.5
mu, v = 10 * 1.5 / 11, 1 / 11
theta = rng.normal(mu, np.sqrt(v), 2_000_000)  # generating values from the exact posterior at ybar = 1.5
ybar = rng.normal(theta, np.sqrt(0.1))
z = (theta - 10 * ybar / 11) * np.sqrt(11)  # standardized: positive means the draw lies in the upper half
res["normal example, ybar = 1.5"] = dict(upper_half_share=float((z > 0).mean()), draws=2_000_000)

json.dump(res, open("results/res_exact_localsbc.json", "w"), indent=1)
print(json.dumps(res, indent=1))
