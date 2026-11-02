"""Chapter 13, two worked illustrations.

1. The probability of necessity from experimental and observational data combined (Section sec:court and the joint law section).
   The joint law of (X, Y(1), Y(0)) has eight cells q[x, a, b], X the exposure people choose, a = Y(1), b = Y(0); the prior is
   Dirichlet with 1/2 on every cell, the choice under which the position u of PNS within its experimental bounds is uniform a priori. Data, the counts of the
   chapter's numerical illustration: an experiment with 500 per arm, 150 exposed and 50 unexposed falling ill, and an observational
   study of 1,000 people, 400 exposed of whom 132 fall ill and 600 unexposed of whom 48 fall ill. Each count is a sum of cells (the
   exposed arm sees Y(1), the unexposed arm Y(0), the observational study X and Y(X)), so the sampler is Gibbs with data
   augmentation: allocate every count to its cells in proportion to q, then draw q from its Dirichlet posterior. Reported: the
   posterior of PN = q[1,1,0] / (q[1,1,0] + q[1,1,1]) and of its lower bound, (q[1,1,0] - q[1,0,1]) / (q[1,1,0] + q[1,1,1]) (Tian and Pearl),
   the posterior probabilities that each exceeds 1/2, and the same with every count multiplied by ten. One million iterations after
   100,000 of burn in (four million for the tenfold data, which mixes more slowly); effective sample sizes by batch means.
2. Rosenbaum's sensitivity value for the sign test in Hammond's (1964) matched pairs of smokers and nonsmokers as analysed by
   Rosenbaum (2002): of 122 pairs in which exactly one man died of lung cancer, the smoker died in 110. Under a bias of at most Gamma
   the smoker's chance of being the one who died is at most Gamma / (1 + Gamma) in each pair, so the upper bound on the one sided
   p value is the binomial tail P(B >= 110), B ~ Bin(122, Gamma / (1 + Gamma)). Reported for Gamma = 1 to 6, and the sensitivity
   value, the Gamma at which the bound reaches 0.05.

Writes results/res_bounds_examples.json. Requires numpy and scipy. About ten minutes.
"""
import json
import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq

# cell index 4x + 2a + b
GROUPS = lambda k: [  # (count, cells)
    (150 * k, [2, 3, 6, 7]), (350 * k, [0, 1, 4, 5]),          # exposed arm: Y(1) = 1, Y(1) = 0
    (50 * k, [1, 3, 5, 7]), (450 * k, [0, 2, 4, 6]),           # unexposed arm: Y(0) = 1, Y(0) = 0
    (132 * k, [6, 7]), (268 * k, [4, 5]), (48 * k, [1, 3]), (552 * k, [0, 2])]  # observational: (x, y) = (1,1), (1,0), (0,1), (0,0)


def ess(x, nb=100):
    m = len(x) // nb; b = x[:m * nb].reshape(nb, m).mean(1)
    return float(len(x) * x.var() / (m * b.var(ddof=1)))


def gibbs(k, iters, burn, seed):
    rng = np.random.default_rng(seed); q = np.full(8, 1 / 8); groups = GROUPS(k); draws = np.empty((iters, 8))
    for it in range(burn + iters):
        c = np.full(8, 0.5)
        for n, cells in groups:
            p = q[cells]; c[cells] += rng.multinomial(n, p / p.sum())
        q = rng.dirichlet(c)
        if it >= burn: draws[it - burn] = q
    pn = draws[:, 6] / (draws[:, 6] + draws[:, 7]); lb = np.maximum(0, (draws[:, 6] - draws[:, 5]) / (draws[:, 6] + draws[:, 7]))
    summ = lambda v: dict(median=float(np.median(v)), mean=float(v.mean()), ci=np.quantile(v, [0.025, 0.975]).tolist(),
                          p_above_half=float(np.mean(v > 0.5)), ess=ess(v))
    return dict(scale=k, iterations=iters, burn_in=burn, pn=summ(pn), lower_bound=summ(lb))


out = {"pn": [gibbs(1, 1000000, 100000, 12), gibbs(10, 4000000, 100000, 13)]}
G = [1, 2, 3, 4, 5, 6]
out["rosenbaum"] = dict(pairs=122, smoker_died=110, gamma=G, p_upper=[float(binom.sf(109, 122, g / (1 + g))) for g in G],
                        sensitivity_value=float(brentq(lambda g: binom.sf(109, 122, g / (1 + g)) - 0.05, 1, 20)))
json.dump(out, open("results/res_bounds_examples.json", "w"), indent=1)
print(json.dumps(out, indent=1))
