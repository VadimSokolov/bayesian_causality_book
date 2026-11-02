# Check of the design behind the Figure fig:iv simulation in Chapter 12 (Instruments), as stated in the text:
# 500 replications of size 500, T = 0.5 Z + U + nu, Y = tau T + 2 U + eps, tau = 1, all of Z, U, nu, eps independent standard normal.
# Reports the mean of least squares, the median of the instrumental variable estimate, their standard deviations, the first stage F statistic
# and the smallest sample Cov(Z, T), the numbers quoted in the text about the strength of the instrument and the spread of the estimates.
# Run from the repository root: python book_scripts/iv_fig_check.py. Writes results/res_iv_figcheck.json (seeded, about a second).
import numpy as np, json
from scipy import stats

rng = np.random.default_rng(2024)
n, R, tau = 500, 500, 1.0
ols, iv, F, cov = [], [], [], []
for _ in range(R):
    Z, U, nu, eps = rng.normal(size=(4, n))
    T = 0.5 * Z + U + nu
    Y = tau * T + 2 * U + eps
    c = np.cov(Z, T)[0, 1]
    ols.append(np.cov(T, Y)[0, 1] / np.var(T, ddof=1))
    iv.append(np.cov(Z, Y)[0, 1] / c)
    zc = Z - Z.mean(); b = c / np.var(Z, ddof=1)
    r = (T - T.mean()) - b * zc
    F.append(b ** 2 * (zc @ zc) / (r @ r / (n - 2)))
    cov.append(c)
ols, iv, F, cov = map(np.array, (ols, iv, F, cov))
out = {
    'ols_mean': float(ols.mean()), 'iv_median': float(np.median(iv)),
    'ols_sd': float(ols.std(ddof=1)), 'iv_sd': float(iv.std(ddof=1)), 'sd_ratio': float(iv.std(ddof=1) / ols.std(ddof=1)),
    'F_mean': float(F.mean()), 'F_min': float(F.min()), 'cov_ZT_min': float(cov.min()),
    'iv_min': float(iv.min()), 'iv_max': float(iv.max()), 'iv_excess_kurtosis': float(stats.kurtosis(iv)),
    'theory_ols_mean': 1 + 2 / 2.25, 'theory_concentration': 500 * 0.25 / 2,
}
json.dump(out, open('results/res_iv_figcheck.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
