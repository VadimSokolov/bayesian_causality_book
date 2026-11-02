"""Prior scale numbers quoted in Section comp:sec:horseshoe (Chapter 9).

The experiment is the one of the book Generative Bayesian Computation (Polson and Sokolov 2026, the chapter on Bayesian inference, example
"Horseshoe Sparse Regression"): a fixed design of n = 50 rows and d = 10 columns of independent standard normals (generator seeded [2026, 0]),
error standard deviation sigma = 0.5, and N = 80,000 draws of (tau_g, lambda, beta) from the horseshoe prior with beta_j | lambda_j, tau_g ~
N(0, lambda_j^2 tau_g^2) and standard half Cauchy tau_g and lambda_j (generator seeded [2026, 1]). This script repeats that cell and records what
the chapter quotes: the range over the ten coefficients of the sample standard deviation of the 80,000 draws, the median of |beta_j| over all
draws, the share of draws within 0.01 standard deviations of the mean after standardizing each coefficient, the error in beta_j that an error of
0.001 on the standardized scale of the smallest-spread coefficient implies, and the range over the coefficients of the standard deviation
sigma * sqrt{(X'X)^{-1}_jj} of beta_j - b_j given beta. The evaluation tables of the experiment (calibration, widths and Kolmogorov and Smirnov
distances of the networks, the coverage and the running time of the Gibbs sampler) are in results/gbc_book_hs, copied from the cache of that book.

usage (from the repository root): python book_scripts/hs_scale_check.py
writes results/res_hs_scale.json
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

d, n, sigma, N = 10, 50, 0.5, 80_000
X = np.random.default_rng([2026, 0]).standard_normal((n, d))


def prior_draws(rng, m):
    """m draws of beta from the horseshoe prior and of y given beta."""
    tau_g = np.abs(rng.standard_cauchy(m))
    lam = np.abs(rng.standard_cauchy((m, d)))
    beta = rng.normal(0.0, lam * tau_g[:, None])
    y = beta @ X.T + sigma * rng.standard_normal((m, n))
    return beta, y


def main():
    beta, y = prior_draws(np.random.default_rng([2026, 1]), N)
    b = np.linalg.solve(X.T @ X, X.T @ y.T).T
    sd = beta.std(axis=0)
    out = dict(
        design="n=50, d=10 standard normal design (seed [2026, 0]), sigma=0.5, 80,000 horseshoe prior draws (seed [2026, 1])",
        sd_beta_min=float(sd.min()), sd_beta_max=float(sd.max()),
        median_abs_beta=float(np.median(np.abs(beta))),
        percent_within_0_01_sd_of_mean=float(100 * np.mean(np.abs(beta - beta.mean(axis=0)) < 0.01 * sd)),
        error_in_beta_for_0_001_standardized_min=float(0.001 * sd.min()),
        sd_b_equals_sd_beta_within_1_percent=bool(np.allclose(b.std(axis=0), sd, rtol=0.01)),
        sd_of_b_minus_beta_min=float((sigma * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))).min()),
        sd_of_b_minus_beta_max=float((sigma * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))).max()),
    )
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "res_hs_scale.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
