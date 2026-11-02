#!/usr/bin/env python3
"""Cross-sectional simulation: DML vs Bayesian DML with horseshoe prior.

Usage: python3 -u sim_cross_section.py --rep N
Output: results/cross_section_repN.csv

Compares OLS, LASSO-DML, RF-DML, and B-DML-HS (horseshoe) estimators
across 9 configurations: k_e in {5, 25, 50} x delta in {0.5, 1.0, 2.0}.
"""
import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from sklearn.linear_model import LassoCV, LogisticRegressionCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

warnings.filterwarnings("ignore")


# ======================================================================
# Gaussian draw from a Cholesky factor
# ======================================================================
def draw_from_precision(A, mean_rhs, lower=False):
    """Return x ~ N(m, A^{-1}) with m = A^{-1} mean_rhs.

    scipy's cho_factor returns an UPPER factor U with A = U^T U by default (lower=False), or a lower factor L
    with A = L L^T. Since A^{-1} = U^{-1} U^{-T}, the draw is m + U^{-1} z (solve with U, no transpose);
    for a lower factor it is m + L^{-T} z (solve with L, transposed). The earlier code solved with the
    transpose of the upper factor, which has covariance (U U^T)^{-1}, not A^{-1}.
    """
    c, low = cho_factor(A, lower=lower)
    m = cho_solve((c, low), mean_rhs)
    z = np.random.randn(len(mean_rhs))
    return m + solve_triangular(c, z, lower=low, trans=('T' if low else 'N'))


# ======================================================================
# Polya-Gamma sampling (series truncation for PG(1, c))
# ======================================================================
def sample_pg_batch(c_vec, K=50):
    """Sample PG(1, c_i) for each element of c_vec using series truncation.

    PG(1, c) = (1/2pi^2) sum_{k=0}^{K-1} G_k / ((k+0.5)^2 + c^2/(4pi^2))
    where G_k ~ Exp(1).
    """
    n = len(c_vec)
    g = np.random.exponential(1.0, (n, K))
    k_vals = np.arange(K)[None, :]           # 1 x K
    c2 = (c_vec[:, None] / (2 * np.pi)) ** 2  # n x 1
    denom = (k_vals + 0.5) ** 2 + c2          # n x K
    return np.sum(g / denom, axis=1) / (2 * np.pi ** 2)


# ======================================================================
# Horseshoe logistic regression via Polya-Gamma Gibbs sampler
# ======================================================================
def horseshoe_logistic(X, D, n_iter=600, n_burn=200):
    """Horseshoe logistic regression. Returns posterior mean of beta.

    Model: D_i ~ Bernoulli(sigma(X_i' beta))
    Prior: beta_j | lam_j, tau ~ N(0, lam_j^2 tau^2)
           lam_j ~ C+(0,1), tau ~ C+(0,1)
    Sampler: Polya-Gamma data augmentation.
    """
    n, p = X.shape

    # Initialize
    beta = np.zeros(p)
    lam2 = np.ones(p)
    nu = np.ones(p)
    tau2 = 1.0
    xi = 1.0

    n_keep = n_iter - n_burn
    beta_sum = np.zeros(p)

    for t in range(n_iter):
        # 1. PG augmentation
        psi = X @ beta
        psi = np.clip(psi, -20, 20)
        omega = sample_pg_batch(np.abs(psi), K=50)
        omega = np.clip(omega, 1e-12, 1e6)

        # 2. Sample beta | omega, lam, tau
        kappa = D - 0.5
        prior_prec = 1.0 / (lam2 * tau2 + 1e-12)
        V_inv = (X.T * omega) @ X + np.diag(prior_prec)
        V_inv = (V_inv + V_inv.T) / 2
        try:
            beta = draw_from_precision(V_inv, X.T @ kappa)
        except np.linalg.LinAlgError:
            V_inv += 1e-6 * np.eye(p)
            beta = draw_from_precision(V_inv, X.T @ kappa)

        # 3. Sample local scales (vectorized)
        rates_lam = beta ** 2 / (2 * tau2 + 1e-12) + 1.0 / (nu + 1e-12)
        lam2 = 1.0 / np.random.gamma(1.0, 1.0 / np.maximum(rates_lam, 1e-12))
        nu = 1.0 / np.random.gamma(1.0, 1.0 / (1.0 + 1.0 / np.maximum(lam2, 1e-12)))

        # 4. Sample global scale
        rate_tau = np.sum(beta ** 2 / (2 * np.maximum(lam2, 1e-12))) + 1.0 / (xi + 1e-12)
        tau2 = 1.0 / np.random.gamma((p + 1) / 2, 1.0 / max(rate_tau, 1e-12))
        xi = 1.0 / np.random.gamma(1.0, 1.0 / (1.0 + 1.0 / max(tau2, 1e-12)))

        if t >= n_burn:
            beta_sum += beta

    return beta_sum / n_keep


# ======================================================================
# Horseshoe linear regression via Gibbs sampler
# ======================================================================
def horseshoe_linear(X, Y, n_iter=600, n_burn=200):
    """Horseshoe linear regression. Returns posterior mean of beta.

    Model: Y_i = X_i' beta + eps_i, eps_i ~ N(0, sigma^2)
    Prior: beta_j | lam_j, tau ~ N(0, lam_j^2 tau^2 sigma^2)
    """
    n, p = X.shape

    beta = np.zeros(p)
    sigma2 = np.var(Y) + 1e-4
    lam2 = np.ones(p)
    nu = np.ones(p)
    tau2 = 1.0
    xi = 1.0

    XtX = X.T @ X
    XtY = X.T @ Y

    n_keep = n_iter - n_burn
    beta_sum = np.zeros(p)

    for t in range(n_iter):
        # 1. Sample beta
        prior_prec = 1.0 / (lam2 * tau2 * sigma2 + 1e-12)
        V_inv = XtX / sigma2 + np.diag(prior_prec)
        V_inv = (V_inv + V_inv.T) / 2
        try:
            beta = draw_from_precision(V_inv, XtY / sigma2)
        except np.linalg.LinAlgError:
            V_inv += 1e-6 * np.eye(p)
            beta = draw_from_precision(V_inv, XtY / sigma2)

        # 2. Sample sigma2 | beta, lam, tau. With beta_j ~ N(0, lam_j^2 tau^2 sigma^2) and p(sigma^2) proportional
        # to 1/sigma^2 the full conditional is inverse gamma, shape (n + p)/2, rate {RSS + sum_j beta_j^2/(lam_j^2 tau^2)}/2
        # (as in book_scripts/ric_sim.py).
        resid = Y - X @ beta
        shape = (n + p) / 2
        rate = (np.sum(resid ** 2) + np.sum(beta ** 2 / np.maximum(lam2 * tau2, 1e-12))) / 2
        sigma2 = 1.0 / np.random.gamma(shape, 1.0 / max(rate, 1e-12))
        sigma2 = np.clip(sigma2, 1e-6, 1e6)

        # 3. Local scales
        rates_lam = beta ** 2 / (2 * sigma2 * tau2 + 1e-12) + 1.0 / (nu + 1e-12)
        lam2 = 1.0 / np.random.gamma(1.0, 1.0 / np.maximum(rates_lam, 1e-12))
        nu = 1.0 / np.random.gamma(1.0, 1.0 / (1.0 + 1.0 / np.maximum(lam2, 1e-12)))

        # 4. Global scale
        rate_tau = np.sum(beta ** 2 / (2 * sigma2 * np.maximum(lam2, 1e-12))) + 1.0 / (xi + 1e-12)
        tau2 = 1.0 / np.random.gamma((p + 1) / 2, 1.0 / max(rate_tau, 1e-12))
        xi = 1.0 / np.random.gamma(1.0, 1.0 / (1.0 + 1.0 / max(tau2, 1e-12)))

        if t >= n_burn:
            beta_sum += beta

    return beta_sum / n_keep


# ======================================================================
# DML cross-fitting infrastructure
# ======================================================================
def dml_plr(Y, D, X, est_outcome, est_propensity, K=5, rng=None):
    """DML estimator for the PLR model with K-fold cross-fitting.

    Returns (theta_hat, se).
    """
    n = len(Y)
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.permutation(n)
    folds = np.array_split(indices, K)

    # Collect residuals across all folds
    Y_tilde = np.zeros(n)
    V_hat = np.zeros(n)

    for k in range(K):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != k])

        g_hat = est_outcome(X[train_idx], Y[train_idx], X[test_idx])
        m_hat = est_propensity(X[train_idx], D[train_idx], X[test_idx])

        Y_tilde[test_idx] = Y[test_idx] - g_hat
        V_hat[test_idx] = D[test_idx] - m_hat

    # PLR estimate
    theta = np.sum(V_hat * Y_tilde) / np.sum(V_hat ** 2)

    # Sandwich SE
    psi = V_hat * (Y_tilde - theta * V_hat)
    J = np.mean(V_hat ** 2)
    se = np.sqrt(np.mean(psi ** 2)) / (J * np.sqrt(n))

    return theta, se


# ======================================================================
# Nuisance estimators (outcome and propensity)
# ======================================================================
def lasso_outcome(X_tr, Y_tr, X_te):
    m = LassoCV(cv=5, max_iter=5000, n_jobs=1, random_state=42)
    m.fit(X_tr, Y_tr)
    return m.predict(X_te)

def lasso_propensity(X_tr, D_tr, X_te):
    m = LogisticRegressionCV(
        penalty="l1", solver="saga", cv=5, max_iter=5000,
        n_jobs=1, random_state=42, Cs=20
    )
    m.fit(X_tr, D_tr.astype(int))
    return np.clip(m.predict_proba(X_te)[:, 1], 0.01, 0.99)

def rf_outcome(X_tr, Y_tr, X_te):
    m = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=1)
    m.fit(X_tr, Y_tr)
    return m.predict(X_te)

def rf_propensity(X_tr, D_tr, X_te):
    m = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=1)
    m.fit(X_tr, D_tr.astype(int))
    p = m.predict_proba(X_te)
    return np.clip(p[:, 1] if p.shape[1] > 1 else p[:, 0], 0.01, 0.99)

def hs_outcome(X_tr, Y_tr, X_te):
    beta = horseshoe_linear(X_tr, Y_tr, n_iter=600, n_burn=200)
    return X_te @ beta

def hs_propensity(X_tr, D_tr, X_te):
    beta = horseshoe_logistic(X_tr, D_tr, n_iter=600, n_burn=200)
    return np.clip(expit(X_te @ beta), 0.01, 0.99)


# ======================================================================
# Main simulation loop
# ======================================================================
def run_one_config(n, p, k_e, k_g, delta, tau_true, rng):
    """Generate data and run all estimators for one configuration."""
    # DGP
    X = rng.standard_normal((n, p))
    beta_e = np.zeros(p)
    beta_e[:k_e] = 1.0 / np.sqrt(k_e)
    beta_g = np.zeros(p)
    beta_g[:k_g] = 1.0 / np.sqrt(k_g)

    prob = expit(delta * X @ beta_e)
    D = rng.binomial(1, prob, n).astype(float)
    Y = tau_true * D + X @ beta_g + rng.standard_normal(n)

    results = {}

    # 1. OLS (regress Y on [D, X])
    XD = np.column_stack([D.reshape(-1, 1), X])
    ols = LinearRegression(fit_intercept=False).fit(XD, Y)
    results["OLS"] = (ols.coef_[0], np.nan)

    # 2. LASSO-DML
    fold_rng = np.random.default_rng(rng.integers(0, 2**31))
    results["LASSO_DML"] = dml_plr(Y, D, X, lasso_outcome, lasso_propensity, K=5, rng=fold_rng)

    # 3. RF-DML
    fold_rng = np.random.default_rng(rng.integers(0, 2**31))
    results["RF_DML"] = dml_plr(Y, D, X, rf_outcome, rf_propensity, K=5, rng=fold_rng)

    # 4. B-DML-HS
    fold_rng = np.random.default_rng(rng.integers(0, 2**31))
    results["B_DML_HS"] = dml_plr(Y, D, X, hs_outcome, hs_propensity, K=5, rng=fold_rng)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, required=True)
    args = parser.parse_args()

    seed = args.rep * 31 + 7
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    n = 1000
    p = 100
    tau_true = 1.0
    k_g = 5

    all_results = []
    t0 = time.time()

    for k_e in [5, 25, 50]:
        for delta in [0.5, 1.0, 2.0]:
            t1 = time.time()
            res = run_one_config(n, p, k_e, k_g, delta, tau_true, rng)
            elapsed = time.time() - t1

            for method, (theta, se) in res.items():
                bias = theta - tau_true
                covered = np.abs(bias) < 1.96 * se if not np.isnan(se) else np.nan
                all_results.append({
                    "rep": args.rep, "k_e": k_e, "delta": delta,
                    "method": method, "theta": theta, "se": se,
                    "bias": bias, "covered": covered,
                })

            print(
                f"k_e={k_e:2d} delta={delta:.1f} | "
                + " | ".join(f"{m}={res[m][0]:.4f}" for m in res)
                + f" | {elapsed:.1f}s"
            )

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(all_results).to_csv(
        f"results/cross_section_rep{args.rep}.csv", index=False
    )
    print(f"Rep {args.rep} done in {time.time() - t0:.1f}s")
