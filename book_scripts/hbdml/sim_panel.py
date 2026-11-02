#!/usr/bin/env python3
"""Panel simulation: DML vs Bayesian DML with horseshoe prior and MAR dropout.

Usage: python3 -u sim_panel.py --rep N [--nrep M]
Output: results/panel_rep{r}.csv for the M replications r = (N-1)*M+1, ..., N*M (default M = 1).

Book copy, version of 2026-10-23 (round 1 review items F1, F8, F15 of Chapter 8; the earlier versions are the
baseline and the 2026-10-13 change to panel_baipw_dml, which fitted the dropout hazard with an intercept and
evaluated it at each unit's own lagged outcome).

Changes in this version
  F1   the horseshoe samplers in sim_cross_section.py draw beta with covariance A^{-1} (they drew with (U U^T)^{-1})
       and horseshoe_linear updates sigma^2 with shape (n+p)/2 and rate {RSS + sum_j beta_j^2/(lam_j^2 tau^2)}/2.
  F8   a direct within OLS row: demeaned Y on demeaned treatment and all p demeaned covariates, no cross fitting,
       unit clustered standard error. The earlier "OLS" row (DML with least squares nuisances) is renamed
       DML_LS_FE. Units observed in fewer than two periods are dropped before demeaning, for every estimator.
  F15  cross fitting by unit: each unit's whole history is in one of five folds (the inner five folds that tune the
       lasso penalty are also formed by unit, GroupKFold). The dropout hazard of the weighted estimator is fitted on
       the units of the four training folds and evaluated for the units of the validation fold. Standard errors are
       clustered by unit: V = G/(G-1) * sum_i S_i^2 / (sum_it w_it V_it^2)^2, S_i = sum_t w_it V_it (Ytilde_it - theta V_it)
       with w = 1 for the unweighted rows and G the number of units used. Three dropout conditions (none,
       expit(-2 + 0.5 Y_{t-1}), expit(-2 + Y_{t-1})), 200 and 1,000 units, and an extra row that uses the true
       survival weights (B_AIPW_ORACLE). The 2,000 replications are split over 500 array tasks of four.
"""
import argparse
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# Import horseshoe samplers from the cross-section script
# (same directory on Hopper)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_cross_section import (
    horseshoe_linear, horseshoe_logistic, sample_pg_batch,
)


# Panel-specific nuisance estimators (both outcome and propensity are continuous after demeaning).
# Each takes (X_train, y_train, X_test, groups_train); groups tune the lasso penalty by unit.
def _lasso(X_tr, y_tr, X_te, groups):
    cv = list(GroupKFold(n_splits=5).split(X_tr, y_tr, groups))
    m = LassoCV(cv=cv, max_iter=5000, n_jobs=1, random_state=42)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)

def panel_lasso_outcome(X_tr, Y_tr, X_te, groups):
    return _lasso(X_tr, Y_tr, X_te, groups)

def panel_lasso_propensity(X_tr, D_tr, X_te, groups):
    """After within-demeaning, D is continuous, so use linear regression."""
    return _lasso(X_tr, D_tr, X_te, groups)

def panel_hs_outcome(X_tr, Y_tr, X_te, groups):
    beta = horseshoe_linear(X_tr, Y_tr, n_iter=600, n_burn=200)
    return X_te @ beta

def panel_hs_propensity(X_tr, D_tr, X_te, groups):
    beta = horseshoe_linear(X_tr, D_tr, n_iter=600, n_burn=200)
    return X_te @ beta

def panel_ls(X_tr, y_tr, X_te, groups):
    return LinearRegression(fit_intercept=False).fit(X_tr, y_tr).predict(X_te)


# ======================================================================
# Panel DGP
# ======================================================================
def generate_panel(n_units, T, p, k_e, k_g, delta, theta_true, dropout_gamma, rng):
    """Generate panel data with optional MAR monotone dropout.

    Args:
        dropout_gamma: (gamma_0, gamma_1) for P(R_{it}=0|R_{i,t-1}=1) = sigma(gamma_0 + gamma_1*Y_{i,t-1}).
                       Set gamma_1=0 for no dropout.
    Returns:
        Y_full, D_full, X_full, alpha, R: arrays of shape (n_units, T) or (n_units, T, p)
    """
    gamma_0, gamma_1 = dropout_gamma

    beta_e = np.zeros(p)
    beta_e[:k_e] = 1.0 / np.sqrt(k_e)
    beta_g = np.zeros(p)
    beta_g[:k_g] = 1.0 / np.sqrt(k_g)

    alpha = rng.standard_normal(n_units)
    X = rng.standard_normal((n_units, T, p))
    eps = rng.standard_normal((n_units, T))

    # Treatment
    prob = expit(delta * np.einsum("itp,p->it", X, beta_e))
    D = rng.binomial(1, prob).astype(float)

    # Outcome
    Y = alpha[:, None] + theta_true * D + np.einsum("itp,p->it", X, beta_g) + eps

    # Monotone MAR dropout
    R = np.ones((n_units, T), dtype=int)
    if gamma_1 != 0:
        for t in range(1, T):
            dropout_prob = expit(gamma_0 + gamma_1 * Y[:, t - 1])
            drop = rng.binomial(1, dropout_prob)
            R[:, t] = R[:, t - 1] * (1 - drop)

    return Y, D, X, alpha, R


# ======================================================================
# Within transformation and unit clustered estimators
# ======================================================================
def within_demean(Y, D, X, R, weights=None):
    """Within-unit demeaning using observed periods only (weighted means when weights are given).

    Units observed in fewer than two periods are dropped. Returns demeaned Y, D, X stacked over the observed
    unit periods, the unit index of every row and the weight of every row (1 when weights is None).
    """
    n_units, T = Y.shape
    if weights is None:
        weights = np.ones((n_units, T))
    Yl, Dl, Xl, Ul, Wl = [], [], [], [], []
    for i in range(n_units):
        obs = R[i] == 1
        if obs.sum() < 2:
            continue
        wi = weights[i, obs]
        Yl.append(Y[i, obs] - np.average(Y[i, obs], weights=wi))
        Dl.append(D[i, obs] - np.average(D[i, obs], weights=wi))
        Xl.append(X[i, obs] - np.average(X[i, obs], axis=0, weights=wi))
        Ul.append(np.full(obs.sum(), i))
        Wl.append(wi)
    return (np.concatenate(Yl), np.concatenate(Dl), np.concatenate(Xl),
            np.concatenate(Ul), np.concatenate(Wl))


def cluster_score_estimate(Yt, V, W, uid):
    """theta = sum w V Yt / sum w V^2 and its unit clustered standard error.

    var = G/(G-1) * sum_i S_i^2 / (sum w V^2)^2 with S_i = sum_t w V (Yt - theta V) over the rows of unit i.
    """
    denom = np.sum(W * V ** 2)
    if denom < 1e-10:
        return np.nan, np.nan
    theta = np.sum(W * V * Yt) / denom
    psi = W * V * (Yt - theta * V)
    units, inv = np.unique(uid, return_inverse=True)
    G = len(units)
    S = np.bincount(inv, weights=psi, minlength=G)
    var = G / (G - 1.0) * np.sum(S ** 2) / denom ** 2
    return theta, np.sqrt(var)


def panel_within_ols(Y, D, X, R):
    """Direct within estimator: demeaned Y on demeaned D and all demeaned X, unit clustered standard error."""
    Yd, Dd, Xd, uid, _ = within_demean(Y, D, X, R)
    if len(Yd) < 20:
        return np.nan, np.nan
    Z = np.column_stack([Dd, Xd])
    coef = np.linalg.lstsq(Z, Yd, rcond=None)[0]
    e = Yd - Z @ coef
    units, inv = np.unique(uid, return_inverse=True)
    G = len(units)
    S = np.zeros((G, Z.shape[1]))
    np.add.at(S, inv, Z * e[:, None])
    bread = np.linalg.inv(Z.T @ Z)
    V = G / (G - 1.0) * bread @ (S.T @ S) @ bread
    return coef[0], np.sqrt(V[0, 0])


def unit_folds(n_units, K, rng):
    """Random assignment of every unit to one of K folds (as equal as possible)."""
    fold_of_unit = np.zeros(n_units, dtype=int)
    for k, idx in enumerate(np.array_split(rng.permutation(n_units), K)):
        fold_of_unit[idx] = k
    return fold_of_unit


def panel_dml_fe(Y, D, X, R, est_outcome, est_propensity, K=5, rng=None, weights=None, fold_of_unit=None):
    """Panel FE-DML with cross fitting by unit: within-demean, then cross fitted partialling out.

    weights (n_units x T) turns on the weighted demeaning and the weighted score; fold_of_unit may be passed
    (it is when the dropout model was cross fitted on the same folds).
    """
    Yd, Dd, Xd, uid, W = within_demean(Y, D, X, R, weights)
    n = len(Yd)
    if n < 20:
        return np.nan, np.nan
    if fold_of_unit is None:
        if rng is None:
            rng = np.random.default_rng()
        fold_of_unit = unit_folds(Y.shape[0], K, rng)
    fold_row = fold_of_unit[uid]

    Y_tilde = np.zeros(n)
    V_hat = np.zeros(n)
    for k in range(K):
        te = fold_row == k
        tr = ~te
        if te.sum() == 0:
            continue
        g_hat = est_outcome(Xd[tr], Yd[tr], Xd[te], uid[tr])
        m_hat = est_propensity(Xd[tr], Dd[tr], Xd[te], uid[tr])
        Y_tilde[te] = Yd[te] - g_hat
        V_hat[te] = Dd[te] - m_hat

    return cluster_score_estimate(Y_tilde, V_hat, W, uid)


# ======================================================================
# IPW-corrected panel DML (B-AIPW-DML)
# ======================================================================
def survival_weights(Y, beta_drop):
    """1 / P(still observed at t) = 1 / prod_{s=1}^t (1 - h_s), h_s = expit(b0 + b1 Y_{s-1}) clipped to [0.01, 0.99]."""
    h = np.clip(expit(beta_drop[0] + beta_drop[1] * Y[:, :-1]), 0.01, 0.99)
    cum = np.cumprod(1.0 - h, axis=1)
    w = np.ones(Y.shape)
    w[:, 1:] = 1.0 / np.maximum(cum, 0.01)
    return w


def fit_hazard(Y, R, units):
    """Horseshoe logistic fit of P(drop at t | alive at t-1, Y_{t-1}) = expit(b0 + b1 Y_{t-1}) on the given units."""
    Yu, Ru = Y[units], R[units]
    at_risk = Ru[:, :-1] == 1
    drop_Y = Yu[:, :-1][at_risk].reshape(-1, 1)
    drop_R = (1 - Ru[:, 1:])[at_risk]
    drop_Z = np.column_stack([np.ones(len(drop_Y)), drop_Y])
    if drop_R.sum() > 5 and (1 - drop_R).sum() > 5:
        return horseshoe_logistic(drop_Z, drop_R, n_iter=400, n_burn=150)
    rate = np.clip(drop_R.mean() if len(drop_R) else 0.5, 0.01, 0.99)
    return np.array([np.log(rate / (1 - rate)), 0.0])


def panel_baipw_dml(Y, D, X, R, K=5, rng=None, true_gamma=None):
    """B-AIPW-DML with horseshoe dropout correction, cross fitted by unit.

    1. Assign units to K folds. For every fold, fit the horseshoe logistic hazard on the units of the other folds
       and compute the inverse survival weights of the units in the fold (true_gamma given: use the true hazard, no fit).
    2. Weighted within demeaning and weighted, unit clustered, cross fitted partialling out with horseshoe nuisances
       on the same folds.
    """
    n_units = Y.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    fold_of_unit = unit_folds(n_units, K, rng)
    weights = np.ones(Y.shape)
    if true_gamma is not None:
        weights = survival_weights(Y, true_gamma)
    else:
        for k in range(K):
            val = np.where(fold_of_unit == k)[0]
            if len(val) == 0:
                continue
            b = fit_hazard(Y, R, np.where(fold_of_unit != k)[0])
            weights[val] = survival_weights(Y[val], b)
    return panel_dml_fe(Y, D, X, R, panel_hs_outcome, panel_hs_propensity, K=K,
                        weights=weights, fold_of_unit=fold_of_unit)


# ======================================================================
# Main simulation
# ======================================================================
def run_panel_sim(n_units, T, p, k_e, k_g, delta, theta_true, dropout_gamma, rng):
    """Run one panel replication for a given dropout level."""
    Y, D, X, alpha, R = generate_panel(
        n_units, T, p, k_e, k_g, delta, theta_true, dropout_gamma, rng
    )
    obs_rate = R.mean()
    units_used = float((R.sum(axis=1) >= 2).mean())

    results = {}
    results["WITHIN_OLS"] = panel_within_ols(Y, D, X, R)

    def fr():
        return np.random.default_rng(rng.integers(0, 2**31))

    results["DML_LS_FE"] = panel_dml_fe(Y, D, X, R, panel_ls, panel_ls, rng=fr())
    results["LASSO_DML_FE"] = panel_dml_fe(Y, D, X, R, panel_lasso_outcome, panel_lasso_propensity, rng=fr())
    results["B_DML_HS_FE"] = panel_dml_fe(Y, D, X, R, panel_hs_outcome, panel_hs_propensity, rng=fr())

    if dropout_gamma[1] != 0:
        results["B_AIPW_DML"] = panel_baipw_dml(Y, D, X, R, rng=fr())
        results["B_AIPW_ORACLE"] = panel_baipw_dml(Y, D, X, R, rng=fr(), true_gamma=dropout_gamma)

    return results, obs_rate, units_used


CONDITIONS = [(n, label, g) for n in (200, 1000)
              for label, g in (("no_dropout", (0.0, 0.0)), ("mar_mild", (-2.0, 0.5)), ("mar_strong", (-2.0, 1.0)))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--nrep", type=int, default=1)
    args = parser.parse_args()

    T = 5
    p = 50
    k_e = 10
    k_g = 5
    delta = 1.0
    theta_true = 1.0

    os.makedirs("results", exist_ok=True)
    for rep in range((args.rep - 1) * args.nrep + 1, args.rep * args.nrep + 1):
        out = f"results/panel_rep{rep}.csv"
        if os.path.exists(out):
            continue
        seed = rep * 37 + 13
        all_results = []
        t0 = time.time()
        for ci, (n_units, label, dropout_gamma) in enumerate(CONDITIONS):
            np.random.seed(seed * 10 + ci)
            rng = np.random.default_rng([seed, ci])
            t1 = time.time()
            res, obs_rate, units_used = run_panel_sim(
                n_units, T, p, k_e, k_g, delta, theta_true, dropout_gamma, rng
            )
            elapsed = time.time() - t1
            for method, (theta, se) in res.items():
                bias = theta - theta_true
                covered = np.abs(bias) < 1.96 * se if not (np.isnan(se) or np.isnan(theta)) else np.nan
                all_results.append({
                    "rep": rep, "n_units": n_units, "dropout": label, "obs_rate": obs_rate,
                    "units_used": units_used, "method": method, "theta": theta, "se": se,
                    "bias": bias, "covered": covered,
                })
            print(
                f"rep {rep} N={n_units} {label} (obs={obs_rate:.2f}) | "
                + " | ".join(f"{m}={res[m][0]:.4f}" for m in res)
                + f" | {elapsed:.1f}s", flush=True
            )
        pd.DataFrame(all_results).to_csv(out, index=False)
        print(f"Rep {rep} done in {time.time() - t0:.1f}s", flush=True)
