"""Shared pieces of critique_refit_bootstrap.py and critique_learners.py (Chapter 7, the review loop of 23 October).

The generating law is the one of the source paper's simulator (dmlsim.make_design), copied so that the book repository does not depend on it:
n x p standard normal X, treatment D = X alpha + nu with nu ~ N(0, sigma_nu^2), outcome Y = tau D + X gamma + eps with eps ~ N(0, 0.948^2), tau = 1,
alpha_j = 1 and gamma_j ~ U(0.8, 1.2) on the first s coordinates (the sparse designs), or, in the dense design, alpha_j = sqrt(15/p) and
gamma_j = alpha_j u_j for all p coordinates. The replication seed is zlib.crc32('<design>|<r>') mod 2^31, as in the source simulator, so the data
sets are those of the 500 replication runs in the paper (replications 0 to 499) and extend them.
"""
import math, zlib
import numpy as np

THETA0, SIGMA_EPS = 1.0, 0.948
DESIGNS = {'sparse_s5': dict(s=5, sigma_nu=1.0), 'sparse_s15': dict(s=15, sigma_nu=1.0), 'sparse_s30': dict(s=30, sigma_nu=1.0),
           'weak_overlap': dict(s=15, sigma_nu=0.3), 'dense': dict(s=15, sigma_nu=1.0, dense=True)}


def rep_seed(design, r):
    return zlib.crc32(('%s|%d' % (design, r)).encode()) % (2 ** 31)


def make_design(rng, n=300, p=400, s=15, sigma_nu=1.0, dense=False, s_ref=15):
    X = rng.standard_normal((n, p)); alpha = np.zeros(p); gamma = np.zeros(p)
    if dense:
        sc = math.sqrt(s_ref / p); alpha[:] = sc; gamma[:] = sc * rng.uniform(0.8, 1.2, p)
    else:
        alpha[:s] = 1.0; gamma[:s] = rng.uniform(0.8, 1.2, s)
    nu = sigma_nu * rng.standard_normal(n); D = X @ alpha + nu
    eps = SIGMA_EPS * rng.standard_normal(n); Y = THETA0 * D + X @ gamma + eps
    return dict(X=X, D=D, Y=Y, alpha=alpha, gamma=gamma, varsigma=THETA0 * alpha + gamma, sigma_nu=sigma_nu)


def _ig(rng, shape, rate):
    g = rng.gamma(shape, 1.0 / np.maximum(rate, 1e-300))
    return np.clip(1.0 / np.maximum(g, 1e-300), 1e-12, 1e12)


def horseshoe_mean(X, y, rng, n_iter=500, burn=150):
    """Posterior mean of the coefficients of a horseshoe linear regression (Gibbs, Makalic and Schmidt scale mixtures; coefficients by Cholesky of the
    p x p precision when p <= n and by the Bhattacharya, Chakraborty and Mallick draw otherwise). y must be centred."""
    n, p = X.shape; lam2 = np.ones(p); om2 = 1.0; nu_l = np.ones(p); xi = 1.0; sig2 = 1.0; acc = np.zeros(p); cnt = 0
    G = X.T @ X if p <= n else None
    for it in range(n_iter):
        Dd = sig2 * om2 * lam2
        if p <= n:
            A = G / sig2 + np.diag(1.0 / Dd); L = np.linalg.cholesky(A)
            mu = np.linalg.solve(L.T, np.linalg.solve(L, X.T @ y / sig2)); beta = mu + np.linalg.solve(L.T, rng.standard_normal(p))
        else:
            Phi = X / math.sqrt(sig2); u = np.sqrt(Dd) * rng.standard_normal(p); v = Phi @ u + rng.standard_normal(n)
            M = (Phi * Dd) @ Phi.T; M[np.diag_indices_from(M)] += 1.0
            beta = u + (Phi * Dd).T @ np.linalg.solve(M, y / math.sqrt(sig2) - v)
        res = y - X @ beta
        sig2 = (res @ res + np.sum(beta ** 2 / (om2 * lam2))) / 2.0 * _ig(rng, (n + p) / 2.0, 1.0)
        lam2 = _ig(rng, 1.0, 1.0 / nu_l + beta ** 2 / (2.0 * sig2 * om2)); nu_l = _ig(rng, 1.0, 1.0 + 1.0 / lam2)
        om2 = _ig(rng, (p + 1) / 2.0, 1.0 / xi + np.sum(beta ** 2 / lam2) / (2.0 * sig2)); xi = _ig(rng, 1.0, 1.0 + 1.0 / om2)
        if it >= burn:
            acc += beta; cnt += 1
    return acc / cnt


def mc_se_mean(x):
    x = np.asarray(x, float); return float(x.std(ddof=1) / math.sqrt(len(x)))
