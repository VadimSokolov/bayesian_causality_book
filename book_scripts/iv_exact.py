"""Posterior computations for the structural instrumental variable model of Chapter 12 (Sections iv:sec:weak, iv:sec:many and iv:sec:card).

The model is the one of Rossi, Allenby and McCulloch (2005), with controls W (including the intercept) and instruments Z:

    x = Z pi + W gamma_x + e1,   y = tau x + W gamma_y + e2,   (e1, e2) ~ N(0, Sigma),

with independent N(0, 10^4) priors on tau and on every coefficient (so almost flat) and Sigma ~ inverse Wishart(3, S0), S0 = 0.1 diag(var x, var y).
Two ways of computing the posterior of tau are in this file.

exact_tau_cdf: Sigma, gamma_x, gamma_y and pi are integrated out analytically, which leaves a one dimensional density of tau that is evaluated on a
grid. With flat priors on pi, gamma_x and gamma_y the integral over Sigma gives |S0 + E'E|^{-(n + 3)/2}, the integral over the control coefficients
replaces n by n - q (q the number of columns of W) and E by its residual after W, and |S0 + E'E| is a quadratic function of pi, whose integral is
a multivariate t integral in closed form. The result is exact up to the (negligible) difference between the N(0, 10^4) prior and a flat prior on
the coefficients other than tau. The density is

    p(tau | data) proportional to exp(-tau^2 / 2e4) delta(tau)^{-(m - k)/2} |Q(tau)|^{-1/2},   m = n + 3 - q,   k = number of instruments,

with c = s2 + |e2|^2, e2 = ytilde - tau xtilde, a = Ztilde'e2, Q = c Ztilde'Ztilde - a a', h = c Ztilde'xtilde - a (xtilde'e2),
delta = s1 c + xtilde'xtilde c - (xtilde'e2)^2 - h'Q^{-1}h (tilde: after partialling out W).

biv_book: the Gibbs sampler of iv_card.py as it was (one coordinate block at a time). biv_block: the same posterior sampled with the coefficient of
the structural equation and the regression of e2 on e1 drawn together, which removes the ridge on which the one at a time sampler moves in steps of
about 0.02 (see iv_chain_check.py for the comparison).
"""
import numpy as np


def partial_out(W, *vs):
    coef = np.linalg.lstsq(W, np.column_stack(vs), rcond=None)[0]
    res = np.column_stack(vs) - W @ coef
    return [res[:, j] for j in range(res.shape[1])]


def exact_tau_cdf(y, x, Z, W, nu0=3, prec_tau=1e-4, umax=4.0, G=16001, scale=10.0, center=0.0, s0_mult=0.1):
    """Grid, density and distribution function of tau under the structural prior with flat priors on the first stage (see the module docstring)."""
    n, k, q = len(y), Z.shape[1], W.shape[1]
    s1, s2 = s0_mult * np.var(x), s0_mult * np.var(y)
    cols = partial_out(W, x, y, *[Z[:, j] for j in range(k)])
    xt, yt, Zt = cols[0], cols[1], np.column_stack(cols[2:])
    Sxx, Sxy, Syy = xt @ xt, xt @ yt, yt @ yt
    Zx, Zy, ZZ = Zt.T @ xt, Zt.T @ yt, Zt.T @ Zt
    m = n + nu0 - q
    u = np.linspace(-umax, umax, G)
    tau = center + scale * np.sinh(u)
    c = s2 + Syy - 2 * tau * Sxy + tau ** 2 * Sxx
    a = Zy[None, :] - tau[:, None] * Zx[None, :]
    xe = Sxy - tau * Sxx
    Q = c[:, None, None] * ZZ[None] - a[:, :, None] * a[:, None, :]
    h = c[:, None] * Zx[None, :] - a * xe[:, None]
    sol = np.linalg.solve(Q, h[:, :, None])[:, :, 0]
    delta = s1 * c + c * Sxx - xe ** 2 - np.einsum('gk,gk->g', h, sol)
    logdetQ = np.linalg.slogdet(Q)[1]
    logp = -(m - k) / 2 * np.log(delta) - 0.5 * logdetQ - 0.5 * prec_tau * tau ** 2
    p = np.exp(logp - logp.max())
    assert max(p[0], p[-1]) < 1e-6, 'the grid does not reach the tails of the posterior'
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(tau))])
    cdf /= cdf[-1]
    dens = p / (getattr(np, 'trapezoid', None) or np.trapz)(p, tau)
    return tau, dens, cdf


def cdf_quantiles(tau, cdf, probs):
    return np.interp(probs, cdf, tau)


def cdf_prob_greater(tau, cdf, t0):
    return float(1.0 - np.interp(t0, tau, cdf))


def biv_book(y, x, Zm, Wm, iters=6000, burn=1000, seed=1, tau0=None):
    """The sampler of iv_card.py, unchanged except for the optional starting value tau0 of the structural coefficient."""
    r = np.random.default_rng(seed)
    A = np.column_stack([Zm, Wm]); B = np.column_stack([x, Wm])
    kA, kB = A.shape[1], B.shape[1]; n = len(y)
    pd = np.linalg.lstsq(A, x, rcond=None)[0]; pb = np.linalg.lstsq(B, y, rcond=None)[0]
    if tau0 is not None:
        pb = pb.copy(); pb[0] = tau0
    Sig = np.eye(2) * np.var(x); Sig[1, 1] = np.var(y); nu0 = 3; S0 = np.diag([np.var(x), np.var(y)]) * 0.1
    prec0 = 1e-4
    out = []
    for it in range(iters):
        e1 = x - A @ pd
        c = Sig[0, 1] / Sig[0, 0]; v = Sig[1, 1] - Sig[0, 1] ** 2 / Sig[0, 0]
        yt = y - c * e1
        P = B.T @ B / v + prec0 * np.eye(kB); m = np.linalg.solve(P, B.T @ yt / v)
        L = np.linalg.cholesky(np.linalg.inv(P)); pb = m + L @ r.normal(size=kB)
        e2 = y - B @ pb
        c2 = Sig[0, 1] / Sig[1, 1]; v2 = Sig[0, 0] - Sig[0, 1] ** 2 / Sig[1, 1]
        xt = x - c2 * e2
        P = A.T @ A / v2 + prec0 * np.eye(kA); m = np.linalg.solve(P, A.T @ xt / v2)
        L = np.linalg.cholesky(np.linalg.inv(P)); pd = m + L @ r.normal(size=kA)
        E = np.column_stack([x - A @ pd, y - B @ pb])
        Sn = S0 + E.T @ E
        Wd = np.linalg.inv(Sn); G = np.linalg.cholesky(Wd)
        Xs = (G @ r.normal(size=(2, nu0 + n))).T; Sig = np.linalg.inv(Xs.T @ Xs)
        if it >= burn:
            out.append((pb[0], pd[0], Sig[0, 1] / np.sqrt(Sig[0, 0] * Sig[1, 1])))
    return np.array(out)


def biv_block(y, x, Zm, Wm, iters=3000, burn=500, seed=1, hier=False, tau0=None, nu0=3, s0_mult=0.1, horseshoe=False):
    """Blocked Gibbs sampler for the same posterior.

    Sigma is written through sigma11 = var(e1), c = cov(e1, e2)/sigma11 and v = var(e2 | e1). The inverse Wishart(nu0, S0) prior on Sigma is then
    sigma11 ~ IG((nu0 - 1)/2, S11/2), v ~ IG(nu0/2, S22/2), c | v ~ N(0, v/S11). The blocks are (tau, gamma_y, c), then v, then sigma11, then
    (pi, gamma_x) (the regression of x on D = [Z, W] with the information from both equations), and with hier=True the first stage prior pi ~
    N(0, lam2 I) with lam2 ~ IG(1, 1e-4) has its own step, as in the sampler of iv_weak2.py, and with horseshoe=True the first stage coefficients have the
    horseshoe prior pi_j ~ N(0, lambda_j^2 tau_g^2) with half Cauchy lambda_j and tau_g (the scale mixture form of Makalic and Schmidt 2016, and no
    scaling by the error variance). Returns the draws of (tau, pi_1, error correlation).
    """
    r = np.random.default_rng(seed)
    n, k = len(y), Zm.shape[1]
    D = np.column_stack([Zm, Wm]); kD = D.shape[1]
    S11, S22 = s0_mult * np.var(x), s0_mult * np.var(y)
    prec0 = 1e-4
    theta = np.linalg.lstsq(D, x, rcond=None)[0]                 # (pi, gamma_x)
    tau = np.linalg.lstsq(np.column_stack([x, Wm]), y, rcond=None)[0][0] if tau0 is None else tau0
    gy = np.linalg.lstsq(np.column_stack([x, Wm]), y, rcond=None)[0][1:]
    s11, v, cc = np.var(x), np.var(y), 0.0
    lam2 = np.var(x) / max(k, 1)
    lam_l = np.ones(k); nu_l = np.ones(k); tg2 = 1.0; xi_g = 1.0       # horseshoe scales of the first stage coefficients
    DtD = D.T @ D
    out = []
    for it in range(iters):
        e1 = x - D @ theta
        # block 1: (tau, gamma_y, c) | rest, a regression of y on [x, W, e1] with noise variance v
        R = np.column_stack([x, Wm, e1])
        prior_prec = np.concatenate([np.full(1 + Wm.shape[1], prec0), [S11 / v]])
        P = R.T @ R / v + np.diag(prior_prec)
        mu = np.linalg.solve(P, R.T @ y / v)
        b = mu + np.linalg.solve(np.linalg.cholesky(P).T, r.normal(size=P.shape[0]))
        tau, gy, cc = b[0], b[1:-1], b[-1]
        # block 2: v | rest
        eta = y - R[:, :-1] @ b[:-1] - cc * e1
        v = 1.0 / r.gamma((nu0 + 1 + n) / 2, 1.0 / ((S22 + S11 * cc ** 2 + eta @ eta) / 2))
        # block 3: sigma11 | rest
        s11 = 1.0 / r.gamma((nu0 - 1 + n) / 2, 1.0 / ((S11 + e1 @ e1) / 2))
        # block 4: (pi, gamma_x) | rest: x = D theta + e1 (variance s11) and y - (tau + c) x - W gamma_y = -c D theta + eta (variance v)
        prec_th = np.full(kD, prec0)
        if hier:
            prec_th[:k] = 1.0 / lam2
        if horseshoe:
            prec_th[:k] = 1.0 / (lam_l * tg2)
        P = DtD * (1.0 / s11 + cc ** 2 / v) + np.diag(prec_th)
        resid_y = y - (tau + cc) * x - Wm @ gy
        rhs = D.T @ x / s11 + (-cc) * (D.T @ resid_y) / v
        mu = np.linalg.solve(P, rhs)
        theta = mu + np.linalg.solve(np.linalg.cholesky(P).T, r.normal(size=kD))
        if hier:
            lam2 = 1.0 / r.gamma(1 + k / 2, 1.0 / (1e-4 + 0.5 * theta[:k] @ theta[:k]))
        if horseshoe:
            pk = theta[:k]
            lam_l = 1.0 / r.gamma(1.0, 1.0 / (1.0 / nu_l + pk ** 2 / (2.0 * tg2)))
            nu_l = 1.0 / r.gamma(1.0, 1.0 / (1.0 + 1.0 / lam_l))
            tg2 = 1.0 / r.gamma((k + 1) / 2.0, 1.0 / (1.0 / xi_g + np.sum(pk ** 2 / lam_l) / 2.0))
            xi_g = 1.0 / r.gamma(1.0, 1.0 / (1.0 + 1.0 / tg2))
        if it >= burn:
            s12 = cc * s11
            s22 = v + cc ** 2 * s11
            out.append((tau, theta[0], s12 / np.sqrt(s11 * s22)))
    return np.array(out)


def autocov(v):
    n = len(v)
    v = v - v.mean()
    f = np.fft.rfft(v, 2 * n)
    return np.fft.irfft(f * np.conj(f))[:n] / n


def split_rhat_ess(chains):
    """chains: (m, n). Each chain is cut in half; returns the classic split R-hat and Geyer's truncated effective sample size."""
    m, n = chains.shape
    h = n // 2
    x = np.concatenate([chains[:, :h], chains[:, h:2 * h]], axis=0)
    M, N = x.shape
    W = x.var(axis=1, ddof=1).mean()
    B = N * x.mean(axis=1).var(ddof=1)
    var_plus = (N - 1) / N * W + B / N
    rhat = float(np.sqrt(var_plus / W))
    mean_acov = np.mean([autocov(x[i]) * N / (N - 1) for i in range(M)], axis=0)
    rho = 1.0 - (W - mean_acov) / var_plus
    s = 0.0
    for t in range(1, N - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        s += pair
    tau_int = 1.0 + 2.0 * s
    ess = float(M * N / max(tau_int, 1.0 / np.log10(M * N)))
    return rhat, ess
