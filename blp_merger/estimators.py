"""
Estimators compared in the paper.

1. logit_iv     : plain logit with two stage least squares (Berry 1994 inversion).
2. blp_gmm      : Berry Levinsohn Pakes nested fixed point GMM, two step, sandwich SE.
3. jmr_mcmc     : Bayesian BLP with the inversion likelihood (Jiang, Manchanda and Rossi 2009),
                  a likelihood based benchmark in the spirit of Viard, Gron and Polson.
4. GBC          : generative Bayesian computation. Simulate (theta, conduct, data) from the
                  prior and the equilibrium model, learn theta = H(S) + U from summary
                  statistics S by a neural network, and invert by noise outsourcing.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import invwishart
from blp_core import contraction, ind_probs, alpha_i, markups, CONDUCTS

# ----------------------------------------------------------------------------- utilities

def flat(a):
    return a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a.reshape(-1)


def tsls(y, X, Z):
    PZ = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
    Xh = PZ @ X
    b = np.linalg.solve(Xh.T @ X, Xh.T @ y)
    return b, y - X @ b


# ----------------------------------------------------------------------------- 1. logit IV

def logit_iv(d, s, p):
    s0 = 1 - s.sum(1, keepdims=True)
    y = flat(np.log(s) - np.log(s0))
    X = np.column_stack([flat(d.X), -flat(p)])
    b, e = tsls(y, X, flat(d.Z))
    return dict(beta=b[:3], alpha=b[3], resid=e)


def logit_markups(d, s, p, alpha, H):
    """Logit is the BLP model with sigma_p = sigma_x = 0."""
    s0 = 1 - s.sum(1, keepdims=True)
    delta = np.log(s) - np.log(s0) + alpha * p
    sij = ind_probs(delta, p, d, alpha, 0.0, 0.0)
    return markups(s, sij, alpha_i(alpha, 0.0, d.nu), H)


# ----------------------------------------------------------------------------- 2. BLP GMM

class BLPGMM:
    def __init__(self, d, s, p):
        self.d, self.s, self.p = d, s, p
        self.Z = flat(d.Z)
        self.X = flat(d.X)
        self.W = np.linalg.inv(self.Z.T @ self.Z / len(self.Z))
        self.dl = None

    def xi(self, th2):
        a, sp, sx = np.exp(th2[0]), abs(th2[1]), abs(th2[2])
        self.dl = contraction(self.s, self.p, self.d, a, sp, sx, delta0=self.dl, tol=1e-12)
        dv = flat(self.dl)
        ZX = self.Z.T @ self.X
        b = np.linalg.solve(ZX.T @ self.W @ ZX, ZX.T @ self.W @ (self.Z.T @ dv))
        return dv - self.X @ b, b

    def obj(self, th2):
        e, _ = self.xi(th2)
        g = self.Z.T @ e / len(e)
        return len(e) * g @ self.W @ g

    def fit(self, start=(0.0, 0.3, 0.5)):
        best = None
        for st in [start, (np.log(1.5), 0.6, 1.0), (np.log(0.7), 0.1, 0.2)]:
            self.dl = None
            r = minimize(self.obj, np.array(st), method="Nelder-Mead",
                         options=dict(xatol=1e-7, fatol=1e-10, maxiter=3000))
            if best is None or r.fun < best.fun:
                best = r
        e, _ = self.xi(best.x)
        G = self.Z * e[:, None]
        self.W = np.linalg.inv(G.T @ G / len(e))
        self.dl = None
        r = minimize(self.obj, best.x, method="Nelder-Mead",
                     options=dict(xatol=1e-8, fatol=1e-11, maxiter=3000))
        self.th2 = np.array([np.exp(r.x[0]), abs(r.x[1]), abs(r.x[2])])
        e, b = self.xi(r.x)
        self.beta, self.resid, self.J_stat = b, e, r.fun
        self.cov = self.sandwich(r.x, b)
        return self

    def sandwich(self, x, b):
        """GMM sandwich covariance of (alpha, sigma_p, sigma_x) by finite differences."""
        n = len(self.Z)
        base = np.array([np.exp(x[0]), abs(x[1]), abs(x[2])])
        def xi_nat(th):
            dl = contraction(self.s, self.p, self.d, th[0], th[1], th[2], delta0=self.dl)
            return flat(dl)
        d0 = xi_nat(base)
        J = np.zeros((n, 3))
        for k in range(3):
            h = 1e-5
            th = base.copy(); th[k] += h
            J[:, k] = (xi_nat(th) - d0) / h
        Jfull = np.column_stack([J, -self.X])
        G = self.Z.T @ Jfull / n
        e = d0 - self.X @ b
        S = (self.Z * e[:, None]).T @ (self.Z * e[:, None]) / n
        W = self.W
        A = np.linalg.inv(G.T @ W @ G)
        V = A @ G.T @ W @ S @ W @ G @ A / n
        return V[:3, :3]


# ----------------------------------------------------------------------------- 3. JMR MCMC

def jmr_mcmc(d, s, p, n_iter=4000, burn=1000, seed=1, step=(0.04, 0.06, 0.06)):
    """Bayesian BLP with the inversion likelihood and a reduced form price equation.

    delta(theta2) = X beta + xi,   p = Z pi + v,   (xi, v) ~ N(0, Sigma),
    likelihood includes the Jacobian |d s / d delta|^{-1} of the share inversion.
    theta2 = (log alpha, sigma_p, sigma_x) by random walk Metropolis; beta, pi, Sigma by Gibbs.
    """
    rng = np.random.default_rng(seed)
    X, Z, P = flat(d.X), flat(d.Z), flat(p)
    n, kx, kz = len(X), X.shape[1], Z.shape[1]
    T, J = s.shape

    def loglik(th2, beta, pi, Sig, dl0):
        a, sp, sx = np.exp(th2[0]), th2[1], th2[2]
        if sp < 0 or sx < 0 or sp > 2.0 or sx > 3.0:
            return -np.inf, dl0
        dl = contraction(s, p, d, a, sp, sx, delta0=dl0, tol=1e-11)
        sij = ind_probs(dl, p, d, a, sp, sx)
        Ds = np.einsum("tji,tki->tjk", sij, sij) / sij.shape[-1]
        Ds = -Ds
        Ds[:, range(J), range(J)] += s
        _, ld = np.linalg.slogdet(Ds)
        xi = flat(dl) - X @ beta
        v = P - Z @ pi
        R = np.column_stack([xi, v])
        Si = np.linalg.inv(Sig)
        q = np.einsum("ni,ij,nj->", R, Si, R)
        return -0.5 * q - ld.sum(), dl

    # start from logit IV
    th2 = np.array([0.0, 0.2, 0.3])
    dl = contraction(s, p, d, 1.0, 0.2, 0.3)
    beta = np.linalg.lstsq(X, flat(dl), rcond=None)[0]
    pi = np.linalg.lstsq(Z, P, rcond=None)[0]
    Sig = np.eye(2) * 0.3
    ll, dl = loglik(th2, beta, pi, Sig, dl)
    out, acc = [], 0
    prior_logalpha = lambda la: -0.5 * (la / 2.0) ** 2
    for it in range(n_iter):
        # Metropolis for theta2
        prop = th2 + rng.normal(0, 1, 3) * np.array(step)
        llp, dlp = loglik(prop, beta, pi, Sig, dl)
        if np.log(rng.uniform()) < llp + prior_logalpha(prop[0]) - ll - prior_logalpha(th2[0]):
            th2, ll, dl = prop, llp, dlp
            acc += 1
        a, sp, sx = np.exp(th2[0]), th2[1], th2[2]
        dv = flat(dl)
        # Gibbs: beta | pi, Sigma
        v = P - Z @ pi
        c = Sig[0, 1] / Sig[1, 1]
        s2 = Sig[0, 0] - c * Sig[0, 1]
        y = dv - c * v
        Vb = np.linalg.inv(X.T @ X / s2 + np.eye(kx) / 100.0)
        beta = rng.multivariate_normal(Vb @ X.T @ y / s2, Vb)
        # Gibbs: pi | beta, Sigma
        xi = dv - X @ beta
        c2 = Sig[0, 1] / Sig[0, 0]
        s22 = Sig[1, 1] - c2 * Sig[0, 1]
        y2 = P - c2 * xi
        Vp = np.linalg.inv(Z.T @ Z / s22 + np.eye(kz) / 100.0)
        pi = rng.multivariate_normal(Vp @ Z.T @ y2 / s22, Vp)
        # Gibbs: Sigma
        R = np.column_stack([xi, P - Z @ pi])
        Sig = invwishart.rvs(df=n + 4, scale=R.T @ R + np.eye(2) * 0.1, random_state=rng)
        ll, _ = loglik(th2, beta, pi, Sig, dl)
        if it >= burn:
            out.append([a, sp, sx])
    return np.array(out), acc / n_iter


# ----------------------------------------------------------------------------- 4. GBC

PARAMS = ["b0", "b1", "b2", "alpha", "sp", "sx", "g0", "g1", "g2", "sxi", "somega", "rho"]
PRIOR = dict(b0=(0.5, 3.5), b1=(0.0, 2.0), b2=(-0.5, 1.5), alpha=(0.4, 2.0), sp=(0.0, 1.0),
             sx=(0.0, 1.5), g0=(1.0, 3.0), g1=(0.0, 1.0), g2=(0.0, 0.6), sxi=(0.2, 1.0),
             somega=(0.1, 0.6), rho=(-0.5, 0.5))


def draw_prior(rng):
    return {k: rng.uniform(*PRIOR[k]) for k in PARAMS}


def summaries(d, s, p, micro=None):
    """Summary statistics. The markup regression block is what identifies conduct:
    under logit, single product markups are 1/(alpha(1 - s_j)), multiproduct Bertrand
    markups are 1/(alpha(1 - s_f)) and joint monopoly markups are 1/(alpha s_0)."""
    T, J = s.shape
    s0 = 1 - s.sum(1, keepdims=True)
    y = flat(np.log(s) - np.log(s0))
    X = flat(d.X)
    P = flat(p)
    Z = flat(d.Z)
    out = []
    # logit IV and OLS
    b, e = tsls(y, np.column_stack([X, -P]), Z)
    out += list(b) + [e.std()]
    bo = np.linalg.lstsq(np.column_stack([X, -P]), y, rcond=None)[0]
    out += [bo[3]]
    # first stage
    fs = np.linalg.lstsq(Z[:, :7], P, rcond=None)[0]
    out += list(fs[1:])
    # markup regressions
    sf = np.einsum("tk,jk->tj", s, d.H_own)
    r1 = 1 / (1 - flat(s))
    r2 = 1 / (1 - flat(sf))
    r3 = 1 / flat(np.broadcast_to(s0, s.shape))
    M = np.column_stack([np.ones_like(P), flat(d.w), flat(d.x1), r1, r2 - r1, r3 - r2])
    bm = np.linalg.lstsq(M, P, rcond=None)[0]
    out += list(bm[3:]) + [(P - M @ bm).std()]
    # same regression with IV for share terms (cost shifters of rivals move rival shares)
    Mz = np.column_stack([np.ones_like(P), flat(d.w), flat(d.x1), flat(d.riv_x1), flat(d.own_x1),
                          flat(d.gh_w), flat(d.gh_x), flat(d.x2)])
    bmi, _ = tsls(P, M, Mz)
    out += list(np.clip(bmi[3:], -20, 20))
    # BLP style curvature: logit residual against differentiation instruments and price
    g = np.linalg.lstsq(np.column_stack([np.ones_like(P), flat(d.gh_x), flat(d.gh_w), P,
                                         P * flat(d.x1)]), e, rcond=None)[0]
    out += list(g[1:])
    out += [np.log(s0).mean(), np.log(s0).std(), P.mean(), P.std(), np.corrcoef(P, y)[0, 1]]
    out = np.array(out)
    if micro is not None:
        out = np.concatenate([out, micro])
    return out


class GBC:
    """Noise outsourcing posterior: theta = H(S) + U, with U approximately independent of S.

    H is a neural network fitted on simulated (theta, S); the posterior at S_obs is
    H(S_obs) + U_i for the simulations i whose fitted values lie closest to H(S_obs)
    (a regression adjusted, kernel localised version of noise outsourcing).
    Conduct is a discrete parameter with its own neural classifier.
    """

    def __init__(self, sims, use_micro=True, seed=0, n_folds=3):
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        from sklearn.preprocessing import StandardScaler
        self.use_micro = use_micro
        S = np.array([x["S"] for x in sims])
        if not use_micro:
            S = S[:, :-2]
        Th = np.array([[x["theta"][k] for k in PARAMS] for x in sims])
        C = np.array([CONDUCTS.index(x["conduct"]) for x in sims])
        self.sS = StandardScaler().fit(S)
        self.sT = StandardScaler().fit(Th)
        Ss = self.sS.transform(S)
        Ts = self.sT.transform(Th)
        self.clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=600, early_stopping=True,
                                 random_state=seed).fit(Ss, C)
        # cross fitting: the outsourced noise U must be computed out of sample,
        # otherwise in sample residuals are too small and intervals undercover
        self.reg, self.oof = {}, {}
        for c in range(3):
            idx = np.where(C == c)[0]
            folds = np.arange(len(idx)) % n_folds
            oof = np.zeros((len(idx), Ts.shape[1]))
            models = []
            for f in range(n_folds):
                tr, te = idx[folds != f], idx[folds == f]
                mdl = MLPRegressor(hidden_layer_sizes=(128, 128), alpha=1e-3, max_iter=1500,
                                   early_stopping=True, random_state=seed + f).fit(Ss[tr], Ts[tr])
                oof[folds == f] = mdl.predict(Ss[te])
                models.append(mdl)
            self.reg[c], self.oof[c] = models, oof
        self.S, self.Ss, self.Ts, self.C, self.Th = S, Ss, Ts, C, Th

    def conduct_probs(self, S_obs):
        S_obs = S_obs if self.use_micro else S_obs[:-2]
        return self.clf.predict_proba(self.sS.transform(S_obs[None]))[0]

    def posterior(self, S_obs, c, frac=0.15, cross_fit=True):
        S_obs = S_obs if self.use_micro else S_obs[:-2]
        so = self.sS.transform(S_obs[None])
        idx = np.where(self.C == c)[0]
        fit = self.oof[c] if cross_fit else np.mean([m.predict(self.Ss[idx]) for m in self.reg[c]], axis=0)
        f0 = np.mean([m.predict(so)[0] for m in self.reg[c]], axis=0)
        dist = np.sqrt(((fit - f0) ** 2).sum(1))
        k = max(50, int(frac * len(idx)))
        keep = np.argsort(dist)[:k]
        h = dist[keep].max()
        wts = 1 - (dist[keep] / h) ** 2
        U = self.Ts[idx][keep] - fit[keep]
        draws = self.sT.inverse_transform(f0[None] + U)
        return draws, wts / wts.sum()

    def sample(self, S_obs, n, rng, conduct=None):
        """Model averaged joint posterior draws of (theta, conduct)."""
        pc = self.conduct_probs(S_obs)
        out_t, out_c = [], []
        cs = rng.choice(3, size=n, p=pc) if conduct is None else np.full(n, CONDUCTS.index(conduct))
        for c in range(3):
            m = (cs == c).sum()
            if m == 0:
                continue
            dr, w = self.posterior(S_obs, c)
            pick = rng.choice(len(dr), size=m, p=w)
            out_t.append(dr[pick]); out_c += [c] * m
        return np.vstack(out_t), np.array(out_c), pc


class GBCTargets:
    """Cross fitted noise outsourcing for an arbitrary target vector Y (parameters and/or
    antitrust estimands): Y = H(S) + U, with U represented by out of fold residuals of the
    simulations whose fitted values are nearest to H(S_obs)."""

    def __init__(self, S, Y, n_folds=3, hidden=(256, 256, 128), seed=0):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        self.ss, self.sy = StandardScaler().fit(S), StandardScaler().fit(Y)
        Ss, Ys = self.ss.transform(S), self.sy.transform(Y)
        folds = np.arange(len(S)) % n_folds
        self.fit = np.zeros_like(Ys)
        self.nets = []
        for f in range(n_folds):
            net = MLPRegressor(hidden_layer_sizes=hidden, alpha=1e-3, max_iter=2500,
                               early_stopping=True, random_state=seed + f).fit(Ss[folds != f], Ys[folds != f])
            self.fit[folds == f] = net.predict(Ss[folds == f])
            self.nets.append(net)
        self.Ys = Ys

    def posterior(self, S_obs, k=1500):
        so = self.ss.transform(S_obs[None])
        f0 = np.mean([n.predict(so)[0] for n in self.nets], axis=0)
        dist = np.sqrt(((self.fit - f0) ** 2).sum(1))
        keep = np.argsort(dist)[:k]
        return self.sy.inverse_transform(f0[None] + self.Ys[keep] - self.fit[keep])
