"""
Core random coefficients logit (BLP) machinery.

Utility of consumer i for product j in market t
    u_ijt = delta_jt - alpha_i p_jt + sigma_x eta_i x1_jt + eps_ijt,
    delta_jt = X_jt beta + xi_jt,
    alpha_i = alpha * exp(-sigma_p nu_i),
where nu_i is a standardised log income draw (observed demographic, as in Petrin 2002)
and eta_i is an unobserved taste for x1. The price coefficient is lognormal and so
always negative, which guarantees an equilibrium exists.

Supply: mc_jt = gamma0 + gamma1 w_jt + gamma2 x1_jt + omega_jt.
Conduct is an ownership (internalisation) matrix H:
    "bertrand"  : H_jk = 1 if j and k share an owner
    "collusion" : H_jk = 1 for all j, k (joint profit maximisation)
    "perfect"   : markups are zero, p = mc
First order conditions:  s + (H * D') (p - mc) = 0,  D_jk = ds_j / dp_k,
so the markup vector is   m = -(H * D')^{-1} s.
"""
import numpy as np

CONDUCTS = ["bertrand", "collusion", "perfect"]


class Design:
    """Observed exogenous design of the industry: characteristics, cost shifters, owners."""

    def __init__(self, T=40, J=6, firms=(0, 0, 1, 1, 2, 2), ns=200, seed=0):
        rng = np.random.default_rng(seed)
        self.T, self.J, self.ns = T, J, ns
        self.firms = np.array(firms)
        self.x1 = rng.normal(0, 1, (T, J))
        self.x2 = rng.normal(0, 1, (T, J))
        self.w = rng.normal(0, 1, (T, J))
        self.X = np.stack([np.ones((T, J)), self.x1, self.x2], axis=-1)  # T,J,3
        self.W = np.stack([np.ones((T, J)), self.w, self.x1], axis=-1)  # T,J,3
        # simulation draws, common across markets (quasi random through fixed seed)
        dr = np.random.default_rng(seed + 1)
        self.nu = dr.normal(0, 1, ns)
        self.eta = dr.normal(0, 1, ns)
        self.H_own = (self.firms[:, None] == self.firms[None, :]).astype(float)
        self.instruments()

    def ownership(self, conduct, firms=None):
        f = self.firms if firms is None else np.asarray(firms)
        if conduct == "bertrand":
            return (f[:, None] == f[None, :]).astype(float)
        if conduct == "collusion":
            return np.ones((self.J, self.J))
        if conduct == "partial":
            # partial rival internalization (kappa = 0.5), used only in the misspecification stress test
            own = (f[:, None] == f[None, :]).astype(float)
            return own + 0.5 * (1.0 - own)
        return None

    def instruments(self):
        """BLP sums of rival and own firm characteristics and Gandhi Houde differentiation IVs."""
        T, J = self.T, self.J
        same = self.H_own[None, :, :] - np.eye(J)[None]
        rival = 1.0 - self.H_own[None, :, :]
        own_x1 = np.einsum("tk,tjk->tj", self.x1, np.broadcast_to(same, (T, J, J)))
        riv_x1 = np.einsum("tk,tjk->tj", self.x1, np.broadcast_to(rival, (T, J, J)))
        dx = (self.x1[:, :, None] - self.x1[:, None, :]) ** 2
        dw = (self.w[:, :, None] - self.w[:, None, :]) ** 2
        gh_x = dx.sum(-1)
        gh_w = (dw * rival).sum(-1)
        self.riv_x1, self.own_x1, self.gh_x, self.gh_w = riv_x1, own_x1, gh_x, gh_w
        self.Z = np.stack(
            [np.ones((T, J)), self.x1, self.x2, self.w, self.w ** 2, riv_x1, own_x1, gh_x, gh_w],
            axis=-1,
        )


def alpha_i(alpha, sp, nu):
    return alpha * np.exp(-sp * nu)


def ind_probs(delta, p, d, alpha, sp, sx):
    """Individual choice probabilities, shape (T, J, ns)."""
    a = alpha_i(alpha, sp, d.nu)
    V = delta[:, :, None] - a[None, None, :] * p[:, :, None] + sx * d.eta[None, None, :] * d.x1[:, :, None]
    m = np.maximum(V.max(axis=1), 0.0)
    e = np.exp(V - m[:, None, :])
    den = np.exp(-m) + e.sum(axis=1)
    return e / den[:, None, :]


def shares(delta, p, d, alpha, sp, sx):
    return ind_probs(delta, p, d, alpha, sp, sx).mean(-1)


def price_derivs(sij, a):
    """D[t,j,k] = d s_j / d p_k."""
    As = sij * a[None, None, :]
    Lam = As.mean(-1)
    Gam = np.einsum("tji,tki->tjk", As, sij) / sij.shape[-1]
    T, J = Lam.shape
    D = -Lam[:, :, None] * np.eye(J)[None] + Gam
    return D, Lam, Gam


def markups(s, sij, a, H):
    """m = -(H * D')^{-1} s for each market. H None means perfect competition."""
    if H is None:
        return np.zeros_like(s)
    D, _, _ = price_derivs(sij, a)
    A = H[None] * np.transpose(D, (0, 2, 1))
    return -np.linalg.solve(A, s[:, :, None])[:, :, 0]


def contraction(s_obs, p, d, alpha, sp, sx, delta0=None, tol=1e-12, maxit=3000):
    """BLP fixed point delta <- delta + log s_obs - log s(delta)."""
    ls = np.log(s_obs)
    if delta0 is None:
        s0 = 1 - s_obs.sum(1, keepdims=True)
        delta0 = ls - np.log(s0)
    delta = delta0.copy()
    for it in range(maxit):
        s = shares(delta, p, d, alpha, sp, sx)
        step = ls - np.log(s)
        delta += step
        if np.max(np.abs(step)) < tol:
            break
    return delta


def solve_equilibrium(delta, c, d, alpha, sp, sx, H, p0=None, tol=1e-10, maxit=2000):
    """Morrow and Skerlos zeta fixed point  p <- c + Lam^{-1}(s + (H*Gam)(p - c))."""
    if H is None:
        return c.copy(), True
    a = alpha_i(alpha, sp, d.nu)
    p = (c + 1.0 / alpha) if p0 is None else p0.copy()
    for it in range(maxit):
        sij = ind_probs(delta, p, d, alpha, sp, sx)
        s = sij.mean(-1)
        _, Lam, Gam = price_derivs(sij, a)
        m = (s + np.einsum("tjk,tk->tj", H[None] * Gam, p - c)) / Lam
        pn = c + m
        if not np.all(np.isfinite(pn)):
            return pn, False
        if np.max(np.abs(pn - p)) < tol:
            return pn, True
        p = pn
    return p, False


def consumer_surplus(delta, p, d, alpha, sp, sx):
    """Expected consumer surplus per market (in price units), averaged over consumers."""
    a = alpha_i(alpha, sp, d.nu)
    V = delta[:, :, None] - a[None, None, :] * p[:, :, None] + sx * d.eta[None, None, :] * d.x1[:, :, None]
    m = np.maximum(V.max(axis=1), 0.0)
    lse = m + np.log(np.exp(-m) + np.exp(V - m[:, None, :]).sum(1))
    return (lse / a[None, :]).mean(-1)


def simulate_market(d, theta, conduct, rng):
    """Draw unobservables and solve for equilibrium prices and shares.

    theta keys: b0,b1,b2,alpha,sp,sx,g0,g1,g2,sxi,somega,rho
    """
    T, J = d.T, d.J
    L = np.array([[theta["sxi"], 0.0],
                  [theta["rho"] * theta["somega"], np.sqrt(1 - theta["rho"] ** 2) * theta["somega"]]])
    e = rng.normal(size=(T, J, 2)) @ L.T
    xi, om = e[..., 0], e[..., 1]
    delta = theta["b0"] + theta["b1"] * d.x1 + theta["b2"] * d.x2 + xi
    c = theta["g0"] + theta["g1"] * d.w + theta["g2"] * d.x1 + om
    H = d.ownership(conduct)
    p, ok = solve_equilibrium(delta, c, d, theta["alpha"], theta["sp"], theta["sx"], H)
    sij = ind_probs(delta, p, d, theta["alpha"], theta["sp"], theta["sx"])
    s = sij.mean(-1)
    # Petrin style micro moments: income of inside good buyers, income times price paid
    inside = 1 - sij.sum(1)  # outside prob, T x ns
    buy = 1 - inside
    wts = buy / buy.sum()
    micro1 = (wts * d.nu[None, :]).sum()
    pp = (sij * p[:, :, None]).sum(1) / np.maximum(buy, 1e-12)
    micro2 = (wts * d.nu[None, :] * pp).sum()
    # rejection reason, recorded for the acceptance logs (it does not change which draws are accepted)
    if not ok:
        why = "solver"
    elif not (np.all(np.isfinite(p)) and np.all(p > 0) and np.all(p < 60)):
        why = "price bound"
    elif not np.all(s > 1e-7):
        why = "share floor"
    elif not np.all(s.sum(1) < 0.995):
        why = "inside share"
    else:
        why = None
    ok = ok and np.all(np.isfinite(p)) and np.all(p > 0) and np.all(p < 60) and np.all(s > 1e-7) \
        and np.all(s.sum(1) < 0.995)
    return dict(p=p, s=s, xi=xi, omega=om, delta=delta, c=c, micro=np.array([micro1, micro2]), ok=ok, why=why)


def tally(log, key, n=1):
    """Count an event in an acceptance log (a plain dict)."""
    if log is not None:
        log[key] = log.get(key, 0) + n


def merge_logs(logs):
    out = {}
    for lg in logs:
        for k, v in lg.items():
            out[k] = out.get(k, 0) + v
    return out
