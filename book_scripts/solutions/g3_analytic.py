"""Deterministic checks for ex:ch07:8, ex:ch14:3, ex:chbart:4 and a Monte Carlo check of ex:chbaipw:2."""
import numpy as np
from scipy import integrate, optimize, special, stats

# ---------- ex:ch07:8, shrinkage on a log scale grid ----------
LS = np.linspace(-40, 25, 6501)  # log s, s = lambda g
DL = LS[1] - LS[0]
S = np.exp(LS)


def dens_logs(g, plus):
    """Density of log s, s = local scale times g."""
    u = S / g
    if plus:
        # density of u: 4 log u / (pi^2 (u^2 - 1)), limit 2/pi^2 at u = 1
        lu = np.log(u)
        with np.errstate(all="ignore"):
            d = np.where(np.abs(lu) < 1e-8, 2 / np.pi**2, 4 * lu / (np.pi**2 * (u**2 - 1)))
        # avoid overflow in u^2 for large u
        return d * u
    return 2 / (np.pi * (1 + u**2)) * u


def post_kappa(y, g, plus):
    y = np.atleast_1d(y)
    w = dens_logs(g, plus)
    v = 1 + S**2
    k = 1 / v
    like = np.exp(-0.5 * np.outer(y**2, k)) * np.sqrt(k)
    den = like @ w
    return (like @ (w * (1 - k))) / den  # E(1 - kappa | y)


def risk(g, plus, nsig, p, mu):
    gridy = np.linspace(-14, 14, 561)
    sh = post_kappa(gridy, g, plus)
    est = sh * gridy
    dy = gridy[1] - gridy[0]
    null = np.sum(est**2 * stats.norm.pdf(gridy)) * dy
    sig = np.sum((est - mu) ** 2 * stats.norm.pdf(gridy - mu)) * dy
    return (p - nsig) * null + nsig * sig, null, sig


print("mass check", np.sum(dens_logs(0.01, False)) * DL, np.sum(dens_logs(0.01, True)) * DL)

# prior mass at g = 1 and small g
for g in [1.0, 0.0075, 0.0013]:
    for plus in [False, True]:
        d = dens_logs(g, plus)
        k = 1 / (1 + S**2)
        print(f"g={g} plus={plus} P(k>0.99)={np.sum(d[k>0.99])*DL:.4f} P(0.1<k<0.9)={np.sum(d[(k>0.1)&(k<0.9)])*DL:.4f} P(k<0.5)={np.sum(d[k<0.5])*DL:.5f}")
for g in [0.0075, 0.0013]:
    print("closed P(k<.5) HS", 2 / np.pi * np.arctan(g), "HS+ approx", 4 / np.pi**2 * (np.log(1 / g) + 1) * g)

# crossover of R(r) = log r (r+1)/((r-1) pi)
x = optimize.brentq(lambda x: x / np.tanh(x / 2) - np.pi, 0.1, 10)
print("R=1 at log r=", x, "r=", np.exp(x), " min R", 2 / np.pi)

# density formula check, g general
for g, kk in [(0.1, 0.3), (0.1, 0.97), (0.01, 0.999)]:
    r = (1 - kk) / (g**2 * kk)
    hs = g / np.pi * kk**-0.5 * (1 - kk) ** -0.5 / (1 - kk + g**2 * kk)
    pl = g / np.pi**2 * kk**-0.5 * (1 - kk) ** -0.5 * np.log(r) / (1 - kk - g**2 * kk)
    # numeric density by differencing the cdf on the grid
    k = 1 / (1 + S**2)
    h = 1e-4 * min(kk, 1 - kk)
    for plus, f in [(False, hs), (True, pl)]:
        d = dens_logs(g, plus)
        s1 = np.sqrt((1 - kk - h) / (kk + h))
        s2 = np.sqrt((1 - kk + h) / (kk - h))
        val, _ = integrate.quad(lambda ls: np.interp(ls, LS, d), np.log(s1), np.log(s2))
        print(f"density check g={g} k={kk} plus={plus} formula={f:.5f} numeric={val/(2*h):.5f}")

# y(y-E) at g = 1
ys = np.array([8.0, 10, 20, 50])
print("y(y-Eb) g=1 HS", ys * ys * (1 - post_kappa(ys, 1.0, False)))
print("y(y-Eb) g=1 HS+", ys * ys * (1 - post_kappa(ys, 1.0, True)))
print("2-1/log y", 2 - 1 / np.log(ys))

# oracle g
for nsig, p, mu in [(2, 2000, 5.0), (2, 2000, 3.0), (1, 5000, 5.0), (20, 200, 5.0)]:
    out = []
    for plus in [False, True]:
        res = optimize.minimize_scalar(lambda lg: risk(np.exp(lg), plus, nsig, p, mu)[0], bounds=(-14, 0), method="bounded", options=dict(xatol=1e-2))
        g = np.exp(res.x)
        out.append((g, *risk(g, plus, nsig, p, mu)))
    print(f"nsig={nsig} p={p} mu={mu}: HS g={out[0][0]:.5f} tot={out[0][1]:.2f} null={out[0][2]:.6f} sig={out[0][3]:.3f}; HS+ g={out[1][0]:.5f} tot={out[1][1]:.2f} null={out[1][2]:.6f} sig={out[1][3]:.3f}")

# posterior shrinkage table at scales
ys = np.array([0.0, 1, 2, 3, 3.5, 4, 5, 6, 8])
for g, pl in [(0.0075, False), (0.0013, True), (0.0075, True)]:
    print(f"g={g} plus={pl} E(1-k|y):", np.round(post_kappa(ys, g, pl), 4))
print("approx null HS 2g/pi", 2 * 0.0075 / np.pi, "HS+ (4g/pi^2)(log(1/g)-1)", 4 * 0.0013 / np.pi**2 * (np.log(1 / 0.0013) - 1), 4 * 0.0075 / np.pi**2 * (np.log(1 / 0.0075) - 1))

# RIC bias contrast for two confounders: mean shrinkage error at signal mu when y ~ N(mu,1)
gridy = np.linspace(-6, 20, 521)
dy = gridy[1] - gridy[0]
for mu in [2.0, 3.0, 4.0, 5.0, 8.0]:
    r = []
    for g, pl in [(0.0075, False), (0.0013, True)]:
        est = post_kappa(gridy, g, pl) * gridy
        r.append(mu - np.sum(est * stats.norm.pdf(gridy - mu)) * dy)
    print(f"mean shrink error at mu={mu}: HS {r[0]:.3f} HS+ {r[1]:.3f}")

# ---------- ex:ch14:3 ----------
print("\nLaplace constant, Gamma form, k=a-1")
for k in [25, 100, 400, 1600]:
    a = k + 1
    logZ = special.gammaln(a)
    logL = k * np.log(k) - k + 0.5 * np.log(2 * np.pi * k)  # b=1
    print(k, "rel err const", np.exp(logL - logZ) - 1, "pred -1/(12k)", -1 / (12 * k))
    for z in [1.0, 2.0, -2.0]:
        th = k * (1 + z / np.sqrt(k))
        lp = stats.gamma(a).logpdf(th)
        lq = stats.norm(k, k / np.sqrt(k)).logpdf(th)
        print(f"   z={z} p/q-1={np.exp(lp-lq)-1:.5f}  z3/(3 sqrt k)={z**3/(3*np.sqrt(k)):.5f}")

# non gamma check of the correction formula: logistic-type integrand, g = n(theta - e^theta) + log pi with pi N(0,1)
for n in [10, 40, 160, 640]:
    logpi = lambda t: stats.norm.logpdf(t)
    g = lambda t: n * (t - np.exp(t)) + logpi(t)
    res = optimize.minimize_scalar(lambda t: -g(t), bounds=(-3, 3), method="bounded", options=dict(xatol=1e-11))
    m = res.x
    d2 = -n * np.exp(m) - 1
    d3 = -n * np.exp(m)
    d4 = -n * np.exp(m)
    gm = g(m)
    val, _ = integrate.quad(lambda t: np.exp(g(t) - gm), m - 2, m + 2, epsabs=1e-13, epsrel=1e-13, limit=500)
    lap = np.sqrt(2 * np.pi / -d2)
    corr = d4 / (8 * d2**2) + 5 * d3**2 / (24 * (-d2) ** 3)
    print(f"n={n} rel err (exact/Lap - 1)={val/lap-1:.3e} predicted corr={corr:.3e} ratio n*err={n*(val/lap-1):.4f}")

# ---------- ex:chbart:4, exact ridge identity ----------
rng = np.random.default_rng(20261101)
n, q = 300, 25
B = (rng.normal(size=(n, q)) > 0).astype(float)
th0 = rng.normal(size=q)
mu = B @ th0
T = (rng.uniform(size=n) < stats.norm.cdf(0.5 * (mu - mu.mean()))).astype(float)
eps = rng.normal(size=n)
tau, P, W = 1.0, 3.0, 2.0  # P = sigma2/v, W = sigma2/w
Y = tau * T + mu + eps
X = np.column_stack([T, B])
A = X.T @ X + np.diag([W] + [P] * q)
tauhat = np.linalg.solve(A, X.T @ Y)[0]
H = B @ np.linalg.solve(B.T @ B + P * np.eye(q), B.T)
M = np.eye(n) - H
den = T @ M @ T + W
print("\nridge identity:", tauhat - tau, (T @ M @ mu + T @ M @ eps - W * tau) / den, "RIC part", T @ M @ mu / den)

# ---------- ex:chbaipw:2, Monte Carlo check of the exact remainder ----------
rng = np.random.default_rng(20261102)
N = 3_000_000
for case in ["clean", "history in residuals"]:
    Hv = rng.normal(size=N)
    Xv = rng.normal(size=N)
    tau0 = 1.3
    m0 = np.tanh(Xv)
    if case == "clean":
        Tv = m0 + rng.normal(size=N)
        epsv = (1 + 0.5 * np.abs(Hv)) * rng.normal(size=N)
    else:
        # E(V0 | X, H) and E(eps | X, H) nonzero but E(V0 | X) = E(eps | X, T) = 0 needs H independent of X:
        # V0 = 0.4 H + noise, eps = 0.5 H' with H' the part of H orthogonal to V0
        nu = rng.normal(size=N)
        V = 0.4 * Hv + nu
        Tv = m0 + V
        hp = Hv - 0.4 * V / (0.16 + 1)  # uncorrelated with V, jointly normal so independent
        epsv = 0.5 * hp + rng.normal(size=N)
    l0 = tau0 * m0 + np.sin(Xv)  # l0 = E(Y | X), g = sin
    Yv = l0 + tau0 * (Tv - m0) + epsv
    pi0 = 0.2 + 0.7 * special.expit(Hv)
    R = rng.uniform(size=N) < pi0
    dm = 0.15 * np.cos(Xv)
    dl = 0.2 * Xv / (1 + Xv**2) + 0.1
    pih = np.clip(pi0 + 0.05 * np.sin(2 * Hv), 0.05, 0.99)
    muh = 0.3 * np.tanh(Hv)
    Vh = Tv - m0 - dm
    Yh = Yv - l0 - dl
    Sh = Vh * (Yh - tau0 * Vh)
    S0 = (Tv - m0) * (Yv - l0 - tau0 * (Tv - m0))
    psi = R / pih * Sh - (R / pih - 1) * muh
    # Sbar(H) = E(Sh | H), estimated by exact conditional integration over bins of H
    edges = np.quantile(Hv, np.linspace(0, 1, 2001))
    idx = np.clip(np.searchsorted(edges, Hv) - 1, 0, 1999)
    cnt = np.bincount(idx, minlength=2000)
    Sbar = (np.bincount(idx, Sh, minlength=2000) / cnt)[idx]
    A0 = (np.bincount(idx, Sh - dm * dl + tau0 * dm**2, minlength=2000) / cnt)[idx]
    lhs = psi.mean() - S0.mean()
    t1 = (dm * dl).mean()
    t2 = -tau0 * (dm**2).mean()
    t3 = ((pi0 - pih) / pih * (Sbar - muh)).mean()
    se = (psi - S0).std() / np.sqrt(N)
    print(f"{case}: lhs={lhs:.5f} (se {se:.5f}) rhs={t1+t2+t3:.5f} dm*dl={t1:.5f} -tau*dm^2={t2:.5f} pi*(Sbar-mu)={t3:.5f}; E[(pi0-pih)/pih*(Sbar-mu0-second order)] first-order part={((pi0-pih)/pih*A0).mean():.5f}, mean S0={S0.mean():.5f}")
