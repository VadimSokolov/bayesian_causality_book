"""Round 12 checks of codex findings on fill-g1 and fill-g2 (no DML here; see g12_dml.py)."""
import numpy as np
from scipy import stats, integrate, optimize
from scipy.special import expit
# g1 F1: marginal likelihood of two equal differences at lambda=1 and 1/2, kappa=sigma=1, flat tau
s = 2.0
dens = lambda z, l: 2 * stats.norm.pdf(z, 0, s) * (l * stats.norm.cdf(z / s) + (1 - l) * stats.norm.cdf(-z / s))
ml = {l: integrate.quad(lambda z: dens(z, l) ** 2, -40, 40, epsabs=1e-12)[0] for l in (1., .5)}
print("g1F1 marginal lik", ml, "ratio", ml[1.] / ml[.5])
print("g1F3 shift lam=.9", .8 * 2 / np.sqrt(np.pi), "post sd", 2 / np.sqrt(50))
# g2 F1: horseshoe, truncated at 45 versus full tails
gx, gw = np.polynomial.hermite_e.hermegauss(40); gw = gw / gw.sum()
lb = np.linspace(-45, 45, 6001); dl = lb[1] - lb[0]; lam = np.exp(lb); plam = 2 / (np.pi * (1 + lam ** 2)) * lam
def prior(b, g):
    sc = lam * g
    return np.array([np.sum(np.exp(-.5 * (x / sc) ** 2) / (sc * np.sqrt(2 * np.pi)) * plam) * dl for x in b])
def grid(top):
    pos = np.exp(np.linspace(np.log(1e-5), np.log(top), 6000))
    return pos
def pm(g, a, M, S, sig, top):
    pos = grid(top); pr = prior(pos, g)
    beta = np.r_[-pos[::-1], pos]; p2 = np.r_[pr[::-1], pr]; wb = np.gradient(beta)
    if sig == 0:
        q = expit(a + beta)
    else:
        q = (expit((a + beta)[:, None] + sig * gx) * gw).sum(1)
    q = np.clip(q, 1e-300, 1 - 1e-16)
    ll = S * np.log(q) + (M - S) * np.log1p(-q); w = wb * p2 * np.exp(ll - ll.max())
    return np.sum(w * q) / np.sum(w)
print("g2F1 M=1,S=0,a=0,sig=.3,g=1e6 top45", pm(1e6, 0, 1, 0, .3, 45), "top1e13", pm(1e6, 0, 1, 0, .3, 1e13))
for a in (1, -1):
    print("line17 M50 S15 sig0 a", a, [round(pm(g, a, 50, 15, 0, 1e13), 4) for g in (1, 10, 1e3, 1e6)], "trunc g=1e6", round(pm(1e6, a, 50, 15, 0, 45), 4))
# g2 F4: logistic fit on selected, X in -1,0,1, e=.5+.3X, theta=(.1,.8,.1)
X = np.array([-1., 0, 1]); e = .5 + .3 * X; th = np.array([.1, .8, .1])
def sc(p):
    q = expit(p[0] + p[1] * X); r = e * (th - q)
    return [r.sum(), (r * X).sum()]
p = optimize.fsolve(sc, [0, 0], xtol=1e-12); q = expit(p[0] + p[1] * X)
print("g2F4 fit", p, "q", q, "mean prediction", q.mean(), "target", th.mean())
Z = 1 / e; m = np.column_stack([np.ones(3), Z])
def sc2(p):
    q = expit(m @ p); return m.T @ (e * (th - q))
p2 = optimize.fsolve(sc2, [0, 0], xtol=1e-12); print("with 1/e covariate, no X: mean prediction", expit(m @ p2).mean())
# g2 F5: tail means
b, c = 1, 1000; a = c / (2 * np.pi); ks = np.arange(61, 2000001)
print("g2F5 script tail", b / 60 / (2 * np.pi ** 2), "exact", b * np.sum(1 / ((ks - .5) ** 2 + a ** 2)) / (2 * np.pi ** 2), "closed", b * np.arctan2(a, 60) / a / (2 * np.pi ** 2))
