"""Checks for fill-g1: ex:ch04:5 simulation, and numbers for ex:chframeworks:5, 8, ex:ch03:6."""
import sys, json, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "book_scripts"))
import numpy as np
from scipy import stats, integrate, optimize
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
import ric_sim

def data(rep, n=200, p=50):
    rng = np.random.default_rng(ric_sim.SEED0 + rep)
    X = rng.normal(size=(n, p)); g = np.zeros(p); g[:5] = 1; b = g.copy()
    t = X @ g + rng.normal(size=n); eps = rng.normal(size=n)
    return rng, X, t, t + X @ b + eps

def dml(X, t, y, s):
    n = len(y); ry = np.zeros(n); rt = np.zeros(n)
    for tr, te in KFold(5, shuffle=True, random_state=s).split(X):
        ry[te] = y[te] - LassoCV(cv=5).fit(X[tr], y[tr]).predict(X[te])
        rt[te] = t[te] - LassoCV(cv=5).fit(X[tr], t[tr]).predict(X[te])
    th = rt @ ry / (rt @ rt); psi = rt * (ry - th * rt)
    return th, np.sqrt((psi ** 2).sum()) / (rt @ rt)

# one data set: replication 1 of the book's design, same stream as ric_sim.one_rep
rng, X, t, y = data(1)
saved = json.load(open(ROOT / "results/ric/rep_1.json"))
res = np.array([dml(X, t, y, s) for s in range(1, 11)])
print("DML splits est", np.round(res[:, 0], 4), "se", np.round(res[:, 1], 4))
print("min %.4f max %.4f range %.4f sd %.4f median %.4f mean se %.4f" % (res[:, 0].min(), res[:, 0].max(), np.ptp(res[:, 0]), res[:, 0].std(ddof=1), np.median(res[:, 0]), res[:, 1].mean()))
print("saved dml split 1", saved["est"]["dml"], saved["se_or_posterior_sd"]["dml"], "mine", res[0])
Z = np.column_stack([t, X]); ric_sim.hs_gibbs(y, Z, unpen={0}, rng=rng)
that = LassoCV(cv=5).fit(X, t).predict(X); Z2 = np.column_stack([t, that, X])
d2 = ric_sim.hs_gibbs(y, Z2, unpen={0, 1}, rng=rng)
print("coupled mean %.4f sd %.4f; saved %.4f %.4f" % (d2[:, 0].mean(), d2[:, 0].std(), saved["est"]["hs_coupled"], saved["se_or_posterior_sd"]["hs_coupled"]))
q = np.quantile(d2[:, 0], [.025, .975]); print("interval", q)
chains = []
for c in range(1, 5):
    d = ric_sim.hs_gibbs(y, Z2, unpen={0, 1}, rng=np.random.default_rng(7000 + c))
    chains.append((d[:, 0].mean(), d[:, 0].std()))
print("extra chains mean, sd", np.round(chains, 4))
print("saved cross fitted", saved["est"]["hs_coupled_cf"], saved["se_or_posterior_sd"]["hs_coupled_cf"])

# other data sets: split spread against posterior sd
rows = []
for rep in range(2, 21):
    _, Xr, tr_, yr = data(rep)
    e = np.array([dml(Xr, tr_, yr, s)[0] for s in range(1, 11)])
    r = json.load(open(ROOT / ("results/ric/rep_%d.json" % rep)))
    rows.append((e.std(ddof=1), np.ptp(e), r["se_or_posterior_sd"]["hs_coupled"], r["se_or_posterior_sd"]["dml"]))
rows = np.array(rows)
print("reps 2-20: median split sd %.4f max %.4f, median range %.4f max %.4f, median post sd %.4f min %.4f, median dml se %.4f; max ratio split sd/post sd %.3f" % (
    np.median(rows[:, 0]), rows[:, 0].max(), np.median(rows[:, 1]), rows[:, 1].max(), np.median(rows[:, 2]), rows[:, 2].min(), np.median(rows[:, 3]), (rows[:, 0] / rows[:, 2]).max()))
allr = [json.load(open(f)) for f in glob.glob(str(ROOT / "results/ric/rep_*.json"))]
ps = np.array([r["se_or_posterior_sd"]["hs_coupled"] for r in allr]); ds = np.array([r["se_or_posterior_sd"]["dml"] for r in allr])
ed = np.array([r["est"]["dml"] for r in allr]); ec = np.array([r["est"]["hs_coupled"] for r in allr])
print("saved %d reps: median post sd %.4f, median dml se %.4f, sd of dml est %.4f, sd of coupled mean %.4f" % (len(allr), np.median(ps), np.median(ds), ed.std(ddof=1), ec.std(ddof=1)))

# ex:chframeworks:5 Makarov style bounds for Y(0)~N(0,1), Y(1)=Y(0)+-1 with type independent of Y(0)
F0 = stats.norm.cdf; F1 = lambda y: .5 * F0(y - 1) + .5 * F0(y + 1)
lo = optimize.minimize_scalar(lambda y: -(F1(y) - F0(y)), bounds=(-4, 0), method="bounded")
hi = optimize.minimize_scalar(lambda y: -(F0(y) - F1(y)), bounds=(0, 4), method="bounded")
print("harm lower %.4f at %.3f; F0-F1 sup %.4f at %.3f; var check" % (-lo.fun, lo.x, -hi.fun, hi.x))

# ex:chframeworks:8 pair model
kap, sig, lam = 1., 1., .9
s = np.sqrt(2 * kap**2 + 2 * sig**2); al = kap / sig
dens = lambda z, l: 2 * stats.norm.pdf(z, 0, s) * (l * stats.norm.cdf(al * z / s) + (1 - l) * stats.norm.cdf(-al * z / s))
print("mass", integrate.quad(dens, -30, 30, args=(lam,))[0], "mean", integrate.quad(lambda z: z * dens(z, lam), -30, 30)[0], "formula", (2 * lam - 1) * 2 * kap / np.sqrt(np.pi))
# simulation check of the density: seed 5
r = np.random.default_rng(5); N = 10**6
V = r.normal(0, np.sqrt(2) * kap, N); sgn = np.where(r.uniform(size=N) < lam, 1, -1); W = sgn * np.abs(V); D = W + r.normal(0, np.sqrt(2) * sig, N)
print("mc P(D<0.5) %.4f density integral %.4f" % ((D < .5).mean(), integrate.quad(dens, -30, .5, args=(lam,))[0]))
# posterior of tau on grid, one data set m=50
r = np.random.default_rng(20261101); m = 50; tau_true = 1.
V = r.normal(0, np.sqrt(2) * kap, m); sgn = np.where(r.uniform(size=m) < lam, 1, -1); D = tau_true + sgn * np.abs(V) + r.normal(0, np.sqrt(2) * sig, m)
grid = np.linspace(-4, 6, 8001)
def post(l):
    lp = np.array([np.log(dens(D - g, l)).sum() for g in grid]); w = np.exp(lp - lp.max()); w /= w.sum()
    mu = (w * grid).sum(); return mu, np.sqrt((w * (grid - mu) ** 2).sum())
print("Dbar %.3f s/sqrt(m) %.3f" % (D.mean(), s / np.sqrt(m)), "post lam .5", post(.5), "post lam .9", post(.9), "post lam 1", post(1.))

# ex:ch03:6 numbers
za = stats.norm.ppf(.95); zb = stats.norm.ppf(.8); B = za - stats.norm.ppf(.6)
a1, a0 = .8, .05; b0 = .4; b1 = stats.norm.cdf(za + zb + B - za)
print("shift delta/sd %.3f B/sd %.3f b1 %.4f b0 check %.4f" % (za + zb, B, b1, stats.norm.sf(za - B)))
pi, c = .5, .8; ell = c / (1 - c) * (1 - pi) / pi
b1r = round(b1, 3)
LR = lambda q: (q * a1 + (1 - q) * b1r) / (q * a0 + (1 - q) * b0)
qs = (b1r - ell * b0) / ((b1r - a1) + ell * (a0 * -1 + b0)) ; print("ell", ell, "a1/a0", a1 / a0, "b1/b0", b1r / b0, "N LRs", (1 - a1) / (1 - a0), (1 - b1r) / (1 - b0))
print("q* via root", optimize.brentq(lambda q: LR(q) - ell, 0, 1), "LR(.3)", LR(.3), "post(.3)", LR(.3) / (LR(.3) + 1), "LR(.9)", LR(.9))
g = pi * a1 + (1 - pi) * a0; gP = pi * b1r + (1 - pi) * b0
Vc = pi * a1 * (1 - c) - (1 - pi) * a0 * c; Vp = pi * b1r * (1 - c) - (1 - pi) * b0 * c
print("g %.4f gP %.4f V clean %.4f V partisan %.4f mix q=.9 %.4f" % (g, gP, Vc, Vp, .9 * Vc + .1 * Vp))
