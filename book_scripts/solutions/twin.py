import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(1958)
N = 2_000_000
f1 = rng.uniform(size=N); lPR = np.abs(rng.normal(0, 2.0, N)); lR = np.abs(rng.normal(0, 2.0, N)); PR = np.exp(lPR); R = np.exp(lR)
print("sigma check: median, q95 of half normal(2):", np.exp(0.6745*2), np.exp(1.96*2))
def gshare(f1, PR, pb):
    f0 = f1 / PR; f = pb * f1 + (1 - pb) * f0
    p1 = pb * f1 / f; p0 = pb * (1 - f1) / (1 - f)
    p1 = np.clip(p1, 1e-12, 1 - 1e-12); p0 = np.clip(p0, 1e-12, 1 - 1e-12)
    d = norm.ppf(p1) - norm.ppf(p0); v = f * (1 - f) * d ** 2
    return v / (1 + v), f, d
B = (1 + f1 * (R - 1)) / (1 + f1 / PR * (R - 1))
print("sceptical: P(B>=9)=", np.mean(B >= 9), "P(B>=60)=", np.mean(B >= 60), "PR median, q95", np.median(PR), np.quantile(PR, .95))
for c in [(.80, .69)]:
    r = [np.sin(np.pi * (x - .5)) for x in c]; print("concordance", c, "tetrachoric", r, "2(rMZ-rDZ)", 2 * (r[0] - r[1]))
for pb in (0.5, 0.3, 0.7):
    g, f, d = gshare(f1, PR, pb)
    print("pb", pb, "g quantiles", np.quantile(g, [.1, .25, .5, .75, .9, .99]).round(3))
    for cap in (0.7, 0.5, 0.3, 0.1, 0.05, 0.02):
        s = g <= cap
        print(f" cap {cap}: mass {s.mean():.3f} PR median {np.median(PR[s]):.2f} q95 {np.quantile(PR[s], .95):.2f} P(B>=9) {np.mean(B[s] >= 9):.2e} P(B>=60) {np.mean(B[s] >= 60):.2e} f1 mean {f1[s].mean():.3f}")
# table of g at selected (f1, PR), pb=.5
for f in (0.05, 0.2, 0.5, 0.9):
    print("f1", f, [(pr, round(float(gshare(np.array([f]), np.array([pr]), .5)[0][0]), 3)) for pr in (1.6, 2, 4, 9, 20, 50)])
# exact check of the concordance implied by a binary U with share g (normal residual), MZ share U & DZ corr 1/2, rest zero genetic correlation in MZ and DZ
def conc(f, g, rhoU, rrest, nn=2000000, seed=1):
    r = np.random.default_rng(seed)
    dl = np.sqrt(g / (f * (1 - f))); sig = np.sqrt(1 - g)
    # U pair with correlation rhoU
    p11 = f * f + rhoU * f * (1 - f)
    u1 = r.random(nn) < f
    pc = np.where(u1, p11 / f, (f - p11) / (1 - f))
    u2 = r.random(nn) < pc
    e = r.multivariate_normal([0, 0], [[1, rrest], [rrest, 1]], nn)
    l1 = dl * (u1 - f) + sig * e[:, 0]; l2 = dl * (u2 - f) + sig * e[:, 1]
    c = np.quantile(l1, .5); s1 = l1 > c; s2 = l2 > c
    return (s1 & s2).sum() / s1.sum()
print("exact concordance, f=.2, g=.5, MZ rho=1 rest .6, DZ rho=.5 rest .6:", conc(.2, .5, 1, .6), conc(.2, .5, .5, .6))
ms = [conc(.2, .5, 1, .6), conc(.2, .5, .5, .6)]; r = [np.sin(np.pi * (x - .5)) for x in ms]; print(" falconer from these", 2 * (r[0] - r[1]))
ms = [conc(.05, .5, 1, .6), conc(.05, .5, .5, .6)]; r = [np.sin(np.pi * (x - .5)) for x in ms]; print(" f=.05 falconer", 2 * (r[0] - r[1]))
