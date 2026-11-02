import numpy as np
from scipy.stats import norm, multivariate_normal as mvn
rng = np.random.default_rng(1958)
N = 2_000_000
f1 = rng.uniform(size=N); PR = np.exp(np.abs(rng.normal(0, 2.0, N))); R = np.exp(np.abs(rng.normal(0, 2.0, N)))
B = (1 + f1 * (R - 1)) / (1 + f1 / PR * (R - 1))
def phi2(f1, PR, pb):
    f0 = f1 / PR; f = pb * f1 + (1 - pb) * f0
    return (f1 - f0) ** 2 * pb * (1 - pb) / (f * (1 - f))
c = (.80, .69)
for pb in (.5, .3, .7):
    ph = [(x - pb) / (1 - pb) for x in c] if pb == .5 else None
    v = phi2(f1, PR, pb)
    print("pb", pb, "phi2 quantiles", np.quantile(v, [.1, .5, .9, .99]).round(3))
    for cap in (1.0, 0.44, 0.22, 0.11, 0.044, 0.02):
        s = v <= cap
        print(f" cap {cap}: mass {s.mean():.3f} PR med {np.median(PR[s]):.2f} q95 {np.quantile(PR[s], .95):.1f} P(B>=9) {np.mean(B[s] >= 9):.2e} P(B>=60) {np.mean(B[s] >= 60):.2e} P(PR>=9) {np.mean(PR[s]>=9):.3f} meanf1 {f1[s].mean():.3f}")
print("largest phi2 at PR=inf, pb=.5, f1=.1,.3,.5,.9:", [round(f / (2 - f), 3) for f in (.1, .3, .5, .9)])
for f in (.1, .3, .5, .9):
    print("f1", f, "phi2 at PR 1.6,2,4,9,inf (pb .5):", [round(float(phi2(np.array([f]), np.array([p]), .5)[0]), 3) for p in (1.6, 2, 4, 9, 1e9)])
# exact check of phi2(1-rhoU) <= phi_MZ - phi_DZ in the threshold model, bivariate normal residuals, rhoU = 1 (MZ), 1/2 and 1/4 (DZ)
def K(a, b, r):
    return mvn.cdf([a, b], mean=[0, 0], cov=[[1, r], [r, 1]])
def pair(f, d, t, rU, r):
    # thresholds on residual scale: P(S=1|u) = Phi(t + d u). P(both) given (u1,u2)
    p = [norm.cdf(t), norm.cdf(t + d)]; p11 = f * f + rU * f * (1 - f); pm = {(1, 1): p11, (1, 0): f - p11, (0, 1): f - p11, (0, 0): 1 - 2 * f + p11}
    pb = (1 - f) * p[0] + f * p[1]
    both = sum(pm[(u, v)] * K(t + d * u, t + d * v, r) for u in (0, 1) for v in (0, 1))
    return pb, both / pb
from scipy.optimize import brentq
print("check: f,g,rho_rest_DZ,rho_rest_MZ, phi2, phiMZ-phiDZ, ratio (must be >= 1-rhoU=.5)")
for f, g, rd, rm in [(.2, .5, .6, .6), (.05, .5, .6, .6), (.5, .2, .3, .6), (.1, .1, .0, .0), (.02, .8, .5, .5), (.3, .3, .2, .5), (.8, .4, .4, .4), (.1, .6, .9, .9)]:
    dl = np.sqrt(g / (f * (1 - f))); d = dl / np.sqrt(1 - g)
    t = brentq(lambda t: (1 - f) * norm.cdf(t) + f * norm.cdf(t + d) - .5, -20, 20)
    p0, p1 = norm.cdf(t), norm.cdf(t + d)
    ph2 = f * (1 - f) * (p1 - p0) ** 2 / .25
    pbm, cm = pair(f, d, t, 1.0, rm); pbd, cd = pair(f, d, t, .5, rd)
    phm, phd = (cm - .5) / .5, (cd - .5) / .5
    print(f, g, rd, rm, round(ph2, 4), round(phm - phd, 4), round((phm - phd) / ph2, 3), "falconer tetra", round(2 * (np.sin(np.pi * (cm - .5)) - np.sin(np.pi * (cd - .5))), 3))
