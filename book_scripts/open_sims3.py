"""Further work on the open problems: Problems 2, 3 and 5.

Revised 23 October 2026 (r1023): Problem 2 has a third outcome prior, "tuned kernel, true noise" (amplitude and length scales tuned as before, noise variance fixed at its true value 1); same 40 replications and seed 303, and the random stream is unchanged because the priors consume no random numbers. Standard errors added.
Produces figs/open_gpcouple.pdf, figs/open_rrw_unknown.pdf, figs/open_transport.pdf and results/res_open3.json.
Requires numpy, scipy, scikit-learn and matplotlib. About ten minutes on two cores.
"""
import numpy as np, json, time, os, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
PAL = ["#1f5a96", "#d1495b", "#edae49", "#00798c", "#66a182", "#8d5fa8"]
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "pdf.fonttype": 42, "axes.prop_cycle": plt.cycler(color=PAL)})
rng = np.random.default_rng(303); FIG = "figs/"; os.makedirs("results", exist_ok=True); R = {}; t0 = time.time()
expit = lambda z: 1 / (1 + np.exp(-z)); logit = lambda p: np.log(p / (1 - p))

# ------------------------------------------------ Problem 2: coupling a Gaussian process through the propensity direction
def gp_tau(y, t, X, mhat, mode, ell=1.0, amp=1.0, lam=0.0, noise=1.0):
    """Partially linear model y = tau t + g(X) + e with a Gaussian process prior on g.
    mode 'plain': kernel k; 'coupled': kernel k + lam * mhat mhat^T; 'covariate': mhat as an unpenalized column.
    Returns the posterior mean and sd of tau (noise variance treated as known)."""
    n = len(y); D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    K = amp * np.exp(-D / (2 * ell ** 2))
    if mode == "coupled": K = K + lam * np.outer(mhat, mhat)
    F = np.column_stack([np.ones(n), t] + ([mhat] if mode == "covariate" else []))
    # integrate g out: y ~ N(F b, K + noise I) with flat prior on b
    S = K + noise * np.eye(n); L = np.linalg.cholesky(S)
    Si = lambda M: np.linalg.solve(L.T, np.linalg.solve(L, M))
    A = F.T @ Si(F); bhat = np.linalg.solve(A, F.T @ Si(y)); cov = np.linalg.inv(A)
    return bhat[1], np.sqrt(cov[1, 1])

def design(n, p=5):
    X = rng.normal(size=(n, p)); m = np.sin(1.5 * X[:, 0]) + 0.5 * X[:, 1] ** 2 - 0.5 + 0.8 * X[:, 2]
    t = m + rng.normal(size=n); g = 2.0 * m + 0.5 * np.cos(X[:, 3])   # confounding: g shares the treatment surface
    y = 1.0 * t + g + rng.normal(size=n)
    return X, t, y, m

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
def gp_tau_ard(y, t, X, mhat, mode, ells, amp, noise, lam=0.0):
    n = len(y); D = ((X[:, None, :] - X[None, :, :]) ** 2 / (2 * ells ** 2)).sum(-1); K = amp * np.exp(-D)
    if mode == "coupled": K = K + lam * np.outer(mhat, mhat)
    F = np.column_stack([np.ones(n), t] + ([mhat] if mode == "covariate" else []))
    S = K + noise * np.eye(n); L = np.linalg.cholesky(S); Si = lambda M: np.linalg.solve(L.T, np.linalg.solve(L, M))
    A = F.T @ Si(F); b = np.linalg.solve(A, F.T @ Si(y)); cov = np.linalg.inv(A); return b[1], np.sqrt(cov[1, 1])

rows = []
for r in range(40):
    X, t, y, m = design(300); n = len(t)
    # propensity surface: a Gaussian process of t on X with tuned ARD kernel, cross fitted in two folds
    gpt = GaussianProcessRegressor(C(1.0) * RBF(np.ones(5)) + WhiteKernel(1.0), normalize_y=True, random_state=0).fit(X, t)
    mhat = np.zeros(n); idx = rng.permutation(n); folds = np.array_split(idx, 2)
    for k in range(2):
        tr, te = folds[1 - k], folds[k]
        mhat[te] = GaussianProcessRegressor(gpt.kernel_, normalize_y=True, optimizer=None).fit(X[tr], t[tr]).predict(X[te])
    # outcome prior, tuned: ARD kernel fitted to y on X (a crude but standard choice)
    gpy = GaussianProcessRegressor(C(1.0) * RBF(np.ones(5)) + WhiteKernel(1.0), normalize_y=True, random_state=0).fit(X, y - y.mean())
    kern = gpy.kernel_; amp_t = kern.k1.k1.constant_value * np.var(y); ells_t = kern.k1.k2.length_scale; noise_t = max(kern.k2.noise_level * np.var(y), 0.3)
    out = [r, np.corrcoef(mhat, m)[0, 1]]
    for prior in ["strong", "tuned", "tuned_true_noise"]:
        # r1023: third prior, amplitude and length scales tuned as in "tuned", noise variance fixed at its true value 1
        amp, ells, noise = {"strong": (1.0, np.full(5, 1.5), 1.0), "tuned": (amp_t, ells_t, noise_t), "tuned_true_noise": (amp_t, ells_t, 1.0)}[prior]
        for mode, lam in [("plain", 0), ("coupled", 10.0), ("coupled", 1000.0), ("covariate", 0)]:
            tau, sd = gp_tau_ard(y, t, X, mhat, mode, ells, amp, noise, lam)
            out += [tau - 1, float(abs(tau - 1) <= 1.96 * sd), 2 * 1.96 * sd]
    rows.append(out); print("P2", r, np.round(out[2::3], 3), round(time.time() - t0), flush=True)
rows = np.array(rows)
names = ["plain GP", "GP, lambda = 10", "GP, lambda = 1000", "propensity covariate"]
R["P2"] = {"n_rep": 40, "se_note": "bias_se and rmse_se are Monte Carlo standard errors over the replications", "corr_mhat": float(rows[:, 1].mean())}
for pi_, prior in enumerate(["strong", "tuned", "tuned_true_noise"]):
    R["P2"][prior] = {nm: dict(bias=float(rows[:, 2 + 12 * pi_ + 3 * i].mean()), rmse=float(np.sqrt((rows[:, 2 + 12 * pi_ + 3 * i] ** 2).mean())),
                               bias_se=float(rows[:, 2 + 12 * pi_ + 3 * i].std(ddof=1) / np.sqrt(len(rows))), rmse_se=float((rows[:, 2 + 12 * pi_ + 3 * i] ** 2).std(ddof=1) / np.sqrt(len(rows)) / (2 * np.sqrt((rows[:, 2 + 12 * pi_ + 3 * i] ** 2).mean()))), n_cover=int(rows[:, 3 + 12 * pi_ + 3 * i].sum()),
                               cover=float(rows[:, 3 + 12 * pi_ + 3 * i].mean()), width=float(rows[:, 4 + 12 * pi_ + 3 * i].mean())) for i, nm in enumerate(names)}
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
for ax, pi_, ttl in [(axs[0], 0, "strong prior on $g$"), (axs[1], 1, "tuned prior on $g$"), (axs[2], 2, "tuned, true noise")]:
    bp = ax.boxplot([rows[:, 2 + 12 * pi_ + 3 * i] for i in range(4)], tick_labels=["plain", "$\\lambda = 10$", "$\\lambda = 10^3$", "covariate"], patch_artist=True)
    for patch, c in zip(bp["boxes"], PAL): patch.set_facecolor(c); patch.set_alpha(0.5)
    ax.axhline(0, color="grey", lw=0.6); ax.set_title(ttl, fontsize=9); ax.tick_params(axis="x", labelsize=8)
axs[0].set_ylabel("estimate minus truth")
fig.tight_layout(); fig.savefig(FIG + "open_gpcouple.pdf"); plt.close()
print("P2", R["P2"], flush=True)

# ------------------------------------------------ Problem 3: Robins and Ritov with an unknown propensity
def rrw(K=20000, n=2000, smooth=True, m=10.0):
    z = rng.uniform(-1, 1, K)
    pi = expit(2.5 * z) * 0.9 + 0.05 if smooth else rng.uniform(0.05, 0.95, K)
    theta = expit(-1 + 6 * (pi - 0.5) + rng.normal(0, 0.3, K)); psi = theta.mean()
    cells = rng.integers(0, K, n); r = rng.binomial(1, pi[cells]); y = rng.binomial(1, theta[cells])
    ht = np.mean(r * y / pi[cells])
    obs = cells[r == 1]; yo = y[r == 1]
    s = np.bincount(obs, weights=yo, minlength=K); c = np.bincount(obs, minlength=K)
    def coupled(pihat):
        lr = LogisticRegression(C=1e6).fit(logit(pihat[obs]).reshape(-1, 1), yo)
        prior_mean = lr.predict_proba(logit(pihat).reshape(-1, 1))[:, 1]
        return ((m * prior_mean + s) / (m + c)).mean()
    # propensity learned from (z, r) on every unit by a logistic regression in z, as the smooth prior would
    lr_pi = LogisticRegression(C=1e6).fit(z[cells].reshape(-1, 1), r)
    pihat = np.clip(lr_pi.predict_proba(z.reshape(-1, 1))[:, 1], 0.02, 0.98)
    ht_hat = np.mean(r * y / pihat[cells])
    return psi, ht, coupled(pi), coupled(pihat), ht_hat
res = {}
for smooth in [True, False]:
    out = np.array([rrw(smooth=smooth) for _ in range(100)]); err = out[:, 1:] - out[:, [0]]
    res["smooth" if smooth else "rough"] = dict(names=["HT, known pi", "coupled, known pi", "coupled, estimated pi", "HT, estimated pi"],
                                                bias=err.mean(0).tolist(), rmse=np.sqrt((err ** 2).mean(0)).tolist())
    print("P3", smooth, res["smooth" if smooth else "rough"], round(time.time() - t0), flush=True)
    res["smooth" if smooth else "rough"]["err"] = err.tolist()
R["P3"] = {k: {kk: vv for kk, vv in v.items() if kk != "err"} for k, v in res.items()}
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
for ax, key, ttl in [(axs[0], "smooth", "propensity smooth in $z$"), (axs[1], "rough", "propensity unrelated to $z$")]:
    err = np.array(res[key]["err"])
    bp = ax.boxplot([err[:, i] for i in range(4)], tick_labels=["HT\nknown", "coupled\nknown", "coupled\nestimated", "HT\nestimated"], patch_artist=True)
    for patch, c in zip(bp["boxes"], PAL): patch.set_facecolor(c); patch.set_alpha(0.5)
    ax.axhline(0, color="grey", lw=0.6); ax.set_title(ttl, fontsize=9); ax.tick_params(axis="x", labelsize=7)
axs[0].set_ylabel("estimate minus truth")
fig.tight_layout(); fig.savefig(FIG + "open_rrw_unknown.pdf"); plt.close()

# ------------------------------------------------ Problem 5: transport with negative controls
# Experiment in A: arm probabilities P_A(y_x), P_A(y_x'); observational study in B gives P_B(x, y).
# Transport: logit P_B(y_x) = logit P_A(y_x) + d_x, d_x ~ N(0, s^2); K negative control outcomes share the same s.
true_s = 0.5
# population B: a binary confounder U, treatment X and outcome Y, all probabilities chosen so the law is coherent
pU = 0.5; pX_U = {1: 0.6, 0: 0.2}; pY_XU = {(1, 1): 0.7, (1, 0): 0.4, (0, 1): 0.3, (0, 0): 0.1}
pB1 = pU * pY_XU[(1, 1)] + (1 - pU) * pY_XU[(1, 0)]; pB0 = pU * pY_XU[(0, 1)] + (1 - pU) * pY_XU[(0, 0)]
px1 = pU * pX_U[1] + (1 - pU) * pX_U[0]
py_x1 = (pU * pX_U[1] * pY_XU[(1, 1)] + (1 - pU) * pX_U[0] * pY_XU[(1, 0)]) / px1
py_x0 = (pU * (1 - pX_U[1]) * pY_XU[(0, 1)] + (1 - pU) * (1 - pX_U[0]) * pY_XU[(0, 0)]) / (1 - px1)
py = px1 * py_x1 + (1 - px1) * py_x0
# the experiment was run in population A, whose arm probabilities differ from B's by transport discrepancies
dB1, dB0 = rng.normal(0, true_s, 2)
pA1, pA0 = expit(logit(pB1) - dB1), expit(logit(pB0) - dB0)
def pn_lower(pb1, pb0):
    # Tian and Pearl: PN >= max{0, (P(y) - P(y_x')) / P(x, y)}, PN <= min{1, (P(y'_x') - P(x', y')) / P(x, y)}  with x' untreated
    pxy = px1 * py_x1; pxpyp = (1 - px1) * (1 - py_x0)
    return max(0.0, (py - pb0) / pxy), min(1.0, (1 - pb0 - pxpyp) / pxy)
true_lo, true_hi = pn_lower(pB1, pB0)
Kneg = 6; dneg_true = rng.normal(0, true_s, Kneg)
nA = nB = 2000; negA = expit(rng.normal(0, 1, Kneg)); negB = expit(logit(negA) + dneg_true)
yA = rng.binomial(nA, negA); yB = rng.binomial(nB, negB)
# posterior for s from the negative controls (grid), then posterior for the PN bounds at each s
sg = np.linspace(0.02, 2.0, 100); logpost = np.zeros_like(sg)
for j, sv in enumerate(sg):
    # integrate the discrepancy for each control numerically
    for k in range(Kneg):
        dgrid = np.linspace(-4, 4, 321); pa = yA[k] / nA
        like = stats.binom.pmf(yB[k], nB, expit(logit(pa) + dgrid)) * stats.norm.pdf(dgrid, 0, sv)
        logpost[j] += np.log(np.trapezoid(like, dgrid) + 1e-300)
post_s = np.exp(logpost - logpost.max()); post_s /= np.trapezoid(post_s, sg)
# Only the untreated arm P_B(y_x') enters the bounds. B's observational law allows P_B(x', y) <= P_B(y_x') <= P_B(x', y) + P_B(x), so a
# transported value outside that interval is incompatible with what B shows, and the prior is restricted to the interval: given s,
# logit P_B(y_x') is normal with mean logit P_A(y_x') and sd s, truncated to the logits of the two ends. Both bounds are nonincreasing
# functions of P_B(y_x'), so the u quantile of a bound is the bound at the 1 - u quantile of P_B(y_x'), and every quantile below is exact
# (an earlier version used 4,000 rejection draws, whose Monte Carlo error moved printed values across a rounding boundary).
pxpy = (1 - px1) * py_x0; ok_lo, ok_hi = pxpy, pxpy + px1
la, lb, m0 = logit(ok_lo), logit(ok_hi), logit(pA0)
def kept_mass(sv):
    return stats.norm.cdf((lb - m0) / sv) - stats.norm.cdf((la - m0) / sv)
def pb0_cdf(c, sv):
    # P(P_B(y_x') <= c | s) under the restricted prior, for ok_lo <= c <= ok_hi; vectorized over sv
    sv = np.asarray(sv, dtype=float)
    Fa = stats.norm.cdf((la - m0) / sv)
    return (stats.norm.cdf((logit(c) - m0) / sv) - Fa) / kept_mass(sv)
def pb0_quantile(u, sv):
    Fa, Fb = stats.norm.cdf((la - m0) / sv), stats.norm.cdf((lb - m0) / sv)
    return float(expit(m0 + sv * stats.norm.ppf(Fa + u * (Fb - Fa))))
def bound_quantiles(q):
    # q(u) is the u quantile of P_B(y_x'); returns the 5 and 50 percent points of the lower bound and the 50 and 95 percent points of the upper
    return [pn_lower(pB1, q(0.95))[0], pn_lower(pB1, q(0.5))[0], pn_lower(pB1, q(0.5))[1], pn_lower(pB1, q(0.05))[1]]
svals = [0.0, 0.25, 0.5, 1.0, 2.0]
curve = {sv: bound_quantiles(lambda u, sv=sv: pb0_quantile(u, max(sv, 1e-6))) for sv in svals}
rejected_share = {sv: float(1 - kept_mass(max(sv, 1e-6))) for sv in svals}
# posterior predictive bounds integrating s over its negative control posterior: the mixture over the grid, inverted by root finding
from scipy import optimize
w_s = post_s / post_s.sum()
def mix_quantile(u):
    return optimize.brentq(lambda c: float(np.sum(w_s * pb0_cdf(c, sg))) - u, ok_lo + 1e-12, ok_hi - 1e-12, xtol=1e-12)
integrated = bound_quantiles(mix_quantile)
R["P5"] = dict(true_s=true_s, true_bounds=[float(true_lo), float(true_hi)], transported_bounds=[float(x) for x in pn_lower(pA1, pA0)],
               post_s_mean=float(np.trapezoid(sg * post_s, sg)), post_s_ci=[float(sg[np.searchsorted(np.cumsum(post_s) / post_s.sum(), q)]) for q in [0.025, 0.975]],
               bounds_by_s={str(sv): [float(x) for x in curve[sv]] for sv in svals},
               integrated=[float(x) for x in integrated], Kneg=Kneg,
               compatible_interval=[float(ok_lo), float(ok_hi)], rejected_share_by_s={str(sv): rejected_share[sv] for sv in svals},
               quantiles="exact, from the truncated normal law of logit P_B(y_x') given s; the integrated row mixes over the grid posterior of s")
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9))
# Gray scale print: black posterior and dashed gray true value; the 5 to 95 percent bars are light gray with dark median segments (black for the posterior
# bar), and the band of the true bounds is a light fill whose edges are dashed lines, so that it survives a faint print
axs[0].plot(sg, post_s, color="k"); axs[0].axvline(true_s, color="0.3", ls="--", lw=1); axs[0].set_xlabel("transport scale $s$"); axs[0].set_ylabel("posterior density"); axs[0].set_yticks([])
axs[0].set_title(f"$s$ from {Kneg} negative controls", fontsize=9)
xs = np.arange(len(svals))
for i, sv in enumerate(svals):
    q05_lo, q50_lo, q50_hi, q95_hi = curve[sv]
    axs[1].plot([i, i], [q05_lo, q95_hi], color="0.8", lw=6)
    axs[1].plot([i, i], [q50_lo, q50_hi], color="0.3", lw=2.5)
axs[1].plot([len(svals), len(svals)], [integrated[0], integrated[3]], color="0.8", lw=6)
axs[1].plot([len(svals), len(svals)], [integrated[1], integrated[2]], color="k", lw=2.5)
axs[1].axhspan(true_lo, true_hi, color="0.93")
axs[1].axhline(true_lo, color="0.3", ls="--", lw=0.8); axs[1].axhline(true_hi, color="0.3", ls="--", lw=0.8)
axs[1].set_xticks(list(xs) + [len(svals)]); axs[1].set_xticklabels([str(v) for v in svals] + ["post."], fontsize=8)
axs[1].set_xlabel("assumed transport scale $s$"); axs[1].set_ylabel("PN in population B"); axs[1].set_title("bounds on PN", fontsize=9)
fig.tight_layout(); fig.savefig(FIG + "open_transport.pdf"); plt.close()
R["secs"] = round(time.time() - t0)
json.dump(R, open("results/res_open3.json", "w"), indent=1, default=float); print(json.dumps(R, default=float, indent=1))
