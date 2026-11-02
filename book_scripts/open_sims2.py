"""Further work on the open problems of the final chapter: Problems 1, 4, 6, 7, 8 and 9.

Produces figs/open_bvm.pdf, figs/open_arcsine.pdf, figs/open_twins2.pdf,
figs/open_rates.pdf, figs/open_product2.pdf and results/res_open2.json.
Requires numpy, scipy, scikit-learn, matplotlib and torch. About 30 minutes on two cores (Problem 8 now runs separately, see below).
"""
import numpy as np, json, time, warnings, os
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats, integrate
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
PAL = ["#1f5a96", "#d1495b", "#edae49", "#00798c", "#66a182", "#8d5fa8"]
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "pdf.fonttype": 42, "axes.prop_cycle": plt.cycler(color=PAL)})
rng = np.random.default_rng(2026); FIG = "figs/"; os.makedirs("results", exist_ok=True); R = {}; t0 = time.time()
IG = lambda a, b: 1 / rng.gamma(a, 1 / b)

def joint_gibbs(y, t, X, prior, iters=1500, burn=500, tdf=None, extra=None):
    """Joint outcome and treatment regressions. prior in {'indep','shared','product'}.
    Outcome: y = tau t + c + X beta (+ delta * extra) + e ; treatment: t = X gamma + nu.
    tdf: if not None, Student t errors in the outcome via scale mixture."""
    n, p = X.shape
    cols = [t, np.ones(n)] + ([extra] if extra is not None else [])
    k0 = len(cols); Zy = np.column_stack(cols + [X]); q = Zy.shape[1]
    lam2 = np.ones(p); nul = np.ones(p); eta2 = np.ones(p); nue = np.ones(p)
    lt2 = np.ones(p); nut = np.ones(p)
    ty2 = tt2 = 1.0; xy = xt = 1.0; sy2 = np.var(y); st2 = np.var(t); w = np.ones(n)
    XtX = X.T @ X; Xtt = X.T @ t; out = []
    for it in range(iters):
        if prior == "indep": vb = lam2; vg = lt2
        elif prior == "shared": vb = lam2; vg = lam2
        else: vb = lam2 * eta2; vg = lam2
        W = Zy * w[:, None]; A = Zy.T @ W + np.diag(np.r_[np.full(k0, 1e-8), 1 / (vb * ty2)])
        L = np.linalg.cholesky(A); b = np.linalg.solve(A, W.T @ y) + np.sqrt(sy2) * np.linalg.solve(L.T, rng.normal(size=q))
        beta = b[k0:]; ry = y - Zy @ b
        Bm = XtX + np.diag(1 / (vg * tt2)); Lb = np.linalg.cholesky(Bm)
        g = np.linalg.solve(Bm, Xtt) + np.sqrt(st2) * np.linalg.solve(Lb.T, rng.normal(size=p)); rt = t - X @ g
        sy2 = IG((n + p) / 2, (w * ry ** 2).sum() / 2 + (beta ** 2 / (vb * ty2)).sum() / 2)
        st2 = IG((n + p) / 2, rt @ rt / 2 + (g ** 2 / (vg * tt2)).sum() / 2)
        if tdf is not None:
            w = rng.gamma((tdf + 1) / 2, 1 / ((tdf + ry ** 2 / sy2) / 2))
        Bq = beta ** 2 / (2 * ty2 * sy2); Gq = g ** 2 / (2 * tt2 * st2)
        if prior == "indep":
            lam2 = IG(1, 1 / nul + Bq); lt2 = IG(1, 1 / nut + Gq); nut = IG(1, 1 + 1 / lt2)
        elif prior == "shared":
            lam2 = IG(1.5, 1 / nul + Bq + Gq)
        else:
            lam2 = IG(1.5, 1 / nul + Bq / eta2 + Gq)
            eta2 = IG(1, 1 / nue + Bq / lam2); nue = IG(1, 1 + 1 / eta2)
        nul = IG(1, 1 + 1 / lam2)
        if prior == "indep": vb = lam2; vg = lt2
        elif prior == "shared": vb = lam2; vg = lam2
        else: vb = lam2 * eta2; vg = lam2
        ty2 = IG((p + 1) / 2, 1 / xy + (beta ** 2 / vb).sum() / (2 * sy2)); xy = IG(1, 1 + 1 / ty2)
        tt2 = IG((p + 1) / 2, 1 / xt + (g ** 2 / vg).sum() / (2 * st2)); xt = IG(1, 1 + 1 / tt2)
        if it >= burn: out.append(b[0])
    return np.array(out)

def dml(y, t, X, seed=0):
    n = len(y); kf = KFold(5, shuffle=True, random_state=seed); ry = np.zeros(n); rt = np.zeros(n)
    for tr, te in kf.split(X):
        ry[te] = y[te] - LassoCV(cv=5).fit(X[tr], y[tr]).predict(X[te]); rt[te] = t[te] - LassoCV(cv=5).fit(X[tr], t[tr]).predict(X[te])
    th = (rt @ ry) / (rt @ rt); psi = (ry - th * rt) * rt
    return th, np.sqrt(np.mean(psi ** 2) / np.mean(rt ** 2) ** 2 / n)

def summ(d, truth=1.0):
    lo, hi = np.quantile(d, [0.025, 0.975]); return [d.mean() - truth, float(lo <= truth <= hi), hi - lo]


# ------------------------------------------------ Problem 1: larger n, 40 replications
resB = {}
for n in [400, 800, 1600, 3200]:
    rows = []
    for r in range(40):
        p = 50; X = rng.normal(size=(n, p)); g = np.zeros(p); g[:5] = 1.0; b = np.zeros(p); b[:5] = 1.0
        t = X @ g + rng.normal(size=n); y = t + X @ b + rng.normal(size=n)
        mhat = LassoCV(cv=5).fit(X, t).predict(X)
        d = joint_gibbs(y, t, X, "indep", extra=mhat, iters=1200, burn=400)
        se_or = 1 / np.sqrt(n); th, se = dml(y, t, X, seed=r)
        rows.append([d.mean() - 1, float(np.quantile(d, .025) <= 1 <= np.quantile(d, .975)), d.std() / se_or,
                     th - 1, float(abs(th - 1) <= 1.96 * se), se / se_or, np.sqrt(n) * (d.mean() - 1), stats.shapiro(d[::20])[1]])
    rows = np.array(rows)
    resB[n] = dict(rmse_hs=float(np.sqrt((rows[:, 0] ** 2).mean())), cover_hs=float(rows[:, 1].mean()), ratio_hs=float(rows[:, 2].mean()),
                   rmse_dml=float(np.sqrt((rows[:, 3] ** 2).mean())), cover_dml=float(rows[:, 4].mean()), ratio_dml=float(rows[:, 5].mean()),
                   sd_root_n_bias=float(rows[:, 6].std()), mean_root_n_bias=float(rows[:, 6].mean()), shapiro_p_median=float(np.median(rows[:, 7])))
    print("P1", n, {k: round(v, 3) for k, v in resB[n].items()}, round(time.time() - t0), flush=True)
R["P1"] = resB
ns = sorted(resB); fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.7))
ax[0].plot(ns, [resB[n]["ratio_hs"] for n in ns], "o-", label="coupled horseshoe"); ax[0].plot(ns, [resB[n]["ratio_dml"] for n in ns], "s--", label="DML")
ax[0].axhline(1, color="grey", lw=0.6); ax[0].set_xscale("log"); ax[0].set_xticks(ns); ax[0].set_xticklabels(ns); ax[0].set_xlabel("$n$"); ax[0].set_ylabel("sd relative to efficient sd"); ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(ns, [np.sqrt(n) * resB[n]["rmse_hs"] for n in ns], "o-", label="coupled horseshoe"); ax[1].plot(ns, [np.sqrt(n) * resB[n]["rmse_dml"] for n in ns], "s--", label="DML")
ax[1].axhline(1, color="grey", lw=0.6); ax[1].set_xscale("log"); ax[1].set_xticks(ns); ax[1].set_xticklabels(ns); ax[1].set_xlabel("$n$"); ax[1].set_ylabel("$\\sqrt{n}$ RMSE"); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG + "open_bvm.pdf"); plt.close()

# ------------------------------------------------ Problem 4: induced priors on the fraction helped
delta = 0.5; helped = lambda rho: stats.norm.cdf(delta / np.sqrt(np.maximum(2 * (1 - rho), 1e-12)))
rho_u = rng.uniform(-1, 1, 200000); rho_a = np.sin(np.pi * (rng.uniform(size=200000) - 0.5))
fig, ax = plt.subplots(figsize=(5.2, 3))
ax.hist(helped(rho_u), bins=60, density=True, alpha=0.5, label="uniform prior on $\\rho$")
ax.hist(helped(rho_a), bins=60, density=True, alpha=0.5, label="uniform prior on Kendall's $\\tau_K$ (arcsine on $\\rho$)")
ax.set_xlabel("fraction helped, $\\delta = 0.5$ standard deviations"); ax.set_yticks([]); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIG + "open_arcsine.pdf"); plt.close()
R["P4"] = dict(mean_helped_uniform=float(helped(rho_u).mean()), mean_helped_arcsine=float(helped(rho_a).mean()),
               p_helped_above_09_uniform=float(np.mean(helped(rho_u) > 0.9)), p_helped_above_09_arcsine=float(np.mean(helped(rho_a) > 0.9)))

# ------------------------------------------------ Problem 7: Makarov bounds under the shared component model
shares = np.linspace(0, 0.95, 60); s = np.sqrt(1 - shares)
mak_lo = 2 * stats.norm.cdf(delta / (2 * s)) - 1
gauss_lo = helped(np.maximum(2 * shares - 1, -1))
fig, ax = plt.subplots(figsize=(5.2, 3.1))
ax.fill_between(shares, mak_lo, 1, alpha=0.25, color=PAL[1], label="any coupling of the individual terms (Makarov)")
ax.fill_between(shares, gauss_lo, 1, alpha=0.35, color=PAL[0], label="Gaussian coupling")
ax.set_xlabel("share of outcome variance shared by twins"); ax.set_ylabel("$P(Y(1) > Y(0))$"); ax.legend(frameon=False, fontsize=7, loc="upper left"); ax.set_ylim(0.15, 1.12)
fig.tight_layout(); fig.savefig(FIG + "open_twins2.pdf"); plt.close()
# a twin trial with the share estimated: 100 pairs, true share 0.6, Bayesian bootstrap of the moments
npair, Vh = 100, 0.6; h = rng.normal(0, np.sqrt(Vh), npair)
y0 = h + rng.normal(0, np.sqrt(1 - Vh), npair); y1 = delta + h + rng.normal(0, np.sqrt(1 - Vh), npair)
draws = []
for b in range(4000):
    w = rng.dirichlet(np.ones(npair))
    m0, m1 = w @ y0, w @ y1; v0 = w @ (y0 - m0) ** 2; v1 = w @ (y1 - m1) ** 2; c = w @ ((y0 - m0) * (y1 - m1))
    c = min(max(c, 0), np.sqrt(v0 * v1) * 0.999); s0, s1 = np.sqrt(v0 - c), np.sqrt(v1 - c); d = m1 - m0
    # Makarov lower bound on P(d + e1 - e0 > 0) with e_t ~ N(0, s_t^2), computed by search over y
    yy = np.linspace(-4, 4, 801); lo = 1 - max(0, np.max(stats.norm.cdf(yy, 0, s0) - stats.norm.cdf(yy - d, 0, s1))) if False else None
    FA = lambda x: stats.norm.cdf(x, d, s1); FB = lambda x: stats.norm.cdf(x, 0, s0)
    upper_FDelta0 = 1 + np.min(FA(yy) - FB(yy)); lower_FDelta0 = max(0, np.max(FA(yy) - FB(yy)))
    draws.append([1 - upper_FDelta0, 1 - lower_FDelta0, c / np.sqrt(v0 * v1), d])
draws = np.array(draws)
R["P7"] = dict(npair=npair, true_share=Vh, true_delta=delta, mak_lo_true=float(2 * stats.norm.cdf(delta / (2 * np.sqrt(1 - Vh))) - 1),
               post_mak_lo_mean=float(draws[:, 0].mean()), post_mak_lo_ci=[float(x) for x in np.quantile(draws[:, 0], [0.025, 0.975])],
               post_share_mean=float(draws[:, 2].mean()), post_delta_mean=float(draws[:, 3].mean()),
               lo_share_0=float(mak_lo[0]), lo_share_0_5=float(2 * stats.norm.cdf(delta / (2 * np.sqrt(0.5))) - 1), lo_share_0_8=float(2 * stats.norm.cdf(delta / (2 * np.sqrt(0.2))) - 1))
print("P7", R["P7"], flush=True)

# ------------------------------------------------ Problem 8: rates under infinite variance errors
# Revised 23 October 2026 (r1023): the simulation (n = 100 to 6,400, 200 replications, t errors on 1.5 and 2 degrees of freedom and Gaussian errors, with the
# oracle score) moved to book_scripts/open_rates_hr.py (array job) and open_rates_hr_merge.py, which write results/res_open_rates.json and figs/open_rates.pdf.
# The earlier block (20 replications, n = 100 to 1,600) is in open_sims2.py.before-r1023. Here only the summary is copied into res_open2.json.
if os.path.exists("results/res_open_rates.json"):
    R["P8"] = json.load(open("results/res_open_rates.json"))["slopes"]

# ------------------------------------------------ Problem 9: shrinkage of a null coefficient as the shared scale grows
def shrink(y, c):
    """E[kappa | y] for beta ~ N(0, c^2 eta^2), eta ~ C+(0,1), y ~ N(beta, 1); kappa = 1/(1 + c^2 eta^2)."""
    f = lambda e: stats.norm.pdf(y, 0, np.sqrt(1 + c * c * e * e)) * 2 / (np.pi * (1 + e * e))
    num = integrate.quad(lambda e: f(e) / (1 + c * c * e * e), 0, np.inf, limit=200)[0]
    den = integrate.quad(f, 0, np.inf, limit=200)[0]
    return num / den
cs = np.logspace(-1, 2, 40)
fig, ax = plt.subplots(figsize=(5.2, 3.1))
for y, ls in [(0.0, "-"), (1.0, "--"), (2.0, ":")]:
    ax.semilogx(cs, [shrink(y, c) for c in cs], ls, label=f"$y = {y:.0f}$")
ax.set_xlabel("effective global scale $c = \\lambda_j \\tau_{\\mathrm{g},Y}$"); ax.set_ylabel("posterior shrinkage $E(\\kappa_j \\mid y)$"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG + "open_product2.pdf"); plt.close()
R["P9"] = {f"shrink_y0_c{c}": float(shrink(0.0, c)) for c in [0.3, 1, 3, 10, 30]}
R["P9"].update({f"shrink_y2_c{c}": float(shrink(2.0, c)) for c in [0.3, 1, 3, 10, 30]})
R["secs"] = round(time.time() - t0)
json.dump(R, open("results/res_open2.json", "w"), indent=1, default=float); print(json.dumps(R, default=float, indent=1))
