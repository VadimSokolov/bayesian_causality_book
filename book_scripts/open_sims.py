import numpy as np, json, time, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(1515); FIG = "figs/"; R = {}; t0 = time.time()
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

# A. Problem 9: product prior
resA = []
for r in range(40):
    n, p = 200, 50; X = rng.normal(size=(n, p))
    g = np.zeros(p); g[:5] = 0.5; g[5:10] = 1.0
    b = np.zeros(p); b[:5] = 0.5; b[10:15] = 1.0
    t = X @ g + rng.normal(size=n); y = t + X @ b + rng.normal(size=n)
    row = []
    for pr in ["indep", "shared", "product"]:
        row += summ(joint_gibbs(y, t, X, pr))
    mhat = LassoCV(cv=5).fit(X, t).predict(X)
    row += summ(joint_gibbs(y, t, X, "indep", extra=mhat))
    resA.append(row); print("A", r, np.round(row[::3], 3), round(time.time() - t0), flush=True)
resA = np.array(resA)
names = ["independent", "shared", "product", "propensity covariate"]
R["A"] = {nm: dict(bias=float(resA[:, 3 * i].mean()), rmse=float(np.sqrt((resA[:, 3 * i] ** 2).mean())), cover=float(resA[:, 3 * i + 1].mean()), width=float(resA[:, 3 * i + 2].mean())) for i, nm in enumerate(names)}
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.boxplot([resA[:, 3 * i] for i in range(4)], tick_labels=["independent", "shared", "product", "propensity\ncovariate"])
ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("Posterior mean minus truth")
fig.tight_layout(); fig.savefig(FIG + "open_product.pdf"); plt.close()

# B. Problem 1: coverage and efficiency of the coupled horseshoe as n grows
resB = {}
for n in [100, 200, 400, 800]:
    rows = []
    for r in range(20):
        p = 50; X = rng.normal(size=(n, p)); g = np.zeros(p); g[:5] = 1.0; b = np.zeros(p); b[:5] = 1.0
        t = X @ g + rng.normal(size=n); y = t + X @ b + rng.normal(size=n)
        mhat = LassoCV(cv=5).fit(X, t).predict(X)
        d = joint_gibbs(y, t, X, "indep", extra=mhat, iters=1200, burn=400)
        se_or = 1 / np.sqrt(n)            # oracle efficient sd: sigma / sqrt(n Var(nu)) with sigma = Var(nu) = 1
        th, se = dml(y, t, X, seed=r)
        rows.append([d.mean() - 1, float(np.quantile(d, .025) <= 1 <= np.quantile(d, .975)), d.std() / se_or, th - 1, float(abs(th - 1) <= 1.96 * se), se / se_or])
    rows = np.array(rows); resB[n] = rows.mean(0).tolist() + [float(np.sqrt((rows[:, 0] ** 2).mean())), float(np.sqrt((rows[:, 3] ** 2).mean()))]
    print("B", n, np.round(resB[n], 3), round(time.time() - t0), flush=True)
R["B"] = resB

# C. Problem 8: heavy tailed errors
resC = []
for r in range(30):
    n, p = 300, 30; X = rng.normal(size=(n, p)); g = np.zeros(p); g[:5] = 1.0; b = np.zeros(p); b[:5] = 1.0
    t = X @ g + rng.normal(size=n); y = t + X @ b + stats.t.rvs(2, size=n, random_state=rng)
    mhat = LassoCV(cv=5).fit(X, t).predict(X)
    dg = joint_gibbs(y, t, X, "indep", extra=mhat, iters=1500, burn=500)
    dt = joint_gibbs(y, t, X, "indep", extra=mhat, iters=1500, burn=500, tdf=3)
    th, se = dml(y, t, X, seed=r)
    resC.append(summ(dg) + summ(dt) + [th - 1, float(abs(th - 1) <= 1.96 * se), 3.92 * se])
    print("C", r, np.round(np.array(resC[-1])[::3], 3), round(time.time() - t0), flush=True)
resC = np.array(resC)
R["C"] = {nm: dict(bias=float(resC[:, 3 * i].mean()), rmse=float(np.sqrt((resC[:, 3 * i] ** 2).mean())), cover=float(resC[:, 3 * i + 1].mean()), width=float(resC[:, 3 * i + 2].mean())) for i, nm in enumerate(["gaussian", "student", "dml"])}
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.boxplot([resC[:, 0], resC[:, 3], resC[:, 6]], tick_labels=["horseshoe, Gaussian", "horseshoe, Student t", "DML"])
ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("Estimate minus truth")
fig.tight_layout(); fig.savefig(FIG + "open_heavy.pdf"); plt.close()

# D. Problem 7: twins narrow the coupling
# Y(t) = mu_t + h + e_t, Var(h) = Vh shared by twins, Var(e_t) = 1 - Vh, sigma_t = 1; delta = mu1 - mu0 = 0.5
shares = np.linspace(0, 0.95, 60); delta = 0.5
lo_r = (shares - (1 - shares)); hi_r = np.ones_like(shares)   # rho in [Vh - s0 s1, Vh + s0 s1] with s0 s1 = 1 - Vh
helped = lambda rho: stats.norm.cdf(delta / np.sqrt(np.maximum(2 * (1 - rho), 1e-9)))
fig, ax = plt.subplots(figsize=(5, 3.1))
ax.fill_between(shares, helped(np.maximum(lo_r, -1)), helped(np.minimum(hi_r, 0.999999)), alpha=0.3, label="twin trial, Gaussian copula range")
ax.axhline(helped(-1.0), color="grey", ls=":", lw=0.8); ax.axhline(1.0, color="grey", ls=":", lw=0.8)
ax.set_xlabel("Share of outcome variance shared by twins"); ax.set_ylabel("P(Y(1) > Y(0))")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIG + "open_twins.pdf"); plt.close()
R["D"] = {"share_0.5": [float(helped(0.0)), 1.0], "share_0.8": [float(helped(0.6)), 1.0], "unpaired": [float(helped(-1.0)), 1.0]}
json.dump(R, open("results/res_open.json", "w"), indent=1, default=float); print(json.dumps(R, default=float))
