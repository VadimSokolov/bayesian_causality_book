import numpy as np, json, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate, stats
from sklearn.linear_model import LassoCV, LogisticRegressionCV, LinearRegression
from sklearn.model_selection import KFold

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
OUT = "figs/"
R = {}
expit = lambda z: 1 / (1 + np.exp(-z))
logit = lambda p: np.log(p / (1 - p))

# 1. Noncollapsibility of the odds ratio
def marg_or(cond_or, spread):
    a = np.array([-spread, spread])
    p0 = expit(a); p1 = expit(a + np.log(cond_or))
    m0, m1 = p0.mean(), p1.mean()
    return (m1 / (1 - m1)) / (m0 / (1 - m0))
spreads = np.linspace(0, 4, 100)
fig, ax = plt.subplots(figsize=(5, 3.2))
# Gray scale print: one line style and gray level per odds ratio (the dotted gray lines mark the conditional odds ratios)
for c, ls, col in zip([2, 4, 8], ["-", "--", "-."], ["k", "0.3", "0.55"]):
    ax.plot(spreads, [marg_or(c, s) for s in spreads], ls, color=col, lw=1.6, label=f"conditional OR = {c}")
    ax.axhline(c, color="grey", lw=0.5, ls=":")
ax.set_xlabel("Spread of baseline log odds across two strata")
ax.set_ylabel("Marginal odds ratio")
ax.legend(frameon=False, handlelength=3)
fig.tight_layout(); fig.savefig(OUT + "noncollapse.pdf"); plt.close()

# 2. Milk style selection: teachers move small children into treatment
rng = np.random.default_rng(1930)
n = 2000
size = rng.normal(0, 1, n)
gain0 = 1.0 - 0.3 * size + rng.normal(0, 0.5, n)   # small children grow more
true_effect = 0.2
p_treat = expit(-1.2 * size)
t = rng.binomial(1, p_treat)
gain = gain0 + true_effect * t
naive = gain[t == 1].mean() - gain[t == 0].mean()
X = np.column_stack([t, size])
adj = LinearRegression().fit(X, gain).coef_[0]
t_r = rng.binomial(1, 0.5, n)
rand = (gain0 + true_effect * t_r)[t_r == 1].mean() - (gain0 + true_effect * t_r)[t_r == 0].mean()
R["milk"] = dict(naive=float(naive), adjusted=float(adj), randomized=float(rand), truth=true_effect)
fig, axs = plt.subplots(1, 2, figsize=(6, 2.8))
axs[0].hist(size[t == 1], bins=30, alpha=0.6, density=True, label="milk")
axs[0].hist(size[t == 0], bins=30, alpha=0.6, density=True, label="control")
axs[0].set_xlabel("Initial size (standardised)"); axs[0].legend(frameon=False)
axs[1].bar(["naive", "model", "randomized"], [naive, adj, rand], color=["#b55", "#58a", "#5a5"])
axs[1].axhline(true_effect, color="k", ls="--", lw=0.8)
axs[1].set_ylabel("Estimated effect on gain")
fig.tight_layout(); fig.savefig(OUT + "milk_sim.pdf"); plt.close()

# 3. Shrinkage profiles for the normal means problem
def hs_postmean(y):
    # horseshoe: kappa ~ Beta(1/2,1/2); y | kappa ~ N(0, 1/kappa)
    num = integrate.quad(lambda k: (1 - k) * np.sqrt(k) * np.exp(-k * y * y / 2) * k ** -0.5 * (1 - k) ** -0.5, 0, 1, limit=200)[0]
    den = integrate.quad(lambda k: np.sqrt(k) * np.exp(-k * y * y / 2) * k ** -0.5 * (1 - k) ** -0.5, 0, 1, limit=200)[0]
    return y * num / den
ys = np.linspace(0, 8, 120)
hs = np.array([hs_postmean(y) for y in ys])
ridge = ys / 2
lasso = np.maximum(ys - 1.5, 0)
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot(ys, ys, color="grey", lw=0.6, ls=":", label="no shrinkage")
# Gray scale print: one line style and gray level per curve (the horseshoe is solid black)
ax.plot(ys, hs, "-", color="k", lw=1.8, label="horseshoe")
ax.plot(ys, ridge, "--", color="0.3", lw=1.6, label="ridge (normal prior)")
ax.plot(ys, lasso, "-.", color="0.55", lw=1.6, label="lasso, soft threshold")
ax.set_xlabel("Observation y"); ax.set_ylabel("Estimate of mean")
ax.legend(frameon=False, handlelength=3)
fig.tight_layout(); fig.savefig(OUT + "shrink_profiles.pdf"); plt.close()
R["hs_profile"] = {str(y): float(hs_postmean(y)) for y in [1, 2, 4, 6, 8]}
import sys
if "--figs-only" in sys.argv:    # redraw noncollapse.pdf and shrink_profiles.pdf only (analytic, no random numbers); the long simulations below are skipped, results/results.json is not written, and the profile is compared with the saved one
    saved = json.load(open("results/results.json")).get("hs_profile", {})
    print("hs_profile: largest difference from results/results.json:", max(abs(R["hs_profile"][k] - saved[k]) for k in saved) if saved else "no saved value"); sys.exit(0)

# 4. Inverse weighting variance under weakening overlap
ns = 1000
strengths = np.linspace(0.5, 4, 8)
sds = []
maxw = []
for s in strengths:
    est = []
    mw = []
    for r in range(300):
        x = rng.normal(0, 1, ns)
        e = expit(s * x)
        tt = rng.binomial(1, e)
        y = x + 1.0 * tt + rng.normal(0, 1, ns)
        est.append(np.mean(tt * y / e - (1 - tt) * y / (1 - e)))
        mw.append(np.max(np.where(tt == 1, 1 / e, 1 / (1 - e))))
    sds.append(np.std(est)); maxw.append(np.median(mw))
R["ipw"] = dict(strengths=strengths.tolist(), sd=list(map(float, sds)), maxw=list(map(float, maxw)))
# Drawn at 6.4 in, like the other figures of the book, so that it prints at about 0.8 of its size; gray scale styles.
# Right panel: the replication sd against the exact sd {4 + exp(s^2/2)(2s^2 - 2s + 5)}/n (text and Exercise ex:ch05:3), log scale.
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9))
axs[0].plot(strengths, maxw, marker="o", color="0.3")
axs[0].set_xlabel("Logit slope of selection"); axs[0].set_ylabel("Median largest weight")
exact_sd = np.sqrt((4 + np.exp(strengths ** 2 / 2) * (2 * strengths ** 2 - 2 * strengths + 5)) / ns)
axs[1].plot(strengths, sds, marker="o", color="0.3", label="sd of 300 estimates")
axs[1].plot(strengths, exact_sd, color="k", ls="--", label="exact sd")
axs[1].set_yscale("log"); axs[1].legend(frameon=False, fontsize=8)
axs[1].set_xlabel("Logit slope of selection"); axs[1].set_ylabel("Standard deviation")
fig.tight_layout(); fig.savefig(OUT + "ipw_var.pdf"); plt.close()

# 5. Robins Ritov style example: many cells, few observations, known propensity
def rr_once(rng, K=20000, n=2000):
    pi = rng.uniform(0.05, 0.95, K)
    theta = expit(-1 + 3 * (pi - 0.5) * 2 + rng.normal(0, 0.3, K)) # outcome correlated with selection
    psi = theta.mean()
    cells = rng.integers(0, K, n)
    r = rng.binomial(1, pi[cells])
    y = rng.binomial(1, theta[cells])
    ht = np.mean(r * y / pi[cells])
    # independent Beta(1,1) prior per cell: unobserved cells get 0.5
    est_cell = np.full(K, 0.5)
    obs = cells[r == 1]; yo = y[r == 1]
    s = np.bincount(obs, weights=yo, minlength=K); c = np.bincount(obs, minlength=K)
    est_cell = (1 + s) / (2 + c)
    naive_bayes = est_cell.mean()
    # coupled prior: logit theta = a + b logit pi + noise; fit a,b by logistic regression on observed
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6).fit(logit(pi[obs]).reshape(-1, 1), yo)
    prior_mean = lr.predict_proba(logit(pi).reshape(-1, 1))[:, 1]
    m = 10.0
    coupled = ((m * prior_mean + s) / (m + c)).mean()
    return psi, ht, naive_bayes, coupled
res = np.array([rr_once(rng) for _ in range(200)])
err = res[:, 1:] - res[:, [0]]
R["rr"] = dict(bias=err.mean(0).tolist(), rmse=np.sqrt((err ** 2).mean(0)).tolist())
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.boxplot([err[:, 0], err[:, 1], err[:, 2]], tick_labels=["Horvitz Thompson", "independent\npriors", "prior coupled\nto $\\pi(x)$"])
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("Estimate minus truth")
fig.tight_layout(); fig.savefig(OUT + "robins_ritov.pdf"); plt.close()

# 6. SUPERSEDED by ric_sim.py (500 replications, seven estimators, Chapter 8). Kept only so that the random
# stream of the sections below is unchanged; it no longer writes ric.pdf, which ric_sim.py makes.
def hs_gibbs(y, X, unpen, iters=1500, burn=500, rng=None):
    n, p = X.shape
    pen = np.array([j not in unpen for j in range(p)])
    lam2 = np.ones(p); nu = np.ones(p); tau2 = 1.0; xi = 1.0; sig2 = 1.0
    XtX = X.T @ X; Xty = X.T @ y
    draws = []
    for it in range(iters):
        prec = np.where(pen, 1 / (lam2 * tau2), 1e-6)
        A = XtX + np.diag(prec)
        L = np.linalg.cholesky(A)
        mu = np.linalg.solve(A, Xty)
        b = mu + np.sqrt(sig2) * np.linalg.solve(L.T, rng.normal(size=p))
        res = y - X @ b
        bp = b[pen]
        sig2 = 1 / rng.gamma((n + pen.sum()) / 2, 1 / (res @ res / 2 + (bp ** 2 / (lam2[pen] * tau2)).sum() / 2))
        lam2[pen] = 1 / rng.gamma(1, 1 / (1 / nu[pen] + bp ** 2 / (2 * tau2 * sig2)))
        nu[pen] = 1 / rng.gamma(1, 1 / (1 + 1 / lam2[pen]))
        tau2 = 1 / rng.gamma((pen.sum() + 1) / 2, 1 / (1 / xi + (bp ** 2 / lam2[pen]).sum() / (2 * sig2)))
        xi = 1 / rng.gamma(1, 1 / (1 + 1 / tau2))
        if it >= burn:
            draws.append(b.copy())
    return np.array(draws)

def ric_once(rng, n=200, p=50, tau=1.0):
    X = rng.normal(size=(n, p))
    g = np.zeros(p); g[:5] = 1.0
    b = np.zeros(p); b[:5] = 1.0
    t = X @ g + rng.normal(size=n)
    y = tau * t + X @ b + rng.normal(size=n)
    out = {}
    # lasso plug in: lasso on y ~ t + X, read off coefficient of t
    Z = np.column_stack([t, X])
    out["lasso"] = LassoCV(cv=5).fit(Z, y).coef_[0]
    # DML partially linear with cross fitting
    kf = KFold(5, shuffle=True, random_state=0)
    ry = np.zeros(n); rt = np.zeros(n)
    for tr, te in kf.split(X):
        ry[te] = y[te] - LassoCV(cv=5).fit(X[tr], y[tr]).predict(X[te])
        rt[te] = t[te] - LassoCV(cv=5).fit(X[tr], t[tr]).predict(X[te])
    out["dml"] = (rt @ ry) / (rt @ rt)
    t0 = time.time()
    d = hs_gibbs(y, Z, unpen={0}, rng=rng)
    out["hs_time"] = time.time() - t0
    out["hs"] = d[:, 0].mean()
    that = LassoCV(cv=5).fit(X, t).predict(X)
    Z2 = np.column_stack([t, that, X])
    d2 = hs_gibbs(y, Z2, unpen={0, 1}, rng=rng)
    out["hs_coupled"] = d2[:, 0].mean()
    lo, hi = np.quantile(d2[:, 0], [0.025, 0.975])
    out["cover"] = float(lo <= tau <= hi)
    return out

reps = [ric_once(rng) for _ in range(40)]
keys = ["lasso", "dml", "hs", "hs_coupled"]
E = np.array([[r[k] for k in keys] for r in reps]) - 1.0
R["ric_40rep_superseded"] = dict(bias=E.mean(0).tolist(), rmse=np.sqrt((E ** 2).mean(0)).tolist(),
                cover=float(np.mean([r["cover"] for r in reps])),
                hs_time=float(np.mean([r["hs_time"] for r in reps])))
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.boxplot([E[:, i] for i in range(4)], tick_labels=["lasso plug in", "DML", "horseshoe", "horseshoe + e(x)"])
ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("Estimate minus truth")
plt.close()

# 7. Quantile treatment effects with Bayesian bootstrap bands
n = 1500
tq = rng.binomial(1, 0.5, n)
y0 = rng.lognormal(0, 0.5, n)
yq = np.where(tq == 1, y0 * (1 + 0.3 * (y0 > np.median(y0))), y0)
qs = np.linspace(0.05, 0.95, 37)
def wq(v, w, q):
    o = np.argsort(v); cw = np.cumsum(w[o]); cw /= cw[-1]
    return np.interp(q, cw, v[o])
bb = []
for s in range(400):
    w = rng.exponential(size=n)
    bb.append(wq(yq[tq == 1], w[tq == 1], qs) - wq(yq[tq == 0], w[tq == 0], qs))
bb = np.array(bb)
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.fill_between(qs, *np.quantile(bb, [0.025, 0.975], axis=0), alpha=0.3)
ax.plot(qs, bb.mean(0))
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Quantile level"); ax.set_ylabel("Quantile treatment effect")
fig.tight_layout(); fig.savefig(OUT + "qte.pdf"); plt.close()

# 8. Frechet Hoeffding bounds on P(Y1 > Y0) for normal marginals
deltas = np.linspace(-2, 2, 81)
lo, hi = [], []
grid = np.linspace(-8, 8, 4001)
for dlt in deltas:
    # P(Y1 - Y0 > 0) bounds via Makarov: sup/inf over y of F0(y) - F1(y)
    F0 = stats.norm.cdf(grid); F1 = stats.norm.cdf(grid - dlt)
    lo.append(max(0, np.max(F0 - F1)))
    hi.append(1 + min(0, np.min(F0 - F1)))
comon = (deltas > 0).astype(float)    # 0 for shifts up to zero, 1 above: the jump sits at zero, with no plateau of 0.5
indep = stats.norm.cdf(deltas / np.sqrt(2))
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.fill_between(deltas, lo, hi, alpha=0.25, label="Makarov bounds")
ax.plot(deltas, indep, label="independent coupling")
ax.step(deltas, comon, where="pre", label="comonotone coupling")
ax.set_xlabel("Location shift of treated marginal"); ax.set_ylabel("P(Y(1) > Y(0))")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "makarov.pdf"); plt.close()

# 9. Tian Pearl bounds and a posterior within them
p_yx = 0.30; p_yxp = 0.10
lb = max(0, p_yx - p_yxp); ub = min(p_yx, 1 - p_yxp)
# posterior for experimental arms with 500 per arm, then PNS = lb + u (ub - lb), u ~ Beta prior
a1 = rng.beta(1 + 150, 1 + 350, 20000); a0 = rng.beta(1 + 50, 1 + 450, 20000)
LB = np.maximum(0, a1 - a0); UB = np.minimum(a1, 1 - a0)
fig, ax = plt.subplots(figsize=(5, 3.2))
for (pa, pb, lab) in [(1, 1, "uniform within bounds"), (0.5, 0.5, "horseshoe type Beta(1/2,1/2)")]:
    u = rng.beta(pa, pb, 20000)
    ax.hist(LB + u * (UB - LB), bins=80, density=True, histtype="step", label=lab)
ax.axvline(lb, color="k", ls=":"); ax.axvline(ub, color="k", ls=":")
ax.set_xlabel("PNS"); ax.set_ylabel("Posterior density"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "pns.pdf"); plt.close()
R["pns"] = dict(lb=lb, ub=ub)

# 10. Cornfield: required prevalence ratio and E value
rr = np.linspace(1, 15, 200)
evalue = rr + np.sqrt(rr * (rr - 1))
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot(rr, rr, label="Cornfield: prevalence ratio of hidden factor")
ax.plot(rr, evalue, label="E value: confounder risk ratios")
ax.set_xlabel("Observed risk ratio"); ax.set_ylabel("Required strength of confounding")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "cornfield.pdf"); plt.close()

# 11. Front door simulation with a hidden genotype
def fd_once(rng, n=20000):
    u = rng.binomial(1, 0.3, n)
    s = rng.binomial(1, expit(-0.5 + 2.0 * u))
    tar = rng.binomial(1, expit(-2 + 4 * s))
    c = rng.binomial(1, expit(-3 + 1.5 * tar + 1.5 * u))
    # truth by intervention
    def do(sv):
        tr = rng.binomial(1, expit(-2 + 4 * sv), n)
        return expit(-3 + 1.5 * tr + 1.5 * u).mean()
    truth_mc = do(1) - do(0)   # sample genotypes with fresh tar draws; kept only so the random stream, and so the data, are unchanged
    naive = c[s == 1].mean() - c[s == 0].mean()
    def fd(sv):
        tot = 0
        for m in [0, 1]:
            pm = np.mean(tar[s == sv] == m)
            inner = sum(np.mean(c[(tar == m) & (s == xp)]) * np.mean(s == xp) for xp in [0, 1])
            tot += pm * inner
        return tot
    return truth_mc, naive, fd(1) - fd(0)
fdr = np.array([fd_once(rng) for _ in range(100)])
FD_TRUTH = json.load(open("results/res_frontdoor_dgp.json"))["risk_difference"]   # exact population value, 0.14453
R["frontdoor"] = dict(truth=float(FD_TRUTH), truth_mc_mean_before_r1023=float(fdr[:, 0].mean()), naive=float(fdr[:, 1].mean()), fd=float(fdr[:, 2].mean()),
                      naive_minus_truth_sd=float((fdr[:, 1] - FD_TRUTH).std(ddof=1)), fd_minus_truth_sd=float((fdr[:, 2] - FD_TRUTH).std(ddof=1)),
                      fd_minus_truth_mean=float((fdr[:, 2] - FD_TRUTH).mean()), reps=100)
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.boxplot([fdr[:, 1] - FD_TRUTH, fdr[:, 2] - FD_TRUTH], tick_labels=["smoker vs non smoker contrast", "front door formula"])
ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("Estimate minus exact risk difference")
fig.tight_layout(); fig.savefig(OUT + "frontdoor.pdf"); plt.close()

# 12. Instruments: OLS vs IV across replications
ols, iv = [], []
for r in range(500):
    z = rng.normal(size=500); u = rng.normal(size=500)
    x = 0.5 * z + u + rng.normal(size=500)
    y = 1.0 * x + 2 * u + rng.normal(size=500)
    ols.append((x @ y) / (x @ x)); iv.append((z @ y) / (z @ x))
R["iv"] = dict(ols_mean=float(np.mean(ols)), iv_median=float(np.median(iv)))
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.hist(ols, bins=40, alpha=0.6, density=True, label="least squares")
ax.hist(np.clip(iv, -1, 3), bins=60, alpha=0.6, density=True, label="instrumental variable")
ax.axvline(1, color="k", ls="--", lw=0.8); ax.set_xlabel("Estimated effect"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT + "iv.pdf"); plt.close()

json.dump(R, open("results/results.json", "w"), indent=1)
print(json.dumps(R, indent=1))
