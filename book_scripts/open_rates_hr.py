"""Problem 8 (rates under heavy tailed errors), rerun of 23 October 2026 (r1023). Replaces the P8 block of open_sims2.py.

Design. Partially linear model y = tau t + X b + e, t = X g + nu, with tau = 1, p = 30 independent standard normal covariates, five unit
confounders (g_j = b_j = 1 for j < 5, zero otherwise), nu ~ N(0, 1). Errors e: Student t on 1.5 degrees of freedom, Student t on 2 degrees of
freedom (both unscaled), or N(0, 1). n in {100, 200, 400, 800, 1600, 3200, 6400}; 200 replications per cell. Each replication has its own seed,
numpy default_rng([20261023, n, error_index, replication]), and KFold seed = replication, so the result does not depend on how the work is
split into array tasks.
Methods: (1) joint horseshoe regression with a Student t working likelihood on 3 degrees of freedom (the scale mixture sampler of joint_gibbs,
independent scales, the lasso propensity fit as an extra covariate, 1,200 iterations, 400 burn in), posterior mean and 95 percent equal tail interval;
(2) the same with a Gaussian working likelihood; (3) DML with lasso nuisances (5 folds, LassoCV with 5 inner folds), partialling out
estimator with the sandwich standard error and a 95 percent Wald interval; (4) the oracle score estimator sum(nu e) / sum(nu^2) with the true
nuisances (error variables nu = t - X g, e), with the sandwich interval sqrt(sum(nu^2 e^2)) / sum(nu^2) (not a valid interval when the
variance of e is infinite; reported for completeness).
Per replication we save the signed error, the interval coverage and the interval width of each method.
Usage: python book_scripts/open_rates_hr.py TASK   (TASK = 0..209: 7 n values x 3 error types x 10 chunks of 20 replications)
Output: results/rates_hr/part_<TASK>.json; the merge is book_scripts/open_rates_hr_merge.py.
"""
import sys, json, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

NS = [100, 200, 400, 800, 1600, 3200, 6400]; ERR = ["t1.5", "t2", "gauss"]; CHUNK = 20; P = 30
rng = np.random.default_rng(0)
IG = lambda a, b: 1 / rng.gamma(a, 1 / b)

def joint_gibbs(y, t, X, iters=1200, burn=400, tdf=None, extra=None):
    """Joint outcome and treatment regressions with independent horseshoe scales (the 'indep' prior of open_sims2.py).
    Outcome y = tau t + c + X beta + delta * extra + e, treatment t = X gamma + nu. tdf: Student t errors in the outcome by scale mixture."""
    n, p = X.shape
    cols = [t, np.ones(n)] + ([extra] if extra is not None else [])
    k0 = len(cols); Zy = np.column_stack(cols + [X]); q = Zy.shape[1]
    lam2 = np.ones(p); nul = np.ones(p); lt2 = np.ones(p); nut = np.ones(p)
    ty2 = tt2 = 1.0; xy = xt = 1.0; sy2 = np.var(y); st2 = np.var(t); w = np.ones(n)
    XtX = X.T @ X; Xtt = X.T @ t; out = []
    for it in range(iters):
        vb = lam2; vg = lt2
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
        lam2 = IG(1, 1 / nul + Bq); lt2 = IG(1, 1 / nut + Gq); nut = IG(1, 1 + 1 / lt2)
        nul = IG(1, 1 + 1 / lam2)
        vb = lam2; vg = lt2
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

def one(n, ei, r):
    global rng
    rng = np.random.default_rng([20261023, n, ei, r])
    X = rng.normal(size=(n, P)); g = np.zeros(P); g[:5] = 1.0; b = np.zeros(P); b[:5] = 1.0
    nu = rng.normal(size=n); t = X @ g + nu
    e = {0: lambda: stats.t.rvs(1.5, size=n, random_state=rng), 1: lambda: stats.t.rvs(2, size=n, random_state=rng), 2: lambda: rng.normal(size=n)}[ei]()
    y = t + X @ b + e
    mhat = LassoCV(cv=5).fit(X, t).predict(X)
    out = {}
    for name, tdf in [("student", 3), ("gauss_post", None)]:
        d = joint_gibbs(y, t, X, tdf=tdf, extra=mhat); lo, hi = np.quantile(d, [0.025, 0.975])
        out[name] = [d.mean() - 1, float(lo <= 1 <= hi), hi - lo]
    th, se = dml(y, t, X, seed=r)
    out["dml"] = [th - 1, float(abs(th - 1) <= 1.96 * se), 3.92 * se]
    so = (nu * e).sum() / (nu ** 2).sum(); seo = np.sqrt(((nu * e) ** 2).sum()) / (nu ** 2).sum()
    out["oracle"] = [so, float(abs(so) <= 1.96 * seo), 3.92 * seo]
    return out

if __name__ == "__main__":
    task = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"]); ni, rest = divmod(task, 30); ei, ch = divmod(rest, 10); n = NS[ni]
    os.makedirs("results/rates_hr", exist_ok=True); t0 = time.time(); reps = []
    for r in range(ch * CHUNK, (ch + 1) * CHUNK):
        reps.append(one(n, ei, r)); print(task, n, ERR[ei], r, round(time.time() - t0), flush=True)
    json.dump(dict(n=n, err=ERR[ei], chunk=ch, reps=reps, secs=time.time() - t0), open(f"results/rates_hr/part_{task}.json", "w"))
