import numpy as np, json, time, warnings, sys
warnings.filterwarnings("ignore")
from polyagamma import random_polyagamma
from sklearn.linear_model import LassoCV, LogisticRegressionCV, LinearRegression
from sklearn.model_selection import StratifiedKFold
rng = np.random.default_rng(2026); t0 = time.time()
expit = lambda z: 1 / (1 + np.exp(-z))
IG = lambda a, b: 1 / rng.gamma(a, 1 / b)

def pg_hs_logit(X, d, sweeps=600, burn=200):
    """Horseshoe logistic regression by Polya Gamma Gibbs; intercept unpenalized. Returns posterior mean of beta draws."""
    n, p = X.shape; Z = np.column_stack([np.ones(n), X]); q = p + 1
    b = np.zeros(q); lam2 = np.ones(p); nu = np.ones(p); tau2 = 1.0; xi = 1.0; kap = d - 0.5; out = []
    for s in range(sweeps):
        om = random_polyagamma(1, Z @ b, random_state=rng)
        prec = np.r_[1e-4, 1 / (lam2 * tau2)]
        A = Z.T @ (Z * om[:, None]) + np.diag(prec); L = np.linalg.cholesky(A)
        b = np.linalg.solve(A, Z.T @ kap) + np.linalg.solve(L.T, rng.normal(size=q))
        bb = b[1:]
        lam2 = IG(1, 1 / nu + bb ** 2 / (2 * tau2)); nu = IG(1, 1 + 1 / lam2)
        tau2 = IG((p + 1) / 2, 1 / xi + (bb ** 2 / lam2).sum() / 2); xi = IG(1, 1 + 1 / tau2)
        if s >= burn: out.append(b.copy())
    return np.array(out)

def hs_lin(X, y, sweeps=600, burn=200):
    n, p = X.shape; Z = np.column_stack([np.ones(n), X]); q = p + 1
    ZtZ = Z.T @ Z; Zty = Z.T @ y; yty = y @ y
    lam2 = np.ones(p); nu = np.ones(p); tau2 = 1.0; xi = 1.0; s2 = np.var(y); out = []
    for s in range(sweeps):
        A = ZtZ + np.diag(np.r_[1e-8, 1 / (lam2 * tau2)]); L = np.linalg.cholesky(A)
        b = np.linalg.solve(A, Zty) + np.sqrt(s2) * np.linalg.solve(L.T, rng.normal(size=q))
        bb = b[1:]; rss = yty - 2 * b @ Zty + b @ ZtZ @ b
        s2 = IG((n + p) / 2, rss / 2 + (bb ** 2 / (lam2 * tau2)).sum() / 2)
        lam2 = IG(1, 1 / nu + bb ** 2 / (2 * tau2 * s2)); nu = IG(1, 1 + 1 / lam2)
        tau2 = IG((p + 1) / 2, 1 / xi + (bb ** 2 / lam2).sum() / (2 * s2)); xi = IG(1, 1 + 1 / tau2)
        if s >= burn: out.append(b.copy())
    return np.array(out).mean(0)

def aipw(X, D, Y, method, K=5, seed=0):
    n = len(Y); e = np.zeros(n); m1 = np.zeros(n); m0 = np.zeros(n)
    for tr, te in StratifiedKFold(K, shuffle=True, random_state=seed).split(X, D):
        if method == "lasso":
            e[te] = LogisticRegressionCV(Cs=10, cv=5, penalty="l1", solver="saga", max_iter=3000, random_state=seed).fit(X[tr], D[tr]).predict_proba(X[te])[:, 1]
            for t, arr in [(1, m1), (0, m0)]:
                s = tr[D[tr] == t]; arr[te] = LassoCV(cv=5).fit(X[s], Y[s]).predict(X[te])
        else:
            B = pg_hs_logit(X[tr], D[tr]); Zte = np.column_stack([np.ones(len(te)), X[te]])
            e[te] = expit(Zte @ B.T).mean(1)
            for t, arr in [(1, m1), (0, m0)]:
                s = tr[D[tr] == t]; b = hs_lin(X[s], Y[s]); arr[te] = Zte @ b
    w = np.where(D == 1, 1 / e, 1 / (1 - e))
    psi = m1 - m0 + D * (Y - m1) / e - (1 - D) * (Y - m0) / (1 - e)
    tau = psi.mean(); se = psi.std() / np.sqrt(n)
    ess = w.sum() ** 2 / (w ** 2).sum()
    return tau, se, float(np.quantile(w, 0.99)), float(ess), float(e.min()), float(e.max())

def design(kind, n, p=100):
    k = {"I": 5, "II": 30, "III": 10}[kind]
    X = rng.normal(size=(n, p)); s = X[:, :k].sum(1) / np.sqrt(k)
    lin = 2 * s if kind == "III" else s
    e0 = expit(lin); D = rng.binomial(1, e0).astype(float)
    Y = 1.0 * D + s + rng.normal(size=n)
    return X, D, Y, e0

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200   # the book prints 200 replications per design (about 85 minutes on four Intel cores)
R = {}
for kind, n in [("I", 500), ("II", 500), ("III", 1000)]:
    rows = []
    for r in range(reps):
        X, D, Y, e0 = design(kind, n)
        ols = LinearRegression().fit(np.column_stack([D, X]), Y).coef_[0]
        lt = aipw(X, D, Y, "lasso", seed=r); ht = aipw(X, D, Y, "hs", seed=r)
        rows.append([ols - 1, lt[0] - 1, float(abs(lt[0] - 1) <= 1.96 * lt[1]), 3.92 * lt[1], lt[2], lt[3],
                     ht[0] - 1, float(abs(ht[0] - 1) <= 1.96 * ht[1]), 3.92 * ht[1], ht[2], ht[3], ht[4], ht[5], float(e0.min()), float(e0.max())])
        print(kind, r, np.round(rows[-1], 3), round(time.time() - t0), flush=True)
    A = np.array(rows)
    R[kind] = dict(n=n, reps=reps,
        ols=dict(bias=float(A[:, 0].mean()), rmse=float(np.sqrt((A[:, 0] ** 2).mean()))),
        lasso=dict(bias=float(A[:, 1].mean()), rmse=float(np.sqrt((A[:, 1] ** 2).mean())), cover=float(A[:, 2].mean()), length=float(A[:, 3].mean()), w99=float(A[:, 4].mean()), ess=float(A[:, 5].mean())),
        hs=dict(bias=float(A[:, 6].mean()), rmse=float(np.sqrt((A[:, 6] ** 2).mean())), cover=float(A[:, 7].mean()), length=float(A[:, 8].mean()), w99=float(A[:, 9].mean()), ess=float(A[:, 10].mean()),
                emin=float(A[:, 11].min()), emax=float(A[:, 12].max())),
        e0min=float(A[:, 13].min()), e0max=float(A[:, 14].max()))
    json.dump(R, open("results/res_baipw.json", "w"), indent=1)

# LaLonde experimental sample, as in the paper's Table 7
from causaldata import nsw_mixtape
d = nsw_mixtape.load_pandas().data
cov = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]
X = d[cov].astype(float).values; X = (X - X.mean(0)) / X.std(0); D = d.treat.values.astype(float); Y = d.re78.values.astype(float)
L = {}
L["diff"] = float(Y[D == 1].mean() - Y[D == 0].mean())
L["diff_se"] = float(np.sqrt(Y[D == 1].var(ddof=1) / (D == 1).sum() + Y[D == 0].var(ddof=1) / (D == 0).sum()))
L["ols"] = float(LinearRegression().fit(np.column_stack([D, X]), Y).coef_[0])
for nm in ["lasso", "hs"]:
    vals = [aipw(X, D, Y, nm, seed=s) for s in range(3)]
    L[nm] = [[float(v[0]), float(v[1]), float(v[2])] for v in vals]
R["lalonde"] = L
json.dump(R, open("results/res_baipw.json", "w"), indent=1); print(json.dumps(R["lalonde"]), "done", round(time.time() - t0))
