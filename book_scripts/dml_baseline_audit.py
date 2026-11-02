"""Audit of the baseline simulations of the DML critique (Chapter 4 Section "What DML delivers in practice", Chapter 7 tables).

A port of the simulator of the working paper "Orthogonality is not enough" (Polson and Sokolov 2026b; dml/sim/dmlsim.py there, itself a
reconstruction because the original code was lost), restricted to the lasso arms, with 5,000 replications for each design.
Data generating process: X has n rows of p independent N(0, 1) columns; the first s coordinates are confounders, alpha_j = 1 in the treatment
equation D = X alpha + nu, nu ~ N(0, sigma_nu^2); Y = tau_0 D + X gamma + eps, gamma_j = u_j with u_j ~ U(0.8, 1.2) drawn afresh in every
replication for j <= s, tau_0 = 1, eps ~ N(0, SIGMA_EPS^2) with SIGMA_EPS = 0.948 (the value recovered in the working paper from nine published
interval widths: the original outcome noise standard deviation was never recorded, so 0.948, variance 0.899, is a reconstruction).
r1023b: the dense design (every coordinate of alpha and gamma nonzero, total signal of s = 15 spread over all p, as dmlsim.py), the tau = 0.5 and 2 designs and the mixed sign designs are added to DESIGNS and run by the same script; agg() still summarises the original five.
Designs (n, p, s, sigma_nu): s5 (300, 400, 5, 1), s15 (300, 400, 15, 1), s30 (300, 400, 30, 1), weak (300, 400, 15, 0.3), bigN (1200, 200, 15, 1).
Every replication r of a design uses the seed crc32("design|r"), and all arms below see the same data set:
  dml_K5, dml_K2   cross fitted partially linear score, cross validated lasso (LassoCV, 50 alphas, 5 inner folds, fitted inside the training
                   folds only) for both nuisances, K = 5 and K = 2 folds, one random split, sandwich standard error;
  rs_K5, rs_K2     the repeated split median of Chernozhukov et al. (2018) with S = 20 splits, point estimate the median, standard error the
                   median of sqrt(se_s^2 + (theta_s - median)^2);
  pds_theory       post double selection, lasso penalty at its theoretical level 1.1 sqrt(2 log(2p/0.1)/n) times the scaled lasso noise level
                   (iterated to convergence), least squares on the union of the two supports, HC1 standard error;
  pds_cv           the same with the cross validated penalty.
Per replication output: results/dml_baseline_audit/DESIGN_START_END.jsonl. Summary (bias, empirical sd, mean sandwich se, coverage of nominal 95
percent intervals with binomial standard error, mean width, Monte Carlo standard error of the bias): results/res_dml_baseline_audit.json.
Usage (repository root):  python book_scripts/dml_baseline_audit.py run DESIGN START END     |     python book_scripts/dml_baseline_audit.py agg
Requires numpy and scikit-learn.
"""
import glob, json, math, os, sys, time, warnings, zlib
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import Lasso, LassoCV

SIGMA_EPS = 0.948
Z95 = 1.959963984540054
DESIGNS = {"s5": dict(n=300, p=400, s=5, sigma_nu=1.0), "s15": dict(n=300, p=400, s=15, sigma_nu=1.0), "s30": dict(n=300, p=400, s=30, sigma_nu=1.0),
           "weak": dict(n=300, p=400, s=15, sigma_nu=0.3), "bigN": dict(n=1200, p=200, s=15, sigma_nu=1.0),
           # r1023b: designs added for the dense row and the channel measurements (the first five are the audit's own and unchanged)
           "dense": dict(n=300, p=400, s=15, sigma_nu=1.0, dense=True), "tau05_s15": dict(n=300, p=400, s=15, sigma_nu=1.0, theta0=0.5),
           "tau20_s15": dict(n=300, p=400, s=15, sigma_nu=1.0, theta0=2.0), "mixed_s15": dict(n=300, p=400, s=15, sigma_nu=1.0, gamma_profile="mixed"),
           "mixed_s30": dict(n=300, p=400, s=30, sigma_nu=1.0, gamma_profile="mixed")}
ARMS = ["dml_K5", "dml_K2", "rs_K5", "rs_K2", "pds_theory", "pds_cv"]
OUTD = "results/dml_baseline_audit/"


def make_design_full(rng, n, p, s, sigma_nu, theta0=1.0, dense=False, gamma_profile="template", c=1.0, s_ref=15):
    """Same stream of random numbers as the original make_design for the default arguments (r1023b: also the dense, tau and mixed sign designs
    of dml/sim/dmlsim.py). Returns X, D, Y, alpha, gamma."""
    X = rng.standard_normal((n, p)); alpha = np.zeros(p); gamma = np.zeros(p)
    if dense:
        scale = c * math.sqrt(s_ref / p); alpha[:] = scale; gamma[:] = scale * rng.uniform(0.8, 1.2, p)
    elif gamma_profile == "mixed":
        alpha[:s] = c; sgn = np.where(np.arange(s) % 2 == 0, 1.0, -1.0); gamma[:s] = c * sgn * rng.uniform(0.8, 1.2, s)
    else:
        alpha[:s] = 1.0; gamma[:s] = rng.uniform(0.8, 1.2, s)
    D = X @ alpha + sigma_nu * rng.standard_normal(n)
    Y = theta0 * D + X @ gamma + SIGMA_EPS * rng.standard_normal(n)
    return X, D, Y, alpha, gamma


def make_design(rng, n, p, s, sigma_nu, **kw):
    return make_design_full(rng, n, p, s, sigma_nu, **kw)[:3]


def lasso_fit(Xtr, ytr, rng, theory=False):
    if theory:
        n, p = Xtr.shape; base = math.sqrt(2 * math.log(2 * p / 0.1) / n); sig = float(np.std(ytr)); mod = None
        for _ in range(200):          # scaled lasso, iterated to convergence from the conservative start sd(y)
            mod = Lasso(alpha=1.1 * sig * base, max_iter=20000).fit(Xtr, ytr)
            new = float(np.std(ytr - mod.predict(Xtr))); done = abs(new - sig) < 1e-8 * max(sig, 1e-8); sig = new
            if done: break
    else:
        mod = LassoCV(alphas=50, cv=5, max_iter=20000, random_state=int(rng.integers(2 ** 31))).fit(Xtr, ytr)
    return mod.coef_


def resid(X, D, Y, rng, K):
    n = len(D); idx = rng.permutation(n); m = np.empty(n); l = np.empty(n)
    for f in np.array_split(idx, K):
        tr = np.setdiff1d(idx, f)
        m[f] = X[f] @ lasso_fit(X[tr], D[tr], rng); l[f] = X[f] @ lasso_fit(X[tr], Y[tr], rng)
    return D - m, Y - l


def plr(rD, rY):
    n = len(rD); J = np.mean(rD ** 2); th = float(np.mean(rD * rY) / J); psi = rD * (rY - th * rD)
    return th, float(math.sqrt(np.mean(psi ** 2) / n) / J)


def dml(X, D, Y, rng, K): return plr(*resid(X, D, Y, rng, K))


def rsplit(X, D, Y, rng, K, S=20):
    th = np.empty(S); se = np.empty(S)
    for i in range(S): th[i], se[i] = plr(*resid(X, D, Y, rng, K))
    med = float(np.median(th)); return med, float(np.median(np.sqrt(se ** 2 + (th - med) ** 2)))


def pds(X, D, Y, rng, theory):
    n = len(D); sel = (np.abs(lasso_fit(X, D, rng, theory)) > 1e-10) | (np.abs(lasso_fit(X, Y, rng, theory)) > 1e-10)
    W = np.column_stack([D, X[:, sel]]) if sel.any() else D.reshape(-1, 1); q = W.shape[1]
    XtXi = np.linalg.pinv(W.T @ W); b = XtXi @ (W.T @ Y); r = Y - W @ b
    V = XtXi @ ((W * (r ** 2)[:, None]).T @ W) @ XtXi * (n / max(n - q, 1))
    return float(b[0]), float(math.sqrt(V[0, 0]))


def one_rep(design, r):
    seed = zlib.crc32(("%s|%d" % (design, r)).encode()) % (2 ** 31); X, D, Y = make_design(np.random.default_rng(seed), **DESIGNS[design])
    rec = dict(design=design, rep=r, seed=seed)
    rec["dml_K5"] = dml(X, D, Y, np.random.default_rng(seed + 1), 5); rec["dml_K2"] = dml(X, D, Y, np.random.default_rng(seed + 2), 2)
    rec["rs_K5"] = rsplit(X, D, Y, np.random.default_rng(seed + 5), 5); rec["rs_K2"] = rsplit(X, D, Y, np.random.default_rng(seed + 6), 2)
    rec["pds_theory"] = pds(X, D, Y, np.random.default_rng(seed + 7), True); rec["pds_cv"] = pds(X, D, Y, np.random.default_rng(seed + 8), False)
    return rec


def agg():
    out = {}
    for d in ["s5", "s15", "s30", "weak", "bigN"]:
        recs = [json.loads(l) for f in sorted(glob.glob(OUTD + d + "_*.jsonl")) for l in open(f)]
        if not recs: continue
        out[d] = dict(spec=DESIGNS[d], reps=len(recs), sigma_eps=SIGMA_EPS)
        for a in ARMS:
            th = np.array([r[a][0] for r in recs]); se = np.array([r[a][1] for r in recs]); cov = np.abs(th - 1) <= Z95 * se; c = cov.mean()
            out[d][a] = dict(bias=float(th.mean() - 1), bias_mcse=float(th.std(ddof=1) / math.sqrt(len(th))), sd=float(th.std(ddof=1)), mean_se=float(se.mean()),
                             coverage=float(c), coverage_se=float(math.sqrt(c * (1 - c) / len(th))), width=float(2 * Z95 * se.mean()), rmse=float(np.sqrt(np.mean((th - 1) ** 2))))
        for a, b in [("dml_K5", "dml_K2"), ("rs_K5", "dml_K5")]:   # paired difference in coverage
            ca = np.array([abs(r[a][0] - 1) <= Z95 * r[a][1] for r in recs], float); cb = np.array([abs(r[b][0] - 1) <= Z95 * r[b][1] for r in recs], float)
            out[d]["paired_coverage_diff_%s_minus_%s" % (a, b)] = [float((ca - cb).mean()), float((ca - cb).std(ddof=1) / math.sqrt(len(ca)))]
    json.dump(out, open("results/res_dml_baseline_audit.json", "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        d, a, b = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); os.makedirs(OUTD, exist_ok=True); t0 = time.time()
        with open(OUTD + "%s_%05d_%05d.jsonl" % (d, a, b), "w") as fh:
            for r in range(a, b):
                rec = one_rep(d, r); rec["elapsed"] = round(time.time() - t0, 1); fh.write(json.dumps(rec) + "\n"); fh.flush()
        print("done", d, a, b, round(time.time() - t0))
    else: agg()
