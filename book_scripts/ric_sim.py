"""Regularization induced confounding: seven estimators of a treatment effect (Chapter 8, table tab:ric, figure fig:ric).

Design (from section 6 of make_figs.py, unchanged): n = 200, p = 50 covariates, X ~ N(0, I). Five covariates have unit coefficients in both the treatment equation t = X g + N(0, 1) and the outcome equation y = tau t + X b + N(0, 1), with tau = 1. The data generating process has no intercept.
Estimators of tau, in table order (all but the lasso plug in come with a 95 percent interval):
  ols_all        least squares of y on (t, X) without intercept, coefficient of t, classical t interval with n - p - 1 degrees of freedom (n - 51 in the default design); not defined when n - p - 1 is not positive
  ols_oracle     least squares of y on (t, X[:, :5]), the five true confounders only, classical t interval with n - 6 degrees of freedom (an unattainable benchmark); in the weak30 design the oracle uses the 30 confounders, with n - 31 degrees of freedom
  lasso          coefficient of t in a LassoCV fit of y on (t, X)
  dml            five fold cross fitted partially linear estimator with LassoCV nuisances for y on X and t on X, theta = sum(rt ry) / sum(rt^2), sandwich interval theta +- 1.96 se
  hs             horseshoe regression of y on (t, X) with the coefficient of t unpenalized, posterior mean and central 95 percent posterior interval
  hs_coupled     the same with the LassoCV fitted treatment regression m_hat(X), fitted on the whole sample, added as a second unpenalized covariate (the coupled prior of Chapter 8)
  hs_coupled_cf  the same with m_hat(X) cross fitted: the out of fold LassoCV fit of t on X, on the same KFold as DML
The Gibbs sampler runs 1500 sweeps and discards 500. In one_rep the estimators lasso, dml, hs and hs_coupled are computed first and in the order of the four estimator version of this script, and the other three after them, so that adding estimators left the random stream, and therefore every earlier estimate, unchanged.

Mechanism diagnostics (added later, no random numbers drawn, so every estimate is unchanged): each replication file also holds "diag" with, for hs_coupled and hs_coupled_cf, eps_term and leak_term (the two terms of eq:pcbias), corr_Ttil_Xg and l1_conf, l1_all; --aggregate adds their means as "mechanism" to the results JSON.

The named design --design weak30 (n = 200, p = 150, 30 weak confounders; the oracle uses the 30 confounders with n - 31 degrees of freedom) writes results/ric_weak30/, results/res_ric_weak30.json, book_tex/chapters/ric_weak30_table.tex (label tab:ric_weak30) and misc/review-loop/newch/ric_weak30_table.md.

Other designs (options --n INT, --p INT and --tag STR, all optional; without them the script is the one described above, with the same data, seeds and files): the same data generating process, estimators, penalties and sampler with the sample size n and the number of covariates p of the command line, for example the high dimensional designs --n 200 --p 150 --tag hd150 and --n 200 --p 300 --tag hd300 (p close to and above n). The data of replication N come from the same seed as in the default design, np.random.default_rng(20261013 + N), and X = rng.normal(size=(n, p)) has the new shape, so a design with another p is a fresh draw of the covariates and not an extension of the p = 50 data. A design other than n = 200, p = 50 needs --tag, and with a tag the files are
  results/ric_STR/rep_N.json, results/res_ric_STR.json, figs/ric_STR.pdf, book_tex/chapters/ric_STR_table.tex (table label tab:ric_STR)
so that the default files are never touched. Least squares on all the covariates needs n - p - 1 > 0: where it is not defined (p = 300 with n = 200) its estimate, interval and standard error are null (JSON has no NaN) in the replication file, its row of the table reads n/a, it is left out of the figure and its paired comparisons are skipped. The tagged designs also report the paired comparisons hs minus dml and lasso minus dml, after the eight of the default design.

Usage (paths are resolved from the location of this file, so the working directory does not matter):
  python3 -I ric_sim.py --rep N      one replication, seed np.random.default_rng(20261013 + N) and KFold(5, shuffle=True, random_state=N), writes results/ric/rep_N.json (skipped when the file exists, --force overwrites)
  python3 -I ric_sim.py --aggregate  reads every results/ric/rep_*.json, writes results/res_ric.json, figs/ric.pdf and book_tex/chapters/ric_table.tex
  python3 -I ric_sim.py --rep N --n 200 --p 150 --tag hd150     and     python3 -I ric_sim.py --aggregate --tag hd150     the same for the design tagged hd150 (--aggregate takes n and p from the replication files and checks --n and --p against them if they are given)
Run the replications N = 1, ..., 500 in parallel, for example: OMP_NUM_THREADS=1 seq 1 500 | xargs -P 4 -I{} python3 -I book_scripts/ric_sim.py --rep {}
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, json, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

N, P, TAU = 200, 50, 1.0
SEED0 = 20261013
# table and figure order
KEYS = ["ols_all", "ols_oracle", "lasso", "dml", "hs", "hs_coupled", "hs_coupled_cf"]
INTERVAL_KEYS = ["ols_all", "ols_oracle", "dml", "hs", "hs_coupled", "hs_coupled_cf"]
ROW_NAMES = {"ols_all": "Least squares, all covariates",
             "ols_oracle": "Least squares, five true confounders (oracle)",
             "lasso": "Lasso plug in",
             "dml": "DML, lasso nuisances",
             "hs": "Horseshoe, independent prior",
             "hs_coupled": "Horseshoe, coupled to $\\hat m(X)$",
             "hs_coupled_cf": "Horseshoe, coupled to cross fitted $\\hat m(X)$"}
ROW_NAMES_WEAK30 = dict(ROW_NAMES, ols_oracle="Least squares, 30 true confounders (oracle)")
FIG_LABELS = ["least squares,\nall covariates", "least squares,\noracle", "lasso plug in", "DML",
              "horseshoe,\nindependent prior", "horseshoe, coupled\nto $\\hat m(X)$", "horseshoe, coupled to\ncross fitted $\\hat m(X)$"]
INTERVALS = {"ols_all": "classical t interval, n - 51 degrees of freedom", "ols_oracle": "classical t interval, n - 6 degrees of freedom",
             "dml": "sandwich interval, theta +- 1.96 se", "hs": "central 95 percent posterior interval",
             "hs_coupled": "central 95 percent posterior interval", "hs_coupled_cf": "central 95 percent posterior interval"}
# paired comparisons (a, b): squared error of a minus squared error of b, replication by replication
PAIRS = [("hs_coupled", "dml"), ("hs_coupled_cf", "dml"), ("hs_coupled_cf", "hs_coupled"), ("hs_coupled", "hs"),
         ("hs_coupled_cf", "hs"), ("ols_all", "dml"), ("ols_all", "hs_coupled"), ("ols_oracle", "dml")]
# added, after those, for the tagged designs only (the default results keep the eight above)
PAIRS_TAGGED_EXTRA = [("hs", "dml"), ("lasso", "dml")]


# Design weak30 (--design weak30): many weak confounders, the DGP of the new draft chapter (n = 200, p = 150): 30 confounders with coefficient 0.3 in both
# the treatment and the outcome equation, and four further covariates (columns 30 to 33) with coefficient 1 in the outcome equation only. Oracle: least
# squares on t and the 30 confounders (n - 31 degrees of freedom). Everything else (estimators, priors, sampler, seeds, KFold) is as in the default design.
WEAK30 = dict(n=200, p=150, tag="weak30", n_conf=30, gamma=0.3, beta=0.3, n_outcome_only=4, beta_outcome_only=1.0)
WEAK30_MD = ROOT / "misc" / "review-loop" / "newch" / "ric_weak30_table.md"


def design_paths(tag=None):
    """The files of a design: the default ones without a tag, results/ric_<tag>/ and so on with one."""
    if tag == "weak30":
        return dict(rep_dir=ROOT / "results" / "ric_weak30", res_json=ROOT / "results" / "res_ric_weak30.json", fig_pdf=None,
                    table_tex=ROOT / "book_tex" / "chapters" / "ric_weak30_table.tex", md=WEAK30_MD, label="tab:ric_weak30")
    if tag is None:
        return dict(rep_dir=ROOT / "results" / "ric", res_json=ROOT / "results" / "res_ric.json", fig_pdf=ROOT / "figs" / "ric.pdf",
                    table_tex=ROOT / "book_tex" / "chapters" / "ric_table.tex", label="tab:ric")
    return dict(rep_dir=ROOT / "results" / ("ric_" + tag), res_json=ROOT / "results" / ("res_ric_" + tag + ".json"),
                fig_pdf=ROOT / "figs" / ("ric_" + tag + ".pdf"), table_tex=ROOT / "book_tex" / "chapters" / ("ric_" + tag + "_table.tex"),
                label="tab:ric_" + tag)


# Regularization induced confounding: horseshoe Gibbs sampler, copied from make_figs.py section 6 without change.
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


def write_json(path, obj):
    """Write a JSON file atomically, so that a replication killed by a timeout never leaves a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp%d" % os.getpid())
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def mech_diag(t, mhat, X, b, g, eps, d, k_conf):
    """Mechanism diagnostics of one coupled fit (uses no random numbers). Ttil is the residual of t on the propensity covariate mhat; bhat is the posterior mean of the
    covariate coefficients d[:, 2:]. eps_term = Ttil'eps / Ttil'Ttil and leak_term = Ttil'X(b - bhat) / Ttil'Ttil are the two terms of eq:pcbias; corr_Ttil_Xg is the
    correlation of Ttil with X g; l1_conf and l1_all are the l1 norms of b - bhat over the confounders (first k_conf columns) and over all covariates."""
    Tt = t - mhat * (mhat @ t) / (mhat @ mhat)
    tt = Tt @ Tt
    delta = b - d[:, 2:].mean(axis=0)
    return dict(eps_term=float(Tt @ eps / tt), leak_term=float(Tt @ (X @ delta) / tt), corr_Ttil_Xg=float(np.corrcoef(Tt, X @ g)[0, 1]),
                l1_conf=float(np.abs(delta[:k_conf]).sum()), l1_all=float(np.abs(delta).sum()))


def one_rep(rep, n=N, p=P, tau=TAU, weak30=False):
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import KFold
    from scipy import stats
    t_start = time.time()
    rng = np.random.default_rng(SEED0 + rep)
    X = rng.normal(size=(n, p))
    g = np.zeros(p); g[:5] = 1.0
    b = np.zeros(p); b[:5] = 1.0
    k_conf = 5
    if weak30:
        k_conf = WEAK30["n_conf"]
        g = np.zeros(p); g[:k_conf] = WEAK30["gamma"]
        b = np.zeros(p); b[:k_conf] = WEAK30["beta"]; b[k_conf:k_conf + WEAK30["n_outcome_only"]] = WEAK30["beta_outcome_only"]
    t = X @ g + rng.normal(size=n)
    eps = rng.normal(size=n)
    y = tau * t + X @ b + eps
    est, lo, hi, sd = {}, {}, {}, {}
    # (1) lasso plug in: lasso on y ~ t + X, read off the coefficient of t
    Z = np.column_stack([t, X])
    est["lasso"] = float(LassoCV(cv=5).fit(Z, y).coef_[0])
    # (2) DML partially linear with cross fitting, sandwich interval
    kf = KFold(5, shuffle=True, random_state=rep)
    ry = np.zeros(n); rt = np.zeros(n)
    for tr, te in kf.split(X):
        ry[te] = y[te] - LassoCV(cv=5).fit(X[tr], y[tr]).predict(X[te])
        rt[te] = t[te] - LassoCV(cv=5).fit(X[tr], t[tr]).predict(X[te])
    theta = (rt @ ry) / (rt @ rt)
    psi = rt * (ry - theta * rt)
    se = np.sqrt(np.sum(psi ** 2)) / np.sum(rt ** 2)
    est["dml"] = float(theta)
    lo["dml"], hi["dml"] = float(theta - 1.96 * se), float(theta + 1.96 * se)
    sd["dml"] = float(se)
    # (3) horseshoe, independent prior: the coefficient of t is unpenalized
    t0 = time.time()
    d = hs_gibbs(y, Z, unpen={0}, rng=rng)
    sec_hs = time.time() - t0
    est["hs"] = float(d[:, 0].mean())
    q = np.quantile(d[:, 0], [0.025, 0.975])
    lo["hs"], hi["hs"], sd["hs"] = float(q[0]), float(q[1]), float(d[:, 0].std())
    # (4) horseshoe coupled to the fitted treatment regression, fitted on the whole sample, as a second unpenalized covariate
    t0 = time.time()
    that = LassoCV(cv=5).fit(X, t).predict(X)
    Z2 = np.column_stack([t, that, X])
    d2 = hs_gibbs(y, Z2, unpen={0, 1}, rng=rng)
    sec_hsc = time.time() - t0
    est["hs_coupled"] = float(d2[:, 0].mean())
    q2 = np.quantile(d2[:, 0], [0.025, 0.975])
    lo["hs_coupled"], hi["hs_coupled"], sd["hs_coupled"] = float(q2[0]), float(q2[1]), float(d2[:, 0].std())
    # Everything below is appended after the four estimators above, and the ols steps use no random numbers, so the random stream of (1) to (4) is unchanged.
    # (5) least squares of y on (t, X), and (6) of y on (t, X[:, :5]) with the five true confounders (oracle), no intercept, classical t intervals; in the weak30 design the oracle uses the 30 confounders X[:, :30] (k_conf), with n - 31 degrees of freedom
    for key, Zm in (("ols_all", Z), ("ols_oracle", np.column_stack([t, X[:, :k_conf]]))):
        dof = n - Zm.shape[1]
        if dof <= 0:
            # no residual degrees of freedom (p + 1 >= n): the classical estimate and interval do not exist; null in the JSON file, NaN once loaded
            est[key], lo[key], hi[key], sd[key] = None, None, None, None
            continue
        beta = np.linalg.lstsq(Zm, y, rcond=None)[0]
        resid = y - Zm @ beta
        se_ols = np.sqrt(resid @ resid / dof * np.linalg.inv(Zm.T @ Zm)[0, 0])
        half = stats.t.ppf(0.975, dof) * se_ols
        est[key] = float(beta[0])
        lo[key], hi[key], sd[key] = float(beta[0] - half), float(beta[0] + half), float(se_ols)
    # (7) horseshoe coupled to the cross fitted treatment regression: the out of fold LassoCV fit of t on X, same folds as DML
    t0 = time.time()
    mcf = np.zeros(n)
    for tr, te in kf.split(X):
        mcf[te] = LassoCV(cv=5).fit(X[tr], t[tr]).predict(X[te])
    Z3 = np.column_stack([t, mcf, X])
    d3 = hs_gibbs(y, Z3, unpen={0, 1}, rng=rng)
    sec_hscf = time.time() - t0
    est["hs_coupled_cf"] = float(d3[:, 0].mean())
    q3 = np.quantile(d3[:, 0], [0.025, 0.975])
    lo["hs_coupled_cf"], hi["hs_coupled_cf"], sd["hs_coupled_cf"] = float(q3[0]), float(q3[1]), float(d3[:, 0].std())
    diag = dict(hs_coupled=mech_diag(t, that, X, b, g, eps, d2, k_conf), hs_coupled_cf=mech_diag(t, mcf, X, b, g, eps, d3, k_conf))
    return dict(rep=rep, seed=SEED0 + rep, n=n, p=p, tau=tau, **(dict(design="weak30", n_conf=k_conf) if weak30 else {}), diag=diag, est=est, lo=lo, hi=hi,
                se_or_posterior_sd=sd,
                seconds=dict(total=time.time() - t_start, hs=sec_hs, hs_coupled=sec_hsc, hs_coupled_cf=sec_hscf))


def fmt3(x):
    """Three decimals, with the TeX minus sign for negative numbers (no minus sign on a printed zero)."""
    s = "%.3f" % abs(x)
    return "$-$" + s if (x < 0 and float(s) != 0.0) else s


def load_reps(rep_dir):
    files = sorted(rep_dir.glob("rep_*.json"), key=lambda f: int(re.search(r"rep_(\d+)\.json$", f.name).group(1)))
    reps = []
    for f in files:
        with open(f) as fh:
            reps.append(json.load(fh))
    return reps


def paired(err, a, b):
    """Replication by replication comparison of estimator a with estimator b: squared error a minus squared error b, with its standard error, the matching RMSE difference (delta method), and the share of replications in which a is closer to the truth."""
    ea, eb = err[a], err[b]
    n = len(ea)
    d = ea ** 2 - eb ** 2
    se = d.std(ddof=1) / np.sqrt(n)
    ra, rb = np.sqrt(np.mean(ea ** 2)), np.sqrt(np.mean(eb ** 2))
    share = float(np.mean(np.abs(ea) < np.abs(eb)))
    ds = ea - eb
    return dict(a=a, b=b, mse_diff=float(d.mean()), mse_diff_se=float(se), mse_diff_t=float(d.mean() / se),
                rmse_diff=float(ra - rb), rmse_diff_se=float(se / (ra + rb)),
                share_a_closer=share, share_a_closer_mcse=float(np.sqrt(share * (1 - share) / n)),
                err_diff_mean=float(ds.mean()), err_diff_se=float(ds.std(ddof=1) / np.sqrt(n)))


def aggregate(paths, tag=None, n_arg=None, p_arg=None):
    rep_dir = paths["rep_dir"]
    reps = load_reps(rep_dir)
    nrep = len(reps)
    if nrep < 2:
        sys.exit("fewer than two replication files in %s" % rep_dir)
    stale = [r["rep"] for r in reps if any(k not in r["est"] for k in KEYS)]
    if stale:
        sys.exit("%d replication files lack some of the %d estimators (first: %s); rerun them with --force" % (len(stale), len(KEYS), stale[:5]))
    ids = [r["rep"] for r in reps]
    assert len(set(ids)) == nrep
    ns, ps = {r["n"] for r in reps}, {r["p"] for r in reps}
    if len(ns) != 1 or len(ps) != 1:
        sys.exit("replication files of different designs in %s: n in %s, p in %s" % (rep_dir, sorted(ns), sorted(ps)))
    n, p = ns.pop(), ps.pop()
    if (n_arg is not None and n_arg != n) or (p_arg is not None and p_arg != p):
        sys.exit("--n %s --p %s do not match the replication files in %s (n = %d, p = %d)" % (n_arg, p_arg, rep_dir, n, p))
    if tag is None and (n, p) != (N, P):
        sys.exit("files of the design n = %d, p = %d in the default folder %s" % (n, p, rep_dir))
    # least squares on all the covariates needs n - p - 1 > 0; where it does not exist the replication files hold null for it
    undefined = [] if n - p - 1 > 0 else ["ols_all"]
    keys = [k for k in KEYS if k not in undefined]
    for k in KEYS:
        n_null = sum(r["est"][k] is None for r in reps)
        if n_null != (nrep if k in undefined else 0):
            sys.exit("%s is null in %d of %d replication files of %s (n = %d, p = %d)" % (k, n_null, nrep, rep_dir, n, p))
    import scipy, sklearn
    err = {k: np.array([r["est"][k] for r in reps], dtype=float) - TAU for k in keys}
    names = ROW_NAMES_WEAK30 if tag == "weak30" else ROW_NAMES
    intervals = dict(INTERVALS)
    intervals["ols_all"] = ("classical t interval, n - %d degrees of freedom" % (p + 1)) if not undefined else \
        "not defined: n - p - 1 = %d degrees of freedom" % (n - p - 1)
    if tag == "weak30":
        intervals["ols_oracle"] = "classical t interval, n - %d degrees of freedom" % (WEAK30["n_conf"] + 1)
    truth = "tau = 1; five covariates with unit coefficients in the treatment and outcome equations, no intercept"
    if tag == "weak30":
        if {r.get("design") for r in reps} != {"weak30"}:
            sys.exit("replication files in %s are not all of design weak30" % rep_dir)
        truth = "tau = 1; 30 confounders with coefficient 0.3 in the treatment and outcome equations, 4 further covariates with coefficient 1 in the outcome equation only, no intercept; oracle = least squares on t and the 30 confounders"
    design = dict(n=n, p=p, tau=TAU, gibbs_sweeps=1500, burn_in=500, seed_base=SEED0, folds=5,
                  truth=truth,
                  estimators_in_table_order=KEYS, intervals=intervals,
                  versions=dict(numpy=np.__version__, scipy=scipy.__version__, scikit_learn=sklearn.__version__))
    if tag is not None:
        design["tag"] = tag
        design["ols_all_defined"] = not undefined
    out = dict(design=design, n_reps=nrep, rep_min=min(ids), rep_max=max(ids), estimators={}, paired=[])
    for k in KEYS:
        if k in undefined:
            out["estimators"][k] = dict(label=names[k], n_reps=nrep, bias=None, bias_mcse=None, rmse=None, rmse_mcse=None, sd=None,
                                        coverage=None, coverage_mcse=None, median_interval_length=None, mean_interval_length=None,
                                        miss_interval_above_truth=None, miss_interval_below_truth=None, defined=False)
            continue
        e = err[k]
        bias = e.mean()
        e2 = e ** 2
        rmse = np.sqrt(e2.mean())
        d = dict(label=names[k], n_reps=nrep, bias=bias, bias_mcse=e.std(ddof=1) / np.sqrt(nrep),
                 rmse=rmse, rmse_mcse=e2.std(ddof=1) / (2 * rmse * np.sqrt(nrep)), sd=e.std(ddof=1),
                 coverage=None, coverage_mcse=None, median_interval_length=None, mean_interval_length=None,
                 miss_interval_above_truth=None, miss_interval_below_truth=None)
        if k in INTERVAL_KEYS:
            lo = np.array([r["lo"][k] for r in reps], dtype=float); hi = np.array([r["hi"][k] for r in reps], dtype=float)
            cov = float(np.mean((lo <= TAU) & (TAU <= hi)))
            d.update(coverage=cov, coverage_mcse=np.sqrt(cov * (1 - cov) / nrep),
                     median_interval_length=np.median(hi - lo), mean_interval_length=np.mean(hi - lo),
                     miss_interval_above_truth=float(np.mean(lo > TAU)), miss_interval_below_truth=float(np.mean(hi < TAU)))
        out["estimators"][k] = {a: (float(v) if isinstance(v, (float, np.floating)) else v) for a, v in d.items()}
    pairs = [(a, b) for a, b in PAIRS + (PAIRS_TAGGED_EXTRA if tag is not None else []) if a not in undefined and b not in undefined]
    out["paired"] = [paired(err, a, b) for a, b in pairs]
    if all("diag" in r for r in reps):
        out["mechanism"] = {k: {m: float(np.mean([r["diag"][k][m] for r in reps])) for m in reps[0]["diag"][k]} for k in ("hs_coupled", "hs_coupled_cf")}
        for k, v in out["mechanism"].items():
            print("mechanism %-14s" % k, "  ".join("%s %.3f" % (m, x) for m, x in v.items()))
    secs = np.array([r["seconds"]["total"] for r in reps])
    out["seconds_per_replication"] = dict(mean=float(secs.mean()), median=float(np.median(secs)), max=float(secs.max()),
                                          hs_mean=float(np.mean([r["seconds"]["hs"] for r in reps])),
                                          hs_coupled_mean=float(np.mean([r["seconds"]["hs_coupled"] for r in reps])),
                                          hs_coupled_cf_mean=float(np.mean([r["seconds"]["hs_coupled_cf"] for r in reps])),
                                          note="wall clock on a shared, loaded machine")
    for v in list(out["estimators"].values()) + out["paired"]:
        for a, x in v.items():
            if isinstance(x, float) and not np.isfinite(x):
                sys.exit("non finite value in %s" % a)
    write_json(paths["res_json"], out)
    if tag == "weak30":
        make_table(out, nrep, n, p, tag, undefined, paths, names)
        make_md_table(out, nrep, n, p, paths, names)
    else:
        make_figure(np.column_stack([err[k] for k in keys]), [KEYS.index(k) for k in keys], paths["fig_pdf"])
        make_table(out, nrep, n, p, tag, undefined, paths, names)
    for k in KEYS:
        v = out["estimators"][k]
        if k in undefined:
            print("%-48s n/a (n - p - 1 = %d degrees of freedom)" % (v["label"].replace("$", ""), n - p - 1))
            continue
        cv = "" if v["coverage"] is None else "  cover %.3f (%.3f)  medlen %.3f" % (v["coverage"], v["coverage_mcse"], v["median_interval_length"])
        print("%-48s bias %7.3f (%.3f)  rmse %.3f (%.4f)  sd %.3f%s" % (v["label"].replace("$", ""), v["bias"], v["bias_mcse"], v["rmse"], v["rmse_mcse"], v["sd"], cv))
    for q in out["paired"]:
        print("%s minus %s: mse diff %+.5f (se %.5f, t %+.2f)  rmse diff %+.4f (se %.4f)  share a closer %.3f (%.3f)" % (
            q["a"], q["b"], q["mse_diff"], q["mse_diff_se"], q["mse_diff_t"], q["rmse_diff"], q["rmse_diff_se"], q["share_a_closer"], q["share_a_closer_mcse"]))
    print("n_reps", nrep, "seconds per rep (mean, median, max): %.1f %.1f %.1f" % (secs.mean(), np.median(secs), secs.max()))


def make_figure(E, idx, fig_pdf):
    """Box plots of estimate minus truth; E has one column per estimator that exists in the design, idx their positions in KEYS."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    # black outlines, light gray fill and black median lines, so that the boxes read the same in grayscale
    style = dict(patch_artist=True, widths=0.6,
                 boxprops=dict(facecolor="0.88", edgecolor="black", linewidth=1.0),
                 medianprops=dict(color="black", linewidth=1.8),
                 whiskerprops=dict(color="black", linewidth=1.0), capprops=dict(color="black", linewidth=1.0),
                 flierprops=dict(marker="o", markersize=2.5, markerfacecolor="0.45", markeredgecolor="0.45"))
    data = [E[:, i] for i in range(len(idx))]
    try:
        ax.boxplot(data, orientation="horizontal", **style)
    except TypeError:
        ax.boxplot(data, vert=False, **style)
    ax.set_yticks(range(1, len(idx) + 1)); ax.set_yticklabels([FIG_LABELS[i] for i in idx])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Estimate minus truth")
    fig.tight_layout(); fig.savefig(fig_pdf); plt.close()


def make_md_table(out, nrep, n, p, paths, names):
    est = out["estimators"]
    L = ["Design weak30: %d replications, n = %d, p = %d, tau = 1; 30 confounders with gamma = beta = 0.3, 4 outcome-only covariates with beta = 1." % (nrep, n, p), "",
         "| Estimator | Bias | RMSE | Coverage | Median width |", "|---|---|---|---|---|"]
    for k in KEYS:
        v = est[k]
        cov = "" if v["coverage"] is None else "%.3f" % v["coverage"]
        w = "" if v["median_interval_length"] is None else "%.3f" % v["median_interval_length"]
        L.append("| %s | %.3f | %.3f | %s | %s |" % (names[k].replace("$", ""), v["bias"], v["rmse"], cov, w))
    paths["md"].parent.mkdir(parents=True, exist_ok=True)
    paths["md"].write_text("\n".join(L) + "\n")


def make_table(out, nrep, n, p, tag, undefined, paths, names=ROW_NAMES):
    est = out["estimators"]
    mx = lambda key: max(est[k][key] for k in KEYS if est[k][key] is not None)
    cap_head = ("Bias, root mean squared error (RMSE) and coverage of seven estimators of $\\tau = 1$ over %d simulated data sets with $n = %d$ and $p = %d$. "
                "Coverage is for nominal 95\\%% intervals: the classical $t$ interval for least squares, the sandwich interval for DML and the central 95\\%% posterior interval for each horseshoe. The lasso plug in has no interval. "
                "The oracle regression uses the five true confounders only and is not available in practice. ") % (nrep, n, p)
    if tag == "weak30":
        cap_head = cap_head.replace("$p = %d$. " % p, "$p = %d$, 30 confounders with $\\gamma_j = \\beta_j = 0.3$ and four outcome only covariates with $\\beta_j = 1$. " % p)
        cap_head = cap_head.replace("The oracle regression uses the five true confounders only and is not available in practice. ", "The oracle regression uses the 30 confounders only, so the four outcome only covariates stay in its error, and it is not available in practice. ")
    if tag is None:
        cap_mid = ""
    elif undefined:
        cap_mid = ("Least squares on all the covariates needs $n - p - 1 > 0$ degrees of freedom and is not defined here ($n - p - 1 = %d$), "
                   "so its row reads n/a. ") % (n - p - 1)
    else:
        cap_mid = "The least squares interval uses $n - p - 1 = %d$ degrees of freedom. " % (n - p - 1)
    cap_tail = ("Monte Carlo standard errors are at most %.3f for the biases, %.3f for the RMSEs and %.3f for the coverages.") % (
        mx("bias_mcse"), mx("rmse_mcse"), mx("coverage_mcse"))
    cap = cap_head + cap_mid + cap_tail
    lines = ["\\begin{table}[ht]", "\\centering", "\\small", "\\caption{" + cap + "}", "\\label{" + paths["label"] + "}",
             "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{2.3in}ccc@{}}", "\\toprule", "Estimator & Bias & RMSE & Coverage \\\\", "\\midrule"]
    for k in KEYS:
        v = est[k]
        if k in undefined:
            lines.append("%s & n/a & n/a & n/a \\\\" % names[k])
            continue
        c = "" if v["coverage"] is None else fmt3(v["coverage"])
        lines.append("%s & %s & %s & %s \\\\" % (names[k], fmt3(v["bias"]), fmt3(v["rmse"]), c))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    paths["table_tex"].write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--rep", type=int, metavar="N", help="run replication N")
    grp.add_argument("--aggregate", action="store_true", help="aggregate results/ric/rep_*.json (results/ric_TAG/rep_*.json with --tag)")
    ap.add_argument("--force", action="store_true", help="recompute a replication whose file exists")
    ap.add_argument("--n", type=int, default=None, metavar="INT", help="sample size (default %d)" % N)
    ap.add_argument("--p", type=int, default=None, metavar="INT", help="number of covariates (default %d, at least 5)" % P)
    ap.add_argument("--design", type=str, default=None, choices=["weak30"], help="named design: weak30 = 30 weak confounders (n = 200, p = 150, tag weak30); results/ric_weak30/, results/res_ric_weak30.json, book_tex/chapters/ric_weak30_table.tex (label tab:ric_weak30), misc/review-loop/newch/ric_weak30_table.md")
    ap.add_argument("--tag", type=str, default=None, metavar="STR", help="design name; files go to results/ric_STR/, results/res_ric_STR.json, figs/ric_STR.pdf, book_tex/chapters/ric_STR_table.tex; required when n or p is not the default")
    a = ap.parse_args()
    if a.design == "weak30":
        if a.tag not in (None, "weak30") or a.n not in (None, WEAK30["n"]) or a.p not in (None, WEAK30["p"]):
            ap.error("--design weak30 fixes n = 200, p = 150, tag weak30")
        a.tag, a.n, a.p = "weak30", WEAK30["n"], WEAK30["p"]
    elif a.tag == "weak30":
        ap.error("the tag weak30 is reserved for --design weak30")
    if a.tag is not None and not re.fullmatch(r"[A-Za-z0-9_]+", a.tag):
        ap.error("--tag must consist of letters, digits and underscores")
    paths = design_paths(a.tag)
    if a.aggregate:
        aggregate(paths, a.tag, a.n, a.p)
        return
    n = N if a.n is None else a.n
    p = P if a.p is None else a.p
    if a.tag is None and (n, p) != (N, P):
        ap.error("--n %d --p %d is not the default design (n = %d, p = %d): give --tag, so that the default files are not overwritten" % (n, p, N, P))
    if p < 5 or n < 20:
        ap.error("need p >= 5 (five confounders) and n >= 20 (five fold cross fitting)")
    path = paths["rep_dir"] / ("rep_%d.json" % a.rep)
    if path.exists() and not a.force:
        with open(path) as fh:
            old = json.load(fh)
        if (old["n"], old["p"]) != (n, p):
            sys.exit("%s holds a replication with n = %d, p = %d, not n = %d, p = %d; use another --tag or --force" % (path, old["n"], old["p"], n, p))
        print("rep %d exists, skipped" % a.rep)
        return
    r = one_rep(a.rep, n=n, p=p, weak30=(a.design == "weak30"))
    write_json(path, r)
    print("rep %d done in %.1f s" % (a.rep, r["seconds"]["total"]) + ("" if a.tag is None else " (tag %s, n = %d, p = %d)" % (a.tag, n, p)))


if __name__ == "__main__":
    main()
