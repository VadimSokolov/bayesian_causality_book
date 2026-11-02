"""Chapter 16: quantile regressions of 401(k) net financial assets on eligibility, to read the Student t result of k401_student_t.py (review loop F10 version).
Data and design as in k401_student_t.py (1991 SIPP, n = 9,915; nine standardized covariates, squares and products: 54 terms; outcome in thousands of dollars). Each regression has eligibility, an intercept, the estimated propensity and the 54 terms, with no penalty (sklearn QuantileRegressor, alpha = 0, HiGHS), at the quantile levels 0.1, 0.25, 0.5, 0.75 and 0.9. The point estimates use the cross validated logistic propensity fitted once on the whole sample, as in the coupled horseshoe.
The coefficient on eligibility is the shift in that conditional quantile of net financial assets, not an unconditional quantile effect. Inference is a paired household bootstrap: B = 2,000 resamples of the households (resample b drawn from default_rng([4300, b]), so any split into tasks gives the same resamples); in each resample the propensity is refitted (cross validated logistic regression on the resample) and all five quantile regressions are fitted on that same resample, so the draws of the coefficients are common across the levels. Intervals are the 2.5 and 97.5 percentiles of the draws, for the five coefficients and for the contrasts tau_0.5 - tau_0.25 and tau_0.9 - tau_0.5 (and tau_0.75 - tau_0.5 and tau_0.9 - tau_0.1).
Saved: coefficients, intervals, bootstrap sd and the contrast intervals by quantile level, and descriptives of the outcome.
usage (repository root): python3 book_scripts/k401_quantiles.py run --start 0 --end 50 --out DIR   (one array task per block of resamples)
                         python3 book_scripts/k401_quantiles.py merge DIR      (writes results/res_k401_quantiles.json; the 100 resample run is results/res_k401_quantiles.json.before-r1023)
"""
import argparse, glob, json, os, time, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegressionCV, QuantileRegressor

warnings.filterwarnings("ignore")
LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]
B = 2000
d = pd.read_stata("data/sipp1991.dta")
xv = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]
Y = d.net_tfa.values.astype(float) / 1000.0
D = d.e401.values.astype(float)
X = StandardScaler().fit_transform(d[xv].values.astype(float))
P = PolynomialFeatures(2, include_bias=False).fit_transform(X)
_, keep = np.unique(np.round(P, 8), axis=1, return_index=True)
P = StandardScaler().fit_transform(P[:, np.sort(keep)])
n, p = P.shape


def prop_of(idx):
    return LogisticRegressionCV(cv=5, Cs=10, max_iter=2000).fit(P[idx], D[idx]).predict_proba(P[idx])[:, 1]


def fit(Zm, idx, q):
    return float(QuantileRegressor(quantile=q, alpha=0.0, solver="highs", fit_intercept=True).fit(Zm, Y[idx]).coef_[0])


def run(start, end, outdir):
    os.makedirs(outdir, exist_ok=True); rows = []; t0 = time.time()
    for b in range(start, end):
        idx = np.arange(n) if b < 0 else np.random.default_rng([4300, b]).integers(0, n, n)
        Zm = np.column_stack([D[idx], prop_of(idx), P[idx]]); rows.append([fit(Zm, idx, q) for q in LEVELS]); print(b, time.time() - t0, flush=True)
    np.save(os.path.join(outdir, "boot_%05d_%05d.npy" % (start, end)), np.array(rows))


def merge(outdir):
    boot = np.vstack([np.load(f) for f in sorted(glob.glob(os.path.join(outdir, "boot_*.npy")))]).T
    idx = np.arange(n); Zm = np.column_stack([D, prop_of(idx), P]); point = [fit(Zm, idx, q) for q in LEVELS]
    ci = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
    pairs = {"tau50_minus_tau25": (2, 1), "tau90_minus_tau50": (4, 2), "tau75_minus_tau50": (3, 2), "tau90_minus_tau10": (4, 0)}
    con = {k: dict(point=point[a] - point[b], lo=ci(boot[a] - boot[b])[0], hi=ci(boot[a] - boot[b])[1], boot_sd=float((boot[a] - boot[b]).std(ddof=1)),
                   share_positive=float(np.mean(boot[a] - boot[b] > 0))) for k, (a, b) in pairs.items()}
    out = {"n": int(n), "p_basis": int(p), "bootstrap_resamples": int(boot.shape[1]), "propensity_refitted_in_each_resample": True, "levels": LEVELS, "coef_eligibility": point,
           "lo": [ci(boot[k])[0] for k in range(5)], "hi": [ci(boot[k])[1] for k in range(5)], "boot_sd": [float(boot[k].std(ddof=1)) for k in range(5)], "contrasts": con,
           "outcome": {"mean": float(Y.mean()), "median": float(np.median(Y)), "min": float(Y.min()), "max": float(Y.max()),
                       "quantile_levels": [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], "quantiles": [float(v) for v in np.quantile(Y, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])]}}
    with open("results/res_k401_quantiles.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("coef_eligibility", "lo", "hi", "boot_sd", "contrasts", "bootstrap_resamples")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("dir", nargs="?"); ap.add_argument("--start", type=int); ap.add_argument("--end", type=int); ap.add_argument("--out"); a = ap.parse_args()
    run(a.start, a.end, a.out) if a.cmd == "run" else merge(a.dir)
