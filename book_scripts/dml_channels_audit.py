"""Channel and selection measurements of Chapter 7 on the audit's seeds (r1023b, item RR4).

Data and seeds as dml_baseline_audit.py: replication r of design d uses crc32("d|r") (designs s5, s15, s30, weak are the audit's own and so are paired
with its arms; the variations tau05_s15 (tau = 0.5), tau20_s15 (tau = 2), mixed_s15 and mixed_s30 (the signs of the outcome coefficients alternate along
the support) use their own names as seeds), n = 300, p = 400, outcome noise sd 0.948, 5,000 replications each. Per replication and arm the record holds
the estimate and sandwich standard error and, measured against the truth on the pooled held out predictions of the fold fits, delta_m = m_hat - X alpha
and delta_l = l_hat - X varsigma (varsigma = tau alpha + gamma): own = mean delta_m^2, tau_own = tau * own (the channel tau ||delta_m||^2), cross =
mean delta_m delta_l, el2 = mean delta_l^2, rho = cross / sqrt(own el2); and, for the lasso arms, the mean over folds of the size of the support
selected for the treatment equation (n_sel_m) and the outcome equation (n_sel_l), the fraction of folds with exact support recovery of the true
treatment support, and its false positives and negatives. Arms: lasso_K5, lasso_K2 (cross validated penalty; same rng as the audit's dml_K5 and dml_K2,
so the estimates agree with the audit), theory_K5, theory_K2 (lasso at the theory penalty of Belloni, Chernozhukov and Hansen), hs_K5, hs_K2 (horseshoe
nuisance, 1,200 sweeps, 400 burn in, same rng as dml_posterior_audit.py), and the post double selection fits at both penalties (pds_theory, pds_cv:
estimate, standard error and the sizes of the treatment, outcome and union supports; same rng as the audit).
Output results/dml_chan/DESIGN_START_END.jsonl. Usage (repository root, PYTHONPATH=.):  python book_scripts/dml_channels_audit.py run DESIGN START END  |  agg
"""
import glob, json, math, os, sys, time, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dml_baseline_audit as A
import dml_posterior_audit as P
from dml_baseline_audit import DESIGNS, Z95

OUTD = "results/dml_chan/"
DES = ["s5", "s15", "s30", "weak", "tau05_s15", "tau20_s15", "mixed_s15", "mixed_s30"]


def fit(X, D, Y, alpha, vs, th0, rng, K, learner):
    n = len(D); idx = rng.permutation(n); m = np.empty(n); l = np.empty(n); sm, sl, ex, fp, fn = [], [], [], [], []; S = np.abs(alpha) > 0
    for f in np.array_split(idx, K):
        tr = np.setdiff1d(idx, f)
        if learner == "hs":
            cm = P.horseshoe_gibbs(X[tr], D[tr], rng, n_iter=1200, burn=400)["mean"]; cl = P.horseshoe_gibbs(X[tr], Y[tr], rng, n_iter=1200, burn=400)["mean"]
        else:
            cm = A.lasso_fit(X[tr], D[tr], rng, learner == "theory"); cl = A.lasso_fit(X[tr], Y[tr], rng, learner == "theory")
            v = np.abs(cm) > 1e-8; sm.append(int(v.sum())); sl.append(int((np.abs(cl) > 1e-8).sum())); ex.append(float(np.array_equal(v, S))); fp.append(int((v & ~S).sum())); fn.append(int((~v & S).sum()))
        m[f] = X[f] @ cm; l[f] = X[f] @ cl
    th, se = A.plr(D - m, Y - l); em = m - X @ alpha; el = l - X @ vs
    own, cross, el2 = float(np.mean(em ** 2)), float(np.mean(em * el)), float(np.mean(el ** 2))
    r = dict(theta=th, se=se, own=own, tau_own=th0 * own, cross=cross, el2=el2, rho=cross / math.sqrt(own * el2) if own > 0 and el2 > 0 else float("nan"))
    if sm: r.update(n_sel_m=float(np.mean(sm)), n_sel_l=float(np.mean(sl)), support_exact=float(np.mean(ex)), false_pos=float(np.mean(fp)), false_neg=float(np.mean(fn)))
    return r


def pds_sizes(X, D, Y, alpha, rng, theory):
    n = len(D); cD = A.lasso_fit(X, D, rng, theory); cY = A.lasso_fit(X, Y, rng, theory); sD = np.abs(cD) > 1e-10; sY = np.abs(cY) > 1e-10; sel = sD | sY
    W = np.column_stack([D, X[:, sel]]) if sel.any() else D.reshape(-1, 1); q = W.shape[1]; XtXi = np.linalg.pinv(W.T @ W); b = XtXi @ (W.T @ Y); r = Y - W @ b
    V = XtXi @ ((W * (r ** 2)[:, None]).T @ W) @ XtXi * (n / max(n - q, 1))
    return dict(theta=float(b[0]), se=float(math.sqrt(V[0, 0])), n_sel=int(sel.sum()), n_sel_D=int(sD.sum()), n_sel_Y=int(sY.sum()), false_neg=int((~sel & (np.abs(alpha) > 0)).sum()))


def one_rep(design, r):
    seed = zlib.crc32(("%s|%d" % (design, r)).encode()) % (2 ** 31); spec = DESIGNS[design]; th0 = spec.get("theta0", 1.0)
    X, D, Y, alpha, gamma = A.make_design_full(np.random.default_rng(seed), **spec); vs = th0 * alpha + gamma; rec = dict(design=design, rep=r, seed=seed)
    rec["lasso_K5"] = fit(X, D, Y, alpha, vs, th0, np.random.default_rng(seed + 1), 5, "cv"); rec["lasso_K2"] = fit(X, D, Y, alpha, vs, th0, np.random.default_rng(seed + 2), 2, "cv")
    rec["theory_K5"] = fit(X, D, Y, alpha, vs, th0, np.random.default_rng(seed + 11), 5, "theory"); rec["theory_K2"] = fit(X, D, Y, alpha, vs, th0, np.random.default_rng(seed + 12), 2, "theory")
    rec["hs_K5"] = fit(X, D, Y, alpha, vs, th0, np.random.default_rng(seed + 2), 5, "hs"); rec["hs_K2"] = fit(X, D, Y, alpha, vs, th0, np.random.default_rng(seed + 22), 2, "hs")
    rec["pds_theory"] = pds_sizes(X, D, Y, alpha, np.random.default_rng(seed + 7), True); rec["pds_cv"] = pds_sizes(X, D, Y, alpha, np.random.default_rng(seed + 8), False)
    return rec


def agg():
    lines, out = [], {}
    for d in DES:
        recs = [json.loads(l) for f in sorted(glob.glob(OUTD + d + "_*.jsonl")) for l in open(f)]
        if not recs: continue
        N = len(recs); out[d] = {}; lines += ["## %s (%d replications)" % (d, N), "| arm | bias (MCSE) | coverage (SE) | s.e./s.d. | own | tau_own | cross | rho | n_sel_m | n_sel_l | exact | fp | fn |", "|" + "---|" * 13]
        th0 = DESIGNS[d].get("theta0", 1.0)
        for a in ["lasso_K5", "lasso_K2", "theory_K5", "theory_K2", "hs_K5", "hs_K2", "pds_theory", "pds_cv"]:
            th = np.array([x[a]["theta"] for x in recs]); se = np.array([x[a]["se"] for x in recs]); c = np.abs(th - th0) <= Z95 * se; cc = c.mean(); sd = th.std(ddof=1)
            g = lambda k: float(np.mean([x[a][k] for x in recs])) if k in recs[0][a] else float("nan")
            q = out[d][a] = dict(bias=float(th.mean() - th0), bias_mcse=float(sd / math.sqrt(N)), coverage=float(cc), coverage_se=float(math.sqrt(cc * (1 - cc) / N)), se_over_sd=float(se.mean() / sd),
                                 **{k: g(k) for k in ("own", "tau_own", "cross", "rho", "n_sel_m", "n_sel_l", "support_exact", "false_pos", "false_neg", "n_sel", "n_sel_D", "n_sel_Y")})
            lines.append("| %s | %.3f (%.3f) | %.3f (%.3f) | %.2f | %.3f | %.3f | %.3f | %.3f | %.1f | %.1f | %.3f | %.1f | %.1f |" % (a, q["bias"], q["bias_mcse"], q["coverage"], q["coverage_se"], q["se_over_sd"], q["own"], q["tau_own"], q["cross"], q["rho"],
                         q["n_sel_m"] if a[:3] != "pds" else q["n_sel_D"], q["n_sel_l"] if a[:3] != "pds" else q["n_sel_Y"], q["support_exact"], q["false_pos"], q["false_neg"]))
        lines.append("")
    json.dump(out, open("results/res_dml_chan.json", "w"), indent=1); open("results/dml_chan_tables.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        d, a, b = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); os.makedirs(OUTD, exist_ok=True); t0 = time.time()
        with open(OUTD + "%s_%05d_%05d.jsonl" % (d, a, b), "w") as fh:
            for r in range(a, b):
                rec = one_rep(d, r); rec["elapsed"] = round(time.time() - t0, 1); fh.write(json.dumps(rec) + "\n"); fh.flush()
        print("done", d, a, b, round(time.time() - t0))
    else: agg()
