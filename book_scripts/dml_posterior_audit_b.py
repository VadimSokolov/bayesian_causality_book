"""Second batch of the DML audit (r1023b): cross fitted reduced form posterior, dense design, n = 1200, horseshoe convergence check, timing.

Companion of dml_posterior_audit.py (samplers copied from dml/sim/dmlsim.py after the r1023 fixes) and dml_baseline_audit.py (data, seeds). Every
replication r of design d uses the audit's seed crc32("d|r") and so sees the data set of the audit's lasso, post double selection and repeated split
arms (n = 300, p = 400, true effect 1, outcome noise sd 0.948; designs s5, s15, s30, weak, bigN with n = 1200, p = 200, s = 15, and dense, in which
every coordinate of the treatment and outcome coefficients is nonzero and the total signal of s = 15 is spread over all p coordinates). Modes, each
writing results/dml_<mode>/DESIGN_START_END.jsonl:
  cf5    rf_cf5: the cross fitted reduced form posterior at K = 5 folds (folds: permutation with rng seed + 14; the horseshoe reduced form fitted on the
         four training folds, the effect from the held out fold, draw t of the five fold chains paired, th_hat = sum num / sum den, tau ~ N(th_hat,
         mean s2 / den) as in xfit_bayes). Four chains per fold from the default start, 10,000 sweeps with half discarded, doubled up to 80,000 until
         split R-hat of th_hat is at most 1.01 and the multichain ESS at least 400.
  cf2    rf_cf2: the same at K = 2, tau drawn jointly with varsigma (block_theta), four chains per fold from overdispersed scales (0.1, 1, 10 and 0.1/10
         times the scales of a 3,000 sweep pilot) and tau = 0, from 100,000 sweeps doubled up to 800,000, same convergence rule.
  dpost  rf and nb on the dense design, four chains from overdispersed starts (reduced form: scales 0.1, 1 and 10 times the stationary region
         s_nu2 = 17.5, s_ep2 = 1.7 and tau = -3, 1.92, 6, 0; naive Bayes: sigma^2 0.1, 1, 10 and 100 times the mean of a 2,000 sweep pilot), 100,000
         sweeps doubled up to 800,000 until R-hat <= 1.01 and ESS >= 400.
  light  hs_K5, hs_K2 (DML with a horseshoe nuisance, 1,200 sweeps, 400 burn in) and boot (multiplier bootstrap at the audit's K = 5 cross fit), as in
         dml_posterior_audit.py, for the dense and bigN designs.
  hs5000 the horseshoe nuisance arms at 5,000 sweeps with 1,000 burn in, same seeds as hs_K5, hs_K2 (to compare coverage with the 1,200 sweep values).
  timing ten data sets of s15 on one core: cross fitted lasso K = 5, one published chain (2,000 sweeps), the four chain reduced form protocol, the
         naive Bayes protocol, the horseshoe nuisance at K = 5.
Usage (repository root, PYTHONPATH=.):  python book_scripts/dml_posterior_audit_b.py MODE DESIGN START END  |  agg MODE
Environment: R1B_PROCS (processes per task for cf2 and dpost), R1B_K2_L0, R1B_K2_LMAX, R1B_D_L0, R1B_D_LMAX, R1_L0, R1_LMAX (smoke tests).
"""
import glob, json, math, os, sys, time, zlib
import multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dml_baseline_audit as A
import dml_posterior_audit as P
from dml_baseline_audit import DESIGNS, Z95, plr

PROCS = int(os.environ.get("R1B_PROCS", 1))
K2_L0, K2_LMAX = int(os.environ.get("R1B_K2_L0", 100000)), int(os.environ.get("R1B_K2_LMAX", 800000))
D_L0, D_LMAX = int(os.environ.get("R1B_D_L0", 100000)), int(os.environ.get("R1B_D_LMAX", 800000))


def _seed(d, r): return zlib.crc32(("%s|%d" % (d, r)).encode()) % (2 ** 31)


def _data(design, seed):
    spec = DESIGNS[design]; X, D, Y, alpha, gamma = A.make_design_full(np.random.default_rng(seed), **spec)
    return X, D, Y, alpha, spec.get("theta0", 1.0) * alpha + gamma


# ------------------------------------------------------------------ cross fitted posterior
def _fold_job(a):
    design, seed, K, fold, c, L, od, block = a
    X, D, Y, alpha, vs = _data(design, seed); n = len(D)
    idx = np.random.default_rng(seed + 14).permutation(n); f = np.array_split(idx, K)[fold]; tr = np.setdiff1d(idx, f); init = None
    if od:
        pl = P._rf_gibbs(X[tr], D[tr], Y[tr], np.random.default_rng(seed + 99 + fold), 3000, 1500)
        rn, re_ = float(np.mean(pl["sig_nu2"])), float(np.mean(pl["sig_ep2"]))
        init = [dict(s_nu2=.1 * rn, s_ep2=.1 * re_, theta=0.0), dict(s_nu2=rn, s_ep2=re_, theta=0.0), dict(s_nu2=10 * rn, s_ep2=10 * re_, theta=0.0),
                dict(s_nu2=.1 * rn, s_ep2=10 * re_, theta=0.0)][c]
    rng = np.random.default_rng(seed + 14 + 10007 * (c + 1) + 1000003 * fold)
    o = P._rf_gibbs(X[tr], D[tr], Y[tr], rng, L, L // 2, eval_block=(X[f], D[f], Y[f]), eval_truth=(X[f] @ alpha, X[f] @ vs), init=init, block_theta=block)
    return {k: o[k] for k in ("ev_num", "ev_den", "ev_s2", "ev_own", "ev_cross", "ev_el2")}


def xfit_run(design, seed, K, L, Lmax, od, block, pool=None):
    mapper = pool.map if pool is not None else map
    while True:
        res = list(mapper(_fold_job, [(design, seed, K, f, c, L, od, block) for c in range(4) for f in range(K)]))
        R = {(c, f): res[c * K + f] for c in range(4) for f in range(K)}
        thh, ths, cat = [], [], {k: [] for k in ("ev_own", "ev_cross", "ev_el2")}; rng = np.random.default_rng(seed + 14 + 77)
        for c in range(4):
            num = sum(R[c, f]["ev_num"] for f in range(K)); den = sum(R[c, f]["ev_den"] for f in range(K)); s2 = sum(R[c, f]["ev_s2"] for f in range(K)) / K
            thh.append(num / den); ths.append(rng.normal(num / den, np.sqrt(s2 / den)))
            for k in cat: cat[k].append(sum(R[c, f][k] for f in range(K)) / K)
        thh = np.array(thh); rh, es = P.split_rhat_ess(thh)
        if (rh <= 1.01 and es >= 400) or L >= Lmax: break
        L *= 2
    th = np.concatenate(ths); lo, hi = np.quantile(th, [0.025, 0.975])
    o, cr, e = (float(np.mean(np.concatenate(cat[k]))) for k in ("ev_own", "ev_cross", "ev_el2"))
    return dict(theta=float(th.mean()), se=float(th.std(ddof=1)), lo=float(lo), hi=float(hi), K=K, L=L, rhat=rh, ess=es, converged=bool(rh <= 1.01 and es >= 400),
                chain_means=[float(x.mean()) for x in thh], own=o, cross=cr, el2=e, rho=float(cr / math.sqrt(o * e)) if o > 0 and e > 0 else float("nan"))


# ------------------------------------------------------------------ dense design, overdispersed posterior
REG = (17.5, 1.7)
RF_STARTS = [dict(s_nu2=.1 * REG[0], s_ep2=.1 * REG[1], theta=-3.0), dict(s_nu2=REG[0], s_ep2=REG[1], theta=1.92), dict(s_nu2=10 * REG[0], s_ep2=10 * REG[1], theta=6.0),
             dict(s_nu2=.1 * REG[0], s_ep2=10 * REG[1], theta=0.0)]


def _dchain(a):
    kind, design, seed, c, L, sig0 = a
    X, D, Y, alpha, vs = _data(design, seed)
    if kind == "rf":
        o = P._rf_gibbs(X, D, Y, np.random.default_rng(seed + 3 + 10007 * c), L, L // 2, init=RF_STARTS[c]); return dict(theta=o["theta"], om=o["omega2"])
    Z = np.column_stack([D, X])
    o = P.horseshoe_gibbs(Z, Y, np.random.default_rng(seed + 4 + 10007 * c), n_iter=L, burn=L // 2, unpenalised=0, keep_scales=True, init_sig2=sig0 * [.1, 1., 10., 100.][c])
    return dict(theta=o["draws"][:, 0], om=o["omega2"])


def dense_post(design, seed, pool):
    X, D, Y, alpha, vs = _data(design, seed); out = {}
    sig0 = float(np.mean(P.horseshoe_gibbs(np.column_stack([D, X]), Y, np.random.default_rng(seed + 55), n_iter=2000, burn=1000, unpenalised=0, keep_scales=True)["sigma2"]))
    for kind, key in (("rf", "rf"), ("nb", "nb")):
        L = D_L0
        while True:
            outs = pool.map(_dchain, [(kind, design, seed, c, L, sig0) for c in range(4)]); th4 = np.array([o["theta"] for o in outs]); rh, es = P.split_rhat_ess(th4)
            if (rh <= 1.01 and es >= 400) or L >= D_LMAX: break
            L *= 2
        out[key] = P.summarize(th4, P.split_rhat_ess(np.log([o["om"] for o in outs]))[1], L); out[key]["overdispersed"] = True
    return out


# ------------------------------------------------------------------ light arms, hs5000
def dml_hs_b(X, D, Y, rng, K, hs_iter, burn):
    n = len(D); idx = rng.permutation(n); m = np.empty(n); l = np.empty(n)
    for f in np.array_split(idx, K):
        tr = np.setdiff1d(idx, f)
        cm = P.horseshoe_gibbs(X[tr], D[tr], rng, n_iter=hs_iter, burn=burn)["mean"]; cl = P.horseshoe_gibbs(X[tr], Y[tr], rng, n_iter=hs_iter, burn=burn)["mean"]
        m[f] = X[f] @ cm; l[f] = X[f] @ cl
    return plr(D - m, Y - l)


def light_rep(design, r, seed):
    X, D, Y, alpha, vs = _data(design, seed); n = len(D); rec = {}
    rec["hs_K5"] = dml_hs_b(X, D, Y, np.random.default_rng(seed + 2), 5, 1200, 400); rec["hs_K2"] = dml_hs_b(X, D, Y, np.random.default_rng(seed + 22), 2, 1200, 400)
    rD, rY = A.resid(X, D, Y, np.random.default_rng(seed + 1), 5); th, se = plr(rD, rY)
    J = np.mean(rD ** 2); psi = rD * (rY - th * rD) / J; wr = np.random.default_rng(seed + 9)
    a = (1 - math.sqrt(5)) / 2; b = (1 + math.sqrt(5)) / 2; pr = (math.sqrt(5) + 1) / (2 * math.sqrt(5))
    u = np.where(wr.random((999, n)) < pr, a, b); draws = th + (u - 1.0) @ psi / n; lo, hi = np.quantile(draws, [0.025, 0.975])
    rec["boot"] = dict(theta=th, se=se, lo=float(lo), hi=float(hi), se_boot=float(np.std(draws, ddof=1)))
    return rec


def hs5000_rep(design, r, seed):
    X, D, Y, alpha, vs = _data(design, seed)
    return dict(hs5000_K5=dml_hs_b(X, D, Y, np.random.default_rng(seed + 2), 5, 5000, 1000), hs5000_K2=dml_hs_b(X, D, Y, np.random.default_rng(seed + 22), 2, 5000, 1000))


def timing_rep(design, r, seed):
    X, D, Y, alpha, vs = _data(design, seed); t = {}
    def tm(name, fn):
        t0 = time.time(); fn(); t[name] = round(time.time() - t0, 2)
    tm("lasso_K5", lambda: A.dml(X, D, Y, np.random.default_rng(seed + 1), 5))
    tm("rf_one_chain_2000", lambda: P._rf_gibbs(X, D, Y, np.random.default_rng(seed + 3), 2000, 700))
    tm("rf_4chain_protocol", lambda: P.run_posterior(lambda rng, L, b: P._rf_gibbs(X, D, Y, rng, L, b), lambda o: o["theta"], lambda o: o["omega2"], seed + 3))
    tm("nb_4chain_protocol", lambda: P.run_nb(np.column_stack([D, X]), Y, seed + 4))
    tm("hs_K5_1200", lambda: dml_hs_b(X, D, Y, np.random.default_rng(seed + 2), 5, 1200, 400))
    return dict(timing=t)


def main_run(mode, design, a, b):
    outd = "results/dml_%s/" % mode; os.makedirs(outd, exist_ok=True); t0 = time.time()
    pool = mp.get_context("fork").Pool(PROCS) if (PROCS > 1 and mode in ("cf2", "dpost")) else None
    if mode == "dpost" and pool is None: raise SystemExit("dpost needs R1B_PROCS >= 4")
    with open(outd + "%s_%05d_%05d.jsonl" % (design, a, b), "w") as fh:
        for r in range(a, b):
            seed = _seed(design, r); rec = dict(design=design, rep=r, seed=seed)
            if mode == "cf5": rec["rf_cf5"] = xfit_run(design, seed, 5, P.L0, P.LMAX, False, False)
            elif mode == "cf2": rec["rf_cf2"] = xfit_run(design, seed, 2, K2_L0, K2_LMAX, True, True, pool)
            elif mode == "dpost": rec.update(dense_post(design, seed, pool))
            elif mode == "light": rec.update(light_rep(design, r, seed))
            elif mode == "hs5000": rec.update(hs5000_rep(design, r, seed))
            elif mode == "timing": rec.update(timing_rep(design, r, seed))
            rec["elapsed"] = round(time.time() - t0, 1); fh.write(json.dumps(rec) + "\n"); fh.flush()
    print("done", mode, design, a, b, round(time.time() - t0))


# ------------------------------------------------------------------ aggregation
def _load(pattern): return {json.loads(l)["rep"]: json.loads(l) for f in sorted(glob.glob(pattern)) for l in open(f)}


def _iv(x):
    if isinstance(x, dict): return x["theta"], x["lo"], x["hi"], (x["se_boot"] if "se_boot" in x else x["se"])
    return x[0], x[0] - Z95 * x[1], x[0] + Z95 * x[1], x[1]


def agg(mode):
    from scipy.stats import binomtest
    lines, out = [], {}
    for d in ["s5", "s15", "s30", "weak", "dense", "bigN"]:
        new = _load("results/dml_%s/%s_*.jsonl" % (mode, d))
        if not new: continue
        reps = sorted(new); N = len(reps); lines.append("## %s %s (%d replications)" % (mode, d, N)); out[d] = {}
        # reference arms: R1 (posterior, horseshoe, bootstrap) and the audit (lasso K5)
        ref = {r: {} for r in reps}
        for r, x in _load("results/dml_posterior_audit/%s_*.jsonl" % d).items():
            if r in ref: ref[r].update(x)
        for r, x in _load("results/dml_baseline_audit/%s_*.jsonl" % d).items():
            if r in ref: ref[r].update({k: v for k, v in x.items() if k not in ref[r]})
        arms = sorted({k for x in new.values() for k, v in x.items() if isinstance(v, (dict, list)) and k not in ("timing",)})
        lines.append("| arm | bias (MCSE) | coverage (SE) | s.e./s.d. | s.e./RMSE | width | converged | median ESS |"); lines.append("|---|---|---|---|---|---|---|---|")
        cov = {}
        for a_ in arms:
            v = np.array([_iv(new[r][a_]) for r in reps]); th, lo, hi, se = v.T; c = (lo <= 1) & (1 <= hi); cov[a_] = c; sd = th.std(ddof=1)
            cv = [new[r][a_].get("converged") for r in reps if isinstance(new[r][a_], dict) and "converged" in new[r][a_]]
            es = [new[r][a_]["ess"] for r in reps if isinstance(new[r][a_], dict) and "ess" in new[r][a_]]
            q = out[d][a_] = dict(bias=float(th.mean() - 1), bias_mcse=float(sd / math.sqrt(N)), coverage=float(c.mean()), coverage_se=float(math.sqrt(c.mean() * (1 - c.mean()) / N)), se_over_sd=float(se.mean() / sd),
                                  se_over_rmse=float(se.mean() / math.sqrt(np.mean((th - 1) ** 2))), width=float((hi - lo).mean()), share_converged=(float(np.mean(cv)) if cv else None),
                                  n_converged=(int(np.sum(cv)) if cv else None), median_ess=(float(np.median(es)) if es else None))
            if "L" in new[reps[0]][a_] if isinstance(new[reps[0]][a_], dict) else False:
                Ls = np.array([new[r][a_]["L"] for r in reps]); q["max_L"] = int(Ls.max()); q["share_L_above_first"] = float((Ls > Ls.min()).mean())
            lines.append("| %s | %.3f (%.3f) | %.3f (%.3f) | %.2f | %.2f | %.3f | %s | %s |" % (a_, q["bias"], q["bias_mcse"], q["coverage"], q["coverage_se"], q["se_over_sd"], q["se_over_rmse"], q["width"],
                         "NA" if q["share_converged"] is None else "%.3f (%d of %d)" % (q["share_converged"], q["n_converged"], N), "NA" if q["median_ess"] is None else "%.0f" % q["median_ess"]))
            if "own" in new[reps[0]][a_]: lines.append("  channels %s: own %.3f, cross %.3f, rho %.3f (means over replications)" % (a_, np.mean([new[r][a_]["own"] for r in reps]), np.mean([new[r][a_]["cross"] for r in reps]), np.mean([new[r][a_]["rho"] for r in reps])))
        # paired comparisons with the R1 / audit arms available on the same replications
        refarms = {"rf_cf5": ["rf", "dml_K5"], "rf_cf2": ["rf", "rf_cf5", "dml_K2"], "rf": ["dml_K5", "hs_K5"], "nb": ["rf"], "hs5000_K5": ["hs_K5"], "hs5000_K2": ["hs_K2"], "hs_K5": ["dml_K5"], "boot": ["dml_K5"]}
        for a_ in arms:
            for b_ in refarms.get(a_, []):
                if b_ in arms: cb = cov[b_]
                elif all(b_ in ref[r] for r in reps): cb = np.array([(lambda t: t[1] <= 1 <= t[2])(_iv(ref[r][b_])) for r in reps])
                else: continue
                ca = cov[a_]; x = int((ca & ~cb).sum()); y = int((~ca & cb).sum()); df = ca.astype(float) - cb
                lines.append("McNemar %s vs %s: %d to %d, exact p %.3g, paired coverage difference %.3f (SE %.3f)" % (a_, b_, x, y, binomtest(min(x, y), x + y, .5).pvalue if x + y else 1.0, df.mean(), df.std(ddof=1) / math.sqrt(N)))
        lines.append("")
    if mode == "timing":
        recs = [json.loads(l) for f in sorted(glob.glob("results/dml_timing/*.jsonl")) for l in open(f)]
        if recs:
            keys = recs[0]["timing"].keys(); lines.append("## timing, %d data sets of s15, one core, seconds (mean, min, max)" % len(recs))
            for k in keys: v = [x["timing"][k] for x in recs]; lines.append("%s: %.1f %.1f %.1f" % (k, np.mean(v), min(v), max(v)))
    json.dump(out, open("results/res_dml_%s.json" % mode, "w"), indent=1); open("results/dml_%s_tables.md" % mode, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "agg": agg(sys.argv[2])
    else: main_run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
