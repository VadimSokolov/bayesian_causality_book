"""Repeated sampling comparison of BLP GMM and GBC, and conduct misspecification experiments.

Staged version of the October 2026 rerun (the original is mc.py.before-r1023). Stages, run from this folder:
  python mc.py table                    the amortized Bertrand reference table for T = 40 and T = 15 (6,000 industries each,
                                        prior draws, so it does not depend on the true value used in the replications)
  python mc.py reps T SP LO HI          replications LO..HI of design T with true sigma_p = SP (a Pool over the cores)
  python mc.py agg SP                   collects the parts, trains the GBC target networks, writes results/mc.pkl (SP = 0.4) or
                                        results/mc_sp005.pkl (SP = 0.05); for SP = 0.4 it also runs the conduct misspecification
                                        experiment of the book (one industry each from joint monopoly and perfect competition)
  python mc.py partial                  stress test: 100 industries with partial rival internalization (kappa = 0.5), T = 40
The replications fix the exogenous design (seed 0) and redraw the unobservables and the micro survey noise; seeds are
31000 + 1000 T + i (plus 700000 when SP = 0.05). Every replication also stores the GMM plug in estimand (the estimands
evaluated at the GMM point estimate), next to the median of the Krinsky and Robb draws."""
import sys, os, glob, json, pickle, time, numpy as np
from multiprocessing import Pool
from blp_core import Design, simulate_market, CONDUCTS, tally, merge_logs
from estimators import BLPGMM, GBC, GBCTargets, PARAMS, PRIOR, draw_prior, summaries
from antitrust import structural, estimands

TRUE = dict(b0=2.0, b1=1.0, b2=0.5, alpha=1.0, sp=0.4, sx=0.7, g0=2.0, g1=0.5, g2=0.3,
            sxi=0.5, somega=0.3, rho=0.3)
DESIGNS = {40: Design(T=40, seed=0), 15: Design(T=15, seed=0)}
N_KR = 100
IA = [PARAMS.index(k) for k in ("alpha", "sp", "sx")] + [12, 13, 14]
SPTAG = {0.4: "", 0.05: "_sp005"}


def noisy_micro(r, rng):
    return r["micro"] + rng.normal(0, [0.03, 0.10])


def true_estimands(d, r, th, conduct):
    """Estimands from the simulated truth (known delta and costs), no inversion needed."""
    from blp_core import solve_equilibrium, consumer_surplus
    from antitrust import POST_FIRMS, MERGING
    a, sp, sx = th["alpha"], th["sp"], th["sx"]
    if conduct == "bertrand":
        p1, ok = solve_equilibrium(r["delta"], r["c"], d, a, sp, sx, d.ownership("bertrand", POST_FIRMS), p0=r["p"])
        if not ok:
            return None
    else:
        p1 = r["p"]
    dcs = consumer_surplus(r["delta"], p1, d, a, sp, sx) - consumer_surplus(r["delta"], r["p"], d, a, sp, sx)
    return np.array([((r["p"] - r["c"]) / r["p"]).mean(), ((p1 - r["p"]) / r["p"])[:, MERGING].mean(), dcs.mean()])


def table_one(args, log=None):
    """One draw of the amortised Bertrand reference table with its estimands."""
    T, i = args
    d = DESIGNS[T]
    rng = np.random.default_rng(900_000 + 100_000 * T + i)
    for _ in range(30):
        tally(log, "attempts")
        th = draw_prior(rng)
        r = simulate_market(d, th, "bertrand", rng)
        if not r["ok"]:
            tally(log, "rejected: " + r["why"])
            continue
        tr = true_estimands(d, r, th, "bertrand")
        if tr is None:
            tally(log, "rejected: post merger solve")
            continue
        S = summaries(d, r["s"], r["p"], noisy_micro(r, rng))
        if np.all(np.isfinite(S)):
            tally(log, "accepted")
            return np.concatenate([[th[k] for k in PARAMS], tr]), S
        tally(log, "rejected: non finite summary")
    tally(log, "jobs at the 30 try cap")
    return None


def table_one_logged(args):
    lg = {}
    return table_one(args, lg), lg


def rep(args):
    """One Monte Carlo replication: new unobservables, same exogenous design."""
    T, i, sp = args
    d = DESIGNS[T]
    tru = dict(TRUE, sp=sp)
    rng = np.random.default_rng(31_000 + 1000 * T + i + (0 if sp == 0.4 else 700_000))
    while True:
        r = simulate_market(d, tru, "bertrand", rng)
        if r["ok"]:
            break
    s, p = r["s"], r["p"]
    tr = structural(d, s, p, (tru["alpha"], tru["sp"], tru["sx"]), "bertrand")
    S = summaries(d, s, p, noisy_micro(r, rng))
    t0 = time.time()
    try:
        g = BLPGMM(d, s, p).fit()
        th, cov = g.th2, g.cov
        ok = np.all(np.isfinite(cov)) and np.all(np.diag(cov) > 0)
    except Exception:
        th, cov, ok = np.full(3, np.nan), np.full((3, 3), np.nan), False
    est = []
    plug = np.full(3, np.nan)
    if ok:
        dr = np.abs(rng.multivariate_normal(th, cov, N_KR))
        dl = None
        for x in dr:
            q = structural(d, s, p, x, "bertrand", dl)
            dl = q["delta"]
            if q["ok"]:
                est.append(estimands(q))
        q = structural(d, s, p, np.abs(th), "bertrand")       # the GMM plug in point prediction
        if q["ok"]:
            plug = estimands(q)
    return dict(T=T, i=i, sp=sp, S=S, truth=estimands(tr), gmm_th=th, gmm_cov=cov, gmm_ok=ok,
                gmm_est=np.array(est), gmm_plug=plug, secs=time.time() - t0)


def misspecification(sims, rng):
    """Data generated under joint monopoly and under perfect competition. BLP GMM with an
    assumed Bertrand Nash supply side versus GBC with conduct as a parameter."""
    d = DESIGNS[40]
    G = GBC(sims)
    out = {}
    for conduct in ["collusion", "perfect"]:
        while True:
            r = simulate_market(d, TRUE, conduct, rng)
            if r["ok"]:
                break
        s, p = r["s"], r["p"]
        truth = estimands(structural(d, s, p, (TRUE["alpha"], TRUE["sp"], TRUE["sx"]), conduct))
        g = BLPGMM(d, s, p).fit()
        gm = estimands(structural(d, s, p, g.th2, "bertrand"))
        S = summaries(d, s, p, noisy_micro(r, rng))
        th, cs, pc = G.sample(S, 150, rng)
        ia = [PARAMS.index(k) for k in ("alpha", "sp", "sx")]
        ge, dl = [], None
        for x, c in zip(np.clip(th[:, ia], 1e-4, None), cs):
            q = structural(d, s, p, x, CONDUCTS[c], dl)
            dl = q["delta"]
            if q["ok"]:
                ge.append(estimands(q))
        out[conduct] = dict(truth=truth, gmm_th=g.th2, gmm_bertrand=gm, gbc_pc=pc, gbc_est=np.array(ge))
        print(conduct, "truth", truth.round(3), "gmm+bertrand", gm.round(3), "gbc pc", pc.round(3),
              "gbc", np.array(ge).mean(0).round(3))
    return out


GPART = None


def partial_rep(i):
    """One industry with partial rival internalization (kappa = 0.5): H = 1 within owner, 0.5 across owners."""
    d = DESIGNS[40]
    rng = np.random.default_rng(61_000 + i)
    while True:
        r = simulate_market(d, TRUE, "partial", rng)
        if r["ok"]:
            break
    s, p = r["s"], r["p"]
    th2 = (TRUE["alpha"], TRUE["sp"], TRUE["sx"])
    truth = estimands(structural(d, s, p, th2, "partial"))
    truth_b = estimands(structural(d, s, p, th2, "bertrand"))       # what a Bertrand analyst would get at the true theta2
    try:
        g = BLPGMM(d, s, p).fit(); gth = g.th2
        gm = estimands(structural(d, s, p, np.abs(gth), "bertrand"))
    except Exception:
        gth, gm = np.full(3, np.nan), np.full(3, np.nan)
    S = summaries(d, s, p, noisy_micro(r, rng))
    th, cs, pc = GPART.sample(S, 100, rng)
    ia = [PARAMS.index(k) for k in ("alpha", "sp", "sx")]
    ge, dl = [], None
    for x, c in zip(np.clip(th[:, ia], 1e-4, None), cs):
        q = structural(d, s, p, x, CONDUCTS[c], dl)
        dl = q["delta"]
        if q["ok"]:
            ge.append(estimands(q))
    return dict(i=i, truth=truth, truth_bertrand=truth_b, gmm_th=gth, gmm_bertrand=gm, gbc_pc=pc, gbc_est=np.array(ge))


if __name__ == "__main__":
    stage = sys.argv[1]
    os.makedirs("results/mc_parts", exist_ok=True)
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    if stage == "table":
        res = {}
        with Pool(ncpu) as pool:
            for T in [40, 15]:
                t0 = time.time()
                out = pool.map(table_one_logged, [(T, i) for i in range(6000)], chunksize=25)
                tab = [x[0] for x in out if x[0] is not None]
                lg = merge_logs([x[1] for x in out]); lg["jobs"] = 6000; lg["final count"] = len(tab)
                lg["acceptance rate per attempt"] = lg["accepted"] / lg["attempts"]
                res[T] = dict(Y=np.array([x[0] for x in tab]), S=np.array([x[1] for x in tab]), log=lg)
                print(T, "table", len(tab), f"{time.time() - t0:.0f}s", lg, flush=True)
        pickle.dump(res, open("results/mc_table.pkl", "wb"))
        json.dump({str(T): res[T]["log"] for T in res}, open("results/acceptance_mc_table.json", "w"), indent=1)
    elif stage == "reps":
        T, sp, lo, hi = int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        t0 = time.time()
        with Pool(ncpu) as pool:
            reps = pool.map(rep, [(T, i, sp) for i in range(lo, hi)], chunksize=1)
        pickle.dump(reps, open(f"results/mc_parts/reps_T{T}_sp{sp}_{lo}.pkl", "wb"))
        print(T, sp, lo, hi, "done", f"{time.time() - t0:.0f}s", flush=True)
    elif stage == "agg":
        sp = float(sys.argv[2])
        tab = pickle.load(open("results/mc_table.pkl", "rb"))
        res = {}
        for T in [40, 15]:
            reps = []
            for f in sorted(glob.glob(f"results/mc_parts/reps_T{T}_sp{sp}_*.pkl"), key=lambda x: int(x.split("_")[-1][:-4])):
                reps += pickle.load(open(f, "rb"))
            Y, S = tab[T]["Y"], tab[T]["S"]
            gt = GBCTargets(S, Y); gn = GBCTargets(S[:, :-2], Y)
            for rr in reps:
                rr["gbc"] = gt.posterior(rr["S"])[:, IA]
                rr["gbc_nomicro"] = gn.posterior(rr["S"][:-2])[:, IA]
            res[T] = dict(reps=reps, Y=Y, S=S)
            print(T, "reps", len(reps), flush=True)
        if sp == 0.4:
            sims = pickle.load(open("results/reference_table.pkl", "rb"))
            res["misspec"] = misspecification(sims, np.random.default_rng(4040))
        with open(f"results/mc{SPTAG[sp]}.pkl", "wb") as f:
            pickle.dump(res, f)
    elif stage == "partial":
        GPART = GBC(pickle.load(open("results/reference_table.pkl", "rb")))
        t0 = time.time()
        with Pool(ncpu) as pool:
            out = pool.map(partial_rep, range(100), chunksize=1)
        pickle.dump(out, open("results/mc_partial.pkl", "wb"))
        print("partial done", len(out), f"{time.time() - t0:.0f}s", flush=True)
