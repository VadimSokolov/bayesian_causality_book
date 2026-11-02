"""Build the GBC reference table: draws of (theta, conduct) from the prior, equilibrium data,
and summary statistics. Usage: python simulate_reference_table.py N_PER_CONDUCT
Also writes results/acceptance_reference_table.json: per conduct, the number of attempts, the
rejections by reason (solver, price bound, share floor, inside share, non finite summary), the
number of jobs that hit the 20 try cap, and the final count (added in the October 2026 rerun; the
random streams are unchanged, so the table is the same as before)."""
import sys, json, pickle, numpy as np
from multiprocessing import Pool
from blp_core import Design, simulate_market, CONDUCTS, tally, merge_logs
from estimators import draw_prior, summaries

D = Design(seed=0)


def one(args, log=None):
    i, conduct = args
    rng = np.random.default_rng(10_000 + i)
    for _ in range(20):
        tally(log, "attempts")
        th = draw_prior(rng)
        r = simulate_market(D, th, conduct, rng)
        if r["ok"]:
            S = summaries(D, r["s"], r["p"], r["micro"] + rng.normal(0, [0.03, 0.10]))
            if np.all(np.isfinite(S)):
                tally(log, "accepted")
                return dict(theta=th, conduct=conduct, S=S)
            tally(log, "rejected: non finite summary")
        else:
            tally(log, "rejected: " + r["why"])
    tally(log, "jobs at the 20 try cap")
    return None


def one_logged(args):
    lg = {}
    return one(args, lg), lg, args[1]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    jobs = [(k * n + i, c) for k, c in enumerate(CONDUCTS) for i in range(n)]
    with Pool() as pool:
        out = pool.map(one_logged, jobs, chunksize=20)
    sims = [x[0] for x in out if x[0] is not None]
    acc = {}
    for c in CONDUCTS:
        lg = merge_logs([x[1] for x in out if x[2] == c])
        lg["jobs"] = n; lg["final count"] = sum(1 for x in out if x[2] == c and x[0] is not None)
        lg["acceptance rate per attempt"] = lg.get("accepted", 0) / lg["attempts"]
        acc[c] = lg
    json.dump(acc, open("results/acceptance_reference_table.json", "w"), indent=1)
    print(json.dumps(acc, indent=1))
    with open("results/reference_table.pkl", "wb") as f:
        pickle.dump(sims, f)
    print(len(sims), "simulations saved")
