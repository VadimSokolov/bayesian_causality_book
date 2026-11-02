"""Second round of GBC: a local reference table drawn from the prior truncated to a box that
covers the first round posterior, with the antitrust estimands (Lerner index, merger price
effect, consumer surplus change) recorded in every simulation. The posterior of these
estimands is then learned directly, without passing through a plug in of parameters."""
import pickle, numpy as np
from multiprocessing import Pool
from blp_core import (Design, simulate_market, solve_equilibrium, consumer_surplus, CONDUCTS, tally, merge_logs)
from estimators import PARAMS, PRIOR, summaries
from run_experiments import POST_FIRMS, MERGING, structural, summarise_draws

D = Design(seed=0)
R = pickle.load(open("results/experiments.pkl", "rb"))
th1 = R["gbc"]["theta"][R["gbc"]["conduct"] == 0]
lo, hi = np.percentile(th1, 0.5, axis=0), np.percentile(th1, 99.5, axis=0)
pad = 0.5 * (hi - lo)
BOX = {k: (max(PRIOR[k][0], lo[i] - pad[i]), min(PRIOR[k][1], hi[i] + pad[i])) for i, k in enumerate(PARAMS)}


def one(i, log=None):
    rng = np.random.default_rng(500_000 + i)
    for _ in range(20):
        tally(log, "attempts")
        th = {k: rng.uniform(*BOX[k]) for k in PARAMS}
        r = simulate_market(D, th, "bertrand", rng)
        if not r["ok"]:
            tally(log, "rejected: " + r["why"])
            continue
        Hp = D.ownership("bertrand", POST_FIRMS)
        p1, ok = solve_equilibrium(r["delta"], r["c"], D, th["alpha"], th["sp"], th["sx"], Hp, p0=r["p"])
        if not ok:
            tally(log, "rejected: post merger solve")
            continue
        cs0 = consumer_surplus(r["delta"], r["p"], D, th["alpha"], th["sp"], th["sx"])
        cs1 = consumer_surplus(r["delta"], p1, D, th["alpha"], th["sp"], th["sx"])
        est = np.array([((r["p"] - r["c"]) / r["p"]).mean(),
                        ((p1 - r["p"]) / r["p"])[:, MERGING].mean(),
                        (cs1 - cs0).mean()])
        S = summaries(D, r["s"], r["p"], r["micro"] + rng.normal(0, [0.03, 0.10]))
        if np.all(np.isfinite(S)):
            tally(log, "accepted")
            return dict(theta=th, S=S, est=est)
        tally(log, "rejected: non finite summary")
    tally(log, "jobs at the 20 try cap")
    return None


def one_logged(i):
    lg = {}
    return one(i, lg), lg


if __name__ == "__main__":
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    with Pool() as pool:
        out = pool.map(one_logged, range(8000), chunksize=20)
    sims = [x[0] for x in out if x[0] is not None]
    import json
    lg = merge_logs([x[1] for x in out]); lg["jobs"] = 8000; lg["final count"] = len(sims)
    lg["acceptance rate per attempt"] = lg["accepted"] / lg["attempts"]
    json.dump(lg, open("results/acceptance_gbc_refine.json", "w"), indent=1); print(lg)
    print(len(sims), "local simulations")
    S = np.array([x["S"] for x in sims])
    Y = np.column_stack([np.array([[x["theta"][k] for k in PARAMS] for x in sims]),
                         np.array([x["est"] for x in sims])])
    out = {}
    for name, cols in [("gbc2", slice(None)), ("gbc2_nomicro", slice(0, -2))]:
        Sx = S[:, cols]
        so = R["S_obs"][cols]
        ss, sy = StandardScaler().fit(Sx), StandardScaler().fit(Y)
        Ssx, Sy = ss.transform(Sx), sy.transform(Y)
        folds = np.arange(len(Sx)) % 3
        fit = np.zeros_like(Sy); nets = []
        for f in range(3):
            net = MLPRegressor(hidden_layer_sizes=(256, 256, 128), alpha=1e-3, max_iter=2500,
                               early_stopping=True, random_state=f).fit(Ssx[folds != f], Sy[folds != f])
            fit[folds == f] = net.predict(Ssx[folds == f]); nets.append(net)
        f0 = np.mean([n_.predict(ss.transform(so[None]))[0] for n_ in nets], axis=0)
        dist = np.sqrt(((fit - f0) ** 2).sum(1))
        keep = np.argsort(dist)[:1500]
        U = sy.transform(Y)[keep] - fit[keep]
        draws = sy.inverse_transform(f0[None] + U)
        ia = [PARAMS.index(k) for k in ("alpha", "sp", "sx")]
        th2 = np.clip(draws[:, ia], 1e-4, None)
        out[name] = dict(theta=draws[:, :len(PARAMS)], est=draws[:, len(PARAMS):],
                         draws=summarise_draws(th2, np.zeros(len(th2), int), name))
        print(name, th2.mean(0).round(3), th2.std(0).round(3), "estimands",
              draws[:, -3:].mean(0).round(4), draws[:, -3:].std(0).round(4))
    out["box"] = BOX
    tr = R["truth"]
    print("truth estimands", tr["lerner"].mean(), tr["dp"][:, MERGING].mean(), tr["dcs"].mean())
    R.update(out)
    with open("results/experiments.pkl", "wb") as f:
        pickle.dump(R, f)
