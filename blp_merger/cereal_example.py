"""Antitrust example: a ready to eat cereal style industry with a dominant firm, a merger of
the second and third firms (in the spirit of the Post and Nabisco case studied by Nevo 2000),
and a divestiture remedy. Compares GBC with BLP GMM and logit IV on the merger effect, the
remedy and the challenge decision. Writes results/cereal.pkl and results/numbers_cereal.tex
and the figure figures/fig_cereal.pdf.

Industry: J = 8 brands, owners (A, A, A, B, B, C, C, D), T = 30 city markets.
Merger: B and C merge. Remedy: the merged firm divests brand C2 to the fringe firm D."""
import sys, pickle, time, numpy as np
from multiprocessing import Pool
from blp_core import (Design, simulate_market, solve_equilibrium, consumer_surplus, contraction,
                      ind_probs, alpha_i, markups, tally, merge_logs)
from estimators import (BLPGMM, GBCTargets, PARAMS, draw_prior, summaries, logit_iv, flat)

FIRMS = (0, 0, 0, 1, 1, 2, 2, 3)
POST = (0, 0, 0, 1, 1, 1, 1, 3)
REMEDY = (0, 0, 0, 1, 1, 1, 3, 3)
MERGING = [3, 4, 5, 6]
D = Design(T=30, J=8, firms=FIRMS, ns=200, seed=11)
TRUE = dict(b0=1.2, b1=1.0, b2=0.5, alpha=1.0, sp=0.4, sx=0.7, g0=2.0, g1=0.5, g2=0.3,
            sxi=0.5, somega=0.3, rho=0.3)
BRANDS = ["A1", "A2", "A3", "B1", "B2", "C1", "C2", "D1"]


def scenario_effects(delta, c, p, th2):
    """Price effects and surplus changes under the merger and under the remedy."""
    a, sp, sx = th2
    out, cs0 = [], consumer_surplus(delta, p, D, a, sp, sx)
    for firms in (POST, REMEDY):
        p1, ok = solve_equilibrium(delta, c, D, a, sp, sx, D.ownership("bertrand", firms), p0=p)
        if not ok:
            return None
        out.append(((p1 - p) / p))
        out.append(consumer_surplus(delta, p1, D, a, sp, sx) - cs0)
    dpm, dcsm, dpr, dcsr = out
    return dpm, dcsm, dpr, dcsr


def estimand_vector(lerner, eff):
    dpm, dcsm, dpr, dcsr = eff
    return np.array([lerner.mean(), dpm[:, MERGING].mean(), dpr[:, MERGING].mean(),
                     dcsm.mean(), dcsr.mean()])


def table_one(i, log=None):
    rng = np.random.default_rng(700_000 + i)
    for _ in range(30):
        tally(log, "attempts")
        th = draw_prior(rng)
        r = simulate_market(D, th, "bertrand", rng)
        if not r["ok"]:
            tally(log, "rejected: " + r["why"])
            continue
        eff = scenario_effects(r["delta"], r["c"], r["p"], (th["alpha"], th["sp"], th["sx"]))
        if eff is None:
            tally(log, "rejected: post merger solve")
            continue
        S = summaries(D, r["s"], r["p"], r["micro"] + rng.normal(0, [0.03, 0.10]))
        if np.all(np.isfinite(S)):
            e = estimand_vector((r["p"] - r["c"]) / r["p"], eff)
            tally(log, "accepted")
            return np.concatenate([[th[k] for k in PARAMS], e]), S
        tally(log, "rejected: non finite summary")
    tally(log, "jobs at the 30 try cap")
    return None


def table_one_logged(i):
    lg = {}
    return table_one(i, lg), lg


def plug_in(s, p, th2, dl=None):
    """Recover delta and costs from the data under Bertrand, then run both scenarios."""
    a, sp, sx = th2
    dl = contraction(s, p, D, a, sp, sx, delta0=dl, tol=1e-11)
    sij = ind_probs(dl, p, D, a, sp, sx)
    m = markups(s, sij, alpha_i(a, sp, D.nu), D.ownership("bertrand"))
    eff = scenario_effects(dl, p - m, p, th2)
    if eff is None:
        return None, dl
    return estimand_vector(m / p, eff), dl, eff


if __name__ == "__main__":
    t0 = time.time()
    with Pool() as pool:
        out = pool.map(table_one_logged, range(6000), chunksize=25)
    tab = [x[0] for x in out if x[0] is not None]
    import json
    lg = merge_logs([x[1] for x in out]); lg["jobs"] = 6000; lg["final count"] = len(tab)
    lg["acceptance rate per attempt"] = lg["accepted"] / lg["attempts"]
    json.dump(lg, open("results/acceptance_cereal.json", "w"), indent=1); print(lg)
    Y = np.array([x[0] for x in tab]); S = np.array([x[1] for x in tab])
    print("table", len(tab), f"{time.time() - t0:.0f}s")
    rng = np.random.default_rng(1993)
    while True:
        r = simulate_market(D, TRUE, "bertrand", rng)
        if r["ok"]:
            break
    s, p = r["s"], r["p"]
    eff_true = scenario_effects(r["delta"], r["c"], p, (TRUE["alpha"], TRUE["sp"], TRUE["sx"]))
    truth = estimand_vector((p - r["c"]) / p, eff_true)
    S_obs = summaries(D, s, p, r["micro"] + rng.normal(0, [0.03, 0.10]))
    gt = GBCTargets(S, Y)
    post = gt.posterior(S_obs)
    gbc_est = post[:, 12:]
    ia = [PARAMS.index(k) for k in ("alpha", "sp", "sx")]
    gbc_th = post[:, ia]
    # brand level effects from GBC parameter draws (plug in through Propositions 1 and 2)
    brand, dl = [], None
    for th2 in np.clip(gbc_th[::10], 1e-4, None):
        res = plug_in(s, p, th2, dl)
        if res[0] is not None:
            dl = res[1]; brand.append(res[2][0].mean(0))
    brand = np.array(brand)
    # BLP GMM with Krinsky and Robb
    g = BLPGMM(D, s, p).fit()
    kr, dl = [], None
    for th2 in np.abs(rng.multivariate_normal(g.th2, g.cov, 150)):
        res = plug_in(s, p, th2, dl)
        if res[0] is not None:
            dl = res[1]; kr.append(res[0])
    kr = np.array(kr)
    gmm_point = plug_in(s, p, g.th2)[0]
    # logit IV
    lg = logit_iv(D, s, p)
    lgt = plug_in(s, p, (lg["alpha"], 0.0, 0.0))[0]
    shares = s.mean(0)
    out = dict(truth=truth, gbc_est=gbc_est, gbc_th=gbc_th, brand=brand, brand_true=eff_true[0].mean(0),
               gmm_th=g.th2, gmm_cov=g.cov, gmm_kr=kr, gmm_point=gmm_point, logit=lgt,
               logit_alpha=lg["alpha"], shares=shares, prices=p.mean(0))
    pickle.dump(out, open("results/cereal.pkl", "wb"))
    print("truth", truth.round(4)); print("gbc", gbc_est.mean(0).round(4), gbc_est.std(0).round(4))
    print("gmm", gmm_point.round(4), "kr sd", kr.std(0).round(4)); print("logit", lgt.round(4))
    print("shares", shares.round(3), "firm", [shares[[0,1,2]].sum(), shares[[3,4]].sum(), shares[[5,6]].sum(), shares[7]])
