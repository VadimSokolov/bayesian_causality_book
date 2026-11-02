"""Run every estimator on the minivan style dataset, compute posterior markups, conduct,
diversion, GUPPI and merger effects, and a calibration study for GBC.
Writes results/experiments.pkl"""
import pickle, time, numpy as np
from blp_core import (Design, simulate_market, contraction, ind_probs, alpha_i, markups,
                      price_derivs, solve_equilibrium, consumer_surplus, CONDUCTS)
from estimators import (logit_iv, BLPGMM, jmr_mcmc, GBC, summaries, draw_prior, PARAMS, flat)

D = Design(seed=0)
TRUE = dict(b0=2.0, b1=1.0, b2=0.5, alpha=1.0, sp=0.4, sx=0.7, g0=2.0, g1=0.5, g2=0.3,
            sxi=0.5, somega=0.3, rho=0.3)
POST_FIRMS = (0, 0, 0, 0, 2, 2)     # firms A and B merge
MERGING = [0, 1, 2, 3]
rng = np.random.default_rng(2026)
data = simulate_market(D, TRUE, "bertrand", rng)
s, p = data["s"], data["p"]
micro = data["micro"] + rng.normal(0, [0.03, 0.10])
S_obs = summaries(D, s, p, micro)


def structural(th2, conduct, dl0=None):
    """Everything antitrust needs from (alpha, sigma_p, sigma_x, conduct) and the data:
    markups, marginal cost, Lerner index, diversion, GUPPI, merger price and welfare effects."""
    a, sp, sx = th2
    dl = contraction(s, p, D, a, sp, sx, delta0=dl0, tol=1e-11)
    ai = alpha_i(a, sp, D.nu)
    sij = ind_probs(dl, p, D, a, sp, sx)
    H = D.ownership(conduct)
    m = markups(s, sij, ai, H)
    c = p - m
    Dm, _, _ = price_derivs(sij, ai)
    own = np.diagonal(Dm, axis1=1, axis2=2)
    div = -np.transpose(Dm, (0, 2, 1)) / own[:, :, None]          # div[t,j,k] = -(ds_k/dp_j)/(ds_j/dp_j)
    div[:, range(D.J), range(D.J)] = 0.0
    # GUPPI against the merger partner, using the conduct implied margins
    guppi = np.zeros_like(p)
    for j in MERGING:
        partners = [k for k in MERGING if D.firms[k] != D.firms[j]]
        guppi[:, j] = (div[:, j, partners] * m[:, partners]).sum(1) / p[:, j]
    # merger: conduct maintained, merged firm internalises all four products
    if conduct == "bertrand":
        Hp = D.ownership("bertrand", POST_FIRMS)
        p1, ok = solve_equilibrium(dl, c, D, a, sp, sx, Hp, p0=p)
    else:
        p1, ok = p.copy(), True                 # collusion and perfect competition: no change
    cs0 = consumer_surplus(dl, p, D, a, sp, sx)
    cs1 = consumer_surplus(dl, p1, D, a, sp, sx)
    elas = Dm * p[:, None, :] / s[:, :, None]   # elas[t,j,k] = ds_j/dp_k * p_k / s_j
    return dict(m=m, c=c, lerner=m / p, div=div, guppi=guppi, dp=(p1 - p) / p, dcs=cs1 - cs0,
                elas=elas, ok=ok, delta=dl)


def summarise_draws(draws, conducts, label, ndraw=250):
    idx = np.linspace(0, len(draws) - 1, min(ndraw, len(draws))).astype(int)
    res = []
    dl = None
    t0 = time.time()
    for i in idx:
        th2 = draws[i]
        r = structural(th2, CONDUCTS[conducts[i]], dl)
        dl = r.pop("delta")
        if r["ok"]:
            r["th2"] = th2; r["conduct"] = conducts[i]
            res.append(r)
    print(f"{label}: {len(res)} structural draws in {time.time() - t0:.0f}s")
    out = {k: np.array([r[k] for r in res]) for k in res[0]}
    return out


if __name__ == "__main__":
    R = dict(data=data, S_obs=S_obs, true=TRUE, micro=micro)
    # ---- truth
    tr = structural((TRUE["alpha"], TRUE["sp"], TRUE["sx"]), "bertrand")
    tr.pop("delta")
    R["truth"] = tr
    # ---- logit IV with robust standard error, Krinsky Robb draws
    lg = logit_iv(D, s, p)
    Xf = np.column_stack([flat(D.X), -flat(p)]); Zf = flat(D.Z)
    PZ = Zf @ np.linalg.pinv(Zf.T @ Zf) @ Zf.T; Xh = PZ @ Xf
    A = np.linalg.inv(Xh.T @ Xh)
    V = A @ (Xh * lg["resid"][:, None] ** 2).T @ Xh @ A
    se_a = np.sqrt(V[3, 3])
    a_dr = np.abs(rng.normal(lg["alpha"], se_a, 250))
    R["logit"] = dict(alpha=lg["alpha"], se=se_a,
                      draws=summarise_draws(np.column_stack([a_dr, 0 * a_dr, 0 * a_dr]),
                                            np.zeros(250, int), "logit"))
    # ---- BLP GMM with Krinsky Robb draws from the sandwich covariance
    t0 = time.time()
    g = BLPGMM(D, s, p).fit()
    print("BLP GMM", g.th2, np.sqrt(np.diag(g.cov)), f"{time.time() - t0:.0f}s")
    gd = rng.multivariate_normal(g.th2, g.cov, 250)
    gd = np.abs(gd)
    R["gmm"] = dict(th2=g.th2, cov=g.cov, J=g.J_stat,
                    draws=summarise_draws(gd, np.zeros(250, int), "gmm"))
    # ---- Bayesian inversion likelihood MCMC
    t0 = time.time()
    ch, acc = jmr_mcmc(D, s, p, n_iter=5000, burn=1500)
    print("JMR", ch.mean(0), ch.std(0), acc, f"{time.time() - t0:.0f}s")
    R["jmr"] = dict(chain=ch, acc=acc, draws=summarise_draws(ch, np.zeros(len(ch), int), "jmr"))
    # ---- GBC with and without Petrin micro moments
    sims = pickle.load(open("results/reference_table.pkl", "rb"))
    for name, um in [("gbc", True), ("gbc_nomicro", False)]:
        G = GBC(sims, use_micro=um)
        th, cs, pc = G.sample(S_obs, 2000, rng)
        ia = [PARAMS.index(k) for k in ("alpha", "sp", "sx")]
        th2 = np.clip(th[:, ia], 1e-4, None)
        R[name] = dict(theta=th, conduct=cs, pc=pc, draws=summarise_draws(th2, cs, name))
        print(name, "conduct probs", pc.round(4), "theta2 mean", th2.mean(0), th2.std(0))
        if um:
            Gmain = G
    # ---- teaching data: markup regression coefficients in the reference table
    Sref = np.array([x["S"] for x in sims])
    Cref = np.array([CONDUCTS.index(x["conduct"]) for x in sims])
    Tref = np.array([[x["theta"][k] for k in PARAMS] for x in sims])
    R["ref"] = dict(S=Sref, C=Cref, T=Tref)
    # ---- calibration on fresh simulations (amortisation makes this cheap)
    cal = dict(cover_naive=[], cover=[], conduct_true=[], conduct_pred=[], probs=[], theta=[], q=[])
    crng = np.random.default_rng(77)
    for k in range(450):
        conduct = CONDUCTS[k % 3]
        th0 = draw_prior(crng)
        r = simulate_market(D, th0, conduct, crng)
        if not r["ok"]:
            continue
        So = summaries(D, r["s"], r["p"], r["micro"] + crng.normal(0, [0.03, 0.10]))
        pc = Gmain.conduct_probs(So)
        c = CONDUCTS.index(conduct)
        dr, w = Gmain.posterior(So, c)
        tv = np.array([th0[x] for x in PARAMS])
        o = np.argsort(dr, axis=0)
        lo, hi = [], []
        for j in range(dr.shape[1]):
            cw = np.cumsum(w[o[:, j]])
            lo.append(dr[o[np.searchsorted(cw, 0.05), j], j])
            hi.append(dr[o[min(np.searchsorted(cw, 0.95), len(cw) - 1), j], j])
        qq = np.array([(w * (dr[:, j] <= tv[j])).sum() for j in range(len(tv))])
        cal["cover"].append((tv >= np.array(lo)) & (tv <= np.array(hi)))
        drn, wn = Gmain.posterior(So, c, cross_fit=False)
        lon, hin = [], []
        for j in range(drn.shape[1]):
            oo = np.argsort(drn[:, j]); cw = np.cumsum(wn[oo])
            lon.append(drn[oo[np.searchsorted(cw, 0.05)], j])
            hin.append(drn[oo[min(np.searchsorted(cw, 0.95), len(cw) - 1)], j])
        cal["cover_naive"].append((tv >= np.array(lon)) & (tv <= np.array(hin)))
        cal["conduct_true"].append(c); cal["conduct_pred"].append(pc.argmax()); cal["probs"].append(pc)
        cal["theta"].append(tv); cal["q"].append(qq)
    R["calibration"] = {k: np.array(v) for k, v in cal.items()}
    print("coverage", R["calibration"]["cover"].mean(0).round(2))
    print("conduct accuracy", (R["calibration"]["conduct_true"] == R["calibration"]["conduct_pred"]).mean())
    with open("results/experiments.pkl", "wb") as f:
        pickle.dump(R, f)
