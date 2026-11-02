"""Castle doctrine fit with four independent chains, convergence checks, and an AR(1) error variant (rerun r1023, group C, ch15 F7).

Design.  The panel of did_hs.py (castle doctrine, 50 states, 2000 to 2010, outcome the log homicide rate) and its design matrix
(unit and period flat effects, flat event time path, horseshoe cohort by event time deviations, horseshoe leads at -5 to -2, horseshoe
unit linear trends, each block with its own global scale).  Same Gibbs sampler as group_hs of did_hs.py, rewritten as a chain with state so
that it can be continued.  Each chain starts from an over-dispersed random state (log local scales N(0, 1.5^2), log global scales
N(log 0.1, 1.5^2), error variance var(y) exp(N(0, 1)), rho uniform on (-0.5, 0.5) in the AR(1) variant) and has its own generator
(SeedSequence(20261023).spawn(8)); the quantity recorded at each sweep is the average effect on the treated (the cell weighted mean of
path plus deviation, as att_from), the ten event time effects of the chapter's event study (0 to 5 and the leads -5 to -2) and, for AR(1), rho.
(a) independent errors: four chains of 20,000 sweeps, the first 5,000 of each discarded; the chains are then extended by 20,000 sweeps
each, and so on, until the Monte Carlo standard error of Pr(ATT > 0), from the effective sample size of the indicator, is below 0.005
(cap 400,000 sweeps per chain).  Reported at 20,000 sweeps and at the end: posterior mean, 95 percent interval, Pr(ATT > 0), split R-hat,
bulk effective sample size and Monte Carlo standard errors (mean, interval endpoints, probability).
(b) AR(1) errors within state: e_i ~ N(0, s2 R_rho), (R_rho)_{ts} = rho^|t - s|, uniform prior on rho over (-1, 1).  The coefficient update
uses the whitened design (Prais Winsten, first period unchanged and later periods (z_t - rho z_{t-1})/sqrt(1 - rho^2)); rho has a griddy
Gibbs step on 2,001 points of (-1, 1) with the determinant term (1 - rho^2)^{-N(T-1)/2} kept; four chains, same rules, cap 100,000 sweeps.
Writes results/res_did_castle2.json, results/did_castle2_draws.npz (ATT, event and rho draws after burn in) and figs/castle_4chain.pdf.
Run on Hopper (about 20 to 60 minutes), 1 thread:  python book_scripts/did_castle2.py [indep|ar1]   (default both, one file each:
results/res_did_castle2_indep.json, results/res_did_castle2_ar1.json; merge with --merge into res_did_castle2.json).
"""
import sys, json, os
import numpy as np, pandas as pd

SRC = "book_scripts/did_hs.py"; MARK = "# ---------------- Simulation"
src = open(SRC).read(); assert src.count(MARK) == 1
exec(compile(src.split(MARK)[0], SRC, "exec"))     # imports, group_hs, build, att_from, twfe, cs_att

d = castle.load_pandas().data.copy()
first = d[d.post > 0].groupby("sid").year.min()
d["cohort"] = d.sid.map(first).fillna(0).astype(int)
d = d.sort_values(["sid", "year"]).reset_index(drop=True)
unit = d.sid.values; year = d.year.values; y = d.l_homicide.values.astype(float); coh = d.cohort.values
N = len(np.unique(unit)); T = len(np.unique(year)); assert len(d) == N * T
ev = np.where(coh > 0, year - coh, -999)
Z, grp, names, _, cells = build(unit, year, coh, y, leads=(-5, -4, -3, -2))
pen = grp >= 0; G = grp.max() + 1; q = Z.shape[1]; n = len(y)
pidx = {nm[1]: i for i, nm in enumerate(names) if nm[0] == "path"}
didx = {nm[1]: i for i, nm in enumerate(names) if nm[0] == "dev"}
w_cell = np.array([np.sum((coh == c) & (ev == e)) for (c, e) in cells], float)
p_i = np.array([pidx[e] for (c, e) in cells]); d_i = np.array([didx[(c, e)] for (c, e) in cells])
ev_cells = {e: [k for k, (c, ee) in enumerate(cells) if ee == e] for e in range(0, 6)}
EV_AVAIL = [e for e in range(0, 6) if ev_cells[e]]
lead_i = {l: [i for i, nm in enumerate(names) if nm == ("lead", l)][0] for l in (-5, -4, -3, -2)}
GRID = np.linspace(-1, 1, 2003)[1:-1]; DR = GRID[1] - GRID[0]


def functionals(b):
    att = float(((b[p_i] + b[d_i]) * w_cell).sum() / w_cell.sum())
    es = []
    for e in EV_AVAIL:
        idx = ev_cells[e]; w = w_cell[idx]; es.append(float(b[pidx[e]] + (b[d_i[idx]] * w).sum() / w.sum()))
    es += [float(b[lead_i[l]]) for l in (-5, -4, -3, -2)]
    return att, es


class Chain:
    def __init__(self, seed, ar):
        self.rng = np.random.default_rng(seed); self.ar = ar; r = self.rng
        self.lam2 = np.exp(r.normal(0, 1.5, q)); self.nu = np.ones(q)
        self.tau2 = np.exp(r.normal(np.log(0.1), 1.5, G)); self.xi = np.ones(G); self.s2 = np.var(y) * np.exp(r.normal())
        self.rho = r.uniform(-0.5, 0.5) if ar else 0.0
        self.Z3 = Z.reshape(N, T, q); self.y2 = y.reshape(N, T)
        if not ar:
            self.ZtZ = Z.T @ Z; self.Zty = Z.T @ y; self.yty = y @ y
        self.att, self.es, self.rhos = [], [], []

    def whiten(self, rho):
        c = 1 / np.sqrt(1 - rho ** 2)
        Zs = self.Z3.copy(); Zs[:, 1:] = (self.Z3[:, 1:] - rho * self.Z3[:, :-1]) * c
        ys = self.y2.copy(); ys[:, 1:] = (self.y2[:, 1:] - rho * self.y2[:, :-1]) * c
        return Zs.reshape(n, q), ys.reshape(n)

    def sweep(self):
        r = self.rng
        if self.ar:
            Zs, ys = self.whiten(self.rho); ZtZ = Zs.T @ Zs; Zty = Zs.T @ ys; yty = ys @ ys
        else:
            ZtZ, Zty, yty = self.ZtZ, self.Zty, self.yty
        lam2, nu, tau2, xi = self.lam2, self.nu, self.tau2, self.xi
        pv = np.where(pen, lam2 * tau2[np.maximum(grp, 0)], np.inf)
        A = ZtZ + np.diag(np.where(pen, 1 / pv, 1e-8)); L = np.linalg.cholesky(A)
        b = np.linalg.solve(A, Zty) + np.sqrt(self.s2) * np.linalg.solve(L.T, r.normal(size=q))
        rss = yty - 2 * b @ Zty + b @ ZtZ @ b
        bp = b[pen]; tp = tau2[grp[pen]]
        s2 = 1 / r.gamma((n + pen.sum()) / 2, 1 / (rss / 2 + (bp ** 2 / (lam2[pen] * tp)).sum() / 2))
        lam2[pen] = 1 / r.gamma(1, 1 / (1 / nu[pen] + bp ** 2 / (2 * tp * s2)))
        nu[pen] = 1 / r.gamma(1, 1 / (1 + 1 / lam2[pen]))
        for g in range(G):
            s = grp == g
            tau2[g] = 1 / r.gamma((s.sum() + 1) / 2, 1 / (1 / xi[g] + (b[s] ** 2 / lam2[s]).sum() / (2 * s2)))
            xi[g] = 1 / r.gamma(1, 1 / (1 + 1 / tau2[g]))
        self.s2 = s2
        if self.ar:                                       # griddy step for rho given b and s2, uniform prior on (-1, 1)
            e = (self.y2 - self.Z3 @ b)
            A0 = (e[:, 1:] ** 2).sum(); B0 = (e[:, 1:] * e[:, :-1]).sum(); C0 = (e[:, :-1] ** 2).sum(); e1 = (e[:, 0] ** 2).sum()
            Q = e1 + (A0 - 2 * GRID * B0 + GRID ** 2 * C0) / (1 - GRID ** 2)
            lp = -0.5 * N * (T - 1) * np.log(1 - GRID ** 2) - Q / (2 * s2)
            pw = np.exp(lp - lp.max()); pw /= pw.sum()
            self.rho = float(np.clip(GRID[r.choice(len(GRID), p=pw)] + r.uniform(-DR / 2, DR / 2), -0.9995, 0.9995))
        a, es = functionals(b)
        self.att.append(a); self.es.append(es); self.rhos.append(self.rho)


def ess_bulk(x):
    """Effective sample size of x (chains by draws) after splitting each chain in two (Vehtari et al. 2021, Geyer initial monotone sequence)."""
    x = np.asarray(x, float); m, k = x.shape; h = k // 2
    x = np.concatenate([x[:, :h], x[:, h:2 * h]], 0); m, k = x.shape
    cm = x.mean(1); W = x.var(1, ddof=1).mean(); B = k * cm.var(ddof=1)
    vp = (k - 1) / k * W + B / k
    if vp <= 0: return float(m * k)
    f = np.fft.rfft(x - cm[:, None], 2 * k, axis=1); ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :k] / k
    ac = ac * k / (k - np.arange(k))                        # unbiased autocovariance
    rho_t = 1 - (W - ac.mean(0)) / vp; rho_t[0] = 1.0
    tau, prev = -1.0, np.inf
    for t in range(0, k - 1, 2):
        P = rho_t[t] + rho_t[t + 1]
        if P < 0: break
        P = min(P, prev); prev = P; tau += 2 * P
    return float(m * k / max(tau, 1e-3))


def rhat(x):
    x = np.asarray(x, float); m, k = x.shape; h = k // 2
    x = np.concatenate([x[:, :h], x[:, h:2 * h]], 0); m, k = x.shape
    W = x.var(1, ddof=1).mean(); B = k * x.mean(1).var(ddof=1)
    return float(np.sqrt(((k - 1) / k * W + B / k) / W))


def summarize(att):
    """att: chains by draws.  Posterior summaries with Monte Carlo standard errors."""
    flat = att.ravel(); sd = flat.std(ddof=1)
    out = dict(chains=int(att.shape[0]), draws_per_chain=int(att.shape[1]), mean=float(flat.mean()), sd=float(sd),
               rhat=rhat(att), ess_mean=ess_bulk(att))
    out["mcse_mean"] = float(sd / np.sqrt(out["ess_mean"]))
    pp = float((flat > 0).mean()); ess_p = ess_bulk((att > 0).astype(float))
    out["prob_pos"] = pp; out["ess_prob_pos"] = ess_p; out["mcse_prob_pos"] = float(np.sqrt(max(pp * (1 - pp), 1e-12) / ess_p))
    h = 0.05 * sd
    for nm, p in (("q025", 0.025), ("q975", 0.975)):
        qv = float(np.quantile(flat, p)); ess_q = ess_bulk((att <= qv).astype(float))
        dens = float(((flat <= qv + h).mean() - (flat <= qv - h).mean()) / (2 * h))
        out[nm] = qv; out["ess_" + nm] = ess_q; out["mcse_" + nm] = float(np.sqrt(p * (1 - p) / ess_q) / dens)
    out["chain_means"] = att.mean(1).tolist(); out["chain_prob_pos"] = (att > 0).mean(1).tolist()
    return out


def run(variant):
    ar = variant == "ar1"; cap = int(os.environ.get("CASTLE_CAP", 100_000 if ar else 400_000))
    seeds = np.random.SeedSequence(20261023).spawn(8)[4 * ar: 4 * ar + 4]
    chains = [Chain(s, ar) for s in seeds]; STEP = int(os.environ.get("CASTLE_STEP", 20000)); BURN = min(5000, STEP // 4)
    snaps, total = {}, 0
    while True:
        for c in chains:
            for _ in range(STEP): c.sweep()
        total += STEP
        att = np.array([c.att[BURN:] for c in chains]); s = summarize(att)
        if ar:
            rh = np.array([c.rhos[BURN:] for c in chains]); s["rho"] = dict(mean=float(rh.mean()), q025=float(np.quantile(rh, .025)),
                q975=float(np.quantile(rh, .975)), sd=float(rh.std(ddof=1)), rhat=rhat(rh), ess=ess_bulk(rh))
        snaps[str(total)] = s
        print(variant, "sweeps", total, {k: (round(v, 4) if isinstance(v, float) else None) for k, v in s.items() if k in ("mean", "q025", "q975", "prob_pos", "mcse_prob_pos", "rhat", "ess_mean")}, flush=True)
        if total >= cap or s["mcse_prob_pos"] < 0.005 or (ar and total >= 40000 and s["mcse_prob_pos"] < 0.01): break
    att = np.array([c.att[BURN:] for c in chains]); es = np.array([c.es[BURN:] for c in chains]); rh = np.array([c.rhos[BURN:] for c in chains])
    np.savez_compressed(f"results/did_castle2_draws_{variant}.npz", att=att.astype("float32"), es=es.astype("float32"), rho=rh.astype("float32"))
    out = dict(variant=variant, sweeps_per_chain=total, burn=BURN, snapshots=snaps, final=snaps[str(total)])
    es_flat = es.reshape(-1, es.shape[-1])
    out["event"] = {str(k): [float(es_flat[:, j].mean()), float(np.quantile(es_flat[:, j], .025)), float(np.quantile(es_flat[:, j], .975))]
                    for j, k in enumerate(EV_AVAIL + [-5, -4, -3, -2])}
    out["twfe"] = float(twfe(unit, year, y, ((coh > 0) & (year >= coh)).astype(float))); out["cs"] = cs_att(unit, year, coh, y)
    json.dump(out, open(f"results/res_did_castle2_{variant}.json", "w"), indent=1)
    if not ar:                                           # redraw the chapter's figure from the pooled draws of the four chains
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        flat = att.ravel(); ks = sorted(EV_AVAIL + [-5, -4, -3, -2, -1])
        m = [out["event"][str(k)][0] if k != -1 else 0.0 for k in ks]; lo = [out["event"][str(k)][1] if k != -1 else 0.0 for k in ks]
        hi = [out["event"][str(k)][2] if k != -1 else 0.0 for k in ks]
        fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.9))
        axs[0].errorbar(ks, m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt="o", capsize=3); axs[0].axhline(0, color="k", lw=0.5); axs[0].axvline(-0.5, color="grey", ls=":")
        axs[0].set_xlabel("Years since adoption"); axs[0].set_ylabel("Effect on log homicide rate")
        axs[1].hist(flat, bins=50, density=True, histtype="step"); axs[1].axvline(out["twfe"], color="k", ls="--", lw=0.8, label="TWFE")
        axs[1].axvline(out["cs"], color="C3", ls=":", lw=0.9, label="cohort by time"); axs[1].legend(frameon=False, fontsize=7); axs[1].set_xlabel("ATT, log homicide rate")
        fig.tight_layout(); fig.savefig("figs/castle_4chain.pdf"); plt.close()


if __name__ == "__main__":
    if "--merge" in sys.argv:
        json.dump({v: json.load(open(f"results/res_did_castle2_{v}.json")) for v in ("indep", "ar1")}, open("results/res_did_castle2.json", "w"), indent=1)
        sys.exit(0)
    for v in ([a for a in sys.argv[1:] if a in ("indep", "ar1")] or ["indep", "ar1"]): run(v)
