"""sol4 F2. Castle refit with a variance floor: Var(e_it) = s2 * ((1 - eta) + eta / w_i), eta ~ Uniform(0, 1), griddy Gibbs on 41 points. eta = 0 is equal noise, eta = 1 the inverse population model. Usage: castle_eta.py VARIANT SWEEPS (VARIANT contains x for covariates, r for region by year; n for neither)."""
import sys, os, json, time
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); sys.path.insert(0, os.getcwd()); sys.dont_write_bytecode = True; os.makedirs("results/solutions", exist_ok=True)
import numpy as np, pandas as pd
SRC = "book_scripts/did_hs.py"; MARK = "# ---------------- Simulation"
src = open(SRC).read(); assert src.count(MARK) == 1
exec(compile(src.split(MARK)[0], SRC, "exec"))
variant = sys.argv[1]; SW = int(sys.argv[2]); BURN = SW // 4
d = pd.read_stata("data/castle.dta")
first = d[d.post > 0].groupby("sid").year.min(); d["cohort"] = d.sid.map(first).fillna(0).astype(int)
d = d.sort_values(["sid", "year"]).reset_index(drop=True)
unit = d.sid.values; year = d.year.values; y0 = d.l_homicide.values.astype(float); coh = d.cohort.values
ev = np.where(coh > 0, year - coh, -999)
Z0, grp, names, _, cells = build(unit, year, coh, y0, leads=(-5, -4, -3, -2))
CTRL = ["blackm_15_24", "whitem_15_24", "blackm_25_44", "whitem_25_44", "l_exp_subsidy", "l_exp_pubwelfare", "l_police", "unemployrt", "poverty", "l_income", "l_prisoner", "l_lagprisoner"]
RY = [c for c in d.columns if c.startswith("r200") or c.startswith("r201")]
cols = []
if "x" in variant:
    X = d[CTRL].astype(float); X = (X - X.mean()) / X.std(); cols.append(X.values)
if "r" in variant:
    R = d[RY].astype(float).values; M = np.column_stack([Z0[:, grp < 0]] + cols); keep = []
    for j in range(R.shape[1]):
        M2 = np.column_stack([M, R[:, j]])
        if np.linalg.matrix_rank(M2, tol=1e-8) > np.linalg.matrix_rank(M, tol=1e-8): M = M2; keep.append(j)
    cols.append(R[:, keep])
Z = np.column_stack([Z0] + cols) if cols else Z0
grp2 = np.concatenate([grp, -np.ones(Z.shape[1] - Z0.shape[1], int)])
popw = d.popwt.values.astype(float); w = popw / popw.mean()
q = Z.shape[1]; n = len(y0); pen = grp2 >= 0; G = grp2.max() + 1
pidx = {nm[1]: i for i, nm in enumerate(names) if nm[0] == "path"}; didx = {nm[1]: i for i, nm in enumerate(names) if nm[0] == "dev"}
p_i = np.array([pidx[e] for c, e in cells]); d_i = np.array([didx[(c, e)] for c, e in cells])
wc_n = np.array([np.sum((coh == c) & (ev == e)) for c, e in cells], float)
grid = np.linspace(0, 1, 41); lw = np.log(np.r_[.5, np.ones(39), .5])
V = np.array([(1 - e) + e / w for e in grid]); PREC = 1 / V; LOGDET = np.log(V).sum(1)
ZtZ = [(Z * p[:, None]).T @ Z for p in PREC]; Zty = [(Z * p[:, None]).T @ y0 for p in PREC]
def chain(seed):
    r = np.random.default_rng(seed)
    lam2 = np.exp(r.normal(0, 1.5, q)); nu = np.ones(q); tau2 = np.exp(r.normal(np.log(0.1), 1.5, G)); xi = np.ones(G); s2 = np.var(y0) * np.exp(r.normal()); k = r.integers(41)
    att, eta = [], []
    for it in range(SW):
        pv = np.where(pen, lam2 * tau2[np.maximum(grp2, 0)], np.inf)
        A = ZtZ[k] + np.diag(np.where(pen, 1 / pv, 1e-8)); L = np.linalg.cholesky(A)
        b = np.linalg.solve(A, Zty[k]) + np.sqrt(s2) * np.linalg.solve(L.T, r.normal(size=q))
        e2 = (y0 - Z @ b) ** 2
        bp = b[pen]; tp = tau2[grp2[pen]]
        s2 = 1 / r.gamma((n + pen.sum()) / 2, 1 / ((PREC[k] @ e2) / 2 + (bp ** 2 / (lam2[pen] * tp)).sum() / 2))
        lp = lw - .5 * LOGDET - .5 * (PREC @ e2) / s2; pr = np.exp(lp - lp.max()); k = r.choice(41, p=pr / pr.sum())
        lam2[pen] = 1 / r.gamma(1, 1 / (1 / nu[pen] + bp ** 2 / (2 * tp * s2)))
        nu[pen] = 1 / r.gamma(1, 1 / (1 + 1 / lam2[pen]))
        for g in range(G):
            s = grp2 == g
            tau2[g] = 1 / r.gamma((s.sum() + 1) / 2, 1 / (1 / xi[g] + (b[s] ** 2 / lam2[s]).sum() / (2 * s2)))
            xi[g] = 1 / r.gamma(1, 1 / (1 + 1 / tau2[g]))
        if it >= BURN:
            cell = b[p_i] + b[d_i]; att.append((cell * wc_n).sum() / wc_n.sum()); eta.append(grid[k])
    return np.array(att), np.array(eta)
def rhat(x):
    x = np.asarray(x); h = x.shape[1] // 2; x = np.concatenate([x[:, :h], x[:, h:2 * h]]); W = x.var(1, ddof=1).mean(); B = h * x.mean(1).var(ddof=1)
    return float(np.sqrt(((h - 1) / h * W + B / h) / W))
def ess(x):
    f = x.ravel(); m = x.shape[0]; h = x.shape[1]; v = x.var(); tot = 0.
    for lag in range(1, min(h - 1, 5000)):
        rho = np.mean([np.corrcoef(c[:-lag], c[lag:])[0, 1] for c in x])
        if rho < .01: break
        tot += rho
    return float(m * h / (1 + 2 * tot))
t0 = time.time(); ss = np.random.SeedSequence(20261102).spawn(4); res = [chain(s) for s in ss]
A = np.array([x[0] for x in res]); E = np.array([x[1] for x in res])
def s(a): f = a.ravel(); return dict(mean=float(f.mean()), med=float(np.median(f)), lo=float(np.quantile(f, .025)), hi=float(np.quantile(f, .975)), p=float((f > 0).mean()), chain_mean=a.mean(1).round(4).tolist(), chain_p=(a > 0).mean(1).round(3).tolist(), rhat=rhat(a), ess=ess(a))
f = E.ravel(); ep = s(E); ep["p_gt_50"] = float((f > .5).mean()); ep["p_gt_90"] = float((f > .9).mean())
# state mean of the implied variance at a floor: the eta share of the noise that scales with 1 / w
out = dict(variant=variant, q=q, sweeps=SW, burn=BURN, att=s(A), eta=ep, corr_att_eta=float(np.corrcoef(A.ravel(), f)[0, 1]), sec=time.time() - t0)
for lo, hi in [(0, .5), (.5, .9), (.9, 1.01)]:
    m = (f >= lo) & (f < hi); out[f"att_eta_{lo}_{hi}"] = dict(share=float(m.mean()), mean=float(A.ravel()[m].mean()) if m.any() else None, p=float((A.ravel()[m] > 0).mean()) if m.any() else None)
# popwt against the state's mean of the population implied by the homicide count and rate
if {"homicide_c", "homicide"} <= set(d.columns):
    pop = d.homicide_c / d.homicide * 1e5; g = pd.DataFrame(dict(u=unit, pop=pop, w=popw)).groupby("u"); out["popwt_vs_mean_pop_max_rel_err"] = float((g.w.first() / g["pop"].mean() - 1).abs().max()); out["popwt_vs_mean_pop_median_rel_err"] = float((g.w.first() / g["pop"].mean() - 1).abs().median())
print(json.dumps(out)); json.dump(out, open(f"results/solutions/castle_eta_{variant}.json", "w"), indent=1)
