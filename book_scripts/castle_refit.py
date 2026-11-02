"""ex:chdid:3. Castle refit: horseshoe DiD with flat covariates, population weights, region by year. Usage: castle_refit.py VARIANT SWEEPS."""
import sys, os, json, time
sys.path.insert(0, os.getcwd()); sys.dont_write_bytecode = True
import numpy as np, pandas as pd
SRC = "book_scripts/did_hs.py"; MARK = "# ---------------- Simulation"
src = open(SRC).read(); assert src.count(MARK) == 1
exec(compile(src.split(MARK)[0], SRC, "exec"))
variant = sys.argv[1]; SW = int(sys.argv[2]); BURN = SW // 4
d = pd.read_stata("data/castle.dta")
DROP = [int(x) for x in os.environ.get("DROP", "").split(",") if x]; d = d[~d.sid.isin(DROP)]; TAG = os.environ.get("TAG", "")
first = d[d.post > 0].groupby("sid").year.min(); d["cohort"] = d.sid.map(first).fillna(0).astype(int)
d = d.sort_values(["sid", "year"]).reset_index(drop=True)
unit = d.sid.values; year = d.year.values; y0 = d.l_homicide.values.astype(float); coh = d.cohort.values
ev = np.where(coh > 0, year - coh, -999)
Z0, grp, names, _, cells = build(unit, year, coh, y0, leads=(-5, -4, -3, -2))
CTRL = ["blackm_15_24", "whitem_15_24", "blackm_25_44", "whitem_25_44", "l_exp_subsidy", "l_exp_pubwelfare", "l_police", "unemployrt", "poverty", "l_income", "l_prisoner", "l_lagprisoner"]
RY = [c for c in d.columns if c.startswith("r200") or c.startswith("r201")]
use_x = "x" in variant; use_w = "w" in variant; use_r = "r" in variant
cols = []
if use_x:
    X = d[CTRL].astype(float); print("missing", X.isna().sum().sum()); X = (X - X.mean()) / X.std(); cols.append(X.values)
if use_r:
    R = d[RY].astype(float).values
    # drop region by year columns collinear with year dummies and each other
    keep = []; M = np.column_stack([Z0[:, grp < 0]] + cols + [np.ones((len(d), 0))])
    for j in range(R.shape[1]):
        M2 = np.column_stack([M, R[:, j]])
        if np.linalg.matrix_rank(M2, tol=1e-8) > M.shape[1] if np.linalg.matrix_rank(M, tol=1e-8) == M.shape[1] else np.linalg.matrix_rank(M2, tol=1e-8) > np.linalg.matrix_rank(M, tol=1e-8):
            M = M2; keep.append(j)
    print("region by year kept", len(keep), "of", R.shape[1]); cols.append(R[:, keep])
Z = np.column_stack([Z0] + cols) if cols else Z0
grp2 = np.concatenate([grp, -np.ones(Z.shape[1] - Z0.shape[1], int)])
popw = d.popwt.values.astype(float); w = popw / popw.mean() if use_w else np.ones(len(d))
sw = np.sqrt(w); Zw = Z * sw[:, None]; yw = y0 * sw
q = Z.shape[1]; n = len(yw); pen = grp2 >= 0; G = grp2.max() + 1
pidx = {nm[1]: i for i, nm in enumerate(names) if nm[0] == "path"}; didx = {nm[1]: i for i, nm in enumerate(names) if nm[0] == "dev"}
p_i = np.array([pidx[e] for c, e in cells]); d_i = np.array([didx[(c, e)] for c, e in cells])
wc_n = np.array([np.sum((coh == c) & (ev == e)) for c, e in cells], float)
wc_p = np.array([popw[(coh == c) & (ev == e)].sum() for c, e in cells], float)
ZtZ = Zw.T @ Zw; Zty = Zw.T @ yw; yty = yw @ yw
def chain(seed):
    r = np.random.default_rng(seed)
    lam2 = np.exp(r.normal(0, 1.5, q)); nu = np.ones(q); tau2 = np.exp(r.normal(np.log(0.1), 1.5, G)); xi = np.ones(G); s2 = np.var(yw) * np.exp(r.normal())
    a1, a2, tr = [], [], []
    for it in range(SW):
        pv = np.where(pen, lam2 * tau2[np.maximum(grp2, 0)], np.inf)
        A = ZtZ + np.diag(np.where(pen, 1 / pv, 1e-8)); L = np.linalg.cholesky(A)
        b = np.linalg.solve(A, Zty) + np.sqrt(s2) * np.linalg.solve(L.T, r.normal(size=q))
        rss = yty - 2 * b @ Zty + b @ ZtZ @ b
        bp = b[pen]; tp = tau2[grp2[pen]]
        s2 = 1 / r.gamma((n + pen.sum()) / 2, 1 / (rss / 2 + (bp ** 2 / (lam2[pen] * tp)).sum() / 2))
        lam2[pen] = 1 / r.gamma(1, 1 / (1 / nu[pen] + bp ** 2 / (2 * tp * s2)))
        nu[pen] = 1 / r.gamma(1, 1 / (1 + 1 / lam2[pen]))
        for g in range(G):
            s = grp2 == g
            tau2[g] = 1 / r.gamma((s.sum() + 1) / 2, 1 / (1 / xi[g] + (b[s] ** 2 / lam2[s]).sum() / (2 * s2)))
            xi[g] = 1 / r.gamma(1, 1 / (1 + 1 / tau2[g]))
        if it >= BURN:
            cell = b[p_i] + b[d_i]; a1.append((cell * wc_n).sum() / wc_n.sum()); a2.append((cell * wc_p).sum() / wc_p.sum())
            tr.append(np.abs(b[grp2 == 2]).max())
    return np.array(a1), np.array(a2), np.array(tr)
# weighted two way fixed effects with the same covariates, fractional post as in the data and binary post
def wls(postvar):
    D = pd.get_dummies(unit).values.astype(float); T_ = pd.get_dummies(year).values[:, 1:].astype(float)
    M = np.column_stack([postvar, D, T_] + cols); b = np.linalg.lstsq(M * sw[:, None], y0 * sw, rcond=None)[0]; return float(b[0])
t0 = time.time()
ss = np.random.SeedSequence(20261101).spawn(4)
res = [chain(s) for s in ss]
def rhat(x):
    x = np.asarray(x); k = x.shape[1] // 2; x = np.concatenate([x[:, :k], x[:, k:2 * k]]); W = x.var(1, ddof=1).mean(); B = k * x.mean(1).var(ddof=1)
    return float(np.sqrt(((k - 1) / k * W + B / k) / W))
A1 = np.array([r[0] for r in res]); A2 = np.array([r[1] for r in res]); Tr = np.array([r[2] for r in res])
def s(a): f = a.ravel(); return dict(mean=float(f.mean()), lo=float(np.quantile(f, .025)), hi=float(np.quantile(f, .975)), p=float((f > 0).mean()), chain_p=(a > 0).mean(1).round(3).tolist(), chain_mean=a.mean(1).round(4).tolist(), rhat=rhat(a))
out = dict(variant=variant, q=q, sweeps=SW, burn=BURN, att_cells=s(A1), att_pop=s(A2), max_trend=float(Tr.mean()), twfe_binary=wls((d.post.values > 0).astype(float)), twfe_frac=wls(d.post.values.astype(float)), sec=time.time() - t0)
tb = np.linalg.lstsq(Z0 * 1.0 if False else Z, y0, rcond=None)[0]
b_ = None
popshare = {int(c): float(popw[(coh == c) & (ev >= 0)].sum() / popw[(coh > 0) & (ev >= 0)].sum()) for c in sorted(set(coh[coh > 0]))}
ps = pd.Series(popw[(coh > 0) & (year == 2006)], index=unit[(coh > 0) & (year == 2006)]).sort_values(ascending=False); ta = float(ps.sum())
out["cohort_pop_share"] = popshare; out["top_adopters"] = [(int(k), round(float(v / ta), 3), int(coh[unit == k][0])) for k, v in ps.head(5).items()]
tot06 = popw[year == 2006]; out["adopter_pop_share_all"] = float(ta / tot06.sum()); out["w_max_over_mean"] = float((popw / popw.mean()).max())
# log squared residual of the unweighted least squares two way plus trends fit against log population
Zf = np.column_stack([Z0[:, [i for i, nm in enumerate(names) if nm[0] in ("unit", "year", "trend")]], ((coh > 0) & (ev >= 0)).astype(float)])
rs = y0 - Zf @ np.linalg.lstsq(Zf, y0, rcond=None)[0]; lp = np.log(popw); XX = np.column_stack([np.ones(len(lp)), lp]); out["slope_logres2_logpop"] = float(np.linalg.lstsq(XX, np.log(rs ** 2 + 1e-8), rcond=None)[0][1])
sm = pd.DataFrame(dict(u=unit, r=rs ** 2, p=lp)).groupby("u").mean(); out["slope_state_logvar_logpop"] = float(np.polyfit(sm.p, np.log(sm.r), 1)[0])
print(json.dumps(out)); json.dump(out, open(f"results/res_castle_{variant}{TAG}.json", "w"), indent=1)
