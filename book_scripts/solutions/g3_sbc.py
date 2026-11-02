"""ex:ch14:5. SBC for the proper horseshoe regression. Usage: g3_sbc.py MODE  (pilot | run THIN | control).
Fixed design n=100, p=20; first column binary treatment (50 ones), others iid N(0,1), seed 20261105.
Prior: sigma2 ~ IG(2,1), tau|sigma2 ~ N(0,100 sigma2), beta_j|.. ~ N(0, lam_j^2 g^2 sigma2), lam_j, g half Cauchy.
"""
import os; os.makedirs("results/solutions", exist_ok=True)
import json
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import multiprocessing as mp

import numpy as np
from scipy import linalg, stats

N, P = 100, 20
rngZ = np.random.default_rng(20261105)
Z = rngZ.normal(size=(N, P))
Z[:, 0] = 0.0
Z[rngZ.permutation(N)[:50], 0] = 1.0
ZZ = Z.T @ Z
FLOOR, CEIL = 1e-30, 1e30


def draw_prior(rng):
    s2 = 1.0 / rng.gamma(2.0, 1.0)
    g = abs(rng.standard_cauchy())
    lam = np.abs(rng.standard_cauchy(P - 1))
    tau = rng.normal(0, 10 * np.sqrt(s2))
    beta = rng.normal(size=P - 1) * lam * g * np.sqrt(s2)
    return s2, g, lam, tau, beta


def sweep_factory(Y, rng):
    ZY = Z.T @ Y
    YY = Y @ Y
    lam2 = np.ones(P - 1)
    nu = np.ones(P - 1)
    g2 = 1.0
    xi = 1.0
    s2 = max(np.var(Y), 1e-3)
    beta = np.zeros(P)

    def sweep():
        nonlocal lam2, nu, g2, xi, s2, beta
        # beta: stable form, A^{-1} = D B^{-1} D with B = I + D Z'Z D, D^2 = prior variance factor
        Lam = np.empty(P)
        Lam[0] = 100.0
        Lam[1:] = np.clip(lam2 * g2, FLOOR, CEIL)
        D = np.sqrt(Lam)
        B = (D[:, None] * ZZ) * D[None, :]
        B[np.diag_indices(P)] += 1.0
        L = linalg.cholesky(B, lower=True, check_finite=False)
        w = linalg.solve_triangular(L, D * ZY, lower=True, check_finite=False)
        w = w + np.sqrt(s2) * rng.standard_normal(P)
        beta = D * linalg.solve_triangular(L.T, w, lower=False, check_finite=False)
        # sigma2 given beta and all scales
        rss = YY - 2 * beta @ ZY + beta @ ZZ @ beta
        pen = np.sum(beta**2 / Lam)
        s2 = 1.0 / rng.gamma(2.0 + (N + P) / 2.0, 1.0 / (1.0 + 0.5 * (max(rss, 0) + pen)))
        s2 = float(np.clip(s2, 1e-30, 1e30))
        b = beta[1:]
        # local scales
        lam2 = 1.0 / rng.gamma(1.0, 1.0 / (1.0 / nu + b**2 / (2 * g2 * s2)))
        lam2 = np.clip(lam2, FLOOR, CEIL)
        nu = 1.0 / rng.gamma(1.0, 1.0 / (1.0 + 1.0 / lam2))
        # global scale
        g2 = 1.0 / rng.gamma((P - 1 + 1) / 2.0, 1.0 / (1.0 / xi + np.sum(b**2 / lam2) / (2 * s2)))
        g2 = float(np.clip(g2, FLOOR, CEIL))
        xi = 1.0 / rng.gamma(1.0, 1.0 / (1.0 + 1.0 / g2))
        return beta[0], np.sqrt(g2), s2

    return sweep


def acf1(x, lag=1):
    x = x - x.mean()
    d = x @ x
    return 0.0 if d == 0 else float(x[:-lag] @ x[lag:] / d)


def rep_run(args):
    r, burn, thin, L = args
    rng = np.random.default_rng([20261105, r])
    s2, g, lam, tau, beta = draw_prior(rng)
    Y = Z @ np.r_[tau, beta] + np.sqrt(s2) * rng.standard_normal(N)
    sw = sweep_factory(Y, rng)
    for _ in range(burn):
        sw()
    tr = np.empty((L, 3))
    for i in range(L):
        for _ in range(thin):
            o = sw()
        tr[i] = o
    return dict(r=r, rt=int(np.sum(tr[:, 0] < tau)), rg=int(np.sum(tr[:, 1] < g)), rs=int(np.sum(tr[:, 2] < s2)), g0=float(g), tau0=float(tau),
                ac_t=acf1(tr[:, 0]), ac_g=acf1(np.log(tr[:, 1])), ac_s=acf1(np.log(tr[:, 2])), kept=tr.tolist())


def pilot_run(args):
    r, burn, n = args
    rng = np.random.default_rng([20261105, r])
    s2, g, lam, tau, beta = draw_prior(rng)
    Y = Z @ np.r_[tau, beta] + np.sqrt(s2) * rng.standard_normal(N)
    sw = sweep_factory(Y, rng)
    for _ in range(burn):
        sw()
    tr = np.array([sw() for _ in range(n)])
    lags = [1, 10, 50, 100, 200, 500, 1000]
    return dict(r=r, g0=float(g), t=[acf1(tr[:, 0], l) for l in lags], g=[acf1(np.log(tr[:, 1]), l) for l in lags], lags=lags)


def report(res, L):
    rt = np.array([x["rt"] for x in res])
    rg = np.array([x["rg"] for x in res])
    rs = np.array([x["rs"] for x in res])
    out = {}
    for name, rk in [("tau", rt), ("g", rg), ("s2", rs)]:
        c = np.bincount(rk, minlength=L + 1)
        chi = stats.chisquare(c)
        c5 = np.add.reduce([c[i::1][:0] for i in []]) if False else None
        # coarse bins: 5 ranks each
        co = np.array([c[i:i + 4].sum() for i in range(0, L + 1, 4)])
        chic = stats.chisquare(co)
        ks = np.mean(rk <= L // 2)
        ext = np.mean((rk == 0) | (rk == L))
        out[name] = dict(counts=c.tolist(), chi=float(chi.statistic), p=float(chi.pvalue), coarse=co.tolist(), chic=float(chic.statistic), pc=float(chic.pvalue), lower_half=float(ks), extreme=float(ext), mean_rank=float(rk.mean()))
    for k in ["ac_t", "ac_g", "ac_s"]:
        v = np.array([x[k] for x in res])
        out[k] = dict(mean=float(v.mean()), q05=float(np.quantile(v, .05)), q95=float(np.quantile(v, .95)), se_null=1 / np.sqrt(L))
    # pooled lag 1 rank autocorrelation of kept draws: rank transform within rep
    def pooled(idx, lg):
        a, b = [], []
        for x in res:
            t = np.asarray(x["kept"])[:, idx]
            rr = stats.rankdata(t) - (L + 1) / 2
            a.append(rr[:-1])
            b.append(rr[1:])
        a = np.concatenate(a)
        b = np.concatenate(b)
        return float(a @ b / (a @ a))
    out["pooled_rank_ac"] = dict(tau=pooled(0, 0), g=pooled(1, 0), s2=pooled(2, 0))
    # expected under independence for ranks within a rep of size L: -1/(L-1) approx
    out["null_rank_ac"] = -1 / (L - 1)
    # ranks in strata of g0
    g0 = np.array([x["g0"] for x in res])
    out["g0_quant"] = np.quantile(g0, [.05, .25, .5, .75, .95]).tolist()
    small = g0 < np.quantile(g0, 1 / 3)
    out["rg_small_g0_chi"] = dict(c=[int(c) for c in [np.bincount(rg[small], minlength=L + 1)[i:i + 4].sum() for i in range(0, L + 1, 4)]])
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "pilot":
        with mp.Pool(8) as p:
            res = p.map(pilot_run, [(r, 2000, 20000) for r in range(10000, 10024)])
        lags = res[0]["lags"]
        t = np.array([x["t"] for x in res])
        g = np.array([x["g"] for x in res])
        print("lags", lags)
        print("tau mean acf", np.round(t.mean(0), 3), "max", np.round(t.max(0), 3))
        print("log g mean acf", np.round(g.mean(0), 3), "max", np.round(g.max(0), 3))
        print("g0 of worst:", [(round(x["g0"], 4), round(x["g"][3], 2)) for x in sorted(res, key=lambda x: -x["g"][3])[:6]])
    elif mode == "time":
        import time
        t0 = time.time()
        rep_run((0, 2000, 1, 19))
        print("sweeps/s", 2019 / (time.time() - t0))
    else:
        thin = int(sys.argv[2])
        nrep = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        burn = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
        with mp.Pool(8) as p:
            res = p.map(rep_run, [(r, burn, thin, 19) for r in range(nrep)], chunksize=10)
        out = report(res, 19)
        out.update(thin=thin, nrep=nrep, burn=burn)
        fn = f"results/solutions/sbc_thin{thin}.json"
        json.dump(out, open(fn, "w"), indent=1)
        print(json.dumps({k: (v if k not in ("tau", "g", "s2") else {a: b for a, b in v.items()}) for k, v in out.items()}))
