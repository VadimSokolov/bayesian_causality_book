"""ex:chapter_blp:9, second step: fit GBCTargets (default hidden layers (256, 256, 128)) to the simulated draws of blp_kappa.py and report the posterior of kappa for the minivan data.
Usage from the repository root: python book_scripts/solutions/blp_fit.py   (30,000 draws, 400 held out; writes results/solutions/kappa.json)"""
import os, sys, json, time, pickle
sys.dont_write_bytecode = True
import numpy as np
from blp_kappa import D, obs, TRUE, PARAMS, GBCTargets, simulate_market, summaries, NC
out = []
for c in range(NC): out += [x for x in pickle.load(open(f"results/solutions/ka/{c}.pkl", "rb")) if x is not None]
Th = np.array([x[0] for x in out]); S = np.array([x[1] for x in out]); kq = Th[:, -1]; kc = np.clip(kq, 1e-4, 1 - 1e-4)
Y = np.column_stack([Th[:, :-1], np.log(kc / (1 - kc))]); test = 400; tr = len(Th) - test
fn = "results/solutions/kappa_G.pkl"
if os.path.exists(fn): G = pickle.load(open(fn, "rb"))
else:
    t0 = time.time(); G = GBCTargets(S[:tr], Y[:tr]); pickle.dump(G, open(fn, "wb")); print("nets done", time.time() - t0, flush=True)
res = dict(n=len(Th), prior_q=np.quantile(kq, [.05, .25, .5, .75, .95]).tolist(), prior_mean=float(kq.mean()))
k = max(300, tr // 20); res["k"] = k; ex = lambda z: 1 / (1 + np.exp(-z))
dr = G.posterior(obs(), k=k); ka = ex(dr[:, -1]); ia = PARAMS.index("alpha"); isp = PARAMS.index("sp"); isx = PARAMS.index("sx")
res["minivan"] = dict(mean=float(ka.mean()), q=np.quantile(ka, [.025, .05, .5, .95, .975]).tolist(), p_lt_01=float((ka < .1).mean()), p_lt_025=float((ka < .25).mean()), alpha=[float(dr[:, ia].mean()), *np.quantile(dr[:, ia], [.025, .975]).tolist()], sp=[float(dr[:, isp].mean()), *np.quantile(dr[:, isp], [.025, .975]).tolist()], sx=[float(dr[:, isx].mean()), *np.quantile(dr[:, isx], [.025, .975]).tolist()])
cov, pm, wid = [], [], []
for i in range(tr, len(Th)):
    d_ = ex(G.posterior(S[i], k=k)[:, -1]); lo, hi = np.quantile(d_, [.025, .975]); cov.append(lo <= kq[i] <= hi); pm.append(d_.mean()); wid.append(hi - lo)
pm = np.array(pm); kt = kq[tr:]; cov = np.array(cov)
res["calib"] = dict(n=test, coverage95=float(cov.mean()), rmse=float(np.sqrt(((pm - kt) ** 2).mean())), width=float(np.mean(wid)), corr=float(np.corrcoef(pm, kt)[0, 1]), sd_prior=float(kt.std()), by={f"{a}-{b}": dict(rmse=float(np.sqrt(((pm - kt)[(kt >= a) & (kt < b)] ** 2).mean())), cover=float(cov[(kt >= a) & (kt < b)].mean()), n=int(((kt >= a) & (kt < b)).sum())) for a, b in [(0, .33), (.33, .67), (.67, 1.01)]})
for nm, kv in [("k0.5", .5), ("k1.0", 1.0)]:
    rng = np.random.default_rng(2027); D.kappa = kv; r = simulate_market(D, TRUE, "kappa", rng); so = summaries(D, r["s"], r["p"], r["micro"] + rng.normal(0, [0.03, 0.10])); d_ = ex(G.posterior(so, k=k)[:, -1]); res[nm] = dict(ok=bool(r["ok"]), mean=float(d_.mean()), q=np.quantile(d_, [.025, .975]).tolist())
print(json.dumps(res)); json.dump(res, open("results/solutions/kappa.json", "w"), indent=1)
