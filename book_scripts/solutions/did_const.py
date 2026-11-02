"""ex:chdid:5 contrast. Panels of design (a), N=40, p=0.1 with tau_{g,e}=1.12 in place of 0.5+0.15e. Both parametrizations, 1500 sweeps, 500 burn. 200 panels.
Usage from the repository root: python book_scripts/solutions/did_const.py panel P | agg   (panel P runs panels P to P+CHUNK-1 with CHUNK = env DC_CHUNK, default 1; agg writes results/solutions/did_const.json)"""
import os, sys, json, time
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_CONCURRENT_THREADS"): os.environ[k] = "1"
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); sys.path.insert(0, os.getcwd()); sys.dont_write_bytecode = True
import numpy as np
G = {"__name__": "s2"}
sys.argv, ARGV = ["x"], sys.argv
exec(compile(open("book_scripts/did_sim2.py").read(), "did_sim2", "exec"), G)
OUT = "results/solutions/dc"; NP = 200
def one(p):
    gen = np.random.default_rng(np.random.SeedSequence([20261101, 0, p]))
    unit, year, cc, ev, y, tr, _ = G["make_panel"](gen, 40, 0.1, "a")
    y = y - np.where(tr, 0.5 + 0.15 * ev, 0.0) + np.where(tr, 1.12, 0.0)
    G["rng"] = np.random.default_rng(np.random.SeedSequence([20261101, 1, p]))
    Z, g, names, _, cells = G["build"](unit, year, cc, y); a = G["att_from"](G["group_hs"](Z, y, g, iters=1500, burn=500), names, cells, cc, ev)
    G["rng"] = np.random.default_rng(np.random.SeedSequence([20261101, 2, p]))
    Z, g, names, _, cells = G["build_single"](unit, year, cc, y); b = G["att_single"](G["group_hs"](Z, y, g, iters=1500, burn=500), names, cells, cc, ev)
    return [a.mean() - 1.12, *(np.quantile(a, [.025, .975]) - 1.12), b.mean() - 1.12, *(np.quantile(b, [.025, .975]) - 1.12)]
def job(p):
    fn = f"{OUT}/{p}.json"
    if os.path.exists(fn): return
    r = one(p); json.dump([float(x) for x in r], open(fn, "w"))
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if ARGV[1] == "panel":
        p0 = int(ARGV[2]); ch = int(os.environ.get("DC_CHUNK", 1))
        for p in range(p0, min(p0 + ch, NP)): job(p); print("panel", p, flush=True)
    else:
        A = np.array([json.load(open(f"{OUT}/{p}.json")) for p in range(NP)])
        out = {}
        for nm, i in (("path", 0), ("single", 3)):
            e = A[:, i]; c = (A[:, i + 1] <= 0) & (A[:, i + 2] >= 0)
            out[nm] = dict(n=NP, bias=float(e.mean()), bias_se=float(e.std(ddof=1) / np.sqrt(len(e))), rmse=float(np.sqrt((e ** 2).mean())), cover=int(c.sum()), width=float((A[:, i + 2] - A[:, i + 1]).mean()))
        print(json.dumps(out)); json.dump(out, open("results/solutions/did_const.json", "w"))
