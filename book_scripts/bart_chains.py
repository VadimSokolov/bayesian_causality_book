"""Chapter 16, review loop F9: do the BART and BCF averages of the LaLonde and 401(k) applications converge across independent chains?

Design. The four tree fits of book_scripts/bart_apps.py (LaLonde: Hill's BART, BART with the estimated propensity, BCF on the 185 NSW treated and 15,992 CPS controls;
401(k): BCF on the 9,915 households, propensity from the cross validated logistic regression on the standardized nine covariates), each run as four independently seeded
chains (seeds 101 to 104), every chain with 20 grow from root sweeps followed by 2,000 retained draws (the main run of bart_apps.py keeps one chain of 1,000 draws, 500 for the 401(k)
BCF, seed 7). Recorded per chain: the draws of the reported averages: LaLonde, the average over the treated of observed minus imputed control earnings (att) and, for BART, the
conditional average effect on the treated (catt); 401(k), the average effect over all households (ate), the treated (att), the untreated (atu) and the ten income decile averages.
The aggregator pools the four chains and reports, per quantity, the split R-hat and the effective sample size of iv_exact.split_rhat_ess (Geyer's initial positive sequence,
chains cut in halves), the pooled mean and 95% interval, the mean and interval of each chain, and the spread of the four chain means.

The 401(k) BCF in the book pools eight chains (seeds 101 to 108) of 50,000 retained draws (--nm 50000, about 100 GB of memory per chain on the cluster).

usage (repository root): PYTHONPATH=. python3 book_scripts/bart_chains.py run --fit lal_bart --seed 101 --out DIR ; python3 book_scripts/bart_chains.py agg DIR
writes results/res_bart_chains.json (agg)
"""
import argparse, glob, json, os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iv_exact import split_rhat_ess
warnings.filterwarnings("ignore")
from stochtree import BARTModel, BCFModel
from sklearn.linear_model import LogisticRegressionCV
NM = 2000


def lalonde():
    from causaldata import nsw_mixtape, cps_mixtape
    cov = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]
    def prep(df):
        X = df[cov].astype(float).copy(); X["u74"] = (X.re74 == 0).astype(float); X["u75"] = (X.re75 == 0).astype(float)
        return X.values, df.treat.values.astype(float), df.re78.values.astype(float) / 1000
    nsw = nsw_mixtape.load_pandas().data; cps = cps_mixtape.load_pandas().data
    obs = pd.concat([nsw[nsw.treat == 1], cps[cps.treat == 0]]); return prep(obs)


def run(fit, seed):
    if fit.startswith("lal"):
        X, T, Y = lalonde(); tr = T == 1; ps = LogisticRegressionCV(cv=5, max_iter=3000).fit(X, T).predict_proba(X)[:, 1]
        if fit == "lal_bcf":
            b = BCFModel(); b.sample(X_train=X, Z_train=T, y_train=Y, propensity_train=ps, num_gfr=20, num_mcmc=NM, general_params={"random_seed": seed}); return {"att": b.tau_hat_train[tr].mean(0)}
        Xa = np.column_stack([X, T]) if fit == "lal_bart" else np.column_stack([X, T, ps]); m = BARTModel()
        m.sample(X_train=Xa, y_train=Y, num_gfr=20, num_mcmc=NM, general_params={"random_seed": seed})
        X1 = Xa[tr].copy(); X0 = Xa[tr].copy(); X0[:, X.shape[1]] = 0; f1 = m.predict(X1); f0 = m.predict(X0)
        f1 = f1["y_hat"] if isinstance(f1, dict) else f1; f0 = f0["y_hat"] if isinstance(f0, dict) else f0
        return {"att": (Y[tr][:, None] - f0).mean(0), "catt": (f1 - f0).mean(0)}
    d = pd.read_stata("data/sipp1991.dta"); xv = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]
    Xk = d[xv].values.astype(float); Tk = d.e401.values.astype(float); Yk = d.net_tfa.values.astype(float) / 1000
    pk = LogisticRegressionCV(cv=5, max_iter=3000).fit((Xk - Xk.mean(0)) / Xk.std(0), Tk).predict_proba((Xk - Xk.mean(0)) / Xk.std(0))[:, 1]
    b = BCFModel(); b.sample(X_train=Xk, Z_train=Tk, y_train=Yk, propensity_train=pk, num_gfr=20, num_mcmc=NM, general_params={"random_seed": seed}); tk = b.tau_hat_train
    inc = d.inc.values / 1000; bins = np.quantile(inc, np.linspace(0, 1, 11)); out = {"ate": tk.mean(0), "att": tk[Tk == 1].mean(0), "atu": tk[Tk == 0].mean(0)}
    for k in range(10): out["decile%d" % (k + 1)] = tk[(inc >= bins[k]) & (inc <= bins[k + 1])].mean(0)
    return out


def aggregate(dr):
    res = {}
    for fit in ("lal_bart", "lal_bartps", "lal_bcf", "k401_bcf"):
        fs = sorted(glob.glob(os.path.join(dr, "%s_*.npz" % fit)))
        if not fs: continue
        Z = [np.load(f) for f in fs]; res[fit] = {"chains": len(fs), "draws_per_chain": int(len(Z[0][Z[0].files[0]]))}
        for q in Z[0].files:
            C = np.array([z[q] for z in Z]); rh, es = split_rhat_ess(C); p = C.ravel(); cm = C.mean(1)
            res[fit][q] = dict(rhat=rh, ess=es, mean=float(p.mean()), lo=float(np.quantile(p, .025)), hi=float(np.quantile(p, .975)), chain_means=cm.tolist(), chain_mean_range=float(cm.max() - cm.min()),
                               chain_lo=[float(np.quantile(c, .025)) for c in C], chain_hi=[float(np.quantile(c, .975)) for c in C])
    json.dump(res, open("results/res_bart_chains.json", "w"), indent=1)
    for fit, r in res.items():
        print(fit, r["chains"], r["draws_per_chain"])
        for q, v in r.items():
            if isinstance(v, dict): print("  %-9s rhat %.4f ess %7.0f mean %.3f [%.3f, %.3f] chain-mean range %.3f" % (q, v["rhat"], v["ess"], v["mean"], v["lo"], v["hi"], v["chain_mean_range"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("dir", nargs="?"); ap.add_argument("--fit"); ap.add_argument("--seed", type=int); ap.add_argument("--out"); ap.add_argument("--nm", type=int, default=2000); a = ap.parse_args()
    if a.cmd == "agg": aggregate(a.dir)
    else:
        os.makedirs(a.out, exist_ok=True); NM = a.nm; r = run(a.fit, a.seed); np.savez(os.path.join(a.out, "%s_%d.npz" % (a.fit, a.seed)), **r); print("done", a.fit, a.seed)
