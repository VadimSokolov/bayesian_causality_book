"""ex:chbart:6. Hill and Su common support versus propensity trimming.
Usage from the repository root: python book_scripts/solutions/g3_bart.py run R | agg   (one data set R of 0 to 199 per call into results/solutions/bart/R.json, CHAINS chains of NMCMC retained draws pooled;
the full run uses 4 chains of 2,000 draws; agg summarizes all data sets into results/solutions/bart_summary.txt). BART_NMCMC and BART_CHAINS override the defaults."""
import json
import os
import sys
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import multiprocessing as mp

import numpy as np

warnings.filterwarnings("ignore")
expit = lambda x: 1 / (1 + np.exp(-x))
SEED = 20261106
CUT = 0.90


def design(rng, n):
    X = rng.normal(size=(n, 5))
    x1, x2 = X[:, 0], X[:, 1]
    # region F: x1 > 1, propensity 0.97, prognostic surface flat in x1 there
    # region S: x2 > 1.2, no control at all in the region (propensity 0.995) but the linear logit does not see it; surface steep in x2
    e = expit(1.2 * np.minimum(x1, 1.0) + 2.3 * (x1 > 1))
    pocket = x2 > 1.2
    e = np.where(pocket, 0.995, e)
    T = (rng.uniform(size=n) < e).astype(float)
    mu = np.minimum(x1, 1.0) + 2 * np.maximum(x2 - 1.2, 0) + 0.5 * X[:, 2]
    tau = 1 + 1.0 * (x1 > 1) + 6 * np.maximum(x2 - 1.2, 0)
    Y = mu + tau * T + rng.normal(size=n)
    return X, T, Y, e, tau, pocket, x1 > 1


def one(args):
    r, n, nm = args[:3]; chains = args[3] if len(args) > 3 else 1
    from sklearn.linear_model import LogisticRegression
    from stochtree import BARTModel

    rng = np.random.default_rng([SEED, r])
    X, T, Y, e, tau, pocket, flat = design(rng, n)
    ps = LogisticRegression(C=1e6, max_iter=2000).fit(X, T).predict_proba(X)[:, 1]
    Xa = np.column_stack([X, T])
    X1 = Xa.copy(); X1[:, -1] = 1
    X0 = Xa.copy(); X0[:, -1] = 0
    F1, F0 = [], []
    for ch in range(chains):
        m = BARTModel()
        m.sample(X_train=Xa, y_train=Y, num_gfr=10, num_mcmc=nm, general_params={"random_seed": 100000 + r + 1000000 * ch})
        f1 = m.predict(X1); f0 = m.predict(X0)
        F1.append(f1["y_hat"] if isinstance(f1, dict) else f1)
        F0.append(f0["y_hat"] if isinstance(f0, dict) else f0)
    f1 = np.concatenate(F1, axis=1); f0 = np.concatenate(F0, axis=1)
    tr = T == 1
    s1 = f1.std(axis=1); s0 = f0.std(axis=1)
    hill = s0[tr] <= s1[tr].max() + s1[tr].std(ddof=1)
    q99 = s0[~tr]
    hill99 = s0[tr] <= np.quantile(q99, 0.99)
    tr_hat = ps[tr] <= CUT
    tr_ora = e[tr] <= CUT
    tt = tau[tr]
    d = (f1 - f0)[tr]  # units by draws
    out = dict(r=r, nt=int(tr.sum()), nc=int((~tr).sum()))
    out["minctrl_F"] = int(((~tr) & flat).sum()); out["ctrl_P"] = int(((~tr) & pocket).sum())
    sets = dict(all=np.ones(tr.sum(), bool), hill=hill, hill99=hill99, trim=tr_hat, oracle=tr_ora)
    cov_ok = lambda draws, truth: float(np.quantile(draws, .025) <= truth <= np.quantile(draws, .975))
    for k, s in sets.items():
        out[f"n_{k}"] = int(s.sum())
        if s.sum() == 0:
            continue
        draws = d[s].mean(axis=0)
        out[f"est_{k}"] = float(draws.mean())
        out[f"true_{k}"] = float(tt[s].mean())
        out[f"cov_{k}"] = cov_ok(draws, tt[s].mean())
        out[f"width_{k}"] = float(np.quantile(draws, .975) - np.quantile(draws, .025))
    out["true_all_T"] = float(tt.mean())
    p = pocket[tr]; f = flat[tr]
    for k, s in sets.items():
        out[f"kept_pocket_{k}"] = float(s[p].mean()) if p.sum() else np.nan
        out[f"kept_flat_{k}"] = float(s[f & ~p].mean()) if (f & ~p).sum() else np.nan
        rest = ~p & ~f
        out[f"kept_rest_{k}"] = float(s[rest].mean()) if rest.sum() else np.nan
    out["np"] = int(p.sum()); out["nf"] = int((f & ~p).sum())
    out["jacc_hill_trim"] = float(np.sum((~hill) & (~tr_hat)) / max(1, np.sum((~hill) | (~tr_hat))))
    out["only_hill"] = int(np.sum((~hill) & tr_hat)); out["only_trim"] = int(np.sum(hill & (~tr_hat))); out["both"] = int(np.sum((~hill) & (~tr_hat)))
    out["only_hill99"] = int(np.sum((~hill99) & tr_hat)); out["only_trim99"] = int(np.sum(hill99 & (~tr_hat))); out["both99"] = int(np.sum((~hill99) & (~tr_hat)))
    # s0 by region for diagnosis
    so = s0[tr]
    out["s0_p"] = float(so[p].mean()) if p.sum() else np.nan
    out["s0_f"] = float(so[f & ~p].mean()) if (f & ~p).sum() else np.nan
    out["s0_r"] = float(so[~p & ~f].mean())
    out["cut_hill"] = float(s1[tr].max() + s1[tr].std(ddof=1))
    out["cut_99"] = float(np.quantile(q99, .99))
    out["ps_p"] = float(ps[tr][p].mean()) if p.sum() else np.nan
    out["ps_f"] = float(ps[tr][f & ~p].mean()) if (f & ~p).sum() else np.nan
    # IPW ATT on trimmed sample, Hajek odds weighted controls, with controls restricted to ps<=CUT
    keep_c = (~tr) & (ps <= CUT)
    w = ps[keep_c] / (1 - ps[keep_c])
    Ytr = Y[tr][tr_hat]
    out["est_ipw_trim"] = float(Ytr.mean() - np.sum(w * Y[keep_c]) / w.sum())
    w0 = ps[~tr] / (1 - ps[~tr])
    out["est_ipw_all"] = float(Y[tr].mean() - np.sum(w0 * Y[~tr]) / w0.sum())
    return out


def summarize(res):
    keys = sorted({k for x in res for k in x if k != "r"})
    mean = {k: float(np.nanmean([x.get(k, np.nan) for x in res])) for k in keys}
    print(json.dumps({k: round(v, 3) for k, v in mean.items()}))
    print("set    n_kept  truth   est    bias   rmse  mcse_bias cov  width  bias_vs_allATT")
    for k in ["all", "hill", "hill99", "trim", "oracle"]:
        e = np.array([x.get(f"est_{k}", np.nan) for x in res]); t = np.array([x.get(f"true_{k}", np.nan) for x in res]); a = np.array([x["true_all_T"] for x in res])
        ok = np.isfinite(e)
        print(f"{k:7s} {mean['n_'+k]:6.1f} {t[ok].mean():6.3f} {e[ok].mean():6.3f} {(e-t)[ok].mean():6.3f} {np.sqrt(((e-t)[ok]**2).mean()):6.3f} {(e-t)[ok].std()/np.sqrt(ok.sum()):6.3f} {np.nanmean([x.get('cov_'+k,np.nan) for x in res]):5.3f} {mean['width_'+k]:6.3f} {(e-a)[ok].mean():6.3f}")
    e = np.array([x["est_ipw_trim"] for x in res]); t = np.array([x["true_trim"] for x in res]); print("ipw trim: bias", (e - t).mean(), "rmse", np.sqrt(((e - t) ** 2).mean()))
    e = np.array([x["est_ipw_all"] for x in res]); t = np.array([x["true_all"] for x in res]); print("ipw all: bias", (e - t).mean(), "rmse", np.sqrt(((e - t) ** 2).mean()))


if __name__ == "__main__":
    import glob
    OUT = "results/solutions/bart"
    if sys.argv[1] == "run":
        r = int(sys.argv[2]); os.makedirs(OUT, exist_ok=True)
        x = one((r, 1000, int(os.environ.get("BART_NMCMC", 2000)), int(os.environ.get("BART_CHAINS", 4))))
        json.dump(x, open(f"{OUT}/{r}.json", "w"))
    else:
        res = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/*.json"), key=lambda f: int(os.path.basename(f)[:-5]))]
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print("data sets", len(res)); summarize(res)
        print(buf.getvalue()); open("results/solutions/bart_summary.txt", "w").write(buf.getvalue())
