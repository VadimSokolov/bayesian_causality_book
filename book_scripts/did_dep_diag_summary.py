"""Summary of results/res_did_dep_diag.json (did_dep_diag.py merge): bias, RMSE and coverage with Monte Carlo standard errors for each variant,
paired difference of the error against the base horseshoe, shrinkage factors and global scales.  Writes results/res_did_dep_diag_summary.json and prints it."""
import json, numpy as np
R = json.load(open("results/res_did_dep_diag.json")); VAR = ["base", "long", "tau_a", "tau_b", "tau_c", "reg3", "reg10", "flat", "oracle"]
out = {}
for key, rows in R.items():
    n = len(rows); d = {"n_panels": n, "mean_n_depart": float(np.mean([r["n_dep"] for r in rows])), "variants": {}}
    base = np.array([r["base"]["err"] for r in rows])
    for v in VAR:
        e = np.array([r[v]["err"] for r in rows]); lo = np.array([r[v]["lo"] for r in rows]); hi = np.array([r[v]["hi"] for r in rows])
        cov = (lo <= 0) & (hi >= 0); diff = e - base
        d["variants"][v] = dict(bias=e.mean(), bias_se=e.std(ddof=1) / np.sqrt(n), rmse=float(np.sqrt((e ** 2).mean())),
                                cover=cov.mean(), cover_se=float(np.sqrt(cov.mean() * (1 - cov.mean()) / n)), width=float((hi - lo).mean()),
                                bias_minus_base=diff.mean(), bias_minus_base_se=diff.std(ddof=1) / np.sqrt(n))
    shr = {}
    for v in ("base", "long", "tau_a", "tau_b", "tau_c", "reg3", "reg10", "flat"):
        s = {}
        for k in ("dep4", "dep8", "dep12", "dep_all", "nodep_adopt", "nodep_nevertreated"):
            x = np.array([r[v]["shr"][k] for r in rows if r[v]["shr"][k] is not None], float)
            s[k] = (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), len(x))
        t = np.array([r[v]["tau"] for r in rows]); s["tau_trend"] = (float(np.median(t[:, 2])), float(t[:, 2].mean())); s["tau_dev"] = float(np.median(t[:, 0])); s["tau_lead"] = float(np.median(t[:, 1]))
        shr[v] = s
    d["shrinkage"] = shr
    # bias split: panels with at least one departing unit in cohort 4 etc. is not needed; share of departing treated cells by cohort is design
    out[key] = d
json.dump(out, open("results/res_did_dep_diag_summary.json", "w"), indent=1, default=float)
for key, d in out.items():
    print(key, d["n_panels"], "panels, mean departing units", round(d["mean_n_depart"], 1))
    for v, e in d["variants"].items():
        print(f"  {v:8s} bias {e['bias']:.3f} ({e['bias_se']:.3f}) rmse {e['rmse']:.3f} cover {e['cover']:.2f} ({e['cover_se']:.2f}) width {e['width']:.2f}  vs base {e['bias_minus_base']:.3f} ({e['bias_minus_base_se']:.3f})")
    for v, s in d["shrinkage"].items():
        print(f"  shrink {v:8s} " + " ".join(f"{k}={s[k][0]:.2f}({s[k][1]:.2f})" for k in ("dep4", "dep8", "dep12", "dep_all", "nodep_adopt", "nodep_nevertreated")) + f" tau_trend med {s['tau_trend'][0]:.3f} mean {s['tau_trend'][1]:.3f} tau_dev {s['tau_dev']:.3f} tau_lead {s['tau_lead']:.3f}")
