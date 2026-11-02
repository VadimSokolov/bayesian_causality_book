"""Merge the 210 parts of open_rates_hr.py: summaries, bootstrap intervals, slopes, figure.

Per cell (n, error type, method): median absolute error with a 95 percent bootstrap interval (2,000 resamples of the 200 replications),
mean coverage with its Monte Carlo standard error sqrt(c (1 - c) / 200), median interval width, RMSE. Slope of log median absolute error on
log n over n >= 400 by least squares on the five cell medians, with a 95 percent bootstrap interval (resample replications within each n,
recompute medians and the slope; 2,000 resamples, seed 7). Writes results/res_open_rates.json and figs/open_rates.pdf (t errors on 1.5
degrees of freedom: Student t posterior mean, DML, and the oracle line) and figs/open_rates_all.pdf (all three error types).
"""
import json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
NS = [100, 200, 400, 800, 1600, 3200, 6400]; ERR = ["t1.5", "t2", "gauss"]; METH = ["student", "gauss_post", "dml", "oracle"]
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
D = {}
for f in sorted(glob.glob("results/rates_hr/part_*.json")):
    d = json.load(open(f)); D.setdefault((d["n"], d["err"]), []).extend(d["reps"])
rng = np.random.default_rng(7); B = 2000; R = {"B": B, "cells": {}, "slopes": {}}
arr = {}
for (n, e), reps in D.items():
    for m in METH:
        a = np.array([r[m] for r in reps]); arr[(n, e, m)] = a
        err = np.abs(a[:, 0]); k = len(err)
        meds = np.median(err[rng.integers(0, k, (B, k))], axis=1)
        c = a[:, 1].mean()
        R["cells"][f"{e}|{n}|{m}"] = dict(reps=k, mad=float(np.median(err)), mad_lo=float(np.quantile(meds, .025)), mad_hi=float(np.quantile(meds, .975)),
                                          cover=float(c), cover_se=float(np.sqrt(c * (1 - c) / k)), n_cover=int(a[:, 1].sum()), width_med=float(np.median(a[:, 2])),
                                          rmse=float(np.sqrt((a[:, 0] ** 2).mean())))
big = [n for n in NS if n >= 400]; ln = np.log(big)
for e in ERR:
    for m in METH:
        if not all((n, e, m) in arr for n in big): continue
        errs = [np.abs(arr[(n, e, m)][:, 0]) for n in big]
        slope = np.polyfit(ln, np.log([np.median(x) for x in errs]), 1)[0]
        bs = []
        for _ in range(B):
            bs.append(np.polyfit(ln, np.log([np.median(x[rng.integers(0, len(x), len(x))]) for x in errs]), 1)[0])
        R["slopes"][f"{e}|{m}"] = dict(slope=float(slope), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)))
json.dump(R, open("results/res_open_rates.json", "w"), indent=1)
def plot(ax, e, oracle):
    for m, mk, lab in [("student", "o-", "Student $t$ posterior mean"), ("dml", "s--", "DML")] + ([("oracle", "^:", "oracle score")] if oracle else []):
        y = [R["cells"][f"{e}|{n}|{m}"]["mad"] for n in NS]
        lo = [R["cells"][f"{e}|{n}|{m}"]["mad_lo"] for n in NS]; hi = [R["cells"][f"{e}|{n}|{m}"]["mad_hi"] for n in NS]
        s = R["slopes"][f"{e}|{m}"]["slope"]
        ax.errorbar(NS, y, yerr=[np.array(y) - lo, np.array(hi) - y], fmt=mk, ms=4, capsize=2, label=f"{lab}, slope {s:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(NS, 0.4 / np.sqrt(NS), ":", color="grey", label="$n^{-1/2}$"); ax.plot(NS, 0.6 * np.array(NS, float) ** (-1 / 3), "-.", color="grey", label="$n^{-1/3}$")
    ax.set_xlabel("$n$"); ax.set_ylabel("median absolute error"); ax.legend(frameon=False, fontsize=7)
fig, ax = plt.subplots(figsize=(5.2, 3.1)); plot(ax, "t1.5", True); fig.tight_layout(); fig.savefig("figs/open_rates.pdf"); plt.close()
fig, axs = plt.subplots(1, 3, figsize=(9, 2.9), sharey=True)
for ax, e in zip(axs, ERR): plot(ax, e, True); ax.set_title({"t1.5": "$t$, 1.5 df", "t2": "$t$, 2 df", "gauss": "Gaussian"}[e], fontsize=9)
fig.tight_layout(); fig.savefig("figs/open_rates_all.pdf"); plt.close()
for e in ERR:
    print(e, {k.split("|")[1]: v for k, v in R["slopes"].items() if k.startswith(e)})
