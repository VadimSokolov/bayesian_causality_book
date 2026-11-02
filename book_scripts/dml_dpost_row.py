"""Chapter 7, Table tab:crit:head: the dense posterior row. Of 1,000 replications in results/dml_dpost, 103 did not converge and 7 converged with standard errors of at least 1; the row summarizes the other 890. Writes results/res_dml_dpost_row.json."""
import glob, json, math
import numpy as np

R = [json.loads(l)["rf"] for f in sorted(glob.glob("results/dml_dpost/dense_*.jsonl")) for l in open(f)]
conv = [x for x in R if x["converged"]]
keep = [x for x in conv if x["se"] < 1]
th, se, lo, hi = (np.array([x[k] for x in keep]) for k in ("theta", "se", "lo", "hi"))
out = dict(n_reps=len(R), n_converged=len(conv), n_used=len(keep), excluded_se=sorted(x["se"] for x in conv if x["se"] >= 1), bias=float(th.mean() - 1), coverage=float(((lo <= 1) & (hi >= 1)).mean()), se_over_sd=float(se.mean() / th.std(ddof=1)), se_over_rmse=float(se.mean() / math.sqrt(((th - 1) ** 2).mean())))
json.dump(out, open("results/res_dml_dpost_row.json", "w"), indent=1)
print(out)
