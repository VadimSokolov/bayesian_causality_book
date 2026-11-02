"""Reliability of the conduct probabilities and calibration of the posterior probability of a merger price increase above r.

(1) Conduct. Uses the 450 calibration industries of run_experiments.py (results/experiments.pkl, key "calibration": true conduct
and the classifier's probabilities at the summaries of each industry). Reports the Brier score sum_k (p_k - 1[c = k])^2 averaged over
industries (0 is perfect, 2/3 is the score of the uniform probabilities (1/3, 1/3, 1/3), and 0 for always right), the accuracy, and a
reliability table over 10 equal width bins of the predicted probability pooled over the three conducts (mean predicted, observed
frequency, count).
(2) Merger price effect. The calibration industries of run_experiments.py do not store the estimands, so the check uses the amortized
Bertrand reference table of mc.py (results/mc_table.pkl, prior draws, with the true estimands recorded): for each design T = 40 and
T = 15, GBCTargets is trained on the first 5,000 industries and applied to the last 1,000, held out. For r = 5, 12 and 15 percent the
predicted probability is the share of the 1,500 posterior draws of the merger price effect above r, the event is whether the true
effect exceeds r. Reports the Brier score, its value for the constant prediction at the base rate, and a reliability table over 10 bins
of equal count (mean predicted, observed frequency, count), and the mean absolute calibration gap. Writes results/res_calib_extra.json."""
import json, pickle, numpy as np
from estimators import GBCTargets

out = {}
R = pickle.load(open("results/experiments.pkl", "rb"))["calibration"]
P, c = np.asarray(R["probs"]), np.asarray(R["conduct_true"]); n = len(c)
Y1 = np.eye(3)[c]
bins = np.linspace(0, 1, 11); pp, yy = P.ravel(), Y1.ravel(); rel = []
for k in range(10):
    m = (pp >= bins[k]) & (pp < bins[k + 1] if k < 9 else pp <= 1)
    if m.sum():
        rel.append(dict(lo=float(bins[k]), hi=float(bins[k + 1]), n=int(m.sum()), mean_pred=float(pp[m].mean()), observed=float(yy[m].mean())))
out["conduct"] = dict(n=int(n), brier=float(((P - Y1) ** 2).sum(1).mean()), brier_uniform=2 / 3, accuracy=float((P.argmax(1) == c).mean()),
                      brier_by_conduct=[float(((P - Y1) ** 2).sum(1)[c == k].mean()) for k in range(3)], reliability=rel,
                      mean_abs_gap=float(sum(r_["n"] * abs(r_["mean_pred"] - r_["observed"]) for r_ in rel) / len(pp)))
print(json.dumps(out["conduct"], indent=1), flush=True)
tab = pickle.load(open("results/mc_table.pkl", "rb")); out["dp"] = {}
for T in (40, 15):
    Y, S = tab[T]["Y"], tab[T]["S"]
    gt = GBCTargets(S[:5000], Y[:5000]); res = {}
    post = [gt.posterior(S[i])[:, 13] for i in range(5000, len(S))]
    truth = Y[5000:, 13]
    for r in (0.05, 0.12, 0.15):
        pr = np.array([(x > r).mean() for x in post]); ev = (truth > r).astype(float)
        o = np.argsort(pr); groups = np.array_split(o, 10); rel = []
        for g in groups:
            rel.append(dict(n=int(len(g)), mean_pred=float(pr[g].mean()), observed=float(ev[g].mean())))
        base = ev.mean()
        res[str(r)] = dict(n=int(len(ev)), base_rate=float(base), brier=float(((pr - ev) ** 2).mean()), brier_base_rate=float(((base - ev) ** 2).mean()),
                           mean_abs_gap=float(sum(x["n"] * abs(x["mean_pred"] - x["observed"]) for x in rel) / len(ev)), reliability=rel)
        print(T, r, {k: v for k, v in res[str(r)].items() if k != "reliability"}, flush=True)
    out["dp"][str(T)] = res
json.dump(out, open("results/res_calib_extra.json", "w"), indent=1)
