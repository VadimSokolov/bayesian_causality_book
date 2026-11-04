# Moscow election observers: the field experiment of Chapter 1, "Causal Questions" (label ch:predictive, section sec:fieldexp). Writes results/res_moscow.json and figs/moscow_observers.pdf.
# Data: data/PNAS_data_2011.csv, the replication file of Enikolopov, Korovkin, Petrova, Sonin and Zakharov (2013), one row per Moscow polling station (3,164 stations) in the Duma election of 4 December 2011.
# Treatment: s = 1 for the 185 stations initially assigned to observation (within each of the 125 territorial commissions, stations sorted by official number and every 25th taken starting from the first; the paper's Materials and Methods), 0 for the other 2,979. Observers attended only 156 of the 185 (all_observers = 1, all inside s = 1) for lack of volunteers, so the main contrast is by assignment (intention to treat). The column pseudo equals s; n_obs (244 stations) is not the assignment and is not used. We also report the ratio of the assignment effect to the share of assigned stations attended (the Wald ratio, the effect at an attended station if assignment acts only through attendance) and the raw comparison by attendance, the paper's 10.8 points.
# Outcomes: er_share, United Russia's share of all ballots cast (er / ballots, so invalid ballots are in the denominator, as in the paper), and turnout_share, ballots over registered voters. The other parties' shares are summarized in the JSON only.
# Model 1 (normal): within each arm the outcomes are independent normal with mean mu_t and standard deviation sigma_t, with the flat prior p(mu_t, log sigma_t) proportional to 1, so mu_t given the data is Student t with n_t - 1 degrees of freedom, location the arm mean and scale s_t / sqrt(n_t). 200,000 draws of each mu_t, default_rng(20111204); the effect is mu_1 - mu_0.
# Model 2 (Bayesian bootstrap): Dirichlet(1, ..., 1) weights over the stations of each arm, 20,000 draws of the two weighted means, same generator. It assumes no shape for the distribution of shares.
# Model 3 (district adjusted): least squares of the outcome on the treatment and indicators of the 125 territorial election commissions (column tik), flat prior on the coefficients and Jeffreys' prior on the error variance, so the treatment coefficient is Student t (exact, no draws).
# Reported: posterior mean, 95 percent interval and posterior probability that assignment lowered the outcome, for each model and outcome; for the normal model also the Wald ratio.
# Run from the repository root: PYTHONPATH=. python book_scripts/moscow_observers.py (a few seconds).
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
rng = np.random.default_rng(20111204)
d = pd.read_csv("data/PNAS_data_2011.csv")
T = d["s"].to_numpy()
A = d["all_observers"].to_numpy()
assert len(d) == 3164 and T.sum() == 185 and A.sum() == 156 and A[T == 0].sum() == 0
S_NORM, S_BB = 200_000, 20_000
R = {"n_stations": int(len(d)), "n_treated": int(T.sum()), "n_control": int((1 - T).sum()),
     "draws_normal": S_NORM, "draws_bb": S_BB, "seed": 20111204, "n_tik": int(d["tik"].nunique()), "n_attended": int(A.sum()), "treatment": "s (initial assignment)"}


def summ(x):
    x = np.asarray(x)
    return {"mean": float(x.mean()), "lo": float(np.quantile(x, 0.025)), "hi": float(np.quantile(x, 0.975)),
            "p_neg": float((x < 0).mean())}


for col in ["er_share", "turnout_share", "sr_share", "ldpr_share", "com_share", "apple_share"]:
    y = d[col].to_numpy()
    y1, y0 = y[T == 1], y[T == 0]
    out = {"mean_assigned": float(y1.mean()), "mean_unassigned": float(y0.mean()),
           "sd_assigned": float(y1.std(ddof=1)), "sd_unassigned": float(y0.std(ddof=1))}
    mu = {}
    for t, yy in ((1, y1), (0, y0)):
        n = len(yy)
        mu[t] = yy.mean() + yy.std(ddof=1) / np.sqrt(n) * rng.standard_t(n - 1, S_NORM)
    out["normal"] = summ(mu[1] - mu[0])
    bb = {}
    for t, yy in ((1, y1), (0, y0)):
        w = rng.dirichlet(np.ones(len(yy)), S_BB)
        bb[t] = w @ yy
    out["bb"] = summ(bb[1] - bb[0])
    X = np.column_stack([T, pd.get_dummies(d["tik"]).to_numpy(float)])
    beta, rss, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    nu = len(y) - rank
    s2 = float(np.sum((y - X @ beta) ** 2)) / nu
    se = np.sqrt(s2 * np.linalg.pinv(X.T @ X)[0, 0])
    tq = stats.t.ppf(0.975, nu)
    out["tik"] = {"mean": float(beta[0]), "lo": float(beta[0] - tq * se), "hi": float(beta[0] + tq * se),
                  "p_neg": float(stats.t.cdf(-beta[0] / se, nu)), "df": int(nu)}
    # Wald ratio, a sensitivity calculation: assignment effect over the attended share, Beta(157, 30) posterior for that share (uniform prior), drawn independently of the contrast; it assumes comparable assignment groups, no effect of assignment except through attendance, and no interference between stations (the paper reports spillovers to neighbouring stations)
    comp = rng.beta(A[T == 1].sum() + 1, (T == 1).sum() - A[T == 1].sum() + 1, S_NORM)
    out["wald"] = summ((mu[1] - mu[0]) / comp)
    out["attended_minus_rest_raw"] = float(y[A == 1].mean() - y[A == 0].mean())
    R[col] = out

# share of stations reporting United Russia above 60 percent, by assignment
R["er_above_60"] = {"assigned": float((d.er_share[T == 1] > 0.6).mean()), "unassigned": float((d.er_share[T == 0] > 0.6).mean())}

with open("results/res_moscow.json", "w") as fh:
    json.dump(R, fh, indent=1)
for k in ["er_share", "turnout_share"]:
    print(k, json.dumps(R[k], indent=None))
print(R["er_above_60"])

# Figure: United Russia's share at the stations assigned and not assigned to observers (histograms on a density scale).
fig, ax = plt.subplots(figsize=(5.2, 2.6))
bins = np.linspace(0, 1, 41)
ax.hist(d.er_share[T == 0], bins=bins, density=True, color="0.75", label="not assigned (2,979 stations)")
ax.hist(d.er_share[T == 1], bins=bins, density=True, histtype="step", color="k", lw=1.6,
        label="assigned to observers (185 stations)")
ax.set_xlabel("United Russia share of the vote")
ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig("figs/moscow_observers.pdf", metadata={"CreationDate": None})
