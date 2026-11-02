"""Chapter 17 (NHEFS), item F7 of round 1 (October 2026): the Bayesian sensitivity analysis of Section sec:sensitivity (McCandless, Gustafson and Levy, 2007) applied to the effect of quitting smoking on weight gain.
Run from the repository root with PYTHONPATH=. (needs stochtree); writes results/res_nhefs_sensitivity.json, results/nhefs_effect_draws.npz (the posterior draws of the average effect) and figs/nhefs_sens.pdf.

Data and fits: the 1,566 smokers of apps2.py; the coupled horseshoe with effect modifiers (3,000 sweeps, 1,000 dropped, 2,000 draws, seed 1971) and the Bayesian causal forest (20 warm start sweeps and 1,000 draws, seed 3), with the settings of nhefs_validation.py; the draws of the average effect are saved.
Parameterization of the section: an unmeasured confounder U with unit variance shifts the estimated effect by a*b, where a is the coefficient of treatment in the regression of U on treatment and the measured covariates and b the effect of U on weight change in kilograms per standard deviation of U, so the effect given (a, b) is tau - a*b.
Benchmark: for each of the nine measured covariates (age, cigarettes per day, years of smoking, 1971 weight, sex, race, education, exercise, activity) take its component c_k of the additive least squares regression (all terms of the basis that involve the covariate, so squares are included), standardize it to unit variance, set b_k = sd(c_k) (kilograms per standard deviation of the component), and a_k = the coefficient of treatment in the regression of the standardized component on treatment and the terms of the other covariates.
The benchmark is the covariate with the largest |a_k b_k|. The prior at multiple m: a ~ N(0, (m a_k)^2) and b ~ N(0, (m b_k)^2) independent and symmetric (row "symmetric"), and a second row "adverse" in which the omitted confounder can only have inflated the estimate, a = m |a_k| |z_a| and b = m b_k |z_b| with z_a, z_b standard normal, so the effect is tau - a*b <= tau (the worst direction for the conclusion; both parameters are scaled by m, so the bias scales with m squared).
Each posterior draw of the effect is paired with 100 independent prior draws of (a, b). Multiples m = 0, 0.5, 1, 2, 4. Reported for each fit and prior: the posterior mean, the 95% interval, and P(effect > 0) and P(effect > 1 kg); the quantities for a grid of m from 0 to 6 go into the figure. The data cannot update (a, b), so the posterior of (a, b) is its prior and every number is conditional on that belief.
"""
import sys, json, time
import numpy as np
sys.path.insert(0, "book_scripts")
import nhefs_validation as nv
t0 = time.time(); n = nv.n; Y = nv.Y0; T = nv.T0
e = nv.propensity(T)
tau_hs, Zh, dr, ss = nv.fit_hs(Y, T, e, np.random.default_rng(1971))
tau_bcf, _ = nv.fit_bcf(Y, T, e, 3)
np.savez("results/nhefs_effect_draws.npz", horseshoe=tau_hs, bcf=tau_bcf)
print("fits", tau_hs.mean(), tau_bcf.mean(), time.time() - t0, flush=True)

# benchmark components from the additive least squares regression (treatment, intercept, Hernan Robins basis)
d = nv.d; X1 = np.column_stack([np.ones(n), T, nv.Xhr]); beta = np.linalg.lstsq(X1, Y, rcond=None)[0]; bx = beta[2:]
cols = {}
nm_num = ["age", "smokeintensity", "smokeyrs", "wt71"]
for i, nme in enumerate(nm_num): cols[nme] = [i, 4 + i]
off = 8
for nme in ["sex", "race", "education", "exercise", "active"]:
    k = len(np.unique(d[nme].astype(int))) - 1; cols[nme] = list(range(off, off + k)); off += k
assert off == nv.Xhr.shape[1]
bench = {}
for nme, idx in cols.items():
    c = nv.Xhr[:, idx] @ bx[idx]; sd = c.std(); z = (c - c.mean()) / sd
    other = [j for j in range(nv.Xhr.shape[1]) if j not in idx]
    Xo = np.column_stack([np.ones(n), T, nv.Xhr[:, other]]); a_k = float(np.linalg.lstsq(Xo, z, rcond=None)[0][1])
    bench[nme] = {"a": a_k, "b": float(sd), "product": a_k * float(sd)}
top = max(bench, key=lambda k: abs(bench[k]["product"])); ak, bk = bench[top]["a"], bench[top]["b"]
print("benchmark", top, bench[top], time.time() - t0, flush=True)

rng = np.random.default_rng(20261025); REP = 100
sgn = np.sign(ak * bk) if ak * bk != 0 else 1.0
def shifted(tau, m, kind):
    t = np.repeat(tau, REP)
    if kind == "symmetric":
        a = rng.normal(0, 1, t.size) * m * abs(ak); b = rng.normal(0, 1, t.size) * m * bk
        return t - a * b
    za = np.abs(rng.normal(size=t.size)); zb = np.abs(rng.normal(size=t.size))
    return t - m * abs(ak) * za * m * bk * zb                  # adverse: the omitted confounder inflates the estimate, so the effect is lowered by |a||b|
def summarize(x): return {"mean": float(x.mean()), "lo": float(np.quantile(x, .025)), "hi": float(np.quantile(x, .975)), "p_gt0": float((x > 0).mean()), "p_gt1": float((x > 1).mean())}
out = {"n": int(n), "benchmark_covariate": top, "benchmark_all": bench, "a_k": ak, "b_k": bk, "sign_of_ak_bk": float(sgn), "table": {}}
for fit, tau in [("horseshoe", tau_hs), ("bcf", tau_bcf)]:
    out["table"][fit] = {"posterior": summarize(tau)}
    for kind in ["symmetric", "adverse"]:
        out["table"][fit][kind] = {str(m): summarize(shifted(tau, m, kind)) for m in [0, 0.5, 1, 2, 4]}
grid = np.linspace(0, 6, 25); out["grid"] = grid.tolist(); out["curves"] = {}
for fit, tau in [("horseshoe", tau_hs), ("bcf", tau_bcf)]:
    for kind in ["symmetric", "adverse"]:
        S = [summarize(shifted(tau, m, kind)) for m in grid]
        out["curves"][f"{fit}_{kind}"] = {"p_gt0": [s["p_gt0"] for s in S], "p_gt1": [s["p_gt1"] for s in S]}
json.dump(out, open("results/res_nhefs_sensitivity.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.9), sharey=True)
for ax, kind, ttl in [(axs[0], "symmetric", "Symmetric prior"), (axs[1], "adverse", "Adverse direction")]:
    for fit, col in [("horseshoe", "C0"), ("bcf", "C3")]:
        ax.plot(grid, out["curves"][f"{fit}_{kind}"]["p_gt0"], color=col, label=f"{fit}, effect > 0")
        ax.plot(grid, out["curves"][f"{fit}_{kind}"]["p_gt1"], color=col, ls="--", label=f"{fit}, effect > 1 kg")
    ax.set_xlabel(f"Multiple of the strength of {top}"); ax.set_title(ttl, fontsize=9)
axs[0].set_ylabel("Posterior probability"); axs[0].legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig("figs/nhefs_sens.pdf"); plt.close()
print(json.dumps(out["table"], indent=1)); print("done", time.time() - t0)
