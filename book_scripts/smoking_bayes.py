"""Bayesian illustration for the smoking chapter (m10.tex).

An illustration with stated priors, not an analysis of any cohort.

Part A. The bias model of Bounds and Sensitivity (m09.tex, section sec:sensitivity) at the
observed risk ratios RR = 9 and 60. A binary unmeasured genotype U has prevalence f1 among
smokers and f0 = f1/PR among nonsmokers (PR is the prevalence ratio) and multiplies the cancer
risk by R in both groups; PR >= 1 and R >= 1 by the coding of Cornfield's theorem (U = 1 raises
the risk and is the commoner state among smokers). Within levels of U, smoking has risk ratio RR/B,
    B = (1 + f1 (R - 1)) / (1 + f0 (R - 1)),
the factor in the proof of Cornfield's inequality. In a large cohort the data pin down RR, so by
Proposition prop:partial the posterior of the effect is the prior on (f1, PR, R) pushed through B.
Under a continuous prior the probability of exactly no effect is zero; the quantity reported is the
probability of no positive effect, Pr(B >= RR), the prior mass of Cornfield's region. Without
exogeneity PN is bounded, not identified: each prior draw implies the experimental margin Pr(y_x')
and so the Tian and Pearl bounds of m09.tex (computed here from the cell probabilities), and
PN = L + u (H - L) with the position u uniform on (0, 1), independent of the draw. Inputs plugged
in for the observational cells: Pr(smoker) = 0.5 and the risk of nonsmokers 0.0002, the risk of
smokers being RR times that; the bounds do not depend on them over the grid checked below.

Part B. One simulated data set from the front door example of m10.tex (the data generating
process of make_figs.py, section 11), Beta(1, 1) priors on P(s), P(m | s) and P(c | m, s),
each posterior draw pushed through the front door formula to give a posterior for the
risk difference; the truth is computed exactly from the equations.

Writes results/res_smoking_bayes.json and book_tex/chapters/smoking_bayes_table.tex.
Run: timeout 600 python3 book_scripts/smoking_bayes.py   (NumPy only, about five seconds).
"""
import json
import os
import time

import numpy as np

T0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 1959
Z975 = 1.959963984540054          # standard normal 97.5th percentile
Z75 = 0.6744897501960817          # standard normal 75th percentile
expit = lambda z: 1.0 / (1.0 + np.exp(-z))
kids = np.random.SeedSequence(SEED).spawn(5)

# ---------------------------------------------------------------- settings of part A
RRS = (9, 60)                                   # observed risk ratios of the chapter
# log PR and log R are half normal, log X = s |Z|, with s set so that the 95th percentile of X
# is q95. f1 = Pr(U = 1 | smoker) is uniform on (0, 1) under both priors; f0 = f1 / PR.
# The twin informed prior bounds PR (twin studies speak to smoking, not to cancer) and keeps
# the sceptical prior for R. The values are illustrative, not elicited.
PRIORS = {
    "Sceptical": dict(q95_pr=50.0, q95_r=50.0),
    "Twin informed": dict(q95_pr=4.0, q95_r=50.0),
}
PI_SMOKE, P0 = 0.5, 2e-4        # cohort inputs for the Tian and Pearl bounds: Pr(smoker), risk of nonsmokers
N_DRAWS = 4_000_000             # prior draws per prior
N_CHECK = N_DRAWS               # draws used for the cell-probability check of the bounds (all of them)


def s_of(q95):
    """Scale of the half normal on the log scale whose 95th percentile of X is q95."""
    return np.log(q95) / Z975


def bias_factor(f1, pr, r):
    """Factor by which U inflates the risk ratio: (1 + f1 (R-1)) / (1 + f0 (R-1)), f0 = f1/PR."""
    return (1 + f1 * (r - 1)) / (1 + (f1 / pr) * (r - 1))


def max_bias(pr, r):
    """Largest bias factor at given PR and R (reached at f1 = 1): PR R / (PR + R - 1)."""
    return pr * r / (pr + r - 1)


def evalue(rr):
    return rr + np.sqrt(rr * (rr - 1))


def f1_needed(pr, r, rr):
    """Smallest f1 with B >= rr at given PR and R (Cornfield's region is f1 above this)."""
    return (rr - 1) * pr / ((r - 1) * (pr - rr))


def f1_min_feasible(pr, r, rr, p0):
    """Smallest f1 for which both stratum risks at the cohort inputs lie in [0, 1]:
    f1 >= max{0, (p1 R - 1)/(R - 1), PR (p0 R - 1)/(R - 1)} with p1 = rr p0 (smokers with U = 1 and nonsmokers with U = 1)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (rr * p0 * r - 1) / (r - 1)
        b = pr * (p0 * r - 1) / (r - 1)
        out = np.maximum(0.0, np.maximum(a, b))
    return np.where(np.isfinite(out), out, 0.0)


def _grid(lo, hi, s, m):
    a = np.linspace(lo, hi, m)
    w = np.full(m, a[1] - a[0])
    w[[0, -1]] /= 2
    return a, w * 2 / (s * np.sqrt(2 * np.pi)) * np.exp(-a ** 2 / (2 * s ** 2))


def region_mass(rr, s_pr, s_r, m=2500, k=9.0, p0=None):
    """Pr(B >= rr) under f1 ~ U(0,1) and half normal log PR, log R, by a product trapezoid rule.
    Cornfield's inequality puts the region inside PR > rr, R > rr; for such (PR, R) the region in f1
    is [f1_needed, 1], which has prior mass 1 - f1_needed when that is positive.
    With p0 given, the prior is truncated to the draws whose stratum risks lie in [0, 1] at the cohort
    inputs (nonsmoker risk p0, smoker risk rr p0): the region in f1 becomes [max(f1_needed, fmin), 1] and
    the result is divided by the feasible prior mass, the integral of 1 - fmin."""
    lo = np.log(rr)
    if k * s_pr <= lo or k * s_r <= lo:
        return 0.0
    a, da = _grid(lo, k * s_pr, s_pr, m)
    b, db = _grid(lo, k * s_r, s_r, m)
    pr, r = np.exp(a), np.exp(b)
    tot = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for i0 in range(0, m, 250):
            fn = f1_needed(pr[i0:i0 + 250, None], r[None, :], rr)
            if p0 is not None:
                fn = np.maximum(fn, f1_min_feasible(pr[i0:i0 + 250, None], r[None, :], rr, p0))
            mass = 1.0 - fn
            mass = np.where(np.isfinite(mass), np.clip(mass, 0.0, 1.0), 0.0)
            tot += float(da[i0:i0 + 250] @ mass @ db)
    if p0 is None:
        return tot
    a0, da0 = _grid(0.0, k * s_pr, s_pr, m)
    b0, db0 = _grid(0.0, k * s_r, s_r, m)
    pr0, r0 = np.exp(a0), np.exp(b0)
    den = 0.0
    for i0 in range(0, m, 250):
        mass = np.clip(1.0 - f1_min_feasible(pr0[i0:i0 + 250, None], r0[None, :], rr, p0), 0.0, 1.0)
        den += float(da0[i0:i0 + 250] @ mass @ db0)
    return tot / den


def tian_pearl_cells(rr, f1, pr, r, pi, p0):
    """Tian and Pearl bounds on PN from the cell probabilities one prior draw implies.
    Observational cells: Pr(x) = pi, Pr(y | x') = p0, Pr(y | x) = rr p0. The experimental margin
    Pr(y_x') is the risk if nobody smoked, sum_u Pr(u) Pr(y | x', u)."""
    f0 = f1 / pr
    p1 = rr * p0
    r00 = p0 / (1 + f0 * (r - 1))                   # risk of a nonsmoker with U = 0
    f = pi * f1 + (1 - pi) * f0                      # Pr(U = 1)
    py_x0 = r00 * (1 + f * (r - 1))                  # Pr(y_x')
    Pxy, Pxp_yp = pi * p1, (1 - pi) * (1 - p0)       # Pr(x, y), Pr(x', y')
    Py = pi * p1 + (1 - pi) * p0                     # Pr(y)
    lo = np.maximum(0.0, (Py - py_x0) / Pxy)
    hi = np.minimum(1.0, ((1 - py_x0) - Pxp_yp) / Pxy)
    return lo, hi


def infeasible_fraction(rr, f1, pr, r, p0):
    """Share of draws that would need a stratum risk above one at the cohort inputs."""
    f0 = f1 / pr
    r11 = rr * p0 * r / (1 + f1 * (r - 1))           # smokers with U = 1
    r01 = p0 * r / (1 + f0 * (r - 1))                # nonsmokers with U = 1
    return float(np.mean((r11 > 1) | (r01 > 1)))


# ---------------------------------------------------------------- part A
res = dict(seed=SEED, n_draws=N_DRAWS)
res["e_values"] = {str(rr): float(evalue(rr)) for rr in RRS}
res["exogeneity_lower_bound"] = {str(rr): 1 - 1 / rr for rr in RRS}
res["cohort_inputs"] = dict(pr_smoker=PI_SMOKE, risk_nonsmokers=P0)
chk = {}
for rr in RRS:                                       # the E value is the common strength PR = R with max bias = rr
    e = evalue(rr)
    assert abs(max_bias(e, e) - rr) < 1e-9
    chk[f"max_bias_at_evalue_rr{rr}"] = float(max_bias(e, e))
    chk[f"f1_needed_at_evalue_rr{rr}"] = float(f1_needed(e, e, rr))
res["priors"] = {}
res["bias_model"] = {}
for (name, spec), kid in zip(PRIORS.items(), kids[:2]):
    rng = np.random.default_rng(kid)
    s_pr, s_r = s_of(spec["q95_pr"]), s_of(spec["q95_r"])
    res["priors"][name] = dict(spec, scale_log_pr=float(s_pr), scale_log_r=float(s_r),
                               median_pr=float(np.exp(Z75 * s_pr)), median_r=float(np.exp(Z75 * s_r)),
                               f1="uniform(0, 1)")
    f1 = rng.uniform(size=N_DRAWS)
    pr = np.exp(s_pr * np.abs(rng.standard_normal(N_DRAWS)))
    r = np.exp(s_r * np.abs(rng.standard_normal(N_DRAWS)))
    u = rng.uniform(size=N_DRAWS)                    # position of PN within the Tian and Pearl bounds
    f1_all, pr_all, r_all, u_all = f1, pr, r, u
    res["bias_model"][name] = {}
    for rr in RRS:
        # truncate the prior to draws whose stratum risks lie in [0, 1] at the cohort inputs and renormalize
        keep = f1_all >= f1_min_feasible(pr_all, r_all, rr, P0)
        f1, pr, r, u = f1_all[keep], pr_all[keep], r_all[keep], u_all[keep]
        n_keep = int(keep.sum())
        B = bias_factor(f1, pr, r)
        # Cornfield's inequality on every draw: B cannot exceed PR or R, and B >= 1 because PR, R >= 1
        assert np.all(B <= np.minimum(pr, r) * (1 + 1e-12)) and np.all(B >= 1 - 1e-12)
        inreg = B >= rr
        assert np.all(pr[inreg] >= rr) and np.all(r[inreg] >= rr)      # the theorem, draw by draw
        quad = region_mass(rr, s_pr, s_r, p0=P0)
        mc = float(np.mean(inreg))
        assert abs(mc - quad) <= 5 * np.sqrt(max(quad, 1.0 / n_keep) / n_keep), (name, rr, mc, quad)
        L = np.maximum(0.0, 1 - B / rr)              # Tian and Pearl lower bound; the upper bound is 1 (checked below)
        pn = L + u * (1 - L)
        # check the closed form against the cell-probability bounds over a grid of cohort inputs, on all kept draws
        err, hmin = 0.0, 1.0
        for pi in (0.2, 0.5, 0.8):
            for p0 in (1e-4, 2e-4, 5e-4, 1e-3):
                lo_c, hi_c = tian_pearl_cells(rr, f1, pr, r, pi, p0)
                err = max(err, float(np.max(np.abs(lo_c - L))))
                hmin = min(hmin, float(hi_c.min()))
        assert err < 1e-8 and hmin == 1.0, (name, rr, err, hmin)
        rrt = rr / B
        res["bias_model"][name][str(rr)] = dict(
            p_no_positive_effect=quad, p_no_positive_effect_mc=mc, n_draws_kept=n_keep, n_check=int(len(f1)),
            mc_se=float(np.sqrt(mc * (1 - mc) / n_keep)),
            adjusted_rr_median=float(np.median(rrt)),
            adjusted_rr_q025=float(np.quantile(rrt, 0.025)), adjusted_rr_q975=float(np.quantile(rrt, 0.975)),
            L_mean=float(L.mean()), p_L_gt_half=float(np.mean(L > 0.5)),
            pn_mean=float(pn.mean()), pn_q025=float(np.quantile(pn, 0.025)),
            pn_q975=float(np.quantile(pn, 0.975)), p_pn_gt_half=float(np.mean(pn > 0.5)),
            infeasible_risk_fraction=infeasible_fraction(rr, f1_all, pr_all, r_all, P0))
        chk[f"tp_lower_bound_max_abs_error_{name}_rr{rr}"] = err
        chk[f"tp_upper_bound_min_{name}_rr{rr}"] = hmin
res["checks"] = chk

# how the twin informed answer depends on the stated bound on PR (R keeps its sceptical prior)
sweep = {}
for q in (2, 3, 4, 6, 9, 15, 25, 50):
    sweep[str(q)] = {str(rr): region_mass(rr, s_of(q), s_of(50.0), p0=P0) for rr in RRS}
res["twin_bound_sweep_p_no_positive_effect"] = sweep

# ---------------------------------------------------------------- part B: the front door
N_FD, TAU = 20000, 0.3                               # sample size and Pr(U = 1) of the chapter's simulation


def fd_data(rng, n=N_FD):
    """The data generating process of make_figs.py, section 11 (U hidden)."""
    u = rng.binomial(1, TAU, n)
    s = rng.binomial(1, expit(-0.5 + 2.0 * u))
    m = rng.binomial(1, expit(-2 + 4 * s))
    c = rng.binomial(1, expit(-3 + 1.5 * m + 1.5 * u))
    return s, m, c


def fd_truth():
    """Risk difference Pr(c | do(S=1)) - Pr(c | do(S=0)) computed exactly from the equations."""
    def risk(sv):
        pm1 = expit(-2 + 4 * sv)
        return sum(pm * sum(pu * expit(-3 + 1.5 * mv + 1.5 * uv) for uv, pu in ((0, 1 - TAU), (1, TAU)))
                   for mv, pm in ((0, 1 - pm1), (1, pm1)))
    return risk(1) - risk(0)


def fd_posterior(s, m, c, rng, draws):
    """Beta(1,1) posteriors for Pr(s), Pr(m | s) and Pr(c | m, s) pushed through the front door formula."""
    n = len(s)
    th = rng.beta(1 + s.sum(), 1 + n - s.sum(), draws)                       # Pr(S = 1)
    al = {sv: rng.beta(1 + m[s == sv].sum(), 1 + (s == sv).sum() - m[s == sv].sum(), draws) for sv in (0, 1)}
    ga = {}
    for mv in (0, 1):
        for sv in (0, 1):
            k = (m == mv) & (s == sv)
            ga[mv, sv] = rng.beta(1 + c[k].sum(), 1 + k.sum() - c[k].sum(), draws)   # Pr(C = 1 | M = mv, S = sv)

    def fd(sv):
        tot = 0.0
        for mv in (0, 1):
            pm = al[sv] if mv == 1 else 1 - al[sv]
            tot = tot + pm * (ga[mv, 0] * (1 - th) + ga[mv, 1] * th)         # sum over s' of Pr(c | m, s') Pr(s')
        return tot
    naive = (rng.beta(1 + c[s == 1].sum(), 1 + (s == 1).sum() - c[s == 1].sum(), draws)
             - rng.beta(1 + c[s == 0].sum(), 1 + (s == 0).sum() - c[s == 0].sum(), draws))
    return fd(1) - fd(0), naive


truth = fd_truth()
s, m, c = fd_data(np.random.default_rng(kids[2]))
post, naive_post = fd_posterior(s, m, c, np.random.default_rng(kids[3]), 200000)
iv = lambda x: [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]
# frequentist calibration of the 95 percent interval over repeated data sets
rng_rep, cover, width, R_REP = np.random.default_rng(kids[4]), 0, 0.0, 500
for _ in range(R_REP):
    p_, _n = fd_posterior(*fd_data(rng_rep), rng_rep, 4000)
    lo_, hi_ = iv(p_)
    cover += lo_ <= truth <= hi_
    width += hi_ - lo_
res["front_door"] = dict(
    n=N_FD, truth=float(truth), posterior_mean=float(post.mean()), posterior_sd=float(post.std()),
    posterior_interval95=iv(post), truth_in_interval=bool(iv(post)[0] <= truth <= iv(post)[1]),
    naive_contrast=float(c[s == 1].mean() - c[s == 0].mean()),
    naive_posterior_mean=float(naive_post.mean()), naive_posterior_interval95=iv(naive_post),
    share_smokers=float(s.mean()), share_cancer=float(c.mean()),
    repeated=dict(data_sets=R_REP, coverage=float(cover) / R_REP, mean_interval_width=float(width) / R_REP))

# ---------------------------------------------------------------- the table fragment
def fmt_p(p):
    """Probability with two significant digits; scientific notation below 0.001; '< 10^-6' below 1e-6."""
    if p < 1e-6:
        return "$< 10^{-6}$"
    if p >= 0.001:
        return f"{p:.2g}"
    mant, ex = f"{p:.1e}".split("e")
    return f"${mant} \\times 10^{{{int(ex)}}}$"


rows = []
for rr in RRS:
    rows.append(f"\\midrule\n\\multicolumn{{5}}{{@{{}}l}}{{Observed risk ratio {rr}, E value {res['e_values'][str(rr)]:.1f}}} \\\\")
    for name in PRIORS:
        d = res["bias_model"][name][str(rr)]
        rows.append(f"{name} & {fmt_p(d['p_no_positive_effect'])} & {d['p_L_gt_half']:.3f} & "
                    f"{d['pn_mean']:.3f} & [{d['pn_q025']:.3f}, {d['pn_q975']:.3f}] \\\\")
table = "\n".join([
    "\\begin{table}[ht]",
    "\\centering",
    "\\caption{Limiting posterior of the bias model of Section~\\ref{sec:sensitivity} under two illustrative priors for the genotype. "
    "$L$ is the Tian and Pearl lower bound on $\\PN$ without exogeneity, and $\\PN$ has its position within the bounds uniform. "
    f"The probability of no positive effect comes from numerical integration and the other summaries from {N_DRAWS:,} prior draws, both restricted to values whose stratum risks lie between zero and one, with a cohort of smoking prevalence {PI_SMOKE:g} and nonsmoker risk {P0:g}.}}",
    "\\label{tab:smokingbayes}",
    "\\small",
    "\\begin{tabular}{@{}lcccc@{}}",
    "\\toprule",
    "Prior & \\shortstack{$\\Prob$(no positive\\\\effect)} & $\\Prob(L > 1/2)$ & \\shortstack{$\\PN$\\\\mean} & \\shortstack{$\\PN$ 95\\%\\\\interval} \\\\",
    *rows,
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}",
]) + "\n"

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "res_smoking_bayes.json"), "w") as fh:
    json.dump(res, fh, indent=1)
with open(os.path.join(ROOT, "book_tex", "chapters", "smoking_bayes_table.tex"), "w") as fh:
    fh.write(table)

# ---------------------------------------------------------------- printed record
def show(path, val):
    print(f"{path:75s} {val}")

print(f"seed {SEED}, {N_DRAWS} prior draws per prior; cohort inputs Pr(smoker) = {PI_SMOKE}, nonsmoker risk = {P0}")
for rr in RRS:
    show(f"e_values.{rr}", f"{res['e_values'][str(rr)]:.4f}")
    show(f"exogeneity_lower_bound.{rr}", f"{res['exogeneity_lower_bound'][str(rr)]:.4f}")
    show(f"checks.f1_needed_at_evalue_rr{rr}", f"{chk[f'f1_needed_at_evalue_rr{rr}']:.5f}")
for name in PRIORS:
    p = res["priors"][name]
    show(f"priors.{name}", f"PR median {p['median_pr']:.2f}, R median {p['median_r']:.2f}, "
                           f"95th percentiles {p['q95_pr']:g} and {p['q95_r']:g}")
    for rr in RRS:
        d = res["bias_model"][name][str(rr)]
        for key, val in d.items():
            show(f"bias_model.{name}.{rr}.{key}", f"{val:.6g}")
show("twin_bound_sweep (q95 of PR: P(no positive effect) at RR 9, 60)", "")
for q, v in sweep.items():
    show(f"  q95_pr = {q}", f"{v['9']:.3e}  {v['60']:.3e}")
for key, val in chk.items():
    show(f"checks.{key}", f"{val:.6g}")
for key, val in res["front_door"].items():
    show(f"front_door.{key}", val)
print(f"table written; {time.time() - T0:.1f} seconds")
