# Urns, Sanov rates and the case mix example for Chapter 4, "Groups, Urns and Information" (label ch:groups). Data free. Three figures and results/res_groups_urn_tau.json.
# (1) Sanov against inverse Sanov: p0 = 0.30 and the set {p >= q = 0.45}. The sampling rate is -(1/n) log P(phat >= q | p0), the exact binomial tail; the posterior rate is -(1/n) log Pi(p >= q | phat = p0), the exact Beta(1 + p0 n, 1 + (1 - p0) n) tail from a flat prior, for n = 20, 40, ..., 1600. The limits are KL(q || p0) and KL(p0 || q), and all four numbers in the figure are computed. The relative difference of the two limits is also stored for q = 0.60 and q = 0.90.
# (2) Heath and Sudderth: total variation between k draws without replacement from an urn of N units with N/2 ones (hypergeometric) and k independent draws with probability 1/2 (binomial), N = 100, 1000 and 10,000, 40 log spaced values of k from 2 to N/2 (duplicates dropped), against the bound min(2k/n, 1). That is ck/n for c = 2 values in total variation, the norm of the curves. Diaconis and Freedman state 2ck/n in their norm, which is twice the total variation, so 4k/n for c = 2 there and 2k/n here. The legend writes the urn size as n.
# (3) Case mix: every unit has the same conditional odds ratio exp(psi), psi = 1, and baseline log odds mu + s z with z standard normal, so the average effect over the baseline distribution, a case mix, is tau_Q = E[expit(mu + s z + psi) - expit(mu + s z)]. It is computed with 400,000 draws of z (default_rng(1), one set shared by every row and every point of the figure) for six (mu, s) rows and for the curves s = 0.5, 1.5 and 3.0 on 71 values of mu from -4 to 3. The line is tanh(psi/4) = expit(psi/2) - expit(-psi/2), the largest risk difference any baseline can show for that odds ratio.
# Writes: results/res_groups_urn_tau.json and figs/sanov_vs_inverse.pdf, heath_sudderth_tv.pdf, tau_case_mix.pdf (each 6.4 in wide or less, text at least 8.5 pt, gray styles: solid black, dashed 0.3, dash-dot 0.55).
# Run from the repository root: python book_scripts/groups_urn_tau.py (about 2 seconds). The kidney stone analysis and its figures are in groups_kidney.py.
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import hypergeom, binom, beta

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
OUT, RES = "figs/", "results/"
T0 = time.time()
R = {}


def plain(x):
    if isinstance(x, dict): return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    if isinstance(x, np.ndarray): return plain(x.tolist())
    if isinstance(x, np.bool_): return bool(x)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    return x


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT + name, metadata={"CreationDate": None}); plt.close(fig)


# ------------------------------------------------------------------ 1. Sanov against inverse Sanov
p0, q = 0.30, 0.45
kl = lambda a, b: a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
nn = np.arange(20, 1601, 20)
fwd = np.array([binom.logsf(int(np.ceil(q * n - 1e-9)) - 1, n, p0) / n for n in nn])        # P(phat >= q | p0)
inv = np.array([beta.logsf(q, 1 + p0 * n, 1 + (1 - p0) * n) / n for n in nn])                # Pi(p >= q | data with phat = p0)
KL1, KL2 = kl(q, p0), kl(p0, q)
R["sanov"] = dict(p0=p0, q=q, KL_q_p0=KL1, KL_p0_q=KL2, rel_diff=(KL1 - KL2) / KL1, rate_n1600=dict(sanov=-fwd[-1], inverse=-inv[-1]), n_grid="20 to 1600 step 20",
                  tails={f"q{qq:.2f}": dict(KL_q_p0=kl(qq, p0), KL_p0_q=kl(p0, qq), rel_diff=(kl(qq, p0) - kl(p0, qq)) / kl(qq, p0)) for qq in (0.45, 0.60, 0.90)})
print("sanov", {k: v for k, v in R["sanov"].items() if k != "tails"}, "tails", R["sanov"]["tails"])
fig, ax = plt.subplots(figsize=(6.4, 3.3))
ax.plot(nn, -fwd, color="k", ls="-", lw=2.0, label=r"Sanov: $-\frac{1}{n}\log P(\hat p\geq%.2f\mid p=%.2f)$" % (q, p0))
ax.plot(nn, -inv, color="0.3", ls="--", lw=2.0, label=r"inverse Sanov: $-\frac{1}{n}\log \Pi(p\geq%.2f\mid \hat p=%.2f)$" % (q, p0))
ax.axhline(KL1, color="k", ls=":", lw=1.4, label=r"limit $\mathrm{KL}(%.2f\Vert%.2f)=%.4f$" % (q, p0, KL1))
ax.axhline(KL2, color="0.55", ls="-.", lw=1.4, label=r"limit $\mathrm{KL}(%.2f\Vert%.2f)=%.4f$" % (p0, q, KL2))
ax.set_xlabel("$n$"); ax.set_ylabel("exponential rate"); ax.set_ylim(0.04, 0.075)
ax.legend(loc="upper right", frameon=False, fontsize=8.5, handlelength=3)
save(fig, "sanov_vs_inverse.pdf")

# ------------------------------------------------------------------ 2. Heath and Sudderth urns
R["heath_sudderth"] = {}
fig, ax = plt.subplots(figsize=(6.4, 3.3))
sty = {100: dict(color="0.6", lw=2.6), 1000: dict(color="0.35", lw=2.0), 10000: dict(color="k", lw=1.6)}
bcol = {100: "0.45", 1000: "0.3", 10000: "0.1"}
for Nu in (100, 1000, 10000):
    r = Nu // 2; k = np.unique(np.rint(np.logspace(np.log10(2), np.log10(Nu // 2), 40)).astype(int))
    tv = np.array([0.5 * np.abs(hypergeom.pmf(np.arange(kk + 1), Nu, r, kk) - binom.pmf(np.arange(kk + 1), kk, r / Nu)).sum() for kk in k])
    bound = np.minimum(2 * k / Nu, 1)           # ck/n for c = 2 in total variation, the norm of the curves (Diaconis and Freedman state 2ck/n in a norm twice as large, which is 4k/n for c = 2)
    # max_tv_over_bound is the largest ratio of the curve to the bound drawn, min(2k/n, 1). The ratio is the same in either norm. On this grid it peaks at the last point k = n/2, where the bound is 1, so it equals tv_last
    R["heath_sudderth"][Nu] = dict(k_first=k[0], k_last=k[-1], tv_over_kN_first=tv[0] / (k[0] / Nu), tv_over_kN_last=tv[-1] / (k[-1] / Nu), max_tv_over_bound=(tv / bound).max(),
                                   loglog_slope_first10=np.polyfit(np.log(k[:10]), np.log(tv[:10]), 1)[0], tv_first=tv[0], tv_last=tv[-1])
    ax.loglog(k, tv, ls="-", label=f"exact, $n={Nu}$", **sty[Nu])
    ax.loglog(k, bound, ls="--", color=bcol[Nu], lw=1.1)
ax.plot([], [], color="0.3", ls="--", lw=1.1, label=r"bound $2k/n$")
ax.set_xlabel(r"treated units observed $k$"); ax.set_ylabel("total variation")
ax.legend(loc="lower right", frameon=False, fontsize=8.5, handlelength=3)
save(fig, "heath_sudderth_tv.pdf")
print("heath_sudderth", {Nu: {a: round(float(b), 4) for a, b in v.items()} for Nu, v in R["heath_sudderth"].items()})

# ------------------------------------------------------------------ 3. same odds ratio, different average effects
psi = 1.0; seed = 1; draws = 400000
z = np.random.default_rng(seed).standard_normal(draws)
tau = lambda m, s: np.mean(expit(m + s * z + psi) - expit(m + s * z))
rows = []
for m, s in [(-3, .5), (-2, .5), (-.5, .5), (-.5, 2), (-.5, 3), (2, .5)]:
    r0, r1 = np.mean(expit(m + s * z)), np.mean(expit(m + s * z + psi))
    rows.append(dict(mu=m, s=s, control_risk=r0, tau=tau(m, s), odds_ratio=np.exp(psi), marginal_odds_ratio=(r1 / (1 - r1)) / (r0 / (1 - r0))))
    print(m, s, round(r0, 3), round(rows[-1]["tau"], 3), round(rows[-1]["marginal_odds_ratio"], 3))
taus = np.array([r["tau"] for r in rows])
R["tau"] = dict(psi=psi, draws=draws, seed=seed, bound_tanh=np.tanh(psi / 4), range_ratio=taus.max() / taus.min(), rows=rows)
print("bound tanh(psi/4) =", np.tanh(psi / 4), "range ratio", taus.max() / taus.min())
mus = np.linspace(-4, 3, 71)
fig, ax = plt.subplots(figsize=(6.0, 3.1))
for s, st in [(0.5, dict(color="k", ls="-", lw=1.8)), (1.5, dict(color="0.3", ls="--", lw=1.8)), (3.0, dict(color="0.55", ls="-.", lw=2.0))]:
    ax.plot(mus, [tau(m, s) for m in mus], label=f"spread {s}", **st)
ax.axhline(np.tanh(psi / 4), color="0.45", lw=1.0, ls=":")
ax.text(mus[0] + 0.05, np.tanh(psi / 4) + 0.004, r"bound $\tanh(\psi/4)$", fontsize=8.5, color="0.25", va="bottom")
ax.set_xlabel(r"mean baseline log odds $\mu$"); ax.set_ylabel(r"$\tau_Q$ (risk difference)"); ax.set_ylim(0, 0.28)
ax.legend(loc="lower center", frameon=False, fontsize=8.5, handlelength=3)
save(fig, "tau_case_mix.pdf")
with open(RES + "res_groups_urn_tau.json", "w") as fh: json.dump(plain(R), fh, indent=1)
print("done in %.0f s" % (time.time() - T0))
