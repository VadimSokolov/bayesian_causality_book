"""Coverage of intervals for the average effect as overlap deteriorates (Chapter 6, Section on weights under weak overlap).

Design of Figure fig:ipw: n = 1000, X ~ N(0, 1), propensity e(x) = expit(s x) known, slope s = 0.5, 1, ..., 4, T | X ~ Bernoulli(e(X)),
outcome Y = g(X) + T + eps with eps ~ N(0, 1), so the average effect is tau = 1. Two outcome functions: g(x) = x (the design of the figure) and
g(x) = x + 0.25 x^3 (the cubic term makes every linear outcome regression misspecified). 20,000 replications for each slope and each g.
Estimators of tau and their 95 percent intervals:
  HT          Horvitz and Thompson with the true propensity; influence function standard error.
  Hajek       the normalised version; influence function standard error.
  oracle AIPW the augmented score with the true outcome means and the true propensity; influence function standard error.
  AIPW 5 fold the augmented score with arm by arm linear regressions of Y on (1, X) fitted on the other four folds, known propensity;
              influence function standard error.
  Bayes lin   Bayesian linear regression of Y on (1, T, X), N(0, 10^2) priors on the three coefficients, noise variance known (equal to 1);
              posterior mean and the central 95 percent credible interval of the coefficient of T.
Reported for each estimator, slope and g: coverage of tau (binomial standard error), median interval width (bootstrap standard error of the
median), and bias (Monte Carlo standard error). Fixed seeds: the seed of configuration c is SeedSequence([20261023, c]).
Usage, from the repository root (PYTHONPATH=.):
  python book_scripts/ipw_overlap_coverage.py run C [REPS]   configuration C = 0..15 (g index * 8 + slope index): results/ipw_overlap_cov/cfg_CC.json
  python book_scripts/ipw_overlap_coverage.py agg            results/res_ipw_overlap_coverage.json and figs/ipw_overlap_coverage.pdf
"""
import glob, json, os, sys
import numpy as np

N, SLOPES = 1000, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
G = [lambda x: x, lambda x: x + 0.25 * x ** 3]
NAMES = ["Horvitz Thompson", "Hajek", "oracle AIPW", "AIPW 5 fold", "Bayes linear"]
Z = 1.959963984540054
OUTD = "results/ipw_overlap_cov/"


def block(rng, gi, s, R):
    """R replications: returns the estimates and standard errors, shape (R, 5) each."""
    X = rng.standard_normal((R, N)); e = 1 / (1 + np.exp(-s * X)); T = (rng.random((R, N)) < e).astype(float)
    g = G[gi](X); Y = g + T + rng.standard_normal((R, N))
    est = np.empty((R, 5)); se = np.empty((R, 5))
    def if_se(psi): return psi.std(1, ddof=1) / np.sqrt(N)
    w1 = T / e; w0 = (1 - T) / (1 - e)
    ht = (w1 * Y - w0 * Y); est[:, 0] = ht.mean(1); se[:, 0] = if_se(ht)
    m1 = (w1 * Y).sum(1) / w1.sum(1); m0 = (w0 * Y).sum(1) / w0.sum(1); est[:, 1] = m1 - m0
    psi = w1 * (Y - m1[:, None]) / w1.mean(1)[:, None] - w0 * (Y - m0[:, None]) / w0.mean(1)[:, None]; se[:, 1] = if_se(psi)
    def aipw(mu1, mu0):
        return mu1 - mu0 + w1 * (Y - mu1) - w0 * (Y - mu0)
    psi = aipw(g + 1, g); est[:, 2] = psi.mean(1); se[:, 2] = if_se(psi)
    fold = np.arange(N) % 5; mu1 = np.empty((R, N)); mu0 = np.empty((R, N))
    for k in range(5):
        tr = (fold != k)[None, :]; te = fold == k
        for arm, mu in ((1.0, mu1), (0.0, mu0)):
            m = tr * (T == arm)
            c = m.sum(1); sx = (m * X).sum(1); sy = (m * Y).sum(1); sxx = (m * X * X).sum(1); sxy = (m * X * Y).sum(1)
            det = c * sxx - sx ** 2 + 1e-9; b1 = (c * sxy - sx * sy) / det; b0 = (sy - b1 * sx) / np.maximum(c, 1)
            mu[:, te] = b0[:, None] + b1[:, None] * X[:, te]
    psi = aipw(mu1, mu0); est[:, 3] = psi.mean(1); se[:, 3] = if_se(psi)
    A = np.stack([np.ones((R, N)), T, X], axis=2)                       # (R, N, 3)
    prec = np.einsum("rni,rnj->rij", A, A) + np.eye(3) / 100.0; V = np.linalg.inv(prec)
    mean = np.einsum("rij,rj->ri", V, np.einsum("rni,rn->ri", A, Y))
    est[:, 4] = mean[:, 1]; se[:, 4] = np.sqrt(V[:, 1, 1])
    return est, se


def run(c, reps=20000):
    gi, si = divmod(c, 8); s = SLOPES[si]; rng = np.random.default_rng(np.random.SeedSequence([20261023, c]))
    E, S = [], []
    for _ in range(reps // 1000):
        e_, s_ = block(rng, gi, s, 1000); E.append(e_); S.append(s_)
    E = np.concatenate(E) - 1.0; S = np.concatenate(S); cover = np.abs(E) <= Z * S; width = 2 * Z * S
    bs = np.random.default_rng(1); out = dict(g=gi, slope=s, reps=len(E), names=NAMES)
    out["coverage"] = cover.mean(0).tolist(); out["coverage_se"] = np.sqrt(cover.mean(0) * (1 - cover.mean(0)) / len(E)).tolist()
    out["median_width"] = np.median(width, 0).tolist()
    out["median_width_se"] = [float(np.std([np.median(width[bs.integers(0, len(E), len(E)), j]) for _ in range(100)], ddof=1)) for j in range(5)]
    out["bias"] = E.mean(0).tolist(); out["bias_se"] = (E.std(0, ddof=1) / np.sqrt(len(E))).tolist()
    out["sd"] = E.std(0, ddof=1).tolist(); out["mean_se"] = S.mean(0).tolist()
    os.makedirs(OUTD, exist_ok=True); json.dump(out, open(OUTD + "cfg_%02d.json" % c, "w"), indent=1)
    print(c, gi, s, [round(v, 3) for v in out["coverage"]], flush=True)


def agg():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    cf = [json.load(open(f)) for f in sorted(glob.glob(OUTD + "cfg_*.json"))]
    json.dump(cf, open("results/res_ipw_overlap_coverage.json", "w"), indent=1)
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 4.8), sharex=True)
    sty = [("k", "-", "o"), ("0.4", "--", "s"), ("0.6", ":", "^"), ("0.2", "-.", "v"), ("k", "-", "D")]
    for gi in range(2):
        rows = sorted([r for r in cf if r["g"] == gi], key=lambda r: r["slope"]); sl = [r["slope"] for r in rows]
        for j, nm in enumerate(NAMES):
            col, ls, mk = sty[j]; kw = dict(color=col, ls=ls, marker=mk, ms=3.5, lw=1.2, mfc="white" if j == 4 else col, label=nm)
            axs[0, gi].errorbar(sl, [r["coverage"][j] for r in rows], yerr=[1.96 * r["coverage_se"][j] for r in rows], capsize=1.5, **kw)
            axs[1, gi].plot(sl, [r["median_width"][j] for r in rows], **kw)
        axs[0, gi].axhline(0.95, color="grey", lw=0.6); axs[0, gi].set_ylim(0, 1.02)
        axs[0, gi].set_title(["g(x) = x", "g(x) = x + 0.25 x$^3$"][gi], fontsize=9); axs[1, gi].set_yscale("log"); axs[1, gi].set_xlabel("selection slope s")
    axs[0, 0].set_ylabel("coverage of 95% interval"); axs[1, 0].set_ylabel("median width"); axs[0, 0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout(); fig.savefig("figs/ipw_overlap_coverage.pdf")


if __name__ == "__main__":
    if sys.argv[1] == "run": run(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 20000)
    else: agg()
