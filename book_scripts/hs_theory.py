"""Appendix on posterior guarantees: horseshoe risk in the sparse normal means model (quadratic and predictive risk) and, new in the October 2026 rerun
(review item F10), the coverage of horseshoe credible sets.

Model. x_i = theta_i + sqrt(v) z_i, i = 1..n, n = 2000, z_i standard normal, theta_i = A for i <= p and 0 otherwise. Horseshoe prior theta_i | lambda_i ~ N(0, lambda_i^2 g^2),
lambda_i independent standard half Cauchy, global scale g fixed at its oracle value p / n (not estimated, so no hyperprior and no truncation of the global scale enters).
The posterior of theta_i given x_i is a scale mixture over lambda_i: given lambda, N(x_i s2/(v+s2), v s2/(v+s2)), s2 = (lambda g)^2. The mixing integral is a sum over
1,200 log spaced values of lambda in [1e-4, 1e4] (the quadrature; the stage `grid` doubles it to 2,400 and compares).
Panels (each design is 500 replications; seeds are recorded as default_rng([2014, panel, p, k]) with panel 1 = quadratic, 2 = predictive, 3 = uncertainty and k the index of v or a):
  quad  A = 1.5 sqrt(2 log(n/p)), v = 1, p in {5, 10, 20, 40, 80, 160}: the mean over replications of the squared error of the posterior mean, with its Monte Carlo s.e.
  pred  p in {5, 20, 80}, A = 1.5 sqrt(2 log(n/p)): the KL risk of the horseshoe predictive density through the George, Liang and Xu identity with v_x = v_y = 1: half the integral over
        v in [0.5, 1] of risk(v) / v^2, by the trapezoid rule on six values of v, the six squared errors of a replication sharing the same z (common random numbers).
  unc   p in {5, 20, 80}, A = a sqrt(2 log(n/p)), a = 0.25, 0.5, ..., 2.5, v = 1: the 95 percent credible ball {theta: |theta - theta_hat| <= L r} with theta_hat the posterior mean and r the 0.95
        quantile of |theta - theta_hat| over 400 posterior draws, for L = 1 and L = 2 (coverage of the true theta), and the 95 percent central posterior interval of each of the p nonzero
        coordinates (exact mixture distribution function: the interval covers theta_i when the posterior cdf at theta_i is in [0.025, 0.975]); marginal coverage is the mean over the nonzero
        coordinates and replications. Monte Carlo s.e. of a ball coverage is sqrt(c(1-c)/500); of the marginal coverage the standard deviation over replications of the per replication
        fraction divided by sqrt(500).
Usage (repository root): python book_scripts/hs_theory.py quad P | pred P | unc P | grid | agg ; `agg` writes results/res_hs_theory.json (same keys as before: quad, pred; new: unc, grid)
and the four macros of appG.tex into results/hs_theory_macros.tex. Parts are in results/hs_theory_parts/."""
import sys, os, json, numpy as np

n = 2000; R = 500; V = 1.0
os.makedirs("results/hs_theory_parts", exist_ok=True)


def make_grid(m=1200):
    lg = np.linspace(np.log(1e-4), np.log(1e4), m); lam = np.exp(lg)
    wl = lam * (2 / np.pi) / (1 + lam ** 2) * (lg[1] - lg[0])        # half Cauchy density times d lambda on a log grid
    return lam, wl


def weights(x, tau, v, lam, wl):
    s2 = (lam * tau) ** 2; tot = v + s2
    L = -0.5 * np.log(tot)[None, :] - 0.5 * (x[:, None] ** 2) / tot[None, :] + np.log(wl)[None, :]
    L -= L.max(1, keepdims=True); W = np.exp(L); W /= W.sum(1, keepdims=True)
    return W, s2, tot


def hs_mean(x, tau, v, lam, wl):
    W, s2, tot = weights(x, tau, v, lam, wl)
    return x * (W @ (s2 / tot))


def stage_quad(p):
    lam, wl = make_grid(); tau = p / n; A = np.sqrt(2 * np.log(n / p)) * 1.5
    rng = np.random.default_rng([2014, 1, p, 0]); th = np.zeros(n); th[:p] = A; loss = []
    for _ in range(R):
        x = th + rng.normal(size=n); loss.append(np.sum((hs_mean(x, tau, 1.0, lam, wl) - th) ** 2))
    loss = np.array(loss)
    json.dump(dict(p=p, risk=float(loss.mean()), se=float(loss.std(ddof=1) / np.sqrt(R)), plog=p * np.log(n / p), minimax=2 * p * np.log(n / p), R=R),
              open(f"results/hs_theory_parts/quad_p{p}.json", "w"))


def stage_pred(p):
    lam, wl = make_grid(); tau = p / n; A = np.sqrt(2 * np.log(n / p)) * 1.5; vs = np.linspace(0.5, 1.0, 6)
    rng = np.random.default_rng([2014, 2, p, 0]); th = np.zeros(n); th[:p] = A; kl = []
    for _ in range(R):
        z = rng.normal(size=n)
        l = np.array([np.sum((hs_mean(th + np.sqrt(v) * z, tau, v, lam, wl) - th) ** 2) for v in vs])
        kl.append(0.5 * np.trapezoid(l / vs ** 2, vs))
    kl = np.array(kl)
    json.dump(dict(p=p, kl=float(kl.mean()), se=float(kl.std(ddof=1) / np.sqrt(R)), plog=p * np.log(n / p), bound_shape=0.5 * p * np.log(n / p) * np.log(2), R=R),
              open(f"results/hs_theory_parts/pred_p{p}.json", "w"))


def one_rep(x, th, tau, lam, wl, rng, ndraw=400):
    """Posterior mean, credible ball radius, and the cdf of each true nonzero coordinate for one data set."""
    from scipy.special import ndtr
    W, s2, tot = weights(x, tau, V, lam, wl)
    mu = x[:, None] * (s2 / tot)[None, :]; sd = np.sqrt(s2 * V / tot)
    m = (W * mu).sum(1)
    cw = np.cumsum(W, 1); cw[:, -1] = 1.0
    u = rng.uniform(size=(ndraw, n)); idx = np.empty((ndraw, n), dtype=np.int64)
    for i in range(n):
        idx[:, i] = np.searchsorted(cw[i], u[:, i])
    dr = mu[np.arange(n)[None, :], idx] + sd[idx] * rng.normal(size=(ndraw, n))
    r = np.quantile(np.sqrt(((dr - m[None, :]) ** 2).sum(1)), 0.95)
    return m, r, W, mu, sd


def stage_unc(p):
    from scipy.special import ndtr
    lam, wl = make_grid(); tau = p / n; rows = []
    for k, a in enumerate(np.arange(1, 11) * 0.25):
        A = a * np.sqrt(2 * np.log(n / p)); rng = np.random.default_rng([2014, 3, p, k])
        th = np.zeros(n); th[:p] = A; c1, c2, mg, rad, err = [], [], [], [], []
        for _ in range(R):
            x = th + rng.normal(size=n)
            m, r, W, mu, sd = one_rep(x, th, tau, lam, wl, rng)
            e = np.sqrt(((m - th) ** 2).sum()); c1.append(e <= r); c2.append(e <= 2 * r); rad.append(r); err.append(e)
            F = (W[:p] * ndtr((th[:p, None] - mu[:p]) / sd[None, :])).sum(1)
            mg.append(np.mean((F >= 0.025) & (F <= 0.975)))
        c1, c2, mg = np.array(c1, float), np.array(c2, float), np.array(mg)
        rows.append(dict(a=float(a), A=float(A), ball_L1=float(c1.mean()), ball_L1_se=float(np.sqrt(c1.mean() * (1 - c1.mean()) / R)),
                         ball_L2=float(c2.mean()), ball_L2_se=float(np.sqrt(c2.mean() * (1 - c2.mean()) / R)),
                         marginal=float(mg.mean()), marginal_se=float(mg.std(ddof=1) / np.sqrt(R)), mean_radius=float(np.mean(rad)), mean_error=float(np.mean(err)), R=R))
        print(p, rows[-1], flush=True)
    json.dump(dict(p=p, rows=rows), open(f"results/hs_theory_parts/unc_p{p}.json", "w"))


def stage_grid():
    """Doubling check of the local scale quadrature: 1,200 against 2,400 log spaced values, 100 replications at p = 20, a = 1 and 2:
    the largest difference in the posterior mean (max over coordinates and replications), in the cdf of the true nonzero coordinates, and in the credible ball radius
    (the same posterior draws are not shared, so the radius differs by Monte Carlo noise of about 1 percent as well; the cdf and the mean are exact)."""
    from scipy.special import ndtr
    p = 20; tau = p / n; out = []
    l1, w1 = make_grid(1200); l2, w2 = make_grid(2400)
    for a in (1.0, 2.0):
        A = a * np.sqrt(2 * np.log(n / p)); rng = np.random.default_rng([2014, 4, p, int(a)]); th = np.zeros(n); th[:p] = A; dm, dc, dn = 0.0, 0.0, 0.0
        for _ in range(100):
            x = th + rng.normal(size=n); res = []
            for lam, wl in ((l1, w1), (l2, w2)):
                W, s2, tot = weights(x, tau, V, lam, wl); mu = x[:, None] * (s2 / tot)[None, :]; sd = np.sqrt(s2 * V / tot)
                m = (W * mu).sum(1); F = (W[:p] * ndtr((th[:p, None] - mu[:p]) / sd[None, :])).sum(1)
                var = (W * (sd ** 2 + mu ** 2)).sum(1) - m ** 2
                res.append((m, F, np.sqrt(var.sum())))
            dm = max(dm, float(np.abs(res[0][0] - res[1][0]).max())); dc = max(dc, float(np.abs(res[0][1] - res[1][1]).max()))
            dn = max(dn, float(abs(res[0][2] - res[1][2]) / res[0][2]))
        out.append(dict(a=a, max_abs_diff_posterior_mean=dm, max_abs_diff_cdf=dc, max_rel_diff_posterior_sd_norm=dn))
    json.dump(out, open("results/hs_theory_parts/grid.json", "w")); print(out)


def stage_agg():
    res = {"quad": [], "pred": [], "unc": []}
    for p in (5, 10, 20, 40, 80, 160):
        res["quad"].append(json.load(open(f"results/hs_theory_parts/quad_p{p}.json")))
    for p in (5, 20, 80):
        res["pred"].append(json.load(open(f"results/hs_theory_parts/pred_p{p}.json")))
        res["unc"].append(json.load(open(f"results/hs_theory_parts/unc_p{p}.json")))
    if os.path.exists("results/hs_theory_parts/grid.json"):
        res["grid"] = json.load(open("results/hs_theory_parts/grid.json"))
    q = [d["risk"] / d["plog"] for d in res["quad"]]; k = [d["kl"] / d["plog"] for d in res["pred"]]
    res["macros"] = dict(pgqmin=round(min(q), 2), pgqmax=round(max(q), 2), pgkmin=round(min(k), 2), pgkmax=round(max(k), 2))
    json.dump(res, open("results/res_hs_theory.json", "w"), indent=1)
    open("results/hs_theory_macros.tex", "w").write("".join(f"\\newcommand{{\\{a}}}{{{b:.2f}}}\n" for a, b in res["macros"].items()))
    print(json.dumps(res["macros"]))


if __name__ == "__main__":
    st = sys.argv[1]
    {"quad": lambda: stage_quad(int(sys.argv[2])), "pred": lambda: stage_pred(int(sys.argv[2])), "unc": lambda: stage_unc(int(sys.argv[2])),
     "grid": stage_grid, "agg": stage_agg}[st]()
