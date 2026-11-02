"""Generative Bayesian computation for an average effect: a conjugate causal model with an exact posterior.

Worked example for the chapter on computation and generative Bayesian computation. The design is the stylized supplement trial of
Chapter 16: n = 200 children, a covariate Z_i ~ N(0,1) drawn once and held fixed, an assignment W_i drawn once from
Bernoulli(expit(G0 + G1 Z_i)) and held fixed, and Y_i = mu_w + alpha_w Z_i + eps_i with w = W_i and eps_i ~ N(0, 1), sigma known.
The parameters theta = (mu_0, alpha_0, mu_1, alpha_1) have the prior N(0, 4 I). The targets are the model based average effect
psi = (mu_1 - mu_0) + (alpha_1 - alpha_0) Zbar and the conditional effect tau(z) = (mu_1 - mu_0) + (alpha_1 - alpha_0) z. Both are linear
in theta and the two arms are independent conjugate normal regressions, so the exact posterior is normal with closed form mean and
variance. The summary S (the four within arm least squares coefficients) is sufficient given the design and sigma, so every error of a
trained network is amortization error and not summary loss.

Networks: the quantile network of qnn_book.py (cosine embedding of the level, product with an MLP embedding of the summary, MLP head,
K = 32, width 64), trained on N simulated pairs (theta from the prior, data from the model, S from the data) with the check loss, 40 epochs.
Inputs and targets are divided by their training standard deviations, which changes nothing but the conditioning of the optimization.

What it computes, by stage (each stage writes its own part of results/res_gbc_ate.json):
  ate    (1) the ATE network against the exact posterior at one observed data set (mean, sd, 5/50/95 percent points, Wasserstein-1 on
         999 levels); (2) the rank check over 500 simulated studies with 100 draws each (10 bin histogram, chi squared, coverage of the
         central 50 and 90 percent intervals), for the network and for the exact posterior; (4) the held out check loss of the network and
         of the exact posterior quantile function, the excess, and the check that twice the excess equals the Cramer distance;
         (5) the rank histogram of the 50 studies with the largest Mahalanobis distance of S against the rest, and an extrapolation ray.
  blind  (3) a network trained on psi with S replaced by a constant (it learns the prior): flat ranks, the prior's interval width,
         its held out loss and excess.
  cate   (6) a network with input (level, S, z) for tau(z), z ~ N(0,1) per training pair, against the exact posterior at z = -1.5, 0, 1.5.
  joint  (7) a triangular pair (Delta mu given S; Delta alpha given S and Delta mu) and a marginal network (Delta alpha given S):
         the posterior sd of tau(z) from joint draws and from independent draws against the exact one.
  fig    figs/gbc_ate.pdf from the JSON of the other stages.
  scale2 (extra) October 2026 rerun for the chapter on computation (item F16): ATE and CATE networks at N = 10,000 to 400,000, ten seeds (one per call, --seed k),
         nested training sets, 2,000 held out studies, Wasserstein-1 over the exact posterior sd (median and 90th percentile over studies), coverage; scale2agg collects it.
  scale  (extra, not part of `all`) the ATE network at N = 10,000 to 400,000 with three training seeds: how the error falls with N.
Design check: a grid of assignment intercepts and slopes with the exact posterior correlation of (Delta mu, Delta alpha), stored in
the JSON under design. With G0 = 0 the correlation is about zero for every slope (the arms are mirror images), so the intercept G0 is
what makes independent draws misstate the spread of tau(z). G0 = 1.0 is the smallest intercept (steps of 0.25, either sign) at
the specified slope 0.8 for which independent draws misstate the sd of tau(z) by at least 20% at z = -1.5 or 1.5.

Trained networks are cached in --cache (outside the repository); delete it or pass --retrain to retrain from scratch. A cached network
is used only if design, N, epochs and seed match. Run from the repository root, one stage per command:
  timeout 600 python3 -I /abs/path/book_scripts/gbc_ate.py --stage ate     (then blind, cate, joint, fig)
Produces results/res_gbc_ate.json and figs/gbc_ate.pdf. Requires numpy, scipy, matplotlib and torch (CPU only).
Every stage records the design and a hash of this file (script_sha256_12) in its part of the JSON, so a stale part can be detected.
Run time: about 90 s for ate, blind, cate, joint and fig together (11 s per network at N = 100,000), about 4 min for scale; Apple M2, 4 threads.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import special, stats
import torch, torch.nn as nn

torch.set_num_threads(2)  # the order of the sums inside a multithreaded kernel follows the thread count
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "pdf.fonttype": 42})
PAL = ["#1f5a96", "#d1495b", "#edae49", "#00798c", "#66a182", "#8d5fa8"]

n, SIGMA, PRIOR_SD = 200, 1.0, 2.0          # units, known noise sd, prior sd of each coordinate of theta
G0, G1 = 1.0, 0.8                            # P(W=1 | Z) = expit(G0 + G1 Z); G0 = 0 (mirror image arms) gives a zero posterior correlation of the effect components
SEED = 2025
ZS = (-1.5, 0.0, 1.5)
LEV = np.arange(1, 1000) / 1000.0            # 999 quantile levels
L99 = np.arange(1, 100) / 100.0
NRANK, LRANK, NHELD, KLEV, NDRAW = 500, 100, 20000, 8, 20000


# ---------------------------------------------------------------- the model and its exact posterior
def post_stats(Z, W):
    """Design matrix, least squares map, exact posterior covariance and posterior mean map of the conjugate model."""
    D = np.zeros((n, 4))
    for w in (0, 1):
        D[W == w, 2 * w] = 1.0
        D[W == w, 2 * w + 1] = Z[W == w]
    DtD = D.T @ D
    A = np.linalg.solve(DtD, D.T)                                        # S = A y: least squares within each arm
    V = np.linalg.inv(DtD / SIGMA ** 2 + np.eye(4) / PRIOR_SD ** 2)      # posterior covariance (does not depend on the data)
    M = V @ DtD / SIGMA ** 2                                             # posterior mean = M S
    return dict(D=D, A=A, V=V, M=M, Z=Z, W=W, zbar=float(Z.mean()))


def make_design(g0, g1, seed=SEED):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)
    W = (rng.uniform(size=n) < special.expit(g0 + g1 * Z)).astype(int)
    return post_stats(Z, W)


def effect_moments(V):
    """Posterior moments of (Delta mu, Delta alpha) = (mu_1 - mu_0, alpha_1 - alpha_0); the arms are independent."""
    vm, va, cv = V[2, 2] + V[0, 0], V[3, 3] + V[1, 1], V[2, 3] + V[0, 1]
    return dict(var_dmu=float(vm), var_dal=float(va), cov=float(cv), rho=float(cv / np.sqrt(vm * va)))


def sd_tau(e, z, joint=True):
    return float(np.sqrt(e["var_dmu"] + z * z * e["var_dal"] + (2 * z * e["cov"] if joint else 0.0)))


def design_section(DES):
    c = np.array([-1.0, -DES["zbar"], 1.0, DES["zbar"]]); e = effect_moments(DES["V"])

    def row(g0, g1):
        d = make_design(g0, g1); ee = effect_moments(d["V"]); cc = np.array([-1.0, -d["zbar"], 1.0, d["zbar"]])
        return dict(g0=float(g0), g1=float(g1), n1=int(d["W"].sum()), rho=ee["rho"], sd_psi=float(np.sqrt(cc @ d["V"] @ cc)),
                    ratio_m15=sd_tau(ee, -1.5, False) / sd_tau(ee, -1.5), ratio_p15=sd_tau(ee, 1.5, False) / sd_tau(ee, 1.5))
    scan_intercept = [row(g0, 0.8) for g0 in np.arange(-2.0, 2.01, 0.25)]          # slope 0.8 as specified, intercept varied
    scan_slope = [row(0.0, g1) for g1 in (0.4, 0.8, 1.6, 2.4, 4.0)]                  # intercept 0 as specified, slope varied
    rho0 = np.array([effect_moments(make_design(0.0, 0.8, sd)["V"])["rho"] for sd in range(300)])
    W, Z = DES["W"], DES["Z"]
    return dict(n=n, sigma=SIGMA, prior_sd=PRIOR_SD, g0=G0, g1=G1, seed=SEED, n_treated=int(W.sum()), n_control=int((1 - W).sum()),
                zbar=DES["zbar"], mean_z_treated=float(Z[W == 1].mean()), mean_z_control=float(Z[W == 0].mean()),
                prior_sd_psi=float(PRIOR_SD * np.linalg.norm(c)), post_sd_psi=float(np.sqrt(c @ DES["V"] @ c)),
                post_sd_dmu=float(np.sqrt(e["var_dmu"])), post_sd_dal=float(np.sqrt(e["var_dal"])), post_rho=e["rho"],
                exact_sd_tau={str(z): sd_tau(e, z) for z in ZS}, indep_sd_tau={str(z): sd_tau(e, z, False) for z in ZS},
                indep_over_exact_sd_tau={str(z): sd_tau(e, z, False) / sd_tau(e, z) for z in ZS},
                shrinkage_factor=float(PRIOR_SD * np.linalg.norm(c) / np.sqrt(c @ DES["V"] @ c)),
                scan_intercept_at_slope_0_8=scan_intercept, scan_slope_at_intercept_0=scan_slope,
                spec_design_300_seeds=dict(rho_quantiles_5_25_50_75_95=np.quantile(rho0, [0.05, 0.25, 0.5, 0.75, 0.95]),
                                           mean_abs_rho=float(np.abs(rho0).mean()), max_abs_rho=float(np.abs(rho0).max())))


def simulate(DES, theta, rng, chunk=20000):
    """Draw the data at each row of theta from the model and return the sufficient summary S (least squares coefficients)."""
    out = np.empty_like(theta)
    for i in range(0, len(theta), chunk):
        th = theta[i:i + chunk]
        out[i:i + chunk] = (th @ DES["D"].T + SIGMA * rng.normal(size=(len(th), n))) @ DES["A"].T
    return out


def rng_for(key):
    return np.random.default_rng([SEED, key])


def prior_pairs(DES, N, key):
    rng = rng_for(key)
    theta = PRIOR_SD * rng.normal(size=(N, 4))
    return theta, simulate(DES, theta, rng)


def exact_post(DES, S, C):
    """Exact posterior mean and sd of c'theta given S; S is (m, 4), C is (4,) or (m, 4)."""
    S = np.atleast_2d(S); C = np.broadcast_to(np.atleast_2d(C), S.shape)
    return np.einsum("ij,ij->i", S @ DES["M"].T, C), np.sqrt(np.einsum("ij,jk,ik->i", C, DES["V"], C))


def make_sets(DES):
    """The observed data set, the rank check studies and the held out pairs, fixed by seeds and identical in every stage."""
    s = {}
    r = rng_for(21)
    s["theta_obs"] = PRIOR_SD * r.normal(size=(1, 4))
    s["Y_obs"] = s["theta_obs"] @ DES["D"].T + SIGMA * r.normal(size=(1, n))
    s["S_obs"] = s["Y_obs"] @ DES["A"].T
    s["theta_r"], s["S_r"] = prior_pairs(DES, NRANK, 22)
    s["theta_h"], s["S_h"] = prior_pairs(DES, NHELD, 23)
    r = rng_for(24); s["U_r"] = r.uniform(size=(NRANK, LRANK)); s["V_r"] = r.uniform(size=NRANK)
    s["U_h"] = rng_for(25).uniform(size=(NHELD, KLEV))
    s["z_r"] = rng_for(26).normal(size=NRANK)
    s["U_obs"] = rng_for(27).uniform(size=(3, NDRAW))
    return s


# ---------------------------------------------------------------- the quantile network (as in qnn_book.py)
class QNN(nn.Module):
    """f(tau, h): cosine embedding of tau, MLP embedding of h, elementwise product, MLP head."""
    def __init__(self, dh, width=64, K=32):
        super().__init__()
        self.register_buffer("k", torch.arange(1, K + 1).float() * np.pi)
        self.tau_net = nn.Sequential(nn.Linear(K, width), nn.ReLU())
        self.h_net = nn.Sequential(nn.Linear(dh, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, tau, h):
        e = self.tau_net(torch.cos(tau[:, None] * self.k))
        return self.head(e * self.h_net(h)).squeeze(-1)


def pinball(u, tau):
    return torch.mean(torch.maximum(tau * u, (tau - 1) * u))


class Fitted:
    """A trained QNN with the scaling of its inputs and target; f(u, H) returns quantiles on the original scale."""
    def __init__(self, net, hm, hs, ym, ys, hist, secs, cached=False):
        self.net, self.hm, self.hs, self.ym, self.ys = net, np.asarray(hm), np.asarray(hs), float(ym), float(ys)
        self.hist, self.secs, self.cached = hist, float(secs), cached

    def _h(self, H):
        return torch.from_numpy(((np.atleast_2d(H) - self.hm) / self.hs).astype(np.float32))

    def __call__(self, u, H, chunk=200000):
        u = np.asarray(u, dtype=np.float64).ravel(); H = np.atleast_2d(H)
        if len(H) == 1 and len(u) > 1:
            H = np.repeat(H, len(u), 0)
        out = np.empty(len(u))
        with torch.no_grad():
            for i in range(0, len(u), chunk):
                out[i:i + chunk] = self.net(torch.from_numpy(u[i:i + chunk].astype(np.float32)), self._h(H[i:i + chunk])).numpy()
        return self.ym + self.ys * out

    def grid(self, H, levels, chunk=100):
        """Quantile functions on a grid of levels, one row per summary (the embedding of H is computed once per row)."""
        H = np.atleast_2d(H); lv = torch.tensor(np.asarray(levels), dtype=torch.float32); out = np.empty((len(H), len(lv)))
        with torch.no_grad():
            e = self.net.tau_net(torch.cos(lv[:, None] * self.net.k))
            for i in range(0, len(H), chunk):
                h = self.net.h_net(self._h(H[i:i + chunk]))
                out[i:i + chunk] = self.net.head(e[None] * h[:, None]).squeeze(-1).numpy()
        return self.ym + self.ys * out


def fit(y, H, epochs, seed, batch=512, ntau=4, lr=2e-3):
    t0 = time.time(); torch.manual_seed(seed)
    hm = H.mean(0); hs = H.std(0); hs = np.where(hs > 0, hs, 1.0); ym, ys = float(y.mean()), float(y.std())
    th = torch.tensor((y - ym) / ys, dtype=torch.float32); Hh = torch.tensor((H - hm) / hs, dtype=torch.float32)
    net = QNN(Hh.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    m, hist = len(th), []
    for ep in range(epochs):
        perm = torch.randperm(m); tot = 0.0
        for i in range(0, m, batch):
            idx = perm[i:i + batch]
            t = th[idx].repeat(ntau); hb = Hh[idx].repeat(ntau, 1)
            tau = torch.rand(len(t))
            loss = pinball(t - net(tau, hb), tau)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step(); hist.append(tot / m * ys)                       # training check loss in the units of the target
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"    epoch {ep + 1}/{epochs}: training check loss {hist[-1]:.5f} ({time.time() - t0:.0f} s)", flush=True)
    return Fitted(net, hm, hs, ym, ys, hist, time.time() - t0)


def get_net(ctx, name, y, H, off):
    cfg = f"{name}|{G0}|{G1}|{SEED}|{ctx['N']}|{ctx['EP']}|v1"
    os.makedirs(ctx["cache"], exist_ok=True)
    path = os.path.join(ctx["cache"], f"{name}_{hashlib.md5(cfg.encode()).hexdigest()[:8]}.pt")
    if os.path.exists(path) and not ctx["retrain"]:
        d = torch.load(path); net = QNN(H.shape[1]); net.load_state_dict(d["state"])
        f = Fitted(net, d["hm"], d["hs"], d["ym"], d["ys"], d["hist"], d["secs"], cached=True)
        print(f"  [{name}] loaded from cache (it was trained in {f.secs:.0f} s)", flush=True)
    else:
        print(f"  [{name}] training on {len(y)} pairs for {ctx['EP']} epochs", flush=True)
        f = fit(y, H, ctx["EP"], seed=1000 + off)
        torch.save(dict(state=f.net.state_dict(), hm=f.hm.tolist(), hs=f.hs.tolist(), ym=f.ym, ys=f.ys, hist=f.hist, secs=f.secs), path)
        print(f"  [{name}] trained in {f.secs:.0f} s", flush=True)
    return f


# ---------------------------------------------------------------- checks
def check_loss(y, q, u):
    r = y - q
    return np.maximum(u * r, (u - 1) * r)


def rb_loss(q, m, s, u):
    """E rho_u(theta - q) for theta ~ N(m, s^2), in closed form."""
    a = (q - m) / s
    return (m - q) * (u - special.ndtr(a)) + s * stats.norm.pdf(a)


def rank_stats(draws, truth, jit, nb=10):
    """Rank of the truth among the draws, jittered inside its cell so that the rank is uniform on (0,1) under calibration."""
    m, L = draws.shape
    r = (draws < truth[:, None]).sum(1)
    v = (r + jit) / (L + 1)
    cnt = np.histogram(v, bins=np.linspace(0, 1, nb + 1))[0]
    chi = stats.chisquare(cnt)
    q = np.quantile(draws, [0.05, 0.25, 0.75, 0.95], axis=1, method="weibull")      # Weibull positions: nominal expected coverage
    return dict(n=m, counts=cnt, chisq=float(chi.statistic), p=float(chi.pvalue), ks_p=float(stats.kstest(v, "uniform").pvalue),
                cov50=float(np.mean((truth >= q[1]) & (truth <= q[2]))), cov90=float(np.mean((truth >= q[0]) & (truth <= q[3]))),
                width50=float(np.mean(q[2] - q[1])), width90=float(np.mean(q[3] - q[0])))


def cramer(Qs, mean, sd, levels=LEV, npts=20001):
    """Integral of (F_hat - F)^2 dx per row: F_hat the distribution function through the sorted quantile points, F = N(mean, sd^2)."""
    out = np.empty(len(Qs))
    for k in range(len(Qs)):
        lo = min(mean[k] - 8 * sd, Qs[k, 0]) - 0.05; hi = max(mean[k] + 8 * sd, Qs[k, -1]) + 0.05
        x = np.linspace(lo, hi, npts)
        out[k] = np.trapezoid((np.interp(x, Qs[k], levels, left=0.0, right=1.0) - special.ndtr((x - mean[k]) / sd)) ** 2, x)
    return out


def amortization(DES, c, sets, s_post, qfun, gridfun, chunk=2000):
    """Held out check loss against the exact posterior quantile function; twice the excess against the Cramer distance.

    raw: the network function f(u, S) as it is; rearranged: its sorted version, the quantile function of the draws f(U, S).
    The identity (excess = half the Cramer distance) holds for the rearranged function; the raw excess is larger by the crossing penalty."""
    Sh, Uh = sets["S_h"], sets["U_h"]
    psi_h = sets["theta_h"] @ c; mh = exact_post(DES, Sh, c)[0]
    q, qx = qfun(Uh, Sh), mh[:, None] + s_post * special.ndtri(Uh)
    Ln, Lx = check_loss(psi_h[:, None], q, Uh), check_loss(psi_h[:, None], qx, Uh)
    d = (Ln - Lx).mean(1)
    rb = (rb_loss(q, mh[:, None], s_post, Uh) - s_post * stats.norm.pdf(special.ndtri(Uh))).mean(1)
    g0 = s_post * stats.norm.pdf(special.ndtri(LEV))[None, :]
    ex_raw, ex_srt, cr, qr, nonmono, disp = [], [], [], np.empty_like(Uh), [], []
    for i in range(0, NHELD, chunk):
        sl = slice(i, i + chunk); Qraw = gridfun(Sh[sl]); Qs = np.sort(Qraw, axis=1); mk = mh[sl][:, None]
        ex_raw.append((rb_loss(Qraw, mk, s_post, LEV[None, :]) - g0).sum(1) / 1000)
        ex_srt.append((rb_loss(Qs, mk, s_post, LEV[None, :]) - g0).sum(1) / 1000)
        cr.append(cramer(Qs, mh[sl], s_post)); nonmono.append((np.diff(Qraw, axis=1) < -1e-9).any(1)); disp.append(np.abs(Qraw - Qs).mean(1))
        pos = np.clip(Uh[sl] * 1000 - 1, 0, 998); i0 = np.floor(pos).astype(int); i1 = np.minimum(i0 + 1, 998); fr = pos - i0
        rows = np.arange(len(Qs))[:, None]; qr[sl] = Qs[rows, i0] * (1 - fr) + Qs[rows, i1] * fr      # rearranged function at the held out levels
    ex_raw, ex_srt, cr = np.concatenate(ex_raw), np.concatenate(ex_srt), np.concatenate(cr)
    Lr = check_loss(psi_h[:, None], qr, Uh); dr = (Lr - Lx).mean(1)
    se = lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))
    cm = float(cr.mean())
    out = dict(loss_exact=float(Lx.mean()), loss_exact_formula=float(s_post / (2 * np.sqrt(np.pi))), cramer=cm, cramer_se=se(cr), n_summaries=NHELD,
                raw=dict(loss=float(Ln.mean()), excess=float(d.mean()), excess_se=se(d), excess_rb=float(rb.mean()), excess_rb_se=se(rb), excess_grid=float(ex_raw.mean())),
                rearranged=dict(loss=float(Lr.mean()), excess=float(dr.mean()), excess_se=se(dr), excess_grid=float(ex_srt.mean())),
                crossing=dict(share_nonmonotone=float(np.concatenate(nonmono).mean()), mean_abs_rearrangement_over_sd=float(np.concatenate(disp).mean() / s_post)),
                twice_excess_over_cramer=dict(raw_empirical=float(2 * d.mean() / cm), raw_rb=float(2 * rb.mean() / cm), raw_grid=float(2 * ex_raw.mean() / cm),
                                              rearranged_empirical=float(2 * dr.mean() / cm), rearranged_grid=float(2 * ex_srt.mean() / cm)))
    mn = out["loss_exact_formula"]
    for k in ("raw", "rearranged"):
        out[k].update(excess_over_se=out[k]["excess"] / out[k]["excess_se"], excess_grid_over_min_loss=out[k]["excess_grid"] / mn)
    out["crossing"]["share_of_raw_excess"] = 1 - out["rearranged"]["excess_grid"] / out["raw"]["excess_grid"]
    return out


# ---------------------------------------------------------------- stages
def stage_ate(ctx):
    DES, c, sets = ctx["DES"], ctx["c"], ctx["sets"]
    net = get_net(ctx, "ate", ctx["theta"] @ c, ctx["S"], 1)
    s_post = float(np.sqrt(c @ DES["V"] @ c)); R = dict(train_secs=net.secs, train_loss=net.hist, cached=net.cached, N=ctx["N"], epochs=ctx["EP"])
    # sufficiency check: the posterior computed from the full data equals the posterior computed from S
    P = DES["D"].T @ DES["D"] / SIGMA ** 2 + np.eye(4) / PRIOR_SD ** 2
    assert np.allclose(np.linalg.solve(P, DES["D"].T @ sets["Y_obs"][0] / SIGMA ** 2), DES["M"] @ sets["S_obs"][0]) and np.allclose(DES["A"] @ DES["D"], np.eye(4))
    # (1) the observed data set
    S_obs = sets["S_obs"]; m_obs = float(exact_post(DES, S_obs, c)[0][0]); pc = [0.05, 0.5, 0.95]
    draws = net(sets["U_obs"][0], S_obs); qn = net.grid(S_obs, LEV)[0]; qe = m_obs + s_post * special.ndtri(LEV)
    edges = np.linspace(m_obs - 3.5 * s_post, m_obs + 3.5 * s_post, 31)
    R["obs"] = dict(psi_true=float(sets["theta_obs"][0] @ c), S_obs=S_obs[0], exact_mean=m_obs, exact_sd=s_post, net_mean=float(draws.mean()), net_sd=float(draws.std()),
                    w1=float(np.mean(np.abs(np.sort(qn) - qe))), w1_raw=float(np.mean(np.abs(qn - qe))), exact_q=m_obs + s_post * special.ndtri(pc),
                    net_q=np.quantile(draws, pc), hist=dict(edges=edges, dens=np.histogram(draws, edges, density=True)[0]),
                    levels99=L99, exact99=m_obs + s_post * special.ndtri(L99), net99=net.grid(S_obs, L99)[0])
    R["obs"].update(w1_over_sd=R["obs"]["w1"] / s_post, w1_raw_over_sd=R["obs"]["w1_raw"] / s_post, net_sd_over_exact_sd=R["obs"]["net_sd"] / s_post)
    # (2) rank check, network and exact posterior, same levels and jitter
    Sr, psi_r, U, V = sets["S_r"], sets["theta_r"] @ c, sets["U_r"], sets["V_r"]; mr = exact_post(DES, Sr, c)[0]
    d_net = net(U.ravel(), np.repeat(Sr, LRANK, 0)).reshape(NRANK, LRANK); d_ex = mr[:, None] + s_post * special.ndtri(U)
    R["rank"] = dict(L=LRANK, n=NRANK, net=rank_stats(d_net, psi_r, V), exact=rank_stats(d_ex, psi_r, V),
                     exact_closed_form_pit_hist=np.histogram(special.ndtr((psi_r - mr) / s_post), bins=np.linspace(0, 1, 11))[0],
                     width90_exact_formula=float(2 * 1.6448536269514722 * s_post))
    # (5) local check: the 50 studies with the largest Mahalanobis distance of S from the training summaries
    Str = ctx["S"]; mu_S = Str.mean(0); Cov = np.cov(Str.T); Cinv = np.linalg.inv(Cov)
    mahal = lambda X: np.sqrt(np.einsum("ij,jk,ik->i", X - mu_S, Cinv, X - mu_S))
    d_r = mahal(Sr); far = np.zeros(NRANK, bool); far[np.argsort(-d_r)[:50]] = True
    w1_r = np.abs(np.sort(net.grid(Sr, LEV), axis=1) - (mr[:, None] + s_post * special.ndtri(LEV)[None])).mean(1) / s_post
    # extrapolation ray: from the training center in the direction along which the posterior mean of psi moves fastest
    g = DES["M"].T @ c; Lc = np.linalg.cholesky(Cov); e = Lc.T @ g; e /= np.linalg.norm(e); ray = []
    for rad in (0, 2, 4, 6, 8, 10):
        Sx = mu_S + rad * (Lc @ e); mx = float(g @ Sx); qx = np.sort(net.grid(Sx, LEV)[0])
        ray.append(dict(radius=rad, exact_mean=mx, net_median=float(qx[499]), w1_over_sd=float(np.mean(np.abs(qx - (mx + s_post * special.ndtri(LEV)))) / s_post)))
    R["local"] = dict(n_far=50, d_far_min=float(np.sort(d_r)[-50]), d_train_max=float(mahal(Str).max()), d_rank_max=float(d_r.max()),
                      far=rank_stats(d_net[far], psi_r[far], V[far]), rest=rank_stats(d_net[~far], psi_r[~far], V[~far]),
                      w1_over_sd_far=float(w1_r[far].mean()), w1_over_sd_rest=float(w1_r[~far].mean()), w1_over_sd_all=float(w1_r.mean()),
                      d=d_r, w1_over_sd=w1_r, is_far=far, ray=ray)
    R["local"]["w1_far_over_rest"] = R["local"]["w1_over_sd_far"] / R["local"]["w1_over_sd_rest"]
    # (4) amortization error
    qfun = lambda Uh, Sh: net(Uh.ravel(), np.repeat(Sh, KLEV, 0)).reshape(Uh.shape)
    R["amort"] = amortization(DES, c, sets, s_post, qfun, lambda S: net.grid(S, LEV))
    return R


def stage_blind(ctx):
    DES, c, sets = ctx["DES"], ctx["c"], ctx["sets"]; N = ctx["N"]
    net0 = get_net(ctx, "prior", ctx["theta"] @ c, np.zeros((N, 1)), 2)
    s_post = float(np.sqrt(c @ DES["V"] @ c)); s_prior = float(PRIOR_SD * np.linalg.norm(c)); Z0 = np.zeros((1, 1))
    q0 = net0.grid(Z0, LEV)[0]; R = dict(train_secs=net0.secs, train_loss=net0.hist, cached=net0.cached, N=N, epochs=ctx["EP"],
                                         prior_sd=s_prior, post_sd=s_post, net_sd_from_quantiles=float(np.std(q0)),
                                         w1_to_exact_prior=float(np.mean(np.abs(np.sort(q0) - s_prior * special.ndtri(LEV)))), quantiles99=net0.grid(Z0, L99)[0])
    Sr, psi_r, U, V = sets["S_r"], sets["theta_r"] @ c, sets["U_r"], sets["V_r"]
    d0 = net0(U.ravel(), np.zeros((NRANK * LRANK, 1))).reshape(NRANK, LRANK)
    R["rank"] = rank_stats(d0, psi_r, V)
    R["rank"]["width90_prior_formula"] = float(2 * 1.6448536269514722 * s_prior); R["rank"]["width90_exact_formula"] = float(2 * 1.6448536269514722 * s_post)
    R["rank"]["width90_over_exact_formula_width"] = R["rank"]["width90"] / R["rank"]["width90_exact_formula"]; R["prior_over_post_sd"] = s_prior / s_post
    qfun = lambda Uh, Sh: net0(Uh.ravel(), np.zeros((Uh.size, 1))).reshape(Uh.shape)
    R["amort"] = amortization(DES, c, sets, s_post, qfun, lambda S: np.tile(q0, (len(S), 1)))
    Uh = sets["U_h"]; psi_h = sets["theta_h"] @ c
    R["amort"]["loss_prior_exact_quantile_fn"] = float(check_loss(psi_h[:, None], s_prior * special.ndtri(Uh), Uh).mean())
    R["amort"]["loss_prior_formula"] = float(s_prior / (2 * np.sqrt(np.pi)))
    return R


def stage_cate(ctx):
    DES, sets, N = ctx["DES"], ctx["sets"], ctx["N"]
    rng = rng_for(31); z_tr = rng.normal(size=N); C_tr = np.column_stack([-np.ones(N), -z_tr, np.ones(N), z_tr])
    net = get_net(ctx, "cate", np.einsum("ij,ij->i", ctx["theta"], C_tr), np.column_stack([ctx["S"], z_tr]), 3)
    cz = lambda z: np.array([-1.0, -z, 1.0, z]); S_obs = sets["S_obs"]; U = sets["U_obs"][0]
    R = dict(train_secs=net.secs, train_loss=net.hist, cached=net.cached, N=N, epochs=ctx["EP"], at_z={})
    for z in ZS:
        m_e, s_e = exact_post(DES, S_obs, cz(z)); m_e, s_e = float(m_e[0]), float(s_e[0]); H = np.column_stack([S_obs, [z]])
        dr = net(U, H); qn = np.sort(net.grid(H, LEV)[0])
        R["at_z"][str(z)] = dict(exact_mean=m_e, exact_sd=s_e, net_mean=float(dr.mean()), net_sd=float(dr.std()),
                                 w1=float(np.mean(np.abs(qn - (m_e + s_e * special.ndtri(LEV))))), exact_q=m_e + s_e * special.ndtri([0.05, 0.5, 0.95]),
                                 net_q=np.quantile(dr, [0.05, 0.5, 0.95]))
        R["at_z"][str(z)].update(w1_over_sd=R["at_z"][str(z)]["w1"] / s_e, net_sd_over_exact_sd=R["at_z"][str(z)]["net_sd"] / s_e)
    ZG = np.linspace(-2.5, 2.5, 41); Hg = np.column_stack([np.repeat(S_obs, len(ZG), 0), ZG]); qg = net.grid(Hg, [0.05, 0.5, 0.95])
    ex = [exact_post(DES, S_obs, cz(z)) for z in ZG]
    R["curve"] = dict(z=ZG, net_q05=qg[:, 0], net_q50=qg[:, 1], net_q95=qg[:, 2], exact_mean=[float(a[0][0]) for a in ex], exact_sd=[float(a[1][0]) for a in ex])
    # rank check of tau(z) at a random z for each of the 500 studies
    z_r = sets["z_r"]; C_r = np.column_stack([-np.ones(NRANK), -z_r, np.ones(NRANK), z_r]); tau_r = np.einsum("ij,ij->i", sets["theta_r"], C_r)
    dn = net(sets["U_r"].ravel(), np.column_stack([np.repeat(sets["S_r"], LRANK, 0), np.repeat(z_r, LRANK)])).reshape(NRANK, LRANK)
    me, se = exact_post(DES, sets["S_r"], C_r); de = me[:, None] + se[:, None] * special.ndtri(sets["U_r"])
    R["rank"] = dict(net=rank_stats(dn, tau_r, sets["V_r"]), exact=rank_stats(de, tau_r, sets["V_r"]))
    return R


def stage_joint(ctx):
    DES, sets, S, th = ctx["DES"], ctx["sets"], ctx["S"], ctx["theta"]
    dmu, dal = th[:, 2] - th[:, 0], th[:, 3] - th[:, 1]
    nA = get_net(ctx, "dmu", dmu, S, 4); nB = get_net(ctx, "dal_given_dmu", dal, np.column_stack([S, dmu]), 5); nM = get_net(ctx, "dal_marginal", dal, S, 6)
    S_obs, U = sets["S_obs"], sets["U_obs"]; B = np.array([[-1.0, 0, 1, 0], [0, -1.0, 0, 1]])
    mu_ex = B @ DES["M"] @ S_obs[0]; cv = B @ DES["V"] @ B.T; e = effect_moments(DES["V"])
    dm = nA(U[0], S_obs); da_j = nB(U[1], np.column_stack([np.repeat(S_obs, NDRAW, 0), dm])); da_i = nM(U[2], S_obs)
    R = dict(train_secs=dict(dmu=nA.secs, dal_given_dmu=nB.secs, dal_marginal=nM.secs), cached=[nA.cached, nB.cached, nM.cached], N=ctx["N"], epochs=ctx["EP"],
             train_loss=dict(dmu=nA.hist, dal_given_dmu=nB.hist, dal_marginal=nM.hist),
             exact=dict(mean_dmu=float(mu_ex[0]), mean_dal=float(mu_ex[1]), sd_dmu=float(np.sqrt(cv[0, 0])), sd_dal=float(np.sqrt(cv[1, 1])), rho=e["rho"]),
             joint=dict(mean_dmu=float(dm.mean()), mean_dal=float(da_j.mean()), sd_dmu=float(dm.std()), sd_dal=float(da_j.std()), rho=float(np.corrcoef(dm, da_j)[0, 1])),
             indep=dict(mean_dal=float(da_i.mean()), sd_dal=float(da_i.std()), rho=float(np.corrcoef(dm, da_i)[0, 1])),
             w1_dmu=float(np.mean(np.abs(np.sort(nA.grid(S_obs, LEV)[0]) - (mu_ex[0] + np.sqrt(cv[0, 0]) * special.ndtri(LEV))))),
             w1_dal_marginal=float(np.mean(np.abs(np.sort(nM.grid(S_obs, LEV)[0]) - (mu_ex[1] + np.sqrt(cv[1, 1]) * special.ndtri(LEV))))), at_z={})
    # the conditional network: is the dependence on Delta mu learned? exact conditional of Delta alpha given Delta mu = x
    slope = cv[0, 1] / cv[0, 0]; csd = float(np.sqrt(cv[1, 1] - cv[0, 1] ** 2 / cv[0, 0])); cond = []
    for k in (-1.0, 0.0, 1.0):
        x = mu_ex[0] + k * np.sqrt(cv[0, 0]); qn = np.sort(nB.grid(np.column_stack([S_obs, [x]]), LEV)[0]); me = mu_ex[1] + slope * (x - mu_ex[0])
        cond.append(dict(k_sd=k, exact_mean=float(me), exact_sd=csd, net_median=float(qn[499]), net_sd=float(np.std(qn)), w1=float(np.mean(np.abs(qn - (me + csd * special.ndtri(LEV)))))))
    R["conditional"] = dict(exact_slope=float(slope), exact_cond_sd=csd,
                            net_slope=float((cond[2]["net_median"] - cond[0]["net_median"]) / (2 * np.sqrt(cv[0, 0]))), points=cond)
    for z in ZS:
        R["at_z"][str(z)] = dict(exact_sd=sd_tau(e, z), exact_sd_if_independent=sd_tau(e, z, False), joint_sd=float(np.std(dm + z * da_j)), indep_sd=float(np.std(dm + z * da_i)),
                                 exact_mean=float(mu_ex[0] + z * mu_ex[1]), joint_mean=float(np.mean(dm + z * da_j)), indep_mean=float(np.mean(dm + z * da_i)))
        d_ = R["at_z"][str(z)]; d_.update(joint_over_exact=d_["joint_sd"] / d_["exact_sd"], indep_over_exact=d_["indep_sd"] / d_["exact_sd"], exact_if_independent_over_exact=d_["exact_sd_if_independent"] / d_["exact_sd"])
    ZG = np.linspace(-2.5, 2.5, 41)
    R["curve"] = dict(z=ZG, exact_sd=[sd_tau(e, z) for z in ZG], exact_sd_if_independent=[sd_tau(e, z, False) for z in ZG],
                      joint_sd=np.std(dm[:, None] + ZG[None] * da_j[:, None], axis=0), indep_sd=np.std(dm[:, None] + ZG[None] * da_i[:, None], axis=0))
    return R


def stage_scale(ctx):
    """Amortization error of the ATE network against the number of simulations N and the training seed (not part of `all`)."""
    DES, c, sets = ctx["DES"], ctx["c"], ctx["sets"]; s_post = float(np.sqrt(c @ DES["V"] @ c)); zq = special.ndtri(LEV)
    S_obs = sets["S_obs"]; qe_obs = float(exact_post(DES, S_obs, c)[0][0]) + s_post * zq
    Sr = sets["S_r"]; qe_r = exact_post(DES, Sr, c)[0][:, None] + s_post * zq[None]
    Sh, Uh, psi_h = sets["S_h"], sets["U_h"], sets["theta_h"] @ c
    Sk = Sh[:2000]; mk = exact_post(DES, Sk, c)[0][:, None]; g0 = s_post * stats.norm.pdf(zq)[None]
    rows = []
    for N in (10000, 40000, 100000, 400000):
        th, S = prior_pairs(DES, N, 10); c2 = dict(ctx, N=N)
        for sd in range(3):
            net = get_net(c2, f"scale_s{sd}", th @ c, S, 100 + sd)
            Qs = np.sort(net.grid(Sk, LEV), axis=1); q = net(Uh.ravel(), np.repeat(Sh, KLEV, 0)).reshape(Uh.shape)
            rows.append(dict(N=N, seed=sd, train_secs=net.secs, train_loss=net.hist[-1],
                             w1_obs=float(np.mean(np.abs(np.sort(net.grid(S_obs, LEV)[0]) - qe_obs))),
                             w1_over_sd_500=float(np.mean(np.abs(np.sort(net.grid(Sr, LEV), axis=1) - qe_r)) / s_post),
                             twice_excess_rearranged=float(2 * ((rb_loss(Qs, mk, s_post, LEV[None]) - g0).sum(1) / 1000).mean()),
                             heldout_loss=float(check_loss(psi_h[:, None], q, Uh).mean())))
    by_N = {}
    for N in sorted({r["N"] for r in rows}):
        rr = [r for r in rows if r["N"] == N]
        by_N[str(N)] = {k: dict(mean=float(np.mean([r[k] for r in rr])), sd=float(np.std([r[k] for r in rr], ddof=1))) for k in ("train_secs", "w1_obs", "w1_over_sd_500", "twice_excess_rearranged", "heldout_loss")}
    sp = stats.spearmanr([r["heldout_loss"] for r in rows], [r["w1_over_sd_500"] for r in rows])
    Ns = np.array(sorted({r["N"] for r in rows}), float)
    slope = lambda k: float(np.polyfit(np.log(Ns), np.log([by_N[str(int(N))][k]["mean"] for N in Ns]), 1)[0])
    return dict(rows=rows, by_N=by_N, spearman_heldout_loss_vs_w1=float(sp.statistic), spearman_p=float(sp.pvalue), loss_exact_formula=float(s_post / (2 * np.sqrt(np.pi))),
                loglog_slope_w1_over_sd_500=slope("w1_over_sd_500"), loglog_slope_twice_excess=slope("twice_excess_rearranged"))


SCALE2_N = (10000, 40000, 100000, 400000)
SCALE2_SEEDS = 10
NEVAL = 2000


def stage_scale2(ctx):
    """Review item F16 (October 2026 rerun). ATE and CATE networks at N = 10,000, 40,000, 100,000 and 400,000, one training seed per call
    (--seed k, k = 0..9; ten seeds in the full run), on nested training sets (the first N rows of one 400,000 row set of prior pairs, and of one
    400,000 draw of the covariate z used by the CATE network), the fixed design, the architecture and 40 epochs of the other stages.
    The seed changes the initial weights and the order of the minibatches, not the data. Evaluation on the first 2,000 of the held out studies
    (sets['S_h']) and, for the CATE network, at z = -1.5, 0, 1.5: Wasserstein-1 distance between the sorted network quantiles on the 999 levels
    i/1000 (so the endpoints are the levels 0.001 and 0.999) and the exact posterior quantiles, divided by the exact posterior sd; its median
    and 90th percentile over the studies; coverage of the central 90 percent interval (levels 0.05 and 0.95) for the true effect of each study; and the
    same distance at the observed data set. Writes results/gbc_ate_scale2/seed{k}.json."""
    DES, c, sets = ctx["DES"], ctx["c"], ctx["sets"]; sd = ctx["seed2"]; zq = special.ndtri(LEV)
    Nmax = max(SCALE2_N); th_all, S_all = prior_pairs(DES, Nmax, 10); z_all = rng_for(31).normal(size=Nmax)
    Sh, thh = sets["S_h"][:NEVAL], sets["theta_h"][:NEVAL]; S_obs = sets["S_obs"]
    cz = lambda z: np.array([-1.0, -z, 1.0, z])
    targets = {"ate": (c, None)}
    for z in ZS:
        targets[f"cate_z{z}"] = (cz(z), z)
    exact = {}
    for k, (cv, z) in targets.items():
        m, sdp = exact_post(DES, Sh, cv); mo, so = exact_post(DES, S_obs, cv)
        exact[k] = dict(m=m, sd=float(sdp[0]), qe=m[:, None] + sdp[0] * zq[None], truth=thh @ cv, mo=float(mo[0]), so=float(so[0]))
    rows = []
    for N in SCALE2_N:
        th, S, zt = th_all[:N], S_all[:N], z_all[:N]
        for kind in ("ate", "cate"):
            if kind == "ate":
                net = fit(th @ c, S, ctx["EP"], seed=5000 + sd)
            else:
                C_tr = np.column_stack([-np.ones(N), -zt, np.ones(N), zt])
                net = fit(np.einsum("ij,ij->i", th, C_tr), np.column_stack([S, zt]), ctx["EP"], seed=6000 + sd)
            for k, (cv, z) in targets.items():
                if (kind == "ate") != (z is None):
                    continue
                H = Sh if z is None else np.column_stack([Sh, np.full(len(Sh), z)]); Ho = S_obs if z is None else np.column_stack([S_obs, [z]])
                Q = np.sort(net.grid(H, LEV), axis=1); e = exact[k]
                w1 = np.abs(Q - e["qe"]).mean(1) / e["sd"]
                qo = np.sort(net.grid(Ho, LEV)[0]); w1o = float(np.abs(qo - (e["mo"] + e["so"] * zq)).mean() / e["so"])
                cov = float(((e["truth"] >= Q[:, 49]) & (e["truth"] <= Q[:, 949])).mean())
                rows.append(dict(N=N, seed=sd, target=k, median_w1_over_sd=float(np.median(w1)), p90_w1_over_sd=float(np.quantile(w1, 0.9)),
                                 mean_w1_over_sd=float(w1.mean()), coverage90=cov, w1_obs_over_sd=w1o, train_secs=net.secs, train_loss=net.hist[-1]))
            print(f"  scale2 seed {sd} N {N} {kind} done", flush=True)
    os.makedirs(os.path.join(ctx["root"], "results", "gbc_ate_scale2"), exist_ok=True)
    json.dump(jsonable(dict(rows=rows, neval=NEVAL, levels="i/1000, i = 1..999", script_sha256_12=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12])),
              open(os.path.join(ctx["root"], "results", "gbc_ate_scale2", f"seed{sd}.json"), "w"), indent=1)
    return None


def stage_scale2agg(ctx):
    """Collects results/gbc_ate_scale2/seed*.json: mean and sd over seeds of each statistic, by N and target; figs/gbc_ate_scale2.pdf."""
    import glob
    rows = []
    for f in sorted(glob.glob(os.path.join(ctx["root"], "results", "gbc_ate_scale2", "seed*.json"))):
        rows += json.load(open(f))["rows"]
    tg = sorted({r["target"] for r in rows}); by = {}
    for t in tg:
        by[t] = {}
        for N in SCALE2_N:
            rr = [r for r in rows if r["target"] == t and r["N"] == N]
            by[t][str(N)] = dict(n_seeds=len(rr), **{k: dict(mean=float(np.mean([r[k] for r in rr])), sd=float(np.std([r[k] for r in rr], ddof=1)),
                                                              min=float(np.min([r[k] for r in rr])), max=float(np.max([r[k] for r in rr])))
                                                     for k in ("median_w1_over_sd", "p90_w1_over_sd", "coverage90", "w1_obs_over_sd", "train_secs")})
    fig, axs = plt.subplots(1, 4, figsize=(6.4, 2.3), sharey=True)
    for ax, t, ttl in zip(axs, ["ate"] + [f"cate_z{z}" for z in ZS], ["ATE", "CATE, z = -1.5", "CATE, z = 0", "CATE, z = 1.5"]):
        for k, ls, lab in (("median_w1_over_sd", "-", "median over studies"), ("p90_w1_over_sd", "--", "90th percentile")):
            m = np.array([by[t][str(N)][k]["mean"] for N in SCALE2_N]); lo = np.array([by[t][str(N)][k]["min"] for N in SCALE2_N]); hi = np.array([by[t][str(N)][k]["max"] for N in SCALE2_N])
            ax.plot(SCALE2_N, m, ls, color="k", lw=1, marker="o", ms=3, label=lab); ax.fill_between(SCALE2_N, lo, hi, color="0.75", alpha=0.6, lw=0)
        ax.plot(SCALE2_N, [by[t][str(N)]["w1_obs_over_sd"]["mean"] for N in SCALE2_N], ":", color="k", lw=1, marker="s", ms=3, label="observed data set")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(ttl, fontsize=8); ax.set_xlabel("simulations $N$")
    axs[0].set_ylabel("Wasserstein-1 / posterior sd"); axs[0].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(ctx["root"], "figs", "gbc_ate_scale2.pdf")); plt.close(fig)
    return dict(by_N=by, seeds=sorted({r["seed"] for r in rows}), neval=NEVAL, N=list(SCALE2_N))


def stage_fig(ctx):
    J = load_json(ctx); fig, axs = plt.subplots(3, 2, figsize=(5.5, 7.6)); ok = lambda *k: all(x in J for x in k)
    grey = "0.25"
    if ok("ate"):                                                                  # (a) posterior of the ATE at the observed data
        a = J["ate"]["obs"]; ax = axs[0, 0]; ed, de = np.array(a["hist"]["edges"]), np.array(a["hist"]["dens"])
        xs = np.linspace(ed[0], ed[-1], 300)
        ax.stairs(de, ed, fill=True, color="0.8"); ax.stairs(de, ed, color="0.45", lw=0.6)
        ax.plot(xs, stats.norm.pdf(xs, a["exact_mean"], a["exact_sd"]), color=grey, lw=1.3)
        ax.axvline(a["psi_true"], color=PAL[1], lw=0.9, ls=(0, (4, 2))); ax.set_yticks([])
        ax.set_xlabel(r"$\psi$"); ax.set_title("(a) Posterior of the average effect", fontsize=8); ax.set_ylim(0, 1.5 * max(de.max(), stats.norm.pdf(0, 0, a["exact_sd"])))
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="0.8", ec="0.45", lw=0.6), plt.Line2D([], [], color=grey, lw=1.3), plt.Line2D([], [], color=PAL[1], lw=0.9, ls=(0, (4, 2)))],
                  labels=["network draws", "exact", "true value"], frameon=False, fontsize=6.5, loc="upper right")
    if ok("ate", "blind"):                                                         # (b) rank histograms
        ax = axs[0, 1]; w = 0.28; x = np.arange(10)
        for k, (lab, cnt, fc, ec, ht) in enumerate([("ATE network", J["ate"]["rank"]["net"]["counts"], PAL[3], PAL[3], None),
                                                    ("exact", J["ate"]["rank"]["exact"]["counts"], "white", grey, "//////"),
                                                    ("prior only", J["blind"]["rank"]["counts"], "0.82", "0.35", "xxxx")]):
            ax.bar(x + (k - 1) * w, cnt, w, fc=fc, ec=ec, hatch=ht, lw=0.5, label=lab, alpha=0.6 if ht is None else 1.0)
        ax.axhline(NRANK / 10, color=PAL[1], lw=0.9); ax.set_ylim(0, 1.75 * NRANK / 10); ax.set_xticks(x[::3]); ax.set_xticklabels([f"{0.1 * (i + 0.5):.2f}" for i in x[::3]])
        ax.set_xlabel("rank of the true value (10 bins)"); ax.set_ylabel("count"); ax.set_title("(b) Rank check, 500 studies", fontsize=8)
        ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", columnspacing=0.8, handlelength=1.2)
    if ok("ate", "blind"):                                                         # (c) quantile functions at the observed data
        ax = axs[1, 0]; a = J["ate"]["obs"]
        ax.plot(a["levels99"], a["exact99"], color="0.6", lw=3.0, label="exact posterior")
        ax.plot(a["levels99"], a["net99"], color=PAL[0], lw=1.0, ls="--", label="ATE network")
        ax.plot(a["levels99"], J["blind"]["quantiles99"], color=PAL[1], lw=1.2, ls=":", label="prior only network")
        ax.set_xlabel("level"); ax.set_ylabel(r"$Q(u)$ of $\psi$"); ax.set_title("(c) Quantile functions, observed data", fontsize=8); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    if ok("ate"):                                                                  # (d) error against distance from the training summaries
        ax = axs[1, 1]; L = J["ate"]["local"]; d, w, fr = np.array(L["d"]), np.array(L["w1_over_sd"]), np.array(L["is_far"], bool)
        ax.scatter(d[~fr], w[~fr], s=5, fc="none", ec="0.45", lw=0.5, label="450 other studies")
        ax.scatter(d[fr], w[fr], s=7, marker="s", fc=PAL[1], ec="none", label="50 farthest")
        ry = L["ray"]; ax.plot([r["radius"] for r in ry], [r["w1_over_sd"] for r in ry], color=grey, lw=0.9, marker="D", ms=2.8, label="along a ray")
        ax.axvline(L["d_train_max"], color="0.5", lw=0.7, ls=":"); ax.set_yscale("log"); ax.set_xlabel("Mahalanobis distance of $S$")
        ax.set_ylabel("Wasserstein-1 / posterior sd"); ax.set_title("(d) Error away from the training data", fontsize=8); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    if ok("cate"):                                                                 # (e) CATE band at the observed data
        ax = axs[2, 0]; c = J["cate"]["curve"]; z = np.array(c["z"]); m, s = np.array(c["exact_mean"]), np.array(c["exact_sd"])
        ax.fill_between(z, m - 1.6449 * s, m + 1.6449 * s, color="0.8", lw=0, label="exact 90% band")
        ax.plot(z, m, color=grey, lw=1.0, label="exact mean")
        ax.plot(z, c["net_q50"], color=PAL[0], lw=1.0, ls="--", label="network median")
        ax.plot(z, c["net_q05"], color=PAL[0], lw=0.8, ls=":", label="network 5%, 95%"); ax.plot(z, c["net_q95"], color=PAL[0], lw=0.8, ls=":")
        ax.set_xlabel("$z$"); ax.set_ylabel(r"$\tau(z)$"); ax.set_title("(e) Conditional effect, observed data", fontsize=8); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    if ok("joint"):                                                                # (f) joint against independent draws
        ax = axs[2, 1]; c = J["joint"]["curve"]; z = np.array(c["z"])
        ax.plot(z, c["exact_sd"], color="0.6", lw=3.0, label="exact")
        ax.plot(z, c["exact_sd_if_independent"], color="0.2", lw=0.7, ls="-.", label="exact marginals, independent")
        ax.plot(z, c["joint_sd"], color=PAL[0], lw=1.1, ls="--", label="joint draws (triangular pair)")
        ax.plot(z, c["indep_sd"], color=PAL[1], lw=1.2, ls=":", label="independent draws")
        ax.set_xlabel("$z$"); ax.set_ylabel(r"posterior sd of $\tau(z)$"); ax.set_title("(f) Joint against independent draws", fontsize=8); lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + 0.3 * (hi_ - lo_)); ax.legend(frameon=False, fontsize=6.5, loc="upper center")
    fig.tight_layout(); path = os.path.join(ctx["root"], "figs", "gbc_ate.pdf"); os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path); plt.close()
    return dict(file="figs/gbc_ate.pdf", width_in=5.5, height_in=7.6, built_from=[k for k in ("ate", "blind", "cate", "joint") if k in J])


# ---------------------------------------------------------------- bookkeeping
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def brief(o):
    if isinstance(o, dict):
        return {k: brief(v) for k, v in o.items()}
    if isinstance(o, list):
        return [brief(v) for v in o] if len(o) <= 10 else f"<list of {len(o)}>"
    return round(o, 5) if isinstance(o, float) else o


def json_path(ctx):
    return os.path.join(ctx["root"], "results", "res_gbc_ate.json")


def load_json(ctx):
    return json.load(open(json_path(ctx))) if os.path.exists(json_path(ctx)) else {}


def save_stage(ctx, name, res):
    J = load_json(ctx); os.makedirs(os.path.dirname(json_path(ctx)), exist_ok=True)
    J["design"] = jsonable(design_section(ctx["DES"]))
    J["versions"] = dict(python=sys.version.split()[0], numpy=np.__version__, scipy=__import__("scipy").__version__, matplotlib=matplotlib.__version__,
                         torch=torch.__version__, torch_threads=torch.get_num_threads())
    J[name] = jsonable(res)
    json.dump(J, open(json_path(ctx), "w"), indent=1)


STAGES = dict(ate=stage_ate, blind=stage_blind, cate=stage_cate, joint=stage_joint, fig=stage_fig, scale=stage_scale, scale2=stage_scale2, scale2agg=stage_scale2agg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES) + ["all"])
    ap.add_argument("--N", type=int, default=100000); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--root", default=".", help="directory holding results/ and figs/ (the repository root)")
    ap.add_argument("--cache", default="/tmp/gbc_ate_cache"); ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="training seed of stage scale2 (0..9)")
    ap.add_argument("--quick", action="store_true", help="N = 3000 and 3 epochs, for testing the code path only")
    args = ap.parse_args()
    DES = make_design(G0, G1); N, EP = (3000, 3) if args.quick else (args.N, args.epochs)
    ctx = dict(DES=DES, N=N, EP=EP, seed2=args.seed, cache=args.cache, retrain=args.retrain, root=args.root, c=np.array([-1.0, -DES["zbar"], 1.0, DES["zbar"]]))
    todo = ["ate", "blind", "cate", "joint", "fig"] if args.stage == "all" else [args.stage]
    if todo not in (["fig"], ["scale2agg"]):
        ctx["theta"], ctx["S"] = prior_pairs(DES, N, 10); ctx["sets"] = make_sets(DES)
    for st in todo:
        t0 = time.time(); print(f"stage {st} (N = {N}, epochs = {EP})", flush=True)
        res = STAGES[st](ctx)
        if res is not None:
            res["stage_secs"] = time.time() - t0; res["design_g0_g1"] = [G0, G1]
            res["script_sha256_12"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]
            save_stage(ctx, st, res); print(json.dumps(brief(jsonable(res)), indent=1))
        print(f"stage {st} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
