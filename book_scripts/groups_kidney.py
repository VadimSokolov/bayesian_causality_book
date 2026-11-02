# Groups, urns and information: the kidney stone analysis of Chapter 4, "Groups, Urns and Information" (label ch:groups). Six figures and results/res_groups_kidney.json.
# Data: the kidney stone counts of Charig et al. (1986), 700 patients, open surgery (A, treated, t=1) against percutaneous nephrolithotomy (PCNL, B, control), successes / patients by stone size. Small stones: A 81/87, B 234/270. Large stones: A 192/263, B 55/80. The counts are typed in below (the book's Table fw:tab:simpson holds the same ones).
# Models: a stable effect psi on three scales, p(x, t=0) = g^{-1}(gamma_x) and p(x, t=1) = g^{-1}(gamma_x + psi) for the two stone sizes x, with g the identity (risk difference, RD), the log (log risk ratio) or the logit (log odds ratio), fitted by maximum likelihood with Nelder-Mead (xatol 1e-10, fatol 1e-12). The heterogeneous model has the four stratum probabilities free. The information gap is D = deviance / (2 N) against the heterogeneous fit.
# Aranda Ordaz scans: the link h_lambda(u) = log{((1-u)^(-lambda) - 1) / lambda} applied to the success probability or to the failure probability, lambda = 0 being the complementary log log link and lambda = 1 the logit, on 0 and the grid 10^(k/20), k = -40..50 (0.01 to 316.2, contains 1.0 exactly). The success side stops at lambda = 19.95 because the gap is flat from there (checked at lambda = 100) and the chart overflows double precision near lambda = 265.
# Priors: flat on (gamma_1, gamma_2, psi) on each chart, restricted to the region where the four probabilities lie in (0, 1); Beta(1, 1) on each stratum probability for the saturated analysis.
# Class weights: the weights of the four classes (RD, log RR, log OR, heterogeneous) are exp(-n D_G - d_G log(n)/2) with equal prior model probabilities, normalised over the four, d_G = 3, 3, 3 and 4. They are leading order posterior weights, not exact posteriors. The crossing points are solved exactly (closed form for 19 to 1, brentq for the heterogeneous class), with no grid.
# Posterior of psi: given psi the two strata are independent, so the marginal posterior is a product of two one dimensional integrals over the baselines. It is computed by quadrature (20,001 baselines times 801 to 901 values of psi, no random numbers). These exact values of P(psi <= 0) are the ones the book prints.
# Seeded chain: random walk Metropolis on (gamma_1, gamma_2, psi), steps 0.02, 0.03 and 0.25 for RD, log RR and log OR, 260,000 steps with 20,000 burn in per chart (240,000 kept draws), default_rng(7), one stream for the three chains and then for the 400,000 Beta draws of the saturated posteriors (2 x 400,000 per arm).
# The chain draws only the histogram of the effect figure (every fifth draw, 48,000 per chart, is kept in results/res_groups_kidney_draws.npz). Both probabilities, the exact one and the chain's, go into the JSON together with the chain's Monte Carlo standard error (from the effective sample size of the indicator).
# Writes: results/res_groups_kidney.json, results/res_groups_kidney_draws.npz and figs/kidney_forest.pdf, kidney_inverse_sanov.pdf, kidney_closed_groups.pdf, kidney_effect_ld.pdf, kidney_information_pythagoras.pdf, kidney_ate_urn.pdf (the information quantities, JSON key info, have no figure of their own since the chapter dropped the information figure). Every figure is drawn 6.4 in wide or less with no text below 8.5 pt, and in gray styles.
# Run from the repository root: python book_scripts/groups_kidney.py (about 20 seconds). With --figs-only the chain is not run, the saved draws are read and only the figures are written. The Sanov, urn and case mix figures are in groups_urn_tau.py.
import json, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import minimize, brentq
from scipy.special import expit, logit, xlogy, logsumexp
from scipy.stats import norm

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
OUT, RES = "figs/", "results/"
FIGS_ONLY = "--figs-only" in sys.argv
trap = getattr(np, "trapezoid", None) or np.trapz
T0 = time.time()
R = {}
# Gray scale styles, the same for a scale in every figure: log risk ratio solid black, risk difference dashed 0.3, log odds ratio dash-dot 0.55, heterogeneous dotted.
ST = {"logRR": dict(color="k", ls="-", lw=1.8), "RD": dict(color="0.3", ls="--", lw=1.8),
      "logOR": dict(color="0.55", ls="-.", lw=2.0), "sat": dict(color="0.15", ls=":", lw=2.2)}
LAB = {"RD": "risk difference", "logRR": "log risk ratio", "logOR": "log odds ratio", "sat": "heterogeneous"}


def plain(x):
    if isinstance(x, dict): return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    if isinstance(x, np.ndarray): return plain(x.tolist())
    if isinstance(x, np.bool_): return bool(x)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    return x


def save(fig, name, rect=None):
    if rect: fig.tight_layout(rect=rect)
    else: fig.tight_layout()
    fig.savefig(OUT + name, metadata={"CreationDate": None}); plt.close(fig)


# ------------------------------------------------------------------ 1. data, links, maximum likelihood fits
yA = np.array([81, 192]); nA = np.array([87, 263]); yB = np.array([234, 55]); nB = np.array([270, 80])   # (small, large); A open surgery, B PCNL
N = int(nA.sum() + nB.sum())
SCALES = ["RD", "logRR", "logOR"]
links = {"RD": (lambda p: p, lambda e: e), "logRR": (np.log, np.exp), "logOR": (logit, expit)}


def nll(theta, link, y1, n1, y0, n0):
    f, finv = links[link]; g = theta[:2]; psi = theta[2]
    p0 = finv(g); p1 = finv(g + psi)
    if np.any(p0 <= 0) or np.any(p0 >= 1) or np.any(p1 <= 0) or np.any(p1 >= 1): return 1e10
    return -np.sum(xlogy(y1, p1) + xlogy(n1 - y1, 1 - p1) + xlogy(y0, p0) + xlogy(n0 - y0, 1 - p0))


def sat_ll(y1, n1, y0, n0):
    p1 = y1 / n1; p0 = y0 / n0
    return np.sum(xlogy(y1, p1) + xlogy(n1 - y1, 1 - p1) + xlogy(y0, p0) + xlogy(n0 - y0, 1 - p0))


def fit(link, y1, n1, y0, n0):
    f, _ = links[link]; p0 = np.clip(y0 / n0, .02, .98); p1 = np.clip(y1 / n1, .02, .98)
    x0 = np.r_[f(p0), np.mean(f(p1) - f(p0))]
    r = minimize(nll, x0, args=(link, y1, n1, y0, n0), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    return r.x, r.fun


LS = sat_ll(yA, nA, yB, nB)
FIT = {L: fit(L, yA, nA, yB, nB) for L in SCALES}

# ------------------------------------------------------------------ 2. strata, collapsed contrast, weights, information gaps
p1 = yA / nA; p0 = yB / nB
R["data"] = dict(yA=yA, nA=nA, yB=yB, nB=nB, N=N, strata=["small stones", "large stones"], A="open surgery", B="PCNL")
R["strata"] = dict(p1=p1, p0=p0, RD=p1 - p0, RR=p1 / p0, OR=(p1 / (1 - p1)) / (p0 / (1 - p0)))
P1 = yA.sum() / nA.sum(); P0 = yB.sum() / nB.sum()
R["marginal"] = dict(P1=P1, P0=P0, RD=P1 - P0, RR=P1 / P0, OR=(P1 / (1 - P1)) / (P0 / (1 - P0)))
wt = (nA[0] + nB[0]) / N                                  # share of small stones in the trial, 357/700
R["weights"] = dict(A=nA[0] / nA.sum(), B=nB[0] / nB.sum(), trial=wt, trial_large=1 - wt)
gap = {}
for L in SCALES:
    th, f = FIT[L]; dev = 2 * (f + LS)
    gap[L] = dict(psi=th[2], dev=dev, D=dev / (2 * N))
gap["ratio_OR_RR"] = gap["logOR"]["D"] / gap["logRR"]["D"]
gap["ratio_RD_RR"] = gap["RD"]["D"] / gap["logRR"]["D"]
gap["ratio_OR_RD"] = gap["logOR"]["D"] / gap["RD"]["D"]
R["gap"] = gap
print("gaps", {L: (round(gap[L]["psi"], 6), round(gap[L]["dev"], 6), "%.5e" % gap[L]["D"]) for L in SCALES}, "ratio OR/RR %.2f" % gap["ratio_OR_RR"])


# ------------------------------------------------------------------ 3. forest plot: stratum effects with Wald intervals, stable estimate, collapsed contrast
def eff(L, y1, n1, y0, n0):
    q1 = y1 / n1; q0 = y0 / n0
    if L == "RD": return q1 - q0, np.sqrt(q1 * (1 - q1) / n1 + q0 * (1 - q0) / n0)
    if L == "logRR": return np.log(q1 / q0), np.sqrt((1 - q1) / y1 + (1 - q0) / y0)
    return logit(q1) - logit(q0), np.sqrt(1 / y1 + 1 / (n1 - y1) + 1 / y0 + 1 / (n0 - y0))


R["forest"] = {}
for L in SCALES:
    e, se = eff(L, yA, nA, yB, nB)
    em, sem = eff(L, np.array([yA.sum()]), np.array([nA.sum()]), np.array([yB.sum()]), np.array([nB.sum()]))
    R["forest"][L] = dict(est=e, se=se, lo=e - 1.96 * se, hi=e + 1.96 * se, stable=FIT[L][0][2], marginal=em[0], marginal_se=sem[0],
                          marginal_lo=em[0] - 1.96 * sem[0], marginal_hi=em[0] + 1.96 * sem[0])

# ------------------------------------------------------------------ 4. leading order posterior weights of the stability classes as the trial is scaled up, exact crossing points
dims = {"RD": 3, "logRR": 3, "logOR": 3, "sat": 4}; keys = ["RD", "logRR", "logOR", "sat"]
Dg = {k: (gap[k]["D"] if k != "sat" else 0.0) for k in keys}


def post(n):
    lp = np.array([-(n * Dg[k]) - dims[k] / 2 * np.log(n) for k in keys])
    return np.exp(lp - logsumexp(lp))


ns = np.logspace(np.log10(300), 7, 200); PM = np.array([post(n) for n in ns]).T
MP = {"ns": ns, "P": {k: PM[i] for i, k in enumerate(keys)}}
n19 = np.log(19) / (Dg["logOR"] - Dg["logRR"])                                    # posterior odds RR : OR = exp{n (D_OR - D_RR)}, the dimensions are equal
n_sat = brentq(lambda n: n * Dg["logRR"] - 0.5 * np.log(n), 1e3, 1e9)           # equal evidence, heterogeneous (4 parameters) against risk ratio (3)
n_half = brentq(lambda n: post(n)[3] - 0.5, 1e3, 1e9)
R["model_post"] = dict(note="leading order posterior weights exp(-n D_G - d_G log(n)/2) with equal prior model probabilities, normalised over the four classes (d_G = 3, 3, 3 and 4), not exact posteriors", n700=dict(zip(keys, post(700.0))), n_RR_OR_19to1=n19, n_RR_OR_19to1_over_trial=n19 / N, n_sat_overtakes_RR=n_sat, n_sat_post_half=n_half, dims=dims)
print("model posterior at n=700", R["model_post"]["n700"], "n(19:1) %.1f" % n19, "n(sat) %.0f" % n_sat, "n(P sat=1/2) %.0f" % n_half)

# ------------------------------------------------------------------ 5. Aranda Ordaz scans, closed groups on (0, 1)
LAM = 10.0 ** (np.arange(-40, 51) / 20.0)                                          # 0.01 to 316.2, 1.0 is a grid point


def ao_u(h, lam):                                                                  # probability the family acts on, as a function of the chart value
    with np.errstate(all="ignore"):
        return 1 - np.exp(-np.exp(h)) if lam < 1e-6 else 1 - (1 + lam * np.exp(h)) ** (-1.0 / lam)


def ao_h(u, lam):
    with np.errstate(all="ignore"):
        return np.log(-np.log(1 - u)) if lam < 1e-6 else np.log(((1 - u) ** (-lam) - 1) / lam)


def ao_p(h, lam, side):
    u = ao_u(h, lam)
    return u if side == "success" else 1 - u


def ao_nll(th, lam, side):
    q0 = ao_p(th[:2], lam, side); q1 = ao_p(th[:2] + th[2], lam, side)
    if not (np.all(np.isfinite(q0)) and np.all(np.isfinite(q1))): return 1e10
    if np.any(q0 <= 0) or np.any(q0 >= 1) or np.any(q1 <= 0) or np.any(q1 >= 1): return 1e10
    return -np.sum(xlogy(yA, q1) + xlogy(nA - yA, 1 - q1) + xlogy(yB, q0) + xlogy(nB - yB, 1 - q0))


def ao_scan(lams, side):
    out = []
    for lam in lams:
        c0 = ao_h(p0 if side == "success" else 1 - p0, lam); c1 = ao_h(p1 if side == "success" else 1 - p1, lam)
        r = minimize(ao_nll, np.r_[c0, np.mean(c1 - c0)], args=(lam, side), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 20000})
        out.append((lam, (r.fun + LS) / N, r.x[2]))
    return np.array(out)


AO = {"success": ao_scan(np.r_[0.0, LAM[LAM <= 20]], "success"), "failure": ao_scan(np.r_[0.0, LAM], "failure")}
lam100 = ao_scan([100.0], "success")[0]


def ao_summary(a, side):
    lam, g, ps = a[:, 0], a[:, 1], a[:, 2]; d = np.diff(g); i1 = int(np.where(lam == 1.0)[0][0])
    out = dict(gap_lambda0=g[0], lambda_last=lam[-1], gap_last=g[-1], gap_at_logit=g[i1], n_points=len(lam), steps_up=int((d > 0).sum()), steps_down=int((d < 0).sum()))
    if side == "success": out.update(monotone_increasing=bool(np.all(d > 0)), gap_lambda100=lam100[1], psi_lambda0=ps[0])
    else: out.update(monotone_decreasing=bool(np.all(d < 0)), minus_psi_over_lambda_last=-ps[-1] / lam[-1], psi_last=ps[-1], psi_lambda0=ps[0])
    return out


R["ao"] = dict(success=ao_summary(AO["success"], "success"), failure=ao_summary(AO["failure"], "failure"), logRR_psi=gap["logRR"]["psi"], logit_gap_reference=gap["logOR"]["D"],
               grid="lambda = 0 and 10^(k/20), k = -40..50 (0.01 to 316.2) on the failure side, the same grid up to 19.95 on the success side")
print("AO", {k: {a: (round(b, 8) if isinstance(b, float) else b) for a, b in v.items()} for k, v in [("success", R["ao"]["success"]), ("failure", R["ao"]["failure"])]})

# ------------------------------------------------------------------ 6. urn view: Frechet segment for small stones, ATE as a function of the urn's case mix
ps1, ps0 = p1[0], p0[0]; tau_s = ps1 - ps0
lo, hi = max(0, tau_s), min(ps1, 1 - ps0); hh = np.linspace(lo, hi, 200); harmed = hh - tau_s; S2 = (hh + harmed) - tau_s ** 2
FR = dict(h=hh, tau=tau_s, harmed=harmed, S2=S2)
R["frechet"] = dict(tau=tau_s, lo=lo, hi=hi, S2_min=S2.min(), S2_max=S2.max())
ate = {}
for L in SCALES:
    f, finv = links[L]; th = FIT[L][0]; d = finv(th[:2] + th[2]) - finv(th[:2])
    ate[L] = dict(small=d[0], large=d[1], trial=wt * d[0] + (1 - wt) * d[1])
R["ate_urn"] = dict(standardised=wt * (p1 - p0)[0] + (1 - wt) * (p1 - p0)[1], collapsed=P1 - P0, stable=ate)

# ------------------------------------------------------------------ 7. information on the (X, T, Y) table and the Pythagoras check
cnt = np.zeros((2, 2, 2)); cnt[:, 0, 1] = yA; cnt[:, 0, 0] = nA - yA; cnt[:, 1, 1] = yB; cnt[:, 1, 0] = nB - yB
Pj = cnt / cnt.sum()


def H(p):
    p = p[p > 0]; return -np.sum(p * np.log(p))


Px = Pj.sum((1, 2)); Pt = Pj.sum((0, 2)); Py = Pj.sum((0, 1)); Pxt = Pj.sum(2); Pxy = Pj.sum(1); Pty = Pj.sum(0)
I_xt = H(Px) + H(Pt) - H(Pxt.ravel()); I_ty = H(Pt) + H(Py) - H(Pty.ravel()); I_xy = H(Px) + H(Py) - H(Pxy.ravel())
I_ty_x = H(Pxt.ravel()) + H(Pxy.ravel()) - H(Pj.ravel()) - H(Px)
I_xt_y = H(Pxy.ravel()) + H(Pty.ravel()) - H(Pj.ravel()) - H(Py)
INFO = dict(I_XT=I_xt, I_TY=I_ty, I_TY_given_X=I_ty_x, I_XT_given_Y=I_xt_y, I_XY=I_xy, coinfo=I_ty - I_ty_x, coinfo_check=I_xt - I_xt_y,
            n_I_XT=N * I_xt, exp_minus_nI=np.exp(-N * I_xt), log10_prob=-N * I_xt / np.log(10), ratio_XT_over_TYgivenX=I_xt / I_ty_x, ratio_TY_over_TYgivenX=I_ty / I_ty_x)
R["info"] = INFO
w1 = nA / N; w0 = nB / N


def wKL(a1, a0, b1, b0):
    kl = lambda a, b: xlogy(a, a / b) + xlogy(1 - a, (1 - a) / (1 - b))
    return np.sum(w1 * kl(a1, b1) + w0 * kl(a0, b0))


PYTH = {}; R["pythagoras"] = dict(max_abs_resid={}, max_lhs={})
for L in ["logOR", "logRR"]:
    f, finv = links[L]; th = FIT[L][0]; s1 = finv(th[:2] + th[2]); s0 = finv(th[:2]); rows = []
    for dpsi in (np.linspace(-0.6, 0.6, 25) if L == "logOR" else np.linspace(-0.06, 0.06, 25)):
        for dg in [np.array([0.0, 0.0]), (np.array([0.15, -0.1]) if L == "logOR" else np.array([0.01, -0.01]))]:
            g = th[:2] + dg; q1 = finv(g + th[2] + dpsi); q0 = finv(g)
            rows.append((dpsi, dg[0], wKL(p1, p0, q1, q0), wKL(p1, p0, s1, s0) + wKL(s1, s0, q1, q0)))
    PYTH[L] = np.array(rows)
    R["pythagoras"]["max_abs_resid"][L] = np.abs(PYTH[L][:, 2] - PYTH[L][:, 3]).max(); R["pythagoras"]["max_lhs"][L] = PYTH[L][:, 2].max()
print("info", {k: round(float(v), 5) for k, v in INFO.items()}, "pythagoras", R["pythagoras"]["max_abs_resid"])


# ------------------------------------------------------------------ 8. profile rate functions J_G, exact posterior of psi by quadrature, seeded Metropolis chain
def profile_J(L, grid):
    th, f0 = FIT[L]

    def prof1(ps, start):
        g = start.copy()
        for _ in range(60):
            if nll(np.r_[g, ps], L, yA, nA, yB, nB) < 1e9: break
            g = g - 0.02 * np.maximum(1, abs(g)) * np.sign(ps if ps != 0 else 1)
        return minimize(lambda gg: nll(np.r_[gg, ps], L, yA, nA, yB, nB), g, method="Nelder-Mead", options={"xatol": 1e-11, "fatol": 1e-13, "maxiter": 5000})

    J = np.full(len(grid), np.nan); i0 = np.argmin(abs(grid - th[2]))
    for order in [range(i0, len(grid)), range(i0, -1, -1)]:
        g0 = th[:2].copy()
        for i in order:
            r = prof1(grid[i], g0)
            if r.fun < 1e9: J[i] = (r.fun - f0) / N; g0 = r.x
    return J


def stratum_logM(L, psis, gam, y1, n1, y0, n0):                                    # log of the integral over the baseline gamma of one stratum's likelihood, for each psi
    finv = links[L][1]; dg = gam[1] - gam[0]; q0 = finv(gam); out = np.empty(len(psis))
    for i, ps in enumerate(psis):
        q1 = finv(gam + ps); ok = (q0 > 0) & (q0 < 1) & (q1 > 0) & (q1 < 1)
        ll = np.full(gam.shape, -np.inf)
        ll[ok] = xlogy(y1, q1[ok]) + xlogy(n1 - y1, 1 - q1[ok]) + xlogy(y0, q0[ok]) + xlogy(n0 - y0, 1 - q0[ok])
        out[i] = logsumexp(ll) + np.log(dg)
    return out


def ess(x):                                                                        # effective sample size, initial positive sequence of the autocorrelations
    x = np.asarray(x, float) - np.mean(x); n = len(x)
    f = np.fft.rfft(x, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n]; rho = ac / ac[0]
    t_, k = -1.0, 0
    while k + 1 < n:
        pr = rho[k] + rho[k + 1]
        if pr <= 0: break
        t_ += 2 * pr; k += 2
    return n / t_


PGRID = {"RD": np.linspace(-0.07, 0.17, 121), "logRR": np.linspace(-0.08, 0.2, 121), "logOR": np.linspace(-0.6, 1.4, 121)}
QGRID = {"RD": (np.linspace(0, 1, 20001), np.linspace(-0.40, 0.50, 901)), "logRR": (np.linspace(-9, 0, 20001), np.linspace(-0.45, 0.65, 881)),
         "logOR": (np.linspace(-9, 9, 20001), np.linspace(-1.5, 2.5, 801))}
PROF = {}; R["profile"] = {}
for L in SCALES:
    th = FIT[L][0]; J = profile_J(L, PGRID[L]); PROF[L] = (PGRID[L], J, th[2])
    J0 = np.interp(0, PGRID[L], J); r0 = np.sqrt(2 * N * J0)
    gam, psis = QGRID[L]
    lm = sum(stratum_logM(L, psis, gam, yA[x], nA[x], yB[x], nB[x]) for x in (0, 1))
    dens = np.exp(lm - lm.max()); Z = trap(dens, psis); m = psis <= 1e-12
    pq = trap(dens[m], psis[m]) / Z; mq = trap(dens * psis, psis) / Z
    R["profile"][L] = dict(psi_hat=th[2], J0=J0, exp_nJ0=np.exp(-N * J0), phi_signed_root=norm.cdf(-r0), phi_minus_exact=norm.cdf(-r0) - pq,
                           post_le0_exact=pq, post_mean_exact=mq, J_nan_count=int(np.isnan(J).sum()))
    print(L, "psi_hat %.4f J0 %.5f exp(-nJ0) %.4f Phi(-r) %.4f | exact P(psi<=0) %.4f mean %.4f | nan %d" % (th[2], J0, np.exp(-N * J0), norm.cdf(-r0), pq, mq, np.isnan(J).sum()), flush=True)
THIN = 5
R["profile_summary"] = dict(post_le0_exact_min=min(R["profile"][L]["post_le0_exact"] for L in SCALES), post_le0_exact_max=max(R["profile"][L]["post_le0_exact"] for L in SCALES),
                            chain=dict(seed=7, steps=260000, burn_in=20000, kept=240000, thin_for_histogram=THIN, step_sd=dict(RD=0.02, logRR=0.03, logOR=0.25), prior="flat on (gamma_1, gamma_2, psi) on each chart"),
                            quadrature=dict(baseline_points=20001, psi_points=dict(RD=901, logRR=881, logOR=801)))
CH = {}
if FIGS_ONLY:
    dz = np.load(RES + "res_groups_kidney_draws.npz"); CH = {L: dz[L] for L in SCALES}
else:
    rng = np.random.default_rng(7); STEP = {"RD": 0.02, "logRR": 0.03, "logOR": 0.25}
    for L in SCALES:
        th, _ = FIT[L]; x = th.copy(); lp = -nll(x, L, yA, nA, yB, nB); S = []; na = 0
        for it in range(260000):
            z = x + STEP[L] * rng.standard_normal(3) * np.array([1, 1, 0.8]); lz = -nll(z, L, yA, nA, yB, nB)
            if np.log(rng.random()) < lz - lp: x, lp = z, lz; na += 1
            if it >= 20000: S.append(x[2])
        S = np.array(S); ind = (S <= 0).astype(float); e_ind = ess(ind)
        R["profile"][L].update(post_le0_metropolis=ind.mean(), metropolis_mc_se=np.sqrt(ind.var() / e_ind), metropolis_ess=e_ind, post_mean_metropolis=S.mean(),
                               metropolis_accept=na / 260000, metropolis_n_draws=len(S))
        CH[L] = S[::THIN]
        print(L, "chain P(psi<=0) %.4f se %.4f ess %.0f accept %.3f mean %.4f  (%.0f s)" % (ind.mean(), np.sqrt(ind.var() / e_ind), e_ind, na / 260000, S.mean(), time.time() - T0), flush=True)
    # -------------------------------------------------------------- 9. saturated Beta(1, 1) posteriors, collapsed analysis, transport
    M = 400000
    pA = rng.beta(1 + yA[:, None], 1 + nA[:, None] - yA[:, None], (2, M)); pB = rng.beta(1 + yB[:, None], 1 + nB[:, None] - yB[:, None], (2, M))
    wx = (nA + nB) / N
    cond = (wx[:, None] * (pA - pB)).sum(0)
    marg = (nA[:, None] / nA.sum() * pA).sum(0) - (nB[:, None] / nB.sum() * pB).sum(0)
    pc, pm = (cond <= 0).mean(), (marg >= 0).mean()
    R["saturated"] = dict(draws=M, std_rd_mean=np.sum(wx * ((yA + 1) / (nA + 2) - (yB + 1) / (nB + 2))), std_rd_mean_mc=cond.mean(), std_rd_P_le0=pc, std_rd_P_le0_mc_se=np.sqrt(pc * (1 - pc) / M),
                          collapsed_rd_mean=np.sum(nA / nA.sum() * (yA + 1) / (nA + 2)) - np.sum(nB / nB.sum() * (yB + 1) / (nB + 2)), collapsed_rd_mean_mc=marg.mean(),
                          collapsed_P_open_no_worse=pm, collapsed_P_open_no_worse_mc_se=np.sqrt(pm * (1 - pm) / M), collapsed_P_pcnl_no_worse=(marg <= 0).mean())
    wx = (nA + nB) / N
    effs = {"RD": lambda a, b: a - b, "logRR": lambda a, b: np.log(a / b), "logOR": lambda a, b: np.log(a / (1 - a)) - np.log(b / (1 - b))}
    R["transport"] = {}
    for L in SCALES:
        t = effs[L](p1, p0); Rg = abs(t[0] - t[1]); R["transport"][L] = {}
        for wp, name in [(np.array([1.0, 0.0]), "all_small"), (np.array([0.0, 1.0]), "all_large"), (np.array([0.8, 0.2]), "p80_small")]:
            kl = np.sum(wp[wp > 0] * np.log(wp[wp > 0] / wx[wp > 0])); tv = 0.5 * np.abs(wp - wx).sum(); shift = abs(wp @ t - wx @ t)
            R["transport"][L][name] = dict(KL=kl, shift=shift, tv_bound=Rg * tv, pinsker=Rg * np.sqrt(kl / 2), shift_over_psi=shift / FIT[L][0][2], range=Rg)
    print("saturated", {k: round(float(v), 5) for k, v in R["saturated"].items()})
    print("transport", {L: {k: round(float(v["shift"]), 4) for k, v in R["transport"][L].items()} for L in SCALES})
    with open(RES + "res_groups_kidney.json", "w") as fh: json.dump(plain(R), fh, indent=1)
    np.savez_compressed(RES + "res_groups_kidney_draws.npz", **CH)


# ------------------------------------------------------------------ 10. figures (6.4 in wide or less, text at least 8.5 pt, gray styles)
def fig_forest():
    fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.9), sharey=True)
    rows = ["small\nstones", "large\nstones", "pooled\nstable", "collapsed\n(marginal)"]
    for a, L in zip(ax, SCALES):
        F = R["forest"][L]
        for i in range(2):
            a.errorbar(F["est"][i], 3 - i, xerr=1.96 * F["se"][i], fmt="o", color="k", ms=5, capsize=2.5, lw=1.2)
        a.plot([F["stable"]], [1], "D", color="k", ms=6)
        a.errorbar(F["marginal"], 0, xerr=1.96 * F["marginal_se"], fmt="s", color="0.3", mfc="white", mec="0.1", ms=5.5, capsize=2.5, lw=1.0)
        a.axvline(0, color="0.55", lw=0.9, ls=":")
        a.set_title(LAB[L], fontsize=10); a.set_yticks([3, 2, 1, 0]); a.set_yticklabels(rows, fontsize=9); a.set_ylim(-0.6, 3.6)
        a.tick_params(axis="x", labelsize=9)
    save(fig, "kidney_forest.pdf")


def fig_inverse_sanov():
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.1), gridspec_kw={"width_ratios": [1, 1.7]})
    ks = ["logRR", "RD", "logOR"]; short = {"logRR": "log RR", "RD": "RD", "logOR": "log OR"}
    fills = {"logRR": "0.1", "RD": "0.5", "logOR": "0.85"}
    a = ax[0]
    for i, k in enumerate(ks):
        v = gap[k]["D"] * 1e4
        a.bar(i, v, width=0.62, color=fills[k], edgecolor="k", lw=0.8)
        a.text(i, v + 0.15, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    a.set_xticks(range(3)); a.set_xticklabels([short[k] for k in ks], fontsize=9)
    a.set_ylabel(r"information gap $D_G\times10^{4}$", fontsize=9.5); a.set_ylim(0, 8.6)
    b = ax[1]
    for k in ["logRR", "RD", "logOR", "sat"]:
        b.plot(MP["ns"], MP["P"][k], label=LAB[k], **ST[k])
    b.set_xscale("log"); b.axvline(N, color="0.55", lw=0.9, ls="--")
    b.text(N * 1.12, 0.97, r"observed $n=%d$" % N, fontsize=8.5, va="top")
    b.set_xlabel(r"number of patients $n$, same cell frequencies", fontsize=9.5); b.set_ylabel("leading order\nposterior weight", fontsize=9.5)
    b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8.5, handlelength=3, columnspacing=1.2)
    save(fig, "kidney_inverse_sanov.pdf")


def fig_closed():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    s, f = AO["success"], AO["failure"]
    ax.plot(s[1:, 0], s[1:, 1] * 1e4, color="k", ls="-", lw=2.0, label=r"success side, $h_\lambda(p)$")
    ax.plot(f[1:, 0], f[1:, 1] * 1e4, color="0.3", ls="--", lw=2.0, label=r"failure side, $h_\lambda(1-p)$")
    ax.axhline(gap["RD"]["D"] * 1e4, color="0.3", ls=":", lw=1.5, label="risk difference")
    ax.axhline(gap["logRR"]["D"] * 1e4, color="0.55", ls="-.", lw=1.5, label="risk ratio")
    i1 = int(np.where(s[:, 0] == 1.0)[0][0]); ax.plot([1.0], [s[i1, 1] * 1e4], "o", color="k", ms=6)
    ax.annotate(r"logit, $\lambda=1$", (1.0, s[i1, 1] * 1e4), (0.12, 9.2), fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xscale("log"); ax.set_ylim(0, 10)
    ax.set_xlabel(r"link parameter $\lambda$"); ax.set_ylabel(r"information gap $D_\lambda\times10^4$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8.5, handlelength=3, columnspacing=1.2)
    save(fig, "kidney_closed_groups.pdf")


def fig_effect():
    fig, ax = plt.subplots(2, 3, figsize=(6.4, 4.9), gridspec_kw={"height_ratios": [1, 1.1]}, sharex="col")
    for j, L in enumerate(SCALES):
        g, J, ph = PROF[L]; S = CH[L]; PL = R["profile"][L]
        a = ax[0, j]; a.plot(g, J * 1e3, color="k", lw=1.8)
        a.axvline(0, color="0.55", lw=0.9, ls=":"); a.plot([0], [PL["J0"] * 1e3], "o", color="k", ms=4)
        a.annotate(r"$J_G(0)=$" + "\n" + r"$%.2f\times10^{-3}$" % (PL["J0"] * 1e3), (0, PL["J0"] * 1e3), (0.5, 0.62), textcoords="axes fraction", ha="center", va="center",
                   fontsize=8.5, arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
        a.set_title(LAB[L], fontsize=10)
        if j == 0: a.set_ylabel(r"rate $J_G(\psi)\times10^{3}$", fontsize=9.5)
        b = ax[1, j]; dens = np.exp(-N * (J - np.nanmin(J))); dens /= trap(dens, g)
        b.hist(S, bins=50, density=True, color="0.85", edgecolor="none")
        m = g <= 0; b.fill_between(g[m], dens[m], color="0.55", lw=0)
        b.plot(g, dens, color="k", lw=1.8); b.axvline(0, color="0.55", lw=0.9, ls=":")
        b.text(0.97, 0.95, r"$\Pi(\psi\leq0)=%.3f$" % PL["post_le0_exact"], transform=b.transAxes, ha="right", va="top", fontsize=8.5)
        b.set_ylim(0, 1.4 * dens.max()); b.set_xlim(g[0], g[-1]); b.set_xlabel(r"stable effect $\psi$", fontsize=9.5)
        if j == 0: b.set_ylabel("posterior density", fontsize=9.5)
    hs = [Patch(facecolor="0.85", edgecolor="none", label="Metropolis draws"), Line2D([], [], color="k", lw=1.8, label=r"$\propto e^{-nJ_G(\psi)}$"),
          Patch(facecolor="0.55", edgecolor="none", label=r"area for $\psi\leq0$")]
    fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False, fontsize=8.5, handlelength=3)
    save(fig, "kidney_effect_ld.pdf", rect=(0, 0.06, 1, 1))


def fig_pythagoras():
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    sty = {"logRR": dict(marker="^", mfc="white", mec="k", ms=5.5), "logOR": dict(marker="o", mfc="k", mec="k", ms=4.5)}
    for L in ["logRR", "logOR"]:
        r = PYTH[L]; x = r[:, 0] / np.abs(r[:, 0]).max()
        ax.plot(x, (r[:, 2] - r[:, 3]) * 1e4, ls="none", label=LAB[L] + " family", **sty[L])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"position along the family (scaled $\psi-\hat\psi$)", fontsize=9.5); ax.set_ylabel(r"Pythagorean residual $\times10^{4}$")
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    save(fig, "kidney_information_pythagoras.pdf")


def fig_ate():
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 5.4))
    a = ax[0]
    a.plot(FR["h"], np.full_like(FR["h"], FR["tau"]), color="k", lw=2.6, label=r"ATE $\tau_{\mathrm{fp}}=p_1-p_0$")
    a.plot(FR["h"], FR["h"], color="0.3", ls="--", lw=1.8, label=r"helped fraction $p_{10}$ (PNS)")
    a.plot(FR["h"], FR["harmed"], color="0.55", ls="-.", lw=2.0, label=r"harmed fraction $p_{01}$")
    a.plot(FR["h"], FR["S2"], color="0.15", ls=":", lw=2.2, label=r"$V_\tau$, variance of unit effects")
    a.set_xlabel(r"helped fraction $p_{10}$ along the Fréchet segment"); a.set_ylabel("value"); a.set_ylim(0, 0.235)
    a.set_xlim(FR["h"][0] - 0.003, FR["h"][-1] + 0.003); a.legend(loc="upper left", frameon=False, fontsize=8.5, handlelength=3)
    b = ax[1]; mix = np.linspace(0, 1, 201)
    for L, lab in [("logRR", "stable risk ratio"), ("RD", "stable risk difference"), ("logOR", "stable odds ratio")]:
        f, finv = links[L]; th = FIT[L][0]; d = finv(th[:2] + th[2]) - finv(th[:2])
        b.plot(mix, mix * d[0] + (1 - mix) * d[1], label=lab, **ST[L])
    b.axvline(wt, color="0.55", lw=0.9, ls=":"); b.text(wt + 0.01, 0.097, "trial case mix", fontsize=8.5)
    b.plot([wt], [R["marginal"]["RD"]], "s", color="k", mfc="white", ms=6.5)
    b.annotate("collapsed contrast", (wt, R["marginal"]["RD"]), (0.62, -0.036), fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.6))
    b.axhline(0, color="k", lw=0.5); b.set_ylim(-0.06, 0.11)
    b.set_xlabel(r"share of small stones in the case mix $Q$"); b.set_ylabel(r"$\tau_Q$ (risk difference)")
    b.legend(loc="lower left", frameon=False, fontsize=8.5, handlelength=3)
    save(fig, "kidney_ate_urn.pdf")


for fn in [fig_forest, fig_inverse_sanov, fig_closed, fig_effect, fig_pythagoras, fig_ate]: fn()
print("done in %.0f s" % (time.time() - T0))
