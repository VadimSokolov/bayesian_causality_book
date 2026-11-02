"""Posterior arms and multiplier bootstrap of the DML audit (Chapter 7 head table), on the audit's own data sets.

Companion of dml_baseline_audit.py (same data generating process, same seeds crc32("design|r"), same 5,000 replications per design, so
every arm here sees the data set that the lasso, post double selection and repeated split arms of the audit saw). Designs s5, s15, s30
and weak: n = 300, p = 400, tau_0 = 1, X ~ N(0, I), the first s columns confound with treatment coefficient 1 and outcome coefficients
U(0.8, 1.2) drawn afresh in every replication, outcome noise sd 0.948, treatment noise sd 1 (0.3 for weak). The dense design and the
n = 1200 design are not run here. Arms, one record per replication:
  rf      reduced form horseshoe posterior (DiTraglia and Liu 2025; Gibbs sampler of Chapter 7 with the corrected variance conditionals
          of the r1023 fix), the posterior mean and standard deviation of tau and its 2.5 and 97.5 percent points;
  nb      naive Bayes: horseshoe on the structural outcome equation, tau unpenalised, same summaries;
  hs_K5, hs_K2   DML with a horseshoe nuisance (Gibbs, 1200 sweeps, burn 400, posterior mean as the fit) at five and two folds, one
          random split, sandwich standard error;
  boot    multiplier bootstrap (Mammen weights, B = 999) of the orthogonal score at the audit's K = 5 cross fit (same split and lasso
          fits as dml_K5 of the audit, so its point estimate equals dml_K5), percentile interval.
Convergence protocol of the posterior arms (rf and nb): 4 chains from the default start with rng seed + 3 + 10007 c (rf) or
seed + 4 + 10007 c (nb), L = 10,000 sweeps with the first half discarded; if split R-hat of tau exceeds 1.01 or the multichain ESS is
below 400, all four chains are rerun with L doubled, up to 80,000. The reported summaries pool the kept draws of the four chains.
Record: L, rhat, ess (tau), ess of log omega^2, converged flag. Output: results/dml_posterior_audit/DESIGN_START_END.jsonl; aggregation
`agg` writes results/res_dml_posterior_audit.json. Usage:  python dml_posterior_audit.py run DESIGN START END  |  agg
Requires numpy, scikit-learn, scipy. The sampler code below is copied verbatim from dml/sim/dmlsim.py (after the r1023 fixes).
"""
import glob, json, math, os, sys, time, warnings, zlib
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dml_baseline_audit import make_design, resid, plr, DESIGNS, Z95

OUTD = "results/dml_posterior_audit/"
AUDIT = "results/dml_baseline_audit/"
L0, LMAX = int(os.environ.get("R1_L0", 10000)), int(os.environ.get("R1_LMAX", 80000))

def _inv_gamma(rng, shape, rate):
    """Draw from InvGamma(shape, rate) as 1/Gamma(shape, scale=1/rate), guarding the
    underflow. When a horseshoe scale is driven hard toward zero the rate becomes
    enormous, numpy's Gamma draw returns exactly 0.0, and the reciprocal raises on a
    scalar or silently becomes inf on an array. Both poison the sweep. Clamping at the
    smallest positive double keeps the chain in the support without changing its law
    anywhere the law is representable."""
    g = rng.gamma(shape, 1.0 / np.maximum(rate, 1e-300))
    # Clip the value, not only the draw. The next sweep forms 1/om2 and 1/lam2 as rates,
    # so a scale that underflows to zero raises there instead of here. [1e-12, 1e12] is
    # far wider than any scale a horseshoe visits at these sample sizes.
    return np.clip(1.0 / np.maximum(g, 1e-300), 1e-12, 1e12)


def _bcm_sample(Phi, D_diag, alpha_vec, rng, gram=None):
    """Bhattacharya, Chakraborty and Mallick (2016): draw from
    N(A^{-1} Phi' alpha, A^{-1}) with A = Phi'Phi + D^{-1}, at cost O(n^2 p) when p > n.
    This is the sampler the manuscript names for the coefficient block.

    At p <= n the same draw is taken from the precision directly, by Cholesky of A,
    at cost O(np^2 + p^3). The BCM route forms an n by n matrix and is hopeless there:
    the 401(k) application of Section 16.4 has n = 9915. The two routes target the
    same conditional, and every simulated design in this file has p = 400 > n = 300,
    so the branch below is reached only by the application.
    """
    n, p = Phi.shape
    if p <= n:
        A = (Phi.T @ Phi) if gram is None else np.array(gram, float)
        A[np.diag_indices_from(A)] += 1.0 / D_diag
        L = np.linalg.cholesky(A)
        b = Phi.T @ alpha_vec
        mu = np.linalg.solve(L.T, np.linalg.solve(L, b))
        return mu + np.linalg.solve(L.T, rng.standard_normal(p))
    u = np.sqrt(D_diag) * rng.standard_normal(p)
    delta = rng.standard_normal(n)
    v = Phi @ u + delta
    PhiD = Phi * D_diag                      # Phi @ diag(D)
    M = PhiD @ Phi.T
    M[np.diag_indices_from(M)] += 1.0
    w = np.linalg.solve(M, alpha_vec - v)
    return u + PhiD.T @ w


def horseshoe_gibbs(X, y, rng, n_iter=1500, burn=500, thin=1, sigma2_fixed=None,
                    unpenalised=None, keep_scales=False, init_sig2=None):
    """Horseshoe linear regression by Gibbs, with the half Cauchy scales made
    conditionally inverse gamma by the Makalic and Schmidt (2016) augmentation and the
    coefficient block drawn by Bhattacharya et al. Both are the samplers Section 12 names.

    `unpenalised`, if given, is a column index that receives a flat prior instead of the
    horseshoe: this is how the naive arm puts a horseshoe on the structural equation while
    leaving theta unshrunk."""
    n, p = X.shape
    beta = np.zeros(p)
    lam2 = np.ones(p)
    om2 = 1.0
    nu_l = np.ones(p)
    xi = 1.0
    sig2 = (1.0 if init_sig2 is None else init_sig2) if sigma2_fixed is None else sigma2_fixed
    # r1023: the flat coefficient carries no horseshoe prior, so it is excluded from the
    # shape of the variance draw (p_pen, not p), from the quadratic penalty, and from the
    # local and global scale updates. Before the fix it counted in all four.
    pen = np.ones(p, dtype=bool)
    if unpenalised is not None:
        pen[unpenalised] = False
    p_pen = int(pen.sum())

    draws = np.empty((0, p))
    kept, om_kept, sig_kept = [], [], []
    for it in range(n_iter):
        prior_var = sig2 * om2 * lam2
        if unpenalised is not None:
            prior_var[unpenalised] = 1e6 * sig2
        Phi = X / math.sqrt(sig2)
        # D_diag is the prior variance of beta itself. Whitening the likelihood by sig2
        # rescales Phi and y and leaves the prior alone, so prior_var is passed as it is.
        beta = _bcm_sample(Phi, prior_var, y / math.sqrt(sig2), rng)

        resid = y - X @ beta
        if sigma2_fixed is None:
            shape = (n + p_pen) / 2.0
            quad = resid @ resid + np.sum(beta[pen] ** 2 / (om2 * lam2[pen]))
            sig2 = quad / 2.0 * _inv_gamma(rng, shape, 1.0)

        b2 = beta ** 2 / (2.0 * sig2 * om2)
        lam2 = _inv_gamma(rng, 1.0, 1.0 / nu_l + b2)
        nu_l = _inv_gamma(rng, 1.0, 1.0 + 1.0 / lam2)
        lam2[~pen] = 1.0
        nu_l[~pen] = 1.0
        om2 = _inv_gamma(rng, (p_pen + 1) / 2.0,
                         1.0 / xi + np.sum(beta[pen] ** 2 / lam2[pen]) / (2.0 * sig2))
        xi = _inv_gamma(rng, 1.0, 1.0 + 1.0 / om2)

        if it >= burn and (it - burn) % thin == 0:
            kept.append(beta.copy())
            om_kept.append(om2)
            sig_kept.append(sig2)
    draws = np.asarray(kept)
    out = dict(mean=draws.mean(0), draws=draws)
    if keep_scales:
        out['omega2'] = np.asarray(om_kept)
        out['sigma2'] = np.asarray(sig_kept)
    return out


def ess(x):
    """Effective sample size by the initial positive sequence rule."""
    x = np.asarray(x, float)
    m = len(x)
    if m < 8:
        return float(m)
    x = x - x.mean()
    v = x @ x / m
    if v <= 0:
        return float(m)
    f = np.fft.rfft(x, 2 * m)
    ac = np.fft.irfft(f * np.conjugate(f))[:m] / (m * v)
    s = 0.0
    for k in range(1, m - 1, 2):
        pair = ac[k] + ac[k + 1]
        if pair < 0:
            break
        s += pair
    return float(m / (1.0 + 2.0 * s))




def _rf_gibbs(X, D, Y, rng, n_iter=2000, burn=700, eval_block=None, eval_truth=None,
              t_df=0.0, block_theta=False, trace=False, init=None):
    """The reduced form horseshoe of eq:rf and eq:defaultprior, sampled by the Gibbs
    sweep of Section 12: alpha from eq:alphacond, varsigma the regression conditional of
    Y - theta nu on X, and theta ~ N(thetahat(alpha,varsigma), sig_eps^2/||nu||^2).

    `eval_block`, if given, is a held out (X, D, Y) on which the two residuals are formed
    from every kept draw of (alpha, varsigma). It is how the cross fitted variant of
    the cross fitted variant evaluates a posterior fitted elsewhere. `eval_truth`, the pair
    (X alpha_0, X varsigma_0) on that block, adds the two channels of eq:twochannel for
    the draw, which is that comment's other half: the paper's own decomposition applied
    to the paper's own estimator. Neither consumes a random number, so the chain is the
    one this sampler would have run without them.

    `t_df`, if positive, replaces the Gaussian outcome likelihood by a Student t on
    `t_df` degrees of freedom through the usual scale mixture: eps_i ~ N(0, sig_eps^2/w_i)
    with w_i ~ Gamma(t_df/2, t_df/2), so every conditional in the outcome equation becomes
    its weighted version. The treatment equation is untouched. It exists because a heavy
    tailed outcome, which net financial assets is, otherwise drives the reported width
    through a handful of observations."""
    n, p = X.shape
    # At p <= n the coefficient block is drawn from the precision, and both Gram
    # matrices are a scalar multiple of X'X, so it is formed once. At p > n, which
    # is every simulated design here, this is None and the BCM route is taken.
    XtX = (X.T @ X) if p <= n else None
    alpha = np.zeros(p)
    vsig = np.zeros(p)
    theta = 0.0
    s_nu2, s_ep2 = 1.0, 1.0
    if init is not None:        # r1023c: overdispersed starting values, default behaviour unchanged
        theta = float(init.get('theta', theta))
        s_nu2 = float(init.get('s_nu2', s_nu2))
        s_ep2 = float(init.get('s_ep2', s_ep2))
    lam2, om2, nu_l, xi = np.ones(p), 1.0, np.ones(p), 1.0
    tl2, tom2, tnu_l, txi = np.ones(p), 1.0, np.ones(p), 1.0
    wt = np.ones(n)

    th_keep, om_keep, tom_keep = [], [], []
    tr_all = []
    sn_keep, se_keep = [], []      # r1023: kept so the sampler can be checked against theory
    ev_num, ev_den, ev_s2 = [], [], []
    ev_own, ev_cross, ev_el2 = [], [], []
    # the terms the cross fitting argument kills: they are not zero for a pooled fit
    ev_nu_el, ev_em_eps, ev_nu_eps, ev_nu_em = [], [], [], []
    for it in range(n_iter):
        # alpha: enters both equations, eq:alphacond
        w = 1.0 / s_nu2 + theta ** 2 / s_ep2
        Lam_a = s_nu2 * om2 * lam2
        tgt = D / s_nu2 - theta * (Y - X @ vsig - theta * D) / s_ep2
        Phi = X * math.sqrt(w)
        alpha = _bcm_sample(Phi, Lam_a, tgt / math.sqrt(w), rng,
                            gram=None if XtX is None else w * XtX)

        s_nu2 = max(s_nu2, 1e-12)
        s_ep2 = max(s_ep2, 1e-12)
        nu_v = D - X @ alpha

        Lam_v = s_ep2 * tom2 * tl2
        sw = np.sqrt(wt)
        if block_theta:
            # r1023b: (theta, varsigma) drawn jointly. Given alpha the outcome equation is
            # the regression Y = theta nu + X varsigma + eps with a flat prior on theta
            # (variance 1e6, as the structural arm's flat coefficient) and the scaled
            # horseshoe on varsigma, so the pair is one Gaussian block. The same target as the
            # two step draw below; it removes the theta | varsigma coupling.
            Z = np.column_stack([nu_v, X])
            Phi_z = (Z * sw[:, None]) / math.sqrt(s_ep2)
            Dz = np.concatenate(([1e6 * s_ep2], Lam_v))
            zz = _bcm_sample(Phi_z, Dz, (Y * sw) / math.sqrt(s_ep2), rng)
            theta, vsig = float(zz[0]), zz[1:]
        else:
            # varsigma: regression of Y - theta nu on X
            targ = Y - theta * nu_v
            Phi_v = (X * sw[:, None]) / math.sqrt(s_ep2)
            vsig = _bcm_sample(Phi_v, Lam_v, (targ * sw) / math.sqrt(s_ep2), rng,
                               gram=None if (XtX is None or t_df > 0) else XtX / s_ep2)

            # theta
            r = Y - X @ vsig
            nn = float(nu_v @ (wt * nu_v))
            theta = float(rng.normal(float((wt * nu_v) @ r) / nn, math.sqrt(s_ep2 / nn)))

        # variances, Jeffreys. r1023: the coefficient priors are N(0, s2 * om2 * lam2), so
        # each variance appears in its prior as well as its likelihood, and the full
        # conditionals are inverse gamma with shape (n + p)/2 and rate
        # (residual sum of squares + sum_j coef_j^2 / (om2 lam2_j)) / 2. The first line
        # of the v1 sampler used shape n/2 and the residual sum of squares alone.
        # s_nu2 is conditionally inverse gamma although alpha also enters the outcome
        # equation: it does so through nu = D - X alpha, and s_nu2 appears nowhere in the
        # outcome likelihood, whose variance is s_ep2.
        rv = Y - X @ vsig - theta * nu_v
        q_ep = float(rv @ (wt * rv)) + float(np.sum(vsig ** 2 / (tom2 * tl2)))
        s_ep2 = q_ep / 2.0 * _inv_gamma(rng, (n + p) / 2.0, 1.0)
        q_nu = float(nu_v @ nu_v) + float(np.sum(alpha ** 2 / (om2 * lam2)))
        s_nu2 = q_nu / 2.0 * _inv_gamma(rng, (n + p) / 2.0, 1.0)

        if t_df > 0:
            # w_i | rest ~ Gamma((df+1)/2, (df + r_i^2/sig_eps^2)/2), which is the scale
            # mixture representation of the t. Drawn after s_ep2 so the two agree.
            wt = rng.gamma((t_df + 1.0) / 2.0, 2.0 / (t_df + rv ** 2 / s_ep2), size=n)

        # half Cauchy scales, Makalic and Schmidt
        b2 = alpha ** 2 / (2.0 * s_nu2 * om2)
        lam2 = _inv_gamma(rng, 1.0, 1.0 / nu_l + b2)
        nu_l = _inv_gamma(rng, 1.0, 1.0 + 1.0 / lam2)
        om2 = _inv_gamma(rng, (p + 1) / 2.0,
                         1.0 / xi + np.sum(alpha ** 2 / lam2) / (2.0 * s_nu2))
        xi = _inv_gamma(rng, 1.0, 1.0 + 1.0 / om2)

        tb2 = vsig ** 2 / (2.0 * s_ep2 * tom2)
        tl2 = _inv_gamma(rng, 1.0, 1.0 / tnu_l + tb2)
        tnu_l = _inv_gamma(rng, 1.0, 1.0 + 1.0 / tl2)
        tom2 = _inv_gamma(rng, (p + 1) / 2.0,
                          1.0 / txi + np.sum(vsig ** 2 / tl2) / (2.0 * s_ep2))
        txi = _inv_gamma(rng, 1.0, 1.0 + 1.0 / tom2)

        if trace:
            tr_all.append((theta, om2, tom2, s_nu2, s_ep2))
        if it >= burn:
            th_keep.append(theta)
            om_keep.append(om2)
            tom_keep.append(tom2)
            sn_keep.append(s_nu2)
            se_keep.append(s_ep2)
            if eval_block is not None:
                Xe, De, Ye = eval_block
                m_e, l_e = Xe @ alpha, Xe @ vsig
                nu_e, r_e = De - m_e, Ye - l_e
                ev_num.append(float(nu_e @ r_e))
                ev_den.append(float(nu_e @ nu_e))
                ev_s2.append(float(s_ep2))
                if eval_truth is not None:
                    em, el = m_e - eval_truth[0], l_e - eval_truth[1]
                    nb = len(De)
                    ev_own.append(float(em @ em) / nb)
                    ev_cross.append(float(em @ el) / nb)
                    ev_el2.append(float(el @ el) / nb)
                    # nu and eps on the evaluation block, from the truth the design knows.
                    # theta_0 is 1 in every design here, and l_0 = theta_0 m_0 + g_0, so
                    # eps = (Y - l_0) - theta_0 (D - m_0).
                    nu_t = De - eval_truth[0]
                    eps_t = (Ye - eval_truth[1]) - nu_t
                    ev_nu_el.append(float(nu_t @ el) / nb)
                    ev_em_eps.append(float(em @ eps_t) / nb)
                    ev_nu_eps.append(float(nu_t @ eps_t) / nb)
                    ev_nu_em.append(float(nu_t @ em) / nb)

    return dict(trace=np.asarray(tr_all), theta=np.asarray(th_keep), omega2=np.asarray(om_keep),
                omega2_tilde=np.asarray(tom_keep), sig_nu2=np.asarray(sn_keep),
                sig_ep2=np.asarray(se_keep), ev_num=np.asarray(ev_num),
                ev_den=np.asarray(ev_den), ev_s2=np.asarray(ev_s2),
                ev_own=np.asarray(ev_own), ev_cross=np.asarray(ev_cross),
                ev_el2=np.asarray(ev_el2), ev_nu_el=np.asarray(ev_nu_el),
                ev_em_eps=np.asarray(ev_em_eps), ev_nu_eps=np.asarray(ev_nu_eps),
                ev_nu_em=np.asarray(ev_nu_em))




def _acov(x):
    n = x.shape[-1]
    x = x - x.mean(-1, keepdims=True)
    f = np.fft.rfft(x, 2 * n, axis=-1)
    return np.fft.irfft(f * np.conjugate(f), axis=-1)[..., :n] / n


def split_rhat_ess(ch):
    """ch is (chains, draws). Classic split R-hat and the multichain ESS of Vehtari et al. (2021),
    without the rank normalisation."""
    ch = np.asarray(ch, float)
    C, N = ch.shape
    h = N // 2
    sp = np.concatenate([ch[:, :h], ch[:, h:2 * h]], axis=0)       # (2C, h)
    M, n = sp.shape
    cm = sp.mean(1)
    W = sp.var(1, ddof=1).mean()
    B = n * cm.var(ddof=1)
    vp = (n - 1) / n * W + B / n
    rhat = math.sqrt(vp / W) if W > 0 else float('nan')
    ac = _acov(sp).mean(0) * n / (n - 1)
    rho = 1.0 - (W - ac) / vp
    rho[0] = 1.0
    s = 0.0
    for k in range(1, n - 2, 2):
        pr = rho[k] + rho[k + 1]
        if pr < 0:
            break
        s += pr
    tau = -1.0 + 2.0 * (1.0 + s)
    return rhat, float(M * n / max(tau, 1e-9 + 1.0 / 1e9))



def chains(fn, L, seed0):
    outs = [fn(np.random.default_rng(seed0 + 10007 * c), L, L // 2) for c in range(4)]
    return outs


def summarize(th4, extra_ess, L):
    rh, es = split_rhat_ess(th4); th = th4.ravel(); lo, hi = np.quantile(th, [0.025, 0.975])
    return dict(theta=float(th.mean()), se=float(th.std(ddof=1)), lo=float(lo), hi=float(hi), L=L, rhat=rh, ess=es,
                ess_omega=extra_ess, converged=bool(rh <= 1.01 and es >= 400), chain_means=[float(x) for x in th4.mean(1)])


def run_posterior(fn, key_theta, key_om, seed0):
    L = L0
    while True:
        outs = chains(fn, L, seed0); th4 = np.array([key_theta(o) for o in outs])
        rh, es = split_rhat_ess(th4)
        if (rh <= 1.01 and es >= 400) or L >= LMAX: break
        L *= 2
    eo = split_rhat_ess(np.log([key_om(o) for o in outs]))[1]
    return summarize(th4, eo, L)


def one_rep(design, r):
    seed = zlib.crc32(("%s|%d" % (design, r)).encode()) % (2 ** 31)
    X, D, Y = make_design(np.random.default_rng(seed), **DESIGNS[design]); n = len(D)
    rec = dict(design=design, rep=r, seed=seed)
    rec["rf"] = run_posterior(lambda rng, L, b: _rf_gibbs(X, D, Y, rng, L, b), lambda o: o["theta"], lambda o: o["omega2"], seed + 3)
    Z = np.column_stack([D, X])
    rec["nb"] = run_nb(Z, Y, seed + 4)
    rec["hs_K5"] = dml_hs(X, D, Y, np.random.default_rng(seed + 2), 5); rec["hs_K2"] = dml_hs(X, D, Y, np.random.default_rng(seed + 22), 2)
    rD, rY = resid(X, D, Y, np.random.default_rng(seed + 1), 5); th, se = plr(rD, rY)
    J = np.mean(rD ** 2); psi = rD * (rY - th * rD) / J; wr = np.random.default_rng(seed + 9)
    a = (1 - math.sqrt(5)) / 2; b = (1 + math.sqrt(5)) / 2; pr = (math.sqrt(5) + 1) / (2 * math.sqrt(5))
    u = np.where(wr.random((999, n)) < pr, a, b)
    draws = th + (u - 1.0) @ psi / n; lo, hi = np.quantile(draws, [0.025, 0.975])
    rec["boot"] = dict(theta=th, se=se, lo=float(lo), hi=float(hi), se_boot=float(np.std(draws, ddof=1)))
    return rec


def run_nb(Z, Y, seed0):
    L = L0
    while True:
        outs = [horseshoe_gibbs(Z, Y, np.random.default_rng(seed0 + 10007 * c), n_iter=L, burn=L // 2, unpenalised=0, keep_scales=True) for c in range(4)]
        th4 = np.array([o["draws"][:, 0] for o in outs]); rh, es = split_rhat_ess(th4)
        if (rh <= 1.01 and es >= 400) or L >= LMAX: break
        L *= 2
    eo = split_rhat_ess(np.log([o["omega2"] for o in outs]))[1]
    return summarize(th4, eo, L)


def dml_hs(X, D, Y, rng, K, hs_iter=1200):
    n = len(D); idx = rng.permutation(n); m = np.empty(n); l = np.empty(n)
    for f in np.array_split(idx, K):
        tr = np.setdiff1d(idx, f)
        cm = horseshoe_gibbs(X[tr], D[tr], rng, n_iter=hs_iter, burn=hs_iter // 3)["mean"]
        cl = horseshoe_gibbs(X[tr], Y[tr], rng, n_iter=hs_iter, burn=hs_iter // 3)["mean"]
        m[f] = X[f] @ cm; l[f] = X[f] @ cl
    return plr(D - m, Y - l)


def _load(pattern):
    return {(json.loads(l)["rep"]): json.loads(l) for f in sorted(glob.glob(pattern)) for l in open(f)}


def _iv(rec, arm):
    if arm in ("rf", "nb", "boot"): return rec[arm]["theta"], rec[arm]["lo"], rec[arm]["hi"], (rec[arm]["se_boot"] if arm == "boot" else rec[arm]["se"])
    th, se = rec[arm][0], rec[arm][1]; return th, th - Z95 * se, th + Z95 * se, se


def agg():
    from scipy.stats import binomtest
    names = dict(dml_K5="DML lasso K=5", dml_K2="DML lasso K=2", rs_K5="split median K=5", rs_K2="split median K=2", pds_theory="PDS theory", pds_cv="PDS CV",
                 hs_K5="DML horseshoe K=5", hs_K2="DML horseshoe K=2", nb="naive Bayes", rf="reduced form posterior", boot="multiplier bootstrap")
    pairs = [("rf", "dml_K5"), ("rf", "hs_K5"), ("rf", "pds_cv"), ("rf", "pds_theory"), ("rf", "rs_K5"), ("rf", "boot"), ("hs_K5", "dml_K5"), ("hs_K5", "dml_K2"), ("hs_K2", "hs_K5"),
             ("pds_theory", "dml_K5"), ("rs_K5", "dml_K5"), ("boot", "dml_K5"), ("dml_K5", "dml_K2"), ("pds_cv", "dml_K5"), ("nb", "rf"), ("hs_K5", "pds_theory")]
    out, lines = {}, []
    for d in ["s5", "s15", "s30", "weak"]:
        new = _load(OUTD + d + "_*.jsonl"); old = _load(AUDIT + d + "_*.jsonl"); reps = sorted(set(new) & set(old))
        if not reps: continue
        recs = []
        for r in reps:
            x = dict(old[r]); x.update(new[r]); assert old[r]["seed"] == new[r]["seed"]; recs.append(x)
        N = len(recs); o = out[d] = dict(reps=N, arms={}, mcnemar={})
        lines.append("## %s (%d paired replications)" % (d, N)); lines.append("| arm | bias (MCSE) | coverage (SE) | s.e./s.d. | s.e./RMSE | width | s.d. | mean s.e. |"); lines.append("|---|---|---|---|---|---|---|---|")
        cov = {}
        for a in ["dml_K5", "hs_K5", "hs_K2", "nb", "rf", "dml_K2", "rs_K5", "pds_theory", "pds_cv", "boot"]:
            v = np.array([_iv(x, a) for x in recs]); th, lo, hi, se = v.T; c = (lo <= 1) & (1 <= hi); cov[a] = c
            sd = th.std(ddof=1); rmse = math.sqrt(np.mean((th - 1) ** 2)); cc = c.mean()
            o["arms"][a] = dict(bias=float(th.mean() - 1), bias_mcse=float(sd / math.sqrt(N)), coverage=float(cc), coverage_se=float(math.sqrt(cc * (1 - cc) / N)), sd=float(sd),
                                mean_se=float(se.mean()), se_over_sd=float(se.mean() / sd), se_over_rmse=float(se.mean() / rmse), width=float((hi - lo).mean()))
            q = o["arms"][a]; lines.append("| %s | %.3f (%.3f) | %.3f (%.3f) | %.2f | %.2f | %.3f | %.4f | %.4f |" % (names[a], q["bias"], q["bias_mcse"], q["coverage"], q["coverage_se"], q["se_over_sd"], q["se_over_rmse"], q["width"], q["sd"], q["mean_se"]))
        for a in ("rf", "nb"):
            cv = np.array([x[a]["converged"] for x in recs]); Ls = np.array([x[a]["L"] for x in recs])
            o[a + "_convergence"] = dict(share_converged=float(cv.mean()), n_converged=int(cv.sum()), share_L_gt_10000=float((Ls > L0).mean()), max_L=int(Ls.max()),
                                         median_ess=float(np.median([x[a]["ess"] for x in recs])), min_ess=float(np.min([x[a]["ess"] for x in recs])), max_rhat=float(np.max([x[a]["rhat"] for x in recs])),
                                         median_ess_omega=float(np.median([x[a]["ess_omega"] for x in recs])))
            if cv.any():
                th = np.array([x[a]["theta"] for x in recs])[cv]; se = np.array([x[a]["se"] for x in recs])[cv]
                o[a + "_convergence"]["se_over_sd_converged_only"] = float(se.mean() / th.std(ddof=1)); o[a + "_convergence"]["coverage_converged_only"] = float(cov[a][cv].mean())
            lines.append("convergence %s: %s" % (a, json.dumps(o[a + "_convergence"])))
        for a, b in pairs:
            ca, cb = cov[a], cov[b]; x = int((ca & ~cb).sum()); y = int((~ca & cb).sum()); diff = ca.astype(float) - cb
            p = binomtest(min(x, y), x + y, 0.5).pvalue if x + y > 0 else 1.0
            o["mcnemar"]["%s_vs_%s" % (a, b)] = dict(only_first=x, only_second=y, p=float(p), diff=float(diff.mean()), diff_se=float(diff.std(ddof=1) / math.sqrt(N)))
            lines.append("McNemar %s vs %s: discordant %d (first only) to %d (second only), exact p %.3g, paired coverage difference %.3f (SE %.3f)" % (names[a], names[b], x, y, p, diff.mean(), diff.std(ddof=1) / math.sqrt(N)))
        lines.append("")
    json.dump(out, open("results/res_dml_posterior_audit.json", "w"), indent=1); open("results/dml_posterior_audit_tables.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        d, a, b = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); os.makedirs(OUTD, exist_ok=True); t0 = time.time()
        with open(OUTD + "%s_%05d_%05d.jsonl" % (d, a, b), "w") as fh:
            for r in range(a, b):
                rec = one_rep(d, r); rec["elapsed"] = round(time.time() - t0, 1); fh.write(json.dumps(rec) + "\n"); fh.flush()
        print("done", d, a, b, round(time.time() - t0))
    else: agg()
