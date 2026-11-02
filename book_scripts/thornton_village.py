"""Chapter 17 (Thornton HIV incentive experiment), item F1 of round 1 (October 2026): audit of the assignment across villages, and the logistic models of Section sec:thornton refitted with village intercepts.
Run from the repository root with PYTHONPATH=. (needs polyagamma); writes results/res_thornton_village.json.

Data: the 2,816 respondents of apps2.py (same filter: no missing got, any, distvct, age or tinc, and hiv2004 >= 0), in 119 villages (villnum); four respondents have no village number and each gets an intercept of their own, so there are 123 intercepts.
The district of a village is not in thornton_hiv.dta (the file has villnum, got, distvct, tinc, any, age, hiv2004) and cannot be recovered from the village number, so village intercepts replace district effects.

Audit: share of respondents offered any incentive in each village, the chi squared test of equal shares over the 119 villages (the four respondents without a village number left out), and the same shares and test by bands of village number as a rough proxy for geography.

Models: exactly those of apps2.py (same covariates dz = standardized distance, az = standardized age, same Polya Gamma logistic sampler, independent N(0, 10) priors on the coefficients) with a village intercept u_j ~ N(0, sigma^2) added to the linear predictor, sigma half Cauchy with scale 1 (inverse gamma form, as in the solution of Exercise ex:chthornton:2).
(beta, u) are drawn in one block given the Polya Gamma variables, sigma^2 and its auxiliary variable in two inverse gamma steps. Four chains of 4,000 iterations, the first 1,000 dropped, seeds 1602 to 1605, so 12,000 draws per model; the same sampler with no village term (J = 0) is run with the same seeds as the comparison, so that both columns come from one code path.
Three models: main (intercept, any, dz, az, any x dz), without the interaction (intercept, any, dz, az), and the dose model (intercept, any, tinc, max(tinc - 1, 0), dz, az).
Quantities, each as [mean, sd, 2.5%, 97.5%] over the pooled draws: the sample average risk difference (each draw's village intercepts held at their draws, averaged over the 2,816 respondents), the risk differences in the nearest and farthest quarters by distance, the conditional odds ratio exp(beta_any) (at mean distance), the marginal odds ratio formed from the sample averaged probabilities, their gap, the interaction coefficient and its posterior probability of being positive, sigma, and the dose curve on the 31 point grid of apps2.py and at the amounts 0.09456 (the smallest offered), 1 and the largest, 2.8368.
Split R hat and effective sample size are reported for the risk difference and for sigma.
"""
import json, time, warnings
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency
from polyagamma import random_polyagamma
from causaldata import thornton_hiv
warnings.filterwarnings("ignore")
t0 = time.time()
expit = lambda z: 1 / (1 + np.exp(-z))
def summ(x): return [float(np.mean(x)), float(np.std(x)), float(np.quantile(x, .025)), float(np.quantile(x, .975))]

h = thornton_hiv.load_pandas().data.dropna(subset=["got", "any", "distvct", "age", "tinc"])
h = h[h.hiv2004 >= 0].reset_index(drop=True)
yv = h.got.values.astype(float); av = h["any"].values.astype(float); dist = h.distvct.values.astype(float); age = h.age.values.astype(float); tinc = h.tinc.values.astype(float)
dz = (dist - dist.mean()) / dist.std(); az = (age - age.mean()) / age.std()
R = {"n": int(len(yv))}

# ---------- audit ----------
vn = h.villnum.values
has = ~np.isnan(vn)
tab = pd.crosstab(vn[has], av[has])
chi, pval, dof, _ = chi2_contingency(tab.values, correction=False)
share = tab[1.0] / tab.sum(1)
R["audit"] = {"n_villages": int(len(tab)), "n_without_village": int((~has).sum()), "overall_share_any": float(av.mean()),
              "village_share_quantiles": {str(q): float(share.quantile(q)) for q in [0, .1, .25, .5, .75, .9, 1]},
              "village_size_quantiles": {str(q): float(tab.sum(1).quantile(q)) for q in [0, .25, .5, .75, 1]},
              "n_villages_all_any": int((share == 1).sum()), "n_villages_none_any": int((share == 0).sum()),
              "chi2": float(chi), "df": int(dof), "p": float(pval)}
bands = pd.cut(vn[has], [0, 29, 59, 89, 119, 149]); tb = pd.crosstab(bands, av[has]); c2, p2, d2, _ = chi2_contingency(tb.values, correction=False)
R["audit"]["by_village_number_band"] = {str(k): [int(tb.loc[k].sum()), float(tb.loc[k, 1.0] / tb.loc[k].sum())] for k in tb.index}
R["audit"]["band_chi2"] = [float(c2), int(d2), float(p2)]
# the village shares of any incentive against the village mean distance (a design variable that might explain the variation)
vd = pd.Series(dist[has]).groupby(vn[has]).mean()
R["audit"]["corr_share_any_mean_distance"] = float(np.corrcoef(share.values, vd.loc[share.index].values)[0, 1])
print("audit", R["audit"]["chi2"], R["audit"]["df"], R["audit"]["p"], round(time.time() - t0, 1), flush=True)

# ---------- samplers ----------
vid = np.zeros(len(yv), int); nxt = int(np.nanmax(vn)) + 1
lab = {}
for i, v in enumerate(vn):
    key = ("v", v) if not np.isnan(v) else ("s", i)
    if key not in lab: lab[key] = len(lab)
    vid[i] = lab[key]
J = len(lab); R["n_intercepts"] = J
W = np.zeros((len(yv), J)); W[np.arange(len(yv)), vid] = 1.0

def pg_re(X, y, rng, village, iters=4000, burn=1000, v0=10.0):
    nn, q = X.shape
    Z = np.column_stack([X, W]) if village else X
    qq = Z.shape[1]; kap = y - 0.5; b = np.zeros(qq); s2 = 1.0; nu = 1.0; Bs, Us, Ss = [], [], []
    for it in range(iters):
        om = random_polyagamma(1, Z @ b, random_state=rng)
        prec = np.concatenate([np.full(q, 1 / v0), np.full(qq - q, 1 / s2)])
        V = np.linalg.inv(Z.T @ (Z * om[:, None]) + np.diag(prec)); V = (V + V.T) / 2
        b = rng.multivariate_normal(V @ (Z.T @ kap), V)
        if village:
            u = b[q:]
            s2 = 1 / rng.gamma((J + 1) / 2, 1 / (1 / nu + (u ** 2).sum() / 2)); nu = 1 / rng.gamma(1, 1 / (1 + 1 / s2))
        if it >= burn:
            Bs.append(b[:q].copy()); Us.append(b[q:].copy() if village else np.zeros(J)); Ss.append(np.sqrt(s2) if village else 0.0)
    return np.array(Bs), np.array(Us), np.array(Ss)

def rhat_ess(ch):
    ch = np.asarray(ch, float); m, nn = ch.shape; hh = nn // 2
    parts = np.vstack([ch[:, :hh], ch[:, hh:2 * hh]]); W_ = parts.var(axis=1, ddof=1).mean(); B_ = hh * parts.mean(axis=1).var(ddof=1)
    rh = float(np.sqrt(((hh - 1) / hh * W_ + B_ / hh) / W_))
    x = ch - ch.mean(axis=1, keepdims=True); f = np.fft.rfft(x, 2 * nn, axis=1); ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :nn].mean(axis=0)
    ac = ac / ac[0]; s = 0.0
    for k in range(1, nn - 1, 2):
        pr = ac[k] + ac[k + 1]
        if pr < 0: break
        s += pr
    return rh, float(m * nn / (1 + 2 * s))

def run(X, village, tag):
    out = []
    for c in range(4):
        rng = np.random.default_rng(1602 + c)
        out.append(pg_re(X, yv, rng, village))
        print(tag, "village" if village else "plain", "chain", c, round(time.time() - t0, 1), flush=True)
    return out

def cat(ch, k): return np.vstack([c[k] for c in ch])

near = dist < np.quantile(dist, 0.25); far = dist > np.quantile(dist, 0.75)
one = np.ones(len(yv))
X_main = np.column_stack([one, av, dz, az, av * dz]); X_noint = X_main[:, :4]
X_dose = np.column_stack([one, av, tinc, np.maximum(tinc - 1, 0), dz, az])
R["fits"] = {}
for village in [False, True]:
    key = "village" if village else "plain"
    F = {}
    # main model
    ch = run(X_main, village, "main"); B = cat(ch, 0); U = cat(ch, 1); S = cat(ch, 2)
    Z1 = X_main.copy(); Z1[:, 1] = 1; Z1[:, 4] = dz; Z0 = X_main.copy(); Z0[:, 1] = 0; Z0[:, 4] = 0
    rd_all, rd_near, rd_far, m1l, m0l = [], [], [], [], []
    for a in range(0, len(B), 1000):
        off = U[a:a + 1000][:, vid] if village else 0.0
        p1 = expit(B[a:a + 1000] @ Z1.T + off); p0 = expit(B[a:a + 1000] @ Z0.T + off); dd = p1 - p0
        rd_all.append(dd.mean(1)); rd_near.append(dd[:, near].mean(1)); rd_far.append(dd[:, far].mean(1)); m1l.append(p1.mean(1)); m0l.append(p0.mean(1))
    rd = np.concatenate(rd_all); m1_ = np.concatenate(m1l); m0_ = np.concatenate(m0l)
    mor = (m1_ / (1 - m1_)) / (m0_ / (1 - m0_)); cor = np.exp(B[:, 1])
    F["rd"] = summ(rd); F["rd_near"] = summ(np.concatenate(rd_near)); F["rd_far"] = summ(np.concatenate(rd_far))
    F["cond_or"] = summ(cor); F["marg_or"] = summ(mor); F["gap"] = summ(cor - mor)
    F["delta"] = summ(B[:, 4]); F["delta_ppos"] = float((B[:, 4] > 0).mean()); F["p0_mean"] = summ(m0_)
    F["rd_rhat_ess"] = rhat_ess(rd.reshape(4, -1))
    if village:
        F["sigma"] = summ(S); F["sigma_rhat_ess"] = rhat_ess(S.reshape(4, -1))
    # no interaction
    ch = run(X_noint, village, "noint"); B = cat(ch, 0); U = cat(ch, 1)
    Z1 = X_noint.copy(); Z1[:, 1] = 1; Z0 = X_noint.copy(); Z0[:, 1] = 0
    m1l, m0l = [], []
    for a in range(0, len(B), 1000):
        off = U[a:a + 1000][:, vid] if village else 0.0
        m1l.append(expit(B[a:a + 1000] @ Z1.T + off).mean(1)); m0l.append(expit(B[a:a + 1000] @ Z0.T + off).mean(1))
    m1_ = np.concatenate(m1l); m0_ = np.concatenate(m0l); mor = (m1_ / (1 - m1_)) / (m0_ / (1 - m0_)); cor = np.exp(B[:, 1])
    F["nointer_cond_or"] = summ(cor); F["nointer_marg_or"] = summ(mor); F["nointer_gap"] = summ(cor - mor); F["nointer_rd"] = summ(m1_ - m0_)
    # dose model
    ch = run(X_dose, village, "dose"); B = cat(ch, 0); U = cat(ch, 1)
    grid = np.linspace(0, tinc.max(), 31); extra = [float(np.min(tinc[tinc > 0])), 1.0, float(tinc.max())]
    def curve(g):
        Zg = X_dose.copy(); Zg[:, 1] = float(g > 0); Zg[:, 2] = g; Zg[:, 3] = max(g - 1, 0)
        return np.concatenate([expit(B[a:a + 1000] @ Zg.T + (U[a:a + 1000][:, vid] if village else 0.0)).mean(1) for a in range(0, len(B), 1000)])
    C = np.array([curve(g) for g in grid])
    F["dose"] = {"grid": grid.tolist(), "mean": C.mean(1).tolist(), "lo": np.quantile(C, .025, 1).tolist(), "hi": np.quantile(C, .975, 1).tolist()}
    F["dose_at"] = {str(round(g, 5)): summ(curve(g)) for g in [0.0] + extra}
    F["dose_jump"] = summ(curve(extra[0]) - curve(0.0))
    # the amount that is best per unit spent: smallest positive offered, per draw (check of the chapter's statement)
    gp = np.array([g for g in grid if g > 0]); ratio = np.array([(curve(g) - curve(0.0)) / (g * curve(g)) for g in gp])
    F["share_draws_smallest_positive_best_ratio"] = float((ratio.argmax(0) == 0).mean())
    R["fits"][key] = F
    print(key, "rd", F["rd"], "cond_or", F["cond_or"], "marg_or", F["marg_or"], round(time.time() - t0, 1), flush=True)
R["seconds"] = round(time.time() - t0, 1)
json.dump(R, open("results/res_thornton_village.json", "w"), indent=1)
print(json.dumps(R["audit"], indent=1)); print("done", R["seconds"])
