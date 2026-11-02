"""Chapter 17: a check on the Thornton logistic model of apps2.py.

The posterior of the treatment by distance interaction, and the gap between the conditional odds ratio at
mean distance and the marginal odds ratio, with and without the interaction. Same sample, design and N(0, 10)
priors as apps2.py, but the posterior is the Laplace (normal) approximation at the posterior mode, so the
script needs no polyagamma; 20,000 draws, seed 1971. With n = 2,816 and five coefficients the approximation
is close to the Polya Gamma sampler. When apps2.py is rerun, its th_delta and th_nointer_gap keys give the
Polya Gamma version of these numbers.
Run from the repository root; writes results/res_thornton_laplace.json.
"""
import json
import numpy as np, pandas as pd
h = pd.read_stata('data/thornton_hiv.dta').dropna(subset=["got", "any", "distvct", "age", "tinc"])
h = h[h.hiv2004 >= 0] if "hiv2004" in h else h
yv = h.got.values.astype(float); av = h["any"].values.astype(float); dist = h.distvct.values.astype(float); age = h.age.values.astype(float)
dz = (dist - dist.mean()) / dist.std(); az = (age - age.mean()) / age.std()
expit = lambda z: 1 / (1 + np.exp(-z))
rng = np.random.default_rng(1971)
summ = lambda x: [round(float(np.mean(x)), 4), round(float(np.quantile(x, .025)), 4), round(float(np.quantile(x, .975)), 4)]


def laplace(Zm, y, v0=10.0, draws=20000):
    q = Zm.shape[1]; b = np.zeros(q)
    for _ in range(100):
        p = expit(Zm @ b); g = Zm.T @ (y - p) - b / v0; H = Zm.T @ (Zm * (p * (1 - p))[:, None]) + np.eye(q) / v0
        step = np.linalg.solve(H, g); b = b + step
        if np.abs(step).max() < 1e-10: break
    V = np.linalg.inv(H); V = (V + V.T) / 2
    return rng.multivariate_normal(b, V, size=draws)


out = {'n': int(len(yv)), 'raw': [float(yv[av == 1].mean()), float(yv[av == 0].mean())]}
for name, inter in (('with_interaction', True), ('no_interaction', False)):
    cols = [np.ones(len(yv)), av, dz, az] + ([av * dz] if inter else [])
    Zt = np.column_stack(cols); B = laplace(Zt, yv)
    Z1 = Zt.copy(); Z1[:, 1] = 1; Z0 = Zt.copy(); Z0[:, 1] = 0
    if inter: Z1[:, 4] = dz; Z0[:, 4] = 0
    p1 = expit(B @ Z1.T); p0 = expit(B @ Z0.T)
    m1, m0 = p1.mean(1), p0.mean(1)
    mor = (m1 / (1 - m1)) / (m0 / (1 - m0)); cor = np.exp(B[:, 1])
    r = {'cond_or': summ(cor), 'marg_or': summ(mor), 'cond_minus_marg': summ(cor - mor), 'rd': summ(m1 - m0),
         'p0_unit_quantiles_05_50_95': [round(float(np.quantile(p0.mean(0), q)), 4) for q in (0.05, 0.5, 0.95)],
         'p1_unit_quantiles_05_50_95': [round(float(np.quantile(p1.mean(0), q)), 4) for q in (0.05, 0.5, 0.95)]}
    if inter:
        r['delta'] = summ(B[:, 4]); r['p_delta_pos'] = round(float((B[:, 4] > 0).mean()), 3)
        r['cond_or_at_dz_q125_q875'] = [summ(np.exp(B[:, 1] + B[:, 4] * np.quantile(dz, 0.125))), summ(np.exp(B[:, 1] + B[:, 4] * np.quantile(dz, 0.875)))]
    out[name] = r
json.dump(out, open('results/res_thornton_laplace.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
