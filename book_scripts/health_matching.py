"""Chapter 6 (The Inverse Weighting Lineage), section "Matching and Bayesian imputation: a care management example".

Synthetic claims file, n = 20,000, seed 7. Covariates: age ~ N(58, 10^2), HbA1c ~ N(7.8, 1.2^2),
comorbidities ~ Poisson(1.5), log prior cost = N(8.6, 0.9^2) + 0.15 comorb + 0.10 (HbA1c - 7.8).
Enrollment is logistic with logit -1.6 + 0.2 age_s + 0.5 hba1c_c + 0.35 comorb_c + 0.9 logprior_c.
Next year cost y0 = exp(8.8 + 0.06 age_s + 0.12 hba1c_c + 0.10 comorb_c + 0.65 logprior_c + 0.8 eps),
and y1 = y0 exp(-0.12). The fee is $400. Estimators: naive difference, 1:1 caliper matching on
the logit of the estimated propensity (caliper 0.2 sd, without replacement, greedy in decreasing
order of the logit, each enrolled member paired with the nearest unused control), and a Bayesian log
normal imputation model fitted to nonenrolled members.
Writes book_tex/chapters/health_numbers.tex (the hc macros). Run from the repository root.
All numbers are synthetic and illustrative.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(7)
n = 20000
PROGRAM_COST = 400.0  # dollars per enrolled member per year

age = rng.normal(58, 10, n)
hba1c = rng.normal(7.8, 1.2, n)
comorb = rng.poisson(1.5, n)
log_prior = rng.normal(8.6, 0.9, n) + 0.15 * comorb + 0.10 * (hba1c - 7.8)
X = np.column_stack([(age - 58) / 10, hba1c - 7.8, comorb - 1.5, log_prior - 8.8])

# enrollment: sicker and costlier members are targeted
lin = -1.6 + 0.2 * X[:, 0] + 0.5 * X[:, 1] + 0.35 * X[:, 2] + 0.9 * X[:, 3]
e_true = 1 / (1 + np.exp(-lin))
Z = rng.binomial(1, e_true)

# next year log cost: persistence of prior cost and a true 12 percent saving
mu0 = 8.8 + 0.06 * X[:, 0] + 0.12 * X[:, 1] + 0.10 * X[:, 2] + 0.65 * X[:, 3]
tau = -0.12
sd = 0.8
y0 = np.exp(mu0 + sd * rng.standard_normal(n))
y1 = y0 * np.exp(tau)
Y = np.where(Z == 1, y1, y0)

true_att = np.mean(y1[Z == 1] - y0[Z == 1])

# 1. naive comparison
naive = Y[Z == 1].mean() - Y[Z == 0].mean()

# 2. propensity score, 1:1 nearest neighbour matching on logit e with caliper
lr = LogisticRegression(C=1e6, max_iter=1000).fit(X, Z)
ehat = lr.predict_proba(X)[:, 1]
le = np.log(ehat / (1 - ehat))
cal = 0.2 * le.std()
t_idx = np.where(Z == 1)[0]
c_idx = np.where(Z == 0)[0]
order = np.argsort(le[c_idx])
c_sorted = c_idx[order]
le_c = le[c_sorted]
used = np.zeros(len(c_sorted), bool)
pairs = []
for i in t_idx[np.argsort(-le[t_idx])]:  # hardest cases first
    dist = np.abs(le_c - le[i])
    dist[used] = np.inf  # search every unused control, not a rank window
    best = int(np.argmin(dist))
    if dist[best] < cal:
        used[best] = True
        pairs.append((i, c_sorted[best]))
pairs = np.array(pairs)
d = Y[pairs[:, 0]] - Y[pairs[:, 1]]
match_att = d.mean()
match_se = d.std(ddof=1) / np.sqrt(len(d))
true_att_matched = np.mean(y1[pairs[:, 0]] - y0[pairs[:, 0]])


def smd(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


smd_before = [smd(X[Z == 1, j], X[Z == 0, j]) for j in range(4)]
smd_after = [smd(X[pairs[:, 0], j], X[pairs[:, 1], j]) for j in range(4)]

# 3. Bayesian imputation (Rubin 1978): log normal outcome model fitted to controls
# on the four covariates; impute y0 for all enrolled members.
D = np.column_stack([np.ones(n), X])
Dc, yc = D[Z == 0], np.log(Y[Z == 0])
V = np.linalg.inv(Dc.T @ Dc)
bhat = V @ Dc.T @ yc
res = yc - Dc @ bhat
nu = len(yc) - D.shape[1]
s2 = res @ res / nu
Dt = D[Z == 1]
draws = []
for _ in range(2000):
    sig2 = nu * s2 / rng.chisquare(nu)
    b = rng.multivariate_normal(bhat, sig2 * V)
    y0_imp = np.exp(Dt @ b + np.sqrt(sig2) * rng.standard_normal(len(Dt)))
    draws.append(np.mean(Y[Z == 1] - y0_imp))
draws = np.array(draws)
lo, hi = np.quantile(draws, [0.025, 0.975])
p_pays = np.mean(-draws > PROGRAM_COST)

n_t = int(Z.sum())
lines = {
    "nmem": f"{n:,}".replace(",", "{,}"),
    "ntreat": f"{n_t:,}".replace(",", "{,}"),
    "nmatch": f"{len(pairs):,}".replace(",", "{,}"),
    "meantreat": f"{Y[Z==1].mean():,.0f}".replace(",", "{,}"),
    "meancontrol": f"{Y[Z==0].mean():,.0f}".replace(",", "{,}"),
    "naive": f"{naive:,.0f}".replace(",", "{,}"),
    "match": f"{match_att:,.0f}".replace(",", "{,}"),
    "matchse": f"{match_se:,.0f}".replace(",", "{,}"),
    "bayes": f"{np.median(draws):,.0f}".replace(",", "{,}"),
    "bayeslo": f"{lo:,.0f}".replace(",", "{,}"),
    "bayeshi": f"{hi:,.0f}".replace(",", "{,}"),
    "truth": f"{true_att:,.0f}".replace(",", "{,}"),
    "truthm": f"{true_att_matched:,.0f}".replace(",", "{,}"),
    "netbayes": f"{-np.median(draws) - PROGRAM_COST:,.0f}".replace(",", "{,}"),
    "ppays": f"{p_pays:.2f}",
    "netmatch": f"{-match_att - PROGRAM_COST:,.0f}".replace(",", "{,}"),
    "netbook": f"{(-match_att - PROGRAM_COST) * len(pairs) / 1e6:.2f}",
    "smdpriorB": f"{smd_before[3]:.2f}", "smdpriorA": f"{smd_after[3]:.2f}",
    "smdhbaB": f"{smd_before[1]:.2f}", "smdhbaA": f"{smd_after[1]:.2f}",
    "smdcomB": f"{smd_before[2]:.2f}", "smdcomA": f"{smd_after[2]:.2f}",
    "smdageB": f"{smd_before[0]:.2f}", "smdageA": f"{smd_after[0]:.2f}",
}
with open("book_tex/chapters/health_numbers.tex", "w") as f:
    for k, v in lines.items():
        f.write(f"\\newcommand{{\\hc{k}}}{{{v}}}\n")
for k, v in lines.items():
    print(k, v)
