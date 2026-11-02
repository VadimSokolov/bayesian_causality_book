"""Numbers of Chapter 16 that app401k_book.py and bart_apps.py print or save only in part (book_tex/chapters/m11.tex).

401(k), 1991 SIPP, n = 9,915, the 54 term basis of app401k_book.py (nine standardized covariates, squares, pairwise products, each term standardized):
  ls_basis        least squares of net financial assets (thousands of dollars) on eligibility and the 54 terms;
  lasso_plug_in   LassoCV(cv=5) on eligibility and the 54 terms, eligibility penalized with the rest, the coefficient of eligibility;
  quintile        income quintiles by pandas.qcut(inc, 5): eligibility rate, share of the eligible and of the ineligible in each quintile, and the mean of
                  e(1 - e) with e the cross validated logistic propensity on the basis (LogisticRegressionCV(cv=5, Cs=10), as in app401k_book.py);
  weighted        the average of the five horseshoe quintile means of results/res401k.json with weights proportional to the mean of e(1 - e), and with equal weights.
LaLonde (NSW treated 185, NSW controls 260, CPS controls 15,992; the covariates of bart_apps.py with indicators of zero earnings in 1974 and 1975), thousands of 1978 dollars:
  exp             experimental difference in means, its Welch standard error and 95 percent half width;
  ols             least squares with the ten covariates on the NSW treated and the CPS controls, the coefficient of treatment, its heteroskedasticity robust
                  (HC1) standard error and 95 percent interval;
  dml_att         the DML estimate of the effect on the treated of bart_apps.py (random forests, five folds, seed 1) with the standard errors of the two parts of its
                  score: the treated term and the reweighted controls, and of the whole score.

usage (from the repository root): python book_scripts/k401_checks.py
writes results/res_k401_checks.json
"""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegressionCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
out = {}

# ---------------------------------------------------------------- 401(k)
d = pd.read_stata("data/sipp1991.dta")
xv = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]
Y = d.net_tfa.values.astype(float) / 1000.0
D = d.e401.values.astype(float)
X = StandardScaler().fit_transform(d[xv].values.astype(float))
P = PolynomialFeatures(2, include_bias=False).fit_transform(X)
_, keep = np.unique(np.round(P, 8), axis=1, return_index=True)
P = StandardScaler().fit_transform(P[:, np.sort(keep)])
out["n"] = int(len(Y)); out["p_basis"] = int(P.shape[1])
out["ls_basis"] = float(LinearRegression().fit(np.column_stack([D, P]), Y).coef_[0])
out["lasso_plug_in"] = float(LassoCV(cv=5).fit(np.column_stack([D, P]), Y).coef_[0])

prop = LogisticRegressionCV(cv=5, Cs=10, max_iter=2000).fit(P, D).predict_proba(P)[:, 1]
q = pd.qcut(d.inc, 5, labels=False).values
ee = prop * (1 - prop)
quint = dict(
    eligibility_rate=[float(D[q == k].mean()) for k in range(5)],
    share_of_eligible=[float(np.mean(q[D == 1] == k)) for k in range(5)],
    share_of_ineligible=[float(np.mean(q[D == 0] == k)) for k in range(5)],
    mean_e_1me=[float(ee[q == k].mean()) for k in range(5)])
res = json.load(open("results/res401k.json"))
means = np.array([s[0] for s in res["quintile"]])
w = np.array(quint["mean_e_1me"])
quint["weighted_average_e_1me"] = float(w @ means / w.sum())
quint["equal_weight_average"] = float(means.mean())
out["quintile"] = quint

# ---------------------------------------------------------------- LaLonde
cov = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]


def prep(df):
    Xd = df[cov].astype(float).copy()
    Xd["u74"] = (Xd.re74 == 0).astype(float); Xd["u75"] = (Xd.re75 == 0).astype(float)
    return Xd.values, df.treat.values.astype(float), df.re78.values.astype(float) / 1000


nsw = pd.read_stata("data/nsw_mixtape.dta"); cps = pd.read_stata("data/cps_mixtape.dta")
Xe, Te, Ye = prep(nsw)
d1, d0 = Ye[Te == 1], Ye[Te == 0]
se = float(np.sqrt(d1.var(ddof=1) / len(d1) + d0.var(ddof=1) / len(d0)))
out["exp"] = dict(diff=float(d1.mean() - d0.mean()), welch_se=se, half_width_95=1.96 * se, n_treated=int(len(d1)), n_control=int(len(d0)))

obs = pd.concat([nsw[nsw.treat == 1], cps[cps.treat == 0]])
Xo, To, Yo = prep(obs)
Zo = np.column_stack([To, np.ones(len(Yo)), Xo])
beta = np.linalg.lstsq(Zo, Yo, rcond=None)[0]
r = Yo - Zo @ beta
A = np.linalg.inv(Zo.T @ Zo)
V = A @ (Zo.T * r ** 2) @ Zo @ A * len(Yo) / (len(Yo) - Zo.shape[1])
out["ols"] = dict(estimate=float(beta[0]), hc1_se=float(np.sqrt(V[0, 0])),
                  lo=float(beta[0] - 1.96 * np.sqrt(V[0, 0])), hi=float(beta[0] + 1.96 * np.sqrt(V[0, 0])))

n = len(Yo); kf = KFold(5, shuffle=True, random_state=1); mu0 = np.zeros(n); e = np.zeros(n)
for tr, te in kf.split(Xo):
    c = tr[To[tr] == 0]
    mu0[te] = RandomForestRegressor(300, min_samples_leaf=5, random_state=1).fit(Xo[c], Yo[c]).predict(Xo[te])
    e[te] = RandomForestClassifier(300, min_samples_leaf=5, random_state=1).fit(Xo[tr], To[tr]).predict_proba(Xo[te])[:, 1]
e = np.clip(e, 0.001, 0.999); p1 = To.mean()
att = float((To * (Yo - mu0) / p1 - (1 - To) * e / (1 - e) * (Yo - mu0) / p1).mean())
# influence function of the ATT (Chernozhukov et al. 2018): the treated part is T (Y - mu0 - att) / p, the control part is unchanged
phi_t = To * (Yo - mu0 - att) / p1
psi_c = -(1 - To) * e / (1 - e) * (Yo - mu0) / p1
phi = phi_t + psi_c
out["dml_att"] = dict(estimate=att, se=float(np.sqrt(np.mean(phi ** 2) / n)), se_treated_term=float(phi_t.std() / np.sqrt(n)),
                      se_control_term=float(psi_c.std() / np.sqrt(n)), max_odds=float(np.max(e / (1 - e))))

with open("results/res_k401_checks.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
