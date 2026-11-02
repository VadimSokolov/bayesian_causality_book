import numpy as np, pandas as pd, json, time, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from causaldata import nsw_mixtape, cps_mixtape
from stochtree import BARTModel, BCFModel
from sklearn.linear_model import LogisticRegressionCV, LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
FIG = "figs/"
t0 = time.time(); R = {}
np.random.seed(78)

def summ(x): return [float(np.mean(x)), float(np.std(x)), float(np.quantile(x, .025)), float(np.quantile(x, .975))]
cov = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]
def prep(df):
    X = df[cov].astype(float).copy()
    X["u74"] = (X.re74 == 0).astype(float); X["u75"] = (X.re75 == 0).astype(float)
    return X.values, df.treat.values.astype(float), df.re78.values.astype(float) / 1000
nsw = nsw_mixtape.load_pandas().data; cps = cps_mixtape.load_pandas().data
Xe, Te, Ye = prep(nsw)
obs = pd.concat([nsw[nsw.treat == 1], cps[cps.treat == 0]])
Xo, To, Yo = prep(obs)
R["n_exp"] = int(len(Ye)); R["n_obs"] = int(len(Yo)); R["n_treated"] = int(To.sum())
R["exp_diff"] = float(Ye[Te == 1].mean() - Ye[Te == 0].mean())
R["obs_naive"] = float(Yo[To == 1].mean() - Yo[To == 0].mean())
R["obs_ols"] = float(LinearRegression().fit(np.column_stack([To, Xo]), Yo).coef_[0])

def bart_att(X, T, Y, ps=None, mcmc=1000, burn=0, seed=7):
    Xa = np.column_stack([X, T]) if ps is None else np.column_stack([X, T, ps])
    m = BARTModel()
    m.sample(X_train=Xa, y_train=Y, num_gfr=20, num_burnin=burn, num_mcmc=mcmc,
             general_params={"random_seed": seed})
    tr = T == 1
    X1 = Xa[tr].copy(); X0 = Xa[tr].copy(); X0[:, X.shape[1]] = 0
    f1 = m.predict(X1, terms="y_hat") if False else m.predict(X1)
    f0 = m.predict(X0)
    f1 = f1["y_hat"] if isinstance(f1, dict) else f1
    f0 = f0["y_hat"] if isinstance(f0, dict) else f0
    # ATT: observed treated outcomes minus imputed counterfactual means (Hill 2011 conditional ATT)
    att = (Y[tr][:, None] - f0).mean(0)
    # Hill and Su style common support: counterfactual sd vs factual sd
    sd0 = f0.std(1); sd1 = f1.std(1)
    # r1023 F2: the conditional average effect on the treated, mean over the treated of f1 - f0 (the BCF target), saved beside att
    catt = (f1 - f0).mean(0)
    return att, f0, sd0, sd1, m, catt, f1

# Experimental benchmark with BART
att_e, _, _, _, _, catt_e, _ = bart_att(Xe, Te, Ye)
R["exp_bart_att"] = summ(att_e); R["exp_bart_catt"] = summ(catt_e); print("exp", R["exp_bart_att"], time.time() - t0, flush=True)
# Observational: propensity
psm = LogisticRegressionCV(cv=5, max_iter=3000).fit(Xo, To); ps = psm.predict_proba(Xo)[:, 1]
R["ps_treated_q"] = [float(q) for q in np.quantile(ps[To == 1], [0.05, 0.5, 0.95])]
R["ps_control_q"] = [float(q) for q in np.quantile(ps[To == 0], [0.5, 0.95, 0.99])]
att_b, f0, sd0, sd1, mb, catt_b, f1b = bart_att(Xo, To, Yo)
R["obs_bart_att"] = summ(att_b); R["obs_bart_catt"] = summ(catt_b); print("bart", R["obs_bart_att"], time.time() - t0, flush=True)
att_p, f0p, sd0p, _, _, catt_p, _ = bart_att(Xo, To, Yo, ps=ps)
R["obs_psbart_att"] = summ(att_p); R["obs_psbart_catt"] = summ(catt_p); print("psbart", R["obs_psbart_att"], time.time() - t0, flush=True)
# common support: discard treated units whose counterfactual posterior sd exceeds the largest factual sd among controls
mb_ctrl = mb.predict(np.column_stack([Xo[To == 0], To[To == 0]]))
mb_ctrl = mb_ctrl["y_hat"] if isinstance(mb_ctrl, dict) else mb_ctrl
cut = np.quantile(mb_ctrl.std(1), 0.99)
keep = sd0 <= cut
R["cs_kept"] = int(keep.sum())
att_cs = (Yo[To == 1][keep][:, None] - f0[keep]).mean(0)
R["obs_bart_att_cs"] = summ(att_cs); R["obs_bart_catt_cs"] = summ((f1b - f0)[keep].mean(0))
# BCF
bcf = BCFModel()
bcf.sample(X_train=Xo, Z_train=To, y_train=Yo, propensity_train=ps, num_gfr=20, num_mcmc=1000,
           general_params={"random_seed": 7})
tau = bcf.tau_hat_train
att_bcf = tau[To == 1].mean(0); R["obs_bcf_att"] = summ(att_bcf)
print("bcf", R["obs_bcf_att"], time.time() - t0, flush=True)
# DML ATT (AIPW for ATT) with random forests
n = len(Yo); kf = KFold(5, shuffle=True, random_state=1); mu0 = np.zeros(n); e = np.zeros(n)
for tr, te in kf.split(Xo):
    c = tr[To[tr] == 0]
    mu0[te] = RandomForestRegressor(300, min_samples_leaf=5, random_state=1).fit(Xo[c], Yo[c]).predict(Xo[te])
    e[te] = RandomForestClassifier(300, min_samples_leaf=5, random_state=1).fit(Xo[tr], To[tr]).predict_proba(Xo[te])[:, 1]
e = np.clip(e, 0.001, 0.999); p1 = To.mean()
psi = (To * (Yo - mu0) - (1 - To) * e / (1 - e) * (Yo - mu0)) / p1
att_dml = psi.mean()
# influence function of the ATT (Chernozhukov et al. 2018): psi minus T_i * att / p, so that its mean is zero; psi.std() would drop the T_i * att / p term
phi = psi - To * att_dml / p1; se = np.sqrt(np.mean(phi ** 2) / n)
R["obs_dml_att"] = [float(att_dml), float(se), float(att_dml - 1.96 * se), float(att_dml + 1.96 * se)]
R["dml_max_weight"] = float(np.max((e / (1 - e))[To == 0]))  # largest odds weight among the controls
print("dml", R["obs_dml_att"], time.time() - t0, flush=True)

# Figures
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.hist(ps[To == 0], bins=50, alpha=0.6, density=True, label="CPS controls", log=True)
ax.hist(ps[To == 1], bins=50, alpha=0.6, density=True, label="NSW treated", log=True)
ax.set_xlabel("Estimated propensity"); ax.set_ylabel("Density, log scale"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG + "lalonde_ps.pdf"); plt.close()

labels = ["naive", "least squares", "DML (AIPW)", "BART", "BART + e(x)", "BART, support", "BCF"]
vals = [[R["obs_naive"]] * 3, [R["obs_ols"]] * 3, R["obs_dml_att"][0:1] + R["obs_dml_att"][2:], R["obs_bart_att"][0:1] + R["obs_bart_att"][2:],
        R["obs_psbart_att"][0:1] + R["obs_psbart_att"][2:], R["obs_bart_att_cs"][0:1] + R["obs_bart_att_cs"][2:], R["obs_bcf_att"][0:1] + R["obs_bcf_att"][2:]]
fig, ax = plt.subplots(figsize=(5.4, 3.4))
for i, v in enumerate(vals):
    ax.errorbar(v[0], i, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", capsize=3, color="C0")
ax.axvspan(R["exp_bart_att"][2], R["exp_bart_att"][3], color="grey", alpha=0.2)
ax.axvline(R["exp_diff"], color="k", ls="--", lw=0.8)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.set_xlabel("Effect on the treated, $000 (1978 earnings)")
fig.tight_layout(); fig.savefig(FIG + "lalonde_att.pdf"); plt.close()

# individual effects for treated by re75
tau_tr = tau[To == 1]
fig, ax = plt.subplots(figsize=(5, 3.2))
x75 = Xo[To == 1, 7]
o = np.argsort(x75)
ax.errorbar(x75[o], tau_tr.mean(1)[o], yerr=1.96 * tau_tr.std(1)[o], fmt=".", alpha=0.5, ms=3)
ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("1975 earnings, dollars"); ax.set_ylabel("Posterior effect, $000")
fig.tight_layout(); fig.savefig(FIG + "lalonde_cate.pdf"); plt.close()

# 401(k) with BCF: tau(x) against income
d = pd.read_stata("data/sipp1991.dta")
xv = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]
Xk = d[xv].values.astype(float); Tk = d.e401.values.astype(float); Yk = d.net_tfa.values.astype(float) / 1000
pk = LogisticRegressionCV(cv=5, max_iter=3000).fit((Xk - Xk.mean(0)) / Xk.std(0), Tk).predict_proba((Xk - Xk.mean(0)) / Xk.std(0))[:, 1]
bk = BCFModel()
bk.sample(X_train=Xk, Z_train=Tk, y_train=Yk, propensity_train=pk, num_gfr=20, num_mcmc=500, general_params={"random_seed": 7})
tk = bk.tau_hat_train
R["k401_bcf_ate"] = summ(tk.mean(0)); R["k401_bcf_att"] = summ(tk[Tk == 1].mean(0)); R["k401_bcf_atu"] = summ(tk[Tk == 0].mean(0))
inc = d.inc.values / 1000
bins = np.quantile(inc, np.linspace(0, 1, 11)); mids = []; mm = []; lo = []; hi = []
for k in range(10):
    s = (inc >= bins[k]) & (inc <= bins[k + 1])
    draws = tk[s].mean(0); mids.append(np.median(inc[s])); mm.append(draws.mean()); lo.append(np.quantile(draws, .025)); hi.append(np.quantile(draws, .975))
R["k401_bcf_decile"] = {"mid": list(map(float, mids)), "mean": list(map(float, mm)), "lo": list(map(float, lo)), "hi": list(map(float, hi))}
print("401k bcf", R["k401_bcf_ate"], time.time() - t0, flush=True)
json.dump(R, open("results/res_bart.json", "w"), indent=1)
print("done", time.time() - t0)
