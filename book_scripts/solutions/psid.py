"""ex:chlalonde:1. LaLonde treated against the PSID controls (Dehejia and Wahba file data/psid_controls.dta).
Usage from the repository root: python book_scripts/solutions/psid.py bart SEED | dml PARTITION | agg
bart SEED fits BART, BART with the estimated propensity score and BCF (1,000 retained draws); dml PARTITION is a 5 fold random forest AIPW ATT with one partition seed;
agg collects the pieces into results/solutions/psid.json. The full run uses seeds 7, 11, 13, 17 and partitions 1 to 5."""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
t0 = time.time(); R = {}; np.random.seed(78)
C = "results/solutions/psid/"; os.makedirs(C, exist_ok=True)
NJ = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))
def summ(x): return [float(np.mean(x)), float(np.std(x)), float(np.quantile(x, .025)), float(np.quantile(x, .975))]
cov = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]
def prep(df):
    X = df[cov].astype(float).copy(); X["u74"] = (X.re74 == 0).astype(float); X["u75"] = (X.re75 == 0).astype(float)
    return X.values, df.treat.values.astype(float), df.re78.values.astype(float) / 1000
nsw = pd.read_stata("data/nsw_mixtape.dta")
ps_ = pd.read_stata("data/psid_controls.dta").rename(columns=dict(education="educ", hispanic="hisp", married="marr"))
obs = pd.concat([nsw[nsw.treat == 1], ps_[ps_.treat == 0]], ignore_index=True)
Xo, To, Yo = prep(obs); n = len(Yo); tr = To == 1
R["n"] = n; R["n_treated"] = int(To.sum()); R["naive"] = float(Yo[tr].mean() - Yo[~tr].mean())
Mo = np.column_stack([np.ones(n), To, Xo]); b = np.linalg.lstsq(Mo, Yo, rcond=None)[0]; e_ = Yo - Mo @ b
XtXi = np.linalg.inv(Mo.T @ Mo); V = XtXi @ (Mo.T * e_ ** 2) @ Mo @ XtXi * n / (n - Mo.shape[1]); R["ols"] = [float(b[1]), float(np.sqrt(V[1, 1]))]
R["ols_book_check"] = float(LinearRegression().fit(np.column_stack([To, Xo]), Yo).coef_[0])
psm = LogisticRegressionCV(cv=5, max_iter=3000).fit(Xo, To); ps = psm.predict_proba(Xo)[:, 1]
R["ps_treated_q"] = np.quantile(ps[tr], [.05, .5, .95]).round(3).tolist(); R["ps_control_q"] = np.quantile(ps[~tr], [.5, .95, .99]).round(3).tolist(); R["ctrl_below_min_treated"] = float((ps[~tr] < ps[tr].min()).mean()); R["ctrl_below_001"] = float((ps[~tr] < .001).mean())
def bart_att(X, T, Y, ps=None, mcmc=1000, seed=7):
    from stochtree import BARTModel
    Xa = np.column_stack([X, T]) if ps is None else np.column_stack([X, T, ps])
    m = BARTModel(); m.sample(X_train=Xa, y_train=Y, num_gfr=20, num_burnin=0, num_mcmc=mcmc, general_params={"random_seed": seed})
    t = T == 1; X1 = Xa[t].copy(); X0 = Xa[t].copy(); X0[:, X.shape[1]] = 0
    f1 = m.predict(X1); f0 = m.predict(X0); f1 = f1["y_hat"] if isinstance(f1, dict) else f1; f0 = f0["y_hat"] if isinstance(f0, dict) else f0
    return (Y[t][:, None] - f0).mean(0), f0, f0.std(1), m, (f1 - f0).mean(0)
mode = sys.argv[1] if len(sys.argv) > 1 else "agg"
if mode == "bart":
    seed = int(sys.argv[2]); k = f"s{seed}"
    a, f0, sd0, mb, catt = bart_att(Xo, To, Yo, seed=seed); r = dict(bart=summ(a), catt=summ(catt))
    c = mb.predict(np.column_stack([Xo[~tr], To[~tr]])); c = c["y_hat"] if isinstance(c, dict) else c
    cut = np.quantile(c.std(1), .99); keep = sd0 <= cut; r["kept"] = int(keep.sum()); r["cs"] = summ((Yo[tr][keep][:, None] - f0[keep]).mean(0)); json.dump(r, open(C + "bart" + k, "w")); print(k, r, time.time() - t0, flush=True)
    a, *_ = bart_att(Xo, To, Yo, ps=ps, seed=seed); json.dump(summ(a), open(C + "ps" + k, "w")); print("ps", k, time.time() - t0, flush=True)
    from stochtree import BCFModel
    b_ = BCFModel(); b_.sample(X_train=Xo, Z_train=To, y_train=Yo, propensity_train=ps, num_gfr=20, num_mcmc=1000, general_params={"random_seed": seed}); json.dump(summ(b_.tau_hat_train[tr].mean(0)), open(C + "bcf" + k, "w")); print("bcf", k, time.time() - t0, flush=True)
elif mode == "dml":
    sd = int(sys.argv[2])
    kf = KFold(5, shuffle=True, random_state=sd); mu0 = np.zeros(n); e = np.zeros(n)
    for a, te in kf.split(Xo):
        c = a[To[a] == 0]
        mu0[te] = RandomForestRegressor(300, min_samples_leaf=5, random_state=sd, n_jobs=NJ).fit(Xo[c], Yo[c]).predict(Xo[te])
        e[te] = RandomForestClassifier(300, min_samples_leaf=5, random_state=sd, n_jobs=NJ).fit(Xo[a], To[a]).predict_proba(Xo[te])[:, 1]
    e = np.clip(e, .001, .999); p1 = To.mean(); psi = (To * (Yo - mu0) - (1 - To) * e / (1 - e) * (Yo - mu0)) / p1; att = psi.mean(); phi = psi - To * att / p1; se = np.sqrt(np.mean(phi ** 2) / n)
    json.dump([float(att), float(se), float(att - 1.96 * se), float(att + 1.96 * se), float((e / (1 - e))[~tr].max())], open(C + f"dml{sd}", "w")); print("dml", sd, time.time() - t0, flush=True)
else:
    for f in sorted(os.listdir(C)): R[f] = json.load(open(C + f))
    json.dump(R, open("results/solutions/psid.json", "w"), indent=1); print(json.dumps(R, indent=1))
