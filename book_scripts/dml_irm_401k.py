# Chapter 10 and 16: DML for the 401(k) data with the interactive (augmented inverse propensity weighted) score, which estimates the unweighted
# average effect, the quantity the generative model averages; the partially linear DML rows of Table tab:k401 weight the effects by e(1 - e).
# Same data, basis, learners and fold partitions as app401k_book.py and dml_partitions.py. Lasso nuisances (on the 54 term basis): LassoCV for the two
# outcome regressions, L1 logistic regression for the propensity; forest nuisances (on the nine standardized covariates, as in the forest rows of the book): the forest of
# app401k_book.py for all three. Propensities are clipped to [0.01, 0.99]. Review loop F8 version: 100 partitions per learner (partition s uses seed s, so the
# first 20 lasso and 5 forest partitions are those of the earlier run), and for every partition also the quintile group effects of Table tab:k401q: with
# psi_i the AIPW score, the effect in income quintile j (pandas.qcut of inc, the quintiles of the table) is the mean of psi over the households in the quintile, which is the
# score restricted to the quintile and divided by its share, with standard error sd(psi in j)/sqrt(n_j); the same nuisances serve all groups. The ATT and ATU scores
# and the numbers of clipped propensities (overall and by quintile) are recorded as well. The book's median aggregation over partitions is done by the merge step.
# usage (repository root): python3 book_scripts/dml_irm_401k.py run DIR --learner lasso --start 0 --end 4 ; python3 book_scripts/dml_irm_401k.py merge DIR
# merge writes results/res_dml_irm_401k.json (the earlier 20 + 5 partition file is results/res_dml_irm_401k.json.before-r1023).
import argparse, glob, json, os, numpy as np
src = open("book_scripts/app401k_book.py").read()
exec(src.split('import os\nif os.path.exists("results/res401k.json")')[0])
qid = pd.qcut(d.inc, 5, labels=False).values

def irm(Z, learner, seed, clip=0.01):
    kf = KFold(5, shuffle=True, random_state=seed)
    m0 = np.zeros(n); m1 = np.zeros(n); e = np.zeros(n)
    for tr, te in kf.split(Z):
        i1, i0 = tr[D[tr] == 1], tr[D[tr] == 0]
        if learner == "rf":
            f = lambda: RandomForestRegressor(200, min_samples_leaf=5, max_features=4, random_state=seed)
            m1[te] = f().fit(Z[i1], Y[i1]).predict(Z[te]); m0[te] = f().fit(Z[i0], Y[i0]).predict(Z[te]); e[te] = f().fit(Z[tr], D[tr]).predict(Z[te])
        else:
            m1[te] = LassoCV(cv=5).fit(Z[i1], Y[i1]).predict(Z[te]); m0[te] = LassoCV(cv=5).fit(Z[i0], Y[i0]).predict(Z[te])
            e[te] = LogisticRegressionCV(Cs=10, cv=5, penalty="l1", solver="liblinear", random_state=seed).fit(Z[tr], D[tr]).predict_proba(Z[te])[:, 1]
    ec = np.clip(e, clip, 1 - clip); cl = (e < clip) | (e > 1 - clip)
    psi = m1 - m0 + D * (Y - m1) / ec - (1 - D) * (Y - m0) / (1 - ec)
    p1 = D.mean(); att = float(np.mean(D * (Y - m0) - (1 - D) * ec / (1 - ec) * (Y - m0)) / p1)
    phi_a = (D * (Y - m0 - att) - (1 - D) * ec / (1 - ec) * (Y - m0)) / p1
    return dict(theta=float(psi.mean()), se=float(psi.std() / np.sqrt(n)), clipped=int(cl.sum()), clipped_by_quintile=[int(cl[qid == j].sum()) for j in range(5)],
                max_weight=float(np.max(np.where(D == 1, 1 / ec, 1 / (1 - ec)))), att=att, att_se=float(np.sqrt(np.mean(phi_a ** 2) / n)),
                quintile=[dict(theta=float(psi[qid == j].mean()), se=float(psi[qid == j].std() / np.sqrt((qid == j).sum())), n=int((qid == j).sum())) for j in range(5)])

def tasks(learner, s0, s1):
    return [((P if learner == "lasso" else X), learner, s) for s in range(s0, s1)]

def agg(runs):
    th = np.array([r["theta"] for r in runs]); out = dict(partition1=runs[1] if len(runs) > 1 else None, median=float(np.median(th)), min=float(th.min()), max=float(th.max()),
                                                       median_se=float(np.median([r["se"] for r in runs])), partitions=len(runs))
    med = lambda v, se: (float(np.median(v)), float(np.median(np.sqrt(np.array(se) ** 2 + (np.array(v) - np.median(v)) ** 2))))
    out["ate_aggregated"] = med(th, [r["se"] for r in runs]); out["quintile_aggregated"] = []
    for j in range(5):
        v = [r["quintile"][j]["theta"] for r in runs]; se = [r["quintile"][j]["se"] for r in runs]; m, s = med(v, se)
        out["quintile_aggregated"].append(dict(theta=m, se=s, lo=m - 1.959964 * s, hi=m + 1.959964 * s, n=runs[0]["quintile"][j]["n"], min=float(min(v)), max=float(max(v))))
    out["clipped_max"] = max(r["clipped"] for r in runs); out["clipped_median"] = float(np.median([r["clipped"] for r in runs]))
    out["clipped_by_quintile_max"] = [max(r["clipped_by_quintile"][j] for r in runs) for j in range(5)]
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("dir"); ap.add_argument("--learner"); ap.add_argument("--start", type=int); ap.add_argument("--end", type=int); a = ap.parse_args()
    if a.cmd == "run":
        os.makedirs(a.dir, exist_ok=True)
        runs = [irm(*t) for t in tasks(a.learner, a.start, a.end)]
        json.dump(runs, open(os.path.join(a.dir, "%s_%03d_%03d.json" % (a.learner, a.start, a.end)), "w"))
    else:
        res = {}
        for k in ["lasso", "rf"]:
            runs = []
            for f in sorted(glob.glob(os.path.join(a.dir, k + "_*.json"))): runs += json.load(open(f))
            res[k] = runs; res[k + "_summary"] = agg(runs); print(k, json.dumps(res[k + "_summary"]), flush=True)
        json.dump(res, open("results/res_dml_irm_401k.json", "w"), indent=1)
