"""ex:chcrit:4. Extra random-state partitions of the 401(k) lasso DML estimate against the coupled horseshoe posterior of book_scripts/app401k_book.py.
Usage from the repository root: python book_scripts/solutions/g2_401k.py head | states A B | agg
head fits the coupled horseshoe (saves its quantiles); states A B runs the lasso DML partitions with random states A to B-1 into results/solutions/g2_401k/ (states 0 to 19 are those saved in results/res_dml_partitions.json);
agg compares all partitions with the posterior (the full run uses states 20 to 199, 200 in total)."""
import os, sys, json, time, numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); sys.path.insert(0, os.getcwd())
OUT = "results/solutions/g2_401k"; os.makedirs(OUT, exist_ok=True)
mode = sys.argv[1]
src = open("book_scripts/app401k_book.py").read()
exec(src.split('import os\nif os.path.exists("results/res401k.json")')[0])
exec('def hs(' + src.split('def hs(')[1].split('def summ')[0])
if mode == "head":
    one = np.ones(n); Z1 = np.column_stack([D, one, P]); dr1 = hs(Z1, Y, {0, 1})
    prop = LogisticRegressionCV(cv=5, Cs=10, max_iter=2000).fit(P, D).predict_proba(P)[:, 1]
    Z2 = np.column_stack([D, one, prop, P]); dr2 = hs(Z2, Y, {0, 1, 2}); t = dr2[:, 0]
    q = np.quantile(t, [.025, .25, .5, .75, .975]); print('coupled mean sd', t.mean(), t.std(), 'q', q, flush=True)
    json.dump(dict(mean=float(t.mean()), sd=float(t.std()), q=q.tolist()), open(OUT + "/head.json", "w"))
elif mode == "states":
    for s in range(int(sys.argv[2]), int(sys.argv[3])):
        fn = f"{OUT}/s{s}.json"
        if not os.path.exists(fn): json.dump(float(dml(P, 'lasso', s)[0]), open(fn, "w")); print("state", s, flush=True)
else:
    from scipy.stats import norm
    h = json.load(open(OUT + "/head.json")); q = h["q"]; lo, hi = q[1], q[3]
    res = json.load(open('results/res_dml_partitions.json'))['lasso']; th = np.array([a for a, b in res])
    print('saved outside', np.mean((th < lo) | (th > hi)), th.min(), th.max(), th.mean(), th.std(ddof=1))
    ss = sorted(int(f[1:-5]) for f in os.listdir(OUT) if f.startswith("s")); ex = np.array([json.load(open(f"{OUT}/s{s}.json")) for s in ss]); allt = np.r_[th, ex]
    out = dict(coupled=h, n_saved=len(th), n_extra=len(ex), extra_outside=int(((ex < lo) | (ex > hi)).sum()), all_n=len(allt), all_min=float(allt.min()), all_max=float(allt.max()), all_mean=float(allt.mean()), all_sd=float(allt.std(ddof=1)), all_median=float(np.median(allt)), saved_median=float(np.median(th)),
               upper95=float(1 - .05 ** (1 / len(allt))), upper95_20=float(1 - .05 ** (1 / 20)), normal_tail=float(norm.cdf((lo - allt.mean()) / allt.std(ddof=1)) + norm.sf((hi - allt.mean()) / allt.std(ddof=1))),
               all_outside=int(((allt < lo) | (allt > hi)).sum()), extra_min=float(ex.min()), extra_max=float(ex.max()))
    print(json.dumps(out)); json.dump(out, open("results/solutions/g2_401k.json", "w"), indent=1)
