"""Deep generative distributional analysis of 401(k) eligibility, with a weighted Bayesian bootstrap
and a high dimensional stress test using partial least squares.

Produces figs/deep401k_qte.pdf, figs/deep401k_cate.pdf, figs/deep401k_hd.pdf and results/res_deep401k.json.
The bands of the network fits are the mean of the refits plus or minus 1.96 standard deviations: the 2.5 and 97.5 percent points of a few
refits sit near the extremes and covered less in the coverage check of deep_401k_coverage.py (Chapter 10). The weighted quantile bands of
Firpo come from 200 draws and are their 2.5 and 97.5 percent points.
Each network input set (the nine covariates "base", the 209 inputs "raw", five partial least squares directions "pls") is refitted 200 times, each
refit under its own Bayesian bootstrap weight vector (exponential weights normalised to mean one, seed [11, label, b]) and its own training seed,
and all draws are saved, so that the summaries of the first 20 refits (the earlier version of the book) and of all 200 can be compared.
Usage, from the repository root (the refits are split into shards so that a SLURM array can run them):
  python book_scripts/deep_401k.py shard LABEL START COUNT   refits START .. START + COUNT - 1 of LABEL in {base, raw, pls}: results/deep401k_shards/
  python book_scripts/deep_401k.py optvar                    the optimisation variance: the first 20 base weight vectors, each refitted under 3 training seeds
  python book_scripts/deep_401k.py assemble                  the figures, results/res_deep401k.json and results/deep401k_draws.npz from the shards
  python book_scripts/deep_401k.py --figs-only               redraw deep401k_hd.pdf from the saved draws
Requires numpy, pandas, scipy, scikit-learn, matplotlib and torch. About 4 to 15 seconds per refit on two CPU cores.
"""
import glob, json, os, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression

PAL = ["#1f5a96", "#d1495b", "#edae49", "#00798c", "#66a182", "#8d5fa8"]
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "pdf.fonttype": 42, "axes.prop_cycle": plt.cycler(color=PAL)})
OUT = "figs/"; os.makedirs("results", exist_ok=True)
def band(a):
    """The band printed for the refits of a network: mean plus or minus 1.96 standard deviations over the refits (axis 0)."""
    m = a.mean(0); s = a.std(0, ddof=1); return m - 1.96 * s, m + 1.96 * s
def plot_hd(QS, qte, qte_raw, qte_pls):
    """deep401k_hd.pdf. Gray scale print: the 9 covariate fit is a solid black line on a light gray band; the raw network is a dashed dark gray line and
    the PLS fit a dash-dot mid gray line, and their 95 percent bands are drawn as thin lines in the same style (overlapping fills cannot be read in gray)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for arr, col, ls, lab in [(qte, "k", "-", "9 covariates"), (qte_raw, "0.3", "--", "209 covariates, raw network"),
                              (qte_pls, "0.55", "-.", "209 covariates, 5 PLS directions")]:
        lo, hi = band(arr)
        if ls == "-": ax.fill_between(QS, lo, hi, color="0.85", lw=0)
        else: ax.plot(QS, lo, ls, color=col, lw=1.0); ax.plot(QS, hi, ls, color=col, lw=1.0)
        ax.plot(QS, arr.mean(0), ls, color=col, lw=2, zorder=3 if ls == "-" else 2, label=lab)
    ax.axhline(0, color="grey", lw=0.6); ax.set_xlabel("quantile level"); ax.set_ylabel("effect, thousands of dollars")
    ax.legend(frameon=False, fontsize=8, loc="upper left", handlelength=3.2)
    fig.tight_layout(); fig.savefig(OUT + "deep401k_hd.pdf"); plt.close()


import sys
if "--figs-only" in sys.argv:    # redraw deep401k_hd.pdf from the saved draws: no network is refit and nothing is written to results/
    sv = np.load("results/deep401k_draws.npz"); plot_hd(sv["QS"], sv["qte"], sv["qte_raw"], sv["qte_pls"]); sys.exit(0)

class QNN(nn.Module):
    def __init__(self, dh, width=64, K=32):
        super().__init__()
        self.register_buffer("k", torch.arange(1, K + 1).float() * np.pi)
        self.tau_net = nn.Sequential(nn.Linear(K, width), nn.ReLU())
        self.h_net = nn.Sequential(nn.Linear(dh, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
    def forward(self, tau, h):
        return self.head(self.tau_net(torch.cos(tau[:, None] * self.k)) * self.h_net(h)).squeeze(-1)


def fit(H, y, w, epochs=int(os.environ.get("DEEP_EPOCHS", 40)), batch=512, ntau=4, lr=2e-3, wd=1e-4):
    """Weighted pinball loss: the weights w are a Bayesian bootstrap draw."""
    H = torch.tensor(H, dtype=torch.float32); y = torch.tensor(y, dtype=torch.float32); w = torch.tensor(w, dtype=torch.float32)
    net = QNN(H.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), batch):
            idx = perm[i:i + batch]
            yy, hh, ww = y[idx].repeat(ntau), H[idx].repeat(ntau, 1), w[idx].repeat(ntau)
            tau = torch.rand(len(yy)); u = yy - net(tau, hh)
            loss = torch.sum(ww * torch.maximum(tau * u, (tau - 1) * u)) / torch.sum(ww)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return net


def potential(net, H0, H1, M=10):
    """Draws of Y(0) and Y(1) for every unit, and unit level mean effects on a tau grid."""
    with torch.no_grad():
        A0 = torch.tensor(H0, dtype=torch.float32); A1 = torch.tensor(H1, dtype=torch.float32)
        U = torch.rand(len(H0) * M)
        y0 = net(U, A0.repeat(M, 1)).numpy() * ys; y1 = net(U, A1.repeat(M, 1)).numpy() * ys
        g = torch.linspace(0.005, 0.995, 100)
        eff = np.zeros(len(H0))
        for tq in g:
            tt = torch.full((len(H0),), float(tq))
            eff += (net(tt, A1) - net(tt, A0)).numpy() * ys / len(g)
    return y0, y1, eff


t0 = time.time(); torch.set_num_threads(2)
SH = "results/deep401k_shards/"; os.makedirs(SH, exist_ok=True)
d = pd.read_stata("data/sipp1991.dta")
cols = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]
X = d[cols].to_numpy(float); T = d["e401"].to_numpy(float); Y = d["net_tfa"].to_numpy(float) / 1000
n = len(Y); Xs = (X - X.mean(0)) / X.std(0); ys = Y.std(); Ysc = Y / ys
QS = np.linspace(0.05, 0.95, 19)
inc_q = pd.qcut(d["inc"], 5, labels=False).to_numpy()
age_g = pd.cut(d["age"], [24, 35, 45, 55, 65], labels=False).to_numpy()
LAB = {"base": 0, "raw": 1, "pls": 2}
# the 200 noise covariates are those of the earlier version, whose generator was advanced past 20 weight vectors and 200 Firpo draws
rng0 = np.random.default_rng(11)
for _ in range(20): rng0.exponential(size=n)
rng_ipw = np.random.default_rng(11); [rng_ipw.exponential(size=n) for _ in range(20)]      # same stream: the Firpo draws are those of the earlier version
for _ in range(200): rng0.exponential(size=n)
Znoise = rng0.normal(size=(n, 200)); Z = np.column_stack([Xs, Znoise])
pls = PLSRegression(n_components=5, scale=True).fit(Z, np.column_stack([Ysc, T]))
S = pls.transform(Z); w2 = pls.x_weights_ ** 2; share_real = float(w2[:9].sum() / w2.sum()); Sz = (S - S.mean(0)) / S.std(0)
INPUTS = {"base": Xs, "raw": Z, "pls": Sz}
def weights(label, b):
    w = np.random.default_rng([11, LAB[label], b]).exponential(size=n); return w / w.mean()


def one_fit(Hx, w, seed):
    """One refit under weights w and training seed; returns the quantile effects, the average effect, the household effects."""
    H0 = np.column_stack([Hx, np.zeros(n)]); H1 = np.column_stack([Hx, np.ones(n)]); H = np.column_stack([Hx, T])
    torch.manual_seed(seed); net = fit(H, Ysc, w); y0, y1, eff = potential(net, H0, H1)
    return np.quantile(y1, QS) - np.quantile(y0, QS), float(np.mean(eff)), eff


def shard(label, start, count):
    out = dict(qte=[], ate=[], inc=[], age=[])
    for b in range(start, start + count):
        q, a, eff = one_fit(INPUTS[label], weights(label, b), 100000 * (LAB[label] + 1) + b)
        out["qte"].append(q); out["ate"].append(a); out["inc"].append([eff[inc_q == k].mean() for k in range(5)]); out["age"].append([eff[age_g == k].mean() for k in range(4)])
        print(label, b, round(a, 2), round(time.time() - t0), flush=True)
    np.savez(SH + "%s_%04d.npz" % (label, start), **{k: np.array(v) for k, v in out.items()})


def optvar():
    """Optimisation variance: 20 fixed weight vectors (the first 20 of the base fit), each refitted under 3 training seeds."""
    ate = np.zeros((20, 3)); qte = np.zeros((20, 3, 19))
    for b in range(20):
        for s_ in range(3):
            q, a, _ = one_fit(Xs, weights("base", b), 900000 + 100 * b + s_); ate[b, s_] = a; qte[b, s_] = q
        print("optvar", b, ate[b].round(2), round(time.time() - t0), flush=True)
    np.savez(SH + "optvar.npz", ate=ate, qte=qte)


def load(label):
    fs = sorted(glob.glob(SH + label + "_[0-9]*.npz")); P = [np.load(f) for f in fs]
    return {k: np.concatenate([p[k] for p in P]) for k in ["qte", "ate", "inc", "age"]}


def wq(y, w, q):
    o = np.argsort(y); c = np.cumsum(w[o]) / w.sum(); return np.interp(q, c, y[o])


def assemble():
    D = {l: load(l) for l in LAB}; Bn = len(D["base"]["ate"]); R = {}
    qte, ate = D["base"]["qte"], D["base"]["ate"]
    e = LogisticRegression(max_iter=2000).fit(Xs, T).predict_proba(Xs)[:, 1]; ipw = []
    for b in range(200):
        w = rng_ipw.exponential(size=n)
        ipw.append(wq(Y[T == 1], (w / e)[T == 1], QS) - wq(Y[T == 0], (w / (1 - e))[T == 0], QS))
    ipw = np.array(ipw)
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for arr, col, lab in [(ipw, PAL[2], "weighted quantiles (Firpo)"), (qte, PAL[0], "deep generative model, Bayesian bootstrap")]:
        lo, hi = band(arr) if arr is qte else np.quantile(arr, [0.025, 0.975], axis=0)
        ax.fill_between(QS, lo, hi, color=col, alpha=0.25, lw=0); ax.plot(QS, arr.mean(0), color=col, lw=2, label=lab)
    ax.axhline(0, color="grey", lw=0.6); ax.set_xlabel("quantile level"); ax.set_ylabel("effect, thousands of dollars")
    ax.legend(frameon=False, fontsize=8, loc="upper left"); fig.tight_layout(); fig.savefig(OUT + "deep401k_qte.pdf"); plt.close()
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for a, G, labs, col, ttl in [(axs[0], D["base"]["inc"], ["1", "2", "3", "4", "5"], PAL[3], "by income quintile"),
                                 (axs[1], D["base"]["age"], ["25 to 35", "36 to 45", "46 to 55", "56 to 64"], PAL[1], "by age")]:
        lo, hi = band(G); m = G.mean(0)
        a.bar(range(len(labs)), m, color=col, alpha=0.85); a.errorbar(range(len(labs)), m, yerr=[m - lo, hi - m], fmt="none", ecolor="k", capsize=3, lw=1)
        a.set_xticks(range(len(labs))); a.set_xticklabels(labs, fontsize=8); a.set_title(ttl, fontsize=10)
    axs[0].set_ylabel("mean effect, thousands of dollars"); fig.tight_layout(); fig.savefig(OUT + "deep401k_cate.pdf"); plt.close()
    plot_hd(QS, qte, D["raw"]["qte"], D["pls"]["qte"])
    def summ(m):
        """Summaries of the first m refits (m = None: all)."""
        sl = slice(m); g = D["base"]; q = g["qte"][sl]; a = g["ate"][sl]; I = g["inc"][sl]; A = g["age"][sl]; idx = [np.argmin(abs(QS - v)) for v in [0.1, 0.25, 0.5, 0.75, 0.9]]
        def rms(x): return float(np.sqrt(np.mean((x.mean(0) - q.mean(0)) ** 2)))
        def width(x): return float(np.mean(2 * 1.96 * x.std(0, ddof=1)))
        r_, p_ = D["raw"], D["pls"]
        return dict(B=len(a), ate_mean=float(a.mean()), ate_sd=float(a.std(ddof=1)), ate_ci=[float(x) for x in band(a)], qte_q=[0.1, 0.25, 0.5, 0.75, 0.9],
                    qte_deep=[float(q.mean(0)[i]) for i in idx], qte_ipw=[float(ipw.mean(0)[i]) for i in idx],
                    inc_quintile=[float(v) for v in I.mean(0)], inc_quintile_ci=[[float(v) for v in band(I[:, k])] for k in range(5)],
                    age=[float(v) for v in A.mean(0)], age_ci=[[float(v) for v in band(A[:, k])] for k in range(4)], qte_sd_mean=float(q.std(0, ddof=1).mean()),
                    hd=dict(p_noise=200, ate_raw=float(r_["ate"][sl].mean()), ate_raw_ci=[float(x) for x in band(r_["ate"][sl])],
                            ate_pls=float(p_["ate"][sl].mean()), ate_pls_ci=[float(x) for x in band(p_["ate"][sl])], rms_raw=rms(r_["qte"][sl]), rms_pls=rms(p_["qte"][sl]),
                            width_base=width(q), width_raw=width(r_["qte"][sl]), width_pls=width(p_["qte"][sl]), pls_weight_share_real=share_real,
                            pls_q95_range=[float(p_["qte"][sl][:, 18].min()), float(p_["qte"][sl][:, 18].max())]))
    R["base"] = summ(None); R["hd"] = R["base"]["hd"]; R["first20"] = summ(20); R["refits"] = Bn
    np.savez("results/deep401k_draws.npz", QS=QS, qte=qte, ate=ate, ipw=ipw, qte_raw=D["raw"]["qte"], qte_pls=D["pls"]["qte"], ate_raw=D["raw"]["ate"], ate_pls=D["pls"]["ate"],
             inc_effects=D["base"]["inc"], age_effects=D["base"]["age"])
    if os.path.exists(SH + "optvar.npz"):
        o = np.load(SH + "optvar.npz"); a = o["ate"]; q = o["qte"]; m_ = a.shape[1]
        def share(x):
            """One way random effects: the optimisation variance (within weight vector) as a share of the sum of it and the variance across weight vectors."""
            msw = x.var(1, ddof=1).mean(0); msb = m_ * x.mean(1).var(0, ddof=1); sw2 = np.maximum((msb - msw) / m_, 0); return msw, sw2, msw / (msw + sw2)
        msw, sw2, sh = share(a); mq, sq, shq = share(q)
        R["optvar"] = dict(weight_vectors=int(a.shape[0]), seeds=int(m_), ate_opt_var=float(msw), ate_weight_var=float(sw2), ate_opt_share=float(sh), ate_opt_sd=float(np.sqrt(msw)), ate_weight_sd=float(np.sqrt(sw2)),
                           ate_seed_means=[float(v) for v in a.mean(0)], qte_opt_share_mean=float(shq.mean()), qte_opt_share_min=float(shq.min()), qte_opt_share_max=float(shq.max()),
                           ate_seed0_sd_across_weights=float(a[:, 0].std(ddof=1)), ate_sd_of_200_refits=float(ate.std(ddof=1)))
    R["secs"] = round(time.time() - t0); json.dump(R, open("results/res_deep401k.json", "w"), indent=1); print(json.dumps(R, indent=1))


if sys.argv[1] == "shard": shard(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
elif sys.argv[1] == "optvar": optvar()
elif sys.argv[1] == "assemble": assemble()
