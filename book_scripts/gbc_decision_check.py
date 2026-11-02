"""A finite check of the distorted value computed from a trained quantile network, on a conjugate two arm trial (review item F11, Chapter 19).

Design. Two arms with p_0, p_1 ~ Beta(1,1) independent, K_w ~ Binomial(n, p_w) given p_w, n in {25, 100, 400} per arm. The exact posterior
is p_w | D ~ Beta(K_w + 1, n - K_w + 1) independently, and the target is tau = p_1 - p_0. The decision is to treat or not: the value of
treating is V_eta(1) = CVaR_eta(tau | D) - c, of not treating V_eta(0) = 0, for c in {0, 0.05, 0.2} and eta in {0.05, 0.1, 0.25, 1}, where
CVaR_eta is the mean of the lowest fraction eta of the posterior of tau (eta = 1 is the posterior mean). The planner treats when V_eta(1) > 0.
Because the prior is uniform, the prior predictive distribution of (K_0, K_1) is uniform over the (n+1)^2 pairs, so a fresh trial is a uniform pair.

Network. The quantile network of Section comp:sec:network (the one of qnn_book.py and gbc_ate.py: cosine embedding of the level with K = 32,
product with an MLP embedding of the inputs, width 64, MLP head), trained with the check loss on 300,000 prior predictive trials, 100,000 for
each n (balanced), inputs (n, K_0, K_1) standardized by their training standard deviations, target tau divided by its training standard
deviation, 40 epochs, batch 512, 4 levels per row, Adam with learning rate 0.002 and cosine decay. Five training seeds s = 0..4; seed s draws
its own training set (data seed 1000 + s) and initial weights (torch seed s). The network is not the gbc package's IQN.

Evaluation. The network quantile function is evaluated on 1,000 levels (j + 1/2)/1000 and rearranged (sorted), as GBC.md section 12 advises;
the network CVaR_eta is the mean of the rearranged quantiles at the levels below eta (the mean of all of them for eta = 1). Before sorting we
record the crossing dip: the largest decrease of the raw quantile from one level to the next, relative to the exact posterior standard deviation.
Exact reference: CVaR_eta = max_t {t - eta^{-1} int^t F(s) ds} with F the distribution function of tau, F(t) = E_{p_0} F_1(p_0 + t), computed
by the midpoint rule over 600 equal probability nodes of the posterior of p_0 and 1,201 values of t spread over the posterior mean +- 7 sd (clipped to
[-1, 1]); the mean (eta = 1) is exact. The numerical tolerance is the largest difference to a run with 2,400 nodes and 4,801 values of t on
200 random trials, stored under reference_tolerance.
Test sets, for each n: 10,000 ordinary trials (uniform pairs, seed 7) and, for each (c, eta), 1,000 boundary trials, uniform among the pairs with
|V_eta(1)| <= 0.02 under the exact reference (all of them when fewer than 1,000 pairs qualify; candidates are screened by a normal approximation and
checked with the exact reference). Reported, by n, c, eta and set: bias and RMSE of the network CVaR (c cancels, so by eta), action disagreement with
the exact rule, and the mean reference regret (the exact value lost by following the network's action, max(V, 0) - V 1{network treats}), each
averaged over trials within a seed, then mean and sd across the five seeds. Also training and evaluation seconds.
Stages (run from the repository root; PYTHONPATH not needed):
  python book_scripts/gbc_decision_check.py ref            exact reference, writes results/gbc_decision_ref.npz (use a Pool over the cores)
  python book_scripts/gbc_decision_check.py train SEED     trains and evaluates one seed, writes results/gbc_decision_seed{SEED}.json
  python book_scripts/gbc_decision_check.py agg            results/res_gbc_decision_check.json and figs/fig_gbc_decision_check.pdf
Add --quick to any stage for a code path test (small sizes, writes to results/quick_*)."""
import sys, os, json, time, numpy as np
from scipy import special
from multiprocessing import Pool

NS = (25, 100, 400)
CS = (0.0, 0.05, 0.2)
ETAS = (0.05, 0.1, 0.25, 1.0)
QUICK = "--quick" in sys.argv
args = [a for a in sys.argv[1:] if a != "--quick"]
NORD, NBND, NTRAIN, EPOCHS, NLEV = (300, 50, 3000, 2, 1000) if QUICK else (10000, 1000, 100000, 40, 1000)
PFX = "results/quick_" if QUICK else "results/"
BAND = 0.02


# ------------------------------------------------------------------ exact reference
def cvar_exact(n, k0, k1, m=600, nt=1201, etas=ETAS):
    """CVaR_eta of tau = p1 - p0, p_w ~ Beta(K_w + 1, n - K_w + 1), for each eta, by the midpoint rule described in the docstring."""
    a0, b0, a1, b1 = k0 + 1.0, n - k0 + 1.0, k1 + 1.0, n - k1 + 1.0
    mean = a1 / (a1 + b1) - a0 / (a0 + b0)
    var = a1 * b1 / ((a1 + b1) ** 2 * (a1 + b1 + 1)) + a0 * b0 / ((a0 + b0) ** 2 * (a0 + b0 + 1))
    sd = np.sqrt(var)
    x = special.betaincinv(a0, b0, (np.arange(m) + 0.5) / m)                 # equal probability nodes of p0
    t = np.linspace(max(-1.0, mean - 7 * sd), min(1.0, mean + 7 * sd), nt)
    F = special.betainc(a1, b1, np.clip(x[None, :] + t[:, None], 0.0, 1.0)).mean(1)
    G = np.concatenate([[0.0], np.cumsum(0.5 * (F[1:] + F[:-1]) * np.diff(t))])
    out = []
    for e in etas:
        out.append(mean if e == 1.0 else float(np.max(t - G / e)))
    return np.array(out), float(mean), float(sd)


def ref_one(a):
    return cvar_exact(*a)


def approx_cvar(n, K0, K1):
    """Normal approximation of CVaR_eta for the screening of boundary candidates (arrays K0, K1)."""
    a0, b0, a1, b1 = K0 + 1.0, n - K0 + 1.0, K1 + 1.0, n - K1 + 1.0
    mean = a1 / (a1 + b1) - a0 / (a0 + b0)
    sd = np.sqrt(a1 * b1 / ((a1 + b1) ** 2 * (a1 + b1 + 1)) + a0 * b0 / ((a0 + b0) ** 2 * (a0 + b0 + 1)))
    from scipy import stats
    return np.stack([mean if e == 1.0 else mean - sd * stats.norm.pdf(stats.norm.ppf(e)) / e for e in ETAS], 1)


def stage_ref():
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    rng = np.random.default_rng(7)
    out = {}; t0 = time.time()
    with Pool(ncpu) as pool:
        for n in NS:
            ordinary = rng.integers(0, n + 1, size=(NORD, 2))
            allp = np.array([(i, j) for i in range(n + 1) for j in range(n + 1)])
            ap = approx_cvar(n, allp[:, 0].astype(float), allp[:, 1].astype(float))
            cand = {}
            for ie, e in enumerate(ETAS):
                for ic, c in enumerate(CS):
                    idx = np.where(np.abs(ap[:, ie] - c) <= BAND + 0.03)[0]
                    cand[(ie, ic)] = rng.permutation(idx)[:int(NBND * 2.2) + 50]
            pairs = {tuple(p) for p in ordinary}
            for v in cand.values():
                pairs |= {tuple(allp[i]) for i in v}
            pairs = sorted(pairs)
            res = pool.map(ref_one, [(n, p[0], p[1]) for p in pairs], chunksize=50)
            ref = {p: r for p, r in zip(pairs, res)}
            # boundary sets
            bnd = {}
            for (ie, ic), v in cand.items():
                good = [tuple(allp[i]) for i in v if abs(ref[tuple(allp[i])][0][ie] - CS[ic]) <= BAND]
                bnd[(ie, ic)] = good[:NBND]
            tol = None
            if n == NS[0]:
                tst = [(n,) + tuple(p) for p in [pairs[i] for i in rng.integers(0, len(pairs), 200)]]
                hi = pool.map(ref_one_hi, tst)
                tol = float(max(np.abs(h[0] - ref[(t[1], t[2])][0]).max() for h, t in zip(hi, tst)))
            tolhi = []
            if n != NS[0]:
                tst = [(n,) + tuple(p) for p in [pairs[i] for i in rng.integers(0, len(pairs), 200)]]
                hi = pool.map(ref_one_hi, tst)
                tolhi = float(max(np.abs(h[0] - ref[(t[1], t[2])][0]).max() for h, t in zip(hi, tst)))
            out[n] = dict(pairs=np.array(pairs), cvar=np.array([ref[p][0] for p in pairs]), mean=np.array([ref[p][1] for p in pairs]),
                          sd=np.array([ref[p][2] for p in pairs]), ordinary=ordinary,
                          bnd={f"{ie}_{ic}": np.array(g) for (ie, ic), g in bnd.items()}, tol=tol if tol is not None else tolhi)
            print(n, len(pairs), "pairs", f"{time.time() - t0:.0f}s", "tol", out[n]["tol"],
                  "boundary sizes", {k: len(v) for k, v in out[n]["bnd"].items()}, flush=True)
    flat = {}
    for n, d in out.items():
        for k, v in d.items():
            if k == "bnd":
                for kk, vv in v.items():
                    flat[f"n{n}_bnd_{kk}"] = vv
            else:
                flat[f"n{n}_{k}"] = np.asarray(v)
    np.savez(PFX + "gbc_decision_ref.npz", **flat)


def ref_one_hi(a):
    return cvar_exact(*a, m=2400, nt=4801)


# ------------------------------------------------------------------ network
def stage_train(seed):
    import torch, torch.nn as nn
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    R = np.load(PFX + "gbc_decision_ref.npz")
    rng = np.random.default_rng(1000 + seed)
    X, Y = [], []
    for n in NS:
        p0, p1 = rng.uniform(size=NTRAIN), rng.uniform(size=NTRAIN)
        k0, k1 = rng.binomial(n, p0), rng.binomial(n, p1)
        X.append(np.column_stack([np.full(NTRAIN, n), k0, k1])); Y.append(p1 - p0)
    X = np.vstack(X).astype(np.float32); Y = np.concatenate(Y).astype(np.float32)
    xm, xs, ym, ys = X.mean(0), X.std(0), float(Y.mean()), float(Y.std())

    class QNN(nn.Module):
        def __init__(self, dh, width=64, K=32):
            super().__init__()
            self.register_buffer("k", torch.arange(1, K + 1).float() * np.pi)
            self.tau_net = nn.Sequential(nn.Linear(K, width), nn.ReLU())
            self.h_net = nn.Sequential(nn.Linear(dh, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

        def forward(self, tau, h):
            return self.head(self.tau_net(torch.cos(tau[:, None] * self.k)) * self.h_net(h)).squeeze(-1)

    torch.manual_seed(seed)
    net = QNN(3); opt = torch.optim.Adam(net.parameters(), lr=2e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    th = torch.tensor((Y - ym) / ys); H = torch.tensor((X - xm) / xs); m = len(th); t0 = time.time(); hist = []
    for ep in range(EPOCHS):
        perm = torch.randperm(m); tot = 0.0
        for i in range(0, m, 512):
            idx = perm[i:i + 512]
            t = th[idx].repeat(4); hb = H[idx].repeat(4, 1); tau = torch.rand(len(t))
            u = t - net(tau, hb)
            loss = torch.mean(torch.maximum(tau * u, (tau - 1) * u))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step(); hist.append(tot / m * ys)
    train_secs = time.time() - t0
    lev = (np.arange(NLEV) + 0.5) / NLEV; lv = torch.tensor(lev, dtype=torch.float32)
    res = dict(seed=seed, train_secs=train_secs, final_train_loss=hist[-1], epochs=EPOCHS, n_train=int(len(th)), by_n={})
    ev_secs = 0.0
    for n in NS:
        pairs = R[f"n{n}_pairs"]; cv = R[f"n{n}_cvar"]; sd = R[f"n{n}_sd"]; key = {tuple(p): i for i, p in enumerate(pairs)}
        t1 = time.time()
        Xn = np.column_stack([np.full(len(pairs), n), pairs]).astype(np.float32)
        Q = np.empty((len(pairs), NLEV), dtype=np.float32)
        with torch.no_grad():
            e = net.tau_net(torch.cos(lv[:, None] * net.k))
            for i in range(0, len(pairs), 500):
                h = net.h_net(torch.tensor((Xn[i:i + 500] - xm) / xs))
                Q[i:i + 500] = net.head(e[None] * h[:, None]).squeeze(-1).numpy() * ys + ym
        dips = np.maximum(Q[:, :-1] - Q[:, 1:], 0).max(1); Qs = np.sort(Q, axis=1)
        net_cvar = np.stack([Qs[:, :int(round(e_ * NLEV))].mean(1) if e_ < 1 else Qs.mean(1) for e_ in ETAS], 1)
        ev_secs += time.time() - t1
        sets = {"ordinary": [key[tuple(p)] for p in R[f"n{n}_ordinary"]]}
        for ie in range(len(ETAS)):
            for ic in range(len(CS)):
                sets[f"bnd_{ie}_{ic}"] = [key[tuple(p)] for p in R[f"n{n}_bnd_{ie}_{ic}"]]
        d = {}
        for ie, e_ in enumerate(ETAS):
            idx = np.array(sets["ordinary"]); err = net_cvar[idx, ie] - cv[idx, ie]
            d[f"eta{e_}"] = dict(ordinary=dict(bias=float(err.mean()), rmse=float(np.sqrt((err ** 2).mean()))))
            for ic, c in enumerate(CS):
                for name, ids in (("ordinary", np.array(sets["ordinary"])), ("boundary", np.array(sets[f"bnd_{ie}_{ic}"], dtype=int))):
                    if len(ids) == 0:
                        d[f"eta{e_}"][f"c{c}_{name}"] = None; continue
                    ve, vn = cv[ids, ie] - c, net_cvar[ids, ie] - c
                    treat = vn > 0
                    reg = np.maximum(ve, 0) - np.where(treat, ve, 0)
                    cell = dict(n_trials=int(len(ids)), disagreement=float((treat != (ve > 0)).mean()), regret=float(reg.mean()),
                                bias=float((vn - ve).mean()), rmse=float(np.sqrt(((vn - ve) ** 2).mean())))
                    d[f"eta{e_}"][f"c{c}_{name}"] = cell
            bi = np.concatenate([np.array(sets[f"bnd_{ie}_{ic}"], dtype=int) for ic in range(len(CS))])
            if len(bi):
                err = net_cvar[bi, ie] - cv[bi, ie]
                d[f"eta{e_}"]["boundary_pooled"] = dict(bias=float(err.mean()), rmse=float(np.sqrt((err ** 2).mean())), n_trials=int(len(bi)))
        ids = np.array(sets["ordinary"])
        d["crossing"] = dict(median_max_dip_over_sd=float(np.median(dips[ids] / sd[ids])), p99_max_dip_over_sd=float(np.quantile(dips[ids] / sd[ids], 0.99)),
                             max_dip_over_sd=float((dips[ids] / sd[ids]).max()), share_with_a_crossing=float((dips[ids] > 1e-6).mean()),
                             mean_abs_rearrangement_over_sd=float((np.abs(Q[ids] - Qs[ids]).mean(1) / sd[ids]).mean()))
        d["reference_tolerance"] = float(R[f"n{n}_tol"])
        res["by_n"][str(n)] = d
    res["eval_secs"] = ev_secs
    json.dump(res, open(PFX + f"gbc_decision_seed{seed}.json", "w"), indent=1)
    print("seed", seed, f"train {train_secs:.0f}s eval {ev_secs:.0f}s", flush=True)


def stage_agg():
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import glob
    seeds = sorted(int(f.split("seed")[1].split(".")[0]) for f in glob.glob(PFX + "gbc_decision_seed*.json"))
    S = [json.load(open(PFX + f"gbc_decision_seed{s}.json")) for s in seeds]
    ms = lambda v: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
    out = dict(seeds=seeds, train_secs=ms([s["train_secs"] for s in S]), eval_secs=ms([s["eval_secs"] for s in S]), by_n={}, boundary_note="c cancels in bias and RMSE")
    for n in NS:
        out["by_n"][str(n)] = {}
        d0 = S[0]["by_n"][str(n)]
        for k, v in d0.items():
            if k == "reference_tolerance":
                out["by_n"][str(n)][k] = v; continue
            if k == "crossing":
                out["by_n"][str(n)][k] = {kk: ms([s["by_n"][str(n)][k][kk] for s in S]) for kk in v}; continue
            out["by_n"][str(n)][k] = {}
            for kk, cell in v.items():
                if cell is None:
                    continue
                out["by_n"][str(n)][k][kk] = {sub: (ms([s["by_n"][str(n)][k][kk][sub] for s in S]) if sub != "n_trials" else cell[sub]) for sub in cell}
    json.dump(out, open(PFX + "res_gbc_decision_check.json", "w"), indent=1)
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.5)); sty = {0.05: ("o", "-"), 0.1: ("s", "--"), 0.25: ("^", ":"), 1.0: ("d", "-.")}
    for e_ in ETAS:
        mk, ls = sty[e_]
        m_ = [out["by_n"][str(n)][f"eta{e_}"]["ordinary"]["rmse"]["mean"] for n in NS]; s_ = [out["by_n"][str(n)][f"eta{e_}"]["ordinary"]["rmse"]["sd"] for n in NS]
        axs[0].errorbar(NS, m_, yerr=s_, marker=mk, ls=ls, color="k", ms=4, lw=1, capsize=2, label=rf"$\eta={e_:g}$")
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("patients per arm $n$"); axs[0].set_ylabel("RMSE of the network CVaR"); axs[0].legend(frameon=False, fontsize=7)
    for ic, c in enumerate(CS):
        mk = ["o", "s", "^"][ic]
        for name, ls, col in (("boundary", "-", "k"), ("ordinary", ":", "0.5")):
            m_, s_ = [], []
            for n in NS:
                cell = out["by_n"][str(n)]["eta0.25"].get(f"c{c}_{name}")
                m_.append(cell["disagreement"]["mean"] if cell else np.nan); s_.append(cell["disagreement"]["sd"] if cell else np.nan)
            axs[1].errorbar(NS, m_, yerr=s_, marker=mk, ls=ls, color=col, ms=4, lw=1, capsize=2, label=(f"$c={c:g}$" if name == "boundary" else None))
    axs[1].set_xscale("log"); axs[1].set_xlabel("patients per arm $n$"); axs[1].set_ylabel(r"disagreement with the exact rule, $\eta=0.25$")
    axs[1].legend(frameon=False, fontsize=7, title="solid: boundary trials, dotted: ordinary", title_fontsize=6.5)
    fig.tight_layout(); fig.savefig(("figs/quick_" if QUICK else "figs/") + "fig_gbc_decision_check.pdf")
    print(json.dumps(out["by_n"]["100"]["eta0.25"]["ordinary"]))


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True); os.makedirs("figs", exist_ok=True)
    st = args[0]
    if st == "ref":
        stage_ref()
    elif st == "train":
        stage_train(int(args[1]))
    elif st == "agg":
        stage_agg()
