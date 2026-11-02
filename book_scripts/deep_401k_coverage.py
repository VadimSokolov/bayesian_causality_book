"""Coverage check of the generative bootstrap bands of Chapter 10 (401(k) eligibility) on simulated truths that resemble the data.

Truth 1 is the quantile network of deep_401k.py, fitted once to the 401(k) data with unit weights. A simulated data set keeps the 9,915
households' covariates and eligibility and draws net financial assets from that network, Y = Q(U | x, t) with U uniform, so that unconfoundedness
holds and every effect is known. Truth 2 lies outside the network family: with x the standardized covariates (x1 age, x2 income, x3 education,
x4 family size) and t the observed eligibility,
  Y(t) = 3 x1 + 2 sin x2 + x3 x4 + t {2 + 2 * 1(x2 > 0)} + (1 + 0.5 |x1|)(1 + 0.5 t) eps,  eps = {exp Z - exp(1/2)} / sqrt{e (e - 1)}, Z ~ N(0, 1),
a skewed, heteroskedastic noise with mean zero and variance one; the average effect and the income quintile effects are closed form and the
quantile effects are computed from 1,000 noise draws per household. Each data set is analysed as the chapter analyses the real data: refits of
the network under Bayesian bootstrap weights, band = mean of the refits plus or minus 1.96 standard deviations (and the 2.5 and 97.5 percent
points). Reported, for the average effect, the 19 quantile effects of deep_401k.py and the five income quintile effects: bias, RMSE, coverage
and band width with Monte Carlo standard errors.
Jobs (name: truth, inputs, refits per data set; the data sets of a job have seeds 1000 + r (truth 1) or 5000 + r (truth 2)):
  f9           t1, nine covariates, 200 refits, 1,000 data sets (the first 20 refits of the first 200 data sets repeat the earlier study)
  t1raw, t1plsonce, t1plsrefit   t1, the 9 covariates plus 200 independent N(0,1) noise covariates (regenerated for every data set) fed to the
               network ("raw"), or five partial least squares directions fitted once per data set ("plsonce") or refitted with the Bayesian bootstrap
               weights inside every refit ("plsrefit"), 20 refits, 200 data sets
  t2nine, t2raw, t2plsonce       the same on truth 2, 20 refits, 200 data sets;  t2nine200: truth 2, nine covariates, 200 refits, 200 data sets
  big_raw, big_plsonce, big_plsrefit   t1, 100 refits on the first 20 data sets, to show what more refits buy
Usage, from the repository root:
  python book_scripts/deep_401k_coverage.py truth                  truth 1: results/deep401k_cov/truth.pt, truth.npz (exists from the first study)
  python book_scripts/deep_401k_coverage.py truth2                 truth 2: results/deep401k_cov/truth2.npz
  python book_scripts/deep_401k_coverage.py job JOB START COUNT    data sets START .. START + COUNT - 1 of the job: results/deep401k_cov/JOB/rep_NNNN.npz
  python book_scripts/deep_401k_coverage.py agg                    results/res_deep401k_coverage.json (f9) and results/res_deep401k_coverage_hd.json (the rest)
The options --dir DIR, --refits B and --epochs E change the folder, the number of refits and the training epochs (40), for a quick test.
Requires numpy, pandas, scipy, scikit-learn and torch.
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from scipy.stats import chisquare
from sklearn.cross_decomposition import PLSRegression

args = sys.argv[1:]
def option(name, default):
    if name not in args: return default
    i = args.index(name); value = args[i + 1]; del args[i:i + 2]; return type(default)(value)
OUTD = option("--dir", "results/deep401k_cov/").rstrip("/") + "/"; B_OPT = option("--refits", 0); EPOCHS = option("--epochs", 40)
os.makedirs(OUTD, exist_ok=True); os.makedirs("results", exist_ok=True); torch.set_num_threads(2)

d = pd.read_stata("data/sipp1991.dta")
cols = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]
X = d[cols].to_numpy(float); T = d["e401"].to_numpy(float); Y = d["net_tfa"].to_numpy(float) / 1000
n = len(Y); Xs = (X - X.mean(0)) / X.std(0)
QS = np.linspace(0.05, 0.95, 19)
inc_q = pd.qcut(d["inc"], 5, labels=False).to_numpy()
H = np.column_stack([Xs, T]); H0 = np.column_stack([Xs, np.zeros(n)]); H1 = np.column_stack([Xs, np.ones(n)])


class QNN(nn.Module):
    """The network of deep_401k.py."""
    def __init__(self, dh, width=64, K=32):
        super().__init__()
        self.register_buffer("k", torch.arange(1, K + 1).float() * np.pi)
        self.tau_net = nn.Sequential(nn.Linear(K, width), nn.ReLU())
        self.h_net = nn.Sequential(nn.Linear(dh, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
    def forward(self, tau, h):
        return self.head(self.tau_net(torch.cos(tau[:, None] * self.k)) * self.h_net(h)).squeeze(-1)


def fit(H, y, w, epochs=40, batch=512, ntau=4, lr=2e-3, wd=1e-4):
    """The training of deep_401k.py: weighted pinball loss, the weights w a Bayesian bootstrap draw."""
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


def potential(net, H0, H1, ys, M=10):
    """As in deep_401k.py, with the outcome scale ys passed in: draws of Y(0) and Y(1) for every unit, and unit mean effects."""
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


def truth():
    """One network fitted to the real data with unit weights, and its effects on the 9,915 households."""
    torch.manual_seed(2016); ys = Y.std()
    net = fit(H, Y / ys, np.ones(n), epochs=EPOCHS); torch.save(net.state_dict(), OUTD + "truth.pt")
    A0 = torch.tensor(H0, dtype=torch.float32); A1 = torch.tensor(H1, dtype=torch.float32)
    with torch.no_grad():
        eff = np.zeros(n)        # E{Y(1) - Y(0) | x}, the integral over u at the midpoints of 1,000 equal cells
        for u in (np.arange(1000) + 0.5) / 1000:
            tt = torch.full((n,), float(u)); eff += (net(tt, A1) - net(tt, A0)).numpy() * ys / 1000
        y0, y1 = [], []          # the laws of Y(0) and Y(1) over the households, 400 draws per household
        for m in range(400):
            U = torch.rand(n); y0.append(net(U, A0).numpy() * ys); y1.append(net(U, A1).numpy() * ys)
    y0 = np.concatenate(y0); y1 = np.concatenate(y1)
    np.savez(OUTD + "truth.npz", ys=ys, eff=eff, ate=np.array([eff.mean()]), qte=np.quantile(y1, QS) - np.quantile(y0, QS),
             inc=np.array([eff[inc_q == k].mean() for k in range(5)]))
    print("truth: average effect", round(float(eff.mean()), 3), flush=True)



def wpls_scores(Z, R, w, k=5):
    """Scores on k partial least squares directions with observation weights w (the Bayesian bootstrap weights): weighted centring and scaling,
    PLS2 deflation with the leading singular vector of the weighted cross covariance, then the scores standardized."""
    sw = w / w.sum(); mz = sw @ Z; sz = np.sqrt(sw @ (Z - mz) ** 2); E = (Z - mz) / sz
    mr = sw @ R; sr = np.sqrt(sw @ (R - mr) ** 2); F = (R - mr) / sr; cols = []
    for _ in range(k):
        wk = np.linalg.svd(E.T @ (sw[:, None] * F), full_matrices=False)[0][:, 0]; t = E @ wk; nt = float(sw @ (t * t))
        E = E - np.outer(t, E.T @ (sw * t) / nt); F = F - np.outer(t, F.T @ (sw * t) / nt); cols.append(t)
    S = np.column_stack(cols); return (S - S.mean(0)) / S.std(0)


JOBS = {"f9": ("t1", "nine", 200), "t1raw": ("t1", "raw", 20), "t1plsonce": ("t1", "plsonce", 20), "t1plsrefit": ("t1", "plsrefit", 20),
        "t2nine": ("t2", "nine", 20), "t2raw": ("t2", "raw", 20), "t2plsonce": ("t2", "plsonce", 20), "t2nine200": ("t2", "nine", 200),
        "big_raw": ("t1", "raw", 100), "big_plsonce": ("t1", "plsonce", 100), "big_plsrefit": ("t1", "plsrefit", 100)}


def truth2():
    """Truth 2 (outside the network family): closed form average and income quintile effects, quantile effects by simulation."""
    c2 = 2 + 2 * (Xs[:, 1] > 0); rng = np.random.default_rng(99); f0 = 3 * Xs[:, 0] + 2 * np.sin(Xs[:, 1]) + Xs[:, 2] * Xs[:, 3]
    y0, y1 = [], []
    for m in range(1000):
        eps = (np.exp(rng.standard_normal(n)) - np.exp(0.5)) / np.sqrt(np.e * (np.e - 1)); sc = 1 + 0.5 * np.abs(Xs[:, 0])
        y0.append(f0 + sc * eps); y1.append(f0 + c2 + sc * 1.5 * eps)
    y0 = np.concatenate(y0); y1 = np.concatenate(y1)
    np.savez(OUTD + "truth2.npz", ate=np.array([c2.mean()]), qte=np.quantile(y1, QS) - np.quantile(y0, QS), inc=np.array([c2[inc_q == k].mean() for k in range(5)]))
    print("truth 2: average effect", round(float(c2.mean()), 3), flush=True)


def simulate(truth_id, r, net0, ys0):
    """The outcome of simulated data set r."""
    if truth_id == "t1":
        torch.manual_seed(1000 + r)
        with torch.no_grad(): return net0(torch.rand(n), torch.tensor(H, dtype=torch.float32)).numpy().astype(float) * ys0
    rng = np.random.default_rng(5000 + r); eps = (np.exp(rng.standard_normal(n)) - np.exp(0.5)) / np.sqrt(np.e * (np.e - 1))
    return (3 * Xs[:, 0] + 2 * np.sin(Xs[:, 1]) + Xs[:, 2] * Xs[:, 3] + T * (2 + 2 * (Xs[:, 1] > 0)) + (1 + 0.5 * np.abs(Xs[:, 0])) * (1 + 0.5 * T) * eps)


def job(name, start, count):
    truth_id, inputs, Bj = JOBS[name]; Bj = B_OPT or Bj; jid = sorted(JOBS).index(name)
    net0 = QNN(H.shape[1]); net0.load_state_dict(torch.load(OUTD + "truth.pt")); ys0 = float(np.load(OUTD + "truth.npz")["ys"])
    os.makedirs(OUTD + name, exist_ok=True)
    for r in range(start, start + count):
        t0 = time.time(); Ys = simulate(truth_id, r, net0, ys0); ys = Ys.std()
        if name == "f9": rng = np.random.default_rng(1000 + r)        # the stream of the first study: its first 20 refits are repeated
        else: torch.manual_seed(7000 + 31 * r + jid); rng = np.random.default_rng([1000 + r, jid])
        Z = Xs if inputs == "nine" else np.column_stack([Xs, np.random.default_rng([7, r]).normal(size=(n, 200))])
        Hx = Z if inputs in ("nine", "raw") else None
        if inputs == "plsonce":
            S = PLSRegression(n_components=5, scale=True).fit(Z, np.column_stack([Ys / ys, T])).transform(Z); Hx = (S - S.mean(0)) / S.std(0)
        ate, qte, inc = [], [], []
        for b in range(Bj):
            w = rng.exponential(size=n); w /= w.mean()
            if inputs == "plsrefit": Hx = wpls_scores(Z, np.column_stack([Ys / ys, T]), w)
            net = fit(np.column_stack([Hx, T]), Ys / ys, w, epochs=EPOCHS)
            y0, y1, eff = potential(net, np.column_stack([Hx, np.zeros(n)]), np.column_stack([Hx, np.ones(n)]), ys)
            ate.append([eff.mean()]); qte.append(np.quantile(y1, QS) - np.quantile(y0, QS)); inc.append([eff[inc_q == k].mean() for k in range(5)])
        np.savez(OUTD + name + "/rep_%04d.npz" % r, ate=np.array(ate), qte=np.array(qte), inc=np.array(inc), secs=time.time() - t0)
        print(name, "rep", r, round(float(np.mean(ate)), 3), round(time.time() - t0), flush=True)


def stats(D, t, Bk=None):
    """D: data sets x refits x quantities; t: truth. Bias, RMSE, coverage of mean +- 1.96 sd and of the 2.5 and 97.5 percent points, width,
    each with its Monte Carlo standard error over data sets, for the first Bk refits."""
    D = D[:, :Bk] if Bk else D; R_ = D.shape[0]; m = D.mean(1); s = D.std(1, ddof=1); lo, hi = np.quantile(D, [0.025, 0.975], axis=1)
    err = m - t; cn = (np.abs(err) <= 1.96 * s).astype(float); cp = ((lo <= t) & (t <= hi)).astype(float)
    def se(x): return (x.std(0, ddof=1) / np.sqrt(R_)).tolist()
    return dict(refits=int(D.shape[1]), data_sets=int(R_), bias=err.mean(0).tolist(), bias_se=se(err), rmse=np.sqrt((err ** 2).mean(0)).tolist(),
                rmse_se=(((err ** 2).std(0, ddof=1) / np.sqrt(R_)) / (2 * np.sqrt((err ** 2).mean(0)))).tolist(),
                coverage_normal=cn.mean(0).tolist(), coverage_normal_se=se(cn), coverage_pct=cp.mean(0).tolist(), coverage_pct_se=se(cp),
                width_normal=(2 * 1.96 * s).mean(0).tolist(), width_normal_se=se(2 * 1.96 * s), width_pct=(hi - lo).mean(0).tolist(), sd=s.mean(0).tolist())


def agg():
    tr = {"t1": np.load(OUTD + "truth.npz")}
    if os.path.exists(OUTD + "truth2.npz"): tr["t2"] = np.load(OUTD + "truth2.npz")
    res, hd = {}, {}
    for name, (tid, inputs, Bj) in JOBS.items():
        files = sorted(glob.glob(OUTD + name + "/rep_*.npz"))
        if not files: continue
        reps = [np.load(f) for f in files]; o = dict(job=name, truth=tid, inputs=inputs, secs_per_data_set=float(np.mean([float(x["secs"]) for x in reps])))
        for key in ["ate", "qte", "inc"]:
            D = np.array([x[key] for x in reps]); t = tr[tid][key]; o[key + "_truth"] = t.tolist()
            o[key] = stats(D, t)
            if D.shape[1] > 20: o[key + "_first20"] = stats(D, t, 20)
            if D.shape[1] > 100: o[key + "_first100"] = stats(D, t, 100)
            if D.shape[1] > 50: o[key + "_first50"] = stats(D, t, 50)
            if len(reps) > 200 and name == "f9": o[key + "_first200sets_20refits"] = stats(D[:200], t, 20); o[key + "_first200sets"] = stats(D[:200], t)
            if len(reps) >= 20: o[key + "_first20sets"] = stats(D[:20], t)
            if name == "f9" and key != "qte" or name == "f9":
                Bk = D.shape[1]; rank = (D < t).sum(1)
                o[key]["rank_chi2"] = [float(chisquare(np.bincount(rank[:, j], minlength=Bk + 1)).statistic) for j in range(D.shape[2])]
                o[key]["rank_below_all"] = (rank == 0).mean(0).tolist(); o[key]["rank_above_all"] = (rank == Bk).mean(0).tolist()
        (res if name == "f9" else hd)[name] = o
    json.dump(res, open("results/res_deep401k_coverage.json", "w"), indent=1); json.dump(hd, open("results/res_deep401k_coverage_hd.json", "w"), indent=1)
    for name, o in {**res, **hd}.items():
        print(name, "sets", o["ate"]["data_sets"], "refits", o["ate"]["refits"], "ATE cov %.3f bias %.3f rmse %.3f width %.2f" % (o["ate"]["coverage_normal"][0], o["ate"]["bias"][0], o["ate"]["rmse"][0], o["ate"]["width_normal"][0]),
              "QTE mean cov %.3f" % np.mean(o["qte"]["coverage_normal"]), flush=True)


if __name__ == "__main__":
    if args[0] == "truth": truth()
    elif args[0] == "truth2": truth2()
    elif args[0] == "job": job(args[1], int(args[2]), int(args[3]))
    elif args[0] == "agg": agg()
    else: sys.exit("usage: deep_401k_coverage.py truth | truth2 | job JOB START COUNT | agg")
