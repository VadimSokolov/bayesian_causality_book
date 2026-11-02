"""Quantile neural network figures for the GBC section.

Produces figs/qnn_loss.pdf, figs/qnn_normal.pdf, figs/qnn_trial.pdf and results/res_qnn.json.
Requires numpy, scipy, matplotlib and torch (CPU is enough; runs in about two minutes).
"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import torch, torch.nn as nn

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "pdf.fonttype": 42})
PAL = ["#1f5a96", "#d1495b", "#edae49", "#00798c", "#66a182", "#8d5fa8"]
OUT = "figs/"; os.makedirs("results", exist_ok=True)
torch.manual_seed(1); torch.set_num_threads(2); rng = np.random.default_rng(1)
R = {}


class QNN(nn.Module):
    """f(tau, h): cosine embedding of tau, MLP embedding of h, elementwise product, MLP head."""
    def __init__(self, dh, width=64, K=32):
        super().__init__()
        self.register_buffer("k", torch.arange(1, K + 1).float() * np.pi)
        self.tau_net = nn.Sequential(nn.Linear(K, width), nn.ReLU())
        self.h_net = nn.Sequential(nn.Linear(dh, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, tau, h):
        e = self.tau_net(torch.cos(tau[:, None] * self.k))
        return self.head(e * self.h_net(h)).squeeze(-1)


def pinball(u, tau):
    return torch.mean(torch.maximum(tau * u, (tau - 1) * u))


def train(theta, h, epochs=60, batch=512, ntau=4, lr=2e-3):
    th = torch.tensor(theta, dtype=torch.float32); H = torch.tensor(h, dtype=torch.float32)
    net = QNN(H.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(th); hist = []
    for ep in range(epochs):
        perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            t = th[idx].repeat(ntau); hb = H[idx].repeat(ntau, 1)
            tau = torch.rand(len(t))
            loss = pinball(t - net(tau, hb), tau)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step(); hist.append(tot / n)
    return net, hist


def quantiles(net, h, taus):
    with torch.no_grad():
        T = torch.tensor(taus, dtype=torch.float32)
        Hh = torch.tensor(np.repeat(np.atleast_2d(h), len(taus), 0), dtype=torch.float32)
        return net(T, Hh).numpy()


t0 = time.time()
# ---------------------------------------------------------------- Figure 1: the check loss
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.6))
u = np.linspace(-2, 2, 400)
for (tau, ls), col in zip([(0.1, "-"), (0.5, "--"), (0.9, ":")], PAL):
    axs[0].plot(u, np.maximum(tau * u, (tau - 1) * u), ls, color=col, label=rf"$\tau = {tau}$")
axs[0].set_xlabel(r"$u = \theta - q$"); axs[0].set_ylabel(r"$\rho_\tau(u)$"); axs[0].legend(frameon=False)
axs[0].set_title("Check loss", fontsize=10)
z = rng.gamma(2.0, 1.0, 4000); qs = np.linspace(0, 8, 400)
for (tau, ls), col in zip([(0.1, "-"), (0.5, "--"), (0.9, ":")], PAL):
    risk = [np.mean(np.maximum(tau * (z - q), (tau - 1) * (z - q))) for q in qs]
    axs[1].plot(qs, risk, ls, color=col)
    qstar = stats.gamma(2).ppf(tau); axs[1].axvline(qstar, color=col, lw=0.6)
axs[1].set_xlabel(r"$q$"); axs[1].set_ylabel(r"$E\,\rho_\tau(\theta - q)$")
axs[1].set_title(r"Expected loss, $\theta \sim$ Gamma(2)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT + "qnn_loss.pdf"); plt.close()

# ---------------------------------------------------------------- Figure 2: normal learning
# theta ~ N(0, 1); y_1..y_n | theta ~ N(theta, 1), n = 10; h = ybar (sufficient)
S, n = 40000, 10
theta = rng.normal(0, 1, S); ybar = theta + rng.normal(0, 1 / np.sqrt(n), S)
net, hist = train(theta, ybar[:, None])
post = lambda hb: stats.norm(n * hb / (n + 1), 1 / np.sqrt(n + 1))
taus = np.linspace(0.01, 0.99, 99)
fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.4))
w1 = []
# Gray scale print: the exact quantile functions are thick gray lines and the learned ones thin black lines that differ by style
# (in the colour version the tan dotted line vanished against the gray); the histogram is light gray and the exact density black
for hb, ls in [(-1.5, "-"), (0.0, "--"), (1.5, ":")]:
    q = quantiles(net, [hb], taus)
    axs[0].plot(taus, post(hb).ppf(taus), color="grey", lw=2.5, alpha=0.5)
    axs[0].plot(taus, q, ls, color="k", lw=1.2, label=rf"$\bar y = {hb}$")
    w1.append(float(np.mean(np.abs(q - post(hb).ppf(taus)))))
# each pair is labelled in the empty space above its left end: a legend in the upper left sat on the curve for ybar = 1.5
for hb, yl in [(-1.5, -1.15), (0.0, 0.25), (1.5, 1.65)]:
    axs[0].text(0.03, yl, rf"$\bar y = {hb}$", fontsize=7)
axs[0].set_xlabel(r"$\tau$"); axs[0].set_ylabel(r"$Q(\tau \mid \bar y)$")
axs[0].set_title("Quantile functions", fontsize=10)
hb = 1.5; V = rng.uniform(size=20000); draws = quantiles(net, [hb], V)
xs = np.linspace(0.3, 2.4, 200)
axs[1].hist(draws, bins=50, density=True, color="0.75", edgecolor="none")
axs[1].plot(xs, post(hb).pdf(xs), color="k"); axs[1].set_yticks([])
axs[1].set_xlabel(r"$\theta$"); axs[1].set_title(r"Draws at $\bar y = 1.5$", fontsize=10)
axs[2].plot(np.arange(1, len(hist) + 1), hist, color="k"); axs[2].set_xlabel("epoch")
axs[2].set_ylabel("check loss"); axs[2].set_title("Training", fontsize=10)
fig.tight_layout(); fig.savefig(OUT + "qnn_normal.pdf"); plt.close()
R["normal"] = dict(S=S, n=n, w1=w1, draw_mean=float(draws.mean()), draw_sd=float(draws.std()),
                   exact_mean=float(post(hb).mean()), exact_sd=float(post(hb).std()))

# ---------------------------------------------------------------- Figure 3: a two arm trial
# p0, p1 ~ Beta(1,1); y_t ~ Bin(m, p_t), m = 50; target tau = p1 - p0; h = (y0/m, y1/m)
S, m = 60000, 50
p0, p1 = rng.uniform(size=S), rng.uniform(size=S)
y0, y1 = rng.binomial(m, p0), rng.binomial(m, p1)
h = np.column_stack([y0, y1]) / m; tau_true = p1 - p0
net2, _ = train(tau_true, h, epochs=60)
obs = (18, 27)  # observed successes out of 50 in control and treatment
exact = rng.beta(1 + obs[1], 1 + m - obs[1], 200000) - rng.beta(1 + obs[0], 1 + m - obs[0], 200000)
V = rng.uniform(size=20000); qd = quantiles(net2, [obs[0] / m, obs[1] / m], V)
# simulation based calibration: rank of the true tau among L posterior draws
L, nrep = 19, 500
ranks = []
for r in range(nrep):
    a, b = rng.uniform(), rng.uniform()
    hh = [rng.binomial(m, a) / m, rng.binomial(m, b) / m]
    dd = quantiles(net2, hh, rng.uniform(size=L))
    ranks.append(int(np.sum(dd < b - a)))
ranks = np.array(ranks)
chi = stats.chisquare(np.bincount(ranks, minlength=L + 1))
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.6))
xs = np.linspace(-0.3, 0.6, 200)
axs[0].hist(qd, bins=50, density=True, color=PAL[3], alpha=0.45, edgecolor="none", label="QNN draws")
kde = stats.gaussian_kde(exact); axs[0].plot(xs, kde(xs), color=PAL[1], label="exact")
axs[0].axvline(0, color="grey", lw=0.6); axs[0].legend(frameon=False, fontsize=8); axs[0].set_yticks([])
axs[0].set_xlabel(r"$\tau = p_1 - p_0$"); axs[0].set_title("Posterior of the risk difference", fontsize=10)
axs[1].bar(np.arange(L + 1), np.bincount(ranks, minlength=L + 1), color=PAL[3], alpha=0.45, edgecolor="none")
axs[1].axhline(nrep / (L + 1), color=PAL[1], lw=1)
axs[1].set_xlabel("rank of true $\\tau$ among 19 draws"); axs[1].set_ylabel("count")
axs[1].set_title("Calibration check", fontsize=10)
fig.tight_layout(); fig.savefig(OUT + "qnn_trial.pdf"); plt.close()
R["trial"] = dict(S=S, m=m, obs=obs, qnn_mean=float(qd.mean()), qnn_q=[float(x) for x in np.quantile(qd, [0.05, 0.5, 0.95])],
                  exact_mean=float(exact.mean()), exact_q=[float(x) for x in np.quantile(exact, [0.05, 0.5, 0.95])],
                  qnn_ppos=float(np.mean(qd > 0)), exact_ppos=float(np.mean(exact > 0)),
                  sbc_chisq=float(chi.statistic), sbc_p=float(chi.pvalue), nrep=nrep, L=L)
R["secs"] = round(time.time() - t0)
json.dump(R, open("results/res_qnn.json", "w"), indent=1); print(json.dumps(R, indent=1))
