"""Problem 6: local simulation based calibration of an amortized posterior, on the two arm trial network.

Produces figs/open_localsbc.pdf and results/res_localsbc.json. Requires torch.
Revised 23 October 2026 (r1023): the earlier version drew p0 independently of tau and clipped p1 to [0, 1]; the pairs now come from the
joint posterior (p0 | tau, D); the figure overlays the exact posterior's rank histogram (black steps) on the network's bars.
"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
import torch, torch.nn as nn
PAL = ["#1f5a96", "#d1495b", "#edae49", "#00798c", "#66a182", "#8d5fa8"]
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "pdf.fonttype": 42})
OUT = "figs/"; os.makedirs("results", exist_ok=True); torch.manual_seed(1); torch.set_num_threads(2); rng = np.random.default_rng(1); R = {}

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



S, m = 60000, 50
p0, p1 = rng.uniform(size=S), rng.uniform(size=S)
y0, y1 = rng.binomial(m, p0), rng.binomial(m, p1)
h = np.column_stack([y0, y1]) / m; tau_true = p1 - p0
net, _ = train(tau_true, h, epochs=60)

def draw_p0_given_tau(tau, y0, y1, G=20001):
    """Draw p0 from pi(p0 | tau, D), proportional to p0^y0 (1-p0)^(m-y0) (p0+tau)^y1 (1-p0-tau)^(m-y1) on
    max(0, -tau) < p0 < min(1, 1-tau): the posterior of p0 given tau and the data when (p0, p1) has the flat prior on
    the unit square and p1 = p0 + tau (the Jacobian of (p0, p1) -> (p0, tau) is one). Inverse cdf on a grid of G
    cells with a uniform jitter inside the chosen cell; the grid error is far below the Monte Carlo error."""
    lo, hi = max(0.0, -tau), min(1.0, 1.0 - tau); dx = (hi - lo) / G
    g = lo + dx * (np.arange(G) + 0.5)
    ld = y0 * np.log(g) + (m - y0) * np.log1p(-g) + y1 * np.log(g + tau) + (m - y1) * np.log1p(-g - tau)
    cdf = np.cumsum(np.exp(ld - ld.max())); k = min(int(np.searchsorted(cdf, rng.uniform() * cdf[-1])), G - 1)
    return float(g[k] + dx * (rng.uniform() - 0.5))


def exact_ranks(obs, M=200000, L=19):
    """The same local protocol with the exact posterior in place of the network: (p0, p1) from the independent beta
    posteriors at obs, a trial simulated from them, the exact posterior of tau recomputed, rank of tau among L draws.
    Returns the rank counts scaled to M = 400 draws (the reference histogram of the figure)."""
    a = rng.beta(1 + obs[0], 1 + m - obs[0], M); b = rng.beta(1 + obs[1], 1 + m - obs[1], M)
    z0, z1 = rng.binomial(m, a), rng.binomial(m, b)
    d1 = rng.beta(1 + z1[:, None], 1 + m - z1[:, None], (M, L)); d0 = rng.beta(1 + z0[:, None], 1 + m - z0[:, None], (M, L))
    r = ((d1 - d0) < (b - a)[:, None]).sum(1)
    return np.bincount(r, minlength=L + 1) * 400.0 / M

def local_sbc(obs, M=400, L=19):
    """Draw M values of the risk difference tau from the amortized posterior at obs, simulate a trial from each,
    recompute the amortized posterior and record the rank of the generating tau. To simulate a trial we also need
    (p0, p1), so p0 is drawn from pi(p0 | tau, D) (draw_p0_given_tau) and p1 = p0 + tau, with no clipping. A draw
    of tau outside (-1, 1) has no valid (p0, p1) (zero prior mass), so it is redrawn from the network (rejection
    onto the valid set); the number of redraws is returned."""
    ranks = []; hobs = [obs[0] / m, obs[1] / m]; redraws = 0
    pool = list(quantiles(net, hobs, rng.uniform(size=M)))
    while len(ranks) < M:
        if not pool:
            pool = list(quantiles(net, hobs, rng.uniform(size=M)))
        tau = float(pool.pop())
        if not (-1.0 < tau < 1.0):
            redraws += 1; continue
        a = draw_p0_given_tau(tau, obs[0], obs[1]); b = a + tau
        hh = [rng.binomial(m, a) / m, rng.binomial(m, b) / m]
        dd = quantiles(net, hh, rng.uniform(size=L)); ranks.append(int(np.sum(dd < tau)))
    ranks = np.array(ranks); counts = np.bincount(ranks, minlength=L + 1); chi = stats.chisquare(counts)
    return ranks, float(chi.statistic), float(chi.pvalue), redraws

cases = {"typical (18 of 50, 27 of 50)": (18, 27), "extreme (1 of 50, 49 of 50)": (1, 49)}
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.6))
for ax, (lab, obs) in zip(axs, cases.items()):
    ranks, chi, pv, redraws = local_sbc(obs); ex = exact_ranks(obs)
    ax.bar(np.arange(20), np.bincount(ranks, minlength=20), color=PAL[3], alpha=0.6)
    ax.axhline(len(ranks) / 20, color=PAL[1], lw=1, ls=":"); ax.step(np.arange(21) - 0.5, np.append(ex, ex[-1]), where="post", color="k", lw=1); ax.set_title(lab, fontsize=9); ax.set_xlabel("rank among 19 draws")
    # also compare the amortized posterior with the exact one at obs
    V = rng.uniform(size=20000); qd = quantiles(net, [obs[0] / m, obs[1] / m], V)
    exact = rng.beta(1 + obs[1], 1 + m - obs[1], 200000) - rng.beta(1 + obs[0], 1 + m - obs[0], 200000)
    R[lab] = dict(obs=obs, chi=chi, p=pv, redrawn_tau=redraws, rank_counts=[int(x) for x in np.bincount(ranks, minlength=20)],
                  exact_rank_counts_per_400=[float(x) for x in ex], upper_half_share=float((ranks > 9).mean()), exact_upper_half_share=float(ex[10:].sum() / ex.sum()), amortized_q=[float(x) for x in np.quantile(qd, [0.05, 0.5, 0.95])],
                  exact_q=[float(x) for x in np.quantile(exact, [0.05, 0.5, 0.95])], w1=float(np.mean(np.abs(np.sort(qd[:20000]) - np.sort(exact[:20000])))))
axs[0].set_ylabel("count")
fig.tight_layout(); fig.savefig(OUT + "open_localsbc.pdf"); plt.close()
json.dump(R, open("results/res_localsbc.json", "w"), indent=1); print(json.dumps(R, indent=1))
