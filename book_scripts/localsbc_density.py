"""Problem 6 follow-up (r1023): does the calibration error of the amortized posterior track the training mass or the position in the summary space?

Design. Two arm trial, m = 50 per arm, flat priors on p0 and p1, summaries s = (y0, y1) on the 51 x 51 grid. The exact posterior of the risk
difference tau = p1 - p0 given s is the difference of independent Beta(1 + y0, 51 - y0) and Beta(1 + y1, 51 - y1) variables, so it is the same
under every training design; only the distribution of the summaries in the training set changes. Two designs for the summaries:
  uniform: s uniform over the 2,601 grid points (this is the marginal of the original design, where p0, p1 are uniform and y is binomial);
  gaussian: q(s) proportional to exp{-8 [(y0 - 25)^2 + (y1 - 25)^2] / 2500} on the grid (a bell with scale 12.5 around (25, 25), standard deviation 11.1 on the grid).
For each sampled summary p0 and p1 are drawn from their exact beta posteriors and the target is p1 - p0. The network is the one of
open_localsbc.py (cosine embedding of the quantile level, MLP of (y0, y1)/50, product, head; width 64, K = 32; pinball loss, 60 epochs, batch 512,
4 quantile levels per example, Adam 2e-3 with cosine decay). Training sizes N = 15,000, 60,000, 240,000; ten seeds (1 to 10) each, so 60 networks.
Evaluation on every grid summary, two metrics, each divided by the exact posterior standard deviation (closed form) and saved per network as a 51 x 51 grid:
  ratio:        mean absolute difference between the network quantile curve as output (200 midpoint levels, NOT sorted) and the exact quantile function
                (100,000 exact draws per summary, fixed seed); an upper bound on the Wasserstein-1 distance where the network curve crosses;
  ratio_sorted: the same after sorting the network curve over the 200 levels (the rearrangement), which is the discrete Wasserstein-1 distance between
                the network's quantile function and the exact posterior. (r1023b: added; the training and the seeds are those of the first run.)
Usage: python book_scripts/localsbc_density.py exact          (precompute, writes results/localsbc_density/exact.npz)
       python book_scripts/localsbc_density.py TASK           (TASK = 0..59, or SLURM_ARRAY_TASK_ID when no argument is given; = size_index * 20 + design_index * 10 + seed - 1)
Merge: book_scripts/localsbc_density_merge.py.
"""
import sys, os, time
import numpy as np
M = 50; SIZES = [15000, 60000, 240000]; DESIGNS = ["uniform", "gaussian"]; OUT = "results/localsbc_density"; U = (np.arange(200) + 0.5) / 200
YY0, YY1 = np.meshgrid(np.arange(M + 1), np.arange(M + 1), indexing="ij")   # grid[y0, y1]

def qmass():
    q = np.exp(-8 * ((YY0 - 25) ** 2 + (YY1 - 25) ** 2) / 2500.0); return q / q.sum()

def exact():
    rng = np.random.default_rng(20261023); os.makedirs(OUT, exist_ok=True)
    Q = np.zeros((M + 1, M + 1, len(U))); SD = np.zeros((M + 1, M + 1))
    for a in range(M + 1):
        for b in range(M + 1):
            d = rng.beta(1 + b, M + 1 - b, 100000) - rng.beta(1 + a, M + 1 - a, 100000)
            Q[a, b] = np.quantile(d, U)
            v = lambda k: (1 + k) * (M + 1 - k) / ((M + 2) ** 2 * (M + 3))
            SD[a, b] = np.sqrt(v(a) + v(b))
    np.savez(f"{OUT}/exact.npz", Q=Q, SD=SD, qmass=qmass())

def run(task):
    import torch, torch.nn as nn
    si, rest = divmod(task, 20); di, seed0 = divmod(rest, 10); seed = seed0 + 1; N = SIZES[si]
    torch.manual_seed(seed); torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    rng = np.random.default_rng([20261023, N, di, seed])
    p = np.full((M + 1) ** 2, 1.0 / (M + 1) ** 2) if di == 0 else qmass().ravel()
    idx = rng.choice((M + 1) ** 2, size=N, p=p); y0, y1 = idx // (M + 1), idx % (M + 1)
    p0 = rng.beta(1 + y0, M + 1 - y0); p1 = rng.beta(1 + y1, M + 1 - y1)
    h = np.column_stack([y0, y1]) / M; target = p1 - p0

    class QNN(nn.Module):
        def __init__(self, dh, width=64, K=32):
            super().__init__()
            self.register_buffer("k", torch.arange(1, K + 1).float() * np.pi)
            self.tau_net = nn.Sequential(nn.Linear(K, width), nn.ReLU())
            self.h_net = nn.Sequential(nn.Linear(dh, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
        def forward(self, tau, hh):
            return self.head(self.tau_net(torch.cos(tau[:, None] * self.k)) * self.h_net(hh)).squeeze(-1)
    pinball = lambda u, tau: torch.mean(torch.maximum(tau * u, (tau - 1) * u))
    th = torch.tensor(target, dtype=torch.float32); H = torch.tensor(h, dtype=torch.float32)
    net = QNN(2); opt = torch.optim.Adam(net.parameters(), lr=2e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60); t0 = time.time()
    for ep in range(60):
        perm = torch.randperm(N)
        for i in range(0, N, 512):
            b = perm[i:i + 512]; t = th[b].repeat(4); hb = H[b].repeat(4, 1); tau = torch.rand(len(t))
            loss = pinball(t - net(tau, hb), tau); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    E = np.load(f"{OUT}/exact.npz"); ratio = np.zeros((M + 1, M + 1))
    with torch.no_grad():
        Tt = torch.tensor(np.tile(U, (M + 1) ** 2), dtype=torch.float32)
        Hh = torch.tensor(np.repeat(np.column_stack([YY0.ravel(), YY1.ravel()]) / M, len(U), 0), dtype=torch.float32)
        qn = net(Tt, Hh).numpy().reshape(M + 1, M + 1, len(U))
    ratio = np.abs(qn - E["Q"]).mean(-1) / E["SD"]
    ratio_sorted = np.abs(np.sort(qn, -1) - E["Q"]).mean(-1) / E["SD"]
    np.savez(f"{OUT}/net_{task}.npz", ratio=ratio, ratio_sorted=ratio_sorted, N=N, design=di, seed=seed, secs=time.time() - t0, final_pinball=float(loss.item()))
    print("task", task, "N", N, DESIGNS[di], "seed", seed, "median ratio", np.median(ratio), "sorted", np.median(ratio_sorted), "typical", ratio[18, 27], ratio_sorted[18, 27], "extreme", ratio[1, 49], ratio_sorted[1, 49], "secs", round(time.time() - t0), flush=True)

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else os.environ["SLURM_ARRAY_TASK_ID"]
    exact() if arg == "exact" else run(int(arg))
