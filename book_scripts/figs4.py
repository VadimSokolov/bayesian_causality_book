"""Additional figures for version 18: orthogonality, the partially linear estimand,
Gaussian process causal inference, and the probability of necessity.

Produces figs/orth_bias.pdf, figs/plm_weights.pdf, figs/gp_causal.pdf, figs/pn_bound.pdf
and results/res_figs4.json. Requires numpy, scipy, matplotlib, scikit-learn.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, integrate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "pdf.fonttype": 42})
PAL = ["#1f5a96", "#d1495b", "#edae49", "#00798c", "#66a182", "#8d5fa8"]
OUT = "figs/"; os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(7); R = {}
expit = lambda z: 1 / (1 + np.exp(-z))

# ------------------------------------------------ 1. plug in versus orthogonal bias
n, tau = 200000, 1.0
x = rng.normal(size=n); m = x; g = x
T = m + rng.normal(size=n); Y = tau * T + g + rng.normal(size=n)
ell = tau * m + g
cs = np.linspace(-0.5, 0.5, 41); bp, bo = [], []
for c in cs:
    gh = g + c * x; mh = m + c * x; lh = tau * mh + gh
    bp.append(np.sum(T * (Y - gh)) / np.sum(T * T) - tau)
    r = T - mh; bo.append(np.sum(r * (Y - lh)) / np.sum(r * r) - tau)
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.7))
axs[0].plot(cs, bp, color=PAL[0], label="plug in")
axs[0].plot(cs, bo, "--", color=PAL[1], label="orthogonal score")
axs[0].axhline(0, color="grey", lw=0.5); axs[0].legend(frameon=False, fontsize=8)
axs[0].set_xlabel("size of nuisance error $c$"); axs[0].set_ylabel("bias of $\\hat\\tau$")
axs[1].plot(cs, bo, "--", color=PAL[1]); axs[1].plot(cs, cs ** 2, ":", color="grey", label="$c^2$")
axs[1].set_xlabel("size of nuisance error $c$"); axs[1].set_title("orthogonal score against $c^2$", fontsize=10)
axs[1].legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(OUT + "orth_bias.pdf"); plt.close()
R["orth"] = dict(c=0.2, plug=float(np.interp(0.2, cs, bp)), orth=float(np.interp(0.2, cs, bo)))

# ------------------------------------------------ 2. what the partially linear model estimates
phi = stats.norm.pdf
def E(f): return integrate.quad(lambda z: f(z) * phi(z), -8, 8)[0]
gams = np.linspace(0, 3, 31); est = []; veff = []
for gm in gams:
    e = lambda z: expit(1.0 + gm * z); tf = lambda z: 1 + z
    w = lambda z: e(z) * (1 - e(z))
    est.append(E(lambda z: w(z) * tf(z)) / E(w))
    veff.append((E(lambda z: 1 / e(z)), E(lambda z: 1 / (1 - e(z))), E(lambda z: z ** 2)))
veff = np.array(veff)
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.7))
axs[0].plot(gams, est, color=PAL[0], label="partially linear estimand")
axs[0].axhline(1, color="grey", ls="--", label="average effect")
axs[0].set_xlabel("strength of selection $\\gamma$"); axs[0].set_ylabel("estimand")
axs[0].legend(frameon=False, fontsize=8)
axs[1].plot(gams, veff[:, 0], color=PAL[0], label="$E\\{1/e(X)\\}$")
axs[1].plot(gams, veff[:, 1], "--", color=PAL[1], label="$E\\{1/(1-e(X))\\}$")
axs[1].plot(gams, veff[:, 2], ":", color=PAL[3], label="$E(\\tau(X)-\\tau)^2$")
axs[1].set_yscale("log"); axs[1].set_xlabel("strength of selection $\\gamma$")
axs[1].set_ylabel("component of $V_{\\mathrm{eff}}$"); axs[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "plm_weights.pdf"); plt.close()
R["plm"] = dict(gamma=[0, 1, 2, 3], est=[float(est[i]) for i in [0, 10, 20, 30]],
                inv_e=[float(veff[i, 0]) for i in [0, 10, 20, 30]], inv_1me=[float(veff[i, 1]) for i in [0, 10, 20, 30]])

# ------------------------------------------------ 3. Gaussian processes: discontinuity and overlap
class gp:
    """Fit hyperparameters with a white noise term, then predict the latent function without it."""
    def fit(self, X, y):
        g = GaussianProcessRegressor(C(1.0) * RBF(0.5, (0.05, 5)) + WhiteKernel(0.1), normalize_y=True,
                                     n_restarts_optimizer=3, random_state=0).fit(X, y)
        self.g = GaussianProcessRegressor(g.kernel_.k1, alpha=g.kernel_.k2.noise_level, normalize_y=True,
                                          optimizer=None).fit(X, y)
        return self
    def predict(self, X, return_std=False):
        return self.g.predict(X, return_std=return_std)
nr = 300; r = rng.uniform(-1, 1, nr); jump = 0.5
y = 0.8 * r + 0.6 * np.sin(2 * r) + jump * (r >= 0) + rng.normal(0, 0.3, nr)
L, Rr = r < 0, r >= 0
gl = gp().fit(r[L, None], y[L]); gr = gp().fit(r[Rr, None], y[Rr])
xl = np.linspace(-1, 0, 100)[:, None]; xr = np.linspace(0, 1, 100)[:, None]
ml, sl = gl.predict(xl, return_std=True); mr, sr = gr.predict(xr, return_std=True)
a0 = gl.predict([[0.0]], return_std=True); b0 = gr.predict([[0.0]], return_std=True)
jm = float(b0[0][0] - a0[0][0]); js = float(np.hypot(b0[1][0], a0[1][0]))
# overlap: controls only for x < 0.4, treated only for x > -0.4
nx = 300; xc = rng.uniform(-2, 0.4, nx); xt = rng.uniform(-0.4, 2, nx)
f0 = lambda z: np.sin(1.5 * z); yc = f0(xc) + rng.normal(0, 0.25, nx)
g0 = gp().fit(xc[:, None], yc); xs = np.linspace(-2, 2, 200)[:, None]
m0, s0 = g0.predict(xs, return_std=True)
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.8))
axs[0].scatter(r, y, s=4, color=PAL[3], alpha=0.5)
for xx, mm, ss in [(xl, ml, sl), (xr, mr, sr)]:
    axs[0].plot(xx[:, 0], mm, color=PAL[0]); axs[0].fill_between(xx[:, 0], mm - 2 * ss, mm + 2 * ss, color=PAL[0], alpha=0.15, lw=0)
axs[0].axvline(0, color="grey", lw=0.6, ls="--"); axs[0].set_xlabel("running variable $R$"); axs[0].set_ylabel("$Y$")
axs[0].set_title("Regression discontinuity", fontsize=10)
axs[1].scatter(xc, yc, s=4, color=PAL[3], alpha=0.5, label="controls")
axs[1].plot(xs[:, 0], m0, color=PAL[0], label="posterior mean of $\\mu_0$")
axs[1].fill_between(xs[:, 0], m0 - 2 * s0, m0 + 2 * s0, color=PAL[0], alpha=0.15, lw=0)
axs[1].plot(xs[:, 0], f0(xs[:, 0]), ":", color=PAL[1], label="true $\\mu_0$")
axs[1].axvspan(0.4, 2, color=PAL[2], alpha=0.15)
axs[1].text(1.2, axs[1].get_ylim()[0] + 0.2 if False else -2.6, "treated only", ha="center", fontsize=8)
axs[1].set_ylim(-3, 3); axs[1].set_xlabel("$x$"); axs[1].legend(frameon=False, fontsize=7, loc="upper left")
axs[1].set_title("Extrapolating a control surface", fontsize=10)
fig.tight_layout(); fig.savefig(OUT + "gp_causal.pdf"); plt.close()
R["gp"] = dict(jump_true=jump, jump_mean=jm, jump_sd=js, sd_inside=float(s0[np.argmin(np.abs(xs[:, 0] + 1))]),
               sd_outside=float(s0[np.argmin(np.abs(xs[:, 0] - 1.5))]))

# ------------------------------------------------ 4. probability of necessity lower bound
rr = np.linspace(1, 70, 500)
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(rr, 1 - 1 / rr, color=PAL[0])
for v, lab, xt, yt in [(9, "all smokers, RR 9", 1.6, 0.97), (60, "heavy smokers, RR 60", 12, 0.70)]:
    ax.plot(v, 1 - 1 / v, "o", color=PAL[1], ms=4); ax.annotate(f"{lab}: {1 - 1 / v:.2f}", (v, 1 - 1 / v), xytext=(xt, yt), fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.5))
ax.axhline(0.5, color="grey", ls=":", lw=0.8); ax.text(12, 0.52, "more probable than not", fontsize=8, color="grey")
ax.set_xscale("log"); ax.set_xlabel("risk ratio"); ax.set_ylabel("lower bound on PN"); ax.set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig(OUT + "pn_bound.pdf"); plt.close()
json.dump(R, open("results/res_figs4.json", "w"), indent=1); print(json.dumps(R, indent=1))

# ------------------------------------------------ 5. DML estimates across random fold partitions (401(k) data)
P = json.load(open("results/res_dml_partitions.json"))
fig, ax = plt.subplots(figsize=(5.6, 3.4))
yy = 0
for key, lab, mk in [("lasso", "lasso", "o"), ("rf", "random forest", "s")]:
    est = np.array(P[key])
    for j, (b, se) in enumerate(est):
        ax.plot([b - 1.96 * se, b + 1.96 * se], [yy, yy], color=PAL[0] if key == "lasso" else PAL[1], alpha=0.5, lw=0.6)
        ax.plot(b, yy, mk, color=PAL[0] if key == "lasso" else PAL[1], ms=3, label=lab if j == 0 else None)
        yy += 1
    yy += 2
ax.set_yticks([]); ax.set_xlabel("effect of 401(k) eligibility, thousands of dollars")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(OUT + "dml_splits.pdf"); plt.close()
