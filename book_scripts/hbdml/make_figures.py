#!/usr/bin/env python3
"""Generate figures for the HB-DML paper.

Reads simulation results from results/ and produces:
  - fig/rmse_cross_section.pdf  (RMSE by k_e, panels by delta)
  - fig/coverage_cross_section.pdf  (Coverage by k_e, panels by delta)
"""
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "fig")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style: clean, Econometrica-friendly ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

# ── Load cross-section results ──
cs_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "cross_section_rep*.csv")))
print(f"Found {len(cs_files)} cross-section result files")
cs = pd.concat([pd.read_csv(f) for f in cs_files], ignore_index=True)

# Exclude OLS (oracle, not a DML method)
cs_dml = cs[cs["method"] != "OLS"].copy()

# Aggregate: RMSE and coverage by (k_e, delta, method)
def rmse(x):
    return np.sqrt(np.mean(x**2))

agg = cs_dml.groupby(["k_e", "delta", "method"]).agg(
    RMSE=("bias", rmse),
    Coverage=("covered", "mean"),
    n_reps=("rep", "nunique"),
).reset_index()

print(f"Reps per cell: {agg['n_reps'].min()}--{agg['n_reps'].max()}")
print(agg.pivot_table(index=["k_e", "delta"], columns="method", values="RMSE").round(3))

# ── Method display config ──
METHOD_LABELS = {
    "LASSO_DML": "LASSO-DML",
    "RF_DML": "RF-DML",
    "B_DML_HS": "B-DML-HS",
}
METHOD_STYLES = {
    "LASSO_DML": {"color": "0.45", "marker": "s", "linestyle": "--", "linewidth": 1.5},
    "RF_DML":    {"color": "0.7",  "marker": "^", "linestyle": ":",  "linewidth": 1.5},
    "B_DML_HS":  {"color": "0.0",  "marker": "o", "linestyle": "-",  "linewidth": 2.0},
}
METHODS_ORDER = ["LASSO_DML", "RF_DML", "B_DML_HS"]
DELTAS = [0.5, 1.0, 2.0]
KES = [5, 25, 50]

# ── Figure 1: RMSE ──
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)

for ax_idx, delta in enumerate(DELTAS):
    ax = axes[ax_idx]
    sub = agg[agg["delta"] == delta]
    for method in METHODS_ORDER:
        row = sub[sub["method"] == method].sort_values("k_e")
        style = METHOD_STYLES[method]
        ax.plot(row["k_e"], row["RMSE"], label=METHOD_LABELS[method],
                markersize=6, **style)
    ax.set_xlabel("$k_e$ (propensity sparsity)")
    ax.set_title(f"$\\delta = {delta}$")
    ax.set_xticks(KES)

axes[0].set_ylabel("RMSE")
axes[2].legend(loc="upper left", framealpha=0.9)
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(FIG_DIR, "rmse_cross_section.pdf"))
print(f"Saved {os.path.join(FIG_DIR, 'rmse_cross_section.pdf')}")
plt.close(fig)

# ── Figure 2: Coverage ──
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)

for ax_idx, delta in enumerate(DELTAS):
    ax = axes[ax_idx]
    sub = agg[agg["delta"] == delta]
    for method in METHODS_ORDER:
        row = sub[sub["method"] == method].sort_values("k_e")
        style = METHOD_STYLES[method]
        ax.plot(row["k_e"], row["Coverage"], label=METHOD_LABELS[method],
                markersize=6, **style)
    ax.axhline(0.95, color="0.5", linewidth=0.8, linestyle="-", zorder=0)
    ax.set_xlabel("$k_e$ (propensity sparsity)")
    ax.set_title(f"$\\delta = {delta}$")
    ax.set_xticks(KES)
    ax.set_ylim(0, 1.02)

axes[0].set_ylabel("Coverage (nominal 0.95)")
axes[2].legend(loc="lower left", framealpha=0.9)
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(FIG_DIR, "coverage_cross_section.pdf"))
print(f"Saved {os.path.join(FIG_DIR, 'coverage_cross_section.pdf')}")
plt.close(fig)

print("Done.")
