"""Figure did_sim.pdf of Chapter 15 from the replication level results of did_sim2.py (results/did_sim2_reps.json).
Design: the chapter's configuration (10 percent of units depart from parallel trends, effect design a), 500 panels at 40 units and
at 200 units; boxplots of the estimate minus the true average effect on the treated for five estimators.  No random numbers.
Run from the repository root: python book_scripts/did_sim2_fig.py"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
reps = json.load(open("results/did_sim2_reps.json"))
labs = ["TWFE", "cohort\nby time", "imputation", "imputation\n+ trends", "horseshoe"]
order = [0, 1, 3, 4, 2]            # columns 1..5 of a row are twfe, cs, hs, bjs, bjs_trend
fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.0), sharey=True)
for ax, N in zip(axs, (40, 200)):
    A = np.array(reps[f"N{N}_dep0.1_a"])[:, 1:6]
    ax.boxplot([A[:, j] for j in order], tick_labels=labs, showfliers=False)
    ax.axhline(0, color="k", lw=0.6); ax.set_title(f"{N} units, 500 panels", fontsize=9); ax.tick_params(axis="x", labelsize=7)
axs[0].set_ylabel("Estimate minus true ATT")
fig.tight_layout(); fig.savefig("figs/did_sim.pdf"); plt.close()
