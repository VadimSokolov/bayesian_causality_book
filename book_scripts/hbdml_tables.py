#!/usr/bin/env python3
"""Tables for the section "A horseshoe inside double machine learning" of Chapter 8 (label sec:baipw).

Reads the per replication CSV files of the companion simulation, book_scripts/hbdml/results/cross_section_rep*.csv
and panel_rep*.csv (500 replications for the cross section, 2,000 for the panel, produced by sim_cross_section.py and sim_panel.py in book_scripts/hbdml/),
and writes

  results/res_hbdml.json                    n, bias, RMSE, coverage and Monte Carlo standard errors for every cell and method
  book_tex/chapters/hbdml_cs_table.tex      Table tab:hbdml:cs, the cross section grid
  book_tex/chapters/hbdml_panel_table.tex   Table tab:hbdml:panel, the panel with and without MAR dropout

The panel files come from the version of sim_panel.py of 2026-10-23 (cross fitting by unit, unit clustered standard
errors, within OLS row, three dropout conditions, 200 and 1,000 units, 2,000 replications, see its docstring).

Monte Carlo standard errors: bias, sd(error)/sqrt(n); RMSE, sd(error^2)/(2 RMSE sqrt(n)) (delta method);
coverage, sqrt(p(1-p)/n).

Usage from the repository root (nothing is simulated here, the script only reads CSV files):
    timeout 600 python3 -I book_scripts/hbdml_tables.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, "hbdml", "results")
ORIG = os.path.join(HERE, "hbdml", "results_original_panel_baipw.csv")
OUT_JSON = os.path.join(ROOT, "results", "res_hbdml.json")
OUT_CS = os.path.join(ROOT, "book_tex", "chapters", "hbdml_cs_table.tex")
OUT_PANEL = os.path.join(ROOT, "book_tex", "chapters", "hbdml_panel_table.tex")

KE = [5, 25, 50]
DELTA = [0.5, 1.0, 2.0]
CS_METHODS = ["OLS", "LASSO_DML", "RF_DML", "B_DML_HS"]
PANEL_ROWS = [("WITHIN_OLS", "Within OLS"), ("DML_LS_FE", "DML, least squares nuisances"), ("LASSO_DML_FE", "Lasso DML"),
              ("B_DML_HS_FE", "Horseshoe DML"), ("B_AIPW_DML", "Horseshoe DML, estimated weights"),
              ("B_AIPW_ORACLE", "Horseshoe DML, oracle hazard")]

CS_CAPTION = (
    r"Cross section grid of the companion simulation: $n = 1000$, $p = 100$, true effect 1, 500 replications per cell. "
    r"Treatment is Bernoulli with logit $\delta\,k_e^{-1/2}\sum_{j\le k_e}X_j$ and the outcome is "
    r"$Y = T + 5^{-1/2}\sum_{j\le 5}X_j + \varepsilon$. "
    r"OLS is least squares of $Y$ on $T$ and all 100 covariates, which is correctly specified here and gives no interval. "
    r"Lasso, RF and HS are DML with the partialling out score, five fold cross fitting, and lasso, random forest or "
    r"horseshoe nuisance models. Bias is the mean error, RMSE the root mean squared error, and coverage that of nominal "
    r"95\% intervals, whose Monte Carlo standard error is about 0.01 near 0.93."
)


def load(prefix):
    files = sorted(glob.glob(os.path.join(RES, prefix + "_rep*.csv")))
    if not files:
        sys.exit("no %s_rep*.csv files in %s" % (prefix, RES))
    df = pd.concat([pd.read_csv(f, dtype={"covered": str}) for f in files], ignore_index=True)
    df["cov"] = df["covered"].map({"True": 1.0, "False": 0.0})
    return df, len(files)


def stats(g):
    b_all = g["bias"].to_numpy(dtype=float)
    b = b_all[~np.isnan(b_all)]
    n = b.size
    rmse = float(np.sqrt(np.mean(b ** 2)))
    out = {
        "n": int(n),
        "n_missing": int(b_all.size - n),
        "bias": float(b.mean()),
        "bias_mcse": float(b.std(ddof=1) / np.sqrt(n)),
        "rmse": rmse,
        "rmse_mcse": float((b ** 2).std(ddof=1) / (2 * rmse * np.sqrt(n))),
        "sd_error": float(b.std(ddof=1)),
    }
    c = g["cov"].to_numpy(dtype=float)
    c = c[~np.isnan(c)]
    if c.size:
        p = float(c.mean())
        out["coverage"] = p
        out["coverage_mcse"] = float(np.sqrt(p * (1 - p) / c.size))
        out["n_interval"] = int(c.size)
    else:
        out["coverage"] = None
        out["coverage_mcse"] = None
        out["n_interval"] = 0
    s = g["se"].to_numpy(dtype=float)
    s = s[~np.isnan(s)]
    out["mean_se"] = float(s.mean()) if s.size else None
    return out


def f3(x):
    s = "%.3f" % abs(x)
    return ("$-$" + s) if (x < 0 and float(s) != 0.0) else s


def write(path, lines):
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------------------
# cross section
# ----------------------------------------------------------------------------------------------
cs_df, n_cs_files = load("cross_section")
cs = {}
for k in KE:
    for d in DELTA:
        for m in CS_METHODS:
            sub = cs_df[(cs_df.k_e == k) & np.isclose(cs_df.delta, d) & (cs_df.method == m)]
            if sub.empty:
                sys.exit("empty cell %s %s %s" % (k, d, m))
            cs[(k, d, m)] = stats(sub)

cells = [(k, d) for k in KE for d in DELTA]
cs_summary = {
    "ols_smallest_rmse_cells": [list(c) for c in cells
                                if cs[c + ("OLS",)]["rmse"] <= min(cs[c + (m,)]["rmse"] for m in CS_METHODS)],
    "hs_lower_rmse_than_lasso_cells": [list(c) for c in cells
                                       if cs[c + ("B_DML_HS",)]["rmse"] < cs[c + ("LASSO_DML",)]["rmse"]],
    "rf_largest_rmse_cells": [list(c) for c in cells
                              if cs[c + ("RF_DML",)]["rmse"] >= max(cs[c + (m,)]["rmse"] for m in CS_METHODS)],
    "lasso_coverage_below_0.90_cells": [list(c) for c in cells if cs[c + ("LASSO_DML",)]["coverage"] < 0.90],
    "hs_coverage_within_2mcse_of_0.95_cells": [list(c) for c in cells
                                               if abs(cs[c + ("B_DML_HS",)]["coverage"] - 0.95)
                                               <= 2 * cs[c + ("B_DML_HS",)]["coverage_mcse"]],
    "lasso_coverage_within_2mcse_of_0.95_cells": [list(c) for c in cells
                                                  if abs(cs[c + ("LASSO_DML",)]["coverage"] - 0.95)
                                                  <= 2 * cs[c + ("LASSO_DML",)]["coverage_mcse"]],
    "hs_coverage_range": [min(cs[c + ("B_DML_HS",)]["coverage"] for c in cells),
                          max(cs[c + ("B_DML_HS",)]["coverage"] for c in cells)],
    "lasso_coverage_range": [min(cs[c + ("LASSO_DML",)]["coverage"] for c in cells),
                             max(cs[c + ("LASSO_DML",)]["coverage"] for c in cells)],
    "rf_coverage_range": [min(cs[c + ("RF_DML",)]["coverage"] for c in cells),
                          max(cs[c + ("RF_DML",)]["coverage"] for c in cells)],
    "ols_rmse_range": [min(cs[c + ("OLS",)]["rmse"] for c in cells), max(cs[c + ("OLS",)]["rmse"] for c in cells)],
    "reps_per_cell": sorted({cs[c + (m,)]["n"] for c in cells for m in CS_METHODS}),
}

L = [r"\begin{table}[ht]", r"\centering", r"\caption{" + CS_CAPTION + "}", r"\label{tab:hbdml:cs}",
     r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{rrrrrrrrrrr}", r"\toprule",
     r" & & \multicolumn{2}{c}{Bias} & \multicolumn{4}{c}{RMSE} & \multicolumn{3}{c}{Coverage} \\",
     r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}\cmidrule(lr){9-11}",
     r"$k_e$ & $\delta$ & Lasso & HS & OLS & Lasso & RF & HS & Lasso & RF & HS \\", r"\midrule"]
for i, k in enumerate(KE):
    if i:
        L.append(r"\addlinespace")
    for j, d in enumerate(DELTA):
        g = lambda m: cs[(k, d, m)]
        row = [str(k) if j == 0 else "", "%g" % d,
               f3(g("LASSO_DML")["bias"]), f3(g("B_DML_HS")["bias"]),
               f3(g("OLS")["rmse"]), f3(g("LASSO_DML")["rmse"]), f3(g("RF_DML")["rmse"]), f3(g("B_DML_HS")["rmse"]),
               f3(g("LASSO_DML")["coverage"]), f3(g("RF_DML")["coverage"]), f3(g("B_DML_HS")["coverage"])]
        L.append(" & ".join(row) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write(OUT_CS, L)

# ----------------------------------------------------------------------------------------------
# panel
# ----------------------------------------------------------------------------------------------
pn_df, n_pn_files = load("panel")
n_reps_panel = int(pn_df["rep"].nunique())
NU = [200, 1000]
DROPS = [("no_dropout", "no dropout"), ("mar_mild", r"MAR dropout, hazard $\mathrm{expit}(-2 + 0.5\,Y_{t-1})$"),
         ("mar_strong", r"MAR dropout, hazard $\mathrm{expit}(-2 + Y_{t-1})$")]

pn = {}
share = {}
for n_u in NU:
    for dropout, _ in DROPS:
        cond = pn_df[(pn_df.n_units == n_u) & (pn_df.dropout == dropout)]
        if cond.empty:
            continue
        per_rep = cond.groupby("rep")[["obs_rate", "units_used"]].first()
        share[(n_u, dropout)] = (float(per_rep["obs_rate"].mean()), float(per_rep["units_used"].mean()))
        for m, _ in PANEL_ROWS:
            sub = cond[cond.method == m]
            if not sub.empty:
                pn[(n_u, dropout, m)] = stats(sub)

pL = [r"\begin{table}[ht]", r"\centering",
      (r"\caption{Panel simulation of the companion paper: five periods, $p = 50$ covariates "
       r"of which the first 10 drive treatment and the first 5 drive the outcome, unit effects $N(0,1)$, standard normal errors, true effect 1, "
       r"%d replications per block, 200 or 1,000 units. A unit still observed leaves before the next period with probability "
       r"$\mathrm{expit}(-2 + 0.5\,Y_{t-1})$ (mild) or $\mathrm{expit}(-2 + Y_{t-1})$ (strong); the share of unit periods observed is given in each block heading. "
       r"Units observed in fewer than two periods are dropped before demeaning, for every estimator. "
       r"Within OLS is least squares of the demeaned outcome on the demeaned treatment and all 50 demeaned covariates, with no cross fitting. "
       r"The other rows are DML on the demeaned data with nuisance fits by least squares, lasso or horseshoe regression. "
       r"Cross fitting is by unit: each unit's whole history lies in one of five folds, the lasso penalty is tuned on unit folds, and the "
       r"dropout hazard of the weighted rows is fitted on the training units only. "
       r"The weighted rows weight unit periods by the inverse of the probability of still being observed, computed from an estimated horseshoe logistic hazard or the oracle hazard, with each period's hazard clipped to $[0.01, 0.99]$ and the probability of still being observed floored at 0.01, so that no weight exceeds 100. "
       r"Every standard error is clustered by unit, the square root of $G/(G-1)\sum_i S_i^2/(\sum_{it} w_{it}V_{it}^2)^2$ with $G$ the number of units used and $S_i$ the unit sum of the scores. "
       r"Bias is the mean error, SD the standard deviation of the estimates, SE the mean reported standard error, and coverage that of nominal 95\%% intervals, "
       r"whose Monte Carlo standard error is about 0.005 near 0.95.}"
       % n_reps_panel),
      r"\label{tab:hbdml:panel}", r"\scriptsize", r"\begin{tabular}{lccccc}", r"\toprule",
      r"Estimator & Bias & SD & SE & RMSE & Coverage \\"]
for n_u in NU:
    for dropout, title in DROPS:
        if (n_u, dropout) not in share:
            continue
        pL.append(r"\midrule")
        pL.append(r"\multicolumn{6}{l}{\emph{%d units, %s; %.0f percent of unit periods observed}} \\"
                  % (n_u, title, 100 * share[(n_u, dropout)][0]))
        for m, name in PANEL_ROWS:
            if (n_u, dropout, m) not in pn:
                continue
            r = pn[(n_u, dropout, m)]
            pL.append(" & ".join([name, f3(r["bias"]), f3(r["sd_error"]), f3(r["mean_se"]), f3(r["rmse"]), f3(r["coverage"])]) + r" \\")
pL += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write(OUT_PANEL, pL)

# ----------------------------------------------------------------------------------------------
# json
# ----------------------------------------------------------------------------------------------
doc = {
    "meta": {
        "source": "book_scripts/hbdml/results/*.csv",
        "script": "book_scripts/hbdml_tables.py",
        "cross_section_files": n_cs_files,
        "panel_files": n_pn_files,
        "panel_replications": n_reps_panel,
        "panel_observed_share": {"%d_%s" % k: v[0] for k, v in share.items()},
        "panel_units_used_share": {"%d_%s" % k: v[1] for k, v in share.items()},
        "true_effect": 1.0,
        "mcse": "bias: sd/sqrt(n); rmse: sd(e^2)/(2 rmse sqrt(n)); coverage: sqrt(p(1-p)/n)",
    },
    "cross_section_summary": cs_summary,
    "cross_section": [dict(k_e=k, delta=d, method=m, **cs[(k, d, m)]) for k in KE for d in DELTA for m in CS_METHODS],
    "panel": [dict(n_units=k[0], dropout=k[1], method=k[2], **v) for k, v in pn.items()],
}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(doc, fh, indent=1)

# ----------------------------------------------------------------------------------------------
# what the text may say (printed for the editor)
# ----------------------------------------------------------------------------------------------
print("cross section files %d, panel files %d, panel reps %d" % (n_cs_files, n_pn_files, n_reps_panel))
print("k_e delta | bias OLS Lasso RF HS | rmse OLS Lasso RF HS | cov Lasso RF HS | sd(err) Lasso HS | mean se Lasso HS")
for k in KE:
    for d in DELTA:
        g = lambda m: cs[(k, d, m)]
        print("%3d %4.1f | %7.3f %7.3f %7.3f %7.3f | %6.3f %6.3f %6.3f %6.3f | %5.3f %5.3f %5.3f | %6.3f %6.3f | %6.3f %6.3f" % (
            k, d, g("OLS")["bias"], g("LASSO_DML")["bias"], g("RF_DML")["bias"], g("B_DML_HS")["bias"],
            g("OLS")["rmse"], g("LASSO_DML")["rmse"], g("RF_DML")["rmse"], g("B_DML_HS")["rmse"],
            g("LASSO_DML")["coverage"], g("RF_DML")["coverage"], g("B_DML_HS")["coverage"],
            g("LASSO_DML")["sd_error"], g("B_DML_HS")["sd_error"], g("LASSO_DML")["mean_se"], g("B_DML_HS")["mean_se"]))
print("bias MC s.e. (OLS, Lasso, RF, HS) range:",
      [round(min(cs[c + (m,)]["bias_mcse"] for c in cells), 4) for m in CS_METHODS],
      [round(max(cs[c + (m,)]["bias_mcse"] for c in cells), 4) for m in CS_METHODS])
print("coverage MC s.e. HS range %.4f to %.4f, Lasso %.4f to %.4f" % (
    min(cs[c + ("B_DML_HS",)]["coverage_mcse"] for c in cells), max(cs[c + ("B_DML_HS",)]["coverage_mcse"] for c in cells),
    min(cs[c + ("LASSO_DML",)]["coverage_mcse"] for c in cells), max(cs[c + ("LASSO_DML",)]["coverage_mcse"] for c in cells)))
for key, val in cs_summary.items():
    print(key, val)
print("panel (share of unit periods observed, share of units used)")
for (n_u, dr, m), r in pn.items():
    print("N=%4d %-11s %-14s n=%d miss=%d bias %7.4f (mcse %.4f) sd %.4f mean se %.4f rmse %.4f cov %.3f (mcse %.3f) obs %.3f" % (
        n_u, dr, m, r["n"], r["n_missing"], r["bias"], r["bias_mcse"], r["sd_error"], r["mean_se"], r["rmse"], r["coverage"],
        r["coverage_mcse"], share[(n_u, dr)][0]))
print("wrote", OUT_JSON, OUT_CS, OUT_PANEL)
