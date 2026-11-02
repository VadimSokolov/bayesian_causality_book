#!/usr/bin/env python3
"""Aggregate simulation results into tables for the paper.

Usage: python3 aggregate.py [--cs_dir results/] [--panel_dir results/]
"""
import argparse
import glob
import numpy as np
import pandas as pd
import os


def aggregate_cross_section(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "cross_section_rep*.csv")))
    if not files:
        print("No cross-section results found.")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    n_reps = df["rep"].nunique()
    print(f"\n=== Cross-Section Results ({n_reps} reps) ===\n")

    # RMSE table
    pivot = df.groupby(["k_e", "delta", "method"]).agg(
        bias=("bias", "mean"),
        rmse=("bias", lambda x: np.sqrt(np.mean(x ** 2))),
        coverage=("covered", "mean"),
    ).reset_index()

    # Print RMSE table in paper format
    print("Table 1: RMSE by configuration\n")
    print(f"{'k_e':>4} {'delta':>6} | {'OLS':>8} {'LASSO':>8} {'RF':>8} {'HS':>8}")
    print("-" * 55)
    for k_e in [5, 25, 50]:
        for delta in [0.5, 1.0, 2.0]:
            row = {}
            for method in ["OLS", "LASSO_DML", "RF_DML", "B_DML_HS"]:
                r = pivot[(pivot.k_e == k_e) & (pivot.delta == delta) & (pivot.method == method)]
                row[method] = r["rmse"].values[0] if len(r) > 0 else np.nan
            print(f"{k_e:4d} {delta:6.1f} | {row.get('OLS', np.nan):8.3f} {row.get('LASSO_DML', np.nan):8.3f} {row.get('RF_DML', np.nan):8.3f} {row.get('B_DML_HS', np.nan):8.3f}")

    # Coverage table
    print(f"\nTable: 95% CI Coverage\n")
    print(f"{'k_e':>4} {'delta':>6} | {'LASSO':>8} {'RF':>8} {'HS':>8}")
    print("-" * 45)
    for k_e in [5, 25, 50]:
        for delta in [0.5, 1.0, 2.0]:
            row = {}
            for method in ["LASSO_DML", "RF_DML", "B_DML_HS"]:
                r = pivot[(pivot.k_e == k_e) & (pivot.delta == delta) & (pivot.method == method)]
                row[method] = r["coverage"].values[0] if len(r) > 0 else np.nan
            print(f"{k_e:4d} {delta:6.1f} | {row.get('LASSO_DML', np.nan):8.3f} {row.get('RF_DML', np.nan):8.3f} {row.get('B_DML_HS', np.nan):8.3f}")

    # LaTeX table
    print("\n% LaTeX table for paper:")
    print("% \\begin{tabular}{llcccc}")
    print("% \\toprule")
    print("% $k_e$ & $\\delta$ & OLS & LASSO-DML & RF-DML & B-DML-HS \\\\")
    print("% \\midrule")
    for k_e in [5, 25, 50]:
        for delta in [0.5, 1.0, 2.0]:
            vals = []
            for method in ["OLS", "LASSO_DML", "RF_DML", "B_DML_HS"]:
                r = pivot[(pivot.k_e == k_e) & (pivot.delta == delta) & (pivot.method == method)]
                vals.append(f"{r['rmse'].values[0]:.3f}" if len(r) > 0 else "--")
            print(f"% {k_e} & {delta} & {' & '.join(vals)} \\\\")
    print("% \\bottomrule")
    print("% \\end{tabular}")

    return df


def aggregate_panel(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "panel_rep*.csv")))
    if not files:
        print("No panel results found.")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    n_reps = df["rep"].nunique()
    print(f"\n=== Panel Results ({n_reps} reps) ===\n")

    pivot = df.groupby(["dropout", "method"]).agg(
        bias=("bias", "mean"),
        rmse=("bias", lambda x: np.sqrt(np.mean(x ** 2))),
        coverage=("covered", "mean"),
    ).reset_index()

    print(f"{'Dropout':>15} {'Method':>20} | {'Bias':>8} {'RMSE':>8} {'Cov':>8}")
    print("-" * 65)
    for dropout in ["no_dropout", "mar_dropout"]:
        for method in ["OLS_FE", "LASSO_DML_FE", "B_DML_HS_FE", "B_AIPW_DML"]:
            r = pivot[(pivot.dropout == dropout) & (pivot.method == method)]
            if len(r) > 0:
                print(f"{dropout:>15} {method:>20} | {r['bias'].values[0]:8.3f} {r['rmse'].values[0]:8.3f} {r['coverage'].values[0]:8.3f}")

    # LaTeX
    print("\n% LaTeX table for paper:")
    for dropout in ["no_dropout", "mar_dropout"]:
        label = "No dropout ($\\pi = 0$)" if dropout == "no_dropout" else "MAR dropout ($\\pi = 0.4$)"
        print(f"% \\multicolumn{{4}}{{l}}{{\\emph{{{label}}}}} \\\\")
        for method in ["OLS_FE", "LASSO_DML_FE", "B_DML_HS_FE", "B_AIPW_DML"]:
            r = pivot[(pivot.dropout == dropout) & (pivot.method == method)]
            if len(r) > 0:
                name = method.replace("_", "-")
                b = r['bias'].values[0]
                rm = r['rmse'].values[0]
                c = r['coverage'].values[0]
                print(f"% {name} & {b:.3f} & {rm:.3f} & {c:.3f} \\\\")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results")
    args = parser.parse_args()
    aggregate_cross_section(args.dir)
    aggregate_panel(args.dir)
