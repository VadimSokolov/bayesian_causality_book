#!/usr/bin/env python3
"""Regenerate the TeX fragments that carry computed numbers into the book, from the saved results in results/*.json.

Usage: python3 tools/make_numbers.py [--out DIR] [--check] [--results DIR]

Without --check each fragment is written to DIR (default book_tex/chapters). With --check nothing is written and the generated text is compared with the files already in DIR. One line per fragment is printed: OK or DIFFERS (with the first differing macro or line) under --check, WROTE (new file, or changed with the first difference) or UNCHANGED otherwise, and NO SOURCE in both modes when the JSON the fragment needs is not saved. The exit status is 1 if --check finds a DIFFERS or a fragment fails, else 0. A file that is already identical is not rewritten.

A fragment whose JSON is missing is never written, so the existing file stays as it is, and its NO SOURCE line names the script to rerun. Rerun that script on the Hopper cluster and copy its JSON into results/ (the multi-hour run_all.sh does not run locally). Warnings that a sentence of prose inside a fragment is no longer supported by the regenerated numbers go to stderr, so stdout stays at one line per fragment.

Sources (JSON file in results/, script in book_scripts/ that writes it, keys read):
- iv_numbers.tex: res_iv.json from iv_card.py (wright, card, weak_one, weak_many) and iv_weak2.py (rf_one, hier_many, card_rf), res_iv_chains.json from iv_chain_check.py, res_iv_hs_r1023.json from iv_horseshoe.py (cells c1.8_s20 and c135_s3); run iv_card.py first.
- iv_hs_table.tex: res_iv_hs_r1023.json from iv_horseshoe.py (six cells c{1.8,20,135}_s{3,20}: tsls, flat, hier, horseshoe).
- shared_numbers.tex: results2.json from make_figs2.py (shared: bias, rmse, cover).
- tauQ_numbers.tex: results2.json from make_figs2.py (tauQ: mean and sd per covariate law).
- baipw_table.tex: res_baipw.json from baipw_rep.py (I, II, III: n, reps, ols, lasso, hs, e0min, e0max).
- baipw_lalonde.tex: res_baipw.json from baipw_rep.py (lalonde: diff, diff_se, ols, lasso, hs); the sample size comes from the header of data/nsw_mixtape.dta.
- castle_table.tex: res_did.json from did_hs.py (castle_twfe, castle_cs, castle_hs_att) and res_castle_w.json from castle_refit.py (twfe_binary, att_cells).
- did_sim_table.tex, did_sim_grid_table.tex and did_sim_dep_table.tex (departures that go with adoption, r1023b): res_did_sim2.json from did_sim2.py (merge step and --merge-dep), the enlarged simulation of r1023 (the 30 panel did_hs.py table is kept in res_did.json).

Counts that the JSON does not store are read from the scripts: the replications of make_figs2.py (shared) and did_hs.py (sim) are the N of the loop `for r in range(N):` that follows `res = []`.
"""
import argparse
import json
import math
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class NoSource(Exception):
    """The saved results (or the script or data they depend on) needed for a fragment are not available."""


def shown(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


class Results:
    """Reads results/*.json once and fetches nested keys, turning any gap into a NoSource that names the script to rerun."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.cache = {}

    def file(self, name, script):
        if name not in self.cache:
            path = self.dir / name
            if not path.exists():
                raise NoSource(f"{shown(path)} is not saved; rerun book_scripts/{script} on Hopper")
            self.cache[name] = json.loads(path.read_text())
        return self.cache[name]

    def get(self, name, script, *keys):
        x = self.file(name, script)
        for k in keys:
            try:
                x = x[k]
            except (KeyError, IndexError, TypeError):
                raise NoSource(f"{name} has no entry {'/'.join(map(str, keys))}; rerun book_scripts/{script} on Hopper") from None
        return x


# ---------------------------------------------------------------- formatting

def dec(x, d):
    """Fixed decimals with a TeX minus: dec(-0.503, 2) is $-$0.50, and a negative that rounds to zero loses its sign."""
    s = f"{x:.{d}f}"
    if s.startswith("-"):
        return s[1:] if float(s) == 0 else "$-$" + s[1:]
    return s


def pct(x):
    return f"{100 * x:.0f}" + "\\%"


def thousands(n):
    return f"{int(n):,}".replace(",", "{,}")


def whole(x, what):
    k = round(x)
    if abs(x - k) > 1e-6:
        raise ValueError(f"{what} is {x}, not a whole number")
    return k


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def fill(template, **values):
    """Replace each <<name>> in the template; the names in the template and in the arguments must match exactly."""
    names = set(re.findall(r"<<(\w+)>>", template))
    if names != set(values):
        raise ValueError(f"template fields and values differ: {sorted(names ^ set(values))}")
    return re.sub(r"<<(\w+)>>", lambda m: values[m.group(1)], template)


def script_reps(script):
    """Replications of the simulation loop of a book script: the N of `for r in range(N):` right after `res = []`."""
    path = ROOT / "book_scripts" / script
    m = re.search(r"^res = \[\]\nfor r in range\((\d+)\):", path.read_text(), re.M) if path.exists() else None
    if not m:
        raise NoSource(f"cannot read the number of replications from book_scripts/{script}")
    return int(m.group(1))


def nsw_size():
    """Rows of data/nsw_mixtape.dta, from the <N> tag of the Stata 117 or later header."""
    path = ROOT / "data" / "nsw_mixtape.dta"
    head = path.read_bytes()[:512] if path.exists() else b""
    order = re.search(rb"<byteorder>(MSF|LSF)</byteorder>", head)
    n = re.search(rb"<N>(.*?)</N>", head, re.S)
    if not (order and n):
        raise NoSource("cannot read the number of units from data/nsw_mixtape.dta")
    return int.from_bytes(n.group(1), "little" if order.group(1) == b"LSF" else "big")


# ---------------------------------------------------------------- iv_numbers.tex

IV_CARD, IV_WEAK = "iv_card.py", "iv_weak2.py"
# macro, script that writes the key, key path in res_iv.json, format (decimals, "%" for percent, "n" for a count with a thousands separator)
IV_MACROS = [
    ("ivwrightols", IV_CARD, ("wright", "ols_slope"), 2),
    ("ivwrightiv", IV_CARD, ("wright", "iv_slope"), 2),
    ("ivcardn", IV_CARD, ("card", "n"), "n"),
    ("ivcardols", IV_CARD, ("card", "ols"), 3),
    ("ivcardolsse", IV_CARD, ("card", "ols_se"), 3),
    ("ivcardfs", IV_CARD, ("card", "first_stage"), 2),
    ("ivcardfsse", IV_CARD, ("card", "first_stage_se"), 2),
    ("ivcardF", IV_CARD, ("card", "F"), 1),
    ("ivcardtsls", IV_CARD, ("card", "tsls"), 3),
    ("ivcardtslsse", IV_CARD, ("card", "tsls_se"), 3),
    ("ivcardrfmed", IV_WEAK, ("card_rf", "q50"), 3),
    ("ivcardrflo", IV_WEAK, ("card_rf", "q05"), 3),
    ("ivcardrfhi", IV_WEAK, ("card_rf", "q95"), 3),
    ("ivcardrfpgt", IV_WEAK, ("card_rf", "p_gt_ols"), 2),
    ("ivcardsmed", IV_CARD, ("card", "bayes", "q50"), 3),
    ("ivcardslo", IV_CARD, ("card", "bayes", "q05"), 3),
    ("ivcardshi", IV_CARD, ("card", "bayes", "q95"), 3),
    ("ivcardspgt", IV_CARD, ("card", "bayes", "p_gt_ols"), 2),
    ("ivcardrho", IV_CARD, ("card", "bayes", "rho_mean"), 2),
    ("ivcardrholo", IV_CARD, ("card", "bayes", "rho_q05"), 2),
    ("ivcardrhohi", IV_CARD, ("card", "bayes", "rho_q95"), 2),
    ("ivweakF", IV_CARD, ("weak_one", "F_med"), 1),
    ("ivmanyF", IV_CARD, ("weak_many", "F_med"), 1),
    ("ivweakols", IV_CARD, ("weak_one", "ols_med"), 2),
    ("ivweakiqrlo", IV_CARD, ("weak_one", "tsls_iqr", 0), 2),
    ("ivweakiqrhi", IV_CARD, ("weak_one", "tsls_iqr", 1), 2),
    ("ivmanytsls", IV_CARD, ("weak_many", "tsls_med"), 2),
    ("ivweakbmed", IV_CARD, ("weak_one", "bayes_med"), 2),
    ("ivweakbcov", IV_CARD, ("weak_one", "bayes_cover"), "%"),
    ("ivmanybcov", IV_CARD, ("weak_many", "bayes_cover"), "%"),
    ("ivhiercov", IV_WEAK, ("hier_many", "cover"), "%"),
    ("ivrflen", IV_WEAK, ("rf_one", "len_med"), 1),
    ("ivrfcov", IV_WEAK, ("rf_one", "cover"), "%"),
    ("ivrfmed", IV_WEAK, ("rf_one", "med"), 2),
    ("ivweakblen", IV_CARD, ("weak_one", "bayes_len_med"), 1),
    ("ivmanybmed", IV_CARD, ("weak_many", "bayes_med"), 2),
    ("ivmanyblen", IV_CARD, ("weak_many", "bayes_len_med"), 2),
    ("ivhiermed", IV_WEAK, ("hier_many", "med"), 2),
    ("ivhierlen", IV_WEAK, ("hier_many", "len_med"), 1),
    ("ivhierrhat", IV_WEAK, ("hier_many", "rhat_max"), 2),
    ("ivscalesmall", IV_CARD, ("weak_one", "prior_scale", "0.01", "bayes_len_med"), 0),
    ("ivscaleten", IV_CARD, ("weak_one", "prior_scale", "1.0", "bayes_len_med"), 1),
    ("ivscalehund", IV_CARD, ("weak_one", "prior_scale", "10.0", "bayes_len_med"), 2),
    ("ivscalehundcov", IV_CARD, ("weak_one", "prior_scale", "10.0", "bayes_cover"), "%"),
    ("ivcardrhopneg", IV_CARD, ("card", "bayes", "rho_p_neg"), 2),
    ("ivcardrhat", IV_CARD, ("card", "bayes", "sampler_rhat_tau"), 3),
    ("ivcardess", IV_CARD, ("card", "bayes", "sampler_ess_tau"), "n"),
    ("ivchexactmed", "iv_chain_check.py", ("weak_one_first_20_data_sets", "exact", "median_of_medians"), 2, "res_iv_chains.json"),
    ("ivchexactlen", "iv_chain_check.py", ("weak_one_first_20_data_sets", "exact", "length_median"), 1, "res_iv_chains.json"),
    ("ivchexactcov", "iv_chain_check.py", ("weak_one_first_20_data_sets", "exact", "coverage"), "%", "res_iv_chains.json"),
    ("ivchblockdiff", "iv_chain_check.py", ("weak_one_first_20_data_sets", "block", "median_abs_diff_from_exact_median"), 2, "res_iv_chains.json"),
    ("ivcholdmed", "iv_chain_check.py", ("weak_one_first_20_data_sets", "old_1200", "median_of_medians"), 2, "res_iv_chains.json"),
    ("ivcholdlen", "iv_chain_check.py", ("weak_one_first_20_data_sets", "old_1200", "length_median"), 2, "res_iv_chains.json"),
    ("ivcholdcov", "iv_chain_check.py", ("weak_one_first_20_data_sets", "old_1200", "coverage"), "%", "res_iv_chains.json"),
    ("ivcholddiff", "iv_chain_check.py", ("weak_one_first_20_data_sets", "old_1200", "median_abs_diff_from_exact_median"), 2, "res_iv_chains.json"),
    ("ivcholdlonglen", "iv_chain_check.py", ("weak_one_first_20_data_sets", "old_20000", "length_median"), 1, "res_iv_chains.json"),
    ("ivcholdlongcov", "iv_chain_check.py", ("weak_one_first_20_data_sets", "old_20000", "coverage"), "%", "res_iv_chains.json"),
    ("ivchcardsd", "iv_chain_check.py", ("card", "exact", "sd"), 3, "res_iv_chains.json"),
    ("ivchcardoldrhat", "iv_chain_check.py", ("card", "old", "rhat"), 2, "res_iv_chains.json"),
    ("ivchcardoldess", "iv_chain_check.py", ("card", "old", "ess"), 0, "res_iv_chains.json"),
    ("ivchcardoldlag", "iv_chain_check.py", ("card", "old", "lag1_autocorrelation"), 3, "res_iv_chains.json"),
    ("ivchcardoldstep", "iv_chain_check.py", ("card", "old", "sd_of_one_step_change"), 3, "res_iv_chains.json"),
    ("ivscaledeflen", IV_CARD, ("weak_one", "prior_scale", "0.1", "bayes_len_med"), 1),
    ("ivscaledefcov", IV_CARD, ("weak_one", "prior_scale", "0.1", "bayes_cover"), "%"),
    ("ivscalecovsmall", IV_CARD, ("weak_one", "prior_scale", "0.01", "bayes_cover"), "%"),
    ("ivscalecovone", IV_CARD, ("weak_one", "prior_scale", "1.0", "bayes_cover"), "%"),
    ("ivscalemedsmall", IV_CARD, ("weak_one", "prior_scale", "0.01", "bayes_med"), 2),
    ("ivscalemedhund", IV_CARD, ("weak_one", "prior_scale", "10.0", "bayes_med"), 2),
    # twenty instruments, weak design (hw) and sparse design (hs): res_iv_hs_r1023.json from iv_horseshoe.py (cover is [value, se]; the share of converged chains is 1 minus share_rhat_above_1_01_final)
    ("ivhwtsls", "iv_horseshoe.py", ("c1.8_s20", "tsls", "med"), 2, "res_iv_hs_r1023.json"),
    ("ivhwflatcov", "iv_horseshoe.py", ("c1.8_s20", "flat", "cover"), "%", "res_iv_hs_r1023.json"),
    ("ivhwflatlen", "iv_horseshoe.py", ("c1.8_s20", "flat", "len_med"), 2, "res_iv_hs_r1023.json"),
    ("ivhwhiermed", "iv_horseshoe.py", ("c1.8_s20", "hier", "med"), 2, "res_iv_hs_r1023.json"),
    ("ivhwhiercov", "iv_horseshoe.py", ("c1.8_s20", "hier", "cover"), "%", "res_iv_hs_r1023.json"),
    ("ivhwhierlen", "iv_horseshoe.py", ("c1.8_s20", "hier", "len_med"), 1, "res_iv_hs_r1023.json"),
    ("ivhwhierrhat", "iv_horseshoe.py", ("c1.8_s20", "hier", "rhat_max"), 2, "res_iv_hs_r1023.json"),
    ("ivhwhiersh", "iv_horseshoe.py", ("c1.8_s20", "hier", "share_rhat_below_1_01"), "%1", "res_iv_hs_r1023.json"),
    ("ivhwhsmed", "iv_horseshoe.py", ("c1.8_s20", "horseshoe", "med"), 2, "res_iv_hs_r1023.json"),
    ("ivhwhscov", "iv_horseshoe.py", ("c1.8_s20", "horseshoe", "cover"), "%", "res_iv_hs_r1023.json"),
    ("ivhwhslen", "iv_horseshoe.py", ("c1.8_s20", "horseshoe", "len_med"), 1, "res_iv_hs_r1023.json"),
    ("ivhwhsrhat", "iv_horseshoe.py", ("c1.8_s20", "horseshoe", "rhat_max"), 2, "res_iv_hs_r1023.json"),
    ("ivhwhssh", "iv_horseshoe.py", ("c1.8_s20", "horseshoe", "share_rhat_below_1_01"), "%1", "res_iv_hs_r1023.json"),
    ("ivhstsls", "iv_horseshoe.py", ("c135_s3", "tsls", "med"), 2, "res_iv_hs_r1023.json"),
    ("ivhsflatmed", "iv_horseshoe.py", ("c135_s3", "flat", "med"), 2, "res_iv_hs_r1023.json"),
    ("ivhsflatcov", "iv_horseshoe.py", ("c135_s3", "flat", "cover"), "%", "res_iv_hs_r1023.json"),
    ("ivhsflatlen", "iv_horseshoe.py", ("c135_s3", "flat", "len_med"), 2, "res_iv_hs_r1023.json"),
    ("ivhshiermed", "iv_horseshoe.py", ("c135_s3", "hier", "med"), 2, "res_iv_hs_r1023.json"),
    ("ivhshiercov", "iv_horseshoe.py", ("c135_s3", "hier", "cover"), "%", "res_iv_hs_r1023.json"),
    ("ivhshierlen", "iv_horseshoe.py", ("c135_s3", "hier", "len_med"), 2, "res_iv_hs_r1023.json"),
    ("ivhshierrhat", "iv_horseshoe.py", ("c135_s3", "hier", "rhat_max"), 2, "res_iv_hs_r1023.json"),
    ("ivhshsmed", "iv_horseshoe.py", ("c135_s3", "horseshoe", "med"), 2, "res_iv_hs_r1023.json"),
    ("ivhshscov", "iv_horseshoe.py", ("c135_s3", "horseshoe", "cover"), "%", "res_iv_hs_r1023.json"),
    ("ivhshslen", "iv_horseshoe.py", ("c135_s3", "horseshoe", "len_med"), 2, "res_iv_hs_r1023.json"),
    ("ivhsrhat", "iv_horseshoe.py", ("c135_s3", "horseshoe", "rhat_max"), 2, "res_iv_hs_r1023.json"),
]


def build_iv(R):
    lines = []
    for macro, script, keys, how, *rest in IV_MACROS:
        fname = rest[0] if rest else "res_iv.json"
        if fname == IV_HS_FILE:
            x = hs_value(lambda *k: R.get(fname, script, *k), keys)
        else:
            x = R.get(fname, script, *keys)
        value = pct1(x) if how == "%1" else pct(x) if how == "%" else thousands(x) if how == "n" else dec(x, how)
        lines.append(f"\\newcommand{{\\{macro}}}{{{value}}}")
    return "\n".join(lines) + "\n", []


# ---------------------------------------------------------------- iv_hs_table.tex

IV_HS_FILE = "res_iv_hs_r1023.json"  # six cells c{concentration}_s{active}, 1,000 data sets each; cover and bias are [value, se]


def hs_value(g, keys):
    """A value of IV_HS_FILE: the coverage is the first of [value, se]; the share of converged chains is 1 minus the share above 1.01."""
    if keys[-1] == "cover":
        return g(*keys)[0]
    if keys[-1] == "share_rhat_below_1_01":
        return 1 - g(*keys[:-1], "share_rhat_above_1_01_final")
    return g(*keys)


def half_up(x, d):
    """Round half up on the decimal text of x (0.945 to one decimal place of a percent is 94.5 and gives 95)."""
    return Decimal(repr(round(x, 12))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)


def pct1(x):
    return f"{half_up(100 * x, 1)}" + "\\%"


IV_HS_TABLE = r"""\begin{table}[ht]
\centering
\caption{Twenty instruments, <<reps>> data sets per design, $n = 2{,}000$, $\tau = 0.1$ and error correlation 0.6. The concentration parameter $c$ is carried by $s$ active instruments, each with first stage coefficient $\sqrt{c/(ns)}$; the weak design of the text is $c = 1.8$ with $s = 20$, the sparse design $c = 135$ with $s = 3$. For two stage least squares the median estimate over data sets; for each first stage prior the median over data sets of the posterior median, the share of equal tailed 90\% intervals that cover the truth, and the median interval length. The Monte Carlo standard error of a coverage near 90\% is about one point.}
\label{tab:ivmany}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{Three active, $s = 3$} & \multicolumn{3}{c}{Twenty active, $s = 20$} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
First stage prior & Median & Coverage & Length & Median & Coverage & Length \\
\midrule
<<rows>>
\bottomrule
\end{tabular}
\end{table}
"""


def build_iv_hs_table(R):
    g = lambda *k: R.get(IV_HS_FILE, "iv_horseshoe.py", *k)
    reps = g("c1.8_s3", "R")
    m = lambda x: dec(x, 2).replace("$-$", "$-") + "$" if x < -0.005 else dec(x, 2)
    cells_by_c = [("1.8", "1.8"), ("20", "20"), ("135", "135")]
    rows = []
    for c, label in cells_by_c:
        if rows:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{7}{@{}l}{\textit{Concentration $c = " + label + r"$}} \\")
        for name, key in [("Two stage least squares", "tsls"), ("Diffuse, exact", "flat"), ("Hierarchical normal", "hier"), ("Horseshoe", "horseshoe")]:
            cells = []
            for s in (3, 20):
                cell = f"c{c}_s{s}"
                if g(cell, "R") != reps:
                    raise ValueError("the cells have different numbers of data sets")
                e = g(cell, key)
                cover = f"{half_up(100 * e['cover'][0], 0)}\\%" if key != "tsls" else ""
                cells += [m(e["med"]), "", ""] if key == "tsls" else [m(e["med"]), cover, dec(e["len_med"], 2)]
            rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    return fill(IV_HS_TABLE, reps=f"{reps:,}", rows="\n".join(rows)), []


# ---------------------------------------------------------------- shared_numbers.tex

SHARED = r"""Figure~\ref{fig:shared} and the table below summarize the result.

\begin{center}
\begin{tabular}{lccc}
\toprule
Prior & Bias & RMSE & 95\% coverage \\
\midrule
<<rows>>
\bottomrule
\end{tabular}
\end{center}
"""


def build_shared(R):
    get = lambda key: R.get("results2.json", "make_figs2.py", "shared", key)
    bias, rmse, cover = get("bias"), get("rmse"), get("cover")
    reps = script_reps("make_figs2.py")
    rows = []
    for i, name in enumerate(["Independent scales", "Shared scales"]):
        k = whole(cover[i] * reps, "shared coverage count")
        rows.append(rf"{name} & {dec(bias[i], 3)} & {dec(rmse[i], 3)} & {k} of {reps} \\")
    return fill(SHARED, rows="\n".join(rows)), []


# ---------------------------------------------------------------- tauQ_numbers.tex

TAUQ = r"""The posterior means and standard deviations are <<sample_m>> and <<sample_sd>> for the sample, <<treated_m>> and <<treated_sd>> for the treated units, and <<older_m>> and <<older_sd>> for the older cohort. The effect for the sample is clearly positive; the effect for the older cohort is indistinguishable from zero.
"""


def build_tauq(R):
    q = {k: R.get("results2.json", "make_figs2.py", "tauQ", k) for k in ("sample", "treated", "older cohort")}
    text = fill(TAUQ,
                sample_m=dec(q["sample"][0], 3), sample_sd=dec(q["sample"][1], 3),
                treated_m=dec(q["treated"][0], 3), treated_sd=dec(q["treated"][1], 3),
                older_m=dec(q["older cohort"][0], 3), older_sd=dec(q["older cohort"][1], 3))
    warns = []
    if not q["sample"][0] > 2 * q["sample"][1]:
        warns.append("the sample effect is no longer more than 2 sd above zero, but the text says it is clearly positive")
    if not abs(q["older cohort"][0]) < 2 * q["older cohort"][1]:
        warns.append("the older cohort effect is now at least 2 sd from zero, but the text says it is indistinguishable from zero")
    return text, warns


# ---------------------------------------------------------------- baipw_table.tex

BAIPW_TABLE = r"""\begin{table}[ht]
\centering
\caption{Three designs of our own, <<reps>> replications each, true effect 1. OLS is least squares on all covariates. Coverage of nominal 95\% intervals is a count out of <<reps>>; $w_{99}$ is the 99th percentile inverse weight and ESS the effective sample size of the weights, both averaged over replications.}
\label{tab:baipw}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccccc}
\toprule
Design & Estimator & Bias & RMSE & Coverage & $w_{99}$ & ESS \\
\midrule
<<rows>>
\bottomrule
\end{tabular}
\end{table}

\paragraph{Sparse confounding.} In Design I (Table~\ref{tab:baipw}) the horseshoe version is nearly unbiased, <<I_hs_bias>>, covers in <<I_hs_cover>> of <<reps>> replications and has error <<I_hs_rmse>> against <<I_lasso_rmse>> for the lasso version. Most of the lasso version's error is variance, a bias of <<I_lasso_bias>> against an RMSE of <<I_lasso_rmse>>, which fits its heavier weights: a 99th percentile of <<I_lasso_w99>> against <<I_hs_w99>> and an effective sample size of <<I_lasso_ess>> against <<I_hs_ess>>.

\paragraph{Dense confounding in both equations.} Design II is where the horseshoe fails. The lasso version breaks down through extreme weights (99th percentile <<II_lasso_w99>>), with bias <<II_lasso_bias>> and RMSE <<II_lasso_rmse>>, though its intervals, <<II_lasso_len>> wide on average, still cover in <<II_lasso_cover>> of <<reps>> replications. The horseshoe version has bias <<II_hs_bias>> and covers in <<II_hs_cover>> of <<reps>>. Its weights, with a 99th percentile of <<II_hs_w99>> and an effective sample size of <<II_hs_ess>> out of <<II_n>>, are those of a propensity shrunk close to a constant, and with thirty weak coefficients in the outcome equation as well, neither nuisance removes the confounding. The grid above also has a dense propensity, up to fifty covariates, and the horseshoe copes. What differs is the outcome, which has five active covariates there and thirty here. In our experiments the horseshoe's trouble is a dense outcome, not a dense propensity. The dense design of Chapter~\ref{ch:critique}, with every coordinate nonzero, points the same way: DML with a horseshoe nuisance is biased by 0.68 there and covers in none of 500 replications (Table~\ref{tab:crit:head}). The grid and Design II also differ in the score and in the sample size, 1,000 against <<II_n>>, so we read the contrast as a hypothesis for a designed test and not as a finding.

\paragraph{Weak overlap.} In Design III the horseshoe version has the better coverage, <<III_hs_cover>> of <<reps>> against <<III_lasso_cover>> of <<reps>>, and the lower error, <<III_hs_rmse>> against <<III_lasso_rmse>>, but heavier weights, a 99th percentile of <<III_hs_w99>> against <<III_lasso_w99>>. Its estimated propensities range from <<III_emin>> to <<III_emax>>, against true values from <<III_e0min>> to <<III_e0max>>. The weight bound does not hold in finite samples, as explained above, and here it is the lasso's stronger shrinkage of the logit, not the horseshoe's, that keeps the weights small.

\paragraph{The benchmark.} In all three designs the outcome is linear in the covariates, so least squares with all covariates is correctly specified, and it has the smallest error in every one: <<I_ols_rmse>>, <<II_ols_rmse>> and <<III_ols_rmse>>. Horseshoe nuisances do not substitute for a correctly specified outcome model.
"""


def build_baipw_table(R):
    g = lambda *k: R.get("res_baipw.json", "baipw_rep.py", *k)
    kinds = ("I", "II", "III")
    all_reps = {k: g(k, "reps") for k in kinds}
    if len(set(all_reps.values())) != 1:
        raise ValueError(f"the designs have different numbers of replications: {all_reps}")
    reps = all_reps["I"]
    d = {k: {"ols": {f: g(k, "ols", f) for f in ("bias", "rmse")},
             "lasso": {f: g(k, "lasso", f) for f in ("bias", "rmse", "cover", "length", "w99", "ess")},
             "hs": {f: g(k, "hs", f) for f in ("bias", "rmse", "cover", "length", "w99", "ess", "emin", "emax")}}
         for k in kinds}
    count = lambda e: whole(e["cover"] * reps, "coverage count")
    blocks = []
    for k in kinds:
        block = [rf"{k} & OLS & {dec(d[k]['ols']['bias'], 3)} & {dec(d[k]['ols']['rmse'], 3)} & & & \\"]
        for label, who in (("AIPW, lasso", "lasso"), ("AIPW, horseshoe", "hs")):
            e = d[k][who]
            block.append(rf" & {label} & {dec(e['bias'], 3)} & {dec(e['rmse'], 3)} & {count(e)}/{reps} & {dec(e['w99'], 1)} & {dec(e['ess'], 0)} \\")
        blocks.append("\n".join(block))
    I, II, III = d["I"], d["II"], d["III"]
    text = fill(BAIPW_TABLE,
                reps=str(reps), rows="\n\\midrule\n".join(blocks),
                I_hs_bias=dec(I["hs"]["bias"], 3), I_hs_cover=str(count(I["hs"])), I_hs_rmse=dec(I["hs"]["rmse"], 3),
                I_lasso_rmse=dec(I["lasso"]["rmse"], 3), I_lasso_bias=dec(I["lasso"]["bias"], 3),
                I_lasso_w99=dec(I["lasso"]["w99"], 1), I_hs_w99=dec(I["hs"]["w99"], 1),
                I_lasso_ess=dec(I["lasso"]["ess"], 0), I_hs_ess=dec(I["hs"]["ess"], 0),
                II_lasso_w99=dec(II["lasso"]["w99"], 1), II_lasso_bias=dec(II["lasso"]["bias"], 3), II_lasso_rmse=dec(II["lasso"]["rmse"], 3),
                II_lasso_len=dec(II["lasso"]["length"], 1), II_lasso_cover=str(count(II["lasso"])),
                II_hs_bias=dec(II["hs"]["bias"], 3), II_hs_cover=str(count(II["hs"])), II_hs_w99=dec(II["hs"]["w99"], 1),
                II_hs_ess=dec(II["hs"]["ess"], 0), II_n=str(g("II", "n")),
                III_hs_cover=str(count(III["hs"])), III_lasso_cover=str(count(III["lasso"])),
                III_hs_rmse=dec(III["hs"]["rmse"], 3), III_lasso_rmse=dec(III["lasso"]["rmse"], 3),
                III_hs_w99=dec(III["hs"]["w99"], 1), III_lasso_w99=dec(III["lasso"]["w99"], 1),
                III_emin=dec(III["hs"]["emin"], 4), III_emax=dec(III["hs"]["emax"], 4),
                III_e0min=dec(g("III", "e0min"), 4), III_e0max=dec(g("III", "e0max"), 4),
                I_ols_rmse=dec(I["ols"]["rmse"], 3), II_ols_rmse=dec(II["ols"]["rmse"], 3), III_ols_rmse=dec(III["ols"]["rmse"], 3))
    warns = []
    if not all(d[k]["ols"]["rmse"] < min(d[k]["lasso"]["rmse"], d[k]["hs"]["rmse"]) for k in kinds):
        warns.append("OLS no longer has the smallest RMSE in every design, but the benchmark paragraph says it does")
    if not (I["hs"]["rmse"] < I["lasso"]["rmse"] and I["hs"]["w99"] < I["lasso"]["w99"] and I["hs"]["ess"] > I["lasso"]["ess"]):
        warns.append("Design I: the horseshoe no longer has smaller error, smaller weights and a larger effective sample size than the lasso, but the text says it does")
    if not I["lasso"]["bias"] ** 2 < I["lasso"]["rmse"] ** 2 / 2:
        warns.append("Design I: most of the lasso error is no longer variance, but the text says it is")
    if not (II["lasso"]["rmse"] > 1 and II["lasso"]["w99"] > 20 and II["hs"]["cover"] < 0.5):
        warns.append("Design II: the text says the lasso broke down through extreme weights and the horseshoe fails (RMSE above 1, 99th percentile weight above 20, horseshoe coverage below one half), and the numbers no longer show all three")
    if not (III["hs"]["cover"] > III["lasso"]["cover"] and III["hs"]["rmse"] < III["lasso"]["rmse"] and III["hs"]["w99"] > III["lasso"]["w99"]):
        warns.append("Design III: the horseshoe no longer has better coverage, lower error and heavier weights than the lasso, but the text says it does")
    return text, warns


# ---------------------------------------------------------------- baipw_lalonde.tex

BAIPW_LALONDE = r"""\begin{table}[ht]
\centering
\caption{The randomized NSW sample, <<n>> units, 1978 earnings in dollars, eight standardized covariates. The AIPW rows give the range over <<folds>> random fold assignments.}
\label{tab:baipwlalonde}
\small
\begin{tabular}{lcc}
\toprule
Estimator & Estimate & Standard error \\
\midrule
Difference in means & <<diff>> & <<diff_se>> \\
Least squares with covariates & <<ols>> & \\
AIPW, lasso nuisances & <<lasso_est>> & <<lasso_se>> \\
AIPW, horseshoe nuisances & <<hs_est>> & <<hs_se>> \\
\bottomrule
\end{tabular}
\end{table}

All estimators agree with the experimental difference in means to within sampling error, and their standard errors are close to the experimental one, <<se_lo>> to <<se_hi>> dollars against <<diff_se>>. Randomization makes the unadjusted difference unbiased, and the covariates turn out to be only weakly prognostic for 1978 earnings, so adjustment barely changes precision. The observational version of the problem, with survey controls, is the one in which the choice of estimator matters, and Section~\ref{sec:lalonde} treats it.
"""


def build_baipw_lalonde(R):
    g = lambda *k: R.get("res_baipw.json", "baipw_rep.py", "lalonde", *k)
    diff, diff_se, ols = g("diff"), g("diff_se"), g("ols")
    runs = {who: g(who) for who in ("lasso", "hs")}  # per fold assignment: [estimate, standard error, 99th percentile weight]
    if len(runs["lasso"]) != len(runs["hs"]):
        raise ValueError("lasso and horseshoe have different numbers of fold assignments")
    est = {who: [r[0] for r in runs[who]] for who in runs}
    se = {who: [r[1] for r in runs[who]] for who in runs}
    span = lambda v: f"{min(v):.0f} to {max(v):.0f}"
    folds = len(runs["lasso"])
    text = fill(BAIPW_LALONDE,
                n=str(nsw_size()), folds=WORDS.get(folds, str(folds)),
                diff=f"{diff:.0f}", diff_se=f"{diff_se:.0f}", ols=f"{ols:.0f}",
                lasso_est=span(est["lasso"]), lasso_se=span(se["lasso"]),
                hs_est=span(est["hs"]), hs_se=span(se["hs"]),
                se_lo=f"{min(se['lasso'] + se['hs']):.0f}", se_hi=f"{max(se['lasso'] + se['hs']):.0f}")
    warns = []
    if any(abs(e - diff) > 2 * diff_se for e in [ols] + est["lasso"] + est["hs"]):
        warns.append("an estimate is more than two standard errors from the difference in means, but the text says all estimators agree within sampling error")
    if any(abs(s / diff_se - 1) > 0.15 for s in se["lasso"] + se["hs"]):
        warns.append("a standard error is more than 15 percent from the experimental one, but the text says they are all close to it")
    try:
        exp = R.get("res_lalonde.json", "lalonde_book.py", "experimental")
        if abs(exp - diff) > 0.5:
            warns.append(f"the difference in means {diff:.2f} differs from the experimental value {exp:.2f} saved in res_lalonde.json")
    except NoSource:
        pass
    return text, warns


# ---------------------------------------------------------------- castle_table.tex and did_sim_table.tex

CASTLE = r"""\begin{table}[ht]
\centering
\caption{Effect of castle doctrine laws on the log homicide rate.}
\label{tab:castle}
\small
\begin{tabular}{lcc}
\toprule
Estimator & Estimate & Interval \\
\midrule
Two way fixed effects & <<twfe>> & \\
Cohort by time, never treated controls & <<cs>> & \\
Horseshoe difference in differences, equal noise & <<hs>> & [<<lo>>, <<hi>>] \\
\addlinespace
Two way fixed effects, population weights & <<twfe_w>> & \\
Horseshoe, population weights & <<hs_w>> & [<<lo_w>>, <<hi_w>>] \\
\bottomrule
\end{tabular}
\end{table}
"""

DID_SIM = r"""\begin{table}[ht]
\centering
\caption{Simulation results for the average effect on the treated in the chapter's design (<<dep>> of units depart from parallel trends, effects growing with exposure): <<panels>> panels per column group and the estimators on the same panels. Cohort by time uses never treated controls, Imputation is the estimator of Borusyak, Jaravel and Spiess (2024), the next row adds a linear trend for every unit, and Horseshoe is the estimator of this chapter. Coverage and width are for the 95 percent interval and are shown only for the estimators that have one. Monte Carlo standard errors are at most <<mcse_bias>> for the biases and <<mcse_cover>> for the coverages.}
\label{tab:didsim}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcccccccc@{}}
\toprule
& \multicolumn{4}{c}{40 units} & \multicolumn{4}{c}{200 units} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Estimator & Bias & RMSE & Cover & Width & Bias & RMSE & Cover & Width \\
\midrule
<<rows>>
\bottomrule
\end{tabular}
\end{table}
"""

DID_GRID = r"""\begin{table}[ht]
\centering
\caption{Root mean squared error of five estimators of the average effect on the treated over <<panels>> panels per row, and coverage of the 95 percent interval of the three that have one. TWFE is two way fixed effects, Cohort the cohort by time estimator, Imput.\ the imputation estimator, Trend the imputation estimator with unit trends and HS the horseshoe. $N$ is the number of units, Dep the share of units whose trend departs from the common path, and the effect design is (a) the common path, (b) the path with one sparse cohort deviation, (c) the path with dense cohort deviations.}
\label{tab:didsimgrid}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}rclcccccccc@{}}
\toprule
& & & \multicolumn{5}{c}{Root mean squared error} & \multicolumn{3}{c}{Coverage} \\
\cmidrule(lr){4-8}\cmidrule(lr){9-11}
$N$ & Dep & Design & TWFE & Cohort & Imput. & Trend & HS & HS & Imput. & Trend \\
\midrule
<<rows>>
\bottomrule
\end{tabular}
\end{table}
"""


def build_castle(R):
    g = lambda *k: R.get("res_did.json", "did_hs.py", *k)
    w = lambda *k: R.get("res_castle_w.json", "castle_refit.py", *k)
    mean, _sd, lo, hi = g("castle_hs_att")  # posterior mean, sd, 2.5 and 97.5 percent points of the horseshoe ATT
    text = fill(CASTLE, twfe=dec(g("castle_twfe"), 3), cs=dec(g("castle_cs"), 3),
                hs=dec(mean, 3), lo=dec(lo, 3), hi=dec(hi, 3),
                twfe_w=dec(w("twfe_binary"), 3), hs_w=dec(w("att_cells", "mean"), 3),
                lo_w=dec(w("att_cells", "lo"), 3), hi_w=dec(w("att_cells", "hi"), 3))
    return text, []


def _sim2(R):
    return {k: v for k, v in R.file("res_did_sim2.json", "did_sim2.py").items()}


def build_did_sim(R):
    S = _sim2(R)
    names = [("twfe", "Two way fixed effects"), ("cs", "Cohort by time"), ("bjs", "Imputation"),
             ("bjs_trend", "Imputation, unit trends"), ("hs", "Horseshoe")]
    rows = []
    for m, nm in names:
        cells = []
        for N in (40, 200):
            d = S[f"N{N}_dep0.1_a"][m]
            cells += [dec(d["bias"], 2), dec(d["rmse"], 2), dec(d["cover"], 2) if "cover" in d else "", f"{d['width']:.2f}" if "width" in d else ""]
        rows.append(f"{nm} & " + " & ".join(cells) + r" \\")
    n = S["N40_dep0.1_a"]["n_panels"]
    mb = max(S[k][m]["bias_se"] for k in S for m in ("twfe", "cs", "bjs", "bjs_trend", "hs") if k.endswith("dep0.1_a"))
    mc = max(S[k][m]["cover_se"] for k in S for m in ("bjs", "bjs_trend", "hs") if k.endswith("dep0.1_a"))
    return fill(DID_SIM, dep="10 percent", panels=str(n), mcse_bias=f"{mb:.3f}", mcse_cover=f"{mc:.3f}", rows="\n".join(rows)), []


def build_did_grid(R):
    S = _sim2(R)
    rows = []
    for N in (40, 200):
        for dep in ("0.0", "0.1", "0.3"):
            for des in "abc":
                d = S[f"N{N}_dep{dep}_{des}"]
                rows.append(f"{N} & {dep} & ({des}) & " + " & ".join(f"{d[m]['rmse']:.2f}" for m in ("twfe", "cs", "bjs", "bjs_trend", "hs")) + " & "
                            + " & ".join(f"{d[m]['cover']:.2f}" for m in ("hs", "bjs", "bjs_trend")) + r" \\")
            rows.append(r"\addlinespace" if (N, dep) != (200, "0.3") else "")
    rows = [r for r in rows if r]
    return fill(DID_GRID, panels=str(S["N40_dep0.0_a"]["n_panels"]), rows="\n".join(rows)), []


DID_DEP = r"""\begin{table}[ht]
\centering
\caption{Departures from parallel trends that go with adoption: <<panels>> panels per block, effects growing with exposure (design (a)), the same five estimators on the same panels. In the first two blocks a unit departs with probability 0.3 if it adopts and never if it is never treated, so about 22 percent of all units depart; <<sparse>>in the last two a unit departs with probability 0.3 if it is in the period 12 cohort and never otherwise. Bias and root mean squared error are for the average effect on the treated, coverage is for the 95 percent interval, and the last column is the mean of the squared error of the estimator minus that of the horseshoe on the same panel. Monte Carlo standard errors are in parentheses.}
\label{tab:didsimdep}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
Estimator & Bias & RMSE & Cover & Squared error minus HS \\
\midrule
<<rows>>
\bottomrule
\end{tabular}
\end{table}
"""


def build_did_dep(R):
    S = _sim2(R)
    names = [("twfe", "Two way fixed effects"), ("cs", "Cohort by time"), ("bjs", "Imputation"),
             ("bjs_trend", "Imputation, unit trends"), ("hs", "Horseshoe")]
    rows = []; first = True
    deps = [("adopt0.3", "Adopters depart with probability 0.3")]; sparse = ""
    if "N40_depadopt0.1_a" in S:      # sparse case added in r1023c4
        deps.append(("adopt0.1", "Adopters depart with probability 0.1"))
        sparse = "in the next two a unit departs with probability 0.1 if it adopts, so about 7.5 percent of units depart and all of them are treated; "
    deps.append(("c12_0.3", "Period 12 cohort departs with probability 0.3"))
    for dep, lab in deps:
        for N in (40, 200):
            d = S[f"N{N}_dep{dep}_a"]
            if not first: rows.append(r"\addlinespace")
            first = False
            rows.append(r"\multicolumn{5}{@{}l}{\emph{%s, %d units}} \\" % (lab, N))
            for m, nm in names:
                e = d[m]; pd_ = d["paired_sq_error_minus_hs"].get(m)
                cov = f"{e['cover']:.2f} ({e['cover_se']:.2f})" if "cover" in e else ""
                pr = f"{pd_['mean']:.4f} ({pd_['se']:.4f})" if pd_ else ""
                rows.append(f"{nm} & {e['bias']:.3f} ({e['bias_se']:.3f}) & {e['rmse']:.3f} & {cov} & {pr}" + r" \\")
    return fill(DID_DEP, panels=str(S["N40_depadopt0.3_a"]["n_panels"]), rows="\n".join(rows), sparse=sparse), []


FRAGMENTS = [
    ("iv_numbers.tex", build_iv),
    ("iv_hs_table.tex", build_iv_hs_table),
    ("shared_numbers.tex", build_shared),
    ("tauQ_numbers.tex", build_tauq),
    ("baipw_table.tex", build_baipw_table),
    ("baipw_lalonde.tex", build_baipw_lalonde),
    ("castle_table.tex", build_castle),
    ("did_sim_table.tex", build_did_sim),
    ("did_sim_grid_table.tex", build_did_grid),
    ("did_sim_dep_table.tex", build_did_dep),
]


# ---------------------------------------------------------------- comparison and main

MACRO = re.compile(r"^\\newcommand\{\\(\w+)\}\{(.*)\}$")


def snippet(s, other, width=30):
    if s is None:
        return "nothing"
    j = 0
    if other is not None:
        while j < min(len(s), len(other)) and s[j] == other[j]:
            j += 1
    lo = max(0, j - width)
    return '"' + ("..." if lo else "") + s[lo:j + width] + ("..." if j + width < len(s) else "") + '"'


def first_diff(old, new):
    """Describe the first differing macro (for \\newcommand files) or line."""
    a, b = old.split("\n"), new.split("\n")
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else None
        y = b[i] if i < len(b) else None
        if x == y:
            continue
        mx, my = MACRO.match(x or ""), MACRO.match(y or "")
        if mx and my and mx.group(1) == my.group(1):
            return f"\\{mx.group(1)} is {mx.group(2)} in the file and {my.group(2)} when generated"
        return f"line {i + 1}: the file has {snippet(x, y)} and the generated text has {snippet(y, x)}"
    return "the texts differ"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Regenerate the TeX number fragments of the book from results/*.json.")
    ap.add_argument("--out", default=str(ROOT / "book_tex" / "chapters"), help="directory of the fragments (default book_tex/chapters)")
    ap.add_argument("--check", action="store_true", help="compare with the files in --out and write nothing")
    ap.add_argument("--results", default=str(ROOT / "results"), help="directory of the saved JSON results (default results)")
    args = ap.parse_args(argv)
    out, R = Path(args.out), Results(args.results)
    bad = 0
    for name, build in FRAGMENTS:
        try:
            text, warns = build(R)
        except NoSource as e:
            print(f"NO SOURCE  {name}: {e}")
            continue
        except Exception as e:  # a malformed JSON entry: report it and go on with the other fragments
            print(f"ERROR      {name}: {type(e).__name__}: {e}")
            bad += 1
            continue
        for w in warns:
            print(f"warning: {name}: {w}", file=sys.stderr)
        target = out / name
        old = target.read_bytes().decode("utf-8") if target.exists() else None
        diff = None if old == text else "no existing file" if old is None else first_diff(old, text)
        if args.check:
            if diff is None:
                print(f"OK         {name}")
            else:
                print(f"DIFFERS    {name}: {diff}")
                bad += 1
        elif diff is None:
            print(f"UNCHANGED  {name}")
        else:
            out.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))
            print(f"WROTE      {name} ({'new file' if old is None else 'changed: ' + diff})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
