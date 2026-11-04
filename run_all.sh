#!/bin/sh
# Rerun every analysis in the book. Run from this folder. Total time is several hours on two cores;
# the slow steps are marked. Each script can also be run on its own in the same way.
set -e
export PYTHONPATH=.
mkdir -p figs results
for s in make_figs make_figs2 figs3 ipw_sim gbc_distort hs_theory hs_theory_plot qnn_book comp_figs exact_localsbc hs_scale_check benchmark_401k frontdoor_dgp iv_card_ar iv_fig_check thornton_laplace milk_unrecorded groups_urn_tau groups_kidney moscow_observers; do
  echo "== $s"; python book_scripts/$s.py
done
for s in app401k_book dml_partitions dml_irm_401k bart_apps k401_checks k401_student_t k401_quantiles sim_bart lalonde_book workflow_figs apps2 did_hs did_checks baipw_rep open_sims iv_card iv_weak2 iv_horseshoe iv_chain_check smoking_bayes bounds_examples; do
  echo "== $s (slow)"; python book_scripts/$s.py
done
echo "== gbc_ate (about 3 minutes on two threads of an Intel node)"; python book_scripts/gbc_ate.py --stage all
echo "== figs4"; python book_scripts/figs4.py
echo "== ric_sim (500 replications, about 8 minutes on four cores)"; seq 1 500 | OMP_NUM_THREADS=1 xargs -P 4 -I{} python book_scripts/ric_sim.py --rep {} && python book_scripts/ric_sim.py --aggregate
echo "== ric_sim, high dimensional design n = 200, p = 150 (500 replications, about 10 minutes on four cores; the saved replications were run as a SLURM array job, one per task)"; seq 1 500 | OMP_NUM_THREADS=1 xargs -P 4 -I{} python book_scripts/ric_sim.py --rep {} --n 200 --p 150 --tag hd150 && python book_scripts/ric_sim.py --aggregate --tag hd150
echo "== ric_sim, high dimensional design n = 200, p = 300 (500 replications, about 17 minutes on four cores)"; seq 1 500 | OMP_NUM_THREADS=1 xargs -P 4 -I{} python book_scripts/ric_sim.py --rep {} --n 200 --p 300 --tag hd300 && python book_scripts/ric_sim.py --aggregate --tag hd300
echo "== time_fits (the timings of Chapter 9, about 2 minutes on one thread; wall clock, so run it on an otherwise idle machine)"; python book_scripts/time_fits.py
echo "== ric_chain_check (split R-hat and effective sample size of the horseshoe samplers of Chapters 8, 9 and 16, about 25 minutes on one core)"; python book_scripts/ric_chain_check.py
echo "== hbdml tables (from the saved replications in book_scripts/hbdml/results; rerun them with the SLURM scripts in that folder)"; python book_scripts/hbdml_tables.py
echo "== deep_401k (about 4 minutes)"; python book_scripts/deep_401k.py && python book_scripts/intro_fig.py
echo "== deep_401k_coverage (200 simulated data sets of twenty refits, about five hours on two threads; the saved run was a SLURM array of 50 tasks of 4)"; python book_scripts/deep_401k_coverage.py truth && python book_scripts/deep_401k_coverage.py rep 0 200 && python book_scripts/deep_401k_coverage.py agg
echo "== open problems, further work (about 10 minutes)"; python book_scripts/open_sims2.py && python book_scripts/open_localsbc.py && python book_scripts/open_sims3.py
cd blp_merger && mkdir -p results figures
python simulate_reference_table.py 3000 && python run_experiments.py && python gbc_refine.py && python make_figures.py
python mc.py 60 && python mc_nomicro.py && python make_figures_mc.py
python cereal_example.py && python make_figures_cereal.py
