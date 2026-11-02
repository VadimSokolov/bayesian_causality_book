# Bayesian BLP Demand Estimation: code

Polson and Sokolov, "Bayesian BLP Demand Estimation: Markups, Conduct and Merger
Simulation" (October 2026).

Requires Python 3 with numpy, scipy, scikit learn and matplotlib.

## Files

* `blp_core.py`  random coefficients logit demand with a lognormal, income driven price
  coefficient (Petrin style), BLP contraction, price derivatives, markup inversion
  `m = -(H * D')^{-1} s` for Bertrand Nash, joint monopoly and perfect competition, the
  Morrow Skerlos equilibrium solver, consumer surplus, and the data simulator with Petrin
  micro moments.
* `estimators.py`  logit IV, BLP GMM (two step, sandwich covariance), the Bayesian inversion
  likelihood MCMC (Jiang, Manchanda and Rossi), the markup centred summary statistics, and the
  GBC estimator (conduct classifier plus cross fitted noise outsourcing posterior).
* `simulate_reference_table.py`  amortised GBC reference table (3,000 industries per conduct).
* `run_experiments.py`  the minivan merger application: every estimator, posterior markups,
  diversion, GUPPI, merger simulation, and the calibration study.
* `gbc_refine.py`  local second GBC round with the antitrust estimands learned directly.
* `antitrust.py`  markups, costs, diversion, GUPPI, merger prices and consumer surplus as
  functions of (theta2, conduct) and the data.
* `mc.py`  Monte Carlo comparison of BLP GMM and GBC at T = 40 and T = 15 markets, and the
  conduct misspecification experiment; `mc_nomicro.py` adds GBC without micro moments.
* `cereal_example.py`, `make_figures_cereal.py`  the antitrust case: a cereal style merger of the
  second and third firms with a divestiture remedy, GBC against BLP GMM and logit.
* `make_figures.py`, `make_figures_mc.py`  all figures, tables and the numbers quoted in the paper.

## Reproduce

Run from this folder (`blp_merger`). Outputs go to `results/` and `figures/` here;
`reference_outputs/` holds the number files, tables and figures used in the book (Chapter 20).

```
python simulate_reference_table.py 3000   # about 2 minutes on 2 cores
python run_experiments.py                 # about 6 minutes
python gbc_refine.py                      # about 5 minutes
python make_figures.py
python mc.py 60                           # about 35 minutes on 2 cores
python mc_nomicro.py
python make_figures_mc.py
python cereal_example.py                  # about 6 minutes
python make_figures_cereal.py
```
