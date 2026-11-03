# Chapter 9. Computing the Posterior

<!-- cite:begin -->
From *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning* by Nick Polson and Vadim Sokolov (2026), chapter 9. Cite as Polson and Sokolov (2026), *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning*, edition 1, https://doi.org/10.5281/zenodo.23905002. Book page: https://vsokolov.org/books/bayesian_causality_book/. Source: https://github.com/VadimSokolov/bayesian_causality_book.
<!-- cite:end -->

This chapter takes up the objection that Bayesian computation is too costly. Horseshoe Gibbs samplers were fast in the Gaussian designs of the book, so at that size computation is no reason to prefer DML, while at 300 or 400 covariates DML with a lasso nuisance is cheaper. When the likelihood cannot be evaluated cheaply, a posterior is a map from noise to parameters: a quantile network trained on simulated pairs of parameter and data with the check loss turns uniform noise and a summary of the data into posterior draws. The chapter gives the algorithm, what the network sees through summaries, how to get posteriors of effects from simulated pairs, and why one usually does not learn the horseshoe posterior this way. Diagnostics, including rank statistics, are part of the computation, and a closing table says which engine to use when.

## From the literature

The chapter presents or builds on noise outsourcing (Kallenberg 2002), quantile regression and the check loss (Koenker and Bassett 1978), simulation based calibration (Talts, Betancourt, Simpson, Vehtari and Gelman 2018), generative Bayesian computation (Polson and Sokolov 2023), Polya Gamma augmentation (Polson, Scott and Windle 2013).

## New results

<!-- results:begin -->
No new results of this chapter are catalogued.
<!-- results:end -->

## Key terms

Bayesian computation, Gibbs sampler, horseshoe sampler, generative Bayesian computation, quantile neural network, check loss, noise outsourcing, amortized inference, simulation based calibration, rank statistic, computational cost, DML timing
