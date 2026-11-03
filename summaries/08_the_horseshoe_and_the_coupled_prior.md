# Chapter 8. The Horseshoe and the Coupled Prior

<!-- cite:begin -->
From *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning* by Nick Polson and Vadim Sokolov (2026), chapter 8. Cite as Polson and Sokolov (2026), *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning*, edition 1, https://doi.org/10.5281/zenodo.23905002. Book page: https://vsokolov.org/books/bayesian_causality_book/. Source: https://github.com/VadimSokolov/bayesian_causality_book.
<!-- cite:end -->

The horseshoe shrinks noise hard and leaves large signals almost untouched, and this chapter uses it for causal regression. Independent shrinkage of confounder coefficients creates regularization induced confounding, because a shrunk coefficient lets the treatment absorb part of the confounding. Coupling the outcome prior to the treatment model through the estimated treatment regression reduces the bias. The chapter reports simulations at 200 observations: with 50 covariates the coupled posterior trails DML slightly on the average effect and also returns every other summary, with 150 and 300 covariates DML is clearly more accurate, and with thirty weak confounders among 150 covariates the order reverses. It also covers shared local scales and instruments, the global scale, comparison with spike and slab priors, nonlinear nuisances, and a horseshoe nuisance estimator inside DML.

## From the literature

The chapter presents or builds on the horseshoe prior (Carvalho, Polson and Scott 2010), shrink globally and act locally (Polson and Scott 2011), the horseshoe plus estimator (Bhadra, Datta, Polson and Willard 2017), posterior contraction of the horseshoe (van der Pas, Kleijn and van der Vaart 2014), regularization induced confounding (Hahn, Carvalho, Puelz and He 2018), the simulation comparison of DML and hierarchical Bayes (Nareklishvili, Polson and Sokolov 2026a).

## New results

<!-- results:begin -->
No new results of this chapter are catalogued.
<!-- results:end -->

## Key terms

horseshoe prior, global local shrinkage, regularization induced confounding, coupled horseshoe, partially exchangeable prior, hierarchical Bayes, DML versus Bayes, sparse regression, spike and slab, shared local scales, coverage, simulation
