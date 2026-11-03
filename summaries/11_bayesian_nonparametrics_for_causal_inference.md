# Chapter 11. Bayesian Nonparametrics for Causal Inference

<!-- cite:begin -->
From *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning* by Nick Polson and Vadim Sokolov (2026), chapter 11. Cite as Polson and Sokolov (2026), *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning*, edition 1, https://doi.org/10.5281/zenodo.23905002. Book page: https://vsokolov.org/books/bayesian_causality_book/. Source: https://github.com/VadimSokolov/bayesian_causality_book.
<!-- cite:end -->

Under ignorability, causal inference is regression of the potential outcomes on the covariates, and Bayesian nonparametrics puts a prior on the whole response surface. This chapter covers Bayesian additive regression trees, Hill's causal BART, the overlap problem that plain BART meets, and Bayesian causal forests, which separate prognosis from effect and put the propensity score into the prior. It reviews posterior contraction for forests, runs a simulation in the spirit of Hill, and treats Gaussian process causal inference. Every effect, average, conditional or on the treated, is read off one fitted surface with its uncertainty attached. The chapter also looks at heterogeneous effects, instruments and sensitivity in these models, the prior as a selector of structure, the role of the propensity score in a Bayesian model, and trees compared with the horseshoe.

## From the literature

The chapter presents or builds on Bayesian additive regression trees (Chipman, George and McCulloch 2010), BART for causal inference (Hill 2011), Bayesian causal forests (Hahn, Murray and Carvalho 2020), posterior contraction of Bayesian forests (Rockova and van der Pas 2020).

## New results

<!-- results:begin -->
No new results of this chapter are catalogued.
<!-- results:end -->

## Key terms

Bayesian nonparametrics, BART, Bayesian additive regression trees, Bayesian causal forest, Gaussian process, response surface modelling, heterogeneous treatment effects, overlap, propensity score in the prior, posterior contraction, Hill simulation
