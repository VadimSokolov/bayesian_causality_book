# Chapter 20. The Causal Effect of a Merger

<!-- cite:begin -->
From *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning* by Nick Polson and Vadim Sokolov (2026), chapter 20. Cite as Polson and Sokolov (2026), *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning*, edition 1, https://doi.org/10.5281/zenodo.23905002. Book page: https://vsokolov.org/books/bayesian_causality_book/. Source: https://github.com/VadimSokolov/bayesian_causality_book.
<!-- cite:end -->

A merger is a treatment applied once, to one industry, and never at random, so its price effect can only come from a model. This chapter defines the merger as the intervention do(H := H post) on the form of competition and shows that, under modularity and given the premerger data, the price effect depends on three nonlinear demand parameters and the form of competition alone. The agency's answer is then a pushforward posterior. Markups are the hinge: the Lerner index, upward pricing pressure and the merger price effect follow from the markup map. The chapter explains why orthogonal moments fall short for a nonlinear fixed point, then learns the posterior of the estimands and of conduct by simulating industries and training a network on summary statistics. Calibration on fresh industries compares the posterior with the usual BLP moment estimator, and a section covers wrong conduct.

## From the literature

The chapter presents or builds on the random coefficients logit demand model (Berry, Levinsohn and Pakes 1995), merger simulation and upward pricing pressure (Nevo 2000; Farrell and Shapiro 2010; Miller and Weinberg 2017), Bayesian estimation of BLP models (Viard, Gron and Polson 2015), approximate Bayesian computation with summary statistics (Frazier, Martin, Robert and Rousseau 2018).

## New results

<!-- results:begin -->
- [Identification of the merger effect](../RESULTS.md#merger_effect_depends_only_on_demand_and_conduct) (Proposition 20.5, page 343; proved): Given the premerger data, the price effect of a merger is a function of the nonlinear demand parameters and the form of competition alone, so the agency question is a question about their posterior. New in this book.
- [Falsifying a conduct model](../RESULTS.md#falsifying_a_conduct_model_under_logit) (Proposition 20.7, page 343; proved): Under logit demand a candidate form of competition is rejected by the instruments unless its markup function differs from the truth only by a combination of cost shifters. New in this book.
- [First order flatness at zero heterogeneity](../RESULTS.md#first_order_flatness_at_zero_heterogeneity) (Proposition 20.11, page 344; proved): At zero taste heterogeneity the market shares of the random coefficients logit are flat to first order in the heterogeneity scale, so moments built on shares carry no first order information about it. New in this book.
- [Markup pressure versus GUPPI under logit](../RESULTS.md#markup_pressure_exceeds_guppi_under_logit) (Lemma 20.13, page 345; proved): Under logit demand and Bertrand Nash pricing the exact markup increase from a merger exceeds the gross upward pricing pressure index whenever the merger partner has a positive share. New in this book.
- [A Bayes rule for merger review](../RESULTS.md#bayes_rule_for_merger_review) (Proposition 20.15, page 346; proved): The agency should challenge a merger when the posterior probability of harm exceeds a ratio of the two error costs, or, for graded costs, when expected excess harm outweighs the cost of blocking. New in this book.
- [Model averaging over conduct](../RESULTS.md#model_averaging_over_conduct_variance) (Proposition 20.16, page 347; proved): The posterior variance of a merger estimand splits into a within conduct term and a between conduct term, and a procedure that fixes one conduct omits the second. New in this book.
<!-- results:end -->

## Key terms

merger simulation, BLP demand, random coefficients logit, upward pricing pressure, GUPPI, markups, conduct, Bertrand Nash, antitrust, merger retrospective, Bayesian decision rule, generative Bayesian computation, model averaging
