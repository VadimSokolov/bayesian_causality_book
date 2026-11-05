# Chapter 4. Groups, Urns and Information

<!-- cite:begin -->
From *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning* by Nick Polson and Vadim Sokolov (2026), chapter 4. Cite as Polson and Sokolov (2026), *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning*, edition 1, https://doi.org/10.5281/zenodo.23985916. Book page: https://vsokolov.org/books/bayesian_causality_book/. Source: https://github.com/VadimSokolov/bayesian_causality_book.
<!-- cite:end -->

A treatment effect that is stable across populations is an element of a group acting on the baseline, and an average of it only describes the case mix. This chapter works that out with groups, urns and information. One odds ratio supports a whole range of average effects as the case mix changes. A randomized experiment is an urn from which about half the units are drawn, so the finite and infinite forms of exchangeability disagree, and when units are sampled from a population the experiment identifies only the two arm laws. A Haar prior gives exact coverage in a transformation model with known baselines. The information gap, in nats, measures how far the data are from stability on a scale, and the posterior odds of one scale against another move at an exponential rate set by the gaps. Kidney stone data illustrate the choice of scale.

## From the literature

The chapter presents or builds on exchangeability and urns (de Finetti 1937; Heath and Sudderth 1976), Sanov's theorem (Sanov 1957) and its inverse for Dirichlet priors (Ganesh and O'Connell 1999), error exponents of tests (Hoeffding 1965; Bahadur 1967), the treatment effect as a group action (McCullagh 2002), bounds on the probability of necessity and sufficiency (Tian and Pearl 2000), stable effects on binary outcomes (Battey and Edgar 2026).

## New results

<!-- results:begin -->
- [Range of the average effect under a stable odds ratio](../RESULTS.md#average_effect_range_under_case_mix) (Proposition 4.3, page 55; proved): With one conditional odds ratio e^psi for every unit, the average risk difference over all possible case mixes fills the interval from 0 to tanh(psi/4), so the same stable effect supports very different averages. New in this book.
- [Separability](../RESULTS.md#separability_of_functionals_on_frechet_class) (Theorem 4.8, page 59; proved): The expectation of a function of two potential outcomes is determined by the two arm laws alone exactly when that function is a sum of a function of each outcome. New in this book.
- [Identified estimands are functions of the margins](../RESULTS.md#identified_estimands_are_functions_of_the_margins) (Corollary 4.9, page 59; proved): Any functional of the potential outcome distribution that an experiment identifies depends on the data only through the two arm laws. New in this book.
- [Stratified urns and the reversal](../RESULTS.md#stratified_urns_and_the_reversal) (Proposition 4.11, page 60; proved): In a stratified randomized experiment the collapsed contrast of arm means is unbiased for the finite population effect for every urn if and only if both arms are allocated in proportion to the stratum sizes. New in this book.
- [The posterior selects the group](../RESULTS.md#posterior_selects_the_group) (Theorem 4.19, page 65; proved): The log posterior odds of one effect scale against another, or against the saturated model, move linearly in n at a rate equal to the difference of their information gaps, with a log n term for the difference in the number of parameters. New in this book.
- [Transport of a stable effect across case mixes](../RESULTS.md#effect_transport_bound_by_total_variation) (Proposition 4.27, page 72; proved): The average effect on the scale of a group moves between two case mixes by at most the range of the stratum effects times the total variation distance between the mixes. New in this book.
<!-- results:end -->

## Key terms

treatment effect as a group action, McCullagh stability, case mix, average effect, odds ratio, urn, finite population correction, Neyman variance, Frechet bounds, Haar prior, transformation model, information gap, Pythagorean identity, large deviations, kidney stones
