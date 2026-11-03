# Chapter 15. Difference in Differences with the Horseshoe

<!-- cite:begin -->
From *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning* by Nick Polson and Vadim Sokolov (2026), chapter 15. Cite as Polson and Sokolov (2026), *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning*, edition 1, https://doi.org/10.5281/zenodo.23905002. Book page: https://vsokolov.org/books/bayesian_causality_book/. Source: https://github.com/VadimSokolov/bayesian_causality_book.
<!-- cite:end -->

Two way fixed effects estimates a weighted average of cohort by time effects with possibly negative weights, and this chapter replaces it with a model of the cells. The horseshoe difference in differences estimator keeps an unshrunk event time path, puts horseshoe priors on cohort deviations and on departures from parallel trends, and reports every summary, such as the effect on the treated or an event study, as an accounting of the posterior. A simulation studies its behaviour. The application is the castle doctrine laws and homicide: with equal noise in every state the posterior mean is 0.025 with probability 0.76 of a positive effect, and with population weights, which the residuals support, it is 0.079 with probability 0.998, close to the published eight percent. The chapter relates the estimator to synthetic control and states the assumptions it needs, among them parallel trends after the modelled departures and no anticipation.

## From the literature

The chapter presents or builds on two way fixed effects and its weights (Goodman Bacon 2021; de Chaisemartin and D'Haultfoeuille 2020), heterogeneity robust estimators (Callaway and Sant'Anna 2021; Sun and Abraham 2021), the castle doctrine study (Cheng and Hoekstra 2013), synthetic control (Abadie, Diamond and Hainmueller 2010).

## New results

<!-- results:begin -->
No new results of this chapter are catalogued.
<!-- results:end -->

## Key terms

difference in differences, two way fixed effects, staggered adoption, negative weights, horseshoe prior, event study, parallel trends, castle doctrine, homicide, synthetic control, cohort effects, heterogeneity robust estimators
