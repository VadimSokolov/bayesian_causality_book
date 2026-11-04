# Chapter 10. Generative Treatment Effects and Joint Potential Outcomes

<!-- cite:begin -->
From *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning* by Nick Polson and Vadim Sokolov (2026), chapter 10. Cite as Polson and Sokolov (2026), *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning*, edition 1, https://doi.org/10.5281/zenodo.23961396. Book page: https://vsokolov.org/books/bayesian_causality_book/. Source: https://github.com/VadimSokolov/bayesian_causality_book.
<!-- cite:end -->

Noise outsourcing represents any conditional law as a map from noise to outcomes. Learning that map for treated and untreated outcomes gives every conditional and quantile effect and every other functional of the two marginal laws, which is what decisions under concave utility need and the average effect does not supply. This chapter presents the generative treatment effect learner, benchmarks it, treats uncertainty for a learned map and deep learning in high dimensions, and covers individualized decisions, heterogeneity with the horseshoe, quantile regression as a likelihood and comparison with conformal prediction. The joint law of the two potential outcomes is a separate question: the marginals bound it without identifying it, optimal transport supplies a canonical coupling, and a copula prior covers the rest. A coverage study checks the bands on the 401(k) data.

## From the literature

The chapter presents or builds on generative causal inference and generative learners for distributional effects (Nareklishvili, Polson and Sokolov 2023a, 2025), the Makarov bounds (Makarov 1981), optimal transport (Villani 2009), conformal prediction (Lei and Candes 2021), deep partial least squares (Nareklishvili, Polson and Sokolov 2023b).

## New results

<!-- results:begin -->
No new results of this chapter are catalogued.
<!-- results:end -->

## Key terms

generative treatment effects, noise outsourcing, quantile treatment effect, distributional effects, joint potential outcomes, Makarov bounds, optimal transport coupling, copula prior, deep learning, individualized treatment rules, policy learning, conformal prediction
