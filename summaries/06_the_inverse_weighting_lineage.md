# Chapter 6. The Inverse Weighting Lineage

<!-- cite:begin -->
From *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning* by Nick Polson and Vadim Sokolov (2026), chapter 6. Cite as Polson and Sokolov (2026), *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning*, edition 1, https://doi.org/10.5281/zenodo.23961396. Book page: https://vsokolov.org/books/bayesian_causality_book/. Source: https://github.com/VadimSokolov/bayesian_causality_book.
<!-- cite:end -->

Inverse probability weighting has a long lineage, and this chapter follows it from Horvitz and Thompson to the augmented score. Every improvement has moved weight off the weights and onto an outcome model. Weights explode under weak overlap, trimming changes the estimand, and Basu's elephants show that unbiasedness over the design says little about the sample in hand. Double robustness and balancing weights are presented next to Bayesian views of the propensity score and a matching example from care management. The weighting estimator is shown to be a posterior mean under a suitable prior. In the Robins, Ritov and Wasserman example, a prior that ignores the known propensity is badly off, a prior coupled to it matches Horvitz and Thompson, and a hierarchical estimator removes the factor of one over the propensity from the Horvitz Thompson and Hájek variances.

## From the literature

The chapter presents or builds on the Horvitz Thompson estimator (Horvitz and Thompson 1952), Basu's elephants (Basu 1971), the Hájek estimator (Hájek 1971), doubly robust estimation (Robins, Rotnitzky and Zhao 1994), the Kang and Schafer example (Kang and Schafer 2007), the Robins and Ritov challenge (Robins and Ritov 1997; Wasserman 2004), balancing weights (Hainmueller 2012).

## New results

<!-- results:begin -->
No new results of this chapter are catalogued.
<!-- results:end -->

## Key terms

inverse probability weighting, Horvitz Thompson, Hajek estimator, stabilized weights, propensity score, overlap, trimming, double robustness, Basu elephants, Robins Ritov Wasserman, balancing weights, matching, Bayesian propensity score
