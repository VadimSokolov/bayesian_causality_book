# Appendix C. Posterior Guarantees

<!-- cite:begin -->
From *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning* by Nick Polson and Vadim Sokolov (2026), appendix C. Cite as Polson and Sokolov (2026), *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning*, edition 1, https://doi.org/10.5281/zenodo.23985916. Book page: https://vsokolov.org/books/bayesian_causality_book/. Source: https://github.com/VadimSokolov/bayesian_causality_book.
<!-- cite:end -->

This appendix gives the posterior guarantees behind the horseshoe analyses. It states posterior concentration theorems, Bernstein von Mises theorems, the behaviour of the horseshoe in the sparse normal means model, predictive densities, and the consequences for treatment effects. It then applies them to difference in differences, where cohort deviations from an event path are sparse, and to groups, scales and stability. Several theorems are quoted from the literature with their authors and the exact result used, and the corollaries that follow from them, such as the coverage of credible intervals under misspecification and the predictive risk of the horseshoe, are derived here. The guarantees are asymptotic and sparse; the chapters say where simulation shows them failing in finite samples.

## From the literature

The chapter presents or builds on posterior concentration (Ghosal, Ghosh and van der Vaart 2000), Bernstein von Mises under misspecification (Kleijn and van der Vaart 2012), the horseshoe in the sparse normal means model (van der Pas, Kleijn and van der Vaart 2014; van der Pas, Szabo and van der Vaart 2017a, 2017b), predictive density estimation (Brown, George and Xu 2008).

## New results

<!-- results:begin -->
No new results of this appendix are catalogued.
<!-- results:end -->

## Key terms

posterior concentration, posterior contraction rate, Bernstein von Mises, misspecification, sandwich variance, horseshoe, sparse normal means, minimax rate, predictive risk, credible interval coverage, difference in differences guarantees
