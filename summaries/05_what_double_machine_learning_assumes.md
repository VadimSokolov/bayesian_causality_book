# Chapter 5. What Double Machine Learning Assumes

<!-- cite:begin -->
From *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning* by Nick Polson and Vadim Sokolov (2026), chapter 5. Cite as Polson and Sokolov (2026), *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning*, edition 1, https://doi.org/10.5281/zenodo.23902093. Book page: https://vsokolov.org/books/bayesian_causality_book/. Source: https://github.com/VadimSokolov/bayesian_causality_book.
<!-- cite:end -->

Double machine learning estimates one effect while flexible learners fit the nuisance functions around it. This chapter states the method and what it assumes. In the partially linear model the estimator is a regression of an outcome residual on a treatment residual, a Neyman orthogonal score makes nuisance errors enter as products, and cross fitting removes overfitting bias. The error of the estimator decomposes exactly into a noise term, a cross term and a bias term built from the two fitting errors, and the main theorem gives root n normal inference when the treatment keeps some variation after adjustment and the product rate condition holds. The partially linear estimand is a variance weighted average of effects, and the denominator of the estimator is not an overlap diagnostic. The 401(k) data serve as the running example, and a list says what DML assumes.

## From the literature

The chapter presents or builds on the partially linear model and partialling out (Robinson 1988), Neyman orthogonality (Neyman 1959), double machine learning with cross fitting and its main theorem (Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey and Robins 2018), the interactive model and augmented inverse probability weighting (Robins, Rotnitzky and Zhao 1994), the variance weighting of regression adjustment (Angrist 1998; Aronow and Samii 2016), the 401(k) application (Poterba, Venti and Wise 1995).

## New results

<!-- results:begin -->
- [Exact error decomposition of the partialling out estimator](../RESULTS.md#exact_error_decomposition_of_the_robinson_estimator) (Lemma 5.3, page 96; proved): The error of the partialling out estimator splits exactly, with no approximation, into a noise term, a cross term with mean zero under cross fitting and a bias term built from the two fitting errors, namely their product minus the effect times the squared treatment error. First stated in Polson and Sokolov (2026b), Orthogonality is not enough: double machine learning and the Bayesian alternative, working paper.
- [Two channels](../RESULTS.md#two_channels_of_the_dml_bias) (Corollary 5.4, page 98; proved): With the fits held fixed the estimator converges to the truth plus a bias fixed by the inner product of the two fitting errors and the squared treatment error. First stated in Polson and Sokolov (2026b), Orthogonality is not enough: double machine learning and the Bayesian alternative, working paper.
<!-- results:end -->

## Key terms

double machine learning, DML, partially linear model, Neyman orthogonality, cross fitting, nuisance functions, exact error decomposition, product rate condition, root n inference, variance weighted average, augmented IPW, 401(k) eligibility
