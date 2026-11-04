# Chapter 16. Two Benchmarks

<!-- cite:begin -->
From *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning* by Nick Polson and Vadim Sokolov (2026), chapter 16. Cite as Polson and Sokolov (2026), *Bayesian Causality: Models, Accounting and the Bayesian Alternative to Double Machine Learning*, edition 1, https://doi.org/10.5281/zenodo.23961396. Book page: https://vsokolov.org/books/bayesian_causality_book/. Source: https://github.com/VadimSokolov/bayesian_causality_book.
<!-- cite:end -->

Two benchmark data sets test the methods of the book. On the 401(k) data, DML and the horseshoe report one effect for everyone and put it near the same range, well above least squares with nine covariates. The posterior adds what that single number hides: effects that rise with income and an effect on the untreated that differs from the effect on the treated. A Bayesian causal forest did not converge on these data, so the chapter reports no forest estimates and the income profile still lacks a check by a second model class. Overlap is adequate but thin at low incomes. On the LaLonde benchmark the randomized experiment gives a benchmark effect, the raw difference with survey controls is badly off, and flexible methods land much closer. The chapter reports the numbers with their posteriors, discusses overlap and the outcome distribution, and says where the methods still differ.

## From the literature

The chapter presents or builds on the 401(k) study (Poterba, Venti and Wise 1995; Chernozhukov and Hansen 2004), the LaLonde benchmark (LaLonde 1986; Dehejia and Wahba 1999), DML (Chernozhukov and coauthors 2018), Bayesian causal forests (Hahn, Murray and Carvalho 2020).

## New results

<!-- results:begin -->
No new results of this chapter are catalogued.
<!-- results:end -->

## Key terms

401(k) eligibility and savings, LaLonde benchmark, NSW experiment, CPS comparison group, heterogeneous effects by income, effect on the treated and untreated, Bayesian causal forest, overlap, DML comparison, observational benchmark
