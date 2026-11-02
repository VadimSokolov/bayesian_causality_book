## dpost dense (1000 replications)
| arm | bias (MCSE) | coverage (SE) | s.e./s.d. | s.e./RMSE | width | converged | median ESS |
|---|---|---|---|---|---|---|---|
| nb | 0.935 (0.001) | 0.000 (0.000) | 0.99 | 0.02 | 0.083 | 1.000 (1000 of 1000) | 50637 |
| rf | 3075.718 (5046.468) | 0.071 (0.008) | 0.42 | 0.42 | 260263.560 | 0.897 (897 of 1000) | 13393 |
McNemar nb vs rf: 0 to 71, exact p 8.47e-22, paired coverage difference -0.071 (SE 0.008)
McNemar rf vs dml_K5: 71 to 0, exact p 8.47e-22, paired coverage difference 0.071 (SE 0.008)

