## s5 (5000 paired replications)
| arm | bias (MCSE) | coverage (SE) | s.e./s.d. | s.e./RMSE | width | s.d. | mean s.e. |
|---|---|---|---|---|---|---|---|
| DML lasso K=5 | 0.017 (0.001) | 0.917 (0.004) | 0.91 | 0.88 | 0.209 | 0.0586 | 0.0533 |
| DML horseshoe K=5 | -0.011 (0.001) | 0.928 (0.004) | 0.95 | 0.93 | 0.216 | 0.0582 | 0.0550 |
| DML horseshoe K=2 | -0.085 (0.002) | 0.577 (0.007) | 0.45 | 0.38 | 0.247 | 0.1409 | 0.0631 |
| naive Bayes | 0.032 (0.001) | 0.921 (0.004) | 1.00 | 0.87 | 0.224 | 0.0575 | 0.0572 |
| reduced form posterior | 0.041 (0.001) | 0.913 (0.004) | 1.05 | 0.85 | 0.238 | 0.0580 | 0.0608 |
| DML lasso K=2 | 0.023 (0.001) | 0.865 (0.005) | 0.81 | 0.76 | 0.207 | 0.0654 | 0.0529 |
| split median K=5 | 0.017 (0.001) | 0.936 (0.003) | 0.99 | 0.94 | 0.215 | 0.0557 | 0.0548 |
| PDS theory | -0.002 (0.001) | 0.950 (0.003) | 1.01 | 1.01 | 0.216 | 0.0546 | 0.0550 |
| PDS CV | -0.086 (0.001) | 0.669 (0.007) | 0.71 | 0.51 | 0.249 | 0.0891 | 0.0636 |
| multiplier bootstrap | 0.017 (0.001) | 0.911 (0.004) | 0.91 | 0.88 | 0.208 | 0.0586 | 0.0533 |
convergence rf: {"share_converged": 1.0, "n_converged": 5000, "share_L_gt_10000": 0.0224, "max_L": 40000, "median_ess": 1463.6135416541629, "min_ess": 400.02407005060303, "max_rhat": 1.0099921539542607, "median_ess_omega": 72.8147307509223, "se_over_sd_converged_only": 1.0478789158426878, "coverage_converged_only": 0.9128}
convergence nb: {"share_converged": 1.0, "n_converged": 5000, "share_L_gt_10000": 0.0, "max_L": 10000, "median_ess": 11295.99698064716, "min_ess": 1203.0901285426348, "max_rhat": 1.0030961151310294, "median_ess_omega": 70.961926077476, "se_over_sd_converged_only": 0.9958027406339511, "coverage_converged_only": 0.9208}
McNemar reduced form posterior vs DML lasso K=5: discordant 155 (first only) to 177 (second only), exact p 0.249, paired coverage difference -0.004 (SE 0.004)
McNemar reduced form posterior vs DML horseshoe K=5: discordant 238 (first only) to 315 (second only), exact p 0.00121, paired coverage difference -0.015 (SE 0.005)
McNemar reduced form posterior vs PDS CV: discordant 1602 (first only) to 382 (second only), exact p 4.33e-177, paired coverage difference 0.244 (SE 0.008)
McNemar reduced form posterior vs PDS theory: discordant 121 (first only) to 305 (second only), exact p 2.04e-19, paired coverage difference -0.037 (SE 0.004)
McNemar reduced form posterior vs split median K=5: discordant 73 (first only) to 191 (second only), exact p 2.38e-13, paired coverage difference -0.024 (SE 0.003)
McNemar reduced form posterior vs multiplier bootstrap: discordant 176 (first only) to 169 (second only), exact p 0.747, paired coverage difference 0.001 (SE 0.004)
McNemar DML horseshoe K=5 vs DML lasso K=5: discordant 235 (first only) to 180 (second only), exact p 0.00795, paired coverage difference 0.011 (SE 0.004)
McNemar DML horseshoe K=5 vs DML lasso K=2: discordant 494 (first only) to 179 (second only), exact p 5.44e-35, paired coverage difference 0.063 (SE 0.005)
McNemar DML horseshoe K=2 vs DML horseshoe K=5: discordant 136 (first only) to 1892 (second only), exact p 0, paired coverage difference -0.351 (SE 0.008)
McNemar PDS theory vs DML lasso K=5: discordant 239 (first only) to 77 (second only), exact p 1.84e-20, paired coverage difference 0.032 (SE 0.004)
McNemar split median K=5 vs DML lasso K=5: discordant 142 (first only) to 46 (second only), exact p 1.36e-12, paired coverage difference 0.019 (SE 0.003)
McNemar multiplier bootstrap vs DML lasso K=5: discordant 15 (first only) to 44 (second only), exact p 0.000204, paired coverage difference -0.006 (SE 0.002)
McNemar DML lasso K=5 vs DML lasso K=2: discordant 398 (first only) to 138 (second only), exact p 3.18e-30, paired coverage difference 0.052 (SE 0.005)
McNemar PDS CV vs DML lasso K=5: discordant 303 (first only) to 1545 (second only), exact p 3.84e-200, paired coverage difference -0.248 (SE 0.008)
McNemar naive Bayes vs reduced form posterior: discordant 93 (first only) to 53 (second only), exact p 0.00117, paired coverage difference 0.008 (SE 0.002)
McNemar DML horseshoe K=5 vs PDS theory: discordant 61 (first only) to 168 (second only), exact p 9.48e-13, paired coverage difference -0.021 (SE 0.003)

## s15 (5000 paired replications)
| arm | bias (MCSE) | coverage (SE) | s.e./s.d. | s.e./RMSE | width | s.d. | mean s.e. |
|---|---|---|---|---|---|---|---|
| DML lasso K=5 | 0.037 (0.001) | 0.826 (0.005) | 0.81 | 0.70 | 0.205 | 0.0644 | 0.0523 |
| DML horseshoe K=5 | -0.012 (0.001) | 0.895 (0.004) | 0.85 | 0.83 | 0.217 | 0.0653 | 0.0552 |
| DML horseshoe K=2 | -0.069 (0.002) | 0.587 (0.007) | 0.49 | 0.43 | 0.236 | 0.1224 | 0.0603 |
| naive Bayes | 0.362 (0.004) | 0.446 (0.007) | 0.56 | 0.36 | 0.490 | 0.3053 | 0.1705 |
| reduced form posterior | 0.076 (0.001) | 0.787 (0.006) | 1.02 | 0.66 | 0.258 | 0.0648 | 0.0658 |
| DML lasso K=2 | 0.053 (0.001) | 0.694 (0.007) | 0.65 | 0.54 | 0.201 | 0.0786 | 0.0513 |
| split median K=5 | 0.038 (0.001) | 0.870 (0.005) | 0.93 | 0.78 | 0.218 | 0.0599 | 0.0555 |
| PDS theory | 0.798 (0.002) | 0.010 (0.001) | 0.26 | 0.04 | 0.127 | 0.1242 | 0.0323 |
| PDS CV | -0.110 (0.001) | 0.629 (0.007) | 0.72 | 0.49 | 0.285 | 0.1005 | 0.0727 |
| multiplier bootstrap | 0.037 (0.001) | 0.823 (0.005) | 0.81 | 0.70 | 0.204 | 0.0644 | 0.0523 |
convergence rf: {"share_converged": 1.0, "n_converged": 5000, "share_L_gt_10000": 0.0006, "max_L": 20000, "median_ess": 2264.5660199085655, "min_ess": 513.2734877354045, "max_rhat": 1.0095845817508529, "median_ess_omega": 189.46370984204663, "se_over_sd_converged_only": 1.0161703262130517, "coverage_converged_only": 0.7868}
convergence nb: {"share_converged": 0.589, "n_converged": 2945, "share_L_gt_10000": 0.5082, "max_L": 80000, "median_ess": 1276.5702241181502, "min_ess": 4.028301938525034, "max_rhat": 11.99408530560981, "median_ess_omega": 77.72051185335378, "se_over_sd_converged_only": 0.1621395684382093, "coverage_converged_only": 0.5079796264855687}
McNemar reduced form posterior vs DML lasso K=5: discordant 272 (first only) to 469 (second only), exact p 4.41e-13, paired coverage difference -0.039 (SE 0.005)
McNemar reduced form posterior vs DML horseshoe K=5: discordant 366 (first only) to 907 (second only), exact p 2.34e-53, paired coverage difference -0.108 (SE 0.007)
McNemar reduced form posterior vs PDS CV: discordant 1774 (first only) to 987 (second only), exact p 2.87e-51, paired coverage difference 0.157 (SE 0.010)
McNemar reduced form posterior vs PDS theory: discordant 3894 (first only) to 12 (second only), exact p 0, paired coverage difference 0.776 (SE 0.006)
McNemar reduced form posterior vs split median K=5: discordant 112 (first only) to 530 (second only), exact p 6.59e-66, paired coverage difference -0.084 (SE 0.005)
McNemar reduced form posterior vs multiplier bootstrap: discordant 279 (first only) to 461 (second only), exact p 2.28e-11, paired coverage difference -0.036 (SE 0.005)
McNemar DML horseshoe K=5 vs DML lasso K=5: discordant 659 (first only) to 315 (second only), exact p 1.14e-28, paired coverage difference 0.069 (SE 0.006)
McNemar DML horseshoe K=5 vs DML lasso K=2: discordant 1325 (first only) to 318 (second only), exact p 6.44e-146, paired coverage difference 0.201 (SE 0.008)
McNemar DML horseshoe K=2 vs DML horseshoe K=5: discordant 214 (first only) to 1754 (second only), exact p 1.98e-300, paired coverage difference -0.308 (SE 0.008)
McNemar PDS theory vs DML lasso K=5: discordant 6 (first only) to 4085 (second only), exact p 0, paired coverage difference -0.816 (SE 0.006)
McNemar split median K=5 vs DML lasso K=5: discordant 342 (first only) to 121 (second only), exact p 1.76e-25, paired coverage difference 0.044 (SE 0.004)
McNemar multiplier bootstrap vs DML lasso K=5: discordant 28 (first only) to 43 (second only), exact p 0.0959, paired coverage difference -0.003 (SE 0.002)
McNemar DML lasso K=5 vs DML lasso K=2: discordant 988 (first only) to 325 (second only), exact p 5.34e-78, paired coverage difference 0.133 (SE 0.007)
McNemar PDS CV vs DML lasso K=5: discordant 723 (first only) to 1707 (second only), exact p 5.35e-91, paired coverage difference -0.197 (SE 0.009)
McNemar naive Bayes vs reduced form posterior: discordant 11 (first only) to 1717 (second only), exact p 0, paired coverage difference -0.341 (SE 0.007)
McNemar DML horseshoe K=5 vs PDS theory: discordant 4426 (first only) to 3 (second only), exact p 0, paired coverage difference 0.885 (SE 0.005)

## s30 (5000 paired replications)
| arm | bias (MCSE) | coverage (SE) | s.e./s.d. | s.e./RMSE | width | s.d. | mean s.e. |
|---|---|---|---|---|---|---|---|
| DML lasso K=5 | 0.050 (0.001) | 0.734 (0.006) | 0.70 | 0.58 | 0.201 | 0.0734 | 0.0513 |
| DML horseshoe K=5 | -0.021 (0.001) | 0.827 (0.005) | 0.73 | 0.70 | 0.217 | 0.0760 | 0.0555 |
| DML horseshoe K=2 | -0.058 (0.001) | 0.671 (0.007) | 0.60 | 0.51 | 0.223 | 0.0943 | 0.0569 |
| naive Bayes | 0.956 (0.000) | 0.000 (0.000) | 0.62 | 0.02 | 0.073 | 0.0300 | 0.0185 |
| reduced form posterior | 0.106 (0.001) | 0.704 (0.006) | 1.00 | 0.57 | 0.292 | 0.0748 | 0.0745 |
| DML lasso K=2 | 0.106 (0.002) | 0.450 (0.007) | 0.38 | 0.29 | 0.191 | 0.1265 | 0.0487 |
| split median K=5 | 0.051 (0.001) | 0.820 (0.005) | 0.88 | 0.69 | 0.224 | 0.0650 | 0.0571 |
| PDS theory | 0.960 (0.000) | 0.000 (0.000) | 0.63 | 0.02 | 0.066 | 0.0269 | 0.0168 |
| PDS CV | -0.120 (0.002) | 0.651 (0.007) | 0.73 | 0.51 | 0.337 | 0.1180 | 0.0861 |
| multiplier bootstrap | 0.050 (0.001) | 0.728 (0.006) | 0.70 | 0.58 | 0.200 | 0.0734 | 0.0513 |
convergence rf: {"share_converged": 1.0, "n_converged": 5000, "share_L_gt_10000": 0.0004, "max_L": 20000, "median_ess": 2230.8628607061455, "min_ess": 560.2408930639181, "max_rhat": 1.0090802074436946, "median_ess_omega": 313.7761638143576, "se_over_sd_converged_only": 0.996448410728502, "coverage_converged_only": 0.7036}
convergence nb: {"share_converged": 0.9822, "n_converged": 4911, "share_L_gt_10000": 0.3938, "max_L": 80000, "median_ess": 1062.2941468876797, "min_ess": 61.9204743355402, "max_rhat": 1.0339563557575524, "median_ess_omega": 20.080360228858275, "se_over_sd_converged_only": 0.6150294602495654, "coverage_converged_only": 0.0}
McNemar reduced form posterior vs DML lasso K=5: discordant 482 (first only) to 634 (second only), exact p 5.99e-06, paired coverage difference -0.030 (SE 0.007)
McNemar reduced form posterior vs DML horseshoe K=5: discordant 649 (first only) to 1268 (second only), exact p 5.08e-46, paired coverage difference -0.124 (SE 0.009)
McNemar reduced form posterior vs PDS CV: discordant 1607 (first only) to 1346 (second only), exact p 1.69e-06, paired coverage difference 0.052 (SE 0.011)
McNemar reduced form posterior vs PDS theory: discordant 3518 (first only) to 0 (second only), exact p 0, paired coverage difference 0.704 (SE 0.006)
McNemar reduced form posterior vs split median K=5: discordant 154 (first only) to 737 (second only), exact p 7.45e-92, paired coverage difference -0.117 (SE 0.006)
McNemar reduced form posterior vs multiplier bootstrap: discordant 497 (first only) to 619 (second only), exact p 0.000289, paired coverage difference -0.024 (SE 0.007)
McNemar DML horseshoe K=5 vs DML lasso K=5: discordant 1054 (first only) to 587 (second only), exact p 5.05e-31, paired coverage difference 0.093 (SE 0.008)
McNemar DML horseshoe K=5 vs DML lasso K=2: discordant 2344 (first only) to 456 (second only), exact p 1.57e-304, paired coverage difference 0.378 (SE 0.009)
McNemar DML horseshoe K=2 vs DML horseshoe K=5: discordant 320 (first only) to 1103 (second only), exact p 7.75e-101, paired coverage difference -0.157 (SE 0.007)
McNemar PDS theory vs DML lasso K=5: discordant 0 (first only) to 3670 (second only), exact p 0, paired coverage difference -0.734 (SE 0.006)
McNemar split median K=5 vs DML lasso K=5: discordant 602 (first only) to 171 (second only), exact p 4.91e-57, paired coverage difference 0.086 (SE 0.005)
McNemar multiplier bootstrap vs DML lasso K=5: discordant 24 (first only) to 54 (second only), exact p 0.000901, paired coverage difference -0.006 (SE 0.002)
McNemar DML lasso K=5 vs DML lasso K=2: discordant 1830 (first only) to 409 (second only), exact p 1.08e-213, paired coverage difference 0.284 (SE 0.009)
McNemar PDS CV vs DML lasso K=5: discordant 1093 (first only) to 1506 (second only), exact p 5.57e-16, paired coverage difference -0.083 (SE 0.010)
McNemar naive Bayes vs reduced form posterior: discordant 0 (first only) to 3518 (second only), exact p 0, paired coverage difference -0.704 (SE 0.006)
McNemar DML horseshoe K=5 vs PDS theory: discordant 4137 (first only) to 0 (second only), exact p 0, paired coverage difference 0.827 (SE 0.005)

## weak (5000 paired replications)
| arm | bias (MCSE) | coverage (SE) | s.e./s.d. | s.e./RMSE | width | s.d. | mean s.e. |
|---|---|---|---|---|---|---|---|
| DML lasso K=5 | 0.317 (0.003) | 0.557 (0.007) | 0.91 | 0.47 | 0.687 | 0.1931 | 0.1753 |
| DML horseshoe K=5 | -0.024 (0.003) | 0.908 (0.004) | 0.87 | 0.86 | 0.713 | 0.2089 | 0.1818 |
| DML horseshoe K=2 | -0.040 (0.004) | 0.841 (0.005) | 0.74 | 0.73 | 0.764 | 0.2651 | 0.1949 |
| naive Bayes | 0.987 (0.001) | 0.000 (0.000) | 0.48 | 0.02 | 0.085 | 0.0450 | 0.0217 |
| reduced form posterior | 0.148 (0.003) | 0.906 (0.004) | 1.01 | 0.85 | 0.918 | 0.2313 | 0.2339 |
| DML lasso K=2 | 0.505 (0.003) | 0.235 (0.006) | 0.72 | 0.30 | 0.667 | 0.2374 | 0.1702 |
| split median K=5 | 0.316 (0.003) | 0.603 (0.007) | 1.04 | 0.51 | 0.723 | 0.1770 | 0.1845 |
| PDS theory | 0.924 (0.003) | 0.034 (0.003) | 0.18 | 0.04 | 0.138 | 0.1954 | 0.0352 |
| PDS CV | -0.134 (0.004) | 0.867 (0.005) | 0.88 | 0.79 | 0.925 | 0.2673 | 0.2360 |
| multiplier bootstrap | 0.317 (0.003) | 0.550 (0.007) | 0.91 | 0.47 | 0.684 | 0.1931 | 0.1752 |
convergence rf: {"share_converged": 1.0, "n_converged": 5000, "share_L_gt_10000": 0.0, "max_L": 10000, "median_ess": 4020.9183887757254, "min_ess": 648.9812240179181, "max_rhat": 1.0073522543673614, "median_ess_omega": 189.6166169334091, "se_over_sd_converged_only": 1.0113167540190708, "coverage_converged_only": 0.9064}
convergence nb: {"share_converged": 0.9892, "n_converged": 4946, "share_L_gt_10000": 0.3482, "max_L": 80000, "median_ess": 979.7051883540003, "min_ess": 65.7522758330498, "max_rhat": 1.0318818317124605, "median_ess_omega": 18.79817317472003, "se_over_sd_converged_only": 0.47959833778507727, "coverage_converged_only": 0.0}
McNemar reduced form posterior vs DML lasso K=5: discordant 1776 (first only) to 30 (second only), exact p 0, paired coverage difference 0.349 (SE 0.007)
McNemar reduced form posterior vs DML horseshoe K=5: discordant 276 (first only) to 285 (second only), exact p 0.736, paired coverage difference -0.002 (SE 0.005)
McNemar reduced form posterior vs PDS CV: discordant 593 (first only) to 397 (second only), exact p 5.06e-10, paired coverage difference 0.039 (SE 0.006)
McNemar reduced form posterior vs PDS theory: discordant 4368 (first only) to 8 (second only), exact p 0, paired coverage difference 0.872 (SE 0.005)
McNemar reduced form posterior vs split median K=5: discordant 1545 (first only) to 28 (second only), exact p 0, paired coverage difference 0.303 (SE 0.007)
McNemar reduced form posterior vs multiplier bootstrap: discordant 1811 (first only) to 31 (second only), exact p 0, paired coverage difference 0.356 (SE 0.007)
McNemar DML horseshoe K=5 vs DML lasso K=5: discordant 2045 (first only) to 290 (second only), exact p 0, paired coverage difference 0.351 (SE 0.008)
McNemar DML horseshoe K=5 vs DML lasso K=2: discordant 3588 (first only) to 220 (second only), exact p 0, paired coverage difference 0.674 (SE 0.008)
McNemar DML horseshoe K=2 vs DML horseshoe K=5: discordant 200 (first only) to 538 (second only), exact p 1.31e-36, paired coverage difference -0.068 (SE 0.005)
McNemar PDS theory vs DML lasso K=5: discordant 63 (first only) to 2677 (second only), exact p 0, paired coverage difference -0.523 (SE 0.007)
McNemar split median K=5 vs DML lasso K=5: discordant 477 (first only) to 248 (second only), exact p 1.35e-17, paired coverage difference 0.046 (SE 0.005)
McNemar multiplier bootstrap vs DML lasso K=5: discordant 51 (first only) to 85 (second only), exact p 0.00448, paired coverage difference -0.007 (SE 0.002)
McNemar DML lasso K=5 vs DML lasso K=2: discordant 1771 (first only) to 158 (second only), exact p 0, paired coverage difference 0.323 (SE 0.008)
McNemar PDS CV vs DML lasso K=5: discordant 2126 (first only) to 576 (second only), exact p 2.18e-207, paired coverage difference 0.310 (SE 0.009)
McNemar naive Bayes vs reduced form posterior: discordant 0 (first only) to 4532 (second only), exact p 0, paired coverage difference -0.906 (SE 0.004)
McNemar DML horseshoe K=5 vs PDS theory: discordant 4377 (first only) to 8 (second only), exact p 0, paired coverage difference 0.874 (SE 0.005)

