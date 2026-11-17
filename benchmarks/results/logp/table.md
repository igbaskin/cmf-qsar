Measured logP: Huuskonen set (1847 compounds, spread 1.49) and a class P sample of SangsterLogP (2999 compounds, spread 1.78). Five folds repeated three times on each set; the last two columns fit on the class P sample and predict the Huuskonen set. The setting and the ridge are chosen by the evidence on the training block.

| Model | q², Huuskonen | RMSE, Huuskonen | q², class P | RMSE, class P | R², class P to Huuskonen | RMSE, class P to Huuskonen |
|---|---:|---:|---:|---:|---:|---:|
| invariant, cosine | +0.884 | 0.510 | +0.778 | 0.838 | +0.803 | 0.664 |
| invariant, not normalised | +0.868 | 0.542 | +0.764 | 0.864 | +0.832 | 0.613 |
| invariant + clogP | +0.899 | 0.474 | +0.797 | 0.801 | +0.830 | 0.616 |
| invariant + five descriptors | +0.896 | 0.482 | +0.788 | 0.819 | +0.827 | 0.622 |
| linear regression on clogP | +0.820 | 0.634 | +0.710 | 0.959 | +0.780 | 0.701 |
| field totals | +0.847 | 0.585 | +0.752 | 0.886 | +0.820 | 0.633 |
| eight counts | +0.806 | 0.658 | +0.672 | 1.019 | +0.685 | 0.839 |
| ECFP4 | +0.823 | 0.629 | +0.764 | 0.865 | +0.736 | 0.768 |
| refitted Crippen increments | +0.848 | 0.582 | +0.745 | 0.898 | +0.743 | 0.758 |
| refitted increments + clogP | +0.845 | 0.587 | +0.745 | 0.898 | +0.755 | 0.740 |
| topological autocorrelation | +0.871 | 0.537 | +0.761 | 0.870 | +0.782 | 0.697 |
| topological + clogP | +0.884 | 0.509 | +0.774 | 0.846 | +0.820 | 0.634 |
| invariant + topological | +0.888 | 0.500 | +0.795 | 0.805 | +0.809 | 0.652 |
| invariant + eight counts | +0.906 | 0.458 | +0.810 | 0.775 | +0.841 | 0.595 |

| Difference of q² | Test | Mean [95 % interval] |
|---|---|---|
| invariant, cosine minus linear regression on clogP | Huuskonen, CV | +0.071 [+0.059, +0.084] |
| invariant, cosine minus linear regression on clogP | class P, CV | +0.076 [+0.061, +0.090] |
| invariant, cosine minus linear regression on clogP | class P to Huuskonen | +0.023 [+0.010, +0.036] |
| invariant, not normalised minus invariant, cosine | Huuskonen, CV | -0.010 [-0.021, +0.001] |
| invariant, not normalised minus invariant, cosine | class P, CV | -0.008 [-0.023, +0.006] |
| invariant, not normalised minus invariant, cosine | class P to Huuskonen | +0.029 [+0.016, +0.043] |
| invariant + clogP minus linear regression on clogP | Huuskonen, CV | +0.086 [+0.075, +0.097] |
| invariant + clogP minus linear regression on clogP | class P, CV | +0.091 [+0.080, +0.102] |
| invariant + clogP minus linear regression on clogP | class P to Huuskonen | +0.050 [+0.040, +0.061] |
| invariant + five descriptors minus invariant, cosine | Huuskonen, CV | +0.011 [+0.007, +0.016] |
| invariant + five descriptors minus invariant, cosine | class P, CV | +0.010 [+0.004, +0.015] |
| invariant + five descriptors minus invariant, cosine | class P to Huuskonen | +0.024 [+0.018, +0.030] |
| invariant, cosine minus field totals | Huuskonen, CV | +0.036 [+0.021, +0.050] |
| invariant, cosine minus field totals | class P, CV | +0.030 [+0.017, +0.043] |
| invariant, cosine minus field totals | class P to Huuskonen | -0.018 [-0.030, -0.006] |
| invariant, cosine minus eight counts | Huuskonen, CV | +0.081 [+0.065, +0.099] |
| invariant, cosine minus eight counts | class P, CV | +0.098 [+0.076, +0.119] |
| invariant, cosine minus eight counts | class P to Huuskonen | +0.118 [+0.097, +0.140] |
| invariant, cosine minus ECFP4 | Huuskonen, CV | +0.064 [+0.047, +0.081] |
| invariant, cosine minus ECFP4 | class P, CV | +0.014 [-0.003, +0.032] |
| invariant, cosine minus ECFP4 | class P to Huuskonen | +0.067 [+0.048, +0.088] |
| invariant + clogP minus ECFP4 | Huuskonen, CV | +0.078 [+0.062, +0.096] |
| invariant + clogP minus ECFP4 | class P, CV | +0.029 [+0.012, +0.047] |
| invariant + clogP minus ECFP4 | class P to Huuskonen | +0.094 [+0.075, +0.115] |
| invariant, cosine minus refitted Crippen increments | Huuskonen, CV | +0.042 [+0.030, +0.054] |
| invariant, cosine minus refitted Crippen increments | class P, CV | +0.039 [+0.027, +0.052] |
| invariant, cosine minus refitted Crippen increments | class P to Huuskonen | +0.060 [+0.043, +0.078] |
| invariant + clogP minus refitted increments + clogP | Huuskonen, CV | +0.059 [+0.049, +0.069] |
| invariant + clogP minus refitted increments + clogP | class P, CV | +0.054 [+0.046, +0.063] |
| invariant + clogP minus refitted increments + clogP | class P to Huuskonen | +0.075 [+0.061, +0.090] |
| invariant, cosine minus topological autocorrelation | Huuskonen, CV | +0.019 [+0.010, +0.027] |
| invariant, cosine minus topological autocorrelation | class P, CV | +0.022 [+0.013, +0.033] |
| invariant, cosine minus topological autocorrelation | class P to Huuskonen | +0.020 [+0.008, +0.034] |
| invariant + clogP minus topological + clogP | Huuskonen, CV | +0.021 [+0.013, +0.028] |
| invariant + clogP minus topological + clogP | class P, CV | +0.025 [+0.018, +0.033] |
| invariant + clogP minus topological + clogP | class P to Huuskonen | +0.010 [+0.001, +0.019] |
| topological autocorrelation minus refitted Crippen increments | Huuskonen, CV | +0.023 [+0.011, +0.034] |
| topological autocorrelation minus refitted Crippen increments | class P, CV | +0.017 [+0.004, +0.029] |
| topological autocorrelation minus refitted Crippen increments | class P to Huuskonen | +0.040 [+0.022, +0.057] |
| invariant + topological minus invariant, cosine | Huuskonen, CV | +0.004 [-0.000, +0.008] |
| invariant + topological minus invariant, cosine | class P, CV | +0.014 [+0.007, +0.019] |
| invariant + topological minus invariant, cosine | class P to Huuskonen | +0.007 [-0.001, +0.014] |
| invariant + eight counts minus invariant, cosine | Huuskonen, CV | +0.021 [+0.014, +0.029] |
| invariant + eight counts minus invariant, cosine | class P, CV | +0.035 [+0.026, +0.044] |
| invariant + eight counts minus invariant, cosine | class P to Huuskonen | +0.039 [+0.029, +0.048] |
| invariant + eight counts minus eight counts | Huuskonen, CV | +0.102 [+0.089, +0.117] |
| invariant + eight counts minus eight counts | class P, CV | +0.133 [+0.117, +0.149] |
| invariant + eight counts minus eight counts | class P to Huuskonen | +0.157 [+0.138, +0.176] |
| invariant + clogP minus invariant + eight counts | Huuskonen, CV | -0.007 [-0.015, +0.001] |
| invariant + clogP minus invariant + eight counts | class P, CV | -0.020 [-0.031, -0.009] |
| invariant + clogP minus invariant + eight counts | class P to Huuskonen | -0.012 [-0.022, -0.002] |
