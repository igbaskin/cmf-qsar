Measured water solubility, logS in mol/L: the Delaney set (1114 compounds, spread 2.09) and the AqSolDB compounds of groups G3 and G5 (1002 compounds, spread 2.30). Five folds repeated three times on each set; the last two columns fit on the AqSolDB set and predict the Delaney set. The setting and the ridge are chosen by the evidence on the training block. The published ESOL equation, with no fitting, gives R² +0.810 and RMSE 0.913 on the Delaney set.

| Model | q², Delaney | RMSE, Delaney | q², AqSolDB | RMSE, AqSolDB | R², AqSolDB to Delaney | RMSE, AqSolDB to Delaney |
|---|---:|---:|---:|---:|---:|---:|
| invariant, cosine | +0.875 | 0.739 | +0.831 | 0.946 | +0.827 | 0.871 |
| invariant, not normalised | +0.838 | 0.843 | +0.801 | 1.028 | +0.791 | 0.956 |
| invariant + ESOL descriptors | +0.896 | 0.676 | +0.859 | 0.865 | +0.856 | 0.793 |
| linear regression on clogP | +0.682 | 1.181 | +0.688 | 1.286 | +0.679 | 1.186 |
| ESOL refitted | +0.762 | 1.021 | +0.784 | 1.070 | +0.744 | 1.059 |
| field totals | +0.861 | 0.780 | +0.818 | 0.983 | +0.816 | 0.897 |
| eight counts | +0.853 | 0.804 | +0.832 | 0.945 | +0.778 | 0.987 |
| ECFP4 | +0.775 | 0.994 | +0.718 | 1.223 | +0.663 | 1.215 |
| refitted Crippen increments | +0.824 | 0.878 | +0.799 | 1.033 | +0.766 | 1.013 |
| refitted increments + ESOL descriptors | +0.818 | 0.892 | +0.805 | 1.018 | +0.765 | 1.014 |
| topological autocorrelation | +0.847 | 0.819 | +0.815 | 0.992 | +0.811 | 0.910 |
| topological + ESOL descriptors | +0.884 | 0.714 | +0.852 | 0.885 | +0.855 | 0.797 |

| Difference of q² | Test | Mean [95 % interval] |
|---|---|---|
| invariant, cosine minus ESOL refitted | Delaney, CV | +0.119 [+0.096, +0.144] |
| invariant, cosine minus ESOL refitted | AqSolDB, CV | +0.055 [+0.031, +0.081] |
| invariant, cosine minus ESOL refitted | AqSolDB to Delaney | +0.083 [+0.059, +0.108] |
| invariant + ESOL descriptors minus ESOL refitted | Delaney, CV | +0.138 [+0.118, +0.162] |
| invariant + ESOL descriptors minus ESOL refitted | AqSolDB, CV | +0.076 [+0.062, +0.093] |
| invariant + ESOL descriptors minus ESOL refitted | AqSolDB to Delaney | +0.112 [+0.095, +0.132] |
| invariant, cosine minus linear regression on clogP | Delaney, CV | +0.199 [+0.168, +0.238] |
| invariant, cosine minus linear regression on clogP | AqSolDB, CV | +0.151 [+0.119, +0.187] |
| invariant, cosine minus linear regression on clogP | AqSolDB to Delaney | +0.148 [+0.122, +0.179] |
| invariant, not normalised minus invariant, cosine | Delaney, CV | -0.036 [-0.059, -0.015] |
| invariant, not normalised minus invariant, cosine | AqSolDB, CV | -0.026 [-0.077, +0.012] |
| invariant, not normalised minus invariant, cosine | AqSolDB to Delaney | -0.036 [-0.078, -0.003] |
| invariant, cosine minus field totals | Delaney, CV | +0.017 [+0.001, +0.033] |
| invariant, cosine minus field totals | AqSolDB, CV | +0.015 [-0.005, +0.035] |
| invariant, cosine minus field totals | AqSolDB to Delaney | +0.010 [-0.011, +0.032] |
| invariant, cosine minus eight counts | Delaney, CV | +0.023 [+0.006, +0.042] |
| invariant, cosine minus eight counts | AqSolDB, CV | +0.003 [-0.017, +0.022] |
| invariant, cosine minus eight counts | AqSolDB to Delaney | +0.049 [+0.025, +0.075] |
| invariant, cosine minus ECFP4 | Delaney, CV | +0.101 [+0.082, +0.124] |
| invariant, cosine minus ECFP4 | AqSolDB, CV | +0.115 [+0.090, +0.142] |
| invariant, cosine minus ECFP4 | AqSolDB to Delaney | +0.164 [+0.135, +0.196] |
| invariant + ESOL descriptors minus ECFP4 | Delaney, CV | +0.121 [+0.099, +0.144] |
| invariant + ESOL descriptors minus ECFP4 | AqSolDB, CV | +0.137 [+0.110, +0.166] |
| invariant + ESOL descriptors minus ECFP4 | AqSolDB to Delaney | +0.194 [+0.168, +0.223] |
| invariant, cosine minus refitted Crippen increments | Delaney, CV | +0.056 [+0.036, +0.076] |
| invariant, cosine minus refitted Crippen increments | AqSolDB, CV | +0.037 [+0.009, +0.067] |
| invariant, cosine minus refitted Crippen increments | AqSolDB to Delaney | +0.061 [+0.035, +0.088] |
| invariant + ESOL descriptors minus refitted increments + ESOL descriptors | Delaney, CV | +0.081 [+0.066, +0.097] |
| invariant + ESOL descriptors minus refitted increments + ESOL descriptors | AqSolDB, CV | +0.054 [+0.035, +0.075] |
| invariant + ESOL descriptors minus refitted increments + ESOL descriptors | AqSolDB to Delaney | +0.091 [+0.072, +0.113] |
| invariant, cosine minus topological autocorrelation | Delaney, CV | +0.032 [+0.019, +0.045] |
| invariant, cosine minus topological autocorrelation | AqSolDB, CV | +0.021 [+0.005, +0.037] |
| invariant, cosine minus topological autocorrelation | AqSolDB to Delaney | +0.016 [+0.000, +0.031] |
| invariant + ESOL descriptors minus topological + ESOL descriptors | Delaney, CV | +0.015 [+0.005, +0.025] |
| invariant + ESOL descriptors minus topological + ESOL descriptors | AqSolDB, CV | +0.007 [-0.000, +0.015] |
| invariant + ESOL descriptors minus topological + ESOL descriptors | AqSolDB to Delaney | +0.001 [-0.006, +0.009] |
| topological autocorrelation minus refitted Crippen increments | Delaney, CV | +0.024 [+0.003, +0.046] |
| topological autocorrelation minus refitted Crippen increments | AqSolDB, CV | +0.016 [-0.012, +0.047] |
| topological autocorrelation minus refitted Crippen increments | AqSolDB to Delaney | +0.045 [+0.018, +0.073] |
