Measured vapour pressures, logVP in mmHg near 25 C: the PHYSPROP set curated for OPERA, its training part (2016 compounds, spread 3.55) and test part (677), and vapour pressures from eChemPortal (2884 compounds, spread 3.36). Five folds repeated three times on the training part and on the eChemPortal set; the external tests fit on the whole training part. The setting and the ridge are chosen by the evidence on the training block. Each cell gives q² (cross-validation) or R² (external test), with the RMSE in log units in brackets.

| Model | PHYSPROP training part, CV | eChemPortal, CV | PHYSPROP test part (the OPERA test) | PHYSPROP training part to eChemPortal |
|---|---:|---:|---:|---:|
| invariant, cosine | +0.874 (1.26) | +0.549 (2.26) | +0.873 (1.27) | +0.276 (2.86) |
| invariant, not normalised | +0.851 (1.37) | +0.339 (2.73) | +0.848 (1.38) | -0.699 (4.38) |
| invariant + Joback groups | +0.915 (1.03) | +0.559 (2.23) | +0.902 (1.11) | -0.162 (3.63) |
| linear regression on molecular weight | +0.585 (2.29) | +0.258 (2.90) | +0.553 (2.37) | -0.003 (3.37) |
| Joback refitted | +0.897 (1.14) | +0.459 (2.47) | +0.876 (1.25) | -0.367 (3.93) |
| field totals | +0.793 (1.62) | +0.428 (2.54) | +0.778 (1.67) | +0.261 (2.89) |
| eight counts | +0.893 (1.16) | +0.578 (2.19) | +0.907 (1.08) | +0.477 (2.43) |
| ECFP4 | +0.791 (1.62) | +0.518 (2.33) | +0.768 (1.71) | +0.394 (2.62) |
| refitted Crippen increments | +0.894 (1.16) | +0.469 (2.45) | +0.882 (1.22) | -0.387 (3.96) |
| topological autocorrelation | +0.789 (1.63) | +0.495 (2.39) | +0.808 (1.56) | +0.209 (2.99) |
| topological + Joback groups | +0.909 (1.07) | +0.548 (2.26) | +0.900 (1.12) | -0.093 (3.52) |
| invariant + topological | +0.880 (1.23) | +0.552 (2.25) | +0.885 (1.21) | +0.147 (3.11) |
| invariant + eight counts | +0.913 (1.05) | +0.590 (2.15) | +0.921 (1.00) | +0.441 (2.52) |

| Difference of q² or R² | Test | Mean [95 % interval] |
|---|---|---|
| invariant, cosine minus Joback refitted | PHYSPROP training part, CV | -0.019 [-0.031, -0.006] |
| invariant, cosine minus Joback refitted | eChemPortal, CV | +0.093 [+0.072, +0.116] |
| invariant, cosine minus Joback refitted | PHYSPROP test part (the OPERA test) | -0.003 [-0.027, +0.026] |
| invariant, cosine minus Joback refitted | PHYSPROP training part to eChemPortal | +0.643 [+0.531, +0.766] |
| invariant + Joback groups minus Joback refitted | PHYSPROP training part, CV | +0.021 [+0.014, +0.029] |
| invariant + Joback groups minus Joback refitted | eChemPortal, CV | +0.103 [+0.084, +0.123] |
| invariant + Joback groups minus Joback refitted | PHYSPROP test part (the OPERA test) | +0.027 [+0.010, +0.048] |
| invariant + Joback groups minus Joback refitted | PHYSPROP training part to eChemPortal | +0.205 [+0.160, +0.254] |
| invariant, cosine minus linear regression on molecular weight | PHYSPROP training part, CV | +0.294 [+0.257, +0.336] |
| invariant, cosine minus linear regression on molecular weight | eChemPortal, CV | +0.295 [+0.269, +0.323] |
| invariant, cosine minus linear regression on molecular weight | PHYSPROP test part (the OPERA test) | +0.320 [+0.262, +0.385] |
| invariant, cosine minus linear regression on molecular weight | PHYSPROP training part to eChemPortal | +0.279 [+0.224, +0.338] |
| invariant, not normalised minus invariant, cosine | PHYSPROP training part, CV | -0.015 [-0.040, +0.007] |
| invariant, not normalised minus invariant, cosine | eChemPortal, CV | -0.176 [-0.232, -0.129] |
| invariant, not normalised minus invariant, cosine | PHYSPROP test part (the OPERA test) | -0.025 [-0.059, +0.003] |
| invariant, not normalised minus invariant, cosine | PHYSPROP training part to eChemPortal | -0.975 [-1.185, -0.791] |
| invariant, cosine minus field totals | PHYSPROP training part, CV | +0.084 [+0.065, +0.104] |
| invariant, cosine minus field totals | eChemPortal, CV | +0.119 [+0.097, +0.142] |
| invariant, cosine minus field totals | PHYSPROP test part (the OPERA test) | +0.095 [+0.060, +0.131] |
| invariant, cosine minus field totals | PHYSPROP training part to eChemPortal | +0.015 [-0.041, +0.069] |
| invariant, cosine minus eight counts | PHYSPROP training part, CV | -0.017 [-0.031, -0.001] |
| invariant, cosine minus eight counts | eChemPortal, CV | -0.030 [-0.048, -0.010] |
| invariant, cosine minus eight counts | PHYSPROP test part (the OPERA test) | -0.034 [-0.054, -0.014] |
| invariant, cosine minus eight counts | PHYSPROP training part to eChemPortal | -0.201 [-0.257, -0.150] |
| invariant, cosine minus ECFP4 | PHYSPROP training part, CV | +0.085 [+0.069, +0.101] |
| invariant, cosine minus ECFP4 | eChemPortal, CV | +0.031 [+0.011, +0.052] |
| invariant, cosine minus ECFP4 | PHYSPROP test part (the OPERA test) | +0.105 [+0.074, +0.136] |
| invariant, cosine minus ECFP4 | PHYSPROP training part to eChemPortal | -0.118 [-0.176, -0.062] |
| invariant, cosine minus refitted Crippen increments | PHYSPROP training part, CV | -0.016 [-0.028, -0.003] |
| invariant, cosine minus refitted Crippen increments | eChemPortal, CV | +0.082 [+0.060, +0.104] |
| invariant, cosine minus refitted Crippen increments | PHYSPROP test part (the OPERA test) | -0.009 [-0.031, +0.016] |
| invariant, cosine minus refitted Crippen increments | PHYSPROP training part to eChemPortal | +0.663 [+0.548, +0.790] |
| invariant, cosine minus topological autocorrelation | PHYSPROP training part, CV | +0.088 [+0.071, +0.105] |
| invariant, cosine minus topological autocorrelation | eChemPortal, CV | +0.057 [+0.042, +0.073] |
| invariant, cosine minus topological autocorrelation | PHYSPROP test part (the OPERA test) | +0.065 [+0.039, +0.092] |
| invariant, cosine minus topological autocorrelation | PHYSPROP training part to eChemPortal | +0.067 [+0.029, +0.108] |
| invariant + Joback groups minus topological + Joback groups | PHYSPROP training part, CV | +0.008 [+0.001, +0.016] |
| invariant + Joback groups minus topological + Joback groups | eChemPortal, CV | +0.013 [+0.001, +0.026] |
| invariant + Joback groups minus topological + Joback groups | PHYSPROP test part (the OPERA test) | +0.002 [-0.011, +0.015] |
| invariant + Joback groups minus topological + Joback groups | PHYSPROP training part to eChemPortal | -0.069 [-0.103, -0.036] |
| topological autocorrelation minus refitted Crippen increments | PHYSPROP training part, CV | -0.103 [-0.123, -0.086] |
| topological autocorrelation minus refitted Crippen increments | eChemPortal, CV | +0.025 [-0.000, +0.050] |
| topological autocorrelation minus refitted Crippen increments | PHYSPROP test part (the OPERA test) | -0.074 [-0.105, -0.043] |
| topological autocorrelation minus refitted Crippen increments | PHYSPROP training part to eChemPortal | +0.596 [+0.469, +0.730] |
| invariant + topological minus invariant, cosine | PHYSPROP training part, CV | +0.007 [-0.001, +0.015] |
| invariant + topological minus invariant, cosine | eChemPortal, CV | +0.004 [-0.004, +0.012] |
| invariant + topological minus invariant, cosine | PHYSPROP test part (the OPERA test) | +0.012 [+0.001, +0.024] |
| invariant + topological minus invariant, cosine | PHYSPROP training part to eChemPortal | -0.129 [-0.162, -0.096] |
| invariant + eight counts minus invariant, cosine | PHYSPROP training part, CV | +0.040 [+0.030, +0.050] |
| invariant + eight counts minus invariant, cosine | eChemPortal, CV | +0.046 [+0.030, +0.060] |
| invariant + eight counts minus invariant, cosine | PHYSPROP test part (the OPERA test) | +0.048 [+0.032, +0.064] |
| invariant + eight counts minus invariant, cosine | PHYSPROP training part to eChemPortal | +0.165 [+0.122, +0.210] |
| invariant + eight counts minus eight counts | PHYSPROP training part, CV | +0.023 [+0.015, +0.032] |
| invariant + eight counts minus eight counts | eChemPortal, CV | +0.016 [+0.006, +0.026] |
| invariant + eight counts minus eight counts | PHYSPROP test part (the OPERA test) | +0.014 [+0.002, +0.026] |
| invariant + eight counts minus eight counts | PHYSPROP training part to eChemPortal | -0.036 [-0.057, -0.018] |
| invariant + Joback groups minus invariant + eight counts | PHYSPROP training part, CV | +0.000 [-0.009, +0.009] |
| invariant + Joback groups minus invariant + eight counts | eChemPortal, CV | -0.035 [-0.050, -0.020] |
| invariant + Joback groups minus invariant + eight counts | PHYSPROP test part (the OPERA test) | -0.019 [-0.035, -0.003] |
| invariant + Joback groups minus invariant + eight counts | PHYSPROP training part to eChemPortal | -0.603 [-0.717, -0.497] |
