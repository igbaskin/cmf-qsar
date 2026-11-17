Measured normal boiling points, C: the PHYSPROP set curated for OPERA, its training part (4059 compounds, spread 85.3) and test part (1353), and boiling points from eChemPortal (1617 compounds, spread 85.6). Five folds repeated three times on the training part and on the eChemPortal set; the external tests fit on the whole training part. The setting and the ridge are chosen by the evidence on the training block. Each cell gives q² (cross-validation) or R² (external test), with the RMSE in C in brackets.

| Model | PHYSPROP training part, CV | eChemPortal, CV | PHYSPROP test part (the OPERA test) | PHYSPROP training part to eChemPortal |
|---|---:|---:|---:|---:|
| invariant, cosine | +0.909 (25.7) | +0.448 (63.6) | +0.915 (24.5) | +0.291 (72.1) |
| invariant, not normalised | +0.895 (27.6) | +0.393 (66.6) | +0.892 (27.6) | -0.316 (98.2) |
| invariant + Joback groups | +0.945 (20.1) | +0.514 (59.7) | +0.947 (19.3) | -0.165 (92.4) |
| linear regression on molecular weight | +0.303 (71.2) | +0.183 (77.4) | +0.218 (74.2) | -0.039 (87.3) |
| Joback refitted | +0.882 (29.3) | +0.459 (62.9) | +0.881 (29.0) | -0.963 (119.9) |
| field totals | +0.670 (49.0) | +0.377 (67.6) | +0.677 (47.7) | +0.352 (68.9) |
| eight counts | +0.856 (32.4) | +0.517 (59.5) | +0.862 (31.2) | +0.534 (58.4) |
| ECFP4 | +0.762 (41.6) | +0.471 (62.2) | +0.785 (38.9) | +0.252 (74.0) |
| refitted Crippen increments | +0.889 (28.4) | +0.447 (63.6) | +0.890 (27.8) | -0.970 (120.1) |
| topological autocorrelation | +0.750 (42.7) | +0.477 (61.9) | +0.762 (40.9) | +0.258 (73.7) |
| topological + Joback groups | +0.927 (23.1) | +0.549 (57.5) | +0.928 (22.5) | -0.331 (98.7) |

| Published Joback equation, no fitting | Compounds covered | R² | RMSE, C |
|---|---:|---:|---:|
| physprop_train | 3761 of 4059 | +0.738 | 43.9 |
| physprop_test | 1268 of 1353 | +0.760 | 40.9 |
| echemportal | 1324 of 1617 | -2.683 | 161.3 |

| Difference of q² or R² | Test | Mean [95 % interval] |
|---|---|---|
| invariant, cosine minus Joback refitted | PHYSPROP training part, CV | +0.030 [+0.020, +0.039] |
| invariant, cosine minus Joback refitted | eChemPortal, CV | +0.020 [-0.018, +0.060] |
| invariant, cosine minus Joback refitted | PHYSPROP test part (the OPERA test) | +0.034 [+0.019, +0.048] |
| invariant, cosine minus Joback refitted | PHYSPROP training part to eChemPortal | +1.253 [+0.942, +1.611] |
| invariant + Joback groups minus Joback refitted | PHYSPROP training part, CV | +0.064 [+0.057, +0.072] |
| invariant + Joback groups minus Joback refitted | eChemPortal, CV | +0.073 [+0.048, +0.098] |
| invariant + Joback groups minus Joback refitted | PHYSPROP test part (the OPERA test) | +0.066 [+0.056, +0.077] |
| invariant + Joback groups minus Joback refitted | PHYSPROP training part to eChemPortal | +0.798 [+0.617, +1.006] |
| invariant, cosine minus linear regression on molecular weight | PHYSPROP training part, CV | +0.609 [+0.570, +0.648] |
| invariant, cosine minus linear regression on molecular weight | eChemPortal, CV | +0.299 [+0.249, +0.352] |
| invariant, cosine minus linear regression on molecular weight | PHYSPROP test part (the OPERA test) | +0.697 [+0.614, +0.790] |
| invariant, cosine minus linear regression on molecular weight | PHYSPROP training part to eChemPortal | +0.330 [+0.241, +0.425] |
| invariant, not normalised minus invariant, cosine | PHYSPROP training part, CV | -0.007 [-0.025, +0.006] |
| invariant, not normalised minus invariant, cosine | eChemPortal, CV | -0.045 [-0.099, +0.004] |
| invariant, not normalised minus invariant, cosine | PHYSPROP test part (the OPERA test) | -0.023 [-0.053, +0.000] |
| invariant, not normalised minus invariant, cosine | PHYSPROP training part to eChemPortal | -0.607 [-0.815, -0.411] |
| invariant, cosine minus field totals | PHYSPROP training part, CV | +0.236 [+0.218, +0.255] |
| invariant, cosine minus field totals | eChemPortal, CV | +0.095 [+0.050, +0.139] |
| invariant, cosine minus field totals | PHYSPROP test part (the OPERA test) | +0.238 [+0.205, +0.275] |
| invariant, cosine minus field totals | PHYSPROP training part to eChemPortal | -0.062 [-0.172, +0.039] |
| invariant, cosine minus eight counts | PHYSPROP training part, CV | +0.054 [+0.042, +0.067] |
| invariant, cosine minus eight counts | eChemPortal, CV | -0.053 [-0.100, -0.007] |
| invariant, cosine minus eight counts | PHYSPROP test part (the OPERA test) | +0.053 [+0.033, +0.074] |
| invariant, cosine minus eight counts | PHYSPROP training part to eChemPortal | -0.243 [-0.349, -0.146] |
| invariant, cosine minus ECFP4 | PHYSPROP training part, CV | +0.147 [+0.134, +0.160] |
| invariant, cosine minus ECFP4 | eChemPortal, CV | -0.002 [-0.046, +0.041] |
| invariant, cosine minus ECFP4 | PHYSPROP test part (the OPERA test) | +0.130 [+0.106, +0.156] |
| invariant, cosine minus ECFP4 | PHYSPROP training part to eChemPortal | +0.039 [-0.073, +0.146] |
| invariant, cosine minus refitted Crippen increments | PHYSPROP training part, CV | +0.022 [+0.012, +0.032] |
| invariant, cosine minus refitted Crippen increments | eChemPortal, CV | +0.031 [-0.010, +0.074] |
| invariant, cosine minus refitted Crippen increments | PHYSPROP test part (the OPERA test) | +0.025 [+0.010, +0.040] |
| invariant, cosine minus refitted Crippen increments | PHYSPROP training part to eChemPortal | +1.260 [+0.957, +1.610] |
| invariant, cosine minus topological autocorrelation | PHYSPROP training part, CV | +0.161 [+0.146, +0.176] |
| invariant, cosine minus topological autocorrelation | eChemPortal, CV | -0.007 [-0.044, +0.032] |
| invariant, cosine minus topological autocorrelation | PHYSPROP test part (the OPERA test) | +0.153 [+0.127, +0.180] |
| invariant, cosine minus topological autocorrelation | PHYSPROP training part to eChemPortal | +0.033 [-0.029, +0.094] |
| invariant + Joback groups minus topological + Joback groups | PHYSPROP training part, CV | +0.019 [+0.014, +0.023] |
| invariant + Joback groups minus topological + Joback groups | eChemPortal, CV | -0.027 [-0.056, +0.004] |
| invariant + Joback groups minus topological + Joback groups | PHYSPROP test part (the OPERA test) | +0.019 [+0.011, +0.026] |
| invariant + Joback groups minus topological + Joback groups | PHYSPROP training part to eChemPortal | +0.166 [+0.089, +0.249] |
| topological autocorrelation minus refitted Crippen increments | PHYSPROP training part, CV | -0.138 [-0.157, -0.121] |
| topological autocorrelation minus refitted Crippen increments | eChemPortal, CV | +0.038 [-0.013, +0.090] |
| topological autocorrelation minus refitted Crippen increments | PHYSPROP test part (the OPERA test) | -0.128 [-0.156, -0.103] |
| topological autocorrelation minus refitted Crippen increments | PHYSPROP training part to eChemPortal | +1.228 [+0.900, +1.599] |
