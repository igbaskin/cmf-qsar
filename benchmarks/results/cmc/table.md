Measured critical micelle concentrations, pCMC = -log10(CMC in mol/L), from SurfPro: the training part (1203 surfactants in 405 series, spread 1.08) and the paper's test set (140, spread 1.11). Five folds repeated three times on the training part, once with random folds and once with every series in one fold; the test set is predicted by a fit on the whole training part. The setting and the ridge are chosen by the evidence on the training block. Each cell gives q² (cross-validation) or R² (test set), with the RMSE in pCMC units in brackets; the last column is the mean absolute error (MAE) on the test set.

| Model | training part, CV, random folds | training part, CV, series held out | the SurfPro test set | MAE, test set |
|---|---:|---:|---:|---:|
| invariant, cosine | +0.585 (0.699) | +0.516 (0.754) | +0.629 (0.677) | 0.544 |
| invariant, not normalised | +0.547 (0.729) | +0.445 (0.807) | +0.601 (0.702) | 0.533 |
| invariant + Klevens descriptors | +0.641 (0.649) | +0.557 (0.721) | +0.693 (0.616) | 0.476 |
| invariant + eight counts | +0.732 (0.562) | +0.550 (0.726) | +0.836 (0.450) | 0.331 |
| invariant + topological | +0.657 (0.635) | +0.441 (0.810) | +0.737 (0.570) | 0.442 |
| Klevens refitted | +0.478 (0.783) | +0.445 (0.807) | +0.522 (0.768) | 0.601 |
| linear regression on the tail length | +0.164 (0.991) | +0.153 (0.997) | +0.249 (0.963) | 0.763 |
| field totals | +0.684 (0.609) | +0.613 (0.674) | +0.791 (0.508) | 0.383 |
| eight counts | +0.675 (0.618) | +0.413 (0.830) | +0.823 (0.468) | 0.339 |
| ECFP4 | +0.395 (0.843) | +0.288 (0.914) | +0.414 (0.851) | 0.646 |
| refitted Crippen increments | +0.526 (0.746) | +0.433 (0.816) | +0.574 (0.726) | 0.569 |
| topological autocorrelation | +0.628 (0.661) | +0.384 (0.850) | +0.720 (0.588) | 0.463 |
| topological + Klevens descriptors | +0.681 (0.612) | +0.480 (0.782) | +0.762 (0.542) | 0.395 |

Published errors on the test set (Table 3 of the SurfPro paper, on the first version of the data):

| Published model | MAE | RMSE |
|---|---:|---:|
| AttentiveFP, best (all properties, 96 channels) | 0.235 | 0.346 |
| AttentiveFP, one property, 64 channels | 0.250 | 0.382 |
| RDKit fingerprint, random forest | 0.630 | 0.840 |

| Difference of q² or R² | Test | Mean [95 % interval] |
|---|---|---|
| invariant, cosine minus Klevens refitted | training part, CV, random folds | +0.139 [+0.096, +0.182] |
| invariant, cosine minus Klevens refitted | training part, CV, series held out | +0.105 [+0.025, +0.180] |
| invariant, cosine minus Klevens refitted | the SurfPro test set | +0.107 [+0.008, +0.212] |
| invariant + Klevens descriptors minus Klevens refitted | training part, CV, random folds | +0.186 [+0.153, +0.220] |
| invariant + Klevens descriptors minus Klevens refitted | training part, CV, series held out | +0.146 [+0.086, +0.210] |
| invariant + Klevens descriptors minus Klevens refitted | the SurfPro test set | +0.171 [+0.096, +0.257] |
| invariant, cosine minus linear regression on the tail length | training part, CV, random folds | +0.456 [+0.408, +0.505] |
| invariant, cosine minus linear regression on the tail length | training part, CV, series held out | +0.402 [+0.304, +0.497] |
| invariant, cosine minus linear regression on the tail length | the SurfPro test set | +0.380 [+0.268, +0.500] |
| invariant, not normalised minus invariant, cosine | training part, CV, random folds | -0.060 [-0.089, -0.030] |
| invariant, not normalised minus invariant, cosine | training part, CV, series held out | -0.079 [-0.138, -0.030] |
| invariant, not normalised minus invariant, cosine | the SurfPro test set | -0.028 [-0.108, +0.033] |
| invariant, cosine minus field totals | training part, CV, random folds | -0.069 [-0.101, -0.036] |
| invariant, cosine minus field totals | training part, CV, series held out | -0.072 [-0.123, -0.024] |
| invariant, cosine minus field totals | the SurfPro test set | -0.162 [-0.258, -0.070] |
| invariant, cosine minus eight counts | training part, CV, random folds | -0.069 [-0.107, -0.033] |
| invariant, cosine minus eight counts | training part, CV, series held out | +0.112 [+0.015, +0.214] |
| invariant, cosine minus eight counts | the SurfPro test set | -0.194 [-0.304, -0.100] |
| invariant, cosine minus ECFP4 | training part, CV, random folds | +0.218 [+0.172, +0.268] |
| invariant, cosine minus ECFP4 | training part, CV, series held out | +0.255 [+0.158, +0.349] |
| invariant, cosine minus ECFP4 | the SurfPro test set | +0.215 [+0.069, +0.362] |
| invariant, cosine minus refitted Crippen increments | training part, CV, random folds | +0.092 [+0.055, +0.130] |
| invariant, cosine minus refitted Crippen increments | training part, CV, series held out | +0.105 [+0.019, +0.184] |
| invariant, cosine minus refitted Crippen increments | the SurfPro test set | +0.056 [-0.052, +0.150] |
| invariant, cosine minus topological autocorrelation | training part, CV, random folds | -0.014 [-0.039, +0.011] |
| invariant, cosine minus topological autocorrelation | training part, CV, series held out | +0.133 [+0.046, +0.236] |
| invariant, cosine minus topological autocorrelation | the SurfPro test set | -0.091 [-0.152, -0.033] |
| invariant + Klevens descriptors minus topological + Klevens descriptors | training part, CV, random folds | -0.021 [-0.043, +0.001] |
| invariant + Klevens descriptors minus topological + Klevens descriptors | training part, CV, series held out | +0.086 [+0.028, +0.159] |
| invariant + Klevens descriptors minus topological + Klevens descriptors | the SurfPro test set | -0.069 [-0.119, -0.022] |
| invariant + topological minus invariant, cosine | training part, CV, random folds | +0.056 [+0.034, +0.077] |
| invariant + topological minus invariant, cosine | training part, CV, series held out | -0.068 [-0.172, +0.013] |
| invariant + topological minus invariant, cosine | the SurfPro test set | +0.108 [+0.063, +0.162] |
| invariant + eight counts minus invariant, cosine | training part, CV, random folds | +0.131 [+0.105, +0.158] |
| invariant + eight counts minus invariant, cosine | training part, CV, series held out | +0.031 [-0.042, +0.098] |
| invariant + eight counts minus invariant, cosine | the SurfPro test set | +0.207 [+0.126, +0.305] |
| invariant + eight counts minus eight counts | training part, CV, random folds | +0.062 [+0.041, +0.082] |
| invariant + eight counts minus eight counts | training part, CV, series held out | +0.143 [+0.076, +0.214] |
| invariant + eight counts minus eight counts | the SurfPro test set | +0.013 [-0.025, +0.051] |
| invariant + Klevens descriptors minus invariant + eight counts | training part, CV, random folds | -0.084 [-0.109, -0.059] |
| invariant + Klevens descriptors minus invariant + eight counts | training part, CV, series held out | +0.011 [-0.052, +0.078] |
| invariant + Klevens descriptors minus invariant + eight counts | the SurfPro test set | -0.143 [-0.238, -0.063] |
