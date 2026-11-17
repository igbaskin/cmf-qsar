The four physical-property benchmarks with a model that needs no conformer: the topological autocorrelation (the five fields, distances counted in bonds) plus h times the Gaussian kernel on the eight counts. Same compounds, folds and evidence selection as the benchmarks; the rows of *invariant + eight counts* are those of the benchmarks. Each cell gives q² (cross-validation) or R² (external test), with the RMSE in brackets.

### logP

| Model | Huuskonen, CV | class P, CV | class P to Huuskonen |
|---|---:|---:|---:|
| eight counts | +0.806 (0.658) | +0.672 (1.02) | +0.685 (0.839) |
| topological autocorrelation | +0.871 (0.537) | +0.761 (0.87) | +0.782 (0.697) |
| topological + eight counts | +0.899 (0.476) | +0.804 (0.787) | +0.821 (0.632) |
| invariant + eight counts | +0.906 (0.458) | +0.810 (0.775) | +0.841 (0.595) |

| Difference of q² or R² | Test | Mean [95 % interval] |
|---|---|---|
| topological + eight counts minus eight counts | Huuskonen, CV | +0.092 [+0.078, +0.107] |
| topological + eight counts minus eight counts | class P, CV | +0.124 [+0.106, +0.141] |
| topological + eight counts minus eight counts | class P to Huuskonen | +0.137 [+0.117, +0.158] |
| topological + eight counts minus topological autocorrelation | Huuskonen, CV | +0.030 [+0.022, +0.038] |
| topological + eight counts minus topological autocorrelation | class P, CV | +0.048 [+0.038, +0.058] |
| topological + eight counts minus topological autocorrelation | class P to Huuskonen | +0.039 [+0.029, +0.050] |
| invariant + eight counts minus topological + eight counts | Huuskonen, CV | +0.010 [+0.004, +0.017] |
| invariant + eight counts minus topological + eight counts | class P, CV | +0.009 [+0.002, +0.017] |
| invariant + eight counts minus topological + eight counts | class P to Huuskonen | +0.020 [+0.010, +0.031] |

### water solubility (logS)

| Model | Delaney, CV | AqSolDB, CV | AqSolDB to Delaney |
|---|---:|---:|---:|
| eight counts | +0.853 (0.804) | +0.832 (0.945) | +0.778 (0.987) |
| topological autocorrelation | +0.847 (0.819) | +0.815 (0.992) | +0.811 (0.91) |
| topological + eight counts | +0.902 (0.655) | +0.869 (0.833) | +0.868 (0.762) |
| invariant + eight counts | +0.906 (0.64) | +0.879 (0.802) | +0.867 (0.763) |

| Difference of q² or R² | Test | Mean [95 % interval] |
|---|---|---|
| topological + eight counts minus eight counts | Delaney, CV | +0.048 [+0.036, +0.062] |
| topological + eight counts minus eight counts | AqSolDB, CV | +0.040 [+0.027, +0.053] |
| topological + eight counts minus eight counts | AqSolDB to Delaney | +0.090 [+0.072, +0.110] |
| topological + eight counts minus topological autocorrelation | Delaney, CV | +0.057 [+0.045, +0.070] |
| topological + eight counts minus topological autocorrelation | AqSolDB, CV | +0.058 [+0.043, +0.073] |
| topological + eight counts minus topological autocorrelation | AqSolDB to Delaney | +0.057 [+0.038, +0.075] |
| invariant + eight counts minus topological + eight counts | Delaney, CV | +0.006 [-0.001, +0.012] |
| invariant + eight counts minus topological + eight counts | AqSolDB, CV | +0.011 [+0.001, +0.020] |
| invariant + eight counts minus topological + eight counts | AqSolDB to Delaney | -0.001 [-0.009, +0.009] |

### boiling point, C

| Model | PHYSPROP training part, CV | eChemPortal, CV | PHYSPROP test part (the OPERA test) | PHYSPROP training part to eChemPortal |
|---|---:|---:|---:|---:|
| eight counts | +0.856 (32.4) | +0.517 (59.5) | +0.862 (31.2) | +0.534 (58.4) |
| topological autocorrelation | +0.750 (42.7) | +0.477 (61.9) | +0.762 (40.9) | +0.258 (73.7) |
| topological + eight counts | +0.925 (23.4) | +0.582 (55.3) | +0.933 (21.7) | +0.427 (64.8) |
| invariant + eight counts | +0.943 (20.4) | +0.543 (57.9) | +0.957 (17.4) | +0.569 (56.2) |

| Difference of q² or R² | Test | Mean [95 % interval] |
|---|---|---|
| topological + eight counts minus eight counts | PHYSPROP training part, CV | +0.069 [+0.060, +0.081] |
| topological + eight counts minus eight counts | eChemPortal, CV | +0.066 [+0.037, +0.095] |
| topological + eight counts minus eight counts | PHYSPROP test part (the OPERA test) | +0.071 [+0.055, +0.089] |
| topological + eight counts minus eight counts | PHYSPROP training part to eChemPortal | -0.107 [-0.182, -0.038] |
| topological + eight counts minus topological autocorrelation | PHYSPROP training part, CV | +0.176 [+0.161, +0.192] |
| topological + eight counts minus topological autocorrelation | eChemPortal, CV | +0.112 [+0.075, +0.150] |
| topological + eight counts minus topological autocorrelation | PHYSPROP test part (the OPERA test) | +0.171 [+0.148, +0.196] |
| topological + eight counts minus topological autocorrelation | PHYSPROP training part to eChemPortal | +0.169 [+0.127, +0.213] |
| invariant + eight counts minus topological + eight counts | PHYSPROP training part, CV | +0.018 [+0.014, +0.022] |
| invariant + eight counts minus topological + eight counts | eChemPortal, CV | -0.028 [-0.051, -0.007] |
| invariant + eight counts minus topological + eight counts | PHYSPROP test part (the OPERA test) | +0.024 [+0.017, +0.032] |
| invariant + eight counts minus topological + eight counts | PHYSPROP training part to eChemPortal | +0.142 [+0.096, +0.194] |

### vapour pressure (logVP)

| Model | PHYSPROP training part, CV | eChemPortal, CV | PHYSPROP test part (the OPERA test) | PHYSPROP training part to eChemPortal |
|---|---:|---:|---:|---:|
| eight counts | +0.893 (1.16) | +0.578 (2.19) | +0.907 (1.08) | +0.477 (2.43) |
| topological autocorrelation | +0.789 (1.63) | +0.495 (2.39) | +0.808 (1.56) | +0.209 (2.99) |
| topological + eight counts | +0.906 (1.09) | +0.588 (2.16) | +0.911 (1.06) | +0.383 (2.64) |
| invariant + eight counts | +0.913 (1.05) | +0.590 (2.15) | +0.921 (0.998) | +0.441 (2.52) |

| Difference of q² or R² | Test | Mean [95 % interval] |
|---|---|---|
| topological + eight counts minus eight counts | PHYSPROP training part, CV | +0.014 [+0.008, +0.022] |
| topological + eight counts minus eight counts | eChemPortal, CV | +0.012 [+0.004, +0.021] |
| topological + eight counts minus eight counts | PHYSPROP test part (the OPERA test) | +0.004 [-0.006, +0.014] |
| topological + eight counts minus eight counts | PHYSPROP training part to eChemPortal | -0.094 [-0.123, -0.067] |
| topological + eight counts minus topological autocorrelation | PHYSPROP training part, CV | +0.119 [+0.102, +0.137] |
| topological + eight counts minus topological autocorrelation | eChemPortal, CV | +0.099 [+0.080, +0.118] |
| topological + eight counts minus topological autocorrelation | PHYSPROP test part (the OPERA test) | +0.103 [+0.074, +0.129] |
| topological + eight counts minus topological autocorrelation | PHYSPROP training part to eChemPortal | +0.174 [+0.132, +0.217] |
| invariant + eight counts minus topological + eight counts | PHYSPROP training part, CV | +0.008 [+0.003, +0.014] |
| invariant + eight counts minus topological + eight counts | eChemPortal, CV | +0.004 [-0.005, +0.013] |
| invariant + eight counts minus topological + eight counts | PHYSPROP test part (the OPERA test) | +0.010 [+0.001, +0.019] |
| invariant + eight counts minus topological + eight counts | PHYSPROP training part to eChemPortal | +0.058 [+0.036, +0.079] |
