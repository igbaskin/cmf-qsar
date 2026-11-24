# The continuous reaction field against its references (plan item PO3, stage 3)

5265 reactions of Cyclo-23-TS, reactant and product only. 2667 distinct reactant pairs, 25 distinct sets of bond changes; 0 reactions have their reverse in the set and are kept on one side of every split. Energies in kcal/mol. Every model is a Gaussian kernel on its blocks, fitted by kernel ridge regression, five folds; the settings are chosen inside the training part.

## Activation energy, whole reactants held out (up to 3 sets of folds)

| Model | Sets of folds | MAE | RMSE | q2 |
|---|---|---|---|---|
| CRF + counts + morgan, rows of unit length | 3 | 3.37 | 4.49 | +0.788 |
| counts + morgan | 3 | 3.61 | 4.79 | +0.758 |
| CRF + counts | 3 | 3.66 | 4.87 | +0.750 |
| morgan | 3 | 3.69 | 4.90 | +0.747 |
| rows of unit length, no superposition, + counts | 3 | 3.73 | 4.98 | +0.739 |
| CRF + counts, rows of unit length | 3 | 3.73 | 4.98 | +0.739 |
| CRF, rows of unit length | 3 | 3.81 | 5.11 | +0.726 |
| counts | 3 | 5.05 | 6.62 | +0.539 |
| not usable: linear in true dh | 3 | 5.27 | 6.85 | +0.506 |
| linear in dh predicted by CRF + counts, rows of unit length | 3 | 5.53 | 7.24 | +0.448 |
| mean | 3 | 7.64 | 9.75 | -0.001 |
| reactant mean | 3 | 7.64 | 9.75 | -0.001 |

| Difference (first minus second) | Sets of folds | in q2 [95 % interval] | in MAE [95 % interval] | in q2 on each set of folds |
|---|---|---|---|---|
| CRF + counts, rows of unit length minus counts + morgan | 3 | -0.019 [-0.030, -0.009] | +0.12 [+0.05, +0.20] | -0.011 [-0.023, +0.000]; -0.016 [-0.027, -0.005]; -0.030 [-0.042, -0.020] |
| CRF + counts minus counts | 3 | +0.211 [+0.191, +0.233] | -1.39 [-1.51, -1.28] | +0.208 [+0.185, +0.232]; +0.210 [+0.186, +0.233]; +0.216 [+0.195, +0.237] |
| CRF + counts minus CRF + counts, rows of unit length | 3 | +0.011 [+0.003, +0.019] | -0.08 [-0.12, -0.03] | +0.009 [-0.003, +0.019]; +0.006 [-0.007, +0.015]; +0.019 [+0.011, +0.028] |
| CRF + counts, rows of unit length minus counts | 3 | +0.200 [+0.182, +0.221] | -1.31 [-1.43, -1.20] | +0.200 [+0.180, +0.220]; +0.204 [+0.185, +0.227]; +0.197 [+0.177, +0.217] |
| CRF, rows of unit length minus morgan | 3 | -0.022 [-0.034, -0.010] | +0.12 [+0.04, +0.20] | -0.018 [-0.031, -0.004]; -0.019 [-0.032, -0.005]; -0.029 [-0.041, -0.015] |
| CRF + counts, rows of unit length minus rows of unit length, no superposition, + counts | 3 | -0.000 [-0.004, +0.004] | +0.00 [-0.03, +0.03] | +0.002 [-0.003, +0.007]; -0.003 [-0.008, +0.002]; -0.000 [-0.006, +0.005] |
| CRF + counts + morgan, rows of unit length minus counts + morgan | 3 | +0.030 [+0.024, +0.035] | -0.25 [-0.29, -0.20] | +0.035 [+0.028, +0.041]; +0.032 [+0.025, +0.040]; +0.022 [+0.016, +0.029] |

## Reaction energy, whole reactants held out (up to 3 sets of folds)

| Model | Sets of folds | MAE | RMSE | q2 |
|---|---|---|---|---|
| CRF + counts + morgan, rows of unit length | 3 | 3.61 | 4.84 | +0.950 |
| counts + morgan | 3 | 3.90 | 5.22 | +0.942 |
| CRF + counts | 3 | 4.25 | 5.71 | +0.930 |
| morgan | 3 | 4.27 | 5.69 | +0.931 |
| CRF + counts, rows of unit length | 3 | 4.38 | 5.83 | +0.927 |
| rows of unit length, no superposition, + counts | 3 | 4.38 | 5.83 | +0.927 |
| CRF, rows of unit length | 3 | 4.65 | 6.26 | +0.916 |
| counts | 3 | 8.63 | 11.14 | +0.735 |
| mean | 3 | 17.55 | 21.67 | -0.001 |
| reactant mean | 3 | 17.55 | 21.67 | -0.001 |

| Difference (first minus second) | Sets of folds | in q2 [95 % interval] | in MAE [95 % interval] | in q2 on each set of folds |
|---|---|---|---|---|
| CRF + counts, rows of unit length minus counts + morgan | 3 | -0.014 [-0.018, -0.011] | +0.48 [+0.36, +0.59] | -0.016 [-0.020, -0.012]; -0.013 [-0.018, -0.009]; -0.013 [-0.017, -0.009] |
| CRF + counts minus counts | 3 | +0.195 [+0.180, +0.211] | -4.38 [-4.62, -4.14] | +0.197 [+0.182, +0.214]; +0.194 [+0.178, +0.213]; +0.194 [+0.179, +0.212] |
| CRF + counts minus CRF + counts, rows of unit length | 3 | +0.003 [-0.001, +0.007] | -0.13 [-0.20, -0.05] | +0.004 [+0.001, +0.007]; +0.001 [-0.007, +0.006]; +0.004 [-0.001, +0.008] |
| CRF + counts, rows of unit length minus counts | 3 | +0.192 [+0.178, +0.208] | -4.25 [-4.47, -4.01] | +0.193 [+0.179, +0.211]; +0.193 [+0.178, +0.212]; +0.190 [+0.176, +0.208] |
| CRF, rows of unit length minus morgan | 3 | -0.014 [-0.020, -0.010] | +0.38 [+0.24, +0.51] | -0.013 [-0.017, -0.008]; -0.015 [-0.020, -0.009]; -0.016 [-0.021, -0.011] |
| CRF + counts, rows of unit length minus rows of unit length, no superposition, + counts | 3 | +0.000 [-0.001, +0.001] | -0.00 [-0.05, +0.04] | -0.001 [-0.003, +0.000]; +0.001 [-0.001, +0.002]; +0.001 [-0.001, +0.003] |
| CRF + counts + morgan, rows of unit length minus counts + morgan | 3 | +0.008 [+0.007, +0.010] | -0.29 [-0.35, -0.24] | +0.008 [+0.006, +0.010]; +0.007 [+0.005, +0.010]; +0.009 [+0.007, +0.012] |

### Correlation of whole kernels, whole reactants held out

Centred alignment on 2,000 reactions, with the settings each model chose for the activation energy in the first fold.

| Two models | Correlation |
|---|---|
| CRF + counts, rows of unit length | counts + morgan | 0.711 |
| CRF + counts, rows of unit length | CRF, rows of unit length | 0.899 |
| CRF + counts, rows of unit length | counts | 0.859 |
| CRF + counts, rows of unit length | morgan | 0.377 |
| counts + morgan | CRF, rows of unit length | 0.578 |
| counts + morgan | counts | 0.667 |
| counts + morgan | morgan | 0.870 |
| CRF, rows of unit length | counts | 0.562 |
| CRF, rows of unit length | morgan | 0.408 |
| counts | morgan | 0.226 |

### Settings the search chose, activation energy, whole reactants held out

| Model | Width factor | Ridge | Block weights |
|---|---|---|---|
| CRF + counts, rows of unit length | 0.1/0.3/1 | 0.01/0.1 | odd_n 1/10; conv_n 1/10; dyn_n 0.1/1; bond_n 0.1/1/10; counts 0.1/1/10 |
| counts + morgan | 0.1/0.3/1 | 0.001/0.01/0.1 | counts 0.1/1/10; mdiff 0.1/1/10; msum 1/10 |
| counts | 0.1/0.3 | 0.01/0.1 | counts 1 |
| morgan | 0.1/0.3/1 | 0.01/0.1 | mdiff 0.1/1/10; msum 0.1/1 |
| CRF, rows of unit length | 0.1/0.3/1 | 1e-05/0.01/0.1 | odd_n 1/10; conv_n 1/10; dyn_n 0.1/1/10; bond_n 0.1/1/10 |
| CRF + counts | 0.1/0.3/1 | 0.01/0.1 | odd 0.1/1/10; conv_h 1/10; dyn_h 0.1/1; bond_h 0.1/1/10; counts 0.1/1/10 |
| rows of unit length, no superposition, + counts | 0.1/0.3/1 | 0.01/0.1 | odd_n 0.1/1/10; even0_n 1/10; counts 0.1/1 |
| CRF + counts + morgan, rows of unit length | 0.1/0.3/1 | 1e-05/0.001/0.01/0.1 | odd_n 0.1/1/10; conv_n 1/10; dyn_n 0.1/1; bond_n 0.1/1; counts 0.1/1/10; mdiff 0.1/1/10; msum 0.1/1/10 |

Test reactions whose reactant is also in the training part, whole reactants held out: [0, 0, 0] (per repeat).

## Activation energy, random folds (up to 3 sets of folds)

| Model | Sets of folds | MAE | RMSE | q2 |
|---|---|---|---|---|
| CRF + counts + morgan, rows of unit length | 3 | 3.00 | 4.10 | +0.823 |
| counts + morgan | 3 | 3.26 | 4.42 | +0.795 |
| morgan | 3 | 3.33 | 4.54 | +0.783 |
| CRF + counts | 3 | 3.37 | 4.56 | +0.781 |
| CRF + counts, rows of unit length | 3 | 3.43 | 4.64 | +0.774 |
| rows of unit length, no superposition, + counts | 3 | 3.44 | 4.64 | +0.774 |
| CRF, rows of unit length | 3 | 3.50 | 4.74 | +0.763 |
| counts | 3 | 4.88 | 6.40 | +0.569 |
| reactant mean | 3 | 5.07 | 7.28 | +0.442 |
| not usable: linear in true dh | 3 | 5.26 | 6.85 | +0.506 |
| linear in dh predicted by CRF + counts, rows of unit length | 3 | 5.45 | 7.16 | +0.461 |
| mean | 3 | 7.64 | 9.75 | -0.000 |

| Difference (first minus second) | Sets of folds | in q2 [95 % interval] | in MAE [95 % interval] | in q2 on each set of folds |
|---|---|---|---|---|
| CRF + counts, rows of unit length minus counts + morgan | 3 | -0.021 [-0.030, -0.011] | +0.16 [+0.09, +0.23] | -0.019 [-0.030, -0.008]; -0.021 [-0.031, -0.011]; -0.023 [-0.034, -0.012] |
| CRF + counts minus counts | 3 | +0.211 [+0.192, +0.232] | -1.51 [-1.62, -1.40] | +0.218 [+0.200, +0.239]; +0.210 [+0.188, +0.232]; +0.206 [+0.183, +0.229] |
| CRF + counts minus CRF + counts, rows of unit length | 3 | +0.007 [+0.000, +0.013] | -0.06 [-0.10, -0.02] | +0.014 [+0.007, +0.020]; +0.003 [-0.005, +0.009]; +0.004 [-0.008, +0.014] |
| CRF + counts, rows of unit length minus counts | 3 | +0.205 [+0.186, +0.224] | -1.45 [-1.56, -1.35] | +0.204 [+0.187, +0.224]; +0.207 [+0.190, +0.227]; +0.202 [+0.185, +0.222] |
| CRF, rows of unit length minus morgan | 3 | -0.020 [-0.030, -0.009] | +0.17 [+0.09, +0.25] | -0.015 [-0.027, -0.004]; -0.022 [-0.035, -0.009]; -0.022 [-0.033, -0.010] |
| CRF + counts, rows of unit length minus rows of unit length, no superposition, + counts | 3 | +0.000 [-0.003, +0.003] | -0.02 [-0.04, +0.01] | -0.001 [-0.006, +0.003]; +0.001 [-0.002, +0.005]; +0.000 [-0.006, +0.005] |
| CRF + counts + morgan, rows of unit length minus counts + morgan | 3 | +0.028 [+0.023, +0.034] | -0.26 [-0.30, -0.21] | +0.029 [+0.024, +0.033]; +0.031 [+0.025, +0.037]; +0.026 [+0.019, +0.033] |

## Reaction energy, random folds (up to 3 sets of folds)

| Model | Sets of folds | MAE | RMSE | q2 |
|---|---|---|---|---|
| CRF + counts + morgan, rows of unit length | 3 | 2.98 | 4.06 | +0.965 |
| counts + morgan | 3 | 3.11 | 4.25 | +0.961 |
| morgan | 3 | 3.34 | 4.59 | +0.955 |
| CRF + counts | 3 | 3.78 | 5.08 | +0.945 |
| rows of unit length, no superposition, + counts | 3 | 3.82 | 5.12 | +0.944 |
| CRF + counts, rows of unit length | 3 | 3.83 | 5.13 | +0.944 |
| CRF, rows of unit length | 3 | 4.00 | 5.39 | +0.938 |
| counts | 3 | 7.98 | 10.37 | +0.771 |
| reactant mean | 3 | 8.80 | 13.50 | +0.611 |
| mean | 3 | 17.54 | 21.66 | -0.000 |

| Difference (first minus second) | Sets of folds | in q2 [95 % interval] | in MAE [95 % interval] | in q2 on each set of folds |
|---|---|---|---|---|
| CRF + counts, rows of unit length minus counts + morgan | 3 | -0.017 [-0.020, -0.015] | +0.71 [+0.62, +0.81] | -0.017 [-0.021, -0.014]; -0.018 [-0.021, -0.015]; -0.017 [-0.020, -0.014] |
| CRF + counts minus counts | 3 | +0.174 [+0.161, +0.189] | -4.20 [-4.40, -3.97] | +0.169 [+0.154, +0.185]; +0.180 [+0.165, +0.198]; +0.174 [+0.162, +0.190] |
| CRF + counts minus CRF + counts, rows of unit length | 3 | +0.001 [-0.002, +0.004] | -0.05 [-0.11, +0.01] | +0.000 [-0.006, +0.005]; +0.000 [-0.003, +0.004]; +0.003 [+0.001, +0.005] |
| CRF + counts, rows of unit length minus counts | 3 | +0.173 [+0.160, +0.188] | -4.15 [-4.36, -3.93] | +0.169 [+0.155, +0.184]; +0.180 [+0.165, +0.198]; +0.172 [+0.158, +0.187] |
| CRF, rows of unit length minus morgan | 3 | -0.017 [-0.020, -0.014] | +0.66 [+0.56, +0.77] | -0.016 [-0.020, -0.012]; -0.019 [-0.022, -0.015]; -0.016 [-0.020, -0.013] |
| CRF + counts, rows of unit length minus rows of unit length, no superposition, + counts | 3 | -0.000 [-0.001, +0.001] | +0.01 [-0.03, +0.05] | +0.003 [+0.001, +0.004]; -0.000 [-0.002, +0.001]; -0.003 [-0.004, -0.001] |
| CRF + counts + morgan, rows of unit length minus counts + morgan | 3 | +0.003 [+0.002, +0.005] | -0.13 [-0.18, -0.08] | +0.004 [+0.002, +0.005]; +0.003 [+0.002, +0.005]; +0.003 [+0.002, +0.005] |

### Correlation of whole kernels, random folds

Centred alignment on 2,000 reactions, with the settings each model chose for the activation energy in the first fold.

| Two models | Correlation |
|---|---|
| CRF + counts, rows of unit length | counts + morgan | 0.739 |
| CRF + counts, rows of unit length | CRF, rows of unit length | 0.820 |
| CRF + counts, rows of unit length | counts | 0.836 |
| CRF + counts, rows of unit length | morgan | 0.462 |
| counts + morgan | CRF, rows of unit length | 0.644 |
| counts + morgan | counts | 0.675 |
| counts + morgan | morgan | 0.898 |
| CRF, rows of unit length | counts | 0.505 |
| CRF, rows of unit length | morgan | 0.539 |
| counts | morgan | 0.291 |

### Settings the search chose, activation energy, random folds

| Model | Width factor | Ridge | Block weights |
|---|---|---|---|
| CRF + counts, rows of unit length | 0.1/0.3/1 | 0.01/0.1 | odd_n 1/10; conv_n 1/10; dyn_n 0.1/1/10; bond_n 0.1/1; counts 1/10 |
| counts + morgan | 0.1/0.3/1 | 0.01/0.1 | counts 0.1/1; mdiff 0.1/1/10; msum 0.1/1/10 |
| counts | 0.1/0.3/1 | 0.001/0.01/0.1 | counts 1 |
| morgan | 0.1/0.3/1 | 0.01/0.1 | mdiff 0.1/1/10; msum 0.1/1 |
| CRF, rows of unit length | 0.1/0.3/1 | 1e-05/0.001/0.01/0.1 | odd_n 0.1/1/10; conv_n 1/10; dyn_n 0.1/1/10; bond_n 0.1/1 |
| CRF + counts | 0.1/0.3/1 | 0.01/0.1 | odd 0.1/1/10; conv_h 1/10; dyn_h 0.1/1; bond_h 0.1/1; counts 0.1/1/10 |
| rows of unit length, no superposition, + counts | 0.1/0.3/1 | 0.01/0.1 | odd_n 0.1/1/10; even0_n 1; counts 0.1/1 |
| CRF + counts + morgan, rows of unit length | 0.3/1 | 1e-05/0.001/0.01 | odd_n 0.1/1; conv_n 1/10; dyn_n 0.1/1; bond_n 0.1; counts 0.1/1; mdiff 1/10; msum 0.1/1/10 |

Test reactions whose reactant is also in the training part, random folds: [3655, 3707, 3792] (per repeat).

