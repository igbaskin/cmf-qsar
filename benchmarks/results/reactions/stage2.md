# The continuous reaction field against its references (plan item PO3, stage 2)

11961 reactions of the Grambow set, reactant and product only. 1156 distinct reactants, 1458 distinct sets of bond changes; 116 reactions have their reverse in the set and are kept on one side of every split. Energies in kcal/mol. Every model is a Gaussian kernel on its blocks, fitted by kernel ridge regression, five folds; the settings are chosen inside the training part.

## Activation energy, whole reactants held out (up to 3 sets of folds)

| Model | Sets of folds | MAE | RMSE | q2 |
|---|---|---|---|---|
| CRF + counts, rows of unit length | 3 | 7.18 | 10.70 | +0.779 |
| not usable: CRF + counts + true dh | 1 | 7.20 | 10.45 | +0.789 |
| not usable: counts + morgan + true dh | 1 | 7.30 | 10.57 | +0.784 |
| CRF + counts + morgan | 1 | 7.40 | 10.82 | +0.774 |
| CRF + counts | 3 | 7.74 | 11.36 | +0.751 |
| CRF (centre) + counts | 1 | 7.87 | 11.31 | +0.753 |
| CRF + counts, no bond field | 1 | 7.90 | 11.49 | +0.745 |
| counts + morgan | 3 | 7.92 | 11.50 | +0.745 |
| CRF + counts, one weight | 1 | 7.93 | 11.40 | +0.749 |
| CRF + counts, no superposition | 1 | 8.46 | 12.02 | +0.721 |
| sides + counts | 1 | 8.62 | 12.17 | +0.714 |
| CRF | 1 | 8.89 | 12.77 | +0.685 |
| even blocks only | 1 | 9.29 | 13.61 | +0.642 |
| counts | 3 | 9.40 | 13.42 | +0.652 |
| odd block only | 1 | 11.46 | 15.78 | +0.519 |
| morgan | 1 | 11.61 | 15.72 | +0.523 |
| sides | 1 | 11.83 | 15.79 | +0.518 |
| not usable: linear in true dh | 3 | 16.50 | 20.16 | +0.215 |
| linear in dh predicted by CRF + counts | 3 | 16.82 | 20.63 | +0.178 |
| mean | 3 | 18.03 | 22.76 | -0.000 |
| reactant mean | 3 | 18.03 | 22.76 | -0.000 |

| Difference (first minus second) | Sets of folds | in q2 [95 % interval] | in MAE [95 % interval] | in q2 on each set of folds |
|---|---|---|---|---|
| CRF + counts minus counts + morgan | 3 | +0.006 [-0.001, +0.014] | -0.18 [-0.31, -0.07] | +0.019 [+0.011, +0.029]; -0.002 [-0.010, +0.007]; +0.001 [-0.007, +0.012] |
| CRF + counts, rows of unit length minus counts + morgan | 3 | +0.034 [+0.027, +0.042] | -0.75 [-0.87, -0.63] | +0.048 [+0.041, +0.057]; +0.027 [+0.019, +0.036]; +0.026 [+0.018, +0.035] |
| CRF + counts minus counts | 3 | +0.099 [+0.089, +0.109] | -1.66 [-1.80, -1.53] | +0.107 [+0.098, +0.120]; +0.098 [+0.089, +0.108]; +0.091 [+0.080, +0.103] |
| CRF + counts minus sides + counts | 1 | +0.044 [+0.037, +0.052] | -0.97 [-1.08, -0.86] |  |
| CRF + counts minus CRF + counts, one weight | 1 | +0.009 [+0.004, +0.014] | -0.27 [-0.35, -0.20] |  |
| CRF + counts minus CRF + counts, no superposition | 1 | +0.037 [+0.030, +0.043] | -0.80 [-0.89, -0.71] |  |
| CRF + counts minus CRF + counts, no bond field | 1 | +0.013 [+0.007, +0.019] | -0.24 [-0.33, -0.15] |  |
| CRF + counts minus CRF + counts, rows of unit length | 3 | -0.028 [-0.033, -0.023] | +0.57 [+0.49, +0.63] | -0.029 [-0.035, -0.023]; -0.030 [-0.035, -0.024]; -0.025 [-0.032, -0.018] |
| CRF + counts minus CRF (centre) + counts | 1 | +0.005 [-0.001, +0.011] | -0.21 [-0.30, -0.12] |  |
| CRF + counts + morgan minus counts + morgan | 1 | +0.035 [+0.030, +0.040] | -0.75 [-0.83, -0.67] |  |
| CRF minus sides | 1 | +0.167 [+0.153, +0.179] | -2.95 [-3.12, -2.76] |  |
| CRF minus morgan | 1 | +0.162 [+0.147, +0.177] | -2.72 [-2.94, -2.50] |  |

## Reaction energy, whole reactants held out (up to 3 sets of folds)

| Model | Sets of folds | MAE | RMSE | q2 |
|---|---|---|---|---|
| CRF + counts, rows of unit length | 3 | 6.28 | 9.60 | +0.900 |
| CRF + counts + morgan | 1 | 6.28 | 9.30 | +0.906 |
| CRF (centre) + counts | 1 | 6.87 | 10.40 | +0.883 |
| CRF + counts | 3 | 6.93 | 10.43 | +0.882 |
| CRF + counts, no bond field | 1 | 6.98 | 10.63 | +0.877 |
| CRF + counts, one weight | 1 | 7.03 | 10.67 | +0.876 |
| CRF + counts, no superposition | 1 | 7.08 | 10.80 | +0.873 |
| sides + counts | 1 | 7.17 | 10.91 | +0.871 |
| counts + morgan | 3 | 7.33 | 10.83 | +0.873 |
| CRF | 1 | 7.92 | 11.96 | +0.845 |
| odd block only | 1 | 9.67 | 13.71 | +0.796 |
| sides | 1 | 9.92 | 13.95 | +0.789 |
| morgan | 1 | 9.94 | 13.74 | +0.795 |
| even blocks only | 1 | 10.09 | 14.95 | +0.758 |
| counts | 3 | 10.20 | 14.63 | +0.768 |
| mean | 3 | 25.05 | 30.37 | -0.001 |
| reactant mean | 3 | 25.05 | 30.37 | -0.001 |

| Difference (first minus second) | Sets of folds | in q2 [95 % interval] | in MAE [95 % interval] | in q2 on each set of folds |
|---|---|---|---|---|
| CRF + counts minus counts + morgan | 3 | +0.009 [+0.002, +0.016] | -0.41 [-0.58, -0.24] | +0.009 [+0.002, +0.016]; +0.009 [+0.002, +0.016]; +0.010 [+0.002, +0.018] |
| CRF + counts, rows of unit length minus counts + morgan | 3 | +0.027 [+0.021, +0.034] | -1.06 [-1.23, -0.90] | +0.028 [+0.022, +0.034]; +0.025 [+0.017, +0.031]; +0.029 [+0.023, +0.037] |
| CRF + counts minus counts | 3 | +0.114 [+0.105, +0.123] | -3.27 [-3.44, -3.08] | +0.120 [+0.110, +0.132]; +0.113 [+0.105, +0.123]; +0.109 [+0.098, +0.120] |
| CRF + counts minus sides + counts | 1 | +0.011 [+0.005, +0.017] | -0.23 [-0.37, -0.10] |  |
| CRF + counts minus CRF + counts, one weight | 1 | +0.005 [+0.000, +0.010] | -0.09 [-0.20, +0.02] |  |
| CRF + counts minus CRF + counts, no superposition | 1 | +0.008 [+0.003, +0.014] | -0.14 [-0.24, -0.03] |  |
| CRF + counts minus CRF + counts, no bond field | 1 | +0.004 [+0.000, +0.008] | -0.04 [-0.13, +0.05] |  |
| CRF + counts minus CRF + counts, rows of unit length | 3 | -0.018 [-0.021, -0.014] | +0.65 [+0.56, +0.75] | -0.019 [-0.024, -0.014]; -0.016 [-0.019, -0.012]; -0.019 [-0.024, -0.015] |
| CRF + counts minus CRF (centre) + counts | 1 | -0.001 [-0.004, +0.003] | +0.07 [-0.01, +0.16] |  |
| CRF + counts + morgan minus counts + morgan | 1 | +0.033 [+0.027, +0.039] | -1.07 [-1.21, -0.92] |  |
| CRF minus sides | 1 | +0.056 [+0.046, +0.065] | -2.00 [-2.21, -1.81] |  |
| CRF minus morgan | 1 | +0.050 [+0.037, +0.063] | -2.02 [-2.28, -1.76] |  |

### The gate models again with wider grids, whole reactants held out, the same folds

Width factor 0.03/0.1/0.3/1/3/10/30; ridge 1e-06/1e-05/0.0001/0.001/0.01/0.1/1; block weights 0.01/0.1/1/10/100; the search goes through the blocks 2 times.

| Model, activation energy | MAE | MAE with the first grids | RMSE | q2 |
|---|---|---|---|---|
| CRF + counts, rows of unit length | 6.95 | 7.18 | 10.48 | +0.788 |
| CRF + counts + morgan | 7.52 | 7.40 | 10.92 | +0.769 |
| CRF + counts | 7.79 | 7.74 | 11.36 | +0.751 |
| counts + morgan | 8.14 | 7.92 | 11.63 | +0.739 |
| counts | 9.48 | 9.40 | 13.45 | +0.650 |

| Difference (first minus second) | in q2 [95 % interval] | in MAE [95 % interval] |
|---|---|---|
| CRF + counts minus counts + morgan | +0.012 [+0.004, +0.021] | -0.36 [-0.50, -0.22] |
| CRF + counts, rows of unit length minus counts + morgan | +0.049 [+0.040, +0.058] | -1.19 [-1.34, -1.05] |
| CRF + counts + morgan minus counts + morgan | +0.031 [+0.025, +0.037] | -0.62 [-0.72, -0.53] |
| CRF + counts minus counts | +0.100 [+0.090, +0.111] | -1.69 [-1.84, -1.53] |

| Model, reaction energy | MAE | MAE with the first grids | RMSE | q2 |
|---|---|---|---|---|
| CRF + counts, rows of unit length | 6.31 | 6.28 | 9.59 | +0.900 |
| CRF + counts + morgan | 6.36 | 6.28 | 9.45 | +0.903 |
| CRF + counts | 6.91 | 6.93 | 10.43 | +0.882 |
| counts + morgan | 7.50 | 7.33 | 11.02 | +0.868 |
| counts | 10.40 | 10.20 | 14.82 | +0.762 |

| Difference (first minus second) | in q2 [95 % interval] | in MAE [95 % interval] |
|---|---|---|
| CRF + counts minus counts + morgan | +0.014 [+0.007, +0.021] | -0.59 [-0.79, -0.42] |
| CRF + counts, rows of unit length minus counts + morgan | +0.032 [+0.026, +0.039] | -1.19 [-1.36, -1.00] |
| CRF + counts + morgan minus counts + morgan | +0.035 [+0.029, +0.041] | -1.14 [-1.28, -1.01] |
| CRF + counts minus counts | +0.120 [+0.109, +0.132] | -3.49 [-3.70, -3.29] |

| Model | Width factor | Ridge | Block weights |
|---|---|---|---|
| CRF + counts | 0.3/1 | 0.01/0.1 | odd 0.01/0.1/1; conv_h 0.01/0.1; dyn_h 0.1/1; bond_h 0.01/0.1/1; counts 1/10 |
| counts + morgan | 1 | 0.01/0.1 | counts 10; mdiff 1; msum 0.1/1 |
| CRF + counts, rows of unit length | 1 | 0.01/0.1 | odd_n 0.1/1; conv_n 0.01/0.1; dyn_n 1; bond_n 0.1/1; counts 1 |
| counts | 1/3 | 0.1 | counts 1 |
| CRF + counts + morgan | 0.3/1 | 0.001/0.01/0.1 | odd 0.01/0.1/1; conv_h 0.01/0.1/1; dyn_h 1; bond_h 0.1/1/10; counts 10; mdiff 1; msum 0.01/0.1/1 |

### Correlation of whole kernels, whole reactants held out

Centred alignment on 2,000 reactions, with the settings each model chose for the activation energy in the first fold.

| Two models | Correlation |
|---|---|
| CRF + counts | counts + morgan | 0.670 |
| CRF + counts | sides | 0.523 |
| CRF + counts | CRF | 0.908 |
| CRF + counts | counts | 0.667 |
| CRF + counts | morgan | 0.297 |
| counts + morgan | sides | 0.558 |
| counts + morgan | CRF | 0.410 |
| counts + morgan | counts | 0.992 |
| counts + morgan | morgan | 0.382 |
| sides | CRF | 0.350 |
| sides | counts | 0.533 |
| sides | morgan | 0.302 |
| CRF | counts | 0.405 |
| CRF | morgan | 0.223 |
| counts | morgan | 0.317 |

### Settings the search chose, activation energy, whole reactants held out

| Model | Width factor | Ridge | Block weights |
|---|---|---|---|
| counts | 1/3 | 0.1 | counts 1 |
| CRF + counts | 0.3/1 | 0.01/0.1 | odd 0.1/1; conv_h 0.1; dyn_h 0.1/1; bond_h 0.1/1/10; counts 1/10 |
| morgan | 1/3 | 1e-05/0.01/0.1 | mdiff 1/10; msum 0.1/1 |
| counts + morgan | 1/3 | 1e-05/0.001/0.01/0.1 | counts 10; mdiff 1; msum 0.1/1 |
| sides | 0.1/1 | 0.01/0.1 | odd 1; even0 1 |
| sides + counts | 1 | 0.01/0.1 | odd 0.1; even0 0.1; counts 1/10 |
| CRF | 0.3/1 | 0.01/0.1 | odd 0.1/1/10; conv_h 0.1/1; dyn_h 1/10; bond_h 1/10 |
| CRF (centre) + counts | 0.3/1 | 0.01/0.1 | odd 0.1/1; conv_c 0.1/1; dyn_c 0.1/1; bond_c 0.1/1; counts 1/10 |
| CRF + counts, one weight | 0.3/1 | 0.01/0.1 | odd 0.1/1; conv_h 0.1/1; dyn_h 0.1/1; bond_h 0.1/1; counts 1 |
| CRF + counts, no superposition | 0.3/1 | 0.01/0.1 | odd 0.1/1; even0 0.1; counts 1/10 |
| CRF + counts, no bond field | 0.3/1 | 0.01/0.1 | odd 0.1/1; conv_h 0.1/1; dyn_h 0.1/1; counts 1/10 |
| CRF + counts, rows of unit length | 0.3/1 | 0.01/0.1 | odd_n 0.1/1; conv_n 0.1; dyn_n 0.1/1; bond_n 0.1/1; counts 1/10 |
| odd block only | 0.3/1/3 | 0.01/0.1 | odd 1 |
| even blocks only | 0.3/1/3 | 0.01/0.1 | conv_h 0.1/1; dyn_h 1/10; bond_h 1/10 |
| CRF + counts + morgan | 1 | 0.001/0.01/0.1 | odd 0.1/1; conv_h 0.1/1; dyn_h 0.1/1; bond_h 1; counts 10; mdiff 1; msum 0.1/1 |
| not usable: CRF + counts + true dh | 0.3/1 | 0.01/0.1 | odd 0.1; conv_h 0.1/1; dyn_h 0.1/1; bond_h 0.1/1/10; counts 1/10; dh 1 |
| not usable: counts + morgan + true dh | 1 | 0.01/0.1 | counts 10; mdiff 0.1/1; msum 0.1/1; dh 1/10 |

Test reactions whose reactant is also in the training part, whole reactants held out: [0, 0, 0] (per repeat).

## Activation energy, random folds (up to 1 set of folds)

| Model | Sets of folds | MAE | RMSE | q2 |
|---|---|---|---|---|
| not usable: CRF + counts + true dh | 1 | 6.76 | 10.08 | +0.804 |
| CRF + counts, rows of unit length | 1 | 6.82 | 10.41 | +0.791 |
| not usable: counts + morgan + true dh | 1 | 7.00 | 10.47 | +0.788 |
| CRF + counts + morgan | 1 | 7.36 | 10.77 | +0.776 |
| CRF (centre) + counts | 1 | 7.43 | 10.99 | +0.767 |
| counts + morgan | 1 | 7.65 | 11.23 | +0.757 |
| CRF + counts, one weight | 1 | 7.85 | 11.26 | +0.755 |
| CRF + counts | 1 | 7.87 | 11.31 | +0.753 |
| CRF + counts, no bond field | 1 | 8.00 | 11.60 | +0.740 |
| CRF + counts, no superposition | 1 | 8.13 | 11.84 | +0.729 |
| CRF | 1 | 8.37 | 12.47 | +0.700 |
| sides + counts | 1 | 8.45 | 12.02 | +0.721 |
| even blocks only | 1 | 9.37 | 13.60 | +0.643 |
| counts | 1 | 9.65 | 13.55 | +0.645 |
| sides | 1 | 11.16 | 15.27 | +0.549 |
| odd block only | 1 | 11.21 | 15.53 | +0.534 |
| morgan | 1 | 11.22 | 15.30 | +0.548 |
| not usable: linear in true dh | 1 | 16.50 | 20.16 | +0.215 |
| linear in dh predicted by CRF + counts | 1 | 16.75 | 20.55 | +0.185 |
| reactant mean | 1 | 18.01 | 22.84 | -0.008 |
| mean | 1 | 18.03 | 22.76 | -0.000 |

| Difference (first minus second) | Sets of folds | in q2 [95 % interval] | in MAE [95 % interval] | in q2 on each set of folds |
|---|---|---|---|---|
| CRF + counts minus counts + morgan | 1 | -0.004 [-0.012, +0.006] | +0.22 [+0.09, +0.34] |  |
| CRF + counts, rows of unit length minus counts + morgan | 1 | +0.034 [+0.026, +0.043] | -0.83 [-0.97, -0.69] |  |
| CRF + counts minus counts | 1 | +0.108 [+0.097, +0.120] | -1.78 [-1.93, -1.64] |  |
| CRF + counts minus sides + counts | 1 | +0.032 [+0.025, +0.039] | -0.58 [-0.69, -0.48] |  |
| CRF + counts minus CRF + counts, one weight | 1 | -0.002 [-0.005, +0.002] | +0.02 [-0.03, +0.07] |  |
| CRF + counts minus CRF + counts, no superposition | 1 | +0.024 [+0.016, +0.032] | -0.25 [-0.37, -0.15] |  |
| CRF + counts minus CRF + counts, no bond field | 1 | +0.013 [+0.006, +0.019] | -0.13 [-0.22, -0.04] |  |
| CRF + counts minus CRF + counts, rows of unit length | 1 | -0.038 [-0.044, -0.031] | +1.05 [+0.95, +1.15] |  |
| CRF + counts minus CRF (centre) + counts | 1 | -0.014 [-0.021, -0.006] | +0.44 [+0.33, +0.54] |  |
| CRF + counts + morgan minus counts + morgan | 1 | +0.019 [+0.014, +0.025] | -0.29 [-0.37, -0.22] |  |
| CRF minus sides | 1 | +0.150 [+0.138, +0.164] | -2.79 [-2.95, -2.63] |  |
| CRF minus morgan | 1 | +0.152 [+0.137, +0.166] | -2.85 [-3.06, -2.64] |  |

## Reaction energy, random folds (up to 1 set of folds)

| Model | Sets of folds | MAE | RMSE | q2 |
|---|---|---|---|---|
| CRF + counts + morgan | 1 | 5.67 | 8.65 | +0.919 |
| CRF + counts, rows of unit length | 1 | 6.09 | 9.23 | +0.908 |
| CRF + counts | 1 | 6.51 | 9.91 | +0.893 |
| CRF (centre) + counts | 1 | 6.63 | 9.87 | +0.894 |
| CRF + counts, no superposition | 1 | 6.64 | 10.01 | +0.891 |
| CRF + counts, no bond field | 1 | 6.67 | 10.05 | +0.891 |
| sides + counts | 1 | 6.83 | 10.23 | +0.886 |
| counts + morgan | 1 | 7.01 | 10.26 | +0.886 |
| CRF + counts, one weight | 1 | 7.02 | 10.53 | +0.880 |
| CRF | 1 | 7.39 | 11.25 | +0.863 |
| odd block only | 1 | 8.88 | 12.68 | +0.826 |
| morgan | 1 | 9.11 | 12.80 | +0.822 |
| sides | 1 | 9.14 | 12.99 | +0.817 |
| even blocks only | 1 | 9.94 | 14.80 | +0.762 |
| counts | 1 | 10.44 | 14.56 | +0.770 |
| mean | 1 | 25.04 | 30.37 | -0.000 |
| reactant mean | 1 | 25.31 | 30.95 | -0.039 |

| Difference (first minus second) | Sets of folds | in q2 [95 % interval] | in MAE [95 % interval] | in q2 on each set of folds |
|---|---|---|---|---|
| CRF + counts minus counts + morgan | 1 | +0.008 [+0.001, +0.014] | -0.50 [-0.65, -0.35] |  |
| CRF + counts, rows of unit length minus counts + morgan | 1 | +0.022 [+0.017, +0.027] | -0.92 [-1.05, -0.76] |  |
| CRF + counts minus counts | 1 | +0.123 [+0.113, +0.133] | -3.93 [-4.12, -3.73] |  |
| CRF + counts minus sides + counts | 1 | +0.007 [+0.004, +0.010] | -0.32 [-0.41, -0.24] |  |
| CRF + counts minus CRF + counts, one weight | 1 | +0.014 [+0.010, +0.018] | -0.51 [-0.59, -0.42] |  |
| CRF + counts minus CRF + counts, no superposition | 1 | +0.002 [-0.000, +0.004] | -0.13 [-0.17, -0.08] |  |
| CRF + counts minus CRF + counts, no bond field | 1 | +0.003 [+0.001, +0.005] | -0.15 [-0.20, -0.11] |  |
| CRF + counts minus CRF + counts, rows of unit length | 1 | -0.014 [-0.017, -0.010] | +0.42 [+0.32, +0.52] |  |
| CRF + counts minus CRF (centre) + counts | 1 | -0.001 [-0.004, +0.002] | -0.12 [-0.19, -0.06] |  |
| CRF + counts + morgan minus counts + morgan | 1 | +0.033 [+0.028, +0.038] | -1.33 [-1.45, -1.21] |  |
| CRF minus sides | 1 | +0.046 [+0.038, +0.053] | -1.74 [-1.90, -1.60] |  |
| CRF minus morgan | 1 | +0.040 [+0.031, +0.050] | -1.71 [-1.91, -1.52] |  |

### Correlation of whole kernels, random folds

Centred alignment on 2,000 reactions, with the settings each model chose for the activation energy in the first fold.

| Two models | Correlation |
|---|---|
| CRF + counts | counts + morgan | 0.941 |
| CRF + counts | sides | 0.592 |
| CRF + counts | CRF | 0.622 |
| CRF + counts | counts | 0.933 |
| CRF + counts | morgan | 0.319 |
| counts + morgan | sides | 0.558 |
| counts + morgan | CRF | 0.484 |
| counts + morgan | counts | 0.992 |
| counts + morgan | morgan | 0.382 |
| sides | CRF | 0.644 |
| sides | counts | 0.533 |
| sides | morgan | 0.302 |
| CRF | counts | 0.467 |
| CRF | morgan | 0.280 |
| counts | morgan | 0.317 |

### Settings the search chose, activation energy, random folds

| Model | Width factor | Ridge | Block weights |
|---|---|---|---|
| counts | 1/3 | 0.1 | counts 1 |
| morgan | 3 | 1e-05 | mdiff 10; msum 1 |
| counts + morgan | 1/3 | 1e-05/0.01 | counts 10; mdiff 1; msum 1 |
| sides | 1 | 0.01/0.1 | odd 1; even0 1 |
| sides + counts | 1 | 0.01/0.1 | odd 0.1; even0 0.1; counts 1 |
| CRF | 1/3 | 0.01/0.1 | odd 1/10; conv_h 0.1/1; dyn_h 0.1/1/10; bond_h 1/10 |
| CRF + counts | 0.3/1 | 0.01 | odd 0.1/1; conv_h 0.1/1; dyn_h 0.1/1; bond_h 0.1/1; counts 1/10 |
| CRF (centre) + counts | 0.3/1 | 0.01 | odd 1; conv_c 0.1; dyn_c 0.1/1; bond_c 0.1/1; counts 1/10 |
| CRF + counts, one weight | 0.3/1 | 0.01/0.1 | odd 0.1; conv_h 0.1; dyn_h 0.1; bond_h 0.1; counts 1 |
| CRF + counts, no superposition | 1 | 0.01/0.1 | odd 1; even0 0.1; counts 1/10 |
| CRF + counts, no bond field | 0.3/1 | 0.01/0.1 | odd 1; conv_h 0.1; dyn_h 0.1/1; counts 1/10 |
| CRF + counts, rows of unit length | 1 | 0.01 | odd_n 1; conv_n 0.1; dyn_n 0.1/1; bond_n 0.1/1; counts 1 |
| odd block only | 1/3 | 0.01/0.1 | odd 1 |
| even blocks only | 1/3 | 0.1 | conv_h 0.1; dyn_h 1; bond_h 1 |
| CRF + counts + morgan | 1 | 1e-05/0.01/0.1 | odd 0.1/1; conv_h 0.1; dyn_h 0.1/1/10; bond_h 1/10; counts 10; mdiff 1; msum 1 |
| not usable: CRF + counts + true dh | 1 | 0.01/0.1 | odd 0.1; conv_h 0.1/1; dyn_h 0.1/1; bond_h 0.1/1; counts 1/10; dh 0.1/1/10 |
| not usable: counts + morgan + true dh | 1/3 | 1e-05/0.01/0.1 | counts 10; mdiff 1; msum 1; dh 1/10 |

Test reactions whose reactant is also in the training part, random folds: [11919] (per repeat).

## Activation energy, whole reaction types held out (up to 1 set of folds)

| Model | Sets of folds | MAE | RMSE | q2 |
|---|---|---|---|---|
| not usable: CRF + counts + true dh | 1 | 11.04 | 14.05 | +0.619 |
| not usable: counts + morgan + true dh | 1 | 11.63 | 14.69 | +0.583 |
| CRF + counts + morgan | 1 | 12.45 | 15.94 | +0.509 |
| CRF + counts, rows of unit length | 1 | 12.71 | 16.47 | +0.476 |
| CRF + counts | 1 | 13.07 | 16.89 | +0.449 |
| CRF + counts, one weight | 1 | 13.08 | 16.72 | +0.460 |
| CRF (centre) + counts | 1 | 13.09 | 16.60 | +0.468 |
| counts + morgan | 1 | 13.12 | 16.48 | +0.475 |
| CRF + counts, no bond field | 1 | 13.53 | 17.10 | +0.435 |
| sides + counts | 1 | 13.67 | 17.26 | +0.425 |
| CRF + counts, no superposition | 1 | 13.68 | 17.26 | +0.425 |
| counts | 1 | 14.58 | 18.46 | +0.342 |
| CRF | 1 | 14.65 | 18.75 | +0.321 |
| even blocks only | 1 | 15.31 | 19.49 | +0.266 |
| morgan | 1 | 15.61 | 19.64 | +0.255 |
| sides | 1 | 16.64 | 20.86 | +0.159 |
| not usable: linear in true dh | 1 | 16.68 | 20.34 | +0.201 |
| odd block only | 1 | 16.69 | 20.99 | +0.149 |
| linear in dh predicted by CRF + counts | 1 | 17.13 | 21.00 | +0.148 |
| mean | 1 | 18.16 | 22.90 | -0.013 |
| reactant mean | 1 | 18.52 | 23.47 | -0.064 |

| Difference (first minus second) | Sets of folds | in q2 [95 % interval] | in MAE [95 % interval] | in q2 on each set of folds |
|---|---|---|---|---|
| CRF + counts minus counts + morgan | 1 | -0.026 [-0.038, -0.013] | -0.04 [-0.22, +0.12] |  |
| CRF + counts, rows of unit length minus counts + morgan | 1 | +0.001 [-0.012, +0.014] | -0.41 [-0.59, -0.24] |  |
| CRF + counts minus counts | 1 | +0.107 [+0.092, +0.124] | -1.51 [-1.70, -1.35] |  |
| CRF + counts minus sides + counts | 1 | +0.024 [+0.013, +0.036] | -0.60 [-0.73, -0.46] |  |
| CRF + counts minus CRF + counts, one weight | 1 | -0.011 [-0.018, -0.003] | -0.01 [-0.11, +0.07] |  |
| CRF + counts minus CRF + counts, no superposition | 1 | +0.025 [+0.013, +0.036] | -0.61 [-0.75, -0.47] |  |
| CRF + counts minus CRF + counts, no bond field | 1 | +0.014 [+0.004, +0.025] | -0.45 [-0.59, -0.32] |  |
| CRF + counts minus CRF + counts, rows of unit length | 1 | -0.027 [-0.036, -0.016] | +0.37 [+0.23, +0.48] |  |
| CRF + counts minus CRF (centre) + counts | 1 | -0.019 [-0.030, -0.006] | -0.02 [-0.16, +0.12] |  |
| CRF + counts + morgan minus counts + morgan | 1 | +0.034 [+0.026, +0.042] | -0.67 [-0.78, -0.56] |  |
| CRF minus sides | 1 | +0.161 [+0.143, +0.180] | -2.00 [-2.22, -1.78] |  |
| CRF minus morgan | 1 | +0.065 [+0.047, +0.084] | -0.96 [-1.19, -0.73] |  |

## Reaction energy, whole reaction types held out (up to 1 set of folds)

| Model | Sets of folds | MAE | RMSE | q2 |
|---|---|---|---|---|
| CRF + counts + morgan | 1 | 7.98 | 11.37 | +0.860 |
| CRF (centre) + counts | 1 | 9.14 | 12.73 | +0.824 |
| sides + counts | 1 | 9.19 | 12.99 | +0.817 |
| CRF + counts, no bond field | 1 | 9.24 | 12.92 | +0.819 |
| CRF + counts | 1 | 9.24 | 12.88 | +0.820 |
| CRF + counts, rows of unit length | 1 | 9.32 | 12.90 | +0.819 |
| CRF + counts, no superposition | 1 | 9.41 | 13.12 | +0.813 |
| counts + morgan | 1 | 9.51 | 13.12 | +0.813 |
| CRF + counts, one weight | 1 | 9.94 | 13.55 | +0.801 |
| CRF | 1 | 10.89 | 15.41 | +0.742 |
| morgan | 1 | 11.08 | 15.02 | +0.755 |
| sides | 1 | 12.53 | 16.92 | +0.689 |
| odd block only | 1 | 13.65 | 18.12 | +0.644 |
| even blocks only | 1 | 14.40 | 19.32 | +0.595 |
| counts | 1 | 14.81 | 19.59 | +0.584 |
| mean | 1 | 25.28 | 30.62 | -0.017 |
| reactant mean | 1 | 26.09 | 31.86 | -0.101 |

| Difference (first minus second) | Sets of folds | in q2 [95 % interval] | in MAE [95 % interval] | in q2 on each set of folds |
|---|---|---|---|---|
| CRF + counts minus counts + morgan | 1 | +0.007 [-0.002, +0.016] | -0.27 [-0.45, -0.08] |  |
| CRF + counts, rows of unit length minus counts + morgan | 1 | +0.006 [-0.002, +0.015] | -0.19 [-0.38, -0.00] |  |
| CRF + counts minus counts | 1 | +0.236 [+0.222, +0.250] | -5.56 [-5.80, -5.34] |  |
| CRF + counts minus sides + counts | 1 | +0.003 [-0.001, +0.008] | +0.05 [-0.05, +0.15] |  |
| CRF + counts minus CRF + counts, one weight | 1 | +0.019 [+0.014, +0.024] | -0.69 [-0.79, -0.60] |  |
| CRF + counts minus CRF + counts, no superposition | 1 | +0.007 [+0.004, +0.010] | -0.16 [-0.24, -0.09] |  |
| CRF + counts minus CRF + counts, no bond field | 1 | +0.001 [-0.002, +0.005] | +0.01 [-0.07, +0.08] |  |
| CRF + counts minus CRF + counts, rows of unit length | 1 | +0.001 [-0.006, +0.008] | -0.08 [-0.23, +0.07] |  |
| CRF + counts minus CRF (centre) + counts | 1 | -0.004 [-0.008, -0.001] | +0.10 [+0.01, +0.19] |  |
| CRF + counts + morgan minus counts + morgan | 1 | +0.047 [+0.039, +0.054] | -1.53 [-1.69, -1.37] |  |
| CRF minus sides | 1 | +0.053 [+0.039, +0.068] | -1.64 [-1.89, -1.41] |  |
| CRF minus morgan | 1 | -0.013 [-0.024, -0.000] | -0.19 [-0.46, +0.03] |  |

### Correlation of whole kernels, whole reaction types held out

Centred alignment on 2,000 reactions, with the settings each model chose for the activation energy in the first fold.

| Two models | Correlation |
|---|---|
| CRF + counts | counts + morgan | 0.968 |
| CRF + counts | sides | 0.457 |
| CRF + counts | CRF | 0.524 |
| CRF + counts | counts | 0.968 |
| CRF + counts | morgan | 0.329 |
| counts + morgan | sides | 0.447 |
| counts + morgan | CRF | 0.317 |
| counts + morgan | counts | 0.996 |
| counts + morgan | morgan | 0.358 |
| sides | CRF | 0.193 |
| sides | counts | 0.425 |
| sides | morgan | 0.369 |
| CRF | counts | 0.310 |
| CRF | morgan | 0.181 |
| counts | morgan | 0.295 |

### Settings the search chose, activation energy, whole reaction types held out

| Model | Width factor | Ridge | Block weights |
|---|---|---|---|
| counts | 0.1/0.3 | 0.01/0.1 | counts 1 |
| morgan | 1/3 | 0.1 | mdiff 1/10; msum 0.1/1 |
| counts + morgan | 0.1/0.3/1 | 1e-05/0.01/0.1 | counts 1/10; mdiff 1; msum 0.1/1 |
| sides | 0.1/1/3 | 0.1 | odd 1; even0 1 |
| sides + counts | 0.1/0.3 | 0.01/0.1 | odd 0.1; even0 0.1; counts 1 |
| CRF | 0.1/1/3 | 0.01/0.1 | odd 0.1/1/10; conv_h 0.1/1; dyn_h 0.1/1; bond_h 1/10 |
| CRF + counts | 0.1/0.3/1 | 0.01/0.1 | odd 0.1/1; conv_h 0.1/1; dyn_h 0.1; bond_h 0.1/1/10; counts 1/10 |
| CRF (centre) + counts | 0.1/0.3/1 | 0.01 | odd 0.1/1; conv_c 0.1/1/10; dyn_c 0.1/1; bond_c 0.1/1/10; counts 1/10 |
| CRF + counts, one weight | 0.1/0.3/1 | 0.01/0.1 | odd 0.1/1; conv_h 0.1/1; dyn_h 0.1/1; bond_h 0.1/1; counts 1 |
| CRF + counts, no superposition | 0.1/0.3 | 0.01/0.1 | odd 0.1/1; even0 0.1/1; counts 1/10 |
| CRF + counts, no bond field | 0.1/0.3 | 0.01/0.1 | odd 0.1/1; conv_h 0.1; dyn_h 0.1/1; counts 1/10 |
| CRF + counts, rows of unit length | 0.1/0.3/1 | 0.001/0.01/0.1 | odd_n 0.1/1; conv_n 0.1/1/10; dyn_n 0.1; bond_n 0.1/1; counts 1/10 |
| odd block only | 0.1/1/10 | 0.1 | odd 1 |
| even blocks only | 0.1/1/3 | 0.01/0.1 | conv_h 0.1/1/10; dyn_h 0.1/1; bond_h 1 |
| CRF + counts + morgan | 0.1/0.3/1 | 1e-05/0.01/0.1 | odd 0.1; conv_h 0.1; dyn_h 0.1; bond_h 0.1/1/10; counts 1/10; mdiff 1; msum 0.1/1 |
| not usable: CRF + counts + true dh | 0.1/0.3/1 | 0.01/0.1 | odd 0.1; conv_h 0.1; dyn_h 0.1; bond_h 0.1/1/10; counts 0.1/1/10; dh 0.1/1 |
| not usable: counts + morgan + true dh | 0.3/1 | 0.01/0.1 | counts 1/10; mdiff 0.1/1; msum 0.1/1; dh 1/10 |

Test reactions whose reactant is also in the training part, whole reaction types held out: [11916] (per repeat).

