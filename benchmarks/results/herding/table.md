# Kernel herding against the usual ways of choosing a subset

## Which molecules to measure first

8 Sutherland sets, 10 random splits into a pool (70 %) and a test set (30 %). From the pool, m molecules are chosen without looking at the activity; a model fitted on them alone predicts the test set. `random` is the mean of 5 draws per split. The kernel the subsets are chosen on is the joint kernel at alpha = 0.3; `herding-gauss` uses a Gaussian of the field distance instead. `fingerprints` (max-min on Morgan fingerprints, Tanimoto) and `size` (spread evenly over the number of heavy atoms) need no field. MMD^2 is between the chosen molecules and the pool, in the joint kernel.

| m | rule | R2pred on the test set | RMSE on the test set | MMD^2 to the pool | MMD^2 to the test set | RMSE minus random, mean over sets [95 % interval] | sets where better than random |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | `herding` | +0.118 | 1.326 | 0.0262 | 0.0795 | -0.013 [-0.075, +0.051] | 5 of 8 |
| 10 | `herding-gauss` | +0.060 | 1.343 | 0.0208 | 0.0749 | +0.004 [-0.044, +0.061] | 5 of 8 |
| 10 | `maxmin` | +0.161 | 1.394 | 0.2060 | 0.2526 | +0.056 [-0.010, +0.137] | 4 of 8 |
| 10 | `kmedoids` | +0.094 | 1.325 | 0.0598 | 0.1154 | -0.014 [-0.086, +0.057] | 4 of 8 |
| 10 | `pivoted_cholesky` | +0.165 | 1.362 | 0.1785 | 0.2261 | +0.023 [-0.070, +0.114] | 3 of 8 |
| 10 | `fingerprints` | +0.130 | 1.304 | 0.0874 | 0.1324 | -0.035 [-0.076, +0.010] | 5 of 8 |
| 10 | `size` | +0.051 | 1.337 | 0.0838 | 0.1381 | -0.002 [-0.060, +0.049] | 3 of 8 |
| 10 | `random` | +0.079 | 1.339 | 0.0994 | 0.1503 | | |
| 20 | `herding` | +0.142 | 1.253 | 0.0076 | 0.0580 | +0.026 [+0.003, +0.051] | 3 of 8 |
| 20 | `herding-gauss` | +0.190 | 1.213 | 0.0068 | 0.0588 | -0.013 [-0.059, +0.035] | 5 of 8 |
| 20 | `maxmin` | +0.252 | 1.234 | 0.0927 | 0.1402 | +0.008 [-0.051, +0.068] | 4 of 8 |
| 20 | `kmedoids` | +0.234 | 1.188 | 0.0441 | 0.0939 | -0.039 [-0.096, +0.016] | 4 of 8 |
| 20 | `pivoted_cholesky` | +0.225 | 1.242 | 0.0856 | 0.1333 | +0.015 [-0.040, +0.068] | 3 of 8 |
| 20 | `fingerprints` | +0.213 | 1.218 | 0.0458 | 0.0874 | -0.009 [-0.060, +0.040] | 4 of 8 |
| 20 | `size` | +0.198 | 1.221 | 0.0328 | 0.0804 | -0.006 [-0.023, +0.011] | 4 of 8 |
| 20 | `random` | +0.186 | 1.227 | 0.0405 | 0.0929 | | |
| 40 | `herding` | +0.381 | 1.074 | 0.0020 | 0.0521 | -0.039 [-0.059, -0.022] | 7 of 8 |
| 40 | `herding-gauss` | +0.381 | 1.072 | 0.0018 | 0.0522 | -0.041 [-0.075, -0.004] | 6 of 8 |
| 40 | `maxmin` | +0.390 | 1.078 | 0.0285 | 0.0757 | -0.036 [-0.073, -0.003] | 7 of 8 |
| 40 | `kmedoids` | +0.395 | 1.057 | 0.0198 | 0.0679 | -0.057 [-0.085, -0.028] | 7 of 8 |
| 40 | `pivoted_cholesky` | +0.367 | 1.093 | 0.0290 | 0.0746 | -0.020 [-0.060, +0.022] | 5 of 8 |
| 40 | `fingerprints` | +0.402 | 1.052 | 0.0188 | 0.0619 | -0.061 [-0.104, -0.019] | 6 of 8 |
| 40 | `size` | +0.319 | 1.115 | 0.0101 | 0.0623 | +0.002 [-0.054, +0.067] | 3 of 8 |
| 40 | `random` | +0.328 | 1.113 | 0.0144 | 0.0663 | | |

The model on the whole pool: R2pred +0.482, RMSE 0.978. The activity's standard deviation averages 1.36.

### R2pred by set, m = 20

| set | n | `herding` | `herding-gauss` | `maxmin` | `kmedoids` | `pivoted_cholesky` | `fingerprints` | `size` | `random` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ACE | 114 | +0.35 | +0.43 | +0.35 | +0.44 | +0.41 | +0.29 | +0.32 | +0.35 |
| AChE | 111 | +0.18 | -0.15 | +0.19 | +0.19 | +0.08 | +0.31 | +0.17 | +0.10 |
| BZR | 147 | -0.10 | -0.04 | -0.07 | -0.04 | +0.01 | -0.05 | +0.09 | -0.04 |
| COX2 | 282 | +0.09 | +0.26 | +0.25 | -0.02 | +0.28 | +0.24 | +0.09 | +0.10 |
| DHFR | 361 | +0.12 | +0.22 | +0.14 | +0.31 | +0.13 | +0.21 | +0.14 | +0.18 |
| GPB | 66 | +0.31 | +0.44 | +0.45 | +0.45 | +0.33 | +0.32 | +0.45 | +0.45 |
| THER | 76 | +0.10 | +0.10 | +0.34 | +0.17 | +0.32 | +0.20 | +0.17 | +0.18 |
| THR | 88 | +0.08 | +0.27 | +0.37 | +0.38 | +0.24 | +0.17 | +0.15 | +0.17 |

## Which molecules to use as Nystroem landmarks

The kernel of each whole set, at the attenuation exact cross-validation chooses, is replaced by its Nystroem approximation on m landmark molecules, chosen on that kernel. Error is ||K - K~|| / ||K|| (Frobenius); q2 is 10-fold kernel ridge regression on K~, with the best ridge of the grid, as for the exact kernel. `random` is the mean of ten draws. Exact q2 averages +0.570.

| landmarks | rule | relative error of the kernel | q2 | q2 minus exact |
|---:|---|---:|---:|---:|
| 5 | `herding` | 0.084 | +0.267 | -0.303 |
| 5 | `herding-gauss` | 0.070 | +0.260 | -0.310 |
| 5 | `maxmin` | 0.116 | +0.196 | -0.375 |
| 5 | `kmedoids` | 0.068 | +0.263 | -0.308 |
| 5 | `pivoted_cholesky` | 0.158 | +0.134 | -0.436 |
| 5 | `random` | 0.129 | +0.221 | -0.350 |
| 10 | `herding` | 0.048 | +0.361 | -0.210 |
| 10 | `herding-gauss` | 0.044 | +0.398 | -0.173 |
| 10 | `maxmin` | 0.077 | +0.291 | -0.279 |
| 10 | `kmedoids` | 0.040 | +0.405 | -0.165 |
| 10 | `pivoted_cholesky` | 0.091 | +0.271 | -0.299 |
| 10 | `random` | 0.074 | +0.353 | -0.217 |
| 20 | `herding` | 0.026 | +0.457 | -0.114 |
| 20 | `herding-gauss` | 0.025 | +0.499 | -0.072 |
| 20 | `maxmin` | 0.036 | +0.419 | -0.151 |
| 20 | `kmedoids` | 0.020 | +0.465 | -0.105 |
| 20 | `pivoted_cholesky` | 0.039 | +0.427 | -0.144 |
| 20 | `random` | 0.040 | +0.451 | -0.120 |
| 40 | `herding` | 0.012 | +0.537 | -0.033 |
| 40 | `herding-gauss` | 0.012 | +0.534 | -0.037 |
| 40 | `maxmin` | 0.012 | +0.509 | -0.062 |
| 40 | `kmedoids` | 0.009 | +0.529 | -0.041 |
| 40 | `pivoted_cholesky` | 0.013 | +0.498 | -0.072 |
| 40 | `random` | 0.019 | +0.520 | -0.050 |

## Which solvation shells to keep

The 62 electrolytes of item BT5, each the weighted mean of 15 or 16 shells. Distance is ||mean field of the kept shells - mean field of all shells|| / ||mean field of all shells||, in the framed field kernel at alpha = 0.5, averaged over the electrolytes (random: 50 draws each).

| shells kept | the heaviest | herding, equal weights | herding, fitted weights | greedy, equal weights | greedy, fitted weights | random |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.407 | 0.516 | 0.516 | 0.398 | 0.398 | 0.615 |
| 2 | 0.272 | 0.331 | 0.292 | 0.257 | 0.242 | 0.476 |
| 4 | 0.165 | 0.200 | 0.160 | 0.161 | 0.139 | 0.316 |
| 8 | 0.065 | 0.118 | 0.071 | 0.099 | 0.062 | 0.192 |

The alignment-free model of BT5 on the kept shells, nested cross-validation as in `benchmarks/run_ensembles.py`. Mean absolute error; the interval is the paired interval over the 62 electrolytes of the gain over the heaviest shells of the same number, positive when the variant has the smaller error. The heaviest shells are the stored records of item MS5.

### log10 conductivity at 298 K, random folds

| shells | mean absolute error | gain over the same number of heaviest | gain over all 16 |
|---|---:|---:|---:|
| heaviest-1 | 0.1098 |  | -0.0010 [-0.0217, +0.0191] |
| herding-1 | 0.1025 | +0.0073 [-0.0080, +0.0228] | +0.0062 [-0.0159, +0.0262] |
| greedy-1 | 0.1174 | -0.0074 [-0.0159, +0.0017] | -0.0084 [-0.0317, +0.0134] |
| heaviest-2 | 0.1038 |  | +0.0051 [-0.0118, +0.0227] |
| herding-2 | 0.1130 | -0.0094 [-0.0256, +0.0058] | -0.0043 [-0.0247, +0.0153] |
| greedy-2 | 0.1318 | -0.0282 [-0.0501, -0.0088] | -0.0231 [-0.0469, -0.0020] |
| heaviest-4 | 0.1102 |  | -0.0013 [-0.0215, +0.0186] |
| herding-4 | 0.1089 | +0.0010 [-0.0133, +0.0150] | -0.0003 [-0.0186, +0.0184] |
| greedy-4 | 0.1177 | -0.0079 [-0.0281, +0.0111] | -0.0092 [-0.0286, +0.0093] |
| random-4-a | 0.1204 | -0.0097 [-0.0249, +0.0057] | -0.0110 [-0.0381, +0.0148] |
| random-4-b | 0.1159 | -0.0053 [-0.0177, +0.0075] | -0.0066 [-0.0249, +0.0127] |
| random-4-c | 0.1079 | +0.0022 [-0.0070, +0.0119] | +0.0009 [-0.0192, +0.0209] |
| heaviest-8 | 0.0946 |  | +0.0142 [+0.0005, +0.0293] |
| greedy-8 | 0.1081 | -0.0131 [-0.0278, +0.0001] | +0.0011 [-0.0079, +0.0105] |
| heaviest-16 | 0.1091 |  |  |

### log10 conductivity at 298 K, by-composition folds

| shells | mean absolute error | gain over the same number of heaviest | gain over all 16 |
|---|---:|---:|---:|
| heaviest-1 | 0.1157 |  | -0.0124 [-0.0301, +0.0044] |
| herding-1 | 0.1117 | +0.0041 [-0.0152, +0.0212] | -0.0082 [-0.0260, +0.0085] |
| greedy-1 | 0.1217 | -0.0063 [-0.0188, +0.0061] | -0.0187 [-0.0374, -0.0013] |
| heaviest-2 | 0.1250 |  | -0.0209 [-0.0418, +0.0000] |
| herding-2 | 0.1148 | +0.0095 [-0.0131, +0.0304] | -0.0115 [-0.0292, +0.0053] |
| greedy-2 | 0.1298 | -0.0059 [-0.0300, +0.0162] | -0.0268 [-0.0455, -0.0088] |
| heaviest-4 | 0.1261 |  | -0.0227 [-0.0428, -0.0043] |
| herding-4 | 0.1141 | +0.0116 [-0.0084, +0.0326] | -0.0110 [-0.0276, +0.0055] |
| greedy-4 | 0.1137 | +0.0125 [-0.0048, +0.0295] | -0.0102 [-0.0243, +0.0049] |
| random-4-a | 0.1268 | -0.0009 [-0.0176, +0.0154] | -0.0236 [-0.0452, -0.0026] |
| random-4-b | 0.1174 | +0.0087 [-0.0094, +0.0271] | -0.0139 [-0.0301, +0.0013] |
| random-4-c | 0.1043 | +0.0216 [+0.0060, +0.0374] | -0.0011 [-0.0170, +0.0142] |
| heaviest-8 | 0.1020 |  | +0.0012 [-0.0086, +0.0124] |
| greedy-8 | 0.1011 | +0.0010 [-0.0136, +0.0159] | +0.0022 [-0.0095, +0.0141] |
| heaviest-16 | 0.1034 |  |  |

### activation energy at 298 K, random folds

| shells | mean absolute error | gain over the same number of heaviest | gain over all 16 |
|---|---:|---:|---:|
| heaviest-1 | 2.0515 |  | -0.4516 [-0.8554, -0.0656] |
| herding-1 | 2.6456 | -0.5828 [-1.0045, -0.1626] | -1.0344 [-1.6121, -0.4935] |
| greedy-1 | 2.2485 | -0.1790 [-0.5017, +0.1635] | -0.6306 [-1.1473, -0.1279] |
| heaviest-2 | 2.0655 |  | -0.4527 [-0.9768, +0.0111] |
| herding-2 | 2.6840 | -0.6114 [-1.1073, -0.1871] | -1.0641 [-1.6378, -0.5188] |
| greedy-2 | 1.9499 | +0.1109 [-0.3794, +0.6140] | -0.3418 [-0.6817, +0.0024] |
| heaviest-4 | 1.9381 |  | -0.3268 [-0.7881, +0.0984] |
| herding-4 | 1.9113 | +0.0260 [-0.2330, +0.3105] | -0.3009 [-0.7296, +0.0934] |
| greedy-4 | 2.0483 | -0.1245 [-0.5848, +0.3316] | -0.4513 [-0.8302, -0.1036] |
| random-4-a | 2.3199 | -0.3783 [-0.7791, +0.0200] | -0.7052 [-1.2229, -0.1950] |
| random-4-b | 2.1926 | -0.2633 [-0.6277, +0.0788] | -0.5901 [-1.1808, -0.0889] |
| random-4-c | 2.3454 | -0.4064 [-0.7910, -0.0510] | -0.7332 [-1.2093, -0.2946] |
| heaviest-8 | 1.6557 |  | -0.0403 [-0.2276, +0.1477] |
| greedy-8 | 1.7214 | -0.0729 [-0.2795, +0.1281] | -0.1132 [-0.3002, +0.0809] |
| heaviest-16 | 1.6121 |  |  |

### activation energy at 298 K, by-composition folds

| shells | mean absolute error | gain over the same number of heaviest | gain over all 16 |
|---|---:|---:|---:|
| heaviest-1 | 2.1542 |  | -0.4737 [-0.8515, -0.0913] |
| herding-1 | 2.6483 | -0.4864 [-0.9901, -0.0068] | -0.9601 [-1.6042, -0.3799] |
| greedy-1 | 2.3440 | -0.1946 [-0.5366, +0.1296] | -0.6683 [-1.0549, -0.2866] |
| heaviest-2 | 2.1977 |  | -0.5184 [-1.0204, -0.0709] |
| herding-2 | 2.4371 | -0.2367 [-0.5759, +0.1065] | -0.7551 [-1.3177, -0.2762] |
| greedy-2 | 2.0818 | +0.1202 [-0.2749, +0.5338] | -0.3982 [-0.7317, -0.0673] |
| heaviest-4 | 2.0357 |  | -0.3564 [-0.7101, -0.0028] |
| herding-4 | 2.0296 | +0.0055 [-0.3795, +0.3695] | -0.3509 [-0.8042, +0.0741] |
| greedy-4 | 2.0174 | +0.0244 [-0.3038, +0.3452] | -0.3320 [-0.7052, +0.0172] |
| random-4-a | 2.7312 | -0.7122 [-1.3362, -0.1543] | -1.0687 [-1.6352, -0.5391] |
| random-4-b | 2.4670 | -0.4313 [-0.9938, +0.0100] | -0.7877 [-1.3887, -0.2944] |
| random-4-c | 2.5885 | -0.5483 [-1.0088, -0.1294] | -0.9048 [-1.4128, -0.4478] |
| heaviest-8 | 1.6670 |  | +0.0134 [-0.2050, +0.2359] |
| greedy-8 | 1.7095 | -0.0420 [-0.2798, +0.1939] | -0.0286 [-0.2003, +0.1545] |
| heaviest-16 | 1.6845 |  |  |

Control: the 4 heaviest shells run again against the stored records, largest difference 3.52e-13.

