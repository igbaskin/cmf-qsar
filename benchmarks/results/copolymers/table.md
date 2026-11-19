Plan item PL4: 1575 copolymers of the VIPEA set (computed electron affinity and ionization potential, eV), each described by the 1384 trimers of its local environments. Five random folds and five folds of whole dibromides, each repeated 5 times; the setting and the ridge are chosen by the evidence on the training block, among all ridges ('evidence') or among the ridges of at least 0.1 ('ridge floor'). 'Architecture r' is the correlation of the predicted with the computed differences between the architectures of one pair of monomers at one composition, with dibromides held out.

**electron affinity, evidence** (1575 copolymers, spread 0.669 eV)

| Model | q², random folds | RMSE, random folds | q², dibromides held out | RMSE, dibromides held out | architecture r |
|---|---:|---:|---:|---:|---:|
| field kernel, sequence ensemble | +0.938 | 0.166 | +0.422 | 0.507 | +0.16 |
| field kernel, mean of the homopolymer units | +0.916 | 0.194 | +0.591 | 0.427 | -0.01 |
| bond-count kernel, sequence ensemble | +0.925 | 0.183 | +0.419 | 0.508 | +0.19 |
| weighted counts | +0.951 | 0.148 | +0.605 | 0.420 | +0.00 |
| weighted counts + architecture | +0.897 | 0.214 | -0.334 | 0.744 | +0.63 |
| weighted ECFP4 | +0.963 | 0.129 | +0.658 | 0.391 | +0.00 |
| sequence ensemble + weighted counts | +0.985 | 0.081 | +0.652 | 0.393 | +0.46 |
| homopolymer mean + weighted counts | +0.970 | 0.115 | +0.717 | 0.355 | -0.01 |
| bond-count ensemble + weighted counts | +0.973 | 0.109 | -0.445 | 0.800 | +0.28 |

| Difference of q² | Folds | Mean [95 % interval] |
|---|---|---|
| sequence ensemble + weighted counts minus weighted counts + architecture | random | +0.089 [+0.078, +0.099] |
| sequence ensemble + weighted counts minus weighted counts + architecture | group | +0.799 [+0.074, +2.447] |
| sequence ensemble + weighted counts minus weighted counts | random | +0.032 [+0.028, +0.036] |
| sequence ensemble + weighted counts minus weighted counts | group | +0.063 [+0.007, +0.127] |
| field kernel, sequence ensemble minus field kernel, mean of the homopolymer units | random | +0.025 [+0.019, +0.032] |
| field kernel, sequence ensemble minus field kernel, mean of the homopolymer units | group | -0.086 [-0.218, +0.043] |
| sequence ensemble + weighted counts minus homopolymer mean + weighted counts | random | +0.018 [+0.015, +0.021] |
| sequence ensemble + weighted counts minus homopolymer mean + weighted counts | group | -0.044 [-0.185, +0.085] |
| sequence ensemble + weighted counts minus bond-count ensemble + weighted counts | random | +0.013 [+0.011, +0.016] |
| sequence ensemble + weighted counts minus bond-count ensemble + weighted counts | group | +0.547 [+0.051, +1.322] |
| sequence ensemble + weighted counts minus weighted ECFP4 | random | +0.025 [+0.021, +0.029] |
| sequence ensemble + weighted counts minus weighted ECFP4 | group | +0.031 [-0.097, +0.160] |

**electron affinity, ridge floor** (1575 copolymers, spread 0.669 eV)

| Model | q², random folds | RMSE, random folds | q², dibromides held out | RMSE, dibromides held out | architecture r |
|---|---:|---:|---:|---:|---:|
| field kernel, sequence ensemble | +0.939 | 0.165 | +0.639 | 0.401 | +0.17 |
| field kernel, mean of the homopolymer units | +0.916 | 0.194 | +0.590 | 0.428 | -0.02 |
| bond-count kernel, sequence ensemble | +0.917 | 0.192 | +0.513 | 0.466 | +0.24 |
| weighted counts | +0.938 | 0.166 | +0.648 | 0.397 | +0.00 |
| weighted counts + architecture | +0.887 | 0.224 | +0.603 | 0.420 | +0.60 |
| weighted ECFP4 | +0.963 | 0.129 | +0.658 | 0.391 | +0.00 |
| sequence ensemble + weighted counts | +0.983 | 0.087 | +0.714 | 0.357 | +0.44 |
| homopolymer mean + weighted counts | +0.970 | 0.115 | +0.758 | 0.329 | -0.01 |
| bond-count ensemble + weighted counts | +0.970 | 0.116 | +0.753 | 0.332 | +0.43 |

| Difference of q² | Folds | Mean [95 % interval] |
|---|---|---|
| sequence ensemble + weighted counts minus weighted counts + architecture | random | +0.094 [+0.083, +0.106] |
| sequence ensemble + weighted counts minus weighted counts + architecture | group | +0.096 [+0.001, +0.222] |
| sequence ensemble + weighted counts minus weighted counts | random | +0.045 [+0.040, +0.051] |
| sequence ensemble + weighted counts minus weighted counts | group | +0.049 [-0.002, +0.104] |
| field kernel, sequence ensemble minus field kernel, mean of the homopolymer units | random | +0.025 [+0.019, +0.031] |
| field kernel, sequence ensemble minus field kernel, mean of the homopolymer units | group | +0.052 [-0.080, +0.226] |
| sequence ensemble + weighted counts minus homopolymer mean + weighted counts | random | +0.014 [+0.012, +0.017] |
| sequence ensemble + weighted counts minus homopolymer mean + weighted counts | group | -0.039 [-0.132, +0.048] |
| sequence ensemble + weighted counts minus bond-count ensemble + weighted counts | random | +0.014 [+0.011, +0.016] |
| sequence ensemble + weighted counts minus bond-count ensemble + weighted counts | group | -0.040 [-0.088, +0.005] |
| sequence ensemble + weighted counts minus weighted ECFP4 | random | +0.022 [+0.018, +0.026] |
| sequence ensemble + weighted counts minus weighted ECFP4 | group | +0.067 [-0.040, +0.175] |

**ionization potential, evidence** (1575 copolymers, spread 0.569 eV)

| Model | q², random folds | RMSE, random folds | q², dibromides held out | RMSE, dibromides held out | architecture r |
|---|---:|---:|---:|---:|---:|
| field kernel, sequence ensemble | +0.919 | 0.162 | +0.771 | 0.272 | +0.19 |
| field kernel, mean of the homopolymer units | +0.904 | 0.176 | +0.700 | 0.311 | +0.01 |
| bond-count kernel, sequence ensemble | +0.894 | 0.185 | +0.729 | 0.294 | +0.17 |
| weighted counts | +0.956 | 0.119 | +0.690 | 0.316 | +0.00 |
| weighted counts + architecture | +0.924 | 0.156 | +0.531 | 0.385 | +0.70 |
| weighted ECFP4 | +0.961 | 0.112 | +0.742 | 0.287 | +0.00 |
| sequence ensemble + weighted counts | +0.987 | 0.065 | +0.614 | 0.352 | +0.72 |
| homopolymer mean + weighted counts | +0.963 | 0.110 | +0.742 | 0.288 | -0.00 |
| bond-count ensemble + weighted counts | +0.983 | 0.075 | +0.713 | 0.304 | +0.69 |

| Difference of q² | Folds | Mean [95 % interval] |
|---|---|---|
| sequence ensemble + weighted counts minus weighted counts + architecture | random | +0.065 [+0.057, +0.073] |
| sequence ensemble + weighted counts minus weighted counts + architecture | group | -0.003 [-0.112, +0.111] |
| sequence ensemble + weighted counts minus weighted counts | random | +0.030 [+0.026, +0.035] |
| sequence ensemble + weighted counts minus weighted counts | group | -0.067 [-0.180, +0.027] |
| field kernel, sequence ensemble minus field kernel, mean of the homopolymer units | random | +0.016 [+0.009, +0.022] |
| field kernel, sequence ensemble minus field kernel, mean of the homopolymer units | group | +0.075 [-0.002, +0.167] |
| sequence ensemble + weighted counts minus homopolymer mean + weighted counts | random | +0.027 [+0.023, +0.032] |
| sequence ensemble + weighted counts minus homopolymer mean + weighted counts | group | -0.088 [-0.199, +0.003] |
| sequence ensemble + weighted counts minus bond-count ensemble + weighted counts | random | +0.005 [+0.004, +0.007] |
| sequence ensemble + weighted counts minus bond-count ensemble + weighted counts | group | -0.097 [-0.194, -0.035] |
| sequence ensemble + weighted counts minus weighted ECFP4 | random | +0.028 [+0.024, +0.033] |
| sequence ensemble + weighted counts minus weighted ECFP4 | group | -0.099 [-0.252, +0.019] |

**ionization potential, ridge floor** (1575 copolymers, spread 0.569 eV)

| Model | q², random folds | RMSE, random folds | q², dibromides held out | RMSE, dibromides held out | architecture r |
|---|---:|---:|---:|---:|---:|
| field kernel, sequence ensemble | +0.919 | 0.162 | +0.771 | 0.272 | +0.19 |
| field kernel, mean of the homopolymer units | +0.904 | 0.176 | +0.700 | 0.311 | +0.01 |
| bond-count kernel, sequence ensemble | +0.892 | 0.187 | +0.811 | 0.247 | +0.23 |
| weighted counts | +0.944 | 0.135 | +0.689 | 0.316 | +0.00 |
| weighted counts + architecture | +0.909 | 0.171 | +0.717 | 0.302 | +0.76 |
| weighted ECFP4 | +0.961 | 0.112 | +0.742 | 0.287 | +0.00 |
| sequence ensemble + weighted counts | +0.984 | 0.071 | +0.675 | 0.323 | +0.75 |
| homopolymer mean + weighted counts | +0.963 | 0.110 | +0.742 | 0.288 | -0.00 |
| bond-count ensemble + weighted counts | +0.976 | 0.087 | +0.734 | 0.292 | +0.65 |

| Difference of q² | Folds | Mean [95 % interval] |
|---|---|---|
| sequence ensemble + weighted counts minus weighted counts + architecture | random | +0.073 [+0.064, +0.081] |
| sequence ensemble + weighted counts minus weighted counts + architecture | group | -0.027 [-0.105, +0.037] |
| sequence ensemble + weighted counts minus weighted counts | random | +0.041 [+0.036, +0.047] |
| sequence ensemble + weighted counts minus weighted counts | group | -0.005 [-0.088, +0.069] |
| field kernel, sequence ensemble minus field kernel, mean of the homopolymer units | random | +0.016 [+0.009, +0.022] |
| field kernel, sequence ensemble minus field kernel, mean of the homopolymer units | group | +0.075 [-0.002, +0.167] |
| sequence ensemble + weighted counts minus homopolymer mean + weighted counts | random | +0.024 [+0.020, +0.028] |
| sequence ensemble + weighted counts minus homopolymer mean + weighted counts | group | -0.040 [-0.119, +0.028] |
| sequence ensemble + weighted counts minus bond-count ensemble + weighted counts | random | +0.008 [+0.006, +0.010] |
| sequence ensemble + weighted counts minus bond-count ensemble + weighted counts | group | -0.053 [-0.130, +0.001] |
| sequence ensemble + weighted counts minus weighted ECFP4 | random | +0.024 [+0.020, +0.029] |
| sequence ensemble + weighted counts minus weighted ECFP4 | group | -0.050 [-0.176, +0.045] |
