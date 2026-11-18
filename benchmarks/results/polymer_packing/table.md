Plan item PL2: density at 300 K from molecular dynamics (RadonPy PI1070) and the measured gas permeability of polymers (log10 Barrer). Each repeat unit is built as a trimer capped with hydrogen. Five random folds and five folds of whole groups (RadonPy polymer classes for the density, ECFP4 Butina clusters for the permeability), each repeated 10 times; the setting and the ridge are chosen by the evidence on the training block.

**density** (1075 polymers, 20 groups, g/cm3, spread 0.180)

| Model | q², random folds | RMSE, random folds | q², groups held out |
|---|---:|---:|---:|
| field kernel, whole trimer | +0.876 | 0.063 | +0.809 |
| field kernel, central unit | +0.872 | 0.064 | +0.795 |
| bond-count kernel, central unit | +0.879 | 0.063 | +0.785 |
| eight counts | +0.664 | 0.105 | +0.021 |
| ECFP4 of the repeat unit | +0.892 | 0.059 | +0.788 |
| whole trimer + eight counts | +0.897 | 0.058 | +0.756 |
| central unit + eight counts | +0.896 | 0.058 | +0.769 |
| bond-count kernel + eight counts | +0.907 | 0.055 | +0.711 |

| Difference of q² | Folds | Mean [95 % interval] |
|---|---|---|
| central unit + eight counts minus bond-count kernel + eight counts | random | -0.009 [-0.030, +0.009] |
| central unit + eight counts minus bond-count kernel + eight counts | group | +0.043 [-0.023, +0.134] |
| whole trimer + eight counts minus bond-count kernel + eight counts | random | -0.008 [-0.033, +0.013] |
| whole trimer + eight counts minus bond-count kernel + eight counts | group | +0.033 [-0.045, +0.139] |
| field kernel, central unit minus bond-count kernel, central unit | random | -0.005 [-0.033, +0.024] |
| field kernel, central unit minus bond-count kernel, central unit | group | +0.002 [-0.044, +0.054] |
| central unit + eight counts minus eight counts | random | +0.232 [+0.176, +0.292] |
| central unit + eight counts minus eight counts | group | +0.644 [+0.098, +1.532] |
| central unit + eight counts minus ECFP4 of the repeat unit | random | +0.007 [-0.010, +0.025] |
| central unit + eight counts minus ECFP4 of the repeat unit | group | -0.006 [-0.192, +0.101] |
| field kernel, central unit minus field kernel, whole trimer | random | -0.004 [-0.012, +0.004] |
| field kernel, central unit minus field kernel, whole trimer | group | -0.014 [-0.048, +0.003] |

**CH4** (600 polymers, 102 groups, log10 Barrer, spread 1.414)

| Model | q², random folds | RMSE, random folds | q², groups held out |
|---|---:|---:|---:|
| field kernel, whole trimer | +0.724 | 0.744 | +0.494 |
| field kernel, central unit | +0.710 | 0.761 | +0.475 |
| bond-count kernel, central unit | +0.664 | 0.820 | +0.432 |
| eight counts | +0.694 | 0.782 | +0.440 |
| ECFP4 of the repeat unit | +0.856 | 0.536 | +0.669 |
| whole trimer + eight counts | +0.795 | 0.641 | +0.575 |
| central unit + eight counts | +0.788 | 0.652 | +0.557 |
| bond-count kernel + eight counts | +0.746 | 0.712 | +0.480 |

| Difference of q² | Folds | Mean [95 % interval] |
|---|---|---|
| central unit + eight counts minus bond-count kernel + eight counts | random | +0.039 [+0.013, +0.065] |
| central unit + eight counts minus bond-count kernel + eight counts | group | +0.059 [+0.017, +0.103] |
| whole trimer + eight counts minus bond-count kernel + eight counts | random | +0.046 [+0.019, +0.073] |
| whole trimer + eight counts minus bond-count kernel + eight counts | group | +0.078 [+0.034, +0.126] |
| field kernel, central unit minus bond-count kernel, central unit | random | +0.047 [+0.013, +0.086] |
| field kernel, central unit minus bond-count kernel, central unit | group | +0.026 [-0.055, +0.162] |
| central unit + eight counts minus eight counts | random | +0.091 [+0.061, +0.125] |
| central unit + eight counts minus eight counts | group | +0.101 [+0.055, +0.185] |
| central unit + eight counts minus ECFP4 of the repeat unit | random | -0.068 [-0.094, -0.045] |
| central unit + eight counts minus ECFP4 of the repeat unit | group | -0.099 [-0.219, -0.053] |
| field kernel, central unit minus field kernel, whole trimer | random | -0.014 [-0.029, +0.001] |
| field kernel, central unit minus field kernel, whole trimer | group | -0.021 [-0.040, -0.001] |

**CO2** (674 polymers, 116 groups, log10 Barrer, spread 1.166)

| Model | q², random folds | RMSE, random folds | q², groups held out |
|---|---:|---:|---:|
| field kernel, whole trimer | +0.625 | 0.714 | +0.450 |
| field kernel, central unit | +0.602 | 0.736 | +0.409 |
| bond-count kernel, central unit | +0.528 | 0.801 | +0.274 |
| eight counts | +0.521 | 0.807 | +0.340 |
| ECFP4 of the repeat unit | +0.795 | 0.528 | +0.608 |
| whole trimer + eight counts | +0.680 | 0.659 | +0.539 |
| central unit + eight counts | +0.681 | 0.659 | +0.520 |
| bond-count kernel + eight counts | +0.634 | 0.705 | +0.405 |

| Difference of q² | Folds | Mean [95 % interval] |
|---|---|---|
| central unit + eight counts minus bond-count kernel + eight counts | random | +0.031 [-0.004, +0.068] |
| central unit + eight counts minus bond-count kernel + eight counts | group | +0.083 [+0.031, +0.134] |
| whole trimer + eight counts minus bond-count kernel + eight counts | random | +0.031 [-0.003, +0.069] |
| whole trimer + eight counts minus bond-count kernel + eight counts | group | +0.101 [+0.044, +0.152] |
| field kernel, central unit minus bond-count kernel, central unit | random | +0.065 [+0.016, +0.115] |
| field kernel, central unit minus bond-count kernel, central unit | group | +0.095 [+0.020, +0.212] |
| central unit + eight counts minus eight counts | random | +0.159 [+0.118, +0.201] |
| central unit + eight counts minus eight counts | group | +0.170 [+0.120, +0.247] |
| central unit + eight counts minus ECFP4 of the repeat unit | random | -0.102 [-0.137, -0.070] |
| central unit + eight counts minus ECFP4 of the repeat unit | group | -0.068 [-0.142, -0.020] |
| field kernel, central unit minus field kernel, whole trimer | random | -0.022 [-0.046, -0.001] |
| field kernel, central unit minus field kernel, whole trimer | group | -0.042 [-0.079, +0.000] |

**H2** (444 polymers, 79 groups, log10 Barrer, spread 0.960)

| Model | q², random folds | RMSE, random folds | q², groups held out |
|---|---:|---:|---:|
| field kernel, whole trimer | +0.702 | 0.524 | +0.421 |
| field kernel, central unit | +0.682 | 0.541 | +0.378 |
| bond-count kernel, central unit | +0.615 | 0.595 | +0.259 |
| eight counts | +0.699 | 0.526 | +0.375 |
| ECFP4 of the repeat unit | +0.825 | 0.402 | +0.544 |
| whole trimer + eight counts | +0.779 | 0.451 | +0.531 |
| central unit + eight counts | +0.772 | 0.458 | +0.496 |
| bond-count kernel + eight counts | +0.749 | 0.480 | +0.398 |

| Difference of q² | Folds | Mean [95 % interval] |
|---|---|---|
| central unit + eight counts minus bond-count kernel + eight counts | random | +0.012 [-0.010, +0.034] |
| central unit + eight counts minus bond-count kernel + eight counts | group | +0.072 [+0.015, +0.135] |
| whole trimer + eight counts minus bond-count kernel + eight counts | random | +0.021 [-0.004, +0.046] |
| whole trimer + eight counts minus bond-count kernel + eight counts | group | +0.106 [+0.031, +0.191] |
| field kernel, central unit minus bond-count kernel, central unit | random | +0.046 [-0.003, +0.092] |
| field kernel, central unit minus bond-count kernel, central unit | group | +0.056 [-0.027, +0.227] |
| central unit + eight counts minus eight counts | random | +0.060 [+0.035, +0.089] |
| central unit + eight counts minus eight counts | group | +0.110 [+0.071, +0.208] |
| central unit + eight counts minus ECFP4 of the repeat unit | random | -0.049 [-0.082, -0.017] |
| central unit + eight counts minus ECFP4 of the repeat unit | group | -0.040 [-0.180, +0.047] |
| field kernel, central unit minus field kernel, whole trimer | random | -0.021 [-0.042, +0.000] |
| field kernel, central unit minus field kernel, whole trimer | group | -0.046 [-0.088, +0.021] |

**N2** (709 polymers, 124 groups, log10 Barrer, spread 1.255)

| Model | q², random folds | RMSE, random folds | q², groups held out |
|---|---:|---:|---:|
| field kernel, whole trimer | +0.690 | 0.698 | +0.466 |
| field kernel, central unit | +0.652 | 0.740 | +0.411 |
| bond-count kernel, central unit | +0.566 | 0.827 | +0.324 |
| eight counts | +0.623 | 0.770 | +0.372 |
| ECFP4 of the repeat unit | +0.826 | 0.523 | +0.635 |
| whole trimer + eight counts | +0.755 | 0.622 | +0.573 |
| central unit + eight counts | +0.743 | 0.636 | +0.540 |
| bond-count kernel + eight counts | +0.684 | 0.706 | +0.425 |

| Difference of q² | Folds | Mean [95 % interval] |
|---|---|---|
| central unit + eight counts minus bond-count kernel + eight counts | random | +0.043 [+0.016, +0.072] |
| central unit + eight counts minus bond-count kernel + eight counts | group | +0.084 [+0.024, +0.137] |
| whole trimer + eight counts minus bond-count kernel + eight counts | random | +0.051 [+0.023, +0.079] |
| whole trimer + eight counts minus bond-count kernel + eight counts | group | +0.116 [+0.035, +0.187] |
| field kernel, central unit minus bond-count kernel, central unit | random | +0.063 [+0.019, +0.110] |
| field kernel, central unit minus bond-count kernel, central unit | group | +0.050 [-0.025, +0.189] |
| central unit + eight counts minus eight counts | random | +0.118 [+0.086, +0.151] |
| central unit + eight counts minus eight counts | group | +0.155 [+0.115, +0.241] |
| central unit + eight counts minus ECFP4 of the repeat unit | random | -0.074 [-0.102, -0.046] |
| central unit + eight counts minus ECFP4 of the repeat unit | group | -0.079 [-0.179, -0.032] |
| field kernel, central unit minus field kernel, whole trimer | random | -0.037 [-0.058, -0.019] |
| field kernel, central unit minus field kernel, whole trimer | group | -0.055 [-0.082, -0.024] |

**O2** (721 polymers, 124 groups, log10 Barrer, spread 1.170)

| Model | q², random folds | RMSE, random folds | q², groups held out |
|---|---:|---:|---:|
| field kernel, whole trimer | +0.659 | 0.683 | +0.475 |
| field kernel, central unit | +0.634 | 0.707 | +0.439 |
| bond-count kernel, central unit | +0.583 | 0.756 | +0.356 |
| eight counts | +0.571 | 0.766 | +0.337 |
| ECFP4 of the repeat unit | +0.810 | 0.510 | +0.599 |
| whole trimer + eight counts | +0.719 | 0.620 | +0.549 |
| central unit + eight counts | +0.723 | 0.616 | +0.540 |
| bond-count kernel + eight counts | +0.678 | 0.663 | +0.434 |

| Difference of q² | Folds | Mean [95 % interval] |
|---|---|---|
| central unit + eight counts minus bond-count kernel + eight counts | random | +0.039 [+0.011, +0.066] |
| central unit + eight counts minus bond-count kernel + eight counts | group | +0.081 [+0.013, +0.139] |
| whole trimer + eight counts minus bond-count kernel + eight counts | random | +0.034 [+0.005, +0.064] |
| whole trimer + eight counts minus bond-count kernel + eight counts | group | +0.089 [+0.008, +0.158] |
| field kernel, central unit minus bond-count kernel, central unit | random | +0.052 [+0.010, +0.093] |
| field kernel, central unit minus bond-count kernel, central unit | group | +0.064 [+0.008, +0.153] |
| central unit + eight counts minus eight counts | random | +0.143 [+0.105, +0.184] |
| central unit + eight counts minus eight counts | group | +0.187 [+0.142, +0.260] |
| central unit + eight counts minus ECFP4 of the repeat unit | random | -0.077 [-0.112, -0.044] |
| central unit + eight counts minus ECFP4 of the repeat unit | group | -0.033 [-0.132, +0.023] |
| field kernel, central unit minus field kernel, whole trimer | random | -0.025 [-0.045, -0.004] |
| field kernel, central unit minus field kernel, whole trimer | group | -0.037 [-0.068, +0.003] |
