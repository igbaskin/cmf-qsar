Plan item PL6: ladder polymers. The measured permeability (log10 Barrer) of 762 polymers, 27 of them ladder polymers; 102 repeat units with a repaired structure. Five random folds and five folds of ECFP4 Butina clusters, each repeated 10 times; for the ladders held out, the models are trained on the linear polymers only. The setting and the ridge are chosen by the evidence on the training block.

Built here: 27 ladder trimers (0 with geometry faults) and 70 repaired linear trimers (0 with faults). End-to-end distance of the ladder trimers: 4.0 to 55.1 A.

**CH4** (600 linear polymers, 25 ladders; mean 0.18 for the linear polymers and 1.90 for the ladders)

| Model | q², linear, random folds | q², linear, clusters held out | q², all, random folds | q², all, clusters held out | RMSE on the ladders, held out | mean error on the ladders, held out | RMSE on the ladders, random folds of all |
|---|---:|---:|---:|---:|---:|---:|---:|
| field kernel, central unit | +0.711 | +0.479 | +0.710 | +0.423 | 2.124 | -1.685 | 1.015 |
| bond-count kernel, central unit | +0.665 | +0.421 | +0.664 | +0.410 | 1.621 | -1.062 | 1.072 |
| eight counts | +0.702 | +0.466 | +0.711 | +0.422 | 2.041 | -1.633 | 0.925 |
| ECFP4 of the repeat unit | +0.853 | +0.634 | +0.858 | +0.587 | 1.709 | -1.343 | 0.703 |
| central unit + eight counts | +0.791 | +0.573 | +0.790 | +0.503 | 2.106 | -1.694 | 0.843 |
| bond-count kernel + eight counts | +0.752 | +0.499 | +0.760 | +0.474 | 1.626 | -1.210 | 0.860 |
| eight counts + side groups | +0.753 | +0.592 | +0.753 | +0.514 | 2.101 | -1.677 | 0.942 |
| central unit + eight counts + side groups | +0.809 | +0.663 | +0.804 | +0.588 | 2.048 | -1.615 | 0.883 |
| bond-count kernel + eight counts + side groups | +0.796 | +0.619 | +0.797 | +0.593 | 1.950 | -1.532 | 0.836 |

| Difference | Polymers | Folds | Measure | Mean [95 % interval] |
|---|---|---|---|---|
| central unit + eight counts minus bond-count kernel + eight counts | linear | random | q2 | +0.036 [+0.012, +0.061] |
| central unit + eight counts minus bond-count kernel + eight counts | linear | group | q2 | +0.058 [+0.015, +0.113] |
| central unit + eight counts minus ECFP4 of the repeat unit | linear | random | q2 | -0.063 [-0.087, -0.040] |
| central unit + eight counts minus ECFP4 of the repeat unit | linear | group | q2 | -0.049 [-0.173, +0.000] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | linear | random | q2 | +0.011 [-0.009, +0.031] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | linear | group | q2 | +0.029 [-0.010, +0.092] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | linear | random | q2 | -0.042 [-0.063, -0.022] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | linear | group | q2 | +0.037 [-0.104, +0.138] |
| central unit + eight counts minus bond-count kernel + eight counts | all | random | q2 | +0.030 [+0.010, +0.052] |
| central unit + eight counts minus bond-count kernel + eight counts | all | group | q2 | +0.009 [-0.086, +0.071] |
| central unit + eight counts minus bond-count kernel + eight counts | all | ladders held out | RMSE | +0.479 [+0.197, +0.749] |
| central unit + eight counts minus ECFP4 of the repeat unit | all | random | q2 | -0.066 [-0.091, -0.042] |
| central unit + eight counts minus ECFP4 of the repeat unit | all | group | q2 | -0.077 [-0.208, -0.014] |
| central unit + eight counts minus ECFP4 of the repeat unit | all | ladders held out | RMSE | +0.397 [+0.088, +0.680] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | all | random | q2 | +0.006 [-0.011, +0.024] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | all | group | q2 | -0.015 [-0.082, +0.033] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | all | ladders held out | RMSE | +0.098 [-0.154, +0.335] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | all | random | q2 | -0.050 [-0.072, -0.028] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | all | group | q2 | +0.012 [-0.128, +0.134] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | all | ladders held out | RMSE | +0.339 [+0.088, +0.572] |

**CO2** (674 linear polymers, 24 ladders; mean 1.37 for the linear polymers and 3.10 for the ladders)

| Model | q², linear, random folds | q², linear, clusters held out | q², all, random folds | q², all, clusters held out | RMSE on the ladders, held out | mean error on the ladders, held out | RMSE on the ladders, random folds of all |
|---|---:|---:|---:|---:|---:|---:|---:|
| field kernel, central unit | +0.607 | +0.359 | +0.623 | +0.346 | 2.035 | -1.703 | 0.858 |
| bond-count kernel, central unit | +0.524 | +0.175 | +0.548 | +0.233 | 1.287 | -0.912 | 0.981 |
| eight counts | +0.533 | +0.319 | +0.558 | +0.304 | 2.017 | -1.741 | 0.914 |
| ECFP4 of the repeat unit | +0.788 | +0.578 | +0.793 | +0.554 | 1.751 | -1.427 | 0.656 |
| central unit + eight counts | +0.680 | +0.467 | +0.703 | +0.452 | 2.131 | -1.837 | 0.727 |
| bond-count kernel + eight counts | +0.638 | +0.359 | +0.653 | +0.393 | 1.527 | -1.276 | 0.773 |
| eight counts + side groups | +0.577 | +0.391 | +0.594 | +0.397 | 1.573 | -1.143 | 0.899 |
| central unit + eight counts + side groups | +0.719 | +0.556 | +0.731 | +0.563 | 1.504 | -1.010 | 0.714 |
| bond-count kernel + eight counts + side groups | +0.675 | +0.442 | +0.683 | +0.477 | 1.360 | -0.983 | 0.787 |

| Difference | Polymers | Folds | Measure | Mean [95 % interval] |
|---|---|---|---|---|
| central unit + eight counts minus bond-count kernel + eight counts | linear | random | q2 | +0.024 [-0.010, +0.059] |
| central unit + eight counts minus bond-count kernel + eight counts | linear | group | q2 | +0.063 [+0.008, +0.117] |
| central unit + eight counts minus ECFP4 of the repeat unit | linear | random | q2 | -0.096 [-0.132, -0.062] |
| central unit + eight counts minus ECFP4 of the repeat unit | linear | group | q2 | -0.084 [-0.160, -0.033] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | linear | random | q2 | +0.027 [-0.001, +0.057] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | linear | group | q2 | +0.072 [+0.016, +0.145] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | linear | random | q2 | -0.062 [-0.091, -0.034] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | linear | group | q2 | +0.004 [-0.097, +0.088] |
| central unit + eight counts minus bond-count kernel + eight counts | all | random | q2 | +0.035 [+0.004, +0.067] |
| central unit + eight counts minus bond-count kernel + eight counts | all | group | q2 | +0.019 [-0.092, +0.094] |
| central unit + eight counts minus bond-count kernel + eight counts | all | ladders held out | RMSE | +0.604 [+0.348, +0.871] |
| central unit + eight counts minus ECFP4 of the repeat unit | all | random | q2 | -0.081 [-0.112, -0.051] |
| central unit + eight counts minus ECFP4 of the repeat unit | all | group | q2 | -0.074 [-0.139, -0.026] |
| central unit + eight counts minus ECFP4 of the repeat unit | all | ladders held out | RMSE | +0.380 [+0.189, +0.559] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | all | random | q2 | +0.030 [+0.004, +0.057] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | all | group | q2 | +0.047 [-0.023, +0.107] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | all | ladders held out | RMSE | +0.143 [-0.107, +0.376] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | all | random | q2 | -0.056 [-0.083, -0.031] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | all | group | q2 | +0.025 [-0.074, +0.108] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | all | ladders held out | RMSE | -0.248 [-0.443, -0.058] |

**H2** (444 linear polymers, 23 ladders; mean 1.70 for the linear polymers and 3.26 for the ladders)

| Model | q², linear, random folds | q², linear, clusters held out | q², all, random folds | q², all, clusters held out | RMSE on the ladders, held out | mean error on the ladders, held out | RMSE on the ladders, random folds of all |
|---|---:|---:|---:|---:|---:|---:|---:|
| field kernel, central unit | +0.678 | +0.350 | +0.692 | +0.331 | 1.647 | -1.488 | 0.704 |
| bond-count kernel, central unit | +0.610 | +0.234 | +0.588 | +0.198 | 1.473 | -1.172 | 0.740 |
| eight counts | +0.709 | +0.361 | +0.715 | +0.329 | 1.666 | -1.623 | 0.684 |
| ECFP4 of the repeat unit | +0.817 | +0.541 | +0.830 | +0.544 | 1.331 | -1.241 | 0.470 |
| central unit + eight counts | +0.765 | +0.441 | +0.772 | +0.424 | 1.678 | -1.606 | 0.638 |
| bond-count kernel + eight counts | +0.747 | +0.388 | +0.739 | +0.381 | 1.393 | -1.272 | 0.600 |
| eight counts + side groups | +0.742 | +0.491 | +0.757 | +0.450 | 1.614 | -1.563 | 0.580 |
| central unit + eight counts + side groups | +0.767 | +0.524 | +0.779 | +0.485 | 1.617 | -1.564 | 0.556 |
| bond-count kernel + eight counts + side groups | +0.761 | +0.518 | +0.763 | +0.483 | 1.451 | -1.378 | 0.528 |

| Difference | Polymers | Folds | Measure | Mean [95 % interval] |
|---|---|---|---|---|
| central unit + eight counts minus bond-count kernel + eight counts | linear | random | q2 | +0.006 [-0.014, +0.026] |
| central unit + eight counts minus bond-count kernel + eight counts | linear | group | q2 | +0.025 [-0.051, +0.110] |
| central unit + eight counts minus ECFP4 of the repeat unit | linear | random | q2 | -0.046 [-0.079, -0.015] |
| central unit + eight counts minus ECFP4 of the repeat unit | linear | group | q2 | -0.082 [-0.226, +0.004] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | linear | random | q2 | -0.001 [-0.018, +0.015] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | linear | group | q2 | -0.016 [-0.065, +0.042] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | linear | random | q2 | -0.037 [-0.071, -0.004] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | linear | group | q2 | +0.014 [-0.156, +0.113] |
| central unit + eight counts minus bond-count kernel + eight counts | all | random | q2 | +0.011 [-0.007, +0.028] |
| central unit + eight counts minus bond-count kernel + eight counts | all | group | q2 | +0.008 [-0.097, +0.095] |
| central unit + eight counts minus bond-count kernel + eight counts | all | ladders held out | RMSE | +0.284 [+0.103, +0.460] |
| central unit + eight counts minus ECFP4 of the repeat unit | all | random | q2 | -0.052 [-0.084, -0.024] |
| central unit + eight counts minus ECFP4 of the repeat unit | all | group | q2 | -0.098 [-0.233, -0.004] |
| central unit + eight counts minus ECFP4 of the repeat unit | all | ladders held out | RMSE | +0.346 [+0.217, +0.474] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | all | random | q2 | +0.007 [-0.007, +0.020] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | all | group | q2 | -0.021 [-0.086, +0.042] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | all | ladders held out | RMSE | +0.165 [+0.058, +0.267] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | all | random | q2 | -0.040 [-0.074, -0.008] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | all | group | q2 | -0.025 [-0.160, +0.083] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | all | ladders held out | RMSE | +0.285 [+0.194, +0.377] |

**N2** (709 linear polymers, 27 ladders; mean 0.08 for the linear polymers and 1.81 for the ladders)

| Model | q², linear, random folds | q², linear, clusters held out | q², all, random folds | q², all, clusters held out | RMSE on the ladders, held out | mean error on the ladders, held out | RMSE on the ladders, random folds of all |
|---|---:|---:|---:|---:|---:|---:|---:|
| field kernel, central unit | +0.651 | +0.427 | +0.657 | +0.401 | 1.935 | -1.633 | 0.928 |
| bond-count kernel, central unit | +0.565 | +0.295 | +0.572 | +0.310 | 1.305 | -0.786 | 1.007 |
| eight counts | +0.637 | +0.406 | +0.663 | +0.402 | 1.815 | -1.493 | 0.775 |
| ECFP4 of the repeat unit | +0.820 | +0.618 | +0.824 | +0.603 | 1.722 | -1.430 | 0.696 |
| central unit + eight counts | +0.747 | +0.547 | +0.757 | +0.533 | 1.730 | -1.419 | 0.716 |
| bond-count kernel + eight counts | +0.690 | +0.438 | +0.698 | +0.460 | 1.271 | -0.927 | 0.729 |
| eight counts + side groups | +0.704 | +0.531 | +0.699 | +0.518 | 1.722 | -1.360 | 0.763 |
| central unit + eight counts + side groups | +0.770 | +0.636 | +0.772 | +0.614 | 1.638 | -1.296 | 0.731 |
| bond-count kernel + eight counts + side groups | +0.747 | +0.570 | +0.747 | +0.556 | 1.592 | -1.218 | 0.703 |

| Difference | Polymers | Folds | Measure | Mean [95 % interval] |
|---|---|---|---|---|
| central unit + eight counts minus bond-count kernel + eight counts | linear | random | q2 | +0.040 [+0.014, +0.069] |
| central unit + eight counts minus bond-count kernel + eight counts | linear | group | q2 | +0.073 [+0.030, +0.131] |
| central unit + eight counts minus ECFP4 of the repeat unit | linear | random | q2 | -0.065 [-0.092, -0.039] |
| central unit + eight counts minus ECFP4 of the repeat unit | linear | group | q2 | -0.049 [-0.155, -0.001] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | linear | random | q2 | +0.013 [-0.010, +0.037] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | linear | group | q2 | +0.040 [+0.000, +0.100] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | linear | random | q2 | -0.046 [-0.069, -0.023] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | linear | group | q2 | +0.033 [-0.100, +0.130] |
| central unit + eight counts minus bond-count kernel + eight counts | all | random | q2 | +0.043 [+0.019, +0.069] |
| central unit + eight counts minus bond-count kernel + eight counts | all | group | q2 | +0.040 [-0.045, +0.104] |
| central unit + eight counts minus bond-count kernel + eight counts | all | ladders held out | RMSE | +0.459 [+0.129, +0.750] |
| central unit + eight counts minus ECFP4 of the repeat unit | all | random | q2 | -0.062 [-0.090, -0.036] |
| central unit + eight counts minus ECFP4 of the repeat unit | all | group | q2 | -0.049 [-0.143, -0.001] |
| central unit + eight counts minus ECFP4 of the repeat unit | all | ladders held out | RMSE | +0.008 [-0.252, +0.237] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | all | random | q2 | +0.015 [-0.006, +0.038] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | all | group | q2 | +0.031 [-0.019, +0.070] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | all | ladders held out | RMSE | +0.046 [-0.185, +0.247] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | all | random | q2 | -0.045 [-0.070, -0.021] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | all | group | q2 | +0.028 [-0.096, +0.125] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | all | ladders held out | RMSE | -0.084 [-0.276, +0.091] |

**O2** (721 linear polymers, 27 ladders; mean 0.70 for the linear polymers and 2.43 for the ladders)

| Model | q², linear, random folds | q², linear, clusters held out | q², all, random folds | q², all, clusters held out | RMSE on the ladders, held out | mean error on the ladders, held out | RMSE on the ladders, random folds of all |
|---|---:|---:|---:|---:|---:|---:|---:|
| field kernel, central unit | +0.637 | +0.433 | +0.649 | +0.409 | 1.930 | -1.630 | 0.889 |
| bond-count kernel, central unit | +0.579 | +0.313 | +0.587 | +0.319 | 1.359 | -0.915 | 1.013 |
| eight counts | +0.587 | +0.360 | +0.630 | +0.394 | 1.699 | -1.409 | 0.736 |
| ECFP4 of the repeat unit | +0.802 | +0.583 | +0.806 | +0.570 | 1.733 | -1.454 | 0.658 |
| central unit + eight counts | +0.720 | +0.535 | +0.737 | +0.542 | 1.582 | -1.312 | 0.677 |
| bond-count kernel + eight counts | +0.685 | +0.437 | +0.694 | +0.478 | 1.327 | -0.940 | 0.733 |
| eight counts + side groups | +0.638 | +0.455 | +0.669 | +0.465 | 1.645 | -1.293 | 0.763 |
| central unit + eight counts + side groups | +0.752 | +0.618 | +0.760 | +0.608 | 1.506 | -1.174 | 0.698 |
| bond-count kernel + eight counts + side groups | +0.733 | +0.553 | +0.743 | +0.552 | 1.603 | -1.230 | 0.705 |

| Difference | Polymers | Folds | Measure | Mean [95 % interval] |
|---|---|---|---|---|
| central unit + eight counts minus bond-count kernel + eight counts | linear | random | q2 | +0.025 [-0.003, +0.053] |
| central unit + eight counts minus bond-count kernel + eight counts | linear | group | q2 | +0.073 [+0.004, +0.128] |
| central unit + eight counts minus ECFP4 of the repeat unit | linear | random | q2 | -0.073 [-0.107, -0.039] |
| central unit + eight counts minus ECFP4 of the repeat unit | linear | group | q2 | -0.023 [-0.121, +0.035] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | linear | random | q2 | +0.008 [-0.015, +0.031] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | linear | group | q2 | +0.046 [-0.007, +0.095] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | linear | random | q2 | -0.044 [-0.073, -0.014] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | linear | group | q2 | +0.052 [-0.059, +0.129] |
| central unit + eight counts minus bond-count kernel + eight counts | all | random | q2 | +0.040 [+0.009, +0.069] |
| central unit + eight counts minus bond-count kernel + eight counts | all | group | q2 | +0.038 [-0.054, +0.110] |
| central unit + eight counts minus bond-count kernel + eight counts | all | ladders held out | RMSE | +0.255 [-0.097, +0.584] |
| central unit + eight counts minus ECFP4 of the repeat unit | all | random | q2 | -0.065 [-0.094, -0.037] |
| central unit + eight counts minus ECFP4 of the repeat unit | all | group | q2 | -0.009 [-0.077, +0.037] |
| central unit + eight counts minus ECFP4 of the repeat unit | all | ladders held out | RMSE | -0.151 [-0.360, +0.043] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | all | random | q2 | +0.011 [-0.013, +0.034] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | all | group | q2 | +0.031 [-0.022, +0.074] |
| central unit + eight counts + side groups minus bond-count kernel + eight counts + side groups | all | ladders held out | RMSE | -0.097 [-0.333, +0.123] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | all | random | q2 | -0.043 [-0.067, -0.018] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | all | group | q2 | +0.052 [-0.033, +0.120] |
| central unit + eight counts + side groups minus ECFP4 of the repeat unit | all | ladders held out | RMSE | -0.227 [-0.389, -0.083] |
