# Melting points of molecular crystals (plan item M8)

665 crystals of `examples/melting_points` (ONS melting points matched to COD structures by the perceived molecules); melting point -142 to 438 C, spread 82.2 C. Five folds repeated three times; the window width or kernel width and the ridge chosen by the evidence on the training block.

| Kernel | Needs the crystal | q², mean of 3 repeats | RMSE, C | Settings chosen (folds) |
|---|---|---:|---:|---|
| ECFP4 fingerprint | no | +0.616 | 50.9 | Tanimoto, ridge 0.1 (15) |
| eight counts | no | +0.704 | 44.7 | width 2, ridge 0.1 (13), width 4, ridge 0.01 (2) |
| inorganic, periodic | yes | +0.280 | 69.7 | sigma 0.07, ridge 1 (6), sigma 0.1, ridge 1 (9) |
| molecular, both | yes | +0.631 | 49.9 | sigma 0.75 with 0.05, ridge 0.1 (4), sigma 0.75 with 0.1, ridge 0.1 (4), sigma 0.75 with 0.15, ridge 0.1 (7) |
| molecular, molecule only | yes | +0.654 | 48.3 | sigma 0.05, ridge 0.1 (8), sigma 0.1, ridge 0.1 (4), sigma 0.15, ridge 0.1 (3) |
| molecular, periodic | yes | +0.285 | 69.4 | sigma 0.25, ridge 1 (1), sigma 0.35, ridge 1 (7), sigma 0.75, ridge 1 (7) |

| Difference of q² | Mean [95 % interval] | Interval above zero |
|---|---|---|
| molecular, periodic minus inorganic, periodic | +0.009 [-0.052, +0.070] | no |
| molecular, both minus molecular, molecule only | -0.022 [-0.041, -0.002] | no |
| molecular, periodic minus molecular, molecule only | -0.374 [-0.430, -0.321] | no |
| molecular, both minus ECFP4 fingerprint | +0.030 [-0.014, +0.073] | no |
| molecular, both minus eight counts | -0.052 [-0.088, -0.015] | no |

Gate (molecular, periodic minus inorganic, periodic, interval above zero): not met.
