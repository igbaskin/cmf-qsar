# Hop barriers from density functional theory (nebDFT2k)

Data: Dembitskiy et al., npj Comput. Mater. 11, 131 (2025), https://github.com/AIRI-Institute/LiTraj (MIT licence). 1652 hops in 852 structures are used, of 1681 hops in 867 structures of the set; the others were left out when the set was prepared (8 hops: length differs from the trajectory; 21 hops: structure left out).
The DFT barriers have the mean 0.813 eV and the standard deviation 0.793 eV. The bond-valence barriers of the same hops have the mean 1.256 eV and the standard deviation 1.591 eV, and correlate with the DFT barriers with r = 0.815.

Published split: 1436 hops to fit and 216 to test, 0 structures in both parts. Folds: 10 folds that hold out whole structures, over all hops.

An interval is a 95 % interval of the mean absolute error (MAE) from a bootstrap that draws whole structures.

| Model | Published test part: MAE, eV | RMSE, eV | R2 | Folds: MAE, eV | RMSE, eV | R2 |
|---|---|---|---|---|---|---|
| random forest, invariant field about the midpoint | 0.212 [0.175, 0.259] | 0.310 | 0.86 | 0.228 [0.213, 0.245] | 0.359 | 0.80 |
| random forest, invariant field about the midpoint, with the bond-valence barrier | 0.203 [0.163, 0.253] | 0.313 | 0.85 | 0.222 [0.207, 0.237] | 0.348 | 0.81 |
| kernel ridge, fields in the frame of the hop (item BT7), with the bond-valence barrier | 0.225 [0.187, 0.273] | 0.335 | 0.83 | 0.236 [0.220, 0.251] | 0.356 | 0.80 |
| random forest, hop length and bottleneck, with the bond-valence barrier | 0.242 [0.200, 0.292] | 0.361 | 0.81 | 0.271 [0.252, 0.291] | 0.410 | 0.73 |
| kernel ridge, invariant field about the midpoint, with the bond-valence barrier | 0.251 [0.210, 0.297] | 0.368 | 0.80 | 0.242 [0.228, 0.258] | 0.354 | 0.80 |
| kernel ridge, hop length and bottleneck, with the bond-valence barrier | 0.264 [0.223, 0.313] | 0.385 | 0.78 | 0.275 [0.255, 0.295] | 0.418 | 0.72 |
| random forest, counts about the midpoint, with the bond-valence barrier | 0.256 [0.210, 0.315] | 0.394 | 0.77 | 0.263 [0.243, 0.282] | 0.405 | 0.74 |
| kernel ridge on the bond-valence barrier alone | 0.300 [0.253, 0.358] | 0.435 | 0.72 | 0.302 [0.281, 0.322] | 0.443 | 0.69 |
| kernel ridge, fields in the frame of the hop (item BT7) | 0.294 [0.242, 0.353] | 0.460 | 0.69 | 0.288 [0.270, 0.307] | 0.439 | 0.69 |
| kernel ridge, invariant field about the midpoint | 0.326 [0.276, 0.389] | 0.486 | 0.65 | 0.289 [0.271, 0.307] | 0.428 | 0.71 |
| random forest, counts about the midpoint | 0.315 [0.256, 0.382] | 0.501 | 0.63 | 0.320 [0.293, 0.346] | 0.530 | 0.55 |
| random forest on the bond-valence barrier alone | 0.369 [0.314, 0.430] | 0.520 | 0.60 | 0.365 [0.343, 0.387] | 0.523 | 0.56 |
| kernel ridge, hop length and bottleneck | 0.352 [0.292, 0.421] | 0.557 | 0.54 | 0.362 [0.336, 0.388] | 0.564 | 0.49 |
| kernel ridge, counts about the midpoint | 0.396 [0.332, 0.469] | 0.564 | 0.53 | 0.395 [0.370, 0.421] | 0.594 | 0.44 |
| random forest, hop length and bottleneck | 0.388 [0.323, 0.465] | 0.601 | 0.46 | 0.377 [0.349, 0.404] | 0.591 | 0.44 |
| kernel ridge, the two end sites | 0.508 [0.421, 0.607] | 0.769 | 0.12 | 0.493 [0.462, 0.526] | 0.723 | 0.17 |
| the mean of the fit | 0.546 [0.460, 0.645] | 0.824 | -0.01 | 0.557 [0.527, 0.589] | 0.793 | -0.00 |
| the bond-valence barrier taken as the prediction | 0.688 [0.561, 0.835] | 1.173 | -1.04 | 0.606 [0.554, 0.663] | 1.140 | -1.07 |

## The published models on the test part (Table 3 of the paper)

| Model of the paper | MAE, eV | RMSE, eV | R2 |
|---|---|---|---|
| RF-tl | 0.25 | 0.38 | 0.80 |
| KR-tl | 0.25 | 0.38 | 0.80 |
| XGBRF-tl | 0.25 | 0.38 | 0.80 |
| KR | 0.27 | 0.41 | 0.76 |
| SVR-tl | 0.26 | 0.42 | 0.75 |
| M3GNet-tl | 0.28 | 0.44 | 0.73 |
| SVR | 0.28 | 0.45 | 0.72 |
| KNN-tl | 0.30 | 0.47 | 0.69 |
| RF | 0.29 | 0.48 | 0.68 |
| KNN | 0.31 | 0.50 | 0.65 |
| XGBRF | 0.30 | 0.50 | 0.65 |
| M3GNet | 0.40 | 0.64 | 0.43 |

"-tl" marks a model of the paper that is given the bond-valence barrier. Machine-learned potentials that run the elastic band themselves reach 0.07 to 0.08 eV without transition metals, 0.20 to 0.21 eV with them, 0.10 to 0.11 eV after fine-tuning (mean absolute error, the text of the paper).

## Differences in the mean absolute error, folds

A 95 % interval from a bootstrap that draws whole structures.

| First model minus second model | Difference, eV |
|---|---|
| kernel ridge, fields in the frame of the hop (item BT7) minus kernel ridge, invariant field about the midpoint | -0.001 [-0.016, +0.013] |
| kernel ridge, fields in the frame of the hop (item BT7) minus kernel ridge, hop length and bottleneck | -0.074 [-0.095, -0.052] |
| kernel ridge, fields in the frame of the hop (item BT7), with the bond-valence barrier minus kernel ridge, fields in the frame of the hop (item BT7) | -0.052 [-0.066, -0.039] |
| kernel ridge, invariant field about the midpoint, with the bond-valence barrier minus kernel ridge, invariant field about the midpoint | -0.047 [-0.060, -0.034] |
| kernel ridge, fields in the frame of the hop (item BT7), with the bond-valence barrier minus kernel ridge on the bond-valence barrier alone | -0.066 [-0.081, -0.052] |
| kernel ridge, invariant field about the midpoint minus random forest, invariant field about the midpoint | +0.061 [+0.047, +0.075] |
| kernel ridge, fields in the frame of the hop (item BT7), with the bond-valence barrier minus random forest, invariant field about the midpoint, with the bond-valence barrier | +0.014 [+0.004, +0.024] |
| random forest, invariant field about the midpoint, with the bond-valence barrier minus random forest, invariant field about the midpoint | -0.006 [-0.012, -0.001] |
| random forest, invariant field about the midpoint minus kernel ridge, fields in the frame of the hop (item BT7) | -0.060 [-0.075, -0.046] |
| random forest, invariant field about the midpoint minus kernel ridge on the bond-valence barrier alone | -0.074 [-0.090, -0.057] |
| kernel ridge, fields in the frame of the hop (item BT7) minus kernel ridge on the bond-valence barrier alone | -0.014 [-0.033, +0.006] |

## What the kernel models chose (folds)

- kernel ridge on the bond-valence barrier alone: Gaussian 0.25 (9), Gaussian 1 (1)
- kernel ridge, hop length and bottleneck: Gaussian 16 (10)
- kernel ridge, hop length and bottleneck, with the bond-valence barrier: weight 0.5, Gaussian 0.25 (6), weight 2, Gaussian 1 (3), weight 2, Gaussian 0.25 (1)
- kernel ridge, counts about the midpoint: linear (10)
- kernel ridge, the two end sites: Gaussian 1 (4), Gaussian 0.25 (4), linear (2)
- kernel ridge, invariant field about the midpoint: Gaussian 0.25 (9), Gaussian 1 (1)
- kernel ridge, invariant field about the midpoint, with the bond-valence barrier: weight 0.5, Gaussian 0.25 (8), weight 0.5, Gaussian 1 (1), weight 2, Gaussian 0.25 (1)
- kernel ridge, fields in the frame of the hop (item BT7): Gaussian 1 (9), Gaussian 0.25 (1)
- kernel ridge, fields in the frame of the hop (item BT7), with the bond-valence barrier: weight 0.5, Gaussian 0.25 (9), weight 2, Gaussian 0.25 (1)

The ridge of a final fit was at an end of its grid in 5 % of the fits.

