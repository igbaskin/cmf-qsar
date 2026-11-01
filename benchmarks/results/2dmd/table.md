# A defect series on one host lattice: the periodic overlap with the host symmetry (item M14)

2DMD (Huang et al. 2023), the low-density sets: one to three point defects on an 8 x 8 supercell of MoS2 or WSe2, formation energy of the cell in eV from DFT. The structures are the unrelaxed ideal supercell, so every cell stands on one lattice exactly. Five random folds; the setting (attenuation or distance width, and the bandwidth of the Gaussian layer) and the ridge chosen by five-fold cross-validation inside each training part. MAE per site is the error of the cell divided by its number of defects, which is the quantity the literature on this set reports.

## MoS2: 2000 cells

Of the 2000 cells, 2000 have an arrangement that no cell of their training fold shares under the host symmetry ("unseen"); the rest are, to a symmetry-averaged kernel, structures the model was trained on, and their error is the reproducibility of the DFT labels rather than a prediction.

| kernel | R2 | MAE, eV per cell | MAE, meV per site | MAE on unseen arrangements, meV per site | R2 on unseen | settings chosen |
|---|---:|---:|---:|---:|---:|---|
| fixed | +0.993 | 0.1759 | 60.2 | 60.2 | +0.993 | alpha 0.5, gauss 0.25 |
| counts | +0.993 | 0.1209 | 41.1 | 41.1 | +0.993 | alpha 1, gauss 0.25, alpha 2, gauss 0.25 |
| arrangement | +1.000 | 0.0054 | 1.8 | 1.8 | +1.000 | alpha 1, gamma 0.25, cosine, alpha 2, gamma 0.25 |
| autocorrelation | +1.000 | 0.0213 | 8.3 | 8.3 | +1.000 | sigma 0.35, gauss 0.25, sigma 0.75, gauss 0.25, sigma 0.75, gauss 0.5 |

Against the autocorrelation, paired over the cells (bootstrap, 95 %): the difference in R2 and in MAE per site.

| kernel | R2 difference | MAE per site difference, meV |
|---|---:|---:|
| fixed | -0.007 [-0.008, -0.005] | +51.9 [+48.3, +55.7] |
| counts | -0.006 [-0.008, -0.005] | +32.8 [+28.7, +37.4] |
| arrangement | +0.000 [+0.000, +0.000] | -6.5 [-7.7, -5.3] |

**From one and two defects to three.** Trained on the cells with one or two defects, predicting those with three.

| kernel | cells trained on | R2 | MAE, eV per cell | MAE, meV per site | setting |
|---|---:|---:|---:|---:|---|
| fixed | 131 | +0.832 | 1.2377 | 412.6 | alpha 0.5, gauss 0.5, lam 1e-08 |
| counts | 131 | +0.993 | 0.1241 | 41.4 | alpha 2, linear, lam 0.001 |
| arrangement | 131 | +1.000 | 0.0130 | 4.3 | alpha 1, gamma 0.25, cosine, lam 1e-08 |
| autocorrelation | 131 | -0.015 | 3.6664 | 1222.1 | sigma 0.35, gauss 0.5, lam 0.001 |

**What each kind of defect costs** (the per-defect contributions of the symmetry-averaged linear model, decomposition rule 3; 400 cells, alpha 2, training R2 +0.994). A vacancy is one row (the species removed); a substitution is two rows on one site, the species removed and the species added, whose sum is its cost. The spread is over the cells and should be zero: under the host symmetry a defect's contribution depends on its kind only.

| row | cells | contribution, eV | spread |
|---|---:|---:|---:|
| +Se | 356 | -2.280 | 0.0000 |
| +W | 160 | -6.785 | 0.0000 |
| -Mo | 331 | +6.978 | 0.0000 |
| -S | 734 | +2.594 | 0.0000 |

## MoS2: 5933 cells

Settings searched: attenuation alpha in {1, 2}; bandwidth of the Gaussian layer gamma in {0.125, 0.25} times the median squared distance; autocorrelation width sigma in {0.35, 0.75} A; ridge in {1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1} times the mean self-kernel.

Of the 5933 cells, 5933 have an arrangement that no cell of their training fold shares under the host symmetry ("unseen"); the rest are, to a symmetry-averaged kernel, structures the model was trained on, and their error is the reproducibility of the DFT labels rather than a prediction.

| kernel | R2 | MAE, eV per cell | MAE, meV per site | MAE on unseen arrangements, meV per site | R2 on unseen | settings chosen |
|---|---:|---:|---:|---:|---:|---|
| fixed | +0.995 | 0.1250 | 41.9 | 41.9 | +0.995 | alpha 1, gauss 0.125 |
| counts | +0.993 | 0.1223 | 41.0 | 41.0 | +0.993 | alpha 2, gauss 0.125 |
| arrangement | +1.000 | 0.0058 | 1.9 | 1.9 | +1.000 | alpha 1, gamma 0.125, alpha 1, gamma 0.125, cosine, alpha 2, gamma 0.25 |
| autocorrelation | +1.000 | 0.0084 | 3.0 | 3.0 | +1.000 | sigma 0.75, gauss 0.125 |

Against the autocorrelation, paired over the cells (bootstrap, 95 %): the difference in R2 and in MAE per site.

| kernel | R2 difference | MAE per site difference, meV |
|---|---:|---:|
| fixed | -0.005 [-0.006, -0.005] | +39.0 [+37.0, +41.1] |
| counts | -0.007 [-0.008, -0.006] | +38.0 [+35.6, +40.5] |
| arrangement | +0.000 [+0.000, +0.000] | -1.0 [-1.2, -0.8] |

**From one and two defects to three.** Trained on the cells with one or two defects, predicting those with three.

| kernel | cells trained on | R2 | MAE, eV per cell | MAE, meV per site | setting |
|---|---:|---:|---:|---:|---|
| fixed | 131 | +0.622 | 1.8418 | 613.9 | alpha 1, gauss 0.25, lam 1e-08 |
| counts | 131 | +0.993 | 0.1235 | 41.2 | alpha 2, linear, lam 0.001 |
| arrangement | 131 | +1.000 | 0.0141 | 4.7 | alpha 1, gamma 0.125, cosine, lam 1e-08 |
| autocorrelation | 131 | -0.108 | 3.8781 | 1292.7 | sigma 0.35, gauss 0.25, lam 0.001 |

**What each kind of defect costs** (the per-defect contributions of the symmetry-averaged linear model, decomposition rule 3; 400 cells, alpha 2, training R2 +0.994). A vacancy is one row (the species removed); a substitution is two rows on one site, the species removed and the species added, whose sum is its cost. The spread is over the cells and should be zero: under the host symmetry a defect's contribution depends on its kind only.

| row | cells | contribution, eV | spread |
|---|---:|---:|---:|
| +Se | 356 | -2.280 | 0.0000 |
| +W | 160 | -6.785 | 0.0000 |
| -Mo | 331 | +6.978 | 0.0000 |
| -S | 734 | +2.594 | 0.0000 |

## WSe2: 2000 cells

Of the 2000 cells, 2000 have an arrangement that no cell of their training fold shares under the host symmetry ("unseen"); the rest are, to a symmetry-averaged kernel, structures the model was trained on, and their error is the reproducibility of the DFT labels rather than a prediction.

| kernel | R2 | MAE, eV per cell | MAE, meV per site | MAE on unseen arrangements, meV per site | R2 on unseen | settings chosen |
|---|---:|---:|---:|---:|---:|---|
| fixed | +0.993 | 0.1680 | 57.4 | 57.4 | +0.993 | alpha 0.5, gauss 0.25 |
| counts | +0.994 | 0.1227 | 41.7 | 41.7 | +0.994 | alpha 0.5, gauss 0.25, alpha 2, gauss 0.25 |
| arrangement | +1.000 | 0.0150 | 5.2 | 5.2 | +1.000 | alpha 1, gamma 0.25, alpha 2, gamma 0.5 |
| autocorrelation | +0.999 | 0.0444 | 15.7 | 15.7 | +0.999 | sigma 0.35, gauss 0.25, sigma 0.5, gauss 0.25, sigma 0.5, linear, sigma 0.75, gauss 0.25 |

Against the autocorrelation, paired over the cells (bootstrap, 95 %): the difference in R2 and in MAE per site.

| kernel | R2 difference | MAE per site difference, meV |
|---|---:|---:|
| fixed | -0.006 [-0.007, -0.004] | +41.8 [+38.2, +45.3] |
| counts | -0.005 [-0.007, -0.004] | +26.0 [+22.2, +29.7] |
| arrangement | +0.001 [+0.000, +0.002] | -10.5 [-12.1, -9.1] |

**From one and two defects to three.** Trained on the cells with one or two defects, predicting those with three.

| kernel | cells trained on | R2 | MAE, eV per cell | MAE, meV per site | setting |
|---|---:|---:|---:|---:|---|
| fixed | 131 | +0.798 | 1.2700 | 423.3 | alpha 0.5, gauss 0.25, lam 1e-08 |
| counts | 131 | +0.993 | 0.1147 | 38.2 | alpha 2, linear, lam 0.001 |
| arrangement | 131 | +0.996 | 0.1633 | 54.4 | alpha 0.5, gamma 0.25, lam 0.0001 |
| autocorrelation | 131 | +0.060 | 3.2883 | 1096.1 | sigma 0.35, gauss 0.25, lam 0.001 |

**What each kind of defect costs** (the per-defect contributions of the symmetry-averaged linear model, decomposition rule 3; 400 cells, alpha 2, training R2 +0.995). A vacancy is one row (the species removed); a substitution is two rows on one site, the species removed and the species added, whose sum is its cost. The spread is over the cells and should be zero: under the host symmetry a defect's contribution depends on its kind only.

| row | cells | contribution, eV | spread |
|---|---:|---:|---:|
| +Mo | 160 | -5.506 | 0.0000 |
| +S | 379 | -3.027 | 0.0000 |
| -Se | 734 | +2.619 | 0.0000 |
| -W | 331 | +5.091 | 0.0000 |

## WSe2: 5933 cells

Settings searched: attenuation alpha in {1, 2}; bandwidth of the Gaussian layer gamma in {0.125, 0.25} times the median squared distance; autocorrelation width sigma in {0.35, 0.75} A; ridge in {1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1} times the mean self-kernel.

Of the 5933 cells, 5933 have an arrangement that no cell of their training fold shares under the host symmetry ("unseen"); the rest are, to a symmetry-averaged kernel, structures the model was trained on, and their error is the reproducibility of the DFT labels rather than a prediction.

| kernel | R2 | MAE, eV per cell | MAE, meV per site | MAE on unseen arrangements, meV per site | R2 on unseen | settings chosen |
|---|---:|---:|---:|---:|---:|---|
| fixed | +0.995 | 0.1257 | 42.2 | 42.2 | +0.995 | alpha 1, gauss 0.125 |
| counts | +0.994 | 0.1225 | 41.1 | 41.1 | +0.994 | alpha 1, gauss 0.125, alpha 2, gauss 0.125 |
| arrangement | +1.000 | 0.0132 | 4.4 | 4.4 | +1.000 | alpha 1, gamma 0.25, alpha 2, gamma 0.125, alpha 2, gamma 0.25 |
| autocorrelation | +1.000 | 0.0173 | 6.0 | 6.0 | +1.000 | sigma 0.35, gauss 0.125, sigma 0.75, gauss 0.125 |

Against the autocorrelation, paired over the cells (bootstrap, 95 %): the difference in R2 and in MAE per site.

| kernel | R2 difference | MAE per site difference, meV |
|---|---:|---:|
| fixed | -0.005 [-0.006, -0.004] | +36.2 [+34.5, +38.1] |
| counts | -0.006 [-0.007, -0.005] | +35.1 [+33.0, +37.2] |
| arrangement | +0.000 [+0.000, +0.000] | -1.5 [-1.9, -1.3] |

**From one and two defects to three.** Trained on the cells with one or two defects, predicting those with three.

| kernel | cells trained on | R2 | MAE, eV per cell | MAE, meV per site | setting |
|---|---:|---:|---:|---:|---|
| fixed | 131 | +0.562 | 1.8927 | 630.9 | alpha 1, gauss 0.25, lam 1e-08 |
| counts | 131 | +0.993 | 0.1148 | 38.3 | alpha 2, linear, lam 0.001 |
| arrangement | 131 | +0.998 | 0.1130 | 37.7 | alpha 1, gamma 0.125, lam 1e-05 |
| autocorrelation | 131 | +0.071 | 3.2989 | 1099.6 | sigma 0.35, gauss 0.25, lam 0.001 |

**What each kind of defect costs** (the per-defect contributions of the symmetry-averaged linear model, decomposition rule 3; 400 cells, alpha 2, training R2 +0.995). A vacancy is one row (the species removed); a substitution is two rows on one site, the species removed and the species added, whose sum is its cost. The spread is over the cells and should be zero: under the host symmetry a defect's contribution depends on its kind only.

| row | cells | contribution, eV | spread |
|---|---:|---:|---:|
| +Mo | 160 | -5.506 | 0.0000 |
| +S | 379 | -3.027 | 0.0000 |
| -Se | 734 | +2.619 | 0.0000 |
| -W | 331 | +5.091 | 0.0000 |

## Context: the published errors on this set

Kazeev et al. 2023, their own test split (not these folds), MAE of the formation energy per site in meV: MoS2 -- sparse graph network 2.6, gradient boosting on descriptors 11.8, whole-structure graph networks 34 to 48; WSe2 -- 4.8, 14.7 and 31 to 55.
