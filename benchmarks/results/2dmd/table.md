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

## A pair expansion as the reference

One energy for each kind of defect and one for each pair of kinds in each shell of host-site distances up to the cut-off, fitted by linear regression with the ridge chosen inside each fold (a cluster expansion cut at pairs), with a fitted constant or with the constant at zero, the formation energy of a cell without defects. The cells and folds are those of the kernel models above, so the differences are paired over the cells (bootstrap, 95 %, MAE per site in meV, negative when the pair expansion has the smaller error).

### MoS2 | 2000 cells

| model | columns | MAE per site, meV | R2 | from one and two defects to three, meV | R2 there |
|---|---:|---:|---:|---:|---:|
| fixed kernel (stored) | | 60.2 | +0.9931 | 412.6 | +0.832 |
| counts kernel (stored) | | 41.1 | +0.9933 | 41.4 | +0.993 |
| arrangement kernel (stored) | | 1.8 | +1.0000 | 4.3 | +1.000 |
| autocorrelation kernel (stored) | | 8.3 | +0.9997 | 1222.1 | -0.015 |
| pairs within 4 A, with a constant | 21 | 14.1 | +0.9994 | 272.4 | +0.931 |
| pairs within 6 A, with a constant | 35 | 3.5 | +1.0000 | 265.8 | +0.934 |
| pairs within 8 A, with a constant | 52 | 3.0 | +1.0000 | 260.4 | +0.937 |
| pairs within 12 A, with a constant | 96 | 2.4 | +1.0000 | 244.9 | +0.943 |
| pairs within 4 A, constant zero | 21 | 13.8 | +0.9994 | 142.2 | +0.983 |
| pairs within 6 A, constant zero | 35 | 3.1 | +1.0000 | 24.8 | +1.000 |
| pairs within 8 A, constant zero | 52 | 2.6 | +1.0000 | 24.8 | +1.000 |
| pairs within 12 A, constant zero | 96 | 2.0 | +1.0000 | 8.1 | +1.000 |

| pair expansion | minus | random folds | from one and two defects to three |
|---|---|---|---|
| within 4 A, with a constant | arrangement | +12.3 [+11.0, +13.6] | +268.0 [+259.5, +276.3] |
| within 4 A, with a constant | autocorrelation | +5.8 [+4.2, +7.5] | -949.8 [-964.2, -935.8] |
| within 4 A, with a constant | counts | -27.0 [-31.3, -23.0] | +231.0 [+223.0, +239.2] |
| within 6 A, with a constant | arrangement | +1.7 [+1.3, +2.3] | +261.5 [+253.1, +269.6] |
| within 6 A, with a constant | autocorrelation | -4.8 [-5.9, -3.8] | -956.3 [-970.3, -942.6] |
| within 6 A, with a constant | counts | -37.6 [-42.0, -33.6] | +224.5 [+216.6, +232.5] |
| within 8 A, with a constant | arrangement | +1.2 [+0.8, +1.8] | +256.1 [+247.9, +264.1] |
| within 8 A, with a constant | autocorrelation | -5.3 [-6.4, -4.2] | -961.7 [-975.8, -947.9] |
| within 8 A, with a constant | counts | -38.1 [-42.4, -34.1] | +219.1 [+211.2, +227.1] |
| within 12 A, with a constant | arrangement | +0.6 [+0.2, +1.2] | +240.6 [+232.7, +248.2] |
| within 12 A, with a constant | autocorrelation | -5.9 [-7.0, -4.8] | -977.2 [-991.2, -963.1] |
| within 12 A, with a constant | counts | -38.7 [-43.1, -34.7] | +203.5 [+195.9, +211.4] |
| within 4 A, constant zero | arrangement | +11.9 [+10.7, +13.2] | +137.9 [+134.4, +141.4] |
| within 4 A, constant zero | autocorrelation | +5.5 [+3.8, +7.3] | -1080.0 [-1095.0, -1065.2] |
| within 4 A, constant zero | counts | -27.3 [-31.6, -23.3] | +100.8 [+95.8, +105.6] |
| within 6 A, constant zero | arrangement | +1.3 [+1.1, +1.4] | +20.5 [+20.0, +20.9] |
| within 6 A, constant zero | autocorrelation | -5.2 [-6.4, -4.0] | -1197.3 [-1213.1, -1181.5] |
| within 6 A, constant zero | counts | -38.0 [-42.4, -33.9] | -16.6 [-21.2, -12.1] |
| within 8 A, constant zero | arrangement | +0.8 [+0.7, +0.9] | +20.4 [+20.0, +20.9] |
| within 8 A, constant zero | autocorrelation | -5.7 [-6.9, -4.5] | -1197.4 [-1213.1, -1181.6] |
| within 8 A, constant zero | counts | -38.5 [-42.8, -34.5] | -16.6 [-21.2, -12.2] |
| within 12 A, constant zero | arrangement | +0.2 [+0.1, +0.3] | +3.8 [+3.7, +3.9] |
| within 12 A, constant zero | autocorrelation | -6.3 [-7.5, -5.1] | -1214.0 [-1230.0, -1198.0] |
| within 12 A, constant zero | counts | -39.1 [-43.4, -35.1] | -33.3 [-38.0, -28.8] |

### MoS2 | 5933 cells

| model | columns | MAE per site, meV | R2 | from one and two defects to three, meV | R2 there |
|---|---:|---:|---:|---:|---:|
| fixed kernel (stored) | | 41.9 | +0.9946 | 613.9 | +0.622 |
| counts kernel (stored) | | 41.0 | +0.9933 | 41.2 | +0.993 |
| arrangement kernel (stored) | | 1.9 | +0.9999 | 4.7 | +1.000 |
| autocorrelation kernel (stored) | | 3.0 | +0.9999 | 1292.7 | -0.108 |
| pairs within 4 A, with a constant | 21 | 14.7 | +0.9992 | 274.8 | +0.929 |
| pairs within 6 A, with a constant | 35 | 3.4 | +0.9999 | 267.9 | +0.933 |
| pairs within 8 A, with a constant | 52 | 2.9 | +0.9999 | 262.5 | +0.936 |
| pairs within 12 A, with a constant | 96 | 2.2 | +0.9999 | 248.4 | +0.942 |
| pairs within 4 A, constant zero | 21 | 14.7 | +0.9992 | 143.8 | +0.983 |
| pairs within 6 A, constant zero | 35 | 3.4 | +0.9999 | 25.1 | +0.999 |
| pairs within 8 A, constant zero | 52 | 2.8 | +0.9999 | 25.1 | +0.999 |
| pairs within 12 A, constant zero | 96 | 2.2 | +0.9999 | 8.2 | +1.000 |

| pair expansion | minus | random folds | from one and two defects to three |
|---|---|---|---|
| within 4 A, with a constant | arrangement | +12.7 [+12.0, +13.5] | +270.1 [+265.4, +275.0] |
| within 4 A, with a constant | autocorrelation | +11.7 [+11.0, +12.5] | -1017.9 [-1024.7, -1011.2] |
| within 4 A, with a constant | counts | -26.3 [-28.7, -24.1] | +233.7 [+229.1, +238.4] |
| within 6 A, with a constant | arrangement | +1.4 [+1.3, +1.5] | +263.2 [+258.6, +267.8] |
| within 6 A, with a constant | autocorrelation | +0.4 [+0.2, +0.6] | -1024.8 [-1031.4, -1018.3] |
| within 6 A, with a constant | counts | -37.6 [-40.1, -35.2] | +226.8 [+222.3, +231.2] |
| within 8 A, with a constant | arrangement | +0.9 [+0.8, +1.0] | +257.8 [+253.2, +262.4] |
| within 8 A, with a constant | autocorrelation | -0.1 [-0.3, +0.1] | -1030.2 [-1036.8, -1023.7] |
| within 8 A, with a constant | counts | -38.2 [-40.6, -35.8] | +221.3 [+216.9, +225.8] |
| within 12 A, with a constant | arrangement | +0.3 [+0.2, +0.3] | +243.7 [+239.4, +248.2] |
| within 12 A, with a constant | autocorrelation | -0.8 [-1.0, -0.6] | -1044.3 [-1051.2, -1037.6] |
| within 12 A, with a constant | counts | -38.8 [-41.3, -36.4] | +207.3 [+203.0, +211.7] |
| within 4 A, constant zero | arrangement | +12.7 [+12.0, +13.5] | +139.1 [+137.1, +141.1] |
| within 4 A, constant zero | autocorrelation | +11.7 [+11.0, +12.5] | -1148.9 [-1155.9, -1142.2] |
| within 4 A, constant zero | counts | -26.3 [-28.7, -24.1] | +102.6 [+99.8, +105.2] |
| within 6 A, constant zero | arrangement | +1.4 [+1.3, +1.5] | +20.4 [+20.1, +20.6] |
| within 6 A, constant zero | autocorrelation | +0.4 [+0.2, +0.6] | -1267.6 [-1275.0, -1260.4] |
| within 6 A, constant zero | counts | -37.6 [-40.1, -35.2] | -16.1 [-18.6, -13.7] |
| within 8 A, constant zero | arrangement | +0.9 [+0.8, +1.0] | +20.4 [+20.1, +20.6] |
| within 8 A, constant zero | autocorrelation | -0.1 [-0.3, +0.1] | -1267.6 [-1275.1, -1260.4] |
| within 8 A, constant zero | counts | -38.2 [-40.6, -35.8] | -16.1 [-18.6, -13.7] |
| within 12 A, constant zero | arrangement | +0.2 [+0.2, +0.3] | +3.5 [+3.5, +3.6] |
| within 12 A, constant zero | autocorrelation | -0.8 [-1.0, -0.6] | -1284.5 [-1292.0, -1277.1] |
| within 12 A, constant zero | counts | -38.8 [-41.3, -36.4] | -32.9 [-35.5, -30.4] |

### WSe2 | 2000 cells

| model | columns | MAE per site, meV | R2 | from one and two defects to three, meV | R2 there |
|---|---:|---:|---:|---:|---:|
| fixed kernel (stored) | | 57.4 | +0.9933 | 423.3 | +0.798 |
| counts kernel (stored) | | 41.7 | +0.9937 | 38.2 | +0.993 |
| arrangement kernel (stored) | | 5.2 | +0.9998 | 54.4 | +0.996 |
| autocorrelation kernel (stored) | | 15.7 | +0.9989 | 1096.1 | +0.060 |
| pairs within 4 A, with a constant | 21 | 19.1 | +0.9987 | 234.6 | +0.935 |
| pairs within 6 A, with a constant | 35 | 6.6 | +0.9998 | 289.7 | +0.935 |
| pairs within 8 A, with a constant | 52 | 6.5 | +0.9998 | 285.2 | +0.937 |
| pairs within 12 A, with a constant | 87 | 5.9 | +0.9998 | 275.0 | +0.941 |
| pairs within 4 A, constant zero | 21 | 18.8 | +0.9987 | 115.6 | +0.985 |
| pairs within 6 A, constant zero | 35 | 6.3 | +0.9998 | 113.5 | +0.986 |
| pairs within 8 A, constant zero | 52 | 6.2 | +0.9998 | 16.1 | +1.000 |
| pairs within 12 A, constant zero | 87 | 5.6 | +0.9998 | 15.1 | +1.000 |

| pair expansion | minus | random folds | from one and two defects to three |
|---|---|---|---|
| within 4 A, with a constant | arrangement | +13.9 [+12.4, +15.5] | +180.1 [+173.1, +187.3] |
| within 4 A, with a constant | autocorrelation | +3.4 [+1.3, +5.5] | -861.5 [-873.9, -849.3] |
| within 4 A, with a constant | counts | -22.6 [-26.0, -19.4] | +196.3 [+188.6, +204.3] |
| within 6 A, with a constant | arrangement | +1.5 [+1.1, +2.0] | +235.2 [+233.4, +237.1] |
| within 6 A, with a constant | autocorrelation | -9.1 [-10.6, -7.6] | -806.4 [-815.6, -797.1] |
| within 6 A, with a constant | counts | -35.1 [-38.5, -31.7] | +251.4 [+247.2, +255.4] |
| within 8 A, with a constant | arrangement | +1.3 [+1.0, +1.8] | +230.8 [+229.0, +232.7] |
| within 8 A, with a constant | autocorrelation | -9.2 [-10.7, -7.8] | -810.9 [-820.0, -801.5] |
| within 8 A, with a constant | counts | -35.2 [-38.7, -31.8] | +247.0 [+242.8, +251.0] |
| within 12 A, with a constant | arrangement | +0.7 [+0.4, +1.2] | +220.6 [+218.2, +222.8] |
| within 12 A, with a constant | autocorrelation | -9.8 [-11.4, -8.4] | -821.1 [-830.4, -811.7] |
| within 12 A, with a constant | counts | -35.8 [-39.3, -32.4] | +236.8 [+232.5, +241.1] |
| within 4 A, constant zero | arrangement | +13.6 [+12.2, +15.1] | +61.2 [+58.0, +64.5] |
| within 4 A, constant zero | autocorrelation | +3.1 [+1.0, +5.1] | -980.5 [-990.9, -970.3] |
| within 4 A, constant zero | counts | -22.9 [-26.3, -19.8] | +77.4 [+72.8, +81.8] |
| within 6 A, constant zero | arrangement | +1.1 [+0.9, +1.4] | +59.1 [+56.8, +61.5] |
| within 6 A, constant zero | autocorrelation | -9.4 [-11.0, -8.1] | -982.6 [-992.4, -972.6] |
| within 6 A, constant zero | counts | -35.4 [-38.8, -31.9] | +75.3 [+70.5, +79.9] |
| within 8 A, constant zero | arrangement | +1.0 [+0.7, +1.3] | -38.3 [-40.6, -36.3] |
| within 8 A, constant zero | autocorrelation | -9.5 [-11.1, -8.2] | -1080.0 [-1090.3, -1069.9] |
| within 8 A, constant zero | counts | -35.6 [-38.9, -32.1] | -22.2 [-26.2, -18.2] |
| within 12 A, constant zero | arrangement | +0.4 [+0.2, +0.6] | -39.4 [-41.6, -37.2] |
| within 12 A, constant zero | autocorrelation | -10.1 [-11.7, -8.8] | -1081.0 [-1091.3, -1070.8] |
| within 12 A, constant zero | counts | -36.1 [-39.6, -32.7] | -23.2 [-27.2, -19.4] |

### WSe2 | 5933 cells

| model | columns | MAE per site, meV | R2 | from one and two defects to three, meV | R2 there |
|---|---:|---:|---:|---:|---:|
| fixed kernel (stored) | | 42.2 | +0.9949 | 630.9 | +0.562 |
| counts kernel (stored) | | 41.1 | +0.9935 | 38.3 | +0.993 |
| arrangement kernel (stored) | | 4.4 | +0.9999 | 37.7 | +0.998 |
| autocorrelation kernel (stored) | | 6.0 | +0.9998 | 1099.6 | +0.071 |
| pairs within 4 A, with a constant | 21 | 19.3 | +0.9986 | 238.3 | +0.933 |
| pairs within 6 A, with a constant | 35 | 5.9 | +0.9999 | 289.8 | +0.936 |
| pairs within 8 A, with a constant | 52 | 5.7 | +0.9999 | 285.4 | +0.938 |
| pairs within 12 A, with a constant | 87 | 4.9 | +0.9999 | 275.1 | +0.942 |
| pairs within 4 A, constant zero | 21 | 19.1 | +0.9986 | 116.8 | +0.985 |
| pairs within 6 A, constant zero | 35 | 5.7 | +0.9999 | 113.8 | +0.987 |
| pairs within 8 A, constant zero | 52 | 5.5 | +0.9999 | 15.9 | +1.000 |
| pairs within 12 A, constant zero | 87 | 4.7 | +0.9999 | 14.9 | +1.000 |

| pair expansion | minus | random folds | from one and two defects to three |
|---|---|---|---|
| within 4 A, with a constant | arrangement | +14.9 [+13.9, +15.9] | +200.6 [+196.4, +204.8] |
| within 4 A, with a constant | autocorrelation | +13.3 [+12.3, +14.3] | -861.4 [-868.2, -854.4] |
| within 4 A, with a constant | counts | -21.8 [-23.9, -19.8] | +200.0 [+195.5, +204.4] |
| within 6 A, with a constant | arrangement | +1.4 [+1.2, +1.8] | +252.1 [+251.2, +253.1] |
| within 6 A, with a constant | autocorrelation | -0.1 [-0.5, +0.3] | -809.8 [-815.0, -804.9] |
| within 6 A, with a constant | counts | -35.2 [-37.3, -33.1] | +251.5 [+249.2, +253.9] |
| within 8 A, with a constant | arrangement | +1.2 [+0.9, +1.6] | +247.7 [+246.7, +248.6] |
| within 8 A, with a constant | autocorrelation | -0.3 [-0.7, +0.1] | -814.2 [-819.4, -809.4] |
| within 8 A, with a constant | counts | -35.4 [-37.5, -33.4] | +247.1 [+244.7, +249.4] |
| within 12 A, with a constant | arrangement | +0.5 [+0.2, +0.7] | +237.4 [+236.3, +238.6] |
| within 12 A, with a constant | autocorrelation | -1.1 [-1.4, -0.7] | -824.5 [-829.8, -819.4] |
| within 12 A, with a constant | counts | -36.2 [-38.3, -34.1] | +236.8 [+234.5, +239.2] |
| within 4 A, constant zero | arrangement | +14.6 [+13.7, +15.6] | +79.1 [+77.2, +81.0] |
| within 4 A, constant zero | autocorrelation | +13.1 [+12.1, +14.1] | -982.8 [-988.7, -977.0] |
| within 4 A, constant zero | counts | -22.0 [-24.0, -20.1] | +78.6 [+76.0, +81.0] |
| within 6 A, constant zero | arrangement | +1.2 [+1.1, +1.4] | +76.1 [+74.6, +77.5] |
| within 6 A, constant zero | autocorrelation | -0.3 [-0.6, -0.1] | -985.9 [-991.5, -980.5] |
| within 6 A, constant zero | counts | -35.4 [-37.5, -33.4] | +75.5 [+73.0, +78.0] |
| within 8 A, constant zero | arrangement | +1.0 [+0.9, +1.1] | -21.8 [-22.7, -20.9] |
| within 8 A, constant zero | autocorrelation | -0.5 [-0.8, -0.3] | -1083.7 [-1089.3, -1078.5] |
| within 8 A, constant zero | counts | -35.6 [-37.7, -33.6] | -22.4 [-24.7, -20.1] |
| within 12 A, constant zero | arrangement | +0.3 [+0.2, +0.3] | -22.8 [-23.7, -21.9] |
| within 12 A, constant zero | autocorrelation | -1.3 [-1.6, -1.0] | -1084.8 [-1090.3, -1079.4] |
| within 12 A, constant zero | counts | -36.4 [-38.5, -34.3] | -23.4 [-25.7, -21.1] |

## Context: the published errors on this set

Kazeev et al. 2023, their own test split (not these folds), MAE of the formation energy per site in meV: MoS2 -- sparse graph network 2.6, gradient boosting on descriptors 11.8, whole-structure graph networks 34 to 48; WSe2 -- 4.8, 14.7 and 31 to 55.
