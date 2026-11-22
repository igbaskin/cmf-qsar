# The co-field as an atom feature of a network that sums over the atoms (ND14c)

Written by `benchmarks/run_device.py --report`. The splits are those of plan items ND1 and ND2: random, 70 % training, 15 % validation, 15 % test. Every score is on the test molecules. "Atom network" is the network of `cmfqsar/device.py`: one network on every atom, a sum over the atoms, a second network; it sees no bonds. "Device" is the QSPR Neural Device, which reads the atoms, the bonds and the paths of two bonds. The co-field is that of the ridge regression of the first row; a training molecule reads the co-field of a model fitted without its fold, unless the row says "not out-of-fold".

## COMPAS-1D: the HOMO-LUMO gap, 5 splits

| Model | Mean absolute error, eV (mean ± sd over the splits) | Test R2 (mean ± sd) | Splits |
|---|---|---|---|
| ridge regression on the fields | 0.114 ± 0.003 | +0.853 ± 0.032 | 5 |
| atom network: field weights of the atom | 0.091 ± 0.003 | +0.906 ± 0.011 | 5 |
| atom network: positions | 0.078 ± 0.003 | +0.931 ± 0.009 | 5 |
| atom network: co-field | 0.083 ± 0.002 | +0.925 ± 0.005 | 5 |
| atom network: co-field, not out-of-fold | 0.074 ± 0.002 | +0.938 ± 0.005 | 5 |
| atom network: co-field and positions | 0.085 ± 0.002 | +0.922 ± 0.002 | 5 |
| device, one-shot: with the co-field | 0.127 ± 0.003 | +0.834 ± 0.019 | 5 |
| device, one-shot: with positions | 0.122 ± 0.004 | +0.845 ± 0.007 | 5 |
| ND1: the device without message passing | 0.183 ± 0.004 | +0.633 ± 0.014 | 5 |
| ND1: the device with message passing | 0.035 ± 0.002 | +0.982 ± 0.009 | 5 |
| ND1: the kernel model on every training molecule | 0.111 ± 0.003 | +0.869 ± 0.016 | 5 |
| ND1: multilayer network on the CMF vector joined with the collectors | 0.029 ± 0.002 | +0.987 ± 0.007 | 5 |

### Paired comparisons on the same test molecules

The difference is the mean absolute error of the first model minus that of the second, in eV; a negative number means that the first model is better. The interval is the 95 % bootstrap interval over the test molecules.

| No. | First model | Second model | Difference [95 % interval] | Splits | What it says |
|---|---|---|---|---|---|
| 1 | atom network: co-field | ridge regression on the fields | -0.0303 [-0.0333, -0.0273] | 5 | the gate: the atom network against the ridge regression on the same fields |
| 2 | atom network: co-field | atom network: field weights of the atom | -0.0091 [-0.0116, -0.0068] | 5 | what the co-field adds to the atom network |
| 3 | atom network: co-field | atom network: positions | +0.0045 [+0.0024, +0.0064] | 5 | the co-field against the bare coordinates |
| 4 | atom network: co-field and positions | atom network: positions | +0.0066 [+0.0047, +0.0085] | 5 | what the co-field adds to the coordinates |
| 5 | atom network: co-field, not out-of-fold | atom network: co-field | -0.0083 [-0.0102, -0.0065] | 5 | what the out-of-fold reading is worth (positive: it helps) |
| 6 | device, one-shot: with the co-field | ridge regression on the fields | +0.0135 [+0.0105, +0.0167] | 5 | the gate with the device without message passing |
| 7 | device, one-shot: with the co-field | device, one-shot | -0.0553 [-0.0594, -0.0512] | 5 | what the co-field adds to the device without message passing |
| 8 | device, one-shot: with the co-field | device, one-shot: with positions | +0.0050 [+0.0025, +0.0075] | 5 | the co-field against the bare coordinates, device without message passing |

### Checks of the co-field values

| Split | Largest difference between the prediction and the sum over the atoms, eV | Training RMSE of the fitted model, eV | Training RMSE out-of-fold, eV | Root mean square change of a co-field value between the full model and the fold model, as a share of the spread of the values |
|---|---|---|---|---|
| 0 | 1.1e-11 | 0.1363 | 0.1589 | 0.228 |
| 1 | 1.2e-11 | 0.1322 | 0.1482 | 0.175 |
| 2 | 1.4e-11 | 0.1335 | 0.1481 | 0.118 |
| 3 | 2.3e-11 | 0.1312 | 0.1453 | 0.117 |
| 4 | 1.1e-11 | 0.1334 | 0.1497 | 0.135 |

### Training time

| Model | Mean time for one split, minutes |
|---|---|
| atom network: field weights of the atom | 15.7 |
| atom network: positions | 9.4 |
| atom network: co-field | 6.7 |
| atom network: co-field, not out-of-fold | 9.7 |
| atom network: co-field and positions | 7.2 |
| device, one-shot: with the co-field | 267.0 |
| device, one-shot: with positions | 409.4 |

### Atom sensors of the device

| Model | Atom sensors in use | Of them read from the file | Offered in the file |
|---|---|---|---|
| device, one-shot: with the co-field | 11 | 9 | 9 |
| device, one-shot: with positions | 5 | 3 | 3 |

## The eight Sutherland series: test R2, mean over the splits

| Series | Splits | ridge regression on the fields | atom network: field weights of the atom | atom network: positions | atom network: co-field | atom network: co-field, not out-of-fold | atom network: co-field and positions |
|---|---|---|---|---|---|---|---|
| ACE | 10 | +0.605 | +0.699 | +0.639 | +0.657 | +0.608 | +0.608 |
| AChE | 10 | +0.321 | -0.079 | +0.161 | +0.065 | +0.293 | +0.156 |
| BZR | 10 | +0.122 | +0.408 | +0.420 | +0.321 | +0.452 | +0.366 |
| COX2 | 10 | +0.480 | +0.351 | +0.482 | +0.464 | +0.475 | +0.444 |
| DHFR | 10 | +0.635 | +0.553 | +0.610 | +0.578 | +0.615 | +0.610 |
| GPB | 10 | +0.462 | -0.377 | +0.134 | -0.008 | +0.320 | +0.130 |
| THER | 10 | +0.247 | +0.092 | +0.336 | +0.315 | +0.313 | +0.270 |
| THR | 10 | +0.430 | +0.098 | +0.559 | +0.345 | +0.480 | +0.376 |
| Mean over the series | | +0.413 | +0.218 | +0.418 | +0.342 | +0.444 | +0.370 |

### Paired comparisons, Sutherland series

The difference of the absolute errors in the units of the activity (pIC50 or pKi), first model minus second; negative means that the first model is better.

| Series | 1: atom network: co-field against ridge regression on the fields | 2: atom network: co-field against atom network: field weights of the atom | 3: atom network: co-field against atom network: positions | 4: atom network: co-field and positions against atom network: positions | 5: atom network: co-field, not out-of-fold against atom network: co-field |
|---|---|---|---|---|---|
| ACE | -0.017 [-0.167, +0.125] (includes zero) | +0.042 [-0.097, +0.195] (includes zero) | +0.031 [-0.122, +0.177] (includes zero) | +0.127 [-0.002, +0.259] (includes zero) | +0.009 [-0.101, +0.119] (includes zero) |
| AChE | +0.171 [+0.081, +0.264] | -0.035 [-0.135, +0.068] (includes zero) | +0.121 [+0.032, +0.212] | +0.061 [-0.021, +0.144] (includes zero) | -0.142 [-0.208, -0.076] |
| BZR | -0.035 [-0.114, +0.042] (includes zero) | +0.019 [-0.036, +0.076] (includes zero) | +0.032 [-0.013, +0.082] (includes zero) | +0.011 [-0.035, +0.050] (includes zero) | -0.039 [-0.094, +0.011] (includes zero) |
| COX2 | -0.015 [-0.061, +0.034] (includes zero) | -0.084 [-0.128, -0.041] | -0.001 [-0.038, +0.037] (includes zero) | +0.011 [-0.024, +0.048] (includes zero) | -0.002 [-0.029, +0.026] (includes zero) |
| DHFR | +0.044 [+0.005, +0.081] | -0.030 [-0.081, +0.024] (includes zero) | +0.012 [-0.035, +0.057] (includes zero) | -0.007 [-0.052, +0.037] (includes zero) | -0.044 [-0.073, -0.015] |
| GPB | +0.251 [+0.128, +0.378] | -0.001 [-0.193, +0.180] (includes zero) | +0.093 [-0.060, +0.244] (includes zero) | +0.038 [-0.068, +0.140] (includes zero) | -0.168 [-0.264, -0.080] |
| THER | -0.114 [-0.327, +0.109] (includes zero) | -0.012 [-0.256, +0.235] (includes zero) | -0.026 [-0.203, +0.145] (includes zero) | +0.064 [-0.110, +0.241] (includes zero) | +0.047 [-0.136, +0.218] (includes zero) |
| THR | +0.094 [-0.020, +0.200] (includes zero) | -0.043 [-0.157, +0.061] (includes zero) | +0.166 [+0.054, +0.285] | +0.115 [+0.005, +0.228] | -0.111 [-0.223, -0.000] |
