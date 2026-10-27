# The field around an atom against a count around an atom (plan item LP2)

The sample is an atom in a molecule. The local descriptor is the molecule's fields under a Gaussian envelope about that atom, turned into the rotation-invariant autocorrelation, so no alignment is needed. The reference of rule 1 is an atom-centred count: the type of the centre and the atom types in distance shells about it, which is what a per-atom increment scheme is. The folds hold out whole molecules, and the interval is bootstrapped over molecules.

## 13C shifts in chloroform (nmrshiftdb2)

5096 measured atoms in 400 molecules; the target runs from 7.00 to 220.50 ppm with a standard deviation of 53.35. 10 folds holding out whole molecules.

| descriptor | envelope width rho | q2 | error | mean absolute error | ridge | atoms per sample | mean kernel between two samples | difference in q2 against the counts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| atom-centred counts (18 atom types in 6 shells) | -- | +0.853 | 20.464 | 14.833 | 0.1 | -- | 0.566 | -- |
| local field | 0.25 | +0.902 | 16.731 | 11.465 | 0.0001 | 28.9 | 0.462 | +0.049 [+0.030, +0.067] |
| local field | 0.5 | +0.962 | 10.404 | 7.196 | 1e-05 | 20.0 | 0.486 | +0.109 [+0.097, +0.123] |
| local field | 1 | +0.973 | 8.824 | 6.077 | 1e-06 | 12.6 | 0.537 | +0.120 [+0.107, +0.133] |
| local field | 2 | +0.969 | 9.460 | 6.584 | 1e-05 | 5.1 | 0.562 | +0.116 [+0.103, +0.130] |

The error is in ppm. A mean kernel close to 1 says the descriptor cannot tell two samples apart, whatever the regression does with it.

* at the module's default width (rho 0.5, local field) it beats the counts: +0.109 [+0.097, +0.123]
* at the best width of the scan (rho 1, local field) it beats the counts: +0.120 [+0.107, +0.133]

## carboxylic acid pKa (Baltruschat and Czodrowski 2020)

912 measured atoms in 912 molecules; the target runs from 1.77 to 6.40 pKa units with a standard deviation of 0.83. 10 folds holding out whole molecules.

| descriptor | envelope width rho | q2 | error | mean absolute error | ridge | atoms per sample | mean kernel between two samples | difference in q2 against the counts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| atom-centred counts (20 atom types in 6 shells) | -- | +0.541 | 0.563 | 0.399 | 0.01 | -- | 0.577 | -- |
| local field | 0.05 | +0.570 | 0.545 | 0.380 | 0.001 | 27.0 | 0.716 | +0.029 [-0.030, +0.086] |
| local difference field | 0.05 | +0.045 | 0.811 | 0.611 | 0.001 | 53.0 | 1.000 | -0.495 [-0.718, -0.356] |
| local field | 0.1 | +0.582 | 0.537 | 0.370 | 0.001 | 24.9 | 0.775 | +0.041 [-0.013, +0.094] |
| local difference field | 0.1 | +0.028 | 0.818 | 0.649 | 0.01 | 48.8 | 1.000 | -0.512 [-0.570, -0.451] |
| local field | 0.25 | +0.553 | 0.555 | 0.390 | 0.001 | 19.3 | 0.877 | +0.013 [-0.052, +0.075] |
| local difference field | 0.25 | +0.014 | 0.824 | 0.656 | 0.1 | 37.7 | 1.000 | -0.526 [-0.579, -0.468] |
| local field | 0.5 | +0.526 | 0.571 | 0.399 | 1e-06 | 13.6 | 0.957 | -0.014 [-0.079, +0.051] |
| local difference field | 0.5 | +0.315 | 0.687 | 0.487 | 1e-06 | 26.3 | 1.000 | -0.226 [-0.383, -0.113] |
| local field | 1 | +0.470 | 0.604 | 0.432 | 1e-05 | 9.4 | 0.990 | -0.070 [-0.126, -0.006] |
| local difference field | 1 | +0.020 | 0.822 | 0.644 | 0.1 | 17.8 | 0.999 | -0.520 [-0.610, -0.445] |
| local field | 2 | +0.395 | 0.646 | 0.471 | 1e-06 | 5.8 | 0.997 | -0.146 [-0.207, -0.081] |
| local difference field | 2 | +0.033 | 0.816 | 0.647 | 1 | 10.6 | 0.998 | -0.508 [-0.561, -0.449] |

The error is in pKa units. A mean kernel close to 1 says the descriptor cannot tell two samples apart, whatever the regression does with it.

* at the module's default width (rho 0.5, local field) it is no better than the counts: -0.014 [-0.079, +0.051]
* at the module's default width (rho 0.5, local difference field) it loses to the counts: -0.226 [-0.383, -0.113]
* at the best width of the scan (rho 0.1, local field) it is no better than the counts: +0.041 [-0.013, +0.094]

