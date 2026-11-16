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

## 13C shifts in chloroform (nmrshiftdb2): the conformations the molecules visit

A measured value of this kind is an average over the conformations a molecule visits, and the single geometry that distance geometry returns is a stand-in for that average. 5 conformations were asked for per molecule and 4.8 were kept on average (2 molecules gave only one); the deposited geometry is the first of them, so the single-conformation row below is exactly the model the sections above measured. Envelope width rho 1, 5096 atoms in 400 molecules, 10 folds holding out whole molecules.

| conformations | weighting | q2 | error (ppm) | ridge | atoms per sample | difference in q2 against one conformation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| one, as prepared | -- | +0.973 | 8.824 | 1e-06 | 13 | -- |
| one, lowest in energy | -- | +0.972 | 8.880 | 1e-06 | 13 | -0.000 [-0.001, +0.000] |
| 5 asked for | uniform | +0.952 | 11.632 | 1e-06 | 71 | -0.020 [-0.024, -0.017] |
| 5 asked for | boltzmann | +0.956 | 11.149 | 1e-06 | 71 | -0.016 [-0.019, -0.014] |

Averaging over the conformations costs: the better of the two weightings is boltzmann, -0.016 [-0.019, -0.014] against the conformation the set was prepared with.

The row that says whether that is about averaging or about the geometries themselves is the second one: a single conformation again, the lowest in energy of the same pool, which scores -0.000 [-0.001, +0.000]. One conformation is one conformation whichever it is, so the loss belongs to the averaging rather than to the geometries.

## carboxylic acid pKa (Baltruschat and Czodrowski 2020): the conformations the molecules visit

A measured value of this kind is an average over the conformations a molecule visits, and the single geometry that distance geometry returns is a stand-in for that average. 5 conformations were asked for per molecule and 4.8 were kept on average (0 molecules gave only one); the deposited geometry is the first of them, so the single-conformation row below is exactly the model the sections above measured. Envelope width rho 0.1, 912 atoms in 912 molecules, 10 folds holding out whole molecules.

| conformations | weighting | q2 | error (pKa units) | ridge | atoms per sample | difference in q2 against one conformation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| one, as prepared | -- | +0.582 | 0.537 | 0.001 | 25 | -- |
| one, lowest in energy | -- | +0.576 | 0.540 | 0.01 | 25 | -0.005 [-0.043, +0.035] |
| 5 asked for | uniform | +0.562 | 0.550 | 0.01 | 127 | -0.020 [-0.061, +0.025] |
| 5 asked for | boltzmann | +0.549 | 0.557 | 0.01 | 127 | -0.032 [-0.070, +0.013] |

Averaging over the conformations changes nothing that can be measured: the better of the two weightings is uniform, -0.020 [-0.061, +0.025] against the conformation the set was prepared with.

The row that says whether that is about averaging or about the geometries themselves is the second one: a single conformation again, the lowest in energy of the same pool, which scores -0.005 [-0.043, +0.035]. One conformation is one conformation whichever it is, so the loss belongs to the averaging rather than to the geometries.

## The reference a spectroscopist would use: a HOSE-code lookup

Added 16 November 2026. The count reference above is the one rule 1 asked for, but it is not how 13C shifts are predicted in practice. The classical method is the HOSE code (Bremser 1978): describe the atom sphere by sphere of bonds, and predict the mean shift of the training atoms that share the deepest sphere the new atom has in common with any of them. It needs no geometry and fits nothing. The codes here are RDKit's Morgan environments, one identifier per atom and number of bonds. Every model below uses the same folds, which hold out whole molecules, and the field models a ridge grid that reaches down to 1e-09. The differences are in q2, against the lookup up to 4 bonds, with a 95 % interval bootstrapped over molecules.

### Every measured carbon: 5096 in 400 molecules

| model | q2 | error (ppm) | mean absolute error (ppm) | ridge | difference against the lookup |
| --- | ---: | ---: | ---: | ---: | ---: |
| HOSE lookup, up to 2 bonds | +0.975 | 8.36 | 5.00 | -- | -- |
| HOSE lookup, up to 3 bonds | +0.975 | 8.35 | 4.95 | -- | -- |
| HOSE lookup, up to 4 bonds | +0.976 | 8.34 | 4.93 | -- | -- |
| HOSE lookup, up to 5 bonds | +0.976 | 8.34 | 4.92 | -- | -- |
| atom-centred counts | +0.853 | 20.46 | 14.83 | 0.1 | -0.123 [-0.136, -0.111] |
| two-point local field, rho 0.5 | +0.962 | 10.40 | 7.20 | 1e-05 | -0.014 [-0.018, -0.009] |
| two-point local field, rho 1 | +0.973 | 8.82 | 6.08 | 1e-06 | -0.003 [-0.008, +0.001] |
| three-distance, fields (r_max 5 A, rho 0.5, sigma 0.35) | +0.990 | 5.45 | 3.55 | 0.0001 | +0.014 [+0.011, +0.017] |
| three-distance, elements (r_max 5 A, rho 0.5, sigma 0.35) | +0.989 | 5.47 | 3.59 | 0.0001 | +0.014 [+0.011, +0.017] |
| three-distance, fields and elements (r_max 5 A, rho 0.5, sigma 0.35) | +0.992 | 4.83 | 3.24 | 0.0001 | +0.016 [+0.014, +0.019] |

The lookup up to 4 bonds read its prediction at: 8.2 % at 0 bonds, 38.6 % at 1 bonds, 28.9 % at 2 bonds, 12.4 % at 3 bonds, 11.9 % at 4 bonds.
In the three-distance kernel the five fields against five element channels (H, C, N, O, other) in their place: +0.000 [-0.002, +0.002].

Control: the counts and the two-point rows of this stage against the same rows stored by the stage above, two point default +0.962 against +0.962, two point 1 +0.973 against +0.973, counts +0.853 against +0.853 -- the same.

### Aromatic ring carbons (plan item LP3): 1225 in 170 molecules

The ring carbons and the pose of LP3: the carbon at the origin, the bond that leaves the ring on +x, the ring in the xy plane, both mirror images averaged. The element channels are swapped in before the pose, so both rows go through the same frames.

| model | q2 | error (ppm) | mean absolute error (ppm) | ridge | difference against the lookup |
| --- | ---: | ---: | ---: | ---: | ---: |
| HOSE lookup, up to 4 bonds | +0.780 | 6.22 | 4.04 | -- | -- |
| atom-centred counts | +0.768 | 6.38 | 4.59 | 0.01 | -0.012 [-0.052, +0.027] |
| posed overlap, fields, rho 0.25 | +0.899 | 4.22 | 2.82 | 0.01 | +0.119 [+0.088, +0.153] |
| posed overlap, elements, rho 0.25 | +0.903 | 4.13 | 2.79 | 0.001 | +0.123 [+0.090, +0.158] |

The five fields against five element channels in the same posed overlap: -0.004 [-0.018, +0.011]. Control: the posed overlap with the fields gives q2 +0.899 here and +0.899 in `results/local_cofield/`.

**What this shows.** **The two-point field of LP2 does not beat the lookup** at its best width (-0.003 [-0.008, +0.001]). The count reference, which it beat by a wide margin, is itself far behind the lookup (-0.123 [-0.136, -0.111]), so that margin measured a weak reference. **The three-distance form of LP5 beats it** (+0.014 [+0.011, +0.017]), and the fields are not why: element channels in their place do as well (+0.000 [-0.002, +0.002]). **On the aromatic ring carbons the posed overlap beats the lookup by +0.119 [+0.088, +0.153]**, and again element channels do as well as the fields (-0.004 [-0.018, +0.011]). What predicts a shift beyond the lookup is where the atoms around the carbon sit, which the envelope, the pose and the three-distance kernel all carry. Put in place of the element of each atom, what the fields say about it does no better. (Added to the elements, the fields gain a little in the three-distance kernel: +0.002 [+0.001, +0.004] in `benchmarks/results/triangles/`.)

### The same ring carbons as the training set grows

Each fold trains on a random share of its training molecules; the test parts are the same as above, and each model's ridge is the one that cross-validates best at that share. 5 random draws per share; the q2 values are their means. The last column counts the draws whose 95 % interval, bootstrapped over molecules, excludes zero.

| share of the training molecules | training molecules per fold | HOSE lookup | posed overlap, element channels | posed overlap, fields | fields minus elements: mean (lowest to highest draw) | draws clear for the fields / for the elements |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 38 | +0.644 | +0.818 | +0.829 | +0.011 (-0.027 to +0.112) | 1 / 0 |
| 0.5 | 76 | +0.716 | +0.870 | +0.874 | +0.003 (-0.010 to +0.011) | 0 / 0 |
| 1 | 153 | +0.780 | +0.903 | +0.899 | -0.004 (-0.004 to -0.004) | 0 / 0 |

With 38 training molecules per fold the fields lead the element channels by +0.011 on average and the lookup by +0.185; with 153 the lead over the elements is -0.004 and over the lookup +0.119. The curve does not show a lead for the fields that grows as the data shrink: at 38 molecules the draws run from -0.027 to +0.112, and the interval is clear for the fields in 1 of 5 draws. What does grow is the lead of either geometric model over the lookup, which needs a near copy of an atom's surroundings among the training atoms and finds fewer of them in a smaller set.

