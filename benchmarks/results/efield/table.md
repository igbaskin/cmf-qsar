# The electric-field kernel against the scalar one

Plan item EC5. The kernel of this field type is the integral of the two
molecules' field *gradients* against each other, in closed form, and it
carries the same charges as the electrostatic field. So every comparison
below is between two kernels on one description. Five outer folds, ten
inner folds for the ridge and the field width.

## Is it the integral it claims, and is it different?

On two molecules of the Hammett series the closed form gives 0.259395 and a grid integral of 2744000 points at 0.129 angstrom gives 0.259395, 0.000 % apart.

The pair integral changes sign at 2.45 angstrom (at the field width alpha = 1), and 72 % of the atom pairs of this series lie beyond it, so most pairs contribute a negative term. This is the only pair integral in the program that can, and the kernel matrix is still positive semidefinite because it is an inner product of two vector fields.

Two molecules of the series are 0.312 +/- 0.314 alike through the scalar kernel and 0.302 +/- 0.312 through the vector one; the two cosine kernels correlate +0.980. A correlation of one would mean the vector kernel orders the pairs exactly as the scalar one does, and no model built on it could differ.

## The Hammett substituent constants

### sigma_m

#### random (training-mean error 0.2170 units)

| model | MAE (units) | standard deviation | R2 |
|---|---|---|---|
| electrostatic | 0.0977 | 0.0163 | +0.730 |
| efield | 0.0963 | 0.0169 | +0.736 |
| physchem | 0.0972 | 0.0148 | +0.748 |
| physchem+efield | 0.0967 | 0.0106 | +0.760 |
| efield-for-es | 0.0986 | 0.0103 | +0.746 |

| comparison | difference (units) | 95 % interval |
|---|---|---|
| efield vs electrostatic | +0.0014 | [-0.0037, +0.0064] |
| physchem+efield vs physchem | +0.0005 | [-0.0037, +0.0051] |
| efield-for-es vs physchem | -0.0015 | [-0.0055, +0.0028] |

### sigma_p

#### random (training-mean error 0.3148 units)

| model | MAE (units) | standard deviation | R2 |
|---|---|---|---|
| electrostatic | 0.1582 | 0.0302 | +0.694 |
| efield | 0.1624 | 0.0281 | +0.680 |
| physchem | 0.1541 | 0.0187 | +0.703 |
| physchem+efield | 0.1473 | 0.0102 | +0.745 |
| efield-for-es | 0.1501 | 0.0175 | +0.727 |

| comparison | difference (units) | 95 % interval |
|---|---|---|
| efield vs electrostatic | -0.0043 | [-0.0140, +0.0048] |
| physchem+efield vs physchem | +0.0068 | [-0.0045, +0.0190] |
| efield-for-es vs physchem | +0.0040 | [-0.0021, +0.0101] |

### pi

#### random (training-mean error 0.9023 units)

| model | MAE (units) | standard deviation | R2 |
|---|---|---|---|
| electrostatic | 0.8373 | 0.1627 | +0.082 |
| efield | 0.7762 | 0.1162 | +0.185 |
| physchem | 0.3129 | 0.0280 | +0.863 |
| physchem+efield | 0.3187 | 0.0501 | +0.860 |
| efield-for-es | 0.3398 | 0.0348 | +0.849 |

| comparison | difference (units) | 95 % interval |
|---|---|---|
| efield vs electrostatic | +0.0620 | [+0.0238, +0.1084] |
| physchem+efield vs physchem | -0.0061 | [-0.0273, +0.0129] |
| efield-for-es vs physchem | -0.0266 | [-0.0531, +0.0016] |

## The nickel complexes of item EC4

### E_ox_AN

#### random (training-mean error 0.4884 V)

| model | MAE (V) | standard deviation | R2 |
|---|---|---|---|
| electrostatic | 0.2234 | 0.0474 | +0.625 |
| efield | 0.2176 | 0.0547 | +0.628 |
| physchem | 0.1501 | 0.0328 | +0.820 |
| physchem+efield | 0.1638 | 0.0306 | +0.792 |
| efield-for-es | 0.1482 | 0.0359 | +0.827 |

| comparison | difference (V) | 95 % interval |
|---|---|---|
| efield vs electrostatic | +0.0057 | [-0.0072, +0.0194] |
| physchem+efield vs physchem | -0.0138 | [-0.0342, -0.0005] |
| efield-for-es vs physchem | +0.0019 | [-0.0031, +0.0077] |

#### leave-one-out (training-mean error 0.4898 V)

| model | MAE (V) | standard deviation | R2 |
|---|---|---|---|
| electrostatic | 0.2803 | 0.2292 | +0.576 |
| efield | 0.2504 | 0.2393 | +0.613 |
| physchem | 0.1572 | 0.1756 | +0.821 |
| physchem+efield | 0.1708 | 0.1886 | +0.791 |
| efield-for-es | 0.1553 | 0.1722 | +0.826 |

| comparison | difference (V) | 95 % interval |
|---|---|---|
| efield vs electrostatic | +0.0299 | [+0.0144, +0.0459] |
| physchem+efield vs physchem | -0.0136 | [-0.0307, -0.0017] |
| efield-for-es vs physchem | +0.0019 | [-0.0040, +0.0075] |

### E_red_AN

#### random (training-mean error 0.0498 V)

| model | MAE (V) | standard deviation | R2 |
|---|---|---|---|
| electrostatic | 0.0294 | 0.0165 | +0.302 |
| efield | 0.0312 | 0.0175 | +0.259 |
| physchem | 0.0293 | 0.0155 | +0.215 |
| physchem+efield | 0.0282 | 0.0154 | +0.274 |
| efield-for-es | 0.0289 | 0.0153 | +0.228 |

| comparison | difference (V) | 95 % interval |
|---|---|---|
| efield vs electrostatic | -0.0018 | [-0.0062, +0.0021] |
| physchem+efield vs physchem | +0.0011 | [-0.0007, +0.0033] |
| efield-for-es vs physchem | +0.0004 | [-0.0001, +0.0010] |

#### leave-one-out (training-mean error 0.0487 V)

| model | MAE (V) | standard deviation | R2 |
|---|---|---|---|
| electrostatic | 0.0292 | 0.0434 | +0.299 |
| efield | 0.0293 | 0.0464 | +0.231 |
| physchem | 0.0327 | 0.0453 | +0.202 |
| physchem+efield | 0.0316 | 0.0435 | +0.261 |
| efield-for-es | 0.0325 | 0.0452 | +0.209 |

| comparison | difference (V) | 95 % interval |
|---|---|---|
| efield vs electrostatic | -0.0000 | [-0.0030, +0.0029] |
| physchem+efield vs physchem | +0.0011 | [-0.0014, +0.0040] |
| efield-for-es vs physchem | +0.0002 | [-0.0003, +0.0008] |

### Where the vector co-field points

The co-field of this kind is an arrow field: at each place it says which way the model wants the field to change. The strongest arrows of the E_ox_AN model, at least 2 angstrom apart:

| length | from the metal (A) | from the nearest donor (A) | pointing away from the metal |
|---|---|---|---|
| 0.3766 | 6.24 | 4.13 | -0.2572 |
| 0.3741 | 6.58 | 4.47 | -0.2820 |
| 0.2872 | 5.58 | 3.60 | -0.1285 |
| 0.2687 | 5.81 | 4.09 | -0.1645 |
| 0.2683 | 3.45 | 1.39 | +0.2241 |
| 0.2670 | 5.66 | 3.97 | -0.1686 |
| 0.2665 | 4.75 | 2.81 | +0.0079 |
| 0.2548 | 4.79 | 3.12 | +0.0033 |

