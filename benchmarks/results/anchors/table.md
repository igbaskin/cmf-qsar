# Fields read at a few anchored points

Plan item EC3. A kernel compares two fields everywhere; these descriptors
read the field and its gradient at a few places of a shared frame. For the
electrostatic field the gradient is the oriented electric field. Five
outer folds, ten inner folds for the ridge and the field width.

## What the columns can carry

A descriptor that is nearly the same in every member cannot separate two of
them, so the spread of each column and its correlation with each target are
measured before any model is fitted. The gradient is also checked against a
finite difference of the value at every anchored point.

### hammett: 95 members, points `meta_site`, `para_site`, `ipso`, `centroid`, `face`

The gradient matches a finite difference to 5.6e-12 at every point of every field.

| source | columns | target | best column | its \|r\| | columns with \|r\| above 0.5 |
|---|---|---|---|---|---|
| field | 100 | sigma_m | `at:ipso:electrostatic:value` | 0.820 | 6 |
| field | 100 | sigma_p | `at:ipso:electrostatic:value` | 0.832 | 3 |
| field | 100 | pi | `at:ipso:hydrophobic:gx` | 0.523 | 1 |
| coulomb | 20 | sigma_m | `at:ipso:coulomb:value` | 0.836 | 8 |
| coulomb | 20 | sigma_p | `at:face:coulomb:gx` | 0.812 | 6 |
| coulomb | 20 | pi | `at:centroid:coulomb:gx` | 0.333 | 0 |

### nickel: 25 members, points `metal`, `substituent_left`, `substituent_right`

The gradient matches a finite difference to 1.9e-11 at every point of every field.

| source | columns | target | best column | its \|r\| | columns with \|r\| above 0.5 |
|---|---|---|---|---|---|
| field | 60 | E_ox_AN | `at:substituent_right:steric:gz` | 0.847 | 24 |
| field | 60 | E_red_AN | `at:metal:steric:value` | 0.810 | 21 |
| coulomb | 12 | E_ox_AN | `at:substituent_right:coulomb:gx` | 0.622 | 4 |
| coulomb | 12 | E_red_AN | `at:metal:coulomb:gz` | 0.606 | 2 |

## The Hammett substituent constants

### sigma_m

#### random (training-mean error 0.2170 units)

| model | MAE (units) | standard deviation | R2 |
|---|---|---|---|
| electrostatic | 0.0977 | 0.0163 | +0.730 |
| physchem | 0.0972 | 0.0148 | +0.748 |
| anchors-near | 0.2328 | 0.1694 | -4.824 |
| anchors-far | 0.1078 | 0.0262 | +0.639 |
| anchors-all | 0.1322 | 0.0221 | +0.475 |
| coulomb-all | 0.1087 | 0.0231 | +0.661 |
| physchem+anchors-all | 0.1001 | 0.0208 | +0.718 |

| comparison | difference (units) | 95 % interval |
|---|---|---|
| anchors-all vs physchem | -0.0350 | [-0.0657, -0.0077] |
| anchors-near vs physchem | -0.1356 | [-0.2844, -0.0438] |
| coulomb-all vs anchors-all | +0.0235 | [+0.0031, +0.0478] |
| physchem+anchors-all vs physchem | -0.0029 | [-0.0190, +0.0127] |
| anchors-all vs electrostatic | -0.0345 | [-0.0636, -0.0073] |

### sigma_p

#### random (training-mean error 0.3148 units)

| model | MAE (units) | standard deviation | R2 |
|---|---|---|---|
| electrostatic | 0.1582 | 0.0302 | +0.694 |
| physchem | 0.1541 | 0.0187 | +0.703 |
| anchors-near | 0.3588 | 0.2178 | -3.825 |
| anchors-far | 0.1812 | 0.0429 | +0.550 |
| anchors-all | 0.2138 | 0.0634 | +0.120 |
| coulomb-all | 0.1801 | 0.0413 | +0.579 |
| physchem+anchors-all | 0.1836 | 0.0273 | +0.475 |

| comparison | difference (units) | 95 % interval |
|---|---|---|
| anchors-all vs physchem | -0.0597 | [-0.1192, -0.0154] |
| anchors-near vs physchem | -0.2047 | [-0.3870, -0.0846] |
| coulomb-all vs anchors-all | +0.0337 | [-0.0052, +0.0827] |
| physchem+anchors-all vs physchem | -0.0296 | [-0.0640, +0.0005] |
| anchors-all vs electrostatic | -0.0556 | [-0.1180, -0.0100] |

### pi

#### random (training-mean error 0.9023 units)

| model | MAE (units) | standard deviation | R2 |
|---|---|---|---|
| electrostatic | 0.8373 | 0.1627 | +0.082 |
| physchem | 0.3129 | 0.0280 | +0.863 |
| anchors-near | 0.9958 | 0.4570 | -2.123 |
| anchors-far | 0.7614 | 0.0742 | +0.169 |
| anchors-all | 0.7841 | 0.1160 | +0.131 |
| coulomb-all | 0.8985 | 0.1712 | -0.053 |
| physchem+anchors-all | 0.4886 | 0.2499 | +0.398 |

| comparison | difference (units) | 95 % interval |
|---|---|---|
| anchors-all vs physchem | -0.4732 | [-0.6366, -0.3268] |
| anchors-near vs physchem | -0.6889 | [-1.1440, -0.3960] |
| coulomb-all vs anchors-all | -0.1128 | [-0.2354, +0.0042] |
| physchem+anchors-all vs physchem | -0.1781 | [-0.3481, -0.0444] |
| anchors-all vs electrostatic | +0.0535 | [-0.0771, +0.1822] |

## The nickel complexes of item EC4

### E_ox_AN

#### random (training-mean error 0.4884 V)

| model | MAE (V) | standard deviation | R2 |
|---|---|---|---|
| electrostatic | 0.2234 | 0.0474 | +0.625 |
| physchem | 0.1501 | 0.0328 | +0.820 |
| anchors-near | 0.4718 | 0.1542 | -0.440 |
| anchors-far | 0.2984 | 0.1616 | -0.322 |
| anchors-all | 0.2801 | 0.2506 | -0.881 |
| coulomb-all | 0.4375 | 0.0910 | -0.204 |
| physchem+anchors-all | 0.2865 | 0.2441 | -0.872 |

| comparison | difference (V) | 95 % interval |
|---|---|---|
| anchors-all vs physchem | -0.1301 | [-0.4180, +0.0340] |
| anchors-near vs physchem | -0.3217 | [-0.5147, -0.1634] |
| coulomb-all vs anchors-all | -0.1574 | [-0.4354, +0.1714] |
| physchem+anchors-all vs physchem | -0.1364 | [-0.4199, +0.0209] |
| anchors-all vs electrostatic | -0.0568 | [-0.3306, +0.1141] |

#### leave-one-out (training-mean error 0.4898 V)

| model | MAE (V) | standard deviation | R2 |
|---|---|---|---|
| electrostatic | 0.2803 | 0.2292 | +0.576 |
| physchem | 0.1572 | 0.1756 | +0.821 |
| anchors-near | 0.5010 | 0.5764 | -0.884 |
| anchors-far | 0.3659 | 0.6574 | -0.829 |
| anchors-all | 0.2854 | 0.7011 | -0.851 |
| coulomb-all | 0.4870 | 0.4694 | -0.478 |
| physchem+anchors-all | 0.2946 | 0.6913 | -0.824 |

| comparison | difference (V) | 95 % interval |
|---|---|---|
| anchors-all vs physchem | -0.1282 | [-0.4097, +0.0349] |
| anchors-near vs physchem | -0.3438 | [-0.5856, -0.1577] |
| coulomb-all vs anchors-all | -0.2016 | [-0.4241, +0.0449] |
| physchem+anchors-all vs physchem | -0.1374 | [-0.4110, +0.0143] |
| anchors-all vs electrostatic | -0.0050 | [-0.2820, +0.1723] |

### E_red_AN

#### random (training-mean error 0.0498 V)

| model | MAE (V) | standard deviation | R2 |
|---|---|---|---|
| electrostatic | 0.0294 | 0.0165 | +0.302 |
| physchem | 0.0293 | 0.0155 | +0.215 |
| anchors-near | 0.0362 | 0.0200 | +0.189 |
| anchors-far | 0.0535 | 0.0221 | -0.585 |
| anchors-all | 0.0440 | 0.0284 | -0.201 |
| coulomb-all | 0.0487 | 0.0303 | -0.637 |
| physchem+anchors-all | 0.0434 | 0.0285 | -0.211 |

| comparison | difference (V) | 95 % interval |
|---|---|---|
| anchors-all vs physchem | -0.0147 | [-0.0229, -0.0062] |
| anchors-near vs physchem | -0.0069 | [-0.0200, +0.0052] |
| coulomb-all vs anchors-all | -0.0047 | [-0.0174, +0.0088] |
| physchem+anchors-all vs physchem | -0.0141 | [-0.0223, -0.0057] |
| anchors-all vs electrostatic | -0.0146 | [-0.0257, -0.0035] |

#### leave-one-out (training-mean error 0.0487 V)

| model | MAE (V) | standard deviation | R2 |
|---|---|---|---|
| electrostatic | 0.0292 | 0.0434 | +0.299 |
| physchem | 0.0327 | 0.0453 | +0.202 |
| anchors-near | 0.0453 | 0.0872 | -1.471 |
| anchors-far | 0.0592 | 0.0625 | -0.896 |
| anchors-all | 0.0480 | 0.0507 | -0.248 |
| coulomb-all | 0.0525 | 0.0726 | -1.057 |
| physchem+anchors-all | 0.0476 | 0.0511 | -0.247 |

| comparison | difference (V) | 95 % interval |
|---|---|---|
| anchors-all vs physchem | -0.0153 | [-0.0222, -0.0076] |
| anchors-near vs physchem | -0.0126 | [-0.0368, +0.0059] |
| coulomb-all vs anchors-all | -0.0045 | [-0.0231, +0.0129] |
| physchem+anchors-all vs physchem | -0.0148 | [-0.0214, -0.0074] |
| anchors-all vs electrostatic | -0.0188 | [-0.0309, -0.0081] |

