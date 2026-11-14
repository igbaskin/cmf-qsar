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
| field | 100 | sigma_m | `at:ipso:electrostatic:value` | 0.823 | 6 |
| field | 100 | sigma_p | `at:ipso:electrostatic:value` | 0.834 | 3 |
| field | 100 | pi | `at:ipso:hydrophobic:gx` | 0.519 | 1 |
| coulomb | 20 | sigma_m | `at:ipso:coulomb:value` | 0.835 | 10 |
| coulomb | 20 | sigma_p | `at:face:coulomb:gx` | 0.808 | 9 |
| coulomb | 20 | pi | `at:meta_site:coulomb:gx` | 0.375 | 0 |

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
| electrostatic | 0.1117 | 0.0254 | +0.664 |
| physchem | 0.1022 | 0.0202 | +0.725 |
| anchors-near | 0.3050 | 0.2959 | -16.363 |
| anchors-far | 0.1145 | 0.0190 | +0.650 |
| anchors-all | 0.1361 | 0.0252 | +0.429 |
| coulomb-all | 0.1108 | 0.0225 | +0.660 |
| physchem+anchors-all | 0.1011 | 0.0218 | +0.703 |

| comparison | difference (units) | 95 % interval |
|---|---|---|
| anchors-all vs physchem | -0.0339 | [-0.0671, -0.0043] |
| anchors-near vs physchem | -0.2027 | [-0.4586, -0.0387] |
| coulomb-all vs anchors-all | +0.0253 | [+0.0017, +0.0522] |
| physchem+anchors-all vs physchem | +0.0011 | [-0.0156, +0.0175] |
| anchors-all vs electrostatic | -0.0244 | [-0.0539, +0.0021] |

### sigma_p

#### random (training-mean error 0.3148 units)

| model | MAE (units) | standard deviation | R2 |
|---|---|---|---|
| electrostatic | 0.1631 | 0.0323 | +0.679 |
| physchem | 0.1501 | 0.0181 | +0.713 |
| anchors-near | 0.6514 | 0.7889 | -58.911 |
| anchors-far | 0.1760 | 0.0428 | +0.569 |
| anchors-all | 0.1871 | 0.0407 | +0.506 |
| coulomb-all | 0.1864 | 0.0594 | +0.466 |
| physchem+anchors-all | 0.2009 | 0.0616 | +0.233 |

| comparison | difference (units) | 95 % interval |
|---|---|---|
| anchors-all vs physchem | -0.0370 | [-0.0728, -0.0055] |
| anchors-near vs physchem | -0.5012 | [-1.2041, -0.0966] |
| coulomb-all vs anchors-all | +0.0007 | [-0.0512, +0.0461] |
| physchem+anchors-all vs physchem | -0.0508 | [-0.1010, -0.0107] |
| anchors-all vs electrostatic | -0.0240 | [-0.0565, +0.0068] |

### pi

#### random (training-mean error 0.9023 units)

| model | MAE (units) | standard deviation | R2 |
|---|---|---|---|
| electrostatic | 0.8371 | 0.1746 | +0.053 |
| physchem | 0.2930 | 0.0429 | +0.877 |
| anchors-near | 0.9891 | 0.3633 | -2.126 |
| anchors-far | 0.7369 | 0.0631 | +0.212 |
| anchors-all | 0.7964 | 0.1185 | +0.065 |
| coulomb-all | 0.8767 | 0.1202 | -0.001 |
| physchem+anchors-all | 0.4478 | 0.1723 | +0.562 |

| comparison | difference (units) | 95 % interval |
|---|---|---|
| anchors-all vs physchem | -0.5059 | [-0.6721, -0.3474] |
| anchors-near vs physchem | -0.7023 | [-1.1498, -0.4091] |
| coulomb-all vs anchors-all | -0.0781 | [-0.1756, +0.0245] |
| physchem+anchors-all vs physchem | -0.1570 | [-0.2982, -0.0459] |
| anchors-all vs electrostatic | +0.0411 | [-0.0904, +0.1761] |

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

