# Transition-metal complexes: the fields against the composition

Plan item EC4. Two series of coordination complexes with computed redox
potentials: 2216 iron(II) complexes from the structural database, which
share the metal and nothing else, and 25 substituted nickel complexes,
which share one ligand skeleton and one coordination sphere. Errors are
mean absolute errors in volts over five outer folds, with the ridge and
the field width chosen inside each fold.

## What the series hold

The iron series holds 2216 complexes with 6.2 dative
bonds each and 377 different first coordination spheres; its
target is +0.57 +/- 0.66 V
(-1.90 to +7.48).
Without the charge repair of this item,
197 of 200 complexes
checked have an electrostatic field that is identically zero, because one metal
atom makes the Gasteiger scheme return nothing for every atom of the molecule.

The nickel series holds 25 complexes, all with the coordination sphere
N3O3. Matching each first coordination sphere onto the first
complex's puts the donors 0.03 A apart on
average (worst 0.23 A), so the series has a real
shared frame. The rule frame of the principal axes puts the same donors
1.82 A from the matched frame
(worst 3.45 A).

| field | mean similarity of two complexes | standard deviation | smallest |
|---|---|---|---|
| electrostatic | 0.953 | 0.032 | 0.856 |
| steric | 0.958 | 0.027 | 0.849 |
| hydrophobic | 0.907 | 0.052 | 0.746 |
| hba | 0.877 | 0.089 | 0.627 |
| hbd | 0.993 | 0.020 | 0.908 |

A mean near one means the field has nearly the same shape in every complex
and cannot separate two of them.

## Iron: every complex of the series

2216 complexes. Only the models that cost little are measured on the whole series; the overlap kernel is measured on the subset below.

### random folds (training-mean error 0.523 V)

| model | MAE (V) | standard deviation | R2 |
|---|---|---|---|
| composition | 0.3132 | 0.0103 | +0.566 |
| site | 0.3590 | 0.0109 | +0.447 |
| invariant | 0.2530 | 0.0128 | +0.690 |

| comparison | difference (V) | 95 % interval |
|---|---|---|
| invariant vs composition | +0.0602 | [+0.0493, +0.0713] |
| composition vs site | +0.0458 | [+0.0374, +0.0547] |

### unseen-sphere folds (training-mean error 0.523 V)

| model | MAE (V) | standard deviation | R2 |
|---|---|---|---|
| composition | 0.3364 | 0.0267 | +0.507 |
| site | 0.3976 | 0.0320 | +0.348 |
| invariant | 0.2840 | 0.0227 | +0.630 |

| comparison | difference (V) | 95 % interval |
|---|---|---|
| invariant vs composition | +0.0554 | [+0.0437, +0.0676] |
| composition vs site | +0.0604 | [+0.0516, +0.0694] |

## Iron: a random subset, with the overlap kernel

800 complexes. The overlap kernel is measured on this random subset and not on the whole series, because its cost grows faster than the square of the number of complexes: one field width took 16 s at 800 complexes and 62 s at 1200.

### random folds (training-mean error 0.528 V)

| model | MAE (V) | standard deviation | R2 |
|---|---|---|---|
| composition | 0.3127 | 0.0247 | +0.603 |
| site | 0.3578 | 0.0216 | +0.476 |
| invariant | 0.2659 | 0.0212 | +0.702 |
| overlap-raw | 0.5235 | 0.0268 | +0.005 |
| overlap-rule | 0.3207 | 0.0200 | +0.596 |

| comparison | difference (V) | 95 % interval |
|---|---|---|
| invariant vs composition | +0.0468 | [+0.0276, +0.0655] |
| overlap-rule vs composition | -0.0080 | [-0.0283, +0.0108] |
| overlap-rule vs overlap-raw | +0.2029 | [+0.1763, +0.2311] |
| invariant vs overlap-rule | +0.0548 | [+0.0367, +0.0736] |
| composition vs site | +0.0451 | [+0.0311, +0.0603] |

### unseen-sphere folds (training-mean error 0.535 V)

| model | MAE (V) | standard deviation | R2 |
|---|---|---|---|
| composition | 0.3420 | 0.0272 | +0.546 |
| site | 0.3869 | 0.0516 | +0.408 |
| invariant | 0.2968 | 0.0217 | +0.661 |
| overlap-raw | 0.5316 | 0.0819 | -0.022 |
| overlap-rule | 0.3493 | 0.0185 | +0.543 |

| comparison | difference (V) | 95 % interval |
|---|---|---|
| invariant vs composition | +0.0493 | [+0.0292, +0.0694] |
| overlap-rule vs composition | -0.0061 | [-0.0276, +0.0151] |
| overlap-rule vs overlap-raw | +0.1845 | [+0.1570, +0.2127] |
| invariant vs overlap-rule | +0.0554 | [+0.0364, +0.0752] |
| composition vs site | +0.0485 | [+0.0335, +0.0649] |

## Nickel: 25 complexes in a shared frame

### E_ox_AN, random (training-mean error 0.488 V)

| model | MAE (V) | standard deviation | R2 |
|---|---|---|---|
| quantum | 0.3449 | 0.0842 | +0.375 |
| site | 0.3704 | 0.0844 | +0.299 |
| invariant | 0.1545 | 0.0540 | +0.825 |
| overlap-rule | 0.1500 | 0.0589 | +0.819 |
| physchem | 0.1501 | 0.0328 | +0.820 |

| comparison | difference (V) | 95 % interval |
|---|---|---|
| physchem vs quantum | +0.1949 | [+0.0682, +0.3211] |
| invariant vs quantum | +0.1905 | [+0.0684, +0.3122] |
| physchem vs site | +0.2204 | [+0.0854, +0.3552] |
| physchem vs invariant | +0.0044 | [-0.0761, +0.0681] |
| overlap-rule vs physchem | +0.0000 | [-0.0432, +0.0468] |

### E_ox_AN, leave-one-out (training-mean error 0.490 V)

| model | MAE (V) | standard deviation | R2 |
|---|---|---|---|
| quantum | 0.3518 | 0.2745 | +0.357 |
| site | 0.3661 | 0.2689 | +0.333 |
| invariant | 0.1910 | 0.1768 | +0.781 |
| overlap-rule | 0.1363 | 0.1903 | +0.823 |
| physchem | 0.1572 | 0.1756 | +0.821 |

| comparison | difference (V) | 95 % interval |
|---|---|---|
| physchem vs quantum | +0.1946 | [+0.0789, +0.3090] |
| invariant vs quantum | +0.1608 | [+0.0424, +0.2773] |
| physchem vs site | +0.2089 | [+0.0864, +0.3242] |
| physchem vs invariant | +0.0338 | [-0.0339, +0.0869] |
| overlap-rule vs physchem | +0.0209 | [-0.0316, +0.0779] |

### E_red_AN, random (training-mean error 0.050 V)

| model | MAE (V) | standard deviation | R2 |
|---|---|---|---|
| quantum | 0.0428 | 0.0196 | -0.059 |
| site | 0.0468 | 0.0227 | -0.003 |
| invariant | 0.0314 | 0.0110 | +0.340 |
| overlap-rule | 0.0418 | 0.0148 | +0.024 |
| physchem | 0.0293 | 0.0155 | +0.215 |

| comparison | difference (V) | 95 % interval |
|---|---|---|
| physchem vs quantum | +0.0134 | [+0.0055, +0.0219] |
| invariant vs quantum | +0.0113 | [+0.0033, +0.0196] |
| physchem vs site | +0.0175 | [+0.0051, +0.0303] |
| physchem vs invariant | +0.0021 | [-0.0041, +0.0080] |
| overlap-rule vs physchem | -0.0124 | [-0.0243, -0.0028] |

### E_red_AN, leave-one-out (training-mean error 0.049 V)

| model | MAE (V) | standard deviation | R2 |
|---|---|---|---|
| quantum | 0.0408 | 0.0564 | -0.239 |
| site | 0.0470 | 0.0406 | +0.012 |
| invariant | 0.0309 | 0.0363 | +0.418 |
| overlap-rule | 0.0429 | 0.0432 | +0.051 |
| physchem | 0.0327 | 0.0453 | +0.202 |

| comparison | difference (V) | 95 % interval |
|---|---|---|
| physchem vs quantum | +0.0081 | [-0.0004, +0.0169] |
| invariant vs quantum | +0.0099 | [-0.0005, +0.0220] |
| physchem vs site | +0.0143 | [+0.0016, +0.0282] |
| physchem vs invariant | -0.0018 | [-0.0092, +0.0056] |
| overlap-rule vs physchem | -0.0102 | [-0.0187, -0.0012] |

## Where the co-field of the nickel series has its weight

| target | field | peak | from the metal (A) | from the nearest N donor (A) | from the nearest water O (A) |
|---|---|---|---|---|---|
| E_ox_AN | Electrostatic | -0.2011 | 6.70 | 4.60 | 7.03 |
| E_ox_AN | Steric | -0.1249 | 7.33 | 5.23 | 7.65 |
| E_ox_AN | Hydrophobic | +0.1787 | 4.52 | 2.42 | 4.96 |
| E_ox_AN | H-bond acceptor | -0.3487 | 5.60 | 3.53 | 5.92 |
| E_ox_AN | H-bond donor | +0.0560 | 3.09 | 3.35 | 1.37 |
| E_red_AN | Electrostatic | +0.0450 | 3.61 | 2.10 | 3.59 |
| E_red_AN | Steric | -0.0218 | 4.43 | 3.43 | 3.60 |
| E_red_AN | Hydrophobic | -0.0265 | 4.72 | 3.60 | 3.98 |
| E_red_AN | H-bond acceptor | -0.0339 | 3.05 | 2.14 | 2.61 |
| E_red_AN | H-bond donor | -0.0324 | 2.65 | 2.62 | 1.58 |

