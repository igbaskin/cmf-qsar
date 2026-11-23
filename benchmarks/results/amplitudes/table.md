# Atom amplitudes of the fields from a network on the atoms (plan item ND14a)

A field puts a Gaussian on every atom, and the amplitude of the Gaussian is a fixed number: a partial charge, a well depth, a hydrogen-bond strength, a one in an indicator field. Here a small network on the atoms of the molecular graph produces a correction to every amplitude (`cmfqsar/amplitudes.py`). The network has shared weights for all atoms, it starts at zero, and a penalty holds it near zero. The model stays the kernel model: its coefficients are the closed-form ridge solution for the current amplitudes, and it keeps its co-field. The attenuations and the field weights are those of the grid search of plan item ND1 and are not trained. The sets and the splits (random, 70 % training, 15 % validation, 15 % test) are those of ND1 and ND2. Every score is on the test rows.

What each row trained: **fixed amplitudes** the ridge only (the reference); **learned, the atom alone** the ridge and a network that reads one atom; **learned, with neighbours** the ridge and a network in which every atom also reads its bonded neighbours, two rounds. These rows keep the evaluation with the lowest error on the validation rows (early stopping). **learned, to the minimum** keeps the evaluation with the lowest objective of the training of *learned, with neighbours*. **Layer** rows have the Gaussian kernel layer on top. The network has 16 hidden units; the penalty is 0.1 times the mean squared correction, in units of the field scale.

## COMPAS-1D (8678 molecules, 5 seeds, 1000 landmark molecules)

The gap between the highest occupied and the lowest unoccupied orbital, in eV. Mean ± standard deviation over the seeds. The amplitudes of `steric`, `prop:q_xtb`, `cif:C.H`, `cif:C.2r` are corrected; the ring fields are not.

| Model | Test R² | Mean absolute error, eV |
|---|---:|---:|
| fixed amplitudes | +0.863 ± 0.012 | 0.114 ± 0.003 |
| learned, the atom alone | +0.883 ± 0.006 | 0.105 ± 0.001 |
| learned, with neighbours | +0.918 ± 0.012 | 0.085 ± 0.003 |
| learned, to the minimum | +0.919 ± 0.012 | 0.085 ± 0.003 |
| fixed amplitudes, layer | +0.935 ± 0.007 | 0.075 ± 0.001 |
| learned, with neighbours, layer | +0.962 ± 0.004 | 0.057 ± 0.003 |
| ND2: nothing trained (the grid search) | +0.853 ± 0.032 | 0.114 ± 0.003 |
| ND2: the ridge trained | +0.863 ± 0.012 | 0.114 ± 0.003 |
| ND2: attenuations, field weights and ridge trained | +0.857 ± 0.022 | 0.114 ± 0.003 |
| ND2: nothing trained, Gaussian kernel layer | +0.934 ± 0.008 | 0.075 ± 0.001 |
| ND2: everything trained, Gaussian kernel layer | +0.960 ± 0.006 | 0.058 ± 0.001 |

The rows of ND2 are copied from its result files for the same test rows.

### Paired differences on COMPAS-1D

Difference of the absolute error on the same test molecules, in eV, with a 95 % bootstrap interval. A molecule in the test part of several seeds counts once. A negative value means that the first model has the smaller error.

| Number | Comparison | Difference, eV |
|---:|---|---:|
| 1 | learned, with neighbours against fixed amplitudes | -0.0273 [-0.0293, -0.0252] |
| 2 | learned, the atom alone against fixed amplitudes | -0.0080 [-0.0093, -0.0067] |
| 3 | learned, with neighbours against learned, the atom alone | -0.0193 [-0.0211, -0.0174] |
| 4 | learned, to the minimum against learned, with neighbours | -0.0005 [-0.0008, -0.0002] |
| 5 | learned, with neighbours, layer against fixed amplitudes, layer | -0.0173 [-0.0187, -0.0158] |

### What the training changed on COMPAS-1D

The criterion is the out-of-fold squared error inside the training part, divided by the variance of the activity. The validation error is in eV.

| Model | Weights of the network | Criterion at the start | Criterion of the kept model | Lowest criterion | Validation RMSE at the start | Validation RMSE of the kept model | Share of the trainings in which early stopping kept the start | Evaluations | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed amplitudes | 0 | 0.1402 | 0.1371 | 0.1357 | 0.1474 | 0.1455 | 20 % | 11 | 42 |
| learned, the atom alone | 372 | 0.1402 | 0.1159 | 0.1157 | 0.1474 | 0.1338 | 0 % | 44 | 2689 |
| learned, with neighbours | 1428 | 0.1402 | 0.0775 | 0.0766 | 0.1474 | 0.1119 | 0 % | 39 | 2398 |
| learned, to the minimum | 1428 | 0.1402 | 0.0766 | 0.0766 | 0.1474 | 0.1141 | 0 % | 39 | 2398 |
| fixed amplitudes, layer | 0 | 0.0700 | 0.0676 | 0.0661 | 0.1018 | 0.1013 | 0 % | 11 | 42 |
| learned, with neighbours, layer | 1428 | 0.0700 | 0.0399 | 0.0389 | 0.1018 | 0.0789 | 0 % | 39 | 2405 |

Root mean square of the correction over the corrected atoms of the training molecules, in units of the field scale (the root mean square of the fixed amplitude):

| Model | steric | prop:q_xtb | cif:C.H | cif:C.2r |
|---|---:|---:|---:|---:|
| learned, the atom alone | 0.04 | 0.06 | 0.10 | 0.16 |
| learned, with neighbours | 0.05 | 0.10 | 0.11 | 0.08 |
| learned, to the minimum | 0.05 | 0.07 | 0.12 | 0.06 |
| learned, with neighbours, layer | 0.08 | 0.25 | 0.09 | 0.13 |

The amplitudes and the co-field of the trained models against those of the model with the fixed amplitudes, over the atoms of the landmark molecules and on a grid of points 1 Å apart; means over the seeds:

| Model | Field | Correlation of the trained with the fixed amplitudes | Correlation of the co-field with that of the fixed amplitudes | Reference: correlation of the co-fields of two splits, fixed amplitudes (ND2) |
|---|---|---:|---:|---:|
| learned, the atom alone | steric | 0.999 | 0.774 | 0.591 |
| learned, the atom alone | prop:q_xtb | 0.998 | 0.825 | 0.800 |
| learned, the atom alone | cif:C.H | - | 0.608 | 0.607 |
| learned, the atom alone | cif:C.2r | - | 0.650 | 0.636 |
| learned, with neighbours | steric | 0.992 | 0.520 | 0.591 |
| learned, with neighbours | prop:q_xtb | 0.999 | 0.661 | 0.800 |
| learned, with neighbours | cif:C.H | - | 0.263 | 0.607 |
| learned, with neighbours | cif:C.2r | - | 0.426 | 0.636 |
| learned, to the minimum | steric | 0.991 | 0.540 | 0.591 |
| learned, to the minimum | prop:q_xtb | 0.999 | 0.683 | 0.800 |
| learned, to the minimum | cif:C.H | - | 0.250 | 0.607 |
| learned, to the minimum | cif:C.2r | - | 0.411 | 0.636 |

Mean correction of *learned, with neighbours* for every atom type, in units of the field scale, with the root mean square in brackets:

| Atom type | steric | prop:q_xtb | cif:C.H | cif:C.2r |
|---|---:|---:|---:|---:|
| C.2r | +0.03 (0.08) | -0.01 (0.07) | - | +0.00 (0.08) |
| C.H | -0.00 (0.04) | -0.01 (0.14) | +0.02 (0.13) | - |
| H | -0.00 (0.05) | +0.04 (0.11) | - | - |

## The Sutherland series (8 series, 10 seeds each)

Every training molecule is a landmark. The amplitudes of all five fields are corrected. Test R², mean over the seeds. The column before the last is the mean over the series. The last column is the median over all splits of all series. The series of the gate is DHFR, named before any run.

| Model | ACE | AChE | BZR | COX2 | DHFR | GPB | THER | THR | Mean over the series | Median over all splits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed amplitudes | +0.631 | +0.332 | +0.135 | +0.486 | +0.645 | +0.450 | +0.275 | +0.443 | +0.425 | +0.522 |
| learned, the atom alone | +0.617 | +0.217 | +0.151 | +0.475 | +0.648 | +0.428 | +0.330 | +0.550 | +0.427 | +0.510 |
| learned, with neighbours | +0.666 | +0.307 | +0.189 | +0.480 | +0.649 | +0.411 | +0.440 | +0.475 | +0.452 | +0.538 |
| learned, to the minimum | +0.656 | +0.113 | +0.185 | +0.461 | +0.643 | +0.235 | +0.294 | +0.402 | +0.374 | +0.473 |
| ND2: nothing trained (the grid search) | +0.605 | +0.321 | +0.121 | +0.480 | +0.632 | +0.462 | +0.247 | +0.430 | +0.412 | +0.504 |
| ND2: the ridge trained | +0.626 | +0.330 | +0.141 | +0.490 | +0.640 | +0.440 | +0.282 | +0.448 | +0.425 | +0.523 |
| ND2: attenuations, field weights and ridge trained | +0.629 | -0.057 | +0.109 | +0.481 | +0.629 | +0.488 | +0.292 | +0.508 | +0.385 | +0.521 |

### Paired differences on the Sutherland series

Difference of the absolute error on the same test molecules, in units of the activity (the negative decimal logarithm of a concentration), with a 95 % bootstrap interval. A negative value means that the first model has the smaller error.

| Number | Comparison | ACE | AChE | BZR | COX2 | DHFR | GPB | THER | THR | Series better | Series worse |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | learned, with neighbours against fixed amplitudes | -0.081 [-0.195, +0.039] | +0.005 [-0.029, +0.039] | -0.006 [-0.042, +0.026] | +0.003 [-0.021, +0.026] | -0.034 [-0.059, -0.010] | -0.001 [-0.062, +0.070] | -0.219 [-0.370, -0.079] | -0.009 [-0.054, +0.037] | 2 | 0 |
| 2 | learned, the atom alone against fixed amplitudes | -0.006 [-0.076, +0.072] | +0.051 [+0.010, +0.095] | -0.003 [-0.024, +0.016] | -0.005 [-0.035, +0.024] | -0.023 [-0.043, -0.003] | +0.003 [-0.032, +0.040] | -0.151 [-0.288, -0.014] | -0.049 [-0.107, +0.011] | 2 | 1 |
| 3 | learned, with neighbours against learned, the atom alone | -0.075 [-0.171, +0.010] | -0.046 [-0.091, -0.005] | -0.003 [-0.039, +0.031] | +0.008 [-0.021, +0.038] | -0.011 [-0.030, +0.008] | -0.004 [-0.079, +0.067] | -0.068 [-0.208, +0.066] | +0.039 [-0.004, +0.083] | 1 | 0 |
| 4 | learned, to the minimum against learned, with neighbours | +0.015 [-0.077, +0.104] | +0.072 [-0.008, +0.151] | +0.005 [-0.055, +0.067] | +0.002 [-0.031, +0.034] | +0.014 [-0.006, +0.033] | +0.079 [+0.001, +0.156] | +0.055 [-0.130, +0.243] | +0.017 [-0.048, +0.086] | 0 | 1 |

### What the training changed on the Sutherland series

Means over all series and seeds. The criterion is the out-of-fold squared error inside the training part, divided by the variance of the activity.

| Model | Weights of the network | Criterion at the start | Criterion of the kept model | Lowest criterion | Validation RMSE at the start | Validation RMSE of the kept model | Share of the trainings in which early stopping kept the start | Evaluations | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed amplitudes | 0 | 0.7381 | 0.7278 | 0.5138 | 0.8180 | 0.8139 | 52 % | 6 | 1 |
| learned, the atom alone | 437 | 0.7381 | 0.4923 | 0.2539 | 0.8180 | 0.7557 | 24 % | 38 | 20 |
| learned, with neighbours | 1493 | 0.7381 | 0.4800 | 0.1932 | 0.8180 | 0.7489 | 24 % | 38 | 20 |
| learned, to the minimum | 1493 | 0.7381 | 0.1932 | 0.1932 | 0.8180 | 0.8911 | 0 % | 38 | 20 |

Root mean square of the correction over the corrected atoms of the training molecules, in units of the field scale:

| Model | electrostatic | steric | hydrophobic | hba | hbd |
|---|---:|---:|---:|---:|---:|
| learned, the atom alone | 0.42 | 0.17 | 0.34 | 0.42 | 0.17 |
| learned, with neighbours | 0.35 | 0.18 | 0.43 | 0.30 | 0.12 |
| learned, to the minimum | 0.50 | 0.32 | 0.50 | 0.46 | 0.21 |

The amplitudes and the co-field of the trained models of DHFR against those of the model with the fixed amplitudes; means over the seeds:

| Model | Field | Correlation of the trained with the fixed amplitudes | Correlation of the co-field with that of the fixed amplitudes | Reference: correlation of the co-fields of two splits, fixed amplitudes (ND2) |
|---|---|---:|---:|---:|
| learned, the atom alone | electrostatic | 0.915 | 0.582 | 0.487 |
| learned, the atom alone | steric | 0.899 | 0.865 | 0.657 |
| learned, the atom alone | hydrophobic | 0.935 | 0.696 | 0.483 |
| learned, the atom alone | hba | 0.912 | 0.730 | 0.530 |
| learned, the atom alone | hbd | 0.999 | 0.914 | 0.485 |
| learned, with neighbours | electrostatic | 0.904 | 0.588 | 0.487 |
| learned, with neighbours | steric | 0.853 | 0.886 | 0.657 |
| learned, with neighbours | hydrophobic | 0.936 | 0.731 | 0.483 |
| learned, with neighbours | hba | 0.845 | 0.745 | 0.530 |
| learned, with neighbours | hbd | 0.971 | 0.871 | 0.485 |
| learned, to the minimum | electrostatic | 0.836 | 0.461 | 0.487 |
| learned, to the minimum | steric | 0.834 | 0.818 | 0.657 |
| learned, to the minimum | hydrophobic | 0.923 | 0.629 | 0.483 |
| learned, to the minimum | hba | 0.819 | 0.637 | 0.530 |
| learned, to the minimum | hbd | 0.935 | 0.800 | 0.485 |

Mean correction of *learned, with neighbours* on DHFR for every atom type that holds at least 1 % of the corrected atoms, in units of the field scale, with the root mean square in brackets:

| Atom type | electrostatic | steric | hydrophobic | hba | hbd |
|---|---:|---:|---:|---:|---:|
| C.3 | -0.33 (0.61) | +0.14 (0.43) | +0.20 (0.43) | - | - |
| C.ar | -0.24 (0.60) | -0.04 (0.26) | +0.19 (0.42) | - | - |
| H | -0.14 (0.38) | +0.05 (0.32) | +0.13 (0.38) | - | -0.00 (0.12) |
| N.ar | +0.11 (0.44) | +0.08 (0.32) | +0.28 (0.47) | -0.02 (0.20) | - |
| N.pl3 | +0.04 (0.54) | +0.21 (0.48) | +0.42 (0.60) | +0.01 (0.37) | - |
| O.3 | +0.17 (0.54) | +0.07 (0.33) | +0.49 (0.66) | -0.16 (0.37) | - |

## The gate

The item continues only if *learned, with neighbours* has a smaller absolute error than *fixed amplitudes* on COMPAS-1D and on DHFR, with an interval that excludes zero on both.

| Set | Difference of the absolute error | Gate |
|---|---:|---|
| COMPAS-1D | -0.0273 [-0.0293, -0.0252] | met |
| DHFR | -0.0342 [-0.0593, -0.0095] | met |

