# The molecule vector of a trained device as a kernel view (plan item ND14b)

A Gaussian kernel on the collectors of the QSPR Neural Device (the sums over its atoms, bonds and two-bond paths), trained on the training molecules of the series, is stacked under the field kernels and gets a weight by the evidence (plan item E6). The devices and the splits are those of plan item ND1; nothing is trained again. The sets are the eight Sutherland series (ten random splits each) and 2,000 molecules of COMPAS-1D (five splits, as in ND7). Every score is on the test molecules. *invariant* is the rotation-invariant autocorrelation kernel of plan item C5.

## The kernel-correlation check (plan item EC5)

The correlation of the kernel on the collectors with the kernel the model already has, over all pairs of two different molecules after both kernels are normalised to a diagonal of one. Mean over the splits. A value near one means that the view adds no new description. *all* uses every molecule of the set, *test* only the test molecules (the collectors of the training molecules are fitted to their activities).

| Set | Splits | Device against the five fields, all | Device against the five fields, test | Device against fields + invariant, all | Device against fields + invariant, test | Invariant against the five fields, all | Device against the device without message passing, all |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACE | 10 | 0.197 | 0.190 | 0.310 | 0.281 | 0.205 | 0.795 |
| AChE | 10 | 0.212 | 0.222 | 0.223 | 0.231 | 0.688 | 0.600 |
| BZR | 10 | 0.246 | 0.199 | 0.243 | 0.198 | 0.603 | 0.442 |
| COX2 | 10 | 0.255 | 0.276 | 0.287 | 0.301 | 0.650 | 0.362 |
| DHFR | 10 | 0.390 | 0.425 | 0.418 | 0.453 | 0.488 | 0.446 |
| GPB | 10 | 0.345 | 0.306 | 0.342 | 0.298 | 0.535 | 0.302 |
| THER | 10 | 0.227 | 0.209 | 0.248 | 0.240 | 0.705 | 0.661 |
| THR | 10 | 0.538 | 0.477 | 0.546 | 0.478 | 0.737 | 0.635 |
| COMPAS-1D | 5 | 0.230 | 0.222 | 0.131 | 0.133 | 0.354 | 0.259 |

## Test R² of every model

Mean over the splits. The weights of the models in the first block are those of the evidence on the training molecules. In the second block the share of the view in the joint kernel and the width of its Gaussian are chosen on the validation molecules.

| Model | ACE | AChE | BZR | COX2 | DHFR | GPB | THER | THR | COMPAS-1D | Mean over the Sutherland series |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fields | +0.624 | +0.350 | +0.286 | +0.488 | +0.596 | +0.432 | +0.370 | +0.591 | +0.412 | +0.467 |
| fields + invariant | +0.663 | +0.372 | +0.380 | +0.525 | +0.595 | +0.456 | +0.376 | +0.572 | +0.907 | +0.492 |
| fields + device | +0.730 | +0.264 | +0.423 | +0.441 | +0.667 | +0.218 | +0.525 | +0.293 | +0.981 | +0.445 |
| fields + invariant + device | +0.729 | +0.270 | +0.443 | +0.447 | +0.662 | +0.224 | +0.537 | +0.304 | +0.981 | +0.452 |
| fields + device, no message passing | +0.714 | +0.295 | +0.389 | +0.500 | +0.684 | +0.481 | +0.377 | +0.559 | - | +0.500 |
| fields + device, weight on validation | +0.697 | +0.181 | +0.410 | +0.473 | +0.689 | +0.234 | +0.455 | +0.524 | +0.981 | +0.458 |
| fields + device, no message passing, weight on validation | +0.671 | +0.217 | +0.374 | +0.476 | +0.652 | +0.487 | +0.277 | +0.492 | +0.769 | +0.456 |
| device alone | +0.727 | -0.159 | +0.409 | +0.433 | +0.664 | +0.140 | +0.387 | +0.149 | +0.983 | +0.344 |
| the device itself | +0.712 | -0.016 | +0.392 | +0.418 | +0.648 | +0.271 | +0.316 | +0.105 | +0.987 | +0.356 |

## Paired differences

Difference of the absolute error on the same test molecules, in units of the activity (for COMPAS-1D in eV), with a 95 % bootstrap interval. A molecule in the test part of several splits counts once. A negative value means that the first model has the smaller error.

| Number | Comparison | ACE | AChE | BZR | COX2 | DHFR | GPB | THER | THR | COMPAS-1D | Sets better | Sets worse |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | fields + device against fields + invariant | -0.068 [-0.210, +0.072] | +0.051 [-0.033, +0.132] | +0.014 [-0.048, +0.068] | +0.018 [-0.039, +0.071] | -0.028 [-0.081, +0.021] | +0.054 [-0.044, +0.151] | -0.165 [-0.396, +0.045] | +0.114 [+0.031, +0.192] | -0.063 [-0.068, -0.057] | 1 | 1 |
| 2 | fields + device against fields | -0.097 [-0.264, +0.063] | +0.039 [-0.037, +0.113] | -0.035 [-0.095, +0.024] | +0.001 [-0.054, +0.054] | -0.028 [-0.081, +0.023] | +0.046 [-0.055, +0.148] | -0.192 [-0.425, +0.011] | +0.108 [+0.026, +0.188] | -0.217 [-0.228, -0.206] | 1 | 1 |
| 3 | fields + invariant + device against fields + invariant | -0.070 [-0.209, +0.069] | +0.042 [-0.042, +0.122] | +0.003 [-0.058, +0.056] | +0.013 [-0.043, +0.065] | -0.024 [-0.077, +0.026] | +0.051 [-0.046, +0.146] | -0.181 [-0.409, +0.026] | +0.100 [+0.014, +0.181] | -0.063 [-0.068, -0.058] | 1 | 1 |
| 4 | fields + device, weight on validation against fields + invariant | -0.032 [-0.152, +0.085] | +0.104 [+0.033, +0.174] | +0.028 [-0.024, +0.073] | +0.012 [-0.036, +0.058] | -0.033 [-0.084, +0.015] | +0.095 [-0.008, +0.194] | -0.138 [-0.344, +0.060] | +0.051 [-0.014, +0.114] | -0.063 [-0.068, -0.058] | 1 | 1 |
| 5 | fields + device, weight on validation against fields | -0.061 [-0.215, +0.083] | +0.092 [+0.027, +0.153] | -0.020 [-0.072, +0.028] | -0.005 [-0.050, +0.041] | -0.032 [-0.083, +0.019] | +0.088 [-0.014, +0.187] | -0.166 [-0.380, +0.034] | +0.046 [-0.017, +0.105] | -0.217 [-0.228, -0.206] | 1 | 1 |
| 6 | fields + device, no message passing against fields + invariant | -0.011 [-0.098, +0.076] | +0.026 [-0.021, +0.073] | +0.009 [-0.035, +0.048] | -0.001 [-0.030, +0.028] | -0.027 [-0.070, +0.012] | +0.010 [-0.020, +0.039] | -0.027 [-0.112, +0.056] | +0.018 [-0.011, +0.048] | - | 0 | 0 |
| 7 | fields + device, no message passing, weight on validation against fields + invariant | +0.056 [-0.050, +0.167] | +0.120 [+0.041, +0.204] | +0.032 [-0.020, +0.081] | +0.012 [-0.025, +0.048] | -0.014 [-0.055, +0.027] | +0.023 [-0.051, +0.097] | +0.008 [-0.228, +0.230] | +0.070 [+0.012, +0.126] | +0.058 [+0.050, +0.065] | 0 | 3 |
| 8 | fields + invariant against fields | -0.029 [-0.118, +0.054] | -0.012 [-0.039, +0.011] | -0.049 [-0.080, -0.017] | -0.017 [-0.040, +0.006] | +0.001 [-0.018, +0.019] | -0.007 [-0.029, +0.018] | -0.028 [-0.069, +0.016] | -0.005 [-0.019, +0.008] | -0.154 [-0.165, -0.143] | 2 | 0 |
| 9 | device alone against the device itself | -0.027 [-0.098, +0.049] | -0.028 [-0.103, +0.045] | -0.008 [-0.035, +0.019] | -0.008 [-0.037, +0.022] | -0.003 [-0.026, +0.018] | +0.012 [-0.040, +0.056] | -0.064 [-0.190, +0.067] | -0.012 [-0.083, +0.050] | +0.000 [-0.001, +0.003] | 0 | 0 |
| 10 | fields + device, weight on validation against the device itself | -0.015 [-0.152, +0.122] | -0.148 [-0.244, -0.054] | -0.022 [-0.056, +0.012] | -0.023 [-0.055, +0.010] | -0.020 [-0.043, +0.003] | +0.015 [-0.051, +0.076] | -0.188 [-0.402, +0.026] | -0.173 [-0.285, -0.064] | +0.000 [-0.002, +0.003] | 2 | 0 |

## The share of the device in the joint kernel

The share is the weight times the scale of the term, divided by the sum over the terms (mean over the splits). The last column is the share chosen on the validation molecules.

| Set | Share by the evidence in fields + device | Share by the evidence in fields + invariant + device | Share of the invariant kernel in fields + invariant | Share chosen on the validation molecules |
|---|---:|---:|---:|---:|
| ACE | 75% | 73% | 22% | 65% |
| AChE | 28% | 30% | 8% | 35% |
| BZR | 76% | 70% | 12% | 47% |
| COX2 | 82% | 80% | 23% | 55% |
| DHFR | 81% | 81% | 27% | 49% |
| GPB | 57% | 57% | 6% | 70% |
| THER | 81% | 80% | 16% | 38% |
| THR | 46% | 47% | 3% | 21% |
| COMPAS-1D | 95% | 94% | 100% | 65% |

## The gate

*fields + device* must have a smaller absolute error than *fields + invariant* on COMPAS-1D and on DHFR, with an interval that excludes zero on both.

| Set | Difference of the absolute error | Gate |
|---|---:|---|
| COMPAS-1D | -0.0627 [-0.0678, -0.0574] | met |
| DHFR | -0.0283 [-0.0812, +0.0214] | not met |

The gate is not met.
