# Item BT9, gate (b): a model of the simulated label

The label of every one of the 62 electrolytes of CALiSol-23 comes from its own simulation (`run_diffusion.py`). A model of the label is read on solvent mixtures that it has not seen: the folds hold out whole mixtures, and five seeds of the folds are averaged. The errors are in the natural logarithm of the label, so an error of 0.1 is about 10 %. The reading rule of the gate was fixed before the models were run (see the docstring of `benchmarks/run_labelmodel.py`).

The labels have a scatter that no model can remove. One start of 4 ns scatters by 24 % in the logarithm of the conductivity and by 7 % in the logarithm of the salt diffusion coefficient. Both come from the six pairs of starts of the concentration series. A model that reproduced the true value exactly would therefore have a mean absolute error of about 0.191 for the conductivity and about 0.057 for the diffusion coefficient.

## Natural logarithm of the salt diffusion coefficient, ideal solution

**New solvent mixtures (the gate).** Predicting the mean costs 0.455 ln(m2/s).

| Model | Mean absolute error | Standardised error, root mean square | Inside one standard deviation | Mean standard deviation |
|---|---:|---:|---:|---:|
| conditions | 0.315 | 1.16 | 58 % | 0.341 |
| composition | 0.045 | 0.78 | 81 % | 0.077 |
| shells | 0.154 | 1.02 | 70 % | 0.196 |
| shells-free | 0.110 | 1.19 | 59 % | 0.115 |
| reference | 0.246 | 1.00 | 74 % | 0.339 |

A positive difference means that the first model has the smaller error. The interval is 95 %, from resampling solvent mixtures.

| Comparison | Difference of the mean absolute error | Interval |
|---|---:|---|
| shells-free vs reference | +0.136 | +0.087 to +0.191 |
| shells vs reference | +0.092 | +0.036 to +0.154 |
| shells-free vs composition | -0.066 | -0.084 to -0.049 |
| shells vs composition | -0.109 | -0.134 to -0.086 |
| composition vs conditions | +0.271 | +0.230 to +0.314 |
| shells-free vs conditions | +0.205 | +0.165 to +0.245 |
| the frame on the shells | -0.044 | -0.061 to -0.026 |

**Random folds, the easier question.** Predicting the mean costs 0.455 ln(m2/s).

| Model | Mean absolute error | Standardised error, root mean square | Inside one standard deviation | Mean standard deviation |
|---|---:|---:|---:|---:|
| conditions | 0.433 | 1.09 | 62 % | 0.455 |
| composition | 0.046 | 1.01 | 71 % | 0.061 |
| shells | 0.158 | 1.04 | 69 % | 0.199 |
| shells-free | 0.102 | 1.10 | 62 % | 0.111 |
| reference | 0.244 | 1.07 | 75 % | 0.333 |

A positive difference means that the first model has the smaller error. The interval is 95 %, from resampling solvent mixtures.

| Comparison | Difference of the mean absolute error | Interval |
|---|---:|---|
| shells-free vs reference | +0.142 | +0.090 to +0.199 |
| shells vs reference | +0.087 | +0.025 to +0.154 |
| shells-free vs composition | -0.057 | -0.075 to -0.040 |
| shells vs composition | -0.112 | -0.145 to -0.082 |
| composition vs conditions | +0.388 | +0.221 to +0.690 |
| shells-free vs conditions | +0.331 | +0.172 to +0.623 |
| the frame on the shells | -0.055 | -0.076 to -0.035 |

**Gate (b) for this label, model shells-free.** The deviation is calibrated (standardised error 1.19, asked for 0.8 to 1.25). The model is ahead of the reference with the measured conductivity. The gate holds.

**The composition is the better model.** The composition alone has a smaller error than the shells, by 0.066 (0.049 to 0.084). The description by shells adds nothing to it, as the plan expected. Its standardised error is 0.78 (asked for 0.8 to 1.25); a value below the range means that the model is less sure than it could be.

## Natural logarithm of the simulated conductivity

**New solvent mixtures (the gate).** Predicting the mean costs 0.432 ln(S/m).

| Model | Mean absolute error | Standardised error, root mean square | Inside one standard deviation | Mean standard deviation |
|---|---:|---:|---:|---:|
| conditions | 0.422 | 1.13 | 63 % | 0.505 |
| composition | 0.298 | 1.15 | 73 % | 0.359 |
| shells | 0.335 | 1.14 | 68 % | 0.410 |
| shells-free | 0.312 | 1.10 | 65 % | 0.367 |
| reference | 0.284 | 1.03 | 69 % | 0.358 |

A positive difference means that the first model has the smaller error. The interval is 95 %, from resampling solvent mixtures.

| Comparison | Difference of the mean absolute error | Interval |
|---|---:|---|
| shells-free vs reference | -0.028 | -0.085 to +0.025 (includes zero) |
| shells vs reference | -0.051 | -0.119 to +0.013 (includes zero) |
| shells-free vs composition | -0.014 | -0.042 to +0.011 (includes zero) |
| shells vs composition | -0.036 | -0.080 to +0.006 (includes zero) |
| composition vs conditions | +0.124 | +0.053 to +0.202 |
| shells-free vs conditions | +0.110 | +0.040 to +0.183 |
| the frame on the shells | -0.023 | -0.058 to +0.014 (includes zero) |

**Random folds, the easier question.** Predicting the mean costs 0.432 ln(S/m).

| Model | Mean absolute error | Standardised error, root mean square | Inside one standard deviation | Mean standard deviation |
|---|---:|---:|---:|---:|
| conditions | 0.410 | 1.05 | 65 % | 0.507 |
| composition | 0.289 | 1.10 | 72 % | 0.347 |
| shells | 0.346 | 1.14 | 65 % | 0.418 |
| shells-free | 0.314 | 1.08 | 64 % | 0.363 |
| reference | 0.279 | 0.99 | 71 % | 0.365 |

A positive difference means that the first model has the smaller error. The interval is 95 %, from resampling solvent mixtures.

| Comparison | Difference of the mean absolute error | Interval |
|---|---:|---|
| shells-free vs reference | -0.035 | -0.087 to +0.015 (includes zero) |
| shells vs reference | -0.067 | -0.141 to +0.000 (includes zero) |
| shells-free vs composition | -0.025 | -0.058 to +0.009 (includes zero) |
| shells vs composition | -0.057 | -0.116 to -0.004 |
| composition vs conditions | +0.121 | +0.041 to +0.208 |
| shells-free vs conditions | +0.096 | +0.020 to +0.177 |
| the frame on the shells | -0.032 | -0.075 to +0.009 (includes zero) |

**Gate (b) for this label, model shells-free.** The deviation is calibrated (standardised error 1.10, asked for 0.8 to 1.25). The model is not clearly ahead of the reference with the measured conductivity. The gate does not hold.

## What the simulation adds to the composition

The target is the measured conductivity of CALiSol-23, in S/m, as the natural logarithm. The test mixture is new. Predicting the mean costs 0.336 ln(S/m).

**New solvent mixtures.**

| Model | Mean absolute error | Standardised error, root mean square |
|---|---:|---:|
| conditions | 0.308 | 1.12 |
| composition | 0.148 | 1.04 |
| simulated | 0.215 | 1.01 |
| composition + simulated | 0.137 | 0.99 |

| Comparison | Difference of the mean absolute error | Interval |
|---|---:|---|
| composition + simulated vs composition | +0.011 | -0.008 to +0.031 (includes zero) |
| simulated vs composition | -0.068 | -0.105 to -0.030 |
| composition vs conditions | +0.161 | +0.105 to +0.222 |
| composition + simulated vs simulated | +0.079 | +0.039 to +0.118 |

**Random folds.**

| Model | Mean absolute error | Standardised error, root mean square |
|---|---:|---:|
| conditions | 0.308 | 1.16 |
| composition | 0.136 | 1.53 |
| simulated | 0.219 | 1.08 |
| composition + simulated | 0.135 | 1.53 |

| Comparison | Difference of the mean absolute error | Interval |
|---|---:|---|
| composition + simulated vs composition | +0.000 | -0.013 to +0.014 (includes zero) |
| simulated vs composition | -0.083 | -0.113 to -0.053 |
| composition vs conditions | +0.173 | +0.109 to +0.239 |
| composition + simulated vs simulated | +0.083 | +0.046 to +0.120 |

