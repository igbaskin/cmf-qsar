# A learnable CMF layer (plan item ND2)

The attenuation of every field (the width of the Gaussian on every atom), the weight of every field in the joint kernel and the ridge are trained by gradient (`cmfqsar/learnable.py`). The coefficients of the model stay the closed-form ridge solution, so the model stays the kernel model and keeps its co-field. The criterion is the squared error of five-fold out-of-fold predictions inside the training part. The sets, the splits (random, 70 % training, 15 % validation, 15 % test) and the start of every training (the settings of the grid search) are those of plan item ND1. Every score is on the test rows.

What each row trained: **grid** nothing; **ridge** the ridge only; **weights** the field weights and the ridge; **widths** the attenuations and the ridge; **widths + weights** all three. These rows keep the evaluation with the lowest error on the validation rows (early stopping). **To the minimum** keeps the evaluation with the lowest criterion of the same training. **Layer** rows have the Gaussian kernel layer on top and train its width with the ridge.

## COMPAS-1D (8678 molecules, 5 seeds, 1000 landmark molecules)

The gap between the highest occupied and the lowest unoccupied orbital, in eV. Mean ± standard deviation over the seeds.

| Model | Test R² | Mean absolute error, eV |
|---|---:|---:|
| grid | +0.853 ± 0.032 | 0.114 ± 0.003 |
| ridge | +0.863 ± 0.012 | 0.114 ± 0.003 |
| weights | +0.863 ± 0.012 | 0.114 ± 0.003 |
| widths | +0.857 ± 0.023 | 0.114 ± 0.003 |
| widths + weights | +0.857 ± 0.022 | 0.114 ± 0.003 |
| widths + weights, to the minimum | +0.857 ± 0.023 | 0.114 ± 0.003 |
| grid, layer | +0.934 ± 0.008 | 0.075 ± 0.001 |
| ridge, layer | +0.938 ± 0.006 | 0.074 ± 0.001 |
| widths + weights, layer | +0.960 ± 0.006 | 0.058 ± 0.001 |
| network, fixed fields | +0.965 ± 0.006 | 0.054 ± 0.002 |
| network, trained fields | +0.959 ± 0.008 | 0.059 ± 0.005 |
| network + collectors, fixed fields | +0.987 ± 0.007 | 0.028 ± 0.001 |
| network + collectors, trained fields | +0.987 ± 0.007 | 0.028 ± 0.001 |
| the kernel model of ND1 on every training molecule | +0.869 ± 0.016 | 0.111 ± 0.003 |
| ND1: ridge regression on the CMF vector | +0.870 ± 0.008 | 0.112 ± 0.002 |
| ND1: the kernel model with the Gaussian layer | +0.966 ± 0.005 | 0.053 ± 0.001 |
| ND1: multilayer network on the CMF vector | +0.963 ± 0.005 | 0.055 ± 0.001 |
| ND1: the device with message passing | +0.982 ± 0.009 | 0.035 ± 0.002 |
| ND1: multilayer network on the CMF vector joined with the collectors | +0.987 ± 0.007 | 0.029 ± 0.002 |

The rows of ND1 are copied from its result files for the same test rows. The first is the kernel model on all 6,074 training molecules; every row of this item is a model over the 1,000 landmarks.

### Paired differences on COMPAS-1D

Difference of the absolute error on the same test molecules, in eV, with a 95 % bootstrap interval. A molecule in the test part of several seeds counts once. A negative value means that the first model has the smaller error.

| Comparison | Difference, eV |
|---|---:|
| ridge against grid | -0.0004 [-0.0013, +0.0002] (includes zero) |
| weights against ridge | -0.0000 [-0.0002, +0.0002] (includes zero) |
| widths against ridge | +0.0001 [-0.0007, +0.0011] (includes zero) |
| widths + weights against widths | +0.0001 [-0.0003, +0.0005] (includes zero) |
| widths + weights against grid | -0.0002 [-0.0009, +0.0005] (includes zero) |
| widths + weights, to the minimum against widths + weights | +0.0000 [-0.0005, +0.0005] (includes zero) |
| ridge, layer against grid, layer | -0.0015 [-0.0022, -0.0008] |
| widths + weights, layer against ridge, layer | -0.0152 [-0.0167, -0.0136] |
| network, trained fields against network, fixed fields | +0.0049 [+0.0036, +0.0062] |
| network + collectors, trained fields against network + collectors, fixed fields | +0.0007 [+0.0002, +0.0012] |
| network + collectors, trained fields against device, message passing | -0.0068 [-0.0077, -0.0060] |

### What the training changed on COMPAS-1D

The criterion is the out-of-fold squared error inside the training part, divided by the variance of the activity. The validation error is that of the model the row keeps.

| Model | Criterion at the start | Lowest criterion | Validation RMSE at the start, eV | Validation RMSE of the kept model, eV | Evaluations | Minutes |
|---|---:|---:|---:|---:|---:|---:|
| ridge | 0.1398 | 0.1374 | 0.1475 | 0.1455 | 19 | 1 |
| weights | 0.1398 | 0.1340 | 0.1475 | 0.1455 | 29 | 1 |
| widths | 0.1398 | 0.1317 | 0.1475 | 0.1468 | 44 | 33 |
| widths + weights | 0.1398 | 0.1313 | 0.1475 | 0.1467 | 42 | 33 |
| widths + weights, to the minimum | 0.1398 | 0.1313 | 0.1475 | 0.1502 | 42 | 33 |
| ridge, layer | 0.0700 | 0.0624 | 0.1018 | 0.0985 | 19 | 1 |
| widths + weights, layer | 0.0700 | 0.0405 | 0.1018 | 0.0802 | 43 | 33 |

### The attenuations and the weights on COMPAS-1D

The attenuation of the grid search is one of 0.1, 0.2, 0.5, 1, 2 and 5 for a group of fields. The trained values are geometric means over the seeds, with the smallest and the largest. The share is the weight of the field divided by the sum of the weights; equal weights give 0.111.

| Field (group) | Grid attenuation of the five seeds | Widths + weights: attenuation | Widths + weights: share of the weight | To the minimum: attenuation | To the minimum: share of the weight |
|---|---|---|---:|---|---:|
| steric (A) | 0.2, 0.5, 1, 1, 1 | 0.682 (0.168 to 1.39) | 0.127 | 0.551 (0.131 to 1.35) | 0.146 |
| prop:q_xtb (B) | 0.1, 0.1, 0.1, 0.1, 0.1 | 0.158 (0.1 to 0.258) | 0.100 | 0.17 (0.127 to 0.201) | 0.064 |
| cif:C.H (C) | 0.1, 0.2, 0.1, 0.2, 0.1 | 0.116 (0.0445 to 0.2) | 0.121 | 0.123 (0.0682 to 0.188) | 0.160 |
| cif:C.2r (C) | 0.1, 0.2, 0.1, 0.2, 0.1 | 0.151 (0.0767 to 0.312) | 0.115 | 0.143 (0.0782 to 0.35) | 0.103 |
| ring:terminal (D) | 0.1, 0.1, 0.1, 0.1, 0.1 | 0.0633 (0.0241 to 0.181) | 0.105 | 0.0519 (0.0153 to 0.185) | 0.140 |
| ring:linear (D) | 0.1, 0.1, 0.1, 0.1, 0.1 | 0.129 (0.0879 to 0.232) | 0.115 | 0.126 (0.0776 to 0.243) | 0.101 |
| ring:angular (D) | 0.1, 0.1, 0.1, 0.1, 0.1 | 0.162 (0.1 to 0.256) | 0.095 | 0.225 (0.113 to 0.346) | 0.085 |
| ring:branched (D) | 0.1, 0.1, 0.1, 0.1, 0.1 | 0.132 (0.1 to 0.17) | 0.129 | 0.135 (0.0641 to 0.257) | 0.131 |
| ring:sextet (E) | 0.2, 0.2, 0.2, 0.2, 0.2 | 0.243 (0.2 to 0.269) | 0.092 | 0.31 (0.267 to 0.448) | 0.071 |

## The Sutherland series (8 series, 10 seeds each)

Every training molecule is a landmark, so every row is the exact kernel model. Test R², mean over the seeds. The column before the last is the mean over the series. The last column is the median over all splits of all series, which one failed split does not move.

| Model | ACE | AChE | BZR | COX2 | DHFR | GPB | THER | THR | Mean over the series | Median over all splits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grid | +0.605 | +0.321 | +0.121 | +0.480 | +0.632 | +0.462 | +0.247 | +0.430 | +0.412 | +0.504 |
| ridge | +0.626 | +0.330 | +0.141 | +0.490 | +0.640 | +0.440 | +0.282 | +0.448 | +0.425 | +0.523 |
| weights | +0.647 | +0.292 | +0.132 | +0.493 | +0.639 | +0.470 | +0.301 | +0.518 | +0.437 | +0.513 |
| widths | +0.618 | +0.270 | +0.166 | +0.516 | +0.648 | +0.497 | +0.316 | +0.486 | +0.440 | +0.514 |
| widths + weights | +0.629 | -0.057 | +0.109 | +0.481 | +0.629 | +0.488 | +0.292 | +0.508 | +0.385 | +0.521 |
| widths + weights, to the minimum | +0.592 | +0.119 | +0.348 | +0.487 | +0.644 | +0.462 | +0.076 | -0.033 | +0.337 | +0.450 |

### Paired differences on the Sutherland series

For every comparison: the number of series in which the first model has the smaller absolute error with an interval that excludes zero, the number in which it has the larger one, and the mean difference of the test R² over the series.

| Comparison | Series better | Series worse | Mean difference of R² |
|---|---:|---:|---:|
| ridge against grid | 6 | 0 | +0.012 |
| weights against ridge | 0 | 1 | +0.012 |
| widths against ridge | 0 | 1 | +0.015 |
| widths + weights against widths | 0 | 0 | -0.055 |
| widths + weights against grid | 0 | 0 | -0.027 |
| widths + weights, to the minimum against widths + weights | 0 | 0 | -0.048 |

### The criterion and the validation error on the Sutherland series

Means over all series and seeds. The criterion is the out-of-fold squared error inside the training part divided by the variance of the activity.

| Model | Criterion at the start | Lowest criterion | Share of the trainings in which early stopping kept the start |
|---|---:|---:|---:|
| ridge | 0.738 | 0.514 | 55 % |
| weights | 0.738 | 0.432 | 40 % |
| widths | 0.738 | 0.402 | 42 % |
| widths + weights | 0.738 | 0.351 | 35 % |

## Where the training of the attenuations ended

The attenuation of a field may move between 0.001 and 100. A trained attenuation is counted as at a limit when it is within 1 % of one of the two values. The share is over the fields and the splits. The last three columns are shares of the trainings, by the reason the optimiser (L-BFGS-B) gave for its end: the criterion or its gradient no longer changed; all the allowed steps were used; the search along the last direction found no lower value of the criterion.

| Set | Model | Attenuations at the lower limit | Attenuations at the upper limit | Trainings ended by the criterion | Trainings ended by the number of steps | Trainings ended by a search that found no lower value |
|---|---|---:|---:|---:|---:|---:|
| COMPAS-1D | widths | 0 % | 0 % | 20 % | 20 % | 60 % |
| COMPAS-1D | widths + weights | 0 % | 0 % | 20 % | 40 % | 40 % |
| COMPAS-1D | widths + weights, to the minimum | 0 % | 0 % | 20 % | 40 % | 40 % |
| COMPAS-1D | widths + weights, layer | 0 % | 0 % | 40 % | 20 % | 40 % |
| Sutherland series | widths | 1 % | 1 % | 60 % | 25 % | 15 % |
| Sutherland series | widths + weights | 0 % | 2 % | 21 % | 54 % | 25 % |
| Sutherland series | widths + weights, to the minimum | 2 % | 4 % | 21 % | 54 % | 25 % |

## The co-field of the trained layer against that of the grid model

The co-field of every field was evaluated on a grid of points 1 Å apart over the landmark molecules, for the grid model and for the trained models of the same split. The number is the correlation coefficient of the two maps. Where early stopping kept the start, the trained model is the grid model and the two maps are equal; these splits are counted and left out of the mean. The mean is over the fields and the remaining splits, with the smallest value of any field and split. The last two columns are the reference: the same correlation between the grid models of two splits of the set, which differ in their training molecules only.

| Set | Splits | Widths + weights: splits with a changed layer | Widths + weights: mean correlation | Widths + weights: smallest correlation | To the minimum: splits with a changed layer | To the minimum: mean correlation | To the minimum: smallest correlation | Grid models of two splits: mean correlation | Grid models of two splits: smallest correlation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compas-1D | 5 | 4 | 0.660 | 0.328 | 5 | 0.654 | 0.050 | 0.712 | 0.316 |
| ACE | 10 | 7 | 0.774 | 0.038 | 10 | 0.364 | -0.016 | 0.469 | 0.131 |
| AChE | 10 | 5 | 0.552 | 0.016 | 10 | 0.399 | -0.364 | 0.427 | -0.136 |
| BZR | 10 | 4 | 0.545 | 0.033 | 10 | 0.317 | 0.014 | 0.416 | 0.009 |
| COX2 | 10 | 9 | 0.733 | -0.018 | 10 | 0.498 | -0.025 | 0.456 | 0.172 |
| DHFR | 10 | 8 | 0.725 | 0.024 | 10 | 0.483 | -0.146 | 0.528 | 0.019 |
| GPB | 10 | 7 | 0.660 | -0.226 | 10 | 0.438 | -0.226 | 0.437 | -0.081 |
| THER | 10 | 6 | 0.800 | -0.011 | 10 | 0.317 | -0.262 | 0.371 | 0.029 |
| THR | 10 | 6 | 0.620 | 0.024 | 10 | 0.386 | -0.221 | 0.390 | 0.017 |

