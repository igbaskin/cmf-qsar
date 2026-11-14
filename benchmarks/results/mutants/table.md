# Point mutants as fields in the frame of the mutated residue (plan item PP1)

657 single substitutions at 174 positions of 4 domains (1O6X, 1UFM, 2K5N, 3ZGK), from the mega-scale data set of Tsuboyama et al. (2023).  The target is ddG_ML in kcal/mol (the unfolding free energy of the variant minus that of the wild type), whose standard deviation over these samples is 1.23.  A difference field has 24 atoms on average.  Of the substitutions measured at a position, 4 enter the measurement, drawn with a fixed seed: the cost of a kernel grows with the square of the sample count and the cost of a fit with its cube, and a position is kept whole because a position is what the folds hold out.

## Before any fit

The correlation of a kernel with the target says whether samples the description finds alike have alike targets (item EC5).

| Kernel | Best correlation with ddG | Setting |
|---|---|---|
| difference field: the new side chain minus the old one | +0.103 | alpha 0.3 |
| the new side chain alone | +0.034 | alpha 3.0 |
| the surroundings of the site, without the replaced side chain | +0.032 | rho 0.1 |

How much of a difference field is the substitution itself: the mean kernel over pairs of samples of each kind, at an attenuation of 1.

| Pairs | Difference field | The new side chain alone |
|---|---|---|
| same substitution, other position | +0.329 | +0.398 |
| same new residue | +0.106 | +0.378 |
| same wild-type residue | +0.181 | +0.229 |
| every pair | +0.005 | +0.233 |

The first row covers 637 pairs.

## Holding out whole positions (174 of them, 10 folds)

| Description | q2 | sd over repeats | RMSE, kcal/mol | MAE, kcal/mol |
|---|---|---|---|---|
| the mean of the training part | -0.010 | 0.002 | 1.239 | 0.983 |
| linear regression on the change of residue volume, hydropathy and charge | +0.181 | 0.007 | 1.116 | 0.873 |
| linear regression on those three columns and the one-hot codes of the two residues | +0.127 | 0.013 | 1.152 | 0.870 |
| linear regression on atom counts in shells about the site, with the one-hot codes | +0.049 | 0.036 | 1.203 | 0.925 |
| linear regression on how many atoms surround the site, with the three columns | +0.300 | 0.009 | 1.032 | 0.814 |
| the surroundings of the site, without the replaced side chain | +0.190 | 0.043 | 1.110 | 0.869 |
| the new side chain alone | +0.065 | 0.022 | 1.192 | 0.940 |
| difference field: the new side chain minus the old one | +0.156 | 0.012 | 1.133 | 0.880 |
| difference field and surroundings, joined | +0.311 | 0.008 | 1.023 | 0.805 |

Differences in q2, with the 95 % bootstrap interval over the held-out unit:

| Comparison | Difference in q2 |
|---|---|
| difference - three columns | -0.025 [-0.072, +0.025] |
| difference - substitution | +0.029 [-0.058, +0.140] |
| difference - counts | +0.107 [-0.086, +0.366] |
| difference - mean | +0.166 [+0.081, +0.240] |
| difference+context - difference | +0.155 [+0.083, +0.232] |
| difference+context - three columns | +0.130 [+0.035, +0.237] |
| difference+context - burial | +0.011 [-0.086, +0.111] |
| context - burial | -0.110 [-0.241, +0.017] |
| burial - three columns | +0.119 [+0.035, +0.199] |
| new - difference | -0.091 [-0.169, -0.008] |
| context - mean | +0.200 [+0.087, +0.301] |

## Holding out whole segments (23 of them, 10 folds)

| Description | q2 | sd over repeats | RMSE, kcal/mol | MAE, kcal/mol |
|---|---|---|---|---|
| the mean of the training part | -0.015 | 0.006 | 1.242 | 0.984 |
| linear regression on the change of residue volume, hydropathy and charge | +0.174 | 0.003 | 1.120 | 0.879 |
| linear regression on those three columns and the one-hot codes of the two residues | +0.133 | 0.011 | 1.149 | 0.868 |
| linear regression on atom counts in shells about the site, with the one-hot codes | +0.027 | 0.029 | 1.217 | 0.952 |
| linear regression on how many atoms surround the site, with the three columns | +0.262 | 0.004 | 1.059 | 0.832 |
| the surroundings of the site, without the replaced side chain | +0.148 | 0.024 | 1.138 | 0.889 |
| the new side chain alone | +0.069 | 0.011 | 1.190 | 0.938 |
| difference field: the new side chain minus the old one | +0.148 | 0.019 | 1.138 | 0.888 |
| difference field and surroundings, joined | +0.291 | 0.008 | 1.039 | 0.817 |

Differences in q2, with the 95 % bootstrap interval over the held-out unit:

| Comparison | Difference in q2 |
|---|---|
| difference - three columns | -0.027 [-0.062, +0.006] |
| difference - substitution | +0.015 [-0.075, +0.131] |
| difference - counts | +0.121 [-0.112, +0.440] |
| difference - mean | +0.163 [+0.091, +0.228] |
| difference+context - difference | +0.143 [+0.088, +0.199] |
| difference+context - three columns | +0.116 [+0.044, +0.198] |
| difference+context - burial | +0.028 [-0.051, +0.114] |
| context - burial | -0.114 [-0.244, +0.014] |
| burial - three columns | +0.088 [-0.014, +0.178] |
| new - difference | -0.079 [-0.133, -0.022] |
| context - mean | +0.163 [+0.059, +0.249] |

## Holding out whole domains (4 of them, 4 folds)

| Description | q2 | sd over repeats | RMSE, kcal/mol | MAE, kcal/mol |
|---|---|---|---|---|
| the mean of the training part | -0.026 | 0.000 | 1.249 | 0.996 |
| linear regression on the change of residue volume, hydropathy and charge | +0.158 | 0.000 | 1.131 | 0.897 |
| linear regression on those three columns and the one-hot codes of the two residues | +0.086 | 0.000 | 1.179 | 0.895 |
| linear regression on atom counts in shells about the site, with the one-hot codes | -0.191 | 0.000 | 1.345 | 1.034 |
| linear regression on how many atoms surround the site, with the three columns | +0.156 | 0.000 | 1.133 | 0.881 |
| the surroundings of the site, without the replaced side chain | +0.051 | 0.000 | 1.201 | 0.919 |
| the new side chain alone | +0.055 | 0.000 | 1.198 | 0.958 |
| difference field: the new side chain minus the old one | +0.139 | 0.000 | 1.144 | 0.911 |
| difference field and surroundings, joined | +0.141 | 0.000 | 1.143 | 0.902 |

Differences in q2, with the 95 % bootstrap interval over the held-out unit:

| Comparison | Difference in q2 |
|---|---|
| difference - three columns | -0.019 [-0.071, +0.036] |
| difference - substitution | +0.053 [+0.007, +0.106] |
| difference - counts | +0.330 [-0.232, +0.624] |
| difference - mean | +0.165 [+0.055, +0.229] |
| difference+context - difference | +0.002 [-0.151, +0.180] |
| difference+context - three columns | -0.017 [-0.189, +0.190] |
| difference+context - burial | -0.015 [-0.113, +0.057] |
| context - burial | -0.105 [-0.250, +0.067] |
| burial - three columns | -0.002 [-0.179, +0.141] |
| new - difference | -0.084 [-0.137, -0.028] |
| context - mean | +0.077 [-0.073, +0.247] |

## The same measurement on another draw of the substitutions

Four substitutions of each position enter the measurement, and which four is a choice. This is the same measurement with the other draw, as a check that a margin is not a property of the draw.

| Description | positions held out | segments held out | domains held out |
|---|---|---|---|
| the mean of the training part | -0.010 / -0.011 | -0.015 / -0.013 | -0.026 / -0.022 |
| linear regression on the change of residue volume, hydropathy and charge | +0.181 / +0.194 | +0.174 / +0.196 | +0.158 / +0.184 |
| linear regression on those three columns and the one-hot codes of the two residues | +0.127 / +0.102 | +0.133 / +0.111 | +0.086 / +0.074 |
| linear regression on atom counts in shells about the site, with the one-hot codes | +0.049 / +0.139 | +0.027 / +0.149 | -0.191 / -0.105 |
| linear regression on how many atoms surround the site, with the three columns | +0.300 / +0.327 | +0.262 / +0.292 | +0.156 / +0.168 |
| the surroundings of the site, without the replaced side chain | +0.190 / +0.245 | +0.148 / +0.211 | +0.051 / +0.130 |
| the new side chain alone | +0.065 / +0.067 | +0.069 / +0.060 | +0.055 / +0.036 |
| difference field: the new side chain minus the old one | +0.156 / +0.150 | +0.148 / +0.160 | +0.139 / +0.140 |
| difference field and surroundings, joined | +0.311 / +0.293 | +0.291 / +0.297 | +0.141 / +0.236 |

The second draw holds 657 substitutions against 657. Each cell is the first draw and then the second.


## A second stage: every position, from the wild type alone

The wild-type structure and the label table of each of the 4 domains give 280 positions. Four substitutions of each position are drawn (1120 substitutions, ddG standard deviation 1.24 kcal/mol), on two draws. No mutant is built, so the difference field is replaced by the three substitution columns, which it equals in the first stage. A Gaussian kernel on standardised columns has the width exp(-d^2 / (2 k)), k being the number of columns, fixed in advance; the context kernel is at rho 0.1. Only the ridge and the mixing weight are chosen inside the training part.

### Draw 0

| Description | q2, positions held out | q2, segments held out | q2, domains held out |
|---|---|---|---|
| the mean of the training part | -0.005 | -0.004 | -0.019 |
| linear regression on the change of residue volume, hydropathy and charge | +0.192 | +0.194 | +0.177 |
| linear regression on six burial numbers and the three columns | +0.346 | +0.338 | +0.314 |
| Gaussian kernel on the three columns | +0.247 | +0.246 | +0.233 |
| Gaussian kernel on one burial count and the three columns | +0.458 | +0.450 | +0.428 |
| Gaussian kernel on the six burial numbers and the three columns | +0.472 | +0.464 | +0.438 |
| the surroundings of the site as a field kernel | +0.189 | +0.186 | +0.126 |
| Gaussian kernel on the three columns, joined with the field kernel of the surroundings | +0.383 | +0.381 | +0.335 |

Differences in q2, with the 95 % bootstrap interval over the held-out unit:

| Comparison | positions held out | segments held out | domains held out |
|---|---|---|---|
| columns kernel - three columns | +0.055 [+0.027, +0.084] | +0.052 [+0.021, +0.085] | +0.055 [+0.035, +0.076] |
| burial kernel - columns kernel | +0.211 [+0.149, +0.271] | +0.204 [+0.136, +0.271] | +0.195 [+0.162, +0.228] |
| burial kernel - burial | +0.112 [+0.075, +0.153] | +0.112 [+0.070, +0.160] | +0.113 [+0.071, +0.187] |
| burial kernel, all columns - columns kernel | +0.225 [+0.162, +0.288] | +0.218 [+0.147, +0.287] | +0.206 [+0.157, +0.250] |
| burial kernel - burial kernel, all columns | -0.014 [-0.040, +0.012] | -0.014 [-0.040, +0.013] | -0.011 [-0.023, +0.004] |
| context - burial | -0.157 [-0.230, -0.079] | -0.152 [-0.227, -0.074] | -0.189 [-0.234, -0.162] |
| context - burial kernel | -0.269 [-0.352, -0.182] | -0.264 [-0.341, -0.189] | -0.302 [-0.391, -0.237] |
| joined - columns kernel | +0.136 [+0.079, +0.192] | +0.135 [+0.086, +0.179] | +0.102 [+0.072, +0.143] |
| joined - burial kernel | -0.075 [-0.139, -0.013] | -0.069 [-0.139, -0.005] | -0.093 [-0.137, -0.068] |
| joined - burial kernel, all columns | -0.089 [-0.151, -0.029] | -0.083 [-0.159, -0.012] | -0.104 [-0.145, -0.075] |
| burial - three columns | +0.154 [+0.097, +0.208] | +0.144 [+0.084, +0.200] | +0.137 [+0.015, +0.207] |

### Draw 1

| Description | q2, positions held out | q2, segments held out | q2, domains held out |
|---|---|---|---|
| the mean of the training part | -0.005 | -0.004 | -0.022 |
| linear regression on the change of residue volume, hydropathy and charge | +0.178 | +0.180 | +0.161 |
| linear regression on six burial numbers and the three columns | +0.348 | +0.333 | +0.295 |
| Gaussian kernel on the three columns | +0.233 | +0.232 | +0.203 |
| Gaussian kernel on one burial count and the three columns | +0.458 | +0.451 | +0.419 |
| Gaussian kernel on the six burial numbers and the three columns | +0.483 | +0.478 | +0.434 |
| the surroundings of the site as a field kernel | +0.218 | +0.201 | +0.156 |
| Gaussian kernel on the three columns, joined with the field kernel of the surroundings | +0.385 | +0.382 | +0.338 |

Differences in q2, with the 95 % bootstrap interval over the held-out unit:

| Comparison | positions held out | segments held out | domains held out |
|---|---|---|---|
| columns kernel - three columns | +0.054 [+0.021, +0.088] | +0.051 [+0.019, +0.084] | +0.042 [+0.016, +0.079] |
| burial kernel - columns kernel | +0.225 [+0.163, +0.290] | +0.219 [+0.154, +0.287] | +0.216 [+0.163, +0.262] |
| burial kernel - burial | +0.110 [+0.066, +0.150] | +0.117 [+0.067, +0.168] | +0.123 [+0.077, +0.192] |
| burial kernel, all columns - columns kernel | +0.250 [+0.185, +0.321] | +0.246 [+0.182, +0.315] | +0.232 [+0.171, +0.285] |
| burial kernel - burial kernel, all columns | -0.025 [-0.058, +0.004] | -0.028 [-0.064, +0.005] | -0.015 [-0.030, +0.001] |
| context - burial | -0.130 [-0.205, -0.061] | -0.132 [-0.205, -0.057] | -0.140 [-0.189, -0.112] |
| context - burial kernel | -0.240 [-0.320, -0.157] | -0.249 [-0.322, -0.180] | -0.263 [-0.377, -0.188] |
| joined - columns kernel | +0.152 [+0.088, +0.213] | +0.150 [+0.095, +0.204] | +0.136 [+0.095, +0.166] |
| joined - burial kernel | -0.073 [-0.134, -0.014] | -0.069 [-0.129, -0.011] | -0.081 [-0.144, -0.041] |
| joined - burial kernel, all columns | -0.098 [-0.156, -0.042] | -0.096 [-0.157, -0.038] | -0.096 [-0.152, -0.046] |
| burial - three columns | +0.170 [+0.112, +0.230] | +0.153 [+0.096, +0.213] | +0.135 [+0.006, +0.209] |

### The width of the kernels on columns

The same measurement on the first draw with the position hold-out, at 0.5, 1.0, 2.0 times the width of the rule.

| Description | width x 0.5 | width x 1.0 | width x 2.0 |
|---|---|---|---|
| Gaussian kernel on the three columns | +0.246 | +0.247 | +0.252 |
| Gaussian kernel on one burial count and the three columns | +0.457 | +0.458 | +0.456 |
| Gaussian kernel on the six burial numbers and the three columns | +0.470 | +0.472 | +0.461 |
| Gaussian kernel on the three columns, joined with the field kernel of the surroundings | +0.381 | +0.383 | +0.385 |

| Comparison | width x 0.5 | width x 1.0 | width x 2.0 |
|---|---|---|---|
| burial kernel - columns kernel | +0.211 [+0.144, +0.275] | +0.211 [+0.149, +0.271] | +0.204 [+0.143, +0.264] |
| joined - burial kernel | -0.077 [-0.141, -0.012] | -0.075 [-0.139, -0.013] | -0.071 [-0.132, -0.010] |
