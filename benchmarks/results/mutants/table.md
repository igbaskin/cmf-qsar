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
