# Ranking by a confidence bound against ranking by the prediction

Early recognition of the actives of a held-out part, a molecule counting as active
when its activity is above the median of the training part. beta is how many
posterior standard deviations are subtracted from the prediction before ranking;
beta = 0 is the point estimate, which is what ranking by a regression model has
always meant. Averaged over the folds of each set, then over the sets.

## random folds

Relative spread of the posterior standard deviation across a held-out part: 0.486. A bound can only act where this is large.

The actives are about half of a held-out part here, so the enrichment factors saturate early: their ceilings are 2.05 at 5 per cent and 2.05 at 10. BEDROC is the metric that is not yet saturated.

| beta | AUC | EF 5% | EF 10% | BEDROC(20) |
|---|---:|---:|---:|---:|
| 0 | 0.857 | 1.966 | 1.874 | 0.917 |
| 0.25 | 0.861 | 1.986 | 1.874 | 0.927 |
| 0.5 | 0.864 | 1.986 | 1.888 | 0.935 |
| 1 | 0.868 | 2.019 | 1.913 | 0.951 |
| 2 | 0.868 | 2.038 | 1.940 | 0.959 |

## chemotype folds

Relative spread of the posterior standard deviation across a held-out part: 0.330. A bound can only act where this is large.

The actives are about half of a held-out part here, so the enrichment factors saturate early: their ceilings are 3.07 at 5 per cent and 2.67 at 10. BEDROC is the metric that is not yet saturated.

| beta | AUC | EF 5% | EF 10% | BEDROC(20) |
|---|---:|---:|---:|---:|
| 0 | 0.732 | 1.626 | 1.578 | 0.741 |
| 0.25 | 0.735 | 1.774 | 1.651 | 0.754 |
| 0.5 | 0.735 | 1.743 | 1.655 | 0.745 |
| 1 | 0.739 | 1.774 | 1.779 | 0.770 |
| 2 | 0.744 | 1.855 | 1.769 | 0.767 |

## The reference: the distance to the nearest training molecule (stage --reference)

The bound subtracts beta posterior standard deviations from the prediction. The reference subtracts, instead, the feature-space distance of the molecule to its nearest training molecule, rescaled to the mean standard deviation of the same held-out part (no activity is used for that). The random split is repeated with 5 seeds (seed 0 is the split above). Every number is a mean over the folds of a set and then over the sets; an interval is a 95 % bootstrap interval over the sets.

### random folds

200 folds over 8 sets. Rank correlation between the standard deviation and the distance on a held-out part: +0.783.

| Score to rank by | BEDROC(20) | AUC |
|---|---:|---:|
| prediction | 0.920 | 0.863 |
| standard deviation, beta 1 | 0.938 | 0.870 |
| nearest distance, beta 1 | 0.944 | 0.866 |
| standard deviation, beta 2 | 0.944 | 0.868 |
| nearest distance, beta 2 | 0.957 | 0.862 |

| First minus second | BEDROC(20) | sets where the first is ahead | AUC |
|---|---:|---:|---:|
| standard deviation, beta 2 minus prediction | +0.024 [+0.003, +0.048] | 7 of 8 | +0.004 [-0.005, +0.012] |
| nearest distance, beta 2 minus prediction | +0.037 [+0.014, +0.070] | 8 of 8 | -0.001 [-0.012, +0.008] |
| standard deviation, beta 2 minus nearest distance, beta 2 | -0.013 [-0.028, +0.000] | 3 of 8 | +0.005 [+0.002, +0.009] |
| standard deviation, beta 1 minus prediction | +0.018 [+0.007, +0.030] | 8 of 8 | +0.007 [+0.000, +0.013] |
| nearest distance, beta 1 minus prediction | +0.024 [+0.011, +0.042] | 8 of 8 | +0.003 [-0.004, +0.008] |
| standard deviation, beta 1 minus nearest distance, beta 1 | -0.007 [-0.014, -0.001] | 2 of 8 | +0.004 [+0.002, +0.006] |

### chemotype folds

50 folds over 8 sets. Rank correlation between the standard deviation and the distance on a held-out part: +0.745.

| Score to rank by | BEDROC(20) | AUC |
|---|---:|---:|
| prediction | 0.741 | 0.732 |
| standard deviation, beta 1 | 0.770 | 0.739 |
| nearest distance, beta 1 | 0.754 | 0.739 |
| standard deviation, beta 2 | 0.767 | 0.744 |
| nearest distance, beta 2 | 0.752 | 0.744 |

| First minus second | BEDROC(20) | sets where the first is ahead | AUC |
|---|---:|---:|---:|
| standard deviation, beta 2 minus prediction | +0.027 [+0.006, +0.051] | 6 of 8 | +0.011 [-0.018, +0.041] |
| nearest distance, beta 2 minus prediction | +0.011 [-0.029, +0.054] | 5 of 8 | +0.012 [-0.019, +0.042] |
| standard deviation, beta 2 minus nearest distance, beta 2 | +0.016 [-0.017, +0.053] | 5 of 8 | -0.000 [-0.014, +0.013] |
| standard deviation, beta 1 minus prediction | +0.030 [+0.004, +0.058] | 6 of 8 | +0.007 [-0.004, +0.019] |
| nearest distance, beta 1 minus prediction | +0.013 [-0.004, +0.038] | 5 of 8 | +0.006 [-0.006, +0.020] |
| standard deviation, beta 1 minus nearest distance, beta 1 | +0.017 [-0.007, +0.049] | 5 of 8 | +0.001 [-0.007, +0.007] |

## Sets

| set | molecules |
|---|---:|
| COX2 | 282 |
| DHFR | 361 |
| BZR | 147 |
| AChE | 111 |
| THR | 88 |
| ACE | 114 |
| GPB | 66 |
| THER | 76 |
