# Ranking by a confidence bound against ranking by the prediction

Early recognition of the actives of a held-out part, a molecule counting as active
when its activity is above the median of the training part. beta is how many
posterior standard deviations are subtracted from the prediction before ranking;
beta = 0 is the point estimate, which is what ranking by a regression model has
always meant. Averaged over the folds of each set, then over the sets.

## random folds

Relative spread of the posterior standard deviation across a held-out part: 0.486. A bound can only act where this is large.

The actives are about half of a held-out part here, so the enrichment factors saturate early: their ceilings are 2.05 at 5 per cent and 2.05 at 10. BEDROC is the metric with room left.

| beta | AUC | EF 5% | EF 10% | BEDROC(20) |
|---|---:|---:|---:|---:|
| 0 | 0.857 | 1.966 | 1.874 | 0.917 |
| 0.25 | 0.861 | 1.986 | 1.874 | 0.927 |
| 0.5 | 0.864 | 1.986 | 1.888 | 0.935 |
| 1 | 0.868 | 2.019 | 1.913 | 0.951 |
| 2 | 0.868 | 2.038 | 1.940 | 0.959 |

## chemotype folds

Relative spread of the posterior standard deviation across a held-out part: 0.330. A bound can only act where this is large.

The actives are about half of a held-out part here, so the enrichment factors saturate early: their ceilings are 3.07 at 5 per cent and 2.67 at 10. BEDROC is the metric with room left.

| beta | AUC | EF 5% | EF 10% | BEDROC(20) |
|---|---:|---:|---:|---:|
| 0 | 0.732 | 1.626 | 1.578 | 0.741 |
| 0.25 | 0.735 | 1.774 | 1.651 | 0.754 |
| 0.5 | 0.735 | 1.743 | 1.655 | 0.745 |
| 1 | 0.739 | 1.774 | 1.779 | 0.770 |
| 2 | 0.744 | 1.855 | 1.769 | 0.767 |

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
