# The expected field at an activity (conditional mean embedding)

Eight Sutherland sets, the joint kernel of five fields at alpha = 0.3, maps on a 0.8 A grid. Co-field: the map of kernel ridge regression. Pattern: the covariance of the field with the activity. Contrast: the expected field at the 90th percentile of the activity minus the one at the 10th (Gaussian kernel on the activity).

## How much of the field variance the activity explains, and how alike the maps are

| set | n | explained, straight line | explained, Gaussian kernel | gamma | co-field vs pattern | pattern vs contrast | co-field vs contrast |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACE | 114 | 6.7 % | 7.5 % | 0.1 | +0.60 | +1.00 | +0.60 |
| AChE | 111 | 1.7 % | 3.6 % | 3 | +0.52 | +0.92 | +0.54 |
| BZR | 147 | 2.0 % | 3.6 % | 1 | +0.47 | +0.99 | +0.43 |
| COX2 | 282 | 3.9 % | 5.7 % | 0.1 | +0.59 | +1.00 | +0.55 |
| DHFR | 361 | 2.4 % | 3.9 % | 0.3 | +0.37 | +0.99 | +0.39 |
| GPB | 66 | 3.0 % | 3.8 % | 0.03 | +0.82 | +1.00 | +0.81 |
| THER | 76 | 5.9 % | 5.9 % | 0.03 | +0.53 | +1.00 | +0.52 |
| THR | 88 | 1.2 % | 1.8 % | 0.3 | +0.72 | +0.97 | +0.72 |

The correlations are over the grid points the molecules occupy, averaged over the five fields.

## Does a map come back from the other half of the data?

Correlation between the maps of two random halves, mean over 20 cuts (standard deviation over the cuts).

| set | co-field | pattern | contrast |
|---|---:|---:|---:|
| ACE | +0.492 (0.070) | +0.850 (0.072) | +0.832 (0.089) |
| AChE | +0.408 (0.104) | +0.528 (0.166) | +0.451 (0.193) |
| BZR | +0.438 (0.102) | +0.740 (0.092) | +0.792 (0.081) |
| COX2 | +0.720 (0.069) | +0.840 (0.074) | +0.852 (0.068) |
| DHFR | +0.618 (0.032) | +0.856 (0.057) | +0.853 (0.064) |
| GPB | +0.553 (0.147) | +0.736 (0.123) | +0.743 (0.125) |
| THER | +0.388 (0.100) | +0.702 (0.096) | +0.726 (0.118) |
| THR | +0.565 (0.057) | +0.602 (0.103) | +0.570 (0.116) |
| **mean** | **+0.523** | **+0.732** | **+0.727** |

## A planted signal at one point of the steric field

5 points per set, 10 noise draws per point.

Linear signal, y = s + noise: where is the peak of the steric map?

| map | median distance of the peak from the planted point | peak within 1.5 A | map at the planted point / its peak |
|---|---:|---:|---:|
| co-field | 1.7 A | 36 % | 0.86 |
| pattern | 2.0 A | 26 % | 0.82 |
| contrast | 2.2 A | 21 % | 0.80 |

A curved signal, y = g(s) + noise: the field at the planted point that each method expects at the 90th (top) and the 10th (bottom) percentile of the activity, minus the Monte Carlo answer, in standard deviations of s. Root mean square error (mean error).

| shape g(s) | end | expected field, Gaussian kernel | expected field, straight line | mean of the tenth of the molecules at that end |
|---|---|---:|---:|---:|
| optimum, -(s - 1)^2 | top | 0.13 (-0.06) | 0.24 (-0.10) | 0.15 (+0.04) |
| optimum, -(s - 1)^2 | bottom | 0.27 (+0.11) | 0.37 (+0.18) | 0.81 (-0.18) |
| plateau, min(s, 0) | top | 0.14 (-0.05) | 0.20 (-0.03) | 0.20 (+0.03) |
| plateau, min(s, 0) | bottom | 0.24 (+0.13) | 0.29 (+0.14) | 0.51 (-0.30) |
| threshold, max(s, 0) | top | 0.30 (-0.16) | 0.39 (-0.25) | 0.76 (+0.53) |
| threshold, max(s, 0) | bottom | 0.12 (+0.02) | 0.21 (-0.04) | 0.14 (-0.03) |

## Do molecules close to the expected field of an active tend to be active?

Five-fold cross-validation, three seeds. Spearman correlation of the score with the activity / the share of the top tenth found in the top tenth of the scores.

| set | regression | expected field, Gaussian | expected field, line | mean of top tenth |
|---|---:|---:|---:|---:|
| ACE | +0.78 / 53 % | +0.46 / 22 % | +0.42 / 3 % | +0.59 / 39 % |
| AChE | +0.74 / 58 % | +0.11 / 3 % | +0.01 / 0 % | +0.38 / 14 % |
| BZR | +0.60 / 7 % | +0.19 / 13 % | +0.14 / 7 % | +0.28 / 24 % |
| COX2 | +0.75 / 50 % | +0.56 / 41 % | +0.57 / 37 % | +0.58 / 49 % |
| DHFR | +0.80 / 63 % | +0.44 / 19 % | +0.43 / 14 % | +0.42 / 22 % |
| GPB | +0.64 / 38 % | +0.49 / 50 % | +0.46 / 42 % | +0.56 / 25 % |
| THER | +0.61 / 12 % | +0.51 / 29 % | +0.51 / 25 % | +0.48 / 17 % |
| THR | +0.73 / 33 % | +0.45 / 15 % | +0.44 / 15 % | +0.52 / 7 % |
| **mean** | **+0.71 / 39 %** | **+0.40 / 24 %** | **+0.37 / 18 %** | **+0.48 / 25 %** |
