# Conformal intervals on the Sutherland sets

8 sets, 10 random 70/30 splits each. Coverage is the share of held-out molecules inside the interval; width is in units of the activity, with the activity's own standard deviation beside it for scale.

| level | rule | coverage | width | width / sd |
|---|---|---:|---:|---:|
| 80% | `cv` | 80.3% | 2.28 | 1.68 |
| 80% | `cv-norm` | 84.7% | 2.95 | 2.17 |
| 80% | `split` | 80.1% | 2.59 | 1.90 |
| 80% | `split-norm` | 78.6% | 2.61 | 1.92 |
| 80% | `gp` | 77.7% **low** | 3.00 | 2.21 |
| 90% | `cv` | 89.9% | 3.10 | 2.28 |
| 90% | `cv-norm` | 93.4% | 4.07 | 2.99 |
| 90% | `split` | 90.1% | 3.61 | 2.66 |
| 90% | `split-norm` | 89.6% | 3.79 | 2.79 |
| 90% | `gp` | 83.1% **low** | 3.85 | 2.83 |
| 95% | `cv` | 95.1% | 4.09 | 3.01 |
| 95% | `cv-norm` | 97.0% | 5.28 | 3.88 |
| 95% | `split` | 94.9% | 4.04 | 2.97 |
| 95% | `split-norm` | 94.2% | 4.11 | 3.02 |
| 95% | `gp` | 86.9% **low** | 4.59 | 3.38 |

The activity's standard deviation averages 1.36 over the sets.

## By set, at 90 %

| set | n | cv | cv-norm | split | split-norm | gp |
|---|---:|---:|---:|---:|---:|---:|
| ACE | 114 | 89% | 95% | 92% | 89% | 99% |
| AChE | 111 | 89% | 92% | 93% | 93% | 75% |
| BZR | 147 | 90% | 93% | 90% | 92% | 80% |
| COX2 | 282 | 91% | 93% | 82% | 77% | 86% |
| DHFR | 361 | 91% | 94% | 90% | 89% | 84% |
| GPB | 66 | 89% | 96% | 96% | 95% | 94% |
| THER | 76 | 90% | 90% | 90% | 87% | 83% |
| THR | 88 | 90% | 94% | 89% | 94% | 63% |
