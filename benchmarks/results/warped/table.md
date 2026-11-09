# A warped Gaussian process: a monotone transform of the activity, learned with the GP (plan item ND11)
## The eight Sutherland sets

8 sets, 5 random 70/30 splits each; alpha and lambda by the evidence on the training part. Central intervals at 90%: the plain GP's, the conformal interval calibrated on the plain fit's folds, and the warped GP's, carried back through the inverse of a transform with 2 tanh terms fitted on the training part. *gain* is the log evidence the warp gained over the identity, in nats; *skew* the skewness of the activities.

| set | n | skew | gain | q2 plain | q2 warped | coverage GP | coverage conformal | coverage warped | width GP | width conformal | width warped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DHFR | 361 | +0.00 | +5.5 | +0.670 | +0.670 | 88% | 92% | 88% | 2.23 | 2.67 | 2.23 |
| COX2 | 282 | -0.42 | +9.9 | +0.508 | +0.507 | 91% | 91% | 91% | 2.72 | 2.77 | 2.70 |
| BZR | 147 | -0.63 | +15.5 | +0.343 | +0.311 | 88% | 93% | 88% | 2.04 | 2.23 | 1.98 |
| ACE | 114 | -0.15 | +17.6 | +0.529 | +0.526 | 82% | 87% | 82% | 4.53 | 4.59 | 4.20 |
| AChE | 111 | -0.21 | +8.2 | +0.534 | +0.534 | 90% | 92% | 90% | 2.73 | 3.18 | 2.73 |
| THR | 88 | -0.41 | +7.3 | +0.540 | +0.540 | 88% | 85% | 88% | 2.11 | 2.10 | 2.11 |
| THER | 76 | -0.14 | +7.5 | +0.364 | +0.364 | 88% | 90% | 88% | 5.28 | 5.46 | 5.28 |
| GPB | 66 | +1.08 | +11.2 | +0.388 | +0.406 | 87% | 86% | 86% | 2.38 | 2.37 | 2.43 |

Over the 8 sets, the warped process against the plain one (mean difference and its 95 % interval from a paired bootstrap over the sets): q2 -0.002 [-0.012, +0.006]; coverage -0.2% [-0.5%, -0.0%]; width -0.044 [-0.135, +0.012]. Against the conformal interval: coverage -1.8% [-3.7%, +0.3%], width -0.213 [-0.340, -0.082]. The warped interval is closer to the promised 90% than the plain GP's on 1 sets, further on 2, the same on 5. The warp gained +10.3 nats of evidence over the identity on average.
