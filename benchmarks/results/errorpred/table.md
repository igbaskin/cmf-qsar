# Which estimate identifies a bad prediction

8 Sutherland sets, 5 random 70/30 splits each. The training part chooses alpha and lambda by the evidence, is fitted, and calibrates the 80% conformal rule on its own out-of-fold errors; every estimate is then scored against the absolute error made on the held-out molecules.

Risk-coverage efficiency: 1 is the ordering the errors themselves give, 0 is no better than ignoring the estimate.

| set | n | GP variance | distance to training set | domain dissimilarity | conformal width |
|---|---:|---:|---:|---:|---:|
| DHFR | 361 | +0.315 | +0.232 | +0.234 | +0.234 |
| COX2 | 282 | +0.125 | +0.195 | +0.145 | +0.145 |
| BZR | 147 | +0.214 | +0.213 | +0.219 | +0.219 |
| ACE | 114 | +0.062 | +0.090 | +0.091 | +0.091 |
| AChE | 111 | +0.380 | +0.377 | +0.449 | +0.449 |
| THR | 88 | +0.329 | +0.251 | +0.255 | +0.255 |
| THER | 76 | -0.057 | -0.061 | -0.093 | -0.093 |
| GPB | 66 | +0.213 | +0.193 | +0.153 | +0.153 |
| **mean** |  | **+0.198** | **+0.186** | **+0.181** | **+0.181** |

Everything each estimate was scored on, averaged over the 40 splits:

| estimate | Spearman rho | Kendall tau | efficiency | RMSE trusted | RMSE doubted | ratio |
|---|---:|---:|---:|---:|---:|---:|
| GP variance | +0.142 | +0.094 | +0.198 | 0.835 | 1.024 | 1.38 |
| distance to training set | +0.125 | +0.086 | +0.186 | 0.853 | 1.009 | 1.31 |
| domain dissimilarity | +0.133 | +0.095 | +0.181 | 0.841 | 1.022 | 1.36 |
| conformal width | +0.133 | +0.095 | +0.181 | 0.841 | 1.022 | 1.36 |

**GP variance** comes first, at +0.198 efficiency against +0.186 for distance to training set: a paired difference of +0.011 with a 95 % interval of [-0.021, +0.044], ahead on 6 of 8 sets. For scale, the AURC of a perfect ordering is 0.423 and ignoring the estimate costs 0.946.

That interval contains zero, which is the first finding: on these sets the three estimates are **indistinguishable as orderings**, and the choice between them does not matter.

Part of the reason is that they are not three independent quantities. The **normalised conformal width and the domain measure must score identically**: the normalisation divides the interval by the difficulty, which is the distance to the training set, so the width is a monotone function of the domain and orders the molecules exactly as it does. The value of the conformal rule is that it is calibrated -- it covers what it says it covers -- and not that it orders. The others are measured rather than argued; the mean Spearman correlation between the estimates themselves, over the same splits, is:

| pair | rho |
|---|---:|
| GP variance vs conformal width | +0.853 |
| GP variance vs distance to training set | +0.799 |
| GP variance vs domain dissimilarity | +0.853 |
| distance to training set vs conformal width | +0.860 |
| distance to training set vs domain dissimilarity | +0.860 |
| domain dissimilarity vs conformal width | +1.000 |

The second finding is the size of all of them. The best captures 20% of what a perfect ordering would, and its RMSE is 0.83 on the half it trusts against 1.02 on the half it doubts, a ratio of 1.38. That is a real and useful difference -- it is what "measure the confident half first" is worth -- and it is nothing like a per-molecule error bar. On one set (THER) every estimate comes out negative, which is the fair reminder that an estimate of error is itself a model and can be wrong about a series.
