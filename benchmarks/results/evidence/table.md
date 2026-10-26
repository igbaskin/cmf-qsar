# Choosing alpha and lambda by the evidence

8 Sutherland sets, 5 random 70/30 splits each. Both criteria search the same grid on the training part alone; each choice is then fitted on that part and asked about the held-out molecules. `agree` is how often the two choose the same point of the grid.

| set | n | R²pred by CV | R²pred by evidence | difference | agree | CV seconds | evidence seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| DHFR | 361 | +0.624 | +0.637 | +0.013 | 40% | 0.1 | 0.06 |
| COX2 | 282 | +0.506 | +0.511 | +0.005 | 60% | 0.1 | 0.04 |
| BZR | 147 | +0.337 | +0.343 | +0.006 | 40% | 0.0 | 0.01 |
| ACE | 114 | +0.559 | +0.529 | -0.031 | 20% | 0.0 | 0.00 |
| AChE | 111 | +0.521 | +0.558 | +0.037 | 40% | 0.0 | 0.01 |
| THR | 88 | +0.533 | +0.540 | +0.006 | 20% | 0.0 | 0.00 |
| THER | 76 | +0.316 | +0.375 | +0.059 | 0% | 0.0 | 0.00 |
| GPB | 66 | +0.373 | +0.336 | -0.037 | 60% | 0.0 | 0.01 |
| **mean** |  |  |  | **+0.007** |  |  | |

Paired over the sets, the evidence choice is +0.007 in R²pred against the cross-validated one, 95 % interval [-0.013, +0.028], better on 6 of 8 sets. It is **4 times faster**, which is the ten folds it does not run. The kernels are built once and shared by both searches, so that ratio is the selection alone; on sets of this size the absolute times are fractions of a second either way, and what the ratio gains is a finer grid -- a per-field alpha, say -- rather than a shorter wait.

The two criteria agree on the point of the grid in 0 to 60 % of the splits and reach the same predictive quality anyway. That deserves attention: it says the surface is flat near its optimum, and that the cost of searching it matters more than which point is picked.
