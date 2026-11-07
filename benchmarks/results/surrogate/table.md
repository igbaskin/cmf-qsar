# The predictive standard deviation as a fast level sees it

8 Sutherland sets, 5 random 70/30 splits each; alpha and lambda by the evidence on the training part. z rms is the root mean square of (measured - predicted) / sd, one for a calibrated deviation and above one for a model surer than it should be; 'corrected' multiplies the test deviations by the factor the folds of the training part gave.

| set | n | test RMSE | mean sd | z rms, folds | z rms, test | z rms, corrected | inside 2 sd | sd vs error (rho) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DHFR | 361 | 0.751 | 0.679 | 1.10 | 1.08 | 0.98 | 93% | +0.20 |
| COX2 | 282 | 0.811 | 0.825 | 0.97 | 0.99 | 1.02 | 93% | +0.04 |
| BZR | 147 | 0.610 | 0.620 | 1.03 | 0.98 | 0.96 | 94% | +0.15 |
| ACE | 114 | 1.512 | 1.378 | 0.99 | 1.15 | 1.16 | 89% | +0.04 |
| AChE | 111 | 0.809 | 0.830 | 1.05 | 0.96 | 0.92 | 94% | +0.28 |
| THR | 88 | 0.672 | 0.641 | 0.96 | 1.02 | 1.06 | 94% | +0.21 |
| THER | 76 | 1.635 | 1.605 | 1.03 | 1.07 | 1.04 | 93% | -0.12 |
| GPB | 66 | 0.767 | 0.722 | 0.99 | 1.10 | 1.11 | 92% | +0.33 |
| **mean** |  | 0.946 | 0.912 | **1.02** | **1.05** | **1.03** | 93% | +0.14 |

The tolerance rule, with the tolerance set to the cross-validated RMSE of the training part (what the model claims as its typical error): the share of the held-out structures the model answers and the RMSE on each side.  The last columns send a fixed 25% of the structures to the fine level instead.

| set | tolerance | answered | RMSE answered | RMSE deferred | RMSE answered at 25% deferred | RMSE of that 25% |
|---|---:|---:|---:|---:|---:|---:|
| DHFR | 0.805 | 80% | 0.678 | 0.989 | 0.671 | 0.944 |
| COX2 | 0.797 | 58% | 0.779 | 0.854 | 0.793 | 0.852 |
| BZR | 0.648 | 80% | 0.583 | 0.687 | 0.570 | 0.699 |
| ACE | 1.380 | 56% | 1.451 | 1.557 | 1.408 | 1.728 |
| AChE | 0.868 | 68% | 0.702 | 0.950 | 0.745 | 0.955 |
| THR | 0.626 | 59% | 0.543 | 0.841 | 0.549 | 0.921 |
| THER | 1.551 | 50% | 1.721 | 1.596 | 1.633 | 1.564 |
| GPB | 0.653 | 34% | 0.750 | 0.741 | 0.668 | 0.976 |
| **mean** | 0.916 | 61% | **0.901** | **1.027** | **0.880** | **1.080** |
