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

## The hop sample of LiTraj: a barrier as the activity, a structure as the fold

1801 hops in 393 structures of the nebBVSE122k sample (`benchmarks/run_litraj.py`), barriers with a standard deviation of 1.90 eV; 10 folds that hold out whole structures, the ridge and the width chosen on an inner split of 5 folds of the same kind. The fine level is the bond-valence calculation that labelled the hops.

| kernel | q2 | held-out RMSE (eV) | mean sd (eV) | z rms, inner folds | z rms, held out | z rms, corrected | inside 2 sd | sd vs error (rho) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | +0.572 | 1.246 | 1.136 | 1.04 | 1.05 | 1.02 | 95% | +0.20 |
| Gaussian | +0.623 | 1.169 | 1.053 | 1.01 | 0.99 | 0.98 | 96% | +0.35 |

The tolerance rule with the tolerance at the inner-fold RMSE of the training part (the error the model claims), and with a fixed 25% of the hops sent back to the calculation.

| kernel | tolerance (eV) | answered | RMSE answered | RMSE deferred | RMSE answered at 25% deferred | RMSE of that 25% |
|---|---:|---:|---:|---:|---:|---:|
| linear | 1.242 | 87% | 1.123 | 1.856 | 1.063 | 1.681 |
| Gaussian | 1.207 | 82% | 0.998 | 1.752 | 0.840 | 1.830 |

## The same deviation on the low-rank path (plan item MS6)

The same splits, the same attenuation and ridge as the table above, with the model fitted in the space of its landmarks instead of the molecules (plan item B6), so the n by n kernel is never built and the scale of the deviation is read from the m by m Gram through the Woodbury identity. *molecules* takes every training molecule as a landmark, which reproduces the exact kernel; *half* and *quarter* take that share of them, drawn at random. z rms as above; the exact path's value is repeated for the comparison, and the last columns are the largest difference from it over the splits.

| set | n | landmarks | rank | z rms, folds | exact | z rms, test | exact | inside 2 sd | largest difference, folds | largest difference, test |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DHFR | 361 | molecules | 1218 | 1.10 | 1.10 | 1.08 | 1.08 | 93% | 1.3e-11 | 6.7e-09 |
| DHFR | 361 | half | 615 | 1.04 | 1.10 | 0.98 | 1.08 | 95% | 8.4e-02 | 1.3e-01 |
| DHFR | 361 | quarter | 309 | 0.89 | 1.10 | 0.80 | 1.08 | 97% | 2.8e-01 | 3.6e-01 |
| COX2 | 282 | molecules | 847 | 0.97 | 0.97 | 0.99 | 0.99 | 93% | 2.5e-12 | 4.1e-09 |
| COX2 | 282 | half | 437 | 0.96 | 0.97 | 0.98 | 0.99 | 93% | 1.7e-02 | 2.8e-02 |
| COX2 | 282 | quarter | 221 | 0.95 | 0.97 | 0.97 | 0.99 | 94% | 4.2e-02 | 5.5e-02 |
| BZR | 147 | molecules | 445 | 1.03 | 1.03 | 0.98 | 0.98 | 94% | 7.2e-13 | 9.0e-09 |
| BZR | 147 | half | 228 | 1.01 | 1.03 | 0.97 | 0.98 | 94% | 2.5e-02 | 5.1e-02 |
| BZR | 147 | quarter | 116 | 0.96 | 1.03 | 0.94 | 0.98 | 96% | 9.0e-02 | 8.6e-02 |
| ACE | 114 | molecules | 382 | 0.99 | 0.99 | 1.15 | 1.15 | 89% | 1.5e-13 | 2.0e-09 |
| ACE | 114 | half | 196 | 0.93 | 0.99 | 1.04 | 1.15 | 94% | 1.3e-01 | 3.1e-01 |
| ACE | 114 | quarter | 98 | 0.83 | 0.99 | 0.92 | 1.15 | 98% | 2.7e-01 | 4.7e-01 |
| AChE | 111 | molecules | 293 | 1.05 | 1.05 | 0.96 | 0.96 | 94% | 1.2e-12 | 3.6e-09 |
| AChE | 111 | half | 152 | 0.92 | 1.05 | 0.80 | 0.96 | 97% | 2.0e-01 | 3.0e-01 |
| AChE | 111 | quarter | 82 | 0.74 | 1.05 | 0.67 | 0.96 | 99% | 3.6e-01 | 4.3e-01 |
| THR | 88 | molecules | 300 | 0.96 | 0.96 | 1.02 | 1.02 | 94% | 7.0e-12 | 1.1e-08 |
| THR | 88 | half | 154 | 0.92 | 0.96 | 0.94 | 1.02 | 96% | 6.5e-02 | 1.4e-01 |
| THR | 88 | quarter | 80 | 0.78 | 0.96 | 0.78 | 1.02 | 98% | 2.3e-01 | 2.9e-01 |
| THER | 76 | molecules | 265 | 1.03 | 1.03 | 1.07 | 1.07 | 93% | 3.3e-15 | 2.8e-14 |
| THER | 76 | half | 130 | 0.99 | 1.03 | 0.99 | 1.07 | 94% | 8.0e-02 | 2.9e-01 |
| THER | 76 | quarter | 65 | 0.91 | 1.03 | 0.93 | 1.07 | 94% | 1.9e-01 | 4.0e-01 |
| GPB | 66 | molecules | 230 | 0.99 | 0.99 | 1.10 | 1.10 | 92% | 2.9e-12 | 9.7e-13 |
| GPB | 66 | half | 115 | 0.84 | 0.99 | 0.82 | 1.10 | 96% | 2.8e-01 | 6.0e-01 |
| GPB | 66 | quarter | 60 | 0.79 | 0.99 | 0.56 | 1.10 | 100% | 5.4e-01 | 1.2e+00 |

The gate of item MS6 is a root mean square of z within 0.1 of the exact path's. Over every set and split, the largest difference is 1.3e-11 on the folds and 1.1e-08 on the test with the molecules; 2.8e-01 on the folds and 6.0e-01 on the test with the half; 5.4e-01 on the folds and 1.2e+00 on the test with the quarter. With the molecules as landmarks the difference is round-off, because the kernel is the same matrix computed another way; with fewer landmarks the model is another model, and the difference is what the approximation does to the calibration of its own deviation. Its direction is the same everywhere: z rms falls below one, so the deviation is larger than the error. That is the term which keeps the variance from falling to zero where the landmarks span nothing -- the exact self-kernel less the part the landmarks reproduce -- being conservative by construction, and the largest single-split differences come from the smallest sets, where a quarter of the molecules is a dozen landmarks.
