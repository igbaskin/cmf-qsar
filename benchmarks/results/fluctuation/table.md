# Is the fluctuation of a field worth anything? (plan item D10)

The ensemble mode uses the mean of the fields of a molecule's conformations. The variance across them is a field of the same class -- a product of two Gaussians is a Gaussian, so it is a sum of Gaussians at the midpoints of atom pairs and at twice the width -- so it can be added as a block of its own with a mixing coefficient. The reference of rule 1 is the mean field: same conformations, same folds, same ridge search.

## the 31 CBG steroids, the classic CoMFA benchmark

31 molecules, 2.6 conformations each, 1461 pseudo-atoms per variance field at attenuation 2 and a pair floor of 0.01 (99.6% of the pair weight kept). Fields in 3 s, kernels in 17 s. 10 folds.

| mixing coefficient | q2 | error | ridge |
| ---: | ---: | ---: | ---: |
| the mean alone | +0.794 | 0.482 | 0.1 |
| 0.10 | +0.785 | 0.493 | 0.01 |
| 0.25 | +0.772 | 0.508 | 0.01 |
| 0.50 | +0.743 | 0.539 | 0.1 |
| 0.75 | +0.706 | 0.576 | 0.1 |
| 0.90 | +0.721 | 0.562 | 0.0001 |
| the variance alone | +0.425 | 0.806 | 1 |

The best mixture is 0.10, q2 +0.785 against +0.794 for the mean alone: -0.009 [-0.079, +0.062].

## glycogen phosphorylase b, 66 molecules

66 molecules, 3.0 conformations each, 1065 pseudo-atoms per variance field at attenuation 2 and a pair floor of 0.01 (99.6% of the pair weight kept). Fields in 1 s, kernels in 39 s. 10 folds.

| mixing coefficient | q2 | error | ridge |
| ---: | ---: | ---: | ---: |
| the mean alone | +0.460 | 0.819 | 1 |
| 0.10 | +0.462 | 0.817 | 1 |
| 0.25 | +0.458 | 0.820 | 1 |
| 0.50 | +0.441 | 0.833 | 0.1 |
| 0.75 | +0.439 | 0.835 | 0.1 |
| 0.90 | +0.402 | 0.862 | 0.1 |
| the variance alone | +0.180 | 1.009 | 0.1 |

The best mixture is 0.10, q2 +0.462 against +0.460 for the mean alone: +0.002 [-0.015, +0.020].

## thrombin, 88 molecules

88 molecules, 3.0 conformations each, 2119 pseudo-atoms per variance field at attenuation 2 and a pair floor of 0.01 (99.1% of the pair weight kept). Fields in 14 s, kernels in 286 s. 10 folds.

| mixing coefficient | q2 | error | ridge |
| ---: | ---: | ---: | ---: |
| the mean alone | +0.308 | 0.837 | 0.0001 |
| 0.10 | +0.302 | 0.841 | 0.0001 |
| 0.25 | +0.273 | 0.858 | 0.0001 |
| 0.50 | +0.197 | 0.901 | 0.0001 |
| 0.75 | +0.088 | 0.961 | 0.01 |
| 0.90 | +0.017 | 0.997 | 1 |
| the variance alone | -0.013 | 1.013 | 1 |

The best mixture is 0.10, q2 +0.302 against +0.308 for the mean alone: -0.006 [-0.032, +0.023].

## What it comes to

**The fluctuation block does not help on any set measured here**: every interval against the mean alone contains zero or lies below it.
The variance alone reaches q2 +0.425 on steroids, q2 +0.180 on GPB, q2 -0.013 on THR. It carries something where the series is small and rigid enough for it to matter, and nothing the mean does not already have.

