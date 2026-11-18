Glass transition temperature of 7302 homopolymers (PolyMetriX, Tg 417 +- 113 K). Each repeat unit is built as a trimer capped with hydrogen. Five random folds and five folds of whole polymer classes, each repeated 3 times; the setting and the ridge are chosen by the evidence on the training block. MAE is the mean absolute error.

| Model | q², random folds | MAE, random folds (K) | q², class folds | MAE, class folds (K) |
|---|---:|---:|---:|---:|
| invariant, oligomer | +0.782 | 39.9 | +0.702 | 46.7 |
| invariant, central unit | +0.780 | 40.0 | +0.692 | 47.6 |
| topological, central unit | +0.698 | 46.3 | +0.579 | 55.0 |
| eight counts | +0.827 | 34.1 | +0.740 | 43.0 |
| PolyMetriX descriptors, kernel | +0.879 | 27.0 | +0.732 | 43.7 |
| PolyMetriX descriptors, gradient boosting | +0.844 | 33.0 | +0.770 | 41.0 |
| ECFP4 of the repeat unit | +0.903 | 23.6 | +0.779 | 39.5 |
| central unit + eight counts | +0.852 | 31.4 | +0.773 | 40.0 |
| central unit + PolyMetriX descriptors | +0.883 | 26.8 | +0.752 | 41.7 |
| topological + eight counts | +0.852 | 31.1 | +0.772 | 40.0 |

Published mean absolute errors (Fig. 6 of the PolyMetriX paper, read off the chart; gradient boosting; random 5-fold and 5 structural clusters, which are not the class folds above):

| Model | MAE, random 5-fold (K) | MAE, 5 clusters (K) |
|---|---:|---:|
| Morgan fingerprint, gradient boosting | 37.0 | 47.0 |
| PolyMetriX descriptors (all three levels), gradient boosting | 33.5 | 41.0 |

| Difference of q² | Folds | Mean [95 % interval] |
|---|---|---|
| invariant, central unit minus PolyMetriX descriptors, gradient boosting | random | -0.063 [-0.071, -0.055] |
| invariant, central unit minus PolyMetriX descriptors, gradient boosting | class | -0.074 [-0.119, -0.026] |
| central unit + PolyMetriX descriptors minus PolyMetriX descriptors, gradient boosting | random | +0.041 [+0.036, +0.047] |
| central unit + PolyMetriX descriptors minus PolyMetriX descriptors, gradient boosting | class | -0.017 [-0.039, +0.037] |
| central unit + eight counts minus PolyMetriX descriptors, gradient boosting | random | +0.009 [+0.004, +0.014] |
| central unit + eight counts minus PolyMetriX descriptors, gradient boosting | class | +0.001 [-0.021, +0.035] |
| invariant, central unit minus invariant, oligomer | random | -0.010 [-0.016, -0.004] |
| invariant, central unit minus invariant, oligomer | class | -0.010 [-0.025, +0.007] |
| invariant, central unit minus topological, central unit | random | +0.081 [+0.071, +0.091] |
| invariant, central unit minus topological, central unit | class | +0.105 [+0.069, +0.233] |
| invariant, central unit minus PolyMetriX descriptors, kernel | random | -0.100 [-0.108, -0.092] |
| invariant, central unit minus PolyMetriX descriptors, kernel | class | -0.037 [-0.094, -0.010] |
| invariant, central unit minus ECFP4 of the repeat unit | random | -0.124 [-0.131, -0.116] |
| invariant, central unit minus ECFP4 of the repeat unit | class | -0.088 [-0.147, -0.063] |
| central unit + eight counts minus eight counts | random | +0.026 [+0.022, +0.031] |
| central unit + eight counts minus eight counts | class | +0.031 [+0.019, +0.065] |
| central unit + eight counts minus invariant, central unit | random | +0.072 [+0.066, +0.078] |
| central unit + eight counts minus invariant, central unit | class | +0.076 [+0.054, +0.110] |
| central unit + eight counts minus topological + eight counts | random | +0.001 [-0.002, +0.004] |
| central unit + eight counts minus topological + eight counts | class | +0.001 [-0.008, +0.015] |
| central unit + PolyMetriX descriptors minus PolyMetriX descriptors, kernel | random | +0.004 [+0.003, +0.006] |
| central unit + PolyMetriX descriptors minus PolyMetriX descriptors, kernel | class | +0.020 [+0.010, +0.031] |
| PolyMetriX descriptors, kernel minus PolyMetriX descriptors, gradient boosting | random | +0.037 [+0.032, +0.042] |
| PolyMetriX descriptors, kernel minus PolyMetriX descriptors, gradient boosting | class | -0.037 [-0.063, +0.017] |
