# Binding, not folding: the frame of a residue at an interface (plan item PP6)

`python benchmarks/run_binding.py`, written 2026-11-19 10:06:26, 78 minutes. SKEMPI 2.0, every single substitution with two numeric affinities: 3992 substitutions at 2793 positions of 308 complexes in 94 families (the largest family has 973 substitutions). The target is the change of binding free energy, mean +0.90 and sd 1.65 kcal/mol; 479 substitutions were measured more than once and are averaged. 475 sites have no partner atom inside the envelope. Left out: {'an affinity is a bound or missing': 283}, {'complex not prepared': 92}.

Every description is joined with the change of volume, hydropathy and charge. 8 folds, 3 repeats; every choice is made in inner folds of the same kind.

| Class of the site (SKEMPI) | Substitutions | Mean ddG, kcal/mol |
|---|---:|---:|
| core of the interface | 1666 | +1.39 |
| rim of the interface | 973 | +0.53 |
| support of the interface | 497 | +1.21 |
| surface away from the interface | 549 | +0.15 |
| interior of a protein | 307 | +0.32 |

## Whole complexes held out (308 groups)

q2, and the median over the complexes with at least 15 substitutions of the correlation of the held-out predictions with ddG inside a complex:

| Description | q2 | RMSE, kcal/mol | Median r inside a complex | Member chosen most often |
|---|---:|---:|---:|---|
| mean | -0.004 | 1.652 |  |  |
| three columns kernel | +0.043 | 1.613 | +0.192 |  |
| one count | +0.108 | 1.557 | +0.426 |  |
| two counts | +0.169 | 1.503 | +0.470 |  |
| interface class | +0.136 | 1.532 | +0.414 |  |
| accessible area | +0.098 | 1.566 | +0.458 |  |
| buried area | +0.198 | 1.476 | +0.464 |  |
| field of the surroundings | +0.095 | 1.568 | +0.286 | weight 0.5 |
| field of the partner | +0.200 | 1.474 | +0.482 | cosine, weight 0.25 |
| buried area and field of the partner | +0.225 | 1.451 | +0.488 | cosine, weight 0.25 |
| buried area and field of the surroundings | +0.216 | 1.459 | +0.474 | weight 0.25 |

Differences in q2, with the 95 % interval over the repeats and the folds; bold where it excludes zero.

| Difference | Value (interval) |
|---|---|
| one count - three columns kernel | **+0.066 (+0.032 to +0.090)** |
| buried area - three columns kernel | **+0.155 (+0.104 to +0.198)** |
| two counts - one count | **+0.061 (+0.021 to +0.095)** |
| buried area - accessible area | **+0.101 (+0.053 to +0.142)** |
| buried area - two counts | +0.029 (-0.005 to +0.070) |
| buried area - interface class | **+0.062 (+0.021 to +0.099)** |
| field of the surroundings - one count | -0.013 (-0.049 to +0.016) |
| field of the partner - two counts | +0.031 (-0.007 to +0.072) |
| field of the partner - buried area | +0.002 (-0.035 to +0.036) |
| buried area and field of the partner - buried area | +0.027 (-0.007 to +0.060) |
| buried area and field of the surroundings - buried area | +0.018 (-0.007 to +0.038) |

## Whole families of complexes held out (94 groups)

q2, and the median over the complexes with at least 15 substitutions of the correlation of the held-out predictions with ddG inside a complex:

| Description | q2 | RMSE, kcal/mol | Median r inside a complex | Member chosen most often |
|---|---:|---:|---:|---|
| mean | -0.007 | 1.654 |  |  |
| three columns kernel | -0.004 | 1.652 | +0.187 |  |
| one count | +0.068 | 1.592 | +0.411 |  |
| two counts | +0.125 | 1.542 | +0.450 |  |
| interface class | +0.094 | 1.569 | +0.413 |  |
| accessible area | +0.044 | 1.611 | +0.433 |  |
| buried area | +0.102 | 1.562 | +0.437 |  |
| field of the surroundings | -0.007 | 1.654 | +0.200 | weight 0.25 |
| field of the partner | +0.131 | 1.536 | +0.470 | cosine, weight 0.5 |
| buried area and field of the partner | +0.133 | 1.535 | +0.476 | cosine, weight 0.5 |
| buried area and field of the surroundings | +0.120 | 1.546 | +0.443 | weight 0.25 |

Differences in q2, with the 95 % interval over the repeats and the folds; bold where it excludes zero.

| Difference | Value (interval) |
|---|---|
| one count - three columns kernel | **+0.072 (+0.054 to +0.094)** |
| buried area - three columns kernel | **+0.107 (+0.076 to +0.192)** |
| two counts - one count | **+0.057 (+0.026 to +0.102)** |
| buried area - accessible area | **+0.058 (+0.028 to +0.094)** |
| buried area - two counts | -0.022 (-0.048 to +0.033) |
| buried area - interface class | +0.008 (-0.008 to +0.048) |
| field of the surroundings - one count | **-0.074 (-0.104 to -0.060)** |
| field of the partner - two counts | +0.006 (-0.003 to +0.015) |
| field of the partner - buried area | +0.029 (-0.032 to +0.056) |
| buried area and field of the partner - buried area | +0.031 (-0.002 to +0.045) |
| buried area and field of the surroundings - buried area | +0.018 (-0.010 to +0.031) |
