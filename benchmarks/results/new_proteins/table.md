# Does a description of a site carry to a new protein? (plan item PP2)

`python benchmarks/run_new_proteins.py`. 24 domains of the mega-scale set (1O6X, 1UFM, 2K5N, 3ZGK, 1AOY, 5KPH, 2M8U, 1QKH, 2JT1, 1F0M, 3DKM, 2LO1, 2O2W, 1V1C, 2KVS, 1YU5, 2KCM, 1ORC, 2LYR, 2KXD, 4C26, 2MCH, 1R69, 1GYZ), 1597 positions, 4 substitutions drawn per position. Whole domains are held out, 8 folds, 3 repeats; every choice is made in inner folds of the same kind. Every description is joined with the change of volume, hydropathy and charge.

## Stage 1: descriptions from the wild-type structures

Written 2026-11-18 21:12:32, 193 minutes.

### Draw 0: 6386 substitutions at 1597 positions, sd of ddG 1.19 kcal/mol

q2 with domains held out, and the median over the domains of the correlation of the held-out predictions with ddG inside a domain (with the lowest and the highest domain):

| Description | q2 | Median r inside a domain | Lowest and highest |
|---|---:|---:|---|
| mean | -0.002 | | |
| three columns kernel | +0.217 | +0.528 | +0.225 to +0.683 |
| one count | +0.421 | +0.685 | +0.511 to +0.781 |
| accessible area | +0.490 | +0.751 | +0.626 to +0.819 |
| field of the surroundings | +0.427 | +0.657 | +0.581 to +0.755 |
| accessible area and field | +0.510 | +0.753 | +0.669 to +0.829 |

Differences in q2, with the 95 % interval over the repeats and the folds; bold where it excludes zero.

| Difference | Value (interval) |
|---|---|
| one count - three columns kernel | **+0.203 (+0.168 to +0.235)** |
| accessible area - three columns kernel | **+0.273 (+0.237 to +0.307)** |
| field of the surroundings - three columns kernel | **+0.209 (+0.167 to +0.253)** |
| field of the surroundings - one count | +0.006 (-0.025 to +0.041) |
| accessible area - one count | **+0.070 (+0.042 to +0.097)** |
| accessible area and field - accessible area | +0.020 (-0.001 to +0.040) |

### Draw 1: 6386 substitutions at 1597 positions, sd of ddG 1.19 kcal/mol

q2 with domains held out, and the median over the domains of the correlation of the held-out predictions with ddG inside a domain (with the lowest and the highest domain):

| Description | q2 | Median r inside a domain | Lowest and highest |
|---|---:|---:|---|
| mean | -0.002 | | |
| three columns kernel | +0.207 | +0.485 | +0.156 to +0.740 |
| one count | +0.413 | +0.671 | +0.519 to +0.789 |
| accessible area | +0.471 | +0.731 | +0.595 to +0.818 |
| field of the surroundings | +0.422 | +0.665 | +0.524 to +0.755 |
| accessible area and field | +0.499 | +0.744 | +0.632 to +0.830 |

Differences in q2, with the 95 % interval over the repeats and the folds; bold where it excludes zero.

| Difference | Value (interval) |
|---|---|
| one count - three columns kernel | **+0.206 (+0.172 to +0.236)** |
| accessible area - three columns kernel | **+0.265 (+0.225 to +0.301)** |
| field of the surroundings - three columns kernel | **+0.216 (+0.175 to +0.257)** |
| field of the surroundings - one count | +0.009 (-0.020 to +0.040) |
| accessible area - one count | **+0.058 (+0.033 to +0.082)** |
| accessible area and field - accessible area | **+0.028 (+0.007 to +0.048)** |

## Stage 2: the energy of the built mutants (plan item PP4)

Written 2026-11-18 23:54:58, 108 minutes.

### Draw 0: 6386 substitutions at 1597 positions, sd of ddG 1.19 kcal/mol

q2 with domains held out, and the median over the domains of the correlation of the held-out predictions with ddG inside a domain (with the lowest and the highest domain):

| Description | q2 | Median r inside a domain | Lowest and highest |
|---|---:|---:|---|
| mean | -0.002 | | |
| three columns kernel | +0.217 | +0.528 | +0.225 to +0.683 |
| one count | +0.421 | +0.685 | +0.511 to +0.781 |
| accessible area | +0.490 | +0.751 | +0.626 to +0.819 |
| energy terms | +0.365 | +0.643 | +0.493 to +0.738 |
| one count and energy terms | +0.505 | +0.750 | +0.588 to +0.813 |
| accessible area and energy terms | +0.524 | +0.754 | +0.655 to +0.842 |

Differences in q2, with the 95 % interval over the repeats and the folds; bold where it excludes zero.

| Difference | Value (interval) |
|---|---|
| energy terms - three columns kernel | **+0.148 (+0.119 to +0.171)** |
| one count and energy terms - one count | **+0.085 (+0.061 to +0.103)** |
| accessible area and energy terms - accessible area | **+0.034 (+0.016 to +0.049)** |
| one count and energy terms - accessible area | +0.015 (-0.014 to +0.042) |

### Draw 1: 6386 substitutions at 1597 positions, sd of ddG 1.19 kcal/mol

q2 with domains held out, and the median over the domains of the correlation of the held-out predictions with ddG inside a domain (with the lowest and the highest domain):

| Description | q2 | Median r inside a domain | Lowest and highest |
|---|---:|---:|---|
| mean | -0.002 | | |
| three columns kernel | +0.207 | +0.485 | +0.156 to +0.740 |
| one count | +0.413 | +0.671 | +0.519 to +0.789 |
| accessible area | +0.471 | +0.731 | +0.595 to +0.818 |
| energy terms | +0.366 | +0.625 | +0.477 to +0.754 |
| one count and energy terms | +0.499 | +0.736 | +0.607 to +0.849 |
| accessible area and energy terms | +0.519 | +0.765 | +0.645 to +0.865 |

Differences in q2, with the 95 % interval over the repeats and the folds; bold where it excludes zero.

| Difference | Value (interval) |
|---|---|
| energy terms - three columns kernel | **+0.159 (+0.132 to +0.183)** |
| one count and energy terms - one count | **+0.086 (+0.068 to +0.102)** |
| accessible area and energy terms - accessible area | **+0.048 (+0.032 to +0.062)** |
| one count and energy terms - accessible area | **+0.027 (+0.002 to +0.050)** |
