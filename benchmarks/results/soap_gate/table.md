# SOAP as the gate for a richer invariant local kernel

SOAP with one channel per element (dscribe), against the two-point local kernel of `cmfqsar/invariant.py`, on the same atoms, the same ridge grid and the same folds holding out whole molecules. Errors are root-mean-square, out of fold, in ppm. Differences are in q2 with a 95% bootstrap interval over molecules.

## All measured carbons of the repository set (plan item LP2)

5096 carbons in 400 molecules, 10 folds.

| model | q2 | error (ppm) | ridge |
|---|---|---|---|
| atom-centred counts | +0.853 | 20.46 | 0.1 |
| two-point local kernel, rho 0.5 | +0.962 | 10.40 | 1e-05 |
| two-point local kernel, rho 1 | +0.973 | 8.82 | 1e-06 |
| two-point local kernel, rho 2 | +0.969 | 9.46 | 1e-05 |
| SOAP r_cut 3 A, l_max 6, zeta 2 | +0.985 | 6.64 | 0.001 |
| SOAP r_cut 4 A, l_max 6, zeta 2 | +0.983 | 6.99 | 0.001 |
| SOAP r_cut 5 A, l_max 6, zeta 2 | +0.981 | 7.26 | 0.001 |
| SOAP r_cut 6 A, l_max 6, zeta 2 | +0.981 | 7.42 | 0.001 |
| SOAP r_cut 3 A, l_max 0, zeta 2 | +0.977 | 8.15 | 0.001 |
| SOAP r_cut 3 A, l_max 1, zeta 2 | +0.981 | 7.35 | 0.001 |
| SOAP r_cut 3 A, l_max 2, zeta 2 | +0.983 | 6.90 | 0.001 |
| SOAP r_cut 3 A, l_max 4, zeta 2 | +0.984 | 6.69 | 0.001 |
| SOAP r_cut 3 A, l_max 6, zeta 2 | +0.985 | 6.64 | 0.001 |
| SOAP r_cut 3 A, l_max 8, zeta 2 | +0.985 | 6.62 | 0.001 |
| SOAP r_cut 3 A, l_max 8, zeta 1, sigma 0.3 | +0.985 | 6.60 | 0.001 |
| SOAP r_cut 3 A, l_max 8, zeta 1, sigma 0.5 | +0.977 | 8.08 | 0.001 |
| SOAP r_cut 3 A, l_max 8, zeta 2, sigma 0.3 | +0.988 | 5.81 | 0.01 |
| SOAP r_cut 3 A, l_max 8, zeta 2, sigma 0.5 | +0.985 | 6.62 | 0.001 |
| SOAP r_cut 3 A, l_max 8, zeta 4, sigma 0.3 | +0.988 | 5.72 | 0.01 |
| SOAP r_cut 3 A, l_max 8, zeta 4, sigma 0.5 | +0.986 | 6.40 | 0.001 |

Differences in q2:

- best SOAP against the best two-point kernel: +0.016 [+0.013, +0.019]
- best SOAP against the counts: +0.136 [+0.124, +0.149]
- two-body SOAP (l_max 0) against the best two-point kernel: +0.004 [-0.001, +0.008]
- best SOAP with l_max > 0 against two-body SOAP, same cutoff: +0.008 [+0.006, +0.011]

## Aromatic ring carbons (plan item LP3)

1225 carbons in 170 molecules, 10 folds.

| model | q2 | error (ppm) | ridge |
|---|---|---|---|
| atom-centred counts | +0.768 | 6.38 | 0.01 |
| two-point local kernel, rho 0.25 | +0.803 | 5.88 | 0.001 |
| two-point local kernel, rho 0.5 | +0.847 | 5.19 | 0.01 |
| two-point local kernel, rho 1 | +0.836 | 5.36 | 0.01 |
| posed overlap kernel, rho 0.25 | +0.899 | 4.22 | 0.01 |
| SOAP r_cut 3 A, l_max 6, zeta 2 | +0.897 | 4.25 | 0.001 |
| SOAP r_cut 4 A, l_max 6, zeta 2 | +0.913 | 3.91 | 0.001 |
| SOAP r_cut 5 A, l_max 6, zeta 2 | +0.917 | 3.83 | 0.01 |
| SOAP r_cut 6 A, l_max 6, zeta 2 | +0.916 | 3.83 | 0.01 |
| SOAP r_cut 5 A, l_max 0, zeta 2 | +0.861 | 4.94 | 0.01 |
| SOAP r_cut 5 A, l_max 1, zeta 2 | +0.889 | 4.41 | 0.01 |
| SOAP r_cut 5 A, l_max 2, zeta 2 | +0.905 | 4.08 | 0.01 |
| SOAP r_cut 5 A, l_max 4, zeta 2 | +0.913 | 3.91 | 0.01 |
| SOAP r_cut 5 A, l_max 6, zeta 2 | +0.917 | 3.83 | 0.01 |
| SOAP r_cut 5 A, l_max 8, zeta 2 | +0.917 | 3.81 | 0.01 |
| SOAP r_cut 5 A, l_max 8, zeta 1, sigma 0.3 | +0.928 | 3.56 | 0.001 |
| SOAP r_cut 5 A, l_max 8, zeta 1, sigma 0.5 | +0.920 | 3.75 | 0.001 |
| SOAP r_cut 5 A, l_max 8, zeta 2, sigma 0.3 | +0.930 | 3.50 | 0.01 |
| SOAP r_cut 5 A, l_max 8, zeta 2, sigma 0.5 | +0.917 | 3.81 | 0.01 |
| SOAP r_cut 5 A, l_max 8, zeta 4, sigma 0.3 | +0.924 | 3.66 | 0.01 |
| SOAP r_cut 5 A, l_max 8, zeta 4, sigma 0.5 | +0.910 | 3.97 | 0.01 |

Differences in q2:

- best SOAP against the best two-point kernel: +0.083 [+0.066, +0.104]
- best SOAP against the counts: +0.162 [+0.130, +0.201]
- two-body SOAP (l_max 0) against the best two-point kernel: +0.014 [-0.005, +0.034]
- best SOAP with l_max > 0 against two-body SOAP, same cutoff: +0.057 [+0.045, +0.070]
- best SOAP against the posed overlap kernel: +0.031 [+0.015, +0.052]
