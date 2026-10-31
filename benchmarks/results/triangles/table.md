# The three-distance autocorrelation against the two-point kernel and SOAP

`cmfqsar/triangles.py` on the same carbons, ridge grid and folds as the SOAP gate. Errors are root-mean-square, out of fold, in ppm. Differences are in q2 with a 95% bootstrap interval over molecules. `rho` is the envelope width parameter, `r_max` the cutoff of the radial grid, `sigma` the window width and `power` the kernel exponent.

## All measured carbons of the repository set (plan item LP2)

5096 carbons in 400 molecules, 10 folds.

| model | q2 | error (ppm) | ridge |
|---|---|---|---|
| atom-centred counts | +0.853 | 20.46 | 0.1 |
| two-point local kernel, rho 1 | +0.973 | 8.82 | 1e-06 |
| SOAP (elements), r_cut 3 A, l_max 8, zeta 4, sigma 0.3 | +0.988 | 5.72 | 0.01 |
| three-distance, fields, r_max 3 A, rho 0.1, sigma 0.5, scaling field, power 1 | +0.971 | 9.08 | 0.001 |
| three-distance, fields, r_max 3 A, rho 0.5, sigma 0.5, scaling field, power 1 | +0.988 | 5.90 | 0.0001 |
| three-distance, fields, r_max 4 A, rho 0.1, sigma 0.5, scaling field, power 1 | +0.961 | 10.60 | 0.001 |
| three-distance, fields, r_max 4 A, rho 0.5, sigma 0.5, scaling field, power 1 | +0.988 | 5.86 | 0.0001 |
| three-distance, fields, r_max 5 A, rho 0.1, sigma 0.5, scaling field, power 1 | +0.957 | 11.09 | 0.001 |
| three-distance, fields, r_max 5 A, rho 0.5, sigma 0.5, scaling field, power 1 | +0.988 | 5.84 | 0.0001 |
| three-distance, fields, r_max 5 A, rho 0.5, sigma 0.35, scaling field, power 1 | +0.990 | 5.45 | 0.0001 |
| three-distance, fields, r_max 5 A, rho 0.5, sigma 0.7, scaling field, power 1 | +0.985 | 6.60 | 0.0001 |
| three-distance, fields, r_max 5 A, rho 0.5, sigma 0.35, scaling pair, power 1 | +0.989 | 5.62 | 0.0001 |
| three-distance, fields, r_max 5 A, rho 0.5, sigma 0.35, scaling none, power 1 | +0.988 | 5.93 | 0.0001 |
| three-distance, fields, r_max 5 A, rho 0.5, sigma 0.35, scaling field, power 2 | +0.990 | 5.46 | 0.01 |
| three-distance, fields, r_max 5 A, rho 0.5, sigma 0.35, scaling field, power 4 | +0.989 | 5.67 | 0.01 |
| three-distance, elements, r_max 5 A, rho 0.5, sigma 0.35, power 1, scaling field | +0.989 | 5.47 | 0.0001 |
| three-distance, elements, r_max 5 A, rho 0.5, sigma 0.35, power 1, scaling pair | +0.989 | 5.70 | 0.001 |
| three-distance, elements, r_max 5 A, rho 0.5, sigma 0.35, power 1, scaling none | +0.989 | 5.58 | 0.0001 |
| three-distance, fields and elements, r_max 5 A, rho 0.5, sigma 0.35, power 1, scaling field | +0.991 | 4.96 | 0.001 |
| three-distance, fields and elements, r_max 5 A, rho 0.5, sigma 0.35, power 1, scaling pair | +0.988 | 5.74 | 0.001 |
| three-distance, fields and elements, r_max 5 A, rho 0.5, sigma 0.35, power 1, scaling none | +0.992 | 4.83 | 0.0001 |

Differences in q2:

- field channels against the two-point kernel: +0.017 [+0.014, +0.020]
- field channels against the counts: +0.137 [+0.124, +0.150]
- element channels against field channels: -0.000 [-0.002, +0.002]
- fields and elements against the fields: +0.002 [+0.001, +0.004]
- fields and elements against the elements: +0.002 [+0.001, +0.004]
- field channels against the best SOAP of the gate: +0.001 [-0.001, +0.003]
- element channels against SOAP (the basis alone): +0.001 [-0.001, +0.003]
- fields and elements against SOAP: +0.003 [+0.002, +0.005]

## Aromatic ring carbons (plan item LP3)

1225 carbons in 170 molecules, 10 folds.

| model | q2 | error (ppm) | ridge |
|---|---|---|---|
| atom-centred counts | +0.768 | 6.38 | 0.01 |
| two-point local kernel, rho 0.5 | +0.847 | 5.19 | 0.01 |
| SOAP (elements), r_cut 5 A, l_max 8, zeta 2, sigma 0.3 | +0.930 | 3.50 | 0.01 |
| three-distance, fields, r_max 3 A, rho 0.1, sigma 0.5, scaling field, power 1 | +0.890 | 4.40 | 0.01 |
| three-distance, fields, r_max 3 A, rho 0.5, sigma 0.5, scaling field, power 1 | +0.882 | 4.55 | 0.01 |
| three-distance, fields, r_max 4 A, rho 0.1, sigma 0.5, scaling field, power 1 | +0.891 | 4.38 | 0.1 |
| three-distance, fields, r_max 4 A, rho 0.5, sigma 0.5, scaling field, power 1 | +0.883 | 4.53 | 0.01 |
| three-distance, fields, r_max 5 A, rho 0.1, sigma 0.5, scaling field, power 1 | +0.891 | 4.37 | 0.1 |
| three-distance, fields, r_max 5 A, rho 0.5, sigma 0.5, scaling field, power 1 | +0.883 | 4.53 | 0.01 |
| three-distance, fields, r_max 5 A, rho 0.1, sigma 0.35, scaling field, power 1 | +0.903 | 4.13 | 0.1 |
| three-distance, fields, r_max 5 A, rho 0.1, sigma 0.7, scaling field, power 1 | +0.881 | 4.57 | 0.01 |
| three-distance, fields, r_max 5 A, rho 0.1, sigma 0.35, scaling pair, power 1 | +0.901 | 4.16 | 0.1 |
| three-distance, fields, r_max 5 A, rho 0.1, sigma 0.35, scaling none, power 1 | +0.896 | 4.27 | 0.1 |
| three-distance, fields, r_max 5 A, rho 0.1, sigma 0.35, scaling field, power 2 | +0.901 | 4.16 | 0.1 |
| three-distance, fields, r_max 5 A, rho 0.1, sigma 0.35, scaling field, power 4 | +0.878 | 4.62 | 0.01 |
| three-distance, elements, r_max 5 A, rho 0.1, sigma 0.35, power 1, scaling field | +0.912 | 3.93 | 0.01 |
| three-distance, elements, r_max 5 A, rho 0.1, sigma 0.35, power 1, scaling pair | +0.905 | 4.08 | 0.1 |
| three-distance, elements, r_max 5 A, rho 0.1, sigma 0.35, power 1, scaling none | +0.923 | 3.67 | 0.01 |
| three-distance, fields and elements, r_max 5 A, rho 0.1, sigma 0.35, power 1, scaling field | +0.915 | 3.86 | 0.1 |
| three-distance, fields and elements, r_max 5 A, rho 0.1, sigma 0.35, power 1, scaling pair | +0.912 | 3.92 | 0.1 |
| three-distance, fields and elements, r_max 5 A, rho 0.1, sigma 0.35, power 1, scaling none | +0.917 | 3.81 | 0.01 |

Differences in q2:

- field channels against the two-point kernel: +0.056 [+0.033, +0.079]
- field channels against the counts: +0.135 [+0.102, +0.172]
- element channels against field channels: +0.020 [+0.001, +0.042]
- fields and elements against the fields: +0.014 [-0.012, +0.039]
- fields and elements against the elements: -0.006 [-0.035, +0.011]
- field channels against the best SOAP of the gate: -0.027 [-0.050, -0.010]
- element channels against SOAP (the basis alone): -0.007 [-0.015, +0.001]
- fields and elements against SOAP: -0.013 [-0.044, +0.005]
