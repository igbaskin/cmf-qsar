# Where the predicted gap of hc_c16h10_0pent_1 comes from

Predicted gap 4.80 eV (computed 6.09 eV) = bias -7.28 eV + the sum of the ring contributions +12.07 eV.

Per-ring contributions (decomposition rule 3 summed over the atoms of every ring; a positive value raises the gap):

| ring | class | Clar share | centroid x, y (A) | contribution (eV) |
|---|---|---|---|---|
| 1 | peri | 0.00 | +2.1, +0.0 | +3.671 |
| 3 | peri | 0.00 | -2.1, -0.0 | +3.306 |
| 2 | peri | 0.00 | +0.0, -1.2 | +2.894 |
| 4 | peri | 0.00 | -0.0, +1.2 | +2.201 |

Rings that raise the gap: 1 (peri), 3 (peri), 2 (peri), 4 (peri).

Co-field extremes in the molecular plane (z = 0) and 1 A above it, by field group (where a positive field weight raises the gap most, and lowers it most):

| field group | plane | most positive at x, y (A) | value | most negative at x, y (A) | value |
|---|---|---|---|---|---|
| steric | z=0 | -3.1, +6.3 | +0.0414 | -7.3, +0.3 | -0.043 |
| electrostatic | z=0 | -0.6, +6.3 | +6.36 | -0.7, +0.8 | -7.09 |
| atom-class indicator | z=0 | +3.8, -0.4 | +0.571 | -0.6, +3.6 | -0.422 |
| ring-class indicator | z=0 | +7.2, -4.2 | +0.503 | +4.1, -0.1 | -1.33 |
| steric | z=1 | -3.9, +6.0 | +0.0429 | -7.6, +0.0 | -0.0355 |
| electrostatic | z=1 | +1.2, -4.2 | +6.28 | -0.7, +0.9 | -4.69 |
| atom-class indicator | z=1 | +3.8, -0.7 | +0.52 | -1.3, -5.8 | -0.332 |
| ring-class indicator | z=1 | +3.5, +6.0 | +0.445 | +4.1, +0.3 | -1.31 |

Training molecules that weigh most in this prediction (decomposition rule 2, a_j K(X, X_j)):

| training molecule | gap (eV) | contribution (eV) |
|---|---|---|
| hc_c28h14_0pent_1 | 3.73 | -56427.266 |
| hc_c34h16_0pent_18 | 3.71 | -49815.434 |
| hc_c34h16_0pent_5 | 3.65 | -43989.463 |
| hc_c34h16_0pent_34 | 4.03 | -43546.519 |
| hc_c24h12_0pent_1 | 6.18 | +42836.154 |
| hc_c38h20_0pent_2063 | 5.23 | +41369.708 |
| hc_c28h14_0pent_3 | 5.81 | +41259.975 |
| hc_c36h18_0pent_233 | 5.69 | +40530.113 |
