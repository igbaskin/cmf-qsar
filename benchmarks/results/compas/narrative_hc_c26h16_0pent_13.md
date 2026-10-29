# Where the predicted gap of hc_c26h16_0pent_13 comes from

Predicted gap 3.51 eV (computed 3.43 eV) = bias 1.17 eV + the sum of the ring contributions +2.34 eV.

Per-ring contributions (decomposition rule 3 summed over the atoms of every ring; a positive value raises the gap):

| ring | class | Clar share | centroid x, y (A) | contribution (eV) |
|---|---|---|---|---|
| 1 | terminal | 0.50 | +3.9, -0.9 | +3.911 |
| 3 | angular | 0.00 | +0.3, +2.3 | +1.121 |
| 2 | linear | 0.50 | +2.1, +0.6 | +0.782 |
| 6 | terminal | 1.00 | -1.8, -2.4 | -0.085 |
| 4 | angular | 1.00 | -2.1, +1.8 | -0.888 |
| 5 | angular | 0.00 | -2.9, -0.5 | -2.497 |

Rings that raise the gap: 1 (terminal), 3 (angular), 2 (linear).
Rings that lower the gap: 6 (terminal), 4 (angular), 5 (angular).

Co-field extremes in the molecular plane (z = 0) and 1 A above it, by field group (where a positive field weight raises the gap most, and lowers it most):

| field group | plane | most positive at x, y (A) | value | most negative at x, y (A) | value |
|---|---|---|---|---|---|
| steric | z=0 | +8.3, -0.4 | +0.396 | +0.1, -3.4 | -0.295 |
| electrostatic | z=0 | +0.4, +5.6 | +6.05 | +4.6, -0.8 | -7.54 |
| atom-class indicator | z=0 | +2.2, -0.4 | +0.0352 | +0.4, +4.1 | -0.0306 |
| ring-class indicator | z=0 | -1.1, -5.5 | +0.284 | +5.0, +1.0 | -0.386 |
| steric | z=1 | +8.5, -0.1 | +0.35 | -0.1, -3.4 | -0.312 |
| electrostatic | z=1 | +0.5, +5.5 | +5.77 | +4.4, -0.5 | -6.98 |
| atom-class indicator | z=1 | +1.9, -0.5 | +0.0375 | +0.7, +4.1 | -0.0276 |
| ring-class indicator | z=1 | +7.0, +5.8 | +0.293 | +5.2, +0.2 | -0.321 |

Training molecules that weigh most in this prediction (decomposition rule 2, a_j K(X, X_j)):

| training molecule | gap (eV) | contribution (eV) |
|---|---|---|
| hc_c42h24_0pent_4570 | 2.65 | -23295.999 |
| hc_c42h24_0pent_741 | 3.22 | +22731.972 |
| hc_c42h24_0pent_6151 | 2.77 | -22487.515 |
| hc_c38h22_0pent_1629 | 2.25 | -22042.859 |
| hc_c42h24_0pent_1958 | 3.42 | +21741.838 |
| hc_c42h24_0pent_4640 | 2.72 | -21497.576 |
| hc_c42h24_0pent_4625 | 2.77 | -20545.402 |
| hc_c42h24_0pent_70 | 3.50 | -20193.821 |
