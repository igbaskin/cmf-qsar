# Can a low-rank kernel reproduce the exact CMF model?

Plan item B6. The exact model holds an n x n matrix per field; the Nyström approximation builds a matrix of rank m from the kernels against m landmarks. The landmarks are either a random subset of the training molecules or a cubic grid of Gaussian centres of the field's own width covering the molecules. Every q² below is 10-fold cross-validated on the same folds; `error` is the relative Frobenius error of the joint training kernel.

## At the selected attenuation, against the exact model

The attenuation and the ridge are the exact grid search's choice for the set. `q²` uses that ridge on the approximate kernel; `q² (own ridge)` re-selects the ridge on the approximate kernel, which is what a user of the approximation would do. For landmark molecules the figures are the mean over 3 random draws, with the range in brackets.

**ACE** -- 114 molecules, 42.5 atoms each, box 17.3 x 19 x 13.4 A; alpha 2, lambda 0.3, exact q²cv **+0.7143**, exact kernel 0.64 s

| landmarks | m | error | q² | q² (own ridge) | kernel time |
|---|---:|---:|---:|---:|---:|
| Gaussians, 4 A | 89 | 5.1e-01 | +0.6116 | +0.6524 (λ 1) | 0.06 s |
| Gaussians, 3 A | 144 | 1.5e-01 | +0.6629 | +0.6896 (λ 1) | 0.09 s |
| Gaussians, 2.5 A | 227 | 1.7e-01 | +0.7000 | +0.7000 (λ 0.3) | 0.19 s |
| Gaussians, 2 A | 347 | 6.4e-02 | +0.7042 | +0.7168 (λ 1) | 0.23 s |
| Gaussians, 1.5 A | 628 | 2.2e-02 | +0.7099 | +0.7126 (λ 1) | 1.13 s |
| Gaussians, 1.25 A | 921 | 1.0e-02 | +0.7106 | +0.7106 (λ 0.3) | 2.52 s |
| molecules | 10 | 4.0e-01 | +0.7076 [+0.678, +0.756] | +0.7076 | ~0.06 s |
| molecules | 20 | 2.3e-01 | +0.7234 [+0.701, +0.758] | +0.7234 | ~0.11 s |
| molecules | 40 | 1.2e-01 | +0.7135 [+0.689, +0.750] | +0.7169 | ~0.22 s |
| molecules | 80 | 3.7e-02 | +0.7032 [+0.696, +0.717] | +0.7134 | ~0.45 s |
| molecules | 114 | 1.1e-11 | +0.7143 | +0.7143 | ~0.64 s |

**AChE** -- 111 molecules, 56.3 atoms each, box 14.2 x 16.9 x 18.3 A; alpha 5, lambda 0.3, exact q²cv **+0.6107**, exact kernel 1.01 s

| landmarks | m | error | q² | q² (own ridge) | kernel time |
|---|---:|---:|---:|---:|---:|
| Gaussians, 4 A | 76 | 6.5e-01 | +0.3110 | +0.3110 (λ 0.3) | 0.08 s |
| Gaussians, 3 A | 123 | 3.6e-01 | +0.1561 | +0.3563 (λ 1) | 0.10 s |
| Gaussians, 2.5 A | 168 | 1.8e-01 | +0.5196 | +0.5196 (λ 0.3) | 0.17 s |
| Gaussians, 2 A | 248 | 1.3e-01 | +0.5697 | +0.5697 (λ 0.3) | 0.31 s |
| Gaussians, 1.5 A | 428 | 9.0e-02 | +0.6080 | +0.6080 (λ 0.3) | 0.76 s |
| Gaussians, 1.25 A | 598 | 6.7e-02 | +0.6080 | +0.6080 (λ 0.3) | 1.12 s |
| molecules | 10 | 1.6e-01 | +0.5752 [+0.551, +0.602] | +0.5977 | ~0.09 s |
| molecules | 20 | 1.0e-01 | +0.6111 [+0.594, +0.626] | +0.6141 | ~0.18 s |
| molecules | 40 | 5.4e-02 | +0.6145 [+0.604, +0.622] | +0.6145 | ~0.36 s |
| molecules | 80 | 1.5e-02 | +0.6148 [+0.610, +0.619] | +0.6148 | ~0.73 s |
| molecules | 111 | 1.7e-11 | +0.6107 | +0.6107 | ~1.01 s |

**BZR** -- 147 molecules, 36.3 atoms each, box 16.1 x 12.5 x 11 A; alpha 0.1, lambda 0.1, exact q²cv **+0.4284**, exact kernel 0.50 s

| landmarks | m | error | q² | q² (own ridge) | kernel time |
|---|---:|---:|---:|---:|---:|
| Gaussians, 4 A | 55 | 1.9e-03 | +0.4302 | +0.4302 (λ 0.1) | 0.05 s |
| Gaussians, 3 A | 87 | 1.5e-04 | +0.4285 | +0.4285 (λ 0.1) | 0.08 s |
| Gaussians, 2.5 A | 121 | 4.9e-05 | +0.4284 | +0.4284 (λ 0.1) | 0.12 s |
| Gaussians, 2 A | 190 | 8.1e-07 | +0.4284 | +0.4284 (λ 0.1) | 0.15 s |
| Gaussians, 1.5 A | 316 | 2.9e-08 | +0.4284 | +0.4284 (λ 0.1) | 0.41 s |
| Gaussians, 1.25 A | 426 | 3.1e-08 | +0.4284 | +0.4284 (λ 0.1) | 0.71 s |
| molecules | 10 | 5.0e-02 | +0.3812 [+0.357, +0.419] | +0.3812 | ~0.03 s |
| molecules | 20 | 2.0e-02 | +0.4331 [+0.429, +0.442] | +0.4334 | ~0.07 s |
| molecules | 40 | 6.8e-03 | +0.4319 [+0.428, +0.434] | +0.4319 | ~0.14 s |
| molecules | 80 | 1.4e-04 | +0.4280 [+0.428, +0.428] | +0.4280 | ~0.27 s |
| molecules | 147 | 4.5e-10 | +0.4284 | +0.4284 | ~0.50 s |

**COX2** -- 282 molecules, 40.7 atoms each, box 15 x 9.6 x 13.8 A; alpha 0.3, lambda 0.3, exact q²cv **+0.5525**, exact kernel 3.16 s

| landmarks | m | error | q² | q² (own ridge) | kernel time |
|---|---:|---:|---:|---:|---:|
| Gaussians, 4 A | 55 | 2.7e-02 | +0.5287 | +0.5297 (λ 0.1) | 0.06 s |
| Gaussians, 3 A | 85 | 7.9e-03 | +0.5568 | +0.5568 (λ 0.3) | 0.06 s |
| Gaussians, 2.5 A | 108 | 3.4e-03 | +0.5485 | +0.5485 (λ 0.3) | 0.07 s |
| Gaussians, 2 A | 169 | 4.8e-04 | +0.5517 | +0.5517 (λ 0.3) | 0.12 s |
| Gaussians, 1.5 A | 295 | 2.2e-05 | +0.5525 | +0.5525 (λ 0.3) | 0.58 s |
| Gaussians, 1.25 A | 411 | 9.2e-06 | +0.5525 | +0.5525 (λ 0.3) | 0.99 s |
| molecules | 10 | 5.0e-02 | +0.4817 [+0.437, +0.510] | +0.4848 | ~0.11 s |
| molecules | 20 | 2.8e-02 | +0.5456 [+0.538, +0.557] | +0.5464 | ~0.22 s |
| molecules | 40 | 1.1e-02 | +0.5492 [+0.545, +0.552] | +0.5492 | ~0.45 s |
| molecules | 80 | 2.7e-03 | +0.5512 [+0.550, +0.553] | +0.5512 | ~0.90 s |
| molecules | 160 | 1.0e-03 | +0.5528 [+0.553, +0.553] | +0.5528 | ~1.79 s |
| molecules | 282 | 7.5e-10 | +0.5525 | +0.5525 | ~3.16 s |

**DHFR** -- 361 molecules, 40.6 atoms each, box 13.4 x 13 x 14.5 A; alpha 0.3, lambda 0.3, exact q²cv **+0.6636**, exact kernel 4.20 s

| landmarks | m | error | q² | q² (own ridge) | kernel time |
|---|---:|---:|---:|---:|---:|
| Gaussians, 4 A | 69 | 2.9e-02 | +0.6526 | +0.6526 (λ 0.3) | 0.06 s |
| Gaussians, 3 A | 110 | 9.1e-03 | +0.6632 | +0.6632 (λ 0.3) | 0.08 s |
| Gaussians, 2.5 A | 155 | 2.0e-03 | +0.6609 | +0.6609 (λ 0.3) | 0.25 s |
| Gaussians, 2 A | 255 | 4.2e-04 | +0.6635 | +0.6635 (λ 0.3) | 0.46 s |
| Gaussians, 1.5 A | 438 | 1.5e-05 | +0.6636 | +0.6636 (λ 0.3) | 0.95 s |
| Gaussians, 1.25 A | 635 | 1.8e-06 | +0.6636 | +0.6636 (λ 0.3) | 1.71 s |
| molecules | 10 | 1.0e-01 | +0.6052 [+0.577, +0.622] | +0.6237 | ~0.12 s |
| molecules | 20 | 6.4e-02 | +0.6451 [+0.637, +0.653] | +0.6560 | ~0.23 s |
| molecules | 40 | 2.3e-02 | +0.6606 [+0.656, +0.669] | +0.6606 | ~0.47 s |
| molecules | 80 | 7.3e-03 | +0.6628 [+0.661, +0.665] | +0.6628 | ~0.93 s |
| molecules | 160 | 3.2e-04 | +0.6636 [+0.664, +0.664] | +0.6636 | ~1.86 s |
| molecules | 361 | 1.0e-09 | +0.6636 | +0.6636 | ~4.20 s |

**GPB** -- 66 molecules, 31.6 atoms each, box 14.4 x 9.8 x 13.2 A; alpha 5, lambda 1, exact q²cv **+0.5840**, exact kernel 0.07 s

| landmarks | m | error | q² | q² (own ridge) | kernel time |
|---|---:|---:|---:|---:|---:|
| Gaussians, 4 A | 46 | 4.7e-01 | +0.1887 | +0.2008 (λ 3) | 0.02 s |
| Gaussians, 3 A | 70 | 2.5e-01 | +0.5244 | +0.5354 (λ 3) | 0.05 s |
| Gaussians, 2.5 A | 86 | 2.6e-01 | +0.3626 | +0.3870 (λ 3) | 0.08 s |
| Gaussians, 2 A | 126 | 1.3e-01 | +0.5784 | +0.5784 (λ 1) | 0.06 s |
| Gaussians, 1.5 A | 219 | 1.0e-01 | +0.6045 | +0.6045 (λ 1) | 0.14 s |
| Gaussians, 1.25 A | 306 | 5.1e-02 | +0.5852 | +0.5852 (λ 1) | 0.29 s |
| molecules | 10 | 2.4e-01 | +0.5391 [+0.497, +0.573] | +0.5415 | ~0.01 s |
| molecules | 20 | 1.3e-01 | +0.5633 [+0.550, +0.573] | +0.5637 | ~0.02 s |
| molecules | 40 | 5.0e-02 | +0.5791 [+0.570, +0.589] | +0.5791 | ~0.04 s |
| molecules | 66 | 4.8e-16 | +0.5840 | +0.5840 | ~0.07 s |

**THER** -- 76 molecules, 51.7 atoms each, box 16.8 x 20.3 x 14.2 A; alpha 1, lambda 0.03, exact q²cv **+0.6093**, exact kernel 0.38 s

| landmarks | m | error | q² | q² (own ridge) | kernel time |
|---|---:|---:|---:|---:|---:|
| Gaussians, 4 A | 86 | 1.5e-01 | +0.3464 | +0.4401 (λ 0.3) | 0.07 s |
| Gaussians, 3 A | 137 | 1.1e-01 | +0.5957 | +0.5957 (λ 0.03) | 0.10 s |
| Gaussians, 2.5 A | 188 | 6.4e-02 | +0.5682 | +0.5682 (λ 0.03) | 0.13 s |
| Gaussians, 2 A | 303 | 2.1e-02 | +0.6073 | +0.6106 (λ 0.01) | 0.31 s |
| Gaussians, 1.5 A | 523 | 5.3e-03 | +0.6119 | +0.6119 (λ 0.03) | 0.89 s |
| Gaussians, 1.25 A | 728 | 1.3e-03 | +0.6096 | +0.6096 (λ 0.03) | 1.23 s |
| molecules | 10 | 1.7e-01 | +0.3535 [+0.151, +0.544] | +0.4835 | ~0.05 s |
| molecules | 20 | 7.2e-02 | +0.4968 [+0.437, +0.589] | +0.5301 | ~0.10 s |
| molecules | 40 | 1.8e-02 | +0.6046 [+0.579, +0.627] | +0.6046 | ~0.20 s |
| molecules | 76 | 7.2e-16 | +0.6093 | +0.6093 | ~0.38 s |

**THR** -- 88 molecules, 68.1 atoms each, box 16.3 x 12.2 x 15.5 A; alpha 0.1, lambda 0.01, exact q²cv **+0.7000**, exact kernel 0.75 s

| landmarks | m | error | q² | q² (own ridge) | kernel time |
|---|---:|---:|---:|---:|---:|
| Gaussians, 4 A | 79 | 8.3e-04 | +0.7053 | +0.7053 (λ 0.01) | 0.07 s |
| Gaussians, 3 A | 136 | 4.0e-05 | +0.6999 | +0.6999 (λ 0.01) | 0.12 s |
| Gaussians, 2.5 A | 178 | 2.4e-06 | +0.7000 | +0.7000 (λ 0.01) | 0.13 s |
| Gaussians, 2 A | 299 | 2.1e-07 | +0.7000 | +0.7000 (λ 0.01) | 0.34 s |
| Gaussians, 1.5 A | 511 | 6.3e-09 | +0.7000 | +0.7000 (λ 0.01) | 0.73 s |
| Gaussians, 1.25 A | 732 | 4.7e-09 | +0.7000 | +0.7000 (λ 0.01) | 1.37 s |
| molecules | 10 | 1.7e-02 | +0.5516 [+0.501, +0.616] | +0.5759 | ~0.09 s |
| molecules | 20 | 4.0e-03 | +0.6714 [+0.621, +0.699] | +0.6714 | ~0.17 s |
| molecules | 40 | 4.1e-04 | +0.7019 [+0.698, +0.709] | +0.7019 | ~0.34 s |
| molecules | 80 | 1.7e-05 | +0.7004 [+0.699, +0.702] | +0.7004 | ~0.68 s |
| molecules | 88 | 2.6e-10 | +0.7000 | +0.7000 | ~0.75 s |

### Summary

The coarsest grid of Gaussian centres whose q² (own ridge) is within 0.01 of the exact one, and the fewest landmark molecules for which every one of the 3 random draws is within 0.02 of it.

| set | n | alpha | exact q² | coarsest grid within 0.01 | m | q² | fewest molecules, every draw within 0.02 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACE | 114 | 2 | +0.714 | 2 A | 347 | +0.717 | 80 of 114 |
| AChE | 111 | 5 | +0.611 | 1.5 A | 428 | +0.608 | 20 of 111 |
| BZR | 147 | 0.1 | +0.428 | 4 A | 55 | +0.430 | 20 of 147 |
| COX2 | 282 | 0.3 | +0.552 | 3 A | 85 | +0.557 | 20 of 282 |
| DHFR | 361 | 0.3 | +0.664 | 3 A | 110 | +0.663 | 40 of 361 |
| GPB | 66 | 5 | +0.584 | 2 A | 126 | +0.578 | 40 of 66 |
| THER | 76 | 1 | +0.609 | 2 A | 303 | +0.611 | 76 of 76 |
| THR | 88 | 0.1 | +0.700 | 4 A | 79 | +0.705 | 40 of 88 |

## The whole grid search on Nyström kernels, centres 2 A apart

The same search over attenuation and ridge, once on exact kernels and once on the approximate ones. The question is whether a user of the approximation would end up with the same model.

| set | n | m | exact: alpha, λ, q² | Nyström: alpha, λ, q² | same choice | exact search | Nyström search |
|---|---:|---:|---|---|---|---:|---:|
| ACE | 114 | 347 | 2, 0.3, +0.7143 | 5, 0.3, +0.7273 | no | 2.7 s | 2.5 s |
| AChE | 111 | 248 | 5, 0.3, +0.6107 | 2, 0.3, +0.6000 | no | 3.8 s | 1.9 s |
| BZR | 147 | 190 | 0.1, 0.1, +0.4284 | 0.1, 0.1, +0.4284 | yes | 3.0 s | 1.3 s |
| COX2 | 282 | 169 | 0.3, 0.3, +0.5525 | 0.3, 0.3, +0.5517 | yes | 13.5 s | 1.3 s |
| DHFR | 361 | 255 | 0.3, 0.3, +0.6636 | 0.3, 0.3, +0.6635 | yes | 22.5 s | 3.3 s |
| GPB | 66 | 126 | 5, 1, +0.5840 | 5, 1, +0.5784 | yes | 0.9 s | 0.5 s |
| THER | 76 | 303 | 1, 0.03, +0.6093 | 1, 0.01, +0.6106 | no | 1.5 s | 2.7 s |
| THR | 88 | 299 | 0.1, 0.01, +0.7000 | 0.1, 0.01, +0.7000 | yes | 4.2 s | 2.3 s |

## A fixed grid of centres against the attenuation

A Gaussian of attenuation alpha has width 1/sqrt(alpha): 3.2 A at 0.1, 1.4 A at 0.5, 0.45 A at 5. A grid of centres spans the fields only while its spacing is not much larger than that width, so the approximation is expected to fail at strong attenuation whatever the number of molecules. The ridge is the exact model's best at each attenuation.

**COX2** -- 282 molecules

| alpha | width (A) | exact q² | 2 A: m, error, q² | 1.5 A: m, error, q² |
|---:|---:|---:|---|---|
| 0.1 | 3.16 | +0.549 | 169, 2e-06, +0.549 | 295, 5e-08, +0.549 |
| 0.3 | 1.83 | +0.552 | 169, 5e-04, +0.552 | 295, 2e-05, +0.552 |
| 0.5 | 1.41 | +0.541 | 169, 3e-03, +0.541 | 295, 2e-04, +0.541 |
| 1 | 1.00 | +0.525 | 169, 2e-02, +0.526 | 295, 3e-03, +0.525 |
| 2 | 0.71 | +0.526 | 169, 8e-02, +0.514 | 295, 2e-02, +0.521 |
| 5 | 0.45 | +0.536 | 169, 1e-01, +0.498 | 295, 9e-02, +0.520 |

**THR** -- 88 molecules

| alpha | width (A) | exact q² | 2 A: m, error, q² | 1.5 A: m, error, q² |
|---:|---:|---:|---|---|
| 0.1 | 3.16 | +0.700 | 299, 2e-07, +0.700 | 511, 6e-09, +0.700 |
| 0.3 | 1.83 | +0.678 | 299, 9e-05, +0.678 | 511, 3e-06, +0.678 |
| 0.5 | 1.41 | +0.679 | 299, 7e-04, +0.680 | 511, 6e-05, +0.679 |
| 1 | 1.00 | +0.682 | 299, 6e-03, +0.685 | 511, 2e-03, +0.685 |
| 2 | 0.71 | +0.671 | 299, 2e-02, +0.677 | 511, 1e-02, +0.687 |
| 5 | 0.45 | +0.643 | 299, 6e-02, +0.629 | 511, 6e-02, +0.675 |

