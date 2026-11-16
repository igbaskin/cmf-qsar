# Random Fourier features of the field kernel

The field kernel is an overlap integral, so it is also an integral of the two fields' Fourier transforms over frequencies. Each of D frequencies gives a molecule two features per field (`cmfqsar/rff.py`). Every q² is 10-fold cross-validated on the folds of the Nyström benchmark (`results/nystrom`); `error` is the relative Frobenius error of the joint training kernel. `columns` is the number of feature columns per field: 2 D for frequencies, m for m Nyström landmarks.

## At the selected attenuation, against the exact model and Nyström

Random frequencies: the mean over 3 seeds. `mc`: independent draws; `orthogonal`: orthogonal triples; `qmc`: a scrambled Sobol sequence. `lattice`: a cubic lattice of frequencies, cut where the spectrum falls below the tolerance, with a period at which the periodic images overlap the molecules by less than the same tolerance. Nyström rows are copied from `results/nystrom/table.md` (landmark molecules: mean of 3 random draws).

**ACE** -- 114 molecules; alpha 2, lambda 0.3, exact q² **+0.7143**, exact kernel 0.78 s

| features | columns per field | error | q² | q² (own ridge) | time |
|---|---:|---:|---:|---:|---:|
| Fourier, mc | 64 | 1.4e-01 | +0.6580 | +0.6822 | 0.02 s |
| Fourier, mc | 256 | 6.9e-02 | +0.6961 | +0.7034 | 0.11 s |
| Fourier, mc | 1024 | 2.9e-02 | +0.7160 | +0.7161 | 0.67 s |
| Fourier, mc | 2048 | 2.2e-02 | +0.7167 | +0.7168 | 1.13 s |
| Fourier, orthogonal | 64 | 1.4e-01 | +0.6401 | +0.6764 | 0.02 s |
| Fourier, orthogonal | 256 | 6.5e-02 | +0.7014 | +0.7055 | 0.10 s |
| Fourier, orthogonal | 1024 | 3.4e-02 | +0.7120 | +0.7124 | 0.69 s |
| Fourier, orthogonal | 2048 | 2.1e-02 | +0.7095 | +0.7109 | 1.14 s |
| Fourier, qmc | 64 | 1.2e-01 | +0.6319 | +0.6623 | 0.02 s |
| Fourier, qmc | 256 | 4.3e-02 | +0.6932 | +0.6967 | 0.09 s |
| Fourier, qmc | 1024 | 1.7e-02 | +0.7141 | +0.7141 | 0.65 s |
| Fourier, qmc | 2048 | 1.0e-02 | +0.7159 | +0.7159 | 1.07 s |
| Fourier, lattice (tolerance 0.001) | 4730/10252 | 6.9e-04 | +0.7140 | +0.7140 | 4.09 s |
| Fourier, lattice (tolerance 1e-06) | 18614/35034 | 1.1e-06 | +0.7143 | +0.7143 | 13.57 s |
| Nyström, molecules | 10 | 4.0e-01 | +0.7076 | +0.7076 | ~0.06 s |
| Nyström, molecules | 20 | 2.3e-01 | +0.7234 | +0.7234 | ~0.11 s |
| Nyström, molecules | 40 | 1.2e-01 | +0.7135 | +0.7169 | ~0.22 s |
| Nyström, molecules | 80 | 3.7e-02 | +0.7032 | +0.7134 | ~0.45 s |
| Nyström, Gaussian centres 2 A | 347 | 6.4e-02 | +0.7042 | +0.7168 | 0.23 s |
| Nyström, Gaussian centres 1.5 A | 628 | 2.2e-02 | +0.7099 | +0.7126 | 1.13 s |

**AChE** -- 111 molecules; alpha 5, lambda 0.3, exact q² **+0.6107**, exact kernel 0.88 s

| features | columns per field | error | q² | q² (own ridge) | time |
|---|---:|---:|---:|---:|---:|
| Fourier, mc | 64 | 7.7e-02 | +0.5786 | +0.5786 | 0.03 s |
| Fourier, mc | 256 | 3.3e-02 | +0.5934 | +0.5934 | 0.12 s |
| Fourier, mc | 1024 | 1.5e-02 | +0.6037 | +0.6037 | 0.88 s |
| Fourier, mc | 2048 | 1.1e-02 | +0.6105 | +0.6105 | 1.48 s |
| Fourier, orthogonal | 64 | 6.4e-02 | +0.5565 | +0.5659 | 0.03 s |
| Fourier, orthogonal | 256 | 3.0e-02 | +0.6038 | +0.6038 | 0.15 s |
| Fourier, orthogonal | 1024 | 1.6e-02 | +0.6056 | +0.6056 | 0.86 s |
| Fourier, orthogonal | 2048 | 1.1e-02 | +0.6085 | +0.6085 | 1.38 s |
| Fourier, qmc | 64 | 5.4e-02 | +0.5508 | +0.5591 | 0.02 s |
| Fourier, qmc | 256 | 2.2e-02 | +0.5980 | +0.5980 | 0.10 s |
| Fourier, qmc | 1024 | 1.3e-02 | +0.6015 | +0.6015 | 0.82 s |
| Fourier, qmc | 2048 | 8.8e-03 | +0.6047 | +0.6047 | 1.30 s |
| Fourier, lattice (tolerance 0.001) | 12294/30096 | 5.4e-04 | +0.6107 | +0.6107 | 15.32 s |
| Fourier, lattice (tolerance 1e-06) | 43508/97714 | 1.2e-06 | +0.6107 | +0.6107 | 51.01 s |
| Nyström, molecules | 10 | 1.6e-01 | +0.5752 | +0.5977 | ~0.09 s |
| Nyström, molecules | 20 | 1.0e-01 | +0.6111 | +0.6141 | ~0.18 s |
| Nyström, molecules | 40 | 5.4e-02 | +0.6145 | +0.6145 | ~0.36 s |
| Nyström, molecules | 80 | 1.5e-02 | +0.6148 | +0.6148 | ~0.73 s |
| Nyström, Gaussian centres 2 A | 248 | 1.3e-01 | +0.5697 | +0.5697 | 0.31 s |
| Nyström, Gaussian centres 1.5 A | 428 | 9.0e-02 | +0.6080 | +0.6080 | 0.76 s |

**BZR** -- 147 molecules; alpha 0.1, lambda 0.1, exact q² **+0.4284**, exact kernel 0.55 s

| features | columns per field | error | q² | q² (own ridge) | time |
|---|---:|---:|---:|---:|---:|
| Fourier, mc | 64 | 4.3e-02 | +0.4202 | +0.4202 | 0.01 s |
| Fourier, mc | 256 | 2.5e-02 | +0.4190 | +0.4190 | 0.08 s |
| Fourier, mc | 1024 | 9.7e-03 | +0.4237 | +0.4237 | 0.63 s |
| Fourier, mc | 2048 | 6.6e-03 | +0.4237 | +0.4237 | 1.08 s |
| Fourier, orthogonal | 64 | 3.6e-02 | +0.4229 | +0.4229 | 0.01 s |
| Fourier, orthogonal | 256 | 1.6e-02 | +0.4314 | +0.4314 | 0.09 s |
| Fourier, orthogonal | 1024 | 8.6e-03 | +0.4241 | +0.4241 | 0.68 s |
| Fourier, orthogonal | 2048 | 6.6e-03 | +0.4232 | +0.4232 | 1.14 s |
| Fourier, qmc | 64 | 2.2e-02 | +0.4259 | +0.4259 | 0.02 s |
| Fourier, qmc | 256 | 6.0e-03 | +0.4307 | +0.4307 | 0.09 s |
| Fourier, qmc | 1024 | 2.1e-03 | +0.4252 | +0.4252 | 0.59 s |
| Fourier, qmc | 2048 | 1.3e-03 | +0.4261 | +0.4261 | 0.93 s |
| Fourier, lattice (tolerance 0.001) | 306/342 | 9.7e-04 | +0.4287 | +0.4287 | 0.08 s |
| Fourier, lattice (tolerance 1e-06) | 1744/1840 | 1.9e-06 | +0.4284 | +0.4284 | 0.89 s |
| Nyström, molecules | 10 | 5.0e-02 | +0.3812 | +0.3812 | ~0.03 s |
| Nyström, molecules | 20 | 2.0e-02 | +0.4331 | +0.4334 | ~0.07 s |
| Nyström, molecules | 40 | 6.8e-03 | +0.4319 | +0.4319 | ~0.14 s |
| Nyström, molecules | 80 | 1.4e-04 | +0.4280 | +0.4280 | ~0.27 s |
| Nyström, Gaussian centres 2 A | 190 | 8.1e-07 | +0.4284 | +0.4284 | 0.15 s |
| Nyström, Gaussian centres 1.5 A | 316 | 2.9e-08 | +0.4284 | +0.4284 | 0.41 s |

**COX2** -- 282 molecules; alpha 0.3, lambda 0.3, exact q² **+0.5525**, exact kernel 2.98 s

| features | columns per field | error | q² | q² (own ridge) | time |
|---|---:|---:|---:|---:|---:|
| Fourier, mc | 64 | 4.5e-02 | +0.5472 | +0.5483 | 0.05 s |
| Fourier, mc | 256 | 2.0e-02 | +0.5497 | +0.5510 | 0.30 s |
| Fourier, mc | 1024 | 1.2e-02 | +0.5542 | +0.5542 | 1.22 s |
| Fourier, mc | 2048 | 8.8e-03 | +0.5530 | +0.5530 | 2.42 s |
| Fourier, orthogonal | 64 | 5.2e-02 | +0.5404 | +0.5404 | 0.05 s |
| Fourier, orthogonal | 256 | 2.3e-02 | +0.5484 | +0.5515 | 0.28 s |
| Fourier, orthogonal | 1024 | 1.1e-02 | +0.5506 | +0.5506 | 1.10 s |
| Fourier, orthogonal | 2048 | 8.2e-03 | +0.5518 | +0.5518 | 2.19 s |
| Fourier, qmc | 64 | 3.1e-02 | +0.5446 | +0.5486 | 0.05 s |
| Fourier, qmc | 256 | 1.3e-02 | +0.5545 | +0.5545 | 0.27 s |
| Fourier, qmc | 1024 | 4.8e-03 | +0.5528 | +0.5528 | 1.04 s |
| Fourier, qmc | 2048 | 3.5e-03 | +0.5553 | +0.5553 | 2.22 s |
| Fourier, lattice (tolerance 0.001) | 516/740 | 9.9e-04 | +0.5529 | +0.5529 | 0.60 s |
| Fourier, lattice (tolerance 1e-06) | 2946/3288 | 2.4e-06 | +0.5525 | +0.5525 | 3.06 s |
| Nyström, molecules | 10 | 5.0e-02 | +0.4817 | +0.4848 | ~0.11 s |
| Nyström, molecules | 20 | 2.8e-02 | +0.5456 | +0.5464 | ~0.22 s |
| Nyström, molecules | 40 | 1.1e-02 | +0.5492 | +0.5492 | ~0.45 s |
| Nyström, molecules | 80 | 2.7e-03 | +0.5512 | +0.5512 | ~0.90 s |
| Nyström, molecules | 160 | 1.0e-03 | +0.5528 | +0.5528 | ~1.79 s |
| Nyström, Gaussian centres 2 A | 169 | 4.8e-04 | +0.5517 | +0.5517 | 0.12 s |
| Nyström, Gaussian centres 1.5 A | 295 | 2.2e-05 | +0.5525 | +0.5525 | 0.58 s |

**DHFR** -- 361 molecules; alpha 0.3, lambda 0.3, exact q² **+0.6636**, exact kernel 4.51 s

| features | columns per field | error | q² | q² (own ridge) | time |
|---|---:|---:|---:|---:|---:|
| Fourier, mc | 64 | 4.9e-02 | +0.6558 | +0.6598 | 0.07 s |
| Fourier, mc | 256 | 2.2e-02 | +0.6587 | +0.6595 | 0.35 s |
| Fourier, mc | 1024 | 1.0e-02 | +0.6638 | +0.6638 | 1.36 s |
| Fourier, mc | 2048 | 6.7e-03 | +0.6641 | +0.6641 | 3.15 s |
| Fourier, orthogonal | 64 | 4.2e-02 | +0.6430 | +0.6430 | 0.09 s |
| Fourier, orthogonal | 256 | 2.1e-02 | +0.6613 | +0.6616 | 0.40 s |
| Fourier, orthogonal | 1024 | 1.1e-02 | +0.6614 | +0.6614 | 1.49 s |
| Fourier, orthogonal | 2048 | 6.7e-03 | +0.6626 | +0.6626 | 2.97 s |
| Fourier, qmc | 64 | 3.8e-02 | +0.6519 | +0.6519 | 0.07 s |
| Fourier, qmc | 256 | 1.4e-02 | +0.6623 | +0.6623 | 0.40 s |
| Fourier, qmc | 1024 | 4.5e-03 | +0.6604 | +0.6604 | 1.45 s |
| Fourier, qmc | 2048 | 2.7e-03 | +0.6622 | +0.6622 | 2.73 s |
| Fourier, lattice (tolerance 0.001) | 516/740 | 1.0e-03 | +0.6631 | +0.6631 | 0.94 s |
| Fourier, lattice (tolerance 1e-06) | 2802/3120 | 2.9e-06 | +0.6636 | +0.6636 | 3.87 s |
| Nyström, molecules | 10 | 1.0e-01 | +0.6052 | +0.6237 | ~0.12 s |
| Nyström, molecules | 20 | 6.4e-02 | +0.6451 | +0.6560 | ~0.23 s |
| Nyström, molecules | 40 | 2.3e-02 | +0.6606 | +0.6606 | ~0.47 s |
| Nyström, molecules | 80 | 7.3e-03 | +0.6628 | +0.6628 | ~0.93 s |
| Nyström, molecules | 160 | 3.2e-04 | +0.6636 | +0.6636 | ~1.86 s |
| Nyström, Gaussian centres 2 A | 255 | 4.2e-04 | +0.6635 | +0.6635 | 0.46 s |
| Nyström, Gaussian centres 1.5 A | 438 | 1.5e-05 | +0.6636 | +0.6636 | 0.95 s |

**GPB** -- 66 molecules; alpha 5, lambda 1, exact q² **+0.5840**, exact kernel 0.11 s

| features | columns per field | error | q² | q² (own ridge) | time |
|---|---:|---:|---:|---:|---:|
| Fourier, mc | 64 | 9.2e-02 | +0.5553 | +0.5616 | 0.01 s |
| Fourier, mc | 256 | 3.9e-02 | +0.5588 | +0.5602 | 0.03 s |
| Fourier, mc | 1024 | 2.2e-02 | +0.5788 | +0.5788 | 0.24 s |
| Fourier, mc | 2048 | 1.3e-02 | +0.5796 | +0.5796 | 0.48 s |
| Fourier, orthogonal | 64 | 1.0e-01 | +0.5911 | +0.6123 | 0.01 s |
| Fourier, orthogonal | 256 | 4.8e-02 | +0.5875 | +0.5880 | 0.02 s |
| Fourier, orthogonal | 1024 | 2.0e-02 | +0.5834 | +0.5834 | 0.23 s |
| Fourier, orthogonal | 2048 | 1.4e-02 | +0.5858 | +0.5858 | 0.52 s |
| Fourier, qmc | 64 | 8.4e-02 | +0.5831 | +0.5875 | 0.01 s |
| Fourier, qmc | 256 | 3.6e-02 | +0.5830 | +0.5883 | 0.03 s |
| Fourier, qmc | 1024 | 1.5e-02 | +0.5852 | +0.5852 | 0.26 s |
| Fourier, qmc | 2048 | 9.3e-03 | +0.5831 | +0.5831 | 0.51 s |
| Fourier, lattice (tolerance 0.001) | 7124/16136 | 8.6e-04 | +0.5839 | +0.5839 | 2.27 s |
| Fourier, lattice (tolerance 1e-06) | 26746/54340 | 1.6e-06 | +0.5840 | +0.5840 | 7.96 s |
| Nyström, molecules | 10 | 2.4e-01 | +0.5391 | +0.5415 | ~0.01 s |
| Nyström, molecules | 20 | 1.3e-01 | +0.5633 | +0.5637 | ~0.02 s |
| Nyström, molecules | 40 | 5.0e-02 | +0.5791 | +0.5791 | ~0.04 s |
| Nyström, Gaussian centres 2 A | 126 | 1.3e-01 | +0.5784 | +0.5784 | 0.06 s |
| Nyström, Gaussian centres 1.5 A | 219 | 1.0e-01 | +0.6045 | +0.6045 | 0.14 s |

**THER** -- 76 molecules; alpha 1, lambda 0.03, exact q² **+0.6093**, exact kernel 0.34 s

| features | columns per field | error | q² | q² (own ridge) | time |
|---|---:|---:|---:|---:|---:|
| Fourier, mc | 64 | 7.5e-02 | +0.5561 | +0.5663 | 0.01 s |
| Fourier, mc | 256 | 4.1e-02 | +0.6010 | +0.6026 | 0.06 s |
| Fourier, mc | 1024 | 1.9e-02 | +0.6174 | +0.6174 | 0.50 s |
| Fourier, mc | 2048 | 1.5e-02 | +0.6133 | +0.6135 | 0.80 s |
| Fourier, orthogonal | 64 | 9.0e-02 | +0.5388 | +0.5625 | 0.01 s |
| Fourier, orthogonal | 256 | 3.6e-02 | +0.5678 | +0.5678 | 0.07 s |
| Fourier, orthogonal | 1024 | 1.9e-02 | +0.5947 | +0.5947 | 0.52 s |
| Fourier, orthogonal | 2048 | 1.4e-02 | +0.6002 | +0.6002 | 0.98 s |
| Fourier, qmc | 64 | 6.5e-02 | +0.5316 | +0.5536 | 0.02 s |
| Fourier, qmc | 256 | 2.0e-02 | +0.5740 | +0.5740 | 0.08 s |
| Fourier, qmc | 1024 | 7.1e-03 | +0.6031 | +0.6031 | 0.55 s |
| Fourier, qmc | 2048 | 4.4e-03 | +0.6051 | +0.6051 | 0.96 s |
| Fourier, lattice (tolerance 0.001) | 2374/5138 | 1.0e-03 | +0.6088 | +0.6088 | 1.77 s |
| Fourier, lattice (tolerance 1e-06) | 9796/18614 | 1.7e-06 | +0.6093 | +0.6093 | 6.62 s |
| Nyström, molecules | 10 | 1.7e-01 | +0.3535 | +0.4835 | ~0.05 s |
| Nyström, molecules | 20 | 7.2e-02 | +0.4968 | +0.5301 | ~0.10 s |
| Nyström, molecules | 40 | 1.8e-02 | +0.6046 | +0.6046 | ~0.20 s |
| Nyström, Gaussian centres 2 A | 303 | 2.1e-02 | +0.6073 | +0.6106 | 0.31 s |
| Nyström, Gaussian centres 1.5 A | 523 | 5.3e-03 | +0.6119 | +0.6119 | 0.89 s |

**THR** -- 88 molecules; alpha 0.1, lambda 0.01, exact q² **+0.7000**, exact kernel 0.77 s

| features | columns per field | error | q² | q² (own ridge) | time |
|---|---:|---:|---:|---:|---:|
| Fourier, mc | 64 | 2.8e-02 | +0.6951 | +0.6951 | 0.02 s |
| Fourier, mc | 256 | 1.1e-02 | +0.7002 | +0.7002 | 0.11 s |
| Fourier, mc | 1024 | 3.7e-03 | +0.6954 | +0.6954 | 0.74 s |
| Fourier, mc | 2048 | 3.4e-03 | +0.6983 | +0.6983 | 1.28 s |
| Fourier, orthogonal | 64 | 1.2e-02 | +0.6737 | +0.6737 | 0.02 s |
| Fourier, orthogonal | 256 | 7.8e-03 | +0.7026 | +0.7026 | 0.11 s |
| Fourier, orthogonal | 1024 | 3.4e-03 | +0.7030 | +0.7030 | 0.69 s |
| Fourier, orthogonal | 2048 | 2.5e-03 | +0.7006 | +0.7006 | 1.20 s |
| Fourier, qmc | 64 | 1.2e-02 | +0.6818 | +0.6818 | 0.02 s |
| Fourier, qmc | 256 | 2.5e-03 | +0.6998 | +0.6998 | 0.11 s |
| Fourier, qmc | 1024 | 1.1e-03 | +0.6997 | +0.6997 | 0.71 s |
| Fourier, qmc | 2048 | 5.7e-04 | +0.7000 | +0.7000 | 1.18 s |
| Fourier, lattice (tolerance 0.001) | 258/366 | 2.1e-04 | +0.7000 | +0.7000 | 0.16 s |
| Fourier, lattice (tolerance 1e-06) | 1648/1744 | 3.1e-07 | +0.7000 | +0.7000 | 0.96 s |
| Nyström, molecules | 10 | 1.7e-02 | +0.5516 | +0.5759 | ~0.09 s |
| Nyström, molecules | 20 | 4.0e-03 | +0.6714 | +0.6714 | ~0.17 s |
| Nyström, molecules | 40 | 4.1e-04 | +0.7019 | +0.7019 | ~0.34 s |
| Nyström, molecules | 80 | 1.7e-05 | +0.7004 | +0.7004 | ~0.68 s |
| Nyström, Gaussian centres 2 A | 299 | 2.1e-07 | +0.7000 | +0.7000 | 0.34 s |
| Nyström, Gaussian centres 1.5 A | 511 | 6.3e-09 | +0.7000 | +0.7000 | 0.73 s |

## The fewest columns per field that bring q² within 0.01 of the exact model

The exact model's ridge is used throughout. A dash means that no size tried reached it (Fourier: up to 2048 columns; Nyström molecules: up to 160 or the whole set; centres: down to 1.25 A).

| set | Fourier, mc | Fourier, orthogonal | Fourier, qmc | Nyström, molecules | Nyström, Gaussian centres |
|---|---:|---:|---:|---:|---:|
| ACE | 512 | 512 | 512 | 10 | 628 |
| AChE | 1024 | 256 | 1024 | 20 | 428 |
| BZR | 64 | 64 | 64 | 20 | 55 |
| COX2 | 64 | 128 | 64 | 20 | 85 |
| DHFR | 64 | 128 | 128 | 40 | 110 |
| GPB | 1024 | 64 | 64 | 40 | 126 |
| THER | 256 | 32 | 1024 | 40 | 303 |
| THR | 64 | 128 | 128 | 40 | 79 |

## Narrow fields

The same sets at attenuations from broad to narrow, each with the ridge that is best for the exact kernel. Fourier features with Sobol frequencies, Nyström with landmark molecules (mean of 3 draws) and with Gaussian centres 2 A apart.

| set | alpha | exact q² | Fourier, 256 columns | Fourier, 1024 columns | Nyström, 40 molecules | Nyström, 80 molecules | Nyström, centres (columns) |
|---|---:|---:|---:|---:|---:|---:|---:|
| COX2 | 0.1 | +0.549 | +0.545 | +0.545 | +0.548 | +0.548 | +0.549 (169) |
| COX2 | 0.3 | +0.552 | +0.554 | +0.553 | +0.549 | +0.551 | +0.552 (169) |
| COX2 | 1 | +0.525 | +0.519 | +0.524 | +0.519 | +0.523 | +0.526 (169) |
| COX2 | 2 | +0.526 | +0.522 | +0.521 | +0.504 | +0.516 | +0.514 (169) |
| COX2 | 5 | +0.536 | +0.537 | +0.538 | +0.496 | +0.510 | +0.498 (169) |
| THR | 0.1 | +0.700 | +0.700 | +0.700 | +0.702 | +0.700 | +0.700 (299) |
| THR | 0.3 | +0.678 | +0.669 | +0.674 | +0.687 | +0.681 | +0.678 (299) |
| THR | 1 | +0.682 | +0.685 | +0.684 | +0.698 | +0.688 | +0.685 (299) |
| THR | 2 | +0.671 | +0.673 | +0.672 | +0.655 | +0.674 | +0.677 (299) |
| THR | 5 | +0.643 | +0.655 | +0.641 | +0.515 | +0.632 | +0.629 (299) |

