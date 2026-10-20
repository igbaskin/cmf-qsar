# The alignment against the topomer rule, on the Mittal sets, whose alignments are automatic

Aligned CMF (the frame the file comes with), topomer CMF (substituents posed by rule,
no superposition) and the alignment-free count baseline, all on the same molecules:
the congeneric series each set contains, found as the Bemis-Murcko framework of its
largest chemotype plus every molecule that framework cuts cleanly off. Same fields (electrostatic, steric, hydrophobic, hba, hbd),
same alpha grid (0.1, 0.3, 0.5, 1, 2, 5), same lambda grid, same 10-fold CV, and both field
models choose (alpha, lambda) by the CV they are scored on, so both are optimistic to
the same degree.

| set | n | series | core | sites | q2 aligned | q2 topomer | q2 counts | loss |
|---|---|---|---|---|---|---|---|---|
| ACE | 51/114 | 45% | `C1CCNC1` | 5 | +0.337 | +0.519 | +0.662 | -0.181 |
| ACHE | 29/111 | 26% | `O=C(NCCC1CC[NH+](Cc2ccccc2)CC1)c1ccccc1` | 20 | +0.426 | +0.503 | +0.345 | -0.078 |
| ACHE_TOX | 10/30 | 33% | `c1ccccc1` | 6 | -0.366 | -0.206 | -0.246 | -0.159 |
| AI | 25/78 | 32% | `O=C1C=C2CCC3C4CCC(=O)C4CCC3C2CC1` | 14 | +0.007 | +0.521 | -0.078 | -0.514 |
| AMPH1 | 130/130 | 100% | `C1CCNC1` | 5 | +0.059 | +0.081 | +0.069 | -0.021 |
| APZ | 15/16 | 94% | `O=C1c2ccccc2-c2n[nH]c3cccc1c23` | 8 | +0.500 | +0.331 | +0.399 | +0.169 |
| ATA | 94/94 | 100% | `c1ccc2ncccc2c1` | 7 | +0.206 | +0.243 | +0.072 | -0.037 |
| BZR | 69/163 | 42% | `O=C1CN=C(c2ccccc2)c2ccccc2N1` | 12 | +0.349 | +0.292 | +0.166 | +0.057 |
| COMT | 91/92 | 99% | `c1ccccc1` | 6 | +0.440 | +0.460 | +0.627 | -0.020 |
| COX2 | 78/322 | 24% | `c1ccc(-c2ccnn2-c2ccccc2)cc1` | 12 | +0.110 | +0.341 | +0.119 | -0.230 |
| DAT | 42/42 | 100% | `c1ccc(C2CC[NH2+]CC2)cc1` | 11 | +0.305 | +0.139 | +0.110 | +0.167 |
| DHFR | 42/397 | 11% | `c1ccc(NCc2cnc3ncncc3c2)cc1` | 11 | +0.162 | +0.180 | -0.051 | -0.018 |
| DIAZEPAM | 22/42 | 52% | `O=C1NCc2cncn2-c2ccccc21` | 8 | +0.588 | +0.680 | +0.214 | -0.093 |
| ECR | 41/50 | 82% | `O=C(NNC(=O)c1ccccc1)c1ccccc1` | 12 | +0.211 | +0.250 | +0.262 | -0.039 |
| EDC | 84/122 | 69% | `c1ccccc1` | 6 | +0.365 | +0.582 | +0.364 | -0.217 |
| EV71 | 16/41 | 39% | `O=C1N(CCCCCOc2ccccc2)CCN1c1ccncc1` | 16 | -0.222 | -0.138 | +0.437 | -0.084 |
| GHS | 31/31 | 100% | `O=C1CCCc2ccccc2N1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1` | 17 | +0.341 | +0.382 | +0.265 | -0.041 |
| GPB | 60/66 | 91% | `C1CCOCC1` | 5 | +0.333 | +0.511 | +0.254 | -0.178 |
| GSK3B | 23/42 | 55% | `N=C1C(=C2C(=O)Nc3ccccc32)Nc2ccccc21` | 11 | +0.165 | +0.353 | -0.105 | -0.188 |
| HIVPR | 72/113 | 64% | `O=C1N(Cc2ccccc2)C(Cc2ccccc2)CCC(Cc2ccccc2)N1Cc1ccccc1` | 28 | +0.056 | +0.108 | +0.122 | -0.052 |
| HIVRT | 62/99 | 63% | `O=c1cc(Sc2ccccc2)[nH]c(=O)[nH]1` | 8 | +0.500 | +0.711 | +0.164 | -0.210 |
| HPTP | 15/135 | 11% | `c1ccc(CCOc2ccc(-c3ccc(-c4c(Cc5ccccc5)oc5ccccc45)cc3)cc2)cc1` | 25 | +0.057 | +0.485 | -0.115 | -0.428 |
| KOA | 36/39 | 92% | `c1cc2c3c(c1)OC1c4[nH]c5ccccc5c4CC4C(C2)[NH+](CC2CC2)CC[C@@]314` | 20 | +0.106 | -0.019 | +0.328 | +0.124 |
| MX | 29/29 | 100% | `O=C1C=CCO1` | 3 | +0.616 | +0.736 | +0.732 | -0.120 |
| PDE | 26/29 | 90% | `c1ccc(C(Cc2ccncc2)c2cccc(OC3CCCC3)c2)cc1` | 20 | -0.145 | -0.038 | -0.094 | -0.107 |
| PTC | 10/40 | 25% | `c1ccc(-c2cccc3c2C[N+]2(CCCC2)Cc2c(-c4ccccc4)cccc2-3)cc1` | 22 | -0.135 | -0.174 | -0.236 | +0.039 |
| TCHK | 10/42 | 24% | `c1ccc(-c2ccccc2)cc1` | 10 | -0.260 | -0.355 | -0.134 | +0.095 |
| THERM | 61/76 | 80% | `c1ccccc1` | 6 | +0.117 | +0.131 | +0.120 | -0.014 |
| THR | 22/88 | 25% | `O=C(C(Cc1ccccc1)NS(=O)(=O)c1ccc2ccccc2c1)N1CCCCC1` | 20 | -0.175 | -0.074 | +0.062 | -0.102 |
| VATP | 13/23 | 57% | `O=C1OCCCCCC=CCc2ccccc21` | 12 | +0.478 | -0.147 | -0.128 | +0.625 |
| YOPH | 28/39 | 72% | `c1ccccc1` | 6 | +0.678 | +0.555 | +0.223 | +0.123 |

Mean loss from giving up the alignment: -0.056 q2 over 31 sets.

Splitting that loss in two. The middle column keeps the same cut substituents at the
same separated sites but leaves each one exactly where the file's own alignment
put it, so the step to its left is the price of the cut itself -- the scaffold's own
field and the geometry between one substituent and another, both of which separated
sites destroy -- and the step to its right is the price of the pose rule alone, with
the same atoms and the same weights in both.

| set | q2 whole molecules | q2 substituents, as aligned | q2 substituents, posed by rule | cut costs | pose costs |
|---|---|---|---|---|---|
| ACE | +0.337 | +0.385 | +0.519 | -0.048 | -0.133 |
| ACHE | +0.426 | +0.541 | +0.503 | -0.116 | +0.038 |
| ACHE_TOX | -0.366 | -0.445 | -0.206 | +0.080 | -0.239 |
| AI | +0.007 | +0.348 | +0.521 | -0.341 | -0.173 |
| AMPH1 | +0.059 | +0.107 | +0.081 | -0.048 | +0.026 |
| APZ | +0.500 | +0.379 | +0.331 | +0.121 | +0.048 |
| ATA | +0.206 | +0.198 | +0.243 | +0.008 | -0.045 |
| BZR | +0.349 | +0.383 | +0.292 | -0.034 | +0.091 |
| COMT | +0.440 | +0.413 | +0.460 | +0.027 | -0.048 |
| COX2 | +0.110 | +0.059 | +0.341 | +0.051 | -0.281 |
| DAT | +0.305 | +0.055 | +0.139 | +0.251 | -0.084 |
| DHFR | +0.162 | +0.212 | +0.180 | -0.049 | +0.032 |
| DIAZEPAM | +0.588 | +0.773 | +0.680 | -0.185 | +0.093 |
| ECR | +0.211 | +0.164 | +0.250 | +0.048 | -0.087 |
| EDC | +0.365 | +0.542 | +0.582 | -0.177 | -0.040 |
| EV71 | -0.222 | -0.178 | -0.138 | -0.044 | -0.040 |
| GHS | +0.341 | +0.355 | +0.382 | -0.014 | -0.027 |
| GPB | +0.333 | +0.392 | +0.511 | -0.059 | -0.119 |
| GSK3B | +0.165 | +0.227 | +0.353 | -0.062 | -0.126 |
| HIVPR | +0.056 | +0.078 | +0.108 | -0.022 | -0.030 |
| HIVRT | +0.500 | +0.490 | +0.711 | +0.011 | -0.221 |
| HPTP | +0.057 | +0.615 | +0.485 | -0.558 | +0.130 |
| KOA | +0.106 | +0.262 | -0.019 | -0.157 | +0.281 |
| MX | +0.616 | +0.679 | +0.736 | -0.063 | -0.057 |
| PDE | -0.145 | -0.073 | -0.038 | -0.072 | -0.035 |
| PTC | -0.135 | +0.070 | -0.174 | -0.205 | +0.244 |
| TCHK | -0.260 | -0.266 | -0.355 | +0.006 | +0.089 |
| THERM | +0.117 | +0.160 | +0.131 | -0.042 | +0.028 |
| THR | -0.175 | -0.201 | -0.074 | +0.026 | -0.128 |
| VATP | +0.478 | +0.417 | -0.147 | +0.061 | +0.564 |
| YOPH | +0.678 | +0.520 | +0.555 | +0.158 | -0.035 |

Mean price of the cut -0.047, mean price of the pose -0.009.

| set | RMSEcv aligned | RMSEcv topomer | aligned on all n |
|---|---|---|---|
| ACE | 1.687 | 1.438 | +0.470 (114) |
| ACHE | 0.870 | 0.809 | +0.243 (111) |
| ACHE_TOX | 1.036 | 0.974 | -0.255 (30) |
| AI | 0.485 | 0.337 | +0.669 (78) |
| AMPH1 | 0.848 | 0.839 | +0.059 (130) |
| APZ | 0.476 | 0.551 | +0.575 (16) |
| ATA | 1.108 | 1.083 | +0.206 (94) |
| BZR | 0.843 | 0.879 | +0.361 (163) |
| COMT | 0.615 | 0.604 | +0.468 (92) |
| COX2 | 1.461 | 1.258 | +0.248 (322) |
| DAT | 0.630 | 0.702 | +0.305 (42) |
| DHFR | 0.651 | 0.644 | +0.490 (397) |
| DIAZEPAM | 0.598 | 0.527 | +0.709 (42) |
| ECR | 0.634 | 0.618 | +0.065 (50) |
| EDC | 1.316 | 1.067 | +0.288 (122) |
| EV71 | 1.034 | 0.997 | -0.030 (41) |
| GHS | 0.768 | 0.743 | +0.341 (31) |
| GPB | 0.757 | 0.648 | +0.260 (66) |
| GSK3B | 0.544 | 0.479 | +0.580 (42) |
| HIVPR | 1.403 | 1.364 | +0.013 (113) |
| HIVRT | 0.889 | 0.676 | +0.474 (99) |
| HPTP | 0.333 | 0.246 | +0.400 (135) |
| KOA | 0.468 | 0.500 | +0.428 (39) |
| MX | 0.946 | 0.784 | +0.616 (29) |
| PDE | 0.658 | 0.627 | -0.122 (29) |
| PTC | 9.158 | 9.313 | +0.449 (40) |
| TCHK | 1.045 | 1.084 | +0.075 (42) |
| THERM | 1.793 | 1.779 | +0.043 (76) |
| THR | 0.793 | 0.758 | +0.021 (88) |
| VATP | 0.899 | 1.333 | +0.652 (23) |
| YOPH | 0.720 | 0.846 | +0.563 (39) |
