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
| ACE | 51/114 | 45% | `C1CCNC1` | 5 | +0.337 | +0.564 | +0.662 | -0.227 |
| ACHE | 29/111 | 26% | `O=C(NCCC1CC[NH+](Cc2ccccc2)CC1)c1ccccc1` | 20 | +0.426 | +0.510 | +0.345 | -0.084 |
| ACHE_TOX | 10/30 | 33% | `c1ccccc1` | 6 | -0.366 | -0.177 | -0.246 | -0.188 |
| AI | 25/78 | 32% | `O=C1C=C2CCC3C4CCC(=O)C4CCC3C2CC1` | 14 | +0.007 | +0.383 | -0.078 | -0.376 |
| AMPH1 | 130/130 | 100% | `C1CCNC1` | 5 | +0.059 | +0.077 | +0.069 | -0.018 |
| APZ | 15/16 | 94% | `O=C1c2ccccc2-c2n[nH]c3cccc1c23` | 8 | +0.500 | +0.296 | +0.399 | +0.205 |
| ATA | 94/94 | 100% | `c1ccc2ncccc2c1` | 7 | +0.206 | +0.235 | +0.072 | -0.030 |
| BZR | 69/163 | 42% | `O=C1CN=C(c2ccccc2)c2ccccc2N1` | 12 | +0.349 | +0.284 | +0.166 | +0.066 |
| COMT | 91/92 | 99% | `c1ccccc1` | 6 | +0.440 | +0.487 | +0.627 | -0.047 |
| COX2 | 78/322 | 24% | `c1ccc(-c2ccnn2-c2ccccc2)cc1` | 12 | +0.110 | +0.349 | +0.119 | -0.238 |
| DAT | 42/42 | 100% | `c1ccc(C2CC[NH2+]CC2)cc1` | 11 | +0.305 | +0.220 | +0.110 | +0.085 |
| DHFR | 42/397 | 11% | `c1ccc(NCc2cnc3ncncc3c2)cc1` | 11 | +0.162 | +0.109 | -0.051 | +0.054 |
| DIAZEPAM | 22/42 | 52% | `O=C1NCc2cncn2-c2ccccc21` | 8 | +0.588 | +0.659 | +0.214 | -0.072 |
| ECR | 41/50 | 82% | `O=C(NNC(=O)c1ccccc1)c1ccccc1` | 12 | +0.211 | +0.231 | +0.262 | -0.020 |
| EDC | 84/122 | 69% | `c1ccccc1` | 6 | +0.365 | +0.601 | +0.364 | -0.236 |
| EV71 | 16/41 | 39% | `O=C1N(CCCCCOc2ccccc2)CCN1c1ccncc1` | 16 | -0.222 | -0.125 | +0.437 | -0.098 |
| GHS | 31/31 | 100% | `O=C1CCCc2ccccc2N1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1` | 17 | +0.341 | +0.371 | +0.265 | -0.030 |
| GPB | 60/66 | 91% | `C1CCOCC1` | 5 | +0.333 | +0.525 | +0.254 | -0.192 |
| GSK3B | 23/42 | 55% | `N=C1C(=C2C(=O)Nc3ccccc32)Nc2ccccc21` | 11 | +0.165 | +0.414 | -0.105 | -0.249 |
| HIVPR | 72/113 | 64% | `O=C1N(Cc2ccccc2)C(Cc2ccccc2)CCC(Cc2ccccc2)N1Cc1ccccc1` | 28 | +0.056 | +0.114 | +0.122 | -0.058 |
| HIVRT | 62/99 | 63% | `O=c1cc(Sc2ccccc2)[nH]c(=O)[nH]1` | 8 | +0.500 | +0.709 | +0.164 | -0.209 |
| HPTP | 15/135 | 11% | `c1ccc(CCOc2ccc(-c3ccc(-c4c(Cc5ccccc5)oc5ccccc45)cc3)cc2)cc1` | 25 | +0.057 | +0.481 | -0.115 | -0.424 |
| KOA | 36/39 | 92% | `c1cc2c3c(c1)OC1c4[nH]c5ccccc5c4CC4C(C2)[NH+](CC2CC2)CC[C@@]314` | 20 | +0.106 | +0.120 | +0.328 | -0.014 |
| MX | 29/29 | 100% | `O=C1C=CCO1` | 3 | +0.616 | +0.739 | +0.732 | -0.123 |
| PDE | 26/29 | 90% | `c1ccc(C(Cc2ccncc2)c2cccc(OC3CCCC3)c2)cc1` | 20 | -0.145 | -0.072 | -0.094 | -0.073 |
| PTC | 10/40 | 25% | `c1ccc(-c2cccc3c2C[N+]2(CCCC2)Cc2c(-c4ccccc4)cccc2-3)cc1` | 22 | -0.135 | -0.187 | -0.236 | +0.052 |
| TCHK | 10/42 | 24% | `c1ccc(-c2ccccc2)cc1` | 10 | -0.260 | -0.380 | -0.134 | +0.120 |
| THERM | 61/76 | 80% | `c1ccccc1` | 6 | +0.117 | +0.315 | +0.120 | -0.198 |
| THR | 22/88 | 25% | `O=C(C(Cc1ccccc1)NS(=O)(=O)c1ccc2ccccc2c1)N1CCCCC1` | 20 | -0.175 | -0.133 | +0.062 | -0.042 |
| VATP | 13/23 | 57% | `O=C1OCCCCCC=CCc2ccccc21` | 12 | +0.478 | -0.205 | -0.128 | +0.683 |
| YOPH | 28/39 | 72% | `c1ccccc1` | 6 | +0.678 | +0.564 | +0.223 | +0.114 |

Mean loss from giving up the alignment: -0.060 q2 over 31 sets.

Splitting that loss in two. The middle column keeps the same cut substituents at the
same separated sites but leaves each one exactly where the file's own alignment
put it, so the step to its left is the cost of the cut itself -- the scaffold's own
field and the geometry between one substituent and another, both of which separated
sites destroy -- and the step to its right is the cost of the pose rule alone, with
the same atoms and the same weights in both.

| set | q2 whole molecules | q2 substituents, as aligned | q2 substituents, posed by rule | cut costs | pose costs |
|---|---|---|---|---|---|
| ACE | +0.337 | +0.483 | +0.564 | -0.145 | -0.081 |
| ACHE | +0.426 | +0.448 | +0.510 | -0.023 | -0.062 |
| ACHE_TOX | -0.366 | -0.480 | -0.177 | +0.115 | -0.303 |
| AI | +0.007 | +0.059 | +0.383 | -0.052 | -0.324 |
| AMPH1 | +0.059 | +0.060 | +0.077 | -0.001 | -0.017 |
| APZ | +0.500 | +0.344 | +0.296 | +0.156 | +0.048 |
| ATA | +0.206 | +0.206 | +0.235 | +0.000 | -0.030 |
| BZR | +0.349 | +0.354 | +0.284 | -0.004 | +0.070 |
| COMT | +0.440 | +0.418 | +0.487 | +0.022 | -0.069 |
| COX2 | +0.110 | +0.165 | +0.349 | -0.055 | -0.183 |
| DAT | +0.305 | +0.081 | +0.220 | +0.225 | -0.140 |
| DHFR | +0.162 | +0.238 | +0.109 | -0.075 | +0.129 |
| DIAZEPAM | +0.588 | +0.663 | +0.659 | -0.075 | +0.004 |
| ECR | +0.211 | +0.162 | +0.231 | +0.050 | -0.069 |
| EDC | +0.365 | +0.501 | +0.601 | -0.137 | -0.100 |
| EV71 | -0.222 | -0.010 | -0.125 | -0.212 | +0.115 |
| GHS | +0.341 | +0.349 | +0.371 | -0.008 | -0.022 |
| GPB | +0.333 | +0.404 | +0.525 | -0.071 | -0.121 |
| GSK3B | +0.165 | +0.260 | +0.414 | -0.095 | -0.154 |
| HIVPR | +0.056 | +0.052 | +0.114 | +0.004 | -0.063 |
| HIVRT | +0.500 | +0.445 | +0.709 | +0.055 | -0.264 |
| HPTP | +0.057 | +0.352 | +0.481 | -0.295 | -0.129 |
| KOA | +0.106 | +0.327 | +0.120 | -0.222 | +0.207 |
| MX | +0.616 | +0.594 | +0.739 | +0.022 | -0.145 |
| PDE | -0.145 | -0.058 | -0.072 | -0.086 | +0.014 |
| PTC | -0.135 | +0.021 | -0.187 | -0.156 | +0.208 |
| TCHK | -0.260 | -0.269 | -0.380 | +0.008 | +0.112 |
| THERM | +0.117 | +0.148 | +0.315 | -0.031 | -0.167 |
| THR | -0.175 | -0.191 | -0.133 | +0.016 | -0.058 |
| VATP | +0.478 | +0.459 | -0.205 | +0.020 | +0.664 |
| YOPH | +0.678 | +0.461 | +0.564 | +0.218 | -0.104 |

Mean cost of the cut -0.027, mean cost of the pose -0.033.

| set | RMSEcv aligned | RMSEcv topomer | aligned on all n |
|---|---|---|---|
| ACE | 1.687 | 1.368 | +0.470 (114) |
| ACHE | 0.870 | 0.803 | +0.243 (111) |
| ACHE_TOX | 1.036 | 0.962 | -0.255 (30) |
| AI | 0.485 | 0.382 | +0.669 (78) |
| AMPH1 | 0.848 | 0.840 | +0.059 (130) |
| APZ | 0.476 | 0.566 | +0.575 (16) |
| ATA | 1.108 | 1.088 | +0.206 (94) |
| BZR | 0.843 | 0.885 | +0.361 (163) |
| COMT | 0.615 | 0.589 | +0.468 (92) |
| COX2 | 1.461 | 1.250 | +0.248 (322) |
| DAT | 0.630 | 0.668 | +0.305 (42) |
| DHFR | 0.651 | 0.671 | +0.490 (397) |
| DIAZEPAM | 0.598 | 0.544 | +0.709 (42) |
| ECR | 0.634 | 0.626 | +0.065 (50) |
| EDC | 1.316 | 1.043 | +0.288 (122) |
| EV71 | 1.034 | 0.992 | -0.030 (41) |
| GHS | 0.768 | 0.750 | +0.341 (31) |
| GPB | 0.757 | 0.639 | +0.260 (66) |
| GSK3B | 0.544 | 0.456 | +0.580 (42) |
| HIVPR | 1.403 | 1.359 | +0.013 (113) |
| HIVRT | 0.889 | 0.678 | +0.474 (99) |
| HPTP | 0.333 | 0.247 | +0.400 (135) |
| KOA | 0.468 | 0.464 | +0.428 (39) |
| MX | 0.946 | 0.779 | +0.616 (29) |
| PDE | 0.658 | 0.637 | -0.122 (29) |
| PTC | 9.158 | 9.366 | +0.449 (40) |
| TCHK | 1.045 | 1.094 | +0.075 (42) |
| THERM | 1.793 | 1.580 | +0.043 (76) |
| THR | 0.793 | 0.779 | +0.021 (88) |
| VATP | 0.899 | 1.366 | +0.652 (23) |
| YOPH | 0.720 | 0.837 | +0.563 (39) |
