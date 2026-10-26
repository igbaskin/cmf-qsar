# Six pharmacophore features against the Sybyl indicators

Indicator fields put a Gaussian of weight one on every atom of one kind. Keyed by Sybyl atom type they help above about forty molecules and not below, because a congeneric series has twenty-odd types and each is rare. These six -- donor, acceptor, aromatic, hydrophobe, cation, anion, from RDKit BaseFeatures.fdef -- are each carried by most molecules of a series. The 38 Mittal sets, whose alignments are automatic, so the absolute numbers are lower than on the hand-aligned sets and only the columns compare. Same attenuation and ridge grids and the same ten-fold cross-validation as `run_alignment.py`.

| set | n | five | five + pharm | pharm | five + cif | cif | best |
|---|---:|---:|---:|---:|---:|---:|---|
| APZ | 16 | +0.575 | +0.601 | **+0.683** | +0.523 | +0.506 | pharm |
| RYR | 17 | +0.355 | +0.430 | **+0.469** | +0.309 | +0.049 | pharm |
| STEROIDS | 21 | +0.515 | +0.557 | **+0.654** | +0.562 | +0.604 | pharm |
| VATP | 23 | +0.652 | **+0.660** | +0.628 | +0.588 | +0.541 | five + pharm |
| TP2A | 25 | +0.068 | +0.081 | +0.052 | +0.136 | **+0.169** | cif |
| D2A | 26 | +0.549 | +0.575 | **+0.580** | +0.505 | +0.451 | pharm |
| ARB | 28 | +0.174 | +0.201 | +0.190 | +0.317 | **+0.320** | cif |
| MX | 29 | **+0.616** | +0.615 | +0.161 | +0.433 | +0.521 | five |
| PDE | 29 | -0.122 | -0.137 | **-0.113** | -0.176 | -0.155 | pharm |
| ACHE_TOX | 30 | -0.255 | -0.292 | -0.189 | -0.169 | **-0.065** | cif |
| CBRA | 31 | +0.246 | +0.317 | +0.358 | +0.328 | **+0.368** | cif |
| GHS | 31 | +0.341 | +0.411 | **+0.531** | +0.338 | +0.328 | pharm |
| DR | 38 | +0.420 | +0.452 | +0.441 | +0.577 | **+0.593** | cif |
| KOA | 39 | **+0.428** | +0.415 | +0.275 | +0.197 | +0.152 | five |
| YOPH | 39 | +0.563 | +0.640 | +0.646 | **+0.695** | +0.636 | five + cif |
| PTC | 40 | +0.449 | +0.416 | +0.328 | +0.528 | **+0.539** | cif |
| EV71 | 41 | -0.030 | -0.014 | -0.018 | +0.129 | **+0.156** | cif |
| DAT | 42 | **+0.305** | +0.244 | +0.007 | +0.218 | +0.075 | five |
| DIAZEPAM | 42 | +0.709 | +0.712 | +0.190 | **+0.725** | +0.646 | five + cif |
| GSK3B | 42 | **+0.580** | +0.556 | +0.065 | +0.554 | +0.532 | five |
| TCHK | 42 | +0.075 | +0.092 | **+0.122** | +0.031 | +0.048 | pharm |
| ECR | 50 | +0.065 | +0.070 | +0.080 | +0.163 | **+0.172** | cif |
| GPB | 66 | +0.260 | +0.280 | +0.263 | +0.378 | **+0.379** | cif |
| THERM | 76 | +0.043 | +0.080 | +0.129 | +0.170 | **+0.199** | cif |
| AI | 78 | +0.669 | +0.653 | +0.572 | **+0.674** | +0.612 | five + cif |
| THR | 88 | +0.021 | **+0.076** | +0.010 | +0.024 | +0.008 | five + pharm |
| COMT | 92 | +0.468 | +0.459 | +0.453 | **+0.628** | +0.561 | five + cif |
| ATA | 94 | +0.206 | **+0.212** | +0.192 | +0.158 | +0.133 | five + pharm |
| HIVRT | 99 | +0.474 | +0.413 | +0.321 | **+0.599** | +0.544 | five + cif |
| ACHE | 111 | +0.243 | **+0.266** | +0.243 | +0.163 | +0.130 | five + pharm |
| HIVPR | 113 | +0.013 | +0.003 | **+0.032** | -0.020 | -0.008 | pharm |
| ACE | 114 | +0.470 | +0.512 | +0.514 | **+0.570** | +0.549 | five + cif |
| EDC | 122 | +0.288 | +0.337 | +0.320 | +0.321 | **+0.347** | cif |
| AMPH1 | 130 | +0.059 | +0.088 | +0.092 | **+0.111** | +0.109 | five + cif |
| HPTP | 135 | +0.400 | +0.408 | +0.367 | +0.476 | **+0.486** | cif |
| BZR | 163 | +0.361 | +0.416 | +0.341 | **+0.510** | +0.430 | five + cif |
| COX2 | 322 | +0.248 | +0.239 | +0.226 | **+0.341** | +0.316 | five + cif |
| DHFR | 397 | +0.490 | **+0.500** | +0.388 | +0.489 | +0.429 | five + pharm |
| **mean** | | +0.316 | +0.330 | +0.279 | +0.345 | +0.327 | |

Split at 40 molecules, which is where the Sybyl indicators were found to start helping:

| sets | n | five | five + pharm | pharm | five + cif | cif |
|---|---:|---:|---:|---:|---:|---:|
| below 40 | 15 | +0.342 | +0.368 | +0.358 | +0.344 | +0.335 |
| at or above 40 | 23 | +0.299 | +0.305 | +0.228 | +0.345 | +0.321 |

**Added to the five fields, the six features are worth +0.014 q² on average (26 of 38 sets better) and the Sybyl indicators +0.029 (23 of 38).** Alone, the six reach +0.279 against +0.327 for the Sybyl ones and +0.316 for the five physico-chemical fields.

## Why: what each vocabulary covers

The share of the molecules of a set that carries each kind. This is the whole argument of the item, and it is visible before any model is fitted.

| set | donor | acceptor | aromatic | hydrophobe | cation | anion | Sybyl types, all | median coverage | types kept | median coverage of those |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| APZ | 100% | 100% | 100% | 100% | 94% | 0% | 10 | 97% | 10 | 97% |
| RYR | 100% | 100% | 94% | 100% | 18% | 0% | 13 | 94% | 7 | 94% |
| STEROIDS | 90% | 100% | 14% | 100% | 0% | 0% | 6 | 88% | 5 | 90% |
| VATP | 100% | 100% | 100% | 100% | 0% | 0% | 10 | 100% | 8 | 100% |
| TP2A | 100% | 100% | 100% | 96% | 100% | 0% | 17 | 12% | 8 | 92% |
| D2A | 100% | 100% | 100% | 100% | 100% | 0% | 5 | 100% | 5 | 100% |
| ARB | 100% | 100% | 100% | 100% | 64% | 79% | 11 | 79% | 8 | 98% |
| MX | 41% | 100% | 0% | 100% | 0% | 0% | 7 | 100% | 7 | 100% |
| PDE | 72% | 100% | 100% | 100% | 0% | 0% | 13 | 48% | 10 | 88% |
| ACHE_TOX | 20% | 100% | 60% | 97% | 0% | 0% | 17 | 27% | 10 | 80% |
| CBRA | 90% | 100% | 26% | 100% | 0% | 0% | 12 | 34% | 7 | 77% |
| GHS | 100% | 100% | 100% | 100% | 100% | 0% | 9 | 100% | 9 | 100% |
| DR | 100% | 71% | 100% | 100% | 100% | 0% | 16 | 30% | 12 | 39% |
| KOA | 100% | 100% | 100% | 100% | 100% | 0% | 15 | 49% | 13 | 62% |
| YOPH | 64% | 100% | 100% | 100% | 0% | 62% | 14 | 42% | 9 | 69% |
| PTC | 0% | 95% | 100% | 100% | 100% | 0% | 7 | 100% | 7 | 100% |
| EV71 | 100% | 100% | 100% | 100% | 0% | 0% | 18 | 23% | 12 | 100% |
| DAT | 100% | 98% | 100% | 100% | 100% | 7% | 13 | 69% | 9 | 100% |
| DIAZEPAM | 93% | 100% | 100% | 100% | 98% | 0% | 14 | 95% | 10 | 99% |
| GSK3B | 100% | 100% | 100% | 100% | 0% | 0% | 14 | 55% | 12 | 77% |
| TCHK | 100% | 100% | 83% | 100% | 21% | 2% | 15 | 36% | 9 | 100% |
| ECR | 100% | 100% | 100% | 100% | 0% | 0% | 16 | 14% | 8 | 100% |
| GPB | 100% | 100% | 18% | 44% | 11% | 2% | 20 | 7% | 10 | 58% |
| THERM | 100% | 100% | 88% | 100% | 13% | 54% | 14 | 70% | 11 | 88% |
| AI | 35% | 100% | 22% | 100% | 1% | 0% | 15 | 10% | 8 | 58% |
| THR | 100% | 100% | 100% | 100% | 100% | 10% | 12 | 99% | 11 | 100% |
| COMT | 100% | 100% | 100% | 100% | 2% | 7% | 18 | 31% | 14 | 70% |
| ATA | 44% | 100% | 100% | 100% | 0% | 11% | 12 | 55% | 10 | 66% |
| HIVRT | 100% | 100% | 100% | 100% | 0% | 0% | 15 | 25% | 10 | 93% |
| ACHE | 100% | 97% | 100% | 100% | 98% | 0% | 15 | 29% | 11 | 93% |
| HIVPR | 100% | 100% | 100% | 100% | 4% | 1% | 17 | 9% | 11 | 100% |
| ACE | 99% | 100% | 60% | 100% | 42% | 97% | 14 | 54% | 13 | 60% |
| EDC | 93% | 98% | 96% | 100% | 5% | 4% | 13 | 15% | 9 | 43% |
| AMPH1 | 100% | 100% | 44% | 100% | 100% | 100% | 13 | 100% | 13 | 100% |
| HPTP | 99% | 100% | 100% | 100% | 1% | 76% | 17 | 19% | 14 | 29% |
| BZR | 67% | 99% | 100% | 100% | 35% | 2% | 20 | 33% | 15 | 66% |
| COX2 | 49% | 100% | 100% | 100% | 30% | 2% | 20 | 22% | 18 | 25% |
| DHFR | 100% | 100% | 100% | 100% | 7% | 4% | 19 | 10% | 16 | 12% |

A pharmacophore feature is carried by 70% of the molecules of a set on average, and the median Sybyl type by 48% -- which is the difference the item was built on. But the Sybyl vocabulary the program actually uses is already pruned: `cif_types_in` keeps only the types in five molecules or more, which on these sets is 10 of 14 types, and the median coverage of what is kept is 94%. **The rarity the item blames was already being filtered out**, so whatever the six features gain is not that.

