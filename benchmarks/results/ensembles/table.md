# Item MS5: an ensemble of structures through the general entry

The gate of item MS5 is that the general path reproduces the numbers of item BT5 from the same frames. The general path reads one SD file with `load_dataset(..., ensemble_by="electrolyte", member_weight="member_weight")` and builds the fields with `conformers="ensemble"` and the weighting "given". BT5's path reads the shell files with the code of `cmfqsar/shells.py`.

## The file

989 records, one per shell, make 62 electrolytes. An electrolyte has 15 to 16 shells, and a shell has 15 to 94 atoms. In 62 of the 62 electrolytes the shells differ in their atoms, so they cannot be conformations of one molecule.

## The tables of the two paths

The framed kernel is the joint overlap kernel of the five fields, with a unit diagonal. The alignment-free rows are the rows of the rotation-invariant autocorrelation. The general path does not use the kernel cache.

| Field width (alpha) | Largest difference in the framed kernel | Largest difference in the alignment-free rows |
|---:|---:|---:|
| 0.3 | 0 | 0 |
| 0.5 | 0 | 0 |
| 1.0 | 0 | 0 |
| 2.0 | 0 | 0 |

## The numbers of BT5, from each path

Nested cross-validation as in `benchmarks/run_shells.py`: the ridge and the field width chosen inside each fold, the field kernel and the molality kernel mixed in equal parts. The mean absolute error is the mean over the five folds. The difference is the largest absolute difference between the new record and the stored one in `benchmarks/results/shells/gate.json`, over every number in the record: the error of each electrolyte, the settings chosen in each fold and the summary numbers.

| Target | Split | Model | Mean absolute error, stored | Mean absolute error, BT5's path now | Difference from the stored record, BT5's path | Mean absolute error, general path | Difference from the stored record, general path |
|---|---|---|---:|---:|---:|---:|---:|
| log10 of the conductivity at 298 K | random | shells | 0.1147 | 0.1147 | 0 | 0.1147 | 0 |
| log10 of the conductivity at 298 K | random | shells-free | 0.1091 | 0.1091 | 0 | 0.1091 | 0 |
| log10 of the conductivity at 298 K | by-composition | shells | 0.1130 | 0.1130 | 0 | 0.1130 | 0 |
| log10 of the conductivity at 298 K | by-composition | shells-free | 0.1034 | 0.1034 | 0 | 0.1034 | 0 |
| activation energy at 298 K, kJ/mol | random | shells | 2.2357 | 2.2357 | 0 | 2.2357 | 0 |
| activation energy at 298 K, kJ/mol | random | shells-free | 1.6121 | 1.6121 | 0 | 1.6121 | 0 |
| activation energy at 298 K, kJ/mol | by-composition | shells | 2.9028 | 2.9028 | 0 | 2.9028 | 0 |
| activation energy at 298 K, kJ/mol | by-composition | shells-free | 1.6845 | 1.6845 | 0 | 1.6845 | 0 |

## The gate

**Met.** Every number of the two models is reproduced exactly by the general path: the tables are the same to the last bit, and so is every record of the nested cross-validation.

The control, BT5's own path run again, differs from the stored file by 0, so the stored file matches the present code.


## How many members the model needs

The alignment-free model of BT5, the better of its two, with the shells read through the general path. Each variant keeps the members of largest weight (`max_conformers`) and uses the given weights, except the last, which weights 16 members equally. The references are the two that BT5 kept: the molality alone, and the molality with the solvent fractions. Mean absolute error over the five outer folds; the interval is the paired interval of rule 4 over the 62 electrolytes, the gain of the variant over the model named, positive when the variant has the smaller error.

### log10 of the conductivity at 298 K, random folds

| Model | Mean absolute error | Gain over 1 member | Gain over the composition |
|---|---|---|---|
| conditions | 0.1360 +/- 0.0481 | | |
| composition | 0.0621 +/- 0.0192 | | |
| 1 member, the heaviest | 0.1098 +/- 0.0473 |  | -0.0479 [-0.0713, -0.0262] |
| 2 members | 0.1038 +/- 0.0487 | +0.0061 [-0.0076, +0.0199] | -0.0417 [-0.0646, -0.0208] |
| 4 members | 0.1102 +/- 0.0480 | -0.0002 [-0.0097, +0.0093] | -0.0481 [-0.0717, -0.0259] |
| 8 members | 0.0946 +/- 0.0511 | +0.0152 [-0.0013, +0.0333] | -0.0326 [-0.0562, -0.0110] |
| 16 members, given weights (the model of the gate) | 0.1091 +/- 0.0404 | +0.0010 [-0.0191, +0.0217] | -0.0468 [-0.0656, -0.0285] |
| 16 members, equal weights | 0.1075 +/- 0.0322 | +0.0023 [-0.0239, +0.0272] | -0.0456 [-0.0640, -0.0288] |

### log10 of the conductivity at 298 K, by-composition folds

| Model | Mean absolute error | Gain over 1 member | Gain over the composition |
|---|---|---|---|
| conditions | 0.1373 +/- 0.0428 | | |
| composition | 0.0748 +/- 0.0174 | | |
| 1 member, the heaviest | 0.1157 +/- 0.0357 |  | -0.0415 [-0.0625, -0.0209] |
| 2 members | 0.1250 +/- 0.0394 | -0.0086 [-0.0218, +0.0046] | -0.0501 [-0.0746, -0.0277] |
| 4 members | 0.1261 +/- 0.0417 | -0.0103 [-0.0208, +0.0005] | -0.0518 [-0.0754, -0.0300] |
| 8 members | 0.1020 +/- 0.0425 | +0.0136 [-0.0048, +0.0335] | -0.0279 [-0.0465, -0.0086] |
| 16 members, given weights (the model of the gate) | 0.1034 +/- 0.0406 | +0.0124 [-0.0044, +0.0301] | -0.0291 [-0.0474, -0.0104] |
| 16 members, equal weights | 0.0957 +/- 0.0295 | +0.0204 [-0.0012, +0.0415] | -0.0211 [-0.0418, -0.0011] |

### activation energy at 298 K, kJ/mol, random folds

| Model | Mean absolute error | Gain over 1 member | Gain over the composition |
|---|---|---|---|
| conditions | 2.9832 +/- 0.8930 | | |
| composition | 1.3744 +/- 0.4856 | | |
| 1 member, the heaviest | 2.0515 +/- 0.5656 |  | -0.6897 [-1.0726, -0.3205] |
| 2 members | 2.0655 +/- 0.5697 | -0.0011 [-0.3355, +0.3350] | -0.6908 [-1.1450, -0.2808] |
| 4 members | 1.9381 +/- 0.4267 | +0.1248 [-0.2035, +0.4626] | -0.5650 [-0.9787, -0.1613] |
| 8 members | 1.6557 +/- 0.4326 | +0.4113 [+0.0403, +0.8259] | -0.2784 [-0.6209, +0.1176] |
| 16 members, given weights (the model of the gate) | 1.6121 +/- 0.3122 | +0.4516 [+0.0656, +0.8554] | -0.2381 [-0.5727, +0.1078] |
| 16 members, equal weights | 1.6072 +/- 0.2967 | +0.4533 [+0.0920, +0.8250] | -0.2365 [-0.5672, +0.1016] |

### activation energy at 298 K, kJ/mol, by-composition folds

| Model | Mean absolute error | Gain over 1 member | Gain over the composition |
|---|---|---|---|
| conditions | 3.0353 +/- 1.1558 | | |
| composition | 1.4287 +/- 0.7043 | | |
| 1 member, the heaviest | 2.1542 +/- 0.4362 |  | -0.7255 [-1.1161, -0.3447] |
| 2 members | 2.1977 +/- 0.6534 | -0.0446 [-0.4333, +0.3415] | -0.7701 [-1.2620, -0.3234] |
| 4 members | 2.0357 +/- 0.4742 | +0.1173 [-0.2670, +0.4899] | -0.6081 [-1.0309, -0.1994] |
| 8 members | 1.6670 +/- 0.4301 | +0.4872 [+0.0970, +0.8896] | -0.2383 [-0.6296, +0.1753] |
| 16 members, given weights (the model of the gate) | 1.6845 +/- 0.3207 | +0.4737 [+0.0913, +0.8515] | -0.2517 [-0.5883, +0.1150] |
| 16 members, equal weights | 1.7113 +/- 0.5326 | +0.4524 [+0.0949, +0.7948] | -0.2730 [-0.6228, +0.0952] |

Control: the 16 given members against the stored `shells-free` records of `benchmarks/results/shells/gate.json`, largest difference 0.
