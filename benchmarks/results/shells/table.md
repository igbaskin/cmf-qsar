# Item BT5: sampled solvation shells against one optimised complex

62 LiPF6 electrolytes over 22 solvent compositions, with measured conductivities from CALiSol-23 and shells sampled from one bulk simulation each.

| What | Value |
|---|---|
| Molecules in a shell | 4.26 +/- 0.35 |
| Donor atoms at the ion | 6.95 +/- 0.40 |
| Share of shells holding an anion | 0.723 +/- 0.199 |
| Density the boxes settled at, g/cm3 | 1.205 +/- 0.081 |
| Block error of the mean field | 0.00019 +/- 0.00037 |
| Cost of keeping 16 members | 0.00652 +/- 0.00702 |
| Distinct shells one optimised complex gives | 12 for 22 compositions |
| Anion in the shell against molality, correlation | +0.729 |
| Donors at the ion against molality, correlation | +0.769 |
| Cost of 16 members against molality, correlation | +0.280 |

## log10 conductivity at 298 K (log10 mS/cm), random folds

| Model | Mean absolute error | R2 |
|---|---|---|
| conditions | 0.136 +/- 0.048 | +0.069 |
| composition | 0.062 +/- 0.019 | +0.776 |
| solvent | 0.090 +/- 0.036 | +0.460 |
| complex | 0.100 +/- 0.037 | +0.422 |
| complex-free | 0.094 +/- 0.040 | +0.462 |
| shells | 0.115 +/- 0.042 | +0.375 |
| shells-free | 0.109 +/- 0.040 | +0.419 |

| Comparison | Gain of the first model | 95 per cent interval |
|---|---|---|
| shells vs complex | -0.0154 | [-0.0341, +0.0028] |
| shells-free vs complex-free | -0.0155 | [-0.0358, +0.0035] |
| shells vs solvent | -0.0251 | [-0.0422, -0.0065] |
| shells-free vs solvent | -0.0195 | [-0.0394, +0.0025] |
| shells-free vs composition | -0.0468 | [-0.0656, -0.0285] |
| shells vs composition | -0.0524 | [-0.0717, -0.0339] |
| shells vs conditions | +0.0212 | [+0.0040, +0.0391] |
| complex vs conditions | +0.0366 | [+0.0153, +0.0581] |
| the frame on the shells | -0.0056 | [-0.0233, +0.0130] |
| the frame on the complex | -0.0056 | [-0.0138, +0.0028] |

## log10 conductivity at 298 K (log10 mS/cm), by-composition folds

| Model | Mean absolute error | R2 |
|---|---|---|
| conditions | 0.137 +/- 0.043 | -0.039 |
| composition | 0.075 +/- 0.017 | +0.727 |
| solvent | 0.082 +/- 0.038 | +0.460 |
| complex | 0.103 +/- 0.031 | +0.352 |
| complex-free | 0.103 +/- 0.043 | +0.337 |
| shells | 0.113 +/- 0.031 | +0.348 |
| shells-free | 0.103 +/- 0.041 | +0.428 |

| Comparison | Gain of the first model | 95 per cent interval |
|---|---|---|
| shells vs complex | -0.0101 | [-0.0299, +0.0099] |
| shells-free vs complex-free | +0.0001 | [-0.0181, +0.0192] |
| shells vs solvent | -0.0308 | [-0.0493, -0.0125] |
| shells-free vs solvent | -0.0213 | [-0.0388, -0.0020] |
| shells-free vs composition | -0.0291 | [-0.0474, -0.0104] |
| shells vs composition | -0.0386 | [-0.0591, -0.0177] |
| shells vs conditions | +0.0246 | [+0.0045, +0.0444] |
| complex vs conditions | +0.0348 | [+0.0132, +0.0568] |
| the frame on the shells | -0.0095 | [-0.0250, +0.0055] |
| the frame on the complex | +0.0007 | [-0.0096, +0.0107] |

## activation energy at 298 K (kJ/mol), random folds

| Model | Mean absolute error | R2 |
|---|---|---|
| conditions | 2.983 +/- 0.893 | +0.248 |
| composition | 1.374 +/- 0.486 | +0.810 |
| solvent | 1.726 +/- 0.341 | +0.674 |
| complex | 2.001 +/- 0.355 | +0.677 |
| complex-free | 1.934 +/- 0.533 | +0.662 |
| shells | 2.236 +/- 0.843 | +0.673 |
| shells-free | 1.612 +/- 0.312 | +0.804 |

| Comparison | Gain of the first model | 95 per cent interval |
|---|---|---|
| shells vs complex | -0.2300 | [-0.6559, +0.1869] |
| shells-free vs complex-free | +0.3227 | [-0.0824, +0.7861] |
| shells vs solvent | -0.5028 | [-0.9170, -0.0441] |
| shells-free vs solvent | +0.1181 | [-0.2839, +0.5831] |
| shells-free vs composition | -0.2381 | [-0.5727, +0.1078] |
| shells vs composition | -0.8590 | [-1.2803, -0.4426] |
| shells vs conditions | +0.7427 | [+0.2071, +1.3789] |
| complex vs conditions | +0.9728 | [+0.5166, +1.4384] |
| the frame on the shells | -0.6209 | [-1.0031, -0.2232] |
| the frame on the complex | -0.0681 | [-0.2178, +0.0874] |

## activation energy at 298 K (kJ/mol), by-composition folds

| Model | Mean absolute error | R2 |
|---|---|---|
| conditions | 3.035 +/- 1.156 | +0.218 |
| composition | 1.429 +/- 0.704 | +0.841 |
| solvent | 2.023 +/- 0.935 | +0.584 |
| complex | 2.424 +/- 0.574 | +0.588 |
| complex-free | 2.232 +/- 0.419 | +0.625 |
| shells | 2.903 +/- 0.991 | +0.399 |
| shells-free | 1.685 +/- 0.321 | +0.825 |

| Comparison | Gain of the first model | 95 per cent interval |
|---|---|---|
| shells vs complex | -0.4934 | [-0.9025, -0.0987] |
| shells-free vs complex-free | +0.5464 | [+0.1371, +1.0377] |
| shells vs solvent | -0.8841 | [-1.2327, -0.5570] |
| shells-free vs solvent | +0.3462 | [-0.1002, +0.8700] |
| shells-free vs composition | -0.2517 | [-0.5883, +0.1150] |
| shells vs composition | -1.4820 | [-2.0401, -0.9504] |
| shells vs conditions | +0.1268 | [-0.3301, +0.6004] |
| complex vs conditions | +0.6202 | [+0.0914, +1.1675] |
| the frame on the shells | -1.2303 | [-1.7916, -0.6968] |
| the frame on the complex | -0.1905 | [-0.3720, -0.0050] |

## The same comparison with the mixture chosen inside each fold

Every model may now keep as much or as little of its field as the inner split prefers. A model that drops its field altogether lands on the `conditions` model, so this reading says how much of each result above survives being allowed to do that.

| Target and split | Comparison | Gain of the first model | 95 per cent interval |
|---|---|---|---|
| log10 conductivity at 298 K, random | shells vs complex | -0.0102 | [-0.0299, +0.0091] |
| log10 conductivity at 298 K, random | shells-free vs complex-free | +0.0021 | [-0.0221, +0.0277] |
| log10 conductivity at 298 K, random | shells vs solvent | -0.0107 | [-0.0285, +0.0084] |
| log10 conductivity at 298 K, random | shells-free vs solvent | -0.0060 | [-0.0285, +0.0207] |
| log10 conductivity at 298 K, random | shells-free vs composition | -0.0430 | [-0.0626, -0.0234] |
| log10 conductivity at 298 K, random | shells vs composition | -0.0477 | [-0.0682, -0.0275] |
| log10 conductivity at 298 K, random | shells vs conditions | +0.0259 | [+0.0077, +0.0450] |
| log10 conductivity at 298 K, random | complex vs conditions | +0.0361 | [+0.0130, +0.0592] |
| log10 conductivity at 298 K, random | the frame on the shells | -0.0048 | [-0.0270, +0.0162] |
| log10 conductivity at 298 K, random | the frame on the complex | +0.0075 | [-0.0048, +0.0200] |
| log10 conductivity at 298 K, by-composition | shells vs complex | -0.0206 | [-0.0423, +0.0007] |
| log10 conductivity at 298 K, by-composition | shells-free vs complex-free | +0.0087 | [-0.0111, +0.0313] |
| log10 conductivity at 298 K, by-composition | shells vs solvent | -0.0398 | [-0.0614, -0.0198] |
| log10 conductivity at 298 K, by-composition | shells-free vs solvent | -0.0119 | [-0.0345, +0.0137] |
| log10 conductivity at 298 K, by-composition | shells-free vs composition | -0.0210 | [-0.0406, -0.0015] |
| log10 conductivity at 298 K, by-composition | shells vs composition | -0.0489 | [-0.0748, -0.0226] |
| log10 conductivity at 298 K, by-composition | shells vs conditions | +0.0143 | [-0.0026, +0.0329] |
| log10 conductivity at 298 K, by-composition | complex vs conditions | +0.0350 | [+0.0133, +0.0568] |
| log10 conductivity at 298 K, by-composition | the frame on the shells | -0.0279 | [-0.0532, -0.0039] |
| log10 conductivity at 298 K, by-composition | the frame on the complex | +0.0014 | [-0.0086, +0.0112] |
| activation energy at 298 K, random | shells vs complex | -0.3651 | [-0.7941, +0.0623] |
| activation energy at 298 K, random | shells-free vs complex-free | +0.4355 | [+0.0566, +0.8894] |
| activation energy at 298 K, random | shells vs solvent | -0.5960 | [-1.0452, -0.1247] |
| activation energy at 298 K, random | shells-free vs solvent | +0.2597 | [-0.1032, +0.7017] |
| activation energy at 298 K, random | shells-free vs composition | -0.1217 | [-0.4150, +0.1772] |
| activation energy at 298 K, random | shells vs composition | -0.9774 | [-1.4764, -0.4802] |
| activation energy at 298 K, random | shells vs conditions | +0.6243 | [+0.2586, +1.0591] |
| activation energy at 298 K, random | complex vs conditions | +0.9894 | [+0.5536, +1.4164] |
| activation energy at 298 K, random | the frame on the shells | -0.8557 | [-1.3884, -0.3346] |
| activation energy at 298 K, random | the frame on the complex | -0.0552 | [-0.2111, +0.1056] |
| activation energy at 298 K, by-composition | shells vs complex | -0.3100 | [-0.7462, +0.0933] |
| activation energy at 298 K, by-composition | shells-free vs complex-free | +0.5861 | [+0.1326, +1.1628] |
| activation energy at 298 K, by-composition | shells vs solvent | -0.7505 | [-1.0584, -0.4333] |
| activation energy at 298 K, by-composition | shells-free vs solvent | +0.4196 | [-0.0769, +1.0224] |
| activation energy at 298 K, by-composition | shells-free vs composition | -0.1548 | [-0.5389, +0.2661] |
| activation energy at 298 K, by-composition | shells vs composition | -1.3249 | [-2.0268, -0.7032] |
| activation energy at 298 K, by-composition | shells vs conditions | +0.2839 | [-0.1086, +0.6786] |
| activation energy at 298 K, by-composition | complex vs conditions | +0.5939 | [+0.0885, +1.1155] |
| activation energy at 298 K, by-composition | the frame on the shells | -1.1701 | [-1.9119, -0.5368] |
| activation energy at 298 K, by-composition | the frame on the complex | -0.2740 | [-0.4630, -0.0790] |

## The co-field of the shell model, read from the ion

The donor atoms of these shells sit 2.14 +/- 0.17 A from the ion, which is what the last column is to be read against.

| Field | Largest magnitude | Distance from the ion, A | Distance from the nearest donor atom, A |
|---|---|---|---|
| Electrostatic | -3.5054 | 3.03 | 0.81 |
| Steric | -0.4050 | 5.04 | 2.85 |
| Hydrophobic | -1.5414 | 5.00 | 2.86 |
| H-bond acceptor | -1.2302 | 4.22 | 1.92 |
| H-bond donor | +0.0000 | 16.12 | 13.72 |
