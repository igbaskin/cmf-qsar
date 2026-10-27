# Posed by rule against relaxed on a slab: the planar aligner on the Fe(110) sets

The relaxed poses are UMA's (`examples/corrosion_fe110`), the normal along z and the top
Fe layer at z = 0. The rule poses are `cmfqsar/planealign.py`: flat with the flattest
face down, or auto (flat on either face, or upright on the most basic nitrogen,
whichever is most similar in field to the one reference pose), refined on that
similarity in the plane only or also in tilt and height. Pose errors are RMSDs over
heavy atoms after the best rotation about the normal and translation in the plane.
Models: KRR on the five fields, 5 outer folds, 10 inner, 10 repeats, one seed, alpha
and lambda by the inner loop; differences with their 95% paired bootstrap interval over
molecules.

## benzimidazole: 72 molecules, 54 chemisorbed, reference Benzimidazole

### Pose recovery (A)

| frame | in-plane RMSD, all | chemisorbed | within 1 A | height RMSD, chemisorbed | rigid floor |
|---|---:|---:|---:|---:|---:|
| relaxed/flat | 2.56 | 2.39 | 0% | 1.33 | 0.00 |
| relaxed/flat-up | 1.99 | 1.30 | 46% | 1.20 | 0.00 |
| relaxed/auto+plane | 2.58 | 1.46 | 42% | 1.14 | 0.00 |
| relaxed/auto+tilt | 2.87 | 2.13 | 38% | 1.52 | 0.00 |
| conformer/flat | 2.61 | 2.22 | 17% | 1.25 | 1.20 |
| conformer/auto+plane | 2.57 | 2.13 | 21% | 1.18 | 1.21 |
| conformer/auto+tilt | 3.00 | 2.32 | 17% | 1.62 | 1.21 |
| relaxed/flat, better face | 1.73 | 1.28 | 46% | 1.16 | 0.00 |

Chemisorbed molecules by the pose UMA kept, in-plane RMSD median (n):

| frame | flat | random | upright |
|---|---:|---:|---:|
| relaxed/flat | 2.18 (35) | 3.67 (9) | 3.30 (10) |
| relaxed/flat-up | 0.36 (35) | 2.76 (9) | 3.07 (10) |
| relaxed/auto+plane | 0.45 (34) | 3.73 (9) | 3.14 (10) |
| relaxed/auto+tilt | 0.55 (34) | 4.07 (9) | 3.09 (10) |
| conformer/flat | 1.88 (35) | 3.63 (9) | 3.27 (10) |
| conformer/auto+plane | 1.56 (34) | 4.63 (9) | 3.17 (10) |
| conformer/auto+tilt | 1.66 (34) | 3.97 (9) | 3.39 (10) |
| relaxed/flat, better face | 0.36 (35) | 2.76 (9) | 3.03 (10) |

### What the relaxed frame carries

The lowest heavy atom of a relaxed pose sits at 1.86 A (median) when the molecule is chemisorbed and at 4.33 A when it detached; E_ads averages -83 kJ/mol for the first and -8 for the second. A straight line on that one height predicts E_ads with a leave-one-out q2 of **0.47**, the chemisorbed indicator alone with 0.35. That is the outcome of the relaxation, not a property of the molecule, and a rule cannot know it; the rows marked *relaxed, lowered* keep the relaxed tilt and face but put every molecule at the common height of 1.9 A, which is what a rule can do.


### Models of E_ads (kJ/mol), all molecules

| kernel | frame | q2 | RMSE | against overlap, relaxed | against surface, relaxed |
|---|---|---:|---:|---|---|
| overlap | relaxed poses as written | +0.505 +- 0.037 | 37.4 | - | -0.15 [-0.27, -0.05] |
| overlap | relaxed, lowered to the common height | +0.266 +- 0.051 | 45.5 | -0.21 [-0.34, -0.07] | -0.36 [-0.53, -0.21] |
| overlap | rule: flat, from the relaxed geometry | +0.260 +- 0.081 | 45.7 | -0.21 [-0.40, -0.03] | -0.36 [-0.54, -0.21] |
| overlap | rule: flat, the other face, relaxed geometry | +0.172 +- 0.080 | 48.4 | -0.28 [-0.46, -0.11] | -0.43 [-0.63, -0.26] |
| overlap | rule: auto by field similarity, refined in the plane, relaxed geometry | +0.256 +- 0.078 | 45.8 | -0.24 [-0.53, -0.05] | -0.39 [-0.66, -0.22] |
| overlap | rule: auto, refined with tilt and height, relaxed geometry | +0.089 +- 0.064 | 50.7 | -0.38 [-0.65, -0.16] | -0.53 [-0.78, -0.36] |
| overlap | rule: flat, from an MMFF conformer | +0.271 +- 0.057 | 45.4 | -0.21 [-0.38, -0.07] | -0.36 [-0.53, -0.22] |
| overlap | rule: auto, refined in the plane, MMFF conformer | +0.119 +- 0.097 | 49.8 | -0.37 [-0.63, -0.14] | -0.52 [-0.77, -0.35] |
| overlap | rule: auto, refined with tilt and height, MMFF conformer | -0.163 +- 0.167 | 57.2 | -0.56 [-0.76, -0.35] | -0.71 [-0.89, -0.56] |
| overlap | MCS alignment of the conformers (the conventional frame) | -0.007 +- 0.092 | 53.3 | -0.47 [-0.64, -0.33] | -0.62 [-0.84, -0.46] |
| surface | relaxed poses as written | +0.655 +- 0.032 | 31.2 | +0.15 [+0.05, +0.27] | - |
| surface | relaxed, lowered to the common height | +0.549 +- 0.021 | 35.7 | +0.05 [-0.10, +0.21] | -0.10 [-0.24, +0.00] |
| surface | rule: flat, from the relaxed geometry | +0.296 +- 0.039 | 44.6 | -0.22 [-0.40, -0.06] | -0.37 [-0.55, -0.24] |
| surface | rule: flat, from an MMFF conformer | +0.310 +- 0.041 | 44.1 | -0.18 [-0.35, -0.04] | -0.33 [-0.52, -0.20] |
| surface | rule: auto, refined with tilt and height, MMFF conformer | -0.009 +- 0.061 | 53.4 | -0.47 [-0.64, -0.31] | -0.62 [-0.76, -0.49] |

### Models of E_ads (kJ/mol), chemisorbed molecules only

| kernel | frame | q2 | RMSE | against overlap, relaxed | against surface, relaxed |
|---|---|---:|---:|---|---|
| overlap | relaxed poses as written | +0.354 +- 0.051 | 38.8 | - | -0.15 [-0.31, -0.01] |
| overlap | relaxed, lowered to the common height | +0.229 +- 0.052 | 42.4 | -0.11 [-0.25, +0.02] | -0.26 [-0.44, -0.08] |
| overlap | rule: flat, from the relaxed geometry | +0.192 +- 0.079 | 43.4 | -0.08 [-0.35, +0.24] | -0.23 [-0.48, +0.00] |
| overlap | rule: flat, the other face, relaxed geometry | -0.061 +- 0.084 | 49.7 | -0.31 [-0.55, -0.04] | -0.47 [-0.69, -0.24] |
| overlap | rule: auto by field similarity, refined in the plane, relaxed geometry | +0.052 +- 0.060 | 47.0 | -0.29 [-0.57, -0.03] | -0.44 [-0.67, -0.24] |
| overlap | rule: auto, refined with tilt and height, relaxed geometry | +0.057 +- 0.060 | 46.9 | -0.25 [-0.53, +0.03] | -0.40 [-0.62, -0.20] |
| overlap | rule: flat, from an MMFF conformer | +0.152 +- 0.078 | 44.5 | -0.17 [-0.36, +0.01] | -0.32 [-0.49, -0.16] |
| overlap | rule: auto, refined in the plane, MMFF conformer | +0.083 +- 0.097 | 46.2 | -0.26 [-0.60, +0.07] | -0.41 [-0.66, -0.18] |
| overlap | rule: auto, refined with tilt and height, MMFF conformer | -0.247 +- 0.164 | 53.9 | -0.55 [-0.88, -0.23] | -0.70 [-0.98, -0.46] |
| overlap | MCS alignment of the conformers (the conventional frame) | +0.008 +- 0.186 | 48.0 | -0.30 [-0.49, -0.09] | -0.45 [-0.65, -0.26] |
| surface | relaxed poses as written | +0.497 +- 0.061 | 34.2 | +0.15 [+0.01, +0.31] | - |
| surface | relaxed, lowered to the common height | +0.404 +- 0.063 | 37.3 | +0.07 [-0.13, +0.29] | -0.08 [-0.25, +0.07] |
| surface | rule: flat, from the relaxed geometry | +0.233 +- 0.074 | 42.3 | -0.11 [-0.39, +0.13] | -0.27 [-0.48, -0.08] |
| surface | rule: flat, from an MMFF conformer | +0.283 +- 0.080 | 40.9 | -0.06 [-0.31, +0.16] | -0.21 [-0.43, -0.00] |
| surface | rule: auto, refined with tilt and height, MMFF conformer | -0.061 +- 0.066 | 49.8 | -0.38 [-0.63, -0.10] | -0.53 [-0.73, -0.33] |

### The co-field in the rule frame (conformer/flat) against the relaxed frame

Correlation over the plane of the two co-fields, per field and height:

| field | z = 2 A | z = 3.5 A |
|---|---:|---:|
| electrostatic | +0.18 | +0.12 |
| steric | +0.23 | +0.00 |
| hydrophobic | +0.59 | -0.22 |
| hba | +0.01 | +0.20 |
| hbd | +0.26 | +0.06 |

Figure: `benchmarks\results\fe110\cofields_benzimidazole.png`

## quinoline: 63 molecules, 31 chemisorbed, reference Oxyquinoline

### Pose recovery (A)

| frame | in-plane RMSD, all | chemisorbed | within 1 A | height RMSD, chemisorbed | rigid floor |
|---|---:|---:|---:|---:|---:|
| relaxed/flat | 3.65 | 3.29 | 6% | 2.05 | 0.00 |
| relaxed/flat-up | 3.82 | 3.43 | 29% | 2.13 | 0.00 |
| relaxed/auto+plane | 3.82 | 3.51 | 27% | 2.27 | 0.00 |
| relaxed/auto+tilt | 4.12 | 3.29 | 13% | 2.72 | 0.00 |
| conformer/flat | 3.78 | 3.17 | 16% | 2.30 | 1.08 |
| conformer/auto+plane | 3.83 | 3.67 | 13% | 2.42 | 1.09 |
| conformer/auto+tilt | 4.22 | 3.45 | 13% | 2.85 | 1.09 |
| relaxed/flat, better face | 2.46 | 1.78 | 35% | 1.57 | 0.00 |

Chemisorbed molecules by the pose UMA kept, in-plane RMSD median (n):

| frame | flat | random | upright |
|---|---:|---:|---:|
| relaxed/flat | 2.83 (17) | 3.29 (7) | 4.54 (7) |
| relaxed/flat-up | 0.99 (17) | 3.51 (7) | 4.60 (7) |
| relaxed/auto+plane | 1.27 (16) | 3.65 (7) | 4.54 (7) |
| relaxed/auto+tilt | 2.33 (16) | 3.32 (7) | 4.28 (7) |
| conformer/flat | 2.50 (17) | 3.67 (7) | 4.42 (7) |
| conformer/auto+plane | 2.64 (16) | 4.04 (7) | 4.42 (7) |
| conformer/auto+tilt | 2.65 (16) | 4.07 (7) | 4.98 (7) |
| relaxed/flat, better face | 0.60 (17) | 2.10 (7) | 4.54 (7) |

### What the relaxed frame carries

The lowest heavy atom of a relaxed pose sits at 1.86 A (median) when the molecule is chemisorbed and at 4.37 A when it detached; E_ads averages -91 kJ/mol for the first and -1 for the second. A straight line on that one height predicts E_ads with a leave-one-out q2 of **0.42**, the chemisorbed indicator alone with 0.33. That is the outcome of the relaxation, not a property of the molecule, and a rule cannot know it; the rows marked *relaxed, lowered* keep the relaxed tilt and face but put every molecule at the common height of 1.9 A, which is what a rule can do.


### Models of E_ads (kJ/mol), all molecules

| kernel | frame | q2 | RMSE | against overlap, relaxed | against surface, relaxed |
|---|---|---:|---:|---|---|
| overlap | relaxed poses as written | +0.560 +- 0.046 | 49.0 | - | +0.03 [-0.16, +0.26] |
| overlap | relaxed, lowered to the common height | -0.011 +- 0.057 | 74.4 | -0.53 [-0.75, -0.39] | -0.50 [-0.80, -0.20] |
| overlap | rule: flat, from the relaxed geometry | -0.008 +- 0.057 | 74.3 | -0.56 [-0.73, -0.40] | -0.53 [-0.82, -0.19] |
| overlap | rule: flat, the other face, relaxed geometry | +0.016 +- 0.034 | 73.4 | -0.51 [-0.75, -0.33] | -0.48 [-0.80, -0.12] |
| overlap | rule: auto by field similarity, refined in the plane, relaxed geometry | +0.270 +- 0.076 | 63.1 | -0.23 [-0.47, -0.05] | -0.20 [-0.52, +0.09] |
| overlap | rule: auto, refined with tilt and height, relaxed geometry | -0.048 +- 0.078 | 75.7 | -0.58 [-0.86, -0.39] | -0.56 [-0.92, -0.21] |
| overlap | rule: flat, from an MMFF conformer | -0.027 +- 0.034 | 75.0 | -0.58 [-0.77, -0.45] | -0.56 [-0.81, -0.25] |
| overlap | rule: auto, refined in the plane, MMFF conformer | -0.004 +- 0.042 | 74.1 | -0.53 [-0.73, -0.40] | -0.51 [-0.77, -0.23] |
| overlap | rule: auto, refined with tilt and height, MMFF conformer | +0.099 +- 0.070 | 70.2 | -0.42 [-0.63, -0.29] | -0.40 [-0.68, -0.12] |
| overlap | MCS alignment of the conformers (the conventional frame) | -0.069 +- 0.088 | 76.4 | -0.61 [-0.83, -0.47] | -0.59 [-0.88, -0.26] |
| surface | relaxed poses as written | +0.517 +- 0.042 | 51.4 | -0.03 [-0.26, +0.16] | - |
| surface | relaxed, lowered to the common height | +0.337 +- 0.034 | 60.2 | -0.21 [-0.40, -0.07] | -0.19 [-0.36, -0.01] |
| surface | rule: flat, from the relaxed geometry | +0.019 +- 0.092 | 73.2 | -0.53 [-0.78, -0.33] | -0.50 [-0.85, -0.13] |
| surface | rule: flat, from an MMFF conformer | -0.003 +- 0.049 | 74.1 | -0.55 [-0.76, -0.39] | -0.53 [-0.81, -0.20] |
| surface | rule: auto, refined with tilt and height, MMFF conformer | -0.017 +- 0.082 | 74.6 | -0.56 [-0.76, -0.40] | -0.54 [-0.80, -0.21] |

### Models of E_ads (kJ/mol), chemisorbed molecules only

| kernel | frame | q2 | RMSE | against overlap, relaxed | against surface, relaxed |
|---|---|---:|---:|---|---|
| overlap | relaxed poses as written | +0.335 +- 0.069 | 67.2 | - | +0.17 [-0.13, +0.51] |
| overlap | relaxed, lowered to the common height | +0.138 +- 0.084 | 76.5 | -0.18 [-0.33, -0.09] | -0.01 [-0.38, +0.37] |
| overlap | rule: flat, from the relaxed geometry | -0.184 +- 0.144 | 89.6 | -0.48 [-0.86, -0.18] | -0.31 [-0.85, +0.26] |
| overlap | rule: flat, the other face, relaxed geometry | +0.035 +- 0.123 | 80.9 | -0.28 [-0.63, -0.01] | -0.12 [-0.61, +0.42] |
| overlap | rule: auto by field similarity, refined in the plane, relaxed geometry | +0.111 +- 0.088 | 77.7 | -0.17 [-0.43, +0.08] | +0.00 [-0.42, +0.46] |
| overlap | rule: auto, refined with tilt and height, relaxed geometry | -0.095 +- 0.048 | 86.3 | -0.43 [-0.76, -0.19] | -0.26 [-0.79, +0.25] |
| overlap | rule: flat, from an MMFF conformer | -0.141 +- 0.097 | 88.0 | -0.43 [-0.77, -0.18] | -0.26 [-0.75, +0.23] |
| overlap | rule: auto, refined in the plane, MMFF conformer | +0.007 +- 0.077 | 82.2 | -0.33 [-0.59, -0.12] | -0.16 [-0.58, +0.27] |
| overlap | rule: auto, refined with tilt and height, MMFF conformer | -0.129 +- 0.094 | 87.6 | -0.43 [-0.88, -0.12] | -0.26 [-0.88, +0.27] |
| overlap | MCS alignment of the conformers (the conventional frame) | -0.325 +- 0.170 | 94.8 | -0.61 [-1.28, -0.26] | -0.44 [-1.22, +0.14] |
| surface | relaxed poses as written | +0.150 +- 0.115 | 75.9 | -0.17 [-0.51, +0.13] | - |
| surface | relaxed, lowered to the common height | +0.165 +- 0.090 | 75.3 | -0.15 [-0.38, +0.06] | +0.02 [-0.12, +0.16] |
| surface | rule: flat, from the relaxed geometry | -0.106 +- 0.146 | 86.6 | -0.40 [-0.97, -0.04] | -0.23 [-0.94, +0.41] |
| surface | rule: flat, from an MMFF conformer | -0.028 +- 0.117 | 83.5 | -0.35 [-0.59, -0.15] | -0.18 [-0.59, +0.27] |
| surface | rule: auto, refined with tilt and height, MMFF conformer | -0.114 +- 0.104 | 87.0 | -0.46 [-0.79, -0.17] | -0.29 [-0.81, +0.27] |

### The co-field in the rule frame (conformer/flat) against the relaxed frame

Correlation over the plane of the two co-fields, per field and height:

| field | z = 2 A | z = 3.5 A |
|---|---:|---:|
| electrostatic | +0.25 | +0.25 |
| steric | +0.61 | -0.29 |
| hydrophobic | +0.30 | +0.01 |
| hba | +0.30 | +0.11 |
| hbd | +0.58 | +0.05 |

Figure: `benchmarks\results\fe110\cofields_quinoline.png`

