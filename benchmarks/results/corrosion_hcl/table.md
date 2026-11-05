# Aligned CMF, topomer CMF and the frameless kernels on the corrosion-inhibitor series, protonated for 1 M HCl

Series: one scaffold-tree node each, activity = MWILA potency. The structures are 2D,
so the alignment is the program's own: every molecule superimposed on the most active
large member of its series through the scaffold atoms. Same fields (electrostatic, steric, hydrophobic, hba, hbd),
same alpha and lambda grids, same 10-fold CV, each model choosing its own (alpha,
lambda) on the CV it reports.

| series | n | scaffold | fit RMSD | q2 aligned | q2 topomer | q2 invariant | q2 invariant, aligned blocks | q2 counts | cuttable |
|---|---|---|---|---|---|---|---|---|---|
| 0_c1ccccc1 | 244 | `c1ccccc1` | 0.01 A | +0.415 | +0.496 | +0.430 | +0.389 | +0.439 | 232/244 |
| 1_c1ccncc1 | 109 | `c1ccncc1` | 0.03 A | +0.341 | -0.046 | +0.205 | +0.190 | +0.067 | 42/109 |
| 2_c1c_nH_cn1 | 96 | `c1c[nH]cn1` | 0.02 A | +0.502 | +0.547 | +0.481 | +0.432 | +0.355 | 20/96 |
| 157_c1ccc2_nH_cnc2c1 | 73 | `c1ccc2[nH]cnc2c1` | 0.01 A | +0.346 | +0.526 | +0.360 | +0.358 | +0.263 | 73/73 |
| 158_c1ccc2ncccc2c1 | 65 | `c1ccc2ncccc2c1` | 0.03 A | +0.149 | +0.482 | +0.034 | +0.020 | +0.019 | 60/65 |
| 4_c1nnc_nH_1 | 40 | `c1nnc[nH]1` | 0.07 A | +0.188 | +0.111 | +0.482 | +0.407 | +0.483 | 35/40 |
| 8_c1cn_nH_c1 | 30 | `c1cn[nH]c1` | 0.06 A | +0.723 | +0.523 | +0.518 | +0.519 | +0.658 | 28/30 |
| 7_c1nncs1 | 29 | `c1nncs1` | 0.04 A | +0.279 | +0.187 | +0.036 | +0.008 | +0.007 | 29/29 |
| 12_c1ccoc1 | 28 | `c1ccoc1` | 0.00 A | +0.658 | +0.410 | +0.413 | +0.373 | +0.388 | 28/28 |
| 159_C_Nc1ccccc1_c1ccccc1 | 28 | `C(=Nc1ccccc1)c1ccccc1` | 1.66 A | +0.261 | +0.355 | +0.220 | +0.097 | -0.013 | 28/28 |
| 3_c1cscn1 | 27 | `c1cscn1` | 0.02 A | +0.465 | - | +0.462 | +0.469 | +0.164 | 7/27 |
| 13_c1c_nH_nn1 | 25 | `c1c[nH]nn1` | 0.09 A | +0.519 | +0.713 | +0.781 | +0.753 | +0.340 | 20/25 |
| 9_c1nc_nH_n1 | 25 | `c1nc[nH]n1` | 0.07 A | +0.090 | +0.024 | +0.042 | +0.043 | +0.289 | 22/25 |
| 14_O_c1cc_nH_c_O_nH_1 | 24 | `O=c1cc[nH]c(=O)[nH]1` | 0.14 A | +0.711 | - | +0.722 | +0.743 | +0.742 | 6/24 |
| 11_c1c_nH_c_nH_1 | 23 | `c1c[nH+]c[nH]1` | 0.03 A | +0.320 | +0.569 | +0.073 | +0.093 | +0.448 | 18/23 |
| 161_c1ccc2scnc2c1 | 20 | `c1ccc2scnc2c1` | 0.03 A | +0.410 | +0.470 | +0.381 | +0.396 | +0.026 | 20/20 |
| 17_c1nnco1 | 17 | `c1nnco1` | 0.01 A | +0.497 | +0.267 | +0.166 | +0.227 | +0.338 | 17/17 |
| 21_O_C1CCN1 | 17 | `O=C1CCN1` | 0.06 A | +0.399 | - | +0.220 | +0.209 | +0.294 | 0/17 |
| 22_C1CNOC1 | 16 | `C1CNOC1` | 0.18 A | +0.398 | - | +0.619 | +0.582 | +0.652 | 4/16 |
| 5_c1cncnc1 | 16 | `c1cncnc1` | 0.02 A | -0.023 | - | -0.044 | -0.046 | -0.077 | 14/16 |
| 160_c1ccc2ccccc2c1 | 15 | `c1ccc2ccccc2c1` | 0.02 A | -0.125 | - | +0.057 | -0.186 | -0.217 | 13/15 |
| 165_O_c1cc_nH_n1_c1ccccc1 | 15 | `O=c1cc[nH]n1-c1ccccc1` | 0.43 A | -0.028 | +0.228 | +0.498 | +0.484 | -0.158 | 15/15 |
| 20_O_c1cc_nH_nH_1 | 15 | `O=c1cc[nH][nH]1` | 0.01 A | -0.045 | +0.229 | +0.498 | +0.484 | -0.158 | 15/15 |

On the 17 series with enough cuttable molecules, measured on exactly those molecules:

| series | whole molecules | substituents, as aligned | substituents, posed by rule | the cut costs | the pose costs |
|---|---|---|---|---|---|
| 0_c1ccccc1 | +0.440 | +0.391 | +0.496 | +0.049 | -0.105 |
| 1_c1ccncc1 | -0.093 | -0.055 | -0.046 | -0.038 | -0.009 |
| 2_c1c_nH_cn1 | +0.563 | +0.376 | +0.547 | +0.186 | -0.171 |
| 157_c1ccc2_nH_cnc2c1 | +0.346 | +0.421 | +0.526 | -0.074 | -0.106 |
| 158_c1ccc2ncccc2c1 | +0.159 | +0.374 | +0.482 | -0.215 | -0.108 |
| 4_c1nnc_nH_1 | +0.366 | -0.063 | +0.111 | +0.428 | -0.174 |
| 8_c1cn_nH_c1 | +0.701 | +0.466 | +0.523 | +0.234 | -0.057 |
| 7_c1nncs1 | +0.279 | +0.237 | +0.187 | +0.043 | +0.050 |
| 12_c1ccoc1 | +0.658 | +0.579 | +0.410 | +0.079 | +0.169 |
| 159_C_Nc1ccccc1_c1ccccc1 | +0.261 | +0.361 | +0.355 | -0.100 | +0.006 |
| 13_c1c_nH_nn1 | +0.471 | +0.723 | +0.713 | -0.252 | +0.010 |
| 9_c1nc_nH_n1 | -0.042 | -0.067 | +0.024 | +0.025 | -0.091 |
| 11_c1c_nH_c_nH_1 | +0.214 | +0.166 | +0.569 | +0.048 | -0.403 |
| 161_c1ccc2scnc2c1 | +0.410 | +0.277 | +0.470 | +0.133 | -0.193 |
| 17_c1nnco1 | +0.497 | +0.418 | +0.267 | +0.079 | +0.151 |
| 165_O_c1cc_nH_n1_c1ccccc1 | -0.028 | -0.030 | +0.228 | +0.002 | -0.258 |
| 20_O_c1cc_nH_nH_1 | -0.045 | -0.044 | +0.229 | -0.001 | -0.274 |

Mean q2: aligned +0.303, topomer +0.358 (difference -0.055); topomer wins 11 of 17. Mean cost of the cut +0.037, of the pose -0.092.

Over all 23 series: aligned +0.324, invariant +0.333, counts +0.233; the invariant kernel wins 9, the aligned model 14.
