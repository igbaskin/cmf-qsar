# Aligned CMF, topomer CMF and the frameless kernels on the corrosion-inhibitor series

Series: one scaffold-tree node each, activity = MWILA potency. The structures are 2D,
so the alignment is the program's own: every molecule superimposed on the most active
large member of its series through the scaffold atoms. Same fields (electrostatic, steric, hydrophobic, hba, hbd),
same alpha and lambda grids, same 10-fold CV, each model choosing its own (alpha,
lambda) on the CV it reports.

| series | n | scaffold | fit RMSD | q2 aligned | q2 topomer | q2 invariant | q2 invariant, aligned blocks | q2 counts | cuttable |
|---|---|---|---|---|---|---|---|---|---|
| 0_c1ccccc1 | 244 | `c1ccccc1` | 0.02 A | +0.386 | +0.466 | +0.376 | - | +0.437 | 232/244 |
| 1_c1ccncc1 | 109 | `c1ccncc1` | 0.02 A | +0.363 | -0.027 | +0.352 | +0.364 | +0.072 | 42/109 |
| 2_c1c_nH_cn1 | 96 | `c1c[nH]cn1` | 0.01 A | +0.460 | +0.244 | +0.435 | - | +0.335 | 20/96 |
| 157_c1ccc2_nH_cnc2c1 | 73 | `c1ccc2[nH]cnc2c1` | 0.01 A | +0.344 | +0.436 | +0.238 | - | +0.268 | 73/73 |
| 158_c1ccc2ncccc2c1 | 65 | `c1ccc2ncccc2c1` | 0.02 A | +0.136 | +0.224 | +0.269 | - | +0.032 | 60/65 |
| 4_c1nnc_nH_1 | 40 | `c1nnc[nH]1` | 0.02 A | +0.142 | +0.052 | +0.250 | - | +0.485 | 35/40 |
| 8_c1cn_nH_c1 | 30 | `c1cn[nH]c1` | 0.06 A | +0.487 | +0.319 | +0.529 | - | +0.644 | 28/30 |
| 7_c1nncs1 | 29 | `c1nncs1` | 0.01 A | -0.037 | +0.110 | -0.014 | - | -0.044 | 29/29 |
| 12_c1ccoc1 | 28 | `c1ccoc1` | 0.00 A | +0.151 | +0.396 | -0.036 | - | +0.339 | 28/28 |
| 159_C_Nc1ccccc1_c1ccccc1 | 28 | `C(=Nc1ccccc1)c1ccccc1` | 1.66 A | +0.408 | +0.255 | +0.267 | - | -0.086 | 28/28 |
| 3_c1cscn1 | 27 | `c1cscn1` | 0.01 A | +0.537 | - | +0.512 | - | +0.158 | 7/27 |
| 13_c1c_nH_nn1 | 25 | `c1c[nH]nn1` | 0.02 A | +0.654 | +0.606 | +0.569 | - | +0.329 | 20/25 |
| 9_c1nc_nH_n1 | 25 | `c1nc[nH]n1` | 0.08 A | +0.339 | +0.015 | +0.106 | - | +0.289 | 22/25 |
| 14_O_c1cc_nH_c_O_nH_1 | 24 | `O=c1cc[nH]c(=O)[nH]1` | 0.12 A | +0.446 | - | +0.425 | - | +0.737 | 6/24 |
| 11_c1c_nH_c_nH_1 | 23 | `c1c[nH+]c[nH]1` | 0.03 A | +0.256 | +0.547 | -0.034 | - | +0.448 | 18/23 |
| 161_c1ccc2scnc2c1 | 20 | `c1ccc2scnc2c1` | 0.01 A | +0.689 | +0.548 | +0.412 | - | +0.043 | 20/20 |
| 17_c1nnco1 | 17 | `c1nnco1` | 0.00 A | -0.000 | +0.072 | +0.320 | - | +0.346 | 17/17 |
| 21_O_C1CCN1 | 17 | `O=C1CCN1` | 0.11 A | +0.398 | - | +0.070 | - | +0.030 | 0/17 |
| 22_C1CNOC1 | 16 | `C1CNOC1` | 0.12 A | +0.001 | - | +0.610 | - | -0.004 | 4/16 |
| 5_c1cncnc1 | 16 | `c1cncnc1` | 0.03 A | +0.073 | - | +0.120 | - | -0.082 | 14/16 |
| 160_c1ccc2ccccc2c1 | 15 | `c1ccc2ccccc2c1` | 0.02 A | -0.163 | - | -0.215 | - | -0.217 | 13/15 |
| 165_O_c1cc_nH_n1_c1ccccc1 | 15 | `O=c1cc[nH]n1-c1ccccc1` | 0.43 A | +0.011 | -0.017 | +0.422 | - | -0.111 | 15/15 |
| 20_O_c1cc_nH_nH_1 | 15 | `O=c1cc[nH][nH]1` | 0.01 A | +0.104 | -0.013 | +0.422 | - | -0.111 | 15/15 |

On the 17 series with enough cuttable molecules, measured on exactly those molecules:

| series | whole molecules | substituents, as aligned | substituents, posed by rule | the cut costs | the pose costs |
|---|---|---|---|---|---|
| 0_c1ccccc1 | +0.388 | +0.345 | +0.466 | +0.043 | -0.121 |
| 1_c1ccncc1 | -0.051 | +0.183 | -0.027 | -0.234 | +0.209 |
| 2_c1c_nH_cn1 | +0.518 | +0.402 | +0.244 | +0.115 | +0.158 |
| 157_c1ccc2_nH_cnc2c1 | +0.344 | +0.288 | +0.436 | +0.056 | -0.148 |
| 158_c1ccc2ncccc2c1 | +0.162 | +0.181 | +0.224 | -0.019 | -0.044 |
| 4_c1nnc_nH_1 | +0.135 | -0.022 | +0.052 | +0.157 | -0.074 |
| 8_c1cn_nH_c1 | +0.450 | +0.313 | +0.319 | +0.137 | -0.007 |
| 7_c1nncs1 | -0.037 | +0.115 | +0.110 | -0.153 | +0.005 |
| 12_c1ccoc1 | +0.151 | +0.129 | +0.396 | +0.022 | -0.268 |
| 159_C_Nc1ccccc1_c1ccccc1 | +0.408 | +0.348 | +0.255 | +0.060 | +0.093 |
| 13_c1c_nH_nn1 | +0.642 | +0.353 | +0.606 | +0.289 | -0.253 |
| 9_c1nc_nH_n1 | +0.095 | -0.048 | +0.015 | +0.142 | -0.063 |
| 11_c1c_nH_c_nH_1 | +0.214 | +0.206 | +0.547 | +0.008 | -0.341 |
| 161_c1ccc2scnc2c1 | +0.689 | +0.463 | +0.548 | +0.226 | -0.085 |
| 17_c1nnco1 | -0.000 | -0.015 | +0.072 | +0.015 | -0.087 |
| 165_O_c1cc_nH_n1_c1ccccc1 | +0.011 | -0.023 | -0.017 | +0.034 | -0.006 |
| 20_O_c1cc_nH_nH_1 | +0.104 | -0.025 | -0.013 | +0.129 | -0.012 |

Mean q2: aligned +0.248, topomer +0.249 (difference -0.001); topomer wins 8 of 17. Mean price of the cut +0.061, of the pose -0.061.

Over all 23 series: aligned +0.269, invariant +0.278, counts +0.189; the invariant kernel wins 9, the aligned model 14.
