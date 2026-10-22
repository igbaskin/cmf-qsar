# Aligned CMF, topomer CMF and the frameless kernels on the corrosion-inhibitor series, protonated for 1 M HCl

Series: one scaffold-tree node each, activity = MWILA potency. The structures are 2D,
so the alignment is the program's own: every molecule superimposed on the most active
large member of its series through the scaffold atoms. Same fields (electrostatic, steric, hydrophobic, hba, hbd),
same alpha and lambda grids, same 10-fold CV, each model choosing its own (alpha,
lambda) on the CV it reports.

| series | n | scaffold | fit RMSD | q2 aligned | q2 topomer | q2 invariant | q2 counts | cuttable |
|---|---|---|---|---|---|---|---|---|
| 0_c1ccccc1 | 244 | `c1ccccc1` | 0.01 A | +0.413 | +0.503 | +0.431 | +0.439 | 232/244 |
| 1_c1ccncc1 | 109 | `c1ccncc1` | 0.03 A | +0.341 | -0.034 | +0.205 | +0.067 | 42/109 |
| 2_c1c_nH_cn1 | 96 | `c1c[nH]cn1` | 0.02 A | +0.502 | +0.554 | +0.481 | +0.355 | 20/96 |
| 157_c1ccc2_nH_cnc2c1 | 73 | `c1ccc2[nH]cnc2c1` | 0.01 A | +0.346 | +0.429 | +0.360 | +0.263 | 73/73 |
| 158_c1ccc2ncccc2c1 | 65 | `c1ccc2ncccc2c1` | 0.03 A | +0.149 | +0.323 | +0.034 | +0.019 | 60/65 |
| 4_c1nnc_nH_1 | 40 | `c1nnc[nH]1` | 0.07 A | +0.188 | -0.009 | +0.482 | +0.483 | 35/40 |
| 8_c1cn_nH_c1 | 30 | `c1cn[nH]c1` | 0.06 A | +0.723 | +0.331 | +0.518 | +0.658 | 28/30 |
| 7_c1nncs1 | 29 | `c1nncs1` | 0.04 A | +0.279 | +0.264 | +0.036 | +0.007 | 29/29 |
| 12_c1ccoc1 | 28 | `c1ccoc1` | 0.00 A | +0.658 | +0.552 | +0.413 | +0.388 | 28/28 |
| 159_C_Nc1ccccc1_c1ccccc1 | 28 | `C(=Nc1ccccc1)c1ccccc1` | 1.66 A | +0.261 | +0.290 | +0.220 | -0.013 | 28/28 |
| 3_c1cscn1 | 27 | `c1cscn1` | 0.02 A | +0.465 | - | +0.462 | +0.164 | 7/27 |
| 13_c1c_nH_nn1 | 25 | `c1c[nH]nn1` | 0.09 A | +0.519 | +0.617 | +0.781 | +0.340 | 20/25 |
| 9_c1nc_nH_n1 | 25 | `c1nc[nH]n1` | 0.07 A | +0.090 | +0.051 | +0.042 | +0.289 | 22/25 |
| 14_O_c1cc_nH_c_O_nH_1 | 24 | `O=c1cc[nH]c(=O)[nH]1` | 0.14 A | +0.711 | - | +0.722 | +0.742 | 6/24 |
| 11_c1c_nH_c_nH_1 | 23 | `c1c[nH+]c[nH]1` | 0.03 A | +0.320 | +0.547 | +0.073 | +0.448 | 18/23 |
| 161_c1ccc2scnc2c1 | 20 | `c1ccc2scnc2c1` | 0.03 A | +0.410 | +0.507 | +0.381 | +0.026 | 20/20 |
| 17_c1nnco1 | 17 | `c1nnco1` | 0.01 A | +0.497 | +0.124 | +0.166 | +0.338 | 17/17 |
| 21_O_C1CCN1 | 17 | `O=C1CCN1` | 0.06 A | +0.399 | - | +0.220 | +0.294 | 0/17 |
| 22_C1CNOC1 | 16 | `C1CNOC1` | 0.18 A | +0.398 | - | +0.619 | +0.652 | 4/16 |
| 5_c1cncnc1 | 16 | `c1cncnc1` | 0.02 A | -0.023 | - | -0.044 | -0.077 | 14/16 |
| 160_c1ccc2ccccc2c1 | 15 | `c1ccc2ccccc2c1` | 0.02 A | -0.125 | - | +0.057 | -0.217 | 13/15 |
| 165_O_c1cc_nH_n1_c1ccccc1 | 15 | `O=c1cc[nH]n1-c1ccccc1` | 0.43 A | -0.028 | -0.046 | +0.498 | -0.158 | 15/15 |
| 20_O_c1cc_nH_nH_1 | 15 | `O=c1cc[nH][nH]1` | 0.01 A | -0.045 | -0.041 | +0.498 | -0.158 | 15/15 |

On the 17 series with enough cuttable molecules, measured on exactly those molecules:

| series | whole molecules | substituents, as aligned | substituents, posed by rule | the cut costs | the pose costs |
|---|---|---|---|---|---|
| 0_c1ccccc1 | +0.437 | +0.399 | +0.503 | +0.038 | -0.104 |
| 1_c1ccncc1 | -0.093 | -0.047 | -0.034 | -0.046 | -0.013 |
| 2_c1c_nH_cn1 | +0.563 | +0.458 | +0.554 | +0.105 | -0.096 |
| 157_c1ccc2_nH_cnc2c1 | +0.346 | +0.318 | +0.429 | +0.028 | -0.111 |
| 158_c1ccc2ncccc2c1 | +0.159 | +0.333 | +0.323 | -0.174 | +0.010 |
| 4_c1nnc_nH_1 | +0.366 | -0.076 | -0.009 | +0.441 | -0.067 |
| 8_c1cn_nH_c1 | +0.701 | +0.489 | +0.331 | +0.211 | +0.158 |
| 7_c1nncs1 | +0.279 | +0.237 | +0.264 | +0.042 | -0.026 |
| 12_c1ccoc1 | +0.658 | +0.701 | +0.552 | -0.043 | +0.149 |
| 159_C_Nc1ccccc1_c1ccccc1 | +0.261 | +0.399 | +0.290 | -0.138 | +0.110 |
| 13_c1c_nH_nn1 | +0.471 | +0.421 | +0.617 | +0.050 | -0.196 |
| 9_c1nc_nH_n1 | -0.042 | -0.038 | +0.051 | -0.004 | -0.090 |
| 11_c1c_nH_c_nH_1 | +0.214 | +0.206 | +0.547 | +0.008 | -0.341 |
| 161_c1ccc2scnc2c1 | +0.410 | +0.263 | +0.507 | +0.147 | -0.244 |
| 17_c1nnco1 | +0.497 | +0.238 | +0.124 | +0.259 | +0.114 |
| 165_O_c1cc_nH_n1_c1ccccc1 | -0.028 | -0.054 | -0.046 | +0.026 | -0.008 |
| 20_O_c1cc_nH_nH_1 | -0.045 | -0.053 | -0.041 | +0.008 | -0.012 |

Mean q2: aligned +0.303, topomer +0.292 (difference +0.011); topomer wins 10 of 17. Mean price of the cut +0.056, of the pose -0.045.

Over all 23 series: aligned +0.324, invariant +0.333, counts +0.233; the invariant kernel wins 9, the aligned model 14.
