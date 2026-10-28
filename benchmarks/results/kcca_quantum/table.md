# What the quantum fields see that shape and charge do not

View A is the three reactivity fields (f+, f− and the dual descriptor, one Gaussian per atom weighted by the condensed index), view B the five empirical fields, both at the attenuation the evidence picks for the empirical model. *Held-out ρ* is the mean canonical correlation over all pairs on held-out molecules; *real* counts the pairs above the 95th percentile of a shuffled pairing; *contained* counts the directions of the quantum view of which the empirical view explains more than half the variance on held-out molecules. The q² values are kernel ridge regression on the whole quantum view, on its contained part and on its own part, with the regularisation chosen inside the folds; the last two columns are the empirical view's whole and its own part relative to the quantum view.

## ORCA: Hirshfeld indices on the corrosion series

| series | n | α | mean ρ | real | contained | q² quantum | q² contained | q² quantum's own | own − contained [95 %] | q² empirical | q² empirical's own |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 7_c1nncs1 | 29 | 2 | +0.476 | 17/23 | 4/28 | -0.132 | -0.120 | -0.169 | -0.050 [-0.441, +0.180] | -0.130 | -0.128 |
| 12_c1ccoc1 | 27 | 2 | +0.504 | 16/21 | 3/26 | -0.350 | -0.664 | -0.182 | +0.483 [-0.087, +1.858] | -0.060 | +0.012 |
| 13_c1c_nH_nn1 | 25 | 0.3 | +0.425 | 12/19 | 3/24 | +0.617 | +0.266 | -0.020 | -0.286 [-0.646, +0.121] | +0.603 | +0.048 |
| 161_c1ccc2scnc2c1 | 20 | 0.5 | +0.524 | 10/15 | 4/19 | +0.145 | +0.093 | -0.013 | -0.105 [-0.606, +0.247] | +0.643 | +0.028 |
| **mean** |  |  |  |  |  | **+0.070** | **-0.106** | **-0.096** | **+0.010** | **+0.264** | **-0.010** |

Over 4 series: 70 % of the canonical pairs hold out above the shuffled pairing, and the empirical view contains 15 % of the quantum view's directions on average. The quantum view's own part predicts the activity at q² **-0.096** on average against **-0.106** for its contained part and **+0.070** for the whole quantum view; the difference (own minus contained) is **+0.010**, with its interval above zero on 0 and below zero on 0 of the 4 series. The empirical view alone reaches **+0.264**, and its own part relative to the quantum view **-0.010**.

## GFN2-xTB: every series the cheap quantum stage was run on

| series | n | α | mean ρ | real | contained | q² quantum | q² contained | q² quantum's own | own − contained [95 %] | q² empirical | q² empirical's own |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| hammett_ens | 464 | 2 | +0.437 | 267/281 | 48/318 | +0.630 | +0.559 | +0.423 | -0.136 [-0.215, -0.066] | +0.893 | +0.870 |
| hammett_sym | 380 | 2 | +0.449 | 207/221 | 37/246 | +0.601 | +0.342 | +0.428 | +0.085 [+0.003, +0.156] | +0.860 | +0.820 |
| DHFR | 361 | 2 | +0.502 | 277/288 | 82/360 | +0.683 | +0.631 | +0.561 | -0.070 [-0.127, -0.014] | +0.639 | +0.579 |
| COX2 | 282 | 2 | +0.439 | 195/225 | 37/280 | +0.380 | +0.436 | +0.338 | -0.098 [-0.166, -0.037] | +0.520 | +0.503 |
| BZR | 147 | 2 | +0.578 | 112/117 | 43/146 | +0.238 | +0.250 | +0.086 | -0.163 [-0.285, -0.051] | +0.394 | +0.185 |
| ACE | 114 | 1 | +0.490 | 83/90 | 25/112 | +0.706 | +0.628 | +0.586 | -0.042 [-0.140, +0.050] | +0.538 | +0.456 |
| AChE | 111 | 2 | +0.563 | 82/88 | 29/110 | +0.537 | +0.445 | +0.098 | -0.347 [-0.551, -0.084] | +0.535 | +0.035 |
| hammett | 95 | 1 | +0.253 | 42/75 | 4/94 | +0.044 | +0.171 | -0.004 | -0.175 [-0.623, +0.129] | +0.599 | +0.626 |
| THR | 88 | 0.5 | +0.526 | 59/70 | 23/87 | +0.413 | +0.381 | +0.028 | -0.353 [-0.521, -0.186] | +0.593 | +0.279 |
| pfkfb3_automap | 87 | 2 | +0.554 | 63/69 | 23/84 | +0.531 | +0.398 | +0.173 | -0.225 [-0.374, -0.086] | +0.457 | -0.017 |
| THER | 75 | 1 | +0.580 | 55/59 | 21/74 | +0.337 | +0.454 | +0.132 | -0.322 [-0.474, -0.193] | +0.337 | +0.133 |
| mcl1_extra_flips | 69 | 2 | +0.661 | 51/55 | 28/68 | +0.668 | +0.137 | +0.255 | +0.118 [-0.296, +0.762] | +0.632 | +0.080 |
| GPB | 66 | 2 | +0.499 | 48/52 | 9/65 | +0.382 | +0.408 | -0.027 | -0.435 [-0.654, -0.178] | +0.498 | +0.434 |
| hif2a_automap | 51 | 1 | +0.617 | 34/40 | 20/50 | +0.253 | +0.333 | -0.129 | -0.463 [-0.740, -0.188] | +0.738 | +0.544 |
| ox2_hip_custcore | 51 | 2 | +0.507 | 35/40 | 4/50 | -0.078 | +0.001 | -0.286 | -0.287 [-0.863, +0.125] | -0.164 | -0.101 |
| syk_4puz_fullmap | 51 | 2 | +0.472 | 33/40 | 6/50 | +0.242 | +0.204 | -0.063 | -0.267 [-0.633, +0.058] | +0.372 | +0.092 |
| syk | 44 | 2 | +0.523 | 31/35 | 6/43 | +0.352 | +0.349 | -0.235 | -0.585 [-0.899, -0.397] | +0.355 | -0.049 |
| hif2a | 42 | 0.3 | +0.461 | 27/33 | 5/41 | -0.367 | -0.098 | -0.305 | -0.207 [-1.170, +0.247] | +0.147 | -0.012 |
| cdk8_5cei_new_helix_loop_extra | 41 | 2 | +0.592 | 28/32 | 12/40 | +0.612 | +0.368 | +0.181 | -0.187 [-0.473, +0.242] | +0.490 | -0.007 |
| pfkfb3 | 40 | 2 | +0.463 | 22/31 | 4/39 | +0.198 | -0.093 | +0.290 | +0.383 [+0.173, +0.590] | -0.111 | -0.146 |
| throm_nozob_hip75 | 39 | 2 | +0.546 | 26/31 | 8/38 | +0.584 | +0.424 | -0.054 | -0.478 [-0.679, -0.294] | +0.549 | -0.003 |
| hif2a-jmc2023 | 38 | 1 | +0.511 | 22/30 | 11/37 | +0.447 | +0.374 | -0.105 | -0.478 [-0.867, -0.180] | +0.583 | +0.185 |
| jnk1_manual_flips | 38 | 2 | +0.688 | 26/30 | 20/37 | +0.613 | +0.599 | -0.196 | -0.794 [-1.008, -0.525] | +0.786 | +0.562 |
| bace | 36 | 0.5 | +0.599 | 24/28 | 10/35 | +0.133 | -0.060 | +0.072 | +0.132 [-0.198, +0.541] | +0.335 | +0.172 |
| pde10 | 35 | 2 | +0.603 | 25/27 | 10/34 | +0.043 | +0.226 | -0.084 | -0.310 [-0.518, -0.034] | +0.237 | +0.243 |
| eg5_extraprotomers | 34 | 0.5 | +0.531 | 21/27 | 6/33 | +0.077 | -0.620 | +0.138 | +0.757 [+0.081, +2.176] | +0.377 | +0.315 |
| p38 | 34 | 0.3 | +0.294 | 13/27 | 4/33 | +0.293 | +0.044 | -0.201 | -0.244 [-0.942, +0.162] | +0.218 | -0.141 |
| bace_ciordia_retro | 33 | 2 | +0.500 | 20/26 | 7/32 | -0.274 | +0.265 | -0.207 | -0.472 [-1.125, +0.028] | +0.299 | -0.098 |
| cdk8 | 33 | 1 | +0.398 | 15/26 | 3/32 | +0.612 | -0.303 | -1.317 | -1.014 [-4.061, +0.583] | +0.384 | +0.002 |
| tnks2_fullmap | 33 | 2 | +0.658 | 24/26 | 13/32 | +0.455 | +0.104 | -0.221 | -0.325 [-0.863, +0.020] | +0.374 | -0.128 |
| bace_hunt | 32 | 2 | +0.512 | 19/25 | 5/31 | +0.072 | -0.104 | -0.005 | +0.099 [-0.265, +0.348] | +0.311 | +0.395 |
| a2a_hip278 | 29 | 2 | +0.744 | 20/23 | 16/28 | +0.405 | +0.571 | -0.119 | -0.690 [-0.951, -0.505] | +0.635 | +0.155 |
| renin_customcore | 29 | 0.3 | +0.472 | 16/23 | 3/28 | -0.138 | -0.086 | +0.086 | +0.172 [-0.068, +0.434] | +0.349 | +0.370 |
| eg5 | 28 | 0.5 | +0.582 | 19/22 | 7/27 | -0.314 | +0.035 | -0.216 | -0.251 [-1.078, +0.168] | -0.169 | +0.451 |
| eg5_alternativeloop | 28 | 0.5 | +0.582 | 19/22 | 7/27 | -0.314 | +0.035 | -0.216 | -0.251 [-1.078, +0.168] | -0.169 | +0.451 |
| fxa_yoshikawa_set | 27 | 2 | +0.544 | 17/21 | 4/26 | -0.111 | -0.047 | -0.200 | -0.153 [-0.525, +0.225] | +0.543 | +0.204 |
| tnks2 | 27 | 1 | +0.628 | 16/21 | 9/26 | +0.299 | +0.074 | -0.433 | -0.506 [-1.330, -0.181] | +0.120 | -0.246 |
| galectin3_extra | 26 | 2 | +0.515 | 15/20 | 5/25 | +0.429 | +0.179 | +0.401 | +0.222 [-0.198, +0.592] | +0.624 | +0.250 |
| shp2 | 26 | 0.5 | +0.493 | 13/20 | 6/25 | -0.004 | +0.030 | -0.215 | -0.245 [-0.548, +0.072] | -0.301 | -0.273 |
| mcl1 | 25 | 0.5 | +0.636 | 16/19 | 8/24 | +0.206 | -0.496 | -1.200 | -0.703 [-2.307, +1.374] | +0.426 | -0.583 |
| cmet | 24 | 2 | +0.691 | 16/19 | 9/23 | +0.547 | +0.702 | -0.247 | -0.949 [-1.544, -0.548] | +0.677 | -0.322 |
| hfaah | 24 | 0.3 | +0.302 | 5/19 | 1/23 | +0.242 | -0.065 | -0.046 | +0.019 [-0.559, +0.322] | +0.649 | +0.606 |
| ptp1b | 23 | 1 | +0.498 | 12/18 | 3/22 | -0.016 | -0.137 | -0.034 | +0.103 [-0.759, +1.578] | -0.211 | -0.087 |
| thrombin | 22 | 2 | +0.389 | 7/17 | 3/21 | +0.119 | +0.351 | -0.394 | -0.745 [-1.296, -0.477] | -0.215 | -3.426 |
| bace_p3_arg368_in | 21 | 2 | +0.874 | 16/16 | 12/20 | +0.359 | +0.403 | -0.040 | -0.443 [-1.114, +0.265] | +0.164 | +0.047 |
| jnk1 | 21 | 0.5 | +0.405 | 8/16 | 3/20 | +0.452 | +0.203 | -0.105 | -0.308 [-0.741, +0.433] | +0.563 | +0.107 |
| pde2 | 21 | 2 | +0.406 | 7/16 | 4/20 | +0.355 | +0.382 | -0.176 | -0.558 [-0.982, -0.069] | +0.347 | -0.169 |
| hne | 20 | 0.5 | +0.518 | 9/15 | 2/19 | +0.635 | -0.039 | +0.191 | +0.230 [-0.667, +0.560] | +0.226 | +0.210 |
| p2y1_meta_sub | 20 | 0.5 | +0.494 | 8/15 | 3/19 | -0.255 | -0.478 | -0.223 | +0.255 [-0.439, +1.560] | -0.197 | -0.152 |
| **mean** |  |  |  |  |  | **+0.272** | **+0.181** | **-0.058** | **-0.239** | **+0.364** | **+0.090** |

Over 49 series: 78 % of the canonical pairs hold out above the shuffled pairing, and the empirical view contains 23 % of the quantum view's directions on average. The quantum view's own part predicts the activity at q² **-0.058** on average against **+0.181** for its contained part and **+0.272** for the whole quantum view; the difference (own minus contained) is **-0.239**, with its interval above zero on 3 and below zero on 20 of the 49 series. The empirical view alone reaches **+0.364**, and its own part relative to the quantum view **+0.090**.

## Verdict

So what the reactivity fields know about the activity of these series is the part of them that shape and charge already contain: the directions that are theirs alone carry little. That is why adding them to the five fields gains little in the earlier measurements, and it is a statement about the condensed, atom-centred form of the fields on these series, not about conceptual DFT.
