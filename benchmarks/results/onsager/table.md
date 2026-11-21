# Item BT10, a pilot: the transference number from the stored boxes

The cation transference number is read from the Onsager coefficients in the frame of the solvent, from the stored centres of the boxes of item BT9 (1,000 solvent molecules, charges of the ions multiplied by 0.7, 4 ns unless the name says long). The definition, the window and the reading are in the docstring of `benchmarks/run_onsager.py`. The error is the standard error over 5 blocks of time origins. The self ratio is D+ / (D+ + D-).

## The six concentrations of the series

| Run | Molarity, mol/L | Transference number from the Onsager coefficients | Self ratio | Measured | Exponent of the charge displacement |
|---|---:|---:|---:|---:|---:|
| conc_c0p1_seed2 | 0.10 | 0.368 +/- 0.140 | 0.388 | 0.373 | 0.94 |
| conc_c0p1 | 0.10 | 0.350 +/- 0.112 | 0.439 | 0.373 | 0.92 |
| conc_c0p5_seed2 | 0.49 | 0.256 +/- 0.157 | 0.445 | 0.307 | 0.96 |
| conc_c0p5 | 0.49 | 0.124 +/- 0.091 | 0.408 | 0.307 | 1.04 |
| conc_c1p1_seed2 | 1.13 | 0.469 +/- 0.132 | 0.395 | 0.201 | 0.84 |
| conc_c1p1 | 1.13 | 0.179 +/- 0.074 | 0.407 | 0.201 | 0.90 |
| conc_c1p1_long | 1.13 | 0.372 +/- 0.032 | 0.398 | 0.201 | 0.96 |
| conc_c1p5_seed2 | 1.48 | 0.256 +/- 0.169 | 0.413 | 0.152 | 0.78 |
| conc_c1p5 | 1.49 | 0.321 +/- 0.238 | 0.425 | 0.152 | 0.59 |
| conc_c2p0 | 1.98 | 0.287 +/- 0.099 | 0.437 | 0.107 | 0.98 |
| conc_c2p0_seed2 | 1.98 | 0.185 +/- 0.096 | 0.434 | 0.107 | 0.87 |
| conc_c3p0_long | 2.97 | 0.309 +/- 0.079 | 0.441 | 0.143 | 0.77 |
| conc_c3p0_seed2 | 2.97 | 0.372 +/- 0.213 | 0.462 | 0.143 | 0.51 |
| conc_c3p0 | 2.97 | 0.049 +/- 0.241 | 0.464 | 0.143 | 0.73 |

* **Scatter of one start.** The two starts of 6 concentrations differ by an amount that corresponds to 0.136 in t+ for one start. The pilot asked for at most 0.05. This is above that.
* **Collective against self.** The number from the Onsager coefficients is on average -0.159 from the self ratio (standard deviation 0.123, 12 boxes).
* **Against the measured number.** The mean difference from the measured number is +0.054 for the Onsager number and +0.213 for the self ratio.
* **Check of the frame.** The conductivity from the charge displacement of this analysis is 1.000000 to 1.000000 times that of item BT9, which uses the frame of the whole box. The number of cations equals that of anions, so the two have to agree.

## The 62 formulations

The transference number from the Onsager coefficients runs from 0.04 to 0.69 over 63 boxes (a mean block error of 0.16); the self ratio runs from 0.36 to 0.48. Their rank correlation is -0.18.

| Formulation | Run | Onsager | Block error | Self ratio |
|---|---|---:|---:|---:|
| LiPF6_DEC20_PC80_m0p282 | form_DEC20_PC80_m0p282 | 0.287 | 0.083 | 0.388 |
| LiPF6_DEC20_PC80_m0p892 | form_DEC20_PC80 | 0.270 | 0.046 | 0.398 |
| LiPF6_DEC20_PC80_m2p373 | form_DEC20_PC80_m2p373 | 0.143 | 0.291 | 0.479 |
| LiPF6_DEC40_PC60_m0p320 | form_DEC40_PC60_m0p320 | 0.277 | 0.133 | 0.361 |
| LiPF6_DEC40_PC60_m0p802 | form_DEC40_PC60 | 0.451 | 0.152 | 0.380 |
| LiPF6_DEC40_PC60_m1p741 | form_DEC40_PC60_m1p741 | 0.050 | 0.082 | 0.435 |
| LiPF6_DEC50_PC50_m0p271 | form_DEC50_PC50_m0p271 | 0.338 | 0.046 | 0.388 |
| LiPF6_DEC50_PC50_m0p386 | form_DEC50_PC50_m0p386 | 0.309 | 0.266 | 0.376 |
| LiPF6_DEC50_PC50_m0p465 | form_DEC50_PC50 | 0.167 | 0.126 | 0.376 |
| LiPF6_DEC60_PC40_m0p245 | form_DEC60_PC40_m0p245 | 0.241 | 0.097 | 0.368 |
| LiPF6_DEC60_PC40_m0p539 | form_DEC60_PC40_m0p539 | 0.251 | 0.095 | 0.409 |
| LiPF6_DEC60_PC40_m0p750 | form_DEC60_PC40 | 0.186 | 0.155 | 0.406 |
| LiPF6_DEC70_PC30_m0p259 | form_DEC70_PC30_m0p259 | 0.079 | 0.636 | 0.364 |
| LiPF6_DEC70_PC30_m0p550 | form_DEC70_PC30_m0p550 | 0.361 | 0.235 | 0.388 |
| LiPF6_DEC70_PC30_m0p898 | form_DEC70_PC30 | 0.411 | 0.047 | 0.396 |
| LiPF6_DMC100_m0p500 | form_DMC100_m0p500 | 0.240 | 0.455 | 0.449 |
| LiPF6_DMC100_m1p500 | form_DMC100 | 0.130 | 0.189 | 0.422 |
| LiPF6_DMC100_m2p000 | form_DMC100_m2p000 | 0.059 | 0.225 | 0.419 |
| LiPF6_DMC50_EC30_MA20_m0p491 | form_DMC50_EC30_MA20_m0p491 | 0.307 | 0.112 | 0.442 |
| LiPF6_DMC50_EC30_MA20_m1p500 | form_DMC50_EC30_MA20 | 0.181 | 0.149 | 0.426 |
| LiPF6_DMC50_EC30_MA20_m2p000 | form_DMC50_EC30_MA20_m2p000 | 0.183 | 0.112 | 0.437 |
| LiPF6_DMC60_EC30_MA10_m0p500 | form_DMC60_EC30_MA10_m0p500 | 0.686 | 0.082 | 0.426 |
| LiPF6_DMC60_EC30_MA10_m1p491 | form_DMC60_EC30_MA10 | 0.267 | 0.120 | 0.418 |
| LiPF6_DMC60_EC30_MA10_m1p982 | form_DMC60_EC30_MA10_m1p982 | 0.279 | 0.063 | 0.424 |
| LiPF6_DMC70_EC25_EMC05_m1p200 | form_DMC70_EC25_EMC05 | 0.391 | 0.091 | 0.404 |
| LiPF6_DMC70_EC30_m0p491 | form_DMC70_EC30_m0p491 | 0.432 | 0.723 | 0.410 |
| LiPF6_DMC70_EC30_m1p000 | form_DMC70_EC30 | 0.340 | 0.105 | 0.407 |
| LiPF6_DMC70_EC30_m2p000 | form_DMC70_EC30_m2p000 | 0.304 | 0.042 | 0.428 |
| LiPF6_DMC80_EC20_m0p500 | form_DMC80_EC20_m0p500 | 0.138 | 0.120 | 0.415 |
| LiPF6_DMC80_EC20_m1p500 | form_DMC80_EC20 | 0.543 | 0.197 | 0.428 |
| LiPF6_DMC80_EC20_m2p000 | form_DMC80_EC20_m2p000 | 0.295 | 0.400 | 0.435 |
| LiPF6_DMC90_EC10_m0p500 | form_DMC90_EC10_m0p500 | 0.115 | 0.101 | 0.433 |
| LiPF6_DMC90_EC10_m1p500 | form_DMC90_EC10 | 0.371 | 0.122 | 0.425 |
| LiPF6_DMC90_EC10_m2p000 | form_DMC90_EC10_m2p000 | 0.176 | 0.039 | 0.410 |
| LiPF6_EC10_EMC90_m0p500 | form_EC10_EMC90_m0p500 | 0.273 | 0.471 | 0.438 |
| LiPF6_EC10_EMC90_m1p500 | form_EC10_EMC90 | 0.250 | 0.077 | 0.428 |
| LiPF6_EC10_EMC90_m2p000 | form_EC10_EMC90_m2p000 | 0.303 | 0.068 | 0.439 |
| LiPF6_EC15_PC85_m0p446 | form_EC15_PC85 | 0.302 | 0.198 | 0.392 |
| LiPF6_EC20_EMC80_m0p454 | form_EC20_EMC80_m0p454 | 0.202 | 0.045 | 0.411 |
| LiPF6_EC20_EMC80_m1p110 | form_EC20_EMC80 | 0.245 | 0.108 | 0.423 |
| LiPF6_EC20_EMC80_m2p000 | form_EC20_EMC80_m2p000 | 0.041 | 0.092 | 0.436 |
| LiPF6_EC30_EMC50_MA20_m0p508 | form_EC30_EMC50_MA20_m0p508 | 0.390 | 0.103 | 0.435 |
| LiPF6_EC30_EMC50_MA20_m1p492 | form_EC30_EMC50_MA20 | 0.264 | 0.087 | 0.422 |
| LiPF6_EC30_EMC50_MA20_m1p992 | form_EC30_EMC50_MA20_m1p992 | 0.137 | 0.145 | 0.429 |
| LiPF6_EC30_EMC60_MA10_m0p570 | form_EC30_EMC60_MA10_m0p570 | 0.194 | 0.200 | 0.395 |
| LiPF6_EC30_EMC60_MA10_m1p596 | form_EC30_EMC60_MA10 | 0.201 | 0.138 | 0.414 |
| LiPF6_EC30_EMC60_MA10_m2p105 | form_EC30_EMC60_MA10_m2p105 | 0.134 | 0.054 | 0.429 |
| LiPF6_EC30_EMC70_m0p443 | form_EC30_EMC70_m0p443 | 0.258 | 0.130 | 0.400 |
| LiPF6_EC30_EMC70_m1p110 | conc_c1p1 | 0.179 | 0.074 | 0.407 |
| LiPF6_EC30_EMC70_m1p110 | conc_c1p1_seed2 | 0.469 | 0.132 | 0.395 |
| LiPF6_EC30_EMC70_m2p009 | form_EC30_EMC70_m2p009 | 0.409 | 0.136 | 0.421 |
| LiPF6_EC40_EMC60_m0p428 | form_EC40_EMC60_m0p428 | 0.508 | 0.078 | 0.375 |
| LiPF6_EC40_EMC60_m1p077 | form_EC40_EMC60 | 0.245 | 0.092 | 0.407 |
| LiPF6_EC40_EMC60_m1p574 | form_EC40_EMC60_m1p574 | 0.464 | 0.219 | 0.413 |
| LiPF6_EC50_EMC50_m0p540 | form_EC50_EMC50_m0p540 | 0.299 | 0.152 | 0.396 |
| LiPF6_EC50_EMC50_m1p212 | form_EC50_EMC50 | 0.405 | 0.323 | 0.416 |
| LiPF6_EC50_EMC50_m1p806 | form_EC50_EMC50_m1p806 | 0.352 | 0.093 | 0.428 |
| LiPF6_EC60_PC40_m0p450 | form_EC60_PC40_m0p450 | 0.463 | 0.146 | 0.407 |
| LiPF6_EC60_PC40_m0p813 | form_EC60_PC40 | 0.217 | 0.060 | 0.384 |
| LiPF6_EC60_PC40_m2p102 | form_EC60_PC40_m2p102 | 0.205 | 0.133 | 0.477 |
| LiPF6_EMC100_m0p500 | form_EMC100_m0p500 | 0.257 | 0.273 | 0.457 |
| LiPF6_EMC100_m1p500 | form_EMC100 | 0.433 | 0.222 | 0.431 |
| LiPF6_EMC100_m2p000 | form_EMC100_m2p000 | 0.322 | 0.153 | 0.421 |

