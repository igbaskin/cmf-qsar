# Item BT11, a pilot: the thermodynamic factor from the stored boxes

The factor is read from the fluctuations of the numbers of molecules in cubes, in the boxes of item BT9 (1,000 solvent molecules, 4 ns, charges of the ions multiplied by 0.7). The definition and what the pilot leaves out are in the docstring of `benchmarks/run_kbpilot.py`. The cubes are 1/4, 1/3 and 1/2 of the side of the box; they are not corrected for the closed box. The error after +/- is the standard error over 8 blocks of frames.

## The six concentrations of the series

| Run | Molarity, mol/L | Measured factor | Cube 1/4 | Cube 1/3 | Cube 1/2 |
|---|---:|---:|---:|---:|---:|
| conc_c0p1_seed2 | 0.10 | 0.54 | 0.92 +/- 0.04 | 0.86 +/- 0.05 | 0.82 +/- 0.08 |
| conc_c0p1 | 0.10 | 0.54 | 0.97 +/- 0.03 | 0.98 +/- 0.04 | 1.11 +/- 0.08 |
| conc_c0p5_seed2 | 0.49 | 1.35 | 1.08 +/- 0.02 | 1.15 +/- 0.03 | 1.31 +/- 0.08 |
| conc_c0p5 | 0.49 | 1.35 | 0.99 +/- 0.02 | 0.93 +/- 0.03 | 0.88 +/- 0.05 |
| conc_c1p1_seed2 | 1.13 | 2.37 | 1.38 +/- 0.03 | 1.49 +/- 0.06 | 1.62 +/- 0.11 |
| conc_c1p1 | 1.13 | 2.37 | 1.41 +/- 0.05 | 1.53 +/- 0.07 | 1.80 +/- 0.11 |
| conc_c1p5_seed2 | 1.48 | 2.92 | 1.59 +/- 0.06 | 1.84 +/- 0.08 | 2.29 +/- 0.13 |
| conc_c1p5 | 1.49 | 2.92 | 1.44 +/- 0.08 | 1.48 +/- 0.11 | 1.42 +/- 0.13 |
| conc_c2p0 | 1.98 | 3.81 | 1.55 +/- 0.04 | 1.52 +/- 0.04 | 1.45 +/- 0.06 |
| conc_c2p0_seed2 | 1.98 | 3.81 | 1.51 +/- 0.10 | 1.48 +/- 0.14 | 1.42 +/- 0.17 |
| conc_c3p0_seed2 | 2.97 | 6.73 | 1.44 +/- 0.05 | 1.52 +/- 0.06 | 1.67 +/- 0.07 |
| conc_c3p0 | 2.97 | 6.74 | 1.38 +/- 0.07 | 1.38 +/- 0.08 | 1.45 +/- 0.09 |

* Cube 0.25 of the side: the rank correlation with the measured factor is +0.75, and the rank correlation with the molarity +0.75.
* Cube 0.33 of the side: the rank correlation with the measured factor is +0.54, and the rank correlation with the molarity +0.54.
* Cube 0.50 of the side: the rank correlation with the measured factor is +0.59, and the rank correlation with the molarity +0.59.

## The factor on the charges of the ions

The older boxes of item BT9 at the reference electrolyte (1.11 mol/kg). The measured factor at this concentration is 2.37. The boxes of the series at 0.7 are in the first table.

| Run | Factor on the charges | Solvent molecules | Cube 1/4 | Cube 1/3 | Cube 1/2 |
|---|---:|---:|---:|---:|---:|
| full | 1 | 250 | 0.92 +/- 0.01 | 1.02 +/- 0.02 | 1.26 +/- 0.07 |
| scaled_small | 0.8 | 125 | 0.91 +/- 0.01 | 0.99 +/- 0.03 | 1.11 +/- 0.03 |
| scaled | 0.8 | 250 | 1.00 +/- 0.02 | 1.11 +/- 0.03 | 1.33 +/- 0.04 |
| scaled_seed2 | 0.8 | 250 | 0.98 +/- 0.01 | 1.07 +/- 0.03 | 1.30 +/- 0.06 |
| scaled_seed3 | 0.8 | 250 | 0.92 +/- 0.02 | 1.02 +/- 0.04 | 1.19 +/- 0.05 |
| scaled_large | 0.8 | 500 | 1.04 +/- 0.02 | 1.10 +/- 0.03 | 1.23 +/- 0.04 |
| scaled_huge | 0.8 | 1000 | 1.07 +/- 0.03 | 1.12 +/- 0.04 | 1.25 +/- 0.06 |
| scaled_07 | 0.7 | 250 | 1.21 +/- 0.05 | 1.33 +/- 0.08 | 1.71 +/- 0.12 |

## The 62 formulations

The cube of 1/2 of the side. There is no measured factor for these electrolytes.

| Formulation | Run | Factor, cube 1/2 | Standard error |
|---|---|---:|---:|
| LiPF6_DEC20_PC80_m0p282 | form_DEC20_PC80_m0p282 | 1.55 | 0.14 |
| LiPF6_DEC20_PC80_m0p892 | form_DEC20_PC80 | 2.09 | 0.09 |
| LiPF6_DEC20_PC80_m2p373 | form_DEC20_PC80_m2p373 | 1.49 | 0.04 |
| LiPF6_DEC40_PC60_m0p320 | form_DEC40_PC60_m0p320 | 1.65 | 0.09 |
| LiPF6_DEC40_PC60_m0p802 | form_DEC40_PC60 | 2.18 | 0.07 |
| LiPF6_DEC40_PC60_m1p741 | form_DEC40_PC60_m1p741 | 1.49 | 0.04 |
| LiPF6_DEC50_PC50_m0p271 | form_DEC50_PC50_m0p271 | 1.41 | 0.04 |
| LiPF6_DEC50_PC50_m0p386 | form_DEC50_PC50_m0p386 | 1.40 | 0.04 |
| LiPF6_DEC50_PC50_m0p465 | form_DEC50_PC50 | 1.55 | 0.06 |
| LiPF6_DEC60_PC40_m0p245 | form_DEC60_PC40_m0p245 | 1.14 | 0.06 |
| LiPF6_DEC60_PC40_m0p539 | form_DEC60_PC40_m0p539 | 1.47 | 0.04 |
| LiPF6_DEC60_PC40_m0p750 | form_DEC60_PC40 | 1.79 | 0.05 |
| LiPF6_DEC70_PC30_m0p259 | form_DEC70_PC30_m0p259 | 1.20 | 0.06 |
| LiPF6_DEC70_PC30_m0p550 | form_DEC70_PC30_m0p550 | 1.61 | 0.11 |
| LiPF6_DEC70_PC30_m0p898 | form_DEC70_PC30 | 1.54 | 0.07 |
| LiPF6_DMC100_m0p500 | form_DMC100_m0p500 | 0.66 | 0.04 |
| LiPF6_DMC100_m1p500 | form_DMC100 | 1.22 | 0.14 |
| LiPF6_DMC100_m2p000 | form_DMC100_m2p000 | 1.62 | 0.11 |
| LiPF6_DMC50_EC30_MA20_m0p491 | form_DMC50_EC30_MA20_m0p491 | 1.07 | 0.08 |
| LiPF6_DMC50_EC30_MA20_m1p500 | form_DMC50_EC30_MA20 | 1.79 | 0.14 |
| LiPF6_DMC50_EC30_MA20_m2p000 | form_DMC50_EC30_MA20_m2p000 | 2.26 | 0.17 |
| LiPF6_DMC60_EC30_MA10_m0p500 | form_DMC60_EC30_MA10_m0p500 | 1.44 | 0.11 |
| LiPF6_DMC60_EC30_MA10_m1p491 | form_DMC60_EC30_MA10 | 2.02 | 0.10 |
| LiPF6_DMC60_EC30_MA10_m1p982 | form_DMC60_EC30_MA10_m1p982 | 2.52 | 0.27 |
| LiPF6_DMC70_EC25_EMC05_m1p200 | form_DMC70_EC25_EMC05 | 1.87 | 0.13 |
| LiPF6_DMC70_EC30_m0p491 | form_DMC70_EC30_m0p491 | 1.43 | 0.08 |
| LiPF6_DMC70_EC30_m1p000 | form_DMC70_EC30 | 1.76 | 0.04 |
| LiPF6_DMC70_EC30_m2p000 | form_DMC70_EC30_m2p000 | 2.11 | 0.18 |
| LiPF6_DMC80_EC20_m0p500 | form_DMC80_EC20_m0p500 | 1.17 | 0.08 |
| LiPF6_DMC80_EC20_m1p500 | form_DMC80_EC20 | 1.64 | 0.12 |
| LiPF6_DMC80_EC20_m2p000 | form_DMC80_EC20_m2p000 | 1.95 | 0.10 |
| LiPF6_DMC90_EC10_m0p500 | form_DMC90_EC10_m0p500 | 0.98 | 0.06 |
| LiPF6_DMC90_EC10_m1p500 | form_DMC90_EC10 | 1.32 | 0.06 |
| LiPF6_DMC90_EC10_m2p000 | form_DMC90_EC10_m2p000 | 1.55 | 0.03 |
| LiPF6_EC10_EMC90_m0p500 | form_EC10_EMC90_m0p500 | 0.93 | 0.07 |
| LiPF6_EC10_EMC90_m1p500 | form_EC10_EMC90 | 1.25 | 0.10 |
| LiPF6_EC10_EMC90_m2p000 | form_EC10_EMC90_m2p000 | 1.58 | 0.09 |
| LiPF6_EC15_PC85_m0p446 | form_EC15_PC85 | 1.21 | 0.08 |
| LiPF6_EC20_EMC80_m0p454 | form_EC20_EMC80_m0p454 | 0.93 | 0.06 |
| LiPF6_EC20_EMC80_m1p110 | form_EC20_EMC80 | 1.56 | 0.08 |
| LiPF6_EC20_EMC80_m2p000 | form_EC20_EMC80_m2p000 | 1.81 | 0.17 |
| LiPF6_EC30_EMC50_MA20_m0p508 | form_EC30_EMC50_MA20_m0p508 | 1.36 | 0.07 |
| LiPF6_EC30_EMC50_MA20_m1p492 | form_EC30_EMC50_MA20 | 1.64 | 0.23 |
| LiPF6_EC30_EMC50_MA20_m1p992 | form_EC30_EMC50_MA20_m1p992 | 1.82 | 0.10 |
| LiPF6_EC30_EMC60_MA10_m0p570 | form_EC30_EMC60_MA10_m0p570 | 1.21 | 0.05 |
| LiPF6_EC30_EMC60_MA10_m1p596 | form_EC30_EMC60_MA10 | 1.38 | 0.23 |
| LiPF6_EC30_EMC60_MA10_m2p105 | form_EC30_EMC60_MA10_m2p105 | 2.76 | 0.14 |
| LiPF6_EC30_EMC70_m0p443 | form_EC30_EMC70_m0p443 | 1.19 | 0.06 |
| LiPF6_EC30_EMC70_m1p110 | conc_c1p1 | 1.80 | 0.11 |
| LiPF6_EC30_EMC70_m1p110 | conc_c1p1_seed2 | 1.62 | 0.11 |
| LiPF6_EC30_EMC70_m2p009 | form_EC30_EMC70_m2p009 | 1.33 | 0.07 |
| LiPF6_EC40_EMC60_m0p428 | form_EC40_EMC60_m0p428 | 1.24 | 0.13 |
| LiPF6_EC40_EMC60_m1p077 | form_EC40_EMC60 | 1.62 | 0.08 |
| LiPF6_EC40_EMC60_m1p574 | form_EC40_EMC60_m1p574 | 1.08 | 0.08 |
| LiPF6_EC50_EMC50_m0p540 | form_EC50_EMC50_m0p540 | 1.41 | 0.04 |
| LiPF6_EC50_EMC50_m1p212 | form_EC50_EMC50 | 1.50 | 0.07 |
| LiPF6_EC50_EMC50_m1p806 | form_EC50_EMC50_m1p806 | 1.35 | 0.04 |
| LiPF6_EC60_PC40_m0p450 | form_EC60_PC40_m0p450 | 1.31 | 0.09 |
| LiPF6_EC60_PC40_m0p813 | form_EC60_PC40 | 1.85 | 0.11 |
| LiPF6_EC60_PC40_m2p102 | form_EC60_PC40_m2p102 | 1.85 | 0.19 |
| LiPF6_EMC100_m0p500 | form_EMC100_m0p500 | 0.69 | 0.02 |
| LiPF6_EMC100_m1p500 | form_EMC100 | 1.03 | 0.08 |
| LiPF6_EMC100_m2p000 | form_EMC100_m2p000 | 1.45 | 0.12 |

