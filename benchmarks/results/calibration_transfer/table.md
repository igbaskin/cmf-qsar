# A model fitted on one instrument reads the spectra of another

Data: two near-infrared sets from https://eigenvector.com/resources/data-sets/ (no licence stated; no file of the sets is in this repository).
- corn: 80 samples on 3 instruments (m5, mp5, mp6), 700 points from 1100 to 2498 nm; properties: moisture, oil, protein, starch; 60 samples to fit and 20 to test, 20 random splits.
- tablets: 655 samples on 2 instruments (1, 2), 501 points from 600 to 1600 nm; properties: assay; 155 samples to fit and 500 to test, 20 random splits.

An error in the tables below is a root-mean-square error divided by the standard deviation of the property over all samples, and then the mean over the properties of the set. A model that predicts the mean has about 1.

## Part 1: what differs between the instruments

Every number but the last two is a root mean square of absorbance over the samples and the axis.

| Pair of instruments | Spread of the samples on the first | Difference between the instruments | After the mean difference is taken off | After an offset, a gain and a sloping baseline per sample | After the fitted shift too | Fitted shift, nm | Fitted difference in resolution, nm |
|---|---|---|---|---|---|---|---|
| corn, m5 to mp5 | 0.0357 | 0.0450 | 0.0077 | 0.0010 | 0.0010 | +0.50 | 0 |
| corn, m5 to mp6 | 0.0357 | 0.0565 | 0.0076 | 0.0011 | 0.0011 | +0.80 | 0 |
| corn, mp5 to mp6 | 0.0337 | 0.0122 | 0.0026 | 0.0003 | 0.0003 | +0.40 | 0 |
| tablets, 1 to 2 | 0.1852 | 0.0618 | 0.0266 | 0.0085 | 0.0085 | +0.10 | 0 |

## Part 2: within one instrument

| Instrument | PLS, plain spectra | PLS, slopes | Kernel ridge, plain spectra | Kernel ridge, slopes | Kernel ridge, slopes, the test spectra centred on the mean of the fit |
|---|---|---|---|---|---|
| corn, m5 | 0.120 | 0.105 | 0.115 | 0.100 | 0.221 |
| corn, mp5 | 0.440 | 0.426 | 0.425 | 0.425 | 0.471 |
| corn, mp6 | 0.454 | 0.415 | 0.431 | 0.398 | 0.445 |
| tablets, 1 | 0.245 | 0.241 | 0.239 | 0.236 | 0.261 |
| tablets, 2 | 0.246 | 0.245 | 0.242 | 0.240 | 0.265 |

The last column is the cost of centring when no instrument is changed: the samples of the test do not have exactly the mean of the samples of the fit.

The same, property by property, as root-mean-square errors in the units of the property. An interval is a 95 % bootstrap interval over the splits.

| Instrument and property | PLS, plain spectra | PLS, slopes | Kernel ridge, plain spectra | Kernel ridge, slopes | Kernel ridge minus PLS, both on the slopes | Kernel ridge on the slopes minus PLS on the plain spectra |
|---|---|---|---|---|---|---|
| corn, m5, moisture | 0.0071 | 0.0115 | 0.0070 | 0.0118 | +0.0003 [-0.0002, +0.0007] | +0.0046 [+0.0038, +0.0053] |
| corn, m5, oil | 0.0277 | 0.0235 | 0.0263 | 0.0218 | -0.0017 [-0.0032, -0.0005] | -0.0059 [-0.0085, -0.0034] |
| corn, m5, protein | 0.0780 | 0.0546 | 0.0751 | 0.0524 | -0.0022 [-0.0049, +0.0004] | -0.0256 [-0.0363, -0.0148] |
| corn, m5, starch | 0.1206 | 0.1183 | 0.1147 | 0.1141 | -0.0042 [-0.0083, +0.0001] | -0.0065 [-0.0168, +0.0040] |
| corn, mp5, moisture | 0.1455 | 0.1440 | 0.1415 | 0.1422 | -0.0018 [-0.0064, +0.0029] | -0.0033 [-0.0108, +0.0065] |
| corn, mp5, oil | 0.1053 | 0.1055 | 0.1001 | 0.1020 | -0.0035 [-0.0063, -0.0006] | -0.0034 [-0.0076, +0.0017] |
| corn, mp5, protein | 0.1585 | 0.1448 | 0.1480 | 0.1430 | -0.0018 [-0.0067, +0.0032] | -0.0155 [-0.0243, -0.0065] |
| corn, mp5, starch | 0.3738 | 0.3516 | 0.3730 | 0.3723 | +0.0207 [+0.0068, +0.0347] | -0.0015 [-0.0169, +0.0125] |
| corn, mp6, moisture | 0.1679 | 0.1304 | 0.1574 | 0.1296 | -0.0008 [-0.0060, +0.0053] | -0.0383 [-0.0490, -0.0278] |
| corn, mp6, oil | 0.1063 | 0.0990 | 0.0969 | 0.0920 | -0.0070 [-0.0095, -0.0044] | -0.0143 [-0.0193, -0.0097] |
| corn, mp6, protein | 0.1521 | 0.1478 | 0.1521 | 0.1437 | -0.0041 [-0.0083, +0.0005] | -0.0084 [-0.0150, -0.0026] |
| corn, mp6, starch | 0.3762 | 0.3692 | 0.3663 | 0.3574 | -0.0118 [-0.0231, -0.0003] | -0.0188 [-0.0566, +0.0106] |
| tablets, 1, assay | 4.1909 | 4.1305 | 4.0898 | 4.0360 | -0.0945 [-0.1245, -0.0676] | -0.1549 [-0.2179, -0.0958] |
| tablets, 2, assay | 4.2197 | 4.2017 | 4.1482 | 4.1027 | -0.0990 [-0.1568, -0.0503] | -0.1170 [-0.2008, -0.0297] |

## Part 2: the crossing, kernel ridge on the slopes

| Source to target | Within the source | Target as measured | Each instrument centred | Shift fitted | Shift fitted, each instrument centred | Mean difference of 8 standards | PDS with 8 standards | PDS with 16 standards | Model fitted on the target |
|---|---|---|---|---|---|---|---|---|---|
| corn, m5 to mp5 | 0.100 | 2.747 | 0.510 | 2.171 | 0.503 | 0.499 | 0.540 | 0.507 | 0.425 |
| corn, m5 to mp6 | 0.100 | 2.682 | 0.523 | 1.720 | 0.528 | 0.522 | 0.560 | 0.526 | 0.398 |
| corn, mp5 to m5 | 0.425 | 2.412 | 0.513 | 1.644 | 0.513 | 0.489 | 0.468 | 0.457 | 0.100 |
| corn, mp5 to mp6 | 0.425 | 0.988 | 0.492 | 0.600 | 0.493 | 0.453 | 0.446 | 0.446 | 0.398 |
| corn, mp6 to m5 | 0.398 | 2.636 | 0.518 | 1.355 | 0.522 | 0.502 | 0.456 | 0.441 | 0.100 |
| corn, mp6 to mp5 | 0.398 | 0.870 | 0.497 | 1.025 | 0.505 | 0.469 | 0.464 | 0.454 | 0.425 |
| tablets, 1 to 2 | 0.236 | 1.382 | 0.373 | 1.405 | 0.375 | 0.407 | 0.341 | 0.320 | 0.240 |
| tablets, 2 to 1 | 0.240 | 2.369 | 0.328 | 2.252 | 0.328 | 0.336 | 0.310 | 0.286 | 0.236 |

## Part 2: the crossing, PLS on the plain spectra

| Source to target | Within the source | Target as measured | Each instrument centred | Shift fitted | Shift fitted, each instrument centred | Mean difference of 8 standards | PDS with 8 standards | PDS with 16 standards | Model fitted on the target |
|---|---|---|---|---|---|---|---|---|---|
| corn, m5 to mp5 | 0.120 | 2.791 | 0.513 | 2.240 | 0.507 | 0.500 | 0.541 | 0.512 | 0.440 |
| corn, m5 to mp6 | 0.120 | 2.846 | 0.543 | 1.915 | 0.539 | 0.542 | 0.570 | 0.537 | 0.454 |
| corn, mp5 to m5 | 0.440 | 3.521 | 0.585 | 3.287 | 0.590 | 0.577 | 0.544 | 0.537 | 0.120 |
| corn, mp5 to mp6 | 0.440 | 1.327 | 0.526 | 0.775 | 0.524 | 0.500 | 0.493 | 0.490 | 0.454 |
| corn, mp6 to m5 | 0.454 | 3.219 | 0.613 | 2.716 | 0.611 | 0.606 | 0.572 | 0.552 | 0.120 |
| corn, mp6 to mp5 | 0.454 | 1.095 | 0.521 | 1.109 | 0.526 | 0.487 | 0.518 | 0.498 | 0.440 |
| tablets, 1 to 2 | 0.245 | 1.754 | 0.472 | 1.751 | 0.476 | 0.537 | 0.407 | 0.366 | 0.246 |
| tablets, 2 to 1 | 0.246 | 3.132 | 0.401 | 3.015 | 0.403 | 0.425 | 0.371 | 0.345 | 0.245 |

## Part 2: the crossing, PLS on the slopes

| Source to target | Within the source | Target as measured | Each instrument centred | Shift fitted | Shift fitted, each instrument centred | Mean difference of 8 standards | PDS with 8 standards | PDS with 16 standards | Model fitted on the target |
|---|---|---|---|---|---|---|---|---|---|
| corn, m5 to mp5 | 0.105 | 2.774 | 0.515 | 2.176 | 0.506 | 0.503 | 0.537 | 0.508 | 0.426 |
| corn, m5 to mp6 | 0.105 | 2.704 | 0.527 | 1.714 | 0.530 | 0.524 | 0.557 | 0.522 | 0.415 |
| corn, mp5 to m5 | 0.426 | 2.416 | 0.528 | 1.617 | 0.528 | 0.503 | 0.475 | 0.462 | 0.105 |
| corn, mp5 to mp6 | 0.426 | 0.954 | 0.494 | 0.579 | 0.496 | 0.456 | 0.447 | 0.447 | 0.415 |
| corn, mp6 to m5 | 0.415 | 2.628 | 0.546 | 1.497 | 0.549 | 0.536 | 0.476 | 0.458 | 0.105 |
| corn, mp6 to mp5 | 0.415 | 0.937 | 0.521 | 1.029 | 0.527 | 0.499 | 0.493 | 0.481 | 0.426 |
| tablets, 1 to 2 | 0.241 | 1.748 | 0.392 | 1.753 | 0.395 | 0.438 | 0.371 | 0.327 | 0.245 |
| tablets, 2 to 1 | 0.245 | 2.132 | 0.336 | 2.039 | 0.337 | 0.352 | 0.325 | 0.305 | 0.241 |

## Part 2: the crossing, kernel ridge on the plain spectra

| Source to target | Within the source | Target as measured | Each instrument centred | Shift fitted | Shift fitted, each instrument centred | Mean difference of 8 standards | PDS with 8 standards | PDS with 16 standards | Model fitted on the target |
|---|---|---|---|---|---|---|---|---|---|
| corn, m5 to mp5 | 0.115 | 2.767 | 0.508 | 2.139 | 0.503 | 0.496 | 0.537 | 0.507 | 0.425 |
| corn, m5 to mp6 | 0.115 | 2.814 | 0.537 | 1.788 | 0.535 | 0.536 | 0.570 | 0.535 | 0.431 |
| corn, mp5 to m5 | 0.425 | 3.266 | 0.536 | 2.869 | 0.540 | 0.518 | 0.495 | 0.487 | 0.115 |
| corn, mp5 to mp6 | 0.425 | 1.191 | 0.507 | 0.709 | 0.506 | 0.476 | 0.473 | 0.469 | 0.431 |
| corn, mp6 to m5 | 0.431 | 3.239 | 0.566 | 2.554 | 0.562 | 0.552 | 0.533 | 0.517 | 0.115 |
| corn, mp6 to mp5 | 0.431 | 0.914 | 0.492 | 1.066 | 0.498 | 0.458 | 0.488 | 0.467 | 0.425 |
| tablets, 1 to 2 | 0.239 | 1.599 | 0.429 | 1.606 | 0.432 | 0.482 | 0.350 | 0.333 | 0.242 |
| tablets, 2 to 1 | 0.242 | 3.260 | 0.374 | 3.135 | 0.375 | 0.387 | 0.372 | 0.328 | 0.239 |

## The gate of the plan item, case by case

A case is one crossing and one property. "Ours" is kernel ridge on the slopes with the shift fitted and each instrument centred, which uses no standard. An interval is a 95 % bootstrap interval over the splits.

| Set | Cases | Ours below PLS on the plain spectra as measured | Ours within the interval of PLS with PDS (8 standards) | Ours below PLS with PDS | Ours above PLS with PDS | Ours above PLS with PDS and 16 standards | Ours within the interval of PLS on the same rows |
|---|---|---|---|---|---|---|---|
| corn | 24 | 24 | 5 | 7 | 12 | 13 | 15 |
| tablets | 2 | 2 | 2 | 0 | 0 | 1 | 1 |

| Crossing and property | Ours | PLS, plain, as measured | PLS with PDS | PLS on the same rows | Ours minus PLS with PDS |
|---|---|---|---|---|---|
| corn, m5 to mp5, moisture | 0.2759 | 1.5468 | 0.3043 | 0.2762 | -0.0284 [-0.0424, -0.0146] |
| corn, m5 to mp5, oil | 0.0770 | 0.7274 | 0.0873 | 0.0778 | -0.0103 [-0.0197, -0.0018] |
| corn, m5 to mp5, protein | 0.1795 | 0.9210 | 0.1617 | 0.1792 | +0.0178 [+0.0042, +0.0335] |
| corn, m5 to mp5, starch | 0.3925 | 0.8789 | 0.4152 | 0.3997 | -0.0227 [-0.0438, +0.0031] |
| corn, m5 to mp6, moisture | 0.3065 | 1.7586 | 0.3378 | 0.3060 | -0.0313 [-0.0470, -0.0164] |
| corn, m5 to mp6, oil | 0.0794 | 0.6417 | 0.0891 | 0.0799 | -0.0098 [-0.0177, -0.0025] |
| corn, m5 to mp6, protein | 0.1831 | 0.9639 | 0.1491 | 0.1816 | +0.0340 [+0.0181, +0.0525] |
| corn, m5 to mp6, starch | 0.3935 | 0.9298 | 0.4201 | 0.4000 | -0.0266 [-0.0497, -0.0003] |
| corn, mp5 to m5, moisture | 0.2780 | 2.2585 | 0.2614 | 0.3043 | +0.0167 [-0.0031, +0.0353] |
| corn, mp5 to m5, oil | 0.0922 | 0.2408 | 0.0948 | 0.1008 | -0.0026 [-0.0132, +0.0067] |
| corn, mp5 to m5, protein | 0.1811 | 1.6808 | 0.1568 | 0.1800 | +0.0243 [+0.0042, +0.0450] |
| corn, mp5 to m5, starch | 0.3494 | 2.7304 | 0.2889 | 0.3040 | +0.0605 [+0.0236, +0.1038] |
| corn, mp5 to mp6, moisture | 0.1967 | 0.9099 | 0.1793 | 0.2003 | +0.0174 [+0.0039, +0.0344] |
| corn, mp5 to mp6, oil | 0.1009 | 0.2217 | 0.0988 | 0.1059 | +0.0021 [-0.0036, +0.0081] |
| corn, mp5 to mp6, protein | 0.1815 | 0.2763 | 0.1457 | 0.1816 | +0.0358 [+0.0217, +0.0519] |
| corn, mp5 to mp6, starch | 0.4175 | 0.8842 | 0.3741 | 0.3963 | +0.0433 [+0.0196, +0.0694] |
| corn, mp6 to m5, moisture | 0.2545 | 1.3898 | 0.2080 | 0.2585 | +0.0465 [+0.0184, +0.0751] |
| corn, mp6 to m5, oil | 0.0966 | 0.4227 | 0.1087 | 0.1113 | -0.0121 [-0.0222, -0.0029] |
| corn, mp6 to m5, protein | 0.1736 | 0.9197 | 0.1443 | 0.1772 | +0.0293 [+0.0137, +0.0476] |
| corn, mp6 to m5, starch | 0.4214 | 4.0277 | 0.3630 | 0.4252 | +0.0584 [+0.0323, +0.0858] |
| corn, mp6 to mp5, moisture | 0.1769 | 0.5869 | 0.1688 | 0.1855 | +0.0081 [-0.0082, +0.0236] |
| corn, mp6 to mp5, oil | 0.0976 | 0.2174 | 0.1156 | 0.1050 | -0.0180 [-0.0258, -0.0109] |
| corn, mp6 to mp5, protein | 0.1874 | 0.2708 | 0.1549 | 0.1874 | +0.0325 [+0.0166, +0.0505] |
| corn, mp6 to mp5, starch | 0.5039 | 0.8524 | 0.4529 | 0.5223 | +0.0510 [+0.0161, +0.0846] |
| tablets, 1 to 2, assay | 6.4285 | 30.0345 | 6.3446 | 6.7585 | +0.0839 [-1.3472, +1.0866] |
| tablets, 2 to 1, assay | 5.6228 | 53.6262 | 5.5704 | 5.7734 | +0.0524 [-0.5688, +0.6810] |

## Part 3: a simulated difference in resolution, the target is the source instrument with a lower resolution

| Set | Model and rows | Target broadened by, nm | Read as measured | Each instrument centred, widths not matched | Widths matched, difference known | Widths matched, difference fitted | Fitted difference, nm |
|---|---|---|---|---|---|---|---|
| corn | PLS, plain spectra | 8 | 1.293 | 0.467 | 0.345 | 0.345 | 8.0 |
| corn | PLS, plain spectra | 16 | 3.794 | 0.668 | 0.357 | 0.357 | 16.0 |
| corn | PLS, plain spectra | 32 | 8.525 | 1.010 | 0.380 | 0.380 | 32.0 |
| corn | PLS, slopes | 8 | 1.444 | 0.479 | 0.325 | 0.325 | 8.0 |
| corn | PLS, slopes | 16 | 3.230 | 0.723 | 0.344 | 0.344 | 16.0 |
| corn | PLS, slopes | 32 | 5.419 | 1.004 | 0.376 | 0.376 | 32.0 |
| corn | kernel ridge, plain spectra | 8 | 1.218 | 0.456 | 0.328 | 0.328 | 8.0 |
| corn | kernel ridge, plain spectra | 16 | 3.488 | 0.652 | 0.341 | 0.341 | 16.0 |
| corn | kernel ridge, plain spectra | 32 | 8.010 | 0.973 | 0.359 | 0.359 | 32.0 |
| corn | kernel ridge, slopes | 8 | 1.394 | 0.475 | 0.312 | 0.312 | 8.0 |
| corn | kernel ridge, slopes | 16 | 2.963 | 0.712 | 0.332 | 0.332 | 16.0 |
| corn | kernel ridge, slopes | 32 | 5.193 | 0.978 | 0.357 | 0.357 | 32.0 |
| tablets | PLS, plain spectra | 8 | 1.266 | 0.322 | 0.245 | 0.245 | 8.0 |
| tablets | PLS, plain spectra | 16 | 3.550 | 0.546 | 0.245 | 0.245 | 16.0 |
| tablets | PLS, plain spectra | 32 | 6.014 | 0.967 | 0.245 | 0.245 | 32.0 |
| tablets | PLS, slopes | 8 | 1.339 | 0.390 | 0.243 | 0.243 | 8.0 |
| tablets | PLS, slopes | 16 | 3.672 | 0.724 | 0.243 | 0.243 | 16.0 |
| tablets | PLS, slopes | 32 | 6.096 | 1.105 | 0.244 | 0.244 | 32.0 |
| tablets | kernel ridge, plain spectra | 8 | 1.246 | 0.335 | 0.239 | 0.239 | 8.0 |
| tablets | kernel ridge, plain spectra | 16 | 3.323 | 0.588 | 0.240 | 0.240 | 16.0 |
| tablets | kernel ridge, plain spectra | 32 | 5.409 | 0.990 | 0.241 | 0.241 | 32.0 |
| tablets | kernel ridge, slopes | 8 | 1.444 | 0.392 | 0.236 | 0.236 | 8.0 |
| tablets | kernel ridge, slopes | 16 | 3.565 | 0.694 | 0.236 | 0.236 | 16.0 |
| tablets | kernel ridge, slopes | 32 | 5.569 | 1.017 | 0.241 | 0.241 | 32.0 |

## Part 3: a simulated difference in resolution, another instrument with a lower resolution

In this table every column has the shift fitted and each instrument centred.

| Set | Model and rows | Target broadened by, nm | Shift fitted and each instrument centred, widths not matched | Widths matched, difference known | Widths matched, difference fitted | Fitted difference, nm |
|---|---|---|---|---|---|---|
| corn | PLS, plain spectra | 8 | 0.581 | 0.557 | 0.557 | 8.0 |
| corn | PLS, plain spectra | 16 | 0.744 | 0.564 | 0.564 | 16.0 |
| corn | PLS, plain spectra | 32 | 1.055 | 0.584 | 0.584 | 32.0 |
| corn | PLS, slopes | 8 | 0.567 | 0.529 | 0.529 | 8.0 |
| corn | PLS, slopes | 16 | 0.777 | 0.544 | 0.544 | 16.0 |
| corn | PLS, slopes | 32 | 1.038 | 0.572 | 0.572 | 32.0 |
| corn | kernel ridge, plain spectra | 8 | 0.556 | 0.531 | 0.531 | 8.0 |
| corn | kernel ridge, plain spectra | 16 | 0.716 | 0.541 | 0.541 | 16.0 |
| corn | kernel ridge, plain spectra | 32 | 1.011 | 0.553 | 0.553 | 32.0 |
| corn | kernel ridge, slopes | 8 | 0.559 | 0.512 | 0.512 | 8.0 |
| corn | kernel ridge, slopes | 16 | 0.764 | 0.527 | 0.527 | 16.0 |
| corn | kernel ridge, slopes | 32 | 1.009 | 0.549 | 0.549 | 32.0 |
| tablets | PLS, plain spectra | 8 | 0.465 | 0.423 | 0.423 | 8.0 |
| tablets | PLS, plain spectra | 16 | 0.625 | 0.448 | 0.448 | 16.0 |
| tablets | PLS, plain spectra | 32 | 1.002 | 0.462 | 0.462 | 32.0 |
| tablets | PLS, slopes | 8 | 0.462 | 0.364 | 0.364 | 8.0 |
| tablets | PLS, slopes | 16 | 0.765 | 0.390 | 0.390 | 16.0 |
| tablets | PLS, slopes | 32 | 1.123 | 0.459 | 0.459 | 32.0 |
| tablets | kernel ridge, plain spectra | 8 | 0.445 | 0.403 | 0.403 | 8.0 |
| tablets | kernel ridge, plain spectra | 16 | 0.643 | 0.416 | 0.416 | 16.0 |
| tablets | kernel ridge, plain spectra | 32 | 1.018 | 0.430 | 0.430 | 32.0 |
| tablets | kernel ridge, slopes | 8 | 0.449 | 0.358 | 0.358 | 8.0 |
| tablets | kernel ridge, slopes | 16 | 0.720 | 0.364 | 0.364 | 16.0 |
| tablets | kernel ridge, slopes | 32 | 1.029 | 0.427 | 0.427 | 32.0 |

