# Mixtures and quantification: the spectrum of a mixture as a sum of fields (plan item SP6, half a)

Every repeat draws 4 components and makes 120 mixtures to fit and 120 to test. The total amount of a mixture is between 0.5 and 2; every mixture has a baseline (an offset up to 0.2 and a rise or fall up to 0.1 over the axis) and noise of standard deviation 0.005, all in units of the strongest band of a pure component. The numbers are root-mean-square errors on the test mixtures, the mean over the repeats with the standard deviation in brackets. The width of the broadening, the ridge, the length of the Gaussian layer and the number of PLS components are chosen by cross-validation inside the mixtures to fit.

The floor is a least-squares fit of every test mixture with the known spectra of the pure components and a baseline. No model that has to learn from mixtures can be better, as long as the response is linear. A difference is given with a 95 % bootstrap interval over the repeats; below zero the first of the two is better.

The identity: the kernel of mixtures made as spectra and the product W K W^T of `spectrum.mixture_kernel` differ by 5.9e-16 of the largest element.

## Components: computed spectra of benzoic acids

Question: the amount of every component. Error of the floor: 0.0022 (0.0007).

| Rows of the description | Kernel ridge, linear kernel | Kernel ridge, Gaussian layer | PLS on the same rows | Linear kernel minus PLS | Gaussian layer minus linear kernel | Linear kernel over the floor |
|---|---|---|---|---|---|---|
| plain, not normalised | 0.0024 (0.0008) | 0.0025 (0.0008) | 0.0024 (0.0008) | +0.0001 [+0.0000, +0.0001] | +0.0000 [+0.0000, +0.0001] | 1.09 |
| plain, cosine | 0.1291 (0.0078) | 0.1156 (0.0076) | 0.1297 (0.0079) | -0.0006 [-0.0013, +0.0001] | -0.0135 [-0.0155, -0.0116] | 62.43 |
| slope, not normalised | 0.0027 (0.0008) | 0.0027 (0.0008) | 0.0027 (0.0008) | -0.0000 [-0.0000, +0.0000] | +0.0001 [+0.0000, +0.0001] | 1.19 |
| slope, cosine | 0.1353 (0.0074) | 0.1171 (0.0096) | 0.1358 (0.0078) | -0.0004 [-0.0016, +0.0008] | -0.0182 [-0.0216, -0.0149] | 65.30 |

Question: the fraction of every component. Error of the floor: 0.0022 (0.0007).

| Rows of the description | Kernel ridge, linear kernel | Kernel ridge, Gaussian layer | PLS on the same rows | Linear kernel minus PLS | Gaussian layer minus linear kernel | Linear kernel over the floor |
|---|---|---|---|---|---|---|
| plain, not normalised, amounts divided by their sum | 0.0024 (0.0008) | 0.0024 (0.0008) | 0.0023 (0.0008) | +0.0001 [+0.0000, +0.0001] | +0.0000 [+0.0000, +0.0001] | 1.07 |
| plain, not normalised | 0.0655 (0.0057) | 0.0236 (0.0025) | 0.0656 (0.0056) | -0.0001 [-0.0002, +0.0001] | -0.0419 [-0.0442, -0.0396] | 32.22 |
| plain, cosine | 0.0623 (0.0058) | 0.0166 (0.0021) | 0.0623 (0.0058) | +0.0001 [-0.0001, +0.0002] | -0.0457 [-0.0481, -0.0435] | 30.54 |
| slope, not normalised, amounts divided by their sum | 0.0026 (0.0009) | 0.0027 (0.0009) | 0.0026 (0.0009) | -0.0000 [-0.0000, +0.0000] | +0.0000 [+0.0000, +0.0001] | 1.18 |
| slope, not normalised | 0.0650 (0.0057) | 0.0114 (0.0023) | 0.0652 (0.0057) | -0.0002 [-0.0005, +0.0001] | -0.0535 [-0.0559, -0.0512] | 31.94 |
| slope, cosine | 0.0140 (0.0055) | 0.0049 (0.0015) | 0.0139 (0.0054) | +0.0001 [+0.0000, +0.0003] | -0.0092 [-0.0117, -0.0069] | 7.28 |

The co-spectrum of the model for an amount (slopes, not normalised) against the net signal of that component, which is the part of its pure spectrum that the other components and a baseline cannot make: cosine 0.9895 (0.0049).

## Components: measured spectra of the Chemotion IR set

Question: the amount of every component. Error of the floor: 0.0013 (0.0003).

| Rows of the description | Kernel ridge, linear kernel | Kernel ridge, Gaussian layer | PLS on the same rows | Linear kernel minus PLS | Gaussian layer minus linear kernel | Linear kernel over the floor |
|---|---|---|---|---|---|---|
| plain, not normalised | 0.0014 (0.0003) | 0.0015 (0.0003) | 0.0014 (0.0003) | +0.0000 [+0.0000, +0.0000] | +0.0001 [+0.0001, +0.0002] | 1.11 |
| plain, cosine | 0.1303 (0.0107) | 0.1250 (0.0110) | 0.1315 (0.0106) | -0.0012 [-0.0022, -0.0003] | -0.0052 [-0.0077, -0.0027] | 106.68 |
| slope, not normalised | 0.0020 (0.0005) | 0.0021 (0.0006) | 0.0020 (0.0006) | -0.0000 [-0.0001, -0.0000] | +0.0001 [+0.0000, +0.0002] | 1.57 |
| slope, cosine | 0.1367 (0.0070) | 0.1224 (0.0065) | 0.1386 (0.0071) | -0.0018 [-0.0030, -0.0006] | -0.0143 [-0.0167, -0.0119] | 112.14 |

Question: the fraction of every component. Error of the floor: 0.0012 (0.0002).

| Rows of the description | Kernel ridge, linear kernel | Kernel ridge, Gaussian layer | PLS on the same rows | Linear kernel minus PLS | Gaussian layer minus linear kernel | Linear kernel over the floor |
|---|---|---|---|---|---|---|
| plain, not normalised, amounts divided by their sum | 0.0014 (0.0003) | 0.0015 (0.0003) | 0.0013 (0.0003) | +0.0000 [+0.0000, +0.0000] | +0.0001 [+0.0001, +0.0001] | 1.10 |
| plain, not normalised | 0.0654 (0.0050) | 0.0184 (0.0031) | 0.0656 (0.0051) | -0.0002 [-0.0005, +0.0001] | -0.0470 [-0.0491, -0.0447] | 54.25 |
| plain, cosine | 0.0417 (0.0072) | 0.0091 (0.0039) | 0.0416 (0.0074) | +0.0001 [-0.0003, +0.0004] | -0.0326 [-0.0348, -0.0304] | 34.40 |
| slope, not normalised, amounts divided by their sum | 0.0019 (0.0005) | 0.0019 (0.0006) | 0.0019 (0.0005) | -0.0000 [-0.0001, -0.0000] | +0.0000 [+0.0000, +0.0001] | 1.51 |
| slope, not normalised | 0.0648 (0.0048) | 0.0129 (0.0030) | 0.0647 (0.0051) | +0.0001 [-0.0002, +0.0004] | -0.0519 [-0.0540, -0.0496] | 53.72 |
| slope, cosine | 0.0227 (0.0038) | 0.0048 (0.0011) | 0.0227 (0.0037) | +0.0001 [-0.0002, +0.0004] | -0.0180 [-0.0194, -0.0166] | 18.94 |

The co-spectrum of the model for an amount (slopes, not normalised) against the net signal of that component, which is the part of its pure spectrum that the other components and a baseline cannot make: cosine 0.9654 (0.0386).

## Components: computed spectra of benzoic acids, with stray light

Question: the amount of every component. Error of the least-squares fit with the pure spectra, which is not a floor here because it takes the response as linear: 0.0535 (0.0094).

| Rows of the description | Kernel ridge, linear kernel | Kernel ridge, Gaussian layer | PLS on the same rows | Linear kernel minus PLS | Gaussian layer minus linear kernel | Linear kernel over the floor |
|---|---|---|---|---|---|---|
| plain, not normalised | 0.0040 (0.0008) | 0.0040 (0.0011) | 0.0048 (0.0022) | -0.0008 [-0.0017, -0.0002] | -0.0000 [-0.0002, +0.0002] | 0.08 |
| plain, cosine | 0.0784 (0.0087) | 0.0544 (0.0103) | 0.0788 (0.0095) | -0.0004 [-0.0020, +0.0011] | -0.0240 [-0.0271, -0.0211] | 1.52 |
| slope, not normalised | 0.0047 (0.0011) | 0.0045 (0.0015) | 0.0045 (0.0011) | +0.0002 [+0.0000, +0.0003] | -0.0002 [-0.0005, +0.0001] | 0.09 |
| slope, cosine | 0.0628 (0.0097) | 0.0462 (0.0128) | 0.0606 (0.0097) | +0.0022 [+0.0013, +0.0031] | -0.0166 [-0.0201, -0.0131] | 1.22 |

Question: the fraction of every component. Error of the least-squares fit with the pure spectra, which is not a floor here because it takes the response as linear: 0.0165 (0.0076).

| Rows of the description | Kernel ridge, linear kernel | Kernel ridge, Gaussian layer | PLS on the same rows | Linear kernel minus PLS | Gaussian layer minus linear kernel | Linear kernel over the floor |
|---|---|---|---|---|---|---|
| plain, not normalised, amounts divided by their sum | 0.0035 (0.0005) | 0.0032 (0.0006) | 0.0038 (0.0011) | -0.0003 [-0.0007, +0.0000] | -0.0003 [-0.0004, -0.0002] | 0.25 |
| plain, not normalised | 0.0470 (0.0049) | 0.0166 (0.0023) | 0.0492 (0.0059) | -0.0022 [-0.0035, -0.0011] | -0.0304 [-0.0321, -0.0287] | 3.42 |
| plain, cosine | 0.0492 (0.0042) | 0.0120 (0.0018) | 0.0494 (0.0041) | -0.0001 [-0.0006, +0.0003] | -0.0373 [-0.0387, -0.0358] | 3.60 |
| slope, not normalised, amounts divided by their sum | 0.0039 (0.0006) | 0.0034 (0.0007) | 0.0037 (0.0006) | +0.0002 [+0.0001, +0.0003] | -0.0005 [-0.0007, -0.0004] | 0.27 |
| slope, not normalised | 0.0466 (0.0049) | 0.0091 (0.0021) | 0.0470 (0.0056) | -0.0004 [-0.0015, +0.0004] | -0.0375 [-0.0397, -0.0355] | 3.39 |
| slope, cosine | 0.0139 (0.0037) | 0.0043 (0.0010) | 0.0142 (0.0040) | -0.0003 [-0.0006, -0.0001] | -0.0096 [-0.0111, -0.0081] | 1.05 |

## Components: computed spectra of benzoic acids, with a baseline only in the test mixtures

Question: the amount of every component. Error of the floor: 0.0023 (0.0009).

| Rows of the description | Kernel ridge, linear kernel | Kernel ridge, Gaussian layer | PLS on the same rows | Linear kernel minus PLS | Gaussian layer minus linear kernel | Linear kernel over the floor |
|---|---|---|---|---|---|---|
| plain, not normalised | 0.1788 (0.0595) | 0.1738 (0.0552) | 0.1819 (0.0768) | -0.0031 [-0.0145, +0.0060] | -0.0050 [-0.0096, -0.0023] | 80.21 |
| plain, cosine | 0.3405 (0.2621) | 0.3071 (0.1357) | 0.3809 (0.5810) | -0.0404 [-0.2205, +0.0730] | -0.0334 [-0.1282, +0.0283] | 143.36 |
| slope, not normalised | 0.0029 (0.0012) | 0.0030 (0.0013) | 0.0029 (0.0012) | +0.0000 [-0.0000, +0.0001] | +0.0001 [+0.0001, +0.0001] | 1.25 |
| slope, cosine | 0.1364 (0.0072) | 0.1159 (0.0082) | 0.1349 (0.0068) | +0.0015 [-0.0010, +0.0045] | -0.0205 [-0.0238, -0.0175] | 64.72 |

Question: the fraction of every component. Error of the floor: 0.0023 (0.0009).

| Rows of the description | Kernel ridge, linear kernel | Kernel ridge, Gaussian layer | PLS on the same rows | Linear kernel minus PLS | Gaussian layer minus linear kernel | Linear kernel over the floor |
|---|---|---|---|---|---|---|
| plain, not normalised, amounts divided by their sum | 0.0958 (0.0433) | 0.0944 (0.0409) | 0.0984 (0.0534) | -0.0026 [-0.0091, +0.0023] | -0.0014 [-0.0036, -0.0000] | 42.29 |
| plain, not normalised | 0.2939 (0.2913) | 0.1965 (0.0408) | 0.2418 (0.2206) | +0.0521 [-0.1005, +0.1985] | -0.0974 [-0.2350, +0.0155] | 139.85 |
| plain, cosine | 0.1486 (0.0432) | 0.1315 (0.0268) | 0.1452 (0.0490) | +0.0034 [-0.0050, +0.0133] | -0.0172 [-0.0368, +0.0007] | 70.33 |
| slope, not normalised, amounts divided by their sum | 0.0029 (0.0012) | 0.0029 (0.0012) | 0.0028 (0.0011) | +0.0000 [-0.0000, +0.0000] | +0.0001 [+0.0000, +0.0001] | 1.23 |
| slope, not normalised | 0.0658 (0.0034) | 0.0116 (0.0034) | 0.0657 (0.0036) | +0.0000 [-0.0003, +0.0004] | -0.0542 [-0.0559, -0.0521] | 31.76 |
| slope, cosine | 0.0131 (0.0044) | 0.0049 (0.0014) | 0.0130 (0.0045) | +0.0000 [-0.0001, +0.0001] | -0.0081 [-0.0104, -0.0059] | 6.82 |

The co-spectrum of the model for an amount (slopes, not normalised) against the net signal of that component, which is the part of its pure spectrum that the other components and a baseline cannot make: cosine 0.9777 (0.0367).

