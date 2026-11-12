# Spectra with two axes: fluorescence maps and peak lists

Data: fluorescence maps of honeys (https://ucphchemometrics.com/honey/) and of olive oils during ageing (https://doi.org/10.17632/g6y69g8gwm.1, CC BY 4.0); no file of the two sets is in this repository. Widths that can be chosen: 0, 10, 20 nm along the excitation axis and 0, 4, 8, 16 nm along the emission axis.

## Part 1: honey, four botanical origins

104 real honeys (47 acacia, 12 linden, 30 meadow, 15 sunflower), maps of 52 by 185 points after the scattered light is taken out; five folds, 20 repeats. A number in brackets is the standard deviation over the repeats.

| Model | Right class, % | Mean over the classes of the share found, % |
|---|---|---|
| always the largest class | 45.2 | 25.0 |
| PLS-DA on the map as measured | 87.0 (3.1) | 86.3 (3.2) |
| PLS-DA, description chosen inside the fit | 86.3 (1.9) | 84.7 (2.4) |
| kernel Fisher discriminant, description chosen inside the fit | 87.9 (2.1) | 86.9 (2.0) |

Kernel Fisher discriminant minus PLS-DA with the same choices, mean over the classes: +2.2 [+0.7, +3.6] points; minus PLS-DA on the map as measured: +0.6 [-0.7, +2.0] points (95 % bootstrap interval over the repeats).

Chosen most often: PLS-DA: 20 nm by 0 nm, each map divided by its length (22 of 100 fits); kernel Fisher discriminant: 20 nm by 0 nm, each map divided by its length (26 of 100 fits).

Where the honeys of a class are put by the kernel Fisher discriminant (rows: the class; columns: what the model said; all repeats together):

| | acacia | linden | meadow | sunflower |
|---|---|---|---|---|
| acacia | 850 | 2 | 84 | 4 |
| linden | 2 | 204 | 33 | 1 |
| meadow | 59 | 16 | 515 | 10 |
| sunflower | 12 | 0 | 29 | 259 |

The places of the map that speak most for (+) or against (-) a class (description: 20 nm by 0 nm, each map divided by its length):

| Class | Excitation / emission, nm, and sign |
|---|---|
| acacia | 370 / 431 (-); 250 / 435 (-); 275 / 369 (-) |
| linden | 240 / 439 (-); 310 / 437 (-); 345 / 461 (-) |
| meadow | 345 / 461 (+); 240 / 383 (+); 350 / 419 (+) |
| sunflower | 240 / 437 (+); 240 / 383 (-); 380 / 437 (+) |

## Part 1: the authentication case

A description of the real honeys of the fit alone; the 6 false honeys against the real honeys that the description has not seen. The number is the share of pairs of a real and a false honey that are put in the right order (0.5 is chance).

| Description of the class | Maps as measured, each divided by its length | Maps broadened by the largest widths |
|---|---|---|
| distance to the mean map | 0.909 (0.004) | 0.884 (0.014) |
| distance to the fifth nearest honey | 0.982 (0.001) | 0.982 (0.003) |
| distance from the main subspace (5 components) | 0.926 (0.003) | 0.966 (0.004) |
| Gaussian process on a constant | 0.965 (0.003) | 0.848 (0.005) |

## Part 2: olive oil, the absorbance from the map

240 oil samples (24 oils at 10 steps of ageing) of 720 maps with 35 by 251 points; 6 folds, 10 repeats. Root-mean-square errors in absorbance; a number in brackets is the standard deviation over the repeats. The standard deviation of the absorbance at 232 nm is 0.284 and at 268 nm 0.075.

| Model and input | 232 nm, whole oils held out | 232 nm, random folds | 268 nm, whole oils held out | 268 nm, random folds |
|---|---|---|---|---|
| the mean of the fit | 0.2915 (0.0034) | 0.2854 (0.0007) | 0.0756 (0.0004) | 0.0754 (0.0002) |
| linear regression on the sum of the intensities | 0.2931 (0.0030) | 0.2862 (0.0008) | 0.0733 (0.0006) | 0.0724 (0.0003) |
| PLS on the map as measured | 0.2560 (0.0233) | 0.1182 (0.0028) | 0.0409 (0.0124) | 0.0217 (0.0006) |
| PLS, description chosen inside the fit | 0.2709 (0.0216) | 0.1146 (0.0031) | 0.0376 (0.0110) | 0.0216 (0.0011) |
| kernel ridge, description chosen inside the fit | 0.2693 (0.0230) | 0.1105 (0.0042) | 0.0320 (0.0064) | 0.0207 (0.0005) |
| PLS, the mean of the three maps of a sample | 0.2762 (0.0198) |  | 0.0307 (0.0079) |  |
| kernel ridge, the mean of the three maps | 0.2614 (0.0194) |  | 0.0279 (0.0036) |  |

Differences with whole oils held out (95 % bootstrap interval over the repeats):

- 232 nm: kernel ridge minus PLS with the same choices -0.0016 [-0.0112, +0.0079]; minus PLS on the map as measured +0.0133 [+0.0057, +0.0229]; the mean of three maps minus one map (kernel ridge) -0.0079 [-0.0153, -0.0018].
- 268 nm: kernel ridge minus PLS with the same choices -0.0056 [-0.0132, +0.0009]; minus PLS on the map as measured -0.0089 [-0.0168, -0.0011]; the mean of three maps minus one map (kernel ridge) -0.0041 [-0.0085, -0.0001].
- Chosen most often with whole oils held out: k232, PLS: 0 nm by 16 nm, each map divided by its length; k268, PLS: no broadening; k232, kernel ridge: 20 nm by 16 nm, each map divided by its length; k268, kernel ridge: no broadening.

## Part 2: within the limits of the regulation or not

168 of the 240 samples are within the three limits for extra virgin oil. Mean over the two classes of the share found, %.

| Model and input | Whole oils held out | Random folds |
|---|---|---|
| PLS-DA on the map as measured | 85.5 (2.7) | 89.8 (1.5) |
| PLS-DA, description chosen inside the fit | 84.7 (1.7) | 89.2 (0.9) |
| kernel Fisher discriminant, description chosen inside the fit | 86.0 (2.3) | 89.9 (0.9) |
| kernel Fisher discriminant, the mean of the three maps | 85.5 (2.5) |  |

Kernel Fisher discriminant minus PLS-DA, whole oils held out: +1.3 [+0.2, +2.4] points.

## Part 3: simulated peak lists

Two classes of 50 lists with 40 peaks on a square of side 100; in one class 4 peaks are stronger by 60 %. The jitter is the standard deviation of the place of a peak along each axis. Right class, %; 10 repeats.

| Jitter | Kernel of the peak lists, kernel Fisher discriminant | Kernel of the peak lists, PLS-DA | Bins, kernel Fisher discriminant | Bins, PLS-DA | PLS-DA, bins of 1 | PLS-DA, bins of 2 | PLS-DA, bins of 4 | PLS-DA, bins of 8 | PLS-DA, bins of 16 | Width chosen | Bin chosen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 91.7 (2.2) | 90.4 (3.0) | 91.6 (2.6) | 91.6 (3.4) | 91.7 (2.6) | 91.6 (3.1) | 91.1 (3.0) | 87.5 (4.1) | 84.3 (3.6) | 0.5 | 1 |
| 0.5 | 90.2 (6.3) | 89.4 (6.3) | 85.1 (7.3) | 86.2 (7.7) | 63.5 (7.3) | 72.1 (9.7) | 83.6 (6.2) | 87.0 (7.2) | 81.1 (7.9) | 2 | 8 |
| 1 | 86.7 (6.1) | 86.2 (5.3) | 81.0 (4.4) | 80.4 (4.9) | 53.3 (6.4) | 57.9 (6.3) | 70.9 (6.2) | 80.1 (5.3) | 79.4 (7.4) | 4 | 8 |
| 2 | 80.4 (6.9) | 78.7 (5.6) | 71.1 (8.3) | 71.1 (8.2) | 51.5 (3.4) | 51.9 (5.0) | 56.8 (6.7) | 66.5 (4.5) | 73.1 (7.9) | 8 | 16 |
| 4 | 75.4 (7.2) | 76.1 (6.9) | 67.1 (7.8) | 66.2 (7.3) | 53.5 (4.8) | 54.8 (4.4) | 56.2 (3.8) | 61.3 (6.4) | 66.2 (7.3) | 8 | 16 |

Kernel of the peak lists with the kernel Fisher discriminant minus PLS-DA on bins, points: jitter 0: +0.1 [-2.1, +2.3]; jitter 0.5: +4.0 [+0.5, +7.1]; jitter 1: +6.3 [+3.9, +8.4]; jitter 2: +9.3 [+4.2, +14.5]; jitter 4: +9.2 [+5.8, +12.4].

## The kernel correlation

The correlation of the kernel of a new description with the kernel of the old one, over the pairs of different samples. Maps: every broadened description against the map as measured, both in the cosine form.

| Set | Lowest correlation over the widths | At the widths | Highest |
|---|---|---|---|
| honey | 0.9581 | 20 nm by 16 nm | 1.0000 |
| olive | 0.9925 | 20 nm by 16 nm | 0.9996 |

Simulated peak lists: the kernel of the peak lists against the kernel of the bins, one draw for every size of the jitter.

| Jitter | width 2, bins of 1 | width 2, bins of 8 | width 8, bins of 1 | width 8, bins of 8 |
|---|---|---|---|---|
| 0 | 0.906 | 0.906 | 0.660 | 0.723 |
| 0.5 | 0.189 | 0.421 | 0.028 | 0.278 |
| 1 | 0.276 | 0.369 | 0.064 | 0.186 |
| 2 | 0.241 | 0.437 | 0.040 | 0.213 |
| 4 | 0.200 | 0.530 | 0.050 | 0.283 |

