# A spectrum as a one-dimensional field: a calculated powder pattern as the sample (plan item SP1)

The Matbench phonons task: 1,265 inorganic crystals with the frequency of the highest peak of the phonon density of states (cm^-1; standard deviation 494 cm^-1).  The powder diffraction pattern of every crystal is calculated from its structure with pymatgen, for X-rays (Cu K-alpha, 1.5418 A) and for neutrons of the same wavelength, between 5 and 90 degrees of the diffraction angle 2-theta.  No measured pattern is used.

A pattern is a stick spectrum: one position and one intensity for every reflection.  The kernel of two patterns is the overlap of the two after every stick is replaced by a Gaussian (`cmfqsar/spectrum.py`), divided by the norms of the two.  The width of the Gaussian (one of 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10 degrees), the power of the intensity (1, or 0.5 for the square root) and the ridge are chosen by 10-fold cross-validation inside every training part.

The crystals have 48 X-ray reflections at the median (2 to 634).

## How alike the kernels are

The correlation between the off-diagonal elements of two kernels, each centred first, on all crystals and at the setting that the folds chose most often.  Near 1, two descriptions find the same pairs of crystals alike, and the second cannot add much to the first.  The last column is the same correlation with the products of the centred targets: how far crystals that a description finds alike have alike frequencies.

| description | setting | structure | composition | G(r) | X-ray pattern | neutron pattern | with the target |
|---|---|---:|---:|---:|---:|---:|---:|
| structure: the periodic autocorrelation kernel | 0.5 | 1 | +0.233 | +0.085 | +0.047 | +0.208 | +0.181 |
| composition: the fractions of the elements | gauss 1 | +0.233 | 1 | +0.043 | +0.050 | +0.126 | +0.140 |
| pair distribution function G(r) to 10 A, X-ray weights (plan item M12) | G(r) | +0.085 | +0.043 | 1 | +0.032 | +0.067 | +0.004 |
| X-ray powder pattern | 10 deg, power 1 | +0.047 | +0.050 | +0.032 | 1 | +0.263 | +0.009 |
| neutron powder pattern | 10 deg, power 1 | +0.208 | +0.126 | +0.067 | +0.263 | 1 | +0.012 |

The kernel of the X-ray patterns by the width of the broadening and the power of the intensity:

| width, degrees | power of the intensity | correlation with the structural kernel | correlation with the composition kernel | correlation with the target |
|---:|---:|---:|---:|---:|
| 0.05 | 1 | +0.040 | +0.019 | +0.004 |
| 0.05 | 0.5 | +0.045 | +0.020 | +0.003 |
| 0.1 | 1 | +0.055 | +0.024 | +0.004 |
| 0.1 | 0.5 | +0.061 | +0.024 | +0.003 |
| 0.2 | 1 | +0.077 | +0.031 | +0.004 |
| 0.2 | 0.5 | +0.083 | +0.029 | +0.004 |
| 0.5 | 1 | +0.122 | +0.046 | +0.006 |
| 0.5 | 0.5 | +0.129 | +0.040 | +0.007 |
| 1 | 1 | +0.163 | +0.060 | +0.011 |
| 1 | 0.5 | +0.177 | +0.052 | +0.013 |
| 2 | 1 | +0.177 | +0.074 | +0.015 |
| 2 | 0.5 | +0.203 | +0.067 | +0.020 |
| 5 | 1 | +0.109 | +0.083 | +0.016 |
| 5 | 0.5 | +0.135 | +0.083 | +0.019 |
| 10 | 1 | +0.047 | +0.050 | +0.009 |
| 10 | 0.5 | +0.061 | +0.069 | +0.013 |

## The error of every description, on the five Matbench folds

MAE is the mean absolute error on the held-out folds.  RMSE and R2 are computed on the pooled held-out predictions.  A setting of a pattern is its width in degrees and the power of the intensity.  With a Gaussian kernel on top it also has a factor: the Gaussian kernel is exp(-factor D^2 / m), with D the distance of two normalised patterns and m the median of D^2 over the training part.  For joined kernels the column gives the mean share of every kernel over the folds.

| description | MAE, cm^-1 (sd over the five folds) | RMSE, cm^-1 | R2 | settings chosen in the five folds |
|---|---:|---:|---:|---|
| mean of the training part | 324.0 (17.7) | 494.2 | -0.001 |  |
| linear regression on the inverse square root of the mass of the lightest atom | 226.5 (12.2) | 355.8 | +0.481 |  |
| composition: the fractions of the elements | 97.6 (16.7) | 201.7 | +0.833 | gauss 1 in every fold |
| structure: the periodic autocorrelation kernel | 92.7 (4.5) | 141.8 | +0.918 | 0.5 in every fold |
| pair distribution function G(r) to 10 A, X-ray weights (plan item M12) | 287.6 (17.3) | 522.4 | -0.119 | G(r) in every fold |
| X-ray powder pattern | 294.2 (14.8) | 466.8 | +0.107 | 10 deg, power 1 in every fold |
| neutron powder pattern | 278.3 (14.8) | 464.3 | +0.116 | 10 deg, power 1 in every fold |
| X-ray powder pattern, Gaussian kernel on top | 257.4 (9.7) | 438.4 | +0.212 | 5 deg, power 1, factor 2; 5 deg, power 1, factor 2; 5 deg, power 1, factor 2; 5 deg, power 1, factor 1; 5 deg, power 1, factor 2 |
| neutron powder pattern, Gaussian kernel on top | 196.1 (13.7) | 360.8 | +0.466 | 5 deg, power 1, factor 2; 5 deg, power 1, factor 2; 5 deg, power 1, factor 1; 5 deg, power 1, factor 4; 5 deg, power 1, factor 2 |
| X-ray and neutron powder patterns | 265.2 (13.6) | 442.7 | +0.197 | shares xray 0.17, neutron 0.83 |
| composition and X-ray powder pattern | 97.0 (17.1) | 203.3 | +0.831 | shares composition 0.71, xray 0.29 |
| structure and X-ray powder pattern | 93.0 (4.8) | 143.2 | +0.916 | shares structure 1.00, xray 0.00 |
| structure and X-ray powder pattern, equal weights | 92.9 (5.2) | 140.5 | +0.919 | shares structure 0.50, xray 0.50 |
| structure, X-ray and neutron powder patterns | 92.1 (3.7) | 141.7 | +0.918 | shares structure 0.99, xray 0.00, neutron 0.01 |

Published errors on the same folds (Matbench v0.1, MAE in cm^-1, read from the Matbench site in October 2026): the mean as the prediction 324.0; RF-SCM/Magpie 67.6; CrabNet, which uses the composition only, 55.1; MODNet 34.3; MEGNet 28.8.

## The error of the crystals with and without hydrogen

62 of the 1,265 crystals hold hydrogen.  Their median frequency is 1754 cm^-1, and the median of the others is 443 cm^-1: a bond to hydrogen vibrates at a higher frequency than any other bond.  A diffraction pattern shows hydrogen weakly, so the table scores the same held-out predictions for the two groups.  Nothing is fitted again.

| description | MAE of the crystals with hydrogen, cm^-1 | MAE of the crystals without hydrogen, cm^-1 |
|---|---:|---:|
| mean of the training part | 1297.1 | 273.8 |
| linear regression on the inverse square root of the mass of the lightest atom | 799.9 | 197.0 |
| composition: the fractions of the elements | 403.0 | 81.9 |
| structure: the periodic autocorrelation kernel | 273.3 | 83.4 |
| pair distribution function G(r) to 10 A, X-ray weights (plan item M12) | 1233.8 | 238.9 |
| X-ray powder pattern | 1193.1 | 247.9 |
| neutron powder pattern | 1252.7 | 228.0 |
| X-ray powder pattern, Gaussian kernel on top | 1164.6 | 210.6 |
| neutron powder pattern, Gaussian kernel on top | 878.8 | 160.9 |
| X-ray and neutron powder patterns | 1209.3 | 216.5 |
| composition and X-ray powder pattern | 424.6 | 80.1 |
| structure and X-ray powder pattern | 266.7 | 84.1 |
| structure and X-ray powder pattern, equal weights | 260.5 | 84.3 |
| structure, X-ray and neutron powder patterns | 261.6 | 83.3 |

## Differences in the mean absolute error

The MAE of the first description minus the MAE of the second, with a 95 % bootstrap interval over the crystals.  Below zero, the first description has the lower error.

| description | compared with | difference in MAE, cm^-1 | is zero inside the interval |
|---|---|---:|---|
| X-ray powder pattern | structure: the periodic autocorrelation kernel | +201.5 [+184.2, +220.4] | no |
| X-ray powder pattern | composition: the fractions of the elements | +196.6 [+179.4, +214.8] | no |
| X-ray powder pattern | mean of the training part | -29.8 [-37.7, -22.1] | no |
| X-ray powder pattern | pair distribution function G(r) to 10 A, X-ray weights (plan item M12) | +6.6 [-12.1, +22.0] | yes |
| neutron powder pattern | X-ray powder pattern | -16.0 [-23.9, -8.1] | no |
| neutron powder pattern | composition: the fractions of the elements | +180.6 [+163.1, +198.7] | no |
| X-ray powder pattern, Gaussian kernel on top | X-ray powder pattern | -36.9 [-45.7, -28.0] | no |
| X-ray powder pattern, Gaussian kernel on top | composition: the fractions of the elements | +159.7 [+143.4, +177.3] | no |
| neutron powder pattern, Gaussian kernel on top | neutron powder pattern | -82.2 [-94.4, -69.4] | no |
| neutron powder pattern, Gaussian kernel on top | composition: the fractions of the elements | +98.4 [+84.2, +113.1] | no |
| X-ray and neutron powder patterns | X-ray powder pattern | -29.0 [-35.8, -22.4] | no |
| composition and X-ray powder pattern | composition: the fractions of the elements | -0.6 [-2.5, +1.3] | yes |
| structure and X-ray powder pattern | structure: the periodic autocorrelation kernel | +0.3 [-1.2, +1.7] | yes |
| structure and X-ray powder pattern, equal weights | structure: the periodic autocorrelation kernel | +0.2 [-1.3, +1.6] | yes |
| structure, X-ray and neutron powder patterns | structure: the periodic autocorrelation kernel | -0.7 [-2.6, +1.2] | yes |
| structure: the periodic autocorrelation kernel | composition: the fractions of the elements | -4.9 [-14.0, +3.5] | yes |

## The pattern as one more kernel beside the structure

The structural kernel and the kernel of the X-ray patterns are joined with weights that maximise the marginal likelihood of the targets (`cmfqsar/mixing.py`).  On all crystals the share of the structural kernel is 1.00 and the share of the pattern is 0.00.  The log marginal likelihood, each at its best ridge, is -8422.3 for the joined kernel, -8420.9 for the structural kernel alone and -9578.6 for the pattern alone.  The log Bayes factor against *the pattern adds nothing* (`mixing.drop_one`: the loss of log marginal likelihood when the pattern is taken away) is -1.4, and against *the structure adds nothing* it is +1156.3.  The pattern adds nothing to the marginal likelihood either.

The same two numbers with every marginal likelihood taken at the best of five fixed ridges (0.001, 0.01, 0.1, 1, 10), as `mixing.drop_one` computed them until 10 November 2026, are +53.9 and +1051.8.  The first of them is not evidence for the pattern.  The best ridge of the structural kernel is 2.1e-05, below the lowest of the five.  The joined kernel has twice the scale of one kernel, which acts as half the ridge, so the joined kernel was nearer to the best ridge and gained from that alone.  This run found the fault, and `drop_one` now searches the ridge relative to the scale of the kernel.

## The co-spectrum of the X-ray patterns

The model on all crystals with the setting that the folds chose most often (width 10 degrees, power 1, ridge 0.003; 22 grid points, 4 degrees apart).  Its prediction is a constant plus the sum over the axis of a weight times the normalised intensity.  The weight is the co-spectrum (`cospectrum.png`).  The effect of a grid point is its weight times the standard deviation of the intensity there over the set.

The largest effect of a grid point is 206.1 cm^-1.  In 20 models fitted to the targets in random order the largest effect is 93.5 cm^-1 at the mean and 144.2 at most.  The model reproduces the targets it was fitted on with R2 +0.120.

The table sums weight times intensity over bands of 10 degrees.  The second number column is the standard deviation of that sum over the crystals: how much the band moves the predictions.  The third is the correlation of that sum with the target.  The last is the correlation of the intensity in the band with the target, which does not use the model.

| 2-theta, degrees | spacing of the lattice planes d, A | standard deviation of what the band adds to the predictions, cm^-1 | correlation of what the band adds with the target | correlation of the intensity in the band with the target |
|---|---|---:|---:|---:|
| 5 to 15 | 17.67 to 5.91 | 224.2 | +0.08 | -0.05 |
| 15 to 25 | 5.91 to 3.56 | 28.1 | -0.12 | +0.02 |
| 25 to 35 | 3.56 to 2.56 | 310.8 | -0.11 | +0.15 |
| 35 to 45 | 2.56 to 2.01 | 53.4 | +0.09 | -0.02 |
| 45 to 55 | 2.01 to 1.67 | 345.0 | +0.16 | -0.16 |
| 55 to 65 | 1.67 to 1.43 | 17.9 | -0.08 | -0.06 |
| 65 to 75 | 1.43 to 1.27 | 144.2 | +0.04 | -0.03 |
| 75 to 85 | 1.27 to 1.14 | 6.1 | +0.09 | -0.19 |
| 85 to 90 | 1.14 to 1.09 | 68.9 | +0.19 | -0.20 |

## Reading

**The kernels.** The kernel of the X-ray patterns correlates with the structural kernel with r +0.047 and with the composition kernel with r +0.050, so it is a different description.  Its correlation with the target is +0.009, and at no width is it above +0.020.  The structural kernel has +0.181 and the composition kernel +0.140.  A kernel that does not find crystals with alike frequencies alike cannot predict the frequency, and the errors below agree with that.

**The pattern alone.** The X-ray pattern predicts the frequency with a mean absolute error of 294.2 cm^-1.  The mean of the training part gives 324.0, the regression on the mass of the lightest atom 226.5, the composition 97.6 and the structural kernel 92.7.  Against the structural kernel the pattern is +201.5 [+184.2, +220.4] cm^-1 (higher), and against the composition +196.6 [+179.4, +214.8] cm^-1 (higher).  The neutron pattern gives 278.3 (-16.0 [-23.9, -8.1] cm^-1 (lower) against the X-ray pattern).  The pair distribution function of plan item M12, which holds the same untyped distances in real space, gives 287.6 (the X-ray pattern against it: +6.6 [-12.1, +22.0] cm^-1 (not separated)).  In every fold the choice inside the folds takes the widest broadening tried, 10 degrees.  A pattern broadened that far holds only how the intensity is spread over the angle, so the places of the single reflections give the model nothing that it can use.  A Gaussian kernel on top lowers the error to 257.4 for the X-ray pattern and to 196.1 for the neutron pattern.  The neutron pattern is then still +98.4 [+84.2, +113.1] cm^-1 (higher) against the composition.

**Why the pattern fails.** The crystals with hydrogen have the highest frequencies, and X-rays show hydrogen weakly.  But hydrogen is not the whole reason.  For the crystals without hydrogen the X-ray pattern has an error of 247.9 cm^-1, the mean of the training part 273.8 and the composition 81.9.  For the crystals with hydrogen the three numbers are 1193.1, 1297.1 and 403.0.  The frequency of the highest phonon peak depends on which elements the crystal holds, most of all on their masses.  A powder pattern holds the geometry of the lattice and the contrast of the scattering strengths, and it does not name the elements.

**The pattern beside another description.** With the structural kernel the X-ray pattern gives 93.0 cm^-1, which is +0.3 [-1.2, +1.7] cm^-1 (not separated) against the structural kernel alone (with equal weights: +0.2 [-1.3, +1.6] cm^-1 (not separated)).  With the composition it gives 97.0, which is -0.6 [-2.5, +1.3] cm^-1 (not separated) against the composition alone.  The weights that maximise the marginal likelihood give the pattern a share of 0.00 beside the structure.

**The structure against the composition.** The structural kernel is -4.9 [-14.0, +3.5] cm^-1 (not separated) against a Gaussian kernel on the fractions of the elements.  On this task the composition carries most of what the structural kernel predicts.

**The co-spectrum.** The model on all X-ray patterns reproduces the targets it was fitted on with R2 +0.120.  Its largest effect is 206 cm^-1, against at most 144 in models fitted to targets in random order, and the intensity of no band of 10 degrees correlates with the target above 0.20 in size.  The weights of neighbouring grid points have opposite signs and cancel, because at this width neighbouring points hold nearly the same intensity.  The co-spectrum of a model that predicts this little cannot be read as a map.

**The gate of the item.** The item is a result when the structural kernel with the pattern has a lower error than the structural kernel alone by more than the interval, or when the pattern alone reaches the composition.  The first condition is not met.  The second condition is not met.  This is the outcome that the plan expected, and it is stronger than expected: a calculated powder pattern, without the types of the atoms, predicts this property little better than the mean does.  The kernel and the co-spectrum of `cmfqsar/spectrum.py` are tested on their own; this property is not one that a diffraction pattern shows.

