# C 1s binding energies of carbon atoms (plan item LP6)

Training: 4074 carbon atoms of 1000 computed molecules of the AugerNet release (5 folds, whole molecules held out). Test: 570 measured values in 113 molecules, split by the release into validation and evaluation molecules. Standard deviation of the values: 1.62 eV computed, 2.57 eV measured.

Mean absolute error in eV. "Small" and "large" are the measured molecules with at most 9 and with more than 9 heavy atoms; the computed molecules have at most 9.

| Model | Computed set, cross-validation | Validation molecules | Evaluation molecules | Small molecules | Large molecules | Setting |
|---|---|---|---|---|---|---|
| mean of the training set | 1.331 | 2.291 | 2.241 | 2.093 | 2.444 |  |
| lookup by bonded environment, 4 bonds | 0.550 | 0.876 | 1.074 | 0.912 | 1.040 | measured atoms read at depth: {-1: 12, 0: 167, 1: 306, 2: 83, 3: 2} |
| linear regression on Gasteiger charge and potential | 0.632 | 0.802 | 0.740 | 0.638 | 0.909 | E = 291.49 +10.77 q -0.14 V |
| Gaussian kernel on Gasteiger charge and potential | 0.591 | 0.649 | 0.593 | 0.484 | 0.762 | width 0.5 x median; ridge 0.1 |
| linear regression on xtb charge and potential | 0.331 | 0.473 | 0.451 | 0.384 | 0.542 | E = 291.31 +23.92 q +11.36 V |
| Gaussian kernel on xtb charge and potential | 0.305 | 0.503 | 0.511 | 0.360 | 0.658 | width 0.5 x median; ridge 0.001 |
| atom-centred counts, linear | 0.350 | 0.506 | 0.500 | 0.505 | 0.501 | 15 atom types in 6 shells; ridge 0.1 |
| atom-centred counts, Gaussian kernel | 0.238 | 0.592 | 0.695 | 0.526 | 0.765 | width 4 x median; ridge 0.0001 |
| counts with xtb charge and potential, linear | 0.248 | 0.546 | 0.671 | 0.501 | 0.720 | ridge 0.0001 |
| counts with xtb charge and potential, Gaussian | 0.117 | 0.397 | 0.708 | 0.329 | 0.782 | width 8 x median; ridge 1e-05 |
| counts to 1.8 A, linear | 0.553 | 0.670 | 0.702 | 0.605 | 0.769 | ridge 1e-09 |
| counts to 1.8 A, Gaussian kernel | 0.484 | 0.598 | 0.638 | 0.508 | 0.732 | width 2 x median; ridge 0.01 |
| counts to 1.8 A, with xtb charge and potential, Gaussian | 0.175 | 0.319 | 0.302 | 0.291 | 0.331 | width 1 x median; ridge 0.001 |
| counts to 2.8 A, linear | 0.484 | 0.610 | 0.665 | 0.580 | 0.697 | ridge 0.01 |
| counts to 2.8 A, Gaussian kernel | 0.379 | 0.542 | 0.541 | 0.418 | 0.668 | width 2 x median; ridge 0.001 |
| counts to 2.8 A, with xtb charge and potential, Gaussian | 0.139 | 0.320 | 0.344 | 0.347 | 0.316 | width 2 x median; ridge 0.001 |
| counts to 3.8 A, linear | 0.418 | 0.489 | 0.602 | 0.524 | 0.568 | ridge 0.01 |
| counts to 3.8 A, Gaussian kernel | 0.297 | 0.549 | 0.495 | 0.421 | 0.626 | width 2 x median; ridge 0.001 |
| counts to 3.8 A, with xtb charge and potential, Gaussian | 0.122 | 0.424 | 0.404 | 0.360 | 0.469 | width 2 x median; ridge 0.0001 |
| local field, rho 0.25, linear | 0.454 | 0.841 | 0.760 | 0.740 | 0.863 | 255 channels, 11.8 atoms per environment, features 7 s; ridge 0.0001 |
| local field, rho 0.25, Gaussian kernel | 0.312 | 0.746 | 0.883 | 0.512 | 1.126 | width 1 x median; ridge 0.01 |
| local field, rho 0.25, with xtb charge and potential, Gaussian | 0.156 | 0.395 | 0.576 | 0.326 | 0.649 | width 2 x median; ridge 0.001 |
| local field, rho 0.5, linear | 0.381 | 0.664 | 0.609 | 0.540 | 0.735 | 255 channels, 10.8 atoms per environment, features 6 s; ridge 1e-06 |
| local field, rho 0.5, Gaussian kernel | 0.306 | 0.523 | 0.693 | 0.468 | 0.752 | width 1 x median; ridge 0.01 |
| local field, rho 0.5, with xtb charge and potential, Gaussian | 0.162 | 0.325 | 0.419 | 0.328 | 0.417 | width 2 x median; ridge 0.001 |
| local field, rho 1, linear | 0.371 | 0.553 | 0.559 | 0.534 | 0.579 | 255 channels, 8.2 atoms per environment, features 6 s; ridge 1e-08 |
| local field, rho 1, Gaussian kernel | 0.319 | 0.422 | 0.465 | 0.423 | 0.464 | width 1 x median; ridge 0.01 |
| local field, rho 1, with xtb charge and potential, Gaussian | 0.158 | 0.280 | 0.303 | 0.310 | 0.273 | width 4 x median; ridge 0.0001 |
| local field, rho 2, linear | 0.397 | 0.628 | 0.614 | 0.563 | 0.681 | 255 channels, 4.6 atoms per environment, features 5 s; ridge 1e-09 |
| local field, rho 2, Gaussian kernel | 0.358 | 0.444 | 0.478 | 0.440 | 0.483 | width 1 x median; ridge 0.01 |
| local field, rho 2, with xtb charge and potential, Gaussian | 0.169 | 0.317 | 0.322 | 0.302 | 0.338 | width 4 x median; ridge 0.0001 |
| local field, rho 4, linear | 0.437 | 0.644 | 0.552 | 0.537 | 0.661 | 255 channels, 3.8 atoms per environment, features 5 s; ridge 1e-09 |
| local field, rho 4, Gaussian kernel | 0.338 | 0.469 | 0.488 | 0.480 | 0.476 | width 0.5 x median; ridge 0.01 |
| local field, rho 4, with xtb charge and potential, Gaussian | 0.157 | 0.333 | 0.335 | 0.365 | 0.302 | width 1 x median; ridge 0.001 |

Field model chosen on the validation molecules: local field, rho 1, with xtb charge and potential, Gaussian. Reference chosen on the validation molecules: counts to 1.8 A, with xtb charge and potential, Gaussian.

Field model minus reference, mean absolute error in eV with a 95 % interval over molecules: +0.001 [-0.047, +0.058] on the evaluation molecules, -0.019 [-0.058, +0.023] on all measured atoms.

Published network on the same split (arXiv 2604.27070, all 2,116 computed molecules): 0.27 eV on the validation and 0.33 eV on the evaluation molecules.
