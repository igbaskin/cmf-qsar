# Item M1 measured on a set of nanotubes (plan item NT4)

## Checks of the chain kernel on tubes

Differences are the largest difference of the periodic autocorrelation rows (cutoff 8 A) divided by the largest entry.

* One cell of the tube (8,4) (112 atoms) against three cells (336 atoms): 1.0e-14.
* The tube (10,0) given as a chain against the same tube in a 20 A box given as a crystal: 6.07. A tube read from a file that states three periodic directions needs `--pbc c`.

A tube against its mirror image. 'Pair channels' is the difference of the periodic autocorrelation rows. The handedness block of item M10 is a triple product of three different fields, so a tube of one element (one field) has none: its length is given. 'Across shells' is a prototype that takes the three vectors from three radial shells of the one field instead (0, 1.3, 2.7 and 4 A; the four channels a < b < c), with the largest difference of the tube's channels from minus its mirror image's.

| Tube | Atoms | Pair channels | Length of the M10 handedness block | Across shells, the four channels | Tube plus its mirror image |
|---|---:|---:|---:|---|---:|
| (8,4) | 112 | 0.0e+00 | 0 | 3.75e-06, 2.27e-06, -1.37e-04, -4.91e-03 | 0.0e+00 |
| (7,5) | 436 | 0.0e+00 | 0 | 2.41e-06, 1.34e-06, -8.81e-05, -3.09e-03 | 0.0e+00 |
| (10,3) | 556 | 0.0e+00 | 0 | 2.30e-06, 1.44e-06, -8.25e-05, -3.01e-03 | 0.0e+00 |
| (12,7) | 1108 | 0.0e+00 | 0 | 3.23e-07, 4.22e-08, -1.20e-05, -3.85e-04 | 0.0e+00 |
| (10,0) | 40 | 0.0e+00 | 0 | -2.73e-20, 4.12e-19, 3.21e-18, -3.81e-17 | 0.0e+00 |
| (6,6) | 24 | 0.0e+00 | 0 | 1.74e-19, -2.94e-19, -6.83e-18, -2.59e-17 | 0.0e+00 |

## What the distances see

147 carbon nanotubes (n, m), 12 to 988 atoms in the cell, diameters 0.41 to 2.71 nm; 75 have n - m divisible by 3. Their tight-binding gaps are 0.000 to 0.367 eV (from curvature) and those of the others 0.32 to 2.20 eV: the two groups overlap at their edges, because curvature opens a larger gap in the thinnest 'metallic' tubes than the widest semiconducting tubes have, so the class is the rule on the indices and not a cut through the gaps. The 1,000-atom limit keeps more of the tubes with n - m divisible by 3, whose translational cell is three times shorter. 5-fold cross-validation repeated with 5 seeds (mean, and the standard deviation over the seeds). References: Gaussian kernels on the diameter, and on the diameter and the chiral angle (both from the indices); their width is chosen with the ridge.

### Is the tube metallic?

| Model | Accuracy | Balanced accuracy | Standard deviation of the balanced accuracy over seeds |
|---|---:|---:|---:|
| periodic autocorrelation, cutoff 8 A | 0.660 | 0.660 | 0.017 |
| periodic autocorrelation, cutoff 12 A | 0.618 | 0.618 | 0.021 |
| three-body channels, radial grid to 4 A | 0.592 | 0.590 | 0.019 |
| Gaussian kernel on the diameter | 0.633 | 0.634 | 0.015 |
| Gaussian kernel on the diameter and the chiral angle | 0.578 | 0.577 | 0.057 |
| the mean of the training part | 0.435 | 0.433 | 0.030 |

Balanced accuracy is the mean of the fractions right in each class: 0.5 is no better than chance whatever the share of each class. The rule 'n - m divisible by 3' gives 1.000 from the indices.

### The gap (147 tubes), eV

| Model | RMSE, eV | Standard deviation over seeds, eV | q2 |
|---|---:|---:|---:|
| periodic autocorrelation, cutoff 8 A | 0.422 | 0.019 | 0.209 |
| periodic autocorrelation, cutoff 12 A | 0.427 | 0.016 | 0.194 |
| three-body channels, radial grid to 4 A | 0.415 | 0.011 | 0.237 |
| Gaussian kernel on the diameter | 0.394 | 0.017 | 0.311 |
| Gaussian kernel on the diameter and the chiral angle | 0.397 | 0.012 | 0.304 |
| the mean of the training part | 0.480 | 0.003 | -0.017 |

### The gap of the semiconducting tubes (72 tubes), eV

| Model | RMSE, eV | Standard deviation over seeds, eV | q2 |
|---|---:|---:|---:|
| periodic autocorrelation, cutoff 8 A | 0.128 | 0.015 | 0.885 |
| periodic autocorrelation, cutoff 12 A | 0.142 | 0.007 | 0.860 |
| three-body channels, radial grid to 4 A | 0.104 | 0.011 | 0.924 |
| Gaussian kernel on the diameter | 0.093 | 0.008 | 0.940 |
| Gaussian kernel on the diameter and the chiral angle | 0.117 | 0.007 | 0.905 |
| the mean of the training part | 0.384 | 0.003 | -0.023 |
| linear regression on 1/d | 0.082 | 0.005 | 0.953 |
