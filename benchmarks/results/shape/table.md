# Size and shape as descriptors (plan item NT3)

## The descriptors against answers known without them

Surface atoms are the atoms a probe sphere of 1.4 A can touch (the Shrake-Rupley method with 200 points per atom, atoms with their covalent radii).

Copper icosahedra of 2 to 12 shells: the surface atoms equal the atoms of the outer shell, 10 (k - 1)^2 + 2, in 11 of 11.

| Shells | Atoms | Surface atoms found | Atoms of the outer shell | Time, s |
|---:|---:|---:|---:|---:|
| 2 | 13 | 12 | 12 | 0.00 |
| 3 | 55 | 42 | 42 | 0.00 |
| 4 | 147 | 92 | 92 | 0.01 |
| 5 | 309 | 162 | 162 | 0.03 |
| 6 | 561 | 252 | 252 | 0.06 |
| 7 | 923 | 362 | 362 | 0.11 |
| 8 | 1,415 | 492 | 492 | 0.18 |
| 9 | 2,057 | 642 | 642 | 0.27 |
| 10 | 2,869 | 812 | 812 | 0.39 |
| 11 | 3,871 | 1,002 | 1,002 | 0.55 |
| 12 | 5,083 | 1,212 | 1,212 | 0.77 |

Other fcc particles: the surface atoms against the atoms with fewer than twelve nearest neighbours.

| Shape | Atoms | Surface atoms found | Atoms with fewer than 12 neighbours | Atoms classed differently |
|---|---:|---:|---:|---:|
| truncated octahedron | 79 | 60 | 60 | 0 |
| truncated octahedron | 459 | 234 | 234 | 0 |
| truncated octahedron | 1,289 | 482 | 482 | 0 |
| truncated octahedron | 2,951 | 876 | 876 | 0 |
| decahedron | 55 | 42 | 42 | 0 |
| decahedron | 309 | 162 | 162 | 0 |
| decahedron | 923 | 362 | 362 | 0 |

39 carbon nanotubes (n, m), 4 <= n <= 9: the diameter (the largest distance between two atom centres across the axis) differs from a sqrt(n^2 + n m + m^2) / pi by at most 1.8e-15 A, and twice the gyration radius by at most 1.8e-15 A.

Blocks of fcc copper of 3 x 3 x 3L cubic cells: aspect ratio against the ratio of the sides.

| Cells | Atoms | Aspect ratio | Ratio of the sides |
|---|---:|---:|---:|
| 3 x 3 x 3 | 108 | 1.000 | 1 |
| 3 x 3 x 6 | 216 | 1.994 | 2 |
| 3 x 3 x 9 | 324 | 2.990 | 3 |
| 3 x 3 x 12 | 432 | 3.985 | 4 |
| 3 x 3 x 18 | 648 | 5.977 | 6 |

| Structure | Atoms | Surface fraction | Diameter, A |
|---|---:|---:|---:|
| bulk copper (crystal) | 32 | 0.00 | none (NaN) |
| graphene sheet (layer) | 32 | 1.00 | 0.00 |

## Size and shape beside the periodic autocorrelation

150 unrelaxed particles of Cu, Ag, Au, Ni, Pd and Pt (13 to 3,564 atoms; the set of item NT2). The size and shape descriptors of all of them took 17.8 s. Kernel ridge regression on the model's own kernel, cutoff 8 A, the ridge parameter chosen by an inner 5-fold cross-validation and the descriptor statistics taken from each training part alone.

### Target: the energy per atom (EMT), 84.96 to 1171 meV per atom

5-fold cross-validation repeated with 5 seeds (mean, and the standard deviation over the seeds):

| Model | RMSE, meV per atom | Standard deviation over seeds, meV per atom | q2 |
|---|---:|---:|---:|
| periodic autocorrelation alone | 28.75 | 2 | 0.9830 |
| periodic autocorrelation and the five size and shape descriptors | 16.19 | 1.4 | 0.9946 |
| linear regression on the five descriptors, an intercept for each metal | 74.62 | 0.85 | 0.8862 |
| size law: linear regression on n^(-1/3) for each metal | 44.89 | 2.3 | 0.9587 |
| linear regression on the surface fraction for each metal | 90.51 | 2.3 | 0.8325 |

5-fold cross-validation that holds out each of the 25 geometries (shape and size) in all six metals at once, repeated with 5 seeds; the inner folds that choose the ridge parameter hold out geometries too:

| Model | RMSE, meV per atom | Standard deviation over seeds, meV per atom | q2 |
|---|---:|---:|---:|
| periodic autocorrelation alone | 74.75 | 95 | 0.7004 |
| periodic autocorrelation and the five size and shape descriptors | 53.85 | 77 | 0.8208 |
| linear regression on the five descriptors, an intercept for each metal | 89.57 | 7 | 0.8350 |
| size law: linear regression on n^(-1/3) for each metal | 45.85 | 1 | 0.9570 |
| linear regression on the surface fraction for each metal | 92.57 | 5.2 | 0.8243 |

The same folds one by one (25 folds: 5 seeds of 5 folds). 'Folds won' counts the folds in which the model's error is below that of the size law.

| Model | Median error of a fold, meV per atom | Largest error of a fold, meV per atom | Held out in that fold | Folds won |
|---|---:|---:|---|---:|
| periodic autocorrelation alone | 30.87 | 587 | deca2, deca4, deca6, ico2, octa8 | 20 of 25 |
| periodic autocorrelation and the five size and shape descriptors | 15.98 | 462 | deca2, deca4, deca6, ico2, octa8 | 24 of 25 |
| linear regression on the five descriptors, an intercept for each metal | 86.09 | 174.4 | deca2, deca4, deca6, ico2, octa8 | 0 of 25 |
| size law: linear regression on n^(-1/3) for each metal | 42.86 | 65.03 | deca3, ico2, ico3, ico5, ico9 | -- |
| linear regression on the surface fraction for each metal | 85.87 | 177.8 | deca2, deca4, deca6, ico2, octa8 | 0 of 25 |

Extrapolation in size: trained on the 96 particles of at most 1,000 atoms, tested on the 54 larger ones (q2 against the mean of the training part):

| Model | RMSE, meV per atom | Largest error, meV per atom | q2 |
|---|---:|---:|---:|
| periodic autocorrelation alone | 33.69 | 91.98 | 0.9816 |
| periodic autocorrelation and the five size and shape descriptors | 48.19 | 110.3 | 0.9623 |
| linear regression on the five descriptors, an intercept for each metal | 323.8 | 729.9 | -0.7023 |
| size law: linear regression on n^(-1/3) for each metal | 36.19 | 92.75 | 0.9787 |
| linear regression on the surface fraction for each metal | 96.07 | 213 | 0.8501 |

### Target: the energy of the particle (EMT), 7.972 to 708.3 eV

5-fold cross-validation repeated with 5 seeds (mean, and the standard deviation over the seeds):

| Model | RMSE, eV | Standard deviation over seeds, eV | q2 |
|---|---:|---:|---:|
| periodic autocorrelation alone | 111.2 | 5.8 | 0.4550 |
| periodic autocorrelation and the five size and shape descriptors | 45.31 | 2.1 | 0.9095 |
| linear regression on the five descriptors, an intercept for each metal | 46.16 | 1.8 | 0.9062 |
| liquid drop: linear regression on n and n^(2/3) for each metal | 63 | 3.1 | 0.8250 |

5-fold cross-validation that holds out each of the 25 geometries (shape and size) in all six metals at once, repeated with 5 seeds; the inner folds that choose the ridge parameter hold out geometries too:

| Model | RMSE, eV | Standard deviation over seeds, eV | q2 |
|---|---:|---:|---:|
| periodic autocorrelation alone | 138.3 | 65 | -0.0286 |
| periodic autocorrelation and the five size and shape descriptors | 60.1 | 12 | 0.8346 |
| linear regression on the five descriptors, an intercept for each metal | 53.23 | 2.1 | 0.8752 |
| liquid drop: linear regression on n and n^(2/3) for each metal | 59.77 | 3.8 | 0.8423 |

The same folds one by one (25 folds: 5 seeds of 5 folds). 'Folds won' counts the folds in which the model's error is below that of the liquid drop.

| Model | Median error of a fold, eV | Largest error of a fold, eV | Held out in that fold | Folds won |
|---|---:|---:|---|---:|
| periodic autocorrelation alone | 86.92 | 555.1 | deca2, deca4, deca6, ico2, octa8 | 0 of 25 |
| periodic autocorrelation and the five size and shape descriptors | 50.29 | 116.2 | deca2, deca4, deca6, ico2, octa8 | 18 of 25 |
| linear regression on the five descriptors, an intercept for each metal | 45.35 | 83.18 | deca3, ico8, ico9, octa18, octa4 | 14 of 25 |
| liquid drop: linear regression on n and n^(2/3) for each metal | 43.54 | 109 | deca9, ico10, ico5, ico6, ico9 | -- |

Extrapolation in size: trained on the 96 particles of at most 1,000 atoms, tested on the 54 larger ones (q2 against the mean of the training part):

| Model | RMSE, eV | Largest error, eV | q2 |
|---|---:|---:|---:|
| periodic autocorrelation alone | 196.1 | 461.5 | 0.4975 |
| periodic autocorrelation and the five size and shape descriptors | 118.3 | 309.8 | 0.8172 |
| linear regression on the five descriptors, an intercept for each metal | 169.4 | 431.9 | 0.6249 |
| liquid drop: linear regression on n and n^(2/3) for each metal | 85.73 | 228.2 | 0.9040 |
