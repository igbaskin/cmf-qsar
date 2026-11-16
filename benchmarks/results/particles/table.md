# Particles of thousands of atoms (plan item NT2)

## How the cost grows with the number of atoms

Copper icosahedra with no cell; carbon nanotubes (6, 6) of growing length ('long tube'); and carbon nanotubes (n, n) of ten repeat units and growing diameter ('wide tube'). A tube is periodic along its axis and has no lattice vector across it, as ASE writes it. Cutoff 8 A, fields steric, prop:electronegativity. Times are the best of three runs (one for the overlap kernel), in seconds. 'ASE' is the neighbour list used before this item; 'k-d tree' is the route such structures take now. 'Same pairs' says whether both give the same atom pairs; the distances then agree to the 'Largest distance difference'. The overlap kernel is that of the structure with a copy of itself whose atoms are moved by up to 0.05 A, open boundaries. 'Pairs alone' is the autocorrelation of a tube without its background term, the uniform density spread along the axis. Peak memory is what Python and NumPy allocate. The old route was not run beyond 5,083 atoms, because its memory grows with the square of the number of atoms.

| Structure | Atoms | ASE neighbour list, s | ASE peak memory, MB | k-d tree, s | Same pairs | Largest distance difference, A | Periodic autocorrelation by the new route, s | Its peak memory, MB | Pairs alone, s | Overlap kernel of two structures, s |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| copper icosahedron | 55 | 0.005 | 1 | 0.000 | yes | 0e+00 | 0.001 | 1 | -- | 0.102 |
| copper icosahedron | 147 | 0.015 | 4 | 0.001 | yes | 0e+00 | 0.004 | 3 | -- | 0.000 |
| copper icosahedron | 309 | 0.051 | 20 | 0.003 | yes | 0e+00 | 0.012 | 8 | -- | 0.002 |
| copper icosahedron | 561 | 0.142 | 65 | 0.006 | yes | 0e+00 | 0.029 | 17 | -- | 0.008 |
| copper icosahedron | 923 | 0.376 | 177 | 0.014 | yes | 0e+00 | 0.061 | 30 | -- | 0.021 |
| copper icosahedron | 1,415 | 0.941 | 416 | 0.023 | yes | 0e+00 | 0.105 | 49 | -- | 0.091 |
| copper icosahedron | 2,057 | 1.963 | 880 | 0.037 | yes | 0e+00 | 0.151 | 75 | -- | 0.173 |
| copper icosahedron | 2,869 | 3.762 | 1711 | 0.063 | yes | 0e+00 | 0.263 | 107 | -- | 0.332 |
| copper icosahedron | 5,083 | 11.6 | 5371 | 0.114 | yes | 0e+00 | 0.478 | 199 | -- | -- |
| copper icosahedron | 8,217 | -- | -- | 0.232 | -- | -- | 0.836 | 332 | -- | -- |
| long tube (6, 6) | 96 | 0.016 | 5 | 0.001 | yes | 2e-15 | 0.007 | 3 | 0.004 | -- |
| long tube (6, 6) | 192 | 0.018 | 9 | 0.002 | yes | 4e-15 | 0.019 | 6 | 0.007 | -- |
| long tube (6, 6) | 384 | 0.029 | 19 | 0.004 | yes | 7e-15 | 0.070 | 18 | 0.017 | -- |
| long tube (6, 6) | 720 | 0.037 | 29 | 0.008 | yes | 1e-14 | 0.224 | 50 | 0.038 | -- |
| long tube (6, 6) | 1,200 | 0.058 | 48 | 0.013 | yes | 1e-14 | 0.612 | 123 | 0.067 | -- |
| long tube (6, 6) | 1,800 | 0.096 | 71 | 0.020 | yes | 3e-14 | 1.286 | 259 | 0.087 | -- |
| long tube (6, 6) | 2,400 | 0.124 | 97 | 0.024 | yes | 3e-14 | 2.223 | 152 | 0.127 | -- |
| long tube (6, 6) | 3,600 | 0.175 | 142 | 0.039 | yes | 3e-14 | 4.973 | 228 | 0.230 | -- |
| long tube (6, 6) | 6,000 | -- | -- | 0.066 | -- | -- | 13.5 | 380 | 0.377 | -- |
| wide tube (n, n) | 240 | 0.018 | 10 | 0.002 | yes | 4e-15 | 0.029 | 9 | 0.009 | -- |
| wide tube (n, n) | 480 | 0.064 | 39 | 0.004 | yes | 4e-15 | 0.063 | 24 | 0.018 | -- |
| wide tube (n, n) | 800 | 0.188 | 109 | 0.008 | yes | 4e-15 | 0.140 | 57 | 0.036 | -- |
| wide tube (n, n) | 1,200 | 0.424 | 244 | 0.012 | yes | 4e-15 | 0.268 | 119 | 0.055 | -- |
| wide tube (n, n) | 1,800 | 0.953 | 550 | 0.020 | yes | 9e-16 | 0.471 | 252 | 0.082 | -- |
| wide tube (n, n) | 2,400 | 1.665 | 978 | 0.025 | yes | 4e-15 | 0.703 | 143 | 0.112 | -- |
| wide tube (n, n) | 3,600 | 3.843 | 2200 | 0.040 | yes | 9e-16 | 1.467 | 215 | 0.204 | -- |

Exponent of the growth with the number of atoms (slope of the logarithm of the time against the logarithm of the atom count, structures of 500 atoms or more): 1 is linear, 2 quadratic.

| Structure | ASE neighbour list | ASE peak memory | k-d tree | Periodic autocorrelation | Its peak memory | Pairs alone | Overlap kernel |
|---|---:|---:|---:|---:|---:|---:|---:|
| copper icosahedra | 2.01 | 2.00 | 1.31 | 1.24 | 1.10 | -- | 2.38 |
| long tubes (6, 6) | 0.99 | 0.99 | 0.99 | 1.93 | 0.83 | 1.10 | -- |
| wide tubes (n, n) | 2.00 | 2.00 | 1.08 | 1.53 | 0.80 | 1.14 | -- |

For a set of 100 particles of 2,869 atoms, the measured times give a training kernel of 28 minutes for the overlap kernel at one field width (5,050 pairs of structures) and 26 s for the periodic autocorrelation of all 100. The model therefore refuses the overlap kernel for a structure of more than 2,000 atoms (`max_overlap_atoms`; None turns the guard off). Molecules are never refused.

## A property of metal particles

150 particles of Cu, Ag, Au, Ni, Pd and Pt (icosahedra, decahedra and truncated octahedra, 13 to 3,564 atoms, 141,618 atoms in all), unrelaxed. Target: the energy per atom from ASE's effective medium theory potential (EMT), 0.08 to 1.17 eV. The periodic autocorrelation of all particles took 14.6 s. 'Size law' is a linear regression on n^(-1/3) (n the number of atoms) with its own intercept and slope for each metal; 'coordination law' the same on the mean number of nearest neighbours of an atom (5.69 to 11.05 here, 12 in the bulk); 'element mean' is the mean of the training particles of the same metal. The autocorrelation model is kernel ridge regression on the cosine-normalised autocorrelation, its ridge parameter chosen by an inner 5-fold cross-validation in each training part.

5-fold cross-validation repeated with 5 seeds (mean, and the standard deviation over the seeds):

| Model | RMSE, meV per atom | Standard deviation over seeds, meV | q2 |
|---|---:|---:|---:|
| element mean | 222.7 | 2.8 | -0.0139 |
| size law | 44.9 | 2.3 | 0.9587 |
| coordination law | 70.1 | 3.5 | 0.8994 |
| autocorrelation | 26.3 | 2.0 | 0.9857 |

Extrapolation in size: trained on the 96 particles of at most 1,000 atoms, tested on the 54 larger ones (q2 against the mean of the training part):

| Model | RMSE, meV per atom | Largest error, meV per atom | q2 |
|---|---:|---:|---:|
| element mean | 251.0 | 396.9 | -0.0232 |
| size law | 36.2 | 92.7 | 0.9787 |
| coordination law | 44.6 | 100.2 | 0.9678 |
| autocorrelation | 26.1 | 76.0 | 0.9889 |

### References that count neighbours

The same particles, folds and inner choice of the ridge. 'Shell counts' is a linear regression on the number of neighbour pairs per atom in each of the first 4 shells of the bulk lattice, with its own coefficients for each metal (a linear kernel within a metal, zero between metals). 'Coordination histogram' is the same on the fraction of atoms with each coordination number from 0 to 12. Both need no field. 'Difference' is the RMSE of the reference minus that of the autocorrelation, with a 95 % bootstrap interval over the particles; negative means the reference is better. In the cross-validation the predictions are averaged over the seeds first.

| Model | RMSE, meV per atom (cross-validation) | Difference | RMSE, meV per atom (extrapolation) | Largest error, meV | Difference |
|---|---:|---:|---:|---:|---:|
| autocorrelation | 26.3 |  | 26.1 | 76.0 |  |
| shell counts | 29.2 | +2.4 [-8.3, +13.0] | 15.0 | 35.9 | -11.1 [-17.4, -4.8] |
| coordination histogram | 95.4 | +61.9 [+39.5, +83.3] | 31.8 | 65.0 | +5.7 [-1.6, +13.1] |
