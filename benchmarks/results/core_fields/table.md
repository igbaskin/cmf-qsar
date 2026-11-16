# Fields for the core of a particle (plan item NT6)

## Checks

* Coordination numbers of the 150 particles of item NT2 (141,618 atoms) against ASE's neighbour list with a cutoff halfway between the first and second shells: 0 atoms differ.
* Atoms with an exposure above zero against atoms with fewer than twelve neighbours: 0 of 141,618 differ.

| Atoms | Coordination number | Generalized coordination number, computed | Published value |
|---|---:|---:|---:|
| Pt fcc(111) terrace, top layer | 9 | 7.500 (spread 0.0e+00) | 7.500 |
| Pt fcc(100) terrace, top layer | 8 | 6.667 (spread 0.0e+00) | 6.667 |
| Bulk Pt | 12 | 12.000 (spread 0.0e+00) | 12.000 |

| Particle | Atoms | Seconds for the three core fields |
|---|---:|---:|
| Cu_ico2 | 13 | 0.00 |
| Cu_octa4 | 38 | 0.01 |
| Ni_ico3 | 55 | 0.01 |
| Cu_ico4 | 147 | 0.05 |
| Cu_ico5 | 309 | 0.12 |
| Cu_octa8 | 314 | 0.15 |
| Ni_ico6 | 561 | 0.33 |
| Cu_ico7 | 923 | 0.64 |
| Cu_octa12 | 1,072 | 0.55 |
| Ni_ico8 | 1,415 | 0.86 |
| Cu_ico9 | 2,057 | 1.23 |
| Cu_octa16 | 2,406 | 1.49 |
| Cu_octa18 | 3,564 | 2.07 |
| Cu_octa18 | 3,564 | 2.04 |

## Particles of one metal (the 150 particles of item NT2)

### Energy per atom

| Model | Random folds: RMSE, meV per atom | Random folds: q2 | Geometry held out: RMSE, meV per atom | Geometry held out: q2 | Larger particles: RMSE, meV per atom |
|---|---:|---:|---:|---:|---:|
| periodic autocorrelation, fields of NT3 | 28.7 | 0.983 | 74.7 | 0.700 | 33.7 |
| periodic autocorrelation, fields of NT3 and the three core fields | 17.2 | 0.994 | 33.1 | 0.960 | 4.4 |
| linear regression on the neighbour pairs in four shells, for each metal | 33.9 | 0.976 | 31.0 | 0.980 | 13.8 |

With geometries held out the core fields lower the error in 22 of 25 folds; worst fold 587.0 without them, 206.6 with them (meV per atom).

### Energy of the particle

| Model | Random folds: RMSE, eV | Random folds: q2 | Geometry held out: RMSE, eV | Geometry held out: q2 | Larger particles: RMSE, eV |
|---|---:|---:|---:|---:|---:|
| periodic autocorrelation, fields of NT3 | 111.2 | 0.455 | 138.3 | -0.029 | 196.1 |
| periodic autocorrelation, fields of NT3 and the three core fields | 64.9 | 0.811 | 170.5 | -2.057 | 151.7 |
| linear regression on the neighbour pairs in four shells, for each metal | 11.5 | 0.994 | 9.6 | 0.996 | 26.4 |

With geometries held out the core fields lower the error in 23 of 25 folds; worst fold 555.1 without them, 1268.6 with them (eV).

## Particles of two metals: the mixing energy

360 particles of 55 to 314 atoms; mixing energy -54 to 176 meV per atom (standard deviation 54).

| Arrangement | Mean mixing energy, meV per atom | Standard deviation |
|---|---:|---:|
| core | 49.5 | 74.1 |
| surface | 26.2 | 60.2 |
| random0 | 23.7 | 30.6 |
| random1 | 21.7 | 30.4 |

| Model | Random folds: RMSE, meV per atom | Random folds: q2 | One geometry held out: RMSE, meV per atom | One geometry held out: q2 |
|---|---:|---:|---:|---:|
| periodic autocorrelation, steric, electronegativity and one indicator per metal | 11.2 | 0.956 | 16.2 | 0.909 |
| the same and the three core fields | 4.6 | 0.993 | 13.2 | 0.939 |
| the same and the coordination number alone | 8.4 | 0.975 | 13.5 | 0.937 |
| the same and the generalized coordination number alone | 7.8 | 0.979 | 13.2 | 0.939 |
| the same and the exposure alone | 5.7 | 0.989 | 13.3 | 0.938 |
| bond counting: linear regression on AA, AB, BB pairs and surface atoms of each metal, for each pair | 23.4 | 0.810 | 24.1 | 0.798 |
| bond counting with coordination classes: linear regression on AA, AB, BB pairs and the atoms of each metal with each coordination number, for each pair | 22.4 | 0.826 | 22.5 | 0.823 |
| Gaussian kernel on the same counts and the composition, for each pair | 6.8 | 0.984 | 12.2 | 0.948 |
| composition alone: linear regression on x and x^2, for each pair | 50.6 | 0.107 | 48.9 | 0.167 |

| Geometry held out | Without the core fields, meV per atom | With them, meV per atom | Gaussian kernel on the counts, meV per atom |
|---|---:|---:|---:|
| deca3 | 16.9 | 21.6 | 31.5 |
| deca4 | 11.6 | 8.5 | 7.8 |
| deca5 | 14.2 | 8.4 | 7.8 |
| ico3 | 35.3 | 21.9 | 10.8 |
| ico4 | 9.4 | 12.1 | 7.3 |
| ico5 | 15.0 | 13.6 | 10.7 |
| octa5 | 14.1 | 10.1 | 5.0 |
| octa6 | 8.1 | 6.9 | 2.9 |
| octa7 | 7.3 | 6.6 | 5.7 |
| octa8 | 10.6 | 11.8 | 4.2 |

The core fields lower the error in 7 of 10 held-out geometries; the Gaussian kernel on the counts has a lower error than the core fields in 9 of them.

The inner cross-validation chose widths [0.03, 0.1, 0.3] from the grid [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0] and ridge parameters [0.0001, 0.001, 0.01] from [1e-05, 0.0001, 0.001, 0.01, 0.1]: none at an edge of its grid.
