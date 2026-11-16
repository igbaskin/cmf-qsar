# Site-resolved models on particles (plan item NT7)

## The data

48 particles, 3070 surface atoms, 1851 distinct sites where the OH is bound atop. 239 more distinct sites are left out because the OH is not bound atop them (its own atom is not the nearest to the O, or is not between 1.6 and 2.3 A away). By the number of neighbours of the site: 1 with 8, 5 with 9, 53 with 10, 180 with 11. These are mostly atoms beside a removed atom, whose normal points into the hole.

| Metal | Sites | Binding energy, eV (lowest to highest) | Standard deviation, eV | Slope on the generalized coordination number, eV | Correlation |
|---|---:|---:|---:|---:|---:|
| Au | 915 | -0.84 to 0.08 | 0.211 | +0.150 | 0.914 |
| Pt | 936 | -1.26 to 0.75 | 0.336 | +0.219 | 0.835 |

## Whole particles held out (5 folds, 5 seeds)

| Model | RMSE, eV | q2 | q2 within Au | q2 within Pt | Strongest site found | Mean excess of the chosen site, eV |
|---|---:|---:|---:|---:|---:|---:|
| site profile, steric, electronegativity and one indicator per metal | 0.150 | 0.744 | 0.785 | 0.688 | 39 of 48 | 0.034 |
| the same and the three core fields of NT6 | 0.079 | 0.929 | 0.929 | 0.918 | 35 of 48 | 0.041 |
| linear regression on the generalized coordination number, for each metal | 0.152 | 0.736 | 0.821 | 0.662 | 37 of 48 | 0.041 |
| linear regression on the coordination number, for each metal | 0.133 | 0.800 | 0.873 | 0.741 | 37 of 48 | 0.038 |
| linear regression on the atoms in four distance shells, for each metal | 0.122 | 0.830 | 0.831 | 0.803 | 31 of 48 | 0.056 |
| Gaussian kernel on the shell counts and both coordination numbers, for each metal | 0.093 | 0.902 | 0.940 | 0.872 | 37 of 48 | 0.040 |

## Each of the eight shapes held out

| Model | RMSE, eV | q2 | q2 within Au | q2 within Pt | Strongest site found | Mean excess of the chosen site, eV |
|---|---:|---:|---:|---:|---:|---:|
| site profile, steric, electronegativity and one indicator per metal | 0.147 | 0.754 | 0.777 | 0.707 | 37 of 48 | 0.037 |
| the same and the three core fields of NT6 | 0.084 | 0.920 | 0.925 | 0.906 | 36 of 48 | 0.026 |
| linear regression on the generalized coordination number, for each metal | 0.159 | 0.711 | 0.812 | 0.627 | 37 of 48 | 0.041 |
| linear regression on the coordination number, for each metal | 0.139 | 0.781 | 0.863 | 0.716 | 37 of 48 | 0.038 |
| linear regression on the atoms in four distance shells, for each metal | 0.132 | 0.802 | 0.810 | 0.768 | 33 of 48 | 0.048 |
| Gaussian kernel on the shell counts and both coordination numbers, for each metal | 0.113 | 0.855 | 0.890 | 0.819 | 34 of 48 | 0.044 |

A site is found when its computed binding is within 0.05 eV of the strongest site of its particle. The q2 within one metal is taken against the mean of that metal.

Ridge parameters the inner cross-validation chose for the site profiles: 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1 (grid 1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1). Width and ridge parameter chosen for the Gaussian kernel on the counts: 0.0003/1e-09, 0.001/1e-09, 0.003/1e-09, 0.003/1e-05, 0.01/1e-09, 0.01/1e-08, 0.03/0.001, 0.03/0.01, 0.1/1e-06, 0.1/1e-05, 0.3/0.0001, 0.3/0.1, 1/0.01, 1/0.1, 3/0.01, 10/0.1 (grids 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 and 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1).

## A whole-particle model read site by site

Target: the mean binding energy over the surface atoms of each of 48 particles (-0.740 to -0.223 eV, standard deviation 0.129). Periodic autocorrelation with the core fields, q2 0.921 in 5-fold cross-validation, ridge parameter 1e-05 (grid 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01).

| Quantity | Value |
|---|---:|
| Median share of the contributions on surface atoms | 0.721 |
| Median share of the atoms that are surface atoms | 0.748 |
| Median rank correlation of contribution and binding over the surface atoms | -0.762 |
| Particles where that correlation is positive | 7 of 48 |
| Median rank correlation of contribution and coordination number over the surface atoms | -0.718 |
| Median rank correlation of coordination number and binding over the surface atoms | +0.960 |
| Largest error of the sum of the contributions | 1.5e-12 |
