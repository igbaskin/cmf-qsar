# The surface kernel on the surface of a particle (plan item NT5)

## Checks

Differences are the largest difference of two rows of channels divided by the largest entry. Heights 2 to 12 A every 2 A, distances along the surface to 12 A every 1 A.

A flat core: a molecule on a gold slab whose top layer is at z = 0, the particle kernel with a plane against the flat surface kernel.

| Core | Molecule | Difference |
|---|---|---:|
| Au(111) | CH3CH2OH | 0.0e+00 |
| Au(111) | C6H6 | 0.0e+00 |
| Au(111) | CH3COOH | 0.0e+00 |
| Au(100) | CH3CH2OH | 0.0e+00 |
| Au(100) | C6H6 | 0.0e+00 |
| Au(100) | CH3COOH | 0.0e+00 |

Invariance: ethanol on a gold particle (sphere) and on a gold wire (cylinder), the whole system turned and moved. The flat kernel, which takes z as the height, is shown for the same pair.

| Core | Particle kernel | Flat surface kernel |
|---|---:|---:|
| sphere | 6.2e-16 | 1.1e+00 |
| cylinder | 8.2e-16 | 1.0e+00 |

Curvature: one flat patch of 40 atoms (heights 2.5 to 10 A, 12 x 12 A across) wrapped onto spheres of growing radius, against the flat kernel of the patch.

| Radius of the sphere, A | Difference |
|---:|---:|
| 10 | 1.2e-02 |
| 20 | 3.0e-03 |
| 50 | 4.8e-04 |
| 100 | 1.2e-04 |
| 200 | 2.9e-05 |
| 500 | 4.9e-06 |

| Core atoms | Shell atoms | Seconds for one shell |
|---:|---:|---:|
| 55 | 64 | 0.01 |
| 55 | 216 | 0.02 |
| 147 | 144 | 0.02 |
| 147 | 480 | 0.11 |
| 309 | 256 | 0.03 |
| 309 | 840 | 0.30 |

## The accessibility of the end groups of a ligand shell

216 generated shells on gold icosahedra of 55, 147, 309 atoms, 8 to 105 ligands, 32 to 840 shell atoms. Target: the mean accessibility of the end atoms to a 1.4 A probe, 0.572 to 0.669. 5-fold cross-validation repeated with 5 seeds (mean, and the standard deviation over the seeds). 'The recipe' is 1/R, the coverage, the chain length and the end group, the numbers each shell was built from.

| Model | RMSE | Standard deviation over seeds | q2 | Seconds for the kernel |
|---|---:|---:|---:|---:|
| particle surface kernel (sphere) | 0.0101 | 0.0004 | 0.706 | 8 |
| particle surface kernel (sphere), heights every 1 A, distances every 0.5 A | 0.0095 | 0.0004 | 0.738 | 15 |
| autocorrelation of the shell alone | 0.0081 | 0.0002 | 0.810 | 6 |
| autocorrelation of the core and the shell | 0.0069 | 0.0001 | 0.862 | 13 |
| Gaussian kernel on the recipe | 0.0075 | 0.0002 | 0.836 | -- |
| linear regression on the recipe | 0.0102 | 0.0001 | 0.701 | -- |
| the mean of the training part | 0.0187 | 0.0000 | -0.011 | -- |
