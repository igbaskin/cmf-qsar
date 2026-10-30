# Judging an alignment against crystal poses

Every ligand of these FEP sets is co-crystallised in one frame, so the pose a
method should have found is known. A model is fitted on the training molecules in
their crystal poses; the held-out molecules then have their poses destroyed, and
each method recovers them without seeing a held-out activity. `crystal` is the
ceiling, `scrambled` the floor, `recovery` the fraction of the pose error a method
removed, and the conformer limit the RMSD the best possible *rigid* placement of
the starting conformation would still leave.

Fields: `electrostatic`, `steric`, `hydrophobic`, `hba`, `hbd`; alpha 0.5, lambda 0.1, 30% held out.

## crystal conformations, pose scrambled, no displacement guard; field-point graph alignment

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | scrambled | mcs | field | fieldgraph | fieldgraph+field | fieldgraph+variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| syk | 44 | 0.00 | 5.95 | 0.05 | 0.60 | 0.73 | 0.60 | 0.13 |
| hif2a | 37 | 0.00 | 4.03 | 0.06 | 0.20 | 0.19 | 0.20 | 0.10 |
| bace | 36 | 0.00 | 5.04 | 0.21 | 0.15 | 0.45 | 0.16 | 0.05 |
| pde10 | 35 | 0.00 | 6.17 | 1.12 | 0.63 | 0.98 | 0.76 | 0.11 |
| bace_hunt | 32 | 0.00 | 4.29 | 0.00 | 0.08 | 0.08 | 0.08 | 0.05 |
| pfkfb3 | 32 | 0.00 | 5.44 | 0.05 | 0.22 | 0.16 | 0.22 | 0.10 |
| cdk8 | 31 | 0.00 | 4.90 | 0.12 | 0.42 | 0.74 | 0.42 | 0.10 |
| p38 | 29 | 0.00 | 5.03 | 0.62 | 0.53 | 0.77 | 0.53 | 0.09 |
| eg5 | 27 | 0.00 | 5.40 | 0.10 | 1.73 | 3.54 | 3.11 | 0.28 |
| tnks2 | 27 | 0.00 | 5.45 | 0.11 | 0.52 | 0.52 | 0.52 | 0.07 |
| mcl1 | 25 | 0.00 | 5.03 | 0.21 | 0.36 | 0.35 | 0.36 | 0.17 |
| shp2 | 24 | 0.00 | 5.22 | 0.70 | 1.02 | 0.83 | 1.01 | 0.21 |
| ptp1b | 22 | 0.00 | 7.74 | 0.10 | 0.18 | 0.27 | 0.18 | 0.10 |
| thrombin | 22 | 0.00 | 4.49 | 0.04 | 0.43 | 0.40 | 0.43 | 0.10 |
| jnk1 | 21 | 0.00 | 5.35 | 0.36 | 0.31 | 0.41 | 0.31 | 0.10 |
| pde2 | 21 | 0.00 | 5.08 | 0.20 | 0.29 | 0.24 | 0.29 | 0.12 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | scrambled | mcs | field | fieldgraph | fieldgraph+field | fieldgraph+variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| syk | 44 | -0.204 | +0.207 | -0.191 | -0.434 | -0.152 | -0.434 | -0.363 |
| hif2a | 37 | +0.477 | -0.183 | +0.476 | +0.571 | +0.097 | +0.192 | +0.177 |
| bace | 36 | +0.274 | -1.669 | +0.254 | +0.258 | +0.329 | +0.258 | +0.256 |
| pde10 | 35 | -0.641 | -1.926 | -0.412 | -0.272 | -0.084 | -0.074 | -0.236 |
| bace_hunt | 32 | +0.406 | -0.678 | +0.428 | +0.352 | +0.411 | +0.352 | +0.374 |
| pfkfb3 | 32 | +0.579 | -1.376 | +0.561 | +0.523 | +0.543 | +0.523 | +0.584 |
| cdk8 | 31 | +0.281 | -0.796 | +0.287 | +0.514 | +0.088 | +0.514 | +0.268 |
| p38 | 29 | +0.105 | -0.646 | -0.001 | -0.048 | -0.013 | -0.048 | +0.105 |
| eg5 | 27 | +0.622 | -0.186 | +0.607 | +0.103 | -0.523 | -0.037 | +0.430 |
| tnks2 | 27 | +0.291 | -0.862 | +0.216 | +0.253 | +0.209 | +0.253 | +0.212 |
| mcl1 | 25 | +0.544 | -0.491 | +0.298 | +0.132 | -0.128 | +0.133 | +0.387 |
| shp2 | 24 | +0.329 | -0.060 | +0.424 | +0.484 | +0.493 | +0.484 | +0.301 |
| ptp1b | 22 | +0.820 | -0.155 | +0.863 | +0.827 | +0.634 | +0.827 | +0.786 |
| thrombin | 22 | +0.401 | -1.861 | +0.401 | +0.474 | +0.352 | +0.474 | +0.440 |
| jnk1 | 21 | +0.625 | -0.132 | +0.578 | +0.573 | +0.614 | +0.573 | +0.625 |
| pde2 | 21 | +0.434 | -0.110 | +0.480 | +0.377 | +0.432 | +0.377 | +0.442 |

### Mean over the sets

| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | conformer limit |
|---|---:|---:|---:|---:|---:|---:|
| crystal | 0.00 A | 100% | 100% | - | +0.334 | 0.00 A |
| scrambled | 5.29 A | 0% | 0% | +0% | -0.683 | 0.00 A |
| mcs | 0.25 A | 94% | 98% | +95% | +0.329 | 0.00 A |
| field | 0.48 A | 89% | 97% | +91% | +0.293 | 0.00 A |
| fieldgraph | 0.67 A | 79% | 94% | +88% | +0.206 | 0.00 A |
| fieldgraph+field | 0.57 A | 85% | 93% | +89% | +0.273 | 0.00 A |
| fieldgraph+variance | 0.12 A | 96% | 96% | +98% | +0.299 | 0.00 A |

## crystal conformations, pose scrambled, no displacement guard

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | scrambled | mcs | field | variance | mcs+variance |
|---|---:|---:|---:|---:|---:|---:|---:|
| syk | 44 | 0.00 | 5.95 | 0.05 | 0.60 | 6.26 | 0.13 |
| hif2a | 37 | 0.00 | 4.03 | 0.06 | 0.20 | 0.12 | 0.10 |
| bace | 36 | 0.00 | 5.04 | 0.21 | 0.15 | 0.05 | 0.05 |
| pde10 | 35 | 0.00 | 6.17 | 1.12 | 0.63 | 1.48 | 0.11 |
| bace_hunt | 32 | 0.00 | 4.29 | 0.00 | 0.08 | 0.05 | 0.05 |
| pfkfb3 | 32 | 0.00 | 5.44 | 0.05 | 0.22 | 0.11 | 0.10 |
| cdk8 | 31 | 0.00 | 4.90 | 0.12 | 0.42 | 0.16 | 0.10 |
| p38 | 29 | 0.00 | 5.03 | 0.62 | 0.53 | 0.32 | 0.09 |
| eg5 | 27 | 0.00 | 5.40 | 0.10 | 1.73 | 0.20 | 0.14 |
| tnks2 | 27 | 0.00 | 5.45 | 0.11 | 0.52 | 0.21 | 0.07 |
| mcl1 | 25 | 0.00 | 5.03 | 0.21 | 0.36 | 0.29 | 0.17 |
| shp2 | 24 | 0.00 | 5.22 | 0.70 | 1.02 | 0.67 | 0.21 |
| ptp1b | 22 | 0.00 | 7.74 | 0.10 | 0.18 | 0.12 | 0.10 |
| thrombin | 22 | 0.00 | 4.49 | 0.04 | 0.43 | 5.00 | 0.10 |
| jnk1 | 21 | 0.00 | 5.35 | 0.36 | 0.31 | 3.49 | 0.10 |
| pde2 | 21 | 0.00 | 5.08 | 0.20 | 0.29 | 3.14 | 0.12 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | scrambled | mcs | field | variance | mcs+variance |
|---|---:|---:|---:|---:|---:|---:|---:|
| syk | 44 | -0.204 | +0.207 | -0.191 | -0.434 | -0.067 | -0.363 |
| hif2a | 37 | +0.477 | -0.183 | +0.476 | +0.571 | +0.644 | +0.578 |
| bace | 36 | +0.274 | -1.669 | +0.254 | +0.258 | +0.256 | +0.256 |
| pde10 | 35 | -0.641 | -1.926 | -0.412 | -0.272 | -0.782 | -0.609 |
| bace_hunt | 32 | +0.406 | -0.678 | +0.428 | +0.352 | +0.374 | +0.374 |
| pfkfb3 | 32 | +0.579 | -1.376 | +0.561 | +0.523 | +0.506 | +0.584 |
| cdk8 | 31 | +0.281 | -0.796 | +0.287 | +0.514 | -0.327 | +0.268 |
| p38 | 29 | +0.105 | -0.646 | -0.001 | -0.048 | -0.813 | +0.105 |
| eg5 | 27 | +0.622 | -0.186 | +0.607 | +0.103 | +0.345 | +0.608 |
| tnks2 | 27 | +0.291 | -0.862 | +0.216 | +0.253 | -1.255 | +0.212 |
| mcl1 | 25 | +0.544 | -0.491 | +0.298 | +0.132 | +0.207 | +0.387 |
| shp2 | 24 | +0.329 | -0.060 | +0.424 | +0.484 | +0.820 | +0.318 |
| ptp1b | 22 | +0.820 | -0.155 | +0.863 | +0.827 | +0.332 | +0.786 |
| thrombin | 22 | +0.401 | -1.861 | +0.401 | +0.474 | -2.804 | +0.440 |
| jnk1 | 21 | +0.625 | -0.132 | +0.578 | +0.573 | -1.174 | +0.625 |
| pde2 | 21 | +0.434 | -0.110 | +0.480 | +0.377 | +0.233 | +0.442 |

### Mean over the sets

| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | conformer limit |
|---|---:|---:|---:|---:|---:|---:|
| crystal | 0.00 A | 100% | 100% | - | +0.334 | 0.00 A |
| scrambled | 5.29 A | 0% | 0% | +0% | -0.683 | 0.00 A |
| mcs | 0.25 A | 94% | 98% | +95% | +0.329 | 0.00 A |
| field | 0.48 A | 89% | 97% | +91% | +0.293 | 0.00 A |
| variance | 1.35 A | 64% | 64% | +74% | -0.219 | 0.00 A |
| mcs+variance | 0.11 A | 99% | 99% | +98% | +0.313 | 0.00 A |

## crystal conformations, pose scrambled, no displacement guard; the pose search restarts 2 times from any orientation

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | scrambled | mcs | field | variance | mcs+variance |
|---|---:|---:|---:|---:|---:|---:|---:|
| mcl1 | 25 | 0.00 | 5.03 | 0.21 | 0.36 | 0.22 | 0.17 |
| thrombin | 22 | 0.00 | 4.49 | 0.04 | 0.43 | 0.11 | 0.10 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | scrambled | mcs | field | variance | mcs+variance |
|---|---:|---:|---:|---:|---:|---:|---:|
| mcl1 | 25 | +0.544 | -0.491 | +0.298 | +0.132 | +0.249 | +0.387 |
| thrombin | 22 | +0.401 | -1.861 | +0.401 | +0.474 | -0.014 | +0.440 |

### Mean over the sets

| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | conformer limit |
|---|---:|---:|---:|---:|---:|---:|
| crystal | 0.00 A | 100% | 100% | - | +0.472 | 0.00 A |
| scrambled | 4.76 A | 0% | 0% | +0% | -1.176 | 0.00 A |
| mcs | 0.13 A | 100% | 100% | +97% | +0.350 | 0.00 A |
| field | 0.39 A | 100% | 100% | +92% | +0.303 | 0.00 A |
| variance | 0.17 A | 79% | 79% | +97% | +0.117 | 0.00 A |
| mcs+variance | 0.14 A | 100% | 100% | +97% | +0.414 | 0.00 A |

## crystal conformations, pose scrambled, no displacement guard; the pose search restarts 20 times from a turn of at most 30 degrees

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | scrambled | mcs | field | variance | mcs+variance |
|---|---:|---:|---:|---:|---:|---:|---:|
| mcl1 | 25 | 0.00 | 5.03 | 0.21 | 0.36 | 0.17 | 0.17 |
| thrombin | 22 | 0.00 | 4.49 | 0.04 | 0.43 | 0.10 | 0.10 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | scrambled | mcs | field | variance | mcs+variance |
|---|---:|---:|---:|---:|---:|---:|---:|
| mcl1 | 25 | +0.544 | -0.491 | +0.298 | +0.132 | +0.387 | +0.387 |
| thrombin | 22 | +0.401 | -1.861 | +0.401 | +0.474 | -0.011 | +0.440 |

### Mean over the sets

| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | conformer limit |
|---|---:|---:|---:|---:|---:|---:|
| crystal | 0.00 A | 100% | 100% | - | +0.472 | 0.00 A |
| scrambled | 4.76 A | 0% | 0% | +0% | -1.176 | 0.00 A |
| mcs | 0.13 A | 100% | 100% | +97% | +0.350 | 0.00 A |
| field | 0.39 A | 100% | 100% | +92% | +0.303 | 0.00 A |
| variance | 0.14 A | 93% | 93% | +97% | +0.188 | 0.00 A |
| mcs+variance | 0.14 A | 100% | 100% | +97% | +0.414 | 0.00 A |

## crystal conformations, pose scrambled, no displacement guard; the pose search restarts 20 times from any orientation

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | scrambled | mcs | field | variance | mcs+variance |
|---|---:|---:|---:|---:|---:|---:|---:|
| syk | 44 | 0.00 | 5.95 | 0.05 | 0.60 | 0.13 | 0.13 |
| hif2a | 37 | 0.00 | 4.03 | 0.06 | 0.20 | 0.10 | 0.10 |
| bace | 36 | 0.00 | 5.04 | 0.21 | 0.15 | 0.05 | 0.05 |
| pde10 | 35 | 0.00 | 6.17 | 1.12 | 0.63 | 0.11 | 0.11 |
| bace_hunt | 32 | 0.00 | 4.29 | 0.00 | 0.08 | 0.05 | 0.05 |
| pfkfb3 | 32 | 0.00 | 5.44 | 0.05 | 0.22 | 0.10 | 0.10 |
| cdk8 | 31 | 0.00 | 4.90 | 0.12 | 0.42 | 0.10 | 0.10 |
| p38 | 29 | 0.00 | 5.03 | 0.62 | 0.53 | 0.09 | 0.09 |
| eg5 | 27 | 0.00 | 5.40 | 0.10 | 1.73 | 0.14 | 0.14 |
| tnks2 | 27 | 0.00 | 5.45 | 0.11 | 0.52 | 0.07 | 0.07 |
| mcl1 | 25 | 0.00 | 5.03 | 0.21 | 0.36 | 0.17 | 0.17 |
| shp2 | 24 | 0.00 | 5.22 | 0.70 | 1.02 | 0.21 | 0.21 |
| ptp1b | 22 | 0.00 | 7.74 | 0.10 | 0.18 | 0.10 | 0.10 |
| thrombin | 22 | 0.00 | 4.49 | 0.04 | 0.43 | 0.10 | 0.10 |
| jnk1 | 21 | 0.00 | 5.35 | 0.36 | 0.31 | 0.10 | 0.10 |
| pde2 | 21 | 0.00 | 5.08 | 0.20 | 0.29 | 0.12 | 0.12 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | scrambled | mcs | field | variance | mcs+variance |
|---|---:|---:|---:|---:|---:|---:|---:|
| syk | 44 | -0.204 | +0.207 | -0.191 | -0.434 | -0.363 | -0.363 |
| hif2a | 37 | +0.477 | -0.183 | +0.476 | +0.571 | +0.578 | +0.578 |
| bace | 36 | +0.274 | -1.669 | +0.254 | +0.258 | +0.256 | +0.256 |
| pde10 | 35 | -0.641 | -1.926 | -0.412 | -0.272 | -0.609 | -0.609 |
| bace_hunt | 32 | +0.406 | -0.678 | +0.428 | +0.352 | +0.374 | +0.374 |
| pfkfb3 | 32 | +0.579 | -1.376 | +0.561 | +0.523 | +0.584 | +0.584 |
| cdk8 | 31 | +0.281 | -0.796 | +0.287 | +0.514 | +0.268 | +0.268 |
| p38 | 29 | +0.105 | -0.646 | -0.001 | -0.048 | +0.105 | +0.105 |
| eg5 | 27 | +0.622 | -0.186 | +0.607 | +0.103 | +0.608 | +0.608 |
| tnks2 | 27 | +0.291 | -0.862 | +0.216 | +0.253 | +0.212 | +0.212 |
| mcl1 | 25 | +0.544 | -0.491 | +0.298 | +0.132 | +0.387 | +0.387 |
| shp2 | 24 | +0.329 | -0.060 | +0.424 | +0.484 | +0.301 | +0.301 |
| ptp1b | 22 | +0.820 | -0.155 | +0.863 | +0.827 | +0.786 | +0.786 |
| thrombin | 22 | +0.401 | -1.861 | +0.401 | +0.474 | +0.440 | +0.440 |
| jnk1 | 21 | +0.625 | -0.132 | +0.578 | +0.573 | +0.625 | +0.625 |
| pde2 | 21 | +0.434 | -0.110 | +0.480 | +0.377 | +0.442 | +0.442 |

### Mean over the sets

| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | conformer limit |
|---|---:|---:|---:|---:|---:|---:|
| crystal | 0.00 A | 100% | 100% | - | +0.334 | 0.00 A |
| scrambled | 5.29 A | 0% | 0% | +0% | -0.683 | 0.00 A |
| mcs | 0.25 A | 94% | 98% | +95% | +0.329 | 0.00 A |
| field | 0.48 A | 89% | 97% | +91% | +0.293 | 0.00 A |
| variance | 0.11 A | 100% | 100% | +98% | +0.312 | 0.00 A |
| mcs+variance | 0.11 A | 100% | 100% | +98% | +0.312 | 0.00 A |

## crystal conformations, pose scrambled, no displacement guard; the receptor pocket of each set

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | scrambled | mcs | variance | mcs+variance | pocket | pocket+variance | mcs+pocket+variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p38 | 29 | 0.00 | 5.03 | 0.62 | 0.32 | 0.09 | 4.40 | 3.73 | 0.13 |
| tnks2 | 27 | 0.00 | 5.45 | 0.11 | 0.21 | 0.07 | 3.45 | 1.26 | 0.09 |
| mcl1 | 25 | 0.00 | 5.03 | 0.21 | 0.29 | 0.17 | 3.00 | 0.31 | 0.22 |
| thrombin | 22 | 0.00 | 4.49 | 0.04 | 5.00 | 0.10 | 6.02 | 7.58 | 0.14 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | scrambled | mcs | variance | mcs+variance | pocket | pocket+variance | mcs+pocket+variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p38 | 29 | +0.105 | -0.646 | -0.001 | -0.813 | +0.105 | -0.971 | -1.319 | +0.105 |
| tnks2 | 27 | +0.291 | -0.862 | +0.216 | -1.255 | +0.212 | -1.070 | -0.354 | +0.233 |
| mcl1 | 25 | +0.544 | -0.491 | +0.298 | +0.207 | +0.387 | -0.041 | +0.141 | +0.269 |
| thrombin | 22 | +0.401 | -1.861 | +0.401 | -2.804 | +0.440 | -1.140 | -1.792 | +0.454 |

### Mean over the sets

| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | conformer limit |
|---|---:|---:|---:|---:|---:|---:|
| crystal | 0.00 A | 100% | 100% | - | +0.335 | 0.00 A |
| scrambled | 5.00 A | 0% | 0% | +0% | -0.965 | 0.00 A |
| mcs | 0.24 A | 97% | 100% | +95% | +0.229 | 0.00 A |
| variance | 1.46 A | 59% | 59% | +68% | -1.166 | 0.00 A |
| mcs+variance | 0.11 A | 100% | 100% | +98% | +0.286 | 0.00 A |
| pocket | 4.22 A | 26% | 31% | +14% | -0.805 | 0.00 A |
| pocket+variance | 3.22 A | 46% | 46% | +32% | -0.831 | 0.00 A |
| mcs+pocket+variance | 0.15 A | 100% | 100% | +97% | +0.265 | 0.00 A |

## crystal conformations, pose scrambled, no displacement guard; the receptor pocket of each set; the pose search restarts 20 times from a turn of at most 30 degrees

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | scrambled | mcs | variance | mcs+variance | pocket | pocket+variance | mcs+pocket+variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mcl1 | 25 | 0.00 | 5.03 | 0.21 | 0.17 | 0.17 | 1.00 | 0.31 | 0.22 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | scrambled | mcs | variance | mcs+variance | pocket | pocket+variance | mcs+pocket+variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mcl1 | 25 | +0.544 | -0.491 | +0.298 | +0.387 | +0.387 | +0.170 | +0.148 | +0.269 |

## crystal conformations, pose scrambled, no displacement guard; the receptor pocket of each set; the pose search restarts 20 times from any orientation

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | scrambled | mcs | variance | mcs+variance | pocket | pocket+variance | mcs+pocket+variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p38 | 29 | 0.00 | 5.03 | 0.62 | 0.09 | 0.09 | 1.12 | 0.12 | 0.12 |
| tnks2 | 27 | 0.00 | 5.45 | 0.11 | 0.07 | 0.07 | 1.70 | 0.11 | 0.09 |
| mcl1 | 25 | 0.00 | 5.03 | 0.21 | 0.17 | 0.17 | 0.93 | 0.22 | 0.22 |
| thrombin | 22 | 0.00 | 4.49 | 0.04 | 0.10 | 0.10 | 0.74 | 0.15 | 0.14 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | scrambled | mcs | variance | mcs+variance | pocket | pocket+variance | mcs+pocket+variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p38 | 29 | +0.105 | -0.646 | -0.001 | +0.105 | +0.105 | +0.228 | +0.105 | +0.105 |
| tnks2 | 27 | +0.291 | -0.862 | +0.216 | +0.212 | +0.212 | -0.882 | -0.480 | +0.233 |
| mcl1 | 25 | +0.544 | -0.491 | +0.298 | +0.387 | +0.387 | +0.180 | +0.268 | +0.269 |
| thrombin | 22 | +0.401 | -1.861 | +0.401 | +0.440 | +0.440 | -0.489 | -0.333 | +0.454 |

### Mean over the sets

| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | conformer limit |
|---|---:|---:|---:|---:|---:|---:|
| crystal | 0.00 A | 100% | 100% | - | +0.335 | 0.00 A |
| scrambled | 5.00 A | 0% | 0% | +0% | -0.965 | 0.00 A |
| mcs | 0.24 A | 97% | 100% | +95% | +0.229 | 0.00 A |
| variance | 0.11 A | 100% | 100% | +98% | +0.286 | 0.00 A |
| mcs+variance | 0.11 A | 100% | 100% | +98% | +0.286 | 0.00 A |
| pocket | 1.12 A | 57% | 74% | +78% | -0.241 | 0.00 A |
| pocket+variance | 0.15 A | 90% | 90% | +97% | -0.110 | 0.00 A |
| mcs+pocket+variance | 0.15 A | 100% | 100% | +97% | +0.265 | 0.00 A |

## crystal conformations, pose scrambled, no displacement guard; the receptor pocket of each set, its term weighted 0.25

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | pocket+variance | mcs+pocket+variance |
|---|---:|---:|---:|---:|
| p38 | 29 | 0.00 | 3.52 | 0.10 |
| tnks2 | 27 | 0.00 | 0.11 | 0.08 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | pocket+variance | mcs+pocket+variance |
|---|---:|---:|---:|---:|
| p38 | 29 | +0.105 | -2.173 | +0.109 |
| tnks2 | 27 | +0.291 | +0.337 | +0.227 |

### Mean over the sets

| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | conformer limit |
|---|---:|---:|---:|---:|---:|---:|
| crystal | 0.00 A | 100% | 100% | - | +0.198 | 0.00 A |
| pocket+variance | 1.81 A | 53% | 53% | +64% | -0.918 | 0.00 A |
| mcs+pocket+variance | 0.09 A | 100% | 100% | +98% | +0.168 | 0.00 A |

## crystal conformations, pose scrambled, no displacement guard; the receptor pocket of each set, its term weighted 4

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | pocket+variance | mcs+pocket+variance |
|---|---:|---:|---:|---:|
| p38 | 29 | 0.00 | 3.63 | 0.24 |
| tnks2 | 27 | 0.00 | 1.32 | 0.26 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | pocket+variance | mcs+pocket+variance |
|---|---:|---:|---:|---:|
| p38 | 29 | +0.105 | -1.477 | +0.089 |
| tnks2 | 27 | +0.291 | -1.138 | +0.216 |

### Mean over the sets

| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | conformer limit |
|---|---:|---:|---:|---:|---:|---:|
| crystal | 0.00 A | 100% | 100% | - | +0.198 | 0.00 A |
| pocket+variance | 2.47 A | 47% | 47% | +52% | -1.307 | 0.00 A |
| mcs+pocket+variance | 0.25 A | 100% | 100% | +95% | +0.152 | 0.00 A |

## crystal conformations, pose scrambled, no displacement guard; the receptor pocket of each set, its term weighted 0

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | pocket+variance |
|---|---:|---:|---:|
| p38 | 29 | 0.00 | 0.29 |
| tnks2 | 27 | 0.00 | 0.07 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | pocket+variance |
|---|---:|---:|---:|
| p38 | 29 | +0.105 | -0.749 |
| tnks2 | 27 | +0.291 | +0.192 |

### Mean over the sets

| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | conformer limit |
|---|---:|---:|---:|---:|---:|---:|
| crystal | 0.00 A | 100% | 100% | - | +0.198 | 0.00 A |
| pocket+variance | 0.18 A | 83% | 83% | +96% | -0.279 | 0.00 A |

## regenerated conformations, the generated conformation alone, no displacement guard

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | scrambled | mcs | mcs+variance | flex | mcs+flex |
|---|---:|---:|---:|---:|---:|---:|---:|
| syk | 44 | 0.00 | 6.25 | 2.67 | 3.06 | 1.46 | 1.21 |
| hif2a | 37 | 0.00 | 5.44 | 2.10 | 2.46 | 2.06 | 1.81 |
| bace | 36 | 0.00 | 6.01 | 2.21 | 2.67 | 1.39 | 1.03 |
| pde10 | 35 | 0.00 | 7.04 | 1.87 | 1.59 | 3.17 | 1.20 |
| bace_hunt | 32 | 0.00 | 5.09 | 0.95 | 1.11 | 1.00 | 0.85 |
| pfkfb3 | 32 | 0.00 | 6.47 | 1.57 | 1.64 | 1.69 | 1.14 |
| cdk8 | 31 | 0.00 | 5.90 | 2.99 | 3.68 | 2.33 | 1.43 |
| p38 | 29 | 0.00 | 6.35 | 1.60 | 1.80 | 2.07 | 1.86 |
| eg5 | 27 | 0.00 | 5.55 | 1.74 | 1.86 | 3.44 | 1.39 |
| tnks2 | 27 | 0.00 | 5.75 | 0.79 | 0.90 | 0.27 | 0.21 |
| mcl1 | 25 | 0.00 | 5.19 | 1.99 | 2.88 | 1.90 | 1.24 |
| shp2 | 24 | 0.00 | 7.18 | 1.66 | 1.52 | 1.32 | 1.32 |
| ptp1b | 22 | 0.00 | 8.14 | 2.44 | 3.23 | 5.05 | 2.64 |
| thrombin | 22 | 0.00 | 4.90 | 2.74 | 3.57 | 3.38 | 2.39 |
| jnk1 | 21 | 0.00 | 6.06 | 1.70 | 2.05 | 3.04 | 1.40 |
| pde2 | 21 | 0.00 | 6.58 | 2.05 | 2.20 | 5.94 | 1.42 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | scrambled | mcs | mcs+variance | flex | mcs+flex |
|---|---:|---:|---:|---:|---:|---:|---:|
| syk | 44 | -0.204 | +0.185 | +0.034 | -0.585 | +0.388 | -0.319 |
| hif2a | 37 | +0.477 | -0.540 | -0.480 | -0.355 | +0.472 | +0.539 |
| bace | 36 | +0.274 | -0.083 | -1.383 | -0.966 | +0.067 | -0.053 |
| pde10 | 35 | -0.641 | -2.407 | -0.504 | -1.413 | -0.881 | -0.482 |
| bace_hunt | 32 | +0.406 | -0.238 | +0.108 | +0.102 | +0.168 | +0.241 |
| pfkfb3 | 32 | +0.579 | -1.226 | +0.206 | +0.177 | +0.167 | +0.325 |
| cdk8 | 31 | +0.281 | -2.387 | -4.619 | -4.293 | -0.305 | -0.929 |
| p38 | 29 | +0.105 | +0.088 | -0.121 | -0.594 | -1.611 | -0.051 |
| eg5 | 27 | +0.622 | +0.050 | +0.637 | +0.637 | +0.049 | +0.213 |
| tnks2 | 27 | +0.291 | -2.021 | +0.388 | +0.461 | +0.067 | +0.432 |
| mcl1 | 25 | +0.544 | -0.383 | -0.803 | -0.146 | -0.047 | +0.203 |
| shp2 | 24 | +0.329 | +0.396 | +0.352 | +0.332 | +0.541 | +0.363 |
| ptp1b | 22 | +0.820 | -1.200 | +0.340 | +0.479 | -0.793 | +0.706 |
| thrombin | 22 | +0.401 | -2.278 | -0.861 | -0.527 | -0.464 | -0.137 |
| jnk1 | 21 | +0.625 | +0.347 | +0.129 | +0.180 | +0.123 | -0.180 |
| pde2 | 21 | +0.434 | +0.559 | +0.317 | +0.174 | -0.508 | +0.168 |

### Mean over the sets

| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | conformer limit |
|---|---:|---:|---:|---:|---:|---:|
| crystal | 0.00 A | 100% | 100% | - | +0.334 | 0.00 A |
| scrambled | 6.12 A | 0% | 1% | +0% | -0.696 | 1.73 A |
| mcs | 1.94 A | 15% | 54% | +68% | -0.391 | 1.73 A |
| mcs+variance | 2.27 A | 12% | 45% | +62% | -0.396 | 1.73 A |
| flex | 2.47 A | 21% | 46% | +60% | -0.160 | 1.39 A |
| mcs+flex | 1.41 A | 35% | 74% | +77% | +0.065 | 1.20 A |

## regenerated conformations, the generated conformation alone, no displacement guard; the pose search restarts 20 times from any orientation

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | scrambled | mcs | mcs+variance | flex | mcs+flex |
|---|---:|---:|---:|---:|---:|---:|---:|
| syk | 44 | 0.00 | 6.25 | 2.67 | 3.42 | 1.43 | 1.39 |
| hif2a | 37 | 0.00 | 5.44 | 2.10 | 2.46 | 1.82 | 1.53 |
| bace | 36 | 0.00 | 6.01 | 2.21 | 2.67 | 0.53 | 0.56 |
| pde10 | 35 | 0.00 | 7.04 | 1.87 | 1.59 | 1.27 | 1.20 |
| bace_hunt | 32 | 0.00 | 5.09 | 0.95 | 1.11 | 0.70 | 0.88 |
| pfkfb3 | 32 | 0.00 | 6.47 | 1.57 | 1.64 | 1.14 | 0.92 |
| cdk8 | 31 | 0.00 | 5.90 | 2.99 | 3.72 | 1.56 | 1.56 |
| p38 | 29 | 0.00 | 6.35 | 1.60 | 1.82 | 1.75 | 1.78 |
| eg5 | 27 | 0.00 | 5.55 | 1.74 | 1.86 | 1.51 | 1.51 |
| tnks2 | 27 | 0.00 | 5.75 | 0.79 | 0.90 | 0.20 | 0.21 |
| mcl1 | 25 | 0.00 | 5.19 | 1.99 | 2.88 | 0.98 | 1.00 |
| shp2 | 24 | 0.00 | 7.18 | 1.66 | 1.47 | 1.32 | 1.32 |
| ptp1b | 22 | 0.00 | 8.14 | 2.44 | 3.78 | 3.59 | 3.59 |
| thrombin | 22 | 0.00 | 4.90 | 2.74 | 3.57 | 2.53 | 2.53 |
| jnk1 | 21 | 0.00 | 6.06 | 1.70 | 2.05 | 1.59 | 1.59 |
| pde2 | 21 | 0.00 | 6.58 | 2.05 | 2.20 | 2.27 | 2.09 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | scrambled | mcs | mcs+variance | flex | mcs+flex |
|---|---:|---:|---:|---:|---:|---:|---:|
| syk | 44 | -0.204 | +0.185 | +0.034 | -0.144 | -0.171 | -0.250 |
| hif2a | 37 | +0.477 | -0.540 | -0.480 | -0.355 | +0.449 | +0.540 |
| bace | 36 | +0.274 | -0.083 | -1.383 | -0.966 | +0.222 | +0.239 |
| pde10 | 35 | -0.641 | -2.407 | -0.504 | -1.413 | -0.518 | -0.687 |
| bace_hunt | 32 | +0.406 | -0.238 | +0.108 | +0.102 | +0.141 | +0.131 |
| pfkfb3 | 32 | +0.579 | -1.226 | +0.206 | +0.177 | +0.452 | +0.416 |
| cdk8 | 31 | +0.281 | -2.387 | -4.619 | -4.355 | -0.889 | -0.901 |
| p38 | 29 | +0.105 | +0.088 | -0.121 | -0.025 | -0.354 | -0.195 |
| eg5 | 27 | +0.622 | +0.050 | +0.637 | +0.637 | -0.178 | -0.156 |
| tnks2 | 27 | +0.291 | -2.021 | +0.388 | +0.461 | +0.239 | +0.170 |
| mcl1 | 25 | +0.544 | -0.383 | -0.803 | -0.146 | +0.349 | +0.218 |
| shp2 | 24 | +0.329 | +0.396 | +0.352 | +0.351 | +0.401 | +0.400 |
| ptp1b | 22 | +0.820 | -1.200 | +0.340 | +0.508 | +0.588 | +0.582 |
| thrombin | 22 | +0.401 | -2.278 | -0.861 | -0.527 | -0.201 | -0.172 |
| jnk1 | 21 | +0.625 | +0.347 | +0.129 | +0.180 | -0.174 | -0.178 |
| pde2 | 21 | +0.434 | +0.559 | +0.317 | +0.258 | +0.073 | +0.134 |

### Mean over the sets

| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | conformer limit |
|---|---:|---:|---:|---:|---:|---:|
| crystal | 0.00 A | 100% | 100% | - | +0.334 | 0.00 A |
| scrambled | 6.12 A | 0% | 1% | +0% | -0.696 | 1.73 A |
| mcs | 1.94 A | 15% | 54% | +68% | -0.391 | 1.73 A |
| mcs+variance | 2.32 A | 13% | 44% | +61% | -0.329 | 1.73 A |
| flex | 1.51 A | 40% | 71% | +75% | +0.027 | 1.22 A |
| mcs+flex | 1.48 A | 40% | 74% | +76% | +0.018 | 1.18 A |

## regenerated conformations, 5 generated starting conformations, no displacement guard

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | scrambled | mcs | mcs+variance | flex | mcs+flex |
|---|---:|---:|---:|---:|---:|---:|---:|
| syk | 44 | 0.00 | 6.25 | 2.67 | 3.06 | 1.46 | 0.81 |
| hif2a | 37 | 0.00 | 5.44 | 2.10 | 2.46 | 1.86 | 0.63 |
| bace | 36 | 0.00 | 6.01 | 2.21 | 2.67 | 0.52 | 0.53 |
| pde10 | 35 | 0.00 | 7.04 | 1.87 | 1.59 | 2.31 | 1.05 |
| bace_hunt | 32 | 0.00 | 5.09 | 0.95 | 1.11 | 0.78 | 0.46 |
| pfkfb3 | 32 | 0.00 | 6.47 | 1.57 | 1.64 | 1.14 | 0.24 |
| cdk8 | 31 | 0.00 | 5.90 | 2.99 | 3.68 | 1.43 | 1.03 |
| p38 | 29 | 0.00 | 6.35 | 1.60 | 1.80 | 1.62 | 1.52 |
| eg5 | 27 | 0.00 | 5.55 | 1.74 | 1.86 | 1.81 | 0.86 |
| tnks2 | 27 | 0.00 | 5.75 | 0.79 | 0.90 | 0.18 | 0.19 |
| mcl1 | 25 | 0.00 | 5.19 | 1.99 | 2.88 | 1.38 | 0.64 |
| shp2 | 24 | 0.00 | 7.18 | 1.66 | 1.52 | 1.73 | 1.32 |
| ptp1b | 22 | 0.00 | 8.14 | 2.44 | 3.23 | 3.47 | 1.51 |
| thrombin | 22 | 0.00 | 4.90 | 2.74 | 3.57 | 2.49 | 0.85 |
| jnk1 | 21 | 0.00 | 6.06 | 1.70 | 2.05 | 1.73 | 0.90 |
| pde2 | 21 | 0.00 | 6.58 | 2.05 | 2.20 | 5.68 | 1.22 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | scrambled | mcs | mcs+variance | flex | mcs+flex |
|---|---:|---:|---:|---:|---:|---:|---:|
| syk | 44 | -0.204 | +0.185 | +0.034 | -0.585 | +0.326 | -0.238 |
| hif2a | 37 | +0.477 | -0.540 | -0.480 | -0.355 | +0.431 | +0.569 |
| bace | 36 | +0.274 | -0.083 | -1.383 | -0.966 | +0.192 | +0.151 |
| pde10 | 35 | -0.641 | -2.407 | -0.504 | -1.413 | -0.719 | -0.066 |
| bace_hunt | 32 | +0.406 | -0.238 | +0.108 | +0.102 | +0.228 | +0.235 |
| pfkfb3 | 32 | +0.579 | -1.226 | +0.206 | +0.177 | +0.090 | +0.517 |
| cdk8 | 31 | +0.281 | -2.387 | -4.619 | -4.293 | -0.168 | +0.128 |
| p38 | 29 | +0.105 | +0.088 | -0.121 | -0.594 | -1.163 | -0.618 |
| eg5 | 27 | +0.622 | +0.050 | +0.637 | +0.637 | -0.179 | +0.641 |
| tnks2 | 27 | +0.291 | -2.021 | +0.388 | +0.461 | +0.383 | +0.370 |
| mcl1 | 25 | +0.544 | -0.383 | -0.803 | -0.146 | -0.028 | +0.240 |
| shp2 | 24 | +0.329 | +0.396 | +0.352 | +0.332 | -0.093 | +0.518 |
| ptp1b | 22 | +0.820 | -1.200 | +0.340 | +0.479 | +0.266 | +0.494 |
| thrombin | 22 | +0.401 | -2.278 | -0.861 | -0.527 | -0.216 | +0.294 |
| jnk1 | 21 | +0.625 | +0.347 | +0.129 | +0.180 | +0.334 | +0.351 |
| pde2 | 21 | +0.434 | +0.559 | +0.317 | +0.174 | -0.554 | +0.642 |

### Mean over the sets

| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | conformer limit |
|---|---:|---:|---:|---:|---:|---:|
| crystal | 0.00 A | 100% | 100% | - | +0.334 | 0.00 A |
| scrambled | 6.12 A | 0% | 1% | +0% | -0.696 | 1.73 A |
| mcs | 1.94 A | 15% | 54% | +68% | -0.391 | 1.73 A |
| mcs+variance | 2.27 A | 12% | 45% | +62% | -0.396 | 1.73 A |
| flex | 1.85 A | 33% | 57% | +70% | -0.054 | 1.22 A |
| mcs+flex | 0.86 A | 59% | 88% | +86% | +0.264 | 0.71 A |

## regenerated conformations, 5 generated starting conformations, displacement guard 1 A

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | scrambled | mcs | mcs+variance | flex | mcs+flex |
|---|---:|---:|---:|---:|---:|---:|---:|
| mcl1 | 25 | 0.00 | 5.19 | 1.99 | 2.17 | 4.18 | 0.83 |
| jnk1 | 21 | 0.00 | 6.06 | 1.70 | 1.69 | 5.58 | 1.51 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | scrambled | mcs | mcs+variance | flex | mcs+flex |
|---|---:|---:|---:|---:|---:|---:|---:|
| mcl1 | 25 | +0.544 | -0.383 | -0.803 | -0.111 | +0.098 | +0.127 |
| jnk1 | 21 | +0.625 | +0.347 | +0.129 | -0.068 | +0.537 | -0.300 |

### Mean over the sets

| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | conformer limit |
|---|---:|---:|---:|---:|---:|---:|
| crystal | 0.00 A | 100% | 100% | - | +0.585 | 0.00 A |
| scrambled | 5.63 A | 0% | 0% | +0% | -0.018 | 1.76 A |
| mcs | 1.85 A | 0% | 67% | +67% | -0.337 | 1.76 A |
| mcs+variance | 1.93 A | 0% | 52% | +65% | -0.089 | 1.76 A |
| flex | 4.88 A | 0% | 12% | +14% | +0.317 | 1.86 A |
| mcs+flex | 1.17 A | 48% | 100% | +80% | -0.086 | 0.92 A |

## regenerated conformations, 5 generated starting conformations, displacement guard 2 A

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | scrambled | mcs | mcs+variance | flex | mcs+flex |
|---|---:|---:|---:|---:|---:|---:|---:|
| mcl1 | 25 | 0.00 | 5.19 | 1.99 | 2.52 | 3.51 | 0.51 |
| shp2 | 24 | 0.00 | 7.18 | 1.66 | 1.67 | 7.69 | 1.30 |
| thrombin | 22 | 0.00 | 4.90 | 2.74 | 3.14 | 4.08 | 1.87 |
| jnk1 | 21 | 0.00 | 6.06 | 1.70 | 1.95 | 5.40 | 0.89 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | scrambled | mcs | mcs+variance | flex | mcs+flex |
|---|---:|---:|---:|---:|---:|---:|---:|
| mcl1 | 25 | +0.544 | -0.383 | -0.803 | +0.002 | +0.090 | +0.285 |
| shp2 | 24 | +0.329 | +0.396 | +0.352 | +0.323 | -0.718 | +0.676 |
| thrombin | 22 | +0.401 | -2.278 | -0.861 | -0.461 | -0.521 | -0.105 |
| jnk1 | 21 | +0.625 | +0.347 | +0.129 | +0.146 | +0.092 | +0.358 |

### Mean over the sets

| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | conformer limit |
|---|---:|---:|---:|---:|---:|---:|
| crystal | 0.00 A | 100% | 100% | - | +0.475 | 0.00 A |
| scrambled | 5.84 A | 0% | 0% | +0% | -0.480 | 1.82 A |
| mcs | 2.02 A | 4% | 48% | +64% | -0.296 | 1.82 A |
| mcs+variance | 2.32 A | 4% | 30% | +58% | +0.003 | 1.82 A |
| flex | 5.17 A | 3% | 21% | +13% | -0.264 | 1.56 A |
| mcs+flex | 1.14 A | 49% | 82% | +80% | +0.303 | 0.98 A |

## regenerated conformations, 5 generated starting conformations, displacement guard 3 A

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | scrambled | mcs | mcs+variance | flex | mcs+flex |
|---|---:|---:|---:|---:|---:|---:|---:|
| mcl1 | 25 | 0.00 | 5.19 | 1.99 | 2.84 | 2.10 | 0.61 |
| jnk1 | 21 | 0.00 | 6.06 | 1.70 | 2.05 | 5.26 | 0.87 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | scrambled | mcs | mcs+variance | flex | mcs+flex |
|---|---:|---:|---:|---:|---:|---:|---:|
| mcl1 | 25 | +0.544 | -0.383 | -0.803 | -0.060 | -0.581 | +0.241 |
| jnk1 | 21 | +0.625 | +0.347 | +0.129 | +0.180 | +0.091 | +0.346 |

### Mean over the sets

| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | conformer limit |
|---|---:|---:|---:|---:|---:|---:|
| crystal | 0.00 A | 100% | 100% | - | +0.585 | 0.00 A |
| scrambled | 5.63 A | 0% | 0% | +0% | -0.018 | 1.76 A |
| mcs | 1.85 A | 0% | 67% | +67% | -0.337 | 1.76 A |
| mcs+variance | 2.45 A | 0% | 31% | +56% | +0.060 | 1.76 A |
| flex | 3.68 A | 12% | 42% | +36% | -0.245 | 1.61 A |
| mcs+flex | 0.74 A | 69% | 92% | +87% | +0.294 | 0.57 A |

## the training frame destroyed and recovered: 4 rounds, guard 1.0 A, rigid E-step

### Median RMSD to the crystal pose, angstrom

| set | n | crystal | scrambled | mcs | coalign | mcs+coalign |
|---|---:|---:|---:|---:|---:|---:|
| syk | 44 | 0.00 | 5.90 | 0.03 | 5.90 | 0.03 |
| hif2a | 37 | 0.00 | 5.28 | 0.07 | 5.51 | 0.07 |
| bace | 36 | 0.00 | 4.84 | 0.10 | 4.84 | 0.09 |
| pde10 | 35 | 0.00 | 6.23 | 0.47 | 6.23 | 0.42 |
| bace_hunt | 32 | 0.00 | 5.05 | 0.02 | 5.05 | 0.02 |
| pfkfb3 | 32 | 0.00 | 5.84 | 0.03 | 5.84 | 0.03 |
| cdk8 | 31 | 0.00 | 4.96 | 0.09 | 4.67 | 0.16 |
| p38 | 29 | 0.00 | 6.49 | 0.25 | 6.49 | 0.32 |
| eg5 | 27 | 0.00 | 5.75 | 0.07 | 5.51 | 0.07 |
| tnks2 | 27 | 0.00 | 5.79 | 0.06 | 5.79 | 0.13 |
| mcl1 | 25 | 0.00 | 5.24 | 0.16 | 5.24 | 0.14 |
| shp2 | 24 | 0.00 | 5.66 | 0.20 | 5.66 | 0.25 |
| ptp1b | 22 | 0.00 | 6.38 | 0.22 | 6.38 | 0.17 |
| thrombin | 22 | 0.00 | 4.95 | 0.04 | 5.23 | 0.04 |
| jnk1 | 21 | 0.00 | 5.33 | 0.28 | 5.33 | 0.23 |
| pde2 | 21 | 0.00 | 5.06 | 0.18 | 5.21 | 0.18 |

### External R2pred of the held-out molecules in those poses

| set | n | crystal | scrambled | mcs | coalign | mcs+coalign |
|---|---:|---:|---:|---:|---:|---:|
| syk | 44 | -0.191 | -0.474 | -0.181 | -0.474 | -0.181 |
| hif2a | 37 | +0.476 | -0.580 | +0.514 | -0.360 | +0.514 |
| bace | 36 | +0.254 | -11.314 | +0.257 | -11.314 | +0.223 |
| pde10 | 35 | -0.412 | -1.803 | -0.672 | -1.803 | -1.016 |
| bace_hunt | 32 | +0.428 | -0.207 | +0.437 | -0.207 | +0.437 |
| pfkfb3 | 32 | +0.561 | -1.568 | +0.541 | -1.568 | +0.541 |
| cdk8 | 31 | +0.287 | -0.641 | +0.244 | -0.780 | +0.350 |
| p38 | 29 | -0.001 | -2.496 | -0.121 | -2.496 | -0.274 |
| eg5 | 27 | +0.607 | -0.695 | +0.617 | -0.093 | +0.617 |
| tnks2 | 27 | +0.216 | -0.341 | +0.236 | -0.341 | +0.125 |
| mcl1 | 25 | +0.298 | -0.464 | +0.375 | -0.464 | +0.403 |
| shp2 | 24 | +0.424 | -0.893 | +0.396 | -0.893 | +0.278 |
| ptp1b | 22 | +0.863 | -9.534 | +0.738 | -9.534 | +0.730 |
| thrombin | 22 | +0.401 | -2.476 | +0.401 | -2.267 | +0.401 |
| jnk1 | 21 | +0.578 | -2.459 | +0.613 | -2.459 | +0.652 |
| pde2 | 21 | +0.480 | -2.485 | +0.479 | -2.140 | +0.479 |

### Mean over the sets

| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | conformer limit |
|---|---:|---:|---:|---:|---:|---:|
| crystal | 0.00 A | 100% | 100% | - | +0.329 | 0.00 A |
| scrambled | 5.55 A | 0% | 2% | +0% | -2.402 | 0.00 A |
| mcs | 0.14 A | 98% | 100% | +97% | +0.305 | 0.00 A |
| coalign | 5.56 A | 0% | 2% | -0% | -2.325 | 0.00 A |
| mcs+coalign | 0.15 A | 97% | 100% | +97% | +0.267 | 0.00 A |

