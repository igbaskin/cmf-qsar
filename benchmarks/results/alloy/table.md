# Alloy surfaces: adsorption of H and OH, fields against element names and properties

Data: the slabs of Mamun et al. (Scientific Data 2019), BEEF-vdW, built by `examples/make_alloy_examples.py`. Energies are in eV and relative to the bare slab and to H2 (and H2O).

How to read the tables:

- **q2** is the out-of-fold coefficient of determination. The training mean has q2 near 0.
- **MAE** is the mean absolute error in eV.
- **MAE against the best reference** is the MAE of the method minus the MAE of the best reference method of the same task and folds, with a paired bootstrap interval (2.5 % to 97.5 %). A negative number means that the method has the smaller error. An interval that contains zero means that the two methods do not differ clearly.
- A *reference* is a model that uses only element names, tabulated element properties or (for sites) counts of the atoms of each element in four shells around the site. A *field method* is a model on a smooth field of the structure.
- **slab** tasks: the sample is the bare slab and the target is the lowest adsorption energy over its sites. **site** tasks: the sample is one site of one slab and the target is the energy at that site.
- Folds: **random** (10 folds), **slab-out** (all sites of a slab in one fold) and **alloy-out** (all slabs with the same elements in one fold).

## slab H [all]: 1918 samples

Mean -0.43 eV, standard deviation 0.61 eV.

### Folds: random

The best reference is *element properties, whole slab* (MAE 0.189 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.000 | 0.454 |  |
| element fractions | reference | +0.644 | 0.218 |  |
| element fractions by layer | reference | +0.642 | 0.220 |  |
| element properties, whole slab | reference | +0.685 | 0.189 |  |
| element properties, top layer and the rest | reference | +0.666 | 0.196 |  |
| periodic autocorrelation, classic fields | field method | +0.573 | 0.245 | +0.056 [+0.047, +0.066] |
| periodic autocorrelation, extended fields | field method | +0.677 | 0.199 | +0.011 [+0.003, +0.018] |
| periodic autocorrelation, indicator fields | field method | +0.624 | 0.219 | +0.030 [+0.022, +0.040] |

### Folds: alloy-out

The best reference is *element properties, whole slab* (MAE 0.190 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.002 | 0.455 |  |
| element fractions | reference | +0.624 | 0.229 |  |
| element fractions by layer | reference | +0.621 | 0.231 |  |
| element properties, whole slab | reference | +0.673 | 0.190 |  |
| element properties, top layer and the rest | reference | +0.648 | 0.200 |  |
| periodic autocorrelation, classic fields | field method | +0.570 | 0.246 | +0.056 [+0.047, +0.066] |
| periodic autocorrelation, extended fields | field method | +0.670 | 0.202 | +0.012 [+0.004, +0.020] |
| periodic autocorrelation, indicator fields | field method | +0.616 | 0.238 | +0.048 [+0.038, +0.057] |

## slab H [converged]: 1909 samples

Mean -0.41 eV, standard deviation 0.62 eV.

### Folds: random

The best reference is *element properties, whole slab* (MAE 0.186 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.001 | 0.459 |  |
| element fractions | reference | +0.637 | 0.219 |  |
| element fractions by layer | reference | +0.635 | 0.222 |  |
| element properties, whole slab | reference | +0.675 | 0.186 |  |
| element properties, top layer and the rest | reference | +0.661 | 0.194 |  |
| periodic autocorrelation, classic fields | field method | +0.567 | 0.247 | +0.060 [+0.051, +0.069] |
| periodic autocorrelation, extended fields | field method | +0.673 | 0.201 | +0.014 [+0.006, +0.022] |
| periodic autocorrelation, indicator fields | field method | +0.614 | 0.223 | +0.037 [+0.028, +0.046] |

### Folds: alloy-out

The best reference is *element properties, whole slab* (MAE 0.190 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.002 | 0.459 |  |
| element fractions | reference | +0.626 | 0.229 |  |
| element fractions by layer | reference | +0.621 | 0.233 |  |
| element properties, whole slab | reference | +0.677 | 0.190 |  |
| element properties, top layer and the rest | reference | +0.650 | 0.201 |  |
| periodic autocorrelation, classic fields | field method | +0.567 | 0.247 | +0.057 [+0.047, +0.067] |
| periodic autocorrelation, extended fields | field method | +0.669 | 0.204 | +0.014 [+0.005, +0.022] |
| periodic autocorrelation, indicator fields | field method | +0.615 | 0.241 | +0.051 [+0.041, +0.061] |

## slab OH [all]: 213 samples

Mean -0.06 eV, standard deviation 0.55 eV.

### Folds: random

The best reference is *element properties, whole slab* (MAE 0.217 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.007 | 0.427 |  |
| element fractions | reference | +0.627 | 0.225 |  |
| element fractions by layer | reference | +0.650 | 0.221 |  |
| element properties, whole slab | reference | +0.620 | 0.217 |  |
| element properties, top layer and the rest | reference | +0.479 | 0.247 |  |
| periodic autocorrelation, classic fields | field method | +0.436 | 0.299 | +0.082 [+0.053, +0.112] |
| periodic autocorrelation, extended fields | field method | +0.481 | 0.259 | +0.042 [+0.008, +0.078] |
| periodic autocorrelation, indicator fields | field method | +0.695 | 0.192 | -0.025 [-0.050, -0.002] |

### Folds: alloy-out

The best reference is *element properties, whole slab* (MAE 0.251 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.022 | 0.432 |  |
| element fractions | reference | +0.538 | 0.274 |  |
| element fractions by layer | reference | +0.540 | 0.274 |  |
| element properties, whole slab | reference | +0.554 | 0.251 |  |
| element properties, top layer and the rest | reference | +0.525 | 0.256 |  |
| periodic autocorrelation, classic fields | field method | +0.325 | 0.324 | +0.073 [+0.039, +0.106] |
| periodic autocorrelation, extended fields | field method | +0.412 | 0.287 | +0.037 [+0.002, +0.069] |
| periodic autocorrelation, indicator fields | field method | +0.474 | 0.292 | +0.041 [+0.009, +0.071] |

## slab OH [converged]: 209 samples

Mean -0.04 eV, standard deviation 0.54 eV.

### Folds: random

The best reference is *element fractions by layer* (MAE 0.200 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.012 | 0.415 |  |
| element fractions | reference | +0.700 | 0.202 |  |
| element fractions by layer | reference | +0.703 | 0.200 |  |
| element properties, whole slab | reference | +0.621 | 0.211 |  |
| element properties, top layer and the rest | reference | +0.582 | 0.214 |  |
| periodic autocorrelation, classic fields | field method | +0.481 | 0.283 | +0.084 [+0.055, +0.114] |
| periodic autocorrelation, extended fields | field method | +0.517 | 0.261 | +0.061 [+0.034, +0.092] |
| periodic autocorrelation, indicator fields | field method | +0.756 | 0.175 | -0.025 [-0.040, -0.010] |

### Folds: alloy-out

The best reference is *element properties, whole slab* (MAE 0.233 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.031 | 0.420 |  |
| element fractions | reference | +0.543 | 0.261 |  |
| element fractions by layer | reference | +0.516 | 0.266 |  |
| element properties, whole slab | reference | +0.599 | 0.233 |  |
| element properties, top layer and the rest | reference | +0.546 | 0.240 |  |
| periodic autocorrelation, classic fields | field method | +0.430 | 0.296 | +0.064 [+0.031, +0.097] |
| periodic autocorrelation, extended fields | field method | +0.482 | 0.267 | +0.034 [-0.001, +0.073] |
| periodic autocorrelation, indicator fields | field method | +0.512 | 0.274 | +0.041 [+0.014, +0.068] |

## site H [all]: 2500 samples

Mean -0.11 eV, standard deviation 0.69 eV.

### Folds: random

The best reference is *neighbour counts, binding elements, site kind and slab elements* (MAE 0.195 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.001 | 0.553 |  |
| binding elements and site kind | reference | +0.615 | 0.252 |  |
| binding elements, site kind and slab elements | reference | +0.706 | 0.215 |  |
| properties of the binding elements and site kind | reference | +0.602 | 0.264 |  |
| properties of the binding elements, site kind and slab elements | reference | +0.715 | 0.206 |  |
| neighbour counts of each element in four shells | reference | +0.751 | 0.207 |  |
| neighbour counts of each element in four shells, and site kind | reference | +0.754 | 0.203 |  |
| neighbour sums of element properties in four shells, and site kind | reference | +0.778 | 0.200 |  |
| neighbour counts, binding elements, site kind and slab elements | reference | +0.769 | 0.195 |  |
| site environment, invariant, classic fields | field method | +0.645 | 0.283 | +0.088 [+0.073, +0.104] |
| site environment, overlap in the lattice frame, classic fields | field method | +0.708 | 0.232 | +0.037 [+0.022, +0.054] |
| site environment, invariant, extended fields | field method | +0.737 | 0.230 | +0.035 [+0.022, +0.049] |
| site environment, overlap in the lattice frame, extended fields | field method | +0.746 | 0.209 | +0.014 [-0.001, +0.031] |

### Folds: slab-out

The best reference is *neighbour counts, binding elements, site kind and slab elements* (MAE 0.244 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.006 | 0.555 |  |
| binding elements and site kind | reference | +0.556 | 0.282 |  |
| binding elements, site kind and slab elements | reference | +0.607 | 0.272 |  |
| properties of the binding elements and site kind | reference | +0.557 | 0.284 |  |
| properties of the binding elements, site kind and slab elements | reference | +0.620 | 0.268 |  |
| neighbour counts of each element in four shells | reference | +0.651 | 0.263 |  |
| neighbour counts of each element in four shells, and site kind | reference | +0.649 | 0.265 |  |
| neighbour sums of element properties in four shells, and site kind | reference | +0.700 | 0.246 |  |
| neighbour counts, binding elements, site kind and slab elements | reference | +0.693 | 0.244 |  |
| site environment, invariant, classic fields | field method | +0.621 | 0.295 | +0.051 [+0.037, +0.065] |
| site environment, overlap in the lattice frame, classic fields | field method | +0.710 | 0.239 | -0.005 [-0.018, +0.008] |
| site environment, invariant, extended fields | field method | +0.692 | 0.252 | +0.009 [-0.004, +0.022] |
| site environment, overlap in the lattice frame, extended fields | field method | +0.741 | 0.216 | -0.027 [-0.040, -0.015] |

### Folds: alloy-out

The best reference is *neighbour counts, binding elements, site kind and slab elements* (MAE 0.243 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.003 | 0.554 |  |
| binding elements and site kind | reference | +0.564 | 0.279 |  |
| binding elements, site kind and slab elements | reference | +0.613 | 0.267 |  |
| properties of the binding elements and site kind | reference | +0.559 | 0.281 |  |
| properties of the binding elements, site kind and slab elements | reference | +0.613 | 0.267 |  |
| neighbour counts of each element in four shells | reference | +0.666 | 0.260 |  |
| neighbour counts of each element in four shells, and site kind | reference | +0.666 | 0.261 |  |
| neighbour sums of element properties in four shells, and site kind | reference | +0.704 | 0.245 |  |
| neighbour counts, binding elements, site kind and slab elements | reference | +0.698 | 0.243 |  |
| site environment, invariant, classic fields | field method | +0.614 | 0.296 | +0.053 [+0.039, +0.067] |
| site environment, overlap in the lattice frame, classic fields | field method | +0.718 | 0.239 | -0.004 [-0.017, +0.008] |
| site environment, invariant, extended fields | field method | +0.688 | 0.253 | +0.010 [-0.003, +0.024] |
| site environment, overlap in the lattice frame, extended fields | field method | +0.745 | 0.216 | -0.027 [-0.040, -0.015] |

## site OH [all]: 2475 samples

Mean +0.59 eV, standard deviation 0.66 eV.

### Folds: random

The best reference is *binding elements, site kind and slab elements* (MAE 0.221 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.001 | 0.514 |  |
| binding elements and site kind | reference | +0.645 | 0.252 |  |
| binding elements, site kind and slab elements | reference | +0.718 | 0.221 |  |
| properties of the binding elements and site kind | reference | +0.645 | 0.252 |  |
| properties of the binding elements, site kind and slab elements | reference | +0.714 | 0.221 |  |
| neighbour counts of each element in four shells | reference | +0.689 | 0.240 |  |
| neighbour counts of each element in four shells, and site kind | reference | +0.695 | 0.235 |  |
| neighbour sums of element properties in four shells, and site kind | reference | +0.715 | 0.222 |  |
| neighbour counts, binding elements, site kind and slab elements | reference | +0.710 | 0.227 |  |
| site environment, invariant, classic fields | field method | +0.520 | 0.321 | +0.100 [+0.084, +0.118] |
| site environment, overlap in the lattice frame, classic fields | field method | +0.615 | 0.281 | +0.061 [+0.049, +0.073] |
| site environment, invariant, extended fields | field method | +0.680 | 0.246 | +0.025 [+0.017, +0.035] |
| site environment, overlap in the lattice frame, extended fields | field method | +0.688 | 0.240 | +0.019 [+0.010, +0.028] |

### Folds: slab-out

The best reference is *neighbour sums of element properties in four shells, and site kind* (MAE 0.240 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.006 | 0.516 |  |
| binding elements and site kind | reference | +0.620 | 0.262 |  |
| binding elements, site kind and slab elements | reference | +0.622 | 0.252 |  |
| properties of the binding elements and site kind | reference | +0.620 | 0.264 |  |
| properties of the binding elements, site kind and slab elements | reference | +0.627 | 0.249 |  |
| neighbour counts of each element in four shells | reference | +0.615 | 0.276 |  |
| neighbour counts of each element in four shells, and site kind | reference | +0.617 | 0.272 |  |
| neighbour sums of element properties in four shells, and site kind | reference | +0.678 | 0.240 |  |
| neighbour counts, binding elements, site kind and slab elements | reference | +0.625 | 0.260 |  |
| site environment, invariant, classic fields | field method | +0.435 | 0.347 | +0.107 [+0.087, +0.130] |
| site environment, overlap in the lattice frame, classic fields | field method | +0.568 | 0.303 | +0.063 [+0.049, +0.079] |
| site environment, invariant, extended fields | field method | +0.612 | 0.272 | +0.032 [+0.018, +0.047] |
| site environment, overlap in the lattice frame, extended fields | field method | +0.673 | 0.252 | +0.012 [-0.000, +0.025] |

### Folds: alloy-out

The best reference is *properties of the binding elements, site kind and slab elements* (MAE 0.232 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.018 | 0.519 |  |
| binding elements and site kind | reference | +0.624 | 0.265 |  |
| binding elements, site kind and slab elements | reference | +0.653 | 0.240 |  |
| properties of the binding elements and site kind | reference | +0.622 | 0.266 |  |
| properties of the binding elements, site kind and slab elements | reference | +0.665 | 0.232 |  |
| neighbour counts of each element in four shells | reference | +0.610 | 0.278 |  |
| neighbour counts of each element in four shells, and site kind | reference | +0.627 | 0.271 |  |
| neighbour sums of element properties in four shells, and site kind | reference | +0.674 | 0.245 |  |
| neighbour counts, binding elements, site kind and slab elements | reference | +0.631 | 0.261 |  |
| site environment, invariant, classic fields | field method | +0.423 | 0.357 | +0.125 [+0.102, +0.149] |
| site environment, overlap in the lattice frame, classic fields | field method | +0.534 | 0.312 | +0.080 [+0.065, +0.096] |
| site environment, invariant, extended fields | field method | +0.614 | 0.278 | +0.046 [+0.028, +0.062] |
| site environment, overlap in the lattice frame, extended fields | field method | +0.668 | 0.255 | +0.022 [+0.008, +0.035] |

## The shells around a site: references added on 19 November 2026

Two more references for the site tasks, with no field and no frame: for each distance shell around the point of the site (up to 2.5, 3.5, 4.5, 6 A), the number of atoms and the sum of their standardised element properties (*shell properties*), or the number of atoms of each element (*shell elements*), each with the kind of site. Same rows, folds and inner choice as the site tasks above. The differences are in MAE (eV); a negative number means that the first method has the smaller error.

### site H [all]

| folds | reference | q2 | MAE (eV) | chosen kernels |
| --- | --- | --- | --- | --- |
| random | shell properties and site kind | +0.817 | 0.174 | shell properties gauss x1 lam=0.1 (9), shell properties gauss x1 lam=0.01 (1) |
| random | shell elements and site kind | +0.799 | 0.188 | shell elements gauss x1 lam=0.1 (10) |
| slab-out | shell properties and site kind | +0.771 | 0.201 | shell properties gauss x4 lam=0.01 (6), shell properties gauss x16 lam=0.001 (1), shell properties gauss x1 lam=0.1 (3) |
| slab-out | shell elements and site kind | +0.698 | 0.241 | shell elements gauss x4 lam=0.1 (8), shell elements gauss x16 lam=0.01 (1), shell elements gauss x1 lam=0.1 (1) |
| alloy-out | shell properties and site kind | +0.774 | 0.200 | shell properties gauss x4 lam=0.01 (4), shell properties gauss x1 lam=0.1 (6) |
| alloy-out | shell elements and site kind | +0.718 | 0.235 | shell elements gauss x4 lam=0.1 (9), shell elements gauss x16 lam=0.01 (1) |

| folds | method | against | MAE difference (eV) |
| --- | --- | --- | --- |
| random | shell properties and site kind | neighbour counts, binding elements, site kind and slab elements | -0.021 [-0.033, -0.009] |
| random | site environment, invariant, classic fields | shell properties and site kind | +0.109 [+0.097, +0.121] |
| random | site environment, overlap in the lattice frame, classic fields | shell properties and site kind | +0.058 [+0.047, +0.070] |
| random | site environment, invariant, extended fields | shell properties and site kind | +0.056 [+0.046, +0.066] |
| random | site environment, overlap in the lattice frame, extended fields | shell properties and site kind | +0.035 [+0.024, +0.046] |
| slab-out | shell properties and site kind | neighbour counts, binding elements, site kind and slab elements | -0.043 [-0.055, -0.032] |
| slab-out | site environment, invariant, classic fields | shell properties and site kind | +0.094 [+0.082, +0.107] |
| slab-out | site environment, overlap in the lattice frame, classic fields | shell properties and site kind | +0.038 [+0.027, +0.049] |
| slab-out | site environment, invariant, extended fields | shell properties and site kind | +0.052 [+0.041, +0.063] |
| slab-out | site environment, overlap in the lattice frame, extended fields | shell properties and site kind | +0.016 [+0.004, +0.027] |
| alloy-out | shell properties and site kind | neighbour counts, binding elements, site kind and slab elements | -0.043 [-0.054, -0.032] |
| alloy-out | site environment, invariant, classic fields | shell properties and site kind | +0.096 [+0.083, +0.109] |
| alloy-out | site environment, overlap in the lattice frame, classic fields | shell properties and site kind | +0.039 [+0.028, +0.050] |
| alloy-out | site environment, invariant, extended fields | shell properties and site kind | +0.053 [+0.042, +0.065] |
| alloy-out | site environment, overlap in the lattice frame, extended fields | shell properties and site kind | +0.016 [+0.005, +0.027] |

### site OH [all]

| folds | reference | q2 | MAE (eV) | chosen kernels |
| --- | --- | --- | --- | --- |
| random | shell properties and site kind | +0.740 | 0.208 | shell properties gauss x1 lam=0.1 (10) |
| random | shell elements and site kind | +0.713 | 0.224 | shell elements gauss x4 lam=0.01 (7), shell elements gauss x1 lam=0.1 (3) |
| slab-out | shell properties and site kind | +0.715 | 0.220 | shell properties gauss x16 lam=0.001 (5), shell properties gauss x4 lam=0.01 (4), shell properties gauss x1 lam=0.1 (1) |
| slab-out | shell elements and site kind | +0.650 | 0.251 | shell elements gauss x4 lam=0.1 (8), shell elements gauss x4 lam=0.01 (2) |
| alloy-out | shell properties and site kind | +0.719 | 0.222 | shell properties gauss x4 lam=0.01 (5), shell properties gauss x1 lam=0.1 (4), shell properties gauss x16 lam=0.001 (1) |
| alloy-out | shell elements and site kind | +0.634 | 0.266 | shell elements gauss x4 lam=0.01 (2), shell elements gauss x4 lam=0.1 (8) |

| folds | method | against | MAE difference (eV) |
| --- | --- | --- | --- |
| random | shell properties and site kind | binding elements, site kind and slab elements | -0.013 [-0.019, -0.007] |
| random | site environment, invariant, classic fields | shell properties and site kind | +0.113 [+0.097, +0.131] |
| random | site environment, overlap in the lattice frame, classic fields | shell properties and site kind | +0.073 [+0.062, +0.087] |
| random | site environment, invariant, extended fields | shell properties and site kind | +0.038 [+0.030, +0.047] |
| random | site environment, overlap in the lattice frame, extended fields | shell properties and site kind | +0.032 [+0.024, +0.040] |
| slab-out | shell properties and site kind | neighbour sums of element properties in four shells, and site kind | -0.020 [-0.029, -0.011] |
| slab-out | site environment, invariant, classic fields | shell properties and site kind | +0.127 [+0.106, +0.149] |
| slab-out | site environment, overlap in the lattice frame, classic fields | shell properties and site kind | +0.083 [+0.070, +0.098] |
| slab-out | site environment, invariant, extended fields | shell properties and site kind | +0.052 [+0.039, +0.066] |
| slab-out | site environment, overlap in the lattice frame, extended fields | shell properties and site kind | +0.032 [+0.022, +0.043] |
| alloy-out | shell properties and site kind | properties of the binding elements, site kind and slab elements | -0.010 [-0.026, +0.002] |
| alloy-out | site environment, invariant, classic fields | shell properties and site kind | +0.135 [+0.116, +0.156] |
| alloy-out | site environment, overlap in the lattice frame, classic fields | shell properties and site kind | +0.090 [+0.074, +0.111] |
| alloy-out | site environment, invariant, extended fields | shell properties and site kind | +0.055 [+0.044, +0.068] |
| alloy-out | site environment, overlap in the lattice frame, extended fields | shell properties and site kind | +0.032 [+0.023, +0.042] |

