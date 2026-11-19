# Alloy surfaces: adsorption of H and OH, fields against element names and properties

Data: the slabs of Mamun et al. (Scientific Data 2019), BEEF-vdW, built by `examples/make_alloy_examples.py`. Energies are in eV and relative to the bare slab and to H2 (and H2O).

How to read the tables:

- **q2** is the out-of-fold coefficient of determination. The training mean has q2 near 0.
- **MAE** is the mean absolute error in eV.
- **MAE against the best reference** is the MAE of the method minus the MAE of the best reference method of the same task and folds, with a paired bootstrap interval (2.5 % to 97.5 %). A negative number means that the method has the smaller error. An interval that contains zero means that the two methods do not differ clearly.
- A *reference* is a model that uses only element names or tabulated element properties. A *field method* is a model on the structure.
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

The best reference is *properties of the binding elements, site kind and slab elements* (MAE 0.206 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.001 | 0.553 |  |
| binding elements and site kind | reference | +0.615 | 0.252 |  |
| binding elements, site kind and slab elements | reference | +0.706 | 0.215 |  |
| properties of the binding elements and site kind | reference | +0.602 | 0.264 |  |
| properties of the binding elements, site kind and slab elements | reference | +0.715 | 0.206 |  |
| site environment, invariant, classic fields | field method | +0.645 | 0.283 | +0.077 [+0.062, +0.093] |
| site environment, overlap in the lattice frame, classic fields | field method | +0.708 | 0.232 | +0.026 [+0.010, +0.042] |
| site environment, invariant, extended fields | field method | +0.737 | 0.230 | +0.024 [+0.010, +0.039] |
| site environment, overlap in the lattice frame, extended fields | field method | +0.746 | 0.209 | +0.003 [-0.013, +0.021] |

### Folds: slab-out

The best reference is *properties of the binding elements, site kind and slab elements* (MAE 0.268 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.006 | 0.555 |  |
| binding elements and site kind | reference | +0.556 | 0.282 |  |
| binding elements, site kind and slab elements | reference | +0.607 | 0.272 |  |
| properties of the binding elements and site kind | reference | +0.557 | 0.284 |  |
| properties of the binding elements, site kind and slab elements | reference | +0.620 | 0.268 |  |
| site environment, invariant, classic fields | field method | +0.621 | 0.295 | +0.027 [+0.011, +0.044] |
| site environment, overlap in the lattice frame, classic fields | field method | +0.710 | 0.239 | -0.029 [-0.045, -0.014] |
| site environment, invariant, extended fields | field method | +0.692 | 0.252 | -0.015 [-0.031, +0.000] |
| site environment, overlap in the lattice frame, extended fields | field method | +0.741 | 0.216 | -0.051 [-0.068, -0.036] |

### Folds: alloy-out

The best reference is *binding elements, site kind and slab elements* (MAE 0.267 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.003 | 0.554 |  |
| binding elements and site kind | reference | +0.564 | 0.279 |  |
| binding elements, site kind and slab elements | reference | +0.613 | 0.267 |  |
| properties of the binding elements and site kind | reference | +0.559 | 0.281 |  |
| properties of the binding elements, site kind and slab elements | reference | +0.613 | 0.267 |  |
| site environment, invariant, classic fields | field method | +0.614 | 0.296 | +0.029 [+0.012, +0.046] |
| site environment, overlap in the lattice frame, classic fields | field method | +0.718 | 0.239 | -0.028 [-0.044, -0.013] |
| site environment, invariant, extended fields | field method | +0.688 | 0.253 | -0.013 [-0.031, +0.003] |
| site environment, overlap in the lattice frame, extended fields | field method | +0.745 | 0.216 | -0.051 [-0.069, -0.034] |

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
| site environment, invariant, classic fields | field method | +0.520 | 0.321 | +0.100 [+0.084, +0.118] |
| site environment, overlap in the lattice frame, classic fields | field method | +0.615 | 0.281 | +0.061 [+0.049, +0.073] |
| site environment, invariant, extended fields | field method | +0.680 | 0.246 | +0.025 [+0.017, +0.035] |
| site environment, overlap in the lattice frame, extended fields | field method | +0.688 | 0.240 | +0.019 [+0.010, +0.028] |

### Folds: slab-out

The best reference is *properties of the binding elements, site kind and slab elements* (MAE 0.249 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.006 | 0.516 |  |
| binding elements and site kind | reference | +0.620 | 0.262 |  |
| binding elements, site kind and slab elements | reference | +0.622 | 0.252 |  |
| properties of the binding elements and site kind | reference | +0.620 | 0.264 |  |
| properties of the binding elements, site kind and slab elements | reference | +0.627 | 0.249 |  |
| site environment, invariant, classic fields | field method | +0.435 | 0.347 | +0.098 [+0.077, +0.120] |
| site environment, overlap in the lattice frame, classic fields | field method | +0.568 | 0.303 | +0.054 [+0.036, +0.071] |
| site environment, invariant, extended fields | field method | +0.612 | 0.272 | +0.023 [+0.008, +0.038] |
| site environment, overlap in the lattice frame, extended fields | field method | +0.673 | 0.252 | +0.003 [-0.013, +0.018] |

### Folds: alloy-out

The best reference is *properties of the binding elements, site kind and slab elements* (MAE 0.232 eV).

| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |
| --- | --- | --- | --- | --- |
| training mean | baseline | -0.018 | 0.519 |  |
| binding elements and site kind | reference | +0.624 | 0.265 |  |
| binding elements, site kind and slab elements | reference | +0.653 | 0.240 |  |
| properties of the binding elements and site kind | reference | +0.622 | 0.266 |  |
| properties of the binding elements, site kind and slab elements | reference | +0.665 | 0.232 |  |
| site environment, invariant, classic fields | field method | +0.423 | 0.357 | +0.125 [+0.102, +0.149] |
| site environment, overlap in the lattice frame, classic fields | field method | +0.534 | 0.312 | +0.080 [+0.065, +0.096] |
| site environment, invariant, extended fields | field method | +0.614 | 0.278 | +0.046 [+0.028, +0.062] |
| site environment, overlap in the lattice frame, extended fields | field method | +0.668 | 0.255 | +0.022 [+0.008, +0.035] |

