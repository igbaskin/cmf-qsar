# The periodic overlap kernel against the periodic autocorrelation (item M4)

`matbench_perovskites`: 1500 ABO3 cells of 5 atoms, formation energy in eV/atom (mean +1.437, spread 0.723). Fields: `electrostatic`, `steric`, `prop:electronegativity`, `prop:inverse_sqrt_mass`, charges from electronegativity differences.

**How far the set is from one lattice.** The relaxed cells have a mean volume of 72.4 A^3, and the largest deviation from the mean is 135 % in volume and 34 % in a lattice vector. The overlap integral compares the two cells point by point, so that spread is a frame error; the one-lattice frame removes it, and scales the bond lengths instead.

## What each kernel predicts

Five Matbench folds, the setting and the ridge chosen by ten-fold cross-validation inside each training part. Lower MAE is better.

| kernel | frame | R2 | RMSE, eV/atom | MAE, eV/atom | settings chosen |
|---|---|---:|---:|---:|---|
| autocorrelation | as given | +0.497 | 0.5124 | 0.3839 | sigma 0.5, sigma 0.75 |
| autocorrelation | one lattice | +0.494 | 0.5138 | 0.4050 | sigma 0.5, sigma 0.75 |
| open | as given | +0.448 | 0.5368 | 0.4040 | alpha 1, alpha 2, alpha 4 |
| open | one lattice | +0.379 | 0.5697 | 0.4347 | alpha 1, alpha 2, alpha 4 |
| overlap | as given | +0.350 | 0.5826 | 0.4487 | alpha 4 |
| overlap | one lattice | +0.479 | 0.5216 | 0.4069 | alpha 4 |

**The two orders of the integral.** The periodic overlap integrates over the cell of one structure and the images of the other, so with cells of different size the two orders differ and the kernel is the average of them. The largest disagreement before the average: overlap | as given, 200.0%; overlap | one lattice, 0.0%.

## What the co-field says

One overlap model on all 1500 cells (frame: one lattice, attenuation 2, ridge 0.001; R2 +0.483 on its own training set, which is not a validation). Decomposition rule 3 gives one contribution per site of every cell; the mean over a sample of 150 cells, by kind of site:

| site | mean contribution, eV/atom | spread | sites |
|---|---:|---:|---:|
| A | +2.3737 | 0.5823 | 150 |
| B | -6.1553 | 0.8119 | 149 |
| X | -3.8685 | 0.5461 | 451 |

The co-field itself is drawn in `cofield.png`, with the cell over it, in the plane of the A site and the X sites (z = 0) and in the plane of the B site and the equatorial X sites (z = a/2).

It is also written as a Gaussian cube over one cell, which carries the atoms beside the values, so a viewer draws the periodic co-field around the structure: `cofield_steric.cube` (steric, 18^3 points at 0.24 A, the atoms of O3PbRu_12).

`cofield_3d.png` draws the isosurfaces of that cube around the cell, read back from the file as an outside viewer would read it.

