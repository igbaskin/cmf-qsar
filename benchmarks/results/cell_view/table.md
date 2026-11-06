# A periodic structure in the application: what it costs to show, and what it reads

`python benchmarks/run_cell_view.py`, written 2026-11-06 21:34:54.

## What one cell costs to show

One row per example set: the cells it holds, and the median over the first 5 cells of the time to expand the periodic images, and of the time the window takes to draw the cell, its atoms and those images. The radius is the one the periodic overlap kernel of the standard inorganic field set reaches to, which is the radius the window offers once a model is fitted.

| Set | Structures | Atoms per cell | Periodic in | Images within the cutoff | Expanding them | Drawing the view | Reading the files |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| disordered: thirteen COD solid electrolytes, thirteen different lattices | 13 | 4 to 256 | 3 directions | 4650 (r = 14.9 A) | 12 ms | 15 ms | 0.5 s |
| hosts: three DFT-relaxed solid-electrolyte supercells | 3 | 112 to 128 | 3 directions | 5532 (r = 14.9 A) | 8 ms | 11 ms | 0.0 s |
| sheets: 139 single-atom sheets on one lattice, periodic in two directions | 139 | 127 to 127 | 2 directions | 836 (r = 14.9 A) | 1 ms | 3 ms | 0.1 s |

## Is the reading exact, and what does it cost

The colouring of the sites is a decomposition of the prediction, so its numbers must sum to the prediction less the bias; the residual below is the largest difference over five cells. The co-field exists for the overlap kernel only, and being the field of a periodic model it must repeat from cell to cell: the periodicity is the largest difference between its value at a point and at the same point one lattice vector away, as a fraction of its peak.

| Model | Cells | Spread of the lattice | q2 (3 folds) | Per-site split | Residual of the sum | Co-field inside the cell | Left out by the cutoff |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| periodic autocorrelation on 13 COD electrolytes | 13 | 110.5 % | 0.071 | 307 ms per cell | 6e-14 (target sd 16.33) | none: the kernel is a distribution of distances | - |
| periodic overlap kernel on 20 single-atom sheets | 20 | 0.4 % | 0.990 | 93 ms per cell | 7e-03 (target sd 0.34) | 30744 points at 0.50 A, 1.4 s | 8e-15 beyond 10.5 A |

The residual of the autocorrelation split is rounding. The residual of the overlap split is not: the periodic overlap integral runs over the cell of one structure and the images of the other, so its two orders agree only on exactly one lattice, and the split is one of the two orders while the kernel is their mean. Relaxed cells stand on one lattice only to the spread shown, and the residual is what that costs; the window reports it beside the colouring whenever it exceeds a tenth of a per cent of the prediction.

The target of the one-lattice set is the height of the metal above the sheet, and the largest per-site contribution of the first cell is its **Ag** site, which is the metal itself: the colouring points at the atom the property belongs to.
