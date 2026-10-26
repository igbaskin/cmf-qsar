# What the kernel costs, and what each shortcut saves

Part B of the plan is about speed. Every pair of numbers below was measured in one process on one data set, with the fast path replaced by the slow one, so the two differ in nothing but the shortcut. The machine is the one that produced every other table in this repository.

## One kernel, 6 attenuations

Time to build the per-field kernels of a set at every attenuation of the grid (0.1, 0.3, 0.5, 1, 2, 5), and how far each row is from the first. The error column is the point: all three shortcuts are exact, and the figures are round-off from summing the atom pairs in a different order.

**COX2** -- 200 molecules, 40.8 atoms each

| shortcut | seconds | speed-up | differs from the first row by |
|---|---:|---:|---:|
| as before | 14.33 | 1.00x | 0e+00 |
| + square root | 10.53 | 1.36x | 3e-16 |
| + symmetry | 4.60 | 3.12x | 8e-16 |
| + shared distances | 3.85 | 3.73x | 8e-16 |

**AMPH1** -- 60 molecules, 168.2 atoms each

| shortcut | seconds | speed-up | differs from the first row by |
|---|---:|---:|---:|
| as before | 42.24 | 1.00x | 0e+00 |
| + square root | 38.74 | 1.09x | 3e-16 |
| + symmetry | 16.02 | 2.64x | 1e-15 |
| + shared distances | 15.34 | 2.75x | 1e-15 |

## The same, end to end

`grid_search` over the same 6 attenuations on 200 molecules of COX2, 5-fold cross-validation. The attenuation chosen and the cross-validated q² are there to show that nothing moved.

| shortcut | seconds | speed-up | alpha chosen | q²cv |
|---|---:|---:|---:|---:|
| as before | 13.8 | 1.00x | 0.1 | +0.504175 |
| + square root | 9.9 | 1.40x | 0.1 | +0.504175 |
| + symmetry | 5.0 | 2.76x | 0.1 | +0.504175 |
| + shared distances | 4.1 | 3.40x | 0.1 | +0.504175 |

**3.4 times faster**, and the selection is identical to six decimal places.

## The cutoff (item B2), which pays only sometimes

A Gaussian overlap falls below 1e-06 of its peak beyond `sqrt(-4 ln tol / b)`, so pairs further apart can be dropped and the sum run over a neighbour list. Whether that is worth doing depends on the molecules. `auto` decides per field by sampling the real distances; `forced` uses the list whatever the geometry.

| set | atoms per molecule | alpha | pairs kept, Gaussian | pairs kept, steric | dense | auto | forced | auto is |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| COX2 | 40.8 | 0.5 | 98% | 100% | 0.95 s | 0.85 s (1.12x) | 4.13 s (0.23x) | the exact sum |
| COX2 | 40.8 | 2 | 56% | 96% | 0.90 s | 0.91 s (0.99x) | 2.86 s (0.31x) | the exact sum |
| COX2 | 40.8 | 5 | 24% | 67% | 0.84 s | 0.82 s (1.02x) | 1.66 s (0.51x) | the exact sum |
| AMPH1 | 168.2 | 0.5 | 62% | 99% | 2.61 s | 2.64 s (0.99x) | 6.92 s (0.38x) | the exact sum |
| AMPH1 | 168.2 | 2 | 14% | 52% | 2.75 s | 2.15 s (1.28x) | 2.70 s (1.02x) | the neighbour list |
| AMPH1 | 168.2 | 5 | 4% | 19% | 2.64 s | 1.21 s (2.18x) | 1.01 s (2.61x) | the neighbour list |

So the cutoff is not a speed-up of the program; it is a speed-up of one case. On a congeneric series superimposed in one frame almost every atom pair is inside the cutoff and `auto` correctly refuses to build a list. On molecules five times longer at a strong attenuation it is worth 2.18 times, and the error it introduces is 2e-07 relative.

## Screening a library in blocks (item B3)

The same library scored against a 200-molecule model, once in blocks and once loaded whole. Peak process memory is the figure to read: the time is similar, and the point of streaming is that the memory does not grow with the file.

| library | how | seconds | molecules per second | peak process memory |
|---|---|---:|---:|---:|
| 4166 molecules (22 MB) | streamed, 500 per block | 48 | 86 | 295 MB |
| 4166 molecules (22 MB) | streamed, 100 per block | 59 | 71 | 232 MB |
| 4166 molecules (22 MB) | loaded whole | 46 | 90 | 571 MB |
| 16664 molecules (87 MB) | streamed, 500 per block | 201 | 83 | 288 MB |
| 16664 molecules (87 MB) | streamed, 100 per block | 237 | 70 | 237 MB |
| 16664 molecules (87 MB) | loaded whole | 178 | 94 | 1706 MB |

Between the two library sizes (4166 and 16664 molecules) the whole-file path grows from 571 to 1706 MB and the streamed one from 232 to 237 MB. The streamed figure is what makes a library of any length possible on a desktop.

## The compiled kernel (item B5)

The same arithmetic as the NumPy path, written as one fused loop over the atom pairs and compiled with Numba, threaded over the molecules on 32 cores. It is **not** faster per core -- NumPy's exponential is vectorised and a compiled scalar loop is not, so serially the compiled loop loses -- and all of its advantage is the other cores.

| set | atoms per molecule | NumPy | compiled | speed-up | differs by |
|---|---:|---:|---:|---:|---:|
| COX2 | 40.8 | 0.50 s | 0.09 s | 5.57x | 4e-15 |
| AMPH1 | 168.2 | 1.99 s | 0.12 s | 16.56x | 2e-14 |

A whole grid search: 2.9 s against 0.7 s (3.99x), choosing the same attenuation (0.1) and the same q² to six decimals.

A screening run over 4166 molecules: 46 s against 12 s (3.82x, 91 against 347 molecules per second). The factor is smaller than on the kernel alone because reading the file and building the fields are now the larger half of the work.

NumPy stays the default everywhere except `screen`, which has no published numbers to reproduce. The two paths differ by about 4e-15 relative, which is the order of the round-off between two orderings of the NumPy sum itself, but a default that depends on whether an optional package is installed is not something a table should rest on.

