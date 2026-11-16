# Grid fields from DFT against the fitted atom-centred fields (plan item M9)

## Crystals

1462 ABO3 perovskite cells of item M4, formation energy of the cell (mean +1.431 eV, spread 0.713 eV). Grids from one static PBE calculation per cell in VASP, on 24 x 24 x 24 points of the cell. Five folds repeated three times; the setting and the ridge chosen by the evidence on the training block.

| Kernel | Frame | R², mean of 3 repeats | RMSE, eV | Settings chosen (folds) |
|---|---|---:|---:|---|
| grid radial | none (radial) | +0.345 | 0.5767 | sigma 0.35, ridge 0.01 (15) |
| fitted grid radial | none (radial) | +0.398 | 0.5531 | sigma 0.35, ridge 0.001 (13), sigma 0.35, ridge 0.01 (2) |
| fields radial | none (radial) | +0.443 | 0.5320 | sigma 0.5, ridge 0.01 (4), sigma 0.75, ridge 0.01 (11) |
| fields and grid radial | none (radial) | +0.543 | 0.4818 | sigma 0.5, ridge 0.01 (2), sigma 0.75, ridge 0.001 (4), sigma 0.75, ridge 0.01 (9) |
| grid overlap | one lattice | +0.825 | 0.2977 | ridge 0.001 (15) |
| fitted grid overlap | one lattice | +0.615 | 0.4422 | ridge 0.01 (15) |
| fields overlap | one lattice | +0.424 | 0.5409 | alpha 8, ridge 0.1 (15) |
| fields and grid overlap | one lattice | +0.815 | 0.3061 | alpha 1, ridge 0.001 (5), alpha 2, ridge 0.001 (10) |
| composition | none (element counts) | +0.362 | 0.5693 | ridge 1 (15) |
| grid density overlap | one lattice | +0.513 | 0.4971 | ridge 0.0001 (15) |
| grid elf overlap | one lattice | +0.646 | 0.4241 | ridge 0.001 (15) |
| grid potential overlap | one lattice | +0.644 | 0.4251 | ridge 0.001 (15) |
| grid fitted with seven widths radial | none (radial) | +0.345 | 0.5769 | sigma 0.35, ridge 0.01 (15) |
| grid fitted with seven widths overlap | one lattice | +0.754 | 0.3532 | ridge 0.001 (15) |

| Difference of R² | What it decides | Mean [95 % interval] | Interval above zero |
|---|---|---|---|
| grid radial minus fields radial | second question (DFT grid against the empirical fields) | -0.105 [-0.153, -0.055] | no |
| grid overlap minus fields overlap | second question (DFT grid against the empirical fields) | +0.409 [+0.365, +0.457] | yes |
| grid radial minus fitted grid radial | the gate (grid against its own atom-centred fit) | -0.055 [-0.072, -0.037] | no |
| grid overlap minus fitted grid overlap | the gate (grid against its own atom-centred fit) | +0.216 [+0.182, +0.248] | yes |
| fields and grid radial minus fields radial | whether the grid adds to the fields | +0.102 [+0.068, +0.134] | yes |
| fields and grid overlap minus fields overlap | whether the grid adds to the fields | +0.402 [+0.357, +0.451] | yes |
| grid overlap minus composition | whether the grid knows more than the elements | +0.472 [+0.433, +0.510] | yes |
| grid radial minus grid fitted with seven widths radial | check of the gate (a richer atom-centred fit) | +0.000 [-0.004, +0.005] | no |
| grid overlap minus grid fitted with seven widths overlap | check of the gate (a richer atom-centred fit) | +0.069 [+0.045, +0.100] | yes |

Relative error of the atom-centred fit, mean over the cells: density 8.6%, elf 20.6%, potential 20.5%, with seven widths density 3.2%, elf 12.7%, potential 7.1%. Mean electron count of the density grids: 31.10.

Correlation of the off-diagonal kernel values: grid_overlap / fields_overlap +0.857; grid_overlap / fitted_overlap +0.999; grid_radial / fields_radial +0.487; grid_radial / fitted_radial +0.997.

### The energy of the same run as the reference

The grid of a cell costs one DFT calculation, and that calculation also gives a total energy. The reference is a ridge regression on that energy and the counts of the 56 elements, on the same folds; it needs nothing the grid does not already need.

| Model | R², mean of 3 repeats | RMSE, eV | Ridge chosen (folds) |
|---|---:|---:|---|
| energy of the run | +0.380 | 0.5611 | ridge 1 (15) |
| energy of the run and element counts | +0.940 | 0.1749 | ridge 1 (15) |
| grid overlap | +0.825 | 0.2977 | ridge 0.001 (15) |
| grid overlap and energy of the run | +0.953 | 0.1545 | ridge 0.01 (15) |

| Difference of R² | Mean [95 % interval] |
|---|---|
| energy of the run and element counts minus grid overlap | +0.103 [+0.084, +0.127] |
| grid overlap and energy of the run minus energy of the run and element counts | +0.014 [+0.011, +0.017] |

Slope of the target on the energy of the run, in one least-squares fit with the element counts: 0.172.

## Molecules

The four corrosion-inhibitor series with ORCA single points. Grids of the neutral molecule from orca_plot on 61³ points; the density without its cores, the potential outside the molecule. q² from five folds repeated three times.

| Series | Molecules | Grid overlap | Fitted grid overlap | Fields overlap | Grid radial | Fitted grid radial | Fields radial |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7_c1nncs1 | 29 | -0.159 | -0.170 | -0.275 | -0.165 | -0.225 | -0.562 |
| 12_c1ccoc1 | 27 | -0.097 | +0.058 | -0.133 | -0.019 | -0.031 | -0.269 |
| 13_c1c_nH_nn1 | 25 | +0.600 | +0.607 | +0.602 | +0.222 | +0.065 | +0.431 |
| 161_c1ccc2scnc2c1 | 20 | +0.562 | +0.635 | +0.580 | +0.331 | +0.443 | +0.238 |

| Series | Grid overlap minus fields overlap | Grid radial minus fields radial | Grid overlap minus fitted grid overlap | Grid radial minus fitted grid radial |
|---|---|---|---|---|
| 7_c1nncs1 | +0.077 [-0.106, +0.293] | +0.321 [+0.031, +1.008] | +0.011 [-0.003, +0.034] | +0.057 [-0.054, +0.257] |
| 12_c1ccoc1 | +0.000 [-0.327, +0.597] | +0.186 [-0.000, +0.452] | -0.152 [-0.303, -0.062] | +0.010 [-0.054, +0.051] |
| 13_c1c_nH_nn1 | +0.005 [-0.235, +0.186] | -0.245 [-0.653, +0.178] | -0.004 [-0.052, +0.049] | +0.112 [-0.088, +0.385] |
| 161_c1ccc2scnc2c1 | -0.058 [-0.270, +0.131] | +0.050 [-0.288, +0.526] | -0.069 [-0.215, +0.035] | -0.073 [-0.240, +0.086] |
