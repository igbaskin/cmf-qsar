# Conceptual-DFT fields: on a grid, fitted to Gaussians, condensed on atoms

Per series and field, the relative error of the Gaussian fit, the correlation of the grid kernel with the fitted one, and the cross-validated q² of the field alone and added to the five classical fields (`base`), for the field kept on the grid, fitted to atom-centred Gaussians, and condensed to one index per atom.

## 161_c1ccc2scnc2c1 (20 molecules, 100³ cubes)

Five classical fields alone: q² 0.69 (RMSEcv 1.06, α 0.3, λ 0.001).

| field | fit error | r(grid, fitted) | grid | base + grid | fitted | base + fitted | condensed | base + condensed |
|---|---|---|---|---|---|---|---|---|
| fukui_plus | 0.81 | +0.93 | 0.64 | 0.67 | 0.64 | 0.66 | 0.55 | 0.68 |
| dual | 0.91 | +0.90 | 0.52 | 0.63 | 0.56 | 0.60 | 0.45 | 0.50 |
