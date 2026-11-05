# A kinetic selectivity as the activity: the Denmark catalyst series

43 catalysts x 25 substrate combinations = 1075 reactions; ddG sd 0.701 kcal/mol. Both C2 images per catalyst. Published: MAD 0.211 kcal/mol on external catalysts, 0.161 on new substrate combinations (Zahrt et al. 2019, a 600/475 design).

| Held out | Model | MAD | RMSE | q2 | Baseline MAD | Baseline q2 | MAD difference [95 %] |
|---|---|---|---|---|---|---|---|
| catalysts | one correlation | 0.224 | 0.296 | +0.822 | 0.458 | +0.168 | -0.236 [-0.340, -0.142] |
| catalysts | substrate covariance | 0.230 | 0.311 | +0.803 | 0.458 | +0.168 | -0.229 [-0.329, -0.137] |
| reactions | one correlation | 0.164 | 0.230 | +0.892 | 0.183 | +0.871 | -0.020 [-0.034, -0.006] |
| reactions | substrate covariance | 0.148 | 0.215 | +0.906 | 0.183 | +0.871 | -0.035 [-0.057, -0.015] |
| substrates | substrate covariance | 0.158 | 0.222 | +0.899 | 0.186 | +0.867 | -0.027 [-0.049, -0.007] |
| imines | substrate covariance | 0.167 | 0.243 | +0.880 | 0.185 | +0.867 | -0.018 [-0.039, -0.000] |
| thiols | substrate covariance | 0.406 | 0.481 | +0.529 | 0.399 | +0.535 | +0.007 [+0.004, +0.010] |

Baselines: catalysts: endpoint mean over the training catalysts; reactions: catalyst effect + substrate effect, least squares; substrates: catalyst effect + imine effect + thiol effect, least squares; imines: catalyst effect + imine effect + thiol effect, least squares; thiols: catalyst effect + imine effect + thiol effect, least squares

## Which field carries the selectivity (catalysts held out, one correlation)

| Kernel | MAD | RMSE | q2 | alpha | lambda | rho |
|---|---|---|---|---|---|---|
| electrostatic | 0.376 | 0.540 | +0.405 | 3 | 0.01 | +0.990 |
| steric | 0.249 | 0.344 | +0.760 | 3 | 0.01 | +0.950 |
| hydrophobic | 0.289 | 0.429 | +0.626 | 0.8 | 0.01 | +0.990 |
| hba | 0.588 | 1.465 | -3.373 | 3 | 0.001 | +0.990 |
| hbd | 0.490 | 0.632 | +0.185 | 3 | 0.1 | +0.990 |
| invariant | 0.419 | 0.604 | +0.255 | 0.3 | 0.01 | +0.990 |
