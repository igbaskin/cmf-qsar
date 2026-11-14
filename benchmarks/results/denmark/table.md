# A kinetic selectivity as the activity: the Denmark catalyst series

43 catalysts x 25 substrate combinations = 1075 reactions; ddG sd 0.701 kcal/mol. Both C2 images per catalyst. Published: MAD 0.211 kcal/mol on external catalysts, 0.161 on new substrate combinations (Zahrt et al. 2019, a 600/475 design).

| Held out | Model | MAD | RMSE | q2 | Baseline MAD | Baseline q2 | MAD difference [95 %] |
|---|---|---|---|---|---|---|---|
| catalysts | one correlation | 0.224 | 0.296 | +0.821 | 0.458 | +0.168 | -0.236 [-0.341, -0.143] |
| catalysts | substrate covariance | 0.231 | 0.311 | +0.803 | 0.458 | +0.168 | -0.229 [-0.329, -0.138] |
| reactions | one correlation | 0.164 | 0.230 | +0.892 | 0.183 | +0.871 | -0.020 [-0.034, -0.006] |
| reactions | substrate covariance | 0.149 | 0.215 | +0.906 | 0.183 | +0.871 | -0.035 [-0.057, -0.014] |
| substrates | substrate covariance | 0.158 | 0.222 | +0.899 | 0.186 | +0.867 | -0.028 [-0.050, -0.007] |
| imines | substrate covariance | 0.167 | 0.244 | +0.879 | 0.185 | +0.867 | -0.018 [-0.039, +0.000] |
| thiols | substrate covariance | 0.406 | 0.481 | +0.529 | 0.399 | +0.535 | +0.007 [+0.003, +0.010] |

Baselines: catalysts: endpoint mean over the training catalysts; reactions: catalyst effect + substrate effect, least squares; substrates: catalyst effect + imine effect + thiol effect, least squares; imines: catalyst effect + imine effect + thiol effect, least squares; thiols: catalyst effect + imine effect + thiol effect, least squares

## Which field carries the selectivity (catalysts held out, one correlation)

| Kernel | MAD | RMSE | q2 | alpha | lambda | rho |
|---|---|---|---|---|---|---|
| electrostatic | 0.404 | 0.570 | +0.339 | 3 | 0.01 | +0.990 |
| steric | 0.250 | 0.346 | +0.756 | 3 | 0.01 | +0.950 |
| hydrophobic | 0.308 | 0.426 | +0.630 | 0.8 | 0.01 | +0.990 |
| hba | 0.418 | 0.627 | +0.199 | 3 | 0.001 | +0.990 |
| hbd | 0.484 | 0.630 | +0.192 | 3 | 0.1 | +0.990 |
| invariant | 0.372 | 0.489 | +0.512 | 0.3 | 0.01 | +0.990 |
