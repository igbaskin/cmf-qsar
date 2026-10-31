# Continuous molecular fields of two-dimensional materials: JARVIS-DFT 2D

1,103 monolayers (JARVIS-DFT 2D, file of 12 December 2022), 5-fold cross-validation (shuffled, seed 0); the width of the channels and the regularisation chosen by 10-fold cross-validation inside every training fold. MAE mean +/- sd over the folds, RMSE and R2 on the pooled held-out predictions; 'mean' is the MAE of predicting the training mean. Vacuum sensitivity: median relative change of the channel vector when the vacuum of the cell is doubled (at the smaller channel width).

| target | variant | flags | background | heights | fields | channels | vacuum sensitivity | MAE | mean | RMSE | R2 | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OptB88vdW band gap (eV) | crystal-bg | crystal | per volume | no | 4 | 210 | 5.0e-01 | 0.82 +/- 0.038 | 1.05 | 1.09 | 0.271 | 85 |
| OptB88vdW band gap (eV) | nobg | layer | kept | no | 4 | 210 | 3.0e-16 | 0.819 +/- 0.029 | 1.05 | 1.09 | 0.276 | 87 |
| OptB88vdW band gap (eV) | layer-bg | layer | per area | no | 4 | 210 | 1.4e-15 | 0.829 +/- 0.029 | 1.05 | 1.1 | 0.267 | 85 |
| OptB88vdW band gap (eV) | heights | layer | per area | +/- 3 A every 1 | 4 | 10290 | 9.6e-16 | 0.785 +/- 0.032 | 1.05 | 1.07 | 0.298 | 14 |
| OptB88vdW band gap (eV) | heights-oriented | layer | per area | +/- 3 A every 1, oriented | 4 | 10290 | 2.0e-15 | 0.789 +/- 0.035 | 1.05 | 1.08 | 0.291 | 112 |
| OptB88vdW band gap (eV) | all-crystal-bg | crystal | per volume | no | 66 | 46431 | 5.0e-01 | 0.68 +/- 0.025 | 1.05 | 0.99 | 0.404 | 146 |
| OptB88vdW band gap (eV) | all-nobg | layer | kept | no | 66 | 46431 | 3.1e-16 | 0.679 +/- 0.029 | 1.05 | 1 | 0.391 | 85 |
| OptB88vdW band gap (eV) | all-layer-bg | layer | per area | no | 66 | 46431 | 1.4e-15 | 0.732 +/- 0.025 | 1.05 | 1.04 | 0.337 | 125 |
| OptB88vdW band gap (eV) | layer-bg-gauss | layer | per area | no | 4 | 210 | 1.4e-15 | 0.68 +/- 0.04 | 1.05 | 1.01 | 0.385 | 324 |
| OptB88vdW band gap (eV) | heights-gauss | layer | per area | +/- 3 A every 1 | 4 | 10290 | 9.6e-16 | 0.692 +/- 0.023 | 1.05 | 1 | 0.387 | 312 |
| OptB88vdW band gap (eV) | all-layer-bg-gauss | layer | per area | no | 66 | 46431 | 1.4e-15 | 0.719 +/- 0.035 | 1.05 | 1.04 | 0.343 | 360 |
| OptB88vdW band gap (eV) | crystal-bg-gauss | crystal | per volume | no | 4 | 210 | 5.0e-01 | 0.67 +/- 0.047 | 1.05 | 1 | 0.389 | 104 |
| OptB88vdW band gap (eV) | nobg-gauss | layer | kept | no | 4 | 210 | 3.0e-16 | 0.668 +/- 0.046 | 1.05 | 0.988 | 0.406 | 96 |
| OptB88vdW band gap (eV) | all-crystal-bg-gauss | crystal | per volume | no | 66 | 46431 | 5.0e-01 | 0.675 +/- 0.032 | 1.05 | 1 | 0.386 | 113 |
| OptB88vdW band gap (eV) | all-nobg-gauss | layer | kept | no | 66 | 46431 | 3.1e-16 | 0.67 +/- 0.034 | 1.05 | 1.03 | 0.357 | 112 |
| formation energy per atom (eV/atom) | crystal-bg | crystal | per volume | no | 4 | 210 | 5.0e-01 | 0.329 +/- 0.02 | 0.698 | 0.489 | 0.703 | 61 |
| formation energy per atom (eV/atom) | nobg | layer | kept | no | 4 | 210 | 3.0e-16 | 0.349 +/- 0.031 | 0.698 | 0.569 | 0.599 | 63 |
| formation energy per atom (eV/atom) | layer-bg | layer | per area | no | 4 | 210 | 1.4e-15 | 0.37 +/- 0.044 | 0.698 | 0.746 | 0.310 | 63 |
| formation energy per atom (eV/atom) | heights | layer | per area | +/- 3 A every 1 | 4 | 10290 | 9.6e-16 | 0.349 +/- 0.027 | 0.698 | 0.525 | 0.658 | 71 |
| formation energy per atom (eV/atom) | heights-oriented | layer | per area | +/- 3 A every 1, oriented | 4 | 10290 | 2.0e-15 | 0.349 +/- 0.028 | 0.698 | 0.526 | 0.657 | 73 |
| formation energy per atom (eV/atom) | all-crystal-bg | crystal | per volume | no | 66 | 46431 | 5.0e-01 | 0.259 +/- 0.019 | 0.698 | 0.453 | 0.745 | 114 |
| formation energy per atom (eV/atom) | all-nobg | layer | kept | no | 66 | 46431 | 3.1e-16 | 0.271 +/- 0.015 | 0.698 | 0.467 | 0.729 | 116 |
| formation energy per atom (eV/atom) | all-layer-bg | layer | per area | no | 66 | 46431 | 1.4e-15 | 0.311 +/- 0.026 | 0.698 | 0.518 | 0.667 | 118 |
| formation energy per atom (eV/atom) | layer-bg-gauss | layer | per area | no | 4 | 210 | 1.4e-15 | 0.283 +/- 0.01 | 0.698 | 0.482 | 0.711 | 325 |
| formation energy per atom (eV/atom) | heights-gauss | layer | per area | +/- 3 A every 1 | 4 | 10290 | 9.6e-16 | 0.286 +/- 0.017 | 0.698 | 0.487 | 0.706 | 255 |
| formation energy per atom (eV/atom) | all-layer-bg-gauss | layer | per area | no | 66 | 46431 | 1.4e-15 | 0.312 +/- 0.026 | 0.698 | 0.52 | 0.664 | 366 |
| formation energy per atom (eV/atom) | crystal-bg-gauss | crystal | per volume | no | 4 | 210 | 5.0e-01 | 0.269 +/- 0.022 | 0.698 | 0.436 | 0.764 | 95 |
| formation energy per atom (eV/atom) | nobg-gauss | layer | kept | no | 4 | 210 | 3.0e-16 | 0.274 +/- 0.019 | 0.698 | 0.45 | 0.748 | 101 |
| formation energy per atom (eV/atom) | all-crystal-bg-gauss | crystal | per volume | no | 66 | 46431 | 5.0e-01 | 0.259 +/- 0.019 | 0.698 | 0.454 | 0.744 | 117 |
| formation energy per atom (eV/atom) | all-nobg-gauss | layer | kept | no | 66 | 46431 | 3.1e-16 | 0.269 +/- 0.017 | 0.698 | 0.472 | 0.724 | 117 |
| exfoliation energy per atom (meV/atom) | crystal-bg | crystal | per volume | no | 4 | 210 | 5.0e-01 | 66.4 +/- 7.9 | 71 | 130 | 0.119 | 29 |
| exfoliation energy per atom (meV/atom) | nobg | layer | kept | no | 4 | 210 | 3.0e-16 | 68.3 +/- 7.6 | 71 | 133 | 0.088 | 29 |
| exfoliation energy per atom (meV/atom) | layer-bg | layer | per area | no | 4 | 210 | 1.4e-15 | 74.7 +/- 9.8 | 71 | 140 | -0.009 | 26 |
| exfoliation energy per atom (meV/atom) | heights | layer | per area | +/- 3 A every 1 | 4 | 10290 | 9.6e-16 | 63.7 +/- 4.5 | 71 | 126 | 0.179 | 34 |
| exfoliation energy per atom (meV/atom) | heights-oriented | layer | per area | +/- 3 A every 1, oriented | 4 | 10290 | 2.0e-15 | 63.9 +/- 2.6 | 71 | 122 | 0.232 | 35 |
| exfoliation energy per atom (meV/atom) | all-crystal-bg | crystal | per volume | no | 66 | 46431 | 5.0e-01 | 64.4 +/- 7.2 | 71 | 139 | 0.005 | 64 |
| exfoliation energy per atom (meV/atom) | all-nobg | layer | kept | no | 66 | 46431 | 3.1e-16 | 67 +/- 7 | 71 | 141 | -0.022 | 61 |
| exfoliation energy per atom (meV/atom) | all-layer-bg | layer | per area | no | 66 | 46431 | 1.4e-15 | 64.3 +/- 8.1 | 71 | 134 | 0.066 | 47 |
| exfoliation energy per atom (meV/atom) | layer-bg-gauss | layer | per area | no | 4 | 210 | 1.4e-15 | 56.1 +/- 5.2 | 71 | 129 | 0.144 | 42 |
| exfoliation energy per atom (meV/atom) | heights-gauss | layer | per area | +/- 3 A every 1 | 4 | 10290 | 9.6e-16 | 51.2 +/- 3.6 | 71 | 116 | 0.303 | 46 |
| exfoliation energy per atom (meV/atom) | all-layer-bg-gauss | layer | per area | no | 66 | 46431 | 1.4e-15 | 63.3 +/- 9.3 | 71 | 136 | 0.043 | 60 |
| exfoliation energy per atom (meV/atom) | crystal-bg-gauss | crystal | per volume | no | 4 | 210 | 5.0e-01 | 53.5 +/- 5.8 | 71 | 121 | 0.242 | 37 |
| exfoliation energy per atom (meV/atom) | nobg-gauss | layer | kept | no | 4 | 210 | 3.0e-16 | 53.7 +/- 5.1 | 71 | 124 | 0.200 | 37 |
| exfoliation energy per atom (meV/atom) | all-crystal-bg-gauss | crystal | per volume | no | 66 | 46431 | 5.0e-01 | 61.1 +/- 7.4 | 71 | 137 | 0.036 | 50 |
| exfoliation energy per atom (meV/atom) | all-nobg-gauss | layer | kept | no | 66 | 46431 | 3.1e-16 | 62.6 +/- 6.1 | 71 | 138 | 0.013 | 48 |

## Co-pattern of `crystal-bg` on the OptB88vdW band gap (sigma 0.5 A, lambda 0.1)

| effect | channel |
|---|---|
| +0.244 | Electrostatic x Steric @ 8.50 A |
| +0.210 | Electrostatic x inverse_sqrt_mass @ 7.50 A |
| -0.199 | Electrostatic x electronegativity @ 7.50 A |
| -0.186 | Electrostatic @ 5.50 A |
| +0.184 | Electrostatic x Steric @ 4.00 A |
| -0.168 | Electrostatic @ 8.00 A |
| +0.160 | electronegativity @ 4.00 A |
| +0.146 | Electrostatic x electronegativity @ 1.50 A |
| -0.145 | electronegativity @ 2.50 A |
| +0.143 | Electrostatic @ 8.50 A |

## Co-pattern of `nobg` on the OptB88vdW band gap (sigma 0.5 A, lambda 0.03)

| effect | channel |
|---|---|
| +0.417 | Electrostatic x Steric @ 8.50 A |
| +0.306 | Electrostatic x Steric @ 4.00 A |
| -0.276 | Electrostatic x Steric @ 9.00 A |
| -0.270 | Electrostatic x electronegativity @ 7.50 A |
| -0.247 | Electrostatic x inverse_sqrt_mass @ 3.00 A |
| -0.244 | Electrostatic @ 5.50 A |
| +0.242 | Electrostatic x inverse_sqrt_mass @ 7.50 A |
| +0.209 | Electrostatic x electronegativity @ 3.50 A |
| -0.205 | Electrostatic x Steric @ 4.50 A |
| -0.203 | Electrostatic x Steric @ 2.00 A |

## Co-pattern of `layer-bg` on the OptB88vdW band gap (sigma 0.75 A, lambda 0.3)

| effect | channel |
|---|---|
| -0.247 | Electrostatic @ 6.00 A |
| -0.237 | electronegativity x inverse_sqrt_mass @ 9.00 A |
| -0.213 | electronegativity x inverse_sqrt_mass @ 8.25 A |
| -0.194 | electronegativity @ 5.25 A |
| -0.181 | electronegativity @ 7.50 A |
| -0.169 | electronegativity @ 2.25 A |
| +0.167 | Steric x electronegativity @ 4.50 A |
| -0.163 | electronegativity @ 6.75 A |
| +0.162 | Electrostatic x Steric @ 8.25 A |
| +0.152 | Electrostatic x Steric @ 3.75 A |

## Co-pattern of `heights` on the OptB88vdW band gap (sigma 0.75 A, lambda 0.3)

| effect | channel |
|---|---|
| +0.057 | Electrostatic x Steric @ rho 8.25 A, z +0.0 / +0.0 A |
| -0.040 | Electrostatic x electronegativity @ rho 3.00 A, z +0.0 / +0.0 A |
| +0.036 | Electrostatic @ rho 6.75 A, z +0.0 / +0.0 A |
| +0.035 | Electrostatic x electronegativity @ rho 1.50 A, z +0.0 / +0.0 A |
| -0.034 | Electrostatic @ rho 8.25 A, z +0.0 / +0.0 A |
| -0.033 | Electrostatic @ rho 6.00 A, z +0.0 / +0.0 A |
| +0.032 | Electrostatic x Steric @ rho 9.00 A, z +0.0 / +0.0 A |
| -0.031 | Electrostatic x inverse_sqrt_mass @ rho 3.75 A, z +0.0 / +0.0 A |
| +0.030 | Electrostatic @ rho 9.00 A, z +0.0 / +0.0 A |
| +0.029 | Electrostatic x inverse_sqrt_mass @ rho 6.75 A, z +0.0 / +0.0 A |

## Co-pattern of `heights-oriented` on the OptB88vdW band gap (sigma 0.75 A, lambda 0.3)

| effect | channel |
|---|---|
| +0.056 | Electrostatic x Steric @ rho 8.25 A, z +0.0 / +0.0 A |
| -0.040 | Electrostatic x electronegativity @ rho 3.00 A, z +0.0 / +0.0 A |
| +0.035 | Electrostatic x electronegativity @ rho 1.50 A, z +0.0 / +0.0 A |
| -0.035 | Electrostatic @ rho 6.00 A, z +0.0 / +0.0 A |
| +0.034 | Electrostatic @ rho 6.75 A, z +0.0 / +0.0 A |
| -0.034 | Electrostatic @ rho 8.25 A, z +0.0 / +0.0 A |
| +0.033 | Electrostatic x Steric @ rho 9.00 A, z +0.0 / +0.0 A |
| -0.030 | Electrostatic x inverse_sqrt_mass @ rho 3.75 A, z +0.0 / +0.0 A |
| +0.030 | electronegativity @ rho 8.25 A, z -1.0 / -1.0 A |
| +0.030 | Electrostatic @ rho 9.00 A, z +0.0 / +0.0 A |

## Co-pattern of `all-crystal-bg` on the OptB88vdW band gap (sigma 0.5 A, lambda 0.3)

| effect | channel |
|---|---|
| +0.038 | CIF Br x CIF Y @ 6.50 A |
| +0.030 | CIF Cl x CIF other @ 6.00 A |
| -0.030 | Electrostatic x CIF I @ 8.00 A |
| +0.027 | CIF Al x CIF Cl @ 7.50 A |
| -0.026 | Electrostatic x CIF other @ 7.50 A |
| -0.025 | Electrostatic x CIF other @ 8.00 A |
| -0.025 | CIF Cl x CIF other @ 5.50 A |
| +0.024 | CIF Al x CIF Cl @ 7.00 A |
| +0.024 | Electrostatic x CIF other @ 3.50 A |
| -0.024 | CIF Cl x CIF other @ 5.00 A |

## Co-pattern of `all-nobg` on the OptB88vdW band gap (sigma 0.5 A, lambda 0.3)

| effect | channel |
|---|---|
| -0.036 | Electrostatic x CIF I @ 8.00 A |
| -0.032 | Electrostatic x CIF other @ 7.50 A |
| +0.031 | CIF Br x CIF Y @ 6.50 A |
| +0.028 | Electrostatic x CIF other @ 3.50 A |
| -0.027 | Electrostatic x CIF other @ 8.00 A |
| +0.027 | CIF Al x CIF Cl @ 7.50 A |
| +0.027 | Electrostatic x CIF I @ 7.00 A |
| -0.025 | CIF Cl x CIF other @ 8.50 A |
| +0.024 | CIF Al x CIF Cl @ 7.00 A |
| +0.024 | Electrostatic x CIF I @ 6.50 A |

## Co-pattern of `all-layer-bg` on the OptB88vdW band gap (sigma 0.5 A, lambda 0.3)

| effect | channel |
|---|---|
| -0.035 | CIF Br x CIF Y @ 7.50 A |
| -0.034 | CIF Cl x CIF other @ 8.00 A |
| +0.026 | CIF Br x CIF Y @ 6.50 A |
| -0.025 | Electrostatic x CIF I @ 8.00 A |
| -0.025 | CIF Cl x CIF other @ 8.50 A |
| -0.022 | CIF B x CIF S @ 8.50 A |
| +0.022 | CIF Cl x CIF other @ 6.00 A |
| -0.022 | electronegativity x CIF other @ 8.50 A |
| -0.021 | Electrostatic x CIF other @ 7.50 A |
| +0.021 | CIF Al x CIF Cl @ 7.50 A |

## Co-pattern of `crystal-bg` on the formation energy per atom (sigma 0.75 A, lambda 0.01)

| effect | channel |
|---|---|
| +0.520 | electronegativity x inverse_sqrt_mass @ 3.75 A |
| -0.487 | Electrostatic x electronegativity @ 1.50 A |
| -0.453 | electronegativity x inverse_sqrt_mass @ 6.00 A |
| -0.439 | electronegativity @ 4.50 A |
| +0.407 | Electrostatic x electronegativity @ 2.25 A |
| -0.336 | electronegativity x inverse_sqrt_mass @ 6.75 A |
| -0.332 | Steric @ 3.75 A |
| -0.330 | Electrostatic x Steric @ 5.25 A |
| +0.319 | Electrostatic x electronegativity @ 3.75 A |
| -0.314 | Electrostatic x Steric @ 0.00 A |

## Co-pattern of `nobg` on the formation energy per atom (sigma 0.75 A, lambda 0.01)

| effect | channel |
|---|---|
| -0.521 | Electrostatic x electronegativity @ 1.50 A |
| -0.458 | Steric @ 3.75 A |
| +0.388 | Electrostatic x electronegativity @ 2.25 A |
| +0.375 | Steric @ 5.25 A |
| -0.370 | electronegativity x inverse_sqrt_mass @ 6.75 A |
| -0.349 | Electrostatic x Steric @ 5.25 A |
| -0.329 | Electrostatic x Steric @ 0.00 A |
| +0.323 | Electrostatic x electronegativity @ 3.75 A |
| -0.314 | electronegativity x inverse_sqrt_mass @ 6.00 A |
| +0.307 | Electrostatic x inverse_sqrt_mass @ 6.75 A |

## Co-pattern of `layer-bg` on the formation energy per atom (sigma 0.75 A, lambda 0.0001)

| effect | channel |
|---|---|
| +1.993 | Steric @ 1.50 A |
| +1.211 | Steric @ 0.00 A |
| -1.161 | electronegativity x inverse_sqrt_mass @ 6.00 A |
| +1.143 | electronegativity @ 6.00 A |
| -1.118 | electronegativity x inverse_sqrt_mass @ 4.50 A |
| -1.113 | Steric @ 0.75 A |
| -1.023 | Electrostatic x electronegativity @ 1.50 A |
| -1.019 | Electrostatic x electronegativity @ 3.00 A |
| -0.998 | Steric x inverse_sqrt_mass @ 1.50 A |
| +0.948 | electronegativity x inverse_sqrt_mass @ 3.75 A |

## Co-pattern of `heights` on the formation energy per atom (sigma 0.75 A, lambda 0.1)

| effect | channel |
|---|---|
| -0.052 | Steric @ rho 7.50 A, z +0.0 / +0.0 A |
| +0.051 | Electrostatic x Steric @ rho 1.50 A, z +0.0 / +0.0 A |
| -0.046 | Electrostatic x electronegativity @ rho 1.50 A, z +0.0 / +0.0 A |
| -0.041 | Electrostatic @ rho 9.00 A, z +0.0 / +0.0 A |
| +0.038 | electronegativity x inverse_sqrt_mass @ rho 7.50 A, z +0.0 / +0.0 A |
| +0.037 | Electrostatic x Steric @ rho 2.25 A, z +0.0 / +0.0 A |
| -0.032 | electronegativity x inverse_sqrt_mass @ rho 6.00 A, z +0.0 / +0.0 A |
| +0.032 | electronegativity x inverse_sqrt_mass @ rho 9.00 A, z +0.0 / +0.0 A |
| +0.031 | electronegativity @ rho 7.50 A, z +0.0 / +0.0 A |
| -0.031 | Electrostatic x electronegativity @ rho 3.75 A, z +0.0 / +0.0 A |

## Co-pattern of `heights-oriented` on the formation energy per atom (sigma 0.75 A, lambda 0.1)

| effect | channel |
|---|---|
| -0.052 | Steric @ rho 7.50 A, z +0.0 / +0.0 A |
| +0.050 | Electrostatic x Steric @ rho 1.50 A, z +0.0 / +0.0 A |
| -0.046 | Electrostatic x electronegativity @ rho 1.50 A, z +0.0 / +0.0 A |
| -0.041 | Electrostatic @ rho 9.00 A, z +0.0 / +0.0 A |
| +0.038 | electronegativity x inverse_sqrt_mass @ rho 7.50 A, z +0.0 / +0.0 A |
| +0.037 | Electrostatic x Steric @ rho 2.25 A, z +0.0 / +0.0 A |
| +0.033 | electronegativity @ rho 7.50 A, z +0.0 / +0.0 A |
| -0.032 | electronegativity x inverse_sqrt_mass @ rho 6.00 A, z +0.0 / +0.0 A |
| -0.032 | Electrostatic x electronegativity @ rho 3.75 A, z +0.0 / +0.0 A |
| +0.031 | electronegativity x inverse_sqrt_mass @ rho 9.00 A, z +0.0 / +0.0 A |

## Co-pattern of `all-crystal-bg` on the formation energy per atom (sigma 0.5 A, lambda 0.03)

| effect | channel |
|---|---|
| +0.038 | CIF N @ 8.50 A |
| -0.029 | CIF O x CIF other @ 7.50 A |
| -0.025 | CIF Br x CIF other @ 6.50 A |
| -0.023 | Steric x CIF O @ 4.50 A |
| -0.023 | CIF Cl x CIF other @ 6.00 A |
| +0.022 | CIF N @ 8.00 A |
| +0.021 | CIF Cl x CIF other @ 2.50 A |
| -0.021 | Steric x CIF O @ 7.50 A |
| +0.020 | inverse_sqrt_mass x CIF N @ 8.50 A |
| +0.019 | electronegativity x CIF N @ 8.50 A |

## Co-pattern of `all-nobg` on the formation energy per atom (sigma 0.5 A, lambda 0.1)

| effect | channel |
|---|---|
| +0.040 | CIF N @ 8.50 A |
| +0.033 | CIF N @ 8.00 A |
| -0.021 | Steric x CIF O @ 7.50 A |
| -0.020 | CIF O x CIF other @ 7.50 A |
| +0.020 | inverse_sqrt_mass x CIF N @ 8.50 A |
| +0.019 | electronegativity x CIF N @ 8.50 A |
| -0.018 | Steric x CIF O @ 8.00 A |
| -0.018 | Steric x CIF O @ 7.00 A |
| -0.017 | Electrostatic x CIF other @ 4.50 A |
| -0.016 | Electrostatic x CIF other @ 4.00 A |

## Co-pattern of `all-layer-bg` on the formation energy per atom (sigma 0.5 A, lambda 0.03)

| effect | channel |
|---|---|
| +0.031 | electronegativity x CIF N @ 8.50 A |
| +0.029 | inverse_sqrt_mass x CIF N @ 8.50 A |
| +0.025 | CIF N @ 8.50 A |
| +0.020 | CIF Cl x CIF other @ 8.00 A |
| -0.018 | Steric x CIF O @ 4.50 A |
| -0.018 | CIF O x CIF other @ 7.50 A |
| -0.018 | CIF C x CIF N @ 5.00 A |
| +0.017 | electronegativity x CIF N @ 8.00 A |
| +0.017 | Steric x CIF O @ 3.50 A |
| -0.017 | CIF C x CIF I @ 7.00 A |

## Co-pattern of `crystal-bg` on the exfoliation energy per atom (sigma 0.75 A, lambda 0.3)

| effect | channel |
|---|---|
| -32.804 | Electrostatic x Steric @ 3.75 A |
| -26.240 | Electrostatic x electronegativity @ 2.25 A |
| -19.312 | Electrostatic x Steric @ 6.00 A |
| -16.781 | Steric @ 6.75 A |
| -15.501 | Steric @ 7.50 A |
| -15.442 | Electrostatic x Steric @ 5.25 A |
| +14.582 | Steric @ 3.00 A |
| -14.241 | Electrostatic x inverse_sqrt_mass @ 7.50 A |
| -13.603 | Electrostatic x Steric @ 7.50 A |
| -13.133 | Electrostatic x Steric @ 1.50 A |

## Co-pattern of `nobg` on the exfoliation energy per atom (sigma 0.75 A, lambda 0.1)

| effect | channel |
|---|---|
| -59.642 | Electrostatic x Steric @ 3.75 A |
| +33.672 | Electrostatic x Steric @ 4.50 A |
| -32.365 | Electrostatic x electronegativity @ 2.25 A |
| -31.611 | Electrostatic x Steric @ 5.25 A |
| +28.471 | Electrostatic @ 6.75 A |
| +27.212 | Electrostatic @ 8.25 A |
| -26.864 | Electrostatic x inverse_sqrt_mass @ 7.50 A |
| -26.221 | Electrostatic x Steric @ 1.50 A |
| -26.023 | Steric @ 7.50 A |
| -25.826 | Electrostatic x Steric @ 6.00 A |

## Co-pattern of `layer-bg` on the exfoliation energy per atom (sigma 0.5 A, lambda 1)

| effect | channel |
|---|---|
| +13.184 | Steric x electronegativity @ 8.50 A |
| -13.139 | Electrostatic x Steric @ 5.50 A |
| +11.983 | Electrostatic x electronegativity @ 3.50 A |
| +10.800 | Electrostatic x Steric @ 4.50 A |
| -10.171 | Steric x inverse_sqrt_mass @ 6.00 A |
| -8.701 | Electrostatic @ 7.50 A |
| +8.269 | Electrostatic x electronegativity @ 4.00 A |
| +7.519 | electronegativity @ 6.00 A |
| +7.507 | Electrostatic x inverse_sqrt_mass @ 5.00 A |
| -7.329 | Electrostatic x inverse_sqrt_mass @ 4.00 A |

## Co-pattern of `heights` on the exfoliation energy per atom (sigma 0.75 A, lambda 1)

| effect | channel |
|---|---|
| +3.549 | Steric x electronegativity @ rho 8.25 A, z +0.0 / +0.0 A |
| -2.644 | Electrostatic x Steric @ rho 5.25 A, z +0.0 / +0.0 A |
| +2.506 | Electrostatic x electronegativity @ rho 1.50 A, z +0.0 / +0.0 A |
| -2.427 | Electrostatic x Steric @ rho 8.25 A, z +0.0 / +0.0 A |
| +2.398 | Electrostatic @ rho 6.00 A, z +0.0 / +0.0 A |
| -2.339 | Electrostatic x inverse_sqrt_mass @ rho 7.50 A, z +0.0 / +0.0 A |
| +2.324 | Electrostatic x Steric @ rho 6.75 A, z +0.0 / +0.0 A |
| +2.267 | Electrostatic x Steric @ rho 3.75 A, z +0.0 / +0.0 A |
| +2.059 | Electrostatic x inverse_sqrt_mass @ rho 1.50 A, z +0.0 / +0.0 A |
| +2.059 | Steric x inverse_sqrt_mass @ rho 8.25 A, z +0.0 / +0.0 A |

## Co-pattern of `heights-oriented` on the exfoliation energy per atom (sigma 0.75 A, lambda 0.3)

| effect | channel |
|---|---|
| +5.529 | Electrostatic x Steric @ rho 6.75 A, z +0.0 / +0.0 A |
| +5.525 | Steric x electronegativity @ rho 8.25 A, z +0.0 / +0.0 A |
| -5.396 | Electrostatic x Steric @ rho 5.25 A, z +0.0 / +0.0 A |
| +5.162 | Electrostatic x electronegativity @ rho 1.50 A, z +0.0 / +0.0 A |
| +4.882 | Electrostatic @ rho 8.25 A, z +0.0 / +0.0 A |
| +4.397 | Electrostatic @ rho 6.00 A, z +0.0 / +0.0 A |
| -3.787 | Electrostatic x Steric @ rho 8.25 A, z +0.0 / +0.0 A |
| +3.556 | Electrostatic x inverse_sqrt_mass @ rho 1.50 A, z +0.0 / +0.0 A |
| -3.494 | Electrostatic x inverse_sqrt_mass @ rho 7.50 A, z +0.0 / +0.0 A |
| +3.481 | Steric x inverse_sqrt_mass @ rho 8.25 A, z +0.0 / +0.0 A |

## Co-pattern of `all-crystal-bg` on the exfoliation energy per atom (sigma 0.5 A, lambda 1)

| effect | channel |
|---|---|
| +5.931 | CIF Mg x CIF Si @ 8.50 A |
| +5.165 | CIF P x CIF Ta @ 5.50 A |
| +3.661 | CIF Mg x CIF Si @ 8.00 A |
| +3.266 | CIF K x CIF Pd @ 8.50 A |
| +2.196 | CIF Al x CIF other @ 5.00 A |
| +2.161 | CIF N x CIF Y @ 6.00 A |
| +2.118 | CIF Al x CIF other @ 4.50 A |
| +2.032 | CIF Mg x CIF Si @ 5.00 A |
| +2.006 | CIF Se x CIF other @ 7.50 A |
| +1.993 | CIF Se x CIF other @ 7.00 A |

## Co-pattern of `all-nobg` on the exfoliation energy per atom (sigma 0.5 A, lambda 0.3)

| effect | channel |
|---|---|
| +8.843 | CIF Mg x CIF Si @ 8.50 A |
| +6.907 | CIF Mg x CIF Si @ 8.00 A |
| +5.599 | CIF K x CIF Pd @ 8.50 A |
| +5.528 | CIF P x CIF Ta @ 5.50 A |
| +3.686 | CIF N x CIF Y @ 8.00 A |
| +3.447 | CIF Se x CIF other @ 7.50 A |
| +3.376 | CIF Se x CIF other @ 7.00 A |
| +3.155 | CIF Mg x CIF Si @ 9.00 A |
| +3.109 | CIF Ti @ 7.50 A |
| +3.108 | CIF Pb x CIF Sb @ 8.50 A |

## Co-pattern of `all-layer-bg` on the exfoliation energy per atom (sigma 0.5 A, lambda 0.1)

| effect | channel |
|---|---|
| +10.463 | CIF Mg x CIF Si @ 8.50 A |
| +6.305 | CIF P x CIF Ta @ 5.50 A |
| -5.881 | CIF N x CIF Y @ 7.00 A |
| -5.211 | CIF Mg x CIF Si @ 7.00 A |
| -4.711 | CIF Se x CIF other @ 8.50 A |
| +4.225 | CIF Br x CIF Cd @ 7.50 A |
| +4.221 | CIF O x CIF Si @ 7.50 A |
| -4.180 | CIF Cd x CIF S @ 7.50 A |
| +3.968 | CIF Pb x CIF Ti @ 6.50 A |
| +3.720 | CIF Mg x CIF Si @ 8.00 A |
