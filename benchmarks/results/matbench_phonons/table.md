# Continuous molecular fields of crystals: Matbench phonons

Last peak of the phonon density of states (cm^-1), 1,265 structures, the five Matbench folds; MAE mean +/- sd over the folds, RMSE and R2 on the pooled held-out predictions. Hyper-parameters (sigma, lambda, gamma) chosen by 10-fold CV inside every training fold.

| variant | fields | MAE (cm^-1) | RMSE | R2 | sigma chosen | time (s) |
|---|---|---|---|---|---|---|
| elements | 55 | 195.8 +/- 7.3 | 340.7 | 0.524 | 0.75, 0.75, 0.5, 0.75, 0.75 | 58 |
| physchem | 3 | 150.2 +/- 10.8 | 243.3 | 0.757 | 0.35, 0.35, 0.5, 0.35, 0.35 | 34 |
| physchem+mass | 4 | 117.3 +/- 10.1 | 191.6 | 0.850 | 0.75, 0.5, 0.5, 0.35, 0.35 | 35 |
| all | 58 | 180.6 +/- 8.5 | 318.0 | 0.585 | 0.75, 0.75, 0.5, 0.75, 0.75 | 63 |
| all+mass | 59 | 179.0 +/- 9.1 | 314.5 | 0.595 | 0.75, 0.75, 0.5, 0.75, 0.75 | 63 |
| all-nobg | 58 | 119.3 +/- 13.3 | 238.7 | 0.766 | 0.75, 0.5, 0.75, 0.75, 0.35 | 46 |
| all-hard | 58 | 204.6 +/- 10.3 | 354.3 | 0.485 | 0.75, 0.5, 0.5, 0.5, 0.5 | 47 |
| all-log | 58 | 174.5 +/- 12.0 | 354.0 | 0.486 | 0.75, 0.75, 0.75, 0.75, 0.75 | 60 |
| all-gauss | 58 | 175.7 +/- 7.5 | 308.0 | 0.611 | 0.75, 0.75, 0.75, 0.75, 0.75 | 198 |
| physchem+mass-gauss | 4 | 105.7 +/- 5.2 | 190.1 | 0.852 | 0.5, 0.75, 0.75, 0.75, 0.5 | 163 |
| all+mass-gauss | 59 | 174.2 +/- 8.9 | 306.1 | 0.616 | 0.75, 0.75, 0.75, 0.75, 0.75 | 201 |

Leaderboard reference (matbench v0.1, MAE in cm^-1, from the Matbench site at the time of writing, October 2026): dummy 324.0; RF-SCM/Magpie 67.6; CGCNN 57.8; CrabNet 55.1; SchNet 39.0; MODNet 34.3; coGN 29.7; ALIGNN 29.5; coNGN 28.9; MegNet 28.8.

## Co-pattern of `elements` (sigma 0.75 A, lambda 0.1)

Weight of each channel in the linear model times its spread over the set: the channels that move the prediction most, sign included.

| effect | channel |
|---|---|
| -25.35 | CIF H x CIF O @ 6.75 A |
| -24.40 | CIF Cl x CIF H @ 7.50 A |
| -24.25 | CIF H x CIF S @ 6.75 A |
| +22.95 | CIF Ca x CIF H @ 8.25 A |
| +22.17 | CIF H x CIF P @ 6.75 A |
| +20.76 | CIF H x CIF O @ 4.50 A |
| +20.52 | CIF H @ 8.25 A |
| +19.31 | CIF K x CIF N @ 6.75 A |
| -19.28 | CIF Cl x CIF O @ 7.50 A |
| -19.11 | CIF H x CIF O @ 3.00 A |
| -18.90 | CIF H x CIF S @ 7.50 A |
| -17.90 | CIF H x CIF O @ 2.25 A |
| -17.71 | CIF H x CIF P @ 8.25 A |
| +16.49 | CIF Ca x CIF H @ 9.00 A |
| +16.46 | CIF Cl x CIF N @ 6.75 A |

## Co-pattern of `physchem` (sigma 0.35 A, lambda 1e-06)

Weight of each channel in the linear model times its spread over the set: the channels that move the prediction most, sign included.

| effect | channel |
|---|---|
| -3846.29 | electronegativity @ 0.35 A |
| +3825.07 | Steric x electronegativity @ 0.00 A |
| +3812.05 | electronegativity @ 0.00 A |
| +3404.88 | Electrostatic @ 0.00 A |
| -3048.01 | Electrostatic @ 0.35 A |
| +2935.76 | Steric x electronegativity @ 1.05 A |
| -2358.57 | Steric x electronegativity @ 0.35 A |
| -2339.32 | Steric x electronegativity @ 0.70 A |
| +2164.95 | electronegativity @ 0.70 A |
| -1965.67 | Steric @ 1.40 A |
| -1566.27 | electronegativity @ 1.05 A |
| -1066.86 | Steric @ 0.00 A |
| +1062.80 | Steric @ 1.75 A |
| -859.22 | Electrostatic x Steric @ 1.75 A |
| +810.24 | Steric @ 1.05 A |

## Co-pattern of `physchem+mass` (sigma 0.35 A, lambda 1e-06)

Weight of each channel in the linear model times its spread over the set: the channels that move the prediction most, sign included.

| effect | channel |
|---|---|
| +1508.38 | Steric x inverse_sqrt_mass @ 0.00 A |
| +1366.12 | Steric x inverse_sqrt_mass @ 1.40 A |
| +1232.93 | Steric x inverse_sqrt_mass @ 2.10 A |
| +1129.66 | electronegativity x inverse_sqrt_mass @ 0.00 A |
| -1114.65 | Steric x inverse_sqrt_mass @ 1.75 A |
| +961.91 | Electrostatic @ 2.10 A |
| +956.93 | Electrostatic @ 2.80 A |
| +922.97 | electronegativity x inverse_sqrt_mass @ 4.20 A |
| -903.93 | inverse_sqrt_mass @ 1.40 A |
| +830.43 | electronegativity x inverse_sqrt_mass @ 1.40 A |
| +822.82 | electronegativity x inverse_sqrt_mass @ 5.60 A |
| +808.77 | Electrostatic x inverse_sqrt_mass @ 0.00 A |
| +807.43 | electronegativity x inverse_sqrt_mass @ 4.90 A |
| +770.87 | electronegativity x inverse_sqrt_mass @ 2.10 A |
| +757.60 | electronegativity x inverse_sqrt_mass @ 3.50 A |

## Co-pattern of `all` (sigma 0.75 A, lambda 0.03)

Weight of each channel in the linear model times its spread over the set: the channels that move the prediction most, sign included.

| effect | channel |
|---|---|
| +26.01 | CIF Ca x CIF H @ 8.25 A |
| -22.74 | CIF Cl x CIF H @ 7.50 A |
| -21.77 | CIF H x CIF O @ 6.75 A |
| +21.61 | CIF H x CIF P @ 6.75 A |
| -20.91 | CIF H x CIF S @ 6.75 A |
| -19.21 | CIF Cl x CIF O @ 7.50 A |
| -17.32 | CIF H x CIF P @ 8.25 A |
| -16.87 | CIF H x CIF S @ 7.50 A |
| +15.90 | CIF K x CIF N @ 6.75 A |
| +15.88 | CIF Cl x CIF N @ 6.75 A |
| +15.86 | CIF Ca x CIF H @ 9.00 A |
| +15.83 | CIF H x CIF N @ 7.50 A |
| +15.36 | CIF H x CIF O @ 4.50 A |
| -15.00 | CIF H x CIF O @ 3.00 A |
| -14.59 | CIF H x CIF O @ 2.25 A |

## Co-pattern of `all+mass` (sigma 0.75 A, lambda 0.03)

Weight of each channel in the linear model times its spread over the set: the channels that move the prediction most, sign included.

| effect | channel |
|---|---|
| +21.15 | CIF Ca x CIF H @ 8.25 A |
| +20.21 | CIF H x CIF P @ 6.75 A |
| -19.73 | CIF Cl x CIF H @ 7.50 A |
| -19.04 | CIF H x CIF O @ 6.75 A |
| -17.74 | CIF Cl x CIF O @ 7.50 A |
| -16.37 | CIF H x CIF S @ 6.75 A |
| -16.32 | CIF H x CIF P @ 8.25 A |
| +15.54 | CIF Cl x CIF N @ 6.75 A |
| -14.39 | CIF H x CIF S @ 7.50 A |
| +14.29 | CIF Cl x CIF N @ 7.50 A |
| +14.21 | CIF H x CIF N @ 7.50 A |
| +13.85 | CIF K x CIF N @ 6.75 A |
| +12.94 | CIF Ca x CIF H @ 9.00 A |
| -12.82 | CIF Br x CIF O @ 9.00 A |
| +12.60 | inverse_sqrt_mass x CIF Ca @ 8.25 A |

## Co-pattern of `all-nobg` (sigma 0.75 A, lambda 0.3)

Weight of each channel in the linear model times its spread over the set: the channels that move the prediction most, sign included.

| effect | channel |
|---|---|
| +27.33 | CIF H x CIF O @ 8.25 A |
| +25.12 | CIF H x CIF O @ 7.50 A |
| +17.32 | CIF H x CIF N @ 7.50 A |
| +15.88 | CIF H x CIF N @ 8.25 A |
| +15.33 | CIF H x CIF S @ 8.25 A |
| +14.54 | CIF H x CIF O @ 6.75 A |
| +13.31 | CIF H x CIF O @ 9.00 A |
| +12.18 | CIF Br x CIF H @ 8.25 A |
| +11.31 | Steric x CIF H @ 7.50 A |
| +11.16 | electronegativity x CIF H @ 8.25 A |
| +11.07 | CIF H x CIF N @ 6.75 A |
| +10.85 | electronegativity x CIF H @ 7.50 A |
| +10.16 | CIF Br x CIF H @ 7.50 A |
| +9.76 | CIF N x CIF Rb @ 7.50 A |
| +9.72 | CIF H x CIF O @ 6.00 A |

## Co-pattern of `all-hard` (sigma 0.5 A, lambda 0.03)

Weight of each channel in the linear model times its spread over the set: the channels that move the prediction most, sign included.

| effect | channel |
|---|---|
| +25.54 | CIF H x CIF O @ 9.00 A |
| +22.61 | CIF H x CIF O @ 9.50 A |
| +21.79 | CIF H x CIF P @ 6.50 A |
| -21.56 | CIF H x CIF O @ 6.50 A |
| -18.79 | CIF Cl x CIF H @ 7.50 A |
| -15.88 | CIF H x CIF S @ 7.00 A |
| -15.39 | CIF Cl x CIF H @ 8.00 A |
| -15.06 | CIF Br x CIF O @ 8.50 A |
| -14.86 | CIF H x CIF Rb @ 9.00 A |
| -14.21 | CIF Li x CIF N @ 9.50 A |
| +14.00 | CIF K x CIF N @ 6.50 A |
| -13.64 | CIF Li x CIF N @ 10.00 A |
| +13.01 | CIF Cl x CIF O @ 9.50 A |
| +12.84 | CIF Ca x CIF H @ 8.50 A |
| +11.87 | CIF H x CIF O @ 8.50 A |

## Co-pattern of `all-log` (sigma 0.75 A, lambda 0.03)

Weight of each channel in the linear model times its spread over the set: the channels that move the prediction most, sign included.

| effect | channel |
|---|---|
| -0.02 | CIF H x CIF S @ 6.75 A |
| -0.02 | CIF Cl x CIF H @ 7.50 A |
| +0.02 | CIF Ag x CIF I @ 8.25 A |
| +0.02 | CIF Cl x CIF N @ 6.75 A |
| +0.02 | Steric x CIF S @ 7.50 A |
| +0.02 | CIF Rb x CIF S @ 7.50 A |
| +0.02 | Electrostatic x CIF Rb @ 7.50 A |
| +0.02 | CIF Ca x CIF H @ 8.25 A |
| -0.02 | CIF H x CIF S @ 7.50 A |
| +0.01 | CIF K x CIF N @ 6.75 A |
| +0.01 | CIF Cs x CIF O @ 8.25 A |
| -0.01 | CIF I x CIF K @ 7.50 A |
| -0.01 | CIF I x CIF Tl @ 7.50 A |
| -0.01 | CIF Cs x CIF I @ 9.00 A |
| -0.01 | CIF B x CIF Bi @ 8.25 A |
