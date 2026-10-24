# Continuous molecular fields of crystals: Matbench phonons

Last peak of the phonon density of states (cm^-1), 1,265 structures, the five Matbench folds; MAE mean +/- sd over the folds, RMSE and R2 on the pooled held-out predictions. Hyper-parameters (sigma, lambda, gamma) chosen by 10-fold CV inside every training fold.

| variant | fields | block weights | MAE (cm^-1) | RMSE | R2 | sigma chosen | time (s) |
|---|---|---|---|---|---|---|---|
| elements | 55 | ones | 195.8 +/- 7.3 | 340.7 | 0.524 | 0.75, 0.75, 0.5, 0.75, 0.75 | 58 |
| physchem | 3 | ones | 150.2 +/- 10.8 | 243.3 | 0.757 | 0.35, 0.35, 0.5, 0.35, 0.35 | 34 |
| physchem+mass | 4 | ones | 117.3 +/- 10.1 | 191.6 | 0.850 | 0.75, 0.5, 0.5, 0.35, 0.35 | 35 |
| all | 58 | ones | 180.6 +/- 8.5 | 318.0 | 0.585 | 0.75, 0.75, 0.5, 0.75, 0.75 | 63 |
| all+mass | 59 | ones | 179.0 +/- 9.1 | 314.5 | 0.595 | 0.75, 0.75, 0.5, 0.75, 0.75 | 63 |
| all-nobg | 58 | ones | 119.3 +/- 13.3 | 238.7 | 0.766 | 0.75, 0.5, 0.75, 0.75, 0.35 | 46 |
| all-hard | 58 | ones | 204.6 +/- 10.3 | 354.3 | 0.485 | 0.75, 0.5, 0.5, 0.5, 0.5 | 47 |
| all-log | 58 | ones | 174.5 +/- 12.0 | 354.0 | 0.486 | 0.75, 0.75, 0.75, 0.75, 0.75 | 60 |
| all-gauss | 58 | ones | 175.7 +/- 7.5 | 308.0 | 0.611 | 0.75, 0.75, 0.75, 0.75, 0.75 | 198 |
| physchem+mass-gauss | 4 | ones | 105.7 +/- 5.2 | 190.1 | 0.852 | 0.5, 0.75, 0.75, 0.75, 0.5 | 163 |
| all+mass-gauss | 59 | ones | 174.2 +/- 8.9 | 306.1 | 0.616 | 0.75, 0.75, 0.75, 0.75, 0.75 | 201 |
| all-balanced | 58 | balanced | 149.3 +/- 7.2 | 263.5 | 0.715 | 0.75, 0.75, 0.75, 0.75, 0.75 | 52 |
| all-align | 58 | align | 154.9 +/- 9.6 | 280.5 | 0.677 | 0.75, 0.75, 0.75, 0.75, 0.75 | 52 |
| all+mass-balanced | 59 | balanced | 148.1 +/- 9.2 | 244.9 | 0.754 | 0.75, 0.75, 0.75, 0.75, 0.75 | 48 |
| all+mass-align | 59 | align | 152.3 +/- 10.1 | 272.5 | 0.696 | 0.75, 0.75, 0.75, 0.75, 0.75 | 51 |
| all+mass-alignf | 59 | alignf | 156.8 +/- 9.5 | 279.5 | 0.680 | 0.75, 0.75, 0.75, 0.75, 0.75 | 152 |
| all+mass-nobg-align | 59 | align | 96.6 +/- 17.8 | 188.5 | 0.854 | 0.5, 0.35, 0.35, 0.35, 0.35 | 58 |
| all+mass-nobg | 59 | ones | 118.7 +/- 12.7 | 239.7 | 0.764 | 0.75, 0.5, 0.75, 0.75, 0.35 | 50 |
| all+mass-nobg-balanced | 59 | balanced | 111.6 +/- 9.5 | 221.4 | 0.799 | 0.35, 0.75, 0.75, 0.35, 0.35 | 51 |
| physchem+mass-nobg | 4 | ones | 92.1 +/- 4.8 | 139.5 | 0.920 | 0.5, 0.5, 0.5, 0.5, 0.5 | 26 |

Leaderboard reference (matbench v0.1, MAE in cm^-1, from the Matbench site at the time of writing, October 2026): dummy 324.0; RF-SCM/Magpie 67.6; CGCNN 57.8; CrabNet 55.1; SchNet 39.0; MODNet 34.3; coGN 29.7; ALIGNN 29.5; coNGN 28.9; MegNet 28.8.

## Block weights by kind of block

The mixing coefficients of the channel blocks (one block per pair of field types), fitted on the whole set with the settings the folds chose: the mean weight over the blocks of each kind and the kind's share of the squared norm of a channel vector. With every block at one the shares are proportional to the numbers of blocks.

| variant | kind of block | blocks | mean weight | share |
|---|---|---|---|---|
| all-balanced | indicator | 1540 | 0.025 | 0.33 |
| all-balanced | physchem | 6 | 0.408 | 0.33 |
| all-balanced | indicator*physchem | 165 | 0.078 | 0.33 |
| all-align | indicator | 1540 | 0.384 | 0.65 |
| all-align | physchem | 6 | 2.400 | 0.03 |
| all-align | indicator*physchem | 165 | 1.200 | 0.32 |
| all+mass-balanced | indicator | 1540 | 0.025 | 0.33 |
| all+mass-balanced | physchem | 10 | 0.316 | 0.33 |
| all+mass-balanced | indicator*physchem | 220 | 0.067 | 0.33 |
| all+mass-align | indicator | 1540 | 0.355 | 0.53 |
| all+mass-align | physchem | 10 | 2.950 | 0.09 |
| all+mass-align | indicator*physchem | 220 | 1.178 | 0.39 |
| all+mass-alignf | indicator | 1540 | 0.413 | 0.63 |
| all+mass-alignf | physchem | 10 | 2.401 | 0.07 |
| all+mass-alignf | indicator*physchem | 220 | 0.872 | 0.31 |
| all+mass-nobg-align | indicator | 1540 | 0.270 | 0.39 |
| all+mass-nobg-align | physchem | 10 | 4.400 | 0.23 |
| all+mass-nobg-align | indicator*physchem | 220 | 0.974 | 0.39 |
| all+mass-nobg-balanced | indicator | 1540 | 0.025 | 0.33 |
| all+mass-nobg-balanced | physchem | 10 | 0.316 | 0.33 |
| all+mass-nobg-balanced | indicator*physchem | 220 | 0.067 | 0.33 |

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

## Co-pattern of `all-balanced` (sigma 0.75 A, lambda 0.03)

Weight of each channel in the linear model times its spread over the set: the channels that move the prediction most, sign included.

| effect | channel |
|---|---|
| -48.42 | electronegativity @ 2.25 A |
| -43.00 | Electrostatic x Steric @ 6.75 A |
| -40.35 | Electrostatic @ 3.75 A |
| +39.58 | Steric x electronegativity @ 7.50 A |
| +37.75 | electronegativity x CIF H @ 8.25 A |
| -37.51 | Electrostatic x Steric @ 9.00 A |
| -37.40 | Electrostatic @ 1.50 A |
| -36.47 | Steric x electronegativity @ 9.00 A |
| +35.46 | Electrostatic @ 3.00 A |
| +34.81 | Steric @ 6.75 A |
| +34.16 | Steric x electronegativity @ 3.00 A |
| +33.25 | Steric x CIF H @ 7.50 A |
| +33.00 | Steric x electronegativity @ 2.25 A |
| +30.29 | Steric @ 9.00 A |
| +29.51 | electronegativity @ 6.00 A |

## Co-pattern of `all-align` (sigma 0.75 A, lambda 0.1)

Weight of each channel in the linear model times its spread over the set: the channels that move the prediction most, sign included.

| effect | channel |
|---|---|
| -36.15 | CIF H x CIF O @ 2.25 A |
| +27.52 | CIF Ca x CIF H @ 8.25 A |
| -26.99 | CIF H x CIF O @ 6.75 A |
| -26.24 | CIF H x CIF O @ 3.00 A |
| +23.66 | CIF H x CIF O @ 4.50 A |
| -21.45 | CIF H x CIF S @ 6.75 A |
| +21.28 | CIF H x CIF P @ 6.75 A |
| -18.97 | CIF C x CIF N @ 5.25 A |
| +18.93 | CIF H @ 8.25 A |
| -18.49 | CIF H x CIF P @ 8.25 A |
| +18.26 | CIF H x CIF N @ 7.50 A |
| -18.06 | CIF H x CIF S @ 7.50 A |
| -17.37 | CIF Cl x CIF H @ 7.50 A |
| +17.24 | CIF Ca x CIF H @ 9.00 A |
| +17.20 | CIF K x CIF N @ 6.75 A |

## Co-pattern of `all+mass-balanced` (sigma 0.75 A, lambda 0.03)

Weight of each channel in the linear model times its spread over the set: the channels that move the prediction most, sign included.

| effect | channel |
|---|---|
| +86.04 | inverse_sqrt_mass @ 8.25 A |
| +76.90 | Steric x inverse_sqrt_mass @ 7.50 A |
| +56.23 | Electrostatic x inverse_sqrt_mass @ 9.00 A |
| +55.13 | inverse_sqrt_mass @ 6.75 A |
| +53.06 | Electrostatic x inverse_sqrt_mass @ 6.75 A |
| +44.47 | Steric x inverse_sqrt_mass @ 3.00 A |
| -44.25 | inverse_sqrt_mass @ 1.50 A |
| -43.48 | electronegativity x inverse_sqrt_mass @ 2.25 A |
| +43.17 | inverse_sqrt_mass @ 0.00 A |
| -42.27 | Steric x inverse_sqrt_mass @ 6.00 A |
| -38.94 | inverse_sqrt_mass @ 2.25 A |
| +37.92 | inverse_sqrt_mass @ 6.00 A |
| -37.03 | Electrostatic @ 3.75 A |
| +35.31 | Electrostatic x electronegativity @ 3.75 A |
| -34.54 | Electrostatic x Steric @ 9.00 A |

## Co-pattern of `all+mass-align` (sigma 0.75 A, lambda 0.03)

Weight of each channel in the linear model times its spread over the set: the channels that move the prediction most, sign included.

| effect | channel |
|---|---|
| -34.90 | CIF H x CIF O @ 2.25 A |
| -30.77 | CIF H x CIF O @ 6.75 A |
| +25.94 | CIF Ca x CIF H @ 8.25 A |
| -25.67 | CIF N @ 8.25 A |
| -24.20 | CIF H x CIF O @ 3.00 A |
| +20.98 | CIF H x CIF P @ 6.75 A |
| +20.35 | inverse_sqrt_mass @ 8.25 A |
| -19.90 | CIF H x CIF S @ 6.75 A |
| -19.27 | CIF C x CIF N @ 5.25 A |
| +18.78 | CIF H x CIF N @ 7.50 A |
| +18.76 | CIF K x CIF N @ 6.75 A |
| -18.40 | CIF H x CIF P @ 8.25 A |
| -17.55 | CIF H x CIF N @ 9.00 A |
| -17.54 | CIF Cl x CIF O @ 7.50 A |
| +16.91 | CIF H x CIF O @ 9.00 A |

## Co-pattern of `all+mass-alignf` (sigma 0.75 A, lambda 0.1)

Weight of each channel in the linear model times its spread over the set: the channels that move the prediction most, sign included.

| effect | channel |
|---|---|
| -41.68 | CIF H x CIF O @ 2.25 A |
| -29.42 | CIF H x CIF O @ 3.00 A |
| -29.10 | CIF H x CIF O @ 6.75 A |
| +26.10 | CIF H x CIF O @ 4.50 A |
| +25.48 | CIF Ca x CIF H @ 8.25 A |
| -23.61 | CIF H x CIF S @ 6.75 A |
| +22.37 | CIF H x CIF P @ 6.75 A |
| -20.75 | CIF C x CIF N @ 5.25 A |
| -20.61 | CIF H x CIF S @ 7.50 A |
| -20.23 | Steric x CIF C @ 5.25 A |
| -19.58 | CIF H x CIF P @ 8.25 A |
| +19.22 | Steric x CIF C @ 3.00 A |
| +18.90 | CIF H x CIF N @ 7.50 A |
| +18.64 | CIF H x CIF O @ 3.75 A |
| -18.21 | CIF Cl x CIF H @ 7.50 A |

## Co-pattern of `all+mass-nobg-align` (sigma 0.35 A, lambda 0.1)

Weight of each channel in the linear model times its spread over the set: the channels that move the prediction most, sign included.

| effect | channel |
|---|---|
| +28.21 | CIF H x CIF O @ 7.70 A |
| +26.56 | CIF H x CIF O @ 8.05 A |
| +20.88 | electronegativity x CIF H @ 8.05 A |
| +19.89 | CIF H x CIF O @ 8.40 A |
| +17.36 | CIF H x CIF O @ 7.35 A |
| -16.53 | Steric @ 7.35 A |
| +16.53 | CIF H x CIF N @ 7.70 A |
| +16.46 | electronegativity x CIF H @ 8.40 A |
| +14.98 | Steric x CIF H @ 7.35 A |
| +14.82 | Steric x CIF H @ 7.00 A |
| +14.43 | CIF H x CIF O @ 8.75 A |
| +14.42 | CIF H x CIF N @ 8.05 A |
| -14.41 | Steric @ 7.70 A |
| -13.01 | Steric x inverse_sqrt_mass @ 8.40 A |
| -11.89 | Steric @ 7.00 A |

## Co-pattern of `all+mass-nobg` (sigma 0.75 A, lambda 0.3)

Weight of each channel in the linear model times its spread over the set: the channels that move the prediction most, sign included.

| effect | channel |
|---|---|
| +24.42 | CIF H x CIF O @ 8.25 A |
| +22.60 | CIF H x CIF O @ 7.50 A |
| +15.24 | CIF H x CIF N @ 7.50 A |
| +14.24 | CIF H x CIF S @ 8.25 A |
| +13.94 | CIF H x CIF N @ 8.25 A |
| +13.18 | CIF H x CIF O @ 6.75 A |
| +11.87 | CIF H x CIF O @ 9.00 A |
| +10.79 | CIF Br x CIF H @ 8.25 A |
| +9.74 | CIF H x CIF N @ 6.75 A |
| +9.11 | CIF Br x CIF H @ 7.50 A |
| +9.07 | CIF N x CIF Rb @ 7.50 A |
| +8.98 | CIF H x CIF S @ 7.50 A |
| +8.96 | CIF H x CIF S @ 9.00 A |
| +8.74 | CIF H x CIF O @ 6.00 A |
| +8.30 | CIF Br x CIF O @ 7.50 A |

## Co-pattern of `all+mass-nobg-balanced` (sigma 0.35 A, lambda 0.1)

Weight of each channel in the linear model times its spread over the set: the channels that move the prediction most, sign included.

| effect | channel |
|---|---|
| +41.54 | Steric x inverse_sqrt_mass @ 7.35 A |
| +36.79 | electronegativity x inverse_sqrt_mass @ 8.05 A |
| +34.66 | Steric x inverse_sqrt_mass @ 7.70 A |
| +34.48 | Steric x inverse_sqrt_mass @ 7.00 A |
| +30.68 | electronegativity x inverse_sqrt_mass @ 8.40 A |
| +27.59 | Electrostatic x inverse_sqrt_mass @ 8.75 A |
| +27.31 | electronegativity x inverse_sqrt_mass @ 8.75 A |
| +26.02 | electronegativity x inverse_sqrt_mass @ 7.70 A |
| -25.69 | inverse_sqrt_mass @ 7.35 A |
| +24.69 | inverse_sqrt_mass @ 8.05 A |
| +23.25 | Steric x inverse_sqrt_mass @ 8.05 A |
| -21.96 | Steric @ 7.35 A |
| +21.78 | Electrostatic x inverse_sqrt_mass @ 7.35 A |
| +21.67 | Electrostatic x inverse_sqrt_mass @ 2.80 A |
| +21.05 | Electrostatic x inverse_sqrt_mass @ 6.30 A |

## Co-pattern of `physchem+mass-nobg` (sigma 0.5 A, lambda 1e-06)

Weight of each channel in the linear model times its spread over the set: the channels that move the prediction most, sign included.

| effect | channel |
|---|---|
| +1223.14 | Electrostatic @ 0.00 A |
| +997.71 | Electrostatic x inverse_sqrt_mass @ 0.00 A |
| -874.02 | inverse_sqrt_mass @ 6.50 A |
| -849.88 | Electrostatic x inverse_sqrt_mass @ 0.50 A |
| -834.60 | Electrostatic @ 0.50 A |
| -718.16 | Electrostatic x Steric @ 0.50 A |
| -718.12 | Electrostatic x electronegativity @ 4.00 A |
| +710.29 | inverse_sqrt_mass @ 7.00 A |
| +642.96 | Electrostatic x Steric @ 0.00 A |
| +628.75 | inverse_sqrt_mass @ 6.00 A |
| -618.23 | inverse_sqrt_mass @ 7.50 A |
| -608.18 | Electrostatic x electronegativity @ 3.00 A |
| +605.25 | Electrostatic x inverse_sqrt_mass @ 1.00 A |
| +603.45 | electronegativity x inverse_sqrt_mass @ 8.50 A |
| +577.02 | electronegativity x inverse_sqrt_mass @ 7.50 A |
