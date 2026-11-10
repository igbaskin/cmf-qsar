# The carbonyl band of substituted benzoic acids as a local target (plan item SP2)

189 acids (the 95 substituents of the Hammett set, meta and para to the carboxyl group), the carbonyl stretch computed with ORCA at r2SCAN-3c and scaled by 0.97, from 1711 to 1737 cm^-1 with a spread (standard deviation) of 4.5 cm^-1; 0 acids ended with an anti carboxyl and 5 with an imaginary frequency.  10 folds holding out whole molecules; the ridge chosen on the same folds for every descriptor.  Differences in q2 carry their 95 % bootstrap interval over the molecules.

The measurement is made twice: on the geometries of the quantum optimisation, which is the calculation that also gives the band, and on the force-field geometries that calculation started from.  Only the second says what the structure alone predicts.

## The band against the Hammett constant

| acids | n | slope, cm^-1 per sigma unit | r |
|---|---:|---:|---:|
| all | 189 | +11.7 | +0.866 |
| meta | 95 | +11.3 | +0.817 |
| para | 95 | +11.5 | +0.910 |

## The references, on the geometries of the quantum optimisation

| descriptor | q2 | RMSE, cm^-1 | note |
|---|---:|---:|---|
| atom-centred counts | +0.662 | 2.60 | 17 atom types in 6 shells |
| linear regression on the Hammett constant, all acids | +0.740 | 2.28 | band = a + rho * sigma, fitted in fold |
| linear regression on the Hammett constant, meta and para separately | +0.781 | 2.09 | band = a + rho * sigma, fitted in fold |
| linear regression on the length of the C=O bond | +0.973 | 0.74 | band = a + b * length, fitted in fold; r -0.987; lengths 1.2088 to 1.2160 A, spread (sd) 0.0012 A; against the counts +0.311 [+0.253, +0.386] |

## The field around the carbonyl carbon by envelope width, on the geometries of the quantum optimisation

| rho, 1/A^2 | width, A | atoms | invariant local q2 | against counts | against the Hammett regression | against the bond length | posed overlap q2 | against counts | against the Hammett regression | against the bond length | posed against invariant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 10.00 | 21.0 | +0.743 | +0.081 [-0.000, +0.152] | +0.003 [-0.060, +0.065] | -0.230 [-0.305, -0.175] | +0.725 | +0.063 [-0.006, +0.132] | -0.016 [-0.071, +0.046] | -0.248 [-0.312, -0.204] | -0.018 [-0.071, +0.040] |
| 0.02 | 7.07 | 21.0 | +0.750 | +0.088 [+0.012, +0.158] | +0.010 [-0.051, +0.075] | -0.223 [-0.293, -0.172] | +0.719 | +0.058 [-0.008, +0.124] | -0.021 [-0.075, +0.042] | -0.253 [-0.307, -0.212] | -0.031 [-0.079, +0.023] |
| 0.05 | 4.47 | 21.0 | +0.770 | +0.108 [+0.031, +0.182] | +0.030 [-0.034, +0.098] | -0.203 [-0.270, -0.155] | +0.733 | +0.072 [+0.005, +0.139] | -0.007 [-0.064, +0.057] | -0.239 [-0.309, -0.193] | -0.037 [-0.087, +0.014] |
| 0.1 | 3.16 | 20.9 | +0.863 | +0.202 [+0.131, +0.276] | +0.123 [+0.062, +0.191] | -0.109 [-0.166, -0.071] | +0.869 | +0.208 [+0.147, +0.274] | +0.129 [+0.079, +0.196] | -0.103 [-0.142, -0.076] | +0.006 [-0.030, +0.050] |
| 0.25 | 2.00 | 16.4 | +0.929 | +0.268 [+0.147, +0.373] | +0.189 [+0.072, +0.290] | -0.043 [-0.151, +0.006] | +0.903 | +0.241 [+0.153, +0.331] | +0.162 [+0.078, +0.245] | -0.070 [-0.145, -0.031] | -0.027 [-0.053, +0.010] |
| 0.5 | 1.41 | 12.0 | +0.960 | +0.298 [+0.240, +0.374] | +0.220 [+0.166, +0.294] | -0.013 [-0.022, -0.006] | +0.963 | +0.301 [+0.243, +0.377] | +0.223 [+0.166, +0.300] | -0.010 [-0.032, +0.003] | +0.003 [-0.013, +0.013] |
| 1 | 1.00 | 9.0 | +0.976 | +0.314 [+0.257, +0.388] | +0.236 [+0.180, +0.314] | +0.003 [+0.001, +0.006] | +0.978 | +0.316 [+0.259, +0.390] | +0.238 [+0.182, +0.316] | +0.005 [+0.002, +0.009] | +0.002 [+0.001, +0.004] |
| 2 | 0.71 | 5.0 | +0.974 | +0.312 [+0.254, +0.386] | +0.234 [+0.178, +0.311] | +0.001 [+0.000, +0.003] | +0.974 | +0.312 [+0.255, +0.386] | +0.234 [+0.178, +0.311] | +0.001 [-0.000, +0.003] | -0.000 [-0.001, +0.001] |

## The references, on the start geometries (force field)

| descriptor | q2 | RMSE, cm^-1 | note |
|---|---:|---:|---|
| atom-centred counts | +0.616 | 2.77 | 17 atom types in 6 shells |
| linear regression on the length of the C=O bond | -0.014 | 4.50 | band = a + b * length, fitted in fold; r -0.006; lengths 1.2162 to 1.2683 A, spread (sd) 0.0079 A; against the counts -0.629 [-0.697, -0.553] |

## The field around the carbonyl carbon by envelope width, on the start geometries (force field)

| rho, 1/A^2 | width, A | atoms | invariant local q2 | against counts | against the Hammett regression | against the bond length | posed overlap q2 | against counts | against the Hammett regression | against the bond length | posed against invariant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 10.00 | 21.0 | +0.746 | +0.130 [+0.043, +0.209] | +0.006 [-0.065, +0.077] | +0.760 [+0.674, +0.829] | +0.702 | +0.087 [+0.002, +0.168] | -0.038 [-0.097, +0.026] | +0.716 [+0.651, +0.777] | -0.044 [-0.108, +0.028] |
| 0.02 | 7.07 | 21.0 | +0.742 | +0.126 [+0.046, +0.202] | +0.002 [-0.068, +0.074] | +0.756 [+0.676, +0.822] | +0.698 | +0.082 [+0.004, +0.163] | -0.042 [-0.099, +0.023] | +0.712 [+0.655, +0.766] | -0.044 [-0.100, +0.019] |
| 0.05 | 4.47 | 21.0 | +0.707 | +0.092 [+0.000, +0.176] | -0.033 [-0.114, +0.040] | +0.721 [+0.628, +0.796] | +0.691 | +0.075 [-0.004, +0.150] | -0.049 [-0.112, +0.017] | +0.705 [+0.625, +0.774] | -0.016 [-0.072, +0.045] |
| 0.1 | 3.16 | 20.9 | +0.650 | +0.034 [-0.063, +0.121] | -0.091 [-0.173, -0.011] | +0.663 [+0.570, +0.740] | +0.663 | +0.047 [-0.033, +0.121] | -0.078 [-0.142, -0.009] | +0.676 [+0.599, +0.744] | +0.013 [-0.039, +0.066] |
| 0.25 | 2.00 | 16.4 | +0.254 | -0.362 [-0.485, -0.242] | -0.486 [-0.594, -0.381] | +0.268 [+0.163, +0.354] | +0.326 | -0.290 [-0.399, -0.173] | -0.415 [-0.515, -0.309] | +0.339 [+0.259, +0.422] | +0.072 [-0.040, +0.198] |
| 0.5 | 1.41 | 11.7 | +0.279 | -0.336 [-0.478, -0.203] | -0.461 [-0.595, -0.331] | +0.293 [+0.168, +0.400] | +0.259 | -0.357 [-0.444, -0.244] | -0.482 [-0.569, -0.365] | +0.272 [+0.220, +0.342] | -0.021 [-0.093, +0.094] |
| 1 | 1.00 | 9.0 | +0.191 | -0.425 [-0.719, -0.231] | -0.550 [-0.829, -0.363] | +0.204 [-0.079, +0.374] | +0.231 | -0.385 [-0.523, -0.248] | -0.509 [-0.644, -0.370] | +0.245 [+0.117, +0.350] | +0.040 [-0.055, +0.213] |
| 2 | 0.71 | 5.0 | +0.218 | -0.398 [-0.533, -0.270] | -0.522 [-0.653, -0.399] | +0.232 [+0.108, +0.331] | +0.191 | -0.424 [-0.723, -0.229] | -0.549 [-0.831, -0.351] | +0.205 [-0.080, +0.373] | -0.026 [-0.198, +0.073] |

## How the co-field tables are read

Values of h_f c_f C_f at the mean position of each ring carbon in the frame of the carbonyl carbon: what one unit of the field at that place adds to the band, in cm^-1.  The number in brackets is the standard deviation that the noise in the bands (taken as the cross-validation error) allows.  *Para minus meta* is |para| - |meta| with its standard deviation and z-score from the same noise.  *Para over meta* is (|para| - |meta|) / (|para| + |meta|), with the mean and standard deviation of a y-scramble null and the z-score against it.

## The co-field at the ring positions, geometries of the quantum optimisation, width 10.00 A (rho 0.01, posed q2 +0.725)

Figure: `cofield_rho_0p01.png`.

| field | ipso | ortho | meta | para | para minus meta (sd, z) | para over meta | null | z |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| electrostatic | +0.277 (0.17) | +1.23 (0.26) | +3.58 (0.58) | +4.91 (0.59) | +1.33 (0.31, +4.3) | +0.16 | +0.02 +/- 0.46 | +0.3 |
| steric | +0.0139 (0.0043) | +0.022 (0.0068) | +0.0224 (0.014) | +0.000265 (0.018) | -0.0221 (0.007, -3.2) | -0.98 | +0.10 +/- 0.48 | -2.3 |
| hydrophobic | +0.00999 (0.018) | +0.0619 (0.052) | +0.0587 (0.16) | +0.154 (0.21) | +0.0955 (0.11, +0.9) | +0.45 | +0.05 +/- 0.37 | +1.1 |
| hba | +0.0256 (0.0019) | -0.000678 (0.0034) | -0.224 (0.05) | -0.717 (0.15) | +0.493 (0.1, +4.8) | +0.52 | +0.43 +/- 0.30 | +0.3 |
| hbd | +0.000974 (0.00027) | +0.000313 (0.00092) | -0.0315 (0.014) | -0.143 (0.058) | +0.111 (0.045, +2.5) | +0.64 | +0.53 +/- 0.31 | +0.4 |

## The co-field at the ring positions, geometries of the quantum optimisation, width 4.47 A (rho 0.05, posed q2 +0.733)

Figure: `cofield_rho_0p05.png`.

| field | ipso | ortho | meta | para | para minus meta (sd, z) | para over meta | null | z |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| electrostatic | -1.09 (1.1) | +3.24 (1.2) | +4.47 (2) | +6.5 (2.4) | +2.03 (1.3, +1.6) | +0.18 | +0.10 +/- 0.50 | +0.2 |
| steric | +0.0766 (0.022) | +0.121 (0.034) | +0.138 (0.064) | +0.0345 (0.085) | -0.104 (0.032, -3.3) | -0.60 | +0.14 +/- 0.34 | -2.1 |
| hydrophobic | -0.187 (0.11) | +0.275 (0.29) | -0.868 (0.7) | -0.665 (0.95) | -0.203 (0.58, -0.3) | -0.13 | +0.12 +/- 0.42 | -0.6 |
| hba | +0.31 (0.024) | +0.0554 (0.015) | -0.735 (0.19) | -2.43 (0.57) | +1.69 (0.39, +4.3) | +0.54 | +0.49 +/- 0.29 | +0.2 |
| hbd | +0.00845 (0.0013) | +0.00265 (0.0036) | -0.102 (0.047) | -0.465 (0.2) | +0.363 (0.15, +2.4) | +0.64 | +0.56 +/- 0.31 | +0.3 |

## The co-field at the ring positions, geometries of the quantum optimisation, width 3.16 A (rho 0.1, posed q2 +0.869)

Figure: `cofield_rho_0p1.png`.

| field | ipso | ortho | meta | para | para minus meta (sd, z) | para over meta | null | z |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| electrostatic | -3.39 (10) | +53.2 (8.2) | +2 (11) | -18 (12) | +16 (22, +0.7) | +0.80 | +0.04 +/- 0.44 | +1.7 |
| steric | +0.623 (0.37) | +0.953 (0.46) | +1.5 (0.69) | +1.04 (0.83) | -0.462 (0.28, -1.6) | -0.18 | +0.10 +/- 0.35 | -0.8 |
| hydrophobic | -16 (2.4) | +3.33 (3.5) | +5.39 (7.2) | -8.54 (6) | +3.15 (12, +0.3) | +0.23 | -0.01 +/- 0.43 | +0.6 |
| hba | +9.01 (0.7) | +2.04 (0.21) | -1.54 (1.6) | -5.73 (4.6) | +4.19 (3.1, +1.3) | +0.58 | +0.44 +/- 0.33 | +0.4 |
| hbd | +0.224 (0.034) | +0.0382 (0.035) | -0.429 (0.42) | -2.05 (1.7) | +1.62 (1.3, +1.2) | +0.65 | +0.52 +/- 0.35 | +0.4 |

## The co-field at the ring positions, start geometries, width 10.00 A (rho 0.01, posed q2 +0.702)

Figure: `start_cofield_rho_0p01.png`.

| field | ipso | ortho | meta | para | para minus meta (sd, z) | para over meta | null | z |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| electrostatic | +0.268 (0.18) | +1.11 (0.27) | +3.68 (0.59) | +5.01 (0.61) | +1.33 (0.31, +4.3) | +0.15 | +0.03 +/- 0.42 | +0.3 |
| steric | +0.0158 (0.0045) | +0.0245 (0.0072) | +0.031 (0.015) | +0.0156 (0.02) | -0.0154 (0.0074, -2.1) | -0.33 | +0.09 +/- 0.42 | -1.0 |
| hydrophobic | +0.0137 (0.018) | +0.0488 (0.053) | +0.0347 (0.17) | +0.136 (0.22) | +0.102 (0.11, +0.9) | +0.59 | +0.08 +/- 0.39 | +1.3 |
| hba | +0.00515 (0.003) | -0.00348 (0.0035) | -0.189 (0.05) | -0.652 (0.15) | +0.462 (0.11, +4.4) | +0.55 | +0.49 +/- 0.23 | +0.3 |
| hbd | +0.00324 (0.0036) | +0.000207 (0.0011) | -0.0351 (0.014) | -0.159 (0.059) | +0.124 (0.045, +2.7) | +0.64 | +0.53 +/- 0.30 | +0.4 |

## The co-field at the ring positions, start geometries, width 4.47 A (rho 0.05, posed q2 +0.691)

Figure: `start_cofield_rho_0p05.png`.

| field | ipso | ortho | meta | para | para minus meta (sd, z) | para over meta | null | z |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| electrostatic | -0.45 (1.2) | +2.41 (1.3) | +5.32 (2.2) | +8.03 (2.6) | +2.71 (1.4, +1.9) | +0.20 | +0.05 +/- 0.49 | +0.3 |
| steric | +0.0995 (0.027) | +0.148 (0.038) | +0.188 (0.071) | +0.107 (0.095) | -0.0813 (0.035, -2.3) | -0.28 | +0.16 +/- 0.37 | -1.2 |
| hydrophobic | +0.00407 (0.13) | +0.147 (0.32) | -1.12 (0.75) | -0.906 (1) | -0.213 (0.64, -0.3) | -0.11 | +0.10 +/- 0.39 | -0.5 |
| hba | +0.0481 (0.028) | -0.00316 (0.016) | -0.631 (0.2) | -2.22 (0.6) | +1.59 (0.42, +3.8) | +0.56 | +0.47 +/- 0.26 | +0.3 |
| hbd | +0.0138 (0.013) | +0.00228 (0.0044) | -0.107 (0.051) | -0.491 (0.21) | +0.384 (0.16, +2.4) | +0.64 | +0.50 +/- 0.33 | +0.4 |

## The co-field at the ring positions, start geometries, width 3.16 A (rho 0.1, posed q2 +0.663)

Figure: `start_cofield_rho_0p1.png`.

| field | ipso | ortho | meta | para | para minus meta (sd, z) | para over meta | null | z |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| electrostatic | -5.54 (6.1) | +7.19 (4.4) | -9.4 (6.7) | -8.86 (8.5) | -0.547 (5, -0.1) | -0.03 | -0.00 +/- 0.48 | -0.1 |
| steric | +0.391 (0.17) | +0.602 (0.19) | +0.99 (0.29) | +0.872 (0.36) | -0.118 (0.12, -1.0) | -0.06 | +0.16 +/- 0.31 | -0.7 |
| hydrophobic | -1.58 (1) | +0.196 (1.5) | -6.57 (3.1) | -9.66 (3.6) | +3.09 (2.4, +1.3) | +0.19 | +0.02 +/- 0.51 | +0.3 |
| hba | +0.495 (0.23) | +0.0742 (0.08) | -2.55 (0.64) | -9.07 (2) | +6.52 (1.4, +4.7) | +0.56 | +0.49 +/- 0.24 | +0.3 |
| hbd | +0.00946 (0.024) | +0.00746 (0.017) | -0.291 (0.17) | -1.32 (0.66) | +1.03 (0.5, +2.0) | +0.64 | +0.52 +/- 0.31 | +0.4 |

**Reading.** A linear regression on the Hammett constant, with one equation for all acids (the Hammett regression), predicts the band with q2 +0.740. On the geometries of the quantum optimisation the count reaches q2 +0.662; the invariant local kernel is best at a width of 1.00 A (q2 +0.976, against the count +0.314 [+0.257, +0.388]), the posed overlap at 1.00 A (q2 +0.978, against the count +0.316 [+0.259, +0.390]). A linear regression on the length of the C=O bond of the same geometries reaches q2 +0.973. The posed overlap at its best width is +0.005 [+0.002, +0.009] against that regression, so nearly all of what the narrow field predicts is one bond length. That length comes from the quantum optimisation, the calculation that also gives the band, so this is not a prediction from the structure. On the start geometries, which hold nothing from the quantum calculation, the count reaches q2 +0.616; the invariant local kernel is best at a width of 10.00 A (q2 +0.746, against the count +0.130 [+0.043, +0.209], against the Hammett regression +0.006 [-0.065, +0.077]), the posed overlap at 10.00 A (q2 +0.702, against the count +0.087 [+0.002, +0.168]). At a width of 1.00 A or less neither kernel is above q2 +0.231 there. The first half of the gate, the local field beating the count by more than the interval, is read on the start geometries and is met: the lower end of the interval is +0.043 for the invariant kernel and +0.002 for the posed overlap, each at a width chosen as the best of the scan. At a width of 10.00 A (posed q2 +0.702) the electrostatic co-field is +3.68 at meta and +5.01 at para, in cm^-1 per unit of field; para minus meta in absolute value is +1.33 with a z of +4.3 against the noise and +0.3 against the scramble. At a width of 4.47 A (posed q2 +0.691) the electrostatic co-field is +5.32 at meta and +8.03 at para, in cm^-1 per unit of field; para minus meta in absolute value is +2.71 with a z of +1.9 against the noise and +0.3 against the scramble. At a width of 3.16 A (posed q2 +0.663) the electrostatic co-field is -9.4 at meta and -8.86 at para, in cm^-1 per unit of field; para minus meta in absolute value is -0.547 with a z of -0.1 against the noise and -0.1 against the scramble.
