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

## Indicator fields: one field per atom type (plan item LP3), on the start geometries

A continuous indicator field of an atom type is one Gaussian on every atom of that type.  15 Sybyl types occur in at least 10 acids (C.1, C.2, C.3, C.ar, Cl, F, H, N.1, N.am, N.pl3, O.2, O.3, O.co2, S.3, S.O2) and each gets a field.  The last field set of each table is a control with a field for every type that occurs in at least 2 acids (25 types; it adds B, Br, I, N.2, N.3, P.3, S.2, S.O, Se, Si).

The cross-validation is repeated 10 times with other folds.  A q2 is the mean over the repeats, and the number in brackets is its standard deviation over the repeats.  The ridge is chosen on the folds of each repeat, for every descriptor alike.  A difference in q2 carries its 95 % bootstrap interval over the units that the folds hold out.

The series holds each substituent twice, meta and para to the carboxyl group (95 substituents in 189 acids).  Over the 94 substituents that occur twice, the band of the meta acid and the band of the para acid correlate with r +0.80, and their Hammett constants with r +0.92.  A fold that holds out one acid therefore usually keeps the other acid of the same substituent in the training set.  The tables are given for both kinds of folds.

The first repeat uses the folds of the tables above.  On those folds the atom-centred counts reach q2 +0.616; over the 10 repeats they lie between +0.616 and +0.666.  The invariant kernel of the five fields at the width of 10.00 A reaches +0.746 on the first repeat and lies between +0.692 and +0.767.  The margin of the five fields over the counts in the tables above is read on that one set of folds.  The column *against the atom-centred counts* below gives it over all repeats.

### Indicator fields, whole acids held out

| reference | q2 (sd over the repeats) | RMSE, cm^-1 |
|---|---:|---:|
| atom-centred counts | +0.650 (0.017) | 2.65 |
| linear regression on the Hammett constant, all acids | +0.742 (0.002) | 2.27 |
| linear regression on the Hammett constant, meta and para separately | +0.782 (0.002) | 2.09 |

| fields | number of fields | width, A | invariant local q2 (sd over the repeats) | against the atom-centred counts | against the five fields | against the Hammett regression, all acids | against the Hammett regressions, meta and para separately | posed overlap q2 (sd over the repeats) | against the five fields | against the Hammett regressions, meta and para separately |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| the five physico-chemical fields | 5 | 10.00 | +0.741 (0.026) | +0.091 [+0.017, +0.162] |  | -0.002 [-0.075, +0.064] | -0.041 [-0.110, +0.022] | +0.707 (0.006) |  | -0.074 [-0.132, -0.017] |
| the five physico-chemical fields | 5 | 7.07 | +0.735 (0.021) | +0.085 [+0.014, +0.154] |  | -0.008 [-0.075, +0.058] | -0.047 [-0.114, +0.015] | +0.699 (0.004) |  | -0.083 [-0.139, -0.024] |
| the five physico-chemical fields | 5 | 4.47 | +0.701 (0.016) | +0.051 [-0.023, +0.120] |  | -0.042 [-0.123, +0.031] | -0.081 [-0.162, -0.012] | +0.698 (0.008) |  | -0.083 [-0.146, -0.020] |
| the five physico-chemical fields | 5 | 3.16 | +0.664 (0.013) | +0.015 [-0.064, +0.090] |  | -0.078 [-0.161, -0.003] | -0.118 [-0.202, -0.045] | +0.670 (0.009) |  | -0.111 [-0.183, -0.045] |
| 15 indicator fields | 15 | 10.00 | +0.749 (0.022) | +0.099 [+0.013, +0.173] | +0.008 [-0.080, +0.094] | +0.006 [-0.083, +0.085] | -0.033 [-0.122, +0.046] | +0.648 (0.027) | -0.059 [-0.194, +0.039] | -0.133 [-0.272, -0.028] |
| 15 indicator fields | 15 | 7.07 | +0.738 (0.022) | +0.088 [-0.013, +0.171] | +0.003 [-0.102, +0.098] | -0.005 [-0.110, +0.085] | -0.044 [-0.149, +0.043] | +0.637 (0.029) | -0.062 [-0.220, +0.045] | -0.145 [-0.294, -0.033] |
| 15 indicator fields | 15 | 4.47 | +0.685 (0.022) | +0.036 [-0.117, +0.145] | -0.015 [-0.172, +0.112] | -0.057 [-0.217, +0.060] | -0.096 [-0.257, +0.013] | +0.598 (0.029) | -0.100 [-0.262, +0.012] | -0.184 [-0.358, -0.063] |
| 15 indicator fields | 15 | 3.16 | +0.618 (0.024) | -0.031 [-0.216, +0.089] | -0.046 [-0.222, +0.086] | -0.124 [-0.317, +0.007] | -0.164 [-0.357, -0.038] | +0.557 (0.029) | -0.114 [-0.290, +0.009] | -0.225 [-0.419, -0.092] |
| the five fields and 15 indicator fields | 20 | 10.00 | +0.776 (0.017) | +0.127 [+0.045, +0.199] | +0.036 [-0.047, +0.116] | +0.034 [-0.053, +0.116] | -0.005 [-0.090, +0.072] | +0.718 (0.025) | +0.011 [-0.088, +0.083] | -0.064 [-0.166, +0.021] |
| the five fields and 15 indicator fields | 20 | 7.07 | +0.767 (0.017) | +0.117 [+0.020, +0.197] | +0.032 [-0.065, +0.121] | +0.024 [-0.077, +0.114] | -0.015 [-0.116, +0.068] | +0.713 (0.025) | +0.015 [-0.104, +0.102] | -0.068 [-0.182, +0.025] |
| the five fields and 15 indicator fields | 20 | 4.47 | +0.720 (0.018) | +0.071 [-0.068, +0.170] | +0.020 [-0.122, +0.138] | -0.022 [-0.172, +0.088] | -0.061 [-0.209, +0.043] | +0.673 (0.026) | -0.025 [-0.152, +0.065] | -0.109 [-0.251, -0.003] |
| the five fields and 15 indicator fields | 20 | 3.16 | +0.634 (0.021) | -0.015 [-0.191, +0.103] | -0.030 [-0.202, +0.096] | -0.108 [-0.303, +0.020] | -0.148 [-0.339, -0.023] | +0.586 (0.030) | -0.084 [-0.251, +0.030] | -0.195 [-0.380, -0.069] |
| the five fields and 25 indicator fields | 30 | 10.00 | +0.784 (0.018) | +0.135 [+0.049, +0.210] | +0.044 [-0.037, +0.123] | +0.042 [-0.050, +0.123] | +0.003 [-0.086, +0.080] | +0.700 (0.021) | -0.007 [-0.106, +0.067] | -0.082 [-0.187, +0.003] |
| the five fields and 25 indicator fields | 30 | 7.07 | +0.775 (0.018) | +0.126 [+0.026, +0.208] | +0.040 [-0.062, +0.128] | +0.033 [-0.078, +0.122] | -0.007 [-0.114, +0.079] | +0.689 (0.025) | -0.010 [-0.126, +0.074] | -0.093 [-0.213, -0.003] |
| the five fields and 25 indicator fields | 30 | 4.47 | +0.730 (0.019) | +0.081 [-0.070, +0.187] | +0.030 [-0.126, +0.154] | -0.012 [-0.177, +0.101] | -0.051 [-0.213, +0.059] | +0.651 (0.031) | -0.047 [-0.182, +0.050] | -0.130 [-0.288, -0.019] |
| the five fields and 25 indicator fields | 30 | 3.16 | +0.658 (0.023) | +0.009 [-0.170, +0.131] | -0.006 [-0.184, +0.127] | -0.084 [-0.283, +0.045] | -0.123 [-0.316, +0.005] | +0.585 (0.033) | -0.086 [-0.258, +0.034] | -0.197 [-0.386, -0.067] |

### Indicator fields, whole substituents held out (both acids of a substituent together)

| reference | q2 (sd over the repeats) | RMSE, cm^-1 |
|---|---:|---:|
| atom-centred counts | +0.623 (0.009) | 2.74 |
| linear regression on the Hammett constant, all acids | +0.742 (0.002) | 2.27 |
| linear regression on the Hammett constant, meta and para separately | +0.782 (0.002) | 2.09 |

| fields | number of fields | width, A | invariant local q2 (sd over the repeats) | against the atom-centred counts | against the five fields | against the Hammett regression, all acids | against the Hammett regressions, meta and para separately | posed overlap q2 (sd over the repeats) | against the five fields | against the Hammett regressions, meta and para separately |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| the five physico-chemical fields | 5 | 10.00 | +0.676 (0.015) | +0.053 [-0.097, +0.168] |  | -0.066 [-0.201, +0.031] | -0.106 [-0.243, -0.011] | +0.703 (0.011) |  | -0.078 [-0.141, -0.016] |
| the five physico-chemical fields | 5 | 7.07 | +0.674 (0.023) | +0.051 [-0.076, +0.154] |  | -0.068 [-0.177, +0.019] | -0.108 [-0.221, -0.024] | +0.695 (0.009) |  | -0.086 [-0.146, -0.025] |
| the five physico-chemical fields | 5 | 4.47 | +0.648 (0.017) | +0.025 [-0.088, +0.124] |  | -0.094 [-0.196, -0.006] | -0.133 [-0.238, -0.049] | +0.695 (0.012) |  | -0.086 [-0.161, -0.022] |
| the five physico-chemical fields | 5 | 3.16 | +0.672 (0.016) | +0.048 [-0.054, +0.139] |  | -0.070 [-0.138, -0.005] | -0.110 [-0.185, -0.043] | +0.670 (0.008) |  | -0.111 [-0.187, -0.045] |
| 15 indicator fields | 15 | 10.00 | +0.675 (0.014) | +0.052 [-0.060, +0.150] | -0.000 [-0.165, +0.160] | -0.066 [-0.214, +0.044] | -0.106 [-0.262, +0.005] | +0.630 (0.014) | -0.073 [-0.226, +0.039] | -0.151 [-0.312, -0.037] |
| 15 indicator fields | 15 | 7.07 | +0.660 (0.016) | +0.036 [-0.089, +0.145] | -0.014 [-0.187, +0.135] | -0.082 [-0.253, +0.038] | -0.122 [-0.296, -0.001] | +0.621 (0.016) | -0.075 [-0.248, +0.049] | -0.161 [-0.345, -0.030] |
| 15 indicator fields | 15 | 4.47 | +0.597 (0.017) | -0.026 [-0.209, +0.108] | -0.051 [-0.269, +0.108] | -0.145 [-0.375, +0.004] | -0.185 [-0.417, -0.037] | +0.586 (0.020) | -0.110 [-0.292, +0.012] | -0.196 [-0.412, -0.054] |
| 15 indicator fields | 15 | 3.16 | +0.552 (0.015) | -0.071 [-0.283, +0.077] | -0.120 [-0.361, +0.035] | -0.190 [-0.448, -0.029] | -0.230 [-0.492, -0.072] | +0.553 (0.021) | -0.117 [-0.320, +0.010] | -0.229 [-0.464, -0.079] |
| the five fields and 15 indicator fields | 20 | 10.00 | +0.705 (0.015) | +0.082 [-0.027, +0.177] | +0.029 [-0.130, +0.181] | -0.037 [-0.185, +0.073] | -0.076 [-0.229, +0.033] | +0.697 (0.015) | -0.006 [-0.120, +0.082] | -0.085 [-0.211, +0.015] |
| the five fields and 15 indicator fields | 20 | 7.07 | +0.693 (0.017) | +0.070 [-0.052, +0.172] | +0.020 [-0.144, +0.154] | -0.049 [-0.212, +0.068] | -0.088 [-0.257, +0.028] | +0.689 (0.015) | -0.006 [-0.137, +0.095] | -0.092 [-0.235, +0.019] |
| the five fields and 15 indicator fields | 20 | 4.47 | +0.645 (0.024) | +0.022 [-0.127, +0.136] | -0.003 [-0.184, +0.136] | -0.097 [-0.292, +0.034] | -0.137 [-0.337, -0.007] | +0.647 (0.016) | -0.048 [-0.199, +0.057] | -0.135 [-0.321, -0.007] |
| the five fields and 15 indicator fields | 20 | 3.16 | +0.576 (0.015) | -0.047 [-0.250, +0.096] | -0.096 [-0.332, +0.051] | -0.166 [-0.420, -0.011] | -0.206 [-0.464, -0.052] | +0.568 (0.020) | -0.103 [-0.300, +0.023] | -0.214 [-0.447, -0.064] |
| the five fields and 25 indicator fields | 30 | 10.00 | +0.709 (0.015) | +0.086 [-0.039, +0.188] | +0.033 [-0.135, +0.186] | -0.033 [-0.195, +0.082] | -0.073 [-0.238, +0.042] | +0.678 (0.012) | -0.025 [-0.146, +0.064] | -0.103 [-0.232, -0.011] |
| the five fields and 25 indicator fields | 30 | 7.07 | +0.691 (0.017) | +0.068 [-0.076, +0.181] | +0.018 [-0.161, +0.159] | -0.051 [-0.235, +0.076] | -0.090 [-0.281, +0.036] | +0.662 (0.013) | -0.033 [-0.179, +0.070] | -0.120 [-0.271, -0.008] |
| the five fields and 25 indicator fields | 30 | 4.47 | +0.636 (0.021) | +0.012 [-0.178, +0.150] | -0.012 [-0.231, +0.134] | -0.106 [-0.342, +0.045] | -0.146 [-0.383, +0.003] | +0.623 (0.019) | -0.073 [-0.251, +0.043] | -0.159 [-0.368, -0.017] |
| the five fields and 25 indicator fields | 30 | 3.16 | +0.583 (0.016) | -0.040 [-0.253, +0.109] | -0.088 [-0.338, +0.062] | -0.159 [-0.416, +0.003] | -0.198 [-0.462, -0.037] | +0.571 (0.022) | -0.099 [-0.313, +0.030] | -0.210 [-0.456, -0.059] |

### How alike the kernel of the indicator fields and the kernel of the five fields are

The correlation between the off-diagonal elements of the two kernels, each centred first.  Near 1, the indicator fields would order the pairs of acids as the five fields do and could add little.

| width, A | invariant local kernel | posed overlap kernel |
|---:|---:|---:|
| 10.00 | +0.333 | +0.377 |
| 7.07 | +0.306 | +0.359 |
| 4.47 | +0.211 | +0.272 |
| 3.16 | +0.120 | +0.132 |

### What one atom of a type adds to the band at the place of the first atom of the substituent, width 10.00 A

The posed overlap model with the five fields and the 15 indicator fields (q2 +0.718, ridge 1).  The places are the mean positions of the first atom of the substituent in the frame of the carbonyl carbon: meta (carbonyl side), 5.01 A from the carbonyl carbon, 44 acids, envelope 0.78; meta (hydroxyl side), 5.10 A from the carbonyl carbon, 50 acids, envelope 0.77; para, 5.84 A from the carbonyl carbon, 94 acids, envelope 0.71.  *Effect* is what one atom of the type at that place adds to the band, in cm^-1, with the standard deviation that the noise in the bands allows.  *Acids* is the number of acids whose substituent starts with an atom of that type at that place, and *mean shift* is their mean band minus the band of the unsubstituted acid (1727.9 cm^-1).  The model was fitted on these bands, so the comparison is a check of what the model has stored and not a prediction.

| atom type | effect, meta (carbonyl side) | acids | mean shift, cm^-1 | effect, meta (hydroxyl side) | acids | mean shift, cm^-1 | effect, para | acids | mean shift, cm^-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C.1 | +2.29 (1.30) | 2 | +5.2 | -0.47 (1.75) | 0 |  | -0.07 (1.22) | 2 | +0.8 |
| C.2 | +0.37 (0.50) | 6 | +3.8 | +0.27 (0.65) | 5 | +2.0 | -0.53 (0.61) | 11 | +1.5 |
| C.3 | -0.17 (0.59) | 15 | -0.9 | -0.13 (0.57) | 11 | +0.9 | -0.78 (0.51) | 26 | -1.2 |
| C.ar | -0.03 (0.11) | 1 | -1.0 | -0.14 (0.06) | 0 |  | -0.25 (0.12) | 1 | -3.9 |
| Cl | +0.46 (1.16) | 0 |  | +1.63 (2.16) | 1 | +3.3 | +0.01 (2.11) | 1 | +0.3 |
| F | +0.75 (0.84) | 0 |  | +1.12 (0.98) | 1 | +3.0 | +0.32 (0.83) | 1 | -0.5 |
| H | -0.26 (0.15) | 0 |  | -0.53 (0.17) | 0 |  | -0.15 (0.21) | 0 |  |
| N.1 | +4.26 (2.26) | 1 | +5.5 | +1.84 (1.90) | 0 |  | +2.97 (2.09) | 1 | +3.1 |
| N.am | -2.72 (1.45) | 4 | -2.3 | +1.82 (1.40) | 4 | +3.1 | -1.12 (1.07) | 8 | -3.9 |
| N.pl3 | -1.66 (1.32) | 4 | -3.1 | -0.42 (1.47) | 3 | +0.1 | -4.94 (1.05) | 7 | -9.0 |
| O.2 | +2.74 (1.15) | 0 |  | +2.88 (1.44) | 0 |  | +2.71 (1.38) | 0 |  |
| O.3 | -0.77 (0.52) | 3 | -3.0 | -0.04 (0.54) | 6 | +1.0 | -0.47 (0.54) | 9 | -4.7 |
| O.co2 | +0.53 (0.16) | 0 |  | +1.04 (0.49) | 0 |  | +2.22 (0.48) | 0 |  |
| S.3 | -1.30 (1.40) | 5 | -0.7 | +0.15 (1.68) | 3 | +3.2 | -3.97 (1.15) | 8 | -2.6 |
| S.O2 | +3.74 (2.25) | 1 | +9.1 | +2.39 (1.45) | 5 | +5.0 | +0.93 (1.49) | 6 | +5.3 |

Over the 22 pairs of a type and a place with at least 2 acids, the effects correlate with the mean shifts with r +0.85 and with the mean Hammett constants of those acids with r +0.59.

**Reading of the indicator fields.** The statements are made at the width of 10.00 A, fixed before the measurement as the width where the five fields were best, and for the invariant local kernel. With whole acids held out the five fields reach q2 +0.741, the indicator fields alone +0.749 and both together +0.776. Both together are +0.036 [-0.047, +0.116] against the five fields (not separated from them) and -0.005 [-0.090, +0.072] against the Hammett regressions for meta and para separately, which reach +0.782 (not separated from them). Against the atom-centred counts (+0.650) the five fields are +0.091 [+0.017, +0.162] (above them) and both together +0.127 [+0.045, +0.199] (above them). With whole substituents held out the five fields reach q2 +0.676, the indicator fields alone +0.675 and both together +0.705. Both together are +0.029 [-0.130, +0.181] against the five fields (not separated from them) and -0.076 [-0.229, +0.033] against the Hammett regressions for meta and para separately, which reach +0.782 (not separated from them). Against the atom-centred counts (+0.623) the five fields are +0.053 [-0.097, +0.168] (not separated from them) and both together +0.082 [-0.027, +0.177] (not separated from them). The posed overlap with both field sets reaches q2 +0.718 with whole acids held out, against +0.707 with the five fields (+0.011 [-0.088, +0.083], not separated from them). The kernel of the indicator fields correlates with the kernel of the five fields with r +0.33 (invariant) and +0.38 (posed), so the indicator fields are a different description and not a copy of the five fields. Holding out whole substituents lowers the q2 of the five fields by 0.065 and of both field sets by 0.071. It does not change the Hammett regressions, because a Hammett constant is a measured value of the substituent, and a field model has to learn a substituent from the other substituents. The effects of single atoms follow the mean observed shifts by the type and the place of the first atom (r +0.85 over 22 pairs), so the model has stored the substituent effects by atom type and place. **Answer to the question of plan item SP2.** The five fields with the indicator fields do not pass the Hammett regressions for meta and para separately: the difference in q2 is -0.005 [-0.090, +0.072] with whole acids held out and -0.076 [-0.229, +0.033] with whole substituents held out.

## Quantum charges: the electrostatic field from GFN2-xTB charges, on the start geometries

The electrostatic field of the five fields is built from Gasteiger charges.  That scheme computes the charges from the bonds alone and moves charge along the sigma bonds only.  It does not hold the effect of a substituent through the pi electrons of the ring (resonance).  In this part the charge of every atom is taken from one calculation with the semi-empirical quantum method GFN2-xTB acc=1 (program xtb) on the start geometry.  The geometry is not optimised, so the description holds nothing about the geometry of the ORCA run that gives the band.  The charges are in `examples/spectra/benzoic_acids_xtb.sdf`.

The same calculation gives the two Fukui indices of every atom: the change of the charge of the atom when one electron is added to the molecule, and when one electron is removed.  One field set adds a field for each of the two indices (fukui_plus, fukui_minus).  Another adds the 15 indicator fields of the part above.

The cross-validation is repeated 10 times with other folds, with whole acids held out and with whole substituents held out, as in the part above.  A difference in q2 carries its 95 % bootstrap interval over the units that the folds hold out.

### The charges of the four atoms of the carboxyl group

| atom | quantum charge, mean (sd over the acids), e | quantum charge: r with the band | quantum charge: r with the Hammett constant | Gasteiger charge, mean (sd over the acids), e | Gasteiger charge: r with the band |
|---|---:|---:|---:|---:|---:|
| carbonyl carbon | +0.350 (0.0018) | +0.590 | +0.608 | +0.335 (0.0001) | -0.024 |
| carbonyl oxygen | -0.418 (0.0083) | +0.875 | +0.909 | -0.246 (0.0000) | -0.067 |
| hydroxyl oxygen | -0.378 (0.0041) | +0.797 | +0.869 | -0.478 (0.0000) | -0.067 |
| hydroxyl hydrogen | +0.335 (0.0036) | +0.801 | +0.857 | +0.297 (0.0000) | -0.083 |

A multiple linear regression of the band on the four quantum charges, fitted on all acids, has r2 0.805.  The charges of the carboxyl group are computed numbers that could hold the whole answer, as the length of the C=O bond did on the optimised geometries.  The references below therefore include the linear regression on the charge of the carbonyl oxygen and the multiple linear regression on the four charges, both fitted inside the folds.  A field with quantum charges that is no better than these regressions has used the charges of the carboxyl group and not the field of the acid.

### Quantum charges, whole acids held out: the references

| reference | q2 (sd over the repeats) | RMSE, cm^-1 | RMSE of the meta acids, cm^-1 | RMSE of the para acids, cm^-1 | acid with the largest error (its RMSE, cm^-1; its part of the squared error) | q2 scored without the acids of SF5 (sd over the repeats) |
|---|---:|---:|---:|---:|---|---:|
| atom-centred counts | +0.650 (0.017) | 2.65 | 2.45 | 2.83 | para-N(C2H5)2 (9.2; 6 %) | +0.649 (0.016) |
| linear regression on the Hammett constant, all acids | +0.742 (0.002) | 2.27 | 2.30 | 2.23 | para-N(C2H5)2 (6.2; 4 %) | +0.738 (0.002) |
| linear regression on the Hammett constant, meta and para separately | +0.782 (0.002) | 2.09 | 2.13 | 2.03 | meta-N(CH3)COCH3 (6.4; 5 %) | +0.778 (0.002) |
| linear regression on the quantum charge of the carbonyl oxygen | +0.758 (0.003) | 2.20 | 1.96 | 2.40 | para-NHCH3 (6.4; 5 %) | +0.753 (0.003) |
| multiple linear regression on the quantum charges of the four atoms of the carboxyl group | +0.786 (0.004) | 2.07 | 1.88 | 2.24 | para-SeCH3 (7.4; 7 %) | +0.790 (0.003) |

### Quantum charges, whole acids held out: the width and the ridge chosen inside the folds, invariant local kernel

Every training part is divided again into 5 folds over the same unit.  The width of the envelope (one of 10.00, 4.47, 2.00, 1.00 A) and the ridge are the pair with the best q2 on these inner folds, so the held-out acids take no part in the choice.  These are the rows that the reading below uses.

| fields | number of fields | q2 (sd over the repeats) | number of folds that chose the width 10.00 / 4.47 / 2.00 / 1.00 A | RMSE of the meta acids, cm^-1 | RMSE of the para acids, cm^-1 | against the five fields with Gasteiger charges | against the five fields with quantum charges | against the Hammett regressions, meta and para separately | against the regression on the four charges of the carboxyl group |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| the five fields, Gasteiger charges | 5 | +0.718 (0.033) | 99 / 1 / 0 / 0 | 2.17 | 2.57 |  |  | -0.064 [-0.135, +0.000] | -0.068 [-0.129, -0.013] |
| the electrostatic field alone, Gasteiger charges | 1 | +0.562 (0.044) | 22 / 78 / 0 / 0 | 3.01 | 2.91 | -0.156 [-0.290, -0.051] |  | -0.220 [-0.358, -0.117] | -0.224 [-0.357, -0.120] |
| the electrostatic field alone, quantum charges | 1 | +0.824 (0.004) | 0 / 0 / 100 / 0 | 1.98 | 1.77 |  |  | +0.042 [-0.005, +0.093] | +0.038 [-0.001, +0.084] |
| the five fields, quantum charges | 5 | +0.651 (0.238) | 0 / 21 / 75 / 4 | 3.08 | 2.12 | -0.066 [-0.414, +0.129] |  | -0.130 [-0.487, +0.064] | -0.135 [-0.486, +0.042] |
| the five fields with quantum charges and the two Fukui fields | 7 | +0.837 (0.022) | 0 / 13 / 71 / 16 | 1.84 | 1.77 | +0.120 [+0.063, +0.184] | +0.186 [+0.024, +0.502] | +0.056 [-0.006, +0.120] | +0.052 [-0.003, +0.108] |
| the five fields with quantum charges and 15 indicator fields | 20 | +0.692 (0.127) | 14 / 1 / 69 / 16 | 2.66 | 2.29 | -0.026 [-0.194, +0.099] | +0.040 [-0.082, +0.233] | -0.090 [-0.261, +0.034] | -0.094 [-0.263, +0.018] |

Where the error of these rows is.  The last three columns score the same predictions without the acids of the substituent SF5; nothing is fitted again.

| fields | acid with the largest error (its RMSE, cm^-1; its part of the squared error) | q2 scored without the acids of SF5 (sd over the repeats) | without them, against the Hammett regressions, meta and para separately | without them, against the regression on the four charges of the carboxyl group |
|---|---|---:|---:|---:|
| the five fields, Gasteiger charges | para-SeCH3 (8.1; 6 %) | +0.719 (0.035) | -0.059 [-0.129, +0.003] | -0.071 [-0.135, -0.015] |
| the electrostatic field alone, Gasteiger charges | meta-C(CF3)3 (13.7; 11 %) | +0.555 (0.046) | -0.223 [-0.381, -0.122] | -0.235 [-0.386, -0.135] |
| the electrostatic field alone, quantum charges | meta-CN (5.7; 5 %) | +0.822 (0.002) | +0.044 [-0.004, +0.095] | +0.032 [-0.006, +0.076] |
| the five fields, quantum charges | meta-SF5 (24.0; 44 %) | +0.815 (0.011) | +0.036 [-0.014, +0.091] | +0.024 [-0.004, +0.054] |
| the five fields with quantum charges and the two Fukui fields | meta-SF5 (7.6; 9 %) | +0.853 (0.007) | +0.075 [+0.016, +0.134] | +0.063 [+0.017, +0.110] |
| the five fields with quantum charges and 15 indicator fields | meta-SF5 (16.4; 23 %) | +0.780 (0.030) | +0.001 [-0.061, +0.064] | -0.011 [-0.065, +0.041] |

### Quantum charges, whole acids held out: the width and the ridge chosen inside the folds, posed overlap kernel

| fields | number of fields | q2 (sd over the repeats) | number of folds that chose the width 10.00 / 4.47 / 2.00 / 1.00 A | RMSE of the meta acids, cm^-1 | RMSE of the para acids, cm^-1 | against the five fields with Gasteiger charges | against the five fields with quantum charges | against the Hammett regressions, meta and para separately | against the regression on the four charges of the carboxyl group |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| the five fields, Gasteiger charges | 5 | +0.688 (0.014) | 55 / 45 / 0 / 0 | 2.43 | 2.57 |  |  | -0.094 [-0.155, -0.033] | -0.098 [-0.164, -0.030] |
| the electrostatic field alone, Gasteiger charges | 1 | +0.652 (0.009) | 0 / 99 / 1 / 0 | 2.45 | 2.82 | -0.036 [-0.084, +0.016] |  | -0.130 [-0.209, -0.046] | -0.134 [-0.208, -0.055] |
| the electrostatic field alone, quantum charges | 1 | +0.484 (0.145) | 0 / 57 / 27 / 16 | 1.93 | 4.12 |  |  | -0.297 [-1.096, +0.066] | -0.302 [-1.083, +0.041] |
| the five fields, quantum charges | 5 | +0.225 (0.268) | 0 / 84 / 16 / 0 | 1.81 | 5.27 | -0.463 [-1.958, +0.167] |  | -0.557 [-2.032, +0.072] | -0.561 [-2.019, +0.051] |
| the five fields with quantum charges and the two Fukui fields | 7 | +0.709 (0.131) | 0 / 89 / 11 / 0 | 1.87 | 2.86 | +0.021 [-0.278, +0.196] | +0.484 [-0.037, +1.679] | -0.073 [-0.360, +0.102] | -0.077 [-0.346, +0.086] |
| the five fields with quantum charges and 15 indicator fields | 20 | -0.293 (1.348) | 0 / 8 / 17 / 75 | 2.19 | 6.86 | -0.981 [-3.637, +0.105] | -0.518 [-1.682, -0.034] | -1.074 [-3.713, +0.013] | -1.079 [-3.721, -0.016] |

| fields | acid with the largest error (its RMSE, cm^-1; its part of the squared error) | q2 scored without the acids of SF5 (sd over the repeats) | without them, against the Hammett regressions, meta and para separately | without them, against the regression on the four charges of the carboxyl group |
|---|---|---:|---:|---:|
| the five fields, Gasteiger charges | para-N(C2H5)2 (8.1; 6 %) | +0.682 (0.014) | -0.097 [-0.159, -0.036] | -0.109 [-0.173, -0.044] |
| the electrostatic field alone, Gasteiger charges | para-N(C2H5)2 (9.4; 7 %) | +0.645 (0.009) | -0.133 [-0.215, -0.049] | -0.145 [-0.220, -0.068] |
| the electrostatic field alone, quantum charges | para-SF5 (34.7; 62 %) | +0.813 (0.011) | +0.034 [-0.014, +0.092] | +0.022 [-0.007, +0.058] |
| the five fields, quantum charges | para-SF5 (47.1; 76 %) | +0.810 (0.009) | +0.032 [-0.013, +0.085] | +0.020 [-0.019, +0.064] |
| the five fields with quantum charges and the two Fukui fields | para-SF5 (20.9; 40 %) | +0.852 (0.007) | +0.074 [+0.030, +0.135] | +0.062 [+0.016, +0.120] |
| the five fields with quantum charges and 15 indicator fields | para-SF5 (62.9; 81 %) | +0.754 (0.021) | -0.024 [-0.082, +0.033] | -0.036 [-0.074, -0.002] |

### Quantum charges, whole acids held out: every width

Here the width is fixed and the ridge is the one with the best q2 on the folds of each repeat, as in the tables of the other parts.  A row of this table shows what a width holds.  The best row of a field set is not a result, because it was chosen by its q2.

| fields | width, A | invariant local q2 (sd over the repeats) | against the five fields with Gasteiger charges | against the Hammett regressions, meta and para separately | against the regression on the four charges of the carboxyl group | posed overlap q2 (sd over the repeats) | posed overlap against the Hammett regressions, meta and para separately |
|---|---:|---:|---:|---:|---:|---:|---:|
| the five fields, Gasteiger charges | 10.00 | +0.741 (0.026) |  | -0.041 [-0.110, +0.022] | -0.045 [-0.101, +0.011] | +0.707 (0.006) | -0.074 [-0.132, -0.017] |
| the five fields, Gasteiger charges | 4.47 | +0.701 (0.016) |  | -0.081 [-0.162, -0.012] | -0.085 [-0.164, -0.016] | +0.698 (0.008) | -0.083 [-0.146, -0.020] |
| the five fields, Gasteiger charges | 2.00 | +0.227 (0.016) |  | -0.555 [-0.694, -0.449] | -0.559 [-0.683, -0.454] | +0.278 (0.065) | -0.504 [-0.597, -0.406] |
| the five fields, Gasteiger charges | 1.00 | +0.164 (0.017) |  | -0.618 [-0.935, -0.423] | -0.622 [-0.921, -0.436] | +0.197 (0.019) | -0.585 [-0.785, -0.430] |
| the electrostatic field alone, Gasteiger charges | 10.00 | +0.547 (0.011) | -0.194 [-0.337, -0.094] | -0.235 [-0.382, -0.129] | -0.239 [-0.379, -0.140] | +0.631 (0.007) | -0.151 [-0.240, -0.063] |
| the electrostatic field alone, Gasteiger charges | 4.47 | +0.604 (0.026) | -0.097 [-0.179, -0.011] | -0.177 [-0.285, -0.086] | -0.182 [-0.285, -0.087] | +0.654 (0.008) | -0.128 [-0.206, -0.044] |
| the electrostatic field alone, Gasteiger charges | 2.00 | +0.355 (0.025) | +0.128 [-0.156, +0.327] | -0.427 [-0.673, -0.263] | -0.431 [-0.678, -0.261] | +0.300 (0.077) | -0.482 [-0.702, -0.300] |
| the electrostatic field alone, Gasteiger charges | 1.00 | +0.093 (0.037) | -0.070 [-0.175, +0.081] | -0.689 [-0.870, -0.545] | -0.693 [-0.852, -0.565] | +0.227 (0.057) | -0.554 [-0.776, -0.396] |
| the electrostatic field alone, quantum charges | 10.00 | +0.612 (0.008) |  | -0.169 [-0.260, -0.096] | -0.174 [-0.259, -0.101] | +0.757 (0.011) | -0.024 [-0.101, +0.045] |
| the electrostatic field alone, quantum charges | 4.47 | +0.709 (0.005) |  | -0.072 [-0.145, -0.016] | -0.077 [-0.149, -0.011] | +0.808 (0.009) | +0.026 [-0.028, +0.090] |
| the electrostatic field alone, quantum charges | 2.00 | +0.825 (0.004) |  | +0.043 [-0.004, +0.094] | +0.039 [+0.000, +0.085] | +0.789 (0.010) | +0.007 [-0.055, +0.068] |
| the electrostatic field alone, quantum charges | 1.00 | +0.764 (0.003) |  | -0.017 [-0.069, +0.037] | -0.022 [-0.049, +0.011] | +0.798 (0.005) | +0.016 [-0.034, +0.071] |
| the five fields, quantum charges | 10.00 | +0.776 (0.008) | +0.035 [+0.002, +0.077] | -0.006 [-0.061, +0.053] | -0.010 [-0.055, +0.041] | +0.723 (0.014) | -0.058 [-0.129, +0.004] |
| the five fields, quantum charges | 4.47 | +0.814 (0.010) | +0.114 [+0.060, +0.178] | +0.033 [-0.023, +0.088] | +0.029 [-0.012, +0.072] | +0.815 (0.005) | +0.034 [-0.012, +0.089] |
| the five fields, quantum charges | 2.00 | +0.816 (0.006) | +0.589 [+0.486, +0.727] | +0.034 [-0.020, +0.094] | +0.030 [-0.010, +0.073] | +0.783 (0.013) | +0.001 [-0.073, +0.066] |
| the five fields, quantum charges | 1.00 | +0.793 (0.008) | +0.630 [+0.440, +0.948] | +0.012 [-0.038, +0.066] | +0.007 [-0.015, +0.035] | +0.758 (0.008) | -0.024 [-0.079, +0.030] |
| the five fields with quantum charges and the two Fukui fields | 10.00 | +0.829 (0.012) | +0.089 [+0.047, +0.141] | +0.048 [-0.003, +0.102] | +0.043 [-0.003, +0.094] | +0.774 (0.009) | -0.007 [-0.071, +0.053] |
| the five fields with quantum charges and the two Fukui fields | 4.47 | +0.853 (0.007) | +0.152 [+0.097, +0.226] | +0.071 [+0.023, +0.127] | +0.067 [+0.020, +0.120] | +0.856 (0.003) | +0.075 [+0.027, +0.136] |
| the five fields with quantum charges and the two Fukui fields | 2.00 | +0.862 (0.011) | +0.635 [+0.541, +0.762] | +0.080 [+0.025, +0.146] | +0.076 [+0.028, +0.132] | +0.824 (0.009) | +0.042 [-0.008, +0.101] |
| the five fields with quantum charges and the two Fukui fields | 1.00 | +0.854 (0.010) | +0.690 [+0.500, +1.007] | +0.072 [+0.020, +0.133] | +0.068 [+0.029, +0.116] | +0.801 (0.010) | +0.019 [-0.045, +0.083] |
| the five fields with quantum charges and 15 indicator fields | 10.00 | +0.791 (0.019) | +0.050 [-0.022, +0.128] | +0.009 [-0.066, +0.084] | +0.005 [-0.071, +0.080] | +0.722 (0.024) | -0.060 [-0.172, +0.035] |
| the five fields with quantum charges and 15 indicator fields | 4.47 | +0.774 (0.017) | +0.074 [-0.029, +0.177] | -0.007 [-0.115, +0.079] | -0.011 [-0.118, +0.080] | +0.731 (0.026) | -0.051 [-0.162, +0.037] |
| the five fields with quantum charges and 15 indicator fields | 2.00 | +0.814 (0.021) | +0.587 [+0.480, +0.727] | +0.032 [-0.027, +0.091] | +0.028 [-0.026, +0.082] | +0.758 (0.032) | -0.024 [-0.095, +0.035] |
| the five fields with quantum charges and 15 indicator fields | 1.00 | +0.773 (0.004) | +0.610 [+0.419, +0.926] | -0.009 [-0.059, +0.045] | -0.013 [-0.039, +0.017] | +0.760 (0.007) | -0.022 [-0.078, +0.032] |

### Quantum charges, whole substituents held out (both acids of a substituent together): the references

| reference | q2 (sd over the repeats) | RMSE, cm^-1 | RMSE of the meta acids, cm^-1 | RMSE of the para acids, cm^-1 | acid with the largest error (its RMSE, cm^-1; its part of the squared error) | q2 scored without the acids of SF5 (sd over the repeats) |
|---|---:|---:|---:|---:|---|---:|
| atom-centred counts | +0.623 (0.009) | 2.74 | 2.39 | 3.06 | para-N(CH3)2 (9.9; 7 %) | +0.626 (0.008) |
| linear regression on the Hammett constant, all acids | +0.742 (0.002) | 2.27 | 2.30 | 2.23 | para-N(C2H5)2 (6.3; 4 %) | +0.738 (0.002) |
| linear regression on the Hammett constant, meta and para separately | +0.782 (0.002) | 2.09 | 2.14 | 2.03 | meta-N(CH3)COCH3 (6.4; 5 %) | +0.778 (0.002) |
| linear regression on the quantum charge of the carbonyl oxygen | +0.760 (0.002) | 2.19 | 1.96 | 2.39 | para-NHCH3 (6.4; 4 %) | +0.754 (0.002) |
| multiple linear regression on the quantum charges of the four atoms of the carboxyl group | +0.783 (0.011) | 2.08 | 1.88 | 2.25 | para-SeCH3 (7.9; 8 %) | +0.787 (0.010) |

### Quantum charges, whole substituents held out (both acids of a substituent together): the width and the ridge chosen inside the folds, invariant local kernel

Every training part is divided again into 5 folds over the same unit.  The width of the envelope (one of 10.00, 4.47, 2.00, 1.00 A) and the ridge are the pair with the best q2 on these inner folds, so the held-out acids take no part in the choice.  These are the rows that the reading below uses.

| fields | number of fields | q2 (sd over the repeats) | number of folds that chose the width 10.00 / 4.47 / 2.00 / 1.00 A | RMSE of the meta acids, cm^-1 | RMSE of the para acids, cm^-1 | against the five fields with Gasteiger charges | against the five fields with quantum charges | against the Hammett regressions, meta and para separately | against the regression on the four charges of the carboxyl group |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| the five fields, Gasteiger charges | 5 | +0.640 (0.034) | 85 / 15 / 0 / 0 | 2.43 | 2.92 |  |  | -0.141 [-0.278, -0.043] | -0.143 [-0.274, -0.052] |
| the electrostatic field alone, Gasteiger charges | 1 | +0.130 (0.127) | 9 / 91 / 0 / 0 | 4.28 | 4.07 | -0.510 [-1.275, -0.046] |  | -0.652 [-1.447, -0.173] | -0.653 [-1.439, -0.180] |
| the electrostatic field alone, quantum charges | 1 | +0.799 (0.032) | 0 / 6 / 86 / 8 | 2.16 | 1.84 |  |  | +0.017 [-0.038, +0.074] | +0.016 [-0.014, +0.056] |
| the five fields, quantum charges | 5 | +0.361 (0.324) | 0 / 11 / 88 / 1 | 4.27 | 2.72 | -0.279 [-1.418, +0.276] |  | -0.420 [-1.577, +0.089] | -0.422 [-1.566, +0.065] |
| the five fields with quantum charges and the two Fukui fields | 7 | +0.823 (0.023) | 0 / 19 / 80 / 1 | 1.91 | 1.86 | +0.183 [+0.090, +0.316] | +0.462 [-0.042, +1.604] | +0.042 [-0.050, +0.115] | +0.040 [-0.013, +0.088] |
| the five fields with quantum charges and 15 indicator fields | 20 | +0.611 (0.171) | 0 / 0 / 79 / 21 | 3.31 | 2.13 | -0.029 [-0.469, +0.248] | +0.250 [-0.072, +0.960] | -0.170 [-0.622, +0.054] | -0.172 [-0.609, +0.040] |

Where the error of these rows is.  The last three columns score the same predictions without the acids of the substituent SF5; nothing is fitted again.

| fields | acid with the largest error (its RMSE, cm^-1; its part of the squared error) | q2 scored without the acids of SF5 (sd over the repeats) | without them, against the Hammett regressions, meta and para separately | without them, against the regression on the four charges of the carboxyl group |
|---|---|---:|---:|---:|
| the five fields, Gasteiger charges | meta-B(OH)2 (10.2; 8 %) | +0.642 (0.036) | -0.136 [-0.275, -0.033] | -0.144 [-0.277, -0.050] |
| the electrostatic field alone, Gasteiger charges | meta-C(CF3)3 (28.7; 25 %) | +0.119 (0.132) | -0.659 [-1.496, -0.173] | -0.668 [-1.513, -0.181] |
| the electrostatic field alone, quantum charges | meta-SF5 (6.9; 6 %) | +0.807 (0.015) | +0.029 [-0.020, +0.083] | +0.021 [-0.011, +0.061] |
| the five fields, quantum charges | meta-SF5 (37.7; 59 %) | +0.825 (0.010) | +0.046 [-0.002, +0.104] | +0.038 [+0.008, +0.074] |
| the five fields with quantum charges and the two Fukui fields | meta-SeCH3 (9.7; 14 %) | +0.830 (0.029) | +0.052 [-0.034, +0.127] | +0.043 [-0.009, +0.097] |
| the five fields with quantum charges and 15 indicator fields | meta-SF5 (24.6; 41 %) | +0.791 (0.019) | +0.013 [-0.046, +0.067] | +0.004 [-0.045, +0.053] |

### Quantum charges, whole substituents held out (both acids of a substituent together): the width and the ridge chosen inside the folds, posed overlap kernel

| fields | number of fields | q2 (sd over the repeats) | number of folds that chose the width 10.00 / 4.47 / 2.00 / 1.00 A | RMSE of the meta acids, cm^-1 | RMSE of the para acids, cm^-1 | against the five fields with Gasteiger charges | against the five fields with quantum charges | against the Hammett regressions, meta and para separately | against the regression on the four charges of the carboxyl group |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| the five fields, Gasteiger charges | 5 | +0.682 (0.027) | 72 / 28 / 0 / 0 | 2.42 | 2.63 |  |  | -0.100 [-0.173, -0.033] | -0.102 [-0.166, -0.035] |
| the electrostatic field alone, Gasteiger charges | 1 | -0.573 (1.990) | 0 / 96 / 4 / 0 | 3.91 | 6.92 | -1.255 [-4.283, +0.002] |  | -1.355 [-4.369, -0.070] | -1.356 [-4.346, -0.087] |
| the electrostatic field alone, quantum charges | 1 | +0.421 (0.403) | 0 / 59 / 21 / 20 | 1.81 | 4.47 |  |  | -0.361 [-1.355, +0.089] | -0.362 [-1.344, +0.068] |
| the five fields, quantum charges | 5 | +0.224 (0.527) | 0 / 86 / 14 / 0 | 1.79 | 5.28 | -0.457 [-1.959, +0.177] |  | -0.557 [-2.023, +0.078] | -0.559 [-2.023, +0.066] |
| the five fields with quantum charges and the two Fukui fields | 7 | +0.730 (0.054) | 0 / 87 / 12 / 1 | 1.75 | 2.79 | +0.048 [-0.274, +0.225] | +0.505 [+0.023, +1.678] | -0.052 [-0.362, +0.123] | -0.053 [-0.358, +0.122] |
| the five fields with quantum charges and 15 indicator fields | 20 | -1.088 (1.451) | 6 / 1 / 23 / 70 | 2.46 | 8.82 | -1.769 [-6.398, +0.123] | -1.312 [-4.447, -0.031] | -1.869 [-6.461, +0.024] | -1.871 [-6.460, +0.005] |

| fields | acid with the largest error (its RMSE, cm^-1; its part of the squared error) | q2 scored without the acids of SF5 (sd over the repeats) | without them, against the Hammett regressions, meta and para separately | without them, against the regression on the four charges of the carboxyl group |
|---|---|---:|---:|---:|
| the five fields, Gasteiger charges | para-N(C2H5)2 (7.9; 5 %) | +0.675 (0.027) | -0.104 [-0.175, -0.038] | -0.112 [-0.179, -0.046] |
| the electrostatic field alone, Gasteiger charges | para-SF5 (61.1; 63 %) | +0.639 (0.010) | -0.139 [-0.231, -0.053] | -0.148 [-0.228, -0.069] |
| the electrostatic field alone, quantum charges | para-SF5 (38.6; 68 %) | +0.822 (0.011) | +0.044 [-0.001, +0.102] | +0.035 [+0.003, +0.078] |
| the five fields, quantum charges | para-SF5 (47.3; 76 %) | +0.816 (0.009) | +0.037 [-0.010, +0.089] | +0.029 [-0.011, +0.078] |
| the five fields with quantum charges and the two Fukui fields | para-SF5 (20.5; 41 %) | +0.857 (0.014) | +0.078 [+0.033, +0.136] | +0.070 [+0.020, +0.133] |
| the five fields with quantum charges and 15 indicator fields | para-SF5 (82.8; 87 %) | +0.756 (0.033) | -0.022 [-0.094, +0.038] | -0.031 [-0.091, +0.012] |

### Quantum charges, whole substituents held out (both acids of a substituent together): every width

Here the width is fixed and the ridge is the one with the best q2 on the folds of each repeat, as in the tables of the other parts.  A row of this table shows what a width holds.  The best row of a field set is not a result, because it was chosen by its q2.

| fields | width, A | invariant local q2 (sd over the repeats) | against the five fields with Gasteiger charges | against the Hammett regressions, meta and para separately | against the regression on the four charges of the carboxyl group | posed overlap q2 (sd over the repeats) | posed overlap against the Hammett regressions, meta and para separately |
|---|---:|---:|---:|---:|---:|---:|---:|
| the five fields, Gasteiger charges | 10.00 | +0.676 (0.015) |  | -0.106 [-0.243, -0.011] | -0.107 [-0.235, -0.019] | +0.703 (0.011) | -0.078 [-0.141, -0.016] |
| the five fields, Gasteiger charges | 4.47 | +0.648 (0.017) |  | -0.133 [-0.238, -0.049] | -0.135 [-0.233, -0.053] | +0.695 (0.012) | -0.086 [-0.161, -0.022] |
| the five fields, Gasteiger charges | 2.00 | +0.210 (0.043) |  | -0.572 [-0.830, -0.407] | -0.573 [-0.818, -0.416] | +0.114 (0.030) | -0.668 [-0.774, -0.561] |
| the five fields, Gasteiger charges | 1.00 | +0.142 (0.065) |  | -0.639 [-0.878, -0.483] | -0.641 [-0.869, -0.484] | +0.182 (0.020) | -0.599 [-0.781, -0.462] |
| the electrostatic field alone, Gasteiger charges | 10.00 | +0.323 (0.029) | -0.353 [-0.740, -0.127] | -0.458 [-0.857, -0.229] | -0.460 [-0.847, -0.238] | +0.627 (0.007) | -0.154 [-0.245, -0.065] |
| the electrostatic field alone, Gasteiger charges | 4.47 | +0.517 (0.028) | -0.131 [-0.272, -0.014] | -0.265 [-0.449, -0.129] | -0.266 [-0.437, -0.135] | +0.648 (0.008) | -0.133 [-0.223, -0.048] |
| the electrostatic field alone, Gasteiger charges | 2.00 | +0.362 (0.016) | +0.152 [-0.134, +0.461] | -0.420 [-0.672, -0.266] | -0.422 [-0.665, -0.262] | +0.134 (0.020) | -0.647 [-0.836, -0.501] |
| the electrostatic field alone, Gasteiger charges | 1.00 | +0.022 (0.021) | -0.120 [-0.240, +0.050] | -0.760 [-0.853, -0.672] | -0.761 [-0.846, -0.672] | +0.100 (0.019) | -0.682 [-0.987, -0.494] |
| the electrostatic field alone, quantum charges | 10.00 | +0.573 (0.021) |  | -0.209 [-0.353, -0.104] | -0.211 [-0.334, -0.125] | +0.771 (0.016) | -0.010 [-0.088, +0.059] |
| the electrostatic field alone, quantum charges | 4.47 | +0.713 (0.008) |  | -0.068 [-0.144, -0.007] | -0.070 [-0.141, -0.009] | +0.817 (0.008) | +0.036 [-0.012, +0.099] |
| the electrostatic field alone, quantum charges | 2.00 | +0.826 (0.003) |  | +0.045 [-0.003, +0.099] | +0.043 [+0.006, +0.091] | +0.800 (0.007) | +0.019 [-0.041, +0.082] |
| the electrostatic field alone, quantum charges | 1.00 | +0.766 (0.003) |  | -0.016 [-0.068, +0.043] | -0.017 [-0.045, +0.018] | +0.804 (0.002) | +0.022 [-0.027, +0.080] |
| the five fields, quantum charges | 10.00 | +0.745 (0.006) | +0.070 [+0.013, +0.152] | -0.036 [-0.112, +0.033] | -0.038 [-0.109, +0.028] | +0.731 (0.013) | -0.050 [-0.133, +0.019] |
| the five fields, quantum charges | 4.47 | +0.796 (0.005) | +0.148 [+0.075, +0.249] | +0.014 [-0.033, +0.074] | +0.012 [-0.029, +0.063] | +0.823 (0.004) | +0.041 [-0.002, +0.093] |
| the five fields, quantum charges | 2.00 | +0.815 (0.006) | +0.605 [+0.464, +0.831] | +0.034 [-0.026, +0.095] | +0.032 [-0.001, +0.075] | +0.793 (0.010) | +0.012 [-0.059, +0.078] |
| the five fields, quantum charges | 1.00 | +0.792 (0.007) | +0.649 [+0.491, +0.891] | +0.010 [-0.042, +0.068] | +0.008 [-0.016, +0.043] | +0.764 (0.005) | -0.017 [-0.071, +0.041] |
| the five fields with quantum charges and the two Fukui fields | 10.00 | +0.787 (0.006) | +0.111 [+0.045, +0.211] | +0.005 [-0.058, +0.066] | +0.004 [-0.059, +0.070] | +0.783 (0.013) | +0.002 [-0.062, +0.064] |
| the five fields with quantum charges and the two Fukui fields | 4.47 | +0.836 (0.009) | +0.188 [+0.113, +0.288] | +0.054 [-0.001, +0.116] | +0.053 [+0.004, +0.112] | +0.861 (0.006) | +0.080 [+0.036, +0.135] |
| the five fields with quantum charges and the two Fukui fields | 2.00 | +0.859 (0.004) | +0.649 [+0.499, +0.905] | +0.078 [+0.028, +0.131] | +0.076 [+0.042, +0.121] | +0.834 (0.008) | +0.052 [-0.003, +0.112] |
| the five fields with quantum charges and the two Fukui fields | 1.00 | +0.830 (0.011) | +0.688 [+0.537, +0.931] | +0.048 [-0.004, +0.101] | +0.046 [+0.015, +0.084] | +0.808 (0.014) | +0.026 [-0.034, +0.085] |
| the five fields with quantum charges and 15 indicator fields | 10.00 | +0.724 (0.013) | +0.048 [-0.106, +0.202] | -0.058 [-0.205, +0.052] | -0.059 [-0.203, +0.052] | +0.707 (0.016) | -0.074 [-0.210, +0.033] |
| the five fields with quantum charges and 15 indicator fields | 4.47 | +0.736 (0.020) | +0.088 [-0.017, +0.194] | -0.045 [-0.162, +0.045] | -0.047 [-0.157, +0.042] | +0.727 (0.013) | -0.054 [-0.177, +0.044] |
| the five fields with quantum charges and 15 indicator fields | 2.00 | +0.812 (0.013) | +0.602 [+0.437, +0.867] | +0.030 [-0.026, +0.082] | +0.029 [-0.027, +0.083] | +0.768 (0.024) | -0.013 [-0.090, +0.050] |
| the five fields with quantum charges and 15 indicator fields | 1.00 | +0.769 (0.012) | +0.627 [+0.472, +0.858] | -0.013 [-0.069, +0.044] | -0.014 [-0.039, +0.017] | +0.768 (0.012) | -0.014 [-0.071, +0.045] |

### The difference between the para acid and the meta acid of one substituent

Resonance is the part of a substituent effect that differs between the para and the meta position.  Over the 94 substituents that occur twice, the band of the para acid minus the band of the meta acid has a mean of -2.6 cm^-1 and a standard deviation of 2.9 cm^-1, and it correlates with the difference of the two Hammett constants with r +0.665.  The table gives how well each description predicts this difference from its out-of-fold predictions of the two acids (invariant local kernel, width and ridge chosen inside the folds).  A q2 below zero means that the description predicts the difference worse than its mean does.

| description | whole acids held out: q2 of the predicted difference | whole acids held out: RMSE of the difference, cm^-1 | whole substituents held out: q2 of the predicted difference | whole substituents held out: RMSE of the difference, cm^-1 |
|---|---:|---:|---:|---:|
| atom-centred counts | -0.27 | 3.30 | -0.17 | 3.16 |
| linear regression on the Hammett constant, all acids | +0.02 | 2.89 | +0.05 | 2.85 |
| linear regression on the Hammett constant, meta and para separately | +0.42 | 2.23 | +0.42 | 2.23 |
| linear regression on the quantum charge of the carbonyl oxygen | -0.09 | 3.06 | -0.06 | 3.01 |
| multiple linear regression on the quantum charges of the four atoms of the carboxyl group | +0.22 | 2.58 | +0.27 | 2.50 |
| the five fields, Gasteiger charges | -0.30 | 3.33 | +0.18 | 2.65 |
| the electrostatic field alone, Gasteiger charges | -0.41 | 3.47 | -0.10 | 3.06 |
| the electrostatic field alone, quantum charges | +0.41 | 2.25 | +0.44 | 2.18 |
| the five fields, quantum charges | +0.08 | 2.81 | +0.14 | 2.72 |
| the five fields with quantum charges and the two Fukui fields | +0.42 | 2.22 | +0.60 | 1.84 |
| the five fields with quantum charges and 15 indicator fields | +0.01 | 2.91 | -0.49 | 3.57 |

### How alike the kernel with quantum charges and the kernel with Gasteiger charges are

The correlation between the off-diagonal elements of the two kernels, each centred first.  Near 1, the change of the charges does not change which acids the description finds alike.

| width, A | electrostatic field alone, invariant local kernel | electrostatic field alone, posed overlap kernel | five fields, invariant local kernel | five fields, posed overlap kernel |
|---:|---:|---:|---:|---:|
| 10.00 | +0.697 | +0.466 | +0.932 | +0.874 |
| 4.47 | +0.493 | +0.490 | +0.897 | +0.895 |
| 2.00 | +0.920 | +0.420 | +0.817 | +0.810 |
| 1.00 | +0.657 | +0.259 | +1.000 | +1.000 |

**Reading of the quantum charges.** The Gasteiger charges of the four atoms of the carboxyl group are the same in every acid (the largest standard deviation is 0.0001 e), so the default electrostatic field holds a substituent only where the substituent is. The quantum charge of the carbonyl oxygen varies (standard deviation 0.0083 e) and correlates with the band with r +0.875 and with the Hammett constant with r +0.909.

This changes what the width of the envelope does. With Gasteiger charges the electrostatic field alone reaches q2 +0.355 at 2.00 A and +0.547 at 10.00 A (whole acids held out, ridge chosen by the q2), and its best width is 4.47 A: the envelope has to reach the substituent. With quantum charges it reaches +0.825 at 2.00 A and +0.612 at 10.00 A. The regression on the four charges of the carboxyl group reaches +0.786, and the difference of the field at 2.00 A from it is +0.039 [+0.000, +0.085]. The quantum calculation has carried the effect of the substituent to the atoms of the carboxyl group, and a narrow field reads it there. The width therefore decides the result. It is chosen inside the folds, and the statements below are made for those rows and for the invariant local kernel.

With whole acids held out, the Hammett regressions for meta and para separately reach q2 +0.782 and the regression on the four charges of the carboxyl group reaches +0.786. The electrostatic field alone reaches +0.562 with Gasteiger charges and +0.824 with quantum charges (difference +0.262 [+0.166, +0.396]). With quantum charges it is +0.042 [-0.005, +0.093] against the separate Hammett regressions (not separated from them) and +0.038 [-0.001, +0.084] against the regression on the four charges (not separated from it). The five fields reach +0.718 with Gasteiger charges and +0.651 with quantum charges, with a standard deviation over the repeats of 0.238 (see the paragraph on one substituent below). With the two Fukui fields added the q2 is +0.837: +0.120 [+0.063, +0.184] against the five fields with Gasteiger charges (above them), +0.056 [-0.006, +0.120] against the separate Hammett regressions (not separated from them) and +0.052 [-0.003, +0.108] against the regression on the four charges (not separated from it). With the indicator fields added the q2 is +0.692 (+0.040 [-0.082, +0.233] against the five fields with quantum charges, not separated from them).

With whole substituents held out, the Hammett regressions for meta and para separately reach q2 +0.782 and the regression on the four charges of the carboxyl group reaches +0.783. The electrostatic field alone reaches +0.130 with Gasteiger charges and +0.799 with quantum charges (difference +0.669 [+0.203, +1.461]). With quantum charges it is +0.017 [-0.038, +0.074] against the separate Hammett regressions (not separated from them) and +0.016 [-0.014, +0.056] against the regression on the four charges (not separated from it). The five fields reach +0.640 with Gasteiger charges and +0.361 with quantum charges, with a standard deviation over the repeats of 0.324 (see the paragraph on one substituent below). With the two Fukui fields added the q2 is +0.823: +0.183 [+0.090, +0.316] against the five fields with Gasteiger charges (above them), +0.042 [-0.050, +0.115] against the separate Hammett regressions (not separated from them) and +0.040 [-0.013, +0.088] against the regression on the four charges (not separated from it). With the indicator fields added the q2 is +0.611 (+0.250 [-0.072, +0.960] against the five fields with quantum charges, not separated from them).

**One substituent decides the result of the five fields with quantum charges.** The acids of the substituent SF5 (para-SF5, meta-SF5) hold 73 % of the squared error of that row with whole substituents held out. The acid with the largest error is meta-SF5, with a root mean square error of 37.7 cm^-1 over the repeats. The largest quantum charge of an atom in these acids is +2.05 e. The largest in all other acids is +1.40 e (meta-SO2F). When these acids are held out together, the training part holds no acid like them. The inner folds then choose a narrow width with a small ridge, and that model extrapolates far. Scored without these acids, the same predictions give q2 +0.815 with whole acids held out and +0.825 with whole substituents held out (+0.036 [-0.014, +0.091] and +0.046 [-0.002, +0.104] against the separate Hammett regressions scored in the same way). The set with the two Fukui fields, scored in the same way, gives +0.075 [+0.016, +0.134] and +0.052 [-0.034, +0.127]. With the posed overlap kernel, an acid of this substituent has the largest error in 9 of the 12 rows of the two tables. The substituent was found by its error, after the result was known, so the scores without it describe the failure and are not results. The tables with a fixed width do not show this failure, because there the ridge is chosen by the q2 of the folds that hold these acids out. A field from quantum charges therefore needs a check that the charges of a new molecule are inside the range of the training set. A tabulated Hammett constant needs no such check.

**Resonance.** The difference between the para acid and the meta acid of one substituent is the part of the band that resonance decides. The five fields with Gasteiger charges predict it with q2 -0.30 (whole acids held out) and +0.18 (whole substituents held out). The electrostatic field alone with quantum charges reaches +0.41 and +0.44, and the five fields with quantum charges and the two Fukui fields reach +0.42 and +0.60. The separate Hammett regressions reach +0.42 and +0.42, and the regression on the four charges of the carboxyl group +0.22 and +0.27.

**Answer.** With the width and the ridge chosen inside the folds, no field set with quantum charges passes the Hammett regressions for meta and para separately on both kinds of folds. The electrostatic field alone with quantum charges is +0.042 [-0.005, +0.093] against them with whole acids held out (not separated from them) and +0.017 [-0.038, +0.074] with whole substituents held out (not separated from them). The five fields with quantum charges and the two Fukui fields are +0.056 [-0.006, +0.120] against them with whole acids held out (not separated from them) and +0.042 [-0.050, +0.115] with whole substituents held out (not separated from them). The five fields with quantum charges alone are -0.130 [-0.487, +0.064] against them with whole acids held out (not separated from them) and -0.420 [-1.577, +0.089] with whole substituents held out (not separated from them). The interval of the last set is wide because of the substituent SF5. In the table with a fixed width, the best width of the set with the Fukui fields (2.00 A with whole acids held out, 2.00 A with whole substituents held out) gives +0.080 [+0.025, +0.146] and +0.078 [+0.028, +0.131] against the separate Hammett regressions. The difference between these numbers and the numbers above is what a choice of the width and the ridge by the result adds. Quantum charges do what Gasteiger charges cannot do on this series: they bring the fields to the level of the separate Hammett regressions, and they predict the difference between the para and the meta acid. They do not bring the q2 of the fields above the regression on the four charges of the carboxyl group: every interval of that difference holds zero. So most of the gain is in the charges of the carboxyl group themselves. On the difference between the para and the meta acid the two best field sets reach a higher q2 than this regression (see the paragraph on resonance); no interval was computed for that difference. The quantum charges come from a quantum calculation, as the band does. This part therefore measures how much of a costly quantum result a quantum calculation of a few seconds per acid already holds. It does not measure what the structure alone predicts.
