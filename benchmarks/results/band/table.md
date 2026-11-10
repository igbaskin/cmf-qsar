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
