# Crown ethers and cryptands in the frame of the cavity (plan item SC5)

79 hosts in 9 families: 12-crown-4 4, 15-crown-5 17, benzo-15-crown-5 6, aza-15-crown-5 8, 18-crown-6 12, benzo-18-crown-6 6, aza-18-crown-6 8, larger ring 9, cryptand 9.

| Target | Lowest | Highest | Mean | Standard deviation |
|---|---|---|---|---|
| log K of K+ | 0.90 | 10.61 | 4.18 | 1.85 |
| log K of Na+ | 1.10 | 9.47 | 3.59 | 1.53 |
| log K(K+) - log K(Na+) | -4.10 | 2.80 | 0.59 | 1.09 |

The correlation of the two constants is +0.810.

## The geometries of the complexes

A geometry is the host wrapped around the ion: every donor of the ring was held at the sum of the ionic radii from the ion while the complex was optimised. The misfit is the root-mean-square difference between the ion-donor distances and that sum.

| Ion | Sum of the ionic radii (A) | Hosts with every ring donor in contact | Mean misfit (A) | Largest misfit (A) | Mean distance of the ion from the donor centroid (A) |
|---|---|---|---|---|---|
| Na+ | 2.42 | 79 of 79 | 0.01 | 0.15 | 0.48 |
| K+ | 2.78 | 79 of 79 | 0.01 | 0.07 | 0.88 |

| Family | Radius of the donor ring with Na+ (A) | With K+ (A) | Distance of Na+ from the donor centroid (A) | Of K+ (A) |
|---|---|---|---|---|
| 12-crown-4 | 1.96 | 1.99 | 1.40 | 1.92 |
| 15-crown-5 | 2.30 | 2.40 | 0.74 | 1.37 |
| benzo-15-crown-5 | 2.27 | 2.36 | 0.83 | 1.45 |
| aza-15-crown-5 | 2.32 | 2.44 | 0.64 | 1.31 |
| 18-crown-6 | 2.40 | 2.70 | 0.27 | 0.56 |
| benzo-18-crown-6 | 2.41 | 2.67 | 0.23 | 0.69 |
| aza-18-crown-6 | 2.41 | 2.74 | 0.27 | 0.40 |
| larger ring | 2.44 | 2.77 | 0.14 | 0.20 |
| cryptand | 2.44 | 2.72 | 0.18 | 0.43 |

| Number read from the complex | Correlation with log K of Na+ | With log K of K+ |
|---|---|---|
| ring donors in contact | +0.24 | +0.57 |
| misfit of the donor distances | +0.09 | +0.35 |
| radius of the donor ring | +0.35 | +0.64 |
| ion to donor centroid | -0.35 | -0.61 |

| Number read from the formula | Correlation with log K of K+ | With log K of Na+ | With the selectivity |
|---|---|---|---|
| ring size | +0.35 | -0.07 | +0.69 |
| ring donors | +0.57 | +0.24 | +0.64 |
| ring nitrogens | +0.62 | +0.61 | +0.19 |
| ring sulfurs | -0.03 | -0.08 | +0.05 |
| fused aromatic rings | -0.09 | -0.18 | +0.10 |
| side-arm donors | -0.06 | +0.10 | -0.26 |
| bicyclic | +0.60 | +0.60 | +0.18 |

## Where the probe field puts the ion

| Ion | Deepest point of the probe field: mean distance from the ion (A) | Within 1 A | Donor centroid: mean distance from the ion (A) | Within 1 A | Probe closer than the centroid |
|---|---|---|---|---|---|
| Na+ | 0.34 | 100 % | 0.48 | 95 % | 63 % |
| K+ | 0.47 | 92 % | 0.88 | 49 % | 68 % |

## Cross-validated q2, five random folds, ten repeats

| Description | log K of K+ | log K of Na+ | log K(K+) - log K(Na+) |
|---|---|---|---|
| ring counts | +0.718 ± 0.070 (1.00) | +0.483 ± 0.259 (1.11) | +0.410 ± 0.109 (0.85) |
| complex shape | +0.166 ± 0.853 (1.71) | -0.466 ± 0.954 (1.87) | +0.472 ± 0.050 (0.80) |
| ring counts + complex shape | +0.589 ± 0.085 (1.20) | +0.516 ± 0.116 (1.07) | +0.378 ± 0.126 (0.87) |
| fingerprint | +0.649 ± 0.045 (1.11) | +0.707 ± 0.056 (0.84) | +0.295 ± 0.148 (0.93) |
| electrostatic field | +0.420 ± 0.037 (1.43) | +0.229 ± 0.066 (1.36) | +0.386 ± 0.058 (0.86) |
| ion probe field | +0.214 ± 0.066 (1.66) | -0.041 ± 0.056 (1.58) | +0.157 ± 0.115 (1.01) |
| five fields | +0.423 ± 0.040 (1.42) | +0.256 ± 0.064 (1.33) | +0.349 ± 0.074 (0.89) |
| five fields + probe | +0.418 ± 0.039 (1.43) | +0.243 ± 0.069 (1.34) | +0.346 ± 0.076 (0.89) |
| five fields, ion side | +0.420 ± 0.048 (1.43) | +0.310 ± 0.054 (1.28) | +0.305 ± 0.103 (0.92) |
| five fields, no frame | +0.449 ± 0.055 (1.39) | +0.046 ± 0.118 (1.51) | +0.308 ± 0.094 (0.92) |
| five fields, amounts kept | +0.399 ± 0.046 (1.45) | +0.270 ± 0.081 (1.32) | +0.361 ± 0.061 (0.88) |
| five fields + ring counts | +0.722 ± 0.066 (0.99) | +0.566 ± 0.096 (1.02) | +0.418 ± 0.101 (0.84) |

In brackets: the root-mean-square error in log K units.

### Paired differences of the mean absolute error, five random folds, ten repeats

A negative value means the first description has the smaller error. The interval is a 95 % bootstrap interval over the hosts.

| First description, second description | log K of K+ | log K of Na+ | log K(K+) - log K(Na+) |
|---|---|---|---|
| five fields, ring counts | +0.369 [+0.191, +0.550] | +0.307 [+0.115, +0.499] | +0.089 [+0.008, +0.168] |
| five fields, complex shape | -0.015 [-0.192, +0.156] (includes zero) | -0.130 [-0.346, +0.064] (includes zero) | +0.102 [+0.037, +0.168] |
| five fields, ring counts + complex shape | +0.266 [+0.110, +0.430] | +0.244 [+0.056, +0.425] | +0.079 [+0.010, +0.144] |
| five fields, fingerprint | +0.272 [+0.097, +0.450] | +0.388 [+0.213, +0.559] | +0.035 [-0.042, +0.111] (includes zero) |
| five fields, five fields, no frame | -0.102 [-0.243, +0.044] (includes zero) | -0.139 [-0.286, +0.000] (includes zero) | -0.020 [-0.102, +0.059] (includes zero) |
| five fields, five fields, ion side | -0.036 [-0.142, +0.069] (includes zero) | -0.005 [-0.110, +0.108] (includes zero) | +0.003 [-0.045, +0.053] (includes zero) |
| five fields, five fields, amounts kept | -0.025 [-0.070, +0.016] (includes zero) | +0.013 [-0.026, +0.053] (includes zero) | +0.002 [-0.015, +0.020] (includes zero) |
| five fields, amounts kept, ring counts | +0.394 [+0.208, +0.583] | +0.294 [+0.110, +0.479] | +0.087 [+0.005, +0.168] |
| five fields, electrostatic field | -0.048 [-0.121, +0.025] (includes zero) | -0.021 [-0.077, +0.036] (includes zero) | -0.006 [-0.039, +0.028] (includes zero) |
| five fields, ion probe field | -0.306 [-0.423, -0.190] | -0.168 [-0.281, -0.049] | -0.143 [-0.209, -0.080] |
| ion probe field, electrostatic field | +0.259 [+0.135, +0.384] | +0.148 [+0.021, +0.268] | +0.137 [+0.075, +0.200] |
| five fields + probe, five fields | +0.010 [-0.007, +0.027] (includes zero) | +0.008 [-0.012, +0.031] (includes zero) | +0.003 [-0.010, +0.015] (includes zero) |
| five fields + ring counts, ring counts | +0.038 [-0.011, +0.089] (includes zero) | +0.031 [-0.050, +0.101] (includes zero) | +0.019 [-0.023, +0.057] (includes zero) |
| five fields + ring counts, five fields | -0.331 [-0.497, -0.169] | -0.276 [-0.451, -0.100] | -0.070 [-0.140, -0.004] |

## Cross-validated q2, one family of hosts held out at a time

| Description | log K of K+ | log K of Na+ | log K(K+) - log K(Na+) |
|---|---|---|---|
| ring counts | +0.176 (1.82) | +0.008 (1.63) | -0.111 (1.23) |
| complex shape | -0.187 (2.19) | -1.248 (2.45) | +0.332 (0.95) |
| ring counts + complex shape | +0.547 (1.35) | -0.045 (1.67) | -0.118 (1.24) |
| fingerprint | +0.276 (1.71) | +0.023 (1.61) | +0.210 (1.04) |
| electrostatic field | +0.163 (1.84) | -0.104 (1.72) | +0.375 (0.92) |
| ion probe field | -0.002 (2.01) | -0.533 (2.02) | +0.164 (1.07) |
| five fields | +0.158 (1.84) | -0.201 (1.79) | +0.320 (0.96) |
| five fields + probe | +0.184 (1.81) | -0.232 (1.81) | +0.287 (0.99) |
| five fields, ion side | +0.292 (1.69) | -0.378 (1.92) | +0.323 (0.96) |
| five fields, no frame | +0.291 (1.69) | -0.294 (1.86) | +0.218 (1.03) |
| five fields, amounts kept | +0.189 (1.81) | -0.153 (1.75) | +0.337 (0.95) |
| five fields + ring counts | +0.097 (1.91) | -0.085 (1.70) | -0.006 (1.17) |

In brackets: the root-mean-square error in log K units.

### Paired differences of the mean absolute error, one family of hosts held out at a time

A negative value means the first description has the smaller error. The interval is a 95 % bootstrap interval over the hosts.

| First description, second description | log K of K+ | log K of Na+ | log K(K+) - log K(Na+) |
|---|---|---|---|
| five fields, ring counts | +0.015 [-0.150, +0.182] (includes zero) | +0.149 [-0.023, +0.325] (includes zero) | -0.191 [-0.335, -0.055] |
| five fields, complex shape | -0.355 [-0.564, -0.166] | -0.411 [-0.692, -0.162] | +0.083 [-0.001, +0.164] (includes zero) |
| five fields, ring counts + complex shape | +0.274 [+0.040, +0.512] | +0.139 [-0.037, +0.326] (includes zero) | -0.119 [-0.313, +0.061] (includes zero) |
| five fields, fingerprint | +0.000 [-0.186, +0.185] (includes zero) | +0.060 [-0.102, +0.229] (includes zero) | -0.029 [-0.136, +0.075] (includes zero) |
| five fields, five fields, no frame | +0.013 [-0.143, +0.174] (includes zero) | -0.092 [-0.238, +0.060] (includes zero) | -0.021 [-0.140, +0.096] (includes zero) |
| five fields, five fields, ion side | +0.068 [-0.069, +0.197] (includes zero) | -0.124 [-0.277, +0.020] (includes zero) | +0.028 [-0.031, +0.090] (includes zero) |
| five fields, five fields, amounts kept | +0.021 [-0.033, +0.077] (includes zero) | +0.008 [-0.044, +0.062] (includes zero) | +0.002 [-0.025, +0.029] (includes zero) |
| five fields, amounts kept, ring counts | -0.006 [-0.161, +0.150] (includes zero) | +0.141 [-0.032, +0.308] (includes zero) | -0.192 [-0.334, -0.052] |
| five fields, electrostatic field | -0.021 [-0.135, +0.091] (includes zero) | +0.069 [-0.022, +0.159] (includes zero) | +0.025 [-0.034, +0.086] (includes zero) |
| five fields, ion probe field | -0.267 [-0.426, -0.107] | -0.215 [-0.348, -0.089] | -0.111 [-0.201, -0.026] |
| ion probe field, electrostatic field | +0.246 [+0.068, +0.417] | +0.284 [+0.122, +0.455] | +0.136 [+0.032, +0.243] |
| five fields + probe, five fields | -0.007 [-0.043, +0.028] (includes zero) | +0.028 [-0.008, +0.072] (includes zero) | +0.011 [-0.010, +0.032] (includes zero) |
| five fields + ring counts, ring counts | +0.028 [-0.069, +0.123] (includes zero) | +0.055 [-0.034, +0.149] (includes zero) | -0.000 [-0.136, +0.118] (includes zero) |
| five fields + ring counts, five fields | +0.013 [-0.149, +0.170] (includes zero) | -0.094 [-0.242, +0.052] (includes zero) | +0.190 [+0.080, +0.297] |

## The selectivity by three routes

| Route | q2, random folds | q2, families held out |
|---|---|---|
| ring counts, a model of the difference | +0.410 ± 0.109 (0.85) | -0.111 (1.23) |
| ring counts, two separate models | -0.101 ± 1.009 (1.16) | +0.160 (1.07) |
| complex shape, a model of the difference | +0.472 ± 0.050 (0.80) | +0.332 (0.95) |
| complex shape, two separate models | -1.854 ± 2.137 (1.86) | -0.531 (1.45) |
| five fields, a model of the difference | +0.349 ± 0.074 (0.89) | +0.320 (0.96) |
| five fields, two separate models | -0.345 ± 0.151 (1.28) | -0.245 (1.30) |
| five fields, the two-task model | +0.374 ± 0.059 (0.87) | +0.348 (0.94) |

The correlation of the two endpoints chosen inside the folds is 0.82 on average.

| Two-task model against | Random folds | Families held out |
|---|---|---|
| five fields, a model of the difference | -0.007 [-0.023, +0.010] (includes zero) | -0.011 [-0.050, +0.028] (includes zero) |
| five fields, two separate models | -0.315 [-0.416, -0.208] | -0.306 [-0.436, -0.177] |
| ring counts, a model of the difference | +0.082 [+0.001, +0.163] | -0.201 [-0.350, -0.061] |
| complex shape, a model of the difference | +0.095 [+0.029, +0.159] | +0.073 [-0.031, +0.164] (includes zero) |

## The co-field of the five-field model, read in the frame

The acceptor, hydrophobic and electrostatic co-fields are read as the effect of one atom placed at a point (one acceptor atom, one hydrophobic atom, one negative unit charge), in log K units; the steric co-field is read as it is. The plane is the set of heights of at most 0.5 A, and every value is averaged over the turn about z.

| Ion and hosts | Field | Radius of the largest gain in the plane (A) | 95 % interval from bootstrap samples of the hosts | Gain there (log K) | Radius at which the gain turns from negative to positive (A) | 95 % interval | Correlation of the maps of two halves of the hosts | The same with shuffled constants |
|---|---|---|---|---|---|---|---|---|
| Na+ | electrostatic | 3.00 | 2.75 to 3.51 | +0.258 | 2.11 | 1.76 to 3.79 | +0.25 ± 0.12 | +0.02 ± 0.15 |
| Na+ | steric | 3.75 | 0.00 to 7.00 | +0.016 | 3.13 | 2.52 to 4.25 | +0.24 ± 0.18 | +0.05 ± 0.18 |
| Na+ | hydrophobic | 0.50 | 0.00 to 7.00 | +0.104 | 4.32 | 3.60 to 4.95 | +0.26 ± 0.12 | +0.01 ± 0.16 |
| Na+ | acceptor | 3.00 | 0.00 to 7.00 | +0.043 | 2.67 | 2.26 to 6.93 | +0.23 ± 0.14 | +0.01 ± 0.19 |
| Na+, monocyclic hosts | electrostatic | 3.00 | 2.75 to 3.26 | +0.081 | 2.28 | 2.00 to 2.93 | +0.18 ± 0.07 | +0.05 ± 0.15 |
| Na+, monocyclic hosts | steric | 3.50 | 3.25 to 3.75 | +0.037 | 2.78 | 2.43 to 3.22 | +0.18 ± 0.11 | -0.02 ± 0.10 |
| Na+, monocyclic hosts | hydrophobic | 1.75 | 1.50 to 4.50 | +0.112 | 4.03 | 3.83 to 4.47 | +0.27 ± 0.08 | +0.04 ± 0.12 |
| Na+, monocyclic hosts | acceptor | 3.00 | 2.75 to 3.76 | +0.039 | 2.58 | 2.21 to 3.82 | +0.28 ± 0.08 | +0.06 ± 0.14 |
| K+ | electrostatic | 3.25 | 3.00 to 3.25 | +0.206 | 2.59 | 2.40 to 2.69 | +0.40 ± 0.06 | -0.03 ± 0.12 |
| K+ | steric | 4.25 | 4.00 to 7.00 | +0.001 | 4.18 | 3.24 to 6.32 | +0.52 ± 0.11 | +0.00 ± 0.19 |
| K+ | hydrophobic | 1.75 | 1.75 to 2.00 | +0.207 | 4.48 | 4.18 to 4.81 | +0.35 ± 0.10 | -0.01 ± 0.10 |
| K+ | acceptor | 3.25 | 3.25 to 3.50 | +0.095 | 2.87 | 2.61 to 3.17 | +0.49 ± 0.10 | -0.02 ± 0.13 |
| K+, monocyclic hosts | electrostatic | 3.00 | 3.00 to 3.00 | +0.082 | 2.47 | 2.38 to 2.58 | +0.41 ± 0.07 | -0.01 ± 0.09 |
| K+, monocyclic hosts | steric | 3.75 | 3.50 to 3.75 | +0.034 | 3.12 | 2.85 to 3.29 | +0.35 ± 0.06 | +0.01 ± 0.12 |
| K+, monocyclic hosts | hydrophobic | 2.00 | 2.00 to 2.25 | +0.072 | 4.17 | 4.05 to 4.39 | +0.26 ± 0.08 | +0.01 ± 0.06 |
| K+, monocyclic hosts | acceptor | 3.00 | 3.00 to 3.00 | +0.057 | 2.60 | 2.48 to 2.71 | +0.44 ± 0.07 | -0.01 ± 0.10 |

| Ion and hosts | Field | Largest gain anywhere: distance from the axis (A) | Height (A) | Gain | Largest loss anywhere: distance from the axis (A) | Height (A) | Loss | Variation of the gain with the turn, in the plane (sd / mean) |
|---|---|---|---|---|---|---|---|---|
| Na+ | electrostatic | 0.00 | +1.75 | +0.904 | 1.00 | -0.25 | -0.411 | 1.56 |
| Na+ | steric | 0.00 | +2.50 | +0.140 | 1.25 | -0.25 | -0.093 | 4.87 |
| Na+ | hydrophobic | 4.25 | +1.25 | +0.187 | 0.00 | +2.25 | -0.894 | 0.42 |
| Na+ | acceptor | 0.00 | +1.75 | +0.739 | 1.00 | -0.25 | -0.487 | 9.30 |
| Na+, monocyclic hosts | electrostatic | 2.50 | +0.75 | +0.119 | 1.50 | -0.25 | -0.163 | 1.30 |
| Na+, monocyclic hosts | steric | 3.50 | +0.75 | +0.044 | 1.50 | +0.00 | -0.085 | 1.69 |
| Na+, monocyclic hosts | hydrophobic | 1.75 | +0.00 | +0.126 | 3.25 | +0.00 | -0.163 | 0.67 |
| Na+, monocyclic hosts | acceptor | 2.75 | +0.75 | +0.079 | 1.50 | +0.00 | -0.204 | 2.06 |
| K+ | electrostatic | 1.00 | +2.25 | +0.418 | 1.75 | +0.25 | -0.360 | 0.71 |
| K+ | steric | 0.00 | +3.00 | +0.279 | 2.25 | +0.50 | -0.219 | 78.18 |
| K+ | hydrophobic | 2.00 | +0.00 | +0.230 | 0.00 | +2.50 | -0.460 | 0.38 |
| K+ | acceptor | 0.00 | +2.50 | +0.590 | 1.75 | +0.00 | -0.443 | 1.49 |
| K+, monocyclic hosts | electrostatic | 3.00 | +0.00 | +0.086 | 2.00 | +0.00 | -0.131 | 0.70 |
| K+, monocyclic hosts | steric | 3.25 | -1.00 | +0.110 | 2.00 | +0.25 | -0.192 | 3.02 |
| K+, monocyclic hosts | hydrophobic | 2.00 | +0.00 | +0.087 | 3.25 | +0.25 | -0.101 | 0.28 |
| K+, monocyclic hosts | acceptor | 2.75 | +1.00 | +0.064 | 2.00 | +0.00 | -0.156 | 0.94 |

| Ion and hosts | Hosts | Field width alpha | Sum of the ionic radii (A) | Mean radius of the donor ring (A) | Ten strongest hosts (A) | Donors within the plane | Concentration of the donors in the turn (0 none, 1 complete) |
|---|---|---|---|---|---|---|---|
| Na+ | 79 | 2 | 2.42 | 2.35 | 2.39 | 48 % | 0.01 |
| Na+, monocyclic hosts | 70 | 4 | 2.42 | 2.34 | 2.37 | 51 % | 0.01 |
| K+ | 79 | 4 | 2.78 | 2.56 | 2.77 | 61 % | 0.00 |
| K+, monocyclic hosts | 70 | 8 | 2.78 | 2.54 | 2.74 | 68 % | 0.00 |

| Ion and hosts | Cryptands with an O, N or S atom within 1.5 A of the largest acceptor gain | Other hosts with one | Mean log K with | Without |
|---|---|---|---|---|
| Na+ | 6 of 9 | 4 of 70 | +4.81 | +3.42 |
| K+ | 7 of 9 | 4 of 70 | +6.33 | +3.83 |

Mean radius of the donor ring by family (A): Na+ 12-crown-4 1.96, 15-crown-5 2.30, benzo-15-crown-5 2.27, aza-15-crown-5 2.32, 18-crown-6 2.40, benzo-18-crown-6 2.41, aza-18-crown-6 2.41, larger ring 2.44, cryptand 2.44; K+ 12-crown-4 1.99, 15-crown-5 2.40, benzo-15-crown-5 2.36, aza-15-crown-5 2.44, 18-crown-6 2.70, benzo-18-crown-6 2.67, aza-18-crown-6 2.74, larger ring 2.77, cryptand 2.72.

