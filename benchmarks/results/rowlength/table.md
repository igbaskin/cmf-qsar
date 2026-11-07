# What the row length is worth (plan item M15)

The invariant kernel divides every feature row by its own length. These are the four ways of treating that one number, each with the setting and the ridge chosen inside the training part, on the samples and folds of the benchmark each series comes from. The difference is in q2 against the first arm, which is what the program does today, with the bootstrap interval of rule 4.

## What it says

| series | the length against the target: Pearson | comparisons | the length kept: better / worse / inside the interval | the length as an extra channel: better / worse / inside | the largest gain from keeping it |
|---|---|---|---|---|---|
| lithium | -0.25 to +0.48 | 1 | 0 / 0 / 0 (1 with no interval) | 0 / 0 / 0 (1 with no interval) | no interval |
| sac | -0.54 to +0.14 | 12 | 0 / 3 / 9 | 1 / 2 / 9 | +0.037 [-0.055, +0.134] |
| hops | +0.08 to +0.60 | 3 | 1 / 0 / 2 | 2 / 0 / 1 | +0.144 [+0.096, +0.198] |
| defects | -0.93 to -0.88 | 2 | 2 / 0 / 0 | 2 / 0 / 0 | +0.001 [+0.001, +0.001] |
| molecules | +0.11 to +0.12 | 1 | 0 / 1 / 0 | 0 / 0 / 1 | -0.043 [-0.073, -0.015] |

On the defects series q2 is already +0.9989, so it cannot show much. The error per defect site is what moves: from 36.5 to 2.4 meV on MoS2, from 30.8 to 6.3 meV on WSe2, when the length is kept.

## lithium

site profile (M5), 104 samples.

| description | target | arm | q2 | error | difference against the first arm |
|---|---|---|---|---|---|
| site profile (M5) | site energy | the length divided out | +0.484 | 0.080 | the reference |
| site profile (M5) | site energy | the length kept | -0.863 | 0.151 | -1.347, no interval |
| site profile (M5) | site energy | the length as an extra channel | +0.439 | 0.083 | -0.045, no interval |
| site profile (M5) | site energy | the length alone | -158.563 | 1.401 | -159.047, no interval |

| setting | spread of the length, per cent of its mean | with the target: Pearson | Spearman |
|---|---|---|---|
| r=4 A | 5 | -0.212 | -0.198 |
| r=6 A | 9 | -0.248 | -0.531 |
| r=8 A | 13 | +0.476 | +0.372 |

## sac

site profile (M5), 139 samples.

| description | target | arm | q2 | error | difference against the first arm |
|---|---|---|---|---|---|
| site profile (M5), random | dE_H | the length divided out | +0.441 | 0.716 | the reference |
| site profile (M5), random | dE_H | the length kept | +0.445 | 0.713 | +0.004 [-0.056, +0.072] |
| site profile (M5), random | dE_H | the length as an extra channel | +0.450 | 0.710 | +0.009 [-0.027, +0.045] |
| site profile (M5), random | dE_H | the length alone | -0.041 | 0.976 | -0.482 [-0.617, -0.336] |
| site profile (M5), random | dE_O | the length divided out | +0.818 | 0.942 | the reference |
| site profile (M5), random | dE_O | the length kept | +0.824 | 0.927 | +0.006 [-0.025, +0.036] |
| site profile (M5), random | dE_O | the length as an extra channel | +0.825 | 0.925 | +0.007 [-0.005, +0.021] |
| site profile (M5), random | dE_O | the length alone | +0.075 | 2.126 | -0.743 [-0.843, -0.666] |
| site profile (M5), random | dE_OH | the length divided out | +0.906 | 0.449 | the reference |
| site profile (M5), random | dE_OH | the length kept | +0.896 | 0.473 | -0.010 [-0.032, +0.010] |
| site profile (M5), random | dE_OH | the length as an extra channel | +0.907 | 0.446 | +0.001 [-0.003, +0.006] |
| site profile (M5), random | dE_OH | the length alone | +0.236 | 1.282 | -0.670 [-0.791, -0.584] |
| site profile (M5), random | dE_OOH | the length divided out | +0.715 | 0.721 | the reference |
| site profile (M5), random | dE_OOH | the length kept | +0.695 | 0.746 | -0.021 [-0.058, +0.016] |
| site profile (M5), random | dE_OOH | the length as an extra channel | +0.709 | 0.729 | -0.006 [-0.028, +0.017] |
| site profile (M5), random | dE_OOH | the length alone | +0.260 | 1.161 | -0.455 [-0.591, -0.333] |
| site profile (M5), metal-out | dE_H | the length divided out | +0.250 | 0.829 | the reference |
| site profile (M5), metal-out | dE_H | the length kept | +0.286 | 0.809 | +0.037 [-0.055, +0.134] |
| site profile (M5), metal-out | dE_H | the length as an extra channel | +0.268 | 0.819 | +0.018 [-0.027, +0.064] |
| site profile (M5), metal-out | dE_H | the length alone | -0.107 | 1.007 | -0.356 [-0.503, -0.183] |
| site profile (M5), metal-out | dE_O | the length divided out | +0.543 | 1.495 | the reference |
| site profile (M5), metal-out | dE_O | the length kept | +0.436 | 1.661 | -0.107 [-0.183, -0.046] |
| site profile (M5), metal-out | dE_O | the length as an extra channel | +0.518 | 1.535 | -0.025 [-0.046, -0.006] |
| site profile (M5), metal-out | dE_O | the length alone | -0.044 | 2.260 | -0.587 [-0.722, -0.438] |
| site profile (M5), metal-out | dE_OH | the length divided out | +0.850 | 0.568 | the reference |
| site profile (M5), metal-out | dE_OH | the length kept | +0.766 | 0.709 | -0.084 [-0.171, -0.017] |
| site profile (M5), metal-out | dE_OH | the length as an extra channel | +0.855 | 0.559 | +0.004 [-0.010, +0.022] |
| site profile (M5), metal-out | dE_OH | the length alone | +0.077 | 1.409 | -0.773 [-0.981, -0.622] |
| site profile (M5), metal-out | dE_OOH | the length divided out | +0.615 | 0.838 | the reference |
| site profile (M5), metal-out | dE_OOH | the length kept | +0.552 | 0.904 | -0.063 [-0.146, +0.008] |
| site profile (M5), metal-out | dE_OOH | the length as an extra channel | +0.544 | 0.912 | -0.071 [-0.135, -0.022] |
| site profile (M5), metal-out | dE_OOH | the length alone | +0.125 | 1.263 | -0.490 [-0.741, -0.256] |
| site profile (M5), motif-out | dE_H | the length divided out | +0.345 | 0.775 | the reference |
| site profile (M5), motif-out | dE_H | the length kept | +0.217 | 0.847 | -0.128 [-0.321, +0.028] |
| site profile (M5), motif-out | dE_H | the length as an extra channel | +0.336 | 0.780 | -0.009 [-0.094, +0.070] |
| site profile (M5), motif-out | dE_H | the length alone | -0.090 | 0.999 | -0.435 [-0.584, -0.266] |
| site profile (M5), motif-out | dE_O | the length divided out | +0.833 | 0.904 | the reference |
| site profile (M5), motif-out | dE_O | the length kept | +0.811 | 0.961 | -0.022 [-0.060, +0.015] |
| site profile (M5), motif-out | dE_O | the length as an extra channel | +0.823 | 0.931 | -0.010 [-0.030, +0.007] |
| site profile (M5), motif-out | dE_O | the length alone | +0.097 | 2.101 | -0.736 [-0.830, -0.660] |
| site profile (M5), motif-out | dE_OH | the length divided out | +0.832 | 0.601 | the reference |
| site profile (M5), motif-out | dE_OH | the length kept | +0.807 | 0.644 | -0.025 [-0.079, +0.023] |
| site profile (M5), motif-out | dE_OH | the length as an extra channel | +0.850 | 0.568 | +0.018 [+0.002, +0.041] |
| site profile (M5), motif-out | dE_OH | the length alone | +0.263 | 1.259 | -0.569 [-0.694, -0.472] |
| site profile (M5), motif-out | dE_OOH | the length divided out | +0.647 | 0.802 | the reference |
| site profile (M5), motif-out | dE_OOH | the length kept | +0.570 | 0.885 | -0.077 [-0.144, -0.015] |
| site profile (M5), motif-out | dE_OOH | the length as an extra channel | +0.641 | 0.810 | -0.007 [-0.049, +0.031] |
| site profile (M5), motif-out | dE_OOH | the length alone | +0.266 | 1.157 | -0.381 [-0.534, -0.248] |

| setting | spread of the length, per cent of its mean | with the target: Pearson | Spearman |
|---|---|---|---|
| dE_H, r=4 A | 39 | +0.141 | +0.017 |
| dE_H, r=6 A | 37 | +0.132 | -0.012 |
| dE_H, r=8 A | 35 | +0.121 | -0.015 |
| dE_O, r=4 A | 39 | -0.284 | -0.393 |
| dE_O, r=6 A | 37 | -0.306 | -0.446 |
| dE_O, r=8 A | 35 | -0.321 | -0.457 |
| dE_OH, r=4 A | 39 | -0.473 | -0.587 |
| dE_OH, r=6 A | 37 | -0.501 | -0.642 |
| dE_OH, r=8 A | 35 | -0.520 | -0.652 |
| dE_OOH, r=4 A | 40 | -0.497 | -0.560 |
| dE_OOH, r=6 A | 38 | -0.525 | -0.611 |
| dE_OOH, r=8 A | 36 | -0.542 | -0.622 |

## hops

1801 samples.

| description | target | arm | q2 | error | difference against the first arm |
|---|---|---|---|---|---|
| counts about the midpoint | barrier | the length divided out | +0.541 | 1.291 | the reference |
| counts about the midpoint | barrier | the length kept | +0.557 | 1.267 | +0.017 [-0.030, +0.055] |
| counts about the midpoint | barrier | the length as an extra channel | +0.543 | 1.288 | +0.002 [-0.005, +0.010] |
| counts about the midpoint | barrier | the length alone | +0.030 | 1.875 | -0.510 [-0.575, -0.435] |
| the two end sites (M5) | barrier | the length divided out | +0.114 | 1.793 | the reference |
| the two end sites (M5) | barrier | the length kept | +0.106 | 1.801 | -0.008 [-0.035, +0.014] |
| the two end sites (M5) | barrier | the length as an extra channel | +0.129 | 1.777 | +0.015 [+0.002, +0.032] |
| the two end sites (M5) | barrier | the length alone | -0.000 | 1.905 | -0.114 [-0.155, -0.073] |
| the hop environment (BT7) | barrier | the length divided out | +0.572 | 1.246 | the reference |
| the hop environment (BT7) | barrier | the length kept | +0.716 | 1.016 | +0.144 [+0.096, +0.198] |
| the hop environment (BT7) | barrier | the length as an extra channel | +0.641 | 1.140 | +0.070 [+0.027, +0.114] |
| the hop environment (BT7) | barrier | the length alone | +0.354 | 1.530 | -0.218 [-0.314, -0.123] |

| setting | spread of the length, per cent of its mean | with the target: Pearson | Spearman |
|---|---|---|---|
| counts about the midpoint, counts | 81 | +0.194 | +0.235 |
| the hop environment (BT7), rho=0.25 | 65 | +0.601 | +0.587 |
| the two end sites (M5), r=6 A | 44 | +0.076 | +0.098 |

## defects

whole-cell autocorrelation, 4000 samples.

| description | target | arm | q2 | error | meV per site | difference against the first arm |
|---|---|---|---|---|---|---|
| whole-cell autocorrelation, MoS2 | formation energy | the length divided out | +0.9989 | 0.125 | 36.5 | the reference |
| whole-cell autocorrelation, MoS2 | formation energy | the length kept | +1.0000 | 0.016 | 2.4 | +0.0011 [+0.0009, +0.0012] |
| whole-cell autocorrelation, MoS2 | formation energy | the length as an extra channel | +0.9997 | 0.063 | 13.7 | +0.0008 [+0.0007, +0.0009] |
| whole-cell autocorrelation, MoS2 | formation energy | the length alone | +0.7794 | 1.782 | 494.3 | -0.2195 [-0.2427, -0.1982] |
| whole-cell autocorrelation, WSe2 | formation energy | the length divided out | +0.9988 | 0.121 | 30.8 | the reference |
| whole-cell autocorrelation, WSe2 | formation energy | the length kept | +0.9998 | 0.053 | 6.3 | +0.0010 [+0.0008, +0.0012] |
| whole-cell autocorrelation, WSe2 | formation energy | the length as an extra channel | +0.9995 | 0.076 | 13.0 | +0.0007 [+0.0006, +0.0009] |
| whole-cell autocorrelation, WSe2 | formation energy | the length alone | +0.8581 | 1.299 | 344.9 | -0.1407 [-0.1565, -0.1260] |

| setting | spread of the length, per cent of its mean | with the target: Pearson | Spearman |
|---|---|---|---|
| MoS2, sigma 0.35 | 16 | -0.882 | -0.862 |
| MoS2, sigma 0.5 | 16 | -0.882 | -0.862 |
| MoS2, sigma 0.75 | 16 | -0.883 | -0.862 |
| WSe2, sigma 0.35 | 16 | -0.925 | -0.932 |
| WSe2, sigma 0.5 | 16 | -0.926 | -0.931 |
| WSe2, sigma 0.75 | 16 | -0.926 | -0.933 |

## molecules

whole-molecule autocorrelation, 1162 samples.

| description | target | arm | q2 | error | difference against the first arm |
|---|---|---|---|---|---|
| whole-molecule autocorrelation | log K | the length divided out | +0.477 | 0.587 | the reference |
| whole-molecule autocorrelation | log K | the length kept | +0.435 | 0.611 | -0.043 [-0.073, -0.015] |
| whole-molecule autocorrelation | log K | the length as an extra channel | +0.474 | 0.589 | -0.003 [-0.007, +0.000] |
| whole-molecule autocorrelation | log K | the length alone | +0.010 | 0.809 | -0.468 [-0.515, -0.413] |

| setting | spread of the length, per cent of its mean | with the target: Pearson | Spearman |
|---|---|---|---|
| width 0.5 A | 64 | +0.109 | +0.123 |
| width 0.75 A | 66 | +0.112 | +0.132 |
| width 1 A | 68 | +0.115 | +0.140 |

