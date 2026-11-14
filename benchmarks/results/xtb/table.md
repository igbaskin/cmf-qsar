# A cheap quantum stage over many series

GFN2-xTB per-atom properties as fields, against the five physico-chemical fields alone, on the same alignment, the same alpha and lambda grids and the same 10-fold cross-validation. Plan item D3 asks whether any series in the repository is one a quantum field helps; every difference carries the bootstrap interval of rule 4, so a gain below the interval is not counted.

## What it costs

On 40 molecules of 20 heavy atoms on average, the same molecules ORCA was run on in `benchmarks/results/corrosion_qm`: **0.25 s** per molecule with xtb against **360 s** with ORCA, a factor of **1437**. Both numbers include the two ionic states a Fukui index is built from.

## Whether it says the same thing as the expensive method

Per-atom quantities over every atom of 101 molecules of the corrosion series, xtb against ORCA.

| quantity | atoms | correlation | rank correlation |
| --- | ---: | ---: | ---: |
| fukui_plus | 3071 | +0.782 | +0.692 |
| fukui_minus | 3071 | +0.708 | +0.599 |
| dual | 3071 | +0.777 | +0.697 |
| charge | 3071 | +0.948 | +0.886 |

The charges are a Mulliken-like tight-binding population against a Hirshfeld charge, so they are different quantities; what the correlation says is whether they order the atoms the same way.

## Whether it helps

52 series of 3511 molecules, from 3 collections.

| model | series ahead of the five fields | series behind | median change in q2 |
| --- | ---: | ---: | ---: |
| base+fukui | 7 | 1 | +0.014 |
| xtb_charges | 3 | 1 | +0.007 |
| base+desc | 6 | 5 | -0.009 |

### Every series

| collection | series | n | q2 base | q2 base+fukui | q2 xtb_charges | q2 base+desc | best change over base |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fep | a2a_hip278 | 29 | +0.694 | +0.714 | +0.661 | +0.683 | base+fukui +0.020 [-0.075, +0.219] |
| fep | bace | 36 | +0.361 | +0.389 | +0.360 | +0.177 | base+fukui +0.028 [-0.081, +0.190] |
| fep | bace_ciordia_retro | 33 | +0.309 | +0.376 | +0.403 | +0.291 | xtb_charges +0.095 [-0.206, +0.310] |
| fep | bace_hunt | 32 | +0.321 | +0.289 | +0.340 | +0.409 | base+desc +0.088 [-0.108, +0.341] |
| fep | bace_keranen_p2 | 18 | +0.403 | +0.559 | +0.358 | -0.002 | base+fukui +0.156 [-0.243, +1.003] |
| fep | bace_p3_arg368_in | 21 | +0.379 | +0.387 | +0.436 | +0.386 | xtb_charges +0.057 [-0.002, +0.134] |
| fep | cdk2 | 16 | +0.818 | +0.785 | +0.783 | +0.717 | base+fukui -0.034 [-0.147, +0.047] |
| fep | cdk8 | 31 | +0.531 | +0.642 | +0.487 | +0.534 | base+fukui +0.111 [-0.023, +0.314] |
| fep | cdk8_5cei_new_helix_loop_extra | 41 | +0.543 | +0.612 | +0.645 | +0.583 | xtb_charges +0.102 [-0.080, +0.215] |
| fep | cmet | 24 | +0.782 | +0.773 | +0.795 | +0.721 | xtb_charges +0.013 [-0.004, +0.035] |
| fep | eg5 | 27 | +0.464 | +0.455 | +0.488 | +0.387 | xtb_charges +0.024 [-0.251, +0.147] |
| fep | eg5_alternativeloop | 28 | +0.280 | +0.227 | +0.322 | +0.215 | xtb_charges +0.042 [-0.001, +0.130] |
| fep | eg5_extraprotomers | 34 | +0.414 | +0.426 | +0.475 | +0.446 | xtb_charges +0.061 [-0.021, +0.210] |
| fep | fxa_yoshikawa_set | 27 | +0.579 | +0.639 | +0.608 | +0.629 | base+fukui +0.060 [-0.003, +0.128] |
| fep | galectin3_extra | 26 | +0.684 | +0.634 | +0.692 | +0.567 | xtb_charges +0.008 [-0.070, +0.079] |
| fep | hfaah | 24 | +0.764 | +0.732 | +0.630 | +0.781 | base+desc +0.017 [-0.159, +0.216] |
| fep | hif2a | 37 | +0.384 | +0.206 | +0.409 | +0.264 | xtb_charges +0.026 [-0.116, +0.190] |
| fep | hif2a-jmc2023 | 38 | +0.648 | +0.560 | +0.655 | +0.664 | base+desc +0.016 [-0.092, +0.120] |
| fep | hif2a_automap | 51 | +0.790 | +0.718 | +0.821 | +0.713 | xtb_charges +0.031 [-0.028, +0.113] |
| fep | hne | 20 | +0.176 | +0.375 | +0.159 | +0.047 | base+fukui +0.199 [-0.025, +0.400] |
| fep | jnk1 | 21 | +0.682 | +0.628 | +0.698 | +0.527 | xtb_charges +0.016 [-0.029, +0.101] |
| fep | jnk1_manual_flips | 38 | +0.938 | +0.853 | +0.933 | +0.916 | xtb_charges -0.005 [-0.011, +0.003] |
| fep | mcl1 | 25 | +0.575 | +0.609 | +0.539 | +0.667 | base+desc +0.092 [+0.021, +0.214] |
| fep | mcl1_extra_flips | 69 | +0.669 | +0.762 | +0.668 | +0.663 | base+fukui +0.092 [+0.020, +0.164] |
| fep | ox2_hip_custcore | 51 | +0.072 | +0.023 | +0.101 | +0.026 | xtb_charges +0.029 [-0.192, +0.192] |
| fep | p2y1_meta_sub | 20 | -0.152 | -0.213 | -0.149 | -0.226 | xtb_charges +0.003 [-0.022, +0.035] |
| fep | p38 | 29 | +0.261 | +0.105 | +0.294 | +0.196 | xtb_charges +0.033 [-0.128, +0.170] |
| fep | pde10 | 35 | +0.294 | +0.257 | +0.278 | +0.291 | base+desc -0.003 [-0.068, +0.062] |
| fep | pde2 | 21 | +0.424 | +0.518 | +0.496 | +0.402 | base+fukui +0.094 [-0.054, +0.295] |
| fep | pfkfb3 | 32 | +0.594 | +0.538 | +0.513 | +0.637 | base+desc +0.043 [-0.044, +0.120] |
| fep | pfkfb3_automap | 87 | +0.575 | +0.637 | +0.593 | +0.554 | base+fukui +0.063 [-0.005, +0.136] |
| fep | ptp1b | 22 | +0.694 | +0.719 | +0.773 | +0.846 | base+desc +0.152 [-0.051, +0.504] |
| fep | renin_customcore | 29 | +0.326 | +0.247 | +0.329 | +0.230 | xtb_charges +0.003 [-0.008, +0.019] |
| fep | shp2 | 24 | +0.300 | +0.071 | +0.381 | +0.083 | xtb_charges +0.081 [+0.016, +0.221] |
| fep | syk | 44 | +0.417 | +0.488 | +0.530 | +0.406 | xtb_charges +0.113 [-0.054, +0.335] |
| fep | syk_4puz_fullmap | 51 | +0.398 | +0.481 | +0.365 | +0.427 | base+fukui +0.083 [-0.013, +0.180] |
| fep | throm_nozob_hip75 | 39 | +0.652 | +0.781 | +0.665 | +0.577 | base+fukui +0.129 [+0.025, +0.238] |
| fep | thrombin | 22 | +0.059 | +0.674 | +0.043 | +0.443 | base+fukui +0.616 [-0.448, +1.405] |
| fep | tnks2 | 27 | +0.484 | +0.480 | +0.583 | +0.433 | xtb_charges +0.100 [-0.089, +0.282] |
| fep | tnks2_fullmap | 33 | +0.538 | +0.719 | +0.593 | +0.539 | base+fukui +0.181 [+0.061, +0.330] |
| fep | tyk2 | 16 | +0.630 | +0.621 | +0.614 | +0.576 | base+fukui -0.009 [-0.128, +0.142] |
| hammett | hammett | 95 | +0.748 | +0.753 | +0.798 | +0.848 | base+desc +0.101 [+0.032, +0.188] |
| hammett | hammett_ens | 464 | +0.932 | +0.955 | +0.914 | +0.950 | base+fukui +0.023 [+0.013, +0.034] |
| hammett | hammett_sym | 380 | +0.895 | +0.924 | +0.876 | +0.906 | base+fukui +0.029 [+0.019, +0.041] |
| sutherland | ACE | 114 | +0.651 | +0.717 | +0.649 | +0.724 | base+desc +0.073 [+0.018, +0.138] |
| sutherland | AChE | 111 | +0.547 | +0.579 | +0.539 | +0.552 | base+fukui +0.032 [-0.044, +0.112] |
| sutherland | BZR | 147 | +0.424 | +0.467 | +0.410 | +0.433 | base+fukui +0.043 [-0.040, +0.126] |
| sutherland | COX2 | 282 | +0.552 | +0.567 | +0.544 | +0.565 | base+fukui +0.015 [-0.024, +0.059] |
| sutherland | DHFR | 361 | +0.651 | +0.693 | +0.663 | +0.664 | base+fukui +0.042 [+0.026, +0.059] |
| sutherland | GPB | 66 | +0.593 | +0.594 | +0.588 | +0.587 | base+fukui +0.001 [-0.130, +0.078] |
| sutherland | THER | 75 | +0.498 | +0.538 | +0.506 | +0.584 | base+desc +0.085 [-0.022, +0.205] |
| sutherland | THR | 88 | +0.701 | +0.706 | +0.674 | +0.656 | base+fukui +0.005 [-0.072, +0.079] |

## What it settles

**The premise of the item holds.** A quantum stage that costs 0.25 s per molecule instead of 360 s is a stage that can be run over every series in the repository, and this is the first time that has been done.

**7 of 52 series (13%) are helped beyond the interval**, the best of the four models being `base+fukui`, and 1 series is made worse. A gain on a minority of series is what the item expected: the Hammett result says the gain belongs to series where resonance moves electron density, and most series are not of that kind. The series named in the table above are the ones to look at.

Read that count against chance: a 95% interval is cleared on one side by about 1.3 of 52 series when there is nothing to find, so 7 is well above what chance would give. The four models are counted separately for the same reason -- taking the best of four per series and counting that would find four times as much by chance, and the `best change over base` column of the table above is there to be looked at rather than counted.

The median change in q2 for `base+fukui` over all series is **+0.014**, with a range from -0.229 to +0.616. A quantum field is not free: it adds a kernel block, and on a series it does not suit it dilutes the ones that work.

