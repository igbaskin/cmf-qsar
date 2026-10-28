# What the ligand field carries that its contacts with the pocket do not

View A is the five fields of the ligand at the attenuation the evidence picks for the series; view B is the linear kernel of its interaction profile with the stored binding site, the complementarity of the fields resolved by receptor residue and interaction type. *Held-out ρ* is the mean canonical correlation over all pairs on held-out molecules; *real* counts the pairs above the 95th percentile of a shuffled pairing; *contained* counts the directions of the field of which the profile explains more than half the variance on held-out molecules. The q² values are kernel ridge regression on the whole field, on its part inside the contained directions (the contacts), on the rest, and on the profile alone, with the regularisation chosen inside the folds.

| series | n | α | residues | mean ρ | real | contained | q² field | q² contacts | q² field's own | own − contacts [95 %] | q² profile alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| openff/syk | 44 | 2 | 62 | +0.369 | 23/35 | 4/43 | +0.371 | -0.100 | +0.086 | +0.187 [-0.303, +0.577] | -0.152 |
| schindler/syk | 44 | 2 | 64 | +0.364 | 22/35 | 4/43 | +0.355 | -0.039 | -0.040 | -0.002 [-0.511, +0.520] | -0.304 |
| schindler/hif2a | 42 | 0.3 | 70 | +0.898 | 33/33 | 32/41 | +0.147 | +0.023 | +0.071 | +0.048 [-0.547, +0.636] | +0.093 |
| schindler/pfkfb3 | 40 | 2 | 71 | +0.416 | 21/31 | 6/39 | -0.111 | -0.199 | -0.062 | +0.137 [-0.023, +0.292] | -0.382 |
| schindler/hif2a-jmc2023 | 38 | 1 | 66 | +0.648 | 27/30 | 14/37 | +0.583 | +0.337 | +0.435 | +0.098 [-0.150, +0.339] | +0.360 |
| openff/hif2a | 37 | 2 | 70 | +0.592 | 26/29 | 12/36 | -0.093 | -0.115 | -0.066 | +0.048 [-0.117, +0.365] | -0.171 |
| openff/bace | 36 | 1 | 38 | +0.650 | 25/28 | 16/35 | +0.283 | +0.113 | +0.127 | +0.014 [-0.437, +0.315] | +0.333 |
| openff/pde10 | 35 | 2 | 81 | +0.467 | 19/27 | 6/34 | +0.237 | -0.120 | +0.147 | +0.266 [-0.106, +0.591] | -0.879 |
| schindler/cdk8 | 33 | 1 | 62 | +0.655 | 24/26 | 12/32 | +0.384 | +0.217 | -0.210 | -0.427 [-0.699, -0.170] | -0.533 |
| openff/bace_hunt | 32 | 2 | 68 | +0.426 | 18/25 | 5/29 | +0.311 | -0.065 | -0.782 | -0.717 [-3.152, +0.455] | -0.001 |
| openff/pfkfb3 | 32 | 2 | 66 | +0.461 | 15/25 | 8/31 | +0.538 | +0.502 | -0.049 | -0.551 [-0.735, -0.378] | +0.386 |
| openff/cdk8 | 31 | 2 | 62 | +0.527 | 20/24 | 5/30 | +0.414 | -0.397 | -0.307 | +0.090 [-1.009, +1.792] | +0.004 |
| openff/p38 | 29 | 0.3 | 66 | +0.784 | 23/23 | 19/28 | -0.061 | -0.047 | -0.067 | -0.020 [-0.141, +0.146] | +0.035 |
| schindler/eg5 | 28 | 0.5 | 100 | +0.780 | 22/22 | 15/27 | -0.169 | +0.145 | +0.046 | -0.099 [-0.456, +0.572] | +0.148 |
| openff/eg5 | 27 | 0.3 | 77 | +0.717 | 21/21 | 9/26 | +0.153 | +0.215 | +0.119 | -0.096 [-0.428, +0.213] | +0.211 |
| openff/tnks2 | 27 | 2 | 60 | +0.494 | 15/21 | 5/26 | +0.082 | -0.285 | +0.177 | +0.462 [-0.229, +1.059] | +0.344 |
| schindler/shp2 | 26 | 0.5 | 61 | +0.729 | 18/20 | 12/25 | -0.301 | -0.118 | +0.435 | +0.553 [+0.048, +1.051] | -0.125 |
| openff/mcl1 | 25 | 0.5 | 61 | +0.774 | 18/19 | 13/24 | +0.426 | +0.499 | +0.172 | -0.326 [-0.771, -0.061] | +0.610 |
| openff/shp2 | 24 | 0.3 | 59 | +0.807 | 19/19 | 14/23 | -0.142 | -0.197 | -0.062 | +0.134 [-0.334, +0.557] | -0.083 |
| openff/ptp1b | 22 | 0.5 | 57 | +0.650 | 15/17 | 8/21 | +0.596 | +0.334 | +0.090 | -0.244 [-0.629, +0.470] | +0.371 |
| openff/thrombin | 22 | 2 | 67 | +0.456 | 10/17 | 4/21 | -0.215 | -0.918 | +0.111 | +1.029 [+0.179, +1.726] | -0.568 |
| **mean** |  |  |  |  |  |  | **+0.180** | **-0.010** | **+0.018** | **+0.028** | **-0.015** |

Over 21 series: 83 % of the canonical pairs hold out above the shuffled pairing, and the profile contains 35 % of the field's directions on average. The contacts predict the activity at q² **-0.010** on average, the field's own part at **+0.018**, the whole field at **+0.180** and the profile alone at **-0.015**; the difference (own minus contacts) is **+0.028**, with its interval above zero on 2 and below zero on 3 of the 21 series.

So on 16 of the 21 series the interval covers zero, and the few that do not fall on both sides: on sets of twenty to forty ligands neither part can be said to carry more of the activity (rule 4). What the table does show is that the contact profile is not a predictor on its own, and that the whole field predicts better than either of its parts.

Series whose ligands sit more than 6 Å from the stored pocket are left out, because the structure is not in their frame; a series that is another's molecules again, against a pocket file differing in a loop the ligands do not touch, is counted once.
