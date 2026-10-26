# Can a direction on every atom represent a Fukui function?

Plan item D1 proposes multiplying each atom's Gaussian by a linear factor, which gives every atom three p functions and keeps the overlaps closed-form (`cmfqsar/anisotropic.py`). It sets two conditions for keeping the extension: the fit error has to drop, and the model has to notice. Measured on the corrosion series with finished ORCA runs, on the valence part of each field, with the four widths the probe fields share.

## Does the fit improve?

| series | n | field | error, s only | error, s and p | coefficients per atom | fitted vs exact kernel, s | s and p |
|---|---:|---|---:|---:|---|---:|---:|
| 161_c1ccc2scnc2c1 | 20 | fukui_plus | 81% | **61%** | 4 → 16 | +0.933 | +0.962 |
| 161_c1ccc2scnc2c1 | 20 | dual | 91% | **68%** | 4 → 16 | +0.899 | +0.955 |
| 12_c1ccoc1 | 27 | fukui_plus | 78% | **61%** | 4 → 16 | +0.949 | +0.988 |
| 12_c1ccoc1 | 27 | dual | 87% | **66%** | 4 → 16 | +0.941 | +0.988 |
| 13_c1c_nH_nn1 | 25 | fukui_plus | 79% | **61%** | 4 → 16 | +0.972 | +0.994 |
| 13_c1c_nH_nn1 | 25 | dual | 87% | **67%** | 4 → 16 | +0.867 | +0.964 |
| 7_c1nncs1 | 29 | fukui_plus | 80% | **63%** | 4 → 16 | +0.934 | +0.989 |
| 7_c1nncs1 | 29 | dual | 91% | **69%** | 4 → 16 | +0.866 | +0.978 |

**The fit does improve**: the relative error falls by 20% on average (from 84% to 64%), for four times as many coefficients per atom. The kernel the fit produces also agrees better with the one the exact field produces, by +0.057 in correlation.

## Does the model notice?

q² of a ten-fold cross-validated model on each kernel: the field alone, and the field added to the five ordinary ones. `grid` keeps the field exactly and is the ceiling a fit is trying to reach.

| series | field | five fields | grid | s alone | s and p alone | five + s | five + s and p |
|---|---|---:|---:|---:|---:|---:|---:|
| 161_c1ccc2scnc2c1 | fukui_plus | +0.689 | +0.638 | +0.636 | +0.627 | +0.658 | +0.655 |
| 161_c1ccc2scnc2c1 | dual | +0.689 | +0.522 | +0.556 | +0.573 | +0.596 | +0.624 |
| 12_c1ccoc1 | fukui_plus | +0.220 | +0.052 | +0.103 | +0.072 | +0.250 | +0.235 |
| 12_c1ccoc1 | dual | +0.220 | +0.028 | +0.057 | +0.012 | +0.272 | +0.241 |
| 13_c1c_nH_nn1 | fukui_plus | +0.654 | +0.636 | +0.685 | +0.684 | +0.672 | +0.681 |
| 13_c1c_nH_nn1 | dual | +0.654 | +0.575 | +0.505 | +0.567 | +0.649 | +0.664 |
| 7_c1nncs1 | fukui_plus | -0.037 | +0.121 | +0.090 | +0.027 | -0.012 | -0.024 |
| 7_c1nncs1 | dual | -0.037 | -0.031 | +0.045 | -0.021 | -0.001 | -0.022 |

**The model does not notice.** Over the 8 series-field pairs the p functions change q² of the field alone by **-0.017** on average (2 of 8 better), and of the field added to the five ordinary ones by **-0.004**. The extension passes the first of the item's two conditions and fails the second.

Why the two measurements disagree is the same reason `run_cdft_kernels.py` found for the isotropic fit: a kernel is an inner product between molecules, and an error that falls the same way for every molecule cancels in it. The p functions recover shape that every molecule of a congeneric series has in the same place, so they improve every field by about the same amount and leave the distances between the molecules where they were.

