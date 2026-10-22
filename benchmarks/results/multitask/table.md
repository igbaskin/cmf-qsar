# Two endpoints on one kernel

One row per series and pair of endpoints. *n* is molecules with the first / the second measurement, *r* their agreement where both exist; q² per endpoint with the endpoints fitted separately (ρ = 0) and with the task correlation ρ chosen by cross-validation over molecules; and the q² of the predicted difference on the molecules that carry both, separate and shared.

| series | endpoints | n | r | q² separate | ρ | q² shared | difference q² (separate → shared) |
|---|---|---|---|---|---|---|---|
| 0_c1ccccc1 | activity / dG_ads | 244 / 207 | -0.59 | +0.39 / +0.35 | +0 | +0.39 / +0.35 | +0.38 → +0.38 (n 207) |
| 0_c1ccccc1 | mwila_hcl_adj / mwila_h2so4_adj | 244 / 40 | +0.93 | +0.06 / +0.44 | +0 | +0.06 / +0.44 | -1.32 → -1.32 (n 40) |
| 11_c1c_nH_c_nH_1 | activity / dG_ads | 23 / 22 | -0.94 | +0.25 / +0.30 | +0.9 | +0.34 / +0.26 | +0.29 → +0.28 (n 22) |
| 12_c1ccoc1 | activity / dG_ads | 28 / 12 | -0.98 | +0.06 / -0.03 | -0.4 | +0.07 / -0.03 | -0.05 → -0.07 (n 12) |
| 13_c1c_nH_nn1 | activity / dG_ads | 25 / 23 | +0.44 | +0.63 / +0.61 | +0.6 | +0.65 / +0.61 | +0.50 → +0.49 (n 23) |
| 161_c1ccc2scnc2c1 | activity / dG_ads | 20 / 16 | -0.99 | +0.69 / +0.74 | -0.95 | +0.73 / +0.76 | +0.76 → +0.77 (n 16) |
| 2_c1c_nH_cn1 | activity / dG_ads | 96 / 72 | -0.78 | +0.49 / +0.33 | -0.6 | +0.53 / +0.33 | +0.41 → +0.42 (n 72) |
