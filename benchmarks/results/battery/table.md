# The average voltage of an insertion electrode, from the cell

The pairs are reconstructed from public formation energies (`cmfqsar/battery.py`), keeping the phases within 0.03 eV per atom of the convex hull. Four textbook cathodes, against the voltages the literature gives, over the first insertion (up to 1 working ion per host formula unit):

| cathode | published, V | reconstructed, V | steps of the ladder |
|---|---:|---:|---:|
| LiCoO2, layered | 4.10 | 3.77 | 5 |
| LiFePO4, olivine | 3.45 | 3.33 | 3 |
| LiMn2O4, spinel | 4.00 | 3.23 | 13 |
| LiNiO2, layered | 3.80 | 3.47 | 10 |

Mean absolute difference from the published values: **0.39 V**, and the reconstruction is lower than the published value every time. Two reasons: the energies are the published GGA/GGA+U ones without the corrections a battery database applies, and the average here runs over the intermediate phases of the ladder, which a two-phase measurement does not see.

How the set and the agreement move with the hull window, which is what fixes it at 0.03:

| hull window, eV/atom | pairs | frameworks | published off by, V |
|---|---:|---:|---:|
| 0.01 | 83 | 51 | 0.39 |
| 0.02 | 197 | 112 | 0.39 |
| 0.03 | 303 | 157 | 0.39 |
| 0.05 | 528 | 276 | 0.42 |
| 0.08 | 789 | 391 | 0.57 |

303 insertion pairs over 157 host frameworks, five folds grouped by framework, 23 fields. Predicting the mean voltage of the set would give a mean absolute error of 1.203 V.

| model | MAE, V | spread over the folds | R2 | pooled RMSE, V |
|---|---:|---:|---:|---:|
| composition | 0.569 | 0.115 | +0.728 | 0.766 |
| fields | 0.650 | 0.146 | +0.624 | 0.900 |
| fields+composition | 0.582 | 0.117 | +0.684 | 0.824 |

By family of hosts, for the model that uses both:

| family | pairs | MAE, V | R2 | mean voltage, V |
|---|---:|---:|---:|---:|
| fluoride | 21 | 0.678 | +0.093 | 4.23 |
| other | 51 | 0.603 | -6.281 | 0.34 |
| oxide | 105 | 0.664 | +0.162 | 2.83 |
| phosphate | 113 | 0.509 | +0.321 | 3.61 |
| silicate | 3 | 0.178 | -0.059 | 2.81 |
| sulfide | 10 | 0.466 | -3.488 | 1.86 |

