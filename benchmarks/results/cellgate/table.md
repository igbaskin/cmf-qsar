# Item BT9, gate (c): the capacity of the cell with the predicted diffusion coefficient

The salt diffusion coefficient of LiPF6 in EC:EMC 3:7 is predicted from the composition with every formulation of that mixture held out. The cell is the full model of item MS9. The truth is the capacity of the cell with all four transport parameters as Landesfeind and Gasteiger measured them. The reading of the gate was fixed before the cell was run (see the docstring of `benchmarks/run_cellgate.py`).

The model uses the kernel `gaussian, width 0.25` with the ridge 0.01, chosen on 59 formulations of other mixtures. The deviation of the label from the measurement, over the six concentrations of the series, is 0.54 in the natural logarithm.

| Concentration, mol/L | Molality, mol/kg | Predicted salt diffusion coefficient, m2/s | Measured, m2/s | Ratio | Standard deviation of the logarithm |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.09 | 1.64e-10 | 5.07e-10 | 0.32 | 0.083 |
| 0.25 | 0.24 | 2.35e-10 | 4.62e-10 | 0.51 | 0.080 |
| 0.50 | 0.48 | 2.99e-10 | 3.95e-10 | 0.76 | 0.077 |
| 0.75 | 0.73 | 3.14e-10 | 3.38e-10 | 0.93 | 0.076 |
| 1.00 | 0.99 | 3.04e-10 | 2.89e-10 | 1.05 | 0.075 |
| 1.25 | 1.25 | 2.80e-10 | 2.47e-10 | 1.13 | 0.075 |
| 1.50 | 1.52 | 2.52e-10 | 2.12e-10 | 1.19 | 0.076 |
| 2.00 | 2.09 | 1.95e-10 | 1.55e-10 | 1.26 | 0.080 |
| 2.50 | 2.73 | 1.50e-10 | 1.14e-10 | 1.32 | 0.098 |
| 3.00 | 3.32 | 1.31e-10 | 8.31e-11 | 1.58 | 0.136 |

Capacity of the last discharge in A.h. Reading A uses the standard deviation of the model alone, reading B adds the deviation of the label.

| Temperature, K | Rate, C | Published cell | Truth | Median of the prediction | Interval A | A holds the truth | A useful | Interval B | B holds the truth |
|---:|---:|---:|---:|---:|---|---|---|---|---|
| 298 | 1 | 4.991 | 4.987 | 4.988 | 4.985 to 4.990 | yes | yes | 4.484 to 4.997 | yes |
| 298 | 1.5 | 4.905 | 4.899 | 4.901 | 4.894 to 4.907 | yes | yes | 2.220 to 4.922 | yes |
| 298 | 2 | 4.783 | 4.789 | 4.793 | 4.758 to 4.808 | yes | yes | 1.104 to 4.841 | yes |
| 298 | 2.5 | 3.675 | 4.609 | 4.535 | 3.853 to 4.662 | yes | no | 0.519 to 4.746 | yes |
| 263 | 1 | 4.960 | 4.755 | 4.670 | 4.031 to 4.862 | yes | no | 0.696 to 4.947 | yes |
| 263 | 1.5 | 4.863 | 3.512 | 2.652 | 2.059 to 3.360 | no | no | 0.258 to 4.834 | yes |
| 253 | 1 | 4.949 | 3.788 | 3.283 | 2.611 to 3.877 | yes | no | 0.352 to 4.912 | yes |
| 253 | 1.5 | 4.848 | 2.409 | 1.650 | 1.218 to 2.140 | no | no | 0.205 to 4.638 | yes |

**Gate (c).** The interval of reading A holds the truth under 6 of the 8 conditions, and the interval of reading B under 8. The gate does not hold. The interval of reading A is useful (narrower than 5 % of the published capacity) under 3 of 8. The median of the prediction is at most 17.7 % of the published capacity away from the truth.

