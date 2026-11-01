# How nonlinear is the Gaussian layer, and where

`benchmarks/run_nonlinearity.py` on the Hammett set (one conformer per substituent, Gasteiger charges, five fields). The non-linearity index of a field is the share of the squared norm of the molecule-dependent co-fields that their spread over the training molecules makes up: 0 for a linear model. `sigma_p*pi` is a target made non-additive on purpose. Stability is the mean correlation between the maps of five refits on random four-fifths of the set.

| target | n | q² linear | q² gauss | γ | index joint | index electrostatic | index steric | index hydrophobic | mean cosine joint | spread map stability | mean map stability | index of the refits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sigma_m | 95 | 0.70 | 0.80 | 1 | 0.199 | 0.178 | 0.305 | 0.188 | 0.847 | 1.00 | 0.99 | 0.201 |
| sigma_p | 95 | 0.66 | 0.78 | 0.5 | 0.100 | 0.083 | 0.199 | 0.106 | 0.936 | 0.99 | 0.99 | 0.100 |
| pi | 69 | 0.72 | 0.88 | 1 | 0.316 | 0.412 | 0.228 | 0.265 | 0.706 | 0.87 | 0.54 | 0.313 |
| sigma_p*pi | 69 | 0.19 | 0.44 | 1 | 0.413 | 0.416 | 0.451 | 0.409 | 0.566 | 0.96 | 0.80 | 0.410 |

The figure `maps.png` cuts the ring plane: the linear co-field, the mean of the local co-fields and their spread, for the electrostatic field.
