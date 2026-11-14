# How nonlinear is the Gaussian layer, and where

`benchmarks/run_nonlinearity.py` on the Hammett set (one conformer per substituent, Gasteiger charges, five fields). The non-linearity index of a field is the share of the squared norm of the molecule-dependent co-fields that their spread over the training molecules makes up: 0 for a linear model. `sigma_p*pi` is a target made non-additive on purpose. Stability is the mean correlation between the maps of five refits on random four-fifths of the set.

| target | n | q² linear | q² gauss | γ | index joint | index electrostatic | index steric | index hydrophobic | mean cosine joint | spread map stability | mean map stability | index of the refits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sigma_m | 95 | 0.69 | 0.80 | 1 | 0.203 | 0.180 | 0.271 | 0.196 | 0.842 | 1.00 | 0.99 | 0.204 |
| sigma_p | 95 | 0.67 | 0.79 | 0.5 | 0.102 | 0.084 | 0.176 | 0.115 | 0.934 | 0.99 | 0.99 | 0.102 |
| pi | 69 | 0.75 | 0.88 | 1 | 0.317 | 0.421 | 0.232 | 0.269 | 0.706 | 0.85 | 0.49 | 0.312 |
| sigma_p*pi | 69 | 0.17 | 0.46 | 1 | 0.408 | 0.407 | 0.451 | 0.405 | 0.572 | 0.97 | 0.83 | 0.407 |

The figure `maps.png` cuts the ring plane: the linear co-field, the mean of the local co-fields and their spread, for the electrostatic field.
