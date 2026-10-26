## Protonation ensembles on the corrosion series in HCl

A protonation ensemble is the same mathematics as a conformational ensemble: the mean field of the states, weighted by their pKa populations. `file` is the state the data file gives, `dominant` the likeliest state at this pH by pKa, `ensemble` the weighted mean over the states, and `tautomer` the mean over the tautomers of the file's state. Same five fields, same attenuation and ridge grids, same ten-fold cross-validation, same scaffold alignment as `run_corrosion.py`.

### Where an ensemble can make a difference

Share of the molecules of a series that is a real mixture (a second state above 1% of the molecule), and how often the likeliest state carries the charge the file gives it. No model is fitted for this table.

| pH | mixtures | likeliest state agrees with the file | mean charge by pKa | mean charge in the file |
| --- | --- | --- | --- | --- |
| 0 | 1% | 52% | +1.57 | +0.77 |
| 1 | 1% | 52% | +1.57 | +0.77 |
| 2 | 18% | 53% | +1.55 | +0.77 |
| 3 | 83% | 53% | +1.47 | +0.77 |
| 4 | 83% | 50% | +1.09 | +0.77 |
| 5 | 83% | 31% | +0.31 | +0.77 |
| 7.4 | 27% | 24% | -0.11 | +0.77 |

23 series, 997 molecules in all.

### The models at pH 0

| set | n | mixtures | state agrees | several tautomers | q2 file | q2 dominant | q2 ensemble | q2 tautomer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0_c1ccccc1 | 244 | 2% | 93% | 1% | **+0.413** | +0.391 | +0.391 | +0.383 |
| 1_c1ccncc1 | 109 | 2% | 61% | 1% | **+0.341** | +0.295 | +0.295 | +0.326 |
| 2_c1c_nH_cn1 | 96 | 0% | 60% | 7% | +0.502 | +0.483 | +0.483 | **+0.503** |
| 157_c1ccc2_nH_cnc2c1 | 73 | 0% | 58% | 3% | +0.346 | **+0.424** | +0.424 | +0.341 |
| 158_c1ccc2ncccc2c1 | 65 | 3% | 51% | 2% | **+0.149** | +0.042 | +0.042 | +0.128 |
| 4_c1nnc_nH_1 | 40 | 0% | 0% | 5% | +0.188 | **+0.228** | +0.228 | +0.191 |
| 8_c1cn_nH_c1 | 30 | 3% | 23% | 10% | +0.723 | +0.649 | +0.649 | **+0.739** |
| 7_c1nncs1 | 29 | 0% | 0% | 0% | **+0.279** | +0.105 | +0.105 | +0.260 |
| 12_c1ccoc1 | 28 | 0% | 96% | 0% | +0.658 | +0.406 | +0.406 | **+0.661** |
| 159_C_Nc1ccccc1_c1ccccc1 | 28 | 0% | 82% | 0% | **+0.261** | +0.136 | +0.136 | +0.236 |
| 3_c1cscn1 | 27 | 0% | 44% | 0% | **+0.465** | +0.428 | +0.428 | +0.434 |
| 13_c1c_nH_nn1 | 25 | 0% | 0% | 0% | **+0.519** | +0.481 | +0.481 | +0.515 |
| 9_c1nc_nH_n1 | 25 | 0% | 0% | 48% | +0.090 | **+0.147** | +0.147 | +0.106 |
| 14_O_c1cc_nH_c_O_nH_1 | 24 | 4% | 71% | 0% | +0.711 | +0.612 | +0.612 | **+0.712** |
| 11_c1c_nH_c_nH_1 | 23 | 0% | 100% | 0% | **+0.320** | +0.312 | +0.312 | +0.316 |
| 161_c1ccc2scnc2c1 | 20 | 0% | 40% | 0% | **+0.410** | +0.322 | +0.322 | +0.402 |
| 17_c1nnco1 | 17 | 0% | 0% | 0% | **+0.497** | -0.062 | -0.062 | +0.447 |
| 21_O_C1CCN1 | 17 | 0% | 59% | 35% | **+0.399** | +0.305 | +0.305 | +0.336 |
| 22_C1CNOC1 | 16 | 0% | 75% | 0% | +0.398 | +0.341 | +0.341 | **+0.399** |
| 5_c1cncnc1 | 16 | 19% | 0% | 6% | -0.023 | +0.145 | **+0.169** | -0.025 |
| 160_c1ccc2ccccc2c1 | 15 | 0% | 93% | 0% | -0.125 | -0.133 | -0.133 | **-0.120** |
| 165_O_c1cc_nH_n1_c1ccccc1 | 15 | 0% | 100% | 0% | **-0.028** | -0.031 | -0.031 | -0.032 |
| 20_O_c1cc_nH_nH_1 | 15 | 0% | 100% | 0% | **-0.045** | -0.046 | -0.046 | -0.046 |
| **mean** | 997 | 1% | 52% | 5% | **+0.324** | +0.260 | +0.261 | +0.314 |

Best on 13 of 23 series for `file`, 3 of 23 series for `dominant`, 1 of 23 series for `ensemble`, 6 of 23 series for `tautomer`.

Mean difference: `dominant` - `file` -0.064, `ensemble` - `dominant` +0.001, `ensemble` - `file` -0.063, `tautomer` - `file` -0.010.

### The models at pH 3

| set | n | mixtures | state agrees | several tautomers | q2 file | q2 dominant | q2 ensemble | q2 tautomer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0_c1ccccc1 | 244 | 41% | 91% | 1% | +0.413 | +0.393 | **+0.452** | +0.383 |
| 1_c1ccncc1 | 109 | 100% | 62% | 1% | **+0.341** | +0.293 | +0.294 | +0.326 |
| 2_c1c_nH_cn1 | 96 | 100% | 60% | 7% | +0.502 | +0.483 | +0.471 | **+0.503** |
| 157_c1ccc2_nH_cnc2c1 | 73 | 100% | 58% | 3% | +0.346 | **+0.424** | +0.422 | +0.341 |
| 158_c1ccc2ncccc2c1 | 65 | 100% | 52% | 2% | **+0.149** | +0.045 | +0.048 | +0.128 |
| 4_c1nnc_nH_1 | 40 | 100% | 0% | 5% | +0.188 | **+0.228** | +0.205 | +0.191 |
| 8_c1cn_nH_c1 | 30 | 100% | 23% | 10% | +0.723 | +0.649 | +0.649 | **+0.739** |
| 7_c1nncs1 | 29 | 100% | 0% | 0% | **+0.279** | +0.105 | +0.111 | +0.260 |
| 12_c1ccoc1 | 28 | 25% | 96% | 0% | +0.658 | +0.406 | +0.255 | **+0.661** |
| 159_C_Nc1ccccc1_c1ccccc1 | 28 | 32% | 82% | 0% | **+0.261** | +0.136 | +0.137 | +0.236 |
| 3_c1cscn1 | 27 | 100% | 44% | 0% | **+0.465** | +0.428 | +0.428 | +0.434 |
| 13_c1c_nH_nn1 | 25 | 100% | 0% | 0% | **+0.519** | +0.481 | +0.437 | +0.515 |
| 9_c1nc_nH_n1 | 25 | 100% | 0% | 48% | +0.090 | +0.147 | **+0.167** | +0.106 |
| 14_O_c1cc_nH_c_O_nH_1 | 24 | 75% | 71% | 0% | +0.711 | +0.582 | +0.584 | **+0.712** |
| 11_c1c_nH_c_nH_1 | 23 | 100% | 100% | 0% | **+0.320** | +0.312 | +0.313 | +0.316 |
| 161_c1ccc2scnc2c1 | 20 | 100% | 40% | 0% | **+0.410** | +0.322 | +0.312 | +0.402 |
| 17_c1nnco1 | 17 | 100% | 0% | 0% | **+0.497** | -0.062 | -0.060 | +0.447 |
| 21_O_C1CCN1 | 17 | 100% | 59% | 35% | **+0.399** | +0.305 | +0.287 | +0.336 |
| 22_C1CNOC1 | 16 | 0% | 75% | 0% | +0.398 | +0.341 | +0.341 | **+0.399** |
| 5_c1cncnc1 | 16 | 100% | 6% | 6% | -0.023 | **+0.063** | +0.012 | -0.025 |
| 160_c1ccc2ccccc2c1 | 15 | 47% | 93% | 0% | -0.125 | -0.133 | -0.133 | **-0.120** |
| 165_O_c1cc_nH_n1_c1ccccc1 | 15 | 100% | 100% | 0% | **-0.028** | -0.031 | -0.030 | -0.032 |
| 20_O_c1cc_nH_nH_1 | 15 | 100% | 100% | 0% | -0.045 | -0.046 | **-0.045** | -0.046 |
| **mean** | 997 | 83% | 53% | 5% | **+0.324** | +0.255 | +0.246 | +0.314 |

Best on 11 of 23 series for `file`, 3 of 23 series for `dominant`, 3 of 23 series for `ensemble`, 6 of 23 series for `tautomer`.

Mean difference: `dominant` - `file` -0.069, `ensemble` - `dominant` -0.009, `ensemble` - `file` -0.078, `tautomer` - `file` -0.010.

Three limits of the weights bear on these numbers. Each ionisable site is treated as independent, which overestimates multiple ionisation: a second proton is harder to add than its own pKa says, so the mean charge by pKa is an upper bound. The pKa values are Dimorphite-DL's regression values for these groups, not measurements of these molecules. And the tautomer weights are not populations at all: the forms RDKit scores highest are counted the same, because no free energies are computed here.

