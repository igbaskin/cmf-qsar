## COMPAS-1D (B3LYP-D3BJ/def2-SVP): the HOMO-LUMO gap, 8678 molecules

Charges: GFN2-xTB Mulliken charges (cmfqsar.xtb, no Fukui run, input geometry).  Target: GAP_eV (eV).  Random splits 80/10/10, 5 seeds (mean +- standard deviation); size-extrapolation splits train on fewer than N rings and test on N rings.

### Field sets

| field set | fields | q2 (5-fold CV, train) | test R2 | test RMSE (eV) | test MAE (eV) | N=8 R2 (train 179, test 440) | N=8 RMSE | N=9 R2 (train 619, test 1651) | N=9 RMSE | N=10 R2 (train 2270, test 6408) | N=10 RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| {A} | steric | 0.433 +- 0.011 | 0.418 +- 0.041 | 0.303 +- 0.017 | 0.235 +- 0.004 | -0.483 | 0.512 | -0.399 | 0.473 | -1.467 | 0.610 |
| {A,B} | steric, prop:q_xtb | 0.524 +- 0.009 | 0.513 +- 0.031 | 0.277 +- 0.013 | 0.215 +- 0.008 | -0.692 | 0.547 | -0.401 | 0.473 | -0.995 | 0.549 |
| {C} | cif:C.H, cif:C.2r | 0.493 +- 0.002 | 0.486 +- 0.015 | 0.285 +- 0.005 | 0.224 +- 0.003 | -0.019 | 0.425 | -0.488 | 0.488 | -1.149 | 0.569 |
| {D} | ring:terminal, ring:linear, ring:angular, ring:branched | 0.664 +- 0.007 | 0.646 +- 0.023 | 0.236 +- 0.011 | 0.182 +- 0.003 | 0.517 | 0.292 | 0.512 | 0.279 | -0.035 | 0.395 |
| {A,B,C} | steric, prop:q_xtb, cif:C.H, cif:C.2r | 0.603 +- 0.021 | 0.595 +- 0.045 | 0.253 +- 0.014 | 0.197 +- 0.011 | -0.045 | 0.430 | -0.123 | 0.424 | -1.070 | 0.559 |
| {A,B,C,D} | steric, prop:q_xtb, cif:C.H, cif:C.2r, ring:terminal, ring:linear, ring:angular, ring:branched | 0.825 +- 0.002 | 0.780 +- 0.105 | 0.184 +- 0.044 | 0.130 +- 0.002 | 0.591 | 0.269 | 0.531 | 0.274 | 0.441 | 0.290 |
| {C,D} | cif:C.H, cif:C.2r, ring:terminal, ring:linear, ring:angular, ring:branched | 0.753 +- 0.006 | 0.713 +- 0.081 | 0.212 +- 0.032 | 0.159 +- 0.002 | 0.501 | 0.297 | 0.503 | 0.282 | 0.305 | 0.324 |
| {A,B,C,D,E} | steric, prop:q_xtb, cif:C.H, cif:C.2r, ring:terminal, ring:linear, ring:angular, ring:branched, ring:sextet | 0.874 +- 0.002 | 0.869 +- 0.018 | 0.144 +- 0.012 | 0.108 +- 0.002 | 0.672 | 0.241 | 0.780 | 0.187 | 0.700 | 0.213 |

Test RMSE (eV) over the molecules of three rings and more, random splits: {A,B,C,D} 0.164 +- 0.003; {A,B,C,D,E} 0.138 +- 0.002.  The one- and two-ring molecules in a random test part (random0: hc_c6h6_0pent_1) are predicted by extrapolation from a training part that holds at most one molecule of their size.

### Attenuations and ridge chosen for {A,B,C,D}

| split | choice | alpha per group (A^-2) | lambda | q2 | valid R2 | test R2 |
|---|---|---|---|---|---|---|
| random0 | per-group | A: 0.5, B: 0.1, C: 0.1, D: 0.1 | 0.0001 | 0.829 | 0.832 | 0.593 |
| random1 | shared | A: 0.1, B: 0.1, C: 0.1, D: 0.1 | 0.0001 | 0.823 | 0.794 | 0.833 |
| random2 | shared | A: 0.1, B: 0.1, C: 0.1, D: 0.1 | 0.0001 | 0.826 | 0.771 | 0.820 |
| random3 | per-group | A: 0.5, B: 0.1, C: 0.2, D: 0.1 | 0.0003 | 0.823 | 0.839 | 0.832 |
| random4 | per-group | A: 0.5, B: 0.1, C: 0.1, D: 0.1 | 0.0003 | 0.824 | 0.821 | 0.821 |
| size8 | per-group | A: 0.5, B: 0.2, C: 0.5, D: 0.1 | 0.01 | 0.799 | - | 0.591 |
| size9 | per-group | A: 1, B: 0.1, C: 0.5, D: 0.1 | 0.01 | 0.808 | - | 0.531 |
| size10 | per-group | A: 0.5, B: 0.1, C: 0.2, D: 0.1 | 0.0001 | 0.835 | - | 0.441 |

### Baselines on the same splits

| model | test R2 | test RMSE (eV) | test MAE (eV) | N=8 R2 | N=8 RMSE | N=9 R2 | N=9 RMSE | N=10 R2 | N=10 RMSE |
|---|---|---|---|---|---|---|---|---|---|
| ridge on atom-class and ring-class counts (additivity scheme) | 0.699 +- 0.013 | 0.218 +- 0.008 | 0.166 +- 0.003 | 0.626 | 0.257 | 0.646 | 0.238 | 0.668 | 0.224 |
| straight line in the number of rings | 0.021 +- 0.029 | 0.393 +- 0.005 | 0.312 +- 0.006 | -0.184 | 0.458 | -0.076 | 0.415 | -0.022 | 0.393 |
| random forest on 66 RDKit descriptors | 0.913 +- 0.024 | 0.117 +- 0.017 | 0.074 +- 0.003 | 0.319 | 0.347 | 0.347 | 0.323 | -0.160 | 0.418 |

### Published models of the gap on the same molecules (random splits; no size-extrapolation numbers are published)

| model | source | protocol | test MAE (eV) | test RMSE (eV) | test R2 |
|---|---|---|---|---|---|
| SE(3)-transformer on a graph of rings (SE(3)-GOR) | Weiss, Wahab, Bronstein, Gershoni-Poranne, J. Org. Chem. 2023, 88, 9645, Table 1 | COMPAS-1D, 8,678 molecules, random 81/9/10, seed 42 | 0.012 +- 0.002 | - | - |
| SE(3)-transformer on a graph of atoms (SE(3)-GOA) | same | same | 0.011 +- 0.004 | - | - |
| SphereNet / DimeNet++ / SchNet (geometric graph networks) | same | same | 0.013 / 0.015 / 0.022 | - | - |
| CHEM-BERT (SMILES language model) / AttentiveFP (topological graph) | same | same | 0.033 / 0.073 | - | - |
| RNN on one-hot augmented LALA annulation strings | Fite, Wahab, Paenurk, Gross, Gershoni-Poranne, J. Phys. Org. Chem. 2023, 36, e4458, Fig. 3 | COMPAS-1D, 7,674 training molecules (90 %) | 0.05 (all model/representation pairs: 0.04 to 0.26) | - | - |
| extra-trees / random forest on selected RDKit descriptors | Zheng, Zhu, Lu, Yan, Jiang, Sun, Chem. Phys. Lett. 2023, 814, 140358 | COMPAS-1D without benzene and naphthalene (8,676), random 90/10, 100 repeats | - | about 0.12 (MSE about 0.014) | about 0.91 |

The random forest of this study (0.074 eV MAE, 0.117 eV RMSE, R2 0.91) reproduces the last row.  The field models of this study (MAE 0.13 eV for {A,B,C,D}, 0.11 eV with the sextet field) sit at the lower end of the annulation-string models and an order of magnitude above the geometric graph networks on these random splits.

### Resolution: a wider attenuation grid, a smaller ridge and the Gaussian kernel layer

Attenuation grid (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0, 20.0), ridge grid down to 1e-06, Gaussian layer widths (0.25, 0.5, 1.0, 2.0, 4.0, 8.0) (relative to the median squared distance); random splits, mean +- sd over the seeds.

| field set | configuration | alpha per group (A^-2), gamma | lambda | q2 | test R2 | test RMSE (eV) | test MAE (eV) |
|---|---|---|---|---|---|---|---|
| {A,B,C,D,E} | coarse grid, as in the main table | - | - | 0.874 +- 0.002 | 0.869 +- 0.018 | 0.144 +- 0.012 | 0.108 +- 0.002 |
| {A,B,C,D,E} | shared (first seed's choice shown) | A: 0.05, B: 0.05, C: 0.05, D: 0.05, E: 0.05 | 0.0001 | 0.867 +- 0.004 | 0.868 +- 0.012 | 0.147 +- 0.009 | 0.110 +- 0.002 |
| {A,B,C,D,E} | per-group (first seed's choice shown) | A: 0.2, B: 0.05, C: 0.05, D: 0.02, E: 0.1 | 0.0001 | 0.868 +- 0.005 | 0.831 +- 0.071 | 0.164 +- 0.037 | 0.112 +- 0.004 |
| {A,B,C,D,E} | gauss (first seed's choice shown) | A: 0.2, B: 0.05, C: 0.05, D: 0.02, E: 0.1; gamma 0.25 | 0.0001/0.001/3e-05 | 0.961 +- 0.004 | 0.964 +- 0.005 | 0.076 +- 0.006 | 0.056 +- 0.004 |
| {A,B,C,D} | coarse grid, as in the main table | - | - | 0.825 +- 0.002 | 0.780 +- 0.105 | 0.184 +- 0.044 | 0.130 +- 0.002 |
| {A,B,C,D} | shared (first seed's choice shown) | A: 0.1, B: 0.1, C: 0.1, D: 0.1 | 0.0001 | 0.824 | 0.770 | 0.198 | 0.132 |
| {A,B,C,D} | per-group (first seed's choice shown) | A: 0.35, B: 0.05, C: 0.1, D: 0.1 | 0.0001 | 0.825 | 0.778 | 0.194 | 0.131 |
| {A,B,C,D} | gauss (first seed's choice shown) | A: 0.35, B: 0.05, C: 0.1, D: 0.1; gamma 0.25 | 0.0001 | 0.937 | 0.923 | 0.115 | 0.073 |

### Nystroem approximation (landmark molecules)

| set | molecules | m landmarks | rank | q2 | test R2 | test RMSE (eV) | fit time (s) |
|---|---|---|---|---|---|---|---|
| COMPAS-1x | 34072 | 500 | 2244 | 0.782 | 0.803 | 0.133 | 902 |
| COMPAS-1x | 34072 | 1000 | 2570 | 0.782 | 0.803 | 0.133 | 1296 |
| COMPAS-1x | 34072 | 2000 | 2722 | 0.782 | 0.803 | 0.133 | 3319 |

### Interpretation

Reference molecule hc_c26h16_0pent_13 (6 rings, gap 3.43 eV); co-field figure `benchmarks/results/compas/figures/cofield_gap_hc_c26h16_0pent_13.png`; per-ring contributions `benchmarks/results/compas/figures/ring_contributions_gap.png`.  The CMFModel refitted in float64 for the figures reproduces the table's test predictions (float32 kernels) to 5.7e-02 eV at worst; its own test R2 is 0.593, RMSE 0.263 eV.

