## COMPAS-1D (B3LYP-D3BJ/def2-SVP): the HOMO-LUMO gap, 8678 molecules

Charges: GFN2-xTB Mulliken charges (cmfqsar.xtb, no Fukui run, input geometry).  Target: GAP_eV (eV).  Random splits 80/10/10, 5 seeds (mean +- standard deviation); size-extrapolation splits train on fewer than N rings and test on N rings.

### Field sets

Field group F is the ring field weighted by the HOMA index of every ring (from the bond lengths of the DFT geometry).  A row '+ Gaussian layer' is the same kernel under exp(-gamma d^2) (the resolution section explains it); the sets added later were run on three seeds.

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
| {A,B,C,D,F} (3 seeds) | steric, prop:q_xtb, cif:C.H, cif:C.2r, ring:terminal, ring:linear, ring:angular, ring:branched, ring:homa | 0.903 +- 0.001 | 0.844 +- 0.120 | 0.153 +- 0.061 | 0.097 +- 0.006 | 0.716 | 0.224 | 0.780 | 0.187 | 0.825 | 0.163 |
| {A,B,C,D,F} + Gaussian layer (3 seeds) | the same kernel, exp(-gamma d^2) | 0.957 +- 0.000 | 0.935 +- 0.041 | 0.100 +- 0.033 | 0.062 +- 0.002 | 0.771 | 0.201 | 0.831 | 0.164 | 0.879 | 0.135 |
| {A,B,C,D,E,F} (3 seeds) | steric, prop:q_xtb, cif:C.H, cif:C.2r, ring:terminal, ring:linear, ring:angular, ring:branched, ring:sextet, ring:homa | 0.920 +- 0.001 | 0.879 +- 0.084 | 0.135 +- 0.049 | 0.085 +- 0.004 | 0.756 | 0.208 | 0.820 | 0.170 | 0.843 | 0.154 |
| {A,B,C,D,E,F} + Gaussian layer (3 seeds) | the same kernel, exp(-gamma d^2) | 0.971 +- 0.001 | 0.961 +- 0.022 | 0.077 +- 0.023 | 0.050 +- 0.003 | 0.808 | 0.184 | 0.890 | 0.133 | 0.927 | 0.105 |

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
| ridge on 10 counts of atom classes, ring classes and HOMA (additivity scheme) | 0.798 +- 0.015 | 0.179 +- 0.010 | 0.138 +- 0.003 | 0.750 | 0.210 | 0.745 | 0.202 | 0.766 | 0.188 |
| straight line in the number of rings | 0.021 +- 0.029 | 0.393 +- 0.005 | 0.312 +- 0.006 | -0.184 | 0.458 | -0.076 | 0.415 | -0.022 | 0.393 |
| random forest on 66 RDKit descriptors | 0.913 +- 0.024 | 0.117 +- 0.017 | 0.074 +- 0.003 | 0.319 | 0.347 | 0.347 | 0.323 | -0.160 | 0.418 |
| linear correction of the GFN2-xTB gap (slope and intercept fitted on the training part) | 0.986 +- 0.001 | 0.047 +- 0.001 | 0.036 +- 0.001 | 0.987 | 0.048 | 0.986 | 0.047 | 0.986 | 0.045 |

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

Size-extrapolation splits (train on fewer than N rings, test on N), the same three configurations against the coarse-grid linear model of the main table:

| field set | split | coarse linear R2 / RMSE | wide-grid linear R2 / RMSE | Gaussian layer R2 / RMSE | layer MAE (eV) | gamma |
|---|---|---|---|---|---|---|
| {A,B,C,D,E} | size8 | 0.672 / 0.241 | 0.705 / 0.228 | 0.803 / 0.187 | 0.145 | 0.25 |
| {A,B,C,D,E} | size9 | 0.780 / 0.187 | 0.778 / 0.188 | 0.891 / 0.132 | 0.100 | 0.25 |
| {A,B,C,D,E} | size10 | 0.700 / 0.213 | 0.702 / 0.212 | 0.916 / 0.113 | 0.086 | 0.25 |
| {A,B,C,D} | size8 | 0.591 / 0.269 | 0.589 / 0.270 | 0.691 / 0.234 | 0.179 | 0.25 |
| {A,B,C,D} | size9 | 0.531 / 0.274 | 0.570 / 0.262 | 0.809 / 0.175 | 0.132 | 0.25 |
| {A,B,C,D} | size10 | 0.441 / 0.290 | 0.441 / 0.290 | 0.840 / 0.155 | 0.118 | 0.25 |

### Nystroem approximation (landmark molecules)

| set | molecules | m landmarks | rank | q2 | test R2 | test RMSE (eV) | fit time (s) |
|---|---|---|---|---|---|---|---|
| COMPAS-1x | 34072 | 500 | 2244 | 0.782 | 0.803 | 0.133 | 902 |
| COMPAS-1x | 34072 | 1000 | 2570 | 0.782 | 0.803 | 0.133 | 1296 |
| COMPAS-1x | 34072 | 2000 | 2722 | 0.782 | 0.803 | 0.133 | 3319 |

### Interpretation

Reference molecule hc_c26h16_0pent_13 (6 rings, gap 3.43 eV); co-field figure `benchmarks/results/compas/figures/cofield_gap_hc_c26h16_0pent_13.png`; per-ring contributions `benchmarks/results/compas/figures/ring_contributions_gap.png`.  The CMFModel refitted in float64 for the figures reproduces the table's test predictions (float32 kernels) to 5.7e-02 eV at worst; its own test R2 is 0.593, RMSE 0.263 eV.

### Delta-learning: the DFT gap from the GFN2-xTB gap and the fields (COMPAS-1D)

Target: the DFT gap minus a straight line in the GFN2-xTB gap, the line fitted on the training part of each split (the xTB gaps from the x table of the same family).  The metrics are on the gap scale, since adding the line back changes no difference.  The first row is the line alone.

| field set | fields | q2 (5-fold CV, train) | test R2 | test RMSE (eV) | test MAE (eV) | N=8 R2 (train 179, test 440) | N=8 RMSE | N=9 R2 (train 619, test 1651) | N=9 RMSE | N=10 R2 (train 2270, test 6408) | N=10 RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| {A,B,C,D,E} (3 seeds) | steric, prop:q_xtb, cif:C.H, cif:C.2r, ring:terminal, ring:linear, ring:angular, ring:branched, ring:sextet | 0.463 +- 0.005 | 0.405 +- 0.169 | 0.035 +- 0.006 | 0.025 +- 0.001 | -0.285 | 0.053 | 0.260 | 0.040 | 0.071 | 0.043 |
| {A,B,C,D,E} + Gaussian layer (3 seeds) | the same kernel, exp(-gamma d^2) | 0.644 +- 0.006 | 0.684 +- 0.034 | 0.026 +- 0.002 | 0.018 +- 0.001 | 0.107 | 0.045 | 0.489 | 0.033 | 0.556 | 0.030 |

The line alone (every random seed): test R2 0.986 +- 0.001, RMSE 0.047 +- 0.001 eV, MAE 0.036 +- 0.001 eV; size splits: size8 R2 0.987, RMSE 0.048; size9 R2 0.986, RMSE 0.047; size10 R2 0.986, RMSE 0.045.

### NICS(1) against the HOMA index and the Clar-sextet share, ring by ring (COMPAS-1D)

ORCA, B3LYP D3BJ def2-SVP NMR TightSCF, one ghost atom 1 A above every ring centroid, 179 molecules of up to 7 rings (1158 rings; 532 min with 10 serial jobs).  Correlation coefficients over the rings: HOMA against NICS(1)zz -0.769, HOMA against NICS(1) -0.749, Clar share against NICS(1)zz -0.495, Clar share against HOMA 0.817.

| ring class | rings | mean NICS(1)zz (ppm) | mean HOMA |
|---|---|---|---|
| angular | 431 | -18.3 | 0.456 |
| branched | 73 | -8.4 | 0.084 |
| linear | 224 | -30.1 | 0.682 |
| terminal | 430 | -25.3 | 0.786 |

## COMPAS-3D (CAM-B3LYP-D3BJ/def2-SVP geometries (xyz headers), energies as in the COMPAS-3 paper): the peri-condensed polybenzenoid hydrocarbons of COMPAS-3D, 8844 molecules

Charges: GFN2-xTB Mulliken charges (cmfqsar.xtb, no Fukui run, input geometry).  Rings by annulation class: angular 14309, branched 1853, linear 6629, peri 44187, terminal 18982.  Largest heavy-atom distance from the molecular plane: median 1.21 A, 95th percentile 2.41 A, maximum 3.36 A.

### Field sets (three random seeds and the size-extrapolation splits)

| field set | fields | q2 (5-fold CV, train) | test R2 | test RMSE (eV) | test MAE (eV) | N=8 R2 (train 87, test 329) | N=8 RMSE | N=9 R2 (train 416, test 1537) | N=9 RMSE | N=10 R2 (train 1953, test 6891) | N=10 RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| {D} (3 seeds) | ring:terminal, ring:linear, ring:angular, ring:branched, ring:peri | 0.436 +- 0.005 | 0.459 +- 0.012 | 0.328 +- 0.011 | 0.256 +- 0.009 | -0.655 | 0.656 | -0.058 | 0.483 | -0.151 | 0.457 |
| {D} + Gaussian layer (3 seeds) | the same kernel, exp(-gamma d^2) | 0.606 +- 0.008 | 0.634 +- 0.020 | 0.270 +- 0.003 | 0.204 +- 0.004 | 0.242 | 0.444 | 0.251 | 0.407 | 0.411 | 0.327 |
| {A,B,C,D} (3 seeds) | steric, prop:q_xtb, cif:C.H, cif:C.2r, cif:C.3r, ring:terminal, ring:linear, ring:angular, ring:branched, ring:peri | 0.584 +- 0.007 | 0.610 +- 0.011 | 0.279 +- 0.009 | 0.217 +- 0.006 | 0.131 | 0.475 | 0.321 | 0.387 | 0.218 | 0.377 |
| {A,B,C,D} + Gaussian layer (3 seeds) | the same kernel, exp(-gamma d^2) | 0.814 +- 0.006 | 0.837 +- 0.017 | 0.180 +- 0.015 | 0.133 +- 0.008 | 0.442 | 0.381 | 0.570 | 0.308 | 0.578 | 0.277 |
| {A,B,C,D,E} (3 seeds) | steric, prop:q_xtb, cif:C.H, cif:C.2r, cif:C.3r, ring:terminal, ring:linear, ring:angular, ring:branched, ring:peri, ring:sextet | 0.809 +- 0.003 | 0.827 +- 0.009 | 0.186 +- 0.003 | 0.147 +- 0.003 | 0.729 | 0.266 | 0.711 | 0.253 | 0.700 | 0.234 |
| {A,B,C,D,E} + Gaussian layer (3 seeds) | the same kernel, exp(-gamma d^2) | 0.931 +- 0.002 | 0.937 +- 0.001 | 0.112 +- 0.003 | 0.083 +- 0.002 | 0.698 | 0.280 | 0.811 | 0.204 | 0.821 | 0.180 |
| {A,B,C,D,E,F} (3 seeds) | steric, prop:q_xtb, cif:C.H, cif:C.2r, cif:C.3r, ring:terminal, ring:linear, ring:angular, ring:branched, ring:peri, ring:sextet, ring:homa | 0.884 +- 0.001 | 0.896 +- 0.002 | 0.144 +- 0.004 | 0.113 +- 0.004 | 0.617 | 0.316 | 0.784 | 0.219 | 0.795 | 0.193 |
| {A,B,C,D,E,F} + Gaussian layer (3 seeds) | the same kernel, exp(-gamma d^2) | 0.945 +- 0.002 | 0.953 +- 0.003 | 0.097 +- 0.005 | 0.072 +- 0.003 | 0.751 | 0.255 | 0.834 | 0.191 | 0.861 | 0.159 |

### Baselines on the same splits

| model | test R2 | test RMSE (eV) | test MAE (eV) | N=8 R2 | N=8 RMSE | N=9 R2 | N=9 RMSE | N=10 R2 | N=10 RMSE |
|---|---|---|---|---|---|---|---|---|---|
| ridge on 11 counts of atom classes, ring classes and HOMA (additivity scheme) | 0.790 +- 0.012 | 0.203 +- 0.002 | 0.163 +- 0.001 | 0.836 | 0.206 | 0.791 | 0.215 | 0.752 | 0.212 |
| straight line in the number of rings | 0.004 +- 0.005 | 0.443 +- 0.011 | 0.365 +- 0.009 | -0.072 | 0.528 | -0.047 | 0.481 | -0.037 | 0.434 |
| random forest on 65 RDKit descriptors | 0.836 +- 0.006 | 0.180 +- 0.007 | 0.129 +- 0.004 | -0.536 | 0.632 | -1.835 | 0.791 | -2.263 | 0.770 |
| linear correction of the GFN2-xTB gap (slope and intercept fitted on the training part) | 0.984 +- 0.001 | 0.057 +- 0.001 | 0.044 +- 0.001 | 0.986 | 0.060 | 0.985 | 0.057 | 0.982 | 0.057 |

### Interpretation

Reference molecule hc_c16h10_0pent_1 (4 rings, gap 6.09 eV); co-field figure `benchmarks/results/compas/figures/cofield_gap_hc_c16h10_0pent_1.png`; per-ring contributions `benchmarks/results/compas/figures/ring_contributions_gap_3D.png`.  The CMFModel refitted in float64 for the figures reproduces the table's test predictions (float32 kernels) to 8.9e-01 eV at worst; its own test R2 is 0.631, RMSE 0.268 eV.
A difference of that size means the linear model at this ridge is ill-conditioned on this set: the contributions of decomposition rule 2 are large and cancel, and the per-ring reading is not reliable.

## COMPAS-2D (CAM-B3LYP-D3BJ/def2-SVP (COMPAS-2 paper, Sci. Data 2024, 11, 97)): the cata-condensed hetero-polycyclic systems of COMPAS-2D, 10000 molecules

Charges: GFN2-xTB Mulliken charges (cmfqsar.xtb, no Fukui run, input geometry).  Rings by annulation class: angular 23893, branched 7917, linear 21230, terminal 27917.  Rings by type: benzene 41644, 14diborinine 6350, thiophene 4441, pyrrole 4343, furan 4316, pyridine 4281, borole 4175, pyrazine 4116, cyclobutadiene 3866, borinine 3425.  Atom types of the indicator fields: C.H, C.2r, N, N.H, O, S, B, B.H.  A random subset (seed 0) of the 52,000 neutral molecules.  Largest heavy-atom distance from the molecular plane: median 0.13 A, 95th percentile 1.84 A, maximum 4.10 A.

### Field sets (three random seeds and the size-extrapolation splits)

| field set | fields | q2 (5-fold CV, train) | test R2 | test RMSE (eV) | test MAE (eV) | N=8 R2 (train 2910, test 2983) | N=8 RMSE | N=9 R2 (train 5893, test 2987) | N=9 RMSE | N=10 R2 (train 8880, test 1120) | N=10 RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| {C} (3 seeds) | cif:C.H, cif:C.2r, cif:N, cif:N.H, cif:O, cif:S, cif:B, cif:B.H | 0.465 +- 0.002 | 0.443 +- 0.011 | 0.507 +- 0.008 | 0.404 +- 0.003 | 0.333 | 0.534 | 0.359 | 0.516 | 0.273 | 0.518 |
| {C} + Gaussian layer (3 seeds) | the same kernel, exp(-gamma d^2) | 0.561 +- 0.008 | 0.553 +- 0.011 | 0.454 +- 0.014 | 0.361 +- 0.013 | 0.448 | 0.485 | 0.478 | 0.466 | 0.471 | 0.442 |
| {D} (3 seeds) | ring:terminal, ring:linear, ring:angular, ring:branched | 0.164 +- 0.002 | 0.167 +- 0.014 | 0.620 +- 0.008 | 0.498 +- 0.008 | -0.337 | 0.755 | -0.061 | 0.664 | -0.126 | 0.644 |
| {D} + Gaussian layer (3 seeds) | the same kernel, exp(-gamma d^2) | 0.192 +- 0.007 | 0.207 +- 0.023 | 0.604 +- 0.011 | 0.491 +- 0.007 | -0.103 | 0.686 | 0.032 | 0.635 | -0.014 | 0.612 |
| {R} (3 seeds) | ring:benzene, ring:pyridine, ring:pyrazine, ring:borinine, ring:14diborinine, ring:pyrrole, ring:furan, ring:thiophene, ring:borole, ring:cyclobutadiene | 0.441 +- 0.006 | 0.415 +- 0.006 | 0.519 +- 0.012 | 0.411 +- 0.010 | 0.364 | 0.521 | 0.364 | 0.514 | 0.304 | 0.507 |
| {R} + Gaussian layer (3 seeds) | the same kernel, exp(-gamma d^2) | 0.502 +- 0.010 | 0.484 +- 0.019 | 0.488 +- 0.019 | 0.384 +- 0.014 | 0.438 | 0.490 | 0.430 | 0.487 | 0.395 | 0.472 |
| {A,B,C,D} (3 seeds) | steric, prop:q_xtb, cif:C.H, cif:C.2r, cif:N, cif:N.H, cif:O, cif:S, cif:B, cif:B.H, ring:terminal, ring:linear, ring:angular, ring:branched | 0.480 +- 0.004 | 0.459 +- 0.004 | 0.499 +- 0.009 | 0.399 +- 0.007 | 0.377 | 0.515 | 0.414 | 0.494 | 0.281 | 0.515 |
| {A,B,C,D} + Gaussian layer (3 seeds) | the same kernel, exp(-gamma d^2) | 0.605 +- 0.005 | 0.619 +- 0.022 | 0.419 +- 0.017 | 0.330 +- 0.014 | 0.473 | 0.474 | 0.522 | 0.446 | 0.511 | 0.425 |
| {A,B,C,D,R} (3 seeds) | steric, prop:q_xtb, cif:C.H, cif:C.2r, cif:N, cif:N.H, cif:O, cif:S, cif:B, cif:B.H, ring:terminal, ring:linear, ring:angular, ring:branched, ring:benzene, ring:pyridine, ring:pyrazine, ring:borinine, ring:14diborinine, ring:pyrrole, ring:furan, ring:thiophene, ring:borole, ring:cyclobutadiene | 0.495 +- 0.005 | 0.471 +- 0.019 | 0.494 +- 0.011 | 0.395 +- 0.014 | 0.412 | 0.501 | 0.423 | 0.490 | 0.322 | 0.500 |
| {A,B,C,D,R} + Gaussian layer (3 seeds) | the same kernel, exp(-gamma d^2) | 0.632 +- 0.008 | 0.637 +- 0.018 | 0.409 +- 0.016 | 0.322 +- 0.013 | 0.555 | 0.436 | 0.557 | 0.429 | 0.543 | 0.411 |
| {A,B,C,D,F,R} (3 seeds) | steric, prop:q_xtb, cif:C.H, cif:C.2r, cif:N, cif:N.H, cif:O, cif:S, cif:B, cif:B.H, ring:terminal, ring:linear, ring:angular, ring:branched, ring:homa, ring:benzene, ring:pyridine, ring:pyrazine, ring:borinine, ring:14diborinine, ring:pyrrole, ring:furan, ring:thiophene, ring:borole, ring:cyclobutadiene | 0.543 +- 0.007 | 0.531 +- 0.020 | 0.465 +- 0.010 | 0.366 +- 0.011 | 0.476 | 0.473 | 0.483 | 0.464 | 0.398 | 0.471 |
| {A,B,C,D,F,R} + Gaussian layer (3 seeds) | the same kernel, exp(-gamma d^2) | 0.681 +- 0.006 | 0.689 +- 0.020 | 0.379 +- 0.017 | 0.297 +- 0.014 | 0.606 | 0.410 | 0.591 | 0.413 | 0.599 | 0.384 |

### Baselines on the same splits

| model | test R2 | test RMSE (eV) | test MAE (eV) | N=8 R2 | N=8 RMSE | N=9 R2 | N=9 RMSE | N=10 R2 | N=10 RMSE |
|---|---|---|---|---|---|---|---|---|---|
| ridge on 24 counts of atom classes, ring classes, ring types and HOMA (additivity scheme) | 0.506 +- 0.011 | 0.477 +- 0.009 | 0.372 +- 0.007 | 0.454 | 0.483 | 0.448 | 0.479 | 0.360 | 0.486 |
| straight line in the number of rings | 0.062 +- 0.018 | 0.658 +- 0.015 | 0.527 +- 0.010 | -0.003 | 0.654 | -0.002 | 0.646 | -0.021 | 0.614 |
| random forest on 109 RDKit descriptors | 0.563 +- 0.017 | 0.449 +- 0.011 | 0.349 +- 0.006 | 0.481 | 0.471 | 0.456 | 0.476 | 0.434 | 0.457 |

