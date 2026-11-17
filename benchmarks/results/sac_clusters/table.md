# The rest of EC8: capped clusters, a readable co-field, a symmetry-averaged frame

`python benchmarks/run_sac_clusters.py --stage gate symmetry --force`, written 2026-11-17 19:19:54.

## The gate: the capped cluster against the references

139 catalysts; the targets have dE_H 133, dE_O 133, dE_OH 133, dE_OOH 119 values. The ligand of every cluster takes the first charge that closes its shell, and the metal the rest of the cluster's zero charge. q2 in cross-validation; every method chooses its settings and its ridge in inner folds of the same kind as the outer ones.

| Bond shells around the metal | Atoms | Hydrogen caps | Formal charge of the metal, by motif |
|---:|---:|---:|---|
| 3 | 25 | 10 | C2N2 +2, C3N +3, C4 +4, CN3 +1, N4 +0 |
| 4 | 45 | 14 | C2N2 +2, C3N +3, C4 +4, CN3 +1, N4 +0 |
| 5 | 57 | 16 | C2N2 +2, C3N +3, C4 +4, CN3 +1, N4 +0 |

**random**

| Method | dE_H | dE_O | dE_OH | dE_OOH | Mean |
|---|---:|---:|---:|---:|---:|
| capped cluster, overlap (EC4 path) | +0.25 | +0.63 | +0.81 | +0.67 | +0.592 |
| capped cluster, invariant | +0.26 | +0.61 | +0.75 | +0.60 | +0.554 |
| capped cluster overlap + metal properties | +0.57 | +0.92 | +0.91 | +0.74 | +0.784 |
| whole cell autocorrelation | +0.58 | +0.91 | +0.94 | +0.76 | +0.797 |
| site profile (M5) | +0.58 | +0.93 | +0.94 | +0.75 | +0.799 |
| site overlap, extended (pilot) | +0.52 | +0.86 | +0.92 | +0.73 | +0.759 |
| metal properties and N count | +0.52 | +0.91 | +0.89 | +0.70 | +0.754 |

**metal-out**

| Method | dE_H | dE_O | dE_OH | dE_OOH | Mean |
|---|---:|---:|---:|---:|---:|
| capped cluster, overlap (EC4 path) | +0.29 | +0.59 | +0.82 | +0.58 | +0.569 |
| capped cluster, invariant | +0.10 | +0.53 | +0.72 | +0.56 | +0.474 |
| capped cluster overlap + metal properties | +0.52 | +0.88 | +0.91 | +0.66 | +0.744 |
| whole cell autocorrelation | +0.49 | +0.87 | +0.90 | +0.72 | +0.744 |
| site profile (M5) | +0.45 | +0.89 | +0.91 | +0.58 | +0.709 |
| site overlap, extended (pilot) | +0.46 | +0.70 | +0.89 | +0.66 | +0.678 |
| metal properties and N count | +0.55 | +0.86 | +0.89 | +0.66 | +0.740 |

**motif-out**

| Method | dE_H | dE_O | dE_OH | dE_OOH | Mean |
|---|---:|---:|---:|---:|---:|
| capped cluster, overlap (EC4 path) | -0.40 | -0.61 | -4.26 | -1.23 | -1.623 |
| capped cluster, invariant | -0.01 | +0.13 | +0.27 | +0.31 | +0.174 |
| capped cluster overlap + metal properties | +0.38 | +0.85 | +0.84 | +0.69 | +0.691 |
| whole cell autocorrelation | +0.51 | +0.89 | +0.91 | +0.63 | +0.736 |
| site profile (M5) | +0.52 | +0.91 | +0.93 | +0.69 | +0.764 |
| site overlap, extended (pilot) | +0.50 | +0.82 | +0.86 | +0.67 | +0.712 |
| metal properties and N count | +0.51 | +0.91 | +0.90 | +0.73 | +0.763 |

Differences in q2 with a 95 % bootstrap interval over the catalysts, the cluster method minus the reference; bold where the interval excludes zero.

| Method | Against | Design | dE_H | dE_O | dE_OH | dE_OOH |
|---|---|---|---:|---:|---:|---:|
| capped cluster, overlap (EC4 path) | metal properties and N count | random | **-0.27 [-0.42, -0.09]** | **-0.28 [-0.39, -0.20]** | -0.07 [-0.15, +0.01] | -0.02 [-0.12, +0.08] |
| capped cluster, overlap (EC4 path) | metal properties and N count | metal-out | **-0.26 [-0.40, -0.12]** | **-0.27 [-0.38, -0.18]** | **-0.07 [-0.13, -0.03]** | -0.07 [-0.19, +0.04] |
| capped cluster, overlap (EC4 path) | metal properties and N count | motif-out | **-0.91 [-1.31, -0.58]** | **-1.53 [-2.18, -1.00]** | **-5.16 [-7.39, -3.32]** | **-1.95 [-2.82, -1.25]** |
| capped cluster, overlap (EC4 path) | site overlap, extended (pilot) | random | **-0.27 [-0.39, -0.16]** | **-0.23 [-0.31, -0.16]** | **-0.11 [-0.16, -0.07]** | -0.06 [-0.14, +0.03] |
| capped cluster, overlap (EC4 path) | site overlap, extended (pilot) | metal-out | **-0.17 [-0.25, -0.07]** | **-0.11 [-0.22, -0.02]** | **-0.07 [-0.12, -0.04]** | -0.08 [-0.18, +0.02] |
| capped cluster, overlap (EC4 path) | site overlap, extended (pilot) | motif-out | **-0.90 [-1.24, -0.60]** | **-1.43 [-2.06, -0.91]** | **-5.12 [-7.35, -3.27]** | **-1.90 [-2.79, -1.20]** |
| capped cluster, overlap (EC4 path) | site profile (M5) | random | **-0.33 [-0.47, -0.19]** | **-0.30 [-0.40, -0.22]** | **-0.12 [-0.18, -0.08]** | -0.08 [-0.19, +0.05] |
| capped cluster, overlap (EC4 path) | site profile (M5) | metal-out | **-0.16 [-0.30, -0.02]** | **-0.30 [-0.41, -0.22]** | **-0.09 [-0.14, -0.05]** | -0.00 [-0.13, +0.13] |
| capped cluster, overlap (EC4 path) | site profile (M5) | motif-out | **-0.92 [-1.33, -0.59]** | **-1.52 [-2.17, -0.99]** | **-5.19 [-7.43, -3.36]** | **-1.92 [-2.79, -1.20]** |
| capped cluster, overlap (EC4 path) | whole cell autocorrelation | random | **-0.33 [-0.49, -0.18]** | **-0.27 [-0.37, -0.20]** | **-0.13 [-0.19, -0.08]** | **-0.08 [-0.16, -0.00]** |
| capped cluster, overlap (EC4 path) | whole cell autocorrelation | metal-out | **-0.20 [-0.35, -0.04]** | **-0.28 [-0.37, -0.21]** | **-0.09 [-0.14, -0.04]** | **-0.14 [-0.22, -0.06]** |
| capped cluster, overlap (EC4 path) | whole cell autocorrelation | motif-out | **-0.91 [-1.32, -0.58]** | **-1.50 [-2.15, -0.98]** | **-5.17 [-7.39, -3.33]** | **-1.86 [-2.74, -1.15]** |
| capped cluster, invariant | metal properties and N count | random | **-0.26 [-0.42, -0.09]** | **-0.30 [-0.39, -0.23]** | **-0.14 [-0.21, -0.06]** | -0.10 [-0.23, +0.06] |
| capped cluster, invariant | metal properties and N count | metal-out | **-0.46 [-0.58, -0.33]** | **-0.34 [-0.45, -0.24]** | **-0.17 [-0.23, -0.12]** | -0.10 [-0.23, +0.05] |
| capped cluster, invariant | metal properties and N count | motif-out | **-0.52 [-0.62, -0.38]** | **-0.79 [-0.92, -0.69]** | **-0.63 [-0.75, -0.54]** | **-0.42 [-0.65, -0.18]** |
| capped cluster, invariant | site overlap, extended (pilot) | random | **-0.26 [-0.39, -0.13]** | **-0.24 [-0.32, -0.17]** | **-0.18 [-0.23, -0.13]** | -0.13 [-0.27, +0.01] |
| capped cluster, invariant | site overlap, extended (pilot) | metal-out | **-0.36 [-0.52, -0.21]** | **-0.18 [-0.29, -0.07]** | **-0.17 [-0.23, -0.12]** | -0.10 [-0.22, +0.04] |
| capped cluster, invariant | site overlap, extended (pilot) | motif-out | **-0.51 [-0.65, -0.36]** | **-0.69 [-0.82, -0.59]** | **-0.59 [-0.72, -0.48]** | **-0.36 [-0.61, -0.11]** |
| capped cluster, invariant | site profile (M5) | random | **-0.33 [-0.45, -0.21]** | **-0.31 [-0.40, -0.24]** | **-0.19 [-0.24, -0.15]** | -0.15 [-0.30, +0.03] |
| capped cluster, invariant | site profile (M5) | metal-out | **-0.36 [-0.50, -0.22]** | **-0.37 [-0.47, -0.28]** | **-0.19 [-0.25, -0.14]** | -0.02 [-0.17, +0.16] |
| capped cluster, invariant | site profile (M5) | motif-out | **-0.53 [-0.66, -0.38]** | **-0.78 [-0.92, -0.68]** | **-0.66 [-0.78, -0.56]** | **-0.39 [-0.63, -0.12]** |
| capped cluster, invariant | whole cell autocorrelation | random | **-0.33 [-0.46, -0.20]** | **-0.29 [-0.38, -0.22]** | **-0.19 [-0.24, -0.15]** | **-0.16 [-0.27, -0.04]** |
| capped cluster, invariant | whole cell autocorrelation | metal-out | **-0.39 [-0.54, -0.25]** | **-0.34 [-0.44, -0.26]** | **-0.19 [-0.25, -0.14]** | **-0.16 [-0.26, -0.05]** |
| capped cluster, invariant | whole cell autocorrelation | motif-out | **-0.52 [-0.65, -0.36]** | **-0.77 [-0.91, -0.66]** | **-0.64 [-0.75, -0.54]** | **-0.33 [-0.56, -0.07]** |
| capped cluster overlap + metal properties | metal properties and N count | random | +0.05 [-0.02, +0.13] | +0.01 [-0.02, +0.03] | +0.02 [-0.01, +0.06] | **+0.05 [+0.00, +0.10]** |
| capped cluster overlap + metal properties | metal properties and N count | metal-out | -0.03 [-0.07, +0.01] | **+0.02 [+0.00, +0.04]** | **+0.02 [+0.01, +0.04]** | +0.00 [-0.05, +0.05] |
| capped cluster overlap + metal properties | metal properties and N count | motif-out | **-0.13 [-0.28, -0.00]** | **-0.07 [-0.11, -0.03]** | **-0.07 [-0.11, -0.03]** | -0.03 [-0.08, +0.02] |
| capped cluster overlap + metal properties | site overlap, extended (pilot) | random | +0.05 [-0.05, +0.14] | **+0.06 [+0.03, +0.11]** | -0.02 [-0.06, +0.01] | +0.01 [-0.05, +0.07] |
| capped cluster overlap + metal properties | site overlap, extended (pilot) | metal-out | +0.07 [-0.06, +0.20] | **+0.18 [+0.11, +0.26]** | +0.02 [-0.01, +0.05] | -0.00 [-0.09, +0.08] |
| capped cluster overlap + metal properties | site overlap, extended (pilot) | motif-out | -0.12 [-0.31, +0.05] | +0.03 [-0.02, +0.09] | -0.02 [-0.09, +0.04] | +0.02 [-0.06, +0.10] |
| capped cluster overlap + metal properties | site profile (M5) | random | -0.01 [-0.10, +0.07] | -0.01 [-0.03, +0.01] | **-0.03 [-0.07, -0.01]** | -0.01 [-0.08, +0.07] |
| capped cluster overlap + metal properties | site profile (M5) | metal-out | +0.07 [-0.05, +0.18] | -0.01 [-0.05, +0.02] | +0.00 [-0.03, +0.04] | **+0.08 [+0.01, +0.15]** |
| capped cluster overlap + metal properties | site profile (M5) | motif-out | -0.14 [-0.32, +0.00] | **-0.06 [-0.11, -0.02]** | **-0.09 [-0.15, -0.05]** | +0.00 [-0.04, +0.05] |
| capped cluster overlap + metal properties | whole cell autocorrelation | random | -0.02 [-0.11, +0.08] | +0.02 [-0.01, +0.04] | **-0.03 [-0.07, -0.01]** | -0.01 [-0.06, +0.05] |
| capped cluster overlap + metal properties | whole cell autocorrelation | metal-out | +0.04 [-0.08, +0.15] | +0.01 [-0.02, +0.05] | +0.01 [-0.03, +0.04] | -0.06 [-0.13, +0.01] |
| capped cluster overlap + metal properties | whole cell autocorrelation | motif-out | -0.13 [-0.29, +0.01] | **-0.04 [-0.09, -0.01]** | **-0.07 [-0.12, -0.02]** | +0.06 [-0.02, +0.17] |

## The co-field of dE_H - dE_OH: smooth against sharp

The difference of the two co-fields, in eV per unit of the field, read at the metal and averaged over the first and the second shell of Fe@N4, fitted on all 133 catalysts and on 100 random halves. 'Same sign' is how often a half gives the sign of the full fit.

| Kernel | Field | Place | All | Halves, 5 to 95 % | Same sign |
|---|---|---|---:|---|---:|
| smooth (the pilot's) | electrostatic | metal | +0.226 | [+0.071, +0.238] | 100 % |
| smooth (the pilot's) | electrostatic | first shell | +0.016 | [-0.019, +0.023] | 66 % |
| smooth (the pilot's) | electrostatic | second shell | -0.016 | [-0.041, +0.010] | 81 % |
| smooth (the pilot's) | steric | metal | +0.005 | [-0.000, +0.008] | 94 % |
| smooth (the pilot's) | steric | first shell | +0.001 | [-0.002, +0.004] | 76 % |
| smooth (the pilot's) | steric | second shell | +0.000 | [-0.001, +0.002] | 72 % |
| smooth (the pilot's) | prop:electronegativity | metal | -0.004 | [-0.006, +0.002] | 79 % |
| smooth (the pilot's) | prop:electronegativity | first shell | -0.004 | [-0.004, -0.000] | 97 % |
| smooth (the pilot's) | prop:electronegativity | second shell | -0.001 | [-0.002, +0.000] | 86 % |
| smooth (the pilot's) | prop:d_count | metal | -0.005 | [-0.015, -0.001] | 97 % |
| smooth (the pilot's) | prop:d_count | first shell | +0.000 | [-0.004, +0.000] | 9 % |
| smooth (the pilot's) | prop:d_count | second shell | +0.000 | [-0.002, +0.000] | 11 % |
| smooth (the pilot's) | prop:valence | metal | -0.003 | [-0.006, +0.000] | 93 % |
| smooth (the pilot's) | prop:valence | first shell | -0.001 | [-0.003, -0.000] | 95 % |
| smooth (the pilot's) | prop:valence | second shell | -0.000 | [-0.001, +0.000] | 92 % |
| smooth (the pilot's) | prop:ionisation | metal | +0.001 | [-0.001, +0.001] | 62 % |
| smooth (the pilot's) | prop:ionisation | first shell | -0.000 | [-0.001, +0.000] | 62 % |
| smooth (the pilot's) | prop:ionisation | second shell | +0.000 | [-0.000, +0.000] | 45 % |
| smooth (the pilot's) | prop:radius | metal | -0.010 | [-0.024, +0.013] | 63 % |
| smooth (the pilot's) | prop:radius | first shell | -0.015 | [-0.016, -0.002] | 99 % |
| smooth (the pilot's) | prop:radius | second shell | -0.006 | [-0.006, -0.001] | 96 % |
| sharp (chosen for accuracy) | electrostatic | metal | +0.341 | [-4.409, +3.282] | 50 % |
| sharp (chosen for accuracy) | electrostatic | first shell | -0.757 | [-0.980, +1.194] | 22 % |
| sharp (chosen for accuracy) | electrostatic | second shell | +0.039 | [-0.665, +0.359] | 44 % |
| sharp (chosen for accuracy) | steric | metal | +0.864 | [-2.369, +4.065] | 65 % |
| sharp (chosen for accuracy) | steric | first shell | -1.915 | [-3.375, +0.608] | 84 % |
| sharp (chosen for accuracy) | steric | second shell | -0.976 | [-1.888, +0.249] | 90 % |
| sharp (chosen for accuracy) | prop:electronegativity | metal | -9.332 | [-9.253, -1.669] | 99 % |
| sharp (chosen for accuracy) | prop:electronegativity | first shell | +0.278 | [-1.333, +1.895] | 66 % |
| sharp (chosen for accuracy) | prop:electronegativity | second shell | +0.057 | [-0.211, +1.792] | 85 % |
| sharp (chosen for accuracy) | prop:d_count | metal | +3.016 | [-1.828, +3.743] | 63 % |
| sharp (chosen for accuracy) | prop:d_count | first shell | +0.040 | [-0.077, +0.052] | 38 % |
| sharp (chosen for accuracy) | prop:d_count | second shell | +0.001 | [-0.002, +0.001] | 39 % |
| sharp (chosen for accuracy) | prop:valence | metal | +1.198 | [-4.951, +4.219] | 43 % |
| sharp (chosen for accuracy) | prop:valence | first shell | -0.166 | [-1.365, +1.461] | 46 % |
| sharp (chosen for accuracy) | prop:valence | second shell | +1.298 | [-0.181, +1.893] | 89 % |
| sharp (chosen for accuracy) | prop:ionisation | metal | -3.812 | [-4.429, +4.137] | 53 % |
| sharp (chosen for accuracy) | prop:ionisation | first shell | -1.120 | [-3.399, +0.236] | 90 % |
| sharp (chosen for accuracy) | prop:ionisation | second shell | -2.033 | [-2.016, +0.498] | 86 % |
| sharp (chosen for accuracy) | prop:radius | metal | -5.420 | [-5.620, +1.139] | 86 % |
| sharp (chosen for accuracy) | prop:radius | first shell | -1.134 | [-1.972, +1.016] | 74 % |
| sharp (chosen for accuracy) | prop:radius | second shell | -0.234 | [-0.897, +1.163] | 55 % |

## The frame averaged over the symmetry of the site

The overlap kernels averaged over the 4 operations of the divacancy that keep the normal (identity, the turn by 180 degrees, the two mirrors), against the same kernels in one frame.

**random**

| Method | dE_H | dE_O | dE_OH | dE_OOH | Mean |
|---|---:|---:|---:|---:|---:|
| site overlap, extended, one frame | +0.59 | +0.88 | +0.93 | +0.73 | +0.785 |
| site overlap, extended, averaged over D2h | +0.56 | +0.89 | +0.94 | +0.74 | +0.781 |
| capped cluster (4 shells), one frame | +0.24 | +0.65 | +0.83 | +0.67 | +0.598 |
| capped cluster (4 shells), averaged over D2h | +0.31 | +0.69 | +0.83 | +0.70 | +0.634 |

**metal-out**

| Method | dE_H | dE_O | dE_OH | dE_OOH | Mean |
|---|---:|---:|---:|---:|---:|
| site overlap, extended, one frame | +0.46 | +0.80 | +0.88 | +0.66 | +0.699 |
| site overlap, extended, averaged over D2h | +0.43 | +0.82 | +0.89 | +0.65 | +0.697 |
| capped cluster (4 shells), one frame | +0.24 | +0.57 | +0.81 | +0.58 | +0.549 |
| capped cluster (4 shells), averaged over D2h | +0.19 | +0.57 | +0.78 | +0.57 | +0.528 |

**motif-out**

| Method | dE_H | dE_O | dE_OH | dE_OOH | Mean |
|---|---:|---:|---:|---:|---:|
| site overlap, extended, one frame | +0.54 | +0.61 | +0.86 | +0.72 | +0.682 |
| site overlap, extended, averaged over D2h | +0.57 | +0.86 | +0.94 | +0.68 | +0.762 |
| capped cluster (4 shells), one frame | -0.72 | +0.44 | +0.50 | +0.58 | +0.201 |
| capped cluster (4 shells), averaged over D2h | +0.15 | +0.49 | +0.72 | +0.59 | +0.489 |

| Method | Against | Design | dE_H | dE_O | dE_OH | dE_OOH |
|---|---|---|---:|---:|---:|---:|
| capped cluster (4 shells), averaged over D2h | capped cluster (4 shells), one frame | random | +0.07 [-0.02, +0.14] | **+0.04 [+0.01, +0.09]** | +0.00 [-0.01, +0.02] | **+0.03 [+0.00, +0.06]** |
| capped cluster (4 shells), averaged over D2h | capped cluster (4 shells), one frame | metal-out | **-0.06 [-0.12, -0.00]** | +0.00 [-0.03, +0.04] | **-0.03 [-0.05, -0.01]** | -0.01 [-0.07, +0.05] |
| capped cluster (4 shells), averaged over D2h | capped cluster (4 shells), one frame | motif-out | **+0.87 [+0.53, +1.31]** | +0.05 [-0.09, +0.20] | **+0.22 [+0.04, +0.42]** | +0.01 [-0.09, +0.10] |
| site overlap, extended, averaged over D2h | site overlap, extended, one frame | random | -0.03 [-0.08, +0.01] | +0.00 [-0.01, +0.02] | **+0.01 [+0.00, +0.01]** | +0.00 [-0.02, +0.02] |
| site overlap, extended, averaged over D2h | site overlap, extended, one frame | metal-out | -0.02 [-0.09, +0.05] | +0.02 [-0.01, +0.04] | **+0.01 [+0.00, +0.02]** | -0.01 [-0.03, +0.00] |
| site overlap, extended, averaged over D2h | site overlap, extended, one frame | motif-out | +0.03 [-0.06, +0.12] | **+0.25 [+0.15, +0.39]** | **+0.08 [+0.05, +0.12]** | -0.04 [-0.09, +0.01] |
