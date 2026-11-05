# Item EC1: quantum fields on a phenazine reduction-potential series

69 substituted phenazines with a computed one-electron reduction potential, from -2.092 to -0.160 V (sd 0.392 V). The mean of the training set alone makes an error of 0.285 V.

Quantum stage: `B3LYP D3BJ def2-SVP RIJCOSX def2/J TightSCF` with the continuum solvent `DME`, at the geometry each source gives, never optimised.

| What | Value |
|---|---|
| Core of one molecule against the reference core | 0.08 +/- 0.07 A |
| Spin populations of the radical anion, summed | 1.0000 +/- 0.0000 (a doublet must give 1) |
| Share of it on the two core nitrogens | 34.5% +/- 5.9% |
| Correlation of the target with `EA` | +0.766 |
| Correlation of the target with `E_lumo` | -0.757 |
| Correlation of the target with `E_homo` | -0.302 |

### How much does each field vary across the series?

The kernel is a cosine-normalised overlap, so this is what it sees. A mean near 1 means the field is nearly the same shape in every molecule, and a model cannot tell two of them apart by it.

| field alone | similarity of two molecules | smallest |
|---|---:|---:|
| `fukui_plus` | 0.953 +/- 0.032 | +0.771 |
| `spin_anion` | 0.910 +/- 0.127 | -0.082 |
| `steric` | 0.902 +/- 0.057 | +0.702 |
| `fukui_minus` | 0.701 +/- 0.201 | +0.096 |
| `q_hirshfeld` | 0.344 +/- 0.182 | -0.323 |
| `electrostatic` | 0.315 +/- 0.168 | -0.105 |
| `dual` | 0.308 +/- 0.307 | -0.895 |

## What each description predicts

Mean absolute error over the folds, in volts. Every model chooses its ridge and its field width inside each fold.

| model | random | one source table held out |
|---|---:|---:|
| `ea` | 0.1937 +/- 0.0229 | 0.2383 +/- 0.0800 |
| `descriptors` | 0.1810 +/- 0.0303 | 0.2312 +/- 0.0957 |
| `physchem` | 0.1582 +/- 0.0186 | 0.2325 +/- 0.1021 |
| `hirshfeld` | 0.1527 +/- 0.0173 | 0.2168 +/- 0.1070 |
| `fukui` | 0.1505 +/- 0.0205 | 0.2405 +/- 0.1040 |
| `spin` | 0.1609 +/- 0.0212 | 0.2296 +/- 0.1063 |
| `quantum` | 0.1507 +/- 0.0165 | 0.2303 +/- 0.1199 |
| `spin-only` | 0.2901 +/- 0.0391 | 0.2697 +/- 0.1004 |

## The comparisons

The paired difference of the two models' per-molecule absolute errors, with a 95 % bootstrap interval. A positive mean means the second model is the better one. An interval that holds zero is not a difference.

| comparison | random | one source table held out |
|---|---|---|
| hirshfeld vs physchem | +0.0055 [-0.0155, +0.0281] | +0.0148 [-0.0083, +0.0377] |
| fukui vs physchem | +0.0071 [-0.0124, +0.0286] | -0.0083 [-0.0231, +0.0070] |
| spin vs physchem | -0.0027 [-0.0078, +0.0022] | +0.0022 [-0.0059, +0.0115] |
| quantum vs physchem | +0.0073 [-0.0149, +0.0310] | -0.0011 [-0.0261, +0.0258] |
| spin vs descriptors | +0.0207 [-0.0250, +0.0648] | -0.0005 [-0.0490, +0.0463] |
| quantum vs descriptors | +0.0307 [-0.0183, +0.0791] | -0.0037 [-0.0438, +0.0374] |
| physchem vs descriptors | +0.0234 [-0.0243, +0.0691] | -0.0026 [-0.0509, +0.0446] |
| quantum vs ea | +0.0428 [-0.0027, +0.0895] | -0.0000 [-0.0391, +0.0408] |
| spin vs ea | +0.0328 [-0.0102, +0.0751] | +0.0032 [-0.0412, +0.0508] |
| spin-only vs descriptors | -0.1081 [-0.1647, -0.0536] | -0.0393 [-0.0707, -0.0061] |

## The co-field of the quantum model

The point of largest magnitude of each field's co-field, and how far it is from the nearest core nitrogen. The spin population of the radical anion sits on those nitrogens, so that distance is what makes the position readable.

| field | peak | distance from a core nitrogen, A | from the core centre, A |
|---|---:|---:|---:|
| Steric | +0.0237 | 2.98 | 4.19 |
| Hydrophobic | +0.0940 | 4.78 | 5.40 |
| H-bond acceptor | +0.0787 | 2.24 | 3.65 |
| H-bond donor | +0.1222 | 0.91 | 2.45 |
| q_hirshfeld | +0.2008 | 2.87 | 3.99 |
| fukui_plus | -0.0275 | 2.43 | 3.21 |
| fukui_minus | -0.0380 | 3.03 | 3.42 |
| dual | +0.0661 | 1.09 | 1.18 |
| spin_anion | +0.0366 | 2.87 | 3.98 |

