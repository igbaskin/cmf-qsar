# A single-atom site as the sample: the field around the metal against the H, O, OH and OOH binding energies

The 139 catalysts of He, Wang and Ouyang (Chem. Mater. 2023) with a relaxed bare sheet: 28 metals (Ag, Au, Cd, Co, Cr, Cu, Fe, Hf, Ir, La, Mn, Mo, Nb, Ni, Os, Pd, Pt, Re, Rh, Ru, Sc, Ta, Ti, V, W, Y, Zn, Zr) in the motifs C2N2, C3N, C4, CN3, N4. The targets are electronic binding energies against computed gas references (H2 -6.772 eV, H2O -14.234 eV, PBE+D2 at the settings of the set). Folds: 10 random, one per metal, one per motif; every method chooses its kernel and ridge in an inner cross-validation of the same kind. Differences carry a 95% bootstrap interval over catalysts.

* 40 bare sheets were relaxed with DFT+U in the source data while every adsorbed state was relaxed without it; 0 of them have a plain-PBE energy recomputed here and 40 carry no binding energy
* 1 bare sheets have a source run with an energy inconsistent with its adsorbed states; 0 of them have a plain-PBE energy recomputed here and 1 carry no binding energy
* 15 adsorbed states ended as another species (an O-O or O-H bond broken) and carry no binding energy: Os@C3N OOH, Re@C3N OOH, Os@C2N2 OOH, Re@C2N2 OOH, W@C2N2 OOH, Cr@CN3 OOH, Os@CN3 OOH, Re@CN3 OOH, Ru@CN3 OOH, Ta@CN3 OOH, Nb@N4 OOH, Os@N4 OOH, Re@N4 OOH, Ta@N4 OOH, Ti@N4 OOH
* Ag@C4: OOH ended 2.91 A from the metal, on the sheet rather than on the site; kept
* Au@C4: OOH ended 2.69 A from the metal, on the sheet rather than on the site; kept
* Ag@C3N: OOH ended 2.85 A from the metal, on the sheet rather than on the site; kept
* Au@C3N: OOH ended 2.82 A from the metal, on the sheet rather than on the site; kept
* La@C2N2: H ended 2.93 A from the metal, on the sheet rather than on the site; kept
* Sc@C2N2: H ended 2.60 A from the metal, on the sheet rather than on the site; kept
* Au@CN3: OOH ended 2.73 A from the metal, on the sheet rather than on the site; kept

## The targets

| target | n | mean, eV | sd, eV |
|---|---:|---:|---:|
| dE_H | 98 | +0.07 | 1.08 |
| dE_O | 98 | +1.15 | 2.31 |
| dE_OH | 98 | -0.44 | 1.58 |
| dE_OOH | 85 | +2.87 | 1.40 |

## q² by method, random folds

| method | dE_H | dE_O | dE_OH | dE_OOH | mean |
|---|---:|---:|---:|---:|---:|
| one-hot metal and motif | +0.44 | +0.80 | +0.85 | +0.85 | +0.74 |
| metal properties and N count | +0.52 | +0.91 | +0.91 | +0.93 | +0.82 |
| periodic autocorrelation, whole cell | +0.55 | +0.93 | +0.94 | +0.95 | +0.84 |
| site invariant, classic | +0.42 | +0.79 | +0.84 | +0.86 | +0.73 |
| site overlap, classic | +0.46 | +0.85 | +0.89 | +0.89 | +0.77 |
| site invariant, extended | +0.63 | +0.92 | +0.94 | +0.95 | +0.86 |
| site overlap, extended | +0.58 | +0.90 | +0.93 | +0.94 | +0.84 |
| site invariant, indicator | +0.63 | +0.88 | +0.90 | +0.90 | +0.83 |
| site overlap, indicator | +0.60 | +0.88 | +0.93 | +0.92 | +0.83 |

RMSE in eV, random folds:

| method | dE_H | dE_O | dE_OH | dE_OOH |
|---|---:|---:|---:|---:|
| one-hot metal and motif | 0.81 | 1.03 | 0.61 | 0.53 |
| metal properties and N count | 0.75 | 0.70 | 0.46 | 0.36 |
| periodic autocorrelation, whole cell | 0.73 | 0.60 | 0.39 | 0.31 |
| site invariant, classic | 0.82 | 1.07 | 0.64 | 0.53 |
| site overlap, classic | 0.79 | 0.90 | 0.51 | 0.47 |
| site invariant, extended | 0.66 | 0.65 | 0.38 | 0.30 |
| site overlap, extended | 0.70 | 0.73 | 0.41 | 0.36 |
| site invariant, indicator | 0.66 | 0.80 | 0.49 | 0.44 |
| site overlap, indicator | 0.68 | 0.81 | 0.42 | 0.41 |

## q² by method, metal-out folds

| method | dE_H | dE_O | dE_OH | dE_OOH | mean |
|---|---:|---:|---:|---:|---:|
| one-hot metal and motif | -0.03 | -0.09 | -0.09 | -0.11 | -0.08 |
| metal properties and N count | +0.56 | +0.91 | +0.92 | +0.84 | +0.81 |
| periodic autocorrelation, whole cell | +0.44 | +0.89 | +0.91 | +0.89 | +0.78 |
| site invariant, classic | +0.14 | +0.56 | +0.45 | +0.29 | +0.36 |
| site overlap, classic | +0.36 | +0.72 | +0.90 | +0.86 | +0.71 |
| site invariant, extended | +0.50 | +0.87 | +0.91 | +0.92 | +0.80 |
| site overlap, extended | +0.44 | +0.72 | +0.91 | +0.88 | +0.74 |
| site invariant, indicator | -0.03 | +0.09 | +0.06 | +0.17 | +0.07 |
| site overlap, indicator | +0.02 | +0.28 | +0.46 | +0.27 | +0.26 |

RMSE in eV, metal-out folds:

| method | dE_H | dE_O | dE_OH | dE_OOH |
|---|---:|---:|---:|---:|
| one-hot metal and motif | 1.09 | 2.41 | 1.64 | 1.47 |
| metal properties and N count | 0.72 | 0.69 | 0.46 | 0.56 |
| periodic autocorrelation, whole cell | 0.81 | 0.77 | 0.48 | 0.46 |
| site invariant, classic | 1.00 | 1.53 | 1.17 | 1.18 |
| site overlap, classic | 0.87 | 1.23 | 0.51 | 0.53 |
| site invariant, extended | 0.77 | 0.82 | 0.48 | 0.40 |
| site overlap, extended | 0.81 | 1.23 | 0.46 | 0.47 |
| site invariant, indicator | 1.09 | 2.20 | 1.53 | 1.27 |
| site overlap, indicator | 1.07 | 1.96 | 1.16 | 1.19 |

## q² by method, motif-out folds

| method | dE_H | dE_O | dE_OH | dE_OOH | mean |
|---|---:|---:|---:|---:|---:|
| one-hot metal and motif | +0.39 | +0.81 | +0.85 | +0.85 | +0.73 |
| metal properties and N count | +0.57 | +0.89 | +0.91 | +0.92 | +0.82 |
| periodic autocorrelation, whole cell | +0.58 | +0.89 | +0.92 | +0.92 | +0.83 |
| site invariant, classic | +0.29 | +0.56 | +0.74 | +0.78 | +0.59 |
| site overlap, classic | +0.48 | +0.62 | +0.81 | +0.85 | +0.69 |
| site invariant, extended | +0.50 | +0.87 | +0.93 | +0.94 | +0.81 |
| site overlap, extended | +0.56 | +0.84 | +0.92 | +0.94 | +0.82 |
| site invariant, indicator | +0.60 | +0.87 | +0.92 | +0.92 | +0.83 |
| site overlap, indicator | +0.53 | +0.88 | +0.93 | +0.90 | +0.81 |

RMSE in eV, motif-out folds:

| method | dE_H | dE_O | dE_OH | dE_OOH |
|---|---:|---:|---:|---:|
| one-hot metal and motif | 0.84 | 0.99 | 0.61 | 0.54 |
| metal properties and N count | 0.71 | 0.75 | 0.48 | 0.39 |
| periodic autocorrelation, whole cell | 0.70 | 0.78 | 0.44 | 0.39 |
| site invariant, classic | 0.91 | 1.52 | 0.80 | 0.66 |
| site overlap, classic | 0.78 | 1.42 | 0.69 | 0.54 |
| site invariant, extended | 0.76 | 0.82 | 0.40 | 0.35 |
| site overlap, extended | 0.71 | 0.91 | 0.43 | 0.35 |
| site invariant, indicator | 0.68 | 0.83 | 0.44 | 0.41 |
| site overlap, indicator | 0.74 | 0.79 | 0.42 | 0.45 |

## site overlap, extended against the references: difference in q² with its interval

| design | reference | dE_H | dE_O | dE_OH | dE_OOH |
|---|---|---:|---:|---:|---:|
| random | one-hot metal and motif | +0.14 [-0.02, +0.30] | +0.10 [+0.06, +0.16] | +0.08 [+0.04, +0.13] | +0.08 [+0.04, +0.13] |
| random | metal properties and N count | +0.06 [-0.07, +0.21] | -0.01 [-0.05, +0.03] | +0.02 [-0.02, +0.05] | +0.00 [-0.02, +0.03] |
| random | periodic autocorrelation, whole cell | +0.03 [-0.10, +0.16] | -0.03 [-0.07, -0.00] | -0.01 [-0.03, +0.01] | -0.01 [-0.04, +0.01] |
| random | site invariant, classic | +0.15 [+0.01, +0.32] | +0.11 [+0.05, +0.19] | +0.09 [+0.05, +0.15] | +0.08 [+0.04, +0.13] |
| random | site invariant, extended | -0.05 [-0.14, +0.03] | -0.02 [-0.05, +0.00] | -0.01 [-0.03, +0.01] | -0.02 [-0.04, -0.00] |
| random | site invariant, indicator | -0.05 [-0.18, +0.08] | +0.02 [-0.02, +0.07] | +0.03 [-0.00, +0.06] | +0.04 [+0.01, +0.07] |
| metal-out | one-hot metal and motif | +0.47 [+0.30, +0.63] | +0.81 [+0.71, +0.91] | +1.00 [+0.95, +1.07] | +0.99 [+0.95, +1.06] |
| metal-out | metal properties and N count | -0.11 [-0.27, +0.03] | -0.19 [-0.30, -0.11] | -0.00 [-0.04, +0.04] | +0.05 [-0.01, +0.12] |
| metal-out | periodic autocorrelation, whole cell | -0.00 [-0.16, +0.18] | -0.17 [-0.26, -0.10] | +0.00 [-0.02, +0.03] | -0.01 [-0.05, +0.03] |
| metal-out | site invariant, classic | +0.30 [+0.11, +0.51] | +0.16 [+0.03, +0.30] | +0.47 [+0.31, +0.68] | +0.59 [+0.42, +0.82] |
| metal-out | site invariant, extended | -0.05 [-0.19, +0.09] | -0.16 [-0.24, -0.08] | +0.01 [-0.03, +0.04] | -0.03 [-0.07, -0.00] |
| metal-out | site invariant, indicator | +0.47 [+0.30, +0.65] | +0.63 [+0.51, +0.77] | +0.85 [+0.78, +0.99] | +0.71 [+0.60, +0.83] |
| motif-out | one-hot metal and motif | +0.17 [-0.00, +0.34] | +0.03 [-0.02, +0.08] | +0.08 [+0.03, +0.13] | +0.09 [+0.05, +0.14] |
| motif-out | metal properties and N count | -0.01 [-0.15, +0.15] | -0.05 [-0.10, -0.00] | +0.02 [-0.02, +0.08] | +0.02 [-0.01, +0.05] |
| motif-out | periodic autocorrelation, whole cell | -0.02 [-0.17, +0.15] | -0.04 [-0.09, +0.01] | +0.00 [-0.03, +0.03] | +0.02 [-0.01, +0.04] |
| motif-out | site invariant, classic | +0.27 [+0.09, +0.45] | +0.28 [+0.17, +0.41] | +0.18 [+0.11, +0.27] | +0.16 [+0.10, +0.25] |
| motif-out | site invariant, extended | +0.06 [-0.09, +0.20] | -0.03 [-0.07, +0.01] | -0.01 [-0.03, +0.01] | +0.00 [-0.02, +0.02] |
| motif-out | site invariant, indicator | -0.04 [-0.19, +0.11] | -0.03 [-0.07, +0.02] | +0.00 [-0.02, +0.04] | +0.02 [-0.01, +0.06] |

## What the overlap models chose

* random, dE_H: rho=0.2 alpha=2 cosine lam=0.01 (3 folds); rho=0.2 alpha=2 raw lam=0.01 (2 folds); rho=0.1 alpha=2 cosine lam=0.001 (2 folds)
* random, dE_O: rho=0.2 alpha=2 cosine lam=0.001 (7 folds); rho=0.1 alpha=2 cosine lam=0.001 (2 folds); rho=0.2 alpha=1 raw lam=0.001 (1 folds)
* random, dE_OH: rho=0.2 alpha=1 raw lam=0.01 (3 folds); rho=0.2 alpha=0.5 raw lam=0.01 (3 folds); rho=0.2 alpha=0.5 raw lam=0.001 (2 folds)
* random, dE_OOH: rho=0.2 alpha=0.5 raw lam=0.001 (5 folds); rho=0.2 alpha=1 raw lam=0.01 (1 folds); rho=0.2 alpha=2 raw lam=0.01 (1 folds)
* metal-out, dE_H: rho=0.1 alpha=2 cosine lam=0.01 (5 folds); rho=0.2 alpha=0.5 raw lam=0.01 (4 folds); rho=0.2 alpha=2 cosine lam=0.01 (4 folds)
* metal-out, dE_O: rho=0.2 alpha=2 raw lam=0.01 (7 folds); rho=0.1 alpha=1 raw lam=0.001 (3 folds); rho=0.2 alpha=2 cosine lam=0.001 (2 folds)
* metal-out, dE_OH: rho=0.2 alpha=0.5 raw lam=0.001 (12 folds); rho=0.2 alpha=0.5 raw lam=0.1 (1 folds); rho=0.2 alpha=1 raw lam=0.01 (1 folds)
* metal-out, dE_OOH: rho=0.2 alpha=0.5 raw lam=0.001 (7 folds); rho=0.1 alpha=0.5 raw lam=0.001 (6 folds); rho=0.2 alpha=2 raw lam=0.001 (2 folds)
* motif-out, dE_H: rho=0.1 alpha=1 raw lam=0.01 (2 folds); rho=0.1 alpha=2 raw lam=0.01 (1 folds); rho=0.2 alpha=2 raw lam=0.1 (1 folds)
* motif-out, dE_O: rho=0.2 alpha=2 raw lam=0.01 (2 folds); rho=0.1 alpha=2 raw lam=0.01 (1 folds); rho=0.2 alpha=1 raw lam=0.01 (1 folds)
* motif-out, dE_OH: rho=0.2 alpha=2 raw lam=0.01 (2 folds); rho=0.2 alpha=1 raw lam=0.01 (2 folds); rho=0.2 alpha=0.5 raw lam=0.01 (1 folds)
* motif-out, dE_OOH: rho=0.2 alpha=2 raw lam=0.01 (3 folds); rho=0.1 alpha=2 raw lam=0.01 (1 folds); rho=0.2 alpha=0.5 raw lam=0.01 (1 folds)

## The GP variance against the error

| design | target | 95% interval covers | rank correlation of predicted sd with error |
|---|---|---:|---:|
| metal-out | dE_H | 90% | +0.17 |
| metal-out | dE_O | 86% | +0.34 |
| metal-out | dE_OH | 92% | +0.06 |
| metal-out | dE_OOH | 84% | +0.16 |
| random | dE_H | 88% | +0.05 |
| random | dE_O | 93% | +0.11 |
| random | dE_OH | 92% | -0.08 |
| random | dE_OOH | 91% | -0.13 |

## The four endpoints together (kernel rho=0.2 alpha=0.5 raw)

Multi-task fit with the task correlation chosen by cross-validation: rho 0, ridge 0.001. Mean q² over the four endpoints +0.847 shared against +0.847 with four separate fits.

| endpoint | q² shared | q² alone |
|---|---:|---:|
| dE_H | +0.630 | +0.630 |
| dE_O | +0.877 | +0.877 |
| dE_OH | +0.938 | +0.938 |
| dE_OOH | +0.942 | +0.942 |

The bifunctional difference dE_H - dE_OH (sd 1.30 eV, 98 catalysts): q² +0.822 (RMSE 0.55 eV) from the shared fit, +0.787 (RMSE 0.60 eV) as the difference of the two single-endpoint predictions.

## Overpotentials from the out-of-fold binding energies

Free energies with the corrections {'H': 0.24, 'O': 0.05, 'OH': 0.35, 'OOH': 0.4} eV; eta_HER = |dG_H|, eta_ORR = 1.23 V minus the limiting potential of the four-electron associative path. The bifunctional score of a site is the larger of its two overpotentials.

| design | HER mae, V | HER rank r | ORR mae, V | ORR rank r | bifunctional rank r | best 10 recovered |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.47 | +0.57 | 0.30 | +0.90 | +0.88 | 7/10 |
| metal-out | 0.54 | +0.38 | 0.42 | +0.79 | +0.81 | 7/10 |
| motif-out | 0.45 | +0.60 | 0.32 | +0.84 | +0.87 | 6/10 |

Best 10 bifunctional sites by DFT: Rh@N4, Rh@CN3, Ir@CN3, Ir@N4, Ag@N4, Pt@C3N, Pd@C4, Rh@C3N, Pt@C4, Cd@N4.
Best 10 by the random-fold predictions: Rh@CN3, Ir@CN3, Pt@C4, Pt@C3N, Cd@CN3, Pd@C4, Rh@N4, Ir@N4, Zn@CN3, Cu@N4.

## The co-fields at named points (extended fields, rho 0.2, alpha 0.5, raw kernel)

Value of the co-field per unit of field weight, in eV, averaged over the atoms of a shell (Fe@N4 as the reference geometry). A positive value says more of that field at that place raises the binding energy, that is weakens the binding.

| field | place | H | OH | OOH | H - OH |
|---|---|---:|---:|---:|---:|
| electrostatic | metal | -0.077 | -0.471 | -0.541 | +0.393 |
| electrostatic | first shell | -0.013 | -0.121 | -0.065 | +0.107 |
| electrostatic | second shell | -0.002 | -0.041 | -0.068 | +0.039 |
| electrostatic | 2 A above the metal | +0.392 | +0.485 | +0.482 | -0.093 |
| steric | metal | -0.014 | -0.019 | -0.017 | +0.005 |
| steric | first shell | -0.014 | -0.010 | -0.011 | -0.004 |
| steric | second shell | -0.009 | -0.006 | -0.010 | -0.003 |
| steric | 2 A above the metal | +0.004 | -0.027 | -0.045 | +0.031 |
| prop:electronegativity | metal | -0.063 | -0.047 | -0.071 | -0.016 |
| prop:electronegativity | first shell | -0.035 | -0.022 | -0.033 | -0.013 |
| prop:electronegativity | second shell | -0.014 | -0.011 | -0.014 | -0.004 |
| prop:electronegativity | 2 A above the metal | -0.002 | -0.021 | -0.023 | +0.019 |
| prop:d_count | metal | -0.010 | -0.004 | -0.014 | -0.006 |
| prop:d_count | first shell | +0.010 | +0.008 | +0.005 | +0.002 |
| prop:d_count | second shell | +0.004 | +0.003 | +0.002 | +0.001 |
| prop:d_count | 2 A above the metal | -0.055 | +0.001 | +0.002 | -0.056 |
| prop:valence | metal | +0.008 | -0.004 | -0.009 | +0.012 |
| prop:valence | first shell | +0.000 | -0.003 | -0.011 | +0.004 |
| prop:valence | second shell | -0.001 | -0.002 | -0.005 | +0.002 |
| prop:valence | 2 A above the metal | +0.002 | +0.003 | +0.018 | -0.001 |
| prop:ionisation | metal | -0.013 | -0.012 | -0.011 | -0.002 |
| prop:ionisation | first shell | -0.007 | -0.006 | -0.006 | -0.001 |
| prop:ionisation | second shell | -0.003 | -0.003 | -0.003 | -0.000 |
| prop:ionisation | 2 A above the metal | -0.003 | -0.004 | -0.004 | +0.001 |
| prop:radius | metal | +0.188 | +0.187 | +0.284 | +0.001 |
| prop:radius | first shell | +0.032 | +0.057 | +0.061 | -0.025 |
| prop:radius | second shell | +0.006 | +0.014 | +0.014 | -0.007 |
| prop:radius | 2 A above the metal | +0.171 | +0.094 | +0.250 | +0.077 |

![co-fields](cofields.png)

![cross-validation](cv_scatter.png)

Run time 11.4 min.
