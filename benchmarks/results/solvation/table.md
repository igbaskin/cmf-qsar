# A cation with its first shell as the sample (plan item BT4)

## The builder against a published relaxation

150 of the Li(+)-ether complexes of `examples/ether/`, GFN2-xTB acc=1 opt. The first shell is counted inside 2.6 A.

| Quantity | The published relaxation | xTB from that geometry | The builder's own complex |
|---|---|---|---|
| oxygens reaching the ion | 4.24 +/- 0.87 | 4.38 +/- 0.78 | 3.81 +/- 0.84 |
| nearest oxygen (A) | 1.93 +/- 0.05 | 1.87 +/- 0.05 | 1.86 +/- 0.05 |
| binding energy (eV) | -3.89 +/- 0.56 | -5.22 +/- 0.62 | -4.81 +/- 0.49 |

* Against the published binding energy, xTB on the published geometry: correlation +0.907, mean error -1.323 eV, mean absolute error 1.323 eV on 150 molecules.
* Against the published binding energy, the builder's own complex: correlation +0.555, mean error -0.919 eV, mean absolute error 0.938 eV on 150 molecules.

## The gate: does the shell carry what the solvent alone does not

1048 complexes, the published binding energy as the target, five outer folds with the ridge and the field width chosen inside each. The ion has 4.16 +/- 0.97 donors inside 2.6 A. Predicting the mean of the set costs 0.454 eV.

| Model | MAE (eV) | R2 |
|---|---|---|
| solvent, no frame | 0.124 +/- 0.016 | +0.894 |
| complex, no frame | 0.114 +/- 0.013 | +0.914 |
| complex, cation frame | 0.145 +/- 0.018 | +0.868 |
| solvent, cation frame | 0.143 +/- 0.009 | +0.886 |

* Paired over the same molecules, complex frame beats solvent no frame: +0.021 eV [+0.011, +0.030], excludes zero.
* Paired over the same molecules, complex no frame beats solvent no frame: -0.010 eV [-0.016, -0.004], excludes zero.
* Paired over the same molecules, the frame alone: +0.018 eV [+0.010, +0.027], excludes zero.
* Paired over the same molecules, the ion inside the frame: +0.002 eV [-0.003, +0.008], includes zero.

In 0.3% of the complexes the two smallest moments of the polyhedron agree to within one per cent, which leaves the frame free to rotate about an axis.

### What the co-field names

The fitted co-field of the complex model, evaluated on a grid: where its largest magnitude lies.

| Field | Distance from the ion (A) | Distance from the nearest donor (A) |
|---|---|---|
| Electrostatic | 1.37 | 1.37 |
| Steric | 4.01 | 2.40 |
| Hydrophobic | 3.04 | 1.54 |
| H-bond acceptor | 1.03 | 1.55 |
| H-bond donor | 12.49 | 11.35 |

## The series computed here

Every complex built by `cmfqsar/solvation.py` and optimised with GFN2-xTB acc=1 opt. The binding energy is E(complex) - E(ion) - n E(solvent) and the desolvation energy E(n-1) + E(solvent) - E(n), both in eV.

| Family | Solvent | n | Binding (eV) | Per solvent (eV) | Desolvation (eV) | Oxygens at the ion | Nearest O (A) |
|---|---|---:|---:|---:|---:|---:|---:|
| carbonate | EC | 1 | -2.55 | -2.55 | - | 1 | 1.66 |
| carbonate | EC | 2 | -4.31 | -2.15 | +1.76 | 2 | 1.71 |
| carbonate | EC | 3 | -5.36 | -1.79 | +1.05 | 3 | 1.79 |
| carbonate | EC | 4 | -6.71 | -1.68 | +1.35 | 4 | 1.90 |
| carbonate | PC | 1 | -2.63 | -2.63 | - | 1 | 1.65 |
| carbonate | PC | 2 | -4.44 | -2.22 | +1.80 | 2 | 1.70 |
| carbonate | PC | 3 | -5.53 | -1.84 | +1.10 | 3 | 1.78 |
| carbonate | PC | 4 | -6.86 | -1.72 | +1.33 | 6 | 1.78 |
| carbonate | DMC | 1 | -2.44 | -2.44 | - | 1 | 1.66 |
| carbonate | DMC | 2 | -4.05 | -2.03 | +1.61 | 2 | 1.70 |
| carbonate | DMC | 3 | -5.07 | -1.69 | +1.02 | 3 | 1.78 |
| carbonate | DMC | 4 | -5.99 | -1.50 | +0.92 | 4 | 1.84 |
| carbonate | DEC | 1 | -2.58 | -2.58 | - | 1 | 1.65 |
| carbonate | DEC | 2 | -4.29 | -2.15 | +1.71 | 2 | 1.69 |
| carbonate | DEC | 3 | -5.42 | -1.81 | +1.13 | 3 | 1.77 |
| carbonate | DEC | 4 | -6.67 | -1.67 | +1.25 | 4 | 1.82 |
| carbonate | EMC | 1 | -2.52 | -2.52 | - | 1 | 1.66 |
| carbonate | EMC | 2 | -4.13 | -2.07 | +1.61 | 2 | 1.70 |
| carbonate | EMC | 3 | -5.28 | -1.76 | +1.15 | 3 | 1.76 |
| carbonate | EMC | 4 | -6.20 | -1.55 | +0.92 | 4 | 1.82 |
| carbonate | FEC | 1 | -2.26 | -2.26 | - | 1 | 1.67 |
| carbonate | FEC | 2 | -3.87 | -1.93 | +1.61 | 2 | 1.72 |
| carbonate | FEC | 3 | -5.11 | -1.70 | +1.24 | 3 | 1.81 |
| carbonate | FEC | 4 | -6.14 | -1.53 | +1.03 | 5 | 1.82 |
| carbonate | VC | 1 | -2.43 | -2.43 | - | 1 | 1.66 |
| carbonate | VC | 2 | -4.12 | -2.06 | +1.70 | 2 | 1.71 |
| carbonate | VC | 3 | -5.20 | -1.73 | +1.08 | 3 | 1.78 |
| carbonate | VC | 4 | -6.42 | -1.61 | +1.22 | 4 | 1.88 |
| ether | DME | 1 | -2.12 | -2.12 | - | 1 | 1.77 |
| ether | DME | 2 | -3.55 | -1.77 | +1.42 | 2 | 1.78 |
| ether | DME | 3 | -4.67 | -1.56 | +1.12 | 3 | 1.87 |
| ether | DME | 4 | -5.98 | -1.49 | +1.31 | 3 | 1.97 |
| ether | DEE | 1 | -3.53 | -3.53 | - | 2 | 1.79 |
| ether | DEE | 2 | -5.58 | -2.79 | +2.06 | 4 | 1.85 |
| ether | DEE | 3 | -5.99 | -2.00 | +0.41 | 4 | 1.97 |
| ether | DEE | 4 | -4.02 | -1.01 | -1.97 | 4 | 1.97 |
| ether | DOL | 1 | -2.08 | -2.08 | - | 2 | 1.97 |
| ether | DOL | 2 | -3.40 | -1.70 | +1.31 | 3 | 1.81 |
| ether | DOL | 3 | -4.55 | -1.52 | +1.15 | 3 | 1.90 |
| ether | DOL | 4 | -5.45 | -1.36 | +0.90 | 4 | 1.88 |
| ether | THF | 1 | -2.16 | -2.16 | - | 1 | 1.73 |
| ether | THF | 2 | -3.70 | -1.85 | +1.54 | 2 | 1.76 |
| ether | THF | 3 | -4.87 | -1.62 | +1.17 | 3 | 1.81 |
| ether | THF | 4 | -5.87 | -1.47 | +0.99 | 4 | 1.87 |
| ether | diglyme | 1 | -4.16 | -4.16 | - | 3 | 1.92 |
| ether | diglyme | 2 | -6.29 | -3.15 | +2.13 | 6 | 1.94 |
| ether | diglyme | 3 | -6.74 | -2.25 | +0.45 | 6 | 1.97 |
| ether | diglyme | 4 | -7.05 | -1.76 | +0.31 | 5 | 1.96 |
| ether | triglyme | 1 | -4.87 | -4.87 | - | 4 | 1.84 |
| ether | triglyme | 2 | -5.46 | -2.73 | +0.59 | 5 | 1.96 |
| ether | triglyme | 3 | -6.29 | -2.10 | +0.83 | 5 | 1.98 |
| ether | triglyme | 4 | -6.56 | -1.64 | +0.27 | 4 | 1.95 |
| ether | tetraglyme | 1 | -2.12 | -2.12 | - | 1 | 1.77 |
| ether | tetraglyme | 2 | -5.58 | -2.79 | +3.46 | 4 | 1.98 |
| ether | tetraglyme | 3 | -5.57 | -1.86 | -0.00 | 4 | 1.91 |
| ether | tetraglyme | 4 | -7.09 | -1.77 | +1.52 | 4 | 1.87 |
| fluorinated ether | FDME | 1 | -2.84 | -2.84 | - | 0 | nan |
| fluorinated ether | FDME | 2 | -4.38 | -2.19 | +1.53 | 3 | 2.09 |
| fluorinated ether | FDME | 3 | -5.47 | -1.82 | +1.10 | 3 | 2.02 |
| fluorinated ether | FDME | 4 | -5.65 | -1.41 | +0.18 | 4 | 1.94 |
| fluorinated ether | TFEE | 1 | -2.31 | -2.31 | - | 1 | 2.09 |
| fluorinated ether | TFEE | 2 | -3.83 | -1.91 | +1.52 | 2 | 1.95 |
| fluorinated ether | TFEE | 3 | -4.96 | -1.65 | +1.13 | 3 | 2.01 |
| fluorinated ether | BTFE | 1 | -3.40 | -3.40 | - | 2 | 1.96 |
| fluorinated ether | BTFE | 2 | -5.29 | -2.65 | +1.89 | 4 | 1.96 |
| fluorinated ether | BTFE | 3 | -6.00 | -2.00 | +0.71 | 3 | 1.98 |
| fluorinated ether | BTFE | 4 | -6.15 | -1.54 | +0.15 | 3 | 1.95 |
| fluorinated ether | TTE | 1 | -3.79 | -3.79 | - | 2 | 1.89 |
| fluorinated ether | TTE | 2 | -4.82 | -2.41 | +1.04 | 3 | 1.89 |
| fluorinated ether | TTE | 3 | -6.01 | -2.00 | +1.19 | 4 | 1.95 |
| fluorinated ether | TTE | 4 | -6.04 | -1.51 | +0.02 | 3 | 1.87 |
| fluorinated ether | HFE | 1 | -2.20 | -2.20 | - | 1 | 1.86 |
| fluorinated ether | HFE | 2 | -3.72 | -1.86 | +1.53 | 2 | 1.87 |
| fluorinated ether | HFE | 3 | -4.60 | -1.53 | +0.87 | 3 | 1.93 |
| fluorinated ether | HFE | 4 | -4.93 | -1.23 | +0.33 | 3 | 1.95 |

1 complexes failed to build or optimise and are named in `series.json`.

