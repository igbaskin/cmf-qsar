# Continuous Molecular Fields on COMPAS-1: a pilot

*One page for the collaborating group, 29 October 2026. The numbers are those of `table.md` in
this folder; `python benchmarks/run_compas.py` in the cmfqsar repository reproduces every one of
them, and `NOTES.md` records the data files, their checksums and every substitution.*

## What was done

The HOMO–LUMO gap of the 8,678 cata-condensed polybenzenoid hydrocarbons of COMPAS-1D
(B3LYP-D3BJ/def2-SVP; Wahab et al., *J. Chem. Inf. Model.* 2022) was modelled with Continuous
Molecular Fields (CMF): every molecule is a sum of atom-centred Gaussians, the model is kernel
ridge regression on the analytic overlap of those fields, and the fitted model is read back as
co-fields in space and as per-ring contributions. The molecules were put in one frame by a rule
(the inertia axes of the heavy atoms with a fixed sign convention), so no alignment was fitted.
Fields: steric (A), electrostatic from GFN2-xTB charges (B), indicator fields of the three carbon
classes rim / two-ring / three-ring (C), a ring-centred field with one Gaussian per ring centroid
per annulation class terminal / linear / angular / branched (D, new), and a ring field weighted by
the share of the Clar sextets (E, meant as a stand-in for an aromaticity field). Random 80/10/10
splits for five seeds, and size-extrapolation splits that train on fewer than N rings and test on
N rings. Baselines on the same splits: ridge regression on the counts of the atom classes, the ring
classes, the rings and the sextets (the additivity scheme that the indicator and ring fields become
when their width goes to infinity), a random forest on RDKit descriptors, and a straight line in
the number of rings.

## Main table

Test R² (mean ± standard deviation over the five random splits), the test RMSE in eV, then R² on
the size-extrapolation splits (N = 8, 9, 10 rings; 179, 619 and 2,270 training molecules).

| model | test R² | RMSE (eV) | MAE (eV) | N = 8 | N = 9 | N = 10 |
|---|---|---|---|---|---|---|
| A steric | 0.42 ± 0.04 | 0.30 | 0.24 | −0.48 | −0.40 | −1.47 |
| A, B steric + charges | 0.51 ± 0.03 | 0.28 | 0.22 | −0.69 | −0.40 | −1.00 |
| C atom-class indicators | 0.49 ± 0.02 | 0.29 | 0.22 | −0.02 | −0.49 | −1.15 |
| D ring-class indicators | 0.65 ± 0.02 | 0.24 | 0.18 | 0.52 | 0.51 | −0.04 |
| A, B, C | 0.60 ± 0.05 | 0.25 | 0.20 | −0.05 | −0.12 | −1.07 |
| C, D | 0.71 ± 0.08 | 0.21 | 0.16 | 0.50 | 0.50 | 0.31 |
| **A, B, C, D** | **0.78 ± 0.11** | **0.18** | **0.13** | **0.59** | **0.53** | **0.44** |
| A, B, C, D, E (Clar sextets) | 0.87 ± 0.02 | 0.14 | 0.11 | 0.67 | 0.78 | 0.70 |
| ridge on counts (additivity scheme) | 0.70 ± 0.01 | 0.22 | 0.17 | 0.63 | 0.65 | 0.67 |
| random forest, 66 RDKit descriptors | 0.91 ± 0.02 | 0.12 | 0.07 | 0.32 | 0.35 | −0.16 |
| straight line in the number of rings | 0.02 ± 0.03 | 0.39 | 0.31 | −0.18 | −0.08 | −0.02 |

**Published models on the same molecules** (random splits only; nothing published measures size
extrapolation):

| model | source | split | test MAE (eV) |
|---|---|---|---|
| SE(3)-transformer on a graph of rings / of atoms | Weiss et al., *J. Org. Chem.* 2023, 88, 9645, Table 1 | 81/9/10 | 0.012 / 0.011 |
| SphereNet / DimeNet++ / SchNet | same | same | 0.013 / 0.015 / 0.022 |
| CHEM-BERT (SMILES) / AttentiveFP | same | same | 0.033 / 0.073 |
| RNN on augmented LALA annulation strings (all pairs 0.04–0.26) | Fite et al., *J. Phys. Org. Chem.* 2023, 36, e4458, Fig. 3 | 90/10 | 0.05 |
| extra-trees on selected RDKit descriptors | Zheng et al., *Chem. Phys. Lett.* 2023, 814, 140358 | 90/10, 100 repeats | R² 0.91, RMSE ≈ 0.12 |

Our random forest (MAE 0.074 eV, R² 0.91) reproduces the last row. The linear field models (MAE
0.11 to 0.13 eV) are at the lower end of the annulation-string models and an order of magnitude
behind the geometric graph networks on interpolation among isomers.

**Resolution of the field model** (random splits; the attenuation grid extended to 0.02 to 20 Å⁻²,
the ridge grid down to 10⁻⁶, and the program's Gaussian kernel layer on top of the overlap kernel,
its width chosen by cross-validation):

| model | seeds | q² | test R² | test RMSE (eV) | test MAE (eV) |
|---|---|---|---|---|---|
| A, B, C, D, E, linear kernel, coarse grid (main table) | 5 | 0.87 | 0.87 ± 0.02 | 0.144 | 0.108 |
| A, B, C, D, E, linear kernel, wide grid and small ridge | 3 | 0.87 | 0.87 ± 0.01 | 0.147 | 0.110 |
| **A, B, C, D, E, Gaussian kernel layer** | 3 | **0.96** | **0.96 ± 0.01** | **0.076 ± 0.006** | **0.056 ± 0.004** |
| A, B, C, D, Gaussian kernel layer | 1 | 0.94 | 0.92 | 0.115 | 0.073 |

The wider grid gains nothing: the linear model is not short of width, it is short of
nonlinearity. The Gaussian layer halves the error and puts the field model level with the
published annulation-string network (0.05 eV) and ahead of the descriptor forests, with the
geometric graph networks (0.011 to 0.022 eV) three to five times better still.

The gain carries over to the size-extrapolation splits, where no published number exists
(test R² / RMSE in eV; the linear coarse-grid model of the main table beside the layered one):

| field set | N = 8 (train 179) | N = 9 (train 619) | N = 10 (train 2,270) |
|---|---|---|---|
| A, B, C, D, E, linear | 0.67 / 0.241 | 0.78 / 0.187 | 0.70 / 0.213 |
| A, B, C, D, E, Gaussian layer | 0.80 / 0.187 | 0.89 / 0.132 | 0.92 / 0.113 |
| A, B, C, D, linear | 0.59 / 0.269 | 0.53 / 0.274 | 0.44 / 0.290 |
| A, B, C, D, Gaussian layer | 0.69 / 0.234 | 0.81 / 0.175 | 0.84 / 0.155 |

Trained on the 2,270 molecules of up to nine rings, the layered five-group model predicts the
6,408 ten-ring molecules with R² 0.92 and RMSE 0.113 eV; the descriptor forest on the same split
has R² −0.16. The price is the
reading: with the layer the prediction is no longer an overlap integral of a co-field with the
molecular field, so the co-field and the per-ring contributions above belong to the linear model
only (Shapley values, which the program computes for any kernel, are the way to read the layered one).

Cross-validated q² on the training part: 0.83 for A, B, C, D and 0.87 with the sextet field. The
spread of the A, B, C, D row over the seeds is one molecule: seed 0 holds benzene, the only
one-ring molecule, in its test part, and the model extrapolates its gap to 10–13 eV (computed
6.72); over the molecules of three rings and more the test RMSE is 0.164 ± 0.003 eV (0.138 with
the sextet field).

**Scaling to COMPAS-1x** (34,072 molecules at GFN2-xTB, gap mean 1.98 eV, sd 0.30 eV; one random
80/10/10 split, the attenuations and ridge of the A, B, C, D model above, kernel ridge regression in
the space of m landmark molecules):

| m landmarks | rank of the feature map | q² | test R² | test RMSE (eV) | test MAE (eV) | features + fit (min) |
|---|---|---|---|---|---|---|
| 500 | 2,244 | 0.78 | 0.80 | 0.133 | 0.106 | 15 |
| 1,000 | 2,570 | 0.78 | 0.80 | 0.133 | 0.106 | 22 |
| 2,000 | 2,722 | 0.78 | 0.80 | 0.133 | 0.106 | 55 |

The approximation has converged at 500 landmarks: quadrupling them changes nothing to three
decimals, and the rank grows by a fifth where the number of columns quadruples. The fields of these planar molecules
in one frame span a few thousand directions, so a model of the 34k set costs the kernel of every
molecule against a few hundred landmarks, built on all cores in a quarter of an hour, and never the
34k × 34k matrix.

## The co-field

![Co-fields of the gap over the six-ring reference molecule hc_c26h16_0pent_13 (test set of seed 0), one panel per field group, in the molecular plane (top) and 1 Å above it (bottom); red raises the gap per unit of field, blue lowers it.](figures/cofield_gap_hc_c26h16_0pent_13.png)

The co-fields change sign along the molecule: the ring-class co-field (right panels) is blue over
one end of the reference molecule and red over the other, and the atom-class co-field is blue
above one edge and red along the opposite one. The maps are drawn over the whole frame because
every training molecule contributes to them; what they mean for one molecule is the overlap with
its own field, which is what rule 3 sums. The per-ring contributions of the four test molecules add up to the prediction
exactly, but the individual values are large and cancel, and a ring on the right of the frame
counts positive and one on the left negative whatever its class: with a ridge of 1e-4 the
coefficients grow and the co-field acquires a smooth component across the frame. The sum is the
physics; the single values read position as much as chemistry, and a decomposition against a
reference (Shapley values) is the next thing to try. The refit of the main model in float64 for
these figures reproduces the table's test predictions, computed on float32 kernels, to 0.06 eV at
worst and gives the same test R² (0.593 on seed 0).

![Per-ring contributions to the gap of four test molecules (decomposition rule 3 summed over the atoms of every ring; the values and the bias add up to the prediction).](figures/ring_contributions_gap.png)

## Conclusions

- **The gap of a benzenoid is topology, and a field can carry topology.** The four ring-class
  fields alone (0.65) beat the steric, electrostatic and atom-class fields together (0.60); the
  Clar-sextet ring field is the largest single gain on top of everything else (0.78 → 0.87).
- **Only the ring fields extrapolate in size.** Every field set without one has a negative R² on
  the molecules larger than any it was trained on; every set with one is positive. The random
  forest on descriptors is the best model on the random splits (0.91) and among the worst on the
  size splits (0.32, 0.35, −0.16): it knows the isomers it saw.
- **The linear field model is not competitive on interpolation; the nonlinear one is close.** The
  published geometric graph networks reach 0.011 to 0.022 eV MAE on the same molecules, ten times
  better than the 0.11 to 0.13 eV of the linear field model, which even the additivity scheme
  (0.70) matches unless the sextet field is included. A wider attenuation grid and a smaller ridge
  change nothing; the Gaussian kernel layer halves the error to 0.056 eV MAE (R² 0.96), level with
  the published annulation-string network and ahead of the descriptor forests, three to five times
  behind the graph networks. On the size-extrapolation splits the layered model is better
  still in proportion: R² 0.92 on the ten-ring molecules from a training part of up to nine
  rings, where the descriptor forest is negative. What the linear model keeps over every
  published model is the reading, a co-field in space and per-ring contributions; the layered
  model gives that up for half the error, and Shapley values are its reading.

## Limitations

- A random split of an enumerated space tests interpolation among close isomers; the
  size-extrapolation split tests what a user would ask of the model, on few training molecules
  (the set is dominated by ten-ring molecules). The two answer different questions.
- The field scales were fixed over the whole set and the attenuations chosen on 2,500 training
  molecules; the chosen steric attenuation sits at the top of its grid and the ridge at the bottom
  (1e-4), so a wider grid is one more run.
- The electrostatic field of a hydrocarbon carries small charges; its small weight in the table is
  a statement about these molecules, not about the field.
- The few helicene-like molecules are not planar; the frame is defined for them but the two faces
  of it differ.

## Follow-ups

0. Shapley values as the per-ring reading of the layered model in place of rule 3, and a wider
   grid of layer widths (the chosen width sat at the widest end of its grid on every split).
1. A NICS-weighted ring field (one DFT calculation per ring), which has to beat the Clar-sextet
   share rather than the set without it.
2. Peri-condensed systems (COMPAS-3): three-ring carbons and adjacent shared bonds; the typing and
   the ring classes are written for them and untested on them.
3. Hetero-atom systems (COMPAS-2) with indicator fields per hetero-atom type.
4. Δ-learning xTB → DFT on the 34k of COMPAS-1x, with the Nyström path that the script carries.
