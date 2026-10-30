# Continuous Molecular Fields on COMPAS-1: a pilot

*One page for the collaborating group, 29 October 2026, with the four follow-ups measured
on 30 October. The numbers are those of `table.md` in
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

## The four follow-ups, measured (30 October 2026)

The four follow-ups of the first report were run the next day with the same protocol (random
80/10/10 splits for three seeds, the size-extrapolation splits, the attenuations and the ridge
chosen by cross-validation, the Gaussian kernel layer as a third configuration). Every number is
in `table.md`; `NOTES.md` lists the data files and the choices.

### A geometric aromaticity field, and NICS as its calibration

A NICS field for the whole set would cost one DFT calculation per molecule with a ghost atom
above every ring: 55 s for naphthalene and 5.5 min for a six-ring molecule with the serial ORCA
of this machine, about a week for the 8,678 molecules. The field that was built instead costs
nothing: the **HOMA index** of every ring (harmonic-oscillator model of aromaticity, Krygowski
1993, from the bond lengths of the DFT geometry, 1 for benzene and below 0 for a ring with
alternating bonds) as the weight of a ring-centred field, group F. NICS(1) and NICS(1)zz were then
computed for the 179 molecules of up to seven rings (B3LYP-D3BJ/def2-SVP, GIAO, one ghost
hydrogen 1 Å above every centroid) to see how far the geometric index stands in for the magnetic
one.

The answer on the 1,158 rings: the HOMA index follows NICS(1)zz with r = −0.77 (NICS(1):
−0.75), the Clar-sextet share only with r = −0.50, so the geometric index is the better
stand-in of the two. By ring class the two measures disagree on one point. The linear rings
are the most shielded (mean NICS(1)zz −30.1 ppm, against −25.3 for a terminal ring, −18.3
angular and −8.4 branched) and not the most bond-equalised (HOMA 0.68 against 0.79 for a
terminal ring): the ring-current picture of an acene, which a NICS field would weight more
than the HOMA field does. The 179 molecules took 8.9 hours with ten serial ORCA jobs.

Across the rings of COMPAS-1D the HOMA index follows the Clar-sextet share (r = 0.82; mean HOMA
0.80 for a terminal ring, 0.70 linear, 0.48 angular, 0.11 branched), so the two ring fields carry
the same ordering, one as a step function and one as a continuous one. In the model they are not
the same (test metrics over three seeds, then R² / RMSE on the size splits):

| model | test R² | RMSE (eV) | MAE (eV) | N = 8 | N = 9 | N = 10 |
|---|---|---|---|---|---|---|
| A, B, C, D, E (Clar), linear | 0.87 ± 0.02 | 0.144 | 0.108 | 0.67 / 0.241 | 0.78 / 0.187 | 0.70 / 0.213 |
| A, B, C, D, F (HOMA), linear | 0.84 ± 0.12 | 0.153 | 0.097 | 0.72 / 0.224 | 0.78 / 0.187 | 0.83 / 0.163 |
| A, B, C, D, E, F, linear | 0.88 ± 0.08 | 0.135 | 0.085 | 0.76 / 0.208 | 0.82 / 0.170 | 0.84 / 0.154 |
| A, B, C, D, E, Gaussian layer | 0.96 ± 0.01 | 0.076 | 0.056 | 0.80 / 0.187 | 0.89 / 0.132 | 0.92 / 0.113 |
| A, B, C, D, F, Gaussian layer | 0.94 ± 0.04 | 0.100 | 0.062 | 0.77 / 0.201 | 0.83 / 0.164 | 0.88 / 0.135 |
| **A, B, C, D, E, F, Gaussian layer** | **0.96 ± 0.02** | **0.077** | **0.050** | **0.81 / 0.184** | **0.89 / 0.133** | **0.93 / 0.105** |
| ridge on the counts, with the sum of the HOMA indices added | 0.80 ± 0.02 | 0.179 | 0.138 | 0.75 / 0.210 | 0.75 / 0.202 | 0.77 / 0.188 |

The spread of the HOMA rows over the seeds is again benzene in the test part of seed 0; by the MAE
and on the size splits the HOMA field beats the Clar field in the linear model (0.097 against
0.108 eV; R² 0.83 against 0.70 on the ten-ring molecules), and the two together are the best
linear model of the study. Under the Gaussian layer the Clar field alone is as good as both. The
additivity scheme gains most of all: the sum of the HOMA indices added to the ten counts takes the
ridge on counts from 0.70 to 0.80, which says that a large part of what the HOMA field carries is
a per-molecule sum, and that the field is worth 0.05 eV of MAE on top of that sum.

### Δ-learning from the GFN2-xTB gap

COMPAS-1x holds the GFN2-xTB gap of every COMPAS-1D molecule. A straight line in it, fitted on
the training part of every split, leaves a residual of 0.047 eV RMSE and 0.036 eV MAE (R² 0.986),
better than every field model of the first report and than every published model but the
geometric graph networks. The field model was then trained on that residual (the `corrected`
columns of COMPAS-1x were not used: they are fitted to another functional and sit 2.2 eV from the
B3LYP gaps). On the gap scale, three seeds:

| model | RMSE (eV) | MAE (eV) | N = 8 RMSE | N = 9 RMSE | N = 10 RMSE |
|---|---|---|---|---|---|
| line in the xTB gap alone | 0.047 ± 0.001 | 0.036 ± 0.001 | 0.048 | 0.047 | 0.045 |
| A, B, C, D, E on the residual, linear | 0.035 ± 0.006 | 0.025 ± 0.001 | 0.053 | 0.040 | 0.043 |
| **A, B, C, D, E on the residual, Gaussian layer** | **0.026 ± 0.002** | **0.018 ± 0.001** | 0.045 | 0.033 | 0.030 |

The layered field model explains 68 % of the variance of the residual and halves the error of
the line. At 0.018 eV MAE the Δ-model stands inside the range of the published geometric graph
networks (0.011 to 0.022 eV) for the price of one xTB calculation per molecule (0.3 s). The
multi-task form (one model of the two levels) was not run; the residual form is the standard one
and the xTB value exists for every molecule.

### COMPAS-3D: the peri-condensed set

COMPAS-3D (Wahab and Gershoni-Poranne, *Phys. Chem. Chem. Phys.* 2024, 26, 15344) holds 8,844
peri-condensed polybenzenoid hydrocarbons of four to ten rings (6,891 with ten) at
CAM-B3LYP-D3BJ/def2-SVP, gap mean 4.99 eV, sd 0.44 eV. The same reader, frame and fields apply,
with two additions that the first report had written and not tested: the three-ring carbon
`C.3r` is present, and a ring two of whose shared bonds touch, or fused to four or more rings, is
a fifth annulation class, `peri` (44,187 of the 85,960 rings; every molecule has at least one).
The Clar-sextet and HOMA fields are defined as before. Three seeds and the size splits (87, 416
and 1,953 training molecules):

| model | test R² | RMSE (eV) | MAE (eV) | N = 8 | N = 9 | N = 10 |
|---|---|---|---|---|---|---|
| D ring classes, linear | 0.46 ± 0.01 | 0.328 | 0.256 | −0.66 / 0.656 | −0.06 / 0.483 | −0.15 / 0.457 |
| A, B, C, D, linear | 0.61 ± 0.01 | 0.279 | 0.217 | 0.13 / 0.475 | 0.32 / 0.387 | 0.22 / 0.377 |
| A, B, C, D, E, linear | 0.83 ± 0.01 | 0.186 | 0.147 | 0.73 / 0.266 | 0.71 / 0.253 | 0.70 / 0.234 |
| A, B, C, D, E, F, linear | 0.90 ± 0.00 | 0.144 | 0.113 | 0.62 / 0.316 | 0.78 / 0.219 | 0.80 / 0.193 |
| A, B, C, D, Gaussian layer | 0.84 ± 0.02 | 0.180 | 0.133 | 0.44 / 0.381 | 0.57 / 0.308 | 0.58 / 0.277 |
| A, B, C, D, E, Gaussian layer | 0.94 ± 0.00 | 0.112 | 0.083 | 0.70 / 0.280 | 0.81 / 0.204 | 0.82 / 0.180 |
| **A, B, C, D, E, F, Gaussian layer** | **0.95 ± 0.00** | **0.097** | **0.072** | **0.75 / 0.255** | **0.83 / 0.191** | **0.86 / 0.159** |
| ridge on 11 counts with the HOMA sum (additivity scheme) | 0.79 ± 0.01 | 0.203 | 0.163 | 0.84 / 0.206 | 0.79 / 0.215 | 0.75 / 0.212 |
| random forest on 65 RDKit descriptors | 0.84 ± 0.01 | 0.180 | 0.129 | −0.54 / 0.632 | −1.84 / 0.791 | −2.26 / 0.770 |
| line in the xTB gap | 0.98 ± 0.00 | 0.057 | 0.044 | 0.99 / 0.060 | 0.99 / 0.057 | 0.98 / 0.057 |

The picture of the cata-condensed set repeats. The ring classes alone are weaker here (0.46
against 0.65), because five classes over mostly peri rings say less than four classes over a
chain; the Clar field is again the largest gain (0.61 to 0.83 linear), the HOMA field adds on top
of it on this set (0.83 to 0.90), and the layer takes the best set to 0.097 eV RMSE and 0.072 eV
MAE, a quarter worse than on COMPAS-1D for a target of a tenth more spread. On the
size-extrapolation splits the layered six-group model predicts the 6,891 ten-ring molecules from
1,953 smaller ones with R² 0.86, where the descriptor forest is at −2.3 and the additivity scheme
at 0.75. No published model on COMPAS-3D is used for comparison here. The reading of the linear model does not survive this set: the four-group model refitted for
the figures (seed 0, ridge 10⁻⁴) has rule-2 contributions of ±5·10⁴ eV that cancel to a
prediction, and in float64 it moves one test prediction by 0.89 eV against the float32 run
(0.06 eV on COMPAS-1D). The per-ring figure (`figures/ring_contributions_gap_3D.png`) is drawn
and is not a reading; the Shapley values of the layered model are the reading for this set.

### COMPAS-2D: hetero atoms

COMPAS-2D (Mayo Yanes, Chakraborty and Gershoni-Poranne, *Sci. Data* 2024, 11, 97) holds 52,000 neutral
cata-condensed poly(hetero)cyclic systems of up to ten rings at CAM-B3LYP-D3BJ/def2-SVP, with
nitrogen, oxygen, sulfur and boron, and rings of four to six atoms (benzene, pyridine, pyrazine,
borinine, 1,4-diborinine, 1,2-dihydro-1,2-diborinine, pyrrole, furan, thiophene, borole,
cyclobutadiene). A random subset of 10,000 (gap mean 4.76 eV, sd 0.68 eV) keeps the stored-kernel
protocol of COMPAS-1D. The atom-class fields gain the hetero types N, N.H, O, S, B and B.H; the
ring-class fields keep the four annulation classes; a new group R holds one ring-centred field per
ring type; the HOMA field is defined for every ring without a bond to boron (weight 0 there); no
Clar field, which is undefined for hetero rings. Three seeds and the size splits (2,910, 5,893
and 8,880 training molecules):

| model | test R² | RMSE (eV) | MAE (eV) | N = 8 | N = 9 | N = 10 |
|---|---|---|---|---|---|---|
| C atom classes, linear | 0.44 ± 0.01 | 0.507 | 0.404 | 0.33 / 0.534 | 0.36 / 0.516 | 0.27 / 0.518 |
| D ring classes, linear | 0.17 ± 0.01 | 0.620 | 0.498 | −0.34 / 0.755 | −0.06 / 0.664 | −0.13 / 0.644 |
| R ring types, linear | 0.42 ± 0.01 | 0.519 | 0.411 | 0.36 / 0.521 | 0.36 / 0.514 | 0.30 / 0.507 |
| A, B, C, D, linear | 0.46 ± 0.00 | 0.499 | 0.399 | 0.38 / 0.515 | 0.41 / 0.494 | 0.28 / 0.515 |
| A, B, C, D, R, linear | 0.47 ± 0.02 | 0.494 | 0.395 | 0.41 / 0.501 | 0.42 / 0.490 | 0.32 / 0.500 |
| A, B, C, D, F, R, linear | 0.53 ± 0.02 | 0.465 | 0.366 | 0.48 / 0.473 | 0.48 / 0.464 | 0.40 / 0.471 |
| C, Gaussian layer | 0.55 ± 0.01 | 0.454 | 0.361 | 0.45 / 0.485 | 0.48 / 0.466 | 0.47 / 0.442 |
| A, B, C, D, R, Gaussian layer | 0.64 ± 0.02 | 0.409 | 0.322 | 0.56 / 0.436 | 0.56 / 0.429 | 0.54 / 0.411 |
| **A, B, C, D, F, R, Gaussian layer** | **0.69 ± 0.02** | **0.379** | **0.297** | **0.61 / 0.410** | **0.59 / 0.413** | **0.60 / 0.384** |
| ridge on 24 counts with the HOMA sum (additivity scheme) | 0.51 ± 0.01 | 0.477 | 0.372 | 0.45 / 0.483 | 0.45 / 0.479 | 0.36 / 0.486 |
| random forest on 109 RDKit descriptors | 0.56 ± 0.02 | 0.449 | 0.349 | 0.48 / 0.471 | 0.46 / 0.476 | 0.43 / 0.457 |

The hetero-atom set is where the frame by rule stops working. The best model explains 69 % of
the variance (0.38 eV RMSE on a spread of 0.68 eV) where the benzenoid sets reach 95 %, and the
chosen parameters say why: on every combined set the steric attenuation sits at the narrow end of
its grid (5 Å⁻²) and the ridge at 0.1, a hundred times the ridge of the benzenoid sets or more, which turns the steric kernel into a
near-identity and leaves the fit to the indicator fields. The inertia axes depend on the atomic
masses, so a sulfur or a boron at the rim moves the frame, and two molecules of the same skeleton
with different hetero atoms no longer lie on top of each other; the atom types alone (C, 0.55
under the layer) and the ring types alone (0.48) carry nearly all of what the full set carries.
The ring-type field R is worth little beside the atom types (0.62 to 0.64), and the HOMA field
is the one clear gain (0.64 to 0.69): a hetero ring's bond lengths say what its type and its
atoms do not. The fields still lead both baselines: the descriptor forest reaches 0.56 and the
additivity scheme on the 24 counts 0.51, against 0.69 for the layered model, and on the ten-ring
molecules 0.43 and 0.36 against 0.60. The forest does not collapse on the size splits of this set
(0.43 on the ten-ring molecules, against −2.3 on COMPAS-3D).

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
- **The four follow-ups, measured.** A ring field weighted by the HOMA index beats the Clar
  field in the linear model, and the two together under the layer reach 0.050 eV MAE (R² 0.93
  on the ten-ring molecules). Δ-learning on the residual of a straight line in the GFN2-xTB
  gap reaches 0.018 eV MAE, inside the range of the published graph networks, for one xTB
  calculation per molecule. The peri-condensed set COMPAS-3D repeats the picture (0.072 eV MAE,
  R² 0.86 on the ten-ring molecules). The hetero-atom set COMPAS-2D is where the frame by rule
  stops working (R² 0.69): the inertia frame depends on the masses, and a frame from the
  skeleton is the next thing to try.

## Limitations

- A random split of an enumerated space tests interpolation among close isomers; the
  size-extrapolation split tests what a user would ask of the model, on few training molecules
  (the set is dominated by ten-ring molecules). The two answer different questions.
- The field scales were fixed over the whole set and the attenuations chosen on 2,500 training
  molecules; the chosen steric attenuation sits at the top of its grid and the ridge at the bottom
  (1e-4), so a wider grid is one more run.
- The electrostatic field of a hydrocarbon carries small charges; its small weight in the table is
  a statement about these molecules, not about the field.
- Most of the molecules are not planar: 89 % have a heavy atom more than 0.5 Å from their best
  plane (median of the largest distance 1.48 Å), a fjord in most ten-ring isomers. The frame is
  defined for them, the two faces of it differ, and a mirror image lands in the mirrored frame,
  so the model is not symmetric under reflection although the gap is.

## Follow-ups

0. Shapley values as the per-ring reading of the layered model in place of rule 3, and a wider
   grid of layer widths (the chosen width sat at the widest end of its grid on every split).
1. The NICS field itself, on a cluster, and HOMA parameters for bonds to boron (the section
   *The four follow-ups, measured* has the calibration on 179 molecules).
2. A frame from the skeleton rather than from the masses, for the hetero-atom set, and then
   the full 52,000 molecules of COMPAS-2 through the landmark path.
3. A kernel averaged over the two faces of the frame (the z-flip of one molecule), so that
   a mirror image gets the same prediction.
4. The multi-task form of Δ-learning (`cmfqsar/multitask.py`) beside the residual form.
