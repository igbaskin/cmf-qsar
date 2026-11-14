# Chiral phosphoric acid catalysts with a measured enantioselectivity (plan item PO1)

Built by `examples/make_denmark_examples.py --fetch` from the supplementary data of

> Zahrt A. F., Henle J. J., Rose B. T., Wang Y., Darrow W. T., Denmark S. E. "Prediction of
> higher-selectivity catalysts by computer-driven workflow and machine learning", Science 363
> (2019) eaau5631.

as redistributed in the DeepReac repository, <https://github.com/bm2-lab/DeepReac>, folder
`Data/DatasetC` (the repository's own code is under the Apache 2.0 licence; the data files are
the paper's). The status of this copy is therefore **unclear** in the sense of
`docs/public_3dqsar_datasets.md`: the facts are the paper's, redistribution is customary in the
field but not explicitly permitted, and the script rebuilds every file here from the original
fragments, whose joint SHA-256 it records. The fragments themselves stay in `upstream/`, which
is not kept in the repository.

## The series

43 BINOL-derived chiral phosphoric acids, each run against the 25 combinations of 5 N-acyl
imines and 5 thiols, forming N,S-acetals: 1,075 reactions, each with its enantiomeric excess.
Every catalyst is the same phosphoric acid on one of two scaffolds (`i`, the binaphthyl;
`vi`, its octahydro form) with one substituent at both the 3 and 3' positions; 38 substituents
occur: 32 catalysts are on `i` and 11 on `vi`.

The published target is the free-energy difference between the two diastereomeric transition
states, `ddG = R T ln((1 + ee) / (1 - ee))` at 298.15 K, in kcal/mol. The paper reports a mean
absolute deviation of 0.161 kcal/mol for new substrate combinations of training catalysts and
0.211 kcal/mol for catalysts outside the training set, from a model trained on 600 reactions of
24 catalysts.

## The files

| File | Content |
|---|---|
| `catalysts.sdf` | the 43 catalysts, one conformer each, in the frame of the phosphoric acid; properties `catalyst`, `substituent`, `scaffold`, `smiles`, `mmff_energy`, `conformers_tried`, `biaryl_dihedral`, `force_field` |
| `catalysts_sym.sdf` | the same, with the two images under the catalyst's C2 axis as two records of one name (`symmetry_image` C2+ and C2-), for the ensemble mode |
| `imines.sdf`, `thiols.sdf` | the 5 imines (named 1 to 5) and the 5 thiols (A to E), lowest MMFF conformer, not in any frame |
| `reactions.csv` | the 1,075 reactions: `reaction` (the source's row name), `catalyst`, `substituent`, `scaffold`, `imine`, `thiol`, `ee_percent`, `ddG_kcal` |

## How the catalysts were built

Each catalyst is assembled from its scaffold and two copies of its substituent, bonded at the
ring carbon next to the phosphate oxygen that is not the biaryl carbon. The molecule is embedded
with ETKDG (30 conformers), each conformer is minimised with MMFF94, and the lowest is kept.

One catalyst, `181_i`, has a pentafluorosulfanyl group on each of its two aryl substituents, and
no force field of RDKit holds that group. MMFF94 has no parameters for it. UFF, which the builder
used for this catalyst until 14 November 2026, moved the fluorines 6 to 8 A away from the sulfur,
put two of them on one point and opened the biaryl dihedral to 71 degrees, where the other 31
catalysts of the scaffold have 54 to 59. The builder now finds the conformers of this catalyst
with a CF3 group in place of each SF5 group, with MMFF94 as for the other 42, and then sets each
SF5 group to a regular octahedron on its ring carbon (S-F 1.59 A, C-S 1.80 A; `regular_sf5` of
`make_hammett_examples.py`). Its biaryl dihedral is 54.5 degrees. The property `force_field` of
the record says so, and `mmff_energy` is the energy of the stand-in with CF3. The other 42
records did not change. `tests/test_denmark_example.py` checks every bond length and every
close contact of the four structure files.

A 2D file does not carry the axial configuration of the binaphthyl, and ETKDG assigns it at
random, so every conformer is reflected where needed until the biaryl dihedral C2-C1-C1'-C2' is
positive. Which enantiomer that is cannot be known from the source; what the model needs is that
all 43 are the same one, as they were in the screen, so that the sign of ddG means the same
thing for every catalyst.

**The frame.** Phosphorus at the origin; z from the midpoint of the two ring oxygens through
the phosphorus, so that the P=O and P-OH oxygens point to +z; x along the line from one ring
oxygen to the other; y completing the right-handed set. The 3,3'-substituents then flank the
+z pocket in which the imine and the thiol meet. The catalyst has a C2 axis along z, so which
ring oxygen is on +x is an arbitrary choice; the single-conformer file puts the hydroxyl oxygen
on +y, and the symmetrised file carries both images, which the ensemble mode averages. Averaging
the two images also averages the P=O and P-OH oxygens, which the acid exchanges far faster than
the reaction proceeds.

## Reading the numbers

ddG runs from -0.42 to +3.14 kcal/mol (mean 0.99, standard deviation 0.70). The benchmark
`benchmarks/run_denmark.py` reads this folder; its results are in `benchmarks/results/denmark/`.
