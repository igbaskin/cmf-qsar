# Measured critical micelle concentrations

Measured critical micelle concentrations (CMC) of surfactants, each surfactant ion with one 3D
structure. The value is pCMC, the negative decimal logarithm of the CMC in mol/L: a surfactant
that forms micelles at a lower concentration has a higher pCMC. The sets are made by
`benchmarks/prepare_cmc.py` and read by `benchmarks/run_cmc.py`.

| file | what it holds |
| --- | --- |
| `train.sdf` | the training part of SurfPro: 1,203 surfactants, every row of `surfpro_train.csv` with a pCMC |
| `test.sdf` | the test set of the SurfPro paper: 140 surfactants, drawn with stratification by surfactant type |
| `LICENSE_SurfPro.md` | the licence of the SurfPro data, CC BY-NC-SA 4.0, as the SurfPro repository gives it |
| `provenance.json` | the source commit, the MD5 checksums, the types and temperatures, and how many rows were refused and why |

Every record carries `pCMC`, `smiles` (the surfactant ion), `salt_smiles` (the salt as the
source writes it), `counterion` (empty for a non-ionic or zwitterionic surfactant),
`surfactant_type`, `temperature` (C) and `inchikey14` (the first block of the InChIKey of the
salt). The training records also carry `fold`, the cross-validation fold of the paper.

**Source.** SurfPro: S. L. Hoedl, L. Hermans, P. F. J. Dankloff, A. Piruska, W. T. S. Huck,
W. E. Robinson, "SurfPro - a curated database and predictive model of experimental properties of
surfactants", *Digital Discovery* 4 (2025) 1176, doi:10.1039/D4DD00393D. The files
`data/surfpro_train.csv` and `data/surfpro_test.csv` of github.com/BigChemistry-RobotLab/SurfPro
at commit 35b8906 (20 May 2026). This is the second version of the data (SurfProV2), in which the
authors checked the structures against the primary papers and corrected many of them after the
paper was published.

**Licence.** The SurfPro data are distributed under CC BY-NC-SA 4.0. The files of this folder are
adapted from them and are under the same licence: they may be shared and adapted for
non-commercial use, with attribution to the paper above, and an adapted version must carry the
same licence. This differs from the MIT licence of the program. The licence text is in
`LICENSE_SurfPro.md`, copied from the SurfPro repository at the same commit.

**Published accuracy on the same test set.** The paper reports, for pCMC on the 140 test
surfactants (Table 3, an ensemble of the ten models of its cross-validation folds): MAE 0.235 and
RMSE 0.346 for its best graph neural network (AttentiveFP trained on all properties), MAE 0.250
and RMSE 0.382 for the same network trained on pCMC alone, and MAE 0.630 and RMSE 0.840 for an
RDKit fingerprint with a random forest. These numbers are for the first version of the data.

**What was changed.**

- A row of the source is a salt. The 3D structure is that of the surfactant ion alone, with its
  net charge; the counterions (Na+, K+, Li+, NH4+, Cl-, Br-, I-) are kept in `counterion`.
- The rows without a pCMC are left out (131 rows of the training file).
- One surfactant ion with 319 heavy atoms is left out; the next largest has 135.
- There is no limit on the molecular weight below that. 166 training surfactants and 16 test
  surfactants weigh more than 700, mostly gemini and non-ionic surfactants.
- A second row of the same salt (the same isomeric SMILES) is left out. Stereoisomers are kept:
  they are different compounds, although the first block of the InChIKey does not tell them
  apart.

**Structures.** One conformer per surfactant ion: RDKit ETKDGv3 with seed 42, then MMFF94
(`cmfqsar.align.embed_3d`), as for the other physical property sets. A surfactant has a long
flexible tail, often several, and one conformer is one arbitrary choice among very many.

**What to keep in mind.**

- Most values are at 25 C, but not all: the temperatures are between 20 and 40 C. The values are
  used as they are, as in the paper.
- The counterion changes the CMC of an ionic surfactant, but the 3D structure does not carry
  it. In `benchmarks/run_cmc.py` only the Klevens descriptors see it.
- The set is not balanced: 438 of the training surfactants are gemini cationic surfactants, and
  many belong to homologous series that differ only in the length of the tail.
- No salt of the test set is in the training part. Three test salts are stereoisomers of a
  training salt, and six test ions are in the training part with another counterion or another
  stereochemistry.
