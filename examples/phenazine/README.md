# Phenazine derivatives with a computed reduction potential (plan item EC1)

69 substituted phenazines with their one-electron reduction potential in 1,2-dimethoxyethane,
from the data repository of Ghule, Dash, Bagchi, Joshi and Vanka, *Predicting the Redox
Potentials of Phenazine Derivatives Using DFT-Assisted Machine Learning*, ACS Omega 7 (2022)
11742-11755, <https://github.com/siddharth-ncl-work/ml_redox_potential-DATA>. The potential is
computed, not measured.

| File | Content |
|---|---|
| `phenazine.sdf` | the 69 molecules in the geometry of the source, superimposed on the phenazine core, with the potential in `E0_V` |
| `phenazine_qm.sdf` | the same, with the quantum properties of plan item EC1 as per-atom and per-molecule columns (ORCA, B3LYP-D3BJ/def2-SVP in the continuum solvent, at the geometry of the file) |

`examples/make_phenazine_examples.py` builds both and describes the three faults of the source
files that it repairs (a missing version string, a missing end line, and a double bond on the
wrong oxygen of two sulfonic acids). The distributor states no licence, so the source files are
fetched and not kept here.

## Two source structures with atoms on top of each other

The geometry files of the source are drawn structures. They were not optimised before they were
distributed, and this repository does not optimise them either: the quantum properties are
computed at the geometry of the file. For 67 of the 69 molecules that geometry is a reasonable
structure. For two it is not. The geometry check of this repository
(`cmfqsar.orca.geometry_faults`: no two atoms without a bond closer than 1.3 Å) finds:

| Record | Closest pair | What it is |
|---|---|---|
| `three_groups_7_CH2=CH_CF3_COOH_R2_R3_R1` | H24–F27, 0.86 Å | a hydrogen of the vinyl group lies on a fluorine of the neighbouring CF3 group |
| `three_groups_11_COOH_P_CH3_2_C6H5_R2_R8_R7` | H31–C34, 0.97 Å | a hydrogen of a methyl group on phosphorus lies on a carbon of the neighbouring phenyl ring |

In both, two substituents on neighbouring ring positions were drawn in one plane. The same pairs
are in the source files, with the same distances (checked on 14 November 2026).

The coordinates are kept, because the series is the source's series. Two consequences are
stated here so that a reader can judge them:

* The quantum properties of these two records in `phenazine_qm.sdf` (charges, Fukui indices,
  spin populations, orbital energies) were computed on a structure in which two atoms overlap.
  They are not properties of a real conformation of the molecule.
* `benchmarks/run_phenazine.py` was run once more without the two records, as a test of how much
  they matter (14 November 2026; the folds of 67 molecules are not the folds of 69, so the two
  runs are not a paired comparison). The mean absolute error of the five ordinary fields with
  random folds goes from 0.158 V to 0.141 V, and that of every other field model changes by
  less than 0.02 V. No field model becomes better than the five ordinary fields, which is the
  answer of the item, and it does not change. One reading does change: with random folds the
  quantum fields and the spin field are then better than the electron affinity alone by an
  interval that excludes zero (+0.050 V [+0.007, +0.093] and +0.048 V [+0.005, +0.091]), where
  the stored table (`benchmarks/results/phenazine/table.md`) has intervals that include zero
  (+0.043 V [−0.003, +0.090] and +0.033 V [−0.010, +0.075]). With one source table held out the
  intervals of these two comparisons include zero in both runs.

The two records (four, counting both files) are listed in `examples/geometry_exceptions.csv`.
