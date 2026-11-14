# The AMPA / glycine-NMDA selectivity set of 2003

The data of Baskin, Tikhonova, Palyulin and Zefirov, *Selectivity fields: comparative molecular
field analysis (CoMFA) of the glycine/NMDA and AMPA receptors*, J. Med. Chem. 46 (2003)
4063-4069: 57 tricyclic antagonists (52 training, 5 test) with inhibition constants against
[3H]AMPA and [3H]glycine binding.

| File | Content |
|---|---|
| `ampa_glycine.sdf` | the 57 compounds superimposed on the tricyclic core, with `pKi_AMPA`, `pKi_glycine` and their difference |
| `docked_ampa.sdf` | the same compounds as docked poses in the frame of the AMPA receptor model |
| `docked_glycine.sdf` | the same compounds as docked poses in the frame of the glycine-site model |
| `receptor_ampa.pdb`, `receptor_glycine.pdb` | the two receptor models |

The three-dimensional structures are the Sybyl models of I. Tikhonova. They are docked poses.
`examples/make_selectivity_examples.py` converts them and describes the columns, the corrections
to Table 1 of the paper and the state of the file charges. `ampa_glycine.sdf` holds the
conformations of the AMPA poses, superimposed as rigid bodies.

## Three poses with atoms on top of each other

The geometry check of this repository (`cmfqsar.orca.geometry_faults`: no two atoms without a
bond closer than 1.3 Å) fails for three of the 57 AMPA poses. The glycine poses of the same
compounds pass.

| Compound | Closest pair in the AMPA pose | What it is |
|---|---|---|
| `cpd_30` | N22–C27, 0.52 Å | the methyl group of the ethyl ester lies on a nitrogen of the triazole ring: six pairs of atoms are below 1.3 Å, one of them a pair of heavy atoms |
| `cpd_37` | H23–H33, 0.62 Å | two hydrogens of the side chain |
| `cpd_57` | H23–H25, 1.16 Å | two hydrogens |

The same pairs, with the same distances, are in the original MOL2 files of the 2003 study
(checked on 14 November 2026). So this is a property of the published models and not of the
conversion. The pose of `cpd_30` is not a possible structure of the molecule; the other two differ
from a possible structure by the position of one hydrogen.

The coordinates are kept, because the files are here to reproduce the published study on its own
structures. A reader who builds a new model on `ampa_glycine.sdf` should know that the fields of
`cpd_30` near its ester group come from this pose. The six records (three compounds in two files)
are listed in `examples/geometry_exceptions.csv`.
