# Measured water solubility

Two sets of measured water solubilities, each compound with one 3D structure. The value is logS,
the decimal logarithm of the solubility in mol/L, near 25 C. The sets are made by
`benchmarks/prepare_solubility.py` and read by `benchmarks/run_solubility.py`.

| file | what it holds |
| --- | --- |
| `delaney.sdf` | the Delaney set: 1114 compounds |
| `aqsoldb_reliable.sdf` | 1002 compounds of AqSolDB whose solubility was measured more than once with agreeing results; none of them is in the Delaney set |
| `provenance.json` | the source files, their MD5 checksums, and how many rows were refused and why |

Every record carries `logS`, `smiles`, `compound` (the name in the source) and `inchikey14` (the
first block of the InChIKey). The Delaney records also carry `esol_published`, the value of the
published ESOL equation as the source file gives it. The AqSolDB records also carry `sd` (the
standard deviation of the measurements), `occurrences` (how many measurements were found) and
`group` (the reliability group, below).

**Sources.**

- The Delaney set: J. S. Delaney, *J. Chem. Inf. Comput. Sci.* 44 (2004) 1000-1005, as
  MoleculeNet distributes it (`delaney-processed.csv`, 1,128 compounds). It is the classic
  solubility benchmark and is often called "ESOL" after the equation of the paper. No licence
  is stated with the file. The same set is part of AqSolDB, which is released under CC0.
- AqSolDB: M. C. Sorkun, A. Khetan, S. Er, *Scientific Data* 6 (2019) 143,
  doi:10.1038/s41597-019-0151-1; Harvard Dataverse, doi:10.7910/DVN/OVHAW8, licence CC0 1.0.
  It merges nine public sets into 9,982 compounds.

The source files are not kept here. The preparation script downloads them into
`benchmarks/cache/solubility/` and checks the AqSolDB file against the checksum that Harvard
Dataverse publishes.

**Reliability groups of AqSolDB.** The AqSolDB paper sorts the compounds by how often they were
measured. G1: found in one source only (7,746 compounds). G2 and G3: found twice, with a standard
deviation of the two values of at least 0.5 log units (G2, 235) or below it (G3, 1,182). G4 and
G5: found three times or more, with a standard deviation of at least 0.5 (G4, 183) or below it
(G5, 636). Only G3 and G5 are used here. For a G3 compound the paper keeps the one of the two
values that is closer to the prediction of the ALOGPS program, so the chosen value depends a
little on a model. The two values differ by less than about 0.7 log units in any case.

**What was refused.**

- The Delaney file: methane, which has fewer than two heavy atoms. 13 compounds appear twice
  under two names with the same first block of the InChIKey; the second copy is dropped. Most
  are synonyms with the same value (2-bromonaphthalene, guaiacol). Three are stereoisomers with
  different values (androsterone and epiandrosterone, mannitol and sorbitol, cis- and
  trans-2-pentene).
- AqSolDB, groups G3 and G5 (1,818 compounds): 175 with more than one component (salts and
  hydrates), 8 with an element outside H, B, C, N, O, F, Si, P, S, Cl, Br and I, 2 with a net
  charge, 1 with fewer than two heavy atoms, 7 with a molecular weight above 700, and 7
  stereoisomers of a compound already kept. 616 more are in the Delaney set.

**Structures.** One conformer per compound: RDKit ETKDGv3 with seed 42, then MMFF94
(`cmfqsar.align.embed_3d`), as for the logP sets.

**What to keep in mind.**

- The solubility of a solid depends on its crystal as well as on the molecule. No structure
  here describes the crystal. Both sets mix liquids and solids.
- Many of the reliable AqSolDB compounds are in the Delaney set: 616 of 1,618 after the
  refusals. After their removal the AqSolDB set holds fewer simple, well-studied compounds than
  the Delaney set.
- AqSolDB mixes intrinsic solubility (of the neutral form) with the solubility measured in
  water, as Llompart et al. note (*Scientific Data* 11 (2024) 303). For an acid or a base the two
  can differ.
