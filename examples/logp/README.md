# Measured logP

Two sets of measured octanol-water partition coefficients (logP), each compound with one 3D
structure. They are made by `benchmarks/prepare_logp.py` and read by `benchmarks/run_logp.py`.

| file | what it holds |
| --- | --- |
| `huuskonen.sdf` | the Huuskonen set: 1847 compounds |
| `sangster_p.sdf` | 2999 compounds sampled at random from class P of SangsterLogP; none of them is in the Huuskonen set |
| `provenance.json` | the source file, its MD5 checksum, and how many rows were refused and why |

Every record carries `logP`, `smiles`, `compound` (the name in the source), `source` (the
reference of the measurement) and `inchikey14` (the first block of the InChIKey). The class P
records also carry `method` (the measurement method as coded by Sangster) and `recommended`
("yes" when Sangster marked the value as the preferred one).

**Source.** The SangsterLogP data set: T. Cirino, J. Sangster, L. Charochkina, I. Tetko,
Zenodo record 19387552, doi:10.5281/zenodo.19387552, licence CC BY 4.0. It is described in
Cirino et al., *Scientific Data* (2026), doi:10.1038/s41597-026-07357-2. The spreadsheet holds
the main table (23,520 compounds) and the two sets the authors used for validation: the
Huuskonen set (J. Huuskonen et al., *J. Chem. Inf. Comput. Sci.* 2000) and a prospective set of
327 compounds. The licence asks for attribution, which this paragraph gives. The spreadsheet is
not kept here; the preparation script downloads it into `benchmarks/cache/logp/`.

**Class P.** The data descriptor sorts the main table into classes by how the value was
measured. Class P holds values that the original authors reported as logP. The authors call it
the most reliable class. Class N holds logD values measured at a pH where the compound is nearly
neutral. Classes M and H hold logD values of ionised compounds that were corrected to logP. Only
class P is used here.

**What was refused.**

- 21 rows of the Huuskonen sheet have no logP value. A comma in the compound name moved the rest
  of the name into the logP column when the sheet was made, and the number is not in the row.
  One more row of that sheet is empty, and 2 compounds appear twice (the second copy is dropped).
- Compounds with a net charge, a zwitterion, more than one component, an element outside H, B,
  C, N, O, F, Si, P, S, Cl, Br and I, or fewer than two heavy atoms. No row of the Huuskonen
  sheet and no row of class P was refused for these reasons. A nitro group or an
  N-oxide is written with a +1 and a -1 on bonded atoms; such a compound is neutral and is kept.
- The 1389 class P compounds that are also in the Huuskonen set (same first block of the
  InChIKey). Compounds with a molecular weight above 700 would also be left out, but class P has
  none. The sample of 3000 was then drawn with a fixed seed from the remaining 12,423 compounds
  of the 13,812 in class P, and 1 of them could not be embedded.

**Structures.** One conformer per compound: RDKit ETKDGv3 with seed 42, then MMFF94
(`cmfqsar.align.embed_3d`). logP is an equilibrium property, so in principle it is an average
over conformers. One conformer is the program's default and the simplest choice.

**What to keep in mind.** The data descriptor says that the Huuskonen measurements come from the
same period as Sangster's, so the two sets are separate compounds but probably not separate
sources. Crippen's
clogP, which the program's hydrophobic field uses as its atomic increments, was fitted on a large
logP set that probably includes many of these compounds; a model that uses those increments has
seen information from outside the training folds.
