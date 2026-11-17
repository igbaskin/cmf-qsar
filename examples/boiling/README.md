# Measured boiling points

Two sets of measured normal boiling points, in degrees C, each compound with one 3D structure.
They are made by `benchmarks/prepare_boiling.py` and read by `benchmarks/run_boiling.py`.

| file | what it holds |
| --- | --- |
| `physprop.sdf` | the PHYSPROP boiling points curated for the OPERA models: 5412 compounds, 4059 in the training part and 1353 in the test part of the OPERA paper |
| `echemportal.sdf` | 1617 boiling points from registration dossiers, added in version 2.9 of OPERA; none of them is in `physprop.sdf` |
| `provenance.json` | the source archive, the MD5 checksums, and how many rows were refused and why |

Every record carries `bp` (the boiling point, C), `smiles`, `compound` (the name in the source),
`cas` and `inchikey14` (the first block of the InChIKey). The PHYSPROP records also carry `split`
(`train` or `test`, as in the OPERA paper) and `dtxsid` (the identifier of the EPA DSSTox
database). The eChemPortal records also carry `sources` (the sources the value was taken from,
with the number of values from each) and `curation_status` (the status the OPERA curators gave
the row).

**Source.** The data of the OPERA models: K. Mansouri, C. M. Grulke, R. S. Judson, A. J. Williams,
*J. Cheminform.* 10 (2018) 10, doi:10.1186/s13321-018-0263-1. The data are in the file
`OPERA_Data.zip` of the repository github.com/kmansouri/OPERA, under the MIT licence. Two files of
that archive are used. `BP_QR.sdf` holds the 5,434 PHYSPROP boiling points of the paper, with
"QSAR-ready" structures (salts and solvents removed) and the split into a training part (75 %)
and a test part (25 %). `BP_2.9_update_checked_targeted.xlsx` holds the 2,426 boiling points that
version 2.9 added, with the value the curators checked and a status for each row. The archive is
not kept here; the preparation script downloads it into `benchmarks/cache/boiling/`.

**Published accuracy on the same split.** The OPERA paper reports, for its boiling point model
with 13 descriptors: Q² 0.93 and RMSE 22.46 C in fivefold cross-validation on the training part,
and R² 0.93 and RMSE 22.08 C on the test part.

**What was refused.**

- PHYSPROP (5,434 rows): 18 with an element outside H, B, C, N, O, F, Si, P, S, Cl, Br and I, 2
  with a molecular weight above 700, 1 with fewer than two heavy atoms, and 1 second copy of a
  structure.
- The update (2,426 rows): 699 without eChemPortal among their sources. Most of these come from
  web catalogues (LookChem, ChemicalBook), whose pages often give a predicted value. Then 6 that
  the curators flagged or left for a manual review (one is a decomposition temperature, one was
  measured at another pressure), 50 with a net charge, 3 zwitterions, 1 without a readable
  SMILES, 15 above a molecular weight of 700, 30 second copies of a structure, and 4 compounds
  already in PHYSPROP. One of the remaining 1,618 could not be embedded.

**Structures.** One conformer per compound: RDKit ETKDGv3 with seed 42, then MMFF94
(`cmfqsar.align.embed_3d`), as for the logP and solubility sets.

**What to keep in mind.**

- Most values of the update were not checked against a second source: 1,720 of its eChemPortal
  rows have the status "retained_original_not_externally_checked". A value in a registration
  dossier can be a measurement at reduced pressure converted to normal pressure, or an estimate.
- The two sets differ. The eChemPortal compounds are larger industrial chemicals: their mean
  boiling point is 261 C, against 189 C for PHYSPROP, so a model fitted on PHYSPROP and tested on
  eChemPortal is partly extrapolating.
- A row's value may combine values from several sources (the `sources` field says which).
