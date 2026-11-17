# Measured vapour pressures

Two sets of measured vapour pressures, each compound with one 3D structure. The value is logVP,
the decimal logarithm of the vapour pressure in mmHg, near 25 C. The sets are made by
`benchmarks/prepare_vapour.py` and read by `benchmarks/run_vapour.py`.

| file | what it holds |
| --- | --- |
| `physprop.sdf` | the PHYSPROP vapour pressures curated for the OPERA models: 2693 compounds, 2016 in the training part and 677 in the test part of the OPERA paper |
| `echemportal.sdf` | 2884 vapour pressures from registration dossiers, added in version 2.9 of OPERA; none of them is in `physprop.sdf` |
| `provenance.json` | the source archive, the MD5 checksums, the temperatures of the PHYSPROP values, and how many rows were refused and why |

Every record carries `logVP`, `smiles`, `compound` (the name in the source), `cas` and
`inchikey14` (the first block of the InChIKey). The PHYSPROP records also carry `split` (`train`
or `test`, as in the OPERA paper), `temperature` (the temperature of the measurement, C) and
`reference` (the publication of the value). The eChemPortal records also carry `sources` (the
sources the value was taken from) and `curation_status` (the status the OPERA curators gave the
row).

**Source.** The data of the OPERA models: K. Mansouri, C. M. Grulke, R. S. Judson, A. J. Williams,
*J. Cheminform.* 10 (2018) 10, doi:10.1186/s13321-018-0263-1, in the file `OPERA_Data.zip` of the
repository github.com/kmansouri/OPERA, under the MIT licence. Two files of that archive are used:
`VP_QR.sdf`, the 2,819 PHYSPROP vapour pressures of the paper with "QSAR-ready" structures and
its split into a training part and a test part, and
`VP_2.9_update_checked_chemicalbook_log10mmHg.xlsx`, the 3,945 vapour pressures that version
2.9 added, with the curators' value and status. The archive is not kept here; the preparation
script uses the copy the boiling point preparation downloaded, or downloads it.

**Published accuracy on the same split.** The OPERA paper reports, for its vapour pressure model
with 12 descriptors: Q² 0.91 and RMSE 1.08 log units in fivefold cross-validation on the training
part, and R² 0.92 and RMSE 1.00 on the test part.

**What was refused.**

- PHYSPROP (2,819 rows): 108 records in neither part of the OPERA split, which also have no
  temperature; 9 with an element outside H, B, C, N, O, F, Si, P, S, Cl, Br and I; 3 with a
  molecular weight above 700; 2 with a net charge; 1 zwitterion; 1 with fewer than two heavy
  atoms; and 2 second copies of a structure.
- The update (3,945 rows): 793 without eChemPortal among their sources; 5 whose status marks the
  value as a range, a value at 50 C, a value that is practically zero, an implausible value or
  values that differ between CAS numbers; 77 with a net charge, 18 zwitterions, 3 without a
  readable SMILES, 65 above a molecular weight of 700, 97 second copies of a structure, and 2
  compounds already in PHYSPROP. One of the remaining 2,885 could not be embedded.

**Structures.** One conformer per compound: RDKit ETKDGv3 with seed 42, then MMFF94
(`cmfqsar.align.embed_3d`), as for the other physical property sets.

**What to keep in mind.**

- Not all PHYSPROP values are at 25 C: 2,355 are, 283 are at 20 C, and 55 are at other
  temperatures between 15 and 31 C. The values are used as they are, as in the OPERA paper. Five
  degrees change a vapour pressure by roughly 0.1 to 0.3 log units, which is small against the
  spread of the values (3.5 log units).
- 236 of the eChemPortal values, and 32 of the PHYSPROP values, are a whole power of ten in
  pascals (their logVP ends in .125 after the conversion to mmHg). In registration dossiers such
  a value is often a limit ("below 0.001 Pa") rather than a measurement. The files do not say
  which ones are limits, so they are kept.
- Almost no eChemPortal value was checked against a second source: the curators checked 34 rows
  of the update.
- The vapour pressure of a solid includes the cost of leaving the crystal. No structure here
  describes the crystal.
