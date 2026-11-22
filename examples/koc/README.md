# Measured soil sorption coefficients (logKoc)

One set of measured soil sorption coefficients, each compound with one 3D structure. It is made
by `benchmarks/prepare_koc.py` and read by `benchmarks/run_koc.py`.

Koc is the soil organic carbon-water partition coefficient: the concentration of a compound
sorbed to soil, per unit mass of soil organic carbon, divided by its concentration in the soil
water at equilibrium, in L/kg. The files hold its decimal logarithm, logKoc. Dividing by the
organic carbon content removes most of the dependence on the individual soil, so logKoc is close
to a property of the molecule alone.

| file | what it holds |
| --- | --- |
| `koc.sdf` | the PHYSPROP logKoc values curated for the OPERA models: 719 compounds, 536 in the training part and 183 in the test part of the OPERA paper |
| `provenance.json` | the source archive, the MD5 checksums, and how many rows were refused and why |

Every record carries `logKoc`, `smiles`, `compound` (the name in the source), `cas`, `dtxsid`
(the identifier of the EPA DSSTox database), `reference` (the source of the measured value, as
OPERA records it), `inchikey14` (the first block of the InChIKey) and `split` (`train` or
`test`, as in the OPERA paper). The values range from 0.0 to 6.5.

**Source.** The data of the OPERA models: K. Mansouri, C. M. Grulke, R. S. Judson, A. J. Williams,
*J. Cheminform.* 10 (2018) 10, doi:10.1186/s13321-018-0263-1. The data are in the file
`OPERA_Data.zip` of the repository github.com/kmansouri/OPERA, under the MIT licence. The member
`KOC_QR.sdf` holds 728 logKoc values with "QSAR-ready" structures (salts and solvents removed)
and the split into a training part (75 %) and a test part (25 %). The archive is not kept here;
the preparation script downloads it into `benchmarks/cache/koc/`, from one fixed commit of the
repository.

**Published accuracy on the same split.** The model report of OPERA (QMRF Q17-26-0017, in
`QMRFs.zip` of the same repository) gives, for its model with 12 descriptors: R² 0.81 and RMSE
0.54 on the training part, Q² 0.81 and RMSE 0.55 in fivefold cross-validation on the training
part, and R² 0.71 and RMSE 0.61 on the test part. The report counts 545 training and 184 test
compounds; the file holds 544 and 184.

**What was refused.** Of the 728 rows: 2 with an element outside H, B, C, N, O, F, Si, P, S, Cl,
Br and I (both hold arsenic), 2 with a net charge, and 5 second copies of a structure (the same
first block of the InChIKey; `provenance.json` names them). All of the remaining 719 could be
embedded.

**Structures.** One conformer per compound: RDKit ETKDGv3 with seed 42, then MMFF94
(`cmfqsar.align.embed_3d`), as for the logP, solubility and boiling point sets.

**What to keep in mind.**

- The sorption of an acid or a base depends on the pH of the soil. The file gives one value per
  compound and does not record the pH, so the values of ionisable compounds are less reliable.
- A Koc value depends on the soils it was measured on, and many values are averages over several
  soils. The scatter between laboratories is a few tenths of a log unit.
- Crippen's clogP, which the program's hydrophobic field uses as its atomic increments, was
  fitted on measured logP values. logKoc follows logP closely, so a model that uses those
  increments has information from outside the training folds.
