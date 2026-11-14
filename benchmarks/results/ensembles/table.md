# Item MS5: an ensemble of structures through the general entry

The gate of item MS5 is that the general path reproduces the numbers of item BT5 from the same frames. The general path reads one SD file with `load_dataset(..., ensemble_by="electrolyte", member_weight="member_weight")` and builds the fields with `conformers="ensemble"` and the weighting "given". BT5's path reads the shell files with the code of `cmfqsar/shells.py`.

## The file

989 records, one per shell, make 62 electrolytes. An electrolyte has 15 to 16 shells, and a shell has 15 to 94 atoms. In 62 of the 62 electrolytes the shells differ in their atoms, so they cannot be conformations of one molecule.

## The tables of the two paths

The framed kernel is the joint overlap kernel of the five fields, with a unit diagonal. The alignment-free rows are the rows of the rotation-invariant autocorrelation. The general path does not use the kernel cache.

| Field width (alpha) | Largest difference in the framed kernel | Largest difference in the alignment-free rows |
|---:|---:|---:|
| 0.3 | 0 | 0 |
| 0.5 | 0 | 0 |
| 1.0 | 0 | 0 |
| 2.0 | 0 | 0 |

## The numbers of BT5, from each path

Nested cross-validation as in `benchmarks/run_shells.py`: the ridge and the field width chosen inside each fold, the field kernel and the molality kernel mixed in equal parts. The mean absolute error is the mean over the five folds. The difference is the largest absolute difference between the new record and the stored one in `benchmarks/results/shells/gate.json`, over every number in the record: the error of each electrolyte, the settings chosen in each fold and the summary numbers.

| Target | Split | Model | Mean absolute error, stored | Mean absolute error, BT5's path now | Difference from the stored record, BT5's path | Mean absolute error, general path | Difference from the stored record, general path |
|---|---|---|---:|---:|---:|---:|---:|
| log10 of the conductivity at 298 K | random | shells | 0.1147 | 0.1147 | 0 | 0.1147 | 0 |
| log10 of the conductivity at 298 K | random | shells-free | 0.1091 | 0.1091 | 0 | 0.1091 | 0 |
| log10 of the conductivity at 298 K | by-composition | shells | 0.1130 | 0.1130 | 0 | 0.1130 | 0 |
| log10 of the conductivity at 298 K | by-composition | shells-free | 0.1034 | 0.1034 | 0 | 0.1034 | 0 |
| activation energy at 298 K, kJ/mol | random | shells | 2.2357 | 2.2357 | 0 | 2.2357 | 0 |
| activation energy at 298 K, kJ/mol | random | shells-free | 1.6121 | 1.6121 | 0 | 1.6121 | 0 |
| activation energy at 298 K, kJ/mol | by-composition | shells | 2.9028 | 2.9028 | 0 | 2.9028 | 0 |
| activation energy at 298 K, kJ/mol | by-composition | shells-free | 1.6845 | 1.6845 | 0 | 1.6845 | 0 |

## The gate

**Met.** Every number of the two models is reproduced exactly by the general path: the tables are the same to the last bit, and so is every record of the nested cross-validation.

The control, BT5's own path run again, differs from the stored file by 0, so the stored file matches the present code.
