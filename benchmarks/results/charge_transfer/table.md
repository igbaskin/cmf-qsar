# Measured activation energies of charge transfer at an electrode (plan item BT12)

The values were collected from the literature and are in `examples/charge_transfer/activation_energies.csv`, one row per value, with the paper and the place in it. A row is *primary* when the number was read in the paper that measured it, and *secondary* when it was read in another paper that cites it. All values below are in kJ/mol; a value given in eV was multiplied by 96.485 and a value given in kcal/mol by 4.184. No model is fitted here.

The table holds 90 rows from 16 studies: 88 primary and 2 secondary. By class of electrode: graphite 39, layered oxide 21, lithium titanate 14, non-graphitizable carbon 3, olivine 1, solid electrolyte 7, spinel oxide 5.

## How well laboratories agree on one system

The reference system is a graphite electrode with its surface film after formation, in a carbonate electrolyte that contains ethylene carbonate, with no additive. One value per study, the mean of its rows for this system.

| Study | Activation energy, kJ/mol |
|---|---:|
| Abe2004 | 56.0 |
| Jow2018 | 57.7 |
| Keefe2020 | 58.4 |
| Xu2007 | 68.5 |
| Yamada2009 | 58.0 |

Over these 5 studies the mean is 59.7 kJ/mol, the standard deviation 5.0 kJ/mol and the range 56.0 to 68.5 kJ/mol. The studies differ in the salt, in the second solvent and in the kind of graphite, so this spread is an upper estimate of the repeatability of one system.

## How much the value changes inside one study

| Study | Electrode | Values | Lowest | Highest | Range | What differs between the rows | Read as |
|---|---|---:|---:|---:|---:|---|---|
| Abe2004 | graphite | 2 | 25.0 | 25.0 | 0.0 | solvent, state | primary |
| Abe2005 | solid electrolyte | 4 | 25.0 | 69.9 | 44.9 | solvent, salt, form of the electrode | primary |
| Abraham2006 | layered oxide | 2 | 41.5 | 44.4 | 2.9 | film | primary |
| Abraham2008 | lithium titanate | 3 | 36.8 | 52.7 | 15.9 | form of the electrode, temperature range | primary |
| Chekushkin2021 | layered oxide | 2 | 15.4 | 61.8 | 46.3 | solvent, film | primary |
| Chekushkin2021 | spinel oxide | 4 | 19.3 | 34.7 | 15.4 | solvent, salt, film | primary |
| Doi2004 | non-graphitizable carbon | 3 | 72.0 | 75.8 | 3.8 | solvent, salt | primary |
| Ishihara2014 | lithium titanate | 3 | 37.0 | 64.0 | 27.0 | solvent | primary |
| Jow2012 | graphite | 2 | 67.0 | 68.0 | 1.0 | state | primary |
| Jow2018 | graphite | 6 | 57.7 | 69.1 | 11.4 | additive | primary |
| Jow2018 | layered oxide | 6 | 61.2 | 100.0 | 38.8 | additive | primary |
| Keefe2020 | graphite | 12 | 42.5 | 62.7 | 20.3 | additive, film, state | primary |
| Keefe2020 | layered oxide | 8 | 53.1 | 68.5 | 15.4 | additive, film, state | primary |
| Sawayama2025 | graphite | 5 | 28.5 | 57.6 | 29.1 | solvent | primary |
| Xu2007 | graphite | 4 | 53.1 | 71.1 | 18.0 | solvent | primary |
| Xu2010 | lithium titanate | 8 | 40.1 | 54.9 | 14.9 | solvent, salt | primary |
| Yamada2009 | graphite | 6 | 40.0 | 58.0 | 18.0 | solvent, film | primary |
| YamadaSagane2009 | solid electrolyte | 2 | 32.0 | 51.0 | 19.0 | solvent | secondary |

## The two electrodes of a cell with a carbonate electrolyte

Films after formation, every additive, one value per row. The published value is the activation energy that the cell model of plan item MS9 holds.

| Electrode | Rows | Mean | Standard deviation | Lowest | Highest | Published value | Mean minus published |
|---|---:|---:|---:|---:|---:|---:|---:|
| negative | 18 | 61.7 | 6.1 | 48.2 | 71.1 | 35.0 | +26.7 |
| positive | 12 | 68.0 | 13.0 | 52.0 | 100.0 | 17.8 | +50.2 |

What that difference costs a cold cell, from the stage `--stage exchange` of `run_cellmodel.py`. The stage ran stated activation energies in steps of 10 kJ/mol; the column gives the one nearest to the mean and the difference of the capacity, in per cent of the capacity with the published values.

| Temperature, K | Rate | Negative electrode alone | Positive electrode alone | Both electrodes |
|---:|---:|---|---|---|
| 263 | 1C | 60 kJ/mol: -1.8 % | 70 kJ/mol: -3.9 % | 60 kJ/mol: -5.2 % |
| 263 | 1.5C | 60 kJ/mol: -1.9 % | 70 kJ/mol: -4.5 % | 60 kJ/mol: -6.1 % |
| 253 | 1C | 60 kJ/mol: -2.6 % | 70 kJ/mol: -5.7 % | 60 kJ/mol: -7.7 % |
| 253 | 1.5C | 60 kJ/mol: -2.1 % | 70 kJ/mol: -5.5 % | 60 kJ/mol: -7.8 % |

## The series in which only the solvent changes

A series is a set of rows of one study that share the electrode and the film, or the way the film was made, and differ in the solvent. Part (c) of the gate of the item needs such series: it centres each study on its own mean and holds out every electrolyte with one solvent together.

| Series | Rows | Solvents, from the lowest value to the highest | Lowest | Highest | Range | Read as |
|---|---:|---|---:|---:|---:|---|
| Abe2005: solid electrolyte, no film | 3 | FEC; PC; DMSO | 31.5 | 69.9 | 38.4 | primary |
| Chekushkin2021: LiCoO2 | 2 | water; PC | 15.4 | 61.8 | 46.3 | primary |
| Chekushkin2021: LiMn2O4 | 4 | water; acetonitrile; EC:DMC; diglyme | 19.3 | 34.7 | 15.4 | primary |
| Ishihara2014: lithium titanate, no film | 3 | DMC; EC:DMC 1:1; PC | 37.0 | 64.0 | 27.0 | primary |
| Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 5 | 2,2,2-trifluoroethyl acetate; THF; acetonitrile; PC; DMF | 28.5 | 57.6 | 29.1 | primary |
| Xu2007: graphite, film formed in each electrolyte | 4 | EMC; EC:EMC 20:80; EC:EMC 30:70; EC | 53.1 | 71.1 | 18.0 | primary |
| Xu2010: lithium titanate, no film | 8 | THF; poly(ethylene glycol) of molar mass 222; EC:DMC 60:40; DMC; EC:DMC 20:80; EC:DMC 1:1; EC:DMC 30:70; EC:DMC 70:30 | 40.1 | 54.9 | 14.9 | primary |
| Yamada2009: graphite, one film formed in EC:DMC 1:1 | 3 | DMC; EC:DMC 1:9; EC:DMC 1:1 | 40.0 | 58.0 | 18.0 | primary |
| Yamada2009: graphite, one film formed with pentafluorostyrene | 2 | DMC; EC:DMC 1:1 | 52.0 | 56.0 | 4.0 | primary |
| YamadaSagane2009: solid electrolyte, no film | 2 | DMC; EC:DMC | 32.0 | 51.0 | 19.0 | secondary |

The 10 series hold 36 rows, and the largest holds 8. They hold 22 different solvents or mixtures, of which 13 are single solvents: 2,2,2-trifluoroethyl acetate, DMC, DMF, DMSO, EC, EMC, FEC, PC, THF, acetonitrile, diglyme, poly(ethylene glycol) of molar mass 222, water. 5 single solvents were measured in two or more series: DMC, PC, THF, acetonitrile, water.

### One solvent in more than one series

The value of a single solvent in each series that measured it. The series differ in the electrode, in the film, in the salt and in its concentration.

| Solvent | Series | Activation energy, kJ/mol |
|---|---|---:|
| DMC | Ishihara2014: lithium titanate, no film | 37.0 |
| DMC | Xu2010: lithium titanate, no film | 50.7 |
| DMC | Yamada2009: graphite, one film formed in EC:DMC 1:1 | 40.0 |
| DMC | Yamada2009: graphite, one film formed with pentafluorostyrene | 52.0 |
| DMC | YamadaSagane2009: solid electrolyte, no film | 32.0 |
| PC | Abe2005: solid electrolyte, no film | 57.3 |
| PC | Chekushkin2021: LiCoO2 | 61.8 |
| PC | Ishihara2014: lithium titanate, no film | 64.0 |
| PC | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 45.7 |
| THF | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 42.4 |
| THF | Xu2010: lithium titanate, no film | 40.1 |
| acetonitrile | Chekushkin2021: LiMn2O4 | 26.1 |
| acetonitrile | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 43.7 |
| water | Chekushkin2021: LiCoO2 | 15.4 |
| water | Chekushkin2021: LiMn2O4 | 19.3 |

### One pair of solvents in more than one series

For every pair of solvents that two or more series measured: the value in the first solvent minus the value in the second, in each series. A difference that changes from series to series is not a property of the two solvents alone.

| First solvent | Second solvent | Series | First minus second, kJ/mol |
|---|---|---|---:|
| DMC | EC:DMC 1:1 | Ishihara2014: lithium titanate, no film | -17.0 |
| DMC | EC:DMC 1:1 | Xu2010: lithium titanate, no film | -1.4 |
| DMC | EC:DMC 1:1 | Yamada2009: graphite, one film formed in EC:DMC 1:1 | -18.0 |
| DMC | EC:DMC 1:1 | Yamada2009: graphite, one film formed with pentafluorostyrene | -4.0 |

## Papers that are cited or known and have not been read

| Key | Reference | State |
|---|---|---|
| KyotoRepository | file 1.3247598.pdf of record 2433/109925 in the Kyoto University repository; the site returned no content to an automated reader | not read; no value taken |
| YamadaSagane2009 | Y. Yamada, F. Sagane, Y. Iriyama, T. Abe, Z. Ogumi, J. Phys. Chem. C, 2009, 113, 14528-14532 | cited only; to be read |
