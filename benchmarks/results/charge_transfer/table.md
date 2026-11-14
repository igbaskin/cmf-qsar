# Measured activation energies of charge transfer at an electrode (plan item BT12)

The values were collected from the literature and are in `examples/charge_transfer/activation_energies.csv`, one row per value, with the paper and the place in it. A row is *primary* when the number was read in the paper that measured it, and *secondary* when it was read in another paper that cites it. All values below are in kJ/mol; a value given in eV was multiplied by 96.485 and a value given in kcal/mol by 4.184. No model is fitted here.

The table holds 92 rows from 16 studies: 92 primary and 0 secondary. By class of electrode: graphite 39, layered oxide 21, lithium titanate 14, non-graphitizable carbon 3, olivine 1, solid electrolyte 9, spinel oxide 5.

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
| YamadaSagane2009 | solid electrolyte | 4 | 32.0 | 53.0 | 21.0 | solvent, temperature range | primary |

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
| YamadaSagane2009: solid electrolyte, no film | 4 | DMC; EC:DMC 1:14; EC:DMC 1:1; EC:DMC 1:9 | 32.0 | 53.0 | 21.0 | primary |

The 10 series hold 38 rows, and the largest holds 8. They hold 23 different solvents or mixtures, of which 13 are single solvents: 2,2,2-trifluoroethyl acetate, DMC, DMF, DMSO, EC, EMC, FEC, PC, THF, acetonitrile, diglyme, poly(ethylene glycol) of molar mass 222, water. 5 single solvents were measured in two or more series: DMC, PC, THF, acetonitrile, water.

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

### Pairs of solvents in more than one series

For every pair of solvents that two or more series measured: the value in the first solvent minus the value in the second, in each series. A difference that changes from series to series is not a property of the two solvents alone.

| First solvent | Second solvent | Series | First minus second, kJ/mol |
|---|---|---|---:|
| DMC | EC:DMC 1:1 | Ishihara2014: lithium titanate, no film | -17.0 |
| DMC | EC:DMC 1:1 | Xu2010: lithium titanate, no film | -1.4 |
| DMC | EC:DMC 1:1 | Yamada2009: graphite, one film formed in EC:DMC 1:1 | -18.0 |
| DMC | EC:DMC 1:1 | Yamada2009: graphite, one film formed with pentafluorostyrene | -4.0 |
| DMC | EC:DMC 1:1 | YamadaSagane2009: solid electrolyte, no film | -19.0 |
| DMC | EC:DMC 1:9 | Yamada2009: graphite, one film formed in EC:DMC 1:1 | -15.0 |
| DMC | EC:DMC 1:9 | YamadaSagane2009: solid electrolyte, no film | -21.0 |
| EC:DMC 1:1 | EC:DMC 1:9 | Yamada2009: graphite, one film formed in EC:DMC 1:1 | +3.0 |
| EC:DMC 1:1 | EC:DMC 1:9 | YamadaSagane2009: solid electrolyte, no film | -2.0 |

## The reference: a regression on the binding energy of Li+ with the solvent

The hypothesis that the removal of the solvent shell sets the barrier has a version with one number: the activation energy rises with the binding energy of Li+ with one molecule of the solvent. It is the reference that a field model of the solvent has to be better than (part (c) of the gate).

### The binding energy of Li+ with one molecule of each solvent

Computed with ORCA: the geometry with `B97-3c TightSCF`, the energy with `B3LYP D3BJ def2-TZVPP RIJCOSX def2/J TightSCF`. The value is the energy of the ion and of the free molecule minus the energy of the complex, each at its own optimised geometry, in the gas phase and with no thermal correction. A larger number is a stronger binding. The starting geometries come from the builder of plan item BT4, which optimises them with GFN2-xTB from 5 starts. *Published* is the value that a paper of the label table gives for the same complex, calculated with B3LYP: a reaction enthalpy or a difference of electronic energies, and, where marked, the position of a marker in a figure. The ORCA runs of the 30 species took 224 minutes of processor time in all.

| Solvent | Binding energy, kJ/mol | Geometry from | Donor atoms at the ion | Shortest distance to the ion, angstrom | With GFN2-xTB alone, kJ/mol | Published, kJ/mol |
|---|---:|---|---:|---:|---:|---|
| water | 155.5 | ORCA | 1 | 1.89 | 187.8 | none |
| THF | 196.7 | ORCA | 1 | 1.83 | 208.5 | 182.5 (figure) |
| acetonitrile | 197.0 | ORCA | 1 | 1.93 | 209.3 | 185.2 (figure) |
| DMC | 197.5 | ORCA | 1 | 1.78 | 235.3 | 185.8 |
| FEC | 199.1 | ORCA | 1 | 1.80 | 217.9 | 190.3 |
| EMC | 201.6 | ORCA | 1 | 1.77 | 243.0 | none |
| 2,2,2-trifluoroethyl acetate | 207.5 | ORCA | 2 | 1.84 | 258.9 | 161.1 (figure) |
| EC | 220.4 | ORCA | 1 | 1.78 | 245.8 | 210.3 |
| PC | 226.9 | ORCA | 1 | 1.78 | 254.1 | 217.1, 213.7 (figure) |
| DMF | 243.6 | ORCA | 1 | 1.76 | 280.7 | 243.1 (figure) |
| DMSO | 250.2 | ORCA | 1 | 1.75 | 306.9 | 235.5 |
| hexamethylphosphoramide (in no series) | 268.3 | GFN2-xTB | 1 | 1.65 | 317.3 | 273.7 |
| diglyme | 363.0 | GFN2-xTB | 3 | 1.83 | 405.2 | none |
| poly(ethylene glycol) of molar mass 222 | 488.4 | GFN2-xTB | 5 | 1.93 | 524.4 | none |

A solvent of more than 20 atoms was not optimised with ORCA, because the ORCA build of this machine runs on one core. It keeps the geometry of GFN2-xTB and ORCA gives the energy only. What that changes was measured on the 11 smaller solvents, which were computed both ways: the binding energy at the geometry of GFN2-xTB is different by -1.9 kJ/mol on average, with a standard deviation of 1.4 kJ/mol and a largest difference of 5.8 kJ/mol.

The polyether of the label table is given by its paper with a molar mass of 222 and no formula. Tetraethylene glycol dimethyl ether has that molar mass and was computed in its place.

Against the 11 published values, the values of ORCA have a correlation of 0.936, and they are higher by 12.3 kJ/mol on average, with a standard deviation of the difference of 12.8 kJ/mol. The values of GFN2-xTB, the method of plan item BT4, have a correlation of 0.803, a mean difference of 44.6 kJ/mol and a standard deviation of 22.0 kJ/mol. For this reason the regression uses the values of ORCA.

A solvent can bind the ion in two ways. The table gives the complex of lower energy, and a paper may have computed the other one. For 2,2,2-trifluoroethyl acetate the two complexes give 177.3 and 207.5 kJ/mol; the one of lower energy holds the ion with 2 atoms. Without this solvent, the 10 other published values and the values of ORCA have a correlation of 0.986, a mean difference of 8.9 kJ/mol and a standard deviation of the difference of 6.4 kJ/mol; for GFN2-xTB the three numbers are 0.948, 39.3 and 13.8.

### The regression, with one solvent held out at a time

The regression has one slope for all series and one intercept for each series, so it describes only the differences inside a series. Every solvent is held out once. The *mean of the series* predicts a held-out value by the mean of the other values of its series, and the regression adds the slope times the distance of the binding energy from the mean of those other values. The error is the mean absolute error over the held-out values, in kJ/mol. The *gain* is the error of the mean minus the error of the regression, with a 95 % interval from a bootstrap over the held-out solvents (2000 draws). The last column is a test against chance: the binding energies are exchanged at random between the solvents, and the column gives the share of the exchanges whose error is not above the error with the true binding energies.

| Points of the regression | Series | Values | Solvents | Slope | Correlation inside the series | Error of the mean of the series | Error of the regression | Gain, with its 95 % interval | Exchanges that do as well |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| single solvents only | 7 | 20 | 13 | +0.073 | 0.39 | 16.6 | 17.1 | -0.5 [-6.8, +3.3] | 34.1 % |
| single solvents, without the two that wrap the ion | 7 | 18 | 11 | +0.559 | 0.87 | 17.9 | 10.3 | +7.6 [+0.2, +13.2] | 0.0 % |
| single solvents and mixtures; a mixture takes its strongest solvent | 10 | 38 | 13 | +0.064 | 0.33 | 12.0 | 14.3 | -2.3 [-6.6, +1.5] | 80.3 % |
| the same, without the two solvents that wrap the ion | 10 | 36 | 11 | +0.491 | 0.84 | 13.4 | 7.0 | +6.4 [+4.2, +9.2] | 0.1 % |
| single solvents, with the binding energies that the papers give | 4 | 12 | 8 | +0.456 | 0.85 | 15.0 | 10.3 | +4.7 [+0.7, +9.5] | 0.9 % |

The repeatability between laboratories, from the reference system above, is 5.0 kJ/mol. The slope is in kJ/mol of activation energy per kJ/mol of binding energy. A mixture is held out once for each of its solvents, so it is counted more than once in the rows with mixtures.

### What a model of the solvent alone cannot remove

Some electrolytes were held out in two or more series. The errors of the regression for one electrolyte differ between its series. A better model of the solvent can move these errors by one amount for all series and no more, so the mean distance of the errors from their median is the smallest error that any model of the solvent alone can reach on these values.

| Points of the regression | Values of electrolytes in two or more series | Error of the regression on them | Smallest error of a model of the solvent alone |
|---|---:|---:|---:|
| single solvents only | 12 | 14.4 | 11.9 |
| single solvents, without the two that wrap the ion | 12 | 11.3 | 10.4 |
| single solvents and mixtures; a mixture takes its strongest solvent | 21 | 13.5 | 9.6 |
| the same, without the two solvents that wrap the ion | 21 | 7.5 | 6.6 |
| single solvents, with the binding energies that the papers give | 7 | 9.6 | 8.8 |

### The slope of each series, single solvents only

| Series | Values | Range of the binding energy, kJ/mol | Range of the activation energy, kJ/mol | Slope of the series alone |
|---|---:|---:|---:|---:|
| Abe2005: solid electrolyte, no film | 3 | 51.1 | 38.4 | +0.758 |
| Chekushkin2021: LiCoO2 | 2 | 71.4 | 46.3 | +0.649 |
| Chekushkin2021: LiMn2O4 | 3 | 207.5 | 15.4 | +0.068 |
| Ishihara2014: lithium titanate, no film | 2 | 29.4 | 27.0 | +0.918 |
| Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 5 | 46.9 | 29.1 | +0.334 |
| Xu2007: graphite, film formed in each electrolyte | 2 | 18.9 | 18.0 | +0.952 |
| Xu2010: lithium titanate, no film | 3 | 291.8 | 10.6 | +0.017 |

### The held-out values: single solvents only

| Held-out solvent | Series | Measured, kJ/mol | Mean of the series | Regression | Error of the regression |
|---|---|---:|---:|---:|---:|
| 2,2,2-trifluoroethyl acetate | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 28.5 | 47.4 | 46.7 | +18.2 |
| DMC | Xu2010: lithium titanate, no film | 50.7 | 45.1 | 32.1 | -18.5 |
| DMC | Ishihara2014: lithium titanate, no film | 37.0 | 64.0 | 61.4 | +24.4 |
| DMF | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 57.6 | 40.1 | 42.6 | -15.0 |
| DMSO | Abe2005: solid electrolyte, no film | 69.9 | 44.4 | 46.9 | -23.0 |
| EC | Xu2007: graphite, film formed in each electrolyte | 71.1 | 53.1 | 54.5 | -16.6 |
| EMC | Xu2007: graphite, film formed in each electrolyte | 53.1 | 71.1 | 69.8 | +16.6 |
| FEC | Abe2005: solid electrolyte, no film | 31.5 | 63.6 | 61.1 | +29.6 |
| PC | Abe2005: solid electrolyte, no film | 57.3 | 50.7 | 50.8 | -6.5 |
| PC | Ishihara2014: lithium titanate, no film | 64.0 | 37.0 | 38.5 | -25.5 |
| PC | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 45.7 | 43.1 | 43.8 | -1.9 |
| PC | Chekushkin2021: LiCoO2 | 61.8 | 15.4 | 19.1 | -42.7 |
| THF | Xu2010: lithium titanate, no film | 40.1 | 50.4 | 39.7 | -0.4 |
| THF | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 42.4 | 43.9 | 42.3 | -0.1 |
| acetonitrile | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 43.7 | 43.6 | 41.9 | -1.8 |
| acetonitrile | Chekushkin2021: LiMn2O4 | 26.1 | 27.0 | 22.3 | -3.7 |
| diglyme | Chekushkin2021: LiMn2O4 | 34.7 | 22.7 | 36.9 | +2.2 |
| poly(ethylene glycol) of molar mass 222 | Xu2010: lithium titanate, no film | 50.2 | 45.4 | 97.5 | +47.4 |
| water | Chekushkin2021: LiCoO2 | 15.4 | 61.8 | 58.1 | +42.7 |
| water | Chekushkin2021: LiMn2O4 | 19.3 | 30.4 | 24.0 | +4.8 |

### The held-out values: single solvents, without the two that wrap the ion

| Held-out solvent | Series | Measured, kJ/mol | Mean of the series | Regression | Error of the regression |
|---|---|---:|---:|---:|---:|
| 2,2,2-trifluoroethyl acetate | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 28.5 | 47.4 | 42.7 | +14.2 |
| DMC | Xu2010: lithium titanate, no film | 50.7 | 40.1 | 40.5 | -10.1 |
| DMC | Ishihara2014: lithium titanate, no film | 37.0 | 64.0 | 48.3 | +11.3 |
| DMF | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 57.6 | 40.1 | 61.1 | +3.5 |
| DMSO | Abe2005: solid electrolyte, no film | 69.9 | 44.4 | 64.5 | -5.4 |
| EC | Xu2007: graphite, film formed in each electrolyte | 71.1 | 53.1 | 63.5 | -7.6 |
| EMC | Xu2007: graphite, film formed in each electrolyte | 53.1 | 71.1 | 60.7 | +7.6 |
| FEC | Abe2005: solid electrolyte, no film | 31.5 | 63.6 | 43.3 | +11.8 |
| PC | Abe2005: solid electrolyte, no film | 57.3 | 50.7 | 51.8 | -5.5 |
| PC | Ishihara2014: lithium titanate, no film | 64.0 | 37.0 | 51.0 | -13.0 |
| PC | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 45.7 | 43.1 | 50.5 | +4.8 |
| PC | Chekushkin2021: LiCoO2 | 61.8 | 15.4 | 49.4 | -12.3 |
| THF | Xu2010: lithium titanate, no film | 40.1 | 50.7 | 50.2 | +10.1 |
| THF | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 42.4 | 43.9 | 30.9 | -11.5 |
| acetonitrile | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 43.7 | 43.6 | 29.4 | -14.3 |
| acetonitrile | Chekushkin2021: LiMn2O4 | 26.1 | 19.3 | 46.5 | +20.4 |
| water | Chekushkin2021: LiCoO2 | 15.4 | 61.8 | 19.6 | +4.2 |
| water | Chekushkin2021: LiMn2O4 | 19.3 | 26.1 | 1.6 | -17.7 |

## Papers that are cited or known and have not been read

| Key | Reference | State |
|---|---|---|
| KyotoRepository | file 1.3247598.pdf of record 2433/109925 in the Kyoto University repository; the site returned no content to an automated reader | not read; no value taken |
