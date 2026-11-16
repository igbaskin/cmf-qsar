# Measured activation energies of charge transfer at an electrode (plan item BT12)

The values were collected from the literature and are in `examples/charge_transfer/activation_energies.csv`, one row per value, with the paper and the place in it. A row is *primary* when the number was read in the paper that measured it, and *secondary* when it was read in another paper that cites it. All values below are in kJ/mol; a value given in eV was multiplied by 96.485 and a value given in kcal/mol by 4.184. No model is fitted here.

The table holds 162 rows from 34 studies: 162 primary and 0 secondary. By class of electrode: carbonaceous thin film 3, graphite 47, layered oxide 25, lithium metal 6, lithium titanate 27, non-graphitizable carbon 13, olivine 1, silicon monoxide 4, solid electrolyte 25, spinel oxide 11.

## How well laboratories agree on one system

The reference system is a graphite electrode with its surface film after formation, in a carbonate electrolyte that contains ethylene carbonate and LiPF6 or LiClO4 at 0.8 to 1.5 mol/L, with no additive. One value per study, the mean of its rows for this system.

| Study | Activation energy, kJ/mol |
|---|---:|
| Abe2004 | 56.0 |
| Jow2018 | 57.7 |
| Keefe2020 | 58.4 |
| Son2025 | 53.6 |
| Uchida2020 | 62.1 |
| Xu2007 | 68.5 |
| Yamada2009 | 58.0 |

Over these 7 studies the mean is 59.2 kJ/mol, the standard deviation 4.8 kJ/mol and the range 53.6 to 68.5 kJ/mol. The studies differ in the salt, in the second solvent and in the kind of graphite, so this spread is an upper estimate of the repeatability of one system.

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
| Doi2004Carbon | non-graphitizable carbon | 4 | 55.8 | 70.0 | 14.2 | solvent, form of the electrode | primary |
| Doi2005 | non-graphitizable carbon | 6 | 47.5 | 67.5 | 20.0 | solvent, additive, form of the electrode | primary |
| Doi2022 | solid electrolyte | 6 | 34.0 | 70.4 | 36.4 | nothing | primary |
| Fukuma2024 | lithium titanate | 2 | 12.5 | 18.1 | 5.6 | film, state | primary |
| Fukutsuka2010 | carbonaceous thin film | 3 | 60.2 | 82.3 | 22.1 | solvent, salt, film | primary |
| Iriyama2004 | layered oxide | 2 | 47.3 | 61.8 | 14.5 | nothing | primary |
| Ishihara2014 | lithium titanate | 3 | 37.0 | 64.0 | 27.0 | solvent | primary |
| Ishihara2014JES | lithium titanate | 5 | 69.0 | 73.0 | 4.0 | solvent, salt | primary |
| Jow2012 | graphite | 2 | 67.0 | 68.0 | 1.0 | state | primary |
| Jow2018 | graphite | 6 | 57.7 | 69.1 | 11.4 | additive | primary |
| Jow2018 | layered oxide | 6 | 61.2 | 100.0 | 38.8 | additive | primary |
| Keefe2020 | graphite | 12 | 42.5 | 62.7 | 20.3 | additive, film, state | primary |
| Keefe2020 | layered oxide | 8 | 53.1 | 68.5 | 15.4 | additive, film, state | primary |
| Nakayama2007 | spinel oxide | 5 | 23.0 | 50.0 | 27.0 | solvent, salt | primary |
| Nasara2020 | lithium titanate | 6 | 36.0 | 67.0 | 31.0 | solvent, film, state | primary |
| Sagane2005 | solid electrolyte | 2 | 56.2 | 97.4 | 41.2 | solvent, form of the electrode | primary |
| Sagane2012 | solid electrolyte | 7 | 25.0 | 72.0 | 47.0 | solvent, salt | primary |
| Sawayama2025 | graphite | 5 | 28.5 | 57.6 | 29.1 | solvent | primary |
| Son2025 | graphite | 4 | 44.2 | 58.9 | 14.7 | nothing | primary |
| Uchida2020 | graphite | 4 | 36.6 | 71.2 | 34.6 | solvent, salt | primary |
| Xu2007 | graphite | 4 | 53.1 | 71.1 | 18.0 | solvent | primary |
| Xu2010 | lithium titanate | 8 | 40.1 | 54.9 | 14.9 | solvent, salt | primary |
| Yamada2007 | layered oxide | 2 | 46.0 | 48.0 | 2.0 | nothing | primary |
| Yamada2009 | graphite | 6 | 40.0 | 58.0 | 18.0 | solvent, film | primary |
| Yamada2010 | silicon monoxide | 4 | 29.0 | 32.0 | 3.0 | solvent, salt | primary |
| YamadaSagane2009 | solid electrolyte | 4 | 32.0 | 53.0 | 21.0 | solvent, temperature range | primary |
| Zabara2024 | lithium metal | 6 | 47.7 | 69.5 | 21.8 | solvent, film, state | primary |

## The two electrodes of a cell with a carbonate electrolyte

Films after formation, every additive, one value per row. The published value is the activation energy that the cell model of plan item MS9 holds.

| Electrode | Rows | Mean | Standard deviation | Lowest | Highest | Published value | Mean minus published |
|---|---:|---:|---:|---:|---:|---:|---:|
| negative | 21 | 61.4 | 6.5 | 48.2 | 71.2 | 35.0 | +26.4 |
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
| Doi2004Carbon: carbon treated by hot isostatic pressing, film formed in each electrolyte | 2 | PC; EC:DEC 1:1 | 55.8 | 61.5 | 5.7 | primary |
| Doi2004Carbon: glassy carbon GC-20 not heat-treated, film formed in each electrolyte | 2 | EC:DEC 1:1; PC | 68.7 | 70.0 | 1.3 | primary |
| Doi2005: glassy carbon GC-10, film formed in each electrolyte | 2 | EC:DEC 1:1; PC | 47.5 | 49.0 | 1.5 | primary |
| Doi2005: glassy carbon GC-20 heat-treated at 773 K, film formed in each electrolyte | 2 | EC:DEC 1:1; PC | 47.9 | 51.6 | 3.7 | primary |
| Fukutsuka2010: carbonaceous thin film, one film formed in EC:DEC 1:1 | 3 | EC:DEC 1:1; PC; DMSO | 60.2 | 82.3 | 22.1 | primary |
| Ishihara2014: lithium titanate, no film | 3 | DMC; EC:DMC 1:1; PC | 37.0 | 64.0 | 27.0 | primary |
| Nakayama2007: LiMn2O4 thin film, water and PC | 5 | water; water; water; water; PC | 23.0 | 50.0 | 27.0 | primary |
| Nasara2020: lithium titanate thin film, film formed in each electrolyte | 3 | FEC:DMC 3:7; EC:DEC 1:1; PC | 44.0 | 67.0 | 23.0 | primary |
| Nasara2020: lithium titanate thin film, no film | 3 | FEC:DMC 3:7; EC:DEC 1:1; PC | 36.0 | 66.0 | 30.0 | primary |
| Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 5 | 2,2,2-trifluoroethyl acetate; THF; acetonitrile; PC; DMF | 28.5 | 57.6 | 29.1 | primary |
| Uchida2020: graphite, LiFSA, film formed in each electrolyte | 2 | EC:DMC 1:9; EC:DMC 1:1 | 36.6 | 50.0 | 13.4 | primary |
| Uchida2020: graphite, LiPF6, film formed in each electrolyte | 2 | EC:DMC 1:1; EC:DMC 1:9 | 53.0 | 71.2 | 18.2 | primary |
| Xu2007: graphite, film formed in each electrolyte | 4 | EMC; EC:EMC 20:80; EC:EMC 30:70; EC | 53.1 | 71.1 | 18.0 | primary |
| Xu2010: lithium titanate, no film | 8 | THF; poly(ethylene glycol) of molar mass 222; EC:DMC 60:40; DMC; EC:DMC 20:80; EC:DMC 1:1; EC:DMC 30:70; EC:DMC 70:30 | 40.1 | 54.9 | 14.9 | primary |
| Yamada2009: graphite, one film formed in EC:DMC 1:1 | 3 | DMC; EC:DMC 1:9; EC:DMC 1:1 | 40.0 | 58.0 | 18.0 | primary |
| Yamada2009: graphite, one film formed with pentafluorostyrene | 2 | DMC; EC:DMC 1:1 | 52.0 | 56.0 | 4.0 | primary |
| Yamada2010: SiO, film formed in each electrolyte | 4 | DMSO; PC; DMC; EC:DMC 1:1 | 29.0 | 32.0 | 3.0 | primary |
| YamadaSagane2009: solid electrolyte, no film | 4 | DMC; EC:DMC 1:14; EC:DMC 1:1; EC:DMC 1:9 | 32.0 | 53.0 | 21.0 | primary |
| Zabara2024: lithium metal, after ten cycles, film formed in each electrolyte | 3 | EC:DMC 1:1; EC:DEC 1:1; EC:EMC 7:3 | 47.7 | 64.4 | 16.7 | primary |
| Zabara2024: lithium metal, as assembled, film formed in each electrolyte | 3 | EC:DMC 1:1; EC:DEC 1:1; EC:EMC 7:3 | 67.6 | 69.5 | 1.9 | primary |

The 23 series hold 74 rows, and the largest holds 8. They hold 26 different solvents or mixtures, of which 13 are single solvents: 2,2,2-trifluoroethyl acetate, DMC, DMF, DMSO, EC, EMC, FEC, PC, THF, acetonitrile, diglyme, poly(ethylene glycol) of molar mass 222, water. 6 single solvents were measured in two or more series: DMC, DMSO, PC, THF, acetonitrile, water.

### One solvent in more than one series

The value of a single solvent in each series that measured it. The series differ in the electrode, in the film, in the salt and in its concentration.

| Solvent | Series | Activation energy, kJ/mol |
|---|---|---:|
| DMC | Ishihara2014: lithium titanate, no film | 37.0 |
| DMC | Xu2010: lithium titanate, no film | 50.7 |
| DMC | Yamada2009: graphite, one film formed in EC:DMC 1:1 | 40.0 |
| DMC | Yamada2009: graphite, one film formed with pentafluorostyrene | 52.0 |
| DMC | Yamada2010: SiO, film formed in each electrolyte | 31.0 |
| DMC | YamadaSagane2009: solid electrolyte, no film | 32.0 |
| DMSO | Abe2005: solid electrolyte, no film | 69.9 |
| DMSO | Fukutsuka2010: carbonaceous thin film, one film formed in EC:DEC 1:1 | 82.3 |
| DMSO | Yamada2010: SiO, film formed in each electrolyte | 29.0 |
| PC | Abe2005: solid electrolyte, no film | 57.3 |
| PC | Chekushkin2021: LiCoO2 | 61.8 |
| PC | Doi2004Carbon: carbon treated by hot isostatic pressing, film formed in each electrolyte | 55.8 |
| PC | Doi2004Carbon: glassy carbon GC-20 not heat-treated, film formed in each electrolyte | 70.0 |
| PC | Doi2005: glassy carbon GC-10, film formed in each electrolyte | 49.0 |
| PC | Doi2005: glassy carbon GC-20 heat-treated at 773 K, film formed in each electrolyte | 51.6 |
| PC | Fukutsuka2010: carbonaceous thin film, one film formed in EC:DEC 1:1 | 71.2 |
| PC | Ishihara2014: lithium titanate, no film | 64.0 |
| PC | Nakayama2007: LiMn2O4 thin film, water and PC | 50.0 |
| PC | Nasara2020: lithium titanate thin film, film formed in each electrolyte | 67.0 |
| PC | Nasara2020: lithium titanate thin film, no film | 66.0 |
| PC | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 45.7 |
| PC | Yamada2010: SiO, film formed in each electrolyte | 30.0 |
| THF | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 42.4 |
| THF | Xu2010: lithium titanate, no film | 40.1 |
| acetonitrile | Chekushkin2021: LiMn2O4 | 26.1 |
| acetonitrile | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 43.7 |
| water | Chekushkin2021: LiCoO2 | 15.4 |
| water | Chekushkin2021: LiMn2O4 | 19.3 |
| water | Nakayama2007: LiMn2O4 thin film, water and PC | 24.0 |

### Pairs of solvents in more than one series

For every pair of solvents that two or more series measured: the value in the first solvent minus the value in the second, in each series. A difference that changes from series to series is not a property of the two solvents alone.

| First solvent | Second solvent | Series | First minus second, kJ/mol |
|---|---|---|---:|
| DMC | EC:DMC 1:1 | Ishihara2014: lithium titanate, no film | -17.0 |
| DMC | EC:DMC 1:1 | Xu2010: lithium titanate, no film | -1.4 |
| DMC | EC:DMC 1:1 | Yamada2009: graphite, one film formed in EC:DMC 1:1 | -18.0 |
| DMC | EC:DMC 1:1 | Yamada2009: graphite, one film formed with pentafluorostyrene | -4.0 |
| DMC | EC:DMC 1:1 | Yamada2010: SiO, film formed in each electrolyte | -1.0 |
| DMC | EC:DMC 1:1 | YamadaSagane2009: solid electrolyte, no film | -19.0 |
| DMC | EC:DMC 1:9 | Yamada2009: graphite, one film formed in EC:DMC 1:1 | -15.0 |
| DMC | EC:DMC 1:9 | YamadaSagane2009: solid electrolyte, no film | -21.0 |
| DMC | PC | Ishihara2014: lithium titanate, no film | -27.0 |
| DMC | PC | Yamada2010: SiO, film formed in each electrolyte | +1.0 |
| DMSO | PC | Abe2005: solid electrolyte, no film | +12.6 |
| DMSO | PC | Fukutsuka2010: carbonaceous thin film, one film formed in EC:DEC 1:1 | +11.1 |
| DMSO | PC | Yamada2010: SiO, film formed in each electrolyte | -1.0 |
| EC:DEC 1:1 | EC:DMC 1:1 | Zabara2024: lithium metal, after ten cycles, film formed in each electrolyte | +0.2 |
| EC:DEC 1:1 | EC:DMC 1:1 | Zabara2024: lithium metal, as assembled, film formed in each electrolyte | +1.1 |
| EC:DEC 1:1 | EC:EMC 7:3 | Zabara2024: lithium metal, after ten cycles, film formed in each electrolyte | -16.6 |
| EC:DEC 1:1 | EC:EMC 7:3 | Zabara2024: lithium metal, as assembled, film formed in each electrolyte | -0.8 |
| EC:DEC 1:1 | FEC:DMC 3:7 | Nasara2020: lithium titanate thin film, film formed in each electrolyte | +23.0 |
| EC:DEC 1:1 | FEC:DMC 3:7 | Nasara2020: lithium titanate thin film, no film | +20.0 |
| EC:DEC 1:1 | PC | Doi2004Carbon: carbon treated by hot isostatic pressing, film formed in each electrolyte | +5.7 |
| EC:DEC 1:1 | PC | Doi2004Carbon: glassy carbon GC-20 not heat-treated, film formed in each electrolyte | -1.3 |
| EC:DEC 1:1 | PC | Doi2005: glassy carbon GC-10, film formed in each electrolyte | -1.5 |
| EC:DEC 1:1 | PC | Doi2005: glassy carbon GC-20 heat-treated at 773 K, film formed in each electrolyte | -3.7 |
| EC:DEC 1:1 | PC | Fukutsuka2010: carbonaceous thin film, one film formed in EC:DEC 1:1 | -11.0 |
| EC:DEC 1:1 | PC | Nasara2020: lithium titanate thin film, film formed in each electrolyte | +0.0 |
| EC:DEC 1:1 | PC | Nasara2020: lithium titanate thin film, no film | -10.0 |
| EC:DMC 1:1 | EC:DMC 1:9 | Uchida2020: graphite, LiFSA, film formed in each electrolyte | +13.4 |
| EC:DMC 1:1 | EC:DMC 1:9 | Uchida2020: graphite, LiPF6, film formed in each electrolyte | -18.2 |
| EC:DMC 1:1 | EC:DMC 1:9 | Yamada2009: graphite, one film formed in EC:DMC 1:1 | +3.0 |
| EC:DMC 1:1 | EC:DMC 1:9 | YamadaSagane2009: solid electrolyte, no film | -2.0 |
| EC:DMC 1:1 | EC:EMC 7:3 | Zabara2024: lithium metal, after ten cycles, film formed in each electrolyte | -16.7 |
| EC:DMC 1:1 | EC:EMC 7:3 | Zabara2024: lithium metal, as assembled, film formed in each electrolyte | -1.9 |
| EC:DMC 1:1 | PC | Ishihara2014: lithium titanate, no film | -10.0 |
| EC:DMC 1:1 | PC | Yamada2010: SiO, film formed in each electrolyte | +2.0 |
| FEC:DMC 3:7 | PC | Nasara2020: lithium titanate thin film, film formed in each electrolyte | -23.0 |
| FEC:DMC 3:7 | PC | Nasara2020: lithium titanate thin film, no film | -30.0 |
| PC | water | Chekushkin2021: LiCoO2 | +46.3 |
| PC | water | Nakayama2007: LiMn2O4 thin film, water and PC | +26.0 |

## The reference: a regression on the binding energy of Li+ with the solvent

The hypothesis that the removal of the solvent shell sets the barrier has a version with one number: the activation energy rises with the binding energy of Li+ with one molecule of the solvent. It is the reference that a field model of the solvent has to be better than (part (c) of the gate).

### The binding energy of Li+ with one molecule of each solvent

Computed with ORCA: the geometry with `B97-3c TightSCF`, the energy with `B3LYP D3BJ def2-TZVPP RIJCOSX def2/J TightSCF`. The value is the energy of the ion and of the free molecule minus the energy of the complex, each at its own optimised geometry, in the gas phase and with no thermal correction. A larger number is a stronger binding. The starting geometries come from the builder of plan item BT4, which optimises them with GFN2-xTB from 5 starts. *Published* is the value that a paper of the label table gives for the same complex, calculated with B3LYP: a reaction enthalpy or a difference of electronic energies, and, where marked, the position of a marker in a figure. The ORCA runs of the 34 species took 243 minutes of processor time in all.

| Solvent | Binding energy, kJ/mol | Geometry from | Donor atoms at the ion | Shortest distance to the ion, angstrom | With GFN2-xTB alone, kJ/mol | Published, kJ/mol |
|---|---:|---|---:|---:|---:|---|
| water | 155.5 | ORCA | 1 | 1.89 | 187.8 | none |
| dimethyl ether (in no series) | 173.4 | ORCA | 1 | 1.85 | 192.9 | none |
| THF | 196.7 | ORCA | 1 | 1.83 | 208.5 | 182.5 (figure) |
| acetonitrile | 197.0 | ORCA | 1 | 1.93 | 209.3 | 185.2 (figure) |
| DMC | 197.5 | ORCA | 1 | 1.78 | 235.3 | 185.8 |
| FEC | 199.1 | ORCA | 1 | 1.80 | 217.9 | 190.3 |
| EMC | 201.6 | ORCA | 1 | 1.77 | 243.0 | none |
| 2,2,2-trifluoroethyl acetate | 207.5 | ORCA | 2 | 1.84 | 258.9 | 161.1 (figure) |
| DEC | 207.8 | ORCA | 1 | 1.77 | 249.7 | none |
| EC | 220.4 | ORCA | 1 | 1.78 | 245.8 | 210.3 |
| PC | 226.9 | ORCA | 1 | 1.78 | 254.1 | 217.1, 213.7 (figure) |
| DMF | 243.6 | ORCA | 1 | 1.76 | 280.7 | 243.1 (figure) |
| DMSO | 250.2 | ORCA | 1 | 1.75 | 306.9 | 235.5 |
| hexamethylphosphoramide (in no series) | 268.3 | GFN2-xTB | 1 | 1.65 | 317.3 | 273.7 |
| diglyme | 363.0 | GFN2-xTB | 3 | 1.83 | 405.2 | none |
| poly(ethylene glycol) of molar mass 222 | 488.4 | GFN2-xTB | 5 | 1.93 | 524.4 | none |

A solvent of more than 20 atoms was not optimised with ORCA, because the ORCA build of this machine runs on one core. It keeps the geometry of GFN2-xTB and ORCA gives the energy only. What that changes was measured on the 13 smaller solvents, which were computed both ways: the binding energy at the geometry of GFN2-xTB is different by -1.8 kJ/mol on average, with a standard deviation of 1.3 kJ/mol and a largest difference of 5.8 kJ/mol.

The polyether of the label table is given by its paper with a molar mass of 222 and no formula. Tetraethylene glycol dimethyl ether has that molar mass and was computed in its place.

Against the 11 published values, the values of ORCA have a correlation of 0.936, and they are higher by 12.3 kJ/mol on average, with a standard deviation of the difference of 12.8 kJ/mol. The values of GFN2-xTB, the method of plan item BT4, have a correlation of 0.803, a mean difference of 44.6 kJ/mol and a standard deviation of 22.0 kJ/mol. For this reason the regression uses the values of ORCA.

A solvent can bind the ion in two ways. The table gives the complex of lower energy, and a paper may have computed the other one. For 2,2,2-trifluoroethyl acetate the two complexes give 177.3 and 207.5 kJ/mol; the one of lower energy holds the ion with 2 atoms. Without this solvent, the 10 other published values and the values of ORCA have a correlation of 0.986, a mean difference of 8.9 kJ/mol and a standard deviation of the difference of 6.4 kJ/mol; for GFN2-xTB the three numbers are 0.948, 39.3 and 13.8.

### The regression, with one solvent held out at a time

The regression has one slope for all series and one intercept for each series, so it describes only the differences inside a series. Every solvent is held out once. The *mean of the series* predicts a held-out value by the mean of the other values of its series, and the regression adds the slope times the distance of the binding energy from the mean of those other values. The error is the mean absolute error over the held-out values, in kJ/mol. The *gain* is the error of the mean minus the error of the regression, with a 95 % interval from a bootstrap over the held-out solvents (2000 draws). The last column is a test against chance: the binding energies are exchanged at random between the solvents, and the column gives the share of the exchanges whose error is not above the error with the true binding energies.

| Points of the regression | Series | Values | Solvents | Slope | Correlation inside the series | Error of the mean of the series | Error of the regression | Gain, with its 95 % interval | Exchanges that do as well |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| single solvents only | 10 | 30 | 13 | +0.085 | 0.43 | 16.2 | 16.5 | -0.2 [-5.9, +2.8] | 27.5 % |
| single solvents, without the two that wrap the ion | 10 | 28 | 11 | +0.430 | 0.82 | 17.0 | 9.4 | +7.7 [+0.8, +12.8] | 0.1 % |
| single solvents and mixtures; a mixture takes its strongest solvent | 23 | 74 | 14 | +0.084 | 0.36 | 11.0 | 12.2 | -1.1 [-3.9, +1.2] | 68.3 % |
| the same, without the two solvents that wrap the ion | 23 | 72 | 12 | +0.443 | 0.78 | 11.7 | 6.9 | +4.8 [+3.1, +7.3] | 0.1 % |
| single solvents; the two that wrap the ion take the binding energy of one dimethyl ether | 10 | 30 | 13 | +0.410 | 0.78 | 16.2 | 9.5 | +6.7 [-0.8, +12.9] | 0.2 % |
| single solvents and mixtures; the two that wrap the ion take the binding energy of one dimethyl ether | 23 | 74 | 14 | +0.398 | 0.73 | 11.0 | 7.7 | +3.4 [+0.7, +7.5] | 0.0 % |
| single solvents, with the binding energies that the papers give | 6 | 17 | 8 | +0.370 | 0.76 | 12.1 | 9.3 | +2.8 [+0.1, +6.7] | 0.6 % |

The repeatability between laboratories, from the reference system above, is 4.8 kJ/mol. The slope is in kJ/mol of activation energy per kJ/mol of binding energy. A mixture is held out once for each of its solvents, so it is counted more than once in the rows with mixtures.

### What a model of the solvent alone cannot remove

Some electrolytes were held out in two or more series. The errors of the regression for one electrolyte differ between its series. A better model of the solvent can move these errors by one amount for all series and no more, so the mean distance of the errors from their median is the smallest error that any model of the solvent alone can reach on these values.

| Points of the regression | Values of electrolytes in two or more series | Error of the regression on them | Smallest error of a model of the solvent alone |
|---|---:|---:|---:|
| single solvents only | 23 | 14.9 | 9.4 |
| single solvents, without the two that wrap the ion | 23 | 9.1 | 8.0 |
| single solvents and mixtures; a mixture takes its strongest solvent | 64 | 10.9 | 6.8 |
| the same, without the two solvents that wrap the ion | 64 | 7.1 | 5.3 |
| single solvents; the two that wrap the ion take the binding energy of one dimethyl ether | 23 | 8.7 | 8.2 |
| single solvents and mixtures; the two that wrap the ion take the binding energy of one dimethyl ether | 64 | 6.9 | 5.4 |
| single solvents, with the binding energies that the papers give | 13 | 9.4 | 8.3 |

### The slope of each series, single solvents only

| Series | Values | Range of the binding energy, kJ/mol | Range of the activation energy, kJ/mol | Slope of the series alone |
|---|---:|---:|---:|---:|
| Abe2005: solid electrolyte, no film | 3 | 51.1 | 38.4 | +0.758 |
| Chekushkin2021: LiCoO2 | 2 | 71.4 | 46.3 | +0.649 |
| Chekushkin2021: LiMn2O4 | 3 | 207.5 | 15.4 | +0.068 |
| Fukutsuka2010: carbonaceous thin film, one film formed in EC:DEC 1:1 | 2 | 23.3 | 11.1 | +0.477 |
| Ishihara2014: lithium titanate, no film | 2 | 29.4 | 27.0 | +0.918 |
| Nakayama2007: LiMn2O4 thin film, water and PC | 5 | 71.4 | 27.0 | +0.364 |
| Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 5 | 46.9 | 29.1 | +0.334 |
| Xu2007: graphite, film formed in each electrolyte | 2 | 18.9 | 18.0 | +0.952 |
| Xu2010: lithium titanate, no film | 3 | 291.8 | 10.6 | +0.017 |
| Yamada2010: SiO, film formed in each electrolyte | 3 | 52.7 | 2.0 | -0.038 |

### The two solvents that wrap the ion, with one donor group

A solvent that wraps the ion with several oxygen atoms has a binding energy that is the sum over its donors. The column *Regression, total* gives it that value, and the column *Regression, one donor* gives it the binding energy of one dimethyl ether, the molecule with one such group. Each solvent is held out once, as in the other sections.

| Held-out solvent | Series | Measured, kJ/mol | Mean of the series | Regression, total | Regression, one donor |
|---|---|---:|---:|---:|---:|
| diglyme | Chekushkin2021: LiMn2O4 | 34.7 | 22.7 | 39.9 | 21.5 |
| poly(ethylene glycol) of molar mass 222 | Xu2010: lithium titanate, no film | 50.2 | 45.4 | 101.9 | 35.3 |

### The held-out values: single solvents only

| Held-out solvent | Series | Measured, kJ/mol | Mean of the series | Regression | Error of the regression |
|---|---|---:|---:|---:|---:|
| 2,2,2-trifluoroethyl acetate | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 28.5 | 47.4 | 46.6 | +18.1 |
| DMC | Xu2010: lithium titanate, no film | 50.7 | 45.1 | 29.9 | -20.8 |
| DMC | Ishihara2014: lithium titanate, no film | 37.0 | 64.0 | 60.9 | +23.9 |
| DMC | Yamada2010: SiO, film formed in each electrolyte | 31.0 | 29.5 | 25.2 | -5.8 |
| DMF | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 57.6 | 40.1 | 43.0 | -14.6 |
| DMSO | Abe2005: solid electrolyte, no film | 69.9 | 44.4 | 47.4 | -22.5 |
| DMSO | Fukutsuka2010: carbonaceous thin film, one film formed in EC:DEC 1:1 | 82.3 | 71.2 | 73.0 | -9.3 |
| DMSO | Yamada2010: SiO, film formed in each electrolyte | 29.0 | 30.5 | 33.5 | +4.5 |
| EC | Xu2007: graphite, film formed in each electrolyte | 71.1 | 53.1 | 54.7 | -16.4 |
| EMC | Xu2007: graphite, film formed in each electrolyte | 53.1 | 71.1 | 69.5 | +16.4 |
| FEC | Abe2005: solid electrolyte, no film | 31.5 | 63.6 | 60.6 | +29.1 |
| PC | Abe2005: solid electrolyte, no film | 57.3 | 50.7 | 50.8 | -6.5 |
| PC | Ishihara2014: lithium titanate, no film | 64.0 | 37.0 | 38.5 | -25.5 |
| PC | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 45.7 | 43.1 | 43.8 | -1.9 |
| PC | Chekushkin2021: LiCoO2 | 61.8 | 15.4 | 19.0 | -42.8 |
| PC | Fukutsuka2010: carbonaceous thin film, one film formed in EC:DEC 1:1 | 71.2 | 82.3 | 81.2 | +10.0 |
| PC | Nakayama2007: LiMn2O4 thin film, water and PC | 50.0 | 24.0 | 27.5 | -22.5 |
| PC | Yamada2010: SiO, film formed in each electrolyte | 30.0 | 30.0 | 30.2 | +0.2 |
| THF | Xu2010: lithium titanate, no film | 40.1 | 50.4 | 37.5 | -2.6 |
| THF | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 42.4 | 43.9 | 41.9 | -0.5 |
| acetonitrile | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 43.7 | 43.6 | 41.6 | -2.1 |
| acetonitrile | Chekushkin2021: LiMn2O4 | 26.1 | 27.0 | 21.6 | -4.5 |
| diglyme | Chekushkin2021: LiMn2O4 | 34.7 | 22.7 | 39.9 | +5.2 |
| poly(ethylene glycol) of molar mass 222 | Xu2010: lithium titanate, no film | 50.2 | 45.4 | 101.9 | +51.7 |
| water | Chekushkin2021: LiCoO2 | 15.4 | 61.8 | 58.1 | +42.7 |
| water | Chekushkin2021: LiMn2O4 | 19.3 | 30.4 | 24.1 | +4.8 |
| water | Nakayama2007: LiMn2O4 thin film, water and PC | 24.0 | 50.0 | 46.4 | +22.4 |
| water | Nakayama2007: LiMn2O4 thin film, water and PC | 24.0 | 50.0 | 46.4 | +22.4 |
| water | Nakayama2007: LiMn2O4 thin film, water and PC | 25.0 | 50.0 | 46.4 | +21.4 |
| water | Nakayama2007: LiMn2O4 thin film, water and PC | 23.0 | 50.0 | 46.4 | +23.4 |

### The held-out values: single solvents, without the two that wrap the ion

| Held-out solvent | Series | Measured, kJ/mol | Mean of the series | Regression | Error of the regression |
|---|---|---:|---:|---:|---:|
| 2,2,2-trifluoroethyl acetate | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 28.5 | 47.4 | 43.8 | +15.3 |
| DMC | Xu2010: lithium titanate, no film | 50.7 | 40.1 | 40.5 | -10.2 |
| DMC | Ishihara2014: lithium titanate, no film | 37.0 | 64.0 | 50.6 | +13.6 |
| DMC | Yamada2010: SiO, film formed in each electrolyte | 31.0 | 29.5 | 10.7 | -20.3 |
| DMF | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 57.6 | 40.1 | 55.6 | -2.0 |
| DMSO | Abe2005: solid electrolyte, no film | 69.9 | 44.4 | 61.1 | -8.8 |
| DMSO | Fukutsuka2010: carbonaceous thin film, one film formed in EC:DEC 1:1 | 82.3 | 71.2 | 81.7 | -0.6 |
| DMSO | Yamada2010: SiO, film formed in each electrolyte | 29.0 | 30.5 | 47.5 | +18.5 |
| EC | Xu2007: graphite, film formed in each electrolyte | 71.1 | 53.1 | 61.1 | -10.0 |
| EMC | Xu2007: graphite, film formed in each electrolyte | 53.1 | 71.1 | 63.1 | +10.0 |
| FEC | Abe2005: solid electrolyte, no film | 31.5 | 63.6 | 48.0 | +16.5 |
| PC | Abe2005: solid electrolyte, no film | 57.3 | 50.7 | 51.5 | -5.8 |
| PC | Ishihara2014: lithium titanate, no film | 64.0 | 37.0 | 47.0 | -17.0 |
| PC | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 45.7 | 43.1 | 48.4 | +2.7 |
| PC | Chekushkin2021: LiCoO2 | 61.8 | 15.4 | 39.7 | -22.1 |
| PC | Fukutsuka2010: carbonaceous thin film, one film formed in EC:DEC 1:1 | 71.2 | 82.3 | 74.4 | +3.2 |
| PC | Nakayama2007: LiMn2O4 thin film, water and PC | 50.0 | 24.0 | 48.2 | -1.8 |
| PC | Yamada2010: SiO, film formed in each electrolyte | 30.0 | 30.0 | 31.0 | +1.0 |
| THF | Xu2010: lithium titanate, no film | 40.1 | 50.7 | 50.3 | +10.2 |
| THF | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 42.4 | 43.9 | 34.1 | -8.3 |
| acetonitrile | Sawayama2025: graphite, film formed in each electrolyte with ethylene sulfite | 43.7 | 43.6 | 33.5 | -10.2 |
| acetonitrile | Chekushkin2021: LiMn2O4 | 26.1 | 19.3 | 38.5 | +12.5 |
| water | Chekushkin2021: LiCoO2 | 15.4 | 61.8 | 31.9 | +16.5 |
| water | Chekushkin2021: LiMn2O4 | 19.3 | 26.1 | 8.7 | -10.6 |
| water | Nakayama2007: LiMn2O4 thin film, water and PC | 24.0 | 50.0 | 20.2 | -3.8 |
| water | Nakayama2007: LiMn2O4 thin film, water and PC | 24.0 | 50.0 | 20.2 | -3.8 |
| water | Nakayama2007: LiMn2O4 thin film, water and PC | 25.0 | 50.0 | 20.2 | -4.8 |
| water | Nakayama2007: LiMn2O4 thin film, water and PC | 23.0 | 50.0 | 20.2 | -2.8 |

## Papers that are cited or known and have not been read

| Key | Reference | State |
|---|---|---|
