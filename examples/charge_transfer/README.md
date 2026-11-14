# Measured activation energies of charge transfer at an electrode

This folder holds the labels of plan item BT12: activation energies of the charge-transfer
reaction of Li+ at the surface of an electrode, collected from the literature on 13 and 14
November 2026. The *activation energy* is the slope of an Arrhenius plot of the charge-transfer
resistance, and the resistance is read from an impedance spectrum. Unlike the exchange-current
density itself, it does not depend on the active area of the electrode, which is not known for a
porous electrode.

    python benchmarks/run_charge_transfer.py     # the summary, in benchmarks/results/charge_transfer

## The files

* `activation_energies.csv`: one row per value, 90 rows from 16 studies.
* `sources.csv`: one row per paper, with its DOI, its licence and what was done with it.
* `papers/`: the place for the PDF files of the papers. The files are not in the repository:
  `.gitignore` leaves out every PDF except the manual.

## How a value was taken

A value is in the table only if it was seen in a document that was read for this collection.
Nothing was written from memory.

* A row is **primary** (`read_from`) when the number was read in the paper that measured it.
  There are 88 such rows, from 15 papers. Five of the papers are open: four under CC BY 4.0
  and one as an open file on the site of its society. The other ten are behind a paywall and
  were read from files that the author of the program supplied.
* A row is **secondary** when the number was read in another paper that cites it. There are 2
  such rows: two values of Yamada, Sagane 2009 that Table I of Ishihara 2014 cites. `source`
  is then the paper that measured the value, and `cited_in` names the paper and the place
  where the number was read. A secondary value has to be checked against its own paper before
  it is used for more than a first estimate.
* The first collection, of 13 November, held 20 secondary rows. All of them were checked
  against their own papers on 14 November, and two differ from their papers. For graphite in
  1 M LiClO4 in EC:DMC 1:1 the review gives 55 kJ/mol; the paper of Yamada 2009 gives 58, and
  55 for EC:DMC 1:9. For the film formed with vinylene carbonate a table of another paper
  gives 0.60 eV; the paper gives 56 kJ/mol. The `note` of each such row says so.
* Four papers print most of their values only inside a figure, in its legend or next to its
  lines: Xu 2007, Xu 2010, Abraham 2006 and Abraham 2008. `location` then names the figure.
* One paper, Sawayama 2025, does not print its values in the article. They were read from the
  image of its Figure 3b: the position of each marker was measured against the tick marks of
  the axis. Such a reading is good to about 1 kJ/mol, and the `note` of each row says so.
* The text of every file was extracted with `pdftotext`. A table was read a second time in the
  reading order of the page (`pdftotext -raw`), because the layout mode moved the rows of one
  table against each other. A figure was rendered with PyMuPDF and read by eye.

## The columns of `activation_energies.csv`

| Column | Meaning |
|---|---|
| `id` | the paper and a running number |
| `electrode`, `electrode_class` | the material as the paper names it, and its class: graphite, non-graphitizable carbon, layered oxide, spinel oxide, olivine, lithium titanate, solid electrolyte |
| `form` | composite electrode, thin film, symmetric cell, full cell with a reference electrode, and so on, as the paper describes it |
| `state` | the potential or the state of the cell, and its history where the paper gives one |
| `salt`, `salt_molar` | the lithium salt and its concentration in mol/L; empty where the paper gives mol per kg of solvent or no concentration, and the `note` then says which |
| `solvent`, `solvent_class` | the solvents with their ratio, and a class: EC-based carbonate, linear carbonate only, PC, FEC, ether, DMSO, nitrile, amide, fluorinated ester, water, ionic liquid, polymer |
| `additive` | an additive of the electrolyte, or `none` |
| `film` | the surface film: `formed` (after the formation cycles in the electrolyte of the row), `cycled`, `formed and cycled` (a mean over both), `aged`, `preformed in ...` (formed in another electrolyte), `none expected`, `not stated` |
| `ea_reported`, `ea_sd_reported` | the value and its stated deviation, in the unit of the paper |
| `ea_low_reported`, `ea_high_reported` | the two ends, when the paper gives a range and no single value |
| `unit` | `kJ/mol`, `eV` or `kcal/mol`; the report multiplies eV by 96.485 and kcal/mol by 4.184 |
| `temperature_range` | the temperatures of the Arrhenius fit, where read |
| `method` | the measurement and which resistance was fitted |
| `source`, `location` | the paper that measured the value (a key of `sources.csv`) and the place in it |
| `read_from`, `cited_in` | `primary` or `secondary`, and for a secondary value where it was read |
| `solvent_series` | the name of a *series*: a set of rows of one study that share the electrode and the film, or the way the film was made, and differ in the solvent; empty for a row that is in no series |
| `note` | what a user of the row has to know |

## What the set is and is not

* The values are facts taken from published papers, and every row names its paper. No text,
  table or figure of a paper is copied here.
* Seven rows (`A05-*` and `YS09-*`) are not electrodes: they are the transfer of the ion
  between a solid and a liquid electrolyte. They are kept because they show the effect of the
  solvent with no electrode and no surface film, and the report leaves them out of the
  electrode means.
* Four rows (`K19-*`, `A04-01` and `A05-04`) are ranges and not single values. The report
  uses the middle of a range only where it says so.
* The papers do not all fit the same resistance. Most fit the charge-transfer resistance of an
  equivalent circuit. Abraham 2006 and 2008 take the width of the whole arc at middle
  frequencies, and Chekushkin 2021 calls its value the activation energy of a kinetic
  resistance. `method` says which.
* The series do not agree with each other on the one pair of solvents that more than one of
  them measured. With no surface film on lithium titanate, DMC is lower than EC:DMC 1:1 by
  17 kJ/mol in one study and by 1.4 kJ/mol in another. The report prints this comparison.
* One paper argues against the hypothesis that desolvation alone sets the barrier:
  Chekushkin 2021 (Electrochim. Acta 372, 137843). Its six values are in the table.
* The set is not complete. Two documents are known and have not been read: Yamada, Sagane
  2009, and one file of the Kyoto University repository. The supporting information of
  Sawayama 2025 was not read either.
* `sources.csv` gives the DOI of every paper that has one. The DOIs were taken from the
  Crossref registry and checked against the journal, the volume and the page.
