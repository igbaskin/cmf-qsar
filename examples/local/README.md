# Three targets that belong to atoms

Every file in this folder carries a measured value **per atom** rather than per molecule, which is
what plan item LP1 added to the data model and what plan item LP2 builds a model on. The value sits
on the atoms themselves, written in RDKit's own convention: the molecule property
`atom.dprop.<name>` holds one field per atom, with `n/a` where the atom was not measured. The
molecule property `activity` holds a per-molecule number as well, so that each file can also be
read as an ordinary data set.

`examples/make_local_examples.py` produced all three files from free downloads; its docstring gives
the download commands. The downloads themselves are not in this repository.

## `shifts_cdcl3.sdf` — 400 molecules, 5096 assigned carbons

13C NMR shifts, in ppm, on the carbon each was measured for (`atom.dprop.shift`).

* **Source.** nmrshiftdb2, the open NMR database of the University of Cologne,
  <https://nmrshiftdb.nmr.uni-koeln.de/>, file `nmrshiftdb2withsignals.sd` as distributed through
  SourceForge (<https://sourceforge.net/projects/nmrshiftdb2/files/data/>, 158 569 118 bytes). The
  record identifier is kept as the molecule name, so every molecule here can be looked up there.
* **Selection.** Of its 58 187 records, the 43 258 with a 13C spectrum were filtered to those
  measured in deuterated chloroform (the solvent of the first spectrum, which most records leave
  unreported, and a shift depends on it), neutral single-piece molecules of C, H, N, O, F, S, Cl,
  Br and I with 5 to 24 heavy atoms, at least three assigned carbons, every assigned atom a carbon,
  every shift between -20 and 240 ppm, and one molecule per InChI key. The first 400 such records
  were kept.
* **What was changed.** The deposited structures are two-dimensional. A conformation was generated
  here with ETKDG and refined with MMFF94, with the atom order left untouched so that a measured
  atom keeps the index the spectrum assigns it. Where a record reports several signals for one
  atom, their mean is used.
* **Licence.** The project states that its spectra may be used without restrictions and names an
  "nmrshiftdb2 Data License" for the database as a whole. That licence page could not be retrieved
  on 27 October 2026 (the server refused the connection), so the exact terms are not recorded here
  and the status of this copy is **unclear** in the sense of `docs/public_3dqsar_datasets.md`:
  redistribution is customary and the project's own statements invite it, but an explicit permission
  has not been read. The file is a derived subset, not the database. Replacing it costs one new
  download and one run of the script.

## `carboxylic_pka.sdf` — 912 carboxylic acids

The acidity of the carboxyl group, in pKa units, on the carboxyl carbon (`atom.dprop.pka`).

* **Source.** The data sets collected by Marcel Baltruschat and Paul Czodrowski, *Machine learning
  meets pKa*, F1000Research 9:113 (2020), <https://doi.org/10.5281/zenodo.7884512>, files
  `AvLiLuMoVe.sdf` (literature values for neutral molecules) and
  `combined_training_datasets_unique.sdf` (values extracted from ChEMBL, molecules stored in the
  ionised form) of
  <https://github.com/czodrowskilab/Machine-learning-meets-pKa/tree/master/datasets>.
* **Licence.** CC BY 4.0, so this copy may be redistributed; the attribution above is the
  condition, and the changes below are the "indicate changes" part of it.
* **Selection.** Molecules with exactly one carboxylic acid group, no other acidic group
  (sulfonic, phosphonic, sulfinic, tetrazole), neutral once the carboxylate is protonated back,
  single-piece, of the same elements, 4 to 30 heavy atoms, and a reported value between 0.5 and
  6.5. That window is what makes the site unambiguous: a molecule of this shape with a value below
  6.5 can only have been measured on its carboxyl group, because a protonated amine is a base and
  sits above 7. One molecule per InChI key, with the mean of repeated values; the repeats agree to
  within 0.35 units.
* **What was changed.** A carboxylate was protonated back to the acid, since acidity is a property
  of the acid. The structures were given a three-dimensional conformation here in the same way as
  the shift set. The target was placed on the carboxyl carbon found by substructure match.

## `metabolism.sdf` — 636 drug-like molecules, 1301 sites among 14 488 heavy atoms

Where a cytochrome P450 attacks the molecule, as one number per heavy atom (`atom.dprop.som`):
one if the atom is a site of metabolism, zero if it is not. Hydrogens are left unmeasured, so a
model is asked which heavy atom is attacked.

* **Source.** The benchmark of Jed Zaretzki, Matthew Matlock and S. Joshua Swamidass, *XenoSite:
  accurately predicting CYP-mediated sites of metabolism with neural networks*, J. Chem. Inf.
  Model. 53 (2013) 3373, <https://doi.org/10.1021/ci400518g>, which collects the substrates of
  nine P450 isoforms from the literature. The copy used here is the preprocessed file
  `data/zaretzki_preprocessed.sdf` of the FAME.AL work of Molecular Informatics Vienna,
  <https://github.com/molinfo-vienna/FAME.AL>, 657 records.
* **Licence.** The FAME.AL repository is MIT licensed, so this copy may be redistributed; the
  attribution above is the condition.
* **Selection.** Records whose site list is readable and points at heavy atoms only, neutral
  single-piece molecules of the usual elements with 5 to 70 heavy atoms, one molecule per InChI
  key, and a conformation that could be generated. That leaves 636 of the 657.
* **What was changed.** The sites are merged over the isoforms, as the deposited file already
  merges them, so the target here is whether the atom is attacked at all and not which enzyme
  attacks it. The deposited structures are two-dimensional; a conformation was generated here with
  ETKDG and refined with MMFF94, with the atom order left untouched so that a site keeps the index
  the record gives it. The per-molecule `activity` property holds the number of sites, which is
  not a useful target and is there only so that the file also reads as an ordinary data set.

## What they are used for

`benchmarks/run_local.py` measures the shift and the acidity sets: the field around an atom
(`cmfqsar/localenv.py`) against an atom-centred count of atom types in distance shells, on folds
that hold out whole molecules. `benchmarks/results/local/table.md` holds the result.
`benchmarks/run_metabolism.py` measures the metabolism set, where the question is a ranking inside
each molecule rather than a number, and adds a third reference: the same atom types counted over
**bonds** instead of over distance, which is what a model with no geometry has.
`benchmarks/results/metabolism/table.md` holds that result. `cmf local` fits any of the three sets
from the command line.
