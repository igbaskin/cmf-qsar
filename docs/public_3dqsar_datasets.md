# Data statement

*Compiled 22 October 2026 (item A3 of `docs/development_plan.md`), from the survey of public
3D-QSAR collections of 15 October 2026, which is kept at the end. For every data set under
`examples/` this says what it is, where it came from, under what terms, which script produced it
and from what, which benchmark uses it, and what its status is under the rule of the plan: a copy
is kept only where redistribution is clear; otherwise the script and the pointer to the original
are the way to obtain it.*

The tests under `tests/` use none of these files; they build their molecules themselves. The
benchmarks under `benchmarks/` and the worked examples of the manual do use them, as the last
column says.

## Status, in three words

* **free** — an explicit open licence (MIT, CC0, ODbL) or produced by this work, so the copy may be
  redistributed with the program. Where the licence is share-alike, as ODbL is, the obligation is
  named in the set's own row and README.
* **derived** — the structures or properties were produced by this work, but from a published
  table of SMILES, constants or activities whose distributor states no licence; the facts are
  the paper's, the files are ours.
* **unclear** — a converted copy of a file whose distributor states no licence; redistribution
  is customary in the field but not explicitly permitted.

## The sets

| Folder or file | Content | Origin | Status | Script | Used by |
|---|---|---|---|---|---|
| `benzamides_synthetic.sdf` | 109 aligned N-phenylbenzamides with a synthetic activity | this work | free | `make_example_dataset.py` | manual, GUI autotest, CI |
| `hammett/` (6 files, 6.0 MB) | 95 monosubstituted benzenes in one frame, with σ_m, σ_p, π; symmetry images; ORCA charges | structures and charges this work; constants Hansch, Leo & Taft 1991 | derived | `make_hammett_examples.py`, `orca_atom_properties.py` | `run_hammett.py` |
| `steroids_cbg.sdf` | 31 CBG steroids, aligned, with log 1/K and the 21/10 split | structures: Erlangen corrected set (Wagener, Sadowski & Gasteiger 1995); affinities Cramer 1988; split Coats 1998 | unclear | none (re-superimposed here) | `run_alignment.py` |
| `sutherland/` (8 files, 6.3 MB) | the eight sets of the 2013 paper, 1245 molecules, aligned coordinates, Sybyl types and MNDO ESP charges, published split | ACS Supporting Information of Sutherland, O'Brien & Weaver 2004 | unclear | `make_sutherland_examples.py` from the MOL2 distribution | `run_topomer.py`, `run_alignment.py`, `docs/benchmark.md`; `run_sutherland.py` needs the MOL2 originals themselves |
| `mittal/` (38 files, 15 MB) | 38 of the 40 Mittal sets, 3D conformers generated and MCS-aligned here | ACS Supporting Information of Mittal, McKinnon & Sorich 2009 (SMILES and activities only) | derived | `make_mittal_examples.py` | `run_topomer.py --mittal`, `run_alignment.py` |
| `openff/` (21 sets, 513 ligands, 2.5 MB) | congeneric FEP series posed in the crystal site | Open Force Field protein-ligand-benchmark | free (MIT, licence file in the folder) | `make_fep_examples.py` | 86-set benchmark (`docs/benchmark.md`, third part) |
| `schindler/` (10 sets, 1.5 MB) | the complete Merck FEP benchmark with LigPrep poses | MCompChem fep-benchmark | free (MIT, licence file in the folder) | `make_schindler_examples.py` | 86-set benchmark |
| `ross/` (67 series, 1254 ligands, 5.8 MB) | the Schrodinger public binding free energy benchmark | schrodinger/public_binding_free_energy_benchmark | free (MIT, licence file in the folder; upstream `benchmark_metadata.csv` kept) | `make_ross_examples.py` | 86-set benchmark |
| `pockets/` (73 files, 2.9 MB) | binding sites trimmed from the crystal structures the FEP sets name, in their frame | RCSB PDB entries, provenance in each file's `REMARK 400` | free (CC0) | `make_pocket_examples.py` | GUI pocket display, complementarity, docking |
| `decoys/` (8 files, 3.5 MB) | posed background sets for ACE, AChE, GPB, THR, plain and property-matched | actives from `sutherland/`, decoys from the other example series | inherits `sutherland/` | `make_decoy_sets.py`, `_matched_decoys.py` | screening and one-class measurements of the README |
| `thrombin_atom_properties.sdf` | 40 THR molecules carrying per-atom Gasteiger and Crippen properties | `sutherland/THR.sdf` | inherits `sutherland/` | `make_property_example.py` | manual (`prop:` fields) |
| `am-eu/` (MOL2 + table) | 46 Am/Eu extractants of the 2015 paper with lg SF, superimposed, heavy atoms | the authors' own data (Sitnikov et al. 2015); the author of this program is one of them | free | none (as prepared for that paper) | `run_am_eu.py` |
| `corrosion/`, `corrosion_hcl/` (23 CSV each, 997 rows each) | congeneric series of iron corrosion inhibitors with the MWILA potency, per medium, and the Langmuir ΔG; the `_hcl` folder protonated at pH 0 | the author's own database of literature measurements, exported from its scaffold tree | free (the author's compilation); the underlying values are published, see below | `scaffold_tree_export.py` | `run_corrosion.py`, `run_corrosion_qm.py`, `run_multitask.py`, `run_cdft_kernels.py` |
| `cyclodextrin/` (7 tables, 2671 guests, 0.7 MB; `ph7/` the same protonated, 0.8 MB; `upstream/` 4.1 MB) | association constants with a fixed cyclodextrin host, one table per host, activity = log K; the host-guest case, with no frame | Tahıl et al., Chemical Data Collections 45 (2023) 101022, Zenodo 10.5281/zenodo.7575539 | free (ODbL 1.0, share-alike; licence file in `upstream/`) | `make_cyclodextrin_examples.py --fetch --protonate` | `run_cyclodextrin.py` |

Sizes are of the tracked files. The whole of `examples/` is 51 MB, of which the two items of
unclear status and the two that inherit it are 10 MB, and `mittal/`, derived, is 15 MB.

## Each set in detail

### Synthetic benzamides

Generated by `examples/make_example_dataset.py`: a congeneric series enumerated from a scaffold,
embedded with ETKDG and MMFF94, aligned on the scaffold, with an activity that is the overlap of
the molecular fields with a hidden pseudo-receptor co-field plus noise. Nothing in it comes from
outside the program. It is the example of the manual, the data set of the GUI autotest and so of
every CI job.

### Hammett substituents

`examples/make_hammett_examples.py` builds the structures and places them in the ring frame;
`examples/orca_atom_properties.py` computed the Hirshfeld, Mulliken, Löwdin and CHELPG charges
(r2SCAN-3c single points) carried by the `_qm` files; the `_sym` and `_ens` files are the symmetry
images and the MMFF conformer ensembles of the same molecules. All of that is this work. The
constants σ_m and σ_p are the measured values compiled by Hansch, Leo and Taft (Chem. Rev. 91
(1991) 165, Table I). They were first written down from memory of that table. **Checked against
the paper on 25 October 2026**: 39 of the 95 substituents were matched to a row of Table I and
38 agreed to the last digit; the one that did not, CH2OCH3, was written here as σ_m 0.02 /
σ_p 0.03 and is given by row 223 as **0.08 / 0.01**, which was confirmed on the page image before
the value was changed in the script and in all five SDF files. The other 56 could not be matched
automatically — the scan's character recognition mangles the substituent names, Cl coming out as
"c1" and SO2Me as "SOIMe" — and remain as good as the memory they came from.

**π is not from that paper.** The 1991 review tabulates σ, F and R and mentions π only in its
introduction; the hydrophobic constants here come from the Hansch–Leo compilations and have not
been checked against any of them. The script's provenance line said otherwise until the same date
and now says this.

### The CoMFA steroids

The 31 steroids of Cramer, Patterson and Bunce with their CBG affinities are the oldest benchmark
of the field and have been redistributed in many places; the structures here are the corrected
set of Wagener, Sadowski and Gasteiger (J. Am. Chem. Soc. 117 (1995) 7769), as served by the
TORVS service of the University of Erlangen, re-superimposed on the steroid skeleton by this
work, with the customary 21/10 split of Coats (1998). No licence was stated by the distributor;
whether the Erlangen service still serves the file was not checked. Another copy is in the
exeResearch GitHub collection (see the survey below). Status unclear; the set is one file of 31 molecules and is
used only by `run_alignment.py`, where it is one of nine targets.

### The Sutherland sets

The eight aligned sets of Sutherland, O'Brien and Weaver (J. Med. Chem. 47 (2004) 5541) are the
data of Table 2 of the 2013 paper and of most of this repository's measurements. The ACS
Supporting Information of the paper (`3dqsar_mol2.tar.gz`) is available free of charge but
carries no licence statement, so a converted copy cannot be said to be redistributable, however
widely these files circulate. The copies in `examples/sutherland/` were converted by
`examples/make_sutherland_examples.py` from the MOL2 distribution and the activity tables as the
2013 paper used them (a local copy; see the memory note on the literature data sets). The
Readme of the SDF mirror at the University of California, Irvine (Baldi group,
`https://ics.uci.edu/~baldig/learning/QSAR_Sutherland/`, checked 22 October 2026, files dated
2005) asks that the original compilers be cited beside the 2004 paper:

* ACE: DePriest, Mayer, Naylor, Marshall, J. Am. Chem. Soc. 115 (1993) 5372.
* GPB and THER: Gohlke, Klebe, J. Med. Chem. 45 (2002) 4153.
* THR: Böhm, Stürzebecher, Klebe, J. Med. Chem. 42 (1999) 458.
* AChE, BZR, COX-2, DHFR: compiled by Sutherland et al. themselves.

`benchmarks/run_sutherland.py`, which reproduces Table 2 of the 2013 paper, reads the MOL2
originals and their activity tables, not the SDF copies; it needs the ACS archive (or the local
copy) under `--data-root`. `run_topomer.py` and `run_alignment.py` read the SDF copies, as do the
decoy sets and the per-atom-property example, which are derived from them.

### The Mittal sets

The Supporting Information of Mittal, McKinnon and Sorich (J. Chem. Inf. Model. 49 (2009) 1810)
is a zip of 40 text files with SMILES and activities, again without a licence statement. What is
in `examples/mittal/` is not that file: the conformers were generated and the alignment made by
`examples/make_mittal_examples.py`, so the three-dimensional content is this work and only the
SMILES and the activity values are the paper's. Two sets are not produced because their SMILES
are corrupted in the SI. These are automatic alignments, meant for method comparison and not for
reproducing literature statistics, which the README says at each use.

### The free-energy collections

Three public collections of congeneric series posed in the crystal site, all under the MIT
licence, with the licence file kept in each folder: the Open Force Field
protein-ligand-benchmark (`openff/`, 21 targets, produced by `make_fep_examples.py` from the
repository's `data` tree, part of it from release 0.2.1 whose files are under git-lfs), the Merck
KGaA FEP benchmark of Schindler et al. (`schindler/`, the complete series, by
`make_schindler_examples.py`), and the Schrodinger public binding free energy benchmark of Ross
et al. (`ross/`, 67 series of five or more ligands, by `make_ross_examples.py`). The scripts
convert the affinities to a p-scale and leave the coordinates untouched; every record keeps the
tags that name its source, its PDB entry and, where the collection gives it, the DOI of the
measurement. The pockets in `examples/pockets/` are trimmed by `make_pocket_examples.py` from
the RCSB entries those tags name, which are public domain; each file records its entry, the
cutoff and the closest approach in `REMARK 400` lines.

### Decoys and the per-atom-property example

`examples/decoys/` holds, for four of the Sutherland targets, their actives and a background of
ligands of the other example series posed in the actives' frame (`make_decoy_sets.py`) and a
property-matched background in the manner of DUD-E (`_matched_decoys.py`).
`examples/thrombin_atom_properties.sdf` is forty THR molecules with computed per-atom
properties (`make_property_example.py`). Both are derived from `examples/sutherland/` and have
its status; both are regenerated by their scripts in a minute.

### The Am/Eu extractants

`examples/am-eu/AmEu1-all.mol2` and `AmEu1-all.txt` are the 46 superimposed extractants and
their lg SF(Am/Eu) of Sitnikov, Zhokhova, Ustynyuk, Varnek and Baskin (J. Comput.-Aided Mol.
Des. 29 (2015) 233), in the form the authors prepared for that paper (Sybyl types, Gasteiger
charges, no hydrogens, aromatic typing). The author of this program is an author of that paper
and the files are the authors' own; the separation factors were compiled from the literature
the 2015 paper cites. `run_am_eu.py` rebuilds the hydrogens for the
quantum stage and writes `benchmarks/results/am_eu/am_eu_qm.sdf`.

### The corrosion-inhibitor series

The 23 series in `examples/corrosion/` and their protonated forms in `examples/corrosion_hcl/`
are exported by `examples/scaffold_tree_export.py` from the scaffold tree of the author's
database of corrosion inhibitors of iron-based alloys, a compilation of published measurements:
the MWILA potency ln[θ/(C(1−θ))] per compound, pooled and per medium, and the Langmuir adsorption
free energy. The tree itself is a private page and the database is not public, so the CSV files
are the only form in which these series exist outside the author's hands, and redistributing
them is the author's decision alone. What they lack for a data statement is the reference of each
measurement: the rows carry the number of papers (`n_papers`) and a database id (`db_id`), not
the papers. Before the corrosion paper, the export should add a column with the DOIs of the
sources of every compound, or the database should be deposited and cited.

### The cyclodextrin guest sets

`examples/cyclodextrin/` is the only host-guest collection here and the only one with no frame at
all. `examples/make_cyclodextrin_examples.py --fetch` downloads the archive deposited by Tahıl,
Delorme, Le Berre, Monflier, Sayede and Tilloy (Chemical Data Collections 45 (2023) 101022,
<https://doi.org/10.5281/zenodo.7575539>, 488 965 bytes, MD5 `9260a9a8a86db1da967d91a22ece8cb4`),
checks that digest, unpacks it into `upstream/`, and reduces their curated table of 3459
measurements over 15 hosts to one activity table per host: 1162 β-CD guests, 827 α-CD, 292 γ-CD,
135 HP-β-CD, 135 SBE-β-CD, 77 methyl-β-CD and 43 2,6-di-O-methyl-β-CD. `hosts.csv` keeps the eight
hosts that fell below the twenty-guest threshold visible. The selection is one window of
conditions, 298 ± 2 K and pH 7 ± 0.5, and the median of log₁₀ K per guest inside it, with the count
and the spread of the pooled measurements in their own columns.

The licence is ODbL 1.0, which permits the copy kept here and asks for attribution, that the
licence travel with the data, and that a publicly used derived database be offered under the same
terms. The per-host tables are such a derived database and `examples/cyclodextrin/README.md` offers
them under ODbL 1.0 accordingly. This is the first set in the repository with a share-alike
condition, and anything built from these tables and then published inherits it.

Three things about these files are approximations rather than data, and the README says so at
length: the guest SMILES at the top level are the neutral forms as deposited in PubChem, so an acid
measured at pH 7 is written protonated; repeated measurements are reduced to their median,
excellent agreement in general but 3.2 log units apart for one α-CD guest; and 1:1 stoichiometry is
assumed throughout.

The first of those has the remedy that `examples/corrosion_hcl/` uses for 1 M HCl.
`--protonate` writes `cyclodextrin/ph7/`, the same seven tables with the same activities and the
structures in their state at pH 7 from Dimorphite-DL, the neutral form kept in `smiles_neutral`
and the net charge in `charge`. 882 of the 2671 guests change: 793 anions, 342 cations, 1536
unchanged. There is deliberately no cap on added protons, unlike the corrosion export, because the
reason for that cap is specific to pH 0, where the rules charge every ring nitrogen although the
first protonation makes the next site far less basic; at pH 7 only sites with a rule pKa above 7
are charged. It is worth having both copies: the neutral tables are what the published models of
this endpoint were built on and so the comparable ones, while the protonated tables are the honest
structures. Which predicts better is not settled: on a single split of the three natural hosts the
protonated copies were ahead on all three, but `run_cyclodextrin.py`, over all seven hosts with a
grid and ten-fold cross-validation, has the invariant kernel improving on four and losing on three,
the two largest sets among those that improve.

They are here for the alignment question. The host is fixed and the guests share no scaffold, so no
superposition exists to be curated, which is the regime where `run_alignment.py` finds the
rotation-invariant autocorrelation kernel of `cmfqsar/invariant.py` ahead of the aligned one. The
published field-based models of this endpoint are the benchmark to beat: CoMFA with COSMOsar3D
fields on 60 α-CD guests, GRIND on 126 and on 233 β-CD guests, all an order of magnitude smaller
than the tables here. `benchmarks/run_cyclodextrin.py` measures the aligned, invariant and count
kernels on every host, in both variants, with the grids of `run_alignment.py` so that the two
tables can be read beside each other.

## What is not in the repository

* The MOL2 originals and activity tables of the Sutherland sets, and the original CMF R scripts
  of the 2013 paper, which `run_sutherland.py` and `docs/benchmark.md` rely on. They are the ACS
  Supporting Information (`3dqsar_mol2.tar.gz`) and the archive of the 2013 paper; a local copy
  exists on the author's machine.
* The Mittal SI zip (`ci900117m_si_001.zip`), from which `make_mittal_examples.py` runs.
* The three free-energy repositories, which `make_fep_examples.py`, `make_schindler_examples.py`
  and `make_ross_examples.py` convert; each script takes the clone or download under
  `--data-root`.
* The corrosion database and its scaffold tree.
* ORCA outputs of the quantum stages (ignored by git); the SDF files that carry their per-atom
  results are kept.

## Before the repository is public

1. **Decide the four items of unclear status**: `sutherland/`, `steroids_cbg.sdf`, and the
   two derived from the first, `decoys/` and `thrombin_atom_properties.sdf`. The choices are to
   ask ACS and the Erlangen group for permission, to replace the copies by the scripts and a
   note that says where to download the originals (`run_topomer.py` and `run_alignment.py`
   would then need `make_sutherland_examples.py --data-root` to have been run first, and the
   decoys and the property example would be regenerated by their scripts), or to keep them on
   the ground that the files have circulated for twenty years and the UCI mirror serves them
   openly. The plan's rule says the second; the decision is the author's. **Decided 22 October
   2026: the copies stay while the repository is private; the choice is made again before it
   goes public.**
2. `mittal/` is derived and can stay, with the sentence above about what in it is the paper's.
3. Add the per-compound references to the corrosion exports, or deposit the database.
4. ~~Check the Hammett constants against the table.~~ Done 25 October 2026 for σ_m and σ_p,
   as far as the scan allows: 39 of 95 checked, one corrected, 56 unmatched. π is unchecked and
   its attribution is corrected.
5. Item A4 of the plan adds the hashes of every file here to the manifest of the benchmark
   results, so that a changed data set is told apart from a changed program.

---

## Where the public collections are (survey of 15 October 2026)

Status column: "verified" = directory listing or file was fetched and inspected during
compilation; "reported" = location cited by a paper or index page but not opened.

### 1. Sutherland, O'Brien & Weaver 2004 benchmark (8 aligned sets)

J. Med. Chem. 2004, 47, 5541-5554, DOI 10.1021/jm0497141. The sets used in the CMF
papers (ACE, AChE, BZR, COX2, DHFR, GPB, THERM, THR). Aligned 3D conformations,
ionisation states, partial charges, designed train/test splits.

| Location | Content | Status |
|---|---|---|
| ACS Supporting Information, https://pubs.acs.org/doi/10.1021/jm0497141 | `3dqsar_mol2.tar.gz` (keeps MNDO ESP charges), `3dqsar_sd.tar.gz`, `corina_sd.tar.gz`, `25d_descriptors.tar.gz` | reported (ACS blocks scripted access; open in a browser) |
| UCI ChemDB mirror, https://ics.uci.edu/~baldig/learning/QSAR_Sutherland/ | `ace_3dqsar.sdf`, `ache_3dqsar.sdf`, `bzr_3dqsar.sdf`, `cox2_3dqsar.sdf`, `dhfr_3dqsar.sdf`, `gpb_3dqsar.sdf`, `therm_3dqsar.sdf`, `thr_3dqsar.sdf` plus `*_25d_desc.txt`, `Readme.txt`, `INFO.txt` | verified listing (again 22 October 2026) |
| Local copy (OneDrive) `...\cheminf\r\bases\suzerland\<set>_data\` | mol2 with charges, `<set>-data-sybyl-all.txt` activity/split table | local, see memory note |
| Py-CoMSIA repo, https://github.com/clhaga/pycomsia/tree/main/sdfs | ACE, AChE, THERM, THR pre-aligned SDFs (taken from Ragno's Py-CoMFA work) plus ATA and CCR5 sets | reported, folder contents not inspected |
| OpenEye Orion 3D-QSAR docs, https://docs.eyesopen.com/floe/modules/models_3dqsar/docs/source/benchmarks/results.html | Train/validation sizes of the 8 sets (ACE 76/38, COX2 188/94, ...) for checking a copy; data itself in `.oedb` only | verified |

Notes
- The SDF mirror cannot carry the mol2 charge column except as a property field; use the mol2 archive (or the local copy) when charges matter.
- Related but different: https://ics.uci.edu/~baldig/learning/Sutherland/ holds the larger 2003 J. Chem. Inf. Comput. Sci. sets (BZR 405, COX2 467, DHFR 756, ER) as `*_3d.sdf.gz`. These are 2D/2.5D QSAR sets, not aligned.

### 2. CoMFA steroids (Cramer 1988; corrected by Coats 1998)

| Location | Content | Status |
|---|---|---|
| https://github.com/exeResearch/datasets/blob/master/Coates_CoMFA/Coats_CoMFA_steroidsBenchmarkDataset.sdf | 32 records, 3D coordinates, fields `index`, `name`, `systematicName`, `ChemSpiderID`, `isomericSMILES`, `pKi_CBG`, `pKi_TeBG`, `bindingClass`; no train/test flag (classic split: 1-21 train, 22-31 test) | verified |
| Coats, E. A. "The CoMFA steroids as a benchmark dataset..." in 3D QSAR in Drug Design vol. 3 (1998), https://link.springer.com/chapter/10.1007/0-306-46858-1_13 | Corrected structures and affinities | reported |

Note: the SDF has 32 records; index 23 is "cortisol-21-acetate" and index 32 is
"cortisolacetat". Check whether 32 is a duplicate before using the set as 31 compounds.
The affinities are stored with a negative sign (e.g. aldosterone pKi_CBG = -6.279), i.e.
as log K rather than pK; flip the sign.

### 3. Congeneric series with crystal-structure-based poses (FEP benchmarks)

| Location | Content | Status |
|---|---|---|
| https://github.com/openforcefield/protein-ligand-benchmark (git-lfs) | 21 targets, 559 ligands; layout `data/<date>_<target>/02_ligands/lig_<name>/crd/lig_<name>.sdf`, protein in `01_protein/crd/protein.pdb`, affinities in `00_data/ligands.yml`. Includes the Wang 2015 "JACS set" targets. | verified; converted into `examples/openff/` |
| https://github.com/MCompChem/fep-benchmark (MIT) | Merck set: cdk8, cmet, eg5, hif2a, pfkfb3, shp2, syk, tnks2; per target `ligands.sdf` (LigPrep poses in the site) + prepared protein PDB + experimental/FEP results CSV | verified; converted into `examples/schindler/` |
| https://github.com/schrodinger/public_binding_free_energy_benchmark (MIT) | 92 congeneric series, 1343 ligands, 13 groups | verified; converted into `examples/ross/` |
| BindingDB validation sets, https://www.bindingdb.org/rwd/validation_sets/index.jsp | ~1200 congeneric series (5-400 compounds), each tied to a PDB complex; per set 2D SDF, 3D SDF (Vconf conformers, NOT aligned) and TSV; CC-BY-SA 3.0 | verified, not used |

### 4. Large non-aligned collections (need alignment first)

| Location | Content | Status |
|---|---|---|
| https://github.com/cimm-kzn/3D-MIL-QSAR (`datasets/`) | 200 ChEMBL bioactivity sets used for 2D vs 3D (conformer-ensemble) comparison | reported |
| https://github.com/ci-lab-cz/ibenchmark | Synthetic + ChEMBL sets for QSAR interpretation benchmarking | reported |
| http://www.3d-qsar.com (Ragno, Py-CoMFA) | 30 pre-aligned public data sets inside the portal; academic login required, no public download page | reported |

### 5. Not found / dead ends

- cheminformatics.org dataset archive (old mirror of the Sutherland sets): site offline, Wayback not reachable from here.
- Open3DQSAR / Open3DALIGN: no bundled example data sets found on the project pages.
- No Zenodo or figshare deposit of the Sutherland sets was found; papers that reuse them cite the ACS SI or "the archive of Zhokhova and Baskin".
