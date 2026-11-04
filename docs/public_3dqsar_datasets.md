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
| `md/` (9 files, 5.4 MB) | conformational ensembles of the 25 mcl1 ligands (entry 4HW3), the 29 p38 ligands (entry 3FLY) and the 36 hif2a ligands (entry 5TBM) from 2 ns simulations of each in its own site, each conformation weighted by the share of frames in its cluster; the `_ownframe` file is the same frames without a common reference, for the control, and `<set>_water.json` the hydration sites of every ligand for plan item D9 | ligands and activities from `openff/`; receptors 4HW3, 3FLY and 5TBM from RCSB; the simulations this work | derived (MIT ligands, CC0 receptor) | `make_md_trajectories.py` (needs OpenMM and the OpenFF stack) | `run_md.py` (D8), `run_solvent.py` (D9) |
| `pockets/` (73 files, 2.9 MB) | binding sites trimmed from the crystal structures the FEP sets name, in their frame | RCSB PDB entries, provenance in each file's `REMARK 400` | free (CC0) | `make_pocket_examples.py` | GUI pocket display, complementarity, docking |
| `decoys/` (8 files, 3.5 MB) | posed background sets for ACE, AChE, GPB, THR, plain and property-matched | actives from `sutherland/`, decoys from the other example series | inherits `sutherland/` | `make_decoy_sets.py`, `_matched_decoys.py` | screening and one-class measurements of the README |
| `litpcba/` (5 files, 15 MB) | 152 measured actives and 2999 measured inactives for two LIT-PCBA targets (ESR1 antagonists, p53), embedded and posed here; the `_decoys_similar` file is the same p53 inactives posed in the way most favourable to them (1500 of them), for the control | SMILES and the active/inactive verdicts from LIT-PCBA AVE-unbiased (Tran-Nguyen, Jacquemard & Rognan 2020), itself from PubChem dose-response assays; conformations and poses this work | derived | `make_litpcba_examples.py` | `run_screening.py` |
| `thrombin_atom_properties.sdf` | 40 THR molecules carrying per-atom Gasteiger and Crippen properties | `sutherland/THR.sdf` | inherits `sutherland/` | `make_property_example.py` | manual (`prop:` fields) |
| `am-eu/` (MOL2 + table) | 46 Am/Eu extractants of the 2015 paper with lg SF, superimposed, heavy atoms | the authors' own data (Sitnikov et al. 2015); the author of this program is one of them | free | none (as prepared for that paper) | `run_am_eu.py` |
| `selectivity/` (3 SDF + 2 PDB, 1.1 MB) | 57 AMPA / glycine-NMDA antagonists of the 2003 selectivity-fields paper with both pKi values, their difference and the published split; superimposed on the tricyclic core, plus the docked poses and the two receptor models | the authors' own Sybyl models (I. Tikhonova; the author of this program is the first author of that paper); activities from Catarzi et al. 1999, 2001 and Varano et al. 2002 as tabulated in the paper | free (the authors' files; the measurements are published) | `make_selectivity_examples.py` from the authors' MOL2 files | `run_selectivity.py` |
| `corrosion/`, `corrosion_hcl/` (23 CSV each, 997 rows each) | congeneric series of iron corrosion inhibitors with the MWILA potency, per medium, and the Langmuir ΔG; the `_hcl` folder protonated at pH 0 | the author's own database of literature measurements, exported from its scaffold tree | free (the author's compilation); the underlying values are published, see below | `scaffold_tree_export.py` | `run_corrosion.py`, `run_corrosion_qm.py`, `run_multitask.py`, `run_cdft_kernels.py` |
| `cyclodextrin/` (7 tables, 2671 guests, 0.7 MB; `ph7/` the same protonated, 0.8 MB; `upstream/` 4.1 MB) | association constants with a fixed cyclodextrin host, one table per host, activity = log K; the host-guest case, with no frame | Tahıl et al., Chemical Data Collections 45 (2023) 101022, Zenodo 10.5281/zenodo.7575539 | free (ODbL 1.0, share-alike; licence file in `upstream/`) | `make_cyclodextrin_examples.py --fetch --protonate` | `run_cyclodextrin.py`, `run_tasks.py` (the seven hosts as seven endpoints) |

| `local/shifts_cdcl3.sdf` (1.6 MB) | 400 molecules with 5096 assigned 13C shifts in ppm, on the carbon each belongs to; conformations generated here | nmrshiftdb2 (`nmrshiftdb2withsignals.sd`), filtered to the chloroform spectra | unclear | `make_local_examples.py --shifts` | `run_local.py` |
| `local/carboxylic_pka.sdf` (2.5 MB) | 912 carboxylic acids with the acidity of the carboxyl group on its carbon; conformations generated here | Baltruschat & Czodrowski 2020, `AvLiLuMoVe.sdf` and `combined_training_datasets_unique.sdf` | free (CC BY 4.0, attribution; the terms and the changes are in `local/README.md`) | `make_local_examples.py --pka-dir` | `run_local.py` |
| `ether/` (1 table and 1 SDF, 4.6 MB; `upstream/` 2.0 MB, not kept here) | 1,085 saturated acyclic ethers with the binding energy of Li(+), Na(+) and Mg(2+) at two levels of theory, 1,050 of them in the relaxed geometry of the lithium complex with the ion position kept; the cation-probe case (plan item EC7) | Wang, Yang, Harris & Gomez-Bombarelli, Chem. Commun. 56 (2020) 8920; data at github.com/learningmatter-mit/MLElectrolytes | unclear (the distributor states no licence; the SHA-256 of both files is in the script, and `upstream/` is fetched rather than redistributed) | `make_ether_examples.py --fetch` | `run_ether.py` |
| `electrolyte/` (1 table, 62 structure files and 1 JSON, 4.5 MB; `upstream/` 2.8 MB, not kept here) | 62 LiPF6 formulations with their measured conductivity and activation energy at 298.15 K, and 16 representative solvation shells of each cut from a 1 ns bulk simulation; the sampled-shell case (plan item BT5) | conductivities from CALiSol-23 (de Blasio, Elsborg, Vegge, Flores & Bhowmik, Sci. Data 11 (2024) 1-11; data at doi.org/10.11583/DTU.24559960, CC BY 4.0); the simulations and the shells this work | free (CC BY 4.0 asks for the citation, which this row and `examples/electrolyte/README.md` carry; `upstream/` is fetched rather than redistributed, and its SHA-256 is in the script) | `make_electrolyte_shells.py --fetch --select --simulate` (needs OpenMM, packmol and AmberTools) | `run_shells.py` |
| `hops/` (3 extended-XYZ files and 2 tables, 0.1 MB) | the three hosts of the Li interface study (beta-Li3PO4, Li3BP2O8, alpha-LiBO2) as DFT-relaxed supercells with net Bader charges per atom, the bond-valence barriers of their 1040 Li hops and two DFT elastic-band barriers | the author's own VASP runs (October 2026), read by `cmfqsar.vasp` | free | none (written by `cmf hops ... --out`) | `run_hops.py` |
| `benchmarks/results/hops/prepared_mp.json` (not under `examples/`) | ternary compounds of lithium and oxygen (Li, O and one more element) of the Materials Project within 0.05 eV/atom of the hull, with every Li hop of 1.5 to 4.5 A and the bond-valence barrier of up to 12 distinct hops per structure | Materials Project through its OPTIMADE interface (no key), structures and energies above the hull | free (CC BY 4.0; the structures carry their `mp-` identifiers) | `benchmarks/prepare_hops.py` | `run_hops.py` |
| `hops/disordered/` (13 CIF files and 1 table, 0.2 MB) | thirteen solid-electrolyte structure types (garnets, perovskites, LGPS, NASICON, LISICON, thio-phosphate, lithium nitride, Li3PO4) with the partial occupancies their refinements report, and the bond-valence barriers of their Li hops | Crystallography Open Database, one entry per file, named by its COD number | free (CC0) | none (the CIFs as downloaded; the table by `cmf hops ... --out`) | `run_hops.py` |
Sizes are of the tracked files. The whole of `examples/` is 73 MB, of which the three items of
unclear status and the two that inherit it are 12 MB, and `mittal/`, derived, is 15 MB.

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
(1991) 165, Table I). They were first written down from memory of that table. **All 95 have now
been checked against the paper, on 31 October 2026, and 92 agree to the last digit.** Each
substituent was found in the printed table by reading the page images of the scan, because its
character recognition distorts the substituent names (Cl comes out as "c1", SO2Me as "SOIMe") and
loses most of them on the last four of the eight table pages, 79 names of 234 rows surviving
there against 263 of 297 on the first four; the two values of every row were then
read twice, once from the image and once from the text layer, which decides the column of a
number by its position on the page. Three disagreed and were corrected in the script and in all
six SDF files:

| substituent | row of Table I | written here | Table I |
|---|---|---|---|
| CH2OCH3 | 223 | σ_m 0.02 / σ_p 0.03 | **0.08 / 0.01** |
| i-C3H7 | 288 | σ_m −0.07 | **−0.04** |
| N(CF3)2 | 169 | σ_m 0.47 | **0.40** |

The review puts brackets round a value it suspects of being inaccurate. One value used here is
bracketed in the paper: the σ_p of O-i-C3H7 (row 293, −0.45).

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
convert the affinities to a p-scale and leave the coordinates unchanged; every record keeps the
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
its status; both are regenerated by their scripts in about one minute.

### LIT-PCBA: inactives that were measured

The only background in this repository whose inactivity is an experimental result rather than an
assumption. LIT-PCBA is assembled from high-confidence PubChem dose-response assays -- 15 targets,
7761 actives and 382674 inactives -- and the AVE-unbiased subsets used here are the form its
authors recommend, the asymmetric validation embedding of Wallach & Heifets having removed the
analogue bias that lets a screen score well merely by recognising that actives resemble actives.

Two of the fifteen targets are kept, the ones whose inactive pools are small enough to pose with
a meaningful sample rather than a small symbolic one: **ESR1 antagonists** (88 actives, 3820
inactives in the pool) and **p53** (64, 3345). A seeded sample of 1500 inactives per target is
taken.

What the script adds is what LIT-PCBA does not provide and a field method requires:

- 3D conformations, from ETKDG followed by MMFF;
- a pose. The actives **and** the inactives are placed by one rule -- maximum CMF field similarity
  to the target's crystal ligand, which the archive includes -- so the frame is the receptor's and
  the placement uses no information about which class a molecule belongs to.

`--pose-against actives` instead poses the inactives to maximise their similarity to the posed
actives, the way `make_decoy_sets.py` does for `decoys/`. That is the placement most favourable to
the inactives, and `TP53_decoys_similar.sdf` keeps it so that the two can be compared: a molecule
placed where it most resembles the actives is in field space inside their description by
construction, and the difference between the two files is a measurement of how much a screening
number depends on its posing. See `benchmarks/run_screening.py` and the manual.

The 57 MB archive (`AVE_unbiased.tgz`) is **not** redistributed here -- `examples/litpcba/*.tgz` is
in `.gitignore` -- and the maker script downloads it when needed from
https://drugdesign.unistra.fr/LIT-PCBA/Files/AVE_unbiased.tgz. Every molecule written
carries a `source` tag naming the paper and the script. The derived structures are offered on the
same terms as the upstream collection, which is free for academic use.

### The Am/Eu extractants

`examples/am-eu/AmEu1-all.mol2` and `AmEu1-all.txt` are the 46 superimposed extractants and
their lg SF(Am/Eu) of Sitnikov, Zhokhova, Ustynyuk, Varnek and Baskin (J. Comput.-Aided Mol.
Des. 29 (2015) 233), in the form the authors prepared for that paper (Sybyl types, Gasteiger
charges, no hydrogens, aromatic typing). The author of this program is an author of that paper
and the files are the authors' own; the separation factors were compiled from the literature
the 2015 paper cites. `run_am_eu.py` rebuilds the hydrogens for the
quantum stage and writes `benchmarks/results/am_eu/am_eu_qm.sdf`.

### The AMPA / glycine-NMDA selectivity set

`examples/selectivity/` holds the data set of Baskin, Tikhonova, Palyulin and Zefirov, J. Med.
Chem. 46 (2003) 4063, the paper that introduced selectivity fields: 57 antagonists of the AMPA
and the glycine/NMDA receptors with both inhibition constants (Tables 1 and 2 of the paper, from
Catarzi et al., J. Med. Chem. 42 (1999) 2478 and 44 (2001) 3157, and Varano et al., J. Med.
Chem. 45 (2002) 1035). The three-dimensional structures are the Sybyl models built by
I. Tikhonova for that work (Gasteiger-Hückel charges), which the author of this program holds
as the docked poses in the two receptor models; `make_selectivity_examples.py` superimposes
them on the tricyclic core by the rule of Figure 1 of the paper and writes the aligned file,
the two docked files and the two receptor models (the AMPA ligand-binding core after PDB 1FTL
and a homology model of the glycine site). The files are the authors' own. The script's
docstring records where the printed Table 1 disagrees with the modelled structures, that the
published training set contains two compounds twice, that the file charges of sixteen compounds
do not sum to the formal charge, and that two hydrogens missing from one file were added.

### The corrosion-inhibitor series

The 23 series in `examples/corrosion/` and their protonated forms in `examples/corrosion_hcl/`
are exported by `examples/scaffold_tree_export.py` from the scaffold tree of the author's
database of corrosion inhibitors of iron-based alloys, a compilation of published measurements:
the MWILA potency ln[θ/(C(1−θ))] per compound, pooled and per medium, and the Langmuir adsorption
free energy. The tree itself is a private page and the database is not public, so the CSV files
are the only form in which these series exist outside the author's own files, and redistributing
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
hosts that fell below the twenty-guest threshold visible. The selection is one range of
conditions, 298 ± 2 K and pH 7 ± 0.5, and the median of log₁₀ K per guest inside it, with the count
and the spread of the pooled measurements in their own columns.

The licence is ODbL 1.0, which permits the copy kept here and asks for attribution, that the
licence accompany the data, and that a publicly used derived database be offered under the same
terms. The per-host tables are such a derived database and `examples/cyclodextrin/README.md` offers
them under ODbL 1.0 accordingly. This is the first set in the repository with a share-alike
condition, and anything built from these tables and then published inherits it.

Three things about these files are approximations rather than data, and the README says so at
length: the guest SMILES at the top level are the neutral forms as deposited in PubChem, so an acid
measured at pH 7 is written protonated; repeated measurements are reduced to their median,
excellent agreement in general but 3.2 log units apart for one α-CD guest; and 1:1 stoichiometry is
assumed throughout.

The first of those is corrected in the same way that `examples/corrosion_hcl/` uses for 1 M HCl.
`--protonate` writes `cyclodextrin/ph7/`, the same seven tables with the same activities and the
structures in their state at pH 7 from Dimorphite-DL, the neutral form kept in `smiles_neutral`
and the net charge in `charge`. 882 of the 2671 guests change: 793 anions, 342 cations, 1536
unchanged. There is deliberately no limit on added protons, unlike the corrosion export, because
the reason for that limit is specific to pH 0, where the rules charge every ring nitrogen although
the first protonation makes the next site far less basic; at pH 7 only sites with a rule pKa above
7 are charged. It is worth having both copies: the neutral tables are what the published models of
this endpoint were built on and so the comparable ones, while the protonated tables are the
physically correct structures. Which predicts better is not settled: on a single split of the three
natural hosts the protonated copies were better on all three, but `run_cyclodextrin.py`, over all
seven hosts with a grid and ten-fold cross-validation, has the invariant kernel improving on four
and worsening on three, the two largest sets among those that improve.

They are here for the alignment question. The host is fixed and the guests share no scaffold, so no
superposition exists to be curated, which is the regime where `run_alignment.py` finds the
rotation-invariant autocorrelation kernel of `cmfqsar/invariant.py` better than the aligned one. The
published field-based models of this endpoint are the reference to surpass: CoMFA with COSMOsar3D
fields on 60 α-CD guests, GRIND on 126 and on 233 β-CD guests, all an order of magnitude smaller
than the tables here. `benchmarks/run_cyclodextrin.py` measures the aligned, invariant and count
kernels on every host, in both variants, with the grids of `run_alignment.py` so that the two
tables can be read beside each other.

### The two per-atom sets

`examples/local/` holds the only two sets in the repository whose measured value belongs to an
**atom** rather than to a molecule, which is what plan items LP1 and LP2 are about:
`shifts_cdcl3.sdf` carries 5096 measured 13C shifts on the carbons they were measured for, and
`carboxylic_pka.sdf` carries 912 carboxyl acidities on the carboxyl carbon.
`examples/local/README.md` is the full record: the two downloads, every filter, what was changed
and the licence of each. In short:

* The shifts come from **nmrshiftdb2**, the open NMR database of the University of Cologne, from
  its distributed dump of 58 187 records. Only the 13C spectra measured in deuterated chloroform
  are kept, because a shift depends on the solvent and most records do not report one. The project
  states that its spectra may be used without restrictions and names an "nmrshiftdb2 Data License"
  for the database; that licence page could not be retrieved on 27 October 2026, the server
  refusing the connection, so the status here is **unclear** in the sense above: redistribution is
  invited by the project's own statements but an explicit permission has not been read. The file is
  a derived subset of 400 molecules, not the database, and replacing it costs one download and one
  run of the script.
* The acidities come from the collection of **Baltruschat and Czodrowski** (F1000Research 9:113,
  2020), which is **CC BY 4.0**, so the copy here is free to redistribute with attribution and a
  statement of the changes; both are in `local/README.md`. Only molecules with exactly one
  carboxylic acid group and a value below 6.5 are kept, which is what makes the site unambiguous.

Both sets had to be given three-dimensional coordinates here, with ETKDG and MMFF94, because both
sources distribute two-dimensional structures. The atom order is left untouched, so a measured atom
keeps the index its source assigns it. That is the weakest part of both files: a shift responds to
the geometry it was measured in, and one force-field conformation is a poor stand-in for it.

### The hop sets

Plan item BT7 makes a migration path the sample, and two sets carry its labels.
`examples/hops/` holds the three hosts of the Li interface study as DFT-relaxed supercells with
their Bader charges, together with the bond-valence barriers of every Li hop that `cmf hops`
wrote and the two DFT elastic-band barriers the study computed; `examples/hops/README.md` is
the record. `benchmarks/results/hops/prepared_mp.json` holds the public set:
the ternary compounds of lithium and oxygen (Li, O and one more element) of the **Materials Project** within 0.05 eV/atom of the convex hull,
fetched through its OPTIMADE interface, which needs no key, with every symmetry-distinct Li hop
and its bond-valence site-energy barrier computed here with `bvlain`. The Materials Project
distributes its data under **CC BY 4.0**, so the prepared file is free to redistribute with
attribution, which the `mp-` identifier on every structure gives; `benchmarks/prepare_hops.py`
rebuilds it from the network in about an hour. `examples/hops/disordered/` holds thirteen
conductors of the **Crystallography Open Database** with their partial occupancies (plan item
M2); COD is public domain (CC0), and `examples/hops/README.md` lists the entries and what is
disordered in each.

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
* The LIT-PCBA AVE-unbiased archive (`AVE_unbiased.tgz`, 57 MB), which
  `make_litpcba_examples.py` downloads when needed from
  https://drugdesign.unistra.fr/LIT-PCBA/Files/AVE_unbiased.tgz; `examples/litpcba/*.tgz` is in
  `.gitignore`. The posed structures derived from it are kept.
* ORCA outputs of the quantum stages (ignored by git); the SDF files that carry their per-atom
  results are kept.
* The nmrshiftdb2 dump (`nmrshiftdb2withsignals.sd`, 159 MB, the free download named in
  `examples/make_local_examples.py`) and the larger shift sets built from it. The 400
  molecules in `examples/local/` are kept; plan item LP5 needs a set of thousands, which
  `make_local_examples.py --max-shifts 6000 --any-solvent --out DIR` writes outside the
  repository and `benchmarks/run_local_scale.py --data-root DIR` reads. That set is 6000
  molecules and 58 705 carbons, 19 MB, and its spectra are not restricted to one solvent,
  so it is noisier than the kept one by a few ppm.

## Before the repository is public

1. **Decide the four items of unclear status**: `sutherland/`, `steroids_cbg.sdf`, and the
   two derived from the first, `decoys/` and `thrombin_atom_properties.sdf`. The choices are to
   ask ACS and the Erlangen group for permission, to replace the copies by the scripts and a
   note that says where to download the originals (`run_topomer.py` and `run_alignment.py`
   would then need `make_sutherland_examples.py --data-root` to have been run first, and the
   decoys and the property example would be regenerated by their scripts), or to keep them
   because the files have circulated for twenty years and the UCI mirror serves them
   openly. The plan's rule says the second; the decision is the author's. **Decided 22 October
   2026: the copies stay while the repository is private; the choice is made again before it
   goes public.**
   **Decided again on 2 November 2026, for the public release: the copies stay.** The reason is
   written here so that a reader can judge it. The three distributors state no licence and none
   of them forbids redistribution; the files are the supporting information of papers published
   between 1988 and 2004, they have been passed between research groups for twenty years, and the
   UCI Machine Learning Repository serves the Sutherland sets openly today. Removing the copies
   would cost the reader rather than the author: the tables of `run_topomer.py`,
   `run_alignment.py` and `docs/benchmark.md` could then only be reproduced after a manual
   download, and 12 MB of the examples would leave the manifest, so a changed file could no
   longer be told from a changed program. Every one of the four items keeps its row in the table
   above, which names the origin, the paper the facts belong to and the script that rebuilds the
   copy from the original distribution. If a distributor asks for a copy to be removed, that
   script is what stays, and the row says where the original is downloaded from. The status word
   in the table stays **unclear**, because it describes the licence and not this decision.
2. `mittal/` is derived and can stay, with the sentence above about what in it is the paper's.
3. Add the per-compound references to the corrosion exports, or deposit the database.
4. ~~Check the Hammett constants against the table.~~ Done for σ_m and σ_p: 39 of 95 on
   25 October 2026, the remaining 56 on 31 October 2026 by reading the page images, so all 95
   are now checked and three are corrected (CH2OCH3, i-C3H7, N(CF3)2). π is not in that paper,
   remains unchecked, and its attribution is corrected.
5. Item A4 of the plan adds the hashes of every file here to the manifest of the benchmark
   results, so that a changed data set is distinguished from a changed program.

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
as log K rather than pK; reverse the sign.

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

### 5. Not found / searches without result

- cheminformatics.org dataset archive (old mirror of the Sutherland sets): site offline, Wayback not reachable from here.
- Open3DQSAR / Open3DALIGN: no bundled example data sets found on the project pages.
- No Zenodo or figshare deposit of the Sutherland sets was found; papers that reuse them cite the ACS SI or "the archive of Zhokhova and Baskin".
