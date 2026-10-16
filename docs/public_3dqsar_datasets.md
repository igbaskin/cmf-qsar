# Public congeneric data sets for 3D-QSAR benchmarking

Compiled 2026-10-15. Status column: "verified" = directory listing or file was fetched
and inspected during compilation; "reported" = location cited by a paper or index page
but not opened.

## 1. Sutherland, O'Brien & Weaver 2004 benchmark (8 aligned sets)

J. Med. Chem. 2004, 47, 5541-5554, DOI 10.1021/jm0497141. The sets used in the CMF
papers (ACE, AChE, BZR, COX2, DHFR, GPB, THERM, THR). Aligned 3D conformations,
ionisation states, partial charges, designed train/test splits.

| Location | Content | Status |
|---|---|---|
| ACS Supporting Information, https://pubs.acs.org/doi/10.1021/jm0497141 | `3dqsar_mol2.tar.gz` (keeps MNDO ESP charges), `3dqsar_sd.tar.gz`, `corina_sd.tar.gz`, `25d_descriptors.tar.gz` | reported (ACS blocks scripted access; open in a browser) |
| UCI ChemDB mirror, https://ics.uci.edu/~baldig/learning/QSAR_Sutherland/ | `ace_3dqsar.sdf`, `ache_3dqsar.sdf`, `bzr_3dqsar.sdf`, `cox2_3dqsar.sdf`, `dhfr_3dqsar.sdf`, `gpb_3dqsar.sdf`, `therm_3dqsar.sdf`, `thr_3dqsar.sdf` plus `*_25d_desc.txt`, `Readme.txt` | verified listing |
| Local copy (OneDrive) `...\cheminf\r\bases\suzerland\<set>_data\` | mol2 with charges, `<set>-data-sybyl-all.txt` activity/split table | local, see memory note |
| Py-CoMSIA repo, https://github.com/clhaga/pycomsia/tree/main/sdfs | ACE, AChE, THERM, THR pre-aligned SDFs (taken from Ragno's Py-CoMFA work) plus ATA and CCR5 sets | reported, folder contents not inspected |
| OpenEye Orion 3D-QSAR docs, https://docs.eyesopen.com/floe/modules/models_3dqsar/docs/source/benchmarks/results.html | Train/validation sizes of the 8 sets (ACE 76/38, COX2 188/94, ...) for checking a copy; data itself in `.oedb` only | verified |

Notes
- The SDF mirror cannot carry the mol2 charge column except as a property field; use the mol2 archive (or the local copy) when charges matter.
- Original data sources to cite alongside: ACE from DePriest et al. JACS 1993; GPB and THERM from Gohlke & Klebe J. Med. Chem. 2002; THR from Böhm, Stürzebecher & Klebe J. Med. Chem. 1999.
- Related but different: https://ics.uci.edu/~baldig/learning/Sutherland/ holds the larger 2003 J. Chem. Inf. Comput. Sci. sets (BZR 405, COX2 467, DHFR 756, ER) as `*_3d.sdf.gz`. These are 2D/2.5D QSAR sets, not aligned.

## 2. CoMFA steroids (Cramer 1988; corrected by Coats 1998)

| Location | Content | Status |
|---|---|---|
| https://github.com/exeResearch/datasets/blob/master/Coates_CoMFA/Coats_CoMFA_steroidsBenchmarkDataset.sdf (raw: https://raw.githubusercontent.com/exeResearch/datasets/master/Coates_CoMFA/Coats_CoMFA_steroidsBenchmarkDataset.sdf) | 32 records, 3D coordinates, fields `index`, `name`, `systematicName`, `ChemSpiderID`, `isomericSMILES`, `pKi_CBG`, `pKi_TeBG`, `bindingClass`; no train/test flag (classic split: 1-21 train, 22-31 test) | verified |
| Coats, E. A. "The CoMFA steroids as a benchmark dataset..." in 3D QSAR in Drug Design vol. 3 (1998), https://link.springer.com/chapter/10.1007/0-306-46858-1_13 | Corrected structures and affinities | reported |

Note: the SDF has 32 records; index 23 is "cortisol-21-acetate" and index 32 is
"cortisolacetat". Check whether 32 is a duplicate before using the set as 31 compounds.
The affinities are stored with a negative sign (e.g. aldosterone pKi_CBG = -6.279), i.e.
as log K rather than pK; flip the sign.

## 3. Congeneric series with crystal-structure-based poses (FEP benchmarks)

Not classic 3D-QSAR sets, but aligned ligand poses in a common binding-site frame with
measured affinities, ideal for receptor-frame CMF models.

| Location | Content | Status |
|---|---|---|
| https://github.com/openforcefield/protein-ligand-benchmark (git-lfs) | 21 targets, 559 ligands: bace (36), bace_hunt (32), bace_p2 (12), cdk2 (16), cdk8 (33), cmet (12), eg5 (28), galectin (8), hif2a (42), jnk1 (21), mcl1 (42), p38 (34), pde10 (35), pde2 (21), pfkfb3 (40), ptp1b (23), shp2 (26), syk (44), thrombin (11), tnks2 (27), tyk2 (16). Layout `data/<date>_<target>/02_ligands/lig_<name>/crd/lig_<name>.sdf`, protein in `01_protein/crd/protein.pdb`, affinities in `00_data/ligands.yml`. Includes the Wang 2015 "JACS set" targets. | verified |
| https://github.com/MCompChem/fep-benchmark (MIT) | Merck set: cdk8, cmet, eg5, hif2a, pfkfb3, shp2, syk, tnks2; per target `ligands.sdf` (LigPrep poses in the site) + prepared protein PDB + experimental/FEP results CSV | verified (cdk8 folder) |
| BindingDB validation sets, https://www.bindingdb.org/rwd/validation_sets/index.jsp | ~1200 congeneric series (5-400 compounds), each tied to a PDB complex; per set 2D SDF, 3D SDF (Vconf conformers, NOT aligned) and TSV: `/rwd/validation_sets/MCS/<PDB>-<LIG>_Validation_Affinities[_3D].sdf`; index file `BDB_Validation_Sets_v1.tsv`; CC-BY-SA 3.0 | verified |

## 4. Large non-aligned collections (need alignment first)

| Location | Content | Status |
|---|---|---|
| https://github.com/cimm-kzn/3D-MIL-QSAR (`datasets/`) | 200 ChEMBL bioactivity sets used for 2D vs 3D (conformer-ensemble) comparison | reported |
| https://github.com/ci-lab-cz/ibenchmark | Synthetic + ChEMBL sets for QSAR interpretation benchmarking | reported |
| http://www.3d-qsar.com (Ragno, Py-CoMFA) | 30 pre-aligned public data sets inside the portal; academic login required, no public download page | reported |

## 5. Not found / dead ends

- cheminformatics.org dataset archive (old mirror of the Sutherland sets): site offline, Wayback not reachable from here.
- Open3DQSAR / Open3DALIGN: no bundled example data sets found on the project pages.
- No Zenodo or figshare deposit of the Sutherland sets was found; papers that reuse them cite the ACS SI or "the archive of Zhokhova and Baskin".
