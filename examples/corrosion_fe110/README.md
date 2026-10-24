# Adsorption on Fe(110) with UMA: the screening tier

Adsorption energies and relaxed adsorbed poses of the corrosion inhibitors of a scaffold-tree
node (`examples/corrosion/*.csv`) on bcc Fe(110), computed with Meta's UMA machine-learned
potential (small model, OC20 task head) by `examples/uma_fe110_adsorption.py`.  The SDF written
here is an aligned, labelled data set for the CMF application: the molecules alone, in the
surface frame (normal along z, the top Fe layer at z = 0, the cell centre at x = y = 0), with
`E_ads_kJmol` as the activity.

    load_dataset("examples/corrosion_fe110/157_benzimidazole_uma.sdf", activity_prop="E_ads_kJmol")

The slow calibration tier (VASP, same slab, same poses as starting points) is not part of this.

## Environment (WSL, Ubuntu 24.04, created 2026-10-22)

`~/uma-env`, a virtualenv of the system Python 3.12 (`python3 -m virtualenv ~/uma-env`; the
`python3-venv` package is not installed, `virtualenv` is):

| package | version | note |
|---|---|---|
| torch | 2.13.0+cu126 | fairchem-core 2.24 pins `torch~=2.13.0`; the PyPI wheel is a CUDA 13.0 build, which the driver (566.24, CUDA 12.7) cannot run, so torch is force-installed from the cu126 index |
| fairchem-core | 2.24.0 | |
| ase | 3.29.0 | |
| rdkit | 2026.9.1 | |
| huggingface_hub | 2.2.0 | the `hf` command |

    python3 -m virtualenv ~/uma-env
    source ~/uma-env/bin/activate
    pip install fairchem-core ase rdkit huggingface_hub pandas
    pip install --force-reinstall --no-deps "torch==2.13.0+cu126" --index-url https://download.pytorch.org/whl/cu126
    pip install "torch==2.13.0+cu126" --index-url https://download.pytorch.org/whl/cu126   # its CUDA 12.6 libraries
    python -c "import torch; print(torch.cuda.is_available())"                            # True, RTX 4070 Laptop GPU

Known noise: `warp-lang` (pulled in by fairchem's `nvalchemi-toolkit-ops`) warns at import that it
wants a CUDA 13 driver; pip reports a `cuda-python` / `cuda-bindings` version conflict.  Neither
affects the UMA calculator.

The UMA weights are gated on Hugging Face under the FAIR Chemistry licence: accept the terms on
https://huggingface.co/facebook/UMA (a form, filled in once), create a read token under
Settings > Access Tokens, then, in WSL,

    source ~/uma-env/bin/activate && hf auth login

and paste the token.  After that the first run downloads `uma-s-1p2p1.pt` into
`~/.cache/huggingface`.

## Method (examples/uma_fe110_adsorption.py)

- **Model.** UMA small 1.2.1 (`uma-s-1p2p1`), task head `oc20`, through fairchem's ASE
  calculator, single GPU (`inference_settings` default).  The same head gives all three energies.
- **Slab.** bcc Fe(110), 4 layers, orthogonal 6x6 cell (16.7 x 11.8 A, 144 Fe), 15 A vacuum,
  a = 2.78 A.  The lattice constant is the oc20 head's own minimum for bcc Fe (cubic fit through
  2.66-2.86 A; the omat head puts it at 2.89 A with a magnetic jump at 2.79 A; PBE: 2.83 A,
  experiment: 2.87 A).  The cell grows (nx by 1, ny by 2) when a molecule's flat footprint plus a
  6 A margin would not fit; 2-phenylbenzimidazole gets 7x6.  Initial moments 2.2 mu_B are
  carried for the later VASP tier; the oc20 head ignores them.
- **All slab atoms fixed.**  Found during validation: with the bottom two layers fixed, every
  adsorption run gave E_ads of -14 to -21 eV.  The energy came from the slab, not the molecule:
  for the oc20 head the ideal Fe(110) surface is a saddle point, and once the adsorbate (or a
  0.05 A random shake of a clean slab) breaks the symmetry, the two free layers slide laterally
  (rms 0.3-0.6 A, Fe-Fe contacts of 2.1-2.3 A against 2.41 A in bcc) and the energy drops by
  0.25 eV per free atom, at a = 2.78 and 2.83 A alike.  bcc Fe(110) does not reconstruct in
  reality or in DFT, so this is a model artefact; the rigid ideal slab avoids it and keeps the
  three energies consistent (the clean slab relaxed from the symmetric start converges in 2 steps
  and never finds the reconstruction, which is why the reference looked fine on its own).
  `--fixed-layers 2` restores the relaxed-surface protocol for a model that behaves.
- **Molecules.** RDKit, neutral form (`rdMolStandardize.Uncharger`; permanent charges stay and
  are written as `charge`), ETKDG v3 + MMFF, re-sanitised.  Gas-phase reference: the molecule
  relaxed in a 20 A periodic box with the same head (checked: the free molecule's energy is the
  same in a 12, 20 or 30 A box, in the slab cell, and placed 12 A above the slab, to 0.1 eV).
- **Poses.** flat (plane of the two longest principal axes parallel to the surface, long axis
  along x, lowest heavy atom 2.2 A above the top layer), upright (the lone pair of the most
  basic nitrogen pointing down, tilted by the smallest angle that makes the N the lowest atom,
  N at 2.0 A; the rule: aliphatic amine > pyridine-type N > imine > aniline > pyrrole-type NH,
  never amide/nitro/quaternary; S or O when there is no N), random (seeded rotation, lowest
  heavy atom at 2.2 A).
- **Docking, then relaxation.** The oc20 head is trained on RPBE data, which has no dispersion
  and therefore no physisorption well: a pose that starts 2.5 A above the surface usually
  drifts off to 4-5 A and ends with E_ads ~ 0 (in the first rigid-slab run all three poses of
  2-methylbenzimidazole did, its methyl hydrogens holding the ring at 3.4 A).  So each pose is
  first relaxed for up to 40 steps (fmax 0.1) with the heavy atoms of the molecule confined to
  their height (ASE `FixedPlane`; hydrogens and the lateral position move), then freely: BFGS,
  fmax 0.05 eV/A, 300 steps.  The reported energy is the free local minimum.  A pose whose bonds
  stretched by more than 30 % (`!` in `poses_eV`) is never chosen while an intact one exists;
  a pose that ended unbound (all heavy atoms above 4 A, `~`) is a valid "no chemisorption"
  result, and the SDF/CSV column `bound` says whether the chosen pose is adsorbed.
- **Output.** SDF of the chosen pose per molecule in the surface frame (z = 0 at the top Fe
  layer, x = y = 0 at the cell centre), with `E_ads_eV`, `E_ads_kJmol`, `pose_label`,
  `n_steps`, `E_slab_eV`, `E_mol_eV`, `poses_eV`, `cell`, `model`, `charge` and the input CSV
  columns; a CSV summary; `<out>_work/` with every pose's final slab+molecule geometry
  (`*.xyz`, the starting points for the VASP tier), BFGS logs and per-molecule JSON records.
  A stopped run resumes from the records; `--fresh` recomputes.

## Validation (three molecules, 2026-10-22)

    python examples/uma_fe110_adsorption.py examples/corrosion/157_c1ccc2_nH_cnc2c1.csv         -o examples/corrosion_fe110/157_benzimidazole_uma.sdf         --select Benzimidazole,2-Methylbenzimidazole,2-Phenylbenzimidazole

Rigid 6x6 slab (7x6 for 2-phenylbenzimidazole), E_slab = -980.8347 eV in every run (-1144.3155 eV
for 7x6), model load 7-8 s, log in `validation_3mol.log`.

| molecule | E_ads, eV | kJ/mol | pose | height of the lowest heavy atom, A | other poses, eV | steps | time, s | GPU peak, MB |
|---|---|---|---|---|---|---|---|---|
| Benzimidazole | -1.297 | -125.1 | flat | 1.72 | upright 0.004 (floated to 3.9 A), random -0.027 (unbound) | 118 | 118 | 2827 |
| 2-Methylbenzimidazole | -0.781 | -75.4 | flat | 1.91 | upright -0.058 (unbound), random -0.093 (unbound) | 107 | 81 | 2862 |
| 2-Phenylbenzimidazole | -1.355 | -130.8 | flat | 1.84 | upright +0.211 (a bond broke), random +0.021 (unbound) | 100 | 172 | 3210 |

Sanity checks: E_ads negative for all three; benzimidazole in the -1 to -3 eV range with the
ring flat on the surface (heavy atoms 1.7-2.1 A above the top Fe layer); the clean-slab energy
is bit-identical between runs; ~2 min per molecule (three poses plus the gas-phase reference).
The N-down upright poses do not stay bound for this head: the azole N is not an anchor on
Fe(110) at the RPBE level without dispersion, the flat pi-bonded geometry is.  Values are
RPBE-like (no dispersion): a D3 correction or the VASP tier will deepen them by roughly
0.5-1 eV for molecules of this size, which matters for the absolute scale, less for the
ranking within the node.

The first attempt, with the bottom two layers fixed and the top two free, gave -14 to -21 eV
for every pose: the slab reconstruction described under "All slab atoms fixed".

## The benzimidazole node (73 molecules, 2026-10-22/23)

    python examples/uma_fe110_adsorption.py examples/corrosion/157_c1ccc2_nH_cnc2c1.csv \
        -o examples/corrosion_fe110/157_benzimidazole_uma.sdf

Output: `157_benzimidazole_uma.sdf` (the adsorbed poses, loads with
`load_dataset(..., activity_prop="E_ads_kJmol")`: 73 molecules, 15-91 atoms, 73 activities),
`157_benzimidazole_uma.csv` (the summary), `157_benzimidazole_uma.log`, and the git-ignored
`157_benzimidazole_uma_work/` with every pose's final geometry.

- **Time.** 189.5 min for 73 molecules on the RTX 4070 Laptop GPU: mean 161 s per molecule
  (65 s for the small ones, 691 s for the 91-atom bis-dodecyl compound), model load 8 s.
  The cell grew for 49 molecules (7x6 to 11x10, up to 440 Fe atoms); GPU peak 2.8 GB for the
  6x6 cell, 6.7 GB for 11x10.  The gas-phase reference plus three poses cost 100-500 BFGS
  steps per molecule.
- **Energies.** E_ads from -2.19 eV (2,6-bis(benzimidazol-2-yl)pyridine, flat) to about
  +0.2 eV; median -0.56 eV.  55 of the 73 chosen poses are chemisorbed (`bound` = 1), 18
  ended unbound with E_ads within +-0.3 eV of zero: bulky N1-substituted and amide/sulfonamide
  compounds whose aromatic rings cannot lie on the surface, for which the dispersion-free oc20
  head has no binding at all.  Flat poses win 44 times, upright 14, random 15; the upright
  winners are mostly long-chain or multi-ring compounds where the "flat" start was a poor guess
  for a non-planar molecule.
- **Dissociation.** 32 molecules had at least one pose in which a bond broke on the surface
  (`!` in `poses_eV`), typically an N-H, S-S or C-S cleavage; the chosen pose is intact for all
  73, because CMF needs the whole molecule.  Where the dissociated pose is much lower (the
  bis(benzimidazol-2-yl) disulphide: -4.8 eV dissociated against +0.36 eV intact), the intact
  E_ads understates the real affinity; `poses_eV` keeps the number.
- **One artefact.** 1-Dodecyl-2-((dodecylthio)methyl)-1H-benzimidazole (91 atoms): the oc20
  head rearranged the free molecule during the gas-phase relaxation (`gas_intact` = 0), so its
  reference is too low and E_ads = +2.52 eV is meaningless; drop it or recompute the reference
  with the omol head.
- **Against the literature columns.** Pearson r between E_ads and the electrochemical
  potency (`activity`) is 0.18, and -0.20 against the tabulated `dG_ads` (59 values): no
  simple linear relation, as expected for a descriptor that ignores solvation, protonation
  (all molecules are neutral here, but the medium is 1 M HCl) and the dispersion contribution.
  The modelling use is as a second training target and as the alignment, not as a predictor.

## CMF models: the adsorbed-pose frame against the MCS alignment (2026-10-23)

`compare_alignments.py` fits the same KRR model (five physicochemical fields, Gasteiger
charges, alpha and lambda by 10-fold inner cross-validation) in three frames and reports the
external 5-fold cross-validation, 10 repeats, identical folds everywhere; results in
`alignment_comparison.csv`.

- `mcs`: gas-phase conformers superimposed on benzimidazole by their maximum common
  substructure, in 3D (the conventional frame, as the exporter checks it).
- `pose`: the UMA poses as written: the surface normal and height are physical, the in-plane
  rotation and position are wherever the relaxation left them.
- `pose+mcs`: the UMA poses registered in the surface plane only (MCS atoms superimposed on
  the reference pose by a rotation about z and an x, y shift; median in-plane rmsd 0.95 A).

| target | frame | n | q2 (external CV, mean +- sd over repeats) | RMSE | sd of y | y-scrambled q2, max of 5 |
|---|---|---|---|---|---|---|
| potency (1 M HCl) | mcs | 73 | 0.19 +- 0.07 | 1.67 | 1.86 | -0.01 |
| potency | pose | 73 | 0.09 +- 0.05 | 1.77 | 1.86 | 0.00 |
| potency | pose+mcs | 73 | 0.16 +- 0.05 | 1.71 | 1.86 | -0.02 |
| E_ads, kJ/mol | mcs | 72 | 0.13 +- 0.06 | 49.7 | 53.2 | 0.01 |
| E_ads | pose | 72 | 0.51 +- 0.04 | 37.4 | 53.2 | -0.01 |
| E_ads | pose+mcs | 72 | 0.52 +- 0.04 | 36.9 | 53.2 | 0.04 |

On the 54 chemisorbed molecules alone (the unbound ones all sit near E_ads = 0 and their
geometry is arbitrary): potency q2 0.13 (mcs) and 0.11 (pose+mcs); E_ads q2 0.07 (mcs) and
0.26 (pose+mcs).

What this says:

- **The adsorbed frame carries the adsorption energy.** A field model in the surface frame
  recovers the UMA E_ads with q2 = 0.5 (0.26 within the chemisorbed subset, where the easy
  bound/unbound distinction is gone), while the same model in the MCS frame does not (0.13 and
  0.07).  The vertical placement and the registration against the surface are the signal; the
  in-plane MCS registration adds little (0.51 -> 0.52).  So the frame is informative, and the
  E_ads target is learnable by CMF.
- **It does not help the potency.** The electrochemical potency in 1 M HCl is weakly
  modelled in every frame (q2 0.09-0.19; scrambled models give 0, so the small values are
  real), and the MCS frame is, if anything, slightly better.  Consistent with r = 0.14
  between potency and E_ads: the neutral-molecule, dispersion-free adsorption energy on a bare
  Fe(110) terrace is not what sets inhibition in acid, where protonation, the chloride-covered
  surface, solvation and the molecule's hydrophobic footprint all enter.
- **Where to go.** Use E_ads as a second task or co-field beside the potency rather than as the
  frame of the potency model; add dispersion and the protonated forms before expecting the
  adsorption tier to rank inhibitors; and treat the 18 unbound molecules separately, since
  their poses are not adsorption geometries.

## E_ads as a second input to the potency model (2026-10-23)

`eads_as_second_target.py` (results in `eads_second_target.csv`) lets the UMA energy into the
potency model in the two ways the package offers, in both frames, same external 5x10 repeated
cross-validation; the artefact molecule is excluded (n = 72; 58 for dG_ads).

| target | frame | fields alone | + E_ads descriptor (h_d = 1 / 4) | two tasks, rho = 0 | two tasks, rho searched | E_ads alone |
|---|---|---|---|---|---|---|
| potency (raw MWILA, 1 M HCl) | mcs | 0.18 | 0.18 / 0.18 | 0.19 | 0.19 (rho 0) | -0.05 |
| potency | pose+mcs | 0.15 | 0.13 / 0.12 | 0.17 | 0.17 (rho 0) | -0.05 |
| MW-adjusted MWILA | mcs | -0.12 | -0.12 / -0.12 | -0.09 | -0.09 (rho 0) | -0.06 |
| MW-adjusted MWILA | pose+mcs | -0.03 | -0.05 / -0.06 | -0.02 | -0.02 (rho 0) | -0.06 |
| literature dG_ads | mcs | -0.02 | -0.04 / -0.04 | -0.02 | -0.02 (rho 0) | -0.07 |
| literature dG_ads | pose+mcs | 0.09 | 0.04 / 0.04 | 0.10 | 0.08 (rho 0) | -0.07 |

(q2 of the external cross-validation, mean over 10 repeats; the standard deviations are
0.04-0.09.)

- **E_ads adds nothing to the potency.**  As a descriptor block it changes q2 by at most
  +-0.03, within the repeat scatter, in both frames and at both mixing weights.  As a second
  task the inner cross-validation picks rho = 0, i.e. no sharing, in every one of the 50 outer
  folds for every target; the rho-searched and single-task numbers coincide.  Alone it
  predicts nothing (q2 < 0).
- **The alternative targets are worse, not better.**  The molecular-weight-adjusted MWILA
  removes the size trend that the fields partly see, and what is left is not predictable
  (q2 < 0 everywhere).  The papers' Langmuir dG_ads is predicted at q2 0.1 at best (pose frame)
  and not at all from the MCS frame.
- **Conclusion for the study.**  The neutral-molecule, dispersion-free adsorption energy on a
  bare Fe(110) terrace carries no information about inhibition in 1 M HCl beyond what the
  molecular fields already hold.  Before more molecules are run, the tier has to model the
  medium: the protonated inhibitor, the chloride-covered surface, and dispersion.  The frame
  result stands on its own: the UMA poses are a self-consistent, learnable geometry (q2 0.5 for
  E_ads), so the pipeline is sound; it is the physics of the target that is missing.

## What the potency can support: reproducibility and single-laboratory series (2026-10-23)

`potency_series.py` works from `node157_experiments.csv`, the per-experiment rows behind the
node (1600 rows, 73 compounds, 62 articles, exported read-only from the database by the parent
session; 1210 rows in 1 M HCl with a MWILA).  Results in `potency_noise.csv` and
`potency_series.csv`.

**Reproducibility of the target.**

| quantity | value |
|---|---|
| sd of the 73 compound medians | 1.87 |
| within-paper sd of a compound's MWILA over concentrations and methods (98 compound-paper pairs) | 1.23 |
| between-paper sd of the same compound (20 compounds in 2-16 papers) | 1.20 |
| split-half correlation of a compound's value across papers | 0.64 |
| implied ceiling q2 for a model of the medians | 0.57 |
| PDP vs EIS on the same compound, paper and concentration (190 pairs) | r 0.98, PDP lower by 0.22 |
| PDP vs weight loss (137 pairs) | r 0.95, PDP lower by 0.33 |
| slope of MWILA vs ln C within a compound-paper series (80 series) | -0.60 (IQR -0.71 to -0.39; Langmuir would give 0) |

So the pooled potency is reproducible to r 0.64 between laboratories, and a perfect model of
the medians would reach q2 about 0.6: the 0.19 of the pooled models is far below what the data
allow, i.e. the fields are missing information rather than fighting noise.  The methods agree
with each other; the concentration dependence is the systematic part: efficiencies saturate
less than Langmuir predicts, so MWILA falls by 0.6 per e-fold of concentration and a compound
measured at high concentrations gets a low median.  Correcting every compound to 1 mM with a
common slope (57 compounds with molar concentrations) did not help the models (q2 0.13 in both
frames): the concentration choice is not what separates the laboratories.

**Single-laboratory series** (the series' own medians; leave-one-out outer, 10-fold inner).

| series | compounds | MCS frame q2 | adsorbed-pose frame q2 | r(E_ads, potency) |
|---|---|---|---|---|
| Hu 2011 (weight loss, 298 K, 2-/4-/5-substituted benzimidazoles) | 18 | **0.66** (scrambled max -0.13) | 0.26 (scrambled max -0.23) | 0.15 |
| Raicheva 1993 (weight loss, 293 K) | 12 | 0.04 | 0.20 | 0.27 |
| Popova laboratory, 4 papers pooled | 13 | -0.14 | 0.28 | 0.08 |
| Dutta laboratory, 3 papers pooled (sd of y only 0.42) | 13 | -0.23 | -0.69 | -0.22 |
| Gutierrez 2016 | 6 | -0.58 | -0.48 | 0.46 |

- On a clean series measured under one protocol, CMF in the conventional MCS frame works:
  q2 0.66 on Hu 2011's 18 compounds, above the pooled node's ceiling.  The twelve-compound
  series are too small and too narrow to decide anything (q2 within +-0.3 of zero in both
  frames), and the Dutta set has almost no spread to model.
- The adsorbed-pose frame does not beat the MCS frame on the one series that can tell
  (0.26 against 0.66), and E_ads correlates with the potency at r 0.1-0.3 everywhere.
- The loss from 0.66 on one laboratory to 0.19 on the node is therefore the heterogeneity
  between laboratories (different alloys, times, concentrations and ways of reading the
  efficiency), not a limit of the fields.  Pooling needs a laboratory offset: compounds
  measured in several papers (20 here) are the only link between papers, and a mixed model
  with compound and paper effects is the next thing to try on the target side.

## A pooled potency without the laboratory offsets (2026-10-23)

`potency_mixed.py` fits, by REML on the 1210 rows in 1 M HCl,

    MWILA = mu + a_compound + b ln(C / 1 mM) + u_paper + e,   u_paper ~ N(0, s_p^2),

compound effects fixed, one random intercept per paper, one concentration slope (ppm converted
with the molecular weight; all 1210 rows have a concentration then).  `potency_mixed.csv` holds
mu + a_i per compound, its potency at 1 mM in an average laboratory.

| component | value |
|---|---|
| concentration slope b | -0.60 per ln C |
| sd of the paper offsets (random effect) | 1.05 |
| residual sd within paper and compound, after concentration | 0.77 |
| sd of the pooled potency over the 73 compounds (raw medians: 1.87) | 1.38 |
| correlation with the raw medians | 0.87 |

Only one component of the compound-paper graph is connected: 39 compounds in 27 papers that
share compounds.  The other 34 compounds sit in 20 single-paper islands (Hu 2011's 18 among
them), whose offsets the random effect shrinks to zero, so those compounds keep their own means.
The paper offsets of the linked papers span -2.4 to +2.1, i.e. the laboratories disagree by as
much as the compounds differ.

Frame comparison with the pooled potency (5-fold external CV x 10, as before):

| target | compounds | MCS frame q2 | adsorbed-pose frame q2 |
|---|---|---|---|
| raw medians | 73 | 0.19 | 0.16 |
| mixed model | 73 | 0.21 | 0.17 |
| raw medians | linked 39 | -0.08 | 0.01 |
| mixed model | linked 39 | 0.00 | -0.04 |

- Removing the laboratory offsets changes the target (r 0.87 with the medians, a third less
  spread) but not its predictability: q2 0.21 against 0.19.  The between-laboratory
  disagreement was not what hid the structure-potency relation.
- The 39 compounds measured in several laboratories are the ones no model predicts, in either
  target and either frame (q2 about 0 with a scatter of 0.15-0.2 between repeats).  They are
  the much-studied, chemically diverse members of the node (benzimidazole and its 2-amino,
  2-mercapto, 2-phenyl, bis- and N-substituted relatives), one or two per chemotype.  What the
  pooled models predict comes from the single-paper islands, which are congeneric series
  measured under one protocol, Hu 2011 above all (q2 0.66 on its own).
- So the node behaves as a set of within-series trends that do not share one potency scale,
  rather than as one series with noisy labels.  A CMF model of inhibition at this level needs
  either a consistent measurement across chemotypes (one laboratory, one protocol, matched
  concentrations, the Hu 2011 design extended) or a target that is explicitly relative within a
  paper (paper-centred potencies, which the mixed model provides as the residual structure).
  Neither the adsorbed-pose frame nor E_ads changes this conclusion; they were tested at every
  step.

## Ranking analogues within a series: leave-paper-out on centred potencies (2026-10-24)

`potency_within_paper.py` (table in `potency_within_paper.csv`): every (compound, paper) pair
in 1 M HCl from a paper with at least three of the node's compounds is one observation, the
paper-wise median MWILA minus the paper's mean (88 observations, 15 papers, 48 compounds, sd
1.31).  The laboratory offsets are gone by construction and the target is "how much better
than its paper's average is this compound".  One paper is held out at a time; lenient keeps
every other row for training, strict also removes the held-out paper's compounds from the
training rows, so the model must rank analogues it has never seen.  KRR on the five fields,
alpha and lambda by a grouped inner cross-validation.

| model | q2 of the within-paper variance | mean Spearman within the 7 papers of 5+ compounds |
|---|---|---|
| E_ads centred within the paper (one slope, no CV) | 0.05 | 0.07 |
| fields, MCS frame, lenient | 0.28 | 0.25 |
| fields, MCS frame, strict (unseen compounds) | -0.13 | 0.08 |
| fields, adsorbed-pose frame, lenient | 0.19 | 0.27 |
| fields, adsorbed-pose frame, strict | -0.12 | -0.26 |

Per paper (lenient, MCS): Raicheva 1993 rho 0.80, Popova 1996 0.83, Popova 2004 0.43, Hu 2011
0.06, Dutta 2017 -0.80; strict: all within +-0.3 of zero except Dutta 2017 (0.60, five
compounds).

- The within-series order is predicted only for compounds that appear in other papers'
  series as well: the lenient models carry a compound's relative standing from one laboratory
  to another (which is real information, and consistent with the between-paper reliability of
  0.64), but with the compounds removed nothing transfers.  Hu 2011's 18 compounds, which the
  fields model at q2 0.66 from inside the series, are ranked at rho 0.06 (lenient) and -0.17
  (strict) from the other fourteen papers: the structure-potency trend of one series does not
  carry to another.
- E_ads does not rank analogues within a series either (rho 0.07 pooled, within +-0.3 in every
  paper).  This is the most direct test the adsorption tier can be put to, and it fails it.
- Taken with the mixed-model result: the node holds series-specific trends, each learnable on
  its own, that neither share a potency scale nor share a structure-potency relation.  A
  cross-chemotype CMF model of inhibition in 1 M HCl needs measurements made to be compared,
  not a better alignment or a computed adsorption energy.

## The co-fields of the E_ads model (2026-10-24)

`cofields_eads.py` fits the E_ads model once in the adsorbed-pose frame (72 molecules, five
fields, alpha 0.8 and lambda 1 by 10-fold cross-validation: q2 0.58, RMSEcv 34.6 kJ/mol, fitted
r2 0.96) and draws h_f c_f C_f(x, y, z), the regression coefficient of each field, on the
planes z = 2.0 A (the height of a chemisorbed ring) and 3.5 A above the top Fe layer, with
benzimidazole, the reference of the frame, drawn in (`cofields_eads.png`; the ranges are in
`cofields_eads.txt`).  Blue: field density there makes E_ads more negative, i.e. stronger
adsorption; red: weaker.  The scale is kJ/mol per unit of field density, which differs between
fields, so the panels are read for shape rather than compared in magnitude.

- **Electrostatic** carries the model (range +-12 at 2.0 A, the largest by far): at 2.0 A a
  strong blue lobe at the C2 position between the two ring nitrogens and red lobes at the far
  side of the benzo ring (positions 5 and 6) and beyond C2; at 3.5 A blue over the whole ring
  system.  So positive electrostatic field at C2 and above the rings, and negative field at
  the benzo positions, strengthen adsorption on the model's own terms: an electron-poor C2
  with electron-rich benzo substituents is what the UMA energies reward.
- **Hydrogen-bond donors** (range +-7): blue around N1/C2 at both heights, the position of an
  N-H or a 2-amino group; a donor there means a nitrogen lone pair that the surface can bind.
- **Steric** is flat (+-0.3): size and shape as such do not change the energy, which agrees
  with the descriptor check (five size descriptors predict E_ads at q2 -0.12).  Hydrophobic
  (+-1.4) and acceptor (+-3) fields contribute little and without a clear pattern.

The picture is chemically sensible for a flat-lying azole on iron, and it is the kind of
output the method is for; it describes the UMA potential, not inhibition.

## Quantum-chemical fields for the E_ads model (2026-10-24)

`examples/orca_atom_properties.py` was run on the adsorbed-pose SDF (r2SCAN-3c, the N, N+1 and
N-1 states at the UMA geometries, so every property sits in the surface frame): 73 molecules,
six at a time in the serial ORCA build, 3.6 h in all, mean 12 min per molecule (2 min for
benzimidazole, 2.6 h for the 91-atom bis-dodecyl compound).  Output
`157_benzimidazole_uma_qm.sdf` with per-atom Hirshfeld, Mulliken, Loewdin and CHELPG charges,
HOMO/LUMO populations, Fukui f+, f-, f0, the dual descriptor, local softness and
electrophilicity, plus the global indices.  `quantum_fields_eads.py` rebuilds the E_ads model
with them (same molecules, registration and nested cross-validation as before; table in
`quantum_fields_eads.csv`).

| fields | q2 (external CV, 10 repeats) | RMSE, kJ/mol |
|---|---|---|
| classical five (Gasteiger electrostatics, steric, hydrophobic, hba, hbd) | 0.52 +- 0.04 | 36.9 |
| Hirshfeld charges in place of Gasteiger | 0.51 +- 0.03 | 37.2 |
| classical + f- | 0.53 +- 0.03 | 36.6 |
| classical + f- + HOMO population | 0.50 +- 0.03 | 37.5 |
| reactivity only: f-, f+, HOMO population, local softness s- | 0.53 +- 0.02 | 36.5 |
| Hirshfeld + f- + HOMO population | 0.50 +- 0.02 | 37.6 |

- **The quantum fields neither help nor hurt**: every set sits at q2 0.50-0.53, inside the
  scatter between repeats.  Four reactivity fields with no classical field at all reach the
  same 0.53, which says what the model is learning: the registration of atoms against the
  surface in this frame, carried by whichever field type marks the atoms, not the chemistry of
  the field.  That is consistent with the rigid-slab, dispersion-free UMA energies being
  largely a geometric quantity, and it sets the useful ceiling of this target at about 0.5.
- **The co-fields agree with the classical ones** (`cofields_eads_qm.png`, reactivity set;
  `cofields_eads_qm_hirshfeld.png`, Hirshfeld + f- + HOMO; blue = stronger adsorption).  The
  Hirshfeld co-field repeats the Gasteiger picture: positive charge at the benzo 5/6 positions
  and beyond C2 weakens adsorption, i.e. electron-rich substituents there strengthen it.  f-
  and the HOMO population add that donor density at the 2-substituent and on the benzo
  carbons strengthens adsorption.  In the reactivity set the local softness s- carries nearly
  everything (it is f- times the molecule's global softness, so it encodes "a soft molecule
  with donor density at the periphery"), with blue lobes at the benzo periphery and at the
  2-substituent position at 2.0 A.
- So the quantum properties confirm the reading, cost 3.6 h of ORCA, and do not change the
  number; they would matter for a target that is electronic rather than geometric, which E_ads
  from this tier is not.

## The surface kernel: no in-plane alignment (2026-10-24)

A new kernel source in the package, `kernel_source="surface"` (`cmfqsar/invariant.py`,
`SurfaceSpec`; `cmf build --kernel surface`): the two-point distribution of the field weights
over the *in-plane* distance, resolved by the heights of both atoms above a surface at z = 0,

    Phi_fg(rho, z, z') = sum_lm w_lf w_mg exp(-(rho_lm - rho)^2 / 2 s^2) exp(-(z_l - z)^2 / 2 s_z^2) exp(-(z_m - z')^2 / 2 s_z^2),

on a grid of in-plane distances (0-12 A in steps of 0.75) and heights (1-6 A in steps of 1).
It is invariant to translation in the plane and rotation about the normal and to nothing
else; averaged over the direction of rho it is, by Wiener-Khinchin, the in-plane power spectrum
of the fields at each height: reciprocal space in the plane, real space along the normal.  The
kernel is the cosine-normalised inner product of the channels, scaled per field pair, exactly
as the three-dimensional autocorrelation kernel is built.  Tests in
`tests/test_surface_invariant.py`; the fix that went with it: `nested_cv` rebuilt its fold
models from a short list of settings that dropped the kernel source, so any invariant kernel
validated through it had silently been the overlap kernel at one alpha (`CMFModel.fresh_copy`
now carries every setting, with a regression test).

`surface_kernel_eads.py` on the UMA poses exactly as written (`surface_kernel_eads.csv`):

| model | alignment | q2 (external CV, 10 repeats) | RMSE, kJ/mol |
|---|---|---|---|
| overlap fields | poses registered in the plane by MCS | 0.52 +- 0.04 | 36.9 |
| overlap fields | poses as written | 0.51 +- 0.04 | 37.4 |
| overlap fields | poses shuffled in the plane | 0.48 +- 0.04 | 38.3 |
| **surface kernel** (s 0.75, z 1-6 step 1) | none; poses as written | **0.66 +- 0.03** | 31.2 |
| surface kernel (s 0.5, z step 0.5) | none | 0.65 +- 0.02 | 31.5 |
| surface kernel (s 1.0, z 1-7 step 1.5) | none | 0.66 +- 0.03 | 31.2 |
| surface kernel | poses shuffled in the plane | 0.66 +- 0.03 (identical predictions) | 31.2 |
| 3D autocorrelation (height thrown away) | none | 0.11 +- 0.08 | 50.2 |

- With no alignment step at all the surface kernel beats the registered overlap model, 0.66
  against 0.52, and its numbers do not change when every pose is turned and shifted at random
  in the plane.  The grids hardly matter (0.65-0.66 over three settings).
- The height is the information: the same descriptor with the height axis removed (the
  three-dimensional autocorrelation, invariant to every rigid motion) falls to 0.11.  Which
  field is at which height above the surface, and how far apart in the plane, is what the UMA
  energies depend on; where the molecule sits over the lattice is not.
- This answers the alignment question for the adsorption tier in the cleanest way available:
  the surface frame needs only the normal, and the in-plane registration that MCS provided
  was never doing anything.  It also raises the ceiling of the E_ads target from about 0.5 to
  about 0.65, which is the number to quote for "can CMF learn an ML adsorption energy".
- What is lost is the co-field: a model on this kernel has no field in a fixed frame to draw.
  The decomposition that survives is per channel (which pair of fields, at which in-plane
  distance and heights), as for the three-dimensional kernel.  The lattice-registry variant
  (projection on the Fe(110) reciprocal lattice) was not built; the result above says the
  registry carries little for this target.

## Baselines on global descriptors (2026-10-24)

`descriptor_baselines.py` (table in `descriptor_baselines.csv`): the QSPR the corrosion
literature builds, on the same molecules and folds as the field models.  Descriptors: the
eleven global quantum indices of the ORCA run (E_HOMO, E_LUMO, gap, dipole, IP, EA, hardness,
softness, electrophilicity, chemical potential, dN towards iron) and 21 RDKit descriptors (size,
logP, polar surface, counts, topological indices).  Learners: ridge, kernel ridge (RBF), random
forest; the first two tuned in an inner cross-validation.  5-fold external CV x 10 repeats on
the node, leave-one-out on Hu 2011.

| target | descriptors | ridge | kernel ridge | random forest | best field model |
|---|---|---|---|---|---|
| E_ads (72) | quantum (11) | 0.33 | 0.35 | 0.31 | surface kernel 0.66; overlap 0.52 |
| E_ads | RDKit (21) | -0.07 | -0.04 | 0.25 | |
| E_ads | both (32) | 0.38 | 0.39 | 0.37 | |
| potency, pooled (73) | quantum | -0.54 | -0.33 | 0.06 | overlap, MCS frame 0.19 |
| potency, pooled | RDKit | 0.10 | 0.25 | **0.38** | |
| potency, pooled | both | 0.29 | -0.25 | 0.34 | |
| potency, Hu 2011 (18) | quantum | **0.75** | 0.74 | 0.53 | overlap, MCS frame 0.66 |
| potency, Hu 2011 | RDKit | 0.47 | 0.75 | 0.34 | |
| potency, Hu 2011 | both | **0.80** | 0.63 | 0.61 | |

Single indices: for E_ads the gap (r 0.61); for the pooled potency ln MW (0.60), logP (0.54)
and IP (-0.48); for Hu 2011 the ionisation potential (r 0.89).

- **E_ads: the fields win, but not for the reason given above.**  The global quantum indices
  alone explain a third of the energy (q2 0.35), the gap being the carrier: a small gap means
  a soft molecule that chemisorbs.  The fields explain more (0.52 in the aligned frame, 0.66
  with the surface kernel), so the geometry over the surface adds to the electronic structure
  rather than replacing it; the earlier reading "the model learns only the registration" was
  too strong.  One caveat cuts the other way: the indices were computed at the *adsorbed*
  geometries, and a chemisorbed ring has a smaller gap than a free one (mean gap 2.8 eV for
  the bound molecules, 3.8 eV for the unbound), so part of r = 0.61 is the adsorption showing
  in the descriptor rather than predicting it (within the bound set r = 0.48).  A fair
  baseline needs the indices at the gas-phase geometries: a neutral-state ORCA run of about an
  hour.
- **Pooled potency: simple descriptors beat the fields.**  A random forest on RDKit
  descriptors reaches q2 0.38, and ln MW with logP alone, as a quadratic, 0.28: the pooled
  potency has a size-and-lipophilicity trend (r 0.60 with ln MW, which the scaffold tree's
  "MW-adjusted MWILA" was made to remove) that the field models at 0.19 capture only partly.
  The quantum indices alone predict nothing pooled.  So on heterogeneous literature data the
  honest ranking is: coarse physical descriptors with a non-linear learner first, fields
  second, frontier orbitals last.
- **Hu 2011: the frontier orbitals win.**  On the one clean series the ionisation potential
  alone correlates at 0.89 with the potency and ridge on the quantum indices gives q2 0.75
  (0.80 with the RDKit descriptors added), against 0.66 for the fields.  That is the
  corrosion literature's claim, reproduced where it should hold: within one protocol and one
  chemotype, a substituent's effect on E_HOMO tracks its effect on inhibition.  Eighteen
  molecules and eleven descriptors in a leave-one-out loop leave room for optimism even with
  the inner tuning, so the margin over the fields is not to be over-read.
- **What this does to the conclusions.**  The alignment results stand (the surface kernel is
  the best E_ads model by a margin over every descriptor model).  The claim that the fields
  are the right tool for the potency does not: on the pooled node a forest on size and logP
  does better, and on the clean series the global quantum indices do better.  A paper has to
  show these columns beside the field ones.

## The same baselines with gas-phase indices (2026-10-24)

The quantum indices above were computed at the adsorbed geometries.  `157_benzimidazole_gas_qm.sdf`
holds a neutral-state r2SCAN-3c run on the script's own ETKDG+MMFF conformers (73 molecules,
45 min, no N+-1 states), so its reactivity indices are Koopmans ones, IP = -E_HOMO and
EA = -E_LUMO, as the corrosion literature computes them; for a like-for-like pair the adsorbed
set was rerun in the same form (`--koopmans`).  Tables: `descriptor_baselines_gas.csv`,
`descriptor_baselines_koopmans.csv`.

| target | descriptors | learner | adsorbed, Delta-SCF | adsorbed, Koopmans | gas phase, Koopmans |
|---|---|---|---|---|---|
| E_ads | quantum (11) | ridge / kernel ridge / forest | 0.33 / 0.35 / 0.31 | 0.28 / 0.31 / 0.23 | -0.03 / -0.09 / -0.08 |
| E_ads | both (32) | ridge / kernel ridge / forest | 0.38 / 0.39 / 0.37 | 0.39 / 0.33 / 0.29 | -0.04 / 0.04 / 0.21 |
| potency, pooled | quantum | ridge / kernel ridge / forest | -0.54 / -0.33 / 0.06 | 0.04 / 0.00 / -0.11 | 0.03 / -0.13 / 0.24 |
| potency, pooled | both | ridge / kernel ridge / forest | 0.29 / -0.25 / 0.34 | 0.27 / 0.32 / 0.37 | 0.35 / 0.30 / 0.40 |
| potency, Hu 2011 | quantum | ridge / kernel ridge / forest | 0.75 / 0.74 / 0.53 | 0.22 / 0.66 / 0.56 | 0.65 / 0.58 / 0.59 |
| potency, Hu 2011 | both | ridge / kernel ridge / forest | 0.80 / 0.63 / 0.61 | 0.80 / 0.66 / 0.62 | 0.80 / 0.74 / 0.61 |

Strongest single index at the gas-phase geometry: for E_ads the hardness at r 0.24 (0.61 at
the adsorbed geometry); for the pooled potency dN at 0.40; for Hu 2011 dN at 0.90.

- **E_ads: the caveat was the whole effect.**  At the free-molecule geometry the global quantum
  indices predict nothing of the adsorption energy (q2 about 0 by every learner); the 0.35 at
  the adsorbed geometry was the chemisorbed ring's smaller gap reporting that it had
  chemisorbed.  So the fields' margin is larger than the previous section said, and the reading
  that the E_ads model learns the arrangement over the surface rather than the molecule's
  electronic structure is restored: no global index of the free molecule anticipates the UMA
  energy, the surface kernel reaches 0.66.
- **Hu 2011: the frontier-orbital model stands in gas phase.**  Ridge on the eleven Koopmans
  indices gives 0.65 (0.58-0.59 for the other learners), 0.80 with the RDKit descriptors added,
  and dN, the fraction of electrons transferred to iron, alone correlates at 0.90 with the
  potency of the series.  That is the literature's claim, and it holds where the literature
  tests it, within one series under one protocol; the field model's 0.66 is on a par with the
  quantum indices alone and below the combined descriptors.
- **Pooled potency: unchanged.**  The quantum indices alone predict nothing pooled in any form;
  with the RDKit descriptors the forest stays at 0.37-0.40, the size-and-lipophilicity trend.

## Follow-ups

- Dispersion: fairchem ships a D3 wrapper (`fairchem.core.calculate.dftd3_calculator`,
  needs `torch-dftd`); adding it would deepen E_ads by ~0.5-1 eV for molecules of this size
  and give the bulky compounds a physisorbed minimum.
- The VASP tier: `*_work/<name>_<k>_<pose>.xyz` are the starting structures; the slab there
  should keep the bottom two layers fixed and relax the top two, as planned, since PBE has no
  reason to reconstruct Fe(110).
- Protonated forms (examples/corrosion_hcl) need the omol head for the gas-phase reference
  (charge in `atoms.info`) and a counter-ion or a charged-slab scheme; not attempted.
- On the target side: a mixed model of MWILA with compound and paper effects, fitted on the
  per-experiment rows, to pool the laboratories without their offsets; and single-laboratory
  series as the test beds for the alignment question (Hu 2011 is the one that works).
