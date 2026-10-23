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

## Follow-ups

- Dispersion: fairchem ships a D3 wrapper (`fairchem.core.calculate.dftd3_calculator`,
  needs `torch-dftd`); adding it would deepen E_ads by ~0.5-1 eV for molecules of this size
  and give the bulky compounds a physisorbed minimum.
- The VASP tier: `*_work/<name>_<k>_<pose>.xyz` are the starting structures; the slab there
  should keep the bottom two layers fixed and relax the top two, as planned, since PBE has no
  reason to reconstruct Fe(110).
- Protonated forms (examples/corrosion_hcl) need the omol head for the gas-phase reference
  (charge in `atoms.info`) and a counter-ion or a charged-slab scheme; not attempted.
