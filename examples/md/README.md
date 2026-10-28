# Short ligand-in-pocket simulations

Conformational ensembles of free-energy-set ligands, taken from molecular-dynamics simulations
of each ligand inside its own crystal site. Made for plan item D8, which asks whether such an
ensemble models a series better than the single crystal pose it started from;
`benchmarks/run_md.py` is the measurement and `docs/development_plan.md` holds the answer.

## What each file is

One SDF per set, named `<set>_md.sdf`. It holds one record per kept conformation, every record
under the ligand's own title, which is how a multi-conformer file is read back:

```bash
cmf build examples/md/mcl1_md.sdf --activity pAffinity --ensemble-by name \
    --conformations ensemble --conformer-weighting given
```

Each molecule carries `conformer_weights`, the share of simulation frames that fell in that
conformation's cluster, and the activity of the ligand as the source collection gives it.

## How they were made

Every ligand was simulated for **2 ns** at 300 K and 1 bar, after 200 ps of equilibration with the alpha carbons restrained at 5 kcal/mol/A^2. The force fields are gaff-2.11 for the ligand with AM1-BCC charges, amber14-all.xml for the protein and TIP3P water with 0.15 M of ions, in a box with 10 A of padding; the time step is 4 fs with hydrogen mass repartitioning and a frame was written every 10 ps. The frames were then superimposed on the binding-site backbone and thinned to at most 20 conformations at a clustering radius of 0.75 A. Where the crystal entry holds several copies of the protein, only the chain lining the site the ligands occupy was kept.

`examples/make_md_trajectories.py` is the script, and it is the whole recipe: nothing was done
by hand. The trajectories themselves are not in the repository -- they come to 1.3 GB -- but the script
rebuilds them from the data here and the entries named below.

## Where the data comes from

* The ligands and their activities: the Open Force Field protein-ligand-benchmark, free under
  MIT, already in `examples/openff/` with its licence; see `docs/public_3dqsar_datasets.md`.
* The receptors: the PDB entries named below, downloaded from RCSB. The wwPDB distributes its
  holdings in the public domain under CC0, so they carry no restrictions.

* `mcl1_md.sdf`: 25 ligands, 283 records, receptor **4HW3**, chain B of it. The kept ligands sit 0.93 A from their crystal pose on average, at most 1.39 A.
