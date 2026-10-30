# Poses found without the crystal pose

One SDF per set and posing method: every ligand of a free-energy set in the pose that was found
for it **while it was held out**. Plan item D8 asks what a conformational ensemble is worth where
the pose is not known, and these files are the starting point for that question;
`examples/make_md_trajectories.py --ligands` simulates them and `benchmarks/run_md.py --docked`
measures them.

## What each file is

`<set>_<method>.sdf`, one record per ligand under the ligand's own title, carrying

* `pAffinity`, the activity as the source collection gives it,
* `pose_rmsd`, how far the pose that was found is from the crystal pose, in angstrom, measured
  in the frame both are already in and without superposition,
* `pose_method`, the posing method, and `pose_fold`, the fold the ligand was held out in.

`<set>_<method>.json` holds the same numbers as a record of the run, including every ligand's
`pose_rmsd`.

## How they were made

`examples/make_docked_poses.py` is the script and the whole recipe: nothing was done by hand. The ligands of a set are split into **5 folds** with seed 0. For each fold a CMF model is fitted on the other ligands in their crystal poses (fields electrostatic, steric, hydrophobic, hba, hbd, attenuation 0.5, lambda 0.1), the fold's ligands have their poses destroyed -- a random rigid placement within 2 A of the site, the same destruction `benchmarks/run_poses.py` uses -- and one posing method puts them back, from the orientation the destruction left it in. So no ligand is posed with the help of its own crystal pose or its own activity. What is still taken from the crystal structure is the *conformation*: the posing is rigid, so only the placement is searched.

* `mcl1_pocket.sdf`: 25 ligands, method **pocket**, restarts 2 of at most 30 degrees (1007 atoms of the binding site). The poses sit a median 1.12 A from the crystal poses, at most 8.26 A; 36% within 1 A, 56% within 2 A.
* `mcl1_pocket_rot180.sdf`: 25 ligands, method **pocket**, restarts 20 of at most 180 degrees (1007 atoms of the binding site). The poses sit a median 0.91 A from the crystal poses, at most 1.21 A; 68% within 1 A, 100% within 2 A.

## Where the data comes from

* The ligands and their activities: the Open Force Field protein-ligand-benchmark, free under
  MIT, already in `examples/openff/` with its licence.
* The binding sites: `examples/pockets/openff/`, cut from the PDB entries those files name. The
  wwPDB distributes its holdings in the public domain under CC0.
