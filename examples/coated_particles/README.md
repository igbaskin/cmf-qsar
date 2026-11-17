# Thiolate-coated gold particles: core and ligand shell in one file

A small generated set for the particle surface kernel on the command line and in the crystal
window (plan item NT11). It is a computed data set; it is not a measurement.

## The particles

`particles.extxyz` holds 72 particles, written by `make_coated_particles.py` in this folder. Each
is a gold icosahedron of 55 or 147 atoms with a shell of chains on it:

| Part of the recipe | Values |
|---|---|
| Core | icosahedron of 55 or 147 gold atoms |
| Anchor | a sulfur atom 2.4 Å above a surface gold atom |
| Chain | 2, 4 or 6 carbon atoms |
| End group | O, N or C |
| Coverage | 20, 35, 50 or 65 % of the surface gold atoms |

Every chain points along the radial direction, tilted by up to 25 degrees in a random direction;
a chain that would come within 2.5 Å of another is tried again, up to 20 times, then left out.
There is one shell of each of the 72 recipes. The shells are those of plan item NT5, built by
`coated()` in `benchmarks/run_ligand_shells.py` (the first of its two random shells per recipe),
so this file and that benchmark describe the same particles.

Each structure sits in a box with 8 Å of vacuum and is periodic in no direction. Its info holds:

* `name` -- core size, coverage in per cent, chain length and end group, as `Au147-c65-n4-O`;
* `accessibility` -- the target: the mean fraction of each end atom's sphere (covalent radius
  plus a 1.4 Å probe) that no other atom of the core or the shell covers, computed with the
  Shrake-Rupley method of plan item NT3.

## Use

    cmf build-crystals examples/coated_particles/particles.extxyz --kernel particle-surface --core Au --charges electronegativity --pbc none --cutoff 12 --sigma 1 --height-step 2 --folds 5 --save shells.cmf
    cmf predict shells.cmf examples/coated_particles/particles.extxyz

`--core Au` says which atoms are the core; the default `--core metals` finds the same atoms
here. The first command gives q2 = 0.402 in 5-fold cross-validation; the same command with
`--kernel autocorrelation` (the core and the shell together, open boundaries) gives 0.569.
`--charges electronegativity` is there because guessing oxidation states is slow for a whole gold
particle with its ligands: up to 7 s for one of the smallest particles, and the autocorrelation
run with guessed oxidation states had not finished after ten minutes.

To write the file again:

    python examples/coated_particles/make_coated_particles.py
