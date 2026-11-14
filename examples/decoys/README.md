# Background (decoy) sets for four targets

`examples/make_decoy_sets.py` writes these files. For each of the targets ACE, AChE, GPB and THR
the background is 100 molecules drawn from the *other* series of `examples/sutherland/`: ligands
with a measured activity against another protein, which are presumed inactive here. Every decoy
is moved as a rigid body into the frame of the training actives, to the pose in which its fields
are most similar to theirs. The `*_matched.sdf` files hold decoys chosen to match the actives in
molecular weight, lipophilicity, hydrogen-bond donors and acceptors, rotatable bonds and net
charge (`examples/_matched_decoys.py`).

The records carry `subset = decoy`, so the program reads them as a background set.

## Close pairs of atoms

A decoy keeps the internal coordinates of its Sutherland structure; only its position and
orientation change. Twelve records of four files therefore have the close pair of hydrogens (or
of a hydrogen and an oxygen) that their source structure has, between 0.97 and 1.25 Å:

| File | Records with a close pair |
|---|---|
| `ACE_decoys.sdf` | 2 |
| `AChE_decoys.sdf` | 4 |
| `GPB_decoys.sdf` | 4 |
| `THR_decoys.sdf` | 2 |

Ten of the twelve come from `sutherland/DHFR.sdf` and two from `sutherland/BZR.sdf`.
`examples/sutherland/README.md` describes the pairs and why the published coordinates are kept.
The records are listed in `examples/geometry_exceptions.csv`.
