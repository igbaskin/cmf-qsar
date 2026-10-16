# Binding-site pockets of the free-energy benchmark sets

Each file here is the binding site of one data set in `examples/`, trimmed from the crystal
structure that the data set names in its `pdb` property and left in the coordinates of that
structure.

The free-energy collections give their ligands in the frame of the reference entry, so **nothing has
to be superimposed**: opening `examples/openff/cdk2.sdf` in the application loads
`pockets/openff/cdk2.pdb` automatically and the protein appears around the molecules, around their
fields and around the co-fields of a model built from them.

Contents of a file: every residue with an atom within 8 Å of any ligand of the set, taken whole,
with crystallographic waters removed and the co-crystallised ligand removed where it occupies the
same space as the data set. Ions and cofactors lining the site are kept. The provenance of each file
is in its `REMARK 400` lines: the source entry, the cutoff and the closest approach measured between
the ligands and the protein.

Structures come from RCSB and carry no restrictions, the wwPDB distributing its holdings in the
public domain under CC0.

Rebuild or extend them with

```bash
python examples/make_pocket_examples.py                 # openff and schindler
python examples/make_pocket_examples.py --all           # the Schrodinger (ross) sets as well
python examples/make_pocket_examples.py --cutoff 10 --overwrite
```

Sets whose ligands do not sit in the frame of the entry they name are reported and skipped rather
than written, which is why a few targets are missing here. The application makes the same check when
a structure is loaded and states the closest approach, so a mismatched frame is visible immediately
instead of producing a picture that quietly means nothing.
