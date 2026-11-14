# Ether molecules with cation binding energies (plan item EC7)

Built by `examples/make_ether_examples.py --fetch` from the MLElectrolytes distribution,
<https://github.com/learningmatter-mit/MLElectrolytes>, branch `master`, of

> Wang W., Yang T., Harris W. H., Gomez-Bombarelli R. "Active learning and neural network
> potentials accelerate molecular screening of ether-based solvate ionic liquids",
> Chem. Commun. 56 (2020) 8920.

The distributor states no licence, so the status of this copy is **unclear** in the sense of
`docs/public_3dqsar_datasets.md`: the facts are the paper's, redistribution is customary in the
field but not explicitly permitted, and the script above rebuilds both files from the original
distribution.

## The files

| File | Content |
|---|---|
| `ether_ions.csv` | 1085 saturated acyclic ethers: SMILES, oxygen count, carbon-to-oxygen ratio, HOMO energy, and the binding energy of Li(+), Na(+) and Mg(2+) in eV at wB97X-D3/def2-TZVP and at BP86-D3/def2-SVP |
| `geometries.sdf` | 1050 of the same molecules in the conformation of the relaxed lithium complex, with the ion position in the `ion_xyz` property and the binding energies repeated as properties |

An empty cell means the quantity is missing upstream, not zero. Of the 1085 molecules,
1048 carry a lithium binding energy, 1013 a sodium one and 1014 a magnesium one at
wB97X-D3/def2-TZVP.

## What the geometries are, and what that costs

The coordinates are the relaxed **complex**: the ether wrapped around the lithium ion, with the
ion removed and its position kept. That is the conformation whose energy was measured, so a
field computed on it is a field of the molecule the number belongs to. The distribution holds
lithium complexes only, so a sodium or magnesium model built on these coordinates uses a
lithium-shaped conformer. The binding energies of the three ions correlate (+0.80 for lithium
against sodium, +0.85 against magnesium), so the error this causes is not visible in a
single-ion model; it is a limit on what the selectivity between two ions can be read as.

Three of the complexes (`e0017`, `e0697` and `e0768`) have two hydrogens of one carbon 1.03 Å
apart; the hydrogens of a methyl or methylene group are 1.75 to 1.8 Å apart. The distributed
geometries hold the same pairs, so this is a property of the source, and the coordinates are
kept as they are. The check that finds them is `cmfqsar.orca.geometry_faults`, and the three
records are listed in `examples/geometry_exceptions.csv`.

## The simple baselines

The oxygen count and the carbon-to-oxygen ratio are columns of the table because any model of
these energies has to be compared with them. On the molecules that carry a lithium binding
energy the oxygen count alone has a correlation of -0.62 with it, and the carbon-to-oxygen ratio
+0.04. The difference between the lithium and sodium energies -- the selectivity -- has a
correlation of -0.05 with the oxygen count, so the count says nothing about it.
