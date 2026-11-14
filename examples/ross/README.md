# The Schrödinger binding free energy benchmark

67 congeneric series (1,254 ligands) from the public benchmark of Ross, Lu, Scarabelli and
co-workers, *The maximal and current accuracy of rigorous protein-ligand binding free energy
calculations*, Commun. Chem. 6 (2023) 222,
<https://github.com/schrodinger/public_binding_free_energy_benchmark> (MIT licence, see
`LICENSE_schrodinger_benchmark.txt`).

Each file holds the prepared ligand poses of one series in the binding site of its protein, so a
file is an aligned congeneric set. `examples/make_ross_examples.py` converts the measured binding
free energy to `pAffinity` and adds the protein, the PDB entry, the group and the source of the
series (`benchmark_metadata.csv`). **It does not change the coordinates.** Series with fewer than
five ligands are left out.

## Close pairs of hydrogens in ten poses

The geometry check of this repository (`cmfqsar.orca.geometry_faults`) asks that every bond
length is between 0.7 and 1.3 times the sum of the covalent radii and that no two atoms without
a bond are closer than 1.3 Å. Every bond of the 67 files passes. Ten poses in seven files have a
close pair:

| File | Record | Closest pair |
|---|---|---|
| `gpcrs/ox2_hip_custcore.sdf` | `set 1: 13` | H–H, 0.60 Å |
| `merck/syk_4puz_fullmap.sdf` | `CHEMBL3265011 flip` | H–H, 0.85 Å (four pairs below 1.3 Å, one of them C–H) |
| `merck/syk_4puz_fullmap.sdf` | `CHEMBL3265009` | H–H, 1.20 Å |
| `merck/syk_4puz_fullmap.sdf` | `CHEMBL3265015 flip` | H–H, 1.27 Å |
| `merck/eg5_extraprotomers.sdf` | `CHEMBL1084678` | H–H, 1.12 Å |
| `misc/cdk8_koehler.sdf` | `21` | H–H, 1.13 Å |
| `jacs_set/mcl1_extra_flips.sdf` | `45 flip` | H–H, 1.22 Å |
| `misc/galectin3_extra.sdf` | `18f` | F–H, 1.26 Å |
| `fragments/frag_liga_auto.sdf` | `12` and `11` | H–H, 1.26 Å |

No pair of heavy atoms is involved. The poses are inputs of free energy calculations, which
relax every structure before anything is measured, so a close pair of hydrogens in an input is
not a problem there. Here the poses are read as they are. They are kept unchanged because the files
are a copy of a published benchmark, and the ten records are listed in
`examples/geometry_exceptions.csv`. The pose of `CHEMBL1084678` is the same one as in
`examples/schindler/eg5.sdf`.
