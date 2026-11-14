# The Merck KGaA free energy benchmark

Ten files (330 ligands) from Schindler, Baumann, Blum and co-workers, *Large-scale assessment of
binding free energy calculations in active drug discovery projects*, J. Chem. Inf. Model. 60
(2020) 5457-5474, <https://github.com/MCompChem/fep-benchmark> (MIT licence, see
`LICENSE_MCompChem_fep-benchmark.txt`): eight congeneric series (CDK8, c-Met, Eg5, HIF-2α,
PFKFB3, SHP-2, SYK, TNKS2) and two variants (Eg5 with the alternative loop conformation, and
the HIF-2α series of 2023).

Each file holds the ligand poses of one series in the binding site, prepared by the authors, so
a file is an aligned congeneric set. `examples/make_schindler_examples.py` adds `pAffinity`
(−log10 of the measured IC50 or Kd in M), the measured and the calculated binding free energies,
the target and the PDB entry. **It does not change the coordinates.**

## One pose with two close hydrogens

The geometry check of this repository (`cmfqsar.orca.geometry_faults`: every bond length between
0.7 and 1.3 times the sum of the covalent radii, no two atoms without a bond closer than 1.3 Å)
fails for one ligand, `CHEMBL1084678`, which is in `eg5.sdf` and in `eg5_alternativeloop.sdf`
with the same pose: two hydrogens six bonds apart are 1.12 Å from each other. No heavy atom is
involved.

The pose is kept as the authors distributed it. The two records are listed in
`examples/geometry_exceptions.csv`. The same pose is in `examples/ross/merck/eg5_extraprotomers.sdf`.
