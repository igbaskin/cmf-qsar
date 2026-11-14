# The eight Sutherland data sets

Eight series with published three-dimensional alignments, from the supplementary material of
Sutherland, O'Brien and Weaver, *A comparison of methods for modeling quantitative
structure-activity relationships*, J. Med. Chem. 47 (2004) 5541-5554.

| File | Molecules | Activity |
|---|---|---|
| `ACE.sdf` | 114 | pIC50, angiotensin-converting enzyme |
| `AChE.sdf` | 111 | pIC50, acetylcholinesterase |
| `BZR.sdf` | 147 | pIC50, benzodiazepine receptor |
| `COX2.sdf` | 282 | pIC50, cyclooxygenase 2 |
| `DHFR.sdf` | 361 | pIC50, dihydrofolate reductase |
| `GPB.sdf` | 66 | pKi, glycogen phosphorylase b |
| `THER.sdf` | 76 | pKi, thermolysin |
| `THR.sdf` | 88 | pKi, thrombin |

`examples/make_sutherland_examples.py` converts the original MOL2 files and activity tables.
Every record carries the activity, the published split into training and test set (`subset`),
and the Sybyl atom types and partial charges of the source as per-atom lists. **The coordinates
are exactly the published ones.** Nothing is aligned again and nothing is refined.

## Close pairs of atoms in the published structures

The geometry check of this repository (`cmfqsar.orca.geometry_faults`) asks two things of a
structure: every bond length is between 0.7 and 1.3 times the sum of the covalent radii, and no
two atoms without a bond between them are closer than 1.3 Å. Two hydrogens on one carbon are
1.75 Å apart, so a real molecule has no pair below that limit.

Every bond of the eight files passes. 30 of the 1,245 structures have one close pair:

| File | Structures with a close pair | Closest pair |
|---|---|---|
| `DHFR.sdf` | 25 of 361 | 0.85 Å |
| `COX2.sdf` | 2 of 282 | 0.92 Å |
| `ACE.sdf` | 1 of 114 | 0.96 Å |
| `BZR.sdf` | 2 of 147 | 1.12 Å |

In all 30 the pair is two hydrogens, or a hydrogen and an oxygen, that are five to seven bonds
apart: two neighbouring groups of the molecule whose hydrogens were not moved apart when the
structure was built. No pair of heavy atoms is involved. The same pairs are in the original MOL2
files, with the same distances (checked on 14 November 2026).

These structures are kept as they are. The sets are used here to compare methods on the
published alignment, and a changed coordinate would make the comparison one with a different
data set. The 30 records are listed by name in `examples/geometry_exceptions.csv`, and
`tests/test_example_geometries.py` fails if any other record of any example file fails the check.

The background sets in `examples/decoys/` take molecules from these files with their
coordinates, so twelve of their records have the same close pairs.
