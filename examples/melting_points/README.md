# Melting points with crystal structures

The data set of plan item M8: 665 organic compounds, each with its melting point and one
crystal structure of it. It is made by `benchmarks/prepare_molcrystals.py` and read by
`benchmarks/run_molcrystals.py`.

| file | what it holds |
| --- | --- |
| `crystals.extxyz` | the unit cells, one frame per compound, in the order of the table; `info["name"]` is the row name |
| `melting_points.csv` | `name`, the COD entry, the ChemSpider identifier, the compound name, its SMILES, the melting point (mean, low and high, degrees C), atoms and molecules in the cell |
| `provenance.json` | how many compounds of the source were matched and why the others were not |

**Sources.** The melting points are those of the ONS Open Melting Point Collection (ONSMP029:
J.-C. Bradley, A. Lang, A. Williams, E. Curtin; figshare, record 93086), which keeps a compound
only when at least two measurements agree; licence CC BY 4.0. The structures are entries of the
Crystallography Open Database (COD, www.crystallography.net), which is in the public domain; each
row names its entry, and the cell is the entry's CIF as ASE reads it.

**How a structure was chosen.** COD was searched by the sum formula of each compound. Entries
measured under pressure, marked as duplicates or without coordinates were left out, and the rest
were tried in order (entries whose name is the compound's name first, then by R factor), at most
five per compound. An entry was accepted when the molecules perceived from its cell by
`cmfqsar.molcrystal` -- bonds across the cell boundary, then bond orders and charges -- are the
compound: every distinct molecule has the InChI skeleton of a component of the compound's SMILES,
and every component is found. That excludes isomers with the same formula, solvates and
co-crystals, and it is a test of the perception.

**What was not matched**, of the 2701 compounds with organic elements: 673 have
no COD entry with their formula (many are liquids at room temperature); for 1105 every entry
tried was another compound; 223 were refused by the perception (no located hydrogens, a
disordered or partly occupied structure, charges that do not add up to zero, which is usually an
acidic hydrogen that was not located); 35 could not be read (a space-group symbol ASE does
not know, an incomplete file, a failed download).

**What to keep in mind.** One structure per compound, so the polymorph is the one COD gave first,
not necessarily the one that melts. The melting points range from -142 to
438 C; compounds that are liquid at room temperature have structures measured cold.
