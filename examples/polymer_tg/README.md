# Glass transition temperatures of homopolymers

The glass transition temperature (Tg) of 7,316 homopolymers, each given by its repeat unit. The
table is made by `benchmarks/prepare_polymer_tg.py` and read by `benchmarks/run_polymer_tg.py`.

| file | what it holds |
| --- | --- |
| `tg.csv` | one polymer per row: `id`, `psmiles`, `tg_k`, `polymer_class`, `reliability`, `source` |
| `provenance.json` | the source record, the MD5 checksum of its file, the licence, how many rows were refused and why, and the size of each polymer class |

**The columns.** `psmiles` is the repeat unit written as SMILES with two attachment points, `[*]`,
for example `[*]CC([*])c1ccccc1` for polystyrene. `tg_k` is the curated Tg in kelvin. Where the
sources gave more than one value for a repeat unit, the curators took the median.
`polymer_class` is one of 22 classes given by the source (polyimides, polyoxides, polyesters,
polyacrylics and so on). `reliability` is the curators' label: `black` for a polymer with one
value, `yellow` for two agreeing values, `gold` for more than two agreeing values. `source` names
the compilation the value came from. `id` is the row number of the source file, written as
`tg0001` and so on.

**Source.** S. Kunchapu and K. M. Jablonka, *Curated Glass Transition Temperature for Polymers*,
Zenodo record 15789599 (doi:10.5281/zenodo.15789599), licence CC BY 4.0. The data set is
described in S. Kunchapu and K. M. Jablonka, "PolyMetriX: an ecosystem for digital polymer
chemistry", *npj Computational Materials* 11, 312 (2025), doi:10.1038/s41524-025-01823-y. The
licence asks for attribution, which this paragraph gives. The source file also holds the 100
hierarchical descriptors of the PolyMetriX library and two language-model embeddings. It is not
kept here (44 MB). The preparation script downloads it into `benchmarks/cache/polymer_tg/raw/` and
checks its checksum, and the benchmark reads the descriptors from there.

**Refused rows.** 51 of the 7,367 rows are left out. 36 contain an element for which the
Gasteiger charge scheme gives no charges (tin, sodium, selenium, germanium and seven others),
which would leave the whole molecule without an electrostatic field. 15 carry a net formal
charge, a repeat unit whose counter-ion is not written. The counts are in `provenance.json`.

**What the data do not say.** The molecular weight, its distribution, the tacticity, the
measurement method and the heating rate are not given. All of them change Tg. Two polymers with the
same repeat unit are one row here, so the spread of values between the sources is part of the
error of any model trained on this table.

**The 3D structures.** They are not stored. `cmf polymers` (or `cmfqsar.polymer.build`) turns each
repeat unit into a trimer capped with hydrogen; the benchmark does this itself and keeps the
result in its cache.
