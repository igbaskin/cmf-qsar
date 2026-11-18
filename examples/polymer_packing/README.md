# Polymer properties set by how chains pack

Two tables for plan item PL2: the density of polymers from molecular dynamics, and their measured
gas permeability. Both are made by `benchmarks/prepare_polymer_packing.py` and read by
`benchmarks/run_polymer_packing.py`.

| file | what it holds |
| --- | --- |
| `pi1070.csv` | 1,077 amorphous homopolymers with their density at 300 K from molecular dynamics |
| `permeability.csv` | 742 polymers with the measured permeability to five gases |
| `LICENSE_RadonPy.txt` | the BSD 3-Clause licence of the density data |
| `LICENSE_torch-molecule.txt` | the MIT licence of the permeability data |
| `provenance.json` | the sources, their versions, how many rows were refused and why |

**Repeat units.** `psmiles` is the repeat unit written as SMILES with two attachment points. For
example, polystyrene is

```
*C(C*)c1ccccc1
```

## Density: RadonPy PI1070

**Columns.** `id` is RadonPy's monomer identifier. `polymer_class` is RadonPy's class number
(1 to 21; 20 occur). `tacticity` is what the simulation used (`atactic` or `none`). `density` is
the mean density in g/cm³ at 300 K and 1 atm, and `density_std` its standard deviation over the
analysed parts of the run. `degree_of_polymerisation` and `mn` are the chain length and the
number-average molar mass of the simulated chains.

**How the values were made.** All values come from the same all-atom protocol, as the paper
describes it: the GAFF2 force field, ten chains of about 1,000 atoms each in an amorphous cell of
about 10,000 atoms, equilibrated and run at 300 K. They are computed, not measured. The dataset
card of the Hugging Face copy (`liuganghuggingface/moldata-radonpy_pi1070`) reports that on 66
polymers shared with an experimental set, 44 % agree within 0.05 g/cm³ (correlation 0.98).

**Source.** Y. Hayashi, J. Shiomi, J. Morikawa and R. Yoshida, "RadonPy: automated physical
property calculation using all-atom classical molecular dynamics simulations for polymer
informatics", *npj Computational Materials* 8, 222 (2022), doi:10.1038/s41524-022-00906-4. The
file is `data/PI1070.csv` of github.com/RadonPy/RadonPy at commit 1387bc8. Licence: BSD 3-Clause,
reproduced in `LICENSE_RadonPy.txt` as the licence requires.

**What is not used.** The table also gives a van der Waals volume per repeat unit, from which
the packing coefficient (the fraction of the volume the atoms fill) would follow. For several
nylons that fraction comes out above 1, which is impossible, so the volume is wrong there and the
packing coefficient is not offered as a target.

## Gas permeability

**Columns.** `id` is the row number of the source, written `perm001` and so on.
`log10_CH4`, `log10_CO2`, `log10_H2`, `log10_N2` and `log10_O2` are the decimal logarithms of the
permeability in Barrer (1 Barrer = 10⁻¹⁰ cm³(STP)·cm/(cm²·s·cmHg)). An empty cell means the
permeability to that gas was not measured. The counts are 607, 681, 450, 716 and 728.

**Source.** The polymer gas permeability set released with torch-molecule
(github.com/liugangcode/torch-molecule; Gang Liu, Yihan Zhu and Eric Inae), read from its copy on
Hugging Face, `liuganghuggingface/moldata-polymer_gas_permeability`, at revision 2492824. The
values are measured and were compiled from the literature. The release states no unit; the
dataset card shows by comparison with another set that they are in Barrer. Licence: MIT,
reproduced in `LICENSE_torch-molecule.txt`.

**Refused rows.** 44 of the 786 rows are left out:

- 41 repeat units do not have exactly two attachment points. 31 of them have four: these are
  ladder polymers, among them the polymers of intrinsic microporosity, which are the most
  permeable class. The oligomer builder handles linear chains only.
- Two contain germanium, which has no Gasteiger charges.
- One carries a net charge.

**What the data do not say.** The film preparation, the thermal history, the temperature and the
pressure of each measurement are not given. Permeability can change by a factor of several with
these, so part of the spread between similar polymers is not in the repeat unit at all.
