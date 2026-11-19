# Bimetallic alloy slabs with H and OH adsorption energies

Adsorption energies of H and OH on 1,928 slabs of metal alloys. The hydrogen energy is a
descriptor of the hydrogen evolution reaction (HER). The hydroxyl energy is the first descriptor
of the oxygen reduction reaction (ORR). The set has no OOH, so the limiting potential of the ORR
cannot be computed from it.

The data are not computed here. They are read from the raw output of Quantum ESPRESSO (BEEF-vdW)
that Mamun et al. published. `examples/make_alloy_examples.py` builds this folder from that output,
and `benchmarks/run_alloy.py` reads it.

| file | what it holds |
| --- | --- |
| `slabs.extxyz` | the 1,928 bare slabs, relaxed, with the formula, the energy and the source log in the comment line |
| `adsorption.csv` | one row for each adsorbed slab that could be matched to its bare slab (16,115 rows) |
| `gas_references.json` | the energies of H2 and H2O that the adsorption energies use |

## The slabs

Every slab is a 2 x 2 cell of three metal layers (12 atoms). The cell is 24 to 26 Angstrom tall,
so the vacuum is at least 18.9 Angstrom. The slabs contain 37 elements. A slab has one, two or
three elements.

## Columns of `adsorption.csv`

| column | meaning |
| --- | --- |
| `log` | the name of the log in the archive of the source (the numbers carry no meaning) |
| `slab` | the number of the bare slab in `slabs.extxyz` (`slab` in its comment line) |
| `formula` | the composition of the bare slab |
| `adsorbate` | `H` or `OH` |
| `e_ads` | the adsorption energy in eV (see below) |
| `x`, `y`, `z` | the position of the adsorbate atom (the H, or the O of OH) in the frame of the bare slab, in Angstrom |
| `height` | the height of that atom over the mean height of the top layer, in Angstrom |
| `site` | `top`, `bridge`, `fcc`, `hcp` or `subsurface`; see below |
| `n_bind` | the number of top-layer atoms that bind the adsorbate |
| `bind_elements` | the elements of those atoms, in alphabetical order |
| `d_nearest` | the distance to the nearest top-layer atom, in Angstrom |
| `match_distance` | how well the adsorbed slab matches its bare slab: the largest distance of any atom, in Angstrom |
| `steps` | the number of steps of the relaxation |
| `step_limit` | `True` if the relaxation stopped at the step limit of 50, so the geometry may not be relaxed |
| `duplicate_of` | empty for a kept row; otherwise the log of the row with the same geometry and energy that was kept |

The adsorption energies are

    E_ads(H)  = E(slab + H)  - E(slab) - 1/2 E(H2)
    E_ads(OH) = E(slab + OH) - E(slab) - (E(H2O) - 1/2 E(H2))

All energies are electronic energies. There is no correction for zero-point energy or entropy.
For the free energy of H add about 0.24 eV, and for OH about 0.35 eV; the single-atom catalyst
benchmark of this repository uses these two numbers.

## How the data were made

1. Every log of `H.tar.gz` (17,212 logs) and of `references.tar.gz` (4,060 logs) is read once.
   The last computed structure and the final energy are kept.
2. A slab with one H (13 atoms) or one H and one O (14 atoms) is matched to the bare slab with the
   same elements. The match allows the 12 rotations and mirrors of the hexagonal layer and every
   shift in the plane, because the two logs can use different cell vectors and origins. A
   match needs every atom to be nearer than 1 Angstrom to its partner.
3. The energies are computed as above, with H2 and H2O from the reference archive.
4. The site is found from the position of the adsorbate in the frame of the bare slab: top,
   bridge, or hollow with 3 neighbours. A hollow site is `hcp` if an atom of the second layer is
   directly below it and `fcc` if not. An adsorbate more than 0.3 Angstrom below the top layer is
   `subsurface`.
5. Two rows of the same slab and adsorbate are duplicates if the adsorbate is within 0.4 Angstrom in
   the plane and 0.3 Angstrom in height and the energies differ by less than 0.02 eV. The
   lowest-energy row stays.

## What the benchmark found

`benchmarks/run_alloy.py` compares field kernels with models that use only element names or
tabulated element properties (the *references*). The full tables are in
`benchmarks/results/alloy/table.md`. Differences are in mean absolute error (MAE) with a paired
bootstrap interval of 95 %.

- **One value for each slab (the lowest energy of its sites).** For H the best reference, the mean
  of tabulated element properties over the whole slab, has an MAE of 0.189 eV. The best field
  method, the periodic autocorrelation with extended fields, has 0.199 eV. The field method is
  slightly worse (+0.011 eV, interval +0.003 to +0.018). For OH (213 slabs) the autocorrelation of
  indicator fields is better with random folds (-0.025 eV, interval -0.050 to -0.002) but not
  when whole alloys are held out.
- **One value for each site.** The site references are of three kinds, and none uses a field:
  - *element references*: the names or properties of the binding elements, the site kind and the
    elements of the slab;
  - *neighbour counts* (`--task counts`): the number of atoms of each element in four shells (up
    to 2.3, 3.3, 4.5 and 6.0 Angstrom) around the same point that the field methods use, and
    variants of it;
  - *shell properties* (`--task shells`): for four shells (up to 2.5, 3.5, 4.5 and 6.0 Angstrom)
    around the same point, the number of atoms and the sum of their standardised element
    properties, with the site kind. The columns are standardised before the Gaussian kernel.

  The best field method is the overlap kernel with extended fields. The table compares it with
  the best neighbour-count reference and with the shell properties (MAE in eV; the last column is
  a paired bootstrap interval of 95 %, with whole slabs resampled):

  | adsorbate | folds | overlap kernel, extended fields | best neighbour-count reference | shell properties | overlap kernel minus shell properties |
  | --- | --- | --- | --- | --- | --- |
  | H | random | 0.209 | 0.195 | 0.174 | +0.035 [+0.024, +0.046] |
  | H | slab-out | 0.216 | 0.244 | 0.201 | +0.016 [+0.004, +0.027] |
  | H | alloy-out | 0.216 | 0.243 | 0.200 | +0.016 [+0.005, +0.027] |
  | OH | random | 0.240 | 0.222 | 0.208 | +0.032 [+0.024, +0.040] |
  | OH | slab-out | 0.252 | 0.240 | 0.220 | +0.032 [+0.022, +0.043] |
  | OH | alloy-out | 0.255 | 0.245 | 0.222 | +0.032 [+0.023, +0.042] |

  Against the element references and the neighbour counts, the overlap kernel looks better for H
  on new slabs and new alloys (-0.027 eV against the best neighbour-count reference, interval
  -0.040 to -0.015). Against the shell properties it is worse with every kind of fold, for H and
  for OH. The neighbour counts are weaker than the shell properties mainly because they leave out
  the number of atoms in each shell and do not standardise their columns. Without
  standardisation the site kind (columns of 0 and 1) has almost no weight beside the sums. In some
  folds the shell properties chose the widest Gaussian together with the smallest ridge value (1
  of 10 folds for H with slabs held out, 1 to 5 of 10 for OH). A wider grid could only make this
  reference better.
- **What this means.** The names and properties of the elements, counted in shells around the
  site, explain most of the variance (q2 about 0.7 to 0.8). A smooth field of the environment
  around the site adds nothing to them, for H or for OH. The fields also add nothing for the
  lowest energy of a whole slab.

## What to keep in mind

- **Many rows are duplicates.** The source warns that many relaxed geometries are repeated. Of
  16,115 matched rows, 4,118 are duplicates, so 11,997 rows are kept.
- **Some rows are lost.** 775 adsorbed slabs have no bare slab within 1 Angstrom, 165 have no bare
  slab with their elements, and 25 logs did not finish. They are not in the file.
- **Some relaxations are not finished.** 2,441 logs of the source stopped at the step limit of 50.
  The column `step_limit` marks them. The benchmark is run with and without them.
- **Subsurface hydrogen is in the file** (111 rows) but the benchmark leaves it out, because the
  HER is a surface reaction.
- **Only 213 slabs have OH.** The H set covers 1,918 slabs.
- **There is no OOH or O.** The ORR limiting potential cannot be computed without them.
- **The source is DFT at one level of theory** (BEEF-vdW, a 2 x 2 cell of three layers, one
  coverage). The numbers are not measurements.

## Four faults of the logs that the builder handles

1. In a relaxation log the last block of atomic positions belongs to the next step, which was never
   computed. The builder uses the block before it.
2. The reader of ASE fails on 2,196 logs (an error in its array of eigenvalues) and uses another value
   of the Rydberg constant than the value of CODATA 2018. The builder has its own reader and takes the
   constant of ASE, so that the energies equal those of ASE.
3. 458 logs are several single-point runs written one after another. The builder takes the last
   run that has an energy.
4. The adsorbed slab and the bare slab can be written with different cell vectors, a different
   origin, or as a rotated or mirrored copy. A match by rounded positions finds only half of
   them.

## Source and licence

O. Mamun, K. T. Winther, J. R. Boes and T. Bligaard, "High-throughput calculations of catalytic
properties of bimetallic alloy surfaces", *Scientific Data* 6 (2019) 76, doi
10.1038/s41597-019-0080-z. The raw files are in the Materials Cloud record xxdhq-f1v83 (doi
10.24435/materialscloud:2019.0015), licence CC BY 4.0. This folder holds a derived table and the
structures; please cite the paper.
