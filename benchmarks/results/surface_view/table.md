# The surface side of the program on screen: what it costs, and what it reads

`python benchmarks/run_surface_view.py --force`, written 2026-11-07 00:24:32.

## The slab behind the molecules

The slab is 144 atoms of Fe(110) in 4 layers, the one the adsorption energies were relaxed on. A patch is the surface atoms within the radius of a molecule, measured **in the plane**, so the surface stays a surface under a molecule that sits high above it. One row per radius: the median over the first 5 molecules of each set.

| Set | Patch radius | Surface atoms drawn | Choosing them | Drawing the whole view |
| --- | ---: | ---: | ---: | ---: |
| benzimidazole (73 molecules) | 6 A | 93 | 1 ms | 3 ms |
| benzimidazole (73 molecules) | 10 A | 196 | 1 ms | 4 ms |
| benzimidazole (73 molecules) | 16 A | 421 | 1 ms | 6 ms |
| benzimidazole (73 molecules) | 24 A | 845 | 1 ms | 11 ms |
| quinoline (65 molecules) | 6 A | 140 | 1 ms | 5 ms |
| quinoline (65 molecules) | 10 A | 259 | 1 ms | 7 ms |
| quinoline (65 molecules) | 16 A | 513 | 1 ms | 9 ms |
| quinoline (65 molecules) | 24 A | 979 | 2 ms | 15 ms |

On the benzimidazole set the highest drawn atom is at z = +0.000 A, which is the top atomic layer of the frame, and 100 % of the drawn atoms are atoms of the complementarity receptor itself: the picture shows what the score is computed against.
On the quinoline set the highest drawn atom is at z = +0.000 A, which is the top atomic layer of the frame, and 100 % of the drawn atoms are atoms of the complementarity receptor itself: the picture shows what the score is computed against.

## The surface frame: the two numbers the panel states

The window states the height of the lowest heavy atom and the tilt of the molecule's own plane. The height is checked against the `height_A` column the relaxation wrote into the data set, which the window never reads. The rule pose of `cmfqsar.planealign` lays a molecule flat, so its tilt is 0 by construction and the last column is how far the relaxed pose is from flat.

| Set | Molecules | Height of the relaxed poses | Agreement with the written height | Tilt of the relaxed poses (median, 90th) | Molecules with a flat plane | Flat rule pose minus relaxed tilt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| benzimidazole | 73 | 1.23 to 4.83 A | 0.005 A at worst (73 matched) | 18.1, 71.7 deg | 29 % | 18.1 deg |
| quinoline | 65 | 1.24 to 4.83 A | 0.005 A at worst (65 matched) | 27.3, 65.5 deg | 14 % | 27.3 deg |

The agreement with the written height is the check that matters: the window computes the number from the coordinates and the relaxation wrote it from its own trajectory, so they agree only if the window is reading the frame the poses are given in.

## The vector co-field as arrows

The electric-field kind as the only field of a model on 95 Hammett benzenes, target sigma_m. The arrows are the 16 longest gradients of the co-field, kept 1.5 A apart. 'Longest arrow' and 'largest gradient on the grid' must agree, or the arrows are not the strongest places. The weakest arrow drawn says how far down the reading goes, as a fraction of the longest and as the share of the volume whose gradient is at least that strong. The direction change is the mean angle between an arrow of the coarsest grid and the nearest arrow of this one.

| Grid spacing | Grid points | Gradient | Arrows | Longest arrow | Largest gradient on the grid | Weakest arrow | Volume at least that strong | Direction change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.70 A | 8740 | 0.27 s | 16 | 0.2468 | 0.2468 | 45 % of the longest | 1.8 % | reference |
| 0.50 A | 21600 | 0.70 s | 16 | 0.2502 | 0.2502 | 50 % of the longest | 1.5 % | 16.3 deg (16 matched) |
| 0.35 A | 61560 | 1.96 s | 16 | 0.2557 | 0.2557 | 40 % of the longest | 2.2 % | 16.0 deg (15 matched) |

The longest arrow is the largest gradient the grid holds, on every grid, and the arrows of a finer grid stand within one grid spacing of the arrows of the coarsest one: the places the drawing points at are not an artefact of the sampling. The direction is another matter. Refining the grid turns the arrows by about 16 degrees on average, because an arrow is taken at the grid point nearest a peak and a finer grid has a nearer one. So an arrow states a direction to roughly fifteen degrees and not better. That is enough for the question it answers -- which way the field should point -- and not enough to measure an angle off the picture.

## The co-pattern chart against the channel table

The chart draws the channel weights of one pair of field types against distance. Two things make it a reading rather than a picture. The weights must reproduce the prediction: the bias plus the sum over every channel of weight times channel is the prediction, and the residual below is the largest difference over the training structures. And where a kernel resolves a distance further -- the surface kernel resolves it by the heights of both atoms -- a point of the curve is the sum over what is left, which the last column states.

| Kernel | Structures | Channels | Pairs of field types | Points per curve | Residual of the prediction | Leading pair | Share of the reading | Channels summed per point |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| periodic autocorrelation, 13 COD electrolytes | 13 | 126 | 6 | 21 | 8e-14 (target sd 16.3) | Electrostatic | 26 % | 1 |
| surface autocorrelation, 73 Fe(110) adsorbates | 73 | 3672 | 6 | 17 | 2e-12 (target sd 63.8) | Electrostatic x Hydrophobic | 28 % | 36 |

On the periodic autocorrelation, 13 COD electrolytes the curve of the leading pair peaks at 8.00 with +3.2588, and one curve costs 0.1 ms to build, so the chart follows the box without a wait.
On the surface autocorrelation, 73 Fe(110) adsorbates the curve of the leading pair peaks at 2.25 with +16.7619, and one curve costs 1.2 ms to build, so the chart follows the box without a wait.

## The bonds the crystal view draws

A structure file says where the atoms are and nothing about which of them are bonded, so the bonds are perceived from the distances, for drawing only. The molecules of the two Fe(110) sets came from SDF files, which do carry a bond table, so the perception has a ground truth on real molecules with their hydrogens. 'In the open' is the molecule alone; 'in a cell that cuts it' is the same molecule placed on the Fe(110) slab in its cell and wrapped so that it straddles the cell boundary, which is what the perception over the periodic images is for.

| Set | Molecules | Bond table reproduced, in the open | ... in a cell that cuts the molecule | ... of those that fit in the cell | ... of those longer than its edge | Sticks leaving the cell (median) | Perceiving, in the open | ... in a cell |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| benzimidazole | 73 | 72 (99 %) | 68 (93 %) | 65 of 65 | 3 of 8 | 14 | 0.1 ms | 1 ms |
| quinoline | 65 | 63 (97 %) | 58 (89 %) | 57 of 58 | 1 of 7 | 14 | 0.1 ms | 1 ms |

**The two columns at the right are the result.** For every molecule that fits inside the surface cell (16.68 x 11.79 A) the perception over the periodic images returns the molecule's own bond table, and the molecules it gets wrong are almost exactly the molecules that are longer than a cell edge. Those touch their own periodic image, so two atoms far apart in the molecule become neighbours across the boundary and the rule joins them. That is a property of the cell and not of the rule, and it is worth knowing before a long inhibitor is drawn in a small surface cell: widen the cell, or read the sticks with care. The few misses in the open are long bonds in a relaxed pose that the covalent radii plus 0.45 A do not reach across.

No bond reaches the 144-atom slab in any of these cells, which is the rule the drawing rests on: a distance rule would give every atom of a close-packed metal about twelve neighbours, and twelve sticks per atom hide the structure they are meant to show. The stage asserts it rather than reporting it, so the benchmark fails if a bond ever reaches the metal.

What a metal gets instead is its first coordination shell, drawn as a polyhedron. The shell is found without any chemistry: the nearest neighbour sets a distance and every neighbour within 1.3 times that distance is a vertex, with the non-metal neighbours preferred where the atom has any.

| Structure | Metal atoms | Polyhedra drawn | Left out: a shell of fewer than three | Coordination numbers found | Expected | Polyhedra | Bonds, and their time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rock salt NaCl, 6-coordinate by construction | 4 | 4 | 0 | 6 | 6 | 5 ms | - |
| cod_1545083 (256 atoms, from the COD) | 160 | 64 | 96 | 4, 6, 8 | not known in advance | 26 ms | 0 in 2 ms |
| cod_1552152 (256 atoms, from the COD) | 160 | 160 | 0 | 4, 6, 8 | not known in advance | 51 ms | 0 in 2 ms |
| cod_1545085 (192 atoms, from the COD) | 96 | 80 | 16 | 3, 4, 6, 8 | not known in advance | 31 ms | 0 in 2 ms |

The fourth column is why a cell can have more metal atoms than polyhedra, and it is a limit of the ratio rule worth stating. On a disordered conductor the lithium sites are partly occupied and their distances to oxygen are very uneven, so for 96 of the 160 lithium atoms of `cod_1545083` the nearest oxygen is more than 1.3 times closer than the next one and the shell comes out as a single atom, which is not a shape. Those atoms get no polyhedron. Widening the ratio would draw them and would also inflate every well-behaved shell, so the convention is kept and the count is reported.

The three cells from the COD are the largest of `examples/hops/disordered/`, which is where a window would be slow if it were going to be. They are garnets and related conductors of lithium, lanthanum, zirconium and oxygen: only the oxygen is a bonding element and no two oxygen atoms are within 2.4 A, so the distance rule gives them no bonds at all and the polyhedra are the whole of their reading. That is the case the two options were built as a pair for.
