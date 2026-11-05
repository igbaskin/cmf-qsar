# A metal slab as the receptor

Plan item EC6. The Fe(110) sets hold the relaxed adsorbed pose and the
adsorption energy of every molecule from the UMA potential, so this item
has reference poses that no protein set has. Two halves: does docking
constrained to the surface recover the pose, and does a complementarity
score track the energy.

## The receptor

The slab is 144 atoms of Fe in 4 layers (6 x 6 surface cell, a = 2.78 angstrom), its top layer at z = 0. The receptor is its top
two layers with periodic images out to 12.0 angstrom: 1080 atoms.

**Fields the slab carries:** steric. **Identically zero:** electrostatic, hydrophobic, hba, hbd — a one-element cell has no electronegativity differences, so every charge model gives zero, and the hydrophobic and hydrogen-bond increments are fitted to organic fragments. All four complementary pairs are therefore zero against a clean metal and what is left is burial.

**The image charge**, which is what a metal has instead. For the first molecule of the series it is -0.0328 eV at its relaxed pose and -0.0030 eV three angstrom higher. It changes by 4.2e-17 eV under an in-plane shift and 1.4e-17 eV under a turn about the normal, so it cannot say where on the lattice a molecule sits; only the shape term can.

| surface sites of the top layer | count |
|---|---|
| bridge | 89 |
| hollow | 54 |
| top | 36 |

## The pose

Median error against the relaxed pose over heavy atoms: in-plane RMSD after the
best rotation about the normal and translation in the plane (the lattice site a
molecule sits over is not part of a pose), and the RMSD of the heights alone.
`rule` is item EC2's flat rule pose; `docked` refines it against the slab.

### benzimidazole (72 molecules)

| start | in-plane (A) | in-plane, bound (A) | height (A) | within 1 A |
|---|---|---|---|---|
| rule | 2.46 | 2.41 | 1.79 | 0 % |
| docked | 2.50 | 2.39 | 1.91 | 0 % |
| docked-both-faces | 2.37 | 2.20 | 1.77 | 7 % |
| docked-with-image | 2.45 | 2.40 | 1.89 | 0 % |
| conformer-rule | 2.47 | 2.17 | 1.62 | 12 % |
| conformer-docked | 2.74 | 2.30 | 1.85 | 14 % |

### quinoline (63 molecules)

| start | in-plane (A) | in-plane, bound (A) | height (A) | within 1 A |
|---|---|---|---|---|
| rule | 3.45 | 3.15 | 2.65 | 3 % |
| docked | 3.66 | 3.15 | 2.78 | 2 % |
| docked-both-faces | 3.23 | 3.15 | 2.88 | 5 % |
| docked-with-image | 3.78 | 3.23 | 2.82 | 2 % |
| conformer-rule | 3.59 | 3.19 | 2.56 | 8 % |
| conformer-docked | 3.73 | 3.30 | 2.71 | 8 % |

## The energy

Leave-one-out q2 of a straight line on each term against `E_ads_kJmol`. The two
controls are item EC2's: the lowest heavy-atom height of the relaxed pose, and
the chemisorbed flag. A score that does not beat the height has found nothing
the relaxation had not already put into the frame.

### benzimidazole (72 molecules, 54 chemisorbed)

| term | leave-one-out q2 | r |
|---|---|---|
| burial | +0.638 | -0.809 |
| image | +0.287 | +0.580 |
| height | +0.472 | +0.703 |
| burial+image | +0.661 | |
| burial_docked | +0.115 | -0.416 |
| image_docked | -0.028 | +0.151 |
| height_docked | -0.057 | -0.104 |
| burial+image_docked | +0.108 | |
| control_relaxed_height | +0.472 | |
| control_bound_flag | +0.345 | |

### quinoline (63 molecules, 31 chemisorbed)

| term | leave-one-out q2 | r |
|---|---|---|
| burial | +0.624 | -0.809 |
| image | +0.408 | +0.691 |
| height | +0.423 | +0.678 |
| burial+image | +0.647 | |
| burial_docked | +0.049 | -0.338 |
| image_docked | -0.073 | +0.037 |
| height_docked | -0.049 | +0.123 |
| burial+image_docked | +0.010 | |
| control_relaxed_height | +0.423 | |
| control_bound_flag | +0.328 | |

