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

## The measured sets: 23 corrosion-inhibitor series

The Fe(110) target came from the same relaxation as the pose, so a score read
there cannot be told from a readout of the relaxation. These series have no such
circularity: a measured potency in acid and, for most molecules, a measured
Langmuir adsorption free energy. The slab terms are read at a pose this program
produces itself: embedded, laid flat by the rule of item EC2, then refined
against the slab. `counts` is atom-type counts with no geometry at all, and the
grids, the fold count and the selection convention are those of
`run_corrosion.py`, whose published means over these series are counts +0.189,
the aligned five-field model +0.269 and the invariant kernel +0.278.

### activity (23 series)

| model | mean q2 over the series | series where it beats counts |
|---|---|---|
| counts | +0.189 | |
| fields | +0.279 | 15 of 23 |
| surface | +0.032 | 7 of 23 |
| fields+surface | +0.190 | 12 of 23 |

The surface block adds to the five fields in 5 of 23 series.

### dG_ads (18 series)

| model | mean q2 over the series | series where it beats counts |
|---|---|---|
| counts | +0.126 | |
| fields | +0.201 | 11 of 18 |
| surface | -0.057 | 3 of 18 |
| fields+surface | +0.080 | 8 of 18 |

The surface block adds to the five fields in 1 of 18 series.

### Every series

| series | n | target | counts | fields | surface | fields+surface | best single term |
|---|---|---|---|---|---|---|
| 0_c1ccccc1 | 244 | activity | +0.437 | +0.363 | +0.134 | +0.358 | burial (0.38) |
| 0_c1ccccc1 | 207 | dG_ads | +0.105 | +0.091 | +0.065 | +0.090 | burial (0.26) |
| 11_c1c_nH_c_nH_1 | 23 | activity | +0.448 | +0.369 | -0.146 | +0.210 | burial (0.31) |
| 11_c1c_nH_c_nH_1 | 22 | dG_ads | +0.358 | +0.346 | -0.031 | +0.183 | over_hollow (0.21) |
| 12_c1ccoc1 | 28 | activity | +0.339 | +0.152 | -0.112 | +0.112 | height (0.34) |
| 13_c1c_nH_nn1 | 25 | activity | +0.329 | +0.788 | +0.052 | +0.345 | burial (0.61) |
| 13_c1c_nH_nn1 | 23 | dG_ads | +0.864 | +0.294 | -0.041 | +0.122 | over_hollow (0.31) |
| 14_O_c1cc_nH_c_O_nH_1 | 24 | activity | +0.737 | +0.354 | -0.044 | +0.437 | burial (0.40) |
| 157_c1ccc2_nH_cnc2c1 | 73 | activity | +0.268 | +0.272 | +0.124 | +0.218 | burial (0.47) |
| 157_c1ccc2_nH_cnc2c1 | 59 | dG_ads | +0.038 | +0.140 | +0.043 | +0.072 | burial (0.37) |
| 158_c1ccc2ncccc2c1 | 65 | activity | +0.032 | +0.316 | -0.077 | +0.241 | height (0.15) |
| 158_c1ccc2ncccc2c1 | 63 | dG_ads | +0.095 | +0.452 | -0.024 | +0.339 | height (0.26) |
| 159_C_Nc1ccccc1_c1ccccc1 | 28 | activity | -0.086 | +0.229 | -0.070 | +0.125 | burial (0.35) |
| 159_C_Nc1ccccc1_c1ccccc1 | 21 | dG_ads | -0.131 | -0.109 | -0.274 | -0.267 | image_eV (0.25) |
| 160_c1ccc2ccccc2c1 | 15 | activity | -0.217 | -0.005 | -0.182 | -0.208 | burial (0.40) |
| 161_c1ccc2scnc2c1 | 20 | activity | +0.043 | +0.131 | +0.088 | +0.082 | burial (0.48) |
| 161_c1ccc2scnc2c1 | 16 | dG_ads | +0.131 | +0.425 | +0.074 | +0.423 | burial (0.45) |
| 165_O_c1cc_nH_n1_c1ccccc1 | 15 | activity | -0.111 | +0.300 | -0.068 | -0.040 | burial (0.52) |
| 165_O_c1cc_nH_n1_c1ccccc1 | 15 | dG_ads | +0.182 | +0.399 | -0.002 | +0.089 | image_eV (0.58) |
| 17_c1nnco1 | 17 | activity | +0.346 | +0.071 | -0.094 | -0.057 | image_eV (0.38) |
| 17_c1nnco1 | 15 | dG_ads | +0.068 | -0.002 | -0.221 | +0.150 | burial (0.25) |
| 1_c1ccncc1 | 109 | activity | +0.072 | +0.327 | +0.028 | +0.291 | burial (0.29) |
| 1_c1ccncc1 | 102 | dG_ads | +0.018 | +0.169 | +0.024 | +0.153 | burial (0.33) |
| 20_O_c1cc_nH_nH_1 | 15 | activity | -0.111 | +0.300 | -0.068 | -0.040 | burial (0.52) |
| 20_O_c1cc_nH_nH_1 | 15 | dG_ads | +0.182 | +0.399 | -0.002 | +0.089 | image_eV (0.58) |
| 21_O_C1CCN1 | 17 | activity | +0.030 | +0.247 | -0.173 | -0.129 | height (0.11) |
| 21_O_C1CCN1 | 17 | dG_ads | +0.121 | -0.101 | -0.195 | -0.193 | over_top (0.28) |
| 22_C1CNOC1 | 16 | activity | -0.004 | +0.288 | -0.047 | -0.004 | image_eV (0.35) |
| 2_c1c_nH_cn1 | 96 | activity | +0.335 | +0.521 | +0.259 | +0.522 | over_bridge (0.39) |
| 2_c1c_nH_cn1 | 72 | dG_ads | +0.089 | +0.322 | +0.089 | +0.272 | over_bridge (0.29) |
| 3_c1cscn1 | 27 | activity | +0.158 | +0.452 | +0.092 | +0.502 | burial (0.47) |
| 3_c1cscn1 | 21 | dG_ads | -0.154 | +0.482 | -0.256 | -0.100 | burial (0.46) |
| 4_c1nnc_nH_1 | 40 | activity | +0.485 | +0.276 | +0.007 | +0.270 | burial (0.39) |
| 4_c1nnc_nH_1 | 34 | dG_ads | +0.629 | +0.254 | -0.024 | +0.162 | image_eV (0.32) |
| 5_c1cncnc1 | 16 | activity | -0.082 | +0.161 | +0.544 | +0.594 | over_top (0.74) |
| 7_c1nncs1 | 29 | activity | -0.044 | +0.038 | +0.032 | +0.100 | image_eV (0.44) |
| 7_c1nncs1 | 25 | dG_ads | -0.067 | -0.051 | -0.106 | -0.089 | over_hollow (0.29) |
| 8_c1cn_nH_c1 | 30 | activity | +0.644 | +0.515 | +0.521 | +0.501 | burial (0.47) |
| 8_c1cn_nH_c1 | 21 | dG_ads | -0.155 | +0.216 | +0.005 | +0.099 | image_eV (0.14) |
| 9_c1nc_nH_n1 | 25 | activity | +0.289 | -0.048 | -0.071 | -0.065 | image_eV (0.23) |
| 9_c1nc_nH_n1 | 20 | dG_ads | -0.097 | -0.110 | -0.152 | -0.152 | over_top (0.22) |

