# Molecules on any slab: the generator of the adsorption benchmarks (plan item BT8)

Two adsorption energies are given. *Against the molecule alone* is the Fe(110) script's: E(slab + molecule) - E(slab) - E(molecule relaxed alone in a box). *Against the molecule far above* is E(slab + molecule) - E(slab + the same relaxed molecule 20 A above), both in a cell 40 A taller, so the two systems have the same atoms in the same cell. The *offset of the model* is E(slab + molecule far above) - E(slab) - E(molecule alone): zero for a model whose energy adds over parts that do not interact. UMA's oc20 head, trained on molecules on surfaces, gives -0.10 to -0.33 eV for the solvents and +0.03 to +0.15 eV for the benzimidazoles, so the first energy carries it and the second does not. The pose of every molecule is chosen against the first, which does not change the choice, since the poses of one molecule on one surface share it.

## The Fe(110) recipe against the Fe(110) script

The first molecules of `examples/corrosion/157_c1ccc2_nH_cnc2c1.csv`, run by this script with `--surface fe110`, against the energies the Fe(110) script stored in `examples/corrosion_fe110/157_benzimidazole_uma.csv`.

| Molecule | Stored E_ads, eV | Now, eV | Difference, meV | Pose, stored and now | Against the molecule far above, eV | Offset of the model, eV |
|---|---:|---:|---:|---|---:|---:|
| Benzimidazole | -1.2967 | -1.2966 | +0.1 | flat, flat | -1.369 | +0.081 |
| 2-Methylbenzimidazole | -0.7810 | -0.7815 | -0.5 | flat, flat | -0.803 | +0.032 |
| 5-Nitrobenzimidazole | -1.1344 | -1.1338 | +0.6 | flat, flat | -1.279 | +0.153 |

## Electrolyte solvents on Li(100)

Lattice constant 3.4408 A, cell 5x5x4, model uma-s-1p2p1/oc20, 16 minutes of relaxations (the sum over the molecules). E_ads < 0 is binding. A pose whose heavy atoms all end more than 4 A above the surface is unbound; 'intact' says whether every bond of the molecule survived the relaxation.

| Solvent | E_ads against the molecule far above, eV | kJ/mol | E_ads against the molecule alone, eV | Offset of the model, eV | Pose | Height of the lowest heavy atom, A | Bound | Intact | All poses against the molecule alone, eV |
|---|---:|---:|---:|---:|---|---:|---|---|---|
| EC | -0.504 | -48.7 | -0.796 | -0.276 | flat | 1.89 | yes | yes | flat:-0.796;upright:-0.710;random:-0.671 |
| AN | -0.414 | -40.0 | -0.529 | -0.099 | random | 1.94 | yes | yes | flat:-0.516;upright:-0.192;random:-0.529 |
| PC | -0.363 | -35.0 | -0.621 | -0.242 | upright | 1.96 | yes | yes | flat:-0.404;upright:-0.621;random:-0.323~ |
| MA | -0.173 | -16.7 | -0.333 | -0.143 | random | 2.43 | yes | yes | flat:-0.325;upright:-0.179;random:-0.333 |
| DOL | -0.155 | -15.0 | -0.319 | -0.147 | flat | 2.68 | yes | yes | flat:-0.319;upright:-0.233;random:-0.242 |
| DMC | -0.083 | -8.0 | -0.297 | -0.198 | upright | 3.28 | yes | yes | flat:-0.260~;upright:-0.297;random:-0.265 |
| THF | -0.073 | -7.1 | -0.205 | -0.117 | upright | 3.08 | yes | yes | flat:-0.194;upright:-0.205;random:-0.157~ |
| DME | -0.062 | -6.0 | -0.214 | -0.135 | flat | 3.21 | yes | yes | flat:-0.214;upright:-0.205;random:-0.172~ |
| DEC | -0.057 | -5.5 | -0.249 | -0.177 | flat | 3.55 | yes | yes | flat:-0.249;upright:-0.243;random:-0.207 |
| EMC | -0.036 | -3.5 | -0.233 | -0.182 | upright | 3.30 | yes | yes | flat:-0.232;upright:-0.233;random:-0.209 |
| EA | -0.018 | -1.8 | -0.192 | -0.159 | random | 4.39 | no | yes | flat:-0.185;upright:-0.190;random:-0.192~ |

## Electrolyte solvents on graphite (0001)

Lattice constant 2.4600 A, cell 7x4x3, model uma-s-1p2p1/oc20, 160 minutes of relaxations (the sum over the molecules). E_ads < 0 is binding. A pose whose heavy atoms all end more than 4 A above the surface is unbound; 'intact' says whether every bond of the molecule survived the relaxation.

| Solvent | E_ads against the molecule far above, eV | kJ/mol | E_ads against the molecule alone, eV | Offset of the model, eV | Pose | Height of the lowest heavy atom, A | Bound | Intact | All poses against the molecule alone, eV |
|---|---:|---:|---:|---:|---|---:|---|---|---|
| EC | -0.052 | -5.0 | -0.214 | -0.218 | random | 3.65 | yes | yes | flat:-0.179~;upright:-0.180;random:-0.214 |
| PC | -0.048 | -4.6 | -0.211 | -0.214 | flat | 4.46 | no | yes | flat:-0.211~;upright:-0.201~;random:-0.195~ |
| DOL | -0.022 | -2.1 | -0.216 | -0.243 | random | 4.13 | no | yes | flat:-0.200~;upright:-0.212;random:-0.216~ |
| AN | -0.021 | -2.0 | -0.155 | -0.185 | upright | 3.86 | yes | yes | flat:-0.140~;upright:-0.155;random:-0.141 |
| DEC | -0.018 | -1.7 | -0.270 | -0.303 | flat | 4.73 | no | yes | flat:-0.270~;upright:-0.259~;random:-0.219 |
| THF | -0.017 | -1.6 | -0.228 | -0.264 | upright | 3.91 | yes | yes | flat:-0.212~;upright:-0.228;random:-0.222~ |
| MA | -0.016 | -1.5 | -0.155 | -0.196 | flat | 5.25 | no | yes | flat:-0.155~;upright:-0.150~;random:-0.144~ |
| EA | -0.013 | -1.3 | -0.201 | -0.239 | flat | 5.11 | no | yes | flat:-0.201~;upright:-0.188~;random:-0.192~ |
| DME | -0.008 | -0.8 | -0.284 | -0.325 | upright | 4.02 | no | yes | flat:-0.240~;upright:-0.284~;random:-0.279~ |
| EMC | -0.002 | -0.2 | -0.227 | -0.277 | random | 5.00 | no | yes | flat:-0.224~;upright:-0.193~;random:-0.227~ |
| DMC | -0.001 | -0.1 | -0.205 | -0.255 | upright | 4.33 | no | yes | flat:-0.164~;upright:-0.205~;random:-0.185~ |
