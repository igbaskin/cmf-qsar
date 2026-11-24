# The continuous reaction field, stage 1: the construction (plan item PO3)

2000 reactions of the Grambow set (wB97X-D3/def2-TZVP), drawn with seed 0. Fields: electrostatic, steric, hydrophobic, hba, hbd; attenuation 0.5 per square angstrom. No model is fitted at this stage.

## Superposition of the product on the reactant

| Atoms the fit uses | Reactions built | RMSD over the fitted atoms, median (A) | 90 % below (A) | Displacement of all atoms, median (A) | 90 % below (A) |
|---|---|---|---|---|---|
| all heavy atoms | 2000 | 0.712 | 1.254 | 1.183 | 1.762 |
| atoms of the changed bonds | 2000 | 1.094 | 1.695 | 1.565 | 2.568 |

## The reactions

Atoms per reaction: median 14, largest 23. Changed bonds per reaction: median 4, largest 10; 0 reactions have none. The count reference has 28 columns.

| Bond change | Number in the sample |
|---|---|
| C-H broken | 1278 |
| C-C broken | 1145 |
| C-C raised | 1019 |
| C-H made | 669 |
| C-N lowered | 534 |
| C-N raised | 476 |
| C-C lowered | 473 |
| C-O raised | 453 |
| C-O broken | 417 |
| C-O lowered | 372 |
| C-C made | 344 |
| H-O made | 289 |

## Share of the two sides that is left in the dynamic channel

The self-overlap of the dynamic channel divided by the sum of the self-overlaps of the two sides. Zero means that nothing changes. 1 means that the two sides do not overlap at all. A value above 1 means that the two sides overlap with opposite signs, which a field with signed weights allows.

| Field | Median | 90 % below | Largest |
|---|---|---|---|
| electrostatic | 0.386 | 0.882 | 1.588 |
| steric | 0.003 | 0.013 | 0.077 |
| hydrophobic | 0.254 | 0.892 | 1.837 |
| hba | 0.093 | 0.444 | 1.000 |
| hbd | 0.151 | 1.000 | 1.000 |

## The two identities, and the cost

| Quantity | Value |
|---|---|
| Identity (1), overlap kernel: largest deviation, relative to the largest entry | 6.6e-16 |
| Identity (2), overlap kernel, 200 reversed reactions: largest deviation | 2.9e-16 |
| Identity (1), invariant rows: largest deviation | 7.7e-16 |
| The odd invariant row is the same under both superposition rules | yes |
| Correlation of the dynamic invariant rows between the two rules | 0.698 |
| Overlap kernels of five parts, time (s) | 291 |
| Invariant rows of five parts, time (s) | 10 |
| Invariant columns per part | 255 |
| Largest memory held by the process (GB) | 1.52 |
| Full set of 11961 reactions: one part, all fields, in memory (GB) | 5.3 |
| Full set: overlap kernels of five parts, estimated time (h) | 2.9 |
| Full set: invariant rows, estimated time (min) | 1 |
