# One attenuation per field, or one shared by all of them

8 Sutherland sets, 5 random 70/30 splits each. On the training part the shared grid is searched by the evidence; the per-field coordinate descent then starts from the width it chose and is scored by the evidence too. Both choices are fitted on that part and asked about the held-out molecules.

| set | n | R²pred shared | R²pred per-field | difference | log evidence gained | fields moved |
|---|---:|---:|---:|---:|---:|---:|
| DHFR | 361 | +0.670 | +0.626 | -0.044 | +4.4 | 3.0 of 5 |
| COX2 | 282 | +0.508 | +0.492 | -0.016 | +3.0 | 3.2 of 5 |
| BZR | 147 | +0.343 | +0.382 | +0.039 | +2.8 | 4.8 of 5 |
| ACE | 114 | +0.529 | +0.525 | -0.004 | +1.8 | 3.8 of 5 |
| AChE | 111 | +0.534 | +0.548 | +0.014 | +3.3 | 3.2 of 5 |
| THR | 88 | +0.540 | +0.560 | +0.020 | +3.9 | 4.6 of 5 |
| THER | 76 | +0.364 | +0.387 | +0.023 | +0.6 | 3.0 of 5 |
| GPB | 66 | +0.388 | +0.352 | -0.036 | +0.8 | 2.0 of 5 |
| **mean** |  |  |  | **-0.001** | +2.6 | 3.5 of 5 |

Paired over the sets, the per-field widths are -0.001 in R²pred against the shared one, 95 % interval [-0.020, +0.018], better on 4 of 8 sets -- while gaining +2.6 in log evidence, which is the criterion that chose them. The descent moved 3.5 of the 5 fields on average and left them all at the shared width in 0 % of the splits.

Which width is chosen for each field, averaged over every split (the grid runs from 0.2 to 4; larger is sharper):

| field | mean alpha chosen |
|---|---:|
| electrostatic | 1.50 |
| steric | 2.33 |
| hydrophobic | 1.59 |
| hba | 2.09 |
| hbd | 1.16 |

Per split rather than per set, which is the finer count: 40 splits, the per-field widths ahead on 20 of them, mean -0.0005 in R²pred with a 95 % interval of [-0.014, +0.013]. The correlation between what a split gained in evidence and what it gained in R²pred is -0.07: the criterion that chose the widths does not know whether they helped.

The reading is that the freedom is real and worthless. The descent moves most of the widths and improves the evidence on every split, the widths it settles on are chemically sensible -- the steric field sharpest, the donor field widest -- and the molecules held out cannot tell the difference. That is the same flat surface item B4 found when the two selection criteria disagreed about the shared width and predicted equally well: around its optimum the attenuation simply does not decide much, and five of them decide no more than one. Use --alpha-search per-field to ask the question of a new series, not as a default.
