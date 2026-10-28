# Is the water worth a field of its own?

Plan item D9. The places a ligand's water occupies in a simulation, turned into a field of Gaussians at the hydration sites and mixed with the kernel of the five physico-chemical fields under one coefficient. Every difference is against the five fields alone, with the bootstrap interval of rule 4. Two controls carry the weight of the design: the same sites moved at random, and one ligand's sites given to every ligand.

## mcl1

25 ligands, 7.8 hydration sites each (from 4 to 12), chosen at an attenuation of 0.1. A site is a point occupied at least 50% of the frames, against 14% for bulk water; the strongest site of a ligand averages 73%.

Every model is given its own best attenuation over the same grid, so the reference is not held back by a choice made to suit the water block.

| model | q2 | attenuation | mixing coefficient | change against the five fields |
| --- | ---: | ---: | ---: | --- |
| the five fields alone | +0.575 | 0.1 | -- | the reference |
| the five fields and the water | +0.558 | 2 | 0.3 | -0.017 [-0.186, +0.198] |
| the water alone | +0.231 | 2 | 1.0 | -0.344 [-0.711, -0.012] |
| with sites moved at random (control) | +0.461 | 2 | 0.1 | -0.114 [-0.341, +0.114] |
| with one ligand's water for every ligand (control) | +0.575 | 0.1 | 0.9 | -0.000 [-0.000, +0.000] |
| random sites alone (control) | -0.034 | 0.3 | 1.0 | -0.609 [-0.892, -0.338] |
| one shared set of sites alone (control) | -0.138 | 0.1 | 1.0 | -0.713 [-0.960, -0.439] |

The mixing coefficient at an attenuation of 2, where the water block did best, from the fields alone to the water alone:

| coefficient | q2 | RMSEcv |
| ---: | ---: | ---: |
| 0.0 | +0.508 | 0.583 |
| 0.1 | +0.524 | 0.574 |
| 0.2 | +0.546 | 0.560 |
| 0.3 | +0.558 | 0.553 |
| 0.5 | +0.551 | 0.557 |
| 0.7 | +0.507 | 0.584 |
| 0.9 | +0.427 | 0.630 |
| 1.0 | +0.231 | 0.729 |

### Where the model asks for water

The sites of every ligand gathered into places, with the weight the water block puts on each. A positive weight asks for **more** water there and a negative one for less, which is where displacing a water is worth trying. Only places shared by at least three ligands are listed, and under rule 5 this is a display tool: the table above says the block does not improve the model, so a place here is something to look at and not a prediction.

| place (x, y, z) | ligands with a site there | mean occupancy | weight |
| --- | ---: | ---: | ---: |
| 68.6, -26.5, 20.1 | 5 | 56% | -0.129 |
| 66.7, -42.8, 36.3 | 5 | 54% | -0.111 |
| 74.4, -31.3, 18.7 | 6 | 52% | -0.103 |
| 61.1, -40.1, 38.2 | 10 | 54% | +0.089 |
| 71.5, -36.1, 25.5 | 8 | 55% | -0.083 |
| 59.9, -31.1, 37.5 | 10 | 60% | -0.078 |
| 58.2, -24.4, 34.3 | 3 | 61% | +0.037 |
| 55.8, -28.6, 31.7 | 21 | 68% | +0.033 |
| 56.3, -25.1, 28.1 | 18 | 64% | +0.028 |
| 74.5, -41.9, 27.4 | 17 | 56% | +0.025 |
| 69.9, -24.0, 21.0 | 23 | 62% | +0.017 |
| 72.5, -28.0, 26.1 | 23 | 63% | +0.015 |

## What it settles

**On mcl1 the water field adds nothing that can be measured.** The five fields reach +0.575 and the best mixture +0.558, a change of -0.017 [-0.186, +0.198], an interval that contains zero. On a set of 25 ligands an interval that wide cannot show a small gain, so what this says is that the water field is not worth a simulation here, not that the solvent carries nothing.

The water alone reaches +0.231. That is the number to read first: it says how much of the activity the places a ligand's water occupies can account for by themselves, with no field of the molecule in the kernel at all.

**The positions are carrying something, and that is the result worth keeping.** The water field on its own reaches +0.231, while the same number of sites moved to random places in the same box or one ligand's sites given to every ligand reach random sites alone -0.034 and one shared set of sites alone -0.138. Both controls are below zero, which is what a block that cannot tell the ligands apart must give, so the signal in the real maps is in **where the water sits** and not in how many sites a ligand has.

A note on reading the mixed controls: one ligand's sites given to every ligand make every kernel entry the same, and a constant block added to a kernel is absorbed by the fit's own bias. That control therefore lands exactly on the reference by construction, which is a check that the mixing does nothing of itself rather than a result about water.

**What this rests on.** One set of 25 ligands, 2 ns of simulation each, and a definition of a hydration site -- a point occupied at least half the frames, with sites no closer than one water across. A longer simulation would place the sites more surely; it would not make 25 ligands a larger set, and the interval above is set by the 25 and not by the nanoseconds.

