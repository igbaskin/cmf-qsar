# Is the water worth a field of its own?

Plan item D9. The places a ligand's water occupies in a simulation, turned into a field of Gaussians at the hydration sites and mixed with the kernel of the five physico-chemical fields under one coefficient. Every difference is against the five fields alone, with the bootstrap interval of rule 4. Two controls carry the weight of the design: the same sites moved at random, and one ligand's sites given to every ligand.

## mcl1

25 ligands, 7.8 hydration sites each (from 4 to 12), chosen at an attenuation of 0.1. A site is a point occupied at least 50% of the frames, against 14% for bulk water; the strongest site of a ligand averages 73%.

Every model is given its own best attenuation over the same grid, so the reference is not held back by a choice made to suit the water block.

| model | q2 | attenuation | mixing coefficient | change against the five fields |
| --- | ---: | ---: | ---: | --- |
| the five fields alone | +0.575 | 0.1 | -- | the reference |
| the five fields and the water | +0.560 | 2 | 0.3 | -0.015 [-0.178, +0.186] |
| the water alone | +0.263 | 2 | 1.0 | -0.312 [-0.744, +0.031] |
| with sites moved at random (control) | +0.458 | 2 | 0.1 | -0.117 [-0.354, +0.112] |
| with one ligand's water for every ligand (control) | +0.575 | 0.1 | 0.9 | -0.000 [-0.000, +0.000] |
| random sites alone (control) | -0.028 | 0.3 | 1.0 | -0.603 [-0.880, -0.330] |
| one shared set of sites alone (control) | -0.138 | 2 | 1.0 | -0.713 [-0.960, -0.439] |

The mixing coefficient at an attenuation of 2, where the water block did best, from the fields alone to the water alone:

| coefficient | q2 | RMSEcv |
| ---: | ---: | ---: |
| 0.0 | +0.508 | 0.583 |
| 0.1 | +0.527 | 0.572 |
| 0.2 | +0.550 | 0.558 |
| 0.3 | +0.560 | 0.552 |
| 0.5 | +0.552 | 0.557 |
| 0.7 | +0.499 | 0.589 |
| 0.9 | +0.405 | 0.642 |
| 1.0 | +0.263 | 0.714 |

### Where the model asks for water

The sites of every ligand gathered into places, with the weight the water block puts on each. A positive weight asks for **more** water there and a negative one for less, which is where displacing a water is worth trying. Only places shared by at least three ligands are listed, and under rule 5 this is a display tool: the table above says the block does not improve the model, so a place here is something to look at and not a prediction.

| place (x, y, z) | ligands with a site there | mean occupancy | weight |
| --- | ---: | ---: | ---: |
| 68.6, -26.5, 20.1 | 5 | 56% | -0.453 |
| 66.7, -42.8, 36.3 | 5 | 54% | -0.346 |
| 74.4, -31.3, 18.7 | 6 | 52% | -0.297 |
| 59.9, -31.1, 37.5 | 10 | 60% | -0.247 |
| 61.1, -40.1, 38.2 | 10 | 54% | +0.224 |
| 71.5, -36.1, 25.5 | 8 | 55% | -0.151 |
| 56.3, -25.1, 28.1 | 18 | 64% | +0.150 |
| 58.2, -24.4, 34.3 | 3 | 61% | +0.135 |
| 55.8, -28.6, 31.7 | 21 | 68% | +0.132 |
| 74.5, -41.9, 27.4 | 17 | 56% | +0.093 |
| 72.5, -28.0, 26.1 | 23 | 63% | +0.050 |
| 69.9, -24.0, 21.0 | 23 | 62% | +0.033 |

### How much the definition of a site decides

Two numbers decide what a hydration site is, and neither follows from anything: the share of frames a point has to be occupied, and how close two sites may be. Here the whole comparison is run again at every pair of them, each time with the attenuation and the mixing coefficient chosen as before. The reference is the five fields at their best, +0.575.

| occupancy | spacing | sites per ligand | with water | change against the fields | water alone | random sites alone |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 30% | 1.8 A | 39.2 | +0.579 | +0.004 [-0.158, +0.212] | +0.284 | -0.067 |
| 30% | 2.4 A | 37.5 | +0.570 | -0.005 [-0.177, +0.197] | +0.312 | +0.047 |
| 30% | 3.0 A | 31.5 | +0.647 | +0.072 [-0.051, +0.239] | +0.371 | -0.139 |
| 30% | 3.6 A | 24.8 | +0.542 | -0.033 [-0.222, +0.114] | +0.090 | +0.438 |
| 40% | 1.8 A | 17.6 | +0.500 | -0.075 [-0.293, +0.158] | +0.123 | -0.038 |
| 40% | 2.4 A | 17.4 | +0.493 | -0.082 [-0.297, +0.151] | +0.130 | -0.081 |
| 40% | 3.0 A | 15.8 | +0.453 | -0.122 [-0.332, +0.091] | +0.093 | +0.004 |
| 40% | 3.6 A | 14.2 | +0.445 | -0.130 [-0.352, +0.089] | -0.117 | -0.121 |
| 50% | 1.8 A | 7.8 | +0.564 | -0.011 [-0.173, +0.191] | +0.263 | -0.049 |
| 50% | 2.4 A | 7.8 | +0.560 | -0.015 [-0.178, +0.186] | +0.263 | -0.028 |
| 50% | 3.0 A | 7.3 | +0.529 | -0.046 [-0.216, +0.164] | +0.129 | -0.085 |
| 50% | 3.6 A | 7.2 | +0.538 | -0.037 [-0.206, +0.172] | +0.131 | -0.142 |
| 60% | 1.8 A | 3.6 | +0.617 | +0.042 [-0.190, +0.299] | +0.554 | -0.096 |
| 60% | 2.4 A | 3.6 | +0.617 | +0.042 [-0.190, +0.299] | +0.554 | -0.096 |
| 60% | 3.0 A | 3.6 | +0.640 | +0.065 [-0.141, +0.320] | +0.557 | -0.055 |
| 60% | 3.6 A | 3.6 | +0.662 | +0.087 [-0.108, +0.347] | +0.607 | -0.082 |
| 70% | 1.8 A | -- | -- | 7 ligand(s) with no site | -- | -- |
| 70% | 2.4 A | -- | -- | 7 ligand(s) with no site | -- | -- |
| 70% | 3.0 A | -- | -- | 7 ligand(s) with no site | -- | -- |
| 70% | 3.6 A | -- | -- | 7 ligand(s) with no site | -- | -- |

**The first conclusion does not depend on the definition.** Of the 16 definitions that give every ligand a site, none beats the five fields beyond the interval and none falls behind it. The change against the reference runs from -0.130 to +0.087, and every interval contains zero.

**The second conclusion does depend on it, and this is the finding of the sweep.** The water field on its own runs from -0.117 to +0.607 across the 16 definitions, and is not positive at 1 of them. What was reported for the default definition alone was therefore one point of a range, not a property of the water.

**At 1 of the 16 definitions the random-site control does as well as the real maps or better**, the worst being an occupancy of 30% and a spacing of 3.6 A, where the control reaches +0.438 against +0.090 for the measured sites. A loose definition puts tens of weakly occupied points in the field, and a block of that many plausible numbers carries whatever a block of that many random numbers carries. Any claim about where the water sits has to be made at a definition where the control is beaten, and the table says which those are.

**The sensitivity has a direction: the stronger the sites, the more they carry.** Averaged over the spacings, the water alone goes 30% +0.264, 40% +0.057, 50% +0.197, 60% +0.568. A stricter threshold keeps fewer sites and they carry more, which says the signal is in the few places a water really sits and that adding weakly occupied points dilutes it.

The strongest the water alone gets is +0.607, at an occupancy threshold of 60% and a spacing of 3.6 A, with 3.6 sites per ligand. **That number should not be quoted as this item's result.** It is the largest of 16 tried on 25 molecules, chosen after seeing them, and the honest estimate of what a strict definition is worth would need a set this one was not used to choose on. The defaults of the module, 50% and 2.4 A, were fixed before any of this was measured and are left alone for that reason.

One thing the sweep does settle about the strict end: at the strictest threshold that still leaves every ligand a site (60%, 3.6 sites per ligand), the water alone reaches +0.554 to +0.607 against a control of -0.096 to -0.055, and it does so at all 4 spacings rather than at one. A threshold above that leaves ligands with no site at all, which is the end of the range.

## What it settles

**On mcl1 the water field adds nothing that can be measured.** The five fields reach +0.575 and the best mixture +0.560, a change of -0.015 [-0.178, +0.186], an interval that contains zero. On a set of 25 ligands an interval that wide cannot show a small gain, so what this says is that the water field is not worth a simulation here, not that the solvent carries nothing.

The water alone reaches +0.263. That is the number to read first: it says how much of the activity the places a ligand's water occupies can account for by themselves, with no field of the molecule in the kernel at all.

**The positions are carrying something, and that is the result worth keeping.** The water field on its own reaches +0.263, while the same number of sites moved to random places in the same box or one ligand's sites given to every ligand reach random sites alone -0.028 and one shared set of sites alone -0.138. Both controls are below zero, which is what a block that cannot tell the ligands apart must give, so the signal in the real maps is in **where the water sits** and not in how many sites a ligand has.

A note on reading the mixed controls: one ligand's sites given to every ligand make every kernel entry the same, and a constant block added to a kernel is absorbed by the fit's own bias. That control therefore lands exactly on the reference by construction, which is a check that the mixing does nothing of itself rather than a result about water.

**What this rests on.** One set of 25 ligands, 2 ns of simulation each, and a definition of a hydration site -- a point occupied at least half the frames, with sites no closer than one water across. A longer simulation would place the sites more surely; it would not make 25 ligands a larger set, and the interval above is set by the 25 and not by the nanoseconds.

