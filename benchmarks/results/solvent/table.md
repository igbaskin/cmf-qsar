# Is the water worth a field of its own?

Plan item D9. The places a ligand's water occupies in a simulation, turned into a field of Gaussians at the hydration sites and mixed with the kernel of the five physico-chemical fields under one coefficient. Every difference is against the five fields alone, with the bootstrap interval of rule 4. Two controls carry the weight of the design: the same sites moved at random, and one ligand's sites given to every ligand.

## hif2a

36 ligands, 5.0 hydration sites each (from 5 to 5), chosen at an attenuation of 0.3. A site is a point occupied at least 30% of the frames, against 14% for bulk water; the strongest site of a ligand averages 83%.

Every model is given its own best attenuation over the same grid, so the reference is not held back by a choice made to suit the water block.

| model | q2 | attenuation | mixing coefficient | change against the five fields |
| --- | ---: | ---: | ---: | --- |
| the five fields alone | +0.342 | 0.3 | -- | the reference |
| the five fields and the water | +0.262 | 0.3 | 0.1 | -0.079 [-0.287, -0.007] |
| the water alone | +0.059 | 2 | 1.0 | -0.283 [-0.685, +0.086] |
| with sites moved at random (control) | +0.470 | 0.3 | 0.1 | +0.128 [+0.038, +0.339] |
| with one ligand's water for every ligand (control) | +0.345 | 0.1 | 0.7 | +0.003 [-0.270, +0.156] |
| random sites alone (control) | +0.177 | 0.3 | 1.0 | -0.165 [-0.435, +0.310] |
| one shared set of sites alone (control) | -0.078 | 0.1 | 1.0 | -0.420 [-0.671, +0.207] |

The mixing coefficient at an attenuation of 0.3, where the water block did best, from the fields alone to the water alone:

| coefficient | q2 | RMSEcv |
| ---: | ---: | ---: |
| 0.0 | +0.342 | 0.590 |
| 0.1 | +0.262 | 0.624 |
| 0.2 | +0.196 | 0.652 |
| 0.3 | +0.175 | 0.660 |
| 0.5 | +0.144 | 0.673 |
| 0.7 | +0.101 | 0.689 |
| 0.9 | +0.036 | 0.714 |
| 1.0 | -0.018 | 0.734 |

### Where the model asks for water

The sites of every ligand gathered into places, with the weight the water block puts on each. A positive weight asks for **more** water there and a negative one for less, which is where displacing a water is worth trying. Only places shared by at least three ligands are listed, and under rule 5 this is a display tool: the table above says the block does not improve the model, so a place here is something to look at and not a prediction.

| place (x, y, z) | ligands with a site there | mean occupancy | weight |
| --- | ---: | ---: | ---: |
| 30.9, 0.4, -2.5 | 12 | 66% | -0.513 |
| 29.6, 7.1, -15.1 | 3 | 60% | +0.312 |
| 26.4, 4.7, -9.8 | 13 | 73% | +0.225 |
| 33.5, -4.2, -2.5 | 7 | 57% | +0.068 |
| 19.6, 8.6, -19.0 | 6 | 64% | +0.049 |
| 11.5, 4.2, -18.0 | 19 | 63% | +0.041 |
| 23.8, -8.6, -4.5 | 36 | 69% | +0.040 |
| 24.1, 9.1, -15.8 | 6 | 60% | -0.040 |
| 14.8, -9.6, -20.1 | 35 | 68% | +0.020 |
| 28.8, -8.8, -11.3 | 34 | 81% | +0.011 |

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

**Whether the water field improves the model does not depend on the definition: it does not, at any of them.** Of the 16 definitions that give every ligand a site, none beats the five fields beyond the interval and none falls behind it. The change against the reference runs from -0.130 to +0.087, and every interval contains zero.

**The second conclusion does depend on it, and this is the finding of the sweep.** The water field on its own runs from -0.117 to +0.607 across the 16 definitions, and is not positive at 1 of them. What was reported for the default definition alone was therefore one point of a range, not a property of the water.

**At 1 of the 16 definitions the random-site control does as well as the real maps or better**, the worst being an occupancy of 30% and a spacing of 3.6 A, where the control reaches +0.438 against +0.090 for the measured sites. A loose definition puts tens of weakly occupied points in the field, and a block of that many plausible numbers carries whatever a block of that many random numbers carries. Any claim about where the water sits has to be made at a definition where the control is beaten, and the table says which those are.

**The sensitivity has a direction: the stronger the sites, the more they carry.** Averaged over the spacings, the water alone goes 30% +0.264, 40% +0.057, 50% +0.197, 60% +0.568. A stricter threshold keeps fewer sites and they carry more, which says the signal is in the few places a water really sits and that adding weakly occupied points dilutes it.

The strongest the water alone gets is +0.607, at an occupancy threshold of 60% and a spacing of 3.6 A, with 3.6 sites per ligand. **That number should not be quoted as this item's result.** It is the largest of 16 tried on 25 molecules, chosen after seeing them, and an honest estimate of what a strict definition is worth needs a set it was not chosen on. The defaults of the module, 50% and 2.4 A, were fixed before any of this was measured and are left as they are for that reason.

One thing the sweep does settle about the strict end: at the strictest threshold that still leaves every ligand a site (60%, 3.6 sites per ligand), the water alone reaches +0.554 to +0.607 against a control of -0.096 to -0.055, and it does so at all 4 spacings rather than at one. A threshold above that leaves ligands with no site at all, which is the end of the range.

**What the threshold is standing in for is the number of sites.** The water alone clearly carries something at 9 of the 16 definitions, and all of them are at an occupancy of 30%, 50%, 60%, leaving 3.6 to 39.2 sites per ligand. Every other definition leaves 7.2 to 24.8 and gives at best +0.131, with 6 of them marginal rather than flat. A threshold is a proxy for how few sites survive, and the proxy is target-specific: the same share of frames leaves a different number of sites on a site that holds much ordered water than on one that holds little. The count is the quantity to compare across targets, not the threshold.

## p38

29 ligands, 11.4 hydration sites each (from 6 to 20), chosen at an attenuation of 0.1. A site is a point occupied at least 60% of the frames, against 14% for bulk water; the strongest site of a ligand averages 90%.

Every model is given its own best attenuation over the same grid, so the reference is not held back by a choice made to suit the water block.

| model | q2 | attenuation | mixing coefficient | change against the five fields |
| --- | ---: | ---: | ---: | --- |
| the five fields alone | +0.261 | 0.1 | -- | the reference |
| the five fields and the water | +0.424 | 0.5 | 0.5 | +0.164 [-0.080, +0.473] |
| the water alone | -0.043 | 0.1 | 1.0 | -0.303 [-0.633, +0.096] |
| with sites moved at random (control) | +0.361 | 0.1 | 0.1 | +0.100 [-0.031, +0.291] |
| with one ligand's water for every ligand (control) | +0.281 | 0.1 | 0.7 | +0.020 [-0.143, +0.122] |
| random sites alone (control) | +0.002 | 0.1 | 1.0 | -0.259 [-0.580, +0.147] |
| one shared set of sites alone (control) | -0.051 | 0.1 | 1.0 | -0.312 [-0.578, +0.065] |

The mixing coefficient at an attenuation of 0.5, where the water block did best, from the fields alone to the water alone:

| coefficient | q2 | RMSEcv |
| ---: | ---: | ---: |
| 0.0 | +0.201 | 0.623 |
| 0.1 | +0.258 | 0.601 |
| 0.2 | +0.329 | 0.571 |
| 0.3 | +0.375 | 0.551 |
| 0.5 | +0.424 | 0.529 |
| 0.7 | +0.405 | 0.538 |
| 0.9 | +0.211 | 0.620 |
| 1.0 | -0.050 | 0.715 |

### Where the model asks for water

The sites of every ligand gathered into places, with the weight the water block puts on each. A positive weight asks for **more** water there and a negative one for less, which is where displacing a water is worth trying. Only places shared by at least three ligands are listed, and under rule 5 this is a display tool: the table above says the block does not improve the model, so a place here is something to look at and not a prediction.

| place (x, y, z) | ligands with a site there | mean occupancy | weight |
| --- | ---: | ---: | ---: |
| 30.5, 13.6, 26.2 | 6 | 69% | -0.026 |
| 21.8, 0.5, 37.6 | 7 | 64% | -0.018 |
| 23.1, 2.8, 22.6 | 12 | 74% | -0.018 |
| 22.7, 13.1, 41.0 | 8 | 68% | +0.017 |
| 22.1, 1.3, 39.6 | 13 | 63% | +0.017 |
| 8.4, 6.4, 24.9 | 9 | 68% | +0.016 |
| 19.3, -0.2, 30.1 | 9 | 67% | +0.013 |
| 20.1, 20.7, 24.7 | 6 | 78% | +0.013 |
| 26.5, 3.4, 33.2 | 5 | 64% | -0.011 |
| 17.2, 18.0, 40.0 | 4 | 73% | +0.010 |
| 7.9, 1.8, 31.8 | 17 | 71% | +0.010 |
| 30.3, 19.4, 25.9 | 6 | 64% | -0.008 |

### How much the definition of a site decides

Two numbers decide what a hydration site is, and neither follows from anything: the share of frames a point has to be occupied, and how close two sites may be. Here the whole comparison is run again at every pair of them, each time with the attenuation and the mixing coefficient chosen as before. The reference is the five fields at their best, +0.261.

| occupancy | spacing | sites per ligand | with water | change against the fields | water alone | random sites alone |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 30% | 1.8 A | 40.0 | +0.206 | -0.054 [-0.083, -0.031] | -0.058 | +0.060 |
| 30% | 2.4 A | 40.0 | +0.196 | -0.065 [-0.096, -0.037] | -0.060 | +0.043 |
| 30% | 3.0 A | 40.0 | +0.261 | +0.000 [-0.031, +0.042] | +0.053 | +0.017 |
| 30% | 3.6 A | 39.4 | +0.242 | -0.018 [-0.041, +0.020] | +0.092 | -0.011 |
| 40% | 1.8 A | 34.4 | +0.225 | -0.035 [-0.084, +0.006] | +0.001 | -0.048 |
| 40% | 2.4 A | 33.9 | +0.225 | -0.036 [-0.085, +0.006] | +0.015 | -0.044 |
| 40% | 3.0 A | 30.1 | +0.289 | +0.028 [-0.221, +0.184] | -0.037 | +0.113 |
| 40% | 3.6 A | 24.4 | +0.319 | +0.059 [-0.183, +0.222] | -0.049 | +0.076 |
| 50% | 1.8 A | 21.3 | +0.194 | -0.066 [-0.314, +0.176] | -0.067 | +0.005 |
| 50% | 2.4 A | 21.1 | +0.203 | -0.058 [-0.353, +0.221] | -0.067 | +0.017 |
| 50% | 3.0 A | 18.9 | +0.174 | -0.087 [-0.176, +0.000] | -0.068 | -0.029 |
| 50% | 3.6 A | 16.2 | +0.201 | -0.059 [-0.150, +0.044] | -0.068 | -0.051 |
| 60% | 1.8 A | 11.5 | +0.425 | +0.165 [-0.079, +0.475] | -0.043 | -0.022 |
| 60% | 2.4 A | 11.4 | +0.424 | +0.164 [-0.080, +0.473] | -0.043 | +0.002 |
| 60% | 3.0 A | 10.8 | +0.405 | +0.144 [-0.101, +0.456] | -0.055 | -0.057 |
| 60% | 3.6 A | 9.8 | +0.360 | +0.099 [-0.123, +0.322] | -0.054 | +0.014 |
| 70% | 1.8 A | 6.7 | +0.536 | +0.276 [+0.037, +0.560] | +0.312 | -0.032 |
| 70% | 2.4 A | 6.7 | +0.536 | +0.276 [+0.037, +0.560] | +0.312 | -0.032 |
| 70% | 3.0 A | 6.4 | +0.559 | +0.298 [+0.063, +0.588] | +0.340 | -0.014 |
| 70% | 3.6 A | 6.3 | +0.560 | +0.299 [+0.063, +0.593] | +0.341 | -0.010 |

**Whether the water field improves the model does depend on the definition.** Of the 20 definitions that give every ligand a site, **4 beat the five fields beyond the interval** and 2 fall behind it. The change against the reference runs from -0.087 to +0.299. The strongest is an occupancy of 70% and a spacing of 3.6 A, +0.299 [+0.063, +0.593], with 6.3 sites per ligand. Those definitions are the strict end of the sweep, and they are the only place in this item where a water block has cleared the interval.

**The second conclusion does depend on it, and this is the finding of the sweep.** The water field on its own runs from -0.068 to +0.341 across the 20 definitions, and is not positive at 12 of them. What was reported for the default definition alone was therefore one point of a range, not a property of the water.

**At 11 of the 20 definitions the random-site control does as well as the real maps or better**, the worst being an occupancy of 40% and a spacing of 3.0 A, where the control reaches +0.113 against -0.037 for the measured sites. A loose definition puts tens of weakly occupied points in the field, and a block of that many plausible numbers carries whatever a block of that many random numbers carries. Any claim about where the water sits has to be made at a definition where the control is beaten, and the table says which those are.

**The sensitivity has a direction: the stronger the sites, the more they carry.** Averaged over the spacings, the water alone goes 30% +0.007, 40% -0.017, 50% -0.067, 60% -0.049, 70% +0.326. A stricter threshold keeps fewer sites and they carry more, which says the signal is in the few places a water really sits and that adding weakly occupied points dilutes it.

The strongest the water alone gets is +0.341, at an occupancy threshold of 70% and a spacing of 3.6 A, with 6.3 sites per ligand. **That number should not be quoted as this item's result.** It is the largest of 20 tried on 29 molecules, chosen after seeing them, and an honest estimate of what a strict definition is worth needs a set it was not chosen on. The defaults of the module, 50% and 2.4 A, were fixed before any of this was measured and are left as they are for that reason.

One thing the sweep does settle about the strict end: at the strictest threshold that still leaves every ligand a site (70%, 6.7 sites per ligand), the water alone reaches +0.312 to +0.341 against a control of -0.032 to -0.010, and it does so at all 4 spacings rather than at one. A threshold above that leaves ligands with no site at all, which is the end of the range.

**What the threshold is standing in for is the number of sites.** The water alone clearly carries something at 4 of the 20 definitions, and all of them are at an occupancy of 70%, leaving 6.3 to 6.7 sites per ligand. Every other definition leaves 9.8 to 40.0 and gives at best +0.092, with 2 of them marginal rather than flat. A threshold is a proxy for how few sites survive, and the proxy is target-specific: the same share of frames leaves a different number of sites on a site that holds much ordered water than on one that holds little. The count is the quantity to compare across targets, not the threshold.

## The definition given as a count of sites

The sweep above left a hypothesis and not a result: what appeared to decide whether the water field carries anything was **how many strongly occupied sites survive**, not the occupancy threshold, because the same threshold leaves a different number of sites on two targets. A threshold is therefore a target-specific proxy for a count, and a count is what can mean the same thing on two sets. This section fixes it: the **5 most occupied sites a ligand**, an occupancy floor of 30% so that the count and not the floor decides, and the module's own spacing of 2.4 A. Everything else is unchanged.

**hif2a** is the set the count was not chosen on. Its definition was written down before the simulations were started, so its row is a test that could fail. The other rows are the sets the hypothesis was read from, which cannot confirm it -- but they can still refute it, by giving less at the fixed count than they gave at their own threshold.

| set | ligands | sites a ligand | the five fields | with the water | change against the five fields | the water alone | random sites alone | one shared set alone |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| **hif2a** | 36 | 5.0 | +0.342 | +0.262 | -0.079 [-0.287, -0.007] | +0.058 | +0.177 | -0.078 |
| mcl1 | 25 | 5.0 | +0.575 | +0.455 | -0.120 [-0.316, +0.074] | +0.172 | -0.134 | -0.138 |
| p38 | 29 | 5.0 | +0.261 | +0.267 | +0.006 [-0.100, +0.133] | -0.043 | +0.089 | -0.051 |

**The water block improves no set beyond the interval** at this definition. The change runs from -0.120 to +0.006, and on 1 of the 3 sets (hif2a) the water block makes the model **worse** beyond the interval, which no definition tried before this one had done. A fixed count does not make the water field worth adding.

The water field **on its own** carries something on 1 of the 3 sets (mcl1): mcl1 +0.172 against -0.134 for the same number of sites moved at random.

**What the held-out set says.** On hif2a the water alone reaches +0.058 against +0.177 for random sites, and the mixture -0.079 [-0.287, -0.007] against the five fields. The hypothesis predicted that a count of 5 would carry something here, and it does not. The count-based statement is therefore **not supported** on the one set that could test it.

**The count does not reproduce what the threshold gave, on 2 of the 3 sets**: mcl1 +0.172 at 5.0 sites against +0.607 at 3.6 sites (60% occupancy); p38 -0.043 at 5.0 sites against +0.341 at 6.3 sites (70% occupancy). That is the clearest result here. A fixed count was put forward as the definition the two sets agreed on, and it gives less on both of the sets it was read from, so the reformulation was a mistake in reading the sweep rather than a property that failed to transfer. What the working definitions have in common is not how many sites survive but that **every site kept is strongly occupied**; the number of survivors is a consequence of that floor and differs between targets. Fixing the count forces a ligand with one strongly held water to take four weak ones as well, which is why a count behaves like a loose threshold.

> **What is fixed and what is not.** The count, the floor and the spacing were written down before the held-out set was simulated, so its row is an estimate and not the best of several. Why one target needs a floor of 60% and another 70% is still unexplained, and a floor chosen per target is not a definition anyone can fix in advance. The honest reading is in the verdict below.

## The rule that was meant to pick both working definitions

A count was the wrong reading of the sweep. What was left of it was a **rule** rather than a number: the definitions where the water alone carried anything looked like the **strictest occupancy floor that still leaves every ligand at least one site**. A rule of that form can be carried to a new set, which a number cannot. It is applied here as a rule: the floor is chosen from the occupancy maps alone, reading no activity and no q2, and the five models are then measured once at that floor. Nothing is searched over after the choice.

**hif2a** is again the set nothing was read from, so its row is an estimate. The other rows are the sets the rule was read from: they can only show whether the rule picks the definition it was meant to pick, which is a check on the rule and not evidence for it.

| set | ligands | floor the rule picks | sites a ligand | the five fields | with the water | change against the five fields | the water alone | random sites alone |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| **hif2a** | 36 | 60% | 4.6 | +0.342 | +0.396 | +0.054 [-0.216, +0.330] | -0.074 | +0.277 |
| mcl1 | 25 | 60% | 3.6 | +0.575 | +0.617 | +0.042 [-0.190, +0.299] | +0.554 | -0.096 |
| p38 | 29 | 80% | 3.0 | +0.261 | +0.301 | +0.040 [-0.176, +0.220] | +0.169 | +0.100 |

**The rule does not pick the definition it was read from, on 1 of the 3 sets**: on p38 it picks 80% and the water alone reaches +0.169 there, while the sweep's best was +0.341 at 70%. That refutes the rule before any held-out set is consulted, and it shows where the rule came from: the sweep's grid stopped at 70%, so a floor that was the strictest **tried** was taken for the strictest **possible**. Carried out properly, the rule keeps raising the floor while every ligand still has a site, and it goes past the definition the sweep had found.

**The water alone clearly beats its own random-site control on 1 of the 3 sets** (mcl1), by +0.650. On one further set (p38) it is ahead by less than 0.20, which on sets of this size is not a difference worth reading: p38 +0.169 against +0.100.

The water block improves no set beyond the interval even here: the change runs from +0.040 to +0.054, and every interval contains zero.

**What the held-out set says.** The rule picks a floor of 60% on hif2a, which leaves 4.6 sites a ligand. There the water alone reaches -0.074 against +0.277 for random sites, and the mixture +0.054 [-0.216, +0.330] against the five fields. The rule does not hold on the set it was not read from: at the definition it chooses, the water field carries nothing that random sites do not. Both readings of the sweep have now failed a test on a set they were not read from, and what the sweep found on two sets is best treated as a property of those two sets.

> **What this does not settle.** A floor chosen by this rule still depends on the set, so two targets get different definitions and the rule gives no number anyone can quote. The rule itself was read from two sets and is tested here on one. And whether the water block is worth adding to a model is answered by the change column, not by the water alone: that column is where rule 1 applies.

## What it settles

**On hif2a the water field makes the model worse.** -0.079 [-0.287, -0.007] against the five fields, which clears the interval in the wrong direction.

The water alone reaches +0.059. That is the number to read first: it says how much of the activity the places a ligand's water occupies can account for by themselves, with no field of the molecule in the kernel at all.

**The positions are carrying nothing.** The water field on its own reaches +0.059 and the controls reach random sites alone +0.177 and one shared set of sites alone -0.078, so a block of plausible numbers does as well as the measured maps and the places the water occupies are not what the model is using.

**On mcl1 the water field adds nothing that can be measured.** The five fields reach +0.575 and the best mixture +0.560, a change of -0.015 [-0.178, +0.186], an interval that contains zero. On a set of 25 ligands an interval that wide cannot show a small gain, so what this says is that the water field is not worth a simulation here, not that the solvent carries nothing.

The water alone reaches +0.263. That is the number to read first: it says how much of the activity the places a ligand's water occupies can account for by themselves, with no field of the molecule in the kernel at all.

**The positions are carrying something, and that is the result worth keeping.** The water field on its own reaches +0.263, while the same number of sites moved to random places in the same box or one ligand's sites given to every ligand reach random sites alone -0.028 and one shared set of sites alone -0.138. Both controls are below zero, which is what a block that cannot tell the ligands apart must give, so the signal in the real maps is in **where the water sits** and not in how many sites a ligand has.

A note on reading the mixed controls: one ligand's sites given to every ligand make every kernel entry the same, and a constant block added to a kernel is absorbed by the fit's own bias. That control therefore lands exactly on the reference by construction, which is a check that the mixing does nothing of itself rather than a result about water.

**On p38 the water field adds nothing that can be measured.** The five fields reach +0.261 and the best mixture +0.424, a change of +0.164 [-0.080, +0.473], an interval that contains zero. On a set of 29 ligands an interval that wide cannot show a small gain, so what this says is that the water field is not worth a simulation here, not that the solvent carries nothing.

The water alone reaches -0.043. That is the number to read first: it says how much of the activity the places a ligand's water occupies can account for by themselves, with no field of the molecule in the kernel at all.

**The positions are carrying nothing.** The water field on its own reaches -0.043 and the controls reach random sites alone +0.002 and one shared set of sites alone -0.051, so a block of plausible numbers does as well as the measured maps and the places the water occupies are not what the model is using.

**What this rests on.** One set of 25 ligands, 2 ns of simulation each, and a definition of a hydration site -- a point occupied at least half the frames, with sites no closer than one water across. A longer simulation would place the sites more surely; it would not make 25 ligands a larger set, and the interval above is set by the 25 and not by the nanoseconds.

