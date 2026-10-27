# Does the co-field know which groups are interchangeable? (plan item MC1)

A bioisostere table answers which groups can replace each other at a position, so that one of them can be chosen for a property the model does not describe. The prediction of the overlap kernel is a sum over the atoms, so the part of it that belongs to a substituent is the sum over that substituent's own atoms, and exchanging one group for another is the difference of two such sums -- one integral per pair of groups rather than one model fit per group.

The pairs are matched molecular pairs from RDKit's `rdMMPA`. The three measures are the ladder of rule 1: the fingerprint distance between the groups (no geometry, no model), the field distance between the two posed groups restricted to their own atoms (geometry, no model), and the predicted change (geometry and the co-field). All three are compared with the size of the measured activity difference by rank correlation. For every pair the dual coefficients are refitted without both of its members, so the predicted change has not seen the two activities it is judged against.

## GPB: glycogen phosphorylase b, 66 molecules

730 matched pairs among 66 molecules; the model reaches q2 +0.533 at alpha 2 and ridge 1, with a conformal half width of 1.30 log units at 90 per cent.

| measure | rank correlation with the size of the measured change |
| --- | ---: |
| fingerprint distance | -0.025 |
| field distance | +0.005 |
| predicted change, the pair left out | +0.561 |
| predicted change, the whole model | +0.631 |

| the co-field against | difference in rank correlation |
| --- | ---: |
| the fingerprint distance | +0.586 [+0.505, +0.669] |
| the field distance | +0.556 [+0.479, +0.632] |

The **signed** change against the signed measurement: +0.712.

A table of 35 groups on one core is in `table_GPB.csv`. At twice the conformal half width (2.60 log units), 35 of the 35 groups have at least one partner the model cannot tell them apart from.

## THER: thermolysin, 76 molecules

142 matched pairs among 76 molecules; the model reaches q2 +0.581 at alpha 1 and ridge 0.1, with a conformal half width of 2.06 log units at 90 per cent.

| measure | rank correlation with the size of the measured change |
| --- | ---: |
| fingerprint distance | +0.084 |
| field distance | +0.162 |
| predicted change, the pair left out | +0.277 |
| predicted change, the whole model | +0.641 |

| the co-field against | difference in rank correlation |
| --- | ---: |
| the fingerprint distance | +0.193 [-0.059, +0.437] |
| the field distance | +0.116 [-0.089, +0.313] |

The **signed** change against the signed measurement: +0.626.

A table of 9 groups on one core is in `table_THER.csv`. At twice the conformal half width (4.12 log units), 9 of the 9 groups have at least one partner the model cannot tell them apart from.

## THR: thrombin, 88 molecules

751 matched pairs among 88 molecules; the model reaches q2 +0.680 at alpha 2 and ridge 0.01, with a conformal half width of 0.97 log units at 90 per cent.

| measure | rank correlation with the size of the measured change |
| --- | ---: |
| fingerprint distance | +0.144 |
| field distance | +0.242 |
| predicted change, the pair left out | +0.335 |
| predicted change, the whole model | +0.711 |

| the co-field against | difference in rank correlation |
| --- | ---: |
| the fingerprint distance | +0.191 [+0.106, +0.279] |
| the field distance | +0.093 [+0.008, +0.181] |

The **signed** change against the signed measurement: +0.644.

A table of 35 groups on one core is in `table_THR.csv`. At twice the conformal half width (1.95 log units), 35 of the 35 groups have at least one partner the model cannot tell them apart from.

## What the sets say together

| set | pairs | fingerprint | field | co-field (pair left out) | co-field (whole model) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPB | 730 | -0.025 | +0.005 | +0.561 | +0.631 |
| THER | 142 | +0.084 | +0.162 | +0.277 | +0.641 |
| THR | 751 | +0.144 | +0.242 | +0.335 | +0.711 |

## The verdict

**The co-field weighting earns its place.** On 2 of the 3 sets (GPB, THR) the predicted change ranks the measured activity difference better than both the fingerprint distance and the field distance by more than the interval of rule 4, and on no set is it worse than either. On THER it is ahead of both as well, but the interval contains zero; that is the set with the fewest pairs (142 against 751 on the largest).

The reading is that a bioisostere table is worth computing from a model. The two references are not weak ones -- the field distance compares the two groups **in their poses**, so it already knows the shape and the charge each of them puts where -- and neither carries much: the fingerprint distance ranks nothing at all on GPB. What the co-field adds is the part the references cannot have, which is whether the activity depends on the place where the two groups differ.

**How much of it is memory** is the difference between the last two columns. The whole model, which has seen both activities of every pair it is judging, reaches +0.631 on GPB, +0.641 on THER, +0.711 on THR; refitting without the two members of each pair leaves +0.561, +0.277, +0.335. The gap is large, and it is the reason the honest column is the one reported above. Two things still come from the whole set: the field type, attenuation and ridge were chosen on all the molecules, and the alignment is the published one.

The column that matters for the chemist is not the ranking but the size. A predicted change smaller than twice the conformal half width is a change the model cannot see, and the groups that fall inside it are the ones the table calls interchangeable. That part of the table is the model's own error bar rather than a claim about the chemistry, so it is sound whatever the ranking says.

