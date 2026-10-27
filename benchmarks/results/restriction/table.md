# Does a ring closure remove the field that moves? (plan item MC9)

A conformational restriction is a ring closure that stops a bond rotating. The variance field of plan item D10 says where a molecule places field only part of the time; weighted by the square of the co-field it says how much of the **activity-relevant** field is intermittent. A pair of a flexible molecule and its restricted analogue is scored by how much of that intermittent part the restriction removes, and the question is whether that tracks the measured potency gain.

The pairs are found by a mechanical rule, not by eye: one ring more, at least one rotatable bond fewer, at most two heavy atoms more, an atom composition differing by at most two atoms, and a common substructure covering at least 80 per cent of the smaller member. Both members get the same conformer generator with the same settings.

## THR: thrombin, 88 molecules

33 pairs among 88 molecules; the model of the set reaches q2 +0.578 at ridge 0.01, and its co-field is the weight. 3 conformations per molecule, a grid of 12350 points at 1.00 A.

| score | correlation with the potency gain |
| --- | ---: |
| rotatable bonds removed | -0.407 [-0.686, -0.044] |
| variance removed, unweighted | -0.058 [-0.387, +0.250] |
| variance removed, weighted | +0.111 [-0.141, +0.367] |
| variance removed, weighted, pair left out | +0.061 [-0.218, +0.420] |
| relevant field kept, weighted | -0.216 [-0.511, +0.102] |

## COX2: cyclooxygenase 2, 322 molecules

145 pairs among 282 molecules; the model of the set reaches q2 +0.519 at ridge 0.1, and its co-field is the weight. 3 conformations per molecule, a grid of 7497 points at 1.00 A.

| score | correlation with the potency gain |
| --- | ---: |
| rotatable bonds removed | +0.199 [+0.039, +0.355] |
| variance removed, unweighted | +0.033 [-0.178, +0.227] |
| variance removed, weighted | +0.208 [+0.077, +0.320] |
| variance removed, weighted, pair left out | +0.177 [+0.020, +0.306] |
| relevant field kept, weighted | +0.432 [+0.248, +0.585] |

## DHFR: dihydrofolate reductase, 397 molecules

190 pairs among 361 molecules; the model of the set reaches q2 +0.585 at ridge 0.1, and its co-field is the weight. 3 conformations per molecule, a grid of 11040 points at 1.00 A.

| score | correlation with the potency gain |
| --- | ---: |
| rotatable bonds removed | +0.466 [+0.299, +0.594] |
| variance removed, unweighted | +0.147 [+0.021, +0.282] |
| variance removed, weighted | -0.066 [-0.213, +0.075] |
| variance removed, weighted, pair left out | -0.065 [-0.211, +0.075] |
| relevant field kept, weighted | +0.186 [+0.049, +0.325] |

## ACE: angiotensin converting enzyme, 114 molecules

71 pairs among 114 molecules; the model of the set reaches q2 +0.571 at ridge 1, and its co-field is the weight. 3 conformations per molecule, a grid of 13440 points at 1.00 A.

| score | correlation with the potency gain |
| --- | ---: |
| rotatable bonds removed | +0.084 [-0.138, +0.337] |
| variance removed, unweighted | -0.170 [-0.308, +0.002] |
| variance removed, weighted | +0.050 [-0.138, +0.260] |
| variance removed, weighted, pair left out | -0.068 [-0.237, +0.118] |
| relevant field kept, weighted | +0.212 [+0.075, +0.366] |

## What the sets say together

| set | pairs | rotatable bonds | variance removed, unweighted | weighted | weighted, pair left out |
| --- | ---: | ---: | ---: | ---: | ---: |
| THR | 33 | -0.407 | -0.058 | +0.111 | +0.061 |
| COX2 | 145 | +0.199 | +0.033 | +0.208 | +0.177 |
| DHFR | 190 | +0.466 | +0.147 | -0.066 | -0.065 |
| ACE | 71 | +0.084 | -0.170 | +0.050 | -0.068 |

### All the pairs together

One log unit of potency means the same thing in every series, so the pairs can be put in one column. This is the single number the gate of the item asks for.

| score | pairs | correlation with the potency gain |
| --- | ---: | ---: |
| rotatable bonds removed | 439 | +0.236 [+0.151, +0.317] |
| variance removed, unweighted | 439 | -0.034 [-0.140, +0.062] |
| variance removed, weighted | 439 | +0.043 [-0.030, +0.111] |
| variance removed, weighted, pair left out | 439 | +0.020 [-0.061, +0.093] |
| relevant field kept, weighted | 439 | +0.220 [+0.120, +0.315] |


## The verdict

**The score does not pass the gate.** Over all 439 pairs the share of intermittent field a ring closure removes correlates with the potency gain at +0.043 [-0.030, +0.111]. Counting the rotatable bonds the closure removed reaches +0.236 [+0.151, +0.317] on the same pairs, and that needs no geometry, no conformations and no model. Leaving both members of a pair out of the co-field changes the score to +0.020 [-0.061, +0.093], so the result is not an artefact of the weight: there is simply little there.

The picture is the same set by set: the interval excludes zero on 1 of the 4 sets in the direction the item expects and on 0 in the opposite direction, and contains zero on the rest. A score that reaches the gate on one series out of four describes that series rather than conformational restriction.

**What does track the potency gain is the other half of the score.** How much of the activity-relevant field the restricted analogue still places -- the **kept** row -- reaches +0.220 [+0.120, +0.315] over the same pairs. That is the mean field and not the variance, and it says the expected thing: a ring closure helps when the molecule still puts its field where the model reads it, and hurts when the ring holds it somewhere else. The measurement here is of the variance, and the variance is not where the answer is.

This is the third place in this repository where the conformational ensemble carries less than it promises. Plan item D9 found that averaging a field over conformations does not predict better than one geometry; plan item D10 found that the variance across conformations adds nothing as a block of the kernel; and this item finds that it does not explain what a conformational restriction did either. The variance field stays what D10 made it: a map of where a series is rigid, worth drawing and not worth scoring with.

Two limits of the measurement are worth stating. The ensembles are three conformations from a distance-geometry generator superimposed on the pose the series was aligned to, which is a narrow view of what a molecule does in water. And a ring closure is never purely conformational in practice: it changes the shape and often an atom or two as well, and the pair rule here allows up to two atoms of difference. A set of pairs that differ in nothing but a single bond would be a better test, and no public set of that kind with measured potencies was found.

