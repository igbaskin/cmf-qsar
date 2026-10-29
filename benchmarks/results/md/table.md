# Does a trajectory beat the crystal pose?

Plan item D8. Five models of the same ligands on the same folds, with the same alpha and lambda grids: the crystal pose alone, an ensemble of generated conformations (what the fourth-dimension study did), and three readings of a short simulation of each ligand in its own crystal site. Every difference is against the crystal pose, with the bootstrap interval of rule 4.

## hif2a

36 ligands, 10.0 trajectory conformations each (from 4 to 16).

| model | q2 | RMSEcv | attenuation | change against the crystal pose |
| --- | ---: | ---: | ---: | --- |
| the crystal pose | +0.342 | 0.590 | 0.3 | the reference |
| generated conformations | +0.311 | 0.604 | 1 | -0.031 [-0.246, +0.433] |
| trajectory, clusters equal | +0.471 | 0.529 | 0.3 | +0.129 [-0.076, +0.365] |
| trajectory, by population | +0.534 | 0.496 | 0.3 | +0.192 [-0.068, +0.561] |
| trajectory, multi-instance | +0.520 | 0.504 | 0.3 | +0.178 [+0.003, +0.465] |
| trajectory, each in its own frame | +0.297 | 0.610 | 0.1 | -0.045 [-0.323, +0.214] |

## mcl1

25 ligands, 11.3 trajectory conformations each (from 4 to 20).

| model | q2 | RMSEcv | attenuation | change against the crystal pose |
| --- | ---: | ---: | ---: | --- |
| the crystal pose | +0.575 | 0.542 | 0.1 | the reference |
| generated conformations | +0.556 | 0.554 | 2 | -0.019 [-0.271, +0.279] |
| trajectory, clusters equal | +0.391 | 0.649 | 2 | -0.184 [-0.526, +0.136] |
| trajectory, by population | +0.180 | 0.753 | 2 | -0.395 [-0.830, -0.053] |
| trajectory, multi-instance | +0.463 | 0.609 | 2 | -0.112 [-0.384, +0.169] |
| trajectory, each in its own frame | +0.277 | 0.707 | 2 | -0.298 [-0.670, -0.032] |

## p38

29 ligands, 11.0 trajectory conformations each (from 3 to 20).

| model | q2 | RMSEcv | attenuation | change against the crystal pose |
| --- | ---: | ---: | ---: | --- |
| the crystal pose | +0.261 | 0.600 | 0.1 | the reference |
| generated conformations | +0.082 | 0.668 | 0.3 | -0.178 [-0.414, +0.182] |
| trajectory, clusters equal | +0.239 | 0.608 | 0.5 | -0.022 [-0.310, +0.299] |
| trajectory, by population | +0.091 | 0.665 | 0.5 | -0.169 [-0.402, +0.181] |
| trajectory, multi-instance | +0.455 | 0.515 | 2 | +0.195 [-0.133, +0.522] |
| trajectory, each in its own frame | +0.368 | 0.555 | 0.5 | +0.107 [-0.351, +0.498] |

## What it settles

**On hif2a the trajectory neither beats the crystal pose nor loses to it.** The best reading of it, trajectory, by population, differs by +0.192 [-0.068, +0.561], an interval that contains zero. The crystal pose reaches +0.342 and it reaches +0.534.

**A different reading of the same frames does clear the interval.** On hif2a, trajectory, multi-instance is +0.178 [+0.003, +0.465] ahead of the crystal pose, while the largest gain, trajectory, by population, is not. At beta = 0 the multi-instance fit is the uniform ensemble, so the reading to hold it against is trajectory, clusters equal, which is the same model: that gives +0.129 [-0.076, +0.365] and does not clear the interval. What separates them is where the kernel is normalised, which the last section of this table measures. So what this set supports is the weaker statement: some reading of the trajectory is ahead here, and which reading that is was not fixed before the measurement.

The control says where that comes from. An ensemble of generated conformations -- what the fourth-dimension study built -- changes q2 by -0.031 [-0.246, +0.433] against the same crystal pose. 

How the clusters are weighted changes little: equal weights give +0.471 and weights by population +0.534, a gap of 0.063, smaller than the 0.129 between the better of them and the crystal pose.

Letting the model choose the frames gives +0.520, +0.178 [+0.003, +0.465] against the crystal pose. The search chose beta = 0, which is the plain ensemble with every conformation weighing the same, so on this set letting the model pick a bioactive frame gained nothing over averaging them.

**On mcl1 the trajectory neither beats the crystal pose nor loses to it.** The best reading of it, trajectory, multi-instance, differs by -0.112 [-0.384, +0.169], an interval that contains zero, so on this set the simulation is not worth its cost. The crystal pose reaches +0.575 and it reaches +0.463.

The control says where that comes from. An ensemble of generated conformations -- what the fourth-dimension study built -- changes q2 by -0.019 [-0.271, +0.279] against the same crystal pose. 

**How the clusters are weighted matters more than whether the trajectory is used at all.** Equal weights give +0.391 and weights by population +0.180, a gap of 0.211, which is larger than the 0.184 between the better of the two and the crystal pose. Equal weights is the better reading here. The item asks for equal weights; weights by population are what `cmfqsar/trajectory.py` writes, on the argument that an equilibrated trajectory has already sampled the distribution. That argument is about the physics and this measurement is about the model, and on this set they disagree.

Letting the model choose the frames gives +0.463, -0.112 [-0.384, +0.169] against the crystal pose. The search chose beta = 0, which is the plain ensemble with every conformation weighing the same, so on this set letting the model pick a bioactive frame gained nothing over averaging them.

**On p38 the trajectory neither beats the crystal pose nor loses to it.** The best reading of it, trajectory, multi-instance, differs by +0.195 [-0.133, +0.522], an interval that contains zero, so on this set the simulation is not worth its cost. The crystal pose reaches +0.261 and it reaches +0.455.

The control says where that comes from. An ensemble of generated conformations -- what the fourth-dimension study built -- changes q2 by -0.178 [-0.414, +0.182] against the same crystal pose. 

**How the clusters are weighted matters more than whether the trajectory is used at all.** Equal weights give +0.239 and weights by population +0.091, a gap of 0.147, which is larger than the 0.022 between the better of the two and the crystal pose. Equal weights is the better reading here. The item asks for equal weights; weights by population are what `cmfqsar/trajectory.py` writes, on the argument that an equilibrated trajectory has already sampled the distribution. That argument is about the physics and this measurement is about the model, and on this set they disagree.

Letting the model choose the frames gives +0.455, +0.195 [-0.133, +0.522] against the crystal pose. The search chose beta = 1, so the model did concentrate the weight on some frames rather than averaging them evenly.

**The sets disagree.** Some reading of the trajectory beats the crystal pose beyond the interval on 1 of the 3 sets (hif2a) and on none of the others. The largest gain on each set: **hif2a** (36 ligands) +0.192 [-0.068, +0.561], trajectory, by population; **mcl1** (25 ligands) -0.112 [-0.384, +0.169], trajectory, multi-instance; **p38** (29 ligands) +0.195 [-0.133, +0.522], trajectory, multi-instance. A result that holds on some sets and not on others is not a method yet; what separates them is the next thing to measure. **Read that carefully.** 9 readings of a trajectory were compared with their crystal pose across the 3 sets, and the nearest edge of a clearing interval is +0.003 from zero. One result at the edge of 9 comparisons is about what chance produces, so what the sets support is a weak statement and not a method.

**What this rests on.** 3 sets, each simulated with GAFF in TIP3P water for the length stated in `examples/md/README.md`. A few nanoseconds samples the ligand's motion around its crystal pose and not a change of binding mode, which is the fair scope of the question asked here. The ligands stayed where they started -- the data statement gives the median distance from the crystal pose -- so this measures what an ensemble around a correct pose is worth, which is exactly the question, and not whether a simulation can find a pose.

**What follows for the plan.** The item said that if the trajectory mean does not beat the crystal pose then plan items D9 and D10 are not built on it. No reading of the trajectory beats the crystal pose on 2 of the 3 sets (mcl1, p38), so D9 -- solvent as a field, read off these trajectories -- does not follow from this result. D10 was built before this measurement and on conformer ensembles rather than trajectories, and its own verdict already stands.

The gate is no longer a plain no, because some reading is ahead beyond the interval on hif2a. D9 was built in any case, at the author's instruction, and it failed on its own evidence: two readings of its definition sweep were written down beforehand and both failed on a set they were not read from. So the gate reopening changes what is still open here, which is the trajectory itself, and not the verdict of D9.

## One reading fixed in advance

Every gain above is the largest of several readings of the same frames. This section reports one reading instead: the trajectory with every conformation weighing the same, the "trajectory, clusters equal" row of the tables above. It was fixed before the sets were compared with each other, and it contains no choice -- there is no weighting to pick and no beta to search -- so it cannot be the reading that happened to win. It is not the largest gain on any of the 3 sets.

| set | ligands | the crystal pose | trajectory, clusters equal | change against the crystal pose |
| --- | ---: | ---: | ---: | --- |
| hif2a | 36 | +0.342 | +0.471 | +0.129 [-0.076, +0.365] |
| mcl1 | 25 | +0.575 | +0.391 | -0.184 [-0.526, +0.136] |
| p38 | 29 | +0.261 | +0.239 | -0.022 [-0.310, +0.299] |

**Combined over the 3 sets: -0.006 [-0.157, +0.163]**, over 90 molecules. Each set is weighted by the molecules it contributes, and the molecules are resampled inside their own set, never across sets: two sets have different targets and their activities are not on one scale.

**Fixed in advance, the trajectory shows no gain.** It clears the interval on none of the 3 sets, and the combined estimate does not clear it either. The gains in the tables above are what taking the best of several readings produces. This is the number the item should be read as giving.

This is not a pre-registration. The sets had been measured before the rule was written down, and no rule written afterwards can undo that. What it does avoid is the other fault: deciding which of several readings to believe after seeing which of them won. The attenuation and lambda are still chosen by cross-validation inside every model, which the crystal pose gets in the same measure, so that choice is not what separates the two columns.

## What the common frame is worth

The same frames, read twice, with the same weighting (clusters equal). Everywhere above, a trajectory is superimposed on the one structure all the runs of its set were built from, so the whole series sits in a single frame. Here each trajectory is superimposed on its own first frame instead, which leaves every ligand in a frame of its own.

| set | ligands | one common frame | each in its own frame | change against the common frame |
| --- | ---: | ---: | ---: | --- |
| hif2a | 36 | +0.471 | +0.297 | -0.175 [-0.391, -0.020] |
| mcl1 | 25 | +0.391 | +0.277 | -0.114 [-0.386, +0.081] |
| p38 | 29 | +0.239 | +0.368 | +0.129 [-0.207, +0.369] |

**Dropping the common frame does not change q2 in one direction.** The change is between -0.175 and +0.129 of q2 over the 3 sets, and the sign is not the same on all of them. The interval excludes zero on 1 of the 3 sets (hif2a), so only there is the change larger than the resampling of the molecules explains. Two runs of the same protein equilibrate to conformations a few tenths of an angstrom apart, so fitting each ligand to its own run leaves the series in as many frames as it has molecules. A field kernel was expected to lose from that, and on one set it does. On the others it does not, so the measurement does not support the general statement that a common frame is what makes this comparison possible. What an alignment of a few tenths of an angstrom is worth is set by set here, not by the method. One more reason to read the gains of the tables above with care: on 1 of the 3 sets this control is itself above the crystal pose (p38 +0.107), and it was built to be wrong.

## Poses that are not known

Every number above rests on crystal poses, and the item itself says that an ensemble should help most where the pose is *not* known. On **mcl1** it is asked there too. Every ligand was posed while it was held out: its pose was destroyed, a random placement within 2 A of the site, and then the receptor's own fields put it back where the pocket prefers it -- no model, no activity of its own and no crystal frame of any kind. Those poses sit a median 1.12 A from the crystal poses (36% within 1 A, 56% within 2 A, at most 8.26 A). Each ligand was then simulated from its found pose, for the same length as everywhere above.

25 ligands, 11.4 conformations each.

| model | q2 | change against the pose that was found | change against the crystal pose |
| --- | ---: | --- | --- |
| the crystal pose | +0.575 | +0.173 [-0.144, +0.567] | the ceiling |
| the pose that was found | +0.402 | the reference | -0.173 [-0.567, +0.144] |
| trajectory from it, clusters equal | -0.040 | -0.442 [-1.461, +0.272] | -0.615 [-1.470, -0.107] |
| trajectory from it, by population | +0.024 | -0.378 [-0.833, +0.185] | -0.551 [-0.957, -0.197] |
| trajectory from it, multi-instance | -0.012 | -0.413 [-1.134, +0.198] | -0.587 [-1.140, -0.215] |

**Did the simulation repair the pose?** The nearest kept conformation of each ligand is a median 0.79 A from its crystal pose and the middle conformation 1.51 A, against 1.12 A for the pose the simulation started from. So two nanoseconds in the site bring the ligand nearer to the pose it should have had. The nearest conformation is chosen after the fact, out of 11.4 of them, so part of that is a smallest of several and not repair. The middle conformation is farther from the crystal pose than the starting pose was, so the ensemble as a whole did not move towards it.

**The ensemble gains nothing here either.** The reading fixed in advance gives -0.442 [-1.461, +0.272] against the single pose it started from, which does not clear the interval. So the place where an ensemble should help most does not show it helping, on this set and at this length. The largest of the three readings is trajectory from it, by population, -0.378 [-0.833, +0.185], and taking the largest of three is what the section above measured the cost of.

What the whole exercise costs: posing the ligands instead of reading their crystal poses gives -0.173 [-0.567, +0.144] of q2. That is the size of the thing an ensemble would have to repair.

## Where the kernel is normalised

The multi-instance fit at beta = 0 is the uniform ensemble: every conformation weighs the same. The two are the same model and they do not give the same q2. Each row is one attenuation (0.3) and one lambda (0.1), with no grid and nothing chosen, so these numbers are not comparable with the table above, where every model is taken at its own best attenuation.

| set | ligands | ensemble | multi-instance, beta = 0 | ensemble, no normalisation | multi-instance, no normalisation |
| --- | ---: | ---: | ---: | ---: | ---: |
| hif2a | 36 | +0.471 | +0.520 | +0.550 | +0.552 |
| mcl1 | 25 | -0.068 | -0.049 | +0.168 | +0.183 |
| p38 | 29 | +0.198 | +0.223 | +0.255 | +0.264 |

**The two paths differ by 0.019 to 0.049 of q2, and the difference narrows to 0.002 to 0.015 when the normalisation is switched off** -- on 3 of the 3 sets. So most of what separates two paths through one model is where the kernel is normalised: the ensemble path normalises the kernel of the whole ensemble, the multi-instance path normalises every conformation and then averages. Up to 0.015 is left over, so the normalisation is not all of it.

Both places are defensible, which is why this matters for reading the table: a gain of this size can come from the choice alone. Where the only reading of a set that clears the interval is the multi-instance fit at beta = 0, that reading is the uniform ensemble with the kernel normalised in the other place, and the uniform ensemble itself does not clear it.

