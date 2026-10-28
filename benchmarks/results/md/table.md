# Does a trajectory beat the crystal pose?

Plan item D8. Five models of the same ligands on the same folds, with the same alpha and lambda grids: the crystal pose alone, an ensemble of generated conformations (what the fourth-dimension study did), and three readings of a short simulation of each ligand in its own crystal site. Every difference is against the crystal pose, with the bootstrap interval of rule 4.

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

## What it settles

**On mcl1 the trajectory neither beats the crystal pose nor loses to it.** The best reading of it, trajectory, multi-instance, differs by -0.112 [-0.384, +0.169], an interval that contains zero, so on this set the simulation is not worth its cost. The crystal pose reaches +0.575 and it reaches +0.463.

The control says where that comes from. An ensemble of generated conformations -- what the fourth-dimension study built -- changes q2 by -0.019 [-0.271, +0.279] against the same crystal pose. 

**How the clusters are weighted matters more than whether the trajectory is used at all.** Equal weights give +0.391 and weights by population +0.180, a gap of 0.211, which is larger than the 0.184 between the better of the two and the crystal pose. Equal weights is the better reading here. The item asks for equal weights; weights by population are what `cmfqsar/trajectory.py` writes, on the argument that an equilibrated trajectory has already sampled the distribution. That argument is about the physics and this measurement is about the model, and on this set they disagree.

Letting the model choose the frames gives +0.463, -0.112 [-0.384, +0.169] against the crystal pose. The search chose beta = 0, which is the plain ensemble with every conformation weighing the same, so on this set letting the model pick a bioactive frame gained nothing over averaging them.

**The common frame is what makes any of this measurable.** Read the same frames with each trajectory superimposed on its own first frame instead of on the one structure every run was built from, and q2 falls from +0.391 to +0.277. Two runs equilibrate to protein conformations a few tenths of an angstrom apart, so fitting each ligand to its own leaves the series in as many frames as it has molecules. That is 0.114 of q2 lost to a misalignment far smaller than the ones part C of the plan is about, which is a measure of how much a field kernel asks of an alignment.

**What this rests on.** One set, simulated with GAFF in TIP3P water for the length stated in `examples/md/README.md`. A few nanoseconds samples the ligand's motion around its crystal pose and not a change of binding mode, which is the honest scope of the question asked here. The ligands stayed where they started -- the data statement gives the median distance from the crystal pose -- so this measures what an ensemble around a correct pose is worth, which is exactly the question, and not whether a simulation can find a pose.

**What follows for the plan.** The item said that if the trajectory mean does not beat the crystal pose then plan items D9 and D10 are not built on it. No reading of the trajectory beats the crystal pose here, so D9 -- solvent as a field, read off these trajectories -- does not follow from this result. D10 was built before this measurement and on conformer ensembles rather than trajectories, and its own verdict already stands.

