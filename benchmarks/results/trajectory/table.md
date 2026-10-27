# Reading a trajectory as a conformational ensemble

Plan item D7 is the reader; plan item D8 is the measurement it exists for, and this is not that measurement. What is measured here belongs to the reader itself: that the two readers agree, what the thinning of the frames loses, and what the frames cost. The frames are conformations of real ligands written to a real trajectory file, not a simulation, because this repository has none.

## The two readers agree

On 36 trajectories read by both mdtraj and MDAnalysis, the largest difference in any coordinate is **3.6e-07 A**. Which of the two is installed therefore does not change a result.

## What the thinning loses

36 molecules of 60 frames each. The reference is the mean field over every frame; the error is of the joint kernel, which is what a model is fitted to.

| clustering radius | conformations kept | error of the kernel | kernel time |
| ---: | ---: | ---: | ---: |
| 0.25 A | 34.5 | 1.01e-02 | 87.3 s |
| 0.50 A | 31.2 | 1.67e-02 | 77.7 s |
| 0.75 A | 25.2 | 3.84e-02 | 59.1 s |
| 1.00 A | 21.1 | 4.74e-02 | 43.7 s |
| 1.50 A | 12.6 | 8.28e-02 | 17.3 s |
| 2.00 A | 6.4 | 9.66e-02 | 4.2 s |
| 3.00 A | 1.9 | 1.21e-01 | 0.2 s |

All 60 frames take 175.9 s for the same kernel.

## What the frames cost

| conformations asked for | kept | kernel time |
| ---: | ---: | ---: |
| 1 | 1.0 | 0.06 s |
| 2 | 1.6 | 0.09 s |
| 5 | 3.7 | 0.55 s |
| 10 | 6.0 | 1.50 s |
| 20 | 13.2 | 9.29 s |
| 40 | 27.1 | 37.11 s |

## What it settles

**The thinning trades accuracy for time in a straight line, with no radius at which it is free.** Over 36 molecules of 60 frames, the kernel error runs from 1.0e-02 at 0.25 A (34 conformations kept) to 1.2e-01 at 3.00 A (2 kept). It does not flatten out: keeping more frames keeps improving the kernel, so the number to keep is a decision about cost and not a point where the curve stops moving.

At the default radius of 1 A the ensemble keeps 21 of 60 conformations, the kernel is within **4.7e-02** of the one built from every frame, and it takes 44 s against 176 s for all of them. A few per cent of the kernel for a quarter of the time is the trade the default makes. The radius is the control, but what the cost follows is the number of conformations it leaves, and that number depends on how flexible the molecules are: the same radius keeps far more frames of a flexible ligand than of a rigid one.

**The cost follows the square of the number of conformations, as the algebra says it must.** From 1 to 27 conformations is 27 times as many and 658 times the kernel time, against 734 for an exact square. The kernel of two ensembles is a double sum over their members, so the thinning belongs inside the reader rather than being left to the caller.

**Which reader is installed does not matter.** mdtraj and MDAnalysis differ by at most 4e-07 A on any coordinate of 36 trajectories.

**What this does not say.** Nothing here compares a trajectory with a crystal pose, because the frames are conformations of the ligands and not a simulation. That comparison is plan item D8, and it needs short simulations on a set whose ligands have crystal poses.

