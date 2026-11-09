# The gradient of a prediction

Written by `python benchmarks/run_gradient.py`.

## Is it the derivative? Relative disagreement with central finite differences

| set | molecule | atoms | step 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 |
|---|---|---:|---:|---:|---:|---:|---:|
| steroids | prednisolone | 54 | 3.4e-04 | 3.4e-06 | 3.4e-08 | 6.2e-09 | 4.1e-08 |
| THR | 01 | 75 | 5.4e-05 | 5.4e-07 | 5.4e-09 | 2.3e-08 | 2.1e-07 |

The disagreement falls as the square of the step and rises again below about 1e-4 angstrom, where round-off in the predictions takes over. That shape is the signature of a correct gradient; the smallest value in each row is the accuracy of the comparison, not of the gradient.

## What does it cost?

| set | atoms | degrees of freedom | analytic | finite differences | ratio |
|---|---:|---:|---:|---:|---:|
| steroids | 54 | 432 | 5.6 ms | 1068 ms | 190x |
| THR | 75 | 600 | 28.8 ms | 7448 ms | 258x |

## What happens if you follow it?

Steepest ascent on the atom positions, no atom moving more than 0.05 A per step.

### steroids, molecule prednisolone (measured +7.51)

| step | prediction | largest displacement | rmsd | posterior sd | worst bond length change |
|---:|---:|---:|---:|---:|---:|
| 0 | +7.934 | 0.00 A | 0.00 A | 0.234 | 0.00 A |
| 1 | +8.149 | 0.05 A | 0.01 A | 0.247 | 0.06 A |
| 2 | +8.349 | 0.10 A | 0.03 A | 0.268 | 0.12 A |
| 5 | +8.875 | 0.25 A | 0.07 A | 0.356 | 0.29 A |
| 10 | +9.534 | 0.50 A | 0.12 A | 0.508 | 0.54 A |
| 20 | +10.329 | 1.00 A | 0.22 A | 0.701 | 0.80 A |

### THR, molecule 01 (measured +8.38)

| step | prediction | largest displacement | rmsd | posterior sd | worst bond length change |
|---:|---:|---:|---:|---:|---:|
| 0 | +7.321 | 0.00 A | 0.00 A | 0.281 | 0.00 A |
| 1 | +7.420 | 0.05 A | 0.01 A | 0.275 | 0.02 A |
| 2 | +7.518 | 0.10 A | 0.02 A | 0.271 | 0.03 A |
| 5 | +7.817 | 0.25 A | 0.05 A | 0.268 | 0.08 A |
| 10 | +8.324 | 0.50 A | 0.10 A | 0.300 | 0.18 A |
| 20 | +9.336 | 0.97 A | 0.22 A | 0.455 | 0.44 A |

The prediction rises, and so does the posterior standard deviation: the molecule is leaving the training set while the number is getting better. The last column says what else is happening -- the bond lengths are being pulled apart, because nothing in the gradient knows that a molecule has bonds. Both columns are the reason this gradient is a term to add to a generator or a force field that enforces geometry, and not an optimiser to run on its own.

## The same gradient on the rotation-invariant kernel

Written by `python benchmarks/run_gradient.py --stage invariant`. The invariant kernel has no frame: its channels are the distance distribution of the field weights inside one structure, so this gradient moves a conformation and not a pose. The last row is the case the gradient was asked for, a local environment of a crystal around a lithium jump (item BT7).

| set | sample | atoms | fields | channels | step 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| steroids | prednisolone | 54 | 5 | 255 | 2.4e-04 | 2.4e-06 | 2.4e-08 | 3.1e-09 | 2.3e-08 |
| THR | 01 | 75 | 5 | 255 | 4.8e-05 | 4.8e-07 | 5.2e-09 | 1.4e-08 | 1.5e-07 |
| hops | mp-10620 LiO2Pr: 3 -> 1, 3.20 A | 22 | 10 | 935 | 5.5e-02 | 6.2e-04 | 6.2e-06 | 6.2e-08 | 1.9e-08 |

The disagreement falls as the square of the step -- a decade of step is two decades of error -- until round-off in the predictions takes over, between 1e-4 and 1e-6 angstrom depending on the sample. That shape is the signature of a correct gradient, and the smallest number in a row is what the comparison can resolve, not the error of the gradient.

| set | degrees of freedom | analytic | finite differences | ratio |
|---|---:|---:|---:|---:|
| steroids | 432 | 14.0 ms | 6068 ms | 432x |
| THR | 600 | 21.0 ms | 11868 ms | 565x |
| hops | 286 | 22.1 ms | 9136 ms | 414x |

## Relaxing a pose under the gradient

Written by `python benchmarks/run_gradient.py --stage relax`. The molecule is moved as a rigid body -- six parameters about its own centroid -- from a pose it did not arrive in. `cmfqsar.relax.relax_pose` raises the prediction; `cmfqsar.varalign` moves the same molecule from the same start by the posterior variance instead. The guard is how far the atoms may move before a penalty holds them back.

### Does the gradient reach what the same objective reaches without it?

| set | molecule | guard | prediction at the start | prediction with the gradient | prediction without it | evaluations with the gradient | evaluations without it | distance between the two poses |
|---|---|---|---:|---:|---:|---:|---:|---:|
| steroids | prednisolone | 2 A | +5.261 | +9.928 | +9.928 | 29 | 2508 | 0.00 A |
| steroids | prednisolone | none | +5.261 | +9.928 | +9.928 | 31 | 1375 | 0.00 A |
| THR | 01 | 2 A | +5.608 | +8.857 | +8.851 | 87 | 2116 | 0.09 A |
| THR | 01 | none | +5.608 | +9.295 | +9.295 | 81 | 2041 | 0.00 A |

The two searches reach the same value of the objective, and they reach it from 24 to 86 times fewer evaluations when the gradient drives them. Where a guard is active the optimum sits on the guard's sphere, where the objective is nearly flat, so the two searches can stop at poses that differ while their values do not: the value columns are the comparison, the pose column is the shape of the optimum.

### Is that the pose the confidence aligner finds?

| set | guard | variance at the start | variance at the confidence pose | variance at the prediction's pose | prediction at the prediction's pose | prediction at the confidence pose | distance between the two poses |
|---|---|---:|---:|---:|---:|---:|---:|
| steroids | 2 A | 1.967 | 0.041 | 1.336 | +9.928 | +7.773 | 1.33 A |
| steroids | none | 1.967 | 0.041 | 1.336 | +9.928 | +7.773 | 1.33 A |
| THR | 2 A | 1.162 | 0.074 | 0.833 | +8.857 | +7.447 | 0.97 A |
| THR | none | 1.162 | 0.074 | 1.361 | +9.295 | +7.447 | 2.19 A |

**They do not agree, and that is not a fault in either.** Raising the prediction moves a molecule towards the training molecules that are active; raising confidence moves it towards the training set as a whole, with redundant members down-weighted by the inverse kernel. The two poses end one to two angstroms apart, the variance at the prediction's pose is 11 to 33 times the variance the aligner reaches, and the prediction at the confidence pose is 1.4 to 2.2 units below the one the relaxation reaches. So the gradient is usable as a driver -- that is the table above -- but the pose it drives to answers a different question from the pose the aligner finds. For a pose to predict from, use the confidence, which uses no activity value of the molecule being placed; the relaxation answers where a molecule would have to sit for the model to call it active.
