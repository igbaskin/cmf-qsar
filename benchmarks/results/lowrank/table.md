# Fitting in the space of the landmarks (plan item B6)

The exact model holds one n by n matrix per field and factorises the n by n joint kernel, which is what stops it at ten to twenty thousand molecules. The Nystrom approximation replaces every field kernel by the Gram matrix of m features built from m landmarks, and the model is fitted in the space of those features. Plan item B6 assessed the method and measured the approximation; this measures the implementation in `cmfqsar/lowrank.py`.

## Nothing approximated

With every molecule its own landmark the approximation is an identity, so the two paths must agree. On GPB (66 molecules, rank 330), the largest disagreement is:

| what | difference |
| --- | ---: |
| the joint kernel | 1.7e-14 |
| the predictions | 8.2e-14 |
| the log evidence | 4.7e-13 |
| the posterior variance | 3.4e-13 |
| the co-field on a grid | 1.9e-11 |

The last row is the one that matters beyond arithmetic: the co-field of a low-rank fit is a sum over the **landmarks** rather than over the training molecules, and it is the same function. What the program reads off a co-field -- the map, the atom contributions, the pharmacophore -- therefore keeps its meaning.

## What the approximation costs

q2 under 0.1 ridge and attenuation 0.5, ten folds, against the exact model on the same folds. **Every row is at one attenuation**, and a broad one: plan item B6 measured what happens at 2 and above, where a grid of this spacing loses 0.01 to 0.04 of q2 because a grid spans a field only while its spacing is not much larger than the field's own width. This table measures the implementation, not that dependence.

| set | molecules | landmarks | rank | q2 | against the exact model |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACE (grid 2.0 A) | 114 | 347 | 1735 | +0.581 | -0.004 |
| ACE (grid 1.5 A) | 114 | 628 | 3129 | +0.585 | +0.000 |
| ACE (20 molecules) | 114 | 20 | 98 | +0.640 | +0.055 |
| ACE (40 molecules) | 114 | 40 | 193 | +0.560 | -0.025 |
| ACE (80 molecules) | 114 | 80 | 380 | +0.586 | +0.001 |
| ACE (every molecule) | 114 | 114 | 531 | +0.585 | +0.000 |
| AChE (grid 2.0 A) | 111 | 248 | 1240 | +0.531 | +0.002 |
| AChE (grid 1.5 A) | 111 | 428 | 2140 | +0.530 | +0.001 |
| AChE (20 molecules) | 111 | 20 | 83 | +0.528 | -0.001 |
| AChE (40 molecules) | 111 | 40 | 160 | +0.512 | -0.017 |
| AChE (80 molecules) | 111 | 80 | 304 | +0.530 | +0.001 |
| AChE (every molecule) | 111 | 111 | 408 | +0.529 | +0.000 |
| BZR (grid 2.0 A) | 147 | 190 | 950 | +0.360 | -0.003 |
| BZR (grid 1.5 A) | 147 | 316 | 1580 | +0.362 | -0.001 |
| BZR (20 molecules) | 147 | 20 | 89 | +0.347 | -0.017 |
| BZR (40 molecules) | 147 | 40 | 177 | +0.288 | -0.076 |
| BZR (80 molecules) | 147 | 80 | 348 | +0.365 | +0.001 |
| BZR (every molecule) | 147 | 147 | 614 | +0.364 | -0.000 |
| COX2 (grid 2.0 A) | 282 | 169 | 845 | +0.506 | -0.001 |
| COX2 (grid 1.5 A) | 282 | 295 | 1475 | +0.507 | -0.000 |
| COX2 (20 molecules) | 282 | 20 | 93 | +0.525 | +0.017 |
| COX2 (40 molecules) | 282 | 40 | 181 | +0.518 | +0.011 |
| COX2 (80 molecules) | 282 | 80 | 349 | +0.513 | +0.006 |
| COX2 (every molecule) | 282 | 282 | 990 | +0.507 | +0.000 |
| DHFR (grid 2.0 A) | 361 | 255 | 1275 | +0.620 | +0.007 |
| DHFR (grid 1.5 A) | 361 | 438 | 2187 | +0.614 | +0.001 |
| DHFR (20 molecules) | 361 | 20 | 99 | +0.615 | +0.002 |
| DHFR (40 molecules) | 361 | 40 | 191 | +0.607 | -0.006 |
| DHFR (80 molecules) | 361 | 80 | 380 | +0.602 | -0.011 |
| DHFR (every molecule) | 361 | 361 | 1366 | +0.613 | -0.000 |
| GPB (grid 2.0 A) | 66 | 126 | 630 | +0.373 | +0.005 |
| GPB (grid 1.5 A) | 66 | 219 | 1095 | +0.368 | -0.001 |
| GPB (20 molecules) | 66 | 20 | 100 | +0.367 | -0.001 |
| GPB (40 molecules) | 66 | 40 | 200 | +0.348 | -0.020 |
| GPB (every molecule) | 66 | 66 | 330 | +0.368 | -0.000 |
| THER (grid 2.0 A) | 76 | 303 | 1515 | +0.445 | -0.004 |
| THER (grid 1.5 A) | 76 | 523 | 2613 | +0.448 | -0.000 |
| THER (20 molecules) | 76 | 20 | 100 | +0.426 | -0.023 |
| THER (40 molecules) | 76 | 40 | 200 | +0.438 | -0.011 |
| THER (every molecule) | 76 | 76 | 380 | +0.449 | -0.000 |
| THR (grid 2.0 A) | 88 | 299 | 1495 | +0.646 | +0.001 |
| THR (grid 1.5 A) | 88 | 511 | 2548 | +0.646 | +0.000 |
| THR (20 molecules) | 88 | 20 | 99 | +0.604 | -0.042 |
| THR (40 molecules) | 88 | 40 | 196 | +0.638 | -0.007 |
| THR (80 molecules) | 88 | 80 | 379 | +0.645 | +0.000 |
| THR (every molecule) | 88 | 88 | 416 | +0.645 | -0.000 |

## Where a grid fails, and what follows the atoms instead

A grid spans a field while its spacing is not much larger than the field's own width, so at a strong attenuation it would need tens of thousands of centres. Putting a landmark on every atom of the molecules, thinned to a spacing, follows the field instead of covering the box. The number to read is the error of the kernel: these sets are small and a q2 moves by more than the approximation does.

| set | attenuation | landmarks | how many | error of the kernel | q2 | exact q2 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| GPB | 0.5 | a grid 2.0 A apart | 126 | 1.5e-02 | +0.373 | +0.368 |
| GPB | 0.5 | the atoms, 0.75 A apart | 171 | 1.2e-03 | +0.364 | +0.368 |
| GPB | 0.5 | the atoms, 0.5 A apart | 259 | 1.3e-04 | +0.367 | +0.368 |
| GPB | 0.5 | 40 molecules | 40 | 2.6e-03 | +0.348 | +0.368 |
| GPB | 2 | a grid 2.0 A apart | 126 | 3.2e-01 | +0.475 | +0.526 |
| GPB | 2 | the atoms, 0.75 A apart | 171 | 1.3e-02 | +0.516 | +0.526 |
| GPB | 2 | the atoms, 0.5 A apart | 259 | 4.0e-03 | +0.527 | +0.526 |
| GPB | 2 | 40 molecules | 40 | 9.3e-03 | +0.520 | +0.526 |
| GPB | 5 | a grid 2.0 A apart | 126 | 6.5e-01 | +0.548 | +0.594 |
| GPB | 5 | the atoms, 0.75 A apart | 171 | 4.5e-02 | +0.581 | +0.594 |
| GPB | 5 | the atoms, 0.5 A apart | 259 | 1.8e-02 | +0.593 | +0.594 |
| GPB | 5 | 40 molecules | 40 | 1.8e-02 | +0.575 | +0.594 |
| THR | 0.5 | a grid 2.0 A apart | 299 | 1.3e-02 | +0.646 | +0.645 |
| THR | 0.5 | the atoms, 0.75 A apart | 428 | 1.9e-03 | +0.647 | +0.645 |
| THR | 0.5 | the atoms, 0.5 A apart | 628 | 1.4e-04 | +0.646 | +0.645 |
| THR | 0.5 | 40 molecules | 40 | 3.2e-03 | +0.638 | +0.645 |
| THR | 2 | a grid 2.0 A apart | 299 | 3.8e-01 | +0.667 | +0.653 |
| THR | 2 | the atoms, 0.75 A apart | 428 | 1.8e-02 | +0.669 | +0.653 |
| THR | 2 | the atoms, 0.5 A apart | 628 | 4.2e-03 | +0.659 | +0.653 |
| THR | 2 | 40 molecules | 40 | 6.2e-03 | +0.640 | +0.653 |
| THR | 5 | a grid 2.0 A apart | 299 | 7.0e-01 | +0.617 | +0.627 |
| THR | 5 | the atoms, 0.75 A apart | 428 | 5.8e-02 | +0.661 | +0.627 |
| THR | 5 | the atoms, 0.5 A apart | 628 | 2.0e-02 | +0.647 | +0.627 |
| THR | 5 | 40 molecules | 40 | 8.9e-03 | +0.590 | +0.627 |
| AChE | 0.5 | a grid 2.0 A apart | 248 | 1.1e-02 | +0.531 | +0.529 |
| AChE | 0.5 | the atoms, 0.75 A apart | 317 | 1.1e-04 | +0.529 | +0.529 |
| AChE | 0.5 | the atoms, 0.5 A apart | 451 | 6.0e-06 | +0.529 | +0.529 |
| AChE | 0.5 | 40 molecules | 40 | 3.7e-03 | +0.512 | +0.529 |
| AChE | 2 | a grid 2.0 A apart | 248 | 3.8e-01 | +0.549 | +0.544 |
| AChE | 2 | the atoms, 0.75 A apart | 317 | 1.5e-02 | +0.532 | +0.544 |
| AChE | 2 | the atoms, 0.5 A apart | 451 | 4.9e-03 | +0.543 | +0.544 |
| AChE | 2 | 40 molecules | 40 | 7.4e-03 | +0.479 | +0.544 |
| AChE | 5 | a grid 2.0 A apart | 248 | 6.7e-01 | +0.503 | +0.532 |
| AChE | 5 | the atoms, 0.75 A apart | 317 | 6.0e-02 | +0.493 | +0.532 |
| AChE | 5 | the atoms, 0.5 A apart | 451 | 3.0e-02 | +0.530 | +0.532 |
| AChE | 5 | 40 molecules | 40 | 9.9e-03 | +0.525 | +0.532 |


## What it saves

The memory of a fit is the rank squared whatever the number of molecules, because the features are accumulated a block at a time and never held at once.

| molecules | rank | exact, seconds | low rank, seconds | exact, MB | low rank, MB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 820 | 0.0 | 0.1 | 0.1 | 5.4 |
| 100 | 920 | 0.2 | 0.1 | 0.5 | 6.8 |
| 200 | 1095 | 0.7 | 0.1 | 1.9 | 9.6 |
| 361 | 1275 | 2.3 | 0.3 | 6.3 | 13.0 |

## The variance where the landmarks do not reach

20 molecules of THR held out, the landmarks drawn from the rest (30 of them). The exact posterior variance of those molecules averages 0.092. Keeping the exact self-kernel gives 0.105 (off by 0.013); dropping it, which is the plain subset-of-regressors form, gives 0.067 (off by 0.025).

## The verdict

**The implementation is the exact model when nothing is approximated**, to 2e-11 in every quantity it computes, the co-field included.

**What the approximation costs depends on the landmarks, as plan item B6 predicted.** The worst a grid of Gaussian centres does here is -0.004 of q2 against the exact model; the worst landmark molecules do is -0.076. A grid spans the field while its spacing is not much larger than the field's own width, so it is the attenuation and not the number of molecules that decides whether a grid is enough.

**What it saves, measured rather than expected.** At 361 molecules -- the largest aligned set in the repository -- the exact kernels need 6 MB and the low-rank fit needs 13 MB, so at this size the exact path is the smaller of the two. The low-rank memory is the rank squared and does not grow with the molecules at all, so the two cross at about 1275 molecules, which is where a set that needs this path begins. It is also 8 times faster here (0.3 s against 2.3 s), which was not the point of the item but follows from the same thing: a kernel entry against a one-atom centre costs a thirtieth of one against a whole molecule.

**At a narrow field the landmarks have to follow the atoms, and that works.** At an attenuation of 2 and above, a grid 2 A apart reproduces the kernel to 5e-01 while the atoms of the same molecules, thinned to 0.5 to 0.75 A, reach 2e-02 with a landmark count of the same order. That is the case plan item B6 named and left open: a grid covers the box, the atoms cover the field, and only the second survives a Gaussian narrow enough to fall between grid points.

**The variance survives the approximation** because the self-kernel is never approximated: on molecules the landmarks do not reach it is off by 0.013 against 0.025 for the form that drops that term, which is the collapse the item warned about.

**What is still not done**, and was not attempted here: the path is a command of its own (`cmf lowrank`) and not a flag on `build`, so a low-rank model cannot yet be saved, predicted from or opened in the application; the landmarks are a grid or whole molecules, not the atoms of a subset, which is the case between the two that the item names for narrow fields; the inducing Gaussians take the field's own width, where a broader basis might need fewer centres; and the largest aligned set in the repository is 361 molecules, so the saving is shown on a set that does not need it. The last of those is the alignment problem of part C and not this item's to solve.

