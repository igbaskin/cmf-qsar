# The hop models on the published nebBVSE122k benchmark (the external check of item BT7)

`python benchmarks/run_litraj.py`, reading `benchmarks/results/litraj/prepared.json` (written by `python benchmarks/prepare_litraj.py --fetch`), seed 0, 1000 bootstrap draws.

The collection is nebBVSE122k of Dembitskiy et al., npj Comput. Mater. 11, 131 (2025), MIT licence, https://github.com/AIRI-Institute/LiTraj: 122421 bond-valence barriers of single Li hops in 11290 structures of the Materials Project over 1644 chemical systems. Its label is the barrier of one hop on a rigid framework, as in item BT7, but relaxed with the nudged elastic band method instead of taken as the minimax path on a grid.

A kernel model holds one number per pair of samples, so this run reads a sample: 400 structures drawn at random with at most 6 of their hops each, which leaves 1801 hops in 393 structures. Hop lengths 2.14 to 9.63 A. 10 folds holding out whole structures; environments within 6 A of the midpoint. Fields: electrostatic, steric, prop:electronegativity, cif:B, cif:C, cif:Co, cif:Cr, cif:F, cif:Fe, cif:Li, cif:Mn, cif:Ni, cif:O, cif:P, cif:Si, cif:V, cif:other.

**The barriers of this sample spread more widely than those of the public set of item BT7**: standard deviation 1.905 eV against 0.807 eV, mean 1.448 eV, because the collection covers every chemistry that holds lithium rather than ternary lithium oxides alone. q2 is measured against the spread of the sample, so the same error in eV gives a higher q2 here: **the error in eV is the column to compare with item BT7**. Against the published models of the next section both columns hold, because this sample and the whole collection spread almost alike (1.90 eV against 1.97 eV).

| descriptor | envelope width rho | q2 | error (eV) | mean absolute error (eV) | ridge | against the counts | against length and bottleneck | against the two end sites |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| counts about the midpoint | -- | +0.468 | 1.389 | 0.812 | 0.01 | -- | -- | -- |
| hop length and bottleneck | -- | +0.179 | 1.725 | 1.050 | 1 | -- | -- | -- |
| the two end sites (M5), cutoff 6 A | -- | +0.114 | 1.793 | 1.075 | 1 | -- | -- | -- |
| invariant local, rho 0.25 | 0.25 | +0.600 | 1.204 | 0.726 | 0.1 | +0.132 [+0.056, +0.209] | +0.421 [+0.345, +0.487] | +0.486 [+0.413, +0.549] |
| rotation-averaged overlap, rho 0.25 | 0.25 | +0.580 | 1.234 | 0.730 | 1 | +0.112 [+0.036, +0.197] | +0.401 [+0.341, +0.454] | +0.466 [+0.397, +0.529] |

Overlap in the hop frame against the invariant kernel at the same width: -0.020 [-0.073, +0.039].

Learning curve of the rotation-averaged overlap model, three random draws of the training structures at each size. Each draw has a label spread of its own, so the error in eV is not comparable from row to row and q2 is the column to read:

| training structures | q2 | error (eV) |
| ---: | ---: | ---: |
| 25 | +0.284 | 1.159 |
| 50 | +0.220 | 1.186 |
| 100 | +0.435 | 1.425 |
| 200 | +0.553 | 1.350 |
| 393 | +0.580 | 1.234 |

## What the published models reach on the whole collection

Table 2 of the paper, on its test part (13167 hops of 1172 structures, none of them in the training part), fitted on 96849 hops of 8945 structures:

| model | mean absolute error (eV) | error (eV) | R2 |
| --- | ---: | ---: | ---: |
| M3GNet (graph network, dummy atom at the hop midpoint) | 0.21 | 0.44 | 0.95 |
| XGBRF (gradient-boosted forest, Voronoi features of the hop) | 0.26 | 0.49 | 0.94 |
| RF (random forest, the same features) | 0.26 | 0.50 | 0.94 |
| Nequip (graph network) | 0.25 | 0.52 | 0.93 |
| KNN (nearest neighbours, the same features) | 0.27 | 0.58 | 0.92 |
| Allegro (graph network) | 0.38 | 0.65 | 0.89 |

Those models are fitted on about forty times as many hops as the sample above, which is the one difference the protocol does not remove, so the learning curve is what says how much of the distance the size of the training set explains.

Between 200 and 393 training structures the overlap model gains +0.027 in q2, which is +0.028 for every doubling of the training set. The training part of the collection holds 8945 structures, 4.5 doublings beyond this sample.
That rate carries the model to about q2 +0.71, short of the +0.95 the graph network reaches. A learning curve is not a law and may bend either way, but at the rate measured here the size of the training set does not account for the distance.

## The same descriptions, fitted the way the published models are fitted

The two published models that come closest to the graph network are a random forest and a gradient-boosted forest on hand-made features of the hop, not kernel models. The table at the top therefore mixes two questions: whether this program's **description** of a hop is behind, or whether fitting a description with a **kernel** is. The rows below fit a forest on this program's own descriptions, over the same hops and the same folds, which separates the two. The random forest is the same kind of model the paper reports; the boosted forest is the histogram boosting of scikit-learn, standing in for the paper's XGBoost.

The comparison favours the kernel. Every kernel row above is given the ridge that maximises the same out-of-fold q2 the row reports, so its figure is optimistic by whatever that search is worth; the forests are fitted at fixed settings and search for nothing.

Both kernels here divide every feature row by its own length, so a kernel entry is the cosine of the angle between two rows and the length of a row is divided out. A forest handed the raw rows would be handed more than the kernel ever had, and the comparison would no longer be about the fitting method. The **rows** column says which was given: *as the kernel* is the arm that compares the two fitting methods on one description, and *raw* asks separately what the row length is worth. The geometry kernel divides out no row length, so it has only one form.

| description | rows | columns | fitted with | q2 | error (eV) | mean absolute error (eV) | against the same description fitted with a kernel |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| counts about the midpoint | as the kernel | 399 | random forest | +0.559 | 1.264 | 0.643 | +0.091 [+0.024, +0.164] |
| counts about the midpoint | as the kernel | 399 | boosted forest | +0.454 | 1.408 | 0.797 | -0.015 [-0.090, +0.079] |
| counts about the midpoint | raw | 399 | random forest | +0.561 | 1.262 | 0.626 | +0.093 [+0.029, +0.166] |
| counts about the midpoint | raw | 399 | boosted forest | +0.440 | 1.425 | 0.809 | -0.028 [-0.107, +0.066] |
| hop length and bottleneck | as the kernel | 2 | random forest | +0.186 | 1.719 | 0.916 | +0.007 [-0.116, +0.103] |
| hop length and bottleneck | as the kernel | 2 | boosted forest | +0.202 | 1.702 | 0.916 | +0.023 [-0.094, +0.119] |
| invariant local | as the kernel | 2601 | random forest | +0.742 | 0.967 | 0.481 | +0.142 [+0.091, +0.198] |
| invariant local | as the kernel | 2601 | boosted forest | +0.758 | 0.937 | 0.488 | +0.158 [+0.110, +0.212] |
| invariant local | raw | 2601 | random forest | +0.844 | 0.753 | 0.374 | +0.243 [+0.188, +0.305] |
| invariant local | raw | 2601 | boosted forest | +0.840 | 0.761 | 0.379 | +0.240 [+0.183, +0.305] |

The best of these forests against the best model of item BT7, the overlap in the hop frame: +0.263 [+0.212, +0.319].

**What these rows say.** The local description is the one the field kernels of this program are built on, and it carries two findings, which the rows column keeps apart.

*The fitting method is worth a large part of the distance.* On exactly the rows the kernel is built from -- the same description, nothing added -- the boosted forest reaches q2 +0.758 where the kernel reaches +0.600, and the mean absolute error falls from 0.726 to 0.488 eV (+0.158 [+0.110, +0.212]). So this program's description of a hop was not what was behind the published models on this set; fitting it with a kernel was.

*The cosine normalisation throws away something that carries the barrier.* Handing the forest the raw rows as well, so that it can see how large an environment's features are and not only their direction, adds a further +0.086 in q2 and takes the mean absolute error down to 0.374 eV. The kernel cannot use that length by construction: it divides every row by it. This is a property of the kernel of this program, not of the hop models, so it is worth testing wherever that kernel is used.

Set beside the published models, this program's description of a hop fitted with a forest reaches a mean absolute error of 0.374 eV on 1801 hops, against 0.21 eV for the graph network and 0.26 eV for the published forest, both fitted on 96849 hops.

Both model classes learn at a rate of their own, and the rate is what the distance to the published training set has to be crossed at. The two curves below are measured on the same drawn subsets of structures:

| training structures | q2, overlap in the hop frame | q2, invariant local, rows raw, random forest |
| ---: | ---: | ---: |
| 25 | +0.284 | +0.593 |
| 50 | +0.220 | +0.486 |
| 100 | +0.435 | +0.745 |
| 200 | +0.553 | +0.792 |
| 393 | +0.580 | +0.844 |

The forest gains +0.053 in q2 for every doubling of the training set over the last step of its curve, against +0.028 for the overlap model.
