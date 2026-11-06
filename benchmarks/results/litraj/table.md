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

### Can a kernel hold what the forests found?

The kernels of this program are linear. `cmfqsar.invariant.autocorrelation_kernel` rescales the channels, divides every row by its own length and takes the inner product, so ridge regression on it is a linear model in the feature space, while a forest is not. The first guess is therefore that the barrier is not a linear function of these features. A Gaussian kernel on the same rows tests that guess and stays a kernel method: the rows are unit vectors, so the squared distance between two of them is 2 - 2K and the whole family is a function of the matrix already computed.

These rows are fitted more honestly than the rows above. The width and the ridge are chosen inside each training part, on an inner split that holds out whole structures, and the test fold is touched once with the choice already made. The linear kernel is given the same treatment, so the pair can be compared without the ridge search of the first table favouring either. That is why a nested linear row can score below the same kernel in the first table: the first table chose its ridge on the folds it reports.

| description | kernel | q2 | error (eV) | mean absolute error (eV) | widths chosen | against the same kernel in the first table |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| counts about the midpoint | linear | +0.445 | 1.419 | 0.818 | linear | -0.024 [-0.070, +0.003] |
| counts about the midpoint | Gaussian | +0.430 | 1.438 | 0.849 | gamma 0.125, gamma 0.25, gamma 0.5 | -0.038 [-0.080, +0.000] |
| invariant local | linear | +0.572 | 1.246 | 0.736 | linear | -0.029 [-0.048, -0.006] |
| invariant local | Gaussian | +0.623 | 1.169 | 0.653 | gamma 0.125, gamma 0.25, gamma 0.5, gamma 1 | +0.023 [-0.002, +0.053] |

**The guess is part of the answer, and the smaller part.** On the local description the width is worth +0.052 [+0.023, +0.080] over the linear kernel given the same treatment, which is clear of zero but small: the whole distance from that kernel to the random forest on the very same rows is +0.171 in q2, so a curved response to these features accounts for about 30% of it. The Gaussian kernel is still -0.119 [-0.170, -0.071] behind that forest.

On the counts the width is worth nothing at all (-0.015 [-0.060, +0.032]), so this is not a general gain from curving a kernel; it is specific to the description that has many columns.

**What the rest is more likely to be: the metric.** A Gaussian kernel measures one distance over all the columns at once, so the few columns that carry the barrier are diluted by the many that do not, and no single width repairs that. A tree picks columns one at a time and ignores the rest. On this reading the question is which columns count and how they are weighted, which is a different repair from putting a width on the kernel, and it is a reading and not yet a measurement. Item M15 of the plan is where it is recorded.

One number in passing. The nested linear kernel scores -0.029 [-0.048, -0.006] against the same kernel in the first table, which is what choosing the ridge on the folds it reports is worth. Every kernel row of the first table is overstated by about that much and the forest rows by nothing, so the forest's advantage there is understated.

### The window the ion passes through

A barrier is set by the narrowest place along the path: a window of three or four framework atoms that the ion has to pass. The two-point description of an environment carries the distance between two neighbours but not the two sides that meet at the centre, so it does not resolve the angle there, and two different windows can leave the same distance histogram. `cmfqsar.triangles` (plan item C5) resolves it for an environment whose first atom is the centre, and the first atom of a hop environment is a marker at the midpoint, so the triangles it builds are exactly the midpoint with two framework atoms. This stays a linear kernel on the fields, so the co-field survives, which a forest and a Gaussian width do not offer.

The reach, the grid spacing and the exponent are chosen inside each training part, as in the section above. The mixture row searches a convex sum of the two-point and three-distance kernels whose grid includes both ends, so it can fall back on either.

| description | q2 | error (eV) | mean absolute error (eV) | chosen inside the folds |
| --- | ---: | ---: | ---: | --- |
| three-distance | +0.789 | 0.875 | 0.421 | reach 4, step 0.75, power 1, row length kept |
| two-point and three-distance | +0.784 | 0.884 | 0.466 | weight 1 |

The three-distance kernel against:

* the two-point description, nested, on the same environments: **+0.217 [+0.159, +0.286]**
* the random forest on the two-point rows: **+0.047 [-0.004, +0.096]**
* the boosted forest on the two-point rows: **+0.031 [-0.022, +0.083]**
* the random forest on the **raw** two-point rows, which has the row length every normalised kernel here divides out: **-0.055 [-0.090, -0.023]**
* the overlap kernel in the hop frame, the best model of item BT7: **+0.209 [+0.145, +0.274]**

Mixing the two-point kernel back in is worth -0.004 [-0.032, +0.028] over the three-distance kernel alone.

**What this says.** Against the same two-point description fitted the same way, resolving the angle at the midpoint is worth +0.217 and putting a width on the kernel is worth +0.052, so the angle is worth more. It is also the repair that keeps the model a linear kernel on fields, which a forest and a Gaussian width do not.

**Does the gain depend on the grid?** Each setting on its own, with the ridge chosen on the folds it reports, so these are the optimistic kind and are here only to show how flat the choice is:

| setting | channels | q2 | mean absolute error (eV) |
| --- | ---: | ---: | ---: |
| reach 4, step 0.75, power 1 | 63036 | +0.784 | 0.466 |
| reach 4, step 0.75, power 2 | 63036 | +0.761 | 0.474 |
| reach 3, step 0.5, power 1 | 92820 | +0.787 | 0.466 |
| reach 3, step 0.5, power 2 | 92820 | +0.760 | 0.474 |
| reach 5, step 0.75, power 1 | 130424 | +0.785 | 0.466 |
| reach 5, step 0.75, power 2 | 130424 | +0.761 | 0.474 |
| reach 4, step 0.75, power 1, row length kept | 63036 | +0.797 | 0.427 |

The whole span of those settings is 0.036 in q2, so the gain does not rest on the grid being tuned.

**And the row length, which item M15 asks about.** The same description with the row length left in rather than divided out scores q2 +0.797 against +0.787 for the best normalised setting, and the nested arm chose it in at least one fold -- so the row length carries something here as well, and the two findings of this section and the last add together rather than replacing each other.

**One caution about this section.** The range of the grid was placed after a first look at this same sample. Nothing was chosen on a test fold, but the grid itself is not independent of the data, so a fresh set would settle it; how flat the settings are is what makes this a small worry rather than a large one.

### The free volume at the three sites, as a cell and as a field

The strongest single feature of the published models on this benchmark is the **smallest volume of the Voronoi polyhedra** around the three sites of the hop -- its start, its midpoint and its end -- with the largest of the three next in their ranking. That quantity is a minimum over atoms, which looks unlike a field, but it is the narrow-width limit of a ratio of fields. Writing d(i) for the squared distance from a point to atom i less that atom's squared radius, the radical Voronoi cell of the centre is where d(centre) is smallest, and

        w(r) = exp(-beta d(centre)) / sum over i of exp(-beta d(i))

tends to the indicator function of that cell as beta grows. The denominator is a steric field of this program with unit weights and a Gaussian of width 1/sqrt(2 beta), so the volume is the integral of a ratio of fields and the cell is its sharp limit. Both are measured below: the cell is the published feature and a width is this program's version of it.

Each setting on its own, on the four columns the three sites give (their smallest, largest and mean volume, and the midpoint alone). The ridge is chosen on the folds these rows report, so they are the optimistic kind:

| setting | q2 | mean absolute error (eV) | mean volume at the midpoint (A^3) |
| --- | ---: | ---: | ---: |
| Voronoi cell | +0.465 | 0.824 | 18.22 |
| field, beta 32 | +0.465 | 0.823 | 18.22 |
| field, beta 8 | +0.468 | 0.820 | 18.20 |
| field, beta 2 | +0.467 | 0.817 | 17.86 |
| field, beta 0.5 | +0.376 | 0.894 | 15.20 |

**Can a field carry it?** The cell scores +0.465 and the best field version +0.468 (field, beta 8), a difference of -0.003: the field version loses nothing worth having.

With the width and the ridge chosen inside each training part:

| description | q2 | error (eV) | mean absolute error (eV) | chosen inside the folds |
| --- | ---: | ---: | ---: | --- |
| free volume | +0.468 | 1.389 | 0.820 | field, beta 8 |
| free volume with hop length and bottleneck | +0.489 | 1.362 | 0.805 | field, beta 8 |
| three-distance and free volume | +0.799 | 0.854 | 0.418 | weight 0.1, weight 0.5 |

* Four numbers of free volume against the counts reference of rule 1: **-0.000 [-0.074, +0.092]**
* against the hop length with the bottleneck: **+0.289 [+0.238, +0.338]**
* against the whole two-point description fitted the same way: **-0.103 [-0.168, -0.039]**
* Adding the cell to the three-distance kernel: **+0.010 [-0.005, +0.023]**
* That mixture against the random forest on the raw two-point rows: **-0.045 [-0.080, -0.013]**

**What this says.** Four numbers of free volume are level with the counts reference of the first table, which has hundreds of channels (-0.000 [-0.074, +0.092]), so the published feature is a strong descriptor in few terms. But adding it to the three-distance kernel is worth +0.010 [-0.005, +0.023], which does not clear zero: the three-distance block already carries what the free volume says, which is what a description of the triangles around the midpoint should carry. So the published feature is not what this program was missing.

**One caution.** The volumes are integrated on a cubic grid of 0.25 angstrom inside a ball of 3.5 angstrom, which reproduces an exactly known cell to under one per cent but leaves the sharp cell with a little grid noise that the smooth versions mostly avoid. That favours the field versions slightly, so a cell that wins anyway has won against the arrangement.

### A forest on the window, which closes the description side

Two things closed most of the distance to the published models in the sections above. One is the **description**: the angle at the midpoint, and the length of a row. The other is the **fitting method**: a forest instead of a linear kernel on the same rows. The best model so far mixes the two, being a forest given the raw two-point rows, so it is not clear which of them it owes its lead to. The rows below separate them by fitting a forest on the three-distance rows, over the same hops and the same folds. If the better description is worth to a forest what it was worth to the kernel, the description side is still open; if it is worth nothing, what is left is the fitting method alone.

The rows are the ones the winning kernel of the last section is built from: the channels scaled as that kernel scales them, with the row length left in, which is the setting the nested arm chose in every fold. The normalised form is not measured here -- it would pair with a setting that was never chosen, and this feature matrix is 63036 columns wide, which costs hours per arm.

| rows | fitting method | columns | q2 | error (eV) | mean absolute error (eV) |
| --- | --- | ---: | ---: | ---: | ---: |
| raw | random forest | 63036 | +0.842 | 0.758 | 0.356 |
| raw | boosted forest | 63036 | +0.846 | 0.746 | 0.354 |

* Random forest on the rows raw, against the kernel on the same description: **+0.053 [+0.015, +0.091]**
* Boosted forest on the rows raw, against the kernel on the same description: **+0.058 [+0.029, +0.089]**
* The best of them against the random forest on the raw two-point rows: **+0.003 [-0.026, +0.028]**
* The best of them against the best kernel model of this item: **+0.047 [+0.019, +0.083]**

**What this says.** The best forest on the three-distance rows is +0.058 [+0.029, +0.089] against the kernel on the same rows, which is clear of zero: a forest still finds something in this description that a linear kernel on it does not, so the fitting method is part of what remains and resolving the angle at the midpoint has not closed it.

**And the forest that was unaccounted for.** The best forest on these rows is +0.003 [-0.026, +0.028] against the random forest on the raw two-point rows, so the forest that was ahead of every kernel in the sections above is no longer ahead of a forest given this description.

Both model classes learn at a rate of their own, and the rate is what the distance to the published training set has to be crossed at. The two curves below are measured on the same drawn subsets of structures:

| training structures | q2, overlap in the hop frame | q2, invariant local, rows raw, random forest |
| ---: | ---: | ---: |
| 25 | +0.284 | +0.593 |
| 50 | +0.220 | +0.486 |
| 100 | +0.435 | +0.745 |
| 200 | +0.553 | +0.792 |
| 393 | +0.580 | +0.844 |

The forest gains +0.053 in q2 for every doubling of the training set over the last step of its curve, against +0.028 for the overlap model.
