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

Between 200 and 393 training structures the overlap model gains +0.027 in q2, which is +0.028 for every doubling of the training set. The training part of the collection holds 8945 structures, 4.5 doublings beyond this sample, so that rate carries the model to about q2 +0.71, short of the +0.95 the graph network reaches. A learning curve is not a law and may bend either way, but at the rate measured here the size of the training set does not account for the distance.
