# Laplacian-regularised kernel ridge regression (plan item E10)

Do molecules without a measured activity improve a model? `--method lapkrr` adds the smoothness penalty of manifold regularisation (Belkin, Niyogi & Sindhwani 2006) to kernel ridge regression, so that unlabelled molecules become rows of the kernel and terms of the model's expansion. Produced by `benchmarks/run_lapkrr.py`.

## Part 1: half of the training activities hidden

The eight Sutherland sets. Half of the published training molecules keep their activities; the other half are given to `lapkrr` with no activity. All three models are scored on the published test set, which none of them saw. Every learner chooses its own settings on a 10-fold cross-validation of the molecules it may see, the weight of the penalty among 0, 0.03, 0.1, 0.3, 1, 3 and the ridge among 0.01, 0.03, 0.1, 0.3, 1, 3; the figures are means over 5 halvings and the intervals are paired bootstraps over the test molecules.

| set | training molecules | test molecules | ridge, visible half | lapkrr, hidden half unlabelled | ridge, every activity (ceiling) | lapkrr - visible half | ceiling - visible half | share of the gap recovered |
|---|---:|---:|---:|---:|---:|---|---|---:|
| **ACE** | 76 | 38 | 0.393 | 0.364 | 0.574 | **-0.029** [-0.101, +0.015] | **+0.181** [+0.063, +0.307] | -0.16 |
| **AChE** | 74 | 37 | 0.199 | 0.193 | 0.246 | **-0.006** [-0.047, +0.024] | **+0.048** [-0.314, +0.348] | -0.12 |
| **BZR** | 98 | 49 | -0.185 | -0.218 | -0.021 | **-0.032** [-0.062, -0.007] | **+0.164** [+0.082, +0.260] | -0.20 |
| **COX2** | 188 | 94 | 0.185 | 0.189 | 0.259 | **+0.004** [-0.007, +0.016] | **+0.073** [+0.031, +0.123] | 0.06 |
| **DHFR** | 237 | 124 | 0.506 | 0.506 | 0.578 | **+0.000** [-0.001, +0.001] | **+0.072** [+0.036, +0.116] | 0.00 |
| **GPB** | 44 | 22 | 0.461 | 0.410 | 0.539 | **-0.051** [-0.177, -0.004] | **+0.077** [-0.109, +0.201] | -0.66 |
| **THER** | 51 | 25 | 0.117 | 0.054 | 0.345 | **-0.064** [-0.164, +0.006] | **+0.228** [+0.008, +0.571] | -0.28 |
| **THR** | 59 | 29 | 0.391 | 0.384 | 0.492 | **-0.007** [-0.018, +0.008] | **+0.101** [+0.033, +0.177] | -0.07 |
| **mean of 8 sets** | | | **0.258** | **0.235** | **0.376** | **-0.023** | **+0.118** | |

**The gate's first part fails.** The penalty gains -0.023 of R2pred on average, is ahead of ridge on the visible half on 2 of the 8 sets, and on no set does its interval exclude zero above; on 2 sets the interval excludes zero *below*, that is the penalty measurably loses. The hidden activities themselves are worth +0.118 of R2pred, so the penalty recovers -18% of that gap -- it recovers none of it.

The cross-validation inside the fits chose a penalty weight of zero, that is no penalty at all, in 50% of the 40 fits. Where it chose a weight above zero the model it produced was, on average, worse on the test set than the one with no penalty.

The same models read on the hidden molecules rather than on the test set. These two columns do not measure the same thing: `lapkrr` has the hidden molecules in its expansion and is predicting molecules whose fields it has seen, which is the transductive setting manifold regularisation is usually stated in, while ridge on the visible half has never met them.

| set | ridge, visible half | lapkrr, hidden half unlabelled | ridge, every activity (fitted on them) |
|---|---:|---:|---:|
| **ACE** | 0.386 | 0.418 | 0.848 |
| **AChE** | 0.225 | 0.276 | 0.927 |
| **BZR** | 0.315 | 0.328 | 0.610 |
| **COX2** | 0.521 | 0.525 | 0.690 |
| **DHFR** | 0.588 | 0.588 | 0.774 |
| **GPB** | 0.268 | 0.277 | 0.882 |
| **THER** | 0.162 | 0.132 | 0.818 |
| **THR** | 0.529 | 0.518 | 0.915 |
| **mean** | **0.374** | **0.383** | **0.808** |

The last column is fitted on the molecules it is scored on, so it is a training figure and not a prediction; it is here only to show where the other two stand.

### The same two learners with their settings chosen on the test set

**These numbers cannot be used.** The settings are chosen by looking at the activities being predicted, so they are an upper bound and not a result. They are here to answer one question: when the penalty gains nothing, is that the method failing or the cross-validation failing to find the weight at which it works? If even the best weight on the test set gains nothing, it is the method.

| set | ridge, best ridge on the test set | lapkrr, best settings on the test set | difference |
|---|---:|---:|---|
| **ACE** | 0.499 | 0.499 | **+0.000** [-0.000, +0.000] |
| **AChE** | 0.243 | 0.252 | **+0.010** [-0.021, +0.048] |
| **BZR** | -0.096 | -0.064 | **+0.032** [-0.069, +0.142] |
| **COX2** | 0.247 | 0.257 | **+0.010** [-0.009, +0.033] |
| **DHFR** | 0.527 | 0.537 | **+0.010** [-0.004, +0.026] |
| **GPB** | 0.470 | 0.474 | **+0.004** [-0.007, +0.051] |
| **THER** | 0.240 | 0.241 | **+0.002** [-0.008, +0.016] |
| **THR** | 0.404 | 0.412 | **+0.008** [-0.004, +0.028] |
| **mean** | **0.317** | **0.326** | **0.009** |

Even with the weight chosen on the test set the penalty gains +0.009 of R2pred on average, and the interval excludes zero above on 0 of the 8 sets. **The method is what gains nothing here, not the choice of its weight.**

### Activity cliffs, so that a loss can be read against them

A pair of labelled molecules counts as **similar** when the cosine of their fields is 0.9 or more, and as an **activity cliff** when it is similar and the two activities also differ by 1 log unit or more. The cosine is the measure of plan item F3, applied to two molecules instead of two co-fields; the two cutoffs are this benchmark's choice. The penalty assumes the activity varies smoothly along the data, which is exactly what a cliff breaks.

| set | labelled pairs | similar pairs | activity cliffs | cliffs among the similar pairs | mean activity difference of a similar pair | lapkrr - visible half |
|---|---:|---:|---:|---:|---:|---|
| **ACE** | 2850 | 109 | 48 | 0.440 | 1.30 | **-0.029** [-0.101, +0.015] |
| **AChE** | 2701 | 728 | 335 | 0.460 | 1.10 | **-0.006** [-0.047, +0.024] |
| **BZR** | 4753 | 543 | 121 | 0.223 | 0.59 | **-0.032** [-0.062, -0.007] |
| **COX2** | 17578 | 1544 | 504 | 0.326 | 0.82 | **+0.004** [-0.007, +0.016] |
| **DHFR** | 27966 | 1238 | 382 | 0.309 | 0.81 | **+0.000** [-0.001, +0.001] |
| **GPB** | 946 | 134 | 53 | 0.396 | 0.91 | **-0.051** [-0.177, -0.004] |
| **THER** | 1275 | 112 | 71 | 0.634 | 1.65 | **-0.064** [-0.164, +0.006] |
| **THR** | 1711 | 980 | 413 | 0.421 | 1.00 | **-0.007** [-0.018, +0.008] |

**The cliffs do not explain the loss.** Over the 8 sets the rank correlation between the share of similar pairs that are cliffs and what the penalty gains is -0.33, and between the mean activity difference of a similar pair and that gain -0.38: both weak, and neither in a direction that would let a cliff count predict whether the penalty is safe to use. What does line up is the **size of the set**: the rank correlation between the number of labelled molecules and the gain is +0.74. The penalty loses most on the smallest sets and nothing at all on the largest, which points at the choice of its weight on few molecules rather than at the smoothness assumption.


### How local the penalty should be

Part 1 repeated at several sizes of the neighbourhood; 0 keeps every pair, which is the untruncated penalty the plan entry warns against.

| set | 5 neighbours | 10 neighbours | 20 neighbours | every pair |
|---|---:|---:|---:|---:|
| **DHFR** | 0.506 | 0.506 | 0.506 | 0.514 |
| **GPB** | 0.452 | 0.410 | 0.428 | 0.419 |

## Part 2: the library unlabelled (screening)

The two LIT-PCBA sets of plan item E4, under the protocol of `benchmarks/run_kfda.py`: one 5-fold split stratified by class, each learner choosing its setting inside every fold by out-of-fold AUC, 95 % paired bootstrap intervals over the molecules. In every fold the held-out molecules are given to `lapkrr` with no label, which is what a screen offers. Enrichment at the top of the ranking is the measure the gate names; AUC and the balanced accuracy at the common cut are beside it.

**ESR1_ant**: 88 actives among 1587 molecules (5.5% active).

| learner | AUC | enrichment in the top 1 % | enrichment in the top 5 % | BEDROC (alpha 20) | balanced accuracy, common cut |
|---|---:|---:|---:|---:|---:|
| lapkrr, the held-out fold unlabelled | 0.619 | 2.25 | 2.28 | 0.158 | 0.540 |
| kernel ridge regression on +1/-1 labels | 0.619 | 2.25 | 2.28 | 0.158 | 0.540 |
| C-SVC, the default classifier | 0.611 | 1.13 | 2.28 | 0.168 | 0.529 |

The folds chose a penalty weight of zero, that is no penalty at all, in 100% of the 5 fits. `lapkrr` is therefore the same model as `krr` here, which is why every difference between them is exactly zero.

| difference | AUC | enrichment in the top 1 % | enrichment in the top 5 % | BEDROC (alpha 20) | balanced accuracy, common cut |
|---|---|---|---|---|---|
| lapkrr - krr | **+0.000** [+0.000, +0.000] | **+0.00** [+0.00, +0.00] | **+0.00** [+0.00, +0.00] | **+0.000** [+0.000, +0.000] | **+0.000** [+0.000, +0.000] |
| lapkrr - svc | **+0.008** [-0.041, +0.057] | **+1.13** [+0.00, +3.81] | **+0.00** [-1.48, +1.53] | **-0.010** [-0.078, +0.057] | **+0.010** [-0.022, +0.049] |

**TP53**: 64 actives among 1564 molecules (4.1% active).

| learner | AUC | enrichment in the top 1 % | enrichment in the top 5 % | BEDROC (alpha 20) | balanced accuracy, common cut |
|---|---:|---:|---:|---:|---:|
| lapkrr, the held-out fold unlabelled | 0.588 | 0.00 | 1.57 | 0.109 | 0.507 |
| kernel ridge regression on +1/-1 labels | 0.569 | 1.53 | 1.88 | 0.106 | 0.519 |
| C-SVC, the default classifier | 0.604 | 3.05 | 2.82 | 0.172 | 0.504 |

The folds chose a penalty weight of zero, that is no penalty at all, in 40% of the 5 fits.

| difference | AUC | enrichment in the top 1 % | enrichment in the top 5 % | BEDROC (alpha 20) | balanced accuracy, common cut |
|---|---|---|---|---|---|
| lapkrr - krr | **+0.019** [-0.005, +0.045] | **-1.53** [-3.76, +1.66] | **-0.31** [-1.02, +0.57] | **+0.003** [-0.033, +0.031] | **-0.013** [-0.057, +0.037] |
| lapkrr - svc | **-0.016** [-0.084, +0.051] | **-3.05** [-8.15, +0.00] | **-1.25** [-2.67, +0.77] | **-0.063** [-0.151, +0.022] | **+0.002** [-0.045, +0.055] |

**No enrichment is lost that the interval can resolve**: no difference in AUC, enrichment or BEDROC against either reference excludes zero below it.

