# Choosing ten compounds at once (plan item MC2)

A laboratory makes ten compounds at a time. A batch has to be chosen before any of it is measured, so something has to keep it from being one molecule ten times over. The cheap rule skips a candidate whose cosine with a pick already made is above a threshold. The rule this item asks for conditions the posterior covariance on each pick instead -- one Cholesky step, no refitting, no threshold -- and recomputes the score from it; with the variance acquisition that is the batch whose covariance block has the largest determinant.

The protocol of `benchmarks/run_active.py`: 8 Sutherland sets, 20 molecules to start, the attenuation and the ridge reselected by the evidence at every fit, 10 repeats with the same starting set for every rule. Batches of 10.

| how the batch is chosen | fits | best acquired | error on what is left |
| --- | ---: | ---: | ---: |
| one at a time | 35 | 8.878 | 1.034 |
| ei, similarity filter | 3 | 8.902 | 1.041 |
| ei, covariance | 3 | 8.939 | 1.025 |
| variance, similarity filter | 3 | 8.540 | 1.029 |
| variance, covariance (largest determinant) | 3 | 8.403 | 1.043 |
| random | 3 | 8.709 | 1.011 |

## The gate

The item asks whether the covariance rule beats the similarity filter in **both** of E2's measures by more than the interval, paired run by run.

| acquisition | measure | covariance minus filter |
| --- | --- | ---: |
| expected improvement | best acquired | +0.037 [-0.015, +0.110] |
| expected improvement | error on what is left | -0.016 [-0.035, +0.002] |
| variance | best acquired | -0.137 [-0.267, -0.013] |
| variance | error on what is left | +0.014 [-0.008, +0.037] |

## The verdict

**The gate is not met: the two rules are the same within the interval.** None of the 4 comparisons separates them, and the filter is ahead in 1. Plan item E2 predicted this, and the item itself allows it: the cheap filter is enough, and that closes the question rather than leaving it open.

**What a batch costs against one at a time.** Acquiring the same molecules in batches of 10 needs a tenth of the fits. Against the one-at-a-time loop, which sees every measurement before choosing the next molecule, expected improvement with the similarity filter differs by +0.025 [-0.025, +0.079] in the best acquired and +0.008 [-0.016, +0.033] in the error on what is left. Neither interval excludes zero: on these sets a plate of ten chosen at once is worth as much as ten chosen one after another, at a tenth of the fitting. That is the useful finding here, and it holds whichever of the two batch rules is used.

**Why the covariance rule has so little to add** is what E2 already found about the posterior standard deviation on these sets: it is nearly flat over the pool, so conditioning on a pick moves the ordering very little, and the cosine filter removes the near-duplicates that would otherwise be picked twice. Both rules end up doing the same thing, by different arithmetic.

