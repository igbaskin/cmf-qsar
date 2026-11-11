# The graph neural tangent kernel as a kernel view (plan item ND7)

A random subset of COMPAS-1D, the fields of the COMPAS benchmark's set ABC, 10 folds, the ridge by the evidence. Every variant stacks its kernels under the weights of plan item E6 (the evidence); *graph* is the graph neural tangent kernel of the bond graph with 2 blocks, *invariant* the rotation-invariant autocorrelation of item C5. The graph view's share of the joint kernel and the Bayes factor against it doing nothing are given for the variant that holds it with the fields.

| set | n | q2 fields | q2 + invariant | q2 + graph | q2 + both | graph share | Bayes factor, graph | graph vs fields | graph vs invariant | both vs invariant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COMPAS-1D | 2000 | +0.439 | +0.904 | +0.908 | +0.929 | 100% | +1857.5 | +0.470 [+0.438, +0.503] | +0.004 [-0.002, +0.011] | +0.025 [+0.020, +0.030] |

Differences in q2 with their 95 % paired bootstrap intervals over the molecules. Against the fields alone the graph view wins on 1 of the 1 sets and loses on 0 (an interval that excludes zero); against the fields with the invariant kernel it wins on 0 and loses on 0. The plan reads the second comparison: a gain the invariant kernel already gives is not a result.
