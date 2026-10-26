# Screening against measured inactives

LIT-PCBA, AVE-unbiased, embedded and posed by `examples/make_litpcba_examples.py`: the actives and the inactives placed by one rule, maximum field similarity to the target's crystal ligand, which knows nothing about which of them is active. The one-class methods and the fingerprint see the training actives alone; the supervised classifier and the PU model are also offered half of the inactives, and every method is scored on the held-out actives against the other half.

| set | actives | inactives | AUC | BEDROC(20) | EF@1 % | EF@5 % | method |
|---|---:|---:|---:|---:|---:|---:|---|
| ESR1_ant | 27 | 750 | 0.536 | 0.098 | 3.6 | 1.5 | distance to the centre of the actives |
| ESR1_ant | 27 | 750 | 0.515 | 0.055 | 0.0 | 0.7 | distance to the k-th active |
| ESR1_ant | 27 | 750 | 0.542 | 0.065 | 3.6 | 0.7 | one-class Gaussian process |
| ESR1_ant | 27 | 750 | 0.537 | 0.041 | 0.0 | 0.0 | distance from the kernel-PCA subspace |
| ESR1_ant | 27 | 750 | 0.668 | 0.273 | 3.6 | 5.2 | ECFP4 Tanimoto to the actives |
| ESR1_ant | 27 | 750 | 0.544 | 0.173 | 0.0 | 3.0 | classifier, inactives = negative |
| ESR1_ant | 27 | 750 | 0.662 | 0.174 | 7.2 | 2.2 | classifier, inactives = unlabelled (bagging PU) |
| TP53 | 20 | 750 | 0.365 | 0.047 | 0.0 | 1.0 | distance to the centre of the actives |
| TP53 | 20 | 750 | 0.382 | 0.042 | 0.0 | 0.0 | distance to the k-th active |
| TP53 | 20 | 750 | 0.546 | 0.059 | 0.0 | 1.0 | one-class Gaussian process |
| TP53 | 20 | 750 | 0.444 | 0.136 | 0.0 | 3.0 | distance from the kernel-PCA subspace |
| TP53 | 20 | 750 | 0.701 | 0.355 | 14.4 | 7.1 | ECFP4 Tanimoto to the actives |
| TP53 | 20 | 750 | 0.613 | 0.201 | 4.8 | 4.1 | classifier, inactives = negative |
| TP53 | 20 | 750 | 0.715 | 0.136 | 4.8 | 2.0 | classifier, inactives = unlabelled (bagging PU) |

Averaged over the sets:

| method | AUC | BEDROC(20) | EF@1 % | EF@5 % |
|---|---:|---:|---:|---:|
| distance to the centre of the actives | 0.451 | 0.073 | 1.8 | 1.2 |
| distance to the k-th active | 0.449 | 0.049 | 0.0 | 0.4 |
| one-class Gaussian process | 0.544 | 0.062 | 1.8 | 0.9 |
| distance from the kernel-PCA subspace | 0.490 | 0.088 | 0.0 | 1.5 |
| ECFP4 Tanimoto to the actives | 0.685 | 0.314 | 9.0 | 6.1 |
| classifier, inactives = negative | 0.579 | 0.187 | 2.4 | 3.5 |
| classifier, inactives = unlabelled (bagging PU) | 0.689 | 0.155 | 6.0 | 2.1 |

The ceiling for EF@1 % here is 33.6: the top percent of a library this size cannot hold more actives than it has places.

**The fields do not beat the fingerprint.** The best one-class description reaches 0.544 AUC against 0.685 for the maximum ECFP4 Tanimoto to the same actives, which is the comparison any three-dimensional method must give its reader and which this repository had not made against measured inactives before.

**And not with every parameter at its best setting either.** Every setting that can change the order of the ranking was tried -- the field attenuation over 0.3, 0.5, 1 and each description's own parameter, 45 fits in all -- and the value that scores best *on the test set* kept, which no fair protocol could pick, so this is an upper bound rather than a result. It lifts the best description only to 0.574 AUC (one-class Gaussian process, EF@1 % 3.6, averaged over each set's own best setting: ESR1_ant at alpha=0.3, lam=0.01, score=mean_over_sd; TP53 at alpha=0.3, lam=0.01, score=mean), still short of the fingerprint at its single default setting. The gap is not a badly chosen parameter.

nu is deliberately absent from that sweep. It places the threshold so that a fraction nu of the training actives falls outside the description, which shifts every decision value by the same constant: AUC, BEDROC and the enrichment factors cannot depend on it, and a sweep over it measures nothing.

**Some of them are worse than chance** -- distance to the centre of the actives scores 0.365 on TP53, which means the held-out actives look *less* typical of the training actives than the measured inactives do. That is not a bug; it is what the AVE debiasing was designed to produce. The embedding removes the analogue bias by construction, so a method whose whole content is 'this molecule resembles the actives' has had its signal taken away, and with it any reason for the result to stay above 0.5.

**Treating the inactives as unlabelled is worth +0.110 AUC and +3.6 in EF@1 %** against calling them negative. The bagging has an effect: an unlabelled molecule that looks like the actives is called negative in few bags and scores high in the rest. And it is not the kernel that is failing: the same CMF kernel, used discriminatively, reaches 0.689 against the fingerprint's 0.685. What fails against measured inactives is the one-class *description* of the actives, not the field similarity it is built from.

## What the posing is worth, which is most of the result

The background sets elsewhere in this repository are posed to maximise their field similarity *to the actives*, so that no decoy can be rejected for having been posed badly. The same molecules posed both ways -- against the crystal ligand, and against the actives -- say what that choice is worth:

| set | method | AUC, posed by the crystal ligand | AUC, posed to resemble the actives | difference |
|---|---|---:|---:|---:|
| TP53 | distance to the centre of the actives | 0.365 | 0.505 | +0.140 |
| TP53 | distance to the k-th active | 0.382 | 0.518 | +0.136 |
| TP53 | one-class Gaussian process | 0.546 | 0.735 | +0.190 |
| TP53 | distance from the kernel-PCA subspace | 0.444 | 0.540 | +0.096 |
| TP53 | ECFP4 Tanimoto to the actives | 0.701 | 0.701 | +0.000 |
| TP53 | classifier, inactives = negative | 0.613 | 0.637 | +0.024 |
| TP53 | classifier, inactives = unlabelled (bagging PU) | 0.715 | 0.783 | +0.067 |

**The generous placement is worth +0.141 AUC to the field descriptions and +0.000 to the fingerprint**, which never looks at a coordinate and so returns exactly the same number. The direction deserves attention, because it is not the obvious one. Posing a decoy where it most resembles the actives ought to make it harder to reject, and it does -- but the pose search also *fails* on a molecule whose field cannot be made to match, and the similarity it ends up with is then a shape-matching score in its own right. Optimising every candidate's pose against the actives turns the alignment into part of the scoring function, which is what shape-based screening does on purpose and what a field model measured this way is getting without cost.

Two consequences. A screening figure measured against a similarity-posed background is a statement about the posing as much as about the model, so the sets built here are posed by the crystal ligand instead, by the same rule as the actives. And the generous condition leaks information: `--ref-max` samples the posed actives without regard to the train/test split, so the decoys' poses were optimised against molecules that include the held-out actives. Both are reasons to read the crystal-posed table as the result and this one as its control.

A caveat that no amount of care removes: a real screening run has to generate and choose poses for molecules nobody has placed, and will do worse than any of this.
