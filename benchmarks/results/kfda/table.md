# Kernel Fisher discriminant analysis against the other classifiers (plan item E8)

## The gate: the classification sets of plan item E4

LIT-PCBA, measured actives against measured inactives, posed by `examples/make_litpcba_examples.py`. One kernel per set (five fields, attenuation 0.5, cosine-normalised), one 5-fold split stratified by class, and each learner choosing its own setting on a 5-fold cross-validation of the training molecules of every fold, scored by AUC. Balanced accuracy, AUC and the Brier score are computed on the pooled out-of-fold predictions.

Two balanced accuracies are given. *Own boundary* is where each learner leaves its decision boundary: the discriminant at the cut its own calibration chose, the support-vector classifier at the margin its quadratic program found, kernel ridge regression at zero on the labels +1 and -1. *Common rule* gives all four the same boundary, the best balanced accuracy on out-of-fold scores of the training molecules of each fold. On a set with one active in twenty a boundary left at zero calls nearly every molecule inactive, so only the second column compares learners rather than conventions.

| set | molecules | actives | learner | balanced accuracy, own boundary | balanced accuracy, common rule | AUC | sensitivity, common rule | specificity, common rule | Brier score |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ESR1_ant | 1587 | 88 | kernel Fisher discriminant (ridge by CV) | 0.612 | 0.549 | 0.631 | 0.227 | 0.871 | 0.0520 |
|  |  |  | C-SVC (C by CV) | 0.499 | 0.529 | 0.611 | 0.114 | 0.945 | - |
|  |  |  | Gaussian-process classifier (amplitude by CV) | 0.504 | 0.555 | 0.604 | 0.352 | 0.758 | 0.0630 |
|  |  |  | kernel ridge regression on +1/-1 labels (ridge by CV) | 0.500 | 0.531 | 0.627 | 0.182 | 0.879 | - |
|  |  |  | *the share of actives, predicted for every molecule* | 0.500 | 0.500 | 0.500 | - | - | 0.0524 |
| TP53 | 1564 | 64 | kernel Fisher discriminant (ridge by CV) | 0.552 | 0.508 | 0.579 | 0.266 | 0.750 | 0.0394 |
|  |  |  | C-SVC (C by CV) | 0.531 | 0.504 | 0.604 | 0.203 | 0.805 | - |
|  |  |  | Gaussian-process classifier (amplitude by CV) | 0.496 | 0.495 | 0.531 | 0.203 | 0.787 | 0.0784 |
|  |  |  | kernel ridge regression on +1/-1 labels (ridge by CV) | 0.491 | 0.528 | 0.572 | 0.344 | 0.711 | - |
|  |  |  | *the share of actives, predicted for every molecule* | 0.500 | 0.500 | 0.500 | - | - | 0.0392 |

Averaged over the sets:

| learner | balanced accuracy, own boundary | balanced accuracy, common rule | AUC | sensitivity, common rule | specificity, common rule | Brier score |
|---|---:|---:|---:|---:|---:|---:|
| kernel Fisher discriminant (ridge by CV) | 0.582 | 0.529 | 0.605 | 0.246 | 0.811 | 0.0457 |
| C-SVC (C by CV) | 0.515 | 0.517 | 0.607 | 0.158 | 0.875 | - |
| Gaussian-process classifier (amplitude by CV) | 0.500 | 0.525 | 0.567 | 0.278 | 0.773 | 0.0707 |
| kernel ridge regression on +1/-1 labels (ridge by CV) | 0.495 | 0.529 | 0.600 | 0.263 | 0.795 | - |
| *the share of actives, predicted for every molecule* | 0.500 | 0.500 | 0.500 | - | - | 0.0458 |

The Brier score is the mean squared error of the probabilities: 0 is perfect and 0.25 is no better than chance. The support-vector and ridge rows have none, because a decision value is not a probability. The last row of each set is the reference a Brier score has to be read against: a model that predicts the share of actives for every molecule and says nothing else. On a set with one active in twenty that model already scores a small Brier score, so a small Brier score is not by itself evidence of a useful probability.

### The differences, with 95 % paired bootstrap intervals

The molecules are resampled once per draw and both learners scored on the same resample. A difference whose interval contains zero is not a difference.

| comparison | metric | ESR1_ant | TP53 | mean |
|---|---|---|---|---|
| kfda - svc | balanced accuracy, own boundary | +0.112 [+0.058, +0.165] | +0.021 [-0.048, +0.091] | +0.067 [+0.005, +0.128] |
| kfda - svc | balanced accuracy, common rule | +0.020 [-0.013, +0.055] | +0.004 [-0.081, +0.086] | +0.012 [-0.047, +0.070] |
| kfda - svc | AUC | +0.020 [-0.032, +0.071] | -0.025 [-0.110, +0.059] | -0.003 [-0.071, +0.065] |
| kfda - gpc | balanced accuracy, own boundary | +0.108 [+0.053, +0.161] | +0.056 [-0.007, +0.118] | +0.082 [+0.023, +0.140] |
| kfda - gpc | balanced accuracy, common rule | -0.006 [-0.060, +0.045] | +0.013 [-0.014, +0.045] | +0.003 [-0.037, +0.045] |
| kfda - gpc | AUC | +0.027 [-0.017, +0.071] | +0.048 [-0.046, +0.146] | +0.038 [-0.032, +0.108] |
| kfda - gpc | Brier score | -0.011 [-0.014, -0.008] | -0.039 [-0.044, -0.034] | -0.025 [-0.029, -0.021] |
| kfda - krr | balanced accuracy, own boundary | +0.112 [+0.056, +0.163] | +0.062 [+0.003, +0.123] | +0.087 [+0.029, +0.143] |
| kfda - krr | balanced accuracy, common rule | +0.019 [-0.018, +0.057] | -0.020 [-0.115, +0.077] | -0.001 [-0.066, +0.067] |
| kfda - krr | AUC | +0.003 [-0.028, +0.034] | +0.007 [-0.075, +0.091] | +0.005 [-0.051, +0.063] |

For the Brier score a negative difference is the better one, because it is an error; for the other two a positive difference is.

### The co-field of the discriminant against the support-vector classifier's

Both classifiers fitted on all the molecules of a set, their co-fields compared by the cosine of plan item F3. The support-vector classifier has no co-field in the sense of rule 2 -- its coefficients are not a linear map of the labels -- so this cosine compares the discriminant's picture with the kernel expansion the support-vector classifier happens to produce, which is why it is read as a check that the two ask the chemistry for the same thing and not as a validation.

| set | overall cosine | electrostatic | steric | hydrophobic | hba | hbd |
|---|---:|---:|---:|---:|---:|---:|
| ESR1_ant | 0.681 | 0.634 | 0.785 | 0.672 | 0.652 | 0.737 |
| TP53 | 0.233 | 0.292 | 0.285 | 0.030 | 0.324 | 0.206 |

## Where to cut the score: the two threshold rules

Eight balanced two-class problems from the Sutherland sets, a molecule being active when its activity is at or above the median of its set. The direction and the ridge are identical in the two columns -- only the bias differs -- so the AUC is the same and only the balanced accuracy can move.

| set | molecules | median | midpoint of the class means | best out-of-fold cut | difference, midpoint - out-of-fold |
|---|---:|---:|---:|---:|---|
| ACE | 114 | 6.39 | 0.842 | 0.851 | -0.009 [-0.045, +0.031] |
| AChE | 111 | 6.84 | 0.739 | 0.747 | -0.009 [-0.040, +0.020] |
| BZR | 147 | 7.89 | 0.701 | 0.708 | -0.007 [-0.046, +0.029] |
| COX2 | 282 | 6.89 | 0.815 | 0.827 | -0.011 [-0.043, +0.022] |
| DHFR | 361 | 6.34 | 0.845 | 0.820 | +0.025 [+0.002, +0.049] |
| GPB | 66 | 2.45 | 0.727 | 0.773 | -0.045 [-0.131, +0.030] |
| THER | 76 | 5.36 | 0.763 | 0.711 | +0.053 [-0.011, +0.118] |
| THR | 88 | 6.69 | 0.727 | 0.750 | -0.023 [-0.069, +0.020] |
| **mean** | | | **0.770** | **0.773** | **-0.003** |

The midpoint is ahead on 2 of the 8 sets and the out-of-fold cut on 6, a mean difference of -0.003 balanced accuracy. The interval contains zero on 7 of the 8 sets, so the two rules cannot be told apart here: searching for the cut gains nothing measurable over the Fisher threshold, and loses nothing either.

