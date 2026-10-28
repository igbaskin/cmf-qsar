# Kernel logistic regression against the other classifiers (plan item E9)

The protocol of plan item E8, imported from `run_kfda.py`: the two LIT-PCBA classification sets of plan item E4, one kernel per set (five fields, attenuation 0.5, cosine-normalised), one 5-fold split stratified by class, every learner choosing its own setting inside each fold on a 5-fold cross-validation scored by AUC. Balanced accuracy is given under E8's common threshold rule, because where a learner leaves its own boundary dominates that statistic on these sets.

| set | molecules | actives | learner | AUC | Brier score | balanced accuracy | sensitivity | specificity |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| ESR1_ant | 1587 | 88 | kernel logistic regression (ridge by CV) | 0.630 | 0.0579 | 0.515 | 0.125 | 0.905 |
|  |  |  | GP classifier, logistic likelihood (amplitude by CV) | 0.631 | 0.0613 | 0.515 | 0.125 | 0.905 |
|  |  |  | GP classifier, logistic likelihood (amplitude by the marginal likelihood) | 0.608 | 0.0537 | 0.523 | 0.125 | 0.921 |
|  |  |  | C-SVC (C by CV) | 0.611 | - | 0.529 | 0.114 | 0.945 |
|  |  |  | kernel Fisher discriminant (ridge by CV) | 0.631 | 0.0520 | 0.549 | 0.227 | 0.871 |
|  |  |  | *the share of actives, predicted for every molecule* | 0.500 | 0.0524 | 0.500 | - | - |
| TP53 | 1564 | 64 | kernel logistic regression (ridge by CV) | 0.563 | 0.0460 | 0.508 | 0.250 | 0.765 |
|  |  |  | GP classifier, logistic likelihood (amplitude by CV) | 0.544 | 0.0506 | 0.501 | 0.297 | 0.706 |
|  |  |  | GP classifier, logistic likelihood (amplitude by the marginal likelihood) | 0.589 | 0.0407 | 0.506 | 0.234 | 0.777 |
|  |  |  | C-SVC (C by CV) | 0.604 | - | 0.504 | 0.203 | 0.805 |
|  |  |  | kernel Fisher discriminant (ridge by CV) | 0.579 | 0.0394 | 0.508 | 0.266 | 0.750 |
|  |  |  | *the share of actives, predicted for every molecule* | 0.500 | 0.0392 | 0.500 | - | - |

Averaged over the sets:

| learner | AUC | Brier score | balanced accuracy | sensitivity | specificity |
|---|---:|---:|---:|---:|---:|
| kernel logistic regression (ridge by CV) | 0.596 | 0.0519 | 0.511 | 0.188 | 0.835 |
| GP classifier, logistic likelihood (amplitude by CV) | 0.587 | 0.0559 | 0.508 | 0.211 | 0.806 |
| GP classifier, logistic likelihood (amplitude by the marginal likelihood) | 0.598 | 0.0472 | 0.514 | 0.180 | 0.849 |
| C-SVC (C by CV) | 0.607 | - | 0.517 | 0.158 | 0.875 |
| kernel Fisher discriminant (ridge by CV) | 0.605 | 0.0457 | 0.529 | 0.246 | 0.811 |
| *the share of actives, predicted for every molecule* | 0.500 | 0.0458 | 0.500 | - | - |

### The differences, with 95 % paired bootstrap intervals

The molecules are resampled once per draw and both learners scored on the same resample. A difference whose interval contains zero is not a difference. For the Brier score a negative difference is the better one, because it is an error.

| comparison | metric | ESR1_ant | TP53 | mean |
|---|---|---|---|---|
| klr - gpc | AUC | -0.001 [-0.012, +0.010] | +0.019 [-0.059, +0.085] | +0.009 [-0.035, +0.048] |
| klr - gpc | Brier score | -0.0034 [-0.0056, -0.0010] | -0.0046 [-0.0075, -0.0015] | -0.0040 [-0.0066, -0.0012] |
| klr - gpc | balanced accuracy, common rule | -0.000 [-0.001, +0.000] | +0.006 [-0.042, +0.051] | +0.003 [-0.022, +0.025] |
| klr - gpc-evidence | AUC | +0.022 [-0.004, +0.048] | -0.025 [-0.105, +0.047] | -0.002 [-0.055, +0.048] |
| klr - gpc-evidence | Brier score | +0.0041 [+0.0020, +0.0064] | +0.0053 [+0.0025, +0.0083] | +0.0047 [+0.0023, +0.0074] |
| klr - gpc-evidence | balanced accuracy, common rule | -0.008 [-0.044, +0.029] | +0.002 [-0.032, +0.038] | -0.003 [-0.038, +0.034] |
| klr - svc | AUC | +0.018 [-0.037, +0.074] | -0.040 [-0.086, +0.009] | -0.011 [-0.061, +0.041] |
| klr - svc | balanced accuracy, common rule | -0.015 [-0.048, +0.021] | +0.003 [-0.081, +0.083] | -0.006 [-0.064, +0.052] |
| klr - kfda | AUC | -0.001 [-0.035, +0.033] | -0.016 [-0.096, +0.063] | -0.008 [-0.065, +0.048] |
| klr - kfda | Brier score | +0.0059 [+0.0036, +0.0084] | +0.0066 [+0.0035, +0.0097] | +0.0062 [+0.0035, +0.0091] |
| klr - kfda | balanced accuracy, common rule | -0.034 [-0.078, +0.007] | -0.000 [-0.041, +0.040] | -0.017 [-0.060, +0.024] |

### What "cheaper" is worth, in milliseconds

The plan entry says this learner is cheaper than the Gaussian process by the variance it does not compute. Measured on these kernels: Newton's method takes the same number of steps either way, so the fit saves only the factorisation the variance is read from and the determinant the marginal likelihood needs, while a prediction saves a triangular solve for every molecule scored.

| set | molecules | one fit, klr | one fit, GP | ratio | all molecules scored, klr | all molecules scored, GP | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| ESR1_ant | 1587 | 490 ms | 567 ms | 1.16 | 0.6 ms | 37.9 ms | 61 |
| TP53 | 1564 | 499 ms | 531 ms | 1.06 | 0.3 ms | 30.2 ms | 112 |

So the saving is real but it is at prediction time, not at fit time: the fit gains a few tens of per cent and scoring a library gains an order of magnitude. A screen is where it matters.


### Are the probabilities any good: predicted against observed

Each row is one learner on one set: the share of molecules that turned out active, among those it gave a probability in that range. A calibrated classifier reproduces the bin. Empty cells are bins no molecule fell in, which on a set with one active in twenty is most of the upper range.

| set | learner | 0.0-0.2 | 0.2-0.4 | 0.4-0.6 | 0.6-0.8 | 0.8-1.0 |
|---|---|---:|---:|---:|---:|---:|
| ESR1_ant | kernel logistic regression (ridge by CV) | 0.05 (1501) | 0.09 (55) | 0.09 (22) | 0.11 (9) | - |
| ESR1_ant | GP classifier, logistic likelihood (amplitude by CV) | 0.05 (1321) | 0.11 (224) | 0.05 (38) | 0.25 (4) | - |
| ESR1_ant | GP classifier, logistic likelihood (amplitude by the marginal likelihood) | 0.05 (1526) | 0.10 (60) | 0.00 (1) | - | - |
| ESR1_ant | kernel Fisher discriminant (ridge by CV) | 0.06 (1584) | 0.00 (3) | - | - | - |
| TP53 | kernel logistic regression (ridge by CV) | 0.04 (1515) | 0.00 (26) | 0.07 (14) | 0.25 (4) | 0.00 (5) |
| TP53 | GP classifier, logistic likelihood (amplitude by CV) | 0.04 (1364) | 0.06 (165) | 0.06 (33) | 0.00 (2) | - |
| TP53 | GP classifier, logistic likelihood (amplitude by the marginal likelihood) | 0.04 (1532) | 0.09 (32) | - | - | - |
| TP53 | kernel Fisher discriminant (ridge by CV) | 0.04 (1563) | 0.00 (1) | - | - | - |

