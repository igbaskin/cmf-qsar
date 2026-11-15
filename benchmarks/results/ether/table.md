# A cation probe on ether binding energies (plan item EC7)

## Where the probe says the ion sits

The deepest point of the lithium probe field against the position the relaxation gave, on 1050 complexes, searched on a grid of 0.4 A.

| Prediction of the ion position | Distance from the relaxed ion (A) | Within 1.0 A | Within 1.5 A |
|---|---|---|---|
| deepest point of the probe field | 0.36 +/- 0.55 | 97% | 98% |
| centroid of the oxygens | 0.65 +/- 0.74 | 86% | 96% |
| centroid of the heavy atoms | 0.67 +/- 0.80 | 82% | 93% |

## The binding energy

Nested cross-validation, 5 outer folds, the ridge and the field width chosen on 10 inner folds. Mean absolute error over the folds, in eV.

### Li+: 1048 molecules, mean -3.84 eV, standard deviation 0.58

**Folds: random.** Predicting the mean of the set costs 0.454 eV.

| Model | MAE (eV) | R2 |
|---|---|---|
| counts | 0.296 +/- 0.012 | +0.431 |
| counts+homo | 0.254 +/- 0.015 | +0.547 |
| electrostatic | 0.209 +/- 0.019 | +0.770 |
| fields | 0.124 +/- 0.016 | +0.894 |
| probe | 0.140 +/- 0.013 | +0.884 |
| fields+probe | 0.117 +/- 0.014 | +0.910 |

* Paired over the same molecules, fields minus counts: -0.172 eV [-0.191, -0.152], excludes zero.
* Paired over the same molecules, probe minus counts: -0.156 eV [-0.175, -0.139], excludes zero.
* Paired over the same molecules, probe minus the electrostatic field alone: -0.069 eV [-0.078, -0.060], excludes zero.
* Paired over the same molecules, probe minus fields: +0.015 eV [+0.009, +0.022], excludes zero.
* Paired over the same molecules, fields+probe minus counts: -0.179 eV [-0.197, -0.161], excludes zero.

**Folds: one oxygen count held out.** Predicting the mean of the set costs 0.454 eV.

| Model | MAE (eV) | R2 |
|---|---|---|
| counts | 0.284 +/- 0.062 | +0.415 |
| counts+homo | 0.264 +/- 0.034 | +0.521 |
| electrostatic | 0.256 +/- 0.028 | +0.689 |
| fields | 0.186 +/- 0.102 | +0.864 |
| probe | 0.177 +/- 0.056 | +0.857 |
| fields+probe | 0.129 +/- 0.020 | +0.896 |

* Paired over the same molecules, fields minus counts: -0.148 eV [-0.168, -0.129], excludes zero.
* Paired over the same molecules, probe minus counts: -0.145 eV [-0.164, -0.128], excludes zero.
* Paired over the same molecules, probe minus the electrostatic field alone: -0.087 eV [-0.098, -0.076], excludes zero.
* Paired over the same molecules, probe minus fields: +0.003 eV [-0.005, +0.011], includes zero.
* Paired over the same molecules, fields+probe minus counts: -0.168 eV [-0.187, -0.149], excludes zero.

### Na+: 980 molecules, mean -2.78 eV, standard deviation 0.57

**Folds: random.** Predicting the mean of the set costs 0.449 eV.

| Model | MAE (eV) | R2 |
|---|---|---|
| counts | 0.279 +/- 0.015 | +0.442 |
| counts+homo | 0.246 +/- 0.015 | +0.513 |
| electrostatic | 0.248 +/- 0.014 | +0.575 |
| fields | 0.181 +/- 0.008 | +0.688 |
| probe | 0.193 +/- 0.011 | +0.664 |
| fields+probe | 0.171 +/- 0.009 | +0.706 |

* Paired over the same molecules, fields minus counts: -0.098 eV [-0.113, -0.084], excludes zero.
* Paired over the same molecules, probe minus counts: -0.086 eV [-0.100, -0.073], excludes zero.
* Paired over the same molecules, probe minus the electrostatic field alone: -0.055 eV [-0.063, -0.046], excludes zero.
* Paired over the same molecules, probe minus fields: +0.012 eV [+0.004, +0.020], excludes zero.
* Paired over the same molecules, fields+probe minus counts: -0.108 eV [-0.122, -0.094], excludes zero.

**Folds: one oxygen count held out.** Predicting the mean of the set costs 0.449 eV.

| Model | MAE (eV) | R2 |
|---|---|---|
| counts | 0.291 +/- 0.029 | +0.429 |
| counts+homo | 0.254 +/- 0.025 | +0.502 |
| electrostatic | 0.277 +/- 0.041 | +0.533 |
| fields | 0.248 +/- 0.122 | +0.658 |
| probe | 0.243 +/- 0.061 | +0.591 |
| fields+probe | 0.204 +/- 0.056 | +0.677 |

* Paired over the same molecules, fields minus counts: -0.082 eV [-0.098, -0.068], excludes zero.
* Paired over the same molecules, probe minus counts: -0.060 eV [-0.074, -0.047], excludes zero.
* Paired over the same molecules, probe minus the electrostatic field alone: -0.045 eV [-0.055, -0.034], excludes zero.
* Paired over the same molecules, probe minus fields: +0.022 eV [+0.011, +0.033], excludes zero.
* Paired over the same molecules, fields+probe minus counts: -0.095 eV [-0.111, -0.080], excludes zero.

### Mg2+: 982 molecules, mean -11.68 eV, standard deviation 1.58

**Folds: random.** Predicting the mean of the set costs 1.235 eV.

| Model | MAE (eV) | R2 |
|---|---|---|
| counts | 0.505 +/- 0.011 | +0.726 |
| counts+homo | 0.444 +/- 0.022 | +0.763 |
| electrostatic | 0.557 +/- 0.032 | +0.688 |
| fields | 0.266 +/- 0.024 | +0.853 |
| probe | 0.321 +/- 0.023 | +0.821 |
| fields+probe | 0.254 +/- 0.018 | +0.856 |

* Paired over the same molecules, fields minus counts: -0.239 eV [-0.268, -0.214], excludes zero.
* Paired over the same molecules, probe minus counts: -0.184 eV [-0.213, -0.155], excludes zero.
* Paired over the same molecules, probe minus the electrostatic field alone: -0.236 eV [-0.263, -0.210], excludes zero.
* Paired over the same molecules, probe minus fields: +0.055 eV [+0.039, +0.072], excludes zero.
* Paired over the same molecules, fields+probe minus counts: -0.250 eV [-0.278, -0.224], excludes zero.

**Folds: one oxygen count held out.** Predicting the mean of the set costs 1.235 eV.

| Model | MAE (eV) | R2 |
|---|---|---|
| counts | 0.571 +/- 0.124 | +0.718 |
| counts+homo | 0.557 +/- 0.146 | +0.745 |
| electrostatic | 0.754 +/- 0.176 | +0.603 |
| fields | 0.366 +/- 0.130 | +0.821 |
| probe | 0.474 +/- 0.188 | +0.789 |
| fields+probe | 0.279 +/- 0.025 | +0.823 |

* Paired over the same molecules, fields minus counts: -0.201 eV [-0.228, -0.176], excludes zero.
* Paired over the same molecules, probe minus counts: -0.117 eV [-0.148, -0.086], excludes zero.
* Paired over the same molecules, probe minus the electrostatic field alone: -0.258 eV [-0.289, -0.227], excludes zero.
* Paired over the same molecules, probe minus fields: +0.084 eV [+0.063, +0.105], excludes zero.
* Paired over the same molecules, fields+probe minus counts: -0.232 eV [-0.257, -0.208], excludes zero.

## The selectivity between lithium and sodium

The difference of the two binding energies on the 980 molecules that carry both: mean -1.070 eV, standard deviation 0.364, correlation with the oxygen count -0.049. Predicting the mean costs 0.190 eV.

| Model | MAE (eV) | R2 |
|---|---|---|
| counts | 0.188 +/- 0.013 | +0.006 |
| counts+homo | 0.188 +/- 0.012 | +0.012 |
| electrostatic | 0.181 +/- 0.011 | +0.159 |
| fields | 0.175 +/- 0.012 | +0.210 |
| probe | 0.174 +/- 0.013 | +0.205 |
| fields+probe | 0.171 +/- 0.011 | +0.218 |
| two probes | 0.176 +/- 0.013 | +0.192 |

* Paired, two probes minus counts: -0.012 eV [-0.022, -0.003], excludes zero.
* Paired, two probes minus the lithium probe alone: +0.001 eV [-0.000, +0.003], includes zero.

## The three ions as three endpoints

One kernel, the covariance between the endpoints as named. The descriptor covariance takes it from the ion's ionic radius, charge, polarisability.

| Covariance | MAE (eV) | q2 Li+ | q2 Na+ | q2 Mg2+ |
|---|---|---|---|---|
| independent | 0.204 | +0.880 | +0.680 | +0.842 |
| one correlation 0.5 | 0.198 | +0.888 | +0.691 | +0.844 |
| one correlation 0.9 | 0.207 | +0.880 | +0.688 | +0.839 |
| descriptors | 0.188 | +0.893 | +0.698 | +0.852 |

**An ion removed entirely** and predicted from the other two through the descriptor covariance, beside the reference that needs no model: the mean of the ions that were measured.

| Ion removed | n | MAE (eV) | R2 | MAE of the other ions' mean (eV) |
|---|---|---|---|---|
| Li+ | 1048 | 1.484 | -5.807 | 3.499 |
| Na+ | 980 | 0.539 | -0.327 | 4.747 |
| Mg2+ | 982 | refused | refused | 8.315 |

Mg2+ is refused: the charge is the same in both ions that remain, so the design matrix of the mean has two identical rows and the mean of the removed ion is not a quantity the data determine. A divalent ion cannot be predicted from two monovalent ones.

## The oxygens at the ion as a reference

Contacts: the oxygens within 2.5 A of the ion, their mean distance (the nearest oxygen when none is that close) and the sum of 1/d over the oxygens within 3.5 A, beside the three counts of stage 2. At the relaxed ion the description reads the position the data carry, which no field model sees; at the probe minimum it reads the deepest point of the lithium probe field and nothing else. The number of contacts is the same at the two places for 89% of the molecules. The Gaussian kernels choose their width inside each fold. The fields, the probe and the counts are the stored models of stage 2 on the same folds.

### Li+, random (1048 molecules)

| Model | MAE (eV) | R2 |
|---|---|---|
| counts (stage 2) | 0.296 | +0.431 |
| fields (stage 2) | 0.124 | +0.894 |
| probe (stage 2) | 0.140 | +0.884 |
| fields+probe (stage 2) | 0.117 | +0.910 |
| counts, Gaussian | 0.240 +/- 0.023 | +0.635 |
| contacts at the relaxed ion | 0.208 +/- 0.017 | +0.771 |
| contacts at the relaxed ion, Gaussian | 0.177 +/- 0.010 | +0.845 |
| contacts at the probe minimum, Gaussian | 0.194 +/- 0.008 | +0.810 |

* contacts at the relaxed ion, Gaussian minus counts: -0.119 eV [-0.139, -0.100], excludes zero.
* contacts at the relaxed ion, Gaussian minus fields: +0.052 eV [+0.042, +0.062], excludes zero.
* contacts at the relaxed ion, Gaussian minus probe: +0.037 eV [+0.027, +0.046], excludes zero.
* contacts at the probe minimum, Gaussian minus counts: -0.102 eV [-0.121, -0.085], excludes zero.
* contacts at the probe minimum, Gaussian minus fields: +0.069 eV [+0.059, +0.080], excludes zero.
* contacts at the probe minimum, Gaussian minus probe: +0.054 eV [+0.045, +0.063], excludes zero.

### Li+, one oxygen count held out (1048 molecules)

| Model | MAE (eV) | R2 |
|---|---|---|
| counts (stage 2) | 0.284 | +0.415 |
| fields (stage 2) | 0.186 | +0.864 |
| probe (stage 2) | 0.177 | +0.857 |
| fields+probe (stage 2) | 0.129 | +0.896 |
| counts, Gaussian | 0.433 +/- 0.197 | +0.134 |
| contacts at the relaxed ion | 0.206 +/- 0.044 | +0.734 |
| contacts at the relaxed ion, Gaussian | 0.266 +/- 0.120 | +0.611 |
| contacts at the probe minimum, Gaussian | 0.232 +/- 0.053 | +0.767 |

* contacts at the relaxed ion, Gaussian minus counts: -0.018 eV [-0.041, +0.005], includes zero.
* contacts at the relaxed ion, Gaussian minus fields: +0.130 eV [+0.115, +0.145], excludes zero.
* contacts at the relaxed ion, Gaussian minus probe: +0.127 eV [+0.113, +0.142], excludes zero.
* contacts at the probe minimum, Gaussian minus counts: -0.082 eV [-0.101, -0.065], excludes zero.
* contacts at the probe minimum, Gaussian minus fields: +0.066 eV [+0.054, +0.077], excludes zero.
* contacts at the probe minimum, Gaussian minus probe: +0.063 eV [+0.052, +0.073], excludes zero.

### Na+, random (980 molecules)

| Model | MAE (eV) | R2 |
|---|---|---|
| counts (stage 2) | 0.279 | +0.442 |
| fields (stage 2) | 0.181 | +0.688 |
| probe (stage 2) | 0.193 | +0.664 |
| fields+probe (stage 2) | 0.171 | +0.706 |
| counts, Gaussian | 0.258 +/- 0.018 | +0.509 |
| contacts at the relaxed ion | 0.254 +/- 0.010 | +0.539 |
| contacts at the relaxed ion, Gaussian | 0.224 +/- 0.016 | +0.602 |
| contacts at the probe minimum, Gaussian | 0.224 +/- 0.015 | +0.624 |

* contacts at the relaxed ion, Gaussian minus counts: -0.055 eV [-0.069, -0.041], excludes zero.
* contacts at the relaxed ion, Gaussian minus fields: +0.043 eV [+0.033, +0.053], excludes zero.
* contacts at the relaxed ion, Gaussian minus probe: +0.031 eV [+0.021, +0.043], excludes zero.
* contacts at the probe minimum, Gaussian minus counts: -0.055 eV [-0.068, -0.042], excludes zero.
* contacts at the probe minimum, Gaussian minus fields: +0.043 eV [+0.032, +0.052], excludes zero.
* contacts at the probe minimum, Gaussian minus probe: +0.031 eV [+0.021, +0.041], excludes zero.

### Na+, one oxygen count held out (980 molecules)

| Model | MAE (eV) | R2 |
|---|---|---|
| counts (stage 2) | 0.291 | +0.429 |
| fields (stage 2) | 0.248 | +0.658 |
| probe (stage 2) | 0.243 | +0.591 |
| fields+probe (stage 2) | 0.204 | +0.677 |
| counts, Gaussian | 0.440 +/- 0.198 | +0.036 |
| contacts at the relaxed ion | 0.285 +/- 0.031 | +0.506 |
| contacts at the relaxed ion, Gaussian | 0.346 +/- 0.200 | +0.170 |
| contacts at the probe minimum, Gaussian | 0.413 +/- 0.302 | -0.166 |

* contacts at the relaxed ion, Gaussian minus counts: +0.086 eV [+0.066, +0.107], excludes zero.
* contacts at the relaxed ion, Gaussian minus fields: +0.169 eV [+0.148, +0.190], excludes zero.
* contacts at the relaxed ion, Gaussian minus probe: +0.147 eV [+0.128, +0.167], excludes zero.
* contacts at the probe minimum, Gaussian minus counts: +0.144 eV [+0.115, +0.173], excludes zero.
* contacts at the probe minimum, Gaussian minus fields: +0.227 eV [+0.200, +0.254], excludes zero.
* contacts at the probe minimum, Gaussian minus probe: +0.204 eV [+0.177, +0.232], excludes zero.

### Mg2+, random (982 molecules)

| Model | MAE (eV) | R2 |
|---|---|---|
| counts (stage 2) | 0.505 | +0.726 |
| fields (stage 2) | 0.266 | +0.853 |
| probe (stage 2) | 0.321 | +0.821 |
| fields+probe (stage 2) | 0.254 | +0.856 |
| counts, Gaussian | 0.428 +/- 0.024 | +0.787 |
| contacts at the relaxed ion | 0.418 +/- 0.017 | +0.779 |
| contacts at the relaxed ion, Gaussian | 0.379 +/- 0.013 | +0.812 |
| contacts at the probe minimum, Gaussian | 0.400 +/- 0.029 | +0.806 |

* contacts at the relaxed ion, Gaussian minus counts: -0.126 eV [-0.149, -0.104], excludes zero.
* contacts at the relaxed ion, Gaussian minus fields: +0.113 eV [+0.096, +0.131], excludes zero.
* contacts at the relaxed ion, Gaussian minus probe: +0.058 eV [+0.039, +0.077], excludes zero.
* contacts at the probe minimum, Gaussian minus counts: -0.104 eV [-0.127, -0.082], excludes zero.
* contacts at the probe minimum, Gaussian minus fields: +0.135 eV [+0.116, +0.155], excludes zero.
* contacts at the probe minimum, Gaussian minus probe: +0.080 eV [+0.059, +0.100], excludes zero.

### Mg2+, one oxygen count held out (982 molecules)

| Model | MAE (eV) | R2 |
|---|---|---|
| counts (stage 2) | 0.571 | +0.718 |
| fields (stage 2) | 0.366 | +0.821 |
| probe (stage 2) | 0.474 | +0.789 |
| fields+probe (stage 2) | 0.279 | +0.823 |
| counts, Gaussian | 0.600 +/- 0.178 | +0.716 |
| contacts at the relaxed ion | 0.476 +/- 0.087 | +0.754 |
| contacts at the relaxed ion, Gaussian | 0.502 +/- 0.105 | +0.749 |
| contacts at the probe minimum, Gaussian | 0.686 +/- 0.307 | +0.690 |

* contacts at the relaxed ion, Gaussian minus counts: -0.039 eV [-0.070, -0.008], excludes zero.
* contacts at the relaxed ion, Gaussian minus fields: +0.162 eV [+0.140, +0.185], excludes zero.
* contacts at the relaxed ion, Gaussian minus probe: +0.078 eV [+0.049, +0.107], excludes zero.
* contacts at the probe minimum, Gaussian minus counts: +0.056 eV [+0.021, +0.089], excludes zero.
* contacts at the probe minimum, Gaussian minus fields: +0.257 eV [+0.231, +0.284], excludes zero.
* contacts at the probe minimum, Gaussian minus probe: +0.173 eV [+0.141, +0.206], excludes zero.

