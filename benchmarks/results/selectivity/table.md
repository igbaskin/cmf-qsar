## Selectivity fields: the AMPA / glycine-NMDA set of the 2003 paper

57 tricyclic antagonists (52 training, 5 test; the training set carries two compounds twice, as published) superimposed on the tricyclic core. Three single-endpoint models per setting, alpha and lambda chosen by cross-validation; `comfa` is the two CoMFA fields with leave-one-out, `default` the five fields with ten folds, `file` the five fields with the Sybyl charges of the source files (unreliable for sixteen compounds); `-dedup` removes the two duplicates. The paper's CoMFA row is Table 1 of the paper (leave-one-out PLS, steric and electrostatic fields).

| setting | endpoint | alpha | lambda | r2 | q2 | RMSEcv | R2pred (5) | RMSEpred (5) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| paper (CoMFA) | AMPA | | 4 components | 0.877 | +0.520 | 0.258 | | |
| paper (CoMFA) | glycine | | 5 components | 0.890 | +0.528 | 0.276 | | |
| paper (CoMFA) | Delta | | 5 components | 0.894 | +0.608 | 0.308 | | |
| comfa | AMPA | 1 | 0.1 | 0.807 | +0.480 | 0.506 | -0.039 | 0.318 |
| comfa | glycine | 3 | 0.1 | 0.804 | +0.382 | 0.641 | -0.390 | 0.337 |
| comfa | Delta | 3 | 0.1 | 0.795 | +0.324 | 0.719 | -1.450 | 0.417 |
| comfa-dedup | AMPA | 1 | 0.1 | 0.806 | +0.467 | 0.517 | -0.035 | 0.317 |
| comfa-dedup | glycine | 3 | 0.1 | 0.801 | +0.337 | 0.659 | -0.319 | 0.328 |
| comfa-dedup | Delta | 3 | 0.1 | 0.792 | +0.290 | 0.743 | -1.356 | 0.409 |
| default | AMPA | 1 | 0.1 | 0.915 | +0.377 | 0.553 | +0.608 | 0.195 |
| default | glycine | 0.5 | 0.1 | 0.909 | +0.477 | 0.590 | -3.074 | 0.577 |
| default | Delta | 0.2 | 0.1 | 0.823 | +0.533 | 0.598 | -3.323 | 0.554 |
| default-dedup | AMPA | 0.7 | 0.1 | 0.907 | +0.453 | 0.524 | +0.644 | 0.186 |
| default-dedup | glycine | 0.3 | 0.1 | 0.883 | +0.419 | 0.617 | -3.000 | 0.571 |
| default-dedup | Delta | 0.2 | 0.1 | 0.826 | +0.528 | 0.605 | -3.193 | 0.546 |
| file | AMPA | 1 | 0.1 | 0.904 | +0.418 | 0.535 | +0.779 | 0.147 |
| file | glycine | 0.7 | 0.1 | 0.906 | +0.492 | 0.581 | -2.860 | 0.561 |
| file | Delta | 0.3 | 0.1 | 0.846 | +0.583 | 0.565 | -3.289 | 0.552 |
| file-dedup | AMPA | 1 | 0.1 | 0.902 | +0.452 | 0.524 | +0.775 | 0.148 |
| file-dedup | glycine | 0.5 | 0.1 | 0.891 | +0.429 | 0.612 | -2.529 | 0.537 |
| file-dedup | Delta | 0.2 | 0.1 | 0.830 | +0.542 | 0.597 | -3.316 | 0.554 |

The selectivity three ways: the direct model of Delta, the difference of the AMPA and glycine models, and the two-task fit (both affinities on one kernel, correlation rho by the marginal likelihood, the alpha and lambda of the AMPA model). q2 and RMSEcv on the training set with whole molecules held out; RMSE on the five test compounds.

| setting | Delta model q2 / RMSEcv / RMSEtest | AMPA - glycine q2 / RMSEcv / RMSEtest | two-task rho | two-task AMPA q2 (alone) | glycine q2 (alone) | two-task Delta q2 / RMSEcv / RMSEtest |
|---|---|---|---:|---|---|---|
| comfa | +0.324 / 0.719 / 0.417 | +0.331 / 0.716 / 0.447 | +0.87 | +0.452 (+0.493) | +0.322 (+0.312) | +0.191 / 0.787 / 0.381 |
| comfa-dedup | +0.290 / 0.743 / 0.409 | +0.298 / 0.738 / 0.429 | +0.87 | +0.442 (+0.482) | +0.302 (+0.286) | +0.201 / 0.788 / 0.382 |
| default | +0.533 / 0.598 / 0.554 | +0.461 / 0.642 / 0.526 | +0.46 | +0.458 (+0.487) | +0.522 (+0.528) | +0.532 / 0.598 / 0.636 |
| default-dedup | +0.528 / 0.605 / 0.546 | +0.518 / 0.612 / 0.520 | +0.54 | +0.391 (+0.424) | +0.509 (+0.505) | +0.473 / 0.640 / 0.608 |
| file | +0.583 / 0.565 / 0.552 | +0.547 / 0.589 / 0.574 | +0.63 | +0.467 (+0.479) | +0.546 (+0.540) | +0.518 / 0.607 / 0.548 |
| file-dedup | +0.542 / 0.597 / 0.554 | +0.501 / 0.622 / 0.540 | +0.63 | +0.398 (+0.412) | +0.530 (+0.516) | +0.474 / 0.639 / 0.539 |

The five test compounds (Table 2 of the paper): observed Delta, the paper's selectivity model and difference of models, and the same from the `comfa` and `default` settings here.

| compound | observed | paper: Delta model | paper: difference | comfa: Delta model | comfa: difference | comfa: two-task | default: Delta model | default: difference | default: two-task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cpd_53 | -0.63 | -0.22 | -0.19 | +0.00 | +0.02 | +0.10 | +0.26 | +0.19 | +0.48 |
| cpd_54 | -0.30 | -0.10 | +0.07 | -0.36 | -0.26 | -0.20 | -0.36 | -0.34 | -0.16 |
| cpd_55 | +0.05 | +0.16 | +0.39 | -0.15 | +0.43 | +0.08 | -0.50 | -0.39 | -0.45 |
| cpd_56 | +0.12 | +0.30 | +0.50 | +0.70 | +0.65 | +0.15 | +0.66 | +0.76 | +0.69 |
| cpd_57 | -0.21 | -0.23 | +0.73 | -0.50 | -0.60 | -0.64 | -0.59 | -0.54 | -0.66 |
| RMSE | | 0.22 | 0.54 | 0.42 | 0.45 | 0.38 | 0.55 | 0.53 | 0.64 |

