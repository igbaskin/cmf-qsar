# Seven endpoints on one kernel

`examples/cyclodextrin/`: 1669 distinct guests measured against the 7 cyclodextrin hosts, 2671 measurements in all (beta 1162, alpha 827, gamma 292, hp_beta 135, sbe_beta 135, methyl_beta 77, dimethyl_beta 43). One kernel for every row -- the rotation-invariant field autocorrelation at alpha 0.1, chosen by the independent model, which has no covariance to favour -- and 10-fold cross-validation over **guests**, so that no guest has one host in the training set and another in the test set.

| task covariance | mean q² | beta | alpha | gamma | hp_beta | sbe_beta | methyl_beta | dimethyl_beta | what it is |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| independent | **+0.412** | +0.448 | +0.338 | +0.473 | +0.545 | +0.331 | +0.377 | +0.374 | rho = 0 |
| one rho, CV | **+0.454** | +0.448 | +0.338 | +0.524 | +0.586 | +0.389 | +0.441 | +0.454 | rho = +0.90 |
| one rho, evidence | **+0.454** | +0.449 | +0.341 | +0.523 | +0.592 | +0.391 | +0.435 | +0.451 | rho = +0.857 over 25 values |
| free rank 1 | **+0.475** | +0.447 | +0.339 | +0.525 | +0.604 | +0.444 | +0.504 | +0.465 | 2 x 1 starts, 1603 evaluations of the evidence; cross-validation would have cost ten fits each |
| free rank 2 | **+0.488** | +0.447 | +0.340 | +0.527 | +0.624 | +0.452 | +0.536 | +0.490 | 2 x 1 starts, 2274 evaluations of the evidence; cross-validation would have cost ten fits each |
| descriptors | **+0.448** | +0.446 | +0.342 | +0.505 | +0.608 | +0.390 | +0.430 | +0.413 | rbf, length scale 4, nugget 0.1 |

**Sharing is worth +0.076 mean q²** (+0.412 independent against +0.488 for free rank 2).

The evidence ranks them in this order, which is the ranking it was asked for and not the one that matters:

| task covariance | log evidence | mean q² |
|---|---:|---:|
| free rank 2 | -2605.3 | +0.488 |
| free rank 1 | -2608.8 | +0.475 |
| one rho, evidence | -2635.3 | +0.454 |
| one rho, CV | -2635.8 | +0.454 |
| descriptors | -2647.6 | +0.448 |
| independent | -2713.1 | +0.412 |

## An endpoint with no measurements

Each host removed entirely -- the whole endpoint, not a fold of its guests -- and predicted from the other six through the descriptor covariance, with a mean linear in the descriptors so that there is something to extrapolate. Spearman ρ is the column to read: the absolute scale of an endpoint nobody measured is not recoverable, and a screen ranks.

| host removed | n | nearest host by descriptors | ρ, descriptor covariance | ρ, nearest host alone | ρ, mean of the measured hosts | q², descriptor |
|---|---:|---|---:|---:|---:|---:|
| beta | 1162 | sbe_beta | +0.577 | +0.320 | +0.623 (615 guests) | +0.231 |
| alpha | 827 | beta | +0.431 | +0.433 | +0.470 (522 guests) | +0.066 |
| gamma | 292 | sbe_beta | -0.284 | +0.145 | +0.533 (225 guests) | -5.499 |
| hp_beta | 135 | sbe_beta | +0.664 | +0.294 | +0.806 (105 guests) | +0.483 |
| sbe_beta | 135 | beta | +0.516 | +0.515 | +0.498 (74 guests) | +0.200 |
| methyl_beta | 77 | dimethyl_beta | +0.595 | +0.529 | +0.894 (73 guests) | +0.416 |
| dimethyl_beta | 43 | methyl_beta | +0.703 | +0.708 | +0.863 (38 guests) | +0.044 |

Mean ρ over the 7 hosts: **+0.457** through the descriptor covariance against **+0.421** for the nearest host's own model.

