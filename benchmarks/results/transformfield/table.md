# Does a transformation carry its effect from one series to another? (plan item MC6)

A matched molecular pair in a fixed pose is a difference of two fields: a local object with its geometry kept, rather than a number attached to an atom type. This asks whether that object transfers to a series the model has never seen. Plan item D5 is the reference and the warning: increments fitted on one series did not transfer.

3814 matched pairs from 17 congeneric series with a crystal pose for every ligand (the Merck KGaA FEP sets and the Schrodinger JACS sets). 2782 distinct transformations, 98 of them in more than one series. The fields are electrostatic, steric, hydrophobic at an attenuation of 1; the folds hold out a whole series and the ridge is chosen inside the training series the same way; the ridge has no intercept, so the model predicts exactly the opposite change for the reverse of a pair.

| what predicts the change | q2 against each series' own mean |
| --- | ---: |
| no change at all | -0.025 |
| the transformation as a field, inside one series (no transfer) | +0.465 |
| the transformation as a number (plan item D5) | -0.073 |
| the transformation as a field, without a frame | -0.025 |
| the transformation as a field | -0.025 |

The field against the number: +0.048 [-0.017, +0.108], resampling whole series. The field against predicting no change: -0.025 [-0.118, +0.035].

The transformation of a held-out pair had been seen in another series for 11.4% of the pairs, so the reference of D5 can say nothing at all about the rest and predicts no change for them.

| series | pairs | spread of the change | q2, field | q2, no frame | q2, number | q2 inside the series | transformation seen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bace | 218 | 0.65 | -0.015 | -0.004 | -0.370 | +0.321 | 25% |
| cdk2 | 120 | 1.26 | +0.051 | -0.003 | -0.044 | +0.761 | 18% |
| cdk8 | 261 | 1.35 | -0.032 | -0.010 | -0.013 | +0.575 | 2% |
| cmet | 85 | 1.15 | +0.034 | -0.003 | -0.014 | +0.078 | 8% |
| eg5 | 49 | 0.45 | -0.019 | -0.009 | -1.496 | +0.044 | 8% |
| hif2a | 39 | 0.71 | -0.112 | -0.010 | -0.391 | +0.382 | 56% |
| hif2a-jmc2023 | 695 | 0.70 | -0.319 | -0.057 | -0.134 | +0.492 | 7% |
| jnk1_manual_flips | 318 | 0.79 | -0.009 | -0.001 | -0.097 | +0.705 | 28% |
| mcl1_extra_flips | 610 | 0.99 | +0.036 | -0.005 | -0.005 | +0.426 | 6% |
| p38 | 196 | 0.87 | -0.127 | -0.081 | -0.103 | +0.148 | 6% |
| pfkfb3 | 465 | 1.18 | +0.080 | -0.001 | -0.021 | +0.385 | 6% |
| ptp1b | 159 | 0.81 | +0.019 | -0.078 | -0.143 | +0.640 | 6% |
| shp2 | 48 | 0.71 | -0.082 | -0.120 | -0.175 | +0.066 | 19% |
| syk | 334 | 0.74 | +0.028 | -0.005 | -0.031 | +0.519 | 3% |
| thrombin_core | 55 | 0.51 | -0.094 | -0.208 | -0.620 | +0.454 | 33% |
| tnks2 | 42 | 0.87 | -0.001 | -0.007 | -0.081 | +0.087 | 36% |
| tyk2 | 120 | 1.25 | -0.146 | -0.154 | -0.220 | +0.524 | 32% |

## The same transformation in several series

Before any model: what did one transformation actually do, series by series? The mean change is taken inside each series first, so a series with many pairs does not outweigh the rest.

| transformation | series | pairs | mean change | lowest | highest |
| --- | ---: | ---: | ---: | ---: | ---: |
| C[*:1] -> Cl[*:1] | 10 | 23 | +0.15 | -0.93 | +1.07 |
| C[*:1] -> CC[*:1] | 8 | 13 | -0.17 | -1.15 | +1.01 |
| F[*:1] -> C[*:1] | 7 | 14 | +0.12 | -0.53 | +1.37 |
| Cl[*:1] -> C[*:1] | 10 | 23 | -0.15 | -1.07 | +0.93 |
| C[*:1] -> F[*:1] | 7 | 14 | -0.12 | -1.37 | +0.53 |
| C[*:1] -> C1CC1[*:1] | 4 | 6 | +0.39 | +0.06 | +0.65 |
| F[*:1] -> N#C[*:1] | 5 | 12 | -0.16 | -0.61 | +0.81 |
| Cl[*:1] -> N#C[*:1] | 5 | 12 | -0.48 | -1.05 | +0.09 |
| F[*:1] -> Cl[*:1] | 6 | 17 | -0.01 | -0.49 | +0.47 |
| CC(C)[*:1] -> CC[*:1] | 5 | 8 | +0.30 | -0.64 | +1.62 |
| Cl[*:1] -> F[*:1] | 6 | 17 | +0.01 | -0.47 | +0.49 |
| C[*:1] -> CO[*:1] | 4 | 13 | -0.14 | -1.49 | +0.40 |

11 of these 12 transformations raise the activity in one series and lower it in another. That is the item's question answered by the data before any model is fitted: a transformation of this kind does not have one effect to carry.

## The verdict

**A transformation does not carry its effect across series.** The field model reaches q2 -0.025 against each series' own mean change, with an interval of -0.118 to +0.035 over the series, which contains zero. The gate of the item is not passed, and the result stands beside plan item D5 rather than replacing it: what a group does depends on the pocket it does it in, and neither a number nor a field carries that from one protein to another.

**It is not the frame.** The same pairs described without one, as the rotation-invariant autocorrelation of the difference field, reach q2 -0.025. A frame convention that put the groups in the wrong place would show up as a difference between those two rows.

Against the reference of D5 -- the same transformation as a number, averaged over the series that have it -- the field is not ahead: q2 -0.025 against -0.073, a difference of +0.048 [-0.017, +0.108]. The number has a second limit that the field does not: it can say nothing about a transformation it has never seen, which is 89 per cent of the pairs here, while a field is read for what it looks like and not for its name.

**Is the descriptor itself the problem?** Trained inside each series in five folds, where nothing has to transfer, the same difference fields reach q2 +0.465. Those folds hold out whole molecules, not pairs, since two pairs that share a molecule share half of their field and part of their measurement; the row covers the 100 per cent of the pairs that such a split can test. Its ridge is chosen on the same folds it is scored on, so it is a best case. That is a working model of a change, and it stops working as soon as it is asked to cross from one protein to another.

**The narrow activity ranges are the known risk of these sets**, and the table above shows it: the spread of the change inside a series is often well under one log unit, so there is little for any model to explain. A series whose changes are all the same size cannot separate a good prediction from a mediocre one.

