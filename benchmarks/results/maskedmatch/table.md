# Does a masked co-field recognise a scaffold the model has never seen?

A model is fitted on one chemotype of a series. The molecules of every other chemotype that are at least as active have to be ranked above a background. Nothing of those scaffolds is in the training set. AUC of the ranking; q2 is the model's own cross-validation on the one chemotype it was fitted on.

Mask of 10 atoms; every size is in the last table.

## Hops against the weak

The background is the weakly active molecules of the same other chemotypes -- real molecules of the same target in the same frame, so nothing separates them from the hops but the activity.

| set | training chemotype | n train | q2 | positives | negatives | masked cosine | masked overlap | field similarity | prediction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AChE | `O=C(NCCC1CC[NH+](Cc2ccccc2)CC1)c` | 20 | +0.096 | 53 | 30 | 0.643 | 0.650 | 0.424 | 0.672 |
| THR | `O=C(C(Cc1ccccc1)NS(=O)(=O)c1ccc2` | 16 | -0.138 | 40 | 24 | 0.623 | 0.630 | 0.789 | 0.665 |
| ACE | `C1CCNC1` | 19 | -0.420 | 74 | 25 | 0.788 | 0.832 | 0.482 | 0.658 |
| GPB | `C1CCOCC1` | 46 | +0.587 | 13 | 5 | 0.862 | 0.862 | 0.569 | 0.877 |
| **mean** | | | | | | 0.729 | 0.743 | 0.566 | 0.718 |

The masked cosine is +0.163 AUC against the unmasked field similarity, ahead on 3 of 4 sets.

## Hops against the decoys

The background is the 100 property-matched presumed inactives. This is the gate item MC5 states, and it decides nothing: the unmasked field similarity is already at its ceiling, because molecules from other targets differ from the actives in size and shape and not only in activity.

| set | training chemotype | n train | q2 | positives | negatives | masked cosine | masked overlap | field similarity | prediction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AChE | `O=C(NCCC1CC[NH+](Cc2ccccc2)CC1)c` | 20 | +0.096 | 53 | 100 | 0.469 | 0.544 | 0.985 | 0.605 |
| THR | `O=C(C(Cc1ccccc1)NS(=O)(=O)c1ccc2` | 16 | -0.138 | 40 | 100 | 0.483 | 0.478 | 1.000 | 0.595 |
| ACE | `C1CCNC1` | 19 | -0.420 | 74 | 100 | 0.758 | 0.842 | 0.688 | 0.733 |
| GPB | `C1CCOCC1` | 46 | +0.587 | 13 | 100 | 0.625 | 0.685 | 1.000 | 0.694 |
| **mean** | | | | | | 0.584 | 0.637 | 0.918 | 0.657 |

The masked cosine is -0.334 AUC against the unmasked field similarity, ahead on 1 of 4 sets.

## How much of the prediction comes from inside the mask

The second score the item asks for: a candidate whose promise is all outside the region the model rewards is one the model is extrapolating about. Mean over the molecules of each kind, in the units of the activity, on the harder task.

| set | mask atoms | inside, hops | inside, weak | outside, hops | outside, weak |
|---|---:|---:|---:|---:|---:|
| AChE | 5 | +0.076 | -0.083 | +0.974 | +0.300 |
| AChE | 10 | +0.309 | +0.078 | +0.740 | +0.139 |
| AChE | 20 | +0.416 | +0.185 | +0.633 | +0.032 |
| THR | 5 | -0.562 | -1.453 | +5.093 | +3.947 |
| THR | 10 | -0.381 | -1.223 | +4.912 | +3.716 |
| THR | 20 | -0.582 | -1.544 | +5.113 | +4.038 |
| ACE | 5 | +1.980 | +1.237 | +3.514 | +1.354 |
| ACE | 10 | +1.069 | +0.621 | +4.426 | +1.970 |
| ACE | 20 | +1.979 | +1.236 | +3.516 | +1.355 |
| GPB | 5 | +0.471 | +0.111 | +0.993 | -1.473 |
| GPB | 10 | +0.477 | +0.121 | +0.986 | -1.484 |
| GPB | 20 | +0.869 | +0.190 | +0.595 | -1.552 |

## Every mask size, on the harder task

| set | mask atoms | masked cosine | masked overlap | field similarity | prediction |
|---|---:|---:|---:|---:|---:|
| AChE | 5 | 0.640 | 0.642 | 0.424 | 0.672 |
| AChE | 10 | 0.643 | 0.650 | 0.424 | 0.672 |
| AChE | 20 | 0.659 | 0.677 | 0.424 | 0.672 |
| THR | 5 | 0.621 | 0.631 | 0.789 | 0.665 |
| THR | 10 | 0.623 | 0.630 | 0.789 | 0.665 |
| THR | 20 | 0.624 | 0.635 | 0.789 | 0.665 |
| ACE | 5 | 0.752 | 0.784 | 0.482 | 0.658 |
| ACE | 10 | 0.788 | 0.832 | 0.482 | 0.658 |
| ACE | 20 | 0.750 | 0.782 | 0.482 | 0.658 |
| GPB | 5 | 0.862 | 0.862 | 0.569 | 0.877 |
| GPB | 10 | 0.862 | 0.862 | 0.569 | 0.877 |
| GPB | 20 | 0.877 | 0.846 | 0.569 | 0.877 |

