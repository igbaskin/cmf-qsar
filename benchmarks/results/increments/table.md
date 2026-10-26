## Increments fitted to the data, and whether they transfer

The hydrophobic field and the two hydrogen-bond fields are built from one increment per (Sybyl atom type, hydrogen count). `published` takes them from the table, `learned` fits them on this set's training molecules, and `transferred` applies the increments fitted on each of the other seven sets. Eight Sutherland sets on their published training and test split, same five fields, same attenuation and ridge grids, same ten-fold cross-validation.

| set | n (train/test) | types | q2 published | q2 learned | R2pred published | R2pred learned | R2pred transferred | R2pred best other |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACE | 76/38 | 24 | +0.737 | +0.914 | +0.479 | +0.437 | +0.245 | +0.520 |
| AChE | 74/37 | 21 | +0.450 | +0.767 | +0.424 | +0.277 | +0.503 | +0.587 |
| BZR | 98/49 | 29 | +0.518 | +0.834 | -0.077 | -0.208 | +0.051 | +0.092 |
| COX2 | 188/94 | 25 | +0.628 | +0.713 | +0.196 | +0.211 | +0.128 | +0.184 |
| DHFR | 237/124 | 27 | +0.647 | +0.745 | +0.643 | +0.655 | +0.609 | +0.638 |
| GPB | 44/22 | 28 | +0.559 | +0.935 | +0.574 | +0.306 | +0.558 | +0.615 |
| THER | 51/25 | 22 | +0.542 | +0.725 | +0.238 | +0.235 | +0.245 | +0.333 |
| THR | 59/29 | 22 | +0.683 | +0.891 | +0.582 | +0.566 | +0.562 | +0.621 |
| **mean** | | | **+0.596** | +0.816 | +0.383 | +0.310 | +0.363 | +0.449 |

Fitting lowers the cross-validated error it is fitted on: q2 goes from +0.596 to +0.816 on the training set. On the test set, which the fitting never saw, R2pred goes from +0.383 to +0.310. The difference between those two movements is the selection bias of fitting the increments on the cross-validation that reports the q2.

Increments fitted on another set give +0.363 on average and +0.449 for the best of the seven, against +0.383 for the published table.

### How far the fitted increments move, and whether they agree with each other

Cosine between the fitted and the published increments of a field, per set: one means the fit only rescaled them, zero means it learned something unrelated.

| set | penalty | hydrophobic | hba | hbd | squared error, published -> fitted |
| --- | ---: | ---: | ---: | ---: | --- |
| ACE | 0.3 | +0.97 | +0.48 | +0.56 | 109.5 -> 39.2 |
| AChE | 300 | +1.00 | +1.00 | +0.99 | 59.3 -> 25.7 |
| BZR | 30 | +1.00 | +1.00 | +0.85 | 20.6 -> 7.1 |
| COX2 | 30 | +1.00 | +0.99 | +0.93 | 72.3 -> 55.8 |
| DHFR | 30 | +1.00 | +0.98 | +0.90 | 133.9 -> 96.9 |
| GPB | 0.3 | +0.98 | +0.81 | +0.21 | 22.1 -> 3.6 |
| THER | 300 | +1.00 | +1.00 | +0.98 | 82.2 -> 49.3 |
| THR | 0.3 | +0.99 | +0.66 | +0.21 | 16.9 -> 5.8 |

Agreement between the increments fitted on different sets, as the mean cosine over every pair of sets and the types they have in common: hydrophobic +0.89, hba +0.72, hbd +0.54. For comparison, every fit starts from the published table, whose cosine with itself is +1.00.

