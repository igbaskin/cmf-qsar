# What the field sees that the fingerprint does not

Each Sutherland set, the five-field kernel at the attenuation the evidence picks for the set against the Tanimoto kernel of 2048-bit ECFP4. *Held-out ρ* is the mean canonical correlation over all pairs on molecules the directions were not fitted on; *real* counts the pairs above the 95th percentile of a shuffled pairing; *contained* counts the directions of the field of which the fingerprint explains more than half the variance on held-out molecules. The three q² values are kernel ridge regression on the whole field kernel, on its part inside the contained directions, and on the rest, the regularisation chosen inside the folds; *random* is a subspace of the same dimension as the contained part, drawn at random 5 times.

| set | n | α | ridge | held-out ρ (first three) | mean ρ | real | contained | q² whole | q² contained | q² own | q² random, same size | own − contained [95 %] |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| DHFR | 361 | 2 | 0.001 | +0.99 +0.99 +0.99 | +0.364 | 251/288 | 45/360 | +0.611 | +0.641 | +0.239 | +0.055 | -0.402 [-0.476, -0.329] |
| COX2 | 282 | 2 | 0.01 | +0.99 +0.99 +0.95 | +0.384 | 197/224 | 29/277 | +0.491 | +0.494 | +0.337 | +0.022 | -0.157 [-0.231, -0.085] |
| BZR | 147 | 1 | 0.001 | +0.99 +0.99 +0.97 | +0.500 | 109/117 | 28/146 | +0.396 | +0.259 | +0.107 | +0.039 | -0.151 [-0.313, +0.039] |
| ACE | 114 | 1 | 0.001 | +0.98 +0.95 +0.94 | +0.391 | 70/83 | 10/101 | +0.603 | +0.563 | +0.340 | -0.017 | -0.223 [-0.389, -0.068] |
| AChE | 111 | 1 | 0.001 | +0.98 +0.97 +0.95 | +0.450 | 75/88 | 14/110 | +0.549 | +0.486 | -0.055 | -0.028 | -0.541 [-0.702, -0.381] |
| THR | 88 | 1 | 0.001 | +0.95 +0.97 +0.94 | +0.508 | 63/69 | 16/85 | +0.622 | +0.427 | +0.106 | +0.083 | -0.320 [-0.589, -0.039] |
| THER | 76 | 1 | 0.1 | +0.98 +0.94 +0.95 | +0.506 | 48/57 | 15/69 | +0.438 | +0.421 | +0.011 | -0.006 | -0.410 [-0.720, -0.207] |
| GPB | 66 | 2 | 0.001 | +0.93 +0.91 +0.85 | +0.268 | 23/45 | 4/52 | +0.511 | +0.534 | +0.208 | +0.040 | -0.326 [-0.540, -0.133] |
| **mean** |  |  |  |  |  |  |  | **+0.528** | **+0.478** | **+0.162** | **+0.024** | **-0.316** |

**(i) The correlations are real.** On average 83 % of the canonical pairs hold out above the shuffled pairing, and the leading pairs hold out at correlations near one on every set: the field and the fingerprint describe the same molecules and agree about most of what varies between them. The fingerprint *contains* -- explains more than half the variance of -- 14 % of the field's directions on average; the rest of the field is its own.

**(ii) Where the activity is.** The part of the field the fingerprint does not contain predicts the activity at q² **+0.162** on average, against **+0.478** for the part it contains and **+0.528** for the whole field; a random subspace of the contained part's size reaches +0.024. The difference (own minus contained) is **-0.316** on average, with its interval above zero on 0 of 8 sets and below zero on 7 of 8.
So on these sets the activity sits mostly in the field directions the fingerprint already contains: the three-dimensional part of the description adds less than the shared part carries. That is a measured answer of the same kind as D5, and it says what the item asked to be told either way.

## The components are readable: the same field in moved poses

A fingerprint has no co-field, so the third gate is measured on two field views of the same molecules on the same fields: the poses of the file against the same poses moved by a random rigid motion of up to 10 degrees and 0.5 Å. *Same* is the mean F3 cosine between the co-fields of the two sides of one component over the first 3; *different* is the mean absolute cosine between the co-fields of two different components.

| set | ridge | held-out ρ (first three) | real | co-field cosine, same component | different components |
|---|---:|---|---:|---:|---:|
| DHFR | 0.003 | +0.99 +0.99 +0.99 | 259/288 | +0.666 | 0.072 |
| COX2 | 0.003 | +1.00 +1.00 +0.99 | 170/225 | +0.693 | 0.060 |
| BZR | 0.001 | +1.00 +1.00 +0.99 | 98/117 | +0.824 | 0.165 |
| ACE | 0.01 | +1.00 +0.99 +1.00 | 88/90 | +0.903 | 0.009 |
| AChE | 0.001 | +0.99 +1.00 +0.99 | 79/88 | +0.857 | 0.016 |
| THR | 0.001 | +0.99 +0.99 +0.98 | 64/70 | +0.892 | 0.016 |
| THER | 0.03 | +1.00 +1.00 +0.99 | 55/60 | +0.950 | 0.006 |
| GPB | 0.001 | +0.99 +0.97 +0.98 | 49/52 | +0.869 | 0.018 |
| **mean** |  |  |  | **+0.832** | **0.045** |

**(iii)** The two co-fields of one component agree at a cosine of **+0.832** on average, against 0.045 between different components, on 8 of 8 sets. A canonical component is one picture, drawn from either side.
