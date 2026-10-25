# Do the signed three-point channels buy anything?

Leave-one-chemotype-out on the five chemotype-split sets, the regime the alignment-free kernel exists for. Three regularisations by three distance grids for every variant, chosen inside the training part of each fold; the number of radial shells is a column rather than a search, so the signed channels never get more freedom than the two-point kernel they extend.

| set | n | chemotypes | two-point | signed, 1 shell | signed, 2 shells | signed, 4 shells | signed, 4 shells, balanced |
|---|---:|---:|---:|---:|---:|---:|---:|
| DHFR | 361 | 31 | +0.381 | +0.413 | +0.402 | **+0.429** | +nan |
| COX2 | 282 | 14 | +0.199 | **+0.397** | +0.394 | +0.386 | +nan |
| BZR | 147 | 12 | **-0.119** | -0.140 | -0.122 | -0.159 | +nan |
| AChE | 111 | 7 | **+0.154** | +0.103 | +0.092 | +0.100 | +nan |
| mittal/ACHE | 111 | 7 | -0.299 | -0.364 | -0.288 | **-0.217** | +nan |
| **mean** |  |  | +0.063 | +0.082 | +0.095 | **+0.108** | +nan |

Channels per molecule, averaged over the sets: two-point 255, signed, 1 shell 265, signed, 2 shells 275, signed, 4 shells 295, signed, 4 shells, balanced 0.

## Does the difference survive the scatter?

Paired against the two-point kernel, set by set, with a bootstrap interval over the five sets. Five is a small number of sets and the interval says so.

| variant | mean difference | 95 % interval | better on |
|---|---:|---|---:|
| `signed, 1 shell` | +0.019 | [-0.051, +0.113] | 2/5 |
| `signed, 2 shells` | +0.033 | [-0.031, +0.121] | 3/5 |
| `signed, 4 shells` | +0.045 | [-0.028, +0.121] | 3/5 |

At signed, 4 shells, balanced the mean difference is +nan, of which **AChE supplies +nan**; over the other four sets it is +nan.

## The experiment the five sets cannot do

20 pairs of molecules that differ in handedness and in nothing else -- one conformation and its exact mirror image, so the two-point channels of a pair are identical (their kernel entry is 1 - 1e-15). The activity is `(1 - w) * composition + w * handedness`, so `w` is the share of it that no two-point descriptor can reach. Random 10-fold folds, the grid and lambda chosen inside each training part.

| w | two-point | signed, 1 shell | signed, 2 shells | signed, 4 shells | signed, 4 shells, balanced |
|---:|---:|---:|---:|---:|---:|
| 0.00 | **+0.970** | +0.926 | +0.905 | +0.818 | +0.566 |
| 0.25 | **+0.648** | +0.602 | +0.590 | +0.381 | +0.105 |
| 0.50 | **+0.019** | -0.092 | -0.040 | -0.059 | -0.184 |
| 0.75 | -0.312 | -0.292 | **-0.281** | -0.309 | -0.346 |
| 1.00 | -0.391 | -0.363 | **-0.339** | -0.366 | -0.353 |

**Why.** A kernel method carries what it learnt from one molecule to another along a direction in feature space. The signed channels of a molecule and its own mirror image are exact negatives -- that is what the block is for -- but between *different* molecules they share nothing: the cosine of the signed block is -0.016 for two molecules of the same handedness and +0.016 for two of opposite handedness, a difference of -0.033 against a spread of 0.552 over 760 pairs. Handedness is not a direction in this space, so there is nothing for a linear model to transfer.
