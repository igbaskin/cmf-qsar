# Can a field kernel see handedness?

Two constructions: C5's signed three-point channels, summed over every triple of atoms in the molecule, and C7's stereo centres, read at one atom at a time. The five real sets below cannot separate them, because none of them contains an enantiomeric pair; the synthetic experiment at the bottom is what does.

Leave-one-chemotype-out on the five chemotype-split sets, the regime the alignment-free kernel exists for. Three regularisations by three distance grids for every variant, chosen inside the training part of each fold; the number of radial shells is a column rather than a search, so the signed channels never get more freedom than the two-point kernel they extend.

| set | n | chemotypes | two-point | signed, 1 shell | signed, 2 shells | signed, 4 shells | signed, 4 shells, balanced | stereo centres (C7) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DHFR | 361 | 31 | +0.381 | +0.413 | +0.402 | **+0.429** | +0.331 | +0.362 |
| COX2 | 282 | 14 | +0.199 | **+0.397** | +0.394 | +0.386 | +0.328 | +0.202 |
| BZR | 147 | 12 | -0.119 | -0.140 | -0.122 | -0.159 | **-0.034** | -0.044 |
| AChE | 111 | 7 | **+0.154** | +0.103 | +0.092 | +0.100 | +0.032 | -0.086 |
| mittal/ACHE | 111 | 7 | -0.299 | -0.364 | -0.288 | **-0.217** | -0.279 | -0.238 |
| **mean** |  |  | +0.063 | +0.082 | +0.095 | **+0.108** | +0.076 | +0.039 |

Channels per molecule, averaged over the sets: two-point 255, signed, 1 shell 265, signed, 2 shells 275, signed, 4 shells 295, signed, 4 shells, balanced 295, stereo centres (C7) 280.

## Is the difference larger than the scatter?

Paired against the two-point kernel, set by set, with a bootstrap interval over the five sets. Five is a small number of sets and the interval says so.

| variant | mean difference | 95 % interval | better on |
|---|---:|---|---:|
| `signed, 1 shell` | +0.019 | [-0.051, +0.113] | 2/5 |
| `signed, 2 shells` | +0.033 | [-0.031, +0.121] | 3/5 |
| `signed, 4 shells` | +0.045 | [-0.028, +0.121] | 3/5 |
| `signed, 4 shells, balanced` | +0.013 | [-0.066, +0.090] | 3/5 |
| `stereo centres (C7)` | -0.024 | [-0.136, +0.055] | 3/5 |

At stereo centres (C7) the mean difference is -0.024, of which **AChE supplies -0.240**; over the other four sets it is +0.030.

## The experiment the five sets cannot do

20 pairs of molecules that differ in handedness and in nothing else -- one conformation and its exact mirror image, so the two-point channels of a pair are identical (their kernel entry is 1 - 1e-15). The activity is `(1 - w) * composition + w * handedness`, so `w` is the share of it that no two-point descriptor can reach. Random 10-fold folds, the grid and lambda chosen inside each training part.

| w | two-point | signed, 1 shell | signed, 2 shells | signed, 4 shells | signed, 4 shells, balanced | stereo centres (C7) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | **+0.970** | +0.926 | +0.905 | +0.818 | +0.566 | +0.924 |
| 0.25 | +0.648 | +0.602 | +0.590 | +0.381 | +0.105 | **+0.790** |
| 0.50 | +0.019 | -0.092 | -0.040 | -0.059 | -0.184 | **+0.504** |
| 0.75 | -0.312 | -0.292 | -0.281 | -0.309 | -0.346 | **+0.598** |
| 1.00 | -0.391 | -0.363 | -0.339 | -0.366 | -0.353 | **+0.721** |

**C5's signed three-point channels:** same handedness -0.016, opposite +0.016, a difference of -0.033 against a spread of 0.552 over 760 pairs.

**C7's stereo centres:** same handedness +0.468, opposite -0.468, a difference of +0.935 against a spread of 0.702 over 760 pairs.

**Why.** A kernel method carries what it learnt from one molecule to another along a direction in feature space. The signed channels of a molecule and its own mirror image are exact negatives -- that is what the block is for -- but between *different* molecules they share nothing: the cosine of the signed block is -0.016 for two molecules of the same handedness and +0.016 for two of opposite handedness, a difference of -0.033 against a spread of 0.552 over 760 pairs. Handedness is not a direction in this space, so there is nothing for a linear model to transfer -- which explains all of C5's failure, and what C7 was built to fix by reading the configuration at one centre instead of summing over the molecule.
