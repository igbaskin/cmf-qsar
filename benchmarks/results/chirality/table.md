# Do the signed three-point channels buy anything?

Leave-one-chemotype-out on the five chemotype-split sets, the regime the alignment-free kernel exists for. Three regularisations by three distance grids for every variant, chosen inside the training part of each fold; the number of radial shells is a column rather than a search, so the signed channels never get more freedom than the two-point kernel they extend.

| set | n | chemotypes | two-point | signed, 1 shell | signed, 2 shells | signed, 4 shells |
|---|---:|---:|---:|---:|---:|---:|
| DHFR | 361 | 31 | +0.381 | +0.413 | +0.402 | **+0.429** |
| COX2 | 282 | 14 | +0.199 | **+0.397** | +0.394 | +0.386 |
| BZR | 147 | 12 | **-0.119** | -0.140 | -0.122 | -0.159 |
| AChE | 111 | 7 | **+0.154** | +0.103 | +0.092 | +0.100 |
| mittal/ACHE | 111 | 7 | -0.299 | -0.364 | -0.288 | **-0.217** |
| **mean** |  |  | +0.063 | +0.082 | +0.095 | **+0.108** |

Channels per molecule, averaged over the sets: two-point 255, signed, 1 shell 265, signed, 2 shells 275, signed, 4 shells 295.

## Does the difference survive the scatter?

Paired against the two-point kernel, set by set, with a bootstrap interval over the five sets. Five is a small number of sets and the interval says so.

| variant | mean difference | 95 % interval | better on |
|---|---:|---|---:|
| `signed, 1 shell` | +0.019 | [-0.051, +0.113] | 2/5 |
| `signed, 2 shells` | +0.033 | [-0.031, +0.121] | 3/5 |
| `signed, 4 shells` | +0.045 | [-0.028, +0.121] | 3/5 |

At signed, 4 shells the mean difference is +0.045, of which **COX2 supplies +0.187**; over the other four sets it is +0.009.
