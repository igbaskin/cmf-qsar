# Two questions about a set, asked with its kernel

## Is this the same population as that? (MMD)

20 random half-splits of each set, 499 permutations each. There is nothing to find in a half-split, so the share called significant at 5 % is the test's false-positive rate and should be near 5 %.

| set | n | half-splits called significant |
|---|---:|---:|
| ACE | 114 | 5% |
| AChE | 111 | 5% |
| BZR | 147 | 10% |
| COX2 | 282 | 0% |
| DHFR | 361 | 5% |
| GPB | 66 | 15% |
| THER | 76 | 5% |
| THR | 88 | 0% |
| **mean** |  | **5.6%** |

Against that, all 28 pairs of *different* targets: **100%** are called significant, median z = +171. The test is calibrated where there is nothing and saturated where there is.

## What are this set's real groups? (spectral clustering)

The adjusted Rand index between the scaffold groups and the clusters of the kernel, both cut into the same number of groups, and what holding out a whole group of each kind costs a model. The eigengap column is how many clusters the kernel would have chosen for itself.

| set | groups | k by eigengap | ARI | q² random | q² by scaffold | q² by field cluster |
|---|---:|---:|---:|---:|---:|---:|
| ACE | 7 | 2 | +0.136 | +0.585 | +0.349 | +0.164 |
| AChE | 7 | 2 | +0.156 | +0.529 | +0.221 | +0.128 |
| BZR | 12 | 2 | +0.204 | +0.364 | +0.010 | -0.740 |
| COX2 | 14 | 2 | +0.229 | +0.507 | +0.207 | +0.448 |
| DHFR | 31 | 2 | +0.257 | +0.613 | +0.423 | +0.374 |
| GPB | 4 | 2 | +0.022 | +0.368 | +0.150 | +0.037 |
| THER | 5 | 2 | +0.070 | +0.449 | +0.302 | -0.293 |
| THR | 9 | 2 | +0.046 | +0.645 | +0.507 | +0.342 |
| **mean** |  |  | **+0.140** | **+0.508** | **+0.271** | **+0.057** |
