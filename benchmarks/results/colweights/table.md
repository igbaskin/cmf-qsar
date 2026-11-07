# Which columns count: a weighted distance over the channels (plan item M15)

Every arm is the same cosine-normalised linear kernel on the same scaled rows and differs only in the weight each column is given. Every weight is computed inside the training part of each outer fold. The difference is in q2 against equal weights, which is what the program does today, with the bootstrap interval of rule 4. The last two rows of each table are forests: on the rows the kernel really uses, which is the gap the arms have to close, and on the rows as they are, which also carry the row length that the first half of item M15 measured.

## hops

the hop environment (BT7), target barrier, 1801 samples, 2601 columns in 153 blocks, 10 folds.

| arm | q2 | error | mean absolute error | columns used | difference against equal weights | share of the forest's lead |
|---|---|---|---|---|---|---|
| equal weights | +0.572 | 1.246 | 0.736 | 2601 | the reference | -- |
| block align | +0.658 | 1.114 | 0.663 | 2414 to 2465 | +0.086 [+0.042, +0.133] | 51 per cent |
| block alignf | +0.605 | 1.197 | 0.708 | 1904 to 2006 | +0.033 [-0.004, +0.072] | 19 per cent |
| column align | +0.612 | 1.186 | 0.706 | 2347 to 2392 | +0.040 [-0.007, +0.089] | 24 per cent |
| best columns | +0.601 | 1.203 | 0.729 | 52 to 260 | +0.029 [-0.031, +0.089] | 17 per cent |
| forest importances | +0.628 | 1.162 | 0.664 | 2348 to 2378 | +0.056 [-0.001, +0.115] | 33 per cent |
| a forest on the scaled rows (the length kept) | +0.845 | 0.751 | 0.375 | -- | +0.273 [+0.217, +0.333] | -- |
| a forest on the kernel's own rows | +0.742 | 0.967 | 0.481 | -- | +0.171 [+0.116, +0.226] | -- |

The forest's lead over equal weights on this series, on the kernel's own rows: +0.171. On the rows as they are, which also carry the row length of the first half of item M15: +0.273.

## molecules

whole-molecule autocorrelation, target log K, 1162 samples, 255 columns in 15 blocks, 10 folds.

| arm | q2 | error | mean absolute error | columns used | difference against equal weights | share of the forest's lead |
|---|---|---|---|---|---|---|
| equal weights | +0.461 | 0.597 | 0.452 | 255 | the reference | -- |
| block align | +0.484 | 0.584 | 0.442 | 255 | +0.024 [+0.003, +0.043] | 25 per cent |
| block alignf | +0.468 | 0.593 | 0.450 | 170 to 187 | +0.008 [-0.015, +0.030] | 8 per cent |
| column align | +0.469 | 0.592 | 0.447 | 255 | +0.008 [-0.021, +0.034] | 8 per cent |
| best columns | +0.480 | 0.586 | 0.442 | 153 | +0.019 [-0.005, +0.042] | 20 per cent |
| forest importances | +0.444 | 0.606 | 0.460 | 255 | -0.016 [-0.039, +0.003] | -17 per cent |
| a forest on the scaled rows (the length kept) | +0.585 | 0.523 | 0.387 | -- | +0.124 [+0.095, +0.154] | -- |
| a forest on the kernel's own rows | +0.556 | 0.541 | 0.405 | -- | +0.096 [+0.066, +0.125] | -- |

The forest's lead over equal weights on this series, on the kernel's own rows: +0.096. On the rows as they are, which also carry the row length of the first half of item M15: +0.124.

## sac

site profile (M5), 6 A, target dE_OH, 139 samples, 78 columns in 6 blocks, 27 folds.

| arm | q2 | error | mean absolute error | columns used | difference against equal weights | share of the forest's lead |
|---|---|---|---|---|---|---|
| equal weights | +0.847 | 0.575 | 0.446 | 78 | the reference | -- |
| block align | +0.851 | 0.567 | 0.441 | 78 | +0.004 [+0.002, +0.007] | -- |
| block alignf | +0.855 | 0.559 | 0.435 | 78 | +0.008 [+0.002, +0.014] | -- |
| column align | +0.866 | 0.537 | 0.417 | 78 | +0.020 [+0.007, +0.031] | -- |
| best columns | +0.891 | 0.484 | 0.383 | 23 to 47 | +0.044 [+0.007, +0.083] | -- |
| forest importances | +0.875 | 0.518 | 0.397 | 78 | +0.029 [+0.010, +0.049] | -- |
| a forest on the scaled rows (the length kept) | +0.623 | 0.900 | 0.705 | -- | -0.223 [-0.324, -0.139] | -- |
| a forest on the kernel's own rows | +0.596 | 0.932 | 0.714 | -- | -0.250 [-0.412, -0.139] | -- |

On this series the forest is behind the kernel, so there is no lead to share and that column is left empty.

The forest's lead over equal weights on this series, on the kernel's own rows: -0.250. On the rows as they are, which also carry the row length of the first half of item M15: -0.223.

