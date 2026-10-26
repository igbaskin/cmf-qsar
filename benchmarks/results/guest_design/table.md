## What to make next, on a table of guests

`cmf design --kernel autocorrelation` ranks candidates built as whole molecules: the rotation-invariant kernel needs no frame, so the guests need no common scaffold and nothing is superimposed on anything. **Leave one group out**: for each row, every guest of the beta-cyclodextrin table (1162 of them) that carries that group is removed, a model is fitted on what is left, and the library is ranked by it, so a group's prediction comes from a model that never saw a molecule containing it. A group carried by more than 30% of the table cannot have all of its guests removed, and the row says so.

| group | guests dropped | what was dropped | predicted | rank | measured |
| --- | ---: | --- | ---: | ---: | ---: |
| methyl | 1 | the molecule only | +1.545 | 11/11 | +1.301 |
| propyl | 1 | the molecule only | +1.864 | 10/11 | +1.505 |
| butyl | 1 | the molecule only | +2.022 | 9/11 | +1.954 |
| pentyl | 1 | the molecule only | +2.397 | 6/11 | +2.467 |
| hexyl | 261 | every guest that carries it | +2.483 | 3/11 | +2.041 |
| phenyl | 1 | the molecule only | +2.081 | 8/11 | +2.127 |
| 4-methylphenyl | 1 | the molecule only | +2.435 | 5/11 | +3.225 |
| cyclopentyl | 83 | every guest that carries it | +2.134 | 7/11 | +3.204 |
| cyclohexyl | 122 | every guest that carries it | +2.243 | 6/11 | +3.620 |
| cyclohexylmethyl | 111 | every guest that carries it | +2.613 | 2/11 | +3.519 |
| adamantyl | 6 | every guest that carries it | +3.297 | 1/11 | - |

Over the 10 groups whose R-COOH is a measured guest the ranking follows the measurements: Spearman +0.709, R2 +0.236.

**The two kinds of row are not equally hard.** Where only the molecule was removed the model had seen hundreds of its relatives, and it is close: mean absolute error 0.26 over 6 groups. Where every guest carrying the group was removed (adamantyl, cyclohexylmethyl, hexyl, cyclohexyl, cyclopentyl) it is not: mean absolute error 0.95 over 4, and the bias is -0.73 -- a group the table no longer contains is pulled towards the mean.

**What the ranking still gets right is the order.** Adamantyl, whose six guests in the table measure 2.9 to 4.3 and whose own acid is not in it at all, comes first of the 11 with every one of those six removed; cyclohexylmethyl, the strongest of the measured groups that were fully held out, comes second. Taking every row together, the 4 groups measuring 3.0 or more have mean rank 5.0 of 11, so the ranking is useful at the top and unreliable in the middle.

