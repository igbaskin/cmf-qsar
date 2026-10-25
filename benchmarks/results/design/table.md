# Would the design loop have suggested the right substituent

Leave one substituent out: every molecule carrying a given R-group at the varied position is removed, the topomer model is refitted on what is left, and the whole library of groups is enumerated and ranked. The group's measured activity is the mean over the molecules that carry it.

| set | groups | position | q² of the models | ρ of ranking with measurement | top suggestion | library mean | library best |
|---|---:|---:|---:|---:|---:|---:|---:|
| THER | 46 of 46 | 0 | +0.251 | +0.279 | 6.53 | 5.24 | 10.17 |
| GPB | 41 of 47 | 2 | +0.566 | +0.516 | 4.40 | 2.94 | 6.80 |
| **mean** |  |  |  | **+0.397** |  |  |  |

Over 87 held-out groups on 2 sets, the ranking agrees with the measurements at a mean Spearman ρ of **+0.397**, and the group the loop puts first is worth **+1.37** log units more than the library average -- it lands in the best 20 % of the library **48 %** of the time.

Sets the loop cannot reach, and why:

| set | why |
|---|---|
| ACE | the core comes off cleanly in only 44% of the series |
| AChE | the core comes off cleanly in only 26% of the series |
| BZR | the core comes off cleanly in only 42% of the series |
| COX2 | the core comes off cleanly in only 22% of the series |
| DHFR | the core comes off cleanly in only 12% of the series |
| THR | the core comes off cleanly in only 25% of the series |

That is half the answer to how far this goes. A design loop over substituents needs a series with a scaffold that comes off at single acyclic bonds, and most public QSAR sets are not that: they are collections of several chemotypes with a common framework found by force. On the sets that are congeneric, the loop works; on the others there is nothing to vary and the honest answer is to say so rather than to rank whatever the core leaves behind.
