# Is the co-field pharmacophore a usable query

8 Sutherland sets, 5 random 70/30 splits each. The pharmacophore is read out of the training model's co-fields, masked at 2 standard deviations first, and its strongest 4 features are matched against the molecules held out. The actives are the most active 25 %.

| set | n | features | AUC query | AUC random centres | AUC the model |
|---|---:|---:|---:|---:|---:|
| DHFR | 361 | 7.4 | 0.573 | 0.498 | 0.901 |
| COX2 | 282 | 8.4 | 0.662 | 0.478 | 0.867 |
| BZR | 147 | 7.8 | 0.518 | 0.514 | 0.797 |
| ACE | 114 | 7.0 | 0.789 | 0.552 | 0.927 |
| AChE | 111 | 7.8 | 0.631 | 0.484 | 0.903 |
| THR | 88 | 8.4 | 0.595 | 0.468 | 0.780 |
| THER | 76 | 7.2 | 0.711 | 0.482 | 0.812 |
| GPB | 66 | 6.8 | 0.838 | 0.484 | 0.902 |
| **mean** |  |  | **0.665** | **0.495** | **0.861** |

The query separates the active quarter at an AUC of 0.665 against 0.495 for the same features moved to random places in the same box: a paired difference of +0.170 with a 95 % interval of [+0.102, +0.238], so the places the co-field picked are doing the work. The query is ahead of its own features in random places on 36 of 40 splits. The model it was read from reaches 0.861 on the same molecules.

Taking every split of every set together, the top tenth by the query holds 1.77 times as many actives as chance would put there, and the top tenth by the model 2.87 times, against a ceiling of 4.0 -- which is what a perfect ranking would reach when the actives are the top 25 %. The model is most of the way there and the query is a third of the way.

That gap is the price of the summary. A pharmacophore is four spheres and a rule for counting them; the model is every training molecule's field weighed by a fitted coefficient. What the query buys is that it leaves this program: it goes into a screening tool and runs over libraries no kernel method here could hold.

The matching rule used here is the simple one -- an atom of the right type inside the sphere -- with no conformer search, since the held-out molecules are already posed in the training frame. A real screening run has to generate conformations and will not do better than this.
