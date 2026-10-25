# Does the co-field cosine say whether the SAR transfers

Every set is split into its chemotypes, a model is fitted on each group of at least 15 molecules with the attenuation shared across the set (it has to be, or two co-fields are not functions of the same shape), and every ordered pair is scored twice: by the cosine of the two co-fields, and by how well one model orders the other group's molecules (Spearman's rho). The largest chemotype is also cut in two at random, as the control for what agreement looks like when the SAR really is the same.

| set | groups | pairs | mean cosine | mean transfer ρ | ρ of the two |
|---|---:|---:|---:|---:|---:|
| DHFR | 5 | 20 | +0.166 | +0.324 | +0.731 |
| COX2 | 5 | 20 | +0.201 | +0.385 | +0.385 |
| BZR | 3 | 6 | +0.036 | +0.023 | +0.257 |
| ACE | 3 | 6 | +0.098 | +0.104 | +0.371 |
| AChE | 3 | 6 | +0.251 | +0.502 | +0.657 |
| THR | 2 | 2 | +0.280 | +0.429 | +nan |
| THER | 2 | 2 | +0.370 | +0.582 | +nan |
| GPB | 2 | 2 | +0.207 | +0.714 | +nan |
| **all** |  | 64 | **+0.178** | **+0.334** | **+0.667** |

**The control first.** Two halves of one chemotype, which share their SAR by construction, agree at a mean cosine of **+0.401** and transfer at rho **+0.673** over 16 pairs. Two different chemotypes of the same target agree at **+0.178** and transfer at **+0.334**. The cosine separates the two cases by +0.223, which is what it is for: two models of one SAR look alike and two models of two SARs do not.

**Within the chemotype pairs, the cosine and the transfer correlate at +0.667** over 64 pairs. That is the claim of the item, measured: a table of co-field overlaps is a map of SAR transfer.

A caveat that cannot be removed by any amount of care: the whole comparison rests on the two training sets being posed in one frame. Here they are, because they come from one file; between two files they would have to be aligned first, and nothing in the cosine would reveal that they had not been.
