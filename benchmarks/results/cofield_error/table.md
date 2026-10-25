# Does the co-field's error bar know which lobes are real

8 Sutherland sets. Each model's co-fields are computed on a 0.8 Å grid together with their closed-form standard deviation, and the mask keeps what stands more than 2 of them from zero. The training set is then resampled with replacement 30 times and refitted; a voxel is **stable** when the sign of its co-field survives that. Only the drawn part of each picture is counted -- a tenth of the peak and above, which is well below any isosurface anyone draws.

| set | n | q² | kept by the mask | stability kept | stability cut | difference |
|---|---:|---:|---:|---:|---:|---:|
| DHFR | 361 | +0.637 | 26% | 0.957 | 0.880 | +0.077 |
| COX2 | 282 | +0.543 | 43% | 0.977 | 0.904 | +0.073 |
| BZR | 147 | +0.388 | 36% | 0.982 | 0.920 | +0.062 |
| ACE | 114 | +0.657 | 33% | 0.972 | 0.915 | +0.057 |
| AChE | 111 | +0.564 | 17% | 0.991 | 0.888 | +0.103 |
| THR | 88 | +0.658 | 32% | 0.970 | 0.910 | +0.059 |
| THER | 76 | +0.436 | 24% | 0.992 | 0.925 | +0.067 |
| GPB | 66 | +0.580 | 30% | 0.983 | 0.902 | +0.081 |
| **mean** |  |  | **30%** | **0.978** | **0.906** | **+0.072** |

The mask keeps 30% of what would have been drawn. What it keeps holds its sign under resampling 97.8% of the time against 90.6% for what it cuts: a difference of +0.072 with a 95 % interval of [+0.064, +0.083], in the same direction on 8 of 8 sets.

By field, over every set:

| field | kept by the mask | stability kept | stability cut | bootstrap sd / closed form | rank correlation of the two |
|---|---:|---:|---:|---:|---:|
| electrostatic | 28% | 0.981 | 0.910 | 0.62 | +0.83 |
| steric | 39% | 0.993 | 0.920 | 0.62 | +0.85 |
| hydrophobic | 31% | 0.984 | 0.919 | 0.63 | +0.82 |
| hba | 27% | 0.971 | 0.887 | 0.62 | +0.82 |
| hbd | 26% | 0.957 | 0.891 | 0.65 | +0.75 |

The last two columns are what the error bar can and cannot claim. The bootstrap moves the molecules as well as the activities, and it moves the co-field about 0.6 times as far as the closed form does -- less far than the closed form allows for, so the bar drawn here is conservative: cutting at 2 of its standard deviations is cutting at about 3.2 of the bootstrap's. That follows from the noise level it is given -- the out-of-fold error, which is the assay's noise and the model's own error together, and so is larger than the activities' noise alone. Pass a smaller sigma to cut less.

What the formula does get right is *where* the uncertainty is: the two orderings agree across the grid at a rank correlation of +0.82. That is what a mask needs, because a mask is an ordering with a line drawn through it, and it is why the kept region holds its sign under a perturbation the formula never modelled.
