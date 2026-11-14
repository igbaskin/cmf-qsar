# Following the advice: was it right?

The diagnosis is run on the training part alone, so the advice never sees the
molecules it is judged on; every route is then fitted on that part and asked about
the rest. `advice` is the route recommended, `best` the route that actually won.

| set | split | advised | R2pred advised | best route | R2pred best | cost |
|---|---:|---|---:|---|---:|---:|
| ACE | 0 | descriptors | +0.712 | descriptors | +0.712 | +0.000 |
| ACE | 1 | invariant | +0.595 | descriptors | +0.726 | +0.131 |
| ACE | 2 | descriptors | +0.646 | descriptors | +0.646 | +0.000 |
| AChE | 0 | aligned | +0.658 | topomer | +0.754 | +0.096 |
| AChE | 1 | aligned | +0.599 | aligned | +0.599 | +0.000 |
| AChE | 2 | invariant | +0.443 | aligned | +0.632 | +0.189 |
| BZR | 0 | invariant | +0.178 | topomer | +0.536 | +0.358 |
| BZR | 1 | invariant | +0.374 | topomer | +0.668 | +0.294 |
| BZR | 2 | invariant | +0.253 | topomer | +0.644 | +0.391 |
| COX2 | 0 | invariant | +0.553 | invariant | +0.553 | +0.000 |
| COX2 | 1 | invariant | +0.507 | invariant | +0.507 | +0.000 |
| COX2 | 2 | invariant | +0.593 | invariant | +0.593 | +0.000 |
| DHFR | 0 | aligned | +0.587 | aligned | +0.587 | +0.000 |
| DHFR | 1 | invariant | +0.711 | invariant | +0.711 | +0.000 |
| DHFR | 2 | invariant | +0.615 | invariant | +0.615 | +0.000 |
| GPB | 0 | aligned | +0.475 | invariant | +0.570 | +0.095 |
| GPB | 1 | aligned | +0.635 | topomer | +0.650 | +0.015 |
| GPB | 2 | aligned | +0.686 | aligned | +0.686 | +0.000 |
| THER | 0 | descriptors | +0.276 | aligned | +0.555 | +0.280 |
| THER | 1 | descriptors | +0.270 | aligned | +0.509 | +0.239 |
| THER | 2 | aligned | +0.396 | invariant | +0.509 | +0.113 |
| THR | 0 | aligned | +0.524 | invariant | +0.573 | +0.049 |
| THR | 1 | invariant | +0.482 | aligned | +0.569 | +0.087 |
| THR | 2 | invariant | +0.376 | aligned | +0.604 | +0.228 |
| corrosion/0_c1ccccc1 | 0 | aligned | +0.405 | counts | +0.615 | +0.211 |
| corrosion/0_c1ccccc1 | 1 | counts | +0.451 | topomer | +0.480 | +0.029 |
| corrosion/0_c1ccccc1 | 2 | counts | +0.241 | topomer | +0.447 | +0.206 |
| corrosion/7_c1nncs1 | 0 | no route; scored as aligned | +0.326 | aligned | +0.326 | +0.000 |
| corrosion/7_c1nncs1 | 1 | no route; scored as aligned | -0.702 | invariant | +0.030 | +0.732 |
| corrosion/7_c1nncs1 | 2 | no route; scored as aligned | -0.076 | counts | +0.047 | +0.123 |
| corrosion/13_c1c_nH_nn1 | 0 | descriptors | +0.374 | invariant | +0.715 | +0.341 |
| corrosion/13_c1c_nH_nn1 | 1 | descriptors | +0.221 | aligned | +0.382 | +0.161 |
| corrosion/13_c1c_nH_nn1 | 2 | descriptors | +0.388 | invariant | +0.515 | +0.127 |
| corrosion/161_c1ccc2scnc2c1 | 0 | descriptors | +0.567 | descriptors | +0.567 | +0.000 |
| corrosion/161_c1ccc2scnc2c1 | 1 | descriptors | +0.780 | descriptors | +0.780 | +0.000 |
| corrosion/161_c1ccc2scnc2c1 | 2 | descriptors | +0.779 | descriptors | +0.779 | +0.000 |
| am_eu | 0 | aligned | +0.823 | aligned | +0.823 | +0.000 |
| am_eu | 1 | aligned | +0.491 | invariant | +0.564 | +0.073 |
| am_eu | 2 | aligned | +0.507 | aligned | +0.507 | +0.000 |

The advice named the best route on **16 of 39** splits; over all splits it cost
+0.117 in R2pred on average against the best route (worst +0.732).

In 3 of these splits the advice was that no route is worth a model. A user who builds a model anyway builds the aligned one, so such a split is scored as the aligned model, for the advice and for the rule "always align" alike. In 2 of them no route reached an R2pred of 0.1 on the held-out molecules.

## Against always taking the aligned model

No split advised a descriptor route that reads the activity from the columns of the file (`validation_splits` looks for such a split and marks it).

| | splits | sets | advice names the best route | mean cost of the advice | the aligned model is the best route | mean cost of always aligning | advice minus aligned, mean over the sets [95 %] | sets better / worse |
|---|---:|---:|---:|---:|---:|---:|---|---|
| all splits | 39 | 13 | 16 | 0.117 | 12 | 0.175 | +0.058 [-0.048, +0.225] | 5 / 5 |

A set counts as better or worse when the mean difference over its splits is more than 0.02 in R2pred.

