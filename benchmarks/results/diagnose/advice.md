# Following the advice: was it right?

The diagnosis is run on the training part alone, so the advice never sees the
molecules it is judged on; every route is then fitted on that part and asked about
the rest. `advice` is the route recommended, `best` the route that actually won.

| set | split | advised | R2pred advised | best route | R2pred best | cost |
|---|---:|---|---:|---|---:|---:|
| ACE | 0 | descriptors | +0.712 | descriptors | +0.712 | +0.000 |
| ACE | 1 | invariant | +0.595 | descriptors | +0.726 | +0.131 |
| ACE | 2 | descriptors | +0.646 | descriptors | +0.646 | +0.000 |
| AChE | 0 | aligned | +0.658 | topomer | +0.727 | +0.069 |
| AChE | 1 | aligned | +0.599 | aligned | +0.599 | +0.000 |
| AChE | 2 | invariant | +0.443 | aligned | +0.632 | +0.189 |
| BZR | 0 | invariant | +0.178 | topomer | +0.583 | +0.405 |
| BZR | 1 | invariant | +0.374 | topomer | +0.680 | +0.305 |
| BZR | 2 | invariant | +0.253 | topomer | +0.671 | +0.418 |
| COX2 | 0 | invariant | +0.553 | invariant | +0.553 | +0.000 |
| COX2 | 1 | invariant | +0.507 | invariant | +0.507 | +0.000 |
| COX2 | 2 | invariant | +0.593 | invariant | +0.593 | +0.000 |
| DHFR | 0 | aligned | +0.587 | aligned | +0.587 | +0.000 |
| DHFR | 1 | invariant | +0.711 | invariant | +0.711 | +0.000 |
| DHFR | 2 | invariant | +0.615 | invariant | +0.615 | +0.000 |
| GPB | 0 | aligned | +0.475 | invariant | +0.570 | +0.095 |
| GPB | 1 | aligned | +0.635 | aligned | +0.635 | +0.000 |
| GPB | 2 | aligned | +0.686 | aligned | +0.686 | +0.000 |
| THER | 0 | descriptors | +0.276 | aligned | +0.555 | +0.280 |
| THER | 1 | descriptors | +0.270 | aligned | +0.509 | +0.239 |
| THER | 2 | aligned | +0.396 | invariant | +0.509 | +0.113 |
| THR | 0 | aligned | +0.524 | invariant | +0.573 | +0.049 |
| THR | 1 | invariant | +0.482 | aligned | +0.569 | +0.087 |
| THR | 2 | invariant | +0.376 | aligned | +0.604 | +0.228 |
| corrosion/0_c1ccccc1 | 0 | descriptors | +0.963 | descriptors | +0.963 | +0.000 |
| corrosion/0_c1ccccc1 | 1 | descriptors | +0.976 | descriptors | +0.976 | +0.000 |
| corrosion/0_c1ccccc1 | 2 | counts | +0.241 | topomer | +0.447 | +0.205 |
| corrosion/7_c1nncs1 | 0 | aligned | +0.326 | aligned | +0.326 | +0.000 |
| corrosion/7_c1nncs1 | 1 | aligned | -0.702 | invariant | +0.030 | +0.732 |
| corrosion/7_c1nncs1 | 2 | aligned | -0.076 | counts | +0.047 | +0.123 |
| corrosion/13_c1c_nH_nn1 | 0 | descriptors | +0.992 | descriptors | +0.992 | +0.000 |
| corrosion/13_c1c_nH_nn1 | 1 | descriptors | +0.974 | descriptors | +0.974 | +0.000 |
| corrosion/13_c1c_nH_nn1 | 2 | invariant | +0.515 | invariant | +0.515 | +0.000 |
| corrosion/161_c1ccc2scnc2c1 | 0 | invariant | -0.091 | counts | -0.079 | +0.012 |
| corrosion/161_c1ccc2scnc2c1 | 1 | descriptors | +0.976 | descriptors | +0.976 | +0.000 |
| corrosion/161_c1ccc2scnc2c1 | 2 | invariant | +0.489 | invariant | +0.489 | +0.000 |
| am_eu | 0 | aligned | +0.823 | aligned | +0.823 | +0.000 |
| am_eu | 1 | aligned | +0.491 | invariant | +0.564 | +0.073 |
| am_eu | 2 | aligned | +0.507 | aligned | +0.507 | +0.000 |

The advice named the best route on **21 of 39** splits; when it
did not, it cost +0.096 in R2pred on average (worst +0.732).

