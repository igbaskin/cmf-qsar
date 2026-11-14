# What the diagnosis says, on sets whose answers are known

Each route scored by the same 10-fold cross-validation; the recommendation reads
the paired bootstrap intervals, not the point estimates. The warnings column counts
the questions the set failed.

| set | n | aligned | counts | invariant | topomer | descriptors | warnings | recommended |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| DHFR | 361 | +0.632 | +0.351 | +0.641 | +0.009 | +0.163 | 4 | invariant |
| COX2 | 282 | +0.509 | +0.319 | +0.523 | +0.011 | +0.166 | 3 | invariant |
| corrosion/0_c1ccccc1 | 244 | +0.332 | +0.442 | +0.316 | +0.388 | +0.965 | 8 | counts |
| BZR | 147 | +0.281 | +0.195 | +0.406 | +0.555 | +0.134 | 5 | invariant |
| ACE | 114 | +0.605 | +0.660 | +0.677 | +0.460 | +0.644 | 5 | descriptors |
| AChE | 111 | +0.511 | +0.147 | +0.487 | +0.453 | +0.217 | 3 | invariant |
| THR | 88 | +0.616 | +0.051 | +0.511 | -0.507 | +0.057 | 3 | aligned, with the frame checked |
| THER | 76 | +0.437 | +0.229 | +0.354 | +0.290 | +0.233 | 4 | aligned, with the frame checked |
| GPB | 66 | +0.569 | +0.294 | +0.463 | +0.541 | +0.315 | 4 | aligned, with the frame checked |
| am_eu | 46 | +0.666 | -0.093 | +0.423 | - | +0.103 | 3 | aligned, with the frame checked |
| corrosion/7_c1nncs1 | 29 | -0.089 | -0.228 | -0.192 | - | +0.992 | 7 | none |
| corrosion/13_c1c_nH_nn1 | 25 | +0.267 | +0.066 | +0.466 | - | +0.984 | 8 | invariant |
| corrosion/161_c1ccc2scnc2c1 | 20 | -0.072 | -0.013 | +0.375 | - | +0.992 | 7 | invariant |
