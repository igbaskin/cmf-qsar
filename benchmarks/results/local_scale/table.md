# How far the local kernel goes (plan item LP5)

A per-atom target is a large problem where a per-molecule one is a small one. The question is what has to be given up to reach the size the free databases have.

## The invariant local kernel needs no approximation at all

Its feature map is finite: the field autocorrelation of an environment is 255 channels for the five physico-chemical fields, scaled per block and cosine-normalised, and the kernel is the inner product of those rows. So the ridge regression can be solved in the space of the coefficients rather than in the space of the samples. It is the same model, not an approximation of it.

5096 carbons in 400 molecules, 10 folds holding out whole molecules, ridge 1e-06:

| route | q2 | error (ppm) | kernel or rows | memory | the folds |
| --- | ---: | ---: | ---: | ---: | ---: |
| dual, an n by n kernel | +0.972645 | 8.8240 | 0 s | 0.19 GB | 13 s |
| primal, n by d rows | +0.972645 | 8.8240 | 0.0 s | 10 MB | 0.1 s |

The largest difference between the two sets of out-of-fold predictions is 1.5e-06 ppm, which is round-off. The memory is the difference that decides: the kernel grows with the square of the samples and the rows grow with the samples.
At 4 times this many carbons the kernel would be 3 GB and the rows 40 MB.
At 20 times this many carbons the kernel would be 77 GB and the rows 198 MB.

## What more data is worth

58705 carbons in 6000 molecules, from a set built from the nmrshiftdb2 dump that this repository does not ship (see the script this benchmark names). An exact kernel over them would be 25.7 GB, so the primal route is not a convenience here but the only one that runs. Each size is a subset of the next.

| molecules | carbons | q2 | error (ppm) | ridge | the folds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1105 | +0.953 | 10.84 | 1e-06 | 0.2 s |
| 200 | 2131 | +0.961 | 10.13 | 1e-05 | 0.3 s |
| 400 | 4113 | +0.964 | 9.73 | 1e-06 | 0.4 s |
| 800 | 8079 | +0.965 | 9.61 | 1e-06 | 1.0 s |
| 1600 | 15780 | +0.965 | 9.70 | 1e-06 | 1.7 s |
| 3200 | 31221 | +0.964 | 9.83 | 1e-06 | 3.2 s |
| 6000 | 58705 | +0.965 | 9.75 | 1e-06 | 6.8 s |

From 100 molecules to 6000 the error goes from 10.84 to 9.75 ppm, a factor of 1.11 for a factor of 60 in data.

**A flat curve says the error is not the data's.** What is left is the descriptor, and the cheapest test of that is a finer one on the same samples: the distance axis of the autocorrelation at 0.5 A per bin against two finer spacings.

| distance bins | channels | q2 | error (ppm) |
| ---: | ---: | ---: | ---: |
| every 0.5 A | 255 | +0.965 | 9.75 |
| every 0.25 A | 495 | +0.974 | 8.35 |
| every 0.125 A | 975 | +0.978 | 7.76 |

The finest grid gains 1.99 ppm, which is the larger part of what the data could not give: the error of this model is set by what the descriptor can tell apart, not by how many carbons it has seen.


## The kernel that does need landmarks

The posed overlap kernel of plan item LP3 has no finite feature map -- its features are the field itself, a function on space -- so there the dual form is the only exact one and a Nystrom approximation is the route the plan names. On the 1225 aromatic carbons of that item, with landmark environments drawn at random from the samples:

| landmarks | relative error of the kernel | q2 | error (ppm) |
| ---: | ---: | ---: | ---: |
| all 1225 (exact) | 0.000 | +0.899 | 4.217 |
| 25 | 0.072 | +0.801 | 5.908 |
| 50 | 0.031 | +0.859 | 4.970 |
| 100 | 0.001 | +0.883 | 4.539 |
| 200 | 0.000 | +0.899 | 4.203 |
| 400 | 0.000 | +0.898 | 4.231 |
| 800 | 0.000 | +0.899 | 4.217 |

**200 landmarks of 1225 samples come within 0.01 of the exact q2**, at 16 per cent of the rank. The environments of one site are alike, which is why so few span them.

