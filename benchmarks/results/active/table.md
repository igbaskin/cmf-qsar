# Which molecule to make next

8 Sutherland sets, 10 runs each (80 in all). Every run starts from 20 molecules drawn at random and holding no hit, and acquires 23 more one at a time; alpha and lambda are reselected by the evidence at every step from what has been acquired and nothing else. A **hit** is a molecule in the top 5 % of its set. Every strategy sees the same starting sets, so the comparison is paired.

## Finding an active

| rule | best found (of the best possible) | runs that found a hit | median acquisitions to the first hit |
|---|---:|---:|---:|
| random | 8.39 (92.2%) | 79% | 6 |
| maxmin | 8.30 (90.9%) | 64% | 9 |
| variance | 8.29 (90.6%) | 71% | 9 |
| greedy | 8.63 (95.0%) | 100% | 4 |
| ucb | 8.77 (96.6%) | 95% | 4 |
| ei | 8.65 (95.2%) | 89% | 3 |

## Improving the model

| rule | mean RMSE over the run | RMSE at the end | q² at the end |
|---|---:|---:|---:|
| random | 1.122 | 1.064 | +0.366 |
| maxmin | 1.108 | 1.047 | +0.384 |
| variance | 1.108 | 1.042 | +0.386 |
| greedy | 1.179 | 1.195 | +0.158 |
| ucb | 1.144 | 1.127 | +0.279 |
| ei | 1.121 | 1.086 | +0.331 |

## Against picking at random, paired run by run

| rule | hits found, difference | mean RMSE, difference |
|---|---:|---:|
| maxmin | -0.150 [-0.275, -0.037] | -0.014 [-0.035, +0.005] |
| variance | -0.075 [-0.212, +0.050] | -0.014 [-0.034, +0.006] |
| greedy | +0.212 [+0.125, +0.300] | +0.057 [+0.038, +0.078] |
| ucb | +0.163 [+0.062, +0.275] | +0.022 [+0.001, +0.044] |
| ei | +0.100 [-0.013, +0.225] | -0.001 [-0.022, +0.021] |

**ucb** finds the best molecules and **maxmin** builds the best model, and they are not the same rule. That is why the two tables are kept apart: an acquisition rule cannot be judged without saying what the acquiring is for.

Three things in the numbers are worth more than the ranking.

**The uncertainty rules and the diversity rule are one rule.** variance and maxmin come out within 0.000 RMSE of each other and within 0.3% of the same activity found. That is item E5 again, seen from a different direction: the posterior standard deviation of a CMF model is very nearly the distance to the training set in other units, so acquiring by the model's uncertainty and acquiring by field-space diversity are the same experiment with two names. The Gaussian process is not adding information here; it is re-expressing geometry.

**Exploring gains nothing measurable.** The best of the exploration rules (maxmin) reaches 1.108 RMSE against 1.122 for picking at random, a paired difference of -0.014 with a 95 % interval of [-0.034, +0.006] -- zero is inside it. Acquiring the molecule the model is least sure of means acquiring the molecule least like the rest of the series, and what is learned from an outlier does not transfer to the molecules anyone will ask about next. Uncertainty sampling is a rule for covering a space, and these sets are series to be understood rather than spaces to be covered.

**Greedy is not as bad as it should be, and that is luck.** It ignores the posterior, so a candidate unlike anything made is predicted at the training mean and cannot be told from a real answer -- the failure the screening rank guards against by subtracting the spread rather than adding it. On these sets it finds 95.0% of the best activity against 96.6% for ucb. The sets are series of analogues, where nothing is far enough from the training molecules for the training mean to be returned; on a library of diverse candidates that protection is gone, and the expected-improvement rule, which at least makes its optimism depend on the posterior, is the one to use.
