# The field weights of a series, learned from the other series

38 Mittal sets, 10 random 70/30 splits each. For every set a two-level prior over the field weights is learned from the other 37 sets (`cmfqsar.mixing.pool_priors`, Gaussian form, 300 posterior draws per set), never from the set itself. The attenuation is chosen on the training part by the evidence with equal weights; the weights are then set five ways on the training part and scored on the molecules held out (R²pred, mean over the splits).

| set | n | uniform | align | evidence | population | pooled | prior τ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DHFR | 397 | +0.449 | +0.438 | +0.451 | +0.428 | +0.447 | 0.61 |
| COX2 | 322 | +0.100 | +0.119 | +0.100 | +0.104 | +0.104 | 0.83 |
| BZR | 163 | +0.331 | +0.351 | +0.358 | +0.325 | +0.341 | 0.80 |
| HPTP | 135 | +0.259 | +0.267 | +0.307 | +0.327 | +0.329 | 0.81 |
| AMPH1 | 130 | -0.065 | -0.046 | -0.083 | -0.061 | -0.063 | 0.79 |
| EDC | 122 | +0.229 | +0.211 | +0.254 | +0.269 | +0.276 | 0.81 |
| ACE | 114 | +0.412 | +0.402 | +0.487 | +0.455 | +0.483 | 0.50 |
| HIVPR | 113 | -0.082 | -0.116 | -0.127 | -0.116 | -0.120 | 0.79 |
| ACHE | 111 | +0.097 | +0.085 | +0.120 | +0.116 | +0.131 | 0.82 |
| HIVRT | 99 | +0.277 | +0.284 | +0.259 | +0.276 | +0.254 | 0.81 |
| ATA | 94 | +0.070 | +0.064 | +0.025 | +0.041 | +0.043 | 0.80 |
| COMT | 92 | +0.095 | +0.090 | +0.051 | +0.121 | +0.106 | 0.81 |
| THR | 88 | -0.103 | -0.108 | -0.086 | -0.085 | -0.114 | 0.74 |
| AI | 78 | +0.533 | +0.539 | +0.544 | +0.552 | +0.557 | 0.79 |
| THERM | 76 | -0.015 | -0.014 | -0.035 | +0.017 | -0.013 | 0.80 |
| GPB | 66 | +0.185 | +0.182 | +0.171 | +0.202 | +0.179 | 0.80 |
| ECR | 50 | -0.052 | -0.056 | -0.055 | -0.064 | -0.047 | 0.79 |
| DAT | 42 | +0.058 | +0.060 | -0.375 | -0.127 | -0.263 | 0.77 |
| DIAZEPAM | 42 | +0.551 | +0.512 | +0.488 | +0.498 | +0.476 | 0.79 |
| GSK3B | 42 | +0.326 | +0.333 | +0.290 | +0.382 | +0.319 | 0.83 |
| TCHK | 42 | -0.138 | -0.142 | -0.165 | -0.104 | -0.165 | 0.79 |
| EV71 | 41 | -0.970 | -0.938 | -0.833 | -0.851 | -0.876 | 0.79 |
| PTC | 40 | -0.039 | -0.057 | -0.074 | -0.061 | -0.064 | 0.81 |
| KOA | 39 | +0.020 | -0.009 | -0.153 | -0.049 | -0.067 | 0.80 |
| YOPH | 39 | +0.070 | -0.042 | +0.092 | +0.222 | +0.198 | 0.80 |
| DR | 38 | -0.075 | -0.089 | -0.130 | -0.012 | -0.090 | 0.78 |
| CBRA | 31 | -0.085 | -0.077 | -0.093 | -0.033 | -0.041 | 0.79 |
| GHS | 31 | -0.295 | -0.291 | -0.347 | -0.353 | -0.338 | 0.79 |
| ACHE_TOX | 30 | -0.495 | -0.542 | -0.489 | -0.476 | -0.517 | 0.79 |
| MX | 29 | +0.401 | +0.375 | +0.389 | +0.410 | +0.401 | 0.78 |
| PDE | 29 | -0.430 | -0.467 | -0.466 | -0.354 | -0.369 | 0.79 |
| ARB | 28 | -0.076 | -0.096 | -0.091 | -0.047 | -0.048 | 0.80 |
| D2A | 26 | -0.625 | -0.544 | -0.639 | -0.833 | -0.692 | 0.75 |
| TP2A | 25 | -0.282 | -0.243 | -0.231 | -0.299 | -0.291 | 0.78 |
| VATP | 23 | -0.001 | +0.006 | +0.059 | -0.061 | -0.066 | 0.80 |
| STEROIDS | 21 | +0.179 | +0.169 | +0.124 | +0.183 | +0.170 | 0.79 |
| RYR | 17 | -0.594 | -0.482 | -0.623 | -0.432 | -0.521 | 0.79 |
| APZ | 16 | +0.136 | +0.146 | -0.088 | +0.246 | +0.237 | 0.79 |
| **mean, all sets** | 38 | **+0.009** | **+0.007** | **-0.016** | **+0.020** | **+0.008** |  |
| **mean, n ≥ 40** | 23 | **+0.109** | **+0.107** | **+0.090** | **+0.115** | **+0.101** |  |
| **mean, n < 40** | 15 | **-0.144** | **-0.146** | **-0.179** | **-0.126** | **-0.136** |  |

## Paired differences

Mean difference in R²pred over the sets, with a 95 % bootstrap interval over sets:

| comparison | all sets | n ≥ 40 | n < 40 |
|---|---:|---:|---:|
| pooled − uniform | -0.002 [-0.026, +0.019] | -0.008 [-0.044, +0.017] | +0.008 [-0.022, +0.040] |
| pooled − evidence | +0.024 [+0.004, +0.048] | +0.011 [+0.000, +0.024] | +0.043 [-0.003, +0.099] |
| population − uniform | +0.011 [-0.013, +0.033] | +0.006 [-0.019, +0.027] | +0.018 [-0.027, +0.061] |
| pooled − population | -0.012 [-0.025, +0.001] | -0.014 [-0.030, -0.002] | -0.010 [-0.032, +0.017] |
| evidence − uniform | -0.025 [-0.055, -0.001] | -0.019 [-0.065, +0.014] | -0.035 [-0.075, -0.003] |

The pooled weights beat the series' own evidence weights on 26 of 38 sets and equal weights on 19 of 38.

## What the collection says a series needs

The prior is learned 38 times, each time without one set. The spread of the log-shares between series is τ = 0.79 (range 0.50 to 0.83); the fixed prior has 1.5. The importance-reweighting estimate, which makes no Gaussian assumption, gives τ = 0.86 (range 0.77 to 0.92).

| field | typical share (Gaussian) | range over the leave-one-out priors | typical share (importance) |
|---|---:|---:|---:|
| Electrostatic | 0.109 | [0.095, 0.122] | 0.126 |
| Steric | 0.495 | [0.272, 0.564] | 0.457 |
| Hydrophobic | 0.238 | [0.187, 0.386] | 0.226 |
| H-bond acceptor | 0.102 | [0.077, 0.143] | 0.130 |
| H-bond donor | 0.056 | [0.043, 0.080] | 0.062 |

Equal shares would be 0.200.

## Reading

Pooling neither gains nor costs on average: against equal weights the pooled weights are -0.002 in R²pred with an interval of [-0.026, +0.019] that contains zero.
Against the series' own evidence weights the pooled weights gain +0.024 [+0.004, +0.048]: the prior from the other series removes the damage that fitting five weights on one series does.
The population weights alone, with nothing fitted on the series, are +0.011 [-0.013, +0.033] against equal weights.
On the 15 sets below 40 molecules, where a prior should matter most, pooled against uniform is +0.008 [-0.022, +0.040].
