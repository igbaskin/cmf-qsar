# The conditional HSIC test: does the activity depend on the fields beyond a reference?

## Calibration and power on synthetic activities

The 8 Sutherland sets, the joint field kernel at alpha = 0.3. y = g(z) + noise with half of the variance from g, plus a field direction the reference does not explain carrying the share given. At share 0 the activity depends on the reference only, so every rejection is a false positive and a calibrated test rejects 5% of the time. Rejections at the 5 % level, pooled over the sets (400 replicates per set; the model comparison 40 per set, for the heavy-atom count and the 6 sets of at most 150 molecules only).

| reference | strata | share of the field direction | plain test | conditional test | conditional test, line removed within bins | plain test on the straight-line residual | model comparison |
|---|---|---:|---:|---:|---:|---:|---:|
| heavy atoms | values | 0% | 90.8% **too many** | 5.2% | -- | 27.6% **too many** | 0.0% |
| heavy atoms | values | 5% | 98.0% | 83.9% | -- | 76.2% | 2.5% |
| heavy atoms | values | 15% | 99.4% | 99.7% | -- | 92.5% | 34.2% |
| logP, 5 bins | bins | 0% | 98.7% **too many** | 22.2% **too many** | 1.5% | 29.0% **too many** | -- |
| logP, 5 bins | bins | 5% | 92.4% | 81.9% | 69.8% | 81.8% | -- |
| logP, 5 bins | bins | 15% | 99.5% | 99.6% | 98.8% | 99.6% | -- |
| logP, 10 bins | bins | 0% | 98.7% **too many** | 18.7% **too many** | 2.4% | 29.0% **too many** | -- |
| logP, 10 bins | bins | 5% | 91.8% | 81.3% | 72.4% | 78.2% | -- |
| logP, 10 bins | bins | 15% | 98.1% | 99.6% | 98.5% | 99.2% | -- |

**too many** marks a false-positive rate more than two binomial standard errors above 5 %.

## The measured activities

p-values from 999 permutations. The model comparison is the gain in q2 of adding the fields to a Gaussian-kernel model on the reference, out-of-fold over five fold seeds, with its paired 95 % bootstrap interval over the molecules. *Reference alignment* is how much of the field the reference itself explains.

| set | n | p, fields | reference | reference alignment | p, beyond the reference | alignment beyond it | p, beyond it with the line removed within bins | p, straight-line residual | gain in q2 of adding the fields |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ACE | 114 | 0.001 | heavy atoms | 0.37 | 0.012 | 0.095 | -- | 0.001 | +0.106 [+0.021, +0.192] |
| ACE | 114 | 0.001 | logP, 5 bins | 0.21 | 0.001 | 0.258 | 0.001 | 0.001 | +0.504 [+0.335, +0.687] |
| ACE | 114 | 0.001 | logP, 10 bins | 0.21 | 0.001 | 0.266 | 0.001 | 0.001 | +0.504 [+0.335, +0.687] |
| AChE | 111 | 0.001 | heavy atoms | 0.22 | 0.004 | 0.105 | -- | 0.011 | +0.280 [+0.158, +0.414] |
| AChE | 111 | 0.001 | logP, 5 bins | 0.26 | 0.001 | 0.160 | 0.001 | 0.001 | +0.511 [+0.292, +0.740] |
| AChE | 111 | 0.001 | logP, 10 bins | 0.26 | 0.001 | 0.159 | 0.001 | 0.001 | +0.511 [+0.292, +0.740] |
| BZR | 147 | 0.001 | heavy atoms | 0.38 | 0.004 | 0.074 | -- | 0.04 | +0.279 [+0.126, +0.435] |
| BZR | 147 | 0.001 | logP, 5 bins | 0.11 | 0.001 | 0.084 | 0.001 | 0.004 | +0.326 [+0.181, +0.480] |
| BZR | 147 | 0.001 | logP, 10 bins | 0.11 | 0.001 | 0.109 | 0.001 | 0.004 | +0.326 [+0.181, +0.480] |
| COX2 | 282 | 0.001 | heavy atoms | 0.06 | 0.001 | 0.111 | -- | 0.001 | +0.545 [+0.431, +0.671] |
| COX2 | 282 | 0.001 | logP, 5 bins | 0.26 | 0.001 | 0.265 | 0.001 | 0.001 | +0.377 [+0.280, +0.477] |
| COX2 | 282 | 0.001 | logP, 10 bins | 0.26 | 0.001 | 0.279 | 0.001 | 0.001 | +0.377 [+0.280, +0.477] |
| DHFR | 361 | 0.001 | heavy atoms | 0.27 | 0.001 | 0.121 | -- | 0.001 | +0.481 [+0.392, +0.567] |
| DHFR | 361 | 0.001 | logP, 5 bins | 0.19 | 0.001 | 0.137 | 0.001 | 0.001 | +0.482 [+0.398, +0.579] |
| DHFR | 361 | 0.001 | logP, 10 bins | 0.19 | 0.001 | 0.123 | 0.001 | 0.001 | +0.482 [+0.398, +0.579] |
| GPB | 66 | 0.001 | heavy atoms | 0.26 | 0.001 | 0.219 | -- | 0.001 | +0.374 [+0.164, +0.610] |
| GPB | 66 | 0.001 | logP, 5 bins | 0.30 | 0.002 | 0.201 | 0.002 | 0.001 | +0.481 [+0.209, +0.786] |
| GPB | 66 | 0.001 | logP, 10 bins | 0.30 | 0.001 | 0.207 | 0.021 | 0.001 | +0.481 [+0.209, +0.786] |
| THER | 76 | 0.001 | heavy atoms | 0.35 | 0.416 | 0.052 | -- | 0.01 | +0.158 [+0.023, +0.294] |
| THER | 76 | 0.001 | logP, 5 bins | 0.30 | 0.001 | 0.197 | 0.001 | 0.001 | +0.402 [+0.231, +0.585] |
| THER | 76 | 0.001 | logP, 10 bins | 0.30 | 0.001 | 0.195 | 0.001 | 0.001 | +0.402 [+0.231, +0.585] |
| THR | 88 | 0.001 | heavy atoms | 0.27 | 0.001 | 0.185 | -- | 0.001 | +0.324 [+0.161, +0.499] |
| THR | 88 | 0.001 | logP, 5 bins | 0.29 | 0.001 | 0.169 | 0.001 | 0.001 | +0.637 [+0.377, +0.892] |
| THR | 88 | 0.001 | logP, 10 bins | 0.29 | 0.001 | 0.169 | 0.002 | 0.001 | +0.637 [+0.377, +0.892] |

Agreement of the conditional test at 5 % (for logP, the version with the line removed within bins) with the model comparison (positive when its interval excludes zero): heavy atoms 7 of 8, logP, 5 bins 8 of 8, logP, 10 bins 8 of 8.

## The ethers of item EC7

The alignment-free field kernel at alpha = 0.5 of the 1,050 ethers in the conformation of the lithium complex. The stored comparison is the gain of the field model over the Gaussian model on the contacts at the relaxed ion, mean absolute error in eV, random folds (`benchmarks/results/ether/contacts.json`).

| ion | n | p, fields | p, beyond the contacts | alignment beyond the contacts | p, beyond the formula's oxygens | stored: fields better than the contacts by |
|---|---:|---:|---:|---:|---:|---:|
| Li+ | 1048 | 0.001 | 0.001 | 0.139 | 0.001 | +0.052 eV [+0.042, +0.062] |
| Na+ | 980 | 0.001 | 0.001 | 0.068 | 0.001 | +0.043 eV [+0.033, +0.053] |
| Mg2+ | 982 | 0.001 | 0.001 | 0.152 | 0.001 | +0.113 eV [+0.096, +0.131] |

A null on these molecules (the lithium set, an activity of the contact count alone plus noise, 100 replicates): the plain test rejects 100%, the conditional test 1%, the plain test on the straight-line residual 100%.

