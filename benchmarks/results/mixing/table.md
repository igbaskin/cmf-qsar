# A posterior over the field weights

8 Sutherland sets, 5 random 70/30 splits each. The attenuation is chosen by the evidence with equal weights, the weights are then set four ways on the training part, and every choice is scored on the molecules held out. The posterior is 150 slice-sampling draws.

## Does weighing the fields predict better?

| set | n | uniform | align | alignf | evidence |
|---|---:|---:|---:|---:|---:|
| DHFR | 361 | +0.663 | +0.660 | +0.660 | +0.644 |
| COX2 | 282 | +0.511 | +0.515 | +0.517 | +0.512 |
| BZR | 147 | +0.343 | +0.350 | +0.358 | +0.352 |
| ACE | 114 | +0.529 | +0.535 | +0.541 | +0.586 |
| AChE | 111 | +0.558 | +0.565 | +0.566 | +0.537 |
| THR | 88 | +0.540 | +0.530 | +0.520 | +0.619 |
| THER | 76 | +0.371 | +0.360 | +0.350 | +0.363 |
| GPB | 66 | +0.336 | +0.381 | +0.365 | +0.265 |
| **mean** |  | **+0.481** | **+0.487** | **+0.485** | **+0.485** |

Against equal weights, the evidence optimum is +0.003 in R²pred with a 95 % interval of [-0.026, +0.035], which contains zero. This is item D6 again: the evidence takes whatever freedom it is given over the shape of the kernel, improves its own number, and leaves prediction where it was.

## Is the posterior right about which fields matter?

Each field's Bayes factor against *this field does nothing*, beside what actually happens to the held-out molecules when it is removed:

| field | mean share | 90 % interval | Bayes factor | ΔR²pred without it |
|---|---:|---:|---:|---:|
| Electrostatic | 0.179 | [0.024, 0.432] | +1.4 | -0.032 |
| Steric | 0.337 | [0.091, 0.663] | +2.8 | -0.012 |
| Hydrophobic | 0.184 | [0.021, 0.460] | +1.1 | -0.006 |
| H-bond acceptor | 0.159 | [0.032, 0.385] | +1.9 | -0.018 |
| H-bond donor | 0.142 | [0.025, 0.349] | +1.5 | -0.003 |

Over all 200 (set, split, field) cases the Bayes factor and the cost of removing the field correlate at Spearman ρ **+0.143** -- the sign is set so that positive means *the evidence was right*: a field it calls indispensable is one whose removal costs the held-out molecules.

That is weaker than the interpretation claim deserves. The Bayes factor orders the fields by what the training activities say, and the held-out molecules do not fully agree; read it as what this training set can see rather than as a property of the chemistry.

A note on the intervals. They are wide, and that is the answer rather than a failure of the sampler: on a congeneric series the field kernels are strongly correlated, so many mixtures explain the activities about equally well, and a point estimate that hides this -- any of the three point estimates above -- is claiming a precision the data do not support.
