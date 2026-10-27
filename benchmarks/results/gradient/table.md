# The gradient of a prediction

Written by `python benchmarks/run_gradient.py`.

## Is it the derivative? Relative disagreement with central finite differences

| set | molecule | atoms | step 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 |
|---|---|---:|---:|---:|---:|---:|---:|
| steroids | prednisolone | 54 | 3.4e-04 | 3.4e-06 | 3.4e-08 | 6.2e-09 | 4.1e-08 |
| THR | 01 | 75 | 5.4e-05 | 5.4e-07 | 5.4e-09 | 2.3e-08 | 2.1e-07 |

The disagreement falls as the square of the step and rises again below about 1e-4 angstrom, where round-off in the predictions takes over. That shape is the signature of a correct gradient; the smallest value in each row is the accuracy of the comparison, not of the gradient.

## What does it cost?

| set | atoms | degrees of freedom | analytic | finite differences | ratio |
|---|---:|---:|---:|---:|---:|
| steroids | 54 | 432 | 5.6 ms | 1068 ms | 190x |
| THR | 75 | 600 | 28.8 ms | 7448 ms | 258x |

## What happens if you follow it?

Steepest ascent on the atom positions, no atom moving more than 0.05 A per step.

### steroids, molecule prednisolone (measured +7.51)

| step | prediction | largest displacement | rmsd | posterior sd | worst bond length change |
|---:|---:|---:|---:|---:|---:|
| 0 | +7.934 | 0.00 A | 0.00 A | 0.234 | 0.00 A |
| 1 | +8.149 | 0.05 A | 0.01 A | 0.247 | 0.06 A |
| 2 | +8.349 | 0.10 A | 0.03 A | 0.268 | 0.12 A |
| 5 | +8.875 | 0.25 A | 0.07 A | 0.356 | 0.29 A |
| 10 | +9.534 | 0.50 A | 0.12 A | 0.508 | 0.54 A |
| 20 | +10.329 | 1.00 A | 0.22 A | 0.701 | 0.80 A |

### THR, molecule 01 (measured +8.38)

| step | prediction | largest displacement | rmsd | posterior sd | worst bond length change |
|---:|---:|---:|---:|---:|---:|
| 0 | +7.321 | 0.00 A | 0.00 A | 0.281 | 0.00 A |
| 1 | +7.420 | 0.05 A | 0.01 A | 0.275 | 0.02 A |
| 2 | +7.518 | 0.10 A | 0.02 A | 0.271 | 0.03 A |
| 5 | +7.817 | 0.25 A | 0.05 A | 0.268 | 0.08 A |
| 10 | +8.324 | 0.50 A | 0.10 A | 0.300 | 0.18 A |
| 20 | +9.336 | 0.97 A | 0.22 A | 0.455 | 0.44 A |

The prediction rises, and so does the posterior standard deviation: the molecule is leaving the training set while the number is getting better. The last column says what else is happening -- the bond lengths are being pulled apart, because nothing in the gradient knows that a molecule has bonds. Both columns are the reason this gradient is a term to add to a generator or a force field that enforces geometry, and not an optimiser to run on its own.
