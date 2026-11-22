# The co-field as an atom feature on lithium hops (ND14c, LiTraj sample)

Written by `benchmarks/run_device_hops.py --report`. 1801 hops of 393 structures, the barrier of the bond-valence site-energy model in eV, ten folds that hold out structures. In every fold 15 % of the other structures are the validation part, and the rest is the fitting part. "Atom network" is the network of `cmfqsar/device.py`: one network on every atom of the environment of the hop, a sum over the atoms, a second network. The co-field is that of the ridge regression of the first row. The standard deviation of the barriers is 1.905 eV.

| Model | q2 over all hops | Mean absolute error, eV | Root mean square error, eV |
|---|---|---|---|
| ridge regression on the fields | +0.566 | 0.742 | 1.254 |
| atom network: field weights of the atom | +0.706 | 0.564 | 1.033 |
| atom network: positions | +0.710 | 0.535 | 1.026 |
| atom network: co-field | +0.776 | 0.514 | 0.900 |
| atom network: co-field, not out-of-fold | +0.792 | 0.490 | 0.868 |
| atom network: co-field and positions | +0.767 | 0.519 | 0.919 |
| BT7: the same kernel model, ridge chosen on the reported folds, fitted on all nine other folds | +0.580 | 0.730 | 1.234 |
| BT7: random forest on the rows of the rotation-invariant description | +0.844 | 0.374 | 0.753 |
| BT7: boosted forest on the three-distance rows, the best model of the check | +0.846 | 0.354 | 0.746 |

## Paired comparisons on the same hops

The difference is the mean absolute error of the first model minus that of the second, in eV; a negative number means that the first model is better. The interval is the 95 % bootstrap interval over structures.

| No. | First model | Second model | Difference [95 % interval] | What it says |
|---|---|---|---|---|
| 1 | atom network: co-field | ridge regression on the fields | -0.229 [-0.262, -0.195] | the gate: the atom network against the ridge regression on the same fields |
| 2 | atom network: co-field | atom network: field weights of the atom | -0.050 [-0.080, -0.020] | what the co-field adds to the atom network |
| 3 | atom network: co-field | atom network: positions | -0.021 [-0.054, +0.009] (includes zero) | the co-field against the bare place of the atom |
| 4 | atom network: co-field and positions | atom network: positions | -0.016 [-0.044, +0.013] (includes zero) | what the co-field adds to the place of the atom |
| 5 | atom network: co-field, not out-of-fold | atom network: co-field | -0.024 [-0.044, -0.002] | what the out-of-fold reading is worth (positive: it helps) |

## Checks

* The largest difference between the prediction of the ridge regression and the sum of the co-field values times the weights of the atoms, over 200 held-out hops: 3.8e-07 eV.
* The ridge chosen in the ten folds: 1.
* Root mean square error of the kernel model on its fitting part: fitted 1.081 eV, out-of-fold on the inner folds 1.270 eV (means over the ten folds).
* Atoms in the environment of a hop: 78 on average.

## Training

| Model | Epochs (smallest to largest of the ten folds) | Time for the ten folds, minutes |
|---|---|---|
| atom network: field weights of the atom | 176 to 459 | 12.0 |
| atom network: positions | 235 to 508 | 13.5 |
| atom network: co-field | 165 to 731 | 11.5 |
| atom network: co-field, not out-of-fold | 184 to 450 | 12.1 |
| atom network: co-field and positions | 165 to 286 | 9.4 |
