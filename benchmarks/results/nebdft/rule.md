# Which hops need the DFT calculation? The tolerance rule on nebDFT2k

1652 hops in 852 structures, 10 folds that hold out whole structures. The fine level is the DFT barrier, with a standard deviation of 0.793 eV over the hops. All errors are in eV.

## Does the standard deviation mean what it says?

The root mean square of error / standard deviation should be 1, and 95 % of the barriers should be inside two standard deviations. The factor is measured inside the training part and applied.

| Model | Mean absolute error | RMSE | Root mean square of error / sd, as computed | Factor applied | After the factor | Inside 2 sd | Rank correlation of sd and absolute error |
|---|---|---|---|---|---|---|---|
| kernel ridge on the bond-valence barrier alone | 0.301 | 0.441 | 1.01 | 1.00 | 1.01 | 94 % | +0.03 |
| kernel ridge, fields in the frame of the hop | 0.289 | 0.445 | 1.10 | 1.08 | 1.01 | 94 % | +0.16 |
| kernel ridge, fields in the frame of the hop, with the bond-valence barrier | 0.234 | 0.350 | 1.02 | 1.03 | 0.99 | 94 % | +0.11 |
| random forest, invariant field about the midpoint, with the bond-valence barrier | 0.222 | 0.348 | 1.05 | 1.06 | 0.99 | 97 % | +0.45 |

## The rule with the tolerance at the error that the model shows inside its training part

| Model | Tolerance | Hops answered | Mean absolute error of the answered hops | RMSE of the answered hops | RMSE of the hops sent to the calculation |
|---|---|---|---|---|---|
| kernel ridge on the bond-valence barrier alone | 0.440 | 94 % | 0.280 | 0.392 | 0.927 |
| kernel ridge, fields in the frame of the hop | 0.457 | 58 % | 0.248 | 0.388 | 0.514 |
| kernel ridge, fields in the frame of the hop, with the bond-valence barrier | 0.357 | 77 % | 0.212 | 0.305 | 0.472 |
| random forest, invariant field about the midpoint, with the bond-valence barrier | 0.361 | 68 % | 0.157 | 0.236 | 0.511 |

## The mean absolute error of the answered hops, by the share sent to the calculation

The hops with the largest standard deviation go first.

| Model | None sent | 10 % sent | 20 % sent | 30 % sent | 50 % sent |
|---|---|---|---|---|---|
| kernel ridge on the bond-valence barrier alone | 0.301 | 0.280 | 0.281 | 0.278 | 0.289 |
| kernel ridge, fields in the frame of the hop | 0.289 | 0.280 | 0.277 | 0.258 | 0.243 |
| kernel ridge, fields in the frame of the hop, with the bond-valence barrier | 0.234 | 0.219 | 0.213 | 0.206 | 0.205 |
| random forest, invariant field about the midpoint, with the bond-valence barrier | 0.222 | 0.185 | 0.168 | 0.160 | 0.138 |

## What orders the hops best

The mean absolute error of the answered hops when 20 % and 50 % of the hops are sent to the calculation in the order of a signal, and the share that has to be sent so that the error of the answered hops is half of the error of all hops.

kernel ridge, fields in the frame of the hop, with the bond-valence barrier (mean absolute error of all hops 0.234):

| Signal that orders the hops | 20 % sent | 50 % sent | Share to halve the error |
|---|---|---|---|
| own standard deviation | 0.213 | 0.205 | not reached |
| predicted barrier | 0.192 | 0.159 | 99 % |
| disagreement with the model on the bond-valence barrier | 0.207 | 0.201 | not reached |
| random order | 0.237 | 0.237 | not reached |
| true error (the best possible) | 0.136 | 0.074 | 28 % |

random forest, invariant field about the midpoint, with the bond-valence barrier (mean absolute error of all hops 0.222):

| Signal that orders the hops | 20 % sent | 50 % sent | Share to halve the error |
|---|---|---|---|
| own standard deviation | 0.168 | 0.138 | 76 % |
| predicted barrier | 0.182 | 0.154 | 99 % |
| disagreement with the model on the bond-valence barrier | 0.195 | 0.186 | not reached |
| random order | 0.225 | 0.219 | not reached |
| true error (the best possible) | 0.122 | 0.064 | 25 % |

