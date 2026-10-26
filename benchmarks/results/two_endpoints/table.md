## Two cavities in one model

Guests of two cyclodextrins, joined on the structure of the guest: a molecule measured against one cavity and not the other is still a training molecule for both endpoints. One kernel (the rotation-invariant autocorrelation), one correlation chosen by the marginal likelihood, ten-fold cross-validation holding out whole molecules. `alone` is the same code with the correlation set to zero, which is the two independent single-task fits; the interval is a bootstrap over the molecules.

| pair | guests | both | correlation | endpoint | q2 shared | q2 alone | difference |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| alpha and gamma | 938 | 181 | +0.706 | alpha | +0.329 | +0.331 | -0.002 [-0.009, +0.004] |
| | | | | gamma | +0.478 | +0.486 | -0.008 [-0.029, +0.016] |
| beta and alpha | 1493 | 496 | +0.788 | beta | +0.461 | +0.464 | -0.004 [-0.011, +0.004] |
| | | | | alpha | +0.319 | +0.314 | +0.005 [-0.008, +0.019] |
| beta and gamma | 1253 | 201 | +0.869 | beta | +0.457 | +0.457 | -0.001 [-0.006, +0.004] |
| | | | | gamma | +0.513 | +0.481 | +0.032 [-0.011, +0.082] |

| pair | selectivity q2 | guests | correlation with the guest's size, observed | predicted |
| --- | ---: | ---: | ---: | ---: |
| alpha - gamma | +0.389 | 181 | -0.613 | -0.676 |
| beta - alpha | +0.273 | 496 | +0.382 | +0.468 |
| beta - gamma | +0.223 | 201 | -0.480 | -0.739 |

