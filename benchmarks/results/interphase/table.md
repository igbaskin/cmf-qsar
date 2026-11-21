# Part (a) of item BT13: the first electron transfer of the electrolyte species

## Reduction: 2106 species

Mean absolute error of the potential in volts, on species that the model has not seen. In the design 'whole families held out' every chemical family is left out in turn. In the design 'whole formulae held out' the isomers of one formula are left out together, five folds, three repeats.

| Model | Whole families held out, error in V | Whole formulae held out, error in V | Rank correlation, formulae |
|---|---:|---:|---:|
| mean | 1.538 | 1.509 | -0.216 |
| charge state | 0.899 | 0.874 | 0.739 |
| orbital energy | 0.547 | 0.529 | 0.914 |
| orbital energy + charge state | 0.538 | 0.512 | 0.925 |
| global quantum numbers | 0.507 | 0.471 | 0.933 |
| element counts | 0.796 | 0.705 | 0.859 |
| fields | 1.019 | 0.567 | 0.920 |
| fields + orbital energy and charge state | 0.695 | 0.472 | 0.942 |
| fields + Fukui | 0.800 | 0.504 | 0.938 |
| fields + Fukui + orbital energy and charge state | 0.653 | 0.456 | 0.947 |
| fields + spin | 0.618 | 0.478 | 0.942 |
| fields + spin + orbital energy and charge state | 0.522 | 0.422 | 0.952 |
| fields + Fukui + spin | 0.570 | 0.438 | 0.951 |
| fields + Fukui + spin + orbital energy and charge state | 0.489 | 0.390 | 0.957 |

Differences, whole families held out. A positive difference means that the tested model has the smaller error. The interval is 95 %, from resampling whole formulae.

| Comparison | Reference model | Tested model | Difference, V | Interval, V | Interval excludes zero |
|---|---|---|---:|---|---|
| gate: the fields against the orbital energy | orbital energy + charge state | fields + Fukui + spin | -0.032 | -0.058 to -0.003 | yes |
| the fields from the reactant alone against the orbital energy | orbital energy + charge state | fields + Fukui | -0.262 | -0.326 to -0.209 | yes |
| the fields against the global quantum numbers | global quantum numbers | fields + Fukui + spin | -0.063 | -0.093 to -0.032 | yes |
| the fields added to the orbital energy | orbital energy + charge state | fields + Fukui + spin + orbital energy and charge state | 0.049 | 0.030 to 0.070 | yes |
| the spin population added to the fields | fields | fields + spin | 0.401 | 0.324 to 0.491 | yes |
| the Fukui function added to the fields | fields | fields + Fukui | 0.219 | 0.173 to 0.270 | yes |
| the fields against the element counts | element counts | fields | -0.223 | -0.325 to -0.134 | yes |

Differences, whole formulae held out. A positive difference means that the tested model has the smaller error. The interval is 95 %, from resampling whole formulae.

| Comparison | Reference model | Tested model | Difference, V | Interval, V | Interval excludes zero |
|---|---|---|---:|---|---|
| gate: the fields against the orbital energy | orbital energy + charge state | fields + Fukui + spin | 0.074 | 0.044 to 0.100 | yes |
| the fields from the reactant alone against the orbital energy | orbital energy + charge state | fields + Fukui | 0.008 | -0.018 to 0.032 | no |
| the fields against the global quantum numbers | global quantum numbers | fields + Fukui + spin | 0.032 | 0.004 to 0.060 | yes |
| the fields added to the orbital energy | orbital energy + charge state | fields + Fukui + spin + orbital energy and charge state | 0.122 | 0.097 to 0.144 | yes |
| the spin population added to the fields | fields | fields + spin | 0.089 | 0.067 to 0.108 | yes |
| the Fukui function added to the fields | fields | fields + Fukui | 0.063 | 0.048 to 0.077 | yes |
| the fields against the element counts | element counts | fields | 0.137 | 0.110 to 0.165 | yes |

Error of the gate model and of its reference in every family held out, in V.

| Family held out | Orbital energy and charge state | Fields, Fukui and spin |
|---|---:|---:|
| carbonate | 0.408 | 0.445 |
| fluorinated | 0.536 | 0.533 |
| other | 0.498 | 0.566 |
| phosphorus | 0.643 | 0.634 |
| sulfur | 0.959 | 1.068 |

## Oxidation: 2486 species

Mean absolute error of the potential in volts, on species that the model has not seen. In the design 'whole families held out' every chemical family is left out in turn. In the design 'whole formulae held out' the isomers of one formula are left out together, five folds, three repeats.

| Model | Whole families held out, error in V | Whole formulae held out, error in V | Rank correlation, formulae |
|---|---:|---:|---:|
| mean | 1.468 | 1.451 | -0.111 |
| charge state | 0.935 | 0.903 | 0.688 |
| orbital energy | 0.514 | 0.491 | 0.912 |
| orbital energy + charge state | 0.524 | 0.478 | 0.920 |
| global quantum numbers | 0.501 | 0.458 | 0.930 |
| element counts | 0.902 | 0.752 | 0.829 |
| fields | 0.915 | 0.648 | 0.884 |
| fields + orbital energy and charge state | 0.440 | 0.373 | 0.957 |
| fields + Fukui | 0.792 | 0.547 | 0.921 |
| fields + Fukui + orbital energy and charge state | 0.415 | 0.332 | 0.966 |
| fields + spin | 0.623 | 0.497 | 0.935 |
| fields + spin + orbital energy and charge state | 0.425 | 0.331 | 0.965 |
| fields + Fukui + spin | 0.591 | 0.474 | 0.940 |
| fields + Fukui + spin + orbital energy and charge state | 0.403 | 0.284 | 0.976 |

Differences, whole families held out. A positive difference means that the tested model has the smaller error. The interval is 95 %, from resampling whole formulae.

| Comparison | Reference model | Tested model | Difference, V | Interval, V | Interval excludes zero |
|---|---|---|---:|---|---|
| gate: the fields against the orbital energy | orbital energy + charge state | fields + Fukui + spin | -0.066 | -0.103 to -0.035 | yes |
| the fields from the reactant alone against the orbital energy | orbital energy + charge state | fields + Fukui | -0.268 | -0.312 to -0.228 | yes |
| the fields against the global quantum numbers | global quantum numbers | fields + Fukui + spin | -0.090 | -0.125 to -0.060 | yes |
| the fields added to the orbital energy | orbital energy + charge state | fields + Fukui + spin + orbital energy and charge state | 0.122 | 0.101 to 0.142 | yes |
| the spin population added to the fields | fields | fields + spin | 0.292 | 0.254 to 0.330 | yes |
| the Fukui function added to the fields | fields | fields + Fukui | 0.123 | 0.089 to 0.157 | yes |
| the fields against the element counts | element counts | fields | -0.013 | -0.071 to 0.038 | no |

Differences, whole formulae held out. A positive difference means that the tested model has the smaller error. The interval is 95 %, from resampling whole formulae.

| Comparison | Reference model | Tested model | Difference, V | Interval, V | Interval excludes zero |
|---|---|---|---:|---|---|
| gate: the fields against the orbital energy | orbital energy + charge state | fields + Fukui + spin | 0.004 | -0.022 to 0.029 | no |
| the fields from the reactant alone against the orbital energy | orbital energy + charge state | fields + Fukui | -0.069 | -0.094 to -0.044 | yes |
| the fields against the global quantum numbers | global quantum numbers | fields + Fukui + spin | -0.016 | -0.042 to 0.009 | no |
| the fields added to the orbital energy | orbital energy + charge state | fields + Fukui + spin + orbital energy and charge state | 0.194 | 0.176 to 0.212 | yes |
| the spin population added to the fields | fields | fields + spin | 0.152 | 0.124 to 0.175 | yes |
| the Fukui function added to the fields | fields | fields + Fukui | 0.101 | 0.082 to 0.120 | yes |
| the fields against the element counts | element counts | fields | 0.104 | 0.080 to 0.127 | yes |

Error of the gate model and of its reference in every family held out, in V.

| Family held out | Orbital energy and charge state | Fields, Fukui and spin |
|---|---:|---:|
| carbonate | 0.463 | 0.457 |
| fluorinated | 0.443 | 0.583 |
| other | 0.552 | 0.532 |
| phosphorus | 0.606 | 0.692 |
| sulfur | 0.580 | 1.372 |

