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

### Reduction: the atom that the model names

The model is 'fields + Fukui', which needs only the reactant. Out of fold, its prediction is split over the atoms (the hydrogen atoms are added to their heavy neighbour). The named atom is the one with the largest absolute share. The atom of reference is the one on which the radical ion carries most of its spin. The spin populations of all heavy atoms add up to about one.

Design: whole formulae held out, three repeats. Largest difference between the split and the prediction: 2.4e-12 V.

| Pick of the atom | Mean spin population on the picked atom | Share of species where the picked atom has the most spin |
|---|---:|---:|
| co-field: largest absolute share | 0.306 | 0.356 |
| largest Fukui function of the reactant | 0.366 | 0.417 |
| largest share in the direction that makes the step easier | 0.371 | 0.438 |
| random heavy atom (expected value) | 0.159 | 0.168 |
| the atom with the most spin (best possible) | 0.741 | 1.000 |

Differences. A positive difference means that the first pick has the larger value. The interval is 95 %, from resampling whole formulae.

| Comparison | Quantity | Difference | Interval | Interval excludes zero |
|---|---|---:|---|---|
| gate: the co-field against a random heavy atom | spin population on the picked atom | 0.147 | 0.123 to 0.171 | yes |
| the co-field against a random heavy atom | share of species with the picked atom at the most spin | 0.188 | 0.157 to 0.221 | yes |
| the co-field against the largest Fukui function | spin population on the picked atom | -0.060 | -0.088 to -0.033 | yes |
| the co-field against the largest Fukui function | share of species with the picked atom at the most spin | -0.061 | -0.097 to -0.025 | yes |
| the largest Fukui function against a random heavy atom | spin population on the picked atom | 0.207 | 0.185 to 0.229 | yes |
| the largest Fukui function against a random heavy atom | share of species with the picked atom at the most spin | 0.249 | 0.220 to 0.277 | yes |
| the atom of largest share in the easier direction against a random heavy atom | spin population on the picked atom | 0.212 | 0.190 to 0.236 | yes |
| exploratory: the atom of largest share in the easier direction against the largest Fukui function | spin population on the picked atom | 0.005 | -0.020 to 0.030 | no |
| exploratory: the atom of largest share in the easier direction against the largest Fukui function | share of species with the picked atom at the most spin | 0.021 | -0.013 to 0.053 | no |

Element of the atom with the most spin and element of the atom that the co-field names (one repeat), the eight most common pairs.

| Atom with the most spin | Atom named by the co-field | Species |
|---|---|---:|
| C | C | 1113 |
| C | O | 392 |
| P | P | 118 |
| Li | O | 86 |
| C | P | 84 |
| C | Li | 66 |
| Li | C | 61 |
| P | O | 29 |

Design: whole families held out. Largest difference between the split and the prediction: 1.4e-12 V.

| Pick of the atom | Mean spin population on the picked atom | Share of species where the picked atom has the most spin |
|---|---:|---:|
| co-field: largest absolute share | 0.333 | 0.387 |
| largest Fukui function of the reactant | 0.366 | 0.417 |
| largest share in the direction that makes the step easier | 0.372 | 0.458 |
| random heavy atom (expected value) | 0.159 | 0.168 |
| the atom with the most spin (best possible) | 0.741 | 1.000 |

Differences. A positive difference means that the first pick has the larger value. The interval is 95 %, from resampling whole formulae.

| Comparison | Quantity | Difference | Interval | Interval excludes zero |
|---|---|---:|---|---|
| gate: the co-field against a random heavy atom | spin population on the picked atom | 0.174 | 0.151 to 0.199 | yes |
| the co-field against a random heavy atom | share of species with the picked atom at the most spin | 0.218 | 0.186 to 0.255 | yes |
| the co-field against the largest Fukui function | spin population on the picked atom | -0.032 | -0.061 to -0.004 | yes |
| the co-field against the largest Fukui function | share of species with the picked atom at the most spin | -0.030 | -0.069 to 0.010 | no |
| the largest Fukui function against a random heavy atom | spin population on the picked atom | 0.207 | 0.185 to 0.229 | yes |
| the largest Fukui function against a random heavy atom | share of species with the picked atom at the most spin | 0.249 | 0.220 to 0.277 | yes |
| the atom of largest share in the easier direction against a random heavy atom | spin population on the picked atom | 0.213 | 0.190 to 0.238 | yes |
| exploratory: the atom of largest share in the easier direction against the largest Fukui function | spin population on the picked atom | 0.006 | -0.024 to 0.035 | no |
| exploratory: the atom of largest share in the easier direction against the largest Fukui function | share of species with the picked atom at the most spin | 0.041 | 0.001 to 0.080 | yes |

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

### Oxidation: the atom that the model names

The model is 'fields + Fukui', which needs only the reactant. Out of fold, its prediction is split over the atoms (the hydrogen atoms are added to their heavy neighbour). The named atom is the one with the largest absolute share. The atom of reference is the one on which the radical ion carries most of its spin. The spin populations of all heavy atoms add up to about one.

Design: whole formulae held out, three repeats. Largest difference between the split and the prediction: 2.8e-12 V.

| Pick of the atom | Mean spin population on the picked atom | Share of species where the picked atom has the most spin |
|---|---:|---:|
| co-field: largest absolute share | 0.436 | 0.513 |
| largest Fukui function of the reactant | 0.452 | 0.518 |
| largest share in the direction that makes the step easier | 0.554 | 0.683 |
| random heavy atom (expected value) | 0.157 | 0.168 |
| the atom with the most spin (best possible) | 0.703 | 1.000 |

Differences. A positive difference means that the first pick has the larger value. The interval is 95 %, from resampling whole formulae.

| Comparison | Quantity | Difference | Interval | Interval excludes zero |
|---|---|---:|---|---|
| gate: the co-field against a random heavy atom | spin population on the picked atom | 0.279 | 0.251 to 0.308 | yes |
| the co-field against a random heavy atom | share of species with the picked atom at the most spin | 0.346 | 0.311 to 0.382 | yes |
| the co-field against the largest Fukui function | spin population on the picked atom | -0.016 | -0.039 to 0.007 | no |
| the co-field against the largest Fukui function | share of species with the picked atom at the most spin | -0.004 | -0.038 to 0.029 | no |
| the largest Fukui function against a random heavy atom | spin population on the picked atom | 0.295 | 0.274 to 0.316 | yes |
| the largest Fukui function against a random heavy atom | share of species with the picked atom at the most spin | 0.350 | 0.324 to 0.378 | yes |
| the atom of largest share in the easier direction against a random heavy atom | spin population on the picked atom | 0.397 | 0.377 to 0.418 | yes |
| exploratory: the atom of largest share in the easier direction against the largest Fukui function | spin population on the picked atom | 0.102 | 0.083 to 0.120 | yes |
| exploratory: the atom of largest share in the easier direction against the largest Fukui function | share of species with the picked atom at the most spin | 0.166 | 0.138 to 0.194 | yes |

Element of the atom with the most spin and element of the atom that the co-field names (one repeat), the eight most common pairs.

| Atom with the most spin | Atom named by the co-field | Species |
|---|---|---:|
| C | C | 1645 |
| O | C | 177 |
| C | O | 157 |
| O | O | 107 |
| P | C | 70 |
| P | P | 69 |
| P | O | 40 |
| C | Li | 40 |

Design: whole families held out. Largest difference between the split and the prediction: 1.2e-12 V.

| Pick of the atom | Mean spin population on the picked atom | Share of species where the picked atom has the most spin |
|---|---:|---:|
| co-field: largest absolute share | 0.464 | 0.555 |
| largest Fukui function of the reactant | 0.452 | 0.518 |
| largest share in the direction that makes the step easier | 0.554 | 0.687 |
| random heavy atom (expected value) | 0.157 | 0.168 |
| the atom with the most spin (best possible) | 0.703 | 1.000 |

Differences. A positive difference means that the first pick has the larger value. The interval is 95 %, from resampling whole formulae.

| Comparison | Quantity | Difference | Interval | Interval excludes zero |
|---|---|---:|---|---|
| gate: the co-field against a random heavy atom | spin population on the picked atom | 0.308 | 0.281 to 0.335 | yes |
| the co-field against a random heavy atom | share of species with the picked atom at the most spin | 0.387 | 0.353 to 0.423 | yes |
| the co-field against the largest Fukui function | spin population on the picked atom | 0.013 | -0.011 to 0.036 | no |
| the co-field against the largest Fukui function | share of species with the picked atom at the most spin | 0.037 | 0.002 to 0.073 | yes |
| the largest Fukui function against a random heavy atom | spin population on the picked atom | 0.295 | 0.274 to 0.316 | yes |
| the largest Fukui function against a random heavy atom | share of species with the picked atom at the most spin | 0.350 | 0.324 to 0.378 | yes |
| the atom of largest share in the easier direction against a random heavy atom | spin population on the picked atom | 0.397 | 0.376 to 0.418 | yes |
| exploratory: the atom of largest share in the easier direction against the largest Fukui function | spin population on the picked atom | 0.103 | 0.082 to 0.121 | yes |
| exploratory: the atom of largest share in the easier direction against the largest Fukui function | share of species with the picked atom at the most spin | 0.169 | 0.138 to 0.199 | yes |

