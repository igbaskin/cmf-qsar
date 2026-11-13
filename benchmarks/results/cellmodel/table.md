# Item MS9: a parameter set for a cell model, with its deviations

The cell model is PyBaMM's SPMe with the growth of the passivating layer, on the published parameter set of a graphite / nickel-manganese-cobalt oxide cell (`Chen2020`). One cycle is a discharge at the one-hour rate to 2.5 V, a charge back to 4.2 V and a hold. At the published values the cell gives 4.992 A.h at the first of 10 cycles and 4.991 A.h at the last, losing 0.023 % of its capacity.

## What this program predicts, and how sure it is

| parameter | what the program gives | 95 % interval | deviation calibrated? | from |
|---|---|---|---|---|
| Positive particle diffusivity [m2.s-1] | 4.81e-22 m2.s-1 | 3.78e-30 to 8.19e-14 | yes, z rms 1.07 | diffusion coefficient |
| Positive electrode OCP [V] | 0.2768 V | -1.6354 to 2.2003 | yes, z rms 0.98 | open-circuit voltage |
| Electrolyte conductivity [S.m-1] | 0.9866 factor | 0.7627 to 1.2781 | yes, z rms 1.01 | electrolyte conductivity |
| Positive electrode Bruggeman coefficient (electrolyte) | 1.9063 exponent | 1.5933 to 2.2212 | yes, z rms 1.04 | electrode tortuosity |
| Positive electrode exchange-current density [A.m-2] | 1.0000 factor | 0.1024 to 9.8984 | not predicted | exchange current |

The kind of each number matters: a diffusion coefficient and a Bruggeman exponent replace the published value, a conductivity and an exchange current multiply it, and a voltage is added to it. A conductivity and an exchange-current density are functions of concentration and of state of charge in a cell model, so a prediction of one condition can only be a factor on the whole function.

## The capacity-against-cycle curve under each deviation

Each parameter is drawn from its own deviation while the others keep their published values, and the cell model runs once per draw (200 draws). The interval is that of the capacity of the last cycle.

| parameter | interval of the last cycle, A.h | holds the published curve? | runs the cell model refused | share of the variance | elasticity |
|---|---|---|---:|---:|---:|
| Positive electrode OCP [V] | 0.038 to 5.421 | yes | 6 % | 72.8 % | -2.426 |
| Positive particle diffusivity [m2.s-1] | 0.611 to 5.009 | yes | 0 % | 27.2 % | +0.004 |
| Positive electrode exchange-current density [A.m-2] | 4.958 to 4.998 | yes | 0 % | 0.0 % | +0.002 |
| Positive electrode Bruggeman coefficient (electrolyte) | 4.983 to 4.990 | no, 0 of 10 cycles | 0 % | 5.83e-05 % | -0.002 |
| Electrolyte conductivity [S.m-1] | 4.989 to 4.992 | yes | 0 % | 1.17e-05 % | +0.001 |

The two last columns answer two different questions. The share of the variance is what this program's own deviations do: it ranks the parameters by how much better they would have to be known. The elasticity is the relative change of the capacity for a relative change of one tenth in the parameter, around the published value, which does not depend on this program at all and is the reading the cell-model literature states.

The two orders are not the same, and neither is the one the item expected, which is the solid diffusion coefficient and the exchange current ahead of the voltage curve. The open-circuit voltage comes first in both. The reason is the same in both columns and it is a matter of scale: a tenth of a voltage is 0.38 V, which empties a cell, while a tenth of a diffusion coefficient changes almost nothing, and the deviation this program has on a voltage is about one volt, where a measured open-circuit curve is known to millivolts. The literature's ranking is for a cell whose voltage curve has been measured. Between the two transport parameters the expected order does come out: the diffusion coefficient is ahead of the exchange current in both columns.

## How accurate the barrier would have to be

The deviation of the predicted barrier is made smaller by each factor below and the cell model is run again, 60 draws over 3 cycles. An interval is worth using only if it is both narrow -- inside 0.250 A.h, which is 5 % of the published capacity -- and still holds the capacity the published value gives.

| the barrier known to, eV | interval of the capacity, A.h | narrow? | holds the published capacity? |
|---|---|---|---|
| 0.254 | 0.611 to 5.010 | no | yes |
| 0.127 | 0.611 to 1.186 | no | no |
| 0.064 | 0.611 to 0.614 | yes | no |
| 0.025 | 0.611 to 0.611 | yes | no |
| 0.013 | 0.611 to 0.611 | yes | no |
| 0.005 | 0.611 to 0.611 | yes | no |

No row is both. A narrower deviation gives a narrower interval, but the interval closes on the capacity the median prediction gives, and that prediction is wrong: the barrier is over-predicted, so the diffusion coefficient comes out several orders of magnitude too small. The error of this channel is a bias and not a spread, and a tolerance rule cannot repair a bias. What this parameter needs is not a smaller deviation but a better prediction, which for a cathode of this kind means the barrier from the fine level itself.

## The electrolyte at faster discharge and in the cold

At the one-hour rate the cell hardly responds to its electrolyte, so the two electrolyte-side parameters are drawn again where it can matter: faster discharges at room temperature, and colder cells. Each run is one discharge of the full DFN model, 100 draws per parameter, because the single-particle model approximates the electrolyte and is not meant for high rates.

The published electrolyte functions have **no temperature dependence**; of the whole parameter set only the two exchange-current densities depend on temperature. A cold run of the published set is therefore a cold cell with a warm electrolyte. The program supplies what is missing: the composition model of item BT5 predicts the activation energy of the conductivity as well, with the solvent mixture held out.

| quantity, for the cell's electrolyte | measured | predicted | over the 62 electrolytes: error, and for the mean | z rms |
|---|---:|---:|---|---:|
| conductivity at 298 K, log10 mS/cm | 0.946 | 0.971 +/- 0.058 | 0.069 against 0.146 | 1.01 |
| activation energy, kJ/mol | 13.684 | 13.562 +/- 0.914 | 1.466 against 3.535 | 1.13 |

Three capacities are compared under every condition. *Published* is the cell model's own. *Measured electrolyte* replaces its conductivity by the one CALiSol-23 measured for this electrolyte, carried to the cell's temperature by the measured activation energy: that is the curve a prediction is judged against. The intervals come from drawing the predicted parameters.

| temperature, K | rate | published, A.h | measured electrolyte, A.h | interval from the conductivity, A.h | holds the measured electrolyte? | interval from the Bruggeman exponent, A.h | holds the published curve? | share of the Bruggeman exponent |
|---:|---:|---:|---:|---|---|---|---|---:|
| 298 | 1C | 4.991 | 4.991 | 4.990 to 4.993 | yes | 4.983 to 4.990 | no | 85.3 % |
| 298 | 1.5C | 4.905 | 4.904 | 4.901 to 4.908 | yes | 4.837 to 4.900 | no | 98.6 % |
| 298 | 2C | 4.783 | 4.779 | 4.771 to 4.793 | yes | 3.053 to 4.706 | no | 100.0 % |
| 298 | 2.5C | 3.675 | 3.662 | 3.640 to 3.715 | yes | 1.692 to 3.144 | no | 99.7 % |
| 263 | 1C | 4.960 | 4.949 | 4.943 to 4.956 | yes | -- | -- | -- |
| 263 | 1.5C | 4.863 | 4.833 | 4.814 to 4.852 | yes | -- | -- | -- |
| 253 | 1C | 4.949 | 4.930 | 4.921 to 4.940 | yes | -- | -- | -- |
| 253 | 1.5C | 4.848 | 4.793 | 4.756 to 4.824 | yes | -- | -- | -- |

**The conductivity is right, now that it can be seen.** The predicted interval holds the capacity of the measured electrolyte under 8 of 8 conditions. In the cold it also excludes the published curve under 4 of 4: the published set, whose electrolyte does not get colder, gives too much capacity, and the predicted activation energy corrects it by about as much as the measured one does. The effect is modest -- a few hundredths of an ampere-hour, at most about 1 % -- because the conductivity is the only electrolyte property that is carried to the cold; the electrolyte diffusivity and the two solid diffusivities stay at their room values in this parameter set.

**The tortuosity takes over at faster discharge, and its error grows with it.** At 2.5C the Bruggeman exponent explains 99.7 % of the variance, its interval runs from 1.692 to 3.144 A.h, and it misses the published 3.675 A.h by 0.531 A.h. At 1C the same miss was 0.0015 A.h. The generated structures are more tortuous than the 1.5 the cell's authors fitted. How much that costs at a fast discharge depends on the electrolyte the cell is given, as the second reference below shows.

### The second reference: all four transport parameters measured

The published cell takes its conductivity and electrolyte diffusivity from one study at room temperature and holds the transference number (the share of the current the lithium ion carries) and the thermodynamic factor constant. Landesfeind and Gasteiger (2019) measured all four for the same electrolyte, LiPF6 in EC:EMC 3:7 by weight, between 263 and 323 K and between 0.1 and 3.0 mol/L; PyBaMM ships their fitted functions in its ORegan2022 parameter set. *Conductivity only* puts their conductivity alone into the cell, which is the fair reference for the predicted conductivity. *All four* puts in every transport parameter; in colder cells the transference number and the thermodynamic factor are read at 293 K, because the study says these two are less precise below +20 degrees Celsius and its fit gives a negative transference number from 0 degrees down. At room temperature the Bruggeman exponent is drawn again with all four in place.

| temperature, K | rate | published, A.h | measured conductivity only, A.h | interval from the predicted conductivity, A.h | holds it? | all four measured, A.h | interval from the Bruggeman exponent with all four measured, A.h | holds it? |
|---:|---:|---:|---:|---|---|---:|---|---|
| 298 | 1C | 4.991 | 4.991 | 4.990 to 4.993 | yes | 4.987 | 4.978 to 4.985 | no |
| 298 | 1.5C | 4.905 | 4.904 | 4.901 to 4.908 | yes | 4.899 | 4.878 to 4.896 | no |
| 298 | 2C | 4.783 | 4.778 | 4.771 to 4.793 | yes | 4.789 | 4.704 to 4.779 | no |
| 298 | 2.5C | 3.675 | 3.699 | 3.640 to 3.715 | yes | 4.609 | 3.873 to 4.574 | no |
| 263 | 1C | 4.960 | 4.948 | 4.943 to 4.956 | yes | 4.755 | -- | -- |
| 263 | 1.5C | 4.863 | 4.804 | 4.814 to 4.852 | no | 3.512 | -- | -- |
| 253 (outside the measured range) | 1C | 4.949 | 4.921 | 4.921 to 4.940 | yes | 3.788 | -- | -- |
| 253 (outside the measured range) | 1.5C | 4.848 | 4.427 | 4.756 to 4.824 | no | 2.409 | -- | -- |

**The predicted conductivity against the measured temperature dependence.** The predicted interval holds the capacity of the measured conductivity under 6 of 8 conditions. It misses at 263 K and 1.5C by 0.010 A.h; 253 K and 1.5C by 0.329 A.h, below the coldest measured temperature. A single activation energy, which is what the prediction carries to the cold, describes the conductivity near room temperature; the measured conductivity falls faster the colder it gets.

**At fast discharge the published electrolyte decides the verdict on the tortuosity.** At 2.5C the cell with all four measured parameters gives 4.609 A.h against the published 3.675, mostly because the measured diffusivity is larger. With the measured electrolyte in place the Bruggeman interval runs from 3.873 to 4.574 A.h and misses by 0.035 A.h, where it missed the published curve by 0.531. The exponent is still too large, but most of the earlier miss came from the published electrolyte, not from the tortuosity.

**In the cold the parameters that are not predicted decide the capacity.** At 253 K and 1.5C the measured conductivity alone gives 4.427 A.h and all four measured parameters give 2.409, against the published 4.848. The difference between the two measured capacities comes from the diffusivity, the transference number and the thermodynamic factor, for which this program has no prediction.

### How accurate the three parameters without a prediction have to be

This program does not predict the salt diffusion coefficient, the cation transference number or the thermodynamic factor of the electrolyte (plan items BT9, BT10 and BT11). Before any of them is simulated, this stage asks how accurate each one has to be. The cell holds the electrolyte with all four transport parameters as Landesfeind and Gasteiger (2019) measured them. One parameter at a time is then multiplied by a factor (the diffusion coefficient and the thermodynamic factor) or shifted by a constant (the transference number) over a fixed set of points, and the full model is run once per point. Nothing is drawn: the capacity for any spread of the parameter can be read from such a curve.

The *tolerated range* is the connected range around the measured function inside which the discharge capacity stays within the stated share of the capacity with the measured function. The *tolerated deviation* is the largest standard deviation of an error whose 95 % interval stays inside that range: of the natural logarithm for a factor, of the number itself for a shift. The nearer end of the range sets it. An end written with < or > is the outermost point that was run, so the range is at least that wide, and a deviation written with >= is a lower bound for the same reason. An end written with * is the last point before one at which the model gave no usable discharge.

**The capacity may move by 1 %.**

| Temperature, K | Rate | Capacity with the measured electrolyte, A.h | Diffusion coefficient: tolerated factor | Tolerated deviation of its logarithm | Transference number: tolerated shift | Tolerated deviation | Thermodynamic factor: tolerated factor | Tolerated deviation of its logarithm |
|---:|---:|---:|---|---:|---|---:|---|---:|
| 298 | 1C | 4.987 | 0.40 to > 20.00 | 0.470 | < -0.40 to > +0.40 | >= 0.204 | < 0.20 to 3.17 | 0.588 |
| 298 | 1.5C | 4.899 | 0.61 to > 20.00 | 0.250 | -0.28 to > +0.40 | 0.143 | < 0.20 to 2.01 | 0.356 |
| 298 | 2C | 4.789 | 0.81 to 2.53 | 0.105 | -0.11 to +0.30 | 0.057 | 0.30 to 1.40 | 0.171 |
| 298 | 2.5C | 4.609 | 0.93 to 1.14 | 0.034 | -0.04 to +0.06 | 0.021 | 0.78 to 1.16 | 0.078 |
| 263 | 1C | 4.755 | 0.96 to 1.07 | 0.023 | -0.03 to +0.03 | 0.013 | 0.89 to 1.09 | 0.046 |
| 263 | 1.5C | 3.512 | 0.99 to 1.01 | 0.005 | -0.01 to +0.01 | 0.003 | 0.92 to 1.08 | 0.037 |
| 253 | 1C | 3.788 | 0.98 to 1.02 | 0.008 | -0.01 to +0.01 | 0.005 | 0.93 to 1.06 | 0.032 |
| 253 | 1.5C | 2.409 | 0.99 to 1.01 | 0.004 | -0.01 to +0.01 | 0.003 | 0.81 to 1.14 | 0.066 |

**The capacity may move by 5 %.**

| Temperature, K | Rate | Capacity with the measured electrolyte, A.h | Diffusion coefficient: tolerated factor | Tolerated deviation of its logarithm | Transference number: tolerated shift | Tolerated deviation | Thermodynamic factor: tolerated factor | Tolerated deviation of its logarithm |
|---:|---:|---:|---|---:|---|---:|---|---:|
| 298 | 1C | 4.987 | 0.32 to > 20.00 | 0.585 | < -0.40 to > +0.40 | >= 0.204 | < 0.20 to > 5.00 | >= 0.821 |
| 298 | 1.5C | 4.899 | 0.49 to > 20.00 | 0.367 | < -0.40 to > +0.40 | >= 0.204 | < 0.20 to 3.34 | 0.616 |
| 298 | 2C | 4.789 | 0.67 to > 20.00 | 0.206 | -0.28 to > +0.40 | 0.143 | < 0.20 to 2.20 | 0.403 |
| 298 | 2.5C | 4.609 | 0.83 to > 20.00 | 0.098 | -0.13 to > +0.40 | 0.064 | < 0.20 to 1.58 | 0.233 |
| 263 | 1C | 4.755 | 0.85 to > 20.00 | 0.086 | -0.10 to > +0.40 | 0.053 | < 0.20 to 1.41 | 0.174 |
| 263 | 1.5C | 3.512 | 0.96 to 1.05 | 0.023 | -0.03 to +0.03 | 0.017 | 0.48 to 1.37 | 0.160 |
| 253 | 1C | 3.788 | 0.93 to 1.10 | 0.040 | -0.05 to +0.06 | 0.027 | 0.67 to 1.35 | 0.154 |
| 253 | 1.5C | 2.409 | 0.96 to 1.04 | 0.020 | -0.03 to +0.03 | 0.014 | < 0.20 to 1.55 | 0.224 |

For scale: at 1 mol/L and 298 K the measured diffusion coefficient is 2.89e-10 m2/s, the measured transference number 0.221 and the measured thermodynamic factor 2.18.

**Where each parameter has to be most accurate, at 5 % of the capacity:** the salt diffusion coefficient at 253 K and 1.5C, where the tolerated deviation is 0.020 (a factor between 0.96 and 1.04); the cation transference number at 253 K and 1.5C, where the tolerated deviation is 0.014 (an error within 0.026 either way); the thermodynamic factor at 253 K and 1C, where the tolerated deviation is 0.154 (a factor between 0.74 and 1.35).

**At room temperature only a diffusion coefficient that is too small is punished.** A larger one costs less than 5 % of the capacity at every factor that was tried, up to 20. The lower end of the tolerated factor rises with the rate: 0.32 at 1C, 0.49 at 1.5C, 0.67 at 2C, 0.83 at 2.5C. A prediction of this parameter should therefore err on the high side.

### What the cheapest reference of each item costs

Each of the three items names a value that needs no new data. For the diffusion coefficient it is the Nernst-Einstein value, D = 2 t (1 - t) k R T / (F^2 c), from the measured conductivity k and the transference number t of the published set: the coefficient of an ideal solution of fully dissociated ions. For the other two it is the constant of the published set. Each is put in place of the measured function, with the other three as measured, and the difference of the capacity is given in A.h and as a share of the capacity with the measured electrolyte. Two more values of the diffusion coefficient are run to say where the error of the first comes from.

| Temperature, K | Rate | Diffusion coefficient: Nernst-Einstein, ideal solution | Diffusion coefficient: Nernst-Einstein times the measured thermodynamic factor | Diffusion coefficient: function of the published set | Transference number: constant of the published set | Thermodynamic factor: 1 |
|---:|---:|---:|---:|---:|---:|---:|
| 298 | 1C | no usable discharge | -0.008 (-0.2 %) | -0.018 (-0.4 %) | +0.002 (+0.0 %) | +0.009 (+0.2 %) |
| 298 | 1.5C | -1.765 (-36.0 %) | -0.031 (-0.6 %) | -0.055 (-1.1 %) | +0.005 (+0.1 %) | +0.018 (+0.4 %) |
| 298 | 2C | -2.723 (-56.9 %) | -0.757 (-15.8 %) | -0.149 (-3.1 %) | +0.013 (+0.3 %) | +0.040 (+0.8 %) |
| 298 | 2.5C | -3.285 (-71.3 %) | -1.863 (-40.4 %) | -0.905 (-19.6 %) | +0.038 (+0.8 %) | +0.103 (+2.2 %) |
| 263 | 1C | -3.276 (-68.9 %) | -2.322 (-48.8 %) | +0.149 (+3.1 %) | +0.114 (+2.4 %) | +0.159 (+3.3 %) |
| 263 | 1.5C | -2.944 (-83.8 %) | -2.457 (-70.0 %) | +0.965 (+27.5 %) | +0.333 (+9.5 %) | +0.158 (+4.5 %) |
| 253 | 1C | -3.096 (-81.7 %) | -2.662 (-70.3 %) | +1.021 (+26.9 %) | +0.301 (+8.0 %) | +0.390 (+10.3 %) |
| 253 | 1.5C | -2.193 (-91.0 %) | -1.991 (-82.6 %) | +1.488 (+61.8 %) | +0.219 (+9.1 %) | +0.007 (+0.3 %) |

At 1 mol/L and 298 K the Nernst-Einstein value is 9.34e-11 m2/s, 3.1 times below the measured 2.89e-10. Multiplied by the measured thermodynamic factor it is 2.04e-10, 1.4 times below. The function of the published set gives 1.77e-10 at every temperature.

**BT9, the salt diffusion coefficient.** The reference is inside 1 % of the capacity under 0 of 8 conditions and inside 5 % of the capacity under 0 of 8 conditions. Its largest difference is -2.193 A.h (-91.0 %) at 253 K and 1.5C. At 298 K and 1C the cell model gave no usable discharge with this reference: the discharge did not end at the voltage limit, so there is no capacity to compare, and the condition is counted as outside. The reference is outside the tolerance under every condition, so the item does not end here.

**BT10, the cation transference number.** The reference is inside 1 % of the capacity under 4 of 8 conditions and inside 5 % of the capacity under 5 of 8 conditions. Its largest difference is +0.333 A.h (+9.5 %) at 263 K and 1.5C. The reference is not inside the tolerance under every condition, so the item does not end here; the conditions where it is inside are the ones where this parameter does not need a prediction.

**BT11, the thermodynamic factor.** The reference is inside 1 % of the capacity under 4 of 8 conditions and inside 5 % of the capacity under 7 of 8 conditions. Its largest difference is +0.390 A.h (+10.3 %) at 253 K and 1C. The reference is not inside the tolerance under every condition, so the item does not end here; the conditions where it is inside are the ones where this parameter does not need a prediction.

**The reference itself leaves the measured range under 5 of 8 conditions.** The study measured between 100 and 3000 mol/m3. With the measured electrolyte in place, the concentration somewhere in the cell goes outside that range at 298 K and 2.5C, 263 K and 1C, 263 K and 1.5C, 253 K and 1C, 253 K and 1.5C; the highest is 4673 mol/m3 at 263 K and 1C. Under 3 of these the electrolyte is exhausted somewhere in the cell (below 1 mol/m3), which is what ends the discharge early. There the fitted functions are read outside the measurements they were fitted to, so the capacity called measured is the capacity of an extrapolation.

In the colder cells the transference number and the thermodynamic factor called measured are the values at 293 K, as in the second reference, because the study calls its colder values of these two less precise. A tolerance of these two in the cold is therefore a tolerance around an assumed value.

**The reading of the curve, checked by drawing.** A tolerance is read from a curve of a few points with straight lines between them. At 298 K and 2.5C each parameter was drawn 100 times with the deviation tolerated at 5 %, and the cell was run once per draw. Each run is compared with the capacity the curve gives at the same drawn value.

| Parameter | Deviation drawn with | Largest difference between curve and run, A.h | 95 % interval of the runs, A.h | 95 % interval of the curve at the same draws, A.h | Draws whose run left the limit |
|---|---:|---:|---|---|---:|
| Salt diffusion coefficient | 0.098 | 0.019 | 4.502 to 4.668 | 4.484 to 4.666 | 2 % |
| Cation transference number | 0.064 | 0.004 | 4.478 to 4.685 | 4.475 to 4.684 | 2 % |
| Thermodynamic factor | 0.233 | 0.011 | 4.414 to 4.666 | 4.410 to 4.666 | 0 % |

The capacity with the measured electrolyte is 4.609 A.h, so the limit of 5 % is 0.230 A.h. The largest error of reading between points is 0.019 A.h, which is 8 % of that limit. About 2.5 % of the draws are expected to leave the limit, because the deviation is set by the nearer end of the range.

### How accurate the exchange-current densities and their activation energies have to be

This program does not predict the exchange-current density of either electrode (plan item BT12). The exchange-current density is the rate of the charge-transfer reaction at the surface of an electrode particle at equilibrium, per unit of area. This stage asks how accurate it has to be, in the same way as the stage above. The cell holds the measured electrolyte. The exchange-current density of one electrode at a time is multiplied by a factor over 17 points from 0.001 to 100, and the full model is run once per point. The tolerated range and the tolerated deviation are defined as above.

**The capacity may move by 1 %.**

| Temperature, K | Rate | Capacity with the published exchange-current densities, A.h | Negative electrode: tolerated factor | Tolerated deviation of its logarithm | Positive electrode: tolerated factor | Tolerated deviation of its logarithm |
|---:|---:|---:|---|---:|---|---:|
| 298 | 1C | 4.987 | 0.072 to > 100 | 1.343 | 0.044 to > 100 | 1.593 |
| 298 | 1.5C | 4.899 | 0.13 to > 100 | 1.029 | 0.11 to > 100 | 1.134 |
| 298 | 2C | 4.789 | 0.26 to 7.6 | 0.695 | 0.24 to > 100 | 0.725 |
| 298 | 2.5C | 4.609 | 0.39 to 3.3 | 0.476 | 0.39 to 3.2 | 0.476 |
| 263 | 1C | 4.755 | 0.46 to 2.4 | 0.397 | 0.45 to 2.9 | 0.409 |
| 263 | 1.5C | 3.512 | 0.48 to 2.9 | 0.375 | 0.49 to 2.3 | 0.364 |
| 253 | 1C | 3.788 | 0.51 to 2.1 | 0.346 | 0.51 to 2 | 0.341 |
| 253 | 1.5C | 2.409 | 0.37 to 6 | 0.503 | 0.38 to 5.6 | 0.500 |

**The capacity may move by 5 %.**

| Temperature, K | Rate | Capacity with the published exchange-current densities, A.h | Negative electrode: tolerated factor | Tolerated deviation of its logarithm | Positive electrode: tolerated factor | Tolerated deviation of its logarithm |
|---:|---:|---:|---|---:|---|---:|
| 298 | 1C | 4.987 | 0.01* to > 100 | 2.350 | < 0.001 to > 100 | >= 2.350 |
| 298 | 1.5C | 4.899 | 0.01* to > 100 | 2.350 | 0.0013 to > 100 | >= 2.350 |
| 298 | 2C | 4.789 | 0.02* to > 100 | 1.996 | 0.0067 to > 100 | >= 2.350 |
| 298 | 2.5C | 4.609 | 0.035 to > 100 | 1.712 | 0.035 to > 100 | 1.713 |
| 263 | 1C | 4.755 | 0.05* to > 100 | 1.528 | 0.031 to > 100 | 1.769 |
| 263 | 1.5C | 3.512 | 0.05* to > 100 | 1.528 | 0.048 to > 100 | 1.552 |
| 253 | 1C | 3.788 | 0.1* to > 100 | 1.175 | 0.036 to > 100 | 1.695 |
| 253 | 1.5C | 2.409 | 0.1* to > 100 | 1.175 | 0.032 to > 100 | 1.764 |

For scale: the parameter set of this benchmark states a spread of a factor of 10 either way for an exchange-current density. With a factor of 0.1 on one electrode the capacity changes by -3.6 to -0.7 % over the conditions and the two electrodes, and with a factor of 10 by +0.1 to +2.8 %.

An end written with * is not a point where the capacity left the limit. At the next factor the cell model did not start the discharge: its solver failed at the first step. The range is therefore at least that wide, and the deviation beside it is at least that large.

In the published set each exchange-current density follows an Arrhenius law. The activation energy is 35.0 kJ/mol for the negative electrode and 17.8 kJ/mol for the positive one, and the law equals 1 at 298.15 K. The cell model is run at one temperature. A shift of the activation energy is then the same as a factor on the exchange-current density at that temperature, so a tolerated factor is also a tolerated shift of the activation energy. A positive shift is a higher activation energy, which is a slower reaction in a cold cell. At 298 K the activation energy changes nothing, so the tables below hold the cold conditions only. The column *published law* is the factor that the published activation energy puts on the exchange-current density at that temperature.

**The capacity may move by 1 %: the tolerated shift of the activation energy.**

| Temperature, K | Rate | Negative electrode: published law | Tolerated shift, kJ/mol | Tolerated deviation, kJ/mol | Positive electrode: published law | Tolerated shift, kJ/mol | Tolerated deviation, kJ/mol |
|---:|---:|---:|---|---:|---:|---|---:|
| 263 | 1C | 0.153 | -16.7 to +14.5 | 7.4 | 0.385 | -19.6 to +14.9 | 7.6 |
| 263 | 1.5C | 0.153 | -20.1 to +13.7 | 7.0 | 0.385 | -15.6 to +13.3 | 6.8 |
| 253 | 1C | 0.081 | -10.5 to +9.4 | 4.8 | 0.279 | -9.7 to +9.3 | 4.8 |
| 253 | 1.5C | 0.081 | -25.0 to +13.7 | 7.0 | 0.279 | -24.0 to +13.7 | 7.0 |

**The capacity may move by 5 %: the tolerated shift of the activation energy.**

| Temperature, K | Rate | Negative electrode: published law | Tolerated shift, kJ/mol | Tolerated deviation, kJ/mol | Positive electrode: published law | Tolerated shift, kJ/mol | Tolerated deviation, kJ/mol |
|---:|---:|---:|---|---:|---:|---|---:|
| 263 | 1C | 0.153 | < -85.8 to +55.8* | 28.5 | 0.385 | < -85.8 to +64.6 | 33.0 |
| 263 | 1.5C | 0.153 | < -85.8 to +55.8* | 28.5 | 0.385 | < -85.8 to +56.7 | 28.9 |
| 253 | 1C | 0.081 | < -64.2 to +32.1* | 16.4 | 0.279 | < -64.2 to +46.3 | 23.6 |
| 253 | 1.5C | 0.081 | < -64.2 to +32.1* | 16.4 | 0.279 | < -64.2 to +48.2 | 24.6 |

**A stated activation energy in place of the published one.** The numbers are the difference of the capacity, in per cent of the capacity with the published activation energies. *Both* puts the same activation energy on both electrodes.

| Temperature, K | Rate | Electrode | 20 kJ/mol | 30 kJ/mol | 40 kJ/mol | 50 kJ/mol | 60 kJ/mol | 70 kJ/mol |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 263 | 1C | negative | +0.9 % | +0.3 % | -0.3 % | -1.0 % | -1.8 % | -2.5 % |
| 263 | 1C | positive | -0.1 % | -0.8 % | -1.5 % | -2.2 % | -3.0 % | -3.9 % |
| 263 | 1C | both | +0.8 % | -0.5 % | -1.9 % | -3.4 % | -5.2 % | -7.1 % |
| 263 | 1.5C | negative | +0.8 % | +0.3 % | -0.3 % | -1.1 % | -1.9 % | -2.8 % |
| 263 | 1.5C | positive | -0.2 % | -0.9 % | -1.7 % | -2.6 % | -3.5 % | -4.5 % |
| 263 | 1.5C | both | +0.7 % | -0.5 % | -2.1 % | -4.0 % | -6.1 % | -8.4 % |
| 253 | 1C | negative | +1.4 % | +0.5 % | -0.5 % | -1.6 % | -2.6 % | -3.7 % |
| 253 | 1C | positive | -0.2 % | -1.3 % | -2.4 % | -3.4 % | -4.5 % | -5.7 % |
| 253 | 1C | both | +1.1 % | -0.8 % | -2.9 % | -5.1 % | -7.7 % | -10.4 % |
| 253 | 1.5C | negative | +0.7 % | +0.3 % | -0.3 % | -1.1 % | -2.1 % | no usable discharge |
| 253 | 1.5C | positive | -0.1 % | -0.9 % | -1.8 % | -2.9 % | -4.2 % | -5.5 % |
| 253 | 1.5C | both | +0.6 % | -0.5 % | -2.3 % | -4.8 % | -7.8 % | no usable discharge |

## What the gate asked, and what came out

**The first part is not met.** The interval holds the published curve for 4 of 5 parameters, and the gate asks for every parameter whose deviation means what it says. It fails for Positive electrode Bruggeman coefficient (electrolyte), whose deviation is calibrated (z rms 1.04). The interval misses by 6.43e-04 A.h, because the structures this channel was fitted on are more tortuous than the exponent of 1.5 the authors of the cell fitted; the channel says so itself, since its structures are generated and it is marked as outside what the model was fitted on.

**The second part is not met either.** The ranking does not put the solid diffusion coefficient and the exchange current ahead of the voltage curve, under either reading, for the reason given above: this program's deviation on a voltage is about a volt.

**What carries over is the interface.** The writer, the sampler and the report the item describes are built: the deviation is carried through the conversions into the units a cell model reads, and the report says which parameter's deviation moves the capacity most. The tolerance rule says the useful thing: for the diffusion coefficient no deviation is small enough, because the error is a bias; and the electrolyte conductivity, the one channel predicted for the cell's own material under its own conditions, agrees with the published parameter set to 1.3 %.

## What every channel measured

**Positive particle diffusivity [m2.s-1]** (diffusion coefficient, 385.2 s). The forest on the invariant field, fitted on the 1652 density-functional barriers of nebDFT2k (plan item BT7), read as the percolation barrier of the hop network of LiNiO2 (mp-2348641) in 2 directions.
The forest predicts the 1652 training barriers to 0.228 eV with whole structures held out, its deviation has z rms 1.07 and holds 96 % of them inside two standard deviations. LiNiO2 (mp-2348641) has 24 lithium hops, 24 of them distinct, and its network percolates in 2 of three directions, which is the two-dimensional diffusion a layered oxide is known for.
*The attempt frequency is 1e+13 per second and the hop length 2.89 angstrom, neither of them predicted; the cathode of the cell is a nickel-manganese-cobalt oxide and the structure used is the ordered layered nickel oxide, which is the closest the public set holds.*

**Positive electrode OCP [V]** (open-circuit voltage, 2.8 s). The kernel model on the composition of 303 insertion pairs reconstructed from public formation energies (plan item BT3), with the frameworks held out.
Mean absolute error 0.555 V against 1.203 for predicting the mean; the deviation has z rms 0.98 and holds 95 % of the activities inside two standard deviations, where 95 % is expected.
*The shift is the model's own error on the layered nickel oxide, which is the closest material to the cell's cathode that the set holds; the ladder reconstructs 3.47 V and the model on the composition 4.08 V, against 3.80 V published.*

**Electrolyte conductivity [S.m-1]** (electrolyte conductivity, 0.9 s). The kernel model on the composition of 62 measured conductivities of CALiSol-23 (plan item BT5), with the solvent mixture of this electrolyte held out.
Mean absolute error 0.069 on the base-ten logarithmic scale against 0.146 for predicting the mean; the deviation has z rms 1.01 and holds 95 % of the activities inside two standard deviations, where 95 % is expected.
*The cell model holds the conductivity as a function of concentration and temperature, so a prediction of one condition can only be a factor on it; the condition is EC0.30+EMC0.70 at 1.11 mol/kg against the one mole per litre the cell model starts from.*

**Positive electrode Bruggeman coefficient (electrolyte)** (electrode tortuosity, 45.4 s). The kernel on the radial autocorrelation of a two-phase field, fitted on 200 generated structures with their effective conductivity solved on the grid (the microstructure pilot).
Mean absolute error 0.135 on the natural logarithm of the conductivity ratio against 0.296 for predicting the mean; the deviation has z rms 1.04 and holds 96 % of the activities inside two standard deviations, where 95 % is expected.
*The structures are generated and not measured, so what this channel gives is the deviation such a model has, not the exponent of this cell's electrode; the training structures cover the porosity of the cell, as a model fitted for one cell would, and the structures read off are held out; the solid phase carries 0.001 of the conductivity of the electrolyte in the pores.*

**Positive electrode exchange-current density [A.m-2]** (exchange current, 0.0 s). Not predicted by this program: a stated spread.
*A 95 % interval of a factor of 10, which is the order the published values for one electrode material differ by; it is here because the ranking of the parameters needs every parameter in it.*

