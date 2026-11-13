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

**The tortuosity takes over at faster discharge, and its error grows with it.** At 2.5C the Bruggeman exponent explains 99.7 % of the variance, its interval runs from 1.692 to 3.144 A.h, and it misses the published 3.675 A.h by 0.531 A.h. At 1C the same miss was 0.0015 A.h. The generated structures are more tortuous than the 1.5 the cell's authors fitted, and at a fast discharge that difference is worth a large part of the cell's capacity: this is the parameter that most needs measuring on the real electrode.

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

