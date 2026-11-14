# Item MS9 with a field model in every channel

The cell model is PyBaMM's SPMe on the published parameter set `Chen2020`, as in `benchmarks/results/cellmodel/table.md`. That table takes for every parameter the model with the smallest error. This one takes a kernel on continuous fields for every parameter that is predicted, and asks whether the scheme still runs from the structures to the capacity of the cell. It does not ask which model is better.

## The models and their errors on held-out groups

The error is the mean absolute error with whole groups held out: structures for the barriers, frameworks for the voltage, solvent mixtures for the conductivity. z rms is the root mean square of the standardised residual; a deviation of the right size has a z rms near 1, and the rule of plan item MS1 accepts 0.8 to 1.25.

| Parameter | Field model | Error of the field model | z rms | Model of the other table | Its error | Unit of the error |
|---|---|---:|---:|---|---:|---|
| Positive particle diffusivity [m2.s-1] | kernel ridge regression on the invariant field of the hop | 0.298 | 1.05 | random forest on the same field rows | 0.228 | eV |
| Positive electrode OCP [V] | autocorrelation kernel on the periodic fields of the cell | 0.701 | 0.91 | kernel on the composition | 0.555 | V |
| Electrolyte conductivity [S.m-1] | kernel on the fields of the sampled solvation shells, with the molality | 0.113 | 1.23 | kernel on the composition and the molality | 0.069 | log10 |
| Positive electrode Bruggeman coefficient (electrolyte) | kernel on the radial autocorrelation of the two-phase field | 0.135 | 1.04 | the same model | 0.135 | ln |

The unit `log10` is the base-ten logarithm of the conductivity in mS/cm, and `ln` is the natural logarithm of the ratio of the effective conductivity to that of the electrolyte.

## The parameters

A value replaces the published one, multiplies it or is added to it. The column of the published value is in the same terms: the published number, 1 for a factor, 0 for a shift.

| Parameter | Value | 95 % interval | What the value does to the published one | Published value in the same terms | Value in the other table | Deviation calibrated |
|---|---:|---|---|---:|---:|---|
| Positive particle diffusivity [m2.s-1] | 3.9e-14 | 3.1e-21 to 1e-10 | replaces it | 4e-15 | 4.8e-22 | yes |
| Positive electrode OCP [V] | -0.53 | -3.1 to 2.1 | is added to it | 0 | 0.28 | yes |
| Electrolyte conductivity [S.m-1] | 0.91 | 0.57 to 1.5 | multiplies it | 1 | 0.99 | yes |
| Positive electrode Bruggeman coefficient (electrolyte) | 1.9 | 1.6 to 2.2 | replaces it | 1.5 | 1.9 | yes |
| Positive electrode exchange-current density [A.m-2] | 1 | 0.1 to 9.9 | multiplies it | 1 | 1 | -- |

## What each deviation does to the capacity

Each parameter is drawn from its deviation while the others stay published. The published values give 4.991 A.h at the last of 10 cycles. The share is that of the summed variance of the capacity over the five parameters.

| Parameter | Capacity of the last cycle, A.h | Holds the published curve | Draws that the cell model refused | Share of the variance | The same interval in the other table | Holds there |
|---|---|---|---:|---:|---|---|
| Positive electrode OCP [V] | 0.049 to 5.424 | yes | 31.5 % | 59.6 % | 0.038 to 5.421 | yes |
| Positive particle diffusivity [m2.s-1] | 0.612 to 5.009 | yes | 0.0 % | 40.4 % | 0.611 to 5.009 | yes |
| Positive electrode exchange-current density [A.m-2] | 4.958 to 4.998 | yes | 0.0 % | 0.0 % | 4.958 to 4.998 | yes |
| Positive electrode Bruggeman coefficient (electrolyte) | 4.983 to 4.990 | no | 0.0 % | 0.0 % | 4.983 to 4.990 | no |
| Electrolyte conductivity [S.m-1] | 4.986 to 4.993 | yes | 0.0 % | 0.0 % | 4.989 to 4.992 | yes |

## All parameters drawn together

All five parameters are drawn at once, 200 times, and the cell model is run once per draw. 122 runs completed, and the cell model refused 39.0 % of the draws: a drawn voltage far from the published one leaves the cell no capacity inside its voltage limits. Over the runs that completed, the capacity of the last cycle is between 0.049 and 5.403 A.h (95 % interval), with a median of 2.314 A.h. The published curve is inside this interval.

## What this shows and what it does not

* The scheme is implemented with a field model in every predicted channel: fields of a hop in a crystal, fields of a periodic cell, fields of solvation shells and the field of a two-phase structure. Each gives a value with a deviation, a conversion turns it into a parameter of the cell model, and the deviation reaches the capacity.
* It does not show that the intervals are useful. An interval of the capacity is useful only if it is narrow and holds the published value, and the columns above say where it is neither. The reason is the size of two deviations. A hop barrier is known to this program to a few tenths of an electronvolt, which is several orders of magnitude in a diffusion coefficient at room temperature. A voltage is known to a large fraction of a volt, where a measured curve is known to millivolts.

