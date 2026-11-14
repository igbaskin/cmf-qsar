# Item BT9: the salt diffusion coefficient of an electrolyte from simulation

The electrolyte is that of the reference cell: lithium hexafluorophosphate in ethylene carbonate and ethyl methyl carbonate, 3 to 7 by weight, at 1.11 mol of salt per kilogram of solvent and 298 K. Every run is one box of the GAFF force field with charges from the OpenFF graph network, equilibrated at constant pressure and then run at constant volume with a Nose-Hoover thermostat. The unwrapped centre of every molecule is stored once per picosecond, and every number below is computed from those centres (`cmfqsar/transport.py`).

A self-diffusion coefficient is the slope of the mean squared displacement of one species against time, divided by six, fitted between a fortieth and a tenth of the run. The *exponent* is the power with which the mean squared displacement grows in that window: 1 is diffusion, and a value below 1 says that the molecules had not left their cages yet and the coefficient is an overestimate of the long-time value. The error after +/- is the standard error over five blocks of the run. It says how well the run determined the number and says nothing about the force field.

## The boxes

| Run | What it changes | Atoms | Ion pairs | Side of the box, A | Density, g/cm3 | Concentration, mol/L | Mean kinetic temperature, K | Length, ns | Speed, ns/day |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | whole charges on the ions | 3546 | 27 | 34.17 | 1.198 | 1.123 | 298.1 | 4.0 | 139 |
| scaled | charges of the ions multiplied by 0.8 | 3546 | 27 | 34.29 | 1.186 | 1.112 | 298.7 | 4.0 | 137 |
| scaled_small | scaled charges, half as many molecules | 1777 | 14 | 27.20 | 1.194 | 1.156 | 298.6 | 4.0 | 189 |
| scaled_large | scaled charges, twice as many molecules | 7100 | 55 | 43.20 | 1.189 | 1.133 | 298.2 | 4.0 | 148 |
| scaled_07 | charges of the ions multiplied by 0.7 | 3546 | 27 | 34.32 | 1.183 | 1.110 | 298.1 | 4.0 | 142 |
| scaled_seed2 | scaled charges, a second start | 3546 | 27 | 34.28 | 1.186 | 1.113 | 298.4 | 4.0 | 121 |
| scaled_seed3 | scaled charges, a third start | 3546 | 27 | 34.24 | 1.190 | 1.117 | 298.3 | 4.0 | 99 |
| scaled_huge | scaled charges, four times as many molecules | 14200 | 110 | 54.40 | 1.191 | 1.135 | 298.2 | 4.0 | 116 |

## The simulated electrolyte against the measured one

The measured values are those of Landesfeind and Gasteiger (2019) for this electrolyte, read from the functions PyBaMM ships at the concentration of the box: salt diffusion coefficient 2.68e-10 m2/s, conductivity 0.906 S/m, transference number 0.202, thermodynamic factor 2.37. CALiSol-23 gives a conductivity of 0.883 S/m for the same formulation.

**Self-diffusion, m2/s.**

| Run | Cation | Exponent | Anion | Exponent | EC | Exponent | EMC | Exponent |
|---|---|---:|---|---:|---|---:|---|---:|
| full | 3.46e-11 +/- 2.2e-12 | 0.73 | 4.60e-11 +/- 4.6e-12 | 0.70 | 1.11e-10 +/- 3.0e-12 | 0.84 | 1.15e-10 +/- 2.5e-12 | 0.84 |
| scaled | 7.09e-11 +/- 4.8e-12 | 0.82 | 1.01e-10 +/- 6.7e-12 | 0.85 | 1.96e-10 +/- 9.0e-12 | 0.93 | 1.71e-10 +/- 2.7e-12 | 0.89 |
| scaled_small | 4.55e-11 +/- 1.3e-12 | 0.74 | 8.36e-11 +/- 7.9e-12 | 0.86 | 1.30e-10 +/- 8.4e-12 | 0.86 | 1.37e-10 +/- 4.8e-12 | 0.88 |
| scaled_large | 6.80e-11 +/- 1.3e-12 | 0.78 | 9.21e-11 +/- 5.5e-12 | 0.81 | 1.78e-10 +/- 6.0e-12 | 0.89 | 1.66e-10 +/- 5.2e-12 | 0.88 |
| scaled_07 | 9.67e-11 +/- 6.4e-12 | 0.84 | 1.35e-10 +/- 6.2e-12 | 0.83 | 1.96e-10 +/- 7.5e-12 | 0.90 | 1.87e-10 +/- 7.5e-12 | 0.90 |
| scaled_seed2 | 6.47e-11 +/- 2.6e-12 | 0.80 | 9.06e-11 +/- 5.6e-12 | 0.84 | 1.83e-10 +/- 6.2e-12 | 0.91 | 1.69e-10 +/- 3.8e-12 | 0.89 |
| scaled_seed3 | 6.02e-11 +/- 4.7e-12 | 0.80 | 7.66e-11 +/- 5.5e-12 | 0.78 | 1.78e-10 +/- 1.0e-11 | 0.92 | 1.58e-10 +/- 8.3e-12 | 0.87 |
| scaled_huge | 7.15e-11 +/- 1.6e-12 | 0.81 | 8.96e-11 +/- 1.9e-12 | 0.81 | 1.68e-10 +/- 5.8e-12 | 0.88 | 1.74e-10 +/- 2.4e-12 | 0.89 |

**The salt diffusion coefficient, m2/s.** For an ideal solution it is 2 D+ D- / (D+ + D-). A concentrated solution multiplies that by the thermodynamic factor, and the measured factor is used here, because this program has no prediction of it (plan item BT11). The ratio is to the measured coefficient.

| Run | Ideal solution | Ratio to the measured | With the measured thermodynamic factor | Ratio to the measured | Rates at which the cell tolerates it |
|---|---:|---:|---:|---:|---|
| full | 3.95e-11 | 0.15 | 9.35e-11 | 0.35 | 1C |
| scaled | 8.32e-11 | 0.31 | 1.96e-10 | 0.73 | 1C, 1.5C, 2C |
| scaled_small | 5.89e-11 | 0.22 | 1.42e-10 | 0.54 | 1C, 1.5C |
| scaled_large | 7.83e-11 | 0.29 | 1.86e-10 | 0.70 | 1C, 1.5C, 2C |
| scaled_07 | 1.13e-10 | 0.42 | 2.65e-10 | 0.98 | 1C, 1.5C, 2C, 2.5C |
| scaled_seed2 | 7.55e-11 | 0.28 | 1.78e-10 | 0.66 | 1C, 1.5C |
| scaled_seed3 | 6.74e-11 | 0.25 | 1.59e-10 | 0.59 | 1C, 1.5C |
| scaled_huge | 7.95e-11 | 0.30 | 1.90e-10 | 0.71 | 1C, 1.5C, 2C |

The last column uses the tolerance stage of `run_cellmodel.py`: for a capacity that may move by 5 % at 298 K the coefficient may be no lower than 0.32 of the measured value at 1C, 0.49 of the measured value at 1.5C, 0.67 of the measured value at 2C, 0.83 of the measured value at 2.5C, and every larger value that was tried is tolerated.

None of these coefficients is corrected for the size of the box. A periodic box makes a diffusion coefficient too small, so the correction raises every one of them.

**The conductivity, S/m.** The first value is from the displacement of the total charge and includes the motion of ions together. The Nernst-Einstein value is what the two self-diffusion coefficients would give if every ion moved on its own. Their ratio says how far the ions are from moving on their own. Both use the whole charges of the ions, also in the run that scales them in the force field.

| Run | From the total charge | Nernst-Einstein value | Ratio of the two | Ratio of the first to the measured | Share of the current on the cation if ions moved alone | Measured transference number |
|---|---|---:|---:|---:|---:|---:|
| full | 0.093 +/- 0.028 | 0.340 | 0.27 | 0.10 | 0.429 | 0.202 |
| scaled | 0.342 +/- 0.045 | 0.717 | 0.48 | 0.38 | 0.413 | 0.202 |
| scaled_small | 0.133 +/- 0.047 | 0.560 | 0.24 | 0.15 | 0.352 | 0.202 |
| scaled_large | 0.302 +/- 0.052 | 0.681 | 0.44 | 0.33 | 0.425 | 0.202 |
| scaled_07 | 0.610 +/- 0.149 | 0.967 | 0.63 | 0.67 | 0.417 | 0.202 |
| scaled_seed2 | 0.251 +/- 0.085 | 0.649 | 0.39 | 0.28 | 0.417 | 0.202 |
| scaled_seed3 | 0.166 +/- 0.029 | 0.574 | 0.29 | 0.18 | 0.440 | 0.202 |
| scaled_huge | 0.351 +/- 0.157 | 0.686 | 0.51 | 0.39 | 0.444 | 0.202 |

## How the error falls with the length of the run

One run is read at several lengths, each from its start. The window of the fit moves with the length, so a shorter reading also sits earlier in the caged regime, which the exponent shows.

| Run | Length, ns | Cation, m2/s | Exponent | Anion, m2/s | Exponent | Salt, ideal, m2/s | Conductivity, S/m |
|---|---:|---|---:|---|---:|---:|---|
| full | 0.5 | 4.66e-11 +/- 5.6e-12 | 0.64 | 5.96e-11 +/- 3.8e-12 | 0.62 | 5.23e-11 | 0.064 +/- 0.021 |
| full | 1.0 | 4.30e-11 +/- 3.0e-12 | 0.62 | 5.91e-11 +/- 3.1e-12 | 0.62 | 4.98e-11 | 0.095 +/- 0.031 |
| full | 2.0 | 4.10e-11 +/- 4.6e-12 | 0.66 | 5.76e-11 +/- 5.5e-12 | 0.68 | 4.79e-11 | 0.102 +/- 0.014 |
| full | 4.0 | 3.46e-11 +/- 2.2e-12 | 0.73 | 4.60e-11 +/- 4.6e-12 | 0.70 | 3.95e-11 | 0.093 +/- 0.028 |
| scaled | 0.5 | 6.68e-11 +/- 8.9e-12 | 0.61 | 6.79e-11 +/- 8.0e-12 | 0.53 | 6.74e-11 | 0.221 +/- 0.121 |
| scaled | 1.0 | 7.52e-11 +/- 8.2e-12 | 0.68 | 9.22e-11 +/- 1.6e-11 | 0.66 | 8.29e-11 | 0.252 +/- 0.089 |
| scaled | 2.0 | 8.03e-11 +/- 3.8e-12 | 0.77 | 1.09e-10 +/- 1.1e-11 | 0.80 | 9.25e-11 | 0.317 +/- 0.079 |
| scaled | 4.0 | 7.09e-11 +/- 4.8e-12 | 0.82 | 1.01e-10 +/- 6.7e-12 | 0.85 | 8.32e-11 | 0.342 +/- 0.045 |
| scaled_small | 0.5 | 4.72e-11 +/- 1.8e-11 | 0.62 | 7.15e-11 +/- 2.0e-11 | 0.67 | 5.69e-11 | 0.135 +/- 0.049 |
| scaled_small | 1.0 | 5.59e-11 +/- 1.4e-11 | 0.69 | 9.29e-11 +/- 2.4e-11 | 0.77 | 6.98e-11 | 0.397 +/- 0.161 |
| scaled_small | 2.0 | 5.03e-11 +/- 5.1e-12 | 0.70 | 8.27e-11 +/- 1.0e-11 | 0.76 | 6.26e-11 | 0.316 +/- 0.075 |
| scaled_small | 4.0 | 4.55e-11 +/- 1.3e-12 | 0.74 | 8.36e-11 +/- 7.9e-12 | 0.86 | 5.89e-11 | 0.133 +/- 0.047 |
| scaled_large | 0.5 | 8.77e-11 +/- 3.3e-12 | 0.68 | 1.17e-10 +/- 8.9e-12 | 0.72 | 1.00e-10 | 0.356 +/- 0.145 |
| scaled_large | 1.0 | 9.05e-11 +/- 6.5e-12 | 0.75 | 1.06e-10 +/- 8.0e-12 | 0.72 | 9.78e-11 | 0.267 +/- 0.101 |
| scaled_large | 2.0 | 7.88e-11 +/- 3.6e-12 | 0.75 | 9.53e-11 +/- 2.9e-12 | 0.74 | 8.63e-11 | 0.380 +/- 0.042 |
| scaled_large | 4.0 | 6.80e-11 +/- 1.3e-12 | 0.78 | 9.21e-11 +/- 5.5e-12 | 0.81 | 7.83e-11 | 0.302 +/- 0.052 |
| scaled_07 | 0.5 | 9.93e-11 +/- 1.1e-11 | 0.71 | 1.68e-10 +/- 1.5e-11 | 0.82 | 1.25e-10 | 0.317 +/- 0.195 |
| scaled_07 | 1.0 | 9.45e-11 +/- 7.1e-12 | 0.68 | 1.47e-10 +/- 1.3e-11 | 0.78 | 1.15e-10 | 0.269 +/- 0.076 |
| scaled_07 | 2.0 | 9.83e-11 +/- 2.8e-12 | 0.77 | 1.41e-10 +/- 9.1e-12 | 0.81 | 1.16e-10 | 0.294 +/- 0.136 |
| scaled_07 | 4.0 | 9.67e-11 +/- 6.4e-12 | 0.84 | 1.35e-10 +/- 6.2e-12 | 0.83 | 1.13e-10 | 0.610 +/- 0.149 |
| scaled_seed2 | 0.5 | 8.73e-11 +/- 1.4e-11 | 0.74 | 1.09e-10 +/- 1.7e-11 | 0.74 | 9.68e-11 | 0.217 +/- 0.076 |
| scaled_seed2 | 1.0 | 7.99e-11 +/- 1.1e-11 | 0.75 | 1.05e-10 +/- 1.1e-11 | 0.76 | 9.06e-11 | 0.236 +/- 0.057 |
| scaled_seed2 | 2.0 | 7.09e-11 +/- 7.4e-12 | 0.75 | 9.61e-11 +/- 6.7e-12 | 0.78 | 8.16e-11 | 0.207 +/- 0.039 |
| scaled_seed2 | 4.0 | 6.47e-11 +/- 2.6e-12 | 0.80 | 9.06e-11 +/- 5.6e-12 | 0.84 | 7.55e-11 | 0.251 +/- 0.085 |
| scaled_seed3 | 0.5 | 7.99e-11 +/- 6.7e-12 | 0.70 | 9.43e-11 +/- 8.3e-12 | 0.68 | 8.65e-11 | 0.247 +/- 0.084 |
| scaled_seed3 | 1.0 | 7.48e-11 +/- 6.9e-12 | 0.74 | 9.57e-11 +/- 6.5e-12 | 0.74 | 8.40e-11 | 0.212 +/- 0.067 |
| scaled_seed3 | 2.0 | 6.97e-11 +/- 4.9e-12 | 0.77 | 8.83e-11 +/- 5.4e-12 | 0.76 | 7.79e-11 | 0.193 +/- 0.050 |
| scaled_seed3 | 4.0 | 6.02e-11 +/- 4.7e-12 | 0.80 | 7.66e-11 +/- 5.5e-12 | 0.78 | 6.74e-11 | 0.166 +/- 0.029 |
| scaled_huge | 0.5 | 8.02e-11 +/- 5.4e-12 | 0.65 | 9.69e-11 +/- 5.2e-12 | 0.65 | 8.77e-11 | 0.228 +/- 0.058 |
| scaled_huge | 1.0 | 8.06e-11 +/- 3.2e-12 | 0.69 | 1.02e-10 +/- 3.9e-12 | 0.71 | 9.00e-11 | 0.279 +/- 0.135 |
| scaled_huge | 2.0 | 7.55e-11 +/- 3.9e-13 | 0.74 | 9.82e-11 +/- 1.8e-12 | 0.76 | 8.54e-11 | 0.432 +/- 0.065 |
| scaled_huge | 4.0 | 7.15e-11 +/- 1.6e-12 | 0.81 | 8.96e-11 +/- 1.9e-12 | 0.81 | 7.95e-11 | 0.351 +/- 0.157 |

## Repeats of one box

The same box was packed and started 3 times (scaled, scaled_seed2, scaled_seed3). The error over the blocks of one run says how well that run determined its own coefficient. It cannot see a difference between two runs that start from different arrangements of the ions, if an arrangement lasts longer than the run. The standard deviation between the repeats does see it, and the ratio of the two is the test.

| Species | scaled, m2/s | scaled_seed2, m2/s | scaled_seed3, m2/s | Mean, m2/s | Standard deviation between repeats | Mean block error of one run | Ratio of the two |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cation | 7.09e-11 | 6.47e-11 | 6.02e-11 | 6.53e-11 | 5.4e-12 | 4.0e-12 | 1.3 |
| Anion | 1.01e-10 | 9.06e-11 | 7.66e-11 | 8.93e-11 | 1.2e-11 | 5.9e-12 | 2.0 |
| EC | 1.96e-10 | 1.83e-10 | 1.78e-10 | 1.86e-10 | 9.3e-12 | 8.5e-12 | 1.1 |
| EMC | 1.71e-10 | 1.69e-10 | 1.58e-10 | 1.66e-10 | 6.9e-12 | 4.9e-12 | 1.4 |

The median ratio is 1.4. The blocks of one run see the whole error of its coefficient. With 3 repeats a standard deviation is itself known only to within a factor of about two, so this ratio is a size and not a number.

Over the repeats, the salt diffusion coefficient with the measured thermodynamic factor is 0.66 +/- 0.04 of the measured one (single runs: 0.73, 0.66, 0.59), and the conductivity from the total charge is 0.253 +/- 0.051 S/m against the measured 0.908 (single runs: 0.342, 0.251, 0.166; the mean block error of one run is 0.053). The error after +/- here is the standard error of the mean over the repeats.

## The charges of the ions

A force field with fixed charges has no polarisation. It binds a small cation to its neighbours too strongly, and the ions move too slowly. Multiplying the charges of the ions by a factor below one is the usual repair. The table gives the base box at each factor that was run. Where a factor was run several times, the numbers are means over the repeats with the standard error of the mean; where it was run once, the error is the block error of that run, which the repeats found too small by a factor of about 1.4. None of these numbers is corrected for the size of the box.

| Factor on the charges | Runs | Cation, m2/s | Anion, m2/s | Exponent of the cation | Salt coefficient with the measured thermodynamic factor, ratio to the measured | Rates at which the cell tolerates it | Conductivity, S/m | Ratio to the measured | Ratio to the Nernst-Einstein value |
|---:|---:|---|---|---:|---|---|---|---:|---:|
| 1 | 1 | 3.46e-11 +/- 2.21e-12 | 4.60e-11 +/- 4.59e-12 | 0.73 | 0.35 | 1C | 0.093 +/- 0.028 | 0.10 | 0.27 |
| 0.8 | 3 | 6.53e-11 +/- 3.09e-12 | 8.93e-11 +/- 6.98e-12 | 0.81 | 0.66 +/- 0.04 | 1C, 1.5C | 0.253 +/- 0.051 | 0.28 | 0.38 |
| 0.7 | 1 | 9.67e-11 +/- 6.38e-12 | 1.35e-10 +/- 6.20e-12 | 0.84 | 0.98 | 1C, 1.5C, 2C, 2.5C | 0.610 +/- 0.149 | 0.67 | 0.63 |

From a factor of 1 to 0.7 the self-diffusion coefficient of the cation rises 2.8 times and the conductivity 6.5 times. The conductivity rises faster because the ions also pair less: the last column is the share of the conductivity that is left when the motion of ions together is counted. Choosing the factor by the measured conductivity of this one electrolyte is a calibration of one parameter on one number. The test of it is whether the same factor gives the diffusion coefficient here and the conductivities of other electrolytes.

## The size of the box

A periodic box makes a diffusion coefficient too small, because a molecule interacts with its own images through the liquid. The amount is 2.837 k T / (6 pi eta L) for a cubic box of the side L and a liquid of the viscosity eta (Yeh and Hummer, 2004). It is the same amount for every species of one liquid. The runs below differ only in the number of molecules. One straight line per species is fitted against 1 / L, with one slope for all of them, so that the solvents, which have many molecules, fix the slope for the ions, which have few. The value of a line at 1 / L = 0 is the coefficient of an infinite box.

The box of the run `scaled` was repeated, so it enters with the mean of its repeats and the standard error of that mean. The errors of the other boxes are their block errors multiplied by 1.4, the factor the repeats give.

| Run | Ion pairs | Side, A | Cation, m2/s | Anion, m2/s | EC, m2/s | EMC, m2/s | What the correction adds, m2/s |
|---|---:|---:|---|---|---|---|---:|
| scaled_small | 14 | 27.20 | 4.55e-11 +/- 1.8e-12 | 8.36e-11 +/- 1.1e-11 | 1.30e-10 +/- 1.1e-11 | 1.37e-10 +/- 6.6e-12 | 5.02e-11 |
| scaled | 27 | 34.27 | 6.53e-11 +/- 3.1e-12 | 8.93e-11 +/- 7.0e-12 | 1.86e-10 +/- 5.4e-12 | 1.66e-10 +/- 4.0e-12 | 3.98e-11 |
| scaled_large | 55 | 43.20 | 6.80e-11 +/- 1.7e-12 | 9.21e-11 +/- 7.4e-12 | 1.78e-10 +/- 8.1e-12 | 1.66e-10 +/- 7.1e-12 | 3.16e-11 |
| scaled_huge | 110 | 54.40 | 7.15e-11 +/- 2.1e-12 | 8.96e-11 +/- 2.6e-12 | 1.68e-10 +/- 7.9e-12 | 1.74e-10 +/- 3.3e-12 | 2.51e-11 |
| infinite box | | | 9.82e-11 | 1.18e-10 | 2.11e-10 | 2.00e-10 | 0 |

The common slope gives a viscosity of 4.54 mPa s. The lines leave a root-mean-square residual of 1.93 in units of the errors of the coefficients; a value near 1 says that one slope describes all species. The hydrodynamic radius of a species is the radius of a sphere that would diffuse as fast in a liquid of that viscosity: cation 4.9 A, anion 4.1 A, EC 2.3 A, EMC 2.4 A.

Fitted one species at a time, without the common slope, the same boxes give: cation 1.01e-10 m2/s with a viscosity of 4.23 mPa s; anion 9.29e-11 m2/s with a viscosity of 37.78 mPa s; EC 1.96e-10 m2/s with a viscosity of 7.72 mPa s; EMC 2.05e-10 m2/s with a viscosity of 3.98 mPa s. The spread of these says how well the boxes determine the slope.

**In an infinite box** the salt diffusion coefficient of an ideal solution is 1.07e-10 m2/s, 0.40 of the measured 2.66e-10. Multiplied by the measured thermodynamic factor it is 2.55e-10 m2/s, 0.96 of the measured value, which the cell tolerates at 1C, 1.5C, 2C, 2.5C. The Nernst-Einstein conductivity of the infinite box is 0.921 S/m against the measured 0.905, and the share of the current on the cation, if the ions moved alone, is 0.455.

## The size of the box, without the smallest box

The same fit without `scaled_small`, the smallest box. If the two fits disagree, the smallest box is too small for the correction to describe it.

The box of the run `scaled` was repeated, so it enters with the mean of its repeats and the standard error of that mean. The errors of the other boxes are their block errors multiplied by 1.4, the factor the repeats give.

Left out of this fit: `scaled_small`.

| Run | Ion pairs | Side, A | Cation, m2/s | Anion, m2/s | EC, m2/s | EMC, m2/s | What the correction adds, m2/s |
|---|---:|---:|---|---|---|---|---:|
| scaled | 27 | 34.27 | 6.53e-11 +/- 3.1e-12 | 8.93e-11 +/- 7.0e-12 | 1.86e-10 +/- 5.4e-12 | 1.66e-10 +/- 4.0e-12 | 1.10e-11 |
| scaled_large | 55 | 43.20 | 6.80e-11 +/- 1.7e-12 | 9.21e-11 +/- 7.4e-12 | 1.78e-10 +/- 8.1e-12 | 1.66e-10 +/- 7.1e-12 | 8.76e-12 |
| scaled_huge | 110 | 54.40 | 7.15e-11 +/- 2.1e-12 | 8.96e-11 +/- 2.6e-12 | 1.68e-10 +/- 7.9e-12 | 1.74e-10 +/- 3.3e-12 | 6.95e-12 |
| infinite box | | | 7.73e-11 | 9.74e-11 | 1.89e-10 | 1.79e-10 | 0 |

The common slope gives a viscosity of 16.38 mPa s. The lines leave a root-mean-square residual of 1.03 in units of the errors of the coefficients; a value near 1 says that one slope describes all species. The hydrodynamic radius of a species is the radius of a sphere that would diffuse as fast in a liquid of that viscosity: cation 1.7 A, anion 1.4 A, EC 0.7 A, EMC 0.7 A.

Fitted one species at a time, without the common slope, the same boxes give: cation 8.22e-11 m2/s with a viscosity of 10.35 mPa s; anion 8.90e-11 m2/s with a slope of the wrong sign; EC 1.40e-10 m2/s with a slope of the wrong sign; EMC 1.88e-10 m2/s with a viscosity of 7.90 mPa s. The spread of these says how well the boxes determine the slope.

**In an infinite box** the salt diffusion coefficient of an ideal solution is 8.62e-11 m2/s, 0.32 of the measured 2.66e-10. Multiplied by the measured thermodynamic factor it is 2.05e-10 m2/s, 0.77 of the measured value, which the cell tolerates at 1C, 1.5C, 2C. The Nernst-Einstein conductivity of the infinite box is 0.744 S/m against the measured 0.905, and the share of the current on the cation, if the ions moved alone, is 0.442.

## The two fits read together

Neither fit can be preferred. The fit over all boxes gives hydrodynamic radii of a plausible size and takes its slope from the smallest box, which differs from the others in every quantity. The fit without the smallest box describes its boxes well (a residual of 1.03 against 1.93) and gives radii smaller than an atom, which says that the coefficients of the larger boxes depend on the size less than the correction predicts for a liquid that diffuses this fast. The two are therefore reported as a range. In the base box (`scaled`) the correction adds between 1.10e-11 and 3.98e-11 m2/s to every coefficient, which is 17 % to 61 % of the cation's. The salt diffusion coefficient of an infinite box, with the measured thermodynamic factor, is between 0.77 and 0.96 of the measured value; the largest box without any correction gives 0.71. The cell tolerates the lower end at 1C, 1.5C, 2C and the upper end at 1C, 1.5C, 2C, 2.5C.

