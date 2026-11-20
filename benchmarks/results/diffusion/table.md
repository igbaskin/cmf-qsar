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

## The boxes of the concentration series

| Run | What it is | Atoms | Ion pairs | Side of the box, A | Density, g/cm3 | Concentration, mol/L | Length, ns | Speed, ns/day |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| conc_c0p1_seed2 | 0.092 mol/kg, a second start | 13392 | 9 | 53.49 | 1.086 | 0.098 | 4.0 | 390 |
| conc_c0p1 | 0.092 mol/kg, charges multiplied by 0.7, 1,000 solvent molecules | 13392 | 9 | 53.48 | 1.086 | 0.098 | 4.0 | 406 |
| conc_c0p5_seed2 | 0.47 mol/kg, a second start | 13688 | 46 | 53.87 | 1.122 | 0.488 | 4.0 | 391 |
| conc_c0p5 | 0.47 mol/kg, charges multiplied by 0.7, 1,000 solvent molecules | 13688 | 46 | 53.87 | 1.123 | 0.489 | 4.0 | 402 |
| conc_c1p1_seed2 | 1.11 mol/kg, a second start | 14200 | 110 | 54.52 | 1.182 | 1.127 | 4.0 | 373 |
| conc_c1p1 | 1.11 mol/kg, charges multiplied by 0.7, 1,000 solvent molecules | 14200 | 110 | 54.52 | 1.182 | 1.127 | 4.0 | 215 |
| conc_c1p1_long | 1.11 mol/kg, 20 ns | 14200 | 110 | 54.51 | 1.183 | 1.128 | 20.0 | 376 |
| conc_c1p5_seed2 | 1.5 mol/kg, a second start | 14504 | 148 | 54.91 | 1.216 | 1.485 | 4.0 | 369 |
| conc_c1p5 | 1.5 mol/kg, charges multiplied by 0.7, 1,000 solvent molecules | 14504 | 148 | 54.89 | 1.217 | 1.486 | 4.0 | 397 |
| conc_c2p0 | 2.07 mol/kg, charges multiplied by 0.7, 1,000 solvent molecules | 14952 | 204 | 55.52 | 1.259 | 1.979 | 4.0 | 379 |
| conc_c2p0_seed2 | 2.07 mol/kg, a second start | 14952 | 204 | 55.49 | 1.261 | 1.983 | 4.0 | 368 |
| conc_c3p0_long | 3.32 mol/kg, 20 ns | 15944 | 328 | 56.85 | 1.343 | 2.965 | 20.0 | 301 |
| conc_c3p0_seed2 | 3.32 mol/kg, a second start | 15944 | 328 | 56.83 | 1.344 | 2.968 | 4.0 | 364 |
| conc_c3p0 | 3.32 mol/kg, charges multiplied by 0.7, 1,000 solvent molecules | 15944 | 328 | 56.82 | 1.345 | 2.969 | 4.0 | 370 |

## The range of concentrations

The measured study covers salt concentrations from 0.1 to 3.0 mol/L. This series simulates the same six concentrations at 298 K with the charges of the ions multiplied by 0.7, in a box of 1,000 solvent molecules. Every concentration is run from two starts: the box is packed again and given other velocities. Where both starts are there, a number is their mean and the error after +/- is the standard error of that mean, which for two starts is half their difference. Where one start is there, the error is the block error of that run. The measured functions are read at the concentration that the box really has.

**Self-diffusion and the salt diffusion coefficient.**

| Concentration, mol/L | Ion pairs | Starts | Cation, m2/s | Anion, m2/s | Exponent of the cation | Measured thermodynamic factor | Salt coefficient with that factor, m2/s | Measured salt coefficient, m2/s | Ratio to the measured | Within 10 % of the measured |
|---:|---:|---:|---|---|---:|---:|---|---:|---|---|
| 0.10 | 9 | 2 | 2.81e-10 +/- 1.25e-11 | 4.03e-10 +/- 5.96e-11 | 0.96 | 0.54 | 1.79e-10 +/- 1.56e-11 | 5.08e-10 | 0.35 +/- 0.03 | no |
| 0.49 | 46 | 2 | 1.93e-10 +/- 1.02e-11 | 2.59e-10 +/- 5.50e-12 | 0.91 | 1.35 | 2.97e-10 +/- 6.28e-12 | 3.98e-10 | 0.75 +/- 0.02 | no |
| 1.13 | 110 | 2 | 1.07e-10 +/- 1.63e-12 | 1.60e-10 +/- 1.54e-12 | 0.84 | 2.37 | 3.05e-10 +/- 1.59e-12 | 2.67e-10 | 1.14 +/- 0.01 | no |
| 1.49 | 148 | 2 | 7.24e-11 +/- 1.74e-12 | 1.00e-10 +/- 4.79e-12 | 0.78 | 2.92 | 2.46e-10 +/- 8.24e-12 | 2.14e-10 | 1.15 +/- 0.04 | no, but its error reaches it |
| 1.98 | 204 | 2 | 4.54e-11 +/- 9.82e-13 | 5.88e-11 +/- 9.73e-13 | 0.72 | 3.81 | 1.95e-10 +/- 3.63e-12 | 1.57e-10 | 1.24 +/- 0.02 | no |
| 2.97 | 328 | 2 | 1.89e-11 +/- 1.43e-12 | 2.19e-11 +/- 1.57e-12 | 0.56 | 6.74 | 1.37e-10 +/- 1.01e-11 | 8.47e-11 | 1.61 +/- 0.12 | no |

**What the study says about its own uncertainty.** Landesfeind and Gasteiger give no single number for it. Their error bars are the scatter of two cells with two current pulses each, and they write that these do not include the systematic error of building a cell. They report that a repeat at the end of a temperature cycle was up to 10 % lower and call that inside the uncertainty of the measurement. Their fitted function of the diffusion coefficient has a coefficient of determination of 0.978 for this electrolyte. Gate (a) is therefore read with 10 %. The comparison also multiplies by their thermodynamic factor, whose fitted function has a coefficient of determination of 0.943 and, in their words, the highest scatter of the three electrolytes of the study. The last column therefore includes the error of two measured functions.

**Gate (a).** The ratio is within 10 % of the measured value at 0 of the 6 concentrations. Over the series it runs from 0.35 to 1.61. The gate is not met. With a band twice as wide, 20 %, the ratio is inside at 2 of the 6 concentrations.

None of these coefficients is corrected for the size of the box. The correction of Yeh and Hummer, as a share of a coefficient, is the hydrodynamic radius of the species divided by the side of the box, times a constant. It does not contain the viscosity, so the share found at one concentration is a fair estimate at the others. The size runs put it between +8 % and +35 % for this size of box, at a factor of 0.8 on the charges. With that range the ratios become:

| Concentration, mol/L | Ratio as simulated | With the smaller correction | With the larger correction | Rates at which the cell tolerates the three |
|---:|---:|---:|---:|---|
| 0.10 | 0.35 | 0.38 | 0.48 | 1C; 1C; 1C |
| 0.49 | 0.75 | 0.81 | 1.01 | 1C, 1.5C, 2C; 1C, 1.5C, 2C; 1C, 1.5C, 2C, 2.5C |
| 1.13 | 1.14 | 1.24 | 1.54 | 1C, 1.5C, 2C, 2.5C; 1C, 1.5C, 2C, 2.5C; 1C, 1.5C, 2C, 2.5C |
| 1.49 | 1.15 | 1.25 | 1.55 | 1C, 1.5C, 2C, 2.5C; 1C, 1.5C, 2C, 2.5C; 1C, 1.5C, 2C, 2.5C |
| 1.98 | 1.24 | 1.35 | 1.68 | 1C, 1.5C, 2C, 2.5C; 1C, 1.5C, 2C, 2.5C; 1C, 1.5C, 2C, 2.5C |
| 2.97 | 1.61 | 1.75 | 2.17 | 1C, 1.5C, 2C, 2.5C; 1C, 1.5C, 2C, 2.5C; 1C, 1.5C, 2C, 2.5C |

The tolerated rates are those of the tolerance stage, which multiplied the whole measured function by one number. They are a guide here, because the ratio of this series changes with the concentration.

**The conductivity, S/m.** The conductivity is the stricter test, because its measured values carry an error of 3 % and no thermodynamic factor enters. The first value is from the displacement of the total charge.

| Concentration, mol/L | From the total charge | Measured | Ratio to the measured | Nernst-Einstein value | Ratio of the first to the Nernst-Einstein value | Share of the current on the cation if ions moved alone | Measured transference number |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.212 +/- 0.008 | 0.253 | 0.84 | 0.251 | 0.85 | 0.414 | 0.373 |
| 0.49 | 0.741 +/- 0.049 | 0.748 | 0.99 | 0.828 | 0.89 | 0.427 | 0.307 |
| 1.13 | 0.666 +/- 0.044 | 0.906 | 0.74 | 1.132 | 0.59 | 0.401 | 0.201 |
| 1.49 | 0.483 +/- 0.080 | 0.815 | 0.59 | 0.964 | 0.50 | 0.419 | 0.152 |
| 1.98 | 0.612 +/- 0.133 | 0.603 | 1.02 | 0.775 | 0.79 | 0.436 | 0.107 |
| 2.97 | 0.232 +/- 0.065 | 0.292 | 0.79 | 0.456 | 0.50 | 0.463 | 0.143 |

The simulated conductivity is largest at 0.49 mol/L among the concentrations that were run, and the measured one at 1.13 mol/L.

**A second measurement of the same electrolyte.** Nyman, Behm and Lindbergh (2008) measured this electrolyte at 298 K between 0.2 and 2 mol/L, and PyBaMM ships their fitted functions in another parameter set. The table gives their diffusion coefficient beside that of Landesfeind and Gasteiger. The two functions as shipped differ by far more than 10 % at most concentrations. Landesfeind and Gasteiger write that the values of Nyman are slightly higher than theirs, which the shipped function is not, so the two may not define the coefficient in the same way. This program has not read the paper of Nyman, and the difference is recorded here and not used as an uncertainty.

| Concentration, mol/L | Landesfeind and Gasteiger, m2/s | Nyman and others, m2/s | Ratio of the second to the first | Inside the range of the second study | Simulated, ratio to the first |
|---:|---:|---:|---:|---|---|
| 0.10 | 5.08e-10 | 4.48e-10 | 0.88 | no | 0.35 +/- 0.03 |
| 0.49 | 3.98e-10 | 3.13e-10 | 0.79 | yes | 0.75 +/- 0.02 |
| 1.13 | 2.67e-10 | 1.50e-10 | 0.56 | yes | 1.14 +/- 0.01 |
| 1.49 | 2.14e-10 | 9.01e-11 | 0.42 | yes | 1.15 +/- 0.04 |
| 1.98 | 1.57e-10 | 4.45e-11 | 0.28 | yes | 1.24 +/- 0.02 |
| 2.97 | 8.47e-11 | 8.21e-11 | 0.97 | no | 1.61 +/- 0.12 |

## Two sizes of the box at the charge factor of the series

The size runs above were made at a factor of 0.8 on the charges of the ions, and the series uses 0.7. At that factor two sizes exist at the concentration of the reference cell: one run of a box with 27 ion pairs (`scaled_07`, a side of 34.3 A) and 2 starts of the box of the series with 110 ion pairs (a side of 54.5 A). The correction of Yeh and Hummer says that the larger box is faster by the same amount, in m2/s, for every species.

| Species | Smaller box, m2/s | Larger box, m2/s | Larger over smaller | Difference, m2/s |
|---|---|---|---:|---:|
| Cation | 9.67e-11 +/- 6.4e-12 | 1.07e-10 +/- 1.6e-12 | 1.11 | 1.07e-11 |
| Anion | 1.35e-10 +/- 6.2e-12 | 1.60e-10 +/- 1.5e-12 | 1.18 | 2.49e-11 |
| EC | 1.96e-10 +/- 7.5e-12 | 2.25e-10 +/- 3.0e-12 | 1.15 | 2.91e-11 |
| EMC | 1.87e-10 +/- 7.5e-12 | 2.00e-10 +/- 7.8e-13 | 1.07 | 1.30e-11 |

The mean difference is 1.94e-11 +/- 4.5e-12 m2/s; the error is from the scatter over the species. Read as the correction of Yeh and Hummer, this difference is a viscosity of 3.4 mPa s. With that viscosity the box of the series is still too slow by 3.30e-11 m2/s, which is 31 % of the coefficient of the cation, 21 % of that of the anion and 26 % of the salt diffusion coefficient. The size runs at a factor of 0.8 gave a range from +8 % to +35 % for the same box, and this estimate is inside it. The smaller box was run once, so this is an estimate. The range from the size runs stays the one that the other sections use.

## How much a short run overstates

A coefficient is fitted between a fortieth and a tenth of the run. In a run of 4 ns that window is 100 to 400 ps, and in a run of 20 ns it is 500 to 2000 ps. Where the exponent is below 1 in the earlier window, the slope there is steeper than the slope at long times, and the short run gives too large a coefficient. The long run is one start; the short value is the mean of the short starts of the same box.

| Concentration, mol/L | Species | Short runs, m2/s | Exponent | Long run, m2/s | Exponent | Short over long |
|---:|---|---:|---:|---|---:|---:|
| 1.13 | Cation | 1.07e-10 | 0.84 | 1.00e-10 +/- 3.8e-12 | 0.94 | 1.07 |
| 1.13 | Anion | 1.60e-10 | 0.88 | 1.52e-10 +/- 3.6e-12 | 0.97 | 1.06 |
| 1.13 | EC | 2.25e-10 | 0.91 | 2.18e-10 +/- 4.2e-12 | 0.97 | 1.03 |
| 1.13 | EMC | 2.00e-10 | 0.90 | 1.88e-10 +/- 3.6e-12 | 0.96 | 1.06 |
| 2.97 | Cation | 1.89e-11 | 0.56 | 1.19e-11 +/- 3.7e-13 | 0.71 | 1.59 |
| 2.97 | Anion | 2.19e-11 | 0.58 | 1.52e-11 +/- 6.1e-13 | 0.76 | 1.45 |
| 2.97 | EC | 2.77e-11 | 0.63 | 1.99e-11 +/- 6.1e-13 | 0.79 | 1.39 |
| 2.97 | EMC | 2.69e-11 | 0.63 | 1.66e-11 +/- 7.8e-13 | 0.76 | 1.62 |

| Concentration, mol/L | Salt coefficient with the measured thermodynamic factor, ratio to the measured: short runs | The same from the long run | Short over long | Conductivity of the long run, S/m | Measured, S/m |
|---:|---:|---:|---:|---|---:|
| 1.13 | 1.14 | 1.07 | 1.06 | 0.725 +/- 0.212 | 0.906 |
| 2.97 | 1.61 | 1.06 | 1.53 | 0.169 +/- 0.043 | 0.293 |

## The series corrected for the length of the run

The long runs say by how much a run of 4 ns overstates the salt diffusion coefficient, at 2 of the concentrations. For the other concentrations the factor is estimated from the exponent of the cation in the short runs. It is linear in the exponent between the measured points, and it is 1 where the exponent is 1, because a displacement that grows in proportion to the time already has its final slope. Every ratio of the series is divided by its factor. This is an estimate: each long run is one start. A long run whose own exponent is still below 1 also overstates, and the corrected ratio at that concentration is then an upper limit.

| Concentration, mol/L | Exponent of the cation in the short runs | Factor by which the short runs overstate | Where the factor comes from | Ratio to the measured, as simulated | Ratio corrected for the length | Within 10 % of the measured |
|---:|---:|---:|---|---:|---:|---|
| 0.10 | 0.96 | 1.01 | estimated from the exponent | 0.35 | 0.35 | no |
| 0.49 | 0.91 | 1.04 | estimated from the exponent | 0.75 | 0.72 | no |
| 1.13 | 0.84 | 1.06 | the long run (its own exponent: 0.94) | 1.14 | 1.07 | yes |
| 1.49 | 0.78 | 1.17 | estimated from the exponent | 1.15 | 0.98 | yes |
| 1.98 | 0.72 | 1.26 | estimated from the exponent | 1.24 | 0.99 | yes |
| 2.97 | 0.56 | 1.53 | the long run (its own exponent: 0.71) | 1.61 | 1.06 | yes |

With this correction the ratio is within 10 % of the measured value at 4 of the 6 concentrations (1.13 mol/L, 1.49 mol/L, 1.98 mol/L, 2.97 mol/L), and it runs from 0.35 to 1.07. None of these ratios is corrected for the size of the box.

A longer run lowers a coefficient and a larger box raises it, so the two corrections act in opposite directions. With both of them:

| Concentration, mol/L | Ratio corrected for the length | With the smaller correction for the box as well | With the larger correction for the box as well |
|---:|---:|---:|---:|
| 0.10 | 0.35 | 0.38 | 0.47 |
| 0.49 | 0.72 | 0.78 | 0.97 |
| 1.13 | 1.07 | 1.16 | 1.45 |
| 1.49 | 0.98 | 1.07 | 1.33 |
| 1.98 | 0.99 | 1.07 | 1.33 |
| 2.97 | 1.06 | 1.14 | 1.42 |

With both corrections the ratio is within 10 % of the measured value at 2 of the 6 concentrations with the smaller correction for the box, and at 1 with the larger one.

## The simulated coefficient in the cell

Gate (a) compares two coefficients. This section asks what the difference costs. The reference cell is discharged at 298 K with the measured electrolyte, and again with the salt diffusion coefficient replaced by the measured function times the ratio of the series: linear between the six concentrations, constant below the lowest and above the highest. The other three electrolyte functions stay the measured ones. The numbers are the change of the discharge capacity; the tolerance stage allows 5 %.

| Rate | Capacity with the measured coefficient, A h | Lowest and highest concentration in the cell, mol/L | Change as simulated | Change with the smaller correction for the box | Change with the larger correction for the box | Change corrected for the length of the run | Change corrected for the length, with the smaller correction for the box | Change corrected for the length, with the larger correction for the box |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1C | 4.987 | 0.66 to 1.64 | +0.0 % | +0.0 % | +0.1 % | -0.0 % | +0.0 % | +0.1 % |
| 1.5C | 4.899 | 0.49 to 2.12 | +0.0 % | +0.1 % | +0.2 % | -0.0 % | +0.0 % | +0.2 % |
| 2C | 4.789 | 0.32 to 2.73 | +0.1 % | +0.3 % | +0.6 % | -0.4 % | -0.0 % | +0.5 % |
| 2.5C | 4.609 | 0.15 to 3.52 | -1.3 % | +0.5 % | +1.9 % | -9.4 % | -2.3 % | +1.4 % |

Inside the tolerance: as simulated, at 1C, 1.5C, 2C, 2.5C; with the smaller correction for the box, at 1C, 1.5C, 2C, 2.5C; with the larger correction for the box, at 1C, 1.5C, 2C, 2.5C; corrected for the length of the run, at 1C, 1.5C, 2C; corrected for the length, with the smaller correction for the box, at 1C, 1.5C, 2C, 2.5C; corrected for the length, with the larger correction for the box, at 1C, 1.5C, 2C, 2.5C.

## Other solvents: does the charge factor transfer?

Rule 2 asks that the simulation reproduces the measured conductivities of CALiSol-23 in rank and in level before a model is fitted on simulated labels. The factor of 0.7 on the charges of the ions was chosen on one electrolyte, EC:EMC 3:7. Here one formulation of each of the 22 solvent compositions is simulated with the same settings: the one whose molality is nearest to 1.11 mol/kg, so that the solvent changes and the concentration stays near that of the reference cell. Each box is one start of 4 ns, except EC:EMC 3:7, which is the mean of the two starts of the series. The measured value is the fit of CALiSol-23 at 298 K for that formulation. The reading rule was fixed before the boxes were run: the rank part holds if the lower end of the 95 % interval of the rank correlation is above zero, and the level part holds if the geometric mean of the ratios of the other compositions lies inside the range of ratios that the series found over the concentration of EC:EMC 3:7.

| Formulation | Molality, mol/kg | Ion pairs | Measured conductivity, S/m | Simulated, S/m | Ratio to the measured | Exponent of the cation | Ratio of the conductivity to the Nernst-Einstein value |
|---|---:|---:|---:|---|---:|---:|---:|
| LiPF6_EMC100_m1p500 | 1.500 | 156 | 0.523 | 0.293 +/- 0.093 | 0.56 | 0.77 | 0.36 |
| LiPF6_EC15_PC85_m0p446 | 0.446 | 44 | 0.593 | 0.273 +/- 0.051 | 0.46 | 0.86 | 0.47 |
| LiPF6_DEC50_PC50_m0p465 | 0.465 | 51 | 0.677 | 0.254 +/- 0.025 | 0.38 | 0.85 | 0.49 |
| LiPF6_DEC70_PC30_m0p898 | 0.898 | 101 | 0.679 | 0.352 +/- 0.084 | 0.52 | 0.78 | 0.61 |
| LiPF6_DEC20_PC80_m0p892 | 0.892 | 94 | 0.711 | 0.447 +/- 0.065 | 0.63 | 0.83 | 0.63 |
| LiPF6_EC10_EMC90_m1p500 | 1.500 | 153 | 0.713 | 0.730 +/- 0.082 | 1.02 | 0.78 | 0.82 |
| LiPF6_DEC60_PC40_m0p750 | 0.750 | 83 | 0.738 | 0.434 +/- 0.103 | 0.59 | 0.84 | 0.72 |
| LiPF6_EC60_PC40_m0p813 | 0.813 | 76 | 0.751 | 0.976 +/- 0.196 | 1.30 | 0.83 | 0.95 |
| LiPF6_DEC40_PC60_m0p802 | 0.802 | 87 | 0.760 | 0.470 +/- 0.151 | 0.62 | 0.81 | 0.74 |
| LiPF6_EC20_EMC80_m1p110 | 1.110 | 111 | 0.811 | 0.678 +/- 0.143 | 0.84 | 0.84 | 0.65 |
| LiPF6_EC30_EMC70_m1p110 (the reference electrolyte, two starts) | 1.110 | 110 | 0.883 | 0.666 +/- 0.044 | 0.75 | 0.84 | 0.59 |
| LiPF6_EC50_EMC50_m1p212 | 1.212 | 116 | 0.912 | 0.445 +/- 0.083 | 0.49 | 0.82 | 0.41 |
| LiPF6_EC40_EMC60_m1p077 | 1.077 | 105 | 0.918 | 0.748 +/- 0.101 | 0.81 | 0.85 | 0.62 |
| LiPF6_DMC100_m1p500 | 1.500 | 135 | 0.943 | 0.840 +/- 0.223 | 0.89 | 0.87 | 0.51 |
| LiPF6_EC30_EMC60_MA10_m1p596 | 1.596 | 152 | 0.973 | 0.809 +/- 0.147 | 0.83 | 0.78 | 0.68 |
| LiPF6_DMC90_EC10_m1p500 | 1.500 | 135 | 1.107 | 0.965 +/- 0.175 | 0.87 | 0.84 | 0.60 |
| LiPF6_DMC80_EC20_m1p500 | 1.500 | 135 | 1.141 | 0.538 +/- 0.205 | 0.47 | 0.84 | 0.33 |
| LiPF6_EC30_EMC50_MA20_m1p492 | 1.492 | 137 | 1.148 | 0.822 +/- 0.081 | 0.72 | 0.85 | 0.50 |
| LiPF6_DMC70_EC25_EMC05_m1p200 | 1.200 | 108 | 1.198 | 0.897 +/- 0.137 | 0.75 | 0.85 | 0.53 |
| LiPF6_DMC70_EC30_m1p000 | 1.000 | 89 | 1.212 | 2.182 +/- 0.533 | 1.80 | 0.89 | 1.18 |
| LiPF6_DMC60_EC30_MA10_m1p491 | 1.491 | 131 | 1.298 | 1.399 +/- 0.284 | 1.08 | 0.86 | 0.71 |
| LiPF6_DMC50_EC30_MA20_m1p500 | 1.500 | 129 | 1.486 | 1.068 +/- 0.298 | 0.72 | 0.85 | 0.49 |

**Rank.** Over 22 formulations the rank correlation of Spearman between the simulated and the measured conductivity is 0.83 (95 % interval 0.54 to 0.95, from resampling the formulations). One start of 4 ns scatters by 24 % in the logarithm of the conductivity, from the pairs of starts of the series. With that scatter alone, a simulation without any bias would reach a rank correlation of about 0.75. The rank part holds.

**Level.** The geometric mean of the ratios of the other compositions is 0.72. The series of EC:EMC 3:7 gave ratios from 0.59 to 1.02 over the concentration. The level part holds.

**By solvent.** The geometric mean of the ratio over the formulations that hold a solvent at 20 % of the weight or more:

| Solvent | Formulations | Geometric mean of the ratio to the measured |
|---|---:|---:|
| DEC | 5 | 0.54 |
| DMC | 7 | 0.87 |
| EC | 12 | 0.82 |
| EMC | 8 | 0.74 |
| MA | 2 | 0.72 |
| PC | 7 | 0.60 |

The exponent of the cation is below 0.8 for LiPF6_EMC100_m1p500, LiPF6_DEC70_PC30_m0p898, LiPF6_EC10_EMC90_m1p500, LiPF6_EC30_EMC60_MA10_m1p596. The concentration series showed that a run of 4 ns then overstates the coefficients, so these boxes need runs of 20 ns before their values are final.

