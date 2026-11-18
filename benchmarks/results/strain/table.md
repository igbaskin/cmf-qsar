# The strain the mutation leaves (plan item PP4)

`python benchmarks/run_strain.py`, written 2026-11-18 15:37:35, 13 minutes, after stage 1 (`--compute`) had relaxed the wild type at every position and split every energy.

174 wild-type sites and 2436 mutants of four domains of the mega-scale set, Amber ff14SB in vacuum. Each mutant is compared with the wild type relaxed in the same way at the same position. Every description is joined with the change of volume, hydropathy and charge. q2 in cross-validation, the mean of 3 repeats; every choice is made in inner folds of the same kind.

Relaxing the wild type at a site lowered its energy by 183 kJ/mol (median; 32 to 486 for the 5th and 95th percentiles). A mutant compared with the unrelaxed wild type would have counted that drop as part of the substitution.

## Draw 0: 657 substitutions at 174 positions, sd of ddG 1.23 kcal/mol

Each energy column against ddG over the substitutions (kJ/mol; the spread is the 5th and 95th percentiles):

| Energy column | Pearson r with ddG | Spearman rho with ddG | Median | Spread |
|---|---:|---:|---:|---|
| interaction, van der Waals | -0.392 | -0.296 | +0.3 | -25.2 to +38.6 |
| interaction, electrostatic | -0.041 | -0.066 | -3.6 | -778.6 to +681.1 |
| interaction, bonded | -0.251 | -0.157 | +0.3 | -17.6 to +29.4 |
| strain of the surroundings | -0.056 | -0.146 | -0.0 | -143.3 to +107.3 |
| strain of the residue | -0.120 | -0.190 | +3.3 | -19.9 to +33.7 |
| total energy difference | -0.084 | -0.098 | +6.7 | -831.8 to +784.0 |

Log Bayes factor against "this block does nothing", for the two blocks of one sum, on all samples of the draw (not a held-out number): accessible area +119.59, energy terms +31.95.

| Description | q2, segments held out | q2, domains held out | q2, positions held out |
|---|---:|---:|---:|
| mean | -0.015 | -0.026 | -0.010 |
| three columns kernel | +0.193 | +0.188 | +0.197 |
| one count | +0.428 | +0.380 | +0.442 |
| accessible area | +0.479 | +0.421 | +0.496 |
| field of the surroundings | +0.383 | +0.284 | +0.422 |
| total energy difference | +0.201 | +0.169 | +0.220 |
| energy terms | +0.305 | +0.293 | +0.319 |
| one count and energy terms | +0.516 | +0.494 | +0.533 |
| accessible area and energy terms | +0.503 | +0.475 | +0.513 |
| field and energy terms | +0.440 | +0.380 | +0.457 |

Differences in q2, with the 95 % interval over the repeats and the folds; bold where it excludes zero.

| Difference | segments held out | domains held out | positions held out |
|---|---|---|---|
| energy terms - one count | **-0.123 (-0.218 to -0.031)** | **-0.087 (-0.191 to -0.041)** | **-0.122 (-0.234 to -0.009)** |
| energy terms - accessible area | **-0.173 (-0.252 to -0.109)** | **-0.128 (-0.209 to -0.093)** | **-0.177 (-0.266 to -0.092)** |
| energy terms - total energy difference | +0.104 (-0.005 to +0.184) | **+0.123 (+0.061 to +0.252)** | **+0.100 (+0.028 to +0.165)** |
| accessible area and energy terms - accessible area | +0.025 (-0.016 to +0.064) | **+0.055 (+0.034 to +0.066)** | +0.017 (-0.036 to +0.069) |
| one count and energy terms - one count | **+0.088 (+0.032 to +0.142)** | **+0.115 (+0.064 to +0.157)** | **+0.092 (+0.024 to +0.167)** |
| one count and energy terms - accessible area | +0.037 (-0.022 to +0.090) | **+0.074 (+0.046 to +0.097)** | +0.037 (-0.018 to +0.095) |
| field and energy terms - field of the surroundings | **+0.057 (+0.003 to +0.108)** | **+0.096 (+0.038 to +0.150)** | +0.035 (-0.024 to +0.095) |
| accessible area - one count | +0.051 (-0.004 to +0.103) | **+0.041 (+0.015 to +0.062)** | +0.054 (-0.008 to +0.118) |
| field of the surroundings - one count | -0.045 (-0.165 to +0.063) | **-0.096 (-0.180 to -0.054)** | -0.020 (-0.148 to +0.108) |

## Draw 1: 657 substitutions at 174 positions, sd of ddG 1.23 kcal/mol

Each energy column against ddG over the substitutions (kJ/mol; the spread is the 5th and 95th percentiles):

| Energy column | Pearson r with ddG | Spearman rho with ddG | Median | Spread |
|---|---:|---:|---:|---|
| interaction, van der Waals | -0.360 | -0.284 | -1.1 | -26.8 to +39.3 |
| interaction, electrostatic | -0.041 | -0.078 | -1.5 | -613.2 to +693.1 |
| interaction, bonded | -0.225 | -0.120 | +0.3 | -18.4 to +26.7 |
| strain of the surroundings | -0.058 | -0.154 | -0.2 | -143.4 to +89.8 |
| strain of the residue | -0.100 | -0.184 | +3.7 | -16.9 to +32.2 |
| total energy difference | -0.074 | -0.100 | +6.5 | -687.8 to +773.5 |

Log Bayes factor against "this block does nothing", for the two blocks of one sum, on all samples of the draw (not a held-out number): accessible area +132.09, energy terms +38.82.

| Description | q2, segments held out | q2, domains held out | q2, positions held out |
|---|---:|---:|---:|
| mean | -0.013 | -0.022 | -0.011 |
| three columns kernel | +0.197 | +0.186 | +0.200 |
| one count | +0.426 | +0.408 | +0.426 |
| accessible area | +0.493 | +0.455 | +0.489 |
| field of the surroundings | +0.399 | +0.280 | +0.440 |
| total energy difference | +0.208 | +0.170 | +0.210 |
| energy terms | +0.278 | +0.291 | +0.296 |
| one count and energy terms | +0.488 | +0.470 | +0.506 |
| accessible area and energy terms | +0.500 | +0.469 | +0.506 |
| field and energy terms | +0.452 | +0.397 | +0.466 |

Differences in q2, with the 95 % interval over the repeats and the folds; bold where it excludes zero.

| Difference | segments held out | domains held out | positions held out |
|---|---|---|---|
| energy terms - one count | **-0.148 (-0.250 to -0.060)** | **-0.117 (-0.252 to -0.073)** | **-0.130 (-0.233 to -0.031)** |
| energy terms - accessible area | **-0.215 (-0.301 to -0.145)** | **-0.164 (-0.315 to -0.103)** | **-0.193 (-0.287 to -0.114)** |
| energy terms - total energy difference | **+0.070 (+0.010 to +0.120)** | **+0.121 (+0.059 to +0.234)** | **+0.086 (+0.017 to +0.144)** |
| accessible area and energy terms - accessible area | +0.007 (-0.025 to +0.040) | +0.014 (-0.039 to +0.044) | +0.017 (-0.026 to +0.060) |
| one count and energy terms - one count | **+0.061 (+0.018 to +0.107)** | **+0.062 (+0.013 to +0.088)** | **+0.080 (+0.022 to +0.141)** |
| one count and energy terms - accessible area | -0.005 (-0.062 to +0.045) | +0.016 (-0.031 to +0.056) | +0.017 (-0.046 to +0.078) |
| field and energy terms - field of the surroundings | **+0.053 (+0.018 to +0.090)** | **+0.117 (+0.025 to +0.186)** | +0.026 (-0.023 to +0.077) |
| accessible area - one count | **+0.067 (+0.011 to +0.129)** | **+0.047 (+0.023 to +0.084)** | **+0.063 (+0.004 to +0.126)** |
| field of the surroundings - one count | -0.027 (-0.117 to +0.050) | **-0.128 (-0.212 to -0.034)** | +0.013 (-0.096 to +0.131) |
