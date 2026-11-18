# Can a field of the site beat a count of its neighbours? (plan item PP3)

`python benchmarks/run_site_fields.py`, written 2026-11-18 13:27:07, 69 minutes.

280 positions of four domains of the mega-scale set, four substitutions drawn at each, the surroundings from the wild-type structure (envelope 0.1 per square angstrom), every description joined with the change of volume, hydropathy and charge. q2 in cross-validation, the mean of 3 repeats; every choice is made in inner folds of the same kind.

How close a description of the position is to the count of atoms in the envelope (Pearson correlation over the positions):

| Description | Correlation with the count |
|---|---:|
| residue area | -0.753 |
| side-chain area | -0.595 |
| amount of electrostatic | +0.679 |
| amount of steric | +0.868 |
| amount of hydrophobic | +0.685 |
| amount of hba | +0.587 |
| amount of hbd | +0.206 |

## Draw 0: 1120 substitutions, sd of ddG 1.24 kcal/mol

| Description | q2, segments held out | q2, domains held out | q2, positions held out |
|---|---:|---:|---:|
| mean | -0.004 | -0.019 | -0.005 |
| three columns kernel | +0.246 | +0.233 | +0.247 |
| one count | +0.450 | +0.428 | +0.458 |
| burial columns | +0.464 | +0.438 | +0.472 |
| accessible area | +0.522 | +0.497 | +0.527 |
| typed fields, cosine | +0.416 | +0.380 | +0.419 |
| typed fields, raw | +0.414 | +0.387 | +0.421 |
| anisotropy | +0.469 | +0.394 | +0.464 |
| anisotropy without the amounts | +0.470 | +0.389 | +0.475 |
| anchored points | +0.443 | +0.359 | +0.441 |

Differences in q2, with the 95 % interval over the repeats and the folds; bold where it excludes zero.

| Difference | segments held out | domains held out | positions held out |
|---|---|---|---|
| typed fields, cosine - one count | -0.034 (-0.102 to +0.029) | -0.048 (-0.105 to +0.000) | -0.039 (-0.100 to +0.024) |
| typed fields, cosine - burial columns | -0.048 (-0.117 to +0.019) | **-0.059 (-0.104 to -0.025)** | -0.052 (-0.112 to +0.003) |
| typed fields, cosine - accessible area | **-0.106 (-0.161 to -0.056)** | **-0.118 (-0.161 to -0.074)** | **-0.107 (-0.164 to -0.052)** |
| typed fields, raw - one count | -0.036 (-0.103 to +0.028) | -0.041 (-0.092 to +0.006) | -0.037 (-0.099 to +0.025) |
| typed fields, raw - burial columns | -0.050 (-0.120 to +0.018) | **-0.051 (-0.091 to -0.017)** | -0.051 (-0.109 to +0.006) |
| typed fields, raw - accessible area | **-0.108 (-0.164 to -0.056)** | **-0.110 (-0.148 to -0.066)** | **-0.106 (-0.163 to -0.051)** |
| anisotropy - one count | +0.019 (-0.031 to +0.065) | -0.034 (-0.121 to +0.051) | +0.006 (-0.042 to +0.056) |
| anisotropy - burial columns | +0.005 (-0.042 to +0.051) | -0.044 (-0.118 to +0.046) | -0.008 (-0.051 to +0.034) |
| anisotropy - accessible area | **-0.052 (-0.103 to -0.006)** | **-0.103 (-0.173 to -0.036)** | **-0.063 (-0.104 to -0.021)** |
| anisotropy without the amounts - one count | +0.020 (-0.034 to +0.069) | -0.039 (-0.132 to +0.050) | +0.017 (-0.030 to +0.067) |
| anisotropy without the amounts - burial columns | +0.006 (-0.044 to +0.054) | -0.049 (-0.130 to +0.046) | +0.004 (-0.039 to +0.045) |
| anisotropy without the amounts - accessible area | **-0.051 (-0.104 to -0.005)** | **-0.108 (-0.184 to -0.036)** | **-0.051 (-0.091 to -0.012)** |
| anchored points - one count | -0.008 (-0.082 to +0.065) | -0.069 (-0.144 to +0.061) | -0.017 (-0.090 to +0.054) |
| anchored points - burial columns | -0.021 (-0.101 to +0.055) | -0.079 (-0.166 to +0.053) | -0.030 (-0.103 to +0.038) |
| anchored points - accessible area | **-0.079 (-0.141 to -0.016)** | **-0.138 (-0.209 to -0.026)** | **-0.085 (-0.148 to -0.027)** |
| typed fields, raw - typed fields, cosine | -0.002 (-0.009 to +0.004) | **+0.008 (+0.002 to +0.013)** | +0.002 (-0.004 to +0.007) |
| accessible area - one count | **+0.072 (+0.033 to +0.110)** | **+0.069 (+0.052 to +0.088)** | **+0.069 (+0.031 to +0.109)** |

## Draw 1: 1120 substitutions, sd of ddG 1.26 kcal/mol

| Description | q2, segments held out | q2, domains held out | q2, positions held out |
|---|---:|---:|---:|
| mean | -0.004 | -0.022 | -0.005 |
| three columns kernel | +0.232 | +0.203 | +0.233 |
| one count | +0.451 | +0.419 | +0.458 |
| burial columns | +0.478 | +0.434 | +0.483 |
| accessible area | +0.517 | +0.480 | +0.520 |
| typed fields, cosine | +0.413 | +0.343 | +0.419 |
| typed fields, raw | +0.413 | +0.392 | +0.422 |
| anisotropy | +0.473 | +0.402 | +0.474 |
| anisotropy without the amounts | +0.474 | +0.394 | +0.476 |
| anchored points | +0.416 | +0.387 | +0.413 |

Differences in q2, with the 95 % interval over the repeats and the folds; bold where it excludes zero.

| Difference | segments held out | domains held out | positions held out |
|---|---|---|---|
| typed fields, cosine - one count | -0.038 (-0.096 to +0.019) | **-0.075 (-0.162 to -0.013)** | -0.039 (-0.097 to +0.020) |
| typed fields, cosine - burial columns | **-0.066 (-0.121 to -0.012)** | **-0.091 (-0.165 to -0.043)** | **-0.064 (-0.121 to -0.009)** |
| typed fields, cosine - accessible area | **-0.105 (-0.147 to -0.065)** | **-0.137 (-0.236 to -0.064)** | **-0.101 (-0.153 to -0.051)** |
| typed fields, raw - one count | -0.038 (-0.098 to +0.019) | -0.027 (-0.081 to +0.012) | -0.036 (-0.096 to +0.023) |
| typed fields, raw - burial columns | **-0.065 (-0.122 to -0.009)** | **-0.042 (-0.087 to -0.004)** | **-0.061 (-0.119 to -0.007)** |
| typed fields, raw - accessible area | **-0.104 (-0.148 to -0.064)** | **-0.088 (-0.152 to -0.044)** | **-0.099 (-0.152 to -0.049)** |
| anisotropy - one count | +0.022 (-0.031 to +0.073) | -0.017 (-0.076 to +0.047) | +0.017 (-0.031 to +0.068) |
| anisotropy - burial columns | -0.005 (-0.047 to +0.034) | -0.032 (-0.086 to +0.036) | -0.008 (-0.051 to +0.037) |
| anisotropy - accessible area | **-0.044 (-0.086 to -0.003)** | **-0.078 (-0.148 to -0.020)** | **-0.046 (-0.088 to -0.004)** |
| anisotropy without the amounts - one count | +0.023 (-0.030 to +0.073) | -0.025 (-0.093 to +0.037) | +0.018 (-0.030 to +0.071) |
| anisotropy without the amounts - burial columns | -0.005 (-0.046 to +0.034) | -0.040 (-0.102 to +0.025) | -0.007 (-0.052 to +0.041) |
| anisotropy without the amounts - accessible area | **-0.044 (-0.088 to -0.001)** | **-0.086 (-0.164 to -0.035)** | **-0.044 (-0.087 to -0.002)** |
| anchored points - one count | -0.034 (-0.104 to +0.041) | -0.032 (-0.077 to +0.015) | -0.045 (-0.118 to +0.028) |
| anchored points - burial columns | **-0.062 (-0.124 to -0.004)** | -0.048 (-0.108 to +0.012) | **-0.070 (-0.138 to -0.004)** |
| anchored points - accessible area | **-0.101 (-0.158 to -0.043)** | **-0.094 (-0.129 to -0.054)** | **-0.107 (-0.168 to -0.049)** |
| typed fields, raw - typed fields, cosine | +0.000 (-0.007 to +0.007) | **+0.049 (+0.005 to +0.159)** | +0.003 (-0.004 to +0.010) |
| accessible area - one count | **+0.067 (+0.025 to +0.112)** | **+0.061 (+0.052 to +0.073)** | **+0.063 (+0.023 to +0.105)** |
