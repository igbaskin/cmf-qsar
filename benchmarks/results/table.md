| Set | Model | q2 | RMSEcv | R2pred | RMSEpred | q2ex | RMSEcvex | alpha | lambda |
|---|---|---|---|---|---|---|---|---|---|
| **ACE** | paper (Table 2) | 0.72 | 1.24 | 0.65 | 1.24 | 0.67 | 1.31 | | |
| | this work, paper protocol | 0.72 | 1.24 | 0.59 | 1.35 | 0.67 | 1.31 | 0.50 | 20.000 |
| | this work, paper protocol, b = 0 | 0.72 | 1.24 | 0.65 | 1.24 | 0.68 | 1.28 | 1.00 | 20.000 |
| | this work, corrected H counts | 0.72 | 1.24 | 0.60 | 1.34 | 0.67 | 1.30 | 0.75 | 10.570 |
| | this work, RDKit parameterisation | 0.73 | 1.22 | 0.67 | 1.21 | 0.69 | 1.27 | 1.00 | 16.464 |
| | this work, RDKit param., scaled field kernels | 0.71 | 1.27 | 0.62 | 1.30 | 0.68 | 1.28 | 5.00 | 0.010 |
| | this work, program defaults | 0.73 | 1.22 | 0.63 | 1.28 | 0.72 | 1.20 | 3.00 | 0.010 |
| **AChE** | paper (Table 2) | 0.58 | 0.79 | 0.64 | 0.77 | 0.54 | 0.84 | | |
| | this work, paper protocol | 0.58 | 0.79 | 0.64 | 0.77 | 0.54 | 0.84 | 3.00 | 0.417 |
| | this work, corrected H counts | 0.59 | 0.77 | 0.65 | 0.77 | 0.56 | 0.82 | 3.00 | 0.452 |
| | this work, RDKit parameterisation | 0.59 | 0.77 | 0.65 | 0.76 | 0.57 | 0.81 | 3.00 | 0.396 |
| | this work, RDKit param., scaled field kernels | 0.60 | 0.76 | 0.45 | 0.96 | 0.51 | 0.87 | 1.00 | 0.019 |
| | this work, program defaults | 0.59 | 0.78 | 0.36 | 1.03 | 0.55 | 0.83 | 1.00 | 0.010 |
| **BZR** | paper (Table 2) | 0.40 | 0.51 | 0.20 | 0.79 | 0.27 | 0.65 | | |
| | this work, paper protocol | 0.40 | 0.51 | 0.20 | 0.79 | 0.27 | 0.65 | 0.01 | 12.259 |
| | this work, corrected H counts | 0.41 | 0.51 | 0.16 | 0.82 | 0.24 | 0.67 | 3.00 | 0.506 |
| | this work, RDKit parameterisation | 0.42 | 0.50 | 0.08 | 0.85 | 0.30 | 0.64 | 2.00 | 1.198 |
| | this work, RDKit param., scaled field kernels | 0.41 | 0.51 | 0.07 | 0.86 | 0.29 | 0.64 | 0.50 | 0.729 |
| | this work, program defaults | 0.43 | 0.50 | 0.07 | 0.86 | 0.31 | 0.64 | 0.50 | 1.000 |
| **COX2** | paper (Table 2) | 0.57 | 0.67 | 0.14 | 1.23 | 0.41 | 0.89 | | |
| | this work, paper protocol | 0.57 | 0.67 | 0.14 | 1.23 | 0.41 | 0.89 | 2.00 | 0.816 |
| | this work, corrected H counts | 0.60 | 0.64 | 0.18 | 1.20 | 0.46 | 0.86 | 1.00 | 1.282 |
| | this work, RDKit parameterisation | 0.61 | 0.63 | 0.31 | 1.10 | 0.51 | 0.82 | 1.00 | 2.448 |
| | this work, RDKit param., scaled field kernels | 0.61 | 0.64 | 0.27 | 1.13 | 0.50 | 0.83 | 1.00 | 0.263 |
| | this work, program defaults | 0.59 | 0.65 | 0.21 | 1.17 | 0.50 | 0.82 | 0.70 | 1.000 |
| **DHFR** | paper (Table 2) | 0.67 | 0.73 | 0.65 | 0.80 | 0.67 | 0.75 | | |
| | this work, paper protocol | 0.67 | 0.73 | 0.65 | 0.80 | 0.67 | 0.75 | 10.00 | 0.152 |
| | this work, corrected H counts | 0.68 | 0.72 | 0.64 | 0.81 | 0.67 | 0.74 | 5.00 | 0.544 |
| | this work, RDKit parameterisation | 0.65 | 0.75 | 0.51 | 0.95 | 0.62 | 0.80 | 0.10 | 9.293 |
| | this work, RDKit param., scaled field kernels | 0.66 | 0.74 | 0.64 | 0.82 | 0.67 | 0.75 | 10.00 | 0.560 |
| | this work, program defaults | 0.64 | 0.76 | 0.54 | 0.92 | 0.63 | 0.79 | 0.20 | 0.100 |
| **GPB** | paper (Table 2) | 0.69 | 0.60 | 0.51 | 0.84 | 0.59 | 0.71 | | |
| | this work, paper protocol | 0.69 | 0.60 | 0.51 | 0.84 | 0.59 | 0.71 | 10.00 | 0.118 |
| | this work, corrected H counts | 0.68 | 0.60 | 0.54 | 0.81 | 0.58 | 0.72 | 10.00 | 0.148 |
| | this work, RDKit parameterisation | 0.71 | 0.57 | 0.49 | 0.86 | 0.47 | 0.81 | 10.00 | 0.058 |
| | this work, RDKit param., scaled field kernels | 0.72 | 0.56 | 0.47 | 0.88 | 0.57 | 0.73 | 10.00 | 0.016 |
| | this work, program defaults | 0.61 | 0.66 | 0.57 | 0.79 | 0.58 | 0.72 | 3.00 | 1.000 |
| **THER** | paper (Table 2) | 0.60 | 1.18 | 0.31 | 1.86 | 0.40 | 1.56 | | |
| | this work, paper protocol | 0.60 | 1.18 | 0.31 | 1.86 | 0.33 | 1.65 | 10.00 | 0.010 |
| | this work, corrected H counts | 0.62 | 1.16 | 0.32 | 1.84 | 0.41 | 1.55 | 10.00 | 0.010 |
| | this work, RDKit parameterisation | 0.57 | 1.23 | 0.37 | 1.78 | 0.32 | 1.66 | 10.00 | 0.047 |
| | this work, RDKit param., scaled field kernels | 0.60 | 1.19 | 0.39 | 1.75 | 0.33 | 1.65 | 10.00 | 0.069 |
| | this work, program defaults | 0.54 | 1.27 | 0.26 | 1.93 | 0.43 | 1.52 | 1.00 | 0.010 |
| **THR** | paper (Table 2) | 0.73 | 0.50 | 0.63 | 0.66 | 0.71 | 0.54 | | |
| | this work, paper protocol | 0.73 | 0.50 | 0.63 | 0.66 | 0.71 | 0.54 | 0.25 | 6.377 |
| | this work, corrected H counts | 0.67 | 0.55 | 0.59 | 0.71 | 0.63 | 0.61 | 0.75 | 1.078 |
| | this work, RDKit parameterisation | 0.72 | 0.50 | 0.42 | 0.84 | 0.67 | 0.58 | 0.10 | 20.000 |
| | this work, RDKit param., scaled field kernels | 0.76 | 0.46 | 0.29 | 0.93 | 0.54 | 0.68 | 0.10 | 0.038 |
| | this work, program defaults | 0.69 | 0.53 | 0.37 | 0.88 | 0.56 | 0.66 | 0.20 | 0.100 |

Mean absolute deviation from the paper over the eight sets:

| Model | q2 | RMSEcv | R2pred | RMSEpred | q2ex | RMSEcvex |
|---|---|---|---|---|---|---|
| this work, paper protocol | 0.002 | 0.003 | 0.010 | 0.017 | 0.012 | 0.013 |
| this work, paper protocol, b = 0 | 0.001 | 0.000 | 0.005 | 0.005 | 0.010 | 0.027 |
| this work, corrected H counts | 0.019 | 0.017 | 0.029 | 0.032 | 0.025 | 0.022 |
| this work, RDKit parameterisation | 0.020 | 0.022 | 0.094 | 0.082 | 0.057 | 0.055 |
| this work, RDKit param., scaled field kernels | 0.021 | 0.024 | 0.120 | 0.107 | 0.053 | 0.048 |
| this work, program defaults | 0.033 | 0.034 | 0.123 | 0.112 | 0.052 | 0.052 |
