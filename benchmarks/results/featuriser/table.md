# CMF as a featuriser (plan item ND1)

Test rows of a random split into 70 % training, 15 % validation and 15 % test, the split of the device's own protocol; every setting is chosen on the validation rows. R2 is given as the mean and the standard deviation over the seeds.

## COMPAS-1D: the gap of 8678 polybenzenoid hydrocarbons, 5 seeds

6074 training, 1302 validation and 1302 test molecules. The CMF vector has 1000 landmark molecules (rank 861).

| Model | Test R2 | Test RMSE (eV) | Test MAE (eV) |
|---|---|---|---|
| CMF kernel | +0.869 ± 0.016 | 0.144 ± 0.008 | 0.111 ± 0.003 |
| CMF kernel, Gaussian layer | +0.966 ± 0.005 | 0.073 ± 0.005 | 0.053 ± 0.001 |
| device, one-shot | +0.633 ± 0.014 | 0.241 ± 0.005 | 0.183 ± 0.004 |
| device, message passing | +0.982 ± 0.009 | 0.053 ± 0.012 | 0.035 ± 0.002 |
| fingerprint | +0.939 ± 0.013 | 0.098 ± 0.009 | 0.066 ± 0.001 |
| CMF vector, ridge regression | +0.870 ± 0.008 | 0.143 ± 0.004 | 0.112 ± 0.002 |
| CMF vector, multilayer network | +0.963 ± 0.005 | 0.076 ± 0.005 | 0.055 ± 0.001 |
| CMF vector, random forest | +0.768 ± 0.020 | 0.192 ± 0.007 | 0.149 ± 0.003 |
| collectors, one-shot, ridge regression | +0.617 ± 0.023 | 0.246 ± 0.007 | 0.188 ± 0.003 |
| collectors, one-shot, multilayer network | +0.640 ± 0.019 | 0.239 ± 0.006 | 0.179 ± 0.005 |
| collectors, one-shot, random forest | +0.634 ± 0.014 | 0.241 ± 0.004 | 0.178 ± 0.002 |
| collectors, message passing, ridge regression | +0.922 ± 0.020 | 0.110 ± 0.013 | 0.079 ± 0.002 |
| collectors, message passing, multilayer network | +0.983 ± 0.010 | 0.050 ± 0.013 | 0.033 ± 0.001 |
| collectors, message passing, random forest | +0.978 ± 0.013 | 0.057 ± 0.015 | 0.037 ± 0.002 |
| CMF vector + collectors, one-shot, ridge regression | +0.872 ± 0.014 | 0.142 ± 0.007 | 0.109 ± 0.002 |
| CMF vector + collectors, one-shot, multilayer network | +0.964 ± 0.005 | 0.075 ± 0.005 | 0.055 ± 0.002 |
| CMF vector + collectors, one-shot, random forest | +0.785 ± 0.019 | 0.184 ± 0.008 | 0.139 ± 0.004 |
| CMF vector + collectors, message passing, ridge regression | +0.943 ± 0.011 | 0.095 ± 0.008 | 0.070 ± 0.001 |
| CMF vector + collectors, message passing, multilayer network | +0.987 ± 0.007 | 0.045 ± 0.011 | 0.029 ± 0.002 |
| CMF vector + collectors, message passing, random forest | +0.972 ± 0.015 | 0.065 ± 0.016 | 0.044 ± 0.004 |
| mean of CMF kernel and device, one-shot | +0.823 ± 0.007 | 0.168 ± 0.004 | 0.130 ± 0.003 |
| weighted mean of CMF kernel and device, one-shot | +0.871 ± 0.011 | 0.143 ± 0.006 | 0.111 ± 0.002 |
| mean of CMF kernel and device, message passing | +0.962 ± 0.002 | 0.078 ± 0.002 | 0.061 ± 0.002 |
| weighted mean of CMF kernel and device, message passing | +0.983 ± 0.006 | 0.051 ± 0.008 | 0.036 ± 0.001 |

### Paired differences of the absolute error on COMPAS-1D (eV)

A negative value means the first model has the smaller error. The interval is a 95 % bootstrap interval over the test molecules of all seeds. Rows 8 to 10 hold the learner fixed and change only the input. They were added after the first COMPAS seed had been read; the other rows were fixed before any run.

| First model | Second model | Difference |
|---|---|---|
| CMF vector, ridge regression | CMF kernel | +0.001 [+0.000, +0.002] |
| CMF vector, multilayer network | CMF vector, ridge regression | -0.055 [-0.058, -0.053] |
| CMF vector, random forest | CMF vector, ridge regression | +0.037 [+0.034, +0.041] |
| CMF vector + collectors, one-shot, ridge regression | CMF vector, ridge regression | -0.003 [-0.003, -0.002] |
| CMF vector + collectors, message passing, ridge regression | CMF vector, ridge regression | -0.041 [-0.043, -0.038] |
| CMF vector + collectors, one-shot, multilayer network | device, one-shot | -0.127 [-0.131, -0.123] |
| CMF vector + collectors, message passing, multilayer network | device, message passing | -0.006 [-0.007, -0.005] |
| CMF vector + collectors, one-shot, multilayer network | collectors, one-shot, multilayer network | -0.123 [-0.127, -0.119] |
| CMF vector + collectors, message passing, multilayer network | collectors, message passing, multilayer network | -0.003 [-0.004, -0.002] |
| CMF vector + collectors, message passing, multilayer network | CMF vector, multilayer network | -0.026 [-0.027, -0.025] |
| weighted mean of CMF kernel and device, one-shot | CMF kernel | +0.000 [-0.001, +0.001] (includes zero) |
| weighted mean of CMF kernel and device, message passing | CMF kernel | -0.073 [-0.076, -0.071] |
| weighted mean of CMF kernel and device, message passing | device, message passing | +0.001 [+0.001, +0.001] |
| device, message passing | CMF kernel | -0.074 [-0.077, -0.072] |

## The Sutherland series, 10 seeds each

Test R2, the mean over the seeds. The last column is the mean over the series.

| Model | ACE | AChE | BZR | COX2 | DHFR | GPB | THER | THR | Mean |
|---|---|---|---|---|---|---|---|---|---|
| CMF kernel | +0.605 | +0.321 | +0.121 | +0.480 | +0.635 | +0.462 | +0.246 | +0.429 | +0.412 |
| device, one-shot | +0.589 | -0.073 | +0.344 | +0.224 | +0.600 | -0.126 | +0.142 | -0.072 | +0.203 |
| device, message passing | +0.712 | -0.016 | +0.392 | +0.418 | +0.648 | +0.271 | +0.316 | +0.105 | +0.356 |
| fingerprint | +0.626 | +0.302 | +0.436 | +0.500 | +0.715 | +0.157 | +0.433 | +0.261 | +0.429 |
| CMF vector, ridge regression | +0.628 | +0.328 | +0.176 | +0.498 | +0.652 | +0.492 | +0.292 | +0.373 | +0.430 |
| CMF vector, multilayer network | +0.631 | +0.425 | +0.350 | +0.508 | +0.648 | +0.375 | +0.348 | +0.518 | +0.475 |
| CMF vector, random forest | +0.515 | +0.168 | +0.277 | +0.284 | +0.468 | +0.318 | +0.233 | +0.421 | +0.336 |
| collectors, one-shot, ridge regression | +0.666 | +0.030 | +0.378 | +0.232 | +0.529 | -0.122 | +0.316 | -0.129 | +0.238 |
| collectors, one-shot, multilayer network | +0.695 | -0.198 | +0.341 | +0.263 | +0.551 | -0.145 | +0.167 | +0.211 | +0.236 |
| collectors, one-shot, random forest | +0.640 | -0.217 | +0.213 | +0.256 | +0.566 | -0.740 | +0.164 | +0.072 | +0.119 |
| collectors, message passing, ridge regression | +0.705 | -0.110 | +0.352 | +0.400 | +0.620 | +0.140 | +0.347 | +0.029 | +0.310 |
| collectors, message passing, multilayer network | +0.682 | -0.222 | +0.339 | +0.410 | +0.644 | -0.006 | +0.431 | +0.133 | +0.301 |
| collectors, message passing, random forest | +0.676 | -0.076 | +0.443 | +0.429 | +0.684 | +0.010 | +0.410 | +0.072 | +0.331 |
| CMF vector + collectors, one-shot, ridge regression | +0.709 | +0.354 | +0.202 | +0.484 | +0.651 | +0.481 | +0.319 | +0.302 | +0.438 |
| CMF vector + collectors, one-shot, multilayer network | +0.664 | +0.316 | +0.464 | +0.501 | +0.673 | +0.366 | +0.381 | +0.553 | +0.490 |
| CMF vector + collectors, one-shot, random forest | +0.686 | +0.166 | +0.325 | +0.343 | +0.494 | +0.336 | +0.312 | +0.417 | +0.385 |
| CMF vector + collectors, message passing, ridge regression | +0.704 | +0.317 | +0.353 | +0.434 | +0.651 | +0.228 | +0.363 | +0.021 | +0.384 |
| CMF vector + collectors, message passing, multilayer network | +0.685 | +0.242 | +0.470 | +0.524 | +0.685 | +0.190 | +0.486 | +0.235 | +0.440 |
| CMF vector + collectors, message passing, random forest | +0.732 | +0.123 | +0.476 | +0.451 | +0.672 | +0.234 | +0.457 | +0.237 | +0.423 |
| mean of CMF kernel and device, one-shot | +0.673 | +0.388 | +0.338 | +0.433 | +0.682 | +0.388 | +0.372 | +0.421 | +0.462 |
| weighted mean of CMF kernel and device, one-shot | +0.687 | +0.417 | +0.293 | +0.473 | +0.683 | +0.455 | +0.385 | +0.465 | +0.482 |
| mean of CMF kernel and device, message passing | +0.715 | +0.416 | +0.363 | +0.518 | +0.702 | +0.450 | +0.441 | +0.443 | +0.506 |
| weighted mean of CMF kernel and device, message passing | +0.718 | +0.423 | +0.316 | +0.497 | +0.698 | +0.410 | +0.379 | +0.463 | +0.488 |

Molecules: ACE 114, AChE 111, BZR 147, COX2 282, DHFR 361, GPB 66, THER 76, THR 88.

### Paired differences of the absolute error on the Sutherland series

In the units of each activity (pKi or pIC50). A negative value means the first model has the smaller error. Rows 8 to 10 hold the learner fixed and change only the input. They were added after the first COMPAS seed had been read; the other rows were fixed before any run.

| First model | Second model | ACE | AChE | BZR | COX2 | DHFR | GPB | THER | THR | Series where the interval excludes zero |
|---|---|---|---|---|---|---|---|---|---|---|
| CMF vector, ridge regression | CMF kernel | -0.033 (0) | -0.017 | -0.002 (0) | -0.020 | -0.026 | -0.008 (0) | -0.018 (0) | -0.003 (0) | first better in 3, second better in 0 |
| CMF vector, multilayer network | CMF vector, ridge regression | -0.026 (0) | -0.039 (0) | -0.060 | -0.009 (0) | +0.010 (0) | +0.064 | -0.107 (0) | -0.014 (0) | first better in 1, second better in 1 |
| CMF vector, random forest | CMF vector, ridge regression | +0.315 | +0.184 | +0.035 (0) | +0.133 | +0.178 | +0.141 | +0.095 (0) | +0.086 (0) | first better in 0, second better in 5 |
| CMF vector + collectors, one-shot, ridge regression | CMF vector, ridge regression | -0.090 (0) | +0.006 (0) | -0.017 (0) | +0.004 (0) | +0.013 (0) | +0.011 (0) | -0.056 (0) | +0.029 | first better in 0, second better in 1 |
| CMF vector + collectors, message passing, ridge regression | CMF vector, ridge regression | -0.086 (0) | +0.016 (0) | -0.022 (0) | +0.029 (0) | +0.010 (0) | +0.111 | -0.077 (0) | +0.143 | first better in 0, second better in 2 |
| CMF vector + collectors, one-shot, multilayer network | device, one-shot | -0.184 (0) | -0.247 | -0.115 | -0.188 | -0.054 | -0.262 | -0.157 (0) | -0.384 | first better in 6, second better in 0 |
| CMF vector + collectors, message passing, multilayer network | device, message passing | +0.036 (0) | -0.208 | -0.056 | -0.065 | -0.030 (0) | -0.022 (0) | -0.200 (0) | -0.093 | first better in 4, second better in 0 |
| CMF vector + collectors, one-shot, multilayer network | collectors, one-shot, multilayer network | +0.007 (0) | -0.226 | -0.114 | -0.168 | -0.087 | -0.261 | -0.225 | -0.261 | first better in 7, second better in 0 |
| CMF vector + collectors, message passing, multilayer network | collectors, message passing, multilayer network | -0.027 (0) | -0.195 | -0.064 | -0.063 | -0.048 | -0.066 (0) | -0.068 (0) | -0.039 (0) | first better in 4, second better in 0 |
| CMF vector + collectors, message passing, multilayer network | CMF vector, multilayer network | -0.013 (0) | +0.056 (0) | -0.030 (0) | -0.018 (0) | -0.038 | -0.020 (0) | -0.104 (0) | +0.063 (0) | first better in 1, second better in 0 |
| weighted mean of CMF kernel and device, one-shot | CMF kernel | -0.076 (0) | -0.025 (0) | -0.031 | +0.004 (0) | -0.032 | +0.017 (0) | -0.129 | -0.010 (0) | first better in 3, second better in 0 |
| weighted mean of CMF kernel and device, message passing | CMF kernel | -0.145 | -0.030 (0) | -0.050 | -0.018 (0) | -0.057 | +0.022 (0) | -0.130 | -0.013 (0) | first better in 4, second better in 0 |
| weighted mean of CMF kernel and device, message passing | device, message passing | -0.036 (0) | -0.240 | -0.015 (0) | -0.036 | -0.033 | -0.035 (0) | -0.100 (0) | -0.151 | first better in 4, second better in 0 |
| device, message passing | CMF kernel | -0.109 (0) | +0.209 | -0.035 (0) | +0.018 (0) | -0.024 (0) | +0.057 (0) | -0.030 (0) | +0.139 | first better in 0, second better in 2 |

(0) marks a difference whose 95 % interval includes zero.

