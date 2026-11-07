# A microstructure as the sample: field-overlap kernels against two-point statistics

Synthetic two-phase microstructures on a periodic voxel grid (Gaussian-filtered noise cut at the phase fraction; the phase fraction, the grain size, the aspect ratio and the direction of the grains vary from sample to sample). The label is the effective thermal conductivity of the periodic cell from a finite-volume solver (phase 0 at 1, phase 1 at the contrast): the component along x, which depends on the direction of the grains, and the mean over the directions, which does not. Training sets of each size drawn at random, the rest of the set held out; every setting chosen by five-fold cross-validation inside the training part. q2 is R2 on the held-out samples, mean over the draws with the standard deviation over the draws in brackets; RMSE is in units of the conductivity of phase 0.

# The gate: equal budgets, twenty draws, the Gaussian-process baseline

Every method chooses among the same number of settings (40) in the inner cross-validation, except the Gaussian process on the principal components, which chooses the number of components (five choices) and fits its length scales, amplitude and noise by the marginal likelihood. Twenty draws per training size. The paired tables give the difference between a method and a baseline on the same draws.

## 2D, 51 points per side, 400 samples, contrast 10

Labels: k_xx from 1.29 to 7.30 (sd 1.555), mean from 1.51 to 6.83 (sd 1.437). Solver checks: uniform phase 1: largest error 0.0e+00; laminate, layers across x: largest error 1.2e-13; random structure, asymmetry of the tensor 2.2e-16. Settings searched: smoothing width in {0, 1, 2, 5} voxels; layer in {linear, gauss 1} (the Gaussian bandwidth is in units of the median squared distance); ridge in {1e-05, 0.0001, 0.001, 0.01, 0.1} times the mean self-kernel (40 settings per kernel method); for the polynomial baseline {2, 3, 5, 10, 20} components, degree {1, 2}, ridge {1e-08, 0.0001, 0.01, 1} (40 settings); for the GP baseline the components alone, with the length scales, the amplitude and the noise fitted by the marginal likelihood. 20 draws per training size. Run time 3 min.

**Label: the conductivity along x (direction-dependent); from 1.29 to 7.30, sd 1.555.**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.850 (0.032) | +0.879 (0.008) | +0.887 (0.006) | +0.885 (0.008) | 0.599 (0.062) | 0.541 (0.017) | 0.524 (0.015) | 0.524 (0.019) |
| two-point statistics + PCA | +0.904 (0.054) | +0.953 (0.008) | +0.964 (0.003) | +0.968 (0.003) | 0.466 (0.123) | 0.336 (0.028) | 0.294 (0.014) | 0.275 (0.012) |
| two-point statistics + GP | +0.863 (0.080) | +0.932 (0.040) | +0.968 (0.008) | +0.976 (0.003) | 0.555 (0.150) | 0.394 (0.100) | 0.279 (0.033) | 0.236 (0.013) |
| overlap, generating frame | +0.777 (0.056) | +0.825 (0.039) | +0.873 (0.010) | +0.892 (0.013) | 0.728 (0.090) | 0.646 (0.067) | 0.554 (0.023) | 0.508 (0.033) |
| overlap, rotation-averaged | +0.794 (0.048) | +0.819 (0.049) | +0.847 (0.015) | +0.852 (0.009) | 0.701 (0.081) | 0.656 (0.085) | 0.608 (0.029) | 0.593 (0.021) |
| autocorrelation, linear | +0.895 (0.062) | +0.955 (0.011) | +0.966 (0.004) | +0.975 (0.004) | 0.488 (0.129) | 0.326 (0.038) | 0.285 (0.017) | 0.245 (0.020) |
| autocorrelation, normalised | +0.891 (0.052) | +0.939 (0.031) | +0.965 (0.004) | +0.974 (0.004) | 0.503 (0.105) | 0.377 (0.077) | 0.290 (0.019) | 0.251 (0.019) |
| autocorrelation, lattice-rotation-averaged | +0.857 (0.027) | +0.863 (0.059) | +0.882 (0.019) | +0.891 (0.008) | 0.586 (0.057) | 0.565 (0.103) | 0.533 (0.040) | 0.509 (0.018) |
| autocorrelation, radial | +0.855 (0.029) | +0.863 (0.066) | +0.885 (0.014) | +0.891 (0.008) | 0.589 (0.059) | 0.566 (0.106) | 0.527 (0.029) | 0.510 (0.018) |

Paired difference against *two-point statistics + GP* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | -0.013 (0.076) | -0.053 (0.039) | -0.081 (0.012) | -0.092 (0.009) | +0.044 (0.142) | +0.146 (0.097) | +0.245 (0.039) | +0.288 (0.026) | 5 of 20 | 1 of 20 | 0 of 20 | 0 of 20 |
| two-point statistics + PCA | +0.041 (0.086) | +0.021 (0.038) | -0.003 (0.009) | -0.008 (0.003) | -0.089 (0.169) | -0.059 (0.095) | +0.015 (0.036) | +0.038 (0.014) | 15 of 20 | 12 of 20 | 5 of 20 | 1 of 20 |
| overlap, generating frame | -0.086 (0.084) | -0.106 (0.058) | -0.094 (0.013) | -0.085 (0.014) | +0.173 (0.150) | +0.252 (0.125) | +0.275 (0.039) | +0.271 (0.039) | 4 of 20 | 1 of 20 | 0 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.070 (0.091) | -0.113 (0.065) | -0.120 (0.018) | -0.124 (0.010) | +0.146 (0.163) | +0.261 (0.133) | +0.329 (0.048) | +0.357 (0.027) | 3 of 20 | 1 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, linear | +0.031 (0.079) | +0.024 (0.038) | -0.001 (0.007) | -0.002 (0.005) | -0.067 (0.152) | -0.069 (0.096) | +0.007 (0.031) | +0.009 (0.025) | 14 of 20 | 14 of 20 | 8 of 20 | 9 of 20 |
| autocorrelation, normalised | +0.028 (0.079) | +0.007 (0.027) | -0.002 (0.007) | -0.003 (0.004) | -0.052 (0.148) | -0.018 (0.081) | +0.012 (0.028) | +0.015 (0.020) | 12 of 20 | 11 of 20 | 7 of 20 | 5 of 20 |
| autocorrelation, lattice-rotation-averaged | -0.007 (0.085) | -0.068 (0.073) | -0.085 (0.022) | -0.085 (0.009) | +0.031 (0.161) | +0.170 (0.146) | +0.254 (0.055) | +0.272 (0.026) | 8 of 20 | 2 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, radial | -0.008 (0.086) | -0.069 (0.082) | -0.083 (0.017) | -0.086 (0.009) | +0.034 (0.163) | +0.171 (0.156) | +0.249 (0.048) | +0.274 (0.026) | 8 of 20 | 2 of 20 | 0 of 20 | 0 of 20 |

Paired difference against *two-point statistics + PCA* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | -0.054 (0.044) | -0.074 (0.013) | -0.078 (0.007) | -0.084 (0.008) | +0.133 (0.098) | +0.205 (0.037) | +0.230 (0.021) | +0.250 (0.020) | 3 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| two-point statistics + GP | -0.041 (0.086) | -0.021 (0.038) | +0.003 (0.009) | +0.008 (0.003) | +0.089 (0.169) | +0.059 (0.095) | -0.015 (0.036) | -0.038 (0.014) | 5 of 20 | 8 of 20 | 15 of 20 | 19 of 20 |
| overlap, generating frame | -0.127 (0.061) | -0.128 (0.039) | -0.091 (0.011) | -0.077 (0.012) | +0.262 (0.118) | +0.311 (0.068) | +0.260 (0.028) | +0.233 (0.031) | 1 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.111 (0.059) | -0.134 (0.048) | -0.117 (0.016) | -0.116 (0.009) | +0.236 (0.119) | +0.320 (0.083) | +0.314 (0.036) | +0.319 (0.020) | 2 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, linear | -0.010 (0.061) | +0.002 (0.007) | +0.002 (0.005) | +0.006 (0.004) | +0.023 (0.132) | -0.010 (0.024) | -0.008 (0.020) | -0.030 (0.020) | 8 of 20 | 13 of 20 | 12 of 20 | 18 of 20 |
| autocorrelation, normalised | -0.013 (0.050) | -0.014 (0.027) | +0.001 (0.006) | +0.005 (0.004) | +0.037 (0.108) | +0.041 (0.065) | -0.003 (0.025) | -0.024 (0.019) | 3 of 20 | 5 of 20 | 12 of 20 | 18 of 20 |
| autocorrelation, lattice-rotation-averaged | -0.048 (0.053) | -0.090 (0.059) | -0.082 (0.019) | -0.077 (0.008) | +0.120 (0.116) | +0.229 (0.107) | +0.239 (0.043) | +0.234 (0.018) | 4 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, radial | -0.049 (0.054) | -0.090 (0.066) | -0.079 (0.015) | -0.078 (0.007) | +0.123 (0.118) | +0.230 (0.108) | +0.234 (0.035) | +0.235 (0.018) | 4 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (20 draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 1 (1), 2 (3), 5 (16) | 1 (2), 2 (3), 5 (15) | 0 (1), 1 (6), 2 (6), 5 (7) | 1 (11), 2 (6), 5 (3) |
| overlap, rotation-averaged | 0 (1), 1 (1), 2 (3), 5 (15) | 0 (5), 1 (2), 2 (1), 5 (12) | 0 (1), 1 (1), 2 (3), 5 (15) | 0 (2), 1 (1), 2 (1), 5 (16) |
| autocorrelation, linear | 0 (10), 1 (1), 2 (3), 5 (6) | 0 (18), 1 (2) | 0 (19), 1 (1) | 0 (20) |
| autocorrelation, normalised | 0 (8), 1 (1), 2 (3), 5 (8) | 0 (19), 1 (1) | 0 (20) | 0 (20) |
| autocorrelation, lattice-rotation-averaged | 0 (1), 1 (1), 2 (1), 5 (17) | 0 (1), 1 (1), 2 (2), 5 (16) | 0 (2), 1 (1), 2 (3), 5 (14) | 0 (2), 2 (1), 5 (17) |
| autocorrelation, radial | 0 (2), 1 (1), 2 (1), 5 (16) | 0 (2), 1 (1), 2 (5), 5 (12) | 0 (4), 2 (4), 5 (12) | 0 (1), 1 (2), 2 (1), 5 (16) |

**Label: the conductivity, mean over the directions (direction-free); from 1.51 to 6.83, sd 1.437.**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.978 (0.005) | +0.983 (0.001) | +0.984 (0.001) | +0.984 (0.001) | 0.210 (0.022) | 0.189 (0.007) | 0.183 (0.005) | 0.179 (0.009) |
| two-point statistics + PCA | +0.978 (0.005) | +0.981 (0.003) | +0.983 (0.003) | +0.985 (0.002) | 0.211 (0.021) | 0.194 (0.016) | 0.185 (0.013) | 0.175 (0.010) |
| two-point statistics + GP | +0.976 (0.008) | +0.977 (0.017) | +0.983 (0.002) | +0.984 (0.003) | 0.220 (0.035) | 0.212 (0.057) | 0.186 (0.010) | 0.181 (0.016) |
| overlap, generating frame | +0.914 (0.029) | +0.933 (0.015) | +0.957 (0.007) | +0.970 (0.002) | 0.416 (0.063) | 0.370 (0.039) | 0.297 (0.027) | 0.246 (0.012) |
| overlap, rotation-averaged | +0.932 (0.023) | +0.957 (0.012) | +0.972 (0.006) | +0.978 (0.003) | 0.370 (0.061) | 0.297 (0.041) | 0.241 (0.023) | 0.209 (0.013) |
| autocorrelation, linear | +0.975 (0.010) | +0.979 (0.007) | +0.982 (0.003) | +0.985 (0.002) | 0.226 (0.040) | 0.206 (0.032) | 0.190 (0.017) | 0.173 (0.010) |
| autocorrelation, normalised | +0.974 (0.008) | +0.978 (0.007) | +0.982 (0.002) | +0.985 (0.001) | 0.230 (0.032) | 0.212 (0.032) | 0.193 (0.012) | 0.176 (0.011) |
| autocorrelation, lattice-rotation-averaged | +0.974 (0.010) | +0.981 (0.004) | +0.984 (0.002) | +0.985 (0.003) | 0.228 (0.040) | 0.194 (0.021) | 0.184 (0.013) | 0.174 (0.014) |
| autocorrelation, radial | +0.975 (0.014) | +0.981 (0.006) | +0.983 (0.003) | +0.985 (0.003) | 0.220 (0.048) | 0.196 (0.026) | 0.186 (0.017) | 0.175 (0.014) |

Paired difference against *two-point statistics + GP* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.002 (0.006) | +0.006 (0.017) | +0.001 (0.002) | +0.001 (0.003) | -0.010 (0.027) | -0.024 (0.057) | -0.003 (0.010) | -0.003 (0.017) | 13 of 20 | 13 of 20 | 9 of 20 | 6 of 20 |
| two-point statistics + PCA | +0.002 (0.006) | +0.005 (0.018) | +0.000 (0.002) | +0.001 (0.003) | -0.009 (0.024) | -0.018 (0.059) | -0.001 (0.009) | -0.007 (0.016) | 12 of 20 | 12 of 20 | 9 of 20 | 15 of 20 |
| overlap, generating frame | -0.062 (0.028) | -0.044 (0.023) | -0.026 (0.008) | -0.014 (0.003) | +0.196 (0.062) | +0.158 (0.070) | +0.111 (0.031) | +0.065 (0.017) | 0 of 20 | 1 of 20 | 0 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.044 (0.025) | -0.020 (0.018) | -0.012 (0.007) | -0.005 (0.004) | +0.150 (0.074) | +0.084 (0.061) | +0.055 (0.031) | +0.028 (0.017) | 0 of 20 | 1 of 20 | 0 of 20 | 1 of 20 |
| autocorrelation, linear | -0.001 (0.012) | +0.002 (0.013) | -0.001 (0.003) | +0.002 (0.003) | +0.006 (0.051) | -0.007 (0.040) | +0.004 (0.017) | -0.009 (0.017) | 7 of 20 | 13 of 20 | 9 of 20 | 16 of 20 |
| autocorrelation, normalised | -0.002 (0.009) | +0.001 (0.013) | -0.001 (0.003) | +0.001 (0.003) | +0.010 (0.038) | -0.000 (0.042) | +0.007 (0.014) | -0.005 (0.016) | 6 of 20 | 7 of 20 | 6 of 20 | 11 of 20 |
| autocorrelation, lattice-rotation-averaged | -0.002 (0.013) | +0.005 (0.018) | +0.000 (0.002) | +0.001 (0.003) | +0.008 (0.054) | -0.018 (0.058) | -0.003 (0.012) | -0.007 (0.017) | 10 of 20 | 16 of 20 | 14 of 20 | 16 of 20 |
| autocorrelation, radial | -0.001 (0.012) | +0.004 (0.018) | -0.000 (0.003) | +0.001 (0.003) | +0.001 (0.043) | -0.016 (0.061) | -0.000 (0.017) | -0.006 (0.018) | 11 of 20 | 15 of 20 | 13 of 20 | 14 of 20 |

Paired difference against *two-point statistics + PCA* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.000 (0.005) | +0.001 (0.003) | +0.001 (0.003) | -0.001 (0.001) | -0.001 (0.022) | -0.006 (0.017) | -0.003 (0.013) | +0.004 (0.005) | 11 of 20 | 13 of 20 | 11 of 20 | 3 of 20 |
| two-point statistics + GP | -0.002 (0.006) | -0.005 (0.018) | -0.000 (0.002) | -0.001 (0.003) | +0.009 (0.024) | +0.018 (0.059) | +0.001 (0.009) | +0.007 (0.016) | 8 of 20 | 8 of 20 | 11 of 20 | 5 of 20 |
| overlap, generating frame | -0.064 (0.028) | -0.049 (0.013) | -0.026 (0.008) | -0.015 (0.002) | +0.205 (0.063) | +0.175 (0.033) | +0.112 (0.030) | +0.072 (0.011) | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.046 (0.024) | -0.025 (0.012) | -0.012 (0.007) | -0.007 (0.003) | +0.159 (0.065) | +0.102 (0.041) | +0.055 (0.031) | +0.035 (0.014) | 0 of 20 | 0 of 20 | 1 of 20 | 0 of 20 |
| autocorrelation, linear | -0.004 (0.010) | -0.003 (0.008) | -0.001 (0.004) | +0.000 (0.001) | +0.015 (0.041) | +0.011 (0.034) | +0.005 (0.022) | -0.002 (0.007) | 9 of 20 | 10 of 20 | 8 of 20 | 12 of 20 |
| autocorrelation, normalised | -0.004 (0.008) | -0.004 (0.008) | -0.001 (0.003) | -0.000 (0.001) | +0.019 (0.032) | +0.018 (0.035) | +0.008 (0.018) | +0.001 (0.008) | 5 of 20 | 6 of 20 | 5 of 20 | 9 of 20 |
| autocorrelation, lattice-rotation-averaged | -0.004 (0.011) | -0.000 (0.005) | +0.000 (0.002) | -0.000 (0.002) | +0.017 (0.045) | -0.000 (0.024) | -0.002 (0.013) | -0.000 (0.012) | 8 of 20 | 12 of 20 | 13 of 20 | 12 of 20 |
| autocorrelation, radial | -0.003 (0.012) | -0.000 (0.006) | -0.000 (0.004) | -0.000 (0.002) | +0.010 (0.042) | +0.001 (0.029) | +0.001 (0.022) | +0.001 (0.013) | 7 of 20 | 12 of 20 | 13 of 20 | 12 of 20 |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (20 draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 5 (20) | 0 (1), 2 (4), 5 (15) | 1 (16), 2 (1), 5 (3) | 0 (9), 1 (11) |
| overlap, rotation-averaged | 1 (2), 2 (1), 5 (17) | 5 (20) | 2 (1), 5 (19) | 5 (20) |
| autocorrelation, linear | 0 (5), 2 (3), 5 (12) | 0 (6), 1 (1), 2 (1), 5 (12) | 0 (6), 1 (3), 2 (3), 5 (8) | 0 (9), 1 (1), 2 (2), 5 (8) |
| autocorrelation, normalised | 0 (1), 1 (2), 5 (17) | 0 (4), 1 (2), 2 (2), 5 (12) | 0 (2), 2 (3), 5 (15) | 2 (4), 5 (16) |
| autocorrelation, lattice-rotation-averaged | 0 (2), 1 (1), 2 (6), 5 (11) | 0 (2), 2 (3), 5 (15) | 0 (3), 2 (3), 5 (14) | 0 (8), 1 (1), 2 (3), 5 (8) |
| autocorrelation, radial | 0 (4), 1 (1), 2 (2), 5 (13) | 0 (2), 2 (3), 5 (15) | 0 (2), 2 (2), 5 (16) | 0 (6), 2 (4), 5 (10) |

**Label: the PyMKS elastic stiffness C_xx (direction-dependent); from 1.40 to 6.90, sd 1.346.**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.872 (0.033) | +0.900 (0.006) | +0.905 (0.007) | +0.904 (0.006) | 0.478 (0.061) | 0.425 (0.012) | 0.414 (0.015) | 0.414 (0.017) |
| two-point statistics + PCA | +0.912 (0.064) | +0.955 (0.007) | +0.964 (0.003) | +0.968 (0.003) | 0.384 (0.114) | 0.286 (0.020) | 0.255 (0.011) | 0.239 (0.011) |
| two-point statistics + GP | +0.873 (0.085) | +0.939 (0.031) | +0.969 (0.008) | +0.980 (0.003) | 0.458 (0.143) | 0.323 (0.076) | 0.236 (0.029) | 0.190 (0.014) |
| overlap, generating frame | +0.762 (0.097) | +0.842 (0.015) | +0.883 (0.011) | +0.902 (0.010) | 0.645 (0.120) | 0.534 (0.025) | 0.461 (0.023) | 0.418 (0.026) |
| overlap, rotation-averaged | +0.786 (0.060) | +0.838 (0.027) | +0.865 (0.017) | +0.880 (0.010) | 0.616 (0.085) | 0.538 (0.043) | 0.494 (0.032) | 0.462 (0.024) |
| autocorrelation, linear | +0.909 (0.043) | +0.955 (0.014) | +0.969 (0.006) | +0.978 (0.003) | 0.396 (0.088) | 0.282 (0.042) | 0.237 (0.022) | 0.196 (0.016) |
| autocorrelation, normalised | +0.901 (0.044) | +0.944 (0.023) | +0.969 (0.004) | +0.978 (0.003) | 0.416 (0.083) | 0.314 (0.054) | 0.235 (0.016) | 0.200 (0.017) |
| autocorrelation, lattice-rotation-averaged | +0.880 (0.025) | +0.884 (0.055) | +0.907 (0.011) | +0.911 (0.008) | 0.464 (0.050) | 0.449 (0.087) | 0.410 (0.022) | 0.398 (0.015) |
| autocorrelation, radial | +0.877 (0.025) | +0.889 (0.048) | +0.907 (0.010) | +0.911 (0.008) | 0.469 (0.048) | 0.440 (0.075) | 0.409 (0.020) | 0.399 (0.015) |

Paired difference against *two-point statistics + GP* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | -0.001 (0.088) | -0.039 (0.028) | -0.064 (0.011) | -0.076 (0.008) | +0.019 (0.147) | +0.102 (0.070) | +0.178 (0.032) | +0.224 (0.024) | 7 of 20 | 2 of 20 | 0 of 20 | 0 of 20 |
| two-point statistics + PCA | +0.039 (0.109) | +0.016 (0.030) | -0.005 (0.009) | -0.012 (0.003) | -0.074 (0.185) | -0.038 (0.074) | +0.019 (0.032) | +0.049 (0.012) | 12 of 20 | 13 of 20 | 5 of 20 | 0 of 20 |
| overlap, generating frame | -0.111 (0.100) | -0.097 (0.033) | -0.086 (0.014) | -0.077 (0.011) | +0.186 (0.149) | +0.211 (0.078) | +0.225 (0.036) | +0.227 (0.030) | 2 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.086 (0.109) | -0.101 (0.045) | -0.104 (0.017) | -0.099 (0.010) | +0.158 (0.169) | +0.215 (0.093) | +0.258 (0.035) | +0.272 (0.027) | 3 of 20 | 1 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, linear | +0.037 (0.086) | +0.016 (0.034) | -0.000 (0.010) | -0.001 (0.003) | -0.062 (0.148) | -0.042 (0.085) | +0.001 (0.037) | +0.005 (0.016) | 13 of 20 | 11 of 20 | 8 of 20 | 8 of 20 |
| autocorrelation, normalised | +0.028 (0.084) | +0.005 (0.030) | +0.000 (0.009) | -0.002 (0.004) | -0.043 (0.142) | -0.009 (0.078) | -0.001 (0.032) | +0.009 (0.017) | 9 of 20 | 9 of 20 | 9 of 20 | 6 of 20 |
| autocorrelation, lattice-rotation-averaged | +0.007 (0.086) | -0.055 (0.064) | -0.062 (0.014) | -0.068 (0.008) | +0.005 (0.145) | +0.126 (0.117) | +0.174 (0.038) | +0.207 (0.022) | 7 of 20 | 3 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, radial | +0.004 (0.084) | -0.050 (0.057) | -0.062 (0.013) | -0.069 (0.008) | +0.011 (0.140) | +0.117 (0.106) | +0.173 (0.035) | +0.208 (0.022) | 6 of 20 | 3 of 20 | 0 of 20 | 0 of 20 |

Paired difference against *two-point statistics + PCA* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | -0.040 (0.052) | -0.055 (0.008) | -0.059 (0.007) | -0.064 (0.007) | +0.093 (0.090) | +0.140 (0.023) | +0.160 (0.019) | +0.175 (0.018) | 2 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| two-point statistics + GP | -0.039 (0.109) | -0.016 (0.030) | +0.005 (0.009) | +0.012 (0.003) | +0.074 (0.185) | +0.038 (0.074) | -0.019 (0.032) | -0.049 (0.012) | 8 of 20 | 7 of 20 | 15 of 20 | 20 of 20 |
| overlap, generating frame | -0.149 (0.115) | -0.113 (0.017) | -0.081 (0.011) | -0.066 (0.010) | +0.260 (0.160) | +0.248 (0.035) | +0.206 (0.022) | +0.179 (0.024) | 1 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.125 (0.064) | -0.116 (0.027) | -0.099 (0.017) | -0.088 (0.009) | +0.232 (0.102) | +0.253 (0.044) | +0.239 (0.031) | +0.223 (0.021) | 1 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, linear | -0.002 (0.066) | +0.001 (0.012) | +0.005 (0.006) | +0.011 (0.004) | +0.011 (0.117) | -0.004 (0.034) | -0.018 (0.021) | -0.043 (0.015) | 5 of 20 | 13 of 20 | 16 of 20 | 20 of 20 |
| autocorrelation, normalised | -0.011 (0.060) | -0.011 (0.020) | +0.005 (0.005) | +0.010 (0.004) | +0.031 (0.101) | +0.028 (0.048) | -0.019 (0.019) | -0.039 (0.015) | 4 of 20 | 6 of 20 | 18 of 20 | 20 of 20 |
| autocorrelation, lattice-rotation-averaged | -0.032 (0.064) | -0.071 (0.055) | -0.057 (0.009) | -0.057 (0.007) | +0.079 (0.110) | +0.163 (0.088) | +0.155 (0.019) | +0.159 (0.017) | 2 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, radial | -0.034 (0.065) | -0.065 (0.047) | -0.057 (0.008) | -0.057 (0.007) | +0.085 (0.112) | +0.155 (0.076) | +0.155 (0.017) | +0.160 (0.016) | 2 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (20 draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 1 (1), 2 (2), 5 (17) | 2 (5), 5 (15) | 0 (1), 1 (6), 2 (4), 5 (9) | 1 (10), 2 (6), 5 (4) |
| overlap, rotation-averaged | 0 (2), 1 (2), 2 (3), 5 (13) | 0 (4), 2 (2), 5 (14) | 0 (1), 2 (3), 5 (16) | 0 (5), 5 (15) |
| autocorrelation, linear | 0 (7), 1 (2), 2 (3), 5 (8) | 0 (18), 1 (2) | 0 (20) | 0 (20) |
| autocorrelation, normalised | 0 (7), 1 (3), 2 (2), 5 (8) | 0 (17), 1 (1), 2 (2) | 0 (20) | 0 (20) |
| autocorrelation, lattice-rotation-averaged | 0 (2), 2 (4), 5 (14) | 0 (7), 1 (1), 2 (2), 5 (10) | 0 (5), 1 (2), 2 (1), 5 (12) | 0 (3), 1 (1), 2 (2), 5 (14) |
| autocorrelation, radial | 0 (2), 2 (3), 5 (15) | 0 (1), 1 (2), 2 (2), 5 (15) | 0 (3), 2 (2), 5 (15) | 5 (20) |

**Label: the PyMKS elastic stiffness, mean of C_xx and C_yy (direction-free); from 1.47 to 6.41, sd 1.259.**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.979 (0.007) | +0.983 (0.002) | +0.984 (0.001) | +0.985 (0.002) | 0.182 (0.028) | 0.165 (0.008) | 0.159 (0.005) | 0.155 (0.009) |
| two-point statistics + PCA | +0.978 (0.007) | +0.981 (0.007) | +0.985 (0.002) | +0.987 (0.002) | 0.183 (0.029) | 0.171 (0.028) | 0.153 (0.009) | 0.145 (0.010) |
| two-point statistics + GP | +0.973 (0.018) | +0.980 (0.013) | +0.985 (0.002) | +0.987 (0.001) | 0.198 (0.057) | 0.171 (0.044) | 0.156 (0.012) | 0.144 (0.009) |
| overlap, generating frame | +0.887 (0.042) | +0.921 (0.013) | +0.949 (0.008) | +0.968 (0.004) | 0.418 (0.069) | 0.352 (0.031) | 0.283 (0.024) | 0.225 (0.016) |
| overlap, rotation-averaged | +0.902 (0.045) | +0.955 (0.015) | +0.971 (0.006) | +0.978 (0.003) | 0.383 (0.091) | 0.265 (0.041) | 0.212 (0.021) | 0.183 (0.014) |
| autocorrelation, linear | +0.975 (0.011) | +0.982 (0.005) | +0.985 (0.003) | +0.988 (0.002) | 0.194 (0.039) | 0.169 (0.021) | 0.153 (0.016) | 0.135 (0.012) |
| autocorrelation, normalised | +0.974 (0.007) | +0.980 (0.004) | +0.984 (0.002) | +0.987 (0.002) | 0.202 (0.028) | 0.176 (0.020) | 0.157 (0.009) | 0.144 (0.014) |
| autocorrelation, lattice-rotation-averaged | +0.973 (0.018) | +0.984 (0.002) | +0.986 (0.002) | +0.988 (0.002) | 0.199 (0.054) | 0.159 (0.011) | 0.148 (0.013) | 0.135 (0.010) |
| autocorrelation, radial | +0.973 (0.021) | +0.983 (0.003) | +0.985 (0.004) | +0.987 (0.002) | 0.200 (0.059) | 0.163 (0.013) | 0.153 (0.019) | 0.141 (0.010) |

Paired difference against *two-point statistics + GP* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.005 (0.018) | +0.002 (0.013) | -0.000 (0.002) | -0.002 (0.001) | -0.016 (0.056) | -0.006 (0.042) | +0.003 (0.011) | +0.011 (0.007) | 10 of 20 | 5 of 20 | 8 of 20 | 2 of 20 |
| two-point statistics + PCA | +0.005 (0.020) | +0.001 (0.011) | +0.001 (0.002) | -0.000 (0.001) | -0.015 (0.065) | +0.000 (0.036) | -0.003 (0.012) | +0.001 (0.008) | 10 of 20 | 8 of 20 | 10 of 20 | 11 of 20 |
| overlap, generating frame | -0.086 (0.031) | -0.059 (0.019) | -0.035 (0.009) | -0.019 (0.003) | +0.220 (0.055) | +0.181 (0.056) | +0.127 (0.028) | +0.080 (0.012) | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.071 (0.046) | -0.026 (0.019) | -0.013 (0.006) | -0.008 (0.003) | +0.185 (0.099) | +0.094 (0.057) | +0.057 (0.024) | +0.039 (0.011) | 0 of 20 | 1 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, linear | +0.002 (0.022) | +0.001 (0.010) | +0.001 (0.003) | +0.002 (0.002) | -0.004 (0.071) | -0.002 (0.030) | -0.003 (0.012) | -0.009 (0.010) | 10 of 20 | 8 of 20 | 12 of 20 | 16 of 20 |
| autocorrelation, normalised | +0.000 (0.016) | -0.000 (0.011) | -0.000 (0.002) | -0.000 (0.002) | +0.004 (0.047) | +0.005 (0.034) | +0.001 (0.012) | -0.000 (0.012) | 6 of 20 | 4 of 20 | 8 of 20 | 10 of 20 |
| autocorrelation, lattice-rotation-averaged | -0.000 (0.023) | +0.004 (0.013) | +0.001 (0.003) | +0.002 (0.002) | +0.001 (0.068) | -0.012 (0.043) | -0.008 (0.014) | -0.010 (0.012) | 8 of 20 | 9 of 20 | 13 of 20 | 18 of 20 |
| autocorrelation, radial | -0.001 (0.024) | +0.003 (0.013) | +0.000 (0.003) | +0.001 (0.002) | +0.002 (0.064) | -0.008 (0.043) | -0.003 (0.016) | -0.003 (0.012) | 7 of 20 | 9 of 20 | 12 of 20 | 15 of 20 |

Paired difference against *two-point statistics + PCA* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.000 (0.008) | +0.002 (0.006) | -0.001 (0.002) | -0.002 (0.001) | -0.001 (0.030) | -0.006 (0.025) | +0.006 (0.009) | +0.010 (0.006) | 8 of 20 | 7 of 20 | 4 of 20 | 0 of 20 |
| two-point statistics + GP | -0.005 (0.020) | -0.001 (0.011) | -0.001 (0.002) | +0.000 (0.001) | +0.015 (0.065) | -0.000 (0.036) | +0.003 (0.012) | -0.001 (0.008) | 10 of 20 | 12 of 20 | 10 of 20 | 9 of 20 |
| overlap, generating frame | -0.091 (0.042) | -0.060 (0.014) | -0.036 (0.008) | -0.019 (0.003) | +0.234 (0.071) | +0.181 (0.038) | +0.130 (0.025) | +0.080 (0.012) | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.076 (0.044) | -0.026 (0.015) | -0.014 (0.006) | -0.008 (0.002) | +0.200 (0.087) | +0.094 (0.044) | +0.060 (0.024) | +0.038 (0.011) | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, linear | -0.003 (0.012) | +0.001 (0.005) | -0.000 (0.004) | +0.002 (0.002) | +0.011 (0.041) | -0.002 (0.022) | +0.000 (0.017) | -0.010 (0.010) | 8 of 20 | 8 of 20 | 10 of 20 | 17 of 20 |
| autocorrelation, normalised | -0.004 (0.010) | -0.001 (0.006) | -0.001 (0.002) | +0.000 (0.002) | +0.018 (0.039) | +0.005 (0.023) | +0.004 (0.010) | -0.001 (0.010) | 6 of 20 | 5 of 20 | 7 of 20 | 9 of 20 |
| autocorrelation, lattice-rotation-averaged | -0.005 (0.017) | +0.003 (0.007) | +0.001 (0.003) | +0.002 (0.002) | +0.016 (0.051) | -0.012 (0.026) | -0.004 (0.013) | -0.010 (0.010) | 8 of 20 | 11 of 20 | 15 of 20 | 19 of 20 |
| autocorrelation, radial | -0.006 (0.020) | +0.002 (0.007) | -0.000 (0.005) | +0.001 (0.002) | +0.017 (0.055) | -0.008 (0.027) | +0.001 (0.021) | -0.004 (0.010) | 8 of 20 | 10 of 20 | 14 of 20 | 14 of 20 |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (20 draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 5 (20) | 1 (1), 2 (1), 5 (18) | 1 (9), 2 (2), 5 (9) | 1 (15), 5 (5) |
| overlap, rotation-averaged | 1 (2), 2 (2), 5 (16) | 5 (20) | 5 (20) | 5 (20) |
| autocorrelation, linear | 0 (1), 1 (4), 2 (1), 5 (14) | 0 (7), 2 (2), 5 (11) | 0 (7), 1 (1), 2 (7), 5 (5) | 0 (16), 1 (1), 2 (1), 5 (2) |
| autocorrelation, normalised | 0 (1), 2 (1), 5 (18) | 0 (1), 2 (1), 5 (18) | 1 (1), 2 (3), 5 (16) | 2 (7), 5 (13) |
| autocorrelation, lattice-rotation-averaged | 1 (6), 2 (3), 5 (11) | 0 (6), 1 (3), 2 (2), 5 (9) | 0 (4), 2 (7), 5 (9) | 0 (13), 2 (4), 5 (3) |
| autocorrelation, radial | 0 (1), 2 (4), 5 (15) | 0 (2), 1 (1), 2 (1), 5 (16) | 0 (1), 2 (4), 5 (15) | 0 (1), 1 (2), 2 (7), 5 (10) |

## 2D, 51 points per side, 400 samples, contrast 100

Labels: k_xx from 1.33 to 68.14 (sd 17.604), mean from 1.71 to 61.39 (sd 15.527). Solver checks: uniform phase 1: largest error 0.0e+00; laminate, layers across x: largest error 7.6e-13; random structure, asymmetry of the tensor 1.3e-15. Settings searched: smoothing width in {0, 1, 2, 5} voxels; layer in {linear, gauss 1} (the Gaussian bandwidth is in units of the median squared distance); ridge in {1e-05, 0.0001, 0.001, 0.01, 0.1} times the mean self-kernel (40 settings per kernel method); for the polynomial baseline {2, 3, 5, 10, 20} components, degree {1, 2}, ridge {1e-08, 0.0001, 0.01, 1} (40 settings); for the GP baseline the components alone, with the length scales, the amplitude and the noise fitted by the marginal likelihood. 20 draws per training size. Run time 4 min.

**Label: the conductivity along x (direction-dependent); from 1.33 to 68.14, sd 17.604.**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.690 (0.076) | +0.755 (0.016) | +0.770 (0.013) | +0.765 (0.023) | 9.727 (1.162) | 8.704 (0.263) | 8.445 (0.273) | 8.465 (0.356) |
| two-point statistics + PCA | +0.748 (0.103) | +0.828 (0.044) | +0.870 (0.012) | +0.876 (0.018) | 8.684 (1.702) | 7.247 (0.877) | 6.344 (0.295) | 6.127 (0.357) |
| two-point statistics + GP | +0.605 (0.217) | +0.770 (0.105) | +0.856 (0.034) | +0.882 (0.025) | 10.724 (2.805) | 8.271 (1.769) | 6.650 (0.772) | 5.987 (0.682) |
| overlap, generating frame | +0.562 (0.150) | +0.700 (0.039) | +0.753 (0.020) | +0.770 (0.029) | 11.497 (1.855) | 9.620 (0.640) | 8.738 (0.358) | 8.378 (0.518) |
| overlap, rotation-averaged | +0.573 (0.101) | +0.655 (0.073) | +0.686 (0.029) | +0.704 (0.031) | 11.431 (1.320) | 10.286 (1.020) | 9.851 (0.439) | 9.500 (0.403) |
| autocorrelation, linear | +0.750 (0.118) | +0.844 (0.038) | +0.879 (0.017) | +0.894 (0.015) | 8.609 (1.852) | 6.904 (0.817) | 6.120 (0.447) | 5.682 (0.390) |
| autocorrelation, normalised | +0.727 (0.114) | +0.832 (0.052) | +0.880 (0.020) | +0.899 (0.017) | 9.042 (1.791) | 7.137 (1.027) | 6.095 (0.521) | 5.526 (0.431) |
| autocorrelation, lattice-rotation-averaged | +0.669 (0.136) | +0.715 (0.155) | +0.774 (0.030) | +0.783 (0.026) | 9.977 (1.801) | 9.188 (1.992) | 8.358 (0.503) | 8.128 (0.372) |
| autocorrelation, radial | +0.641 (0.219) | +0.750 (0.041) | +0.770 (0.045) | +0.782 (0.026) | 10.269 (2.469) | 8.764 (0.691) | 8.407 (0.690) | 8.144 (0.370) |

Paired difference against *two-point statistics + GP* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.085 (0.221) | -0.015 (0.103) | -0.086 (0.038) | -0.117 (0.036) | -0.997 (2.855) | +0.433 (1.701) | +1.796 (0.813) | +2.478 (0.779) | 11 of 20 | 7 of 20 | 1 of 20 | 0 of 20 |
| two-point statistics + PCA | +0.143 (0.237) | +0.058 (0.111) | +0.014 (0.034) | -0.005 (0.032) | -2.039 (3.151) | -1.024 (1.902) | -0.306 (0.767) | +0.140 (0.813) | 14 of 20 | 13 of 20 | 13 of 20 | 5 of 20 |
| overlap, generating frame | -0.043 (0.236) | -0.070 (0.113) | -0.103 (0.042) | -0.112 (0.042) | +0.774 (2.899) | +1.350 (1.862) | +2.088 (0.857) | +2.391 (0.910) | 6 of 20 | 4 of 20 | 1 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.033 (0.242) | -0.115 (0.123) | -0.170 (0.053) | -0.178 (0.043) | +0.708 (3.027) | +2.015 (1.932) | +3.202 (1.002) | +3.513 (0.848) | 6 of 20 | 2 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, linear | +0.145 (0.237) | +0.074 (0.115) | +0.023 (0.027) | +0.012 (0.027) | -2.115 (3.132) | -1.367 (1.999) | -0.530 (0.602) | -0.305 (0.699) | 15 of 20 | 17 of 20 | 16 of 20 | 12 of 20 |
| autocorrelation, normalised | +0.121 (0.229) | +0.062 (0.101) | +0.024 (0.021) | +0.018 (0.032) | -1.681 (2.957) | -1.133 (1.670) | -0.555 (0.459) | -0.461 (0.846) | 16 of 20 | 16 of 20 | 17 of 20 | 13 of 20 |
| autocorrelation, lattice-rotation-averaged | +0.064 (0.169) | -0.055 (0.201) | -0.082 (0.054) | -0.099 (0.039) | -0.747 (2.309) | +0.917 (2.868) | +1.708 (1.081) | +2.141 (0.850) | 11 of 20 | 8 of 20 | 1 of 20 | 0 of 20 |
| autocorrelation, radial | +0.036 (0.184) | -0.020 (0.120) | -0.086 (0.067) | -0.100 (0.039) | -0.454 (2.433) | +0.493 (2.002) | +1.757 (1.253) | +2.157 (0.843) | 10 of 20 | 8 of 20 | 1 of 20 | 0 of 20 |

Paired difference against *two-point statistics + PCA* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | -0.057 (0.083) | -0.073 (0.044) | -0.100 (0.014) | -0.111 (0.014) | +1.042 (1.391) | +1.457 (0.868) | +2.101 (0.304) | +2.338 (0.307) | 6 of 20 | 2 of 20 | 0 of 20 | 0 of 20 |
| two-point statistics + GP | -0.143 (0.237) | -0.058 (0.111) | -0.014 (0.034) | +0.005 (0.032) | +2.039 (3.151) | +1.024 (1.902) | +0.306 (0.767) | -0.140 (0.813) | 6 of 20 | 7 of 20 | 7 of 20 | 15 of 20 |
| overlap, generating frame | -0.186 (0.156) | -0.128 (0.049) | -0.117 (0.026) | -0.107 (0.021) | +2.813 (2.035) | +2.373 (0.880) | +2.394 (0.515) | +2.251 (0.425) | 2 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.175 (0.122) | -0.173 (0.083) | -0.184 (0.028) | -0.172 (0.022) | +2.747 (1.798) | +3.039 (1.272) | +3.508 (0.453) | +3.373 (0.380) | 1 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, linear | +0.002 (0.107) | +0.016 (0.040) | +0.009 (0.016) | +0.017 (0.016) | -0.076 (1.625) | -0.344 (0.771) | -0.224 (0.386) | -0.445 (0.399) | 10 of 20 | 17 of 20 | 14 of 20 | 18 of 20 |
| autocorrelation, normalised | -0.021 (0.100) | +0.004 (0.040) | +0.010 (0.019) | +0.023 (0.012) | +0.358 (1.530) | -0.110 (0.819) | -0.249 (0.458) | -0.601 (0.311) | 5 of 20 | 12 of 20 | 16 of 20 | 20 of 20 |
| autocorrelation, lattice-rotation-averaged | -0.079 (0.168) | -0.113 (0.164) | -0.096 (0.027) | -0.093 (0.015) | +1.293 (2.340) | +1.941 (2.194) | +2.014 (0.479) | +2.001 (0.277) | 5 of 20 | 2 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, radial | -0.107 (0.248) | -0.078 (0.059) | -0.100 (0.043) | -0.094 (0.014) | +1.585 (2.997) | +1.516 (1.078) | +2.063 (0.670) | +2.017 (0.272) | 5 of 20 | 2 of 20 | 0 of 20 | 0 of 20 |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (20 draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 2 (5), 5 (15) | 2 (6), 5 (14) | 1 (1), 2 (6), 5 (13) | 1 (4), 2 (6), 5 (10) |
| overlap, rotation-averaged | 0 (3), 1 (4), 2 (3), 5 (10) | 0 (7), 2 (4), 5 (9) | 0 (3), 2 (3), 5 (14) | 0 (9), 1 (2), 5 (9) |
| autocorrelation, linear | 0 (6), 1 (2), 2 (6), 5 (6) | 0 (16), 2 (3), 5 (1) | 0 (19), 1 (1) | 0 (20) |
| autocorrelation, normalised | 0 (7), 1 (2), 2 (3), 5 (8) | 0 (15), 2 (2), 5 (3) | 0 (19), 1 (1) | 0 (20) |
| autocorrelation, lattice-rotation-averaged | 0 (2), 2 (2), 5 (16) | 0 (4), 1 (1), 2 (2), 5 (13) | 0 (4), 1 (1), 2 (1), 5 (14) | 0 (1), 1 (3), 2 (1), 5 (15) |
| autocorrelation, radial | 0 (1), 1 (1), 2 (2), 5 (16) | 0 (3), 1 (2), 2 (1), 5 (14) | 0 (1), 2 (3), 5 (16) | 0 (3), 2 (4), 5 (13) |

**Label: the conductivity, mean over the directions (direction-free); from 1.71 to 61.39, sd 15.527.**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.889 (0.026) | +0.907 (0.008) | +0.914 (0.005) | +0.916 (0.008) | 5.133 (0.548) | 4.719 (0.195) | 4.543 (0.122) | 4.458 (0.202) |
| two-point statistics + PCA | +0.881 (0.018) | +0.901 (0.019) | +0.916 (0.012) | +0.923 (0.009) | 5.326 (0.405) | 4.864 (0.467) | 4.475 (0.288) | 4.259 (0.231) |
| two-point statistics + GP | +0.860 (0.041) | +0.888 (0.036) | +0.906 (0.017) | +0.912 (0.025) | 5.746 (0.843) | 5.138 (0.756) | 4.734 (0.422) | 4.521 (0.601) |
| overlap, generating frame | +0.785 (0.044) | +0.830 (0.034) | +0.865 (0.013) | +0.884 (0.009) | 7.146 (0.763) | 6.373 (0.666) | 5.693 (0.307) | 5.248 (0.218) |
| overlap, rotation-averaged | +0.786 (0.080) | +0.851 (0.032) | +0.883 (0.016) | +0.900 (0.012) | 7.063 (1.281) | 5.951 (0.659) | 5.293 (0.345) | 4.847 (0.216) |
| autocorrelation, linear | +0.871 (0.052) | +0.900 (0.032) | +0.915 (0.014) | +0.924 (0.009) | 5.480 (0.984) | 4.852 (0.702) | 4.524 (0.362) | 4.236 (0.235) |
| autocorrelation, normalised | +0.876 (0.033) | +0.896 (0.033) | +0.911 (0.014) | +0.922 (0.008) | 5.409 (0.703) | 4.939 (0.721) | 4.610 (0.366) | 4.289 (0.218) |
| autocorrelation, lattice-rotation-averaged | +0.869 (0.053) | +0.900 (0.024) | +0.916 (0.019) | +0.925 (0.011) | 5.520 (0.991) | 4.869 (0.532) | 4.487 (0.466) | 4.191 (0.272) |
| autocorrelation, radial | +0.881 (0.026) | +0.891 (0.054) | +0.914 (0.018) | +0.925 (0.011) | 5.311 (0.573) | 5.024 (1.015) | 4.524 (0.438) | 4.200 (0.285) |

Paired difference against *two-point statistics + GP* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.029 (0.038) | +0.019 (0.037) | +0.008 (0.018) | +0.003 (0.025) | -0.613 (0.808) | -0.418 (0.789) | -0.191 (0.443) | -0.063 (0.599) | 16 of 20 | 15 of 20 | 11 of 20 | 4 of 20 |
| two-point statistics + PCA | +0.021 (0.036) | +0.013 (0.038) | +0.010 (0.020) | +0.011 (0.024) | -0.420 (0.733) | -0.274 (0.794) | -0.259 (0.489) | -0.262 (0.571) | 13 of 20 | 12 of 20 | 15 of 20 | 15 of 20 |
| overlap, generating frame | -0.074 (0.059) | -0.058 (0.044) | -0.041 (0.022) | -0.029 (0.026) | +1.400 (1.092) | +1.236 (0.866) | +0.959 (0.513) | +0.728 (0.611) | 2 of 20 | 3 of 20 | 1 of 20 | 2 of 20 |
| overlap, rotation-averaged | -0.074 (0.093) | -0.037 (0.032) | -0.023 (0.024) | -0.012 (0.027) | +1.317 (1.577) | +0.813 (0.658) | +0.559 (0.559) | +0.326 (0.641) | 5 of 20 | 3 of 20 | 3 of 20 | 3 of 20 |
| autocorrelation, linear | +0.011 (0.058) | +0.012 (0.048) | +0.008 (0.023) | +0.012 (0.025) | -0.266 (1.091) | -0.286 (1.000) | -0.210 (0.552) | -0.285 (0.589) | 14 of 20 | 15 of 20 | 15 of 20 | 15 of 20 |
| autocorrelation, normalised | +0.016 (0.040) | +0.009 (0.044) | +0.005 (0.022) | +0.010 (0.023) | -0.337 (0.824) | -0.199 (0.917) | -0.124 (0.553) | -0.232 (0.544) | 13 of 20 | 15 of 20 | 13 of 20 | 14 of 20 |
| autocorrelation, lattice-rotation-averaged | +0.009 (0.070) | +0.012 (0.040) | +0.009 (0.027) | +0.013 (0.025) | -0.226 (1.357) | -0.269 (0.853) | -0.247 (0.652) | -0.330 (0.609) | 14 of 20 | 13 of 20 | 14 of 20 | 16 of 20 |
| autocorrelation, radial | +0.021 (0.044) | +0.003 (0.057) | +0.008 (0.027) | +0.013 (0.025) | -0.435 (0.917) | -0.114 (1.101) | -0.210 (0.650) | -0.321 (0.611) | 14 of 20 | 13 of 20 | 13 of 20 | 17 of 20 |

Paired difference against *two-point statistics + PCA* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.008 (0.027) | +0.006 (0.020) | -0.002 (0.010) | -0.007 (0.005) | -0.193 (0.579) | -0.144 (0.473) | +0.068 (0.252) | +0.199 (0.127) | 15 of 20 | 12 of 20 | 5 of 20 | 1 of 20 |
| two-point statistics + GP | -0.021 (0.036) | -0.013 (0.038) | -0.010 (0.020) | -0.011 (0.024) | +0.420 (0.733) | +0.274 (0.794) | +0.259 (0.489) | +0.262 (0.571) | 7 of 20 | 8 of 20 | 5 of 20 | 5 of 20 |
| overlap, generating frame | -0.096 (0.039) | -0.071 (0.030) | -0.051 (0.017) | -0.040 (0.008) | +1.820 (0.647) | +1.509 (0.588) | +1.218 (0.408) | +0.989 (0.223) | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.095 (0.078) | -0.050 (0.030) | -0.033 (0.019) | -0.023 (0.010) | +1.737 (1.230) | +1.087 (0.625) | +0.818 (0.447) | +0.588 (0.242) | 0 of 20 | 1 of 20 | 1 of 20 | 0 of 20 |
| autocorrelation, linear | -0.010 (0.050) | -0.001 (0.035) | -0.002 (0.019) | +0.001 (0.005) | +0.154 (0.922) | -0.012 (0.746) | +0.048 (0.467) | -0.023 (0.140) | 10 of 20 | 13 of 20 | 12 of 20 | 12 of 20 |
| autocorrelation, normalised | -0.005 (0.029) | -0.004 (0.036) | -0.005 (0.020) | -0.001 (0.005) | +0.083 (0.593) | +0.075 (0.775) | +0.135 (0.490) | +0.030 (0.134) | 9 of 20 | 11 of 20 | 9 of 20 | 7 of 20 |
| autocorrelation, lattice-rotation-averaged | -0.012 (0.052) | -0.001 (0.023) | -0.001 (0.023) | +0.002 (0.007) | +0.194 (0.994) | +0.005 (0.512) | +0.012 (0.544) | -0.068 (0.187) | 10 of 20 | 11 of 20 | 12 of 20 | 12 of 20 |
| autocorrelation, radial | -0.000 (0.026) | -0.010 (0.050) | -0.002 (0.022) | +0.002 (0.007) | -0.015 (0.583) | +0.160 (0.924) | +0.048 (0.526) | -0.059 (0.194) | 10 of 20 | 11 of 20 | 12 of 20 | 13 of 20 |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (20 draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 5 (20) | 1 (1), 2 (3), 5 (16) | 1 (2), 2 (8), 5 (10) | 1 (10), 2 (3), 5 (7) |
| overlap, rotation-averaged | 1 (2), 2 (3), 5 (15) | 0 (3), 2 (5), 5 (12) | 0 (5), 5 (15) | 0 (4), 5 (16) |
| autocorrelation, linear | 0 (4), 1 (4), 2 (4), 5 (8) | 0 (6), 2 (1), 5 (13) | 0 (9), 1 (1), 2 (4), 5 (6) | 0 (4), 1 (4), 2 (5), 5 (7) |
| autocorrelation, normalised | 0 (1), 1 (3), 2 (6), 5 (10) | 0 (3), 1 (1), 2 (3), 5 (13) | 0 (3), 1 (1), 2 (11), 5 (5) | 0 (1), 1 (2), 2 (7), 5 (10) |
| autocorrelation, lattice-rotation-averaged | 0 (3), 1 (1), 2 (5), 5 (11) | 0 (2), 1 (1), 2 (3), 5 (14) | 0 (5), 2 (5), 5 (10) | 0 (7), 2 (1), 5 (12) |
| autocorrelation, radial | 0 (4), 2 (3), 5 (13) | 0 (2), 1 (1), 2 (2), 5 (15) | 0 (4), 1 (1), 2 (1), 5 (14) | 0 (1), 1 (2), 2 (5), 5 (12) |

**Label: the PyMKS elastic stiffness C_xx (direction-dependent); from 1.44 to 58.31, sd 12.378.**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.619 (0.130) | +0.712 (0.022) | +0.731 (0.020) | +0.726 (0.025) | 7.537 (1.240) | 6.617 (0.203) | 6.407 (0.273) | 6.427 (0.359) |
| two-point statistics + PCA | +0.657 (0.202) | +0.798 (0.041) | +0.828 (0.022) | +0.843 (0.020) | 7.025 (1.810) | 5.522 (0.572) | 5.115 (0.299) | 4.861 (0.249) |
| two-point statistics + GP | +0.570 (0.192) | +0.700 (0.209) | +0.830 (0.048) | +0.883 (0.022) | 7.927 (1.776) | 6.499 (1.957) | 5.050 (0.696) | 4.182 (0.379) |
| overlap, generating frame | +0.484 (0.189) | +0.637 (0.043) | +0.703 (0.026) | +0.722 (0.030) | 8.750 (1.547) | 7.419 (0.464) | 6.726 (0.323) | 6.470 (0.467) |
| overlap, rotation-averaged | +0.473 (0.157) | +0.579 (0.065) | +0.623 (0.054) | +0.664 (0.036) | 8.884 (1.283) | 7.986 (0.627) | 7.570 (0.508) | 7.103 (0.413) |
| autocorrelation, linear | +0.717 (0.125) | +0.820 (0.052) | +0.862 (0.020) | +0.889 (0.017) | 6.463 (1.271) | 5.188 (0.726) | 4.575 (0.358) | 4.076 (0.310) |
| autocorrelation, normalised | +0.671 (0.171) | +0.807 (0.086) | +0.868 (0.022) | +0.888 (0.019) | 6.904 (1.707) | 5.340 (1.046) | 4.483 (0.379) | 4.088 (0.356) |
| autocorrelation, lattice-rotation-averaged | +0.624 (0.174) | +0.702 (0.109) | +0.747 (0.039) | +0.761 (0.028) | 7.433 (1.529) | 6.653 (1.019) | 6.191 (0.421) | 5.995 (0.307) |
| autocorrelation, radial | +0.611 (0.174) | +0.701 (0.107) | +0.750 (0.037) | +0.758 (0.029) | 7.563 (1.509) | 6.667 (0.946) | 6.156 (0.394) | 6.029 (0.291) |

Paired difference against *two-point statistics + GP* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.049 (0.185) | +0.012 (0.205) | -0.100 (0.054) | -0.157 (0.027) | -0.390 (1.761) | +0.117 (1.878) | +1.357 (0.733) | +2.245 (0.423) | 12 of 20 | 7 of 20 | 1 of 20 | 0 of 20 |
| two-point statistics + PCA | +0.087 (0.256) | +0.098 (0.212) | -0.003 (0.061) | -0.041 (0.021) | -0.902 (2.223) | -0.977 (1.977) | +0.065 (0.862) | +0.679 (0.352) | 13 of 20 | 16 of 20 | 10 of 20 | 1 of 20 |
| overlap, generating frame | -0.086 (0.239) | -0.062 (0.216) | -0.127 (0.055) | -0.161 (0.037) | +0.823 (2.075) | +0.919 (1.991) | +1.677 (0.726) | +2.289 (0.562) | 8 of 20 | 4 of 20 | 1 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.097 (0.219) | -0.121 (0.225) | -0.208 (0.074) | -0.219 (0.037) | +0.957 (2.007) | +1.487 (2.076) | +2.521 (0.836) | +2.921 (0.485) | 4 of 20 | 3 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, linear | +0.147 (0.156) | +0.120 (0.209) | +0.032 (0.052) | +0.006 (0.019) | -1.464 (1.493) | -1.311 (1.906) | -0.475 (0.745) | -0.106 (0.345) | 16 of 20 | 18 of 20 | 16 of 20 | 13 of 20 |
| autocorrelation, normalised | +0.101 (0.162) | +0.107 (0.208) | +0.037 (0.056) | +0.005 (0.023) | -1.023 (1.549) | -1.160 (1.894) | -0.566 (0.817) | -0.094 (0.417) | 14 of 20 | 16 of 20 | 16 of 20 | 11 of 20 |
| autocorrelation, lattice-rotation-averaged | +0.054 (0.202) | +0.002 (0.243) | -0.083 (0.071) | -0.122 (0.029) | -0.494 (1.868) | +0.154 (2.220) | +1.142 (0.917) | +1.813 (0.414) | 12 of 20 | 7 of 20 | 1 of 20 | 0 of 20 |
| autocorrelation, radial | +0.042 (0.198) | +0.001 (0.244) | -0.080 (0.070) | -0.125 (0.029) | -0.363 (1.818) | +0.168 (2.219) | +1.106 (0.903) | +1.847 (0.406) | 12 of 20 | 7 of 20 | 1 of 20 | 0 of 20 |

Paired difference against *two-point statistics + PCA* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | -0.038 (0.159) | -0.086 (0.045) | -0.097 (0.027) | -0.117 (0.023) | +0.512 (1.390) | +1.094 (0.569) | +1.292 (0.372) | +1.566 (0.333) | 7 of 20 | 2 of 20 | 0 of 20 | 0 of 20 |
| two-point statistics + GP | -0.087 (0.256) | -0.098 (0.212) | +0.003 (0.061) | +0.041 (0.021) | +0.902 (2.223) | +0.977 (1.977) | -0.065 (0.862) | -0.679 (0.352) | 7 of 20 | 4 of 20 | 10 of 20 | 19 of 20 |
| overlap, generating frame | -0.173 (0.259) | -0.161 (0.051) | -0.125 (0.034) | -0.120 (0.032) | +1.725 (2.146) | +1.896 (0.584) | +1.612 (0.451) | +1.610 (0.448) | 3 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.184 (0.241) | -0.219 (0.070) | -0.205 (0.052) | -0.178 (0.033) | +1.859 (2.039) | +2.464 (0.711) | +2.455 (0.529) | +2.243 (0.406) | 2 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, linear | +0.060 (0.186) | +0.022 (0.052) | +0.034 (0.024) | +0.047 (0.012) | -0.561 (1.605) | -0.334 (0.695) | -0.540 (0.379) | -0.785 (0.200) | 13 of 20 | 15 of 20 | 19 of 20 | 20 of 20 |
| autocorrelation, normalised | +0.014 (0.232) | +0.009 (0.066) | +0.040 (0.023) | +0.046 (0.016) | -0.121 (2.083) | -0.183 (0.795) | -0.631 (0.349) | -0.772 (0.253) | 12 of 20 | 13 of 20 | 19 of 20 | 20 of 20 |
| autocorrelation, lattice-rotation-averaged | -0.033 (0.242) | -0.096 (0.112) | -0.080 (0.037) | -0.082 (0.020) | +0.408 (2.056) | +1.131 (1.096) | +1.076 (0.428) | +1.134 (0.276) | 5 of 20 | 1 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, radial | -0.046 (0.245) | -0.097 (0.110) | -0.077 (0.035) | -0.085 (0.020) | +0.539 (2.078) | +1.145 (1.037) | +1.041 (0.409) | +1.169 (0.268) | 5 of 20 | 1 of 20 | 0 of 20 | 0 of 20 |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (20 draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 1 (1), 2 (4), 5 (15) | 2 (7), 5 (13) | 1 (2), 2 (4), 5 (14) | 1 (4), 2 (5), 5 (11) |
| overlap, rotation-averaged | 0 (2), 1 (4), 2 (2), 5 (12) | 0 (7), 1 (1), 2 (2), 5 (10) | 0 (3), 1 (1), 2 (2), 5 (14) | 0 (9), 5 (11) |
| autocorrelation, linear | 0 (7), 1 (2), 2 (4), 5 (7) | 0 (17), 1 (1), 2 (1), 5 (1) | 0 (19), 1 (1) | 0 (20) |
| autocorrelation, normalised | 0 (8), 1 (1), 2 (5), 5 (6) | 0 (14), 1 (2), 2 (1), 5 (3) | 0 (19), 2 (1) | 0 (20) |
| autocorrelation, lattice-rotation-averaged | 0 (3), 2 (2), 5 (15) | 0 (7), 1 (1), 2 (1), 5 (11) | 0 (4), 1 (3), 2 (3), 5 (10) | 0 (3), 1 (1), 5 (16) |
| autocorrelation, radial | 0 (1), 1 (1), 2 (3), 5 (15) | 0 (2), 1 (3), 2 (3), 5 (12) | 0 (3), 1 (1), 2 (2), 5 (14) | 5 (20) |

**Label: the PyMKS elastic stiffness, mean of C_xx and C_yy (direction-free); from 1.61 to 50.58, sd 10.742.**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.847 (0.060) | +0.876 (0.012) | +0.888 (0.008) | +0.890 (0.010) | 4.129 (0.719) | 3.764 (0.183) | 3.590 (0.130) | 3.525 (0.226) |
| two-point statistics + PCA | +0.837 (0.061) | +0.880 (0.035) | +0.908 (0.011) | +0.919 (0.009) | 4.266 (0.748) | 3.681 (0.528) | 3.255 (0.189) | 3.025 (0.209) |
| two-point statistics + GP | +0.811 (0.067) | +0.866 (0.044) | +0.900 (0.020) | +0.910 (0.024) | 4.589 (0.799) | 3.872 (0.579) | 3.372 (0.374) | 3.165 (0.407) |
| overlap, generating frame | +0.716 (0.086) | +0.783 (0.043) | +0.829 (0.024) | +0.861 (0.015) | 5.658 (0.863) | 4.969 (0.550) | 4.425 (0.368) | 3.958 (0.276) |
| overlap, rotation-averaged | +0.724 (0.112) | +0.812 (0.042) | +0.850 (0.035) | +0.879 (0.014) | 5.530 (1.063) | 4.621 (0.562) | 4.132 (0.459) | 3.688 (0.234) |
| autocorrelation, linear | +0.840 (0.068) | +0.888 (0.024) | +0.905 (0.026) | +0.925 (0.010) | 4.221 (0.829) | 3.572 (0.405) | 3.288 (0.428) | 2.905 (0.222) |
| autocorrelation, normalised | +0.843 (0.063) | +0.890 (0.025) | +0.903 (0.020) | +0.923 (0.009) | 4.187 (0.787) | 3.537 (0.410) | 3.332 (0.367) | 2.941 (0.226) |
| autocorrelation, lattice-rotation-averaged | +0.832 (0.080) | +0.886 (0.023) | +0.907 (0.038) | +0.926 (0.015) | 4.302 (0.894) | 3.591 (0.363) | 3.237 (0.564) | 2.872 (0.285) |
| autocorrelation, radial | +0.829 (0.090) | +0.887 (0.028) | +0.905 (0.036) | +0.922 (0.014) | 4.333 (0.948) | 3.585 (0.424) | 3.268 (0.531) | 2.955 (0.263) |

Paired difference against *two-point statistics + GP* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.036 (0.052) | +0.010 (0.042) | -0.013 (0.019) | -0.020 (0.023) | -0.460 (0.663) | -0.108 (0.563) | +0.218 (0.305) | +0.360 (0.366) | 16 of 20 | 9 of 20 | 4 of 20 | 3 of 20 |
| two-point statistics + PCA | +0.026 (0.077) | +0.013 (0.040) | +0.007 (0.022) | +0.009 (0.020) | -0.323 (0.919) | -0.192 (0.541) | -0.117 (0.367) | -0.141 (0.312) | 14 of 20 | 14 of 20 | 11 of 20 | 10 of 20 |
| overlap, generating frame | -0.096 (0.099) | -0.083 (0.061) | -0.071 (0.028) | -0.049 (0.026) | +1.069 (1.041) | +1.096 (0.736) | +1.052 (0.401) | +0.793 (0.397) | 4 of 20 | 1 of 20 | 0 of 20 | 2 of 20 |
| overlap, rotation-averaged | -0.087 (0.132) | -0.054 (0.056) | -0.051 (0.040) | -0.031 (0.023) | +0.942 (1.334) | +0.749 (0.706) | +0.760 (0.545) | +0.523 (0.366) | 5 of 20 | 3 of 20 | 1 of 20 | 2 of 20 |
| autocorrelation, linear | +0.028 (0.052) | +0.021 (0.042) | +0.005 (0.023) | +0.015 (0.023) | -0.368 (0.642) | -0.300 (0.557) | -0.085 (0.366) | -0.261 (0.371) | 15 of 20 | 15 of 20 | 14 of 20 | 17 of 20 |
| autocorrelation, normalised | +0.031 (0.047) | +0.024 (0.038) | +0.002 (0.018) | +0.013 (0.023) | -0.402 (0.597) | -0.335 (0.491) | -0.041 (0.286) | -0.225 (0.361) | 15 of 20 | 15 of 20 | 13 of 20 | 17 of 20 |
| autocorrelation, lattice-rotation-averaged | +0.021 (0.078) | +0.020 (0.039) | +0.006 (0.035) | +0.016 (0.027) | -0.287 (0.887) | -0.282 (0.500) | -0.136 (0.506) | -0.293 (0.447) | 13 of 20 | 15 of 20 | 15 of 20 | 18 of 20 |
| autocorrelation, radial | +0.017 (0.084) | +0.020 (0.046) | +0.005 (0.032) | +0.012 (0.026) | -0.256 (0.915) | -0.287 (0.593) | -0.104 (0.472) | -0.210 (0.427) | 14 of 20 | 16 of 20 | 12 of 20 | 18 of 20 |

Paired difference against *two-point statistics + PCA* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.010 (0.066) | -0.004 (0.033) | -0.020 (0.012) | -0.029 (0.007) | -0.138 (0.774) | +0.084 (0.478) | +0.335 (0.204) | +0.501 (0.135) | 11 of 20 | 6 of 20 | 0 of 20 | 0 of 20 |
| two-point statistics + GP | -0.026 (0.077) | -0.013 (0.040) | -0.007 (0.022) | -0.009 (0.020) | +0.323 (0.919) | +0.192 (0.541) | +0.117 (0.367) | +0.141 (0.312) | 6 of 20 | 6 of 20 | 9 of 20 | 10 of 20 |
| overlap, generating frame | -0.121 (0.071) | -0.096 (0.047) | -0.078 (0.026) | -0.058 (0.012) | +1.391 (0.739) | +1.288 (0.597) | +1.169 (0.377) | +0.934 (0.178) | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.113 (0.109) | -0.068 (0.046) | -0.058 (0.037) | -0.040 (0.014) | +1.264 (1.042) | +0.941 (0.607) | +0.877 (0.488) | +0.664 (0.221) | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, linear | +0.003 (0.080) | +0.008 (0.032) | -0.003 (0.029) | +0.006 (0.008) | -0.045 (0.963) | -0.108 (0.482) | +0.032 (0.452) | -0.120 (0.162) | 10 of 20 | 11 of 20 | 12 of 20 | 15 of 20 |
| autocorrelation, normalised | +0.006 (0.069) | +0.010 (0.031) | -0.005 (0.022) | +0.004 (0.009) | -0.079 (0.825) | -0.144 (0.450) | +0.077 (0.361) | -0.084 (0.162) | 11 of 20 | 12 of 20 | 10 of 20 | 15 of 20 |
| autocorrelation, lattice-rotation-averaged | -0.005 (0.083) | +0.007 (0.042) | -0.001 (0.042) | +0.007 (0.015) | +0.036 (0.941) | -0.090 (0.622) | -0.019 (0.605) | -0.152 (0.266) | 12 of 20 | 10 of 20 | 13 of 20 | 16 of 20 |
| autocorrelation, radial | -0.008 (0.090) | +0.007 (0.038) | -0.002 (0.039) | +0.003 (0.013) | +0.067 (0.972) | -0.096 (0.526) | +0.013 (0.569) | -0.069 (0.238) | 10 of 20 | 10 of 20 | 12 of 20 | 14 of 20 |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (20 draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 2 (2), 5 (18) | 2 (1), 5 (19) | 1 (1), 2 (4), 5 (15) | 1 (3), 2 (1), 5 (16) |
| overlap, rotation-averaged | 1 (1), 2 (2), 5 (17) | 0 (2), 1 (1), 2 (3), 5 (14) | 0 (2), 2 (2), 5 (16) | 0 (3), 5 (17) |
| autocorrelation, linear | 0 (2), 1 (2), 2 (5), 5 (11) | 0 (7), 5 (13) | 0 (9), 1 (1), 2 (6), 5 (4) | 0 (12), 1 (3), 2 (2), 5 (3) |
| autocorrelation, normalised | 1 (1), 2 (5), 5 (14) | 0 (6), 1 (1), 2 (2), 5 (11) | 0 (3), 1 (3), 2 (7), 5 (7) | 0 (1), 1 (3), 2 (7), 5 (9) |
| autocorrelation, lattice-rotation-averaged | 0 (3), 2 (6), 5 (11) | 0 (9), 2 (2), 5 (9) | 0 (7), 1 (1), 2 (5), 5 (7) | 0 (11), 2 (4), 5 (5) |
| autocorrelation, radial | 0 (2), 2 (4), 5 (14) | 0 (4), 5 (16) | 0 (3), 2 (3), 5 (14) | 2 (2), 5 (18) |

## 3D, 21 points per side, 400 samples, contrast 10

Labels: k_xx from 1.34 to 8.03 (sd 1.679), mean from 1.68 to 7.31 (sd 1.524). Solver checks: uniform phase 1: largest error 0.0e+00; laminate, layers across x: largest error 1.5e-14; random structure, asymmetry of the tensor 1.1e-16. Settings searched: smoothing width in {0, 1, 2, 5} voxels; layer in {linear, gauss 1} (the Gaussian bandwidth is in units of the median squared distance); ridge in {1e-05, 0.0001, 0.001, 0.01, 0.1} times the mean self-kernel (40 settings per kernel method); for the polynomial baseline {2, 3, 5, 10, 20} components, degree {1, 2}, ridge {1e-08, 0.0001, 0.01, 1} (40 settings); for the GP baseline the components alone, with the length scales, the amplitude and the noise fitted by the marginal likelihood. 20 draws per training size. Run time 7 min.

**Label: the conductivity along x (direction-dependent); from 1.34 to 8.03, sd 1.679.**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.786 (0.029) | +0.809 (0.010) | +0.816 (0.014) | +0.826 (0.017) | 0.774 (0.052) | 0.731 (0.019) | 0.720 (0.026) | 0.697 (0.033) |
| two-point statistics + PCA | +0.956 (0.021) | +0.974 (0.013) | +0.984 (0.002) | +0.987 (0.002) | 0.344 (0.073) | 0.265 (0.053) | 0.212 (0.015) | 0.191 (0.012) |
| two-point statistics + GP | +0.932 (0.056) | +0.972 (0.007) | +0.981 (0.004) | +0.989 (0.001) | 0.415 (0.141) | 0.280 (0.033) | 0.228 (0.026) | 0.178 (0.010) |
| overlap, generating frame | +0.728 (0.102) | +0.833 (0.034) | +0.912 (0.024) | +0.951 (0.003) | 0.863 (0.142) | 0.680 (0.069) | 0.494 (0.064) | 0.370 (0.013) |
| overlap, rotation-averaged | +0.748 (0.090) | +0.778 (0.027) | +0.787 (0.024) | +0.803 (0.022) | 0.833 (0.130) | 0.788 (0.050) | 0.773 (0.044) | 0.743 (0.045) |
| autocorrelation, linear | +0.963 (0.016) | +0.975 (0.009) | +0.984 (0.003) | +0.988 (0.001) | 0.318 (0.062) | 0.262 (0.046) | 0.211 (0.019) | 0.183 (0.009) |
| autocorrelation, normalised | +0.927 (0.042) | +0.956 (0.014) | +0.976 (0.007) | +0.987 (0.002) | 0.438 (0.113) | 0.349 (0.054) | 0.259 (0.036) | 0.190 (0.014) |
| autocorrelation, lattice-rotation-averaged | +0.768 (0.060) | +0.796 (0.040) | +0.805 (0.030) | +0.822 (0.018) | 0.801 (0.100) | 0.754 (0.071) | 0.739 (0.051) | 0.706 (0.037) |
| autocorrelation, radial | +0.773 (0.055) | +0.796 (0.040) | +0.808 (0.024) | +0.822 (0.019) | 0.794 (0.094) | 0.753 (0.070) | 0.734 (0.048) | 0.706 (0.037) |

Paired difference against *two-point statistics + GP* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | -0.146 (0.039) | -0.162 (0.010) | -0.166 (0.013) | -0.163 (0.017) | +0.358 (0.106) | +0.450 (0.030) | +0.492 (0.027) | +0.520 (0.038) | 1 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| two-point statistics + PCA | +0.024 (0.058) | +0.002 (0.013) | +0.003 (0.004) | -0.002 (0.002) | -0.071 (0.148) | -0.015 (0.056) | -0.016 (0.024) | +0.013 (0.013) | 14 of 20 | 13 of 20 | 15 of 20 | 4 of 20 |
| overlap, generating frame | -0.204 (0.118) | -0.139 (0.035) | -0.069 (0.023) | -0.038 (0.003) | +0.448 (0.205) | +0.400 (0.080) | +0.266 (0.061) | +0.193 (0.017) | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.185 (0.107) | -0.194 (0.030) | -0.194 (0.025) | -0.186 (0.022) | +0.418 (0.192) | +0.507 (0.064) | +0.545 (0.053) | +0.565 (0.049) | 1 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, linear | +0.031 (0.061) | +0.003 (0.008) | +0.003 (0.004) | -0.001 (0.002) | -0.097 (0.164) | -0.018 (0.040) | -0.017 (0.029) | +0.006 (0.014) | 17 of 20 | 14 of 20 | 15 of 20 | 7 of 20 |
| autocorrelation, normalised | -0.005 (0.068) | -0.016 (0.011) | -0.006 (0.008) | -0.002 (0.002) | +0.023 (0.169) | +0.068 (0.043) | +0.031 (0.042) | +0.012 (0.017) | 9 of 20 | 1 of 20 | 6 of 20 | 4 of 20 |
| autocorrelation, lattice-rotation-averaged | -0.164 (0.058) | -0.176 (0.040) | -0.176 (0.028) | -0.167 (0.019) | +0.386 (0.124) | +0.473 (0.072) | +0.511 (0.046) | +0.528 (0.042) | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, radial | -0.159 (0.054) | -0.176 (0.040) | -0.173 (0.023) | -0.167 (0.019) | +0.378 (0.120) | +0.473 (0.072) | +0.506 (0.048) | +0.528 (0.043) | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |

Paired difference against *two-point statistics + PCA* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | -0.170 (0.037) | -0.165 (0.015) | -0.168 (0.014) | -0.161 (0.017) | +0.430 (0.091) | +0.465 (0.053) | +0.508 (0.026) | +0.507 (0.036) | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| two-point statistics + GP | -0.024 (0.058) | -0.002 (0.013) | -0.003 (0.004) | +0.002 (0.002) | +0.071 (0.148) | +0.015 (0.056) | +0.016 (0.024) | -0.013 (0.013) | 6 of 20 | 7 of 20 | 5 of 20 | 16 of 20 |
| overlap, generating frame | -0.228 (0.108) | -0.141 (0.037) | -0.072 (0.024) | -0.036 (0.004) | +0.519 (0.176) | +0.415 (0.089) | +0.282 (0.063) | +0.180 (0.018) | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.208 (0.097) | -0.196 (0.032) | -0.197 (0.024) | -0.184 (0.022) | +0.489 (0.167) | +0.523 (0.078) | +0.561 (0.046) | +0.552 (0.049) | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, linear | +0.007 (0.020) | +0.001 (0.012) | +0.000 (0.003) | +0.001 (0.001) | -0.026 (0.070) | -0.003 (0.050) | -0.001 (0.021) | -0.007 (0.011) | 13 of 20 | 11 of 20 | 13 of 20 | 17 of 20 |
| autocorrelation, normalised | -0.029 (0.045) | -0.018 (0.019) | -0.008 (0.006) | +0.000 (0.002) | +0.095 (0.126) | +0.084 (0.075) | +0.047 (0.033) | -0.001 (0.017) | 3 of 20 | 1 of 20 | 0 of 20 | 10 of 20 |
| autocorrelation, lattice-rotation-averaged | -0.188 (0.064) | -0.178 (0.042) | -0.179 (0.029) | -0.165 (0.019) | +0.457 (0.126) | +0.488 (0.087) | +0.527 (0.052) | +0.515 (0.040) | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, radial | -0.183 (0.060) | -0.178 (0.042) | -0.176 (0.024) | -0.165 (0.019) | +0.450 (0.121) | +0.488 (0.086) | +0.523 (0.048) | +0.515 (0.041) | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (20 draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 1 (1), 2 (2), 5 (17) | 1 (2), 2 (4), 5 (14) | 1 (1), 2 (2), 5 (17) | 1 (1), 2 (3), 5 (16) |
| overlap, rotation-averaged | 0 (1), 2 (2), 5 (17) | 0 (1), 1 (1), 2 (1), 5 (17) | 0 (1), 1 (1), 2 (1), 5 (17) | 2 (1), 5 (19) |
| autocorrelation, linear | 0 (13), 1 (2), 2 (1), 5 (4) | 0 (15), 1 (1), 2 (2), 5 (2) | 0 (17), 1 (2), 2 (1) | 0 (17), 1 (3) |
| autocorrelation, normalised | 0 (6), 1 (2), 2 (6), 5 (6) | 0 (9), 1 (2), 2 (4), 5 (5) | 0 (12), 1 (4), 2 (3), 5 (1) | 0 (18), 1 (2) |
| autocorrelation, lattice-rotation-averaged | 0 (4), 2 (4), 5 (12) | 0 (6), 2 (4), 5 (10) | 0 (3), 2 (5), 5 (12) | 0 (1), 1 (1), 2 (4), 5 (14) |
| autocorrelation, radial | 0 (3), 2 (5), 5 (12) | 0 (5), 1 (1), 2 (4), 5 (10) | 0 (4), 2 (3), 5 (13) | 0 (2), 1 (2), 2 (5), 5 (11) |

**Label: the conductivity, mean over the directions (direction-free); from 1.68 to 7.31, sd 1.524.**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.990 (0.001) | +0.991 (0.001) | +0.992 (0.000) | +0.992 (0.001) | 0.150 (0.008) | 0.142 (0.004) | 0.137 (0.003) | 0.135 (0.004) |
| two-point statistics + PCA | +0.991 (0.005) | +0.993 (0.001) | +0.994 (0.000) | +0.995 (0.000) | 0.142 (0.030) | 0.125 (0.012) | 0.113 (0.004) | 0.106 (0.004) |
| two-point statistics + GP | +0.990 (0.006) | +0.992 (0.002) | +0.994 (0.001) | +0.996 (0.000) | 0.151 (0.035) | 0.133 (0.015) | 0.115 (0.007) | 0.101 (0.005) |
| overlap, generating frame | +0.973 (0.004) | +0.974 (0.007) | +0.983 (0.003) | +0.989 (0.001) | 0.249 (0.021) | 0.242 (0.031) | 0.195 (0.020) | 0.158 (0.011) |
| overlap, rotation-averaged | +0.970 (0.029) | +0.985 (0.004) | +0.990 (0.002) | +0.992 (0.001) | 0.249 (0.089) | 0.182 (0.022) | 0.152 (0.015) | 0.137 (0.006) |
| autocorrelation, linear | +0.989 (0.007) | +0.993 (0.002) | +0.994 (0.001) | +0.995 (0.001) | 0.155 (0.044) | 0.122 (0.014) | 0.114 (0.010) | 0.105 (0.009) |
| autocorrelation, normalised | +0.988 (0.006) | +0.993 (0.001) | +0.993 (0.001) | +0.995 (0.001) | 0.162 (0.036) | 0.128 (0.008) | 0.126 (0.010) | 0.110 (0.009) |
| autocorrelation, lattice-rotation-averaged | +0.993 (0.002) | +0.994 (0.001) | +0.995 (0.000) | +0.996 (0.001) | 0.129 (0.015) | 0.119 (0.013) | 0.107 (0.004) | 0.094 (0.007) |
| autocorrelation, radial | +0.993 (0.002) | +0.994 (0.001) | +0.995 (0.000) | +0.995 (0.001) | 0.127 (0.015) | 0.114 (0.009) | 0.109 (0.004) | 0.103 (0.007) |

Paired difference against *two-point statistics + GP* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.001 (0.005) | -0.001 (0.002) | -0.002 (0.001) | -0.003 (0.000) | -0.001 (0.031) | +0.009 (0.014) | +0.023 (0.008) | +0.034 (0.005) | 8 of 20 | 4 of 20 | 1 of 20 | 0 of 20 |
| two-point statistics + PCA | +0.001 (0.007) | +0.001 (0.002) | +0.000 (0.001) | -0.000 (0.000) | -0.009 (0.043) | -0.008 (0.020) | -0.001 (0.007) | +0.006 (0.004) | 11 of 20 | 13 of 20 | 9 of 20 | 1 of 20 |
| overlap, generating frame | -0.017 (0.007) | -0.018 (0.007) | -0.011 (0.003) | -0.006 (0.001) | +0.098 (0.037) | +0.109 (0.035) | +0.081 (0.020) | +0.058 (0.011) | 1 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.020 (0.029) | -0.007 (0.004) | -0.004 (0.002) | -0.004 (0.001) | +0.098 (0.092) | +0.049 (0.027) | +0.038 (0.017) | +0.036 (0.008) | 2 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, linear | -0.001 (0.009) | +0.001 (0.002) | +0.000 (0.001) | -0.000 (0.001) | +0.004 (0.052) | -0.011 (0.020) | -0.000 (0.012) | +0.004 (0.009) | 13 of 20 | 16 of 20 | 13 of 20 | 6 of 20 |
| autocorrelation, normalised | -0.002 (0.007) | +0.001 (0.002) | -0.001 (0.001) | -0.001 (0.001) | +0.011 (0.041) | -0.005 (0.015) | +0.011 (0.012) | +0.009 (0.009) | 5 of 20 | 11 of 20 | 3 of 20 | 3 of 20 |
| autocorrelation, lattice-rotation-averaged | +0.003 (0.006) | +0.002 (0.002) | +0.001 (0.001) | +0.001 (0.001) | -0.022 (0.039) | -0.014 (0.020) | -0.007 (0.008) | -0.006 (0.007) | 16 of 20 | 15 of 20 | 15 of 20 | 15 of 20 |
| autocorrelation, radial | +0.003 (0.006) | +0.002 (0.002) | +0.001 (0.001) | -0.000 (0.001) | -0.024 (0.038) | -0.018 (0.018) | -0.006 (0.008) | +0.002 (0.007) | 16 of 20 | 17 of 20 | 17 of 20 | 7 of 20 |

Paired difference against *two-point statistics + PCA* over the draws: mean of (method minus baseline) in q2 and in RMSE, with the standard deviation over the draws in brackets, and the number of draws the method was ahead in q2.

| method | q2 diff., n_train 20 | q2 diff., n_train 50 | q2 diff., n_train 100 | q2 diff., n_train 200 | RMSE diff., n_train 20 | RMSE diff., n_train 50 | RMSE diff., n_train 100 | RMSE diff., n_train 200 | ahead, n_train 20 | ahead, n_train 50 | ahead, n_train 100 | ahead, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | -0.001 (0.005) | -0.002 (0.001) | -0.003 (0.000) | -0.003 (0.000) | +0.008 (0.030) | +0.017 (0.013) | +0.024 (0.004) | +0.028 (0.003) | 3 of 20 | 3 of 20 | 0 of 20 | 0 of 20 |
| two-point statistics + GP | -0.001 (0.007) | -0.001 (0.002) | -0.000 (0.001) | +0.000 (0.000) | +0.009 (0.043) | +0.008 (0.020) | +0.001 (0.007) | -0.006 (0.004) | 9 of 20 | 7 of 20 | 11 of 20 | 19 of 20 |
| overlap, generating frame | -0.018 (0.006) | -0.019 (0.007) | -0.011 (0.003) | -0.006 (0.001) | +0.107 (0.035) | +0.117 (0.029) | +0.082 (0.018) | +0.052 (0.011) | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| overlap, rotation-averaged | -0.021 (0.030) | -0.008 (0.004) | -0.005 (0.002) | -0.003 (0.001) | +0.107 (0.098) | +0.057 (0.026) | +0.039 (0.016) | +0.030 (0.007) | 1 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| autocorrelation, linear | -0.002 (0.006) | +0.000 (0.002) | -0.000 (0.001) | +0.000 (0.001) | +0.013 (0.035) | -0.003 (0.018) | +0.001 (0.009) | -0.002 (0.008) | 8 of 20 | 14 of 20 | 15 of 20 | 13 of 20 |
| autocorrelation, normalised | -0.003 (0.004) | -0.000 (0.002) | -0.001 (0.001) | -0.000 (0.001) | +0.020 (0.028) | +0.002 (0.016) | +0.012 (0.011) | +0.004 (0.009) | 4 of 20 | 8 of 20 | 1 of 20 | 8 of 20 |
| autocorrelation, lattice-rotation-averaged | +0.002 (0.005) | +0.001 (0.002) | +0.001 (0.001) | +0.001 (0.001) | -0.013 (0.034) | -0.006 (0.018) | -0.006 (0.005) | -0.012 (0.007) | 13 of 20 | 14 of 20 | 18 of 20 | 18 of 20 |
| autocorrelation, radial | +0.002 (0.005) | +0.001 (0.002) | +0.000 (0.001) | +0.000 (0.001) | -0.015 (0.034) | -0.011 (0.017) | -0.004 (0.005) | -0.003 (0.007) | 13 of 20 | 16 of 20 | 17 of 20 | 15 of 20 |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (20 draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 5 (20) | 5 (20) | 5 (20) | 5 (20) |
| overlap, rotation-averaged | 0 (1), 2 (2), 5 (17) | 2 (1), 5 (19) | 2 (1), 5 (19) | 5 (20) |
| autocorrelation, linear | 0 (2), 1 (2), 2 (9), 5 (7) | 0 (1), 1 (1), 2 (7), 5 (11) | 0 (3), 2 (14), 5 (3) | 1 (3), 2 (13), 5 (4) |
| autocorrelation, normalised | 0 (3), 1 (2), 2 (4), 5 (11) | 1 (3), 2 (6), 5 (11) | 0 (1), 1 (3), 2 (7), 5 (9) | 2 (12), 5 (8) |
| autocorrelation, lattice-rotation-averaged | 0 (5), 2 (8), 5 (7) | 0 (7), 1 (3), 2 (3), 5 (7) | 0 (6), 1 (3), 2 (10), 5 (1) | 0 (10), 1 (9), 2 (1) |
| autocorrelation, radial | 0 (6), 1 (1), 2 (6), 5 (7) | 0 (8), 1 (4), 2 (1), 5 (7) | 0 (2), 1 (6), 2 (12) | 0 (7), 1 (7), 2 (6) |

# The other three tests of the gate

## A property of two scales, on 2d51_n400_c10

Phase 1 carries a lower-scale state: a defect concentration c, uniform in [0, 1], that sets its conductivity by k1(c) = 10 / (1 + 4 c), c uniform in [0, 1]. The arrangement is the same 2d51_n400_c10 set; every cell is solved again with its own phase-1 conductivity. Correlation of c with k_xx -0.58, of the phase fraction with k_xx +0.60. The arrangement kernel is the autocorrelation kernel with its width and layer chosen as in the gate; the descriptor kernel is 1 + c c' or a Gaussian in c; the sum and the product combine one candidate of each; the two-width kernel mixes the finest and the coarsest width with weight {0.25, 0.5, 0.75} on the fine one. 20 draws per size. Run time 1 min.

**Label: k_xx (direction-dependent).**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lower-scale descriptor alone | -0.127 (0.744) | +0.226 (0.182) | +0.276 (0.096) | +0.257 (0.147) | 0.773 (0.218) | 0.662 (0.068) | 0.639 (0.047) | 0.642 (0.086) |
| phase fraction + descriptor | +0.824 (0.119) | +0.885 (0.021) | +0.906 (0.013) | +0.904 (0.013) | 0.306 (0.082) | 0.256 (0.025) | 0.230 (0.014) | 0.231 (0.027) |
| two-point statistics + descriptor + PCA | +0.830 (0.080) | +0.874 (0.054) | +0.911 (0.010) | +0.919 (0.012) | 0.304 (0.072) | 0.264 (0.049) | 0.224 (0.015) | 0.213 (0.027) |
| arrangement kernel alone | +0.150 (0.174) | +0.148 (0.266) | +0.328 (0.048) | +0.321 (0.064) | 0.694 (0.062) | 0.691 (0.093) | 0.617 (0.025) | 0.616 (0.061) |
| sum kernel | +0.294 (0.911) | +0.655 (0.100) | +0.735 (0.039) | +0.741 (0.055) | 0.572 (0.265) | 0.441 (0.058) | 0.387 (0.034) | 0.379 (0.058) |
| product kernel | +0.803 (0.208) | +0.938 (0.025) | +0.960 (0.020) | +0.974 (0.011) | 0.310 (0.134) | 0.186 (0.039) | 0.147 (0.036) | 0.118 (0.030) |
| two widths as two channels | +0.151 (0.173) | +0.152 (0.272) | +0.334 (0.048) | +0.326 (0.063) | 0.694 (0.061) | 0.689 (0.095) | 0.614 (0.024) | 0.614 (0.060) |
| two widths, product with descriptor | +0.817 (0.143) | +0.938 (0.034) | +0.963 (0.016) | +0.974 (0.011) | 0.306 (0.110) | 0.184 (0.048) | 0.143 (0.031) | 0.118 (0.030) |

Paired difference in q2 against *arrangement kernel alone* over the draws (mean, standard deviation in brackets, draws ahead):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---:|---:|---:|---:|
| lower-scale descriptor alone | -0.277 (0.761), 9 of 20 | +0.078 (0.334), 12 of 20 | -0.051 (0.103), 6 of 20 | -0.064 (0.162), 8 of 20 |
| phase fraction + descriptor | +0.674 (0.162), 20 of 20 | +0.737 (0.263), 20 of 20 | +0.578 (0.044), 20 of 20 | +0.583 (0.062), 20 of 20 |
| two-point statistics + descriptor + PCA | +0.680 (0.165), 20 of 20 | +0.726 (0.260), 20 of 20 | +0.583 (0.051), 20 of 20 | +0.598 (0.062), 20 of 20 |
| sum kernel | +0.144 (0.808), 16 of 20 | +0.506 (0.196), 20 of 20 | +0.407 (0.059), 20 of 20 | +0.421 (0.072), 20 of 20 |
| product kernel | +0.653 (0.275), 19 of 20 | +0.790 (0.272), 20 of 20 | +0.632 (0.052), 20 of 20 | +0.654 (0.063), 20 of 20 |
| two widths as two channels | +0.001 (0.007), 10 of 20 | +0.003 (0.023), 13 of 20 | +0.006 (0.015), 18 of 20 | +0.005 (0.004), 19 of 20 |
| two widths, product with descriptor | +0.667 (0.219), 20 of 20 | +0.790 (0.274), 20 of 20 | +0.635 (0.051), 20 of 20 | +0.654 (0.065), 20 of 20 |

**Label: the mean over the directions.**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lower-scale descriptor alone | -0.077 (0.997) | +0.246 (0.164) | +0.293 (0.140) | +0.320 (0.067) | 0.710 (0.237) | 0.622 (0.060) | 0.599 (0.060) | 0.582 (0.033) |
| phase fraction + descriptor | +0.921 (0.070) | +0.949 (0.015) | +0.959 (0.005) | +0.961 (0.007) | 0.192 (0.060) | 0.161 (0.025) | 0.144 (0.009) | 0.140 (0.019) |
| two-point statistics + descriptor + PCA | +0.913 (0.037) | +0.938 (0.030) | +0.959 (0.005) | +0.961 (0.007) | 0.208 (0.039) | 0.176 (0.039) | 0.144 (0.009) | 0.140 (0.019) |
| arrangement kernel alone | +0.172 (0.177) | +0.169 (0.272) | +0.339 (0.053) | +0.331 (0.066) | 0.650 (0.065) | 0.648 (0.090) | 0.580 (0.024) | 0.579 (0.050) |
| sum kernel | +0.443 (0.687) | +0.719 (0.073) | +0.777 (0.025) | +0.778 (0.036) | 0.489 (0.214) | 0.378 (0.049) | 0.337 (0.025) | 0.333 (0.040) |
| product kernel | +0.935 (0.080) | +0.982 (0.018) | +0.986 (0.022) | +0.994 (0.003) | 0.161 (0.093) | 0.089 (0.040) | 0.075 (0.044) | 0.056 (0.013) |
| two widths as two channels | +0.183 (0.163) | +0.217 (0.190) | +0.337 (0.053) | +0.329 (0.062) | 0.646 (0.058) | 0.632 (0.070) | 0.581 (0.024) | 0.580 (0.050) |
| two widths, product with descriptor | +0.930 (0.084) | +0.982 (0.017) | +0.992 (0.005) | +0.994 (0.002) | 0.167 (0.099) | 0.092 (0.038) | 0.064 (0.015) | 0.055 (0.012) |

Paired difference in q2 against *arrangement kernel alone* over the draws (mean, standard deviation in brackets, draws ahead):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---:|---:|---:|---:|
| lower-scale descriptor alone | -0.249 (1.021), 9 of 20 | +0.077 (0.344), 11 of 20 | -0.046 (0.149), 8 of 20 | -0.011 (0.105), 9 of 20 |
| phase fraction + descriptor | +0.750 (0.176), 20 of 20 | +0.780 (0.276), 20 of 20 | +0.621 (0.052), 20 of 20 | +0.630 (0.065), 20 of 20 |
| two-point statistics + descriptor + PCA | +0.741 (0.176), 20 of 20 | +0.769 (0.272), 20 of 20 | +0.621 (0.052), 20 of 20 | +0.630 (0.065), 20 of 20 |
| sum kernel | +0.271 (0.645), 17 of 20 | +0.550 (0.275), 20 of 20 | +0.438 (0.052), 20 of 20 | +0.447 (0.068), 20 of 20 |
| product kernel | +0.764 (0.207), 20 of 20 | +0.813 (0.277), 20 of 20 | +0.647 (0.051), 20 of 20 | +0.662 (0.066), 20 of 20 |
| two widths as two channels | +0.011 (0.059), 7 of 20 | +0.048 (0.242), 9 of 20 | -0.002 (0.006), 7 of 20 | -0.002 (0.009), 7 of 20 |
| two widths, product with descriptor | +0.758 (0.210), 20 of 20 | +0.813 (0.278), 20 of 20 | +0.653 (0.052), 20 of 20 | +0.663 (0.066), 20 of 20 |

## The coefficient map of the linear autocorrelation model, on 2d51_n400_c100

The model y = sum over r of w(r) S(r) + b read back on the lag vector r, fitted on all 400 cells with the ridge chosen by five-fold cross-validation; the figure is `map_2d51_n400_c100.png`. The numbers below are the effect w(r) times the spread of S(r) over the samples, for the phase-1 autocorrelation block, along the x axis and along the y axis of the lag, and the mean over the shell of each lag length.

**k_xx**: ridge 0.01, cross-validated R2 +0.891.

| lag, voxels | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| along x | -0.001 | +0.080 | +0.144 | +0.175 | +0.171 | +0.153 | +0.125 | +0.109 | +0.097 | +0.089 | +0.075 | +0.074 | +0.066 |
| along y | -0.001 | -0.078 | -0.126 | -0.136 | -0.115 | -0.086 | -0.058 | -0.038 | -0.026 | -0.018 | -0.009 | -0.002 | +0.010 |
| shell mean | -0.001 | +0.002 | +0.010 | +0.018 | +0.023 | +0.024 | +0.022 | +0.019 | +0.017 | +0.017 | +0.012 | +0.013 | +0.008 |

**mean k**: ridge 0.1, cross-validated R2 +0.927.

| lag, voxels | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| along x | -0.001 | +0.005 | +0.012 | +0.016 | +0.020 | +0.021 | +0.021 | +0.020 | +0.018 | +0.015 | +0.012 | +0.010 | +0.008 |
| along y | -0.001 | +0.007 | +0.014 | +0.019 | +0.021 | +0.021 | +0.020 | +0.018 | +0.015 | +0.013 | +0.013 | +0.013 | +0.012 |
| shell mean | -0.001 | +0.008 | +0.015 | +0.018 | +0.020 | +0.020 | +0.019 | +0.017 | +0.014 | +0.011 | +0.009 | +0.008 | +0.006 |

## Real micrographs: the UHCS set

UHCSDB (DeCost et al. 2017), Zenodo mirror record 23050464, CC BY 3.0 US. 961 micrographs of 44 specimens; the annealing temperature and time are recorded for 598 of them. Each image is cut above its information bar and reduced 4 times by block means to 161 x 121 pixels; the two-point autocorrelation is computed with zero padding and the count of overlapping pairs as the denominator, lags up to 30 pixels. Smoothing widths {0, 1, 2, 4} pixels. Classes: spheroidite 374, network 212, pearlite 124, pearlite+spheroidite 107, spheroidite+widmanstatten 81, martensite 36, pearlite+widmanstatten 27. Two ways of drawing a training set: images at random (the way most published numbers are made; images of one specimen then sit on both sides), and whole specimens (the test images come from specimens the model never saw). 20 draws per size. Run time 150 min.

**anneal temperature, C, training sets of images at random** (mean training count 20, 50, 100, 200, 500).

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | q2, n_train 500 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 | RMSE, n_train 500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grey mean and spread | -1.247 (1.695) | -0.078 (0.131) | -0.001 (0.052) | +0.019 (0.053) | +0.068 (0.036) | 120.840 (38.912) | 87.794 (4.768) | 84.582 (2.314) | 83.619 (2.195) | 82.334 (5.720) |
| two-point statistics + PCA | -762.578 (2873.132) | -22.549 (55.353) | -0.274 (0.886) | -0.498 (1.330) | -1.293 (4.078) | 860.542 (2234.824) | 238.703 (342.001) | 92.757 (22.833) | 98.414 (32.199) | 108.788 (77.167) |
| two-point statistics + GP | -0.189 (0.273) | -0.117 (0.090) | -0.027 (0.071) | +0.062 (0.055) | +0.205 (0.078) | 91.624 (9.430) | 89.477 (3.872) | 85.639 (2.697) | 81.789 (3.241) | 75.871 (5.236) |
| CNN features + ridge | +0.013 (0.131) | +0.152 (0.109) | +0.283 (0.051) | +0.378 (0.031) | +0.543 (0.055) | 83.749 (5.438) | 77.864 (4.727) | 71.541 (2.703) | 66.617 (2.445) | 57.527 (4.514) |
| autocorrelation, grey, linear | -0.568 (0.824) | -0.154 (0.341) | +0.052 (0.093) | +0.168 (0.059) | +0.301 (0.063) | 103.198 (23.790) | 90.285 (11.362) | 82.235 (3.899) | 77.003 (2.833) | 71.240 (5.455) |
| autocorrelation, grey, normalised | -0.110 (0.249) | +0.064 (0.097) | +0.120 (0.079) | +0.200 (0.045) | +0.322 (0.060) | 88.528 (9.302) | 81.843 (4.158) | 79.216 (3.256) | 75.527 (2.980) | 70.092 (4.939) |
| autocorrelation, grey, radial | -1.528 (2.795) | -0.217 (0.345) | -0.052 (0.163) | +0.082 (0.071) | +0.236 (0.075) | 123.553 (53.383) | 92.654 (11.820) | 86.515 (6.385) | 80.854 (2.240) | 74.450 (5.939) |
| autocorrelation, two-level, linear | -0.356 (0.388) | -0.087 (0.103) | +0.025 (0.081) | +0.082 (0.046) | +0.174 (0.072) | 97.533 (13.275) | 88.242 (4.246) | 83.424 (3.369) | 80.910 (2.178) | 77.418 (5.914) |
| autocorrelation, two-level, radial | -0.523 (0.535) | -0.134 (0.214) | -0.014 (0.151) | +0.064 (0.064) | +0.176 (0.066) | 103.033 (16.145) | 89.852 (8.021) | 84.951 (5.973) | 81.701 (3.291) | 77.383 (5.965) |

Paired difference in q2 against *CNN features + ridge* over the draws (mean, standard deviation in brackets, draws ahead):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 | n_train 500 |
|---|---:|---:|---:|---:|---:|
| grey mean and spread | -1.260 (1.733), 1 of 20 | -0.230 (0.160), 2 of 20 | -0.285 (0.061), 0 of 20 | -0.359 (0.049), 0 of 20 | -0.475 (0.074), 0 of 20 |
| two-point statistics + PCA | -762.592 (2873.187), 0 of 20 | -22.701 (55.293), 0 of 20 | -0.558 (0.882), 0 of 20 | -0.876 (1.334), 0 of 20 | -1.835 (4.072), 0 of 20 |
| two-point statistics + GP | -0.202 (0.276), 2 of 20 | -0.269 (0.125), 0 of 20 | -0.311 (0.070), 0 of 20 | -0.316 (0.060), 0 of 20 | -0.337 (0.070), 0 of 20 |
| autocorrelation, grey, linear | -0.581 (0.810), 1 of 20 | -0.306 (0.357), 1 of 20 | -0.231 (0.084), 0 of 20 | -0.210 (0.052), 0 of 20 | -0.242 (0.083), 0 of 20 |
| autocorrelation, grey, normalised | -0.123 (0.217), 7 of 20 | -0.088 (0.112), 2 of 20 | -0.163 (0.087), 1 of 20 | -0.178 (0.037), 0 of 20 | -0.220 (0.055), 0 of 20 |
| autocorrelation, grey, radial | -1.542 (2.794), 0 of 20 | -0.368 (0.381), 1 of 20 | -0.335 (0.172), 0 of 20 | -0.296 (0.075), 0 of 20 | -0.307 (0.081), 0 of 20 |
| autocorrelation, two-level, linear | -0.369 (0.377), 4 of 20 | -0.239 (0.124), 1 of 20 | -0.258 (0.072), 0 of 20 | -0.296 (0.049), 0 of 20 | -0.369 (0.067), 0 of 20 |
| autocorrelation, two-level, radial | -0.536 (0.562), 0 of 20 | -0.286 (0.233), 1 of 20 | -0.298 (0.127), 0 of 20 | -0.314 (0.055), 0 of 20 | -0.367 (0.072), 0 of 20 |

**anneal temperature, C, training sets of whole specimens** (mean training count 28, 64, 110, 207, 509).

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | q2, n_train 500 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 | RMSE, n_train 500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grey mean and spread | -3.004 (5.347) | -0.675 (0.721) | -0.354 (0.294) | -0.113 (0.193) | +nan (nan) | 148.942 (70.858) | 107.613 (17.757) | 99.473 (7.823) | 87.073 (7.752) | 79.399 (23.023) |
| two-point statistics + PCA | -47.681 (158.252) | -6.519 (16.945) | -16.515 (64.562) | -13.301 (45.535) | +nan (nan) | 329.955 (469.440) | 170.441 (152.066) | 196.411 (305.280) | 181.324 (262.790) | 88.890 (21.654) |
| two-point statistics + GP | -1.162 (1.188) | -0.591 (0.842) | -0.331 (0.218) | -0.177 (0.240) | +nan (nan) | 119.418 (27.327) | 104.388 (19.641) | 98.954 (6.662) | 89.356 (8.297) | 77.990 (18.925) |
| CNN features + ridge | -0.870 (0.913) | -0.436 (0.417) | -0.154 (0.298) | -0.025 (0.165) | +nan (nan) | 111.508 (23.997) | 100.667 (11.644) | 91.825 (9.125) | 83.640 (7.941) | 74.175 (18.672) |
| autocorrelation, grey, linear | -1.522 (1.820) | -0.938 (1.936) | -0.214 (0.334) | -0.090 (0.240) | +nan (nan) | 126.122 (37.366) | 111.153 (40.513) | 94.085 (9.673) | 85.917 (8.742) | 76.162 (17.985) |
| autocorrelation, grey, normalised | -1.051 (1.118) | -0.344 (0.548) | -0.140 (0.288) | +0.017 (0.172) | +nan (nan) | 116.048 (27.286) | 96.736 (14.991) | 91.238 (8.149) | 81.857 (7.830) | 73.497 (19.118) |
| autocorrelation, grey, radial | -2.008 (2.856) | -1.031 (1.597) | -0.335 (0.346) | -0.173 (0.229) | +nan (nan) | 134.956 (51.269) | 114.968 (35.974) | 98.663 (10.486) | 89.195 (7.934) | 79.453 (17.678) |
| autocorrelation, two-level, linear | -1.340 (1.327) | -0.529 (0.445) | -0.274 (0.219) | -0.127 (0.194) | +nan (nan) | 123.451 (29.997) | 103.941 (11.698) | 96.832 (6.983) | 87.537 (6.737) | 81.053 (17.677) |
| autocorrelation, two-level, radial | -1.337 (1.152) | -0.617 (0.567) | -0.268 (0.200) | -0.141 (0.192) | +nan (nan) | 124.209 (27.747) | 106.339 (14.133) | 96.609 (6.147) | 88.165 (7.637) | 80.649 (17.793) |

Paired difference in q2 against *CNN features + ridge* over the draws (mean, standard deviation in brackets, draws ahead):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 | n_train 500 |
|---|---:|---:|---:|---:|---:|
| grey mean and spread | -2.134 (5.131), 5 of 20 | -0.239 (0.655), 7 of 20 | -0.200 (0.271), 3 of 20 | -0.088 (0.112), 4 of 20 | +nan (nan), 5 of 20 |
| two-point statistics + PCA | -46.812 (158.029), 2 of 20 | -6.083 (16.792), 4 of 20 | -16.362 (64.588), 0 of 20 | -13.276 (45.528), 0 of 20 | +nan (nan), 3 of 20 |
| two-point statistics + GP | -0.292 (0.522), 5 of 20 | -0.155 (0.660), 7 of 20 | -0.177 (0.252), 3 of 20 | -0.152 (0.176), 2 of 20 | +nan (nan), 10 of 20 |
| autocorrelation, grey, linear | -0.652 (1.440), 7 of 20 | -0.501 (1.924), 11 of 20 | -0.060 (0.127), 5 of 20 | -0.065 (0.193), 9 of 20 | +nan (nan), 9 of 20 |
| autocorrelation, grey, normalised | -0.182 (0.442), 8 of 20 | +0.093 (0.381), 14 of 20 | +0.014 (0.162), 10 of 20 | +0.041 (0.148), 12 of 20 | +nan (nan), 11 of 20 |
| autocorrelation, grey, radial | -1.138 (2.777), 5 of 20 | -0.595 (1.551), 5 of 20 | -0.181 (0.179), 3 of 20 | -0.148 (0.167), 3 of 20 | +nan (nan), 7 of 20 |
| autocorrelation, two-level, linear | -0.470 (0.933), 4 of 20 | -0.092 (0.421), 8 of 20 | -0.121 (0.208), 5 of 20 | -0.102 (0.158), 4 of 20 | +nan (nan), 5 of 20 |
| autocorrelation, two-level, radial | -0.468 (0.740), 4 of 20 | -0.180 (0.432), 5 of 20 | -0.114 (0.175), 5 of 20 | -0.116 (0.164), 3 of 20 | +nan (nan), 4 of 20 |

**log10 anneal time, h, training sets of images at random** (mean training count 20, 50, 100, 200, 500).

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | q2, n_train 500 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 | RMSE, n_train 500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grey mean and spread | -0.603 (0.480) | -0.009 (0.101) | +0.078 (0.040) | +0.107 (0.027) | +0.118 (0.051) | 1.039 (0.157) | 0.834 (0.038) | 0.800 (0.018) | 0.783 (0.017) | 0.766 (0.048) |
| two-point statistics + PCA | -12.140 (49.656) | -14.766 (62.601) | -0.042 (0.089) | -0.173 (0.543) | -0.120 (0.536) | 1.687 (2.552) | 1.652 (2.948) | 0.850 (0.034) | 0.884 (0.158) | 0.850 (0.173) |
| two-point statistics + GP | -0.100 (0.158) | -0.039 (0.067) | -0.014 (0.061) | +0.003 (0.044) | +0.039 (0.066) | 0.868 (0.059) | 0.847 (0.024) | 0.839 (0.024) | 0.828 (0.020) | 0.799 (0.055) |
| CNN features + ridge | -0.013 (0.144) | +0.156 (0.095) | +0.297 (0.050) | +0.416 (0.029) | +0.528 (0.056) | 0.833 (0.056) | 0.763 (0.041) | 0.699 (0.027) | 0.634 (0.020) | 0.559 (0.043) |
| autocorrelation, grey, linear | -0.864 (1.360) | -0.151 (0.292) | -0.047 (0.094) | +0.037 (0.050) | +0.162 (0.095) | 1.088 (0.329) | 0.886 (0.099) | 0.852 (0.038) | 0.814 (0.024) | 0.746 (0.057) |
| autocorrelation, grey, normalised | -0.261 (0.126) | -0.158 (0.099) | -0.104 (0.058) | -0.017 (0.088) | +0.157 (0.081) | 0.930 (0.045) | 0.894 (0.037) | 0.876 (0.025) | 0.836 (0.038) | 0.747 (0.038) |
| autocorrelation, grey, radial | -2.083 (3.626) | -0.541 (1.293) | -0.003 (0.078) | +0.013 (0.053) | +0.082 (0.131) | 1.305 (0.665) | 0.982 (0.317) | 0.834 (0.032) | 0.824 (0.023) | 0.780 (0.066) |
| autocorrelation, two-level, linear | -0.465 (0.303) | -0.233 (0.180) | -0.083 (0.178) | -0.018 (0.075) | +0.080 (0.096) | 0.999 (0.099) | 0.921 (0.068) | 0.866 (0.065) | 0.836 (0.034) | 0.782 (0.070) |
| autocorrelation, two-level, radial | -0.756 (0.778) | -0.209 (0.182) | -0.033 (0.037) | -0.020 (0.075) | +0.069 (0.081) | 1.079 (0.217) | 0.912 (0.066) | 0.847 (0.016) | 0.837 (0.028) | 0.786 (0.055) |

Paired difference in q2 against *CNN features + ridge* over the draws (mean, standard deviation in brackets, draws ahead):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 | n_train 500 |
|---|---:|---:|---:|---:|---:|
| grey mean and spread | -0.590 (0.513), 1 of 20 | -0.164 (0.131), 3 of 20 | -0.219 (0.046), 0 of 20 | -0.308 (0.035), 0 of 20 | -0.411 (0.071), 0 of 20 |
| two-point statistics + PCA | -12.127 (49.678), 4 of 20 | -14.922 (62.606), 1 of 20 | -0.339 (0.083), 0 of 20 | -0.588 (0.544), 0 of 20 | -0.648 (0.538), 0 of 20 |
| two-point statistics + GP | -0.087 (0.190), 6 of 20 | -0.194 (0.099), 2 of 20 | -0.310 (0.071), 0 of 20 | -0.413 (0.052), 0 of 20 | -0.489 (0.087), 0 of 20 |
| autocorrelation, grey, linear | -0.852 (1.366), 3 of 20 | -0.307 (0.303), 2 of 20 | -0.343 (0.105), 0 of 20 | -0.379 (0.050), 0 of 20 | -0.367 (0.095), 0 of 20 |
| autocorrelation, grey, normalised | -0.248 (0.212), 4 of 20 | -0.314 (0.105), 0 of 20 | -0.400 (0.072), 0 of 20 | -0.433 (0.076), 0 of 20 | -0.371 (0.062), 0 of 20 |
| autocorrelation, grey, radial | -2.071 (3.654), 2 of 20 | -0.697 (1.293), 1 of 20 | -0.300 (0.072), 0 of 20 | -0.403 (0.060), 0 of 20 | -0.447 (0.122), 0 of 20 |
| autocorrelation, two-level, linear | -0.452 (0.280), 0 of 20 | -0.389 (0.202), 0 of 20 | -0.380 (0.176), 0 of 20 | -0.434 (0.075), 0 of 20 | -0.448 (0.094), 0 of 20 |
| autocorrelation, two-level, radial | -0.743 (0.810), 0 of 20 | -0.364 (0.186), 1 of 20 | -0.330 (0.062), 0 of 20 | -0.435 (0.079), 0 of 20 | -0.459 (0.082), 0 of 20 |

**log10 anneal time, h, training sets of whole specimens** (mean training count 28, 64, 110, 207, 509).

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | q2, n_train 500 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 | RMSE, n_train 500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grey mean and spread | -1.271 (1.472) | -0.634 (0.770) | -0.498 (0.693) | -0.211 (0.292) | -0.353 (0.747) | 1.195 (0.342) | 1.029 (0.174) | 0.980 (0.159) | 0.869 (0.100) | 0.780 (0.285) |
| two-point statistics + PCA | -17.372 (44.552) | -1.829 (3.523) | -7.949 (32.480) | -4.545 (18.266) | -0.692 (0.729) | 2.418 (2.574) | 1.266 (0.575) | 1.506 (2.075) | 1.274 (1.447) | 0.866 (0.253) |
| two-point statistics + GP | -0.937 (1.166) | -0.570 (0.561) | -0.546 (0.543) | -0.325 (0.274) | -0.481 (0.546) | 1.112 (0.281) | 1.017 (0.131) | 1.004 (0.120) | 0.910 (0.089) | 0.821 (0.242) |
| CNN features + ridge | -0.775 (1.149) | -0.442 (0.539) | -0.409 (0.588) | -0.187 (0.253) | -0.379 (0.587) | 1.063 (0.280) | 0.975 (0.121) | 0.954 (0.130) | 0.860 (0.076) | 0.774 (0.208) |
| autocorrelation, grey, linear | -1.102 (1.262) | -0.770 (0.694) | -0.492 (0.428) | -0.398 (0.337) | -0.620 (0.660) | 1.156 (0.314) | 1.077 (0.179) | 0.991 (0.091) | 0.931 (0.093) | 0.846 (0.232) |
| autocorrelation, grey, normalised | -0.992 (1.181) | -0.755 (0.782) | -0.690 (0.559) | -0.525 (0.332) | -0.730 (0.629) | 1.128 (0.289) | 1.067 (0.164) | 1.050 (0.112) | 0.975 (0.097) | 0.879 (0.238) |
| autocorrelation, grey, radial | -1.877 (3.034) | -0.812 (0.762) | -0.598 (0.471) | -0.434 (0.426) | -0.665 (0.649) | 1.282 (0.589) | 1.088 (0.193) | 1.024 (0.109) | 0.939 (0.116) | 0.860 (0.237) |
| autocorrelation, two-level, linear | -1.145 (1.224) | -0.597 (0.420) | -0.629 (0.517) | -0.452 (0.351) | -0.731 (0.696) | 1.170 (0.302) | 1.034 (0.100) | 1.033 (0.121) | 0.950 (0.103) | 0.884 (0.261) |
| autocorrelation, two-level, radial | -1.251 (1.278) | -0.641 (0.444) | -0.763 (0.624) | -0.510 (0.434) | -0.723 (0.593) | 1.198 (0.322) | 1.047 (0.097) | 1.071 (0.134) | 0.963 (0.077) | 0.885 (0.247) |

Paired difference in q2 against *CNN features + ridge* over the draws (mean, standard deviation in brackets, draws ahead):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 | n_train 500 |
|---|---:|---:|---:|---:|---:|
| grey mean and spread | -0.496 (1.037), 6 of 20 | -0.192 (0.444), 6 of 20 | -0.088 (0.457), 7 of 20 | -0.024 (0.222), 8 of 20 | +0.026 (0.577), 10 of 20 |
| two-point statistics + PCA | -16.597 (44.426), 3 of 20 | -1.387 (3.521), 3 of 20 | -7.540 (32.543), 7 of 20 | -4.358 (18.253), 2 of 20 | -0.313 (0.574), 5 of 20 |
| two-point statistics + GP | -0.162 (0.439), 6 of 20 | -0.127 (0.306), 4 of 20 | -0.136 (0.394), 4 of 20 | -0.138 (0.208), 8 of 20 | -0.102 (0.395), 7 of 20 |
| autocorrelation, grey, linear | -0.327 (0.778), 7 of 20 | -0.328 (0.549), 4 of 20 | -0.083 (0.314), 6 of 20 | -0.211 (0.201), 2 of 20 | -0.242 (0.302), 5 of 20 |
| autocorrelation, grey, normalised | -0.217 (0.519), 1 of 20 | -0.313 (0.449), 3 of 20 | -0.280 (0.282), 1 of 20 | -0.338 (0.191), 0 of 20 | -0.351 (0.246), 2 of 20 |
| autocorrelation, grey, radial | -1.102 (2.878), 6 of 20 | -0.370 (0.573), 4 of 20 | -0.189 (0.364), 5 of 20 | -0.247 (0.285), 2 of 20 | -0.286 (0.398), 4 of 20 |
| autocorrelation, two-level, linear | -0.369 (0.523), 1 of 20 | -0.155 (0.297), 4 of 20 | -0.220 (0.455), 5 of 20 | -0.265 (0.207), 0 of 20 | -0.353 (0.402), 4 of 20 |
| autocorrelation, two-level, radial | -0.475 (0.787), 3 of 20 | -0.199 (0.341), 5 of 20 | -0.353 (0.454), 3 of 20 | -0.324 (0.334), 1 of 20 | -0.345 (0.481), 4 of 20 |

**microconstituent class, training sets of images at random** (mean training count 20, 50, 100, 200, 500).

| method | accuracy, n_train 20 | accuracy, n_train 50 | accuracy, n_train 100 | accuracy, n_train 200 | accuracy, n_train 500 | balanced acc., n_train 20 | balanced acc., n_train 50 | balanced acc., n_train 100 | balanced acc., n_train 200 | balanced acc., n_train 500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grey mean and spread | 0.352 (0.059) | 0.436 (0.040) | 0.478 (0.016) | 0.477 (0.013) | 0.491 (0.013) | 0.202 (0.026) | 0.223 (0.020) | 0.230 (0.011) | 0.223 (0.012) | 0.225 (0.010) |
| two-point statistics + PCA | 0.394 (0.067) | 0.437 (0.055) | 0.469 (0.028) | 0.510 (0.021) | 0.550 (0.020) | 0.191 (0.038) | 0.225 (0.045) | 0.244 (0.023) | 0.286 (0.017) | 0.311 (0.019) |
| two-point statistics + GP | 0.352 (0.060) | 0.355 (0.052) | 0.399 (0.040) | 0.453 (0.023) | 0.509 (0.028) | 0.160 (0.020) | 0.177 (0.025) | 0.195 (0.031) | 0.226 (0.021) | 0.250 (0.022) |
| CNN features + ridge | 0.647 (0.047) | 0.728 (0.027) | 0.771 (0.017) | 0.811 (0.022) | 0.867 (0.014) | 0.433 (0.055) | 0.516 (0.033) | 0.560 (0.037) | 0.635 (0.037) | 0.747 (0.025) |
| autocorrelation, grey, linear | 0.380 (0.058) | 0.458 (0.041) | 0.510 (0.031) | 0.557 (0.024) | 0.610 (0.018) | 0.209 (0.033) | 0.257 (0.029) | 0.296 (0.032) | 0.329 (0.026) | 0.383 (0.025) |
| autocorrelation, grey, normalised | 0.347 (0.047) | 0.401 (0.043) | 0.478 (0.027) | 0.516 (0.024) | 0.584 (0.017) | 0.210 (0.028) | 0.242 (0.024) | 0.268 (0.028) | 0.305 (0.024) | 0.356 (0.021) |
| autocorrelation, grey, radial | 0.401 (0.051) | 0.453 (0.031) | 0.488 (0.034) | 0.546 (0.025) | 0.591 (0.018) | 0.219 (0.029) | 0.248 (0.019) | 0.275 (0.036) | 0.324 (0.029) | 0.377 (0.024) |
| autocorrelation, two-level, linear | 0.371 (0.062) | 0.440 (0.044) | 0.478 (0.022) | 0.503 (0.018) | 0.564 (0.020) | 0.207 (0.036) | 0.238 (0.032) | 0.259 (0.025) | 0.295 (0.035) | 0.355 (0.025) |
| autocorrelation, two-level, radial | 0.379 (0.056) | 0.438 (0.052) | 0.449 (0.037) | 0.488 (0.023) | 0.542 (0.024) | 0.220 (0.040) | 0.243 (0.030) | 0.248 (0.027) | 0.270 (0.032) | 0.323 (0.031) |

Paired difference in accuracy against *CNN features + ridge* over the draws (mean, standard deviation in brackets, draws ahead):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 | n_train 500 |
|---|---:|---:|---:|---:|---:|
| grey mean and spread | -0.295 (0.074), 0 of 20 | -0.292 (0.045), 0 of 20 | -0.293 (0.022), 0 of 20 | -0.334 (0.028), 0 of 20 | -0.377 (0.019), 0 of 20 |
| two-point statistics + PCA | -0.253 (0.086), 0 of 20 | -0.291 (0.049), 0 of 20 | -0.302 (0.027), 0 of 20 | -0.300 (0.028), 0 of 20 | -0.318 (0.017), 0 of 20 |
| two-point statistics + GP | -0.295 (0.074), 0 of 20 | -0.373 (0.053), 0 of 20 | -0.371 (0.045), 0 of 20 | -0.358 (0.035), 0 of 20 | -0.359 (0.031), 0 of 20 |
| autocorrelation, grey, linear | -0.267 (0.074), 0 of 20 | -0.270 (0.045), 0 of 20 | -0.260 (0.032), 0 of 20 | -0.254 (0.037), 0 of 20 | -0.257 (0.021), 0 of 20 |
| autocorrelation, grey, normalised | -0.300 (0.074), 0 of 20 | -0.327 (0.053), 0 of 20 | -0.292 (0.034), 0 of 20 | -0.295 (0.034), 0 of 20 | -0.284 (0.013), 0 of 20 |
| autocorrelation, grey, radial | -0.246 (0.063), 0 of 20 | -0.275 (0.028), 0 of 20 | -0.283 (0.035), 0 of 20 | -0.265 (0.036), 0 of 20 | -0.277 (0.019), 0 of 20 |
| autocorrelation, two-level, linear | -0.276 (0.068), 0 of 20 | -0.288 (0.051), 0 of 20 | -0.293 (0.030), 0 of 20 | -0.308 (0.030), 0 of 20 | -0.303 (0.017), 0 of 20 |
| autocorrelation, two-level, radial | -0.268 (0.074), 0 of 20 | -0.290 (0.057), 0 of 20 | -0.322 (0.037), 0 of 20 | -0.323 (0.034), 0 of 20 | -0.325 (0.021), 0 of 20 |

**microconstituent class, training sets of whole specimens** (mean training count 46, 91, 130, 226, 547).

| method | accuracy, n_train 20 | accuracy, n_train 50 | accuracy, n_train 100 | accuracy, n_train 200 | accuracy, n_train 500 | balanced acc., n_train 20 | balanced acc., n_train 50 | balanced acc., n_train 100 | balanced acc., n_train 200 | balanced acc., n_train 500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grey mean and spread | 0.293 (0.094) | 0.310 (0.083) | 0.344 (0.104) | 0.386 (0.082) | 0.462 (0.067) | 0.170 (0.032) | 0.198 (0.032) | 0.196 (0.034) | 0.213 (0.021) | 0.243 (0.049) |
| two-point statistics + PCA | 0.317 (0.100) | 0.350 (0.098) | 0.376 (0.092) | 0.419 (0.067) | 0.516 (0.067) | 0.172 (0.027) | 0.211 (0.027) | 0.219 (0.036) | 0.237 (0.035) | 0.303 (0.057) |
| two-point statistics + GP | 0.305 (0.096) | 0.311 (0.073) | 0.332 (0.073) | 0.347 (0.066) | 0.483 (0.064) | 0.167 (0.035) | 0.185 (0.043) | 0.192 (0.036) | 0.197 (0.030) | 0.267 (0.051) |
| CNN features + ridge | 0.448 (0.114) | 0.541 (0.064) | 0.582 (0.101) | 0.665 (0.056) | 0.748 (0.043) | 0.274 (0.068) | 0.373 (0.083) | 0.408 (0.073) | 0.471 (0.055) | 0.551 (0.056) |
| autocorrelation, grey, linear | 0.323 (0.095) | 0.372 (0.071) | 0.373 (0.094) | 0.428 (0.065) | 0.548 (0.052) | 0.183 (0.043) | 0.234 (0.048) | 0.235 (0.049) | 0.263 (0.042) | 0.359 (0.047) |
| autocorrelation, grey, normalised | 0.309 (0.081) | 0.336 (0.073) | 0.377 (0.081) | 0.400 (0.058) | 0.480 (0.039) | 0.190 (0.050) | 0.216 (0.039) | 0.239 (0.037) | 0.255 (0.040) | 0.302 (0.046) |
| autocorrelation, grey, radial | 0.306 (0.101) | 0.359 (0.080) | 0.372 (0.088) | 0.423 (0.062) | 0.525 (0.055) | 0.185 (0.046) | 0.228 (0.043) | 0.231 (0.036) | 0.259 (0.034) | 0.314 (0.058) |
| autocorrelation, two-level, linear | 0.311 (0.097) | 0.341 (0.088) | 0.358 (0.066) | 0.395 (0.038) | 0.469 (0.045) | 0.174 (0.041) | 0.216 (0.055) | 0.218 (0.040) | 0.239 (0.045) | 0.287 (0.051) |
| autocorrelation, two-level, radial | 0.307 (0.091) | 0.334 (0.089) | 0.346 (0.069) | 0.386 (0.057) | 0.461 (0.032) | 0.179 (0.039) | 0.212 (0.047) | 0.216 (0.036) | 0.241 (0.043) | 0.280 (0.047) |

Paired difference in accuracy against *CNN features + ridge* over the draws (mean, standard deviation in brackets, draws ahead):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 | n_train 500 |
|---|---:|---:|---:|---:|---:|
| grey mean and spread | -0.155 (0.102), 2 of 20 | -0.231 (0.080), 0 of 20 | -0.237 (0.086), 0 of 20 | -0.279 (0.076), 0 of 20 | -0.286 (0.044), 0 of 20 |
| two-point statistics + PCA | -0.131 (0.074), 2 of 20 | -0.191 (0.085), 0 of 20 | -0.206 (0.066), 0 of 20 | -0.247 (0.074), 0 of 20 | -0.232 (0.038), 0 of 20 |
| two-point statistics + GP | -0.143 (0.065), 1 of 20 | -0.230 (0.072), 0 of 20 | -0.250 (0.068), 0 of 20 | -0.318 (0.067), 0 of 20 | -0.265 (0.047), 0 of 20 |
| autocorrelation, grey, linear | -0.125 (0.063), 0 of 20 | -0.169 (0.047), 0 of 20 | -0.209 (0.059), 0 of 20 | -0.237 (0.069), 0 of 20 | -0.200 (0.036), 0 of 20 |
| autocorrelation, grey, normalised | -0.139 (0.074), 0 of 20 | -0.204 (0.069), 0 of 20 | -0.205 (0.073), 0 of 20 | -0.266 (0.064), 0 of 20 | -0.268 (0.042), 0 of 20 |
| autocorrelation, grey, radial | -0.142 (0.079), 0 of 20 | -0.182 (0.058), 0 of 20 | -0.210 (0.062), 0 of 20 | -0.242 (0.072), 0 of 20 | -0.223 (0.034), 0 of 20 |
| autocorrelation, two-level, linear | -0.137 (0.070), 0 of 20 | -0.200 (0.071), 0 of 20 | -0.224 (0.069), 0 of 20 | -0.270 (0.044), 0 of 20 | -0.279 (0.037), 0 of 20 |
| autocorrelation, two-level, radial | -0.141 (0.068), 1 of 20 | -0.206 (0.078), 0 of 20 | -0.236 (0.072), 0 of 20 | -0.279 (0.066), 0 of 20 | -0.287 (0.041), 0 of 20 |

# The pilot: five draws, 140 settings per kernel method

## 2D, 51 points per side, 400 samples, contrast 10

Labels: k_xx from 1.29 to 7.30 (sd 1.555), mean from 1.51 to 6.83 (sd 1.437). Solver checks: uniform phase 1: largest error 0.0e+00; laminate, layers across x: largest error 1.2e-13; random structure, asymmetry of the tensor 2.2e-16. Settings searched: smoothing width in {0, 1, 2, 3, 5} voxels; layer in {linear, gauss 0.5, gauss 1, gauss 2} (the Gaussian bandwidth is in units of the median squared distance); ridge in {1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1} times the mean self-kernel (140 settings per kernel method); for the polynomial baseline {2, 3, 5, 10, 20} components, degree {1, 2}, ridge {1e-08, 0.0001, 0.01, 1} (40 settings); for the GP baseline the components alone, with the length scales, the amplitude and the noise fitted by the marginal likelihood. 5 draws per training size. Run time 8 min.

**Label: the conductivity along x (direction-dependent); from 1.29 to 7.30, sd 1.555.**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.867 (0.028) | +0.875 (0.008) | +0.884 (0.008) | +0.889 (0.009) | 0.565 (0.059) | 0.545 (0.015) | 0.531 (0.017) | 0.522 (0.026) |
| two-point statistics + PCA | +0.930 (0.014) | +0.950 (0.009) | +0.965 (0.003) | +0.968 (0.003) | 0.408 (0.042) | 0.343 (0.031) | 0.291 (0.010) | 0.279 (0.012) |
| overlap, generating frame | +0.812 (0.042) | +0.845 (0.021) | +0.879 (0.013) | +0.898 (0.017) | 0.670 (0.070) | 0.607 (0.043) | 0.542 (0.027) | 0.499 (0.046) |
| overlap, rotation-averaged | +0.807 (0.028) | +0.823 (0.030) | +0.862 (0.006) | +0.870 (0.012) | 0.681 (0.054) | 0.648 (0.053) | 0.581 (0.018) | 0.564 (0.031) |
| autocorrelation, linear | +0.890 (0.042) | +0.927 (0.062) | +0.971 (0.003) | +0.979 (0.002) | 0.507 (0.091) | 0.395 (0.153) | 0.265 (0.014) | 0.228 (0.014) |
| autocorrelation, normalised | +0.912 (0.024) | +0.931 (0.055) | +0.972 (0.002) | +0.979 (0.002) | 0.456 (0.058) | 0.387 (0.142) | 0.261 (0.009) | 0.229 (0.012) |
| autocorrelation, lattice-rotation-averaged | +0.868 (0.034) | +0.826 (0.122) | +0.890 (0.009) | +0.893 (0.011) | 0.561 (0.070) | 0.619 (0.192) | 0.517 (0.019) | 0.511 (0.026) |
| autocorrelation, radial | +0.864 (0.043) | +0.836 (0.100) | +0.891 (0.009) | +0.894 (0.011) | 0.567 (0.086) | 0.606 (0.164) | 0.515 (0.018) | 0.511 (0.026) |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (5 draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 5 (5) | 2 (1), 5 (4) | 2 (1), 5 (4) | 2 (1), 3 (1), 5 (3) |
| overlap, rotation-averaged | 1 (1), 2 (1), 5 (3) | 0 (3), 3 (1), 5 (1) | 1 (1), 5 (4) | 0 (2), 1 (2), 5 (1) |
| autocorrelation, linear | 0 (1), 2 (1), 3 (1), 5 (2) | 0 (5) | 0 (5) | 0 (5) |
| autocorrelation, normalised | 0 (2), 2 (1), 5 (2) | 0 (5) | 0 (5) | 0 (5) |
| autocorrelation, lattice-rotation-averaged | 3 (1), 5 (4) | 2 (1), 5 (4) | 3 (1), 5 (4) | 0 (1), 2 (1), 5 (3) |
| autocorrelation, radial | 0 (1), 3 (1), 5 (3) | 5 (5) | 0 (1), 3 (2), 5 (2) | 0 (1), 1 (1), 5 (3) |

**Label: the conductivity, mean over the directions (direction-free); from 1.51 to 6.83, sd 1.437.**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.982 (0.001) | +0.983 (0.001) | +0.984 (0.001) | +0.985 (0.002) | 0.194 (0.005) | 0.185 (0.006) | 0.183 (0.006) | 0.175 (0.013) |
| two-point statistics + PCA | +0.980 (0.003) | +0.982 (0.002) | +0.984 (0.002) | +0.986 (0.002) | 0.202 (0.014) | 0.192 (0.011) | 0.179 (0.009) | 0.170 (0.009) |
| overlap, generating frame | +0.931 (0.024) | +0.959 (0.006) | +0.965 (0.004) | +0.973 (0.003) | 0.374 (0.066) | 0.288 (0.019) | 0.268 (0.014) | 0.238 (0.012) |
| overlap, rotation-averaged | +0.934 (0.027) | +0.967 (0.009) | +0.974 (0.004) | +0.982 (0.001) | 0.361 (0.078) | 0.258 (0.033) | 0.230 (0.014) | 0.193 (0.002) |
| autocorrelation, linear | +0.980 (0.002) | +0.974 (0.012) | +0.984 (0.002) | +0.987 (0.002) | 0.205 (0.010) | 0.224 (0.048) | 0.183 (0.008) | 0.166 (0.009) |
| autocorrelation, normalised | +0.971 (0.011) | +0.973 (0.010) | +0.984 (0.001) | +0.986 (0.001) | 0.240 (0.042) | 0.232 (0.042) | 0.182 (0.006) | 0.170 (0.008) |
| autocorrelation, lattice-rotation-averaged | +0.976 (0.008) | +0.979 (0.005) | +0.984 (0.002) | +0.986 (0.002) | 0.223 (0.035) | 0.205 (0.023) | 0.181 (0.011) | 0.169 (0.009) |
| autocorrelation, radial | +0.968 (0.017) | +0.976 (0.009) | +0.984 (0.002) | +0.986 (0.002) | 0.249 (0.067) | 0.218 (0.040) | 0.182 (0.011) | 0.169 (0.009) |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (5 draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 5 (5) | 5 (5) | 2 (3), 5 (2) | 1 (1), 2 (2), 5 (2) |
| overlap, rotation-averaged | 1 (1), 3 (1), 5 (3) | 5 (5) | 3 (1), 5 (4) | 5 (5) |
| autocorrelation, linear | 3 (2), 5 (3) | 0 (2), 2 (1), 5 (2) | 0 (1), 1 (1), 3 (1), 5 (2) | 0 (2), 2 (1), 3 (1), 5 (1) |
| autocorrelation, normalised | 3 (1), 5 (4) | 0 (2), 5 (3) | 3 (1), 5 (4) | 2 (1), 3 (1), 5 (3) |
| autocorrelation, lattice-rotation-averaged | 0 (1), 5 (4) | 3 (2), 5 (3) | 0 (1), 5 (4) | 0 (2), 5 (3) |
| autocorrelation, radial | 2 (1), 3 (2), 5 (2) | 0 (1), 5 (4) | 3 (1), 5 (4) | 0 (2), 3 (1), 5 (2) |

## 2D, 51 points per side, 400 samples, contrast 100

Labels: k_xx from 1.33 to 68.14 (sd 17.604), mean from 1.71 to 61.39 (sd 15.527). Solver checks: uniform phase 1: largest error 0.0e+00; laminate, layers across x: largest error 7.6e-13; random structure, asymmetry of the tensor 1.3e-15. Settings searched: smoothing width in {0, 1, 2, 3, 5} voxels; layer in {linear, gauss 0.5, gauss 1, gauss 2} (the Gaussian bandwidth is in units of the median squared distance); ridge in {1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1} times the mean self-kernel (140 settings per kernel method); for the polynomial baseline {2, 3, 5, 10, 20} components, degree {1, 2}, ridge {1e-08, 0.0001, 0.01, 1} (40 settings); for the GP baseline the components alone, with the length scales, the amplitude and the noise fitted by the marginal likelihood. 5 draws per training size. Run time 7 min.

**Label: the conductivity along x (direction-dependent); from 1.33 to 68.14, sd 17.604.**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.739 (0.048) | +0.744 (0.023) | +0.766 (0.017) | +0.779 (0.020) | 8.938 (0.832) | 8.822 (0.381) | 8.568 (0.352) | 8.384 (0.464) |
| two-point statistics + PCA | +0.805 (0.039) | +0.788 (0.063) | +0.876 (0.012) | +0.883 (0.009) | 7.701 (0.729) | 7.979 (1.222) | 6.218 (0.347) | 6.104 (0.242) |
| overlap, generating frame | +0.657 (0.042) | +0.694 (0.034) | +0.756 (0.020) | +0.795 (0.034) | 10.250 (0.640) | 9.653 (0.627) | 8.737 (0.366) | 8.059 (0.738) |
| overlap, rotation-averaged | +0.614 (0.069) | +0.635 (0.076) | +0.736 (0.021) | +0.754 (0.025) | 10.852 (1.018) | 10.506 (1.022) | 9.092 (0.423) | 8.840 (0.559) |
| autocorrelation, linear | +0.811 (0.028) | +0.827 (0.028) | +0.893 (0.008) | +0.902 (0.014) | 7.600 (0.582) | 7.245 (0.632) | 5.787 (0.230) | 5.562 (0.498) |
| autocorrelation, normalised | +0.787 (0.056) | +0.821 (0.068) | +0.891 (0.016) | +0.907 (0.006) | 8.031 (1.140) | 7.306 (1.381) | 5.840 (0.491) | 5.450 (0.282) |
| autocorrelation, lattice-rotation-averaged | +0.713 (0.061) | +0.621 (0.325) | +0.785 (0.019) | +0.793 (0.025) | 9.348 (0.991) | 10.164 (3.785) | 8.210 (0.362) | 8.093 (0.492) |
| autocorrelation, radial | +0.677 (0.079) | +0.651 (0.258) | +0.785 (0.018) | +0.788 (0.029) | 9.902 (1.225) | 9.879 (3.169) | 8.200 (0.368) | 8.194 (0.526) |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (5 draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 5 (5) | 3 (1), 5 (4) | 2 (1), 5 (4) | 3 (1), 5 (4) |
| overlap, rotation-averaged | 2 (3), 5 (2) | 0 (2), 3 (2), 5 (1) | 5 (5) | 0 (1), 2 (1), 5 (3) |
| autocorrelation, linear | 0 (2), 1 (1), 5 (2) | 0 (4), 2 (1) | 0 (4), 1 (1) | 0 (5) |
| autocorrelation, normalised | 0 (2), 1 (1), 3 (1), 5 (1) | 0 (4), 5 (1) | 0 (4), 1 (1) | 0 (5) |
| autocorrelation, lattice-rotation-averaged | 3 (2), 5 (3) | 0 (1), 2 (1), 5 (3) | 3 (1), 5 (4) | 0 (1), 2 (1), 5 (3) |
| autocorrelation, radial | 1 (1), 2 (1), 5 (3) | 3 (1), 5 (4) | 5 (5) | 0 (1), 2 (1), 3 (1), 5 (2) |

**Label: the conductivity, mean over the directions (direction-free); from 1.71 to 61.39, sd 15.527.**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.905 (0.004) | +0.908 (0.012) | +0.914 (0.006) | +0.921 (0.008) | 4.779 (0.083) | 4.636 (0.270) | 4.533 (0.152) | 4.351 (0.196) |
| two-point statistics + PCA | +0.885 (0.024) | +0.899 (0.015) | +0.921 (0.007) | +0.928 (0.006) | 5.239 (0.508) | 4.854 (0.352) | 4.347 (0.177) | 4.158 (0.148) |
| overlap, generating frame | +0.813 (0.051) | +0.865 (0.020) | +0.883 (0.006) | +0.895 (0.008) | 6.669 (0.935) | 5.633 (0.416) | 5.294 (0.192) | 5.020 (0.130) |
| overlap, rotation-averaged | +0.784 (0.080) | +0.864 (0.046) | +0.884 (0.028) | +0.915 (0.009) | 7.120 (1.287) | 5.615 (0.902) | 5.221 (0.547) | 4.529 (0.080) |
| autocorrelation, linear | +0.898 (0.011) | +0.890 (0.033) | +0.901 (0.025) | +0.929 (0.010) | 4.946 (0.276) | 5.036 (0.709) | 4.822 (0.645) | 4.130 (0.170) |
| autocorrelation, normalised | +0.863 (0.034) | +0.880 (0.044) | +0.902 (0.023) | +0.930 (0.010) | 5.728 (0.728) | 5.250 (0.916) | 4.815 (0.578) | 4.111 (0.186) |
| autocorrelation, lattice-rotation-averaged | +0.896 (0.012) | +0.877 (0.045) | +0.910 (0.021) | +0.930 (0.014) | 5.012 (0.300) | 5.323 (0.959) | 4.620 (0.584) | 4.072 (0.262) |
| autocorrelation, radial | +0.885 (0.028) | +0.847 (0.097) | +0.918 (0.011) | +0.931 (0.015) | 5.244 (0.643) | 5.802 (1.756) | 4.422 (0.265) | 4.066 (0.270) |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (5 draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 5 (5) | 5 (5) | 5 (5) | 2 (1), 5 (4) |
| overlap, rotation-averaged | 1 (1), 3 (1), 5 (3) | 3 (1), 5 (4) | 3 (1), 5 (4) | 5 (5) |
| autocorrelation, linear | 0 (1), 1 (2), 3 (1), 5 (1) | 0 (1), 5 (4) | 2 (3), 5 (2) | 0 (2), 1 (1), 3 (2) |
| autocorrelation, normalised | 3 (3), 5 (2) | 0 (2), 2 (1), 5 (2) | 2 (2), 3 (1), 5 (2) | 1 (1), 3 (3), 5 (1) |
| autocorrelation, lattice-rotation-averaged | 3 (2), 5 (3) | 2 (1), 5 (4) | 2 (1), 3 (1), 5 (3) | 0 (1), 5 (4) |
| autocorrelation, radial | 5 (5) | 0 (1), 5 (4) | 1 (1), 3 (1), 5 (3) | 2 (1), 3 (1), 5 (3) |

## 3D, 21 points per side, 400 samples, contrast 10

Labels: k_xx from 1.34 to 8.03 (sd 1.679), mean from 1.68 to 7.31 (sd 1.524). Solver checks: uniform phase 1: largest error 0.0e+00; laminate, layers across x: largest error 1.5e-14; random structure, asymmetry of the tensor 1.1e-16. Settings searched: smoothing width in {0, 1, 2, 3, 5} voxels; layer in {linear, gauss 0.5, gauss 1, gauss 2} (the Gaussian bandwidth is in units of the median squared distance); ridge in {1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1} times the mean self-kernel (140 settings per kernel method); for the polynomial baseline {2, 3, 5, 10, 20} components, degree {1, 2}, ridge {1e-08, 0.0001, 0.01, 1} (40 settings); for the GP baseline the components alone, with the length scales, the amplitude and the noise fitted by the marginal likelihood. 5 draws per training size. Run time 3 min.

**Label: the conductivity along x (direction-dependent); from 1.34 to 8.03, sd 1.679.**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.770 (0.043) | +0.805 (0.008) | +0.823 (0.014) | +0.830 (0.010) | 0.803 (0.079) | 0.738 (0.022) | 0.699 (0.019) | 0.695 (0.023) |
| two-point statistics + PCA | +0.959 (0.014) | +0.976 (0.004) | +0.984 (0.001) | +0.987 (0.003) | 0.338 (0.060) | 0.257 (0.021) | 0.209 (0.010) | 0.189 (0.020) |
| overlap, generating frame | +0.746 (0.051) | +0.837 (0.027) | +0.931 (0.013) | +0.959 (0.005) | 0.843 (0.084) | 0.672 (0.052) | 0.434 (0.044) | 0.343 (0.018) |
| overlap, rotation-averaged | +0.721 (0.075) | +0.793 (0.012) | +0.810 (0.011) | +0.817 (0.016) | 0.881 (0.114) | 0.759 (0.028) | 0.723 (0.015) | 0.722 (0.037) |
| autocorrelation, linear | +0.968 (0.010) | +0.979 (0.004) | +0.984 (0.003) | +0.989 (0.002) | 0.299 (0.045) | 0.244 (0.018) | 0.207 (0.014) | 0.179 (0.014) |
| autocorrelation, normalised | +0.948 (0.021) | +0.969 (0.011) | +0.980 (0.006) | +0.989 (0.001) | 0.379 (0.080) | 0.290 (0.054) | 0.231 (0.034) | 0.174 (0.008) |
| autocorrelation, lattice-rotation-averaged | +0.750 (0.088) | +0.783 (0.050) | +0.801 (0.048) | +0.825 (0.011) | 0.830 (0.149) | 0.775 (0.095) | 0.736 (0.074) | 0.706 (0.023) |
| autocorrelation, radial | +0.749 (0.092) | +0.783 (0.050) | +0.819 (0.014) | +0.822 (0.016) | 0.832 (0.154) | 0.774 (0.095) | 0.706 (0.024) | 0.711 (0.032) |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (5 draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 3 (2), 5 (3) | 5 (5) | 3 (3), 5 (2) | 3 (2), 5 (3) |
| overlap, rotation-averaged | 0 (1), 3 (1), 5 (3) | 5 (5) | 5 (5) | 5 (5) |
| autocorrelation, linear | 0 (4), 3 (1) | 0 (4), 3 (1) | 0 (4), 1 (1) | 0 (4), 1 (1) |
| autocorrelation, normalised | 0 (1), 1 (2), 5 (2) | 0 (3), 2 (1), 3 (1) | 0 (3), 1 (1), 5 (1) | 0 (3), 1 (2) |
| autocorrelation, lattice-rotation-averaged | 2 (1), 3 (2), 5 (2) | 0 (1), 3 (1), 5 (3) | 0 (1), 3 (1), 5 (3) | 0 (1), 2 (1), 5 (3) |
| autocorrelation, radial | 1 (1), 3 (2), 5 (2) | 0 (2), 5 (3) | 0 (1), 2 (1), 5 (3) | 0 (1), 1 (1), 2 (1), 5 (2) |

**Label: the conductivity, mean over the directions (direction-free); from 1.68 to 7.31, sd 1.524.**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.991 (0.001) | +0.991 (0.000) | +0.992 (0.000) | +0.992 (0.000) | 0.147 (0.009) | 0.142 (0.003) | 0.138 (0.003) | 0.135 (0.003) |
| two-point statistics + PCA | +0.992 (0.001) | +0.992 (0.001) | +0.994 (0.000) | +0.995 (0.000) | 0.132 (0.010) | 0.137 (0.009) | 0.115 (0.004) | 0.107 (0.003) |
| overlap, generating frame | +0.974 (0.008) | +0.986 (0.001) | +0.990 (0.001) | +0.991 (0.001) | 0.246 (0.039) | 0.176 (0.006) | 0.151 (0.005) | 0.140 (0.007) |
| overlap, rotation-averaged | +0.969 (0.025) | +0.985 (0.010) | +0.990 (0.002) | +0.992 (0.000) | 0.257 (0.095) | 0.177 (0.051) | 0.149 (0.015) | 0.135 (0.005) |
| autocorrelation, linear | +0.993 (0.002) | +0.993 (0.002) | +0.994 (0.000) | +0.996 (0.001) | 0.127 (0.015) | 0.121 (0.015) | 0.113 (0.003) | 0.101 (0.009) |
| autocorrelation, normalised | +0.992 (0.001) | +0.993 (0.001) | +0.994 (0.000) | +0.995 (0.001) | 0.139 (0.011) | 0.124 (0.008) | 0.115 (0.004) | 0.102 (0.006) |
| autocorrelation, lattice-rotation-averaged | +0.993 (0.002) | +0.993 (0.002) | +0.995 (0.001) | +0.996 (0.000) | 0.127 (0.014) | 0.121 (0.018) | 0.105 (0.009) | 0.093 (0.006) |
| autocorrelation, radial | +0.994 (0.001) | +0.994 (0.001) | +0.994 (0.001) | +0.996 (0.001) | 0.120 (0.014) | 0.112 (0.006) | 0.113 (0.007) | 0.095 (0.007) |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (5 draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 5 (5) | 5 (5) | 5 (5) | 5 (5) |
| overlap, rotation-averaged | 5 (5) | 5 (5) | 3 (1), 5 (4) | 5 (5) |
| autocorrelation, linear | 1 (1), 2 (2), 3 (1), 5 (1) | 2 (2), 5 (3) | 0 (2), 1 (1), 2 (1), 5 (1) | 0 (2), 2 (2), 3 (1) |
| autocorrelation, normalised | 0 (1), 2 (1), 3 (1), 5 (2) | 2 (2), 3 (3) | 1 (1), 2 (1), 3 (3) | 3 (5) |
| autocorrelation, lattice-rotation-averaged | 0 (1), 2 (1), 3 (3) | 0 (2), 1 (1), 5 (2) | 0 (4), 3 (1) | 1 (4), 2 (1) |
| autocorrelation, radial | 0 (2), 2 (1), 3 (2) | 0 (3), 2 (1), 5 (1) | 0 (1), 2 (2), 3 (1), 5 (1) | 0 (4), 1 (1) |
