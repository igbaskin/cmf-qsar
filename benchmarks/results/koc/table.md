Measured soil sorption coefficients, logKoc (Koc in L/kg): the PHYSPROP set curated for OPERA, its training part (536 compounds, spread 1.25) and test part (183 compounds, spread 1.14). Five folds repeated three times on the training part and on all 719 compounds; the external test fits on the whole training part. The setting and the ridge are chosen by the evidence on the training block. Each cell gives q² (cross-validation) or R² (external test), with the RMSE in brackets.

| Model | Training part, CV | All compounds, CV | Test part (the OPERA test) |
|---|---:|---:|---:|
| invariant, cosine | +0.753 (0.62) | +0.769 (0.59) | +0.792 (0.52) |
| invariant, not normalised | +0.735 (0.64) | +0.769 (0.59) | +0.798 (0.51) |
| invariant + clogP | +0.787 (0.58) | +0.787 (0.56) | +0.812 (0.50) |
| linear regression on clogP | +0.712 (0.67) | +0.706 (0.66) | +0.683 (0.64) |
| field totals | +0.752 (0.62) | +0.752 (0.61) | +0.760 (0.56) |
| eight counts | +0.761 (0.61) | +0.772 (0.58) | +0.732 (0.59) |
| ECFP4 | +0.671 (0.72) | +0.686 (0.69) | +0.646 (0.68) |
| refitted Crippen increments | +0.757 (0.61) | +0.768 (0.59) | +0.774 (0.54) |
| topological autocorrelation | +0.728 (0.65) | +0.738 (0.63) | +0.727 (0.60) |
| topological + clogP | +0.761 (0.61) | +0.764 (0.59) | +0.758 (0.56) |
| invariant + topological | +0.753 (0.62) | +0.770 (0.59) | +0.794 (0.52) |
| invariant + eight counts | +0.802 (0.55) | +0.806 (0.54) | +0.805 (0.51) |
| OPERA model, 12 descriptors, as its report gives it (545 and 184 compounds) | +0.810 (0.55) | -- | +0.710 (0.61) |

| Difference of q² or R² | Test | Mean [95 % interval] |
|---|---|---|
| invariant, cosine minus linear regression on clogP | Training part, CV | +0.053 [+0.017, +0.091] |
| invariant, cosine minus linear regression on clogP | All compounds, CV | +0.080 [+0.047, +0.113] |
| invariant, cosine minus linear regression on clogP | Test part (the OPERA test) | +0.109 [+0.039, +0.192] |
| invariant + clogP minus linear regression on clogP | Training part, CV | +0.081 [+0.058, +0.107] |
| invariant + clogP minus linear regression on clogP | All compounds, CV | +0.095 [+0.070, +0.122] |
| invariant + clogP minus linear regression on clogP | Test part (the OPERA test) | +0.129 [+0.085, +0.184] |
| invariant + clogP minus invariant, cosine | Training part, CV | +0.028 [+0.007, +0.048] |
| invariant + clogP minus invariant, cosine | All compounds, CV | +0.015 [-0.000, +0.028] |
| invariant + clogP minus invariant, cosine | Test part (the OPERA test) | +0.020 [-0.021, +0.061] |
| invariant, not normalised minus invariant, cosine | Training part, CV | -0.017 [-0.050, +0.013] |
| invariant, not normalised minus invariant, cosine | All compounds, CV | -0.006 [-0.030, +0.019] |
| invariant, not normalised minus invariant, cosine | Test part (the OPERA test) | +0.006 [-0.046, +0.056] |
| invariant, cosine minus field totals | Training part, CV | +0.009 [-0.019, +0.039] |
| invariant, cosine minus field totals | All compounds, CV | +0.031 [+0.005, +0.057] |
| invariant, cosine minus field totals | Test part (the OPERA test) | +0.032 [-0.018, +0.086] |
| invariant, cosine minus eight counts | Training part, CV | -0.011 [-0.047, +0.023] |
| invariant, cosine minus eight counts | All compounds, CV | +0.001 [-0.029, +0.029] |
| invariant, cosine minus eight counts | Test part (the OPERA test) | +0.060 [-0.003, +0.129] |
| invariant, cosine minus ECFP4 | Training part, CV | +0.086 [+0.040, +0.130] |
| invariant, cosine minus ECFP4 | All compounds, CV | +0.093 [+0.061, +0.125] |
| invariant, cosine minus ECFP4 | Test part (the OPERA test) | +0.146 [+0.074, +0.219] |
| invariant, cosine minus refitted Crippen increments | Training part, CV | +0.002 [-0.039, +0.048] |
| invariant, cosine minus refitted Crippen increments | All compounds, CV | +0.014 [-0.018, +0.057] |
| invariant, cosine minus refitted Crippen increments | Test part (the OPERA test) | +0.018 [-0.038, +0.080] |
| invariant, cosine minus topological autocorrelation | Training part, CV | +0.030 [+0.005, +0.058] |
| invariant, cosine minus topological autocorrelation | All compounds, CV | +0.042 [+0.018, +0.067] |
| invariant, cosine minus topological autocorrelation | Test part (the OPERA test) | +0.065 [+0.012, +0.121] |
| invariant + clogP minus topological + clogP | Training part, CV | +0.030 [+0.014, +0.047] |
| invariant + clogP minus topological + clogP | All compounds, CV | +0.035 [+0.015, +0.054] |
| invariant + clogP minus topological + clogP | Test part (the OPERA test) | +0.054 [+0.027, +0.085] |
| invariant + topological minus invariant, cosine | Training part, CV | +0.000 [-0.004, +0.005] |
| invariant + topological minus invariant, cosine | All compounds, CV | -0.000 [-0.009, +0.009] |
| invariant + topological minus invariant, cosine | Test part (the OPERA test) | +0.002 [-0.008, +0.012] |
| invariant + eight counts minus invariant, cosine | Training part, CV | +0.050 [+0.026, +0.075] |
| invariant + eight counts minus invariant, cosine | All compounds, CV | +0.035 [+0.017, +0.053] |
| invariant + eight counts minus invariant, cosine | Test part (the OPERA test) | +0.013 [-0.026, +0.060] |
| invariant + eight counts minus eight counts | Training part, CV | +0.038 [+0.022, +0.059] |
| invariant + eight counts minus eight counts | All compounds, CV | +0.036 [+0.018, +0.055] |
| invariant + eight counts minus eight counts | Test part (the OPERA test) | +0.073 [+0.039, +0.117] |
