# Covariate shift on the Sutherland sets

8 sets, 20 splits per set and scenario. Each molecule goes to the test part with a probability that depends on its structure only, so the activity follows the same rule in both parts and the true density ratio is known (the *oracle*). `neighbours` holds out the 30 % of molecules nearest to one molecule and has no oracle.

Coverage counts an infinite interval as covering; `inf` is the share of infinite intervals; widths are means over finite intervals, divided by the standard deviation of the activity (mean 1.36).

## How strong is each shift

| scenario | MMD z | MMD p < 0.05 | ESS true | ESS uLSIF | ESS KMM | ESS KLR | rank corr. uLSIF with true | KMM with true | KLR with true | MMD^2 left after KMM | after uLSIF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | -0.1 | 3% | 100% | 83% | 39% | 95% | +nan | +nan | +nan | 37% | 113% |
| pc1-1 | 10.2 | 84% | 51% | 58% | 28% | 63% | +0.52 | +0.46 | +0.74 | 22% | 62% |
| pc1-2 | 22.3 | 100% | 30% | 41% | 22% | 37% | +0.60 | +0.59 | +0.82 | 25% | 52% |
| size-1 | 3.3 | 60% | 46% | 66% | 32% | 70% | +0.20 | +0.41 | +0.54 | 32% | 99% |
| size-2 | 8.3 | 91% | 20% | 44% | 24% | 36% | +0.30 | +0.50 | +0.62 | 31% | 86% |
| neighbours | 37.7 | 100% | nan% | 17% | 18% | 6% | +nan | +nan | +nan | 43% | 74% |

ESS is the effective sample size of the weights on the training part, as a share of its size. A rank correlation is Spearman's, between estimated and true weights. KLR is kernel logistic regression.

Settings chosen by cross-validation: the uLSIF width factor had median 4 (grid 0.5 to 32, largest chosen in 10% of splits); the classifier's setting was on the edge of its grid in 36% of splits.

## Coverage at 80%

| scenario | `cv` | `cv-norm` | `cv-ulsif` | `cv-klr` | `cv-norm-klr` | `cv-oracle` | `split` | `split-ulsif` | `split-klr` | `split-oracle` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 80.4% | 85.9% | 79.7% | 81.0% | 86.2% | 80.4% | 81.0% | 79.2% (1% inf) | 82.0% | 81.0% |
| pc1-1 | 77.4% **low** | 84.5% | 78.4% (2% inf) | 79.7% (1% inf) | 85.4% (1% inf) | 80.5% | 78.9% | 79.7% (6% inf) | 82.6% (4% inf) | 83.8% (10% inf) |
| pc1-2 | 74.7% **low** | 85.0% | 78.5% (8% inf) | 81.4% (7% inf) | 86.7% (7% inf) | 83.6% (14% inf) | 76.6% **low** | 80.6% (14% inf) | 84.9% (19% inf) | 86.8% (28% inf) |
| size-1 | 79.6% | 85.9% | 77.9% **low** | 80.4% | 86.6% | 81.9% (1% inf) | 81.4% | 79.7% (3% inf) | 83.2% (3% inf) | 84.4% (10% inf) |
| size-2 | 75.6% **low** | 84.3% | 75.6% **low** (2% inf) | 81.7% (4% inf) | 87.0% (4% inf) | 85.0% (16% inf) | 78.5% | 77.0% **low** (8% inf) | 85.5% (18% inf) | 89.1% (39% inf) |
| neighbours | 67.4% **low** | 83.7% | 86.2% (45% inf) | 92.4% (70% inf) | 93.2% (70% inf) | -- | 70.7% **low** | 90.9% (67% inf) | 94.1% (82% inf) | -- |

## Coverage at 90%

| scenario | `cv` | `cv-norm` | `cv-ulsif` | `cv-klr` | `cv-norm-klr` | `cv-oracle` | `split` | `split-ulsif` | `split-klr` | `split-oracle` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 89.7% | 93.6% | 88.5% (1% inf) | 90.0% | 93.6% | 89.7% | 92.2% | 90.1% (3% inf) | 92.3% (2% inf) | 92.2% |
| pc1-1 | 87.8% **low** | 92.9% | 87.6% **low** (4% inf) | 89.7% (3% inf) | 93.6% (3% inf) | 91.0% (7% inf) | 89.6% | 88.6% (13% inf) | 92.0% (17% inf) | 92.4% (26% inf) |
| pc1-2 | 86.1% **low** | 93.0% | 87.9% **low** (11% inf) | 91.2% (15% inf) | 94.1% (15% inf) | 92.1% (21% inf) | 89.4% | 89.9% (27% inf) | 93.6% (37% inf) | 94.2% (47% inf) |
| size-1 | 89.3% | 93.4% | 88.0% **low** (2% inf) | 90.9% (2% inf) | 94.1% (2% inf) | 91.9% (6% inf) | 91.9% | 89.8% (9% inf) | 93.0% (11% inf) | 93.7% (27% inf) |
| size-2 | 86.3% **low** | 92.1% | 85.4% **low** (5% inf) | 90.9% (12% inf) | 94.5% (12% inf) | 93.3% (30% inf) | 89.9% | 87.2% **low** (20% inf) | 94.1% (36% inf) | 95.4% (55% inf) |
| neighbours | 81.5% **low** | 92.5% | 94.3% (60% inf) | 96.4% (78% inf) | 96.7% (78% inf) | -- | 85.9% **low** | 95.3% (79% inf) | 97.2% (88% inf) | -- |

## Coverage at 95%

| scenario | `cv` | `cv-norm` | `cv-ulsif` | `cv-klr` | `cv-norm-klr` | `cv-oracle` | `split` | `split-ulsif` | `split-klr` | `split-oracle` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 94.8% | 97.2% | 93.8% (2% inf) | 95.0% (1% inf) | 97.3% (1% inf) | 94.8% | 97.0% (24% inf) | 95.2% (24% inf) | 97.2% (25% inf) | 97.0% (24% inf) |
| pc1-1 | 93.3% | 96.8% | 92.7% **low** (8% inf) | 94.6% (9% inf) | 96.9% (9% inf) | 95.2% (14% inf) | 95.5% (22% inf) | 94.2% (38% inf) | 96.9% (41% inf) | 97.2% (46% inf) |
| pc1-2 | 92.7% **low** | 96.8% | 92.9% **low** (20% inf) | 95.8% (27% inf) | 97.2% (27% inf) | 96.0% (40% inf) | 95.2% (24% inf) | 94.5% (51% inf) | 98.0% (57% inf) | 97.6% (63% inf) |
| size-1 | 94.7% | 97.1% | 93.7% (5% inf) | 96.0% (6% inf) | 97.4% (6% inf) | 96.1% (17% inf) | 96.7% (24% inf) | 95.0% (32% inf) | 97.3% (39% inf) | 97.7% (50% inf) |
| size-2 | 92.7% **low** | 96.1% | 91.5% **low** (13% inf) | 96.3% (26% inf) | 97.5% (26% inf) | 96.5% (47% inf) | 95.1% (23% inf) | 92.4% **low** (43% inf) | 97.6% (58% inf) | 98.0% (71% inf) |
| neighbours | 88.7% **low** | 97.5% | 97.0% (73% inf) | 98.4% (85% inf) | 98.7% (85% inf) | -- | 92.0% **low** (25% inf) | 97.4% (87% inf) | 98.8% (93% inf) | -- |

## Coverage at 90 % among the finite intervals only

An infinite interval covers by construction. This table leaves those molecules out, so it shows how the weighted rules do on the molecules they do answer for. No guarantee applies to this subset.

| scenario | `cv` | `cv-norm` | `cv-ulsif` | `cv-klr` | `cv-norm-klr` | `cv-oracle` | `split` | `split-ulsif` | `split-klr` | `split-oracle` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 89.7% | 93.6% | 88.3% | 90.0% | 93.6% | 89.7% | 92.2% | 89.5% | 92.1% | 92.2% |
| pc1-1 | 87.8% | 92.9% | 87.2% | 89.4% | 93.5% | 90.3% | 89.6% | 86.6% | 90.5% | 89.3% |
| pc1-2 | 86.1% | 93.0% | 86.4% | 89.7% | 93.0% | 89.9% | 89.4% | 85.6% | 89.7% | 88.6% |
| size-1 | 89.3% | 93.4% | 87.6% | 90.7% | 93.9% | 91.3% | 91.9% | 88.6% | 92.0% | 91.3% |
| size-2 | 86.3% | 92.1% | 84.4% | 89.7% | 93.7% | 90.4% | 89.9% | 83.9% | 90.6% | 89.6% |
| neighbours | 81.5% | 92.5% | 86.4% | 83.8% | 84.9% | -- | 85.9% | 80.0% | 79.0% | -- |

## Width at 90 %, divided by the activity's standard deviation

| scenario | `cv` | `cv-norm` | `cv-ulsif` | `cv-klr` | `cv-norm-klr` | `cv-oracle` | `split` | `split-ulsif` | `split-klr` | `split-oracle` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 2.29 | 2.94 | 2.22 | 2.32 | 2.95 | 2.29 | 2.90 | 2.69 | 2.91 | 2.90 |
| pc1-1 | 2.24 | 2.96 | 2.20 | 2.40 | 3.02 | 2.48 | 2.77 | 2.56 | 2.81 | 2.75 |
| pc1-2 | 2.23 | 3.29 | 2.30 | 2.43 | 3.05 | 2.46 | 2.80 | 2.48 | 2.70 | 2.68 |
| size-1 | 2.30 | 3.08 | 2.23 | 2.45 | 3.16 | 2.58 | 2.92 | 2.67 | 2.98 | 2.85 |
| size-2 | 2.25 | 3.25 | 2.20 | 2.64 | 3.36 | 2.65 | 2.80 | 2.44 | 2.80 | 2.75 |
| neighbours | 2.31 | 3.73 | 2.33 | 2.17 | 2.43 | -- | 2.95 | 2.31 | 2.09 | -- |

## The fit: RMSE on the test part

Mean over sets of the per-set mean RMSE. The difference columns are weighted minus plain (negative is better), with the number of sets, out of those with a value, where the weighted fit was better on average.

| scenario | plain | KMM | uLSIF | KLR | oracle | KMM - plain | uLSIF - plain | KLR - plain | oracle - plain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.937 | 0.956 | 0.964 | 0.939 | 0.937 | +0.019 (2 of 8) | +0.027 (1 of 8) | +0.002 (5 of 8) | +0.000 (1 of 8) |
| pc1-1 | 1.002 | 1.045 | 1.045 | 1.011 | 1.013 | +0.042 (1 of 8) | +0.043 (2 of 8) | +0.009 (2 of 8) | +0.010 (3 of 8) |
| pc1-2 | 1.056 | 1.102 | 1.129 | 1.070 | 1.073 | +0.047 (3 of 8) | +0.073 (2 of 8) | +0.014 (4 of 8) | +0.017 (2 of 8) |
| size-1 | 0.958 | 0.966 | 0.969 | 0.956 | 0.951 | +0.008 (2 of 8) | +0.012 (2 of 8) | -0.001 (5 of 8) | -0.007 (4 of 8) |
| size-2 | 1.044 | 1.039 | 1.022 | 1.034 | 1.030 | -0.006 (5 of 8) | -0.022 (7 of 8) | -0.011 (5 of 8) | -0.014 (4 of 8) |
| neighbours | 1.192 | 1.172 | 1.309 | 1.270 | -- | -0.021 (5 of 8) | +0.116 (1 of 8) | +0.077 (2 of 8) | -- |

## By set: coverage at 90 % on the strongest structural shift (`pc1-2`)

| set | n | `cv` | `cv-norm` | `cv-ulsif` | `cv-klr` | `cv-norm-klr` | `cv-oracle` |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACE | 114 | 79% | 98% | 92% | 96% | 97% | 94% |
| AChE | 111 | 83% | 91% | 83% | 89% | 94% | 91% |
| BZR | 147 | 85% | 92% | 91% | 93% | 95% | 93% |
| COX2 | 282 | 90% | 94% | 87% | 88% | 92% | 88% |
| DHFR | 361 | 94% | 96% | 91% | 93% | 95% | 92% |
| GPB | 66 | 85% | 94% | 87% | 92% | 95% | 94% |
| THER | 76 | 88% | 89% | 89% | 91% | 93% | 91% |
| THR | 88 | 86% | 91% | 83% | 89% | 92% | 93% |
