| Set | Learner | q2 | RMSEcv | R2pred | RMSEpred | q2ex | RMSEcvex | alpha | setting chosen |
|---|---|---|---|---|---|---|---|---|---|
| **ACE** | kernel ridge regression (lambda by CV) | 0.73 | 1.21 | 0.61 | 1.31 | 0.69 | 1.25 | 2.00 | lambda 1.000 |
|  | kernel PLS (components by CV) | 0.73 | 1.22 | 0.57 | 1.39 | 0.66 | 1.33 | 3.00 | 11 comp. |
| **AChE** | kernel ridge regression (lambda by CV) | 0.59 | 0.77 | 0.64 | 0.78 | 0.57 | 0.81 | 3.00 | lambda 0.300 |
|  | kernel PLS (components by CV) | 0.60 | 0.77 | 0.62 | 0.80 | 0.52 | 0.86 | 3.00 | 6 comp. |
| **BZR** | kernel ridge regression (lambda by CV) | 0.42 | 0.50 | 0.08 | 0.85 | 0.29 | 0.64 | 2.00 | lambda 1.000 |
|  | kernel PLS (components by CV) | 0.39 | 0.52 | 0.03 | 0.88 | 0.25 | 0.66 | 2.00 | 6 comp. |
| **COX2** | kernel ridge regression (lambda by CV) | 0.61 | 0.63 | 0.30 | 1.10 | 0.50 | 0.82 | 1.00 | lambda 3.000 |
|  | kernel PLS (components by CV) | 0.58 | 0.66 | 0.22 | 1.17 | 0.45 | 0.86 | 2.00 | 8 comp. |
| **DHFR** | kernel ridge regression (lambda by CV) | 0.65 | 0.75 | 0.62 | 0.84 | 0.65 | 0.78 | 5.00 | lambda 1.000 |
|  | kernel PLS (components by CV) | 0.63 | 0.77 | 0.55 | 0.91 | 0.62 | 0.81 | 10.00 | 4 comp. |
| **GPB** | kernel ridge regression (lambda by CV) | 0.71 | 0.57 | 0.50 | 0.85 | 0.51 | 0.78 | 10.00 | lambda 0.100 |
|  | kernel PLS (components by CV) | 0.71 | 0.57 | 0.46 | 0.89 | 0.53 | 0.76 | 10.00 | 7 comp. |
| **THER** | kernel ridge regression (lambda by CV) | 0.57 | 1.23 | 0.37 | 1.78 | 0.34 | 1.64 | 10.00 | lambda 0.030 |
|  | kernel PLS (components by CV) | 0.58 | 1.22 | 0.35 | 1.81 | 0.33 | 1.65 | 10.00 | 10 comp. |
| **THR** | kernel ridge regression (lambda by CV) | 0.72 | 0.50 | 0.42 | 0.84 | 0.69 | 0.56 | 0.10 | lambda 20.000 |
|  | kernel PLS (components by CV) | 0.74 | 0.48 | 0.07 | 1.06 | 0.66 | 0.58 | 0.05 | 7 comp. |

Mean over the sets with both learners (8 sets), and on how many of them kernel PLS is the better of the two:

| Learner | q2 | RMSEcv | R2pred | RMSEpred | q2ex | RMSEcvex |
|---|---|---|---|---|---|---|
| kernel ridge regression (lambda by CV) | 0.625 | 0.772 | 0.443 | 1.045 | 0.530 | 0.911 |
| kernel PLS (components by CV) | 0.619 | 0.777 | 0.357 | 1.113 | 0.502 | 0.939 |
| kernel PLS better, sets out of 8 | 4 | 4 | 0 | 0 | 1 | 1 |
