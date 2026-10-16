| Set | Model | r2 | RMSE | q2 | RMSEcv | R2ext | R2ext(tr) | RMSEext | q2ext | RMSEext-cv | alpha | lambda |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **THR** | paper, CIF/CMF (Table 1) | 0.956 | 0.21 | 0.772 | 0.49 | 0.311 | 0.431 | 0.75 | 0.712 | 0.54 | | |
|  | paper, PC/CMF (Table 1) | 0.929 | 0.27 | 0.747 | 0.52 | 0.280 | 0.404 | 0.77 | 0.764 | 0.49 | | |
| | this work, paper protocol (Nelder-Mead from (0.25, 5)) | 0.956 | 0.21 | 0.772 | 0.49 | 0.311 | 0.430 | 0.75 | 0.684 | 0.57 | 0.243 | 93.681 |
| | this work, alpha grid + Brent lambda (2013 protocol) | 0.971 | 0.17 | 0.753 | 0.51 | 0.444 | 0.540 | 0.68 | 0.690 | 0.56 | 0.500 | 20.000 |
| | this work, physico-chemical fields, Nelder-Mead | 0.941 | 0.25 | 0.750 | 0.51 | 0.288 | 0.411 | 0.77 | 0.672 | 0.58 | 0.750 | 3.255 |
| | this work, physico-chemical fields, alpha grid + Brent lambda | 0.929 | 0.27 | 0.747 | 0.52 | 0.280 | 0.404 | 0.77 | 0.665 | 0.58 | 0.500 | 7.484 |
| **AmEu** | paper (Table 2, text) | 0.906 | - | 0.707 | 0.24 | 0.745 | 0.767 | 0.20 | 0.668 | 0.25 | | |
| | this work, paper protocol (Nelder-Mead from (0.25, 5)) | 0.913 | 0.13 | 0.687 | 0.24 | 0.744 | 0.767 | 0.20 | 0.665 | 0.25 | 10.000 | 0.975 |
| | this work, paper protocol, b = 0 | 0.906 | 0.13 | 0.707 | 0.24 | 0.745 | 0.767 | 0.20 | 0.664 | 0.25 | 10.000 | 1.108 |
| | this work, alpha grid + Brent lambda (2013 protocol) | 0.894 | 0.14 | 0.675 | 0.25 | 0.727 | 0.751 | 0.21 | 0.666 | 0.25 | 3.000 | 6.482 |
| **DYES** | paper, all atom types (Table 3) | 1.000 | - | 0.956 | 0.69 | - | - | - | 0.909 | 0.99 | | |
|  | paper, hydrogens only (Table 3) | 0.999 | - | 0.940 | 0.80 | - | - | - | 0.822 | 1.39 | | |
| | this work, paper protocol (Nelder-Mead from (0.25, 5)) | 0.981 | 0.45 | 0.924 | 0.90 | - | - | - | 0.906 | 1.00 | 0.010 | 7.163 |
| | this work, paper protocol, Nelder-Mead from (1, 1) | 1.000 | 0.07 | 0.959 | 0.67 | - | - | - | 0.923 | 0.91 | 1.371 | 0.017 |
| | this work, hydrogens only, paper protocol | 0.999 | 0.12 | 0.940 | 0.80 | - | - | - | 0.884 | 1.12 | 10.000 | 0.006 |
| | this work, hydrogens only, Nelder-Mead from (1, 1) | 0.999 | 0.12 | 0.940 | 0.80 | - | - | - | 0.891 | 1.08 | 10.000 | 0.006 |
| | this work, alpha grid + Brent lambda (2013 protocol) | 1.000 | 0.05 | 0.956 | 0.69 | - | - | - | 0.913 | 0.97 | 1.000 | 0.010 |
