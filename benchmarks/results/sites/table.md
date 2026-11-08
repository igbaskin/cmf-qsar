# Site-resolved models of a periodic cell (plan item M5)

## The single-atom catalysts: one measured site per cell

139 catalysts; measured targets dE_H 133, dE_O 133, dE_OH 133, dE_OOH 119.
Every method chose its settings in an inner cross-validation of the same kind as the outer folds, over the cutoffs 4, 6, 8 A and the ridge values.

| Method | Folds | dE_H | dE_O | dE_OH | dE_OOH | Mean of the four |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| site profile (M5) | random | +0.568 | +0.930 | +0.938 | +0.707 | +0.786 |
| site profile (M5) | metal-out | +0.405 | +0.866 | +0.893 | +0.615 | +0.695 |
| site profile (M5) | motif-out | +0.523 | +0.902 | +0.934 | +0.692 | +0.763 |
| whole cell autocorrelation | random | +0.571 | +0.921 | +0.941 | +0.752 | +0.796 |
| whole cell autocorrelation | metal-out | +0.506 | +0.859 | +0.907 | +0.718 | +0.747 |
| whole cell autocorrelation | motif-out | +0.501 | +0.901 | +0.840 | +0.633 | +0.719 |
| counts around the site | random | +0.437 | +0.878 | +0.914 | +0.652 | +0.720 |
| counts around the site | metal-out | -0.008 | +0.124 | +0.280 | +0.180 | +0.144 |
| counts around the site | motif-out | +0.505 | +0.833 | +0.913 | +0.636 | +0.722 |
| site overlap, cut cluster (EC8) | random | +0.520 | +0.868 | +0.924 | +0.734 | +0.761 |
| site overlap, cut cluster (EC8) | metal-out | +0.456 | +0.636 | +0.875 | +0.664 | +0.658 |
| site overlap, cut cluster (EC8) | motif-out | +0.501 | +0.817 | +0.860 | +0.668 | +0.712 |
| site invariant, cut cluster (LP2) | random | +0.597 | +0.922 | +0.940 | +0.750 | +0.802 |
| site invariant, cut cluster (LP2) | metal-out | +0.540 | +0.884 | +0.893 | +0.635 | +0.738 |
| site invariant, cut cluster (LP2) | motif-out | +0.479 | +0.833 | +0.877 | +0.710 | +0.725 |

q2 out of fold.  The paired difference of the site profile against every other method, resampling the catalysts:

| Against | Folds | Target | Difference in q2 | 95 % interval |
| --- | --- | --- | ---: | --- |
| whole cell autocorrelation | random | dE_H | -0.004 | -0.071 to +0.076 |
| counts around the site | random | dE_H | +0.131 | +0.027 to +0.231 |
| site overlap, cut cluster (EC8) | random | dE_H | +0.048 | -0.045 to +0.136 |
| site invariant, cut cluster (LP2) | random | dE_H | -0.029 | -0.075 to +0.023 |
| whole cell autocorrelation | random | dE_O | +0.009 | -0.002 to +0.022 |
| counts around the site | random | dE_O | +0.053 | +0.025 to +0.090 |
| site overlap, cut cluster (EC8) | random | dE_O | +0.062 | +0.039 to +0.095 |
| site invariant, cut cluster (LP2) | random | dE_O | +0.008 | -0.002 to +0.018 |
| whole cell autocorrelation | random | dE_OH | -0.003 | -0.012 to +0.007 |
| counts around the site | random | dE_OH | +0.024 | +0.006 to +0.042 |
| site overlap, cut cluster (EC8) | random | dE_OH | +0.014 | -0.004 to +0.033 |
| site invariant, cut cluster (LP2) | random | dE_OH | -0.002 | -0.010 to +0.006 |
| whole cell autocorrelation | random | dE_OOH | -0.045 | -0.113 to +0.009 |
| counts around the site | random | dE_OOH | +0.056 | -0.009 to +0.134 |
| site overlap, cut cluster (EC8) | random | dE_OOH | -0.027 | -0.095 to +0.037 |
| site invariant, cut cluster (LP2) | random | dE_OOH | -0.043 | -0.078 to -0.009 |
| whole cell autocorrelation | metal-out | dE_H | -0.101 | -0.199 to -0.013 |
| counts around the site | metal-out | dE_H | +0.413 | +0.279 to +0.540 |
| site overlap, cut cluster (EC8) | metal-out | dE_H | -0.051 | -0.186 to +0.080 |
| site invariant, cut cluster (LP2) | metal-out | dE_H | -0.135 | -0.230 to -0.053 |
| whole cell autocorrelation | metal-out | dE_O | +0.008 | -0.022 to +0.036 |
| counts around the site | metal-out | dE_O | +0.743 | +0.667 to +0.825 |
| site overlap, cut cluster (EC8) | metal-out | dE_O | +0.230 | +0.143 to +0.333 |
| site invariant, cut cluster (LP2) | metal-out | dE_O | -0.017 | -0.051 to +0.010 |
| whole cell autocorrelation | metal-out | dE_OH | -0.013 | -0.038 to +0.014 |
| counts around the site | metal-out | dE_OH | +0.613 | +0.544 to +0.695 |
| site overlap, cut cluster (EC8) | metal-out | dE_OH | +0.018 | -0.015 to +0.055 |
| site invariant, cut cluster (LP2) | metal-out | dE_OH | +0.000 | -0.021 to +0.025 |
| whole cell autocorrelation | metal-out | dE_OOH | -0.103 | -0.187 to -0.027 |
| counts around the site | metal-out | dE_OOH | +0.435 | +0.277 to +0.575 |
| site overlap, cut cluster (EC8) | metal-out | dE_OOH | -0.049 | -0.131 to +0.028 |
| site invariant, cut cluster (LP2) | metal-out | dE_OOH | -0.020 | -0.095 to +0.063 |
| whole cell autocorrelation | motif-out | dE_H | +0.021 | -0.044 to +0.092 |
| counts around the site | motif-out | dE_H | +0.018 | -0.120 to +0.152 |
| site overlap, cut cluster (EC8) | motif-out | dE_H | +0.021 | -0.140 to +0.179 |
| site invariant, cut cluster (LP2) | motif-out | dE_H | +0.043 | -0.067 to +0.149 |
| whole cell autocorrelation | motif-out | dE_O | +0.001 | -0.011 to +0.011 |
| counts around the site | motif-out | dE_O | +0.069 | +0.019 to +0.126 |
| site overlap, cut cluster (EC8) | motif-out | dE_O | +0.085 | +0.039 to +0.136 |
| site invariant, cut cluster (LP2) | motif-out | dE_O | +0.069 | +0.028 to +0.113 |
| whole cell autocorrelation | motif-out | dE_OH | +0.093 | +0.051 to +0.147 |
| counts around the site | motif-out | dE_OH | +0.021 | -0.001 to +0.043 |
| site overlap, cut cluster (EC8) | motif-out | dE_OH | +0.074 | +0.041 to +0.116 |
| site invariant, cut cluster (LP2) | motif-out | dE_OH | +0.057 | +0.030 to +0.090 |
| whole cell autocorrelation | motif-out | dE_OOH | +0.059 | -0.008 to +0.141 |
| counts around the site | motif-out | dE_OOH | +0.056 | +0.010 to +0.112 |
| site overlap, cut cluster (EC8) | motif-out | dE_OOH | +0.024 | -0.057 to +0.099 |
| site invariant, cut cluster (LP2) | motif-out | dE_OOH | -0.018 | -0.059 to +0.017 |

## The lithium sites: many measured sites per cell

104 Li sites in 3 compounds (phosphate, borophosphate, borate), 3 folds holding out whole compounds; the site energies have a spread of 0.111 eV.

| Method | q2 | Root mean square error, eV | Mean absolute error, eV |
| --- | ---: | ---: | ---: |
| site profile (M5), r=4 A | -0.071 | 0.115 | 0.108 |
| site profile (M5), r=6 A | -1.309 | 0.168 | 0.161 |
| site profile (M5), r=8 A | -0.235 | 0.123 | 0.115 |
| counts around the site | -1.151 | 0.163 | 0.155 |
| the mean of the other compounds | -1.151 | 0.163 | 0.155 |

By compound, for site profile (M5), r=4 A.  The offset is how far the whole compound is predicted from where it sits; the last two columns leave that offset out and ask only whether the sites inside the compound are ordered correctly.

| Compound | Sites | Mean absolute error, eV | Spread, eV | Offset, eV | q2 about its own mean | Spearman correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| phosphate | 48 | 0.123 | 0.031 | -0.123 | +0.604 | +0.78 |
| borophosphate | 24 | 0.054 | 0.124 | +0.039 | +0.766 | +0.84 |
| borate | 32 | 0.125 | 0.007 | +0.125 | +0.001 | +0.55 |

## A whole-cell model read site by site

A model of the OH binding energy on the whole-cell autocorrelation of 133 catalysts (q2 +0.908 in five folds, 0.444 eV).  Its prediction splits exactly over the sites of the cell.  The metal and its first coordination shell carry 16.4 % of the total weight (the metal alone 8.6 %), where an even spread over every site of the sheet would give 3.9 %: a factor of 4.2.

| Distance from the metal, angstrom | Share of the weight, per cent | Share of the sites, per cent | Times an even spread |
| --- | ---: | ---: | ---: |
| 0.0 to 1.0 | 8.6 | 0.8 | 10.9 |
| 1.0 to 2.4 | 7.8 | 3.1 | 2.5 |
| 2.4 to 3.5 | 10.5 | 7.0 | 1.5 |
| 3.5 to 5.0 | 25.6 | 11.5 | 2.2 |
| 5.0 to 7.0 | 14.6 | 22.4 | 0.7 |
| beyond 7.0 | 32.8 | 55.2 | 0.6 |

