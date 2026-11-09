# Site-resolved models of a periodic cell (plan item M5)

## The map of a cell model, against the site model (plan item MS2)

A whole-cell model's prediction splits into one number per site, exactly. The question of this section is whether those numbers rank the sites as a model fitted on the sites themselves ranks them. Both models use the same kernel, the same cutoff (6 A) and the same ridge value (0.01); only the sample differs. Everything below is out of fold: for each of the 5 folds both models are fitted on the other folds and the held-out cells are split and predicted site by site. The data set has grown since the section below was last computed, so the two sections count different numbers of catalysts for the same target; the comparison here is internal to this section.

**The split adds up to the prediction.** This is the first half of the gate, and it is exact arithmetic rather than a measurement:

| Target | Catalysts | Largest difference between the sum and the prediction |
| --- | ---: | ---: |
| dE_H | 133 | 2.8e-13 |
| dE_O | 133 | 3.5e-13 |
| dE_OH | 133 | 2.0e-13 |
| dE_OOH | 119 | 1.8e-13 |

**Where the split puts its weight.** The binding energy of these catalysts is a property of the metal site, so this has a known right answer. The split finds it:

| Target | The metal is the largest place | Among the first three | Weight on the metal | An even spread would give |
| --- | ---: | ---: | ---: | ---: |
| dE_H | 72 % | 84 % | 11.7 % | 0.8 % |
| dE_O | 62 % | 76 % | 9.0 % | 0.8 % |
| dE_OH | 77 % | 81 % | 8.1 % | 0.8 % |
| dE_OOH | 69 % | 70 % | 8.6 % | 0.8 % |

**The ranking the gate asks about.** Within one cell the split and the site model are compared over all 127 sites, which is the literal reading of the gate; across the cells they are compared at the one site of each cell that was measured:

| Target | Within a cell, mean | Within a cell, median | Cells above +0.7 | Across cells at the measured site |
| --- | ---: | ---: | ---: | ---: |
| dE_H | +0.465 | +0.477 | 1 % | +0.219 |
| dE_O | +0.476 | +0.526 | 13 % | +0.743 |
| dE_OH | +0.350 | +0.444 | 3 % | +0.600 |
| dE_OOH | +0.025 | +0.038 | 0 % | -0.001 |

**The two models, compared as models.** A contribution is a share of one structure's answer and a prediction is an answer of its own, so the rows above compare two different quantities. These rows compare the quantities that are alike: what each model predicts, against the other and against the measurement.

| Target | Cell model against site model | Cell model against the measurement | Site model against the measurement |
| --- | ---: | ---: | ---: |
| dE_H | +0.968 | +0.771 | +0.762 |
| dE_O | +0.987 | +0.927 | +0.927 |
| dE_OH | +0.992 | +0.952 | +0.956 |
| dE_OOH | +0.971 | +0.883 | +0.862 |

Rank correlations (Spearman).

**The second half of the gate is not met, and the measurement says why.** What the split does do is locate the property. The metal site is the largest place of the split in 62 to 77 % of the cells, its median rank is 1 on every target, and it carries 8.1 to 11.7 % of the weight where an even spread over the 127 sites would give it 0.8 %, which is ten to fifteen times its share. What it does not do is reproduce the site model's ordering of all the sites of a cell: the mean rank correlation inside a cell is +0.025 to +0.476 and it passes +0.7 on at most 13 % of the cells. Across the cells, at the one site of each cell that was measured, it is -0.001 to +0.743, which passes +0.7 on one of the four targets.

Two reasons, and neither is a fault in either model. First, the two quantities differ: a contribution says how much of this cell's answer came from this place, and a prediction says what would be measured at that place, so the places a cell model leans on need not be the places that bind most strongly. Second, the site model was fitted on the metal site of every cell and on no other site, so its answers at the carbon and nitrogen sites are extrapolation; the gate compares the split against a model that was never shown the places the comparison runs over. The reading that stays inside both models' training data is the last table, where the two predictions agree closely and both track the measurement.

## The single-atom catalysts: one measured site per cell

139 catalysts; measured targets dE_H 104, dE_O 104, dE_OH 104, dE_OOH 91.
Every method chose its settings in an inner cross-validation of the same kind as the outer folds, over the cutoffs 4, 6, 8 A and the ridge values.

| Method | Folds | dE_H | dE_O | dE_OH | dE_OOH | Mean of the four |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| site profile (M5) | random | +0.556 | +0.924 | +0.929 | +0.761 | +0.792 |
| site profile (M5) | metal-out | +0.452 | +0.862 | +0.903 | +0.666 | +0.721 |
| site profile (M5) | motif-out | +0.552 | +0.819 | +0.903 | +0.746 | +0.755 |
| whole cell autocorrelation | random | +0.558 | +0.923 | +0.925 | +0.770 | +0.794 |
| whole cell autocorrelation | metal-out | +0.448 | +0.907 | +0.916 | +0.751 | +0.756 |
| whole cell autocorrelation | motif-out | +0.524 | +0.834 | +0.837 | +0.779 | +0.744 |
| counts around the site | random | +0.503 | +0.829 | +0.899 | +0.802 | +0.758 |
| counts around the site | metal-out | -0.002 | +0.035 | +0.235 | +0.115 | +0.096 |
| counts around the site | motif-out | +0.496 | +0.792 | +0.900 | +0.806 | +0.749 |
| site overlap, cut cluster (EC8) | random | +0.550 | +0.844 | +0.929 | +0.806 | +0.782 |
| site overlap, cut cluster (EC8) | metal-out | +0.398 | +0.744 | +0.906 | +0.734 | +0.695 |
| site overlap, cut cluster (EC8) | motif-out | +0.481 | +0.747 | +0.853 | +0.815 | +0.724 |
| site invariant, cut cluster (LP2) | random | +0.583 | +0.917 | +0.926 | +0.779 | +0.801 |
| site invariant, cut cluster (LP2) | metal-out | +0.487 | +0.876 | +0.897 | +0.696 | +0.739 |
| site invariant, cut cluster (LP2) | motif-out | +0.477 | +0.807 | +0.841 | +0.723 | +0.712 |

q2 out of fold.  The paired difference of the site profile against every other method, resampling the catalysts:

| Against | Folds | Target | Difference in q2 | 95 % interval |
| --- | --- | --- | ---: | --- |
| whole cell autocorrelation | random | dE_H | -0.003 | -0.064 to +0.050 |
| counts around the site | random | dE_H | +0.052 | -0.063 to +0.167 |
| site overlap, cut cluster (EC8) | random | dE_H | +0.006 | -0.101 to +0.112 |
| site invariant, cut cluster (LP2) | random | dE_H | -0.027 | -0.094 to +0.037 |
| whole cell autocorrelation | random | dE_O | +0.001 | -0.014 to +0.016 |
| counts around the site | random | dE_O | +0.095 | +0.056 to +0.152 |
| site overlap, cut cluster (EC8) | random | dE_O | +0.079 | +0.047 to +0.122 |
| site invariant, cut cluster (LP2) | random | dE_O | +0.007 | -0.005 to +0.021 |
| whole cell autocorrelation | random | dE_OH | +0.003 | -0.009 to +0.017 |
| counts around the site | random | dE_OH | +0.030 | -0.001 to +0.067 |
| site overlap, cut cluster (EC8) | random | dE_OH | -0.000 | -0.021 to +0.020 |
| site invariant, cut cluster (LP2) | random | dE_OH | +0.003 | -0.010 to +0.017 |
| whole cell autocorrelation | random | dE_OOH | -0.009 | -0.065 to +0.025 |
| counts around the site | random | dE_OOH | -0.040 | -0.149 to +0.034 |
| site overlap, cut cluster (EC8) | random | dE_OOH | -0.044 | -0.106 to +0.010 |
| site invariant, cut cluster (LP2) | random | dE_OOH | -0.018 | -0.066 to +0.031 |
| whole cell autocorrelation | metal-out | dE_H | +0.003 | -0.099 to +0.107 |
| counts around the site | metal-out | dE_H | +0.454 | +0.322 to +0.588 |
| site overlap, cut cluster (EC8) | metal-out | dE_H | +0.053 | -0.099 to +0.230 |
| site invariant, cut cluster (LP2) | metal-out | dE_H | -0.035 | -0.134 to +0.070 |
| whole cell autocorrelation | metal-out | dE_O | -0.046 | -0.113 to -0.001 |
| counts around the site | metal-out | dE_O | +0.827 | +0.729 to +0.939 |
| site overlap, cut cluster (EC8) | metal-out | dE_O | +0.118 | +0.045 to +0.195 |
| site invariant, cut cluster (LP2) | metal-out | dE_O | -0.015 | -0.074 to +0.026 |
| whole cell autocorrelation | metal-out | dE_OH | -0.013 | -0.029 to +0.003 |
| counts around the site | metal-out | dE_OH | +0.669 | +0.591 to +0.764 |
| site overlap, cut cluster (EC8) | metal-out | dE_OH | -0.003 | -0.033 to +0.030 |
| site invariant, cut cluster (LP2) | metal-out | dE_OH | +0.006 | -0.013 to +0.032 |
| whole cell autocorrelation | metal-out | dE_OOH | -0.085 | -0.148 to -0.033 |
| counts around the site | metal-out | dE_OOH | +0.551 | +0.344 to +0.724 |
| site overlap, cut cluster (EC8) | metal-out | dE_OOH | -0.068 | -0.143 to -0.001 |
| site invariant, cut cluster (LP2) | metal-out | dE_OOH | -0.030 | -0.084 to +0.023 |
| whole cell autocorrelation | motif-out | dE_H | +0.027 | -0.043 to +0.104 |
| counts around the site | motif-out | dE_H | +0.056 | -0.088 to +0.225 |
| site overlap, cut cluster (EC8) | motif-out | dE_H | +0.070 | -0.107 to +0.260 |
| site invariant, cut cluster (LP2) | motif-out | dE_H | +0.075 | -0.043 to +0.200 |
| whole cell autocorrelation | motif-out | dE_O | -0.015 | -0.053 to +0.015 |
| counts around the site | motif-out | dE_O | +0.028 | -0.035 to +0.083 |
| site overlap, cut cluster (EC8) | motif-out | dE_O | +0.072 | -0.007 to +0.150 |
| site invariant, cut cluster (LP2) | motif-out | dE_O | +0.013 | -0.038 to +0.068 |
| whole cell autocorrelation | motif-out | dE_OH | +0.066 | +0.028 to +0.116 |
| counts around the site | motif-out | dE_OH | +0.003 | -0.048 to +0.045 |
| site overlap, cut cluster (EC8) | motif-out | dE_OH | +0.051 | -0.001 to +0.107 |
| site invariant, cut cluster (LP2) | motif-out | dE_OH | +0.062 | +0.031 to +0.103 |
| whole cell autocorrelation | motif-out | dE_OOH | -0.034 | -0.106 to +0.011 |
| counts around the site | motif-out | dE_OOH | -0.061 | -0.221 to +0.045 |
| site overlap, cut cluster (EC8) | motif-out | dE_OOH | -0.069 | -0.177 to +0.004 |
| site invariant, cut cluster (LP2) | motif-out | dE_OOH | +0.022 | -0.011 to +0.057 |

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

A model of the OH binding energy on the whole-cell autocorrelation of 104 catalysts (q2 +0.893 in five folds, 0.511 eV).  Its prediction splits exactly over the sites of the cell.  The metal and its first coordination shell carry 16.0 % of the total weight (the metal alone 8.9 %), where an even spread over every site of the sheet would give 3.9 %: a factor of 4.1.

| Distance from the metal, angstrom | Share of the weight, per cent | Share of the sites, per cent | Times an even spread |
| --- | ---: | ---: | ---: |
| 0.0 to 1.0 | 8.9 | 0.8 | 11.3 |
| 1.0 to 2.4 | 7.1 | 3.1 | 2.3 |
| 2.4 to 3.5 | 10.6 | 7.0 | 1.5 |
| 3.5 to 5.0 | 29.0 | 11.4 | 2.5 |
| 5.0 to 7.0 | 14.3 | 22.5 | 0.6 |
| beyond 7.0 | 30.2 | 55.2 | 0.5 |

