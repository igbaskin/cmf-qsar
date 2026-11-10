# A spectrum as the sample where the spectrum should hold the answer: infrared spectra against the Hammett constant (plan item SP1)

The 189 substituted benzoic acids of plan item SP2 (95 substituents, meta and para).  The sample is the infrared spectrum of the acid from its ORCA frequency job: one stick for every normal mode, at the scaled harmonic wavenumber, with the infrared intensity (43 sticks between 400 and 4000 cm^-1 at the median, 32 to 86).  The target is the Hammett constant of the substituent at its position (-0.83 to +1.20, standard deviation 0.33), which is a measured number.  The spectrum is computed, and the calculation is not told the target.

The kernel of two spectra is the overlap of the two after every stick is replaced by a Gaussian (`cmfqsar/spectrum.py`), divided by the norms of the two.  The width of the Gaussian (one of 2, 5, 10, 20, 50 cm^-1), the power of the intensity (1, or 0.5 for the square root) and the ridge are chosen on 5 folds inside every training part.  The cross-validation has 10 folds and is repeated 10 times with other folds.  A difference in q2 carries its 95 % bootstrap interval over the units that the folds hold out.

The series holds each substituent twice, meta and para.  A spectrum shows the bands of the substituent itself, so with whole acids held out a model can recognise the substituent from the other acid.  The numbers with whole substituents held out are the ones that the reading uses.

## Control: the length of the C=O bond from the spectrum

The length of the C=O bond of the optimised geometry (120.88 to 121.60 pm, standard deviation 0.116 pm) follows the position of the carbonyl band (r -0.987; Badger's rule).  The position of one line of the spectrum therefore holds the answer, and a kernel that can read the position of a line has to find it.

| description | q2, whole acids held out | q2, whole substituents held out | settings chosen most often, whole substituents held out |
|---|---:|---:|---|
| linear regression on the position of the carbonyl band | +0.973 (0.000) | +0.973 (0.000) |  |
| infrared spectrum, 400 to 4000 cm^-1 | +0.871 (0.008) | +0.858 (0.010) | 10 cm^-1, power 1 (51 %); 5 cm^-1, power 1 (32 %) |
| infrared spectrum, the region of the carbonyl stretch alone | +0.871 (0.197) | +0.895 (0.041) | 5 cm^-1, power 1 (90 %); 10 cm^-1, power 1 (8 %) |
| infrared spectrum without the region of the carbonyl stretch | +0.750 (0.010) | +0.724 (0.010) | 5 cm^-1, power 0.5 (82 %); 2 cm^-1, power 0.5 (18 %) |

## Before any fit: how far each kernel follows the target

The correlation between the centred kernel and the products of the centred Hammett constants, over the pairs of acids, at the setting chosen most often with whole substituents held out.  The first measurement of this item found that this number, which needs no fit, says in advance what the fit finds.

| description | setting | correlation with the target | correlation with the kernel of the whole spectrum |
|---|---|---:|---:|
| atom-centred counts around the carbonyl carbon |  | +0.228 | +0.250 |
| structure: the five fields, Gasteiger charges | 10.00 A | +0.199 | +0.278 |
| structure: the electrostatic field alone, quantum charges | 1.00 A | +0.432 | +0.235 |
| infrared spectrum, 400 to 4000 cm^-1 | 5 cm^-1, power 1 | +0.438 | 1 |
| infrared spectrum, the region of the carbonyl stretch alone | 10 cm^-1, power 1 | +0.629 | +0.595 |
| infrared spectrum without the region of the carbonyl stretch | 2 cm^-1, power 0.5 | +0.228 | +0.657 |

## The Hammett constant from each description

The band of the carbonyl stretch correlates with the Hammett constant with r +0.866.  The last column is the q2 of the predicted difference between the para acid and the meta acid of one substituent, which is the part of the constant that resonance decides.

### The Hammett constant, whole acids held out

| description | q2 (sd over the repeats) | RMSE | settings chosen most often | q2 of the difference between the para and the meta acid |
|---|---:|---:|---|---:|
| atom-centred counts around the carbonyl carbon | +0.683 (0.026) | 0.186 | counts (100 %) | -0.14 |
| linear regression on the position of the carbonyl band | +0.743 (0.002) | 0.167 |  | -0.05 |
| linear regression on the position of the carbonyl band, meta and para separately (two equations; it is told the position) | +0.771 (0.001) | 0.158 |  | +0.32 |
| structure: the five fields, Gasteiger charges | +0.778 (0.028) | 0.155 | 10.00 A (100 %) | -0.40 |
| structure: the electrostatic field alone, quantum charges | +0.835 (0.012) | 0.134 | 2.00 A (63 %); 1.00 A (32 %) | +0.54 |
| infrared spectrum, 400 to 4000 cm^-1 | +0.847 (0.009) | 0.129 | 10 cm^-1, power 1 (51 %); 5 cm^-1, power 1 (44 %) | +0.14 |
| infrared spectrum, Gaussian kernel on top | +0.860 (0.011) | 0.124 | 10 cm^-1, power 0.5, factor 0.25 (24 %); 10 cm^-1, power 1, factor 0.25 (23 %) | +0.31 |
| infrared spectrum, the region of the carbonyl stretch alone | +0.753 (0.014) | 0.164 | 10 cm^-1, power 1 (67 %); 5 cm^-1, power 1 (29 %) | -0.32 |
| infrared spectrum without the region of the carbonyl stretch | +0.735 (0.017) | 0.170 | 5 cm^-1, power 0.5 (75 %); 2 cm^-1, power 0.5 (11 %) | -0.35 |
| structure (five fields) and infrared spectrum, joined | +0.875 (0.011) | 0.117 | 10.00 A + 5 cm^-1, power 1, share 0.5 (30 %); 10.00 A + 10 cm^-1, power 1, share 0.75 (18 %) | +0.24 |
| structure (quantum charges) and infrared spectrum, joined | +0.824 (0.067) | 0.138 | 10.00 A + 10 cm^-1, power 1, share 0.25 (15 %); 1.00 A + 50 cm^-1, power 1, share 0.25 (13 %) | +0.37 |

| description | compared with | difference in q2 | reading |
|---|---|---:|---|
| infrared spectrum, 400 to 4000 cm^-1 | atom-centred counts around the carbonyl carbon | +0.163 [+0.095, +0.246] | above |
| infrared spectrum, 400 to 4000 cm^-1 | linear regression on the position of the carbonyl band | +0.104 [+0.051, +0.168] | above |
| infrared spectrum, 400 to 4000 cm^-1 | linear regression on the position of the carbonyl band, meta and para separately (two equations; it is told the position) | +0.075 [+0.026, +0.134] | above |
| infrared spectrum, 400 to 4000 cm^-1 | structure: the five fields, Gasteiger charges | +0.068 [+0.011, +0.136] | above |
| infrared spectrum, 400 to 4000 cm^-1 | structure: the electrostatic field alone, quantum charges | +0.012 [-0.035, +0.065] | not separated |
| infrared spectrum, Gaussian kernel on top | infrared spectrum, 400 to 4000 cm^-1 | +0.013 [-0.004, +0.032] | not separated |
| infrared spectrum, Gaussian kernel on top | linear regression on the position of the carbonyl band | +0.117 [+0.066, +0.182] | above |
| infrared spectrum, the region of the carbonyl stretch alone | linear regression on the position of the carbonyl band | +0.011 [-0.024, +0.048] | not separated |
| infrared spectrum without the region of the carbonyl stretch | infrared spectrum, 400 to 4000 cm^-1 | -0.112 [-0.163, -0.065] | below |
| infrared spectrum without the region of the carbonyl stretch | atom-centred counts around the carbonyl carbon | +0.052 [-0.024, +0.137] | not separated |
| structure (five fields) and infrared spectrum, joined | structure: the five fields, Gasteiger charges | +0.097 [+0.046, +0.161] | above |
| structure (five fields) and infrared spectrum, joined | infrared spectrum, 400 to 4000 cm^-1 | +0.029 [+0.010, +0.051] | above |
| structure (quantum charges) and infrared spectrum, joined | structure: the electrostatic field alone, quantum charges | -0.011 [-0.066, +0.041] | not separated |

### The Hammett constant, whole substituents held out (both acids of a substituent together)

| description | q2 (sd over the repeats) | RMSE | settings chosen most often | q2 of the difference between the para and the meta acid |
|---|---:|---:|---|---:|
| atom-centred counts around the carbonyl carbon | +0.629 (0.023) | 0.201 | counts (100 %) | -0.07 |
| linear regression on the position of the carbonyl band | +0.742 (0.003) | 0.168 |  | -0.03 |
| linear regression on the position of the carbonyl band, meta and para separately (two equations; it is told the position) | +0.771 (0.002) | 0.158 |  | +0.32 |
| structure: the five fields, Gasteiger charges | +0.697 (0.024) | 0.182 | 10.00 A (99 %); 4.47 A (1 %) | +0.08 |
| structure: the electrostatic field alone, quantum charges | +0.798 (0.020) | 0.148 | 1.00 A (41 %); 2.00 A (38 %) | +0.47 |
| infrared spectrum, 400 to 4000 cm^-1 | +0.809 (0.014) | 0.144 | 5 cm^-1, power 1 (76 %); 10 cm^-1, power 1 (14 %) | +0.37 |
| infrared spectrum, Gaussian kernel on top | +0.808 (0.017) | 0.145 | 5 cm^-1, power 1, factor 0.25 (73 %); 10 cm^-1, power 1, factor 0.25 (18 %) | +0.46 |
| infrared spectrum, the region of the carbonyl stretch alone | +0.737 (0.016) | 0.169 | 10 cm^-1, power 1 (74 %); 5 cm^-1, power 1 (25 %) | -0.07 |
| infrared spectrum without the region of the carbonyl stretch | +0.661 (0.015) | 0.192 | 2 cm^-1, power 0.5 (45 %); 5 cm^-1, power 0.5 (22 %) | -0.03 |
| structure (five fields) and infrared spectrum, joined | +0.831 (0.017) | 0.136 | 10.00 A + 5 cm^-1, power 1, share 0.5 (45 %); 10.00 A + 5 cm^-1, power 1, share 0.75 (23 %) | +0.48 |
| structure (quantum charges) and infrared spectrum, joined | +0.791 (0.029) | 0.151 | 10.00 A + 5 cm^-1, power 1, share 0.5 (21 %); 10.00 A + 5 cm^-1, power 1, share 0.75 (19 %) | +0.40 |

| description | compared with | difference in q2 | reading |
|---|---|---:|---|
| infrared spectrum, 400 to 4000 cm^-1 | atom-centred counts around the carbonyl carbon | +0.179 [+0.086, +0.293] | above |
| infrared spectrum, 400 to 4000 cm^-1 | linear regression on the position of the carbonyl band | +0.067 [+0.015, +0.130] | above |
| infrared spectrum, 400 to 4000 cm^-1 | linear regression on the position of the carbonyl band, meta and para separately (two equations; it is told the position) | +0.037 [-0.015, +0.098] | not separated |
| infrared spectrum, 400 to 4000 cm^-1 | structure: the five fields, Gasteiger charges | +0.111 [+0.023, +0.244] | above |
| infrared spectrum, 400 to 4000 cm^-1 | structure: the electrostatic field alone, quantum charges | +0.010 [-0.063, +0.111] | not separated |
| infrared spectrum, Gaussian kernel on top | infrared spectrum, 400 to 4000 cm^-1 | -0.001 [-0.013, +0.013] | not separated |
| infrared spectrum, Gaussian kernel on top | linear regression on the position of the carbonyl band | +0.066 [+0.013, +0.133] | above |
| infrared spectrum, the region of the carbonyl stretch alone | linear regression on the position of the carbonyl band | -0.005 [-0.046, +0.031] | not separated |
| infrared spectrum without the region of the carbonyl stretch | infrared spectrum, 400 to 4000 cm^-1 | -0.148 [-0.202, -0.101] | below |
| infrared spectrum without the region of the carbonyl stretch | atom-centred counts around the carbonyl carbon | +0.031 [-0.074, +0.150] | not separated |
| structure (five fields) and infrared spectrum, joined | structure: the five fields, Gasteiger charges | +0.134 [+0.056, +0.253] | above |
| structure (five fields) and infrared spectrum, joined | infrared spectrum, 400 to 4000 cm^-1 | +0.022 [-0.003, +0.047] | not separated |
| structure (quantum charges) and infrared spectrum, joined | structure: the electrostatic field alone, quantum charges | -0.007 [-0.071, +0.066] | not separated |

## The co-spectrum

The model on all acids with the setting chosen most often with whole substituents held out (5 cm^-1, power 1; ridge 0.1).  Its prediction is a constant plus the sum over the axis of a weight times the normalised intensity.  The effect of a grid point is its weight times the standard deviation of the intensity there over the acids (`cospectrum.png`).

The largest effect is 0.0283.  In 50 models fitted to the constants in random order the largest effect is 0.0296 at the mean and 0.0568 at most.  0 of the 1801 grid points are above that largest value.

The ten places with the largest effect:

| wavenumber, cm^-1 | effect | region |
|---:|---:|---|
| 1736 | +0.0283 | C=O stretch |
| 1600 | -0.0204 | ring stretch |
| 1170 | +0.0113 | C-O stretch, O-H bend and ring modes |
| 1058 | -0.0102 | C-O stretch, O-H bend and ring modes |
| 1714 | -0.0094 | C=O stretch |
| 1090 | +0.0063 | C-O stretch, O-H bend and ring modes |
| 1328 | -0.0063 | C-O stretch, O-H bend and ring modes |
| 1138 | -0.0061 | C-O stretch, O-H bend and ring modes |
| 2038 | +0.0054 | between the named regions |
| 3646 | -0.0046 | O-H stretch |

What every region adds to a prediction is the sum of weight times intensity inside the region.  The first number column is its standard deviation over the acids, for the model on all acids.  The next is read out of fold: the model is fitted without the substituent of an acid, the sum of the region is computed for that acid, and the sums of all acids are correlated with the Hammett constant.  The last columns give the same correlation for 50 sets of constants in random order.

| region | from, cm^-1 | to, cm^-1 | standard deviation of what the region adds | held-out correlation of what the region adds with the Hammett constant | the same for constants in random order, mean (largest) | above the largest |
|---|---:|---:|---:|---:|---:|---|
| O-H stretch | 3450 | 3800 | 0.025 | +0.82 | -0.02 (+0.13) | yes |
| C-H stretch | 2800 | 3200 | 0.023 | +0.44 | -0.02 (+0.18) | yes |
| C=O stretch | 1650 | 1800 | 0.139 | +0.79 | +0.00 (+0.20) | yes |
| ring stretch | 1550 | 1650 | 0.102 | +0.67 | -0.02 (+0.15) | yes |
| C-O stretch, O-H bend and ring modes | 1000 | 1550 | 0.107 | +0.55 | +0.01 (+0.19) | yes |
| out-of-plane bends and ring deformations | 400 | 1000 | 0.046 | +0.20 | -0.01 (+0.16) | yes |

## Reading

**The control.** With whole substituents held out the spectrum predicts the length of the C=O bond with q2 +0.858, and the regression on the position of the carbonyl band with +0.973 (the spectrum against the regression: -0.114 [-0.154, -0.084]).  The region of the carbonyl stretch alone gives +0.895, and the spectrum without that region +0.724.  So the kernel reads the position of a line, but less exactly than a regression on the position itself: the lengths differ by a tenth of a picometre, the band moves by a few cm^-1, and the kernel sees such a shift only as a small change of the overlap of two broadened lines.

**The Hammett constant from the spectrum.** With whole substituents held out the spectrum reaches q2 +0.809.  The atom-centred counts reach +0.629, the regression on the position of the carbonyl band +0.742, the five fields +0.697 and the electrostatic field with quantum charges +0.798.  The spectrum is +0.067 [+0.015, +0.130] (above) against the regression on its one band, +0.179 [+0.086, +0.293] (above) against the counts, +0.111 [+0.023, +0.244] (above) against the five fields and +0.010 [-0.063, +0.111] (not separated) against the field with quantum charges.  With a Gaussian kernel on top it reaches +0.808 (-0.001 [-0.013, +0.013] (not separated) against the spectrum without it).

**Where the information is.** The region of the carbonyl stretch alone reaches +0.737 (-0.005 [-0.046, +0.031] (not separated) against the regression on the position of the band).  The spectrum without that region reaches +0.661 (-0.148 [-0.202, -0.101] (below) against the whole spectrum).

**Whole acids held out.** There the spectrum reaches +0.847 and the spectrum without the carbonyl region +0.735; the five fields reach +0.778.  The difference from the numbers above is what the model gains when it has seen the other acid of the same substituent.

**The spectrum beside the structure.** Joined with the five fields the q2 is +0.831 with whole substituents held out: +0.134 [+0.056, +0.253] (above) against the five fields alone and +0.022 [-0.003, +0.047] (not separated) against the spectrum alone.  Joined with the field with quantum charges it is +0.791 (-0.007 [-0.071, +0.066] (not separated) against that field alone).

**The co-spectrum.** The place with the largest effect is at 1736 cm^-1 (C=O stretch), with the sign plus: intensity at the upper side of the carbonyl band raises the predicted constant, as the band itself rises with the constant.  No single grid point is above the largest effect in models fitted to the constants in random order (0.028 against 0.057), so the co-spectrum cannot be read point by point.  It can be read by region.  With the substituent of an acid held out, what a region adds to the prediction correlates with the Hammett constant above every model on random constants in these regions: O-H stretch (+0.82), C-H stretch (+0.44), C=O stretch (+0.79), ring stretch (+0.67), C-O stretch, O-H bend and ring modes (+0.55), out-of-plane bends and ring deformations (+0.20).

**The gate of the item.** The item is a result when the structural kernel with the spectrum is above the structural kernel alone by more than the interval, or when the spectrum alone reaches the descriptor that needs no structure in three dimensions (here the atom-centred counts).  With whole substituents held out the first condition is met for the five fields and not met for the field with quantum charges, and the second condition is met.

**What the result does and does not say.** The spectrum of this benchmark is computed with a quantum program on a geometry that the program optimised.  The five fields are built on a geometry of a force field with charges from the bond graph.  The gain of the spectrum over the five fields is therefore what a quantum calculation knows about the acid, and the field with charges from one fast quantum calculation holds the same.  What the result does say is that the kernel of `cmfqsar/spectrum.py` takes a property out of a spectrum that holds it: more than a regression on the one band that a chemist would read, and as much as the best structural description of this series.  A measured spectrum would be the experiment itself and not a calculation, and no measured spectrum is used here.

