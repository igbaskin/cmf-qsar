# The spectrum as the bridge between a structure and a spectrum (plan item SP3)

## Part (b): which acid fits a spectrum

The 189 substituted benzoic acids of plan item SP2.  The library holds one infrared spectrum for every acid, from ORCA (r2SCAN-3c, wavenumbers times 0.97).  A query is the spectrum of one acid from GFN2-xTB, a faster and rougher quantum method, with its own optimised geometry.  The candidates are ranked by the kernel between the query and their library spectra.  The rank of the acid of the query is 1 when it is first.

The two levels put the lines at different places: the carbonyl band of a query is +29.3 cm^-1 from that of its library spectrum (standard deviation 4.4), and its highest line, which is the O-H stretch in most acids, is -182.0 cm^-1 from it (standard deviation 18.7).  The axis of the query is therefore mapped first: not at all, by one factor, or by one factor below 2300 cm^-1 and another above it.

All acids share the bands of benzoic acid.  In the rows with centred spectra the mean normalised spectrum of each level is taken from its spectra before they are compared, which leaves what an acid has beyond those bands.

A map with more factors fits the calibration acids better whether or not it is better.  One row therefore chooses the map inside the calibration acids: they are divided in two, every map is fitted on one half and scored on the other, and the map with the best score is taken.  This is the row that the reading uses.

The factors, the width of the broadening (one of 5, 10, 20, 50, 100, 200 cm^-1), the power of the intensity and the mean spectrum of the queries are fitted on the acids of half of the substituents, where the right answer is used, and applied to the queries of the other half.  The split is repeated 10 times; a number is the mean over the splits, with the standard deviation over the splits in brackets.

A measured-like query is the same spectrum as a curve on a grid of 4 cm^-1, with every line broadened to a standard deviation of 4 cm^-1 and with noise of 2 % of the largest value.

| how the candidates are ranked | first, % of the queries | among the first three, % | median rank | the right isomer above the other isomer of the same substituent, % | first among the candidates with the composition of the query, % | settings chosen most often |
|---|---:|---:|---:|---:|---:|---|
| by chance | 0.5 | | 95 | 50.0 | | |
| by the composition (the molecular formula) | 45.0 | | 1.5 | 50.0 | 45.0 | |
| spectrum kernel, the axis as it is | 27.0 (4.6) | 49.5 (6.4) | 3.8 | 66.2 (4.6) | 63.7 (5.4) | 100 cm^-1, power 0.5; 50 cm^-1, power 0.5 |
| spectrum kernel, the axis times one factor | 26.2 (4.5) | 49.6 (6.8) | 3.6 | 66.6 (4.1) | 64.0 (5.1) | 50 cm^-1, power 0.5, factor 0.990; 50 cm^-1, power 0.5, factor 1.000 |
| spectrum kernel, one factor below and one above 2300 cm^-1 | 23.6 (4.4) | 44.5 (5.4) | 4.5 | 59.7 (2.6) | 56.5 (3.0) | 200 cm^-1, power 0.5, factors 0.980 and 1.040; 200 cm^-1, power 0.5, factors 0.990 and 1.045 |
| centred spectra, the axis as it is | 42.8 (2.6) | 67.7 (4.2) | 2.0 | 74.0 (5.3) | 69.8 (4.5) | 100 cm^-1, power 0.5; 50 cm^-1, power 0.5 |
| centred spectra, the axis times one factor | 43.3 (2.6) | 68.8 (4.2) | 2.0 | 74.8 (5.1) | 71.0 (4.4) | 100 cm^-1, power 0.5, factor 1.000; 50 cm^-1, power 0.5, factor 0.995 |
| centred spectra, one factor below and one above 2300 cm^-1 | 36.0 (4.8) | 61.3 (6.7) | 2.4 | 75.6 (4.9) | 71.8 (4.1) | 100 cm^-1, power 0.5, factors 0.985 and 1.030; 100 cm^-1, power 0.5, factors 0.990 and 1.030 |
| centred spectra, the map of the axis chosen inside the calibration acids | 42.3 (3.2) | 66.9 (5.9) | 2.1 | 74.9 (4.8) | 70.6 (4.0) | map chosen: none in 6 of 10 splits, one factor in 3 of 10 splits, two factors in 1 of 10 splits; 50 cm^-1, power 0.5; 100 cm^-1, power 0.5 |
| the same, the query as a measured-like curve | 37.8 (6.0) | 60.1 (7.1) | 2.4 | 82.9 (5.7) | 78.1 (6.2) | map chosen: none in 7 of 10 splits, one factor in 1 of 10 splits, two factors in 2 of 10 splits; 50 cm^-1, power 1; 100 cm^-1, power 1, factors 0.995 and 1.040 |
| the same in the other direction: an ORCA spectrum as the query, the GFN2-xTB spectra as the library | 48.2 (6.8) | 74.4 (4.6) | 1.6 | 85.6 (2.3) | 81.8 (3.6) | map chosen: none in 7 of 10 splits, one factor in 3 of 10 splits; 50 cm^-1, power 0.5; 50 cm^-1, power 0.5, factor 1.005 |
| the position of the carbonyl band alone, after a linear regression between the two levels | 2.7 (1.4) | 7.8 (2.4) | 26.2 | 70.5 (4.8) |  |  |

99 % of the acids have the composition of at least one other acid of the series (the meta and the para acid of one substituent, and other isomers; the largest group has 4 acids).  The composition alone therefore puts the right acid first in 45.0 % of the cases on average, by drawing among the acids that it cannot tell apart.  The last number column is the practical case: the composition is known, and the spectrum has to choose among the isomers.

## Part (b) again: a spectrum from B97-3c as the query

The same library and the same protocol.  A query is now the spectrum of one acid from B97-3c, a second density functional, with its own optimised geometry and its own scaling factor of the wavenumbers (0.98).  The two levels put the lines at different places: the carbonyl band of a query is -4.8 cm^-1 from that of its library spectrum (standard deviation 3.0), and its highest line, which is the O-H stretch in most acids, is -26.6 cm^-1 from it (standard deviation 1.5).

| how the candidates are ranked | first, % of the queries | among the first three, % | median rank | the right isomer above the other isomer of the same substituent, % | first among the candidates with the composition of the query, % | settings chosen most often |
|---|---:|---:|---:|---:|---:|---|
| by chance | 0.5 | | 95 | 50.0 | | |
| by the composition (the molecular formula) | 45.0 | | 1.5 | 50.0 | 45.0 | |
| spectrum kernel, the axis as it is | 98.6 (1.0) | 99.7 (0.5) | 1.0 | 100.0 (0.0) | 100.0 (0.0) | 20 cm^-1, power 0.5; 10 cm^-1, power 0.5 |
| spectrum kernel, the axis times one factor | 98.6 (1.0) | 99.7 (0.5) | 1.0 | 100.0 (0.0) | 100.0 (0.0) | 20 cm^-1, power 0.5, factor 1.000; 10 cm^-1, power 0.5, factor 1.000 |
| spectrum kernel, one factor below and one above 2300 cm^-1 | 97.6 (1.8) | 99.6 (0.5) | 1.0 | 99.7 (0.5) | 99.7 (0.5) | 20 cm^-1, power 0.5, factors 1.000 and 1.000; 20 cm^-1, power 0.5, factors 0.995 and 1.000 |
| centred spectra, the axis as it is | 99.6 (0.5) | 99.7 (0.5) | 1.0 | 100.0 (0.0) | 100.0 (0.0) | 20 cm^-1, power 0.5; 10 cm^-1, power 0.5 |
| centred spectra, the axis times one factor | 99.3 (0.9) | 99.7 (0.5) | 1.0 | 100.0 (0.0) | 100.0 (0.0) | 20 cm^-1, power 0.5, factor 1.000; 10 cm^-1, power 0.5, factor 1.000 |
| centred spectra, one factor below and one above 2300 cm^-1 | 97.8 (2.0) | 99.7 (0.5) | 1.0 | 99.9 (0.3) | 99.9 (0.3) | 10 cm^-1, power 0.5, factors 1.000 and 1.000; 20 cm^-1, power 0.5, factors 0.990 and 1.000 |
| centred spectra, the map of the axis chosen inside the calibration acids | 99.6 (0.5) | 99.7 (0.5) | 1.0 | 100.0 (0.0) | 100.0 (0.0) | map chosen: none in 10 of 10 splits; 20 cm^-1, power 0.5; 10 cm^-1, power 0.5 |
| the same, the query as a measured-like curve | 96.3 (2.5) | 100.0 (0.0) | 1.0 | 99.8 (0.6) | 99.7 (0.7) | map chosen: none in 10 of 10 splits; 20 cm^-1, power 0.5; 10 cm^-1, power 0.5 |
| the same in the other direction: an ORCA spectrum as the query, the B97-3c spectra as the library | 97.8 (2.9) | 99.6 (0.5) | 1.0 | 99.2 (0.9) | 99.2 (0.9) | map chosen: none in 6 of 10 splits, one factor in 3 of 10 splits, two factors in 1 of 10 splits; 20 cm^-1, power 0.5; 50 cm^-1, power 0.5 |
| the position of the carbonyl band alone, after a linear regression between the two levels | 6.4 (3.3) | 17.2 (7.4) | 13.1 | 83.1 (6.0) |  |  |

## Part (a): which part of the structure moves which part of the spectrum

A kernel canonical correlation analysis of a structural kernel and the kernel of the ORCA spectra (width 5 cm^-1, power 1).  It finds pairs of directions, one in each description, whose values over the acids (the variates) correlate most.  The correlation on the acids that the directions were fitted on is close to 1 whatever the two descriptions hold, so a pair is judged by its correlation on held-out acids: 10 folds that hold out whole substituents.  It is compared with 100 analyses in which the pairing of the two descriptions is shuffled.  The structural kernel is the invariant local kernel around the carbonyl carbon; the width of its envelope and the ridge are the ones with the highest held-out correlation.

### the five fields, Gasteiger charges

The held-out correlation of the first pair by the width of the envelope around the carbonyl carbon: 10.00 A +0.822, 4.47 A +0.817, 2.00 A +0.614, 1.00 A +0.105.  The table is for 10.00 A.

| pair | correlation of the two variates on the acids of the fit | correlation on held-out acids (sd over the folds) | shuffled pairing: mean | shuffled pairing: 95 % | above the shuffled pairing |
|---:|---:|---:|---:|---:|---|
| 1 | +0.958 | +0.822 (0.114) | -0.003 | +0.190 | yes |
| 2 | +0.926 | +0.804 (0.073) | -0.001 | +0.202 | yes |
| 3 | +0.931 | +0.700 (0.135) | -0.017 | +0.212 | yes |
| 4 | +0.910 | +0.521 (0.161) | +0.018 | +0.183 | yes |

What the pairs follow: the correlation of the held-out variates with five properties of the acid.  The sign of a pair is free; the two variates of one pair have the same sign.

| property | pair 1: structure | pair 1: spectrum | pair 2: structure | pair 2: spectrum | pair 3: structure | pair 3: spectrum | pair 4: structure | pair 4: spectrum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| the Hammett constant | -0.79 | -0.83 | -0.11 | -0.18 | +0.15 | +0.17 | -0.08 | -0.12 |
| the position of the carbonyl band | -0.70 | -0.76 | -0.27 | -0.32 | +0.22 | +0.21 | -0.17 | -0.15 |
| para (1) or meta (0) | +0.00 | +0.03 | +0.86 | +0.84 | +0.17 | +0.21 | +0.04 | +0.01 |
| the number of heavy atoms | -0.26 | -0.22 | +0.05 | +0.06 | -0.27 | -0.26 | -0.06 | -0.02 |
| the number of lines of the spectrum | +0.35 | +0.36 | -0.01 | +0.03 | -0.22 | -0.19 | -0.02 | -0.00 |

Which region of the spectrum carries a pair: the correlation, over held-out acids, between what the region adds to the spectral variate and the structural variate of the same pair.

| region | pair 1 | pair 2 | pair 3 | pair 4 |
|---|---:|---:|---:|---:|
| O-H stretch | +0.74 | +0.15 | +0.48 | +0.26 |
| C-H stretch | +0.45 | +0.24 | +0.04 | +0.24 |
| C=O stretch | +0.74 | +0.27 | +0.53 | +0.17 |
| ring stretch | +0.44 | +0.36 | +0.32 | +0.20 |
| C-O stretch, O-H bend and ring modes | +0.48 | +0.66 | +0.57 | +0.46 |
| out-of-plane bends and ring deformations | +0.30 | +0.66 | +0.28 | +0.17 |

### the electrostatic field alone, quantum charges

The held-out correlation of the first pair by the width of the envelope around the carbonyl carbon: 10.00 A +0.700, 4.47 A +0.757, 2.00 A +0.876, 1.00 A +0.893.  The table is for 1.00 A.

| pair | correlation of the two variates on the acids of the fit | correlation on held-out acids (sd over the folds) | shuffled pairing: mean | shuffled pairing: 95 % | above the shuffled pairing |
|---:|---:|---:|---:|---:|---|
| 1 | +0.938 | +0.893 (0.047) | +0.018 | +0.169 | yes |
| 2 | +0.781 | +0.085 (0.187) | +0.010 | +0.215 | no |
| 3 | +0.791 | +0.664 (0.122) | +0.001 | +0.143 | yes |
| 4 | +0.649 | +0.324 (0.175) | +0.006 | +0.147 | yes |

What the pairs follow: the correlation of the held-out variates with five properties of the acid.  The sign of a pair is free; the two variates of one pair have the same sign.

| property | pair 1: structure | pair 1: spectrum | pair 2: structure | pair 2: spectrum | pair 3: structure | pair 3: spectrum | pair 4: structure | pair 4: spectrum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| the Hammett constant | -0.92 | -0.89 | -0.01 | -0.09 | +0.05 | +0.15 | +0.06 | -0.13 |
| the position of the carbonyl band | -0.88 | -0.88 | -0.01 | -0.26 | +0.06 | +0.23 | +0.08 | -0.12 |
| para (1) or meta (0) | +0.11 | +0.16 | +0.01 | +0.05 | -0.19 | -0.35 | +0.01 | -0.12 |
| the number of heavy atoms | -0.27 | -0.17 | -0.02 | +0.02 | -0.11 | -0.03 | +0.11 | -0.09 |
| the number of lines of the spectrum | +0.39 | +0.39 | -0.07 | +0.13 | +0.04 | +0.11 | +0.06 | +0.02 |

Which region of the spectrum carries a pair: the correlation, over held-out acids, between what the region adds to the spectral variate and the structural variate of the same pair.

| region | pair 1 | pair 2 | pair 3 | pair 4 |
|---|---:|---:|---:|---:|
| O-H stretch | +0.79 | -0.03 | +0.26 | +0.12 |
| C-H stretch | +0.46 | +0.12 | +0.10 | +0.09 |
| C=O stretch | +0.80 | +0.04 | +0.45 | +0.21 |
| ring stretch | +0.63 | +0.03 | +0.35 | +0.17 |
| C-O stretch, O-H bend and ring modes | +0.55 | +0.05 | +0.09 | +0.15 |
| out-of-plane bends and ring deformations | +0.30 | -0.16 | +0.13 | +0.02 |

## Reading

**Part (b).** With centred spectra and the map of the axis chosen inside the calibration acids, the spectrum of the second level ranks its own acid first in 42.3 % of the queries and among the first three in 66.9 %.  It puts the right isomer above the other isomer of the same substituent in 74.9 % of the cases.  When only the candidates with the composition of the query are ranked, it is first in 70.6 %; the composition alone gives 45.0 %.  Without the centring the share of first places is 27.0 % with the axis as it is and 23.6 % with two factors.  With centred spectra the three maps give 42.8 % (none), 43.3 % (one factor) and 36.0 % (two factors): the map with the most freedom is the worst on new queries.  The position of the carbonyl band alone gives 2.7 % and chance 0.5 %.  A measured-like curve gives 37.8 %, and the other direction 48.2 %.

**The width that is chosen.** The calibration chooses 50 cm^-1, power 0.5 or 100 cm^-1, power 0.5.  A width of this size compares regions of the spectrum and not single lines.  The two levels of theory do not agree line by line.  A map of the axis can move the carbonyl band and the O-H stretch onto their places, but the other lines stay tens of cm^-1 apart in a way that no smooth map repairs, and the calibration then prefers a width at which the places of single lines do not matter.

**The gate of the item.** A spectrum computed at a second level of theory has to rank its own acid first among the series in most cases, where the composition cannot tell the isomers apart.  With GFN2-xTB as the second level the gate is not met among the whole series.  Among the candidates with the composition of the query, which is the question of the isomers, it is met.  With B97-3c as the second level it is met among the whole series.

**Part (b) with a second density functional.** A spectrum from B97-3c has its carbonyl band -4.8 cm^-1 from that of the library (standard deviation 3.0; GFN2-xTB +29.3, 4.4).  With the same protocol it ranks its own acid first in 99.6 % of the queries (GFN2-xTB 42.3 %) and among the first three in 99.7 %.  It puts the right isomer above the other in 100.0 %, and it is first among the candidates with the composition of the query in 100.0 %.  The calibration chooses 20 cm^-1, power 0.5 or 10 cm^-1, power 0.5.  A measured-like curve gives 96.3 %.  For this level the gate is met among the whole series.  The share of GFN2-xTB was therefore a limit of that method and not of the kernel: with a second level that is near the first, the kernel finds the acid.

**What is needed with a near level.** Without a map of the axis and without centring the spectrum of the second functional is already first in 98.6 % of the queries.  The centring, which decided the result for GFN2-xTB, adds little here (99.6 %).  The position of the carbonyl band alone gives 6.4 %, so it is the whole spectrum that finds the acid and not one band.

**Control: the start geometry.** The spectra of the second functional start from the ORCA geometries of the library, and the GFN2-xTB spectra start from the force-field geometries.  With GFN2-xTB started from the ORCA geometries as well, the same protocol gives 43.3 % of first places (from the force-field geometries 42.3 %) and 74.9 % among the candidates with the composition of the query (70.6 %).  The start geometry is not what separates the two levels; the method is.

**Part (a), the five fields, Gasteiger charges.** The pairs above the shuffled pairing on held-out substituents are: 1, 2, 3, 4 of the first 4.  Pair 1 has a held-out correlation of +0.822.  Of the five properties, its two variates follow the Hammett constant most (r -0.79 for the structure and -0.83 for the spectrum), and the regions of the spectrum that carry it most are: O-H stretch (+0.74) and C=O stretch (+0.74).  Pair 2 has a held-out correlation of +0.804.  Of the five properties, its two variates follow para (1) or meta (0) most (r +0.86 for the structure and +0.84 for the spectrum), and the regions of the spectrum that carry it most are: C-O stretch, O-H bend and ring modes (+0.66) and out-of-plane bends and ring deformations (+0.66).  The held-out correlation of the first pair is +0.105 with an envelope of 1.00 A around the carbonyl carbon and +0.822 with 10.00 A.

**Part (a), the electrostatic field alone, quantum charges.** The pairs above the shuffled pairing on held-out substituents are: 1, 3, 4 of the first 4.  Pair 1 has a held-out correlation of +0.893.  Of the five properties, its two variates follow the Hammett constant most (r -0.92 for the structure and -0.89 for the spectrum), and the regions of the spectrum that carry it most are: C=O stretch (+0.80) and O-H stretch (+0.79).  Pair 3 has a held-out correlation of +0.664.  Its two variates follow none of the five properties closely; the largest correlation is with para (1) or meta (0) (r -0.19 for the structure and -0.35 for the spectrum), and the regions of the spectrum that carry it most are: C=O stretch (+0.45) and ring stretch (+0.35).  The held-out correlation of the first pair is +0.893 with an envelope of 1.00 A around the carbonyl carbon and +0.700 with 10.00 A.

**Where the first pair lives.** The plan expected the first structural direction on the neighbourhood of the carbonyl group and the first spectral direction on the carbonyl band.  On the spectral side the first pair is carried by the carbonyl stretch and by the O-H stretch about equally.  On the structural side the answer depends on the charges.  With quantum charges the pair is strongest with the narrowest envelope (1.00 A: +0.893), which holds little beyond the carboxyl group: the quantum calculation has carried the effect of the substituent to that group.  With Gasteiger charges, which are the same on the carboxyl group of every acid, the pair vanishes there (+0.105) and needs an envelope that reaches the substituent.  This is the finding of plan item SP2, reached here without a target.

