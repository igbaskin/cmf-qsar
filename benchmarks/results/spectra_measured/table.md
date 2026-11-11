# A measured infrared spectrum against other measured spectra of the same compounds

Set: Chemotion IR (https://doi.org/10.22000/OGoEQGlsZGElrgst, CC BY-SA 4.0), fetched by `benchmarks/prepare_chemotion_ir.py`; no file of the set is in this repository.
After the copies were left out: 2079 measured spectra of 1912 compounds. 138 compounds have two or more spectra of different samples (120 with 2, 13 with 3, 1 with 4, 2 with 5, 2 with 6); they give 430 pairs of a query and a library spectrum.
Left out: 19 curves that are a copy of another curve of the same compound, and 16 curves that are 8 files, each attached to the samples of two different compounds.
The library holds one spectrum of every compound. A compound counts once, whatever its number of spectra. The form of the kernel and the width are chosen on half of the compounds with several spectra and scored on the other half, 50 random splits; the number after the sign +/- is the standard deviation over the splits.

## How often the compound of the query is found

| Description of the spectrum | Compound is first | In the first three | In the first ten |
|---|---|---|---|
| chance (one of 1912 compounds) | 0.1 % | 0.2 % | 0.5 % |
| plain kernel of the absorbance, width chosen | 59.9 % +/- 3.4 | 71.1 % +/- 3.2 | 78.7 % +/- 2.7 |
| the mean of every spectrum taken off (the correlation coefficient of two spectra), width chosen | 77.2 % +/- 3.5 | 84.7 % +/- 3.0 | 90.0 % +/- 2.5 |
| kernel of the slopes of the spectra, width chosen | 87.6 % +/- 2.5 | 93.2 % +/- 2.1 | 95.4 % +/- 1.9 |
| form and width chosen | 87.6 % +/- 2.5 | 93.2 % +/- 2.1 | 95.4 % +/- 1.9 |
| the chosen form on 700 to 1800 cm^-1 alone (all compounds, no choice) | 87.9 % | 92.8 % | 95.2 % |
| the chosen form on 1800 to 3850 cm^-1 alone (all compounds, no choice) | 58.6 % | 66.1 % | 70.2 % |

Chosen settings over the splits: slope, width 4 cm^-1 in 40; slope, width 8 cm^-1 in 7; slope, not broadened in 2; slope, width 16 cm^-1 in 1.

Difference in the share of first places, per compound, with a 95 % bootstrap interval over the compounds:

- chosen form minus the form 'plain': +27.9 [+21.8, +34.4] points
- chosen form minus the form 'own mean removed': +10.3 [+5.7, +14.9] points
- the choice among all forms took the form 'slope' in every split, so the two rows are the same

## Every setting on all compounds (no choice, for the reader)

| Form and width | Compound is first | In the first three | In the first ten |
|---|---|---|---|
| plain, not broadened | 59.4 % | 70.9 % | 78.7 % |
| own mean removed, not broadened | 77.1 % | 84.8 % | 89.9 % |
| slope, not broadened | 86.6 % | 94.1 % | 94.4 % |
| plain, width 4 cm^-1 | 55.5 % | 67.7 % | 76.0 % |
| own mean removed, width 4 cm^-1 | 73.0 % | 83.9 % | 88.7 % |
| slope, width 4 cm^-1 | 88.0 % | 93.4 % | 95.5 % |
| plain, width 8 cm^-1 | 51.4 % | 62.2 % | 73.7 % |
| own mean removed, width 8 cm^-1 | 66.6 % | 79.0 % | 88.4 % |
| slope, width 8 cm^-1 | 86.9 % | 92.6 % | 95.4 % |
| plain, width 16 cm^-1 | 44.2 % | 56.9 % | 67.1 % |
| own mean removed, width 16 cm^-1 | 56.4 % | 71.9 % | 83.3 % |
| slope, width 16 cm^-1 | 83.9 % | 92.4 % | 95.7 % |

## What the numbers say about the set and about the kernel

- Two spectra of one compound from different samples have a correlation coefficient of 0.943 (median); a tenth of the pairs is below 0.784.
- The molecular formula alone would name the compound in 82.1 % of the queries: the compounds of the set are diverse, and few share a formula. The formula is not known from a spectrum, so this is a statement about the library and not a rival of the kernel.
- Over the whole library (1909 compounds), the other compound with the most similar spectrum has a Tanimoto similarity of 0.53 on average (median 0.58) and of 0.5 or more in 59.6 %; a pairing at random gives 0.10 and 0.5 %.
- When the kernel (slope, width 4 cm^-1, all compounds) names another compound (76 pairs), the Tanimoto similarity of that compound to the right one is 0.59 on average (median 0.46), against 0.10 for a compound drawn at random. In 42.1 % of these pairs the similarity is 0.5 or more, and in 26.3 % the named compound has the same formula (an isomer).
- The 15 pairs in which the compound is not among the first 10 belong to 7 compounds. The two spectra of such a pair have a correlation coefficient of 0.49 (median; the highest is 0.72), against 0.94 for all pairs: the two samples did not give the same spectrum, whatever the reason.

## Reading

A measured spectrum finds its own compound among 1912 compounds in 87.6 % of the queries and puts it into the first ten in 95.4 %. This is the ceiling for a query against a computed library of the same compounds: there the difference between a calculation and a measurement is added to the difference between two samples.
The plain kernel finds 59.9 %. The difference is +27.9 points [+21.8, +34.4], so the form of the kernel matters on measured spectra: a measured curve has a baseline, and the plain kernel counts it as intensity.

