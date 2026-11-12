# A measured infrared spectrum against a library of computed spectra

Queries: 1782 measured spectra of 1635 compounds of the Chemotion IR set (https://doi.org/10.22000/OGoEQGlsZGElrgst, CC BY-SA 4.0; no file of the set is in this repository).
Library: 1645 computed spectra, one per candidate compound (GFN2-xTB acc=1 ohess, one conformer, one molecule in the gas phase; `benchmarks/prepare_chemotion_xtb.py`). 227 of them keep an imaginary mode.
Compounds without a computed spectrum: 133: more than one molecule (a salt, a solvate or a complex); 49: an element outside the list (a metal in most cases); 4: unpaired electrons; 92: more than 100 atoms; 1: an ion; 1: the SMILES is not read; 0 failed in the calculation.
The form of the kernel, the width, a factor on the computed wavenumbers, the region, the power of the intensities, their division by the wavenumber and the centring (1680 settings) are chosen on half of the compounds and scored on the other half, 20 random splits. A compound counts once. The number after the sign +/- is the standard deviation over the splits.

## How often the compound of the query is found

| What is compared | Compound is first | In the first three | In the first ten |
|---|---|---|---|
| chance (one of 1645 compounds) | 0.1 % | 0.2 % | 0.6 % |
| all settings open to the choice | 5.6 % +/- 0.6 | 11.9 % +/- 0.8 | 24.5 % +/- 1.3 |
| only the kernel of the slopes | 5.7 % +/- 0.7 | 11.9 % +/- 0.8 | 24.6 % +/- 1.4 |
| only the spectra with their mean taken off | 6.0 % +/- 0.7 | 12.6 % +/- 1.1 | 24.3 % +/- 1.1 |
| only the plain kernel | 4.3 % +/- 0.5 | 10.2 % +/- 1.0 | 21.6 % +/- 1.2 |
| only the whole range, 700 to 3850 cm^-1 | 5.6 % +/- 0.6 | 11.9 % +/- 0.8 | 24.5 % +/- 1.3 |
| only the region 700 to 1800 cm^-1 | 2.8 % +/- 0.6 | 7.0 % +/- 0.6 | 16.7 % +/- 1.1 |
| for comparison: a measured spectrum of another sample as the library spectrum, among 1912 compounds (`spectra_measured`) | 87.6 % | | 95.4 % |

Chosen settings over the splits: slope, width 64 cm^-1, wavenumbers times 0.97, the whole range, intensities, computed intensities over the wavenumber, centred in 11; own mean removed, width 32 cm^-1, wavenumbers times 0.97, the whole range, intensities, computed intensities over the wavenumber, centred in 4; slope, width 64 cm^-1, wavenumbers times 0.97, the whole range, square root of the intensities, computed intensities over the wavenumber, centred in 3; own mean removed, width 32 cm^-1, wavenumbers times 0.98, the whole range, intensities, computed intensities over the wavenumber, centred in 1; slope, width 64 cm^-1, wavenumbers times 0.98, the whole range, intensities, computed intensities over the wavenumber, centred in 1.

At the setting chosen most often, on all compounds: first in 6.5 %, in the first ten in 26.2 %, in the best one per cent of the library in 31.4 %; the median rank of the right compound is 56 of 1645.

Difference in the share of first places, per compound, with a 95 % bootstrap interval over the compounds:

- the open choice minus 'only the kernel of the slopes': -0.1 [-0.4, +0.2] points
- the open choice minus 'only the spectra with their mean taken off': -0.3 [-1.2, +0.5] points
- the open choice minus 'only the plain kernel': +1.5 [+0.7, +2.4] points
- the open choice is the same as 'only the whole range, 700 to 3850 cm^-1' in every split
- the open choice minus 'only the region 700 to 1800 cm^-1': +3.0 [+2.2, +3.7] points

## What every option is worth (all compounds, no choice, for the reader)

The best share of first places over all settings that have the value.

| Option | Value | Compound is first |
|---|---|---|
| form of the kernel | plain | 4.9 % |
| form of the kernel | own mean removed | 6.3 % |
| form of the kernel | slope | 6.5 % |
| width, cm^-1 | 4.0 | 3.1 % |
| width, cm^-1 | 8.0 | 3.8 % |
| width, cm^-1 | 16.0 | 4.9 % |
| width, cm^-1 | 32.0 | 6.3 % |
| width, cm^-1 | 64.0 | 6.5 % |
| factor on the computed wavenumbers | 0.95 | 3.7 % |
| factor on the computed wavenumbers | 0.96 | 4.6 % |
| factor on the computed wavenumbers | 0.97 | 6.5 % |
| factor on the computed wavenumbers | 0.98 | 6.1 % |
| factor on the computed wavenumbers | 0.99 | 4.3 % |
| factor on the computed wavenumbers | 1.0 | 2.8 % |
| factor on the computed wavenumbers | 1.01 | 1.5 % |
| region | whole | 6.5 % |
| region | 700-1800 | 3.6 % |
| power of the intensities | 1.0 | 6.5 % |
| power of the intensities | 0.5 | 5.8 % |
| centred | False | 5.3 % |
| centred | True | 6.5 % |
| computed intensities over the wavenumber | False | 5.2 % |
| computed intensities over the wavenumber | True | 6.5 % |

## Which compounds are found (the setting chosen most often, all compounds)

| Compounds | Number | Compound is first | In the first ten |
|---|---|---|---|
| no hydrogen-bond donor | 962 | 7.6 % | 29.5 % |
| one or more hydrogen-bond donors | 673 | 4.8 % | 21.5 % |
| at most 2 rotatable bonds | 575 | 7.0 % | 26.6 % |
| 3 to 5 rotatable bonds | 626 | 6.1 % | 29.7 % |
| 6 or more rotatable bonds | 434 | 6.3 % | 20.6 % |
| at most 40 atoms | 811 | 5.4 % | 25.3 % |
| 41 to 70 atoms | 681 | 7.8 % | 27.1 % |
| 71 or more atoms | 143 | 5.6 % | 26.9 % |
| the computed geometry keeps an imaginary mode | 226 | 3.5 % | 20.8 % |

When the kernel names another compound (1666 queries), the Tanimoto similarity of that compound to the right one is 0.18 on average (median 0.12), against 0.11 for a compound drawn at random. In 7.3 % of these queries the similarity is 0.5 or more, and in 0.9 % the named compound has the same formula (an isomer).

## Two levels of theory on the small compounds

The 230 compounds with at most 25 atoms that have a spectrum at both levels are the candidates; 255 measured spectra of 228 of them are the queries. Both levels start from the same conformer. The settings are chosen for each level on half of the compounds and scored on the other half, the same 20 splits for both.

| Library | Compound is first | In the first three | In the first ten | Median rank of the right compound | Width chosen most often |
|---|---|---|---|---|---|
| chance (one of 230 compounds) | 0.4 % | 1.3 % | 4.3 % | 115.5 | |
| GFN2-xTB acc=1 ohess | 11.3 % +/- 2.4 | 22.3 % +/- 3.7 | 37.2 % +/- 3.4 | 20 | 64 cm^-1 |
| B97-3c TightSCF | 32.3 % +/- 3.4 | 51.4 % +/- 3.5 | 71.9 % +/- 4.2 | 3 | 32 cm^-1 |

Difference in the share of first places, B97-3c TightSCF minus GFN2-xTB acc=1 ohess, per compound: +21.2 [+16.5, +26.2] points (95 % bootstrap interval over the compounds).

- GFN2-xTB acc=1 ohess: chosen most often: slope, width 64 cm^-1, wavenumbers times 0.97, the whole range, square root of the intensities, computed intensities over the wavenumber, centred. Best share of first places by width, all compounds: 4 cm^-1 8.6 %, 8 cm^-1 11.5 %, 16 cm^-1 12.7 %, 32 cm^-1 12.9 %, 64 cm^-1 14.8 %.
- B97-3c TightSCF: chosen most often: slope, width 32 cm^-1, wavenumbers times 0.97, 700 to 1800 cm^-1, square root of the intensities, centred. Best share of first places by width, all compounds: 4 cm^-1 36.5 %, 8 cm^-1 36.4 %, 16 cm^-1 36.2 %, 32 cm^-1 37.4 %, 64 cm^-1 34.8 %.

Groups of compounds, each level at the setting chosen most often (all compounds):

| Compounds | Number | First, GFN2-xTB acc=1 ohess | First, B97-3c TightSCF | In the first ten, GFN2-xTB acc=1 ohess | In the first ten, B97-3c TightSCF |
|---|---|---|---|---|---|
| no hydrogen-bond donor | 112 | 18.0 % | 46.1 % | 43.8 % | 75.9 % |
| one or more hydrogen-bond donors | 116 | 11.6 % | 28.9 % | 33.2 % | 63.8 % |
| at most 2 rotatable bonds | 188 | 13.1 % | 35.7 % | 36.4 % | 67.0 % |
| 3 to 5 rotatable bonds | 40 | 22.5 % | 45.0 % | 47.5 % | 82.5 % |

Of the 228 compounds, 23 have a first place at both levels, 64 only at B97-3c TightSCF, 12 only at GFN2-xTB acc=1 ohess and 129 at neither. For a place among the first ten the four numbers are 73, 87, 15 and 53. A compound with several measured spectra is counted when one of them is found.

