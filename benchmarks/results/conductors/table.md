# From DFT barriers to the activation energy of a conductor

The model: a random forest on the invariant field about the midpoint of a hop (4301 columns), fitted on 1652 DFT barriers of 852 structures (nebDFT2k). On folds that hold out whole structures its mean absolute error is 0.228 eV. The spread of its trees is multiplied by 1.07 to have the right size (96 % of the barriers inside two standard deviations), and 46 % of the squared error is common to the hops of one structure.

The conductors: 6408 hops in 13 structures, 298 distinct by symmetry. Elements without a field of their own in the training set: Ge, La, N, Zr. The predicted barriers have the mean 1.21 eV and a mean standard deviation of 0.50 eV.

Does the model carry over? No DFT barrier of a conductor is known, so this cannot be measured directly. One check costs nothing: the predictions should correlate with the bond-valence barriers of the same hops as they do inside the training set. Inside the training set that correlation is r = +0.84 (predictions for structures that the fit has not seen). For the conductors it is r = +0.09.

A percolation barrier is the lowest level that an ion has to reach to cross the cell. An interval is the 95 % interval over 400 draws of the barriers; the number before it is the median. All values in eV.

| Structure | Compound | Hops (distinct) | Share of hops shorter than any training hop | From the predicted barriers as they are: one / three directions | With the errors drawn: one direction | With the errors drawn: all three directions | Bond-valence barriers: one / three directions | Measured activation energy |
|---|---|---|---|---|---|---|---|---|
| cod_1545083 | LLZO:Ta cubic (Li6.5La3Zr1.5Ta0.5O12) | 1728 (18) | 11 % | 0.99 / 0.99 | 0.63 [0.00, 1.43] | 0.63 [0.00, 1.43] | 0.40 / 0.40 | 0.41 to 0.47 |
| cod_1552152 | LLZO:Ta cubic (Li6.2La3Zr1.2Ta0.8O12) | 1824 (19) | 11 % | 0.81 / 0.81 | 0.66 [0.00, 1.32] | 0.66 [0.00, 1.32] | 0.39 / 0.39 | 0.41 to 0.47 |
| cod_1545085 | LLZO:Ta tetragonal | 416 (13) | 0 % | 0.76 / 1.03 | 0.51 [0.00, 1.32] | 0.80 [0.00, 1.71] | 1.45 / 2.93 |  |
| cod_1525568 | LLTO rhombohedral (La0.5Li0.5TiO3) | 252 (7) | 0 % | 2.52 / 2.53 | 1.84 [0.66, 3.20] | 2.20 [1.05, 3.45] | 1.76 / 1.89 |  |
| cod_1000437 | LLTO tetragonal (La0.56Li0.33TiO3) | 12 (4) | 0 % | 0.70 / 0.82 | 0.49 [0.00, 1.24] | 0.79 [0.03, 1.52] | 0.82 / 15.17 | 0.40 |
| cod_7215190 | LGPS (Li10GeP2S12) | 416 (26) | 4 % | 0.25 / 0.32 | 0.11 [0.00, 0.41] | 0.21 [0.00, 0.61] | 0.53 / 0.59 | 0.22 to 0.28 |
| cod_7222155 | LiTi2(PO4)3 NASICON | 0 | | no hop | no hop | no hop | |  |
| cod_7222156 | Li3.18Ti2(PO4)3 NASICON | 216 (12) | 17 % | 0.66 / 0.66 | 0.52 [0.00, 1.17] | 0.56 [0.04, 1.20] | 0.47 / 0.47 |  |
| cod_4002499 | Li2.92N | 72 (9) | 0 % | 0.39 / 1.00 | 0.32 [0.00, 1.23] | 0.73 [0.00, 1.55] | 0.16 / 0.17 | 0.25 |
| cod_1524557 | LISICON (Li13.7Zn1.2Ge4O16) | 416 (55) | 0 % | 0.60 / 0.76 | 0.32 [0.00, 0.82] | 0.50 [0.00, 1.12] | 0.45 / 0.49 |  |
| cod_9011044 | Li3PO4 (Pmnb) | 144 (18) | 0 % | 0.51 / 0.51 | 0.33 [0.00, 0.66] | 0.49 [0.14, 0.77] | 0.31 / 0.63 |  |
| cod_1570310 | beta-Li3PS4 | 152 (19) | 0 % | 0.28 / 0.37 | 0.07 [0.00, 0.52] | 0.33 [0.00, 0.77] | 0.25 / 0.40 | 0.36 |
| cod_1529523 | Li3.16Zn0.24GeO4 | 760 (98) | 8 % | 0.60 / 0.67 | 0.28 [0.00, 0.73] | 0.44 [0.00, 0.91] | 0.40 / 0.46 |  |

## Against the measured activation energies

6 structures have a measured value. The mean absolute difference between the percolation barrier and the middle of the measured range is 0.24 eV (one direction) and 0.36 eV (three directions) for the predicted barriers as they are, and 0.17 and 0.23 eV for the medians of the draws. A prediction of the mean of the measured values for every structure would give 0.07 eV. For comparison, the bond-valence barriers give 0.16 and 2.55 eV (6 structures).
The interval of the forest, from its lower end for one direction to its upper end for three directions, meets the measured range in 6 of the 6 structures. The interval for one direction is 1.03 eV wide on average.

| Structure | Measured, eV | What was measured | Source | How far the source was checked |
|---|---|---|---|---|
| cod_1545083 (LLZO:Ta cubic (Li6.5La3Zr1.5Ta0.5O12)) | 0.41 to 0.47 | single crystals of Ta-doped cubic garnet | a search summary of single-crystal studies of Li6.5La3Zr1.5Ta0.5O12; pellets of the same compound: 0.41 eV | search result |
| cod_1552152 (LLZO:Ta cubic (Li6.2La3Zr1.2Ta0.8O12)) | 0.41 to 0.47 | the same family, another Ta content | as for cod_1545083; not measured on this composition | search result |
| cod_1000437 (LLTO tetragonal (La0.56Li0.33TiO3)) | 0.40 | bulk part of a polycrystal, impedance | S. Stramare, V. Thangadurai and W. Weppner, Chem. Mater. 15, 3974 (2003), on Li0.34La0.51TiO2.94 (Y. Inaguma et al., Solid State Commun. 86, 689 (1993)) | search result |
| cod_7215190 (LGPS (Li10GeP2S12)) | 0.22 to 0.28 | polycrystals, impedance | R. Iwasaki et al., ISSP Activity Report 2018, on the earlier measurements (N. Kamaya et al., Nat. Mater. 10, 682 (2011) and later) | page read |
| cod_4002499 (Li2.92N) | 0.25 | single crystal, in the Li2N planes | U. v. Alpen, A. Rabenau and G. H. Talat, Appl. Phys. Lett. 30, 621 (1977) | search result |
| cod_1570310 (beta-Li3PS4) | 0.36 | recrystallised pellets, impedance | US patent 8,597,838 of the group of Z. Liu et al. (J. Am. Chem. Soc. 135, 975 (2013)) | search result |

## How the error of one barrier moves the result

The standard deviations of the barriers as they are, and made smaller. The width is that of the 95 % interval of the percolation barrier in one direction, as the mean over the structures.

| Standard deviation of a barrier, mean, eV | Width of the interval of the percolation barrier, eV |
|---|---|
| 0.503 (the model) | 1.12 |
| 0.251 | 0.66 |
| 0.126 | 0.36 |
| 0.050 | 0.15 |

