# Can the program tell two enantiomers apart?

Plan item MC3, on 3834 enantiomeric pairs (7668 molecules) of 54 targets, built by `examples/make_enantiomer_examples.py` from ChEMBL. Both potencies of a pair come from one assay. The two members of a pair are exact reflections of one conformation, so the two-point description gives them identical rows: the largest difference found between the rows of a pair is 0.0e+00. A model without handedness channels therefore cannot rank a pair, and its share of correctly ranked pairs is one half by construction, not by chance.

The mean of each pair is taken off its two potencies, so what is left is half the difference between the two enantiomers. The share below is how often the better enantiomer of a held-out pair is named correctly, with a 95 % interval in brackets; a share whose interval holds 50 % is no better than naming one of the two at random.

**The two fold rules answer two different questions.** With pairs held out, the two members of a held-out pair are not in the training part but other pairs of the same scaffold may be, so the question is whether a new pair of a series the model already knows can be ranked. With scaffolds held out, every molecule of a scaffold leaves together, so the question is whether a pair of a series the model has never seen can be ranked. The second is the harder question and the one the alignment-free kernel exists for. The two give similar answers here, which says the signal is not mostly read off a near neighbour of the held-out pair.

A pair whose two potencies are equal has no better enantiomer and is left out of the share; there are 130 such pairs. A model that gives the two members of a pair the same number names no better enantiomer either, and such a pair counts as half a success, which is what it is worth.

## The difference inside a pair, with pairs held out

Every target is fitted on its own. The last row counts the held-out pairs of all of them together.

| Target | Pairs | Scaffolds | every centre | stereocentres only | every other centre | no handedness channels |
|---|---|---|---|---|---|---|
| 11-beta-hydroxysteroid dehydrogenase 1 | 26 | 19 | 53.8 % (35 to 71) | 46.2 % (29 to 65) | 61.5 % (43 to 78) | 50.0 % (32 to 68) |
| PI3-kinase alpha | 139 | 88 | 60.2 % (52 to 68) | 54.9 % (46 to 63) | 57.9 % (49 to 66) | 50.0 % (42 to 58) |
| PIM1 kinase | 177 | 85 | 50.6 % (43 to 58) | 52.9 % (45 to 60) | 52.9 % (45 to 60) | 50.0 % (43 to 57) |
| PPAR gamma | 101 | 65 | 72.0 % (63 to 80) | 70.5 % (61 to 79) | 59.0 % (49 to 68) | 50.0 % (40 to 60) |
| acetylcholinesterase | 28 | 26 | 46.4 % (30 to 64) | 66.1 % (48 to 81) | 42.9 % (27 to 61) | 50.0 % (33 to 67) |
| alpha-1A adrenergic receptor | 44 | 32 | 74.4 % (60 to 85) | 76.7 % (62 to 87) | 55.8 % (41 to 70) | 50.0 % (36 to 64) |
| alpha-2A adrenergic receptor | 16 | 12 | 18.8 % (7 to 43) | 62.5 % (39 to 82) | 6.2 % (1 to 28) | 50.0 % (28 to 72) |
| androgen receptor | 43 | 29 | 60.5 % (46 to 74) | 46.5 % (33 to 61) | 48.8 % (35 to 63) | 50.0 % (36 to 64) |
| beta-2 adrenergic receptor | 39 | 18 | 92.3 % (80 to 97) | 93.6 % (81 to 98) | 59.0 % (43 to 73) | 50.0 % (35 to 65) |
| beta-secretase 1 | 142 | 77 | 56.0 % (48 to 64) | 62.4 % (54 to 70) | 53.9 % (46 to 62) | 50.0 % (42 to 58) |
| butyrylcholinesterase | 27 | 23 | 51.9 % (34 to 69) | 66.7 % (48 to 81) | 37.0 % (22 to 56) | 50.0 % (32 to 68) |
| cannabinoid CB1 receptor | 55 | 38 | 49.1 % (36 to 62) | 61.8 % (49 to 73) | 36.4 % (25 to 50) | 50.0 % (37 to 63) |
| cannabinoid CB2 receptor | 61 | 44 | 36.1 % (25 to 49) | 46.7 % (35 to 59) | 31.1 % (21 to 44) | 50.0 % (38 to 62) |
| carbonic anhydrase 2 | 58 | 28 | 53.4 % (41 to 66) | 47.4 % (35 to 60) | 51.7 % (39 to 64) | 50.0 % (38 to 62) |
| coagulation factor Xa | 52 | 40 | 50.0 % (37 to 63) | 64.4 % (51 to 76) | 48.1 % (35 to 61) | 50.0 % (37 to 63) |
| cyclin-dependent kinase 2 | 44 | 30 | 45.2 % (31 to 60) | 51.2 % (37 to 66) | 42.9 % (29 to 58) | 50.0 % (36 to 64) |
| cyclooxygenase 2 | 43 | 15 | 58.1 % (43 to 72) | 60.5 % (46 to 74) | 55.8 % (41 to 70) | 50.0 % (36 to 64) |
| delta opioid receptor | 32 | 19 | 58.1 % (41 to 74) | 53.2 % (36 to 69) | 48.4 % (32 to 65) | 50.0 % (33 to 67) |
| dipeptidyl peptidase 4 | 33 | 19 | 60.0 % (42 to 75) | 61.7 % (44 to 77) | 50.0 % (33 to 67) | 50.0 % (33 to 67) |
| dopamine D2 receptor | 234 | 146 | 47.2 % (41 to 54) | 54.5 % (48 to 61) | 45.9 % (40 to 52) | 50.0 % (44 to 56) |
| dopamine D3 receptor | 163 | 87 | 51.5 % (44 to 59) | 62.3 % (55 to 69) | 52.8 % (45 to 60) | 50.0 % (42 to 58) |
| dopamine D4 receptor | 62 | 49 | 50.0 % (38 to 62) | 45.2 % (33 to 57) | 54.8 % (43 to 67) | 50.0 % (38 to 62) |
| dopamine transporter | 147 | 68 | 45.8 % (38 to 54) | 52.8 % (45 to 61) | 44.4 % (37 to 53) | 50.0 % (42 to 58) |
| epidermal growth factor receptor | 111 | 82 | 50.5 % (41 to 60) | 52.0 % (42 to 62) | 52.5 % (43 to 62) | 50.0 % (40 to 60) |
| fatty acid amide hydrolase | 54 | 30 | 88.9 % (78 to 95) | 82.4 % (70 to 90) | 37.0 % (25 to 50) | 50.0 % (37 to 63) |
| glucocorticoid receptor | 20 | 16 | 57.9 % (36 to 77) | 52.6 % (32 to 73) | 57.9 % (36 to 77) | 50.0 % (29 to 71) |
| hERG potassium channel | 143 | 106 | 47.1 % (39 to 55) | 47.1 % (39 to 55) | 48.5 % (40 to 57) | 50.0 % (42 to 58) |
| histamine H1 receptor | 30 | 20 | 56.7 % (39 to 73) | 55.0 % (38 to 71) | 50.0 % (33 to 67) | 50.0 % (33 to 67) |
| histamine H3 receptor | 120 | 67 | 46.2 % (37 to 55) | 52.6 % (44 to 61) | 42.7 % (34 to 52) | 50.0 % (41 to 59) |
| kappa opioid receptor | 100 | 49 | 64.1 % (53 to 74) | 61.5 % (50 to 72) | 57.7 % (47 to 68) | 50.0 % (39 to 61) |
| monoamine oxidase A | 33 | 17 | 48.4 % (32 to 65) | 48.4 % (32 to 65) | 54.8 % (38 to 71) | 50.0 % (33 to 67) |
| monoamine oxidase B | 41 | 20 | 50.0 % (35 to 65) | 52.6 % (37 to 68) | 42.1 % (28 to 58) | 50.0 % (35 to 65) |
| mu opioid receptor | 77 | 47 | 42.4 % (31 to 54) | 48.5 % (37 to 60) | 42.4 % (31 to 54) | 50.0 % (38 to 62) |
| muscarinic M1 receptor | 37 | 24 | 23.5 % (12 to 40) | 48.5 % (33 to 65) | 20.6 % (10 to 37) | 50.0 % (34 to 66) |
| muscarinic M2 receptor | 41 | 26 | 39.5 % (26 to 55) | 51.3 % (36 to 66) | 42.1 % (28 to 58) | 50.0 % (35 to 65) |
| muscarinic M3 receptor | 53 | 19 | 39.2 % (27 to 53) | 70.6 % (57 to 81) | 37.3 % (25 to 51) | 50.0 % (37 to 63) |
| neurokinin NK1 receptor | 58 | 38 | 48.3 % (36 to 61) | 51.7 % (39 to 64) | 50.0 % (38 to 62) | 50.0 % (38 to 62) |
| noradrenaline transporter | 192 | 87 | 48.4 % (41 to 56) | 52.2 % (45 to 59) | 44.5 % (37 to 52) | 50.0 % (43 to 57) |
| oestrogen receptor alpha | 75 | 40 | 37.3 % (27 to 49) | 42.7 % (32 to 54) | 57.3 % (46 to 68) | 50.0 % (39 to 61) |
| phosphodiesterase 4D | 23 | 17 | 69.6 % (49 to 84) | 60.9 % (41 to 78) | 43.5 % (26 to 63) | 50.0 % (31 to 69) |
| phosphodiesterase 5A | 58 | 29 | 60.7 % (48 to 72) | 44.6 % (32 to 58) | 60.7 % (48 to 72) | 50.0 % (37 to 63) |
| protein kinase B | 46 | 28 | 57.8 % (43 to 71) | 52.2 % (38 to 66) | 71.1 % (57 to 82) | 50.0 % (36 to 64) |
| renin | 22 | 18 | 45.5 % (27 to 65) | 29.5 % (15 to 50) | 50.0 % (31 to 69) | 50.0 % (31 to 69) |
| rho-associated kinase 1 | 56 | 38 | 54.7 % (41 to 67) | 46.2 % (34 to 59) | 52.8 % (40 to 66) | 50.0 % (37 to 63) |
| serotonin 5-HT1A receptor | 83 | 53 | 34.6 % (25 to 45) | 44.4 % (34 to 55) | 49.4 % (39 to 60) | 50.0 % (39 to 61) |
| serotonin 5-HT2A receptor | 114 | 69 | 43.3 % (34 to 53) | 48.6 % (39 to 58) | 52.9 % (43 to 62) | 50.0 % (41 to 59) |
| serotonin 5-HT2C receptor | 72 | 44 | 62.9 % (51 to 73) | 67.1 % (56 to 77) | 48.6 % (37 to 60) | 50.0 % (39 to 61) |
| serotonin 5-HT6 receptor | 83 | 50 | 59.3 % (48 to 69) | 65.4 % (55 to 75) | 58.0 % (47 to 68) | 50.0 % (39 to 61) |
| serotonin 5-HT7 receptor | 65 | 44 | 33.3 % (23 to 46) | 37.3 % (26 to 50) | 39.7 % (29 to 52) | 50.0 % (38 to 62) |
| serotonin transporter | 201 | 94 | 50.8 % (44 to 58) | 54.0 % (47 to 61) | 52.8 % (46 to 60) | 50.0 % (43 to 57) |
| thrombin | 23 | 17 | 50.0 % (31 to 69) | 63.6 % (43 to 80) | 50.0 % (31 to 69) | 50.0 % (31 to 69) |
| **every target, each fitted on its own** | 3667 | | **51.5 %** (50 to 53) | **55.2 %** (54 to 57) | **49.5 %** (48 to 51) | **50.0 %** (48 to 52) |

As q2 of the half-difference (at or below zero means the model predicts the difference no better than predicting zero for every pair):

| Target | every centre | stereocentres only | every other centre | no handedness channels |
|---|---|---|---|---|
| 11-beta-hydroxysteroid dehydrogenase 1 | -0.048 | -0.021 | -0.044 | -0.000 |
| PI3-kinase alpha | +0.009 | +0.090 | -0.037 | -0.000 |
| PIM1 kinase | -0.024 | +0.109 | -0.001 | -0.000 |
| PPAR gamma | +0.048 | +0.138 | -0.018 | +0.000 |
| acetylcholinesterase | -0.137 | +0.459 | -0.007 | +0.000 |
| alpha-1A adrenergic receptor | +0.115 | +0.231 | -0.182 | +0.000 |
| alpha-2A adrenergic receptor | -1.416 | -0.074 | -1.513 | +0.000 |
| androgen receptor | -0.116 | -0.178 | -0.341 | -0.000 |
| beta-2 adrenergic receptor | +0.397 | +0.565 | -0.211 | +0.000 |
| beta-secretase 1 | +0.075 | +0.134 | -0.000 | -0.000 |
| butyrylcholinesterase | -0.053 | -2.637 | -0.583 | +0.000 |
| cannabinoid CB1 receptor | -0.035 | +0.135 | -0.038 | +0.000 |
| cannabinoid CB2 receptor | -0.027 | -0.010 | -0.009 | +0.000 |
| carbonic anhydrase 2 | -0.026 | -0.157 | -0.034 | -0.000 |
| coagulation factor Xa | -0.047 | -0.042 | -0.004 | +0.000 |
| cyclin-dependent kinase 2 | -0.090 | +0.110 | -0.280 | +0.000 |
| cyclooxygenase 2 | -0.093 | -0.009 | -0.066 | +0.000 |
| delta opioid receptor | -0.007 | -0.050 | -0.619 | +0.000 |
| dipeptidyl peptidase 4 | +0.083 | +0.092 | -0.141 | +0.000 |
| dopamine D2 receptor | -0.036 | +0.056 | -0.055 | -0.000 |
| dopamine D3 receptor | +0.011 | +0.139 | -0.019 | +0.000 |
| dopamine D4 receptor | -0.277 | +0.007 | -0.019 | -0.000 |
| dopamine transporter | -0.008 | -0.005 | -0.006 | -0.000 |
| epidermal growth factor receptor | -0.027 | -0.009 | -0.034 | +0.000 |
| fatty acid amide hydrolase | +0.538 | +0.501 | -0.006 | +0.000 |
| glucocorticoid receptor | -0.311 | -0.118 | +0.057 | +0.000 |
| hERG potassium channel | -0.006 | -0.005 | -0.012 | +0.000 |
| histamine H1 receptor | -0.021 | -0.022 | -0.032 | +0.000 |
| histamine H3 receptor | -0.004 | -0.220 | -0.005 | -0.000 |
| kappa opioid receptor | -0.147 | +0.067 | -0.124 | -0.000 |
| monoamine oxidase A | -0.072 | -0.002 | -0.038 | +0.000 |
| monoamine oxidase B | -0.085 | -0.052 | +0.001 | -0.000 |
| mu opioid receptor | -0.107 | -0.093 | -0.064 | +0.000 |
| muscarinic M1 receptor | -0.146 | -0.007 | -0.089 | -0.000 |
| muscarinic M2 receptor | -0.004 | +0.001 | -0.004 | -0.000 |
| muscarinic M3 receptor | -0.026 | +0.022 | -0.183 | +0.000 |
| neurokinin NK1 receptor | -0.003 | +0.075 | -0.005 | -0.000 |
| noradrenaline transporter | -0.003 | +0.049 | -0.006 | +0.000 |
| oestrogen receptor alpha | -0.361 | -0.005 | -0.059 | +0.000 |
| phosphodiesterase 4D | -0.090 | -0.073 | -0.099 | +0.000 |
| phosphodiesterase 5A | -0.030 | -0.025 | +0.085 | +0.000 |
| protein kinase B | +0.010 | +0.071 | -0.000 | -0.000 |
| renin | -0.059 | -0.056 | +0.076 | +0.000 |
| rho-associated kinase 1 | +0.067 | -0.004 | +0.136 | -0.000 |
| serotonin 5-HT1A receptor | -0.014 | -0.053 | -0.032 | -0.000 |
| serotonin 5-HT2A receptor | -0.001 | -0.046 | -0.005 | -0.000 |
| serotonin 5-HT2C receptor | +0.026 | +0.200 | -0.055 | +0.000 |
| serotonin 5-HT6 receptor | -0.048 | +0.179 | -0.012 | +0.000 |
| serotonin 5-HT7 receptor | -0.006 | -0.037 | -0.004 | +0.000 |
| serotonin transporter | +0.007 | +0.007 | +0.004 | -0.000 |
| thrombin | -0.122 | -0.395 | -0.044 | -0.000 |

## The difference inside a pair, with scaffolds held out

Every target is fitted on its own. The last row counts the held-out pairs of all of them together.

| Target | Pairs | Scaffolds | every centre | stereocentres only | every other centre | no handedness channels |
|---|---|---|---|---|---|---|
| 11-beta-hydroxysteroid dehydrogenase 1 | 26 | 19 | 57.7 % (39 to 74) | 50.0 % (32 to 68) | 57.7 % (39 to 74) | 50.0 % (32 to 68) |
| PI3-kinase alpha | 139 | 88 | 59.4 % (51 to 67) | 54.9 % (46 to 63) | 55.6 % (47 to 64) | 50.0 % (42 to 58) |
| PIM1 kinase | 177 | 85 | 52.9 % (45 to 60) | 53.4 % (46 to 61) | 52.9 % (45 to 60) | 50.0 % (43 to 57) |
| PPAR gamma | 101 | 65 | 67.0 % (57 to 75) | 67.5 % (58 to 76) | 57.0 % (47 to 66) | 50.0 % (40 to 60) |
| acetylcholinesterase | 28 | 26 | 53.6 % (36 to 70) | 66.1 % (48 to 81) | 46.4 % (30 to 64) | 50.0 % (33 to 67) |
| alpha-1A adrenergic receptor | 44 | 32 | 72.1 % (57 to 83) | 79.1 % (65 to 89) | 58.1 % (43 to 72) | 50.0 % (36 to 64) |
| alpha-2A adrenergic receptor | 16 | 12 | 25.0 % (10 to 49) | 62.5 % (39 to 82) | 25.0 % (10 to 49) | 50.0 % (28 to 72) |
| androgen receptor | 43 | 29 | 48.8 % (35 to 63) | 37.2 % (24 to 52) | 55.8 % (41 to 70) | 50.0 % (36 to 64) |
| beta-2 adrenergic receptor | 39 | 18 | 89.7 % (76 to 96) | 93.6 % (81 to 98) | 53.8 % (39 to 68) | 50.0 % (35 to 65) |
| beta-secretase 1 | 142 | 77 | 55.3 % (47 to 63) | 64.5 % (56 to 72) | 51.8 % (44 to 60) | 50.0 % (42 to 58) |
| butyrylcholinesterase | 27 | 23 | 51.9 % (34 to 69) | 63.0 % (44 to 78) | 40.7 % (25 to 59) | 50.0 % (32 to 68) |
| cannabinoid CB1 receptor | 55 | 38 | 49.1 % (36 to 62) | 62.7 % (50 to 74) | 32.7 % (22 to 46) | 50.0 % (37 to 63) |
| cannabinoid CB2 receptor | 61 | 44 | 42.6 % (31 to 55) | 50.0 % (38 to 62) | 44.3 % (33 to 57) | 50.0 % (38 to 62) |
| carbonic anhydrase 2 | 58 | 28 | 56.9 % (44 to 69) | 50.9 % (38 to 63) | 56.9 % (44 to 69) | 50.0 % (38 to 62) |
| coagulation factor Xa | 52 | 40 | 51.9 % (39 to 65) | 62.5 % (49 to 74) | 48.1 % (35 to 61) | 50.0 % (37 to 63) |
| cyclin-dependent kinase 2 | 44 | 30 | 47.6 % (33 to 62) | 60.7 % (46 to 74) | 54.8 % (40 to 69) | 50.0 % (36 to 64) |
| cyclooxygenase 2 | 43 | 15 | 53.5 % (39 to 67) | 62.8 % (48 to 76) | 55.8 % (41 to 70) | 50.0 % (36 to 64) |
| delta opioid receptor | 32 | 19 | 48.4 % (32 to 65) | 56.5 % (39 to 72) | 51.6 % (35 to 68) | 50.0 % (33 to 67) |
| dipeptidyl peptidase 4 | 33 | 19 | 56.7 % (39 to 73) | 55.0 % (38 to 71) | 43.3 % (27 to 61) | 50.0 % (33 to 67) |
| dopamine D2 receptor | 234 | 146 | 46.8 % (40 to 53) | 53.2 % (47 to 60) | 43.7 % (37 to 50) | 50.0 % (44 to 56) |
| dopamine D3 receptor | 163 | 87 | 54.6 % (47 to 62) | 55.5 % (48 to 63) | 51.5 % (44 to 59) | 50.0 % (42 to 58) |
| dopamine D4 receptor | 62 | 49 | 56.5 % (44 to 68) | 50.0 % (38 to 62) | 59.7 % (47 to 71) | 50.0 % (38 to 62) |
| dopamine transporter | 147 | 68 | 47.9 % (40 to 56) | 48.6 % (41 to 57) | 46.5 % (39 to 55) | 50.0 % (42 to 58) |
| epidermal growth factor receptor | 111 | 82 | 53.5 % (44 to 63) | 53.0 % (43 to 63) | 50.5 % (41 to 60) | 50.0 % (40 to 60) |
| fatty acid amide hydrolase | 54 | 30 | 88.9 % (78 to 95) | 75.0 % (62 to 85) | 44.4 % (32 to 58) | 50.0 % (37 to 63) |
| glucocorticoid receptor | 20 | 16 | 68.4 % (46 to 85) | 57.9 % (36 to 77) | 52.6 % (32 to 73) | 50.0 % (29 to 71) |
| hERG potassium channel | 143 | 106 | 45.6 % (37 to 54) | 48.5 % (40 to 57) | 48.5 % (40 to 57) | 50.0 % (42 to 58) |
| histamine H1 receptor | 30 | 20 | 53.3 % (36 to 70) | 51.7 % (35 to 68) | 53.3 % (36 to 70) | 50.0 % (33 to 67) |
| histamine H3 receptor | 120 | 67 | 43.6 % (35 to 53) | 48.3 % (39 to 57) | 37.6 % (29 to 47) | 50.0 % (41 to 59) |
| kappa opioid receptor | 100 | 49 | 55.1 % (44 to 66) | 59.0 % (48 to 69) | 51.3 % (40 to 62) | 50.0 % (39 to 61) |
| monoamine oxidase A | 33 | 17 | 45.2 % (29 to 62) | 41.9 % (26 to 59) | 54.8 % (38 to 71) | 50.0 % (33 to 67) |
| monoamine oxidase B | 41 | 20 | 55.3 % (40 to 70) | 55.3 % (40 to 70) | 44.7 % (30 to 60) | 50.0 % (35 to 65) |
| mu opioid receptor | 77 | 47 | 31.8 % (22 to 44) | 43.9 % (33 to 56) | 42.4 % (31 to 54) | 50.0 % (38 to 62) |
| muscarinic M1 receptor | 37 | 24 | 35.3 % (21 to 52) | 51.5 % (35 to 67) | 32.4 % (19 to 49) | 50.0 % (34 to 66) |
| muscarinic M2 receptor | 41 | 26 | 42.1 % (28 to 58) | 43.4 % (29 to 59) | 42.1 % (28 to 58) | 50.0 % (35 to 65) |
| muscarinic M3 receptor | 53 | 19 | 33.3 % (22 to 47) | 41.2 % (29 to 55) | 49.0 % (36 to 62) | 50.0 % (37 to 63) |
| neurokinin NK1 receptor | 58 | 38 | 44.8 % (33 to 58) | 44.8 % (33 to 58) | 43.1 % (31 to 56) | 50.0 % (38 to 62) |
| noradrenaline transporter | 192 | 87 | 51.1 % (44 to 58) | 53.8 % (47 to 61) | 52.2 % (45 to 59) | 50.0 % (43 to 57) |
| oestrogen receptor alpha | 75 | 40 | 53.3 % (42 to 64) | 48.0 % (37 to 59) | 53.3 % (42 to 64) | 50.0 % (39 to 61) |
| phosphodiesterase 4D | 23 | 17 | 60.9 % (41 to 78) | 52.2 % (33 to 71) | 47.8 % (29 to 67) | 50.0 % (31 to 69) |
| phosphodiesterase 5A | 58 | 29 | 62.5 % (49 to 74) | 48.2 % (36 to 61) | 62.5 % (49 to 74) | 50.0 % (37 to 63) |
| protein kinase B | 46 | 28 | 64.4 % (50 to 77) | 52.2 % (38 to 66) | 64.4 % (50 to 77) | 50.0 % (36 to 64) |
| renin | 22 | 18 | 36.4 % (20 to 57) | 25.0 % (12 to 46) | 45.5 % (27 to 65) | 50.0 % (31 to 69) |
| rho-associated kinase 1 | 56 | 38 | 35.8 % (24 to 49) | 42.5 % (30 to 56) | 32.1 % (21 to 45) | 50.0 % (37 to 63) |
| serotonin 5-HT1A receptor | 83 | 53 | 38.3 % (28 to 49) | 44.4 % (34 to 55) | 53.1 % (42 to 64) | 50.0 % (39 to 61) |
| serotonin 5-HT2A receptor | 114 | 69 | 41.3 % (32 to 51) | 43.8 % (35 to 53) | 51.9 % (42 to 61) | 50.0 % (41 to 59) |
| serotonin 5-HT2C receptor | 72 | 44 | 62.9 % (51 to 73) | 67.1 % (56 to 77) | 48.6 % (37 to 60) | 50.0 % (39 to 61) |
| serotonin 5-HT6 receptor | 83 | 50 | 61.7 % (51 to 72) | 66.7 % (56 to 76) | 54.3 % (44 to 65) | 50.0 % (39 to 61) |
| serotonin 5-HT7 receptor | 65 | 44 | 34.9 % (24 to 47) | 37.3 % (26 to 50) | 39.7 % (29 to 52) | 50.0 % (38 to 62) |
| serotonin transporter | 201 | 94 | 45.7 % (39 to 53) | 49.5 % (43 to 56) | 54.3 % (47 to 61) | 50.0 % (43 to 57) |
| thrombin | 23 | 17 | 40.9 % (23 to 61) | 63.6 % (43 to 80) | 50.0 % (31 to 69) | 50.0 % (31 to 69) |
| **every target, each fitted on its own** | 3667 | | **51.3 %** (50 to 53) | **53.8 %** (52 to 55) | **49.7 %** (48 to 51) | **50.0 %** (48 to 52) |

As q2 of the half-difference (at or below zero means the model predicts the difference no better than predicting zero for every pair):

| Target | every centre | stereocentres only | every other centre | no handedness channels |
|---|---|---|---|---|
| 11-beta-hydroxysteroid dehydrogenase 1 | -0.025 | -0.648 | -0.018 | +0.000 |
| PI3-kinase alpha | +0.015 | +0.091 | -0.015 | +0.000 |
| PIM1 kinase | -0.047 | +0.129 | +0.005 | +0.000 |
| PPAR gamma | +0.037 | +0.095 | -0.030 | +0.000 |
| acetylcholinesterase | -0.025 | +0.408 | -0.018 | -0.000 |
| alpha-1A adrenergic receptor | +0.065 | +0.219 | +0.003 | -0.000 |
| alpha-2A adrenergic receptor | -1.245 | +0.013 | -1.222 | +0.000 |
| androgen receptor | -0.249 | -0.064 | -0.317 | +0.000 |
| beta-2 adrenergic receptor | +0.074 | +0.474 | -0.034 | +0.000 |
| beta-secretase 1 | +0.054 | +0.141 | -0.011 | +0.000 |
| butyrylcholinesterase | -0.416 | -0.165 | -0.293 | +0.000 |
| cannabinoid CB1 receptor | -0.009 | +0.136 | -0.011 | -0.000 |
| cannabinoid CB2 receptor | -0.181 | -0.209 | -0.004 | -0.000 |
| carbonic anhydrase 2 | -0.005 | -0.207 | -0.002 | -0.000 |
| coagulation factor Xa | -0.016 | -0.052 | -0.005 | +0.000 |
| cyclin-dependent kinase 2 | -0.051 | +0.050 | -0.035 | +0.000 |
| cyclooxygenase 2 | -0.017 | -0.073 | -0.010 | -0.000 |
| delta opioid receptor | -0.044 | -0.153 | -0.021 | -0.000 |
| dipeptidyl peptidase 4 | -0.059 | +0.091 | -0.002 | +0.000 |
| dopamine D2 receptor | -0.021 | -0.011 | -0.034 | -0.000 |
| dopamine D3 receptor | -0.007 | +0.075 | +0.001 | +0.000 |
| dopamine D4 receptor | -0.284 | -0.030 | -0.026 | -0.000 |
| dopamine transporter | -0.008 | -0.011 | -0.009 | -0.000 |
| epidermal growth factor receptor | -17.145 | -0.014 | -0.005 | +0.000 |
| fatty acid amide hydrolase | +0.485 | +0.448 | -0.065 | -0.000 |
| glucocorticoid receptor | -0.031 | -0.116 | -0.162 | +0.000 |
| hERG potassium channel | -0.003 | -0.007 | -0.000 | -0.000 |
| histamine H1 receptor | -0.036 | -0.024 | -0.038 | -0.000 |
| histamine H3 receptor | -0.008 | -0.026 | -0.006 | -0.000 |
| kappa opioid receptor | -0.100 | +0.032 | -0.162 | +0.000 |
| monoamine oxidase A | -0.038 | -0.009 | -0.136 | +0.000 |
| monoamine oxidase B | -0.088 | -0.968 | -0.008 | +0.000 |
| mu opioid receptor | -0.015 | -0.084 | -0.008 | -0.000 |
| muscarinic M1 receptor | -0.010 | -0.017 | -0.004 | +0.000 |
| muscarinic M2 receptor | -0.003 | -0.036 | -0.002 | +0.000 |
| muscarinic M3 receptor | -0.017 | +0.016 | -0.106 | -0.000 |
| neurokinin NK1 receptor | -0.007 | -0.230 | -0.007 | -0.000 |
| noradrenaline transporter | -0.019 | +0.002 | -0.003 | +0.000 |
| oestrogen receptor alpha | -0.037 | -0.000 | -0.156 | +0.000 |
| phosphodiesterase 4D | -0.055 | -0.008 | -0.051 | -0.000 |
| phosphodiesterase 5A | -0.003 | -0.002 | +0.006 | -0.000 |
| protein kinase B | -0.046 | -0.069 | -0.100 | -0.000 |
| renin | -0.321 | -0.014 | -0.211 | -0.000 |
| rho-associated kinase 1 | -0.008 | -0.008 | -0.008 | -0.000 |
| serotonin 5-HT1A receptor | -0.019 | -0.133 | -0.037 | -0.000 |
| serotonin 5-HT2A receptor | -0.012 | -0.058 | -0.017 | +0.000 |
| serotonin 5-HT2C receptor | +0.179 | +0.241 | -0.053 | -0.000 |
| serotonin 5-HT6 receptor | -0.032 | +0.124 | -0.024 | -0.000 |
| serotonin 5-HT7 receptor | -0.008 | -0.221 | -0.004 | +0.000 |
| serotonin transporter | -0.008 | -0.044 | -0.003 | +0.000 |
| thrombin | -0.118 | -0.474 | -0.041 | +0.000 |

Among the pairs whose two potencies differ by at least one log unit, with every target fitted on its own:

| Fold rule | every centre | stereocentres only | every other centre | no handedness channels |
|---|---|---|---|---|
| pairs held out | 57.4 % of 1144 | 60.5 % of 1144 | 53.2 % of 1144 | 50.0 % of 1144 |
| scaffolds held out | 55.6 % of 1144 | 59.1 % of 1144 | 51.8 % of 1144 | 50.0 % of 1144 |

The spread of the difference between the two potencies of a pair is 1.04 log units.

## The potency itself, one target at a time

The ordinary question of item C7, on sets that do hold enantiomeric pairs: does the handedness block add anything to the two-point kernel? Whole pairs are held out, and every variant chooses its distance grid and its regularisation inside the training part of each fold.

| Target | Pairs | two-point | stereo centres (C7) |
|---|---|---|---|
| 11-beta-hydroxysteroid dehydrogenase 1 | 26 | -0.080 | -0.256 |
| PI3-kinase alpha | 139 | +0.482 | +0.520 |
| PIM1 kinase | 177 | -0.033 | -0.032 |
| PPAR gamma | 101 | +0.208 | +0.206 |
| acetylcholinesterase | 28 | +0.228 | +0.239 |
| alpha-1A adrenergic receptor | 44 | -0.044 | -0.014 |
| alpha-2A adrenergic receptor | 16 | +0.184 | -0.407 |
| androgen receptor | 43 | +0.055 | -0.017 |
| beta-2 adrenergic receptor | 39 | +0.379 | +0.426 |
| beta-secretase 1 | 142 | +0.076 | +0.057 |
| butyrylcholinesterase | 27 | -0.265 | -0.075 |
| cannabinoid CB1 receptor | 55 | -0.093 | -0.054 |
| cannabinoid CB2 receptor | 61 | -0.022 | -0.068 |
| carbonic anhydrase 2 | 58 | -0.058 | -0.059 |
| coagulation factor Xa | 52 | -0.089 | -0.188 |
| cyclin-dependent kinase 2 | 44 | -0.104 | -0.112 |
| cyclooxygenase 2 | 43 | +0.039 | -0.117 |
| delta opioid receptor | 32 | -0.158 | -0.215 |
| dipeptidyl peptidase 4 | 33 | -0.094 | +0.011 |
| dopamine D2 receptor | 234 | +0.163 | +0.128 |
| dopamine D3 receptor | 163 | +0.228 | +0.207 |
| dopamine D4 receptor | 62 | -0.099 | -0.202 |
| dopamine transporter | 147 | +0.014 | +0.109 |
| epidermal growth factor receptor | 111 | +0.143 | +0.142 |
| fatty acid amide hydrolase | 54 | +0.339 | +0.379 |
| glucocorticoid receptor | 20 | -0.121 | +0.073 |
| hERG potassium channel | 143 | +0.190 | +0.156 |
| histamine H1 receptor | 30 | -0.114 | -0.149 |
| histamine H3 receptor | 120 | +0.152 | +0.070 |
| kappa opioid receptor | 100 | +0.094 | +0.195 |
| monoamine oxidase A | 33 | +0.026 | -0.150 |
| monoamine oxidase B | 41 | -0.371 | -0.136 |
| mu opioid receptor | 77 | -0.059 | +0.083 |
| muscarinic M1 receptor | 37 | +0.681 | +0.128 |
| muscarinic M2 receptor | 41 | +0.350 | +0.182 |
| muscarinic M3 receptor | 53 | +0.191 | -0.129 |
| neurokinin NK1 receptor | 58 | +0.060 | +0.053 |
| noradrenaline transporter | 192 | -0.107 | -0.121 |
| oestrogen receptor alpha | 75 | -0.012 | -0.075 |
| phosphodiesterase 4D | 23 | -0.313 | -0.757 |
| phosphodiesterase 5A | 58 | +0.476 | +0.410 |
| protein kinase B | 46 | -0.181 | +0.012 |
| renin | 22 | -0.182 | -0.202 |
| rho-associated kinase 1 | 56 | -0.085 | -0.029 |
| serotonin 5-HT1A receptor | 83 | -0.066 | -0.076 |
| serotonin 5-HT2A receptor | 114 | -0.018 | -0.023 |
| serotonin 5-HT2C receptor | 72 | +0.199 | +0.179 |
| serotonin 5-HT6 receptor | 83 | -0.030 | -0.094 |
| serotonin 5-HT7 receptor | 65 | -0.155 | -0.183 |
| serotonin transporter | 201 | +0.044 | +0.003 |
| thrombin | 23 | -0.093 | -0.267 |

