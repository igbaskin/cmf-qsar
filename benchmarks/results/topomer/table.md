# The alignment against the topomer rule, on the Sutherland sets

Aligned CMF (the frame the file comes with), topomer CMF (substituents posed by rule,
no superposition) and the alignment-free count baseline, all on the same molecules:
the congeneric series each set contains, found as the Bemis-Murcko framework of its
largest chemotype plus every molecule that framework cuts cleanly off. Same fields (electrostatic, steric, hydrophobic, hba, hbd),
same alpha grid (0.1, 0.3, 0.5, 1, 2, 5), same lambda grid, same 10-fold CV, and both field
models choose (alpha, lambda) by the CV they are scored on, so both are optimistic to
the same degree.

| set | n | series | core | sites | q2 aligned | q2 topomer | q2 counts | loss |
|---|---|---|---|---|---|---|---|---|
| ACE | 50/114 | 44% | `C1CCNC1` | 5 | +0.645 | +0.530 | +0.722 | +0.115 |
| AChE | 29/111 | 26% | `O=C(NCCC1CC[NH+](Cc2ccccc2)CC1)c1ccccc1` | 20 | +0.499 | +0.514 | +0.376 | -0.014 |
| BZR | 62/147 | 42% | `O=C1CN=C(c2ccccc2)c2ccccc2N1` | 12 | +0.612 | +0.681 | +0.293 | -0.069 |
| COX2 | 62/282 | 22% | `c1ccc(-c2ccnn2-c2ccccc2)cc1` | 12 | +0.416 | +0.157 | +0.108 | +0.259 |
| DHFR | 42/361 | 12% | `c1ccc(NCc2cnc3ncncc3c2)cc1` | 11 | +0.260 | +0.277 | -0.029 | -0.017 |
| GPB | 60/66 | 91% | `C1CCOCC1` | 5 | +0.590 | +0.553 | +0.258 | +0.037 |
| THER | 63/76 | 83% | `c1ccccc1` | 6 | +0.458 | +0.179 | +0.144 | +0.279 |
| THR | 22/88 | 25% | `O=C(C(Cc1ccccc1)NS(=O)(=O)c1ccc2ccccc2c1)N1CCCCC1` | 20 | +0.076 | +0.080 | +0.105 | -0.004 |

Mean loss from giving up the alignment: +0.073 q2 over 8 sets.

Splitting that loss in two. The middle column keeps the same cut substituents at the
same separated sites but leaves each one exactly where the file's own alignment
put it, so the step to its left is the price of the cut itself -- the scaffold's own
field and the geometry between one substituent and another, both of which separated
sites destroy -- and the step to its right is the price of the pose rule alone, with
the same atoms and the same weights in both.

| set | q2 whole molecules | q2 substituents, as aligned | q2 substituents, posed by rule | cut costs | pose costs |
|---|---|---|---|---|---|
| ACE | +0.645 | +0.597 | +0.530 | +0.048 | +0.067 |
| AChE | +0.499 | +0.522 | +0.514 | -0.023 | +0.009 |
| BZR | +0.612 | +0.735 | +0.681 | -0.123 | +0.055 |
| COX2 | +0.416 | +0.238 | +0.157 | +0.177 | +0.082 |
| DHFR | +0.260 | +0.332 | +0.277 | -0.071 | +0.055 |
| GPB | +0.590 | +0.593 | +0.553 | -0.003 | +0.040 |
| THER | +0.458 | +0.199 | +0.179 | +0.259 | +0.020 |
| THR | +0.076 | -0.125 | +0.080 | +0.201 | -0.205 |

Mean price of the cut +0.058, mean price of the pose +0.015.

| set | RMSEcv aligned | RMSEcv topomer | aligned on all n |
|---|---|---|---|
| ACE | 1.244 | 1.432 | +0.651 (114) |
| AChE | 0.813 | 0.801 | +0.547 (111) |
| BZR | 0.465 | 0.422 | +0.424 (147) |
| COX2 | 0.977 | 1.173 | +0.552 (282) |
| DHFR | 0.612 | 0.605 | +0.651 (361) |
| GPB | 0.594 | 0.620 | +0.593 (66) |
| THER | 1.428 | 1.757 | +0.538 (76) |
| THR | 0.703 | 0.701 | +0.701 (88) |
