# A curated alignment against an automatic one

The same nine targets twice: the conformations and superposition distributed with
Sutherland et al. (2004), and the same compounds built from 2D by RDKit ETKDG and
superimposed on a reference molecule by maximum common substructure for the Mittal
et al. (2009) collection. Same fields (electrostatic, steric, hydrophobic, hba, hbd), same alpha
grid, same lambda grid, same 10-fold CV.

| target | n | q2 published | q2 automatic | median fit RMSD | gain from geometry (published) | gain from geometry (automatic) |
|---|---|---|---|---|---|---|
| ACE | 114/114 | +0.651 | +0.470 | 1.27 A | -0.003 | -0.142 |
| AChE | 111/111 | +0.547 | +0.243 | 1.61 A | +0.337 | +0.078 |
| BZR | 147/163 | +0.424 | +0.361 | 1.29 A | +0.162 | +0.039 |
| COX2 | 282/322 | +0.552 | +0.248 | 1.01 A | +0.244 | -0.095 |
| DHFR | 361/397 | +0.651 | +0.490 | 0.50 A | +0.272 | +0.225 |
| GPB | 66/66 | +0.593 | +0.260 | 1.12 A | +0.285 | -0.063 |
| THER | 76/76 | +0.538 | +0.043 | 1.60 A | +0.292 | -0.236 |
| THR | 88/88 | +0.701 | +0.021 | 2.71 A | +0.691 | -0.163 |
| STEROIDS | 31/21 | +0.719 | +0.515 | 1.40 A | +0.219 | +0.065 |

Mean q2: published +0.597, automatic +0.295 (difference +0.303).

The frame against the conformer: the invariant column uses the same coordinates without
any superposition, so where it keeps its q2 and the aligned column does not, the frame is
what went wrong rather than the geometry.

| target | aligned pub. | invariant pub. | counts pub. | aligned aut. | invariant aut. | counts aut. |
|---|---|---|---|---|---|---|
| ACE | +0.651 | +0.677 | +0.654 | +0.470 | +0.536 | +0.611 |
| AChE | +0.547 | +0.549 | +0.210 | +0.243 | +0.124 | +0.165 |
| BZR | +0.424 | +0.475 | +0.262 | +0.361 | +0.389 | +0.322 |
| COX2 | +0.552 | +0.545 | +0.308 | +0.248 | +0.396 | +0.343 |
| DHFR | +0.651 | +0.651 | +0.379 | +0.490 | +0.571 | +0.266 |
| GPB | +0.593 | +0.457 | +0.308 | +0.260 | +0.307 | +0.324 |
| THER | +0.538 | +0.528 | +0.246 | +0.043 | +0.264 | +0.279 |
| THR | +0.701 | +0.442 | +0.010 | +0.021 | +0.452 | +0.184 |
| STEROIDS | +0.719 | +0.694 | +0.500 | +0.515 | +0.302 | +0.451 |
