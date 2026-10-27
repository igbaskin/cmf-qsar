# A potency and an anti-target on the same compounds (plan item MC4)

Where may field be added for the target that is not also added for the off-target? The two-task model answers it as a difference of co-fields. What this item adds is a series in the program's format with both measurements on the same compounds, from one paper and one laboratory: histamine H3 ligands with their hERG activity, from Lazewska and co-workers (Bioorg. Med. Chem. Lett. 21 (2011) 5378), whose own subject is taking the hERG activity out of the series.

78 molecules, pKi_H3 and pIC50_hERG on every one of them. Their spreads are 0.68 and 0.68 log units and their correlation is -0.222, so the two endpoints are nearly independent and a selectivity model has something to find. The fields are the five physico-chemical ones at an attenuation of 0.2 with a ridge of 1, chosen on the primary endpoint alone; the task correlation -0.431 is chosen by the evidence.

## The first part of the gate: the potency must not suffer

| endpoint | molecules | q2 with the other endpoint | q2 on its own |
| --- | ---: | ---: | ---: |
| pKi_H3 | 78 | +0.341 | +0.319 |
| pIC50_hERG | 78 | +0.357 | +0.338 |

On the potency the two-task fit is +0.022 [-0.005, +0.048] against the single fit, paired over the molecules. The selectivity itself is predicted at q2 +0.471 on the 78 molecules that carry both.

## The second part: where the difference co-field puts its weight

What one more atom of each field type, at each kind of place, would do to the predicted selectivity, in log units, with the error the activities allow (a noise level of 0.577 log units, RMSEcv, the out-of-fold error). A negative number is a place where more field moves the compound towards hERG and away from selectivity.

| place | field | effect on the selectivity | atoms |
| --- | --- | ---: | ---: |
| the basic nitrogen | electrostatic | -0.288 [-2.322, +1.746] | 82 |
| the basic nitrogen | steric | +12.839 [+9.684, +15.994] | 82 |
| the basic nitrogen | hydrophobic | +6.961 [+5.457, +8.466] | 82 |
| the basic nitrogen | hba | +0.622 [-1.923, +3.168] | 82 |
| the basic nitrogen | hbd | -0.146 [-0.401, +0.109] | 82 |
| the carbons beside it | electrostatic | +0.383 [-1.205, +1.970] | 825 |
| the carbons beside it | steric | +11.558 [+8.857, +14.259] | 825 |
| the carbons beside it | hydrophobic | +7.350 [+5.919, +8.780] | 825 |
| the carbons beside it | hba | +0.716 [-1.197, +2.630] | 825 |
| the carbons beside it | hbd | -0.444 [-1.315, +0.427] | 825 |
| every other heavy atom | electrostatic | -1.755 [-3.825, +0.315] | 1202 |
| every other heavy atom | steric | +0.031 [-2.391, +2.454] | 1202 |
| every other heavy atom | hydrophobic | +0.702 [-0.768, +2.173] | 1202 |
| every other heavy atom | hba | +4.435 [+2.627, +6.242] | 1202 |
| every other heavy atom | hbd | -2.200 [-5.835, +1.436] | 1202 |

## What a whole-molecule number says, for comparison

| descriptor | against the potency | against the anti-target | against the selectivity |
| --- | ---: | ---: | ---: |
| cLogP | -0.100 | +0.380 | -0.308 |
| basic nitrogens | +0.220 | -0.132 | +0.226 |
| heavy atoms | +0.077 | +0.095 | -0.012 |


## The verdict

**The first part of the gate is met.** Tying the hERG endpoint to the potency changes the potency by +0.022 [-0.005, +0.048], so the selectivity co-field is not bought with a worse model of the thing that matters.

**At the basic nitrogen the difference co-field is of one sign** in 2 of the 5 fields: steric +12.839 [+9.684, +15.994], hydrophobic +6.961 [+5.457, +8.466].

Beside it, among the carbons within five angstroms of that nitrogen, the fields that stand clear are steric +11.558, hydrophobic +7.350.

The reference for both rows is the last block of the table, every other heavy atom: hba +4.435. A co-field that said the same thing everywhere would be saying nothing about the basic nitrogen in particular.

**The sign is not the one the item expected, and the reason is worth stating.** Lipophilicity does drive hERG in this series as the textbook says: cLogP correlates +0.380 with the hERG activity and -0.308 with the selectivity, over a range of 3.0 to 7.0. The co-field nevertheless puts a **positive** weight at the amine region: more field there raises the selectivity. The two are not in conflict, because a co-field is about a place and cLogP is about a molecule. What this series says is that the lipophilicity that costs selectivity is the lipophilicity spread over the rest of the molecule, while the region around the basic nitrogen is where the H3 affinity is made, and the paper's own work was to keep that region and change what is far from it.

**What this measurement is and is not.** It is one series from one paper, 78 compounds whose hERG numbers come from a displacement assay rather than from patch-clamp, and the alignment is by the common substructure rather than by a crystal structure. What it does establish is that the format and the machinery are there: a second endpoint travels with the file, its assay conditions travel as columns, and the difference co-field can be read at the places a chemist asks about. A second series, measured another way, is what would turn the reading into a rule.

