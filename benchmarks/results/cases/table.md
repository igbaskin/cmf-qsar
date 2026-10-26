# Three series, worked through

Free-energy benchmark sets that ship with the crystal structure their ligands were posed in, chosen by measuring every set in `examples/openff` and `examples/schindler` that has a pocket and taking the best models among different target families. The model whose co-fields are read is fitted on the whole series; the attenuation and the ridge were chosen on two thirds of it and the other third -- every third molecule by activity rank -- held out.

| case | target | n | activity range | α | q²cv | R²pred (held out) | y-randomised q², worst of 30 | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ptp1b | protein tyrosine phosphatase 1B | 22 | 3.77 | 3 | +0.499 | +0.147 | +0.539 | 0.065 |
| mcl1 | Mcl-1, a Bcl-2 family protein | 25 | 3.05 | 0.2 | +0.548 | +0.581 | +0.098 | 0.032 |
| cdk8 | cyclin-dependent kinase 8 | 33 | 4.18 | 3 | +0.496 | +0.180 | +0.026 | 0.032 |

**ptp1b: a shuffle of the activities did as well as the activities.** On that set the interpretation below is of noise, and the case is kept for that reason: it is what the check is for.

## Which fields the series needs

Each field removed in turn and the model refitted with the same attenuation and ridge: the number is how much cross-validated q² the removal costs, so a large positive value means the series needs that field and a negative one means the model is better without it. A kernel norm cannot answer this, because the field kernels are scaled to unit mean diagonal before they are mixed and then all five look equally large.

| case | electrostatic | steric | hydrophobic | hba | hbd | needs most |
|---|---:|---:|---:|---:|---:|---|
| ptp1b | -0.009 | +0.113 | -0.016 | +0.043 | +0.025 | steric |
| mcl1 | +0.032 | +0.062 | +0.071 | +0.037 | -0.077 | hydrophobic |
| cdk8 | +0.043 | -0.015 | -0.024 | +0.099 | -0.061 | hba |

Every one of the three can lose at least one field without losing accuracy. **ptp1b** needs steric most, and electrostatic (-0.009), hydrophobic (-0.016) can be removed with no loss; **mcl1** needs hydrophobic most, and hbd (-0.077) can be removed with no loss; **cdk8** needs hba most, and steric (-0.015), hydrophobic (-0.024), hbd (-0.061) can be removed with no loss.

The first of those is worth stopping on: a phosphatase site is charged, and the electrostatic field is the one field this series does not need.

## What the co-fields are made of

The share of the grid points carrying any field at all where the co-field is more than two standard deviations from zero -- that is, the part of the map that the activities' own noise could not have produced (`cofield_with_error`).

| case | electrostatic | steric | hydrophobic | hba | hbd | noise level |
|---|---:|---:|---:|---:|---:|---:|
| ptp1b | 17% | 27% | 17% | 8% | 0% | 0.673 |
| mcl1 | 2% | 23% | 4% | 0% | 0% | 0.559 |
| cdk8 | 4% | 6% | 4% | 7% | 0% | 0.697 |

## What the pocket knows

Field complementarity with the crystal pocket, and the Spearman correlation of that score with the measured activity. `burial` is the steric overlap alone: how deep in the site the ligand sits.

| case | pocket atoms | complementarity | burial | clashes | ρ(score, activity) | ρ(burial, activity) |
|---|---:|---:|---:|---:|---:|---:|
| ptp1b | 943 | +0.003 | 0.015 | 0.0 | +0.046 | -0.368 |
| mcl1 | 1007 | +0.201 | 0.588 | 0.7 | +0.254 | +0.698 |
| cdk8 | 1033 | +0.002 | 0.014 | 0.3 | +0.326 | -0.075 |

The strongest of those is **mcl1**, where how deeply a ligand sits in the site correlates with its activity at +0.698 while the full complementarity score reaches only +0.254. A score that mixes five terms can be worse than one of them.

## Where a prediction comes from

The Shapley decomposition of the most active molecule of each series, by rings and the connected pieces between them. The values add up to the prediction minus the prediction of the bare scaffold, so they are in log units of the activity.

| case | molecule | measured | predicted | the three largest contributions |
|---|---|---:|---:|---|
| ptp1b | lig_23486 | +9.17 | +8.50 | CHOS6 at 10 +2.01, ring 1 +1.22, HN2 at 27 +0.85 |
| mcl1 | lig_53 | +7.26 | +7.15 | CH4 at 37 +2.01, Cl1 at 36 +1.63, ring 2 -1.56 |
| cdk8 | cdk8_43 | +8.70 | +8.29 | ring 4 +1.54, ring 2 -0.75, ring 5 +0.72 |

## What the weakest molecule is missing

The field difference between the least and the most active molecule of each series, as typed lobes. The weights are integrals over the lobe and are not comparable between cases, only within one.

| case | weak | strong | gap | the largest lobes |
|---|---|---|---:|---|
| ptp1b | lig_23467 | lig_23486 | +3.77 | missing steric 8.5, missing hydrophobic 2.1, extra steric 0.9, extra hydrophobic 0.5 |
| mcl1 | lig_47 | lig_53 | +3.05 | missing steric 164.2, missing hydrophobic 42.8, missing hydrophobic 24.8, extra h-bond acceptor 22.7 |
| cdk8 | cdk8_28 | cdk8_43 | +4.18 | missing steric 2.5, extra steric 1.8, extra hydrophobic 1.4, extra steric 1.3 |

What is **not** here is a chemist's verdict on any of it. The plan item asks for these cases to be judged by a medicinal chemist, and that has not been done: what this file holds is every number such a judgement would need.

