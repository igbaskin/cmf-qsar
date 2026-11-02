# Inverse design on prototypes (plan item M11)

## How large the design space is

The five-atom cubic ABO3 prototype with the 52 A and 52 B elements that `matbench_perovskites` uses allows 2,704 candidates. The set holds 18,928 cells with 2,661 distinct A/B pairs, so 98.4 % of the space has a computed formation energy and 43 candidates have none.

One more site class and the space is no longer exhausted. The same prototype doubled, with its two B sites free of each other (A2BB'O6), allows 140,608 candidates, and the repository holds a computed formation energy for none of them. With 4 elements per site the duplicate test runs over the whole enumeration: 64 candidates leave 64 distinct structures, the duplicates being the pairs that differ only in which B site holds which element.

Placements of one substituent on the 16 sites of an fcc supercell, reduced by the symmetry of the prototype:

| Substituted sites | Placements | Distinct arrangements | Duplicates |
|---:|---:|---:|---:|
| 1 | 16 | 1 | 15 |
| 2 | 120 | 5 | 115 |
| 3 | 560 | 10 | 550 |

Of those 10 arrangements with three substituted sites, the descriptors separate fewer:

| Descriptor | Channel sets | Arrangements |
|---|---:|---:|
| three-body 5 A | 8 | 10 |
| three-body 9 A | 8 | 10 |
| two-point 14 A | 8 | 10 |
| two-point 6 A | 8 | 10 |

`Cu:Au@0,1,8` and `Cu:Au@0,3,4` share their channels at every reach above, and they are not the same structure: ASE's symmetry comparator separates them, and they have the same space group, 25, with different Wyckoff positions. An EMT calculation on the unrelaxed cells puts their energies 0.6 meV per atom apart, which is an error no model on these channels can avoid.
`Cu:Au@0,1,12` and `Cu:Au@0,3,12` share their channels at every reach above, and they are not the same structure: ASE's symmetry comparator separates them, and they have space groups 25 and 65. An EMT calculation on the unrelaxed cells puts their energies 0.6 meV per atom apart, which is an error no model on these channels can avoid.

## The gate: a published pool of perovskites

A budget of 100 calculations after the start set, 10 start sets, and the same start sets for every rule and every descriptor. The target is minimised. **Start** is how many cells were computed before the loop began; **Rank** is where the best cell the loop found sits in the whole pool, 1 being the best cell there is.

| The set | Descriptor | Start | Rule | best found | Rank | beat its start set | reached the best 1 % | calls to reach it |
|---|---|---:|---|---:|---:|---:|---:|---:|
| perovskites | three-body | 100 | random | +0.0060 | 14 of 4000 | 5/10 | 6/10 | 37 |
| perovskites | three-body | 100 | greedy | -0.1780 | 1 of 4000 | 7/10 | 8/10 | 8 |
| perovskites | three-body | 100 | ucb | -0.0600 | 9 of 4000 | 7/10 | 8/10 | 36 |
| perovskites | three-body | 100 | ei | +0.0580 | 28 of 4000 | 4/10 | 6/10 | 67 |
| perovskites | three-body | 100 | variance | +0.0720 | 28 of 4000 | 2/10 | 3/10 | 67 |
| perovskites | three-body | 100 | maxmin | +0.0820 | 40 of 4000 | 1/10 | 0/10 | -- |
| perovskites | three-body | 25 | random | +0.0880 | 27 of 4000 | 6/10 | 3/10 | 50 |
| perovskites | three-body | 25 | greedy | -0.1340 | 1 of 4000 | 7/10 | 8/10 | 16 |
| perovskites | three-body | 25 | ucb | -0.0620 | 6 of 4000 | 7/10 | 7/10 | 31 |
| perovskites | three-body | 25 | ei | +0.1360 | 34 of 4000 | 4/10 | 1/10 | 38 |
| perovskites | three-body | 25 | variance | +0.1900 | 34 of 4000 | 2/10 | 3/10 | 93 |
| perovskites | three-body | 25 | maxmin | +0.1860 | 73 of 4000 | 3/10 | 1/10 | 58 |
| perovskites | two-point | 100 | random | +0.0060 | 14 of 4000 | 5/10 | 6/10 | 37 |
| perovskites | two-point | 100 | greedy | -0.1260 | 1 of 4000 | 7/10 | 7/10 | 12 |
| perovskites | two-point | 100 | ucb | -0.0700 | 6 of 4000 | 8/10 | 8/10 | 41 |
| perovskites | two-point | 100 | ei | +0.0800 | 40 of 4000 | 1/10 | 0/10 | -- |
| perovskites | two-point | 100 | variance | +0.0900 | 40 of 4000 | 0/10 | 0/10 | -- |
| perovskites | two-point | 100 | maxmin | +0.0620 | 28 of 4000 | 1/10 | 1/10 | 92 |
| perovskites | two-point | 25 | random | +0.0880 | 27 of 4000 | 6/10 | 3/10 | 50 |
| perovskites | two-point | 25 | greedy | -0.0260 | 15 of 4000 | 7/10 | 6/10 | 32 |
| perovskites | two-point | 25 | ucb | -0.0020 | 17 of 4000 | 7/10 | 6/10 | 44 |
| perovskites | two-point | 25 | ei | +0.1020 | 31 of 4000 | 4/10 | 2/10 | 50 |
| perovskites | two-point | 25 | variance | +0.2180 | 82 of 4000 | 1/10 | 0/10 | -- |
| perovskites | two-point | 25 | maxmin | +0.0640 | 11 of 4000 | 5/10 | 3/10 | 56 |

The best cell of each pool, which no rule is told: perovskites three-body -0.2600 eV per atom.

## The defect series: which arrangement is the most stable

A budget of 100 calculations after the start set, 10 start sets, and the same start sets for every rule and every descriptor. The target is minimised. **Start** is how many cells were computed before the loop began; **Rank** is where the best cell the loop found sits in the whole pool, 1 being the best cell there is.

| The set | Descriptor | Start | Rule | best found | Rank | beat its start set | reached the best 1 % | calls to reach it |
|---|---|---:|---|---:|---:|---:|---:|---:|
| MoS2 | three-body | 100 | random | +0.4719 | 5 of 2000 | 4/10 | 6/10 | 38 |
| MoS2 | three-body | 100 | greedy | +0.5131 | 10 of 2000 | 1/10 | 0/10 | -- |
| MoS2 | three-body | 100 | ucb | +0.5131 | 10 of 2000 | 1/10 | 0/10 | -- |
| MoS2 | three-body | 100 | ei | +0.2715 | 1 of 2000 | 7/10 | 7/10 | 48 |
| MoS2 | three-body | 100 | variance | +0.4938 | 10 of 2000 | 2/10 | 5/10 | 86 |
| MoS2 | three-body | 100 | maxmin | +0.4030 | 8 of 2000 | 5/10 | 10/10 | 68 |
| MoS2 | three-body | 25 | random | +0.4828 | 10 of 2000 | 7/10 | 7/10 | 46 |
| MoS2 | three-body | 25 | greedy | +0.6293 | 37 of 2000 | 7/10 | 0/10 | -- |
| MoS2 | three-body | 25 | ucb | +0.6294 | 37 of 2000 | 7/10 | 0/10 | -- |
| MoS2 | three-body | 25 | ei | +0.5016 | 6 of 2000 | 7/10 | 3/10 | 81 |
| MoS2 | three-body | 25 | variance | +0.5221 | 35 of 2000 | 6/10 | 0/10 | -- |
| MoS2 | three-body | 25 | maxmin | +0.4318 | 16 of 2000 | 7/10 | 9/10 | 5 |
| MoS2 | two-point | 100 | random | +0.4719 | 5 of 2000 | 4/10 | 6/10 | 38 |
| MoS2 | two-point | 100 | greedy | +0.5132 | 10 of 2000 | 1/10 | 0/10 | -- |
| MoS2 | two-point | 100 | ucb | +0.5132 | 10 of 2000 | 1/10 | 0/10 | -- |
| MoS2 | two-point | 100 | ei | +0.4450 | 8 of 2000 | 4/10 | 3/10 | 78 |
| MoS2 | two-point | 100 | variance | +0.4943 | 10 of 2000 | 2/10 | 0/10 | -- |
| MoS2 | two-point | 100 | maxmin | +0.5132 | 10 of 2000 | 1/10 | 0/10 | -- |
| MoS2 | two-point | 25 | random | +0.4828 | 10 of 2000 | 7/10 | 7/10 | 46 |
| MoS2 | two-point | 25 | greedy | +0.6298 | 37 of 2000 | 6/10 | 0/10 | -- |
| MoS2 | two-point | 25 | ucb | +0.6296 | 38 of 2000 | 7/10 | 1/10 | 38 |
| MoS2 | two-point | 25 | ei | +0.5995 | 37 of 2000 | 7/10 | 1/10 | 96 |
| MoS2 | two-point | 25 | variance | +0.6298 | 39 of 2000 | 6/10 | 0/10 | -- |
| MoS2 | two-point | 25 | maxmin | +0.6298 | 40 of 2000 | 6/10 | 0/10 | -- |
| WSe2 | three-body | 100 | random | -1.1763 | 12 of 2000 | 6/10 | 6/10 | 38 |
| WSe2 | three-body | 100 | greedy | -1.1777 | 1 of 2000 | 9/10 | 10/10 | 38 |
| WSe2 | three-body | 100 | ucb | -1.1777 | 1 of 2000 | 9/10 | 10/10 | 38 |
| WSe2 | three-body | 100 | ei | -1.1772 | 4 of 2000 | 9/10 | 10/10 | 45 |
| WSe2 | three-body | 100 | variance | -1.1759 | 21 of 2000 | 5/10 | 1/10 | 67 |
| WSe2 | three-body | 100 | maxmin | -1.1760 | 20 of 2000 | 6/10 | 3/10 | 42 |
| WSe2 | three-body | 25 | random | -1.1760 | 14 of 2000 | 8/10 | 6/10 | 31 |
| WSe2 | three-body | 25 | greedy | -1.1774 | 1 of 2000 | 10/10 | 10/10 | 28 |
| WSe2 | three-body | 25 | ucb | -1.1774 | 1 of 2000 | 10/10 | 10/10 | 20 |
| WSe2 | three-body | 25 | ei | -1.1777 | 1 of 2000 | 10/10 | 10/10 | 60 |
| WSe2 | three-body | 25 | variance | -1.1758 | 23 of 2000 | 7/10 | 0/10 | -- |
| WSe2 | three-body | 25 | maxmin | -1.1759 | 20 of 2000 | 7/10 | 7/10 | 60 |
| WSe2 | two-point | 100 | random | -1.1763 | 12 of 2000 | 6/10 | 6/10 | 38 |
| WSe2 | two-point | 100 | greedy | -1.1775 | 2 of 2000 | 9/10 | 10/10 | 24 |
| WSe2 | two-point | 100 | ucb | -1.1776 | 1 of 2000 | 9/10 | 10/10 | 22 |
| WSe2 | two-point | 100 | ei | -1.1776 | 1 of 2000 | 10/10 | 10/10 | 16 |
| WSe2 | two-point | 100 | variance | -1.1772 | 4 of 2000 | 8/10 | 8/10 | 26 |
| WSe2 | two-point | 100 | maxmin | -1.1779 | 1 of 2000 | 10/10 | 10/10 | 9 |
| WSe2 | two-point | 25 | random | -1.1760 | 14 of 2000 | 8/10 | 6/10 | 31 |
| WSe2 | two-point | 25 | greedy | -1.1770 | 6 of 2000 | 9/10 | 10/10 | 28 |
| WSe2 | two-point | 25 | ucb | -1.1773 | 2 of 2000 | 9/10 | 10/10 | 27 |
| WSe2 | two-point | 25 | ei | -1.1776 | 2 of 2000 | 10/10 | 10/10 | 17 |
| WSe2 | two-point | 25 | variance | -1.1777 | 1 of 2000 | 10/10 | 10/10 | 26 |
| WSe2 | two-point | 25 | maxmin | -1.1779 | 1 of 2000 | 10/10 | 10/10 | 12 |

The best cell of each pool, which no rule is told: MoS2 three-body +0.1667 eV.

## How well the model orders what it has not computed

The same start sets as the loop, the same kernels, and the ridge chosen the same way. **Rank correlation** and **R squared** are measured on the candidates the start set leaves out. **Place of the best candidate** is where the model puts the best candidate of the pool in its own order, 1 meaning first: it is what a rule that exploits acts on, and a budget of 100 calls cannot reach a candidate the model puts much further back.

| The set | Descriptor | Start | rank correlation | R squared | place of the best candidate |
|---|---|---:|---:|---:|---:|
| MoS2 | three-body | 100 | +0.871 | +0.844 | 894 of 1900 |
| MoS2 | three-body | 25 | +0.865 | +0.814 | 929 of 1975 |
| MoS2 | two-point | 100 | +0.876 | +0.842 | 882 of 1900 |
| MoS2 | two-point | 25 | +0.864 | +0.787 | 902 of 1975 |
| WSe2 | three-body | 100 | +0.933 | +0.878 | 59 of 1900 |
| WSe2 | three-body | 25 | +0.927 | +0.858 | 138 of 1975 |
| WSe2 | two-point | 100 | +0.930 | +0.874 | 98 of 1900 |
| WSe2 | two-point | 25 | +0.923 | +0.851 | 212 of 1975 |
| perovskites | three-body | 100 | +0.530 | +0.313 | 227 of 3900 |
| perovskites | three-body | 25 | +0.273 | +0.060 | 846 of 3975 |
| perovskites | two-point | 100 | +0.387 | +0.166 | 185 of 3900 |
| perovskites | two-point | 25 | +0.270 | +0.037 | 802 of 3975 |

## A closed loop, with a calculator and not a table

8 of the 32 sites of an fcc copper cell take Au: 10,518,300 placements, 400 after the symmetry reduction and the sample. Every candidate is relaxed by an ASE EMT calculator inside the loop, at 0.220 s per call, and the whole pool is also computed exhaustively (88 s) so that the rank is exact. The energies span 0.0325 eV per atom.

A start set of 10 candidates, a budget of 100 calls, 5 start sets.

| Rule | best found (eV per atom) | Rank | found the best of the pool |
|---|---:|---:|---:|
| random | +0.0497 | 3 of 400 | 1/5 |
| greedy | +0.0480 | 1 of 400 | 5/5 |
| ucb | +0.0480 | 1 of 400 | 5/5 |
| ei | +0.0480 | 1 of 400 | 5/5 |
| variance | +0.0480 | 1 of 400 | 5/5 |
| maxmin | +0.0480 | 1 of 400 | 5/5 |

The best candidate of the pool is +0.0480 eV per atom.

