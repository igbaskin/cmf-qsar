# Can a field score say what to grow a fragment into? (plan item MC7)

A bound fragment leaves a gap: what the pocket would still reward, after what the fragment already places is taken off. A candidate group is built from its SMILES, placed on the growth vector by rule with the turn about the new bond scanned, and scored by how much of the gap its own field fills where the pocket leaves room. The gate of the item is whether the group that was actually added comes out among the first ten.

25 growing cases from the fragment sets of the Schrodinger free-energy collection, a library of 201 groups, 12 turns of the new bond tried for each. Nothing about the answer enters the geometry: the growth vector is the direction the fragment's own bonds leave empty, and every candidate, the true one included, is built from its SMILES and placed the same way.

| ranking | median rank of the true group | among the first ten | first |
| --- | ---: | ---: | ---: |
| at random | 82 | 0% | 0% |
| by whether it fits, at random inside that | 15 | 40% | 0% |
| by whether it fits, then by the size of the group | 28 | 12% | 0% |
| by whether it fits, then by the field score | 25 | 28% | 0% |

| set | fragment | grown | added | rank by field | by size | candidates | without a clash |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| frag_liga_auto | 5 | 9 | `CC(F)(F)[*:1]` | 18 | 35 | 201 | 57 |
| frag_liga_auto | 5 | 6 | `CO[*:1]` | 44 | 46 | 201 | 57 |
| frag_liga_auto | 5 | 10 | `FC(F)(F)[*:1]` | 21 | 25 | 201 | 57 |
| frag_liga_auto | 5 | 8 | `C1CC([*:1])C1` | 15 | 28 | 201 | 57 |
| frag_liga_auto | 5 | 7 | `C[*:1]` | 38 | 51 | 201 | 57 |
| frag_liga_auto | 5 | 4 | `Cl[*:1]` | 36 | 52 | 201 | 57 |
| frag_liga_auto | 11 | 12 | `OCCN[*:1]` | 74 | 109 | 201 | 131 |
| frag_mup1 | PT | SBT_R | `C[*:1]` | 26 | 44 | 201 | 50 |
| frag_mup1 | PT | SBT_S | `C[*:1]` | 26 | 44 | 201 | 50 |
| frag_p38 | 3 | 1_1W7H | `N[*:1]` | 61 | 112 | 201 | 115 |
| hsp90_frag_2rings | 17 | 19 | `F[*:1]` | 7 | 10 | 201 | 14 |
| hsp90_frag_2rings | 15 | 18 | `F[*:1]` | 7 | 11 | 201 | 15 |
| hsp90_frag_2rings | 14 | 15 | `C[*:1]` | 41 | 49 | 201 | 55 |
| hsp90_frag_single_ring | 3 | 10 | `F[*:1]` | 8 | 13 | 201 | 17 |
| hsp90_frag_single_ring | 9 | 10 | `C[*:1]` | 41 | 50 | 201 | 56 |
| hsp90_frag_single_ring | 8 | 11 | `F[*:1]` | 7 | 11 | 201 | 15 |
| hsp90_frag_single_ring | 11 | 12 | `C[*:1]` | 41 | 49 | 201 | 55 |
| hsp90_frag_single_ring | 2 | 3 | `C[*:1]` | 43 | 52 | 201 | 58 |
| hsp90_frag_single_ring | 2 | 9 | `F[*:1]` | 8 | 13 | 201 | 17 |
| jak2_set1 | 2_3E63 | 9 | `Cl[*:1]` | 18 | 18 | 201 | 1 |
| jak2_set1 | 2_3E63 | 8 | `Cl[*:1]` | 26 | 27 | 201 | 31 |
| jak2_set2_extra | 13_3E64 | 22 | `Cl[*:1]` | 6 | 6 | 201 | 1 |
| jak2_set2_extra | 13_3E64 | 21 | `C[*:1]` | 3 | 7 | 201 | 11 |
| jak2_set2_extra | 13_3E64 | 20 | `Cl[*:1]` | 20 | 22 | 201 | 11 |
| jak2_set2_extra | 13_3E64 | 22 flip | `Cl[*:1]` | 25 | 28 | 201 | 32 |

## The verdict

**The gate is not met.** The group that was really added comes out among the first ten in 28% of the cases, at a median rank of 25 in a library of 201; ranking the same candidates at random puts it 82, and refusing the placements that clash while shuffling the rest puts it 15. What the plan expected of this item is what happened, and rule 5 applies: the gap field stays a picture of where there is room and what the pocket would reward there, which is worth drawing, and not a way of choosing a group.

**All of what the ranking knows is whether the group fits.** Refusing the placements that clash and shuffling the rest puts the true group at a median rank of 15, against 82 with no method at all. Sorting those candidates by the field score moves it to 25, which is **worse**, and the share among the first ten falls from 40% to 28%. The score is not neutral: it prefers the largest group that still fits, and the additions in these cases are small, so ordering by it pushes the right answer down.

**Against the other reference**, the size of the group, the field score is ahead: a median rank of 25 against 28. A bigger group fills more of any gap, so this comparison says whether the fields added anything to counting atoms. The additions in these cases are small -- a fluorine, a methyl, a chlorine -- so a ranking that prefers large groups is working against itself here.

**What the measurement cannot settle.** The library holds 201 groups and the true one is always in it, which is the easy part of a real enumeration; the cases are few, because a public set with a pocket, a fragment and the compound it grew into is rare; and the placement rule gives every candidate one embedded conformation and a turn about the new bond, where a real growing tool would minimise the group in the site. A negative result here is evidence against the one-body field score, not against fragment growing.

