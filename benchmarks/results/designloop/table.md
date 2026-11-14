# A design loop with more than one objective (plan item MC11)

Lead optimisation has more than one objective and since plan items E1 and MC4 they live on one kernel. This is the loop that reads them together: the Pareto front of the predicted objectives, an acquisition that adds the expected improvement of each one, the batch rule of plan item MC2 so that a plate is not one compound ten times over, and a sentence per suggestion saying what it gains and what it loses.

The test is retrospective and the series makes it possible, because it comes in two parts. The loop is trained on the 78 compounds of part 1, with their histamine H3 affinity and their hERG activity. It then scores 427 candidates -- every other H3 ligand in ChEMBL that fits the same frame -- of which 16 are the compounds the same authors went on to make in part 2. The loop sees no activity of any candidate.

The model reaches q2 +0.276 on the potency (alpha 0.2, ridge 1); the task correlation is -0.431. The objectives are pKi_H3 (higher is better, best so far 10.30) and pIC50_hERG (lower is better, best so far 4.36).

| the list the loop reports | candidates | of them made next | share |
| --- | ---: | ---: | ---: |
| the Pareto front | 5 | 0 | 0.0% |
| the first 10 by the acquisition | 10 | 0 | 0.0% |
| the first 20 by the acquisition | 20 | 0 | 0.0% |
| the first 50 by the acquisition | 50 | 0 | 0.0% |
| the batch of 5 taken from the front | 5 | 0 | 0.0% |

A front of 5 cannot hold 16 compounds, so where they sit is the number that says more. The compounds the authors made are at Pareto layer 22 of 51, against 19 for the rest of the pool, and at rank 344 of 427 by the acquisition against 209. The best of them is on layer 2 and at rank 79.

The base rate is 3.7%: that is the share of the pool the authors made. The Pareto front holds 0.00 times that [0.00, 0.00], resampling the front.

## What the loop would order

The batch is taken from the front by the acquisition, with every objective's posterior conditioned on each pick. A change smaller than the model's own error is reported as no change.

| compound | made next | predicted potency | predicted hERG | what it trades |
| --- | :-: | ---: | ---: | --- |
| CHEMBL4571200 |  | 8.04 | 3.94 | loses pKi_H3 -2.26 |
| CHEMBL193607 |  | 8.39 | 4.14 | loses pKi_H3 -1.91 |
| CHEMBL1085782 |  | 9.37 | 4.22 | loses pKi_H3 -0.93 |
| CHEMBL1917641 |  | 9.87 | 4.41 | nothing worth reporting |
| CHEMBL1917464 |  | 9.81 | 4.30 | nothing worth reporting |

## The verdict

**The front contains none of the compounds the authors made.** The gate is not met.

**Nor does it rank them above the pool.** The compounds they made sit at Pareto layer 22 of 51 against 19 for the rest, and at rank 344 of 427 against 209 by the acquisition. The best of them reaches layer 2 and rank 79, so the loop does not miss them altogether; it puts them in the worse half.

**What the authors made was better than the pool, and the model saw part of it.** Their measured H3 affinity averages 9.15 against 7.97 for the rest of the pool; the model predicted 8.68 against 8.57. The order is right and the size of the gap is not, which is what a q2 of +0.276 on a series this narrow buys.

**What sinks them in the ranking is the other objective.** The model predicts their hERG activity at 6.29 against 5.90 for the rest of the pool, and the acquisition adds an expected improvement over the best already made in each objective, so a compound that is good in one and ordinary in the other scores below one that is extreme in either. No hERG number was ever measured for these compounds, so whether the model is right about them cannot be checked -- which is itself the honest statement about a two-objective loop run on a series where only one objective was followed up.

**What this test can and cannot settle.** It is one pair of papers, and the compounds the authors made next were chosen with knowledge the loop does not have -- synthetic access, a chemical idea, the rest of their programme -- so the compounds they did not make are not known failures. The candidates also carry no measured hERG number, so only one of the two objectives can be checked against what happened. What the item asked for and this delivers is the machinery: a front over several objectives on one kernel, a batch taken from it by the rule of plan item MC2, and for every suggestion the trade it makes written out.

