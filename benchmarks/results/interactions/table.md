# Is the pair-interaction field where the pair is

`benchmarks/run_interactions.py`: 105 para-disubstituted benzenes X-C6H4-Y from 14 substituents, with an additive activity σ(X) + σ(Y) and an interacting one that adds 1.5 σ(X) σ(Y); five fields, unnormalised kernels, α, λ and γ by 10-fold cross-validation. Stability: mean correlation of the OCH3/NO2 pair field over five refits on four-fifths of the set.

| activity | n | q² linear | q² gauss | γ | index joint | stability |
|---|---|---|---|---|---|---|
| additive | 105 | 0.99 | 0.99 | 0.25 | 0.026 | 0.98 |
| interacting | 105 | 0.82 | 0.95 | 0.5 | 0.236 | 0.97 |

The measures below are taken on the 78 molecules with X and Y both other than H and different. Cross-site share: the part of the atom interaction matrix (electrostatic field, off-diagonal, absolute) between the X and the Y group, beside the share of all atom pairs those pairs are. Near Y: the part of the pair-interaction field (probe fixed on the first heavy atom of X) within 2 Å of the Y group, beside the share of the grid volume that lies there. Mode spans both: how often the leading eigenmode has at least 15 % of its weight on each group. `interacting − additive` is the difference between the two models' maps on the same molecule, which isolates the designed pair term from the curvature the Gaussian layer has on any target.

| maps of | cross-site share | share of pairs | near Y | volume near Y | mode spans both |
|---|---|---|---|---|---|
| additive model | 0.10 | 0.07 | 0.32 | 0.08 | 31 % |
| interacting model | 0.09 | 0.07 | 0.28 | 0.08 | 14 % |
| interacting − additive | 0.09 | 0.07 | 0.28 | 0.08 | 12 % |

The decisive comparison: the fitted second derivatives against the designed ones. Atoms: the cross block of the atom interaction matrix (k in X, l in Y) against 1.5 c_k c_l, with c the additive model's first derivative per atom; the slope through the origin should be 1 for the interacting model and 0 for the additive one. Field: the pair field within 2 Å of the Y group against 1.5 c_k w_k times the additive model's first-derivative field. Within: the root mean square of the fitted within-X block, which should be zero, over that of the designed cross block.

| model | slope, atoms | correlation, atoms | slope, field | correlation, field | within-group RMS / designed |
|---|---|---|---|---|---|
| additive | 0.06 | 0.30 | 0.08 | 0.42 | 0.35 |
| interacting | 1.87 | 0.73 | 1.43 | 0.67 | 2.39 |

The figure `pair_fields.png` cuts the ring plane for OCH3/NO2: the pair field of the additive and of the interacting model, and their difference, on one colour scale.
