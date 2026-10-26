# The three kernels where no frame exists

Association constants with a fixed cyclodextrin host, `examples/cyclodextrin/`. The
guests of one table share no scaffold, so the superposition the `aligned` column uses
is a maximum common substructure fit onto the largest guest of the most active
quartile, which is the best the program can do without a receptor. The `invariant`
column is the field autocorrelation, which needs no frame; `counts` are atom-type
counts, which need no geometry. Same fields (electrostatic, steric, hydrophobic, hba, hbd), same
alpha and lambda grids and the same 10-fold CV as
`run_alignment.py`, so the columns are comparable with that benchmark.

## The neutral tables

| host | n | aligned | invariant | counts | MCS atoms (median) | heavy atoms (median) |
|---|---|---|---|---|---|---|
| dimethyl_beta | 43 | +0.155 | +0.380 | +0.350 | 9 | 18 |
| methyl_beta | 77 | +0.500 | +0.530 | +0.386 | 8 | 11 |
| sbe_beta | 135 | +0.327 | +0.383 | +0.257 | 9 | 15 |
| hp_beta | 135 | +0.288 | +0.571 | +0.410 | 7 | 15 |
| gamma | 292 | +0.451 | +0.485 | +0.363 | 9 | 19 |
| alpha | 827 | +0.336 | +0.323 | +0.178 | 5 | 12 |
| beta | 1162 | +0.329 | +0.447 | +0.318 | 8 | 14 |

Mean q2 over the 7 hosts measured: aligned +0.341, invariant +0.446, counts +0.323.

The kernel that uses no frame is ahead of the one that does on 6 of the 7 hosts, and ahead of
atom-type counts on all of them. The aligned column falls below counts on 2, which is the
frame failing rather than the fields: the MCS column is the explanation, a superposition built on
half a molecule or less. This is the regime `run_alignment.py` reaches only on its worst targets,
and here it is the whole data set.

## Neutral against the structures at pH 7

The same guests in their protonation state at the pH the constants were measured
at, `examples/cyclodextrin/ph7/`. The `charged` column counts the guests that
carry a formal charge there.

| host | n | charged at pH 7 | aligned neutral | aligned pH 7 | invariant neutral | invariant pH 7 | counts neutral | counts pH 7 |
|---|---|---|---|---|---|---|---|---|
| dimethyl_beta | 43 | 26 | +0.155 | +0.235 | +0.380 | +0.319 | +0.350 | +0.342 |
| methyl_beta | 77 | 21 | +0.500 | +0.334 | +0.530 | +0.534 | +0.386 | +0.384 |
| sbe_beta | 135 | 27 | +0.327 | +0.268 | +0.383 | +0.385 | +0.257 | +0.259 |
| hp_beta | 135 | 47 | +0.288 | +0.347 | +0.571 | +0.499 | +0.410 | +0.437 |
| gamma | 292 | 122 | +0.451 | +0.441 | +0.485 | +0.437 | +0.363 | +0.382 |
| alpha | 827 | 444 | +0.336 | +0.303 | +0.323 | +0.340 | +0.178 | +0.186 |
| beta | 1162 | 601 | +0.329 | +0.335 | +0.447 | +0.490 | +0.318 | +0.329 |

Mean q2 at pH 7: aligned +0.323, invariant +0.429, counts +0.331.

Protonating the guests is not settled by this measurement. The invariant kernel improves on 4 of
the 7 hosts (methyl_beta, sbe_beta, alpha, beta) and loses elsewhere, so the mean barely moves.
The two largest sets, beta and alpha, are among those that improve, which is what one would expect if the
charges carry signal that only a large enough set can use: on the small hosts the extra spread in
the electrostatic field costs more than it returns. Read it as helping where there are guests
enough to learn from.

For scale, the published field-based models of this endpoint: CoMFA with COSMOsar3D
fields on 60 alpha-CD guests (Q2 0.83, r2test 0.70), GRIND on 126 beta-CD guests
(Q2LOO 0.64, r2test 0.68) and GRIND on 233 beta-CD guests (Q2LOO 0.75, r2test 0.74).
Those sets are smaller and hand-curated, so these numbers are not a direct
comparison with them.

