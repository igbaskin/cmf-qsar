# Notes of the COMPAS-1 pilot

Written by benchmarks/run_compas.py on 2026-10-29.

## Data

| file | URL | bytes | SHA-256 | downloaded |
|---|---|---|---|---|
| compas-1D_b3lyp_def2-svp.csv | https://gitlab.com/porannegroup/compas/-/raw/main/COMPAS-1/compas-1D_b3lyp_def2-svp.csv | 1946817 | a2c33ad8e6a581d037057814d9b6ae403d3a01003db32b6660550d25b8b7606b | 2026-10-29 |
| compas-1D_b3lyp_def2-svp.tar.gz | https://gitlab.com/porannegroup/compas/-/raw/main/COMPAS-1/compas-1D_b3lyp_def2-svp.tar.gz | 15519759 | 5c055e63f486fbf5879d8d0bb8daa133f71451c34b21bc29b189aa832d7f6afa | 2026-10-29 |
| README.md | https://gitlab.com/porannegroup/compas/-/raw/main/COMPAS-1/README.md | 5267 | 6943151dde80a04b3b3f1c0be635237c6fc0d42baa374fe40dce15c0d79eea22 | 2026-10-29 |
| compas-1x.csv | https://gitlab.com/porannegroup/compas/-/raw/main/COMPAS-1/compas-1x.csv | 9432211 | b8851c73f8dbddef2c49bf43fe21d1399d1b68e1f8583f1ecf5180f302c8119d | 2026-10-29 |
| compas-1x.tar.gz | https://gitlab.com/porannegroup/compas/-/raw/main/COMPAS-1/compas-1x.tar.gz | 67414863 | f7b643b4b3f1edb6d0128ee419fe47d67d22701be5271f59de20a3450ec8a05e | 2026-10-29 |

COMPAS-1D: 8678 molecules at B3LYP-D3BJ/def2-SVP; the GitLab repository porannegroup/compas, branch main (the raw-file endpoint carries no commit hash; the checksums above identify the files).  Licence CC-BY-NC-SA: the files stay in benchmarks/cache and are not committed.

## Substitutions and deviations from the task text

- cif:C.3r dropped: no carbon of three rings in a cata-condensed set
- No separate package or YAML configuration: the study is this benchmark script with the library module cmfqsar/pbh.py, as every other study of the repository.
- The steric field is the program's (Tripos well depths, widths scaled by the van der Waals radius, hydrogens included), not a unit-weight heavy-atom field.
- The field scales (unit mean kernel diagonal) are taken over the whole set rather than the training part, so that one stored kernel per attenuation serves every split; the scale is a constant per field and the difference is a rescaling of the mixing coefficients.
- The attenuations are chosen on a random subset of 2,500 training molecules (5-fold CV); the ridge is then chosen by 5-fold CV on the whole training part.
- The aromaticity field E is the Clar-sextet share of every ring (placeholder); a NICS-weighted ring field is a follow-up that needs a DFT calculation per ring.
- Literature numbers for the gap on COMPAS-1 are not filled in (table row 'TBD').
- The Nystroem scaling experiment on COMPAS-1x is the second milestone; the stage exists (--stage scaling) and was smoke-tested on a subset only.

## Decisions

- Frame: heavy-atom inertia tensor with atomic masses; x along the smallest moment, z along the largest; the signs of x and y by the moments sum x^3, sum y^3, sum x^2 y, sum x^3 y, sum x y^3 in that order (cmfqsar.pbh.resolve_signs); molecules with equal in-plane moments take x along the largest third (else fourth) moment.
- The kernel-based choice among the four frames (alignment by prediction confidence) is implemented (cmfqsar.pbh.choose_frame_by_kernel) and off by default; the rule leaves nothing to choose for a planar molecule without symmetry.
- Ring classes from the fused-neighbour graph of the six-membered carbon rings; checked against the 'lalas' annulation strings of the table (0 disagreements).
- Kernel ridge regression with the bias of Eq. 12 (2013); inside the cross-validation the same solution is obtained from the eigendecomposition of the centred kernel (checked against krr_fit at every run).
- Attenuation grid (0.1, 0.2, 0.5, 1.0, 2.0, 5.0) A^-2 in the program's convention (exponent alpha/2); ridge grid (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0).
- GFN2-xTB charges with xtb * xtb version 6.7.1pre (5071a88) compiled by 'Marcel@Raven' on 2024-07-23; the input geometry is kept (no optimisation), no Fukui run.

## Open questions

- Whether the gap of the few non-planar molecules (helicene-like fjords) is served by the planar frame; the frame is defined for them but the two faces differ.
- The size-extrapolation split trains on few molecules (the sets are dominated by ten-ring molecules), so its numbers mix extrapolation with a small training set.
- Published COMPAS-1 gap-prediction numbers for the literature row.
- The chosen hyperparameters sit at the edges of the grids: the steric attenuation at the top (its width is scaled by the van der Waals radius, so the same grid value is a wider Gaussian) and the ridge at the bottom (1e-4) for the sets with the ring fields. A wider steric grid and a smaller ridge are worth one more run; the float32 kernels limit the ridge to about 1e-5.
- Seed 0 of the random splits puts benzene, the only one-ring molecule, in the test part; the main model extrapolates it to 10-13 eV (computed 6.72), which is the whole difference between that seed and the other four (table.md gives the RMSE over the molecules of three rings and more).
- The Clar-sextet ring field (E), meant as a placeholder, is the single largest gain on top of {A,B,C,D}; the NICS-weighted field should be compared against it, not against the set without it.
- The per-ring contributions (rule 3) are large and cancel: a ring on the +x side of the frame counts positive and one on the -x side negative, whatever its class, because the co-field has a smooth component across the frame and a ridge of 1e-4 lets the coefficients grow. Their sum is the prediction; the individual values read position as much as chemistry. A decomposition against a reference (the Shapley values of cmfqsar/shapley.py) is the next thing to try for this series.
