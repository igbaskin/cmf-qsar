# The co-field around a generic residue (plan item PP5)

`python benchmarks/run_site_cofield.py`, written 2026-11-19 00:58:03, 8 minutes.

The model of the field of the surroundings joined with the change of volume, hydropathy and charge, fitted on 6386 substitutions at 1597 positions of 24 domains (draw 0 of plan item PP2). Field weight 0.25 and ridge 0.1, chosen with whole domains held out (q2 +0.424, RMSE 0.900 kcal/mol). Grid of 6859 points, -9 to 9 A on every axis of the residue frame, 1 A apart. A point is covered where the surroundings put field at all, and kept where the co-field exceeds 2 times its standard deviation under the noise of the targets.

## The map and its checks

| Field | Covered points | Kept share | Halves: median r | Halves: lowest r | Halves: same sign where kept | Frame alone: median r | Frame alone: same sign |
|---|---:|---:|---:|---:|---:|---:|---:|
| electrostatic | 852 | 0.229 | +0.130 | -0.044 | 0.869 | +0.025 | 0.505 |
| steric | 1914 | 0.577 | +0.840 | +0.786 | 0.966 | +0.035 | 0.507 |
| hydrophobic | 722 | 0.324 | +0.245 | +0.137 | 0.897 | -0.002 | 0.504 |
| acceptor | 784 | 0.246 | +0.285 | +0.160 | 0.907 | +0.021 | 0.513 |
| donor | 613 | 0.250 | +0.221 | +0.094 | 0.882 | -0.030 | 0.467 |

Halves: two models fitted on two disjoint sets of twelve domains, twenty splits. Frame alone: the same after every position is given the surroundings of another position of its domain.

## Where the kept co-field lies

The kept co-field summed in four regions of the frame (in front of the alpha carbon means along the beta carbon), and the share of kept points that are negative (where more field makes a substitution more destabilising):

| Field | in front, within 5 A | in front, 5 to 9 A | behind, within 5 A | behind, 5 to 9 A |
|---|---|---|---|---|
| electrostatic | -4.086 (0.63 negative, 41 points) | -4.045 (0.64 negative, 33 points) | +5.930 (0.30 negative, 37 points) | +3.368 (0.38 negative, 32 points) |
| steric | -29.387 (0.97 negative, 143 points) | -91.819 (1.00 negative, 466 points) | +6.109 (0.09 negative, 65 points) | -4.777 (0.65 negative, 156 points) |
| hydrophobic | -5.661 (0.85 negative, 33 points) | -3.171 (0.94 negative, 17 points) | -0.357 (0.52 negative, 54 points) | +3.514 (0.33 negative, 48 points) |
| acceptor | -7.024 (0.75 negative, 44 points) | -1.974 (0.76 negative, 21 points) | -3.747 (0.67 negative, 39 points) | -3.079 (0.71 negative, 34 points) |
| donor | -0.666 (0.43 negative, 23 points) | +0.960 (0.00 negative, 4 points) | -1.544 (0.50 negative, 48 points) | -2.754 (0.76 negative, 25 points) |

## What a site wants: classes of substitutions

| Class | Substitutions | Positions | Mean ddG, kcal/mol | Halves r, electrostatic | Halves r, steric | Halves r, hydrophobic | Halves r, acceptor | Halves r, donor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adds volume | 1919 | 925 | -0.73 | +0.15 | +0.89 | +0.37 | +0.41 | +0.36 |
| removes volume | 1656 | 891 | -1.32 | +0.24 | +0.92 | +0.44 | +0.43 | +0.11 |
| adds a positive charge | 456 | 419 | -1.21 | +0.11 | +0.90 | +0.18 | +0.27 | +0.11 |
| adds a negative charge | 480 | 441 | -1.41 | +0.25 | +0.94 | +0.41 | +0.54 | +0.33 |
| adds a donor | 1776 | 1038 | -1.03 | +0.18 | +0.91 | +0.46 | +0.43 | +0.26 |
| removes a donor | 1136 | 422 | -0.73 | +0.08 | +0.81 | +0.16 | +0.13 | +0.23 |

Pairs of classes: the correlation of their two co-fields over the covered points, and the first minus the second, averaged in front of the residue (more than 1 A along the beta carbon, within 9 A), where the new side chain sits. Halves: the share of the 20 half-models (ten splits of the domains) whose difference has the sign of the whole.

| First class | Second class | Field | r of the two co-fields | Difference in front | Halves with the same sign | Sign physics expects |
|---|---|---|---:|---:|---:|---|
| adds volume | removes volume | steric | +0.787 | +2.97e-02 | 1.00 | none stated |
| adds a positive charge | adds a negative charge | electrostatic | +0.269 | -1.00e-02 | 0.80 | negative |
| adds a donor | removes a donor | acceptor | +0.371 | -4.26e-02 | 1.00 | positive |

The figure is `docs/figures/site_cofield.png`.
