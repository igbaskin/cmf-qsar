# A migration path as the sample: the barrier of a Li hop from the field around it

`python benchmarks/run_hops.py`, reading `benchmarks/results/hops/prepared_mp.json` (written by `python benchmarks/prepare_hops.py`), seed 0, 1000 bootstrap draws.

Public set: 472 ternary compounds of lithium and oxygen (Li, O and one more element) of the Materials Project within 0.05 eV/atom of the hull, 24004 Li hops of 1.5 to 4.5 A, 1732 distinct by symmetry and labelled with the bond-valence site-energy barrier; 10 folds holding out whole structures; environments within 6.0 A of the midpoint. Fields: electrostatic, steric, prop:electronegativity, cif:Co, cif:Cr, cif:Li, cif:Mn, cif:Ni, cif:O, cif:Ti, cif:V, cif:other.

The site-resolved reference describes each of the two end sites of a hop by the row of the periodic autocorrelation of its cell that belongs to that site (plan item M5) and combines the two rows as their sum and the absolute value of their difference, so that the descriptor does not depend on which end is called first. Cutoffs of the site profile: 4, 6, 8 A. The column *against the two end sites* compares with the one at 6 A, the radius the hop environments themselves use.

The two end sites name 1195 distinct pairs for 1732 hops, and 923 hops share their pair with another hop (largest group 6): a hop and its reverse, and two hops through different images of the cell, get one descriptor. Predicting the mean barrier of every group is therefore the limit of any descriptor of the two end sites alone, at q2 +0.950 (error 0.181 eV).

| descriptor | envelope width rho | q2 | error (eV) | mean absolute error (eV) | ridge | against the counts | against length and bottleneck | against the two end sites |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| counts about the midpoint | -- | +0.491 | 0.576 | 0.403 | 0.01 | -- | -- | -- |
| hop length and bottleneck | -- | +0.103 | 0.765 | 0.510 | 1 | -- | -- | -- |
| the two end sites (M5), cutoff 4 A | -- | +0.439 | 0.604 | 0.392 | 0.1 | -0.052 [-0.128, +0.054] | +0.337 [+0.204, +0.427] | -- |
| the two end sites (M5), cutoff 6 A | -- | +0.446 | 0.601 | 0.385 | 1 | -0.045 [-0.142, +0.065] | +0.344 [+0.178, +0.452] | -- |
| the two end sites (M5), cutoff 8 A | -- | +0.462 | 0.592 | 0.383 | 1 | -0.030 [-0.122, +0.068] | +0.359 [+0.186, +0.469] | -- |
| invariant local, rho 0.05 | 0.05 | +0.553 | 0.540 | 0.357 | 0.1 | +0.062 [-0.003, +0.170] | +0.451 [+0.321, +0.538] | +0.107 [+0.030, +0.218] |
| rotation-averaged overlap, rho 0.05 | 0.05 | +0.630 | 0.491 | 0.335 | 0.01 | +0.138 [+0.085, +0.219] | +0.527 [+0.402, +0.608] | +0.184 [+0.087, +0.298] |
| invariant local, rho 0.1 | 0.1 | +0.624 | 0.495 | 0.327 | 0.01 | +0.133 [+0.066, +0.231] | +0.522 [+0.391, +0.609] | +0.178 [+0.082, +0.289] |
| rotation-averaged overlap, rho 0.1 | 0.1 | +0.649 | 0.478 | 0.325 | 0.01 | +0.158 [+0.096, +0.250] | +0.546 [+0.423, +0.628] | +0.203 [+0.102, +0.318] |
| invariant local, rho 0.25 | 0.25 | +0.657 | 0.473 | 0.307 | 0.01 | +0.165 [+0.097, +0.266] | +0.554 [+0.421, +0.643] | +0.211 [+0.117, +0.329] |
| rotation-averaged overlap, rho 0.25 | 0.25 | +0.662 | 0.469 | 0.322 | 0.01 | +0.170 [+0.112, +0.265] | +0.559 [+0.432, +0.641] | +0.216 [+0.109, +0.334] |

Overlap in the hop frame against the invariant kernel at the same width: rho 0.05: +0.076 [-0.002, +0.145]; rho 0.1: +0.024 [-0.049, +0.099]; rho 0.25: +0.005 [-0.060, +0.071].

Overlap in the hop frame against the two end sites at cutoff 4 A: rho 0.05: +0.190 [+0.108, +0.273]; rho 0.1: +0.210 [+0.124, +0.292]; rho 0.25: +0.222 [+0.134, +0.310].

Overlap in the hop frame against the two end sites at cutoff 6 A: rho 0.05: +0.184 [+0.087, +0.298]; rho 0.1: +0.203 [+0.102, +0.318]; rho 0.25: +0.216 [+0.109, +0.334].

Overlap in the hop frame against the two end sites at cutoff 8 A: rho 0.05: +0.168 [+0.082, +0.277]; rho 0.1: +0.187 [+0.096, +0.297]; rho 0.25: +0.200 [+0.104, +0.322].

Learning curve of the best overlap model (q2 against the number of training structures, three random draws): 25: +0.078; 50: +0.153; 100: +0.441; 200: +0.498; 400: +0.660; 459: +0.662.

## The co-field along the hop (z, A from the midpoint) and away from its axis (r, A), best model at rho 0.25

Values in eV of barrier per unit of normalised field, with the standard deviation the out-of-fold error (0.469 eV) allows in brackets.

**electrostatic**

| z \ r | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| -2 | -1.42 (0.81) | -0.01 (0.45) | -0.39 (0.32) | -0.07 (0.27) | +0.32 (0.26) |
| -1 | -2.62 (1.07) | +0.10 (0.43) | +0.26 (0.31) | +0.36 (0.27) | +0.48 (0.25) |
| +0 | -1.15 (1.11) | +0.38 (0.41) | -0.46 (0.32) | -0.10 (0.23) | -0.16 (0.22) |
| +1 | -1.52 (1.07) | -0.02 (0.43) | +0.16 (0.31) | +0.82 (0.27) | +0.62 (0.24) |
| +2 | -1.77 (0.80) | -0.67 (0.44) | -0.47 (0.33) | +0.35 (0.27) | +0.30 (0.26) |

**steric**

| z \ r | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| -2 | +1.49 (0.74) | -0.23 (0.44) | -1.13 (0.36) | -0.50 (0.25) | +0.02 (0.21) |
| -1 | +2.61 (0.96) | +0.33 (0.56) | -0.86 (0.44) | -0.09 (0.27) | +0.14 (0.20) |
| +0 | +2.15 (1.74) | +0.60 (0.72) | -0.49 (0.47) | +0.24 (0.31) | +0.07 (0.22) |
| +1 | +2.84 (0.94) | +0.70 (0.56) | -0.56 (0.44) | -0.03 (0.27) | +0.08 (0.19) |
| +2 | +1.68 (0.73) | -0.11 (0.43) | -1.04 (0.37) | -0.64 (0.25) | -0.10 (0.21) |

**prop:electronegativity**

| z \ r | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| -2 | -1.20 (0.61) | -1.82 (0.40) | -0.93 (0.33) | -0.34 (0.26) | -0.75 (0.21) |
| -1 | -2.73 (0.89) | -1.36 (0.46) | +0.42 (0.36) | +0.28 (0.27) | -0.75 (0.21) |
| +0 | -2.18 (1.00) | +0.58 (0.43) | +1.45 (0.35) | -0.09 (0.25) | -1.08 (0.23) |
| +1 | -2.57 (0.89) | -1.31 (0.46) | +0.11 (0.36) | +0.34 (0.27) | -0.56 (0.21) |
| +2 | -1.58 (0.60) | -2.13 (0.39) | -1.10 (0.32) | -0.33 (0.26) | -0.79 (0.21) |

**cif:Co**

| z \ r | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| -2 | +0.66 (0.27) | +0.33 (0.29) | -0.14 (0.32) | -0.14 (0.23) | +0.00 (0.23) |
| -1 | +1.90 (0.62) | +0.63 (0.46) | -0.63 (0.31) | +0.31 (0.26) | +0.55 (0.21) |
| +0 | +3.10 (0.85) | +1.14 (0.39) | -0.90 (0.30) | +0.29 (0.20) | +0.59 (0.28) |
| +1 | +1.94 (0.60) | +0.85 (0.40) | -0.45 (0.30) | -0.05 (0.27) | +0.22 (0.22) |
| +2 | +0.76 (0.24) | +0.54 (0.28) | +0.05 (0.33) | -0.21 (0.23) | -0.05 (0.23) |

**cif:Cr**

| z \ r | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| -2 | +0.03 (0.25) | +0.02 (0.29) | +0.00 (0.30) | -0.03 (0.25) | -0.02 (0.22) |
| -1 | +0.39 (0.48) | +0.31 (0.42) | +0.08 (0.28) | -0.04 (0.26) | +0.02 (0.20) |
| +0 | +0.30 (0.64) | +0.10 (0.40) | -0.14 (0.32) | +0.03 (0.21) | +0.18 (0.23) |
| +1 | +0.11 (0.46) | -0.09 (0.41) | -0.37 (0.31) | -0.13 (0.25) | +0.22 (0.20) |
| +2 | +0.02 (0.22) | +0.06 (0.27) | +0.26 (0.29) | +0.43 (0.26) | +0.32 (0.23) |

**cif:Li**

| z \ r | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| -2 | -0.68 (0.79) | -0.95 (0.44) | +0.56 (0.35) | -0.72 (0.28) | -0.36 (0.22) |
| -1 | +1.81 (0.95) | -1.29 (0.47) | +0.25 (0.34) | +1.20 (0.25) | -0.45 (0.24) |
| +0 | +5.15 (1.20) | -0.56 (0.45) | -0.75 (0.29) | -0.49 (0.26) | -0.80 (0.23) |
| +1 | +2.14 (0.95) | -1.23 (0.48) | +0.56 (0.34) | +1.36 (0.25) | -0.76 (0.24) |
| +2 | -0.29 (0.80) | -1.43 (0.45) | +0.14 (0.35) | -0.88 (0.29) | -0.34 (0.22) |

**cif:Mn**

| z \ r | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| -2 | +1.12 (0.31) | +0.54 (0.32) | -0.61 (0.31) | -0.25 (0.26) | +0.25 (0.22) |
| -1 | +1.82 (0.55) | +1.12 (0.43) | +0.34 (0.29) | +0.75 (0.26) | +0.19 (0.23) |
| +0 | +1.09 (0.68) | -0.14 (0.41) | -0.98 (0.30) | -0.08 (0.20) | -0.44 (0.23) |
| +1 | +0.35 (0.58) | -0.15 (0.45) | +0.14 (0.29) | +0.88 (0.27) | +0.15 (0.23) |
| +2 | +0.31 (0.32) | -0.01 (0.34) | -0.40 (0.31) | -0.10 (0.26) | +0.10 (0.23) |

**cif:Ni**

| z \ r | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| -2 | -0.03 (0.25) | -0.15 (0.27) | -0.07 (0.28) | +0.14 (0.22) | +0.27 (0.20) |
| -1 | -0.84 (0.37) | -0.38 (0.30) | +0.53 (0.35) | +0.25 (0.24) | +0.09 (0.20) |
| +0 | -1.51 (0.62) | -0.87 (0.40) | +0.35 (0.24) | -0.20 (0.19) | -0.31 (0.23) |
| +1 | -0.62 (0.37) | -0.16 (0.31) | +0.46 (0.37) | -0.24 (0.25) | -0.30 (0.20) |
| +2 | +0.21 (0.25) | +0.18 (0.25) | +0.28 (0.27) | +0.14 (0.22) | +0.02 (0.21) |

**cif:O**

| z \ r | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| -2 | +0.77 (0.62) | -0.23 (0.39) | +0.56 (0.31) | +0.65 (0.24) | -0.38 (0.20) |
| -1 | +1.96 (0.98) | +0.10 (0.44) | +0.18 (0.35) | +0.89 (0.25) | -0.27 (0.20) |
| +0 | +3.03 (1.06) | +0.42 (0.44) | -1.31 (0.31) | -0.28 (0.25) | -0.50 (0.21) |
| +1 | +1.89 (0.98) | +0.19 (0.44) | -0.13 (0.35) | +0.76 (0.25) | -0.12 (0.20) |
| +2 | +0.67 (0.62) | -0.14 (0.40) | +0.61 (0.31) | +0.63 (0.24) | -0.39 (0.20) |

**cif:Ti**

| z \ r | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| -2 | +0.12 (0.18) | +0.21 (0.25) | +0.42 (0.31) | +0.10 (0.24) | -0.03 (0.23) |
| -1 | -0.28 (0.41) | -0.41 (0.35) | -0.05 (0.31) | -0.16 (0.25) | -0.34 (0.22) |
| +0 | -0.47 (0.60) | -0.62 (0.36) | +0.32 (0.30) | +0.57 (0.22) | -0.13 (0.22) |
| +1 | -0.55 (0.40) | -0.69 (0.36) | -0.14 (0.31) | -0.16 (0.25) | -0.54 (0.23) |
| +2 | -0.04 (0.17) | +0.02 (0.24) | +0.36 (0.32) | +0.20 (0.25) | -0.13 (0.23) |

**cif:V**

| z \ r | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| -2 | -0.11 (0.22) | -0.23 (0.27) | -0.32 (0.30) | +0.14 (0.26) | +0.11 (0.23) |
| -1 | +0.49 (0.42) | +0.48 (0.35) | -0.02 (0.30) | +0.01 (0.27) | +0.28 (0.22) |
| +0 | +0.76 (0.65) | +0.73 (0.42) | -0.21 (0.30) | -0.33 (0.23) | +0.32 (0.21) |
| +1 | +0.40 (0.45) | +0.42 (0.36) | +0.04 (0.29) | +0.07 (0.27) | +0.22 (0.22) |
| +2 | -0.06 (0.24) | -0.05 (0.29) | +0.04 (0.31) | +0.10 (0.25) | -0.26 (0.22) |

**cif:other**

| z \ r | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| -2 | +0.04 (0.43) | +0.03 (0.38) | +0.67 (0.33) | -0.26 (0.27) | -0.42 (0.22) |
| -1 | +0.45 (0.73) | -0.58 (0.47) | -0.13 (0.31) | +0.11 (0.26) | -0.07 (0.24) |
| +0 | +1.69 (0.90) | -0.12 (0.38) | -0.21 (0.32) | +0.39 (0.28) | -0.32 (0.22) |
| +1 | +1.28 (0.74) | +0.05 (0.46) | -0.41 (0.32) | -0.01 (0.27) | -0.02 (0.24) |
| +2 | +0.22 (0.43) | +0.03 (0.40) | +0.40 (0.33) | +0.18 (0.26) | -0.19 (0.21) |

The same co-field on a finer grid is `profile_fine.json`, written by `python benchmarks/run_hops.py --fine-profile`; the manual draws it as a figure.

## From hops to conduction: percolation thresholds of the predicted network

| direction | structures with a finite threshold in both | correlation | mean absolute difference (eV) | finite or not agrees |
| --- | ---: | ---: | ---: | ---: |
| 1D | 164 | +0.808 | 0.380 | 100% |
| 2D | 110 | +0.857 | 0.351 | 100% |
| 3D | 60 | +0.754 | 0.343 | 100% |

## The three hosts of the Li interface study

The public model applied to the 48 distinct hops of Li3PO4, Li3BP2O8 and LiBO2 (bond-valence labels): q2 -1.415, error 0.279 eV, correlation +0.683.
- phosphate: 20 hops, correlation +0.837, mean absolute error 0.231 eV
- borophosphate: 18 hops, correlation +0.784, mean absolute error 0.253 eV
- borate: 10 hops, correlation -0.490, mean absolute error 0.155 eV

- leave one host out, rotation-averaged overlap, leave one host out: q2 -0.120, error 0.190 eV
- leave one host out, counts about the midpoint: q2 -0.115, error 0.190 eV

| host | hop | length (A) | DFT barrier (eV) | DFT reverse | bond-valence | bond-valence reverse | predicted | bottleneck (A) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phosphate | 34 -> 0 | 2.98 | 0.287 | 0.469 | 0.732 | 0.656 | 0.798 | 1.26 (O) |
| borophosphate | 18 -> 8 | 2.64 | 0.003 | 0.276 | 0.393 | 0.347 | 0.561 | 1.48 (O O) |

The DFT barriers are plain elastic-band values for a vacancy hop into a relaxed vacancy, so they are lower bounds and include the relaxation the rigid bond-valence landscape does not; the columns are not expected to agree in value, only in order.

## The disordered conductors (plan item M2)

13 structures of the Crystallography Open Database, 6408 hops, 298 distinct, 298 with a bond-valence barrier. The public model applied as it stands: q2 -1.075, error 1.437 eV, correlation +0.273.
- leave one structure out, rotation-averaged overlap, leave one structure out: q2 -0.064, error 1.030 eV
- leave one structure out, counts about the midpoint: q2 +0.166, error 0.912 eV
- the frame kernel against the counts, leave one structure out: -0.230 [-0.884, +0.143]

| structure | what is disordered | hops | distinct | mean barrier (eV) | correlation with the public model | E_1D / E_2D / E_3D labelled (eV) | public model | within the set |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| LLZO:Ta cubic (Li6.5La3Zr1.5Ta0.5O12) (COD 1545083) | Li sites 0.38/0.62, Zr/Ta shared | 1728 | 18 | 0.22 | -0.09 | -- | -- | -- |
| LLZO:Ta cubic (Li6.2La3Zr1.2Ta0.8O12) (COD 1552152) | Li sites 0.28/0.96, Zr/Ta shared | 1824 | 19 | 0.25 | -0.06 | -- | -- | -- |
| LLZO:Ta tetragonal (COD 1545085) | ordered | 416 | 13 | 1.53 | +0.61 | -- | -- | -- |
| LLTO rhombohedral (La0.5Li0.5TiO3) (COD 1525568) | Li sites 0.16 | 252 | 7 | 1.86 | -0.32 | -- | -- | -- |
| LLTO tetragonal (La0.56Li0.33TiO3) (COD 1000437) | La/Li shared sites | 12 | 4 | 4.00 | -0.76 | -- | -- | -- |
| LGPS (Li10GeP2S12) (COD 7215190) | Li sites 0.47 to 0.89, Ge/P shared | 416 | 26 | 0.40 | +0.52 | -- | -- | -- |
| LiTi2(PO4)3 NASICON (COD 7222155) | ordered | 0 | 0 | nan | -- | inf / inf / inf | inf / inf / inf | inf / inf / inf |
| Li3.18Ti2(PO4)3 NASICON (COD 7222156) | Li sites 0.34/0.72 | 216 | 12 | 0.62 | +0.84 | -- | -- | -- |
| Li2.92N (COD 4002499) | Li site 0.96 | 72 | 9 | 0.14 | +0.06 | -- | -- | -- |
| LISICON (Li13.7Zn1.2Ge4O16) (COD 1524557) | Li sites 0.16 to 0.98, Li/Zn shared | 416 | 55 | 0.52 | +0.23 | -- | -- | -- |
| Li3PO4 (Pmnb) (COD 9011044) | ordered | 144 | 18 | 0.35 | +0.69 | -- | -- | -- |
| beta-Li3PS4 (COD 1570310) | Li sites 0.3/0.7 | 152 | 19 | 0.39 | +0.18 | -- | -- | -- |
| Li3.16Zn0.24GeO4 (COD 1529523) | Li sites 0.06 to 0.92, Li/Zn shared | 760 | 98 | 0.29 | +0.50 | -- | -- | -- |

The bond-valence landscape of a disordered structure is that of its ordered framework (the sites not held by Li, each with its majority species); the fields see every site at its occupancy. A threshold of inf means the hops between 1.5 and 4.5 A do not connect through the cell in that many directions.


## The model read through the interfaces (plan item MS7)

The public model, rebuilt at the stored width (rho 0.25) and ridge (0.01) on 1732 paths, is read twice: by the route this script uses -- environments built here and handed to the model -- and through `cmfqsar.interfaces.HopBarriers`, which is given a structure and builds the environments itself. The structures are the three hosts of the lithium interface study, which the public model has never seen.

**The two routes give the same barriers.** The interface is a way in, not another model:

| Paths of the hosts | Distinct by symmetry | Compared by both routes | Largest difference |
| ---: | ---: | ---: | ---: |
| 1040 | 48 | 48 | 1.2e-11 eV |

**The stored numbers come back.** The two paths the benchmark lists against a DFT elastic-band barrier, as the stored table has them and as the interface gives them:

| Host | Path | Stored | Through the interface | Difference |
| --- | --- | ---: | ---: | ---: |
| phosphate | 34 -> 0 | 0.7979 | 0.7979 | +0.0e+00 |
| borophosphate | 18 -> 8 | 0.5610 | 0.5610 | +0.0e+00 |

In eV.

Both of these paths are equivalent by symmetry to another one of their cell, so the network a simulation reads carries their group's barrier rather than their own; the two differ by up to 0.00 eV, which is the error of the symmetry grouping and not of the interface. A caller who wants one particular jump asks for that path by name, which is what the column above reports.

**And so do the percolation thresholds**, which is what a conductivity estimate starts from: the lowest barrier at which the paths form a connected network in one, two and three dimensions, by this script's route and through the interface.

| Host | E_1D | E_2D | E_3D |
| --- | --- | --- | --- |
| phosphate | 0.403 / 0.403 | 0.452 / 0.452 | 0.798 / 0.798 |
| borophosphate | 0.528 / 0.528 | 0.654 / 0.654 | 1.571 / 1.571 |
| borate | 0.281 / 0.281 | 0.437 / 0.437 | none / none |

In eV above the lowest site, by this script's route / through the interface.

The interface took 106 s on these paths against 113 s for the route inside this script. Both pay for the same two things, the environments and the kernel against the training set; the difference is that the interface works one structure at a time, so it evaluates the kernel in three blocks instead of one.
