# Benchmark against Table 2 of Baskin & Zhokhova, JCAMD 27 (2013) 427–442

This document compares the `cmfqsar` implementation with the CMF models published in the 2013
paper on the eight literature data sets of Sutherland et al. (J. Med. Chem. 47 (2004) 5541):
ACE, AChE, BZR, COX-2, DHFR, GPB, THER and THR, using Sutherland's alignments, ionisation states,
conformations, partial charges and train/test splits.  All numbers below were produced with
`benchmarks/run_sutherland.py`, which drives `cmf_cli.py build`; the raw results (statistics,
hyper-parameters, per-molecule predictions) are in `benchmarks/results/*.json`.

## Summary

* With the protocol of the paper the program reproduces **47 of the 48 numbers of Table 2 to the
  printed precision**; the one exception (THER q²ex 0.33 vs 0.40) is explained below.  Against the
  unrounded values of the original R runs (the author's `results.xlsx`) the agreement is to four
  decimals, e.g. ACE: q² 0.7178, RMSEcv 1.2431, R²pred 0.5918, RMSEpred 1.3530, q²ex 0.6674,
  RMSEcvex 1.3088.  The analytic kernels, the KRR solver, the cross-validation and the nested
  cross-validation are therefore validated against the original code.
* Reproducing the paper required three additions that are now options of the program: the in-house
  field increments of the original R scripts (`--param tripos`), unshuffled interleaved folds
  (`--cv-no-shuffle`) and a continuous Brent minimisation of RMSEcv over λ (`--lam-search brent`);
  the raw kernel sum with h_f = 1 (`--raw-scale`) and the MOL2 charges (`--charges file`) were
  already available.
* The original scripts contain a **hydrogen-counting bug** (`cmf-params.R`: `el2` is read from the
  first atom of the bond), so that in practice every atom received the "no attached hydrogen"
  increment; the published hydrophobic and hydrogen-bond fields are therefore not what the
  parameterisation intended.  `--param tripos-legacy` reproduces this behaviour; `--param tripos`
  counts hydrogens correctly.  The correction changes the statistics by a few hundredths on most sets
  (THER improves, THR deteriorates).
* The default settings of the program (RDKit-based increments, field kernels scaled to unit mean
  diagonal, shuffled folds, λ grid) give models of similar quality in cross-validation but deviate
  from the paper on the single external test sets, mainly because the scaling convention changes
  the implicit weights of the five fields.  For reproducing the papers, the raw convention
  (`--raw-scale`, *Scale fields* unticked in the GUI) must be used.

## Data sets

| Set | Molecules used (train / test) | Charges | Activity |
|---|---|---|---|
| ACE | 114 (76 / 38) | MNDO ESP | pIC50 |
| AChE | 111 (74 / 37) | MNDO ESP | pIC50 |
| BZR | 147 (98 / 49); 16 inactives of the 163 excluded | MNDO ESP | pIC50 |
| COX-2 | 282 (188 / 94); 40 inactives of the 322 excluded | MNDO ESP | pIC50 |
| DHFR | 361 (237 / 124); 36 inactives of the 397 excluded | MNDO ESP | pIC50 |
| GPB | 66 (44 / 22) | MNDO ESP | pKi |
| THER | 76 (51 / 25) | Gasteiger–Marsili | pKi |
| THR | 88 (59 / 29) | MNDO ESP | pKi |

Files: `<set>_data/<prefix>-all.mol2` (multi-molecule MOL2 with Sybyl atom types and charges) and
`<set>_data/<set>-data-sybyl-all.txt` (tab table `NAME ACTIVITY SET`, SET 1 = training, 2 = test),
as distributed with the supplementary R scripts of the paper.  All molecules are read; RDKit's MOL2
reader rejected one ACE molecule (THIOL_28, an indole N.pl3 it cannot kekulise), which is now
handled by a tolerant fallback parser.

## The protocol of the paper, as reconstructed from the original R scripts

The published models were produced with the `cmf-*.R` scripts (`cmf-params.R`, `cmf-kernels.R`,
`cmf-krr.R`, `cmf-tripos.prm`).  Their protocol is:

1. Five fields: partial charges from the MOL2 file (`q`), steric (`vdw`: Tripos well depth E by
   Sybyl type, width α/R² with the Tripos radius by Sybyl type), hydrophobic (`logp`), hydrogen-bond
   acceptor (`abrb`, Abraham B) and donor (`abra`, Abraham A) with in-house one-atom increments keyed
   by Sybyl type and number of attached hydrogens.  Hydrogen atoms carry no increments, so the donor
   field is centred on the heavy atom.
2. Kernels: analytic Gaussian overlaps, `w_l w_m (π/α)^{3/2} exp(−α d²/4)` for ordinary fields and
   `w_l w_m (2π/(b_l+b_m))^{3/2} exp(−b_l b_m d²/(2(b_l+b_m)))` with `b = α/R²` for the steric field;
   the joint kernel is the **raw sum** of the five field kernels (h_f = 1, no normalisation, no
   scaling).
3. KRR with the bias term of Eq. 12 (`set_b_0 = 0`); the ACE values of q², RMSEcv, R²pred and
   RMSEpred in Table 2 come from a run *without* bias (`set_b_0 = 1`), its q²ex/RMSEcvex from the
   run with bias (both appear in the author's `results.xlsx`).
4. Hyper-parameters: α shared by all fields on the grid 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5,
   10; λ (γ in the scripts) by R's `optimize()` (Brent) on [0.01, 20], the objective being the
   smallest RMSEcv over the α grid for the given λ.
5. 10-fold cross-validation with **interleaved folds in file order** (molecule i in fold i mod 10),
   no shuffling; the external 5-fold cross-validation on **all** molecules (training + test) with the
   same interleaved folds, the inner 10-fold loop selecting α and λ per outer fold.

The equivalent `cmf_cli.py` invocation (mode `paper` of the benchmark script):

```bash
python cmf_cli.py build ace-3dqsar-mndo-all.mol2 --activity-file ace-data-sybyl-all.txt --param tripos-legacy \
       --charges file --raw-scale --cv-no-shuffle --lam-search brent \
       --alphas 0.01,0.05,0.1,0.25,0.5,0.75,1,2,3,5,10 --lams 0.01,20 --nested --nested-on all
```

## Results

Modes: *paper protocol* = the command above (`--param tripos-legacy`); *b = 0* = the same without
bias (ACE only, as in the paper); *corrected H counts* = `--param tripos`; *RDKit parameterisation*
= `--param rdkit` (Wildman–Crippen logP contributions, built-in H-bond increments with the donor
weight on the hydrogen, element-based vdW parameters), otherwise the paper protocol; *RDKit param.,
scaled field kernels* = the same with every field kernel scaled to unit mean diagonal before
summation (the program's default convention); *program defaults* = `--param rdkit --charges file`
with the default α grid (0.2 … 3), λ grid (10⁻⁴ … 10), scaled field kernels and shuffled folds
(seed 0).  α and λ are the selected hyper-parameters of the model built on the training set.

<!-- TABLE -->
| Set | Model | q2 | RMSEcv | R2pred | RMSEpred | q2ex | RMSEcvex | alpha | lambda |
|---|---|---|---|---|---|---|---|---|---|
| **ACE** | paper (Table 2) | 0.72 | 1.24 | 0.65 | 1.24 | 0.67 | 1.31 | | |
| | this work, paper protocol | 0.72 | 1.24 | 0.59 | 1.35 | 0.67 | 1.31 | 0.50 | 20.000 |
| | this work, paper protocol, b = 0 | 0.72 | 1.24 | 0.65 | 1.24 | 0.68 | 1.28 | 1.00 | 20.000 |
| | this work, corrected H counts | 0.72 | 1.24 | 0.60 | 1.34 | 0.67 | 1.30 | 0.75 | 10.570 |
| | this work, RDKit parameterisation | 0.73 | 1.22 | 0.67 | 1.21 | 0.69 | 1.27 | 1.00 | 16.464 |
| | this work, RDKit param., scaled field kernels | 0.71 | 1.27 | 0.62 | 1.30 | 0.68 | 1.28 | 5.00 | 0.010 |
| | this work, program defaults | 0.73 | 1.22 | 0.63 | 1.28 | 0.72 | 1.20 | 3.00 | 0.010 |
| **AChE** | paper (Table 2) | 0.58 | 0.79 | 0.64 | 0.77 | 0.54 | 0.84 | | |
| | this work, paper protocol | 0.58 | 0.79 | 0.64 | 0.77 | 0.54 | 0.84 | 3.00 | 0.417 |
| | this work, corrected H counts | 0.59 | 0.77 | 0.65 | 0.77 | 0.56 | 0.82 | 3.00 | 0.452 |
| | this work, RDKit parameterisation | 0.59 | 0.77 | 0.65 | 0.76 | 0.57 | 0.81 | 3.00 | 0.396 |
| | this work, RDKit param., scaled field kernels | 0.60 | 0.76 | 0.45 | 0.96 | 0.51 | 0.87 | 1.00 | 0.019 |
| | this work, program defaults | 0.59 | 0.78 | 0.36 | 1.03 | 0.55 | 0.83 | 1.00 | 0.010 |
| **BZR** | paper (Table 2) | 0.40 | 0.51 | 0.20 | 0.79 | 0.27 | 0.65 | | |
| | this work, paper protocol | 0.40 | 0.51 | 0.20 | 0.79 | 0.27 | 0.65 | 0.01 | 12.259 |
| | this work, corrected H counts | 0.41 | 0.51 | 0.16 | 0.82 | 0.24 | 0.67 | 3.00 | 0.506 |
| | this work, RDKit parameterisation | 0.42 | 0.50 | 0.08 | 0.85 | 0.30 | 0.64 | 2.00 | 1.198 |
| | this work, RDKit param., scaled field kernels | 0.41 | 0.51 | 0.07 | 0.86 | 0.29 | 0.64 | 0.50 | 0.729 |
| | this work, program defaults | 0.43 | 0.50 | 0.07 | 0.86 | 0.31 | 0.64 | 0.50 | 1.000 |
| **COX2** | paper (Table 2) | 0.57 | 0.67 | 0.14 | 1.23 | 0.41 | 0.89 | | |
| | this work, paper protocol | 0.57 | 0.67 | 0.14 | 1.23 | 0.41 | 0.89 | 2.00 | 0.816 |
| | this work, corrected H counts | 0.60 | 0.64 | 0.18 | 1.20 | 0.46 | 0.86 | 1.00 | 1.282 |
| | this work, RDKit parameterisation | 0.61 | 0.63 | 0.31 | 1.10 | 0.51 | 0.82 | 1.00 | 2.448 |
| | this work, RDKit param., scaled field kernels | 0.61 | 0.64 | 0.27 | 1.13 | 0.50 | 0.83 | 1.00 | 0.263 |
| | this work, program defaults | 0.59 | 0.65 | 0.21 | 1.17 | 0.50 | 0.82 | 0.70 | 1.000 |
| **DHFR** | paper (Table 2) | 0.67 | 0.73 | 0.65 | 0.80 | 0.67 | 0.75 | | |
| | this work, paper protocol | 0.67 | 0.73 | 0.65 | 0.80 | 0.67 | 0.75 | 10.00 | 0.152 |
| | this work, corrected H counts | 0.68 | 0.72 | 0.64 | 0.81 | 0.67 | 0.74 | 5.00 | 0.544 |
| | this work, RDKit parameterisation | 0.65 | 0.75 | 0.51 | 0.95 | 0.62 | 0.80 | 0.10 | 9.293 |
| | this work, RDKit param., scaled field kernels | 0.66 | 0.74 | 0.64 | 0.82 | 0.67 | 0.75 | 10.00 | 0.560 |
| | this work, program defaults | 0.64 | 0.76 | 0.54 | 0.92 | 0.63 | 0.79 | 0.20 | 0.100 |
| **GPB** | paper (Table 2) | 0.69 | 0.60 | 0.51 | 0.84 | 0.59 | 0.71 | | |
| | this work, paper protocol | 0.69 | 0.60 | 0.51 | 0.84 | 0.59 | 0.71 | 10.00 | 0.118 |
| | this work, corrected H counts | 0.68 | 0.60 | 0.54 | 0.81 | 0.58 | 0.72 | 10.00 | 0.148 |
| | this work, RDKit parameterisation | 0.71 | 0.57 | 0.49 | 0.86 | 0.47 | 0.81 | 10.00 | 0.058 |
| | this work, RDKit param., scaled field kernels | 0.72 | 0.56 | 0.47 | 0.88 | 0.57 | 0.73 | 10.00 | 0.016 |
| | this work, program defaults | 0.61 | 0.66 | 0.57 | 0.79 | 0.58 | 0.72 | 3.00 | 1.000 |
| **THER** | paper (Table 2) | 0.60 | 1.18 | 0.31 | 1.86 | 0.40 | 1.56 | | |
| | this work, paper protocol | 0.60 | 1.18 | 0.31 | 1.86 | 0.33 | 1.65 | 10.00 | 0.010 |
| | this work, corrected H counts | 0.62 | 1.16 | 0.32 | 1.84 | 0.41 | 1.55 | 10.00 | 0.010 |
| | this work, RDKit parameterisation | 0.57 | 1.23 | 0.37 | 1.78 | 0.32 | 1.66 | 10.00 | 0.047 |
| | this work, RDKit param., scaled field kernels | 0.60 | 1.19 | 0.39 | 1.75 | 0.33 | 1.65 | 10.00 | 0.069 |
| | this work, program defaults | 0.54 | 1.27 | 0.26 | 1.93 | 0.43 | 1.52 | 1.00 | 0.010 |
| **THR** | paper (Table 2) | 0.73 | 0.50 | 0.63 | 0.66 | 0.71 | 0.54 | | |
| | this work, paper protocol | 0.73 | 0.50 | 0.63 | 0.66 | 0.71 | 0.54 | 0.25 | 6.377 |
| | this work, corrected H counts | 0.67 | 0.55 | 0.59 | 0.71 | 0.63 | 0.61 | 0.75 | 1.078 |
| | this work, RDKit parameterisation | 0.72 | 0.50 | 0.42 | 0.84 | 0.67 | 0.58 | 0.10 | 20.000 |
| | this work, RDKit param., scaled field kernels | 0.76 | 0.46 | 0.29 | 0.93 | 0.54 | 0.68 | 0.10 | 0.038 |
| | this work, program defaults | 0.69 | 0.53 | 0.37 | 0.88 | 0.56 | 0.66 | 0.20 | 0.100 |

Mean absolute deviation from the paper over the eight sets:

| Model | q2 | RMSEcv | R2pred | RMSEpred | q2ex | RMSEcvex |
|---|---|---|---|---|---|---|
| this work, paper protocol | 0.002 | 0.003 | 0.010 | 0.017 | 0.012 | 0.013 |
| this work, paper protocol, b = 0 | 0.001 | 0.000 | 0.005 | 0.005 | 0.010 | 0.027 |
| this work, corrected H counts | 0.019 | 0.017 | 0.029 | 0.032 | 0.025 | 0.022 |
| this work, RDKit parameterisation | 0.020 | 0.022 | 0.094 | 0.082 | 0.057 | 0.055 |
| this work, RDKit param., scaled field kernels | 0.021 | 0.024 | 0.120 | 0.107 | 0.053 | 0.048 |
| this work, program defaults | 0.033 | 0.034 | 0.123 | 0.112 | 0.052 | 0.052 |
<!-- /TABLE -->

## Where the numbers deviate, and why

### 1. ACE, R²pred 0.59 vs 0.65 (paper protocol with bias)

The ACE row of Table 2 mixes two runs: q², RMSEcv, R²pred and RMSEpred come from KRR without the
bias term, q²ex and RMSEcvex from KRR with bias.  Both runs are reproduced to four decimals
(`--no-bias`: q² 0.7194, RMSEcv 1.2395, R²pred 0.6545, RMSEpred 1.2448, q²ex 0.6805, RMSEcvex
1.2827; with bias: 0.7178, 1.2431, 0.5918, 1.3530, 0.6674, 1.3088).  Not an implementation
difference.

### 2. THER, q²ex 0.33 vs 0.40

The internal statistics of THER (q² 0.60, RMSEcv 1.18, R²pred 0.31, RMSEpred 1.86) are reproduced
exactly; only the external cross-validation deviates.  The inner objective RMSEcv(λ) (with the best
α chosen inside) is **multimodal** in four of the five outer folds, e.g. fold 4 has local minima at
(λ, α) = (0.01, 10), (0.80, 5), (2.24, 3) and (4.26, 2) with RMSEcv 1.375, 1.488, 1.493 and 1.495.
R's `optimize()` is a local search; depending on where it lands in each fold, q²ex of the
external cross-validation ranges from 0.30 to 0.42 (RMSEcvex 1.54 to 1.69) over the 64 combinations
of local minima.  The combination (0.01, 3), (0.37, 0.5), (0.01, 10), (4.26, 2), (0.37, 5) gives
q²ex 0.4005 and RMSEcvex 1.5595, i.e. the published values; the global minimum in every fold
(found by scipy's bounded Brent and confirmed by a fine λ grid) gives 0.327 / 1.652.  The published
THER q²ex therefore reflects the path of the local optimiser rather than a property of the model,
and the honest value under this protocol is about 0.33.  All other 47 numbers of Table 2 are
reproduced.

### 3. The hydrogen-counting bug of the original parameterisation

`cmf-params.R` counts the hydrogens attached to each atom with

```r
el1 <- mol$atoms[[iat1]]$el
el2 <- mol$atoms[[iat1]]$el     # should be iat2
if (el1 == "H") nchydr[iat2] <- nchydr[iat2] + 1
if (el2 == "H") nchydr[iat1] <- nchydr[iat1] + 1
```

so a hydrogen is only counted when it is the *first* atom of the bond record.  In Sybyl MOL2 files
the heavy atom is written first: in the eight data sets 0 to 62 of 1000–6000 bond records per set
list the hydrogen first.  Consequently practically every atom received the increment of its
"0 attached hydrogens" branch: hydroxyl oxygens were parameterised as ethers (hydrophobic −0.34
instead of −0.61, donor A −0.035 instead of +0.445), NH/NH₂ nitrogens as tertiary amines, methyl
carbons as quaternary carbons, etc.  The hydrogen-bond donor field was thereby almost switched off
(see the kernel magnitudes below).  `--param tripos-legacy` reproduces this, `--param tripos`
counts the hydrogens correctly.  The correction changes the statistics by ±0.01–0.03 on six sets,
improves THER (q²ex 0.33 → 0.41) and deteriorates THR (q² 0.73 → 0.67, q²ex 0.71 → 0.63); THR
contains protonated amidines/guanidines and amide NH groups whose increments change most.

### 4. Field parameterisation: RDKit increments versus the in-house increments

The `rdkit` scheme uses Wildman–Crippen atomic logP contributions for the hydrophobic field and
approximate Abraham-type acceptor/donor increments with the donor weight placed on the polar
hydrogen; the steric and electrostatic fields are the same as in the paper (vdW parameters by
element instead of by Sybyl type, which only differs for sulfone/sulfoxide sulfur, R 1.80 vs 1.70;
the steric width convention b = α/R² is identical in both codes, as the 4-decimal agreement shows).
In the raw h_f = 1 convention the fields enter with very different magnitudes; the mean diagonal
elements of the field kernels, which act as implicit weights, are:

| Set, α | Scheme | electrostatic | steric | hydrophobic | H-bond acceptor | H-bond donor |
|---|---|---|---|---|---|---|
| ACE, 0.5 | tripos-legacy | 36.7 | 329 | 146 | 10.1 | 0.13 |
| ACE, 0.5 | tripos | 36.7 | 329 | 137 | 4.3 | 0.34 |
| ACE, 0.5 | rdkit | 36.7 | 329 | 120 | 41.2 | 10.7 |
| THR, 0.25 | tripos-legacy | 55.0 | 2772 | 472 | 30.8 | 1.2 |
| THR, 0.25 | rdkit | 55.0 | 2819 | 400 | 83.7 | 60.5 |
| DHFR, 10 | tripos-legacy | 1.00 | 0.39 | 0.42 | 0.10 | 0.004 |
| DHFR, 10 | rdkit | 1.00 | 0.39 | 0.82 | 0.23 | 0.023 |

With the in-house increments the published models are essentially steric + hydrophobic +
electrostatic models (the donor field weighs 10⁻³ of the steric field); the RDKit increments give
the hydrogen-bond fields 10–50 times more weight.  The RDKit scheme is better on ACE, AChE, BZR
(q²ex) and COX-2 (q²ex 0.51 vs 0.41, R²pred 0.31 vs 0.14) and worse on DHFR (R²pred 0.51 vs 0.65),
GPB (q²ex 0.47 vs 0.59) and THR (R²pred 0.42 vs 0.63); averaged over the eight sets it deviates
from the paper by 0.02 in q² and 0.05–0.09 in the external statistics.  Neither parameterisation is
uniformly better; the in-house increments are needed to reproduce the papers and are the default
recommendation for that purpose, the RDKit scheme has the advantage of being defined for arbitrary
molecules without Sybyl typing.

### 5. Kernel scaling convention and the program defaults

The program's default convention rescales every field kernel to unit mean diagonal before the
h_f-weighted sum, so that h_f = 1 means "equal weight" instead of "raw magnitude".  Compared with
the raw sum (same RDKit increments, same protocol otherwise) this helps DHFR (R²pred 0.51 → 0.64)
and GPB (q²ex 0.47 → 0.57), changes little on BZR, COX-2 and THER, and hurts ACE (R²pred 0.67 →
0.62), AChE (0.65 → 0.45) and in particular THR (R²pred 0.42 → 0.29, q²ex 0.67 → 0.54): on THR the
raw convention's dominance of the steric field (weight ratio steric : electrostatic ≈ 50 : 1 at
α = 0.25, see the table above) is evidently what the published model relies on.  Averaged over the
sets the scaled convention deviates from the paper by 0.02 in q² and 0.05–0.12 in the external
statistics; its cross-validated q² is on average slightly *higher* than with the raw sum (the
scaled kernels select much smaller λ, 0.01–0.7, so the λ range of the paper is not adequate for
them and the λ grid 10⁻⁴ … 10 of the program defaults should be used).

The *program defaults* row adds shuffled folds (seed 0) and the default α/λ grids to the scaled
convention.  Its statistics are of the same quality as the paper's in cross-validation (mean |Δq²|
0.03) but can differ markedly on the single external test sets (AChE R²pred 0.36 vs 0.64, THR 0.37
vs 0.63), which illustrates the high variance of R²pred on 22–124 molecules that the paper itself
discusses; q²ex, the paper's preferred measure, agrees within 0.05 on average.  Conclusion: the
default settings are a reasonable general-purpose choice, but any comparison with the CMF papers
must use the raw convention, the MOL2 charges and the Tripos increments (the `paper`/`tripos` modes).

## Changes to the program made for this benchmark

* `cmfqsar/increments.py` (new): transcription of the increments of `cmf-params.R`;
  `FieldSet.param_scheme` (`rdkit` | `tripos` | `tripos-legacy`), *Increments* combo box in the GUI,
  `--param` on the command line; vdW parameters by Sybyl type (`vdw_params(symbol, atom_type)`).
* `cmfqsar/dataio.py`: tolerant MOL2 fallback parser (bug fix: a molecule rejected by RDKit was
  silently dropped), `read_activity_table` / `apply_activity_table` / `load_dataset(activity_file=)`
  for separate activity tables carrying the train/test split; `--activity-file`, `--test-file`.
* `cmfqsar/model.py`: KRR without bias (`fit_bias`, `--no-bias`); interleaved folds (`seed=None`,
  `--cv-no-shuffle`); continuous Brent minimisation over λ (`grid_search(lam_search="brent")`,
  `--lam-search brent`); nested CV computes the kernels once per α and slices them for the outer
  folds (`precompute=True`, identical results, about five times faster); `--nested-on all`;
  `--json` output.
* Tests: `tests/test_benchmark_features.py` (increments, Tripos parameterisation, MOL2 fallback,
  activity tables, KRR without bias, interleaved folds, Brent search, precomputed nested CV).

## Reproducing

```bash
python benchmarks/run_sutherland.py --data-root <folder with the *_data sub-folders>
python benchmarks/run_sutherland.py --report benchmarks/results/table.md
```

The full run (six modes, eight sets) takes about 40 minutes on a laptop; results are cached per
set and mode in `benchmarks/results`.

---

# Benchmark against the continuous indicator field (CIF) models of Sitnikov, Zhokhova, Ustynyuk, Varnek & Baskin, JCAMD 29 (2015) 233–247

The 2015 paper does **not** use the Sutherland sets.  Its three case studies are thrombin
inhibitors, Am³⁺/Eu³⁺-separating triazine ligands and azo dyes; the data and the result objects of
the original R runs are in the folder of the CMF R scripts (`...\cheminf\r`, see
`benchmarks/run_sutherland.py --cif-root`).  Every published number of Tables 1–3 was traced to a
stored R result object (`*.RData`, read with R 4.5), so that the comparison below is made both
against the rounded values of the paper and against the unrounded values of the original runs.
All numbers were produced with `python benchmarks/run_sutherland.py --cif-root <folder>`; raw
results are in `benchmarks/results/cif/*.json`, the derived activity tables with the published
splits in `benchmarks/results/cif_data/`.

## Summary

* The CIF kernels of the program agree with the per-type Gram matrices stored by the original
  scripts to machine precision (max |ΔK| ≈ 10⁻¹¹ relative) for all atom types and α values of the
  three sets; the port of R's Nelder–Mead minimiser reproduces R's sequence of function
  evaluations exactly.  With the protocol of the paper (below) the program reproduces the stored
  training models of the thrombin and Am/Eu sets **exactly** (hyper-parameters, r², q², RMSEcv,
  test-set statistics to four decimals) and the external cross-validation to within the
  statistical uncertainty of the reshuffling.
* The published dye models require the Nelder–Mead search to start in a different basin than the
  scripts' default start (see deviation 3); the stored result object of the dye set corresponds to
  the default start and is reproduced exactly.
* Reproducing the paper required three additions: a simultaneous Nelder–Mead search over (α, λ) on
  kernels interpolated between the α grid points (`--alpha-search nelder-mead`, `--nm-start`), the
  statistics of the prediction averaged over reshuffled external cross-validations
  (`q2_ex_aggr` / `rmse_ex_aggr`, printed with `--repeats` > 1), the R²ext variant computed with the
  training-set variance (`r2_pred_train_var`) and an explicit CIF type list (`--cif-types`).

## Data sets

| Set | Molecules (train / test) | Activity | Files (relative to the R-script folder) | CIF atom types (≥ 5 training molecules) |
|---|---|---|---|---|
| THR | 88 (72 / 16) 3-amidinophenylalanine thrombin inhibitors (Böhm et al. 1999) | pKi | `ligands-all.mol2`, `activity-all.txt` (column THROMBIN_PKI); CPD_01–72 train, CPD_73–88 test | C.2, C.3, C.ar, C.cat, H, N.am, N.pl3, O.2, O.3, O.co2, S.o2 (N.4: 4 molecules, N.ar: 1 excluded) |
| AmEu | 46 (36 / 10) polyaza-heterocyclic ligands | log SF(Am/Eu) | `AmEu/_AmEu1-all.mol2`, `AmEu/_AmEu1-all.txt`; test molecules identified by matching the coordinates of `_AmEu1-test.mol2` (positions 3, 8, 9, 18, 22, 25, 34, 39, 42, 44 of the combined file) | C.ar, N.ar, C.3, O.3, Br (H: 1 molecule excluded) |
| DYES | 30 anionic azo dyes, no test set | affinity to cellulose −Δμ° (kJ/mol; stored with positive sign, which affects no statistic) | `gleb/aff12_aligned_babel-p5.mol2` (identical to `azo-all.mol2`), `gleb/aff.txt` | C.2, C.ar, H, N.2, N.am, N.pl3, O.2, O.3, O.co2, S.2, S.3 |

The thrombin structures are the same 88 aligned molecules (identical coordinates) as Sutherland's
THR set, with Böhm's 72/16 split instead of Sutherland's 59/29.  The Am/Eu folder contains two
typings of the same 46 ligands: `_AmEu1-*` (aromatic C.ar/N.ar types, the paper's model, Fig. 7
shows N.ar/C.ar co-fields) and `_AmEu-*` (Kekulé C.2/N.2 types, q² 0.687); the paper's dye type
list names C.3 instead of O.co2, which must be a typo (the file contains no sp³ carbon, the
sulfonate oxygens are O.co2).

## The protocol of the paper, as reconstructed from the scripts and the stored results

The CIF models were produced with `cmf-do-comp-ind-kernels-*.R`, `cmf-do-krr-train-ind.R`,
`cmf-do-krr-pred-ind.R` and `cmf-do-krr-ecvr-ind.R` on top of `cmf-krr.R`/`cmf-kernels.R`:

1. One indicator field per Sybyl atom type of the MOL2 file (`get_syb_types_list_ex`) that occurs
   in at least `min_occ` molecules (7 in the thrombin and dye scripts, 5 in the Am/Eu scripts and
   in the paper; the type lists are the same for both thresholds on these sets); w = 1, kernel
   (π/α)^{3/2} Σ exp(−α d²/4) per type (Eq. 12), h_t = 1, raw sum over types, no physico-chemical
   fields.
2. Kernels are precomputed on the α grid of the 2013 scripts (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1,
   2, 3, 5, 10) and **linearly interpolated in α** between the grid points (the end kernels are used
   beyond the grid).  α and γ (= λ) are optimised **simultaneously by R's `optim()` Nelder–Mead
   simplex** started at (α, γ) = (0.25, 5) (`alpha_grid_search = FALSE`, `gamma_grid_search =
   FALSE`), the objective being RMSEcv of a 10-fold interleaved cross-validation; the final α is
   clamped to the grid range.  Negative γ is not excluded by the scripts (15 of the 500 Am/Eu
   fold models have γ < 0).
3. KRR with the bias term (Eq. 11); the Am/Eu training model alone was built without bias
   (`set_b_0 = TRUE` in `_AmEu1-train.R`, b = 0), its external cross-validation with bias.
4. External validation: 5-fold cross-validation on **all** molecules with random reshuffling
   before every repetition (`cmf_krr_ecvr`, seed 0), the inner search of step 2 in every fold, and
   q²ext/RMSEext-cv computed from the prediction **averaged over the repetitions**
   (`Q2ecv_aggr`).  The stored Am/Eu result uses 100 repetitions as the paper states; the stored
   thrombin CIF result (`ligands-ecvr-ind.RData`) was produced with **10** repetitions and the
   thrombin physico-chemical result (`ligands-ecvr.RData`) with 20.  The mean of the per-repetition
   q² is distinctly lower (thrombin: 0.655 ± 0.073 over the 10 repetitions).
5. Test-set statistics: RMSEext and two R²ext values, "c" = 1 − RSS/Σ(y − ȳ_test)² and
   "d" = 1 − [RSS/(m − 1)]/var(y_train) (`regr_param_ex`).

Equivalent `cmf_cli.py` invocation (tag `cif` of the benchmark script):

```bash
python cmf_cli.py build ligands-all.mol2 --activity-file thrombin-activity.txt --fields , --cif --cif-min 5 \
       --raw-scale --cv-no-shuffle --alpha-search nelder-mead --alphas 0.01,0.05,0.1,0.25,0.5,0.75,1,2,3,5,10 \
       --nested --nested-on all --outer-folds 5 --repeats 100 --seed 0
```

## Validation against the stored results of the original runs

| Set | Stored object | α | γ | r² | RMSE | q² | RMSEcv | R²ext (c / d) | RMSEext | this work (same protocol) |
|---|---|---|---|---|---|---|---|---|---|---|
| THR, CIF | `ligands-model-ind.RData`, `ligands-pred-ind.RData` | 0.242646 | 93.6806 | 0.9564 | 0.2138 | 0.7721 | 0.4888 | 0.3113 / 0.4304 | 0.7534 | identical to all printed decimals |
| THR, PC | `ligands-model.RData`, `ligands-pred.RData` (α grid + Brent γ) | 0.5 | 7.4842 | 0.9288 | 0.2731 | 0.7469 | 0.5150 | 0.2799 / 0.4044 | 0.7704 | identical |
| AmEu | `_AmEu1-model.RData` (b = 0), `_AmEu1-pred.RData` | 10 | 1.10828 | 0.9062 | 0.1338 | 0.7072 | 0.2364 | 0.7447 / 0.7674 | 0.2027 | identical |
| DYES | `aff-ind-model-all.RData` | 0.01 | 7.16311 | 0.9809 | 0.4535 | 0.9245 | 0.9016 | – | – | identical |

The stored external cross-validations give q²ext / RMSEext-cv = 0.7115 / 0.5402 (THR CIF, 10
reshuffles), 0.7644 / 0.4882 (THR PC, 20 reshuffles) and 0.6676 / 0.2478 (AmEu, 100 reshuffles).

## Results

Columns as in the paper: r², RMSE of the fit, q²/RMSEcv of the 10-fold internal cross-validation of
the training model, R²ext computed with the test-set variance ("c") and with the training-set variance
("d"), RMSEext on the test set, q²ext/RMSEext-cv of the external 5-fold cross-validation with 100
reshuffles evaluated on the mean prediction; α and λ are the hyper-parameters of the training model.
*paper protocol* = the command above (`cif`); *b = 0* = the same without bias (Am/Eu training model
of the paper); *alpha grid + Brent lambda* = α on the grid and λ by Brent in [0.01, 20] in both loops
(the protocol of the 2013 paper, `cif-grid`); *physico-chemical fields* = the five fields of the 2013
paper with the in-house increments (`--param tripos-legacy --charges file`, tags `pc`/`pc-grid`);
*Nelder–Mead from (1, 1)* = the paper protocol with another starting point of the simplex.

<!-- CIF-TABLE -->
| Set | Model | r2 | RMSE | q2 | RMSEcv | R2ext | R2ext(tr) | RMSEext | q2ext | RMSEext-cv | alpha | lambda |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **THR** | paper, CIF/CMF (Table 1) | 0.956 | 0.21 | 0.772 | 0.49 | 0.311 | 0.431 | 0.75 | 0.712 | 0.54 | | |
|  | paper, PC/CMF (Table 1) | 0.929 | 0.27 | 0.747 | 0.52 | 0.280 | 0.404 | 0.77 | 0.764 | 0.49 | | |
| | this work, paper protocol (Nelder-Mead from (0.25, 5)) | 0.956 | 0.21 | 0.772 | 0.49 | 0.311 | 0.430 | 0.75 | 0.684 | 0.57 | 0.243 | 93.681 |
| | this work, alpha grid + Brent lambda (2013 protocol) | 0.971 | 0.17 | 0.753 | 0.51 | 0.444 | 0.540 | 0.68 | 0.690 | 0.56 | 0.500 | 20.000 |
| | this work, physico-chemical fields, Nelder-Mead | 0.941 | 0.25 | 0.750 | 0.51 | 0.288 | 0.411 | 0.77 | 0.672 | 0.58 | 0.750 | 3.255 |
| | this work, physico-chemical fields, alpha grid + Brent lambda | 0.929 | 0.27 | 0.747 | 0.52 | 0.280 | 0.404 | 0.77 | 0.665 | 0.58 | 0.500 | 7.484 |
| **AmEu** | paper (Table 2, text) | 0.906 | - | 0.707 | 0.24 | 0.745 | 0.767 | 0.20 | 0.668 | 0.25 | | |
| | this work, paper protocol (Nelder-Mead from (0.25, 5)) | 0.913 | 0.13 | 0.687 | 0.24 | 0.744 | 0.767 | 0.20 | 0.665 | 0.25 | 10.000 | 0.975 |
| | this work, paper protocol, b = 0 | 0.906 | 0.13 | 0.707 | 0.24 | 0.745 | 0.767 | 0.20 | 0.664 | 0.25 | 10.000 | 1.108 |
| | this work, alpha grid + Brent lambda (2013 protocol) | 0.894 | 0.14 | 0.675 | 0.25 | 0.727 | 0.751 | 0.21 | 0.666 | 0.25 | 3.000 | 6.482 |
| **DYES** | paper, all atom types (Table 3) | 1.000 | - | 0.956 | 0.69 | - | - | - | 0.909 | 0.99 | | |
|  | paper, hydrogens only (Table 3) | 0.999 | - | 0.940 | 0.80 | - | - | - | 0.822 | 1.39 | | |
| | this work, paper protocol (Nelder-Mead from (0.25, 5)) | 0.981 | 0.45 | 0.924 | 0.90 | - | - | - | 0.906 | 1.00 | 0.010 | 7.163 |
| | this work, paper protocol, Nelder-Mead from (1, 1) | 1.000 | 0.07 | 0.959 | 0.67 | - | - | - | 0.923 | 0.91 | 1.371 | 0.017 |
| | this work, hydrogens only, paper protocol | 0.999 | 0.12 | 0.940 | 0.80 | - | - | - | 0.884 | 1.12 | 10.000 | 0.006 |
| | this work, hydrogens only, Nelder-Mead from (1, 1) | 0.999 | 0.12 | 0.940 | 0.80 | - | - | - | 0.891 | 1.08 | 10.000 | 0.006 |
| | this work, alpha grid + Brent lambda (2013 protocol) | 1.000 | 0.05 | 0.956 | 0.69 | - | - | - | 0.913 | 0.97 | 1.000 | 0.010 |
<!-- /CIF-TABLE -->

Per-repetition statistics of the external cross-validations (mean ± sd of q² over the 100
reshuffles; the stored R results give 0.655 ± 0.073 for thrombin CIF over 10 reshuffles and
0.613 ± 0.148 for Am/Eu over 100):

| Run | q²ext per repetition | repetitions with q² < 0.3 | fold models with α at the upper grid end | fold models with λ < 0 |
|---|---|---|---|---|
| THR, CIF, paper protocol | 0.620 ± 0.053 | 0 / 100 | 0 / 500 | 0 |
| THR, PC, Nelder–Mead | 0.603 ± 0.104 | 3 | 0 | 0 |
| AmEu, paper protocol | 0.619 ± 0.077 | 0 | 248 | 8 |
| DYES, all types, paper protocol | 0.773 ± 0.397 | 3 | 37 | 38 |
| DYES, all types, Nelder–Mead from (1, 1) | 0.573 ± 2.257 | 3 | 92 | 100 |
| DYES, hydrogens only, paper protocol | 0.774 ± 0.221 | 4 | 23 | 48 |
| DYES, hydrogens only, Nelder–Mead from (1, 1) | 0.730 ± 0.442 | 6 | 47 | 67 |

## Where the numbers deviate, and why

### 1. Training models: all reproduced

The training models of the thrombin (CIF and physico-chemical), Am/Eu and dye sets are
reproduced exactly from the stored result objects, including the non-grid α of the thrombin model
(0.2426) found by the simplex search and the clamped α = 10 of the Am/Eu model.  The paper's
r²ext "d" for thrombin (0.431) and Am/Eu (0.767) is the training-variance formula of
`regr_param_ex`, now available as `r2_pred_train_var`; the program's older "train mean" variant
(0.335 for thrombin) is a different quantity.  The Am/Eu r² of 0.906 and q² of 0.707 require the
model without bias (`--no-bias`, b = 0 as in `_AmEu1-train.R`); with bias the same protocol gives
r² 0.913, q² 0.687 and identical test-set statistics.

### 2. External cross-validation: reshuffling noise and the number of repetitions

The external cross-validations cannot be reproduced number for number because the molecules are
reshuffled with R's random generator; the comparison is statistical.  Am/Eu agrees to the printed
precision (0.665 / 0.249 vs 0.668 / 0.25, 100 reshuffles in both cases; per-repetition mean 0.619
vs 0.613; the fold hyper-parameters show the same pattern, half of the fold models at the upper
end of the α grid and a few with negative λ).  The published thrombin CIF value (0.712 / 0.54)
comes from **10** reshuffles whose per-repetition q² values are 0.50 … 0.75 (mean 0.655 ± 0.073);
our 100 reshuffles give 0.620 ± 0.053 per repetition and 0.684 / 0.566 for the mean prediction.
The difference of the per-repetition means (0.035) is 1.5 standard errors of the 10-reshuffle
estimate, and the distributions of the selected hyper-parameters agree (α median 0.37 vs 0.40, λ
median 54 vs 49, no fold at the grid ends in either run).  R itself, re-run on the stored
permutations, kernels and fold hyper-parameters, reproduces the stored CIF fold models of both sets
exactly, so the protocol is the same and the published thrombin value is at the favourable end of
the reshuffling distribution.  Note that q²ext of the mean prediction is systematically higher than
the per-repetition q² (0.68 vs 0.62 here, 0.71 vs 0.66 in the stored run): averaging the
predictions of differently trained models is an ensemble, and the number of repetitions
influences the result.

### 3. The physico-chemical thrombin model: q²ext 0.764 cannot be traced

The PC/CMF training model of Table 1 (α 0.5 on the grid, γ 7.48 by Brent, q² 0.747) is reproduced
exactly from `ligands-kernels-train.RData`, but its external cross-validation (0.764 / 0.49,
`ligands-ecvr.RData`, 20 reshuffles, Nelder–Mead in every fold) is not: our runs give 0.67 / 0.58
with either search.  Re-running R's own `cv_krr` on the stored permutation, training indices and
hyper-parameters of the first stored fold model with the kernel file present in the folder
(`ligands-kernels-all.RData`, which agrees with the program's kernels to 10⁻¹⁴) gives q² 0.642
instead of the stored 0.762, and the stored cross-validated predictions differ from the
recomputed ones by up to 2 log units.  The stored external cross-validation was therefore
produced with a kernel set that no longer exists in the folder (file dates: kernels June 2013,
result January 2014; the stored fold models cluster at α ≈ 0.25–0.39, γ ≈ 4.5–5.5 with inner q²
0.74–0.81, better than the present kernels allow).  Its 20 repetitions also contain one with
q² = −2.46, a Nelder–Mead run gone astray.  The CIF value of 0.712 and the PC value of 0.764 of
Table 1 are thus not comparable with each other; with the present data the physico-chemical
fields and the CIFs give the same external q² (0.67 vs 0.68).

### 4. The dye models: two basins of the simplex search

The scripts' default start (α, γ) = (0.25, 5) leads the Nelder–Mead search of the dye set into
the basin α → 0 (clamped to 0.01, γ 7.16, q² 0.924, RMSEcv 0.90, r² 0.981), which is the stored
`aff-ind-model-all.RData`, whereas the paper reports q² 0.956, RMSEcv 0.69, r² 1.000.  From
almost any other start (e.g. (1, 1)) the search converges to α 1.37, γ 0.017 with q² 0.959,
RMSEcv 0.67, r² 0.9996, and the α-grid search with Brent γ gives α 1, γ 0.01, q² 0.956, RMSEcv
0.69 (tag `cif-grid`), the published values.  The published dye model therefore belongs to the
second basin and was obtained with a different start or an α grid search; all three variants give
the same external cross-validation (q²ext 0.91–0.92, RMSEext-cv 0.91–1.00 vs 0.909 / 0.99).  The
hydrogen-only model is reproduced from the default start (α ≥ 10 clamped, γ 0.0058, q² 0.940,
RMSEcv 0.80, r² 0.999); its external cross-validation (ours 0.88–0.89 / 1.1, paper 0.822 / 1.39)
is dominated by the instability of fold models with λ → 0 or λ < 0 (48–67 of 500 fold models,
per-repetition q² between −2.9 and 0.95), so that the aggregated value depends on how many such
repetitions a run happens to contain.  Two further remarks: the internal q² values of the dye
set are inflated by the interleaved folds in file order, because `aff.txt` is sorted by activity
and the folds are therefore stratified (shuffled 10-fold folds give q² 0.87–0.90 for the same
models); and the whole-set fit with r² = 1.000 reflects the tiny γ selected by cross-validation
on 30 molecules.

### 5. Simplex search versus grid search

For the thrombin CIF model the simplex search (q² 0.772) beats the α grid + Brent λ protocol
(0.753, λ at the upper bound 20), while on the test set the grid model is better (R²ext 0.44 vs
0.31, RMSEext 0.68 vs 0.75); the external cross-validations agree (0.684 vs 0.690).  On Am/Eu the
grid protocol is slightly worse in all statistics (q² 0.675 vs 0.707); on the dyes it finds the
good basin directly.  Because the kernels of the simplex protocol are linear interpolations
between grid kernels, a model with α between grid points is not the model at that α; the program
stores the grid (`alpha_grid`) with the model and interpolates in the same way for new molecules,
and the co-field plots of such a model are drawn at the nominal α (an approximation).  With two
parameters and a kinked (piecewise linear in α) objective the simplex search is start-dependent,
as deviation 4 shows.

## Changes to the program made for this benchmark

* `cmfqsar/neldermead.py` (new): `nmmin`, a line-by-line port of R's `optim(method =
  "Nelder-Mead")` (validated against R's evaluation trace), and `interp_terms` (linear
  interpolation weights on the α grid).
* `cmfqsar/model.py`: `CMFModel.alpha_grid` (field kernels for α between grid points are linear
  interpolations of the grid kernels, used consistently for training, prediction, atom
  contributions and saved models); `grid_search(alpha_search="nelder-mead", nm_start=...)`
  (simultaneous (α, λ) search on interpolated kernels, the result taken as it is);
  `nested_cv(inner_seed=...)` and the aggregated statistics `q2_ex_aggr`/`rmse_ex_aggr` of the
  prediction averaged over reshuffles plus the selected hyper-parameters of every outer fold
  (`per_repeat[i]["hyper"]`); `ModelStats.r2_pred_train_var` (R²ext with the training-set variance,
  `regr_param_ex`) and `q2_ex_aggr`/`rmse_ex_aggr`.
* `cmf_cli.py`: `--alpha-search nelder-mead`, `--nm-start`, `--cif-types`; with `--repeats` > 1 the
  outer folds are reshuffled before every repetition and the inner folds are random even with
  `--cv-no-shuffle` (bug fix: previously all repetitions used the same interleaved folds, so that
  `--repeats` had no effect with `--cv-no-shuffle`); JSON output gains `q2_ex_aggr`,
  `rmse_ex_aggr`, `r2_pred_train_var`, `cif_types` and `nested_per_repeat`.
* `benchmarks/run_sutherland.py`: `--cif-root`, `--tags`, `--cif-report`, `CIF_RUNS`,
  `prepare_cif_data` (derived activity tables with the published splits in
  `benchmarks/results/cif_data`); `--sets` no longer defaults to the Sutherland list.
* Tests: `tests/test_cif_protocol.py` (Nelder–Mead port against R's results, interpolation
  weights, interpolated kernels, simplex search, reshuffled external CV with aggregated
  statistics, training-variance R²ext).

## Reproducing

```bash
python benchmarks/run_sutherland.py --cif-root <folder of the R scripts>      # all 12 runs, about 40 min
python benchmarks/run_sutherland.py --cif-root <folder> --sets DYES --tags cif,cif-h
python benchmarks/run_sutherland.py --cif-report benchmarks/results/cif/table.md
```


# Benchmark of the nonlinear kernel layer and of alignment-based multiple kernel learning on the public example collections

The two options described in
[*Nonlinear kernel layers and multiple kernel learning*](../README.md#nonlinear-kernel-layers-and-multiple-kernel-learning)
are benchmarked there on the eight Sutherland sets, with the published alignments and the published
train/test splits. This section runs the same two options over the four example collections that
the README does not cover — `examples/openff`, `examples/schindler`, `examples/ross` and
`examples/mittal` — 86 congeneric series and 4549 molecules, which is an order of magnitude more
data sets than the original benchmark and, unlike it, a mixture of good and bad alignments. The
Sutherland numbers are not repeated here; only the conclusions drawn from them are tested.

The short answer is that both conclusions survive, that both effects are small compared with what
the *alignment* of a data set decides, and that one of the two alignment variants should not be
used on these collections.

## Summary

* The **Gaussian kernel layer** is confirmed: it raises q²cv by 0.027 on average (paired SD 0.076
  over 86 sets, t = 3.3) and q²ex by 0.026 (paired SD 0.097, t = 2.5). The direction is the same as
  on the Sutherland sets and the size is about a third of it. It wins on 53 of 86 sets in q²ex,
  which is worth stressing: the layer is a reliable *average* improvement and a coin flip on any
  single set.
* **Alignment-based multiple kernel learning** behaves as reported, too, and the distinction between
  its two modes matters more here. Per-field alignment (`align`) gains 0.014 in q²cv (SD 0.036,
  t = 3.6) and 0.008 in q²ex (t = 1.4) over the same field set with h_f = 1 — the "about +0.01 with
  indicator fields" of the README, reproduced on ten times as many sets. The joint solution
  (`alignf`) does not: it is level in q²cv (+0.010, t = 1.7) and **loses** 0.015 in q²ex, winning
  only 40 of 86 sets. It is also the least stable configuration of the five (mean standard
  deviation over the five reshuffles of the external cross-validation 0.135 against 0.102 for
  `align`), and it fails outright on individual sets — `schindler/cdk8` falls from q²ex 0.470 to
  −0.184. On these collections, use `align`.
* Adding the **continuous indicator fields** at all is worth nothing on average (q²ex +0.005,
  t = 0.3) but that average hides a clean split by size: over the 33 sets with at least 40
  molecules they raise q²ex from 0.220 to 0.248, and over the 53 smaller sets they lower it from
  0.222 to 0.212. Twelve to eighteen extra field types need molecules to pay for themselves. Since
  the alignment runs use the indicator fields, a `cif` column with h_f = 1 is carried through every
  table; without it a comparison of `align` against the five physico-chemical fields would mix
  "added the indicator fields" with "weighted them".
* Over the **five physico-chemical fields alone** mixing is worth nothing, as the README says, but
  it is not harmful either: at the default shrinkage `align` gains 0.004 of q²ex (t = 0.8) and
  `alignf` loses 0.004 (t = −0.6) against h_f = 1. The *unshrunk* joint solution, which is what the
  README's example was, is harmful and by a wide margin — q²ex 0.221 → 0.152, t = −4.7, 42 of the
  86 sets losing more than 0.05 — because it zeroes three or more of the five fields on 43 sets.
  Full numbers
  [below](#does-alignment-based-multiple-kernel-learning-hurt-over-the-five-physico-chemical-fields-alone).
* **None of this is the main effect.** The quality of the alignment is, by a factor of three. The
  five Mittal sets that consist of exactly the same molecules with exactly the same activities as
  their Sutherland counterparts, and differ only in the coordinates, go from a mean q²ex of 0.534
  with Sutherland's published alignment to 0.137 with automatic conformer generation and MCS
  superposition. Against that, 0.026 for a kernel layer is a detail. Where a result below is about
  the alignment rather than about the kernel, it is said so.
* The honest resolution of a single set is about **±0.1 in q²ex**: that is the mean standard
  deviation of q²ex over the five reshuffles of the external cross-validation, and independently it
  is the scale on which two different preparations of one and the same congeneric series disagree
  (`schindler/cmet` and `ross/merck/cmet` are the same 24 ligands in different poses and give q²ex
  0.678 and 0.571). Per-set differences smaller than that carry no information; only the means and
  the win counts over many sets do.

## Data sets

| Collection | Series used | Molecules | Poses | Activity |
|---|---|---|---|---|
| `examples/openff` | 16 of 21 | 465 | crystal-structure-based ligand poses in the binding site (OpenFF protein-ligand benchmark) | pAffinity from the deposited affinities |
| `examples/schindler` | 10 of 10 | 330 | prepared poses in the site (Merck KGaA / Schindler et al. 2020 FEP benchmark) | pAffinity from ΔG |
| `examples/ross` | 24 of 67 | 866 | prepared poses in the site (Schrödinger public binding free energy benchmark, Ross et al. 2023) | pAffinity from ΔG |
| `examples/mittal` | 36 of 38 | 2888 | **automatic**: RDKit ETKDGv3 conformer + MMFF94, superimposed on a reference molecule by its maximum common substructure | activity in log units as given by Mittal et al. 2009 |

Every SDF with at least 20 molecules carrying an activity is used; the activity property is the one
`cmfqsar.dataio.load_dataset` guesses. The series excluded by that threshold are the 43 small
`examples/ross` cases (5 to 18 ligands — many are two- to four-ligand scaffold-hopping or
charge-change pairs), five `examples/openff` targets (5 to 13 ligands) and `APZ` and `RYR` of
`examples/mittal` (16, 17). A q² over six molecules is noise, not a statistic.

Three properties of the collections affect how the tables must be read, all of them verified rather
than assumed:

* `schindler/eg5.sdf` and `schindler/eg5_alternativeloop.sdf` hold **identical ligand coordinates**
  (maximum coordinate difference 0.0 over all 28 ligands); the two cases differ in a protein loop,
  which a ligand-field model never sees. Their rows are therefore bit-identical and the
  `schindler` means count that series twice; read them as nine series, not ten.
* `examples/ross/merck` is the same Merck benchmark as `examples/schindler`, re-prepared by
  Schrödinger, but it is not a copy of it. Only `shp2` (26 ligands) and `cmet` (24) have the same
  composition; the others hold more ligands (extra protomers, "fullmap"/"automap" variants:
  `tnks2` 27/33, `syk` 44/51, `cdk8` 33/41, `hif2a` 42/51, `pfkfb3` 40/87, `eg5` 28/34). `shp2` is
  identical in molecules *and* poses and its two rows agree to the last digit, which is a free
  end-to-end determinism check of the whole pipeline. `cmet` is the same molecules in *different*
  poses and its two rows differ by 0.08 in q²cv and 0.11 in q²ex — the pose-preparation noise
  quoted above.
* `examples/mittal` overlaps `examples/sutherland` in nine targets, and in five of them
  (ACE, AChE/ACHE, GPB, THER/THERM, THR) the molecules and the activities are *identical*. That
  coincidence is what makes the alignment control below possible.

## Protocol

Five configurations per data set. All of them use cosine-normalised kernels, field kernels scaled to
unit mean diagonal, kernel ridge regression with a bias term and the default grids of
`cmf_cli.py build` (α over 0.2 … 3.0 in eight steps, λ over 10⁻⁴ … 10, γ over 0.25 … 4):

| Tag | Fields | Kernel | Mixing coefficients |
|---|---|---|---|
| `pc` | the five physico-chemical fields | linear | h_f = 1 |
| `gauss` | the five physico-chemical fields | Gaussian layer, γ searched with α and λ | h_f = 1 |
| `cif` | physico-chemical + continuous indicator fields | linear | h_f = 1 |
| `align` | physico-chemical + continuous indicator fields | linear | centred kernel alignment per field |
| `alignf` | physico-chemical + continuous indicator fields | linear | joint alignment, shrunk (ρ = 1) |

Three further configurations were added afterwards, to settle the one README claim the five above
leave open — that alignment-based mixing is harmful over the five physico-chemical fields on their
own. They are reported separately, in
[their own section](#does-alignment-based-multiple-kernel-learning-hurt-over-the-five-physico-chemical-fields-alone),
and are not carried through the per-set tables of the five:

| Tag | Fields | Kernel | Mixing coefficients |
|---|---|---|---|
| `pc+align` | the five physico-chemical fields | linear | centred kernel alignment per field |
| `pc+alignf` | the five physico-chemical fields | linear | joint alignment, shrunk (ρ = 1) |
| `pc+alignf0` | the five physico-chemical fields | linear | joint alignment, **unshrunk** (ρ = 0) |

The indicator fields are one per Sybyl atom type present in at least five molecules
(`cif_types_in(..., min_molecules=5)`), which gives 10 to 23 fields in total. Partial charges are
Gasteiger–Marsili throughout, so that the collections are comparable with each other and with the
control run.

Two statistics are reported per set:

* **q²cv / RMSEcv** — 10-fold cross-validation over all molecules, with α, λ, γ and the alignment
  weights chosen on that same cross-validation. It is therefore optimistically biased, and it is
  given because it is the number a user of the program sees. The alignment weights are at least
  re-derived inside every fold by `mkl_cv_predict`, so they never see the left-out activities.
* **q²ex / RMSEex** — external 5-fold cross-validation with the whole hyper-parameter search
  repeated inside every outer fold (`cmfqsar.model.nested_cv`), **averaged over five reshuffles** of
  the outer folds, with the standard deviation over the reshuffles kept in the raw results. A
  single 5-fold split is far too noisy at the 20 to 50 molecules that most of these series have;
  five repeats cost little because the per-field kernels are computed once per α and sliced.

None of these collections has a published train/test split, which is why external
cross-validation is used throughout instead of the single test set of the Sutherland benchmark.
The per-set tables below give q²ex and RMSEex together with the selected α; the q²cv numbers per
set are in the raw results and in the report that the script writes.

One departure from a strictly honest protocol is worth naming: the list of indicator atom types is
derived once from all molecules of a set, not inside each outer fold. It uses structures only and
no activities, so it cannot inflate q²ex through the response, but it is not a fold-local decision
either.

## Results per data set, external cross-validation

q²ex (mean over five reshuffles) and RMSEex for the five configurations, best of the five in bold.
"F(cif)" is the total number of fields of the three indicator-field configurations, "α pc" and
"α cif" the attenuation factors selected by the full-set search.

### `examples/openff`

| Set | n | F(cif) | α pc | α cif | pc | gauss | cif | align | alignf | RMSE pc | gauss | cif | align | alignf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bace | 36 | 17 | 0.5 | 3 | 0.223 | **0.286** | 0.201 | 0.220 | 0.252 | 0.51 | **0.49** | 0.52 | 0.51 | 0.50 |
| bace_hunt | 32 | 18 | 1 | 3 | 0.001 | **0.257** | 0.032 | -0.036 | -0.136 | 0.86 | **0.74** | 0.84 | 0.87 | 0.91 |
| cdk8 | 31 | 13 | 0.7 | 3 | 0.416 | 0.408 | **0.445** | 0.430 | 0.304 | 0.75 | 0.75 | **0.73** | 0.74 | 0.81 |
| eg5 | 27 | 15 | 1 | 3 | 0.189 | **0.323** | 0.226 | 0.182 | 0.156 | 0.58 | **0.53** | 0.56 | 0.58 | 0.59 |
| hif2a | 37 | 16 | 0.2 | 3 | -0.096 | **0.000** | -0.201 | -0.285 | -0.251 | 0.76 | **0.73** | 0.80 | 0.83 | 0.82 |
| jnk1 | 21 | 16 | 3 | 3 | 0.603 | **0.617** | 0.512 | 0.524 | 0.463 | 0.39 | **0.38** | 0.43 | 0.43 | 0.45 |
| mcl1 | 25 | 13 | 0.2 | 3 | 0.433 | 0.369 | 0.686 | **0.711** | 0.642 | 0.63 | 0.66 | 0.46 | **0.45** | 0.50 |
| p38 | 29 | 13 | 1 | 3 | 0.047 | **0.071** | -0.207 | -0.070 | -0.058 | 0.68 | **0.67** | 0.76 | 0.72 | 0.72 |
| pde10 | 35 | 15 | 3 | 3 | **0.171** | 0.157 | 0.145 | 0.077 | 0.136 | **0.79** | 0.79 | 0.80 | 0.83 | 0.80 |
| pde2 | 21 | 12 | 0.2 | 0.5 | 0.260 | 0.069 | 0.287 | **0.289** | 0.271 | 0.54 | 0.61 | 0.53 | **0.53** | 0.54 |
| pfkfb3 | 32 | 13 | 1 | 1 | 0.470 | 0.478 | 0.534 | **0.577** | 0.523 | 0.56 | 0.55 | 0.52 | **0.50** | 0.53 |
| ptp1b | 22 | 17 | 0.2 | 0.5 | **0.416** | 0.396 | 0.139 | 0.274 | 0.180 | **0.72** | 0.73 | 0.88 | 0.80 | 0.85 |
| shp2 | 24 | 12 | 0.2 | 0.2 | **-0.122** | -0.261 | -0.273 | -0.256 | -0.239 | **0.84** | 0.90 | 0.90 | 0.90 | 0.89 |
| syk | 44 | 15 | 1 | 3 | 0.215 | **0.321** | 0.074 | 0.113 | 0.139 | 0.52 | **0.49** | 0.57 | 0.56 | 0.55 |
| thrombin | 22 | 14 | 3 | 2 | -0.180 | -0.145 | -0.105 | **0.002** | -0.057 | 1.05 | 1.04 | 1.02 | **0.97** | 1.00 |
| tnks2 | 27 | 13 | 1 | 3 | 0.074 | 0.091 | 0.316 | 0.397 | **0.494** | 0.71 | 0.71 | 0.61 | 0.57 | **0.53** |
| **mean** | | | | | 0.195 | **0.215** | 0.176 | 0.197 | 0.176 | 0.68 | **0.67** | 0.68 | 0.67 | 0.69 |

### `examples/schindler`

| Set | n | F(cif) | α pc | α cif | pc | gauss | cif | align | alignf | RMSE pc | gauss | cif | align | alignf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cdk8 | 33 | 13 | 2 | 3 | 0.369 | 0.368 | **0.470** | 0.433 | -0.184 | 0.78 | 0.78 | **0.71** | 0.74 | 0.98 |
| cmet | 24 | 15 | 3 | 3 | 0.678 | **0.729** | 0.688 | 0.637 | 0.656 | 0.70 | **0.65** | 0.70 | 0.75 | 0.73 |
| eg5 | 28 | 15 | 0.7 | 0.7 | 0.168 | **0.351** | 0.108 | 0.058 | 0.064 | 0.58 | **0.52** | 0.61 | 0.62 | 0.62 |
| eg5_alternativeloop | 28 | 15 | 0.7 | 0.7 | 0.168 | **0.351** | 0.108 | 0.058 | 0.064 | 0.58 | **0.52** | 0.61 | 0.62 | 0.62 |
| hif2a | 42 | 16 | 0.2 | 1 | **0.211** | 0.202 | -0.035 | -0.063 | -0.050 | **0.73** | 0.73 | 0.84 | 0.85 | 0.84 |
| hif2a-jmc2023 | 38 | 16 | 2 | 0.7 | 0.452 | **0.550** | 0.366 | 0.487 | 0.445 | 0.37 | **0.33** | 0.39 | 0.35 | 0.37 |
| pfkfb3 | 40 | 14 | 3 | 3 | 0.028 | **0.169** | -0.004 | 0.088 | 0.073 | 0.77 | **0.72** | 0.79 | 0.75 | 0.76 |
| shp2 | 26 | 12 | 3 | 0.3 | -0.034 | -0.105 | **-0.034** | -0.124 | -0.054 | 0.86 | 0.89 | **0.86** | 0.90 | 0.87 |
| syk | 44 | 15 | 1.5 | 1 | 0.205 | **0.290** | 0.178 | 0.168 | 0.120 | 0.53 | **0.50** | 0.54 | 0.54 | 0.55 |
| tnks2 | 27 | 13 | 2 | 3 | 0.146 | 0.167 | 0.082 | 0.159 | **0.239** | 0.68 | 0.68 | 0.71 | 0.68 | **0.65** |
| **mean** | | | | | 0.239 | **0.307** | 0.193 | 0.190 | 0.137 | 0.66 | **0.63** | 0.67 | 0.68 | 0.70 |

### `examples/ross`

| Set | n | F(cif) | α pc | α cif | pc | gauss | cif | align | alignf | RMSE pc | gauss | cif | align | alignf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpcrs/a2a_hip278 | 29 | 12 | 2 | 2 | 0.586 | 0.463 | 0.566 | **0.615** | 0.596 | 0.39 | 0.44 | 0.39 | **0.38** | 0.39 |
| gpcrs/ox2_hip_custcore | 51 | 16 | 1.5 | 2 | -0.075 | **0.213** | -0.206 | -0.199 | -0.221 | 0.91 | **0.78** | 0.96 | 0.96 | 0.97 |
| gpcrs/p2y1_meta_sub | 20 | 14 | 1 | 0.5 | -0.407 | **-0.388** | -0.486 | -0.623 | -0.712 | 0.72 | **0.71** | 0.74 | 0.77 | 0.79 |
| jacs_set/bace | 36 | 17 | 0.5 | 2 | **0.189** | 0.111 | 0.169 | 0.176 | 0.188 | 0.51 | 0.53 | 0.51 | 0.51 | **0.50** |
| jacs_set/jnk1_manual_flips | 38 | 19 | 3 | 3 | **0.840** | 0.807 | 0.712 | 0.729 | 0.743 | **0.25** | 0.28 | 0.34 | 0.33 | 0.32 |
| jacs_set/mcl1_extra_flips | 69 | 15 | 3 | 3 | 0.490 | 0.635 | 0.704 | **0.715** | 0.697 | 0.53 | 0.45 | 0.41 | **0.40** | 0.41 |
| jacs_set/p38 | 34 | 15 | 0.5 | 3 | **0.092** | 0.052 | 0.023 | 0.045 | 0.061 | **0.69** | 0.71 | 0.72 | 0.71 | 0.70 |
| jacs_set/ptp1b | 23 | 17 | 2 | 0.2 | -0.111 | 0.076 | 0.083 | **0.089** | 0.062 | 0.98 | **0.88** | 0.90 | 0.90 | 0.91 |
| janssen_bace/bace_ciordia_retro | 33 | 18 | 0.5 | 3 | -0.086 | **0.328** | 0.208 | 0.147 | 0.088 | 0.93 | **0.73** | 0.79 | 0.82 | 0.85 |
| janssen_bace/bace_p3_arg368_in | 21 | 18 | 1 | 2 | 0.154 | -0.097 | 0.306 | 0.334 | **0.376** | 0.98 | 1.13 | 0.89 | 0.88 | **0.85** |
| mcs_docking/hne | 20 | 16 | 0.7 | 3 | -0.059 | **0.024** | -0.011 | -0.007 | -0.009 | 1.13 | **1.09** | 1.11 | 1.10 | 1.10 |
| mcs_docking/renin_customcore | 29 | 15 | 0.2 | 3 | **0.141** | 0.097 | -0.058 | -0.024 | 0.131 | **0.95** | 0.98 | 1.06 | 1.04 | 0.96 |
| merck/cdk8_5cei_new_helix_loop_extra | 41 | 13 | 1 | 3 | 0.349 | 0.358 | **0.527** | 0.434 | 0.065 | 0.77 | 0.76 | **0.66** | 0.71 | 0.86 |
| merck/cmet | 24 | 15 | 3 | 0.2 | 0.571 | **0.581** | 0.448 | 0.500 | 0.574 | 0.81 | **0.81** | 0.92 | 0.87 | 0.81 |
| merck/eg5_extraprotomers | 34 | 19 | 3 | 0.7 | 0.209 | **0.361** | 0.171 | 0.153 | 0.017 | 0.56 | **0.50** | 0.57 | 0.58 | 0.63 |
| merck/hif2a_automap | 51 | 17 | 0.2 | 1 | **0.632** | 0.559 | 0.506 | 0.541 | 0.465 | **0.48** | 0.52 | 0.56 | 0.54 | 0.58 |
| merck/pfkfb3_automap | 87 | 20 | 3 | 3 | 0.426 | 0.426 | 0.487 | **0.509** | 0.504 | 0.57 | 0.57 | 0.54 | **0.52** | 0.53 |
| merck/shp2 | 26 | 12 | 3 | 0.3 | -0.034 | -0.105 | **-0.034** | -0.124 | -0.054 | 0.86 | 0.89 | **0.86** | 0.90 | 0.87 |
| merck/syk_4puz_fullmap | 51 | 15 | 3 | 1.5 | **0.209** | 0.188 | 0.162 | 0.139 | 0.160 | **0.56** | 0.57 | 0.58 | 0.59 | 0.58 |
| merck/tnks2_fullmap | 33 | 14 | 3 | 3 | 0.342 | 0.368 | 0.425 | **0.456** | 0.408 | 0.60 | 0.59 | 0.56 | **0.54** | 0.57 |
| misc/galectin3_extra | 26 | 12 | 1.5 | 0.7 | 0.572 | **0.702** | 0.668 | 0.672 | 0.687 | 0.40 | **0.33** | 0.35 | 0.34 | 0.34 |
| misc/hfaah | 24 | 13 | 0.2 | 0.3 | 0.598 | 0.518 | 0.495 | **0.601** | 0.491 | **0.62** | 0.68 | 0.70 | 0.62 | 0.70 |
| opls_stress/fxa_yoshikawa_set | 27 | 15 | 3 | 3 | 0.202 | **0.370** | -0.021 | 0.130 | 0.047 | 0.96 | **0.85** | 1.08 | 1.00 | 1.05 |
| waterset/throm_nozob_hip75 | 39 | 15 | 3 | 0.7 | 0.413 | **0.424** | 0.318 | 0.328 | 0.302 | 0.61 | **0.60** | 0.65 | 0.65 | 0.66 |
| **mean** | | | | | 0.260 | **0.295** | 0.257 | 0.264 | 0.236 | 0.70 | **0.68** | 0.70 | 0.69 | 0.70 |

### `examples/mittal`

| Set | n | F(cif) | α pc | α cif | pc | gauss | cif | align | alignf | RMSE pc | gauss | cif | align | alignf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ACE | 114 | 18 | 0.3 | 0.2 | 0.414 | 0.402 | **0.498** | 0.484 | 0.386 | 1.74 | 1.75 | **1.61** | 1.63 | 1.78 |
| ACHE | 111 | 16 | 0.2 | 0.2 | **0.189** | 0.165 | 0.108 | 0.130 | 0.123 | **1.11** | 1.13 | 1.17 | 1.15 | 1.16 |
| ACHE_TOX | 30 | 15 | 3 | 0.2 | **-0.141** | -0.184 | -0.235 | -0.279 | -0.253 | **0.93** | 0.94 | 0.96 | 0.98 | 0.97 |
| AI | 78 | 13 | 2 | 1.5 | 0.659 | 0.664 | 0.665 | **0.669** | 0.662 | 0.62 | 0.62 | 0.62 | **0.61** | 0.62 |
| AMPH1 | 130 | 18 | 1.5 | 0.2 | -0.024 | **0.035** | -0.010 | -0.008 | -0.036 | 0.88 | **0.86** | 0.88 | 0.88 | 0.89 |
| ARB | 28 | 13 | 0.5 | 0.2 | 0.077 | 0.100 | **0.282** | 0.262 | 0.279 | 1.00 | 0.98 | 0.88 | 0.89 | **0.88** |
| ATA | 94 | 15 | 3 | 3 | 0.123 | **0.139** | 0.122 | 0.128 | 0.111 | 1.16 | **1.15** | 1.16 | 1.16 | 1.17 |
| BZR | 163 | 20 | 2 | 0.2 | 0.325 | 0.263 | **0.423** | 0.408 | 0.407 | 0.90 | 0.94 | **0.83** | 0.84 | 0.85 |
| CBRA | 31 | 12 | 0.2 | 0.5 | 0.123 | 0.171 | 0.287 | 0.287 | **0.311** | 0.96 | 0.93 | 0.86 | 0.86 | **0.85** |
| COMT | 92 | 19 | 2 | 0.2 | 0.366 | 0.377 | 0.572 | **0.608** | 0.584 | 0.65 | 0.65 | 0.53 | **0.51** | 0.53 |
| COX2 | 322 | 23 | 1 | 0.2 | 0.177 | 0.180 | 0.298 | 0.304 | **0.308** | 1.29 | 1.29 | 1.19 | 1.19 | **1.18** |
| D2A | 26 | 10 | 1 | 0.2 | 0.372 | 0.388 | 0.321 | **0.391** | 0.388 | 1.05 | 1.04 | 1.09 | **1.03** | 1.04 |
| DAT | 42 | 14 | 0.2 | 0.2 | 0.003 | **0.019** | -0.045 | -0.006 | 0.005 | 0.75 | **0.75** | 0.77 | 0.76 | 0.75 |
| DHFR | 397 | 21 | 0.3 | 0.3 | 0.469 | **0.502** | 0.434 | 0.471 | 0.481 | 1.08 | **1.05** | 1.12 | 1.08 | 1.07 |
| DIAZEPAM | 42 | 15 | 1.5 | 0.2 | 0.505 | 0.511 | **0.554** | 0.499 | 0.455 | 0.75 | 0.75 | **0.71** | 0.76 | 0.79 |
| DR | 38 | 17 | 1.5 | 0.5 | 0.362 | 0.487 | **0.591** | 0.538 | 0.485 | 0.96 | 0.86 | **0.77** | 0.81 | 0.86 |
| ECR | 50 | 13 | 0.2 | 0.2 | -0.095 | -0.066 | -0.081 | -0.039 | **-0.023** | 0.77 | 0.77 | 0.77 | 0.76 | **0.75** |
| EDC | 122 | 14 | 0.3 | 1 | 0.178 | 0.189 | **0.236** | 0.214 | 0.193 | 1.59 | 1.58 | **1.53** | 1.55 | 1.57 |
| EV71 | 41 | 17 | 0.2 | 0.2 | -0.110 | -0.157 | 0.087 | 0.080 | **0.103** | 1.00 | 1.02 | 0.90 | 0.91 | **0.89** |
| GHS | 31 | 14 | 1 | 1.5 | 0.141 | **0.176** | 0.122 | 0.090 | -0.043 | 0.87 | **0.86** | 0.88 | 0.90 | 0.96 |
| GPB | 66 | 15 | 3 | 0.7 | 0.199 | 0.162 | 0.293 | 0.313 | **0.313** | 1.00 | 1.02 | 0.94 | 0.92 | **0.92** |
| GSK3B | 42 | 17 | 0.2 | 0.2 | 0.370 | 0.460 | **0.509** | 0.454 | 0.373 | 0.62 | 0.57 | **0.55** | 0.58 | 0.62 |
| HIVPR | 113 | 16 | 3 | 0.5 | -0.089 | -0.091 | **-0.084** | -0.098 | -0.106 | 1.41 | 1.41 | **1.40** | 1.41 | 1.42 |
| HIVRT | 99 | 15 | 0.5 | 0.2 | 0.414 | 0.469 | 0.530 | **0.550** | 0.544 | 0.98 | 0.94 | 0.88 | **0.86** | 0.87 |
| HPTP | 135 | 19 | 1 | 0.2 | 0.341 | 0.340 | 0.403 | **0.417** | 0.414 | 0.50 | 0.50 | 0.47 | **0.47** | 0.47 |
| KOA | 39 | 18 | 0.2 | 1 | **0.401** | 0.354 | 0.076 | 0.196 | 0.151 | **0.54** | 0.56 | 0.67 | 0.62 | 0.64 |
| MX | 29 | 12 | 0.3 | 2 | **0.498** | 0.394 | 0.324 | 0.397 | 0.458 | **1.08** | 1.19 | 1.25 | 1.18 | 1.12 |
| PDE | 29 | 15 | 0.2 | 0.7 | **-0.296** | -0.305 | -0.377 | -0.388 | -0.380 | **0.68** | 0.68 | 0.70 | 0.70 | 0.70 |
| PTC | 40 | 12 | 3 | 0.7 | 0.368 | 0.390 | **0.421** | 0.408 | 0.399 | 21.29 | 20.91 | **20.36** | 20.57 | 20.74 |
| STEROIDS | 21 | 10 | 0.5 | 0.2 | 0.374 | 0.348 | **0.467** | 0.459 | 0.415 | 0.90 | 0.92 | **0.84** | 0.84 | 0.87 |
| TCHK | 42 | 14 | 0.2 | 3 | -0.082 | -0.057 | -0.071 | -0.088 | **-0.054** | 1.02 | 1.01 | 1.02 | 1.03 | **1.01** |
| THERM | 76 | 16 | 0.2 | 0.2 | -0.057 | 0.042 | 0.041 | 0.048 | **0.075** | 2.07 | 1.97 | 1.97 | 1.96 | **1.93** |
| THR | 88 | 16 | 2 | 1.5 | -0.060 | **-0.018** | -0.102 | -0.052 | -0.061 | 1.04 | **1.01** | 1.05 | 1.03 | 1.04 |
| TP2A | 25 | 13 | 2 | 2 | -0.196 | -0.121 | -0.107 | -0.090 | **-0.089** | 0.93 | 0.90 | 0.89 | 0.89 | **0.89** |
| VATP | 23 | 13 | 0.5 | 0.5 | **0.510** | 0.470 | 0.414 | 0.429 | 0.422 | **1.08** | 1.12 | 1.18 | 1.16 | 1.17 |
| YOPH | 39 | 14 | 0.2 | 0.2 | 0.470 | 0.499 | **0.577** | 0.564 | 0.448 | 0.93 | 0.91 | **0.83** | 0.85 | 0.95 |
| **mean** | | | | | 0.202 | 0.214 | 0.237 | **0.243** | 0.229 | 1.56 | 1.54 | **1.51** | 1.51 | 1.53 |

## What wins where

Collected over the sets of each collection, for both statistics, with the number of sets on which
each option beats its own baseline (`gauss` against `pc`, `align` and `alignf` against `cif`):

| Collection | sets | mol. | stat | pc | gauss | cif | align | alignf | gauss > pc | align > cif | alignf > cif |
|---|---|---|---|---|---|---|---|---|---|---|---|
| openff | 16 | 465 | q²cv | 0.422 | **0.457** | 0.369 | 0.382 | 0.388 | 9/16 | 10/16 | 10/16 |
|  |  |  | q²ex | 0.195 | **0.215** | 0.176 | 0.197 | 0.176 | 10/16 | 11/16 | 7/16 |
| schindler | 10 | 330 | q²cv | 0.443 | **0.476** | 0.346 | 0.367 | 0.355 | 5/10 | 8/10 | 6/10 |
|  |  |  | q²ex | 0.239 | **0.307** | 0.193 | 0.190 | 0.137 | 7/10 | 3/10 | 3/10 |
| ross | 24 | 866 | q²cv | 0.464 | **0.489** | 0.442 | 0.457 | 0.458 | 11/24 | 16/24 | 15/24 |
|  |  |  | q²ex | 0.260 | **0.295** | 0.257 | 0.264 | 0.236 | 13/24 | 18/24 | 11/24 |
| mittal | 36 | 2888 | q²cv | 0.300 | 0.323 | 0.328 | **0.340** | 0.331 | 22/36 | 20/36 | 18/36 |
|  |  |  | q²ex | 0.202 | 0.214 | 0.237 | **0.243** | 0.229 | 23/36 | 20/36 | 19/36 |
| ALL | 86 | 4549 | q²cv | 0.385 | **0.412** | 0.370 | 0.384 | 0.380 | 47/86 | 54/86 | 49/86 |
|  |  |  | q²ex | 0.221 | **0.247** | 0.226 | 0.234 | 0.211 | 53/86 | 52/86 | 40/86 |

A mean hides how the effect is distributed, and for a decision about switching an option on the
distribution is what matters. Paired per-set differences against each option's own baseline:

| Difference | mean | sd | t | median | > +0.05 | within ±0.05 | < −0.05 |
|---|---|---|---|---|---|---|---|
| `gauss` − `pc`, q²cv | +0.027 | 0.076 | +3.3 | +0.007 | 23 | 54 | 9 |
| `gauss` − `pc`, q²ex | +0.026 | 0.097 | +2.5 | +0.016 | 25 | 49 | 12 |
| `align` − `cif`, q²cv | +0.014 | 0.036 | +3.6 | +0.005 | 10 | 75 | 1 |
| `align` − `cif`, q²ex | +0.008 | 0.055 | +1.4 | +0.007 | 14 | 60 | 12 |
| `alignf` − `cif`, q²cv | +0.010 | 0.056 | +1.7 | +0.010 | 17 | 59 | 10 |
| `alignf` − `cif`, q²ex | −0.015 | 0.114 | −1.3 | −0.003 | 16 | 54 | 16 |

Three things follow. The **median** set is barely touched by any of the three options: on more than
half of the sets every difference is inside ±0.05, which is well inside the per-set noise. The
options are therefore not model improvements one can demonstrate on one's own series; they are
average improvements over many. Second, `align` is the *tidiest* of the three in
cross-validation — only one set in 86 loses more than 0.05 — but most of that gain does not reach
external validation (t = 3.6 falls to t = 1.4). The weights themselves are fold-safe, since
`mkl_cv_predict` re-derives them inside every fold; what does not survive is the α and λ chosen
alongside them on the same cross-validation. Third, `alignf` is the only option with a heavy tail:
its worst set loses 0.654 of q²ex, where the worst case of `align` loses 0.137 and of `gauss`
0.251.

The gain from the Gaussian layer is also not spread evenly over the sets. It concentrates where the
linear model has little to show: over the 42 sets with a linear q²ex below 0.2 the mean gain is
+0.040, over the 44 above it +0.013 (correlation between the linear q²ex and the gain, −0.21). A
deeper kernel is worth most where a plain overlap integral is failing, and close to nothing where
it already works.

The pattern is consistent and the collections disagree about one thing only.

* The **Gaussian layer** improves the mean of both statistics in all four collections and is the
  single best configuration of the five on the three receptor-frame collections. Its largest gains
  are on sets where the linear model has little: `openff/bace_hunt` q²ex 0.001 → 0.257,
  `ross/janssen_bace/bace_ciordia_retro` −0.086 → 0.328, `ross/gpcrs/ox2_hip_custcore`
  −0.075 → 0.213, `schindler/eg5` 0.168 → 0.351. That is the same signature as on the Sutherland
  sets, where its largest gain was on the worst linear model of the eight.
* The **indicator fields with per-field alignment** win on `examples/mittal`, and there they are
  the best configuration of the five — the only collection where the five physico-chemical fields
  are not the better starting point. This is an alignment effect and is discussed below.
* **`alignf` is the only option that is worse than doing nothing**, and the sets where it collapses
  are instructive: `schindler/cdk8` 0.470 → −0.184, `ross/merck/cdk8_5cei_new_helix_loop_extra`
  0.527 → 0.065, `openff/bace_hunt` 0.032 → −0.136. The joint alignment problem is nearly sparse on
  a congeneric series, because the field kernels are strongly correlated; the shrinkage towards
  equal weights (ρ = 1) keeps that in check on most sets but not on all, and when the solution does
  concentrate on a few indicator fields the model loses the physico-chemical fields that carried it.
  Per-field alignment cannot do this, which is why it is the robust one.

## Do the conclusions of the Sutherland benchmark hold?

Both of them do, in direction, and both effects are smaller here. To make the comparison on equal
terms rather than against numbers obtained with a different protocol, the eight Sutherland sets
were run again under *exactly* the protocol of this section (all molecules, Gasteiger charges,
external cross-validation with five reshuffles instead of the published test set). These numbers
are therefore not comparable with the README's table and are not meant to replace it:

| Set | n | F(cif) | α pc | α cif | pc | gauss | cif | align | alignf | RMSE pc | gauss | cif | align | alignf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ACE | 114 | 17 | 1.5 | 1 | 0.652 | 0.688 | **0.699** | 0.693 | 0.676 | 1.34 | 1.27 | **1.24** | 1.26 | 1.29 |
| AChE | 111 | 14 | 3 | 3 | 0.554 | **0.682** | 0.576 | 0.593 | 0.591 | 0.83 | **0.70** | 0.80 | 0.79 | 0.79 |
| BZR | 147 | 19 | 1.5 | 3 | 0.426 | **0.535** | 0.407 | 0.424 | 0.425 | 0.58 | **0.52** | 0.59 | 0.58 | 0.58 |
| COX2 | 282 | 21 | 3 | 0.2 | 0.552 | **0.586** | 0.558 | 0.570 | 0.550 | 0.78 | **0.75** | 0.77 | 0.76 | 0.78 |
| DHFR | 361 | 20 | 1.5 | 3 | 0.650 | **0.703** | 0.668 | 0.681 | 0.672 | 0.77 | **0.71** | 0.75 | 0.74 | 0.75 |
| GPB | 66 | 16 | 3 | 3 | 0.588 | 0.604 | 0.638 | **0.659** | 0.657 | 0.72 | 0.70 | 0.67 | **0.65** | 0.65 |
| THER | 76 | 16 | 1 | 3 | 0.538 | **0.640** | 0.546 | 0.626 | 0.600 | 1.37 | **1.21** | 1.36 | 1.23 | 1.27 |
| THR | 88 | 17 | 2 | 3 | 0.677 | **0.699** | 0.668 | 0.680 | 0.671 | 0.57 | **0.55** | 0.58 | 0.57 | 0.58 |
| **mean** | | | | | 0.580 | **0.642** | 0.595 | 0.616 | 0.605 | 0.87 | **0.80** | 0.85 | 0.82 | 0.84 |

Under this protocol the Gaussian layer gains 0.062 in q²cv and 0.041 in q²ex on the eight Sutherland
sets, against 0.027 and 0.026 over the 86 sets of the four collections; `align` gains 0.021 in q²cv
and 0.021 in q²ex there, against 0.014 and 0.008 here. So the effects reported in the README are
reproduced on a tenfold larger and much more varied sample, at roughly half to a third of the size.
That the effects shrink on a larger sample is the ordinary fate of a result selected on eight sets,
and it is the reason to prefer the numbers in this section when deciding whether to switch an
option on.

## Does alignment-based multiple kernel learning hurt over the five physico-chemical fields alone?

This is the one claim of the README that the five configurations above cannot touch, because all
three of their mixing variants carry the indicator fields. It was therefore run separately, on
exactly the same 86 sets and the same eight controls and under exactly the same protocol, in three
configurations: `pc+align` and `pc+alignf` are the five physico-chemical fields with the two
alignment modes at the shrinkage used everywhere above (ρ = 1, the program default), and
`pc+alignf0` is the *unshrunk* joint solution (ρ = 0) that the README's own example — thrombin
losing 0.138 of fold-safe q²cv — comes from.

| Collection | sets | mol. | stat | `pc` | `pc+align` | `pc+alignf` | `pc+alignf0` | align > pc | alignf > pc | alignf0 > pc |
|---|---|---|---|---|---|---|---|---|---|---|
| openff | 16 | 465 | q²cv | 0.422 | 0.421 | **0.428** | 0.419 | 8/16 | 8/16 | 10/16 |
|  |  |  | q²ex | **0.195** | 0.187 | 0.185 | 0.136 | 8/16 | 7/16 | 5/16 |
| schindler | 10 | 330 | q²cv | **0.443** | 0.433 | 0.426 | 0.390 | 3/10 | 3/10 | 3/10 |
|  |  |  | q²ex | 0.239 | 0.246 | **0.250** | 0.140 | 3/10 | 5/10 | 2/10 |
| ross | 24 | 866 | q²cv | 0.464 | **0.477** | 0.477 | 0.464 | 15/24 | 16/24 | 15/24 |
|  |  |  | q²ex | 0.260 | **0.275** | 0.267 | 0.181 | 14/24 | 13/24 | 11/24 |
| mittal | 36 | 2888 | q²cv | 0.300 | **0.302** | 0.301 | 0.282 | 16/36 | 16/36 | 15/36 |
|  |  |  | q²ex | **0.202** | 0.202 | 0.190 | 0.143 | 15/36 | 13/36 | 12/36 |
| ALL | 86 | 4549 | q²cv | 0.385 | **0.388** | 0.388 | 0.371 | 42/86 | 43/86 | 43/86 |
|  |  |  | q²ex | 0.221 | **0.225** | 0.217 | 0.152 | 40/86 | 38/86 | 30/86 |
| sutherland | 8 | 1245 | q²cv | 0.580 | **0.582** | 0.579 | 0.581 | 5/8 | 5/8 | 6/8 |
|  |  |  | q²ex | 0.518 | **0.525** | 0.518 | 0.507 | 6/8 | 5/8 | 2/8 |

Paired per-set differences against `pc`, the same statistics as in *What wins where*, over the 86
sets of the four collections:

| Difference | mean | sd | t | median | > +0.05 | within ±0.05 | < −0.05 |
|---|---|---|---|---|---|---|---|
| `pc+align` − `pc`, q²cv | +0.003 | 0.025 | +1.1 | −0.000 | 4 | 81 | 1 |
| `pc+align` − `pc`, q²ex | +0.004 | 0.041 | +0.8 | −0.002 | 12 | 70 | 4 |
| `pc+alignf` − `pc`, q²cv | +0.003 | 0.038 | +0.8 | +0.000 | 8 | 70 | 8 |
| `pc+alignf` − `pc`, q²ex | −0.004 | 0.060 | −0.6 | −0.008 | 19 | 55 | 12 |
| `pc+alignf0` − `pc`, q²cv | −0.014 | 0.108 | −1.2 | +0.002 | 23 | 42 | 21 |
| `pc+alignf0` − `pc`, q²ex | **−0.069** | 0.136 | **−4.7** | −0.045 | 14 | 30 | **42** |

The claim has two halves and they come out differently.

**"Not worth anything" is confirmed.** At the program's default shrinkage neither mode is worth
running over the five physico-chemical fields. Per-field alignment gains +0.003 in q²cv (t = 1.1)
and +0.004 in q²ex (t = 0.8); the joint solution gains +0.003 and loses 0.004, winning 38 of 86
sets. Every one of those numbers is inside the noise and the medians are zero to three decimals.
Against that, the same two modes over the field set that includes the indicator fields gain +0.014
and +0.008 (`align`, t = 3.6 and 1.4). So the ordering the README gives — mixing pays only once
fields that really do differ in relevance are present — is reproduced in direction, with both terms
small: on the eight Sutherland controls under this protocol `align` gains 0.022 of q²ex over `cif`
(t = 2.8) against 0.007 over `pc` (t = 1.8).

**"Actively harmful" is not confirmed at the default shrinkage, and is confirmed without it.** At
ρ = 1 nothing is harmed: `pc+align` loses more than 0.05 of q²cv on one set in 86 and of q²ex on
four, the worst of them `openff/tnks2` at −0.182, while 70 of the 86 move by less than 0.05 either
way. The unshrunk joint solution is a different matter. It loses 0.069 of q²ex on average
(t = −4.7), drops the mean from 0.221 to 0.152, wins only 30 of 86 sets and loses more than 0.05
on 42 of them, with a tail that reaches −0.414
(`ross/merck/hif2a_automap`), −0.375 (`mittal/YOPH`) and −0.369
(`ross/opls_stress/fxa_yoshikawa_set`). It is the only configuration anywhere in this benchmark
that is harmful at a significance worth the name, and it is harmful in all four collections
separately (q²ex −0.059 in openff, −0.099 in schindler, −0.079 in ross, −0.059 in mittal). The mean
standard deviation of q²ex over the five reshuffles rises from 0.106 to 0.138, so it is also the
least stable configuration run here.

The mechanism is the one the README names, and over five fields it can be counted directly. The
unshrunk solution puts 75 % of the total weight on a single field on average (median 76 %, against
20 % for equal weights), passes 90 % on 24 of the 86 sets, and sets h_f to exactly zero for 2.5 of
the five fields — on 43 sets three or more of the five are deleted outright. Shrinkage stops that:
at ρ = 1 the joint solution's largest share averages 38 % and reaches 50 % on 9 sets, and per-field
alignment averages 32 % and passes 50 % once in 86, with mean weights (electrostatic 1.07,
steric 1.06, hydrophobic 1.08, H-bond acceptor 0.96, H-bond donor 0.78) that barely leave h_f = 1.
Five strongly correlated kernels do not contain enough independent information for a data-driven
weighting to exploit, and the unshrunk problem answers that by discarding most of them. With the
indicator fields the same solution concentrates far less (largest share 25 % of 10 to 23 fields,
against 7 % for equal weights), which is why `cif+alignf` is merely useless out of sample where
`pc+alignf0` is damaging.

One detail of the README's wording does **not** reproduce, and it is the symptom rather than the
conclusion. There the harm is reported in the fold-safe q²cv, with thrombin falling from 0.674 to
0.536. Here the unshrunk solution leaves q²cv essentially alone — −0.014 over the 86 sets (t =
−1.2), median +0.002, and on the eight Sutherland sets +0.002, with thrombin at 0.677 against 0.677
for h_f = 1 — and does its damage only in external validation (−0.069, t = −4.7). The likely
explanation, which was not itself tested, is that the README's q²cv was obtained on the published
training set with the published alignment and MOL2 charges, where 88 molecules and one α/λ choice
make a single number volatile; the direction of the effect is right, the size quoted there is not
representative, and the statistic that shows it is q²ex rather than q²cv. The practical advice is
unchanged either way and now rests on 86 sets instead of eight: over the five physico-chemical
fields alone leave h_f = 1, and if the joint solution is used at all, leave the shrinkage on.

## Alignment, not the kernel

Five of the `examples/mittal` sets contain exactly the same molecules with exactly the same
activities as their `examples/sutherland` counterparts and differ **only** in the three-dimensional
coordinates — Sutherland's published alignment against a conformer generated by RDKit ETKDGv3 and
superimposed on a reference molecule by its maximum common substructure. Run under one protocol they
isolate the cost of an automatic alignment from everything else:

| Set | n | pc aligned | gauss aligned | cif aligned | align aligned | alignf aligned | pc auto | gauss auto | cif auto | align auto | alignf auto |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ACE | 114 | 0.586 | 0.604 | 0.657 | 0.645 | 0.626 | 0.414 | 0.402 | 0.498 | 0.484 | 0.386 |
| AChE | 111 | 0.516 | 0.569 | 0.499 | 0.510 | 0.516 | 0.189 | 0.165 | 0.108 | 0.130 | 0.123 |
| GPB | 66 | 0.578 | 0.570 | 0.605 | 0.633 | 0.617 | 0.199 | 0.162 | 0.293 | 0.313 | 0.313 |
| THER | 76 | 0.396 | 0.431 | 0.440 | 0.498 | 0.433 | -0.057 | 0.042 | 0.041 | 0.048 | 0.075 |
| THR | 88 | 0.593 | 0.627 | 0.598 | 0.634 | 0.625 | -0.060 | -0.018 | -0.102 | -0.052 | -0.061 |
| **mean** | | 0.534 | 0.560 | 0.560 | 0.584 | 0.563 | 0.137 | 0.150 | 0.168 | 0.185 | 0.167 |

The whole model collapses. Averaged over the five configurations, mean q²ex falls from 0.56 to 0.16
and mean q²cv from 0.64 to 0.23; on thrombin, where the mean MCS superposition error is 2.9 Å, every
configuration is at or below zero where the published alignment gives 0.59 to 0.63. For scale: the
largest effect any kernel option has in this benchmark is 0.027. The alignment decides fifteen times
as much.

The hyper-parameter search says the same thing in its own language. The attenuation factor α sets
how local a field is, and a kernel built from sharply peaked fields can only work if the molecules
are in a common frame:

| Collection | stat | pc | gauss | cif | align | alignf |
|---|---|---|---|---|---|---|
| openff | median alpha | 1 | 0.5 | 3 | 3 | 1 |
| | % at alpha = 0.2 | 31 | 19 | 6 | 6 | 6 |
| | % at alpha = 3.0 | 19 | 25 | 69 | 69 | 44 |
| schindler | median alpha | 2 | 2 | 1 | 3 | 1 |
| | % at alpha = 0.2 | 10 | 10 | 0 | 0 | 10 |
| | % at alpha = 3.0 | 30 | 40 | 40 | 60 | 20 |
| ross | median alpha | 2 | 2 | 2 | 3 | 3 |
| | % at alpha = 0.2 | 12 | 12 | 8 | 8 | 17 |
| | % at alpha = 3.0 | 42 | 42 | 42 | 54 | 50 |
| mittal | median alpha | 0.5 | 0.5 | 0.2 | 0.2 | 0.3 |
| | % at alpha = 0.2 | 31 | 44 | 53 | 56 | 47 |
| | % at alpha = 3.0 | 14 | 8 | 6 | 8 | 8 |
| sutherland | median alpha | 2 | 1.5 | 3 | 3 | 3 |
| | % at alpha = 0.2 | 0 | 0 | 12 | 12 | 0 |
| | % at alpha = 3.0 | 38 | 12 | 75 | 62 | 50 |

With the published alignment the indicator-field configurations run to the **top** of the α grid
(median 3.0, three quarters of the `cif` runs at the upper edge and one at the lower); with the
automatic alignment of the same molecules they run to the **bottom** (median 0.2, over half at the
lower edge). The lower edge is where the model stops being a field model: as α → 0 a continuous
indicator field tends to the plain count of atoms of that type (Sitnikov et al. 2015), so the
kernel degenerates towards an alignment-free fragment-descriptor kernel. That is exactly why the
indicator fields are the *best* configuration on `examples/mittal` and the worst on the three
receptor-frame collections — on `examples/mittal` they are not being used as fields at all. A q²
gain from `align` on that collection is therefore evidence about a fragment kernel, not about
multiple kernel learning on molecular fields.

Two things should not be over-read from this. The per-set correlation between the mean MCS
superposition error and q²cv *within* `examples/mittal` is only −0.36 over its 36 sets (−0.41 for
the indicator-field configuration): the series also differ in size, activity range and chemotype
diversity, so the cross-set correlation is loose even though the controlled comparison is
unambiguous. The alignment error of a series predicts its q² only weakly; it is the *difference*
between two alignments of one series that is decisive. And the automatic alignment is not uniformly bad —
`mittal/AI` reaches q²ex 0.669 and `mittal/COMT` 0.608 — so the collection is usable for comparing
methods on equal footing, which is what `make_mittal_examples.py` says it is for. It is not usable
for comparing against literature 3D-QSAR statistics.

## Caveats

* **Absolute values are low**, and should be. Mean q²ex is 0.22 to 0.26 on the receptor-frame
  collections against 0.52 on the Sutherland sets under the same protocol. These are not curated
  3D-QSAR data sets: they are FEP benchmarks, chosen for being hard cases of relative binding free
  energy, with narrow activity ranges over 20 to 90 congeners: the standard deviation of pAffinity
  is 0.50 to 1.25 log units (median 0.78) against 0.76 to 2.27 (median 1.24) for the Sutherland
  sets. A q² is a fraction of a variance, and when the variance is small there is little to
  explain. 11 of the 86 sets have a negative q²ex under every one of the five configurations.
* **Per-set numbers are soft.** The mean standard deviation of q²ex over the five reshuffles is
  0.093 (`gauss`) to 0.135 (`alignf`); an independent estimate from the two preparations of
  `cmet` is 0.11. Differences below roughly 0.1 on a single set are not interpretable, and nothing
  in the discussion above rests on one.
* **q²cv is biased** and is reported only because it is what the program's model panel shows. The
  hyper-parameters, and for the alignment runs the mixing coefficients, were selected on the same
  10-fold cross-validation. The gap between the two statistics is the size of that bias: 0.16 on
  average here.
* `mittal/PTC` activities are not in log units (their standard deviation is 26.8 against about 1
  for every other set), so its q² is comparable with the rest but its RMSE, around 20, is not.
* `mittal/AMPH1` is 130 decapeptides of 169 atoms each; its kernels cost about fourteen times a
  normal set, and its model is empty in every configuration (q²cv 0.063 to 0.132), which is what a
  mean MCS superposition error of 3.2 Å over a flexible peptide should give.
* The indicator atom-type list is selected from all molecules of a set rather than per outer fold
  (structures only, no activities), as noted under the protocol.
* The α grid stops at 3.0 and 27 of the 86 `cif` runs selected that upper value, so on the
  well-aligned collections the optimum is in part outside the grid and the indicator-field
  configurations are, if anything, under-tuned there. The grid was left at the program default so
  that these numbers describe what a user gets without tuning.

## Reproducing

Every configuration of this benchmark is reachable from the shipped command line, so no extra code
is needed to repeat a single run. For one data set, the five configurations are

```bash
D=examples/schindler/syk.sdf
C="--normalize --nested --nested-on all --outer-folds 5 --repeats 5 --seed 0"
python cmf_cli.py build $D $C                                              # pc
python cmf_cli.py build $D $C --kernel-layer gauss                         # gauss
python cmf_cli.py build $D $C --cif --cif-min 5                            # cif
python cmf_cli.py build $D $C --cif --cif-min 5 --mkl align                # align
python cmf_cli.py build $D $C --cif --cif-min 5 --mkl alignf               # alignf
```

with the α, λ and γ grids left at their defaults, which is what the tables use; `--json <file>`
writes the statistics of a run for collection. The tables above were produced by a driver that
loops over the SDF files of the four collections and calls `cmfqsar.model.grid_search` and
`cmfqsar.model.nested_cv` directly, so that the per-field kernels of a set are computed once and
reused by all five configurations; that driver is not part of the repository, since it adds no
capability to the program. The whole benchmark is 470 runs, 8.7 hours of processor time, run in about
three and a half hours of wall clock on 32 cores with six concurrent processes. Five sets account
for a third of it: `mittal/AMPH1` 1.9 h (130 decapeptides, 22005 atoms), `mittal/DHFR` 1.2 h,
`sutherland/DHFR` 0.8 h, `sutherland/COX2` 0.8 h and `mittal/COX2` 0.6 h. Everything else is
minutes per set.

No source file of `cmfqsar/` was changed for this benchmark; it reports the behaviour of the
program as it stands.
