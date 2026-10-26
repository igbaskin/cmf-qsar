# Publication plan for the CMF implementation

*Drafted 20 October 2026, revised 22 October 2026 after the Hammett, quantum-field, conceptual-DFT
and Am/Eu results. The plan names what each paper takes from the repository, what it still needs,
and which journal it is aimed at. File references point at the material a paper is written from.*

## What is available

The repository holds, as of 22 October 2026:

* The implementation: analytical kernels for every field type, the GUI, the CLI and the library;
  the typeset manual (`docs/CMF_manual.pdf`, built from `docs/manual.md`).
* Reproductions of the published numbers: Table 2 of the 2013 paper on the eight Sutherland sets,
  the continuous-indicator-field models of the 2015 paper, and the Am/Eu separation set of the
  same paper (`docs/benchmark.md`, `benchmarks/results/am_eu/`). The hydrogen-counting bug of the
  original parameterisation was found during this work.
* The 86-set kernel benchmark on the public congeneric collections: Gaussian kernel layer,
  per-field and joint multiple kernel learning, indicator fields split by set size, and the
  same-molecules-different-coordinates comparison of the Sutherland and Mittal collections
  (`docs/benchmark.md`, third part).
* The alignment and topomer studies: curated against automatic alignment, aligned CMF against the
  topomer rule and the alignment-free kernels, on the Sutherland sets and on the corrosion-inhibitor
  series (`benchmarks/results/alignment/`, `topomer/`, `topomer_mittal/`, `corrosion/`,
  `corrosion_hcl/`).
* The Hammett study: 95 monosubstituted benzenes, σ_m, σ_p and π from the fields, with Gasteiger
  and with Hirshfeld charges, single conformer and symmetry ensemble
  (`benchmarks/results/hammett/`, README section "A field-based Hammett equation").
* Quantum-chemical fields: Fukui fields, Hirshfeld electrostatics and per-molecule descriptors on
  four corrosion series (`benchmarks/results/corrosion_qm/`); the conceptual-DFT measurement that a
  Fukui function cannot be represented by atom-centred Gaussians, and the three-kernel comparison
  (grid, fitted, condensed) on one series (`benchmarks/results/cdft_kernels/`, `cmfqsar/cdft.py`).
* Interpretation and validation tools, implemented and tested but without published case
  studies: Shapley attribution under nonlinear kernels, field differences, pharmacophore extraction
  from co-fields, pocket complementarity, probe fields, descriptor block, y-randomisation,
  applicability domain, bootstrap intervals, Gaussian-process regression and classification,
  one-class models, alignment by prediction confidence, kernel GTM, conformational ensembles.

## The papers

| # | Paper | Source material | Venue | Status |
|---|---|---|---|---|
| 1 | Software paper | Implementation, manual, the three reproductions, descriptor block, probe fields, validation checks, ORCA dialog | J. Chem. Inf. Model. (application note) or J. Cheminformatics (software) | Ready to write |
| 2 | Hammett equation as a CMF model | Hammett study, both charge models, the co-field figures | J. Phys. Org. Chem., J. Org. Chem., or J. Chem. Inf. Model. | Results done; three additions below |
| 3 | Alignment, not the kernel | 86-set benchmark, Sutherland-versus-Mittal drop in q², α at the grid edge as a diagnostic, bootstrap intervals, the ACE descriptor finding | J. Chem. Inf. Model. or JCAMD | Results done; text in `docs/benchmark.md` |
| 4 | Topomer and frameless CMF | Topomer construction, invariant autocorrelation kernel, count baseline, leave-one-chemotype-out validation, cut-versus-pose decomposition; corrosion series as the no-frame case | JCAMD or Mol. Inform. | Results done; a few more heterogeneous sets needed |
| 5 | Quantum-chemical fields in CMF | Corrosion QM runs, CDFT kernel comparison, Am/Eu quantum stage; Hammett as the positive control, cited from paper 2 | J. Chem. Inf. Model. or J. Comput. Chem. | Needs the runs listed below |
| 6 | Corrosion inhibitors, application | Scaffold-tree series, protonation in HCl, frames compared, co-field interpretation of the best series, fields against quantum descriptors | Corrosion Science or J. Mol. Liq. | Results mostly done |
| 7 | Interpretation | Shapley, field difference, pharmacophore, complementarity, probe co-fields; FEP sets with crystal pockets | JCAMD | Needs case studies |
| 8 | Probabilistic CMF | GP variances, Laplace GP classifier, one-class descriptions, alignment by prediction confidence, applicability domain | J. Chem. Inf. Model. | Needs a virtual-screening benchmark |

Short notes, included in papers 1 and 7 unless they become larger: conformational ensembles as
mean-map kernels (the 4D extension with no new mathematics); kernel GTM maps of field space.

### 1. Software paper

The contribution is what is new, not the reproduction: closed-form kernels for every field type
including fitted probe fields, the pocket overlay in the frame of the reference crystal structure,
Shapley attribution that holds under a nonlinear kernel, the descriptor block that keeps the model
linear in its inputs, the three validation checks on precomputed kernels, and the corrected
parameterisation. The reproductions of the 2013, 2015 and Am/Eu numbers are the validation section.

Before submission:

* Public repository, release tag, Zenodo DOI, `pip install`, the manual as supplementary material.
* One entry point that rebuilds every table in `docs/benchmark.md` and a lock file with pinned
  versions, so the numbers reproduce on a reviewer's machine.
* A data statement: where each public set came from, with conversion scripts rather than copies
  where redistribution is unclear (`docs/public_3dqsar_datasets.md` is the basis).

### 2. Hammett equation as a CMF model

Hammett's equation is CMF on a congeneric series with trivial alignment: the shared frame is the
benzene ring, the co-field is the sensitivity map of benzoic-acid ionisation to charge and bulk
around the ring, and it turns out to be a learned form of Topsom's electrostatic model of the field
effect. Fragment charges carry the field effect and miss the resonance effect; whole-molecule DFT
charges supply it, improving σ_p and leaving σ_m unchanged, and the two co-fields then separate
onto the meta carbons and the whole ring.

This is the fastest paper to write and the most distinctive. It goes to ChemRxiv in the same week
as paper 1 so each can cite the other.

Additions reviewers will ask for, all easy to do:

* Held-out substituents beyond cross-validation, and Taft's σ_I and σ_R, to show the model
  separates the inductive and resonance components rather than fitting two numbers.
* One other scaffold, pyridine or naphthalene, to show whether the co-field transfers.
* A comparison with published DFT-based σ predictions (electrostatic-potential and NMR-shift
  correlations) so the q² has a reference point.
* The amide outliers (CONH2, N(CH3)COCH3) resolved or discussed through the ensemble mode.

### 3. Alignment, not the kernel

Written from the third part of `docs/benchmark.md`. The argument: the Gaussian layer is a
reliable average gain and a matter of chance per set; per-field alignment helps a little and the
joint solution does not; indicator fields help only above about forty molecules; and none of this
is the main effect, because the alignment decides fifteen times as much. The bootstrap intervals
now make the ±0.1 resolution claim formal. The ACE finding, that four whole-molecule descriptors
alone are better than the 3D model, joins the alignment-free baseline as a second diagnostic of a
data set.

Before submission:

* One 2D baseline on all 86 sets (ECFP ridge or random forest) so the claim is measured against a
  method that never needed an alignment. CoMFA and CoMSIA numbers for the Sutherland sets are in
  the 2004 paper already.
* Wilcoxon signed-rank beside each paired t-test, and a figure of paired q²ex differences.

### 4. Topomer and frameless CMF

Paper 3 is about kernels versus alignment on sets that come with a frame; this paper is about
what to do when no frame exists. Keep that division strict: the Sutherland-versus-Mittal table
appears in paper 3 only. The corrosion series are the natural no-frame case and belong here for
the frame question only.

### 5. Quantum-chemical fields in CMF

Largely a negative result, and it must be presented as a measurement with a positive control:

* Hammett (paper 2): whole-molecule charges help where resonance matters.
* Corrosion series: Fukui fields, Hirshfeld electrostatics and quantum descriptors add nothing to
  the five fields, and descriptors alone are far worse than fields.
* Conceptual-DFT fields: a Fukui function cannot be fitted by atom-centred isotropic Gaussians
  (relative fit error 0.8 to 0.9), and this does not affect the model, since the exact grid kernel
  does no better than the condensed index.
* Am/Eu: classical fields reproduce the published model and counts add nothing, so geometry is
  decisive; the quantum stage decides whether there is a second positive case.

Runs still needed:

* The Am/Eu quantum stage (`benchmarks/run_am_eu.py --stage orca`, then `--stage quantum`).
* The CDFT kernel comparison on the other three corrosion series that have ORCA results, so the
  claim is supported by more than one series of 20 molecules.
* Bootstrap intervals on the corrosion series. With 20 to 29 molecules the differences are inside
  the resolution, and the paper should say so with numbers.

### 6. Corrosion inhibitors, application

Narrowed to the chemistry: the scaffold-tree series as a way of defining congeneric sets from a
2D database, protonation in 1 M HCl, which frame to use for each series, the co-field
interpretation of the best series, and the corrective finding that fields on an aligned series are
better than the quantum descriptors the corrosion-QSAR literature favours. Papers 4, 5 and 6 all
use the corrosion series; each takes one question (the frame, the quantum fields, the chemistry)
and none repeats another's table.

### 7. Interpretation

Exact Shapley attribution for kernel field models, the field difference as "what a candidate is
missing", the pharmacophore extracted from a co-field, complementarity with the pocket, and the
probe co-field as "where a water would bind" (probe fields add no accuracy and are justified by
interpretation only). Needs two or three case studies on the FEP sets with crystal pockets and a
medicinal-chemistry co-author to judge the interpretations on a real series.

### 8. Probabilistic CMF

The 2019 Conclusions proposed alignment by prediction confidence; it is implemented, together with
GP regression variances, the Laplace GP classifier, the one-class descriptions and the
applicability domain defined from the kernel. Needs a virtual-screening benchmark (DUD-E or LIT-PCBA
subsets) and a comparison of the posterior variance against the domain measure as predictors of
error.

## Sequence

| When | What |
|---|---|
| Months 1 to 2 | Papers 1 and 2 to ChemRxiv in the same week; repository public at submission |
| Months 2 to 5 | Paper 3, submitted once paper 1 has a preprint DOI to cite |
| Months 3 to 6 | Am/Eu quantum stage, remaining CDFT runs, bootstrap on corrosion; then paper 5 |
| Months 4 to 8 | Papers 4 and 6 in parallel |
| Months 6 to 12 | Papers 7 and 8, each with new case studies |
| Year 2 | A review that connects the papers of the series, the successor of the 2014 Springer chapter |

## Risks

* **Overlap.** Papers 3 and 4 both measure the cost of alignment; papers 4, 5 and 6 all use the
  corrosion series. The divisions above are the rule; a table appears in one paper only.
* **Novelty of paper 1.** Six prior publications on the method; the paper must begin with what is
  new in the implementation and treat the reproductions as validation.
* **Negative results in paper 5.** Acceptable with the Hammett positive control shown alongside
  and with intervals that show what the data can and cannot resolve; not acceptable as a plain
  "nothing helped".
* **Small series.** The corrosion sets of 20 to 30 molecules cannot separate differences below
  about 0.1 in q²; every comparison on them is reported with a bootstrap interval.
