# Publication plan for the CMF implementation

*Drafted 20 October 2026, revised 22 October 2026, rewritten 27 October 2026 after the results of
the development plan's parts B to I, and revised 31 October 2026 after the medicinal-chemistry
items, the trajectory and hydration-site studies, the cheap quantum stage, the local-kernel
comparison with SOAP, the COMPAS-1 pilot and the lithium-hop model. The plan names what each
paper takes from the repository, what it still needs, which journal it is aimed at, and in which
order the papers are written. File references point at the material a paper is written from. The
companion document, `docs/development_plan.md`, says what gets built; this one says what gets
written.*

## What changed since 27 October

Four days of work, 65 commits, and three new papers:

* **The medicinal-chemistry items of part I are measured** (MC1 to MC11), and most of them did
  not pass their gate. What passed: the co-field weighting ranks the measured activity difference
  of matched pairs better than a fingerprint distance (`results/bioisostere/`); tying an
  anti-target to a potency costs the potency nothing and puts the difference co-field at the basic
  nitrogen (`results/antitarget/`, the H3/hERG series of `examples/antitarget/`); a masked co-field
  recognises active molecules of a scaffold the model never saw (`results/maskedmatch/`); the
  cheap similarity filter is as good as the posterior rule for a batch of ten (`results/batch/`).
  What did not: a transformation does not carry its effect across series
  (`results/transformfield/`), a field score cannot say what to grow a fragment into
  (`results/fraggrow/`), the variance field does not score a ring closure (`results/restriction/`),
  the two-objective loop puts the compounds the authors made in the worse half
  (`results/designloop/`), and the local field finds a metabolic site next to the count, not
  instead of it (`results/metabolism/`).
* **The fourth dimension is closed.** A trajectory in the crystal site does not beat the crystal
  pose, with one reading fixed in advance (`results/md/`); the water as a field helps at none of
  sixteen definitions of a hydration site, and the pre-registered tests of the site definition
  failed on three sets (`results/solvent/`); where the pose is not known the ensemble is worse,
  not better. With the conformer and protonation results this is now a paper, not a note.
* **A cheap quantum stage over every series** (GFN2-xTB, 0.25 s a molecule against 360 s for
  ORCA): 7 of 52 series are helped beyond the interval and 1 is made worse (`results/xtb/`).
  Kernel canonical correlation says why so few: what the reactivity fields know about the activity
  is the part of them that shape and charge already contain (`results/kcca_quantum/`).
* **What the field sees that a fingerprint does not**: on the Sutherland sets the part of the
  field a fingerprint does not contain predicts the activity at q² +0.16, the part it contains at
  +0.48 (`results/kcca/`). The contact profile with the pocket is not a predictor on its own
  (`results/kcca_pocket/`).
* **Four learners measured and none replaces the ridge**: kernel PLS ties it in cross-validation
  and is behind it on the test sets (`results/kpls/`);
  kernel Fisher discriminant analysis gains no accuracy and is kept because it is the classifier
  that has a co-field (`results/kfda/`); kernel logistic regression ties the GP classifier and no
  classifier beats the base rate on calibration (`results/klr/`); molecules without an activity
  do not help, and measurably hurt on two sets (`results/lapkrr/`).
* **The local kernel against SOAP** on 5096 carbons: the two-point kernel 8.8 ppm, SOAP 5.7, the
  three-distance autocorrelation with the fields as channels 5.5 (`results/soap_gate/`,
  `results/triangles/`); the error is flat from about 800 molecules, so the model is limited by
  its representation and not by data.
* **COMPAS-1**: the HOMO-LUMO gap of 8,678 polybenzenoid hydrocarbons in a frame given by rule,
  with ring-centred fields, test R² 0.87 and positive R² on size extrapolation where every
  representation without a ring field is negative (`results/compas/`, `REPORT.md` there).
* **The lithium hop**: 1732 symmetry-distinct hops in 472 lithium oxides, the barrier from the
  field around the path with the rotation about the path averaged in closed form, q² +0.65
  against +0.49 for the counts (`results/hops/`); partial occupancy as the exact mean field.
* **Surfaces**: a molecule posed on a surface by rule, without the slab, gives q² about 0.3 on
  the benzimidazoles and 0 on the quinolines; the rest of the earlier number was the relaxation
  telling the model which molecules it had bound (`results/fe110/`). Paper 14 states that.
* **Software**: 92 modules, 31 subcommands, 1618 tests; a supported interface of nine names with a
  test that holds their signatures; the low-rank path fits in the space of the landmarks with the
  co-field intact and saves as a model (`results/lowrank/`); project files and undo in the
  application; readers for trajectories.
* **Paper 6 is extended to five scaffold nodes and across chemotypes**
  (`examples/corrosion_fe110/potency_reliability.py`): between different first authors the spread
  of one compound's potency is 1.17 units on the benzimidazole node, 1.19 over five nodes and 1.19
  across chemotypes, so the ceiling of q² 0.6 for pooled node medians is a constant of the
  literature, not a property of one node. A third node with a UMA run is not needed.

## What changed since 22 October

The 22 October plan had eight papers, five of them waiting for runs. Those runs are done, and
the five days since produced results that the old plan had no place for:

* The frameless kernels have a measured verdict on their own ground: seven cyclodextrin tables
  with no scaffold, where the alignment-free kernel beats the aligned one on six of seven
  (`benchmarks/results/cyclodextrin/`); the `cmf diagnose` recommendation checked on sets whose
  answer is known (`results/diagnose/`); handedness, which the signed channels cannot carry
  between molecules and a stereo-centre descriptor can (`results/chirality/`).
* Two kernels that leave the molecule behind: the periodic autocorrelation of a crystal, measured
  on Matbench phonons (`results/matbench_phonons/`), and the in-plane invariant kernel of an
  adsorbed molecule, measured on 72 benzimidazoles and 63 quinolines relaxed on Fe(110) and on
  OC20's DFT energies (`examples/corrosion_fe110/`).
* A target that belongs to an atom: 13C shifts predicted from the field around the carbon
  (q² +0.97 against +0.85 for the atom-centred count), carboxylic acid pKa where the same
  construction gains nothing, and a co-field around one aromatic carbon that reproduces the
  substituent-increment tables (`results/local/`, `results/local_cofield/`).
* Several endpoints on one kernel: the selectivity co-field, two cyclodextrin cavities, seven
  hosts with one never measured, and the AMPA/glycine set of the 2003 J. Med. Chem. paper
  on which the selectivity field was first proposed (`results/multitask/`, `two_endpoints/`,
  `tasks/`, `selectivity/`).
* The uncertainty tools measured: conformal intervals that hold their coverage where the
  Gaussian-process variance does not (`results/conformal/`), which estimate identifies a bad
  prediction (`results/errorpred/`), ranking by a confidence bound (`results/bound/`), the
  evidence as a selector of hyper-parameters (`results/evidence/`, `perfield/`, `mixing/`).
* Screening against measured inactives on LIT-PCBA, where every one-class field description
  loses to a fingerprint and the kernel used with the inactives as unlabelled data draws level
  (`results/screening/`, `fusion/`).
* The interpretation tools measured instead of only implemented: the error bar of a co-field
  (`results/cofield_error/`), the co-field pharmacophore as a query (`results/query/`), the
  co-field cosine as a map of SAR transfer (`results/transfer/`), and three series worked
  through with pockets (`results/cases/`).
* Which molecule to make next: six acquisition rules on the Sutherland sets
  (`results/active/`), the substituent design loop (`results/design/`), the same loop on a table
  without a scaffold (`results/guest_design/`).
* Negative results that close questions: protonation ensembles (`results/protonation/`),
  increments fitted to the data (`results/increments/`), the Am/Eu quantum stage
  (`results/am_eu/`), atoms with a direction (`results/anisotropic/`), a one-body
  complementarity score against a cyclodextrin cavity (`results/host_pocket/`).
* The software side of paper 1 is in place: `pip install`, continuous integration on three
  operating systems (`.github/workflows/ci.yml`), a lock file, a data statement
  (`docs/public_3dqsar_datasets.md`), four tutorials that run in Colab, an HTML API reference,
  and one entry point that rebuilds every table (`benchmarks/run_all.py --manifest`).

What is still not in hand is unchanged: a public release with a DOI, and a medicinal chemist's
judgement of the case studies.

## Rules the plan follows

1. **A table appears in one paper only.** Several papers draw on the same series (the corrosion
   sets, the cyclodextrin tables, the Sutherland sets, the LIT-PCBA sets, the shift set). Each
   paper takes one question from a series and cites the others for the rest. The assignment below
   is the rule.
2. **Each paper has one claim that a reader can state in a sentence.** Results that do not
   serve the claim go to the supplementary material or to another paper.
3. **Negative results are reported with a positive control and an interval.** "Nothing helped"
   is not a paper; "this did not help, that did, and the data can resolve differences down to
   0.1 in q²" is. A gate that was set before the measurement and not met is reported as that.
4. **The repository is cited by DOI from the first paper on.** No paper is submitted before the
   release of item 0 below, and every table in a paper is produced by a script named in it.
5. **Three papers at a time at most**, since there is one author. The tiers below are the order.

## Item 0: the release

Before any submission: a public repository, a release tag, a Zenodo DOI, the package on PyPI,
the manual (`docs/CMF_manual.pdf`) as supplementary material, and a run of
`benchmarks/run_all.py` on a machine that is not this one, so the claim "the numbers reproduce"
has been tested once. The version is 1.0.0 in `cmfqsar/__init__.py` and `CITATION.cff`; the
release should carry the commit the manifest records. The supported interface of nine names
(`tests/test_stable_surface.py`) is what the release promises to keep.

## The papers

| # | Paper, with its one-sentence claim | Source material | Venue | Status |
|---|---|---|---|---|
| 1 | **Software paper.** A library, a tool and an application that give every CMF kernel in closed form, with co-fields, their uncertainty, and the checks around a q² | implementation, manual, tutorials, the three reproductions (`docs/benchmark.md` parts 1 and 2), speed and scale (`results/speed/`, `nystrom/`, `lowrank/`, screening in blocks), the supported interface | J. Chem. Inf. Model. (application note) or J. Cheminformatics | Ready to write once item 0 is done |
| 2 | **Hammett equation as a CMF model.** Substituent constants are a co-field around the ring, and resonance needs whole-molecule charges | `results/hammett/`, README "A field-based Hammett equation" | J. Phys. Org. Chem., J. Org. Chem. or J. Chem. Inf. Model. | Results done; three additions below |
| 3 | **Alignment, not the kernel, and not the learner or the hyper-parameters either.** On 86 series the superposition decides fifteen times as much as the kernel; the evidence chooses as well as cross-validation; kernel PLS does not beat the ridge; and the part of the field a fingerprint does not contain carries little of the activity | `docs/benchmark.md` part 3, `results/alignment/`, `evidence/`, `perfield/`, `mixing/`, `pharmfields*/`, `kpls/`, `kcca/` | J. Chem. Inf. Model. or J. Comput.-Aided Mol. Des. | Results done; a plain 2D baseline still needed |
| 4 | **Working without a frame.** The rotation-invariant autocorrelation kernel, the topomer rule and the diagnosis that says which route a series needs | README "Alignment-free, but geometry-aware", "Topomer fields", `results/topomer*/`, `diagnose/`, `chirality/`, `structure/`, `corrosion/` (the frame columns only) | J. Comput.-Aided Mol. Des. or J. Chem. Inf. Model. | Results done |
| 5 | **Quantum-chemical fields in CMF.** They help on about one series in eight, a cheap semi-empirical stage finds which, and what they know about the activity is mostly what shape and charge already contain | `results/xtb/`, `kcca_quantum/`, `corrosion_qm/`, `cdft_kernels/`, `anisotropic/`, `am_eu/` (the quantum rows), Hammett cited from paper 2 | J. Chem. Inf. Model. or J. Comput. Chem. | Results done |
| 6 | **Corrosion inhibitors: what the literature data can support.** The pooled potency of a scaffold node is reproducible between laboratories (r 0.64) yet no model reaches a third of that ceiling, because the node is a set of series-specific trends with no shared scale; within one protocol fields and frontier-orbital descriptors are level, and a computed adsorption energy predicts nothing of inhibition; the protonation state is one state, not an ensemble | `examples/corrosion_fe110/` (the potency, mixed-model, leave-paper-out, descriptor-baseline, E_ads and reliability rows), `results/corrosion_hcl/`, `protonation/`, `examples/scaffold_tree_export.py` | Corrosion Science or J. Mol. Liq. | Results done on five nodes |
| 7 | **How much of a co-field is real, and what it can be read for.** The co-field has a closed-form error bar; masked by it, it is a usable pharmacophore, a map of SAR transfer, a bioisostere table and a scaffold-hopping score; a transformation read from it does not transfer between series | `results/cofield_error/`, `query/`, `transfer/`, `cases/`, `bioisostere/`, `maskedmatch/`, `transformfield/`, `kfda/` (the co-field rows), README "Three series, worked through", Shapley and field-difference sections | J. Chem. Inf. Model. | Results done; needs a chemist's judgement of the cases |
| 8 | **How certain is one prediction.** Conformal intervals keep their coverage, the GP variance does not, no estimate orders the errors well, no classifier beats the base rate on calibration, and molecules without an activity do not help | `results/conformal/`, `errorpred/`, `bound/`, `ranking/`, `klr/`, `lapkrr/`, README "Chance, domain and uncertainty" | J. Chem. Inf. Model. or J. Cheminformatics | Results done |
| 9 | **Several endpoints on one kernel: the selectivity co-field.** Selectivity is a difference of co-fields; sharing helps only when the endpoints share molecules unevenly; an anti-target beside the potency costs the potency nothing and the difference co-field says where the two part; an endpoint nobody measured can be predicted from its neighbours | `results/selectivity/`, `multitask/` (method rows), `two_endpoints/`, `tasks/`, `antitarget/`, `designloop/`, `cmfqsar/multitask.py`; the 2003 J. Med. Chem. set and the H3/hERG set | J. Med. Chem. (perspective) or J. Chem. Inf. Model. | Results done |
| 10 | **A target that belongs to an atom.** The field around a carbon predicts its shift better than its atom-centred count and, with three distances, as well as SOAP; a frame around one atom gives a co-field that reads as a substituent table; the same construction finds a metabolic site beside the count and not instead of it | `results/local/`, `local_cofield/`, `soap_gate/`, `triangles/`, `metabolism/`, `examples/local/`, `cmfqsar/atomtargets.py`, `localenv.py`, `localalign.py`, `triangles.py` | J. Chem. Inf. Model., Magn. Reson. Chem. or J. Org. Chem. | Results done; needs a published shift predictor as the reference point |
| 11 | **Virtual screening with measured inactives.** One-class field descriptions lose to a fingerprint on LIT-PCBA; the same kernel with the inactives as unlabelled data draws level; four classifiers tie; the posing rule is half of any such figure | `results/screening/`, `fusion/`, `kfda/` and `klr/` (the accuracy rows), `examples/litpcba/`, `cmfqsar/pu.py`, the one-class and GP-classifier sections of the README | J. Chem. Inf. Model. or J. Cheminformatics | Results done; two targets only |
| 12 | **Which molecule to make next.** A confidence bound finds hits and a variance rule improves the model, and the two are different tasks; the substituent loop ranks held-out groups at ρ +0.4; a plate of ten needs only the cheap filter; and three questions the posterior cannot answer: what to grow a fragment into, whether a ring closure will pay, and which transformation to copy | `results/active/`, `design/`, `guest_design/`, `batch/`, `fraggrow/`, `restriction/`, `cmfqsar/design.py` | J. Chem. Inf. Model. or Mol. Inform. | Results done |
| 13 | **Host-guest binding without a frame.** Seven cyclodextrin tables, where the invariant kernel beats the alignment, names the group that enters the cavity, and a one-body complementarity score points the wrong way | `results/cyclodextrin/`, `guest_atoms/`, `host_pocket/`, `examples/cyclodextrin/`; the seven-host sharing cited from paper 9 | Phys. Chem. Chem. Phys., J. Phys. Chem. B or J. Incl. Phenom. | Results done |
| 14 | **Continuous molecular fields of crystals and surfaces.** The periodic autocorrelation on Matbench phonons, and the in-plane invariant kernel that keeps the height of an adsorbate and nothing else: as accurate as the best aligned model with no alignment, better on one node and level on another and on OC20; posed by rule without the slab, the structure alone predicts a third of what the relaxed frame did | `results/matbench_phonons/`, `fe110/`, `cmfqsar/periodic.py`, `planealign.py`, `examples/corrosion_fe110/surface_kernel_eads*.csv`, `surface_channels_eads.py`, `oc20_anchor.py` | Digital Discovery, J. Chem. Theory Comput. or npj Comput. Mater. | Needs two more Matbench tasks and a DFT anchor on the benzimidazole structures |
| 15 | **Review.** The CMF series from 2009 to the present, the successor of the 2014 Springer chapter | papers 1 to 18 | WIREs Comput. Mol. Sci. or J. Comput.-Aided Mol. Des. (perspective) | Year 2 |
| 16 | **The gap of a benzenoid in a frame given by rule.** On 8,678 polybenzenoid hydrocarbons the HOMO-LUMO gap is topology: ring-centred fields carry it, and they are the one representation that extrapolates to larger molecules | `results/compas/` (`table.md`, `REPORT.md`, `NOTES.md`), `cmfqsar/pbh.py`, `benchmarks/run_compas.py`, manual "A frame by rule" | J. Chem. Inf. Model., J. Phys. Chem. A or Phys. Chem. Chem. Phys. | Results done; the gap to the equivariant networks must be stated |
| 17 | **The barrier of a lithium hop from the field around the path.** The hop is the sample, the rotation about it is averaged in closed form, and the field beats the counts and the geometric descriptors on 1732 hops in 472 oxides | `results/hops/`, `cmfqsar/hops.py`, `vasp.py`, `examples/hops/` | Chem. Mater., J. Mater. Chem. A or npj Comput. Mater. | Labels are bond-valence site energies; needs a DFT-labelled subset |
| 18 | **The fourth dimension, measured.** Conformer ensembles, protonation ensembles, a trajectory in the crystal site and the water as a field: none beats one well-posed conformation, and where the pose is unknown the ensemble is worse | README "The verdict on the fourth dimension", `results/protonation/`, `md/`, `solvent/`, `trajectory/`, `examples/md/`, the Hammett ensemble rows | J. Comput.-Aided Mol. Des. or J. Chem. Inf. Model. | Results done |

Short communication, written only if a journal asks for a separate note: increments fitted to
the data, which improve the number they are fitted on and nothing else (`results/increments/`);
otherwise supplementary material of paper 1. Not yet a paper: the CMF-neural integration of the
development plan (stages ND1 to ND3), which has a plan and no measurement; it becomes paper 19
when ND1 and ND2 are measured on COMPAS-1.

### 1. Software paper

The contribution is what is new in the implementation, not the reproduction: closed-form
kernels for every field type including probe fields and ensembles, the invariant, surface,
periodic and local kernels as one framework, the error bar of a co-field, exact Shapley
attribution under a nonlinear kernel, the descriptor block, the checks around a q², the
corrected parameterisation, and a supported interface of nine names whose signatures a test
holds. The reproductions of the 2013, 2015 and Am/Eu numbers are the validation section, and the
hydrogen-counting bug of the original scripts is reported there. Speed belongs here and nowhere
else: the exact shortcuts (3 to 4 times), the compiled kernel (5 to 16 times with the threads),
the low-rank path that fits in the space of the landmarks with the co-field intact (exact to
2e-11 when nothing is approximated, -0.004 in q² at worst with a grid of Gaussian centres), and
screening a library in blocks. The learners that were added and did not replace the ridge
(kernel PLS, the Fisher discriminant, logistic regression, the Laplacian ridge) are listed here as
options; their measurements are papers 3, 7, 8 and 11.

Before submission: item 0; a figure that shows the application on one series with its pocket
(`docs/screenshot.png` is the basis); the data statement as a supplementary table.

### 2. Hammett equation as a CMF model

Unchanged from the 22 October plan. Hammett's equation is CMF on a series with a trivial
alignment; the co-field is the sensitivity map of benzoic-acid ionisation to charge and bulk
around the ring; fragment charges carry the field effect and whole-molecule charges add the
resonance effect. The three additions reviewers will ask for are still open: held-out
substituents and Taft's σ_I and σ_R; one other scaffold (pyridine or naphthalene); a comparison
with published DFT-based σ predictions. The ensemble result is in hand and is one paragraph:
the conformer ensemble changes nothing, and the measurement says why.

The constants themselves are no longer a risk. All 95 σ_m and σ_p values were checked against
Table I of the 1991 review on 31 October 2026 by reading its page images row by row: 92 agree to
the last digit, and three were corrected in the data and the benchmark (CH2OCH3 row 223,
i-C3H7 row 288, N(CF3)2 row 169). One value the paper itself brackets as suspected of being
inaccurate is in the set, the σ_p of O-i-C3H7, and the paper will say so. The 69 hydrophobic
constants π are not in that review at all and are still traced to no page: they are the one
provenance gap left, and the paper either cites the compilation they do come from or drops them.

Papers 10 and 16 give this paper two more readings: the local co-field around an aromatic
carbon is the same construction applied to a shift, and the ring-centred fields of COMPAS-1 are
the same construction applied to a gap. The three cite each other and share no table.

### 3. Alignment, not the kernel

Written from the third part of `docs/benchmark.md`, with the results added since: the
evidence chooses α and λ as well as cross-validation at a quarter of the cost (+0.007 in
R²pred, interval containing zero, `results/evidence/`); one width per field and a posterior over
the field weights both gain nothing (`perfield/`, `mixing/`); six pharmacophore features beat
twenty Sybyl indicators below forty molecules and lose above (`pharmfields*/`); kernel PLS, the
learner of CoMFA, ties the ridge in cross-validation (q² 0.62 against 0.63 over eight sets) and
is behind it on the published test sets (R²pred 0.36 against 0.44, better on none of the eight;
`kpls/`). The claim is broader than in October:
on an aligned series the kernel, its layer, its widths, its weights and its learner all move q²
by less than the bootstrap interval, and the superposition moves it by 0.1 to 0.7.

Kernel canonical correlation (`kcca/`) is the 2D baseline reviewers will ask for, with its
reason. On the eight Sutherland sets the fingerprint contains most of the field: the part of the
field the fingerprint does not contain predicts the activity at q² +0.162, the part it contains
at +0.478 and the whole field at +0.528, so what a 3D model adds over a 2D one on an aligned
congeneric series is small, and the paper says so.

Before submission: one plain 2D baseline on all 86 sets (ECFP ridge or random forest), since the
canonical-correlation result covers eight; Wilcoxon signed-rank beside each paired t-test; a
figure of the paired differences.

### 4. Working without a frame

Paper 3 is about an aligned series; this paper is about a series that has no frame. The
rotation-invariant autocorrelation kernel, the topomer rule, the count baseline, and
`cmf diagnose`, which runs all four routes and reads the paired intervals, checked on thirteen
sets whose answer is known (`results/diagnose/`). The pose rule of the topomer costs 0.066 q²
on average and the cut 0.035; where the alignment is automatic the verdict reverses. Handedness
is the honest limit: the signed three-point channels cannot carry it between molecules, and a
stereo-centre descriptor can (`results/chirality/`); the five real sets cannot separate the two
because none holds an enantiomeric pair, so the synthetic experiment is reported as such.

The corrosion series appear here for the frame question only (`results/corrosion/`: aligned,
topomer, invariant, counts); the cyclodextrin tables for the one line that the invariant kernel
wins on six of seven hosts, with the rest in paper 13. The three-distance autocorrelation of
paper 10 is a local kernel and stays there.

### 5. Quantum-chemical fields in CMF

The claim is sharper than in October, and no longer only negative. A GFN2-xTB stage costs 0.25 s
a molecule against 360 s for ORCA, which is the first time a quantum field could be tried on every
series in the repository: of 52 series, 7 are helped beyond the interval, the best model being
the five fields plus the Fukui field, and 1 is made worse (`results/xtb/`). Kernel canonical
correlation says why the share is small: the directions of the reactivity fields that shape and
charge do not already contain predict the activity at q² -0.058 against +0.181 for the directions
they do contain (`kcca_quantum/`). The earlier results stand as the measurements behind that:
Hammett (paper 2) as the positive control, where resonance needs whole-molecule charges; the
corrosion series with ORCA, where Fukui fields, Hirshfeld electrostatics and descriptors add
nothing; a Fukui function that cannot be fitted by isotropic Gaussians (relative error 0.8 to 0.9),
which a direction on every atom fixes for the fit (0.6) and not for the model (-0.004); the Am/Eu
quantum stage, which reproduces the classical model to the third decimal.

Before submission: bootstrap intervals on every corrosion comparison; the seven helped series
named and their co-fields read, since a reader will ask what they have in common.

### 6. Corrosion inhibitors: what the literature data can support

A critical paper rather than an application, after the benzimidazole node was taken apart
(`examples/corrosion_fe110/README.md`).  Its spine is the per-experiment database (1600
measurements, 62 articles) behind one scaffold node: a compound's potency agrees between
laboratories at r 0.64, so a perfect model of the pooled medians could reach q² 0.6, yet no
model of any kind passes 0.2 on the 73 compounds, and 0.4 is the most that size and
lipophilicity descriptors with a random forest reach.  The reason is measured: a mixed model
with compound and paper effects (paper offsets of sd 1.05 against a within-paper residual of
0.77) does not make the pooled target more predictable, the 39 compounds that link the
laboratories are unpredictable in every frame, and centred potencies with whole papers held
out show that the structure-potency trend of one series does not transfer to another.  Within
one laboratory's series the picture is the conventional one: CMF on the 18 compounds of Hu
2011 reaches q² 0.66, the frontier-orbital descriptors the corrosion literature favours 0.65
to 0.80, level with the fields rather than beaten by them, and ΔN alone correlates at 0.9.
The computed adsorption energy on Fe(110) (paper 14's tier) predicts nothing of inhibition:
r 0.14 with the potency, no gain as a descriptor or as a second task (the two-endpoint fit
chooses ρ = 0 in every fold), Spearman 0.07 for the ranking of analogues within a series.

Kept from the earlier plan: scaffold-tree series as the way to define congeneric sets from a
2D database, and the protonation state at pH 0 as one state (the pKa-weighted ensemble is
worse, +0.261 against +0.324, because only 1% of the molecules are mixtures and
independent-site pKa picks the state badly).  Dropped: the claim that fields beat the quantum
descriptors; where the earlier series showed it, the comparison must be redone with the
descriptor baselines of `descriptor_baselines.py` (gas-phase indices, ridge, kernel ridge and
a forest, same folds) before it is stated.

Added on 31 October: the reproducibility analysis on five nodes at once and on the 35 compounds
measured in 1 M HCl by three or more papers (`potency_reliability.py`, its CSV outputs beside
it). The shared compounds of the new nodes are mostly one group publishing twice, and
same-author pairs agree to 0.63 units; between different first authors the spread is 1.17 on
the benzimidazole node, 1.19 over five nodes and 1.19 across chemotypes, so one paper's value
has a reliability of about 0.6 within a node and 0.7 across chemotypes. The ceiling of q² 0.6
for pooled node medians is therefore general, which is the sentence the paper's title can carry.

Before submission: one single-protocol series beyond Hu 2011 with ten or more compounds, if the
database has one.

Papers 4, 5, 6 and 14 all use corrosion data. The rule: paper 4 takes the frame columns, paper 5
the quantum columns, paper 6 the chemistry, paper 14 the Fe(110) adsorption energies.

### 7. How much of a co-field is real, and what it can be read for

The measurements of October: the closed-form standard deviation of a co-field masks 70% of what
would have been drawn and raises the sign stability of what remains from 0.906 to 0.978
(`cofield_error/`); the pharmacophore read out of a masked co-field separates the active quarter
at AUC 0.665 against 0.495 for the same features in random places (`query/`); the cosine of two
co-fields separates two models of one SAR from two models of two SARs by +0.22 and tracks how
well one orders the other's molecules (`transfer/`).

Three readings added since, two of them positive. The sum of the co-field over a substituent's
atoms is a bioisostere table: on matched molecular pairs it ranks the measured activity
difference better than both the fingerprint distance and the field distance on two of three sets
and worse on none, and the paper reports how much of that is memory (the whole model reaches
+0.63 to +0.71, a model refitted without the pair +0.28 to +0.56; `bioisostere/`). A co-field
multiplied by a soft mask on the atoms that carry the prediction recognises active molecules of
a scaffold the model never saw (AUC 0.729 against 0.566 for the unmasked similarity, against the
weak actives of those scaffolds), and loses against decoys from other targets because the mask
discards size (`maskedmatch/`). A matched pair read as a difference of fields does not carry
its effect to a series the model never saw (q² -0.025, with or without a frame;
`transformfield/`), which stands beside the increment result and says what a co-field is: a
property of one series in one pocket. The Fisher discriminant is here as the classifier that
has a co-field (`kfda/`, its co-field against the support-vector classifier's); its accuracy is
paper 11's.

Shapley attribution, the field difference and pocket complementarity are the tools, shown on
the three worked series, where one of the three fails its own y-randomisation and is kept for
that reason.

Before submission: a medicinal chemist co-author to judge the three cases; this is the one paper
in the first three tiers that cannot be finished alone.

### 8. How certain is one prediction

Conformal intervals hold their nominal coverage at 80, 90 and 95% on the Sutherland sets, and
the Gaussian-process interval under-covers at every level (`conformal/`). Of four error
estimates, none orders the errors well (risk-coverage efficiency 0.18 to 0.20) and the GP
variance is marginally best (`errorpred/`). Ranking by a lower confidence bound improves
BEDROC over the point estimate (`bound/`). Fitting the order instead of the values gains nothing
on a clean set and a little on sets pooled from sources with offsets (`ranking/`). Added since:
kernel logistic regression ties the Gaussian-process classifier on accuracy, and on calibration
no classifier beats the base rate, which the reliability table explains (`klr/`); molecules
without an activity, added through the Laplacian penalty, gain -0.023 in R²pred where the hidden
activities themselves are worth +0.118, and even the best weight chosen on the test set gains
nothing (`lapkrr/`). The claim is modest and reliable: the program says how often it is right,
says it in a way that was checked, and does not pretend that unlabelled molecules are data.

Paper 12 uses the same posterior for acquisition and is the active-learning side of this one;
the two cite each other and share no table.

### 9. Several endpoints on one kernel: the selectivity co-field

The most distinctive paper of the set after paper 2, and the one with the longest history: the
selectivity field was proposed in 2003 on 52 AMPA/glycine antagonists with CoMFA, and the data
set is rebuilt here (`examples/selectivity/`). With five fields the direct model of Δ reaches
q² +0.53 against +0.46 for the difference of two models, and the hydrophobic and hydrogen-bond
fields are what the selectivity needed. The method is the multi-task fit on one kernel with a
task covariance, which gives the selectivity as a difference of co-fields rather than a ratio.
Three regimes, measured: two endpoints with their own molecules (corrosion, cyclodextrin pairs),
where sharing gains nothing the intervals can see; seven endpoints of which five are small, where
sharing gains +0.076 mean q² and a free-form covariance a further +0.034 (`tasks/`); and a host
removed entirely and predicted from the other six through a descriptor covariance.

Added since: the case a medicinal chemist asks for, a potency with an anti-target on the same
compounds from one laboratory (78 histamine H3 ligands with their hERG activity,
`examples/antitarget/`). Tying the anti-target to the potency changes the potency by +0.022 with
an interval that contains zero, so the selectivity co-field costs nothing, and at the basic
nitrogen the difference co-field is of one sign in the steric and hydrophobic fields
(`antitarget/`). The two-objective design loop on the same series is the honest counterpart: run
backwards on the part the authors made later, its Pareto front contains none of their compounds
and ranks them in the worse half (`designloop/`). Reading where two endpoints part is what the
model can do; choosing the next compound from two objectives is not yet.

The 2003 paper is the natural frame for a J. Med. Chem. perspective; otherwise J. Chem. Inf.
Model. The two-cavity selectivity, which follows the size of the guest, is this paper's; the
single-host tables are paper 13's.

### 10. A target that belongs to an atom

The field around a carbon under a Gaussian envelope, turned into the invariant autocorrelation,
predicts 13C shifts at q² +0.97 and 8.8 ppm against +0.85 and 20.5 ppm for the atom-centred
count, on 5096 carbons of 400 molecules with folds that hold out whole molecules (`local/`).
Against the representation the materials community uses, SOAP with one channel per element
reaches 5.7 ppm on the same carbons, and the three-distance autocorrelation with the fields as
channels reaches 5.5 (`soap_gate/`, `triangles/`): a field kernel with angular information is
level with SOAP and keeps the field types as its channels. The error is flat from about 800
molecules (10.8 ppm at 100 molecules, 9.8 at 6000 for the two-point kernel), so the model is
limited by its representation and not by data, and averaging over conformers costs rather than
gains. The same construction gains nothing on 912 carboxylic acid pKa values, and the difference
field of the acid and its anion is worse, because every acid loses its proton the same way. A
frame around one aromatic carbon, fixed by a rule, allows the overlap kernel and a co-field: q²
+0.90 against +0.77, and with one indicator field per atom type the co-field becomes a table of
what an atom of each type at each ring position adds to the shift (`local_cofield/`). On the
site of metabolism (636 molecules, 1301 sites) the local field finds a true site among the top
two in 71.1% of the molecules against 70.9% for the two-dimensional count, and counting the
same atom types over distance reaches 73.9%: the field adds beside the count, not instead of it
(`metabolism/`).

Before submission: one published shift predictor on the same molecules (the HOSE-code method
of nmrshiftdb2 or a graph network) so the 5.5 ppm has a reference point; SOAP is a
representation, not a shift predictor, and does not settle this.

### 11. Virtual screening with measured inactives

The successor of the two 2011 one-class papers, and the least favourable result in the
repository, reported as such. On ESR1 antagonists and p53 from LIT-PCBA, posed by one rule that
knows nothing about the class, the best one-class field description reaches AUC 0.544 against
0.685 for the ECFP4 Tanimoto to the same actives; some descriptions are below chance on p53,
which is what the AVE debiasing of that benchmark does to resemblance methods. What survives is
the kernel used discriminatively: with the inactives treated as unlabelled (bagging PU) it reaches
0.689, level with the fingerprint and +0.110 above the same learner told the inactives were
negative. The posing rule is half of the measurement: a generous placement is worth +0.141 AUC
to the field descriptions and +0.000 to the fingerprint. Data fusion over the similarities to
the individual actives (`fusion/`) beats every fitted one-class description and belongs here.
Added since: the support-vector classifier, the Gaussian-process classifier, the Fisher
discriminant and logistic regression tie on these sets, every AUC difference containing zero
(`kfda/`, `klr/`), and the Laplacian ridge loses no enrichment (`lapkrr/`); the learner is not
where the answer is.

Before submission: three more LIT-PCBA targets, since two decide nothing on their own.

### 12. Which molecule to make next

On 80 runs over the Sutherland sets, the confidence bound and expected improvement find a hit in
three to four acquisitions where random takes six, and the variance rule improves the model where
the greedy rule degrades it (`active/`): finding an active and improving the model are different
tasks and need different rules. A plate of ten chosen at once needs only the cheap similarity
filter: the posterior-conditioned rule is the same within the interval, and the batch loses
nothing measurable against one molecule at a time (`batch/`). The substituent design loop,
leave-one-group-out on two sets, ranks held-out groups at Spearman +0.40 and its first
suggestion is 1.4 log units above the library mean (`design/`); on a table with no scaffold the
invariant kernel puts adamantyl first of 11 with every adamantyl guest removed, and a held-out
group is pulled towards the mean (`guest_design/`).

Three questions the model cannot answer, each with the gate it failed and what did pass it. What
to grow a fragment into: the true group has a median rank of 25 in a library of 201, where
refusing the placements that clash and shuffling the rest gives 15, so the field score is worse
than not scoring (`fraggrow/`). Whether a ring closure will pay: the share of intermittent field
it removes correlates with the potency gain at +0.04, the count of rotatable bonds removed at
+0.24, and the activity-relevant field the restricted analogue still places at +0.22
(`restriction/`). Which transformation to copy from another series: none (paper 7). The paper
is the one place the positive and the negative answers stand together.

### 13. Host-guest binding without a frame

An application paper for a supramolecular readership. Seven cyclodextrin tables, 1669 guests
sharing no scaffold: the invariant kernel beats the best superposition the program can make on
six of seven hosts and the atom-type counts on all (`cyclodextrin/`); its per-atom split names
the three cavity groups positive and the three polar groups negative on beta-cyclodextrin and
not on alpha or gamma (`guest_atoms/`); and a one-body complementarity score with the cavity
correlates at -0.35 with log K where the guest's size alone gives +0.35, because the score has
no term for the water that leaves the cavity (`host_pocket/`). The last is the result a
supramolecular reader will recognise, and the paper is built around it.

### 14. Continuous molecular fields of crystals and surfaces

The periodic autocorrelation kernel on Matbench phonons gives 92 to 97 cm⁻¹ against 324 for the
dummy predictor, 68 for a random forest and 29 for the best graph network, and three findings
worth more than the number: the inverse square root of the atomic mass is the field the
spectrum asks for, the taper beats a hard cutoff, and the background subtraction argued for on
physical grounds costs accuracy. The in-plane invariant kernel keeps the height of an adsorbate
and discards its lateral position, and the claim it supports is bounded on three sides
(`examples/corrosion_fe110/README.md`): on 72 benzimidazoles relaxed on Fe(110) with UMA it
reaches q² 0.66 where the overlap kernel reaches 0.52 with or without a lateral registration;
on 63 quinolines it is level with the unregistered overlap kernel (0.52 to 0.58 against 0.56)
and above the registered one (0.47); on OC20's DFT energies of small fragments on copper it is
level with the other geometric representations (0.76 against 0.71 for the overlap and 0.82 for
the fully invariant kernel) and far below the identity of the fragment (0.92).  What holds
every time: shuffling the poses in the plane changes nothing, scrambled targets give nothing,
and the fully invariant kernel, which throws the height away, loses the signal on both nodes
(0.11 and 0.02).  The model read in its own channels (`surface_channels_eads.png`) says why:
density 2 to 3 Å above the metal strengthens adsorption in every field pair and density at 4
to 6 Å weakens it.  "Alignment-free at no cost, with the height as the carrier" is the
statement; "better" is not.

Added since: the molecule posed on the surface by rule, without the slab (`planealign.py`,
`results/fe110/`). The number to quote for "can CMF predict a machine-learned adsorption energy
from the structure alone" is about 0.3 on the benzimidazole node and 0 on the quinoline node;
the rest of 0.5 to 0.66 was the relaxation telling the model which molecules it had bound. The
rule aligner did not pass the gate the plan set it and stays a display tool. The same README
holds the physics of the tier: the OC20 head of UMA makes the ideal Fe(110) surface a saddle
point (the slab is kept rigid), has no dispersion (the poses are docked before they are
relaxed), and its energies are learnable at q² 0.5 by the overlap kernel in the surface frame
but say nothing about inhibition (paper 6).

Added since: the **periodic overlap kernel** of item M4 and the co-field it gives inside the
cell, which is the one thing this tier could not do before -- an autocorrelation is a distance
distribution and cannot say *where*. Measured on 1,500 cells of `matbench_perovskites` (a second
Matbench task, so the formation-energy row of the list below is partly answered): R² +0.479 on
one lattice against +0.497 for the autocorrelation, +0.350 in the cells as they come. The frame
is the whole finding -- it is worth +0.129 to the overlap kernel, nothing to the invariant one
and -0.069 to the same overlap without the images -- and the kernel does not beat the
autocorrelation, because giving 18,928 different compounds a common lattice takes the lattice
constant out of the kernel. **The section this supports is "the co-field in the cell, and what a
frame costs a set that has none"; "better" is again not the claim.** The MAE is 0.38 eV per
five-atom cell against about 0.03 for the leaderboard, on 8 % of the set and four fields, and
that comparison has to be stated rather than avoided.

Before submission: two more Matbench tasks (formation energy and band gap are the natural ones)
so the kernel is placed on the leaderboard rather than on one task, and the perovskite task on
the **whole** 18,928 cells rather than 1,500, since the present number cannot be compared with a
leaderboard; a DFT anchor on the benzimidazole structures themselves (VASP single points on
eight to ten UMA geometries), since OC20's fragments could not supply one; a co-author from
materials science for the reading of the co-patterns.

**Item M14 is done (1 November 2026) on a public set**, and it is what makes the periodic
overlap kernel a paper of its own rather than a section of this one: 2DMD, one to three point
defects on the 8 x 8 supercell of MoS2 and of WSe2, 5,933 DFT-labelled cells per material on
one lattice, where the frame is free and nothing has to be rescaled to get it. With the kernel
averaged over the 768 operations of the host the arrangement kernel reaches 1.8 and 5.2 meV per
defect site on 2,000 cells against 8.3 and 15.7 for the periodic autocorrelation, 41 for the
count kernel and 60 for the fixed frame, and 1.9 and 4.4 against 3.0 and 6.0 on all 5,933 cells
(the paired intervals exclude zero at both sizes; the lead is a factor of three to four at
2,000 cells and 1.0 and 1.5 meV at 5,933); it extrapolates
from one and two defects to three where the autocorrelation does not; and its per-defect
contributions are the single-defect energies of the set to 0.2 eV. The published errors on this
set are 2.6 and 4.8 meV for a sparse graph network and 11.8 and 14.7 for gradient boosting on
descriptors, on the authors' own split. **The section this supports is "a congeneric series in
the crystal sense": the symmetry average makes a count kernel of the linear overlap and an
arrangement kernel of the layer, and the comparison with the graph networks has to be made under
one split before "better" is claimed.** The beta-Li3PO4 vacancy series remains the case the
conductivity literature argues about, and can be a case study of the same paper at the measured
cost of two nights of one workstation; the co-field of a vacancy energy read on the oxygens that
coordinate it is still the figure the grant paragraph of the Li3PO4/Li3BP2O8/LiBO2 study needs.

**Item M5 is done (1 November 2026)**, and it gives this tier the reading it was missing:
which **site** of the cell carries the property. The autocorrelation of a cell is a sum over
ordered pairs, so it splits exactly into one row per site; a linear model on the whole cell
therefore splits into one number per site with nothing attributed, which is rule 3 for an
invariant kernel. On the single-atom catalysts of item EC8 a model of the OH binding energy
(q² +0.893) puts 8.9 % of its weight on the metal site, 11.3 times an even share of the 127
sites of the sheet, and the channels at that site name the d electron count of the metal and the
electrostatic field in the first shell: a figure for the interpretation section that no
autocorrelation could draw. The same split makes a site a **sample**, and on those catalysts the
site profile taken inside the periodic cell, with nothing cut out of it, is level with the cut
cluster of the pilot and with the whole cell (mean q² over the four endpoints +0.792, +0.721 and
+0.755 in the three fold designs, against +0.801 / +0.739 / +0.712 for the invariant kernel of
the cut cluster, +0.782 / +0.695 / +0.724 for its overlap kernel and +0.794 / +0.756 / +0.744
for the whole cell; of the twelve paired comparisons one is clear of zero in its favour against
the whole cell and two against it). **The section this supports is "reading a crystal model site
by site"**, and what the site model itself supports is "the cut is not needed", not "better".
The per-site model is not a claim of its own yet: it needs a set with many measured sites in
each of many cells, which neither the catalysts (one site per cell) nor the three lithium
electrolytes (three cells) can be.

**Item M6 is done (1 November 2026)**, and it gives the theory section of this paper the piece
it was missing. The periodic autocorrelation and the typed structure factors are one object: the
real-space lattice sum of a Gaussian and the sum over the reciprocal lattice are equal by Poisson
summation, which `parseval_residual` checks to one part in 1e13 on 120 crystals of three Matbench
tasks. **The section this supports is "one descriptor, two resolutions"**: the width of the
Gaussian field is a resolution limit in reciprocal space, exp(-a² q²), and the measurement shows
it behaving as one -- accuracy falls steadily as the field widens (R² +0.785 to +0.587) and
changes by at most 0.002 when the sum over G is taken half as far again. What the channels buy
for prediction is smaller and should be stated as it is: alone they are level with the best
real-space cutoff on all three tasks, with no neighbour list; added to it they gain +0.066
[+0.045, +0.090] on the formation energy of the perovskites, +0.015 [+0.002, +0.027] on the shear
modulus and nothing on the bulk modulus. The result for the methods section is the third one:
taking the real-space cutoff from 4 to 10 Å *loses* about 0.10 of R² on both moduli, because a
longer sum adds many weakly informative distance channels, while the reciprocal channels give it
back without growing in number. That is the honest case for reciprocal space in this program --
not more accuracy, but long-range order at a fixed cost -- and it is also what makes the
scattering item M12 worth starting.

**Item M12 is done (1 November 2026)**, and what it adds to this paper is a section and a
warning, not a result on measured data. The section is **"the measured curve is the untyped
descriptor"**: the reduced pair distribution function G(r) is the one-field periodic
autocorrelation of the same program, with the uniform background kept and the self term dropped,
divided by r, by the mean scattering length squared and by the integral of the channel's
Gaussian, and the background term of the descriptor is exactly the -4 pi r rho_0 that the
standard definition subtracts. The difference against a direct sum is 8.2e-05 of the curve at a
0.05 Å channel and falls with the channel width. That makes a measurement a sample and gives the
program a route to amorphous and nanocrystalline materials, which is worth stating on its own.
The cost is large and must be stated with it: on the three Matbench tasks the one curve an
experiment can measure reaches R² +0.303 and +0.192 on the two elastic moduli against +0.771 and
+0.684 for the typed descriptor, and +0.012 against +0.418 on the perovskite formation energies,
where the curve alone is no better than the mean. The warning is the part a reader will need
most, and it was not expected: a curve as computed, at a channel 0.05 Å wide, carries nothing a
kernel can use (+0.003 on the bulk modulus), while the same curve cut at Q_max = 20 1/A reaches
+0.327 -- the finite resolution of the instrument is what makes two curves comparable -- and a
model fitted on computed curves cannot predict measured ones at all (transfer R² below +0.06 in
all 36 settings tried). Anyone who fits a model to measured curves must compute the training
curves at the resolution of the measurement. **No experimental data is used**, and the paper must
say so: there is no public set of measured curves with measured properties, so this section
reports how good a measurement would have to be, and finding or making that set is the next step
of the item.

**Item M10 is done (2 November 2026)**, and it gives this paper the result that the periodic
sections were missing: a case where the standard description of a crystal does not merely lose
accuracy but fails, and a cheap block that repairs it. Every descriptor of the program so far
counts field weight at a distance from field weight, so two structures with the same multiset of
distances are one structure for it, and a mirror image is indistinguishable from its original.
The three-body block resolves the triangle around every atom -- two distances to a pair of
neighbours and the distance between them -- and the handedness block is the determinant of three
field-weighted vector sums about each centre, which changes sign under a reflection. **The
section this supports is "what a histogram of distances cannot see"**, and its measurement is on
the defect series of M14: trained on the cells with one or two defects and asked about the cells
with three, the two-point description is no better than predicting the mean (R² -0.015, 1,222 meV
per site on MoS2; +0.060 and 1,096 on WSe2) and the three-body channels hold (+0.940 and 240 meV
per site; +0.954 and 198). On ordinary property tasks the gain must be stated as small: +0.089
[+0.041, +0.140] on the perovskite formation energies, where the cells differ by a tilt angle,
and nothing measurable on the two elastic moduli. Two side results belong in the methods section.
The plan expected the handedness block to cost a sum over triples of atoms; with the centre atom
as the origin the factorisation survives the lattice sum, so both blocks cost one pass over the
pairs of neighbours. And the mixed kernel, which fails on the extrapolation split although its
three-body part alone succeeds, is a clean example of a general trap: a setting chosen on a
random inner split cannot see the failure an extrapolating test asks about. **No chiral crystal
set is used**, so the handedness block is reported as tested and not as measured.

### 15. Review

Year 2, after the first four tiers are in print: the CMF series from the 2009 Doklady paper to
the present, organised by what the measurements settled (the alignment decides; the kernel,
the learner, the hyper-parameters and the fourth dimension do not; the co-field has an error
bar and a selectivity; the same kernel reaches atoms, rings, hops, surfaces and crystals).

### 16. The gap of a benzenoid in a frame given by rule

A pilot written for a collaborating group (`results/compas/REPORT.md`), and complete as a
study. The HOMO-LUMO gap of the 8,678 cata-condensed polybenzenoid hydrocarbons of COMPAS-1D,
every molecule in the frame of its own inertia tensor with the signs of the axes fixed by a
rule, so nothing is aligned. The steric field alone reaches a test R² of 0.42, the carbon-class
indicators 0.49, four ring-centred fields (one Gaussian per ring centroid per annulation class)
0.65, the four groups together 0.78, and a ring field weighted by the Clar-sextet share takes
that to 0.87 (MAE 0.11 eV). The result the paper rests on is the size extrapolation: trained on
fewer than N rings and tested on N, every representation without a ring field has negative R²
and every one with it is positive (0.44 to 0.78), where a random forest on descriptors, 0.91 on
random splits, is negative. The co-field over a six-ring reference and the per-ring
contributions are the reading. The follow-ups of 30 October are in the table: the Gaussian layer
gains as much on the extrapolation splits, the resolution was the nonlinearity and not the width,
2,000 landmarks give the same model as 500, and a HOMA ring field stands in for NICS.

What must be stated, since a reader will know it: the equivariant networks of Weiss et al.
reach 0.011 to 0.015 eV on random splits of the same molecules, ten times better. The claim is
not accuracy; it is that a linear model in closed form with ten fields extrapolates in size
where nothing published has been measured to, and can be read ring by ring.

Before submission: the open items of the pilot in order (a skeleton frame, a kernel invariant
to the flip of the plane, a NICS field from the ORCA stage); the collaborating group's reading
of the co-fields. Co-authorship with that group is the natural form.

### 17. The barrier of a lithium hop from the field around the path

The hop is the sample: the environment is cut from the periodic cell about the midpoint of two
lithium sites with the migrating ion removed, the frame is the hop itself, and the rotation about
it is averaged in closed form (the overlap kernel with the cosine replaced by a modified Bessel
function), which is what 96 finite poses approximate at the cost of 96 squared. On 1732
symmetry-distinct hops in 472 ternary lithium oxides of the Materials Project, with folds that
hold out whole structures, the rotation-averaged overlap reaches q² +0.65 and 0.33 eV mean
absolute error against +0.49 for the counts about the midpoint and +0.10 for the hop length and
bottleneck; the difference over the counts is +0.16 with an interval of +0.10 to +0.25
(`hops/`). Partial occupancy enters as the exact mean field, which lets the conductor families
with disordered sites into the set (plan item M2). The co-field along the hop is drawn per field.

The limitation that decides the venue: the labels are bond-valence site-energy barriers, not
DFT. As it stands this is a method paper on a cheap label; it becomes a materials paper when a
subset carries nudged-elastic-band barriers (UMA is the cheap route to them, VASP the anchor).

Before submission: that DFT-labelled subset; the polarisability field the plan names; a
co-author from solid-state ionics. If time runs short the hop table joins paper 14 as its third
case.

### 18. The fourth dimension, measured

Everything the program can average over, against one well-posed conformation, with the bootstrap
interval of rule 4 on every difference. Conformer ensembles on five routes of placement, none
better than the single conformation on an aligned series (README, "The verdict on the fourth
dimension"); protonation ensembles at pH 0, worse than one state (`protonation/`); the Hammett
ensembles, which change nothing and say why; a short simulation of each ligand in its crystal
site read nine ways, where the reading fixed in advance gains nothing, two of three sets disagree
with the third, and the own-frame control corrects what one set seemed to say (`md/`); the water
of that simulation as a field at the hydration sites, which helps at none of sixteen definitions
of a site, and the pre-registered tests of the definition that failed on a second and a third set
(`solvent/`); and the case a user actually has, a pose that is not known, where the ensemble is
worse and not better. The reader for trajectories and what thinning the frames costs
(`trajectory/`) are the methods section.

The methodological contribution is the design: a reading fixed before the measurement, a
control that destroys the frame, and a definition sweep that reports the range rather than one
point. The positive control is the crystal pose itself, which every ensemble is measured against.

### 19. Not yet: CMF as a layer of a neural model

The development plan's ND1 to ND3 (the kernel rows as a featuriser, trainable widths and mixing
weights with the co-field kept exportable, field values as node attributes of an equivariant
network) have a plan and no measurement. The paper exists when ND1 and ND2 are measured on
COMPAS-1 against paper 16's model and the published networks; until then it is not counted.

## Sequence

| Tier | When | Papers | Why these first |
|---|---|---|---|
| 0 | Month 1 | Release (item 0) | Every paper cites the DOI |
| 1 | Months 1 to 3 | 1, 2, 9 | 1 and 2 to ChemRxiv in the same week so each cites the other; 9 is the most distinctive result, continues the 2003 paper and now carries the anti-target case |
| 2 | Months 3 to 6 | 3, 4, 10 | 3 and 4 are one argument in two halves; 10 is complete, level with SOAP, and only needs its reference predictor |
| 3 | Months 6 to 9 | 7, 16, 8 | The co-field read four ways; the COMPAS pilot, which a collaborating group is waiting for; the posterior |
| 4 | Months 9 to 12 | 5, 6, 12 | The quantum stage with its share of helped series; the corrosion data paper, now on five nodes; the design questions with their three negatives |
| 5 | Months 12 to 15 | 13, 18, 11 | Two complete papers that fold into others if needed, and the screening paper once it has five targets |
| 6 | Months 15 to 18 | 14, 17 | Both need data from outside the repository (Matbench tasks, a DFT anchor, NEB barriers) and a co-author outside chemoinformatics |
| 7 | Year 2 | 15, then 19 if ND1 and ND2 are measured | After tiers 1 to 4 are in print |

Eighteen papers in eighteen months is not realistic for one author, and the plan does not
assume it is. The tiers are an order of priority, and rule 5 limits the work in progress to
three. If time runs out, the papers that fold into others are 12 (into 8), 13 (into 4 and 9), 5
(into 6), 17 (into 14), 18 (a short communication) and 11 (a short communication). Papers 1, 2,
3, 4, 9, 10 and 16 do not fold into anything.

## What each paper still needs, in one table

| # | Open item | Effort |
|---|---|---|
| 0 | Public release, Zenodo DOI, PyPI, a reproduction run on another machine | days |
| 1 | Figure of the application; nothing else | days |
| 2 | Taft constants, one more scaffold, published DFT σ predictions as the reference, a source for the π values | one to two weeks of runs |
| 3 | Plain 2D baseline on 86 sets, Wilcoxon tests, paired-difference figure | days |
| 4 | Nothing; the enantiomer limitation is stated | none |
| 5 | Bootstrap intervals on the corrosion comparisons; the seven helped series read | days |
| 6 | One single-protocol series beyond Hu 2011 with ten or more compounds, if the database has one | days |
| 7 | A medicinal chemist's judgement of three cases | a co-author |
| 8 | Nothing | none |
| 9 | Nothing; the 2003 set and the H3/hERG set are in place | none |
| 10 | One published shift predictor on the same molecules | one week |
| 11 | Three more LIT-PCBA targets | one to two weeks of runs |
| 12 | Nothing | none |
| 13 | Nothing | none |
| 14 | Two Matbench tasks, a VASP anchor on the benzimidazole structures, a materials co-author | weeks, and a co-author |
| 16 | Skeleton frame, plane-flip kernel, NICS field; the collaborating group's reading | one to two weeks, and the group |
| 17 | A DFT-labelled subset of hops (NEB), the polarisability field, a co-author in solid-state ionics | weeks, and a co-author |
| 18 | Nothing | none |

## Risks

* **Overlap.** Eighteen papers from one repository will look like one paper cut into pieces
  unless each has a claim of its own. Rule 2 is the control, and the assignment of tables in rule
  1; a reviewer who sees the same table twice is right to object. The new assignments are:
  the H3/hERG table to paper 9, the three design negatives to paper 12, the bioisostere and
  masked-match tables to paper 7, the metabolism table to paper 10, the classifier accuracies to
  paper 11 and their calibration to paper 8.
* **Novelty of paper 1.** Eight prior publications on the method. The paper opens with what is
  new in the implementation and treats the reproductions as validation.
* **Negative results.** Papers 5, 8, 11, 12 and 18 are mostly negative. Each carries its positive
  control (Hammett and the seven helped series; conformal coverage; the PU result; the acquisition
  rules; the crystal pose) and its intervals, and says what the data can and cannot resolve. A
  gate set before the measurement and not met is reported as a gate, with the baseline that did
  pass it.
* **Accuracy against the networks.** Paper 16 is ten times less accurate than the published
  equivariant networks on random splits, and paper 14 sits in the lower middle of the Matbench
  leaderboard. Both papers claim extrapolation, reading and cost, not accuracy, and say so in the
  abstract.
* **Cheap labels.** Paper 17's barriers are bond-valence site energies. It is not sent as a
  materials paper until a DFT-labelled subset exists.
* **Small series.** The corrosion sets of 20 to 30 molecules cannot separate differences below
  about 0.1 in q²; every comparison on them is reported with a bootstrap interval. The three
  worked cases of paper 7 are 22 to 33 molecules, and one of them fails its own check, which the
  paper reports.
* **Two targets, one task.** Paper 11 rests on two LIT-PCBA targets and paper 14 on one Matbench
  task. Neither is submitted until the count is raised; both are in the last tiers for that
  reason.
* **Co-authors.** Papers 7, 14, 16 and 17 need a judgement the author cannot supply alone (a
  medicinal chemist, a materials scientist, the COMPAS group, a solid-state ionics group). All
  but 16 are placed after the papers that need nobody else; 16 is placed early because the
  group is waiting.
* **The release.** Everything cites the DOI. Item 0 is a week of work and is the first thing
  done, not the last.
