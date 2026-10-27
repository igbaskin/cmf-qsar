# Publication plan for the CMF implementation

*Drafted 20 October 2026, revised 22 October 2026, rewritten 27 October 2026 after the results of
the development plan's parts B to I: the alignment-free, surface, periodic and local kernels, the
multi-endpoint models, the uncertainty and screening measurements, the host-guest and per-atom
sets, and the 2003 selectivity set. The plan names what each paper takes from the repository,
what it still needs, which journal it is aimed at, and in which order the papers are written.
File references point at the material a paper is written from. The companion document,
`docs/development_plan.md`, says what gets built; this one says what gets written.*

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
* The software side of paper 1 is in place: `pip install`, 974 tests, continuous integration
  on three operating systems (`.github/workflows/ci.yml`), a lock file, a data statement
  (`docs/public_3dqsar_datasets.md`), four tutorials that run in Colab, an HTML API reference,
  and one entry point that rebuilds every table (`benchmarks/run_all.py --manifest`).

What is still not in hand is unchanged: a public release with a DOI, and a medicinal chemist's
judgement of the case studies.

## Rules the plan follows

1. **A table appears in one paper only.** Several papers draw on the same series (the corrosion
   sets, the cyclodextrin tables, the Sutherland sets). Each paper takes one question from a
   series and cites the others for the rest. The assignment below is the rule.
2. **Each paper has one claim that a reader can state in a sentence.** Results that do not
   serve the claim go to the supplementary material or to another paper.
3. **Negative results are reported with a positive control and an interval.** "Nothing helped"
   is not a paper; "this did not help, that did, and the data can resolve differences down to
   0.1 in q²" is.
4. **The repository is cited by DOI from the first paper on.** No paper is submitted before the
   release of item 0 below, and every table in a paper is produced by a script named in it.
5. **Three papers at a time at most**, since there is one author. The tiers below are the order.

## Item 0: the release

Before any submission: a public repository, a release tag, a Zenodo DOI, the package on PyPI,
the manual (`docs/CMF_manual.pdf`) as supplementary material, and a run of
`benchmarks/run_all.py` on a machine that is not this one, so the claim "the numbers reproduce"
has been tested once. The version is 1.0.0 in `pyproject.toml` and `CITATION.cff`; the release
should carry the commit the manifest records.

## The papers

| # | Paper, with its one-sentence claim | Source material | Venue | Status |
|---|---|---|---|---|
| 1 | **Software paper.** A library, a tool and an application that give every CMF kernel in closed form, with co-fields, their uncertainty, and the checks around a q² | implementation, manual, tutorials, the three reproductions (`docs/benchmark.md` parts 1 and 2), speed and scale (`results/speed/`, `nystrom/`, screening in blocks) | J. Chem. Inf. Model. (application note) or J. Cheminformatics | Ready to write once item 0 is done |
| 2 | **Hammett equation as a CMF model.** Substituent constants are a co-field around the ring, and resonance needs whole-molecule charges | `results/hammett/`, README "A field-based Hammett equation" | J. Phys. Org. Chem., J. Org. Chem. or J. Chem. Inf. Model. | Results done; three additions below |
| 3 | **Alignment, not the kernel, and not the hyper-parameters either.** On 86 series the superposition decides fifteen times as much as the kernel, and the evidence chooses as well as cross-validation | `docs/benchmark.md` part 3, `results/alignment/`, `evidence/`, `perfield/`, `mixing/`, `pharmfields*/` | J. Chem. Inf. Model. or J. Comput.-Aided Mol. Des. | Results done; a 2D baseline still needed |
| 4 | **Working without a frame.** The rotation-invariant autocorrelation kernel, the topomer rule and the diagnosis that says which route a series needs | README "Alignment-free, but geometry-aware", "Topomer fields", `results/topomer*/`, `diagnose/`, `chirality/`, `structure/`, `corrosion/` (the frame columns only) | J. Comput.-Aided Mol. Des. or J. Chem. Inf. Model. | Results done |
| 5 | **Quantum-chemical fields in CMF.** They help where resonance matters and nowhere else that was measured | `results/corrosion_qm/`, `cdft_kernels/`, `anisotropic/`, `am_eu/` (the quantum rows), Hammett cited from paper 2 | J. Chem. Inf. Model. or J. Comput. Chem. | Results done |
| 6 | **Corrosion inhibitors: what the literature data can support.** The pooled potency of a scaffold node is reproducible between laboratories (r 0.64) yet no model reaches a third of that ceiling, because the node is a set of series-specific trends with no shared scale; within one protocol fields and frontier-orbital descriptors are level, and a computed adsorption energy predicts nothing of inhibition; the protonation state is one state, not an ensemble | `examples/corrosion_fe110/` (the potency, mixed-model, leave-paper-out, descriptor-baseline and E_ads rows), `results/corrosion_hcl/`, `protonation/`, `examples/scaffold_tree_export.py` | Corrosion Science or J. Mol. Liq. | Results done |
| 7 | **How much of a co-field is real.** The co-field has a closed-form error bar; masked by it, it is a usable pharmacophore and a map of SAR transfer | `results/cofield_error/`, `query/`, `transfer/`, `cases/`, README "Three series, worked through", Shapley and field-difference sections | J. Chem. Inf. Model. | Results done; needs a chemist's judgement of the cases |
| 8 | **How certain is one prediction.** Conformal intervals keep their coverage, the GP variance does not, and no estimate orders the errors well | `results/conformal/`, `errorpred/`, `bound/`, `ranking/`, README "Chance, domain and uncertainty" | J. Chem. Inf. Model. or J. Cheminformatics | Results done |
| 9 | **Several endpoints on one kernel: the selectivity co-field.** Selectivity is a difference of co-fields; sharing helps only when the endpoints share molecules unevenly; an endpoint nobody measured can be predicted from its neighbours | `results/selectivity/`, `multitask/` (method rows), `two_endpoints/`, `tasks/`, `cmfqsar/multitask.py`; the 2003 J. Med. Chem. set | J. Med. Chem. (perspective) or J. Chem. Inf. Model. | Results done |
| 10 | **A target that belongs to an atom.** The field around a carbon predicts its shift better than its atom-centred count, and a frame around one atom gives a co-field that reads as a substituent table | `results/local/`, `local_cofield/`, `examples/local/`, `cmfqsar/atomtargets.py`, `localenv.py`, `localalign.py` | J. Chem. Inf. Model., Magn. Reson. Chem. or J. Org. Chem. | Results done; needs a published shift predictor as the reference point |
| 11 | **Virtual screening with measured inactives.** One-class field descriptions lose to a fingerprint on LIT-PCBA; the same kernel with the inactives as unlabelled data draws level; the posing rule is half of any such figure | `results/screening/`, `fusion/`, `examples/litpcba/`, `cmfqsar/pu.py`, the one-class and GP-classifier sections of the README | J. Chem. Inf. Model. or J. Cheminformatics | Results done; two targets only |
| 12 | **Which molecule to make next.** A confidence bound finds hits and a variance rule improves the model, and the two are different tasks; the substituent loop ranks held-out groups at ρ +0.4 | `results/active/`, `design/`, `guest_design/`, `cmfqsar/design.py` | J. Chem. Inf. Model. or Mol. Inform. | Results done |
| 13 | **Host-guest binding without a frame.** Seven cyclodextrin tables, where the invariant kernel beats the alignment, names the group that enters the cavity, and a one-body complementarity score points the wrong way | `results/cyclodextrin/`, `guest_atoms/`, `host_pocket/`, `examples/cyclodextrin/`; the seven-host sharing cited from paper 9 | Phys. Chem. Chem. Phys., J. Phys. Chem. B or J. Incl. Phenom. | Results done |
| 14 | **Continuous molecular fields of crystals and surfaces.** The periodic autocorrelation on Matbench phonons, and the in-plane invariant kernel that keeps the height of an adsorbate and nothing else: as accurate as the best aligned model with no alignment, better on one node and level on another and on OC20 | `results/matbench_phonons/`, `cmfqsar/periodic.py`, `examples/corrosion_fe110/surface_kernel_eads*.csv`, `surface_channels_eads.py`, `oc20_anchor.py` | Digital Discovery, J. Chem. Theory Comput. or npj Comput. Mater. | Needs two more Matbench tasks and a DFT anchor on the benzimidazole structures |
| 15 | **Review.** The CMF series from 2009 to the present, the successor of the 2014 Springer chapter | papers 1 to 14 | WIREs Comput. Mol. Sci. or J. Comput.-Aided Mol. Des. (perspective) | Year 2 |

Short communications, written only if a journal asks for a separate note: the fourth dimension
measured (conformational and protonation ensembles, five routes, none better than one chosen
conformation; README "The verdict on the fourth dimension", `results/protonation/`), and
increments fitted to the data, which improve the number they are fitted on and nothing else
(`results/increments/`). Both otherwise go to the supplementary material of papers 1 and 6.

### 1. Software paper

The contribution is what is new in the implementation, not the reproduction: closed-form
kernels for every field type including probe fields and ensembles, the invariant, surface,
periodic and local kernels as one framework, the error bar of a co-field, exact Shapley
attribution under a nonlinear kernel, the descriptor block, the checks around a q², and the
corrected parameterisation. The reproductions of the 2013, 2015 and Am/Eu numbers are the
validation section, and the hydrogen-counting bug of the original scripts is reported there.
Speed belongs here and nowhere else: the exact shortcuts (3 to 4 times), the compiled kernel
(5 to 16 times with the threads), Nyström landmarks, and screening a library in blocks.

Before submission: item 0; a figure that shows the application on one series with its pocket
(`docs/screenshot.png` is the basis); the data statement as a supplementary table.

### 2. Hammett equation as a CMF model

Unchanged from the 22 October plan. Hammett's equation is CMF on a series with a trivial
alignment; the co-field is the sensitivity map of benzoic-acid ionisation to charge and bulk
around the ring; fragment charges carry the field effect and whole-molecule charges add the
resonance effect. The three additions reviewers will ask for are still open: held-out
substituents and Taft's σ_I and σ_R; one other scaffold (pyridine or naphthalene); a comparison
with published DFT-based σ predictions. The ensemble result is now in hand and is one paragraph:
the conformer ensemble changes nothing, and the measurement says why.

Paper 10 gives this paper a second reading: the local co-field around an aromatic carbon is the
same construction applied to a shift instead of a constant. The two papers cite each other and
do not share a table.

### 3. Alignment, not the kernel

Written from the third part of `docs/benchmark.md`, with three results added since: the
evidence chooses α and λ as well as cross-validation at a quarter of the cost (+0.007 in
R²pred, interval containing zero, `results/evidence/`); one width per field and a posterior over
the field weights both gain nothing (`perfield/`, `mixing/`); six pharmacophore features beat
twenty Sybyl indicators below forty molecules and lose above (`pharmfields*/`). The claim is
now broader: on an aligned series the kernel, its layer, its widths and its weights all move
q² by less than the bootstrap interval, and the superposition moves it by 0.1 to 0.7.

Before submission: one 2D baseline on all 86 sets (ECFP ridge or random forest); Wilcoxon
signed-rank beside each paired t-test; a figure of the paired differences.

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
wins on six of seven hosts, with the rest in paper 13.

### 5. Quantum-chemical fields in CMF

All runs are done and the result is as the 22 October plan expected: a measurement with one
positive control. Hammett (paper 2): whole-molecule charges help where resonance matters.
Corrosion series: Fukui fields, Hirshfeld electrostatics and quantum descriptors add nothing to
the five fields, and descriptors alone are far worse. Conceptual-DFT fields: a Fukui function
cannot be fitted by isotropic atom-centred Gaussians (relative error 0.8 to 0.9), giving each atom
a direction fixes the fit (to 0.6) and not the model (-0.004 in q²), and the exact grid kernel does
no better than the condensed index. Am/Eu: quantum charges reproduce the classical model to the
third decimal (+0.803 against +0.803) and the quantum stage adds nothing.

Before submission: bootstrap intervals on every corrosion comparison, since with 20 to 29
molecules the differences are inside the resolution and the paper must say so with numbers.

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

Before submission: the same reproducibility and leave-paper-out analysis on a second node
(the per-experiment rows of the quinoline node are a query away), and one single-protocol
series beyond Hu 2011 with ten or more compounds, if the database has one.

Papers 4, 5, 6 and 14 all use corrosion data. The rule: paper 4 takes the frame columns, paper 5
the quantum columns, paper 6 the chemistry, paper 14 the Fe(110) adsorption energies.

### 7. How much of a co-field is real

This replaces the old "Interpretation" paper, which had tools and no measurements. It now has
three: the closed-form standard deviation of a co-field masks 70% of what would have been drawn
and raises the sign stability of what remains from 0.906 to 0.978 (`cofield_error/`); the
pharmacophore read out of a masked co-field separates the active quarter at AUC 0.665 against
0.495 for the same features in random places (`query/`); and the cosine of two co-fields
separates two models of one SAR from two models of two SARs by +0.22 and tracks how well one
orders the other's molecules (`transfer/`). Shapley attribution, the field difference and pocket
complementarity are the tools, shown on the three worked series, where one of the three fails its
own y-randomisation and is kept for that reason.

Before submission: a medicinal chemist co-author to judge the three cases; this is the one paper
in the first two tiers that cannot be finished alone.

### 8. How certain is one prediction

Conformal intervals hold their nominal coverage at 80, 90 and 95% on the Sutherland sets, and
the Gaussian-process interval under-covers at every level (`conformal/`). Of four error
estimates, none orders the errors well (risk-coverage efficiency 0.18 to 0.20) and the GP
variance is marginally best (`errorpred/`). Ranking by a lower confidence bound improves
BEDROC over the point estimate (`bound/`). Fitting the order instead of the values gains nothing
on a clean set and a little on sets pooled from sources with offsets (`ranking/`). The claim is
modest and reliable: the program says how often it is right, and says it in a way that was
checked.

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

The 2003 paper is the natural frame for a J. Med. Chem. perspective; otherwise J. Chem. Inf.
Model. The two-cavity selectivity, which follows the size of the guest, is this paper's; the
single-host tables are paper 13's.

### 10. A target that belongs to an atom

Three results. The field around a carbon under a Gaussian envelope, turned into the invariant
autocorrelation, predicts 13C shifts at q² +0.97 and 8.8 ppm against +0.85 and 20.5 ppm for the
atom-centred count, on 5096 carbons of 400 molecules with folds that hold out whole molecules
(`local/`). The same construction gains nothing on 912 carboxylic acid pKa values, and the
difference field of the acid and its anion is worse, because every acid loses its proton the same
way. A frame around one aromatic carbon, fixed by a rule, allows the overlap kernel and a
co-field: q² +0.90 against +0.77, and with one indicator field per atom type the co-field becomes
a table of what an atom of each type at each ring position adds to the shift, which is the
substituent-increment table read from the data (`local_cofield/`).

Before submission: one published shift predictor on the same molecules (the HOSE-code method
of nmrshiftdb2 or a graph network) so the 8.8 ppm has a reference point; the 400-molecule sample
enlarged if the comparison asks for it.

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

Before submission: three more LIT-PCBA targets, since two decide nothing on their own.

### 12. Which molecule to make next

On 80 runs over the Sutherland sets, the confidence bound and expected improvement find a hit in
three to four acquisitions where random takes six, and the variance rule improves the model where
the greedy rule degrades it (`active/`): finding an active and improving the model are different
tasks and need different rules. The substituent design loop, leave-one-group-out on two sets,
ranks held-out groups at Spearman +0.40 and its first suggestion is 1.4 log units above the
library mean (`design/`); on a table with no scaffold the invariant kernel puts adamantyl first
of 11 with every adamantyl guest removed, and a held-out group is pulled towards the mean
(`guest_design/`).

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
statement; "better" is not.  The same README holds the physics of the tier: the OC20 head of
UMA makes the ideal Fe(110) surface a saddle point (the slab is kept rigid), has no
dispersion (the poses are docked before they are relaxed), and its energies are learnable at
q² 0.5 by the overlap kernel in the surface frame but say nothing about inhibition (paper 6).

Before submission: two more Matbench tasks (formation energy and band gap are the natural ones)
so the kernel is placed on the leaderboard rather than on one task; a DFT anchor on the
benzimidazole structures themselves (VASP single points on eight to ten UMA geometries), since
OC20's fragments could not supply one; a co-author from materials science for the reading of
the co-patterns.

### 15. Review

Year 2, after the first three tiers are in print: the CMF series from the 2009 Doklady paper to
the present, organised by what the measurements settled (the alignment decides; the kernel,
the hyper-parameters and the fourth dimension do not; the co-field has an error bar and a
selectivity; the same kernel reaches atoms, surfaces and crystals).

## Sequence

| Tier | When | Papers | Why these first |
|---|---|---|---|
| 0 | Month 1 | Release (item 0) | Every paper cites the DOI |
| 1 | Months 1 to 3 | 1, 2, 9 | 1 and 2 to ChemRxiv in the same week so each cites the other; 9 is the most distinctive result and continues the 2003 paper |
| 2 | Months 3 to 6 | 3, 4, 10 | 3 and 4 are one argument in two halves; 10 is complete and only needs its reference predictor |
| 3 | Months 6 to 9 | 7, 8, 12 | The co-field and the posterior, with the acquisition study between them; 7 waits on a chemist co-author |
| 4 | Months 9 to 12 | 5, 6, 13 | The application papers, each citing the method paper its tables came from |
| 5 | Months 12 to 15 | 11, 14 | Both need more data (three LIT-PCBA targets, two Matbench tasks) and a co-author outside chemoinformatics for 14 |
| 6 | Year 2 | 15 | After tiers 1 to 4 are in print |

Fifteen papers in fifteen months is not realistic for one author, and the plan does not assume
it is. The tiers are an order of priority, and rule 5 limits the work in progress to three. If
time runs out, the papers that fold into others are 12 (into 8), 13 (into 4 and 9), 5 (into 6) and
11 (a short communication). Papers 1, 2, 3, 4, 9 and 10 do not fold into anything.

## What each paper still needs, in one table

| # | Open item | Effort |
|---|---|---|
| 0 | Public release, Zenodo DOI, PyPI, a reproduction run on another machine | days |
| 1 | Figure of the application; nothing else | days |
| 2 | Taft constants, one more scaffold, published DFT σ predictions as the reference | one to two weeks of runs |
| 3 | 2D baseline on 86 sets, Wilcoxon tests, paired-difference figure | days |
| 4 | Nothing; the enantiomer limitation is stated | none |
| 5 | Bootstrap intervals on the corrosion comparisons | days |
| 6 | The reproducibility and leave-paper-out analysis on a second node; the descriptor baselines rerun on the earlier series | days |
| 7 | A medicinal chemist's judgement of three cases | a co-author |
| 8 | Nothing | none |
| 9 | Nothing; the 2003 set is rebuilt and modelled | none |
| 10 | One published shift predictor on the same molecules | one week |
| 11 | Three more LIT-PCBA targets | one to two weeks of runs |
| 12 | Nothing | none |
| 13 | Nothing | none |
| 14 | Two Matbench tasks, a VASP anchor on the benzimidazole structures, a materials co-author | weeks, and a co-author |

## Risks

* **Overlap.** Fourteen papers from one repository will look like one paper cut into pieces
  unless each has a claim of its own. Rule 2 is the control, and the assignment of tables in rule
  1; a reviewer who sees the same table twice is right to object.
* **Novelty of paper 1.** Eight prior publications on the method. The paper opens with what is
  new in the implementation and treats the reproductions as validation.
* **Negative results.** Papers 5, 8 and 11 are mostly negative. Each carries its positive control
  (Hammett; conformal coverage; the PU result) and its intervals, and says what the data can and
  cannot resolve.
* **Small series.** The corrosion sets of 20 to 30 molecules cannot separate differences below
  about 0.1 in q²; every comparison on them is reported with a bootstrap interval. The three
  worked cases of paper 7 are 22 to 33 molecules, and one of them fails its own check, which the
  paper reports.
* **Two targets, one task.** Paper 11 rests on two LIT-PCBA targets and paper 14 on one Matbench
  task. Neither is submitted until the count is raised; both are in the last tier for that
  reason.
* **Co-authors.** Papers 7 and 14 need a judgement the author cannot supply alone (a medicinal
  chemist, a materials scientist). Both are placed after the papers that need nobody else.
* **The release.** Everything cites the DOI. Item 0 is a week of work and is the first thing
  done, not the last.
