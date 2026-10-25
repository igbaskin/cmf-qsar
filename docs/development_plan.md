# Development and extension plan for CMF

*Drafted 22 October 2026. The companion of `docs/publication_plan.md`: that document says what gets
written and where it is sent; this one says what gets built, in what order, and what each piece has
to prove before it is allowed to stay. File references point at the code the work starts from.*

## Where the program stands

The repository holds, as of 25 October 2026 (the lists below were written on 22 October and are
kept current; the counts are the ones the repository gives today):

* A library of 53 modules and about 22,600 lines (`cmfqsar/`), a PySide6 application, a
  command-line tool with eighteen subcommands (`build`, `build-crystals`, `diagnose`, `predict`,
  `unalign`, `map`, `compare`, `cluster`, `explain`, `mil`, `hetero`, `pharmacophore`,
  `difference`, `topomer`, `complement`, `report`, `suggest`, `design`), 602 tests, a typeset manual and a README that doubles
  as the record of every measurement made so far.
* Closed-form kernels for the five physico-chemical fields, the indicator fields, fields from any
  per-atom property, fitted probe fields, conformational ensembles (mean-map kernels), the
  rotation-invariant autocorrelation kernel and topomer fields; a Gaussian or arc-cosine layer on
  top, and multiple kernel learning for the mixing coefficients.
* Learners: kernel ridge regression, SVR, Gaussian-process regression with variances, SVC, a Laplace
  Gaussian-process classifier, five one-class descriptions, and a multi-task model on a shared
  kernel whose selectivity is a difference of co-fields (`cmfqsar/multitask.py`, measured on the
  corrosion series in `benchmarks/results/multitask/`).
* Interpretation: co-fields, contributions, exact Shapley values under nonlinear kernels, field
  differences, pharmacophores read out of a co-field, pocket complementarity, kernel GTM maps.
* Validation: nested cross-validation, leave-one-chemotype-out, y-randomisation, applicability
  domain, bootstrap intervals, posed decoys with the early-recognition metrics.

The measurements already in hand settle several questions, and the plan treats them as settled:

| Question | Verdict | Where |
|---|---|---|
| Does the kernel matter? | A Gaussian layer gains on average and is a coin flip per set; the alignment decides fifteen times as much | `docs/benchmark.md`, third part |
| Do conformational ensembles help? | Five routes, none beats one well-chosen conformation on an aligned series | README, "The verdict on the fourth dimension" |
| Do probe fields buy accuracy? | No; they are justified by interpretation only | README, "GRID-like probe fields" |
| Do quantum-chemical fields help? | Not on the corrosion series or on Am/Eu; yes on Hammett, where resonance needs whole-molecule charges | `benchmarks/results/corrosion_qm/`, `am_eu/`, `hammett/` |
| Can a Fukui function be a sum of atom Gaussians? | No (relative fit error 0.8 to 0.9), and the model does not care | `cmfqsar/cdft.py`, `benchmarks/results/cdft_kernels/` |
| When do indicator fields pay? | Above about forty molecules | `docs/benchmark.md` |
| Is the one-class ball a good description of the actives? | It comes last of five | README, "One-class models" |
| Does realigning the training set by confidence help? | Not reliably; one round with a 1 Å guard is the safe default | README, "Alignment by prediction confidence" |

What is implemented but has no measurement behind it: Shapley attribution, field differences,
pharmacophore extraction, complementarity, the GP classifier, the one-class descriptions on a real
screening benchmark, and kernel GTM. What is missing as software: packaging,
continuous integration, a release, a versioned model file, gradients of probe fields
(`cmfqsar/kernels.py` raises `NotImplementedError` there), the marginal likelihood for regression
(it exists for the classifier only, in `cmfqsar/gpc.py`), and a way to screen more molecules than
fit in memory.

## Rules the plan follows

1. **Every extension is measured against the five-field, single-conformation model on the same
   folds before it becomes a default.** The fourth-dimension study is the template: five routes
   tried, all reported, the mode kept because it is exact and costs nothing when unused.
2. **The model stays linear in the field wherever a co-field is wanted.** Anything nonlinear gets
   Shapley values instead, and anything that cannot produce either is not a CMF model.
3. **The kernel stays in closed form.** Grids are for measurement and display; the conceptual-DFT
   study showed that an exact grid kernel does no better than a fitted one.
4. **A difference below the bootstrap interval is not a result.** On a series of twenty to thirty
   molecules that is about 0.1 in q²; every comparison on such a set is reported with its interval.
5. **A module without a benchmark script is a candidate for removal.** The benchmark directory
   is the proof that a feature does something; a feature that cannot be measured is documented as
   a display tool, as the probe fields are.
6. **The rebuild is one command.** Every table in the documentation is produced by a script under
   `benchmarks/`, and the numbers reproduce on a machine that is not this one.

## A. The platform (months 0 to 3)

This is what the software paper needs, and nothing in the later parts should wait for it.

* **A1. Packaging.** Done 22 October 2026 for the part that lives in the repository:
  `pyproject.toml` with the version read from `cmfqsar/__init__.py`, the commands `cmf` and
  `cmf-app`, extras `gui`, `docs`, `dev`, `examples` and `all` so the library installs without Qt
  (there is no `orca` extra, since ORCA is a program and not a package), the command-line tool
  moved into the package as `cmfqsar/cli.py` with `cmf_cli.py` left as a launcher,
  `requirements-lock.txt` with the versions the benchmarks were run under, and `CITATION.cff`.
  Still to do, once the repository is public: the v1.0 tag and the Zenodo DOI.
* **A2. Continuous integration.** Done 22 October 2026: `.github/workflows/ci.yml` runs the
  suite on Ubuntu, Windows and macOS with Python 3.10 and 3.13 from `pyproject.toml`, once more
  from `requirements-lock.txt` with the manual typeset, and the GUI autotest under the offscreen
  Qt platform on every job, failing unless it reports success. The suite takes 23 s and its
  slowest tests are numerical integrations, not embeddings, so the shared fixture this item
  foresaw was not needed. Verified locally on Windows step by step; the Linux and macOS legs run
  for the first time on the first push.
* **A3. The data statement.** Done 22 October 2026 as the record: `docs/public_3dqsar_datasets.md`
  gives every set under `examples/` its origin, terms, script, size, users and a status (free,
  derived, unclear). Four items are of unclear status, `sutherland/`, `steroids_cbg.sdf` and the
  two derived from the first; the document says what replacing each copy by its script would
  cost, and the decision to remove them before the repository is public is the author's, not
  taken here. Two further gaps it named: the corrosion exports lack per-compound references, which stands;
  and the Hammett constants were transcribed from memory, **checked 25 October 2026**. The rows
  of Table I were recovered from the scan by the coordinates of its words, since its text layer
  comes out column by column; 39 of the 95 substituents were matched to a row and 38 agreed to
  the last digit. CH2OCH3 did not -- written as 0.02 / 0.03, given by row 223 as 0.08 / 0.01,
  confirmed on the page image -- and is corrected in the script and in all five SDF files, with
  the benchmark re-run. The remaining 56 could not be matched automatically, because the scan's
  character recognition mangles the names. The hydrophobic constant pi turned out not to be in
  that paper at all: it tabulates sigma, F and R, so the attribution is corrected and pi remains
  unchecked.
* **A4. One entry point for the tables.** Done 22 October 2026. `benchmarks/run_all.py` lists
  the eleven benchmarks with their inputs, runs them (`--compute`), rebuilds every
  `results/*/table.md` from the result files (`--tables`; the two scripts that had no table
  builder got one), splices the two tables of `docs/benchmark.md` and the two-endpoint table of
  the README between markers, writes `benchmarks/manifest.json` with the package versions, platform,
  commit and the hash of every data and result file, and `--check` fails when a rebuilt table or
  spliced document differs from what is on disk or a data file from the manifest; the locked CI
  job runs it. `benchmarks/kernel_cache.py` caches the training kernels on disk, keyed exactly by
  the atom table, the field set and the model settings, and the six scripts that build kernels go
  through it. Not done: the other result tables of the README stay written by hand, since each is
  a summary in its own words of a result file the registry names.
* **A5. A versioned model file.** Done 22 October 2026. The model file is version 2 and carries
  its provenance (program, Python, NumPy and RDKit versions, date, the field list, parameterisation
  and charge model, where every training molecule's charges actually came from, the fraction with
  explicit hydrogens, the box of the training atoms, the activity range); a file from a newer
  program is not opened, a version-1 file loads with a note. `cmfqsar/contract.py` checks
  molecules about to be predicted against the record: file charges missing is a refusal (the
  silent fall-back to Gasteiger charges, found on the way, is now reported by `MoleculeFields.
  charge_source`), hydrogens, frame and empty property fields are warnings. `cmf predict` and
  `cmf explain` stop on a refusal unless `--allow-mismatch`; the application logs the findings
  and refuses in a dialog. Two faults of the file were found on the way and fixed: a loaded
  model could not give predictive variances (its training kernels are not stored and were not
  recomputed), and a loaded model with a descriptor block dropped the block silently, since the
  descriptor statistics were not written; they are now, and a version-1 file refits them. **Done 25 October 2026**: a molecule inside
  the training box but wrongly oriented, which the box test cannot see because a rotation about a
  molecule's own centre moves no centroid. The check compares the molecule with the *shape* the
  training set occupies instead of its box -- the share of its atoms within 1.5 A of some training
  atom, against the fifth percentile of the same quantity measured on each training molecule
  against the others, less a slack, so the rule comes from the set as the applicability domain's
  threshold does. On the eight Sutherland sets it catches 95.5 % of held-out molecules turned 90
  degrees about their own centre, for a false-positive rate of 2.6 % on the molecules left alone.
* **A6. Land the multi-task model.** Done 22 October 2026: `cmfqsar/multitask.py`, its test, the
  selectivity dialog and `benchmarks/run_multitask.py` are committed with the result on the
  corrosion series (README, "Two endpoints on one kernel"; `benchmarks/results/multitask/table.md`).
  The first run of the benchmark had a correlation grid of non-negative values against a pair of
  anti-correlated endpoints and could not have found the sharing; the grid now covers both signs.

## B. Scale and speed (months 2 to 6)

The kernels are vectorised NumPy over atom pairs (`cmfqsar/kernels.py`): a 300-molecule set takes
seconds per α, the 86-set benchmark hours, and the 130 decapeptides of `mittal/AMPH1` cost fourteen
times a normal set. None of this blocks a congeneric series; all of it blocks screening and the
larger searches in parts C and E.

* **B1. Share the distance matrix across α.** The grid search computes a kernel per α; the
  squared distances do not change. Compute them once per molecule pair and take the exponentials for
  every α from the same array; for the steric field the width enters through `b = α/R²`, which is
  the same block with a different scalar.
* **B2. A cutoff with a neighbour list.** A Gaussian overlap is below 10⁻⁶ of its peak beyond
  about `7/√α` Å. A KD-tree neighbour list per molecule pair turns the atom-pair sum from quadratic
  in atoms to linear for large molecules, with an exact fallback and a test that the two agree.
* **B3. Prediction out of core.** Predicting needs only the rows `K(X, X_train)`, so a screening
  run over 10⁵ posed molecules should stream through an SDF and never hold more than one block. The
  CLI gets a `screen` subcommand that does this, with the decoy posing of `cmfqsar/decoys.py`.
* **B4. Hyper-parameters by evidence rather than cross-validation.** The GP marginal likelihood is
  implemented for the classifier (`select_by_evidence` in `cmfqsar/gpc.py`) and not for regression,
  where it is one determinant and one quadratic form on the factorisation KRR already does.
  Selecting α and λ by evidence costs one factorisation per candidate instead of ten, and it opens the door to
  per-field α, which the 2013 protocol shared across fields because a grid over five widths was
  unaffordable. Measure on the 86 sets: evidence against cross-validation, shared α against per-field
  α, with the usual intervals.
  **Done 25 October 2026**: `cmfqsar/evidence.py` and `--select evidence` on `build`. The bias is
  marginalised under a flat prior rather than fitted, since the matching Gaussian process has an
  unknown constant mean and fitting it before scoring would count it twice; the kernel amplitude
  is profiled out at its closed-form optimum, so the criterion depends on α and λ alone and its
  ranking is unmoved by the units of the activity. Both forms are checked against the textbook
  expression in the tests.
  **Measured on the eight Sutherland sets**, five 70/30 splits each, both criteria searching the
  same grid on the training part: R²pred +0.471 chosen by cross-validation against +0.479 chosen
  by the evidence, a paired difference of **+0.007** with a 95 % interval of [−0.013, +0.028],
  better on six of eight sets. Indistinguishable in quality, four times faster at selecting.
  The finding worth keeping is next to it: the two criteria land on the **same point of the grid
  in 0 to 60 % of splits** and predict equally well regardless, so the surface is flat near its
  optimum and the cost of the search matters more than the point it picks.
  **Not done:** the 86-set measurement of the item's own text. The per-field α this makes
  affordable was the reason the item was worth doing, and it was spent at once: D6 below, where
  the saving buys five widths instead of one, the evidence improves on every split and external
  prediction does not move. The second finding is the more useful of the two.
* **B5. Optional compiled kernels.** Only if B1 and B2 leave screening too slow: a Numba path for the
  atom-pair block, behind the same function signature, with the NumPy path as the reference.

## C. The alignment problem (months 3 to 12)

This is the scientific centre of the plan. The 2019 paper's *Future perspective* names alignment and
flexibility as the unsolved problem of 3D-QSAR and proposes solving it by maximising prediction
confidence; the benchmarks since then show that the alignment decides more than anything else; and the
confidence criterion is implemented, recovers scrambled test poses to 0.27 Å on thrombin, but does not
reliably improve a training set. The work is to turn one criterion into a method with a protocol that
can judge it.

* **C1. A protocol that can judge an alignment.** q² cannot, because realigning changes the
  geometry q² is computed on, and the README records q² and external R²pred moving in opposite
  directions. The judge is the crystal pose: the FEP sets with pockets (`examples/openff`,
  `schindler`, `ross`) have one for every ligand. Metrics: RMSD of the recovered pose to the crystal
  pose, and prediction error on held-out molecules posed without their activity. Both, always, and the
  unalign control (`cmf_cli.py unalign`) before every run.
  **Done 22 October 2026** (`e44b271`): `cmfqsar/poses.py` -- `pose_rmsd`, `POSERS`, `judge`,
  `judge_training_frame` -- and `benchmarks/run_poses.py`, with 88 result files and
  `benchmarks/results/poses/table.md`. A model is fitted on the training molecules in their
  crystal poses, the held-out poses are destroyed, and each method recovers them without ever
  seeing a held-out activity; the table carries the RMSD and the external R2pred of the same run
  in one row, and `scrambled` is the first method of every run, so the floor is always there.
  The RMSD is symmetry-aware and taken **without superposition**: `GetBestRMS` would report
  0.00 A for a ligand placed in the wrong site, and `CalcRMS` reports the 1.0 A it has moved.
  **Still to do:** the protocol has been exercised on `examples/openff` only, 16 of its sets.
  `schindler` and `ross` are in the repository with their pockets, and `run_poses.py` hard-codes
  `DATA = examples/openff`; extending it to them is a data path and a run, not new work.
* **C2. Flexible alignment by confidence.** The present search is rigid-body with analytic
  gradients. Add the torsions, with the conformer generator supplying starting points and the
  kernel gradient already in `kernel_gradient` extended by the chain rule over rotatable bonds. The
  objective becomes posterior confidence times a prior on the conformer energy, which is the
  proposal of the 2019 paper in full. Probe-field gradients, now `NotImplementedError`, are filled in
  on the way so every field can take part.
  **Done 22 October 2026** (`e44b271`): `cmfqsar/flexalign.py` turns the rotatable bonds with the
  torsion chain rule `dx_i/dtheta_j = a_j x (x_i - p_j)`, under a one-sided MMFF strain prior and
  from several generated starting conformations; the probe-field branch of `kernel_gradient` is
  filled in and agrees with finite differences to 7e-10. `--align-flexible`,
  `--align-conformers`, `--align-energy-slack`, `--align-energy-weight`, and a tick box in
  *Align by prediction confidence...*. The displacement guard now applies to the rigid motion
  only, since a turning bond is not a translation.
  Judged under C1, with the held-out **conformations regenerated**: median 0.86 A and R2pred
  +0.264, against 1.94 A / -0.391 for substructure alignment and **2.27 A / -0.396 for the rigid
  confidence search**, which is worse than not refining at all. That failure is what the protocol
  was built to expose, and the torsions are what rescue the criterion. The five conformer starts
  are worth 0.55 A of it, better on 15 of 16 sets.
  **What remains is the conformation, not the pose:** +0.264 against the crystal poses' +0.334,
  with a conformational residue of 0.71 A.
* **C3. Alignment inside the pocket.** `cmfqsar/docking.py` places conformations rigidly and
  `cmfqsar/complementarity.py` scores them against the pocket fields. Use complementarity as the
  prior and confidence as the likelihood: the pose must fit the pocket and sit inside the model's
  domain. Measure against the curated frames on the FEP sets, where the curated frame is the
  crystal pose and the ceiling is known.
  **Not done.** `cmfqsar/docking.py` scores `complementarity - clash` by rigid-body optimisation,
  which is the prior half of the objective; the confidence likelihood is not combined with it,
  and nothing has been measured against the curated frames. The module says of itself that it is
  a convenience for getting a pose into the right frame, not a scoring function.
* **C4. Training-set co-alignment done properly.** The present iteration (fit, realign, refit)
  contracts the ensemble and degrades one set in four. Treat it as what it is, an expectation-
  maximisation over poses, and regularise it: the displacement guard, a leave-one-out variance so no
  molecule is aligned against itself, and an annealed λ. Judge under C1 only.
  **Done 24 October 2026** (`6d85fd2`), and the answer is null. `cmfqsar/coalign.py` is the
  expectation-maximisation with all three regularisers -- a leave-fold-out error so no molecule
  is aligned against itself, an annealed lambda, and an acceptance test that keeps a round only
  when it improves that error by 1 % relative, with exact rollback otherwise. Judged under C1 on
  16 FEP sets (`*_frame.json`, four rounds, guard 1.0 A): starting from scrambled training poses
  it recovers **nothing** -- 5.56 A against the 5.55 A floor, recovery -0 %, R2pred -2.325 -- and
  on top of a substructure alignment it is marginally **worse** than leaving it alone, 0.15 A
  against 0.14 A and R2pred +0.267 against +0.305.
  So the regularisation does what it was asked to do, which was to stop the iteration degrading
  one set in four, and the method still does not find a frame. The remaining value of the item is
  the negative result and the acceptance test, which any later co-alignment should keep.
* **C5. A richer invariant kernel.** The field autocorrelation kernel (`cmfqsar/invariant.py`) is
  alignment-free and geometry-aware but two-point, so it is blind to chirality and to the
  orientation of one field with respect to another. A three-point extension (field triples, or the
  cross-correlation of two field types) keeps the closed form and is the natural next term. Measure
  on the five chemotype-split sets under leave-one-chemotype-out, where the alignment-free baseline
  already sets the bar.
  **Built, not measured.** The three-point extension is in `cmfqsar/invariant.py` -- `chirality`,
  `chirality_triples`, the `chirality` and `chirality_shells` fields of `AutocorrelationSpec`, the
  signed triple product resolved over radial shells, with tests and the block scaling that the
  signed channels need more than anything else does (a volume carries the cube of a length). It
  is reachable as `--ac-chirality` and `--ac-chirality-shells`.
  **Measured 25 October 2026**, and the answer is no. `benchmarks/run_chirality.py` runs the five
  chemotype-split sets under leave-one-chemotype-out, three regularisations by three distance
  grids chosen inside the training part of every fold, with the number of shells held out of the
  search so the signed channels never get more freedom than the kernel they extend. Mean q2 over
  the sets: two-point +0.063, signed +0.082 / +0.095 / +0.108 at one, two and four shells. Every
  paired interval over the five sets contains zero ([-0.028, +0.121] at four shells, better on
  three of five), and **COX2 supplies +0.187 of the +0.045 mean difference while the other four
  average +0.009**; AChE and BZR are made worse. One set out of five is not a result.
  **And the experiment those sets cannot do was then run on synthetic data**, which is what
  settles it. Twenty molecules with their exact mirror images -- one conformation reflected, so a
  pair differs in handedness and in nothing else and their two-point channels agree to one part
  in 1e15 -- with the activity mixed between composition and the CIP handedness. At w = 0 the
  model reaches q2 +0.97; at w = 1 **no variant learns it**, two-point -0.391 and the signed
  channels -0.34 to -0.37 whatever the shells and even scaled up to half the kernel.
  The reason is measurable and is the real result of C5: the signed channels of a molecule and
  its own mirror image are exact negatives, but between *different* molecules they share nothing
  -- cosine -0.016 for two molecules of the same handedness against +0.016 for opposite, a
  difference of 0.03 against a spread of 0.55, and 0.10 against 0.48 within one scaffold family.
  Handedness is not a direction in this space, so a linear model has nothing to transfer, and the
  two nulls are one null seen twice.
  **Closed, negatively.** The block is correct about a molecule and its mirror image and useless
  for carrying handedness between molecules. Making it useful needs a construction anchored on
  the stereocentre rather than summed over the whole molecule: that is **C7**, opened out of this
  result, and it inherits this item's two measurements as the gate it has to pass before it goes
  near a real set.
* **C6. A diagnosis of the data set.** Three diagnostics exist in pieces: α at the grid edge, the
  descriptor-only baseline that beat the 3D model on ACE, and the alignment-free baseline. Package
  them as `cmf diagnose`, which says before any modelling whether the frame carries signal, whether
  the series is congeneric enough for a frame to exist, and which of the three routes (aligned,
  topomer, invariant) the set should take. This is also the measurement that papers 3 and 4 share.
  **Done 24 October 2026** (`a980194`): `cmfqsar/diagnose.py`, the `cmf diagnose` command,
  `benchmarks/run_diagnose.py` and a chapter of the manual. Five routes -- aligned, counts,
  autocorrelation, topomer, descriptors -- each choosing its hyper-parameters inside the training
  fold and scored outside it, which costs the aligned model 0.05 on ACE and the topomer route
  0.07 and is the size of the bias that scoring at each route's own best point would hide. Two
  guards it needed: the activity is usually a molecule property, so the descriptor baseline is
  handed the answer, and on the corrosion files two laboratory-adjusted columns reconstruct the
  potency between them, which no per-column test can see -- so the outcome is judged as well.
  **Validated by following the advice**, which is the only check that matters for a
  recommendation: diagnose 70 % of a set, build what it recommends, score it on the rest. The
  advice names the best of the five routes on 21 of 39 splits and falls 0.096 short of the best
  on average; always taking the aligned model names it on 12 and costs 0.201. Per set, the honest
  unit, the advice costs 0.105 less (95 % interval -0.259 to +0.019), better on six of thirteen
  sets and worse on four: a hedge against the frame being meaningless, not an edge.
  **Still hand-chosen:** the q2 < 0.1 cutoff below which the command recommends no route at all.
  It never fired on these 39 splits, so the validation cannot set it.
* **C7. A chirality descriptor anchored on the stereocentre.** Opened and built 25 October
  2026, out of C5's failure. The question C5 was asked -- can a field kernel see handedness -- is still a real
  one: a series in which both enantiomers are measured and differ in potency is an ordinary
  situation in medicinal chemistry, and nothing in this program can model it. What C5 settled is
  only that *its* construction cannot, and why: the signed triple products are summed over every
  triple of atoms in the molecule, so the sign pattern is a property of the whole shape, and two
  molecules of the same CIP handedness are no more aligned in that block than two of opposite
  handedness (cosine -0.016 against +0.016, spread 0.55).
  The construction that could work is local. Around a chiral atom, take the three or four
  substituent directions in a fixed order -- by CIP priority, or by the field weights themselves
  so that the descriptor stays a property of the fields and not of a chemist's convention -- and
  form the signed volume of that small frame, one number per centre per field triple, carried as
  a channel attached to the centre's atom type rather than smeared over radial shells. Two
  molecules with the same configuration at a comparable centre then produce the same sign, which
  is the thing C5's block does not do.
  **Built and gated, 25 October 2026**: `cmfqsar/stereo.py`. Every atom with at least three
  neighbours, the neighbours ordered by their weight in a field, the signed volume of that
  ordered frame; one channel per (ordering field, weighting field), 25 for the five
  physico-chemical fields. A centre whose top three priorities lie within `min_separation` of
  each other contributes nothing rather than a guessed sign, and that guard is not a formality:
  it takes the same-handedness separation from 0.18 to 0.86.
  **It passes both gates, and by a margin.** The cosine between molecules of the same handedness
  is +0.468 against -0.468 for opposite, a difference of **0.935** against a spread of 0.702,
  where C5 manages -0.033. On the synthetic set where the activity is handedness and nothing
  else it reaches **q2 +0.721** where every variant of C5 sits at -0.35; at every mixture from a
  quarter handedness upward it is ahead of the two-point kernel, and at none of it costs little
  (+0.924 against +0.970).
  **On the five real sets it adds nothing**, +0.039 against the two-point kernel's +0.063, which
  is the expected answer and not a disappointment: no set has an enantiomeric pair, so the
  channels carry no signal and their 25 columns dilute the kernel. Off by default, on for a
  series whose stereochemistry varies.
  **What is still open** is the thing neither item can settle from this repository: a real series
  containing enantiomeric pairs that differ in activity. Finding or building one remains part of
  this item, and it is now the only part left.

## D. Fields and kernels (months 3 to 9)

* **D1. Anisotropic atom functions.** The conceptual-DFT study found that isotropic Gaussians on
  atoms cannot represent a Fukui function. Gaussians times first-order polynomials (p-type functions)
  still have closed-form overlaps; one set per atom along the bond directions is the smallest
  extension that can represent a lobe. Repeat the fit-error measurement on the four corrosion series
  with ORCA results, and keep the extension only if the fit error drops and the model notices.
* **D2. Pharmacophoric indicator fields.** Sybyl-type indicator fields pay only above about forty
  molecules because there are many types and each is rare. A vocabulary of six pharmacophore
  features (donor, acceptor, aromatic, hydrophobe, cation, anion) is learnable on small series and
  reads directly as a pharmacophore. Measure on the Sutherland sets against the five fields and
  against the Sybyl indicators, split by set size as before.
* **D3. A cheap quantum stage.** ORCA takes two to three minutes per molecule serially, so the
  quantum fields have been tried on a few series only. GFN2-xTB through `tblite` gives charges and
  Fukui indices in seconds, which puts a quantum stage on all 86 sets within reach. The Hammett
  result says where to expect a gain (resonance) and the corrosion result says where not to; the
  question is whether the 86 sets contain any series of the first kind.
* **D4. Protonation and tautomer ensembles.** The corrosion series in HCl are modelled in one
  protonation state. The mean-map kernel needs only bilinearity, so an ensemble over protonation
  states weighted by pKa is the same mathematics as a conformer ensemble; but `ensemble_fields` in
  `cmfqsar/fields.py` tiles one atom list over the members (`atoms_per_conformer`), and the atom
  contributions and the GUI read it that way. Generalise the builder to members with their own
  atoms, with a conformer ensemble as the special case that must reproduce the present numbers
  exactly, then measure on `examples/corrosion_hcl` against the single state.
* **D5. Learned increments.** The two parameterisations differ by 0.20 in q² on AChE, which says
  the hydrophobic and hydrogen-bond increments are not settled. Fit the per-type increments jointly
  with the model (gradient of the cross-validation loss through the kernel, with the published
  increments as the prior and a penalty on the distance from them), and ask the only question that
  matters: do increments learned on one set transfer to another? If they do not, the result is that
  the increments are a per-set nuisance parameter and the plan says so.
* **D6. Per-field widths.** Follows from B4: once the evidence selects hyper-parameters, α per field
  is five numbers instead of one, and the co-fields become sharper for the field that wants it.
  **Done 25 October 2026, and the answer is no**: `cmfqsar/perfield.py` and
  `--alpha-search per-field`, a coordinate descent on the evidence over one α per field, starting
  from the best shared width. It is affordable for the two reasons B4 promised — the per-field
  kernels are independent, so every one of the 5⁵ assignments is assembled from the matrices the
  shared grid already built, and a candidate costs one factorisation — and it costs a few hundred
  factorisations in practice.
  **Measured on the eight Sutherland sets**, five 70/30 splits each: the descent gains **+2.6 in
  log evidence on every one of the 40 splits**, moves **3.5 of the 5 widths** on average, settles
  on widths that read correctly (steric sharpest at mean α 2.33, donor widest at 1.16) — and
  changes external prediction by **−0.001 R²pred**, 95 % interval [−0.020, +0.018], ahead on four
  of eight sets and on **20 of 40 splits**. The correlation between what a split gained in
  evidence and what it gained in R²pred is **−0.07**.
  So the item is implemented and the hypothesis behind it is refuted: the extra freedom is real,
  the criterion likes it, and the molecules held out cannot tell. Read together with B4 — where
  two criteria disagreed about the shared width and predicted equally well — this says the
  attenuation surface is flat around its optimum, and that the search is not where a CMF model's
  accuracy is won. The flag stays for asking the question of a new series; it is not a default,
  and nothing else in the plan should be built on the assumption that a finer α grid pays.
  The co-field half of the item is untouched: whether a per-field width makes a co-field easier to
  read is a question about interpretation, not accuracy, and it needs an eye rather than a
  benchmark.

**Molecular dynamics as the source of the ensemble (D7 to D10).** The fourth-dimension study used
conformations generated from one input pose, and its verdict is that none of five routes beats one
well-chosen conformation. A trajectory is a different kind of ensemble: its frames are already
Boltzmann-sampled, it holds induced-fit poses when the ligand was simulated in the receptor, and
it carries solvent. Nothing in the formalism needs a single conformation: the kernel is bilinear in
the fields, so a time-averaged field enters it as exactly as a minimised one, and the ensemble
mode, the multi-instance weighting (`cmfqsar/mil.py`) and `charge_model="file"` already accept
frames, weights and force-field charges. What is missing is the reader, and the measurement that
says whether a trajectory does what a generated ensemble could not.

* **D7. A trajectory reader.** One function that takes a topology and a trajectory (XTC, DCD,
  NetCDF, through `mdtraj` or `MDAnalysis` as an optional dependency, in the same way as ORCA and
  xTB), strips solvent and ions, superimposes the frames on the binding-site backbone when there is
  a receptor and on a common core otherwise, subsamples by RMSD clustering to a few dozen
  representatives, and returns one RDKit molecule with the frames as conformers, the frame
  population as `conformer_weights` and the force-field charges as a per-atom property. Weights
  are the cluster populations, never Boltzmann factors of the frame energies: an equilibrated
  trajectory has sampled the distribution once, and weighting it again squares it. The kernel
  costs k² atom-pair integrals for k frames per molecule, which is why the subsampling is part of
  the reader and not left to the user. `max_conformers` and the RMSD pruning in
  `cmfqsar/conformers.py` do the rest once the frames are loaded.
* **D8. The measurement.** Short ligand-in-pocket simulations (a few nanoseconds each, GAFF or
  OpenFF through OpenMM) on one FEP set with a crystal pose for every ligand, so that the
  curated frame is known. Three comparisons on the same folds, with intervals: the crystal pose
  alone; the trajectory mean field, uniform over clusters; the multi-instance fit over the
  clusters. The first is the ceiling the fourth-dimension study never had, because its ensembles
  were built from one pose. The second question is whether the receptor frame removes the
  alignment problem for a flexible series: `mittal/AMPH1` (H) is the test case if the first set
  shows anything, since its 3.2 Å superposition error is what a shared receptor frame would
  remove. If the trajectory mean does not beat the crystal pose, the result joins the verdict in
  the README and D9 and D10 are not built.
* **D9. Solvent as a field.** The 2015 indicator-field formalism accepts any continuous density,
  and a water occupancy map from the trajectory is one. Fit the occupancy by Gaussians on a grid
  of hydration sites (the fitting machinery of `cmfqsar/probes.py` does this for probe fields) and
  add it as a sixth field with its own mixing coefficient, so that the co-field says where
  displacing or keeping a water pays. It is the first CMF field that is not centred on an atom of
  the molecule, which is why it waits for D4's builder with members of their own atoms.
* **D10. Fluctuation as information.** The ensemble mode uses the mean field only. The variance of
  the field across frames is a flexibility map, and the kernel of two variance fields is in closed
  form by the same algebra as the mean (a product of Gaussians is a Gaussian). Add it as a block
  with its own coefficient under multiple kernel learning, and let the mixing coefficient say
  whether it is worth anything; rule 1 applies, and the expectation is that it is a display tool.

Two uses of the kernel on a trajectory need no new field and can be built alongside D7 as
analysis tools: the kernel similarity between frames, with the invariant kernel of
`cmfqsar/invariant.py` when the frames are not superimposed, as a chemically weighted alternative
to RMSD for clustering conformations; and the per-residue complementarity profile of
`cmfqsar/complementarity.py` evaluated frame by frame, which is a field-based interaction
fingerprint along the trajectory. Both are reported, not measured, under rule 5.

## E. Learning (months 4 to 10)

* **E1. Multi-task beyond two tasks.** After A6: a free-form low-rank task covariance for several
  endpoints, task descriptors (the medium, the metal, the temperature) so that an endpoint with no
  measurements at all can be predicted from its neighbours, and the selectivity co-field in the GUI
  as the difference of two co-fields. The Am/Eu problem is the first application; separations of
  other lanthanide and actinide pairs the next.
* **E2. Active learning: which molecule to make next.** The GP variance is in hand. An acquisition
  loop over a set of posed candidates (expected improvement, or variance alone for exploration)
  turns the 2019 statement "maximising the overlap tells how to modify the molecule" into an
  operation with a number. Retrospective benchmark on the Sutherland sets: start from twenty
  molecules, add one at a time by acquisition or at random, and plot the learning curves with
  their intervals.
  **Done 25 October 2026**: `cmfqsar/active.py`, the `suggest` subcommand, and
  `benchmarks/run_active.py`. Six rules — expected improvement, the upper confidence bound,
  the prediction alone, the posterior standard deviation, furthest-first in field space, and
  random — with the expected improvement in closed form and a cosine filter so that a batch is
  not one molecule several times over.
  **The framing that the measurement forced.** An acquisition rule cannot be judged without
  saying what the acquiring is for, so the benchmark reports two things and keeps them apart:
  how good the best molecule acquired is (finding an active) and the RMSE on everything not yet
  acquired (improving the model). The q² of the pool is taken against the variance of the whole
  set, since a rule that acquires the extremes shrinks the pool's variance and would otherwise be
  rewarded for making the remaining molecules harder to tell apart.
  **Measured on the eight Sutherland sets**, ten runs each, starting from twenty molecules that
  hold no hit and acquiring a quarter of what is left, with α and λ reselected by the evidence at
  every step from what has been acquired and nothing else (item B4 is what makes that affordable):
  for **finding an active**, `ucb` reaches **96.6 %** of the best activity in the set against
  **92.2 %** for random, hits in a median of **4** acquisitions against 6, and gains
  **+0.163 [+0.062, +0.275]** in hit rate; `ei` hits soonest, in 3. For **improving the model**,
  the best exploration rule gains **−0.014 RMSE [−0.034, +0.006]** over random — zero is inside
  it, so **exploring buys nothing measurable** — while `greedy` costs **+0.057 [+0.038, +0.078]**.
  **Two findings worth more than the ranking.** `variance` and `maxmin` land within **0.000 RMSE**
  of each other and within 0.3 % of the same activity found: the posterior standard deviation of a
  CMF model is the distance to the training set in other units, which is item E5 from the other
  side, so there is no separate "uncertainty sampling" here to speak of. And `greedy` does better
  than it deserves to, because a series of analogues never puts a candidate far enough away for
  the training mean to be returned in place of a prediction; on a diverse library that protection
  is gone, and `ei` is the rule to use.
  **Not done:** a batch rule that refits between picks (the cosine filter is the cheap stand-in),
  the loop closed over generated candidates rather than a given file (that is E3), and any
  prospective test, which only a laboratory can give.
* **E3. Co-field-guided design.** The field difference says what a candidate is missing and the
  pharmacophore says where. Close the loop: enumerate replacements of a substituent at a position
  from a fragment library, pose them by the topomer rule (`cmfqsar/topomer.py`) so no alignment is
  needed, score by predicted activity and posterior variance, and present the ranked list with the
  field difference of the top candidates. This is the first feature that makes a prediction the
  user did not ask for, and the GUI needs a panel for it.
  **Done 25 October 2026**: `cmfqsar/design.py`, the `design` subcommand and
  `benchmarks/run_design.py`. The command does the whole loop from the series: it finds the
  scaffold, builds a topomer model, chooses alpha and lambda by the evidence, reports the q2 the
  suggestions rest on, enumerates one position against a library (a built-in list of about forty
  ordinary substituents, the groups the series already uses, both, or a file), scores by
  prediction and posterior standard deviation through the acquisition rules of E2, and annotates
  each suggestion: whether the series already uses that group there, whether the whole molecule
  has been made and what it measured, and what its fields have that the parent's do not.
  **Measured by leaving one substituent out** -- every molecule carrying a group at the varied
  position removed, the model refitted, the whole library ranked from one fixed parent: over
  87 held-out groups on 2 sets (GPB, THER) the ranking agrees with the
  measurements at a mean Spearman rho of **+0.397**, and the group the
  loop puts first is worth **+1.37** log units more than the library
  average, landing in the best 20 % of the library
  **48 %** of the time. Of the five best groups held out on each set, the median
  rank is 7 of 42 on GPB and 12 of 46 on THER: the loop finds most of what is worth finding and
  misses one.
  **How far it reaches is the other half of the answer.** Only 2 of the eight
  Sutherland sets have a scaffold that comes off at acyclic single bonds; on the rest the
  Bemis-Murcko framework is fused into the molecules and what comes off is not a substituent.
  The benchmark reports those by name rather than ranking the pieces anyway. A design loop over
  substituents needs a congeneric series, and most public QSAR sets are not one.
  **A defect it turned up, and the numbers that moved.** Gasteiger charges come back as nan for
  every atom of a molecule that carries an attachment point, and the field builder read nan as
  zero -- so every topomer model in the program, and every benchmark built on one, was fitted
  **without its electrostatic field**. `_cap_dummies` in `cmfqsar/fields.py` now caps the open
  valence with hydrogen before the charges are computed, which is what a fragment's charges mean
  anyway, and `benchmarks/run_topomer.py` (both collections), `run_corrosion.py` and
  `run_diagnose.py` were re-run. On the Sutherland sets the mean loss from giving up the
  alignment moved from +0.073 to +0.101 q2, and more usefully the pose rule's cost became
  visible: it was 0.015 q2 on four fields and is 0.066 on five, because electrostatics is the
  field most sensitive to where an atom sits and a model with no charges could not pay for
  misplacing them.
  **Not done:** the GUI panel the item asks for, growing candidates at more than one position at
  a time, and any library beyond substituent replacement -- scaffold hopping needs a different
  construction, since the topomer frame is defined relative to a scaffold that is being held
  fixed.
* **E4. Screening done on a public benchmark.** The one-class, GP-classifier and decoy machinery has
  been measured on the actives of the Sutherland sets against generated decoys only. Paper 8 needs
  DUD-E or LIT-PCBA subsets with posed decoys, the comparison with fingerprint similarity on the
  same task, and positive-unlabelled learning, since the inactives of a screening set are unlabelled
  rather than inactive.
* **E5. Error prediction compared.** Three quantities claim to predict the error of a prediction:
  the GP variance, the kernel applicability-domain measure, and (to be added) a conformal interval
  on top of KRR. Compare them on the 86 sets as predictors of the actual external error, which is
  the measurement paper 8 is missing.
  **Done 25 October 2026**: `cmfqsar/errorpred.py`, `build --error-predictors` and
  `benchmarks/run_errorpred.py`. The conformal interval the item said was still to be added
  arrived with the conformal work earlier in the session, so all three were available to compare.
  Each estimate is scored against the absolute error on held-out molecules three ways: Spearman ρ
  for the order, the risk–coverage efficiency (1 = the ordering the errors themselves give, 0 = no
  better than ignoring the estimate) for what acting on it buys, and the RMSE of the trusted half
  against the doubted half for what a two-bin rule would do.
  **Measured on the eight Sutherland sets**, five 70/30 splits each, α and λ by the evidence:
  efficiency **+0.198** for the GP variance, **+0.186** for the distance to the training set,
  **+0.181** for the domain dissimilarity and the normalised conformal width. The GP variance
  leads by +0.011 with a 95 % interval of [−0.021, +0.044], so **the three are indistinguishable
  as orderings** and the choice between them does not matter.
  Two findings outrank that. The normalised conformal width and the domain measure **score
  identically and must**: the normalisation divides the interval by the difficulty, which is the
  distance to the training set, so the width is a monotone function of the domain. There are three
  estimates here, not four, and the conformal rule's contribution is calibration, not ordering.
  The remaining two are barely two either: the estimates agree with each other at a mean Spearman
  ρ of +0.80 to +0.86 (the conformal width against the domain measure at exactly +1.000), so what
  is being measured three times is one geometric fact — how far the molecule sits from the
  training set — in three sets of units. That is why no ranking between them could be found.
  And all of them are weak: the best captures about a fifth of what the oracle would, the trusted
  half's RMSE is 0.84 against 1.02 for the doubted half (a ratio of 1.4, which is genuinely useful
  for deciding what to measure first), and on THER every estimate is negative. An estimate of
  error is a model and can be wrong about a series.
  **Not done:** the 86 sets of the item's text, and the classification analogue (a probability's
  reliability is a different measurement). This is the measurement paper 8 was missing; what it
  says is that paper 8 should quote the conformal interval for a number and any one of the three
  for an ordering.
* **E6. Bayesian mixing.** Multiple kernel learning picks one set of mixing coefficients. A
  hierarchical prior over `h_f` with the evidence as the objective gives a posterior over the field
  weights, and with it an answer to "does this series care about the steric field at all" with an
  interval rather than a point.

## F. Interpretation (months 4 to 9)

* **F1. Case studies.** Shapley values, field differences, pharmacophores and complementarity have
  tests but no story. Three FEP series with crystal pockets, each worked through end to end in a
  script under `examples/` and a section of the manual, judged by a medicinal chemist. Paper 7 is
  written from these.
* **F2. The uncertainty of a co-field.** The co-field is `Σ a_j X_j(r)`; under the GP the dual
  coefficients have a posterior covariance, so the co-field has a posterior variance at every grid
  point in closed form. Render only the regions where the co-field exceeds twice its standard
  deviation; the rest of the isosurface is noise and should not be shown. This is cheap, new, and
  changes what every co-field picture means.
  **Done 25 October 2026**, with one correction to the item's own wording. `cofield_variance`,
  `cofield_with_error` and `mask_noise` in `cmfqsar/cofields.py`; the **hide noise** tick in the
  3D view, `pharmacophore --min-z`, and the report masking its pictures and its pharmacophore by
  default (`report --cofield-z`, 2 by default, 0 to draw everything).
  **The correction.** What is computed is not a posterior variance. Under the Gaussian process
  whose kernel this is, the prior on the field weight function is white noise on L2: the value at
  a point is not a random variable, and the pointwise posterior variance is infinite everywhere.
  The finite Bayesian quantity is the variance of the overlap with a probe field of non-zero
  width, which is the GP variance of a prediction for a one-atom molecule placed there and is
  already implemented. What is computed instead is the other half of the question: the molecules
  and the kernel held fixed, the noise in the activities propagated through the fit, which is the
  question a co-field picture actually provokes. The item's own formula -- the dual coefficients
  carry a covariance, so the co-field carries a variance -- is right; only the word "posterior"
  was wrong. A unit test checks the closed form against three hundred refits on noisy activities
  and the two agree to a few per cent.
  **Measured on the eight Sutherland sets** (`benchmarks/run_cofield_error.py`), co-fields on a
  0.8 A grid, the training set then resampled with replacement thirty times and refitted -- a
  harder perturbation than the formula models, since it moves the molecules too: the mask keeps
  **30 %** of what would have been drawn, and what it keeps holds its sign under resampling
  **97.8 %** of the time against **90.6 %** for what it cuts, a difference of **+0.072** with a
  95 % interval of [+0.064, +0.083], in the same direction on **8 of 8** sets.
  Two things the measurement adds. The bar is **conservative**: the bootstrap moves the co-field
  only 0.6 times as far as the closed form allows for, so cutting at two standard deviations is
  really cutting at about 3.2 of the bootstrap's, which follows from taking sigma to be the
  out-of-fold error rather than the assay's noise alone (`--sigma` sets it directly). And the
  thing it gets right is *where* the uncertainty is, the two orderings agreeing across the grid
  at a rank correlation of **+0.82** -- which is what a mask needs, a mask being an ordering with
  a line drawn through it.
  So the item's claim stands: it is cheap, it is new, and it changes what every co-field picture
  in the program means, the report's included. Between a half and five sixths of a drawn co-field,
  depending on the field, is shape the measurement error could have produced.
* **F3. Co-fields compared.** The overlap of two co-fields is a closed-form similarity between two
  models: two targets (selectivity, again), two series on one target, or one series across rounds
  of active learning. A table of co-field overlaps is a map of SAR transfer.
* **F4. A model report.** One command that writes an HTML or PDF report of a model: statistics,
  the three validation checks, the co-field pictures from fixed viewpoints, the Shapley table for
  the extremes of the series, and the pharmacophore. This is what a user hands to a colleague who
  will never start the application.
  **Done 25 October 2026**: `cmfqsar/report.py` and `cmf_cli.py report model.cmf`. One
  self-contained HTML page — every picture a base64 PNG inside it, so it survives being emailed or
  attached, and it prints to PDF from any browser. Seven sections: what the model is; the
  statistics with the observed-against-predicted plots beside them; the validation that was
  actually run; the mean absolute contribution of each field; the co-fields projected onto the
  three coordinate planes (the signed extremum along each viewing axis, with the training atoms
  for scale — a projection rather than an isosurface, since a report has no camera); the two
  extreme molecules with their exact field Shapley values and largest atom contributions; the
  pharmacophore and the provenance.
  Two rules it keeps and that are the point of it. Everything is read from the model file, so a
  report costs no refitting and **cannot disagree with the model it describes** (`evaluate_test`
  now keeps the test activities beside their predictions so that the file alone is enough to draw
  the scatter). And **a check that was not run is reported as not run**, never silently omitted,
  which is the difference between a report and an advertisement. Everything else degrades rather
  than fails: a nonlinear kernel loses the co-fields, the contributions and the pharmacophore and
  keeps the rest.
  About three seconds for a 50-molecule model with the co-field pictures, under one with
  `--no-cofields`.
  **Not done:** the PDF writer of the item's text (a browser prints the page, which is enough),
  the 3D isosurface renders a GUI screenshot would give, and F1's case studies, which are what
  would make a report tell a story rather than state facts.
* **F5. Pharmacophore export.** The typed features with tolerance spheres of
  `cmfqsar/pharmacophore.py` in a format a pharmacophore-screening program can read, so the
  co-field can be used as a query outside CMF.

## G. The application and the tool (continuous)

* **G1. Subcommands.** Alignment is spread over flags of `build` and `predict` and the `hetero`
  subcommand; give it a subcommand of its own, `align` (by confidence, by MCS, in the pocket), and
  add `diagnose` (C6), `screen` (B3), `cofield` (grids and cube export without the GUI), ~~`report`
  (F4)~~ **done** and `benchmark` (A4).
* **G2. Projects.** A project file that holds the data set, its poses before and after each
  alignment, the settings, the model and the results, so a session can be reopened; undo of an
  alignment step; a panel that compares two models on the same molecules.
* **G3. The library as an interface.** The classes the README already presents (`CMFModel`,
  `FieldSet`, `AtomTable`, `Dataset`) become the documented stable surface, with a notebook per
  workflow (aligned series, frameless series, screening) under `examples/`.
* **G4. The manual grows with the program.** Each feature lands with its manual section and its
  benchmark script in the same commit, as the probe fields and descriptors did; `docs/make_manual.py`
  rebuilds the PDF.
* **G5. Other platforms.** The application has been run on Windows only. Linux and macOS through A2
  first, then the high-DPI and OpenGL differences the 3D view will meet there.

## H. Beyond medicinal chemistry (year 2)

The 2019 paper names materials design as a target; the Am/Eu result, where geometry is decisive
and counts predict nothing, is the first case.

* **Separations.** Extractant series for other ion pairs with the multi-task model; the
  selectivity co-field as the design tool.
* **Catalysis and physical organic chemistry.** The Hammett paper is the entry: ligand libraries
  with a shared frame (phosphines, N-heterocyclic carbenes) and rate or selectivity as the
  activity. Sybyl typing has to be extended for metal centres before the indicator fields apply;
  that extension is EC4 below, and electrocatalysis is where the series with a free frame are.
* **Peptides.** `mittal/AMPH1` is 130 decapeptides that no configuration models, and the reason is
  a 3.2 Å superposition error over a flexible chain. It is the test case for C2 and C3 together, and
  the plan expects it to stay hard.
* **Field-based similarity for reactions.** Fields of reactant and product in one frame, with the
  difference field as the description of the transformation. Exploratory; no commitment until C is
  done.

### Materials science: the extensions the crystal kernel asks for (25 October 2026)

The periodic autocorrelation kernel (`cmfqsar/periodic.py`, kernel source `periodic`, command
`build-crystals`, measured in `benchmarks/results/matbench_phonons/`) is the first CMF model of a
crystal, and the block mixing coefficients of the invariant kernels came with it. What it settled:
a typed pair-distance kernel per atom of the cell is invariant to cell, origin and orientation and
reads as chemistry through its co-pattern; the inverse square root of the mass is the field that
matters for the top of a phonon spectrum; the taper is needed; the background subtraction, right
for structure comparison, costs accuracy on a property that wants composition times density; and
fifty element-indicator fields dilute the kernel in a way the block weights halve but do not remove.
What it cannot do is the list below, ordered by the code each item needs. Rule 1 applies unchanged
with the four-field periodic model (electrostatic, steric, electronegativity, 1/√m, background
kept: MAE 92 cm⁻¹ on Matbench phonons) as the reference every item has to beat on the same folds.

**Small (days each)**

* **M1. Lower-dimensional periodicity.** Polymers as 1D-periodic repeat units and slabs as
  2D-periodic surfaces already pass through the ASE neighbour list with mixed boundary conditions,
  but the background term of `periodic_autocorrelation` divides by the cell volume. It needs the
  repeat length or the surface area, and the slab case wants the height channels of the surface
  kernel (`SurfaceSpec` in `cmfqsar/invariant.py`). Opens surface energies, work functions and
  facet-dependent adsorption. Measure on a slab set against the surface kernel alone.
* **M2. Partial occupancy and configurational ensembles.** Solid solutions and disordered sites
  need the field weights multiplied by the site occupancy; `structures_from_matbench` takes the
  majority species today. Special quasirandom structures and MD snapshots are the periodic
  analogue of the conformer ensemble and reuse the mean-map averaging of `ensemble_fields`,
  generalised as D4 asks for members with their own atoms. The fourth-dimension verdict says to
  expect little; measure once, on a solid-solution set, and keep the mode only if it is exact and
  costs nothing when unused.
* **M3. DFT arrays as fields without hand-made files.** Bader charges and magnetic moments enter
  as per-atom arrays (`prop:<name>`); a reader for VASP outputs (ACF.dat, OUTCAR magmoms,
  OSZICAR energies) feeds the kernel from the Li interface and Fe(110) runs directly. No new
  science; it removes the step every materials study has had to script by hand.

**Medium (one to two weeks each)**

* **M4. The periodic overlap kernel with co-fields in the cell.** For a fixed parent lattice,
  dopant and vacancy series, the real-space overlap integral with periodic images (a minimum-image
  or Ewald-like sum of the Gaussian pair overlaps of `cmfqsar/kernels.py`) gives the full CMF
  treatment, including a co-field drawn inside the cell, which the grid code of
  `cmfqsar/cofields.py` cannot do with open boundaries. This is the tool for the vacancy formation
  energies of the Li3PO4/Li3BP2O8/LiBO2 interface study, where the reciprocal form is a change of
  basis and the co-field is what the grant paragraph wants to show. Gate: the co-field of the
  vacancy energy must name the coordinating oxygens and the Li neighbours, and the model must beat
  the periodic autocorrelation on the same cells.
* **M5. Site-resolved models.** The periodic channel sum is a sum over the atoms of the cell, so it
  splits exactly into one contribution per site, as the atom contributions split the overlap
  kernel. That gives a local co-pattern (which site carries the property) and a route to per-site
  targets: adsorption energies on different surface atoms, defect energies by site, NMR shifts,
  Bader charges. Needs site-centred features (the row of `T` for one atom), per-site targets in the
  data model and the decomposition in `top_channels`. Measured first on the Fe(110) adsorption
  energies resolved by site and on vacancy energies by site from M4's cells.
* **M6. Explicit reciprocal-space channels.** Order beyond the cutoff is invisible to a pair kernel.
  The low-|G| part of the typed structure factors, with the Gaussian form factor as the resolution
  cutoff, added as extra channels in the manner of the reciprocal half of an Ewald sum, restores
  long-range electrostatic and superstructure information; the same module gives the s-space
  co-pattern, the α-as-resolution reading and the Parseval test that the publication plan's theory
  section needs. Measure on the elastic moduli and formation-energy tasks, where long-range order
  should matter, against the real-space kernel with the same cutoff.
* **M7. Rigid hosts as pockets.** `cmfqsar/protein.py` and `cmfqsar/complementarity.py` read a PDB
  protein; zeolite and MOF frameworks come as CIF files with periodic images and want a
  pore-centred frame. Once in, guest uptake and selectivity follow the ligand path, residue
  profiles becoming framework-atom profiles. The cyclodextrin example (`examples/cyclodextrin`)
  is the finite host that the periodic one must reproduce as a limit.
* **M8. Molecules inside crystals.** Molecular crystals, co-crystals and MOF linkers want the five
  classical fields for the molecule and the periodic kernel for the packing. The missing piece is
  bond and charge perception for molecules extracted from a periodic cell (connectivity across the
  boundary, then RDKit sanitisation), so that `build_fields` can run on a CIF. Measure on a
  sublimation-enthalpy or melting-point set against the inorganic field set.

**Large (a month or more each)**

* **M9. Grid fields from DFT.** Charge density, electron localisation function and electrostatic
  potential on a grid as the fields themselves rather than atom-centred Gaussians: a grid-field
  representation, CHGCAR and cube readers (the cube writer in `cmfqsar/cofields.py` is the model),
  an overlap kernel on grids for a shared frame and the radial autocorrelation for the frame-free
  case. It makes CMF an interpretable kernel directly on the DFT output. Rule 3 says grids are for
  display; this item is the one place the plan allows an exception, and only if the grid kernel
  beats the fitted atom-centred one by more than the interval, which the conceptual-DFT study
  says it may not.
* **M10. Three-body and chirality for solids.** The two-body kernel cannot separate homometric
  structures or enantiomorphs. Typed angle channels, or a periodic analogue of the signed-volume
  block of `chirality` in `cmfqsar/invariant.py`, close the gap at the cost of a cubic sum per
  structure; the factorisation trick of the molecular block does not survive the lattice sum.
  Measure on structure-type classification first, where the deficiency is visible, before any
  property.
* **M11. Inverse design on prototypes.** The co-pattern says which pair at which distance to add.
  Making that actionable needs a candidate generator over prototype structures with substitutions,
  ranking by the lower confidence bound of the GP view (`benchmarks/run_bound.py` measures it), and a
  loop that proposes, computes with VASP or UMA, and refits. The GP exists; the enumerator and the
  loop do not. Gate: the loop must find, on a held-out Matbench task, a structure better than the
  training maximum within a fixed budget of calculations.
* **M12. Learning from scattering data.** Because the kernel is the Debye transform, a measured pair
  distribution function or structure factor is a total, untyped version of the same object. A
  reader for experimental G(r) and S(Q) and a kernel that compares measured curves with computed
  typed ones would allow models on amorphous and nanocrystalline materials where no atomic
  structure exists. Exploratory; no commitment until M6 has shown that the reciprocal channels
  carry signal.
* **M13. The application.** None of this is visible in the GUI: no cell viewer, no co-pattern
  plot, no site colouring. It belongs to G, and it is what makes the materials side usable by
  someone other than the author. It waits for M4 and M5, which decide what there is to draw.

**Order.** M4, M5 and M6 first: together they give the most science per line and supply the theory
section and two of the case studies of the publication plan. M1 and M3 are done as they are needed
by those three. M9 and M12 only after M6 has reported.

### Electrochemistry: the extensions the surface frame and the metal centre ask for (25 October 2026)

Electrochemistry is already in the repository under other names. The corrosion-inhibitor series
(`examples/corrosion`, `examples/corrosion_hcl`; `benchmarks/results/corrosion/`, `corrosion_hcl/`,
`corrosion_qm/`, `multitask/`) are adsorption potencies on a metal in acid; the Fe(110) set
(`examples/corrosion_fe110`) holds UMA adsorption energies and relaxed poses in the surface frame,
normal along z, top layer at z = 0; the surface autocorrelation kernel (`SurfaceSpec` in
`cmfqsar/invariant.py`, `--kernel surface`) models molecules on a plane without alignment and
without co-fields; and the quantum stage (`cmfqsar/orca.py`, `cmfqsar/cdft.py`) brings Hirshfeld
and CHELPG charges, Fukui indices and Fukui grids into the kernel. What that work settled: on the
HCl series the quantum fields and descriptors do not beat the five fields (mean q² 0.38 against
0.38, `corrosion_qm/table.md`), the aligned overlap kernel and the frameless kernels trade places
series by series, and two endpoints on one kernel give a selectivity co-field.

What electrochemistry asks for next is decided by who the partner is. A redox potential of an
isolated molecule is an electronic-structure quantity, and the program already carries what a
congeneric series needs for it: Fukui and charge fields, the ionisation potential and electron
affinity as descriptors, the applied potential and the pH as descriptors, the GP variance for a
series of twenty. Nothing below is required for that; it is a benchmark to run, not code to write.
Three partners the program cannot describe yet are an electrode surface met face-on, a metal centre
with its ligand periphery, and a cation as the receptor. The items are ordered by the code each
needs. Rule 1 holds with two references: the surface autocorrelation kernel on the Fe(110)
adsorption energies for anything on a surface, and the five-field model plus the ORCA descriptors
for anything molecular.

**Small (days each)**

* **EC1. Oxidation state, spin state and solvent in the quantum stage.** `make_input` in
  `cmfqsar/orca.py` takes the charge from RDKit's formal charges and the lowest multiplicity, in
  the gas phase. A catalyst is Fe(II) or Fe(I), triplet or quintet, in acetonitrile or water; a
  radical electrolyte is a doublet. Read `charge` and `multiplicity` from SD properties when they
  are given, add a continuum-solvent keyword and a def2 basis with effective core potentials for
  the heavy elements, parse the Mulliken spin populations that stand next to the charges into
  `prop:spin`, and let `cdft.py` read the potential cube as it reads the density and spin cubes.
  The module's own measurement says Gaussian fits carry long-range tails badly, so the CHELPG
  field, already there, stays the first choice for the potential. Measure on a quinone or
  phenazine redox-potential series with Hirshfeld, Fukui and spin fields against the five fields
  and against the IP/EA descriptors alone; the conceptual-DFT result on corrosion says where not
  to expect a gain, and the Hammett result where to.
* **EC2. Planar alignment.** Nothing aligns a molecule onto a plane. The Fe(110) script places its
  starting poses by rule (flat: the plane of the two longest principal axes parallel to the
  surface; upright: the lone pair of the most basic nitrogen down) and UMA relaxes them. A
  principal-axes aligner in `cmfqsar/align.py` that lays a molecule on z = 0 with a chosen face
  down, followed by `field_align` restricted to in-plane translation, rotation about the normal
  and tilt, gives the aligned overlap kernel a surface frame and therefore the first **surface
  co-field**. Gate: on `examples/corrosion_fe110` the aligner reproduces the UMA-relaxed poses
  to the RMSD of C3's pocket placement, and the overlap kernel in that frame is measured against
  the surface kernel on `E_ads_kJmol` with the interval of rule 4. If it does not beat the
  surface kernel, the co-field is still the thing the surface kernel cannot draw, and the item
  stays as a display tool under rule 5 with that said.
* **EC3. Fields at anchored points.** The potential and its gradient at fixed points of the
  shared frame as descriptors: the positions of the atoms of a bound intermediate, the metal
  site, the reaction centre of a Hammett series. `_gauss_field_at` and `coulomb_potential` in
  `cmfqsar/complementarity.py` already evaluate fields at points; a descriptor source in
  `cmfqsar/descriptors.py` that takes a list of points in the frame and reports value and
  gradient per field makes them inputs. The gradient at the reaction centre is the oriented
  electric field of the electrostatic-catalysis literature, so it arrives with a physical unit.
  Measure first on the Hammett sets, where the through-space component of σ has a literature,
  against the field model alone; the catalyst use waits for EC4.

**Medium (one to two weeks each)**

* **EC4. Transition-metal complexes.** The steric table in `cmfqsar/atomtypes.py` stops at the
  main group, the Sybyl typing knows no metal, and RDKit will not sanitise most coordination
  complexes without dative bonds. Every molecular-electrocatalyst study rests on this, and the
  catalysis bullet of H already names it. Needed: van der Waals parameters and types for Fe, Co,
  Ni, Cu, Mn, Zn, Ru, Rh, Pd, Pt and Ir, zero hydrophobic and hydrogen-bond increments for them,
  a loader that reads complexes from XYZ or MOL files with dative bonds and the charge and spin
  of EC1, and the metal identity as a descriptor or a task. The frame is free: the metal and its
  first coordination sphere align every member of a series. Gate: a five-field model on a
  tabulated series with a shared frame (iron porphyrins for CO₂ reduction with the turnover
  frequency and overpotential of the catalytic Tafel-plot benchmarking; nickel diphosphines with
  their hydricities) measured against the ORCA descriptors alone, with the co-field read for the
  known second-sphere effects: the pendant proton relay, the through-space cation, the steric
  shield. The data are dozens of compounds per reaction, so this is GP territory with intervals,
  never SVR, and leave-one-chemotype-out applies.
* **EC5. The electric-field kernel.** Every field in the program is a scalar; the quantity a
  reaction centre feels is the vector ∇φ. The integral of ∇φ_A·∇φ_B over space, for φ the sum of
  atom-centred Gaussians, is the second derivative of the overlap of `cmfqsar/kernels.py` with
  respect to the pair distance and stays in closed form, so rule 3 is kept. A field type `efield`
  with that kernel, a vector co-field (coefficients on the same atoms, drawn as arrows), and the
  Shapley decomposition unchanged. Measure on the Hammett sets and on the EC4 series against the
  scalar electrostatic kernel; the item is justified only where the vector kernel beats the
  scalar one by more than the interval, which the through-space catalysis literature says it
  should on the cationic porphyrins.
* **EC6. A slab as the receptor.** `cmfqsar/protein.py` reads a PDB or mmCIF protein by residue;
  an electrode is a POSCAR or CIF slab with periodic images. A reader that builds the fields of
  the top layers as a `MoleculeFields` with the images inside the cutoff (the inorganic field set
  of `cmfqsar/periodic.py` supplies the parameters), the docking of `cmfqsar/docking.py`
  constrained to the surface (height, tilt, rotation about the normal, in-plane translation),
  and the complementarity scores with the surface in place of the pocket, residue profiles
  becoming surface-site profiles. The Fe(110) poses are the ground truth no protein set has:
  gate, the constrained docking recovers the UMA-relaxed poses and the complementarity score
  tracks `E_ads_kJmol` across the set. Shares the slab reader with M1 and the site resolution
  with M5; M7 is the same item for a pore.
* **EC7. A cation probe.** The probe module (`cmfqsar/probes.py`) carries water, methyl and
  carbonyl probes fitted as Gaussians. A Li⁺, Na⁺ or K⁺ probe, Coulomb plus repulsion with the
  module's own caveat about the wall, makes the field of an ionophore or an electrolyte solvent
  the energy a cation would feel. Measure on a crown-ether or calixarene binding-constant series
  and on Li⁺ solvation energies of carbonate and ether solvents (computable here with ORCA, so
  the set can be made rather than found), against the electrostatic field; selectivity between
  cations is a multi-task problem and the first electrochemical application of E1's task
  descriptors.

**Large (a month or more each)**

* **EC8. Clusters cut from periodic sites.** A single-atom or metal–nitrogen–carbon site is a
  molecular active site inside a solid. A cut-and-terminate step from a periodic cell to a finite
  cluster with atom types the field builder accepts, then the EC4 path, gives adsorbate binding
  free energies (H, OH, CO: the volcano-plot descriptors) as a cheap surrogate inside a family of
  sites, with the GP variance to choose the next DFT calculation as E2 does for molecules. Gate:
  on a public single-atom-catalyst set, beat the periodic autocorrelation kernel and M5's
  site-resolved model on the same cells. Waits for M5, which decides whether the site resolution
  of the periodic kernel already does the job.
* **EC9. Potential-dependent targets.** A catalyst is a curve, turnover frequency against
  overpotential, not a number. The descriptor block already admits the applied potential as a
  condition, which covers a fixed set of overpotentials; a product kernel over catalysts and
  over potential, fitted on the whole catalytic Tafel plot, is the model of the curve. Exploratory
  in the sense of the reactions bullet of H: no commitment until EC4 has shown that the catalyst
  kernel carries signal at one potential.
* **EC10. The application.** None of this is visible in the GUI: no slab behind the molecules
  (the protein viewer is the model), no vector co-field, no surface frame. It belongs to G and
  waits for EC2, EC5 and EC6, which decide what there is to draw.

**Order.** EC2 and EC1 first: days each, both run on data and modules already in the repository,
and EC2 delivers the first surface co-field, which is the picture the corrosion and modifier
work has been missing. EC4 next, since EC3, EC5, EC8 and EC9 are empty without a catalyst series
that loads. EC6 after M1, EC7 on its own, EC8 after M5, EC9 and EC10 last.

### Battery science: the extensions the cell and the solvation shell ask for (25 October 2026)

Battery science is the one application area where the repository already holds a piece of the
work rather than a plan: the periodic autocorrelation kernel (`cmfqsar/periodic.py`, measured on
Matbench phonons) is a model of a crystal, and the vacancy energies of the
Li₃PO₄/Li₃BP₂O₈/LiBO₂ interface are the stated target of M4. A cell has three phases and CMF
meets each differently. In the liquid electrolyte the partner is Li⁺ and the electrode, and the
right molecule is the cation with its first solvation shell. In organic electrodes and redox-flow
actives the molecule is a congeneric series with a shared frame, and the program already models
its redox potential with Fukui and charge fields and its trade-offs with the two-task kernel. In
the solid, the cell is the molecule, and transport is a property of the path between two sites
rather than of either site. Most of what batteries ask for lands on items that exist: EC7 (Li⁺ as a
probe), EC1 (charge and spin state), E1 (potential, solubility and stability on one kernel), M4
and M5 (the co-field in the cell and the site resolution), M3 (Bader charges as fields), M1
(polymer electrolytes as 1D-periodic chains), EC2 and EC6 (electrolyte molecules on lithium and
on cathode surfaces), and D4 (ensembles whose members have their own atoms), on which every
solvation-shell idea below depends. The items here are what none of those covers, ordered by the
code each needs. Rule 1 holds with the periodic autocorrelation kernel as the reference for every
property of a cell and the five-field model on the complex as the reference for every property of a
shell. Two limits stand over the whole section and are not to be argued away by a q²: transport
properties (conductivity, transference, diffusion) are collective, so CMF models them only as a
surrogate on bulk values that a simulation or a measurement supplies; and the electrochemical
window as measured is kinetic, set by passivation, so models of stability are trained on measured
values and a computed oxidation potential is a descriptor, not a target.

**Small (days each)**

* **BT1. Fractional compositions in the crystal charge model.** `crystal_charges` in
  `cmfqsar/periodic.py` takes pymatgen's first oxidation-state guess on the reduced composition,
  which fails for a partially delithiated cathode (Li₀.₅CoO₂ has no integer assignment). A
  fallback that spreads the charge over the mixed-valence species, and Bader charges taken in
  preference once M3 lands, are needed before any cathode series builds. Test: the charges of
  Li_xCoO₂ for x in {1, 0.75, 0.5} sum to zero and vary monotonically on Co.
* **BT2. Battery descriptors.** The descriptor block has no computed sources for crystals. The
  theoretical capacity from the formula, the Li content, the volume per formula unit and the
  density as per-structure descriptors; the coordination number and the mean Li–O distance as
  per-complex descriptors for the shells of BT4. Trivial once the computed-descriptor registry of
  `cmfqsar/descriptors.py` accepts an ASE `Atoms` as well as an RDKit molecule.
* **BT3. A battery data loader.** `read_structures` reads files through ASE and
  `structures_from_matbench` reads one table. Electrode data from the Materials Project battery
  explorer come as charged and discharged structure pairs with the average voltage, the volume
  change and the stability per pair, and the OPTIMADE route works from this machine without a key. A loader that fetches the pairs,
  keeps both frames and writes them in the form `crystal_table` takes gives the periodic kernel a
  battery benchmark of thousands of entries, which no molecular series has. Measure the
  autocorrelation kernel on the average voltage against a composition baseline, by chemistry
  (oxides, phosphates, sulfides) as the Sutherland sets are measured by size.

**Medium (one to two weeks each)**

* **BT4. A solvation-complex builder.** Li⁺ with its first shell is the molecule for every
  electrolyte property, and nothing builds one. A module that places one to four solvent
  molecules, optionally an anion for a contact ion pair, around a cation, optimises the complex
  (GFN2-xTB through `tblite` when D3 lands, ORCA otherwise, cached as the quantum stage caches),
  and aligns every complex on the cation with the coordination polyhedron fixing the orientation,
  so that the frame comes for free as the metal centre does in EC4. RDKit holds disconnected
  fragments and the field builder needs no change. Targets: Li⁺ binding energy, desolvation energy
  and coordination number on a carbonate, ether and fluorinated-ether series computed here, and
  measured low-temperature or rate data where a series has them. Gate: the model on the complex
  beats the five-field model on the free solvent, which is the test of whether the shell carries
  what the solvent alone does not; the co-field names the donor oxygens.
* **BT5. Solvation shells from a bulk trajectory.** D7 reads a ligand trajectory in a pocket. An
  electrolyte simulation is a periodic box with many cations, so a shell extractor that cuts the
  molecules within a cutoff of each Li⁺, unwraps the periodic boundary, and hands the shells to
  the ensemble builder as members with their own atoms (D4) is new code on top of both. The
  ensemble of shells is the only route from a field model to conductivity, transference and
  diffusion, and the fourth-dimension verdict applies: measure once, on a series with simulated
  or measured transport data, against BT4's single optimised complex, and keep the route only if
  the sampled shells beat the optimised one by more than the interval. The block-averaged error of
  the mean field is reported next to the q², as D8 requires.
* **BT6. Difference kernels on the cell.** The intercalation voltage is the energy difference
  between the lithiated and delithiated cells, and the field-difference module
  (`cmfqsar/fielddiff.py`) has no periodic counterpart. Once M4 gives the periodic overlap kernel
  in a fixed frame, the difference of the two cells' fields is a field in the cell like any
  other, its kernel is the same closed form, and its co-field shows the coordination environment
  the inserted lithium wants. Small once M4 exists; listed here because BT3 supplies the pairs it
  runs on. Measure on the BT3 voltages against the autocorrelation kernel on the discharged cell
  alone.

**Large (a month or more)**

* **BT7. Migration paths as objects.** Ionic conductivity in a solid electrolyte is set by the
  bottleneck between two Li sites, not by either site, and M5 resolves single sites. A path module
  that enumerates neighbouring Li-site pairs, finds the bottleneck on the way between them (the
  `free_space` test of `cmfqsar/complementarity.py` is the seed: the point of least free space on
  the straight or relaxed path), evaluates the fields along the path as a one-dimensional profile,
  and builds a kernel over site pairs from those profiles. Per-pair targets are the migration
  barriers from nudged-elastic-band or bond-valence calculations, and the LLZO, NASICON and
  argyrodite families have public sets. Gate: beat M5's site-resolved model on the same cells, and
  read the co-profile as the bottleneck chemistry (which anions, at what distance) the path
  wants. Waits for M5, which decides whether the site resolution alone already carries the
  barrier.

**Outside the library**

* **BT8. A slab generator for the adsorption benchmarks.** `examples/uma_fe110_adsorption.py` is
  written for one slab. A version under `benchmarks/` that takes any slab (Li(100), a layered
  oxide facet, graphite) and the pose rules of the Fe(110) script is the generator for the EC6
  benchmark on battery surfaces and for the SEI question (which solvent reduces first on lithium).
  Script code under rule 6, not a module.

**Order.** BT1 and BT3 first: days each, and together they give the periodic kernel a battery
benchmark on the voltages before any new kernel is written. BT4 next, since BT5 is empty without a
complex that builds and aligns, and BT2 as BT3 and BT4 need it. BT6 with M4, BT7 after M5, BT5 only
after BT4 has reported and D4 and D7 exist, BT8 whenever EC6 is ready for it.

## Sequence

| When | What |
|---|---|
| Months 0 to 3 | A (platform), A6 first; B1 and B2 as they are needed by the benchmarks |
| Months 2 to 6 | ~~B4 (evidence)~~ **done**, ~~D6~~ **done (null)**, ~~C1 (the protocol) and C2 (flexible alignment)~~ **both done**; ~~F2 (co-field uncertainty)~~ **done** |
| Months 3 to 9 | D1 to D5; C3, ~~C4~~ **done (null)**, ~~C5~~ **done (null)**, ~~C6~~ **done**, ~~C7~~ **built, gates passed**; F1 (case studies) |
| Months 4 to 10 | E1, ~~E2~~ **done**, ~~E3~~ **done**, E4, ~~E5~~ **done**, E6; G1 and G2 alongside |
| Months 6 to 12 | F3, ~~F4~~ **done**, F5; D7 and D8 (trajectories), D9 and D10 only if D8 says so; H begins with separations |
| Year 2 | H; M4 to M6 (periodic overlap kernel, site-resolved models, reciprocal-space channels) first, M1 to M3 as they are needed, M7 to M13 after; EC1 and EC2 (quantum-stage controls, planar alignment and the surface co-field) as soon as the Fe(110) set is benchmarked, EC4 (metal complexes) before any other EC item, EC6 with M1, EC8 after M5; BT1 and BT3 (crystal charges for fractional compositions, the battery data loader) with M4 to give the periodic kernel its voltage benchmark, BT4 (solvation complexes) after EC7, BT6 with M4, BT7 after M5; the review that ties the series together |

The order inside a part follows the dependencies: nothing in C or E is judged without the protocol
of C1 and the intervals of rule 4; D6 and E6 wait for B4; E3 waits for E2 and the topomer rule;
F2 wants the GP view that B4 makes the default; D9 waits for D4 and, with D10, for the
verdict of D8.

## What is not planned

* **Neural networks on fields.** A 3D convolutional network on a field grid gives up the closed-form
  kernel, the co-field and the exactness of the Shapley values at once. The nonlinear kernel layer
  and the Shapley decomposition already cover the case where a linear model is not enough.
* **A web service.** The program is a desktop application and a library; serving it is a different
  project.
* **General docking.** The pocket placement in `cmfqsar/docking.py` exists to pose conformations for
  a field model, not to compete with docking programs; C3 uses it as a prior and no more.

## Risks

* **Breadth.** Twenty-nine modules and one author. Rule 5 is the control: a feature that cannot be
  measured is documented as a display tool or removed, and the test count rises with every module
  that stays.
* **Data for the alignment work.** The FEP sets are the only series with a crystal pose for every
  ligand, and their activity ranges are narrow. Congeneric series from PDBbind with several
  structures per series are the next source and should be collected during A3.
* **The confidence criterion may not be enough.** C2 to C4 may show that confidence alone cannot
  pose a flexible molecule without a pocket. That is a result, not a failure, and the alignment-free
  route of C5 and the diagnosis of C6 are the plan's answer to it.
* **Dependencies.** ORCA is the serial build and xTB is not yet in the tool chain; RDKit's MMFF set-up
  alters aromaticity flags in place, so every new module that optimises geometry must re-sanitise, as
  the conformer generator does.
* **Metals and small series.** EC4 asks RDKit to hold coordination complexes it was not built
  for, and the catalyst series behind it are dozens of compounds with turnover frequencies that
  differ between laboratories by the factor the model is meant to explain. The control is the
  order: the loader and the parameters first, measured on the geometry alone (does the series
  load, align and give a steric kernel that separates the known chemotypes), then one tabulated
  series with GP intervals, and the vector kernel and anchored points only if that series shows
  signal. A q² on twenty catalysts without its interval is not reported.
* **Trajectories cost more than they may return.** D8 needs simulations that take longer than
  every benchmark in the repository together, and the fourth-dimension verdict says that ensembles
  have not paid so far. The control is the order: the reader first, the measurement on one set
  second, and the two new fields only after the measurement; an unconverged trajectory gives a mean
  field with sampling error that would pass for chemistry, so D8 reports the block-averaged error of
  the mean field next to the q² interval.
