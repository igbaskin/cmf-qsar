# Development and extension plan for CMF

*Drafted 22 October 2026. The companion of `docs/publication_plan.md`: that document says what gets
written and where it is sent; this one says what gets built, in what order, and what each piece has
to prove before it is allowed to stay. File references point at the code the work starts from.*

## Where the program stands

The repository holds, as of 29 October 2026 (the lists below were written on 22 October and are
kept current; the counts are the ones the repository gives today):

* A library of 83 modules and about 38,900 lines (`cmfqsar/`), a PySide6 application, a
  command-line tool with thirty subcommands (`build`, `build-crystals`, `diagnose`,
  `align`, `predict`, `screen`, `cofield`, `report`, `benchmark`, `unalign`, `map`, `compare`,
  `cluster`, `explain`, `mil`, `hetero`, `pharmacophore`, `difference`, `topomer`,
  `complement`, `suggest`, `design`, `transfer`, `mixing`, `atoms`, `local`, `restriction`, `isosteres`, `lowrank`, `kcca`), 1499 tests, a typeset manual and a
  README that doubles as the record of every measurement made so far.
* Closed-form kernels for the five physico-chemical fields, the indicator fields, fields from any
  per-atom property, fitted probe fields, conformational, protonation and tautomer ensembles (mean-map kernels), the
  rotation-invariant autocorrelation kernel and topomer fields; a Gaussian or arc-cosine layer on
  top, and multiple kernel learning for the mixing coefficients.
* Learners: kernel ridge regression, kernel partial least squares, SVR, Gaussian-process regression
  with variances, SVC, a Laplace Gaussian-process classifier, kernel Fisher discriminant analysis
  (the classifier that has a co-field), kernel logistic regression, the Laplacian-regularised ridge
  (the only learner here that fits molecules carrying no activity), five one-class descriptions, and a multi-task model on a shared
  kernel whose selectivity is a difference of co-fields (`cmfqsar/multitask.py`, measured on the
  corrosion series in `benchmarks/results/multitask/`).
* Interpretation: co-fields, contributions, exact Shapley values under nonlinear kernels, field
  differences, pharmacophores read out of a co-field, pocket complementarity, kernel GTM maps.
* Validation: nested cross-validation, leave-one-chemotype-out, y-randomisation, applicability
  domain, bootstrap intervals, posed decoys with the early-recognition metrics.

The measurements already in hand settle several questions, and the plan treats them as settled:

| Question | Verdict | Where |
|---|---|---|
| Does the kernel matter? | A Gaussian layer gains on average, but per set the gain is no better than chance; the alignment decides fifteen times as much | `docs/benchmark.md`, third part |
| Do conformational ensembles help? | Five routes, none beats one well-chosen conformation on an aligned series | README, "The verdict on the fourth dimension" |
| Do probe fields improve accuracy? | No; they are justified by interpretation only | README, "GRID-like probe fields" |
| Do quantum-chemical fields help? | Not on the corrosion series or on Am/Eu; yes on Hammett, where resonance needs whole-molecule charges | `benchmarks/results/corrosion_qm/`, `am_eu/`, `hammett/` |
| Does complementarity with a cyclodextrin cavity predict the constant? | No: -0.348 on 1162 beta-CD guests, against +0.348 for the guest's size alone; a one-body score has no term for the water that leaves the cavity | `cmfqsar/protein.py`, `benchmarks/results/host_pocket/` |
| Can a model with no frame say what to make next? | On the beta-cyclodextrin guests, at the top of the list: with every guest carrying it removed, adamantyl comes first of 11 and Spearman is +0.709, but a held-out group is pulled towards the mean (-0.73) | `cmfqsar/design.py`, `benchmarks/results/guest_design/` |
| Does a second endpoint help, and is the selectivity predictable? | Sharing gains nothing on three cyclodextrin pairs (every interval contains zero), but the selectivity is predictable (q2 +0.273 to +0.389) and follows the size of the guest | `cmfqsar/multitask.py`, `benchmarks/results/two_endpoints/` |
| Does an alignment-free model know which group of a guest binds? | On beta-cyclodextrin yes (the three cavity groups +0.0394 per atom against the rest of their molecule, the three polar ones -0.0723), on alpha and gamma no | `cmfqsar/invariant.py`, `benchmarks/results/guest_atoms/` |
| Do increments fitted on one series transfer to another? | No (+0.363 on the test set against +0.383 for the published table; the vectors fitted on two sets agree to +0.89 at best) | `cmfqsar/learnincrements.py`, `benchmarks/results/increments/` |
| Does a pKa-weighted protonation ensemble beat one chosen state? | No on the corrosion series in HCl (+0.261 against +0.324): at pH 0 only 1% of the molecules are mixtures, and independent-site pKa picks the state badly | `cmfqsar/protonation.py`, `benchmarks/results/protonation/` |
| Do six pharmacophore features beat twenty Sybyl indicators? | Below 40 molecules yes (+0.027 against +0.002 added to the five fields), above it no | `cmfqsar/pharmtypes.py`, `benchmarks/results/pharmfields_mittal/` |
| Can a Fukui function be a sum of atom Gaussians? | No (relative fit error 0.8 to 0.9), and the model does not care | `cmfqsar/cdft.py`, `benchmarks/results/cdft_kernels/` |
| Does giving every atom a direction fix that? | The fit, yes (84% -> 64%); the model, no (-0.004 q2) | `cmfqsar/anisotropic.py`, `benchmarks/results/anisotropic/` |
| When do indicator fields help? | Above about forty molecules | `docs/benchmark.md` |
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
  taken here. Two further gaps it named: the corrosion exports lack per-compound references, which remains the case;
  and the Hammett constants were transcribed from memory, **checked 25 October 2026**. The rows
  of Table I were recovered from the scan by the coordinates of its words, since its text layer
  comes out column by column; 39 of the 95 substituents were matched to a row and 38 agreed to
  the last digit. CH2OCH3 did not -- written as 0.02 / 0.03, given by row 223 as 0.08 / 0.01,
  confirmed on the page image -- and is corrected in the script and in all five SDF files, with
  the benchmark re-run. The remaining 56 could not be matched automatically, because the scan's
  character recognition corrupts the names. The hydrophobic constant pi turned out not to be in
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
* **A6. Commit the multi-task model.** Done 22 October 2026: `cmfqsar/multitask.py`, its test, the
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
  **Done 26 October 2026, and the premise was wrong, which is the useful part.** The distance
  matrix is **3 %** of a kernel block, measured on one 40-atom molecule against a 200-molecule
  table, so sharing it across the grid could never have been the factor the item expected. The
  measurement found the cost elsewhere and all three changes are in `cmfqsar/kernels.py`:
  the **symmetry of the training kernel** (only the atom pairs with j >= i, then mirrored:
  56 % of what was left, and it makes the matrix exactly symmetric, which it
  was not), the **radius-scaled prefactor as `z*sqrt(z)` rather than `z**1.5`**
  (27 % of the whole kernel, because a general power is a library call per
  element and the steric field evaluates it for every atom pair), and finally the shared distances
  the item asked for (16 %), through `kernel_matrices_multi` and
  `CMFModel.prepare_training_kernels_grid`, which `grid_search` now uses for a grid of more than
  one α.
  **Measured** (`benchmarks/run_speed.py`): six attenuations on 200 congeneric
  molecules 14.3 s -> 3.8 s
  (3.7x), on 60 decapeptides
  42.2 s -> 15.3 s
  (2.8x); a whole `grid_search` with five-fold cross-validation
  13.8 s -> 4.1 s (**3.4x**) with the same α
  and the same q2cv to six decimals. Every shortcut is exact; the largest difference any of them
  makes to a kernel entry is 1e-15 relative, which is round-off.
  **Not done:** the atom-pair block itself is still one dense NumPy expression per field group,
  and the exponential is now the floor (32 %
  of the original time is in exp and the per-field multiplies). B5 is the item for going below
  that.
* **B2. A cutoff with a neighbour list.** A Gaussian overlap is below 10⁻⁶ of its peak beyond
  about `7/√α` Å. A KD-tree neighbour list per molecule pair turns the atom-pair sum from quadratic
  in atoms to linear for large molecules, with an exact fallback and a test that the two agree.
  **Done 26 October 2026, and it helps only in one case.** `cutoff_radius`, `surviving_share`
  and the neighbour-list branch of `atom_kernel_block` are in `cmfqsar/kernels.py`; `CMFModel`
  carries a `cutoff` and `screen --cutoff auto|R` exposes it. The dense sum is the default and the
  exact reference, and `tests/test_kernel_speed.py` holds the two to the tolerance the cutoff was
  derived from.
  **Why it is off by default.** The molecules of a congeneric series are superimposed in one frame
  about 14 A across, and at the usual α = 0.5 the cutoff is 10.5 A for a shared-width field and
  19 A for the radius-scaled steric one, so it keeps
  98% and 100% of the atom pairs and
  a neighbour list only adds cost -- forcing it on there makes the kernel five times slower.
  The item's
  "linear for large molecules" is right only for large molecules: on
  60 decapeptides at α = 5 the cutoff
  keeps 4% of the pairs and
  `auto` is 2.18x faster.
  **How `auto` decides**: it samples 2048 real atom-pair distances and uses a list only when the
  surviving share is below a threshold that depends on the block size. A geometric estimate from
  the bounding box was tried first and was wrong by a factor of three, because the molecules of a
  series are concentrated in the middle of the box rather than spread through it; and the first
  sampling was nine times larger and showed up as a 12 % overhead where the answer is "do not
  bother".
  **Not done:** the cutoff is per field group, not per molecule pair, so two molecules that do not
  overlap at all are still evaluated; and the error it introduces (3e-07 relative) has not
  been propagated to a q2, only bounded.
* **B3. Prediction out of core.** Predicting needs only the rows `K(X, X_train)`, so a screening
  run over 10⁵ posed molecules should stream through an SDF and never hold more than one block. The
  CLI gets a `screen` subcommand that does this, with the decoy posing of `cmfqsar/decoys.py`.
  **Done 26 October 2026.** `cmfqsar/screen.py` reads the library with a forward supplier (so the
  file is never indexed), builds one block of fields at a time, computes that block's kernel rows,
  writes the scores to a CSV as it goes and keeps only the best in memory; `screen MODEL LIBRARY`
  is the subcommand, with `--chunk`, `--top`, `--beta`, `--pose`, `--pose-conformers` and
  `--cutoff`. A SMILES library is read and embedded as it streams. The posterior standard
  deviation is in every row, because a library is mostly molecules unlike the training set and a
  point estimate cannot be told from the training mean; `--beta` turns the score into the lower
  confidence bound.
  **Measured** on a library of the example sets concatenated, scored against a
  200-molecule model: 16664 molecules
  (87 MB) in 237 s at **237 MB peak
  process memory**, against 178 s and **1706 MB** loading the
  file whole. Over a four-fold library the whole-file path grows 571 ->
  1706 MB (3.0x) and the streamed one 232 ->
  237 MB (1.02x), which is the claim of the item: the memory does not
  grow with the file. The cost is 33 % of wall clock.
  **One fix that came out of it**: the GP posterior variance factorised the training kernel on
  every call, so a streamed run did it once per block and alignment by confidence once per pose.
  It is cached on the model now (`_variance_factor`), keyed on the matrix so a realignment or
  another fold still refactorises.
  **Not done:** 10^5 molecules has not actually been run, only extrapolated from 12 848; the
  posing is still a Python loop at about half a second a molecule, which dominates any run that
  uses it; and nothing parallelises over blocks.
* **B4. Hyper-parameters by evidence rather than cross-validation.** The GP marginal likelihood is
  implemented for the classifier (`select_by_evidence` in `cmfqsar/gpc.py`) and not for regression,
  where it is one determinant and one quadratic form on the factorisation KRR already does.
  Selecting α and λ by evidence costs one factorisation per candidate instead of ten, and it makes
  per-field α possible, which the 2013 protocol shared across fields because a grid over five widths was
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
  The more useful finding is next to it: the two criteria choose the **same point of the grid
  in 0 to 60 % of splits** and predict equally well regardless, so the surface is flat near its
  optimum and the cost of the search matters more than the point it picks.
  **Not done:** the 86-set measurement of the item's own text. The per-field α this makes
  affordable was the reason the item was worth doing, and it was used at once: D6 below, where
  the saved time is used for five widths instead of one, the evidence improves on every split and external
  prediction does not move. The second finding is the more useful of the two.
* **B5. Optional compiled kernels.** Only if B1 and B2 leave screening too slow: a Numba path for the
  atom-pair block, behind the same function signature, with the NumPy path as the reference.
  **Done 26 October 2026, and what it adds is threads, not speed per core.**
  `cmfqsar/fastkernel.py` is one fused loop over the atom pairs -- every field accumulated while
  the pair's distance is still in a register, so no temporaries and one pass over memory instead
  of ten -- compiled with Numba and threaded over the molecules of the first table, which is safe
  because each iteration writes its own row. `kernels.ENGINE`, `kernel_matrices(engine=...)`,
  `CMFModel.engine` and `screen --engine` select it; `plan()` declines probe fields and small
  problems and the call falls back to NumPy by itself.
  **The first finding is negative and worth recording.** Written serially the compiled loop is
  *slower* than NumPy: 0.90 s against 0.67 s on 200 molecules. NumPy's `np.exp` over an array is
  vectorised and the same calls one at a time in a compiled loop are not, unless Intel's SVML is
  installed, which it is not here. The exponentials are about half the kernel, so no amount of
  loop-level cleverness wins on one core. What the fused loop does remove is the memory traffic,
  and `fastmath` recovers the rest: serial with fastmath, 0.55 s against 0.67 s.
  **The threads are the result.** On 32 cores: one kernel of
  200 congeneric molecules 0.50 s ->
  0.09 s (**5.6x**), of 60 decapeptides
  1.99 s -> 0.12 s (**16.6x**, more because
  each thread then has more work to amortise its own start-up); a whole grid search
  2.9 s -> 0.7 s (4.0x) with the
  same attenuation and the same q2cv to six decimals; a screening run over
  4166 molecules 46 s ->
  12 s (3.8x, 91 ->
  347 molecules a second).
  **NumPy stays the default** everywhere except `screen`, which has no published numbers to
  reproduce. The two paths differ by 2e-14 relative, which is the order of the round-off
  between two orderings of the NumPy sum itself; that is not a reason to let a table depend on
  whether an optional package is installed. `tests/test_fastkernel.py` holds the two to agreeing
  and skips itself without Numba.
  **A side benefit**: inside the loop the cutoff of B2 is one comparison per pair and needs no
  neighbour list, so the compiled path gets B2 on exactly the sets where the NumPy neighbour list
  is not worth building.
  **Not done:** SVML (which would make the serial loop competitive and is a packaging question,
  not a code one), a GPU path, and threading the field construction, which is now the larger half
  of a screening run.
* **B6. Low-rank kernels for large training sets.** The exact model holds one n × n matrix per
  field and factorises the n × n joint kernel. On a 32 GB machine that stops at ten to twenty
  thousand molecules: five field kernels of 10 000 molecules take 4 GB, and the alpha grid of B1
  keeps six of them. Memory is the limit, not time: extrapolating from the 0.09 s of B5 for 200
  molecules, the compiled kernel of 10 000 molecules takes about four minutes and that of
  100 000 about six hours.
  **Assessed and measured 26 October 2026, before the implementation reported below.** Of the methods for kernel models
  on large sets, one fits this kernel and the others do not.
  *The Nyström approximation (inducing points) fits.* It replaces every field kernel by
  `K(X, Z) K(Z, Z)^+ K(Z, X)`, a matrix of rank m built from the kernels against m landmarks Z.
  The blocks `K(X, Z)` are what `kernel_matrices(P, Q)` computes and what `screen.py` streams,
  so no new kernel code is needed. The landmarks can be a subset of the training molecules, or
  Gaussians in space: the molecule kernel is the Gram matrix of a Gaussian kernel on R³
  contracted with the atom weights, so a basis of Gaussian centres in the common frame spans the
  field of every molecule that fits in the frame, and the rank it needs depends on the frame and
  the width, not on the number of molecules. Everything downstream has a standard low-rank
  form: the ridge fit in O(n m²), the variance and the evidence through the Woodbury identity
  (in the variational form of Titsias 2009, because the plain Nyström variance goes to zero far
  from the data, which is the opposite of what a lower confidence bound needs), LinearSVR and
  LinearSVC on the features instead of libsvm, the kernel alignment of the mixing coefficients
  from m × m Gram matrices, Nyström kernel PCA for the maps. The interpretation survives: with
  landmark molecules the co-field is a sum over their fields, with Gaussian centres it is
  directly a sum over the basis, so the contributions, the Shapley values, the pharmacophore and
  the co-field error bars keep their form with the index set changed from training molecules to
  landmarks.
  *Random Fourier features do not fit.* They need a shift-invariant kernel on vectors, and the
  CMF kernel has no vector input. The only analogue is sampling the fields on a grid as explicit
  features, which is the grid dependence the 2013 paper removed and breaks rule 3.
  *Matrix-free iterative solvers do not fit.* Every matrix-vector product is a full kernel pass,
  about six hours at 100 000 molecules, and conjugate gradient needs tens of them. Falkon, which
  is Nyström with a preconditioned conjugate gradient, is worth it only when m must exceed about
  10 000, and it needs PyTorch and a GPU.
  **Measured** (`benchmarks/run_nystrom.py`, `benchmarks/results/nystrom/table.md`) on the eight
  Sutherland sets, at the attenuation and ridge the exact grid search chose, on the same ten
  folds:
  - Gaussian centres 2 Å apart (170 to 350 of them) reproduce the exact q²cv to within 0.002 on
    the five sets whose attenuation is 1 or below (BZR, COX2, DHFR, THR, THER), with a relative
    error of the joint kernel between 2e-7 and 2e-2. On the three sets that chose alpha 2 or 5
    (ACE, AChE, GPB) the same grid loses 0.01 to 0.04 of q², and centres 1.5 Å apart (430 to
    630) bring it back to within 0.005 or above the exact value.
  - The reason is the width. A fixed 2 Å grid on COX2 has a kernel error of 2e-6 at alpha 0.1
    (Gaussian width 3.2 Å) and 1e-1 at alpha 5 (width 0.45 Å). A grid of centres spans the field
    only while its spacing is not much larger than the width, so at alpha 5 the frame of a
    Sutherland set would need about 0.5 Å spacing, some 27 000 centres. The rank argument holds
    for broad fields only, and three of the eight sets chose a narrow one.
  - Landmark molecules need no grid and have no width problem: 20 to 80 of them put every one
    of three random draws within 0.02 of the exact q² on seven sets; THER needed all 76. The
    joint kernel is still 1 to 10 % wrong there, so the model needs far less of the kernel than
    the kernel error suggests.
  - The whole grid search on the 2 Å kernels chooses the same attenuation and ridge on five of
    eight sets with the same q² to 0.006. The three it misses are the two alpha ≥ 2 sets (ACE
    chose 5 instead of 2, AChE 2 instead of 5) and THER's ridge (0.01 instead of 0.03, q² equal
    to 0.002). It is faster already at these sizes: DHFR's search 22.5 s → 3.3 s, COX2's
    13.5 s → 1.3 s.
  **What this settles.** For the attenuations most sets choose, 1 and below, a few hundred
  Gaussian centres give the exact model for every practical purpose whatever the number of
  molecules, and the item is safe to implement. At alpha 2 and above the inducing set must be
  finer or must follow the atoms; the atoms of a subset of molecules used as centres, which is
  between the two kinds measured, is the candidate. The implementation is two to four weeks: a
  feature module (the Nyström map per field, streamed over blocks of molecules), a primal path
  in `_Learner` with the Woodbury variance and evidence, the kernel alignment on features, the
  interpretation tools indexed by landmarks, and a model state that stores the landmarks and
  their coefficients instead of the whole training table, which also makes the model files
  smaller.
  **Two limits of the whole question.** A training set of 100 000 molecules needs all of them
  in one frame, so the alignment problem of part C, not the solver, is what stands between CMF
  and large training sets; posed libraries of that size are already handled by `screen.py`. And
  the one kernel built for many molecules without a frame, the autocorrelation kernel of
  `invariant.py`, already has explicit features of a few hundred columns: for it no
  approximation is needed, only a learner that accepts a feature matrix, which is the cheaper
  route for LP5.
  **Implemented 27 October 2026.** `cmfqsar/lowrank.py` and the subcommand `cmf lowrank`.
  `build` makes the feature map of a landmark set -- the whitener `W` with `W W^T = K(Z,Z)^+`
  from a thresholded eigendecomposition, so a landmark set with near-duplicate members cannot
  blow it up -- `transform` turns molecules into feature rows a block at a time, and `gram`
  accumulates `Phi^T Phi` and `Phi^T y` over those blocks, so the features of the whole set are
  never held at once and the memory of a fit is the rank squared whatever n is. `fit` solves
  the bordered primal system that matches the exact solver's bias, `variance` is the
  Gaussian-process variance with the **exact** self-kernel kept (the deterministic
  training-conditional form, which is what stops it collapsing away from the landmarks),
  `log_evidence` is the same quantity as `cmfqsar/evidence.py` through the Woodbury identity at
  m^3 instead of n^3, and `gaussian_centres` and `landmark_molecules` are the two kinds of
  landmark. `cofield_gaussians` returns the same (centres, coefficients, widths) that
  `CMFModel.cofield_coefficients` returns, with the index set changed from training molecules to
  landmarks, so the co-field and everything read off it keep their form. 19 tests.
  **Measured** by `benchmarks/run_lowrank.py` on the eight Sutherland sets at an attenuation of
  0.5 and a ridge of 0.1: with every molecule its own landmark the kernel, the predictions, the
  log evidence, the posterior variance and the co-field on a grid agree with the exact path to
  2e-11; a grid of Gaussian centres 2 A apart is within 0.004 of the exact q2 on every set and
  1.5 A within 0.001; 20 to 80 landmark molecules within 0.076. At 361 molecules the exact
  kernels are 6 MB and the low-rank fit 13 MB -- the rank of a 2 A grid over that frame is 1275,
  so the two cross at about 1275 molecules -- while the fit is already eight times faster, since
  a kernel entry against a one-atom centre costs a thirtieth of one against a whole molecule. On
  held-out molecules the landmarks do not reach, the variance that keeps the self-kernel is off
  by 0.013 against 0.025 for the form that drops it.
  **The flag on `build`, 27 October 2026.** `CMFModel.fit_low_rank` holds the landmarks, the
  whitening and one weight per feature instead of one coefficient per training molecule;
  `predict`, `predict_with_variance` and `cofield_coefficients` take the low-rank route when it
  is there, and `state`/`from_state` carry it, so `cmf build --low-rank grid:2.0 --save` writes
  a file that `cmf predict` and `cmf cofield` read like any other. The training table still
  travels with the file, because it is linear in the number of molecules and the provenance and
  the contract are read off it; the n by n kernel is never built at any point. 28 tests.
  **Landmark atoms, 27 October 2026.** `lowrank.landmark_atoms` puts one landmark on every atom
  of a subset of the molecules, thinned to a spacing, each with the radius of the atom it
  stands on and unit weight in every field; `--low-rank atoms:SPACING` and
  `--landmarks atoms:SPACING` take it. This is the case the assessment named for narrow fields
  and it does what it was expected to: measured by the narrow-field stage of
  `benchmarks/run_lowrank.py` on GPB, THR and AChE at attenuations 0.5, 2 and 5, a grid 2 A
  apart reproduces the kernel to 5e-01 at an attenuation of 2 and above while the atoms of the
  same molecules thinned to 0.5 to 0.75 A reach 2e-02 with a landmark count of the same order.
  On GPB at an attenuation of 5 that is 6.5e-01 against 1.8e-02 for 259 landmark atoms, and q2
  +0.548 against +0.593 where the exact model gives +0.594. The kernel error is the number to
  read: these sets are small and their q2 moves by more than the approximation does. 35 tests.
  **A test set and the conformal interval, 27 October 2026.** The low-rank branch of `build`
  now scores a test set left out by `--test-fraction` and calibrates a prediction interval with
  `--conformal LEVEL`. Neither needed new mathematics: `evaluate_test` asks only for a
  prediction, and the conformal rule asks only for the out-of-fold errors, both of which the
  low-rank fit already produces. The interval is stored in the model file, so
  `cmf predict --conformal` reports it from a saved low-rank model, and `--predictions` writes
  both subsets with `lower` and `upper` columns. `--conformal-normalise` is refused with the
  reason: the distance it divides each error by is read off the n by n kernel this path exists
  to avoid, and measured in `benchmarks/run_conformal.py` normalising gains no sharpness. 42
  tests, including one that checks the test score of the low-rank path equals the exact one
  when every training molecule is a landmark.
  **Still not done:** the low-rank branch of `build` does not take a second endpoint or write
  the JSON report; the application has no control for it; a width for the inducing
  Gaussians other than the field's own; LinearSVR and LinearSVC on the features, the kernel
  alignment from m by m Gram matrices and Nystrom kernel PCA, all of which the assessment names
  as following the same way; and any set larger than 361 molecules, which is where the point of
  the item lies and where nothing in the repository is aligned.

## C. The alignment problem (months 3 to 12)

This is the scientific centre of the plan. The 2019 paper's *Future perspective* names alignment and
flexibility as the unsolved problem of 3D-QSAR and proposes solving it by maximising prediction
confidence; the benchmarks since then show that the alignment decides more than anything else; and the
confidence criterion is implemented, recovers scrambled test poses to 0.27 Å on thrombin, but does not
reliably improve a training set. The work is to turn one criterion into a method with a protocol that
can judge it.

* **C1. A protocol that can judge an alignment.** q² cannot, because realigning changes the
  geometry q² is computed on, and the README records q² and external R²pred moving in opposite
  directions. The reference is the crystal pose: the FEP sets with pockets (`examples/openff`,
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
  was built to expose, and freeing the torsions is what makes the criterion work. The five conformer starts
  account for 0.55 A of that improvement, better on 15 of 16 sets.
  **What remains is the conformation, not the pose:** +0.264 against the crystal poses' +0.334,
  with a conformational residue of 0.71 A.
* **C3. Alignment inside the pocket.** `cmfqsar/docking.py` places conformations rigidly and
  `cmfqsar/complementarity.py` scores them against the pocket fields. Use complementarity as the
  prior and confidence as the likelihood: the pose must fit the pocket and sit inside the model's
  domain. Measure against the curated frames on the FEP sets, where the curated frame is the
  crystal pose and the ceiling is known.
  **Done 26 October 2026, and the combination does not help.** `cmfqsar/pocketalign.py` joins the
  two halves -- `sigma^2(pose) - w [complementarity - clash]` over the six rigid-body parameters,
  with the displacement guard of the confidence aligner -- `predict --align-variance
  --align-pocket STRUCTURE` runs it, and three posers (`pocket`, `pocket+variance`,
  `mcs+pocket+variance`) join `cmfqsar/poses.py` so that the joint objective is judged by the C1
  protocol that judges everything else (`run_poses.py --pocket`).
  **Measured on 4 FEP sets with a receptor structure** (mcl1, p38, thrombin, tnks2), held-out poses
  scrambled and recovered. From a scrambled start: confidence alone
  0.30 A median, the receptor alone 3.93 A, the
  two together 2.50 A -- **adding the receptor makes the
  confidence criterion worse on two of the four sets and better on none.** After a substructure
  frame: 0.10 A against
  0.13 A, R2pred +0.286 against
  +0.265 -- indistinguishable, at about ten times the cost,
  since the pocket term has no analytic gradient and the search falls back on Powell.
  **The control that makes this a result rather than a bug report**: at weight zero the joint
  aligner is the confidence objective optimised by a different method, and it recovers the pose
  as well as the gradient search does (0.29 A on p38 against 0.32 A). Turning
  the weight up is what makes it fail -- on p38, 0.25, 1 and 4 all give 3.5 to 3.7 A -- so what fails
  is the term and not the search.
  **Why.** Complementarity is a weak prior over six dimensions: a ligand that fits a site fits it
  in many placements, so the term adds a broad multi-modal surface to an objective that was
  already multi-modal, and its optimum is not the crystal pose -- not with a scoring function
  that has no desolvation, no entropy and no torsional search.
  So the item is built as it was specified and the specification was optimistic. What is left is
  a narrow use -- a pose that must be defensible to the receptor as well as to the model -- and a
  measurement that says the prior is too weak to improve the likelihood.
  **Not done:** the flexible version (freeing the torsions made the confidence criterion work in C2 and
  might do the same here), an analytic gradient for the pocket term, and the sets without a receptor,
  which is most of them.
* **C4. Training-set co-alignment done properly.** The present iteration (fit, realign, refit)
  contracts the ensemble and degrades one set in four. Treat it as what it is, an expectation-
  maximisation over poses, and regularise it: the displacement guard, a leave-one-out variance so no
  molecule is aligned against itself, and an annealed λ. Judge under C1 only.
  **Done 24 October 2026** (`6d85fd2`), and the result is negative. `cmfqsar/coalign.py` is the
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
  is already the reference.
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
  three of five), and **COX2 accounts for +0.187 of the +0.045 mean difference while the other four
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
  two negative results are one result seen twice.
  **Closed, negatively.** The block is correct about a molecule and its mirror image and useless
  for carrying handedness between molecules. Making it useful needs a construction anchored on
  the stereocentre rather than summed over the whole molecule: that is **C7**, opened out of this
  result, and it inherits this item's two measurements as the gate it has to pass before it is
  tried on a real set.
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
  given the answer, and on the corrosion files two laboratory-adjusted columns reconstruct the
  potency between them, which no per-column test can see -- so the outcome is judged as well.
  **Validated by following the advice**, which is the only check that matters for a
  recommendation: diagnose 70 % of a set, build what it recommends, score it on the rest. The
  advice names the best of the five routes on 21 of 39 splits and falls 0.096 short of the best
  on average; always taking the aligned model names it on 12 and costs 0.201. Per set, the correct
  unit, the advice costs 0.105 less (95 % interval -0.259 to +0.019), better on six of thirteen
  sets and worse on four: insurance against a meaningless frame, not an advantage.
  **Still hand-chosen:** the q2 < 0.1 cutoff below which the command recommends no route at all.
  It was never triggered on these 39 splits, so the validation cannot set it.
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
  **Done 26 October 2026: the first condition passes, the second fails, so the extension is
  implemented and not adopted.** `cmfqsar/anisotropic.py` carries the basis (one s and three
  Cartesian p functions per atom per width), the four closed-form overlap blocks, the fitter and
  the kernel; `tests/test_anisotropic.py` holds every overlap against numerical quadrature, which
  is how the formulas were derived, and checks that the s block reproduces the isotropic overlap
  the program already uses to twelve digits. Cartesian p rather than one p along each bond:
  three components span every direction, so the fit chooses the orientation and a bond direction
  is a special case, at the same cost.
  **The fit improves decisively.** On 4 corrosion series with finished ORCA runs
  (101 molecules, valence part of the field, `benchmarks/run_anisotropic.py`): relative error
  **84% -> 64%**, best case 61%
  (fukui_plus on 13_c1c_nH_nn1), and the correlation between the fitted kernel and the exact
  grid kernel +0.920 -> +0.977. The diagnosis the item was built on was right: what
  the isotropic basis lacked is a direction.
  **The model does not notice.** Over 8 series-field pairs the p functions change q2
  of the field alone by **-0.017** (2 of 8 better) and of the field
  added to the five ordinary fields by **-0.004**, for four times the coefficients per
  atom and sixteen times the terms per atom pair.
  **Why, and it is the same reason as for the isotropic fit:** a kernel is an inner product
  between molecules, so an error that falls the same way for every molecule cancels in it. The p
  functions recover the pi lobes above and below a ring; every molecule of a congeneric series has
  a ring in the same place; every field improves by about the same amount and the distances
  between the molecules do not move.
  So by the item's own test the extension is not adopted: the ordinary field machinery still puts
  one isotropic Gaussian on each atom, and the module is there for the case that would earn its
  cost -- a series whose members differ in the shape of a frontier orbital rather than in where
  their atoms are.
  **Not done:** d functions (the next shell, and the first that could represent a quadrupolar
  lobe without a direction); the p-type basis in the ordinary physico-chemical fields, which is
  what adoption would mean; and a series chosen because its SAR *is* orbital shape, which is the
  experiment that could still overturn this.
* **D2. Pharmacophoric indicator fields.** Sybyl-type indicator fields help only above about forty
  molecules because there are many types and each is rare. A vocabulary of six pharmacophore
  features (donor, acceptor, aromatic, hydrophobe, cation, anion) is learnable on small series and
  reads directly as a pharmacophore. Measure on the Sutherland sets against the five fields and
  against the Sybyl indicators, split by set size as before.
  **Done 26 October 2026, and the item is right -- below forty molecules.** `cmfqsar/pharmtypes.py`
  carries the vocabulary, taken from RDKit's `BaseFeatures.fdef` rather than from SMARTS written
  here, and `pharm:<feature>` is a field like `cif:<type>`:
  `FieldSet.create(pharm_features=...)`, `build --pharm` with `--pharm-min` and
  `--pharm-features`, or `--fields pharm:donor,steric`. The weight is an indicator on every atom
  of a matching group -- six for a benzene ring, three for a carboxylate -- and
  `cmfqsar/pharmacophore.py` knows the six names, so the second half of the item is literal: a
  lobe of the `pharm:donor` co-field comes out as a donor feature with its own meaning.
  **The Sutherland sets cannot answer the question the item asks**, which is worth recording
  because the item asks for them: the smallest of the eight has 66 molecules and the
  claim is about series below forty. So the measurement was repeated on the 38 Mittal
  sets, 15 of which are smaller than that (`run_pharmfields.py --mittal`).
  **Below 40 molecules (15 sets):** five +0.342, five + pharm
  **+0.368** (+0.027), pharm alone
  +0.358, five + cif +0.344
  (+0.002), cif alone +0.335. The six features beat
  the Sybyl indicators, and on their own they beat the five physico-chemical fields.
  **At or above 40 (23 sets):** five +0.299, five + pharm
  +0.305 (+0.006), pharm alone
  +0.228, five + cif **+0.345**
  (+0.047), cif alone +0.321. The order reverses, and
  the eight Sutherland sets agree with this half (+0.023 for cif against
  +0.012 for pharm).
  **The mechanism is not the one the item gave.** It blamed the rarity of individual Sybyl types,
  but `--cif-min` was already dropping those: 11 of 14 types are kept on these
  sets and the median coverage of what is kept is 88%, against 70% for a
  pharmacophore feature. What changes with set size is molecules per field --
  3.4 per Sybyl field below forty against
  6.4 per pharmacophore field, 8.9 against
  20.9 above. At three molecules per field no vocabulary can be
  estimated however common its members are; at nine the more descriptive one wins because it is
  more descriptive.
  **Not done:** the definitions are RDKit's and have not been compared with another program's;
  features are assigned on the molecule as it stands, so a cation or anion depends on how the file
  was protonated; and nothing uses the features' *centres*, which is what a pharmacophore search
  would match -- here they are atom weights like any other field.
* **D3. A cheap quantum stage.** ORCA takes two to three minutes per molecule serially, so the
  quantum fields have been tried on a few series only. GFN2-xTB through `tblite` gives charges and
  Fukui indices in seconds, which makes a quantum stage on all 86 sets feasible. The Hammett
  result says where to expect a gain (resonance) and the corrosion result says where not to; the
  question is whether the 86 sets contain any series of the first kind.
  **Done 27 October 2026, and the answer is yes for a minority of series.** `cmfqsar/xtb.py`
  and `examples/xtb_atom_properties.py`. One call of xtb per molecule
  (`--gfn 2 --vfukui`) gives the GFN2 partial charge `q_xtb`, the atomic polarisability
  `alpha_xtb` and the condensed Fukui indices, cached per molecule, failures reported and
  skipped, the geometry never relaxed unless asked. The Fukui names mean exactly what
  `cmfqsar/orca.py` means by them -- a test runs the three charge states separately and checks
  that xtb's printed f+ is q(N) - q(N+1) of its own charges -- so one `prop:` field can be
  filled by either program; `q_xtb` carries a name of its own because a tight-binding
  population is not a Hirshfeld charge. What is deliberately not computed is the ionisation
  potential and the electron affinity: the energies xtb prints for the ionic states of a Fukui
  run are electronic energies, not total ones, and `--vipea` needs a different Hamiltonian,
  which would describe one molecule with two methods. 19 tests.
  **Measured** by `benchmarks/run_xtb.py` in three stages. *Cost:* on the 40 molecules ORCA was
  run on here, **0.25 s** per molecule against **360 s**, a factor of 1437, both including the
  ionic states. The premise of the item holds. *Agreement:* over every atom of 101 corrosion
  molecules that have both, the charges correlate at +0.95 with the Hirshfeld ones, the dual
  descriptor at +0.78, f+ at +0.78 and f- at +0.71. *Gain:* over **52 series of 3511
  molecules** -- the eight Sutherland sets, the Hammett sets and the FEP collections, every
  series with at least 15 molecules -- the three reactivity fields added to the five
  physico-chemical ones raise q2 beyond the bootstrap interval of rule 4 on **7 series** and
  lower it on 1, where a 95% interval would be cleared by about 1.3 of 52 by chance. The median
  change is +0.014. The largest gains are tnks2 +0.181 [+0.061, +0.330], thrombin with the nozob
  ligands +0.129 [+0.025, +0.238], mcl1 +0.092 [+0.020, +0.164], ACE +0.067 [+0.033, +0.107],
  the symmetrised Hammett set +0.040 [+0.023, +0.058] and DHFR +0.042 [+0.026, +0.059] on 361
  molecules. The Hammett sets were the positive control and they gain, which is what the item
  predicted.
  **What this settles.** A quantum field helps a minority of series and costs the rest a little,
  and the cheap stage is good enough to find which. That could not be known before, because the
  expensive stage could not be run on more than four series.
  **Not done:** the 46 FEP series with fewer than 15 molecules, which cannot carry a 10-fold
  cross-validation; the application has no dialog for xtb as it has for ORCA; a solvent model
  (`--alpb` is implemented and was not measured); and the per-atom polarisability `alpha_xtb` is
  written but was not put in a field of its own in the comparison.
* **D4. Protonation and tautomer ensembles.** The corrosion series in HCl are modelled in one
  protonation state. The mean-map kernel needs only bilinearity, so an ensemble over protonation
  states weighted by pKa is the same mathematics as a conformer ensemble; but `ensemble_fields` in
  `cmfqsar/fields.py` tiles one atom list over the members (`atoms_per_conformer`), and the atom
  contributions and the GUI read it that way. Generalise the builder to members with their own
  atoms, with a conformer ensemble as the special case that must reproduce the present numbers
  exactly, then measure on `examples/corrosion_hcl` against the single state.
  **Done 26 October 2026. The builder works, the ensemble does not pay here, and the reason is the
  pKa model rather than the ensemble.** `combine_members` in `cmfqsar/fields.py` builds one field
  from members that have their own atoms, with `member_sizes`, `member_atom_map` and `member_kind`
  beside the `atoms_per_conformer` the conformational case keeps; `per_atom` follows the map and
  `unmapped_share` reports what it could not place, so nothing is lost silently. A test asserts
  the gate the item sets: a conformational ensemble built through the general builder reproduces
  the tiled one atom for atom and kernel for kernel. `cmfqsar/protonation.py` enumerates the
  states with Dimorphite-DL's patterns and weighs them by the Henderson-Hasselbalch fraction at
  each site's regression pKa, on the molecule's own heavy-atom geometry, so an aligned series stays
  aligned; `cmfqsar/tautomers.py` does the same for tautomers. `build --ph 0`, `--tautomers score`.
  **Measured on all 23 series of `examples/corrosion_hcl` (997 molecules), same fields,
  grids, folds and scaffold alignment as `run_corrosion.py`:** the state the file gives
  +0.324, the likeliest state by pKa +0.260, the pKa-weighted
  ensemble +0.261, the tautomer mean +0.314. The file's
  state is best on 13 of the 23 series.
  **Two numbers explain that.** At pH 0 only 1% of the molecules
  are a real mixture -- a pKa two units from the pH settles the question -- so the averaging itself
  is worth +0.001 and the whole difference is the choice of state.
  And that choice is wrong more often than the curator's: the likeliest state by pKa carries the
  file's charge for 52% of the molecules, with a mean charge of
  +1.57 against +0.77, because
  independent sites protonate a second and a third basic nitrogen that would not both carry a
  proton. **Nor is it the lack of mixtures.** The same measurement at pH 3, where
  83% of the molecules are mixtures with 3.0 states on average, gives
  +0.324 for the file's state, +0.255 for the likeliest and +0.246 for the ensemble --
  -0.009 for the averaging itself. The ensemble mathematics is sound and cheap; the pKa input
  is what fails.
  **Not done:** the two kinds of ensemble cannot be combined with a conformational one (the program
  refuses rather than build the product); site-site interaction is ignored, which is exactly the
  error the mean charge shows; the pKa values cannot be replaced from the command line, so a set
  with measured pKa values cannot yet be modelled with them (the tautomer path does accept given
  energies); the tautomer weights are an assumption, not populations, and would need the energies
  D3 would give; and neither option is in the GUI, only on the command line.
* **D5. Learned increments.** The two parameterisations differ by 0.20 in q² on AChE, which says
  the hydrophobic and hydrogen-bond increments are not settled. Fit the per-type increments jointly
  with the model (gradient of the cross-validation loss through the kernel, with the published
  increments as the prior and a penalty on the distance from them), and ask the only question that
  matters: do increments learned on one set transfer to another? If they do not, the result is that
  the increments are a per-set nuisance parameter and the plan says so.
  **Done 26 October 2026, and they do not transfer: the increments are a per-set nuisance
  parameter.** `cmfqsar/learnincrements.py` uses the one property that makes this affordable: an
  increment gives every atom of a type the same weight, so the kernel is a quadratic form in the
  increments and the tensor of overlap integrals summed over type pairs does not depend on them.
  It is built once per attenuation (25 types on these sets, the largest tensor 380 MB),
  every candidate kernel is one contraction, and the gradient of the cross-validated squared
  error -- the number the model is selected on, with the bias term of Eq. 12 -- is exact through
  the fold solves and the cosine normalisation. Tests compare each link with finite differences
  and the quadratic form with the kernel the program builds the ordinary way. `build
  --learn-increments`, `--increments FILE`, `--save-increments FILE`.
  **Measured on the eight Sutherland sets, on their published train/test split:** published
  increments q2 +0.596 and R2pred **+0.383**; fitted on the set
  itself q2 **+0.816** and R2pred +0.310; fitted on another set and
  transferred here +0.363, the best of the seven donors
  +0.449. Fitted increments predict better than the table on 2
  of the 8 sets, transferred ones on 3. The best-of-seven column is picked
  with the test set in hand, so it is an upper bound rather than a rule.
  **The fitted vectors do not agree with one another**, although every fit starts from the same
  table: the mean cosine between two sets' fitted increments is +0.89
  (hydrophobic), +0.72 (acceptor), +0.54 (donor), and those figures
  are generous because every fit starts from the same table. The fit moves the two hydrogen-bond
  fields much further than the hydrophobic one (+0.99, +0.86,
  +0.70 mean cosine with the published values), which is where the two published
  parameterisations differ most. How far a set moves them does not predict what it loses:
  the correlation between the two over the eight sets is -0.25.
  **Two things the measurement needed.** Only the relative increments of a field are identifiable,
  because a cosine-normalised kernel does not change when they are all multiplied by the same
  number; the fitted values are reported at the published length, which changes no kernel. And the
  penalty cannot be chosen on the cross-validation being minimised, which always prefers the
  freest fit, so it is chosen on training molecules the fitting does not see (the chosen values
  range over 0.3, 30, 300 across the sets).
  **Not done:** the fit is at one attenuation and ridge, the ones the published model selects, not
  jointly with them; the increments of the steric and electrostatic fields are not fitted; and
  nothing in the GUI fits them.
* **D6. Per-field widths.** Follows from B4: once the evidence selects hyper-parameters, α per field
  is five numbers instead of one, and the co-fields become sharper for the field that needs it.
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
  the criterion prefers it, and the held-out molecules show no difference. Read together with B4 — where
  two criteria disagreed about the shared width and predicted equally well — this says the
  attenuation surface is flat around its optimum, and that the search is not what decides a CMF model's
  accuracy. The flag stays for asking the question of a new series; it is not a default,
  and nothing else in the plan should be built on the assumption that a finer α grid helps.
  The co-field half of the item is untouched: whether a per-field width makes a co-field easier to
  read is a question about interpretation, not accuracy, and it needs visual judgement rather than a
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
  **Done 27 October 2026.** `cmfqsar/trajectory.py` and `examples/trajectory_ensemble.py`.
  `read_trajectory` is the one function the item asks for and does the five steps in order:
  read through mdtraj or MDAnalysis (XTC, DCD, NetCDF, TRR, multi-model PDB), strip the solvent
  and the ions, superimpose every frame on the binding-site backbone when the topology holds a
  receptor and on the ligand itself when it does not (the result says which), thin the frames by
  leader clustering with the radius raised until the cap is met, and return one RDKit molecule
  whose conformations are the representatives and whose `conformer_weights` property is their
  cluster populations. The weights are populations and never Boltzmann factors, for the reason
  the item gives; a test checks that no energies are written, so none can be used by mistake.
  Force-field charges are written as `q_mm` where the topology carries them. The script writes
  an SDF of one record per conformation under one title, which `--ensemble-by name` collapses
  again, so the whole path from a trajectory to a fitted ensemble model is one command line.
  44 tests, on trajectories the tests build themselves, so they need no simulation package and
  no downloaded data.
  **Measured** by `benchmarks/run_trajectory.py`, on what belongs to the reader rather than to
  D8. *The two readers agree*: on 36 trajectories read by both, the largest difference in any
  coordinate is 4e-7 A, so which one is installed does not change a result. *The superposition
  works*: a system whose frames differ by nothing but a rigid rotation collapses to a single
  conformation, which is the test that would fail if the binding-site fit did not. *The
  thinning* trades accuracy for time in a straight line with no free radius: over 36 molecules
  of 60 frames the joint kernel is 1.0e-02 from the all-frames one at 0.25 A (34 conformations
  kept, 87 s), 4.7e-02 at 1 A (21 kept, 44 s) and 1.2e-01 at 3 A (2 kept), against 176 s for all
  sixty. *The cost* follows the square of the number kept: 27 times as many conformations is 658
  times the kernel time against 734 for an exact square, which is why the thinning is inside the
  reader.
  **Corrected 28 October 2026 by D8, which found two faults in this reader.** Both would have
  produced a confident false negative there. The superposition had **no common frame**: each
  trajectory was fitted to its own first frame, which is that run's equilibrated protein, so
  several trajectories of one series came back in as many frames as there were molecules;
  `read_trajectory` now takes a `reference`, the one structure every run was built from, and
  checks atom by atom that it describes the same protein. And **periodic images were ignored**:
  a trajectory is written with every atom wrapped into the cell, so a ligand that crossed a face
  appeared a box length away -- 17 of 25 mcl1 ligands looked 58 A from their crystal poses, all
  at the same distance, which is the box size and not chemistry. `make_whole` puts a split
  molecule back together and `reimage` brings the ligand to the image beside the protein, both
  before anything is fitted, and both are tested on constructed cases. With the two fixed the
  ligands sit 0.93 A from their crystal poses in the median, which is what a 2 ns simulation of
  a bound ligand should give. 48 tests.
  **Not done:** a dialog in the application, and a reader for a trajectory too large to hold in
  memory (the stride is the only control over that today).
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
  **Done 28 October 2026, and the answer is no.** `cmfqsar/md.py` builds and runs one
  ligand-in-pocket simulation, `examples/make_md_trajectories.py` is the whole recipe from the
  entry code to the ensemble file, `benchmarks/run_md.py` is the comparison and
  `examples/md/mcl1_md.sdf` is the data it needs, so the table rebuilds on a machine with no
  simulation software.
  **What was run.** The 25 ligands of `examples/openff/mcl1.sdf` in the site of PDB entry 4HW3,
  2 ns each at 300 K and 1 bar after 200 ps of equilibration with the alpha carbons restrained,
  GAFF-2.11 with AM1-BCC charges, Amber ff14SB, TIP3P water and 0.15 M of ions. 20,284 atoms,
  658 ns/day on one consumer GPU, 351 s per ligand including the charges. Two deviations from
  the item's wording, both deliberate: the **whole protein** rather than the trimmed pocket of
  `examples/pockets/`, because every cut leaves a broken peptide bond that preparation turns
  into a charged terminus; and **only the chain lining the site**, because 4HW3 holds twelve
  copies of Mcl-1. The ligands stayed in their poses -- 0.93 A from the crystal pose in the
  median, 1.39 A at worst -- so all 25 are kept and the measurement is about an ensemble around
  a correct pose rather than about finding a pose.
  **Measured.** Six models, same molecules, same folds, same grids, each against the crystal
  pose with the interval of rule 4: the crystal pose **+0.575**; generated conformations +0.556
  (-0.019 [-0.271, +0.279]); the trajectory with clusters equal +0.391 (-0.184 [-0.526,
  +0.136]); by population +0.180 (-0.395 [-0.830, -0.053]); multi-instance +0.463 (-0.112
  [-0.384, +0.169]); and, as a control on the method rather than the science, the same frames
  with each trajectory in its own frame +0.277.
  **What it settles.** No reading of the trajectory beats the crystal pose. The best is
  indistinguishable from it and the population weighting is behind it beyond the interval. The
  control says this is not about molecular dynamics in particular: generated conformations
  change nothing either, which reproduces D6's fourth-dimension result on a set where the pose
  is known. Around a pose that is already right, averaging over the ligand's motion adds no
  information and weighting that average badly subtracts some. The multi-instance search chose
  beta = 0, the plain ensemble, so no frame was worth singling out. **Therefore D9 does not
  follow from this result**, which is what the item said; D10 was built earlier on conformer
  ensembles and its own verdict stands.
  **Two findings about the method, not the question.** How the clusters are weighted matters
  more than whether the trajectory is used: equal weights and weights by population differ by
  0.211, which is larger than the 0.184 between the better of them and the crystal pose. That
  contradicts the physical argument for population weights in `cmfqsar/trajectory.py` and the
  contradiction is reported rather than resolved. And the common frame is what makes any of this
  measurable: 0.114 of q2 is lost when each trajectory is fitted to its own first frame instead
  of to the one structure every run was built from, a misalignment of a few tenths of an
  angstrom. A field kernel notices less than part C is about.
  **Two faults in D7's reader were found by this item and fixed**, both of which would have
  turned into a confident false negative: the superposition had no common frame across
  trajectories, and periodic images were ignored, which made 17 of 25 ligands appear 58 A -- one
  box length -- from their poses. `read_trajectory` now takes a `reference`, and `make_whole`
  and `reimage` undo the wrapping before anything is fitted; both are tested on constructed
  cases. 48 tests for the reader.
  **Two more sets, 29 October 2026, and the answer is not the same on every set.** The
  trajectories built for D9 cover two further free-energy sets, so the same five models were
  measured on them with no new simulation: **p38**, 29 ligands in PDB entry 3FLY, and **hif2a**,
  36 ligands in 5TBM. One ligand of hif2a has no trajectory, because its charges could not be
  computed, so 36 of 37 are compared.
  **p38**, crystal pose +0.261: conformations +0.082 (-0.178 [-0.414, +0.182]), clusters equal
  +0.239 (-0.022 [-0.310, +0.299]), by population +0.091 (-0.169 [-0.402, +0.181]),
  multi-instance +0.455 (**+0.195 [-0.133, +0.522]**, beta = 1).
  **hif2a**, crystal pose +0.342: conformations +0.311 (-0.031 [-0.246, +0.433]), clusters equal
  +0.471 (+0.129 [-0.076, +0.365]), by population +0.534 (+0.192 [-0.068, +0.561]),
  multi-instance +0.520 (**+0.178 [+0.003, +0.465]**, beta = 0).
  **What that changes.** On mcl1 no reading of the trajectory was ahead of the crystal pose. On
  these two the best reading is about +0.19 ahead, and on hif2a one reading clears the interval.
  So "the answer is no" was a statement about mcl1 and not about the method, which is what one
  set can say. It is also the first time the multi-instance search chose a beta above zero: on
  p38 it chose beta = 1, so the model did concentrate the weight on some frames.
  **What the three sets support together is weak, and the item says so.** Nine readings of a
  trajectory were compared with their crystal pose over the three sets and one of them clears
  its interval, with the near edge +0.003 from zero, which is about what nine comparisons produce
  on their own. That one reading is also the uniform ensemble: at beta = 0 the multi-instance
  fit weighs every conformation alike, and "clusters equal" on the same set reaches +0.129
  [-0.076, +0.365] and does not clear. The two are the same model.
  **What separates them is where the kernel is normalised**, measured with `--normalisation` at
  one attenuation and one lambda on all three sets, because a result rests on it: the ensemble
  path normalises the kernel of the whole ensemble, the multi-instance path normalises every
  conformation and then averages. The gap between the two paths is 0.019 to 0.049 of q2 and
  falls to 0.002 to 0.015 when the normalisation is switched off, so most of it is the
  normalisation; up to 0.015 is left over, so not all of it. Both places are defensible, which
  is why a gain of this size can come from the choice alone.
  **Where that left the gate.** Read as the best of three readings, the gate of this item was no
  longer a plain no: on hif2a some reading was ahead beyond the interval, weak as that reading
  is. The paragraph after the next one fixes the reading in advance instead, and closes the gate
  as a no again. D9 was built in any case, at the author's instruction, and failed on its own
  evidence -- two readings of its definition sweep were written down beforehand and both failed
  on a set they were not read from -- so nothing here changes anything about D9.
  **The own-frame control on all three sets, 29 October 2026, and it does not say what one set
  said.** The control reads the same frames with each trajectory superimposed on its own first
  frame instead of on the one structure all the runs of its set were built from, so every ligand
  sits in a frame of its own. It now exists for the two new sets as well, built from the
  trajectories already on disk, and the change it causes has an interval of its own, taken
  against the common-frame ensemble on the same molecules and with the same weighting:
  **hif2a** +0.471 to +0.297 (**-0.175 [-0.391, -0.020]**), **mcl1** +0.391 to +0.277 (-0.114
  [-0.386, +0.081]), **p38** +0.239 to +0.368 (+0.129 [-0.207, +0.369]).
  **Only hif2a loses beyond the interval.** On mcl1 the drop of 0.114 reported above does not
  clear the interval, now that the control has one; on p38 the own frame reads 0.129 *higher*
  than the common one. So the sentence above -- that the common frame is what makes any of this
  measurable -- was a reading of one set, and these three do not support it as a statement about
  the method. What an alignment of a few tenths of an angstrom is worth is decided set by set
  here. On p38 the control is also +0.107 [-0.351, +0.498] above the crystal pose, which is one
  more reason to read the gains of that set with care: there a reading built to be wrong does no
  worse than the pose it was meant to lose to.
  **The control is kept out of the counting.** It is not one of the nine readings compared with
  their crystal pose, and a set is not counted as ahead because the control clears an interval;
  `CONTROLS` in `benchmarks/run_md.py` is what leaves it out, with a test for each. A reading
  built to be wrong must make the evidence look neither stronger nor weaker than it is. An
  own-frame run also no longer overwrites `examples/md/provenance.json`, which describes the
  common-frame ensemble and not this control.
  **One reading fixed in advance, 29 October 2026, and it answers the item: no gain.** Every
  number above is the largest of three readings of the same frames, which is what makes a single
  interval at the edge of zero so weak. The three sets are therefore read once more with one
  reading, the plainest: the trajectory with every conformation weighing the same. It contains
  no choice -- there is no weighting to pick and no beta to search -- and it is not the largest
  gain on any of the three sets, so the rule does not name a winner. **hif2a** +0.342 to +0.471
  (+0.129 [-0.076, +0.365]), **mcl1** +0.575 to +0.391 (-0.184 [-0.526, +0.136]), **p38** +0.261
  to +0.239 (-0.022 [-0.310, +0.299]). None of the three clears its interval.
  **Combined over the three sets: -0.006 [-0.157, +0.163]**, over 90 molecules, each set
  weighted by the molecules it contributes and the molecules resampled inside their own set
  because three targets do not share an activity scale. So a trajectory ensemble around a
  crystal pose is worth nothing once the reading is fixed first, and the interval is now narrow
  enough to say so rather than to leave it open: 0.32 of q2 wide over 90 molecules, against 0.66
  on the 25 molecules of mcl1 alone. **This is the reading the item should be read as giving, and the gate is a no.** The
  +0.003 edge on hif2a is what choosing the best of three readings produces.
  **What this is not.** It is not a pre-registration: the sets had been measured before the rule
  was written down, and no rule written afterwards undoes that. What it avoids is the other
  fault, deciding which of several readings to believe after seeing which one won. The
  attenuation and lambda are still chosen by cross-validation inside every model, which the
  crystal pose gets in the same measure, so that choice is not what separates the columns. The
  combined number pools three targets, which is why the three sets are also given on their own.
  **A set whose poses are not known, 29 October 2026, and the ensemble is worse there rather
  than better.** Every number above starts from a crystal pose, which the item itself names as
  the gap. mcl1 was therefore turned into the other case without a new set:
  `examples/make_docked_poses.py` poses every ligand while it is held out. The pose is destroyed
  -- a random rigid placement within 2 A of the site -- and then the pocket docking of C3 puts it
  back where the receptor's own fields prefer it, with no model, no activity of its own and no
  crystal frame of any kind, which is the protocol C1 defines. Each ligand was then simulated
  from the pose that was found, 2 ns on the same receptor frame as everywhere above, and read in
  the same three ways.
  **What the posing produced is two groups rather than a spread.** The 25 poses sit a median
  1.12 A from the crystal poses, but the median hides the shape: 14 ligands are within 1.3 A,
  and on the other 11 the docking missed the site, 4.0 to 8.3 A away. So this test is harder
  than its median suggests, and a reading of it is also a reading of how often that docking
  fails.
  **Measured**, every model against the pose a modeller would actually have -- the one that was
  found -- and then against the crystal pose as the ceiling: the crystal pose +0.575; the
  **found pose +0.402** (-0.173 [-0.567, +0.144] against the crystal pose, which does not clear
  the interval); the trajectory from it with **clusters equal -0.040** (-0.442 [-1.461, +0.272]
  against the found pose, -0.615 [-1.470, -0.107] against the crystal pose); by population
  +0.024 (-0.378 [-0.833, +0.185]); multi-instance -0.012 (-0.413 [-1.134, +0.198], beta = 1).
  **What it settles: the ensemble does not cover for a pose that is wrong.** It is 0.44 of q2
  below the single pose it started from, and it is below the crystal pose beyond the interval,
  which the found pose itself is not. A pose error of a median 1.12 A costs 0.173 of q2; two
  nanoseconds of motion around those poses cost 0.442 more. The multi-instance fit chose
  beta = 1 here, which is the second time it put weight on some frames rather than none, and it
  did not recover the loss. Read strictly, the comparison against the found pose does not clear
  its own interval, because 25 molecules and an ensemble this scattered make a wide one; the
  statement that does clear its interval is the one against the crystal pose. Either way the
  direction is the opposite of the one this item expected, so the gap is answered and not left
  open.
  **Did the simulation repair the pose?** Partly, and not in a way the model could use. The
  nearest kept conformation of a ligand is a median 0.79 A from its crystal pose, against 1.12 A
  for the pose the run started from, but the nearest of 11.4 conformations is a smallest of
  several and not a repair, and the middle conformation sits at 1.51 A, farther out than the
  start. The 11 ligands the docking placed wrongly did move in, from a median 5.17 A to 3.70 A,
  and none of them came back under 2 A in 2 ns; the 14 that started near drifted out a little,
  from 0.95 A to 1.21 A. The cost was 3.4 hours of one consumer GPU for the 25 simulations and
  232 s for the measurement.
  60 tests for this item.
  **Not done:** one force field and one length on all three sets; poses that are not known on a
  second set, and with a docking that does not miss the site on 11 ligands of 25, since the
  result above is as much about that docking as about the ensemble; why the own frame costs q2
  on hif2a and not on the other two, which is a question about the sets rather than about the
  method; a combined estimate that treats the sets as a sample of
  targets rather than weighting them by size; and the flexible-series question the item names,
  `mittal/AMPH1`, which the item made conditional on a set showing something and which the
  best-of-three reading permits but this one does not support.
* **D9. Solvent as a field.** The 2015 indicator-field formalism accepts any continuous density,
  and a water occupancy map from the trajectory is one. Fit the occupancy by Gaussians on a grid
  of hydration sites (the fitting machinery of `cmfqsar/probes.py` does this for probe fields) and
  add it as a sixth field with its own mixing coefficient, so that the co-field says where
  displacing or keeping a water helps. It is the first CMF field that is not centred on an atom of
  the molecule, which is what D4's `combine_members` builder, now in place, allows.
  **Done 28 October 2026, against the gate of D8, at the author's instruction.** D8 said this
  item does not follow if a trajectory does not beat the crystal pose, and it did not. It was
  built anyway because the quantity is not the same one: D8 averaged the *ligand's* motion into
  a mean field, while this uses the trajectory for something the ligand's field cannot express
  at all, namely where the water sits. The measurement below is therefore its own, with the five
  fields as the reference of rule 1 rather than anything inherited from D8.
  **What it is.** `cmfqsar/solvent.py`. For every point of a grid around the ligand, the share
  of frames in which a water oxygen lies within 1 A of it: an occupancy in [0, 1] whose reading
  needs no normalisation, with bulk water at about 14% of it. The peaks above 50% are taken
  greedily, strongest first, no two closer than 2.4 A, so a site is a place a water occupies and
  not a cluster of molecules; the coefficients are then fitted to the whole map by a
  **non-negative** least squares rather than read off the peak heights, because an occupancy
  cannot be negative and a negative coefficient would read as a place water avoids. The result
  is an ordinary `MoleculeFields` whose atoms are hydration sites, so the existing kernel code
  takes it unchanged -- the same arrangement `cmfqsar/fluctuation.py` uses for D10. It cannot
  share a table with the molecule's own fields, because a table holds one set of atoms and a
  hydration site is not one of the molecule's, so it is a block of its own mixed in with one
  coefficient. On mcl1 it gives 7.8 sites per ligand, from 4 to 12, the strongest averaging 73%.
  30 tests, the two that matter being that a water held at a known place in every frame raises a
  site there within 1 A and that a water which moves every frame raises none.
  **Measured** by `benchmarks/run_solvent.py`, every model at its own best attenuation so that
  the reference is not held back by a choice made to suit the water: the five fields alone
  **+0.575**, the five fields and the water +0.560 (**-0.015 [-0.178, +0.186]**), the water
  alone +0.263, random sites alone -0.028, one shared set of sites alone -0.138.
  **What it settles, and it is not nothing.** The water field does not improve the model, and
  the interval is too wide for 25 ligands to show a small gain either way. But the water field
  **on its own** reaches +0.263 while both controls are below zero, which is what a block that
  cannot tell the ligands apart must give. So at this definition of a site, where the water sits
  does carry information about activity; it is information the ligand's own fields already
  carry, since adding it gains nothing. That is a sharper statement than D8's gate anticipated,
  and it is the reason the item was worth building against the gate.
  **How much the definition decides, 28 October 2026**, measured with `--sweep` over the two
  numbers that define a site -- the share of frames a point must be occupied and how close two
  sites may be -- because neither follows from anything. The two conclusions come apart. *The
  first is firm:* of the sixteen definitions that leave every ligand a site, none beats the five
  fields beyond the interval and none falls behind, the change running from -0.130 to +0.087.
  *The second is not:* the water alone runs from **-0.124 to +0.607** and is not positive at
  three of the sixteen. The sensitivity has a direction -- averaged over the spacings the water
  alone goes 30% +0.080, 40% +0.052, 50% +0.197, 60% +0.568 -- so the signal sits in the few
  places a water really occupies and weakly occupied points dilute it. At the strictest
  threshold that still leaves every ligand a site, 60% with 3.6 sites per ligand, the water
  alone reaches +0.554 to +0.607 against a control of -0.096 to -0.055, at all four spacings.
  **Two cautions the sweep itself produced.** At the loosest definition tried, 30% occupancy and
  3.6 A spacing with about 25 sites per ligand, the **random-site control reaches +0.438 against
  -0.022 for the measured sites**: a block of two dozen plausible numbers carries what a block
  of two dozen random numbers carries, so a claim about positions is only worth making where the
  control is beaten. And +0.607 is the largest of sixteen values tried on 25 molecules after
  seeing them and is not an estimate of anything; the defaults, 50% and 2.4 A, were fixed before
  this was measured and are left alone. One fault of mine was found on the way: a site's
  coefficient is fitted against Gaussians of one width, and the stored file fixed that width at
  0.5 whatever width the kernel then used, so the file now carries a coefficient for every
  attenuation of the grid and the two tables agree where they overlap.
  **The co-field** over the sites is in the table: the sites of every ligand gathered into
  places, each with the weight the block puts on it, positive for more water and negative for
  less. Under rule 5 it is a display tool, because the block does not improve the model.
  **A held-out second set, 28 October 2026, and the pre-registered claim failed.** The strict
  end of the mcl1 sweep was the largest of sixteen values tried, so it was put to a test that
  could fail: the definition was fixed at **60% occupancy and 2.4 A** before anything was run,
  the protocol was left unchanged, and the whole chain ran on **p38** -- 29 ligands, receptor
  3FLY, one chain of 3,113 atoms, 67,423 atoms solvated, 273 ns/day, 13 minutes a ligand, all 29
  ligands keeping their poses. At that definition the water alone reaches **-0.043** against a
  random-site control of **+0.002**: nothing at all, where mcl1 gave +0.554. The number did not
  transfer and would have been wrong to quote.
  **The sweep on p38 explains it and finds what mcl1 did not.** At 60% p38 keeps **11.4 sites a
  ligand** where mcl1 kept 3.6, because its site holds far more ordered water, so the same share
  of frames is a much looser filter there. At **70%**, leaving 6.3 to 6.7 sites, the water alone
  reaches **+0.312 to +0.341** against controls near -0.02, and the mixture reaches +0.536 to
  +0.560: **+0.276 [+0.037, +0.560] to +0.299 [+0.063, +0.593]** against the five fields, which
  **clears the interval** at all four spacings. Those are the only definitions anywhere in this
  item where a water block has cleared it, and they also overturn the first conclusion as a
  general statement: on mcl1 no definition beat the five fields, on p38 four do.
  **What the two sets appeared to agree on, stated as a count rather than a threshold.** The
  threshold looked like a proxy for how few sites survive, and the proxy is target-specific. In
  terms of the count, both sets appeared to say the same: the water field carries something where
  the definition leaves roughly four to seven strongly occupied sites a ligand (mcl1 +0.554 at
  3.6, p38 +0.312 to +0.341 at 6.3 to 6.7) and nothing where it leaves ten or more. That was
  formulated after seeing both sets, so it was a hypothesis with two observations behind it and
  no estimate.
  **A third set, with the definition fixed as a count, 28 October 2026, and the hypothesis
  failed.** A count is testable where a threshold is not, because a count means the same thing
  on two targets. The definition was written down before anything was run: the **five most
  occupied sites a ligand**, an occupancy floor of 30% so that the count and not the floor
  decides, and the module's spacing of 2.4 A. The set was **hif2a**, chosen as the largest one
  available with the smallest receptor: 37 ligands, chain A of entry 5TBM, 1,749 atoms after
  preparation, 22,717 atoms solvated, about 540 ns/day, 5.3 minutes a ligand, 235 minutes in
  all; `antechamber` could not charge lig_43, so 36 ligands remain. The two earlier
  sets were measured again at the same count from the trajectories they already had.
  **The three rows**, each model at its own best attenuation, the change against the five
  fields with the interval of rule 4: **hif2a** (held out) 36 ligands, the five fields +0.342,
  with the water +0.262, **-0.079 [-0.287, -0.007]**, the water alone +0.058, random sites
  +0.177, one shared set -0.078. **mcl1** +0.575, +0.455, -0.120 [-0.316, +0.074], alone +0.172,
  random -0.134, shared -0.138. **p38** +0.261, +0.267, +0.006 [-0.100, +0.133], alone -0.043,
  random +0.089, shared -0.051.
  **What the test says.** On the held-out set the water alone reaches +0.058 while the same five
  sites moved at random reach +0.177, so the measured positions do worse than random ones; and
  the mixture is the first case in this item where adding the water makes a model **worse beyond
  the interval**. The count also loses each earlier set its own result: mcl1 +0.172 where its
  60% floor gave +0.607 at 3.6 sites, p38 -0.043 where its 70% floor gave +0.341 at 6.3 sites.
  A reformulation that loses the result on both of the sets it was formed on is a mistake in
  reading the sweep, not a property that failed to transfer to a third target.
  **Why the count was the wrong variable.** The random-site control keeps each ligand's number
  of sites and moves them, so the count was already held fixed in every comparison in this item:
  at mcl1's 60% floor the measured sites reach +0.554 and that control -0.096, which places the
  difference in the positions of strongly occupied sites and not in how many there are. What the
  two working definitions have in common is the floor itself, that every site kept is occupied
  in most frames; how many survive follows from the floor and differs between targets, 3.6 on
  one and 6.3 on the other. Fixing the count does the opposite: a ligand with one strongly held
  water is made to take four weak places as well, so a count behaves like a loose threshold,
  which is the end of the sweep where the field carries least.
  **The honest status of the item.** A water field does not improve a model at any definition
  that can be fixed in advance, and at one such definition on one set it makes the model
  measurably worse. A high occupancy floor is the only setting in which the field carries
  anything, the floor that works differs between targets, and nothing here explains why. Across
  the two sweeps 36 definitions were tried and one cleared the interval, which is also what
  trying 36 definitions produces on its own. The field and its co-field stay in the program as a
  display tool under rule 5.
  **The rule behind those definitions, tested as a rule, 28 October 2026, and it failed too.**
  One statement survived the count: both definitions where the water alone carried anything
  looked like the **strictest occupancy floor that still leaves every ligand at least one
  site**, 60% on mcl1 and 70% on p38. That is a rule and not a number, so it can be carried to
  a new set, and it reads only the occupancy maps -- no activity and no q2 -- so the choice
  cannot be a search in disguise. `strictest_floor` and `stage_rule` in the benchmark do exactly
  that: raise the floor in steps of 10% while every ligand still has a site, then measure the
  five models once.
  **The three rows.** **hif2a** (held out), floor 60%, 4.6 sites a ligand: the five fields
  +0.342, with the water +0.396, +0.054 [-0.216, +0.330], the water alone **-0.074**, random
  sites **+0.277**. **mcl1**, floor 60%, 3.6 sites: +0.575, +0.617, +0.042 [-0.190, +0.299],
  alone +0.554, random -0.096. **p38**, floor 80%, 3.0 sites: +0.261, +0.301, +0.040 [-0.176,
  +0.220], alone +0.169, random +0.100.
  **The rule is refuted by its own evidence, before the held-out set is consulted.** On p38 it
  picks 80% and not the 70% the sweep had found, and at 80% the water alone reaches +0.169
  against a control of +0.100: nothing. The cause is a mistake in reading the sweep: its grid
  stopped at 70%, so a floor that was the strictest **tried** was taken for the strictest
  **possible**. Applied as stated, the rule walks past the definition the sweep had found. It
  reproduces its evidence only on mcl1, where 70% does leave seven ligands without a site.
  **It failed on the held-out set as well.** At the 60% floor the rule picks for hif2a the water
  alone reaches -0.074 while the same sites moved at random reach +0.277 -- the control ahead by
  a wide margin, the same inversion the loosest definitions of the sweep produced. The mixture
  is +0.054 [-0.216, +0.330], and every interval in the table contains zero.
  **Where this leaves the item.** Two readings of the sweep were written down and tested on a
  set they were not read from, and both failed. A water field does not improve a model at any
  definition that can be fixed in advance. The one place in this item where a water block
  cleared the interval, p38 at a 70% floor with +0.276 [+0.037, +0.560], cannot be reached by
  any rule stated beforehand, and it is one result from 36 definitions tried on two sets, which
  is about what 36 attempts produce on their own. The field and its co-field stay in the program
  as a display tool under rule 5.
  **Not done:** the hydrogens of the water are ignored, so the field says where a water is and
  not which way it points; there is no control for the simulation length, which is what would
  say whether 2 ns places the sites surely enough; and no definition of a hydration site has
  been found that carries the same meaning on two targets, which is what the two failed tests
  were looking for.
* **D10. Fluctuation as information.** The ensemble mode uses the mean field only. The variance of
  the field across frames is a flexibility map, and the kernel of two variance fields is in closed
  form by the same algebra as the mean (a product of Gaussians is a Gaussian). Add it as a block
  with its own coefficient under multiple kernel learning, and let the mixing coefficient say
  whether it is worth anything; rule 1 applies, and the expectation is that it is a display tool.
  **Done 27 October 2026, and the expectation in the item was right: it is a display tool.**
  `cmfqsar/fluctuation.py`. The variance of a field across the conformations of an ensemble is a
  field of the same class, which is the whole point: every term of
  ``(1/k) sum_m X_m^2 - ((1/k) sum_m X_m)^2`` is a product of two Gaussians, and a product of two
  Gaussians is a Gaussian, so the variance is a sum of Gaussians at the midpoints of atom pairs,
  at twice the width, with a coefficient that falls off with the distance between the two atoms.
  `variance_field` builds it, `variance_fieldset` doubles the widths it has to be evaluated with
  and refuses a field whose width is not the same on every atom (the steric field, b = alpha/R^2,
  cannot be written this way), and `kept_share` reports what the pair floor dropped. 11 tests,
  the first of which is the one the item rests on: evaluated on a grid, the field is the variance
  across the frames to a part in a thousand.
  **What it costs.** k conformations of n atoms give k^2 n^2 pseudo-atoms before the floor, so the
  field grows with the square of the molecule and the kernel with the fourth power. On the sets measured here that is 1065 to 2119 pseudo-atoms per molecule at an attenuation of 2 and a floor of 0.01, keeping 99% of the pair weight, and a kernel over a set takes 286 seconds at most. That is
  affordable on sets of this size and is the reason the measurement below uses three conformations,
  heavy atoms and an attenuation of 2 rather than the defaults.
  **Measured** by `benchmarks/run_fluctuation.py` on 3 aligned sets, the mean field as the
  reference of rule 1 and the same conformations, folds and ridge search for every row:
  steroids (31 molecules) mean +0.794, variance alone +0.425, best mixture +0.785 at 0.10, -0.009 [-0.079, +0.062]; GPB (66 molecules) mean +0.460, variance alone +0.180, best mixture +0.462 at 0.10, +0.002 [-0.015, +0.020]; THR (88 molecules) mean +0.308, variance alone -0.013, best mixture +0.302 at 0.10, -0.006 [-0.032, +0.023].
  **What the variance carries by itself runs out as the series grows**: alone it reaches
  q2 +0.425 on the 31 steroids, +0.180 on the 66 of GPB and -0.013 on the 88 of thrombin,
  something on a small set of one rigid scaffold and nothing on a larger and more varied one. Under rule 1 that closes the
  item as the plan expected: **keep it as a map, not as a block**. A flexibility map answers a
  different question from a co-field -- where the series is rigid, not where it is active -- and
  the two are worth looking at together.
  **Not done:** no multiple kernel learning proper. The mixing coefficient here is a grid searched
  inside the cross-validation, not the kernel alignment of `cmfqsar/mixing.py`; with every
  measured mixture at or below the mean alone, a better search for the coefficient would not
  change the answer. And no display in the application: the variance field comes out as a field
  object and `cmfqsar/cofields.py` will draw it, but no panel offers it.

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
  **Done 26 October 2026, and this one helps.** `cmfqsar/multitask.py` gains the multi-endpoint
  **marginal likelihood** (the mean marginalised over the design columns, the amplitude profiled
  out, one factorisation per candidate instead of ten folds), a **free-form low-rank covariance**
  `L L' + diag(d)` -- positive definite for every parameter value, containing the one-rho model at
  rank 1, with a free diagonal so that endpoints in different units get their own amplitudes --
  and a **covariance from task descriptors** `T(t,s) = k(z_t, z_s)` whose length scale and nugget
  the evidence chooses. `Observations` takes `task_features`, which replaces the indicator columns
  of the bias with the descriptors so that the mean extrapolates, and
  `MultiTaskFit.predict_new_task` predicts an endpoint that was never measured (with
  `new_task_coefficients` giving it a co-field like any other). In the GUI the selectivity dialog
  now fits any number of ticked endpoints, offers the covariance as a choice, prints the
  correlation matrix, and lists every endpoint plus the named pair's difference in the Endpoint
  box.
  **Measured on `examples/cyclodextrin/`** (`benchmarks/run_tasks.py`), which has seven real
  endpoints and descriptors for them -- 1669 guests, 2671 association constants
  against seven hosts, and the host is itself a molecule, so the task descriptors are a structure.
  One shared kernel, 10-fold cross-validation over guests: independent
  +0.412 mean q2, one rho by cross-validation +0.454, free rank 2
  +0.488, descriptors +0.448 --
  **sharing is worth +0.076**, the opposite of the corrosion result of A6, and for a reason:
  there two endpoints each had their own molecules, here five small endpoints sit beside two large
  ones.
  **An endpoint with no measurements.** Each host removed entirely and predicted from the other six: mean Spearman
  rho **+0.457** through the descriptor covariance against **+0.421** for the nearest
  host's own model -- so the descriptors carry real information about an endpoint never measured --
  and **+0.670** for simply averaging the other hosts' values of the same guest, which it
  does *not* beat. That baseline is not a model and can only say something about a guest some other host
  was measured against; but it is the number to beat and it has not been beaten. q2 is
  negative on 1 of the 7 hosts,
  so what this produces is a ranking rather than a binding constant. The spread has one cause.
  Five of the seven hosts have seven glucose units (beta and its derivatives), alpha has six and
  gamma has eight: remove one of the five and the covariance interpolates, remove alpha or gamma
  and it must extrapolate outside the range it was built on. The five interpolated hosts average
  rho **+0.611** against **+0.473** for the nearest host alone; the two extrapolated ones give
  +0.431 (alpha) and **-0.284** (gamma, q2 -5.50). **Interpolation between endpoints that were
  measured is what this supports**, and the item's wording -- "an endpoint with no measurements at
  all can be predicted from its neighbours" -- is right exactly to the extent that it has
  neighbours. Seven endpoints are also too few for the evidence to localise the task kernel's
  length scale: it came out at 4 standardised units on all seven fits, the top of the grid, so the
  criterion would prefer it wider still, with every host treated as a neighbour of every other.
  **Not done:** the Am/Eu application itself (the 46 extractants carry one endpoint, lg SF, not the
  two affinities; getting those is a data task, not a modelling one), task descriptors for the
  corrosion media, and a covariance whose descriptors are *fitted* rather than given, which is the
  obvious next parameterisation and the one that would need more than seven endpoints to justify.
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
  it, so **exploring gains nothing measurable** — while `greedy` loses **+0.057 [+0.038, +0.078]**.
  **Two findings worth more than the ranking.** `variance` and `maxmin` end within **0.000 RMSE**
  of each other and within 0.3 % of the same activity found: the posterior standard deviation of a
  CMF model is the distance to the training set in other units, which is item E5 seen from a different
  direction, so there is no separate "uncertainty sampling" here. And `greedy` does better
  than it would in general, because a series of analogues never puts a candidate far enough away for
  the training mean to be returned in place of a prediction; on a diverse library that protection
  is gone, and `ei` is the rule to use.
  **Not done:** a batch rule that refits between picks (the cosine filter is the cheap substitute),
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
  loop puts first is on average **+1.37** log units better than the library
  average, and lies in the best 20 % of the library
  **48 %** of the time. Of the five best groups held out on each set, the median
  rank is 7 of 42 on GPB and 12 of 46 on THER: the loop finds most of what is worth finding and
  misses one.
  **How far it reaches is the other half of the answer.** Only 2 of the eight
  Sutherland sets have a scaffold that comes off at acyclic single bonds; on the rest the
  Bemis-Murcko framework is fused into the molecules and what comes off is not a substituent.
  The benchmark reports those by name rather than ranking the pieces anyway. A design loop over
  substituents needs a congeneric series, and most public QSAR sets are not one.
  **A defect it revealed, and the numbers that changed.** Gasteiger charges come back as nan for
  every atom of a molecule that carries an attachment point, and the field builder read nan as
  zero -- so every topomer model in the program, and every benchmark built on one, was fitted
  **without its electrostatic field**. `_cap_dummies` in `cmfqsar/fields.py` now caps the open
  valence with hydrogen before the charges are computed, which is what a fragment's charges mean
  anyway, and `benchmarks/run_topomer.py` (both collections), `run_corrosion.py` and
  `run_diagnose.py` were re-run. On the Sutherland sets the mean loss from giving up the
  alignment moved from +0.073 to +0.101 q2, and more usefully the pose rule's cost became
  visible: it was 0.015 q2 on four fields and is 0.066 on five, because electrostatics is the
  field most sensitive to where an atom sits and a model with no charges could not be penalised for
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
  **Done 26 October 2026, and it is the least favourable measurement in the repository.**
  DUD-E is unreachable (`dude.docking.org` does not answer), so LIT-PCBA supplied the negatives:
  `examples/make_litpcba_examples.py` reads the AVE-unbiased archive, embeds the molecules with
  ETKDG and MMFF and poses **both classes by one rule** -- maximum field similarity to the
  target's crystal ligand, which knows nothing about which of them is active. `cmfqsar/pu.py`
  adds bagging PU (Mordelet & Vert 2014) on the precomputed kernel, and
  `benchmarks/run_screening.py` scores seven methods on the same split.
  **Measured on two targets** (ESR1_ant, TP53), 152 measured actives and 2999 measured
  inactives, the one-class methods and the fingerprint seeing the training actives alone:
  the best one-class field description reaches **0.544 AUC** against
  **0.685** for the maximum ECFP4 Tanimoto to the same actives.
  **The sweep that makes this a result.** Every setting that can reorder the ranking was tried --
  the attenuation over 0.3, 0.5, 1 and each description's own parameter, 45 fits per set -- and
  the best kept *on the test set*, a bound no protocol could reach. It lifts the best description
  only to 0.574, still short of the fingerprint at its one default
  setting. nu is absent from the sweep on purpose: it places the threshold, shifting every
  decision value by a constant, so no ranking metric can depend on it.
  Some descriptions are **worse than chance** (distance to the centre of the actives at 0.365 on TP53),
  which is what the AVE embedding was built to produce: it removes the analogue bias, and a method
  whose whole content is "this molecule resembles the actives" then has nothing left.
  **The kernel is not what fails.** The same CMF kernel used discriminatively, with the inactives
  treated as unlabelled, reaches **0.689 AUC** --
  +0.110 against calling them negative, and level
  with the fingerprint. What fails against measured inactives is the one-class *description* of
  the active class, not the field similarity it is built from.
  **And the posing is half of any such number.** The same 1500 TP53 inactives posed to maximise
  their similarity to the actives instead -- the generous placement the background sets elsewhere
  in this repository are given -- move the one-class AUCs by +0.141 on average while leaving the
  fingerprint at +0.000, because it never looks at a coordinate. A screening figure measured
  against a similarity-posed background is a statement about the posing.
  **Not done:** the other thirteen LIT-PCBA targets (the two kept are the ones whose inactive
  pools can be posed without a token sample), DUD-E if the site returns, and the follow-up this
  measurement argues for -- a screening score that is discriminative rather than one-class, since
  that is the only use of the kernel which survived.
* **E5. Error prediction compared.** Three quantities claim to predict the error of a prediction:
  the GP variance, the kernel applicability-domain measure, and (to be added) a conformal interval
  on top of KRR. Compare them on the 86 sets as predictors of the actual external error, which is
  the measurement paper 8 is missing.
  **Done 25 October 2026**: `cmfqsar/errorpred.py`, `build --error-predictors` and
  `benchmarks/run_errorpred.py`. The conformal interval the item said was still to be added
  arrived with the conformal work earlier in the session, so all three were available to compare.
  Each estimate is scored against the absolute error on held-out molecules three ways: Spearman ρ
  for the order, the risk–coverage efficiency (1 = the ordering the errors themselves give, 0 = no
  better than ignoring the estimate) for what is gained by acting on it, and the RMSE of the trusted half
  against the doubted half for what a two-bin rule would do.
  **Measured on the eight Sutherland sets**, five 70/30 splits each, α and λ by the evidence:
  efficiency **+0.198** for the GP variance, **+0.186** for the distance to the training set,
  **+0.181** for the domain dissimilarity and the normalised conformal width. The GP variance
  leads by +0.011 with a 95 % interval of [−0.021, +0.044], so **the three are indistinguishable
  as orderings** and the choice between them does not matter.
  Two findings matter more than that. The normalised conformal width and the domain measure **score
  identically and must**: the normalisation divides the interval by the difficulty, which is the
  distance to the training set, so the width is a monotone function of the domain. There are three
  estimates here, not four, and the conformal rule's contribution is calibration, not ordering.
  The remaining two are also nearly the same: the estimates agree with each other at a mean Spearman
  ρ of +0.80 to +0.86 (the conformal width against the domain measure at exactly +1.000), so what
  is being measured three times is one geometric fact — how far the molecule sits from the
  training set — in three sets of units. That is why no ranking between them could be found.
  And all of them are weak: the best captures about a fifth of what a perfect ordering would, the trusted
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
  **Done 26 October 2026**: `cmfqsar/mixing.py`, the `mixing` subcommand and `--mkl evidence`.
  A sum of kernels is the kernel of a sum of independent processes, so h_f is the prior variance
  of the part of the model that field f carries. What is sampled is each field's *share* of the
  kernel, h = F softmax(u), with an exchangeable N(0, 1.5^2) prior on u and log lambda sampled
  beside it; the sampler is a slice sampler, which has no step size for a user to get wrong.
  Three readings: the shares with their intervals, the fields whose whole interval lies below
  1/F, and the drop-one Bayes factor against *this field does nothing*.
  **Measured on the eight Sutherland sets**, five 70/30 splits each
  (`benchmarks/run_mixing.py`). R2pred by equal weights +0.481, by
  per-field alignment +0.487, by joint alignment
  +0.485, by the evidence optimum +0.485: the
  evidence optimum is **+0.003** against equal weights with a 95 % interval of
  [-0.026, +0.035]. That is item D6 again, on the other axis of the kernel: **the evidence
  takes whatever freedom it is given over the shape of the kernel, improves its own number and
  leaves the prediction unchanged.** Choosing the weights is not how a CMF model is made better,
  and the item's premise -- that MKL picking one set of coefficients is a limitation worth
  fixing -- is wrong about accuracy.
  **The interpretation claim is the one that survives, in a qualified form.** Over the
  200 (set, split, field) cases the drop-one Bayes factor and the cost of actually
  removing the field correlate at Spearman rho **+0.121**: the Bayes factor orders the fields by what the training activities can see, which is not the same as what the held-out molecules need.
  **The useful finding is the width of the intervals.** A field's 90 % interval spans
  0.42 of the unit simplex on average, where an equal share is 0.20:
  on a congeneric series the field kernels are strongly correlated, several quite different
  mixtures explain the activities about equally well, and every point estimate of h -- the
  alignment's included -- is claiming a precision the data do not support. That is what the
  posterior adds, and it is an argument against trusting any of the point estimates rather than
  for replacing them.
  **Not done:** the same treatment of the attenuations (D6 settled that by measurement), and a
  GUI reading of the intervals.

* **E7. Kernel partial least squares.** CoMFA's learner, in its kernel form, on the continuous
  field: with the linear kernel it is PLS in the space of the fields, so a CoMFA-against-CMF
  comparison then differs in the representation alone. Regularised by the number of components,
  chosen by cross-validation as λ is.
  **Done 28 October 2026.** `kpls_fit` in `cmfqsar/model.py` (`--method kpls`,
  `--pls-components`, the GUI's method list, the report); the centring is folded into the dual
  coefficients and a bias, so every reader of a kernel expansion works unchanged.
  `benchmarks/run_kpls.py` measures it against kernel ridge regression on the eight Sutherland
  sets under one protocol (`results/kpls/table.md`; the numbers are in the manual, *The learners*).
* **E8. Kernel Fisher discriminant analysis.** The classification counterpart of kernel ridge
  regression: the direction of field space along which the two classes are furthest apart
  relative to their spread. In the dual it is one linear solve, α = (N + λI)⁻¹(m₊ − m₋) with the
  class means and within-class scatter written on the kernel, and it is known to coincide with
  kernel ridge regression on ±1 targets weighted by class size, so the learner costs nothing new
  and the whole chain that reads a kernel expansion (co-field, contributions, Shapley values,
  domain) applies to a classifier without change. The threshold is chosen by cross-validated
  balanced accuracy; probabilities, when wanted, by a one-dimensional logistic fit on the
  cross-validated scores. **Gate:** measured against C-SVC and the Laplace GP classifier on the
  classification sets of E4 under one protocol; the co-field of the discriminant compared with the
  SVC's by the cosine of F3. Kept as a method if its balanced accuracy is within the interval of
  the two; kept as the default classifier only if it is also ahead on calibration.
  **Done 28 October 2026.** `kfda_fit`, `kfda_frame` and `kfda_calibrate` in `cmfqsar/model.py`
  (`--method kfda`, the λ grid, the GUI's classification list, the report), `tests/test_kfda.py`,
  `benchmarks/run_kfda.py`. The score is a kernel expansion, so the co-field, the contributions,
  the Shapley values, the domain and `cmf cofield` read a fitted classifier unchanged, which is
  the whole point of the item. Three implementation decisions the entry did not specify, each
  taken for a reason written down in the docstrings: λ is **relative** (the ridge is λ times the
  mean diagonal of the within-class scatter), so one λ grid means the same amount of
  regularisation whatever the kernel; the coefficients are scaled to unit within-class standard
  deviation of the training scores, without which the scores of different folds carry different
  arbitrary lengths and cannot be pooled (measured: at a weak ridge on GPB the out-of-fold scores
  spread six times as widely as the full fit's); and the balanced-accuracy cut is taken at the
  middle of its plateau rather than at its lower edge, which is where the step function first
  reaches its maximum and moves with one molecule's score.
  **Measured on the two LIT-PCBA sets of E4** (1587 and 1564 molecules, 88 and 64 measured
  actives), one kernel per set, one 5-fold split stratified by class, each of four learners
  choosing its own setting inside every fold by cross-validated AUC. The fourth learner is kernel
  ridge regression on ±1 labels, the simplest reference rule 1 asks for, and it is not a straw
  man: as λ goes to zero the discriminant is that regression with the labels weighted by class
  size, which `tests/test_kfda.py` checks.
  **The gate's condition is met and the result is still a negative one.** Balanced accuracy under
  one threshold rule for all four learners: **+0.012 [−0.047, +0.070]** against C-SVC,
  **+0.003 [−0.037, +0.045]** against the GP classifier, **−0.001 [−0.066, +0.067]** against the
  ridge — inside the interval of the two, so the method is kept. But the AUC differences are
  **−0.003 [−0.071, +0.065]**, **+0.038 [−0.032, +0.108]** and **+0.005 [−0.051, +0.063]**: the
  discriminant does not order the molecules better than any of the three. The ordering is the
  kernel's, and all four share it.
  **The one large difference is a threshold effect, and naming it is the useful part.** Left at
  each learner's own boundary the discriminant leads by +0.067, +0.082 and +0.087 balanced
  accuracy with intervals excluding zero. That is not a better model: it is the only one of the
  four that places a boundary at all, and a boundary left at the fit's own zero calls nearly every
  molecule inactive on a set with one active in twenty, scoring 0.50 whatever lies behind it. The
  advice this measurement produces is to move the threshold of whichever classifier is used on an
  imbalanced set.
  **Calibration: ahead, and not for a good reason.** The Brier score beats the GP classifier's by
  **0.025 [−0.029, −0.021]** on both sets, which satisfies the gate's wording for promotion to
  default. It should not be read that way. Predicting the share of actives for every molecule
  scores 0.0458 over the two sets and the discriminant scores **0.0457** — the base rate to three
  decimal places — while the GP classifier at 0.0707 is *worse* than the base rate. The
  discriminant is better calibrated only in being less overconfident; neither carries usable
  probability information here. **C-SVC therefore stays the default classifier**, a departure from
  the gate as written, taken because the measurement that would trigger the promotion is not
  evidence of anything.
  **The co-field comparison of F3:** cosine **+0.681** on ESR1_ant and **+0.233** on TP53 against
  the SVC's kernel expansion, positive in all five fields on both sets.
  **A second measurement the entry implied but did not ask for.** The threshold rule the entry
  prescribes (out-of-fold balanced accuracy) was compared with the plain Fisher midpoint of the
  two class means on eight balanced two-class problems from the Sutherland sets, cut at each set's
  median: mean balanced accuracy **0.773** against **0.770**, difference **−0.003** with an
  interval containing zero on seven of the eight sets. The search gains nothing measurable over
  the rule that fits nothing. It stays the default because it assumes nothing about the class
  priors, and `kfda_threshold="midpoint"` selects the other.
  **So E8 is kept under rule 5:** it does not beat a simpler reference on accuracy and is labelled
  as what it is — the classification model this program can draw, not a gain in classification.
  Numbers in `benchmarks/results/kfda/table.md`, the manual (*A classifier with a co-field*) and
  the README.
  **Not done:** the multi-class case (one discriminant against the rest), and a probability that
  widens for a molecule outside the training set, which the GP classifier's latent variance gives
  and a logistic map on the score cannot.
* **E9. Kernel logistic regression.** The probabilistic classifier that is linear in the field:
  logistic loss plus λ‖f‖², fitted by Newton's method on the dual. This is the mode that
  `cmfqsar/gpc.py` already finds in its Laplace step, so the learner is that step with λ chosen by
  cross-validation instead of the evidence and without the posterior variance; a method `klr`
  beside `gpc`, cheaper by the variance it does not compute. Several classes by one model per
  class against the rest, when a data task asks for it. **Gate:** Brier score and AUC against
  `gpc` and `svc` on the same sets as E8; expected to tie `gpc` on accuracy and to be the
  reference its probabilities are judged against, since the two share the likelihood and differ
  only in what λ is chosen by.
  **Done 28 October 2026.** `klr_fit` in `cmfqsar/gpc.py` beside the Laplace step it *is*
  (`--method klr`, the λ grid, the GUI's classification list, the report), `klr_threshold` and
  the shared `out_of_fold_scores` in `cmfqsar/model.py`, `tests/test_klr.py` (20 tests),
  `benchmarks/run_klr.py`. The entry said the learner is the Laplace step with λ chosen on the
  folds, so it runs that step rather than a second Newton iteration: `LaplaceGPC` gained an
  `evidence=False` flag that stops before the factorisation the posterior variance and the
  marginal likelihood are read from, and `klr_fit` calls it with the logistic likelihood at
  amplitude 1/λ. The tests check the identity against `LaplaceGPC`, against linear logistic
  regression on explicit features, and against the objective itself (the gradient vanishes and no
  perturbation improves it). The decision boundary is left at f = 0, which is p = 0.5, so the
  class and the probability never disagree — the one wart E8's threshold rule has; `klr_threshold`
  moves it when an imbalanced set needs that, and is off by default.
  **The comparison the gate asks for needed one correction before it could be made.** The entry's
  reasoning rests on `klr` and `gpc` sharing the likelihood, and `gpc` defaults to **probit**.
  Run that way the two differ in three things at once and nothing can be attributed. So the
  benchmark runs `gpc` with the logistic likelihood, and runs it **twice** — amplitude by
  cross-validation and amplitude by the marginal likelihood — which separates the two real
  differences: averaging over the posterior variance, and choosing the hyper-parameter without
  holding anything out. Everything else is E8's protocol, imported from `run_kfda.py`.
  **The expected tie on accuracy holds**: AUC against `gpc` at the same likelihood and the same
  selection is **−0.001 [−0.012, +0.010]** and **+0.019 [−0.059, +0.085]**; against `svc` the mean
  is **−0.011 [−0.061, +0.041]** and against `kfda` **−0.008 [−0.065, +0.048]**. They are the same
  mode, so they are the same classifier, and no reference is beaten on accuracy.
  **The expectation about the probabilities fails, and taking it apart is the result.** The two
  differences pull opposite ways. Dropping the variance averaging gains: Brier
  **−0.0034 [−0.0056, −0.0010]** and **−0.0046 [−0.0075, −0.0015]** against the cross-validated
  Gaussian process, excluding zero on both sets. Choosing the amplitude by the evidence gains
  more in the other direction: the evidence-selected Gaussian process beats `klr` by
  **+0.0041 [+0.0020, +0.0064]** and **+0.0053 [+0.0025, +0.0083]**. **The evidence is the better
  selector for calibration on these sets**, so `klr` does not become the reference the entry
  expected; that is a point for `gpc`, and the second time its evidence has held its own against a
  cross-validated search here (B4 was the first, on ranking).
  **And no classifier's probabilities beat the base rate.** Predicting the share of actives for
  every molecule scores **0.0458** over the two sets; `klr` scores 0.0519, `gpc` by CV 0.0559,
  `gpc` by evidence 0.0472 and `kfda` 0.0457. The reliability table says why: every learner puts
  almost every molecule in the lowest bin, where about 5 % are active, and the few it places
  higher are active about one time in ten rather than the one in two or three predicted. These
  probabilities are a ranking, not a frequency, and the manual says so.
  **"Cheaper" is worth a measured number, and it is not where the entry implies.** On these
  ~1570-molecule kernels one fit takes 490 ms against 567 ms and 499 ms against 531 ms, a gain of
  6 to 16 %, because Newton's method takes the same number of steps either way and only the last
  factorisation is saved. Scoring every molecule takes 0.6 ms against 37.9 ms and 0.3 ms against
  30.2 ms — **61 and 112 times less** — because the variance costs a triangular solve per
  molecule. The saving is at prediction time, which is what a screen spends its time on.
  **So E9 is kept under rule 5**, as E8 was: no accuracy gain, no calibration gain, labelled as
  the plain regularised logistic regression in field space and as the cheap way to put a
  probability on a library. `svc` stays the default classifier and `gpc` stays the method for a
  probability that admits ignorance. Numbers in `benchmarks/results/klr/table.md`, the manual
  (*Kernel logistic regression*) and the README.
  **Not done:** several classes by one model per class against the rest, which the entry made
  conditional on a data task asking for it and none has; and a probability that widens away from
  the training set, which is exactly what this learner gives up and `gpc` keeps.
* **E10. Laplacian-regularised kernel ridge regression.** Molecules without an activity — the
  library being screened, the decoys, the inactives that carry no value — enter the fit through
  the graph Laplacian of the kernel over labelled and unlabelled molecules together (manifold
  regularisation, Belkin, Niyogi and Sindhwani 2006): the penalty γ fᵀLf asks the prediction to
  change slowly between molecules that are close in field space, whether or not they are
  labelled. The dual is one solve over all n molecules, α = (JK + λI + γLK)⁻¹Jy with J the
  indicator of the labelled ones, so the model stays a kernel expansion and every co-field reader
  works; the expansion now includes unlabelled molecules, which is what makes the co-field see the
  library. The normalised Laplacian of `cmfqsar/spectral.py` is the one to use, with the
  neighbourhood truncated so that the penalty is local. **Gate, in two parts:** (i) on the eight
  Sutherland sets hide the activities of half the training molecules and compare with ridge on
  the labelled half alone and with ridge on all of them, the second being the ceiling; (ii) on
  the screening sets of E4 fit with the library unlabelled and score the actives among the
  decoys, against the plain model. Kept if (i) recovers a measurable part of the gap to the
  ceiling on most sets and (ii) does not lose enrichment. The assumption to state is that the
  activity varies smoothly along the data, which an activity cliff breaks; report, per set, how
  many labelled pairs are cliffs by the F3 measure, so that a loss can be read against them.
  **Done 29 October 2026. Part (i) of the gate fails, part (ii) passes, and the method is kept
  under rule 5 as what the measurement makes it.**
  **What was built.** `lapkrr_fit` in `cmfqsar/model.py` -- the dual the entry states,
  `(JK + lam I + gamma L K) a = J y`, with one extra row and column for the bias, as `krr_fit`
  has -- and `knn_laplacian` in `cmfqsar/spectral.py`, the normalised Laplacian of the entry with
  two additions the entry does not state and that both matter: the affinity is the **cosine** of
  the kernel, so that a neighbour is a similar molecule and not a big one, and every molecule
  keeps its ten closest neighbours, which is the truncation the entry asks for.
  **The penalty is taken over the model without its bias**, that is over `K a` and not over
  `K a + b`, and the entry's `γ fᵀLf` is read in that sense throughout. Adding the same constant
  to every activity does not make a model vary faster, so a bias should not be charged for. It is
  worth stating because `L 1` is not zero for a normalised Laplacian, while it is zero for the
  unnormalised one: a bias inside the penalty would be charged for being a constant, and the dual
  above would carry a further `γ L 1 b` term in its extra column. `tests/test_lapkrr.py` fixes the
  choice in both directions.
  `--method lapkrr` with `--unlabelled`, `--manifold-weights` and `--neighbours`;
  `tests/test_lapkrr.py` (27 tests); `benchmarks/run_lapkrr.py`, registered in `run_all.py`.
  **The one design decision the entry leaves open, and it is the reason every co-field reader
  still works.** The activity of an unlabelled molecule is **nan**. A learner needing more than
  `(K, y)` is new in this program, and the alternative -- a second block of molecules handed to
  the learner on the side -- would have put the unlabelled molecules outside the training table,
  where `cofield_coefficients`, `training_contributions` and `atom_contributions` could not reach
  them. Marking the missing values keeps one training table and one expansion over it, so those
  three read the model unchanged and their sums still equal the prediction over all n terms.
  What that convention cost elsewhere was small and is all of it: `_cv_splits` builds the folds
  over the labelled molecules and keeps the unlabelled ones in the training part of every fold;
  `cv_loss`, `CMFModel.fit` and `nested_cv` take their statistics on the labelled molecules;
  `align_mixing` derives the field weights from the labelled molecules; and
  `dual_covariance_factor` gained the system above with one column per labelled molecule, so the
  co-field error band works on this learner too.
  **Part (i) fails, and it is the method that fails.** Half of the published training activities
  hidden, the three models scored on the published test set that none of them saw, settings chosen
  by cross-validation inside what each may see, means over five halvings: ridge on the visible
  half **+0.258** of R2pred, lapkrr with the hidden half unlabelled **+0.235**, ridge on every
  activity **+0.376**. So the hidden measurements are worth **+0.118** and the penalty recovers
  **-0.023** of it -- none. It is ahead on two of the eight sets, its interval excludes zero above
  on none, and on BZR (**-0.032 [-0.062, -0.007]**) and GPB (**-0.051 [-0.177, -0.004]**) it
  measurably loses. **Choosing both settings on the test set**, an oracle that cannot be used and
  is reported only to separate the two explanations, gains **+0.009** with the interval excluding
  zero on no set: there is no weight at which this penalty is worth anything on these series, so
  the failure is not the cross-validation missing it.
  **The cliffs the entry asked for do not explain the loss.** A pair of labelled molecules counts
  as similar at a field cosine of 0.9 or more -- the cosine of F3, applied to two molecules
  instead of two co-fields; F3 fixes the measure and not a cutoff, so the cutoffs are the
  benchmark's and are stated with the counts -- and as a cliff when it is similar and the
  activities also differ by at least one log unit. Between **22 %** (BZR) and **63 %** (THER) of
  the similar pairs are cliffs, but the rank correlation between that share and the gain is
  **-0.33**, and between the mean activity difference of a similar pair and the gain **-0.38**.
  What lines up instead is the **size of the set**: rank correlation **+0.74** between the number
  of labelled molecules and the gain. The penalty loses most on the smallest sets and nothing at
  all on the largest. The smoothness assumption is not what these sets break; there are simply
  too few molecules for a second regularisation parameter to be chosen on.
  **Part (ii) passes, quietly.** The two LIT-PCBA sets under the protocol of E4 and E8, with the
  held-out fold given to the learner as the unlabelled library: no difference in AUC, in the
  enrichment factor at the top 1 % or 5 %, or in BEDROC against either `krr` on +-1 labels or
  `svc` excludes zero below it, so no enrichment is lost. It passes partly because the method
  reduces to its reference: on ESR1_ant the folds chose a penalty weight of zero in all five
  folds, so the model fitted there *is* kernel ridge regression and every difference against it
  is exactly zero. Over part (i) the weight zero was chosen in half of the forty fits.
  **Neither does the neighbourhood size rescue it**: 5, 10, 20 neighbours and the untruncated
  graph give +0.452/+0.410/+0.428/+0.419 on GPB and +0.506/+0.506/+0.506/+0.514 on DHFR, none of
  them reaching ridge on the labelled half.
  **Why it is kept.** The implementation is right, which the tests show rather than claim: on two
  thin curved bands of points with one label each -- the situation manifold regularisation was
  written for -- it turns an R2 of +0.02 into +0.91. What fails is the match between that
  situation and an aligned congeneric series of a few dozen molecules, which is what every data
  set in this repository is. Two things make it worth having anyway. It is the only learner here
  whose co-field is built from molecules that were never measured, so a picture can be drawn over
  a library instead of over a training set. And the weight zero is inside its grid, so trying it
  on a larger or less congeneric collection risks nothing: the cross-validation can switch the
  penalty off, and half the time it did. **Kernel ridge regression stays the default.** Numbers in
  `benchmarks/results/lapkrr/table.md`, the manual (*Molecules without an activity*) and the
  README.
  **Not done:** the graphical interface, which has nowhere to name a second set of molecules, so
  the learner is on the command line and in the library only; the one setting of
  `--unlabelled` where the item might still pay, namely a library far larger and more varied than
  the labelled set, for which this repository has no set with measured activities to test against;
  and the untested guess that a weight shared across sets, or a weight chosen by the evidence
  rather than by the folds, would avoid the loss on the small sets.

## F. Interpretation (months 4 to 9)

* **F1. Case studies.** Shapley values, field differences, pharmacophores and complementarity have
  tests but no worked example. Three FEP series with crystal pockets, each worked through end to end in a
  script under `examples/` and a section of the manual, judged by a medicinal chemist. Paper 7 is
  written from these.
  **Done 26 October 2026, except the part that needs a chemist.** `examples/case_studies.py` runs
  one of three series from end to end -- model, the three checks, co-fields masked at two standard
  deviations, pharmacophore, Shapley decomposition by rings and the pieces between them, field
  difference between the weakest and the strongest molecule, complementarity with the crystal
  pocket, and the HTML report -- and `benchmarks/run_cases.py` keeps the numbers so the manual's
  three sections can be rebuilt from them (`docs/manual.md`, "Three series, worked through").
  **The three cases** were chosen by measuring every set in `examples/openff` and
  `examples/schindler` that ships a pocket and keeping the best models among different target
  families: `openff/ptp1b` (a phosphatase, 22 ligands),
  `openff/mcl1` (a protein-protein interaction groove, 25) and
  `schindler/cdk8` (a kinase, 33, the widest activity range at
  4.18 log units).
  **What they produced**, and the first one is the most useful:
  - `ptp1b` **does not survive its own y-randomisation**: q2cv +0.499
    against a best-of-thirty shuffle of +0.539, p =
    0.065. It is kept as the case that stops at the check, with the
    two warning signs named: a training fit of R2 +0.975, and
    R2pred +0.147 on the held-out third where the selection fold gave
    +0.824.
  - `mcl1` reads cleanly: q2cv +0.548, R2pred
    +0.581, best shuffle +0.098. The
    hydrophobic and steric fields carry it and the donor field is *worse* than useless
    (electrostatic +0.032, steric +0.062, hydrophobic +0.071, hba +0.037, hbd -0.077). **Burial alone correlates with activity at
    +0.698 while the full complementarity score reaches
    +0.254**: a score that averages five terms is three
    times worse than one of them.
  - `cdk8` is acceptor-driven (electrostatic +0.043, steric -0.015, hydrophobic -0.024, hba +0.099, hbd -0.061), and almost none of its map survives its own error
    bar (electrostatic 4%, steric 6%, hydrophobic 4%, hba 7%, hbd 0%). Its pocket file says nothing, because a site trimmed at 8 A around a
    shallow hinge leaves most of a ligand outside it (burial 0.014
    against 0.588 on `mcl1`).
  **One method fix came out of it.** "Which fields does this series need" was being answered by a
  kernel norm, which cannot answer it: the field kernels are scaled to unit mean diagonal before
  they are mixed, so all five come out at about 20 % whatever they contribute. The script removes
  each field and refits instead, and that is the number quoted above.
  **Not done, and it is the half the item cares about:** no medicinal chemist has judged any of
  this. Paper 7 is written from a chemist's reading of these maps, and what exists now is the
  material for that reading, not the reading. Also not done: the other 70 sets that have a pocket,
  and a case where the pharmacophore is tested by screening rather than by inspection.
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
  Two things the measurement adds. The threshold is **conservative**: the bootstrap moves the co-field
  only 0.6 times as far as the closed form allows for, so cutting at two standard deviations is
  really cutting at about 3.2 of the bootstrap's, which follows from taking sigma to be the
  out-of-fold error rather than the assay's noise alone (`--sigma` sets it directly). And the
  thing it gets right is *where* the uncertainty is, the two orderings agreeing across the grid
  at a rank correlation of **+0.82** -- which is what a mask needs, a mask being an ordering with
  a cut-off.
  So the item's claim holds: it is cheap, it is new, and it changes what every co-field picture
  in the program means, the report's included. Between a half and five sixths of a drawn co-field,
  depending on the field, is shape the measurement error could have produced.
* **F3. Co-fields compared.** The overlap of two co-fields is a closed-form similarity between two
  models: two targets (selectivity, again), two series on one target, or one series across rounds
  of active learning. A table of co-field overlaps is a map of SAR transfer.
  **Done 25 October 2026**: `cmfqsar/transfer.py` and the `transfer` subcommand, which takes two
  model files and prints the per-field cosines, or several and prints the matrix. The overlap is
  `a^T K_f(A,B) b` -- one matrix product on the kernel block between the two training sets -- and
  the cosine is the number to read, since it is invariant to the units of either activity, to the
  mixing coefficients and to the field scaling, and so compares the shapes of two demands rather
  than their sizes. A test checks the closed form against the same integral on a 175 000-point
  grid; they agree to four decimals.
  **Measured** (`benchmarks/run_transfer.py`): every Sutherland set split into its chemotypes, a
  model on each group of fifteen or more with the attenuation shared across the set, every
  ordered pair scored by the cosine and by how well one model orders the other group's molecules.
  Two halves of one chemotype -- the control, where the SAR is the same by construction -- agree
  at a mean cosine of **+0.401** and transfer at rho **+0.673**; two different chemotypes of one
  target agree at **+0.178** and transfer at **+0.334**. The cosine separates the two cases by
  **0.22**, and within the chemotype pairs the cosine and the transfer correlate at rho
  **+0.667** over 64 pairs. The item's claim holds: a table of co-field overlaps is a map of SAR
  transfer.
  **What it depends on.** Two co-fields are functions on one space only if their molecules were
  posed in one frame. `incompatible` checks what the model files allow -- both linear, both on
  the overlap kernel, the same fields with the same attenuations, overlapping training boxes --
  and refuses with the reason; it cannot check whether two alignments mean the same thing, and
  nothing in the cosine would reveal that they did not.
  **Not done:** the same comparison for models in different frames, which needs the two training
  sets aligned to each other first and is really the alignment problem again; and the GUI, where
  a matrix of co-field cosines would belong beside the chemical-space map.
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
  which is the difference between a report and an advertisement. For everything else a missing part is reported rather
  than causing a failure: a nonlinear kernel loses the co-fields, the contributions and the pharmacophore and
  keeps the rest.
  About three seconds for a 50-molecule model with the co-field pictures, under one with
  `--no-cofields`.
  **Not done:** the PDF writer of the item's text (a browser prints the page, which is enough),
  the 3D isosurface renders a GUI screenshot would give, and F1's case studies, which are what
  would make a report explain a model rather than only state facts.
* **F5. Pharmacophore export.** The typed features with tolerance spheres of
  `cmfqsar/pharmacophore.py` in a format a pharmacophore-screening program can read, so the
  co-field can be used as a query outside CMF.
  **Done 26 October 2026**: `write_pharmit`, `write_pml` and `write_query` in
  `cmfqsar/pharmacophore.py`, reached from `pharmacophore --query FILE` (the extension picks the
  format: `.json` a Pharmit session, `.pml` the LigandScout-style XML). Three things the export
  is explicit about rather than silent on. A query is a conjunction, so only the strongest
  `--enable-top` features are written switched on and the rest are written disabled. A
  hydrogen-bond feature is written without a direction, because a co-field has none. And three of
  this program's feature types cannot be expressed at all -- `bulk`, which asks for atoms of any
  kind, and the disfavoured `polar`, `acceptor_bad` and `donor_bad`, which name one kind of group
  as unwelcome where a query has only the exclusion sphere, which forbids every atom -- so they
  are reported as left out.
  **Measured** (`benchmarks/run_query.py`): the query, matched by the rule a screening program
  uses, separates the most active quarter of the held-out molecules at an AUC of
  **0.665** against **0.495** for the same features moved to
  random places in the same box -- a paired difference of +0.170, 95 %
  interval [+0.102, +0.238], ahead on 36 of 40 splits. The model
  it was read from reaches **0.861** on the same molecules.
  So the places the co-field picks are better than chance and much worse than the model: the
  export is a lossy summary whose value is that it can be used outside the program. That is what the item
  asked for, and the measurement says what it costs.
  **Not done:** a conformer-generating matcher (the benchmark matches molecules already posed in
  the training frame, which overstates the quality of any query), and the MOE and Phase formats.
* **F6. Kernel canonical correlation analysis: what the field sees that another view does not.**
  A view is one description of the same molecules -- one kernel over them. The program builds
  several: the five-field CMF kernel, the quantum fields of `cmfqsar/cdft.py`, the ECFP4
  Tanimoto kernel of `benchmarks/run_fusion.py`, the pocket complementarity of
  `cmfqsar/complementarity.py`. What no tool here answers is how much two of them say the same
  thing, and, more useful, **which part of the field is the part a second view does not
  contain**. Canonical correlation analysis (CCA) answers exactly that question: given two
  descriptions of the same objects it finds the pair of directions, one in each description,
  whose projections correlate most strongly, then the next such pair uncorrelated with the
  first, and so on. The kernel form (KCCA) does it with kernels in place of coordinates, which
  is what CMF needs because its description of a molecule has no coordinate vector.
  With centred kernels `K` and `L` on the same molecules, the pairs are the solutions of the
  generalised eigenproblem `[[0, KL], [LK, 0]] v = rho [[K^2 + c K, 0], [0, L^2 + c L]] v`,
  where `rho` is the canonical correlation and `c` is a ridge. The ridge is not an option. With
  `c = 0` and invertible kernels every canonical correlation is exactly 1 and the directions
  are arbitrary, which is the classical failure of KCCA and the reason this item is measured
  against a permutation control rather than against its own correlations.
  Two properties make it a CMF tool rather than a general one. A canonical direction on the
  field side is a vector of dual coefficients, the same object as a fitted model's, so
  `CMFModel.cofield_coefficients` turns each component into a co-field with no new
  interpretation code; rule 2 is satisfied because the direction is linear in the field. And
  with the features of B6 both sides become matrices of m columns and the eigenproblem is m by
  m, so the cost does not grow with the number of molecules.
  **The degenerate case has to be stated first, because it is the obvious way to misuse the
  method.** With the activity as the second view, that view has one dimension, and CCA with a
  one-dimensional second view returns the ridge regression of the activity on the first view,
  up to a scale factor. "KCCA-QSAR" in that sense is kernel ridge regression under another
  name and predicts nothing new; the only use of the case here is as a correctness check, that
  the first canonical direction reproduces what `cmf build` fits at a matched ridge. The method
  earns its place only when both views are rich.
  **The pairing to measure first** is the CMF kernel against the ECFP4 Tanimoto kernel, because
  it puts CMF's own premise to the test. The field is three-dimensional and the fingerprint is
  not; the canonical correlations say how much of the field space the fingerprint already
  holds, and the field directions that correlate with nothing on the fingerprint side are the
  part that is genuinely three-dimensional. Those directions have co-fields, so the answer is
  not a number but a picture of where the three-dimensional information sits. Both kernels
  exist today, neither needs an alignment that the repository does not have, and the eight
  Sutherland sets carry them.
  Needed: `cmfqsar/kcca.py` with the centred regularised eigenproblem, solved on the n by n
  kernels for a small set and in B6's feature space for a large one, with the ridge chosen on
  folds by the **held-out** correlation and never by the training correlation; a whole-molecule
  Tanimoto kernel in the library (`cmfqsar/bioisostere.py` has one for substituents only, and
  `benchmarks/run_fusion.py` has `ecfp4_similarity` which should move into the library); a
  permutation null; the F3 cosine to compare the two co-fields of one component; and
  `benchmarks/run_kcca.py`. Two to three weeks, most of it measurement.
  Gate, in three parts, each with the interval of rule 4. (i) **The correlations are real**: the
  canonical correlations on held-out molecules stay above the correlations obtained when the
  pairing between the two views is shuffled. (ii) **The uncorrelated part carries activity**: a
  model built on the field directions that the fingerprint does not contain predicts activity
  better than a model built on the same number of directions that it does contain. (iii) **The
  components are readable**: the two co-fields of one shared component resemble each other, by
  the F3 cosine, more than the co-fields of two components taken at random do.
  What a failure settles is worth as much as a success. If every field direction that carries
  activity is also in the fingerprint, then on these sets the three-dimensional field adds
  nothing a two-dimensional descriptor does not, which is a measured answer of the same kind as
  D5 and belongs in the README whichever way it comes out. If part (i) fails, the ridge was too
  small and the correlations were the degeneracy above, not chemistry.
  **Other pairings, in the order they become possible.** The quantum fields against the
  empirical ones, on the corrosion sets where both are computed and cached
  (`benchmarks/run_cdft_kernels.py`), which asks what the conceptual-DFT field adds beyond the
  shape and charge the empirical fields already carry. The ligand field against the pocket
  field, which is the classical chemogenomics use and the most valuable one, and which this
  repository cannot do yet: `examples/pockets/` holds one pocket per data set, so a kernel over
  targets needs several targets whose ligands share a frame, and that is the alignment problem
  of part C. The item is deliberately not a learning item: using the shared space as features
  for a predictor belongs with E4 and E6 and is not claimed here.
  **Done 28 October 2026**: `cmfqsar/kcca.py`, the `kcca` subcommand, the whole-molecule
  Tanimoto kernel as `cmfqsar.descriptors.fingerprint_kernel` (which `run_fusion.py` now
  uses), and `benchmarks/run_kcca.py`. The eigenproblem is solved through the spectra of the two
  centred kernels, as the singular value decomposition of
  `diag(sqrt(k/(k+c))) U^T V diag(sqrt(l/(l+c)))`; the ridge is relative to the mean eigenvalue
  of each view and chosen by the mean held-out correlation over all pairs, never by the training
  one; the shuffled pairing is the null for every component; `component_model` turns a direction
  into a model whose co-field the existing code draws. The degenerate case is a unit test: against
  a one-dimensional view the first direction gives the fitted values of kernel ridge regression at
  the matched ridge exactly, and the dual vector itself when the kernel has full rank.
  **One deviation from the text of gate (ii).** It asked for a model on the directions the other
  view does not contain against a model on the same number it does contain. Fitted on the
  canonical variates that comparison measures nothing, because the variates are whitened by
  construction and a regression on them has lost how much the molecules vary along each
  direction, which is the prior kernel ridge regression rests on (on the thrombin set a model on
  *all* the variates fell to q² 0.0 where the field model reaches 0.6). The gate is therefore
  measured on a projection: a direction counts as contained when the other view explains more
  than half of its variance on held-out molecules, the field kernel is split into its part inside
  those directions and the rest, and kernel ridge regression is fitted on each with the
  regularisation chosen inside the folds. The two parts have different dimensions; a random
  subspace of the contained part's dimension is run beside them as the control for that.
  **Measured** (`benchmarks/run_kcca.py`, the eight Sutherland sets, the five fields at the
  evidence's attenuation against ECFP4). (i) holds: the leading pairs hold out at **0.93 to
  0.99** on every set against about 0.2 for the shuffled pairing, and 83 % of all pairs clear it.
  (ii) comes out against the premise: the fingerprint contains **14 %** of the field's directions
  on average, and that part predicts the activity at q² **+0.478** against **+0.162** for the
  rest and +0.528 for the whole field; the difference is below zero with its interval on **7 of
  8** sets (−0.32 on average), and a random part of the contained part's size reaches +0.024, so
  it is the directions and not their number. (iii) holds, on two field views of the same molecules
  (the file's poses against poses moved by up to 10° and 0.5 Å, since a fingerprint has no
  co-field): the co-fields of the two sides of one component agree at **+0.83** against 0.05 for
  different components, on all eight sets. So on the Sutherland series a two-dimensional
  descriptor already holds the field directions that carry the activity; the three-dimensional
  remainder adds little, which is the fourth-dimension finding from the other side, and the whole
  field still beats either part.
  **The other two pairings, measured 28 October 2026.** *The quantum fields against the
  empirical ones* (`benchmarks/run_kcca_quantum.py`): view A the three reactivity fields (f+, f−,
  the dual descriptor, one Gaussian per atom weighted by the condensed index), view B the five
  empirical fields. On the 49 series of the GFN2-xTB stage of D3, 78 % of the pairs hold out
  above the shuffled pairing and the empirical view contains **23 %** of the quantum view's
  directions; that contained part predicts the activity at q² **+0.181** against **−0.058** for
  the quantum view's own part, with the interval below zero on **20 of 49** series and above
  zero on 3 (`hammett_sym`, the one series where the quantum field was built to matter, and the
  pfkfb3 and eg5 free-energy sets). The empirical view alone reaches +0.364 and the whole quantum
  view +0.272. So what the reactivity fields know about the activity is the part of them that
  shape and charge already contain, which is why adding them to the five fields gained little in
  D3 and in the corrosion study; it is a statement about the condensed atom-centred form of the
  fields, not about conceptual DFT. The four ORCA corrosion series (20 to 29 molecules) say the
  same with intervals that all cover zero. *The ligand field against the pocket*
  (`benchmarks/run_kcca_pocket.py`): the chemogenomics form needs several targets whose ligands
  share a frame, which the repository does not have, so the pocket's view of a ligand is its
  interaction profile with the stored binding site, the complementarity resolved by residue and
  interaction type (`cmfqsar/complementarity.py`), as a linear kernel. On the 21 free-energy
  series with a binding site in the ligands' frame (two Schindler sets resolved to a pocket 30
  to 50 Å away and are refused by a frame check), 83 % of the pairs hold out and the profile
  contains **35 %** of the field's directions; but the contacts predict at q² −0.010, the
  field's own part at +0.018, the profile alone at −0.015 and the whole field at +0.180, with the
  interval covering zero on 16 of 21 series and the rest on both sides. Under rule 4 that is no
  result for where the activity sits, and it says two other things: the contact profile is not a
  predictor on its own on sets of 22 to 44 ligands, and the whole field predicts better than
  either of its parts, so the split costs more than it tells on sets this small.
  **Not done:** the cross-target form of the pocket pairing, which is the alignment problem of
  part C; the feature-space solver for B6's landmarks, since every set here fits the n by n form;
  and the GUI.
## G. The application and the tool (continuous)

* **G1. Subcommands.** Alignment is spread over flags of `build` and `predict` and the `hetero`
  subcommand; give it a subcommand of its own, `align` (by confidence, by MCS, in the pocket), and
  add `diagnose` (C6), `screen` (B3), `cofield` (grids and cube export without the GUI), ~~`report`
  (F4)~~ **done** and `benchmark` (A4).
  **Done 26 October 2026.** All six exist: `diagnose` arrived with C6, `screen` with B3, `report`
  with F4, and this item added the other three.
  **`align`** takes the four criteria the program has and puts them behind one `--by`: `mcs` (the
  common substructure with a reference molecule), `field` (maximum field similarity to a
  reference, to a model's training set or to the set itself), `variance` (the confidence criterion
  of the 2019 paper, with `--flexible` for the torsions) and `pocket` (that confidence with the
  complementarity term). It prints what it did per molecule, then how far the set moved, and
  writes nothing without `--out`, because an alignment that is not kept is a measurement and a
  command that silently rewrites a file is a trap. `--max-rmsd` is the displacement guard, and
  `--judge` adds the mean field similarity within the set afterwards.
  **`cofield`** writes the co-fields as Gaussian cube files, which any viewer opens, and prints
  the peak of each field and the share of the covered grid that stands more than `--z` standard
  deviations from zero -- the quickest way to see how much of a map is worth looking at.
  `--kind molecule` and `--kind overlap` give one molecule's field in the model's frame and that
  field weighted by the co-field, which is the picture that explains a prediction.
  **`benchmark`** runs `benchmarks/run_all.py` through the tool, with `--list`, `--check`,
  `--tables`, `--manifest`, `--readme`, `--data-root` and `--orca` passed straight through. It
  needs the source checkout and says so rather than failing obscurely.
  `tests/test_cli_subcommands.py` drives all three through `cmfqsar.cli.main`, including the four
  alignment criteria and the four ways each command refuses what it cannot do.
  **Not done:** the alignment flags on `build` and `predict` still exist, because scripts and the
  benchmarks use them; `align` is the new way in, not a removal.
* **G2. Projects.** A project file that holds the data set, its poses before and after each
  alignment, the settings, the model and the results, so a session can be reopened; undo of an
  alignment step; a panel that compares two models on the same molecules.
  **Done 28 October 2026.** `cmfqsar/project.py` holds the pieces without Qt: a pose snapshot
  (every conformer of every molecule), `PoseHistory` (last in, first out, bounded at fifty),
  `Project` with `save_project` / `load_project` (a joblib dump like a model file; the molecules
  travel as RDKit binaries with their properties) and `compare_models` (per-molecule rows, RMSE
  and r² per subset, agreement of the two prediction vectors). The application keeps a snapshot
  before each of the nine steps that move molecules, *Undo last alignment step* (Ctrl+Z) restores
  it and discards the model as any alignment does, *Save project* / *Open project* carry the data
  set, the history, every panel's settings (captured by attribute name, so a new widget needs no
  change), the model with its indices, the search table, the validation report, the log and the
  protein; a `.cmfproj` on the command line reopens the session. *Compare two models on these
  molecules…* opens a sortable table with a CSV export. `tests/test_project.py` covers the round
  trip, undo order and bound, the comparison, and the application's own save, reopen and undo.
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
  Properties of an atom rather than a molecule (pKa, NMR shifts, site reactivity) are the LP
  items below: the same kernels on a sample the program does not yet have.
* **Host–guest chemistry and delivery systems.** The cyclodextrin tables are the first
  fixed-host series, and the invariant kernel is the model for them, since the guests of one
  host share no frame. Binding to an excipient, release and solubility gain are functions of
  the constant the model predicts; the items are the SC section below.
* **Peptides.** `mittal/AMPH1` is 130 decapeptides that no configuration models, and the reason is
  a 3.2 Å superposition error over a flexible chain. It is the test case for C2 and C3 together, and
  the plan expects it to stay hard. Item MC10 of part I is the route that gives up on aligning
  whole peptides.
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
for structure comparison, costs accuracy on a property that depends on composition times density; and
fifty element-indicator fields dilute the kernel in a way the block weights halve but do not remove.
What it cannot do is the list below, ordered by the code each item needs. Rule 1 applies unchanged
with the four-field periodic model (electrostatic, steric, electronegativity, 1/√m, background
kept: MAE 92 cm⁻¹ on Matbench phonons) as the reference every item has to beat on the same folds.

**Small (days each)**

* **M1. Lower-dimensional periodicity.** Polymers as 1D-periodic repeat units and slabs as
  2D-periodic surfaces already pass through the ASE neighbour list with mixed boundary conditions,
  but the background term of `periodic_autocorrelation` divides by the cell volume. It needs the
  repeat length or the surface area, and the slab case needs the height channels of the surface
  kernel (`SurfaceSpec` in `cmfqsar/invariant.py`). Opens surface energies, work functions and
  facet-dependent adsorption. Measure on a slab set against the surface kernel alone.
* **M2. Partial occupancy and configurational ensembles.** Solid solutions and disordered sites
  need the field weights multiplied by the site occupancy; `structures_from_matbench` takes the
  majority species today. Special quasirandom structures and MD snapshots are the periodic
  analogue of the conformer ensemble and reuse the mean-map averaging of `ensemble_fields`,
  generalised by D4's `combine_members` to members with their own atoms. The fourth-dimension verdict says to
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
  basis and the co-field is what the grant paragraph needs to show. Gate: the co-field of the
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
  block of `chirality` in `cmfqsar/invariant.py`, remove the deficiency at the cost of a cubic sum per
  structure; the factorisation trick of the molecular block does not survive the lattice sum.
  Measure on structure-type classification first, where the deficiency is visible, before any
  property.
* **M11. Inverse design on prototypes.** The co-pattern says which pair at which distance to add.
  Making that usable for design needs a candidate generator over prototype structures with substitutions,
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

**Order.** M4, M5 and M6 first: together they give the most results per line of code and supply the theory
section and two of the case studies of the publication plan. M1 and M3 are done as they are needed
by those three. M9 and M12 only after M6 has reported.

### Nanotechnology: the extensions a particle, a tube, a sheet and a coated surface ask for (26 October 2026)

The application is built around one object: an RDKit molecule with one conformation, loaded by
`load_dataset` in `cmfqsar/dataio.py`, given fields by `build_fields`, drawn by `Viewer3D`. Every
nanotechnology use breaks one of those three assumptions. A metal cluster or a quantum dot has no
lattice and no shared frame between particles of different sizes; a nanotube is periodic in one
direction and a sheet in two; a coated particle is an adsorbate problem on a curved surface; a
porous framework is a host. The periodic kernel (M-items above) supplies the kernel for all of
them. What is missing is listed here, grouped by the layer of the program each item touches, with
the module it starts from. The same rule 1 applies: every item is measured against the four-field
periodic model on the same folds before it becomes a default.

Two cautions are recorded before the items. First, the per-atom normalisation of the periodic
kernel removes the particle size, which is often the main variable in nanoscience; size must enter
as a separate descriptor (NT3). Second, the published nano-QSAR sets describe particles by size,
coating, zeta potential and composition rather than by atomic structure, and they come from
different laboratories. The corrosion study showed what that costs. The first study on such a set
tests whether the target is reproducible between sources before any kernel is compared.

**Data model and input**

* **NT1. A structure type that is not a molecule.** `CrystalFields` in `cmfqsar/periodic.py` stands in
  for a molecule inside the model, but `Dataset` in `cmfqsar/dataio.py` holds RDKit molecules only,
  and the Dataset panel reads SDF, MOL2 and SMILES. Needed: a `Dataset` that can hold ASE structures
  with a cell and boundary conditions, readers for CIF, POSCAR, extended XYZ and ASE databases, and
  the name and the target taken from the structure's info fields. Every other NT item depends on
  this one. About a week.
* **NT2. Particles of thousands of atoms.** The overlap kernel is quadratic in the number of atoms,
  and `AtomTable` concatenates every structure's atoms. For a particle the neighbour-list path of
  `periodic_autocorrelation` is the only one that scales. Needed: open boundaries in that function
  (no background term, no cell volume) and a guard in the model that refuses the overlap kernel
  above an atom count. Two days.
* **NT3. Size and shape as descriptors.** `cmfqsar/descriptors.py` computes RDKit descriptors for
  molecules. Structures need their own: atom count, diameter, surface fraction, aspect ratio; and
  the measured values of a nano-QSAR set (hydrodynamic size, zeta potential) read from a table and
  added through the descriptor block that the model already has. Two days.

**Fields and kernels**

* **NT4. Tubes and sheets.** Nanotubes and nanowires are periodic in one direction, sheets in two.
  M1 covers this (the background term with the repeat length or the area, and `PeriodicSpec` told
  which directions repeat). Nanotube handedness is invisible to a pair kernel (M10); the diameter
  and the metallic or semiconducting character are visible in the distances. No new item; NT4 is
  M1 measured on a nanotube set.
* **NT5. Curved surfaces for ligand shells.** The surface kernel (`SurfaceSpec` in
  `cmfqsar/invariant.py`) resolves the in-plane distance by the height above z = 0. For a coated
  particle the height becomes the distance from the particle surface, and the in-plane distance
  becomes the distance between anchor points along the surface. A `ParticleSurfaceSpec` beside
  `SurfaceSpec`, with the core given as a set of atoms, and a test that a flat core reproduces the
  surface kernel exactly. One week.
* **NT6. Fields for the core.** `inorganic_fieldset` gives oxidation states, atomic volume,
  electronegativity, mass and element indicators. A particle core adds the coordination number and
  the surface exposure of each atom, which `cmfqsar/exposure.py` computes for molecules and can be
  reused. Two days.
* **NT7. Site-resolved models on particles.** Catalytic activity and binding belong to sites, not to
  the whole particle. This is M5 applied to particles: per-site rows of the channel tensor,
  per-site targets, and the decomposition that names which site carries the property. Measured
  first on adsorption energies by site on a cluster set. One to two weeks beyond M5.

**Interpretation and display**

* **NT8. A viewer for large structures and cells.** `Viewer3D` in `cmfqsar/gui/viewer3d.py` draws a
  molecule as spheres and bonds with a co-field as isosurfaces. It needs instanced drawing for ten
  thousand atoms, a unit cell outline with periodic images, and no bond perception for metals.
  One to two weeks of Qt and OpenGL work. M13 is the same item seen from the crystal side.
* **NT9. The co-pattern as a plot.** The invariant kernels have no co-field. `top_channels` returns
  the ranked channels; the Results panel needs a chart of weight against distance for the chosen
  pair of field types, and the viewer needs site colouring by contribution (NT7). Three days.
* **NT10. Complementarity with a porous host.** `cmfqsar/protein.py` and
  `cmfqsar/complementarity.py` read a PDB protein and resolve scores by residue. A framework read
  from CIF with periodic images, and scores resolved by framework atom type, extend the same dialog
  to MOFs and zeolites. This is M7; NT10 is its measurement on a gas-uptake set.

**Application panels, command line and proof**

* **NT11. Model panel and command line.** A kernel source entry for "periodic" and "particle
  surface" in the Model panel, with the cutoff, width, taper and background controls that
  `build-crystals` exposes, and the descriptor block shown next to the fields. On the command line,
  `build-crystals` needs the open-boundary and lower-dimensional options and the descriptor table,
  and `predict` needs a path that accepts structure files, since `cmd_predict` reads molecules
  only. Four days.
* **NT12. Benchmarks and tests.** Rule 5 requires a benchmark per module. Candidate sets with public
  atomic structures: metal clusters from the Cambridge Cluster Database for energies, a nanotube
  set for band gaps, and a MOF uptake set such as CoRE MOF with simulated loadings. Each needs a
  script under `benchmarks/`, a results folder, and tests of the invariances (a rotated or
  re-centred particle gives the same channels; a supercell of a tube gives the same channels per
  atom). One week per set.

**What this does not cover.** Properties that depend on long-range shape, such as the plasmon
resonance of a particle above a few nanometres, are set by continuum electrodynamics and not by pair
distances. Self-assembly of DNA structures and of block copolymers is a mesoscale problem where the
atomistic field is the wrong level of description. Neither is planned.

**Order.** NT1, NT2 and NT4 (that is, M1) first, then NT8: nothing can be shown or checked in the
application before structures load and draw. NT5 and NT7 are the two items that carry new
results; the rest is the plumbing the molecular side already had. NT12 runs alongside each item it
measures.

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
0.38, `corrosion_qm/table.md`), the aligned overlap kernel and the frameless kernels alternate in rank
from series to series, and two endpoints on one kernel give a selectivity co-field.

What electrochemistry asks for next is decided by who the partner is. A redox potential of an
isolated molecule is an electronic-structure quantity, and the program already carries what a
congeneric series needs for it: Fukui and charge fields, the ionisation potential and electron
affinity as descriptors, the applied potential and the pH as descriptors, the GP variance for a
series of twenty. Nothing below is required for that; it is a benchmark to run, not code to write.
Three partners the program cannot describe yet are an electrode surface approached face-on, a metal centre
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
  The module's own measurement says Gaussian fits represent long-range tails badly, so the CHELPG
  field, already there, stays the first choice for the potential. Measure on a quinone or
  phenazine redox-potential series with Hirshfeld, Fukui and spin fields against the five fields
  and against the IP/EA descriptors alone; the conceptual-DFT result on corrosion says where not
  to expect a gain, and the Hammett result where to.
* **EC2. Planar alignment.** Built 27 October 2026: `cmfqsar/planealign.py` (flat, flat-up,
  upright and auto poses, the refinement in the plane or with tilt, the in-plane and height
  RMSDs), `cmf align --by plane`, `tests/test_planealign.py`, `benchmarks/run_fe110.py` with
  `benchmarks/results/fe110/table.md`, the README section *Posed by rule: the surface frame
  without the slab* and its manual section. The ask was a principal-axes aligner that lays a
  molecule on z = 0 with a chosen face down, refined by `field_align` restricted to the plane,
  the normal and the tilt, so that the overlap kernel gets a surface frame and a surface
  co-field. **Gates not passed, and the item stays as a display tool, as the ask said it then
  would.** Pose recovery: the chemisorbed benzimidazoles UMA left flat are reproduced to 0.36 Å
  from the relaxed geometry *given the face*, but the face of a planar molecule is a mirror in
  the plane that neither the "flattest face down" rule (which reads chemisorption bending
  backwards, 2.2 Å) nor the field similarity to one reference decides; the upright and oblique
  third are 3 Å from any rule pose, and refining the tilt hurts everywhere. Models: the overlap
  kernel in the rule frame from MMFF conformers gives q² 0.27 on the benzimidazoles, exactly the
  relaxed frame with its heights levelled (0.27), and nothing on the quinolines; the surface
  kernel on the same rule poses gives 0.31 and 0.00, so the rule frame does not beat it. What the
  measurement found instead is that the relaxed frame carries the outcome of the relaxation: a
  line on the lowest heavy-atom height alone predicts E_ads with q² 0.47 and 0.42, so the
  0.5–0.66 quoted for the Fe(110) sets was mostly the relaxation's verdict, and the number for
  E_ads from structure alone is about 0.3 and about 0. The co-field in the rule frame agrees with the
  relaxed-frame one only at the height of the ring and only for some fields (r 0.6 for the
  hydrophobic field on the benzimidazoles, for the steric and donor fields on the quinolines,
  below 0.3 for the rest and for everything 1.5 Å higher).
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
  shield. The data are dozens of compounds per reaction, so this is work for a Gaussian process with intervals,
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
  becoming surface-site profiles. The Fe(110) poses are the reference poses no protein set has:
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

**Order.** EC2 was first and is done: it gives the surface frame and the co-field in it, and
it found that the relaxed frames carry the relaxation's outcome, which changes what the Fe(110)
numbers mean. EC1 next, days on modules already in the repository. EC4 after that, since EC3, EC5,
EC8 and EC9 are empty without a catalyst series that loads. EC6 after M1, EC7 on its own, EC8
after M5, EC9 and EC10 last. The EC2 result also says what EC6 must do differently: a slab as
the receptor has to decide the face and the tilt by complementarity, since no rule and no
reference does.

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
rather than of either site. Most of what batteries ask for falls under items that exist: EC7 (Li⁺ as a
probe), EC1 (charge and spin state), E1 (potential, solubility and stability on one kernel), M4
and M5 (the co-field in the cell and the site resolution), M3 (Bader charges as fields), M1
(polymer electrolytes as 1D-periodic chains), EC2 and EC6 (electrolyte molecules on lithium and
on cathode surfaces), and D4 (ensembles whose members have their own atoms), on which every
solvation-shell idea below depends. The items here are what none of those covers, ordered by the
code each needs. Rule 1 holds with the periodic autocorrelation kernel as the reference for every
property of a cell and the five-field model on the complex as the reference for every property of a
shell. Two limits stand over the whole section and cannot be removed by a good q²: transport
properties (conductivity, transference, diffusion) are collective, so CMF models them only as a
surrogate on bulk values that a simulation or a measurement supplies; and the electrochemical
window as measured is kinetic, set by passivation, so models of stability are trained on measured
values and a computed oxidation potential is a descriptor, not a target.

**Small (days each)**

* **BT1. Fractional compositions in the crystal charge model.** `crystal_charges` in
  `cmfqsar/periodic.py` takes pymatgen's first oxidation-state guess on the reduced composition,
  which fails for a partially delithiated cathode (Li₀.₅CoO₂ has no integer assignment). A
  fallback that spreads the charge over the mixed-valence species, and Bader charges taken in
  preference once M3 is done, are needed before any cathode series builds. Test: the charges of
  Li_xCoO₂ for x in {1, 0.75, 0.5} sum to zero and vary monotonically on Co.
* **BT2. Battery descriptors.** The descriptor block has no computed sources for crystals. The
  theoretical capacity from the formula, the Li content, the volume per formula unit and the
  density as per-structure descriptors; the coordination number and the mean Li–O distance as
  per-complex descriptors for the shells of BT4. Simple once the computed-descriptor registry of
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
  (GFN2-xTB through `tblite` once D3 is done, ORCA otherwise, cached as the quantum stage caches),
  and aligns every complex on the cation with the coordination polyhedron fixing the orientation,
  so that the frame needs no extra work, as with the metal centre in EC4. RDKit holds disconnected
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
  `free_space` test of `cmfqsar/complementarity.py` is the starting point: the point of least free space on
  the straight or relaxed path), evaluates the fields along the path as a one-dimensional profile,
  and builds a kernel over site pairs from those profiles. Per-pair targets are the migration
  barriers from nudged-elastic-band or bond-valence calculations, and the LLZO, NASICON and
  argyrodite families have public sets. Gate: beat M5's site-resolved model on the same cells, and
  read the co-profile as the bottleneck chemistry (which anions, at what distance) the path
  requires. Waits for M5, which decides whether the site resolution alone already carries the
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

### Local atomic properties: the extensions a per-atom target asks for (25 October 2026)

The Hammett study (`examples/hammett`, `benchmarks/results/hammett/`) needed no new code because
its sample was still a molecule: a substituent constant is a molecular label in a frame the ring
fixes, and the co-field came out of the ordinary machinery. A pKa, an NMR shift, an atomic charge
or a site reactivity is a label on an atom in its environment, and that changes the sample. The
program has no object for an atom-in-molecule, no loader for a target with one value per atom, no
fold rule that keeps the atoms of one molecule together, and no co-field drawn around an atom.
Everything else carries over: the kernels in `cmfqsar/kernels.py` are bilinear sums over Gaussians
and do not ask what the Gaussians belong to; the regression, the GP variance, the `prop:` fields
for quantum inputs, the conformer averaging, the Shapley and difference-field tools all apply to
an environment as they apply to a molecule. M5 (site-resolved models) asks for the same three
missing pieces on the periodic side, site-centred features, per-site targets and a per-site
decomposition; the data model and the fold rule are to be designed once and shared. The reference
under rule 1 is an atom-centred count kernel (counts of atom types in distance shells about the
centre), which is what a per-atom increment scheme is, and every item below is measured against it
on the same folds.

**Small (days)**

* **LP1. Per-atom targets in the data model.** A target that is an atom property (one number per
  atom, blank where the atom has none), a sample index (molecule, atom) carried through loading,
  fitting and prediction, and cross-validation folds grouped by molecule, since atoms of one
  molecule are not independent samples; the grouped folds of `cmfqsar/chemotypes.py` are the
  pattern. Output as atom properties on the written SDF, so that a predicted shift or pKa sits
  on the atom it belongs to. No model yet; LP2 and M5 both need it first.
  **Done 27 October 2026, as the data model it was asked to be.** `cmfqsar/atomtargets.py`:
  `read_atom_target` reads one number per atom and *not a number* where the atom has none, in
  both conventions, with a blank field between two commas meaning an unmeasured atom and a list
  as long as the heavy atoms spread over them; a length that is neither is an error naming the
  molecule and both counts. `AtomTargets` is the index of the measured (molecule, atom) pairs and
  keeps the molecule of every sample; `AtomTargets.folds` holds out whole molecules, which is the
  grouped-fold rule of `cmfqsar/chemotypes.py` with the molecule as the group, so a shift of a
  molecule in the training set can never be tested against another shift of the same molecule.
  `scatter` spreads a prediction back over the molecules and `write_atom_predictions` writes an
  SDF carrying the measured and the predicted value on the atoms themselves, in RDKit's
  `atom.dprop.<name>` spelling, which is the only one that survives the writer.
  `Dataset.set_atom_activity` and `load_dataset(..., atom_activity=...)` carry it beside whatever
  molecule-level activity the file has, and `cmf atoms` reports what a file holds, what the folds
  would look like and writes it back. `candidate_targets` lists what *could* be a target, which
  is deliberately a looser rule than the one for a per-atom field: a property with gaps cannot be
  a field and is exactly what a target looks like.
  **No measurement, because there is nothing yet to measure**, which is what the item says: there
  is no model over atoms until the local kernel exists. What stands in its place is 20 tests of
  the properties the later items will rely on -- that a blank is not a zero, that a molecule is
  never split between two folds, and that a value written out comes back on the same atom.
  **Not done:** no model, no per-atom q2, and no GUI; the file for LP2's measurement
  (nmrshiftdb2) is not in the repository and has to be fetched when that item is taken.

**Medium (one to two weeks each)**

* **LP2. Atom-centred environments and the invariant local kernel.** A builder that turns one
  molecule into N samples: the molecule's fields in a frame with the origin on atom a, multiplied
  by a Gaussian envelope of width ρ about a. The product of two Gaussians is a Gaussian, so the
  envelope keeps the closed-form overlap and the atom decomposition; ρ is a hyper-parameter like
  α, short for a shielding (a few Å), longer for an acidity. The frame-free kernel is the
  autocorrelation of `cmfqsar/invariant.py` taken from the centre instead of over all pairs,
  which is the site-centred feature row that M5 names, and it is the first thing to measure
  because it needs no alignment: carbon shifts on a subset of nmrshiftdb2 of a few thousand
  atoms, and carboxylic acid pKa with the difference field of acid and anion (`cmfqsar/fielddiff.py`)
  as the sample, since the free energy of deprotonation is a response to that change. Gate: beats
  the count kernel by more than the interval of rule 4 on both.
  **Done 27 October 2026, and the gate splits: it passes on the shifts and fails on the
  acidity, for a reason the measurement names.** `cmfqsar/localenv.py`: `environment` multiplies
  a molecule's fields by `exp(-rho d^2)` about one atom and returns an ordinary `MoleculeFields`
  with the centre at the origin and the envelope folded into the weights, so an environment is a
  field of the same class as a molecule's and every kernel, learner and decomposition applies to
  it unchanged; `member_atom_map` keeps which atom of the molecule each atom of the environment
  came from, and `kept_share` reports what the 1e-4 floor dropped. `difference_field` is the
  signed counterpart of `combine_members`, which is what the acid-minus-anion sample needs, since
  averaging members requires non-negative weights. `local_features`/`local_kernel` take the
  rotation-invariant autocorrelation of `cmfqsar/invariant.py` on an environment, over a shorter
  distance axis (8 A in steps of 0.5 A). The reference of rule 1 is `shell_counts`: the one-hot
  type of the centre plus the counts of atom types in six distance shells about it, which is a
  per-atom increment scheme written as counts. `cross_validate` and `cross_validate_lams` run the
  folds of `AtomTargets.folds`, which hold out whole molecules, and get every ridge value from one
  eigendecomposition per fold (`_ridge_path`, verified against `krr_fit`); `paired_interval`
  bootstraps over **molecules**, not atoms, because the atoms of one molecule are not independent.
  `cmf local` fits either set from the command line, with `--reference` for the comparison and
  `--out` to write the out-of-fold predictions back onto the atoms. 23 tests.
  Both data sets are new and prepared by `examples/make_local_examples.py` from free downloads,
  with origin, licence and every filter in `examples/local/README.md` and in
  `docs/public_3dqsar_datasets.md`: 400 molecules with 5096 assigned 13C shifts from nmrshiftdb2
  (the chloroform spectra only, because a shift depends on the solvent), and 912 carboxylic acids
  with the acidity of the carboxyl group from the collection of Baltruschat and Czodrowski (2020).
  **Shifts: the gate passes, and by a wide margin.** The local field gives q2 +0.973 and an error
  of 8.8 ppm at an envelope width of 1.0 A, against +0.853 and 20.5 ppm for the atom-centred
  count; the difference in q2 is +0.120 [+0.107, +0.133], and at the module's default width
  (1.4 A, fixed before either set here was measured) +0.109 [+0.097, +0.123]. The width decides the result: +0.902 at 2.0 A width, +0.973 at
  1.0 A, +0.969 at 0.7 A, which is the behaviour a short-range property should show.
  **Acidity: the gate fails, and the reason is worth more than the number.** The best plain
  environment is at a width of 3.2 A -- the plan's expectation that an acidity is felt further away
  than a shift is confirmed in direction -- but it gives +0.582 against +0.541 for the count, a
  difference of +0.041 [-0.013, +0.094], inside the interval, so under rule 4 it is not a result.
  The **difference field**, which the item proposed as the sample, is far worse: +0.028 at that
  width, -0.512 [-0.570, -0.451] against the count. The diagnostic added for this says why: the
  mean kernel between two difference-field samples is 0.998 or more at every width, and 0.9999 at
  the three widest, because every carboxylic acid loses its proton in nearly the same way, so the
  difference of the two fields is
  nearly the same pattern for all 912 of them and the samples are indistinguishable after
  normalisation. The free energy of deprotonation is a response to that change, but what tells two
  acids apart is not the change itself -- it is what surrounds it, which is exactly what the
  difference removes. The same column explains the plain environments: a mean kernel of 0.957 at a
  width of 1.4 A, because the first shell of a carboxyl group is the same group in every molecule.
  **What this says for the items below.** LP3 should take the shifts, where there is signal to
  interpret, and should not expect its co-field from the difference field; its stated gate -- the
  meta/para picture of `benchmarks/results/hammett/` out of per-atom pKa data -- now looks out of
  reach on this set, because the local kernel does not beat a count on it at all. LP4's protonation
  ensembles and LP5's scale both wait on LP3.
  **Not done:** no local alignment and no co-field around an atom (LP3), no panel in the
  application, and no attempt at the shift sets that would make this a shift predictor: 400
  molecules with one force-field conformation each is far from the one to three ppm that programs
  built for the task reach with tens of thousands of spectra and calculated geometries.
* **LP3. Local alignment and the co-field around an atom.** For a congeneric site (the carboxyl
  group, an aromatic carbon with its two neighbours) a superposition on the centre and its first
  shell is well posed, and the overlap kernel in that frame gives what the invariant kernel
  cannot: a co-field around the atom that says where in its surroundings, and in which field, a
  feature raises or lowers the property, the local analogue of the Hammett co-fields. Needs the
  local superposition, `cofields.py` on an environment, and a panel in the application that shows
  the environment with its co-field instead of the molecule. Gate: the pKa co-field of the
  substituted benzoic acids reproduces the meta/para picture of `benchmarks/results/hammett/`
  from per-atom data alone.
  **Done 27 October 2026, and it splits the other way from LP2: the frame gains on the model,
  and the co-field reproduces the substituent table once it is read by atom type instead of by
  charge.** `cmfqsar/localalign.py`:
  `local_frames` fixes the rotation of an environment from two of the centre's bonded neighbours,
  grouped by **atom type and nothing else** -- a frame that depended on which of two ring bonds is
  a hundredth of an angstrom longer would not be a rule -- and every choice the type leaves open
  becomes a pose, which `posed` averages into one mean field over the frames the site allows
  (`combine_members`), the device the symmetrised Hammett set uses for a whole molecule applied to
  one atom. `ring_frames` is the site the item names: an aromatic carbon of a six-membered ring,
  the bond leaving the ring on +x, the ring in the xy plane, the two mirror images kept;
  `RING_POSITIONS` holds the four substituent positions that frame fixes, and a carbon where two
  rings meet is refused rather than posed anyway. `overlap_kernel` builds the `AtomTable` and the
  ordinary CMF overlap kernel on the posed environments, so the co-field machinery applies
  unchanged. `cmfqsar/cofields.py` gains `cofield_at_points`, `cofield_covariance_at` and
  `contrast_at`: the co-field at places a chemist can name, with the error the noise in the
  measured values allows and with the covariance between places included, which a comparison of
  two positions needs and which two separate error bars get wrong. `cmf local --align --site ring
  --cofield FILE.cube` writes the co-field in the frame of the pose. 20 tests, and
  `benchmarks/run_local_cofield.py` on 1225 aromatic carbons of 170 molecules of
  the shift set, folds holding out whole molecules.
  **The frame gains where the envelope is wide and nothing where it is narrow.** Posed overlap
  against the rotation-invariant local kernel at the same width: +0.756 [+0.718, +0.791] at a width of 4.5 A; +0.522 [+0.477, +0.572] at a width of 3.2 A; +0.096 [+0.069, +0.124] at a width of 2.0 A; +0.042 [+0.022, +0.061] at a width of 1.4 A; +0.025 [-0.015, +0.060] at a width of 1.0 A. The last is inside its
  interval. Each at its own best width, posed at rho 0.25 (+0.899) against invariant at rho 0.5 (+0.847), +0.052 [+0.033, +0.071]. Against the atom-centred count the posed
  kernel is +0.131 [+0.098, +0.170] at its best, q2 +0.899 and 4.22 ppm against +0.768 and
  6.38 ppm. The reading is that an invariant descriptor of a **wide** environment has only
  distances, and at a width of 4.5 A the surroundings of an aromatic carbon are nearly the same
  distances in every molecule, while the pose keeps the directions that separate an ortho position
  from a meta one; squeezed to 1 A there is little left but the first shell, which distances
  describe perfectly well.
  **The co-field says the large thing and not the subtle one.** Read at the four ring positions
  with the noise in the shifts propagated, one feature stands far above the rest: the steric field
  at the **ipso** position, +15.5 +- 1.3, twelve standard deviations from zero, where nothing
  else reaches four. It
  says that putting anything bulkier than hydrogen on the measured carbon raises its shift, which
  is the dominant term of the 13C substituent effect and true of nearly every group (methyl +9
  ppm, chlorine +6, nitro +20, methoxy +31). **The alternation around the ring is not
  reproduced**: the electrostatic co-field differs between para and meta by -18.5 +- 24.7, the wrong
  sign and inside the error, and no other field reaches two standard deviations either way. Its
  signs agree with the measured methoxy shifts at 3 of the four positions (r +0.55), but
  the y-scramble null for a correlation over four points is far too wide to call that a result.
  **Why no better was to be expected from these fields:** a nitro group, whose nitrogen is
  *positively* charged, also raises the ipso carbon and raises the para carbon where methoxy
  lowers it, so no model linear in fragment charges can follow both groups -- what separates them
  is resonance, which fragment charges do not carry. That is the same wall `run_hammett.py` met
  for sigma_p, where quantum charges on the whole molecule were the remedy.
  **The gate as written could not be measured, so the item was taken on the shifts, where LP2
  found signal.** Its gate is the pKa co-field of the substituted benzoic acids reproducing
  the meta/para picture of `benchmarks/results/hammett/`. LP2 had already found no signal to interpret on that set, and
  the carboxyl half of this benchmark confirms that the missing frame was not the reason: the
  carboxyl carbon poses uniquely -- two oxygens and one carbon, which the atom types separate, so
  nothing is averaged -- and the posed kernel loses to the count, -0.101 [-0.164, -0.037], and differs
  from the invariant kernel by -0.142 [-0.198, -0.088]. A co-field read off a model that does not predict
  would be an interpretation of noise, so none is drawn.
  **The indicator fields do reproduce the table, which the electrostatic field could not.** A
  partial charge cannot tell a pi-donor from a pi-acceptor; an atom type can. With one continuous
  indicator field per Sybyl type (17 of them, 22 fields in all, envelope width
  4.5 A, q2 +0.882 and 4.55 ppm out of fold, against +0.872 and 4.75 for the five fields alone
  at that width) the
  same machinery reports, for each type and each ring position, **what one atom of that type
  there adds to the predicted shift in ppm**: `cofields.atom_effect`, the closed-form overlap of
  the co-field with that atom's own Gaussian, with the weight the envelope gives it at that
  distance, checked in the tests against what the model's own prediction does when the atom is
  added. Over the 15 (type, ring position) pairs with a measured value -- the ipso position left out, since it is the largest effect in every row -- the correlation is +0.89 at a slope of 0.46, and +0.72 with ipso included. The comparison the electrostatic field could not make comes out: the amino nitrogen lowers the para carbon (-5.9 +- 1.4 ppm against -9.8 measured, four standard deviations from zero) while the sp2 carbon raises it (+0.7 +- 0.7 against +4.2), and the ether oxygen sits with the donor (-1.5 +- 0.9). So the local co-field does carry the substituent table; it was the field it was read from that could not. The sizes are compressed -- root mean square difference per type Cl 0.7, C.2 1.8, C.3 2.5, N.pl3 8.1, O.3 14.0 ppm -- by the ridge, by the envelope (a para atom enters at 0.41 of itself) and by a set whose rings mostly carry several substituents at once.
  **Not done:** no panel in the application (the co-field leaves as a cube file), and no quantum
  charges for the shift set, which is what the *electrostatic* co-field would need to carry the
  resonance that the indicator fields carry by type.
* **LP4. Conformers, tautomers and states for a local target.** A shift is a conformer average
  and a pKa is a property of a site in one protonation state. The ensemble builder gives the
  first with no extra work once an environment is a field; the second needs members with their
  own atoms, since acid and anion differ by an atom, which D4 has built. The site assignment (which atom carries the
  macroscopic pKa, which tautomer is in solution) is data preparation under rule 6, scripted per
  set, not a module.
  **Done 27 October 2026: averaging over conformations costs accuracy on the shifts and
  does not pay on the acidities.**
  `cmfqsar/localenv.ensemble_environment` averages an atom's environment over the conformations of
  its molecule. The averaging needs no new kernel -- an environment is a field, so the mean of the
  per-conformation environments is a field and the kernel on it is the mean-map kernel of
  `combine_members`, which is the item's "no extra work once an environment is a field". What it
  cannot be is the environment of the mean field: the atom is in a different place in every
  conformation, so each conformation has its own environment centred on its own copy of the atom.
  With a frame the order is pose first and average second (`cmfqsar/localalign.posed_ensemble`),
  because the site moves with the conformation; posing a mean field would put one rotation on
  geometries that do not share it. The conformations keep a common atom list, the union of what
  the envelope floor keeps in any of them, so the mean is over the same atoms. `cmf local
  --conformers N --conformer-weighting uniform|boltzmann`, 11 tests.
  **Measured** with 5 conformations per molecule from distance geometry refined by the
  force field, the prepared geometry kept as the first of them, on the same folds as LP2:
  on the shifts, the prepared geometry gives 8.82 ppm and the ensemble 11.63 ppm weighed equally, -0.020 [-0.024, -0.017] in q2, or -0.016 [-0.019, -0.014] weighed by the force field's energies; on the acidities, -0.020 [-0.061, +0.025] and -0.032 [-0.070, +0.013]. Both shift rows are outside their intervals and both acidity rows include zero, so the loss is measurable on the shifts and only a direction on the acidities; none of the four is positive.
  **The control is the second row and it is what makes the result readable.** Using a different
  single conformation -- the lowest in energy of the same pool, which differs from the prepared one
  for 285 of the 400 molecules on the shift set -- costs -0.000 [-0.001, +0.000], nothing.
  So the loss is
  not that the generated geometries are worse than the prepared one; it is the averaging itself.
  The reading is that the mean field smears every atom over the positions it takes in the
  conformations, and at an envelope width of 1.0 Å what is left inside the envelope is the
  first
  shells, which every conformation shares. The averaging therefore adds blur and no information.
  On the acidities the envelope is wide enough (3.2 Å) to reach the parts that move, which is
  where averaging had its best chance; the point estimates go the same way and the intervals
  include zero, so there the honest statement is that it does not pay rather than that it costs.
  **What this says about the earlier items.** LP2 and LP3 both named the single force-field
  conformation as their weakest part. On this evidence it is not: the model is insensitive to
  which conformation it is given and worse when given several. What a shift averages over is the
  thermally populated set in solution, and five conformations ranked by a force field are not
  that; the way to test the premise properly is a populated ensemble from molecular dynamics or
  from a quantum ranking, which is D7's trajectory reader rather than this item.
  **Not done:** no tautomers and no protonation states. The item asks for them, and the site
  assignment they need is data preparation under rule 6, scripted per set. Neither set here
  offers it: the nmrshiftdb2 records are deposited in one tautomer with no evidence that the
  measured compound was a mixture, and for the acidities the protonation state is the acid by
  definition, so weighing acid against anion at a pH would use the answer to build the sample.

**Large (a month or more)**

* **LP5. Scale.** nmrshiftdb2 and the DFT shift sets hold hundreds of thousands of carbons, past
  what an exact kernel handles; a Nyström or inducing-point GP over environments (part B's scale
  work applied to the local kernel) is the route, and it waits until LP2 and LP3 have shown on the
  small sets that the co-field says something a chemist recognises. The ShiftML precedent (SOAP
  with Gaussian-process regression, about 0.5 ppm on protons in molecular crystals) is the number
  to reach, and the co-field is what CMF would add to it.
  **Done 27 October 2026, and it answers the item's premise rather than carrying it out:
  the scale is free and the data is not what is missing.** `cmfqsar/locallarge.py`. The
  rotation-invariant local kernel is a linear kernel on a finite feature map -- the
  autocorrelation of an environment is 255 channels for the five fields, scaled per block
  and cosine-normalised -- so the ridge can be solved in the space of those coefficients rather
  than the space of the samples. `normalised_rows` stops `autocorrelation_kernel` one step early
  and returns the rows whose inner products are the kernel; `primal_fit`, `cross_validate_primal`
  and `choose_lam_primal` are the dual routines of `cmfqsar/localenv.py` computed the other way
  round. It is the same model, not an approximation: on the 5096 carbons of the repository set
  the two sets of out-of-fold predictions differ by 2e-06 ppm, with the folds taking
  13 s and 0.1 s and the memory 0.19 GB against 10 MB. 11 tests,
  and `benchmarks/run_local_scale.py` with `--data-root`, since the set it needs is too large for
  the repository (`docs/public_3dqsar_datasets.md` says how to build it).
  **The learning curve is flat.** On 58705 carbons in 6000 molecules built
  from the whole nmrshiftdb2 dump, cross-validated in 7 s where an exact kernel would
  have been 26 GB: 100 molecules 10.84 ppm; 200 molecules 10.13 ppm; 400 molecules 9.73 ppm; 800 molecules 9.61 ppm; 1600 molecules 9.70 ppm; 3200 molecules 9.83 ppm; 6000 molecules 9.75 ppm. Sixty times the data buys 1.23 ppm and stops paying
  after about eight hundred molecules.
  **What the data could not give, resolution gives.** At the full size, halving the width of the
  distance bins (255 channels to 495) takes the error from 9.75 to
  8.35 ppm, and halving again (975 channels) to 7.76 ppm -- 1.99 ppm
  against the 1.23 ppm that sixty times the data gave. **The model is
  representation-limited, not data-limited**, so the route to the ShiftML number this item names
  is a richer descriptor, which is C5, and not a larger training set, which is this item. The
  scale work is still worth having, because it is what made the question answerable in seconds,
  but it is not the bottleneck it was expected to be.
  **The overlap kernel is the one that needs landmarks, and they work.** Its feature map is a
  function on space, so the dual form is the only exact one; `nystrom_kernel` approximates it from
  landmark environments. On the 1225 aromatic carbons of LP3, 200 landmarks
  (16 per cent of the rank) reproduce the exact q2 of +0.899 and
  100 already give a relative kernel error of 0.001. The environments
  of one site are alike, which is why so few span them. That matters because the exact overlap
  kernel took 71 s on those 1225 samples and grows with the square: the
  58 705 carbons above would be about 45 hours and 26 GB without it.
  **Not done:** no inducing-point Gaussian process, which is what the item asks for by name; the
  landmarks here approximate the kernel and are then used in the ordinary ridge, which is enough
  to measure what the approximation costs but is not a sparse GP with its own variance. No proton
  shifts and no DFT shift sets, and so no comparison with ShiftML on its own ground.

**Order.** LP1 first and shared with M5. LP2 next, invariant kernel and pKa by difference field,
because it needs no alignment and settles whether the environment field carries the property.
LP3 after LP2 has reported, since its co-field is the point of the exercise but costs the alignment
step. LP4 as D4 arrives, LP5 only if LP3 passes its gate.

### Supramolecular chemistry and host–guest delivery systems: the extensions the fixed host asks for (26 October 2026)

A host–guest complex is the receptor–ligand problem with the receptor known to the atom, and most
of the program applies to it as it stands. What exists: `examples/cyclodextrin/` holds 2671 guests
of seven cyclodextrin hosts with log K as the activity, and two of the hosts (HP-β-CD and
SBE-β-CD) are the excipients of marketed formulations. `benchmarks/run_cyclodextrin.py` measures
the three kernels on every host. The rotation-invariant autocorrelation kernel is ahead of the
aligned kernel on six hosts of seven (mean q² +0.446 against +0.341, atom-type counts +0.323),
because the guests of one host share no scaffold, so no superposition exists to curate. The probe
module (`cmfqsar/probes.py`) already carries a +1 cation probe, so the field an ionophore presents
to a cation is one option of the field set. The two-endpoint model of `cmfqsar/multitask.py` was
written for the Am/Eu extractants, which is itself a host–guest problem; it gives the selectivity
of a host between two guests, or of a guest between a carrier and its biological target, as a
co-field. The `complement` command reads a structure in PDB or mmCIF format and scores guests
against it by field overlap, and a macrocyclic host written as PDB goes through the same path.
The conformal option of `build` gives every predicted constant an interval.

Most questions a delivery formulation asks are functions of the binding constant K, so a model of
K answers them. The solubility gain is the intrinsic solubility times (1 + K·[host]) by the
phase-solubility equation. The release half-life follows from K when association is
diffusion-limited, which it is for small guests. The fraction of the complex that survives
dilution in blood, or displacement by cholesterol and serum proteins, is a mass-action
calculation from K and the concentrations. What the method does not reach is listed at the end.
The items below are what the fixed host asks for beyond a drug series, ordered by the code each
needs. Rule 1 applies with two references: on a guest table of one host, the invariant kernel at
the numbers above; on a series of hosts, the five-field aligned model.

**Small (days each)**

* **SC1. The two-endpoint model on the command line.** `cmfqsar/multitask.py` is reached only
  from `benchmarks/run_multitask.py` or from a Python script. Needed: a `--second-endpoint
  <column>` option of `build`, the task correlation chosen by the evidence or given, both
  endpoints reported by `predict`, and the selectivity written beside the model as a co-field
  when the kernel has one (aligned) and as channel weights otherwise (invariant). The measurement
  needs no new data: 496 guests appear in both the α-CD and the β-CD tables, and the selectivity
  between the two cavities is the first supramolecular two-task problem. Gate: on the shared
  guests the two-task fit is not worse than the two single fits by more than the interval of
  rule 4, and the selectivity channels name the size of the guest, which is the known difference
  between the two cavities.
  **Done 26 October 2026. The gate passes on both halves, and sharing gains nothing.**
  `build --second-endpoint COLUMN` reads the second endpoint from a property of the same file,
  chooses the correlation by the marginal likelihood (or takes `--task-rho`), reports the
  ten-fold cross-validation of both endpoints and of their difference with whole molecules held
  out, and installs the two-task coefficients in the model it saves. `predict` prints both
  endpoints and their difference and writes them as two more columns; `cofield --endpoint
  second|selectivity` draws the other endpoint's maps or the selectivity ones, which for the
  aligned kernel is the selectivity co-field and for the invariant one the channel weights.
  `examples/make_two_endpoint_example.py` joins two shipped tables into such a file
  (`beta_with_alpha.csv`, 1162 guests of the beta table, 496 of them also measured
  against alpha).
  **Measured by `benchmarks/run_two_endpoints.py`** on three pairs of cavities, one invariant
  kernel, the correlation by the evidence, the intervals bootstrapped over the molecules.
  Sharing: beta -0.004 [-0.011, +0.004], alpha +0.005 [-0.008, +0.019] for that pair; beta -0.001 [-0.006, +0.004], gamma +0.032 [-0.011, +0.082]; alpha -0.002 [-0.009, +0.004], gamma
  -0.008 [-0.029, +0.016]. Every interval contains zero, so the first half of the gate passes in the form it was
  written -- not worse -- and the fair reading is that an endpoint with a few hundred molecules
  of its own gains nothing from another.
  **The selectivity is predictable and it names the size of the guest**, which is the second half
  of the gate: q2 +0.273 (beta - alpha, 496 guests), +0.223 (beta - gamma) and
  +0.389 (alpha - gamma); the correlation of the predicted selectivity with the heavy-atom
  count is +0.468, -0.739 and -0.676, against +0.382,
  -0.480 and -0.613 for the measured values. The sign follows the ordering of the
  three cavities exactly: a big guest prefers the wider cavity, and nothing told the model that a
  cavity has a size.
  **Not done:** the primary endpoint must be present on every row, since a molecule without it is
  dropped before any model is built, so a guest measured against the second cavity only is not a
  training molecule on the command line (the benchmark, which works at library level, does use
  them); more than two endpoints are still reached from the application or from Python; and the
  channel reading of the selectivity is printed as numbers, not drawn.
* **SC2. Delivery quantities from the constant.** The three formulas above are not printed by the
  program. Needed: a post-processing option of `predict` that takes the host concentration and,
  where given, the intrinsic solubility, and reports the solubility gain, the fraction bound at a
  given dilution and the release half-life under the diffusion-limited assumption, each with the
  conformal interval of K carried through. Under rule 5 this is a display tool unless a
  phase-solubility set is found to test it. The formulas are standard and a spreadsheet does the
  same, so the item is convenience and is kept small.
* **SC3. A macrocyclic host as a pocket.** `complement` reads proteins. A cyclodextrin,
  cucurbituril or calixarene comes as an SDF file or as a CIF file from the Cambridge Structural
  Database, with no residues. Needed: a reader in `cmfqsar/protein.py` for a host given as a
  molecule file, and a split of the host into its monomer units (one glucose, one glycoluril) by
  ring system, so that the residue profiles of the dialog become monomer-unit profiles. Guests
  are posed into the cavity by the pocket placement of `cmfqsar/docking.py`. The measurement is
  the largest a complementarity score has had: 1162 β-CD guests with a measured constant in one
  pocket, where no protein set in the repository offers more than a few dozen ligands per pocket.
  Gate: the complementarity score correlates with log K. If it does not, the result stands as the
  limit of a one-body score on an endpoint that the solvent drives.
  **Done 27 October 2026, and the answer is the one the item allowed for: it does not
  correlate.** `cmfqsar/protein.py` gained `read_molecule_host`, `host_from_molecule`,
  `monomer_units` and `cavity_frame`: a host given as an SDF, MOL, MOL2 or PDB file is read with
  its repeating units as the residues, the units being its small ring systems with every atom
  outside a ring given to the unit it hangs on and the macrocycle that joins them left out, so
  every per-residue tool of the pocket machinery becomes a per-unit tool. `read_structure`
  dispatches on the extension, so `complement --structure host.sdf` works; `--place-in-cavity`
  threads each guest into the channel -- centroid on the centre of the ring of units, longest
  axis along it -- and refines it there with `cmfqsar/docking.py`, which takes the clashes of a
  guest from dozens to about one. `examples/make_host_structures.py` builds the hosts from the
  SMILES in `hosts.csv`, keeping the most symmetric of forty conformations, a rule that uses the
  symmetry of the molecule and nothing about the guests; the three cyclodextrins come out with
  cavity radii of 2.6, 3.0 and 3.9 A for six, seven and eight sugars, against published values
  near 2.4, 3.1 and 4.0.
  **Measured by `benchmarks/run_host_pocket.py`**, the largest set a complementarity score has
  had: 1162 beta-cyclodextrin guests in one cavity, where no protein set here offers more
  than a few dozen ligands per pocket. The score against log K is -0.348 on beta,
  -0.206 on alpha and
  +0.025 on gamma; the burial gives
  -0.077 on beta. The guest's heavy-atom count alone gives
  +0.348, and with the size held constant the score adds -0.285.
  On beta the hydrophobic term, which a hydrophobic cavity should reward, is the most
  negative of the four (-0.405); on gamma, whose cavity is too wide for these guests to
  touch its walls, every correlation is near zero.
  **The result stands as the limit of a one-body score**, which is what the item said it would
  if it came out this way. Binding to a cyclodextrin is driven by the water that leaves the
  cavity, and an overlap between the guest's fields and the host's has no term for the solvent it
  displaces. The score is a check on a pose, not an affinity, and this is the clearest statement
  of that the repository has, because nothing else here has a thousand ligands in one pocket.
  **Not done:** the host structures are embedded conformations, not crystal ones, so a reader
  with a CSD entry should use it instead; the guest gets one conformation and one pose; and the
  monomer split is by ring system, which is right for the hosts tried and would need a rule of
  its own for a host whose unit is not a ring.
* **SC4. Host–guest sets curated by conditions.** `make_cyclodextrin_examples.py` keeps one window
  of temperature and pH and the median of repeated measurements, and the spread between
  laboratories reaches 3.2 log units for one guest. A second host family should be prepared by
  the same script pattern, with the medium, the counter-ion and the stoichiometry as columns:
  cucurbit[7]uril and cucurbit[8]uril from the published tabulations, and the SAMPL host–guest
  challenge sets (CB7, octa-acid, TEMOA), which are small but measured to a tenth of a
  kilocalorie per mole and are the reference sets of the binding free-energy community. Data
  preparation under rule 6, scripted per set. The first run on each set includes the water probe
  field, which is the closest the field set comes to the hydrophobic effect, so that its value is
  settled by measurement and not by argument.

**Medium (one to two weeks each)**

* **SC5. Host series with a frame given by the cavity.** This is the measurement EC7 names, and
  most of its code exists: the +1 probe, the aligned kernel and the two-task model. What is
  missing is a frame convention. For a crown ether, a cryptand or a calixarene the natural frame
  has its origin at the centroid of the donor atoms and its z axis along the principal axis of
  the donor ring. `cmfqsar/align.py` should offer it as a `cavity` frame beside the substructure
  and field superpositions, so that a host series is aligned by its cavity and not by a scaffold
  that differs between hosts. The cation-specific radii of EC7 are the remaining part of the
  probe. Measure on log K of Na⁺ and K⁺ for crown ethers and cryptands in methanol from the Izatt
  tabulations, the electrostatic field against the probe field, and the Na⁺/K⁺ selectivity with
  SC1. A host series has a physical frame, which few drug series have, so the co-field is
  expected to be readable. Gate: the co-field of the K⁺ constant names the cavity size and the
  donor positions.
* **SC6. What an invariant model says about a guest.** The invariant kernel has no co-field, so a
  model of a guest table cannot show where on the guest binding is gained or lost, which is the
  question a formulation chemist asks. The channel sum is a sum over the atoms of the molecule,
  so it splits into one contribution per atom, as the overlap kernel does; this is the
  decomposition M5 asks for, on a molecule rather than a cell. Needed: the per-atom contribution
  in `cmfqsar/invariant.py`, atoms coloured by it in the viewer, and the chart of weight against
  distance per pair of field types that NT9 asks for. Gate: on β-CD the contributions name the
  group that enters the cavity (adamantyl, phenyl, cyclohexyl), which is known chemistry.
  **Done 26 October 2026, and the gate passes on beta-cyclodextrin.**
  `cmfqsar/invariant.py` carries `channel_coefficients`, `atom_contributions` and
  `pair_contributions`: a channel is a double sum over atom pairs and the prediction is a fixed
  linear functional of the channels, so a pair is one term of the prediction, divided equally
  between its two atoms. The numbers add up to the prediction minus the bias exactly, for kernel
  ridge and for a support-vector model alike, and the tests assert that to the last digit.
  `CMFModel.atom_contributions` now answers for this kernel as it does for the aligned one, so
  the report lists the largest contributions and the viewer colours the atoms by them with no
  further work; `--ac-chirality` is refused by name, since a triple has no place in a split over
  two atoms.
  **Measured by `benchmarks/run_guest_atoms.py`** on three cyclodextrin tables, with the
  contributions taken out of fold so that a guest's own constant never enters its own
  explanation, hydrogens reported on their heavy neighbour, and the statistic the mean share per
  heavy atom inside a group minus the mean outside it. On beta-cyclodextrin (1162 guests,
  invariant q2 +0.447) the three groups that enter the cavity are all positive --
  adamantyl +0.0646 (6 guests), cyclohexyl +0.0378 (122), phenyl +0.0158 +-
  0.0041 (817) -- and the three polar groups all negative: carboxyl -0.0366,
  hydroxyl -0.0916, amine -0.0886. The order among the three is the order of the affinity
  they are known to give.
  **On the other two hosts the reading does not hold**: alpha-cyclodextrin, whose cavity is too
  narrow for a benzene ring, gives phenyl -0.0158 and carboxyl +0.0703; gamma, whose
  cavity none of these groups fills, gives all three below zero (-0.0333 phenyl,
  -0.0558 cyclohexyl). The arithmetic is exact on all three, so what differs is whether the
  group a chemist would name is the one that host's data makes the model lean on.
  **It is a statement about a series, not about one molecule.** Against a null that shuffles the
  numbers among the atoms of the same molecule, the group's z is +0.35 for phenyl and
  +1.38 for adamantyl: within one guest the group does not stand out far above the spread
  of its own atoms, and what is solid is the mean over hundreds of guests.
  **Not done:** the chart of weight against distance per pair of field types (NT9) is not drawn;
  the split over field *pairs* is reported as numbers only; and the surface and periodic
  invariant kernels keep their own feature builders, so neither has the split yet.
* **SC7. Design on a guest table.** `cmf design` runs on a topomer model only, because the
  candidate substituents need a pose. On an invariant model no pose is needed: a candidate built
  by substituent replacement is embedded once and its autocorrelation computed, and the ranking by
  prediction, posterior standard deviation and acquisition score runs unchanged. Needed: the
  design module accepting an invariant model, with the topomer step skipped. The field-difference
  annotation needs a frame and is replaced by the contributions of SC6. Gate: leave-one-fragment-
  out on β-CD, with the known strong binders recovered at the top of the list.
  **Done 26 October 2026, and the gate passes at the top of the list.**
  `design(pose="molecule")` builds a candidate as a whole molecule -- `whole_molecule` puts the
  core and its groups back together, which is the inverse of the cut, and `embed` gives it one
  conformation -- and the rotation-invariant kernel scores it without a frame.
  `cmf design --kernel autocorrelation` fits that model on the table and takes this route; the
  field-difference annotation, which needs a frame, is replaced by the atom contributions of
  SC6, reported as how much of the prediction comes from the group that was put in. The route is
  refused for an overlap kernel, where a candidate with no frame cannot be scored, and a flat
  acquisition score -- expected improvement when the best molecule of the table is out of reach,
  which is the normal case here -- is reported as not being a ranking.
  **Measured by `benchmarks/run_guest_design.py`** on the 1162 guests of the beta-cyclodextrin
  table, leave one group out: for each group of an eleven-group library of R-COOH, every guest
  carrying that group is removed and the library is ranked by a model fitted on the rest.
  Spearman +0.709 over the 10 groups whose acid is a measured guest.
  **The two kinds of row are not equally hard**: removing the molecule alone leaves the model its
  relatives (mean absolute error 0.26 over 6), removing every guest
  that carries the group does not (0.95 over 4, bias
  -0.73), and a group the table no longer holds is pulled towards the mean.
  **The gate is about the top of the list and it passes**: adamantyl, whose six guests measure
  2.9 to 4.3 against a median near 2 and whose own acid is not in the table, comes first of
  11 with all six removed; cyclohexylmethyl, the strongest measured group that was fully
  held out, comes second. Over every row the 4 groups measuring 3.0 or more have
  mean rank 5.0, so the list is useful at the top and not in the
  middle.
  **Not done:** the candidate gets one embedded conformation, not an ensemble, so a flexible
  group is represented by one shape; the library is a hand-made list of R groups and nothing
  enumerates it from the table; and the loop still varies one position of one parent.

**Large (a month or more)**

* **SC8. Porous carriers.** Drug loading in a MOF or in mesoporous silica is M7, the periodic pore
  frame, and nothing substitutes for it; SC8 is M7 measured on a drug-loading set. The data are
  the difficulty: published loadings are per formulation, with the solvent and the activation
  differing between studies, and the first task is the reproducibility check that the
  nanotechnology section prescribes.

**What this does not cover.** Carriers without a defined atomic structure: liposomes, micelles,
polymer nanoparticles and dendrimers have no host geometry. Partition into such a carrier can be
modelled from the guest side, but that endpoint is largely lipophilicity, and `cmf diagnose` is
expected to report that a one-dimensional descriptor does as well as a field. Ternary and 1:2
complexes (drug, cyclodextrin and polymer) break the one-guest-one-host sample of the data format
and wait until SC1 to SC7 have reported. The hydrophobic contribution to binding in water is a
property of the medium, not of the guest; the water probe is the closest the fields come, and SC4
measures it. Permeability, uptake and pharmacokinetics are not functions of a molecular field.

**Order.** SC1 and SC4 first: they need no new science, and they give the second host family and
the two-task command that every later item measures with. SC5 next, because a host series with a
cavity frame is the case where the co-field is expected to work, and EC7 has been waiting for its
set. SC6 before SC7, since the design loop on a guest table is unreadable without the
contributions. SC3 as the docking placement allows. SC2 at any time. SC8 is M7.

## I. Medicinal chemistry: the approaches the co-field makes cheap (months 6 to 12)

*Drafted 26 October 2026. Parts A to G build the model and part H leaves medicinal chemistry for
other fields; this part stays inside medicinal chemistry and asks a different question: which task
of lead optimisation does a finished CMF model already answer, and at what price. It is placed last
because it was written last, and not because it comes last. MC1 to MC4 can be done as soon as the
data they name are in hand.*

One identity organises the section. A model linear in the field predicts an activity as the overlap
of the co-field with the molecular field, and a molecular field is a sum over atoms. Two analogues
that share a core in one pose therefore differ in predicted activity by the overlap of the co-field
with the field difference of their two substituents alone:

`y(A) − y(B) = ⟨C, X_A − X_B⟩`, where `C = Σ_j a_j X_j` is the co-field and `X_A`, `X_B` are the
fields of the two substituents and of nothing else.

The identity is exact for the overlap kernel, it is local, and it needs no refitting. It also says
what interchangeability of two groups means in this program: their fields must agree where the
co-field has weight, which is not the same as their fields agreeing. That is the whole content of
field-based bioisosterism, and here it costs one integral per candidate group. Under rule 2 the
Gaussian layer breaks the identity, and the Shapley values of `cmfqsar/shapley.py` take its place
there.

What the items build on is already measured. `cmfqsar/design.py` and the `design` command enumerate
substituents at one position and rank them by prediction and posterior standard deviation (E3);
`cmfqsar/active.py` and `suggest` give six acquisition rules and the finding that the posterior
standard deviation of a CMF model is the distance to the training set in other units (E2);
`cmfqsar/multitask.py` gives the selectivity of two endpoints as a difference of co-fields (A6);
`cmfqsar/stereo.py` sees handedness on a synthetic set and has no real series to prove it on (C7);
`cmfqsar/fielddiff.py` gives the field difference of two molecules and `cmfqsar/complementarity.py`
the field a pocket would reward.

Rule 1 applies with two references: on a whole series the five-field single-conformation aligned
model, and on anything built out of substituents the topomer model of E3, whose cost is known (a
mean loss of 0.101 q² against the aligned model on the eight Sutherland sets). Of those eight sets
only GPB and THER have a scaffold that comes off at acyclic single bonds, so every substituent item
below is measured on those two and reports that limit, as E3's benchmark does.

**Small (days each)**

* **MC1. A bioisostere table weighted by the co-field.** `cmf design` ranks candidate substituents
  by the predicted activity of the whole molecule. The table a medicinal chemist asks for answers
  the other question: which groups are interchangeable at this position, so that one of them can be
  chosen for a property the model does not describe. The identity above gives that table directly,
  as one integral per pair of groups rather than one model fit. Needed: the field difference of two
  posed substituents restricted to their own atoms, which is a mask on the atom sums of
  `cmfqsar/kernels.py`; its overlap with the co-field; and a square table of a library against
  itself holding that distance and the predicted activity change, with the conformal interval
  carried through. Gate: matched molecular pairs taken from GPB and THER with RDKit's `rdMMPA`; the
  co-field-weighted substituent distance ranks the measured activity difference of a pair better
  than the unweighted field distance of the same two substituents, and better than a 2D fingerprint
  distance, by more than the interval of rule 4. If the weighting adds nothing then the ranking is
  shape similarity and needs no model, which is a result of the same kind as D5.
  **Done 27 October 2026, and the gate is passed on two of the three sets with no set against
  it.** `cmfqsar/bioisostere.py` and the subcommand `cmf isosteres`. The table needs no new
  model: the prediction of the overlap kernel is a sum over the atoms (decomposition rule 3), so
  the part of it that belongs to a substituent is the sum over that substituent's own atoms, and
  exchanging group A for group B is the difference of two such sums -- one integral per pair of
  groups rather than one model fit per group. The masking is exact and not an approximation,
  because the kernel is itself a sum over pairs of atoms; `masked_field` builds the subset as a
  field object of its own and a test checks that for any division of a molecule in two,
  K(whole, whole) = K(a,a) + 2K(a,b) + K(b,b). 26 tests.
  **The pairs** are matched molecular pairs from RDKit's `rdMMPA` as the gate asks: one single
  bond cut at a time, both pieces tried as the core, two molecules with the same core and
  different groups, two molecules that match on several cores counted once. 730 pairs on GPB,
  142 on THER, 751 on THR.
  **Measured** by `benchmarks/run_bioisostere.py` as the rank correlation between each measure
  of how far apart two groups are and the size of the measured activity difference, with the
  dual coefficients refitted without **both** members of every pair so that the predicted change
  has not seen the two activities it is judged against. GPB: fingerprint -0.025, field distance
  +0.005, predicted change +0.561. THER: +0.084, +0.162, +0.277. THR: +0.144, +0.242, +0.335.
  The gains of the predicted change over the two references are, on GPB +0.586 [+0.505, +0.669]
  and +0.556 [+0.479, +0.632]; on THR +0.191 [+0.106, +0.279] and +0.093 [+0.008, +0.181]; on
  THER +0.193 [-0.059, +0.437] and +0.116 [-0.089, +0.313], which contains zero and is the set
  with the fewest pairs. On no set is the co-field worse than either reference.
  **What this does and does not show.** The field distance is not a weak reference: it compares
  the two groups **in their poses**, so it already knows the shape and the charge each of them
  puts where, and the co-field still adds to it. What the co-field has that the references cannot
  is whether the activity depends on the place where the two groups differ. Against that, the gap
  between the honest column and the whole model is large (+0.561 against +0.631, +0.277 against
  +0.641, +0.335 against +0.711), so a table built from a model that has seen the compounds it is
  asked about will look better than it is; and the field types, the attenuation and the ridge
  were still chosen on all the molecules.
  **What the table is read for is the size rather than the ranking.** A predicted change smaller
  than twice the conformal half width is a change the model cannot see, and the groups inside
  that bound are the ones the table calls interchangeable; that part is the model's own error bar
  and not a claim about the chemistry. On the GPB model, whose half width is 1.5 log units, the
  bound covers every pair of the 35 groups at that position, which is the table saying honestly
  that this model cannot separate them. What is left is the order, and it is chemically
  reasonable: the groups it puts closest to the carboxamide are the methyl carbamate (+0.02),
  butyramide (+0.04) and acetamide (+0.05).
  This unblocks MC6 and MC7, which the plan puts after MC1.
* **MC2. A batch rule that refits between picks.** E2 names this as not done. A laboratory makes ten
  or twenty compounds at once, and the cosine filter of `cmfqsar/active.py` keeps a batch from being
  one molecule several times over without saying how much it loses. The posterior covariance is in
  hand, so a batch that maximises the determinant of its own covariance block is a greedy selection
  with one Cholesky update per candidate, which costs less than refitting. Gate: on the eight
  Sutherland sets, with the protocol of `benchmarks/run_active.py` and batches of ten, the
  determinant rule beats the cosine filter in both of E2's two measures (the best molecule acquired,
  and the error on what is left) by more than the bootstrap interval. E2's finding about the
  posterior standard deviation predicts that the gain will be small; that the cheap filter is
  enough is an acceptable outcome and closes the item.
  **Done 27 October 2026; the gate is not met, the cheap filter is enough, and the item is
  closed on the outcome it allowed.** `active.condition_on` is one Cholesky step with a
  candidate as the pivot, `active.batch_by_covariance` is the greedy batch built from it -- the
  acquisition recomputed after every pick from the conditional standard deviations, so there is
  no threshold anywhere -- and `active.batch_by_cosine` is the old filter, pulled out as a
  function of its own so that the two can be measured against each other.
  `CMFModel.predict_with_covariance` returns the whole posterior covariance of a block of
  candidates. With the variance acquisition the rule is the greedy maximisation of the
  determinant of the batch's covariance block, and a test checks it against the exhaustive best
  on a small matrix. `learning_curve` acquires a batch between fits and reports the error after
  the last one; `cmf suggest --batch-rule covariance` is the command line. 16 tests.
  **Measured** by `benchmarks/run_batch.py` with the protocol of `run_active.py` on the eight
  Sutherland sets, 20 molecules to start, 10 repeats, the same starting set for every rule, in
  batches of 10. Paired run by run, the covariance rule minus the similarity filter: expected
  improvement +0.037 [-0.015, +0.110] in the best acquired and -0.016 [-0.035, +0.002] in the
  error on what is left; the variance acquisition -0.137 [-0.267, -0.013] and +0.014 [-0.008,
  +0.037]. None of the four favours the covariance rule beyond its interval and one favours the
  filter, so the gate asked for by this item is not met.
  **The useful finding is beside the gate**: a batch costs nothing measurable. Against the
  one-at-a-time loop, which sees every measurement before choosing the next molecule and pays
  thirty-five fits for it, the batch differs by +0.025 [-0.025, +0.079] in the best acquired and
  +0.008 [-0.016, +0.033] in the error. A plate of ten chosen at once is worth as much as ten
  chosen one after another, at a tenth of the fitting.
* **MC3. A series in which both enantiomers are measured.** This is the one part of C7 left open,
  restated here because it is a data task of medicinal chemistry and not a kernel task. Needed: a
  series with enantiomeric pairs and a separate potency for the eutomer and the distomer, prepared
  by a script under `examples/` as rule 6 requires, with the configuration taken from the
  publication and not from a compound name. Beta-blockers, the profens and the published
  eudismic-ratio tabulations are the first places to look. Gate, two parts: the stereo channels of
  `cmfqsar/stereo.py` gain on this set, where they gained nothing on five sets that hold no
  enantiomeric pair; and the model puts the eutomer above the distomer in more pairs than chance,
  with the chiral part of the co-field near the stereocentre and nowhere else.
* **MC4. An anti-target panel and the selectivity co-field.** The two-task model gives the
  selectivity of two endpoints as a difference of co-fields, measured on the Am/Eu extractants, and
  E1 extends it to several tasks. Medicinal chemistry asks the same question of hERG, of a
  cytochrome and of a kinase panel: where may field be added for the target that is not also added
  for the off-target. What is missing is data in the program's format. Needed: one congeneric series
  with a potency and at least one anti-target activity measured on the same compounds, scripted from
  ChEMBL under rule 6, with the assay conditions as columns, because an anti-target number from
  another laboratory is a different measurement. Gate, two parts: the two-task fit of the potency is
  not worse than the single fit by more than the interval of rule 4; and the difference co-field
  places weight of one sign on the groups that are known to drive the off-target, which for hERG is
  the basic nitrogen and the lipophilic region beside it.
  **Done 27 October 2026; both parts of the gate are met, the second with a sign the item did
  not expect.** The data were the missing piece and are now
  `examples/antitarget/h3_herg.sdf`, built by `examples/make_antitarget_examples.py` from one
  ChEMBL document under rule 6: Lazewska and co-workers, *Novel and highly potent histamine H3
  receptor ligands. Part 1: withdrawing of hERG activity*, Bioorg. Med. Chem. Lett. 21 (2011)
  5378, whose own subject is this question. 78 of the paper's 224 compounds with both numbers
  form one series by the common-substructure rule; each carries an H3 Ki (displacement of
  [125I]iodoproxyfan) and a hERG IC50 (displacement of [3H]dofetilide), **one assay each from
  one laboratory**, with the assay descriptions as columns as the item requires. The spreads are
  0.68 log units each and the correlation is -0.22. 12 tests; `examples/antitarget/README.md` is
  the data statement.
  **The first part.** The task correlation by the evidence is -0.431. The potency reaches q2
  +0.341 with the second endpoint against +0.319 alone, which is +0.022 [-0.005, +0.048] paired
  over the molecules: not worse. The selectivity itself is predicted at q2 +0.471, better than
  either endpoint alone, which is the 2003 paper's claim met on another kind of series.
  **The second part**, read with `cofields.atom_effect` as what one more atom at that place
  would do to the predicted selectivity: at the basic nitrogen steric +12.84 [+9.68, +15.99] and
  hydrophobic +6.96 [+5.46, +8.47]; at the carbons within 5 A of it +11.56 and +7.35; over every
  other heavy atom +0.03 [-2.39, +2.45] and +0.70 [-0.77, +2.17], neither clear of the noise.
  The weight is of one sign and concentrated exactly at the groups the item names.
  **The sign is positive, not negative.** Lipophilicity does drive hERG in this series as
  expected -- cLogP correlates +0.38 with the hERG activity and -0.31 with the selectivity over
  a range of 3.0 to 7.0 -- but a co-field is about a place and cLogP is about a molecule. What
  the series says is that the lipophilicity costing selectivity is the lipophilicity far from
  the amine, while the amine region is where the H3 affinity is made; the paper kept that region
  and changed what is far from it. A whole-molecule number cannot make that distinction, which
  is the only thing a co-field adds.
  **What it is not**: one series, one paper, hERG by displacement rather than patch-clamp, and
  an alignment by common substructure rather than by a crystal structure. A second series
  measured another way is what would turn the reading into a rule.

**Medium (one to two weeks each)**

* **MC5. Scoring a candidate only where the co-field has weight.** E3 says what scaffold hopping
  lacks: the topomer frame is defined against a scaffold that is held fixed, so replacing the
  scaffold leaves no frame. The way through is to stop asking for a match over the whole molecule.
  Multiply the co-field by a soft mask, itself a sum of Gaussians placed on the atoms whose
  contribution is largest, and score a candidate by its overlap with the masked co-field alone. A
  product of two Gaussians is a Gaussian, so the masked overlap stays in closed form and rule 3
  holds. The candidate is then free everywhere else, which is what a replacement scaffold needs, and
  the complement of the mask gives the second score: the field the candidate must not place where
  the model rewards none. Needed: a mask beside the co-field in `cmfqsar/cofields.py`, the masked
  overlap in `cmfqsar/kernels.py`, and a screen over a conformer library. Gate: on a published pair
  of scaffolds with the same activity, with the model fitted on the first scaffold only, the second
  scaffold ranks above the posed decoys of `examples/decoys/` by the masked score, and ranks higher
  by the masked score than by the unmasked field similarity.
* **MC6. Matched pairs as transformation fields.** A matched molecular pair in a fixed pose has a
  field difference, and that difference is a local object placed in space rather than a number
  attached to an atom type. Collecting one per pair over many series asks whether a transformation
  carries its effect with it. D5 is both the warning and the reference: increments fitted on one
  series did not transfer, and the vectors fitted on two sets agreed to +0.89 at best. Needed: the
  pair finder of MC1, the difference field of `cmfqsar/fielddiff.py` averaged over the pairs of one
  transformation, and the predicted change for that transformation in a series held out. Gate: the
  held-out prediction beats the mean change of that series by more than the interval of rule 4. The
  FEP sets (`examples/schindler/`, `examples/ross/`) are the only data with a crystal pose for every
  ligand and are where this is measured; their narrow activity ranges are the known risk, and a
  negative result stands beside D5 rather than replacing it.
  **Done 27 October 2026, and the gate is not passed: the negative result stands beside D5, as
  the item allowed.** `cmfqsar/transformfield.py`. A pair becomes one field: the two groups,
  each masked out of its molecule by the rule of MC1, placed in the frame of their attachment
  point (origin at the core atom carrying the group, x along the bond to it, the plane fixed by
  one more neighbour chosen by atom type so that both members pick the same one), the second
  minus the first. The overlap kernel is bilinear, so reversing a pair reverses its kernel row,
  and a ridge with no intercept predicts exactly the opposite change for the reverse of a pair;
  a test fixes that. `average_field` is the transformation field the item asks for, the mean of
  the difference fields of all the pairs of one transformation. 16 tests.
  **Measured** by `benchmarks/run_transformfield.py` on 3814 matched pairs in 17 congeneric
  series with a crystal pose for every ligand (the ten Merck KGaA sets, the duplicate
  `eg5_alternativeloop` left out, and the eight Schrodinger JACS sets), holding out a whole
  series at a time with the ridge chosen inside the training series the same way, and q2 taken
  against each series' own mean change: no change at all -0.025; the transformation as a number,
  which is D5's increment averaged over the series that have it, -0.073; the transformation as a
  field -0.025 [-0.118, +0.035]; the field against the number +0.048 [-0.017, +0.108].
  **Two controls say the construction is not what failed.** Described without a frame, as the
  rotation-invariant autocorrelation of the difference field, the same pairs give the same
  -0.025, so the frame convention is not the problem. Trained **inside** each series, with whole
  molecules held out so that two pairs sharing a molecule cannot leak into each other, the same
  difference fields reach q2 +0.465. The descriptor describes a change; it does not transfer.
  The ridge chosen in every outer fold sits at the strong end of a grid that runs to 1e4, which
  is the model saying it has nothing.
  **The data answer the item before any model is fitted.** Of the twelve transformations that
  occur in the most series, eleven raise the activity in one series and lower it in another:
  methyl to chloro is +0.15 on average over 10 series and runs from -0.93 to +1.07; methyl to
  ethyl -0.17 and runs from -1.15 to +1.01. There is no single effect for a field or a number to
  carry. What a group does depends on the pocket it does it in, which is D5's finding in another
  form, and the place for the difference field is inside one project rather than across them.
  **The narrow activity ranges are the known risk and they are visible**: the spread of the
  change inside a series is often well under one log unit.
* **MC7. Growing and linking a fragment.** `cmfqsar/complementarity.py` scores a ligand against a
  pocket by field overlap. A bound fragment leaves a gap, which is the field the pocket would reward
  minus the field the fragment already places. Needed: that difference as a field object, an
  enumeration of linkers or of growth vectors, and the overlap of each candidate with the gap field,
  ranked as E2 ranks. Gate: on a published fragment-linking case with structures for the two
  fragments and for the linked compound, the true linker is among the first ten of the enumeration.
  This is the item most likely to return nothing, because a one-body field score has never been
  shown to rank linkers, and under rule 5 it becomes a display tool if the gate fails.
  **Done 27 October 2026; the gate is not met and rule 5 applies, which is what the item
  expected.** `cmfqsar/fraggrow.py`. The pocket's offer is its own field with the sign
  convention of `complementarity.COMPLEMENTARY_PAIRS`; the **gap** is that offer in the space
  free of the protein and of the fragment, which is how "minus what the fragment already places"
  is taken. The other reading, subtracting the fragment's field from the pocket's offer, was
  tried first and is wrong: it compares two different quantities and erases the contact region a
  new group would use. Candidates are built from SMILES with ETKDG and a force field, never from
  a crystal structure, placed on the growth vector by rule -- origin at the growing atom, x
  along the direction its own bonds leave empty, the bond length the sum of two covalent radii,
  the turn about the new bond scanned -- and a placement closer than 2.5 A to the protein is
  refused. 17 tests.
  **Measured** by `benchmarks/run_fraggrow.py` on 25 growing cases from the fragment sets of the
  Schrodinger collection (`examples/ross/fragments/`, with the pockets of
  `examples/pockets/ross/fragments/`), a library of 201 groups taken from the ligands of the
  same collection, 12 turns per candidate. The median rank of the group that was really added,
  out of 201: at random 82; refusing the placements that clash and shuffling the rest 15 (among
  the first ten in 40% of the cases); those candidates sorted by size 28 (12%); sorted by the
  field score 25 (28%).
  **The field score is worse than not scoring at all.** All of the ranking comes from the
  geometry -- whether the group fits -- and ordering what fits by the score pushes the right
  answer down, because the score rewards whatever fills most of the gap and so prefers the
  largest group that still fits, while these additions are a fluorine, a methyl, a chlorine.
  Under rule 5 the gap becomes a display tool: a picture of where there is room beside a bound
  fragment and what the pocket would reward there.
  **What the measurement cannot settle**: the library always contains the answer, which is the
  easy part of a real enumeration; the cases are few, because a public set with a pocket, a
  fragment and the compound it grew into is rare; and every candidate gets one embedded
  conformation and a turn, where a real growing tool would minimise the group in the site.
* **MC8. Soft spots: a target that belongs to one atom.** The site of metabolism and the pKa of a
  group are properties of an atom, and LP1 and LP2 are the items that make a per-atom target
  possible. The use in medicinal chemistry is blocking a metabolic position, which is a question
  about one atom and its surroundings. Needed: nothing beyond LP1, LP2 and a data set; the item
  exists so that the measurement is made on drug-like molecules and not only on the small molecules
  those items will use. Gate: the correct site among the first two predicted on a public
  site-of-metabolism set, against a 2D baseline.
  **Done 27 October 2026; the gate is passed, but by the field next to the count and not by the
  field alone.** The data set is `examples/local/metabolism.sdf`, prepared by
  `examples/make_local_examples.py --som`: 636 molecules, 1301 sites among 14 488 heavy atoms,
  from the benchmark of Zaretzki, Matlock and Swamidass (J. Chem. Inf. Model. 53 (2013) 3373) in
  the preprocessed form the FAME.AL project publishes under the MIT licence, the sites merged over
  the cytochrome P450 isoforms. `localenv.topological_counts` is the two-dimensional baseline the
  gate names -- the centre's atom type and the types one to four bonds away, which is what an
  atom's topological fingerprint holds. `localenv.top_hits` and `localenv.top_k_accuracy` are the
  measure: in what share of the molecules is a true site among the one, or the two, atoms the
  model ranks highest, ranked inside each molecule. `cmf local` gained `--rank`, which reports it,
  and `--primal`, because 14 488 atoms give a 1.7 GB kernel and the primal route of LP5 is the
  same model in 10 MB. 13 tests.
  **Measured** by `benchmarks/run_metabolism.py` on 10 folds that hold out whole molecules, the
  ridge searched the same way for every row, as the share of molecules whose site is among the
  first two: count over bonds (the 2D baseline) 70.9%; count over distance 73.9%, +3.0% [+0.3%,
  +5.5%]; the local field at its best width (rho 0.25) 71.1%, +0.2% [-3.3%, +3.8%], which is no
  better than the baseline; the count and the field together **76.9%**, +3.0% [+0.5%, +5.7%] over
  the count alone and +6.0% [+3.3%, +8.8%] over the baseline, which is the gate.
  **What the rows say.** Geometry gains, and it gains in its simplest form: the same atom types
  placed in space rather than along the bond graph are worth three points. The field adds three
  more on top of the count, so the two descriptions hold different things. The field alone is weak
  here for a reason that belongs to the descriptor: the autocorrelation of an environment never
  names its centre, and how easily an atom is oxidised depends first of all on what the atom is.
  Giving the field only the centre's own atom type already reaches 74.8%. On the shift set of LP2
  that costs little because every sample is a carbon; here the candidates are carbons, nitrogens,
  oxygens and sulfurs at once. The width is interior: the field rows rise to rho 0.25 and fall on
  both sides. The mixture that passes the gate lives in the benchmark, not in `cmf local`, which
  fits one descriptor at a time; and the merged isoforms make this a question about the atom
  rather than about the fit of a molecule into one binding site, which is the half of the
  statement an isoform-resolved target would test.
* **MC9. Scoring a conformational restriction.** Rigidifying a molecule is one of the few moves in
  medicinal chemistry whose effect is purely conformational, and the fourth-dimension verdict says
  that ensembles do not improve a prediction. They may still answer a different question. The
  variance field of D10 says where a molecule places field only part of the time; weighted by the
  squared co-field it says how much of the activity-relevant field is intermittent. A restriction is
  then scored by two numbers: how much of the weighted mean field it keeps, and how much of the
  weighted variance it removes. Needed: D10, the weighting, and the score as an option of
  `cmf explain` or of a subcommand of its own. Gate: on pairs of a flexible compound and its
  restricted analogue with measured potencies, the weighted variance removed correlates with the
  potency gain. Both members of a pair must come from the same conformer generator with the same
  settings, since the comparison is between two ensembles and not between two molecules.
  **Done 27 October 2026; the gate is not passed, and the half of the score that works is the
  mean field rather than the variance.** `cmfqsar/restriction.py` and the subcommand
  `cmf restriction`. The square of the co-field is a weight that is positive everywhere and
  large only where the model reads the field, so a molecule gets two amounts on a grid: the
  **relevant** field it places (the weight times the squared mean field) and the
  **intermittent** part of it (the same weight on the variance field of D10). A pair is scored
  by `kept` = relevant(B)/relevant(A) and `removed` = 1 - intermittent(B)/intermittent(A).
  20 tests.
  **The pairs are found by a rule, not by eye**, so the measurement cannot be steered: one ring
  more, at least one rotatable bond fewer, at most two heavy atoms more, an atom composition
  differing by at most two atoms, and a common substructure covering at least 80 per cent of the
  smaller member. A fingerprint is deliberately not used for the likeness. Closing a ring changes
  the environment of every atom near the new bond, so butylphenol and 2-methylchroman -- the same
  eleven heavy atoms, one bond more -- are only 0.15 alike as Morgan fingerprints, and a
  threshold high enough to be useful would throw away exactly the pairs the item is about.
  **Measured** by `benchmarks/run_restriction.py` on 439 pairs from 4 aligned Sutherland sets
  (THR 33, COX2 145, DHFR 190, ACE 71), 3 conformations per molecule from the same generator for
  both members, correlated with the potency gain: the weighted variance removed +0.043 [-0.030,
  +0.111], which contains zero; the same with the pair left out of the co-field +0.020 [-0.061,
  +0.093], so there was nothing for the weight to learn; the unweighted variance removed -0.034
  [-0.140, +0.062]. The reference of rule 1 is the cheapest thing a chemist could use instead,
  the number of rotatable bonds the closure removed, and it reaches +0.236 [+0.151, +0.317]
  without geometry, conformations or a model.
  **What does track the gain is the mean field**: how much of the activity-relevant field the
  restricted analogue still places, +0.220 [+0.120, +0.315]. A ring closure helps when the
  molecule still puts its field where the model reads it and hurts when the ring holds it
  somewhere else, which is the expected statement and does not need the variance to make it.
  This is the third measured place where the conformational ensemble carries less than it
  promises, after D9 (averaging does not predict better) and D10 (the variance adds nothing as a
  block of the kernel); the variance field stays the map D10 made it.
  **Two limits of this measurement**, both stated in the benchmark: the ensembles are three
  conformations from a distance-geometry generator superimposed on the aligned pose, which is a
  narrow view of what a molecule does in water; and a ring closure is never purely conformational
  in practice, since it changes the shape and often an atom or two as well. A public set of pairs
  differing in nothing but one bond, with measured potencies, was looked for and not found.

**Large (a month or more each)**

* **MC10. Peptidomimetics.** The direct route is closed and the repository says so: the 130
  decapeptides of `examples/mittal/` are modelled by no configuration, the reason is a superposition
  error of 3.2 Å over a flexible chain, and C2, C3 and C4 are three measured failures to fix an
  alignment with the means the program has. What is left is a route that never aligns whole
  molecules. The template is the bioactive conformation, taken from a structure or from a
  constrained analogue and not from a model fitted on peptides. The scoring is MC5: match a
  candidate scaffold to the masked co-field of the side-chain positions that carry the weight, and
  ignore the backbone the mimetic is meant to replace. The restriction is scored by MC9, which is
  how a cyclisation, an N-methylation or an α-methylation is compared with another. Needed: MC5,
  MC9, D10 and a template. Gate, and it is hard: on a published peptide with several mimetics, the
  mimetic scaffold that was active ranks above the ones that were not, and the predicted order of
  two restrictions of the same peptide agrees with the measured potencies. Until that gate is passed
  this is a hypothesis, and the plan expects peptides to stay hard.
* **MC11. A design loop with more than one objective.** E2 and E3 close the loop on potency alone.
  Lead optimisation has at least three objectives, and once E1 and MC4 are done they live on one
  kernel: potency, selectivity against the panel, and whatever property of the same series is
  measured. Needed: a scalarised or Pareto acquisition over the tasks, the batch rule of MC2, and a
  report that says for each suggestion which objective it gains and which it loses. Gate:
  retrospectively on a series with two measured endpoints, the Pareto front the loop reports
  contains the compounds the authors went on to make. The risk is plain: three endpoints measured on
  one congeneric series are rare in public data, so this item waits on MC4's data more than on its
  own code.
  **Done 27 October 2026; the machinery is there and the gate is not met.**
  `cmfqsar/multiobjective.py`: `pareto_layers` and `pareto_front` over any number of objectives
  with a direction each; an acquisition that adds the expected improvement of every objective
  against the best already made, which is scale-free because each term is in its own objective's
  units of improvement; `batch`, which is MC2's rule with more than one task -- after a pick,
  **every** objective's posterior covariance is conditioned on it and the acquisition recomputed
  -- and `trade`, which says for one suggestion what it gains and what it loses against the best
  compound already made, with a change smaller than the model's own error reported as no change.
  `CMFModel.predict_with_covariance` (added for MC2) is what makes the batch possible on a real
  model. 17 tests, one of which fixes the property that surprises: conditioning removes the
  uncertainty and not the promise, so two compounds the model is sure are a log unit better are
  both still worth making.
  **The data for the retrospective test** came with MC4's series, which is in two parts. The
  loop is trained on the 78 compounds of part 1 with both endpoints and scores 425 candidates
  (`examples/antitarget/h3_candidates.sdf`, built by the same script with `--candidates`): every
  other H3 ligand in ChEMBL that fits the frame, of which 16 are the compounds the same authors
  made next, in part 2 of the paper. The loop sees no activity of any candidate.
  **Measured** by `benchmarks/run_designloop.py`: the Pareto front holds 5 of the 425 and none
  of the 16; nor do the first 50 by the acquisition. The compounds they made sit at Pareto layer
  23 of 47 against 19 for the rest and at rank 343 of 425 against 208, so the loop puts them in
  the worse half; the best of them reaches layer 2 and rank 79.
  **Why, in three parts.** The model does see that they are good -- measured affinity 9.15
  against 7.98 for the pool, predicted 8.68 against 8.59, the order right and the size not,
  which is what q2 +0.28 on a narrow series buys. What sinks them is the second objective: their
  hERG is predicted at 6.29 against 5.89, and an acquisition that adds improvements prefers a
  candidate extreme in one objective to one that is good in both. And no hERG was ever measured
  for them, so that prediction cannot be checked.
  **What the test cannot settle**: the compounds the authors made were chosen with knowledge the
  loop does not have, so the ones they did not make are not known failures; and only one of the
  two objectives can be checked against what happened. A series with both endpoints measured on
  the follow-up compounds would decide it, and the plan's own warning about three endpoints on
  one congeneric series stands.

**What this does not cover.** Equivalence of frontier orbitals, which is the literal form of the
isolobal analogy, needs a frontier-orbital field, and that road has been measured twice: a Fukui
function is not a sum of atom Gaussians (relative fit error 0.8 to 0.9) and the model did not care,
and giving every atom a direction fixed the fit and not the model (−0.004 q²). Quantum-chemical
fields earned their place on the Hammett constants only, where resonance needs whole-molecule
charges, so a field-based equivalence of groups rests on the five physico-chemical fields, which is
MC1. Properties measured across a diverse set, such as solubility, permeability and clearance, have
no superposition and therefore no field model; inside one congeneric series they are extra tasks on
the shared kernel, which is where lead optimisation works anyway, and `cmf diagnose` is expected to
report that a one-dimensional descriptor does as well on the diverse case. Covalent inhibitors and
bifunctional degraders break the sample of one molecule in one pose: a warhead's effect is a
reaction rate, and a degrader binds two proteins through a linker whose conformation is the whole
problem. Which analogue can be made, and in how many steps, is not a property of a field; MC1, MC2
and MC11 rank candidates and a chemist removes the ones nobody will synthesise.

**Order.** MC1 and MC2 first: they need no new science, they use the two loops E2 and E3 have
already measured, and MC1 is the item that answers the question most often asked of a 3D-QSAR model.
MC3 and MC4 are data tasks and run beside them, since every later item that mentions selectivity
waits on MC4's set and C7 cannot be closed without MC3's. MC5 next, because masked matching is the
construction E3 names as missing and MC10 cannot start without it. MC9 after D10, which it needs,
and before MC10. MC6 and MC7 at any time after MC1, and both are allowed to fail. MC8 with LP1 and
LP2. MC11 last, after E1, MC2 and MC4.

## Sequence

| When | What |
|---|---|
| Months 0 to 3 | A (platform), A6 first; B1 and B2 as they are needed by the benchmarks |
| Months 2 to 6 | ~~B1~~, ~~B2~~, ~~B3~~, ~~B4 (evidence)~~ and ~~B5~~ **done**, ~~D6~~ **done (negative result)**, ~~C1 (the protocol) and C2 (flexible alignment)~~ **both done**; ~~F2 (co-field uncertainty)~~ **done** |
| Months 3 to 9 | D1 to D5; ~~C3~~ **done (negative result)**, ~~C4~~ **done (negative result)**, ~~C5~~ **done (negative result)**, ~~C6~~ **done**, ~~C7~~ **built, gates passed**; ~~F1 (case studies)~~ **done, except the chemist** |
| Months 4 to 10 | ~~E1~~ **done**, ~~E2~~ **done**, ~~E3~~ **done**, ~~E4~~ **done**, ~~E5~~ **done**, ~~E6~~ **done (no gain in accuracy)**, ~~E7~~ **done (ridge stays the default)**, ~~E8~~ **done (no gain in accuracy; kept for its co-field)**, ~~E9~~ **done (ties the GP classifier; its evidence calibrates better)**, ~~E10~~ **done (no gain; kept and labelled under rule 5)**; part E complete; ~~G1~~ and ~~G2~~ **done** |
| Months 6 to 12 | ~~F3~~ **done**, ~~F4~~ **done**, ~~F5~~ **done**, ~~F6~~ **done (the fingerprint holds the activity)**; D7 and D8 (trajectories), D9 and D10 only if D8 says so; MC1 and MC2 (the bioisostere table, the batch rule) and the data tasks MC3 and MC4 beside them; H begins with separations |
| Year 2 | H; M4 to M6 (periodic overlap kernel, site-resolved models, reciprocal-space channels) first, M1 to M3 as they are needed, M7 to M13 after; ~~EC2~~ **built, gates not passed, kept as a display tool**; EC1 (quantum-stage controls) next, then EC4 (metal complexes) before any other EC item, EC6 with M1, EC8 after M5; BT1 and BT3 (crystal charges for fractional compositions, the battery data loader) with M4 to give the periodic kernel its voltage benchmark, BT4 (solvation complexes) after EC7, BT6 with M4, BT7 after M5; LP1 (per-atom targets) with M5 and LP2 (atom-centred environments, pKa and shifts) right after, LP3 to LP5 as their gates allow; NT1 to NT3 (the structure data model, open boundaries, size descriptors) with M1, NT8 (the viewer) with M13, NT5 (curved surfaces) and NT7 (sites on particles) after M5, NT12 alongside each; SC1 and SC4 (the two-endpoint command line, the host–guest sets) at once since they need no new science, SC5 (the cavity frame) with EC7, SC6 (per-atom contributions of the invariant kernel) with NT9 and M5, SC7 after SC6, SC8 is M7; MC5 (masked matching) once E3's missing construction is wanted and MC9 after D10, MC6 and MC7 at any time after MC1, MC8 with LP1 and LP2, MC10 after MC5 and MC9, MC11 after E1, MC2 and MC4; the review that ties the series together |

The order inside a part follows the dependencies: nothing in C or E is judged without the protocol
of C1 and the intervals of rule 4; D6 and E6 wait for B4; E3 waits for E2 and the topomer rule;
F2 needs the GP view that B4 makes the default; E8 and E9 are judged on the classification sets
of E4 and E9 reuses the Newton step of the Laplace classifier, E10 waits for both because its
screening gate needs a classifier to compare with; D9 waits for D4 and, with D10, for the
verdict of D8; in part I the substituent items wait for the topomer rule E3 measured, MC9 for
the variance field of D10, MC10 for MC5 and MC9 together, and MC11 for the panel data of MC4. F6 needs the cosine of F3 to compare the two co-fields of a component, and the features of B6 to run on a set large enough for the question to matter.

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
* **Data for the medicinal-chemistry items.** Part I is limited by data and not by code. Four of
  its eleven items need a set the repository does not hold: enantiomeric pairs with two
  potencies (MC3), one series with a potency and an off-target activity (MC4), a
  site-of-metabolism set (MC8) and pairs of a flexible compound with its restricted analogue
  (MC9). The control is the order: MC1 and MC2 are measured on the Sutherland sets that are
  already in place, so the part returns a result even if no new set is found.
* **Metals and small series.** EC4 asks RDKit to hold coordination complexes it was not built
  for, and the catalyst series behind it are dozens of compounds with turnover frequencies that
  differ between laboratories by the factor the model is meant to explain. The control is the
  order: the loader and the parameters first, measured on the geometry alone (does the series
  load, align and give a steric kernel that separates the known chemotypes), then one tabulated
  series with GP intervals, and the vector kernel and anchored points only if that series shows
  signal. A q² on twenty catalysts without its interval is not reported.
* **Trajectories cost more than they may return.** D8 needs simulations that take longer than
  every benchmark in the repository together, and the fourth-dimension verdict says that ensembles
  have not helped so far. The control is the order: the reader first, the measurement on one set
  second, and the two new fields only after the measurement; an unconverged trajectory gives a mean
  field with sampling error that would look like chemistry, so D8 reports the block-averaged error of
  the mean field next to the q² interval.
