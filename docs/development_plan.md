# Development and extension plan for CMF

*Drafted 22 October 2026. The companion of `docs/publication_plan.md`: that document says what gets
written and where it is sent; this one says what gets built, in what order, and what each piece has
to prove before it is allowed to stay. File references point at the code the work starts from.*

## Where the program stands

The repository holds, as of 30 October 2026 (the lists below were written on 22 October and are
kept current; the counts are the ones the repository gives today):

* A library of 92 modules and about 49,500 lines (`cmfqsar/`), a PySide6 application, a
  command-line tool with thirty-one subcommands (`build`, `build-crystals`, `diagnose`,
  `align`, `predict`, `screen`, `cofield`, `report`, `benchmark`, `unalign`, `map`, `compare`,
  `cluster`, `explain`, `mil`, `hetero`, `pharmacophore`, `difference`, `topomer`,
  `complement`, `suggest`, `design`, `transfer`, `mixing`, `atoms`, `local`, `hops`, `restriction`, `isosteres`, `lowrank`, `kcca`), 1703 tests, a typeset manual and a
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
  **Corrected 30 October 2026, and the correction is larger than this item.** The protocol
  destroys a pose by placing the molecule at random within 2 A of the site **and giving it a
  uniformly random orientation**. Three of the four searches it judges restarted from a turn of at
  most about 63 degrees around the pose they had arrived at, and restarted twice. A bounded turn
  cannot undo an orientation drawn uniformly, however many restarts are spent on it, so the
  protocol was measuring the starting points of those searches and not the criteria they optimise.
  `varalign.py`, `pocketalign.py` and `flexalign.py` now take `any_orientation`, which
  `cmfqsar/poses.py` sets whenever the protocol has destroyed the orientation; the protocol's own
  defaults are 20 restarts from any orientation, while the docking keeps the 30 degrees a series
  already placed in the site needs. A user reaches it as
  `align --any-orientation`, `predict --align-any-orientation` and a tick box in *Align by
  prediction confidence...*, which also removes the displacement limit and raises the restarts,
  since a setting nobody can switch on is of no use to anyone.
  **Both halves are needed, and a control measures them apart**: the number of restarts and the
  turn they are allowed, varied separately on two sets (`mcl1_r2.json`, `mcl1_r20_rot30.json`,
  `thrombin_r2.json`, `thrombin_r20_rot30.json`, beside the two files at the defaults). Each
  change on its own brings the median down -- on thrombin from 5.00 A to 0.11 A with the
  orientation alone and to 0.10 A with the restarts alone -- and neither removes the last
  misplaced molecule: one ligand of the seven stays 7.62 A out in all three of those corners, and
  that one ligand holds R2pred at about zero (-2.804, -0.014, -0.011). With 20 restarts from any
  orientation its worst molecule is 0.33 A and the same model predicts at +0.440. On mcl1, which
  never failed outright, 20 restarts are enough whichever turn they are allowed (0.17 A, +0.387).
  So the median is the wrong number to read here, which is why the protocol reports the error of
  the prediction beside it. The bounded
  branch draws the same random numbers as before, so every number measured with it reproduces
  exactly; those files keep their names, a run at other settings writes its own file
  (`*_r20.json`) and its own table section, and every record stores the settings it was measured
  with.
  **What it changes.** The confidence criterion alone (`variance`) on the 16 FEP sets, mean over
  the sets: median RMSD 1.35 A -> **0.11 A**, within 1 A 64 % -> **100 %**, recovery 74 % ->
  **98 %**, external R2pred -0.219 -> **+0.312** against the crystal poses' +0.334. It is now
  level with substructure alignment (0.25 A, +0.329) and needs no substructure to align on. No set
  is worse, and the four sets it had failed on are the four it gains most: syk 6.26 -> 0.13,
  thrombin 5.00 -> 0.10, jnk1 3.49 -> 0.10, pde2 3.14 -> 0.12 A.
  **What it does not change.** `mcs+variance`, which starts from a substructure frame, was already
  at 0.11 A and stays there; and the field-similarity refinement of `cmfqsar/align.py` always drew
  a full random orientation, which is why that search never showed this failure.
  23 tests for the correction, over the four searches, the file naming and the two user routes to
  the option.
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
  **Re-measured 30 October 2026 with the widened search of C1**, from the generated conformation
  alone: `flex` 2.47 A -> **1.51 A** median, within 1 A 21 % -> 40 %, R2pred -0.160 -> +0.027,
  better on fifteen of sixteen sets and worse on none (pde2 5.94 -> 2.27, eg5 3.44 -> 1.51,
  pde10 3.17 -> 1.27, bace 1.39 -> 0.53 A). After a substructure frame it changes nothing for the
  better: `mcs+flex` 1.41 A -> 1.48 A, the wider search leaving a good start as often as it
  improves one. The statement above survives the correction: the rigid confidence search on
  regenerated conformations is 2.27 A at 2 bounded restarts and 2.32 A at 20 unbounded ones, so
  what defeats it there is the conformation and not the pose search, which is what this item was
  about. The five-conformer runs, the 0.86 A among them, were measured with the narrow search and
  have not been repeated.
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
  **Corrected 30 October 2026: most of the failure above was the search and not the term.** The
  numbers above were measured with the search C1's correction describes, 2 restarts of a turn of
  at most 63 degrees, which cannot recover an orientation the protocol drew uniformly. Repeated on
  the same four sets with 20 restarts from any orientation (`*_pocket_r20.json`): the receptor
  alone 4.22 A -> **1.12 A** median, 31 % -> **74 %** within 2 A, recovery 14 % -> **78 %**,
  R2pred -0.805 -> -0.241; the two together 3.22 A -> **0.15 A**, recovery 32 % -> **97 %**. These
  figures are means over the four sets, as `benchmarks/results/poses/table.md` prints them, while
  the RMSD figures in the paragraphs above are medians over the sets and their percentages are
  means; that is why the receptor alone reads 3.93 A there and 4.22 A here, from one measurement.
  **The verdict of the item is unchanged and the evidence for it is now much smaller.** At a search
  that can find the pose, the receptor term still adds nothing to the confidence criterion: 0.15 A
  against 0.11 A for confidence alone on these four sets, worse on all four of them by 0.03 to
  0.05 A, with a worse R2pred (-0.110 against +0.286) and about ten times the cost. So "adding the
  receptor makes the confidence criterion worse" stands, but the sentence **"what fails is the
  term and not the search" is wrong**: the term costs 0.04 A and the search cost 3.07 A. The
  weight-zero control remains valid as what it was, a comparison of two optimisers of one
  objective at one search setting.
  **The two criteria answer the two settings differently, and a control on mcl1 separates them**
  (`mcl1_pocket.json`, `mcl1_pocket_r20_rot30.json`, `mcl1_pocket_r20.json`, 8 held-out ligands).
  The receptor alone gains most from the number of restarts: 3.00 A at 2 restarts of a bounded
  turn, 1.00 A at 20 restarts of the same turn, 0.93 A at 20 restarts from any orientation. One
  ligand stays about 8 A out in all three, which is where the receptor term's own optimum is. The
  two criteria together answer the other way round: 20 bounded restarts change nothing at all
  (0.31 A either way, the same ligand 7.89 A out), and only a restart from a completely different
  orientation brings that ligand back (0.37 A, median 0.22 A). So the receptor term holds a pose in
  an orientation that a bounded turn cannot leave, while the confidence criterion alone on this set
  is recovered by 20 bounded restarts. This is a second reason to keep the free orientation on
  wherever the receptor term is used.
  **The weight sweep repeated at the corrected search, 30 October 2026**
  (`{p38,tnks2}_pocket_w{0,025,4}_r20.json`, the joint aligner only, since the weight changes no
  other method). Median distance, narrow search then corrected search: p38 0.29 -> 0.09 A at weight
  0, 3.52 -> 0.10 at 0.25, 3.73 -> 0.12 at 1, 3.63 -> 0.24 at 4; tnks2 0.07 -> 0.07, 0.11 -> 0.09,
  1.26 -> 0.11, 1.32 -> 0.27. So the first sweep measured the search and not the weight, and the
  answer to the question the item asked is that the weight no longer matters for the median. It
  still matters for one molecule: on tnks2 the worst of the eight distances is 0.33 A at weight zero
  and 2.59 A at weight 0.25, and that single ligand takes R2pred from +0.212 to -0.422 (-0.517 at
  weight 4), against +0.291 for the crystal poses. On p38 no weight loses a molecule and R2pred
  stays at the crystal value. Weight zero also serves as the control of the two optimisers at the
  corrected search: 0.09 A on p38 and 0.07 A on tnks2 are the numbers the confidence criterion alone
  gives, so the joint optimiser is not what fails.
  **Not done:** the flexible version (freeing the torsions made the confidence criterion work in C2 and
  might do the same here), an analytic gradient for the pocket term, the sets without a receptor,
  which is most of them, and the two kinds of section of the pose table that are
  still measured at the narrow search alone: the five-conformer flexible runs and the `_fg`
  field-point runs.
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
  **The other three-point term, 31 October 2026.** The angle at the centre of a local
  environment, which is the part of a triangle the two-point kernel does not resolve and
  which is not handedness, is built as `cmfqsar/triangles.py` and measured under LP5 below:
  it matches SOAP on the carbon shifts with the fields as channels.
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
  -- a random rigid placement within 2 A of the site and a uniformly random orientation -- and
  then the pocket docking of C3 puts it
  back where the receptor's own fields prefer it, with no model, no activity of its own and no
  crystal frame of any kind, which is the protocol C1 defines. Each ligand was then simulated
  from the pose that was found, 2 ns on the same receptor frame as everywhere above, and read in
  the same three ways.
  **What the posing produced is two groups rather than a spread.** The 25 poses sit a median
  1.12 A from the crystal poses, but the median hides the shape: 14 ligands are within 1.3 A,
  and on the other 11 the docking missed the site, 4.0 to 8.3 A away. So this test is harder
  than its median suggests, and a reading of it is also a reading of how often that docking
  fails.
  **That docking has since been fixed (C1, 30 October 2026).** Missing the site on 11 of 25 was
  the bounded restart and not the scoring function: with 20 restarts from any orientation all 25
  mcl1 ligands come back within 2 A of their crystal poses, on two seeds. The poses used here are
  the old ones, so every number in this paragraph and the next three is about them. Re-posing the
  set and simulating from the better poses is 3.4 hours of one consumer GPU and has not been
  done.
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
  second set, and the same measurement at the poses the corrected docking finds, since the result
  above is as much about the old docking as about the ensemble; why the own frame costs q2
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

**Small (one to three days, but it moves published tables)**

* **D11. A charge model that no element can void (5 November 2026).** The Gasteiger scheme has no
  parameters for some elements, and one such atom makes RDKit return "not a number" for *every*
  atom of the molecule, so the electrostatic field of that molecule is identically zero: not an
  error, not a warning, and not a field. Item EC4 found this for the d-block metals and repaired
  it there, by computing the charges with the metal atoms removed and giving each metal its formal
  charge. Item EC3 then found the same fault outside the d block, on the selenomethyl substituent
  of the Hammett series.
  *The scope is measured, and it is smaller than the element list suggests.* The elements the
  scheme cannot do at all are **selenium, tellurium, arsenic, tin, lithium and zinc** (each tested
  on a small molecule: every atom comes back "not a number"). But the fault needs the element to
  be **covalently bonded inside a fragment**: the scheme runs per connected fragment, so a
  disconnected ion gets its formal charge and its neighbours are computed normally. That halves
  the exposure and it was measured, not assumed:
  * **voided**, 5 molecules in all: two selenoethers in `examples/corrosion/0_c1ccccc1.csv` and
    the same two in its `corrosion_hcl` twin (of 244 each); a tetraphenylarsonium cation and an
    arsenic-dithiol compound in `examples/cyclodextrin/beta_cyclodextrin.csv` and
    `beta_with_alpha.csv` (of 1162 each); and the selenomethyl substituent of
    `examples/hammett/` (of 95).
  * **not affected**, although the element is present: `examples/electrolyte/shells/`, where the
    lithium is a separate fragment. Its shells keep an electrostatic field of ordinary size, so
    items BT4 and BT5 are not touched by this. That was worth checking rather than assuming,
    because both use `charge_model="gasteiger"` on lithium-containing files.
  So the repair reaches the corrosion, cyclodextrin and Hammett families, whose tables are
  published, but through five molecules; the likely measured effect is in the third decimal.
  *What to build.* Generalise EC4's repair to any atom the scheme cannot do: compute the charges
  with those atoms removed and give each of them its formal charge, reporting the source as
  something other than `gasteiger` so that a model fitted before and after cannot be confused.
  The electrolyte shells may need nothing, since they read charges from their files; that has to
  be checked rather than assumed.
  *Gate, and it is a reporting gate rather than an accuracy one.* The repair is correct by
  construction -- a field that was identically zero becomes a field -- so what has to be measured
  is **what it moves**. A table that moves is re-measured and said to have moved, never quietly
  rewritten; rule 6 is the reason this cannot be folded into another item's commit.
  **Done 5 November 2026, and the gate's answer is not the one this entry predicted.** The repair
  is in `cmfqsar/fields.py`: the plain scheme is asked first, so every molecule it can do is
  answered exactly as before, and only a molecule that comes back "not a number" is repaired.
  `GASTEIGER_ELEMENTS` is the measured list of fourteen, `_gasteiger_with_unknown_atoms` removes
  the atoms outside it and gives each its formal charge, and an atom left over -- an uncovered
  valence state -- keeps its formal charge too. `tests/test_charges.py` has 22 tests, including one
  that demonstrates the fault in the scheme itself, so the repair cannot outlive the thing it
  repairs.
  *What moved, benchmark by benchmark.* Five of six affected measurements moved by almost nothing,
  as five molecules should: Hammett sigma_m with the electrostatic field alone +0.66 to +0.67 q2
  (and the chosen attenuation 0.5 to 0.3, the fitted r2 0.82 to 0.80); the corrosion root series,
  1 of 23, aligned +0.386 to +0.388 and the invariant kernel +0.478 to +0.479, with the series
  means unchanged; the same series in `corrosion_hcl`, +0.413 to +0.415; the beta-cyclodextrin
  host, 1 of 7, invariant +0.447 to +0.451; and the slab benchmark's corrosion stage not at all.
  No verdict in any of them changed.
  *And item BT5 moved a great deal, which is why this item earned its place.* 717 of its 989
  solvation shells carried a **net charge of +1** in the electrostatic field, because the
  phosphorus of hexafluorophosphate is a valence state the scheme cannot do and was being set to
  zero while the lithium kept its +1. The corrected cosine kernel correlates only **+0.507** with
  the old one, so the electrostatic description of those shells was substantially wrong rather
  than slightly. Three of BT5's 80 paired comparisons change their verdict, all against the
  shells, and one of them is its headline: the gain of sampled shells over one optimised complex
  on the activation energy goes from +0.480 kJ/mol [0.096, 0.960] to +0.323 [-0.082, +0.786] on
  random folds, which no longer excludes zero. The grouped-fold reading survives (+0.546
  [+0.137, +1.038]) and both fold-chosen-weight variants did not move at all, which says the
  weight search had been absorbing part of the fault. BT5's co-field reading improved: its
  electrostatic peak was +3.62 at 0.69 angstrom from the ion, which is on the cation itself, and
  is now -3.51 at 3.03 angstrom from the ion and 0.81 from a donor atom. BT5's entry, its manual
  section, the README and the publication plan were revised with these numbers and say that D11
  revised them.
  *Two things found in passing and left for their own items.* The repair's approximation: removing
  a bonded atom leaves its neighbours an open valence, so the remaining fragment is charged as a
  radical and the total comes out within about 0.25 of an electron of the molecule's formal charge
  rather than exactly on it; capping the open valence with hydrogen would fix that and was not
  done. And the Hammett benchmark's four documented compute runs overwrite each other's output
  files -- two of them write `summary.csv`, and one variant's summary was missing from the
  committed results entirely -- which is a fault in that benchmark's own bookkeeping and not in
  the charges. **Item G6 took the second one on 5 November 2026** and found it to be larger than
  this paragraph says: the registry rebuilt three of the six stored results. It also recomputed the
  two ensemble variants that this item had left with the old charges.

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
  **Pooled across series, 1 November 2026.** The one reading of "hierarchical kernel model"
  that could help prediction is a second level over the series: a series of thirty molecules
  cannot set five weights, but the collection it belongs to can say what a typical series needs
  and how much series differ. `pool_priors` in `cmfqsar/mixing.py` learns `u_s ~ N(mu, tau^2 I)`
  over the centred log-shares by empirical Bayes from the posterior draws of many series (a
  Gaussian summary of each series' likelihood, which is the random-effects model of
  meta-analysis; an importance-reweighting estimate is kept as the check), and
  `evidence_mixing(prior_mean=mu, prior_sd=tau)` gives a series its weights under that prior.
  **Measured on the 38 Mittal sets** (`benchmarks/run_pooled_mixing.py`), each set's prior
  learned from the other 37, ten 70/30 splits per set. The collection does have an opinion:
  a typical series gives the steric field 0.49 of the kernel, the hydrophobic 0.25, the
  electrostatic 0.11, the acceptor 0.10 and the donor 0.06, with a spread between series of
  tau = 0.76 in the log-shares (the fixed prior has 1.5), and both estimators agree on it.
  Prediction does not follow: pooled against equal weights is **-0.003** in R2pred with a 95 %
  interval of [-0.027, +0.017] over the sets, +0.009 [-0.021, +0.042] on the 15 sets below
  forty molecules where a prior should matter most, and the population weights alone are
  +0.010 [-0.014, +0.032]. What pooling does do is repair the per-series fit: the evidence
  optimum on the series alone is -0.025 [-0.055, -0.001] against equal weights on this
  collection, and the pooled weights are +0.022 [+0.002, +0.047] against it, better on 25 of
  the 38 sets. So the hierarchy removes the variance that fitting five weights on one series
  adds, and ends where equal weights already were. The reason is the one the intervals gave:
  the field kernels of a congeneric series are so correlated that the weights hardly change
  the prediction, so no prior over them, however well informed, can move it. One set is an
  outlier in the other direction: ACE has a clearly different answer (its own evidence weights
  gain +0.075), and the prior learned without it is the only one with tau = 0.32.
  **Not done:** the same treatment of the attenuations (D6 settled that by measurement), a
  GUI reading of the intervals, and a `--mkl pooled` flag (there is no reason to add one until
  a collection is found where the weights matter).

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

* **E11. A field autoencoder with an analytic decoder.** Kernel PCA on the CMF kernel is the
  linear autoencoder of the field space, and the program has it twice: as the coordinates of the
  GTM map (`cmfqsar/gtm.py`) and as the `kpca` applicability-domain score of `cmfqsar/oneclass.py`.
  What it cannot give is a pre-image. A field reconstructed by kernel PCA is a sum of training
  fields: it can be drawn on the grid, but it is not a set of atoms, and nothing can be sampled
  from it. This item adds the nonlinear autoencoder without leaving the field space. The
  *encoder* maps a molecule to a code of a few dimensions. Its input is the whitened kernel to a
  few hundred landmarks, the feature vector of `cmfqsar/lowrank.py`, so the encoder sees the
  molecule exactly as the kernel does and needs neither a grid nor an atom order. The *decoder*
  maps a code to a fixed number of pseudo-atoms, each with a position and one weight per field
  type, so its output has the same form as every other field in the program. The *loss* is the
  squared distance between the input field and the decoded field in the field Hilbert space,

      ||f - f'||^2 = k(f, f) - 2 k(f, f') + k(f', f'),

  and all three terms are sums of overlap integrals of atom-centred Gaussians. The loss is
  therefore in closed form and is differentiable in the positions and weights of the pseudo-atoms;
  `cmfqsar/gradients.py` already holds the derivative of a Gaussian overlap with respect to the
  atom positions and weights. Rule 3 holds: no grid enters the training. The pseudo-atoms are the
  pre-image that kernel PCA lacks. They are not a molecule, but they are the field a molecule
  would have to place, and that is what the design items consume: E3, MC5 and MC7 all score a
  candidate by its overlap with a target field.
  Four versions, in the order they are built:
  (i) *plain*: the reconstruction error of a new molecule is a nonlinear applicability-domain
  score, compared with the five scores of `cmfqsar/oneclass.py` under the protocol of E5;
  (ii) *denoising*: the decoder is trained to reconstruct the single-conformation field of a
  molecule from each member of its conformer ensemble (`cmfqsar/conformers.py`), or from the
  field with atoms removed at random, so that the decoded field is a consensus field with the
  pose noise averaged out. Judged by whether a model fitted on decoded fields beats the model on
  the raw fields on the same folds (rule 1). The fourth-dimension verdict says ensembles do not
  help on an aligned series, so the expected outcome is a null result, reported as such;
  (iii) *variational*: a Gaussian prior on the code, so that codes can be sampled and
  interpolated and a decoded sample is a hypothesis field. The overlap of a decoded field with a
  co-field is a differentiable function of the code, because the co-field is linear in the field
  (rule 2), so a code can be moved along the gradient of the predicted activity, decoded, and
  matched against the fragment libraries of E3 and MC7 to find a group whose field fits the
  decoded one. This is the inverse-QSAR step the plan does not have;
  (iv) *semi-supervised*: the encoder is trained on molecules that carry no activity, in the same
  frame, and the code then enters the predictive model as one more kernel view, a Gaussian
  kernel on the codes mixed with the five field kernels by the mixing of E6. The predictive
  model stays a kernel model and rule 2 holds. It is the neural counterpart of E10, whose
  result, that unlabelled molecules gain nothing on a congeneric series of a few dozen, is the
  expected outcome unless the unlabelled set is large and varied.
  **What does not change: the activity model.** The autoencoder is never the regressor. A code
  that enters a prediction enters it as a kernel, so the co-field, the Shapley values and every
  interpretation tool are untouched. That is why this item is not the convolutional network that
  "What is not planned" excludes: there the grid replaces the kernel and the network replaces the
  model; here the loss *is* the closed-form kernel and the model is the one the program has.
  **Needed:** a module `cmfqsar/fieldae.py` with the landmark encoder, the pseudo-atom decoder,
  the reconstruction, denoising and variational losses, and a `sample` and `pre_image` entry
  point; the overlap-kernel derivatives of `cmfqsar/gradients.py` exposed for a set of
  pseudo-atoms against a molecule; PyTorch as an optional dependency, in the same way as ORCA
  and xTB, with the module absent rather than broken when it is not installed; a benchmark script
  `benchmarks/run_fieldae.py`; a `cofield`-style grid export of a decoded field for the viewer.
  No graphical interface until a gate is passed (rule 5).
  **Data.** The item cannot be judged on a series of thirty molecules, where kernel PCA has as many
  components as it needs and any network is underdetermined. The training data are the sets of
  this repository that are large and share a frame: the COMPAS sets of part H (8,678 to 34,072
  molecules in a frame given by rule) and the docked pose sets of C1 where a series has more
  than a few hundred ligands. The invariant kernel of C5 has no decoder of this form, because the
  autocorrelation is not a field that pseudo-atoms can reproduce, so the item is for aligned
  tables only.
  **Gates**, one per version, each with the interval of rule 4:
  (a) on held-out COMPAS-1D molecules the relative reconstruction error
  `||f - f'|| / ||f||` of a decoder with M pseudo-atoms is below the kernel PCA reconstruction
  with the same number of latent dimensions; if it is not, there is no nonlinearity to use, and
  the item closes here;
  (b) the reconstruction error ranks the absolute residuals of a held-out fold at least as well
  as the best of the five one-class scores, by the rank correlation of E5;
  (c) on a series with a held-out active chemotype, a code sampled near the training actives,
  decoded and matched against the fragment library, puts a group of the held-out active in the
  first ten, which is the gate of MC7 used in reverse;
  (d) the semi-supervised kernel view is judged on the folds of E10 and gains nothing until its
  interval excludes zero.
  A version that fails its gate is reported and the module is kept, if at all, for the pre-image
  of (iii) alone, labelled under rule 5 as a display tool.
  **Risks.** The number of pseudo-atoms is fixed and their set is unordered; the overlap loss is
  invariant to their order, which is right, but it lets pseudo-atoms collapse onto one position,
  so a repulsion term or a cap on the weights will probably be needed. The optimiser can place a
  pseudo-atom where no atom could stand, inside another, and the design use of (iii) must filter
  such fields before matching. The whole item depends on PyTorch, which is the first such
  dependency in the library and must stay optional.
  **Version (i) built and measured 31 October 2026, and gate (a) passes at every width, which is
  what keeps the item open.** `cmfqsar/fieldae.py`: `FieldGeometry` and `geometry` (what the
  closed-form overlap needs to know about a field set), `overlap` (the Gaussian pair overlap of
  `kernels.py` written for PyTorch tensors, grouped by width exactly as `atom_kernel_block`
  groups it), `FieldTensors`/`tensors` (every molecule as padded arrays -- a padded atom carries
  zero weight and a zero-weight atom contributes nothing, so a batch of molecules of different
  size needs no mask), `LandmarkEncoder`, `PseudoAtomDecoder` (positions as `box * tanh`, so a
  pseudo-atom cannot leave the frame), `FieldAutoencoder` with `codes`, `pre_image` and `sample`,
  `fit`, `relative_errors`, `fit_pseudo_atoms` (the floor), `kpca_errors` (what it is compared
  with), `decoded_volume` and `export_decoded` (the grid export the item asks for, through the
  cube writer `cofields.py` already had). PyTorch is optional in the way Numba is: the module
  imports without it, `fieldae.available()` is false, and `pyproject.toml` carries a `fieldae`
  extra.
  **The measurement** (`benchmarks/run_fieldae.py`, `results/fieldae/table.md`): COMPAS-1D with
  the field set, attenuations, field scaling and random0 split of the model of record of
  `run_compas.py` (6,942 / 868 / 868). The module's closed-form self-overlap agrees with the
  diagonal of that benchmark's stored kernel to 7e-08 relative over all 8,678 molecules, which is
  the precision of the stored float32 kernel. Gate (a), the relative reconstruction error on the
  held-out part at 32 pseudo-atoms, decoder against kernel PCA at the same number of latent
  dimensions: **2 dimensions 0.291 against 0.340; 4: 0.217 against 0.285; 8: 0.132 against 0.216;
  16: 0.083 against 0.145; 32: 0.070 against 0.082.** The decoder is ahead at every width, by
  most at eight, by least at thirty-two where kernel PCA has components enough; and the
  comparison favours kernel PCA, which gets the training mean free on top of its components and
  need not produce atoms.
  **The floor, which the item did not ask for and the gate cannot be read without.**
  `fit_pseudo_atoms` fits pseudo-atoms to one molecule directly, with no code: 8 pseudo-atoms
  reach 0.188, 16 reach 0.092, 32 reach 0.043 and 64 reach 0.073 on 40 held-out molecules of 56
  atoms. So at 8 pseudo-atoms the decoder (0.200) is at the floor and the pseudo-atoms are the
  limit; at 32 (0.132 against 0.043) the code is the limit; and beyond 32 the direct fit itself
  stops improving, which is the optimiser, not the representation -- the risk the item names
  about pseudo-atoms collapsing, for which `min_pseudo_distance` is the diagnostic and the
  repulsion term of `_repulsion` the remedy.
  **Gate (b) fails, and it is the more informative of the two.** The reconstruction error as an
  applicability-domain score, against the five one-class scores of `cmfqsar/oneclass.py` under
  the protocol of E5 (`cmfqsar/errorpred.py`), scored on the errors the model of record of
  `run_compas.py` made on its own held-out part (R2 +0.593, RMSE 0.263 eV): rank correlation
  **-0.059** for the autoencoder against +0.120 for the distance to the centre of the training
  set, +0.061 for the fifth nearest neighbour, +0.052 for the 1-SVM, +0.050 for the one-class
  Gaussian process and +0.001 for the kernel PCA distance; risk-coverage efficiency 0.467 against
  0.566 for the best. The autoencoder is last of the seven. The explanation is the property that
  wins gate (a): the decoder is trained to reconstruct every molecule of the field space and a
  nonlinear decoder generalises that to molecules unlike the training set, so reconstructing well
  stops being evidence of being near it. Note also that on this task every estimate is weak --
  the best rank correlation is +0.120 and the efficiencies lie between 0.50 and 0.57 -- so what
  the gate establishes is an ordering, not that any of them works well.
  **What the worst case is.** The single worst reconstruction of the held-out part is benzene, at
  a relative error of 1.68: the one molecule of one ring where 652 of the 868 have ten. The
  number of pseudo-atoms is fixed, so a molecule far smaller than the set gets many more of them
  than its field needs. Within the sizes the set is made of the error is steady and grows mildly
  (0.09 at seven rings, 0.14 at ten, rank correlation with the ring count +0.25). A decoder whose
  pseudo-atom count follows the size of the molecule is the obvious remedy and is not built.
  **Not done:** versions (ii) denoising, (iii) variational and (iv) semi-supervised, so gates (c)
  and (d) are not approached and `sample` draws from a Gaussian fitted to the training codes
  rather than from a variational prior -- it is the pre-image entry point the design items need
  and is labelled as what it is; no subcommand and no panel, which rule 5 now allows since gate
  (a) passes; the loss is minimised in single precision, which is reliable while the relative
  error is above about 1e-3 (two orders of magnitude below anything measured here) and would
  need double precision below that; and continuous integration does not exercise the module,
  because installing PyTorch there costs a large download for one optional module.

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
* **F7. The co-field of a nonlinear model.** A kernel layer (D-series, `kernel_layer`) made the
  program refuse the co-field, and the Shapley values were the only map left. But the layered
  prediction `Σ a_j g(K(X, X_j), K(X, X), K(X_j, X_j))` is a differentiable function of the field,
  and its derivative at one molecule is a field: `C_fX = Σ a_j g_s(X, X_j) x_fj` plus, without
  normalisation, a term along X's own field, with `g_s` the layer's derivative (`2γg` for the
  Gaussian layer, `(π − θ)/π` for the arc-cosine one, both closed forms). It is the co-field of
  the linear model tangent to the nonlinear one at X, and it reduces to the ordinary co-field for
  the linear layer. On top of it: the spread of the local co-fields over the training molecules
  (mean and standard deviation on a grid, read off the same per-molecule field matrix the noise
  band of F2 uses) and a non-linearity index per field, the share of the local co-fields' squared
  norm that their spread makes up, from the Gram matrix `B K_f B^T` without a grid.
  **Done 1 November 2026.** `cmfqsar/nonlinearity.py`; `CMFModel.training_field_basis` (the
  co-field basis without the linearity check); `cmf cofield --kind local|spread`;
  `tests/test_nonlinearity.py`, whose decisive test moves one atom's weight and checks the
  prediction moves by the overlap of the local co-field with the change of the normalised field,
  for all three layers and both normalisation conventions; manual section *The co-field of a
  nonlinear model: one per molecule*. Measured on the Hammett set
  (`benchmarks/run_nonlinearity.py`): the index is 0.10 for σp, 0.20 for σm, 0.32 for π and 0.41
  for σp·π, a target made non-additive on purpose; the Gaussian layer gains 0.10 to 0.16 in q² on
  the three constants, so they are not linear in the normalised field (π grows with the size of
  the substituent, which a cosine-normalised kernel does not see — a likely reason, not tested);
  the spread maps are stable under refitting on four-fifths of the set (r ≥ 0.96) except on π
  (0.87; its mean map 0.54 at λ = 10⁻⁴).
  **Second derivatives, done the same day.** Taken with respect to the amplitudes of probe atoms
  (finite, and the kernel normalisation differentiated exactly; `Tangent` in the same module), as
  the pair-interaction field (`pair_interaction_field`, `cmf cofield --kind pair`), the curvature
  field, the gradient field, and the atom interaction matrix with its eigenmodes
  (`atom_interactions`, `interaction_modes`, `--kind modes`); checked against second differences
  of the prediction with probes on the atoms and in free space, both conventions, linear and
  Gaussian layers; the arc-cosine layers are refused (cusp). Validated on a set made for it
  (`benchmarks/run_interactions.py`: 105 para-disubstituted benzenes, σ(X) + σ(Y) against the
  same plus 1.5 σ(X) σ(Y)): the fitted cross derivative regressed on the designed one has slope
  1.87 (r 0.73) for the interacting activity and 0.06 (r 0.30) for the additive one, the pair
  field near Y 1.43 against 0.08, stable under refits (0.97); the layer overstates the pair term
  and is curved within a group too (2.4 times the designed cross RMS), and where the curvature
  sits does not separate the two activities — the signal is the size.
  **Not done:** the interaction modes in the span of the training fields (the n by n eigenproblem
  of the field-space Hessian), which the atom-weight modes replace; the chain rule to atom
  positions; the arc-cosine layers; and the GUI, which still shows the refusal.
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
  **Done 31 October 2026.** The surface is nine names, declared in `cmfqsar/__all__` and imported
  from the package root: the four the item lists, plus `FieldSpec`, `MoleculeFields`, `ModelStats`,
  `build_fields` and `load_dataset`, which is what the manual's own examples use. The promise is
  stated in the package docstring, in the manual (chapter *Using it as a library*, section *The
  stable surface*) and in the README: within a major version a name is not removed or renamed, and
  a recorded argument keeps its name, its position and its default, while new arguments may be
  added after it. `tests/test_stable_surface.py` is what makes that checkable rather than a
  promise in prose: it records the arguments of ten entry points, fails when one disappears or
  changes its default or its place, fits and predicts through the surface alone, and reads the
  manual chapter itself, so an example there cannot go on importing a name the package no longer
  has. Whole signatures are deliberately *not* compared -- a model gains a field almost every
  week, and a test that failed on every new field would be deleted within a month.
  **The notebooks.** Four already existed and the item's third workflow did not:
  `docs/tutorials/05_screening.ipynb` is screening, on the 88 thrombin inhibitors against the 100
  presumed inactives of `examples/decoys/THR_decoys.sdf`. It runs in 3 s and shows what the item
  is really about: the lower confidence bound ranks better than the prediction (AUC 0.92 against
  0.83, enrichment at 10 % of the list 3.42 against 2.74 of a possible 4.45), and it says why the
  number is optimistic -- two COX-2 decoys outrank every inhibitor, the background was posed by
  field similarity, and 129 molecules make the top fractions meaningless.
  **Deviation from the item's text:** the notebooks live under `docs/tutorials/` and not under
  `examples/`, with `run_tutorials.py` to execute them so that the committed outputs come from the
  current code. The manual and the README point there, and moving them now would break the Colab
  links that are already published.
  **Not done:** the notebooks are not executed in continuous integration (`run_tutorials.py
  --check` exists and the workflow does not call it, which needs the `tutorials` extra in
  `requirements-lock.txt`), and the stable surface has no deprecation mechanism -- a name that
  does have to change would have to change in a major version with nothing to warn the caller
  first.
* **G4. The manual grows with the program.** Each feature lands with its manual section and its
  benchmark script in the same commit, as the probe fields and descriptors did; `docs/make_manual.py`
  rebuilds the PDF.
  **Done 31 October 2026, as far as a program can decide it.** No check can say that a section was
  *written*; two checks catch the omissions that matter, and `benchmarks/run_all.py --check` now
  runs both. First, **every registered benchmark is named in the manual** -- by its name, its
  results folder or its script. Five of the 73 were not when the check was written, and the gap it
  found was real: the manual described the ORCA stage at length and never said what it gained.
  It now does, with the numbers (`corrosion_qm`: mean q2 +0.382 for the five ordinary fields
  against +0.348 with the quantum fields and descriptors added, and +0.218 for the descriptors
  alone; `am_eu`: +0.803 against +0.803 for the ORCA charges; against the Hammett series, where
  Hirshfeld charges take sigma-meta from 0.70 to 0.79, because a substituent constant is a
  statement about resonance). `cdft_kernels`, `cofield_error` and `designloop` were quoted without
  a reference to the file the numbers come from, and now carry one. Second, **the typeset manual is
  the manual**: `docs/make_manual.py` writes `docs/CMF_manual.sha256`, the hash of the Markdown it
  read, and both `--check` commands compare it with `docs/manual.md` as it stands. A PDF carries
  its build date, so two builds of one source are different files and the PDF itself cannot be
  compared with a rebuild; the hash of the source can be, and it answers the question that matters
  -- has the manual been edited since the PDF was made? `tests/test_documents.py` runs both checks
  in the suite, and also drives each of them into failure on a temporary copy, because a check
  that has never failed is not known to work.
  **Not done:** nothing checks that a *feature* brought a manual section, only that a benchmark
  did, so a feature with no benchmark can still arrive undocumented; and the same hash mechanism
  is not applied to `docs/figures/` (`make_figures.py --check` redraws them instead, which costs
  minutes) nor to `docs/api.md` (`make_api.py --check` regenerates it, which is cheap and exact).
* **G5. Other platforms.** The application has been run on Windows only. Linux and macOS through A2
  first, then the high-DPI and OpenGL differences the 3D view will meet there.

* **G6. One documented command per stored result (5 and 6 November 2026).** Rule 6 says that one command
  rebuilds every table. For a script that is run once this holds by construction; for a script run
  several times with different options it holds only if two things are true, and for
  `benchmarks/run_hammett.py` neither was. The fault was found while item D11 re-ran the
  benchmarks its charge repair touched.
  **Done 5 November 2026.** The two rules are now stated, implemented and tested. First, every
  option that changes a number enters the tag that names the files a run writes. The tag used to
  leave out the whole protocol -- the targets, the fold count, the α and λ grids and the
  single-field fits -- so a run over one target wrote its seven rows over the twenty-one rows of the
  full run and the table then showed less than it claimed. It also exempted
  `examples/hammett/hammett.sdf` from the tag, so naming the single-conformer set by its path wrote
  over the default run's `summary.csv`, while `--single`, which is the same data set, did not. The
  tag now covers the protocol and gives one name to the two ways of naming that set.
  Second, the registry lists one command per stored summary file. It had listed four, of which two
  were the same command (the default and `--sdf examples/hammett/hammett_sym.sdf`, which *is* the
  default), one wrote `summary_qm_q_hirshfeld.csv`, a file the repository does not keep, and four of
  the six stored summaries came from commands only the README showed, two of them from no document
  at all. So `cmf benchmark --compute hammett` rebuilt three of the six stored results, added a
  seventh the table then grew a block for, and left four untouched. The registry now lists the six.
  Each run also writes `run<tag>.json`, its own command and options, and `table.md` prints that
  command above each table; a stored file with no command says so instead of looking documented.
  `tests/test_hammett_runs.py` has 17 tests and reads the stored file names from `git ls-files`, so
  a file left behind by an exploratory run cannot pass as a stored result.
  *What re-running the six found.* Four of the six came out byte for byte identical, which is worth
  recording in itself: the two with Hirshfeld charges, which the charge repair of D11 cannot touch,
  and the default run, which D11 had already recomputed. The two Gasteiger **ensemble** variants had
  not been recomputed after D11, because D11 restricted its Hammett re-run to the one command whose
  output it could attribute, and they moved by small amounts: with the five fields, uniform weights
  give $q^2$ 0.6982 to 0.6949 for $\sigma_m$, 0.6864 to 0.6890 for $\sigma_p$ and 0.7054 to 0.7115
  for $\pi$, and Boltzmann weights 0.7000 to 0.6981, 0.6855 to 0.6884 and 0.7042 to 0.7104. No
  verdict changes: the conformer ensembles still change nothing on this series.
  *And the comparison table in the README was wrong in four cells*, which is what a table built by
  hand from six files costs. One cell was stale: D11 moved the single-conformer $\pi$ value from
  0.7091 to 0.7154 and left 0.71 in the README, where 0.72 belongs. One was a rounding mistake made
  when the table was written: 0.7567 was entered as 0.75. The other two are the ensemble numbers
  above, 0.6949 and 0.7104, which were entered as 0.70 for $\sigma_m$ and 0.70 for $\pi$. All four
  are corrected. The table itself stays hand-written, so this remains possible; `table.md` is the
  generated record and the README quotes a selection from it.
  **The same two rules on the hop benchmark, 6 November 2026.** `benchmarks/run_hops.py` had the
  fault in a different form, found while asking whether item BT7 was completely implemented. Every
  run wrote the same file names, so the quick look the script's own usage lines offer,
  `--max-structures 60`, wrote its small numbers over the stored measurement; and the registry
  listed one command where the repository keeps the results of two, so `profile_fine.json`, the
  fine co-field profile the manual draws as a figure, was rebuilt by no documented command. The
  tag now covers which prepared set is read, how much of it, the envelope widths, the radius of
  the environment, the cutoffs of the site-resolved reference, the folds, the seed and the number
  of bootstrap draws; the options that only
  choose which parts of a run are done leave the tag alone, because they change no stored number.
  Each run writes `run<tag>.json` and `table.md` prints that command. The registry lists both
  commands, and a new field of the registry names the command that prepares an input a benchmark
  only reads: `prepare_hops.py`, which `--compute` does not run, because it needs the network and
  its result is kept in the repository. `tests/test_hops_runs.py` has 23 tests.
  `benchmarks/run_litraj.py`, the external check of item BT7 added the same day, was written to
  both rules from the start: its tag covers the prepared sample, how much of it, the envelope
  widths, the radius of the environment, the cutoff of the site-resolved reference, the folds, the
  seed and the bootstrap draws, its one command is in the registry beside the command that
  prepares its input (`prepare_litraj.py --fetch`, which downloads someone else's data and is not
  run by `--compute`), and `tests/test_litraj_runs.py` has 22 tests, the last of which asks the
  prepared sample to name the source and the licence of the labels it carries.
  *What re-running the two commands found.* Nothing stale, in contrast to the Hammett set, which
  is itself worth recording: the re-run passes the stored results through every change the library
  has had since 31 October, the charge repair of item D11 included. `hosts.json` and
  `profile_fine.json` came out byte for byte identical. Every number of `public.json` is
  identical and only the recorded run time differs. The disordered conductors moved in the fourth
  decimal (the transfer correlation +0.2730 against +0.2733, the frame kernel leaving one
  structure out $q^2$ -0.0644 against -0.0643), which is the order of the kernel's own summation
  and changes no reading. What the re-run did correct is the cost the registry states: 30 minutes,
  against the 67 and 40 minutes the two commands take on this machine.

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
  **Done 31 October 2026: the background per area or repeat length, the images that read the
  flags, and the height channels of a layer; measured on the 1,103 JARVIS-DFT 2D monolayers.**
  **What was built.** In `cmfqsar/periodic.py` the uniform background is the pair histogram of
  the same weights spread uniformly over the *periodic* directions alone, every atom keeping its
  place across the others: `(1/M) sum_lm w_lf w_mg J_d(h_lm)` with `M` the volume, area or
  repeat length (`periodic_measure`, `CrystalFields.measure`), `h_lm` the separation of the pair
  across the non-periodic directions and `J_d` the shell integral (crystal), the tail of the
  ring integral from `h` (layer: pairs at distance r between atoms a height h apart lie on a
  ring of radius sqrt(r^2 - h^2), so their count is 2 pi r dr for r >= h) or the axis integral
  (chain: 2 du in the axial coordinate), tabulated once per spec in `_background_profile` and
  read by interpolation; the l = m terms and the second moments of a disordered site (M2) are
  in it as before. For a crystal the code path and the numbers are unchanged (the existing tests
  compare against the shell integral to 1e-9). `image_translations` and `cell_images` make no
  translation along a non-periodic direction, so the periodic overlap kernel of a layer does
  not see its own copy across the vacuum. `slab_autocorrelation` is the second half: with
  `z_max > 0` in `PeriodicSpec` (`pbc_z_max`, `pbc_z_step`, `pbc_oriented` on the model;
  `--heights`, `--height-step`, `--oriented` on `build-crystals`) the descriptor of a layer is
  the in-plane distance of each pair resolved by the heights of both atoms above the mean plane
  of the layer, images and self term included, with the background removed per unit area
  exactly (the ring integral times the product of the height-weighted sums), both orders of a
  pair of different field types summed as the surface kernel sums them, the second moments of
  disordered sites kept, and the result averaged over the two faces unless `oriented`. It
  refuses a structure not periodic in exactly two directions. `centre_layer` gathers a layer
  that a file leaves split across the cell boundary; `--pbc ab` sets the flags that a POSCAR
  or CIF does not carry; `lattice_spread` reports the area of a layer. `benchmarks/run_jarvis2d.py`
  is the measurement and is registered in `run_all.py`.
  **Tests** (`tests/test_periodic_layers.py`, 17): a MoS2 layer gives the same channels with 5
  and 25 A of vacuum, under an in-plane supercell, a rotation of the whole frame, a shift along
  the normal and a leaning third lattice vector, with and without the height channels, while
  the same file flagged as a crystal changes with the vacuum; atoms spread at random over a
  plane at random heights give subtracted channels that are noise about zero and a subtracted
  term equal to the formula (the ring integral checked against a simulation), and a random tube
  around a chain axis the same; a chain does not see its box; the images stop at the vacuum and
  the periodic overlap kernel of a layer does not change with it; a split layer is gathered to
  the same descriptor; MoSSe is told from its mirror image by the oriented descriptor and not
  by the default, which is the mean of the two; the height sum recovers the in-plane distance
  distribution with images; a model with height channels fits, predicts, survives a save and a
  load and the command line takes a layer.
  **Measurement** (`benchmarks/run_jarvis2d.py`; `benchmarks/results/jarvis2d/table.md`): the
  1,103 JARVIS-DFT 2D monolayers, band gap, formation energy and exfoliation energy per atom,
  5 outer folds, width/regularisation/Gaussian scale chosen in 10 inner folds. Flagged as
  crystals the channels change by a median 50 % when the vacuum is
  doubled, flagged as layers by 1e-15; the crystal flags lose no accuracy within
  the set because JARVIS uses nearly one vacuum (median 20.6 Å, nine in ten within 15-25 Å). The
  per-area background does not improve the linear model (gap 0.82 /
  0.819 / 0.829 eV for crystal flags / layer flags kept / layer flags
  per area, mean predictor 1.05; formation 0.329 /
  0.349 / 0.37 eV/atom; exfoliation 66.4 /
  68.3 / 74.7 meV/atom): it removes the areal density of each pair of
  field types, as the phonons measurement found for the volume background. With the Gaussian
  layer: 0.68 eV, 0.283 eV/atom,
  56.1 meV/atom per area, against 0.67 /
  0.269 / 53.5 on the crystal flags and
  0.668 / 0.274 / 53.7 kept.
  Under the Gaussian layer the three are within the fold spread: the gain is the kernel's. With
  the 66 fields the per-area background still costs under it (gap
  0.719 against 0.67 eV, formation
  0.312 against 0.269 eV/atom), so
  `--no-background` is worth trying on a layer as on a crystal, and it keeps the invariance to
  the vacuum. The height channels (+/- 3 Å every 1 Å, four fields, 10290 channels)
  gain on every target with the linear kernel (0.785 eV, 0.349
  eV/atom, 63.7 meV/atom) and, with the Gaussian layer, on the exfoliation energy
  (51.2 against 56.1 meV/atom, R²
  0.30 against 0.14; the best result on that target,
  the crystal flags under the same kernel giving 53.5 and R²
  0.24) but not on the gap or the formation energy; oriented = symmetric
  within the fold spread. The
  66-field models: 0.679 eV / 0.271 eV/atom /
  67 meV/atom with the background kept, 0.732 /
  0.311 / 64.3 per area, no gain from the Gaussian layer.
  **Open:** heights with the element indicator fields (memory), nanotubes (NT4), a molecule on a
  periodic slab (EC6), a second monolayer set (C2DB) to measure the transfer across vacuum
  conventions that the crystal flags cannot make.
* **M2. Partial occupancy and configurational ensembles.** Solid solutions and disordered sites
  need the field weights multiplied by the site occupancy; `structures_from_matbench` takes the
  majority species today. Special quasirandom structures and MD snapshots are the periodic
  analogue of the conformer ensemble and reuse the mean-map averaging of `ensemble_fields`,
  generalised by D4's `combine_members` to members with their own atoms. The fourth-dimension verdict says to
  expect little; measure once, on a solid-solution set, and keep the mode only if it is exact and
  costs nothing when unused.
  **Done 31 October 2026, exact and free when unused, and it opens the conductor families to
  BT7.** The ensemble is treated by its mean field, which is what the mean-map kernel does, and
  for independently occupied sites that mean is closed form: `crystal_fields` gives every site
  the occupancy-weighted weights of the species that share it (`periodic.site_species`,
  `read_disordered` for a CIF with partial occupancies, `read_structures` hands such a CIF to
  it), the overlap kernel of two mean fields is the mean kernel of the two ensembles exactly,
  and the periodic autocorrelation keeps the second moments of each site for the three places
  where a site meets itself (the self term, its own periodic images, the background), so that
  the descriptor is the exact mean over the configurations too. Tests enumerate every
  configuration of a small disordered cell and check both to 1e-10; an ordered structure goes
  through the same code with identical results. The oxidation states of a fractional composition
  come from the combination of common states nearest to charge balance, every known state when
  those cannot (mixed valence). In `hops.py` every site the mobile ion holds at any occupancy is
  a site a hop can start from, the hop records both occupancies, the fingerprint of the symmetry
  reduction includes them, and the bond-valence landscape is built from the ordered framework
  (the sites the ion does not hold, each with its majority species), which is exact when only
  the mobile sublattice is disordered. No special quasirandom structures and no snapshots: the
  independent-site mean makes them unnecessary for the kernel, and correlated occupancy is what
  they would add, which this item did not measure.
  **Measured** on thirteen structures of the Crystallography Open Database
  (`examples/hops/disordered/`: Ta-doped cubic and tetragonal garnets, two perovskites, LGPS,
  two NASICON phases, Li2.92N, two LISICON phases, Li3PO4, beta-Li3PS4), ten with partly occupied
  or shared sites: 6408 hops, 298 distinct, 298 with a
  bond-valence barrier. The percolation thresholds of the hop network agree with those of
  `bvlain`'s own grid where the Li sublattice is the disordered part (cubic LLZO 0.40 against
  0.41 eV, LGPS 0.53 to 0.59 against 0.57 to 0.60, beta-Li3PS4 0.25 to 0.40 against 0.27 to
  0.42, Li2.92N 0.16 against 0.16, LISICON 0.45 to 0.49 against 0.47 to 0.52), which is the
  check that the hops between 1.5 and 4.5 A are the paths the landscape uses; the ordered
  NASICON has no Li pair in that window, and the tetragonal garnet and Li3PO4 differ because the
  network measures from the lowest site and the grid from its own minimum. The model is a
  different matter: the public model gives q2 -1.075 (correlation +0.273, error
  1.438 eV) on these hops, and within the set, one structure out at a time, the frame
  kernel gives -0.064 against +0.166 for the counts,
  -0.230 [-0.886, +0.143]: twelve labelled structures dominated by
  one garnet with 98 distinct hops, and barriers up to 15 eV where a straight path runs through
  an atom, are not a set a model can be measured on. The table is in
  `benchmarks/results/hops/table.md` and in the manual's section on migration paths.
  **Not done:** correlated occupancy (a pair of sites never occupied together), which the
  independent mean cannot express and a snapshot ensemble would; the shared La/Li site of the
  perovskites leaves with the migrating ion when it is a hop's end; no GUI.
* **M3. DFT arrays as fields without hand-made files.** Bader charges and magnetic moments enter
  as per-atom arrays (`prop:<name>`); a reader for VASP outputs (ACF.dat, OUTCAR magmoms,
  OSZICAR energies) feeds the kernel from the Li interface and Fe(110) runs directly. No new
  science; it removes the step every materials study has had to script by hand.
  **Done 30 October 2026.** `cmfqsar/vasp.py`: `structure_with_arrays` reads a run directory
  (`CONTCAR`, `ACF.dat`, `OUTCAR`) into an ASE structure whose arrays carry the net Bader
  charges (valence electrons of the pseudopotential, read from the OUTCAR, minus the Bader
  electrons; also set as the initial charges so that they survive extended XYZ), the raw Bader
  column, and the magnetic moments of the last magnetisation block when the run was
  spin-polarised. `crystal_charges` takes them under the charge model `"file"`, so
  `inorganic_fieldset(..., charge_model="file")` builds the electrostatic field from DFT charges
  with nothing written by hand. A count that does not match is refused rather than
  misassigned. 8 tests; the three hosts of the Li interface study are in `examples/hops/` in
  this form (mean Bader charges Li +0.89, P +3.63 to +3.68, B +2.31 to +2.35, O -1.55 to -1.60).
  **Not done:** no OSZICAR reader beyond the final energy of the OUTCAR, and no reader for
  other codes.

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
  **Built and measured 31 October 2026. The kernel is built and its properties are exact; the
  half of the gate that asks it to beat the autocorrelation is not met on the only public set
  large enough to try it on, and the measurement says why.**
  **What was built.** `cmfqsar/periodic.py`: `image_translations` and `cell_images` (the images
  of a cell out to a radius, measured from the box the cell spans so that one expansion serves
  every structure of a series), `expanded_cell_table` (the same for a whole table, cached on it
  because a grid search asks for it once per attenuation), `overlap_cutoff` (the radius at which
  the widest field of the set has fallen to 1e-6 of a peak overlap -- the cutoff rule of item B2,
  read from `kernels.py`), `periodic_overlap_kernels`, `periodic_self_overlaps`,
  `overlap_asymmetry` and `lattice_spread`. The kernel is the ordinary kernel evaluated against
  the image-expanded table, so the grouping by width, the cutoff and the compiled loop of item B5
  carry over untouched. In `cmfqsar/model.py` it is the `fields` kernel source with
  `pbc_overlap=True`, which routes `_pair_kernels`, `_self_overlaps`, `_training_side` and
  `cofield_basis` through the images: the normalisation, the scaling, the learners, the co-field
  and the per-site contributions are then the same code for a crystal as for a molecule, and the
  co-field drawn on a grid is periodic, which `cofields.py` could not do with open boundaries.
  22 tests, each on a property rather than a number: the open-boundary limit (with a box far
  larger than the fields reach the kernel *equals* `kernel_matrices` exactly), writing the atoms
  outside the cell instead of inside it (unchanged, while the open-boundary kernel of the same
  atoms moves by more than 1 %), the supercell (the raw integral multiplies, the cosine kernel
  does not), decomposition rule 3 (the per-site contributions plus the bias are the prediction to
  1e-6), the periodicity of the co-field, and the equivalence of the three oxygens of a cubic
  cell.
  **Two limits the tests pin down rather than hide.** The integral runs over the cell of the row
  and the images of the column, so with cells of different size the two orders are different
  integrals and the kernel is their average; `overlap_asymmetry` reports the disagreement, which
  is 7e-09 on one lattice and 17 % across a 2.5 % lattice mismatch, and is the number that says
  whether a set really is one lattice. And the image sum is truncated like the open-boundary
  kernel, so the invariances hold to that truncation and improve as the radius grows.
  **The measurement** (`benchmarks/run_periodic_overlap.py`, `results/periodic_overlap/table.md`):
  1,500 cells of `matbench_perovskites` with their formation energy -- the largest public set
  whose members share a cell topology -- five Matbench folds, the same four fields (the three of
  `INORGANIC_FIELDS` and the inverse square root of the mass, chosen by measuring the
  autocorrelation with three, four and seventeen fields: R2 +0.370, +0.495, +0.587) and the same
  ridge grid for every row. The cells are relaxed, so their lattice constants differ by up to
  82 % in volume, which is a frame error for a point-by-point comparison; the second frame gives
  every cell the mean lattice constant and keeps its fractional coordinates, which puts the set
  on one lattice exactly (spread 0, asymmetry 0).
  R2, and MAE in eV for the whole five-atom cell, which is what Matbench's target is:
  **autocorrelation as given +0.497 (0.384)**, autocorrelation on one
  lattice +0.494 (0.405), **periodic overlap on one lattice +0.479 (0.407)**, overlap with the
  images left out, as given, +0.448 (0.404), the same on one lattice +0.379 (0.435), **periodic
  overlap as given +0.350 (0.449)**.
  **What the table says.** The frame does to each kernel what it should: nothing to the
  autocorrelation (+0.497 against +0.494), which is invariant; **+0.129 to the periodic
  overlap**, the largest effect in the table and exactly what a kernel that compares fields point
  by point must show; and -0.069 to the same overlap with the images left out, which without them
  compares five atoms as a cluster that the rescaling only distorts. The overlap kernel does not
  beat the autocorrelation, and the reason is in the same column: the operation that gives it a
  frame takes the lattice constant out of it, and for a formation energy that is a strong
  predictor on its own. On a set of different compounds there is no frame to be had for free. So
  the kernel is for the series the item names -- one host lattice with a substitution, a vacancy
  or a dopant moved from site to site -- where the frame is given and nothing has to be rescaled
  to get it.
  **The co-field, which is the other half of the gate and the reason for the kernel.** Drawn
  inside the cell (`results/periodic_overlap/cofield.png`, the planes z = 0 and z = a/2 of a
  model on all 1,500 cells, R2 +0.483 on its own training set) it is periodic and its strongest
  features sit on the sites: the steric co-field has a deep negative lobe at the B site and at
  the oxygen and a positive one at the A site. The per-site form of rule 3, over a sample of 150
  cells: A +2.37 +- 0.58, B -6.16 +- 0.81, X -3.87 +- 0.55 eV, where the means carry a
  constant the bias cancels and the spread is the information -- the B site varies most. So the
  kernel does name sites, which is what the gate asks of the co-field; it names them on
  perovskites rather than on the Li vacancy cells, and the item's own gate is written for the
  latter. The asymmetry of the two orders of the integral over this set is 0.0 % on one lattice
  and 200 % in the cells as they come -- in the raw cells the two orders of one overlap can have
  opposite signs, which is the sharpest way to say that those cells are not a frame.
  **Not done:** the vacancy cells of the Li3PO4/Li3BP2O8/LiBO2 study themselves, which is where
  the item points the kernel and where the first half of the gate belongs: they live in another
  project and rule 6 wants them prepared by a script under `examples/` first, and the six sites
  computed so far carry no interval. No background subtraction of the kind the autocorrelation
  has (the overlap of two cells is dominated by the product of the mean fields, which carries
  composition and not structure; subtracting `V m_f^i m_f^j` would leave the fluctuation
  correlation and needs the same term in the per-site decomposition to keep rule 3). The training
  kernel is computed in full rather than by triangle, because the two tables differ in size, so
  it costs twice what a symmetric variant would. Probe fields are refused and the Shapley
  decomposition over sites with them. And there is no subcommand and no panel: rule 5 keeps them
  until a gate is passed.
* **M5. Site-resolved models.** The periodic channel sum is a sum over the atoms of the cell, so it
  splits exactly into one contribution per site, as the atom contributions split the overlap
  kernel. That gives a local co-pattern (which site carries the property) and a route to per-site
  targets: adsorption energies on different surface atoms, defect energies by site, NMR shifts,
  Bader charges. Needs site-centred features (the row of `T` for one atom), per-site targets in the
  data model and the decomposition in `top_channels`. Measured first on the Fe(110) adsorption
  energies resolved by site and on vacancy energies by site from M4's cells.
  **Done 1 November 2026, on two sets that were already in the repository rather than on the
  Fe(110) adsorption energies, which still need runs nobody has made. The split is exact; on the
  single-atom catalysts the site profile is level with cutting the site out of the sheet and
  level with the whole cell, which answers the question EC8 left open with "the cut is not
  needed" and not with "better"; and the local reading of a whole-cell model puts eleven times
  an even share of the weight on the metal site.**
  **What was built.** `cmfqsar/sites.py`: `site_autocorrelation` splits the periodic
  autocorrelation of a cell into one row per site by grouping the ordered pairs on the atom that
  stands first, with the self term, the second moments of a disordered site (M2) and the uniform
  background of each site in it; the mean of the rows is the descriptor of the cell to machine
  precision, for a crystal, a layer and a chain alike. The rows are not divided by the atom
  count, so a site profile does not change when the site is written in a larger supercell.
  `site_fields` makes a site a **sample**: a `CrystalFields` whose new `centre` field names the
  site, which `periodic.periodic_autocorrelation` answers with that site's row alone (the only
  change in `periodic.py`: four lines and a dataclass field), so `site_table` plus
  `CMFModel(kernel_source="periodic")` fits a set of sites with no new model code -- the
  normalisation, the block weights, the learners, the grid search and the file format are the
  ones a set of cells uses. One split per cell serves all of its sites (a one-entry cache keyed
  by the cell and the spec). `atomtargets.AtomTargets.from_arrays` is the data model of LP1 for a
  target that a structure file cannot carry: one array per cell, *not a number* where a site was
  not measured, with the folds of LP1 unchanged, which hold out whole structures.
  `shell_count_features` is the reference under rule 1 (the element at the site and the counts of
  each element in distance shells around it), and `site_contributions` reads a **whole-cell**
  model site by site: the cell's channels are the mean of its site rows and the linear layer is
  linear, so the prediction minus the bias is a sum of one number per site, exactly, which is
  rule 3 for the autocorrelation kernel; `site_top_channels` splits one site's number over the
  channels.
  **Tests** (`tests/test_sites.py`, 19): the mean of the site rows is the cell descriptor for a
  crystal, a layer, a chain, a supercell and a disordered cell, with and without the background;
  a site record's autocorrelation is its row; equivalent sites of a supercell get the same
  profile and the two kinds of site do not; a supercell, an origin shift and a rotation of the
  frame change nothing; the vacuum of a layer changes nothing; a change beyond the cutoff changes
  nothing while the same change inside it does; the folds hold out whole structures; a model on a
  site table fits, predicts and survives a file; the contributions of a whole-cell model sum to
  its prediction, for an ordered and for a disordered cell, and are equal for equivalent sites;
  the channels of one site sum to its contribution; a nonlinear layer is refused, as are the
  height channels of a layer; one cell is split once however many of its sites are asked for.
  **Measurement** (`benchmarks/run_sites.py`; `benchmarks/results/sites/table.md`), three parts.
  **The single-atom catalysts of EC8**, where one site of every cell is measured: 139 M@C(4-x)Nx
  sheets, the binding energies of H, O and OH on 104 of them and of OOH on 91 (six bare sheets
  had been recomputed with plain PBE when this ran and 29 more since, so neither the pilot's
  numbers nor the rerun of 5 November are comparable and every reference is recomputed here),
  the pilot's three
  fold designs, every descriptor a family choosing its settings in an inner cross-validation of
  the same kind. Mean q² over the four endpoints, random / one metal held out / one motif held
  out: **the site profile +0.792 / +0.721 / +0.755**, the invariant kernel of the cut cluster
  (LP2) +0.801 / +0.739 / +0.712, its overlap kernel (EC8) +0.782 / +0.695 / +0.724, the
  autocorrelation of the whole cell +0.794 / +0.756 / +0.744, counts in shells around the site
  +0.758 / +0.096 / +0.749. In the twelve paired comparisons (four endpoints by three designs,
  the catalysts resampled) the site profile is level with every descriptor built from fields: two
  better and one worse against the cut-cluster overlap, one better and none worse against its
  invariant kernel, one better (the OH binding with a motif held out, +0.066) and two worse (the
  O and the OOH binding with a metal held out, -0.046 and -0.085) against the whole cell. It is
  clearly better only than the counts, in five of the twelve. **So the site resolution of the
  periodic kernel does the work that cutting a cluster out of the sheet does -- which is what
  EC8's gate asked, and the cut is not where the pilot's accuracy came from -- and it buys no
  accuracy over the whole cell on a set whose cells differ only at one site.** The counts
  collapse to +0.096 when the metal is new, as the one-hot did in the pilot, while every
  descriptor built from fields keeps most of its accuracy. The set is small enough to move: one
  recomputed sheet, added between two runs of this benchmark, took every method's OOH q² from
  about +0.93 to about +0.77, so the differences between the methods are what this measurement
  gives and the absolute values are not. **The lithium sites**, where many sites of a cell are
  measured: the bond-valence site energy of all 104 Li sites of the three electrolytes of
  `examples/hops`, a fold being a whole compound. The model cannot place an unseen compound on
  the energy scale (offsets -0.12, +0.04 and +0.13 eV; the overall q² is negative for every
  method, the mean of the other compounds included, because with three compounds nearly all of
  the variance is between them) and inside a compound it ranks the sites (Spearman +0.78, +0.84
  and +0.55; q² about the compound's own mean +0.60 and +0.77 for the two whose sites differ by
  more than 0.01 eV). **A whole-cell model read site by site**: on the OH binding energy of the
  104 catalysts (q² +0.893 in five folds, 0.51 eV) the metal site carries 8.9 % of the weight,
  11.3 times what an even spread over the 127 sites would give, the metal with its four
  neighbours 16.0 % against 3.9 %, and everything beyond 5 Å about half an even share; the
  channels at the metal name the d electron count for Pt@C4 and the electrostatic field at 2.0 Å
  for Fe@N4. The figure is `benchmarks/results/sites/contributions.png`.
  **What is open.** The height channels of a layer (M1) have no site split, so a site of a slab
  is described by distances alone and cannot tell a neighbour above the plane from one below it;
  the Fe(110) adsorption energies by site, which the item names first, still need the UMA or VASP
  runs, and so do defect energies by site; a set with many measured sites per cell **and** more
  than a handful of cells is the measurement the lithium set could not be, since three compounds
  cannot separate the offset from the ranking; and per-site targets have no subcommand and no
  panel, which rule 5 keeps until a gate is passed.
* **M6. Explicit reciprocal-space channels.** Order beyond the cutoff is invisible to a pair kernel.
  The low-|G| part of the typed structure factors, with the Gaussian form factor as the resolution
  cutoff, added as extra channels in the manner of the reciprocal half of an Ewald sum, restores
  long-range electrostatic and superstructure information; the same module gives the s-space
  co-pattern, the α-as-resolution reading and the Parseval test that the publication plan's theory
  section needs. Measure on the elastic moduli and formation-energy tasks, where long-range order
  should matter, against the real-space kernel with the same cutoff.
  **Done 1 November 2026. The channels carry what the item claimed: alone they are level with
  the best real-space cutoff on all three tasks with no neighbour list at all, added to it they
  gain clearly on one task of the three, and -- the result worth keeping -- they carry long-range
  order at a cost that does not grow with the range, where lengthening the real-space cutoff
  costs about 0.10 of R² on both elastic moduli. The width of the field behaves as the resolution
  the item said it was, and the Parseval test holds to one part in 1e13 on real structures.**
  **What was built.** `cmfqsar/reciprocal.py`: `ReciprocalSpec` lays out channels in |G| like the
  distance channels, `structure_factors` and `reciprocal_vectors` build the typed structure
  factors of one cell over the reciprocal lattice out to `q_max` (one plane of indices at a
  time, because the whole box of indices is many times the sphere it contains and a large cell at
  a high cutoff would cost gigabytes; `count_reciprocal_vectors` says what a request will cost
  and the call refuses one above a stated limit), and
  `reciprocal_autocorrelation` makes the channels -- the intensity of a pair of field types
  averaged over the directions of G, per atom of the cell. Three properties are exact rather than
  approximate: the uniform background is the G = 0 term and is left out, so no integral is
  approximated to remove it; the descriptor does not change with the choice of cell, because the
  one quantity that does not depend on G (the self scattering, and the second moments of a
  disordered site, M2) is added from its integral over the channel instead of being summed over a
  grid that a supercell makes denser; and the form factor exp(-a² q²) makes the width of the
  Gaussian field a resolution rather than a cutoff, since every atom enters every channel whatever
  the distances. `parseval_residual` is the identity the theory section needs: the real-space
  lattice sum of a Gaussian against the sum over the reciprocal lattice, computed by the
  neighbour list of `periodic.py` and by the structure factors of this module, so it tests the
  two against each other. `reciprocal_pattern` and `pattern_table` read a fitted model as weight
  times the spread of the channel, the ranking `top_channels` uses. `joint_features` puts the two
  descriptions in one matrix. The hook is `kernel_source="reciprocal"` in `model.py` (four
  options, a spec, a feature branch and the file format) and one dispatch in
  `periodic_channel_labels`; no model code is new.
  **Tests** (`tests/test_reciprocal.py`, 28): the reciprocal lattice is complete and bounded
  correctly for any shape of cell; the structure factor at G = 0 is the sum of the weights; the
  Parseval identity holds for a primitive cell and two supercells and at three widths; a
  supercell, another primitive cell of the same lattice, a shift of the origin and a rotation
  change nothing to 1e-12, while a different structure does change the channels; two cells whose
  substituted sites are 5.4 and 10.9 Å apart are identical to the real-space descriptor with a
  4 Å cutoff and different to these channels, and with 10.9 and 21.7 Å they are identical to it
  with a 10 Å cutoff as well -- which is the cutoff the benchmarks of this program use; the form
  factor falls with q and faster for a wider field; dropping the self term changes only a smooth
  curve and keeping G = 0 adds the forward scattering to the lowest channels; a disordered cell
  uses its second moments and is still independent of the choice of cell; a layer, a site record
  (M5) and a set built from two field sets are refused; the blocks and the labels line up and the
  shared helper of `periodic.py` hands a reciprocal spec to this module; a model fits, predicts,
  survives a file and reads back.
  **Measurement** (`benchmarks/run_reciprocal.py`; `benchmarks/results/reciprocal/table.md`).
  Three Matbench tasks -- the formation energy of `matbench_perovskites` and the two elastic
  moduli `matbench_log_kvrh` and `matbench_log_gvrh` -- 1,200 structures of each, the five
  published folds, the descriptor setting, the mixing weight and the ridge chosen inside every
  training fold, the two descriptions combined as `K = (1-t) K_real + t K_reciprocal`.
  *Alone:* R² +0.518 against +0.528 for the best real-space cutoff (formation energy), +0.778
  against +0.773 (bulk modulus), +0.728 against +0.735 (shear modulus); all three differences
  have an interval containing zero. *Added to the best real-space cutoff:* +0.066
  [+0.045, +0.090] on the formation energy, +0.015 [+0.002, +0.027] on the shear modulus,
  +0.008 [-0.005, +0.022] on the bulk modulus; the folds give the reciprocal kernel a weight of
  0.25 to 0.55. *The cost of range:* taking the real-space cutoff from 4 to 10 Å loses 0.102 and
  0.099 of R² on the two moduli, and adding these channels to the 10 Å descriptor gives back
  +0.102 [+0.052, +0.172] and +0.076 [+0.051, +0.103]; the number of reciprocal channels does not
  grow with the range. *Resolution:* on the bulk modulus, R² +0.785 at a = 0.15 Å falling to
  +0.587 at 0.7 Å, and at most 0.002 of change when the sum over G is taken to 12 Å⁻¹ instead of
  8 -- once the form factor has decided what is visible, summing further changes nothing.
  *The identity:* about 1e-13 over 120 crystals of the three tasks, worst case 2e-11. *Reading
  the model:* on the bulk modulus the channels that move the prediction most sit at q = 2 to
  3.5 Å⁻¹, a spacing of 1.8 to 3.1 Å. A fourth stage draws the concept on two cells built for the
  purpose (`--stage concept`): two substituted sites 5.4 and 10.9 Å apart give a real-space kernel
  of 1.000000 at a 4 Å cutoff and 0.9550 in reciprocal space.
  **What is open.** Three periodic directions only: a layer or a chain has a reciprocal lattice of
  lower dimension and the intensity then depends on the direction across the vacuum, so M1's
  structures are refused rather than approximated, and that is the first thing to add. No site
  split (M5), because a structure factor is a sum over the whole cell. The channels are blind to
  a mirror image, like every two-point descriptor, which is M10. The gain was measured against
  the autocorrelation on one field set of four property fields; indicator fields per element
  would make these channels the partial structure factors a diffraction experiment measures and
  were not affordable in the same comparison. And M12 (learning from measured G(r) and S(Q)) was
  held until this item reported: it carries signal, so M12 is now open.
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
  **Done 2 November 2026. Both blind spots are real and both blocks close them, and the item is
  cheaper than this entry assumed: with the centre atom as the origin, the factorisation of the
  molecular chirality block does survive the lattice sum, because a determinant is linear in each
  of its three arguments. No cubic sum is needed; both blocks cost one pass over the pairs of
  neighbours of each atom. The result that justifies the item is on the defect series of M14.
  Trained on the cells with one or two defects and asked about the cells with three, the two-point
  description fails (R² -0.015 and 1,222 meV per site on MoS2, +0.060 and 1,096 on WSe2) and the
  three-body channels hold (+0.940 and 240 meV per site, +0.954 and 198). On ordinary property
  tasks the gain is small and depends on the task: +0.089 [+0.041, +0.140] on the perovskite
  formation energies, where the signal is a tilt angle, and nothing measurable on the two elastic
  moduli. The handedness block is tested only on constructed structures, because the repository
  holds no set of chiral crystals with a measured property; that set is what the block now needs.**
  **What was built.** `cmfqsar/threebody.py`. `threebody_autocorrelation` takes every atom of the
  cell as a centre, builds the neighbourhood from the ASE neighbour list (the centre itself at
  radius zero, so its own field types enter the channels, and every periodic image within the
  cutoff), and sums over pairs of neighbours the product of two Gaussian radial windows and one
  window on the distance between the two neighbours -- the three sides of the triangle, which fix
  it. Dividing by the atoms in the cell makes a supercell give exactly the same channels. The
  channel layout is the one of `cmfqsar/triangles.py` (item C5), so a model needs no new block
  bookkeeping and `block_group_keys` works unchanged. `periodic_chirality` is the handedness
  block: per centre and per radial window, the vector sum of the field-weighted neighbour
  directions, and then the determinant of three such vectors, one per field type. It is off by
  default. The module states when it is empty: a weight that depends only on the species makes
  every vector a combination of one vector per species, so fewer than three species give
  coplanar vectors and every channel is zero, and a rotation axis through the centre gives zero
  as well. `threebody_features` and `threebody_feature_matrix` build rows for a model,
  `threebody_channel_labels` names the channels, `channel_spread` and `pattern_table` rank them
  for interpretation, and a site record (item M5) gives the channels of one centre, whose mean
  over the centres is the whole-cell row. The model hook is `kernel_source="threebody"` with
  `tb_r_max`, `tb_step`, `tb_sigma`, `tb_chirality` and `tb_chirality_shells`.
  **Tests** (`tests/test_threebody.py`, 26): the neighbourhood holds the centre at radius zero
  and the 1 + 6 + 12 shells of rocksalt, and a layer has no neighbour across the open direction;
  a supercell, a shift of the origin and a rotation change no channel by more than 1e-12; the
  homometric pair gives a two-point kernel that differs from one by less than 1e-12 and a
  three-body kernel below 0.99; a mirror image leaves the three-body channels unchanged and
  negates every handedness channel; a centrosymmetric cell gives handedness below 1e-20 and a
  two-species cell gives handedness that is noise against the four-species value; the layout,
  the labels and the block sizes agree with each other; `include_self` moves the channels it
  should; a site record averages to the cell; a model fits, predicts and survives a file round
  trip; the pattern table ranks the channels; a disordered cell, two field sets, too many
  channels and an empty structure are refused.
  **Measurement** (`benchmarks/run_threebody.py`; `benchmarks/results/threebody/table.md`).
  *The concept stage* is the two blind spots: the homometric pair of five atoms on a ring of
  twelve sites gives a two-point kernel of 1.00000000 and a three-body kernel of 0.9372; a cell
  of four species with no symmetry and its mirror image agree in the three-body channels to
  6.9e-16 of the largest channel and have handedness channels of the same size and the opposite
  sign (kernel -1.000000); rocksalt gives 2.5e-45 in every handedness channel. *The property
  stage* is the three Matbench tasks of M6, 600 structures of each, the five published folds, the
  setting and the ridge chosen inside every training fold: R² +0.453 / +0.781 / +0.699 for the
  two-point descriptor, +0.542 / +0.789 / +0.727 for the three-body channels and +0.515 / +0.799
  / +0.721 for the two kernels mixed, so the three-body gain is +0.089 [+0.041, +0.140] on the
  formation energies and inside the noise on both moduli (+0.009 [-0.022, +0.041], +0.028
  [-0.003, +0.065]). *The defect stage* is the series of M14, 2,000 cells of MoS2 and of WSe2,
  191 atoms per cell, with the two splits of that item. On random folds the two-point description
  is better (8.3 and 15.7 meV per site against 36.0 and 38.3), because it reaches 10 Å where the
  three-body channels reach four to five; on the split from one and two defects to three the
  two-point description fails and the three-body channels hold, as the paragraph above reports.
  The overlap kernel of M14 is better than both (4 and 54 meV per site on the second split) but
  needs one shared lattice and supercell, which the three-body channels do not. *The mixed row*
  is between the two on random folds (25.6 and 23.8 meV per site) and fails with the two-point
  row on the second split (+0.021 and +0.091), although its three-body part alone succeeds there.
  The random folds all chose a quarter of the weight for the three-body kernel and the
  extrapolation split chose three quarters, so the failure is not a matter of too little weight;
  the likely reason is that the two-point kernel is close to one for every pair of cells of one
  lattice, so what it adds is nearly constant and compresses the differences the three-body part
  supplies. A second reading is a lesson about model selection and not about this descriptor --
  the weight and the layer are chosen on a random inner split, where both descriptions look
  nearly perfect, so the choice cannot see the failure that the outer split tests for. *The cost:* a cell of 191 atoms takes 0.26 s for the 14,112 channels
  of the first setting, and the two feature matrices of 2,000 cells are 0.11 and 0.23 GB; one
  host's three-body row -- the features of both settings, twelve kernels over five folds and the
  second split -- took 35 minutes, and its mixed row eleven.
  **What is open.** A set of chiral crystals with a measured property, which is the handedness
  block's own gate: today the block is tested on constructed structures only, and nothing says
  whether a measured property follows it. Inside the program, three things are left. The channel
  count grows as the square of (field types times radial windows), which is what holds the cutoff
  at four to five angstrom; a selection of channels by `channel_spread`, or a random projection,
  would let the block reach further and is not built. The mixed kernel asks for an inner split
  that matches the question the outer split asks -- an extrapolating inner split for an
  extrapolating test -- which belongs to the model-selection code and not here. And a disordered
  cell is refused, because the mean over the species sharing a site is not the mean of the
  product of three weights; the correction needs the pair and triple occupancy correlations,
  which the structure format does not carry.
* **M11. Inverse design on prototypes.** The co-pattern says which pair at which distance to add.
  Making that usable for design needs a candidate generator over prototype structures with substitutions,
  ranking by the lower confidence bound of the GP view (`benchmarks/run_bound.py` measures it), and a
  loop that proposes, computes with VASP or UMA, and refits. The GP exists; the enumerator and the
  loop do not. Gate: the loop must find, on a held-out Matbench task, a structure better than the
  training maximum within a fixed budget of calculations.
  **Done 2 November 2026. The gate is met by the two rules that act on the prediction, and the
  measurement that matters most was not the gate.** On 4,000 cells of `matbench_perovskites`, with the formation
  energy to be made small, a budget of 100 calls after the start set and ten start sets shared by
  every rule, the greedy rule reaches the best cell of the pool (median rank 1 of 4,000) and
  improves on its start set in 7 of the 10 runs; one of the ten start sets already held the best
  cell, so 9 runs could improve at all. It enters the best one per cent after a median of 8 calls.
  The lower confidence bound improves on its start set in 7 of 10 runs as well, at median rank 9,
  while random choice improves in 5 of 10 and the rules that only explore in 1 to 4 of 10.
  With 25 computed cells instead of 100 the three-body kernel of M10 still reaches median rank 1
  and the two-point kernel reaches rank 15, which is the one place in this study where those
  channels decide the outcome. Random choice gives median rank 27 and 14 for the 125 and 200 cells
  it computes, which is chance (22 and 14 expected), and the rules that explore are no better than
  chance on this pool. **The finding to carry forward is the failure.** On the 2DMD defect series
  of M14 the same loop reaches median rank 1 of 2,000 on WSe2 in 10 of 10 runs and fails on MoS2:
  median rank 10 with 100 computed cells, the best one per cent reached in 0 of 10 runs, worse than
  random choice, while the rules that explore do reach it (`ei` median rank 1, `maxmin` in the best
  one per cent in 10 of 10). A stage added for the explanation (`--stage ordering`) measures how
  well the model orders the candidates it has not computed: on MoS2 it is accurate, rank
  correlation +0.87 and R² +0.84, better than the perovskite pool at its best (+0.53 and +0.31),
  and it still puts the best candidate at place 894 of 1,900. On WSe2, at +0.93 and +0.88, the best
  candidate is at place 59. The pools differ, not the models: the best one per cent of WSe2 lies
  within 0.002 eV of the best arrangement, while on MoS2 the best arrangement lies 0.41 eV below
  the rest of the best one per cent, and a kernel ridge model shrinks an isolated value toward the
  mean. **For a design loop the number to report is not the accuracy of the model but the place it
  gives the best candidate**, and the rule has to be chosen for the pool.
  **The closed loop.** A third stage computes every value inside the loop: 8 of the 32 sites of an
  fcc copper cell take gold (one composition, 10,518,300 arrangements; a sample of 3,000 leaves
  2,459 distinct candidates and 400 of them are the pool), an ASE EMT calculator relaxes each
  proposal at 0.22 s per call, and the whole pool is also computed exhaustively (88 s) so the rank
  is exact. Every rule that uses the model finds the best candidate of the 400 in 5 of 5 runs,
  `greedy` at its first call, against 1 of 5 and median rank 3 for random choice. The budget is a
  quarter of the pool, so the stage shows that the loop runs end to end and that the model helps,
  not by how much. The first version of this stage let three elements take four sites, and that
  search was solved by random choice in 3 of 5 runs because the composition carried almost all of
  the energy difference; fixing the composition is what made it a test of the arrangement.
  **What was built.** `cmfqsar/prototypes.py`. A `Prototype` is a cell plus `SiteClass` groups,
  each with the elements its sites may take; a class with no count is a composition choice (every
  site takes the same element) and a class with a count is an arrangement choice (that many sites
  take a substituent), and the element name `vac` removes the atom. Nothing is built until it is
  asked for: `n_candidates` is a product of binomial coefficients, `candidate_at` decodes one index
  into one structure, and `candidates` draws a sample without replacement from an enumeration too
  large to walk. `site_permutations` turns the spglib operations of the prototype into permutations
  of the substitutable sites and `unique` keeps one candidate per orbit; `channel_keys` and
  `unique_by_channels` are the descriptor's own, weaker test. `LookupOracle` and `CalculatorOracle`
  are the two oracles, both counting calls, the second relaxing with any ASE calculator.
  `design_loop` is the loop: choose the ridge by evidence (B4), fit with the helper of the
  acquisition study (E2), score with its rules, ask the oracle, refit. `benchmarks/run_prototypes.py`
  has five stages (`space`, `loop`, `defects`, `ordering`, `closed`) and `tests/test_prototypes.py`
  has 36 tests.
  **What is open.** The pool is fixed: a space too large to enumerate must be sampled before the
  loop starts, and a generator that proposes near what the model likes is the next step. The
  generator substitutes on a fixed cell, so the candidate the model sees is not the relaxed
  structure the oracle returns, and the difference grows as the ions differ more in size. Both tests of
  which candidates are one structure disagree with each other, and the descriptor's test is the
  weaker one: of 10 symmetry-distinct arrangements on 16 fcc sites both descriptors give 8 channel
  sets at every reach tried, and the merged pairs differ by 0.6 meV per atom, which is an error no
  model on these channels can avoid. No DFT oracle was run: VASP and UMA plug into
  `CalculatorOracle` unchanged, but the cost of a real design study was not measured. And the
  perovskite composition space is 98.4 % computed already in `matbench_perovskites`, so the next
  measurement this item needs is a space whose candidates nobody has computed -- the A2BB'O6
  prototype (140,608 candidates, none computed) is the obvious one.
* **M12. Learning from scattering data.** Because the kernel is the Debye transform, a measured pair
  distribution function or structure factor is a total, untyped version of the same object. A
  reader for experimental G(r) and S(Q) and a kernel that compares measured curves with computed
  typed ones would allow models on amorphous and nanocrystalline materials where no atomic
  structure exists. Exploratory; no commitment until M6 has shown that the reciprocal channels
  carry signal.
  **Done 1 November 2026. The identity the item rests on is exact, the reader works, and the
  accuracy cost of losing the types is large: the one curve an experiment can measure reaches
  R² +0.303 where the typed descriptor reaches +0.771 (bulk modulus) and +0.192 where it reaches
  +0.684 (shear modulus), and on the formation energies the curve is no better than the mean. Two results were not expected and matter more than that number: a curve as
  computed is too sharp for a kernel to compare, so the finite range of the instrument is what
  makes the comparison work, and a model fitted on computed curves cannot predict measured ones
  at all. The route to amorphous materials is open, but only with curves computed at the
  resolution of the measurement.**
  **What was built.** `cmfqsar/scattering.py`. `pair_distribution` computes G(r) as the one-field
  periodic autocorrelation of `periodic.py`, with the uniform background kept and the self term
  dropped, divided by r, by the mean scattering length squared and by the integral of the
  channel's Gaussian; the background term of the descriptor is exactly the -4 pi r rho_0 that the
  standard definition subtracts, so no new sum is written. `structure_function` gives F(Q) =
  Q[S(Q) - 1] by two routes, the sine transform of G(r) and the one-field reciprocal channel of
  M6 divided by the mean scattering length squared, which tests them against each other. The
  Faber-Ziman convention is what the self term is for: with it left in, S approaches
  <b²>/<b>² instead of 1 and a straight line in Q is added to every curve of more than one
  element. `scattering_record` builds the one field, with the X-ray weight the atomic number and
  the neutron weight a tabulated scattering length (Sears 1992; the elements whose value depends
  strongly on the wavelength are left out and a structure containing one is refused).
  `measured_like` applies the four things that separate a computed curve from a measured one: a
  finite range in Q, counting noise on F(Q), an error in the number density and a limited range
  in r. `read_curve` reads the two-column text the common measurement programs write, with the
  header keys they use, `resample` puts a curve on the grid a model uses, and `curve_kernel`
  compares measured curves with computed ones. `pair_distribution_residual` is the test of the
  identity against a direct sum over the neighbour list.
  **Tests** (`tests/test_scattering.py`, 22): the X-ray weight is the atomic number and the
  neutron weight the tabulated value; an untabulated element, a disordered cell and an unknown
  source are refused; the descriptor route equals the definition and the difference falls as the
  channel narrows; the peaks of NaCl sit at 2.82 and 3.99 Å; a supercell gives the same curve;
  the two routes to F(Q) agree, F approaches zero at large Q and the sine transform round trip
  returns the curve (correlation above +0.99); every degradation of `measured_like` does what it
  says; a written file reads back to 1e-05 with its header, and odd files (comments, blank lines,
  three columns) read correctly; curves are usable as samples in a kernel and the cross kernel
  picks the structure a curve came from.
  **Measurement** (`benchmarks/run_scattering.py`; `benchmarks/results/scattering/table.md`).
  The three Matbench tasks of M6, 600 structures of each, the same five published folds, the
  curve range and the ridge chosen inside every training fold. *The identity:* 8.2e-05 of the
  curve at a 0.05 Å channel (median; worst 3.2e-04) and 5.5e-06 at 0.02 Å. *What the typing is
  worth:* the typed descriptor R² +0.771 on the bulk modulus, +0.684 on the shear modulus and
  +0.418 on the perovskite formation energies, against +0.303, +0.192 and +0.012 for the X-ray
  curve -- a difference of -0.469 [-0.560, -0.389], -0.492 [-0.579, -0.414] and -0.406 [-0.498,
  -0.311] from 2,000 bootstrap resamples. The neutron curve, on the 78 % and 96.5 % of structures
  its table covers and against a typed baseline on the same structures, is worse still (+0.093,
  +0.086, +0.101), and the likely reason is that the atomic number orders the elements while the
  neutron scattering lengths of neighbours are irregular and change sign. All 30 folds chose the
  10 Å curve over the 20 Å one. *What an instrument does:* the curve as computed predicts nothing
  (+0.003, -0.050, -0.001), cutting the transform at Q_max = 20 1/A raises it to +0.327, +0.235
  and +0.030 and at 10 1/A to +0.432, +0.344 and +0.050, because a kernel over curves 0.05 Å wide
  cannot see two structures as related; 10 % counting noise costs most of that gain (+0.150,
  +0.040) and a 10 % error in the number density costs nothing measurable (+0.336). *Transfer:*
  a model fitted on computed curves and asked about degraded ones never reaches +0.06 over the 36
  settings and is negative in 15, so the computed curves must be degraded to the instrument's
  resolution before fitting. *The reader:* 25 curves written, degraded and read back all match
  their own structure, kernel 0.891 against their own and 0.001 against the others. A concept
  stage (`--stage concept`) draws one cell of BaTiO3 as its 15 typed channel blocks, as the one
  curve a detector can measure, and as an instrument would report it.
  **What is open.** No experimental data, and that is the item's own gate: there is no public set
  of measured curves with measured properties to fit, so the next step is one such set, from a
  collaborator or from a repository that holds both. A partial structure factor per pair of
  elements, which a neutron experiment with isotope substitution does measure, would sit between
  the total curve and the typed descriptor and is not built. The X-ray weight is the atomic
  number rather than the form factor, which is right at low Q and too strong at high Q. A
  disordered cell is refused. And the one thing the measurement asks for is a fitting mode that
  computes the training curves at a stated resolution, so that a measured curve and the computed
  ones are the same object; `measured_like` can do it today but no subcommand does it.
* **M13. The application.** None of this is visible in the GUI: no cell viewer, no co-pattern
  plot, no site colouring. It belongs to G, and it is what makes the materials side usable by
  someone other than the author. It waits for M4 and M5, which decide what there is to draw.
  **Done 6 November 2026.** A window of its own, `cmfqsar/gui/crystal_window.py`, opened from
  *File -> Crystals and other periodic structures...* of the main window; a window and not a
  mode, because a crystal has a lattice instead of a pose, no bond table, and a kernel that sums
  over the images of the cell. (i) *The sample:* `periodic.read_crystal_set` reads one structure
  file or a folder of them (one structure per file, named after the file, which is how a database
  download arrives), with the periodic directions settable for the whole set and a layer gathered
  about its cell; the measured values come from a CSV whose first column is the names, and a
  table of several properties asks which column (`dataio.read_activities_csv(column=...)`,
  `--target-column` / `--activity-column` on the command line). `build-crystals` reads a folder
  the same way. (ii) *The cell viewer:* `Viewer3D.add_cell` draws the twelve edges of the box
  with the lattice vectors in red, green and blue; `add_atoms` draws the atoms of the cell as
  spheres coloured by element, the organic elements as the molecular view draws them and every
  other element in its Jmol colour; `add_points` draws the periodic images as dots out to the
  radius on the control strip, which is set to the fitted kernel's own cutoff
  (`periodic.cell_image_indices`, the expansion the overlap kernel itself uses, refactored out of
  `cell_images`). (iii) *The reading:* the co-field of the overlap kernel evaluated on a grid
  over the box of the selected cell, cut to the atoms along a direction with vacuum, drawn as the
  two isosurfaces of the main window; one sphere per site coloured by its contribution -- the
  atom contributions of rule 3 for the overlap kernel, `sites.site_contributions` of M5 for the
  autocorrelation -- with the scale on the control strip stating the value of the largest
  sphere, and the legend saying whether the numbers sum to the prediction; the Reading tab with
  the co-pattern over the channels and the sites of the selected cell. The autocorrelation has
  no co-field and the window says why instead of offering it. (iv) *`cmf predict` on a crystal:*
  the command decides the path from the model (`pbc_overlap` or `kernel_source="periodic"`), reads
  the structures as `build-crystals` does, refuses the alignment options for a crystal and a
  site model (M5) as a crystal model, and runs `contract.check_crystals`: no lattice is a
  refusal; an element without an indicator field is a warning where `cif:other` collects it and a
  refusal where nothing does; an element the training cells never held, electrostatic weights
  from another source, a cell off the training lattice by more than the set's own spread or
  10 % (overlap kernel), and partial occupancy on one side only are warnings. The tail of
  `predict` is one function for both paths.
  **Measured** (`benchmarks/run_cell_view.py`, `benchmarks/results/cell_view/table.md`; the window
  driven without a screen, as its tests drive it). *What a cell costs to show:* the 256-atom
  garnet cell of the COD set has 6,581 image atoms within 14.9 A, expanded in 6 ms and drawn in
  11 ms; the 112- to 128-atom electrolyte supercells draw in 10 to 13 ms; a 127-atom single-atom
  sheet, periodic in two directions, has 836 images and draws in 3 ms; the thirteen CIFs read in
  0.5 s. *Whether the reading is exact:* the site split of the autocorrelation sums to the
  prediction to 6e-14; the atom split of the overlap kernel on twenty relaxed sheets (lattice
  spread 0.4 %) misses the prediction by 7e-3 on a property of spread 0.34, because the periodic
  integral has two orders that agree only on exactly one lattice, and the window reports the
  difference beside the colouring whenever it exceeds a tenth of a per cent. The co-field inside
  a 127-atom cell takes 1.4 s on 30,744 points at 0.5 A, and the images beyond the kernel's
  10.5 A cutoff change it by 8e-15 of its peak, so the image sum is complete. On the sheets with
  the metal's height as the property, the largest per-site contribution of the first cell is the
  metal itself (-0.39 of a scale of 0.39), which is the check the colouring passes on its own.
  Tests: `tests/test_crystal_window.py` (20, offscreen Qt, `background=False` runs every step in
  the calling thread), `tests/test_crystal_predict.py` (11), `tests/test_cell_view_runs.py` (4).
  **What is open.** One cell at a time; the two whole-cell kernels only (no site-resolved model,
  no difference fields of a defect series, no three-body or reciprocal channels in the window);
  no project file for a crystal session; the images are dots, not spheres; and the main window
  and this one share nothing, so a slab behind molecules (EC10) is still its own item.

* **M14. A vacancy series on one host lattice: the data the periodic overlap kernel needs**
  (added 31 October 2026, out of what M4 measured). M4 built the kernel and measured it on
  `matbench_perovskites`, where it does not beat the periodic autocorrelation, and the
  measurement says why: that set is 18,928 *different compounds*, so it has no frame, and the
  rescaling that gives it one takes the lattice constant -- a strong predictor of a formation
  energy -- out of the kernel. The kernel is for a series where the frame is free, and the
  repository has no such series with enough members. This item is that data set, and it is the
  difference between a section of paper 14 and a paper of its own. (It became that paper on
  6 November 2026: paper 21 of `docs/publication_plan.md`, measured on the public 2DMD series.)
  **What the series is.** One host, one supercell, one lattice: the 2x2x2 cell of beta-Li3PO4
  that the Li3PO4/Li3BP2O8/LiBO2 interface study already relaxed (128 atoms, 48 lithium sites).
  A trivalent cation on a lithium site -- Al3+ or Ga3+ for 3 Li+, the substitution LISICON and
  LLZO chemistry both use -- leaves two lithium vacancies per dopant, and the configurations of
  those two vacancies around the dopant are the series: the same lattice, the same composition,
  a different arrangement. That is a congeneric series in the crystal sense, which is exactly
  what the overlap kernel asks for and what the molecular side of this program has had all
  along. The target is the total energy of the configuration, or equivalently the binding energy
  of the vacancy pair to the dopant, which is the quantity the conductivity literature argues
  about.
  **How many cells, and what they cost.** Enumerate the symmetry-distinct (dopant, vacancy,
  vacancy) configurations within about 6 A of the dopant with `pymatgen`'s enumeration, which
  gives of the order of thirty to forty after symmetry reduction -- enough for the interval of
  rule 4 and for leave-one-shell-out. The cost is measured, not estimated: that study's own
  vacancy relaxations in cells of this size took **1 h 47 to 3 h 25 each** at 8 MPI ranks on this
  desktop, about an hour each on a 32-core node, under 3 GB of memory, fixed cell, 24 to 95 ionic
  steps (`work/claude/li-graded-interface/README.md`, section 6). Thirty-five cells is therefore
  **two nights of this workstation with two concurrent 8-rank runners**, or a day of one node.
  Nothing in it needs a cluster.
  **Look for it before running it.** A public DFT-labelled defect set on one host lattice would
  cost nothing and the plan has already noted that it could not find one for the barriers (BT7's
  "no DFT-labelled public set" for the barriers). Check the Materials Project defect database and the published
  LISICON and LLZO dopant studies first; run the enumeration only if none of them gives thirty
  cells on one lattice.
  **Needed:** a script under `examples/` that builds the cells from the relaxed host, as rule 6
  requires, so that the series travels with the repository and not with another project; the
  VASP runner of `cmfqsar/vasp.py`, which the hop model already uses; and the loader that turns
  the finished cells into a `crystal_table`.
  **Gate, four parts.** (a) The periodic overlap kernel beats the periodic autocorrelation on
  these cells by more than the interval of rule 4 -- the half of M4's gate that the perovskites
  could not test, now on a set with a frame. (b) The co-field inside the cell names the oxygens
  that coordinate the vacancy and the lithium neighbours, which is M4's other half and the
  reading the grant paragraph wants. (c) The per-site contributions of rule 3 put their largest
  magnitudes on the sites that coordinate the vacancy, which is a check against the chemistry
  and not against the model. (d) Leave-one-distance-shell-out: train on the configurations with
  the vacancies near the dopant and predict the far ones, which is the extrapolation a dopant
  study actually makes.
  **Risks.** The configurations may be too similar to separate -- thirty-five cells of one
  composition differing by where two vacancies sit is a narrow series, and a narrow series is
  where a kernel method is strongest and a graph network weakest, which is the argument for the
  paper but also the way it could fail. The relaxation moves the atoms, so the frame is only as
  fixed as the relaxation leaves it; `periodic.overlap_asymmetry` and `lattice_spread` measure
  that and both must be reported. And a referee will ask for the comparison with a graph network
  on the same cells, which is a day of someone else's software and should be in the plan.
  **Done 1 November 2026, on a public set and without a VASP run.** The search the item asked
  for found one: 2DMD (Huang et al., npj 2D Mater. Appl. 2023; Kazeev et al., npj Comput.
  Mater. 2023), one to three point defects -- vacancies and substitutions -- on the 8 x 8
  supercell of a MoS2 or a WSe2 monolayer, 5,933 cells per material with the DFT formation
  energy of the cell, all on the ideal unrelaxed supercell, so every cell stands on one lattice
  exactly. The beta-Li3PO4 enumeration was not run.
  **What was built** (`cmfqsar/periodic.py`, `cmfqsar/fastkernel.py`,
  `CMFModel(pbc_symmetry=...)`). (i) Difference fields: `defect_fields` writes a cell as the
  host plus a signed change on the changed sites (a vacancy is the host row with its weights
  negated, a substitution is that row and the new species on the same site). The kernel is
  bilinear, so the kernel of the differences is exact (8.7e-11 relative on real cells), and a
  model with a bias on the differences is the model on the whole cells; `defect_table` builds
  the series and `normalize=False` is allowed for it. (ii) The host symmetry: `host_symmetry`
  takes the space-group operations of the host from spglib (768 for the 8 x 8 supercell, 64
  translations times 12 point operations) and the kernel is averaged over them. The linear
  average turns out to be a count kernel: additive over the defects and blind to their
  arrangement, because the orbit mean of a field is the same for every arrangement of the same
  defects. The arrangement survives only when the Gaussian layer is averaged over the group
  (`symmetry_averaged_kernels`), which is the orbit mean in the layer's feature space and so
  positive semidefinite. (iii) A compiled loop (`fastkernel.defect_kernels`): every atom pair at
  its nearest image in fractional coordinates, the cutoff capped at half the cell, the mean over
  the operations of the linear kernels and of the layer -- 1.4 s per 100 cells against 153 s
  for the numpy reference, agreement to 1e-6, 125 s for 2,000 cells and 768 operations. The
  per-atom contributions of the model take the same loop (`_compiled_atom_block`): 30 s for 400
  cells, where the image-expanded path took three hours. (iv) `benchmarks/run_2dmd.py`: five
  random folds with the setting and the ridge chosen inside each training part, the paired
  interval of rule 4 against the periodic autocorrelation, the extrapolation from the cells with
  one or two defects to those with three, and the per-defect reading. The arrangement classes of
  the cells under the host group are labelled, and all 5,933 cells of each material are
  distinct, so every test cell is an arrangement the model has not seen.
  **Measured** on 2,000 random cells per material (all 131 with one or two defects, the rest
  with three), `benchmarks/results/2dmd/table.md`. MAE of the formation energy per defect site
  in meV, MoS2 / WSe2: fixed frame 60 / 57, counts 41 / 42, periodic autocorrelation 8.3 / 15.7,
  symmetry-averaged arrangement kernel 1.8 / 5.2. **Gate (a) passes:** against the
  autocorrelation the arrangement kernel is better by 6.5 [5.3, 7.7] meV per site on MoS2 and
  10.5 [9.1, 12.1] on WSe2 (paired bootstrap, 95 %), while the fixed frame and the counts are
  worse by 26 to 52 meV; the R2 of both is above 0.999 and does not separate them. Published on
  this set, on the authors' own split of all 5,933 cells (not these folds): sparse graph network
  2.6 / 4.8 meV, gradient boosting on descriptors 11.8 / 14.7, whole-structure graph networks 31
  to 55. **Gate (d) passes on MoS2 and half on WSe2:** trained on the 131 cells with one or two
  defects and predicting the 1,869 with three, the arrangement kernel gives 4.3 meV per site
  (R2 1.000) on MoS2 and 54 meV (R2 0.996) on WSe2, where the additive count kernel gives 38;
  the autocorrelation does not extrapolate (R2 -0.02 and +0.06, 1.1 to 1.2 eV per site) and
  the fixed frame reaches R2 0.8. **Gates (b) and (c), the reading:** the per-defect
  contributions of the symmetry-averaged linear model (rule 3, 400 cells) are one number per
  kind of defect to four decimals, as the symmetry requires, and they are the chemistry the data
  hold. MoS2: a sulfur vacancy 2.59 eV, a molybdenum vacancy 6.98, selenium for sulfur 0.31,
  tungsten for molybdenum 0.19, against the four single-defect cells of the set at 2.65, 7.12,
  0.28 and 0.17 eV. WSe2: a selenium vacancy 2.62, a tungsten vacancy 5.09, sulfur for selenium
  -0.41, molybdenum for tungsten -0.42, against 2.70, 5.30, -0.38 and -0.41. The co-field was
  not drawn: the orbit-averaged co-field of a host lattice is one value per site type and says
  which species, not where; the per-defect contribution is the reading this series has.
  **Measured on all 5,933 cells** (the same folds and protocol, the reduced grid of attenuation
  1 and 2, bandwidth 0.125 and 0.25, width 0.35 and 0.75 A; 10 h on this machine, 8.9 GB at the
  peak): fixed frame 42 / 42, counts 41 / 41, periodic autocorrelation 3.0 / 6.0, arrangement
  kernel 1.9 / 4.4 meV per site. With three times the training cells the autocorrelation gains
  most, and the lead of the arrangement kernel shrinks to 1.0 [0.8, 1.2] meV per site on MoS2
  and 1.5 [1.3, 1.9] on WSe2. The intervals still exclude zero, so gate (a) holds on the full
  set, but by a small margin instead of a factor of three to four. The fixed frame reaches the
  counts (42 meV) where it stood at 60 on 2,000 cells: with enough cells it learns the orbit
  from the data. Trained on the same 131 cells and predicting the 5,802 with three defects, the
  arrangement kernel gives 4.7 meV per site on MoS2 and 37.7 on WSe2 against 38.3 for the
  counts, so on WSe2 it is level with the additive model rather than behind it; the
  autocorrelation stays at R2 near zero (1.1 to 1.3 eV per site) and the fixed frame at R2 0.6.
  **Not done:** the full grid on all 5,933 cells (the full-set run searched two values of each
  setting), the comparison with a graph network under these folds, a Shapley
  reading of the arrangement model (its layer is not linear, so rule 3 does not apply), the
  correlated-occupancy case, and the beta-Li3PO4 series itself, which this set makes
  unnecessary for the gate but which remains the series the conductivity literature argues
  about.

* **M15. What the invariant kernel's treatment of the features costs: the row length it divides
  out, and the columns it weighs alike** (added 6 November 2026, out of what BT7's external check
  measured). `cmfqsar.invariant.autocorrelation_kernel` rescales the
  channels and then divides every feature row by its own length, so a kernel entry is the cosine
  of the angle between two rows and the length of a row never reaches the model. Every kernel
  built on that function inherits this: the local environment kernel, the count kernel, the site
  profiles of M5, the structure factors of M6. The external check of item BT7 measured what it
  costs on one task. Fitting a forest on exactly the rows the kernel is built from gives q2 +0.758
  on the published hop barriers; handing the same forest the raw rows, so that it can also see how
  large the features are, gives +0.844 and takes the mean absolute error from 0.488 to 0.374 eV.
  The length carried about a third of what the fitting method and the length together added. The
  same mechanism is already recorded as a caution for particles further down this plan -- the
  per-atom normalisation removes the particle size, so size must enter as a separate descriptor
  (NT3) -- which is this question in another setting.
  **What the item is.** Measure, on the benchmarks that already exist rather than on new data,
  what the row length is worth where that kernel is used: the defect series of M14, the site
  profiles of M5, the hop environments of BT7, one molecular set. The row length is one number per
  sample, so the cheapest form is the kernel as it stands plus that number as an extra channel, or
  the kernel without the division where the samples are comparable in size (`normalize=False` is
  already allowed for the defect differences of M14). Rule 1 and rule 4 decide it as usual, and
  rule 6 means the answer arrives as a table.
  **Why it is not simply switched off.** The normalisation is what makes the kernel comparable
  between samples of different sizes, which is why it is there; a molecular set whose members
  differ in atom count is the case it was written for. So the question is not whether to divide but
  **when**, and the answer has to be measured per kind of data, not decided once. Until it is
  measured, nothing changes: every stored table in this repository was produced with the division
  in place.
  **The second half of the item: which columns count** (added the same day, out of the Gaussian
  arms of BT7's check). The row length is not the whole of what the kernel gives away. On the hop
  barriers a forest beats the kernel by +0.171 in q2 on the kernel's own rows, and a Gaussian
  kernel on those rows recovers only about 30 per cent of that (+0.052 [+0.023, +0.080]), so the
  rest is not a curved response to the features. The reading, which this item has to test rather
  than assume, is that the distance itself is wrong: one Euclidean distance over 2601 columns
  dilutes the few columns that carry the barrier among the many that do not, while a tree reads
  columns one at a time. The test is a weighted distance -- per channel, or per channel and
  distance bin -- chosen inside the training part, against the same kernel with equal weights,
  with rule 4's interval deciding it.
  **What is already known about weighting, so that it is not measured twice.** The pooled mixing
  pilot of 1 November 2026 learned field weights over 38 molecular sets and did **not** improve
  prediction over equal weights; it repaired per-series fitting instead. So weight learning as a
  route to accuracy has failed once already, on molecular sets with few channels and a target the
  equal-weight kernel already fitted. This item is a different case in three ways and must say so
  when it reports: 2601 columns rather than a handful of fields, a solid-state target, and a
  measured gap to another learner on the identical rows that the earlier pilot never had. If it
  fails here too, then the answer to "why does a forest win" is neither the length, nor the
  curvature, nor the weights, and the item should say that plainly rather than look for a fourth
  repair.
  **The first half is measured, 7 November 2026: the length is worth keeping where it goes with
  the target, and nowhere else.** `cmfqsar.invariant.row_lengths` returns the number the kernel
  divides out and `length_kernel` makes a kernel of it; `cmfqsar.sites.site_kernel` now takes
  `normalize` as well. `benchmarks/run_rowlength.py` fits four arms -- the length divided out
  (the reference), the length kept (`normalize=False`, the matrix divided by its own mean
  diagonal, which takes no row's length away), the length added as one more channel with the
  mixing weight chosen inside the training part, and the length alone -- over five series, each
  on the samples, folds and ridge grid of the benchmark it comes from.
  *Where it gains.* On the hop environments of item BT7 (1,801 hops of 393 structures) the
  length correlates with the barrier at +0.601 and keeping it takes $q^2$ from **+0.572 to
  +0.716**, a gain of **+0.144 [+0.096, +0.198]**, with the mean absolute error falling from
  0.736 to 0.546 eV; the length alone reaches +0.354. So how crowded the environment is carries
  a third of the barrier, and this is where most of the gap item BT7 measured against a forest
  on the raw rows was hiding. On the defect series of item M14 (2,000 cells each of MoS2 and
  WSe2) the length correlates with the formation energy at -0.882 and -0.925, $q^2$ is already
  +0.9989 and moves by +0.0011 [+0.0009, +0.0012], and the error per defect site falls from
  **36.5 to 2.4 meV** on MoS2 and from 30.8 to 6.3 meV on WSe2.
  *Where it costs.* On the 1,162 beta-cyclodextrin guests, 2 to 42 heavy atoms, the length is 66
  per cent of its own mean and tracks the heavy-atom count at +0.750, but tracks the constant at
  only +0.112: keeping it loses **-0.043 [-0.073, -0.015]**. On the single-atom catalysts of item
  EC8, twelve combinations of four targets and three designs, keeping it is behind the reference
  beyond the interval in three and ahead in none. On the Li site energies of the three
  electrolytes the length varies by 5 to 13 per cent and keeping it loses, although with three
  compounds rule 4 can give no interval, so that series carries point differences only.
  *The rule that comes out of it, and it is cheap.* What decides is the correlation between the
  length and the target, not the spread of the length: the counts about a hop midpoint vary by 81
  per cent of their mean and keeping that length still buys nothing (+0.017 [-0.030, +0.055]).
  One correlation on the training part says which way a new series will go. Keeping the length is
  the stronger of the two ways where it works at all -- as an extra channel the hop environments
  gained +0.070 [+0.027, +0.114] of the +0.144. No default changes: every stored table was
  produced with the division in place, and the manual says so under *What the row length is
  worth, and when the division helps*.
  *Found on the way, and not part of this item.* The autocorrelation features do not depend on
  the field width alpha at all, because they are sums of products of atom weights and those
  weights do not carry alpha. `benchmarks/run_cyclodextrin.py` searches six alphas for its
  invariant kernel and therefore searches six copies of one kernel: its stored $q^2$ values are
  unaffected, but the alpha it records for that column means nothing and its sentence about the
  alpha grid does not hold for that one column.
  **What is left of the item** is the second half above: the weighted distance over the columns.

**Order.** M4, M5 and M6 first: together they give the most results per line of code and supply the theory
section and two of the case studies of the publication plan. M1 and M3 are done as they are needed
by those three. M9 after M6 has reported; ~~M10~~ and ~~M11~~ **done 2 November 2026**, and what they now need is data and not more code -- a set of chiral crystals with a measured property for M10, and a design space whose candidates are not already computed for M11; ~~M12~~ **done 1 November 2026**, and what it now needs is a measured data set and not more code. M14 is the data M4 turned out to need, and
nothing else in this part waits for it. M15 came out of item BT7's external check on 6 November
2026 and needs no new data: it is a measurement on the benchmarks that already exist, and it
touches every kernel this part builds, so it comes before any further channel is added.

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
  **Two of the three are done.** The cell outline with its periodic images came with M13
  (6 November 2026). The bonds came on **7 November 2026**, asked for by the author, because a
  crystal view that draws spheres alone cannot show an adsorbed organic molecule, which is the
  case items EC2 and EC6 are about: `cmfqsar/bonds.py` perceives a bond list from the distances
  over the cell *and* its periodic images (so a bond that crosses the cell boundary is drawn
  rather than lost), the crystal window offers *ball and stick* and *sticks* beside *balls* and
  *dots*, and **a metal atom gets no bonds**, which is this item's own rule: a distance rule
  gives every atom of a close-packed metal about twelve neighbours, so a slab stays spheres while
  the molecule on it gets its bonds. What a metal gets instead is
  `bonds.coordination_polyhedra`, the first coordination shell as a polyhedron, which is how an
  inorganic structure is read. The perception is display-only: no bond order, no sanitisation and
  no charge, which is M8's work and not this.
  **Measured** (`benchmarks/run_surface_view.py`, the `bonds` stage): the molecules of the two
  Fe(110) sets came from SDF files, which carry a bond table, so the rule has a ground truth on
  138 real molecules with their hydrogens. In the open it returns the molecule's own bonds for 72
  of 73 and 63 of 65; placed on the Fe(110) slab in its 16.68 x 11.79 A cell and wrapped so the
  box cuts the molecule in two, for **65 of the 65 and 57 of the 58 molecules that fit inside
  that cell**. The ones it gets wrong are almost exactly the molecules longer than a cell edge,
  which touch their own periodic image, so two atoms far apart in the molecule become neighbours
  across the boundary -- a property of the cell and not of the rule, and the one caution to carry
  into a picture. No bond ever reaches the slab, which the stage asserts. One molecule costs
  0.1 ms in the open and about 1 ms in a cell; the polyhedra of a 48-atom COD cell take 19 ms,
  and rock salt comes out 6-coordinate as it must.
  **What is left of the item** is the first clause: instanced drawing for ten thousand atoms. The
  images are still drawn as points and the spheres are still one mesh per call, so a particle of
  that size is the open part.
* **NT9. The co-pattern as a plot.** The invariant kernels have no co-field. `top_channels` returns
  the ranked channels; the Results panel needs a chart of weight against distance for the chosen
  pair of field types, and the viewer needs site colouring by contribution (NT7). Three days.
  **Done 6 November 2026.** `cmfqsar/copattern.py` turns a fitted model into one curve per pair
  of field types: `copattern` returns the distance grid with the weight and the weight times the
  spread, `copattern_rows` the same numbers as a table, `has_copattern` whether a model has one.
  `cmfqsar/gui/copattern_panel.py` draws it, with boxes for the pair and for which of the two
  quantities is on the y axis, in the new *Co-pattern* tab of the main window's Results panel and
  beside the ranked table in the crystal window's *Reading* tab. The axis is whatever the kernel's
  channels are indexed by and the label says which: the distance between two atoms
  (molecular and periodic autocorrelation), the in-plane distance (surface kernel), the reciprocal
  length q (M6). A kernel that resolves a distance **further** -- the surface kernel resolves it by
  the heights of both atoms -- has a point that is the sum of the weights of the channels at that
  distance, and the note under the chart says how many were summed; the sum is the right summary
  because a prediction is a sum over channels, and the table beside it still names the individual
  channels. Two kernels are refused with a reason instead of being drawn wrongly: the three-body
  channels of M10 (two radii and a separation, not one distance) and the overlap kernel of the
  fields, which has a co-field. The site colouring the entry also asks for was built by M13
  (`crystal_window._draw_sites` for a cell, the atom contributions of rule 3 for a molecule), so
  nothing of NT7 was needed.
  **Measured** (`benchmarks/run_surface_view.py`, the `copattern` stage): on two kernels that have
  no co-field, the identity that makes the weights a reading -- the bias plus the sum over every
  channel of weight times channel is the prediction -- holds to 8e-14 on the periodic
  autocorrelation of the thirteen COD electrolytes (target spread 16.3) and to 2e-12 on the
  surface autocorrelation of the 73 Fe(110) adsorbates (spread 63.8 kJ/mol); both are rounding.
  The leading pair is the electrostatic field with itself, 26 % of the reading, peaking at 8.0 A
  on the conductors, and the electrostatic with the hydrophobic field, 28 %, peaking at 2.25 A in
  the plane on the adsorbates. One curve costs 0.1 to 1.1 ms, so the chart follows the box with no
  wait, even where a pair block is 612 channels.
  **Not done:** one pair at a time and no overlay of two pairs; a chirality block is left out of
  the curve rather than given an axis of its own.
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
  **Done 5 November 2026. The machinery works and the gate is not passed: no quantum field
  improves on the five ordinary fields on this series, and the measurement says why.**
  **What was built.** All five pieces the item lists. `state_from_properties` reads `charge` and
  `multiplicity` from the SD properties (also capitalised, also `total_charge` /
  `spin_multiplicity`), refusing a multiplicity below 1 or of the wrong parity for the electron
  count rather than adjusting it, because the likely cause is a mistake in the file;
  `ion_multiplicity` gives the two ionic states their own spin states, which is what makes a
  Fukui index of an open-shell parent possible at all. `solvent_lines` takes either a name ORCA
  knows (`CPCM(water)`) or one of `SOLVENT_CONSTANTS` or an (ε, n) pair, which becomes a `%cpcm`
  block; `Config.solvent` passes it to every input of a molecule, and `qm_solvent` records it in
  the file so a cached result cannot be mistaken for a gas-phase one. `HEAVY_LEVEL` names
  `def2-ECP` explicitly (the threshold is rubidium, Z = 37, which `needs_ecp` reports). The
  Mulliken spin populations are parsed from the fourth column of the open-shell charge block into
  `prop:spin`, `prop:spin_anion` and `prop:spin_cation`, with `spin_total*` beside them. And
  `cdft.py` reads the potential cube: `orca_plot` type 43, which unlike the density and spin plots
  asks for the saved density **by name** rather than offering one, and writes
  `<job>.scfp.esp.cube`, whose header is identical to the density cube's, so `read_cube` and
  `same_grid_as` needed no change. Established by experiment on the running program, not from the
  manual.
  **The cache was protected.** `cache_key` folds the solvent and the multiplicity in only when
  they are given, so a gas-phase lowest-spin key is byte-identical to the one it had, and the ORCA
  results the other benchmarks have already paid for are not thrown away. A test asserts it.
  **The data set.** 69 substituted phenazines with a computed one-electron reduction potential
  (`examples/phenazine/`, `examples/make_phenazine_examples.py`), −2.092 to −0.160 V against a
  training mean error of 0.285 V, superimposed on the core to 0.08 ± 0.07 Å. Three faults in the
  source geometries were repaired and are reported by the recipe; one molecule was dropped by name
  and reason. The quantum stage cost **10.73 hours** for 69 × 3 single points at
  `B3LYP D3BJ def2-SVP` with CPCM at the constants of dimethoxyethane, 0 failures.
  **The spin populations are correct**, which is a check and not a result: they sum to
  1.0000 ± 0.0000 as a doublet must, 34.5 ± 5.9 % of the unpaired electron sits on the two core
  nitrogens, and on average 4 % of it leaves the core.
  **What was measured** (`benchmarks/run_phenazine.py`,
  `benchmarks/results/phenazine/table.md`): eight descriptions under five random folds and under
  the three source tables held out one at a time, each choosing its ridge and field width inside
  each fold, compared by paired bootstrap intervals over the per-molecule absolute errors. Mean
  absolute error in volts on random folds: the electron affinity alone 0.194, nine global quantum
  numbers 0.181, the five ordinary fields 0.158, with the Hirshfeld charge field 0.153, with the
  Fukui fields 0.151, with the spin field 0.161, with all of them 0.151, the spin field alone
  0.290. **Every quantum-against-ordinary interval holds zero**: Hirshfeld +0.006
  [−0.016, +0.028], Fukui +0.007 [−0.012, +0.029], all together +0.007 [−0.015, +0.031], spin
  −0.003 [−0.008, +0.002]. With a source table held out, nothing separates at all and every model
  sits near 0.23 V against a mean of 0.285.
  **Why, measured rather than argued.** The kernel is a normalised overlap, so what matters is how
  much a field varies across the series. Through the spin field alone two molecules of this series
  are **0.910 ± 0.127** alike, and through f⁺ **0.953 ± 0.032**, against 0.315 for the
  electrostatic field, 0.344 for the Hirshfeld charge field and 0.308 for the dual descriptor. The
  added electron always goes to the same place, so the field that says where it went is nearly a
  constant and cannot separate two molecules by what they share. The co-field says the same from
  the other side: the Hirshfeld charge field carries twice the weight of anything else (+0.201
  against +0.122) and its strongest point sits 2.87 Å from the nearest core nitrogen, among the
  substituents.
  **The circularity, and the control that handles it.** The target is computed and comes from the
  same kind of calculation as the spin field, so a good score would not have been evidence about
  an experiment. The electron affinity alone — one number, no geometry, correlated +0.77 with the
  target — is what separates "the field predicts the potential" from "the field recovers the
  global number". It is a strict control and no field beats it with an interval excluding zero
  (the best, all the quantum fields together, gains +0.043 [−0.003, +0.090]).
  **What this result is worth.** It names the series the spin field is *for*: one whose members
  reduce at **different places** — competing redox centres, or an electron that moves between the
  ring and a substituent — rather than one fixed core decorated differently. A phenazine series is
  the opposite case. The field was cheap once the quantum stage could produce it, and a negative
  result with a measured explanation tells the next person which series not to try.
  **Not done:** the potential cube is read but not measured against the CHELPG field, which the
  item's own text says should stay the first choice; no dialog in the application for the solvent
  or the spin state (the ORCA dialog has neither); the `prop:spin` field of an open-shell *parent*
  is implemented and tested but never exercised on a real radical series; and no quinone series,
  which is the other data set the item names and the one where the two redox centres differ.
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
  **Done 5 November 2026. The machinery works and the gate is not passed, and not narrowly:
  reading a few places loses to reading everywhere on every target measured.**
  *Built:* `field_at_points`, `field_gradient_at_points`, `coulomb_at_points`, `anchor_names`,
  `anchored_row` and `anchored_matrix` in `cmfqsar/descriptors.py`, `tests/test_anchors.py`
  (17 tests), `benchmarks/run_anchors.py` with `benchmarks/results/anchors/table.md`, and the
  manual section *Reading a few places instead of all of them*. The reading is exact — every atom
  contributes, unlike `evaluate_gaussians`, which drops Gaussians below 1e-5 of their peak and is
  the routine that draws a field on a grid. The gradient is checked against a finite difference of
  the value at every anchored point of every field, on real molecules, inside the benchmark as
  well as in the tests: it agrees to 1e-11.
  *Two sources, because the item's "physical unit" claim needs one.* The Gaussian field of a field
  set is not a Coulomb potential, so the gradient of the Gaussian electrostatic field is
  proportional to an electric field near the atoms and is not one in volts per angstrom.
  `source="coulomb"` reads the true point-charge potential and its gradient from the same charges,
  in e/angstrom and e/angstrom^2. The distance-dependent dielectric of
  `cmfqsar.complementarity.coulomb_potential` is deliberately not offered, because its gradient is
  not the physical field.
  *The places are given and not chosen.* For the Hammett series the frame is a rule, so the ring
  positions are exact: the ipso carbon, the centroid, 1.7 angstrom above the face, and 1.5 angstrom
  beyond the meta and para carbons, which is where the reacting group of the sigma definition sits.
  For the nickel complexes of EC4 the metal is at the origin by the alignment, and the two
  substituent points are where EC5's vector co-field put its strongest arrows.
  *The gate.* Mean absolute error against the five-field model: sigma_m 0.132 against 0.097,
  a difference of -0.035 [-0.066, -0.008]; sigma_p 0.214 against 0.154, -0.060 [-0.119, -0.015];
  pi 0.784 against 0.313, -0.473 [-0.637, -0.327]; nickel E_red 0.044 V against 0.029 V,
  -0.015 V [-0.023, -0.006]; nickel E_ox 0.280 V against 0.150 V, -0.130 V [-0.418, +0.034].
  Every interval but the last excludes zero, and the last excludes it under neither design. Added
  to the five fields the block ties on sigma_m (-0.0029 [-0.0190, +0.0127]) and loses elsewhere.
  The reaction-centre points on their own are the worst model in the table, with a negative R2 on
  four targets of five: at 2.9 angstrom from the ring centre the Gaussian field is small, and eight
  columns of small numbers overfit.
  *What survives.* The true Coulomb potential is the better anchored descriptor, by
  +0.0235 units on sigma_m [+0.0031, +0.0478], which is the item's physical argument confirmed in
  the one place it could show. And the field model's advantage is not that it has more numbers: the
  anchored block has 100 columns on 95 substituents and still loses.
  *The methodological result, which is the most useful part.* Item EC5's diagnostic -- ask, before
  fitting, whether a new description can differ from the one the model has -- **misled here**. The
  best single anchored column reaches |r| 0.85 with the nickel oxidation potential and 0.83 with
  sigma_p; the block then gives R2 -0.88 and +0.12 under nested cross-validation. The difference is
  selection: a correlation between two whole kernels is one number computed once, while the best of
  60 columns on 25 molecules is a maximum over 60 tries. A whole-kernel correlation is a fair
  diagnostic and a best-of-N column correlation is not, and this item is the measurement that says
  so.
  **Not done:** the catalyst use waits on data and not on code, since no series with a measured
  catalytic property was found in machine-readable form (EC4 records that search); the application
  has no way to place points in a frame, so the block is available to a script and not to the
  window; and the block cannot be named like the other computed descriptors, because it needs the
  field set and the points as well as the molecule.

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
  **Done 5 November 2026. The machinery works and the gate is passed, on a different pair of
  series from the two the item named, because no machine-readable table of either was found.**
  *Built:* `cmfqsar/complexes.py` (reading a complex from coordinates with dative metal–ligand
  bonds, the ligand charge assignment, the metal site as numbers, and two frames), van der Waals
  parameters for all 29 d-block metals in `cmfqsar/atomtypes.py`, a charge model that survives a
  metal and zero hydrophobic weight on it in `cmfqsar/fields.py`, `.xyz` support in
  `read_molecules` and in the application's file dialog, `tests/test_complexes.py` (36 tests),
  `examples/make_metal_examples.py`, `benchmarks/run_metal.py` with
  `benchmarks/results/metal/table.md`, and the manual section *A metal in the middle:
  coordination complexes*.
  *Three faults found, each silent.* The Gasteiger scheme has no parameters for a d-block metal
  and one metal atom makes RDKit return "not a number" for *every* atom, so the electrostatic
  field of a complex was identically zero: 197 of 200 iron complexes checked. A one-atom ligand —
  a chloride, bromide or iodide — is left neutral by bond perception whatever charge it is asked
  for, so the charge that belonged to the halide went to the metal and a quarter of the iron
  series had the right total charge and the wrong charge inside it. And the steric table stopped
  at the main group, so every metal had the default radius; the parameters now come from the
  universal force field, with its van der Waals distance halved and put on the scale of the
  Tripos radii by the ratio of the two carbon values (iron 1.285 Å against carbon 1.70 Å).
  *The data.* The item's own two series could not be used. The iron-porphyrin turnover
  frequencies are spread over separate papers with different conditions and analysis methods,
  which those papers themselves warn against pooling, and the complete hydricity table of the
  nickel diphosphines sits in a publisher's supporting-information file rather than in a data
  repository, and was not retrieved. Two public series
  with clear licences were used instead: 2216 iron(II) complexes from the structural database
  with a computed Fe(III)/Fe(II) reduction potential (MIT licence), which share the metal and
  nothing else — 377 different first coordination spheres — and 25 substituted triaqua
  nickel–bis(pyrazolyl)pyridine complexes with the computed potentials of two couples (CC BY 4.0),
  which share one ligand skeleton and one N3O3 sphere and carry the natural population analysis
  of the same calculation as their own control. Both targets are computed, not measured.
  *The gate.* On the nickel series the five ordinary fields in the shared frame reach a mean
  absolute error of 0.150 V on the Ni(III)/Ni(II) potential against 0.345 V for the population
  analysis of the calculation that produced the target and 0.370 V for the coordination geometry
  alone, with a training-mean error of 0.488 V; the paired differences are +0.195 V
  [+0.068, +0.321] and +0.220 V [+0.085, +0.355], and leave-one-out agrees. The co-field reads
  the second coordination sphere, as the item asked: its largest weights sit 4.5 to 7.3 Å from
  the nickel atom, among the substituents rather than at the metal or its donors. The other
  couple, Ni(II)/Ni(I), has a standard deviation of 0.06 V and little to predict: the fields
  reach 0.029 V against 0.043 V, which the random folds call a difference and leave-one-out does
  not.
  *The iron series adds two results of its own.* The frameless field autocorrelation reaches
  0.253 V against 0.313 V for the element counts with the metal site and a training-mean error of
  0.523 V, a difference of +0.060 V [+0.049, +0.071] that survives holding whole coordination
  spheres out. This is the first inorganic target in this repository where a field beats a
  composition; items BT1 to BT3 found the opposite on an average cell voltage, and the difference
  is what the target is. And the frame decides whether the overlap kernel works at all: in the
  frame each file happens to carry it scores 0.490 V, barely better than the training mean
  (R² = +0.083), and in the metal-and-principal-axes rule frame it scores 0.302 V (R² = +0.588),
  so the frame is worth +0.188 V [+0.173, +0.202]. That lifts the overlap kernel just past the
  composition (+0.012 V [+0.001, +0.022], and +0.022 V [+0.009, +0.033] with the coordination
  spheres held out) and leaves it behind the kernel that needs no frame
  (+0.049 V [+0.039, +0.058]). Measuring this on a random 800 of the complexes instead, which was
  tried first because the overlap kernel costs most of the stage's hour, made the rule frame tie
  the composition rather than pass it, so the subset answered a different question and every
  model is measured on all 2216.
  *What the two frames are worth.* Matching the first coordination spheres onto a reference
  complex, over every assignment that pairs donors of the same element, puts the nickel donors
  0.03 ± 0.04 Å apart. The principal-axes rule puts the same donors 1.82 Å from where the match
  puts them (worst 3.45 Å), so the rule is not a substitute for the match on a congeneric series;
  on the iron series it is nearly degenerate for 11 % of the complexes. The matched frame is also
  the one place the phenazine symmetry trap of EC1 reappears, and the search over assignments is
  there for it.
  **Not done:** the metal identity is a descriptor the reader provides and neither series
  exercises it, since one is all iron and the other all nickel; no series with a measured
  electrocatalytic property was found in machine-readable form, so the catalysis claim the item
  wanted is not made; the electrostatic field of a complex still uses the charges of the free
  ligands, which understates the donor atoms, and a calculated charge per atom is the first
  extension; and the spin state of EC1 is stored on a complex and never used, because both series
  are measured at one state each.
* **EC5. The electric-field kernel.** Every field in the program is a scalar; the quantity a
  reaction centre feels is the vector ∇φ. The integral of ∇φ_A·∇φ_B over space, for φ the sum of
  atom-centred Gaussians, is the second derivative of the overlap of `cmfqsar/kernels.py` with
  respect to the pair distance and stays in closed form, so rule 3 is kept. A field type `efield`
  with that kernel, a vector co-field (coefficients on the same atoms, drawn as arrows), and the
  Shapley decomposition unchanged. Measure on the Hammett sets and on the EC4 series against the
  scalar electrostatic kernel; the item is justified only where the vector kernel beats the
  scalar one by more than the interval, which the through-space catalysis literature says it
  should on the cationic porphyrins.
  **Done 5 November 2026. The machinery works, it is exact, and the gate is not passed: the
  vector kernel is a near-copy of the scalar one, and added to the five fields it costs accuracy
  rather than adding any.**
  *Built:* the field kind `efield` in `cmfqsar/fields.py` (same charges, different kernel, not one
  of the five defaults and asked for by name), `efield_pair_factor` and the kernel, gradient and
  self-kernel paths in `cmfqsar/kernels.py`, `evaluate_gaussian_gradients`, `cofield_gradient` and
  `cofield_arrows` in `cmfqsar/cofields.py`, `tests/test_efield.py` (26 tests),
  `benchmarks/run_efield.py` with `benchmarks/results/efield/table.md`, and the manual section
  *The field a reaction centre feels*.
  *The closed form.* The pair integral of two Gaussians of the shared width is the ordinary
  overlap times `3 alpha / 2 - alpha^2 d^2 / 4`. The item's own text calls this the second
  derivative of the overlap with respect to the pair distance; it is the negative *Laplacian* with
  respect to the separation of the two centres, which is not the same thing, and the formula here
  is the one that matches a grid integral — to 0.000 % on two real molecules with 2.7 million
  points, and exactly on single pairs in the unit tests. It is the only pair integral in the
  program that changes sign: zero at `d = sqrt(6/alpha)`, 2.45 angstrom at alpha = 1, and 72 % of
  the atom pairs of the Hammett series lie beyond it. The kernel matrix is still positive
  semidefinite, since it is an inner product of two vector fields, and that is tested rather than
  asserted.
  *The reading that predicted the result, used in advance this time.* The cosine-normalised vector
  and scalar kernels correlate **+0.980** over the 4371 pairs of the Hammett series. A description
  that orders a series almost exactly as one already in the model cannot change it much, which is
  the lesson item EC1 learned after the fact.
  *The gate.* Added to the five fields: sigma_m +0.0005 [-0.0037, +0.0051], sigma_p
  +0.0068 [-0.0045, +0.0190], nickel E_ox **-0.0138 V [-0.0342, -0.0005]** and
  -0.0136 V [-0.0307, -0.0017] under leave-one-out, so on the one series where it does anything
  measurable it makes the model worse, which is what a sixth nearly collinear field should do.
  Used *in place of* the electrostatic field it ties the five-field model everywhere (nickel E_ox
  +0.0019 V [-0.0031, +0.0077]), so it is a substitute at the same price. Alone against the scalar
  field alone it wins on the nickel oxidation potential under leave-one-out
  (+0.0299 V [+0.0144, +0.0459]) and not under random folds (+0.0057 V [-0.0072, +0.0194]); that
  is the only measurement consistent with the through-space argument, on the only cationic series
  available, and it is too design-dependent to carry the item. The apparent win on pi
  (+0.0620 [+0.0238, +0.1084], vector alone against scalar alone) is a difference between two bad
  models of a target that is not electrostatic: both score R2 below 0.2 where the five fields
  reach 0.86.
  *The arrows.* `cofield_arrows` returns the strongest arrows of the co-field gradient with a
  minimum separation. On the nickel oxidation potential they sit 5.6 to 6.6 angstrom from the
  metal, among the substituents where item EC4's scalar co-field also put its weight, and point
  inward towards the metal.
  **Kept and labelled under rule 5**, because it is exact, cheap, a working substitute for the
  electrostatic field, and the only field whose co-field is a direction.
  **Not done:** the compiled kernel loop (`cmfqsar/fastkernel.py`) does not implement the kind, so
  a field set containing it falls back to NumPy and the iron series of EC4 was not measured; the
  cationic porphyrins the item names were not available in machine-readable form (see EC4), so the
  through-space prediction is tested only on 25 nickel complexes; the arrows are returned as
  numbers and the viewer does not draw them; and the measurement uncovered a fault it did not fix
  — one of the 95 Hammett molecules (the selenomethyl substituent) has an electrostatic field that
  is identically zero, because the Gasteiger scheme has no parameters for selenium and EC4's
  repair covers the d-block metals only. A general repair would move the published Hammett tables
  and belongs to its own item.
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
  **Done 5 November 2026. The machinery works and the gate is not passed on either half: docking
  against the slab does not place the molecules, and the score that tracks the energy needs the
  pose it was supposed to find.**
  *Built:* `cmfqsar/slab.py` -- `bcc110_slab` and `read_slab`, `slab_receptor` (the top layers
  with their periodic images as a `MoleculeFields`, so the pocket machinery applies unchanged),
  `image_charge_energy`, `surface_transform` and `dock_on_surface` (height, tilt, rotation about
  the normal, translation in the plane, with the tilt bounded and the face a choice),
  `surface_sites` and `site_profile` (top, bridge and hollow sites from a Delaunay triangulation
  of the top layer, so any surface is covered), `surface_score`; `tests/test_slab.py` (18 tests),
  `benchmarks/run_slab.py` with `benchmarks/results/slab/table.md`, and the manual section *A
  metal surface as the receptor*.
  *What a clean metal offers, measured.* Four of the five physicochemical fields are identically
  zero on a one-element slab and not for want of parameters: no electronegativity differences, so
  every charge model gives zero, and the hydrophobic and hydrogen-bond increments are fitted to
  organic fragments (item EC4 made that explicit). All four complementary pairs therefore vanish
  and a complementarity score against a clean metal is a shape score. The image charge is what a
  metal has instead, in closed form, and two measured facts bound its use: it is invariant to
  translation in the plane and to rotation about the normal (4e-17 and 1e-17 eV on a real
  molecule), so it can set a height and a tilt and never a position; and it is -0.033 eV for
  benzimidazole at its relaxed pose against adsorption energies spanning 200 kJ/mol, so it is the
  right physics for physisorption of a polar molecule and two orders of magnitude too small for
  chemisorption on iron.
  *The pose half fails.* Median in-plane error against the relaxed pose, benzimidazole and
  quinoline: EC2's flat rule pose 2.46 and 3.45 angstrom; that pose docked against the slab 2.50
  and 3.66; docked with both faces tried 2.37 and 3.23 (the share within 1 angstrom goes from 0 %
  to 7 % and from 3 % to 5 %); docked with the image term on 2.45 and 3.78; from a conformer of
  the SMILES, docked, 2.74 and 3.73. Shape complementarity with the slab decides neither the face
  nor the in-plane placement, which is EC2's finding reached from the other side.
  *The energy half passes only when it is given the answer.* Leave-one-out q2 of a line on the
  burial at the relaxed pose: +0.638 and +0.624, against EC2's controls of +0.472 and +0.423 (the
  relaxed pose's height) and +0.345 and +0.328 (the chemisorbed flag), with the right sign
  (r = -0.81). Adding the image term gives +0.661 and +0.647. At a pose the program produces
  itself the same score is +0.115 and +0.049. So the burial in the relaxed frame is a *better*
  readout of the relaxation than the height is, which makes EC2's warning worse rather than
  better: a q2 of 0.64 looks like a model and is a measurement of the input.
  *A fourth stage settles it on measured targets.* The Fe(110) target came from the same
  relaxation as the pose, so that half could not separate a score from a readout of its input. The
  corrosion sets have no such circularity: 23 congeneric series, 997 molecules, with a measured
  potency in acid and 830 measured Langmuir adsorption free energies. Six slab terms go in as a
  descriptor block -- burial, the image energy, the height, and the share of the molecule over the
  top, bridge and hollow sites -- read at a pose the program produces itself (embedded from SMILES,
  laid flat by EC2's rule, refined against the slab), with the grids, folds and selection
  convention of `run_corrosion.py`. The agreement of the control is exact: atom-type counts come
  out at +0.189, the value already published for them in
  `benchmarks/results/corrosion/table.md`. Mean q2 over the series, measured potency then
  measured dG_ads: counts +0.189 and +0.126, the five fields in the rule frame +0.279 and +0.201,
  the six slab terms +0.032 and -0.057, the two together +0.190 and +0.080. The slab terms beat
  the no-geometry control in 7 of 23 series and 3 of 18, and add to the five fields in 5 and 1, so
  adding them *lowers* the field model. One 16-molecule series looks like an exception (the slab
  terms alone +0.544, `over_top` at |r| 0.74) and with 23 series tried that is what selection
  produces -- item EC3's lesson applies. Two smaller findings: the measured dG_ads is harder than
  the measured potency for every model including counts, although it is nominally the adsorption
  quantity; and the site profile, which this item had computed and not used, is the term that most
  often carries what little signal there is.
  **Not done:** no surface that offers more than shape was measured -- an oxide or a passivated
  metal, where the electrostatic and hydrogen-bond fields are not zero, and physisorption rather
  than chemisorption, where the image term is the right size, are where this machinery should
  earn its place and neither has a measured series here. That is now the only open question in
  the item: the machinery has been measured against everything the repository holds, computed and
  measured. The slab is also rigid and its own relaxation is not modelled, and M7, the same item
  for a pore, is untouched.
* **EC7. A cation probe.** The probe module (`cmfqsar/probes.py`) carries water, methyl and
  carbonyl probes fitted as Gaussians. A Li⁺, Na⁺ or K⁺ probe, Coulomb plus repulsion with the
  module's own caveat about the wall, makes the field of an ionophore or an electrolyte solvent
  the energy a cation would feel. Measure on a crown-ether or calixarene binding-constant series
  and on Li⁺ solvation energies of carbonate and ether solvents (computable here with ORCA, so
  the set can be made rather than found), against the electrostatic field; selectivity between
  cations is a multi-task problem and the first electrochemical application of E1's task
  descriptors.
  **Done 4 November 2026**, and no set had to be computed: the ether series of Wang, Yang, Harris
  and Gomez-Bombarelli (Chem. Commun. 56 (2020) 8920) is public, holds 1,085 saturated acyclic
  ethers with the binding energy of Li(+), Na(+) and Mg(2+) at wB97X-D3/def2-TZVP, and carries the
  relaxed geometry of 1,050 lithium complexes, which a flexible glyme needs, since a conformer
  from a generator is not the conformer the number belongs to. `examples/make_ether_examples.py
  --fetch` prepares it (`examples/ether/`, status *unclear*: the distributor states no licence, so
  the hashes are checked and `upstream/` is not redistributed) and `benchmarks/run_ether.py`
  measures it in three stages.

  Four probes were added to `cmfqsar/probes.py`: `lithium`, `sodium`, `potassium`, `magnesium`.
  The lithium radius, 0.49 angstrom, is measured rather than assumed: it is the one that puts the
  probe's contact with an oxygen at 2.01 angstrom, which is where the oxygens of those 1,050
  complexes sit (2.007 +/- 0.113 angstrom, 4.16 +/- 0.97 of them within 2.6 angstrom). The others
  follow Shannon's six-coordinate radii. **What separates them is one quantity.** An ion carries
  no hydrogen bond and its dispersion well is two per cent of its electrostatic term, so a +1 ion
  probe is the molecule's electrostatic potential sampled at that ion's contact distance, and the
  three +1 probes differ in the distance alone; the +2 probe multiplies the field by two, which a
  self-overlap-normalised kernel cannot see, so the magnesium probe is the lithium probe at a
  shorter distance. A test holds each of these statements.

  **A fault found and repaired on the way.** A probe field keeps its weights per Gaussian term and
  not per atom, so its atom weight column is zero; the alignment-free kernel of
  `cmfqsar/invariant.py` read that column and returned a kernel of zeros, which states that every
  molecule is alike, and nothing refused it. `expanded_weights` now spreads a probe field over its
  terms as one channel each, which is exact, `feature_matrix` needs the field set whenever a probe
  field is present and refuses without it, and `CMFModel` passes it, so `--kernel autocorrelation
  --probes lithium` works from the command line and the GUI.

  **Where the ion sits: the gate no regression is needed for.** A probe field is a map of an
  energy, so its deepest point is a prediction of the ion's position. Over the 1,050 complexes it
  lies 0.36 +/- 0.55 angstrom from the lithium the relaxation placed and within 1.0 angstrom in
  97 per cent of them, against 0.65 +/- 0.74 angstrom (86 per cent) for the centroid of the
  oxygens and 0.67 +/- 0.80 (82 per cent) for the centroid of the heavy atoms.

  **The binding energy: the gate is passed.** Nested cross-validation, five outer folds, the ridge
  and the field width chosen inside each fold, no frame anywhere. Mean absolute error in eV for
  Li(+), Na(+) and Mg(2+): predicting the mean of the set 0.454, 0.449, 1.235; the oxygen count
  with the carbon-to-oxygen ratio and the heavy-atom count 0.296, 0.279, 0.505; the same with the
  molecule's HOMO energy 0.254, 0.246, 0.444; the electrostatic field alone 0.209, 0.248, 0.557;
  the five fields 0.124, 0.181, 0.266; the matching ion probe alone 0.140, 0.193, 0.321; the five
  fields and the probe 0.117, 0.171, 0.254 (R2 +0.910, +0.706, +0.856). Paired over the same
  molecules, the ion probe beats the electrostatic field alone -- the comparison this item names --
  by 0.069 eV [0.060, 0.078] on lithium, 0.055 [0.046, 0.063] on sodium and 0.236 [0.210, 0.263]
  on magnesium, and the five fields beat the counts by 0.172 eV [0.152, 0.191] on lithium. The
  probe alone is a little behind the five fields (+0.015 eV [+0.009, +0.022]). With one oxygen
  count held out at a time the fields and the probe give 0.129, 0.204, 0.279 against 0.284, 0.291,
  0.571 for the counts. This is the first measurement of the battery and electrochemistry series
  in which the arrangement of the atoms beats what the composition already says.

  **The selectivity: measured, small, and explained.** The lithium-minus-sodium difference over
  the 980 molecules that carry both has a spread of 0.364 eV and a correlation of -0.049 with the
  oxygen count, so the formula says nothing about it (q2 +0.006). Predicting its mean costs
  0.190 eV; the five fields and the probe reach 0.171 (q2 +0.218) and the two ion probes 0.176.
  The field models are ahead of the counts by 0.012 eV [0.003, 0.022], a real and small
  difference. Adding the sodium probe to the lithium probe changes nothing (+0.001 eV
  [-0.000, +0.003]), which is what the probe definitions above predict, and is the measured form
  of the statement that selectivity needs polarisation.

  **The three ions as three endpoints (item E1 applied to ions).** 3,010 measurements over 1,048
  molecules on one kernel: independent endpoints cost 0.204 eV, one shared correlation of 0.5
  gives 0.198, and a covariance taken from the ion's radius, charge and polarisability 0.188, with
  q2 +0.893, +0.698, +0.852. The covariance relates lithium to magnesium (0.46) more closely than
  to sodium (0.27), so what decides which endpoint helps another is the size of the ion (Li+ 0.76,
  Mg2+ 0.72, Na+ 1.02 angstrom) and not its charge. **An ion that was never measured cannot be
  predicted on this set**: magnesium is refused by construction, because the mean of an endpoint
  is a linear function of the charge and two monovalent ions do not determine it at +2, and
  lithium and sodium give 1.484 and 0.539 eV against their own means of 0.454 and 0.449. They beat
  every reference available without the ion (the mean of the measured ions is 3.5 and 4.7 eV away)
  and the part that fails is the mean: lithium's is -3.84 eV and the model places it at -2.36.
  Three endpoints carry one descriptor column, which is a limit of the set.

  **Not done:** the potassium probe has no data here, so it is defined and unmeasured; the
  crown-ether and calixarene log K series this item also names needs the Izatt tabulations, for
  which no machine-readable open copy was found on 4 November 2026 (the NIST stability-constant
  database is licensed), so that half of the item stays with SC5, which needs the same data and
  the cavity frame; and the targets here are gas-phase one-to-one complexes, not solvation free
  energies.

**Large (a month or more each)**

* **EC8. Clusters cut from periodic sites.** A single-atom or metal–nitrogen–carbon site is a
  molecular active site inside a solid. A cut-and-terminate step from a periodic cell to a finite
  cluster with atom types the field builder accepts, then the EC4 path, gives adsorbate binding
  free energies (H, OH, CO: the volcano-plot descriptors) as a cheap surrogate inside a family of
  sites, with the GP variance to choose the next DFT calculation as E2 does for molecules. Gate:
  on a public single-atom-catalyst set, beat the periodic autocorrelation kernel and M5's
  site-resolved model on the same cells. Waits for M5, which decides whether the site resolution
  of the periodic kernel already does the job.
  **Pilot measured 1 November 2026, without the cut-and-terminate step and without M5**
  (`benchmarks/run_sac.py`, `benchmarks/results/sac/table.md`, `examples/sac/README.md`): the
  environment of the metal cut from the periodic sheet by the hop machinery of BT7, in the frame
  the shared 8x8 template gives for free, on the 140 M@C(4-x)Nx catalysts of He, Wang and Ouyang
  (Chem. Mater. 2023) with their H, O, OH and OOH binding energies. The source data have a fault
  the pilot found and works around: the bare sheets of eight metals were relaxed with DFT+U while
  every adsorbed state was relaxed without it, so 40 catalysts carried no binding energy until
  their sheets were recomputed with plain PBE (a VASP queue in WSL, about 45 minutes per sheet;
  finished 5 November for 35 of the 40, tungsten's five excluded because its pseudopotential is
  not on this machine); 15 OOH states dissociated and Os@C4's sheet is 7 eV off. On the 98
  catalysts the pilot had, with
  gas references computed at the set's settings and three fold designs (random, one metal held
  out, one motif held out), the overlap kernel of the site with electrostatic, steric,
  electronegativity, d-count, valence, ionisation and radius fields gives mean q² +0.84, +0.74
  and +0.82 over the four endpoints; OH and OOH are predicted to 0.4 eV (q² 0.93), O to 0.7 eV,
  H to 0.7 eV (q² 0.58, the hard endpoint). **The gate is not passed**: the periodic
  autocorrelation of the whole cell gives +0.84, +0.78, +0.83 and the tabulated metal properties
  with the nitrogen count +0.82, +0.81, +0.82, every difference inside the bootstrap interval
  except for O on unseen metals, where the site kernel is worse by 0.17 to 0.19; the invariant
  local kernel of the same environment matches the frame kernel everywhere, so the frame adds
  nothing to the prediction. The indicator fields of the metal collapse on unseen metals (q²
  +0.26), as a one-hot does. Sharing the four endpoints in one multi-task fit gains nothing
  (the chosen task correlation is zero). What the pilot delivers is what the references cannot:
  the bifunctional difference dE_H - dE_OH is predicted with q² +0.82 (0.55 eV on an sd of
  1.30), the ORR overpotential derived from the out-of-fold energies ranks the sites with a
  rank correlation of +0.90 (+0.79 on unseen metals) and recovers 7 of the 10 best bifunctional
  sites, and the difference co-field says where the two reactions part: the charge at the metal
  (+0.39 eV per unit of the electrostatic field there: an electropositive metal binds OH and OOH
  more and H about the same) and at its four neighbours (+0.11: nitrogen in the first shell
  weakens OH binding more than H binding), while the size of the metal weakens all three alike.
  The GP interval covers 84 to 93 % at the nominal 95 % and its width ranks the errors weakly
  (rank correlation +0.06 to +0.34 on unseen metals), so the variance is not yet the acquisition
  rule E2 would need.
  **Rerun on the full set, 5 November 2026** (133 catalysts with targets, the same command):
  the site overlap kernel gives mean q² +0.76 / +0.68 / +0.71 over the four endpoints on the
  random / metal-out / motif-out designs, against +0.80 / +0.74 / +0.74 for the periodic
  autocorrelation, +0.75 / +0.74 / +0.76 for the metal properties with the nitrogen count and
  +0.79 / +0.74 / +0.74 for the invariant kernel of the same environment; on O in every design
  and on OH with a motif held out the site kernel is worse by 0.04 to 0.17 with the interval
  clear of zero, every other difference is inside its interval. The gate stays failed. Two
  things the full set changed. (1) The random folds alone now choose the indicator field set by
  a margin inside the interval (+0.79 against +0.76), and that set collapses on an unseen metal
  (q² +0.30), so the field set the deliverables read from is chosen by the mean over the three
  designs (`run_sac.py`, the extended set wins at +0.72 against +0.64 and +0.61); choosing on
  one design and reporting on another is the selection trap of M10 in a new form. (2) The
  co-field reading is a property of the kernel's smoothness: at the pilot's kernel (envelope
  0.2, attenuation 0.5, raw, ridge 0.1) the difference co-field of the electrostatic field at
  the metal is +0.23 eV per unit on the 133 and +0.16 on the pilot's 98, the same reading; at
  the sharper kernel the full set chooses by accuracy (attenuation 2, cosine, ridge 0.001) the
  weights are ten times larger and change sign between the two subsets. The two-reaction
  reading holds: dE_H - dE_OH at q² +0.81 (0.53 eV on an sd of 1.22), the ORR overpotential
  ranked at +0.88 (+0.79 on unseen metals), 4 to 6 of the ten best bifunctional sites recovered
  (the DFT top ten is now led by Co@N4, Co@CN3 and Co@C2N2, which the recompute brought in). The
  GP interval covers 86 to 96 % and its width does not rank the errors. Next, in order: the
  co-field drawn at a kernel whose smoothness is fixed rather than chosen by accuracy; a frame
  averaged over the point symmetry of the site (the D2h of the divacancy), which is the M14
  device and is what the frame kernel lacks against the invariant one; the EC4 metal types so
  that the same measurement runs on the molecular macrocycles.
* **EC9. Potential-dependent targets.** A catalyst is a curve, turnover frequency against
  overpotential, not a number. The descriptor block already admits the applied potential as a
  condition, which covers a fixed set of overpotentials; a product kernel over catalysts and
  over potential, fitted on the whole catalytic Tafel plot, is the model of the curve. Exploratory
  in the sense of the reactions bullet of H: no commitment until EC4 has shown that the catalyst
  kernel carries signal at one potential.
* **EC10. The application.** None of this is visible in the GUI: no slab behind the molecules
  (the protein viewer is the model), no vector co-field, no surface frame. It belongs to G and
  waits for EC2, EC5 and EC6, which decide what there is to draw.
  **Done 6 November 2026**, on the window and the viewer item M13 built. Three pieces, each in
  the main window rather than in a window of its own, because an adsorbate *is* a molecule and
  the molecular view is the right place for it.
  (i) *The slab behind the molecules.* `cmfqsar/gui/surface_view.py` reads a slab from a file or
  builds a bcc(110) one with the defaults of the Fe(110) reference poses (*File -> Load a surface
  slab* and *Build a bcc(110) surface slab*), and draws the top layers repeated over the surface
  lattice and cut to the molecules **in the plane** -- not in space, so the surface stays a
  surface under a molecule that sits high above it. The expansion is
  `slab.slab_images`, factored out of `slab_receptor`, so the picture is the atoms the
  complementarity receptor is made of and not a lookalike: 100 % of the drawn atoms are its own.
  The top, bridge and hollow sites of `slab.surface_sites` are drawn too, translated to the
  molecules over the surface lattice.
  (ii) *The surface frame.* `Viewer3D.add_frame` draws the three axes at the origin in the
  colours a cell's lattice vectors carry, and `planealign.pose_geometry` gives the two numbers
  the surface kernel sees -- the height of the lowest heavy atom and the tilt of the molecule's
  own plane -- for the legend under the view and for the *Surface poses* tab, which
  *Tools -> Pose the molecules in the surface frame* fills by running `plane_align_dataset`. A
  molecule with no plane is told apart rather than given a meaningless angle.
  (iii) *The vector co-field.* The electric-field kind of EC5 is now a field in the Model panel
  (unticked by default, because its value is a question per series), and `Viewer3D.add_arrows`
  draws its co-field as arrows from `cofields.cofield_arrows`, scaled so the longest is 2.5 A on
  screen with the legend stating what that length is in activity units per unit of the field per
  angstrom. Ticking the box on a scalar field draws nothing and says why.
  **Measured** (`benchmarks/run_surface_view.py`,
  `benchmarks/results/surface_view/table.md`; the window is driven without a screen):
  a 6 A patch of the Fe(110) slab is 93 surface atoms and a 24 A patch 845, chosen in 1 ms
  either way, the whole view drawn in 3 to 14 ms. The height the panel states agrees with the
  `height_A` column the relaxation wrote -- which the window never reads -- to **0.005 A** on all
  138 molecules of the two nodes, which is the one check of the frame that does not rest on the
  program's own arithmetic. The same stage measured what the relaxed poses are: heights 1.23 to
  4.83 A, median tilt 18 and 27 degrees on the two nodes, and only 29 % and 14 % of the molecules
  flat at all, so a molecule relaxed on iron is **not** lying flat and the flat rule pose differs
  from the relaxed tilt by that median. Sixteen arrows on a 0.5 A grid of 21,600 points take
  0.7 s; the longest arrow is the largest gradient the grid holds at every spacing tried, and the
  weakest drawn is 40 to 50 % of the longest, which the gradient exceeds in 1.5 to 2.2 % of the
  volume. One finding worth recording: refining the grid from 0.7 to 0.35 A turns the arrows by
  **about 16 degrees** on average, because an arrow is taken at the grid point nearest a peak and
  a finer grid has a nearer one, so an arrow states a direction to roughly fifteen degrees and not
  better -- enough for which way the field should point, not enough to measure an angle off the
  picture.
  **Not done:** no docking from the window (the constrained pose search stays in `slab.py` and its
  benchmark), one slab at a time, and the slab is not part of a project file.

**Order.** EC2 was first and is done: it gives the surface frame and the co-field in it, and
it found that the relaxed frames carry the relaxation's outcome, which changes what the Fe(110)
numbers mean. ~~EC1~~ **done 5 November 2026: the quantum stage now has the oxidation state, the
spin state and the solvent, and the spin field it was built for gains nothing on a phenazine
series, because the added electron goes to the same place in every member.** ~~EC4~~ **done
5 November 2026: coordination complexes load with dative metal–ligand bonds, the metals have
parameters, the electrostatic field of a complex is no longer identically zero, and the gate is
passed — on 25 congeneric nickel complexes the five fields beat the population analysis of the
calculation that produced the target. The catalyst series the item named are not in
machine-readable form, so EC8 and EC9 still have no measured catalytic property to aim
at, even though a complex now loads.** ~~EC3~~ **done 5 November 2026: the field and its gradient
at anchored points of a shared frame are readable, exactly, from the Gaussian field or as a true
point-charge potential, and reading a few places loses to reading everywhere on every target
measured; its lasting result is that a best-of-N column correlation is not the fair diagnostic
that a whole-kernel correlation is.** ~~EC5~~ **done 5 November 2026: the vector kernel is exact
and is a near-copy of the scalar one (the two cosine kernels correlate +0.980), so it changes
nothing where it is added and costs accuracy on the one series where it does anything; kept under
rule 5 as a substitute for the electrostatic field and for its arrow co-field.** ~~EC6~~ **done
5 November 2026 after M1: a slab is a receptor the pocket machinery takes unchanged, and on the
Fe(110) sets the gate fails on both halves -- docking against the slab does not place the
molecules, and the burial score that beats EC2's height control does so only at the relaxed pose
it was supposed to find.** ~~EC7~~ **done
4 November 2026, and its gate is passed**, EC8 after M5, EC9 and EC10 last. The EC2 result said
what EC6 had to do differently: a slab as the receptor has to decide the face and the tilt by
complementarity, since no rule and no reference does. EC6 measured that complementarity does not
decide them either.

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
  **Done 3 November 2026.** Two faults, one of them silent. `oxidation_states` now checks that
  pymatgen's guess leaves the cell neutral and drops it when it does not: for Li(0.5)FePO4 the
  guess was Fe(II), which leaves half a charge over, and the field was built from it without a
  word. And when no assignment in whole oxidation states exists, the remainder is spread over the
  redox-active metal: `_mixed_valence_balance` holds every other element at one of its common
  states and divides what is left by the amount of the last one, so Li(0.5)CoO2 is Co(+3.5) and
  Li(0.5)FePO4 is Fe(+2.5), where before the first gave nothing and the electronegativity proxy
  took over. Only a transition metal, a post-transition metal, a lanthanide or an actinide may
  carry the mixed valence; otherwise the arithmetic is happy to make the oxygen of Li(0.5)CoO2
  O(-1.75) or the chlorine of a half-occupied LiCl2 rock salt Cl(-0.5), both of which balance and
  neither of which is chemistry. A cell with no such element is refused as before. The charge
  model `"dft"` is new beside `"oxidation"`, `"electronegativity"` and `"file"`: it reads
  `bader_charges` first, then the other per-atom arrays, and falls back to the formal states, so a
  series of computed cells uses what the calculation gave (item M3) without naming a file, and no
  existing benchmark changes because the default is untouched. The test the item asked for is in
  `tests/test_battery.py`: Li_xCoO2 for x = 1, 0.75 and 0.5 is neutral to a millionth of a charge
  and its cobalt goes +3.0, +3.25, +3.5.
* **BT2. Battery descriptors.** The descriptor block has no computed sources for crystals. The
  theoretical capacity from the formula, the Li content, the volume per formula unit and the
  density as per-structure descriptors; the coordination number and the mean Li–O distance as
  per-complex descriptors for the shells of BT4. Simple once the computed-descriptor registry of
  `cmfqsar/descriptors.py` accepts an ASE `Atoms` as well as an RDKit molecule.
  **Done 3 November 2026.** `descriptors.CRYSTAL` holds eight names under the same `calc:` prefix
  -- `capacity`, `ion_count`, `ion_fraction`, `volume_per_formula`, `volume_per_atom`, `density`,
  `ion_coordination` and `ion_oxygen_distance` -- and `compute` sends an ASE `Atoms` to them
  instead of to RDKit, so one registry serves a molecule and a cell. The working ion is lithium
  unless the name says otherwise: `calc:capacity@Na`. `ion_count` is per *host* formula unit,
  which is the x of Li_xHost and not the formula unit of the cell, so Li2Co4O8 reads 0.5. The two
  shell descriptors use the periodic neighbour list with a cutoff of one and a half times the sum
  of the covalent radii, so a shell across the cell boundary counts and nothing of the second
  shell comes in. `read_values` reads any name that is not a `calc:` one from `atoms.info`, where
  a loader leaves the numbers that came with the structure, and `crystal_fields` fills the
  descriptor row of a cell, so the descriptor block of the kernel works for a crystal as it does
  for a molecule. Checked against the textbook capacities: lithium metal 3862 mAh/g, LiCoO2
  274 mAh/g.
* **BT3. A battery data loader.** `read_structures` reads files through ASE and
  `structures_from_matbench` reads one table. Electrode data from the Materials Project battery
  explorer come as charged and discharged structure pairs with the average voltage, the volume
  change and the stability per pair, and the OPTIMADE route works from this machine without a key. A loader that fetches the pairs,
  keeps both frames and writes them in the form `crystal_table` takes gives the periodic kernel a
  battery benchmark of thousands of entries, which no molecular series has. Measure the
  autocorrelation kernel on the average voltage against a composition baseline, by chemistry
  (oxides, phosphates, sulfides) as the Sutherland sets are measured by size.
  **Done 3 November 2026, and the gate it sets itself is not met: the composition decides the
  voltage.** `cmfqsar/battery.py` and `benchmarks/run_battery.py` (three stages: `fetch`, `check`,
  `voltage`), with 303 pairs measured and 34 tests in
  `tests/test_battery.py`.

  **The data, without an account.** The item assumed the OPTIMADE route would serve the battery
  explorer's pairs. It does not: OPTIMADE serves structures, and the insertion-electrode documents
  need a free API key, which this machine has none of. The way round it is better than the
  original plan. OPTIMADE does serve the **formation energy** of every structure, and a formation
  energy carries the elemental reference, so elemental lithium cancels in the voltage and
  V = -[E_f(Li_yHost) - E_f(Li_xHost)] / (y - x) per host formula unit. One query per chemical
  system (the ternary lithium-metal-oxygen systems, and the phosphates, sulfides, fluorides and
  silicates of the eleven redox metals: 73 systems, 19,412 structures, 58 MB of cache, four
  minutes), the phases within 0.03 eV per atom of the hull kept, every cell reduced to
  its host framework and its lithium content per host formula unit, neighbouring steps paired.
  `insertion_electrodes` fetches the official documents when a key is given, so the official pairs
  are one environment variable away.

  **Checked before it was used.** Over the first insertion the four textbook cathodes come out
  0.39 V low on average and low every time: LiCoO2
  3.77 against 4.10, LiFePO4
  3.33 against 3.45, LiMn2O4
  3.23 against 4.00, LiNiO2
  3.47 against 3.80. The energies carry none of
  the corrections a battery database applies, and the average here runs over the intermediate
  phases of the ladder. The hull window was chosen on that table: the agreement is
  0.39 V at 0.01, 0.02 and 0.03 eV per atom while the set grows
  from 83 to 303 pairs, and 0.57 V
  at 0.08.

  **The measurement.** 303 pairs over 157 frameworks, five folds that keep a framework
  together (two steps of one ladder share most of their structure), the ridge and the width chosen
  inside the training part. Mean absolute error over the folds: the mean of the set
  1.203 V, composition 0.569 +/- 0.115 V,
  fields 0.650 +/- 0.146 V, the two together
0.582 +/- 0.117 V. **The composition decides this property and the arrangement adds nothing.** The
  field kernel is 0.081 V worse than the composition baseline and the two together 0.013 V worse,
  both inside the fold-to-fold spread of 0.146 V, which under rule 4 is not a result. That is the
  answer the data support for an average voltage: it is a thermodynamic quantity of a composition,
  and the geometry enters only through which phase that composition takes. The item is kept under
  rule 5 because it is the loader and the baseline every other battery item needs, and because a
  negative result measured against a baseline is worth more than an unmeasured kernel.

  **Not done:** no topotactic test of a pair (the frameworks are matched by composition, not by
  structure), no official insertion-electrode documents (they need the key), no volume change or
  stability as a second target although the loader carries both, and no cathode series of its own
  -- the set is public phases, not the materials a group is working on.

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
  **Done 4 November 2026. The gate is met by a small margin, the co-field reading passes, and the
  frame the item expected to pay does not.** `cmfqsar/solvation.py` with 27 tests in
  `tests/test_solvation.py`, measured by `benchmarks/run_solvation.py` in three stages.

  **What was built.** `place_shell` puts one to six molecules, and optionally an anion for a
  contact ion pair, on the polyhedron of that coordination number: each molecule turned so that
  its donor's lone pair points at the ion, the donor set at the coordination distance (2.01 A for
  Li and oxygen, the distance measured on the complexes of item EC7), each molecule spun about its
  own ion-donor axis to the angle that leaves the most room. `wrap_ion` is the intramolecular case
  of a glyme. The optimiser is GFN2-xTB through `cmfqsar.xtb`, which already cached by molecule,
  charge, method and starting geometry, so nothing new was needed there and a complex of fifty
  atoms takes under a second. `cation_frame` is the frame the item promised: the origin on the
  ion, the axes along the principal axes of the donor set, the four proper sign choices decided by
  the donors' third moments, and the rotation that a regular polyhedron leaves free settled by the
  heavy atoms of the whole complex; `frame_quality` reports the moment ratios, so a caller can see
  when the frame is not fixed (0.3 per cent of the reference complexes).

  **The builder against a published relaxation**, on 150 of the 1,050 DFT-relaxed Li(+)-ether
  complexes of `examples/ether/`, three xTB runs per molecule and five starting conformers for the
  builder's own complex: oxygens reaching the ion 4.24 +/- 0.87 in the published relaxation,
  4.38 when xTB re-optimises that geometry, 3.81 +/- 0.84 for the builder; mean Li-O distance
  1.995 +/- 0.068 A against 1.926 +/- 0.088; binding energy correlation with the published values
  +0.907 for xTB on the published geometry (mean error -1.323 eV) and +0.555 for the builder's own
  complex (-0.919 eV). Two separate statements: **GFN2-xTB overbinds a lithium ion to an ether by
  about 1.3 eV and still ranks the complexes well**, so it is a ranking method here and not an
  absolute one; and **the builder loses the longer chains**, for a measured reason -- the five
  starting conformers of one molecule spread 1.52 +/- 0.71 eV, so the start decides the answer and
  five are not enough for a chain with five oxygens. What is missing is a conformational search,
  not a better placing rule.

  **The gate.** 1,048 complexes with a published binding energy, five outer folds, the ridge and
  the field width chosen inside each fold; predicting the mean costs 0.454 eV. Mean absolute error
  in eV: the solvent alone through the alignment-free kernel (the model of item EC7) 0.124
  +/- 0.016, the complex through the same kernel 0.114 +/- 0.013, the complex through the overlap
  kernel in the cation frame 0.145 +/- 0.018, and the solvent again in that same frame
  0.143 +/- 0.009. Paired over the same molecules, **putting the ion back gains 0.010 eV
  [0.004, 0.016]**, an interval clear of zero and eight per cent of the error: the gate is met,
  and the margin says that the solvent in the geometry it holds inside the complex already carries
  most of what there is to carry. The fourth model exists to separate the ion from the frame, and
  the separation is clear: **the frame costs 0.018 eV [0.010, 0.027]** against the alignment-free
  kernel, and inside the frame the ion adds nothing (+0.002 eV [-0.003, +0.008]).

  **The co-field reading passes.** A frame is what a co-field needs, and this is what the frame is
  for, since it does not buy accuracy. Fitted on every complex and evaluated on a grid, the
  largest magnitude of the electrostatic co-field lies 1.37 A from the ion and 1.37 A from the
  nearest donor oxygen, and that of the hydrogen-bond acceptor field 1.03 A from the ion: both
  inside the first shell. The hydrogen-bond donor field is the control that costs nothing, since
  an ether cannot fill it, and its largest magnitude lands 12.5 A away in the corner of the box.

  **The series the item names, computed here.** 19 carbonates, ethers and fluorinated ethers with
  one to four molecules around Li(+), 75 complexes, one failure. At one molecule the carbonates
  bind -2.49 +/- 0.12 eV, the ethers -3.01 +/- 1.08 and the fluorinated ethers -2.91 +/- 0.61; the
  ethers' spread is chemistry and not noise, because 12 of the 19 solvents give a complex in which
  the ion touches one oxygen (those 12 agree at -2.36 +/- 0.18 eV) while the other 7 fold and
  donate two or three at once. Two readings worth keeping: fluorination weakens the donor, FEC
  binding 0.29 eV less than EC, and the fluorinated ethers hold the fourth molecule weakly, with a
  desolvation energy of 0.02 to 0.33 eV against 0.92 to 1.35 eV for the carbonates, which is what
  a diluent is chosen to do. Two of the 56 desolvation energies are negative, which means the
  optimiser found no bound complex at that size, and `series.json` names them.

  **Not done:** the conformational search the builder needs; the measured low-temperature or rate
  data the item also names, for which no machine-readable series was looked for here; ORCA as the
  engine (the function refuses any program but xTB with a message that says so); and a target
  other than the binding energy, since the desolvation energies and coordination numbers computed
  here are data and not yet a measurement. Item BT5 is the next one: it asks whether shells sampled
  from a trajectory beat this single optimised complex, and this item is the baseline it needs.
* **BT5. Solvation shells from a bulk trajectory.** D7 reads a ligand trajectory in a pocket. An
  electrolyte simulation is a periodic box with many cations, so a shell extractor that cuts the
  molecules within a cutoff of each Li⁺, unwraps the periodic boundary, and hands the shells to
  the ensemble builder as members with their own atoms (D4) is new code on top of both. The
  ensemble of shells is the only route from a field model to conductivity, transference and
  diffusion, and the fourth-dimension verdict applies: measure once, on a series with simulated
  or measured transport data, against BT4's single optimised complex, and keep the route only if
  the sampled shells beat the optimised one by more than the interval. The block-averaged error of
  the mean field is reported next to the q², as D8 requires.
  **Done 4 November 2026. The gate is met on the activation energy and not on the conductivity,
  and only in the alignment-free kernel.** `cmfqsar/electrolyte.py` and `cmfqsar/shells.py` with
  59 tests in `tests/test_electrolyte.py` and `tests/test_shells.py`, the data recipe in
  `examples/make_electrolyte_shells.py`, measured by `benchmarks/run_shells.py` in four stages.

  **What was built.** Two modules. `cmfqsar/electrolyte.py` builds and runs one bulk box per
  formulation: a `Formulation` is a salt, a set of solvent weight fractions and a molality, and
  from those three `counts` fixes the exact number of molecules, 250 solvents in the proportions
  asked for plus the ion pairs that reproduce the molality, with no density assumed. packmol packs
  the box, the OpenMM topology is then assembled from the same RDKit templates in a fixed species
  order (packmol writes no bond records, so the packed file supplies coordinates only), the
  solvents and the anion are typed with GAFF-2.11 and charged with the OpenFF NAGL graph network,
  the cation takes the Amber ion parameters because GAFF has no lithium, and the run is 150 ps of
  equilibration and 1,000 ps of production under a Monte Carlo barostat with particle-mesh Ewald.
  The net charge is checked against zero before the run starts. `cmfqsar/shells.py` cuts the
  shells: every molecule with a donor atom within 1.3 times the ion-donor coordination distance of
  item BT4 (2.613 A for Li and oxygen), the counter-ion included, since a shell holding the anion
  is a contact ion pair. Both periodic-boundary faults of item D7 are repaired, a molecule split
  across a face by `make_whole` and a shell a box length from its ion by `reimage`, and the
  selection is written to screen every donor atom of the frame at once under the minimum-image
  convention and repair only the residues that pass, because a single electrolyte needs 3,500 to
  31,000 shells and the first version, which repaired every residue first, was too slow to use.
  `cluster_shells` thins the shells to 16 representatives by a fingerprint that does not depend on
  the order the molecules were found in, weighted by the share of sampled shells each represents.

  **Why this needed no new kernel.** A field is a sum of Gaussian functions on atoms and the
  kernel is bilinear, so the weighted mean field of a set of shells is itself a field: its atoms
  are all the shells' atoms, each weight multiplied by its shell's weight. `combine_members` in
  `cmfqsar/fields.py`, written for item D4, already accepted members carrying their own atoms,
  which is what shells are, since two shells need not hold the same molecules. Three things
  follow: the kernel of two mean fields is exactly the weighted mean of the shell-to-shell
  kernels, so a 16-member ensemble costs one kernel evaluation and not 256; the matrix stays
  positive definite; and the model stays linear in the mean field, so the co-field and the Shapley
  decomposition keep their meaning. The weights are populations and never Boltzmann factors, which
  is item D7's rule: an equilibrated trajectory has already sampled the distribution, so weighting
  it again would count the energy twice.

  **The two convergence numbers item D8 asks for.** `convergence` reports them separately, both
  computed exactly from one shell-to-shell overlap matrix with no model fitted: the **block error
  of the mean field**, how far the mean field of a block of frames lies from that of the whole run,
  which falls as the trajectory grows, and the **cost of keeping 16 members**, how far the thinned
  ensemble lies from all the shells, which does not. The first version measured the block error on
  the thinned ensemble and so mixed the two: it rose with more sampling, 0.055 at 40 frames to
  0.145 at 500, because thinning noise is larger than sampling noise and more frames give more
  distinct shapes to discard. A diagnostic that gets worse as the data get better is reporting the
  wrong quantity.

  **What was measured.** `benchmarks/run_shells.py` in four stages (`data`, `gate`, `cofield`,
  `report`) on 62 LiPF6 formulations over 22 solvent compositions, from 0.245 to 2.373 mol/kg,
  with measured conductivities from CALiSol-23 (CC BY 4.0), so that the target is measured and
  only the descriptor is computed. Two targets per formulation, read off a fit of
  log10 k = a + b/T + c/T^2 to that formulation's own measurements: log10 of the conductivity at
  298.15 K and the activation energy there. Every selection rule is in
  `examples/make_electrolyte_shells.py`, the sharpest being that LiBF4 and LiBOB, the larger part
  of that collection, are refused by name because GAFF has no boron. Seven models, two fold
  schemes (random, and grouped so that no solvent composition is in both the training and the test
  set), the ridge and the field width chosen inside each fold, each field kernel mixed in equal
  parts with the kernel on the molality at a fixed weight, with a separate reading at a weight
  chosen inside the fold. The complex and the shells each appear twice, in the overlap kernel in
  the cation frame and in the alignment-free kernel, because the kernel is not part of the item's
  question and item BT4 measured the alignment-free one as the better of the two for its own
  descriptor.

  **The gate is met on the activation energy, in three readings of four, and not on the
  conductivity.** On the activation energy, where predicting the mean costs 3.535 kJ/mol, the
  sampled shells reach 1.612 kJ/mol against the single optimised complex's 1.934 on random folds
  and 1.685 against 2.232 on the grouped folds, both in the alignment-free kernel. Paired, that is
  a gain of 0.323 kJ/mol [-0.082, +0.786] and 0.546 kJ/mol [+0.137, +1.038]: the grouped-fold
  interval is clear of zero and the random-fold one is not, while both variants with the kernel
  weight chosen inside the fold stay clear of it (0.436 [0.057, 0.889] and 0.586 [0.133, 1.163]).
  On the conductivity, where predicting the mean costs 0.146 in log10 mS/cm, there is no gain:
  -0.016 [-0.036, +0.004] and +0.000 [-0.018, +0.019].
  **Revised 5 November 2026 by item D11, which corrected the charges these numbers rest on.**
  Until then 72 % of these shells carried a net charge of +1 in the electrostatic field: the
  phosphorus of the hexafluorophosphate anion is a valence state the Gasteiger scheme has no
  parameters for, so it was set to zero while the lithium kept its +1. Correcting it makes the
  shell model slightly worse on the activation energy (1.456 to 1.612 kJ/mol on random folds) and
  takes the random-fold gain from 0.480 [0.096, 0.960], clear of zero, to 0.323 [-0.082, +0.786],
  which is not; the two fold-chosen-weight variants did not move at all, which says the weight
  search had been absorbing part of the fault. Two further verdicts changed, both against the
  shells: against the free-solvent model on grouped conductivity folds they are now worse by
  0.021 [0.002, 0.039], and against the framed complex on grouped activation-energy folds by
  0.493 [0.099, 0.903]. The co-field reading changed with them, and for the better: the
  electrostatic peak was +3.62 at 0.69 angstrom from the ion, which is on the cation itself, and
  is now -3.51 at 3.03 angstrom from the ion and 0.81 from a donor atom, which is where the
  coordinating atoms are. The difference between the two targets is
  the one `examples/make_electrolyte_shells.py` predicted before the measurement: an ion moves by
  leaving one shell and entering the next, so the barrier is the quantity a distribution of shells
  has a claim on, while a conductivity at one temperature is set largely by how much salt is
  dissolved and how viscous the mixture is.

  **Three readings beside the gate.** *The frame costs an ensemble far more than it costs one
  complex*: in the cation frame the shells are 0.621 kJ/mol [0.223, 1.003] and 1.230
  [0.536, 1.492] worse than their own alignment-free version, while for the single complex the
  same frame costs 0.068 kJ/mol with an interval containing zero. Sixteen shells have sixteen
  coordination polyhedra and one frame cannot hold them all, so the accuracy and the co-field,
  which needs the frame, pull against each other in this item. *The composition is still not
  beaten*: the shell field draws level with the molality and the solvent weight fractions on the
  activation energy (-0.081 [-0.397, +0.256] and -0.159 [-0.502, +0.221]) and is behind them on
  the conductivity, which makes this the fourth battery item in a row where a composition
  descriptor is at least as good as a field. *Against free solvent molecules the gain is not
  established*: +0.276 [-0.110, +0.765] and +0.439 [-0.006, +0.970], and free solvents need no
  simulation at all.

  **The co-field is readable.** The donor atoms of these shells sit 2.14 +/- 0.17 A from the ion,
  and three of the four fields that carry any weight have their largest magnitude within 1.3 A of
  a coordinating atom: the hydrophobic field 0.41 A from one, the hydrogen-bond acceptor field
  1.12 A, the electrostatic field 1.22 A (and 0.69 A from the ion, so between the ion and its
  donors). The steric field is the exception at 4.91 A from the ion, which is the molecule tails
  and the second shell. The hydrogen-bond donor field is the control that costs nothing: a
  carbonate and an ester have no donor, that field carries exactly zero weight, and the point it
  is reported at is a corner of the box 16 A away. The distance to the nearest donor atom was
  added to this reading because the distance from the ion cannot be interpreted without it: 0.69 A
  from the ion is inside the first shell but is not where any atom is.

  **What item D8 asks for, reported next to the accuracy.** Over the 62 electrolytes the block
  error of the mean field is 0.00019 +/- 0.00037 and the cost of keeping 16 members
  0.00652 +/- 0.00702. The two are reported separately on purpose, because the first falls as the
  trajectory grows and the second does not.

  **How this reading was reached, which belongs in the record.** The alignment-free complex model
  was added after the first run showed the framed comparison failing, and it is the model that
  turns the gate from unmet into met. The reason for adding it does not depend on the result --
  comparing a framed ensemble only against a framed complex leaves out the better of item BT4's
  two models -- but the decision came after the first numbers were seen. What argues against it
  being a lucky choice is that the gain holds in both fold schemes, holds with the kernel weight
  chosen inside the fold, and runs in the direction the item predicted in advance.

  **The cost.** 62 bulk boxes of 3,500 +/- 398 atoms, 150 ps of equilibration and 1,000 ps of
  production each, at 456 +/- 49 ns per day, 191 +/- 17 s per box including the shell extraction,
  no failures: about three and a half hours in all on one CUDA device. The trajectories are 1.2 GB
  and are not kept; the 16 representative shells per electrolyte are, at 4.4 MB in all.

  **Not done:** a polarisable force field, so the 6.95 +/- 0.40 donor atoms at the ion are above
  the 4.24 oxygens item BT4 measured on complexes relaxed with density functional theory, and the
  number is recorded rather than corrected; transference numbers and diffusion coefficients, which
  the item also names and for which the trajectories here are too short; a frame that an ensemble
  of different shapes can actually use, which is what would make the co-field cheap instead of
  expensive; and the salts GAFF cannot parameterise, which is the larger part of CALiSol-23.
* **BT6. Difference kernels on the cell.** The intercalation voltage is the energy difference
  between the lithiated and delithiated cells, and the field-difference module
  (`cmfqsar/fielddiff.py`) has no periodic counterpart. Once M4 gives the periodic overlap kernel
  in a fixed frame, the difference of the two cells' fields is a field in the cell like any
  other, its kernel is the same closed form, and its co-field shows the coordination environment
  the inserted lithium wants. Small once M4 exists; listed here because BT3 supplies the pairs it
  runs on. Measure on the BT3 voltages against the autocorrelation kernel on the discharged cell
  alone.
  **Done 3 November 2026, and the gate is not met.** `cmfqsar/celldiff.py` with 18 tests in
  `tests/test_celldiff.py`, measured by `benchmarks/run_battery.py --stage difference`.

  **The construction.** `difference_fields` returns the field difference of a pair as an ordinary
  `CrystalFields`: the discharged cell's atoms with their weights, the charged cell's atoms moved
  into the discharged cell's lattice with their weights negated. The host cancels where the two
  cells agree, and what is left is the inserted ions and the relaxation. Two things do not cancel
  and should not: the formal charge of the metal, which item BT1 makes Co(+3.5) at half lithiation
  against Co(+3) at full, and the electronegativity proxy, which is a deviation from the mean over
  the cell. Every kernel takes the result unchanged, since each is bilinear in the weights.
  `can_difference` is the gate on a pair (one lattice within a tolerance, the same host atoms,
  more of the working ion in the discharged cell) and `usable_pairs` returns both the pairs that
  pass and a record for every refusal, so a benchmark reports what the frame costs.
  `inserted_positions` matches the ions of the two cells and returns the sites with no partner,
  and `shell_of_the_inserted_ion` measures what surrounds them.

  **What the frame costs.** 42 of the 303 pairs of item BT3 have their two cells
  within 5% in cell length with the same host atoms, over 33
  frameworks; 261 are refused, almost every one because the two cells relaxed to
  lattices too far apart -- which is the volume change of the insertion, a quantity a battery
  paper reports rather than an error.

  **The measurement**, five folds grouped by framework, mean absolute error in volts: the mean
  voltage of these pairs 0.791, composition
  0.656 +/- 0.100, the discharged cell alone
  0.699 +/- 0.143, the difference field
  0.797 +/- 0.083, the difference field with the composition
  0.680 +/- 0.118. Paired over the same samples the difference
  field is +0.090 V against the discharged cell
  [-0.034, +0.216], an interval that contains zero, and
  +0.134 V against the composition
  [+0.004, +0.273], which does not. **The item asked the difference
  kernel to beat the autocorrelation on the discharged cell alone; it does not, and it loses to
  the element fractions.** With 42 pairs over 33 frameworks the set resolves
  only large differences, and it agrees with item BT3: a voltage follows the composition.

  **What it gives instead.** The geometric reading, measured on the pairs rather than read off a
  model: the inserted ion sits in 5.0 +/- 1.1
  oxygens at 2.22 +/- 0.13 angstrom, the first shell of
  a lithium in an oxide, over 33 hosts.

  **Not done:** the co-field in the cell, which is the reading the item promised. It needs one
  parent lattice for the whole set, and 33 hosts have none, so the kernel here is
  the autocorrelation and its co-pattern has nothing to read at R² +0.014. The
  map needs a ladder of one host on one lattice, which is the beta-Li3PO4 vacancy series this
  program still has to compute for itself; the overlap kernel of item M4 is ready for it and
  `difference_fields` feeds it unchanged.

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
  **Done 30 October 2026, by the local route rather than through M4 and M5.** The item was
  written for the whole-cell route; what it needed was already in the per-atom items. LP2's
  environment becomes `hops.point_environment`, the same envelope cut from a periodic cell with
  its images, about a *point* (the midpoint of the hop) and without the migrating ion; LP3's
  frame becomes the hop itself, z from the first site to the second, with the rotation about the
  axis **averaged in closed form** rather than chosen: the mean of the overlap kernel over the
  rotation group about the hop is the plain kernel with the cosine term replaced by a modified
  Bessel function (`cylindrical_blocks`, one evaluation per atom pair, equal to 96 finite poses
  to 1e-14 and a thousand times faster); LP1's index becomes (structure, site a, site b, lattice
  offset), with folds that hold out whole structures. The enumeration, the symmetry reduction by
  a fingerprint, the bottleneck (least free space on the straight path, with the atoms that form
  it), the bond-valence labels (the minimax level connecting the two sites through a box around
  the path on the softBV landscape of `bvlain`), a table reader for DFT or measured barriers,
  the co-field as a (z, r) profile with its error, exact atom contributions, and the percolation
  step (union-find with lattice offsets: the level at which the hop network wraps the cell in 1,
  2 and 3 directions) are `cmfqsar/hops.py`; `cmf hops` runs it. 23 tests.
  **Measured** (`benchmarks/run_hops.py`, `benchmarks/results/hops/`): 472
  ternary compounds of lithium and oxygen (Li, O and one more element) of the Materials Project within 0.05 eV/atom of the hull
  (`prepare_hops.py`, through OPTIMADE), 24004 Li hops, 1732 distinct
  and labelled, 10 folds holding out whole structures, environments within
  6 A of the midpoint. The count of atoms in shells about the midpoint (rule 1)
  gives q2 +0.491, the hop length with the bottleneck distance +0.103; the
  rotation-invariant local kernel +0.657 at rho 0.25; the overlap
  kernel in the hop frame +0.662 at rho 0.25 (error 0.469 eV),
  +0.170 [+0.112, +0.265] against the counts and
  +0.005 [-0.060, +0.071] against the invariant kernel at the same
  width. Learning curve of the frame model: 25 structures +0.078; 50 structures +0.153; 100 structures +0.441; 200 structures +0.498; 400 structures +0.660; 459 structures +0.662. The predicted barriers run through the
  percolation step reproduce the thresholds of the labelled network with correlation
  +0.808 (1D, 164 structures), +0.857 (2D) and
  +0.754 (3D), mean absolute difference 0.38, 0.35
  and 0.34 eV. Applied unchanged to the 48 distinct hops of
  the three hosts of the Li interface study the public model gives q2 -1.415 (correlation
  +0.683, error 0.279 eV); within the hosts, leave one host out, the frame
  kernel gives -0.120 against -0.115 for the counts, which three structures
  cannot turn into an interval. The two DFT elastic-band hops (Li3PO4 0.29 eV,
  Li3BP2O8 0.00 eV) sit next to bond-valence values of 0.73 and
  0.39 eV and predictions of 0.80 and 0.56 eV:
  vacancy hops with full relaxation against a rigid landscape, listed and not compared.
  **Since M2** (31 October 2026, the day after this item) the benchmark has a third part: partly
  occupied sites are read as they are refined, and thirteen disordered conductors of the
  Crystallography Open Database are measured -- garnets, perovskites, LGPS, NASICON, LISICON, a
  thio-phosphate and Li3N. What that measurement says, including that the public model does not
  transfer to them, is in M2's entry, which added it.
  **The gate as written could not be applied when the item was done**: there was no M5 model to
  beat, so the reference is the count and the two chemist's numbers, and the co-profile is read
  as the gate asks. What it says is in the manual's section *A migration path as the sample*.
  M5 was done two days later, on 1 November 2026, but not on these cells with these labels: it
  resolves the bond-valence site **energy** of the 104 Li sites of the three hosts, not the
  **barrier** of a hop between two of them on the public set.
  **Gate passed, 6 November 2026**, by a model of the same hops built from the site features of
  M5. `site_pair_features` of `benchmarks/run_hops.py` describes each of the two end sites of a
  hop by the row of the periodic autocorrelation of its cell that belongs to that site, which is
  what M5 calls a site profile (`cmfqsar.sites.site_autocorrelation`), and combines the two rows
  as their sum and the absolute value of their difference, so that the descriptor does not change
  when the hop is reversed. The kernel is `sites.site_kernel`, the one M5 measured with, on the
  same 1732 labelled hops, the same 10 folds holding out whole structures and the same ridge
  grid; the three cutoffs of the site profile are the three M5 chose between. It gives q2
  +0.439 (4 A), +0.446 (6 A) and +0.462 (8 A), against +0.662 for the rotation-averaged overlap
  kernel in the hop frame. The frame kernel is ahead by +0.216 [+0.109, +0.334] against the 6 A
  site model (6 A is the radius the hop environments themselves use), +0.222 [+0.134, +0.310]
  against the 4 A one and +0.200 [+0.104, +0.322] against the 8 A one, clear of zero in all
  three, so the gate is passed; the rotation-invariant kernel on the same environments passes it
  too, by +0.211 [+0.117, +0.329]. Two readings come with the result. The site-resolved model is
  **not** better than the count of atoms in shells about the midpoint (-0.045 [-0.142, +0.065] at
  6 A, -0.030 [-0.122, +0.068] at 8 A), so describing the two sites well is worth about what
  counting the neighbours of the midpoint is worth, and the path kernel is ahead of both. And the
  shortfall is not the descriptor losing hops to each other: the two end sites name 1195 distinct
  pairs for the 1732 hops and 923 hops share their pair with another hop (a hop and its reverse,
  and two hops through different images of the cell, get one row), but predicting the mean
  barrier of every such group would still reach q2 +0.950 at an error of 0.181 eV, far above what
  any model here reaches. The site description therefore loses the barrier for want of the path
  between the sites, not for want of a way to tell two paths apart, which is the answer the item
  was written to get: the site resolution alone does not carry the barrier.
  **Measured against published numbers, 6 November 2026** (`benchmarks/prepare_litraj.py`,
  `benchmarks/run_litraj.py`, `benchmarks/results/litraj/`). The nebBVSE122k part of the LiTraj
  collection (Dembitskiy et al., npj Comput. Mater. 11, 131 (2025), MIT licence) predicts the
  same quantity this item predicts: 122,421 bond-valence barriers of single Li hops in 11,290
  structures of the Materials Project over 1,644 chemical systems, relaxed with the nudged
  elastic band method instead of taken as the minimax path on a grid. **Its split holds out whole
  structures**, which was read from its index rather than from the paper: none of its 1,172 test
  structures appears in its training part. A kernel holds one number per pair of samples, so the
  run reads a sample of 400 structures with at most 6 hops each, 1801 hops of 393 structures, and
  fits the same models with the same protocol as the public set above. The counts about the
  midpoint give q2 +0.468 (error 1.389 eV), the hop length with the bottleneck +0.179, the two
  end sites of M5 +0.114, the rotation-invariant local kernel +0.600 (1.204 eV, mean absolute
  error 0.726 eV) and the overlap kernel in the hop frame +0.580 (1.234 eV). The published models
  on the same labels reach q2 +0.95 at a mean absolute error of 0.21 eV (M3GNet, a graph network
  that marks the hop midpoint with a dummy atom) and +0.94 at 0.26 eV (a gradient-boosted forest
  on Voronoi features of the hop). Three readings. **The ordering inside this program holds**:
  the field kernels are ahead of the count reference by +0.132 [+0.056, +0.209], so rule 1 is met
  on a chemistry the program was not built on. **The distance to the published models is not the
  size of the training set**: between 200 and 393 training structures the model gains +0.028 in q2
  for every doubling, and the 4.5 doublings up to the 8,945 training structures of the collection
  carry it to about +0.71 at that rate, short of +0.95. **And the two findings of this item hold
  on the new set as well**: the hop frame is again worth nothing in accuracy over the
  rotation-invariant kernel (-0.020 [-0.073, +0.039], as it was +0.005 [-0.060, +0.071] on the
  public set), and the site-resolved descriptor falls from +0.446 on ternary lithium oxides to
  +0.114 here, so describing two end sites carries a barrier only inside one chemistry.

  **What the distance is made of, same day.** The two published models closest to the graph
  network are a random forest and a gradient-boosted forest on hand-made features, not kernel
  models, so the comparison above mixes two questions: is the **description** of a hop behind, or
  is fitting it with a **kernel**? `run_litraj.py` now answers them separately by fitting both
  forests on this program's own descriptions, over the same hops and the same folds, and the
  answer revises the reading above. Both kernels of this program divide every feature row by its
  own length, so a forest given the raw rows would be given more than the kernel ever had; the
  arms are therefore measured twice, on the kernel's own rows and on the raw rows. **On exactly
  the kernel's rows the fitting method is worth a large part of the distance**: the boosted forest
  reaches q2 +0.758 against the kernel's +0.600 on the identical description, and the mean
  absolute error falls from 0.726 to 0.488 eV (+0.158 [+0.110, +0.212]; the random forest +0.742,
  +0.142 [+0.091, +0.198]). The comparison favours the kernel, which is given the ridge that
  maximises the same out-of-fold q2 it reports while the forests search for nothing. So **the
  description of a hop was not what was behind the published models; fitting it with a kernel
  was.** **And the cosine normalisation throws away something that carries the barrier**: adding
  the raw row lengths back gains a further +0.086 and takes the mean absolute error to 0.374 eV
  (q2 +0.844, the random forest), which the kernel cannot use by construction. That is a property
  of `cmfqsar.invariant.autocorrelation_kernel`, not of the hop models, so it is worth testing
  wherever that kernel is used; it is now open as item M15.

  **Can a kernel hold what the forests found, same day.** The kernels of this program are linear
  -- `autocorrelation_kernel` rescales the channels, divides every row by its own length and takes
  the inner product -- so ridge regression on one is a linear model in the feature space while a
  forest is not, and the first guess was that the barrier is not a linear function of these
  features. A Gaussian kernel on the same rows tests that guess without leaving the kernel
  framework, because the rows are unit vectors and the squared distance between two of them is
  `2 - 2K`. These arms are fitted more honestly than the rows above: the width and the ridge are
  chosen inside each training part on an inner split that holds out whole structures, as item
  M14's benchmark does, and the linear kernel is given the same treatment. **The guess is the
  smaller part of the answer.** The width is worth +0.052 [+0.023, +0.080] on the local
  description (nested linear +0.572, nested Gaussian +0.623, mean absolute error 0.736 to
  0.653 eV), which is clear of zero but covers about 30 per cent of the +0.171 between that
  kernel and the random forest on the very same rows; the Gaussian kernel is still
  -0.119 [-0.170, -0.071] behind the forest. On the counts the width is worth nothing
  (-0.015 [-0.060, +0.032]), so this is not a general gain from curving a kernel but something
  the description with many columns has. **The reading for the rest is the metric, and it is a
  reading and not a measurement**: a Gaussian kernel measures one distance over all 2601 columns
  at once, so the few that carry the barrier are diluted by the many that do not and no single
  width repairs it, while a tree picks columns one at a time. That makes M15 a question about
  which columns count and how they are weighted, not about linear against nonlinear. One number
  came out in passing: the nested linear kernel scores -0.029 [-0.048, -0.006] against the same
  kernel in the first table, which is what choosing the ridge on the folds it reports is worth, so
  every kernel row of that table is overstated by about that much and the forest rows by nothing.

  **The window the ion passes through, same day, and it is the repair that works.** A barrier is
  set by the narrowest place along the path: a window of three or four framework atoms. The
  two-point description carries the distance between two neighbours but not the two sides that
  meet at the centre, so it does not resolve the angle there and two different windows can leave
  the same distance histogram -- which is what `benchmarks/run_soap_gate.py` already measured on a
  carbon-shift model. `cmfqsar.triangles` (item C5) resolves it for an environment whose first
  atom is the centre, and the first atom of a hop environment is a marker at the midpoint, so its
  triangles are the midpoint with two framework atoms. The reach, the grid spacing, the exponent
  and whether the row length is kept are chosen inside each training part. **The three-distance
  kernel reaches q2 +0.789 at a mean absolute error of 0.421 eV**, against +0.572 and 0.736 eV for
  the two-point description fitted the same way: **+0.217 [+0.159, +0.286]**, four times what a
  Gaussian width was worth, and +0.209 [+0.145, +0.274] over the overlap kernel in the hop frame,
  which was the best model of this item. It passes the random forest on the two-point rows
  (+0.047 [-0.004, +0.096], and +0.031 [-0.022, +0.083] against the boosted one), so a kernel is
  no longer behind a forest on the description it is given; and **it stays a linear kernel on
  fields, so the co-field survives**, which neither a forest nor a Gaussian width offers. The
  nested arm chose the setting that keeps the row length in every fold, so the two findings of the
  day add together: resolving the angle and keeping the length are different pieces of
  information. The gain does not rest on the grid -- reaches of 3, 4 and 5 angstrom and spacings
  of 0.5 and 0.75 land within 0.003 of each other at power 1, and only the exponent moves the
  figure. Mixing the two-point kernel back in is worth -0.004 [-0.032, +0.028] and the arm chose
  pure three-distance in every fold, so the two-point description adds nothing once the
  three-distance one is there.
  **What is still ahead of it, and the caution.** A random forest on the raw two-point rows is
  still ahead by -0.055 [-0.090, -0.023], so a flexible learner is finding something a linear
  kernel on triangles does not; the arm after the free volume below says what that is, and it is
  the fitting method. The range of the three-distance grid was placed after a first look at this
  same
  sample: nothing was chosen on a test fold, but the grid is not independent of the data, and a
  fresh set would settle it. The published models remain ahead at 0.21 eV (graph network) and
  0.26 eV (forest) on 96,849 hops against these 1801 -- but the distance in mean absolute error
  has gone from 0.726 to 0.421 eV with one description already in the program, and M3GNet's own
  three-body terms are the same ingredient.

  **The published models' strongest feature, as a cell and as a field, same day.** Their feature
  ranking puts the **smallest volume of the Voronoi polyhedra** around the three sites of a hop
  first, and the largest of the three fourth. That is a minimum over atoms, which looks unlike a
  field, but it is the narrow-width limit of a ratio of fields: with `d(i)` the squared distance
  to atom *i* less its squared radius, the radical Voronoi cell of a centre is where `d(centre)`
  is smallest, and `exp(-beta d(centre)) / sum_i exp(-beta d(i))` tends to that cell's indicator
  as beta grows, with the denominator a steric field of unit weights and width `1/sqrt(2 beta)`.
  `free_volume` in the benchmark integrates both over a ball of 3.5 angstrom on a 0.25 angstrom
  grid, which reproduces an exactly known cell to under one per cent; four tests pin it to cases
  with a known answer (no neighbour gives the ball, six neighbours on the axes give the cube of
  their separation, the radii move the boundary, and the narrow width converges to the cell).
  **A field carries the quantity with nothing to spare**: the cell scores q2 +0.465 and the field
  version +0.468 at beta 8, +0.465 at beta 32, and the nested arm chose a field over the cell in
  every fold. Only the widest setting fails (+0.376 at beta 0.5), where one point is shared
  between neighbours. **And the feature is strong in few terms**: four numbers -- the smallest,
  largest and mean volume of the three sites and the midpoint alone -- are level with the whole
  count reference (-0.000 [-0.074, +0.092]) and ahead of the hop length with the bottleneck by
  +0.289 [+0.238, +0.338], though behind the whole two-point description
  (-0.103 [-0.168, -0.039]). **But it adds nothing to the three-distance kernel**:
  +0.010 [-0.005, +0.023], which does not clear zero, so the triangles around the midpoint
  already carry what the free volume says -- as a description of those triangles should. The
  mixture is this item's best kernel model at q2 +0.799 and a mean absolute error of 0.418 eV,
  and it is still -0.045 [-0.080, -0.013] behind the random forest on the raw two-point rows. So
  the published feature is not what this program was missing; what the forest has is accounted for
  by the arm below.

  **And what the forest has is the fitting method, not the description (6 November 2026).** The
  last arm of this item gives a forest the three-distance rows: the same hops, the same folds and
  the same forest settings as every other forest arm here. The random forest reaches q2 +0.842 at
  a mean absolute error of 0.356 eV and the boosted one +0.846 at 0.354 eV, which is this item's
  best model of any kind. **The reading is in two comparisons.** Against the kernel on the same
  rows the two forests are +0.053 [+0.015, +0.091] and +0.058 [+0.029, +0.089], both clear of
  zero, so a flexible learner still finds something a linear kernel on the same description does
  not. But against the random forest on the raw two-point rows the best of them is
  +0.003 [-0.026, +0.028], which does not clear zero: **the better description is worth nothing to
  a forest, while it was worth +0.217 to the kernel.** So the description side of the distance is
  closed. What a forest has over this program on this task is not a description of the hop that
  the fields cannot express; it is the fitting method, and the remainder is about +0.05 in q2 or
  0.06 eV, of which the Gaussian width above had already recovered 30 per cent. For the program
  that is a trade and not a loss, because the field kernel keeps the co-field and a forest does
  not, and it says that the place to look next for accuracy on this task is the fitting method and
  no longer the description. **One row form was measured, not two**: the raw rows, because the
  nested kernel chose the setting that keeps the row length in every fold, so those are the rows
  of the model the forests are compared with, and the normalised form would pair with a setting
  that was never chosen and would repeat the row-length question the two-point arms answered. The
  two arms cost 39 and 27 minutes on 63,036 columns, which is why the set is two and not four.

  Set beside the published numbers,
  this program's description of a hop fitted with a forest reaches 0.354 eV on 1801 hops against
  0.21 eV (graph network) and 0.26 eV (published forest) on 96,849 hops. The
  forest also learns faster: +0.053 in q2 per doubling over the last step of its curve against
  +0.028 for the overlap kernel. The whole run costs 124.7 minutes, of which the two forest arms
  on the three-distance rows are 66 and the overlap kernel 30.
  Nothing here changes a default: the field kernels are what the co-field is read from, and the
  co-field is the reason the frame model exists.
  **Not done:** no model fitted on DFT barriers. When this item was written the barriers it names
  were scattered over papers and not in one table; the collection above now supplies one,
  `nebDFT2k`, with 1681 nudged-elastic-band barriers of 876 structures under the same licence,
  which nothing here has read yet. Measuring on it is the next step for this item, and the first
  chance to fit the program on real migration barriers rather than on an empirical landscape; the
  published reference there is a mean absolute error of 0.25 eV for the best structure-to-property
  model and 0.07 to 0.11 eV for a machine-learned potential that runs the elastic band itself.
  Also no polarisability field, no argyrodite among the thirteen disordered conductors, no
  relaxed path (the bottleneck is on the straight line), and no panel in the application.

**Outside the library**

* **BT8. A slab generator for the adsorption benchmarks.** `examples/uma_fe110_adsorption.py` is
  written for one slab. A version under `benchmarks/` that takes any slab (Li(100), a layered
  oxide facet, graphite) and the pose rules of the Fe(110) script is the generator for the EC6
  benchmark on battery surfaces and for the SEI question (which solvent reduces first on lithium).
  Script code under rule 6, not a module.

**Order.** ~~BT1 and BT3 first, with BT2 as they need it~~: **all three done 3 November 2026**, and
the benchmark they give says the composition decides an average voltage. ~~BT6 with M4~~ **done the
same day**, and it does not pay: the difference of the two cells is
indistinguishable from the lithiated cell alone and behind the element fractions. ~~BT4~~ **done
4 November 2026** (the builder works, the gate is met by 0.010 eV, and the cation frame buys the
co-field rather than accuracy). BT4 was next, since
BT5 is empty without a complex that builds and aligns. BT6 with M4, ~~BT7 after M5~~ (BT7 is done
by the
local route, 30 October 2026, and did not
need M5; its gate was applied against an M5-style model of the same hops on 6 November 2026 and
is passed, and the same day the models were measured against published numbers on a public
benchmark of the same labels, which found that the kernel and not the description is what the
distance to those numbers is mostly made of and opened item M15), ~~BT5~~ **done 4 November 2026** (shells sampled from a bulk electrolyte do beat BT4's
single optimised complex on the activation energy, by 0.32 to 0.55 kJ/mol after item D11's charge
correction, clear of zero in three readings of four, and do not beat it on
the conductivity; the cation frame costs an ensemble 0.7 to 1.0 kJ/mol, far more than it costs one
complex, so the co-field and the accuracy pull against each other), BT8
whenever EC6 is ready for it.

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
  **The gate for the richer descriptor, run 31 October 2026, and it passes.**
  `benchmarks/run_soap_gate.py` computes the SOAP power spectrum (Smooth Overlap of Atomic
  Positions; one Gaussian density per element around the atom, from the dscribe package) on
  the same carbons and puts it through the same ridge on the same folds. SOAP resolves the
  triangle of the centre and two neighbours; the two-point kernel here resolves only its
  third side. On the 5096 carbons, SOAP at angular order 0, which is two-body like ours,
  matches the two-point kernel (+0.004 [-0.001, +0.008] in q2); the angular orders then take
  the error from 8.15 to 6.62 ppm at the same density width and kernel exponent (+0.008
  [+0.006, +0.011]), and a narrower density with a higher exponent to 5.72 ppm, against 8.82
  for the two-point kernel. That is a larger gain than the 1.99 ppm two halvings of the
  distance bins gave at 6000 molecules, on one fifteenth of the data. On the 1225 aromatic
  carbons of LP3 the sequence over angular order is the same (4.94 to 3.81 ppm; +0.057
  [+0.045, +0.070]) and the best SOAP, 3.50 ppm, is better than the posed overlap kernel in
  its frame, 4.22 ppm (+0.031 [+0.015, +0.052]). **The angular term is the missing piece.**
  The next step is a CMF form of it: the pair sum of the environment resolved over the two
  distances from the centre as well as the distance between the pair, with the field weights
  as the channels in place of the elements. That is what this program would add to SOAP, and
  the gate did not measure it. Results in `benchmarks/results/soap_gate/`.
  **The CMF form of the angular term, built and measured 31 October 2026.**
  `cmfqsar/triangles.py` is the three-distance autocorrelation: the pair sum of the environment
  resolved over the two distances of a neighbour pair from the centre and the distance between
  them, with one weight per field as the channels (`cmf local --triangles`; `--r-max`,
  `--grid-step`, `--grid-sigma`, `--scaling`, `--power`). Three sides fix a triangle, so it
  carries what the SOAP power spectrum carries, in a different basis. `benchmarks/run_triangles.py`
  measures it on the carbons and folds of the gate, with the element indicators of SOAP as an
  alternative set of channels through the same code. Settings chosen on the sets: cutoff 5 A,
  grid step 0.5 A, window 0.35 A, kernel power 1. On the 5096 carbons the five fields reach
  **5.45 ppm against 5.72 for the best SOAP** (+0.001 [-0.001, +0.003] in q2, the same model)
  and 8.82 for the two-point kernel (+0.017 [+0.014, +0.020]); the element channels give 5.47
  ppm (-0.000 [-0.002, +0.002] against the fields), so the basis reproduces SOAP, and fields
  with elements together reach **4.83 ppm**, better than SOAP (+0.003 [+0.002, +0.005]) and than
  either set of channels alone. On the 1225 aromatic carbons the fields reach 4.13 ppm against
  5.19 for the two-point kernel (+0.056 [+0.033, +0.079]) but stay behind SOAP, 3.50 ppm (-0.027
  [-0.050, -0.010]); the element channels in this basis give 3.67 ppm, inside SOAP's interval
  (-0.007 [-0.015, +0.001]), and adding the fields to them gains nothing (3.81 ppm; -0.006
  [-0.035, +0.011]). So the form is right and the fields are a good set of channels on the
  broad set, where they add to the elements, and a weaker set on the aromatic carbons, where
  the shift follows the substitution pattern more closely than the field weights express it.
  A window of 0.35 A is better than 0.5 or 0.7 A on both sets; raising the kernel to a power,
  which SOAP needs, gains nothing here; scaling the channel blocks per field is best for the
  field channels and no scaling for the element channels. The 5 A cutoff costs 32 340 channels
  for five fields and 128 205 for ten, and ten to twenty-five minutes per cross-validation on
  the 5096 carbons where the two-point kernel takes one; 4 A, the default, loses 0.02 ppm on the
  shift set and nothing on the aromatic carbons. What is still open is the ShiftML comparison
  the item started from, which needs its DFT shift sets. Results in
  `benchmarks/results/triangles/`.

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
  that differs between hosts. **The cation-specific radii of EC7 are done (4 November 2026)**, so
  what is left here is the frame convention and the data.
  Measure on log K of Na⁺ and K⁺ for crown ethers and cryptands in methanol from the Izatt
  tabulations, the electrostatic field against the probe field, and the Na⁺/K⁺ selectivity with
  SC1. No machine-readable open copy of those tabulations was found on 4 November 2026 (the NIST
  stability-constant database is licensed), so this item begins with entering log K from the
  papers by hand. EC7 also measured what to expect of the selectivity half: on an ether series the
  difference between two monovalent ions was barely predictable, and a second +1 probe beside the
  first carried no information the first did not. A host series has a physical frame, which few drug series have, so the co-field is
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

### Physical organic chemistry: a planar series in a frame given by rule (29 October 2026)

The COMPAS-1 study (`cmfqsar/pbh.py`, `benchmarks/run_compas.py`, measured in
`benchmarks/results/compas/`, manual section *A frame by rule: the gap of polybenzenoid
hydrocarbons*) is the first CMF model of a quantum-chemical property on an enumerated chemical
space: the HOMO–LUMO gap of the 8,678 cata-condensed polybenzenoid hydrocarbons of COMPAS-1D
(Gershoni-Poranne group, 2022). It came from an outside brief — a one-page proposal for a
computational-chemistry group — and was built inside the repository rather than as the separate
package the brief proposed, because every piece but five already existed. What it added:

* a planar frame by rule (`canonical_frame`): the inertia axes of the heavy atoms, the signs of
  the in-plane axes fixed by a fixed sequence of odd moments, and the third-, fourth- or
  sixth-moment direction where the in-plane moments are degenerate; the kernel-based choice
  among the four frames (alignment by prediction confidence) as an option;
* the typing scheme `pbh` (rim carbon, two-ring carbon, three-ring carbon) for the indicator
  fields, on the command line and in the application;
* **ring-centred fields**: one Gaussian per ring centroid, in one field per annulation class
  (terminal, linear, angular, branched) or weighted by the Clar-sextet share, carried as
  pseudo-atoms of the molecular field so that the kernel, the co-field and rule 3 need no change
  (`ring:<class>` in a field set, `FieldSet.create(ring_classes=...)`);
* rule 3 summed per ring (`ring_contributions`), and `mae` beside `rmse`;
* a COMPAS reader that matches the SMILES to the xyz atoms by graph and refuses a mismatch.

What it settled. The gap of a benzenoid is topology: the steric field alone reaches a test R² of
0.42 (five random 80/10/10 splits), the carbon-class indicator fields 0.49, the four ring-class
fields alone 0.65, and the four groups together 0.78 (q² 0.83, MAE 0.13 eV); the Clar-sextet
ring field on top takes that to 0.87 (q² 0.87, RMSE 0.14 eV, MAE 0.11 eV). On the
size-extrapolation splits — train on fewer than N rings, test on N — the sets without a ring
field have negative R² and the sets with one are positive (0.44 to 0.78), which is the one
result a proposal can rest on: the ring field is what carries over to larger molecules. The
frame costs nothing: a planar molecule needs no alignment step and the rule leaves nothing to
fit. The co-field of the ring-class fields over a six-ring reference molecule and the per-ring
contributions of four test molecules are drawn by the benchmark.

What it leaves open, in order:

* **PBH1. The Nyström path on COMPAS-1x, done** (34,072 molecules at GFN2-xTB, `--stage
  scaling`): with the attenuations and ridge of the COMPAS-1D model, 500 landmark molecules give
  q² 0.78 and test R² 0.80, and 1,000 and 2,000 give the same to three decimals while the rank
  of the feature map grows only from 2,244 to 2,722. The fields of planar molecules in one frame span a
  few thousand directions, so the 34k set is a quarter of an hour of kernel evaluation against a
  few hundred landmarks. Two things the run taught: the landmark path has to run on the compiled
  kernel engine (the NumPy path is single-threaded, and `fit_low_rank` builds the feature map
  three times), and the attenuations were not searched on the 34k set.
* **PBH2. A NICS-weighted ring field, replaced by a geometric one and calibrated (30 October
  2026).** A NICS field for the set is a week of serial ORCA on one desktop (5.5 min per six-ring
  molecule), so the field built is the HOMA index of every ring from the bond lengths of the DFT
  geometry (`ring:homa`, group F), and NICS(1) was computed for the 179 molecules of up to seven
  rings as its calibration (`--stage nics`): r = −0.77 between HOMA and NICS(1)zz over 1,158
  rings, −0.50 for the Clar share; the linear rings are the most shielded by NICS and not the
  most bond-equalised by HOMA, so a NICS field would weight them more. The HOMA field beats the Clar field in
  the linear model by the MAE (0.097 against 0.108 eV) and on the size splits (R² 0.83 against
  0.70 on the ten-ring molecules); the two together under the Gaussian layer give 0.050 eV MAE
  and R² 0.93 on the ten-ring molecules. The sum of the HOMA indices added to the counts takes
  the additivity scheme from 0.70 to 0.80. What is left is the NICS field itself on a cluster,
  and HOMA parameters for bonds to boron.
* **PBH3. Peri-condensed systems (COMPAS-3), done (30 October 2026)**: 8,844 molecules, the
  `C.3r` carbons and the fifth annulation class `peri` (every molecule holds one). The picture
  of COMPAS-1D repeats: ring classes alone 0.46, with the Clar field 0.83, with Clar and HOMA
  0.90 (linear), 0.097 eV RMSE and 0.072 eV MAE with the layer, R² 0.86 on the 6,891 ten-ring
  molecules from 1,953 smaller ones (descriptor forest −2.3). The linear four-group model is
  ill-conditioned on this set (a float64 refit moves one test prediction by 0.89 eV), so its
  per-ring reading is not usable here.
* **PBH4. Hetero-atom systems (COMPAS-2), measured on a 10,000-molecule subset (30 October
  2026)**: atom types N, N.H, O, S, B, B.H, one ring field per ring type (group R), the HOMA
  field, no Clar field. The best layered model explains 69 % of the variance (0.38 eV RMSE on a
  spread of 0.68 eV; 0.60 on the ten-ring molecules, against 0.56 and 0.43 for the descriptor
  forest and 0.51 and 0.36 for the additivity scheme); the steric attenuation goes to the
  narrow end of its grid and the ridge to a hundred times that of the benzenoid sets or more,
  because the inertia frame depends on the masses and molecules of
  one skeleton with different hetero atoms do not lie on top of each other. The next item on this
  set is a frame from the skeleton, not from the masses; the full 52,000 go through the landmark
  path unchanged.
* **PBH5. Δ-learning xTB → DFT, done in the residual form (30 October 2026)**: a straight line
  in the GFN2-xTB gap leaves 0.047 eV RMSE and 0.036 eV MAE; the layered field model on the
  residual reaches 0.026 ± 0.002 eV RMSE and 0.018 ± 0.001 eV MAE (0.030 eV on the ten-ring
  molecules), inside the range of the published graph networks for one xTB calculation per
  molecule. The multi-task form (`cmfqsar/multitask.py`) was not run.
* **PBH6. The grids, which is now the first item.** The published geometric graph networks
  (Weiss et al., J. Org. Chem. 2023, Table 1) reach a test MAE of 0.011 to 0.022 eV on the same
  8,678 molecules and a random split like ours; the sequence model on augmented annulation
  strings (Fite et al., J. Phys. Org. Chem. 2023) 0.05 eV; the field models here 0.11 to 0.13
  eV, with the steric attenuation at the top of its grid and the ridge at the bottom (1e-4) for
  every set with a ring field. The gap is fixed by the ring arrangement to a precision that the
  overlap of wide Gaussians does not resolve. Measured the same night (`--stage resolution`):
  the attenuation grid extended to 0.02 to 20 Å⁻² and the ridge down to 10⁻⁶ change nothing
  (0.11 eV), and the Gaussian kernel layer on the same kernel halves the error to 0.056 ± 0.004
  eV MAE, R² 0.964 ± 0.005 over three seeds, level with the annulation-string network and
  ahead of the descriptor forests, three to five times behind the graph networks. The
  resolution was the nonlinearity, not the width. On the size-extrapolation splits the layer
  gains as much: trained on the 2,270 molecules of up to nine rings, the layered five-group
  model predicts the 6,408 ten-ring molecules with R² 0.92 and RMSE 0.113 eV (linear 0.70 and
  0.213 eV; the descriptor forest −0.16). The layered model gives up the co-field and rule 3;
  what is left to measure is Shapley values as its per-ring reading and a wider grid of layer
  widths (the chosen gamma sat at the widest end on every split). Nothing published measures
  the size-extrapolation split, which stays the field model's own claim.
* **PBH7. Planarity and reflection.** Measured on 30 October 2026: 89 % of the COMPAS-1D
  molecules have a heavy atom more than 0.5 Å from their best plane (median 1.48 Å), most
  ten-ring cata-condensed isomers holding a fjord; COMPAS-3D median 1.21 Å. The frame is defined
  for them, but a mirror image lands in the mirrored frame and the model is not symmetric under
  reflection although the gap is. A kernel averaged over the two faces (the z-flip of one
  molecule) is one line and was not run.

* **PO1. A catalytic series with a shared frame, done 5 November 2026.** The catalysis bullet of
  H named ligand libraries with a shared frame and a rate or a selectivity as the activity, and
  EC4, EC8 and EC9 recorded that no such series could be found in machine-readable form. One
  was: the screen of Zahrt, Henle, Rose, Wang, Darrow and Denmark (Science 363 (2019) eaau5631),
  43 BINOL-derived chiral phosphoric acids on 25 imine–thiol combinations, 1,075 N,S-acetal
  formations with their enantiomeric excess, public in the DeepReac repository
  (`docs/public_3dqsar_datasets.md`). The target is ΔΔG‡ at 298.15 K, the difference of two
  rate constants, so this is the first measured kinetic selectivity in the repository.
  `examples/make_denmark_examples.py` assembles the catalysts from the published fragments,
  gives all 43 the same axial configuration and places the phosphoric acid in a frame (phosphorus
  at the origin, acid oxygens on +z, the two C2 images as an ensemble); `benchmarks/run_denmark.py`
  is the two-endpoint model of SC1 with 25 endpoints, the covariance by the evidence, and the
  baseline that predicts the same thing without a field on every split. **Measured:** each
  catalyst held out in turn, MAD 0.224 kcal/mol (q² 0.82) against an endpoint-mean baseline of
  0.458 (0.17), difference −0.236 [−0.340, −0.142] over 1000 catalyst bootstraps; reactions
  10-fold, 0.148 to 0.164 against the additive catalyst + endpoint model at 0.183; each substrate
  combination held out, 0.158 against 0.186. The published model reached 0.211 on 19 held-out
  catalysts trained on 24 and 0.161 on new substrate combinations, so the field model is at the
  published accuracy on both kinds of split. **The frame is the result:** on the catalyst holdout
  the steric field alone gives q² 0.76, the hydrophobic 0.63, the electrostatic 0.41, and the same
  fields through the rotation-invariant kernel 0.26; the co-field, averaged over the endpoints,
  puts positive steric weight at the 3,3' ipso carbons flanking the acid and negative weight 3 to
  5 Å over the pocket (manual section *A kinetic selectivity as the activity*, README of the
  same name). **Not done:** a thiol no catalyst was run on is not predicted (0.406 against a
  baseline of 0.399), because the one-hot substrate descriptors know nothing about an unseen
  thiol and the aliphatic and aromatic thiols differ by 0.6 kcal/mol in their means; a field of
  the substrate aligned on its reacting group is the form that could carry it. The two worst
  catalysts have a methylene or a silicon between the scaffold and the aryl group, a conformer
  choice the lowest MMFF structure makes once; the ensemble of D8 on this series is one command
  (`--ensemble`, not yet written for this builder). The sign of the axial configuration is
  consistent across the 43 but not known to be (R) or (S); a co-field drawn for a design would
  need the enantiomer stated. EC9's potential-dependent target is not touched: this series has one
  condition.
### CMF–neural device integration (30 October 2026)

**Goal.** Reuse the Continuous Molecular Fields representation inside the GNN/QSPR framework
(the neural device), so that alignment-based 3D field information and the graph-based model can
be combined and ablated on one data set.

**Rationale.** CMF is a fixed-kernel linear model on field space: it is the infinite-width limit
of a radial-basis-function network whose neurons tile 3D space. The neural device is the learned,
feature-learning end of the same spectrum. Merging the two means letting the device learn what
CMF currently fixes.

**Stages.**

* **ND1. CMF as a featuriser.** Compute the CMF kernel rows of every molecule (or a Nyström or
  random-feature approximation of them) and feed them as an input vector to the existing model.
  Baseline: CMF alone, the device alone, and the two combined.
* **ND2. A learnable CMF layer.** Make the Gaussian widths and the mixing weights of the field
  types trainable end to end; keep the regression-coefficient field (the co-field) exportable as
  a 3D map, so that the reading is not lost.
* **ND3. Field-to-graph fusion (optional, later).** Sample the field values at the atoms or at
  probe points and use them as node or edge attributes of an equivariant GNN, which removes the
  superposition requirement.

**Data set.** COMPAS-1, which is large enough for stages ND2 and ND3 and reuses the specification
of the CMF pilot above.

**Open questions.**

* The alignment protocol for stages ND1 and ND2.
* Which field types to include first: steric, electrostatic, hydrogen bond, CIF.
* How to quantify the loss of interpretability from one stage to the next.
* The interface between the R package and the Python/PyTorch device: export the kernels, or
  reimplement the field kernels in Python (`cmfqsar/kernels.py` already holds them in Python, so
  the export route is the one left to write).

**ND4. Transformers inside CMF (30 October 2026).** The CMF kernel is already a form of
attention. The kernel between two molecules is a double sum over their atoms, where each atom
pair contributes the product of its two atom coefficients times a Gaussian of the distance
between the atoms. A transformer's attention layer is the same object with the pair score learned
and a softmax added (the linear-attention form is a kernel method; see Tsai et al., EMNLP 2019,
and Choromanski et al., ICLR 2021). The question is therefore not whether a transformer fits, but
which part of the model it is allowed to learn. The four options below are ordered by how much of
CMF they keep: the closed-form kernel, the co-field and the exact Shapley values.

* **ND4a. A transformer that produces the atom coefficients.** Today the per-atom values are
  fixed: MMFF charges, van der Waals radii, hydrogen-bond strengths. A transformer on the
  molecular graph, or an equivariant 3D transformer (TorchMD-Net, Equiformer), produces them
  instead. The kernel, the co-field and the Shapley decomposition are untouched, because the
  network only produces a field; this is ND2 with the architecture named, and it stays inside
  the E11 exception, since the model remains the kernel model. End-to-end training is cheap
  because the landmark features of `cmfqsar/lowrank.py` make the ridge solution a
  differentiable function of the atom coefficients; the Gaussian-overlap kernel of
  `cmfqsar/kernels.py` is a few dozen lines in PyTorch. *The small-data constraint:* on a
  congeneric series of a few dozen molecules a transformer with thousands of parameters
  overfits. The remedy is to pretrain it on a per-atom quantity the program already computes
  without activity labels (xTB or ORCA charges, `cmfqsar/xtb.py` and `cmfqsar/orca.py`), then
  fine-tune it as a correction to the MMFF values with a penalty toward zero.
* **ND4b. A learned atom-pair score inside the kernel.** Replace the product of two fixed
  coefficients by the inner product of two learned atom embeddings. The result is still a
  positive-definite kernel, because it is the product of a graph-embedding kernel and the
  spatial Gaussian kernel. The co-field survives, but it becomes a field over space and
  embedding directions, so the embedding must be limited to a few named channels to keep the
  3D reading.
* **ND4c. Learned probe tokens.** The landmark construction of B6 is one cross-attention layer:
  the atoms of a molecule attend to fixed points in the frame, with the Gaussian as the
  attention score and a linear readout. Making the landmark positions and widths trainable and
  adding depth gives a set transformer. The co-field reading survives at the first layer only.
* **ND4d. A full 3D transformer as the regressor.** An equivariant transformer on the atoms,
  with no superposition, is ND3. It gives up the co-field and the exact Shapley values, which
  is why "What is not planned" excludes it as the model, and it is data-hungry. It makes sense
  only on large sets such as COMPAS-1.

Options ND4a to ND4c still need superposed molecules; only ND4d removes that requirement.

**Order.** ND4a first, measured on COMPAS-1 and on one Sutherland set against the fixed MMFF
coefficients inside the same kernel. It answers the one question that matters: do learned atom
values improve on the physics-based ones? ND4b is the second step. ND4d stays excluded for the
medicinal-chemistry use case. *Gate:* ND4a continues only if it improves on the fixed
coefficients on both sets by more than the cross-validation noise; otherwise rule 5 applies and
the result is recorded.

**ND5 to ND10. Six more deep-learning methods, ordered by the CMF cost each removes (30 October
2026).** The four options of ND4 ask which part of the model a network may learn. The six items
below leave the model alone and put a network at one of the places where the method is expensive
or weak: the quality of the geometries and charges, the superposition, the small size of a
series, and the inverse step from a co-field to a molecule. Every one keeps the kernel as the
activity model, so rule 2 holds and the co-field and the exact Shapley values survive. Every one
follows the optional-dependency rule of E11: PyTorch and any model weights are optional, and the
module is absent rather than broken when they are missing.

* **ND5. Machine-learned interatomic potentials for the geometries and charges.** The fields are
  computed from MMFF conformations and MMFF charges, and the xTB study of part C showed that
  better charges help a minority of series. A machine-learned interatomic potential (MACE, UMA,
  ANI) gives a conformation of DFT quality at a cost close to MMFF, and a charge model trained on
  DFT (a charge-equilibration or multipole head of the same potential) gives charges to match.
  Nothing in the model changes: the item is a third source of geometries and charges next to xTB
  and ORCA, filled into the same `prop:` fields. It also supplies the pretraining target that
  ND4a needs, so it is worth running before ND4a. *Needed:* `cmfqsar/mlip.py`, which relaxes a
  conformation and returns charges with the caching and failure handling of `cmfqsar/xtb.py`;
  `benchmarks/run_mlip.py`, built on `run_xtb.py`. *Gate:* on the eight Sutherland sets, the
  five fields from the potential's geometries and charges against the MMFF reference of rule 1
  and against the xTB row, with the interval of rule 4. The potential becomes a default source
  only where it beats xTB at a lower cost per molecule; otherwise it is recorded as a third
  charge source.
* **ND6. Amortised alignment.** Superposition is the step that scales worst, and the one that
  ND4a to ND4c still need. The pose protocol of C1 and the pocket placement of C3 produce a
  judged pose for every training molecule, and those poses are a training set. An equivariant
  network of the kind used for pose prediction (EquiBind, Uni-Mol) trained on them predicts the
  pose of a new molecule in one forward pass. The pose search then starts from the prediction and
  refines it, or runs in full only where the network's own score is low. The training loss is the
  kernel distance between the predicted pose and the judged one, whose gradient
  `cmfqsar/gradients.py` already holds. *Needed:* `cmfqsar/posenet.py`; `benchmarks/run_posenet.py`,
  with the C1 protocol of `run_poses.py` as the judge. *Data:* the docked pose sets of C1 where a
  series has more than a few hundred ligands; COMPAS does not need it, because its frame is given
  by rule. *Gate, in two parts:* (i) on held-out ligands the predicted pose is within the RMSD the
  C1 protocol accepts for a fraction of ligands large enough that prediction plus refinement is
  cheaper than the full search, with the factor stated with the measurement; (ii) q² on the
  predicted poses is within the interval of rule 4 of q² on the searched poses. If (ii) fails the
  network is a starting-point generator for the search only, recorded under rule 5.
* **ND7. The graph neural tangent kernel as a kernel view.** The rationale above calls CMF the
  infinite-width limit of a radial-basis network. The same limit exists for a graph network: the
  graph neural tangent kernel (Du et al., NeurIPS 2019) is the kernel of an infinitely wide graph
  network at initialisation, computed in closed form from the bond graph with no training at
  all. Mixed with the five field kernels by the multiple kernel learning of E6, it adds what the
  bond graph knows and the aligned field does not, at no data cost. The model stays a kernel
  model and every interpretation tool works; the Shapley value of a graph view is per molecule
  rather than per point in space, and the co-field is read from the field views alone. *Needed:*
  `cmfqsar/gntk.py`, a few hundred lines of NumPy with no new dependency; a `gntk` view in
  `cmfqsar/mixing.py`; `benchmarks/run_gntk.py`. *Gate:* on the eight Sutherland sets and on
  COMPAS-1, the five fields plus the graph kernel against the five fields alone and against the
  five fields plus the invariant kernel of C5, with the interval of rule 4. A gain that the
  invariant kernel already gives is not a result.
* **ND8. Pretrained 3D representations as a kernel view.** E11 (iv) trains an encoder on
  unlabelled molecules of the same frame, and E10 found that unlabelled molecules close to a
  congeneric series gain nothing. A pretrained equivariant model (Uni-Mol, or the representation
  layer of a MACE or UMA potential) has already seen millions of molecules, and a Gaussian kernel
  on its per-molecule embedding enters the E6 mixing as one more view. This is the practical form
  of the semi-supervised item, with the unlabelled set large and varied, which is the case E10 did
  not cover. There is no fine-tuning: the embedding is fixed and the kernel width is the only
  parameter, so nothing overfits on thirty molecules. *Needed:* `cmfqsar/pretrained.py`, with one
  cached embedding per molecule; `benchmarks/run_pretrained.py`. *Gate:* as ND7, with one extra
  comparison against ND7, because the two views carry overlapping information; where they tie,
  the cheaper one (ND7) is kept.
* **ND9. Diffusion models guided by the co-field, for inverse QSAR.** E11 (iii) decodes a field
  into pseudo-atoms and then matches fragments to them. A 3D diffusion model over atom positions
  and types (an EDM- or DiffSBDD-type generator) can instead be steered during denoising by the
  gradient of the kernel distance between the field of the current structure and the target
  co-field, in the way such models are steered by a pocket. The steering needs no training,
  because the gradient is the one of `cmfqsar/gradients.py`; only the unconditional generator
  needs training, and a published pretrained one serves. This is the item most likely to turn
  the co-field from a picture into a design tool. It is the generative counterpart of MC1 and
  MC7, which rank existing substituents and fragments rather than make new ones. *Needed:*
  `cmfqsar/fielddiffusion.py`, with a `cmf generate` entry that takes a fitted model and a target
  region; `benchmarks/run_fielddiffusion.py`. *Gate:* a generated molecule must be judged by a
  model that did not steer it. Fit the steering model on one half of a series, generate, and
  score the molecules with the model fitted on the other half; the gain over unsteered generation
  must exceed the interval of rule 4, and the molecules must be chemically valid and pass a
  synthetic-accessibility filter (the SA score of Ertl and Schuffenhauer 2009, below a stated
  cut-off). Without that the item is a display tool under rule 5.
* **ND10. Predicted electron densities as the electrostatic field.** The electrostatic field is a
  sum of Gaussians on point charges, and point charges cannot reproduce the field above an
  aromatic ring or at a lone pair, which is where the ring work of part H put the chemistry. A
  network that predicts the electron density or atomic multipoles from the geometry (a DeepDFT-type
  density model, or the multipole output of a potential trained for it) gives a field with these
  features at the cost of one forward pass. The field enters as a sixth field type with its own
  mixing coefficient, in the way D9 adds a solvent field, and the kernel stays in closed form
  (rule 3) when the density is represented by Gaussians on atoms and bond midpoints with per-site
  coefficients, which is what the multipole output gives after projection. *Needed:*
  `cmfqsar/densityfield.py`; `benchmarks/run_densityfield.py`. *Gate:* on COMPAS-1 and on the
  Sutherland sets where the xTB charges helped, the five fields plus the density field against
  the five fields, and against the five fields with xTB charges, with the interval of rule 4. It
  replaces the point-charge electrostatic field only if it beats it on every set where the point
  charges were measured; otherwise it is a sixth view.

**Order.** ND5 first, because it changes inputs the existing benchmarks score at once and gives
ND4a its pretraining target. Then ND7, which is a few hundred lines of NumPy with no new
dependency. Then ND6, the item that makes the method usable on a library. ND8 to ND10 wait for
the larger sets and for the ND4a gate.

**ND11 to ND13. Normalizing flows (30 October 2026).** A normalizing flow is an invertible
network with an exact likelihood and an exact inverse. CMF has one use for each of these two
properties and one for the degenerate one-dimensional case, which is the only flow that fits a
congeneric series. A flow that warps field space before the Gaussian kernel is not among them:
the kernel would stay positive definite, but the co-field would live in the warped coordinates
and lose its reading. A Boltzmann generator for conformer ensembles is not among them either,
because the fourth-dimension verdict says ensembles do not help, so it has nothing to improve.

* **ND11. A warped Gaussian process: a one-dimensional flow on the activity.** A monotone
  invertible transform of the response learned together with the GP (Snelson, Ghahramani and
  Rasmussen, NIPS 2003) handles a skewed or censored pIC50 scale with a handful of parameters,
  so it fits thirty molecules. The model stays linear in the field in the warped units, and the
  co-field is read in those units, which the output must say. *Needed:* a `--warp` option of the
  GP fit in `cmfqsar/model.py`, the transform stored in the model file so that `predict` and the
  conformal interval of `cmfqsar/conformal.py` report in the original units;
  `benchmarks/run_warped.py`. *Gate:* on the eight Sutherland sets and the 86-set collection,
  q² and the interval coverage and width of the warped GP against the plain GP and the conformal
  interval, with the interval of rule 4; a series whose activities are already close to normal
  must lose nothing. The item is kept as a default only if coverage improves on more series
  than it worsens; otherwise it is an option, recorded under rule 5.
* **ND12. A flow on the landmark features as the density of a series in field space.** A flow
  on the landmark feature vector of B6, which has the fixed dimension a flow needs, learns the
  probability density of the training molecules in field space, and a new molecule gets a
  likelihood under it. This is a proper density, which none of the five scores of
  `cmfqsar/oneclass.py` is (they give a boundary or a distance) and which the GP variance is not
  either. *Needed:* `cmfqsar/fieldflow.py`, PyTorch optional, with a `flow` score next to the
  five of `cmfqsar/oneclass.py`; `benchmarks/run_fieldflow.py` under the protocol of E5 (Spearman
  ρ against the held-out error, risk–coverage efficiency, and the two-bin RMSE). *Data:* the
  docked pose sets of C1 with more than a few hundred ligands and COMPAS-1; on thirty molecules a
  flow is underdetermined, and the gate must show the point at which it stops being so. *Gate:*
  the likelihood ranks the held-out errors better than the GP variance and the best one-class
  score, with the interval of rule 4, on the sets where it is trained on more than a stated
  number of molecules; below that number the item is not used, and the number is recorded.
* **ND13. Flow matching as the generator of ND9, with design in latent space.** An equivariant
  flow over atom positions and types is invertible, so the ridge model on the decoded field is a
  differentiable function of the latent code, and design becomes gradient ascent of the
  predicted activity in latent space followed by an exact decode. That is what E11 (iii) wants and
  what a diffusion model cannot do exactly, because diffusion has no encoder. The classical
  discrete flows (RealNVP, Glow) do poorly on 3D molecules; the equivariant flow of Satorras et
  al. (NeurIPS 2021) works, and the form in use today is flow matching (Lipman et al., ICLR
  2023; FlowMol and EquiFM for molecules), which trains like a diffusion model and samples in a
  few steps. Those few steps are its gain over ND9 at library scale. *Needed:* a second generator
  in `cmfqsar/fielddiffusion.py` behind the same `cmf generate` entry, with a `--latent-ascent`
  mode; the benchmark of ND9 extended with a flow row. *Gate:* the gate of ND9, applied to both
  generators on the same series, plus the cost per accepted molecule; the two share one gate and
  the one that passes at the lower cost is kept, the other recorded under rule 5.

**Order.** ND11 first, because it runs on the existing sets and gates on the calibration the
program already measures. ND12 after ND5 to ND7, on the larger sets. ND13 with ND9.

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
  **Built and measured 31 October 2026. The gate fails on its own background, passes on a harder
  one, and the method is kept under rule 5 for what the measurement makes it: a diagnosis and not
  a better ranking.**
  **What was built.** `kernels.gaussian_product` (the Gaussian expansion of the pointwise product
  of two expansions) and `kernels.expansion_overlap` (the overlap of two expansions, the module's
  own pair formula written for bags of Gaussians that need not be molecules); `cofields.CofieldMask`,
  `cofields.top_atom_mask` and `cofields.masked_cofield`; `cmfqsar/maskedmatch.py` with
  `MaskedScorer`, `score_candidates` and `screen_conformers`, which keeps the best conformation of
  every candidate. Three numbers per candidate: **inside** (the part of the prediction that comes
  from the masked region), **outside** (the rest -- the complement the item asks for, obtained as
  the unmasked overlap minus the masked one, so that `1 - m` never has to be represented) and a
  **cosine** in the inner product the mask defines, which does not grow with the size of the
  molecule. 16 tests, each against something independent: the product against a dense evaluation
  of both factors, its integral against the closed-form overlap, the overlap against the
  quadrature already in the module, and `inside + outside + bias` against the model's own
  `predict`, which is rule 3 holding.
  **The measurement** (`benchmarks/run_maskedmatch.py`, `results/maskedmatch/table.md`): a model
  on the largest chemotype of a Sutherland set, and the molecules of every other chemotype that
  are at least as active to be ranked above a background. Against the posed presumed inactives --
  the gate's own background -- the unmasked field similarity is already at its ceiling (AUC 0.918
  mean, 1.000 on two sets) because those molecules come from other targets and differ in size and
  shape, so the gate cannot distinguish two measures and the mask, which throws size away, is
  worse (0.584). Against the weakly active molecules of the same other chemotypes, where nothing
  but the activity differs, the masked cosine reaches 0.729 and the masked overlap 0.743 against
  0.566 for field similarity: **+0.163 AUC, ahead on 3 of 4 sets**. Neither beats the model's own
  prediction (0.718), and on the two sets whose single-chemotype model has a positive q² (GPB
  +0.587, AChE +0.096) the prediction is ahead of the masked cosine on both.
  **What is kept, and why.** The inside/outside split behaves as the item predicted on every set
  and every mask size -- the active molecules of the unseen scaffolds collect more of their
  prediction inside the mask than the weak ones (GPB +0.477 against +0.121, AChE +0.309 against
  +0.078) -- so the two numbers say how much of a candidate's promise the model has evidence for.
  That is what the construction is for; ranking by it is not better than predicting.
  **Not done:** no subcommand and no panel (the mask is in the library and in the benchmark
  only); the mask is chosen by a count of atoms and not by a share of the prediction, which would
  be the better rule; and the one measurement that could still overturn the verdict is a series
  with a *measured* scaffold hop -- two scaffolds with the same activity and a published pair --
  which is a data task of the same kind as MC3.
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

**How to read this.** The table says what the *entries* say, and nothing else. An item records
its own completion in bold with a date -- "Done 27 October 2026", "Built 27 October 2026",
"measured 1 November 2026" -- so an item named below as not finished is one whose entry carries no
such statement, and one marked *partly* carries a date for a part of itself and says in its entry
what is left. Five entries do not follow that convention and were read by hand: E11 (version (i)
of four measured), EC8 (a pilot measured, without the cut-and-terminate step and without M5), SC5
(the radii of EC7 are done and the frame convention and the data are left), and PBH1 and PBH2,
which record completion in their titles.

The previous version of this table was written by hand and had drifted far enough to mislead: 29
items whose entries record a completion date were still listed in it as pending. It is now
generated from the entries by `docs/plan_status.py`, which prints exactly the rows below, so the
way to keep it true is to regenerate it rather than to edit it:

    python docs/plan_status.py            # these rows, ready to paste
    python docs/plan_status.py open       # only what is not finished, item by item

That script also records the two things that make reading the file by machine unreliable: the
**Order.** paragraph at the end of each section names every item's completion date, so an item's
own text has to stop before it, and five entries state their status in a form of their own.

| Part | Items | Finished | Not finished |
|---|---|---|---|
| A. The platform | 6 | 6 | -- |
| B. Scale and speed | 6 | 6 | -- |
| C. The alignment problem | 7 | 7 | -- |
| D. Fields and kernels | 11 | 11 | -- |
| E. Learning | 11 | 10 | E11 (partly) |
| F. Interpretation | 7 | 7 | -- |
| G. The application and the tool | 6 | 5 | G5 |
| H. Materials and periodicity | 15 | 11 | M7, M8, M9, M15 (partly) |
| H. Nanotechnology | 12 | 1 | NT1, NT2, NT3, NT4, NT5, NT6, NT7, NT8 (partly), NT10, NT11, NT12 |
| H. Electrochemistry | 10 | 8 | EC8 (partly), EC9 |
| H. Battery science | 8 | 7 | BT8 |
| H. Local properties | 5 | 5 | -- |
| H. Supramolecular and host-guest | 8 | 4 | SC2, SC4, SC5 (partly), SC8 |
| H. Physical organic | 8 | 7 | PBH6 |
| H. Deep learning | 16 | 0 | ND1 to ND13 |
| I. Medicinal chemistry | 11 | 9 | MC3, MC10 |

**147 items: 104 finished, 5 partly, 38 open.**
 Parts A to F and the battery, local-property and
periodicity work are substantially done; what is left falls into three kinds, and they are not
equally blocked.

* **Waiting for data, not for code.** EC9 needs a measured catalytic property at more than one
  potential, EC8 one at a periodic site, MC3 a series in which both enantiomers are measured, SC4
  a host-guest set curated by conditions, and NT12 the structure sets the nanotechnology part
  would be judged on. Four items in a row had recorded that no machine-readable series could be
  found for the catalytic case; PO1 found one on 5 November 2026, a measured enantioselectivity
  on 43 molecular catalysts at one condition, which answers the molecular case and not EC8's or
  EC9's. The search is written up in `docs/public_3dqsar_datasets.md`.
* **Waiting for the application.** G5, NT8 and NT11 are what is left of this block: the
  particle side of the window, where a structure that is neither a molecule nor a cell needs a
  data model of its own (NT1), and the model panel and command line that would offer it. Two of
  NT8's three parts are already done (the cell outline with its images, and the bonds with the
  metal rule); what it still asks for is instanced drawing for ten thousand atoms. Neither
  needs a new measurement. Three of the six are done. M13 closed the crystal side on
  6 November 2026; **EC10 and NT9 closed the surface side and the reading on the same day** --
  the slab behind the molecules, the surface frame with its two numbers, the vector co-field as
  arrows, and the chart of channel weight against distance that the kernels without a co-field
  are read by, in both windows. The window, the viewer and the panels those three added are what
  G5, NT8 and NT11 extend.
* **Open on their own merits.** M7 to M9, NT1 to NT7, SC2, SC8, PBH6, MC10, E11's remaining three
  versions, and the whole deep-learning part ND1 to ND13, which has not been started.
* **Closed on 5 November 2026.** D11, the charge model that an element or a valence state could
  void, found by EC4 inside the d block and by EC3 outside it. Its gate moved one published
  conclusion: 72 % of item BT5's solvation shells had carried a spurious net charge of +1, and
  correcting it cost BT5 its headline interval on random folds.


The order inside a part follows the dependencies: nothing in C or E is judged without the protocol
of C1 and the intervals of rule 4; D6 and E6 wait for B4; E3 waits for E2 and the topomer rule; E11 waits for the landmark features of B6 and the COMPAS tables of PBH1, and its gate (c) for the fragment libraries of E3 and MC7;
F2 needs the GP view that B4 makes the default; E8 and E9 are judged on the classification sets
of E4 and E9 reuses the Newton step of the Laplace classifier, E10 waits for both because its
screening gate needs a classifier to compare with; D9 waits for D4 and, with D10, for the
verdict of D8; in part I the substituent items wait for the topomer rule E3 measured, MC9 for
the variance field of D10, MC10 for MC5 and MC9 together, and MC11 for the panel data of MC4. F6 needs the cosine of F3 to compare the two co-fields of a component, and the features of B6 to run on a set large enough for the question to matter.

## What is not planned

* **Neural networks on fields as the model.** A 3D convolutional network on a field grid gives up
  the closed-form kernel, the co-field and the exactness of the Shapley values at once. The
  nonlinear kernel layer and the Shapley decomposition already cover the case where a linear
  model is not enough. The one exception is E11, where a network is trained on the closed-form
  kernel distance and the activity model stays the kernel model; a network that replaces the
  model is still not planned.
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
