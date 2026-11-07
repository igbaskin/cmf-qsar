# cmfqsar API reference

This file is generated from the docstrings of the source by `python docs/make_api.py`; do not edit it by hand. `python docs/make_api.py --check` fails when it is out of date. It covers 114 modules, 225 classes with 743 public methods, and 1127 public functions. A name is public when it does not start with an underscore. The user manual (`docs/manual.md`, chapter *Using it as a library*) explains what the modules are for and shows worked scripts; this file says what every function takes and returns.

cmfqsar - Continuous Molecular Fields (CMF) 3D-QSAR.

Implementation of the CMF approach described in:

* Baskin I.I., Zhokhova N.I. "The continuous molecular fields approach to
  building 3D-QSAR models", J. Comput.-Aided Mol. Des. 27 (2013) 427-442.
* Sitnikov G.V., Zhokhova N.I., Ustynyuk Y.A., Varnek A., Baskin I.I.
  "Continuous indicator fields: a novel universal type of molecular fields",
  J. Comput.-Aided Mol. Des. 29 (2015) 233-247.
* Baskin I.I., Zhokhova N.I. "Continuous molecular fields and the concept of
  molecular co-fields in structure-activity studies", Future Med. Chem. (2019).

Molecular fields are represented as sums of atom-centred Gaussian functions,
kernels between molecules are overlap integrals of those fields (computed
analytically), and kernel ridge regression (or SVR / GP) is used to fit
3D-QSAR models.  The fields of regression coefficients ("molecular co-fields")
are reconstructed from the dual coefficients and can be visualised.

The stable surface
------------------
The names in `__all__` below are the supported interface of this package (plan item G3).  They
are documented in the manual, chapter *Using it as a library*, used by the notebooks under
`docs/tutorials/` and guarded by `tests/test_stable_surface.py`, which records the arguments
each one promises.  Within a major version they are not removed or renamed, and an argument that
is recorded there keeps its name and its default; new arguments may be added after them.

Every other module is internal.  It is documented in `docs/api.md` and may change in any
release, so a script that imports one should say which version of the package it was written
for.

## Modules

| Module | What it is for |
|---|---|
| [cmfqsar](#cmfqsar) | cmfqsar - Continuous Molecular Fields (CMF) 3D-QSAR. |
| [cmfqsar.active](#cmfqsaractive) | Which molecule to make next (plan item E2). |
| [cmfqsar.align](#cmfqsaralign) | 3D structure generation and molecular alignment. |
| [cmfqsar.anisotropic](#cmfqsaranisotropic) | Atom functions with a direction: Gaussians times a linear factor (plan item D1). |
| [cmfqsar.atomtargets](#cmfqsaratomtargets) | A target that belongs to an atom rather than to a molecule (plan item LP1). |
| [cmfqsar.atomtypes](#cmfqsaratomtypes) | Atom typing (Tripos/Sybyl-like) and van der Waals parameters. |
| [cmfqsar.battery](#cmfqsarbattery) | Electrode data for the periodic kernel: insertion pairs, and the voltage between them. |
| [cmfqsar.bioisostere](#cmfqsarbioisostere) | Which groups are interchangeable here? |
| [cmfqsar.bonds](#cmfqsarbonds) | Bonds of a periodic structure, perceived from distances, for drawing only. |
| [cmfqsar.cdft](#cmfqsarcdft) | Conceptual-DFT fields on a grid, and what atom-centred Gaussians can and cannot make of them. |
| [cmfqsar.celldiff](#cmfqsarcelldiff) | The difference between two cells of one ladder as a field in the cell (plan item BT6). |
| [cmfqsar.chemotypes](#cmfqsarchemotypes) | Heterogeneous series: several chemotypes, no single common substructure. |
| [cmfqsar.cli](#cmfqsarcli) | Command-line interface for building CMF 3D-QSAR models (no GUI needed). |
| [cmfqsar.coalign](#cmfqsarcoalign) | Co-alignment of a training set, treated as what it is: expectation-maximisation over poses. |
| [cmfqsar.cofields](#cmfqsarcofields) | Molecular co-fields (fields of regression coefficients), molecular fields and their overlap evaluated on a rectangular grid for visualisation (isosurfaces) and export. |
| [cmfqsar.complementarity](#cmfqsarcomplementarity) | Compatibility of molecular fields and co-fields with the fields of a protein pocket. |
| [cmfqsar.complexes](#cmfqsarcomplexes) | Transition-metal complexes: reading one from coordinates, and the frame its metal gives. |
| [cmfqsar.conformal](#cmfqsarconformal) | Prediction intervals with a coverage guarantee, for any learner and any kernel source. |
| [cmfqsar.conformers](#cmfqsarconformers) | Conformational ensembles: the fourth dimension of the CMF approach. |
| [cmfqsar.contract](#cmfqsarcontract) | What has to match between fitting and predicting, checked rather than assumed. |
| [cmfqsar.copattern](#cmfqsarcopattern) | The co-pattern of an invariant kernel as a curve: weight against distance (plan item NT9). |
| [cmfqsar.dataio](#cmfqsardataio) | Reading structure-activity data sets (SDF / MOL2 / SMILES), saving models and exporting results. |
| [cmfqsar.decoys](#cmfqsardecoys) | Preparing a background (decoy) set for field-based virtual screening. |
| [cmfqsar.descriptors](#cmfqsardescriptors) | Non-field descriptors: one number per molecule, beside the fields rather than instead of them. |
| [cmfqsar.design](#cmfqsardesign) | The last step of the design cycle: what to put at a position, ranked by the model (E3). |
| [cmfqsar.diagnose](#cmfqsardiagnose) | What to ask of a data set before modelling it, and which route it should take. |
| [cmfqsar.docking](#cmfqsardocking) | Place conformations inside the pocket instead of in the gas phase. |
| [cmfqsar.electrolyte](#cmfqsarelectrolyte) | Bulk electrolyte simulations, for plan item BT5. |
| [cmfqsar.errorpred](#cmfqsarerrorpred) | Which of the things that claim to know when a prediction is bad actually knows (plan item E5). |
| [cmfqsar.evidence](#cmfqsarevidence) | Choosing alpha and lambda by the evidence instead of by cross-validation. |
| [cmfqsar.exposure](#cmfqsarexposure) | Solvent exposure as a weight on the fields: the claim of the surface methods, without a surface. |
| [cmfqsar.fastkernel](#cmfqsarfastkernel) | A compiled, threaded atom-pair loop for the field kernels (plan item B5). |
| [cmfqsar.fieldae](#cmfqsarfieldae) | A field autoencoder with an analytic decoder (plan item E11). |
| [cmfqsar.fielddiff](#cmfqsarfielddiff) | What one molecule's fields have that another's do not. |
| [cmfqsar.fieldgraph](#cmfqsarfieldgraph) | Field-point graphs: an alignment that needs neither a common substructure nor a model. |
| [cmfqsar.fields](#cmfqsarfields) | Parameterisation of continuous molecular fields. |
| [cmfqsar.flexalign](#cmfqsarflexalign) | Flexible alignment by confidence: the rigid-body search of 2019, with the torsions. |
| [cmfqsar.fluctuation](#cmfqsarfluctuation) | The variance of a field across the frames of an ensemble (plan item D10). |
| [cmfqsar.fraggrow](#cmfqsarfraggrow) | Growing a bound fragment into the gap the pocket leaves (plan item MC7). |
| [cmfqsar.fusion](#cmfqsarfusion) | Data fusion: one score from the similarities to several known actives. |
| [cmfqsar.gpc](#cmfqsargpc) | Gaussian-process classification by the Laplace approximation. |
| [cmfqsar.gradients](#cmfqsargradients) | The derivative of a prediction with respect to the atoms of the molecule being predicted. |
| [cmfqsar.gtm](#cmfqsargtm) | Chemical space and activity landscapes from CMF kernels: kernel PCA + Generative Topographic Mapping. |
| [cmfqsar.hops](#cmfqsarhops) | A migration path as the sample: the environment of a hop, its frame and its labels (BT7). |
| [cmfqsar.increments](#cmfqsarincrements) | Atomic increments of the original CMF implementation (Baskin & Zhokhova 2013). |
| [cmfqsar.invariant](#cmfqsarinvariant) | Alignment-free but geometry-aware kernels: autocorrelation of the molecular fields. |
| [cmfqsar.kcca](#cmfqsarkcca) | Kernel canonical correlation analysis: what one view of the molecules sees that another does not (plan item F6). |
| [cmfqsar.kernels](#cmfqsarkernels) | Molecular-field kernels. |
| [cmfqsar.learnincrements](#cmfqsarlearnincrements) | Fitting the atomic increments to the data instead of taking them from the table (item D5). |
| [cmfqsar.localalign](#cmfqsarlocalalign) | Local alignment: an environment posed on its centre and first shell (plan item LP3). |
| [cmfqsar.localenv](#cmfqsarlocalenv) | Atom-centred environments and the invariant local kernel (plan item LP2). |
| [cmfqsar.locallarge](#cmfqsarlocallarge) | Many atoms at once: the local kernel without a kernel matrix (plan item LP5). |
| [cmfqsar.lowrank](#cmfqsarlowrank) | Low-rank kernels for large training sets (plan item B6). |
| [cmfqsar.maskedmatch](#cmfqsarmaskedmatch) | Scoring a candidate only where the co-field has weight (plan item MC5). |
| [cmfqsar.md](#cmfqsarmd) | Short ligand-in-pocket simulations, for plan item D8. |
| [cmfqsar.mil](#cmfqsarmil) | Multi-instance learning over conformations: let the model choose the pose. |
| [cmfqsar.mixing](#cmfqsarmixing) | A posterior over the field weights (plan item E6). |
| [cmfqsar.mmd](#cmfqsarmmd) | Is this set of molecules like that one? |
| [cmfqsar.model](#cmfqsarmodel) | CMF regression models: kernel ridge regression (KRR, the method of the 2013 paper), kernel partial least squares (KPLS, the kernel form of the PLS that CoMFA uses), support vector regression and a Gaussian-process view of KRR that provides predictive variances. |
| [cmfqsar.multiobjective](#cmfqsarmultiobjective) | A design loop with more than one objective (plan item MC11). |
| [cmfqsar.multitask](#cmfqsarmultitask) | Two endpoints at once, so that selectivity is a co-field rather than a ratio. |
| [cmfqsar.neldermead](#cmfqsarneldermead) | Nelder-Mead simplex minimiser, a direct port of R's `nmmin` (`optim(method = "Nelder-Mead")`, R 2.x-4.x), which the original CMF R scripts use to optimise the attenuation factor alpha and the regularisation parameter gamma simultaneously (`cmf_krr_train_mem` with `alpha_grid_search = FALSE`). |
| [cmfqsar.nonlinearity](#cmfqsarnonlinearity) | The co-field of a model with a nonlinear kernel layer: one per molecule, and how far they differ. |
| [cmfqsar.oneclass](#cmfqsaroneclass) | One-class scores on a precomputed kernel: four geometries besides the 1-SVM's ball. |
| [cmfqsar.orca](#cmfqsarorca) | Quantum-chemical properties from ORCA: per-atom quantities that become fields, per-molecule ones that become descriptors. |
| [cmfqsar.pbh](#cmfqsarpbh) | Polybenzenoid hydrocarbons: a planar frame, ring classes, ring-centred fields and per-ring contributions. |
| [cmfqsar.perfield](#cmfqsarperfield) | A separate attenuation for every field, chosen by the evidence (plan item D6). |
| [cmfqsar.periodic](#cmfqsarperiodic) | Continuous molecular fields of crystals: the periodic field autocorrelation. |
| [cmfqsar.pharmacophore](#cmfqsarpharmacophore) | A pharmacophore read out of a fitted co-field. |
| [cmfqsar.pharmtypes](#cmfqsarpharmtypes) | Six pharmacophore features as indicator fields (plan item D2). |
| [cmfqsar.planealign](#cmfqsarplanealign) | Alignment onto a plane: the frame of a molecule on a surface, without the surface. |
| [cmfqsar.pocketalign](#cmfqsarpocketalign) | Aligning a molecule inside the pocket (plan item C3). |
| [cmfqsar.poses](#cmfqsarposes) | A protocol that can judge an alignment, because q2 cannot. |
| [cmfqsar.probes](#cmfqsarprobes) | GRID-like probe fields, written as sums of Gaussians so that CMF can keep its algebra. |
| [cmfqsar.project](#cmfqsarproject) | Projects (plan item G2): a session that can be saved and reopened, the poses before every alignment step so that a step can be undone, and two models compared on one set of molecules. |
| [cmfqsar.protein](#cmfqsarprotein) | Protein structures for display behind the molecules, their fields and the co-fields. |
| [cmfqsar.protonation](#cmfqsarprotonation) | Protonation ensembles: one molecule as the weighted mean of its ionisation states (item D4). |
| [cmfqsar.prototypes](#cmfqsarprototypes) | Inverse design on prototypes: candidates, an oracle and a loop (plan item M11). |
| [cmfqsar.pu](#cmfqsarpu) | Learning from positives and unlabelled molecules (part of plan item E4). |
| [cmfqsar.ranking](#cmfqsarranking) | Learning to order molecules instead of to reproduce their numbers. |
| [cmfqsar.reciprocal](#cmfqsarreciprocal) | Reciprocal-space channels of a crystal: the typed structure factors (plan item M6). |
| [cmfqsar.report](#cmfqsarreport) | One HTML file that says what a model is, how well it works and what it says (plan item F4). |
| [cmfqsar.restriction](#cmfqsarrestriction) | Scoring a conformational restriction (plan item MC9). |
| [cmfqsar.scattering](#cmfqsarscattering) | Measured scattering curves as samples: the total, untyped descriptor (plan item M12). |
| [cmfqsar.screen](#cmfqsarscreen) | Screening a library without holding it in memory (plan item B3). |
| [cmfqsar.shapley](#cmfqsarshapley) | Shapley decomposition of a CMF prediction. |
| [cmfqsar.shells](#cmfqsarshells) | Solvation shells cut from a bulk electrolyte trajectory (plan item BT5). |
| [cmfqsar.sites](#cmfqsarsites) | Site-resolved models of a periodic structure (plan item M5). |
| [cmfqsar.slab](#cmfqsarslab) | A metal slab as the receptor (plan item EC6). |
| [cmfqsar.solvation](#cmfqsarsolvation) | A cation with its first solvation shell as one molecule, built, optimised and framed. |
| [cmfqsar.solvent](#cmfqsarsolvent) | Solvent as a field: where a ligand's water sits, as a CMF field (plan item D9). |
| [cmfqsar.spectral](#cmfqsarspectral) | Clustering a set by its fields rather than by its scaffolds. |
| [cmfqsar.stereo](#cmfqsarstereo) | Handedness as a property of a centre, not of a shape. |
| [cmfqsar.surrogate](#cmfqsarsurrogate) | The model as the fast level of a hierarchy: a standard deviation in the units of the activity, its calibration on the folds, and the rule that sends a structure back to the fine level (plan item MS1). |
| [cmfqsar.tautomers](#cmfqsartautomers) | Tautomer ensembles: the other half of plan item D4. |
| [cmfqsar.threebody](#cmfqsarthreebody) | Three-body channels and handedness for a periodic cell (plan item M10). |
| [cmfqsar.topomer](#cmfqsartopomer) | Topomer fields: a frame that comes from the topology, not from a superposition. |
| [cmfqsar.trajectory](#cmfqsartrajectory) | A molecular-dynamics trajectory as a conformational ensemble (plan item D7). |
| [cmfqsar.transfer](#cmfqsartransfer) | Comparing two models by their co-fields (plan item F3). |
| [cmfqsar.transformfield](#cmfqsartransformfield) | A matched pair as a field placed in space (plan item MC6). |
| [cmfqsar.triangles](#cmfqsartriangles) | The three-distance autocorrelation of an environment: the angle at the centre (plan item C5). |
| [cmfqsar.validation](#cmfqsarvalidation) | The three questions a 3D-QSAR model has to answer besides "how good is q²". |
| [cmfqsar.varalign](#cmfqsarvaralign) | Alignment by prediction confidence: the pose of a molecule is chosen so that the posterior variance of its predicted activity is as small as possible. |
| [cmfqsar.vasp](#cmfqsarvasp) | Per-atom arrays from a VASP run, read into an ASE structure (plan item M3). |
| [cmfqsar.xtb](#cmfqsarxtb) | A cheap quantum stage: per-atom properties from GFN2-xTB (plan item D3). |
| [cmfqsar.gui](#cmfqsargui) | PySide6 desktop application for CMF 3D-QSAR. |
| [cmfqsar.gui.app](#cmfqsarguiapp) | Application entry point. |
| [cmfqsar.gui.copattern_panel](#cmfqsarguicopattern_panel) | The co-pattern chart: weight against distance for one pair of field types (plan item NT9). |
| [cmfqsar.gui.crystal_window](#cmfqsarguicrystal_window) | A periodic structure as the sample, in the application (plan item M13). |
| [cmfqsar.gui.main_window](#cmfqsarguimain_window) | Main window of the CMF 3D-QSAR desktop application. |
| [cmfqsar.gui.panels](#cmfqsarguipanels) | Dock panels of the CMF application: data set table, model settings, results, viewer controls. |
| [cmfqsar.gui.surface_view](#cmfqsarguisurface_view) | A surface behind the molecules, in the application (plan item EC10). |
| [cmfqsar.gui.viewer3d](#cmfqsarguiviewer3d) | OpenGL 3D viewer (pyqtgraph.opengl) for molecules, co-field isosurfaces, molecular-field meshes and atom-contribution spheres. |
| [cmfqsar.gui.workers](#cmfqsarguiworkers) | Background execution of long computations with progress reporting and cancellation. |

## cmfqsar

Source: `cmfqsar/__init__.py`

cmfqsar - Continuous Molecular Fields (CMF) 3D-QSAR.

Implementation of the CMF approach described in:

* Baskin I.I., Zhokhova N.I. "The continuous molecular fields approach to
  building 3D-QSAR models", J. Comput.-Aided Mol. Des. 27 (2013) 427-442.
* Sitnikov G.V., Zhokhova N.I., Ustynyuk Y.A., Varnek A., Baskin I.I.
  "Continuous indicator fields: a novel universal type of molecular fields",
  J. Comput.-Aided Mol. Des. 29 (2015) 233-247.
* Baskin I.I., Zhokhova N.I. "Continuous molecular fields and the concept of
  molecular co-fields in structure-activity studies", Future Med. Chem. (2019).

Molecular fields are represented as sums of atom-centred Gaussian functions,
kernels between molecules are overlap integrals of those fields (computed
analytically), and kernel ridge regression (or SVR / GP) is used to fit
3D-QSAR models.  The fields of regression coefficients ("molecular co-fields")
are reconstructed from the dual coefficients and can be visualised.

The stable surface
------------------
The names in `__all__` below are the supported interface of this package (plan item G3).  They
are documented in the manual, chapter *Using it as a library*, used by the notebooks under
`docs/tutorials/` and guarded by `tests/test_stable_surface.py`, which records the arguments
each one promises.  Within a major version they are not removed or renamed, and an argument that
is recorded there keeps its name and its default; new arguments may be added after them.

Every other module is internal.  It is documented in `docs/api.md` and may change in any
release, so a script that imports one should say which version of the package it was written
for.

## cmfqsar.active

Source: `cmfqsar/active.py`

Which molecule to make next (plan item E2).

A fitted CMF model predicts an activity and, through the Gaussian-process view, a posterior
standard deviation beside it.  Those two numbers are enough to answer a question the 2019 paper
poses and leaves as a statement -- *maximising the overlap tells how to modify the molecule* --
as an operation with a number: given a set of posed candidates, which one should be made?

The answer depends on what the asker is doing, and the module keeps the two apart rather than
pretending one rule serves both.

* **To find an active compound** the acquisition is optimistic.  Expected improvement asks how
  much a candidate is expected to beat the best thing already made, counting only the gain:

      EI(x) = (mu - y* - xi) Phi(z) + sigma phi(z),   z = (mu - y* - xi) / sigma

  and the upper confidence bound `mu + beta sigma` is the cruder form of the same idea.  A
  molecule the model is unsure about is *promoted*, because being wrong about it is cheap and
  being right is valuable.

* **To improve the model** the acquisition ignores the prediction entirely and takes the largest
  posterior standard deviation, or -- with no model at all -- the candidate furthest in field
  space from everything already made (`maxmin`, the classical furthest-first design).

Note that this is the opposite of `CMFModel.rank_score`, which ranks a *screening* library
by `mu - beta sigma`: there a molecule the model knows nothing about must be pushed down,
because a kernel method hands back the training mean presented as a prediction and the library
is full of such molecules.  Screening is a decision whose mistakes have a cost; acquisition is a
question that has to be answered.  Optimism is right for one and wrong for the other.

`learning_curve` runs the loop retrospectively on a kernel whose activities are already
known, which is how `benchmarks/run_active.py` measures whether any of this beats picking at
random.

### `class Suggestion  (dataclass)`

One candidate, with the numbers that chose it.

Attributes:

- `index: int`
- `name: str`
- `mean: float`
- `sd: float`
- `score: float`
- `distance: float` = `float('nan')`

#### `Suggestion.describe(kind: str) -> str`

### `expected_improvement(mean: Sequence[float], sd: Sequence[float], best: float, xi: float=0.0) -> np.ndarray`

E[max(y - best - xi, 0)] under the posterior, in closed form.

Zero standard deviation gives the improvement itself when there is one and zero otherwise,
which is the limit of the expression and keeps a molecule the model is certain about from
being discarded for its certainty.

### `acquisition_scores(kind: str, mean: Sequence[float], sd: Sequence[float], best: Optional[float]=None, beta: float=1.0, xi: float=0.0, distance: Optional[Sequence[float]]=None, rng: Optional[np.random.Generator]=None) -> np.ndarray`

The score of every candidate under one rule; larger is always better.

`distance` is the field-space distance of each candidate to the nearest molecule already
made, which only `maxmin` uses; `best` is the best activity so far, which only `ei` uses.

### `suggest(model, candidates, kind: str='ei', n: int=1, beta: float=1.0, xi: float=0.0, names: Optional[Sequence[str]]=None, max_similarity: float=0.98, seed: Optional[int]=0, batch_rule: str='cosine') -> List[Suggestion]`

Rank posed candidates by one acquisition rule and return the best `n`.

The model must be a regression on the linear kernel, since that is where a posterior exists.
With `n` above one a batch is chosen, and `batch_rule` says how (plan item MC2):

* `"cosine"` takes the best scores in order and skips a candidate too similar to one
  already picked (`max_similarity`, a cosine in the model's own kernel).  It is cheap and it
  has a threshold in it.
* `"covariance"` recomputes the score after every pick from the posterior covariance
  conditioned on the picks already made, which needs no threshold and costs one Cholesky
  step per pick.  With `kind="variance"` it is the batch of largest determinant.

Neither refits the model inside the batch, which is what a laboratory would have to pay for;
`benchmarks/run_batch.py` measures what the two rules are worth against that.

### `conditional_variances(cov: np.ndarray) -> np.ndarray`

The diagonal of a covariance matrix, which is where a greedy batch starts.

### `condition_on(cov: np.ndarray, j: int, floor: float=1e-12) -> np.ndarray`

The covariance of everything else once candidate `j` is known.

One step of a Cholesky factorisation with `j` as the pivot:

    cov' = cov - cov[:, j] cov[j, :] / cov[j, j]

The variance of `j` itself becomes zero, and the variance of anything correlated with it
falls.  That is how a batch chosen this way stops being one molecule several times over:
nothing has to be refitted, because the posterior covariance already knows what a second
measurement near the first would add.

### `batch_by_covariance(kind: str, mean: Sequence[float], cov: np.ndarray, size: int, best: Optional[float]=None, beta: float=1.0, xi: float=0.0, distance: Optional[Sequence[float]]=None, rng: Optional[np.random.Generator]=None) -> List[int]`

A batch of `size` candidates, the score recomputed after every pick (plan item MC2).

The cheap rule of `suggest` ranks once and drops a candidate that looks too much like
one already picked, which is a filter with a threshold in it.  This rule has no threshold:
after each pick the posterior covariance is conditioned on that pick
(`condition_on`), the standard deviations are taken from the conditioned covariance,
and the acquisition is evaluated again.  A candidate whose uncertainty the first pick has
already removed scores low by itself.

With `kind="variance"` the score **is** the conditional standard deviation, so this is the
greedy maximisation of the determinant of the batch's own covariance block -- the classical
D-optimal design -- and `log det` of that block is the sum of the logs of the conditional
variances the picks were chosen by.

### `batch_by_cosine(score: Sequence[float], cosine: Optional[np.ndarray], size: int, max_similarity: float=0.98) -> List[int]`

A batch of `size` by score, skipping a candidate too similar to one already picked.

This is the rule `suggest` has always used, as a function of its own so that the two
can be measured against each other.

### `log_determinant(cov: np.ndarray, rows: Sequence[int]) -> float`

log det of the covariance block of a batch, which is what a D-optimal rule maximises.

### `learning_curve(kernels: Dict[float, np.ndarray], y: Sequence[float], kind: str, start: Sequence[int], steps: int, lams: Sequence[float]=(0.01, 0.1, 1.0), beta: float=1.0, xi: float=0.0, seed: int=0, fit_bias: bool=True, batch: int=1, batch_rule: str='cosine', max_similarity: float=0.98) -> Dict[str, list]`

Run the acquisition loop over a set whose activities are already known.

`kernels` maps an attenuation to its full joint kernel over every molecule, so the loop can
reselect the hyper-parameters at every step -- by the evidence, from what has been acquired
and nothing else, which is what item B4 made affordable.  Nothing outside `start` and the
molecules the rule asks for is ever looked at.

With `batch` above one the model is refitted once per batch and not once per molecule,
which is what a laboratory that makes ten compounds at a time does (plan item MC2).
`batch_rule` is then how the rest of the batch is chosen after the first pick: `"cosine"`
takes the next best scores and skips what looks too similar, `"covariance"` recomputes the
score from the posterior covariance conditioned on the picks already made.

Returns, after every acquisition or batch: the RMSE and q2 of the model on the molecules
*not* yet made, the best activity among those made, and which molecules were taken.

## cmfqsar.align

Source: `cmfqsar/align.py`

3D structure generation and molecular alignment.

* embed_3d          - ETKDG conformer + MMFF optimisation for a SMILES-derived molecule
* mcs_atom_map      - the atom correspondence of that maximum common substructure
* mcs_align         - rigid alignment on the maximum common substructure with a reference
* field_align       - rigid-body refinement maximising the (normalised) CMF kernel
                      between a molecule and the reference (the alignment criterion
                      suggested in Baskin & Zhokhova 2013)

### `embed_3d(mol: Chem.Mol, seed: int=42, optimize: bool=True) -> Chem.Mol`

### `transform_coords(coords: np.ndarray, rotvec: np.ndarray, trans: np.ndarray, center: np.ndarray) -> np.ndarray`

### `set_coords(mol: Chem.Mol, coords: np.ndarray) -> None`

### `mcs_atom_map(mol: Chem.Mol, ref: Chem.Mol, timeout: int=10, ring_matches_ring_only: bool=True, complete_rings_only: bool=False, compare_elements: bool=True) -> List[Tuple[int, int]]`

The atom correspondence of the maximum common substructure of `mol` and `ref`.

Returns the pairs (atom of `mol`, atom of `ref`).  Nothing is moved: the map is what
`mcs_align` superimposes on, and it is worth computing once when several
conformations of the same molecule are to be aligned on the same reference.

### `mcs_align(mol: Chem.Mol, ref: Chem.Mol, timeout: int=10, ring_matches_ring_only: bool=True, complete_rings_only: bool=False, compare_elements: bool=True) -> Tuple[float, int]`

Rigidly superimpose `mol` onto `ref` using their maximum common substructure.

Returns (rmsd, number of matched atoms).  `mol` is modified in place.

### `field_align(mol: Chem.Mol, ref: Chem.Mol, fieldset: FieldSet, restarts: int=0, seed: int=0, max_iter: int=300) -> Tuple[float, float]`

Refine the pose of `mol` by maximising the normalised CMF kernel with `ref`.

Returns (score_before, score_after); `mol` is modified in place.

### `align_dataset(mols: Sequence[Chem.Mol], ref_index: int, use_mcs: bool=True, field_refine: bool=False, fieldset: Optional[FieldSet]=None, progress=None) -> List[str]`

Align every molecule to mols[ref_index] in place. Returns a log line per molecule.

### `random_rigid_transform(coords: np.ndarray, rng: np.random.RandomState, max_rotation: Optional[float]=None, max_translation: float=2.0) -> np.ndarray`

Randomly rotate `coords` about their centroid and translate them.

Parameters
----------
max_rotation : maximum rotation angle in degrees.  `None` draws a uniformly
    distributed orientation (the alignment is then destroyed completely);
    a number draws a random axis and an angle uniform in [0, max_rotation].
max_translation : the displacement is a random direction times a length drawn
    uniformly from [0, max_translation] (A).

The transformation is rigid, so all interatomic distances are preserved.

### `unalign(mols: Sequence[MoleculeFields], max_rotation: Optional[float]=None, max_translation: float=2.0, seed: int=0, progress=None) -> List[float]`

Destroy the alignment of a set of molecules by giving each a random pose.

Every molecule is rotated about its own centroid and translated, so the internal
geometry (and therefore the self-kernel of every molecule) is unchanged while the
mutual arrangement is randomised.  This is the control experiment for any alignment
method: scramble the poses, realign, and measure how much of the original arrangement
comes back.

The molecules are modified in place; the displacement (RMSD, A) of each is returned.

## cmfqsar.anisotropic

Source: `cmfqsar/anisotropic.py`

Atom functions with a direction: Gaussians times a linear factor (plan item D1).

Every field in this program is a sum of isotropic Gaussians on the nuclei, and
`cmfqsar.cdft` showed what that cannot do.  A Fukui function is essentially a
frontier-orbital density: on acrolein it is pi*, three quarters of its squared magnitude lies more
than 0.5 A off the molecular plane, and there is a node *in* the plane.  A function that is
positive everywhere and centred on a nucleus cannot make a node, and the fit loses half the field
(relative error 77 % on f+, 89 % on the dual descriptor).

The smallest extension that can make a lobe is to multiply the Gaussian by a linear factor.  With
the width convention this program uses, `exp(-b |r - R|^2 / 2)`, the basis on atom `l` becomes

    s_l(r)   = exp(-b |r - R_l|^2 / 2)
    p_l^u(r) = (r_u - R_l,u) exp(-b |r - R_l|^2 / 2),      u = x, y, z

one s and three p functions per atom per width.  A p function is odd about its own centre, so it
*has* a node through the nucleus and a lobe on either side, which is the shape that was missing.
Three Cartesian components span every direction, so nothing has to be oriented along bonds by
hand: the fit chooses the direction, and a lobe along a bond, along a lone pair or perpendicular
to a ring plane are all the same three numbers in a different combination.

The overlaps stay closed-form, which is the whole reason this is the extension to try.  With
`p = a + b`, `mu = ab/p`, `d = B - A` and the s|s overlap
`S = (2 pi / p)^{3/2} exp(-mu |d|^2 / 2)`:

    <s_A | s_B>     = S
    <p_A^u | s_B>   = (b/p) d_u S
    <s_A | p_B^v>   = -(a/p) d_v S
    <p_A^u | p_B^v> = [delta_uv / p - (ab/p^2) d_u d_v] S

`pair_overlaps` evaluates all four over whole arrays of centres, and
`tests/test_anisotropic.py` holds every one of them against numerical quadrature, which is how
they were derived.  A kernel between two fields is then still a double sum over pairs of atoms and
pairs of basis functions -- four times as many terms per atom pair, with the same cost per term --
so the co-field, the atom contributions and the training-set decomposition all survive.

What this module does not do is decide whether the extension is worth its cost.
`benchmarks/run_anisotropic.py` measures that the way the plan item asks: the fit error of the
conceptual-DFT fields on the four corrosion series that have ORCA results, with and without the p
functions, and then whether a model built on the resulting kernels notices the difference.

### `class AnisotropicFit  (dataclass)`

A field fitted with s and p functions on every atom.

Attributes:

- `coeffs: np.ndarray` = `field(default_factory=lambda: np.zeros((0, 0, 0)))`
- `widths: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`
- `p_type: bool` = `True`
- `rel_error: float` = `float('nan')`
- `r2: float` = `float('nan')`
- `name: str` = `''`

#### `AnisotropicFit.n_coefficients` -> int  (property)

#### `AnisotropicFit.condensed() -> np.ndarray`

What each atom's share of the fitted field integrates to.

Only the s functions contribute: a p function is odd about its own centre and integrates
to zero, which is the point of it -- it moves density from one side of the nucleus to the
other without changing how much there is.

#### `AnisotropicFit.dipoles() -> np.ndarray`

(n_atoms, 3): the first moment the p functions put on each atom.

`int (r - R_l) F_l(r) dr` for atom `l`'s own part of the field, which is the direction
and size of the lobe the fit chose there.  Zero for an isotropic fit, by construction.

#### `AnisotropicFit.evaluate(points: np.ndarray, positions: np.ndarray, chunk: int=100000) -> np.ndarray`

The fitted field at arbitrary points, for checking it against the grid it came from.

### `basis_size(n_widths: int, p_type: bool=True) -> int`

Coefficients per atom for `n_widths` widths, with or without the p functions.

### `design(points: np.ndarray, positions: np.ndarray, widths: np.ndarray, p_type: bool=True) -> np.ndarray`

The basis evaluated at `points`: (n_points, n_atoms, n_widths, 1 or 4).

The last axis is the s function and then the three p components, in the order x, y, z.  It is
kept as a separate axis rather than flattened so that a caller can drop the p block and get
exactly the isotropic basis back.

### `pair_overlaps(A: np.ndarray, B: np.ndarray, a: float, b: float, p_type: bool=True) -> np.ndarray`

Overlap integrals between the basis on centres `A` and the basis on centres `B`.

Returns (n_A, n_B, k, k) with k = 4 (s, px, py, pz) or 1 (s only): entry [l, m, i, j] is
`<f_i on A_l | f_j on B_m>` for Gaussians of width `a` on A and `b` on B.  Derived from the
Gaussian product rule and checked against quadrature in the tests.

### `field_overlap(coeff_a: np.ndarray, pos_a: np.ndarray, coeff_b: np.ndarray, pos_b: np.ndarray, widths: np.ndarray, p_type: bool=True) -> float`

`int F_A(r) F_B(r) dr` for two fitted fields, in closed form.

The coefficients are (n_atoms, n_widths, k) with k as in `design`.  The sum runs over
pairs of atoms, pairs of widths and pairs of basis functions; the atom pairs are the expensive
axis and are vectorised, the rest is a loop over at most 4 x 4 x n_widths^2 scalars.

### `overlap_kernel(coeffs: Sequence[np.ndarray], positions: Sequence[np.ndarray], widths: np.ndarray, p_type: bool=True) -> np.ndarray`

The matrix of `field_overlap` between every pair of fitted fields.

### `fit_field(values: np.ndarray, points: np.ndarray, positions: np.ndarray, widths: np.ndarray, window: Optional[np.ndarray]=None, p_type: bool=True, ridge: float=1e-06, name: str='', chunk: int=100000) -> AnisotropicFit`

Least squares with s and p functions on every atom, in chunks over the grid.

The design matrix of a 20-atom molecule on an 80^3 grid with four widths and four functions
is 160 million entries; what is accumulated instead is the 320 by 320 normal matrix.  `ridge`
is relative to the mean diagonal, because the p blocks of nearby atoms are nearly collinear
and the normal matrix is close to singular without it.

### `fit_cube(values: np.ndarray, cube, positions: Optional[np.ndarray]=None, window: Optional[np.ndarray]=None, n_terms: int=4, p_type: bool=True, ridge: float=1e-06, name: str='', chunk: int=100000) -> AnisotropicFit`

`fit_field` on a `cmfqsar.cdft.Cube`, with the widths the probe fields share.

## cmfqsar.atomtargets

Source: `cmfqsar/atomtargets.py`

A target that belongs to an atom rather than to a molecule (plan item LP1).

Everything else in this program predicts one number per molecule.  A carbon shift, an acidity, a
site's formation energy and a local reactivity are numbers per *atom*, and a data set of them is
not a list: most atoms of a molecule carry no measurement at all, and the atoms that do are not
independent samples of each other, because they share a molecule.

This module is the data model for that, and nothing more -- there is no model here, and the item
that adds one (the local kernel) needs this first.  It holds three things:

* **reading**, with blanks.  `read_atom_target` returns one number per atom and *not a
  number* where the atom has none, which is the case that separates an atom target from the
  per-atom properties a field is built from (`cmfqsar.fields.atom_property`), where every
  atom must have a value.
* **a sample index.**  `AtomTargets` is the list of measured (molecule, atom) pairs with
  their values, which is what a fit iterates over; the molecule each sample came from is kept
  beside it, because it is needed for every honest split.
* **folds that hold out whole molecules.**  Two shifts of one molecule in different folds would
  let a model see the molecule it is being tested on, and a per-atom q2 obtained that way is
  meaningless.  `AtomTargets.folds` splits the molecules, as the grouped folds of
  `cmfqsar.chemotypes` split the chemotypes.

The predictions go back where they came from: `write_atom_predictions` writes an SDF whose
atoms carry the measured and the predicted value, so a shift sits on the carbon it belongs to.

### `class AtomTargets  (dataclass)`

The measured (molecule, atom) pairs of a data set, and what each one measured.

Attributes:

- `molecule: np.ndarray` = `_field(default_factory=lambda: np.zeros(0, int))`
- `atom: np.ndarray` = `_field(default_factory=lambda: np.zeros(0, int))`
- `y: np.ndarray` = `_field(default_factory=lambda: np.zeros(0))`
- `n_mol: int` = `0`
- `name: str` = `''`
- `per_molecule: List[np.ndarray]` = `_field(default_factory=list)`

#### `AtomTargets.n_measured_molecules` -> int  (property)

#### `AtomTargets.from_molecules(mols: Sequence[Chem.Mol], name: str, names: Optional[Sequence[str]]=None) -> 'AtomTargets'`  (class method)

Read the target from every molecule and build the index of what was measured.

#### `AtomTargets.from_arrays(per_structure: Sequence[Sequence[float]], name: str='') -> 'AtomTargets'`  (class method)

The same index from one array per structure, not a number where the atom or site was
not measured.

`from_molecules` reads the values from the molecules themselves, which a periodic
cell cannot carry: a crystal site takes its target from a table instead (plan item M5,
`cmfqsar.sites`).  Everything after this point -- the folds, the subsets, the
scatter back over the structures -- is the same for an atom and for a site.

#### `AtomTargets.subset(rows: Sequence[int]) -> 'AtomTargets'`

#### `AtomTargets.of_molecule(k: int) -> np.ndarray`

The sample rows that belong to molecule `k`.

#### `AtomTargets.counts() -> Dict[int, int]`

How many atoms were measured in each molecule that has any.

#### `AtomTargets.folds(k: int=10, seed: Optional[int]=0) -> List[np.ndarray]`

Test-sample rows for `k` folds that hold out whole **molecules**.

A molecule's atoms never straddle the split: two shifts of one molecule, one in the
training set and one in the test set, would let the model see the molecule it is being
tested on, and the q2 that came out would not mean anything.

#### `AtomTargets.describe() -> str`

### `read_atom_target(mol: Chem.Mol, name: str, mol_name: str='') -> np.ndarray`

One number per atom, *not a number* where the atom has none (plan item LP1).

Two conventions are read, in this order: a property on the atoms themselves, where an atom
without it is simply unmeasured, and a molecule property holding the numbers in atom order,
where a blank field is unmeasured.  A list as long as the molecule is taken atom for atom; a
list as long as its heavy atoms is spread over them, leaving the hydrogens unmeasured, which
is what a file written before the hydrogens were added means; any other length is an error
naming the molecule and both counts, because the alternative is to guess which atom a number
belongs to.

Returns an array of length `mol.GetNumAtoms()`; a molecule carrying nothing comes back as
all *not a number*, which is a molecule with no measurement rather than an error.

### `scatter(targets: AtomTargets, values: Sequence[float]) -> List[np.ndarray]`

Spread one value per sample back over the molecules, as one array per molecule.

### `write_atom_predictions(path: str, mols: Sequence[Chem.Mol], targets: AtomTargets, predicted: Optional[Sequence[float]]=None, names: Optional[Sequence[str]]=None) -> int`

Write an SDF whose atoms carry the measured value and, if given, the predicted one.

A predicted shift belongs on the carbon it was predicted for, not in a column of a table
beside the molecule, so it is written in RDKit's own per-atom convention, the molecule
property `atom.dprop.<name>` holding one field per atom.  RDKit decodes that back into
properties on the atoms themselves when the file is read, an atom whose field is `n/a`
gets nothing, and the writer does not drop it as it drops properties set on atoms directly.

### `candidate_targets(mols: Sequence[Chem.Mol], limit: int=200) -> Dict[str, Tuple[int, int]]`

What in these molecules could be a per-atom target: (molecules carrying it, atoms measured).

This is a looser rule than the one for a per-atom *field*
(`cmfqsar.fields.atom_property_names`), and it has to be: a field needs a value on every
atom, while a target is a measurement that most atoms do not have.  A property counts here
when it is set on at least one atom, or when it is a molecule property whose fields number
one per atom or one per heavy atom once the blanks are counted in.

## cmfqsar.atomtypes

Source: `cmfqsar/atomtypes.py`

Atom typing (Tripos/Sybyl-like) and van der Waals parameters.

Sybyl atom types are used (a) for continuous indicator fields (CIFs) and
(b) to look up Tripos force-field van der Waals parameters for the steric
field (Baskin & Zhokhova 2013, Eq. 10-11).

### `is_transition_metal(symbol: str) -> bool`

Is this element symbol one of the d-block metals (plan item EC4)?

### `vdw_params(symbol: str, atom_type: Optional[str]=None)`

Return (R_vdw, E_vdw) for an element symbol, or for a Sybyl atom type if given.

### `sybyl_atom_type(atom: Chem.Atom) -> str`

Assign an approximate Sybyl (Tripos) atom type to an RDKit atom.

### `element_atom_type(atom: Chem.Atom) -> str`

### `element_aromatic_atom_type(atom: Chem.Atom) -> str`

### `pbh_atom_type(atom: Chem.Atom) -> str`

The carbon classes of a benzenoid hydrocarbon: a rim carbon bonded to a hydrogen
(`C.H`), a carbon shared by two rings (`C.2r`), by three (`C.3r`); see
`cmfqsar.pbh.pbh_atom_type`.

### `assign_atom_types(mol: Chem.Mol, scheme: str='sybyl') -> List[str]`

Return a list with one atom type string per atom in `mol`.

If the molecule carries pre-assigned types in the atom property `_TriposAtomType`
(e.g. read from a MOL2 file) and `scheme == "sybyl"` these are used directly.

## cmfqsar.battery

Source: `cmfqsar/battery.py`

Electrode data for the periodic kernel: insertion pairs, and the voltage between them.

A cathode is measured by its **average voltage**: the work per electron of taking the working ion
out of the host.  For the insertion reaction

    Li_x Host  +  (y - x) Li  ->  Li_y Host

the average voltage over that interval is

    V = - [ E(Li_y Host) - E(Li_x Host) - (y - x) E(Li metal) ] / (y - x),

with the energies per host formula unit and in electronvolts, so that V comes out in volts.  The
reference to lithium metal is what makes the number a voltage against a lithium anode.

This module builds those pairs out of public data, without an account anywhere.  The Materials
Project serves its structures and their **formation energies** through OPTIMADE, which needs no
API key, and a formation energy already carries the elemental reference: it is the energy of the
cell minus the energies of the elements it is made of.  Elemental lithium is one of those
elements, so in the difference above it cancels, and

    V = - [ E_f(Li_y Host) - E_f(Li_x Host) ] / (y - x)

with E_f the formation energy per host formula unit.  Nothing else is needed, and the result is
the same quantity the battery databases publish, up to the corrections they apply (see
*What this is not*).

**How a pair is formed.**  One query per chemical system (`Li`, one metal, `O` -- the ternary
lithium-metal-oxygen systems) returns every structure of that system, the lithium-free host
included.  Each structure is reduced to its **host framework** (its composition with the working
ion removed, divided by the common factor) and its **lithium content per host formula unit**.
Structures that share a framework form a ladder in that content; the lowest structure above the
convex hull is kept at each step, and neighbouring steps make a pair.

**What this is not.**  The pairs are matched by host *composition*, not by host *structure*: two
cells with the same framework formula and different frameworks can be paired, which a topotactic
criterion would refuse.  The energies are the Materials Project's GGA/GGA+U values without the
corrections a battery database applies to them, so a voltage here can differ from the published
one by of the order of 0.1 V.  Both limits are measured rather than argued in
`benchmarks/run_battery.py`, which checks the lithium cobalt oxide and lithium iron phosphate
voltages against the published ones before it uses the set for anything.

The official insertion-electrode documents, with topotactic pairs and the corrections, need a
free Materials Project API key and the `mp-api` package; `insertion_electrodes` fetches
them when a key is given, and the benchmark uses them in place of the reconstructed pairs if it
finds one.

### `class InsertionPair  (dataclass)`

Two steps of one host's ladder and the average voltage between them.

Attributes:

- `framework: str`
- `charged: object`
- `discharged: object`
- `x_charged: float`
- `x_discharged: float`
- `voltage: float`
- `volume_change: float` = `float('nan')`
- `hull_charged: float` = `float('nan')`
- `hull_discharged: float` = `float('nan')`
- `ion: str` = `WORKING_ION`
- `info: Dict[str, object]` = `field(default_factory=dict)`

#### `InsertionPair.name` -> str  (property)

A label a table can carry: the framework and the two contents.

#### `InsertionPair.capacity` -> float  (property)

The specific capacity of the interval, mAh per gram of the discharged cell.

### `fetch_system(elements: Sequence[str], endpoint: str=MP_OPTIMADE, page_limit: int=100, max_pages: int=100, timeout: float=90.0) -> List[dict]`

Every structure whose elements are among those named, as OPTIMADE entries.

`elements HAS ONLY` is the OPTIMADE way of asking for one chemical system: the lithium-free
host, the fully lithiated phase and everything between them come back in one answer, which is
what a ladder needs.

### `insertion_electrodes(key: Optional[str]=None, working_ion: str=WORKING_ION, limit: int=10000) -> List[dict]`

The Materials Project's own insertion-electrode documents, which need a free API key.

The key is taken from the argument, then from `MP_API_KEY` or `PMG_MAPI_KEY` in the
environment.  Without one this raises, and the caller falls back to the pairs this module
reconstructs from formation energies, which need no key.

### `write_records(records: Sequence[dict], path: str) -> str`

Keep the fetched entries, so that a measurement does not depend on a service being up.

### `read_records(path: str) -> List[dict]`

The entries written by `write_records`.

### `atoms_from_optimade(entry: dict)`

One OPTIMADE entry as an ASE `Atoms`, carrying its identifier and its energies.

What travels in `atoms.info`: `id`, `formation_energy_per_atom` and
`energy_above_hull` in eV, and `functional`, the name of the calculation they came from.
The descriptor reader of `cmfqsar.descriptors` looks there, so any of them can enter a
model as a non-field descriptor.

### `host_framework(atoms, ion: str=WORKING_ION) -> Tuple[Tuple[str, int], ...]`

The composition of the cell with the working ion removed, divided by its common factor.

Two cells of one ladder share this: `CoO2` for both `LiCoO2` and `Li0.5CoO2`, however
large the cell each was computed in.

### `host_units(atoms, ion: str=WORKING_ION) -> float`

How many host formula units the cell holds.

### `ion_content(atoms, ion: str=WORKING_ION) -> float`

Working ions per host formula unit: the *x* of Li_x Host.

### `formation_energy_per_host(atoms, ion: str=WORKING_ION) -> float`

The formation energy of the cell divided by the number of host formula units, in eV.

### `average_voltage(charged, discharged, ion: str=WORKING_ION) -> float`

The average voltage between two steps of one ladder, in volts.

`charged` is the cell with less of the working ion in it.  Both formation energies are taken
per host formula unit, so the cells need not be the same size.

### `framework_label(framework: Sequence[Tuple[str, int]]) -> str`

`(("Co", 1), ("O", 2))` as `CoO2`.

### `ladders(structures: Sequence, ion: str=WORKING_ION, tol: float=1e-06) -> Dict[str, Dict[float, object]]`

Group the structures by host framework and keep the most stable cell at each content.

The key of the inner mapping is the working-ion content per host formula unit, rounded to six
decimals so that two cells of the same stoichiometry in different cell sizes meet.  The cell
kept is the one lowest above the convex hull, which is the phase a measurement would see.

### `insertion_pairs(structures: Sequence, ion: str=WORKING_ION, min_delta: float=0.001, max_voltage: float=6.0, min_voltage: float=-1.0) -> List[InsertionPair]`

Every neighbouring pair of every ladder, with its average voltage.

A pair is refused when the two steps hold the same amount of the working ion, when either
formation energy is missing, or when the voltage falls outside `min_voltage` to
`max_voltage`: outside that range the pair is not an insertion electrode but two unrelated
phases that happen to share a framework formula.

### `pair_table(pairs: Sequence[InsertionPair], frame: str='discharged')`

The structures, the voltages and the labels of a set of pairs, ready for `crystal_table`.

`frame` says which cell of the pair describes it: `"discharged"` (the lithiated one),
`"charged"`, or `"both"`, which returns the two cells of every pair in order, each with
the pair's voltage, so a kernel sees the ladder from both ends.

### `chemistry(pair: InsertionPair) -> str`

Which family the host belongs to: an oxide, a phosphate, a sulfide, and so on.

The families are the ones an electrode review groups by, and a pair that is in none of them is
reported as `other` rather than forced into one.

## cmfqsar.bioisostere

Source: `cmfqsar/bioisostere.py`

Which groups are interchangeable here? A bioisostere table weighted by the co-field (MC1).

`cmf design` ranks candidate substituents by the predicted activity of the whole molecule.  The
table a medicinal chemist asks for answers a different question: **which groups can replace each
other at this position**, so that one of them can be chosen for a property the model knows
nothing about -- solubility, cost, a patent, a metabolic liability.

The model answers it without being refitted once per group.  The prediction of the overlap kernel
is a sum over the atoms of the molecule (decomposition rule 3,
`cmfqsar.model.CMFModel.atom_contributions`), so the part of the prediction that belongs to
a substituent is the sum over that substituent's own atoms, and the predicted effect of exchanging
group A for group B at one position is the difference of those two sums.  That is one integral per
pair of groups rather than one model fit per group.

Three distances between two groups
----------------------------------
For every pair of substituents at the same position this module computes:

* **the fingerprint distance**, one minus the Tanimoto similarity of their Morgan fingerprints:
  two dimensions, no geometry, no model;
* **the field distance**, the distance between the two posed groups in the field metric of the
  model's own kernel, restricted to the atoms of the groups themselves: geometry, but no model;
* **the predicted change**, the difference of the two atom-contribution sums: geometry and the
  model's co-field, since the contribution of an atom is its field weighted by the co-field.

The gate of plan item MC1 is whether the third ranks the measured activity difference of a matched
molecular pair better than the first two.  If it does not, then what the table shows is shape
similarity and needs no model at all, which would be a result of the same kind as plan item D5.
`benchmarks/run_bioisostere.py` is that measurement.

Finding the pairs
-----------------
The pairs come from RDKit's `rdMMPA`, which cuts one single bond at a time and reports the two
pieces.  Every cut is taken in both directions, so each piece is tried as the core; two molecules
whose cores are the same molecule and whose groups are different are a matched pair at that
position.  Nothing is chosen by eye.

### `class Substituent  (dataclass)`

One group of one molecule, with the core it hangs on.

Attributes:

- `molecule: int`
- `core: str`
- `group: str`
- `atoms: np.ndarray`

### `class MatchedPair  (dataclass)`

Two molecules that differ in one group at one position.

Attributes:

- `a: int`
- `b: int`
- `core: str`
- `groups: Tuple[str, str]`
- `atoms_a: np.ndarray`
- `atoms_b: np.ndarray`
- `delta: float`
- `names: Tuple[str, str]` = `('', '')`

### `class PairScore  (dataclass)`

One matched pair, by the three measures and by what was measured.

Attributes:

- `pair: MatchedPair`
- `fingerprint: float`
- `field: float`
- `change: float`

#### `PairScore.measured` -> float  (property)

### `class Table  (dataclass)`

A square table of groups against groups at one position.

Attributes:

- `core: str`
- `groups: List[str]`
- `molecules: List[int]`
- `distance: np.ndarray`
- `change: np.ndarray`
- `half_width: float` = `float('nan')`
- `names: List[str]` = `_field(default_factory=list)`

#### `Table.interchangeable(group: str, within: Optional[float]=None) -> List[str]`

The groups whose predicted change against `group` is inside the model's own error.

Without `within`, the bound is twice the conformal half width: the model cannot tell two
predictions apart by less than that, so a smaller change is not a change it can see.

#### `Table.rows()`

(group a, group b, distance, change) for every ordered pair, a != b.

### `fragment(mol, max_substituent: int=MAX_SUBSTITUENT, min_core: float=MIN_CORE_SHARE) -> List[Substituent]`

Every way of cutting one single bond of `mol`, as (core, group, the group's atoms).

Both directions of every cut are reported, because which piece is the core is decided by
what the other molecules of the set have in common and not by this molecule.  The atom
indices refer to `mol` as it was passed in, which is what lets the group be masked out of the
molecule's fields afterwards.

### `substituents(mols: Sequence, **kw) -> List[Substituent]`

`fragment` over a set, with the molecule index filled in.

### `matched_pairs(mols: Sequence, y: Sequence[float], names: Optional[Sequence[str]]=None, max_substituent: int=MAX_SUBSTITUENT, min_core: float=MIN_CORE_SHARE, one_per_couple: bool=True) -> List[MatchedPair]`

Every matched molecular pair of a set: two molecules, one core, two different groups.

`delta` is the activity of the second molecule minus that of the first.  With
`one_per_couple`, two molecules that can be matched on several cores are reported once, on
the largest core, because the same two measured numbers must not enter a correlation twice.

### `masked_field(mol: MoleculeFields, atoms: Sequence[int], name: str='') -> MoleculeFields`

The fields of `mol` restricted to `atoms`, as a field object of its own.

A kernel of this program is a sum over pairs of atoms, so restricting one side of it to a
subset of the atoms is the same as building the fields of that subset and nothing else.  The
atoms keep the coordinates they have in the molecule, so two groups compared this way are
compared **in their poses** and not as detached fragments.

### `group_distance(a: MoleculeFields, b: MoleculeFields, fieldset: FieldSet, scale: Optional[np.ndarray]=None) -> float`

The distance between two posed groups in the field metric, with no model in it.

`d^2 = K(a,a) - 2 K(a,b) + K(b,b)` over the enabled fields, each field scaled as the model
scales it so that the fields are comparable with one another.  This is the reference the
co-field weighting has to beat: it knows where the two groups put their field, but not
whether the activity depends on it.

### `group_contribution(model, mol: MoleculeFields, atoms: Sequence[int], coef: Optional[np.ndarray]=None) -> float`

How much of the prediction for `mol` belongs to the atoms of one group.

The prediction of the overlap kernel is a sum over the atoms (decomposition rule 3), so this
is simply that sum restricted to the group.  `coef` replaces the model's dual coefficients,
which is how a value is obtained from a model that has not seen the molecules being compared.

### `predicted_change(model, mol_a: MoleculeFields, atoms_a: Sequence[int], mol_b: MoleculeFields, atoms_b: Sequence[int], coef: Optional[np.ndarray]=None) -> float`

What the model says exchanging the first group for the second does to the activity.

### `fingerprint_distance(group_a: str, group_b: str, radius: int=2, bits: int=2048) -> float`

One minus the Tanimoto similarity of two groups, written as SMILES with an attachment.

### `score_pairs(model, fields: Sequence[MoleculeFields], pairs: Sequence[MatchedPair], coef_of=None, progress=None) -> List[PairScore]`

The three measures for every pair.

`coef_of(pair)` returns the dual coefficients to use for that pair, which is how the model is
kept away from the two activities it is being compared with; the default uses the model's own.

### `spearman(x: Sequence[float], y: Sequence[float]) -> float`

Rank correlation, which is what "ranks better" means here.

### `rank_gain(better: Sequence[float], worse: Sequence[float], measured: Sequence[float], runs: int=2000, seed: int=0) -> Tuple[float, float, float]`

How much better one measure ranks the measured difference than another, with an interval.

Both measures are compared with the **size** of the measured activity difference, and the
bootstrap resamples the pairs, so the interval is the one rule 4 asks for.

### `build_table(model, fields: Sequence[MoleculeFields], core: str, subs: Sequence[Substituent], half_width: float=float('nan'), names: Optional[Sequence[str]]=None) -> Table`

The square table of one position: every group of `subs` against every other.

One group per molecule is used, so that the table is a table of groups and not of molecules;
where two molecules carry the same group the first is kept.

### `biggest_core(subs: Sequence[Substituent], least: int=3) -> Optional[str]`

The core carrying the most different groups, which is where a table is worth building.

## cmfqsar.bonds

Source: `cmfqsar/bonds.py`

Bonds of a periodic structure, perceived from distances, for drawing only.

A crystal read from a CIF or an extended-XYZ file has no bond table: a file of that kind states
where the atoms are and nothing about which of them are bonded.  The molecular view of this
program draws bonds because an RDKit molecule carries them; the crystal view drew spheres alone.
For an inorganic framework that is the honest picture.  For an **organic molecule adsorbed on a
surface**, which is the case items EC2 and EC6 are about, it is not: a reader cannot tell a ring
from a chain in a cloud of spheres.

This module supplies the missing list.  Everything here is **display-only perception**: it
decides which pairs of atoms to draw a stick between and nothing else.  No molecule is
sanitised, no bond order is assigned and no charge is perceived from it -- those belong to plan
item M8, which turns a periodic structure into a molecule, and a bond drawn here never reaches a
field, a kernel or a prediction.

The rule, and the two things it gets right
------------------------------------------
A pair is bonded when `|r_ij| < R_i + R_j + tolerance` with covalent radii R and a tolerance of
0.45 angstrom, which is the rule `cmfqsar.protein.covalent_bonds` already applies to a
protein.  Two things are added for the periodic case:

* **the images.**  The pairs are taken over the cell *and* its periodic images, so a bond that
  crosses the cell boundary is found rather than lost.  A benzene ring that a file leaves split
  across the boundary is drawn as a ring, with the sticks leaving the box on one side and the
  image atoms they end on drawn with them.
* **metals get no bonds.**  A distance rule gives every atom of a close-packed metal about twelve
  neighbours, and twelve sticks per atom is noise, not information: it hides the structure it is
  meant to show.  So the rule is applied to the non-metal atoms only (plan item NT8 states it as
  "no bond perception for metals").  A slab therefore stays spheres and the organic molecule on
  it gets its bonds, which is exactly the picture the adsorbate case needs.

What a metal is told instead: `coordination_polyhedra` draws the first coordination shell
of a metal atom as a polyhedron, which is how an inorganic structure is read -- a chain of
corner-sharing octahedra says more than any number of sticks would.

### `class DrawableStructure  (dataclass)`

What a viewer needs to draw one cell with its bonds.

The first `n_cell` rows are the atoms of the cell, **in the cell's own order**, so a site
number in the Reading table still names the same row.  The rows after them are the image
atoms that a bond reaches, and only those: an image that is bonded to nothing is not drawn,
because the images have a drawing of their own.

Attributes:

- `coords: np.ndarray`
- `symbols: List[str]`
- `radii: np.ndarray`
- `bonds: np.ndarray`
- `n_cell: int`
- `source: np.ndarray`
- `skipped_metals: int` = `0`
- `n_distinct: int` = `0`

#### `DrawableStructure.n_images_drawn` -> int  (property)

#### `DrawableStructure.n_crossing` -> int  (property)

Sticks that leave the cell: one end in it, the other on an image atom.

#### `DrawableStructure.metal_note(n_atoms: Optional[int]=None) -> str`

How many atoms got no bonds because they are metal atoms.

#### `DrawableStructure.describe() -> str`

### `class Polyhedron  (dataclass)`

The first coordination shell of one atom, as a closed surface to draw.

Attributes:

- `centre: int`
- `element: str`
- `vertices: np.ndarray`
- `faces: np.ndarray`
- `neighbours: List[str]` = `field(default_factory=list)`
- `planar: bool` = `False`

#### `Polyhedron.coordination` -> int  (property)

#### `Polyhedron.label() -> str`

### `covalent_radius(symbol: str) -> float`

The covalent radius of an element in angstrom.

The table of `cmfqsar.protein` first, because the organic elements should have the same
radius in every part of the program; then ASE's table, which covers every element a crystal
may hold; then `DEFAULT_RADIUS`.

### `is_bonding(symbol: str) -> bool`

Whether a stick may be drawn to an atom of this element.  A metal: no.

### `perceive_bonds(coords: np.ndarray, symbols: Sequence[str], tolerance: float=BOND_TOLERANCE, max_bond: float=MAX_BOND, skip_metals: bool=True) -> np.ndarray`

Bonded pairs of an open (non-periodic) set of atoms, as an (n, 2) array of indices.

With `skip_metals` a pair counts only when **both** atoms are in
`BONDING_ELEMENTS`; without it the distance rule is applied to every pair, which is
what a reader asks for when the structure is a molecular crystal of a metal complex and the
metal-ligand bonds are the point.

### `periodic_bonds(crystal, tolerance: float=BOND_TOLERANCE, max_bond: float=MAX_BOND, skip_metals: bool=True) -> DrawableStructure`

The bonds of one cell, with the images a bond reaches (display only).

The images are expanded to `max_bond`, which is enough: no stick is longer than that, so any
bonding partner outside the cell lies within `max_bond` of the box.  A pair of two image
atoms is dropped -- it is a bond of a neighbouring cell and the cell's own copy of it is
already drawn -- and a pair with one end in the cell is kept, so a bond that crosses the
boundary is drawn leaving the box, with the atom it ends on drawn beside it.

### `coordination_numbers(crystal, ratio: float=SHELL_RATIO, max_shell: float=3.6, centres: str='metals') -> List[Tuple[int, str, int]]`

`(atom, element, coordination)` for every candidate centre, small shells included.

`coordination_polyhedra` leaves out a shell too small to be a shape; this reports every
centre, so that a count of the polyhedra drawn can be read against the number of atoms that
could have had one.  The shell rule is the same one, shared, not repeated.

### `coordination_polyhedra(crystal, ratio: float=SHELL_RATIO, max_shell: float=3.6, min_coordination: int=3, centres: str='metals', limit: int=MAX_POLYHEDRA) -> List[Polyhedron]`

The first coordination shell of every metal atom of a cell, as polyhedra.

The shell is found without any chemistry: the nearest neighbour of the centre sets a
distance, and every neighbour within `ratio` times that distance is a vertex.  That is the
usual rule, and it needs no parameter per element.  Neighbours are taken over the periodic
images, so a polyhedron at the edge of the cell is whole; non-metal neighbours are preferred
where the centre has any, which is what makes the vertices the anions of an ionic structure
rather than the next cations out.

`centres` is `"metals"` (the atoms that get no bonds, which is what a polyhedron is for),
`"all"` or `"non-metals"`.  A shell of fewer than `min_coordination` neighbours is left
out, and a flat shell is drawn as a polygon with `Polyhedron.planar` set.

### `polyhedra_mesh(polyhedra: Sequence[Polyhedron]) -> Optional[Tuple[np.ndarray, np.ndarray]]`

Every polyhedron as one mesh, so a cell of hundreds costs one drawing item.

## cmfqsar.cdft

Source: `cmfqsar/cdft.py`

Conceptual-DFT fields on a grid, and what atom-centred Gaussians can and cannot make of them.

The program already carries the *condensed* conceptual-DFT quantities -- a Hirshfeld f+ per atom,
a dual descriptor per atom -- and makes a field of them by putting one Gaussian of the field's own
width on each atom.  That discards the shape: a lone pair is a lobe to one side of the oxygen,
a pi system sits above and below the plane, and a single isotropic Gaussian centred on the nucleus
can say neither.  The quantity those numbers are condensed *from* is a function of position,
f(r) = rho_{N+1}(r) - rho_N(r), and ORCA will write it on a grid.

This module reads those grids (`read_cube`, `cdft_fields`, `run_orca_plot`) and
tries to bring them into CMF's algebra the way the probe fields were brought in -- by fitting
several Gaussians per atom,

    F(r)  ~  sum_l sum_t  c_{l,t} exp(-beta_t |r - R_l|^2 / 2),

with the widths `beta_t` shared across atoms, which is what makes the overlap of two such fields
a closed-form double sum over atom pairs and term pairs, and so keeps the co-field, the
training-set decomposition and the atom contributions.

**The fit reproduces the field badly and the kernel well, and that difference is what this
module shows.**  Taking the two measurements in the order they were made:

*The field is not reproduced.*  On acrolein, fitted
against the valence part of the map (the raw maps peak *on the nuclei* -- every one of f+, f-,
the dual descriptor and the Parr functions has its maximum within 0.2 A of the oxygen, because a
density difference is dominated by core relaxation -- so the target is multiplied by
`valence_window`):

===========================  ==========  ======  ==========
centres                      f+          dual    Parr P+
===========================  ==========  ======  ==========
nuclei only (8)              77%         89%     72%
+ pi centres at 0.8 A (16)   53%         69%     50%
+ pi centres and bonds (19)  47%         62%     41%
===========================  ==========  ======  ==========

the numbers being the relative error in the norm the kernel sees.  The fitter is not what fails:
on a field that *is* a sum of Gaussians on these widths it recovers the coefficients to within a
few per cent (`tests/test_cdft.py`).  The target is what cannot be represented.  A Fukui
function is essentially a frontier-orbital density, and acrolein's is pi*: three quarters of its
squared magnitude lies more than 0.5 A off the molecular plane, with a node *in* the plane, and an
isotropic Gaussian centred on a nucleus cannot make a node.  Putting centres above and below the
plane recovers a third of the error -- which confirms the diagnosis -- at the cost of doubling
the centres, and so quadrupling the kernel, and of giving up the one-to-one correspondence between
a centre and an atom that makes the atom contributions mean anything.

*The kernel is.*  A kernel is an inner product between molecules, and an error that falls the same
way for every molecule cancels in it.  On the twenty benzothiazoles of the corrosion series
(`benchmarks/run_cdft_kernels.py`), with the field kept exactly on a common grid and the kernel
done by quadrature, against the same field fitted and the kernel done in closed form:

=========================  =======  =======  =========  ==========
f+(r)                      alone    + the 5  r to grid  fit error
=========================  =======  =======  =========  ==========
grid quadrature            +0.638   +0.669   --         --
fitted Gaussians           +0.636   +0.658   +0.933     81%
condensed (one per atom)   +0.553   +0.684   +0.724     --
=========================  =======  =======  =========  ==========

and for the dual descriptor +0.522 / +0.556 / +0.448 alone, the fitted kernel again matching the
exact one (r = +0.899) while losing 91% of the field.  So the angular shape that the fit cannot
carry does not reach the kernel: as a *predictor* the fitted field is the grid field, at a
fraction of the cost and with CMF's algebra intact.

What the fit does carry that the condensed field does not is the *radial* shape -- four widths per
atom rather than one -- and that is worth something: alone, the fitted field beats the condensed
one by 0.08 in q2 on f+ and by 0.11 on the dual descriptor, and its kernel is much closer to the
exact one (+0.93 against +0.72).

Worth remembering before building on any of it: on that series none of the three helps.  The
five ordinary fields alone give +0.689, and the best quantum combination is +0.684.

Two practical lessons survive for whoever tries again:

* **Use a narrow width basis.**  With the probe widths (2.0 down to 0.08 A^-2) the coefficients
  reach +/-100 and cancel, on a field whose integral is 1; with 20 down to 0.5 A^-2 the largest
  is 0.5 and the fitted field integrates to +0.95 against a true +1.00.  Broad Gaussians on atoms
  a bond apart are very nearly the same function.
* **What survives the fit is the condensation.**  Once the coefficients are well conditioned, what
  the fit reproduces is each atom's share of the integral (`FieldFit.condensed`) -- which is
  what the Hirshfeld condensation in `cmfqsar.orca` already gives, more cheaply and without
  a grid.  The shape, which was the whole reason to go to the grid, is what is lost.

Nothing here is connected to the field machinery.  It computes the conceptual-DFT fields, fits
them, and reports how well -- by the measure that decides it, which is the kernel and not the
field.

### `class Cube  (dataclass)`

A Gaussian cube, in angstrom: values on a regular grid plus the atoms it was made for.

Attributes:

- `values: np.ndarray`
- `origin: np.ndarray`
- `vectors: np.ndarray`
- `numbers: np.ndarray`
- `positions: np.ndarray`
- `path: str` = `''`

#### `Cube.dV` -> float  (property)

#### `Cube.symbols` -> List[str]  (property)

#### `Cube.points() -> np.ndarray`

(N, 3) coordinates of every grid point, in the cube's order.

#### `Cube.integral() -> float`

#### `Cube.same_grid_as(other: 'Cube') -> bool`

### `class FieldFit  (dataclass)`

The fitted expansion of one conceptual-DFT field of one molecule.

Attributes:

- `coeffs: np.ndarray` = `field(default_factory=lambda: np.zeros((0, 0)))`
- `widths: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`
- `rel_error: float` = `float('nan')`
- `r2: float` = `float('nan')`
- `name: str` = `''`

#### `FieldFit.condensed() -> np.ndarray`

What each atom's share of the fitted field integrates to.

The fit's own answer to the question the Hirshfeld condensation answers, and the natural
thing to compare it with: sum_t c_t (2 pi / beta_t)^{3/2}, one number per atom.

#### `FieldFit.evaluate(points: np.ndarray, positions: np.ndarray, chunk: int=200000) -> np.ndarray`

The fitted field at arbitrary points -- for checking it against the grid it came from.

### `read_cube(path: str) -> Cube`

Read a Gaussian cube written by `orca_plot`; lengths come back in angstrom.

Cube files are in bohr, and a negative atom count marks the (rare) variant that carries a
trailing list of orbital indices -- both are handled here rather than by the caller.

### `valence_window(rho: np.ndarray, rho_core: float=RHO_CORE, power: float=WINDOW_POWER) -> np.ndarray`

1 where the density is low, 0 in the cores, smooth in between.

`w = 1 / (1 + (rho / rho_core)^power)` with `rho` in e/bohr^3.  Smooth matters: a hard
cut-off puts a step into the field, and a step is the one thing a sum of Gaussians cannot
follow, so the fit would spend its terms on the edge instead of on the chemistry.

### `fit_field(values: np.ndarray, cube: Cube, positions: Optional[np.ndarray]=None, window: Optional[np.ndarray]=None, n_terms: int=4, ridge: float=1e-06, name: str='', chunk: int=200000, widths: Optional[np.ndarray]=None) -> FieldFit`

Fit `values` on the grid of `cube` with `n_terms` Gaussians per atom.

`window` multiplies the target before fitting -- pass `valence_window` of the N-electron
density to fit the valence part, which is what these maps mean.  The widths are the ones the
probe fields share, so the resulting coefficients can be used directly in the probe kernel.

The normal equations are accumulated in chunks: the design matrix of a 48-atom molecule on an
80^3 grid is 100 million entries, and the matrix that matters, A'A, is 192 by 192.

### `overlap_kernel(coeffs: Sequence[np.ndarray], positions: Sequence[np.ndarray], widths: np.ndarray) -> np.ndarray`

The closed-form overlap integrals between fitted fields: K[i, j] = int F_i F_j dr.

The same formula the probe branch of `cmfqsar.kernels` evaluates, written here over
explicit coefficient matrices so that a fitted field can be scored without being connected to
a FieldSet.  Two Gaussians of widths b1, b2 a distance D apart overlap by
`(2 pi / (b1 + b2))^{3/2} exp(-b1 b2 D^2 / (2 (b1 + b2)))`; the rest is bookkeeping over
pairs of centres and pairs of terms.

### `cdft_fields(rho_n: Cube, rho_np1: Optional[Cube]=None, rho_nm1: Optional[Cube]=None, spin_np1: Optional[Cube]=None, spin_nm1: Optional[Cube]=None, potential: Optional[Cube]=None) -> dict`

The conceptual-DFT fields as arrays on the common grid of the cubes given.

Finite differences at fixed geometry, which is what the condensed indices of
`cmfqsar.orca` are made of as well -- the same quantities before they are condensed.

`potential` is the electrostatic potential of the N-electron state (plan item EC1), in
hartree per electron.  It is the one field here that is not a difference of densities, and it
is offered for display and for a grid kernel rather than as a replacement for the
atom-centred electrostatic field: this module's own measurement is that a few Gaussians per
atom represent the long-range tail of a potential badly, so the CHELPG charges of
`cmfqsar.orca` remain the first choice for a field built from the potential.

### `fit_cdft(cubes: dict, rho_n: Cube, names: Sequence[str]=CDFT_FIELDS, n_terms: int=4, rho_core: float=RHO_CORE, ridge: float=1e-06) -> dict`

Fit every named conceptual-DFT field of one molecule; returns {name: FieldFit}.

### `run_orca_plot(gbw: str, kind: int=PLOT_DENSITY, intervals: int=80, orca_plot: Optional[str]=None, timeout: Optional[float]=1800.0, reuse: bool=True) -> str`

Write a cube beside `gbw` with `orca_plot`, and return its path.

ORCA 6 has no `%plot` input block -- it is rejected outright -- so the cube can only come
from the separate program, which is menu-driven and is therefore driven by its menu.  The box
is the one orca_plot chooses from the geometry, which is what makes the three states of a
molecule share a grid exactly.

`kind` is `PLOT_DENSITY`, `PLOT_SPIN` or `PLOT_POTENTIAL`.  The potential is
entry 43 of the menu of ORCA 6.1 and differs from the other two in one way that matters here:
instead of offering the default density for the caller to accept, it lists the densities the
job saved and asks for one by name, which is `<job>.scfp` for a single point.  Its cube is
written as `<job>.scfp.esp.cube` and has the same header as the density cube, so
`read_cube` reads all three unchanged.

## cmfqsar.celldiff

Source: `cmfqsar/celldiff.py`

The difference between two cells of one ladder as a field in the cell (plan item BT6).

An intercalation voltage is not a property of a structure but of a **pair**: the energy of the
lithiated cell minus the energy of the delithiated one, per electron.  The fields of this program
are linear in the atoms, so the same difference can be taken of the fields themselves,

    dX_f(r) = X_f(Li_y Host)(r) - X_f(Li_x Host)(r),

and the result is an ordinary field in the cell: the atoms of the lithiated cell with their
weights and the atoms of the delithiated one with their weights negated.  Where the two cells
agree -- most of the host -- the two contributions cancel, and what is left is the inserted ions
and the relaxation the insertion caused.  That is the object the voltage is a property of, and it
is what `difference_fields` builds.

Because a field difference is a field, every kernel in the program takes it unchanged: it is
bilinear in the weights, so nothing has to be re-derived for signed ones.  Two kernels are
available and the set decides which:

* **the periodic autocorrelation** (`cmfqsar.periodic`), which needs no common frame between
  *pairs* and so works across host frameworks.  This is what a collection of different hosts
  allows, and what `benchmarks/run_battery.py --stage difference` measures.
* **the periodic overlap kernel** of plan item M4, which compares fields point by point and so
  needs every pair on one parent lattice.  On a ladder of one host it is available and gives a
  co-field in the cell; across hosts it is not.

**What the two cells of a pair need, and why most pairs do not have it.** A difference is only
defined when the two cells stand on one lattice and hold the same host atoms.  Two cells relaxed
separately do not: their lattice constants differ by the volume change of the insertion, which is
the quantity a battery paper reports.  `difference_fields` therefore rebuilds the
delithiated cell in the lattice of the lithiated one, keeping its fractional coordinates, and
refuses a pair whose cell lengths differ by more than `tol` or whose host atom counts do not
match.  The refusal is not a failure of the method: it says the two cells are too far apart for a
difference to mean anything, and `usable_pairs` reports how much of a set survives it.

The scale of a difference field is not a free parameter but it does not matter either: the field
kernels of this program are cosine-normalised, so a difference field of two inserted ions and one
of one inserted ion are compared by their shape and not by their size.

### `class PairDifference  (dataclass)`

One pair that can carry a difference field, with what the construction found.

Attributes:

- `name: str`
- `fields: CrystalFields`
- `target: float`
- `group: str`
- `lattice_mismatch: float`
- `n_inserted: int`
- `inserted: np.ndarray` = `None`
- `discharged: object` = `None`
- `info: dict` = `None`

### `lattice_mismatch(a, b) -> float`

How far two cells are from one lattice: the largest relative difference of their lengths.

The lengths are sorted before they are compared, so the same lattice written with its vectors
in another order counts as the same lattice.  The angles are not compared, because a pair that
passes the length test and fails an angle test has not been seen in practice and the number
reported here should mean one thing.

### `host_counts(atoms, ion: str) -> dict`

The atoms of the cell by element, the working ion left out.

### `on_lattice(atoms, cell)`

The same structure in another lattice: its fractional coordinates, the given cell.

This is how the two cells of a pair are brought together.  It moves every atom by the
difference of the two lattices, which for a pair that passes `lattice_mismatch` is a
fraction of an angstrom, and it is the only honest way to subtract two fields: a field
difference taken between two different cells would measure the lattices and not the
insertion.

### `can_difference(charged, discharged, ion: str='Li', tol: float=LATTICE_TOL) -> Tuple[bool, str]`

Can these two cells carry a difference field, and if not, which test failed?

### `difference_fields(charged, discharged, fieldset: FieldSet, name: str='', ion: str='Li', tol: float=LATTICE_TOL) -> CrystalFields`

The field difference of one pair, as a `cmfqsar.periodic.CrystalFields`.

The atoms of the discharged cell keep their weights and the atoms of the charged cell, moved
into the discharged cell's lattice, carry theirs negated.  Radii, symbols and the cell come
with them, so every kernel and every feature matrix of the program takes the result as it
takes any other structure.  The charges of the two cells are computed in their own cells
first, which matters: the formal charge of the metal is not the same before and after the
insertion (plan item BT1), and that change is part of what the difference field carries.

### `inserted_positions(charged, discharged, ion: str='Li', tol: float=0.5) -> np.ndarray`

Where the working ions went in: the sites of the discharged cell with no partner.

Every ion of the charged cell is matched to the nearest ion of the discharged cell within
`tol` angstrom, the minimum image taken; the discharged sites left over are the inserted
ones.  It is a reading of the pair, not part of the kernel, and it is what a figure of the
difference field is centred on.

### `usable_pairs(pairs: Sequence, fieldset: FieldSet, tol: float=LATTICE_TOL, ion: str='Li', target=lambda p: p.voltage, group=lambda p: p.framework) -> Tuple[List[PairDifference], List[dict]]`

Every pair of a set that can carry a difference field, and why the others cannot.

`pairs` are the insertion pairs of `cmfqsar.battery`, or anything with `charged` and
`discharged` attributes; `target` and `group` say what is being predicted and what holds
the folds together.  The second list is the refusals, one record each, so a benchmark can
report how much of its set the frame costs instead of quietly dropping it.

### `shell_of_the_inserted_ion(pair: PairDifference, anion: str='O', cutoff: float=3.0) -> dict`

What surrounds the sites the ions went into: how many anions, and how far.

This is the reading the item asks for, as far as a set of different hosts allows one: the
coordination of the site the working ion takes, measured on the discharged cell itself rather
than read off a model.  A co-field in the cell needs every pair on one lattice, which a
collection of different hosts does not give.

## cmfqsar.chemotypes

Source: `cmfqsar/chemotypes.py`

Heterogeneous series: several chemotypes, no single common substructure.

Everything in this program rests on the molecules sharing a frame, and for a congeneric series
superposition on the common substructure supplies one: the alignment experiment rebuilt a published
alignment from scrambled poses to within 0.06 Å for the median molecule.  A series of several
scaffolds has no such substructure to superimpose on, and the usual procedure silently becomes
meaningless: one reference molecule, one maximum common substructure, one co-field, one random
cross-validation that puts a molecule's own scaffold-mates in the training part.

This module is the workflow for that case, in the order the problem imposes:

1. **Cluster by scaffold.**  `chemotype_labels` groups the molecules by their Bemis-Murcko
   framework, either as drawn or made generic so that ring atoms and bond orders stop mattering.
2. **Align within each cluster**, where a common substructure does exist and works well:
   `align_within_groups`.
3. **Place the clusters relative to one another**, which is the part a congeneric series never
   needs: `place_groups` moves each cluster rigidly, as a whole, either by maximising the
   field similarity of its reference with a chosen cluster or by docking that reference into the
   binding site.  Moving the cluster rigidly is what keeps step 2 intact.
4. **Judge the result by leave-one-chemotype-out**, `grouped_folds` and
   `grouped_cv_predict`.  Random folds overstate the quality of a multi-scaffold model,
   because the molecules that look most like the one being predicted are its own scaffold-mates and
   they sit in the training part; holding out a whole chemotype asks the question that matters.
5. **Compare against a baseline that needs no alignment at all**, `alignment_free_q2` over
   `count_kernel`: with a small attenuation factor a continuous indicator field tends to a
   count of atoms of its type, so the kernel becomes a fragment-descriptor kernel and the poses
   stop mattering.  By default the baseline is that limit, written down from the counts
   themselves, so no attenuation has to be chosen and the poses are gone exactly rather than
   nearly.  Two details decide whether an attenuated kernel reaches the same place, and both are
   measured in the tests: the attenuation has to be far smaller than a modelling one, since the
   pose dependence dies only linearly in it (at alpha = 0.02 a rigid motion of every molecule
   still moves the kernel by 0.7, at 2e-4 by 0.013), and the cosine normalisation has to be
   applied to the joint kernel rather than field by field, or the counts cancel.  If the aligned
   model cannot beat that baseline, the alignment is not worth its cost, and on a heterogeneous
   set that is the first thing worth knowing.

### `class HeteroResult  (dataclass)`

What the workflow found.

Attributes:

- `labels: np.ndarray`
- `names: List[str]`
- `q2_random: float` = `float('nan')`
- `q2_grouped: float` = `float('nan')`
- `q2_baseline: float` = `float('nan')`
- `q2_invariant: float` = `float('nan')`
- `rmse_grouped: float` = `float('nan')`
- `alpha: float` = `float('nan')`
- `lam: float` = `float('nan')`
- `per_group: List[dict]` = `_field(default_factory=list)`
- `log: List[str]` = `_field(default_factory=list)`

#### `HeteroResult.verdict() -> str`

### `scaffold_smiles(mol: Chem.Mol, generic: bool=False) -> str`

Bemis-Murcko framework of a molecule as canonical SMILES; "" has no rings.

### `chemotype_labels(mols: Sequence[Chem.Mol], generic: bool=False, min_size: int=1) -> Tuple[np.ndarray, List[str]]`

Group index per molecule and the scaffold of every group, largest group first.

Groups smaller than `min_size` are merged into one "other" group, so that a few
singletons does not turn leave-one-chemotype-out into leave-one-molecule-out.

### `group_members(labels: np.ndarray) -> List[np.ndarray]`

### `summarise_groups(labels: np.ndarray, names: Sequence[str]) -> str`

### `group_reference(mols: Sequence[Chem.Mol], members: Sequence[int]) -> int`

The molecule of a group that others are superimposed on: the largest one.

### `align_within_groups(mols: Sequence[Chem.Mol], labels: np.ndarray, progress=None, cancel=None) -> List[str]`

Superimpose the molecules of every group on that group's reference. In place.

### `apply_transform(mol: Chem.Mol, R: np.ndarray, t: np.ndarray) -> None`

### `place_groups(mols: Sequence[Chem.Mol], labels: np.ndarray, fieldset: FieldSet, mode: str='field', reference_group: int=0, pocket=None, restarts: int=2, seed: int=0, tether_radius: float=8.0, progress=None, cancel=None) -> List[str]`

Move every cluster, as a rigid body, into a common frame.

`field` superimposes each cluster's reference on the reference of `reference_group` by
maximising the CMF kernel between them, which needs no common substructure.  `fieldgraph`
superimposes them on the largest consistent set of their typed field extrema instead
(`cmfqsar.fieldgraph`), which needs no common substructure either and, unlike `field`,
does not have to start from a pose that already roughly overlaps: a kernel maximisation gives
no information where two molecules do not overlap, which is exactly the position two chemotypes
are in.  `dock` places each cluster's reference in the binding site instead, which needs a
pocket but no reference cluster at all.  Either way the transform found for the reference is
applied to
every molecule of its cluster, so the within-cluster alignment of step 2 survives untouched.

### `grouped_folds(labels: np.ndarray, max_folds: int=0) -> List[np.ndarray]`

Folds that never split a chemotype: one fold per group, or groups bundled into `max_folds`.

This is the fair cross-validation for a heterogeneous series.  With random folds a molecule
is predicted from its own scaffold-mates, which is a question nobody is asking.

### `grouped_cv_predict(K: np.ndarray, y: np.ndarray, labels: np.ndarray, method: str='krr', params: Optional[Dict[str, float]]=None, fit_bias: bool=True, max_folds: int=0) -> np.ndarray`

Cross-validated predictions with a whole chemotype held out at a time.

### `count_kernel(mols: Sequence[Chem.Mol], names: Sequence[str], alpha: Optional[float]=None, cif_min: int=3, scheme: str='sybyl') -> np.ndarray`

The kernel of the alignment-free baseline: indicator fields at vanishing attenuation.

The field of Sybyl type *t* is a sum of unit Gaussians on the atoms of that type, so the
overlap integral of two such fields is

    K_t(A, B) = (pi/alpha)^{3/2} sum_{l in A} sum_{m in B} exp(-alpha |r_l - r_m|^2 / 4)
              = (pi/alpha)^{3/2} [ n_t(A) n_t(B) - (alpha/4) sum_lm d_lm^2 + O(alpha^2) ],

so as alpha tends to zero K_t tends to n_t(A) n_t(B) times a factor common to every pair,
which the cosine normalisation then removes.  The joint kernel tends to the dot product of
the atom-type count vectors and nothing of the geometry survives.

`alpha = None`, the default, writes that limit down directly: each field is scaled to unit
mean diagonal, which in the limit is a division by the mean of n_t^2, and the result is the
cosine of the count vectors.  It is the exact limit rather than an approach to it, so there
is no attenuation left to choose badly, and it costs one pass over the atoms instead of a
kernel over every pair of them (10 to 34 times faster on the sets in `examples/`).  Passing
an explicit `alpha` takes the Gaussian route instead, which is the thing the limit is a limit
of; the two agree to 4e-5 by alpha = 2e-6.

Either way the normalisation goes on the joint kernel, not field by field.  A field
normalised on its own tends to n_t(A) n_t(B) / (n_t(A) n_t(B)) = 1, so the counts cancel and
what is left is the indicator of "both molecules contain type t": the baseline would lose the
very descriptors it is supposed to be built from.

### `invariant_q2(molfields: Sequence, y: np.ndarray, labels: np.ndarray, lam=1.0, max_folds: int=0, d_max: float=12.0, step: float=0.75, sigma: float=0.75, chirality: bool=False, chirality_shells: int=4) -> Tuple[float, float]`

Leave-one-chemotype-out q2 of a model that keeps the geometry but needs no alignment.

The kernel is the field autocorrelation of `cmfqsar.invariant`: invariant to rotation
and translation, so no superposition is involved, but still a function of the conformation.
It sits between the aligned model and the count kernel, and says what the geometry is worth
once the frame is no longer assumed.

### `alignment_free_q2(mols: Sequence[Chem.Mol], names: Sequence[str], y: np.ndarray, labels: np.ndarray, alpha: Optional[float]=None, lam=1.0, cif_min: int=3, max_folds: int=0, scheme: str='sybyl') -> Tuple[float, float]`

Leave-one-chemotype-out q2 of a model that ignores the poses entirely.

`alpha` is the attenuation of the indicator fields, or `None` for the limit they tend to,
which is the default and what makes the baseline alignment-free.  `lam` may be one value or a
grid, which is then searched on the same folds the aligned model is tuned on, so that the two
representations are compared under one protocol.  Returns (q2, rmse).

### `evaluate(mols: Sequence[Chem.Mol], names: Sequence[str], y: np.ndarray, labels: np.ndarray, group_names: Sequence[str], fieldset: FieldSet, alphas: Sequence[float]=(0.3, 0.5, 1.0), lams: Sequence[float]=(0.01, 0.1, 1.0), max_folds: int=0, normalize: bool=True, baseline_alpha: Optional[float]=None, progress=None, cancel=None) -> HeteroResult`

Build the aligned model, judge it chemotype by chemotype, and compare with the baseline.

The hyper-parameters are chosen on the same grouped folds the model is reported on, which is
optimistic in the usual way of any selection on the reported statistic; what matters here is
the comparison between representations, which share the protocol.

## cmfqsar.cli

Source: `cmfqsar/cli.py`

Command-line interface for building CMF 3D-QSAR models (no GUI needed).

Installed, it is the `cmf` command (`pyproject.toml`); from a checkout, `python cmf_cli.py`
runs the same `main`.

Examples
--------
  python cmf_cli.py build examples/benzamides_synthetic.sdf --activity pActivity         --alphas 0.3,0.5,0.8,1.2,2 --lams 1e-3,1e-2,0.1,1,10 --test-fraction 0.25 --save model.cmf

  # literature data set with MOL2 charges, a separate activity table carrying the published
  # train/test split (set column 1/2) and the protocol of the 2013 paper
  python cmf_cli.py build ace-all.mol2 --activity-file ace-data.txt --charges file --param tripos         --raw-scale --cv-no-shuffle --lam-search brent --alphas 0.01,0.05,0.1,0.25,0.5,0.75,1,2,3,5,10         --lams 0.01,20 --nested --nested-on all

  python cmf_cli.py predict model.cmf new_molecules.sdf --out predictions.csv

### `cmd_build(args)`

### `cmd_build_crystals(args)`

A model on periodic structures: the periodic field autocorrelation of cmfqsar/periodic.py.

### `cmd_diagnose(args)`

Ask a data set the questions that should precede modelling it (item C6).

### `cmd_screen(args)`

Score a library block by block, never holding more than one block of it.

### `cmd_align(args)`

Put a set of molecules in one frame, by whichever of the four criteria is asked for.

### `cmd_lowrank(args)`

Fit a model in the space of a few hundred landmarks instead of the molecules (item B6).

### `cmd_isosteres(args)`

Which groups are interchangeable at a position, by the co-field (plan item MC1).

### `cmd_restriction(args)`

Score a conformational restriction against a saved model (plan item MC9).

### `cmd_cofield(args)`

The co-fields of a saved model on a grid, as cube files and as numbers.

### `cmd_benchmark(args)`

Run, list or check the benchmarks, which live in benchmarks/run_all.py.

### `cmd_predict(args)`

### `cmd_report(args)`

One self-contained HTML file describing a saved model: item F4 of the plan.

Everything comes out of the model file, so the report cannot disagree with the model and
costs no refitting; what the file does not carry (a validation that was never run) is
reported as missing rather than quietly left out.

### `cmd_suggest(args)`

Rank posed candidates by an acquisition rule: item E2 of the plan.

### `cmd_design(args)`

What to put at a position of a parent compound, ranked by the model: item E3.

The model is a topomer model built here from the series, because the suggestion has to be
posed by the same rule as the molecules it is being compared with, and because a model
fitted on superimposed whole molecules has no way to place a group nobody has made.

`--kernel autocorrelation` takes the other route (plan item SC7): the rotation-invariant
kernel needs no frame, so the series needs no common scaffold and a candidate is built as a
whole molecule and embedded once.  That is what makes a design loop possible on a table of
guests, which share no scaffold at all; the core is then only what names the position being
varied on the parent.

### `cmd_atoms(args)`

What a file says about its atoms: a target that belongs to them (plan item LP1).

There is no model over atoms yet -- the local kernel is the item that adds one -- so what
this does is the part that has to be right first: read the target with its blanks, say what
was measured and where, show the folds that hold out whole molecules, and write the whole
thing back in the form every program can read.

### `cmd_local(args)`

A model over atoms: the field around an atom against a count around it (plan item LP2).

The sample is an atom in a molecule, not the molecule. The descriptor is the molecule's
fields under a Gaussian envelope about the atom, turned into the rotation-invariant
autocorrelation, so nothing has to be aligned. The folds hold out whole molecules, and
`--reference` adds the atom-centred count that every such model has to beat.

### `cmd_hops(args)`

A model over migration paths: the field around a hop against a count around it (BT7).

The sample is a hop of the mobile ion between two sites of a crystal, with the ion itself
removed; the descriptor is the cell's fields under a Gaussian envelope about the midpoint,
in the frame of the hop and averaged over the rotation about it, which is a closed form.
The folds hold out whole structures.  Labels come from the bond-valence landscape
(`--bvse`) or from a table (`--labels`).

### `cmd_transfer(args)`

How much two models ask the same thing of the chemistry: item F3 of the plan.

### `cmd_mixing(args)`

The posterior over the field weights: item E6 of the plan.

### `cmd_explain(args)`

Shapley decomposition of a prediction: works for linear and nonlinear kernels alike.

### `cmd_complement(args)`

Field complementarity of a data set with a protein pocket.

### `cmd_mil(args)`

Multi-instance learning over conformations: let the model choose the pose.

### `cmd_topomer(args)`

Model a series by its substituents, each posed by rule instead of superimposed.

### `cmd_difference(args)`

What one molecule's fields have that another's do not.

### `cmd_pharmacophore(args)`

Discretise the co-fields of a fitted model into typed features with tolerance spheres.

### `cmd_hetero(args)`

The workflow for a series of several chemotypes, in the order the problem imposes.

### `cmd_unalign(args)`

Give every molecule of a structure file a random pose (control for alignment methods).

### `cmd_compare(args)`

Are two sets of molecules drawn from the same population?  A kernel two-sample test.

### `cmd_cluster(args)`

Cluster a set by the field kernel itself, and compare that with its scaffolds.

### `cmd_kcca(args)`

What the field sees that another view does not: item F6 of the plan.

### `cmd_map(args)`

Chemical space map and activity landscape (kernel PCA + GTM) from the CMF kernels.

### `main(argv=None)`

## cmfqsar.coalign

Source: `cmfqsar/coalign.py`

Co-alignment of a training set, treated as what it is: expectation-maximisation over poses.

`cmfqsar.varalign.cv_align_training_set` already aligns a training set by the
posterior-variance criterion, fold by fold so that no molecule is ever aligned against a model
containing it.  Its own docstring records what measuring it found: the iteration converges, the
contraction of the ensemble is mild, and the predictive gain is *not* reliable -- one set improved,
one degraded, two were neutral, and rounds after the first consistently made the external
prediction worse.  So it is run for one round with a displacement guard, and the outcome is left to
chance.

That is not a method, it is a hope with a default setting.  What the loop actually is, is an EM
iteration: the poses are latent variables, the M-step fits the model to the current poses, the
E-step moves each molecule to where the model (fitted without it) is most confident.  Written that
way it needs the three things an EM iteration needs, and this module adds them.

**An objective, and an acceptance test.**  A round is kept only if it improves the quantity the
iteration is allowed to improve: the **leave-fold-out error of the activities**.  This is a valid
objective here, and the reason is worth stating because the obvious alternative is not.  q2 of the
realigned set is invalid -- realigning changes the geometry q2 is computed on, and in the earlier
tests q2 and the external R2 moved in opposite directions.  But the *pose* of each molecule is
chosen from the model's confidence alone and never from its activity, so predicting that
molecule's activity afterwards, with the fold model that did not contain it, is a fair
measurement.  A round that does not improve it is rejected and the previous poses are restored.

**An annealed ridge.**  Early rounds run with a large lambda, where the model is smooth and the
confidence landscape has few spurious minima, and later rounds sharpen it.  Annealing is what
keeps an EM iteration from settling on a bad frame in the first round.

**A contraction stop.**  The degeneracy of the criterion is that superimposing everything makes
every variance zero, so the iteration is watched for it: the mean off-diagonal normalised kernel
(`cmfqsar.varalign.ensemble_similarity`) is reported every round and the run stops if it
rises past a tolerance, whatever the objective says.

The E-step may be rigid or flexible.  The flexible one is the measured better choice -- C2 found
that confidence with the torsions and conformer-ensemble starts recovers held-out poses to 0.86 A
median against 2.27 A for the rigid search -- and it needs the RDKit molecules, which the field
representation does not carry, so `rdmols` must be passed for it.

Judged under C1 only (`benchmarks/run_poses.py --coalign`): the training poses are destroyed,
recovered by this iteration, and compared with the crystal poses they should have found.

### `class CoalignRound  (dataclass)`

What one round of the iteration did, and whether it was kept.

Attributes:

- `round: int` = `0`
- `lam: float` = `float('nan')`
- `moved_mean: float` = `0.0`
- `moved_max: float` = `0.0`
- `drift: float` = `0.0`
- `similarity: float` = `float('nan')`
- `loo_rmse: float` = `float('nan')`
- `accepted: bool` = `False`
- `note: str` = `''`

#### `CoalignRound.describe() -> str`

### `class CoalignResult  (dataclass)`

The outcome: the poses kept, the objective at every round, and why it stopped.

Attributes:

- `rounds: List[CoalignRound]` = `field(default_factory=list)`
- `loo_rmse_start: float` = `float('nan')`
- `similarity_start: float` = `float('nan')`
- `best_round: int` = `0`
- `stopped: str` = `''`

#### `CoalignResult.loo_rmse` -> float  (property)

#### `CoalignResult.improved` -> float  (property)

How much the objective fell; negative means the iteration found nothing.

#### `CoalignResult.describe() -> str`

### `leave_fold_out_error(mols: Sequence[MoleculeFields], y: np.ndarray, model_template, lam: float, folds: int=5, seed: int=0) -> float`

RMSE of the activities predicted fold by fold, with the poses as they currently are.

The objective the iteration is allowed to improve: every molecule is predicted by a model
that does not contain it, and no pose was chosen using an activity.

### `coalign(mols: Sequence[MoleculeFields], y: Sequence[float], model_template, rounds: int=4, folds: int=5, max_rmsd: Optional[float]=1.0, lam_start: Optional[float]=None, lam_factor: float=0.5, flexible: bool=False, rdmols: Optional[Sequence]=None, conformers: int=0, restarts: int=0, max_similarity: Optional[float]=None, energy_slack: float=10.0, tol: float=0.01, seed: int=0, progress: Optional[Callable[[str, float], None]]=None, cancel: Optional[Callable[[], bool]]=None) -> CoalignResult`

Co-align a training set by confidence, as an EM iteration with an acceptance test.

The molecules are modified in place and left in the poses of the last **accepted** round, so
a run that finds nothing leaves them exactly as they arrived.  `lam_start` defaults to four
times the model's own ridge and is multiplied by `lam_factor` each round.

`tol` is the *relative* improvement a round must show to be kept, and it is not a detail.
With a threshold of machine epsilon the first measurement accepted a round on eg5 whose
objective fell by less than one part in a hundred thousand -- leave-fold-out RMSE 0.479 to
0.479 -- while the molecules moved 0.40 A and the frame drifted from 0.07 to 0.25 A from the
crystal one.  An accepted round with no real gain is worse than no round at all, because the
poses move.

## cmfqsar.cofields

Source: `cmfqsar/cofields.py`

Molecular co-fields (fields of regression coefficients), molecular fields and
their overlap evaluated on a rectangular grid for visualisation (isosurfaces)
and export.  Also: Gaussian-cube export.

Co-field of field type f (Baskin & Zhokhova 2013, Eq. 9; "molecular co-field"
in Baskin & Zhokhova 2019):

    C_f(r) = sum_j a_j sum_m w_fjm exp(-b_jm/2 |r - r_jm|^2)

Primal form of the model (Eq. 7):  y = sum_f h_f  INT C_f(r) X_f(r) d^3r + b

### `class Grid  (dataclass)`

Attributes:

- `origin: np.ndarray`
- `spacing: float`
- `shape: Tuple[int, int, int]`

#### `Grid.n_points` -> int  (property)

#### `Grid.axes()`

#### `Grid.points() -> np.ndarray`

#### `Grid.extent`  (property)

### `class CofieldError  (dataclass)`

A co-field with its error bar: the mean map, its standard deviation, and the ratio.

Attributes:

- `mean: np.ndarray`
- `sd: np.ndarray`
- `sigma: float`
- `sigma_from: str`
- `z: float` = `2.0`

#### `CofieldError.zscore` -> np.ndarray  (property)

C_f(r) / sd(r), zero where the training molecules put no field at all.

#### `CofieldError.masked` -> np.ndarray  (property)

The co-field with everything below `z` standard deviations set to zero.

#### `CofieldError.surviving` -> float  (property)

The fraction of the drawn volume the mask keeps.

"Drawn" is the part a picture shows at all -- a tenth of the peak and above, which is
well below any isosurface anyone draws -- so this is the fraction of what a reader would
have seen that is more than `z` standard deviations from zero.

#### `CofieldError.describe() -> str`

### `class CofieldMask  (dataclass)`

A soft region of space: a sum of Gaussians on the atoms the model rewards most.

Everything else in this program compares two molecules over the whole of both of them, which
is right for a congeneric series and wrong for a replacement scaffold.  A candidate that
carries the same field where the co-field has weight is the one worth making, and what it
does elsewhere is its own business -- that is the whole content of plan item MC5, and this is
the region "elsewhere" is measured against.

`centres`, `weights` and `widths` are a Gaussian expansion like any other in this module, so
the mask can be drawn, exported as a cube and multiplied into a co-field with
`cmfqsar.kernels.gaussian_product`.  `atoms` and `contributions` say which atoms it was
built from and what each contributed, because a mask nobody can explain is not useful.

Attributes:

- `centres: np.ndarray`
- `weights: np.ndarray`
- `widths: np.ndarray`
- `atoms: np.ndarray` = `None`
- `molecules: np.ndarray` = `None`
- `contributions: np.ndarray` = `None`

#### `CofieldMask.expansion() -> Tuple[np.ndarray, np.ndarray, np.ndarray]`

#### `CofieldMask.volume(grid: Grid) -> np.ndarray`

The mask itself on a grid, so that it can be drawn beside the co-field it cuts.

#### `CofieldMask.describe() -> str`

#### `CofieldMask.peak() -> float`

The largest value the mask takes at one of its own centres.

### `make_grid(coords: np.ndarray, margin: float=3.0, spacing: float=0.5, max_points: int=1500000) -> Grid`

Rectangular grid enclosing `coords` with a margin; the spacing is coarsened if needed.

### `points_per_chunk(points: np.ndarray, tree, cutoff: float, chunk: int, sample: int=16) -> int`

How many points to put in one neighbour list, from how many centres a point sees.

The count is measured on a few points rather than assumed, because it is the one quantity
that differs by orders of magnitude between a molecular co-field and a periodic one.

### `evaluate_gaussians(points: np.ndarray, centers: np.ndarray, coeffs: np.ndarray, widths: np.ndarray, eps: float=1e-05, chunk: int=20000) -> np.ndarray`

`sum_m coeffs[m] exp(-widths[m]/2 |p - centers[m]|^2)` for every point p (sparse evaluation).

`chunk` is an upper bound on the points evaluated at once; the real one is
`MAX_PAIRS`, which bounds the neighbour list of a chunk and so the memory this needs,
whatever the density of the centres.

### `cofield_volume(model: CMFModel, f: int, grid: Grid, include_scale: bool=False) -> np.ndarray`

C_f(r) on the grid (shape = grid.shape).  If include_scale, multiply by h_f c_f.

### `evaluate_gaussian_gradients(points: np.ndarray, centers: np.ndarray, coeffs: np.ndarray, widths: np.ndarray) -> np.ndarray`

The gradient of `sum_j c_j exp(-w_j |r - r_j|^2 / 2)` at every point.  Shape (n, 3).

This is what the electric-field kind of plan item EC5 asks to see: its kernel pairs the
gradients of two fields, so the quantity with a reading is the gradient of the co-field, not
the co-field itself.  An arrow at a point says which way the model wants the field to change
there, and how strongly.

### `cofield_gradient(model: CMFModel, f: int, grid: Grid, include_scale: bool=False) -> np.ndarray`

The gradient of the co-field of field `f` on the grid.  Shape (3, *grid.shape).

### `cofield_arrows(model: CMFModel, f: int, grid: Grid, count: int=20, min_separation: float=1.5)`

The strongest arrows of the co-field gradient, for drawing or for a table.

Returns `(points, vectors, lengths)`.  The points are taken in order of decreasing arrow
length, and a point is skipped when it lies within `min_separation` angstrom of one already
taken, so that the answer spreads over the molecule instead of crowding one peak.

### `field_volume(mol: MoleculeFields, spec, f: int, grid: Grid) -> np.ndarray`

Molecular field X_f(r) of `mol` (field f described by FieldSpec `spec`) on the grid; no model needed.

### `molfield_volume(model: CMFModel, mol: MoleculeFields, f: int, grid: Grid, primal_scale: bool=False) -> np.ndarray`

Molecular field X_f(r) of `mol` on the grid.

### `difference_volume(candidate: MoleculeFields, reference: MoleculeFields, spec, f: int, grid: Grid) -> np.ndarray`

X_f(candidate) - X_f(reference) on the grid.

Positive where the candidate puts field the reference does not have, negative where it fails
to reproduce field the reference has.  Both molecules have to be in the same frame already,
which is the usual condition of everything here.

### `shapley_volume(mol: MoleculeFields, spec, values: np.ndarray, grid: Grid, normalized: bool=True) -> np.ndarray`

Shapley field: the per-atom values of one field type smeared over its own Gaussians:

    Phi(r) = sum_l phi_l (b_l / 2 pi)^(3/2) exp(-b_l/2 |r - r_l|^2)

With `normalized` each Gaussian integrates to one, so the integral of the map over space is
the sum of the values, which is the part of the prediction attributed to this field: the same
relation that holds between partial charges and the charge density they generate.  Unlike a
co-field this map belongs to one molecule, because a model with a nonlinear kernel has no
co-field to belong to.

### `overlap_volume(model: CMFModel, mol: MoleculeFields, f: int, grid: Grid) -> np.ndarray`

Integrand h_f c_f C_f(r) X~_f(r) whose integral is the contribution of field f to y.

### `isosurface(volume: np.ndarray, level: float, grid: Grid)`

Marching cubes.  Returns (verts, faces, normals) in world coordinates or None.

### `export_cube(path: str, grid: Grid, volume: np.ndarray, mol: Optional[MoleculeFields]=None, title: str='CMF co-field') -> None`

Write a Gaussian cube file (coordinates converted to Bohr).

### `dual_covariance_factor(model: CMFModel) -> np.ndarray`

G, with Cov[a] = sigma^2 G G^T, for the dual coefficients of a kernel ridge fit.

The coefficients are a linear map of the activities -- with a bias, the bordered system of
`cmfqsar.model.krr_fit`, whose inverse's top-left block is this G; without one, simply
(K + lam I)^-1.  So activities carrying independent noise of variance sigma^2 give
coefficients carrying covariance sigma^2 G G^T, and every linear functional of them carries
the matching variance.  Nothing here is a Bayesian posterior: see `cofield_variance`.

The Laplacian-regularised ridge (method `lapkrr`) is the same kind of map, with two
differences: its system matrix carries the smoothness penalty, and only the molecules that
have an activity carry noise, so G has one column per labelled molecule rather than per
training molecule.  `G G^T` is the same size either way, and so is everything downstream.

### `noise_level(model: CMFModel, sigma: Optional[float]=None) -> Tuple[float, str]`

The noise level to propagate, and where it came from.

The out-of-fold error is the default and the reliable estimate: it is what the model's own
predictions miss by, and it includes the part of the error that is the model's fault rather
than the assay's, which errs in the conservative direction.  The residual of the fit would be
far too small -- a model with a small lambda reproduces its training set almost exactly,
which says nothing about the noise in it.

### `per_molecule_fields(model: CMFModel, f: int, points: np.ndarray, chunk: int=0, basis=None) -> np.ndarray`

(n_train, n_points): the normalised field of every training molecule at every point.

This is the basis the co-field is a combination of: `C_f = coef @ X`, and the variance of
the co-field is read off the same matrix, so the two cannot disagree.  `basis` replaces the
Gaussians of `CMFModel.cofield_basis` by another set over the same training molecules,
which is how `atom_effect` reads the same quantity convolved with one atom.

### `cofield_variance(model: CMFModel, f: int, grid: Grid, sigma: Optional[float]=None, include_scale: bool=False, chunk: int=0) -> np.ndarray`

Var[C_f(r)] on the grid: how far the co-field moves when the activities carry noise.

The co-field is linear in the dual coefficients, `C_f(r) = sum_j a_j x_j(r)`, and the
coefficients are a linear map of the activities, so with `Cov[a] = sigma^2 G G^T`

    Var[C_f(r)] = sigma^2 || G^T x(r) ||^2

in closed form: one matrix product per block of grid points, no resampling and no refitting.

**What this is and is not.**  It is tempting to call it the posterior variance of the
co-field, and the plan item that asked for it did.  It is not, and the reason is worth
stating.  Under the Gaussian process whose kernel this is, the prior on the field weight
function is white noise on L^2: its value at a point is not a random variable at all, and the
pointwise posterior variance is infinite everywhere.  What the prior does allow is the
variance of `<w, g>` for a probe field g of non-zero width, which smooths the point into a
Gaussian -- that is exactly the GP variance of a prediction for a one-atom molecule placed
there, and `CMFModel.predict_with_variance` already computes it.

What is computed here is the other half of the question, with the training molecules and the
kernel held fixed and the noise in the activities propagated.  It answers *how much of this
lobe would survive the error in the measurements*, which is the question a co-field picture
raises.  It says nothing about regions where the training set put no field at all:
there the co-field and its error are both zero, and the ratio of the two is not informative.

### `cofield_at_points(model: CMFModel, f: int, points: np.ndarray, include_scale: bool=True) -> np.ndarray`

C_f at a list of points rather than on a grid.

A co-field read at a few places a chemist can name -- the ortho, meta and para positions of a
ring, the four corners of a pocket -- is a table, not a picture, and this is what fills it.

### `cofield_covariance_at(model: CMFModel, f: int, points: np.ndarray, sigma: Optional[float]=None, include_scale: bool=True, chunk: int=0, factor: Optional[np.ndarray]=None) -> np.ndarray`

Cov[C_f] at a list of points, under noise of level `sigma` in the activities.

The diagonal is what `cofield_variance` gives on a grid.  The off-diagonal terms are
what a *difference* between two places needs: the co-field at two points a few angstrom
apart is built from the same training fields, so the two values are correlated and the error
of their difference is not the two errors added -- usually it is far smaller.  Without this
a comparison between two positions would be judged against the wrong error bar.

`factor` is the matrix `dual_covariance_factor` returns.  It does not depend on the
field, and computing it costs a solve with as many right-hand sides as there are training
molecules, so a caller reading the co-fields of twenty fields at the same points should
compute it once and pass it here instead of paying for it twenty times.

### `contrast_at(model: CMFModel, f: int, points: np.ndarray, weights: Sequence[float], sigma: Optional[float]=None, include_scale: bool=True, factor: Optional[np.ndarray]=None) -> Tuple[float, float]`

A weighted combination of the co-field at a few points, and its standard deviation.

`weights` picks the comparison: (0, 0, -1, 1) at the four ring positions is "para minus
meta".  The standard deviation uses the whole covariance of `cofield_covariance_at`, so
the correlation between the points is taken into account rather than ignored.  `factor` is
passed through, so several comparisons on one model share the one expensive solve.

### `probe_basis(centres, coeffs, widths, width: float, weight: float=1.0)`

The same Gaussian basis, convolved with one atom's own Gaussian of width `width`.

The overlap of two Gaussians is a Gaussian, which is what makes every integral in this
program closed form:

    int exp(-b/2 |r - c|^2) exp(-bp/2 |r - p|^2) dr
        = (2 pi / (b + bp))^(3/2) exp(-(b bp) / (2 (b + bp)) |p - c|^2),

the same expression `cmfqsar.kernels` sums over atom pairs.  So a basis evaluated at a
point becomes the overlap with an atom placed there by scaling every coefficient and widening
every width, and nothing else in this module has to change.

### `atom_effect(model: CMFModel, f: int, points: np.ndarray, width: Optional[float]=None, weight: float=1.0, include_scale: bool=True) -> np.ndarray`

What one atom placed at each point would add to the prediction.

A co-field says what the prediction gains per unit of field at a point, which is not a number
a chemist can check against anything: a field is not something one can put somewhere.  An atom
is.  This is the overlap of the co-field with one atom's own Gaussian, so the result is in the
units of the measured value -- ppm for a shift, pKa units for an acidity -- and can be read
beside a table of measured substituent effects.

For a continuous indicator field the weight of an atom of its type is 1, so the default
`weight=1` is one such atom; for the electrostatic field the weight is a partial charge and
the caller should pass it.  `width` defaults to the field's own attenuation, which is the
Gaussian the kernel puts on an atom of that field.

### `atom_effect_covariance(model: CMFModel, f: int, points: np.ndarray, width: Optional[float]=None, weight: float=1.0, sigma: Optional[float]=None, include_scale: bool=True, chunk: int=0, factor: Optional[np.ndarray]=None) -> np.ndarray`

Cov of `atom_effect` under noise of level `sigma` in the measured values.

### `cofield_with_error(model: CMFModel, f: int, grid: Grid, sigma: Optional[float]=None, include_scale: bool=False, z: float=2.0, chunk: int=0) -> CofieldError`

The co-field, its standard deviation and the threshold that decides what is drawn.

### `mask_noise(volume: np.ndarray, sd: np.ndarray, z: float=2.0) -> np.ndarray`

`volume` with everything below `z` standard deviations of zero set to zero.

Zero rather than nan, so the result is still a field an isosurface can be drawn on: a lobe
keeps its shape where it is significant and is cut where it is not.  Where the error is zero
the value is kept, because there the co-field is zero as well and there is nothing to cut.

### `mask_width(model: CMFModel, width: Optional[float]=None) -> float`

The width of the mask's Gaussians: by default the attenuation the kernel itself uses.

A mask is a region and not a field, so it has one width for every field of the model.  The
default is the mean attenuation of the fields that put one Gaussian of a fixed width on an
atom, which is the Gaussian the kernel compares atoms with; a field whose width is per atom
(the radius-scaled steric field) has no single value to contribute and is left out.

### `top_atom_mask(model: CMFModel, mols, k: int=10, width: Optional[float]=None, normalise: bool=True, per_field: bool=False) -> CofieldMask`

A mask on the `k` atoms of `mols` whose contribution to the prediction is largest.

`mols` is one `cmfqsar.fields.MoleculeFields` or several -- the training molecules the
mask should describe, usually the series the model was fitted on.  Atoms are ranked by the
absolute value of their contribution (decomposition rule 3), pooled over the molecules given,
so a superimposed series votes on the region rather than one molecule choosing it.

With `normalise` the weights are scaled so that the mask's largest value is 1, which keeps a
masked overlap in the units of the unmasked one and makes the two comparable.

### `masked_cofield(model: CMFModel, f: int, mask: CofieldMask, tol: float=1e-09)`

The Gaussians of `C_f(r) m(r)`: the co-field where the mask is, and nothing elsewhere.

A product of two Gaussian expansions is a Gaussian expansion
(`cmfqsar.kernels.gaussian_product`), so the masked co-field is the same kind of object
as the co-field and every reader of one works on it.  `tol` drops the product terms whose
weight is a negligible fraction of the largest, which removes the pairs that are far apart.

## cmfqsar.complementarity

Source: `cmfqsar/complementarity.py`

Compatibility of molecular fields and co-fields with the fields of a protein pocket.

A pocket is a set of atoms, so the parameterisation that builds the fields of a ligand builds the
fields of the protein without any new physics, and the overlap of two Gaussian fields has the same
closed form whichever side it comes from.  Four questions can then be answered with numbers rather
than with a glance at a picture:

1. **How complementary is a molecule to the pocket?**  Overlap integrals between the ligand fields
   and the pocket fields, normalised to a cosine so that the size of the molecule has no effect.  Complementarity is not
   similarity: the electrostatic term counts with a minus sign, because a positive ligand region
   should meet a negative protein region, while the hydrophobic term counts with a plus and the
   hydrogen-bond terms pair the donors of one side with the acceptors of the other.
2. **Where on the molecule does it hold?**  The same quantities on the surface of the molecule,
   which gives a score per vertex to colour the surface with, and, for electrostatics, the
   correlation of the two potentials over the surface, which is the published electrostatic
   complementarity score (Bauer & Mackey, J. Comput. Aided Mol. Des. 2009; J. Med. Chem. 62 (2019)
   3036).  Across a congeneric series this score is reported to track activity.
3. **Does the model agree with the protein?**  A co-field says where more field raises activity; the
   pocket says what it offers.  Their weighted correlation over the space the series occupies is a
   test of a model that never saw the protein.  A strong co-field lobe where the pocket offers
   nothing is a sign that the model is fitting the alignment or a correlated substituent instead of
   an interaction.
4. **Which residues say so?**  The overlap integral is a double sum over atom pairs, so keeping
   the per-atom column sums instead of the grand total costs no extra work and resolves every
   score of question 1 by receptor residue.  That turns one cosine into an interaction profile, which can
   be compared with the profile of a reference ligand, averaged over several complexes into a
   consensus pattern, and used to require the conserved contacts of a target rather than every
   contact of one ligand.  A global score cannot say any of this, and neither can the ligand
   RMSD: a pose can move very little and still lose the interaction that mattered.

What this is not: a binding energy.  Desolvation, entropy, bridging waters and side-chain motion are
all absent, and the co-field is unconstrained wherever no molecule of the series reaches, which is
why the agreement is weighted by occupancy.

### `class Complementarity  (dataclass)`

Scores of one molecule against one pocket.

Attributes:

- `name: str`
- `terms: Dict[str, float]` = `_field(default_factory=dict)`
- `raw: Dict[str, float]` = `_field(default_factory=dict)`
- `burial: float` = `float('nan')`
- `clashes: int` = `0`
- `worst_clash: float` = `0.0`
- `surface: Dict[str, float]` = `_field(default_factory=dict)`
- `n_surface: int` = `0`

#### `Complementarity.score` -> float  (property)

Mean of the electrostatic, hydrophobic and hydrogen-bond terms, in [-1, 1].

#### `Complementarity.describe() -> str`

### `class InteractionProfile  (dataclass)`

The complementarity of one molecule with one pocket, resolved by receptor residue.

`terms` is (residues, interactions) and carries the same normalised, signed quantities as
`Complementarity`, so each column sums over the residues to exactly the matching
entry of `Complementarity.terms`.  A global cosine says how well a molecule suits the
pocket; this says which residues say so, and it disagrees usefully with the ligand RMSD,
since a pose can move very little and still lose the one interaction that mattered.

Attributes:

- `name: str`
- `residues: List[str]` = `_field(default_factory=list)`
- `labels: List[str]` = `_field(default_factory=list)`
- `terms: np.ndarray` = `_field(default_factory=lambda: np.zeros((0, 0)))`
- `contacts: np.ndarray` = `_field(default_factory=lambda: np.zeros(0))`

#### `InteractionProfile.vector() -> np.ndarray`

The profile as one flat vector, residue by residue; the order is `residues`.

#### `InteractionProfile.total() -> Dict[str, float]`

Column sums: the whole-pocket term each interaction contributes.

#### `InteractionProfile.per_residue() -> Dict[str, float]`

Row sums: how much each residue contributes, over all interaction types.

#### `InteractionProfile.top(n: int=8, favourable: bool=True, minimum: float=1e-06) -> List[Tuple[str, str, float]]`

The `n` strongest (residue, interaction, value) entries, largest magnitude first.

A term is reported only above `minimum`.  The entries are fractions of a cosine, so a
Gaussian overlap that has decayed to 1e-100 is not a weak interaction but no interaction
at all, and listing it as the strongest one of a distant molecule is simply wrong.

#### `InteractionProfile.bits(threshold: float=0.02) -> np.ndarray`

The profile as a binary interaction fingerprint: one bit per favourable term.

#### `InteractionProfile.describe(n: int=5) -> str`

### `ionise(mol) -> int`

Give the standard side chains their charge at pH 7, in place; returns how many were set.

A PDB file carries no charges, so a reader leaves aspartate and glutamate protonated and
lysine neutral.  Left alone that misplaces both the electrostatic field and, once hydrogens
are added, the donor field, which is exactly what a complementarity score is made of.
Histidine is left neutral, the usual assumption.

### `pocket_molecule(protein: Protein, add_hydrogens: bool=True, ionise_residues: bool=True)`

Build an RDKit molecule from a structure so that the field code can parameterise it.

Crystal structures rarely carry hydrogens, and the donor field is on the polar hydrogen, so
they are added with coordinates unless the structure already has them.

### `pocket_fields(protein: Protein, fieldset: FieldSet, add_hydrogens: bool=True, ionise_residues: bool=True) -> MoleculeFields`

Fields of the pocket, in the same parameterisation as the molecules of the data set.

### `field_overlap(a: MoleculeFields, spec_a, fa: int, b: MoleculeFields, spec_b, fb: int) -> float`

Integral over space of field `fa` of `a` times field `fb` of `b`.

Both fields are sums of Gaussians, so the integral is a sum over atom pairs of
`(2 pi / (b1 + b2))^(3/2) exp(-b1 b2 / (2 (b1 + b2)) |x1 - x2|^2)`, the same closed form the
CMF kernel uses; no grid is involved and the result is exact.

### `clash_count(mol: MoleculeFields, pocket: MoleculeFields, tolerance: float=0.5) -> Tuple[int, float]`

Heavy-atom pairs closer than the sum of their radii minus `tolerance`, and the worst overlap.

A complementarity score says nothing about whether the molecule fits at all; this does.

### `complementarity(mol: MoleculeFields, pocket: MoleculeFields, fieldset: FieldSet, clash_tolerance: float=0.5) -> Complementarity`

Field complementarity of one molecule with the pocket, from exact overlap integrals.

Every term is divided by the norms of the two fields, so it is a cosine in [-1, 1] and does
not grow with the size of the molecule.

### `shape_surface(mol: MoleculeFields, spacing: float=0.4, level: float=0.5, margin: float=3.0, heavy_only: bool=True)`

A smooth van der Waals surface of a molecule, as (vertices, faces).

The density is a sum of per-atom Gaussians scaled so that an isolated atom has the chosen
level exactly on its van der Waals sphere, which puts the surface where a chemist expects it
and closes smoothly between bonded atoms.

### `coulomb_potential(points: np.ndarray, coords: np.ndarray, charges: np.ndarray, dielectric: str='distance', eps: float=4.0, min_distance: float=1.0) -> np.ndarray`

Electrostatic potential of point charges, in arbitrary but consistent units.

`dielectric="distance"` uses the distance-dependent dielectric eps*r customary in
complementarity scoring, which damps the long range; `"constant"` divides by eps instead.

### `surface_complementarity(mol: MoleculeFields, pocket: MoleculeFields, fieldset: FieldSet, spacing: float=0.4, level: float=0.5, charge_field: str='electrostatic')`

Per-vertex and overall complementarity on the surface of the molecule.

Returns (vertices, faces, per-vertex score in [-1, 1], correlations per field).  The
electrostatic part compares potentials rather than the Gaussian charge densities, because the
potential is what the other side actually senses; the reported value is minus the correlation,
so that a positive number means complementary.

### `occupancy(mols: Sequence[MoleculeFields], grid, k: float=0.693) -> np.ndarray`

Where the series actually puts atoms, as a density on the grid.

The co-field is unconstrained by the data wherever this is zero, so it is the weight to use
when the co-field is compared with anything.

### `cofield_agreement(model, pocket: MoleculeFields, grid, weights: np.ndarray, charge_field: str='electrostatic', statistic: str='sign') -> Dict[str, float]`

Weighted correlation between each co-field and what the pocket offers there.

For every field type the pocket has a preferred ligand field: the opposite of its own charge,
the same hydrophobicity, acceptors where the ligand should donate, and space where the ligand
should be bulky.  The co-field says what the *model* prefers.  Their agreement over the
occupied region, in [-1, 1], is how far the model agrees with the protein.

Three statistics are available, all in [-1, 1].  `"sign"` (the default) is the
magnitude-weighted agreement of sign, which is how a co-field is actually read; `"cosine"`
also requires the two maps to have the same shape; `"pearson"` removes the means first and
so asks only whether the variations track each other.

### `free_space(points: np.ndarray, pocket: MoleculeFields, contact: float=3.4, reach: float=3.0) -> np.ndarray`

How much room the pocket leaves at each point, in [-1, 1].

Positive where a ligand atom would sit at a comfortable contact distance or further, negative
where it would clash with the protein.  This is what a steric co-field should be compared
with: a protein offers space, not steric field.

### `series_scores(mols: Sequence[MoleculeFields], pocket: MoleculeFields, fieldset: FieldSet, activity: Optional[Sequence[float]]=None, surface: bool=False, spacing: float=0.5, progress=None) -> Tuple[List[Complementarity], Dict[str, float]]`

Complementarity of every molecule of a series, and how it relates to the activity.

The correlation between the score and the activity is the fair test of whether
complementarity means anything for this series and this structure.

### `field_overlap_by_atom(a: MoleculeFields, spec_a, fa: int, b: MoleculeFields, spec_b, fb: int) -> np.ndarray`

`field_overlap`, split over the atoms of `b`; the sum is the whole integral.

The overlap of two sums of Gaussians is a double sum over atom pairs, so keeping the column
sums instead of the grand total costs no extra work and is exact.  It is what turns one
number for the whole pocket into a number per receptor residue.

### `pocket_residues(protein: Protein, pocket: MoleculeFields, max_distance: float=2.5) -> Tuple[List[str], List[str]]`

Which residue each atom of the pocket fields belongs to.

`pocket_fields` goes through RDKit and adds hydrogens, so its atom order is not the
atom order of the structure and the residue information cannot simply be carried across.
Every field atom is therefore assigned to the nearest atom of the structure, which is exact
for a heavy atom and right for an added hydrogen, since a hydrogen sits about one angstrom
from the atom it was added to and no atom of another residue can be that close.

Returns (label of every pocket atom, the distinct labels in the order the structure lists
them).  The second list is the fixed residue order a profile needs: the requirement that the
residue identities and their ordering be identical for every complex compared is met by
taking that order from the pocket, which a whole series shares.

### `interaction_profile(mol: MoleculeFields, pocket: MoleculeFields, fieldset: FieldSet, residues: Sequence[str], order: Optional[Sequence[str]]=None) -> InteractionProfile`

Decompose the field complementarity of one molecule over the residues of the pocket.

`residues` labels every atom of `pocket` and `order` fixes the rows; both come from
`pocket_residues`.  The normalisation is the one `complementarity` uses, the
product of the two whole-field norms, which is what makes the columns sum to the global
terms rather than to something merely proportional to them.

### `profile_matrix(profiles: Sequence[InteractionProfile]) -> np.ndarray`

One row per profile, flattened; every profile must share the residue order.

### `profile_similarity(a: InteractionProfile, b: InteractionProfile, binary: bool=False, threshold: float=0.02) -> float`

How far two complexes reproduce the same interaction pattern.

The continuous comparison is the cosine of the two flat profiles, which keeps the strength
of every interaction.  `binary=True` reduces each profile to the interactions above
`threshold` and takes the Tanimoto coefficient instead: cruder, but it is the form the
interaction-fingerprint literature reports and it is not affected by the scale of the
terms.

### `consensus_profile(profiles: Sequence[InteractionProfile], threshold: float=0.02) -> Tuple[np.ndarray, np.ndarray]`

(frequency of every interaction over the profiles, its mean strength).

An interaction seen in most of the reference complexes is a property of the target; one seen
in a single complex is more likely a property of that ligand.  The frequencies are therefore
the weights with which a candidate should be credited for reproducing each interaction, and
a threshold on them gives the conservative query: require the conserved contacts and let the
rest vary.  Requiring the union of every complex's contacts, which is the other obvious
choice, makes a query that only close analogues of one reference can satisfy.

## cmfqsar.complexes

Source: `cmfqsar/complexes.py`

Transition-metal complexes: reading one from coordinates, and the frame its metal gives.

Plan item EC4.  A coordination complex breaks three assumptions the rest of the program makes
about a molecule.  Its metal atom has no entry in the steric table, so the steric field used a
default radius for it; the Gasteiger charge scheme has no parameters for a metal, and one metal
atom made RDKit return "not a number" for *every* atom, so the electrostatic field of a complex
was identically zero; and a metal-ligand bond is not a shared electron pair, so RDKit refuses to
sanitise most complexes written with ordinary single bonds.  This module repairs the third point
and `cmfqsar.atomtypes` and `cmfqsar.fields` repair the first two.

What a reader has to decide, and how it is decided here:

* **Which bonds the metal makes.**  They are found by distance (RDKit's own connectivity
  perception) and written as *dative* bonds that point from the donor atom to the metal.  A
  dative bond carries no electrons of the donor's own valence, so the donor keeps the valence it
  has in the free ligand and the molecule sanitises.
* **The bond orders of the ligands.**  The metal is taken out, which leaves the ligands as
  separate fragments, and each fragment's bond orders are determined on its own with
  `rdDetermineBonds`, the same algorithm the program uses for any set of coordinates.  A
  fragment needs its charge before its bond orders can be found (a cyclopentadienyl ring is
  aromatic at -1 and a radical at 0), so the charges are taken from the data set when it states
  them, and otherwise tried in turn.
* **The charge of the metal.**  It is the total charge of the complex minus the sum of the ligand
  charges.  For a complex whose ligand charges are stated this is the oxidation state of the
  metal, and it is written into the molecule as a formal charge, which is what the repaired
  charge model then places on the metal.
* **The frame.**  The metal is the one atom every member of a series has, so the frame is free in
  the sense of plan item EC4: the metal goes to the origin, and the rotation is fixed either by
  matching the first coordination sphere onto a reference complex (`align_on_first_sphere`)
  or, when the members do not share a coordination sphere, by a rule
  (`principal_metal_frame`).

Nothing here computes a field.  The output is an ordinary RDKit molecule with one conformation,
which `cmfqsar.fields.build_fields` reads like any other.

### `class XyzFrame  (dataclass)`

One block of an XYZ file: the elements, the coordinates and the comment line.

The comment line of an XYZ file has no agreed format.  Two readings of it are offered and
both are filled in, because a caller that knows its own file can use whichever fits:
`tags` holds every `key=value` token (the extended-XYZ convention), and `parts` holds the
comment split on the vertical bar, which several published data sets use to separate fields.

Attributes:

- `symbols: List[str]`
- `coords: np.ndarray`
- `comment: str` = `''`
- `name: str` = `''`
- `tags: Dict[str, str]` = `field(default_factory=dict)`
- `parts: List[str]` = `field(default_factory=list)`

#### `XyzFrame.n_atoms` -> int  (property)

### `class ComplexReport  (dataclass)`

What the builder did, so that a data set can be checked instead of trusted.

Attributes:

- `name: str` = `''`
- `metal: str` = `''`
- `metal_charge: Optional[int]` = `None`
- `total_charge: int` = `0`
- `donors: List[int]` = `field(default_factory=list)`
- `donor_symbols: List[str]` = `field(default_factory=list)`
- `ligand_charges: List[int]` = `field(default_factory=list)`
- `ligands_matched: int` = `0`
- `n_ligands: int` = `0`
- `net_charge: Optional[int]` = `None`
- `fault: str` = `''`

#### `ComplexReport.ok` -> bool  (property)

### `parse_comment(comment: str) -> Tuple[Dict[str, str], List[str]]`

The `key=value` tokens of a comment line, and the line split on the vertical bar.

### `read_xyz(path: str, limit: Optional[int]=None) -> List[XyzFrame]`

Every block of a (possibly multi-block) XYZ file.

The file is read literally: no bonds are perceived and no element is renamed.  `limit` stops
after that many blocks, which is what a quick check of a large file needs.

### `molecule_from_coordinates(symbols: Sequence[str], coords: np.ndarray) -> Chem.Mol`

An RDKit molecule with the given elements and coordinates and no bonds at all.

### `build_complex(symbols: Sequence[str], coords: np.ndarray, total_charge: int=0, ligand_charges: Optional[Sequence[int]]=None, ligand_smiles: Optional[Sequence[str]]=None, name: str='', multiplicity: Optional[int]=None, metal: Optional[str]=None) -> Tuple[Optional[Chem.Mol], ComplexReport]`

Build one mononuclear complex from its coordinates, with dative metal-ligand bonds.

`total_charge` is the charge of the whole complex.  `ligand_charges` are the charges of the
ligands as the data set states them, in any order; they are matched to the fragments the
coordinates give, and `ligand_smiles` (same order) lets the match be checked instead of
guessed.  `multiplicity` is stored as a property and not used here, so that the quantum stage
of plan item EC1 can pick the state up later.

Returns the molecule and a report.  The molecule is `None` when the complex could not be
built, and the report then says why; a data set is expected to lose a few members this way
and the number lost is part of its description.

### `metal_index(mol: Chem.Mol) -> Optional[int]`

The index of the one transition-metal atom, or `None` when there is not exactly one.

### `donor_indices(mol: Chem.Mol) -> List[int]`

The atoms bonded to the metal: the first coordination sphere.

### `coordination(mol: Chem.Mol) -> Dict[str, float]`

Numbers that describe the metal site, for use as descriptors or as a check.

`n_donors` is the coordination number, `donor_<element>` the number of donors of that
element, `metal_charge` the formal charge the builder gave the metal, and `d_min`,
`d_mean` and `d_max` the distances from the metal to its donors in angstrom.

### `metal_origin(mol: Chem.Mol) -> Chem.Mol`

The same molecule with its metal atom at the origin.

### `align_on_first_sphere(mols: Sequence[Chem.Mol], ref_index: int=0, max_permutations: int=MAX_DONOR_PERMUTATIONS)`

Put every complex in the frame of one of them, using the metal and its donors.

The metal goes to the origin and the rotation is the one that brings the donor atoms closest
to the reference complex's donor atoms.  Which donor answers to which reference donor is not
known, and a coordination sphere is symmetric, so every assignment that pairs donors of the
same element is tried and the best is kept.  Trying one arbitrary assignment instead would
report several angstrom for two complexes that already lie on top of each other.

Returns the aligned molecules and, for each, the root-mean-square distance between its donors
and the reference donors in angstrom, or `None` when its coordination sphere holds a
different set of elements from the reference's and there is nothing to match.

### `principal_metal_frame(mol: Chem.Mol, degenerate_tolerance: float=0.05)`

A frame for a complex that shares no coordination sphere with any other.

The metal goes to the origin and the axes are the principal axes of the whole complex, with
every atom weighted by its atomic number, in order of decreasing moment.  The sign of each
axis is fixed by the third moment along it, which is the usual way to remove the remaining
ambiguity: an axis and its opposite give the same second moment, but the heavier side of the
molecule decides the sign.

Returns the molecule in that frame and the relative gap between the two closest moments.  A
small gap means the two axes are interchangeable and the frame is arbitrary in that plane, so
the number is reported rather than hidden: it says how much of the frame is a real rule.

## cmfqsar.conformal

Source: `cmfqsar/conformal.py`

Prediction intervals with a coverage guarantee, for any learner and any kernel source.

Everything else the program says about an individual prediction is an estimate without
a guarantee.  The Gaussian process
returns a posterior standard deviation, but it is the standard deviation of *that model's own
assumptions*, and when the kernel is misspecified -- which is the normal case for molecular fields
-- it is not a statement about how often the true value falls inside.  The bootstrap intervals of
`cmfqsar.validation` are intervals on a *statistic*, on q² or on an R²pred, and say nothing
about one molecule.

Conformal prediction (Vovk, Gammerman & Shafer 2005; Lei et al., JASA 113 (2018) 1094) supplies
the missing guarantee, and requires almost nothing in return.  Take the absolute errors the model
makes on molecules it did not fit, let *q* be their (1-alpha) quantile with the finite-sample
correction below, and report y_hat +/- q.  Then, provided the calibration molecules and the new
molecule are exchangeable, the truth lies in that interval at least (1-alpha) of the time --
whatever the learner, whatever the kernel, however wrong the model is.  A badly specified model
does not lose its coverage; its cost is a wider interval, which is a fair cost.

Two ways to get the errors, and they are not equally reliable:

`split`
    Hold out a calibration set, fit on the rest, take the errors there.  The guarantee is exact
    and distribution-free, and the cost is that neither the fit nor the calibration sees all the
    data -- which on a set of thirty molecules is a real cost.

`cv`
    Use the out-of-fold errors of the cross-validation the program runs anyway, then predict with
    the model refitted on everything.  Nothing is held out and nothing extra is computed, but the
    calibration errors come from models fitted on n(1-1/k) molecules while the predictions come
    from a model fitted on n, so exchangeability is only approximate and the guarantee with it.
    It is the variant used in practice, and `benchmarks/run_conformal.py` measures what its
    coverage actually is on real sets rather than asserting that it is fine.

Normalised conformal is the third option, and the one this program is best able to offer: scale
each error by a measure of how hard its molecule is, and the interval widens where the model is
extrapolating instead of staying the same width everywhere.  The measure used here is the
applicability domain the program already computes -- one minus the mean kernel cosine to the
nearest training molecules -- so a molecule outside the domain is not merely flagged, it is given
a wider interval, which is what a user wanted the flag to mean.

Regression only.  Conformal classification produces prediction *sets* rather than intervals and
needs its own vocabulary; it is not implemented here.

### `class Conformal  (dataclass)`

A calibrated interval rule: absolute errors on molecules the model did not fit.

`difficulty` and `beta` turn it into the normalised form, where the nonconformity score is
`|y - y_hat| / (beta + difficulty)` and the half width grows with the difficulty of the molecule
being predicted.  `beta` keeps the denominator away from zero and sets how much of the width
is fixed: with beta large the rule tends back to the plain one.

Attributes:

- `level: float`
- `residuals: np.ndarray`
- `kind: str` = `'cv'`
- `difficulty: Optional[np.ndarray]` = `None`
- `beta: float` = `1.0`

#### `Conformal.n_calibration` -> int  (property)

#### `Conformal.scores` -> np.ndarray  (property)

The nonconformity scores: absolute errors, normalised when a difficulty is given.

#### `Conformal.rank` -> int  (property)

Which order statistic of the scores is taken: ceil((n + 1) * level).

The +1 is what makes the guarantee finite-sample rather than asymptotic, and it is also
what makes a small calibration set refuse to answer: 19 errors are exactly enough for a
95 % interval, since ceil(20 * 0.95) = 19 is the largest of them, and 18 are not enough
at any width, because the rule would have to extrapolate beyond everything it has seen.

#### `Conformal.quantile` -> float  (property)

The half width in units of the nonconformity score; inf when there is too little data.

#### `Conformal.usable` -> bool  (property)

#### `Conformal.min_calibration() -> int`

The smallest calibration set that could support this level.

Counted rather than solved: level / (1 - level) is the answer in exact arithmetic and
9.000000000000002 in floating point, which would reject a calibration set of nine.

#### `Conformal.half_width(difficulty: Optional[Sequence[float]]=None) -> np.ndarray`

The half width of the interval, one per molecule when a difficulty is given.

#### `Conformal.interval(y_pred: Sequence[float], difficulty: Optional[Sequence[float]]=None) -> Tuple[np.ndarray, np.ndarray]`

(lower, upper) around the predictions.

#### `Conformal.describe() -> str`

### `difficulty_from_kernel(K_test_train: np.ndarray, k_test_self: Sequence[float], K_train: np.ndarray, k: int=3) -> np.ndarray`

How far each scored molecule is from the training set, in [0, 2].

One minus the mean kernel cosine to the `k` nearest training molecules: 0 for a molecule that
is a training molecule, larger the further it reaches outside.  This is the quantity the
applicability domain thresholds, reused here as the difficulty of a prediction, so that the
two cannot disagree about which molecules are the extrapolations.

### `calibrate(y_true: Sequence[float], y_pred: Sequence[float], level: float=0.9, difficulty: Optional[Sequence[float]]=None, beta: float=1.0, kind: str='cv') -> Conformal`

Calibrate on predictions the model made for molecules it did not fit.

It is the caller's responsibility that those predictions really are out of sample; calibrating
on the fit would return an interval that covers the training set and nothing else.

### `calibrate_cv(K: np.ndarray, y: Sequence[float], method: str, params: Dict[str, float], level: float=0.9, folds: int=10, seed: Optional[int]=0, fit_bias: bool=True, difficulty: Optional[Sequence[float]]=None, beta: float=1.0) -> Conformal`

Calibrate from the out-of-fold predictions of the cross-validation already being run.

### `coverage(y_true: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> Dict[str, float]`

What the intervals actually did: the share covered, and how wide they had to be.

## cmfqsar.conformers

Source: `cmfqsar/conformers.py`

Conformational ensembles: the fourth dimension of the CMF approach.

A 3D-QSAR model represents every molecule by one conformation, and the result depends on which
one was chosen.  4D-QSAR (Hopfinger et al., J. Am. Chem. Soc. 119 (1997) 10509) replaces that
choice with an ensemble and averages over it, so that the model no longer rests on a single
guessed pose.

In the CMF framework the averaging is exact and costs no new mathematics.  A field is a sum of
atom-centred Gaussians and the kernel is bilinear in the fields, so the mean field of an ensemble

    X_f(r) = sum_k w_k X_f^(k)(r)

is itself a field, and the kernel of two ensembles is the weighted mean of the kernels of their
conformation pairs,

    K(A, B) = sum_k sum_l w_k^A w_l^B K(X_A^(k), X_B^(l)),

which is the mean-map kernel of the two conformational distributions and is positive definite by
construction.  The model stays linear in that mean field, so co-fields, the decomposition rules,
the Shapley machinery and the complementarity scores all continue to hold; contributions simply
arrive per atom per conformation and are summed back onto the atoms with
`cmfqsar.fields.MoleculeFields.per_atom`.

This module builds the ensembles.  The field set decides whether they are used: `conformers` is
`"single"` by default, which is the behaviour of the papers and of every earlier version.

What to be careful about: the kernel cost grows with the square of the number of conformations kept,
and the conformations have to be put somewhere.  That last point is not a detail.  The CMF kernel
compares molecules with each other, so what matters is not where a conformation sits relative to
its own molecule but where it sits relative to the rest of the series.  Superimposing the
conformations of a molecule on that molecule's own first pose - all that
`rdMolAlign.AlignMolConformers` can do - leaves them in arbitrary orientations with respect to
every other molecule, and in a series that its authors aligned by hand that blurs the very
arrangement the model depends on.  `CONFORMER_ALIGNMENTS` lists what is offered instead, all
of it relative to one common reference molecule of the series:

`self`
    superimpose the conformations on the molecule's own first pose over all heavy atoms (what the
    first version of this module did, kept as the default so that nothing changes silently).
`core`
    superimpose them on the molecule's own first pose over the atoms of the maximum common
    substructure with the reference only, so the scaffold stays exactly where the published
    alignment put it and only the flexible part moves.
`reference`
    superimpose each conformation on the reference molecule itself through that same maximum
    common substructure, which forces every conformation of the series into one frame.
`field`
    as `reference`, followed by the rigid-body refinement of `cmfqsar.align.field_align`,
    which maximises the normalised CMF kernel with the reference - the alignment criterion of
    Baskin & Zhokhova (2013) applied to each conformation.

In every case the molecule's first conformation is left exactly where it was, so a series that
arrives aligned keeps its alignment and only the generated conformations are placed.

### `set_conformer_energies(mol: Chem.Mol, energies: Sequence[float], prop: str=ENERGY_PROP) -> None`

Store per-conformer energies in kcal/mol on the molecule, for Boltzmann weighting.

### `get_conformer_energies(mol: Chem.Mol, prop: str=ENERGY_PROP) -> Optional[np.ndarray]`

### `generate_conformers(mol: Chem.Mol, n: int=10, seed: int=0, prune_rmsd: float=0.5, optimise: bool=True, align: bool=True, max_iters: int=200, keep_input: bool=True, align_to: str='self', reference: Optional[Chem.Mol]=None, fieldset: Optional[FieldSet]=None, atom_map: Optional[Sequence[Tuple[int, int]]]=None, window_rmsd: float=0.0) -> Chem.Mol`

Return a copy of `mol` carrying up to `n` conformations, with their energies stored.

The input conformation is kept as the first one by default, so a series that has been
aligned stays where it was and the ensemble is built around that pose.  `align_to` decides
where the other conformations go - see `CONFORMER_ALIGNMENTS` and
`align_conformers_to_reference`; `"self"`, the default, superimposes them on the
molecule's own first pose, which is what the earlier versions did.  Energies come from
MMFF94s where it can be set up and from UFF otherwise, which is enough for Boltzmann weights.

`window_rmsd`, when positive, keeps only the conformations that lie within that heavy-atom
root-mean-square distance of the input pose, measured where they are rather than after a
best fit, so the ensemble covers the local flexibility around the pose the series was aligned
to instead of the whole conformational surface.  A wider pool is embedded when a window is
asked for, since most of it will be discarded.

### `conformer_displacements(mol: Chem.Mol, reference_conf: int=0) -> np.ndarray`

Heavy-atom RMSD of every conformation from `reference_conf`, measured in place.

No superposition: two conformations that have the same shape but sit in different places are
far apart here, which is the distance that matters once a series has been aligned.

### `align_conformers(mol: Chem.Mol, atom_ids: Optional[Sequence[int]]=None) -> List[float]`

Superimpose every conformation of a molecule on its first one; returns the RMSD values.

### `single_conformer(mol: Chem.Mol, index: int=0) -> Chem.Mol`

A copy of `mol` carrying only its conformation number `index`, as conformer 0.

### `align_conformers_to_reference(mol: Chem.Mol, ref: Optional[Chem.Mol]=None, mode: str='self', fieldset: Optional[FieldSet]=None, atom_map: Optional[Sequence[Tuple[int, int]]]=None, timeout: int=10, field_restarts: int=0, field_max_iter: int=150) -> str`

Place the generated conformations of `mol` in the frame of the series, in place.

`mode` is one of `CONFORMER_ALIGNMENTS` and is explained in the module docstring.  Every
mode but `self` needs a reference molecule of the series, in the pose the series is aligned
to, and `field` needs a field set as well.  Conformation 0 is never moved.  `atom_map` is
the maximum common substructure correspondence with the reference, computed here when it is
not supplied; it depends on the molecule and not on its conformations, so it is worth passing
in when the same molecule is aligned more than once.  Returns one line of log.

### `dataset_reference(mols: Sequence[Chem.Mol], index: Optional[int]=None) -> Chem.Mol`

The molecule the conformations of a series are aligned to, as a one-conformation copy.

`index` picks it; without one the largest molecule is taken, its maximum common substructure
with the others being the most likely to hold the scaffold of the series.  The copy is made
before any conformations are generated, so the reference keeps the pose the series arrived in.

### `generate_for_dataset(mols: Sequence[Chem.Mol], n: int=10, seed: int=0, prune_rmsd: float=0.5, optimise: bool=True, align_to: str='self', reference_index: Optional[int]=None, fieldset: Optional[FieldSet]=None, window_rmsd: float=0.0, progress=None, cancel=None) -> Tuple[List[Chem.Mol], List[str]]`

Conformations for a whole series; returns the molecules and a line of log per failure.

`align_to` other than `"self"` places the conformations of every molecule relative to one
common reference molecule of the series instead of relative to the molecule itself, which is
what the kernel actually compares; the reference is taken - in the pose it arrives in, before
anything is generated - from `reference_index`, or the largest molecule if that is `None`.

### `group_conformers(mols: Sequence[Chem.Mol], names: Sequence[str], by: str='name') -> Tuple[List[Chem.Mol], List[str], List[int]]`

Collapse the records of a multi-conformer file into one molecule per compound.

`by="name"` groups records that share a title, which is how most conformer generators write
their output; any other value is read as the name of a property to group by.  The records of
a group must have the same atom order, which is the case when they come from one generator.
Returns the grouped molecules, their names, and the number of conformations in each.

### `ensemble_summary(mols: Sequence[Chem.Mol], fieldset: FieldSet) -> Dict[str, float]`

How many conformations a series carries and how concentrated their weights are.

The effective number of conformations is the exponential of the entropy of the weights: it is
the number of conformations that actually matter, which with Boltzmann weights and a dominant
minimum can be far below the number present.

## cmfqsar.contract

Source: `cmfqsar/contract.py`

What has to match between fitting and predicting, checked rather than assumed.

A saved model predicts a new molecule by the overlap of its fields with the training fields, so
the prediction is only meaningful when the new molecule was prepared the way the training
molecules were.  The model file records how they were prepared (`CMFModel.provenance`,
written by `CMFModel.state`), and this module compares a set of molecules about to be
predicted with that record.  It returns findings of three levels:

* `refuse` -- the prediction would be silently wrong: the model was fitted on partial charges
  read from the file and a molecule carries none, so the program would fall back to Gasteiger
  charges and call the result the same electrostatic field.
* `warn` -- the prediction is probably wrong and the user should look: the molecule has no
  explicit hydrogens where the training set had them (or the reverse), its atoms lie outside the
  frame of the training set, or a per-atom property field is empty for it.
* `note` -- worth knowing: the model was written by another version of the program, or by a
  version that recorded no provenance at all.

The command-line tool stops on a refusal unless told `--allow-mismatch`; the application asks.

### `class Finding  (dataclass)`

Attributes:

- `level: str`
- `code: str`
- `message: str`

### `check_molecules(model: CMFModel, mf: Sequence[MoleculeFields]) -> List[Finding]`

Compare molecules about to be predicted with how the model's training set was prepared.

### `check_crystals(model: CMFModel, crystals: Sequence) -> List[Finding]`

Compare periodic structures about to be predicted with the training cells of the model.

A crystal is prepared differently from a molecule, so the questions are different ones: the
structure must carry a lattice, its elements must be ones the fields can see, the
electrostatic weights must come from where they came from in training, and -- for the
periodic overlap kernel, which compares two cells point by point -- the cells must stand on
about one lattice.  The levels mean what they mean for molecules: `refuse` is a prediction
that would be silently wrong, `warn` is one to look at, `note` is worth knowing.

### `summary(model: CMFModel) -> str`

One line on how the model's training set was prepared, from its provenance.

## cmfqsar.copattern

Source: `cmfqsar/copattern.py`

The co-pattern of an invariant kernel as a curve: weight against distance (plan item NT9).

An overlap kernel has a co-field: a map C(r) that says where more field raises the property.
The invariant kernels of this program -- the molecular autocorrelation
(`cmfqsar.invariant`), the surface autocorrelation of an adsorbate, the periodic
autocorrelation of a cell (`cmfqsar.periodic`) and the reciprocal-space channels
(`cmfqsar.reciprocal`) -- have none, because a distribution of distances cannot say
*where* anything is.  What they have instead is a weight per channel, and a channel is one pair
of field types at one distance.  `cmfqsar.periodic.channel_weights` computes those weights
and `cmfqsar.periodic.top_channels` ranks them.

A ranked list of twenty channels out of several thousand is hard to read as a shape.  The same
numbers, drawn as a curve of weight against distance for one pair of field types, are the
reading these kernels do have: a positive part at 3 angstrom and a negative part at 6 angstrom
says that the property wants those two fields close together and not far apart.  This module
builds that curve; `cmfqsar.gui.copattern_panel.CoPatternChart` draws it.

What the curve is, exactly
--------------------------
A fitted linear model on an invariant kernel predicts

    y = b + sum_c w_c phi_c(x),

with `phi` the scaled and unit-normalised channels of the structure and `w` the weights.
The channels are laid out in blocks, one block per unordered pair of field types, and inside a
block the distances run in order.  The curve of one pair is that block, plotted against the
distance grid of the kernel's own specification.

Two kernels resolve a distance further and their curve is a **sum** over what is left:

* the **surface** autocorrelation resolves the in-plane distance by the heights of both atoms,
  so each in-plane distance carries `n_heights ** 2` channels;
* the **periodic** autocorrelation of a layer does the same when height channels are on
  (`z_max > 0`).

For those, the point drawn at a distance is the sum of the weights of the height channels at
that distance.  The sum is the right summary because the prediction is a sum over channels: the
curve then says what the whole distance contributes, and the Reading table beside it still names
the individual height channels.  `CoPattern.grouped` says how many channels went into each
point, so a reader is never left guessing, and it is 1 whenever nothing was summed.

Two kernels have no distance axis at all.  The three-body channels of plan item M10 index a
channel by two radii and a separation, and the chirality channels of the molecular
autocorrelation by a radial shell of a signed volume; neither is a curve of weight against one
distance, so `copattern` refuses them and says so.

### `class CoPattern  (dataclass)`

One curve: the weights of one pair of field types against distance.

`x` and `y` are what is drawn.  `weight` and `moved` are both carried whichever is
drawn, so that a chart can switch between them without recomputing anything: `weight` is
the coefficient and `moved` is the coefficient times the spread of the channel over the
training structures.

Attributes:

- `pair: str`
- `pair_index: int`
- `x: np.ndarray`
- `weight: np.ndarray`
- `moved: np.ndarray`
- `spread: np.ndarray`
- `x_label: str`
- `y_label: str`
- `value: str`
- `grouped: int`
- `note: str`

#### `CoPattern.y` -> np.ndarray  (property)

#### `CoPattern.describe() -> str`

### `pair_labels(model, fieldset: FieldSet) -> List[str]`

The label of every pair of field types, in the order the channel blocks are laid out.

Raises `ValueError` for a kernel that has no distance channels, with a message that says
what to read instead.

### `copattern(model, fieldset: FieldSet, pair_index: int=0, value: str='weight') -> CoPattern`

The curve of one pair of field types: weight against distance.

`pair_index` indexes `pair_labels`.  `value` is one of `VALUES`.

### `copattern_rows(model, fieldset: FieldSet, pair_index: int=0, value: str='weight') -> Tuple[List[str], List[List[str]]]`

The same curve as a table: the column titles and one row per point.

The chart is the reading; this is what a reader copies into a paper or a spreadsheet.

### `has_copattern(model, fieldset: Optional[FieldSet]=None) -> bool`

Whether this model's channels can be drawn as weight against distance.

## cmfqsar.dataio

Source: `cmfqsar/dataio.py`

Reading structure-activity data sets (SDF / MOL2 / SMILES), saving models and
exporting results.

### `class Dataset  (dataclass)`

Attributes:

- `mols: List[Chem.Mol]` = `field(default_factory=list)`
- `names: List[str]` = `field(default_factory=list)`
- `activity: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`
- `subset: np.ndarray` = `field(default_factory=lambda: np.zeros(0, dtype=object))`
- `properties: List[str]` = `field(default_factory=list)`
- `atom_properties: List[str]` = `field(default_factory=list)`
- `path: str` = `''`
- `activity_prop: str` = `''`
- `atom_activity_prop: str` = `''`
- `atom_targets: Optional['AtomTargets']` = `None`

#### `Dataset.set_atom_activity(prop: str) -> 'AtomTargets'`

Read a per-atom target and keep the index of the (molecule, atom) pairs it measured.

#### `Dataset.train_indices` -> np.ndarray  (property)

#### `Dataset.test_indices` -> np.ndarray  (property)

#### `Dataset.decoy_indices` -> np.ndarray  (property)

Background molecules, presumed inactive; never used for fitting.

#### `Dataset.unlabelled_indices` -> np.ndarray  (property)

Molecules loaded as a set without activities, for a semi-supervised fit.

They are trained on -- they are rows of the kernel and terms of the model's expansion --
but there is no activity to fit or to score them against; see `--unlabelled` and
`--method lapkrr`.  Decoys, in contrast, are never trained on at all.

#### `Dataset.random_split(test_fraction: float=0.25, seed: int=0) -> None`

#### `Dataset.set_activity_from_property(prop: str) -> None`

#### `Dataset.classes_from_threshold(threshold: float, active_above: bool=True) -> np.ndarray`

+1 / -1 class labels obtained by thresholding the activity (NaN stays NaN).

Molecules of the "decoy" subset are presumed inactive whatever their activity.

#### `Dataset.classes_from_property(prop: str, positive=('1', '+1', 'active', 'yes', 'true', 'a', 'p')) -> np.ndarray`

+1 / -1 class labels read from an SD property (numeric or a label such as active/inactive).

#### `Dataset.append(other: 'Dataset', subset: str='test') -> None`

#### `Dataset.remove(indices: Sequence[int]) -> None`

### `mol_from_mol2_block(block: str) -> Optional[Chem.Mol]`

Parse one MOL2 block with RDKit; fall back to the tolerant parser on failure.

### `common_atom_properties(mols, limit: int=25) -> List[str]`

Per-atom properties that every one of the first `limit` molecules carries.

These are what a `prop:` field can be built from; only the properties present throughout
are offered, since a field that exists for half a data set is not a field.

### `read_molecules(path: str, add_hs: bool=True)`

Return (mols, names, property_names). Supports .sdf/.sd/.mol, .mol2, .xyz, .smi/.csv/.txt.

### `numeric_properties(mols: Sequence[Chem.Mol], min_fraction: float=0.5) -> List[str]`

SD properties that are numeric for at least `min_fraction` of the molecules.

### `load_dataset(path: str, activity_prop: Optional[str]=None, add_hs: bool=True, subset: str='train', activity_file: Optional[str]=None, use_subset_column: bool=True, ensemble_by: Optional[str]=None, atom_activity: Optional[str]=None) -> Dataset`

Load a structure file; activities come from an SD property or from a separate table.

`activity_file` is a text table (tab/comma/whitespace separated, header optional) with the
molecule name in the first column, the activity in the column whose header mentions the
activity (else the second column) and, optionally, a set/subset column (1/"train", 2/"test")
defining the published train/test split.

`ensemble_by` collapses a multi-conformer file into one molecule per compound, carrying all
of its conformations: "name" groups records that share a title, anything else is read as the
name of a property to group by.  The conformations are used only in the ensemble mode of the
field set; in the default single mode the first one of each compound is used, as before.

`atom_activity` names a target that belongs to the atoms rather than to the molecule (plan
item LP1): the data set then carries the index of the (molecule, atom) pairs that were
measured, beside whatever molecule-level activity it has.

### `guess_activity_property(cands: Sequence[str]) -> Optional[str]`

Pick the most plausible activity column among numeric SD properties.

### `read_activity_table(path: str)`

Rows (name, activity, subset) of a 'name activity [set]' table (header optional).

Separator is tab, comma or whitespace; quotes around names are removed; rows without a
numeric activity are skipped.  The set column (header matching set/subset/split, or the
third column of a header-less file) is mapped to "train" (1) / "test" (2).

### `apply_activity_table(ds: Dataset, rows, use_subset: bool=True) -> int`

Assign activities (and train/test membership) from `read_activity_table` rows.

Names are matched case-insensitively; if fewer than half of the molecules match by name
but the table has exactly one row per molecule, the rows are applied in order.
Returns the number of molecules that received an activity.

### `read_activities_csv(path: str, column: Optional[str]=None) -> Dict[str, float]`

name,value CSV (header optional) -> dict.

With `column`, the value is taken from the column of that title instead of the second one,
which is what a prepared table of several properties needs: one file, one column per
property, the names in the first column.  A title that is not in the header, or a file with
no header, raises rather than quietly returning the second column.

### `activity_columns(path: str) -> List[str]`

The titles of the columns of a CSV that hold numbers, for a chooser to offer.

The first column is the name, so it is never offered; a column is offered when the first
data row has a number in it.  An empty list means the file has no header, or no numeric
column, and `read_activities_csv` should be used as it is.

### `save_model(path: str, model: CMFModel, extra: Optional[dict]=None) -> None`

Write the model with its provenance: how its training set was prepared, and by what.

The file is a joblib dump of `CMFModel.state`, version 2: everything a prediction
needs, plus the record that `cmfqsar.contract` checks new molecules against.

### `load_model(path: str)`

Read a model file; refuses one written by a newer version of the program.

### `write_predictions_csv(path: str, names: Sequence[str], y: Sequence[float], yp: Sequence[float], subset: Sequence[str], y_cv: Optional[Sequence[float]]=None, var: Optional[Sequence[float]]=None, value_header: str='predicted', domain: Optional[Sequence[float]]=None, domain_threshold: Optional[float]=None, interval: Optional[Sequence[Sequence[float]]]=None, truth_header: str='experimental', extra_columns: Optional[Dict[str, Sequence[float]]]=None) -> None`

Write the predictions table.

`extra_columns` adds one column per entry at the end, in order; a two-endpoint model writes
the second endpoint and the difference of the two there (plan item SC1).

### `write_sdf(path: str, ds: Dataset, activity_prop: str='activity') -> None`

## cmfqsar.decoys

Source: `cmfqsar/decoys.py`

Preparing a background (decoy) set for field-based virtual screening.

A decoy has to be placed in the same frame as the training actives before its fields can be
compared with theirs, and that placement decides its score: a molecule posed badly scores low for
the wrong reason and overstates the model's quality.  The safest choice is therefore to give every
decoy its *best* pose, i.e. to align it so that its field similarity to the training actives is as
large as it can be.  The evaluation is then conservative: whatever enrichment survives is not an
artefact of careless placement.

The criterion maximised here is a smooth maximum (log-sum-exp) of the cosine-normalised CMF kernel
between the decoy and each training molecule,

    S(x) = (1/beta) log sum_j exp(beta * Kn(x, M_j)),   Kn in [0, 1]

which tends to the similarity to the best-matching active as beta grows.  The search is over the
six rigid-body parameters, with analytic gradients and several random starting orientations,
because a molecule of a different chemotype has no reason to lie in the basin of its best pose.

Note what this does *not* do: it does not make a decoy a good ligand, and it cannot repair the fact
that a score computed after such an alignment measures the alignment and the model together.  See
the README for the choice of background sets (DUD-E, DEKOIS, LIT-PCBA, or the other example series).

### `class DecoyAlignResult  (dataclass)`

Attributes:

- `name: str`
- `coords: np.ndarray`
- `similarity_before: float`
- `similarity_after: float`
- `best_match: str` = `''`
- `n_eval: int` = `0`
- `conformer: int` = `0`
- `n_conformers: int` = `1`

### `class DecoyAligner(reference: AtomTable, fieldset: FieldSet, beta: float=20.0)`

Place molecules in the frame of a reference set by maximising their field similarity to it.

Parameters
----------
reference : AtomTable of the aligned molecules that define the frame (the training actives).
fieldset  : the fields to compare with; cosine normalisation is applied internally, so the
    result does not depend on molecular size.
beta : sharpness of the smooth maximum.  Large values align a decoy to its single best-matching
    reference molecule, small values to the ensemble as a whole.

#### `DecoyAligner.similarities(mol: MoleculeFields, coords: np.ndarray, sp: Optional[np.ndarray]=None)`

Normalised similarity of `mol` in the given pose to every reference molecule.

#### `DecoyAligner.objective(mol, X0, c, sp, p)`

Negative smooth maximum of the similarity, and its gradient in the rigid-body parameters.

#### `DecoyAligner.align(mol: MoleculeFields, restarts: int=6, seed: int=0, max_iter: int=200, apply: bool=True) -> DecoyAlignResult`

Give `mol` the pose of highest field similarity to the reference set.

### `align_decoys(mols: Sequence[MoleculeFields], reference: AtomTable, fieldset: FieldSet, restarts: int=6, beta: float=20.0, seed: int=0, progress: Optional[Callable[[int, int], None]]=None, cancel: Optional[Callable[[], bool]]=None) -> List[DecoyAlignResult]`

Align every molecule of `mols` into the frame of `reference`, in place.

### `align_decoys_flexible(mols: Sequence['Chem.Mol'], names: Sequence[str], reference: AtomTable, fieldset: FieldSet, n_conformers: int=8, restarts: int=2, beta: float=20.0, seed: int=0, prune_rmsd: float=0.5, progress: Optional[Callable[[int, int], None]]=None, cancel: Optional[Callable[[], bool]]=None)`

Pose every decoy from several conformations and keep the one that scores highest.

The rigid aligner above can only turn and move the conformation a decoy arrives with, so a
molecule that would resemble the actives in some other conformation never has the opportunity,
and its low score may be an accident of the file it came from.  Here the conformations are
generated (the input pose is kept as the first, so this can never do worse than the rigid
alignment), each is given its best rigid pose, and the best of those is the decoy's pose.

The cost is the product: `n_conformers` times `1 + restarts` optimisations per decoy, which
is why the restarts default lower here than in `align_decoys`.

Returns (fields, results): the chosen `MoleculeFields` of every decoy, already in the
frame of the reference, and one `DecoyAlignResult` each, saying which conformation was
chosen.

## cmfqsar.descriptors

Source: `cmfqsar/descriptors.py`

Non-field descriptors: one number per molecule, beside the fields rather than instead of them.

Everything else in this program describes a molecule as a function of space.  Some things are not
functions of space and never will be -- a measured logP, a molecular weight, a melting point, a
medium or a temperature a series was recorded at, a count of rotatable bonds, an energy from a
quantum calculation.  A 3D-QSAR model that cannot see them has to pretend the activity depends on
geometry alone, which for an adsorption potency or a permeability is simply false.

A descriptor block is therefore added to the kernel as a block of its own:

    K(A, B) = sum_f h_f c_f K_f(A, B)  +  h_d c_d  z(A) . z(B)

where `z` holds the descriptors of a molecule, standardised with the **training set's** mean and
standard deviation so that a descriptor in kilojoules does not dominate one in log units, and
`h_d` is a mixing coefficient like any field's.  The block is a linear kernel, so three things
follow and all three are worth having:

* the model stays linear in its inputs, so the decomposition rules survive -- every prediction
  splits into a field part and a per-descriptor part, and `contributions` reports the latter
  as one number per descriptor;
* a molecule's descriptor vector enters exactly as a field does, so cross-validation, the nested
  loop, the mixing search and the saved model need no special case;
* the block is *scaled*, not cosine-normalised: dividing by the two norms would send a molecule
  that happens to sit at the training mean in every descriptor to a zero row and a zero diagonal,
  i.e. out of the model altogether, which for a perfectly ordinary molecule is absurd.  Instead
  the block is multiplied by a constant that makes its mean diagonal 1 over the training set, so
  it enters on the same basis as a scaled field and the relative magnitudes survive.

What the block is *not* is a way to insert the activity itself into the inputs: a descriptor
correlated with the activity by construction will dominate and the geometry will look worthless.
The program reports the descriptor contributions beside the field ones so that this is visible
rather than hidden.

### `class DescriptorScaler  (dataclass)`

Mean and spread of every descriptor over the training set, and nothing else.

The statistics belong to the training molecules and are stored with the model: standardising a
test molecule by its own set would score two test sets on two different scales, which is the
same mistake as rescaling the field kernels at prediction time.

Attributes:

- `names: Tuple[str, ...]` = `()`
- `mean: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`
- `sd: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`
- `missing: str` = `'error'`

#### `DescriptorScaler.fit(values: np.ndarray, names: Sequence[str], missing: str='error') -> 'DescriptorScaler'`  (class method)

#### `DescriptorScaler.constant` -> List[str]  (property)

Descriptors that do not vary over the training set, and so contribute nothing.

#### `DescriptorScaler.transform(values: np.ndarray) -> np.ndarray`

Standardised descriptors; a missing value becomes the training mean, i.e. zero.

#### `DescriptorScaler.describe() -> str`

### `anchor_names(field_names: Sequence[str], labels: Sequence[str], kinds: Sequence[str]=('value', 'grad')) -> List[str]`

The column names of an anchored-point descriptor block, in the order the matrix has them.

A name says the place, the field and the reading: `at:para_site:electrostatic:gx`.  The
place comes first because a reader of a model's contributions asks "what does this point
carry" before "which field".

### `field_at_points(points: np.ndarray, mol, spec, f: int) -> np.ndarray`

The value of one field of one molecule at each point.  Exact: every atom contributes.

### `field_gradient_at_points(points: np.ndarray, mol, spec, f: int) -> np.ndarray`

The gradient of one field of one molecule at each point.  Shape (n_points, 3).

### `coulomb_at_points(points: np.ndarray, coords: np.ndarray, charges: np.ndarray, eps: float=1.0, min_distance: float=0.5)`

The potential and the electric field of point charges at each point.

Returns `(phi, E)` with `phi = sum_i q_i / (eps r_i)` and `E = -grad phi`, so that phi
is in e/angstrom and E in e/angstrom^2 (multiply by 14.4 for volts and volts per angstrom).
The dielectric is a constant: the distance-dependent form of
`cmfqsar.complementarity.coulomb_potential` damps the long range usefully for a
complementarity score, but its gradient is not the physical field, so it is not offered here.

`min_distance` keeps a point that lands on an atom finite; a point closer than this to a
nucleus is held at that distance, and the benchmark places its points away from atoms.

### `anchored_row(mol, fieldset, points: np.ndarray, kinds: Sequence[str]=('value', 'grad'), source: str='field') -> np.ndarray`

The anchored-point descriptors of one molecule, in the order `anchor_names` gives.

`mol` is a `cmfqsar.fields.MoleculeFields`, so the weights are the ones the field set
built.  With `source="coulomb"` the charges of the field set's electrostatic field are used
as point charges and the field set's other fields are ignored.

### `anchored_matrix(mols, fieldset, points: np.ndarray, kinds: Sequence[str]=('value', 'grad'), source: str='field') -> np.ndarray`

The anchored-point descriptor block of a series.  Shape (molecules, columns).

### `computed_names() -> List[str]`

The computed descriptors, as they are written in a field set: `calc:mw` and so on.

### `crystal_names() -> List[str]`

The crystal descriptors, as they are written in a field set: `calc:capacity` and so on.

### `is_structure(obj) -> bool`

Is this an ASE structure rather than an RDKit molecule?

### `compute_crystal(atoms, name: str) -> float`

One crystal descriptor of one periodic structure; NaN where the cell cannot give it.

`name` is the short name, with an optional working ion after an `@`: `capacity@Na`.

### `rdkit_descriptors() -> Dict[str, object]`

Every descriptor RDKit exposes, by its own name: about 200 of them, as a lookup table.

### `compute(mol, name: str) -> float`

One computed descriptor of one molecule; NaN if RDKit cannot produce it.

The four short names are shortcuts; beyond them any of RDKit's own descriptors may be asked
for by its RDKit name -- `calc:NumHAcceptors`, `calc:FractionCSP3`, `calc:BalabanJ` --
which is why there is no list here to keep up to date.

### `available(mols: Sequence) -> List[str]`

The molecule properties that are numeric on at least one molecule, in file order.

### `read_values(mol, names: Sequence[str]) -> np.ndarray`

The descriptor vector of one molecule or one periodic structure; anything missing is NaN.

A molecule's own properties are read from the file's tags, a structure's from
`atoms.info`, which is where a loader puts the numbers that came with the structure
(plan item BT2).

### `descriptor_kernel(Z_rows: np.ndarray, Z_cols: np.ndarray, normalize: bool=False) -> np.ndarray`

The block itself: a dot product of standardised vectors, cosine-normalised only on request.

Normalisation is off by default and the model leaves it off.  For a field, dividing by the
norms removes molecular size and keeps the shape; for a descriptor vector it removes the one
thing the descriptors are there to carry, and it sends any molecule sitting at the training
mean -- an unremarkable molecule, not a degenerate one -- to a zero row and a zero diagonal.
The magnitude is handled by scaling the whole block instead, exactly as *Scale fields* does.

### `descriptor_self(Z: np.ndarray, normalize: bool=False) -> np.ndarray`

K_d(x, x) for every row, which the one-class scorers and the GP variance need.

### `contributions(Z_mol: np.ndarray, Z_train: np.ndarray, coef: np.ndarray, h: float=1.0, scale: float=1.0, normalize: bool=False) -> np.ndarray`

What each descriptor contributed to one prediction.

The block is a dot product, so the prediction's descriptor part splits exactly:

    h c z(x) . sum_j a_j z(j)  =  sum_d  [ h c z_d(x) (sum_j a_j z_dj) ]

one term per descriptor, and they add up to the block's whole contribution.  With cosine
normalisation the two norms divide in, which is done here as well so that the terms still sum
to the number the model actually used.

### `fingerprint_kernel(mols: Sequence, radius: int=2, bits: int=2048) -> np.ndarray`

The Tanimoto coefficient of Morgan fingerprints, which is a positive semi-definite kernel.

The default is 2048-bit ECFP4 (radius 2), the fingerprint the screening literature's fusion
rules were established on.  It describes a molecule by its substructures and knows nothing
of its shape or its pose, which is what makes it the second view plan item F6 measures the
field against: what the field sees that the fingerprint does not is the part of the
description that is three-dimensional.  `cmfqsar.bioisostere` has the same similarity
for substituents; this one is over whole molecules.

## cmfqsar.design

Source: `cmfqsar/design.py`

The last step of the design cycle: what to put at a position, ranked by the model (E3).

Everything else in this program scores molecules somebody has already drawn.  This module makes
the suggestion itself: take a parent compound of a series, take one of its attachment points,
replace what is there with every fragment of a library, and rank the results by what the model
expects of them.

The pose problem that stops this being easy is solved by not having one.  A new substituent has
no crystal structure and no alignment, and superimposing it on anything would be a guess; the
topomer rule (`cmfqsar.topomer`) builds it into a pose that is a deterministic function of
its own topology, so the candidate is posed by exactly the rule that posed the molecules the
model was fitted on.  The comparison is then between comparable things at one attachment point,
which is all a substituent ranking needs.  It is also the reason this works on a topomer model and
not on an aligned one: a model fitted on superimposed whole molecules has no way to place a group
that was never synthesised.

**On a rotation-invariant model there is no pose problem at all** (plan item SC7).  The field
autocorrelation is a function of the molecule's own conformation and of nothing else, so a
candidate needs no frame and no topomer rule: it is built as a whole molecule, embedded once, and
scored.  That is what makes a design loop possible on a guest table, where the molecules share no
scaffold and the superposition the aligned route would need does not exist.  `pose="molecule"`
selects it and is the default for such a model; the annotation that compares the candidate's
fields with the parent's needs a frame and is replaced there by the atom contributions of the
invariant model (`cmfqsar.invariant.atom_contributions`), which say how much of the
prediction comes from the group that was put in.

Three numbers come back for every candidate and they answer different questions:

* the **prediction**, which is what the model expects;
* the **posterior standard deviation**, which says whether the model has any basis for an
  expectation -- a fragment unlike anything in the series gets the training mean and a large
  spread, and the two together are what stop a design loop proposing its own ignorance;
* an **acquisition score** (`cmfqsar.active`) that combines them according to what the
  chemist is doing, expected improvement by default.

The ranked list is then annotated: whether the fragment is one the series already uses, whether
the whole molecule has already been made (and if so what it measured, which is the only free
check on the suggestion), and what the candidate's fields have that the parent's do not
(`cmfqsar.fielddiff`), so the answer is *put a wider acceptor here* and not only a number.

### `class Candidate  (dataclass)`

One replacement at one position, with everything that was worked out about it.

Attributes:

- `smiles: str`
- `label: int`
- `mean: float` = `float('nan')`
- `sd: float` = `float('nan')`
- `score: float` = `float('nan')`
- `delta: float` = `float('nan')`
- `known: bool` = `False`
- `made: Optional[str]` = `None`
- `measured: float` = `float('nan')`
- `fields: List` = `_field(default_factory=list)`
- `group_value: float` = `float('nan')`
- `group_total: float` = `float('nan')`
- `group_atoms: float` = `float('nan')`
- `molecule: str` = `''`

#### `Candidate.describe() -> str`

### `class DesignResult  (dataclass)`

A ranked set of replacements at one position of one parent.

Attributes:

- `parent: str`
- `label: int`
- `acquire: str`
- `candidates: List[Candidate]`
- `parent_mean: float` = `float('nan')`
- `parent_sd: float` = `float('nan')`
- `parent_measured: float` = `float('nan')`
- `posed: int` = `0`
- `failed: List[str]` = `_field(default_factory=list)`
- `flat_score: bool` = `False`

#### `DesignResult.top(n: int=10) -> List[Candidate]`

#### `DesignResult.describe() -> str`

### `group_contribution(model, fields: MoleculeFields, mol: Chem.Mol, core: Chem.Mol) -> Tuple[float, float, float]`

How much of a prediction comes from the atoms that are not the core (plan item SC7).

Returns (the group's share of the prediction in the units of the activity, the whole molecule's,
the share of the heavy atoms the group holds).  The prediction splits over the atoms exactly
for the invariant kernel, so these are arithmetic rather than an attribution, and the three
numbers together say whether the group earns its size.  The total is the prediction without
the bias, so a candidate predicted below the training mean has a negative one.

### `read_fragments(source: str='common') -> List[str]`

A fragment library: the built-in list, or one SMILES per line from a file.

A line may carry a name after the SMILES, which is ignored: the fragment is identified by
its own canonical form throughout, so that the same group written two ways is one candidate.

### `as_fragment(smiles: str, label: int) -> Optional[Chem.Mol]`

A library SMILES turned into a fragment this attachment point can accept.

The attachment is written `[*]` or `*`; it becomes the dummy atom with the isotope that
`cmfqsar.topomer.substituents` uses to name the core atom a group is attached to, so that
a fragment from a library and a fragment cut off a molecule are the same kind of object.

### `fragment_smiles(frag) -> str`

The canonical name of a fragment, with hydrogens implicit.

A group cut out of an SDF carries its hydrogens as atoms; the same group written into a
library by hand does not, and the two are written differently. Everything here -- which
candidate is which, which are the groups the series already uses, which candidate rebuilds a
molecule somebody has already made -- is decided by this name, so it has to be one name.

### `enumerate_replacements(parent: Chem.Mol, core: Chem.Mol, label: int, fragments: Sequence[str], labels: Optional[Sequence[int]]=None, spec: TopomerSpec=TopomerSpec(), cache: Optional[Dict[str, Chem.Mol]]=None) -> Tuple[List[Tuple[str, Chem.Mol, str]], List[str]]`

Build the parent with each library fragment in place of what sits at `label`.

Returns (candidates, failures), where a candidate is (fragment SMILES, posed assembly, the
key naming the whole substitution pattern).  Everything the parent carries at its other
positions is kept exactly as it is, so the only difference between two candidates is the one
group being varied -- which is what makes the ranking a ranking of substituents.

### `whole_molecule(core: Chem.Mol, frags: Sequence[Chem.Mol]) -> Optional[Chem.Mol]`

Put the core and its substituents back together into one molecule (plan item SC7).

The fragments are the ones `cmfqsar.topomer.substituents` produces, whose dummy atom
carries the index of the core atom it was attached to, so the bond to remake is named by the
fragment itself.  This is the inverse of the cut, and it is what a design loop on a model
that needs no frame builds its candidates with.

### `embed(mol: Chem.Mol, seed: int=1) -> Optional[Chem.Mol]`

One conformation of a molecule, the way the guest tables were embedded.

### `enumerate_molecules(parent: Chem.Mol, core: Chem.Mol, label: int, fragments: Sequence[str], seed: int=1) -> Tuple[List[Tuple[str, Chem.Mol, str]], List[str]]`

Build the parent with each library fragment in place of what sits at `label`.

The candidates are whole molecules, each embedded once; nothing is superimposed on anything,
which is what a model with no frame allows and a guest table needs.

### `design(model, parent: Chem.Mol, core: Chem.Mol, label: int, fragments: Optional[Sequence[str]]=None, labels: Optional[Sequence[int]]=None, acquire: str='ei', beta: float=1.0, xi: float=0.0, spec: TopomerSpec=TopomerSpec(), parent_name: str='the parent', series: Optional[Sequence[Chem.Mol]]=None, series_names: Optional[Sequence[str]]=None, series_y: Optional[Sequence[float]]=None, known: Optional[Sequence[str]]=None, with_fields: int=0, pose: str='auto', seed: int=1, progress=None) -> DesignResult`

Rank replacements at one position of `parent` under a fitted model.

`series`, `series_names` and `series_y` are the molecules the model was fitted on; they are
used only to say which suggestions have already been made and what those measured, which is
the one free check there is on a design loop.

`pose` is how a candidate is built: `"topomer"` poses the substituent by the topomer rule,
which is what a topomer model is fitted on, and `"molecule"` builds the whole molecule and
embeds it once, which a rotation-invariant model can score because it needs no frame (plan
item SC7).  `"auto"`, the default, picks the second for an invariant model and the first
otherwise.

## cmfqsar.diagnose

Source: `cmfqsar/diagnose.py`

What to ask of a data set before modelling it, and which route it should take.

Every expensive mistake this program has recorded was visible before the model was built, in a
number that took seconds to compute.  The corrosion benzenes were modelled with five fields and an
automatic frame, and atom-type counts -- no geometry at all -- predicted better than them; the
ACE set of the 2013 paper is predicted better by four two-dimensional descriptors than by the
whole 3D model; a
thrombin model in the exposure study scored +0.059 with alpha stuck at the top of its grid, which
is what a mis-specified model looks like from outside.  None of these is subtle.  They are simply
never asked, because asking means running three or four baselines by hand.

This module asks them.  It is not a model and it does not improve one: it is the half hour of
scepticism that should precede the modelling, packaged as `diagnose` and `cmf diagnose`.

Five questions, and the numbers that answer each:

1. **Is there anything to model?**  The size of the set and the spread of the activity.  Twenty
   molecules with an activity range of half a log unit cannot support a conclusion, whatever q2
   comes out.
2. **Can a frame exist at all?**  The largest Bemis-Murcko framework and how much of the set and
   of each molecule it covers, the number of chemotypes, and whether the molecules are already
   superimposed.  A series of several scaffolds has no common frame to be aligned into, and the
   program's own alignment will invent one.
3. **Does the frame carry signal?**  The aligned model against the two baselines that need no
   frame: the atom-type count kernel, which has thrown the geometry away entirely, and the
   rotation-invariant field autocorrelation, which keeps it without a pose.  If neither is
   outperformed, the alignment adds nothing and the cheaper route is also the more reliable one.
4. **Does the geometry add anything?**  The same comparison against descriptors alone -- what the
   file carries plus the few the program computes.  This is the question the ACE result forces.
5. **Was the model specified sensibly?**  Whether the chosen alpha sits on the edge of its grid,
   which means the search needed a value it was not offered.

Every comparison is paired over the molecules and reported with a bootstrap interval, because on
twenty to thirty molecules a difference below about 0.1 in q2 is not a difference.  The
recommendation at the end is a reading of those intervals and nothing more: it says which route to
take and which question the set has not answered, never that a model is good.

### `class Route  (dataclass)`

One way of comparing the molecules, and how well it predicted.

Attributes:

- `name: str`
- `q2: float` = `float('nan')`
- `rmse_cv: float` = `float('nan')`
- `alpha: float` = `float('nan')`
- `lam: float` = `float('nan')`
- `predicted: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`
- `note: str` = `''`
- `dropped: List[str]` = `field(default_factory=list)`
- `correlations: List[Tuple[str, float]]` = `field(default_factory=list)`
- `leaked: bool` = `False`
- `subset: Optional[List[int]]` = `None`

#### `Route.row() -> dict`

### `class Finding  (dataclass)`

One answer, with the numbers behind it.

Attributes:

- `code: str`
- `level: str`
- `text: str`
- `numbers: dict` = `field(default_factory=dict)`

### `class Diagnosis  (dataclass)`

Attributes:

- `n: int` = `0`
- `activity_sd: float` = `float('nan')`
- `activity_range: Tuple[float, float]` = `(float('nan'), float('nan'))`
- `routes: Dict[str, Route]` = `field(default_factory=dict)`
- `findings: List[Finding]` = `field(default_factory=list)`
- `recommended: str` = `''`
- `why: str` = `''`
- `nested: bool` = `True`

#### `Diagnosis.warnings() -> List[Finding]`

#### `Diagnosis.report() -> str`

#### `Diagnosis.to_dict() -> dict`

### `aligned_route(mols, names, y, fieldset=None, alphas=ALPHAS, folds=10, seed=0, charge_model='gasteiger', param_scheme='rdkit') -> Route`

The ordinary model: the five fields in the frame the molecules arrive in.

### `counts_route(mols, names, y, folds=10, seed=0) -> Route`

Atom-type counts: no geometry at all, and no frame to get wrong.

### `invariant_route(mols, names, y, alphas=ALPHAS, folds=10, seed=0, charge_model='gasteiger', param_scheme='rdkit') -> Route`

The field autocorrelation: the geometry kept, the frame not needed.

### `topomer_route(mols, names, y, alphas=ALPHAS, folds=10, seed=0, charge_model='gasteiger', param_scheme='rdkit') -> Route`

The substituents cut at the framework and posed by rule: geometry without a frame.

Only the molecules that can be cut at the common framework take part, so the route carries
the indices it covers and the diagnosis compares it with an aligned model fitted on those
same molecules -- comparing a subset against the whole set is the mistake this repository
already made once, and it overstates the quality of whichever route happens to drop the hard
compounds.

### `descriptor_route(mols, y, names_d: Optional[Sequence[str]]=None, folds=10, seed=0, exclude: Sequence[str]=(), leak: float=0.95) -> Route`

Descriptors alone: the question the ACE result forces on every set.

Most files store the activity as a molecule property, so the available descriptors usually
*include the quantity being predicted*; given it, this baseline returns q2 = 1.000 and the
diagnosis recommends descriptors for every set in the world.  Anything in `exclude`, and
anything correlating with the activity above `leak`, is therefore dropped and named.

`leak` is 0.95 and not something nearer 1 because the leak is rarely a copy.  The corrosion
files carry the potency as `activity` and again as `mwila_pooled_raw`, two reductions of
the same measurements, which correlate at 0.989: excluding the activity by name left the
other, and the descriptor route came back at q2 = 0.999 on four sets.  A genuine descriptor
can of course reach 0.95 on a congeneric series, so the dropped columns are named in the
note and raised as a warning rather than quietly discarded -- if one of them really is a
descriptor, name it with `names_d` and run again.

### `paired_interval(a: Route, b: Route, y, runs: int=2000, seed: int=0) -> Tuple[float, float, float]`

(difference in q2, low, high) for route `a` against route `b`, bootstrapped over molecules.

Resampling the molecules rather than comparing two point estimates is what keeps a difference
of 0.04 on twenty-five molecules from being read as a result.

### `frame_findings(mols: Sequence[Chem.Mol], names: Sequence[str]) -> List[Finding]`

Whether a common frame can exist, and whether the molecules are already in one.

### `route_kernels(route, mols, names, alphas=ALPHAS, charge_model='gasteiger', param_scheme='rdkit', fieldset=None, y=None, activity_prop='', leak=0.95)`

[(alpha, K)] over all the molecules for one route, and the molecules it covers.

The routes are scored inside `diagnose` on their own cross-validation; to *follow* the
advice one needs the kernels themselves, so that a model can be fitted on part of a set and
asked about the rest.

### `validate_advice(mols, names, y, test_fraction=0.3, seeds=(0, 1, 2), alphas=ALPHAS, charge_model='gasteiger', param_scheme='rdkit', activity_prop='', folds=5, progress=None)`

Take the advice, build that model, and see whether it did better than the aligned one.

The diagnosis is run on the training part alone, so the advice never sees the molecules it
will be judged on; every route is then fitted on that part and asked about the rest.  This
is the measurement that decides whether the recommendation is worth printing: a tool that
reports four baselines is useful whatever its advice, but advice that has never been
followed is only a rule its author liked.

### `diagnose(mols: Sequence[Chem.Mol], names: Sequence[str], y: Sequence[float], fieldset: Optional[FieldSet]=None, alphas=ALPHAS, folds: int=10, seed: int=0, charge_model: str='gasteiger', param_scheme: str='rdkit', routes: Sequence[str]=('aligned', 'counts', 'invariant', 'topomer', 'descriptors'), activity_prop: str='', nested: bool=True, progress=None) -> Diagnosis`

Ask a data set the questions that should precede modelling it.

## cmfqsar.docking

Source: `cmfqsar/docking.py`

Place conformations inside the pocket instead of in the gas phase.

Every other way of building a 4D ensemble in this program has failed on an aligned series, and for
the same reason each time: conformations generated for an isolated molecule are not the poses a
receptor selects, so averaging over them, or choosing among them, only dilutes or re-finds the pose
the series already carried.  The remaining idea is to make the ensemble bound-like by construction,
which means putting every conformation into the site and letting the site decide where it sits.

This is a small rigid-body docking built from what the program already has: the fields of the
pocket from `cmfqsar.complementarity`, the exact overlap integrals between two sets of
Gaussians, and a soft clash term.  A pose is scored by

    score = complementarity of the fields with the pocket - clash penalty

and the six rigid-body parameters of each conformation are optimised from the pose it arrives in,
with optional random restarts.  The internal geometry is never touched, so the conformations stay
the ones that were generated; only their placement changes.

What this is not is a docking program.  There is no desolvation, no entropy, no torsional search
and no scoring function fitted to binding data, and a real program would do better.  It is enough
to answer the question this project is asking: whether an ensemble placed by the receptor behaves
differently from one placed by superposition on a pose.

### `class DockResult  (dataclass)`

What happened to one conformation.

Attributes:

- `score_before: float`
- `score_after: float`
- `clash_before: float`
- `clash_after: float`
- `moved: float`
- `restarts_used: int` = `0`

### `class PocketScorer(mol: MoleculeFields, pocket: MoleculeFields, fieldset: FieldSet, clash_weight: float=CLASH_WEIGHT, clash_tolerance: float=0.4, heavy_only: bool=True, cutoff: float=12.0, burial_weight: float=BURIAL_WEIGHT, tether_radius: float=TETHER_RADIUS, tether_weight: float=TETHER_WEIGHT)`

Scores a pose against one pocket, with everything that does not move precomputed.

#### `PocketScorer.score(coords: np.ndarray) -> float`

#### `PocketScorer.clash(coords: np.ndarray) -> float`

### `restart_poses(restarts: int, rng, max_shift: float=2.0, max_rotation_deg: float=30.0) -> List[np.ndarray]`

The rigid-body parameters each restart starts from, the arriving pose first.

A restart is a perturbation of the pose that arrived: an axis drawn at random and a turn of
at most `max_rotation_deg` about it, with a small shift. At 180 degrees or more the turn is
replaced by a uniformly random orientation, because a pose whose orientation is unknown
cannot be found by perturbing the wrong one.

### `dock_coords(mol: MoleculeFields, pocket: MoleculeFields, fieldset: FieldSet, restarts: int=0, seed: int=0, max_iter: int=200, max_shift: float=2.0, max_rotation_deg: float=30.0, clash_weight: float=CLASH_WEIGHT, tether_radius: float=TETHER_RADIUS) -> Tuple[np.ndarray, DockResult]`

Optimise the rigid-body placement of one conformation inside the pocket.

The search starts from the pose the conformation arrives in and stays near it: restarts are
bounded perturbations rather than a global search, because the series is already in the site
and the question is where in the site each conformation belongs, not whether it binds.

That assumption fails where the orientation is not known at all. The pose-recovery protocol
of item C1 gives every held-out molecule a uniformly random orientation, and a restart that
turns the conformation by at most 30 degrees can never undo it. Set `max_rotation_deg` to
180 or more and every restart draws a uniformly random orientation instead, which is the
setting for a pose that has to be found rather than refined.

### `dock_conformers(mol: Chem.Mol, pocket: MoleculeFields, fieldset: FieldSet, restarts: int=0, seed: int=0, max_shift: float=2.0, max_rotation_deg: float=30.0, keep_best: int=0, clash_weight: float=CLASH_WEIGHT, tether_radius: float=TETHER_RADIUS) -> Tuple[Chem.Mol, List[DockResult]]`

Dock every conformation of a molecule into the pocket, in place in a copy.

With `keep_best` the conformations are ordered by their docked score and only that many are
kept, which is how a docking program would return its best poses.

### `dock_dataset(mols: Sequence[Chem.Mol], pocket: MoleculeFields, fieldset: FieldSet, restarts: int=0, seed: int=0, keep_best: int=0, max_shift: float=2.0, max_rotation_deg: float=30.0, tether_radius: float=TETHER_RADIUS, progress=None, cancel=None) -> Tuple[List[Chem.Mol], List[DockResult]]`

Dock the conformations of a whole series; returns the molecules and every pose result.

### `summarise(results: Sequence[DockResult]) -> dict`

## cmfqsar.electrolyte

Source: `cmfqsar/electrolyte.py`

Bulk electrolyte simulations, for plan item BT5.

`cmfqsar.shells` cuts solvation shells out of a bulk trajectory. This module produces the
trajectories it cuts them from, so that the question of item BT5 can be asked: does an ensemble
of shells sampled from a liquid model an electrolyte better than the single optimised complex of
item BT4?

`cmfqsar.md` does the same service for item D8, a ligand inside a protein. An electrolyte is
a different system and needs a different recipe, so it gets its own module rather than another
mode of that one.

What one run is, and why each choice was made.

**A box of a named formulation.** The input is a salt, the solvents with their weight fractions
and the molality, which is what a conductivity table reports. Weight fractions and molality fix
the number of molecules exactly, with no density needed: the mass of the solvent follows from the
molecule counts and the molar masses, and the salt count follows from the mass. A table that gives
volume fractions or molarity would need a density and so an assumption; this module takes the
first kind and says so, and the realised molality is reported next to the one asked for, because
rounding the ion count to a whole number changes it by a per cent or two.

**Packed with packmol, closed by the barostat.** The starting box is built at a slightly low
density, which packmol can fill without overlaps, and the pressure coupling then closes it to the
liquid's own density. The density that comes out is reported: it is the first check that the force
field describes the liquid at all, and a box ten per cent off is a result about the force field
rather than about the electrolyte.

**GAFF with graph-network charges.** The solvents and the anion are typed with GAFF-2.11, as item
D8's ligands are. Their partial charges come from the OpenFF graph network, which reproduces
AM1-BCC charges in under a second per species, so no semi-empirical calculation is needed for
each of them. The cation is the Amber ion parameter set, because GAFF has no lithium. Three
consequences are worth stating: boron is outside GAFF, so the borate salts of a conductivity
table cannot be run here and are refused by name; a fixed-charge force field has no polarisation,
which is known to overbind a small cation to a carbonate; and the shells this produces are
therefore a model of the liquid, not a measurement of it.

**The topology is built here and not read back.** packmol writes coordinates without bond records,
and a force field matched to a bondless topology silently fails to find a template or, worse,
finds the wrong one. The topology is therefore assembled from the same molecule templates the box
was packed from, in the same order, and the packed file supplies coordinates only. The atom count
is checked against it.

### `class ElectrolyteError  — bases: RuntimeError`

A formulation cannot be built, or the simulation stack is not here.

### `class Formulation  (dataclass)`

One electrolyte as a conductivity table reports it.

`solvents` maps a name in `SOLVENTS` to its weight fraction of the solvent, and
`molality` is moles of salt per kilogram of solvent.

Attributes:

- `salt: str`
- `solvents: Dict[str, float]`
- `molality: float`
- `temperature: float` = `298.15`
- `name: str` = `''`

#### `Formulation.label() -> str`

A short name that says what the formulation is, usable as a file name.

#### `Formulation.cation` -> str  (property)

#### `Formulation.anion_smiles` -> str  (property)

#### `Formulation.molar_masses() -> Dict[str, float]`

#### `Formulation.counts(n_solvent: int=250, min_ions: int=4) -> Dict[str, int]`

How many molecules of each species the box holds, and how many ion pairs.

The solvent counts follow from the weight fractions and the molar masses, the ion count
from the molality and the solvent mass. Both are whole numbers, so the realised molality
differs a little from the one asked for; `realised_molality` reports it.

#### `Formulation.realised_molality(counts: Dict[str, int]) -> float`

The molality the whole-number counts actually give.

#### `Formulation.mass(counts: Dict[str, int]) -> float`

The mass of the box in grams, times Avogadro's number.

#### `Formulation.start_density() -> float`

The solvent mixture's density by ideal mixing, g/cm3.

#### `Formulation.box_side(counts: Dict[str, int], loosen: float=1.1) -> float`

The side of the starting cube in angstrom, a little larger than the liquid's.

### `class Settings  (dataclass)`

What one run does. The defaults are the ones item BT5's measurement used.

Attributes:

- `temperature: float` = `298.15`
- `pressure: float` = `1.0`
- `timestep_fs: float` = `2.0`
- `cutoff_nm: float` = `1.0`
- `n_solvent: int` = `250`
- `min_ions: int` = `4`
- `equilibration_ps: float` = `150.0`
- `production_ps: float` = `1000.0`
- `frame_ps: float` = `5.0`
- `seed: int` = `1`
- `platform: str` = `'CUDA'`
- `charge_model: str` = `'openff-gnn-am1bcc-0.1.0-rc.3.pt'`
- `small_molecule_forcefield: str` = `'gaff-2.11'`
- `ion_forcefield: str` = `'amber14/tip3p.xml'`
- `extra: Dict[str, float]` = `field(default_factory=dict)`

#### `Settings.production_steps` -> int  (property)

#### `Settings.equilibration_steps` -> int  (property)

#### `Settings.frame_steps` -> int  (property)

### `tools_on_path() -> List[str]`

Put the running interpreter's own directory on PATH, and say what that added.

GAFF atom types come from AmberTools' `antechamber`, which the template generator calls as
a program. In a conda environment that program sits beside the interpreter, and running the
interpreter by its full path does not put it on PATH -- which is the usual reason a machine
that has every package still cannot type a molecule. The fix is small enough to make here so
that the failure cannot happen, and `require` reports what is still missing.

### `available() -> Dict[str, bool]`

Which parts of the stack can be used here.

### `require() -> None`

### `molar_mass(smiles: str) -> float`

The molar mass of a SMILES, hydrogens included, computed rather than transcribed.

### `template(smiles: str, residue: str, seed: int=0) -> Chem.Mol`

One species as an RDKit molecule with a conformation and PDB residue information.

The same object serves three purposes: packmol packs copies of it, the topology is assembled
from its bonds, and `cmfqsar.shells` uses it to give a cut-out shell its bond orders.
That is why the atom order must never be changed between them.

### `templates(form: Formulation, seed: int=0) -> Dict[str, Chem.Mol]`

A template per species of the formulation, keyed by its residue name.

The cation has no template: it is one atom and the force field types it as an ion.

### `ion_pdb(path: str, ion: str) -> None`

A one-atom PDB of the bare cation, for packmol to place.

### `species_order(form: Formulation, counts: Dict[str, int]) -> List[Tuple[str, int]]`

The species and their counts in the order the box is packed and the topology is built.

Fixing this order in one place is what lets the packed coordinates and the assembled
topology be matched atom for atom.

### `pack(form: Formulation, counts: Dict[str, int], workdir: str, *, seed: int=1, tolerance: float=2.2, loosen: float=1.1) -> Tuple[str, float]`

Write the species' PDB files, run packmol, and return (packed PDB, box side).

### `build(form: Formulation, counts: Dict[str, int], packed_pdb: str, side: float, settings: Settings)`

(topology, system, positions) for one packed box.

### `net_charge(system) -> float`

The sum of the partial charges of a built system, which has to be zero.

### `run(form: Formulation, workdir: str, settings: Optional[Settings]=None, progress=None) -> Dict[str, object]`

Pack, equilibrate and run one formulation; returns what was done and where it is.

The result holds the paths of the topology PDB and the DCD of production frames, the
realised molality, the density the barostat settled at, and the speed, so that a later
reading of the trajectory can report the cost of producing it.

### `shell_templates(form: Formulation) -> Dict[str, Chem.Mol]`

The templates `cmfqsar.shells` needs to turn a cut-out shell into a molecule.

The same objects the box was packed from, keyed by the residue name the topology carries.

## cmfqsar.errorpred

Source: `cmfqsar/errorpred.py`

Which of the things that claim to know when a prediction is bad actually knows (plan item E5).

Three quantities in this program offer to tell a chemist that a particular prediction should not be
trusted, and they come from three different places:

* the **Gaussian-process posterior variance**, which is the kernel's own statement about how much
  of this molecule lies in the span of the training set (`CMFModel.predict_with_variance`);
* the **applicability domain**, a geometric statement about how far the molecule is from its
  nearest training neighbours in field space (`cmfqsar.validation`);
* the **conformal interval**, whose normalised form widens the interval of a molecule the model
  finds difficult (`cmfqsar.conformal`).

They are in different units -- a variance, an Angstrom-like distance in field space, a width in log
units of activity -- so they cannot be compared by their numbers.  What can be compared is the
*order* they put the molecules in, because that is all a chemist uses them for: which prediction do
I act on, which do I measure first.  So every estimate is scored against the absolute error it was
supposed to anticipate, by three measures that answer three different questions.

**Does it order the molecules correctly?**  Spearman's rho and Kendall's tau between the estimate
and `|y - yhat|`.  Scale-free, and unaware of how big the errors are.

**What is gained by acting on it?**  The risk--coverage curve: trust the most confident fraction of
the molecules and measure the RMSE of that fraction.  A useful estimate makes the curve rise with
coverage; a useless one leaves it flat at the overall RMSE.  The area under it (AURC) is one
number, and it is read against two references -- the *oracle*, which orders the molecules by the
error they actually made, and *random*, which is the flat curve.  The efficiency

    (AURC_random - AURC) / (AURC_random - AURC_oracle)

is then 1 for the oracle, 0 for an estimate that is no better than ignoring it, and negative for one
that is actively misleading.  It is the number to quote: unlike a correlation, it is in the units of
the thing a chemist cares about and it says how much of the achievable separation was captured.

**What would a two-bin rule do?**  The RMSE of the half the estimate trusts, against the RMSE of
the half it doubts.  This is the crudest and the most practical: the ratio is what "only act on the
confident half" is worth.

A constant estimate -- the unnormalised conformal interval, which gives every molecule the same
width -- is detected and reported as constant rather than scored as if its arbitrary ordering meant
something.

### `class PredictorScore  (dataclass)`

How well one estimate anticipated the errors the model actually made.

Attributes:

- `name: str`
- `n: int`
- `spearman: float`
- `kendall: float`
- `aurc: float`
- `aurc_oracle: float`
- `aurc_random: float`
- `efficiency: float`
- `rmse_trusted: float`
- `rmse_doubted: float`
- `constant: bool` = `False`

#### `PredictorScore.ratio` -> float  (property)

RMSE of the doubted half over the trusted half; 1 means the split gave no information.

#### `PredictorScore.describe() -> str`

### `risk_coverage(estimate: Sequence[float], error: Sequence[float], min_n: int=MIN_COVERAGE) -> Tuple[np.ndarray, np.ndarray]`

(coverage, RMSE) of the most-confident fraction, for every fraction of at least `min_n`.

The molecules are ordered by `estimate` ascending, so the first prefix is the one the estimate
is surest about; the RMSE of each prefix is the risk a chemist would take by acting on exactly
that many predictions.  Ties keep their input order, which is the only neutral choice.

### `aurc(estimate: Sequence[float], error: Sequence[float], min_n: int=MIN_COVERAGE) -> float`

Area under the risk--coverage curve, as the mean RMSE over the coverages it is defined at.

Every coverage point counts the same, which weights the small, confident subsets as heavily as
the whole set; that is deliberate, since the point of the estimate is to be right about the
confident end.

### `score_predictor(name: str, estimate: Sequence[float], error: Sequence[float], min_n: int=MIN_COVERAGE) -> PredictorScore`

Score one estimate against the errors, by order, by risk--coverage and by halves.

### `compare(estimates: Mapping[str, Sequence[float]], y_true: Sequence[float], y_pred: Sequence[float], min_n: int=MIN_COVERAGE) -> List[PredictorScore]`

Score every estimate against the same errors, best efficiency first.

`estimates` maps a name to one number per molecule, oriented so that **larger means a larger
error is expected**: a variance, a distance from the training set, an interval width.  An
estimate that is oriented the other way (a similarity) must be negated by the caller, which is
the one convention this module insists on.

### `table(scores: Sequence[PredictorScore]) -> str`

The comparison as a text table, for the command line and the report.

### `estimates_for(model, test, y_pred: Optional[Sequence[float]]=None) -> Dict[str, np.ndarray]`

Every error estimate a fitted model can offer about `test`, by the module's convention.

Whatever the model cannot provide is left out rather than faked: the posterior variance needs
the linear kernel, the domain needs the field kernels, and the conformal width needs a
calibration (`--conformal`).  The returned names are the ones the manual's table uses.

## cmfqsar.evidence

Source: `cmfqsar/evidence.py`

Choosing alpha and lambda by the evidence instead of by cross-validation.

Every hyper-parameter in this program is chosen by cross-validation: for each candidate, ten
fits and ten predictions, and the best cross-validated score wins.  That is reliable and slow, and
it spends ten factorisations to learn what one can tell.

A kernel ridge model is the posterior mean of a Gaussian process, and a Gaussian process scores
its own hyper-parameters without holding anything out.  The marginal likelihood -- the
probability of the training activities under the model, with the weights integrated away -- is

    log p(y) = -1/2 (y - 1m)' A^-1 (y - 1m) - 1/2 log|A| - 1/2 log(1' A^-1 1) - (n-1)/2 log 2pi

with `A = K + lambda I`.  Every term is read off the Cholesky factorisation that the fit needs
anyway, so a candidate costs one factorisation rather than ten, and nothing is held out.

Two details make it usable here rather than merely correct.

**The constant is marginalised, not fitted.**  The program's KRR carries a bias, solved in a
bordered system, and the matching Gaussian process has an unknown constant mean.  Fitting that
mean and then scoring would count it twice; integrating it out under a flat prior gives the
third term above and the `m` that appears in the first, which is the generalised-least-squares
mean rather than the arithmetic one.

**The amplitude is profiled out.**  The kernels here are cosine-normalised and the activities
arrive in whatever units the assay used, so the scale of the prior matters and nobody knows it.
Scaling the kernel and the noise together by `s` leaves the ridge -- and therefore the fitted
model -- unchanged while moving the evidence, and the maximising `s` has a closed form, so it is
substituted rather than searched.  What is left is a criterion that depends on alpha and lambda
alone.  It is a density, so changing the units of the activity still shifts every value by the
same constant -- the ranking is untouched, and it is the ranking that selects.

What the evidence is not is a cross-validated score.  It answers "how probable are these
activities under this model", not "how well would this model predict a molecule it has not
seen", and the two can disagree when the kernel is misspecified -- which for molecular fields is
the normal case.  `benchmarks/run_evidence.py` measures how far apart they are in practice.

### `profile_amplitude(quadratic: float, n: int) -> float`

The kernel amplitude that maximises the evidence, in closed form.

Scaling the kernel and the noise together by `s` leaves the predictive mean exactly where it
was, so this is not a parameter of the model but of the prior's scale; it has an optimum and
no search is needed for it.

### `log_evidence(K: np.ndarray, y: Sequence[float], lam: float, fit_bias: bool=True, profile: bool=True) -> float`

The log marginal likelihood of `y` under the kernel `K` with ridge `lam`.

With `fit_bias` the constant mean is marginalised under a flat prior, which is the Gaussian
process matching the bordered system the program's KRR solves.  With `profile` the kernel
amplitude is substituted at its maximum, so the value no longer depends on the scale of the
prior -- which nobody knows -- and the ranking over alpha and lambda no longer depends on the
units of the activity.  The value itself is a density and still shifts with those units, by
the same constant for every candidate.

### `evidence_grid(kernels: Sequence[Tuple[float, np.ndarray]], y: Sequence[float], lams: Sequence[float], fit_bias: bool=True) -> Tuple[float, float, Dict[Tuple[float, float], float]]`

(best alpha, best lambda, the whole surface) by evidence over a grid of kernels.

`kernels` is [(alpha, K)]; the evidence is comparable across both axes because the amplitude
is profiled out of each.

### `noise_scale(K: np.ndarray, y: Sequence[float], lam: float, fit_bias: bool=True) -> float`

The amplitude the evidence profiles out, as a number.

The Gaussian process that matches the ridge fit has the prior covariance `s K` and the
noise `s lambda`; this is `s`, the mean squared whitened residual of the training set.
`cmfqsar.surrogate` multiplies the posterior variance by it to put a standard deviation
in the units of the activity.

## cmfqsar.exposure

Source: `cmfqsar/exposure.py`

Solvent exposure as a weight on the fields: the claim of the surface methods, without a surface.

A CMF kernel is an overlap integral over all of space, so it weights an atom buried in the middle
of a molecule exactly as it weights one on the outside -- although a binding partner can never
touch the first.  That observation is what the surface methods are built on (CoMSA and its
relatives map properties onto the molecular surface and compare them there), and it arrived in
this program from another direction: the conceptual-DFT study found density differences dominated
by the buried core, which had to be windowed out before the maps meant anything
(`cmfqsar.cdft`).

Taking a surface literally would cost the thing the program is built around.  Every molecule has
its *own* surface, so there is no shared integration domain -- worse than the grid kernel, which at
least had one -- and the usual repair, projecting both surfaces onto a learned map, cannot be
inverted back into a field in space, so the co-field, the atom contributions and the Shapley
decomposition are all lost with it.

The same physical claim is available at the cost of a scalar per atom.  Multiply each atomic
weight by how exposed that atom is,

    w'_f,l  =  w_f,l * e_l,        e_l = SASA_l / 4 pi (R_l + R_probe)^2   in [0, 1],

and a buried atom contributes little to any field while an exposed one contributes fully.  The
field is still a sum of atom-centred Gaussians, so the kernel is still the closed-form overlap and
everything that reads it keeps working; the co-field of an exposure-weighted field is the co-field
of a field that happens to vanish inside the molecule.  There is a shorter way to say this: a
window that is itself a sum of atom-centred Gaussians leaves the triple-product integral analytic,
and in practice such a window *is* a per-atom weight.

What this does and does not touch: the steric field becomes the bulk a partner can feel rather than
the bulk that exists, which is the intent; indicator and per-atom property fields are scaled the
same way; a probe field's expansion coefficients are scaled per atom too; per-molecule descriptors
are left alone, since they are not fields.  Nothing here is a default -- it is an option to be
measured against the plain fields first, which is what `benchmarks/run_exposure.py` does.

### `atom_exposure(mol: Chem.Mol, probe_radius: float=PROBE_RADIUS, floor: float=0.0, power: float=1.0, united: bool=True) -> np.ndarray`

Relative solvent accessibility of every atom, in [0, 1].

The accessible area divided by the area the same atoms would have alone, so the number is
dimensionless and comparable between a carbon and an iodine.  `floor` keeps a wholly buried
atom from vanishing (a molecule most of whose atoms are buried would otherwise have almost no
field at all), and `power` sharpens or softens the contrast: 0.5 is gentler, 2 harsher.

With `united`, a heavy atom and its hydrogens count as one group and share the group's
exposure.  This is not a detail: measured atom by atom, the carbon of a methyl group is 93%
occluded **by its own hydrogens**, so ethanol comes out two thirds buried and the weighting
would be shielding rather than burial.  A partner meeting a methyl group meets the group.

### `weight_by_exposure(fields, exposure: Sequence[float])`

A copy of one molecule's fields with every atomic weight scaled by its atom's exposure.

Returns a new `cmfqsar.fields.MoleculeFields`; the original is untouched, so the plain
and the weighted representation of a set can be held side by side and compared on the same
molecules.

### `exposed_fields(mol: Chem.Mol, fieldset, name: str='', probe_radius: float=PROBE_RADIUS, floor: float=0.0, power: float=1.0, united: bool=True, **kwargs)`

`build_fields` followed by the exposure weighting, for one molecule.

### `exposed_set(mols: Sequence[Chem.Mol], fieldset, names: Optional[Sequence[str]]=None, probe_radius: float=PROBE_RADIUS, floor: float=0.0, power: float=1.0, united: bool=True)`

The same for a whole set, in the order given.

### `buried_fraction(mol: Chem.Mol, cutoff: float=0.05, heavy_only: bool=True, probe_radius: float=PROBE_RADIUS, united: bool=True) -> float`

The fraction of atoms the weighting almost removes: how much of a molecule is inside.

Reported beside a measurement because it says whether the weighting had anything to do, and
the answer for a ligand is usually *nothing*: measured by group, a drug-sized molecule has a
median exposure near 0.4 and no buried group at all, because thirty heavy atoms are all
surface.  There is no interior to discount until a molecule is much larger than that.

## cmfqsar.fastkernel

Source: `cmfqsar/fastkernel.py`

A compiled, threaded atom-pair loop for the field kernels (plan item B5).

The NumPy path in `cmfqsar.kernels` evaluates one molecule against a table as a dense
`(atoms_of_i, atoms_of_Q)` matrix and then makes one pass over that matrix per field: an
exponential, a scaling, two multiplications and a segment sum each.  On a 40-atom molecule against
200 molecules that matrix holds 318 000 entries, the block costs 6.7 ms, and about 2 ms of that is
passes over memory rather than arithmetic -- 25 MB of traffic for 2.5 MB of data.

What a compiled loop can and cannot do about it
-----------------------------------------------
It cannot beat NumPy at the exponentials, which are half the time.  `np.exp` on an array is
vectorised; the same calls written one at a time in a compiled loop are not, unless Intel's SVML
is installed, and on the machine these numbers come from it is not.  Measured on 200 molecules:
the NumPy path takes 0.67 s, the same arithmetic in a serial compiled loop takes 0.90 s.  **A
compiled kernel is not faster than NumPy for this.**

What it can do is use the other cores.  The NumPy path is single-threaded, and a kernel is a sum
over independent molecule pairs, so the loop over the molecules of the first table is a
`prange`: every iteration writes its own row of the output and nothing is shared.  On 32 cores
the same 0.67 s becomes 0.08 s.  That is the whole value of this module, and it is why it is
written as one fused loop rather than as a translation of the NumPy code: having paid for a
thread, it should not also be making ten passes over memory.

It also gets the cutoff of plan item B2 almost for free.  Dropping a distant atom pair costs one
comparison inside the loop, with no neighbour list to build, so it is useful on exactly the sets
where the NumPy neighbour list is not worth building.

Using it
--------
Numba is an optional dependency: `pip install numba` or `pip install -e .[fast]`.  Without it
`available` is false and everything falls back to NumPy, which stays the reference -- it is
what every table in this repository was produced with, and `tests/test_fastkernel.py` holds the
two paths to agreeing.  The compiled loop is compiled with `fastmath`, so it may reorder a sum;
the measured difference from the NumPy path is 4e-15 relative, which is the same order as the
round-off between two orderings of the NumPy sum itself.

Probe fields (a basis of several Gaussians per atom) are not implemented here and fall back.

### `available() -> bool`

True when the compiled path can be used at all.

### `threads() -> int`

How many threads the compiled loop will use.

### `defect_kernels(P, Q, specs: Sequence, cell: np.ndarray, rotations: np.ndarray, translations: np.ndarray, mixing: np.ndarray, dp: np.ndarray, dq: np.ndarray, gammas: Sequence[float], cutoff=None, symmetric: bool=False) -> Optional[Tuple[np.ndarray, np.ndarray]]`

`(K_lin, K_layer)` of `_defect_kernels` for two tables of difference fields that
stand in one `cell`, or None when the compiled loop cannot take the field set.

`rotations` and `translations` are in fractional coordinates; `mixing` are the per-field
coefficients of the joint kernel; `dp` and `dq` the joint self-kernels of the two tables;
`cutoff` a radius in angstroms, which should not exceed half the cell's smallest width,
because the displacement is taken to the nearest image only.  The caller mirrors the result
when `symmetric`.

### `plan(specs: Sequence, cutoff=None, tol: float=1e-06) -> Optional[Tuple]`

The field layout the compiled loop needs, or None when it cannot take this field set.

A probe field, or a field of a kind the loop does not implement, makes the whole call fall
back to NumPy rather than being handled halfway.  `cutoff` is a radius in angstroms or None;
unlike the NumPy path there is no "auto", because inside the loop a cutoff costs one
comparison per pair and needs no neighbour list, so there is nothing to decide.

### `bounds_of(offsets: np.ndarray, first_mol: int=0) -> np.ndarray`

(n_mol, 2) first and last atom index of every molecule from `first_mol` onwards.

### `kernel_matrices(P, Q, specs: Sequence, symmetric: bool=False, cutoff=None, out: Optional[np.ndarray]=None) -> Optional[np.ndarray]`

(F, nP, nQ) per-field kernels by the compiled loop, or None when it cannot be used.

The caller mirrors the result when `symmetric`; this writes only the upper triangle then.

### `warm_up() -> bool`

Compile the loop on a two-atom problem, so the first real call is not the compilation.

Returns False when Numba is not installed.  The compiled function is cached on disk, so this
costs about a second once per machine and nothing afterwards.

## cmfqsar.fieldae

Source: `cmfqsar/fieldae.py`

A field autoencoder with an analytic decoder (plan item E11).

Kernel principal component analysis on the CMF kernel is the linear autoencoder of the field
space, and this program already has it twice: as the coordinates of the generative topographic
map (`cmfqsar.gtm`) and as the `ockpca` applicability-domain score of
`cmfqsar.oneclass`.  What kernel PCA cannot give is a **pre-image**.  A field reconstructed
from its principal components is a weighted sum of training fields.  It can be drawn on a grid,
but it is not a set of atoms, nothing can be sampled from it, and no design step can take it as
a target.  This module adds a nonlinear autoencoder that never leaves the field space and whose
output *is* a set of atoms.

The three pieces
----------------
**The encoder** maps a molecule to a code of a few dimensions.  Its input is the whitened kernel
against a few hundred landmarks -- the feature vector of `cmfqsar.lowrank`, built here by
`landmark_features` -- so the encoder sees a molecule exactly as the kernel does and needs
neither a grid nor an atom order nor a fixed atom count.

**The decoder** maps a code to a fixed number of *pseudo-atoms*, each with a position and one
weight per field type.  That is the same form as every other field in the program, so a decoded
field can be handed to the overlap kernel, to a co-field reader or to a matcher without any
conversion (`FieldAutoencoder.pre_image`).

**The loss** is the squared distance between the input field and the decoded field in the field
Hilbert space:

    ||f - f'||^2 = k(f, f) - 2 k(f, f') + k(f', f')

and all three terms are sums of overlap integrals of atom-centred Gaussians, the same closed
form as every kernel in `cmfqsar.kernels`.  The loss is therefore exact and differentiable
in the positions and the weights of the pseudo-atoms, and **rule 3 holds: no grid enters the
training**.  `overlap` is that formula written for PyTorch tensors, and
`tests/test_fieldae.py` holds it to `cmfqsar.kernels.kernel_matrices`, which stays the
reference.

What does not change: the activity model
----------------------------------------
The autoencoder is never the regressor.  A code that enters a prediction enters it as one more
kernel view, so the predictive model stays a kernel model, rule 2 holds, and the co-field, the
Shapley values and every interpretation tool are untouched.  That is why this is not the
convolutional network that the plan's "What is not planned" excludes: there a grid replaces the
kernel and a network replaces the model; here the loss *is* the closed-form kernel and the model
is the one the program already has.

What is measured, and what is not
---------------------------------
`benchmarks/run_fieldae.py` measures version (i) of the item, the plain autoencoder, against the
two gates that decide whether the nonlinearity is worth anything:

(a) the relative reconstruction error `||f - f'|| / ||f||` on held-out molecules, against
    kernel PCA with the same number of latent dimensions (`kpca_relative_errors`);
(b) the reconstruction error as an applicability-domain score, against the five scores of
    `cmfqsar.oneclass` under the protocol of item E5 (`cmfqsar.errorpred`).

Versions (ii) denoising, (iii) variational and (iv) semi-supervised are not here.
`FieldAutoencoder.sample` draws a code from the Gaussian fitted to the training codes,
which is the pre-image entry point the design items need; it is not the variational prior of
version (iii), and it is labelled as what it is.

Using it
--------
PyTorch is an optional dependency, in the way Numba, ORCA and xTB are: this module imports
without it, `available` is then false and `fit` raises with a message that says what
to install.  Install it with `pip install torch` or `pip install -e .[fieldae]`.

    >>> from cmfqsar import fieldae
    >>> geom = fieldae.geometry(fieldset, scale=fieldae.field_scales(table, fieldset))
    >>> tens = fieldae.tensors(table, geom)
    >>> feats, low = fieldae.landmark_features(table, fieldset, n_landmarks=256)
    >>> ae = fieldae.FieldAutoencoder(geom, feats.shape[1], code=8, n_pseudo=32)
    >>> hist = fieldae.fit(ae, feats, tens, train_rows, valid_rows)
    >>> err = fieldae.relative_errors(ae, feats, tens, test_rows)

### `class FieldGeometry  (dataclass)`

What the overlap of two sets of Gaussian atoms needs to know about a field set.

`kinds` is "gauss" (one width for every atom, `b = alpha`) or "steric" (a width per atom,
`b = alpha / R^2`).  `scale` is the constant in front of each field in the joint kernel,
`h_f` times whatever field scaling the caller uses; `field_scales` gives the
convention of this program, one over the mean self-kernel of the field over the whole set.

Attributes:

- `names: List[str]`
- `kinds: List[str]`
- `alphas: np.ndarray`
- `scale: np.ndarray`

#### `FieldGeometry.n_fields` -> int  (property)

#### `FieldGeometry.groups() -> Tuple[Dict[float, List[int]], List[int]]`

The fields of shared width grouped by that width, and the radius-scaled ones.

The grouping is what makes the overlap one pass over the distance block per distinct
width rather than one per field, exactly as `cmfqsar.kernels.atom_kernel_block`
does it.

### `class FieldTensors  (dataclass)`

Every molecule of a table as padded arrays, plus its own field norm.

A padded atom carries zero weight in every field, and a zero-weight atom contributes nothing
to an overlap, so a batch of molecules of different size needs **no mask**: the padding is
exact.  Its position is set to the origin and its radius to one, because a radius of zero
would divide by zero in the width of the steric field.

Attributes:

- `coords: np.ndarray`
- `weights: np.ndarray`
- `radii: np.ndarray`
- `self_k: np.ndarray`
- `counts: np.ndarray`
- `names: List[str]`

#### `FieldTensors.n_mol` -> int  (property)

#### `FieldTensors.max_atoms` -> int  (property)

#### `FieldTensors.box(margin: float=1.1) -> np.ndarray`

Half the extent of the frame along each axis, which bounds the pseudo-atoms.

#### `FieldTensors.weight_scale(quantile: float=0.99) -> np.ndarray`

A typical size of an atom weight in each field, used to condition the decoder.

### `class LandmarkEncoder(n_features: int, code: int, hidden: Sequence[int]=(256, 256))  — bases: _Base`

The whitened landmark kernel of a molecule, mapped to a code of a few dimensions.

The input columns are standardised by the training mean and spread, held as buffers so that
they travel with the fitted model; without it the columns of a whitened kernel differ by
orders of magnitude and the first layer cannot see the small ones.

#### `LandmarkEncoder.set_standardisation(features: np.ndarray) -> None`

#### `LandmarkEncoder.forward(features)`

### `class PseudoAtomDecoder(code: int, n_pseudo: int, n_fields: int, hidden: Sequence[int]=(256, 256), box: Sequence[float]=(10.0, 10.0, 5.0), weight_scale: Optional[Sequence[float]]=None)  — bases: _Base`

A code, mapped to a fixed number of pseudo-atoms with a position and a weight per field.

The positions are `box * tanh(raw)`, so a pseudo-atom cannot be pushed out of the frame
the training molecules occupy; the weights are a per-field scale times a linear output, of
either sign, because a charge field has both.

#### `PseudoAtomDecoder.forward(code)`

### `class FieldAutoencoder(geom: FieldGeometry, n_features: int, code: int=8, n_pseudo: int=32, hidden: Sequence[int]=(256, 256), radius: float=PSEUDO_RADIUS, box: Sequence[float]=(10.0, 10.0, 5.0), weight_scale: Optional[Sequence[float]]=None, seed: int=0)  — bases: _Base`

The encoder, the decoder and the field geometry that makes the loss closed form.

#### `FieldAutoencoder.n_pseudo` -> int  (property)

#### `FieldAutoencoder.forward(features)`

(positions, weights) of the pseudo-atoms of every row of `features`.

#### `FieldAutoencoder.squared_error(features, coords, weights, radii, self_k)`

`||f - f'||^2` of every molecule of the batch, in the field Hilbert space.

The three terms are each of the size of `k(f, f)` and their sum is much smaller, so the
subtraction loses precision as the reconstruction improves: in single precision the
result is reliable while the *relative* error is above about 1e-3, which is two orders
of magnitude below anything measured here.  A fit that has to go below that must be run
in double precision.

#### `FieldAutoencoder.relative_error(features, coords, weights, radii, self_k)`

`||f - f'|| / ||f||`, the quantity gate (a) of the item compares.

#### `FieldAutoencoder.codes(features, batch: int=512) -> np.ndarray`

The code of every row of `features`, with no gradient.

#### `FieldAutoencoder.pre_image(code: Sequence[float], name: str='pseudo') -> MoleculeFields`

The pseudo-atoms a code decodes to, as fields the rest of the program can read.

#### `FieldAutoencoder.set_code_distribution(codes: np.ndarray) -> None`

Remember the mean and the spread of the training codes, for `sample`.

#### `FieldAutoencoder.sample(n: int=1, seed: Optional[int]=0, spread: float=1.0) -> List[MoleculeFields]`

`n` decoded fields from codes drawn around the training codes.

This is **not** the variational prior of version (iii) of the item: the codes are drawn
from an independent Gaussian fitted to the training codes, which says nothing about
whether the region between two codes decodes to anything sensible.  It is the pre-image
entry point the design items need, and it is labelled as what it is.

### `class Training  (dataclass)`

What a fit did: the error after every epoch and the state that was kept.

Attributes:

- `epochs: int` = `0`
- `best_epoch: int` = `0`
- `train_error: float` = `float('nan')`
- `valid_error: float` = `float('nan')`
- `history: List[Dict[str, float]]` = `_field(default_factory=list)`
- `seconds: float` = `0.0`
- `min_distance: float` = `float('nan')`
- `stopped_early: bool` = `False`

### `available() -> bool`

Whether PyTorch is installed, so that the autoencoder can be built and fitted.

### `require() -> None`

Raise with an installation message when PyTorch is missing.

### `geometry(fieldset: FieldSet, scale: Optional[Sequence[float]]=None) -> FieldGeometry`

The `FieldGeometry` of an enabled field set.

Probe fields and descriptor blocks are refused rather than ignored: a probe field puts
several Gaussians on every atom and a descriptor block is not a field at all, so neither can
be reproduced by pseudo-atoms that carry one weight per field.

### `field_scales(table: AtomTable, fieldset: FieldSet) -> np.ndarray`

One over the mean self-kernel of every field over the whole table.

This is the field-scaling convention of the program: each field kernel is brought to unit
mean diagonal, so that fields of very different magnitude -- a steric field and a charge
field -- enter the joint kernel on comparable terms.

### `joint_kernel(P: AtomTable, Q: AtomTable, fieldset: FieldSet, geom: FieldGeometry, symmetric: Optional[bool]=None) -> np.ndarray`

The scaled joint kernel of two tables, `sum_f scale_f K_f`, from the NumPy kernels.

### `joint_self(table: AtomTable, fieldset: FieldSet, geom: FieldGeometry) -> np.ndarray`

`k(f, f)` of every molecule of the table, in the same scaling.

### `tensors(table: AtomTable, fieldset: FieldSet, geom: FieldGeometry) -> FieldTensors`

Pad every molecule of `table` to the largest atom count and take its field norm.

### `overlap(xa, wa, ra, xb, wb, rb, geom: FieldGeometry)`

The joint overlap integral of two sets of Gaussian atoms, in closed form.

`xa` is (..., Na, 3), `wa` is (..., Na, F), `ra` is (..., Na), and the same for the second
set, with one shared batch shape.  Returns (...,): `sum_f scale_f K_f`, the same number
`joint_kernel` gives for two tables.

The distances are formed by subtraction rather than by `torch.cdist`, whose gradient is
not defined where two atoms coincide -- which is exactly where a pseudo-atom may be pushed.

### `pseudo_table(positions: np.ndarray, weights: np.ndarray, radius: float=PSEUDO_RADIUS, name: str='pseudo') -> AtomTable`

One set of pseudo-atoms as a one-molecule `cmfqsar.fields.AtomTable`.

This is what lets the NumPy kernels be the reference for the formula above, and what lets a
decoded field be used by anything in the program that takes a table.

### `pre_image_fields(positions: np.ndarray, weights: np.ndarray, radius: float=PSEUDO_RADIUS, name: str='pseudo') -> MoleculeFields`

One set of pseudo-atoms as `cmfqsar.fields.MoleculeFields`.

It is not a molecule: the pseudo-atoms have no elements and no bonds, and their weights can
be of either sign in any field.  It is the field a molecule would have to place.

### `landmark_features(table: AtomTable, fieldset: FieldSet, n_landmarks: int=256, seed: int=0, molecules: int=24, block: int=256, progress: Optional[Callable[[str, float], None]]=None)`

(features, the feature map) of every molecule, from the landmarks of item B6.

The landmarks are the atoms of a few training molecules, thinned so that no two stand on top
of each other (`cmfqsar.lowrank.landmark_atoms`).  That is the kind of landmark a
narrow field wants, and it keeps the feature vector to a few hundred columns per field
whatever the size of the set.

### `min_pseudo_distance(positions: np.ndarray) -> float`

The smallest distance between two pseudo-atoms of a set, which says whether they collapsed.

### `fit(ae: 'FieldAutoencoder', features: np.ndarray, tens: FieldTensors, train_rows: Sequence[int], valid_rows: Optional[Sequence[int]]=None, epochs: int=200, batch: int=128, lr: float=0.001, weight_decay: float=0.0, repulsion: float=0.0, min_distance: float=0.75, patience: int=20, loss: str='relative', seed: int=0, progress: Optional[Callable[[str, float], None]]=None) -> Training`

Fit the autoencoder by Adam on the closed-form reconstruction loss.

`loss` is "relative" -- the mean of `||f - f'||^2 / ||f||^2` -- or "absolute", the mean of
`||f - f'||^2` as the item writes it.  The relative form is the default because the
molecules of a large set differ in size by an order of magnitude, so an absolute loss is
almost entirely the largest molecules, and because the relative error is what gate (a)
measures.  `repulsion` weights the penalty of `_repulsion`, which keeps pseudo-atoms
from collapsing onto one position.

The state of the best epoch on `valid_rows` is restored at the end, so the returned model is
the one the validation part chose and not the one the last epoch left.

### `relative_errors(ae: 'FieldAutoencoder', features: np.ndarray, tens: FieldTensors, rows: Optional[Sequence[int]]=None, batch: int=256) -> np.ndarray`

`||f - f'|| / ||f||` of every molecule, with no gradient.

### `decode_rows(ae: 'FieldAutoencoder', features: np.ndarray, rows: Sequence[int], batch: int=256) -> Tuple[np.ndarray, np.ndarray]`

The pseudo-atom positions and weights of several molecules, as NumPy arrays.

### `fit_pseudo_atoms(tens: FieldTensors, rows: Sequence[int], geom: FieldGeometry, n_pseudo: int=32, steps: int=400, lr: float=0.05, repulsion: float=0.0, min_distance: float=0.75, seed: int=0, progress: Optional[Callable[[str, float], None]]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`

Fit pseudo-atoms to each of several fields directly, with no code in between.

This is the **floor** of the autoencoder: the reconstruction error that `n_pseudo`
pseudo-atoms can reach for one field when nothing is shared between molecules and no
bottleneck restricts them.  It separates the two ways a decoder can fail -- the code is too
narrow, or the pseudo-atoms cannot represent the field at all -- and the plan's gate (a)
cannot be read without it.

It is also useful on its own: it is the pre-image of a *given* field, so it answers "what
set of atoms would place this field?" for a co-field or a decoded field as well as for a
molecule.

Returns (relative errors, positions, weights).  The positions start on the molecule's own
atoms, which is the best starting point available and keeps the fit from having to find the
molecule before it can fit it.  With at least as many pseudo-atoms as atoms the start is the
molecule itself, the spare pseudo-atoms carrying no weight, so the error starts at zero for
every field of shared width and the fit has only to keep it there.

### `field_width(geom: FieldGeometry, f: int, radius: float=PSEUDO_RADIUS) -> float`

The Gaussian width `b` of field `f` for a pseudo-atom of this radius.

### `decoded_volume(mf: MoleculeFields, geom: FieldGeometry, f: int, grid) -> np.ndarray`

Field `f` of a set of pseudo-atoms (or of a molecule) on a grid, for display only.

A grid never enters the loss -- that is rule 3 -- but a chemist has to see the field, and
this is the same evaluation `cmfqsar.cofields` uses for a co-field.

### `export_decoded(path: str, mf: MoleculeFields, geom: FieldGeometry, f: int, spacing: float=0.5, margin: float=3.0, title: str='CMF decoded field') -> str`

Write field `f` of a decoded set of pseudo-atoms as a Gaussian cube file.

The pseudo-atoms are written as dummy atoms of atomic number zero, because they are not
elements; a viewer shows them as points with the field around them.

### `kpca_errors(K_train: np.ndarray, K_test: np.ndarray, k_self: np.ndarray, components: Sequence[int]) -> Dict[int, np.ndarray]`

`kpca_relative_errors` for several numbers of components, from one eigendecomposition.

The eigendecomposition of the centred training kernel is the whole cost of kernel PCA -- a
minute and a half on seven thousand molecules -- and it does not depend on how many
components are then kept, so a comparison over a grid of latent dimensions must do it once.

### `kpca_relative_errors(K_train: np.ndarray, K_test: np.ndarray, k_self: np.ndarray, components: int) -> np.ndarray`

`||f - f'|| / ||f||` of kernel PCA with `components` latent dimensions.

The reconstruction is the training mean plus the projection on the first `components`
principal directions of the centred training kernel (Hoffmann 2007, the `ockpca` score of
`cmfqsar.oneclass` without its offset).  This is the number gate (a) of item E11
compares the decoder with, and the comparison is not quite even: kernel PCA gets the
training mean for free on top of its `components` directions, and its reconstruction is a
sum of training fields, which is not a set of atoms and cannot be sampled from.

`K_train` is the training kernel, `K_test` the test-against-training block, `k_self` the
test molecules' own `k(x, x)`; all in one scaling.

## cmfqsar.fielddiff

Source: `cmfqsar/fielddiff.py`

What one molecule's fields have that another's do not.

A model says how well a candidate scores; it does not say what is wrong with it.  For scaffold
hopping and peptidomimetic work the useful readout is the difference between the fields of the
molecule being imitated and those of the candidate: where the candidate fails to reproduce what
the reference presents, and where it puts something the reference never had.

Two views of the same quantity, both exact:

* **Per field type, in closed form.**  The squared distance between two fields is an overlap
  integral like any other,

      ||X_fa - X_fb||^2 = K_f(a, a) + K_f(b, b) - 2 K_f(a, b),

  so the mismatch of each field needs no grid at all, and the cosine of the two fields comes
  from the same three numbers.  This is what `mismatch` reports.
* **Where in space**, on a grid: `X_f(candidate) - X_f(reference)`, whose negative lobes are
  what the candidate is missing and positive lobes what it adds.  `difference_features`
  turns those lobes into the same typed features with tolerance radii that
  `cmfqsar.pharmacophore` reads out of a co-field, so the report says "missing acceptor at
  (x, y, z), radius 1.4 A" rather than only "cosine 0.71".

The cosine weighs the two molecules alike, which is the wrong question for half of this module's
purpose.  A fragment cut from a lead reproduces part of its fields and nothing else, so the
cosine divides what it does reproduce by the norm of the whole lead and reports a poor match for
a perfect sub-structure.  `tversky` weights the missing part and the added part separately
and so can be asked "how much of the candidate does the reference account for" instead, which is
the measure for growing a fragment or screening smaller molecules against a lead.

Both molecules must already be in the same frame: a difference map of two molecules that were
never aligned measures the alignment.

### `class FieldMismatch  (dataclass)`

How one field type of a candidate compares with the same field of a reference.

Attributes:

- `field: str`
- `label: str`
- `cosine: float`
- `distance: float`
- `norm_ref: float`
- `norm_cand: float`
- `missing: float` = `float('nan')`
- `added: float` = `float('nan')`

#### `FieldMismatch.describe() -> str`

### `class TverskyScore  (dataclass)`

Tversky's index of one field type, with the two one-sided parts it is built from.

Attributes:

- `field: str`
- `label: str`
- `index: float`
- `overlap: float`
- `missing: float`
- `added: float`

#### `TverskyScore.describe() -> str`

### `mismatch(candidate: MoleculeFields, reference: MoleculeFields, fieldset: FieldSet, grid: Optional[Grid]=None, spacing: float=0.5, margin: float=3.0) -> List[FieldMismatch]`

Per-field comparison of a candidate with the molecule it is meant to imitate.

The cosine and the distance are exact overlap integrals.  `missing` and `added` are the
integrals of the negative and positive parts of the difference map and need a grid; pass
`grid=False` to skip them when only the closed-form numbers are wanted.

### `overall_similarity(candidate: MoleculeFields, reference: MoleculeFields, fieldset: FieldSet) -> float`

Cosine of the two molecules over all fields together, with the mixing coefficients.

### `difference_features(candidate: MoleculeFields, reference: MoleculeFields, fieldset: FieldSet, grid: Optional[Grid]=None, level: float=0.5, min_radius: float=0.6, max_features: int=10, spacing: float=0.5, margin: float=3.0)`

The lobes of the difference map, as typed features: what is missing, and what is extra.

The kinds read "missing <field>" where the candidate fails to reproduce the reference and
"extra <field>" where it adds something; the weight is the integral of the difference over
the lobe, negative for what is missing.

### `rank_against(reference: MoleculeFields, candidates: Sequence[MoleculeFields], fieldset: FieldSet, alpha: Optional[float]=None, beta: Optional[float]=None) -> List[Tuple[int, float]]`

(index, similarity) of every candidate against the reference, best first.

The similarity is the overall cosine by default.  Give `alpha` and `beta` to rank by
Tversky's index instead, which is what a library of molecules smaller than the reference
needs; see `tversky` for what the two weights mean.

### `tversky(candidate: MoleculeFields, reference: MoleculeFields, fieldset: FieldSet, alpha: float=1.0, beta: float=1.0) -> List[TverskyScore]`

Asymmetric field similarity per field type: Tversky's index on the overlap integrals.

The cosine of `mismatch` treats the two molecules alike, so a fragment can never
resemble the lead it was cut from: whatever the fragment does reproduce is divided by the
norm of the whole reference.  Tversky's index weights the two one-sided parts separately,

    T = O_cr / (O_cr + alpha * (O_rr - O_cr) + beta * (O_cc - O_cr))

where `O_cr` is the overlap integral of the two fields and `O_cc`, `O_rr` the self-overlaps.
`alpha` therefore weights what the **reference** presents and the candidate fails to
reproduce, and `beta` what the **candidate** adds, which is the vocabulary
`mismatch` already reports as `missing` and `added`.  Four settings are worth knowing:

`alpha = beta = 1`
    the Tanimoto index of the two fields, and the default.
`alpha = beta = 0.5`
    the Dice index.
`alpha = 0, beta = 1`
    `O_cr / O_cc`: how much of the candidate the reference accounts for.  This is the
    setting for growing a fragment or screening a library of smaller molecules against a
    lead, because it penalises the candidate for nothing that is merely absent from it.
`alpha = 1, beta = 0`
    `O_cr / O_rr`: how much of the reference the candidate covers, which asks the opposite
    question and rewards a candidate for leaving nothing out.

The cosine of `mismatch` is exactly the geometric mean of those last two, since their
product is `O_cr^2 / (O_cc O_rr)`.  That is the precise sense in which it is a compromise
between the two questions and answers neither.

**A one-sided index can exceed one, and that is the containment signal rather than a bug.**
An overlap integral is an inner product, not the measure of an intersection, so the
set identity `|A - B| >= 0` has no counterpart here: where the reference field exceeds the
candidate's everywhere, `O_cr` exceeds `O_cc` and `O_cr / O_cc` passes one.  A fragment of
the reference, in the frame it was cut from, therefore scores one or a little above, and the
amount above measures how much reference field surrounds it.  The shape literature reports
Tversky scores the same way.  Nothing is clamped, because clamping would give a tidier range
at the cost of the Tanimoto and Dice identities above.

One further caution.  Tversky's index was defined on sets, and a field with signed
weights is not one: the electrostatic overlap of two molecules can be negative while both
self-overlaps are positive, and the index is then a formal ratio rather than a shared
fraction.  The reading above holds for the non-negative fields, which is the steric,
hydrophobic and hydrogen-bond ones and every continuous indicator field.

### `tversky_similarity(candidate: MoleculeFields, reference: MoleculeFields, fieldset: FieldSet, alpha: float=1.0, beta: float=1.0) -> float`

Tversky's index over all fields together, with the mixing coefficients applied.

The counterpart of `overall_similarity`, which is the cosine of the same three
quantities; `alpha = beta = 1` gives the Tanimoto index of the mixed fields.

## cmfqsar.fieldgraph

Source: `cmfqsar/fieldgraph.py`

Field-point graphs: an alignment that needs neither a common substructure nor a model.

Every alignment this program has is a superposition on something shared.  `mcs_align` needs a
common substructure, which a series of several scaffolds does not have; `field_align` refines a
pose that something else must first put in roughly the right place, because the CMF kernel is flat
where two molecules do not overlap and there is no gradient to follow; `varalign` and
`flexalign` need a fitted model, so they cannot align the training set that the model is fitted
on.  The measurements in this repository all say the same thing about the consequence: the
alignment decides more than the kernel, the learner or the fields, and on a heterogeneous set the
frame is where the method fails.
The construction here is the third way, and it is old (Thorner, Willett, Wright & Taylor 1997 for
field graphs; the field points of Cresset's methodology, and the maximum-common-subgraph matching
of the virtual-screening literature).  Reduce each molecule's *own* fields to a few typed
extrema, treat those as the vertices of a graph whose edges carry the distances between them, and
superimpose two molecules on the largest set of extrema whose mutual distances agree.  Nothing is
shared in advance: not a substructure, not a reference frame, not a model.  What is matched is
what the molecule presents to a binding partner, which is the thing the fields are a description
of in the first place.

How it works, in the order the code does it:

1. **Field points.**  A connected region where one field exceeds a fraction of its own largest
   value is one point, with the position, tolerance radius and integrated weight that
   `cmfqsar.pharmacophore` already computes for a co-field lobe -- the same discretisation,
   applied to a molecule's own field rather than to a model's.  The type of a point is the field
   it came from and the sign of the lobe, so a positive electrostatic extremum can only ever be
   matched to another positive electrostatic extremum.
2. **The correspondence graph.**  One vertex per pair of points of the same type, one from each
   molecule.  Two vertices are joined when they are compatible: they reuse no point, and the
   distance between the two points of one molecule agrees with the distance between their
   partners in the other to within a tolerance.  A clique is then a set of mutually consistent
   pairings, which is a candidate common arrangement.
3. **The cliques.**  Bron-Kerbosch with pivoting, budgeted, since the number of maximal cliques
   is exponential in the worst case and a screening run must not run forever.  Three points are the
   minimum that fixes a rigid body, and the superposition that follows uses a proper rotation, so
   a mirror-image arrangement is never accepted as a match.
4. **Choosing among them.**  A graph can have several maximum cliques and they are not
   interchangeable: they are alternative hypotheses about which features correspond.  Each is
   superimposed by Kabsch on the matched extrema and then scored by the quantity the program
   actually cares about, the cosine of the two molecules' fields.  The best wins.  This is the
   one step that uses the CMF kernel, and it uses it to *rank* complete hypotheses rather than to
   search, which is what keeps it out of the flat region that defeats `field_align`.

**The steric field is left out, and that is measured rather than assumed.**  At every contour
tried, from 0.2 to 0.8 of its peak, the steric field of a drug-sized molecule has exactly one
connected region: it is a single mass covering the molecule.  Its one extremum sits near the
centroid and carries no information about where anything is, so including it would add a vertex
that matches everything and constrains nothing.  The fields that do have localised structure are
the electrostatic one, with two signs, the hydrophobic one, also with two signs because the
Crippen contributions are signed, and the two hydrogen-bond fields; together they give nine to
thirteen points for a ligand of thirty to sixty atoms, which is the regime the field-point
literature works in.

What this does not do: it does not change a conformation, so a molecule can only be placed as
well as the conformation it arrives in allows, and it does not know which of its own points
matter, since a molecule's field says what it presents and not what the target needs.  A fitted
model knows the second thing, which is why the confidence alignment exists and why the two are
worth running in series.

### `class FieldPoint  (dataclass)`

One extremum of one field of one molecule: where it is, how big, and how much of it.

Attributes:

- `field: str`
- `sign: int`
- `centre: np.ndarray`
- `radius: float`
- `weight: float`

#### `FieldPoint.type_key` -> Tuple[str, int]  (property)

What a point may be matched to: its field and the sign of the lobe.

#### `FieldPoint.label` -> str  (property)

#### `FieldPoint.describe() -> str`

### `class FieldGraphMatch  (dataclass)`

One hypothesis about how two molecules correspond, and how good it turned out to be.

Attributes:

- `pairs: List[Tuple[int, int]]` = `_field(default_factory=list)`
- `size: int` = `0`
- `point_rmsd: float` = `float('nan')`
- `similarity: float` = `float('nan')`
- `weight: float` = `0.0`
- `R: Optional[np.ndarray]` = `None`
- `t: Optional[np.ndarray]` = `None`
- `n_points: Tuple[int, int]` = `(0, 0)`
- `n_cliques: int` = `0`

#### `FieldGraphMatch.describe() -> str`

### `field_points(mol: MoleculeFields, fieldset: FieldSet, fields: Optional[Sequence[str]]=None, level: float=0.3, min_radius: float=0.5, spacing: float=0.5, margin: float=2.0, max_points: int=16, grid: Optional[Grid]=None) -> List[FieldPoint]`

The typed extrema of a molecule's own fields, strongest first.

The grid is built around this molecule alone, because a field point is a property of the
molecule and not of any frame it is being compared in; two molecules' point sets are
therefore computed independently and only their *distances* are ever compared.

### `correspondence_graph(a: Sequence[FieldPoint], b: Sequence[FieldPoint], tolerance: float=1.0) -> Tuple[List[Tuple[int, int]], List[set]]`

Vertices (pairings of points of the same type) and their compatibility adjacency.

Two pairings are compatible when they share no point and the distance between the two points
taken from `a` agrees, to within `tolerance`, with the distance between their partners in
`b`.  A clique of this graph is a correspondence whose internal geometry is consistent, which
is a necessary condition for a rigid superposition and not a sufficient one -- the
superposition itself still has to be computed and checked, since distance agreement alone
admits arrangements that no rotation can bring together.

### `maximal_cliques(adj: Sequence[set], min_size: int=3, max_cliques: int=5000, budget: int=200000) -> List[List[int]]`

Maximal cliques of at least `min_size`, by Bron-Kerbosch with pivoting.

The count of maximal cliques is exponential in the worst case, so the search carries a
budget and returns what it found when either limit is reached.  That makes the result a
heuristic when the budget is exhausted and exact when it is not; a caller that cares can
raise both, and a screening run that cares more about finishing can lower them.

### `superpose_pairs(a: Sequence[FieldPoint], b: Sequence[FieldPoint], pairs: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, float]`

The rigid transform taking the matched points of `a` onto those of `b`, and their RMSD.

Kabsch, with the proper-rotation correction, so a correspondence that would need a reflection
is fitted as well as a rotation can fit it and reports the residual rather than being
silently mirrored into place.

### `align(mol: MoleculeFields, reference: MoleculeFields, fieldset: FieldSet, tolerance: float=1.0, min_size: int=3, level: float=0.3, min_radius: float=0.5, spacing: float=0.5, margin: float=2.0, max_points: int=16, max_candidates: int=40, max_cliques: int=5000, fields: Optional[Sequence[str]]=None, ref_points: Optional[Sequence[FieldPoint]]=None) -> Optional[FieldGraphMatch]`

Place `mol` on `reference` by matching their field points; returns the best hypothesis.

`ref_points` lets a caller aligning a whole series compute the reference's points once.
Returns None when the two molecules share fewer than `min_size` compatible points or no
consistent correspondence exists, which is a real answer: two molecules presenting nothing in
common cannot be superimposed by what they present, and a method that returned a frame anyway
would be inventing one.

### `align_dataset(mols: Sequence[MoleculeFields], reference: MoleculeFields, fieldset: FieldSet, progress=None, **kwargs) -> List[Optional[FieldGraphMatch]]`

Align a whole series on one reference, computing the reference's points once.

### `summarise(matches: Sequence[Optional[FieldGraphMatch]]) -> Dict[str, float]`

How the alignment performed over a series: how many were placed, and how well.

## cmfqsar.fields

Source: `cmfqsar/fields.py`

Parameterisation of continuous molecular fields.

Every molecular field of type *f* is approximated by a sum of atom-centred
isotropic Gaussian functions (Baskin & Zhokhova 2013, Eq. 4 and Eq. 10):

    rho_f,il(r) = w_f,il * exp(-1/2 * b_f,il * |r - r_il|^2)

with `b = alpha_f` for ordinary fields and `b = alpha_f / R_vdw^2` for the
steric field (whose width scales with the van der Waals radius of the atom).

Field types implemented
-----------------------


    electrostatic   w = partial atomic charge (Gasteiger, MMFF94 or from file)
    steric          w = Tripos vdW well depth E_vdw, width scaled by R_vdw
    hydrophobic     w = Wildman-Crippen atomic logP contribution
    hba             w = approximate hydrogen-bond acceptor increment (Abraham-B-like)
    hbd             w = approximate hydrogen-bond donor increment (Abraham-A-like),
                    placed on the polar hydrogen atoms

With `FieldSet.param_scheme == "tripos"` the hydrophobic, hba and hbd weights
are instead the in-house increments of the original CMF R scripts (keyed by
Sybyl atom type and number of attached hydrogens, donor increments on the heavy
atom; see `cmfqsar.increments`), and the steric parameters are looked up
by Sybyl atom type.  `"tripos-legacy"` reproduces the 2013 scripts exactly,
including their hydrogen-counting bug (attached hydrogens are only counted when
the hydrogen is the first atom of the bond record, which practically never
happens in Sybyl MOL2 files).

    cif:<type>      continuous indicator field: w = 1 for atoms of Sybyl type <type>
    pharm:<feature> the same for one of six pharmacophore features (donor, acceptor, aromatic,
                    hydrophobe, cation, anion); see cmfqsar/pharmtypes.py
                    (Sitnikov et al. 2015, Eq. 6)
    ring:<class>    one Gaussian per benzene-ring centroid, w = 1 for rings of the annulation
                    class <class> (terminal, linear, angular, branched) or w = the Clar-sextet
                    share of the ring (sextet); the centroids are pseudo-atoms appended by
                    cmfqsar/pbh.py

### `class FieldSpec  (dataclass)`

Description of one molecular field type.

Attributes:

- `name: str`
- `kind: str` = `'gauss'`
- `alpha: float` = `1.0`
- `h: float` = `1.0`
- `enabled: bool` = `True`

#### `FieldSpec.label` -> str  (property)

#### `FieldSpec.is_cif` -> bool  (property)

#### `FieldSpec.cif_type` -> Optional[str]  (property)

#### `FieldSpec.is_pharm` -> bool  (property)

An indicator field of one of the six pharmacophore features (plan item D2).

#### `FieldSpec.pharm_feature` -> Optional[str]  (property)

#### `FieldSpec.is_prop` -> bool  (property)

A field whose atomic weights are read from a property of the molecule.

#### `FieldSpec.prop` -> Optional[str]  (property)

#### `FieldSpec.is_probe` -> bool  (property)

A GRID-like probe field: every atom carries several Gaussians, not one.

#### `FieldSpec.probe` -> Optional[str]  (property)

#### `FieldSpec.is_ring` -> bool  (property)

A field carried by one Gaussian per ring centroid rather than by the atoms
(`ring:<class>`; see `cmfqsar.pbh`).

#### `FieldSpec.ring_class` -> Optional[str]  (property)

### `class FieldSet  (dataclass)`

The complete set of field types used by a model plus parameterisation options.

Attributes:

- `specs: List[FieldSpec]` = `field(default_factory=list)`
- `charge_model: str` = `'gasteiger'`
- `cif_scheme: str` = `'sybyl'`
- `param_scheme: str` = `'rdkit'`
- `increments: Optional[Dict[str, Dict[str, float]]]` = `None`
- `prop_missing: str` = `'error'`
- `descriptors: Tuple[str, ...]` = `()`
- `descriptor_h: float` = `1.0`
- `probe_terms: int` = `4`
- `conformers: str` = `'single'`
- `conformer_weighting: str` = `'uniform'`
- `conformer_kt: float` = `0.6`
- `max_conformers: int` = `0`
- `energy_prop: str` = `'conformer_energies'`
- `weight_prop: str` = `'conformer_weights'`
- `protonation_ph: Optional[float]` = `None`
- `protonation_states: int` = `8`
- `protonation_min: float` = `0.01`
- `tautomers: str` = `''`
- `tautomer_forms: int` = `8`

#### `FieldSet.is_ensemble` -> bool  (property)

#### `FieldSet.is_protonation_ensemble` -> bool  (property)

#### `FieldSet.is_tautomer_ensemble` -> bool  (property)

#### `FieldSet.create(physchem: Sequence[str]=PHYSCHEM_FIELDS, cif_types: Sequence[str]=(), alpha: float=1.0, charge_model: str='gasteiger', cif_scheme: str='sybyl', param_scheme: str='rdkit', conformers: str='single', conformer_weighting: str='uniform', conformer_kt: float=0.6, max_conformers: int=0, atom_properties: Sequence[str]=(), prop_missing: str='error', descriptors: Sequence[str]=(), descriptor_h: float=1.0, probes: Sequence[str]=(), probe_terms: int=4, pharm_features: Sequence[str]=(), ring_classes: Sequence[str]=(), protonation_ph: Optional[float]=None, protonation_states: int=8, protonation_min: float=0.01, tautomers: str='', tautomer_forms: int=8) -> 'FieldSet'`  (class method)

#### `FieldSet.names` -> List[str]  (property)

#### `FieldSet.n_fields` -> int  (property)

#### `FieldSet.set_alpha(alpha: float, names: Optional[Sequence[str]]=None) -> None`

#### `FieldSet.set_mixing(h: Dict[str, float]) -> None`

#### `FieldSet.mixing_vector() -> np.ndarray`

#### `FieldSet.copy() -> 'FieldSet'`

The whole set, with the specs themselves copied so they can be retuned separately.

Every field of the dataclass is carried over by asking the dataclass what its fields are,
not by listing them here: a copy written by hand quietly drops whatever is added to the
class afterwards, and a setting lost on the way to the model -- a descriptor, a probe
width -- shows up as nothing at all, only as a model that is not the one that was asked for.

### `class MissingAtomProperty  — bases: ValueError`

A molecule does not carry a per-atom property that a field asks for.

### `class MoleculeFields  (dataclass)`

Atom-level data of one molecule needed to evaluate its continuous fields.

Attributes:

- `name: str`
- `coords: np.ndarray`
- `weights: np.ndarray`
- `radii: np.ndarray`
- `symbols: List[str]`
- `atom_types: List[str]`
- `bonds: np.ndarray`
- `charges: np.ndarray`
- `molblock: str` = `''`
- `n_conformers: int` = `1`
- `conformer_weights: Optional[np.ndarray]` = `None`
- `atoms_per_conformer: int` = `0`
- `member_sizes: Optional[np.ndarray]` = `None`
- `member_atom_map: Optional[np.ndarray]` = `None`
- `member_kind: str` = `''`
- `descriptors: Optional[np.ndarray]` = `None`
- `probe_coeffs: Dict[str, np.ndarray]` = `field(default_factory=dict)`
- `charge_source: str` = `''`

#### `MoleculeFields.n_atoms` -> int  (property)

#### `MoleculeFields.is_ensemble` -> bool  (property)

#### `MoleculeFields.sizes` -> np.ndarray  (property)

Atoms in every member of the ensemble, whether or not they are all the same.

#### `MoleculeFields.same_atoms` -> bool  (property)

True when every member has the same atoms, which is the conformational case.

#### `MoleculeFields.member_slice(i: int) -> slice`

Where member `i` sits in the concatenated atom list.

#### `MoleculeFields.per_atom(values: np.ndarray) -> np.ndarray`

Sum a per-atom quantity of an ensemble field back onto the atoms of the molecule.

Atom contributions, Shapley values and the like come out per atom *per member*; a chemist
wants them per atom of the molecule, which is the sum over the members.

With members that have their own atoms -- protonation states, tautomers -- "the atoms of
the molecule" are the atoms of the first member, and `member_atom_map`, which the builder
of the ensemble supplies, says which atom of it every other member's atom corresponds to.
A proton that exists in one state only has no counterpart there; the builders in
`cmfqsar.protonation` and `cmfqsar.tautomers` point it at the heavy atom it sits
on, which is the atom a chemist would read it on.  An atom the map leaves at -1 has
nowhere to go and its share is dropped, which `unmapped_share` reports, so that a
caller can say how much was lost instead of losing it quietly.

#### `MoleculeFields.unmapped_share(values: np.ndarray) -> float`

The share of a per-atom quantity that `per_atom` could not place, in [0, 1].

### `class AtomTable  (dataclass)`

Attributes:

- `coords: np.ndarray`
- `weights: np.ndarray`
- `radii: np.ndarray`
- `offsets: np.ndarray`
- `names: List[str]`
- `molecules: List[MoleculeFields]`

#### `AtomTable.n_mol` -> int  (property)

#### `AtomTable.slice(i: int) -> slice`

#### `AtomTable.probe_coeffs` -> Dict[str, np.ndarray]  (property)

(A, n_terms) expansion coefficients of every probe field, over all atoms.

#### `AtomTable.descriptors` -> Optional[np.ndarray]  (property)

(n_mol, n_desc) of raw descriptor values, or None when the field set has none.

#### `AtomTable.subset(idx: Sequence[int]) -> 'AtomTable'`

#### `AtomTable.from_molecules(mols: Sequence[MoleculeFields]) -> 'AtomTable'`  (class method)

### `atom_property_names(mol: Chem.Mol, min_atoms: int=2) -> List[str]`

Names of the per-atom properties a molecule carries, in either convention.

An atom property is one that every atom has (RDKit keeps these and writes them to SDF as
`atom.dprop.<name>`), or a molecule property holding one number per atom, which is how
most quantum-chemistry packages dump a Mulliken charge or a Fukui index.

### `atom_property(mol: Chem.Mol, name: str, n_atoms: Optional[int]=None, policy: str='error', mol_name: str='') -> np.ndarray`

One number per atom, read from the molecule.

Two conventions are understood, in this order: a property set on every atom (RDKit's own,
which survives a round trip through SDF as `atom.dprop.<name>`), and a molecule property
holding the numbers in atom order, separated by whitespace, commas or semicolons.

The hydrogens are the usual source of error.  A structure file without 3D coordinates is embedded on loading
and hydrogens are added, so a list written for the heavy atoms is shorter than the molecule
is now.  A list as long as the molecule is taken atom for atom; a list as long as the heavy
atoms is spread over them in order, leaving the hydrogens at zero; anything else is an error
naming the molecule and both counts, because the alternative is to guess which atom a number
belongs to.

### `cif_types_in(mols: Sequence[Chem.Mol], scheme: str='sybyl', min_molecules: int=5) -> List[str]`

Atom types occurring in at least `min_molecules` molecules (Sitnikov et al. 2015).

### `atom_type_counts(mols: Sequence[Chem.Mol], scheme: str='sybyl') -> Dict[str, int]`

### `conformer_weights(mol: Chem.Mol, fieldset: FieldSet) -> np.ndarray`

Weights of the conformations of one molecule, summing to one.

`uniform` gives every conformation the same weight.  `boltzmann` and `property` read
relative energies in kcal/mol from the molecule property named by `energy_prop`, which the
conformer generator writes, and turn them into exp(-dE / kT) normalised.  `given` reads the
weights themselves from the property named by `weight_prop`, which is how learned weights
travel with a molecule: the multi-instance fit writes them there and everything downstream,
the display and the co-fields included, then uses exactly the weighting the model was fitted
with.  If anything is missing or unusable the weights fall back to uniform, because a silently
wrong weighting would be worse than an even one.

### `combine_members(members: Sequence[MoleculeFields], weights: Sequence[float], kind: str='', atom_maps: Optional[Sequence[Sequence[int]]]=None, name: str='') -> MoleculeFields`

One field that is the weighted mean of several molecules' fields (plan item D4).

The mean-map kernel needs only bilinearity: a CMF field is a sum of atom-centred Gaussians
and the kernel is bilinear in the fields, so the mean field of an ensemble is itself a field
whose atoms are the atoms of every member with their weights multiplied by that member's
weight.  Nothing in that argument requires the members to be the same molecule, which is what
this function and `ensemble_fields` now exploit: the members may be conformations, or
protonation states, or tautomers.

`weights` need not be normalised; they are.  `atom_maps` gives, for each member, the atom of
the first member that each of its atoms corresponds to, with -1 where there is none; without
it the members are assumed to share an atom list, which is checked.

### `ensemble_fields(mol: Chem.Mol, fieldset: FieldSet, name: str='') -> MoleculeFields`

Fields of a conformational ensemble: the weighted mean of the fields of its conformations.

The fourth dimension of 4D-QSAR.  Because a CMF field is a sum of atom-centred Gaussians and
the kernel is bilinear in the fields, the mean field of an ensemble is itself a field whose
atoms are the atoms of every conformation with their weights multiplied by that
conformation's weight.  The kernel of two ensembles is then automatically the weighted mean
of the kernels of their conformation pairs, which is the mean-map kernel of the two
conformational distributions, and it stays positive definite.  Everything downstream, the
co-fields and the decomposition rules included, continues to apply unchanged, because the
model is still linear in this mean field.

The cost is quadratic in the number of conformations kept, so `max_conformers` is worth
setting on large series.

### `build_fields(mol: Chem.Mol, fieldset: FieldSet, name: str='', _single: bool=False) -> MoleculeFields`

Compute atom coordinates and per-field weights for one molecule.

With `conformers="ensemble"` in the field set and a molecule that carries more than one
conformation, this returns the ensemble field instead; see `ensemble_fields`.

## cmfqsar.flexalign

Source: `cmfqsar/flexalign.py`

Flexible alignment by confidence: the rigid-body search of 2019, with the torsions.

`cmfqsar.varalign` places a molecule by minimising the posterior variance of its predicted
activity -- the proposal of Baskin & Zhokhova, Future Med. Chem. (2019) -- but it moves the
molecule as a rigid body, so a flexible ligand can only be placed as well as its one conformation
allows.  This module adds the rotatable bonds to the same search, which is that proposal in full:
the pose and the conformation are chosen together, by confidence.

    minimise   sigma^2(X(p))  +  penalty(strain)  +  penalty(displacement),
    p = (rotation vector, translation, torsion angles)

**Why there is a strain term.**  Confidence alone will fold a molecule into whatever shape the
training set finds familiar, and nothing in the variance knows that a torsion has a barrier.  The
prior is the force field: the conformation may rise freely to `energy_slack` kcal/mol above the
one it started in, and is penalised quadratically beyond that.  One-sided on purpose -- a reward
for going *below* the starting energy would turn the aligner into a conformer optimiser, which is
not what it is for.

**The gradient is analytic throughout.**  The kernel gradient with respect to atom positions is
already in `cmfqsar.kernels.kernel_gradient` (probe fields included, as of this work); the
chain rule to a torsion is the generator of a rotation about the bond axis,

    d x_i / d theta_j  =  a_j x (x_i - p_j)      for the atoms that bond j moves,

evaluated in the current configuration, which is exact when the bonds are applied from the root
outwards: a rotation about bond j carries every bond inside its moving set with it.  The force
field supplies Cartesian gradients for the strain term, and they go through the same Jacobian.

**What it is worth, measured.**  Judged by the C1 protocol (`cmfqsar.poses`) on sixteen FEP
sets whose held-out conformations are *regenerated* rather than read from the crystal -- the case
a flexible method actually faces -- with the pose scrambled first every time:

==================================  ===========  =======  =======  ==========  ===============
method                              median RMSD  < 1 A    < 2 A    R2pred      conformer limit
==================================  ===========  =======  =======  ==========  ===============
crystal poses (upper limit)         0.00 A       100%     100%     +0.334      0.00 A
scrambled (lower limit)             6.12 A         0%       1%     -0.696      1.73 A
substructure (MCS)                  1.94 A        15%      54%     -0.391      1.73 A
MCS + rigid confidence              2.27 A        12%      45%     -0.396      1.73 A
MCS + flexible confidence           1.41 A        35%      74%     +0.065      1.20 A
MCS + flexible, 5 conformer starts  0.86 A        59%      88%     +0.264      0.71 A
==================================  ===========  =======  =======  ==========  ===============

Three things follow, and they set the defaults.  **The torsions make the criterion work**: with a
conformation to fix, the *rigid* confidence search is worse than not refining at all (2.27 against
MCS's 1.94), which is the failure C1 found; the same objective with the bonds free is the best
method available.  **The conformer generator is what makes it work** -- it is worth -0.55 A on the
median pose, better on 15 of 16 sets (95% CI -0.77 to -0.35), and the combination beats plain
substructure alignment on 16 of 16; so `conformers` defaults to five rather than zero.  **The
displacement guard has nothing left to prevent**: with these starts no set is worse than 1.52 A
and the large failures it was meant to catch are gone, a 1 A guard costs 0.6 A on jnk1, and 2 A is
indistinguishable from none, so the 2 A default is kept as insurance rather than as a necessity.

What it does not do is replace knowing the conformation: +0.264 against the crystal poses' +0.334.
The conformational residue is 0.71 A, and that is where the remaining error is.

### `class Torsion  (dataclass)`

One rotatable bond: the axis, and the atoms that turn with it.

Attributes:

- `pivot: int`
- `tip: int`
- `moving: np.ndarray` = `field(default_factory=lambda: np.zeros(0, dtype=int))`

### `class FlexibleAligner(model, max_rmsd: Optional[float]=2.0, penalty: float=50.0, energy_slack: float=10.0, energy_weight: float=1.0, max_torsions: int=12)  — bases: VarianceAligner`

The confidence search over rigid motion *and* the rotatable bonds.

`mol_for` must return the RDKit molecule behind a `cmfqsar.fields.MoleculeFields`,
since the torsions and the force field come from the bond graph, which the field
representation does not carry.

#### `FlexibleAligner.objective_flex(mol, rd: Chem.Mol, X0: np.ndarray, c: np.ndarray, torsions: Sequence[Torsion], strain: _Strain, sp: np.ndarray, kself: float, p: np.ndarray)`

#### `FlexibleAligner.align_flexible(mol, rd: Chem.Mol, restarts: int=2, seed: int=0, max_iter: int=300, apply: bool=True, torsion_start: float=0.4, conformers: int=0, prune_rmsd: float=0.5, any_orientation: bool=False) -> VarianceAlignResult`

Place `mol` by confidence, turning its rotatable bonds as well as moving it.

With `conformers`, the generator supplies that many further starting *configurations*,
each superimposed on the current pose, besides the one the molecule arrived in; the
search runs from each and the best is kept.  `any_orientation` has every restart draw a
uniformly random orientation; see `VarianceAligner.align`.

### `rotatable_torsions(mol: Chem.Mol, max_torsions: int=12) -> List[Torsion]`

The rotatable bonds of a molecule, ordered from the root outwards.

The smaller side of each bond is the one that moves, so the molecule's bulk stays where the
rigid-body parameters put it.  Ordering by the size of the moving set, largest first, is the
root-outward order the chain rule needs: a rotation then carries every later bond with it.

### `apply_torsions(X0: np.ndarray, torsions: Sequence[Torsion], thetas: Sequence[float]) -> np.ndarray`

X0 with every torsion turned by its angle, applied from the root outwards.

### `torsion_gradient(X: np.ndarray, torsions: Sequence[Torsion], g: np.ndarray) -> np.ndarray`

dF/dtheta from dF/dX, by the chain rule over the bond axes at the current geometry.

### `conformer_starts(mol: Chem.Mol, n: int, seed: int=0, prune_rmsd: float=0.5, reference=None)`

`n` generated conformations, each superimposed on the pose the molecule is in.

C2 asks for the conformer generator to supply the starting points, and the measurement says
why: after a flexible alignment the residual error is conformational, so a local torsion
search is only as good as the conformations it is allowed to start from.  Every member is
placed on the current pose first, so the rigid parameters mean the same thing for each.

### `flexible_align(mols: Sequence, rdmols: Sequence[Chem.Mol], model, max_rmsd: Optional[float]=2.0, energy_slack: float=10.0, energy_weight: float=1.0, restarts: int=2, seed: int=0, max_torsions: int=12, conformers: int=5, prune_rmsd: float=0.5, any_orientation: bool=False, progress=None, cancel=None) -> List[VarianceAlignResult]`

Align every molecule in place by confidence over rigid motion and the rotatable bonds.

## cmfqsar.fluctuation

Source: `cmfqsar/fluctuation.py`

The variance of a field across the frames of an ensemble (plan item D10).

The ensemble mode of this program represents a molecule by the **mean** of the fields of its
conformations (`cmfqsar.fields.combine_members`), which is the occupancy averaging of
4D-QSAR.  A mean throws away how much the field moved, and that is a flexibility map: where a
molecule's field is the same in every conformation, and where it is not.

The variance is a field of the same class, which is what makes it usable without a new kernel.
With the Gaussians written as `exp(-b/2 |r - c|^2)`, as everywhere else in this program,

    X_m(r) = sum_i w_i exp(-b/2 |r - r_mi|^2)

for conformation *m*, and the variance across conformations is

    V(r) = (1/k) sum_m X_m(r)^2  -  ( (1/k) sum_m X_m(r) )^2.

Every term is a product of two Gaussians, and a product of two Gaussians is a Gaussian:

    exp(-b/2 |r-p|^2) exp(-b/2 |r-q|^2) = exp(-b |r - (p+q)/2|^2) exp(-b |p-q|^2 / 4),

so V is again a sum of Gaussians -- at the **midpoints of atom pairs**, with **twice the width**,
and with a coefficient that falls off with how far apart the two atoms are.  The same-conformation
pairs enter with weight 1/k - 1/k^2 and the pairs from different conformations with -1/k^2, which
is the two terms of the variance written out.

`variance_field` builds that field, and `variance_fieldset` gives the field set it has
to be used with: the widths are doubled, so a variance field must not be mixed with an ordinary
one in the same kernel call.  Pairs whose coefficient is below a floor are dropped, which is what
keeps the construction from growing with the square of the atoms; `kept_share` says what the
floor cost.

**What this is for.** The plan expects it to be a display tool -- a map of where a series is
rigid and where it is not -- and asks for it to be measured as a block of its own under multiple
kernel learning, so that the mixing coefficient says whether it carries anything the mean does
not.  `benchmarks/run_fluctuation.py` does that.

**What it is not.** Only fields whose Gaussians share one width can be written this way, which is
every "gauss" field of a field set at one attenuation.  The steric field puts a different width on
every atom (b = alpha / R^2), so a pair of atoms of different radii gives a product Gaussian whose
width depends on the pair, which a single field set cannot express; `variance_fieldset`
refuses it rather than approximating.

### `variance_fieldset(fieldset: FieldSet) -> FieldSet`

The field set a variance field has to be evaluated with: the same fields at twice the width.

A product of two Gaussians of width `b` is a Gaussian of width `2b`, so the variance field
lives at twice the attenuation of the field it came from.  A field whose width is not the same
on every atom cannot be written this way and is refused.

### `variance_field(members: Sequence[MoleculeFields], fieldset: FieldSet, weights: Optional[Sequence[float]]=None, floor: float=PAIR_FLOOR, name: str='') -> MoleculeFields`

The variance of the fields of `members` across the ensemble, as a field.

`members` are the fields of one molecule in its conformations, in the same atom order.  The
result is a `cmfqsar.fields.MoleculeFields` whose atoms are pseudo-atoms at the
midpoints of atom pairs; it must be used with `variance_fieldset`, because its Gaussians
are twice as wide as the ones it was built from.

The weights need not be uniform: a Boltzmann-weighted ensemble has the same algebra with
`1/k` replaced by the member's weight.

### `kept_share(members: Sequence[MoleculeFields], fieldset: FieldSet, floor: float=PAIR_FLOOR) -> float`

The share of the pair factors the floor keeps, in [0, 1].

### `evaluate(field: MoleculeFields, fieldset: FieldSet, f: int, points: np.ndarray) -> np.ndarray`

One field of a `cmfqsar.fields.MoleculeFields` at a list of points.

Used to check a variance field against the variance it is supposed to be, and to draw one.

## cmfqsar.fraggrow

Source: `cmfqsar/fraggrow.py`

Growing a bound fragment into the gap the pocket leaves (plan item MC7).

A fragment that binds fills part of a site and leaves the rest empty.  What should be put there?
`cmfqsar.complementarity` already scores a whole ligand against a pocket by field overlap,
with the sign convention that makes a score of this kind mean something: a positive ligand region
should meet a negative protein region, hydrophobic meets hydrophobic, a donor meets an acceptor.
This module turns that into a question about one growth vector:

* **the gap** is what the pocket would still reward, on a grid, after the fragment's own field is
  taken off -- the field a group would have to place to be worth adding;
* **the candidates** are groups built from their SMILES and placed on the vector by rule, with
  the torsion about the new bond scanned, so that nothing about the answer enters the geometry;
* **the score** is the overlap of the grown molecule's added field with the gap, and a placement
  that clashes with the protein is refused.

The gate of plan item MC7 is whether the group that was actually added comes out among the first
ten of such an enumeration.  The plan says this is the item most likely to return nothing,
because a one-body field score has never been shown to rank linkers;
`benchmarks/run_fraggrow.py` is the measurement, and rule 5 then decides what is kept.

Nothing here needs a model: the pocket and the fragment are enough, which is what makes it usable
on a target with no series yet -- the situation a fragment campaign is in.

### `class Vector  (dataclass)`

A place on a fragment where a group can be added, and the frame that places it.

Attributes:

- `atom: int`
- `origin: np.ndarray`
- `rotation: np.ndarray`
- `hydrogen: Optional[int]` = `None`

#### `Vector.to_pocket(local: np.ndarray, turn: float=0.0) -> np.ndarray`

Local coordinates of a group, turned about the new bond, in the pocket's frame.

### `class Case  (dataclass)`

A fragment, the compound it was grown into, and what was added where.

Attributes:

- `fragment: int`
- `grown: int`
- `smiles: str`
- `atom: int`
- `added: List[int]`
- `names: Tuple[str, str]` = `('', '')`

### `class Group  (dataclass)`

A candidate group in its own frame: the attachment point at the origin, the bond on +x.

Attributes:

- `smiles: str`
- `mol: object`
- `local: np.ndarray`
- `first: int`

#### `Group.heavy` -> int  (property)

### `class Gap  (dataclass)`

What a group would still have to place, on a grid around one growth vector.

Attributes:

- `points: np.ndarray`
- `reward: Dict[str, np.ndarray]`
- `room: np.ndarray`
- `cell: float`

### `class Candidate  (dataclass)`

One group placed on one vector, with the best turn found for it.

Attributes:

- `smiles: str`
- `score: float`
- `turn: float`
- `clashes: int`
- `heavy: int`
- `placed: object` = `None`

### `vacant_direction(coords: np.ndarray, atom: int, neighbours: Sequence[int]) -> np.ndarray`

Where the next bond of `atom` would point, from the bonds it already has.

### `growth_vector(mol, atom: int) -> Optional[Vector]`

The frame of the bond that would be made at `atom`, with the hydrogen it replaces.

The direction is the one the atom's own bonds leave empty, so it is fixed by the fragment
alone and nothing about what will be added enters it.  The plane is fixed by the atom's
first heavy neighbour, which makes the frame reproducible.

### `addition(fragment, grown, max_added: int=20) -> Optional[Tuple[str, int, List[int]]]`

(the added group as SMILES, the fragment atom it hangs on, the added atoms of `grown`).

Both molecules are taken without hydrogens, and the fragment must be a substructure of the
grown compound, which is what makes this a growing case rather than a pair of analogues.
The group is written by the same single-bond cut as the matched pairs of
`cmfqsar.bioisostere`, so the two items speak of groups in the same words.

### `build_group(smiles: str, seed: int=1) -> Optional[Group]`

A group written as SMILES with one attachment point, embedded and put in its own frame.

The geometry comes from the SMILES and a force field, never from a crystal structure, so a
candidate that happens to be the right answer is built exactly like every other one.

### `grow(fragment, vector: Vector, group: Group, turn: float=0.0)`

The fragment with the group bonded on, as one molecule with coordinates.

### `gap_points(vector: Vector, reach: float=REACH, spacing: float=SPACING) -> np.ndarray`

A ball of grid points in front of the growth vector, where a group could go.

### `gap(pocket: MoleculeFields, fragment: MoleculeFields, fieldset: FieldSet, vector: Vector, reach: float=REACH, spacing: float=SPACING) -> Gap`

What a group would be rewarded for placing, in the space the fragment leaves free.

One array per ligand field type, with the sign convention of
`cmfqsar.complementarity.COMPLEMENTARY_PAIRS`: the pocket's electrostatic field counts
with a minus, its hydrophobic field with a plus, and its donors pair with the ligand's
acceptors.  `room` is the space left by the protein **and** by the fragment, which is how
"minus what the fragment already places" is taken here: an offer is open only where a new
atom could sit.  Shape needs no field of its own, because that is what the room is.

### `room(points: np.ndarray, pocket: MoleculeFields, contact: float=CONTACT, ramp: float=RAMP) -> np.ndarray`

How much of a ligand atom the pocket allows at each point, from 0 to 1.

This is not `cmfqsar.complementarity.free_space`, which asks where a ligand could sit
comfortably and is right for comparing a co-field with a pocket.  A group that is added to a
bound fragment **touches** the protein -- that contact is the interaction -- so the mask here
has to be open at the contact distance and closed only where an atom would not fit.

### `score(added: MoleculeFields, fieldset: FieldSet, g: Gap, weights: Optional[Dict[str, float]]=None) -> float`

How much of the gap one added group fills, as one number.

The integral is taken only where the pocket leaves room, so a group cannot be rewarded for
placing field inside the protein.

### `clashes(mol, pocket: MoleculeFields, atoms: Optional[Sequence[int]]=None, contact: float=CLASH) -> int`

How many heavy atoms of the molecule sit closer than `contact` to a protein heavy atom.

### `try_group(fragment, vector: Vector, group: Group, pocket: MoleculeFields, fieldset: FieldSet, g: Gap, turns: int=TORSIONS, keep_placed: bool=False) -> Optional[Candidate]`

Place one group on the vector, turn it, and keep the best placement that does not clash.

The molecule is assembled and its fields are built **once**: an atom's type, weight and
radius come from the bond graph, which the turn about the new bond does not change, so every
further turn is a rotation of the added atoms' coordinates and nothing else.

### `enumerate_groups(fragment, vector: Vector, pocket: MoleculeFields, fieldset: FieldSet, groups: Sequence[Group], turns: int=TORSIONS, reach: float=REACH, spacing: float=SPACING, progress=None) -> List[Candidate]`

Every group of the library placed on one vector, best first.

### `rank_of(candidates: Sequence[Candidate], smiles: str) -> Optional[int]`

Where one group came in the enumeration, counting from one.

## cmfqsar.fusion

Source: `cmfqsar/fusion.py`

Data fusion: one score from the similarities to several known actives.

A similarity search needs a query, and a project that has found more than one active has more
than one.  The screening literature answers that with *data fusion*: compute the similarity of
each library molecule to every known active and combine the results by a fixed rule (Willett,
*Data fusion in ligand-based virtual screening*, 2013, and the group-fusion rules of Fox & Shaw
1994 as they reached chemoinformatics through Whittle, Gillet, Willett et al. 2004).  No model is
fitted and no inactive is needed, which is why fusion is the next step beyond a single-query search
when a few actives are all there is.

The input is one matrix: `S[i, j]`, the similarity of library molecule `i` to reference active
`j`.  For this program that is the cosine-normalised CMF kernel between two sets of molecular
fields, so the geometry of the comparison is the overlap integral and nothing here has to know
it; the same rules run unchanged on a fingerprint Tanimoto, which is what makes the two
representations comparable on one task.

Eight rules, and the two that matter most are at the ends of the list:

`max`
    the similarity to the *best-matching* active.  The rule the literature most often finds
    best, and the one this program did not have: a molecule resembling one active strongly is a
    better candidate than one resembling all of them weakly, because a series usually binds through
    one arrangement and not through an average of several.
`min`
    the similarity to the worst-matching active, which asks for resemblance to all of them at
    once.  Included because it is the opposite extreme and bounds what the others can do.
`sum`
    the total, which ranks exactly as the mean does.  This is the similarity ensemble, and it is
    the rule `cmfqsar.oneclass`'s centroid description already implements: for a normalised
    kernel the distance to the centroid of the actives is one minus their mean similarity.  It
    is here for completeness and as the baseline the others have to beat.
`med`
    the median, which is the robust form of `sum` and ignores one outlying reference.
`anz`, `mnz`
    Fox and Shaw's combinations, on the actives whose similarity passes a threshold: `anz` is
    the sum divided by how many passed and `mnz` the sum multiplied by it.  They act in
    opposite directions, `anz` rewarding a molecule that matches few actives well and `mnz`
    one that matches many at all, so the pair bounds how much the *number* of matches should
    count beside their strength.
`euc`
    the Euclidean norm of the similarity vector, which lies between `max` and `sum` and
    weights a strong match more than a linear sum does without discarding the rest.
`rrf`
    reciprocal rank fusion: each reference ranks the whole library, and a molecule scores
    `sum_j 1 / (k + r_ij)`.  The only rule here that uses ranks rather than similarity values,
    so it is the one that does not care whether the similarity scales of two references agree,
    and the only one whose score for a molecule depends on the rest of the library.

A molecule is never its own reference.  When the library being scored *is* the reference set,
pass `training=True` and the diagonal is excluded, exactly as the nearest-neighbour score of
`cmfqsar.oneclass` does; without that every rule involving a maximum returns the
self-similarity and the ranking is meaningless.

### `class FusionScorer  (dataclass)`

A fusion rule with the interface of a fitted one-class description.

It holds no model: the references *are* the description, and `fit` only records them and
places the offset so that a fraction `nu` of them falls outside, which is what makes a
decision value here mean the same thing as one from `cmfqsar.oneclass`.  The purpose
of fusion is that nothing is fitted, so there is nothing to overfit and nothing to
cross-validate; what there is instead is a choice of rule, and the rule has to be chosen on
data of its own if the choice is not to become the fit.

Attributes:

- `kind: str` = `'max'`
- `nu: float` = `0.1`
- `threshold: float` = `0.0`
- `rrf_k: float` = `60.0`
- `top_fraction: Optional[float]` = `None`
- `offset_: float` = `0.0`
- `n_ref_: int` = `0`
- `diag_: Optional[np.ndarray]` = `None`
- `is_linear: bool` = `False`

#### `FusionScorer.label` -> str  (property)

How this rule is named in a message to the user.

#### `FusionScorer.why_not_linear` -> str  (property)

Why no co-field exists, in a clause that can be dropped into an error message.

#### `FusionScorer.fit(K: np.ndarray) -> 'FusionScorer'`

Record the reference set and calibrate the offset on it, self-matches excluded.

#### `FusionScorer.decision(K_test_train: np.ndarray, k_test_self: Optional[np.ndarray]=None, training: bool=False) -> np.ndarray`

Fused score of every molecule; at or above zero is inside the description.

`k_test_self` is accepted and ignored.  Every rule here is a function of the similarity
row alone, so unlike the centroid and subspace distances of `cmfqsar.oneclass` this
needs no self-kernel -- which also means it says nothing unless the kernel is normalised,
since an unnormalised row ranks by molecular size.

### `fuse(S: np.ndarray, rule: str='max', threshold: float=0.0, rrf_k: float=60.0, top_fraction: Optional[float]=None, training: bool=False) -> np.ndarray`

Combine the similarities of every library molecule to several actives into one score.

`S` is (library, references).  Larger is better for every rule, so the result can be ranked,
thresholded or handed to the screening metrics directly.  `threshold` is what counts as a
match for `anz` and `mnz`; `rrf_k` is the constant of reciprocal rank fusion, and
`top_fraction`, if given, restricts the rules that count matches to the actives for which
the molecule falls in that leading fraction of the library, which is the form in which the
rank-based rule is usually quoted.

A row of nothing but missing values scores as missing rather than as zero, which keeps a
molecule that could not be compared with anything out of the top of the list.

### `make(rule: str='max', nu: float=0.1, **kwargs) -> FusionScorer`

Scorer for one of `FUSION_RULES`.

## cmfqsar.gpc

Source: `cmfqsar/gpc.py`

Gaussian-process classification by the Laplace approximation.

The support-vector classifiers of this program return a decision value and nothing else: to get
a probability out of one, a sigmoid has to be fitted to it afterwards, and there is no posterior
at all.  A Gaussian process puts a prior on the latent function instead and gives three things
the hinge loss cannot:

* **calibrated probabilities**, from the likelihood rather than from a sigmoid fitted afterwards;
* **a posterior variance per molecule**, so "active" and "no idea" are different answers, which
  is what the confidence-based alignment of `cmfqsar.varalign` needs for a classification
  set;
* **the marginal likelihood**, which scores a kernel without cross-validation and so can choose
  the attenuation, the amplitude and the mixing coefficients from the training molecules alone.

A Bernoulli likelihood is not conjugate to the Gaussian prior, so the posterior is approximated
by a Gaussian at its mode (Rasmussen and Williams, *Gaussian Processes for Machine Learning*,
chapter 3; the Newton iteration is their Algorithm 3.1, the predictions 3.2 and the marginal
likelihood 3.3).  Everything here works from a precomputed kernel, which is what the CMF
machinery provides, and nothing in it looks at a molecule.

The latent mean stays a kernel expansion, `f(x) = sum_j a_j K(x, x_j)`, so the co-field and the
atom contributions remain exactly as well defined as they are for kernel ridge regression: unlike
a nonlinear kernel layer, this classifier keeps the interpretation intact.

Measured on the six classification sets of `examples/`, the accuracy is equal to that of C-SVC; the
reason to use it is the three things above, not the Matthews correlation.

`klr_fit` (plan item E9) is the same Laplace step read as a learner of its own: kernel
logistic regression is exactly this mode with the logistic likelihood, so it reuses the iteration
here instead of repeating it, and differs only in choosing its ridge on the folds rather than from
the evidence and in not computing the posterior variance.

### `class LaplaceGPC  (dataclass)`

Binary GP classifier on a precomputed kernel, fitted at the posterior mode.

`amplitude` multiplies the kernel and is the only hyper-parameter: a large one lets the
latent function move far from zero, which is what makes the fit confident, and it plays the
role C plays for a support-vector classifier.  Unlike C it can be chosen by the marginal
likelihood, without cross-validation; see `select_by_evidence`.

Attributes:

- `likelihood: str` = `'probit'`
- `amplitude: float` = `1.0`
- `jitter: float` = `1e-08`
- `max_iter: int` = `100`
- `tol: float` = `1e-09`
- `evidence: bool` = `True`
- `y_: Optional[np.ndarray]` = `None`
- `f_: Optional[np.ndarray]` = `None`
- `grad_: Optional[np.ndarray]` = `None`
- `sqrt_W_: Optional[np.ndarray]` = `None`
- `L_: Optional[np.ndarray]` = `None`
- `log_marginal_likelihood: float` = `float('nan')`
- `n_iter: int` = `0`
- `converged: bool` = `False`

#### `LaplaceGPC.fit(K: np.ndarray, y: np.ndarray) -> 'LaplaceGPC'`

Newton iteration on the posterior of the latent values (Algorithm 3.1).

#### `LaplaceGPC.coef` -> np.ndarray  (property)

Dual coefficients of the latent mean, with the amplitude folded in.

The latent mean of a new molecule is `k(x, X) @ coef`, so the rest of the program can
treat this model exactly like kernel ridge regression: same decomposition rules, same
co-field, same atom contributions.

#### `LaplaceGPC.latent(K_test_train: np.ndarray) -> np.ndarray`

#### `LaplaceGPC.latent_variance(K_test_train: np.ndarray, k_test_self: np.ndarray) -> np.ndarray`

Posterior variance of the latent function (Algorithm 3.2).

#### `LaplaceGPC.predict_proba(K_test_train: np.ndarray, k_test_self: Optional[np.ndarray]=None) -> np.ndarray`

p(active) for new molecules, averaged over the latent posterior where it is known.

### `squash(mean: np.ndarray, var: np.ndarray, likelihood: str='probit') -> np.ndarray`

p(active) from the latent mean and variance, averaged over the latent posterior.

For the probit likelihood the average is exact, `Phi(mean / sqrt(1 + var))`; for the
logistic one it is MacKay's approximation.  Either way the variance pulls the probability
towards one half, which is the purpose: an uncertain molecule should not be called confidently.

### `klr_fit(K: np.ndarray, y: np.ndarray, lam: float, max_iter: int=100, tol: float=1e-09) -> Tuple[np.ndarray, float, 'LaplaceGPC']`

Kernel logistic regression (plan item E9): the probabilistic classifier that is linear
in the field.

It minimises the logistic loss of the latent function plus a ridge on its norm in the
reproducing-kernel Hilbert space,

    sum_i log(1 + exp(-y_i f(x_i)))  +  (lam / 2) ||f||^2,      f(x) = sum_j a_j K(x, x_j),

which is a smooth convex problem solved by Newton's method.  **It is the same problem the
Laplace step of `LaplaceGPC` already solves**, because that step finds the mode of
`log p(y | f) - f' (s K)^-1 f / 2` with the logistic likelihood, and the two agree when the
amplitude is `s = 1 / lam`.  So this function runs that step rather than a second Newton
iteration of its own; `tests/test_klr.py` checks that the two give the same coefficients.

Two things differ from `gpc`, and they are the whole of the difference:

* **lam is chosen by cross-validation, not by the marginal likelihood.**  The Gaussian process
  picks its amplitude from the evidence, which needs no held-out molecules; this learner picks
  lam the way kernel ridge regression picks its own, on the folds.
* **there is no posterior variance.**  The probability is the likelihood at the mode,
  `p(active) = 1 / (1 + exp(-f))`, not the likelihood averaged over a posterior, so a
  molecule unlike anything in the training set is *not* pulled towards one half.  That is what
  makes the fit cheaper -- the factorisation the variance is read from is never made -- and it
  is also what the Gaussian process gives that this does not.

The decision boundary is `f = 0`, which is `p(active) = 0.5`: the class the model reports
and its probability are the same statement, unlike a discriminant whose threshold is placed
separately.  On a set with far more inactives than actives that boundary follows the larger
class, and moving it is then worth considering; `cmfqsar.model` takes a `klr_threshold`
parameter for that and leaves it off by default.

Returns (dual coefficients, bias, the fit), the bias being zero.

### `evidence(K: np.ndarray, y: np.ndarray, amplitude: float=1.0, likelihood: str='probit') -> float`

Approximate log marginal likelihood of a kernel, with no cross-validation anywhere.

### `select_by_evidence(kernels: Dict[object, np.ndarray], y: np.ndarray, amplitudes: Sequence[float]=(0.3, 1.0, 3.0, 10.0, 30.0), likelihood: str='probit') -> Tuple[object, float, float]`

Pick the kernel and the amplitude with the largest marginal likelihood.

`kernels` is keyed by whatever labels them, usually the attenuation.  Returns
(key, amplitude, log marginal likelihood).  This is the cross-validation-free alternative to
the grid search: every candidate is fitted once, on all the molecules.

### `brier_score(y: np.ndarray, p: np.ndarray) -> float`

Mean squared error of the probabilities; 0 is perfect, 0.25 is no better than chance.

### `calibration_table(y: np.ndarray, p: np.ndarray, bins: int=5)`

Rows of (bin centre, mean predicted probability, observed fraction, count).

## cmfqsar.gradients

Source: `cmfqsar/gradients.py`

The derivative of a prediction with respect to the atoms of the molecule being predicted.

A CMF model predicts an overlap integral.  The prediction of molecule *x* is

    y(x) = sum_j a_j K(x, j) + b,

and the kernel is a sum of Gaussian overlaps over pairs of atoms, one atom from *x* and one from
training molecule *j*.  A Gaussian overlap is an analytic function of the distance between the two
atoms, so `y` is an analytic function of the coordinates of the atoms of *x* and of the per-atom
weights that carry its fields.  This module differentiates it.

What the derivative is for
--------------------------
Three things, all of which otherwise have to be done by trying one structure at a time:

* **Guiding a generative model.**  A three-dimensional generative model proposes atom positions.
  With a gradient, a fitted CMF model can be used as a guidance term inside that proposal: the
  gradient says which way every atom would have to move to raise the predicted activity.  Nothing
  has to be enumerated first.
* **Refining a pose or a conformation by predicted activity.**  `cmfqsar.varalign` and
  `cmfqsar.flexalign` already move a molecule by *confidence*, with their own gradients.
  This module supplies the other objective, the prediction itself, in the same analytic form.
* **Reading the model at an atom.**  The derivative with respect to an atom's weight is a local
  statement: how much the prediction would change if that one atom carried a little more charge,
  or were a little more hydrophobic.  That is the question a chemist asks of a co-field picture,
  answered as a number per atom instead of as a surface.

The derivatives
---------------
Write the raw kernel of field *f* as a double sum over atoms, with `g_f` the Gaussian overlap of
two atoms as a function of their squared distance:

    K_f(x, j) = sum_l sum_m w_fl w_fjm g_f(|r_l - r_jm|^2)

and the molecule's own self-kernel as `d_f(x) = K_f(x, x)`.  Then

    dK_f(x, j) / dr_l   = 2 sum_m w_fl w_fjm dg_f (r_l - r_jm)
    dK_f(x, j) / dw_fl  = sum_m w_fjm g_f(|r_l - r_jm|^2)
    dd_f(x) / dr_l      = 2 (the same sum, over the atoms of x itself)
    dd_f(x) / dw_fl     = 2 sum_m w_fm g_f(|r_l - r_m|^2)

where `dg_f` is the derivative of the overlap with respect to the squared distance.  The factor
of two in the last two lines is there because atom *l* appears in both arguments of the
self-kernel.  The ratio of that derivative to the overlap itself is a constant per field
(`-alpha_f/4`) for a field whose Gaussians share one width, and a constant per atom pair for the
radius-scaled steric field; that is the part `cmfqsar.kernels.kernel_gradient` already did,
and the coordinate part of this module is a chain rule on top of it.

Everything the model puts between the raw kernel and the prediction is differentiated with it:

* the **cosine normalisation**, through `s_f(x) = d_f(x) ** -0.5` and its derivative
  `-0.5 s_f ** 3 dd_f`.  This term is what makes the gradient of a normalised model different
  from the gradient of its kernel row: moving an atom changes what the molecule is divided by, as
  well as its overlap with the training set;
* the **field scaling and mixing**, `h_f c_f`, which are constants of the fitted model;
* the **nonlinear kernel layer** (`gauss`, `arccos1`, `arccos2`), which depends on the
  linear kernel and on the molecule's own self-kernel, so it contributes through both;
* the **non-field descriptors**, which contribute nothing: they are computed from the molecular
  graph or read from a file property, so they do not change when an atom moves.

What the derivative is not
--------------------------
**A weight is not a molecule.**  The weight of an atom in the electrostatic field is its partial
charge, and the charges of a real molecule are not free: they sum to its total charge and they
follow from its structure.  A positive derivative with respect to one atom's weight says the model
would like more of that quantity there.  It does not say that a molecule exists in which only that
atom's charge has changed.  The same holds for the other fields, whose weights are table
increments of an atom type.  Treat the weight gradient as a direction in the field
representation, to be realised by choosing a different atom or a different substituent, not as a
molecule.

**The gradient is local and the model has a domain.**  A step along the gradient leaves the
training set, which is where a CMF model is least reliable.  The posterior variance
(`cmfqsar.model.CMFModel.predict_with_variance`), the conformal interval
(`cmfqsar.conformal`) and the two-sample test (`cmfqsar.mmd`) are the three things that
say how far a step can be trusted; the lower confidence bound of `cmfqsar.active` is the
objective to follow when the aim is a molecule to make rather than a number to maximise.

**Maximising an overlap is not designing a molecule.**  The overlap of a field with a positive
lobe of a co-field grows without limit as atoms are piled into that lobe.  Nothing in this
gradient knows about valence, bond lengths or sterics.  It is a term to add to a generator or a
force field that does, not an optimiser on its own.

**Probe fields have no per-atom weight.**  In a GRID-like probe field every atom carries several
Gaussians whose coefficients come from the expansion of a probe interaction curve
(`cmfqsar.probes`), not from one weight.  The coordinate gradient covers them; the weight
gradient reports zero for their columns and names them in
`PredictionGradient.weightless_fields`.

Which models are covered
------------------------
Any fitted model whose prediction is a sum over the training molecules plus a constant: the
regression learners, the classifiers (the gradient is then of the decision value, not of the
probability) and the ranking SVM, on the overlap-integral kernel.  The rotation-invariant,
surface and periodic kernels are built from the distances *inside* each molecule and are not
overlap integrals, so this gradient does not apply to them; neither do the one-class and
data-fusion scores, whose decision is not of that form.  Every case that is not covered is
refused with a reason rather than approximated.

Checked against finite differences
----------------------------------
`tests/test_gradients.py` holds the analytic gradient to central finite differences of
`cmfqsar.model.CMFModel.predict` itself, over the combinations that change the chain rule:
normalised and unnormalised kernels, the steric field with its per-atom widths, probe fields,
interpolated attenuations, the three nonlinear layers, a descriptor block and a conformational
ensemble.  The remaining disagreement is that of the finite difference, not of the gradient.

### `class PredictionGradient  (dataclass)`

The prediction of one molecule and its derivatives.

Attributes
----------
value : the prediction itself, the number `cmfqsar.model.CMFModel.predict` gives.
coords : (N, 3) derivative with respect to the atom positions, in activity units per
    angstrom, or None when it was not asked for.  For a molecule that is an ensemble of
    conformations or states, the rows are the atoms of all of its members, one member after
    another, in the order the ensemble was built.
weights : (N, F) derivative with respect to the per-atom field weights, in activity units per
    unit of weight, one column per field of the field set, or None when it was not asked for.
field_names : the field of every column of `weights`.
weightless_fields : fields whose weight column is zero because they have no single per-atom
    weight to differentiate (the probe fields).

Attributes:

- `value: float`
- `coords: Optional[np.ndarray]` = `None`
- `weights: Optional[np.ndarray]` = `None`
- `field_names: List[str]` = `field(default_factory=list)`
- `weightless_fields: List[str]` = `field(default_factory=list)`

#### `PredictionGradient.forces` -> Optional[np.ndarray]  (property)

The negative of the coordinate derivative: the direction in which every atom would
have to move to *lower* the prediction, which is the sign convention of a force.

#### `PredictionGradient.largest_coordinate_terms(n: int=5) -> List[Tuple[int, float]]`

The `n` atoms whose position matters most, as (atom index, length of its derivative).

#### `PredictionGradient.largest_weight_terms(n: int=5) -> List[Tuple[int, str, float]]`

The `n` largest weight derivatives, as (atom index, field name, derivative).

### `prediction_gradient(model: CMFModel, mol: MoleculeFields, coords: Optional[np.ndarray]=None, weights: Optional[np.ndarray]=None, wrt: str='both', chunk: int=4000) -> PredictionGradient`

The prediction of `mol` and its analytic derivatives with respect to its atoms.

Parameters
----------
model : a fitted `cmfqsar.model.CMFModel` on the overlap-integral kernel.  Its
    training poses define the frame, exactly as they do for a prediction.
mol : the molecule, as `cmfqsar.fields.MoleculeFields` from
    `cmfqsar.fields.build_fields`, in the frame of the training set.
coords : positions to differentiate at, (N, 3), instead of the molecule's own.  Nothing is
    written back to `mol`, which is what lets a caller follow a gradient without rebuilding
    the fields at every step.
weights : per-atom weights to differentiate at, (N, F), instead of the molecule's own.
wrt : "coords", "weights" or "both".  Asking for less costs less.
chunk : training atoms per block of the coordinate sum, which bounds the memory it uses.

Returns
-------
`PredictionGradient`

Notes
-----
The per-atom weights of a field are not free parameters of a molecule, and a step along the
gradient leaves the training set.  Read the limits in the module docstring before using
either derivative for design.

### `numerical_prediction_gradient(model: CMFModel, mol: MoleculeFields, h: float=1e-05, wrt: str='both') -> PredictionGradient`

The same derivatives by central finite differences of `CMFModel.predict` (slow).

The reference the analytic gradient is tested against, and the way to check a model this
module has not been tried on.  It costs two predictions per degree of freedom, so it is for
small molecules and for tests, not for a design loop.

### `smoothed_cofield(model: CMFModel, f: int, points: np.ndarray, width: Optional[float]=None, include_scale: bool=True) -> np.ndarray`

The co-field of field `f` at `points`, optionally smoothed by an atom's own Gaussian.

With `width` of None this is the co-field itself, which is what
`cmfqsar.cofields.cofield_at_points` gives, computed exactly rather than with a
neighbour cutoff.  With `width` set to the `b` of an atom's Gaussian (`alpha` for an
ordinary field, `alpha / R_vdw ** 2` for the steric one), the value at a point is what an
atom of unit weight centred there would contribute to the prediction, before the mixing
coefficient and the field scaling that `include_scale` applies.

### `cofield_gradient(model: CMFModel, f: int, points: np.ndarray, width: Optional[float]=None, include_scale: bool=True) -> np.ndarray`

The gradient of that same function, (M, 3).

This is the co-field read as a potential: at a point in the binding site it says which way a
piece of field of this type would have to move to raise the prediction.  It is the quantity a
generative model can be steered by, and it needs no molecule, only a place to stand.

For an unnormalised linear model the two gradients are the same statement: the derivative of
the prediction with respect to the position of atom *l* is `h_f c_f w_fl` times this
gradient, taken at the atom's own position and smoothed by the atom's own Gaussian, summed
over the fields.  `prediction_gradient` is the form to use on a molecule, because it
also carries the normalisation and the nonlinear layer; this one is the form to use at a point
where no atom is yet.

## cmfqsar.gtm

Source: `cmfqsar/gtm.py`

Chemical space and activity landscapes from CMF kernels: kernel PCA + Generative Topographic
Mapping.

GTM (Bishop, Svensen & Williams, Neural Comput. 10 (1998) 215) is a probabilistic latent-variable
map: a square grid of K nodes in a two-dimensional latent space is mapped into data space by
y(x) = W phi(x) with radial basis functions phi, and the data are modelled as a constrained mixture
of Gaussians centred on the mapped nodes, fitted by expectation-maximisation.  Each molecule
therefore gets a posterior distribution over the nodes (its "responsibility"), from which the
chemical-space map, the activity or class landscapes and the applicability domain follow
(Gaspar, Baskin, Marcou, Horvath & Varnek, J. Chem. Inf. Model. 55 (2015) 84; Mol. Inf. 34 (2015) 348).

Molecular fields are not vectors, so GTM cannot be applied to them directly; only the kernel is
available.  The CMF kernel is an inner product in the Hilbert space of the fields, so kernel PCA on
it is functional PCA, and its leading components are coordinates that reproduce the feature-space
distances exactly.  Running the ordinary GTM in those coordinates is equivalent to a kernelised GTM
(cf. Olier, Vellido & Giraldo, ESANN 2010) and keeps the out-of-sample projection a single formula,
which a map needs in order to place new molecules.

Two practical points, both learned from experience elsewhere in this program:

* use cosine-normalised kernels, otherwise the leading components follow molecular size;
* a CMF kernel between molecules that are not aligned in a common frame is meaningless, so a map
  spanning several series would partly show alignment quality.  The way out is the limit of small
  attenuation: as alpha -> 0 a continuous indicator field tends to the count of atoms of that type
  (Sitnikov et al. 2015), so the kernel becomes an alignment-free fragment-descriptor kernel and the
  map can span a whole library.

### `class KernelPCA  (dataclass)`

Principal components of a set of molecules described by a kernel matrix.

Attributes:

- `n_components: int` = `20`
- `alphas: Optional[np.ndarray]` = `None`
- `lambdas: Optional[np.ndarray]` = `None`
- `K_fit_rows_: Optional[np.ndarray]` = `None`
- `K_fit_all_: float` = `0.0`

#### `KernelPCA.fit(K: np.ndarray) -> 'KernelPCA'`

#### `KernelPCA.transform(K_new: Optional[np.ndarray]=None) -> np.ndarray`

Coordinates of molecules whose kernels against the training set are `K_new` (M, N).

With `K_new=None` the coordinates of the training molecules themselves are returned.

#### `KernelPCA.explained_variance_ratio` -> np.ndarray  (property)

### `class GTM  (dataclass)`

Generative Topographic Mapping fitted by expectation-maximisation.

Parameters
----------
grid : the map is grid x grid nodes.
rbf : rbf x rbf radial basis functions constrain the mapping (its smoothness).
rbf_width : width of those functions, in units of the spacing between their centres.
reg : weight-decay regularisation of the mapping (the variational/regularised GTM of
    Olier & Vellido guards against overfitting; this is its simple quadratic form).

Attributes:

- `grid: int` = `20`
- `rbf: int` = `5`
- `rbf_width: float` = `1.0`
- `reg: float` = `0.1`
- `n_iter: int` = `100`
- `tol: float` = `1e-06`
- `seed: int` = `0`
- `W: Optional[np.ndarray]` = `None`
- `beta: float` = `1.0`
- `Phi: Optional[np.ndarray]` = `None`
- `latent: Optional[np.ndarray]` = `None`
- `log_likelihood_: List[float]` = `field(default_factory=list)`

#### `GTM.fit(T: np.ndarray, progress: Optional[Callable[[int, int, float], None]]=None) -> 'GTM'`

#### `GTM.responsibilities(T: np.ndarray) -> np.ndarray`

Posterior probability of every node for every molecule, shape (K, N).

#### `GTM.transform(T: np.ndarray) -> np.ndarray`

Posterior mean position of every molecule in the latent square, shape (N, 2).

#### `GTM.modes(T: np.ndarray) -> np.ndarray`

Most probable node of every molecule, shape (N, 2).

### `class CMFMap  (dataclass)`

A chemical-space map of a set of molecules described by CMF kernels.

Workflow: `fit(table, fieldset)` computes the kernel matrix, its principal components and the
GTM; `project(other)` places further molecules on the same map.

Attributes:

- `n_components: int` = `20`
- `grid: int` = `20`
- `rbf: int` = `5`
- `rbf_width: float` = `1.0`
- `reg: float` = `0.1`
- `n_iter: int` = `100`
- `normalize: bool` = `True`
- `seed: int` = `0`
- `kpca: Optional[KernelPCA]` = `None`
- `gtm: Optional[GTM]` = `None`
- `train_: Optional[object]` = `None`
- `fieldset_: Optional[object]` = `None`
- `coords_: Optional[np.ndarray]` = `None`
- `R_: Optional[np.ndarray]` = `None`
- `scores_: Optional[np.ndarray]` = `None`

#### `CMFMap.fit_pca(table, fieldset, progress: Optional[Callable[[str, float], None]]=None) -> 'CMFMap'`

The kernel and its functional principal components, without the map.

The score plot needs no GTM, and the GTM is the slow half -- a hundred EM iterations
against one eigendecomposition.  It also leaves `gtm` and `R_` as None, so a map
fitted this far cannot be mistaken for one that can carry a landscape.

#### `CMFMap.scores(i: int=0, j: int=1) -> np.ndarray`

The fitted molecules on two principal components, as an (N, 2) array.

Unlike the latent coordinates of the map, these are distances one may read: the
projection is linear and orthogonal, so the plane is a faithful projection of the kernel
geometry as far as its two eigenvalues reach.

#### `CMFMap.fit(table, fieldset, progress: Optional[Callable[[str, float], None]]=None) -> 'CMFMap'`

#### `CMFMap.project(table)`

(latent coordinates, responsibilities) of further molecules on the fitted map.

#### `CMFMap.activity_landscape(values, min_density: float=0.001) -> np.ndarray`

#### `CMFMap.class_landscape(classes, min_density: float=0.001) -> np.ndarray`

#### `CMFMap.density() -> np.ndarray`

#### `CMFMap.as_image(node_values: np.ndarray) -> np.ndarray`

Reshape per-node values into the grid, ready to be drawn.

### `density(R: np.ndarray) -> np.ndarray`

Cumulated responsibility of every node: the data support, i.e. applicability domain.

### `landscape(R: np.ndarray, values: np.ndarray, min_density: float=0.001) -> np.ndarray`

Responsibility-weighted mean of `values` at every node; NaN where there is no support.

### `class_landscape(R: np.ndarray, classes: np.ndarray, min_density: float=0.001) -> np.ndarray`

Responsibility-weighted fraction of the positive class at every node (NaN without support).

### `predict_from_landscape(R_new: np.ndarray, land: np.ndarray) -> np.ndarray`

Predict a property of new molecules as their responsibility-weighted landscape value.

## cmfqsar.hops

Source: `cmfqsar/hops.py`

A migration path as the sample: the environment of a hop, its frame and its labels (BT7).

Ionic conductivity in a solid electrolyte is set by the barrier a mobile ion meets between two
sites, not by either site.  That barrier is a field property in the sense of this program: the
ion moving from one site through the bottleneck to the next feels the electrostatic potential,
the repulsion and the polarisability of the framework around the path.  In the language of the
rest of the program the framework is the receptor, the ion is the ligand and the barrier is the
binding energy.

The sample is therefore one **hop**: a pair of sites of the mobile species, in one structure,
with the migrating ion itself removed.  Everything the per-atom items built (plan items LP1 to
LP3) carries over, with three changes that are this module:

* **The environment is cut from a periodic cell.**  `point_environment` is
  `cmfqsar.localenv.environment` for a periodic structure: it collects every atom and every
  periodic image within the reach of the envelope, measured from a *point* rather than from an
  atom, and returns an ordinary `cmfqsar.fields.MoleculeFields`.  The centre of a hop is
  the midpoint of the two sites, so the first atom of the result is a weightless marker there,
  which keeps the convention of the local modules that the centre is atom 0.
* **The frame is the hop itself.**  The origin is the midpoint and z points from the first site
  to the second, which is the direction the barrier is measured in.  What is left is the rotation
  about z, and `hop_frames` does not choose it: the environment is posed at `n_poses`
  equally spaced angles and averaged, which is a mean-map kernel over the rotation group about
  the path (the device of `cmfqsar.localalign` for a frame ambiguity).  The starting angle
  is fixed by a rule, the nearest framework atom to the midpoint, so that the discrete set is
  itself a rule.  Reversing a hop reverses z, and the forward and the reverse barrier are two
  samples.
* **The sample index is a pair.**  A hop is (structure, site a, site b, lattice offset of b), and
  the folds hold out whole structures (`HopSet.targets` builds an
  `cmfqsar.atomtargets.AtomTargets` whose "molecule" is the structure), since the hops
  of one cell are not independent of each other.

**Labels.**  `bvse_barriers` reads the barrier of every hop from the bond-valence
site-energy landscape (the softBV model, through the `bvlain` package): the lowest energy at
which the two sites are connected through the landscape, which is a minimax path and the
quantity the percolation analysis of that method is built from.  It is cheap, so it labels
thousands of hops, and under rule 1 of the plan it is also the reference a model trained on
DFT barriers has to beat.  Measured or DFT barriers come from a table (`read_hop_labels`).

**From hops to conduction.**  `percolation_thresholds` takes one energy per hop of a
structure and reports the level at which the network of hops first connects through the cell in
one, two and three independent directions, which is what the effective activation energy of a
conductor is read from.  The model predicts per hop; this is the step that turns those
predictions into a number per structure.

**Disorder (plan item M2).**  A conductor's Li sublattice is usually partly occupied, and the
sites of its framework may be shared by two species.  A structure that carries its site species
(`cmfqsar.periodic.site_species`, read from a CIF by
`cmfqsar.periodic.read_disordered`) gives occupancy-weighted fields, which is the exact
mean field of the configurational ensemble, so the environment of a hop sees the *other* Li
sites at their occupancy, as a migrating ion does on average.  Every site that holds the mobile
species at any occupancy is a site a hop can start from or end at, and the hop records both
occupancies.  The bond-valence landscape needs an ordered framework, so it is built from the
sites not held by the mobile species, each with its majority species, which is exact when only
the Li sublattice is disordered and an approximation when the framework is.

**The bottleneck.**  `bottleneck` finds the point of least free space on the straight
path, in the spirit of the free-space test of `cmfqsar.complementarity`, and names the
atoms that form it; it is reported with every hop and is a check that the environment the model
sees is the one a chemist would describe.

### `class Hop  (dataclass)`

One migration path: site `a` of a structure to site `b` in the image `offset`.

Attributes:

- `structure: int`
- `a: int`
- `b: int`
- `offset: Tuple[int, int, int]`
- `start: np.ndarray`
- `vector: np.ndarray`
- `shift: np.ndarray` = `_field(default_factory=lambda: np.zeros(3))`
- `y: float` = `float('nan')`
- `y_reverse: float` = `float('nan')`
- `site_a: float` = `float('nan')`
- `site_b: float` = `float('nan')`
- `saddle: float` = `float('nan')`
- `bottleneck: float` = `float('nan')`
- `bottleneck_at: float` = `float('nan')`
- `bottleneck_atoms: List[str]` = `_field(default_factory=list)`
- `representative: int` = `-1`
- `multiplicity: int` = `1`
- `name: str` = `''`
- `occupancy_a: float` = `1.0`
- `occupancy_b: float` = `1.0`

#### `Hop.length` -> float  (property)

#### `Hop.midpoint` -> np.ndarray  (property)

#### `Hop.end` -> np.ndarray  (property)

#### `Hop.reversed() -> 'Hop'`

The same path the other way: b in its image to a in the home cell.

#### `Hop.label(names: Optional[Sequence[str]]=None) -> str`

### `class HopSet  (dataclass)`

Structures, their hops, and the index that turns the hops into a per-sample target.

Attributes:

- `structures: list`
- `names: List[str]`
- `hops: List[Hop]`
- `mobile: str` = `'Li'`

#### `HopSet.build(structures: Sequence, names: Optional[Sequence[str]]=None, mobile: str='Li', d_min: float=HOP_MIN, d_max: float=HOP_MAX) -> 'HopSet'`  (class method)

#### `HopSet.of_structure(k: int) -> List[int]`

#### `HopSet.representatives() -> List[int]`

#### `HopSet.crystals(fieldset: FieldSet) -> List[CrystalFields]`

#### `HopSet.environments(crystals: Sequence[CrystalFields], rows: Optional[Sequence[int]]=None, rho: float=DEFAULT_RHO, floor: float=ENVELOPE_FLOOR, r_max: Optional[float]=None, posed: bool=False, n_poses: int=DEFAULT_POSES, progress=None) -> List[MoleculeFields]`

The environment of every hop in `rows` (all of them by default), posed if asked.

#### `HopSet.annotate(crystals: Sequence[CrystalFields], rho: float=DEFAULT_RHO, floor: float=ENVELOPE_FLOOR, r_max: Optional[float]=None) -> List[int]`

Bottleneck of every hop and the symmetry reduction; returns the representatives.

#### `HopSet.targets(rows: Optional[Sequence[int]]=None, which: str='forward', name: str='barrier') -> AtomTargets`

The hops in `rows` with a finite label, as an index whose "molecule" is the structure.

The folds of the result hold out whole structures, and every tool of the local modules
that takes an `cmfqsar.atomtargets.AtomTargets` takes this one.  `atom` is the
index of the hop in `hops`.

#### `HopSet.describe() -> str`

### `class HopModel  (dataclass)`

A kernel ridge model on hop environments with the rotation-averaged overlap kernel.

The environments are kept in their hop frames; the kernel is the closed-form average over
the rotation about the hop (`cylindrical_blocks`), one block per field, each divided
by its diagonal so that no field dominates by its units, and summed with the mixing
coefficients of the field set.  `profile` reads the co-field as a function of the
position along the hop and the distance from its axis, which is what the averaged kernel
makes the co-field depend on.

Attributes:

- `fieldset: FieldSet`
- `train: List[MoleculeFields]`
- `K: np.ndarray`
- `diag: np.ndarray`
- `lam: float` = `0.001`
- `coef: Optional[np.ndarray]` = `None`
- `bias: float` = `0.0`
- `y: Optional[np.ndarray]` = `None`
- `K_fields: Optional[np.ndarray]` = `None`

#### `HopModel.build(envs: Sequence[MoleculeFields], fieldset: FieldSet, keep_blocks: bool=False, progress=None, n_jobs: int=1) -> 'HopModel'`  (class method)

#### `HopModel.specs`  (property)

#### `HopModel.fit(y: Sequence[float], lam: Optional[float]=None) -> 'HopModel'`

#### `HopModel.kernel_to(envs: Sequence[MoleculeFields], progress=None) -> np.ndarray`

The (n_test, n_train) joint kernel of new environments against the training ones.

#### `HopModel.predict(envs: Sequence[MoleculeFields], progress=None) -> np.ndarray`

#### `HopModel.profile(f: int, zs: Sequence[float], rs: Sequence[float], sigma: Optional[float]=None) -> Tuple[np.ndarray, Optional[np.ndarray]]`

The co-field of field `f` at every (z, r), as a (len(zs), len(rs)) table, with its
standard deviation under noise of level `sigma` in the barriers when that is given.

The co-field is `sum_j a_j s_fj X_fj(rho, z)` with `s_fj` the normalisation and
`X_fj` the field of environment j averaged about the hop axis; its error follows the
dual coefficients, whose covariance under independent noise of level sigma is
`sigma^2 (K + lam I)^-2`.

#### `HopModel.atom_contributions(env: MoleculeFields) -> np.ndarray`

What every atom of an environment adds to its prediction, the bias apart.

The kernel is a sum over the atoms of the environment, so the prediction splits over
them exactly, as the atom contributions of the overlap model do.

### `point_environment(crystal: CrystalFields, centre: Sequence[float], rho: float=DEFAULT_RHO, floor: float=ENVELOPE_FLOOR, r_max: Optional[float]=None, exclude: Sequence[Tuple[int, Sequence[int]]]=(), name: str='') -> MoleculeFields`

The fields of a periodic structure seen from a point, under a Gaussian envelope.

Every atom of the cell and every periodic image of it within the reach of the envelope is
collected, with the coordinates measured from `centre` and the weights multiplied by
`exp(-rho d^2)`, as `cmfqsar.localenv.environment` does for a molecule.  The reach is
where the envelope falls below `floor`, or `r_max` when that is given and smaller.

`exclude` lists (atom index, lattice offset) pairs to leave out: for a hop these are the two
positions of the migrating ion.  Other images of the same atom are other ions and stay.

The first atom of the result is a weightless marker at the centre, so that the local modules'
convention that the centre is atom 0 holds; `member_atom_map` holds the index in the cell
of every other atom (-1 for the marker), so that a per-atom quantity can be put back on the
cell.  `bonds` is empty: a framework is described by distances, not by a bond graph.

### `enumerate_hops(atoms, mobile: str='Li', d_min: float=HOP_MIN, d_max: float=HOP_MAX, structure: int=0) -> List[Hop]`

Every pair of sites of the mobile species between `d_min` and `d_max` apart, both ways.

Periodic images count, so a hop from an atom to its own image in a small cell is a hop.  Each
unordered pair gives two hops, because the barrier has a direction.

### `mobile_occupancy(atoms, mobile: str='Li') -> np.ndarray`

The occupancy of the mobile species on every site: 1 or 0 in an ordered structure, the
site's share of that species where sites are partly occupied or shared (plan item M2).

### `hop_environment(crystal: CrystalFields, hop: Hop, rho: float=DEFAULT_RHO, floor: float=ENVELOPE_FLOOR, r_max: Optional[float]=None, name: str='') -> MoleculeFields`

The environment of a hop: the cell seen from the midpoint, without the migrating ion.

The two sites of the hop are left out whole.  On a site the mobile species shares with
another (a La/Li site of a perovskite), that other species leaves with it, which is the one
approximation of the disordered case; such sites are rare and the hop records their
occupancies.

### `hop_axis_frame(vector: Sequence[float], reference: Optional[Sequence[float]]=None) -> np.ndarray`

The rotation whose rows are x, y, z with z along the hop and x towards `reference`.

`reference` is a direction whose component perpendicular to the hop fixes x; without one, or
when it is parallel to the hop, x is taken as the lattice-independent choice of the axis of
the global frame least aligned with z.

### `hop_frames(env: MoleculeFields, vector: Sequence[float], n_poses: int=DEFAULT_POSES) -> List[np.ndarray]`

`n_poses` frames of a hop environment: z along the hop, x turned about it in equal steps.

The starting angle points x at the framework atom nearest to the midpoint (the centre is
atom 0 and is skipped), so that the set is fixed by the geometry and not by the coordinates
the file happened to use.

### `posed_hop(env: MoleculeFields, vector: Sequence[float], n_poses: int=DEFAULT_POSES, name: str='') -> MoleculeFields`

The hop environment in its frame, averaged over the rotation about the hop axis.

### `bottleneck(env: MoleculeFields, vector: Sequence[float], n_points: int=41, margin: float=0.3) -> Tuple[float, float, List[str]]`

The point of least free space on the straight path, as (distance, position, atoms).

The path runs from -vector/2 to +vector/2 through the centre of `env`; at every point the
distance to the nearest framework atom is taken, and the smallest of those is the
bottleneck.  `position` is the fraction of the path at which it occurs (0.5 is the midpoint)
and `atoms` are the symbols of the atoms within `margin` angstrom of that nearest distance,
which is the ring the ion has to pass through.

### `fingerprint(env: MoleculeFields, vector: Sequence[float], reach: float=5.0, decimals: int=2, occupancy: Optional[np.ndarray]=None) -> Tuple`

What a hop looks like up to a rotation about its axis: the sorted (symbol, distance from
the midpoint, position along the hop, occupancy) of the atoms within `reach`.

Two hops related by a symmetry of the structure have the same fingerprint; a hop and its
reverse have different ones unless the two sites are equivalent, which is as it should be,
since their barriers then differ.  `occupancy` is the site occupancy of every atom of the
environment (the marker first), when the structure has partial occupancies.

### `environment_occupancy(env: MoleculeFields, crystal: CrystalFields) -> Optional[np.ndarray]`

The site occupancy of every atom of an environment, from the cell it was cut from, or
None when the structure is ordered.

### `reduce_by_symmetry(hops: Sequence[Hop], envs: Sequence[MoleculeFields], crystals: Optional[Sequence[CrystalFields]]=None) -> List[int]`

Mark equivalent hops of one structure and return the indices of the representatives.

Equivalence is equality of `fingerprint`.  Every hop gets its `representative` and the
representative its `multiplicity`, so that a prediction for the representative can be spread
back to all the hops it stands for, which the percolation step needs.  With `crystals` the
occupancies enter the fingerprint, so that two hops that differ only in how full their
surroundings are stay apart.

### `bvse_calculator(atoms, mobile: str='Li', oxidation: Optional[Dict[str, float]]=None, resolution: float=0.25, r_cut: float=10.0)`

A `bvlain` calculator with the landscape of the mobile ion over the framework.

The oxidation states come from `oxidation`, else from the structure's `oxi_states` array,
else from the composition (`cmfqsar.periodic.oxidation_states`).  The framework is the
structure without the mobile species, which is how the bond-valence method is defined.

### `bvse_energy(calc, points: np.ndarray, f: float=0.74) -> np.ndarray`

The bond-valence site energy of the mobile ion at Cartesian `points`, in eV.

This evaluates the same Morse plus screened-Coulomb sum `bvlain` puts on its grid, at
points of the caller's choosing: the points are wrapped into the cell and placed in the
middle of the calculator's supercell, whose neighbour search then applies.

### `path_barrier(energy: Callable[[np.ndarray], np.ndarray], start: np.ndarray, vector: np.ndarray, spacing: float=0.2, pad: float=2.5, site_search: float=SITE_SEARCH) -> Tuple[float, float, float]`

(saddle, energy at the start site, energy at the end site) on a landscape.

A box is laid around the path -- `pad` angstrom beyond each site along it and `pad` to the
sides -- and sampled at `spacing`; the energy of a site is the lowest value within
`site_search` of it, so that a site that the landscape puts a little off the given position
is still that site.  The saddle is the lowest level at which the two minima are connected
through the box, found by bisection on the level with a connected-components labelling at
each step: a minimax path, which is what a migration barrier on a grid is.

### `bvse_barriers(hopset: HopSet, rows: Optional[Sequence[int]]=None, oxidation: Optional[Dict[str, float]]=None, resolution: float=0.25, spacing: float=0.2, pad: float=2.5, progress=None, spread: bool=True, grid_percolation: bool=True) -> Dict[str, Dict[str, float]]`

Label the hops in `rows` with their bond-valence barriers; returns the percolation
energies of every structure's landscape as `bvlain` reports them, by structure name.

With `spread`, a hop that was not labelled itself but has a labelled representative takes
its representative's values, and a hop whose reverse was labelled takes the reverse values.
`grid_percolation` asks `bvlain` for the percolation energies of its own grid, which the
hop network of `structure_percolation` can be compared with; it can take minutes on a
landscape with many pockets, so it can be switched off.

### `spread_labels(hopset: HopSet) -> int`

Give every unlabelled hop the label of its representative, or of its reverse; returns
how many were filled.

### `read_hop_labels(path: str, hopset: HopSet, column: str='barrier', reverse_column: str='barrier_reverse') -> int`

Read barriers from a table onto the hops; returns how many hops were labelled.

The table has the columns `structure` (a name of the set, or its index), `a` and `b`
(atom indices in the structure), optionally `offset_x`, `offset_y` and `offset_z` (the
image of `b`; the nearest image is taken without them), and the barrier column.  A row
naming a hop the set does not contain is an error, because a silent miss would be a label
thrown away.

### `write_hops(path: str, hopset: HopSet, predicted: Optional[Dict[int, float]]=None) -> int`

Write every hop with what is known about it; `predicted` maps hop index to prediction.

### `percolation_thresholds(n_sites: int, hops: Sequence[Hop], levels: Sequence[float]) -> Dict[str, float]`

The lowest level at which the hops connect through the cell in 1, 2 and 3 directions.

`levels` is one number per hop, the energy a path crossing it has to reach -- the saddle
energy relative to the lowest site, or a barrier -- and a hop in either direction carries the
same connection.  Hops are added in order of increasing level, and the result is
`{"E_1D", "E_2D", "E_3D"}` with *infinity* where the network never wraps in that many
independent directions.  The sites are the atoms of the mobile species, indexed as the hops
index them.

### `structure_percolation(hopset: HopSet, k: int, values: Optional[Dict[int, float]]=None, relative: bool=True) -> Dict[str, float]`

Percolation thresholds of structure `k` from the hops' saddle energies, or from `values`.

`values` maps a hop index to the energy a path crossing it has to reach; without it the
bond-valence saddles are used.  With `relative` the thresholds are measured from the lowest
site energy of the structure, which is what a percolation barrier means.

### `saddle_from_barrier(hopset: HopSet, predicted: Dict[int, float]) -> Dict[int, float]`

Turn predicted forward barriers into saddle energies: site energy plus barrier, where
the site energy is known, else the barrier itself.

### `hop_types(envs: Sequence[MoleculeFields], min_atoms: int=20) -> List[str]`

The atom types of the environments worth a column of the count reference.

### `hop_count_features(envs: Sequence[MoleculeFields], types: Sequence[str], edges: Optional[Sequence[float]]=None) -> np.ndarray`

The atom-centred count of `cmfqsar.localenv.shell_counts` about the midpoint of
every hop, which is the reference every hop model has to beat.

### `profile_points(zs: Sequence[float], rs: Sequence[float], n_angles: int=DEFAULT_POSES) -> np.ndarray`

Points of a (z, r) profile in the hop frame: for every z along the hop and every
radius r from the axis, `n_angles` points around the axis, which are averaged.

### `cofield_profile(model, f: int, zs: Sequence[float], rs: Sequence[float], n_angles: int=DEFAULT_POSES, include_scale: bool=True) -> np.ndarray`

The co-field of field `f` as a (len(zs), len(rs)) table in the hop frame.

The posed kernel averages over the rotation about the hop, so the co-field is a function of
the position along the hop and the distance from its axis; this reads it at those places and
averages the residual dependence on the angle that the finite number of poses leaves.

### `format_profile(table: np.ndarray, zs: Sequence[float], rs: Sequence[float], unit: str='') -> str`

A (z, r) table as text: rows along the hop, columns away from its axis.

### `cylindrical_blocks(P: Sequence[MoleculeFields], Q: Sequence[MoleculeFields], specs, chunk: int=40000, progress=None, n_jobs: int=1) -> np.ndarray`

The rotation-averaged overlap kernels between the environments of P and of Q: (F, nP, nQ).

Every environment is in its hop frame, z along the hop.  When `P` is `Q` only the upper
triangle is computed and mirrored.  The geometric part of the kernel is computed once per
group of fields that share a width, and the weights are applied per field.  With `n_jobs`
above one the rows are split over that many processes (joblib), which pays from a few
hundred environments on; the result does not depend on it.

### `cross_validate_hop_model(model: HopModel, targets: AtomTargets, lams: Optional[Sequence[float]]=None, folds: int=10, seed: Optional[int]=0, name: str='')`

The ridge search of `cmfqsar.localenv.choose_lam` on a hop model's kernel.

### `framed_environments(hopset: HopSet, crystals: Sequence[CrystalFields], rows: Optional[Sequence[int]]=None, rho: float=DEFAULT_RHO, floor: float=ENVELOPE_FLOOR, r_max: Optional[float]=None, progress=None) -> List[MoleculeFields]`

The environment of every hop in `rows`, turned into its hop frame (one pose, z along the
hop), which is what the rotation-averaged kernel takes.

## cmfqsar.increments

Source: `cmfqsar/increments.py`

Atomic increments of the original CMF implementation (Baskin & Zhokhova 2013).

The R scripts that produced the models of the 2013 paper parameterised the
hydrophobic field and the two hydrogen-bond fields with in-house one-atom
fragment increments obtained by regression of experimental logP values and of
Abraham's hydrogen-bond acidity (A) and basicity (B) solute constants (the
scripts also carry increments for Abraham's E and S constants).  The increments
are keyed by the Sybyl atom type and by the number of hydrogen atoms attached
to the atom.  Hydrogen atoms themselves carry no increments, i.e. the donor
field is centred on the heavy atom bearing the hydrogen.

This module is a faithful transcription of `cmf-params.R`; the numbers must
not be "improved" here, because their purpose is to reproduce the published
models.  Use `param_scheme="tripos"` of `cmfqsar.fields.FieldSet` to
select this parameterisation.

### `normalize_sybyl_type(syb: str) -> str`

Canonical spelling of a Sybyl atom type (`S.o2`/`S.O2` -> `S.o2`).

### `tripos_increments(syb: str, n_h: int) -> Dict[str, float]`

Increments of one atom of Sybyl type `syb` carrying `n_h` hydrogen atoms.

Returns a dict with keys `hydrophobic` (logP increment), `hbd`
(Abraham A, hydrogen-bond donor), `hba` (Abraham B, acceptor),
`abraham_e` and `abraham_s`.

## cmfqsar.invariant

Source: `cmfqsar/invariant.py`

Alignment-free but geometry-aware kernels: autocorrelation of the molecular fields.

Everything else in this program compares two molecules by the overlap integral of their fields,
which needs them to share a frame.  That is the strength of the approach and its one hard
requirement, and the heterogeneous-series work shows what it costs when the frame is in doubt:
the alignment-free comparator there is the count kernel, which is pose-free only because it has
thrown the geometry away entirely.

This module fills the gap between those two.  For every pair of field types it takes the distance
distribution of the field weights inside one molecule,

    Phi_fg(d) = sum_l sum_m w_lf w_mg exp(-(|r_l - r_m| - d)^2 / 2 sigma^2),

evaluated on a grid of distances d_k.  A rigid motion moves every atom and leaves every
interatomic distance where it was, so the whole descriptor is invariant to rotation and
translation by construction -- no superposition, no reference molecule, no common frame -- while
still describing the three-dimensional arrangement: where a donor sits relative to an acceptor,
how far the positive charge is from the hydrophobic bulk.

The connection to the CMF kernel is exact at one point.  The overlap integral of two Gaussian
fields of the same molecule is

    K_fg(A, A) = (pi/alpha)^{3/2} sum_lm w_lf w_mg exp(-alpha |r_l - r_m|^2 / 4),

which is this descriptor's channel at d = 0 with sigma^2 = 2/alpha, up to the prefactor.  The
further channels resolve what that single number collapses: the overlap integral reports how much
the fields of a molecule coincide, the autocorrelation reports at which distances they do.

A distance distribution is the same for a molecule and its mirror image, so the two-point form
above cannot detect chirality.  The signed quantity that is not is the triple product of three
displacement vectors, and `chirality` adds it as a second block of channels.  Summing the
signed volume over every triple of atoms would cost O(N^3), but the determinant is trilinear, so
the sum factorises into the triple product of three field-weighted first moments and costs one
pass over the atoms instead.  It is off by default, so a model built before it existed is
unchanged; a test asserts that.

What is lost with the frame is the co-field: a map in space needs a space to live in.  The
decomposition that survives is the one in the descriptor's own terms -- which pair of field
types at which distance carries the weight -- and `channel_labels` names those channels so
a fitted model can be read that way.

The channels come in blocks, one per pair of field types, and every block is scaled to unit root
mean square before the blocks are taken together in one cosine-normalised dot product.  That is
the right default for a few fields and the wrong one for sixty: with one indicator field
per element the pairs across the elements outnumber the physico-chemical pairs a hundred to one
and, equally scaled, overwhelm them.  The overlap kernel has its mixing coefficients h_f for this;
the functions under "Mixing coefficients over the channel blocks" below give the invariant
kernels theirs -- one weight per block, set by the kind of block (within a group of fields, across
two groups) or by centred kernel alignment of the per-block kernels, as `model.align_mixing`
sets the h_f.

### `class AutocorrelationSpec  (dataclass)`

The distance grid the field autocorrelation is evaluated on.

Attributes:

- `d_max: float` = `12.0`
- `step: float` = `0.75`
- `sigma: float` = `0.75`
- `include_self: bool` = `True`
- `chirality: bool` = `False`
- `chirality_shells: int` = `4`

#### `AutocorrelationSpec.centres` -> np.ndarray  (property)

#### `AutocorrelationSpec.n_bins` -> int  (property)

#### `AutocorrelationSpec.shell_centres` -> np.ndarray  (property)

Where the radial windows of the chirality channels sit, measured from the centroid.

Radii from the centroid reach about half the longest interatomic distance, so the shells
span `d_max / 2` rather than `d_max`; one shell means no window at all.

#### `AutocorrelationSpec.shell_sigma` -> float  (property)

#### `AutocorrelationSpec.n_shells` -> int  (property)

### `class SurfaceSpec  (dataclass)`

The in-plane distance grid and the height grid of the surface autocorrelation.

A molecule on a surface has one direction that is not like the others: the normal.  Heights
above the surface are physical and stay as they are; what is arbitrary is where the molecule
sits in the plane and how it is turned about the normal.  The descriptor is the two-point
distribution of the field weights over the *in-plane* distance rho, resolved by the heights
of both atoms,

    Phi_fg(rho_k, z_i, z_j) = sum_l sum_m w_lf w_mg exp(-(rho_lm - rho_k)^2 / 2 sigma^2)
                                          exp(-(z_l - z_i)^2 / 2 sigma_z^2)
                                          exp(-(z_m - z_j)^2 / 2 sigma_z^2),

which is invariant to translations in the plane and rotations about the normal, and to
nothing else.  Averaged over the direction of rho it is, by the Wiener-Khinchin theorem,
the same information as the in-plane power spectrum `|F_f(k, z_i) F_g(k, z_j)|` of the fields:
a description of the molecule over the surface in reciprocal space in the plane and in
real space along the normal.  The surface is at z = 0 and the normal is +z, which is how
examples/uma_fe110_adsorption.py writes its poses.

Attributes:

- `d_max: float` = `12.0`
- `step: float` = `0.75`
- `sigma: float` = `0.75`
- `z_min: float` = `1.0`
- `z_max: float` = `6.0`
- `z_step: float` = `1.0`
- `z_sigma: float` = `1.0`
- `include_self: bool` = `True`

#### `SurfaceSpec.centres` -> np.ndarray  (property)

#### `SurfaceSpec.heights` -> np.ndarray  (property)

#### `SurfaceSpec.n_bins` -> int  (property)

Channels per pair of field types: distances times heights squared.

### `surface_autocorrelation(mol: MoleculeFields, spec: SurfaceSpec=SurfaceSpec(), pairs: Optional[Sequence[Tuple[int, int]]]=None) -> np.ndarray`

Surface autocorrelation of one molecule, flattened: (n_pairs * n_rho * n_z * n_z,).

For a pair of different field types both orders are summed (f at z_i with g at z_j, and
g at z_i with f at z_j), as the three-dimensional descriptor counts a pair both ways round.

### `surface_block_sizes(spec: SurfaceSpec, n_fields: int) -> List[int]`

### `surface_channel_labels(fieldset: FieldSet, spec: SurfaceSpec) -> List[str]`

One label per feature: the field types, the in-plane distance and the two heights.

### `surface_feature_matrix(mols: Sequence[MoleculeFields], spec: SurfaceSpec=SurfaceSpec(), n_fields: Optional[int]=None, progress=None) -> np.ndarray`

Surface autocorrelation features of a set of molecules, one row each.

### `field_pairs(n_fields: int) -> List[Tuple[int, int]]`

The unordered pairs of field types, diagonal first: (0,0), (1,1), ..., then (0,1), ...

### `chirality_triples(n_fields: int) -> List[Tuple[int, int, int]]`

The ordered triples f < g < h of distinct field types.

The signed volume is antisymmetric in its three arguments, so it vanishes whenever two of
them are the same field and merely changes sign when two are swapped: the triples with
f < g < h are all the independent ones there are.

### `block_sizes(spec: AutocorrelationSpec, n_fields: int) -> List[int]`

How many columns each channel block occupies, in the order the features are laid out.

The pair blocks all have `n_bins` columns and the chirality blocks `n_shells`, so the blocks
are no longer of one size once chirality is on and the scaling has to be told where they are.

### `expanded_field_labels(fieldset: FieldSet) -> List[str]`

The field labels this module works with: a probe field becomes one label per term.

An ordinary field puts one number on every atom, which is what the autocorrelation above
needs.  A probe field does not: it puts a short sum of Gaussians of different widths on every
atom (see `cmfqsar.probes`), so its atom weight column is empty and an autocorrelation
taken from that column alone is identically zero.  The expansion below treats each term of
the sum as a field of its own, which is exact -- the field is the sum of those terms -- and
gives the kernel the probe's chemistry as several channels instead of none.

### `expanded_field_count(fieldset: FieldSet) -> int`

How many field columns `expanded_weights` produces.

### `expanded_weights(mol: MoleculeFields, fieldset: FieldSet) -> np.ndarray`

`(n_atoms, n_expanded)`: the atom weights with every probe field spread over its terms.

### `channel_labels(fieldset: FieldSet, spec: AutocorrelationSpec) -> List[str]`

One label per feature: which field types, and at which distance or radius.

### `chirality_labels(fieldset: FieldSet, spec: AutocorrelationSpec) -> List[str]`

One label per chirality feature: which three field types, in which radial shell.

### `chirality(mol: MoleculeFields, spec: AutocorrelationSpec=AutocorrelationSpec(), triples: Optional[Sequence[Tuple[int, int, int]]]=None) -> np.ndarray`

The three-point signed channels of one molecule, flattened: (n_triples * n_shells,).

A distance distribution is the same for a molecule and its mirror image, so everything above
cannot detect chirality.  The signed quantity that is not is the triple product of three
displacement vectors, and the obvious way to use it -- summing the signed volume over every
triple of atoms -- would cost O(N^3) per field triple.  It does not have to.  The determinant
is trilinear, so the sum over triples factorises exactly:

    sum_{l,m,n} w_lf w_mg w_nh  det[r_l - c, r_m - c, r_n - c]
        =  det[ A_f, A_g, A_h ],        A_f = sum_l w_lf (r_l - c)

with `c` the centroid of the atoms.  So the whole block costs one pass over the atoms, and
each channel is the signed volume spanned by three field-weighted vectors -- the first moment
of each field about the centroid.  Under a proper rotation all three are rotated and the
determinant is unchanged; under a reflection the determinant changes sign; a translation
moves the centroid with the atoms and cancels.  That is exactly the behaviour chirality
needs, and it is why a planar molecule gives zero in every channel: three vectors in a plane
span no volume, and a planar molecule is achiral.

`chirality_shells` resolves the invariant over radial windows about the centroid, since one
number per field triple describes the handedness of the molecule as a whole and cannot say
at which radius it is located.  The windows depend on `|r_l - c|`, which is itself reflection-
invariant, so every channel keeps the sign behaviour above.

### `autocorrelation(mol: MoleculeFields, spec: AutocorrelationSpec=AutocorrelationSpec(), pairs: Optional[Sequence[Tuple[int, int]]]=None, weights: Optional[np.ndarray]=None) -> np.ndarray`

Field autocorrelation of one molecule, flattened: (n_pairs * n_bins,).

The result depends on the molecule's conformation and on nothing else: not on where it sits,
not on how it is turned, not on any other molecule.

`weights` replaces the molecule's own atom weights, which is how a probe field enters: see
`expanded_weights`.

### `feature_matrix(mols: Sequence[MoleculeFields], spec: AutocorrelationSpec=AutocorrelationSpec(), n_fields: Optional[int]=None, progress=None, fieldset: Optional[FieldSet]=None) -> np.ndarray`

Autocorrelation features of a set of molecules, one row each.

`fieldset` is required when the fields include a probe field, because a probe field carries
its weights per Gaussian term rather than per atom and has to be expanded first
(`expanded_weights`).  Without it the probe field's channels would all be zero, so this
function refuses instead of returning a kernel that silently says every molecule is alike.

### `channel_scale(train: np.ndarray, n_bins: int, blocks: Optional[Sequence[int]]=None) -> np.ndarray`

One scale per channel block, so that no block dominates by its units alone.

The electrostatic field carries charges and the steric one carries ones, so their
autocorrelations differ by orders of magnitude; each block is divided by its root mean square
over the training molecules, which is what `scale_fields` does for the kernels.  The signed
chirality channels need this more than anything else does, since a volume carries the cube of
a length.  `blocks` gives the column counts explicitly, which is required once they are not
all `n_bins` wide; without it the blocks are taken to be uniform, as they were before the
chirality channels existed.

### `autocorrelation_kernel(train: np.ndarray, test: Optional[np.ndarray]=None, scale: Optional[np.ndarray]=None, n_bins: int=1, normalize: bool=True, blocks: Optional[Sequence[int]]=None) -> np.ndarray`

Cosine-normalised linear kernel on the autocorrelation features.

The normalisation is applied to the joint kernel, not block by block: normalising each pair
of field types on its own would divide out the very magnitudes that distinguish molecules,
the same source of error that affects the alignment-free count kernel when it is normalised
field by field.

### `block_fields(n_fields: int, chirality: bool=False) -> List[Tuple[int, ...]]`

The field types of every channel block, in the order the features are laid out: the pairs
of `field_pairs`, then, with chirality, the triples of `chirality_triples`.  The
surface and periodic descriptors have the pair blocks only.

### `default_block_groups(names: Iterable[str]) -> Dict[str, str]`

The grouping of the fields the block weights use unless told otherwise: the continuous
indicator fields (`cif:<type>`, one per element or atom type) in one group, every other
field -- the physico-chemical and property fields -- in the other.

### `group_key(groups: Iterable[str]) -> str`

The name of a kind of block from the groups of its fields: the group's name for a block
within one group, the names joined by `*` in alphabetical order for a block across groups,
so that `"a*b"` and `"b*a"` are the same key.

### `block_group_keys(names: Sequence[str], groups: Optional[Mapping[str, str]]=None, chirality: bool=False) -> List[str]`

The kind of every block of a field set with these field names, in layout order.

`groups` maps field names to group names; a field it does not name falls back to the
default grouping of `default_block_groups`.

### `group_block_weights(keys: Sequence[str], weights: Union[str, Mapping[str, float]]) -> np.ndarray`

One weight per block, by the kind of block.

`weights` is either a mapping from kinds (keys as `group_key` writes them; either
order of the groups of a cross key is accepted) to weights, with 1 for every kind it does not
name, or the word `"balanced"`: every kind present gets `1 / sqrt(its number of blocks)`,
so that the kinds contribute equally to the squared norm of a molecule's channel vector --
the pairs within the four physico-chemical fields as much as the fifteen hundred pairs across
the element indicators -- however many blocks each kind has.  The weights multiply the scaled
features, so a block's share of the kernel goes as the square of its weight, and a weight of
zero removes the block exactly: it enters neither the dot products nor the norms.

### `expand_block_weights(w: Sequence[float], blocks: Sequence[int]) -> np.ndarray`

The per-block weights spread over the columns of their blocks: one multiplier per feature,
to be applied with (or folded into) the channel scale.

### `align_block_weights(X: np.ndarray, y: np.ndarray, blocks: Sequence[int], mode: str='align', rho: float=1.0, max_chunk_cols: int=2048) -> np.ndarray`

Block weights by centred kernel alignment: `model.align_mixing` for the channel
blocks of an invariant kernel.

`X` holds the scaled channel features of the training molecules (after
`channel_scale`), `blocks` the column count of every block.  Every block carries a
linear kernel `K_b = X_b X_b^T`; its centred form is `K_b^c = X_b^c X_b^c^T` with the
columns centred over the molecules, so everything the alignment needs is in the feature
space and the kernels themselves are never formed: `<K_b^c, y^c y^c^T>_F = ||X_b^c^T y^c||^2`
and `||K_b^c||_F = ||X_b^c^T X_b^c||_F`.

  `mode="align"`  weights every block by its own alignment,
    `h_b ~ <K_b^c, y^c y^c^T> / ||K_b^c||_F` (Cristianini et al. 2001);
  `mode="alignf"` maximises the alignment of the combination,
    `min_{h >= 0} || sum_b h_b K_b^c - y^c y^c^T ||_F` (Cortes et al. 2012), with the
    Frobenius Gram matrix of the block kernels formed block pair by block pair and the
    solution shrunk towards equal weights with strength `rho`, as for the fields.

The kernel weights `h_b` are rescaled so that the mean diagonal of the joint kernel is
what it was with every `h_b = 1`, and the returned weights are their square roots, since
they multiply the features and the kernel sees them squared.  A block that is constant over
the training molecules -- a zero block, or one that does not vary -- gets no weight.

### `block_weight_summary(keys: Sequence[str], w: Sequence[float]) -> List[Dict[str, float]]`

The weights by kind of block: how many blocks of each kind there are and the mean,
smallest and largest weight among them, plus the kind's share of the squared norm of a
channel vector whose blocks all have unit root mean square (`sum w^2` over the kind,
normalised over the kinds).

### `channel_coefficients(train: np.ndarray, scale: np.ndarray, dual: Sequence[float], normalize: bool=True) -> np.ndarray`

The model as one vector in channel space.

A kernel ridge or support-vector model on this kernel predicts

    y(x) = sum_j a_j K(x, x_j) + b,

and the kernel is a dot product of the scaled, length-normalised channel rows, so the sum
over the training molecules collapses into one vector:

    c = sum_j a_j (s * F_j) / |s * F_j|      and      y(x) - b = u(x) . c

with `u(x)` the scaled row of the molecule being predicted, divided by its own length.
Everything the model has to say about a new molecule is that one dot product, which is what
makes the decomposition below exact rather than an attribution method.

### `atom_contributions(mol: MoleculeFields, coefficients: Sequence[float], scale: Optional[Sequence[float]]=None, spec: AutocorrelationSpec=AutocorrelationSpec(), pairs: Optional[Sequence[Tuple[int, int]]]=None, normalize: bool=True, per_pair: bool=False) -> np.ndarray`

Split what the invariant model predicts for one molecule over its atoms (item SC6).

Returns (n_atoms,), or (n_pairs, n_atoms) with `per_pair`.  The numbers sum to the prediction
without the bias, exactly, so an atom's number is its share of the predicted value and not a
score of its own.

**Why it is exact.** Each channel of the autocorrelation is a double sum over atom pairs,

    F[(f,g), k] = sum over atoms a, b of  W[a,f] B[a,b,k] W[b,g],

and the prediction is a fixed linear functional of those channels
(`channel_coefficients`).  A pair (a, b) is therefore one term of the prediction, and
the only choice to make is how to divide a pair between its two atoms; it is divided equally,
which is the only split that treats the two the same way.  A self term (a = a) is one atom's
alone.  Nothing is left over: the atom numbers add up to the prediction minus the bias.

`coefficients` is the vector `channel_coefficients` returns, `scale` what every channel
was multiplied by before the kernel was taken, and `normalize` says whether the kernel was
cosine-normalised, which divides the molecule's row by its own length.

### `pair_contributions(mol: MoleculeFields, coefficients: Sequence[float], scale: Optional[Sequence[float]]=None, spec: AutocorrelationSpec=AutocorrelationSpec(), fieldset: Optional[FieldSet]=None, normalize: bool=True) -> List[Tuple[str, float]]`

The same split taken over the channel blocks instead of the atoms.

One number per pair of field types -- "steric with steric", "electrostatic with hba" -- which
says which kind of channel the prediction came from.  The numbers add up to the same total as
`atom_contributions`.

## cmfqsar.kcca

Source: `cmfqsar/kcca.py`

Kernel canonical correlation analysis: what one view of the molecules sees that another does
not (plan item F6).

A *view* is one description of the same molecules, which in this program means one kernel over
them.  The program builds several: the five-field CMF kernel, the quantum fields, the ECFP4
Tanimoto kernel, the pocket complementarity, the same fields over a conformer ensemble.  No tool
here says how much two of them say the same thing, and, more useful, **which part of the field
is the part a second view does not contain**.

Canonical correlation analysis (CCA) answers that question.  Given two descriptions of the same
objects it finds the pair of directions, one in each description, whose projections correlate
most strongly; then the next such pair, uncorrelated with the first; and so on.  The kernel form
(KCCA) does it with kernels in place of coordinates, which is what CMF needs because its
description of a molecule has no coordinate vector.  With centred kernels `K` and `L` on the
same `n` molecules and dual directions `a` and `b` (one coefficient per molecule, like the
coefficients of a fitted model), the pairs solve the generalised eigenproblem

    [ 0    K L ] [a]         [ K^2 + c_K K      0      ] [a]
    [ L K   0  ] [b] = rho   [     0       L^2 + c_L L ] [b]

where `rho` is the canonical correlation and `c_K`, `c_L` are ridges.  **The ridge is not
an option.**  With `c = 0` and invertible kernels every canonical correlation is exactly 1 and
the directions are arbitrary -- any direction of one view can be matched exactly by some
direction of the other, because both have as many directions as there are molecules.  That is the
classical failure of KCCA, and it is why nothing here is judged by its training correlations:
the ridge is chosen by the correlations on **held-out** molecules, and the correlations are
compared with the ones a **shuffled pairing** of the two views produces.

The eigenproblem is solved through the spectra of the two kernels.  With `K = U diag(k) U^T`
and `L = V diag(l) V^T` the substitution `a = U diag(k (k + c_K))^(-1/2) p` turns it into
the singular value decomposition of

    M = diag( sqrt(k / (k + c_K)) )  U^T V  diag( sqrt(l / (l + c_L)) )

whose singular values are the canonical correlations and whose singular vectors give the
directions.  The factor `sqrt(k / (k + c))` is what the ridge does: a direction along which
the molecules barely vary (small `k`) is shrunk towards nothing instead of being inflated to
unit variance, and `||K a_i||^2`, which is 1 without a ridge, says how much of each direction
the ridge left.  That number is reported for every component as its **variance share**.

Two properties make this a CMF tool rather than a general one.  A canonical direction on the
field side is a vector of dual coefficients, the same object as a fitted model's, so
`component_model` turns a component into a model whose co-field
(`cmfqsar.model.CMFModel.cofield_coefficients`) is the picture of that direction, with no
new interpretation code and with rule 2 of the plan kept: the direction is linear in the field.
And when both views are field kernels over the same fields, the two co-fields of one component
are functions on one space and `component_cosines` compares them by the cosine of plan
item F3.

**The degenerate case, stated first because it is the obvious misuse.**  With the activity as the
second view that view has one dimension, and the first canonical direction is the kernel ridge
regression of the activity on the first view at ridge `c_K` -- `a = (K + c_K I)^-1 y`, up to
a scale factor.  "KCCA-QSAR" in that sense is kernel ridge regression under another name and
predicts nothing new.  The test suite uses the identity as a correctness check and nothing else.

**What the field sees that the other view does not.**  A direction of the field view counts
as *contained* in the other view when that view explains more than half of its variance on
held-out molecules (`contained_directions`).  The field kernel is then split in two:
`projected_kernel` restricts it to the span of the contained directions, and the
difference is the kernel of what is left, the part of the field that is the field's own.  The
split is a projection and not a regression on the canonical variates, because the variates are
whitened by construction and a model on them has lost the variance along each direction, which
is the prior kernel ridge regression rests on; the projected kernels keep it.  Kernel ridge
regression on the two parts, with the regularisation chosen inside the folds
(`activity_on_kernel`), answers the question the item asks, with the bootstrap interval of
the difference; and the model on the field's own part has dual coefficients
(`private_coefficients`), so where that part carries the activity is a co-field.

### `class KCCAFit  (dataclass)`

The canonical directions of two views on one set of molecules.

`a` and `b` hold one dual direction per column, in the order of `rho` (largest
first); `u = K_c a` and `v = L_c b` are the canonical variates of the training
molecules.  `variance_a[i] = ||K_c a_i||^2` lies in (0, 1] and says how much of the
direction the ridge left; without a ridge it is 1 for every direction.

Attributes:

- `rho: np.ndarray`
- `a: np.ndarray`
- `b: np.ndarray`
- `u: np.ndarray`
- `v: np.ndarray`
- `variance_a: np.ndarray`
- `variance_b: np.ndarray`
- `ridge: float`
- `ridge_a: float`
- `ridge_b: float`
- `rank_a: int`
- `rank_b: int`
- `col_mean_a: np.ndarray` = `_field(repr=False, default=None)`
- `col_mean_b: np.ndarray` = `_field(repr=False, default=None)`
- `grand_a: float` = `0.0`
- `grand_b: float` = `0.0`

#### `KCCAFit.n_components` -> int  (property)

#### `KCCAFit.project_a(K_test_train: np.ndarray) -> np.ndarray`

Canonical variates of new molecules in view A, from their kernel block to the training set.

#### `KCCAFit.project_b(L_test_train: np.ndarray) -> np.ndarray`

The same for view B.

#### `KCCAFit.shared_variance() -> np.ndarray`

Variance of each direction of view A that view B follows: `v_i rho_i^2`.

#### `KCCAFit.private_variance() -> np.ndarray`

Variance of each direction of view A that view B does not follow: `v_i (1 - rho_i^2)`.

### `class HeldOut  (dataclass)`

Canonical correlations measured on molecules the directions were not fitted on.

Attributes:

- `per_fold: np.ndarray`
- `ridge: float`

#### `HeldOut.mean` -> np.ndarray  (property)

The held-out correlation of each component, averaged over the folds.

#### `HeldOut.n_components` -> int  (property)

### `class RidgeChoice  (dataclass)`

The ridge the held-out correlations chose, and what every candidate scored.

Attributes:

- `ridge: float`
- `scores: Dict[float, float]`
- `components: int`

#### `RidgeChoice.describe() -> str`

### `class KCCAResult  (dataclass)`

The analysis of two views: the fit, what held out, what shuffling gave, and the gates.

Attributes:

- `fit: KCCAFit`
- `held_out: HeldOut`
- `ridge_choice: Optional[RidgeChoice]`
- `null: Optional[np.ndarray]`
- `components: int`
- `n: int`
- `contained: Optional[np.ndarray]` = `None`
- `q2_all: float` = `float('nan')`
- `q2_shared: float` = `float('nan')`
- `q2_private: float` = `float('nan')`
- `q2_difference: Tuple[float, float, float]` = `(float('nan'), float('nan'), float('nan'))`
- `ycv_all: Optional[np.ndarray]` = `None`
- `ycv_shared: Optional[np.ndarray]` = `None`
- `ycv_private: Optional[np.ndarray]` = `None`
- `private_coef: Optional[np.ndarray]` = `None`
- `private_lam: float` = `float('nan')`

#### `KCCAResult.null_quantile(q: float=0.95) -> np.ndarray`

The `q` quantile of the shuffled held-out correlation, per component.

#### `KCCAResult.p_values() -> np.ndarray`

Share of shuffles whose held-out correlation reached the observed one, per component.

#### `KCCAResult.real_components(q: float=0.95) -> int`

How many components hold out above the `q` quantile of the shuffled pairing.

#### `KCCAResult.n_contained() -> int`

#### `KCCAResult.summary() -> Dict[str, object]`

The numbers a benchmark keeps: ridge, held-out correlations, the counts and gate (ii).

#### `KCCAResult.rows(components: Optional[int]=None) -> List[Dict[str, float]]`

One record per leading component, for a table or a file.

#### `KCCAResult.describe() -> str`

### `centre(K: np.ndarray) -> np.ndarray`

`H K H` with `H = I - 11^T / n`: the feature space centred at the mean molecule.

### `centre_against(K_test_train: np.ndarray, K_train: np.ndarray) -> np.ndarray`

The block between new molecules and the training set, centred with the training means.

A new molecule is placed in the feature space the training set was centred in, so the
column means and the grand mean are the training set's, and only the row mean is its own.

### `kcca(K: np.ndarray, L: np.ndarray, ridge: float=0.1, rcond: float=RCOND, max_rank: Optional[int]=None) -> KCCAFit`

Regularised kernel canonical correlation analysis of two kernels on the same molecules.

`K` and `L` are the raw (uncentred) kernels; they are centred here.  `ridge` is relative:
each view's absolute ridge is `ridge` times the mean eigenvalue of its centred kernel, so
that the same number means the same thing for a kernel of overlap integrals and for a
Tanimoto coefficient.  `rcond` drops the eigenvalues that are round-off and `max_rank`, if
given, keeps only that many leading eigenvectors of each view (a cheap second regulariser,
not used by default).

### `held_out_correlations(K: np.ndarray, L: np.ndarray, ridge: float, folds: int=5, seed: Optional[int]=0, components: Optional[int]=None, rcond: float=RCOND, max_rank: Optional[int]=None) -> HeldOut`

Fit the directions on four fifths of the molecules, correlate the variates on the fifth.

This is the number KCCA has to be judged by.  The training correlations approach 1 as the
ridge goes to zero whatever the two views contain; the held-out correlation does not, and
falls when the directions were fitted to the particular molecules rather than to what the
views share.  The correlation is Pearson's, per fold, averaged over the folds.

### `choose_ridge(K: np.ndarray, L: np.ndarray, grid: Sequence[float]=RIDGE_GRID, folds: int=5, seed: Optional[int]=0, components: Optional[int]=None, rcond: float=RCOND) -> RidgeChoice`

The ridge with the highest mean held-out correlation over the pairs.

Over all pairs by default, so that a ridge too small to stop the deep components fitting
noise is told apart from one that lets them hold out; `components` restricts the mean to
the leading pairs.  Never the training correlation: that is highest at the smallest ridge of
any grid.

### `permutation_null(K: np.ndarray, L: np.ndarray, ridge: float, permutations: int=200, folds: int=5, seed: Optional[int]=0, components: Optional[int]=None, rcond: float=RCOND) -> np.ndarray`

Held-out correlations when the pairing between the two views is shuffled.

The molecules of view B are permuted, so every molecule's field is paired with another
molecule's fingerprint, and the analysis is repeated at the same ridge on the same folds.
What comes back, `permutations` rows of `components` values, is what the leading held-out
correlations look like when the two views share nothing but their marginal distributions.
A real correlation has to clear this; one that does not is the degeneracy of the method,
not chemistry.

### `contained_directions(fit: KCCAFit, held_out: HeldOut, null: Optional[np.ndarray]=None, share: float=CONTAINED_SHARE, quantile: float=0.95) -> np.ndarray`

Which directions of view A the other view contains: a boolean per component.

A direction counts as contained when the other view explains more than `share` of its
variance on held-out molecules (a held-out correlation above `sqrt(share)`) and, when the
shuffled pairing is given, that correlation is also above its `quantile`.  A component the
folds could not produce (the full fit has a few more than a fold's) is not contained.

### `projected_kernel(fit: KCCAFit, index: Sequence[int]) -> np.ndarray`

The centred kernel of view A restricted to the span of the directions `index`.

With `E = Phi^T A_S` the feature-space vectors of the directions, the projector onto their
span is `E (E^T E)^-1 E^T` and the kernel of the projected features is

    K_S = (K_c A_S) (A_S^T K_c A_S)^-1 (K_c A_S)^T = U_S G^-1 U_S^T

which keeps the variance of the molecules along those directions as it is.  That is the
point of projecting rather than regressing on the variates: the canonical variates are
whitened by construction, so a regression on them has no way to prefer the directions the
molecules actually vary along, which is the prior kernel ridge regression rests on.
`K_c - K_S` is the kernel of what is left, the part of the field outside the span.

### `activity_on_kernel(K: np.ndarray, y: np.ndarray, folds: int=5, seed: Optional[int]=0, lams: Sequence[float]=LAMS) -> Tuple[float, np.ndarray, float]`

Nested cross-validated q2 of kernel ridge regression on a kernel, with the predictions.

The regularisation is chosen inside every outer fold by an inner cross-validation on the
training part, so the q2 is not helped by the choice.  `lams` are multiples of the mean
diagonal of the kernel.  Returns `(q2, y_cv, lam)` where `lam` is the regularisation an
inner cross-validation on the whole set picks, for a final model.

### `private_coefficients(fit: KCCAFit, contained: np.ndarray, beta: np.ndarray) -> np.ndarray`

Dual coefficients over the training molecules of a model fitted on the private kernel.

A model on `K_P = K_c - K_S` predicts `f(x) = phi(x)^T (I - P_S) Phi_c^T beta`, which is
linear in the field with the weight vector `Phi_c^T (beta - A_S G^-1 U_S^T beta)`; the
centring makes the coefficients sum to zero.  Returned as coefficients on the raw training
Gaussians, so `component_model` turns the private part of the field into a co-field.

### `paired_difference(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, runs: int=2000, seed: Optional[int]=0, level: float=0.95) -> Tuple[float, float, float]`

`q2(pred_a) - q2(pred_b)` with its bootstrap interval over the molecules (rule 4).

The two predictions are resampled together, molecule by molecule, so the interval is that
of the difference and not of two separate numbers.

### `analyse(K: np.ndarray, L: np.ndarray, y: Optional[np.ndarray]=None, ridge: Optional[float]=None, components: int=3, folds: int=5, seed: Optional[int]=0, permutations: int=200, grid: Sequence[float]=RIDGE_GRID, rcond: float=RCOND, share: float=CONTAINED_SHARE, progress=None) -> KCCAResult`

The whole analysis of two views: ridge, held-out correlations, shuffled pairing, gates.

`ridge` of None chooses it on the folds by the mean held-out correlation over all pairs;
`permutations` of 0 skips the shuffled pairing; `y` of None skips gate (ii).  `components`
is how many leading pairs the tables show.  `share` is the share of a direction's variance
the other view has to explain on held-out molecules for the direction to count as contained.

### `component_model(probe, direction: np.ndarray, variate: Optional[np.ndarray]=None)`

A canonical direction of a field view as a model, so that it has a co-field.

`probe` is a `cmfqsar.model.CMFModel` whose training kernels were prepared on the
molecules of the view (`prepare_training_kernels`); the result shares its field set,
normalisation and kernels and has the direction as its dual coefficients, a bias of zero and
the canonical variate as its "activity".  Everything that reads a linear model -- the
co-field export, the atom contributions, the F3 cosine -- works on it unchanged, and it can
be saved with `cmfqsar.dataio.save_model` and drawn with `cmf cofield`.

### `component_cosines(models_a: Sequence, models_b: Sequence) -> np.ndarray`

The F3 cosine between the co-fields of the components of two field views.

Entry `(i, j)` is the cosine between component `i` of view A and component `j` of
view B, over all fields together.  For two views on the same fields in one frame the
diagonal is what gate (iii) of the item reads: the two co-fields of one shared component
should resemble each other more than the co-fields of two different components do.

### `cosine_summary(M: np.ndarray) -> Tuple[float, float]`

Mean signed cosine on the diagonal and mean absolute cosine off it.

## cmfqsar.kernels

Source: `cmfqsar/kernels.py`

Molecular-field kernels.

For two atoms with Gaussian field contributions

    rho_l(r) = w_l exp(-b_l/2 |r - r_l|^2),   rho_m(r) = w_m exp(-b_m/2 |r - r_m|^2)

the overlap integral over R^3 is analytic:

    k(l, m) = w_l w_m (2 pi / (b_l + b_m))^{3/2} exp(-(b_l b_m) / (2 (b_l + b_m)) |r_l - r_m|^2)

For b_l = b_m = alpha this reduces to Eq. 5 of Baskin & Zhokhova (2013):

    k = w_l w_m (pi/alpha)^{3/2} exp(-alpha/4 |r_l - r_m|^2)

and for the radius-scaled steric field (b = alpha / R_vdw^2) to Eq. 11.

The kernel of molecules i, j for field f is the sum over all atom pairs
(Eq. 2); the joint kernel is the h_f-weighted sum over fields (Eq. 1).

Three things make this fast enough to use on more than one series (plan item B1).  They were
chosen by measuring one 40-atom molecule against a 200-molecule table, 318 000 atom pairs:

* **The training kernel is computed once, not twice.**  K_f(i, j) = K_f(j, i), so for a table
  against itself only the upper triangle is evaluated and then mirrored.  This is the largest of
  the three, because it removes half of the work rather than part of it, and it is exact: the two
  triangles used to differ by about 4e-13, which was round-off from summing the atom pairs in two
  different orders.

* **The radius-scaled prefactor uses a square root, not a power.**  The steric field has a width
  per atom, so `(2 pi / (b_l + b_m))^{3/2}` has a different value for every atom pair, and
  `x ** 1.5` on 318 000 elements took 2.3 ms of the block's 9.2 ms.  `z * sqrt(z)` is the same
  number -- the two agree to 1.4e-14 -- and costs a third as much, because a square root is a
  machine instruction and a general power is a library call.

* **The distances are computed once per molecule, not once per alpha** (what B1 asked for).  The
  attenuation enters only through the exponential, so a grid search can share the squared
  distances: `kernel_matrices_multi` takes several field sets and evaluates them all from
  one distance block.  This is the smallest of the three; the distance matrix is 3 % of the time.

A fourth shortcut, the cutoff of plan item B2, is **off unless it is worth switching on**.  A
Gaussian overlap falls below 10^-6 of its peak beyond `sqrt(-4 ln tol / b)`, so atom pairs
further apart than that can be dropped and the sum run over a neighbour list instead of over
every pair.  Whether that saves anything depends on the molecules, and for the series this
program is usually given it does not: the molecules of a congeneric series are superimposed in
one frame about 14 A across, and at the usual alpha = 0.5 the cutoff is 10.5 A, which keeps 97 %
of the atom pairs of the electrostatic field and 100 % of the steric one.  On 60 decapeptides
spread over 33 A the same cutoff keeps 62 %, and at alpha = 2 it keeps 14 %, where the neighbour
list is 2.7 times faster than the dense block.  `atom_kernel_block` therefore takes
`cutoff="auto"`, which compares the cutoff sphere with the bounding box of the molecules and
uses the neighbour list only when it expects to drop most of the pairs.  The dense path stays as
the exact reference, and `tests/test_kernel_speed.py` holds the two to the tolerance the cutoff
was derived from.

### `class Cancelled  — bases: Exception`

### `use_compiled(engine=None) -> bool`

Whether this call should use the compiled loop.

`engine` of None means the module default `ENGINE`.  "numba" asks for it explicitly and
raises when Numba is not installed, which is what a benchmark wants; "auto" falls back.

### `cutoff_radius(coefficient: float, tol: float=CUTOFF_TOL) -> float`

The distance beyond which `exp(-coefficient * d^2)` is below `tol` of its peak.

`coefficient` is what multiplies the squared distance in the exponent: `alpha / 4` for a
field whose Gaussians share one width, and `b_min / 4` for the radius-scaled steric field,
where `b_min = alpha / R_max^2` is the widest Gaussian present -- the widest one decides,
because it is the one that still contributes at long range.

### `surviving_share(xi: np.ndarray, coords: np.ndarray, radius: float, sample: int=16, seed: int=0) -> float`

The share of atom pairs closer than `radius`, estimated from a sample of them.

A geometric estimate from the bounding box is not good enough here, because the molecules of a
series are superimposed and their atoms are concentrated in the middle of the box rather than
spread through it, so the volume ratio underestimates the share by a factor of three or more.
A sample of at most `sample` x 8 `sample` real distances answers the question directly.
The default is 16 x 128 = 2048 pairs, which estimates a share of 0.2 to within 0.01 and costs
about a hundredth of what the block costs; the first version sampled nine times as many pairs
and that showed up as a 12 % overhead on the sets where the answer is "do not bother".

### `pair_distances(xi: np.ndarray, Q: AtomTable, first_mol: int=0) -> np.ndarray`

Squared distances from the atoms of one molecule to the atoms of table Q.

The only part of a field kernel that does not depend on the attenuation, which is why
`kernel_matrices_multi` computes it once for a whole grid of alphas.  `first_mol`
restricts the columns to the molecules from that index onwards.

### `efield_pair_factor(alpha: float, d2: np.ndarray) -> np.ndarray`

The pair integral of the electric-field kind (plan item EC5).

For two atom-centred Gaussians of the shared width that `alpha` names, this is

    integral of grad g_i . grad g_j over space
        = (pi / alpha) ** 1.5 * exp(-alpha d^2 / 4) * (3 alpha / 2 - alpha^2 d^2 / 4),

which is the ordinary overlap of the same two Gaussians multiplied by
`3 alpha / 2 - alpha^2 d^2 / 4`.  It is the negative Laplacian of the overlap with respect
to the separation of the two centres, so it needs no quadrature and rule 3 of the plan is
kept.  The derivation and a check against a grid integral are in
`tests/test_efield.py`.

Unlike every other pair integral in this module this one **changes sign**: it is positive for
two close atoms, zero at `d^2 = 6 / alpha` and negative beyond, because two charges far
apart have gradients that point against each other.  The kernel matrix is still positive
semidefinite, since it is an inner product of two vector fields; the sign only means that a
single atom pair may lower the kernel.

### `atom_kernel_block(xi: np.ndarray, wi: np.ndarray, ri: np.ndarray, Q: AtomTable, specs: Sequence[FieldSpec], ci=None, cq=None, first_mol: int=0, D2: Optional[np.ndarray]=None, cutoff=None, tol: float=CUTOFF_TOL) -> np.ndarray`

Kernels between the atoms of one molecule and every molecule of table Q.

Returns an array of shape (F, Ni, nQ): the contribution of every atom of
the probe molecule to the field kernels with every molecule in Q.

`first_mol` restricts the columns to the molecules of Q from that index onwards, which is what
the symmetric case needs: for a table against itself only molecules j >= i are evaluated.
`D2` is a squared-distance block from `pair_distances` for the same `first_mol`; passing
it in saves recomputing the distances for every attenuation of a grid.

`cutoff` controls the neighbour list of plan item B2.  None or False computes every atom pair,
which is exact and is the reference the tests compare against.  `"auto"` uses a neighbour
list for each group of fields whose cutoff radius is small against the size of the molecules,
True uses one whatever the geometry, and a number is used as the radius itself.  A cutoff
makes the kernel an approximation: every dropped pair contributed less than `tol` of a peak
overlap, and the measured error on a 60-molecule table was 4e-7 relative.  A cutoff cannot be
combined with a precomputed `D2`, which is a dense block by construction.

### `kernel_matrices(P: AtomTable, Q: AtomTable, specs: Sequence[FieldSpec], progress: Optional[Callable[[int, int], None]]=None, cancel: Optional[Callable[[], bool]]=None, symmetric: Optional[bool]=None, cutoff=None, engine=None) -> np.ndarray`

Per-field kernel matrices K_f(i, j) for molecules i in P and j in Q.

Returns an array of shape (F, nP, nQ).

`symmetric` evaluates only the pairs with j >= i and mirrors them, which halves the work and
is exact because the atom-pair overlap is symmetric.  It defaults to true when Q is P -- the
training kernel, where it applies -- and must not be set for two different tables.

`cutoff` is passed to `atom_kernel_block`: `"auto"` for the neighbour list of item B2
where the geometry makes it worth using, None for the exact dense sum.

`engine` chooses the implementation (item B5): None for the module default, "numba" for the
compiled threaded loop, "auto" for that where it is installed.  The compiled loop cannot take
probe fields and declines small problems, and the call then falls back to NumPy by itself.

### `kernel_matrices_multi(P: AtomTable, Q: AtomTable, spec_sets: Sequence[Sequence[FieldSpec]], progress: Optional[Callable[[int, int], None]]=None, cancel: Optional[Callable[[], bool]]=None, symmetric: Optional[bool]=None, engine=None) -> List[np.ndarray]`

The kernels of several field sets, sharing each molecule's distance block (plan item B1).

A grid search over the attenuation computes one kernel per alpha, and the squared interatomic
distances are the same for all of them.  This evaluates every field set of `spec_sets` from
one distance block per molecule, and returns one (F, nP, nQ) array per field set, in order.

The field sets must describe the same fields in the same order and differ only in their
widths; nothing here checks that, because the caller builds them from one field set.

### `self_kernels(P: AtomTable, specs: Sequence[FieldSpec]) -> np.ndarray`

K_f(i, i) for every molecule i of P. Shape (F, nP).

### `self_kernels_multi(P: AtomTable, spec_sets: Sequence[Sequence[FieldSpec]]) -> List[np.ndarray]`

`self_kernels` for several field sets, sharing each molecule's distance block.

### `normalize_kernels(K: np.ndarray, dP: np.ndarray, dQ: np.ndarray) -> np.ndarray`

Cosine-normalised kernels K_f(i,j) / sqrt(K_f(i,i) K_f(j,j)); zero where undefined.

### `combine_kernels(K: np.ndarray, h: np.ndarray, scale: np.ndarray) -> np.ndarray`

Joint kernel K = sum_f h_f * scale_f * K_f.

### `expansion_overlap(x1, w1, b1, x2, w2, b2) -> float`

The overlap integral of two Gaussian expansions, in closed form.

The expansions are given the way the rest of this program gives them: centres `(n, 3)`,
weights `(n,)` and widths `(n,)`.  The integral is the sum of the pair overlaps at the top
of this module, so this is the same formula `atom_kernel_block` evaluates over atom
pairs, written for two bags of Gaussians that need not be molecules at all -- a co-field, a
mask, a masked co-field (plan item MC5).

### `gaussian_product(x1, w1, b1, x2, w2, b2, tol: float=0.0)`

The Gaussian expansion of the pointwise *product* of two Gaussian expansions.

The product of two Gaussians is a Gaussian -- the same fact that makes every overlap in this
program closed form, used on the integrand instead of on the integral:

    exp(-b/2 |r - A|^2) exp(-c/2 |r - B|^2)
        = exp(-(b c) / (2 (b + c)) |A - B|^2) exp(-(b + c)/2 |r - C|^2),
        C = (b A + c B) / (b + c)

So a co-field multiplied by a mask is again a sum of Gaussians, and anything that reads a
Gaussian expansion -- the grid drawing, the cube export, `expansion_overlap` -- works on
it unchanged (plan item MC5).  The result has `n1 * n2` terms; `tol` drops the terms whose
weight is below that fraction of the largest, which is worth doing because two Gaussians far
apart make a term that is numerically zero.

### `numerical_overlap(x1, w1, b1, x2, w2, b2, spacing=0.1, extent=6.0) -> float`

Numerically integrate the product of two Gaussian atom fields (slow; for tests).

### `kernel_gradient(xi: np.ndarray, wi: np.ndarray, ri: np.ndarray, Q: AtomTable, specs: Sequence[FieldSpec], u: np.ndarray, chunk: int=4000, ci=None) -> np.ndarray`

Gradient of `sum_f sum_j u[f, j] K_f(i, j)` with respect to the atom positions of i.

The kernels are sums of Gaussians in the squared interatomic distances, so the
derivative is a weighted sum of the same Gaussians times the interatomic vectors:

    d/dr_il  w_fil w_fjm g(|r_il - r_jm|^2)  =  2 g'(d^2) w_fil w_fjm (r_il - r_jm)

with `g'/g = -alpha_f/4` for ordinary fields and `-b_il b_jm / (2 (b_il + b_jm))`
for the radius-scaled steric field.

Parameters
----------
u : array of shape (F, nQ), the multiplier of every field/molecule pair.
ci : the moving molecule's probe-field expansion coefficients, `{name: (Ni, n_terms)}`,
    needed only when the field set contains probe fields.

Returns
-------
array of shape (Ni, 3)

### `so3_right_jacobian(omega: np.ndarray) -> np.ndarray`

Right Jacobian of the SO(3) exponential map (identity at omega = 0).

## cmfqsar.learnincrements

Source: `cmfqsar/learnincrements.py`

Fitting the atomic increments to the data instead of taking them from the table (item D5).

The hydrophobic field and the two hydrogen-bond fields are built from one-atom increments: a
number per (Sybyl atom type, number of attached hydrogens), taken from the regressions of the 2013
scripts (`cmfqsar.increments`).  Those numbers were fitted to logP and to Abraham's solute
constants, not to the activity being modelled, and the program's other parameterisation ("rdkit")
gives a q2 that differs by 0.20 on one Sutherland set.  That difference says the increments are
not settled.  This module fits them to the data and asks whether what it learns is a property of
the atom types or of the series.

How the fitting is possible
---------------------------
For a Gaussian field the kernel is a double sum over atom pairs:

    K(A, B) = sum over i in A, j in B of  w_i w_j E(|r_i - r_j|)

with E the overlap integral of two Gaussians, which depends on the geometry and the attenuation
alpha but not on the weights.  An increment gives every atom of one type the same weight, so with
`c_t` the increment of type `t`:

    K(A, B) = sum over types t, u of  c_t c_u M_tu(A, B)
    M_tu(A, B) = sum over i in A of type t, j in B of type u of  E(|r_i - r_j|)

The kernel is a quadratic form in the increments, and `M` does not depend on them.  So `M` is
built once per attenuation, every candidate kernel is one small contraction, and the derivative of
anything computed from the kernel can be carried back to the increments exactly.  That is what
makes a gradient method affordable here.

What is optimised
-----------------
The loss is the k-fold cross-validated squared error of kernel ridge regression -- the number the
model is selected on -- plus a penalty on the distance from the published increments:

    L(c) = sum over folds of || y_test - K_test,train (K_train + lam I)^-1 y_train ||^2
           + rho * || c - c_published ||^2

The gradient is analytic through the fold solves, the field scaling and the cosine normalisation.
One warning about the parameterisation: a cosine-normalised kernel does not change when a field's
increments are all multiplied by the same number, so only the *relative* increments of a field are
identifiable.  The fitted vectors are therefore rescaled to the length of the published ones
before they are reported or compared, and the penalty is applied to the rescaled vector.

Usage
-----
    from cmfqsar.learnincrements import fit_increments, increment_types

    fit = fit_increments(mols, names, y, alpha=0.5, lam=0.1)
    fs.increments = fit.increments        # a field set now builds the fields with them

### `class IncrementBasis  (dataclass)`

What a series' increment kernels are built from, at one attenuation.

`M` has shape (n_mol, T, n_mol, T): the overlap integral summed over the atom pairs of two
molecules whose types are t and u.  Any increment vector `c` gives the field kernel
`K = einsum('t,u,mtnu->mn', c, c, M)`, so every candidate costs one contraction instead of a
pass over the atom pairs.

Attributes:

- `keys: List[str]`
- `alpha: float`
- `M: np.ndarray`
- `n_mol: int`
- `counts: Dict[str, int]` = `_field(default_factory=dict)`

#### `IncrementBasis.n_types` -> int  (property)

#### `IncrementBasis.megabytes` -> float  (property)

#### `IncrementBasis.kernel(c: np.ndarray) -> np.ndarray`

The field kernel of one increment vector, (n_mol, n_mol).

#### `IncrementBasis.half(c: np.ndarray) -> np.ndarray`

`S[t][m, n] = sum_u c_u M[m, t, n, u]`, the half-contraction the gradient needs.

### `class IncrementFit  (dataclass)`

What one fitting produced.

Attributes:

- `keys: List[str]`
- `values: np.ndarray`
- `published: np.ndarray`
- `alpha: float`
- `lam: float`
- `rho: float`
- `loss: float`
- `loss_published: float`
- `iterations: int`
- `message: str` = `''`
- `counts: Dict[str, int]` = `_field(default_factory=dict)`

#### `IncrementFit.increments` -> Dict[str, Dict[str, float]]  (property)

The fitted increments as a table a field set can use.

#### `IncrementFit.cosine() -> Dict[str, float]`

How much each fitted field still looks like the published one, in [-1, 1].

#### `IncrementFit.to_json(path: str) -> None`

### `increment_key(syb: str, n_h: int) -> str`

The key the increments are stored under: the Sybyl type and the hydrogen count.

### `increment_types(mol: Chem.Mol, legacy: bool=False) -> List[str]`

The increment key of every atom of a molecule; hydrogens get an empty key.

### `published_increments(keys: Sequence[str]) -> np.ndarray`

The published increment of every key, as (len(keys), 3) in the order of INCREMENT_FIELDS.

### `types_in(mols: Sequence[Chem.Mol], min_atoms: int=MIN_ATOMS, max_types: int=MAX_TYPES, legacy: bool=False) -> Tuple[List[str], Dict[str, int]]`

The increment types worth fitting in a series, and how many atoms carry each.

A type on too few atoms cannot be fitted and would only add a free parameter, so the rare ones
are merged into one group called "other", whose increment is fitted like any other.  The
published value of that group is zero, which is what a type the table does not know would get.

### `type_index(mols: Sequence[Chem.Mol], keys: Sequence[str], legacy: bool=False) -> List[np.ndarray]`

For every molecule, the index into `keys` of each atom, or -1 for an atom with no increment.

An atom whose own type was merged away lands in the "other" group when there is one.

### `basis_from(mols: Sequence[MoleculeFields], index: Sequence[np.ndarray], alpha: float, keys: Sequence[str], counts: Optional[Dict[str, int]]=None, dtype=np.float32) -> IncrementBasis`

Build the (molecule, type) overlap tensor of a series at one attenuation.

`mols` are the built fields (only their coordinates are used) and `index` the per-atom type
index of `type_index`, in the same order.

### `build_basis(mols: Sequence[Chem.Mol], names: Sequence[str], alpha: float, keys: Optional[Sequence[str]]=None, fieldset: Optional[FieldSet]=None, legacy: bool=False, dtype=np.float32) -> IncrementBasis`

The basis of a series given the molecules themselves.

### `normalise(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`

Cosine-normalise a kernel, returning it and the square roots of its diagonal.

### `folds_of(n: int, k: int, seed: int=0) -> List[np.ndarray]`

The folds the model's own cross-validation uses, so the loss is the number it reports.

### `cv_loss_and_grad(K: np.ndarray, y: np.ndarray, lam: float, folds: Sequence[np.ndarray], fit_bias: bool=True) -> Tuple[float, np.ndarray]`

The cross-validated squared error of kernel ridge regression, and its gradient in K.

The learner is the one the model uses: kernel ridge regression with the bias term of Eq. 12
of Baskin and Zhokhova (2013), which solves the bordered system

    [ K + lam I   1 ] [ a ]   [ y ]
    [ 1^T         0 ] [ b ] = [ 0 ]

so the fitted increments are fitted against the model as it is, not against a simpler one.

Returns (loss, G) with `G[i, j] = dL / dK[i, j]`, symmetric, so that the change of the loss
under a symmetric change of the kernel is the sum of G times that change.

### `read_increments(path: str) -> Dict[str, Dict[str, float]]`

Read a table written by `IncrementFit.to_json` (or any table of the same shape).

### `fit_from_basis(bases: Sequence[IncrementBasis], other: Optional[np.ndarray], y: np.ndarray, lam: float=0.1, rho: float=RHO, folds: int=10, seed: int=0, start: Optional[np.ndarray]=None, max_iter: int=200, mixing: Optional[Sequence[float]]=None, use: Optional[Sequence[int]]=None) -> IncrementFit`

Fit the increments of the three fields on a prepared basis.

`bases` holds one `IncrementBasis` per increment field, in the order of
INCREMENT_FIELDS, all with the same types.  `other` is the summed, already normalised kernel
of the fields that have no increments (steric, electrostatic), or None.  The kernels are
cosine-normalised and added with `mixing` (ones by default), which is what the model does.

`use` fits on a subset of the molecules of the basis -- the training set, when the basis was
built over a whole series -- and `y` then holds that subset's activities.

### `fit_increments(mols: Sequence[Chem.Mol], names: Sequence[str], y: np.ndarray, alpha: float=0.5, lam: float=0.1, rho: float=RHO, folds: int=10, seed: int=0, keys: Optional[Sequence[str]]=None, fieldset: Optional[FieldSet]=None, legacy: bool=False, max_iter: int=200) -> IncrementFit`

Fit the increments of one series, including the fields that have no increments.

The other fields of `fieldset` (steric, electrostatic and any others) are built once and
enter the kernel as a constant, so what is fitted is only the three increment fields against
the rest of the model as it stands.

### `other_kernel(built: Sequence[MoleculeFields], fieldset: FieldSet) -> Optional[np.ndarray]`

The normalised, summed kernel of the fields that carry no increments.

### `coverage(table: Dict[str, Dict[str, float]], mols: Sequence[Chem.Mol], legacy: bool=False) -> Dict[str, float]`

How much of a series an increment table covers.

Returns the share of atoms whose own type is in the table, the share that falls back to its
"other" entry and the share that gets nothing.  A table fitted on one series and applied to
another is only as good as this says it is.

### `spread(fits: Sequence[IncrementFit]) -> Dict[str, float]`

How much the increments fitted on different series agree with each other.

The mean cosine between every pair of fitted vectors, per field, over the types the two fits
have in common.  One means the series learned the same increments; zero means they learned
unrelated ones, and the increments are then a property of the series rather than of the atom
types.

## cmfqsar.localalign

Source: `cmfqsar/localalign.py`

Local alignment: an environment posed on its centre and first shell (plan item LP3).

`cmfqsar.localenv` (plan item LP2) makes the sample an atom and its surroundings, and
measures it with the rotation-invariant autocorrelation, which needs no frame.  That kernel has
the same limitation as the invariant kernel for whole molecules: with no frame there is no
co-field, so it cannot say **where** around the atom, and in which field, a feature raises or
lowers the property.

For one site the frame is not a problem.  An aromatic carbon with its two ring neighbours, or a
carboxyl carbon with its two oxygens, fixes a frame to within a mirror image, so the environments
of that site in different molecules can be put in one frame by a rule, with nothing fitted and
nothing to converge.  This module is that rule, and with it the overlap kernel and the co-field of
the rest of the program apply to an environment exactly as they apply to a molecule.

**The rule.**  The centre is the origin already (`cmfqsar.localenv.environment` puts it
there and makes it the first atom).  What is left is the rotation, and it is fixed by two of the
centre's bonded neighbours: the first gives the x axis, the second the xy plane.  Which two is
decided by the atom type, so that the same site of two molecules is posed the same way, and by
nothing else: a frame that depended on which of two ring bonds is a hundredth of an angstrom
longer would not be a rule.  Neighbours of the same type are therefore interchangeable, and when
that leaves the choice open, as it does for the two ring neighbours of an aromatic carbon or the
three hydrogens of a methyl group, every choice is kept and
the environment becomes the **mean of those poses**, which is a mean-map kernel over the frames
the site allows (`cmfqsar.fields.combine_members`).  That is the same device the symmetrised
Hammett set uses for the four images of a monosubstituted benzene: a frame ambiguity is averaged
over rather than resolved by a coin toss.

**The ring site.**  `ring_frames` is the frame for the site the plan names, an aromatic
carbon of a six-membered carbocycle: the carbon at the origin, the bond that leaves the ring on
+x, the ring in the xy plane, and the two mirror images kept.  In that frame the positions of the
ortho, meta and para substituents of the ring are fixed numbers (`RING_POSITIONS`), so a
co-field can be read at them and compared with what substituent effects on an aromatic carbon are
known to be.

### `frame_from(coords: np.ndarray, i: int, j: int, centre: int=0) -> Optional[np.ndarray]`

The rotation whose rows are the x, y and z axes fixed by atoms `i` and `j`.

x is along the centre-to-`i` direction, y is what is left of the centre-to-`j` direction after
x is taken out, and z completes a right-handed set.  Returns None when the three atoms are
collinear, where no frame exists.

### `local_frames(env: MoleculeFields, max_images: int=MAX_IMAGES, centre: int=0) -> List[np.ndarray]`

Every pose of this environment the key allows, at most `max_images` of them.

One rotation is returned for a site the key separates, two for an aromatic carbon with two
equivalent ring neighbours, three for a methyl carbon, and the identity for a centre with
nothing bonded to it, where there is no frame to speak of.

### `ring_frames(env: MoleculeFields, centre: int=0) -> List[np.ndarray]`

The two poses of an aromatic ring carbon: the bond leaving the ring on +x, ring in xy.

This is the site of plan item LP3 and the local analogue of the frame the Hammett set uses for
a whole molecule.  x points away from the ring, so a substituent on the centre sits on +x at
about 1.4 A; the ring is in the xy plane with its two neighbours at about +-120 degrees; z is
the ring normal.  The two poses are mirror images in y, which is the ambiguity of which ring
neighbour is called the first one, and averaging over them is what makes the pose a rule
rather than a choice.

Returns an empty list when the centre is not an aromatic carbon with exactly two aromatic
neighbours, which is what a fused-ring carbon and a non-aromatic centre are.  A five-membered
aromatic ring passes as well, and the frame is as good there, but `RING_POSITIONS` is
the geometry of a six-membered ring and does not apply to it.

### `ring_positions(radius: float=RING_RADIUS, bond: float=SUBSTITUENT_BOND) -> Dict[str, np.ndarray]`

Where a substituent sits, in the frame of `ring_frames`, by ring position.

The centre is a carbon of a regular six-membered ring of circumradius `radius`, so the ring
centroid is at (-radius, 0, 0) and the other five carbons follow from the hexagon.  Each
position is the place the first atom of a substituent on that carbon occupies, `bond` angstrom
out from it along the direction away from the centroid.  The y > 0 member of each pair is
given; the two poses of `ring_frames` are mirror images in y, so the co-field of a set
posed that way is symmetric in y and one of each pair is enough.

### `pose(env: MoleculeFields, R: np.ndarray, name: str='') -> MoleculeFields`

The same environment with its coordinates turned into the frame `R`.

### `posed(env: MoleculeFields, max_images: int=MAX_IMAGES, frames: Optional[Sequence[np.ndarray]]=None, name: str='') -> MoleculeFields`

One field: the environment in every pose its site allows, averaged.

With one pose the result is that pose.  With several it is their mean field, which is a
mean-map kernel over the frames the site allows and removes the arbitrary part of the rule
instead of hiding it.

### `posed_ensemble(members: Sequence[MoleculeFields], atom: int, rho: float=DEFAULT_RHO, floor: float=ENVELOPE_FLOOR, weights: Optional[Sequence[float]]=None, max_images: int=MAX_IMAGES, frames_of: Optional[Callable[[MoleculeFields], List[np.ndarray]]]=None, name: str='') -> MoleculeFields`

One field: an atom's environment in every conformation, each posed in its own frame.

The frame is a property of the site, and the site moves with the conformation, so the pose
has to be taken per conformation and the averaging has to come after it.  Posing the mean
field instead would put one rotation on geometries that do not share it.

The result is the mean over conformations and over the poses each site allows, which is one
ensemble field: the two averages compose, because a mean of means over a shared atom list is
a mean (`cmfqsar.fields.combine_members`).

### `posed_environments(mols: Sequence[MoleculeFields], targets: AtomTargets, rho: float=DEFAULT_RHO, floor: float=ENVELOPE_FLOOR, max_images: int=MAX_IMAGES, frames_of: Optional[Callable[[MoleculeFields], List[np.ndarray]]]=None, progress=None) -> List[MoleculeFields]`

One posed environment per measured atom, in the row order of `targets`.

`frames_of` replaces the general rule by a site rule -- `ring_frames` for an aromatic
carbon -- and is given the environment, whose centre is its first atom.

### `images_of(env: MoleculeFields, max_images: int=MAX_IMAGES, frames_of: Optional[Callable[[MoleculeFields], List[np.ndarray]]]=None) -> int`

How many poses the site of this environment allows: 1 when the key settles it.

### `well_posed(mols: Sequence[MoleculeFields], targets: AtomTargets, rho: float=DEFAULT_RHO, floor: float=ENVELOPE_FLOOR, max_images: int=MAX_IMAGES, frames_of: Optional[Callable[[MoleculeFields], List[np.ndarray]]]=None) -> Dict[int, int]`

How many of the samples have how many poses: {poses: samples}.

A site that gives one pose is settled by the key; two is the usual mirror pair; zero means the
site rule refuses this centre, and such a sample has to be left out rather than posed anyway.

### `overlap_kernel(posed_envs: Sequence[MoleculeFields], fieldset, y=None, progress=None)`

The overlap kernel of a set of posed environments, and the model that holds it.

The environments are in one frame, so the ordinary CMF overlap kernel applies to them exactly
as it applies to molecules: this builds the `cmfqsar.fields.AtomTable` and returns the
fitted-kernel model together with the joint training kernel, which
`cmfqsar.localenv.cross_validate` then runs over folds that hold out whole molecules.

## cmfqsar.localenv

Source: `cmfqsar/localenv.py`

Atom-centred environments and the invariant local kernel (plan item LP2).

Everything else in this program takes a molecule as the sample.  A carbon shift, an acidity or a
site reactivity belongs to an *atom in a molecule*, so the sample has to be the atom and its
surroundings.  `cmfqsar.atomtargets` (plan item LP1) holds the targets of that shape; this
module builds the samples that go with them.

**The environment.**  One molecule gives as many samples as it has measured atoms.  The sample for
atom *a* is the molecule's own fields multiplied by a Gaussian envelope about *a*,

    g(r) = f(r) * exp(-rho |r - r_a|^2),

so an atom close to the centre keeps its weight and one far away loses it.  A CMF field is a sum
of atom-centred Gaussians, and this envelope keeps that form: `environment` returns a
`cmfqsar.fields.MoleculeFields` whose atoms are the atoms of the molecule, whose
coordinates are measured from the centre, and whose weights are damped by the envelope.  Every
tool that works on a molecule's fields therefore works on an environment -- the kernels, the
regression, the variance, the atom decomposition -- because an environment *is* a field.

`rho` is a hyper-parameter like `alpha`, and it says how far the model is allowed to look: the
envelope falls to 1/e at `1/sqrt(rho)`, which `envelope_width` reports.  A shielding is a
short-range property and wants a narrow envelope; an acidity is felt further away and wants a
wider one.

**The kernel.**  The first kernel to measure is the rotation-invariant field autocorrelation of
`cmfqsar.invariant`, taken on the environment instead of on the whole molecule, because it
needs no alignment: two environments of different molecules need not be superposed for their
autocorrelations to be compared.  The autocorrelation is a double sum over atom pairs, so on an
environment it becomes

    sum over atoms a, b of  w_a w_b exp(-rho d_a0^2) exp(-rho d_b0^2) * (distance bin of d_ab),

with `d_a0` the distance of an atom from the centre.  It is a local descriptor built from the
same fields as the global one, and the prediction still splits over the atoms exactly
(`cmfqsar.model.CMFModel.atom_contributions`, plan item SC6).

**The reference.**  Under rule 1 of the plan every item is measured against the simplest thing
that could work, which here is an atom-centred count: the type of the centre itself and the counts
of atom types in distance shells around it, which is what a per-atom increment scheme is.
`count_features` builds it from the same molecules, the same atom types and the same
samples, and `cross_validate` runs both through one cross-validation, so the only difference
between the two numbers is the descriptor.

**The folds.**  `cmfqsar.atomtargets.AtomTargets.folds` holds out whole molecules, and this
module uses nothing else.  Two carbons of one molecule, one in the training set and one in the
test set, would let the model see the molecule it is being tested on.

### `class LocalResult  (dataclass)`

What one descriptor scored on the per-atom target, out of fold.

Attributes:

- `name: str`
- `predicted: np.ndarray`
- `q2: float`
- `error: float`
- `mae: float`
- `lam: float`
- `n_samples: int`
- `n_molecules: int`
- `folds: int`
- `note: str` = `''`
- `molecule: np.ndarray` = `_field(default_factory=lambda: np.zeros(0, int))`

#### `LocalResult.summary() -> str`

### `envelope_width(rho: float=DEFAULT_RHO) -> float`

The distance at which the envelope falls to 1/e, in angstrom.

### `envelope_radius(rho: float=DEFAULT_RHO, floor: float=ENVELOPE_FLOOR) -> float`

The distance beyond which an atom is left out of the environment, in angstrom.

### `envelope(coords: np.ndarray, centre: Sequence[float], rho: float=DEFAULT_RHO) -> np.ndarray`

The envelope factor exp(-rho d^2) of every atom, with d measured from `centre`.

### `environment(mol: MoleculeFields, atom: int, rho: float=DEFAULT_RHO, floor: float=ENVELOPE_FLOOR, name: str='', keep: Optional[Sequence[int]]=None) -> MoleculeFields`

The fields of `mol` seen from `atom`: the same fields under a Gaussian envelope.

The result is an ordinary `cmfqsar.fields.MoleculeFields`, with the centre at the
origin and every weight multiplied by `exp(-rho d^2)`, so the whole program applies to it.
Atoms whose factor is below `floor` are left out, which costs nothing measurable and saves the
cost of carrying them: at the default floor and the default `rho` that is everything beyond
4.3 A, which `envelope_radius` reports.  `kept_share` says what share of the
enveloped weight the floor dropped.

The centre is the first atom of the result, so that `e.coords[0]` is the origin and
`e.atom_types[0]` is the type of the atom the sample belongs to.  Which atom of the parent
molecule every other atom came from is kept in `member_atom_map`, which is otherwise used
for the members of an ensemble and is inert here, so that a per-atom quantity computed on an
environment can be put back on the molecule it came from.

An ensemble field may be passed, and `atom` then indexes its concatenated atom list.  The
result is a plain field rather than an ensemble, because the members' weights are already
multiplied into the atoms they belong to; the atom decomposition of such an environment is
therefore per atom of a member, which is what a difference field (`difference_field`)
means.  To average an environment over the **conformations** of one molecule, use
`ensemble_environment` instead: the atom is in a different place in every conformation,
so each one has its own environment and it is those that are averaged.

`keep` overrides the choice of atoms the floor would make, which is what lets several
conformations of one molecule share an atom list.

### `kept_atoms(mol: MoleculeFields, atom: int, rho: float=DEFAULT_RHO, floor: float=ENVELOPE_FLOOR) -> np.ndarray`

The atoms `environment` would keep, as an index array.

### `ensemble_environment(members: Sequence[MoleculeFields], atom: int, rho: float=DEFAULT_RHO, floor: float=ENVELOPE_FLOOR, weights: Optional[Sequence[float]]=None, name: str='') -> MoleculeFields`

The environment of one atom, averaged over the conformations of its molecule (LP4).

A measured shift is an average over the conformations the molecule visits, and so is an
acidity; the single conformation that a distance-geometry program returns is a stand-in for
that average and a poor one.  The averaging needs no new kernel, because an environment is a
field: the mean field of the per-conformation environments is itself a field, and the kernel
on it is the mean-map kernel of `cmfqsar.fields.combine_members`, exactly as a
conformational ensemble of whole molecules is handled.

What it cannot be is the environment of the mean field of the molecule.  The atom sits in a
different place in every conformation, so there is no single centre to put an envelope on;
each conformation has its own environment, centred on its own copy of the atom, and it is
those that are averaged.

`members` are the fields of one molecule in its conformations, in the same atom order.  They
share one atom list -- the union of what the floor keeps in any of them -- so that the mean is
taken over the same atoms rather than over whichever ones happened to survive.

### `kept_share(mol: MoleculeFields, atom: int, rho: float=DEFAULT_RHO, floor: float=ENVELOPE_FLOOR) -> float`

The share of the enveloped field weight that the floor keeps, in [0, 1].

### `environments(mols: Sequence[MoleculeFields], targets: AtomTargets, rho: float=DEFAULT_RHO, floor: float=ENVELOPE_FLOOR, progress=None) -> List[MoleculeFields]`

One environment per measured atom, in the row order of `targets`.

### `difference_field(a: MoleculeFields, b: MoleculeFields, name: str='') -> MoleculeFields`

The field of `a` minus the field of `b`, as one field (plan item LP2).

A kernel on CMF fields is bilinear, so a difference of two fields is again a field: its atoms
are the atoms of both molecules, with the second molecule's weights negated.  That is the
signed counterpart of `cmfqsar.fields.combine_members`, which averages members and so
needs non-negative weights.

The acidity of a site is the free energy of a change -- the acid loses a proton -- and the
difference field of the acid and its anion is that change as a field, which is why it is the
sample for a pKa.  The two molecules must be in one frame for the difference to mean anything;
the protonation states built by `cmfqsar.protonation` are, because they are built on the
geometry of the parent.

### `local_features(envs: Sequence[MoleculeFields], spec: AutocorrelationSpec=LOCAL_SPEC, progress=None) -> np.ndarray`

Autocorrelation features of a list of environments, one row each.

### `local_kernel(train: np.ndarray, test: Optional[np.ndarray]=None, scale: Optional[np.ndarray]=None, spec: AutocorrelationSpec=LOCAL_SPEC, normalize: bool=True) -> np.ndarray`

Cosine-normalised linear kernel on the environment autocorrelations.

### `types_of(mols: Sequence[MoleculeFields], min_atoms: int=20) -> List[str]`

The atom types worth a column of their own: those on at least `min_atoms` atoms.

### `type_index(types: Sequence[str]) -> Dict[str, int]`

Where each type sits in a count row; everything else shares the last column.

### `shell_counts(mol: MoleculeFields, atom: int, types: Sequence[str], edges: Sequence[float]=SHELL_EDGES) -> np.ndarray`

What is around one atom: its own type, and the atom types in distance shells about it.

This is the reference the local kernel has to beat (rule 1 of the plan): an atom-centred
increment scheme, written as counts.  The row is the one-hot type of the centre followed by
one count per (type, shell), with a last column in each block for every type too rare to have
one of its own.

### `count_features(mols: Sequence[MoleculeFields], targets: AtomTargets, types: Sequence[str], edges: Sequence[float]=SHELL_EDGES) -> np.ndarray`

Count rows of the reference, one per measured atom, in the row order of `targets`.

### `topological_counts(mol, atom: int, types: Sequence[str], shells: int=BOND_SHELLS) -> np.ndarray`

What is around one atom counted over **bonds** rather than over space.

This is the two-dimensional baseline: the type of the centre and the atom types one, two,
three and four bonds away, which is what a topological fingerprint of an atom holds and what
a model without any geometry has to work with.  `shell_counts` is its counterpart over
distance, and the two differ in exactly the thing a field model is supposed to add.

`mol` is an RDKit molecule here rather than a `cmfqsar.fields.MoleculeFields`, because
the bond graph is what it counts over; the types are the Sybyl types of
`cmfqsar.atomtypes`, so that the two baselines name their columns the same way.

### `topological_features(mols, targets: AtomTargets, types: Sequence[str], shells: int=BOND_SHELLS) -> np.ndarray`

Topological count rows, one per measured atom, in the row order of `targets`.

`mols` are RDKit molecules, in the order the targets index them.

### `count_kernel(train: np.ndarray, test: Optional[np.ndarray]=None, normalize: bool=True) -> np.ndarray`

Cosine-normalised linear kernel on the count rows.

Every column is a count, so there is nothing to rescale between them: unlike the field
channels, which carry charges in one block and ones in another, the blocks here are already
in the same units.

### `cross_validate(K: np.ndarray, targets: AtomTargets, lam: float=0.001, folds: int=10, seed: Optional[int]=0, name: str='', note: str='') -> LocalResult`

Out-of-fold predictions of a kernel on a per-atom target, holding out whole molecules.

The folds come from `cmfqsar.atomtargets.AtomTargets.folds`, so every atom of a molecule
is predicted by a model that never saw that molecule.

### `cross_validate_lams(K: np.ndarray, targets: AtomTargets, lams: Sequence[float]=LAMS, folds: int=10, seed: Optional[int]=0, name: str='') -> List[LocalResult]`

One result per ridge value, from one pass over the folds.

### `choose_lam(K: np.ndarray, targets: AtomTargets, lams: Sequence[float]=LAMS, folds: int=10, seed: Optional[int]=0, name: str='') -> LocalResult`

The ridge that cross-validates best, and its result.

Both the local kernel and the count reference are given this same search on the same folds, so
the choice favours neither.  The q2 it reports is the one the search maximised and is
therefore optimistic by the amount any such search is; what is being compared is a difference
between two descriptors treated alike, not an absolute figure.

### `top_hits(targets: AtomTargets, predicted: np.ndarray, k: int=1) -> np.ndarray`

One 0 or 1 per molecule: is a measured atom with a positive value among its top `k`?

This is the measure for a target that asks **which atom** rather than how much: a site of
metabolism, the most reactive position, the atom a reagent reaches first (plan item MC8).  The
ranking happens inside a molecule, so the scale of one molecule's predictions cannot help or
hurt another, and a molecule with no positive value asks no question and is left out.

### `top_k_accuracy(targets: AtomTargets, predicted: np.ndarray, k: int=1) -> float`

The share of molecules whose true atom is among the `k` the model ranks highest.

### `paired_interval(a: LocalResult, b: LocalResult, targets: AtomTargets, runs: int=2000, seed: int=0, statistic: str='q2') -> Tuple[float, float, float]`

(difference, low, high) for `a` against `b`, bootstrapped over **molecules**.

Resampling atoms would overstate the precision, because the atoms of one molecule are not
independent: they share a geometry, a charge model and whatever the molecule's chemistry does
to all of them.  So a bootstrap sample is a sample of molecules, with every measured atom of a
drawn molecule going in with it.  Rule 4 of the plan then applies as it does to molecules: a
difference inside the interval is not a result.

## cmfqsar.locallarge

Source: `cmfqsar/locallarge.py`

Many atoms at once: the local kernel without a kernel matrix (plan item LP5).

A per-atom target is a large problem where a per-molecule one is a small one.  Four hundred
molecules are five thousand carbons; the free NMR databases hold hundreds of thousands, and an
exact kernel over samples costs the square of their number to build and the cube to solve.  Plan
item LP5 asks for a Nystrom or inducing-point approximation to get past that.

For the rotation-invariant local kernel of `cmfqsar.localenv` no approximation is needed, and
that is what this module is.  The kernel there is a **linear kernel on a finite feature map**: the
field autocorrelation of an environment is a fixed number of channels -- fifteen field pairs times
seventeen distance bins is 255 columns for the five physico-chemical fields -- scaled per block and
then cosine-normalised, and the kernel is the inner product of those rows.  So

    K = Z Z^T,   with Z the scaled and normalised feature rows,

and the ridge regression can be solved in the **primal**, in the space of the 255 coefficients,
at a cost linear in the number of samples:

    (Z^T Z + lam I) w = Z^T y,

a d by d system whatever n is, against the n by n system the dual form solves.  The two give the
same predictions -- the tests check that they agree to round-off -- so this is not an
approximation but the same model computed the other way round.  What it changes is the cost:
memory goes from n^2 to n d, and time from n^3 to n d^2.

The overlap kernel of `cmfqsar.localalign` is a different matter.  Its feature map is the
field itself, which is a function on space and has no finite basis, so there the dual form is the
only exact one and a Nystrom approximation is the route; `nystrom_kernel` builds one from a
set of landmark environments.

### `class Curve  (dataclass)`

What a model of this kind is worth at several training sizes.

Attributes:

- `sizes: List[int]`
- `molecules: List[int]`
- `q2: List[float]`
- `error: List[float]`
- `lam: List[float]`
- `seconds: List[float]`

#### `Curve.summary() -> str`

### `normalised_rows(features: np.ndarray, scale: Optional[np.ndarray]=None, n_bins: int=1, blocks: Optional[Sequence[int]]=None) -> Tuple[np.ndarray, np.ndarray]`

The feature rows whose inner products are the kernel: (rows, the scale used).

This is `cmfqsar.invariant.autocorrelation_kernel` stopped one step early.  That
function scales the channel blocks, cosine-normalises and multiplies; this returns the rows
themselves, so that a model can be fitted in the space they live in.  Pass the `scale` back in
to prepare test rows with the training set's scaling, as a kernel would.

### `primal_fit(Z: np.ndarray, y: np.ndarray, lam: float, fit_bias: bool=True) -> Tuple[np.ndarray, float]`

Ridge regression in the feature space: (weights, bias).

The bias is handled as the dual form handles it -- the fit is on centred rows and a centred
target, which is the same model as the bordered system of `cmfqsar.model.krr_fit` for a
kernel that is an inner product.

### `primal_predict(Z: np.ndarray, w: np.ndarray, b: float) -> np.ndarray`

### `cross_validate_primal(Z: np.ndarray, targets: AtomTargets, lam: float=0.001, folds: int=10, seed: Optional[int]=0, name: str='', note: str='') -> LocalResult`

Out-of-fold predictions in the primal, on folds that hold out whole molecules.

The same folds, the same ridge and the same model as
`cmfqsar.localenv.cross_validate`; only the arithmetic differs.

### `choose_lam_primal(Z: np.ndarray, targets: AtomTargets, lams: Sequence[float]=LAMS, folds: int=10, seed: Optional[int]=0, name: str='') -> LocalResult`

The ridge that cross-validates best, in the primal.

The ridge means the same thing in both forms for an inner-product kernel, so the value chosen
here is the value the dual form would choose.

### `learning_curve(Z: np.ndarray, targets: AtomTargets, sizes: Sequence[int], lams: Sequence[float]=LAMS, folds: int=10, seed: Optional[int]=0, progress=None) -> Curve`

Cross-validated error against the number of **molecules** in the training set.

The sizes are numbers of molecules, not of atoms, because that is what a data set is counted
in and because the folds hold out molecules.  Each size is a subset of the next, drawn once
with the given seed, so the curve is nested and the differences along it are not the noise of
redrawing the set.

### `nystrom_kernel(K_xz: np.ndarray, K_zz: np.ndarray) -> np.ndarray`

The rank-m approximation K(X, Z) K(Z, Z)^+ K(Z, X), for a kernel with no finite features.

This is the route plan item LP5 names, and it is needed for the **overlap** kernel of
`cmfqsar.localalign`, whose feature map is a function on space.  For the invariant local
kernel it is not needed at all: that kernel has 255 features and the primal form above is
exact.  The pseudo-inverse is taken through a symmetric eigendecomposition, which is what
keeps a landmark set with near-duplicate members from blowing the approximation up.

## cmfqsar.lowrank

Source: `cmfqsar/lowrank.py`

Low-rank kernels for large training sets (plan item B6).

The exact model holds one n by n matrix per field and factorises the n by n joint kernel.  On a
32 GB machine that stops at ten to twenty thousand molecules, and memory is the limit rather
than time.  Plan item B6 assessed the ways out and measured one of them; this module implements
it.

**The Nystrom approximation.**  Choose m landmarks Z -- a grid of Gaussian centres in the common
frame, or a subset of the training molecules -- and replace every field kernel by

    K(X, X) ~ K(X, Z) K(Z, Z)^+ K(Z, X).

That is the Gram matrix of the **features** `Phi = K(X, Z) W` with `W = V diag(w)^(-1/2)`
from the eigendecomposition of K(Z, Z), so the model can be fitted in the space of the m columns
instead of the space of the n molecules.  Nothing new has to be computed: `K(X, Z)` is what
`cmfqsar.kernels.kernel_matrices` already produces and what `cmfqsar.screen` already
streams a block at a time.

What it costs and what it saves:

    exact      n*n per field, n^3 to factorise      five fields of 10 000 molecules: 4 GB
    low rank   n*m per field, m^3 to factorise      the same with 300 landmarks: 0.12 GB

and the n by m features need never be held at once: `gram` accumulates `Phi^T Phi` and
`Phi^T y` over blocks of molecules, so the memory of a fit is m by m whatever n is.

**What stays exact.**  The self-kernel K(x, x) of a molecule is one molecule's own integral, so
it costs nothing and is never approximated here.  That matters twice: the cosine normalisation
divides by the exact diagonal, and the posterior variance keeps the exact `k(x, x)`, which is
what stops it collapsing far from the landmarks.  Writing the variance out,

    var(x) = k(x,x) - phi^T phi + lam * phi^T (Phi^T Phi + lam I)^-1 phi,

which is the exact Gaussian-process variance with K replaced by its Nystrom approximation, and
is the deterministic-training-conditional form of the literature.

**What this module is not.**  The approximation is only as good as the landmarks, and plan item
B6 measured where that fails: a grid of Gaussian centres spans the field while its spacing is not
much larger than the field's own width, so at an attenuation of 2 or more a Sutherland-sized
frame would need tens of thousands of centres.  Landmark molecules have no width problem.
`gaussian_centres` and `landmark_molecules` build the two kinds and
`benchmarks/run_lowrank.py` measures what each costs in accuracy.

### `class LowRank  (dataclass)`

The landmarks and the whitening that turn a molecule into a row of features.

`transform` gives the (molecules, columns) matrix whose inner products are the approximate
joint kernel, which is all the rest of the model needs.

Attributes:

- `landmarks: AtomTable`
- `specs: Sequence[object]`
- `whiteners: List[np.ndarray]`
- `weights: np.ndarray`
- `normalize: bool` = `True`
- `self_landmarks: Optional[np.ndarray]` = `None`
- `kind: str` = `'landmarks'`

#### `LowRank.rank` -> int  (property)

#### `LowRank.transform(table: AtomTable, block: int=BLOCK, progress=None) -> np.ndarray`

The feature rows of every molecule of `table`, a block of molecules at a time.

### `class Gram  (dataclass)`

What a fit needs from the whole training set, in m by m memory.

Attributes:

- `PtP: np.ndarray`
- `Pty: np.ndarray`
- `Pt1: np.ndarray`
- `n: int`
- `sum_y: float`
- `yty: float`

### `gaussian_centres(table: AtomTable, spacing: float, n_fields: int, radius: float=CENTRE_RADIUS) -> AtomTable`

A cubic grid of one-atom landmarks of unit weight covering the atoms of `table`.

A centre further than one spacing from every atom carries nothing and is left out.  The rank
such a basis needs depends on the frame and the width of the field, not on how many
molecules there are, which is the property that makes the approximation worth having.

### `landmark_molecules(table: AtomTable, n: int, seed: Optional[int]=0) -> Tuple[AtomTable, np.ndarray]`

`n` of the molecules themselves as landmarks, drawn at random without replacement.

Molecules have no width problem: whatever the attenuation, a molecule's own field is in the
span of the molecules.  The price is that the landmarks are tied to the training set.

### `landmark_atoms(table: AtomTable, n_fields: int, molecules: int=0, min_distance: float=0.75, seed: Optional[int]=0, max_landmarks: int=4000) -> AtomTable`

The atoms of a subset of molecules, each one a landmark where it stands.

This is the case between the two above, and it is the one a **narrow** field wants.  A grid
spans a field while its spacing is not much larger than the field's own width, so at a
strong attenuation a grid over a whole frame would need tens of thousands of centres --
but the field is only ever large near an atom, and the atoms are where a new molecule's
atoms will be as well.  Putting a landmark on every atom of a few molecules follows the
field instead of covering the box.

Each landmark carries unit weight in every field and the radius of the atom it stands on,
so it is a basis function of the right width at the right place rather than a copy of that
atom's own field.  `min_distance` thins the set: atoms of different molecules that sit on
top of each other add nothing but rank, and the pseudo-inverse of the feature map would
have to drop them anyway.  `molecules` takes the atoms of that many molecules, drawn at
random; 0 takes them all.

### `whitener(K_zz: np.ndarray, rcond: float=RCOND) -> np.ndarray`

`W` with `W W^T = K(Z, Z)^+`, from a symmetric eigendecomposition.

The pseudo-inverse is what keeps a landmark set with near-duplicate members from blowing the
approximation up: an eigenvalue below `rcond` of the largest is dropped rather than divided
by.

### `build(landmarks: AtomTable, fieldset: FieldSet, scale: Optional[Sequence[float]]=None, normalize: bool=True, rcond: float=RCOND, kind: str='landmarks') -> LowRank`

The feature map of one landmark set, ready to transform any molecule in the same frame.

### `gram(low: LowRank, table: AtomTable, y: Sequence[float], block: int=BLOCK, progress=None) -> Gram`

Accumulate `Phi^T Phi`, `Phi^T y` and `Phi^T 1` over blocks of molecules.

The features of the whole set are never held at once, which is the point of the item: the
memory of a fit is the rank squared, whatever the number of molecules.

### `fit(g: Gram, lam: float, fit_bias: bool=True) -> Tuple[np.ndarray, float]`

(the weights over the features, the bias) of the ridge with this Gram.

The dual ridge `(K + lam I) a = y` with `K = Phi Phi^T` is the primal ridge
`(Phi^T Phi + lam I) w = Phi^T y` with `w = Phi^T a`: the same model, m by m instead of
n by n.  The bias is the constant the bordered system of the exact solver fits, which here
is one more row and column.

### `predict(Phi: np.ndarray, w: np.ndarray, b: float=0.0) -> np.ndarray`

### `variance(g: Gram, Phi: np.ndarray, lam: float, k_self: Optional[Sequence[float]]=None) -> np.ndarray`

The posterior variance of every row of `Phi`, with the exact self-kernel kept.

`var = k(x,x) - phi^T phi + lam phi^T (Phi^T Phi + lam I)^-1 phi`.  The first two terms
are what the landmarks do not span, and they are why this does not fall to zero far from
them; with `k_self` left out they are dropped and what is left is the plain subset-of-
regressors variance, which does.

### `log_evidence(g: Gram, lam: float, fit_bias: bool=True, profile: bool=True) -> float`

The log marginal likelihood through the Woodbury identity.

The same quantity `cmfqsar.evidence.log_evidence` computes from the n by n kernel,
with `(Phi Phi^T + lam I)^-1 = (I - Phi (Phi^T Phi + lam I)^-1 Phi^T) / lam` and
`log|Phi Phi^T + lam I| = log|Phi^T Phi + lam I| + (n - m) log lam`, so it costs m^3 and
not n^3.  The conventions -- the constant mean marginalised, the amplitude profiled out --
are the exact function's, so the two values may be compared.

### `memory(n: int, m: int, n_fields: int, exact: bool=False) -> float`

Bytes the kernels of a fit need, for the exact form and for this one.

### `cross_validate(low: LowRank, table: AtomTable, y: Sequence[float], lam: float, folds: int=10, seed: Optional[int]=0, fit_bias: bool=True, block: int=BLOCK) -> Tuple[np.ndarray, np.ndarray]`

(out-of-fold predictions, the fold of every molecule), the features built once.

### `landmark_coefficients(low: LowRank, w: Sequence[float]) -> List[np.ndarray]`

The weight the fitted model puts on every landmark, one array per field.

The features are `Phi_f = sqrt(h_f c_f) K(X, Z_f) W_f`, so a fit with weights `w` is the
same model as a dual fit whose coefficients sit on the landmarks:
`sqrt(h_f c_f) W_f w_f`.  Everything the program reads off a co-field -- the map itself,
the atom contributions, the pharmacophore -- keeps its form with the index set changed from
training molecules to landmarks, which is what makes the approximation usable and not only
fast.

### `cofield_gaussians(low: LowRank, w: Sequence[float], f: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`

(centres, coefficients, widths) of the co-field of field `f`, over the landmarks.

The same three arrays `cmfqsar.model.CMFModel.cofield_coefficients` returns for an
exact model, so `cmfqsar.cofields.evaluate_gaussians` draws one as readily as the
other.  A landmark that is a molecule contributes its own atoms; a landmark that is a
Gaussian centre contributes itself.

## cmfqsar.maskedmatch

Source: `cmfqsar/maskedmatch.py`

Scoring a candidate only where the co-field has weight (plan item MC5).

Every other comparison in this program is made over the whole of both molecules.  That is right
for a congeneric series, where the molecules differ in a substituent and agree everywhere else,
and it is wrong for the question a chemist asks when a scaffold has to be replaced: *does this
other skeleton put the same field where the model rewards field?*  What the candidate does
elsewhere is not the question, and an overlap taken over the whole molecule answers mostly with
how similar the skeletons are.

Item E3 measured where that leaves scaffold hopping: the topomer frame is defined against a
scaffold that is held fixed, so replacing the scaffold leaves no frame, and the design loop has
nothing to pose the candidate with.  This module is the construction E3 lacked.  It needs no
frame of the candidate's own beyond a pose in the model's frame, and it asks nothing of the
candidate away from the region the model cares about.

How it works
------------
A **mask** is a soft region of space: a sum of Gaussians on the atoms whose contribution to the
prediction is largest (`cmfqsar.cofields.top_atom_mask`).  Multiplying a co-field by it
gives the co-field inside that region and nothing outside, and because the product of two
Gaussians is a Gaussian the masked co-field is again a sum of Gaussians
(`cmfqsar.kernels.gaussian_product`).  So rule 3 holds: every integral below is closed
form, with no grid and no sampling.

Three numbers come out of it, for every candidate:

* **inside** -- the part of the model's prediction that comes from the masked region,
  `sum_f h_f c_f INT C_f(r) m(r) X_f(r) dr`, in the units of the measured activity.  This is
  the score a replacement scaffold should be ranked by.
* **outside** -- the rest of the prediction, `prediction - bias - inside`.  It is the second
  score the item asks for: field the candidate places where the model rewards none.  A candidate
  whose promise is all outside is a candidate the model is extrapolating about.
* **cosine** -- the same comparison with the size of the molecule divided out.  The mask defines
  an inner product `<f, g>_m = INT f g m`, and this is the cosine of the co-field and the
  candidate's field in it.  A raw overlap grows with the number of atoms, so a ranking by
  `inside` alone prefers large molecules; the cosine does not.

What it does not do
-------------------
It does not pose the candidate.  The molecules handed to it must already stand in the model's
frame -- docked, posed by field similarity, or built by a rule -- exactly as for
`cmfqsar.screen.screen`.  A mask makes the *comparison* local; it does not make the
placement free.

### `class MaskedScore  (dataclass)`

What one candidate collects inside the mask, outside it, and how much of its size that is.

Attributes:

- `name: str`
- `inside: float`
- `outside: float`
- `prediction: float`
- `cosine: float`
- `n_atoms: int` = `0`
- `conformer: int` = `0`

### `class MaskedScorer(model: CMFModel, mask: CofieldMask, fields: Optional[Sequence[int]]=None, tol: float=1e-09)`

A model, a mask, and everything about them that does not depend on the candidate.

The masked co-field is the same for every candidate, and it is the expensive part -- one
product per field over the training atoms -- so it is built once here and reused.  Scoring a
candidate is then two overlaps per field.

#### `MaskedScorer.score(mol: MoleculeFields, name: Optional[str]=None, conformer: int=0) -> MaskedScore`

The three numbers for one molecule, which must already stand in the model's frame.

#### `MaskedScorer.prediction_check(mol: MoleculeFields) -> Tuple[float, float]`

(the primal prediction from these overlaps, the model's own) -- a test of rule 3.

The two must agree: the unmasked overlap summed over the fields *is* the prediction, so a
masked score that is built from the same integrals cannot be measuring something else.

### `score_candidates(model: CMFModel, mols: Sequence[MoleculeFields], mask: CofieldMask, fields: Optional[Sequence[int]]=None) -> List[MaskedScore]`

Every molecule of `mols` scored against the masked co-field, best first by the cosine.

### `screen_conformers(model: CMFModel, mols, mask: CofieldMask, names: Optional[Sequence]=None, fields: Optional[Sequence[int]]=None, rank_by: str='cosine') -> List[MaskedScore]`

The same over a conformer library: every conformation scored, the best one kept.

`mols` are RDKit molecules that may carry several conformations each, already placed in the
model's frame.  A candidate is as good as its best conformation, which is the question a
library asks, and the conformer that wins is reported so that it can be looked at.

## cmfqsar.md

Source: `cmfqsar/md.py`

Short ligand-in-pocket simulations, for plan item D8.

`cmfqsar/trajectory.py` (item D7) reads a simulation as a conformational ensemble. This module
produces the simulations that reader was built for, so that the question of item D8 can be
asked: does the mean field of a ligand's trajectory in its pocket model a series better than the
single crystal pose it started from?

What one run is, and why each choice was made.

**The whole protein, not the trimmed pocket.** `examples/pockets/` holds binding sites cut to
the residues near the ligand. Those files are right for display and for the complementarity
score, and wrong for a simulation: every cut leaves a broken peptide bond that a preparation
tool turns into a charged terminus, so a pocket of twenty residues would carry twenty charges
that the protein does not have. The source entry named in the file's `REMARK 400` line is
downloaded instead and simulated whole.

**The ligands keep their frame.** The free-energy collections give their ligands in the
coordinates of the reference entry, so the protein and the ligands are already in one frame and
nothing is docked or superimposed here. That is what makes the comparison of item D8 fair: the
crystal pose and the trajectory describe the same molecule in the same place.

**GAFF through OpenMM**, with AM1-BCC charges, which is the item's own choice; the protein is
Amber ff14SB and the water TIP3P. Hydrogen mass repartitioning allows a 4 fs step.

**The backbone is restrained lightly** (`restraint_k`, on the alpha carbons) through the
equilibration and released for the production run unless `restrain_production` is set. A few
nanoseconds is not long enough for a protein to need holding, but it is long enough for a
loosely packed crystal structure to drift, and the field model cares about the ligand's position
relative to the site rather than about the protein's own motion.

**Nothing is optimised away.** Every run writes its topology, its trajectory and a record of the
settings, and a run already finished is not repeated, so a set can be built over several
sessions.

OpenMM and the OpenFF/AmberTools stack are not dependencies of this program. They are imported
when a simulation is actually run, and `available` says whether they are there.

### `class MDError  — bases: RuntimeError`

### `class Settings  (dataclass)`

Everything that decides what a run is; written beside the trajectory.

Attributes:

- `nanoseconds: float` = `2.0`
- `equilibration_ps: float` = `200.0`
- `timestep_fs: float` = `4.0`
- `temperature: float` = `300.0`
- `pressure: float` = `1.0`
- `padding: float` = `10.0`
- `ionic_strength: float` = `0.15`
- `frame_ps: float` = `10.0`
- `restraint_k: float` = `DEFAULT_RESTRAINT`
- `restrain_production: bool` = `False`
- `small_molecule_forcefield: str` = `'gaff-2.11'`
- `protein_forcefield: str` = `'amber14-all.xml'`
- `water_forcefield: str` = `'amber14/tip3pfb.xml'`
- `platform: str` = `'CUDA'`
- `seed: int` = `0`

#### `Settings.production_steps` -> int  (property)

#### `Settings.equilibration_steps` -> int  (property)

#### `Settings.frame_steps` -> int  (property)

### `class RunOutcome  (dataclass)`

Attributes:

- `name: str`
- `topology: str` = `''`
- `trajectory: str` = `''`
- `frames: int` = `0`
- `seconds: float` = `0.0`
- `ns_per_day: float` = `0.0`
- `atoms: int` = `0`
- `error: str` = `''`
- `reused: bool` = `False`

### `available() -> Dict[str, bool]`

Which parts of the stack can be imported here.

### `charge_toolkit() -> Optional[str]`

The registered toolkit that can assign AM1-BCC charges, or None.

AmberTools registers itself only when `antechamber` is on PATH, which running the
environment's interpreter by its full path does not do; this is the usual reason a run that
has every package still cannot charge a ligand.

### `require(charges: bool=True) -> None`

### `entry_of(pocket_pdb: str) -> Optional[str]`

The source entry a pocket file of `examples/pockets/` names in its REMARK 400 lines.

### `fetch_entry(code: str, directory: str) -> str`

The PDB entry, downloaded from RCSB once and kept. Entries there are public domain.

### `chains_of(pdb_path: str) -> List[str]`

The chain identifier of every atom record, in file order.

### `chains_near(pdb_path: str, ligands: Sequence[Chem.Mol], radius: float=5.0, min_atoms: int=20) -> List[str]`

The chains of an entry whose atoms line the site the ligands occupy.

A crystal entry often holds several copies of the protein -- 4HW3, the receptor of the mcl1
set, holds twelve -- and simulating all of them costs ten times what the question needs. The
ligands of a free-energy collection are given in the entry's own frame, so the chain they sit
in can be found by distance. A chain is kept when at least `min_atoms` of its atoms are
within `radius` of some ligand atom, which keeps both chains of a site formed by two.

### `prepare_receptor(pdb_path: str, out_path: str, ph: float=7.0, keep_chains: Sequence[str]=()) -> str`

The entry made ready to simulate: one copy, no water, missing atoms built, hydrogens on.

Heteroatoms are dropped, the co-crystallised ligand included: the ligand that is simulated
is the one from the data set, placed in the same frame, and keeping the crystal ligand as
well would put two molecules in one site.

### `build(receptor_pdb: str, ligand: Chem.Mol, settings: Settings)`

(simulation-ready topology, system, positions) for one complex in water.

### `add_restraints(system, topology, positions, k: float)`

Harmonic restraints on the alpha carbons; returns the index of the force that was added.

### `simulate(ligand: Chem.Mol, name: str, receptor_pdb: str, outdir: str, settings: Optional[Settings]=None, log=print) -> RunOutcome`

One ligand: build, minimise, equilibrate with restraints, run, write the trajectory.

### `simulate_set(mols: Sequence[Chem.Mol], names: Sequence[str], receptor_pdb: str, outdir: str, settings: Optional[Settings]=None, log=print) -> List[RunOutcome]`

Every ligand of a set, in order; one that fails is reported and the rest go on.

## cmfqsar.mil

Source: `cmfqsar/mil.py`

Multi-instance learning over conformations: let the model choose the pose.

Averaging a molecule's conformations into one mean field, which is what the 4D mode of
`cmfqsar.conformers` does, assumes that every conformation deserves the same weight.  On an
aligned series that assumption is wrong and measurably harmful: the activity is produced by one
conformation, the bioactive one, and the rest dilute it.

The multi-instance formulation says so directly.  A molecule is a *bag* of conformations, the label
belongs to the bag, and the model decides which conformations in it matter.  Here the decision is a
weight per conformation,

    w_ik = exp(beta * f(x_ik)) / sum_l exp(beta * f(x_il)),

driven by the model's own prediction for that conformation, and the molecule is represented as
before by the weighted mean of its conformational fields.  Two limits bracket everything:

* `beta = 0`   every conformation weighs the same: the ensemble mode exactly.
* `beta -> inf` all the weight goes to the single highest-scoring conformation: hard selection,
  the witness assumption of mi-SVM (Andrews, Tsochantaridis & Hofmann 2002) and of multi-instance
  regression (Ray & Page 2001).

In between the weights are soft, which keeps the whole scheme inside the CMF framework: the bag
kernel stays a weighted mean of conformation kernels,

    K(A, B) = sum_k sum_l w_ak w_bl K(X_A^(k), X_B^(l)) = (W K_conf W^T)_{AB},

so it stays positive definite, the model stays linear in the mean field, and co-fields, the
decomposition rules and the Shapley machinery survive exactly as they do for a fixed weighting.
Hard selection is the one limit that leaves that guarantee intact only because a one-hot weighting
is still a weighting.

Fitting alternates, as multi-instance methods do: fit the regression with the current weights,
score every conformation with it, update the weights, repeat.  The conformation kernel is computed
once and every round after that is matrix algebra on it.  The objective is not convex, so the
result depends on where it starts; the default starts from the pose the molecules arrived in, which
makes the single-conformation model the starting point and the reference to beat.

What this module does *not* do is pretend the choice is free: the weights are fitted to the
training activities, so they must be refitted inside every cross-validation fold, which
`cv_predict` does, and a q2 obtained any other way is not comparable with the
single-conformation one.

### `class MILFit  (dataclass)`

A fitted multi-instance weighting and the model that goes with it.

Attributes:

- `weights: np.ndarray`
- `bags: np.ndarray`
- `beta: float`
- `selection: str`
- `coef: np.ndarray` = `_field(default_factory=lambda: np.zeros(0))`
- `bias: float` = `0.0`
- `rounds: int` = `0`
- `moved: List[int]` = `_field(default_factory=list)`
- `objective: List[float]` = `_field(default_factory=list)`

#### `MILFit.n_bags` -> int  (property)

#### `MILFit.W() -> np.ndarray`

#### `MILFit.chosen() -> np.ndarray`

Index of the largest-weight conformation of every bag: the pose the model picked.

#### `MILFit.effective_counts() -> np.ndarray`

Exponential of the entropy of each bag's weights: how many conformations matter.

### `class MILSearch  (dataclass)`

Attributes:

- `best_beta: float`
- `best_lam: float`
- `best_q2: float`
- `best_rmse: float`
- `table: List[dict]`
- `y_cv: np.ndarray`
- `fit: MILFit`

### `flatten_conformers(mols: Sequence, fieldset: FieldSet, names: Sequence[str], max_conformers: int=0) -> Tuple[List[MoleculeFields], np.ndarray]`

Every conformation of every molecule as its own entry, plus the bag it belongs to.

The field set is used in its single-conformation sense here, whatever its mode: the
purpose is that the conformations are kept apart so that the model can weigh them.

### `weight_matrix(weights: np.ndarray, bags: np.ndarray, n_bags: int) -> np.ndarray`

(n_bags, n_conformations) matrix W with the weights of each bag in its own row.

### `uniform_weights(bags: np.ndarray) -> np.ndarray`

### `first_conformer_weights(bags: np.ndarray) -> np.ndarray`

All the weight on the first conformation of each bag: the pose the molecule arrived in.

### `softmax_weights(scores: np.ndarray, bags: np.ndarray, beta: float) -> np.ndarray`

Weights within each bag from its conformation scores; beta = 0 is uniform.

### `argmax_weights(scores: np.ndarray, bags: np.ndarray) -> np.ndarray`

One-hot weights on the highest-scoring conformation of each bag: hard selection.

### `bag_kernel(K_conf: np.ndarray, W_rows: np.ndarray, W_cols: Optional[np.ndarray]=None)`

Kernel between bags from the kernel between conformations: W K W^T.

### `learn_weights(K_conf: np.ndarray, bags: np.ndarray, y: np.ndarray, beta: float=1.0, lam: float=1.0, rounds: int=10, selection: str='soft', init: str='first', fit_bias: bool=True, method: str='krr', params: Optional[Dict[str, float]]=None, tol: float=1e-06, init_weights: Optional[np.ndarray]=None) -> MILFit`

Alternate between fitting the model and re-weighting the conformations of each bag.

`init` is where the alternation starts: `"first"` puts all the weight on the conformation a
molecule arrived with, so the single-conformation model is the starting point, and
`"uniform"` starts from the plain ensemble.  `init_weights` overrides both, which is what
annealing uses: run with a small beta, hand the weights to a larger one, and so on upwards.
The iteration stops when no bag changes its highest-weighted conformation and the objective
settles.

The objective is not convex and the alternation only finds a local optimum, so where it
starts matters and is worth varying.

### `anneal_weights(K_conf: np.ndarray, bags: np.ndarray, y: np.ndarray, betas: Sequence[float]=(0.5, 1.0, 3.0, 10.0), lam: float=1.0, rounds: int=10, selection: str='soft', init: str='uniform', **kw) -> MILFit`

Raise beta in steps, carrying the weights over, instead of jumping to a hard selection.

The alternation is a local search, and a hard rule applied to a model that has not settled
picks the wrong conformation and then trains itself to believe it.  Starting soft keeps every
conformation as a candidate while the model forms, and sharpening afterwards is the usual
remedy.

### `predict_bags(K_new_conf: np.ndarray, fit: MILFit, bags_new: np.ndarray) -> np.ndarray`

Predict new bags: score their conformations with the fitted model, weigh, then combine.

The weighting rule is the one the model was fitted with, applied once: a new molecule has no
activity to fit its weights to, which is exactly the intention.

### `cv_predict(K_conf: np.ndarray, bags: np.ndarray, y: np.ndarray, beta: float=1.0, lam: float=1.0, folds: int=10, seed: Optional[int]=0, rounds: int=10, selection: str='soft', init: str='first', fit_bias: bool=True, method: str='krr', params: Optional[Dict[str, float]]=None) -> np.ndarray`

Cross-validated predictions with the weights relearned inside every fold.

Splitting is by bag, never by conformation: a molecule is in the training part or in the test
part as a whole, and its weights are decided by a model that never saw its activity.

### `search(K_conf: np.ndarray, bags: np.ndarray, y: np.ndarray, betas: Sequence[float]=(0.0, 1.0, 3.0, 10.0), lams: Sequence[float]=(0.1, 1.0), folds: int=10, seed: Optional[int]=0, rounds: int=10, selection: str='soft', init: str='first', progress=None) -> MILSearch`

Choose beta and lambda by cross-validation, the weights relearned in every fold.

beta = 0 is the plain ensemble and is always in the grid by default, so the search can always
fall back to it; hard selection is reached with a large beta or with `selection="hard"`.

## cmfqsar.mixing

Source: `cmfqsar/mixing.py`

A posterior over the field weights (plan item E6).

The joint kernel is a sum, `K = sum_f h_f c_f K_f`, so choosing the mixing coefficients h_f is
the multiple-kernel-learning problem, and `cmfqsar.model.align_mixing` already solves it by
kernel alignment: one number per field, no statement of how sure it is.  Under the Gaussian
process the same sum has an interpretation that does better.  A sum of kernels is the kernel of a
sum of independent processes,

    f(x) = sum_f f_f(x),    f_f ~ GP(0, h_f c_f K_f),

so h_f is the prior variance of the part of the model carried by field f -- how much of the
activity that field is allowed to explain.  Putting a prior on it and integrating gives a
posterior over the weights, and with it an answer to *does this series care about the steric field
at all* that has an interval on it rather than a point.

The parametrisation is the one the question asks for.  What matters is each field's **share** of
the kernel, not the overall size, which the evidence profiles away anyway: the weights are
written `h = F softmax(u)`, so the shares `softmax(u)` sum to one and can be compared with
1/F, the share a field would have if every field mattered equally.  The prior is
`u_f ~ N(0, sigma^2)` independently, which is exchangeable over the fields -- it prefers no
field over any other -- and weakly informative: sigma = 1.5 keeps a field's share within about a
factor of twenty of the others before the data are taken into account.  Adding a constant to every
u_f leaves the shares alone, so the likelihood is flat along that direction and the prior settles
it at the centred representative, which is to say the prior acts on the *spread* of the log-shares
and not on their level.  The noise level is sampled with them, as `log lambda`, because a model
that is allowed to reweigh its fields and not to adjust its regularisation is answering a different
question.

Three things come out, in increasing order of what they cost:

* `evidence_mixing` -- the weights that maximise the posterior.  A point estimate like the
  alignment one, and a direct replacement for it (`--mkl evidence`).
* `sample_mixing` -- the posterior itself, by slice sampling.  Slice sampling because it
  has no step size to tune: a library that asks its user to tune a sampler will be run with the
  default and the default will be wrong for their data.
* `drop_one` -- what the evidence loses when a field is removed outright, which is the
  Bayes factor against *this field does nothing*.  The sharpest form of the question, and the
  cheapest to answer.

**What this does not promise.**  The evidence is not a held-out score, and item D6 measured what
happens when it is given more freedom over the shape of the kernel: it takes the freedom, its own
number improves, and prediction does not move.  The same caution applies here and
`benchmarks/run_mixing.py` measures it.  What a posterior over the weights is *for* is the
interpretation question -- which fields this series needs -- and that question has no
cross-validated answer to be checked against.

Pooling the weights across series
---------------------------------

A single series of thirty molecules cannot say much about five weights, which is why the
intervals above are wide and why a point estimate fitted on one series is noisy.  Other series
can help.  The prior above is centred at equal shares with one fixed spread; the pooled form
lets other series set both:

    u_s ~ N(mu, tau^2 I)    for every series s,

with `u_s` the centred log-shares of series s, `mu` the log-shares a typical series needs and
`tau` how much series differ from each other.  `mu` and `tau` are chosen by empirical Bayes,
that is, to maximise the probability of all the series together (`pool_priors`).  The
posterior draws of each series under the fixed prior are all that is needed: the probability of
a series under the pooled prior is the average over its draws of the ratio of the two priors
(importance reweighting), so no new sampling is done.  A series then gets its weights from
`evidence_mixing` with `prior_mean=mu` and `prior_sd=tau`: a small series is pulled
towards what the others need, a large series keeps its own weights, and `tau` decides where the
balance lies.  With `tau` near zero the weights are the population's and nothing is fitted on
the series; with `tau` large the pooled prior is the fixed one again.
`benchmarks/run_pooled_mixing.py` measures whether this predicts better, with every series'
prior learned from the other series only.

### `class MixingPosterior  (dataclass)`

The posterior over the share of the kernel each field carries.

Attributes:

- `names: List[str]`
- `samples: np.ndarray`
- `lam: np.ndarray`
- `map_shares: np.ndarray`
- `map_lam: float`
- `log_post: np.ndarray`
- `accepted: int` = `0`
- `note: str` = `''`

#### `MixingPosterior.mean` -> np.ndarray  (property)

#### `MixingPosterior.interval(level: float=0.9) -> np.ndarray`

(F, 2) equal-tailed credible interval of every field's share.

#### `MixingPosterior.carries_less_than_its_share(level: float=0.9) -> List[str]`

Fields whose whole interval lies below 1/F: the series needs them least.

#### `MixingPosterior.describe(level: float=0.9) -> str`

### `class PooledPrior  (dataclass)`

The prior over the log-shares that a collection of series supports.

Attributes:

- `mean: np.ndarray`
- `sd: float`
- `names: List[str]`
- `n_series: int`
- `ess: np.ndarray`
- `log_marginal: float`
- `note: str` = `''`
- `method: str` = `'gaussian'`

#### `PooledPrior.shares` -> np.ndarray  (property)

The shares a typical series gives the fields.

#### `PooledPrior.h() -> np.ndarray`

The population weights in the convention of `evidence_mixing` (mean one).

#### `PooledPrior.describe() -> str`

### `shares_to_h(p: Sequence[float], n_fields: int) -> np.ndarray`

Shares that sum to one, written as weights that average one.

### `centred_log_shares(shares: np.ndarray) -> np.ndarray`

Shares (rows summing to one) as centred log-shares, the coordinates the prior acts on.

### `evidence_mixing(K_fields: np.ndarray, y: np.ndarray, scale: Optional[np.ndarray]=None, enabled: Optional[Sequence[bool]]=None, fit_bias: bool=True, prior_sd: float=PRIOR_SD, maxiter: int=400, prior_mean: Optional[Sequence[float]]=None) -> np.ndarray`

The mixing coefficients that maximise the evidence (with the prior above), on the simplex.

Returned in the same convention as `cmfqsar.model.align_mixing`: one number per field,
zero where the field is disabled, scaled so that the joint kernel has the mean diagonal it
would have with h = 1, which keeps the lambda grid comparable.

`prior_mean` (centred log-shares, one per field) and `prior_sd` together are the pooled
prior of `pool_priors`; by default the prior is centred at equal shares.

### `sample_mixing(K_fields: np.ndarray, y: np.ndarray, names: Optional[Sequence[str]]=None, scale: Optional[np.ndarray]=None, enabled: Optional[Sequence[bool]]=None, draws: int=400, burn: int=100, thin: int=1, seed: int=0, fit_bias: bool=True, prior_sd: float=PRIOR_SD, progress=None, prior_mean: Optional[Sequence[float]]=None) -> MixingPosterior`

The posterior over the fields' shares, by slice sampling the log-shares and log lambda.

### `drop_one(K_fields: np.ndarray, y: np.ndarray, names: Optional[Sequence[str]]=None, scale: Optional[np.ndarray]=None, enabled: Optional[Sequence[bool]]=None, fit_bias: bool=True, prior_sd: float=PRIOR_SD) -> Dict[str, float]`

How much log evidence the model loses when each field is taken away entirely.

The Bayes factor against *this field does nothing*: a large positive number means the
activities are much more probable with the field than without it, and a number near zero --
or below it -- means the field is carrying nothing the others do not already carry. This is
the sharpest form of the question and the cheapest to answer, since it needs one optimisation
per field rather than a posterior.

### `pool_priors(shares_by_series: Sequence[np.ndarray], names: Optional[Sequence[str]]=None, prior_sd: float=PRIOR_SD, sd_range: Tuple[float, float]=POOLED_SD_RANGE, method: str='gaussian', min_ess: float=MIN_ESS, maxiter: int=2000) -> PooledPrior`

The population prior `N(mu, tau^2 I)` over the centred log-shares, by empirical Bayes.

`shares_by_series` holds, for every series, the posterior draws of its shares under the
fixed prior (`MixingPosterior.samples`), all with the same fields.  `mu` and `tau` are
chosen to maximise the probability of all the series together, each series' probability
being its likelihood integrated against the candidate prior.  Two series are enough to run
it; the result is only as good as the number of them.

Two ways to get at a series' likelihood from its draws:

* `method="gaussian"` (the default) summarises the draws by their mean and covariance,
  removes the fixed prior's precision, and treats what is left as a Gaussian likelihood.
  The integral against the candidate prior is then in closed form.  This is the
  random-effects model of meta-analysis, with each series' weights as the estimate and its
  posterior covariance as the uncertainty.  A direction in which the series says nothing
  has zero precision and contributes nothing, as it should.
* `method="importance"` takes the log of the average, over the draws, of the ratio of the
  candidate prior to the fixed one (importance reweighting), and needs no Gaussian
  assumption.  A narrow candidate prior gives almost all the weight to the few draws nearest
  its mean, and the optimiser, left alone, finds such a prior and reports a spread far too
  small.  So every series must keep an effective sample size (the number of equally
  weighted draws its average is worth) of at least `min_ess`.  With a few hundred draws
  this stops the spread at around a fifth of `prior_sd`, and the result is noisier than
  the Gaussian one; it is kept as the check that the Gaussian assumption is not doing the
  work.

The draws must come from the prior of width `prior_sd` centred at equal shares, which is
what `sample_mixing` uses by default.

## cmfqsar.mmd

Source: `cmfqsar/mmd.py`

Is this set of molecules like that one?  A two-sample test in the field Hilbert space.

The applicability domain of `cmfqsar.validation` answers the question one molecule at a
time: how close is *this* compound to the training set.  It cannot answer the question a
screening run actually raises, which is about the library as a whole -- are these molecules
drawn from the same population as the ones the model was fitted on, or is the whole exercise an
extrapolation?  Averaging a per-molecule verdict does not answer it either: two sets can have
identical nearest-neighbour similarities and still occupy different regions.

The maximum mean discrepancy (Gretton et al., JMLR 13 (2012) 723) is the natural answer when a
kernel is already available.  Every set of molecules has a mean field -- the mean of its feature
maps, which for CMF is a function of space in its own right -- and the MMD is the distance
between the two mean fields:

    MMD^2(X, Y) = ||mean_i phi(x_i) - mean_j phi(y_j)||^2
                = mean K(x,x') - 2 mean K(x,y) + mean K(y,y')

so it is three averages over blocks of the kernel matrix and nothing else.  The unbiased
estimator drops the diagonal terms, which is what lets it be negative on sets that are really
the same -- a feature, not a defect: a small negative value is how "no difference at all" looks
when the diagonal bias has been removed.

Because the kernel is characteristic in practice, MMD = 0 only when the two field distributions
coincide, so the test has power against differences no single summary statistic would show.  The
null distribution has no closed form worth using at these sample sizes, so it is obtained by
permutation: shuffle the labels, recompute, repeat.  That costs three block means per shuffle
and nothing else, since the kernel is computed once.

The **witness** says *where* the difference lies.  It is the function whose expectation separates
the two sets, w(x) = mean_i K(x, x_i) - mean_j K(x, y_j), evaluated at each molecule: large and
positive for molecules typical of the first set, large and negative for the second.  Ranking a
library by it names the compounds that make it unlike the training set, which is the usable
form of a p-value.

### `class MMDResult  (dataclass)`

The statistic, its null distribution and what it means.

Attributes:

- `mmd2: float`
- `mmd: float`
- `p_value: float`
- `n_x: int`
- `n_y: int`
- `permutations: int`
- `null_mean: float` = `float('nan')`
- `null_sd: float` = `float('nan')`
- `z: float` = `float('nan')`

#### `MMDResult.describe() -> str`

#### `MMDResult.as_dict() -> Dict[str, float]`

### `mmd2(K: np.ndarray, mask: Sequence[bool], unbiased: bool=True) -> float`

MMD^2 between the molecules `mask` selects and the rest, from their kernel matrix.

The unbiased estimator leaves the self-similarities out of the two within-set means, so that
it estimates the squared distance of the mean fields rather than that distance plus a bias
that falls off only as 1/n.  On two samples of the same distribution it is centred on zero
and takes either sign.

### `mmd_test(K: np.ndarray, mask: Sequence[bool], permutations: int=999, seed: int=0, unbiased: bool=True) -> MMDResult`

Permutation test of MMD^2: how often does a reshuffling of the labels do as well?

The kernel is computed once and every permutation is three block means over it, so the test
costs a fraction of the kernel it is built on.  The p-value is (1 + #{null >= observed}) /
(1 + permutations), which never returns zero -- a p-value of 0 would claim more than a finite
number of shuffles can support.

### `witness(K: np.ndarray, mask: Sequence[bool]) -> np.ndarray`

w(x) = mean K(x, first set) - mean K(x, second set), one value per molecule.

Positive where a molecule looks like the first set and negative where it looks like the
second; its largest magnitudes name the compounds that make the two sets differ, which is
what a p-value cannot tell you.

### `assemble(Kxx: np.ndarray, Kyx: np.ndarray, Kyy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`

One kernel matrix and its mask from the three blocks a fitted model can already give.

`Kyx` is the (new, train) block a prediction computes anyway, so comparing a library with a
model's training set costs only the library's own kernel on top of the prediction.

### `kernel_over(mols_a, names_a, mols_b, names_b, fieldset, normalize: bool=True)`

The joint CMF kernel over both sets at once, and the mask selecting the first.

One table of both sets is built so that every molecule is parameterised identically and the
kernel is the one a model would use; comparing sets with kernels built separately would
compare their parameterisations as much as their chemistry.

## cmfqsar.model

Source: `cmfqsar/model.py`

CMF regression models: kernel ridge regression (KRR, the method of the 2013
paper), kernel partial least squares (KPLS, the kernel form of the PLS that
CoMFA uses), support vector regression and a Gaussian-process view of KRR that
provides predictive variances.  The classifiers are here too: the support-vector
classifiers, the Laplace Gaussian-process classifier of `cmfqsar.gpc`,
kernel logistic regression (the same mode with its ridge chosen on the folds)
and the kernel Fisher discriminant (KFDA), whose score is linear in the field
and which therefore has a co-field.  Includes k-fold cross-validation, grid search
over hyper-parameters (alpha, lambda / C / number of PLS components, optionally
mixing coefficients h_f) and external (nested) cross-validation.

### `class ModelStats  (dataclass)`

Attributes:

- `n_train: int` = `0`
- `r2: float` = `float('nan')`
- `rmse_fit: float` = `float('nan')`
- `q2: float` = `float('nan')`
- `rmse_cv: float` = `float('nan')`
- `n_test: int` = `0`
- `r2_pred: float` = `float('nan')`
- `r2_pred_train_mean: float` = `float('nan')`
- `r2_pred_train_var: float` = `float('nan')`
- `rmse_pred: float` = `float('nan')`
- `q2_ex: float` = `float('nan')`
- `rmse_ex: float` = `float('nan')`
- `q2_ex_sd: float` = `float('nan')`
- `rmse_ex_sd: float` = `float('nan')`
- `q2_ex_aggr: float` = `float('nan')`
- `rmse_ex_aggr: float` = `float('nan')`
- `cv_folds: int` = `0`
- `rank_fit: Dict[str, float]` = `field(default_factory=dict)`
- `rank_cv: Dict[str, float]` = `field(default_factory=dict)`
- `rank_test: Dict[str, float]` = `field(default_factory=dict)`
- `extra: Dict[str, object]` = `field(default_factory=dict)`

#### `ModelStats.summary() -> str`

### `class ClassStats  (dataclass)`

Statistics of a classification (binary) or one-class model.

Attributes:

- `task: str` = `'classification'`
- `n_train: int` = `0`
- `n_test: int` = `0`
- `fit: Dict[str, float]` = `field(default_factory=dict)`
- `cv: Dict[str, float]` = `field(default_factory=dict)`
- `test: Dict[str, float]` = `field(default_factory=dict)`
- `cv_folds: int` = `0`
- `inlier_fraction: float` = `float('nan')`
- `n_support: int` = `0`
- `screening: Dict[str, float]` = `field(default_factory=dict)`
- `extra: Dict[str, object]` = `field(default_factory=dict)`

#### `ClassStats.summary() -> str`

### `class KFDACalibration  (dataclass)`

Where a kernel Fisher discriminant cuts its score, and how it turns it into a probability.

Both are fitted on scores of molecules the discriminant that scored them had not seen, so
neither carries the optimism of a threshold read off the training scores.  Both are also
expressed in the class-mean units of `kfda_frame` rather than in raw score units: the
scaled score is (raw score - `centre`) / `spread`, which is 0 at the midpoint of the two
class means and +-1 at the means themselves.

Attributes:

- `threshold: float` = `0.0`
- `slope: float` = `1.0`
- `intercept: float` = `0.0`
- `folds: int` = `0`
- `balanced_accuracy: float` = `float('nan')`
- `scores: Optional[np.ndarray]` = `None`
- `centre: float` = `0.0`
- `spread: float` = `1.0`

#### `KFDACalibration.scale(score: np.ndarray) -> np.ndarray`

Raw scores K(x, train) a expressed in the class-mean units the threshold is kept in.

#### `KFDACalibration.bias() -> float`

The bias b that puts the chosen threshold at zero, so that the class is sign(K a + b).

#### `KFDACalibration.probability(score: np.ndarray) -> np.ndarray`

p(active) for raw scores K(x, train) a, that is for predictions with the bias removed.

#### `KFDACalibration.describe() -> str`

### `class CMFModel  (dataclass)`

A fitted (or to-be-fitted) CMF 3D-QSAR model.

Attributes set by `fit`:
  train      - AtomTable of the training set
  y          - training activities
  coef       - dual coefficients a_j
  bias       - b
  self_diag  - K_f(j, j) for training molecules (for normalised kernels)
  scale      - per-field scaling constants c_f (unit mean diagonal) or ones

Attributes:

- `fieldset: FieldSet`
- `task: str` = `'regression'`
- `method: str` = `'krr'`
- `params: Dict[str, float]` = `field(default_factory=lambda: {'lam': 1.0, 'C': 1.0, 'epsilon': 0.1, 'nu': 0.5, 'components': 5})`
- `normalize: bool` = `False`
- `scale_fields: bool` = `True`
- `cutoff: Optional[object]` = `None`
- `engine: Optional[str]` = `None`
- `kernel_source: str` = `'fields'`
- `ac_sigma: float` = `0.75`
- `ac_d_max: float` = `12.0`
- `ac_chirality: bool` = `False`
- `ac_chirality_shells: int` = `4`
- `surf_z_min: float` = `1.0`
- `surf_z_max: float` = `6.0`
- `surf_z_step: float` = `1.0`
- `pbc_background: bool` = `True`
- `pbc_taper: float` = `2.0`
- `pbc_z_max: float` = `0.0`
- `pbc_z_step: float` = `1.0`
- `pbc_oriented: bool` = `False`
- `rec_q_max: float` = `8.0`
- `rec_step: float` = `0.25`
- `rec_form_sigma: float` = `0.3`
- `rec_self: bool` = `True`
- `tb_r_max: float` = `4.0`
- `tb_step: float` = `0.5`
- `tb_sigma: float` = `0.35`
- `tb_chirality: bool` = `False`
- `tb_chirality_shells: int` = `4`
- `pbc_overlap: bool` = `False`
- `pbc_overlap_r_max: Optional[float]` = `None`
- `pbc_symmetry: Optional[Tuple[np.ndarray, np.ndarray]]` = `None`
- `ac_block_weights: Union[None, str, Dict[str, float]]` = `None`
- `ac_block_groups: Optional[Dict[str, str]]` = `None`
- `ac_block_rho: float` = `1.0`
- `kernel_layer: str` = `'linear'`
- `mkl: Optional[str]` = `None`
- `mkl_rho: float` = `1.0`
- `fit_bias: bool` = `True`
- `alpha_grid: Optional[List[float]]` = `None`
- `train: Optional[AtomTable]` = `None`
- `y: Optional[np.ndarray]` = `None`
- `coef: Optional[np.ndarray]` = `None`
- `bias: float` = `0.0`
- `self_diag: Optional[np.ndarray]` = `None`
- `scale: Optional[np.ndarray]` = `None`
- `K_fields_train: Optional[np.ndarray]` = `None`
- `stats: ModelStats` = `field(default_factory=ModelStats)`
- `learner_: Optional[_Learner]` = `None`
- `desc_scaler_: Optional[object]` = `None`
- `desc_train_: Optional[np.ndarray]` = `None`
- `desc_scale_: float` = `1.0`
- `provenance: Dict[str, object]` = `field(default_factory=dict)`
- `file_version: int` = `0`

#### `CMFModel.uses_fields` -> bool  (property)

True when the kernel is the overlap integral, which is what needs a common frame.

#### `CMFModel.fresh_copy() -> 'CMFModel'`

An unfitted model with every setting of this one: the field set (copied), learner,
parameters, normalisation, bias, kernel layer and kernel source with its widths,
heights and block weights.  The nested cross-validation builds its fold models from
this, so that a model on an invariant kernel is validated as the kernel it is.

#### `CMFModel.on_surface` -> bool  (property)

True for the surface kernel: in-plane invariant, height kept.

#### `CMFModel.is_periodic` -> bool  (property)

True for the periodic kernel: crystals, summed over the neighbour list of the cell.

#### `CMFModel.is_reciprocal` -> bool  (property)

True for the reciprocal-space channels: the typed structure factors of a crystal.

#### `CMFModel.is_threebody` -> bool  (property)

True for the three-body channels of a crystal: the triangles around every atom.

#### `CMFModel.block_weights_aligned` -> bool  (property)

True when the block weights come from the activities by kernel alignment, and so
have to be re-derived inside every cross-validation fold.

#### `CMFModel.ac_column_scale` -> np.ndarray  (property)

What every channel column is multiplied by before the kernel is taken: the
root-mean-square scale of its block times the block's mixing coefficient.

#### `CMFModel.block_weight_report() -> List[Dict[str, float]]`

The fitted block weights by kind of block (see `invariant.block_weight_summary`);
empty for a model whose blocks all weigh one.

#### `CMFModel.training_kernel(train: AtomTable, y: Optional[np.ndarray]=None, progress=None, cancel=None, reuse: bool=False) -> np.ndarray`

The joint training kernel, whichever source it comes from.

`y` is needed by the invariant sources when the block weights are set by kernel
alignment.  With `reuse` the channel features of a table this model has already
featurised are kept; the scale and the block weights are always derived afresh.

#### `CMFModel.test_kernel(test: AtomTable, progress=None, cancel=None) -> np.ndarray`

The (n_test, n_train) joint kernel block, whichever source it comes from.

#### `CMFModel.joint_kernel(K_fields: np.ndarray, desc: Optional[np.ndarray]=None) -> np.ndarray`

Linear (overlap-integral) joint kernel: the sum over field types.

`desc` holds the raw descriptor values of the molecules the *rows* belong to, when the
field set names non-field descriptors.  They enter as one more block, standardised with
the training set's statistics and scaled like a field; passing None leaves the kernel as
the fields alone, which is what the callers that only want a field kernel do.

#### `CMFModel.descriptor_block(desc: Optional[np.ndarray]) -> float`

h_d c_d z(rows) . z(training), or 0.0 when there are no descriptors to add.

#### `CMFModel.descriptor_contributions(mol: 'MoleculeFields') -> List[Tuple[str, float]]`

What each non-field descriptor contributed to one molecule's prediction.

The block is a dot product, so its share of the prediction splits exactly into one term
per descriptor, and the terms sum to the block's whole contribution.  Reported beside the
atom contributions, this is what shows at a glance whether a model rests on its
geometry or on a descriptor that happens to follow the activity.

#### `CMFModel.descriptor_self(desc: Optional[np.ndarray]) -> float`

The descriptor block's own diagonal, which the variance and one-class scores need.

#### `CMFModel.is_linear` -> bool  (property)

True when the kernel is the plain overlap integral, so co-fields are meaningful.

#### `CMFModel.joint_self(table: AtomTable) -> np.ndarray`

Diagonal of the linear joint kernel, i.e. K(x, x) for every molecule of `table`.

#### `CMFModel.self_after_layer(d_linear: np.ndarray) -> np.ndarray`

K(x, x) of the kernel the learner actually sees, from the linear self-kernels.

The distances of the one-class scores are taken in the Hilbert space of the kernel in
use, so they need the self-kernel of that kernel: for the Gaussian layer every molecule
sits on the unit sphere, and the arc-cosine layers leave the self-kernel where it was
(which is what makes the same diagonals serve both of them).

#### `CMFModel.apply_layer(K: np.ndarray, diag_p: np.ndarray, diag_q: np.ndarray) -> np.ndarray`

Apply the nonlinear layer to a block of the linear kernel.



    gauss    : exp(-gamma d^2), a radial basis function in the Hilbert space of the fields.
    arccos1/2: the degree-1 arc-cosine kernel of Cho & Saul (2009), the kernel of an infinitely
               wide one- (or two-) hidden-layer network; its self-kernel equals the base one,
               so the same diagonals serve both layers.

#### `CMFModel.layered_kernel(K_fields: np.ndarray, diag_p: np.ndarray, diag_q: np.ndarray) -> np.ndarray`

#### `CMFModel.prepare_training_kernels(train: AtomTable, progress=None, cancel=None) -> np.ndarray`

Compute (and cache) the per-field training kernel matrices for the current alphas.

#### `CMFModel.prepare_training_kernels_grid(train: AtomTable, alphas: Sequence[float], alpha_fields: Optional[Sequence[str]]=None, progress=None, cancel=None) -> Optional[Dict]`

Training kernels for a whole grid of attenuations, sharing the distances (item B1).

Returns `{alpha: (K_fields, self_diag)}` in the form `grid_search` accepts as
`precomputed`, or None when the shortcut does not apply: a kernel that is not the
overlap integral has no attenuation to sweep, and an attenuation that has to be
interpolated between grid points is two field sets rather than one, which the plain path
already handles.

The squared interatomic distances do not depend on the attenuation, so one pass over the
molecules serves every alpha.  The saving is about a quarter of the time on a six-point
grid; the larger part of the kernel work is the exponential, which has to be done per
alpha.

#### `CMFModel.prepare_descriptors(train: AtomTable) -> None`

Fit the descriptor statistics on the training molecules and scale the block.

Both belong to the training set and are stored with the model: standardising or rescaling
according to the molecules being predicted would score two test sets on two scales. The
scale puts the block's mean diagonal at 1, which is what `scale_fields` does to a field,
so that a descriptor block with ten descriptors does not outweigh the geometry merely by
being longer.

#### `CMFModel.set_training_kernels(train: AtomTable, K_fields: np.ndarray, self_diag: Optional[np.ndarray]=None) -> np.ndarray`

Install precomputed per-field training kernels (already normalised if `normalize`).

#### `CMFModel.fit(train: AtomTable, y: np.ndarray, cv_folds: int=10, cv_seed: Optional[int]=0, progress: Optional[ProgressFn]=None, cancel: Optional[CancelFn]=None, reuse_kernels: bool=False) -> 'CMFModel'`

#### `CMFModel.calibrate_conformal(level: float=0.9, normalise: bool=False, beta: float=1.0)`

Calibrate a prediction interval on the errors of the cross-validation already run.

Nothing is recomputed: `fit` leaves the out-of-fold predictions on the statistics,
and their absolute errors are exactly the nonconformity scores conformal prediction asks
for.  The guarantee that comes with them is approximate rather than exact, because those
errors come from models fitted on a fraction of the training set while the predictions
will come from the model fitted on all of it; `benchmarks/run_conformal.py` measures how
much that costs, and on the Sutherland sets the answer is nothing measurable.

With `normalise` the errors are divided by the distance of their molecule from the rest
of the training set, so that the interval widens where the model is extrapolating.  It
is off by default because, measured, it gains no sharpness on these sets.

#### `CMFModel.predict_interval(test: AtomTable, progress=None, cancel=None)`

(prediction, lower, upper) for new molecules, using the stored calibration.

#### `CMFModel.test_kernels(test: AtomTable, progress=None, cancel=None) -> np.ndarray`

#### `CMFModel.fit_low_rank(train: AtomTable, y: Sequence[float], landmarks: AtomTable, lam: Optional[float]=None, kind: str='landmarks', folds: int=10, seed: Optional[int]=0, block: int=256, progress=None) -> 'CMFModel'`

Fit in the space of the landmarks instead of the space of the molecules (item B6).

The model keeps the landmarks and one weight per feature instead of one coefficient per
training molecule, and everything a prediction needs follows from those: see
`cmfqsar.lowrank`.  The training table is still kept, because it is linear in the
number of molecules and the provenance and the contract are read off it; what is never
built is the n by n kernel.

#### `CMFModel.is_low_rank` -> bool  (property)

True when this model was fitted in the space of its landmarks.

#### `CMFModel.low_rank_features(test: AtomTable, block: int=256, progress=None) -> np.ndarray`

#### `CMFModel.predict(test: AtomTable, return_fields: bool=False, progress=None, cancel=None)`

#### `CMFModel.predict_from_field_kernels(K_fields_test: np.ndarray) -> np.ndarray`

Prediction from precomputed per-field kernels (F, n_test, n_train) with the training set.

#### `CMFModel.predict_with_covariance(test: AtomTable) -> Tuple[np.ndarray, np.ndarray]`

Prediction and the whole posterior covariance over the candidates (plan item MC2).

`predict_with_variance` gives the diagonal of the same matrix.  The rest of it is
what makes a batch more than a list: two candidates the model is equally unsure about
may be unsure about *the same thing*, and only the covariance says so.

#### `CMFModel.predict_with_variance(test: AtomTable) -> Tuple[np.ndarray, np.ndarray]`

Prediction and GP posterior variance (only meaningful for method 'gp'/'krr').

#### `CMFModel.surrogate()`

The Gaussian-process reading of this fit, with its scale in the units of the activity.

Plan item MS1.  It is kept on the model, because a library is predicted in blocks and the
factorisation of the training kernel is the whole cost; a refit discards it.

#### `CMFModel.predict_with_sd(test: AtomTable, with_noise: bool=True) -> Tuple[np.ndarray, np.ndarray]`

(prediction, predictive standard deviation) in the units of the activity.

The prediction is the model's own.  The standard deviation is the posterior deviation of
`predict_with_variance` put into the units of `y` by the noise scale of the fit,
with the uncertainty of the constant mean added; with `with_noise` it is the deviation
of a new measurement, which is what a tolerance on the answer should be compared with.

#### `CMFModel.calibration(folds: int=10, seed: Optional[int]=0)`

Whether the predictive standard deviation means what it says, on the folds.

Returns a `cmfqsar.surrogate.Calibration`: the root mean square of the
standardised out-of-fold residual (one when calibrated), the shares inside one and two
standard deviations, and the factor that would correct the deviation.

#### `CMFModel.rank_score(test: AtomTable, beta: float=0.0, larger_is_better: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`

(score to rank by, prediction, posterior standard deviation) for new molecules.

With `beta` at zero the score is the prediction itself, which is what ranking a library
by a regression model has always meant here.  Above zero it is a confidence bound,

    score = mean - beta * sd        (or mean + beta * sd when smaller is better)

so a molecule is promoted only if the model is both optimistic *and* sure about it.  The
point is not caution for its own sake: a screening library is by construction dominated
by molecules unlike the training set, a kernel method returns the training mean presented
as a prediction for those, and a point estimate cannot tell that answer from a real one.
Subtracting the spread puts the molecules the model knows nothing about where they belong.

This is the same quantity the applicability domain reports and the one-class Gaussian
process already divides by; `beta` is the only choice, and one standard deviation is the
conventional starting point.  It needs the linear kernel, since that is where a
posterior exists.

#### `CMFModel.predict_proba(test: AtomTable) -> np.ndarray`

p(active) for new molecules; the two probabilistic classifiers provide it.

For the Gaussian-process classifier the latent posterior variance is averaged over, so a
molecule the model has nothing to say about is reported near one half rather than
confidently.  For the kernel Fisher discriminant the probability is the logistic map that
was fitted on the out-of-fold discriminant scores, which has no such term: it reports the
rate of actives seen at this score during cross-validation, wherever the molecule lies.

For the discriminant, `p >= 0.5` and the reported class are two different cuts, and on
a set with many more inactives than actives they are far apart.  p = 0.5 is the score at
which actives and inactives were equally frequent; the class comes from the threshold
that maximises balanced accuracy, which weighs the two classes equally and therefore
sits lower.  Read the probability as a frequency and the class as a decision.

#### `CMFModel.log_marginal_likelihood` -> float  (property)

Approximate evidence of the fitted classifier; nan for every other method.

#### `CMFModel.evaluate_test(test: AtomTable, y_test: np.ndarray, progress=None, cancel=None) -> np.ndarray`

#### `CMFModel.training_contributions(mol: MoleculeFields) -> np.ndarray`

Decomposition rule 2: contribution a_j K(X, X_j) of every training molecule.

Valid for any kernel, including a nonlinear layer.

#### `CMFModel.atom_contributions(mol: MoleculeFields, per_field: bool=False) -> np.ndarray`

Decomposition rule 3 (Eq. 14 of Sitnikov et al. 2015): contribution of every atom.

Returns (N,) or, if per_field, (F, N).

Defined for the overlap-integral kernel, where the decomposition into atom pairs is a
property of the kernel, and for the rotation-invariant autocorrelation kernel, whose
channels are also sums over atom pairs (plan item SC6,
`cmfqsar.invariant.atom_contributions`).  Neither survives a nonlinear layer.

#### `CMFModel.invariant_pair_contributions(mol: MoleculeFields) -> List[Tuple[str, float]]`

What each pair of field types contributed to one prediction (plan item SC6).

#### `CMFModel.cofield_basis(f: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`

The Gaussians of every training molecule's field f, before the dual coefficients.

Returns (centres, coefficients, widths, molecule index): the j-th training molecule
contributes `x_j(r) = s_fj sum_m w_fjm exp(-b_jm/2 |r - r_jm|^2)`, so that the co-field
is `C_f(r) = sum_j a_j x_j(r)` and anything else linear in the same basis -- the
variance of the co-field under noise in the activities, say (`cmfqsar.cofields`) --
is built from the same numbers rather than from a second copy of them.

#### `CMFModel.training_field_basis(f: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`

The same four arrays as `cofield_basis`, with no condition on the kernel layer.

The basis describes the normalised training fields, not the model, so it exists for
every model that has fields.  `cmfqsar.nonlinearity` builds the local co-field of
a model with a nonlinear layer from it; `cofield_basis` is this method behind
the check that the ordinary co-field is defined.

#### `CMFModel.cofield_coefficients(f: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`

Centres, coefficients and widths of the Gaussians composing co-field C_f(r):

    C_f(r) = sum_j a_j s_fj sum_m w_fjm exp(-b_jm/2 |r - r_jm|^2),  (Eq. 9)

with s_fj = 1/sqrt(K_f(j,j)) when kernels are normalised, else 1.
The field-scaling constant c_f and mixing coefficient h_f are *not* included.
For a model whose alpha lies between the points of `alpha_grid` (interpolated
kernels) the co-field is evaluated with the nominal alpha, an approximation.

#### `CMFModel.field_scale_for(f: int, mol: Optional[MoleculeFields]=None) -> float`

Factor multiplying the raw molecular field X_f of `mol` in the primal form (Eq. 7).

#### `CMFModel.describe_training(model: 'CMFModel') -> dict`  (static method)

How the training set was prepared, read off the stored training molecules.

This is what the model file records as its provenance (version 2 files), and what the
contract check falls back to for a file that recorded nothing.

#### `CMFModel.state() -> dict`

#### `CMFModel.from_state(st: dict) -> 'CMFModel'`  (class method)

### `class SearchResult  (dataclass)`

Attributes:

- `best_alpha: float`
- `best_params: Dict[str, float]`
- `best_h: Dict[str, float]`
- `best_rmse: float`
- `best_q2: float`
- `table: List[dict]`
- `y_cv: Optional[np.ndarray]` = `None`
- `best_alphas: Optional[Dict[str, float]]` = `None`
- `per_field: Optional[object]` = `None`

### `class NestedCVResult  (dataclass)`

Attributes:

- `q2_ex: float`
- `rmse_ex: float`
- `q2_ex_sd: float`
- `rmse_ex_sd: float`
- `y_pred: np.ndarray`
- `per_repeat: List[dict]`
- `q2_ex_aggr: float` = `float('nan')`
- `rmse_ex_aggr: float` = `float('nan')`

### `r2_score(y: np.ndarray, yp: np.ndarray, y_ref_mean: Optional[float]=None) -> float`

### `rmse(y: np.ndarray, yp: np.ndarray) -> float`

### `mae(y: np.ndarray, yp: np.ndarray) -> float`

The mean absolute error.

### `classification_scores(y_true: np.ndarray, decision: np.ndarray) -> Dict[str, float]`

Metrics of a binary classifier from its decision values.

`y_true` holds +1 for the positive (active) class and -1 for the negative one;
`decision` is the signed decision function, whose sign is the predicted class.

### `rie_bedroc(ranks: np.ndarray, n: int, n_act: int, alpha: float=20.0) -> Tuple[float, float]`

Robust initial enhancement and BEDROC of Truchon and Bayly (2007).

The enrichment factor counts the actives above a cut and ignores where they are; RIE
weights every active by exp(-alpha r / N), so a hit at rank 1 is worth more than one at rank
50, and BEDROC rescales RIE onto [0, 1] given the active rate.  `alpha` sets how early
"early" is: 20 puts 80 % of the weight in the top 8 % of the ranking, 160.9 in the top 1 %.

### `screening_scores(y_true: np.ndarray, decision: np.ndarray, fractions: Sequence[float]=(0.005, 0.01, 0.05, 0.1), alphas: Sequence[float]=(20.0, 160.9)) -> Dict[str, float]`

Virtual-screening metrics of a ranking: AUC and enrichment at the top of the list.

`y_true` is +1 for the actives and -1 for the background (decoys), `decision` the score the
library is ranked by.  The enrichment factor at a fraction f is the share of actives found in
the top f of the ranking divided by f, so 1.0 is random and 1/(active rate) the maximum.

A screening result is a property of the background as much as of the model: decoys that differ
grossly from the actives (in size, charge or shape) inflate every number here, which is what
property-matched sets such as DUD-E, DEKOIS or MUV exist to avoid.

### `enrichment_curve(y_true: np.ndarray, decision: np.ndarray)`

(fraction of the library screened, fraction of the actives found) of a ranking.

### `krr_fit(K: np.ndarray, y: np.ndarray, lam: float, fit_bias: bool=True) -> Tuple[np.ndarray, float]`

Kernel ridge regression with a bias term, Eq. 12 of Baskin & Zhokhova (2013):

    [ K + lam I   1 ] [ a ]   [ y ]
    [ 1^T         0 ] [ b ] = [ 0 ]

With `fit_bias=False` the plain KRR system (K + lam I) a = y is solved and b = 0
(the `set_b_0` option of the original R scripts).

### `kpls_fit(K: np.ndarray, y: np.ndarray, components: int) -> Tuple[np.ndarray, float, int]`

Kernel partial least squares (Rosipal & Trejo 2001) with one response.

The kernel is centred in the field space and the activity is centred; every component is
one NIPALS step (one response needs no iteration), after which the kernel and the activity
residual are deflated by that component's score.  The dual coefficients are
a = U (T' Kc U)^-1 T' yc on the centred kernel Kc, and are then folded, together with the
centring, into coefficients on the *uncentred* kernel plus a bias, so that the prediction
is K(x, train) a + b exactly as for every other learner here.  The coefficients sum to
zero, as those of kernel ridge regression with a bias term do.

With the linear CMF kernel this is PLS regression in the space of the fields, that is CoMFA's
learner with the grid replaced by the continuous field.  Returns (a, b, components used);
fewer components than asked for are used when the residual or the next score vanishes.

### `lapkrr_fit(K: np.ndarray, y: np.ndarray, lam: float, manifold: float=1.0, neighbours: int=LAPKRR_NEIGHBOURS, fit_bias: bool=True) -> Tuple[np.ndarray, float]`

Kernel ridge regression with a smoothness penalty over labelled *and* unlabelled molecules.

Manifold regularisation (Belkin, Niyogi & Sindhwani, J. Mach. Learn. Res. 7 (2006)
2399-2434).  A library being screened, a set of decoys, a batch not yet measured: these
molecules have fields but no activity, and plain kernel ridge regression cannot use them at
all.  Here they enter through a second penalty,

    gamma g' L g,        g = K a,

where g holds the predicted values at all n molecules without the bias term, and L is the
normalised Laplacian of the graph of field similarity
(`cmfqsar.spectral.knn_laplacian`).  That term is small when the prediction changes
slowly between molecules that are close in field space, whether or not they are labelled, so
the unlabelled molecules say where the prediction may vary and where it may not.  Minimising
it together with the squared error on the labelled molecules and the usual ridge gives one
dual solve over all n molecules,

    (J K + lam I + gamma L K) a = J y,

where J is the diagonal indicator of the molecules that have an activity.  With a bias term
the stationarity conditions add one row and one column, exactly as in `krr_fit`.

**The bias stays outside the penalty**, which is what `g = K a` says above.  Adding the
same constant to every activity does not make a model vary faster, so a bias should not be
charged for.  The distinction matters here, because `L 1` is not zero for a normalised
Laplacian, while it is zero for the unnormalised one.  A bias inside the penalty would
therefore be charged for being a constant, and the dual above would carry a further
`gamma L 1 b` term in its extra column.

**The activities of the unlabelled molecules are passed as nan.**  A learner that needs more
than (K, y) is new in this program, and the alternative -- a second kernel block handed to
the learner on the side -- would have put the unlabelled molecules outside the training
table, where the co-field, the molecule contributions and the atom contributions could not
see them.  The expansion `f(x) = sum_j a_j K(x, x_j) + b` runs over all n molecules, so
the training table has to hold all n; marking the missing activities in y is what keeps the
model one kernel expansion over one training table and every reader of it unchanged.

**The assumption.**  gamma g' L g asks the activity to vary smoothly along the data.  An
activity cliff -- two molecules whose fields are nearly the same and whose activities are
not -- breaks that assumption, and the penalty then pulls the prediction of both towards
their average.  How many such pairs a set holds is therefore worth counting before the
method is used on it (`benchmarks/run_lapkrr.py` reports the count per set).

`manifold = 0` switches the penalty off; the unlabelled molecules then get coefficients of
zero and the fit is ridge regression on the labelled molecules alone.

### `kfda_fit(K: np.ndarray, y: np.ndarray, lam: float) -> Tuple[np.ndarray, float]`

Kernel Fisher discriminant analysis for two classes (Mika et al. 1999).

The direction in field space along which the two classes are furthest apart relative to
their own spread.  In the dual it is one linear solve,

    a = (N + lam t I)^-1 (m_+ - m_-),

where m_c is the mean kernel row of class c, N = K W K is the within-class scatter written
on the kernel, W = I - 1_+ 1_+' / n_+ - 1_- 1_-' / n_- is the projector that centres each
class on its own mean, and t = trace(N) / n.

**What the regularisation is for.**  N has rank at most n - 2, so N on its own is singular
and the solve has no unique answer: the ridge is what makes the direction defined at all.
It also bounds the norm of the discriminant, which is the usual control of overfitting --
without it the quotient being maximised can be made arbitrarily large along directions that
the training molecules happen to span and new molecules do not.  lam is relative: the ridge
added is lam times the mean diagonal of N, so one lam grid means the same amount of
regularisation whatever the size of the kernel, as the lam of kernel ridge regression does
on a kernel scaled to unit mean diagonal.

The coefficients are scaled so that the within-class standard deviation of the training
scores is 1.  A discriminant direction is defined only up to its length, and fixing the
length this way makes the score readable (it is measured in within-class standard
deviations) and comparable between fits on different training sets, which is what
`kfda_calibrate` needs when it pools the scores of several folds.

The score is K(x, train) a + b, a kernel expansion like every other learner here, so the
co-field, the molecule contributions, the Shapley values and the applicability domain all
apply to this classifier without change.  That is what the support-vector classifier cannot
offer: its decision function is a kernel expansion too, but it is not linear in the field in
the sense a co-field needs, because the coefficients are the solution of a quadratic program
on the hinge loss rather than a linear map of the class labels.

The bias returned puts the decision at the midpoint of the two projected class means, which
is the Fisher threshold for equal class priors; `kfda_calibrate` replaces it with the
threshold that scores best out of fold.  Returns (dual coefficients, bias).

### `kfda_frame(score: np.ndarray, pos: np.ndarray) -> Tuple[float, float]`

(midpoint, half-separation) of the two class means of a discriminant score.

These two numbers are the units a threshold on this score is kept in: measured from the
midpoint and divided by the half-separation, the mean active scores +1, the mean inactive
scores -1, and a cut expressed that way means the same thing for two discriminants fitted on
different numbers of molecules.  That is what lets `kfda_calibrate` pool the scores of
several folds, and it is needed: the length of the solve is only fixed up to the within-class
spread of the molecules it was fitted on, which on a small fold is not the spread of the
molecules it is scoring.  Fitted on 35 of the 44 training molecules of the GPB set at a weak
ridge, the out-of-fold scores spread six times as widely as the full fit's, and a cut carried
over in raw score units then lands in the wrong place.

### `kfda_calibrate(K: np.ndarray, y: np.ndarray, lam: float, folds: int=KFDA_FOLDS, seed: Optional[int]=0, frame: Optional[Tuple[float, float]]=None) -> KFDACalibration`

Threshold and probability map of a kernel Fisher discriminant, from out-of-fold scores.

A discriminant direction says where the classes differ, not where to cut between them: the
solve of `kfda_fit` is invariant to any shift of the score.  The molecules are split
into folds, each fold is scored by a discriminant fitted on the other folds, and the cut with
the best balanced accuracy on those scores is the one kept.  The same scores give the
one-dimensional logistic map from score to probability.

Every fold's scores are first put into that fold's own class-mean units
(`kfda_frame`), which is what makes pooling them sound, and the cut that comes out is
in those units.  `frame` is the (centre, half-separation) of the fit the threshold will be
used with, so that the same cut can be turned back into a bias on its raw score; it is
computed here from a fit on all the molecules when the caller does not supply it.

### `out_of_fold_scores(K: np.ndarray, y: np.ndarray, fit, folds: int=KFDA_FOLDS, seed: Optional[int]=0) -> Tuple[np.ndarray, int, int]`

Scores of molecules that the fit which scored them had not seen, in class-mean units.

`fit(K_train, y_train)` returns (dual coefficients, bias) for one fold's training molecules.
Every fold's scores are put into that fold's own class-mean units (`kfda_frame`) before
they are pooled, which is what makes pooling them sound: a classifier fitted on four fifths of
the molecules separates them by a different amount than one fitted on all of them, and a cut
carried over in raw score units then lands in the wrong place.

Returns (scores, how many folds contributed, how many folds were asked for).  A fold whose
training part holds only one class is skipped, and with no usable fold the scores are all nan.

### `klr_threshold(K: np.ndarray, y: np.ndarray, lam: float, folds: int=KFDA_FOLDS, seed: Optional[int]=0, frame: Optional[Tuple[float, float]]=None) -> Tuple[float, float]`

Where to cut a kernel logistic regression when p = 0.5 is the wrong place.

The natural boundary of this learner is f = 0, which is p(active) = 0.5, and that is the
default because the class it reports and its probability then say the same thing.  It is also
the boundary that follows the larger class: with one active in twenty, calling a molecule
inactive is right nineteen times in twenty whatever the fields say, and the balanced accuracy
of such a model is 0.5.  This returns the bias that puts the best out-of-fold balanced
accuracy at zero instead, and that accuracy, leaving the probability untouched -- so a model
using it reports a class and a probability that disagree, which is the price and is stated in
`CMFModel.predict_proba`.

Returns (bias, balanced accuracy out of fold); (0.0, nan) when no fold could be fitted.

### `cv_predict(K: np.ndarray, y: np.ndarray, method: str, params: Dict[str, float], folds: int=10, seed: Optional[int]=0, fit_bias: bool=True, groups: Optional[Sequence]=None) -> np.ndarray`

k-fold cross-validated predictions on a precomputed joint kernel matrix.

`seed=None` uses the deterministic interleaved folds (molecule i goes to fold i mod k)
of the original R scripts; an integer seed shuffles the molecules first.

### `median_gamma_scale(K: np.ndarray) -> float`

1 / median squared distance of the training molecules in the field Hilbert space.

The bandwidth of the Gaussian kernel layer is expressed in these units, so that
gamma = 1 is the median-distance bandwidth whatever the size of the fields; the usual
"median heuristic" of kernel methods.

### `align_mixing(K_fields: np.ndarray, y: np.ndarray, scale: Optional[np.ndarray]=None, enabled: Optional[Sequence[bool]]=None, mode: str='align', rho: float=1.0) -> np.ndarray`

Mixing coefficients h_f by centred kernel alignment (multiple kernel learning).

The joint kernel of the CMF approach, K = sum_f h_f c_f K_f, is a *linear* combination of
the per-field overlap-integral kernels, so choosing h_f is exactly the multiple-kernel-
learning problem.  Alignment measures the cosine between a centred kernel and the ideal
kernel y y^T of the activities:

  `mode="align"`  (Cristianini et al. 2001) weights every field by its own alignment,
    h_f ~ <K_f^c, y^c y^c^T>_F / ||K_f^c||_F, which is robust because the fields are
    treated independently;
  `mode="alignf"` (Cortes et al. 2012) maximises the alignment of the *combination*,
    i.e. solves min_{h >= 0} || sum_f h_f K_f^c - y^c y^c^T ||_F.  On congeneric series the
    field kernels are strongly correlated and this solution is nearly sparse, so it is
    shrunk towards equal weights with strength `rho` (rho = 0 is the plain alignf);
  `mode="evidence"` instead maximises the marginal likelihood of the mixture under the
    prior of `cmfqsar.mixing`, which is the Bayesian reading of the same problem: h_f is
    the prior variance of the part of the model that field f carries.  It costs an
    optimisation rather than a least-squares solve, and the same module will give the
    posterior over those weights rather than this point.

The weights are rescaled so that the mean diagonal of the joint kernel is the same as with
h_f = 1, which keeps the regularisation grid comparable.  The combination stays linear, so
co-fields and atom contributions remain valid - that is the difference from a nonlinear
kernel layer, which can gain accuracy but loses the interpretation.

### `mkl_fold_weights(K_fields: np.ndarray, y: np.ndarray, model: 'CMFModel', folds: int=10, seed: Optional[int]=0, mode: str='align', rho: float=1.0) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]`

Per-fold alignment weights: a list of (test indices, training indices, h).

### `mkl_cv_predict(K_fields: np.ndarray, y: np.ndarray, model: 'CMFModel', params: Dict[str, float], folds: int=10, seed: Optional[int]=0, fold_h: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]=None, mode: str='align', rho: float=1.0) -> Tuple[np.ndarray, np.ndarray]`

Cross-validated predictions with the mixing coefficients re-derived inside every fold.

Deriving h_f from all activities and then cross-validating would leak the left-out
activities into the kernel; the weights are therefore computed from the training part of
each fold only.  Returns the predictions and the weights of the full training set.

### `block_fold_kernels(model: 'CMFModel', y: np.ndarray, folds: int=10, seed: Optional[int]=0) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]`

Per-fold kernels of an invariant source whose block weights come from the activities.

Deriving the weights from every activity and then cross-validating would leak the left-out
activities into the kernel, exactly as for the field mixing; so the channel scale and the
block weights are derived from the training part of each fold alone, and the whole
(n, n) kernel is formed under them.  A list of (test indices, training indices, K).

### `block_cv_predict(model: 'CMFModel', y: np.ndarray, params: Dict[str, float], fold_kernels: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray`

Cross-validated predictions over the per-fold kernels of `block_fold_kernels`.

### `cv_loss(model: 'CMFModel', y: np.ndarray, ycv: np.ndarray) -> Tuple[float, float]`

(loss to minimise, quality metric to report) of cross-validated predictions.

Regression: RMSEcv and q2.  Classification: 1 - AUC (falling back to 1 - balanced
accuracy when only one class is present in a fold) and the balanced accuracy.  Ranking:
1 - rho and Spearman's rho, because the score of a ranking model has no units and
minimising its squared error would be selecting on a quantity that does not exist.

### `grid_search(train: AtomTable, y: np.ndarray, model: CMFModel, alphas: Sequence[float], lams: Sequence[float]=(1.0,), Cs: Sequence[float]=(1.0,), epsilons: Sequence[float]=(0.1,), nus: Sequence[float]=(0.5,), gammas: Sequence[float]=(), folds: int=10, seed: Optional[int]=0, optimize_h: bool=False, mkl=False, mkl_rho: float=1.0, h_grid: Sequence[float]=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0), h_passes: int=2, alpha_fields: Optional[Sequence[str]]=None, progress: Optional[ProgressFn]=None, cancel: Optional[CancelFn]=None, lam_search: str='grid', precomputed: Optional[Dict]=None, alpha_search: str='grid', nm_start: Sequence[float]=(0.25, 5.0), nm_maxit: int=500, select: str='cv', components: Sequence[int]=(), manifolds: Sequence[float]=()) -> SearchResult`

Minimise RMSEcv over alpha (shared by all fields unless `alpha_fields` restricts it)
and the learner's regularisation parameters (`lams` for KRR/GP, `Cs` with `epsilons`
or `nus` for the support-vector methods, `components` for KPLS, which falls back to
`DEFAULT_COMPONENTS` when empty, `lams` together with `manifolds` for the
Laplacian-regularised ridge, which falls back to `DEFAULT_MANIFOLDS`).  Optionally refine the mixing
coefficients h_f by coordinate descent (beware of model-selection bias), or set them by
centred kernel alignment (`mkl=True`, see `align_mixing`), in which case the
reported RMSEcv is reliable because the weights are re-derived inside every fold.

`lam_search="brent"` (KRR/GP only) additionally minimises RMSEcv over a *continuous*
lambda in [min(lams), max(lams)] with Brent's bounded method, as the R scripts of the
2013 paper did (`optimize()` over gamma with the alpha grid evaluated inside);
the best of the grid and the Brent result is kept.

`alpha_search="nelder-mead"` (KRR/GP, unnormalised kernels) instead minimises RMSEcv
over the pair (alpha, lambda) with R's Nelder-Mead simplex (`optim`) started at
`nm_start`, the kernels for alphas between the grid points being linear interpolations
of the grid kernels and the end kernels being used beyond the grid, exactly as
`cmf_krr_train_mem` of the original scripts with `alpha_grid_search = FALSE` (the
protocol of the continuous-indicator-field models of Sitnikov et al. 2015).  The result of
the simplex search is used as it is (no comparison with the grid), the model's
`alpha_grid` is set so that kernels for new molecules are interpolated in the same way.

`precomputed` maps alpha -> (K_fields, self_diag) for the molecules of `train`
(used by `nested_cv` to avoid recomputing kernels for every outer fold).
`seed=None` selects the deterministic interleaved folds of the original scripts.

### `nested_cv(train: AtomTable, y: np.ndarray, model_template: CMFModel, alphas: Sequence[float], lams=(1.0,), Cs=(1.0,), epsilons=(0.1,), nus=(0.5,), outer_folds: int=5, inner_folds: int=10, repeats: int=1, seed: Optional[int]=0, optimize_h: bool=False, mkl=False, mkl_rho: float=1.0, alpha_fields=None, gammas: Sequence[float]=(), progress: Optional[ProgressFn]=None, cancel: Optional[CancelFn]=None, lam_search: str='grid', precompute: bool=True, max_precompute_bytes: float=2000000000.0, alpha_search: str='grid', nm_start: Sequence[float]=(0.25, 5.0), inner_seed='same', components: Sequence[int]=(), manifolds: Sequence[float]=()) -> NestedCVResult`

External k-fold cross-validation with the hyper-parameters selected in an
inner cross-validation loop on each working set (Baskin & Zhokhova 2013).

With `precompute` the per-field kernel matrices of the whole set are computed once
for every alpha of the grid and sliced for the outer folds (identical results, much
faster).  `seed=None` uses interleaved (unshuffled) outer and inner folds.  With
`repeats > 1` the molecules are reshuffled before every repeat (`seed` + repeat index;
`seed=None` gives the same interleaved outer folds in every repeat); besides the mean and
standard deviation of the per-repeat statistics the result also holds the statistics of
the prediction averaged over the repeats (`q2_ex_aggr`, `rmse_ex_aggr`), which is
what the original scripts (`cmf_krr_ecvr`) and the 2015 paper report.  `inner_seed`
is the seed of the inner cross-validation (default: the same as `seed`); `None` gives
interleaved inner folds as in the scripts, which reshuffle the molecules only once per
repeat.

## cmfqsar.multiobjective

Source: `cmfqsar/multiobjective.py`

A design loop with more than one objective (plan item MC11).

Plan items E2 and E3 close the loop on potency alone.  Lead optimisation has at least three
objectives -- the potency, the selectivity against an anti-target panel, and whatever property
of the same series is measured -- and since plan items E1 and MC4 those live on one kernel: a
multi-task model predicts them all, with a posterior for each.

What is missing between that model and a chemist's list is three things, and this module is
them:

* **the Pareto front**, the candidates no other candidate beats on every objective at once.  A
  weighted sum has to be given weights, and nobody knows them before the compounds are made; a
  front does not, and the chemist chooses from it.
* **an acquisition over the objectives**, so that the front can be ordered and a batch taken
  from it.  Expected improvement is computed per objective against the best already made, and
  the objectives are combined by a weighted sum of those improvements, which is scale-free
  because each one is already in its own units of improvement.
* **a sentence per suggestion** saying what it gains and what it loses against the compound it
  is to replace.  A loop that reports only a score hides the trade it is making.

The batch rule of plan item MC2 generalises without a change of idea: after a pick, condition
**every** task's posterior covariance on it (`cmfqsar.active.condition_on`) and compute
the acquisition again, so a plate is not one compound ten times over.  What that conditioning
takes away is the uncertainty, not the promise: two compounds the model is sure are a log unit
better than anything made are both still worth making, and only the part of their score that
came from not knowing them falls.  A rule that spread the batch on the predictions instead
would be refusing a compound for resembling a good one.

Nothing here is specific to two objectives or to the fields: it takes predicted means and
posterior covariances, one per objective, and gives back an order.

### `class Objective  (dataclass)`

One thing the loop is optimising, and which way is better.

Attributes:

- `name: str`
- `direction: int` = `MAXIMISE`
- `weight: float` = `1.0`
- `best: Optional[float]` = `None`

#### `Objective.sign` -> float  (property)

### `class Trade  (dataclass)`

One candidate against one reference compound, objective by objective.

The numbers in `gains` and `loses` are improvements **in the direction of the objective**:
positive is better whichever way the objective runs, so a hERG activity that rises by one
log unit appears as -1.00 in `loses`.

Attributes:

- `name: str`
- `against: str`
- `gains: List[Tuple[str, float]]` = `_field(default_factory=list)`
- `loses: List[Tuple[str, float]]` = `_field(default_factory=list)`
- `same: List[str]` = `_field(default_factory=list)`

### `dominates(a: Sequence[float], b: Sequence[float], signs: Sequence[float]) -> bool`

Is `a` at least as good as `b` everywhere and better somewhere?

### `pareto_layers(values: np.ndarray, objectives: Sequence[Objective]) -> np.ndarray`

Which front every candidate belongs to: 1 is the Pareto front, 2 is the next, and so on.

`values` is (candidates, objectives) of predicted values, in the units of the objectives.

### `pareto_front(values: np.ndarray, objectives: Sequence[Objective]) -> np.ndarray`

The indices of the candidates nothing else beats on every objective at once.

### `improvements(values: np.ndarray, sds: np.ndarray, objectives: Sequence[Objective]) -> np.ndarray`

(candidates, objectives) of expected improvement, each in its own objective's units.

### `acquisition(values: np.ndarray, sds: np.ndarray, objectives: Sequence[Objective]) -> np.ndarray`

One score per candidate: the weighted sum of the improvements over the objectives.

Each term is an expected improvement, which is already in the units of its own objective's
improvement, so the weights say how much a log unit of one is worth against a log unit of
another and nothing else is hidden in them.

### `batch(values: np.ndarray, covariances: Sequence[np.ndarray], objectives: Sequence[Objective], size: int, only: Optional[Sequence[int]]=None) -> List[int]`

A batch of `size` candidates, the posterior of every objective conditioned on each pick.

This is the rule of plan item MC2 with more than one task: after a pick, every objective's
covariance is conditioned on it, so a candidate whose uncertainty that pick has already
removed -- in any of the objectives -- scores lower for the next one.  `only` restricts the
choice to a subset, which is how a batch is taken from the Pareto front alone.

### `trade(values: Sequence[float], reference: Sequence[float], objectives: Sequence[Objective], name: str='', against: str='', indifference: float=0.0) -> Trade`

What one candidate gains and what it loses against a reference, objective by objective.

`indifference` is the change below which an objective is called unchanged; the sensible
value is the model's own error, since a loop should not report a trade it cannot see.

### `best_so_far(y: np.ndarray, objectives: Sequence[Objective]) -> List[float]`

The best value of every objective among the compounds already made.

### `with_best(objectives: Sequence[Objective], y: np.ndarray) -> List[Objective]`

The same objectives with `best` filled in from what has been made.

### `report(values: np.ndarray, objectives: Sequence[Objective], names: Sequence[str], picks: Sequence[int], reference: Optional[Sequence[float]]=None, reference_name: str='the best already made', indifference: float=0.0) -> List[Trade]`

One trade per suggestion, against the best compound already made unless told otherwise.

## cmfqsar.multitask

Source: `cmfqsar/multitask.py`

Two endpoints at once, so that selectivity is a co-field rather than a ratio.

A separation problem is not one property but two: an extractant has an affinity for americium and
an affinity for europium, and what matters is the difference.  Modelling the ratio directly works
-- `benchmarks/run_am_eu.py` gets q2 = +0.81 on lg SF -- but it throws information away.  Each
endpoint has its own measurements, often on partly different compounds; a ligand that extracts
both metals strongly looks, to a model of the ratio, exactly like one that extracts neither; and
the ratio cannot say *which* of the two affinities a substituent moved.

This module fits both endpoints on one shared CMF kernel:

    y_t(X)  =  sum_j sum_s a_js T(t,s) K(X, X_j)  +  b_t

with `T` a task covariance -- for two tasks, `[[1, rho], [rho, 1]]` with one correlation to
choose -- and one bias per task.  Writing the inner sum as an effective coefficient per molecule
and task,

    c_jt  =  sum_s a_js T(t,s),        y_t(X) = sum_j c_jt K(X, X_j) + b_t,

each task's prediction is **linear in the molecular field with its own dual coefficients**, which
is the purpose: it reduces to exactly the object the rest of the program already knows how to
read.  Hand `c[:, t]` to `cmfqsar.model.CMFModel.cofield_coefficients` and you get that
endpoint's co-field; hand it `c[:, Am] - c[:, Eu]` and you get the **selectivity co-field**,

    C_Am(r) - C_Eu(r)  =  sum_j (c_j,Am - c_j,Eu) X_j(r),

a map of where more of a field raises the affinity for one metal more than for the other.  Only a
model linear in the field can produce that; it is available here for the same reason co-fields are
available at all.

Three things the implementation has to get right, and does:

* **Cross-validation leaves out molecules, not measurements.**  With a molecule's Am value in the
  training set and its Eu value in the test set, the task correlation hands the answer over.
  `cv_predict` folds over molecules and moves every observation of a held-out molecule
  together.
* **Missing measurements are the normal case**, so the kernel is built over observed (molecule,
  task) pairs rather than as a Kronecker product over a full grid.
* **rho = 0 must reproduce single-task models exactly**, or the sharing cannot be measured against
  anything; `tests/test_multitask.py` holds it to that.

What the selected `rho` means is worth saying: at rho = 0 the endpoints are modelled separately;
as rho rises they share information with each other; and at rho = 1 the task covariance is singular,
the effective coefficients of the two tasks coincide, and the model can only predict a *constant*
selectivity.  So the selectivity structure lives strictly below rho = 1, and the value chosen by
cross-validation says how much of each metal's affinity is the other's.

More than two endpoints
-----------------------

One shared `rho` assumes that every pair of endpoints is related to the same degree.  With two
endpoints that assumption is empty.  With seven it is strong, and usually wrong: some endpoints
are almost the same measurement, others are only distantly related.  Two more general forms are
implemented here.  Both are selected by the **marginal likelihood** instead of
cross-validation, because a covariance with ten free parameters needs one matrix factorisation per
candidate this way and ten per candidate with ten-fold cross-validation.

* **A free-form low-rank covariance.**  `T = L L' + diag(d)`, with `L` of shape (tasks, rank)
  and `d > 0`.  It is positive definite for any parameter values, it has (rank + 1) * tasks
  parameters, and at rank = tasks it can represent any covariance.  `fit_covariance` fits
  it.  Its diagonal is *not* fixed at one, which matters when the endpoints are in different
  units: a potency and a free energy in kJ/mol need separate variances, and one shared rho gives
  them the same variance.

* **A covariance computed from task descriptors.**  `T(t,s) = k(z_t, z_s)`, where `z` is a
  vector of properties of the endpoint -- the medium, the metal, the temperature, or, in a
  host-guest problem, the structure of the host.  `descriptor_covariance` selects the
  kernel's length scale and nugget by the evidence.  Only this form can say anything about an
  endpoint with **no measurements at all**: a free-form row cannot be estimated for an endpoint
  with no data, while a descriptor row is computed from the descriptors alone.
  `MultiTaskFit.predict_new_task` then predicts that endpoint from the related ones.

The mean needs the same change.  A separate bias per endpoint is one indicator column per
endpoint, and an endpoint with no measurements has no bias to estimate.  Passing
`task_features` to `Observations` replaces those indicator columns with the descriptors,
so the mean is a linear function of them and can be evaluated for an endpoint that was never
measured.  Without task features the design matrix is the indicator matrix and the behaviour is
unchanged.

One limitation remains.  An endpoint with no measurements is predicted by *interpolation between
the endpoints that were measured*, so the prediction contains only what those endpoints have in
common.  `benchmarks/run_tasks.py` compares it with two reference values that need no model: the
mean of the measured endpoints, and the single most similar endpoint by descriptors.

### `class Observations  (dataclass)`

Which (molecule, task) pairs were measured, and what was measured.

Attributes:

- `molecule: np.ndarray`
- `task: np.ndarray`
- `y: np.ndarray`
- `n_mol: int` = `0`
- `n_tasks: int` = `0`
- `task_names: Tuple[str, ...]` = `()`
- `task_features: Optional[np.ndarray]` = `None`

#### `Observations.from_columns(values: Sequence[Sequence[float]], names: Sequence[str]=(), task_features: Optional[np.ndarray]=None) -> 'Observations'`  (class method)

From one column per task, with NaN for a measurement that was not made.

#### `Observations.design() -> np.ndarray`

`B`, the (P, d) design matrix of the mean: indicator columns, or the descriptors.

The bordered system of `fit` imposes `B' a = 0`. With indicator columns that is
one constraint per endpoint, namely that its coefficients sum to zero. With descriptor
columns it is the generalised-least-squares normal equation for a mean that is a linear
function of the descriptors.

#### `Observations.task_design() -> np.ndarray`

`(n_tasks, d)`: the row of the design each endpoint would contribute.

#### `Observations.counts() -> List[int]`

#### `Observations.subset(rows: Sequence[int]) -> 'Observations'`

### `class MultiTaskFit  (dataclass)`

A fitted two-or-more-endpoint model and the per-task coefficients it reduces to.

Attributes:

- `alpha: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`
- `bias: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`
- `coef: np.ndarray` = `field(default_factory=lambda: np.zeros((0, 0)))`
- `T: np.ndarray` = `field(default_factory=lambda: np.zeros((0, 0)))`
- `rho: float` = `float('nan')`
- `lam: float` = `float('nan')`
- `obs: Optional[Observations]` = `None`
- `mean_coef: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`

#### `MultiTaskFit.task_coefficients(task: int) -> np.ndarray`

The dual coefficients of one endpoint, over the training molecules.

These go straight into the single-task machinery -- set them as a model's `coef` and its
co-field, atom contributions and cube export are that endpoint's.

#### `MultiTaskFit.selectivity_coefficients(task: int, other: int) -> np.ndarray`

`c[:, task] - c[:, other]`: the dual coefficients of the selectivity co-field.

#### `MultiTaskFit.predict(K_cross: np.ndarray, task: int) -> np.ndarray`

Predict one endpoint for new molecules; `K_cross` is (n_new, n_train_molecules).

#### `MultiTaskFit.predict_selectivity(K_cross: np.ndarray, task: int, other: int) -> np.ndarray`

The difference of the two endpoints, which is what a separation factor is.

#### `MultiTaskFit.new_task_coefficients(t_row: Sequence[float]) -> np.ndarray`

Dual coefficients over the training molecules for an endpoint that was never measured.

`t_row` is the covariance of that endpoint with the fitted ones. Only a covariance
computed from task descriptors can supply it. The coefficients are
`c_j = sum_q a_q T(new, task_q)`, summed over the observations of molecule j. Pass the
result to the co-field functions and the map obtained is that of the new endpoint.

#### `MultiTaskFit.predict_new_task(K_cross: np.ndarray, t_row: Sequence[float], features: Optional[Sequence[float]]=None) -> np.ndarray`

Predict an endpoint with no measurements, from its covariance and its descriptors.

`features` is the row of the task design matrix for that endpoint. Without task features
there is no bias to evaluate, because the indicator column of an unmeasured endpoint was
never estimated. In that case the fitted biases are averaged with weights taken from the
covariance. That average is an approximation, not an estimate of the new endpoint's bias.

### `class CovarianceFit  (dataclass)`

A task covariance chosen by the evidence, and what it cost to choose.

Attributes:

- `T: np.ndarray`
- `lam: float`
- `evidence: float`
- `rank: int`
- `evaluations: int` = `0`
- `kind: str` = `'free'`
- `note: str` = `''`

#### `CovarianceFit.correlations` -> np.ndarray  (property)

`T` scaled to unit diagonal, which is the part that is comparable with `rho`.

#### `CovarianceFit.describe(names: Sequence[str]=()) -> str`

### `class DescriptorCovariance  (dataclass)`

A covariance read off task descriptors, and the rule for a new endpoint's row.

Attributes:

- `T: np.ndarray`
- `lam: float`
- `evidence: float`
- `length_scale: float`
- `nugget: float`
- `kind: str`
- `Z: np.ndarray`
- `mean: np.ndarray`
- `scale: np.ndarray`

#### `DescriptorCovariance.row_for(z_new: Sequence[float]) -> np.ndarray`

The covariance of an endpoint with the trained ones, from its raw descriptors.

#### `DescriptorCovariance.as_fit() -> CovarianceFit`

### `task_covariance(n_tasks: int, rho: float) -> np.ndarray`

`(1 - rho) I + rho 11^T`: one correlation shared by every pair of tasks.

Positive definite for `-1/(n_tasks - 1) < rho < 1`; the caller is expected to keep rho
strictly below 1, where the matrix is singular and selectivity becomes a constant.

### `pair_kernel(K: np.ndarray, rows: Observations, cols: Observations, T: np.ndarray) -> np.ndarray`

The kernel between observed pairs: `K[molecule, molecule] * T[task, task]`.

### `fit(K: np.ndarray, obs: Observations, rho: float=0.0, lam: float=1.0, fit_bias: bool=True, T: Optional[np.ndarray]=None) -> MultiTaskFit`

Kernel ridge regression over the observed (molecule, task) pairs.

The bordered system of the single-task case (Eq. 12 of Baskin & Zhokhova 2013) gains one
column per task, so each endpoint gets its own bias and its own `sum_j a_j = 0`:

    [ G + lam I   B ] [ a ]   [ y ]
    [ B^T         0 ] [ b ] = [ 0 ]

with `G` the pair kernel and `B[p, t] = 1` when observation p belongs to task t.

### `cv_predict(K: np.ndarray, obs: Observations, rho: float=0.0, lam: float=1.0, folds: int=10, seed: Optional[int]=0, fit_bias: bool=True, T: Optional[np.ndarray]=None) -> np.ndarray`

Out-of-fold predictions for every observation, holding out whole **molecules**.

A molecule's measurements never straddle the split: with its americium value in the training
set and its europium value in the test set, the task correlation would hand over the answer
and the result would be meaningless.

### `score_by_task(obs: Observations, pred: np.ndarray) -> List[dict]`

q2 and RMSE per task, computed where the predictions exist.

### `score_selectivity(K: np.ndarray, obs: Observations, task: int, other: int, rho: float, lam: float, folds: int=10, seed: Optional[int]=0, fit_bias: bool=True, T: Optional[np.ndarray]=None) -> dict`

Cross-validated q2 of the *difference* of the two endpoints.

Only molecules carrying both measurements can be scored, since only they have an observed
difference; the model is still fitted on everything, which is the point of doing it this way.

### `search(K: np.ndarray, obs: Observations, rhos: Sequence[float]=(0.0, 0.3, 0.5, 0.7, 0.9), lams: Sequence[float]=(0.001, 0.01, 0.1, 1.0, 10.0), folds: int=10, seed: Optional[int]=0, fit_bias: bool=True, statistic: str='mean') -> Tuple[float, float, float, List[dict]]`

Choose (rho, lambda) by cross-validation; returns (rho, lam, score, per-task scores).

`statistic` is what the search maximises: `"mean"` the mean q2 over the tasks, or the name
of one task to tune on that endpoint alone.

### `log_evidence(K: np.ndarray, obs: Observations, T: np.ndarray, lam: float, fit_bias: bool=True, profile: bool=True) -> float`

The log marginal likelihood of the observations under `T` and `lam`.

With `fit_bias` the mean is integrated out over the design columns of `obs`. This extends the
single-task case, where one unknown constant is integrated out, to one constant per endpoint,
or, with task descriptors, to a mean that is a linear function of them. The degrees of freedom
are reduced by the number of design columns, which puts covariances of different rank on the
same scale.

### `low_rank_covariance(theta: Sequence[float], n_tasks: int, rank: int) -> np.ndarray`

`L L' + diag(exp(2 u))` from a flat parameter vector, normalised to mean diagonal 1.

The result is positive definite for every `theta`, so the optimiser needs no constraints. The
normalisation removes the one direction the evidence cannot distinguish: multiplying `T` and
`lam` by the same factor leaves the fitted model unchanged, so without the normalisation the
optimiser would move along that direction without improving anything. The *relative* sizes of
the diagonal entries stay free, so endpoints in different units get separate variances.

### `fit_covariance(K: np.ndarray, obs: Observations, rank: int=1, lams: Sequence[float]=(0.1,), rhos: Sequence[float]=(0.2, 0.8), fit_bias: bool=True, maxiter: int=80) -> CovarianceFit`

Optimise a free-form low-rank task covariance on the evidence.

Nelder-Mead over the `(rank + 1) * tasks` parameters and `log lam`. It is started from
the rank-1 point that reproduces each `rho` of `rhos` at each `lam` of `lams`, and the best
result is kept. Several starting points are needed because the evidence as a function of the
covariance is not concave: on real data a start at rho = 0.9 and a start at rho = 0.1 end at
different optima.

### `standardise(Z: np.ndarray) -> np.ndarray`

Columns centred and scaled to unit spread, with constant columns dropped to zero.

### `task_kernel(Z: np.ndarray, length_scale: float=1.0, nugget: float=0.1, kind: str='rbf', Z_new: Optional[np.ndarray]=None) -> np.ndarray`

The covariance between endpoints, from their descriptors.

`kind` is `"rbf"`, that is `exp(-|z_t - z_s|^2 / 2 l^2)`, so that endpoints with similar
descriptors are strongly correlated and distant ones are not; or `"linear"`, the normalised
inner product, which can be negative and is the form to use when two endpoints are expected to
be *negatively* correlated (a potency and a free energy). The nugget is added to the diagonal.
Without it, two endpoints with identical descriptors would be perfectly correlated, the
covariance would be singular, and their coefficients would be forced to be equal.

With `Z_new` the result is the (n_new, n_tasks) cross-covariance, which is what is needed for
an endpoint with no measurements.

### `descriptor_covariance(K: np.ndarray, obs: Observations, Z: np.ndarray, length_scales: Sequence[float]=(0.25, 0.5, 1.0, 2.0, 4.0), nuggets: Sequence[float]=(0.05, 0.1, 0.2, 0.4), lams: Sequence[float]=(0.01, 0.1, 1.0), kinds: Sequence[str]=('rbf',), fit_bias: bool=True) -> DescriptorCovariance`

Choose the task kernel's length scale, nugget and ridge by the evidence.

A grid rather than Nelder-Mead, because there are only three or four numbers and the grid is
the thing to show in a table.

### `evidence_rho(K: np.ndarray, obs: Observations, rhos: Sequence[float]=(), lams: Sequence[float]=(0.01, 0.1, 1.0), fit_bias: bool=True) -> CovarianceFit`

The one-correlation covariance, chosen by the evidence instead of by cross-validation.

This is the reference for the free-form and descriptor covariances: the same objective and the
same ridge grid, with one parameter.

## cmfqsar.neldermead

Source: `cmfqsar/neldermead.py`

Nelder-Mead simplex minimiser, a direct port of R's `nmmin` (`optim(method =
"Nelder-Mead")`, R 2.x-4.x), which the original CMF R scripts use to optimise the
attenuation factor alpha and the regularisation parameter gamma simultaneously
(`cmf_krr_train_mem` with `alpha_grid_search = FALSE`).

The port reproduces R's sequence of function evaluations exactly: the initial simplex
with step `0.1 * max |x0|`, the reflection/expansion/contraction/shrink rules and
the convergence test on the spread of the function values only
(`VH <= VL + reltol * (|f(x0)| + reltol)`).

### `nmmin(fn: Callable[[List[float]], float], x0: Sequence[float], abstol: float=-math.inf, reltol: float=1.490116e-08, maxit: int=500, alpha: float=1.0, bet: float=0.5, gamm: float=2.0) -> Tuple[List[float], float, int, int]`

Minimise `fn` from `x0`; returns (x_min, f_min, n_evaluations, fail).

`fail` is 0 on convergence, 1 when `maxit` evaluations were exceeded and 10 when
the simplex could not shrink any further (the codes of R's `optim`).

### `interp_terms(alpha: float, grid: Sequence[float]) -> List[Tuple[float, float]]`

Linear interpolation weights of `alpha` between the points of a sorted `grid`.

Returns `[(grid_lo, c_lo), (grid_hi, c_hi)]` (or a single `(grid_end, 1.0)` term
when `alpha` lies outside the grid), as `cmf_kernels_interpolate` of the original
R scripts: kernel matrices are only available on the alpha grid and are interpolated
linearly in between, with the end values used beyond the ends.

## cmfqsar.nonlinearity

Source: `cmfqsar/nonlinearity.py`

The co-field of a model with a nonlinear kernel layer: one per molecule, and how far they differ.

A model with a kernel layer (`CMFModel.kernel_layer`) predicts

    y(x) = sum_j a_j g( K(x, x_j), K(x, x), K(x_j, x_j) ) + b,                              (1)

where K is the linear joint kernel -- the overlap integrals of the normalised fields, mixed over
the field types -- and g is the layer: the identity for the linear model, exp(-gamma d^2) for
the Gaussian layer, the arc-cosine function of Cho and Saul for the arc-cosine layers.  The
prediction is then no longer the overlap of the field with one fixed co-field, which is why
`CMFModel.cofield_coefficients` refuses such a model.  It is still a differentiable
function of the field, and its derivative with respect to the normalised field of type f, taken
at molecule x, is a field:

    C_fx(r) = sum_j a_j g_s(x, j) x_fj(r)  +  2 [ sum_j a_j g_n(x, j) ] x_fx(r),              (2)

with g_s = dg/dK(x, x_j), g_n = dg/dK(x, x), x_fj the normalised field of training molecule j
(the basis of the ordinary co-field, `CMFModel.training_field_basis`) and x_fx the
normalised field of x itself.  This is the **local co-field**: the co-field of the linear model
tangent to (1) at x.  The prediction of a molecule close to x changes by the overlap of C_fx
with the change of the normalised field, summed over the field types with the usual factors
h_f c_f.  For the linear layer g_s = 1 and g_n = 0, and (2) is the ordinary co-field whatever x
is; for the Gaussian layer g_s = 2 gamma g and g_n = -gamma g, so the training molecules are
weighted by their kernel similarity to x and the field is taken relative to x's own.

The second term of (2) lies along x's own field.  With cosine-normalised kernels (the default)
K(x, x) does not change when the normalised field moves, so the term has no effect on any
prediction and is left out -- as the ordinary co-field leaves out the dependence of the
normalisation on the field.  Without normalisation it is kept.

Two readings of (2) are built on top of the map itself:

* the **spread** of the local co-fields over the training molecules (`cofield_spread`):
  their mean, which is the one co-field to draw for a nonlinear model, and their standard
  deviation point by point, which is a map of where the effect of field depends on the rest of
  the molecule -- zero where the model is additive, large where it is not;
* a **non-linearity index** per field type (`nonlinearity_indices`), the share of the
  squared norm of the local co-fields that their differences from the mean make up, computed in
  closed form from the training kernel matrix: 0 for a linear model, 1 when the local co-fields
  have nothing in common.  The mean pairwise cosine between them is reported beside it.

What the maps are not.  They describe the tangent models, not the chemistry by themselves: the
Gaussian layer makes every model nonlinear to a degree set by its bandwidth, and the spread is
nonzero even when the measured relation is additive, as long as the bandwidth is finite.  Read
the spread against the linear model's cross-validated error (if the linear kernel predicts as
well, the spread is a property of the kernel) and check that it survives the folds, as the noise
band of `cmfqsar.cofields` is checked.

Second derivatives -- how the effect of field at one place depends on the field at another --
are a function of two places, not a field.  The second half of this module takes them with
respect to the amplitudes of probe atoms, which keeps them finite and exact under the kernel
normalisation: the **pair-interaction field** (one probe fixed, the other scanned over a grid),
the **curvature field** (its diagonal), and the **atom interaction matrix** of one molecule with
its eigenmodes, the patterns of atoms the model reads together.

### `class CofieldSpread  (dataclass)`

The local co-fields of the training molecules, summarised point by point.

Attributes:

- `mean: np.ndarray`
- `sd: np.ndarray`
- `n: int`

### `class NonlinearityIndex  (dataclass)`

How far the local co-fields of one field type differ over the training molecules.

Attributes:

- `field: str`
- `spread: float`
- `cosine: float`
- `n: int`

### `class ProbeSet  (dataclass)`

Probe atoms of one field type: Gaussians added to the molecule's raw field.

`centres` (P, 3), `widths` (P,) the exponents b of exp(-b/2 |r - c|^2), and `weights`
(P,) the amount of field one unit of the probe's amplitude puts there: one for an indicator
field, a charge for the electrostatic field.  The derivatives below are per unit amplitude.

Attributes:

- `field: int`
- `centres: np.ndarray`
- `widths: np.ndarray`
- `weights: np.ndarray`

### `class Tangent(model: CMFModel, mol: MoleculeFields)`

The derivatives of one model's prediction at one molecule, to second order.

Holding the layer's derivatives, the kernel row and the molecule's self-overlaps, it answers
for any set of probes: what each probe adds per unit amplitude (`gradient`), and how
the effect of one depends on the amplitude of another (`hessian`).  The maps and the
modes below are built on these two.

#### `Tangent.second()`

#### `Tangent.check_field(f: int) -> None`

#### `Tangent.A_tilde(f: int) -> np.ndarray`

<X_f, x_fj> over the training molecules, with x_fj the (normalised) basis field.

#### `Tangent.terms(p: ProbeSet) -> Tuple[np.ndarray, np.ndarray]`

V (n_train, P): <p_k, x_fj>; u (P,): <p_k, X_f>, the molecule's own raw field.

#### `Tangent.first(p: ProbeSet, V: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`

S (n_train, P) = dS_j/de_k and n1 (P,) = dn/de_k.

#### `Tangent.gradient(p: ProbeSet) -> np.ndarray`

dy/de_k: what one unit of each probe adds to the prediction, to first order.

#### `Tangent.hessian(a: ProbeSet, b: Optional[ProbeSet]=None, terms_a: Optional[tuple]=None) -> np.ndarray`

d2y/de_k de_l between the probes of `a` (rows) and `b` (columns; `a` when None).

#### `Tangent.diagonal(b: ProbeSet) -> np.ndarray`

d2y/de_k^2 for every probe: the curvature along each probe alone.

### `class InteractionMode  (dataclass)`

One eigenpair of the atom interaction matrix.

Attributes:

- `value: float`
- `vector: np.ndarray`

### `layer_derivatives(model: CMFModel, K: np.ndarray, diag_p: np.ndarray, diag_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`

The layer g on a block of the linear kernel, and its derivatives g_s and g_n of Eq. (2).

`K` holds the linear joint kernel between the rows (the molecules the derivative is taken
at, self-kernels `diag_p`) and the columns (the training molecules, self-kernels `diag_q`).
The first array equals `model.apply_layer(K, diag_p, diag_q)`; the second is the
derivative with respect to the entry K(x, x_j); the third the derivative with respect to the
row's own self-kernel K(x, x), the other arguments held fixed.

### `local_duals(model: CMFModel, table: Optional[AtomTable]=None) -> Tuple[np.ndarray, np.ndarray]`

The coefficients of Eq. (2) for the molecules of `table`, or for the training set.

Returns `(B, b0)`: `B[x, j] = a_j g_s(x, j)` weighs the normalised field of training
molecule j in the local co-field at x, and `b0[x] = 2 sum_j a_j g_n(x, j)` weighs x's own
normalised field.  `b0` is zero for a model with normalised kernels (see the module
docstring) and for the linear layer, where `B` has the dual coefficients in every row.
The same two arrays serve every field type, because the layer acts on the joint kernel.

### `local_cofield_coefficients(model: CMFModel, f: int, mol: MoleculeFields) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`

Centres, coefficients and widths of the Gaussians composing the local co-field C_fx.

The same three arrays `CMFModel.cofield_coefficients` returns, in the same
convention: the field-scaling constant c_f and the mixing coefficient h_f are not included.
For a linear model the result is the ordinary co-field whatever `mol` is.  Without
normalisation the molecule's own Gaussians are appended with the weight `b0` of Eq. (2).

### `local_cofield_volume(model: CMFModel, f: int, mol: MoleculeFields, grid, include_scale: bool=False) -> np.ndarray`

C_fx on a grid (shape = grid.shape); with `include_scale`, multiplied by h_f c_f.

### `local_cofield_at_points(model: CMFModel, f: int, mol: MoleculeFields, points: np.ndarray, include_scale: bool=True) -> np.ndarray`

C_fx at a list of points rather than on a grid.

### `cofield_spread(model: CMFModel, f: int, grid, include_scale: bool=False, chunk: int=0) -> CofieldSpread`

Mean and standard deviation of the local co-fields C_fx over the training molecules x.

Both are read off the same matrix the co-field noise band uses -- the normalised fields of
the training molecules at the grid points -- multiplied by the coefficient matrix of
`local_duals`, one block of points at a time.  For a linear model the standard
deviation is zero everywhere and the mean is the co-field.

### `nonlinearity_indices(model: CMFModel, joint: bool=True) -> List[NonlinearityIndex]`

The non-linearity index of every enabled field type, and of all of them together.

The local co-fields of the training molecules are combinations of the training fields,
`C_x = sum_j Bt[x, j] x_j`, so their Gram matrix in the Hilbert space of the fields is
`Bt K_f Bt^T` with `K_f` the (normalised) training kernel of field f, which the model
keeps.  The index is read off that matrix: the mean squared norm is the mean of its
diagonal, the squared norm of the mean local co-field is the mean of all its entries, and
the spread is one minus their ratio.  Nothing is evaluated on a grid.  With `joint` a last
entry named `joint` takes all the fields together, weighted as the joint kernel weighs
them, which is the index of the model as a whole.

### `layer_second_derivatives(model: CMFModel, K: np.ndarray, diag_p: np.ndarray, diag_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`

(g_ss, g_sn, g_nn): the second derivatives of the layer, as `layer_derivatives`.

Defined for the linear kernel (all zero) and the Gaussian layer.  The arc-cosine layers are
refused: their kernel has a cusp where a molecule meets a training molecule (the angle is
zero), and the second derivative diverges there, so a training molecule has no
pair-interaction field under them.

### `probe_set(model: CMFModel, f: int, centres, widths=None, weights=None, radius: float=1.7) -> ProbeSet`

Probes in field `f` at `centres`, with the field's own width unless `widths` is given.

A field with one Gaussian per atom (`kind == "gauss"`) has the width `alpha`; the steric
kind scales it by the atom's radius, for which `radius` stands in.

### `atom_probes(model: CMFModel, f: int, mol: MoleculeFields) -> ProbeSet`

The atoms of `mol` as probes of field `f`: a unit change of each atom's weight.

### `probe_gradient(model: CMFModel, mol: MoleculeFields, probes: ProbeSet) -> np.ndarray`

What one unit of each probe adds to the prediction of `mol`, to first order.

### `probe_hessian(model: CMFModel, mol: MoleculeFields, a: ProbeSet, b: Optional[ProbeSet]=None) -> np.ndarray`

How the effect of each probe of `a` depends on the amplitude of each probe of `b`.

### `pair_interaction_field(model: CMFModel, mol: MoleculeFields, f: int, centre, grid, f_probe: Optional[int]=None, width: Optional[float]=None, weight: float=1.0, chunk: int=0) -> np.ndarray`

The pair-interaction field: one probe fixed at `centre`, the other scanned over the grid.

The value at r is the second derivative of the prediction with respect to the amplitude of a
probe of field `f_probe` (default `f`) at `centre` and of a probe of field `f` at r: how
much adding field at r changes what the fixed probe does.  Positive where the two reinforce
each other, negative where one undoes part of the other, zero where the model treats the two
places additively.  With the fixed probe on an atom of `mol`, `weight` set to that atom's
weight gives the field for *that atom*; with weight one it is per unit of field.

### `gradient_field(model: CMFModel, mol: MoleculeFields, f: int, grid, chunk: int=0) -> np.ndarray`

What one unit of a probe of field `f` adds to the prediction of `mol`, at every point.

The first-derivative companion of the maps below: under a nonlinear layer it is what
`cmfqsar.cofields.atom_effect` is for a linear model, the local co-field convolved
with the probe's own Gaussian, in the units of the activity.

### `curvature_field(model: CMFModel, mol: MoleculeFields, f: int, grid, chunk: int=0) -> np.ndarray`

The diagonal of the pair-interaction field: the second derivative along one probe.

Negative where the model saturates (a second unit of field there adds less than the first),
positive where it accelerates.  For a linear model without normalisation it is zero; with
cosine normalisation it is not, because the normalisation itself saturates.

### `atom_interactions(model: CMFModel, mol: MoleculeFields, f: int) -> Tuple[np.ndarray, np.ndarray]`

The interaction matrix of the atoms of `mol` in field `f`, and their first-order effects.

`H[k, l]` is the second derivative of the prediction with respect to the weights of atoms k
and l in that field (two partial charges, say), `g[k]` the first derivative: what a unit
more of weight on atom k adds, which under a nonlinear layer takes the place of the atom
contribution.  `H` is symmetric; its diagonal is the curvature along one atom, its
off-diagonal part says which atoms the model reads together rather than one by one.

### `interaction_modes(model: CMFModel, mol: MoleculeFields, f: int, k: int=3) -> List[InteractionMode]`

The `k` eigenpairs of `atom_interactions` of largest absolute eigenvalue.

A mode is a pattern of weight changes over the atoms whose effect on the prediction is pure
curvature: changing the weights along it by t moves the prediction by value t^2 / 2 beyond
the linear term.  A positive value means the changes in the pattern reinforce one another,
a negative one that they saturate.  The patterns of the largest absolute values are where
the model is least additive; a mode spread over two sites is a pair the model reads
together.

## cmfqsar.oneclass

Source: `cmfqsar/oneclass.py`

One-class scores on a precomputed kernel: four geometries besides the 1-SVM's ball.

Similarity-based virtual screening fits a description of the actives alone and ranks everything
else by how well it fits (the 2011 papers on the one-class CMF approach).  The program does that
with a one-class support-vector machine, which encloses the actives in a ball in the field
Hilbert space.  A ball is one choice among several, and the others are cheap here because the
kernel is already computed:

`mean`
    distance to the centroid of the actives, `||phi(x) - mu||^2`.  With a normalised kernel
    this is one minus the mean field similarity to the training molecules: the similarity
    ensemble of the screening literature, with no hyper-parameter and nothing fitted.  It is the
    baseline the others have to beat.
`knn`
    distance to the k-th nearest active in the field Hilbert space, the classical similarity
    search, with the number of neighbours as its only parameter.
`gp`
    a Gaussian process regressed on a constant target of one over the actives (Kemmler et al.
    2013): the predictive mean falls off both where a molecule is unlike the actives and where
    the model has nothing to say, and `score="mean_over_sd"` divides by the posterior standard
    deviation to use the second effect explicitly.
`kpca`
    distance from the principal subspace of the actives (Hoffmann 2007).  A congeneric series
    lies on something much flatter than a ball, and a subspace of a few components can describe
    it far better than a radius can.

Every scorer returns a *decision value* with the same convention as the rest of the program: at
or above zero means inside the description.  The offset is placed so that a fraction `nu` of the
training molecules falls outside, which is what nu means for the 1-SVM, so the methods can be
compared without re-tuning anything.

`mean` and `gp` are linear in the kernel row, `score(x) = sum_j a_j K(x, x_j) + b`, so the
co-field, the training-molecule contributions and the atom contributions carry over unchanged.
`knn` and `kpca` are not: they are a minimum and a quadratic form, so no co-field exists for
them, and `OneClassScorer.is_linear` says so.

### `class OneClassScorer  (dataclass)`

A description of one class, fitted on the kernel of its members alone.

Attributes:

- `kind: str` = `'ocmean'`
- `nu: float` = `0.1`
- `k: int` = `5`
- `lam: float` = `0.1`
- `components: int` = `5`
- `score: str` = `'mean'`
- `coef_: Optional[np.ndarray]` = `None`
- `offset_: float` = `0.0`
- `diag_: Optional[np.ndarray]` = `None`
- `K_: Optional[np.ndarray]` = `None`
- `L_: Optional[np.ndarray]` = `None`
- `vecs_: Optional[np.ndarray]` = `None`
- `kmean_: Optional[np.ndarray]` = `None`
- `ktotal_: float` = `0.0`

#### `OneClassScorer.is_linear` -> bool  (property)

True when the score is a kernel expansion, so the decomposition rules still apply.

#### `OneClassScorer.label` -> str  (property)

How this description is named in a message to the user.

#### `OneClassScorer.why_not_linear` -> str  (property)

Why no co-field exists, in a clause that can be dropped into an error message.

#### `OneClassScorer.fit(K: np.ndarray) -> 'OneClassScorer'`

#### `OneClassScorer.decision(K_test_train: np.ndarray, k_test_self: np.ndarray, training: bool=False) -> np.ndarray`

Decision value of new molecules; at or above zero is inside the description.

`training` says that the rows are the training molecules themselves, which only the
nearest-neighbour score needs to know: a molecule is not its own neighbour.

#### `OneClassScorer.coef` -> np.ndarray  (property)

#### `OneClassScorer.bias` -> float  (property)

Constant of the kernel expansion.

For the centroid distance it absorbs the molecule's own self-kernel, which is a constant
only when the kernels are normalised -- and a one-class model normalises them, because
an unnormalised one-class score ranks by molecular size.

### `make(method: str, nu: float=0.1, **kwargs) -> OneClassScorer`

Scorer for one of `ONECLASS_METHODS` other than `ocsvm`.

### `enrichment_factor(y_true: np.ndarray, score: np.ndarray, fraction: float=0.01) -> float`

How many times more actives the top `fraction` of the ranking holds than chance.

## cmfqsar.orca

Source: `cmfqsar/orca.py`

Quantum-chemical properties from ORCA: per-atom quantities that become fields, per-molecule ones
that become descriptors.

The program's own fields are built from what it can work out itself -- Gasteiger charges, Crippen
increments, van der Waals parameters -- and that is a limit.  A quantum calculation raises it: a
Hirshfeld charge, a Fukui index or a local softness says something about an atom that no increment
table knows, and the frontier orbital energies say something about the molecule that no field can
carry.  This module runs ORCA over a set of molecules and brings both back: the per-atom values as
atom property lists, which become `prop:` fields, and the per-molecule values as molecule
properties, which become descriptors.

What it computes, with the default settings: Mulliken, Loewdin, Hirshfeld and CHELPG charges, the
HOMO and LUMO populations of every atom, and -- when the N+1 and N-1 states are run as well -- the
condensed Fukui indices f+, f- and f0, the dual descriptor, local softnesses and the local
electrophilicity.  Per molecule: the total energy, the frontier orbital energies and their gap,
the dipole, and from the ionised states the ionisation potential, electron affinity, chemical
hardness, softness, chemical potential and electrophilicity index.

Three practical things are built in rather than left to the caller.  Every job is **cached** by a
key over the molecule, its charge, the level of theory and the geometry, so an interrupted run
resumes instead of starting again -- which matters when a molecule takes minutes and a set takes
hours.  A molecule that fails is **reported and skipped**, never fatal to the set.  And the
geometry is **not optimised** unless asked: a superimposed series that is optimised molecule by
molecule is no longer superimposed, and the fields built from it would be comparing poses that
have nothing to do with each other.

The command-line front end is `examples/orca_atom_properties.py`; the application runs the same
code from *Tools -> Quantum-chemical properties with ORCA*.

### `class OrcaCancelled  — bases: RuntimeError`

Raised when the caller asked for the run to stop; not a failure of the calculation.

### `class OrcaError  — bases: RuntimeError`

### `class OrcaResult  (dataclass)`

Attributes:

- `terminated: bool` = `False`
- `converged: bool` = `False`
- `energy: float` = `float('nan')`
- `dipole: float` = `float('nan')`
- `mulliken: Optional[np.ndarray]` = `None`
- `loewdin: Optional[np.ndarray]` = `None`
- `hirshfeld: Optional[np.ndarray]` = `None`
- `chelpg: Optional[np.ndarray]` = `None`
- `spin: Optional[np.ndarray]` = `None`
- `homo: float` = `float('nan')`
- `lumo: float` = `float('nan')`
- `homo_index: int` = `-1`
- `lumo_index: int` = `-1`
- `mo_populations: Dict[int, Dict[int, float]]` = `field(default_factory=dict)`
- `symbols: List[str]` = `field(default_factory=list)`
- `coords: Optional[np.ndarray]` = `None`

#### `OrcaResult.n_atoms` -> int  (property)

#### `OrcaResult.mo_population(mo: int, n_atoms: int) -> np.ndarray`

Fraction of orbital `mo` on every atom (0 where ORCA printed nothing).

### `class Config  (dataclass)`

Attributes:

- `orca: str` = `DEFAULT_ORCA`
- `level: str` = `DEFAULT_LEVEL`
- `opt: bool` = `False`
- `fukui: bool` = `True`
- `solvent: Optional[object]` = `None`
- `nprocs: int` = `1`
- `maxcore: int` = `2000`
- `timeout: Optional[float]` = `None`
- `workdir: str` = `'orca_work'`
- `cancel: Optional[object]` = `None`

### `class MoleculeOutcome  (dataclass)`

Attributes:

- `index: int`
- `name: str`
- `mol: Optional[Chem.Mol]` = `None`
- `error: str` = `''`
- `seconds: float` = `0.0`
- `reused: bool` = `False`

### `find_orca(explicit: Optional[str]=None) -> str`

The ORCA executable: the argument, `$ORCA_EXE`, the standard location, or the PATH.

### `orca_has_mpi(exe: str) -> bool`

Whether this ORCA build can run `%pal` jobs (the serial build lacks the MPI helpers).

### `electron_count(mol: Chem.Mol, charge: int) -> int`

### `multiplicity_for(mol: Chem.Mol, charge: int) -> int`

Lowest spin state: singlet for an even number of electrons, doublet for an odd one.

### `state_from_properties(mol: Chem.Mol) -> Tuple[int, int]`

(charge, multiplicity) of a molecule, from its SD properties when it carries them.

A catalyst is Fe(II) or Fe(I) and a triplet or a quintet, and nothing in a connection table
says which; a radical electrolyte is a doublet.  So `charge` and `multiplicity` are read
from the SD properties of the record when they are given, and only otherwise taken from the
formal charges and the lowest spin state (plan item EC1).  A multiplicity whose parity does
not match the electron count is refused rather than silently corrected: it is a mistake in
the data file, and a wrong spin state gives a plausible number for the wrong state.

### `ion_multiplicity(mult: int) -> int`

The spin state of a state with one electron more or fewer, given the parent's.

Adding or removing one electron flips the parity of the electron count, so it flips the
parity of the multiplicity.  The rule here is to pair the new electron where it can be
paired, that is to take `mult - 1` when that is still a spin state and `mult + 1`
otherwise.  For a closed-shell parent this gives the doublet the Fukui functions have always
used; for a quintet parent it gives a quartet, which is the state a caller who wrote
`multiplicity 5` almost always means.

### `needs_ecp(mol: Chem.Mol) -> bool`

Whether this molecule holds an element for which a def2 basis uses a core potential.

### `solvent_lines(solvent) -> Tuple[List[str], List[str]]`

(keywords, input blocks) for a continuum solvent; `([], [])` for the gas phase.

`solvent` is the name of a solvent ORCA's CPCM list knows, a name in
`SOLVENT_CONSTANTS`, or a (dielectric constant, refractive index) pair.  A name ORCA
knows becomes the keyword `CPCM(name)`; the other two become an explicit `%cpcm` block,
because a solvent ORCA cannot name is still usable when its two constants are given.

### `make_input(level: str, charge: int, mult: int, xyz_name: str, nprocs: int=1, opt: bool=False, populations: bool=True, chelpg: bool=True, maxcore: int=2000, solvent=None) -> str`

Text of an ORCA input.

`populations` adds the Hirshfeld analysis and the per-MO Mulliken atom populations; `chelpg`
the CHELPG fit (needed for the N-electron state only).  An open-shell state is run
unrestricted; a geometry optimisation prints nothing but what it needs.  `solvent` puts the
molecule in a continuum solvent, as `solvent_lines` describes (plan item EC1).

### `xyz_block(symbols: Sequence[str], coords: np.ndarray, comment: str='') -> str`

### `read_xyz(path: str) -> Tuple[List[str], np.ndarray]`

### `cache_key(smiles: str, charge: int, level: str, opt: bool, coords: Optional[np.ndarray]=None, solvent=None, mult: Optional[int]=None) -> str`

Identity of a computation: the molecule, its state, the level and the geometry.

The SMILES is canonicalised so that the same compound written two ways shares a key; the
coordinates (to 1e-3 A) are part of it because a different starting geometry is a different
result even when nothing else changed.

`solvent` and `mult` are folded in only when they are given.  That keeps the key of a
gas-phase run at its lowest spin state exactly what it was before plan item EC1 added the
two, so the ORCA results already cached for the other benchmarks of this repository are not
thrown away.  A caller passes `mult` when it is not the lowest spin state, which is the only
case in which two runs of the same molecule in the same charge state differ.

### `parse_orca_output(text: str) -> OrcaResult`

Everything the script needs from an ORCA output, each from the last occurrence of its block.

### `derive_properties(n: OrcaResult, np1: Optional[OrcaResult]=None, nm1: Optional[OrcaResult]=None, charge: int=0, mult: int=1) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]`

Per-atom and global descriptors from the N-electron result and, if given, the N+1 / N-1 ones.

Returns (atom properties, molecule properties); the Fukui-derived entries are present only when
both ionic states are given.  Energies in eV, softness in 1/eV, dipole in Debye, charges in e.

### `annotate(mol: Chem.Mol, atom_props: Dict[str, np.ndarray], mol_props: Dict[str, float], coords: Optional[np.ndarray]=None, extra: Optional[Dict[str, str]]=None) -> Chem.Mol`

A copy of `mol` carrying the properties (and `coords`, when the geometry was optimised).

Atom properties go through RDKit's property lists, which is the only way they reach the SDF
and are read back as atom properties (`atom.dprop.<name>`); the molecule's own properties
(activity, name, ...) are kept.

### `verify_sdf(path: str, expect: Sequence[str]) -> List[str]`

Load the file as the program does and return the per-atom properties it offers.

### `safe_name(name: str, limit: int=40) -> str`

### `run_job(jobdir: str, job: str, inp: str, cfg: Config, need_converged: bool=False) -> OrcaResult`

Run (or reuse) the ORCA job `job` in `jobdir`; returns the parsed output or raises OrcaError.

### `prepare_molecule(mol: Chem.Mol) -> Chem.Mol`

All hydrogens explicit, with coordinates; what ORCA is given and what the SDF will hold.

### `run_molecule(mol: Chem.Mol, name: str, index: int, cfg: Config) -> MoleculeOutcome`

The whole chain for one molecule: geometry, ORCA jobs, parsing, the annotated copy.

### `process_dataset(path: str, out: str, cfg: Config, limit: Optional[int]=None, jobs: int=1, log=print) -> List[MoleculeOutcome]`

Run every molecule of `path` and write the annotated SDF `out`; returns the outcomes.

### `annotate_molecules(mols: Sequence[Chem.Mol], names: Sequence[str], cfg: Config, jobs: int=1, indices: Optional[Sequence[int]]=None, progress=None, cancel=None) -> List[MoleculeOutcome]`

Run ORCA over molecules already in memory and return the annotated copies.

This is what the application calls: `process_dataset` reads and writes files, which a
window cannot do while the user watches.  `indices` restricts the run to part of the set --
the selected molecules, or the ones still missing their properties -- and `progress` is
called with a message and a fraction so the window can show where the run is.

Cancelling stops the ORCA process that is running and returns what is finished; because every
job is cached, starting again continues from there rather than from the beginning.

### `missing_properties(mols: Sequence[Chem.Mol], fukui: bool=True) -> List[int]`

Which molecules do not carry the per-atom properties a run would give them.

### `estimate_seconds(n_molecules: int, fukui: bool=True, jobs: int=1, per_job: float=150.0) -> float`

A rough wall time for a run, so that a dialog can say so before it starts.

`per_job` is the time one single-point calculation takes on this machine; the Fukui indices
need the N+1 and N-1 states as well, which triples it.

### `measured_seconds_per_job(workdir: str, fukui: bool=True) -> Optional[float]`

The mean time one ORCA job took in `workdir`, from the status files a previous run left.

A dialog that has to warn about the cost is more convincing with a number measured on this
machine than with a guess, and after one run the number is available.

## cmfqsar.pbh

Source: `cmfqsar/pbh.py`

Polybenzenoid hydrocarbons: a planar frame, ring classes, ring-centred fields and per-ring
contributions.

A cata-condensed polybenzenoid hydrocarbon (PBH) is a set of fused benzene rings in which no
carbon belongs to three rings.  The COMPAS-1 data set of the Gershoni-Poranne group (Wahab et
al., J. Chem. Inf. Model. 2022, 62, 3704) enumerates every such molecule of one to eleven rings
with optimised geometries and frontier-orbital energies.  It is a good test of the overlap kernel
for two reasons: the property is a quantum-chemical one rather than a binding affinity, and the
molecules are rigid and nearly planar, so the alignment that usually decides a 3D-QSAR model is
reduced to a rule.  This module holds what that study needs beyond the ordinary machinery:

* `read_compas` builds RDKit molecules from the SMILES of the table and the coordinates
  of the xyz files, and checks that the two describe the same atoms;
* `canonical_frame` puts a molecule into its inertial frame (x along the long axis, z
  along the normal) and resolves the signs of the in-plane axes by a fixed rule, so that two
  molecules that are alike in shape are alike in position;
* `pbh_atom_type` is the atom typing of the continuous indicator fields: a ring carbon
  bonded to a hydrogen, a carbon shared by two rings, a carbon shared by three;
* `ring_classes` perceives the rings and classifies them by annulation -- terminal,
  linear, angular, branched -- and `clar_weights` gives every ring its share of the Clar
  sextets;
* `build_pbh_fields` adds one Gaussian per ring centroid to the molecular field, carried
  by the `ring:<class>` fields, which is what `ring:terminal` and the rest mean in a field set;
* `ring_contributions` sums decomposition rule 3 over the atoms of every ring.

The same machinery serves the later COMPAS sets: the peri-condensed benzenoids of COMPAS-3
(a fifth annulation class, `peri`, for a ring whose shared bonds touch) and the hetero-atom
systems of COMPAS-2 (rings of four to six atoms with B, N, O or S, typed by ring type; read by
`read_compas2` from the SDF of that set).  `ring_classes` also gives every ring its
HOMA index, the harmonic-oscillator measure of aromaticity computed from the bond lengths
alone, which the `ring:homa` field uses as weight.

Everything here is read by `benchmarks/run_compas.py`; see the manual, section "A frame by
rule: the gap of polybenzenoid hydrocarbons".

### `class Ring  (dataclass)`

One benzene ring of a benzenoid.

Attributes:

- `atoms: Tuple[int, ...]`
- `centroid: np.ndarray`
- `neighbours: Tuple[int, ...]`
- `cls: str`
- `sextet: float` = `0.0`
- `homa: float` = `float('nan')`
- `ring_type: str` = `'benzene'`

### `parse_xyz(text: str) -> Tuple[List[str], np.ndarray]`

Symbols and coordinates of one xyz block (count line, comment line, atoms).

### `mol_from_smiles_xyz(smiles: str, symbols: Sequence[str], coords: np.ndarray, name: str='', max_bond: float=1.8) -> Chem.Mol`

An RDKit molecule with the bonds of `smiles` and the coordinates of the xyz atoms.

The xyz file and the SMILES list the atoms in different orders.  Bonds are perceived from the
coordinates, the two graphs are matched (elements and connectivity, bond orders ignored), and
the coordinates are copied over the match.  A mismatch -- different elements, a graph that
does not match, a bonded pair further apart than `max_bond` angstroms after the copy -- raises
`ValueError` naming the molecule.

### `read_compas_table(csv_path: str) -> List[Dict[str, str]]`

The rows of a COMPAS-1 csv file, as dictionaries keyed by the column names.

### `read_compas(csv_path: str, geometries: str, limit: Optional[int]=None, names: Optional[Sequence[str]]=None, progress=None)`

The molecules of a COMPAS-1 set as a `cmfqsar.dataio.Dataset`.

`geometries` is the folder of xyz files or the tar.gz archive itself.  Every row of the table
becomes one molecule with the columns of `COMPAS_NUMERIC` and `COMPAS_TEXT` as
properties and `GAP_eV` as the activity.  `limit` keeps the first rows only; `names`
keeps the listed molecules only.  A molecule whose geometry does not match its SMILES stops
the reading with a `ValueError`.

### `read_compas2(csv_path: str, sdf_gz: str, limit: Optional[int]=None, names: Optional[Sequence[str]]=None, progress=None)`

The neutral molecules of a COMPAS-2 set as a `cmfqsar.dataio.Dataset`.

COMPAS-2 stores one row per molecule and charge state (-1, 0, +1) and the geometries in one
gzipped SDF with `name` and `charge` tags.  The neutral rows are kept, the geometry of
the neutral state is taken, and the SDF graph is checked against the SMILES of the table.
The columns `gap`, `homo`, `lumo` and `rings` become `GAP_eV`, `HOMO_eV`,
`LUMO_eV` and `n_rings`, so that the molecules carry the same properties as COMPAS-1.

### `heavy_mask(mol: Chem.Mol) -> np.ndarray`

### `atomic_masses(mol: Chem.Mol) -> np.ndarray`

### `inertial_frame(coords: np.ndarray, masses: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`

Centre of mass, the rows x, y, z of the inertial frame, and the three moments.

x is the axis of the smallest moment of inertia (the largest extent), z the axis of the
largest moment (the normal of a planar molecule); the frame is right-handed.

### `sign_key(xy: np.ndarray) -> np.ndarray`

The five moments that decide the signs of the in-plane axes.

Under x -> -x the first and the last two change sign; under y -> -y the second, third and
the last two do.  A molecule for which all five vanish is symmetric under the flip in
question, and its fields are then the same in either frame.

### `resolve_signs(coords: np.ndarray, heavy: Optional[np.ndarray]=None, tol: float=1e-06) -> Tuple[np.ndarray, Tuple[float, float]]`

Among the four sign choices of a frame, the one with the largest `sign_key`.

The keys are compared in order, each within a tolerance relative to the size of the
molecule; candidates still tied after the fifth key are equivalent by symmetry and the first
in `FLIPS` is taken.  Returns the coordinates and the flip applied.

### `canonical_frame(mol: Chem.Mol, coords: Optional[np.ndarray]=None, degenerate_tol: float=0.001) -> np.ndarray`

The coordinates of `mol` in its canonical planar frame.

The frame is that of the heavy-atom inertia tensor: origin at the centre of mass, x along the
axis of the smallest moment (the long axis), z along the largest (the normal).  When the two
in-plane moments are equal within `degenerate_tol` of their mean (benzene, triphenylene and
the other molecules with a three-fold or higher axis), x is put along the in-plane direction
with the largest third moment of the heavy atoms, or the largest fourth or sixth moment when
the lower ones do not depend on the direction.  The signs of x and y are then fixed by
`resolve_signs`.  Hydrogens follow the heavy atoms and take no part in the rule.

### `frame_candidates(coords: np.ndarray, z_flip: bool=False) -> List[np.ndarray]`

The four (eight with `z_flip`) frames that share the inertial axes of `coords`.

### `choose_frame_by_kernel(mol: Chem.Mol, fieldset: FieldSet, reference: AtomTable, scale: Optional[np.ndarray]=None, z_flip: bool=False, engine=None) -> Tuple[np.ndarray, np.ndarray]`

Alignment by prediction confidence: of the frames that share the inertial axes, the one
whose field is on average most similar to the molecules of `reference`.

The similarity is the cosine-normalised joint kernel, with the field scales `scale` of the
model when given.  Returns the chosen coordinates and the mean similarity of every candidate
in the order of `frame_candidates`.

### `pbh_atom_type(atom: Chem.Atom) -> str`

The carbon classes of a benzenoid: `C.H` on the rim (one ring, bonded to a hydrogen),
`C.2r` shared by two rings, `C.3r` shared by three (absent in a cata-condensed set).
Hydrogens are `H`; a hetero atom is its element symbol, with `.H` appended when it
carries a hydrogen (`N.H` of pyrrole, `B.H` of borole and borinine); `C.1r` is a ring
carbon without a hydrogen and `C` a carbon outside the rings.

### `ring_type_of(mol: Chem.Mol, atoms: Sequence[int]) -> str`

The ring type of COMPAS-2 from the ring size and its hetero atoms, or a generic
`ring:<size>:<symbols>` name for anything else.

### `ring_homa(mol: Chem.Mol, pos: np.ndarray, atoms: Sequence[int]) -> float`

The HOMA index of one ring from its bond lengths, `nan` when a bond of the ring has
no parameters (a bond to boron).

### `ring_classes(mol: Chem.Mol, coords: Optional[np.ndarray]=None, carbon_only: bool=True) -> List[Ring]`

The rings of `mol` with their centroids, annulation classes, HOMA indices and types.

With `carbon_only` (the default) only six-membered carbon rings are taken, as in a
benzenoid; otherwise every ring of four to six atoms, which is what the hetero-atom systems
of COMPAS-2 need.  Two rings are fused when they share a bond.  A ring with one fused
neighbour (or none: benzene) is `terminal`; with two, it is `linear` when the two shared
bonds face each other across the ring (anthracene's middle ring) and `angular` when they
do not (phenanthrene's; any five-membered ring); with three it is `branched`
(triphenylene's centre).  Two shared bonds that touch make a carbon of three rings, which is
peri-condensation: such a ring, and a ring fused to four or more, is `peri`.

### `ring_class_counts(rings: Sequence[Ring]) -> Dict[str, int]`

### `annulation_counts(lalas: str) -> Tuple[int, int]`

(linear, angular-or-branched) rings announced by a COMPAS annulation string.

The `lalas` column writes one letter per inner ring: `L` for a linearly annulated ring,
`A` for an angular one; a branch is written in parentheses after the ring it hangs from,
and that ring is also written `A`.  Terminal rings are not written.

### `clar_structures(mol: Chem.Mol, rings: Sequence[Ring]) -> List[Tuple[int, ...]]`

Every Clar structure with the largest number of sextets, as tuples of ring indices.

A Clar structure places aromatic sextets on rings that share no atom and covers every other
carbon by a double bond.  All independent sets of rings are enumerated (at most eleven rings
in COMPAS-1) and those whose remaining carbons admit a perfect matching are kept; the ones of
the largest size are returned.

### `clar_weights(mol: Chem.Mol, rings: Sequence[Ring]) -> np.ndarray`

The share of the maximal Clar structures in which every ring carries a sextet, in [0, 1];
also written into `Ring.sextet`.  Naphthalene gives 0.5 and 0.5, anthracene 1/3 each,
phenanthrene 1, 0, 1.

### `ring_field_weight(ring: Ring, field_name: str) -> float`

The weight of one ring in a `ring:<class>`, `ring:<type>`, `ring:sextet` or
`ring:homa` field.

### `with_ring_centres(mf: MoleculeFields, rings: Sequence[Ring], fieldset: FieldSet) -> MoleculeFields`

`mf` with one pseudo-atom per ring centroid appended, carrying the `ring:` fields.

The pseudo-atoms have weight zero in every atom field and the real atoms weight zero in the
ring fields, so the kernel of a ring field is a sum over ring pairs only.  They are appended
in the order of `rings`, which `ring_contributions` relies on.

### `build_pbh_fields(mol: Chem.Mol, fieldset: FieldSet, name: str='', rings: Optional[Sequence[Ring]]=None) -> MoleculeFields`

`cmfqsar.fields.build_fields` followed by the ring-centre pseudo-atoms when the
field set has `ring:` fields.  The Clar weights are computed when `ring:sextet` asks.

### `n_real_atoms(mf: MoleculeFields) -> int`

Atoms of `mf` that are not ring-centre pseudo-atoms.

### `ring_contributions(model, mf: MoleculeFields, rings: Sequence[Ring]) -> Tuple[np.ndarray, np.ndarray, float]`

Decomposition rule 3 summed per ring.

Every atom's contribution (`cmfqsar.model.CMFModel.atom_contributions`) goes to the
rings it belongs to, shared equally when it belongs to two; a hydrogen goes with the heavy
atom it is bonded to; a ring-centre pseudo-atom goes to its ring.  Returns the per-ring sums,
the per-atom contributions they were summed from, and what could not be placed (atoms in no
ring), which is zero for a benzenoid.  The per-ring sums plus the unplaced part equal the
prediction minus the bias.

### `count_names(atom_types: Sequence[str]=PBH_TYPES, ring_classes_: Sequence[str]=RING_CLASSES, ring_types: Sequence[str]=(), sextet: bool=True, homa: bool=False) -> Tuple[str, ...]`

The names of `count_features` for the given classes.

### `count_features(mol: Chem.Mol, rings: Optional[Sequence[Ring]]=None, atom_types: Sequence[str]=PBH_TYPES, ring_classes_: Sequence[str]=RING_CLASSES, ring_types: Sequence[str]=(), sextet: bool=True, homa: bool=False, carbon_only: bool=True) -> np.ndarray`

Counts of the atom classes, the ring classes (and ring types), the rings and the Clar
sextets, and the sum of the HOMA indices when asked: what the indicator and ring fields
reduce to when their attenuation goes to zero.

### `random_split(n: int, seed: int, fractions: Sequence[float]=(0.8, 0.1, 0.1)) -> Dict[str, np.ndarray]`

Indices of a shuffled train / validation / test split with the given fractions.

### `size_split(n_rings: Sequence[int], test_rings: int) -> Dict[str, np.ndarray]`

Train on the molecules with fewer than `test_rings` rings, test on those with exactly
`test_rings`: the size-extrapolation split.

## cmfqsar.perfield

Source: `cmfqsar/perfield.py`

A separate attenuation for every field, chosen by the evidence (plan item D6).

The attenuation alpha_f decides how far a field reaches: large alpha makes it sharp, so two
molecules must match closely before the overlap integral calls them similar, and small alpha makes
it diffuse.  There is no reason for one number to be right for all five fields.  Electrostatics is
a long-range property and its field is the one a crystallographer would draw widest; a hydrogen-bond
donor is a point with a direction, and a field around it that reaches 4 A is making a claim about
the molecule that the chemistry does not support.  The 2013 protocol nevertheless shared one alpha
across the fields, and said why: a grid over five widths is five times the search, and a grid over
one width per field is 5^5 of it.

What makes the second grid affordable is two observations rather than a faster computer.

The first is that the per-field kernels are independent.  The joint kernel is a weighted sum,
`K = sum_f h_f c_f K_f`, and K_f depends on alpha_f alone, so the kernels for a grid of A widths
over F fields are the same FA matrices the shared search already computes -- it simply throws away
every combination but the diagonal one.  Assembling the kernel for any assignment of widths to
fields is then a weighted sum of matrices already in memory, not a rebuild.

The second is that the evidence (`cmfqsar.evidence`) scores an assignment with one
factorisation instead of ten, which is what item B4 measured.  A coordinate descent over the
widths -- hold the others, try every width for one field, keep the best, and go round again --
costs `passes * F * A * L` factorisations, a few hundred, where the same descent on a
cross-validated score would cost ten times that and the full grid 5^5 times.

What this does not do is hold anything out, and over 5^5 assignments the evidence has room to
favour itself.  The descent is therefore deliberately short (two passes by default, from the best
shared width rather than from nothing), and whether the extra freedom helps is a question for a
test set, not for the criterion that chose it: see `benchmarks/run_perfield.py`.

Fields whose kernel does not depend on alpha at all -- the probe fields, whose widths come from
the expansion in `cmfqsar.probes` -- are detected and skipped rather than searched uselessly.

### `class PerFieldResult  (dataclass)`

The outcome of the descent: the widths, the regularisation that goes with them, and the
trace, including the shared-width configuration it started from so that the gain in evidence
is readable off the result.

Attributes:

- `alphas: Dict[str, float]`
- `params: Dict[str, float]`
- `log_evidence: float`
- `shared_alpha: float`
- `shared_params: Dict[str, float]`
- `shared_log_evidence: float`
- `tuned: List[str]` = `_field(default_factory=list)`
- `skipped: List[str]` = `_field(default_factory=list)`
- `rows: List[dict]` = `_field(default_factory=list)`
- `n_eval: int` = `0`
- `passes_used: int` = `0`

#### `PerFieldResult.gain` -> float  (property)

How much log evidence the per-field widths gained over the shared one.

#### `PerFieldResult.is_shared` -> bool  (property)

True when the descent came back to one width for every field.

### `assemble(cache: Dict[float, tuple], choice: Sequence[float]) -> Tuple[np.ndarray, Optional[np.ndarray]]`

The per-field kernel stack for one width per field, taken out of the shared-search cache.

`cache` maps a width to `(K_fields, scale, self_diag)` as `cmfqsar.model.grid_search`
stores it, with the fields along the first axis; `choice` gives the width of each field in
that order.  The normalisation diagonals are selected field by field in the same way, because
a field normalised at one width and kerneled at another is not a kernel at all.

### `tunable_fields(cache: Dict[float, tuple], alphas: Sequence[float], enabled: Sequence[bool]) -> Tuple[List[int], List[int]]`

Which fields the width actually moves: the enabled ones whose kernel changes with alpha.

A probe field's kernel is built from its own expansion and ignores alpha, so searching over
its width spends factorisations to rediscover the same number; a disabled field (h = 0) is
multiplied out of the joint kernel entirely.  Both are reported as skipped rather than
silently tuned.

### `descend(train, y: np.ndarray, model, cache: Dict[float, tuple], alphas: Sequence[float], grid: Sequence[Dict[str, float]], start_alpha: float, start_params: Dict[str, float], start_evidence: Optional[float]=None, passes: int=2, progress: Optional[Callable[[str, float], None]]=None, cancel: Optional[Callable[[], bool]]=None) -> PerFieldResult`

Coordinate descent on the evidence over one attenuation per field.

Starts from `start_alpha` for every field -- the width the shared search chose, so the descent
can only improve on the configuration the program would have used -- and visits the tunable
fields in turn, trying every width of `alphas` and every regularisation of `grid` and keeping
whatever the evidence scores best.  A pass that changes nothing ends the search.

The model is left with the chosen widths, the kernels assembled from them and the
regularisation that goes with them, so a fit follows immediately.  `cache` is not modified.

## cmfqsar.periodic

Source: `cmfqsar/periodic.py`

Continuous molecular fields of crystals: the periodic field autocorrelation.

The autocorrelation kernel of `invariant.py` compares two molecules by the distance
distributions of their field weights, which is the real-space form of the orientation-averaged
product of their typed structure factors (the Debye scattering equation).  A crystal is the same
object with three differences, and this module is those differences:

* **The pair sum runs over a neighbour list.**  A molecule has finitely many pairs; a crystal has
  one atom per cell times every neighbour out to a cutoff radius, periodic images included, and
  the sum is divided by the number of atoms in the cell so that the descriptor is intensive: a
  supercell describes the same material, and gives the same channels.

* **The uniform background is removed.**  In an infinite solid the number of pairs at distance r
  grows as r^2, the same for every structure, and dominates everything else at long distance.  What
  the Faber-Ziman partial structure factors keep is the deviation from a uniform distribution of
  each field over the cell volume, and that is what is subtracted here, pair of field types by
  pair of field types.  The l = m self term stays, as it does in the molecular descriptor: it
  carries the composition, the mean of w_f w_g over the atoms.

* **The cutoff is tapered.**  A hard cutoff produces oscillations; the last `taper` angstrom
  before `r_max` are weighted by a cosine that falls smoothly to zero, so that a pair leaving the
  sphere leaves the descriptor continuously.

Everything else is shared: the Gaussian along the distance axis, the pair-of-field-types
channels, the per-block scaling and cosine normalisation of the kernel, the channel labels, and
the learners.  The descriptor is invariant to rotation, translation, the choice of origin and of
unit cell by construction, and cannot see what any pair-distance distribution cannot see:
enantiomorphs and homometric structures look the same, and so does anything beyond `r_max`.

Fields for inorganic solids
---------------------------
The classical hydrophobic and hydrogen-bond fields mean nothing in an oxide.  What carries over
is the *electrostatic* field with oxidation states as weights, the *steric* field with the atomic
volume (the cube of the covalent radius), the *electronegativity*, the atomic *mass* or its inverse
square root (`prop:mass`, `prop:inverse_sqrt_mass`) as property fields, and above all the
*continuous indicator fields* of the 2015 paper, one per element: their pair products are
the partial structure factors of each element pair, which is what a diffraction experiment
measures.  `inorganic_fieldset` builds such a set and `crystal_fields` evaluates the
weights for one structure.  Any per-atom array a structure file carries (Bader charges, magnetic
moments, ...) can be a field as `prop:<name>`.

Partial occupancy (plan item M2)
-------------------------------
A disordered site -- a Li site occupied in 54 % of the cells, a site shared by Zr and Ta -- is
a configurational ensemble, and the kernels of this program treat an ensemble by its mean field
(the mean-map kernel of `cmfqsar.fields.combine_members`).  When the sites are occupied
independently of each other, the mean field over all configurations is exact and costs nothing:
the weight of every field at a site is the occupancy-weighted mean of the weights of the species
that share it, `w_f = sum_s p_s w_f(s)`, which is what `crystal_fields` computes when a
structure carries its site species (`site_species`; `read_disordered` reads a CIF
with partial occupancies through pymatgen).  The overlap kernel is bilinear in the fields, so
the kernel of two mean fields is the mean of the kernel over the two ensembles, exactly.  The
periodic autocorrelation is quadratic in one field, so its mean over the ensemble differs from
the descriptor of the mean field where a site meets itself: in the self term at zero distance,
in the pairs of a site with its own periodic images and in the background, where the mean of a
product is not the product of the means,
`E[w_f w_g] = sum_s p_s w_f(s) w_g(s)` on each site.  `CrystalFields` carries those
second moments and `periodic_autocorrelation` uses them, so that the descriptor is the
exact mean over the configurations as well, per expected atom of the cell.  A structure without disorder carries none of this and goes through the
same code unchanged.

Layers and chains (plan item M1)
--------------------------------
A structure periodic in two directions (a monolayer, a slab) or in one (a polymer chain, a
nanowire) goes through the same neighbour list, which respects the periodicity flags of the
structure (`pbc`), so no pair is counted across the vacuum whatever its width.  Two things do
depend on the number of periodic directions, and both read the flags:

* **The uniform background.**  What is subtracted is the pair histogram of the same weights
  spread uniformly over the *periodic* directions only; across the others every atom keeps its
  place.  For a crystal that is the shell integral over the cell volume; for a layer the count
  of pairs at distance r between two atoms a height h apart grows as a ring, 2 pi r dr over the
  cell area, for r >= h; for a chain as 2 r dr / sqrt(r^2 - t^2) over the repeat length, with t
  the separation from the axis.  `_background_profile` holds those integrals per channel
  and `CrystalFields.measure` the volume, area or length that divides them.  The
  descriptor then does not change with the vacuum, with an in-plane supercell or with a shift
  of the layer along the normal.

* **The height channels of a layer.**  The distance descriptor cannot see the normal.  With
  `z_max > 0` in the spec the descriptor of a layer is resolved by the in-plane distance and
  the heights of both atoms above the layer's mean plane, as the surface kernel of
  `invariant.py` resolves a molecule above a surface, with the background removed per unit
  area.  A free-standing layer has no physical "up", so the descriptor is averaged over the two
  faces unless the spec says `oriented`; an oriented spec keeps the file's own up (the third
  lattice vector) and tells a Janus layer from its mirror image.

`centre_layer` gathers a layer that a database file leaves split across the cell boundary.
The images of the periodic overlap kernel (`image_translations`) read the same flags, so
a layer is never compared with its own copy across the vacuum.

ASE is needed for the neighbour list and to read structure files; pymatgen is needed for the
oxidation-state guess, the electronegativities and the reading of partial occupancies and
nothing else.  Both are imported lazily, so that the rest of the program does not depend on
them.

### `class PeriodicSpec  (dataclass)`

The distance grid, cutoff, taper and background treatment of the periodic autocorrelation.

The grid is laid out like `invariant.AutocorrelationSpec`: channels at 0, step, 2 step,
... up to `r_max`, each a Gaussian of width `sigma` along the distance axis.  `chirality`
exists only so that the block and label helpers of `invariant.py` accept this spec; the
two-point descriptor of a crystal has no signed block.

Attributes:

- `r_max: float` = `10.0`
- `step: float` = `0.5`
- `sigma: float` = `0.5`
- `include_self: bool` = `True`
- `background: bool` = `True`
- `taper: float` = `2.0`
- `chirality: bool` = `False`
- `z_max: float` = `0.0`
- `z_step: float` = `1.0`
- `z_sigma: float` = `0.0`
- `oriented: bool` = `False`

#### `PeriodicSpec.d_max` -> float  (property)

#### `PeriodicSpec.has_heights` -> bool  (property)

#### `PeriodicSpec.heights` -> np.ndarray  (property)

The height channels, symmetric about the mean plane; one channel at 0 without them.

#### `PeriodicSpec.height_sigma` -> float  (property)

#### `PeriodicSpec.centres` -> np.ndarray  (property)

#### `PeriodicSpec.n_bins` -> int  (property)

Channels per pair of field types: the distances, times the heights squared for a
layer with height channels.

#### `PeriodicSpec.window(r: np.ndarray) -> np.ndarray`

1 inside the sphere, a half-cosine over the last `taper` angstrom, 0 beyond.

#### `PeriodicSpec.shell_integral(n_points: int=4000) -> np.ndarray`

`∫ 4π r² exp(-(r - d_k)² / 2σ²) w(r) dr` for every channel d_k.

The pair count a uniform distribution of unit density puts into each channel.  It depends
on the grid, the width and the window only, so it is computed once per spec.

#### `PeriodicSpec.ring_integral(n_points: int=4000) -> np.ndarray`

`∫ 2π ρ exp(-(ρ - d_k)² / 2σ²) w(ρ) dρ` for every channel: the in-plane analogue of
the shell integral, the pair count a unit density spread over a plane puts into each
channel of the in-plane distance.

### `class CrystalFields  (dataclass)`

Atom-level data of one periodic structure: what `fields.MoleculeFields` is to a
molecule.  It carries what an `fields.AtomTable` and the periodic kernel need, and
the unit cell.

Attributes:

- `name: str`
- `coords: np.ndarray`
- `weights: np.ndarray`
- `radii: np.ndarray`
- `symbols: List[str]`
- `cell: np.ndarray`
- `pbc: Tuple[bool, bool, bool]` = `(True, True, True)`
- `charges: Optional[np.ndarray]` = `None`
- `charge_source: str` = `''`
- `atom_types: List[str]` = `field(default_factory=list)`
- `bonds: np.ndarray` = `field(default_factory=lambda: np.zeros((0, 2), dtype=int))`
- `molblock: str` = `''`
- `n_conformers: int` = `1`
- `conformer_weights: Optional[np.ndarray]` = `None`
- `atoms_per_conformer: int` = `0`
- `descriptors: Optional[np.ndarray]` = `None`
- `probe_coeffs: Dict[str, np.ndarray]` = `field(default_factory=dict)`
- `occupancy: Optional[np.ndarray]` = `None`
- `site_species: Optional[List[Dict[str, float]]]` = `None`
- `second_moments: Optional[np.ndarray]` = `None`
- `n_defects: Optional[int]` = `None`
- `relative_to: str` = `''`
- `centre: Optional[int]` = `None`

#### `CrystalFields.n_atoms` -> int  (property)

#### `CrystalFields.is_ensemble` -> bool  (property)

#### `CrystalFields.is_disordered` -> bool  (property)

#### `CrystalFields.volume` -> float  (property)

#### `CrystalFields.n_periodic` -> int  (property)

How many lattice directions are periodic: 3 for a crystal, 2 for a layer, 1 for a
chain, 0 for a molecule in a box.

#### `CrystalFields.measure` -> Optional[float]  (property)

The size of the cell along its periodic directions: the volume of a crystal, the area
of a layer, the repeat length of a chain; None for a structure with no periodic
direction, which has no uniform background to remove.

#### `CrystalFields.to_atoms()`

### `periodic_measure(cell, pbc) -> Optional[float]`

The volume, area or repeat length of a cell along its periodic directions; None when
no direction is periodic.

### `layer_normal(cell, pbc) -> np.ndarray`

The unit normal of a structure periodic in two directions, pointing along the
non-periodic lattice vector: the file's own "up".

### `centre_layer(atoms)`

A copy of `atoms` with the structure gathered and centred along every non-periodic
lattice direction.

A layer read from a database may sit across the cell boundary along the normal, one part at
the bottom of the cell and the rest at the top, which breaks the heights and the neighbour
list of an open direction.  Along each non-periodic direction the atoms are shifted so that
the widest empty gap of the fractional coordinate, taken around the cycle, straddles the
boundary, and the structure sits at the middle of the cell.  A structure periodic in every
direction is returned unchanged.

### `periodic_autocorrelation(crystal: CrystalFields, spec: PeriodicSpec=PeriodicSpec(), pairs: Optional[Sequence[Tuple[int, int]]]=None) -> np.ndarray`

Periodic field autocorrelation of one structure, flattened: (n_pairs * n_bins,).

For a pair of field types (f, g) and a channel at distance d_k,     ::

    Phi_fg(d_k) = (1/N) [ sum_{l in cell} sum_{m: r_lm < r_max} w_lf w_mg
                                 exp(-(r_lm - d_k)^2 / 2 sigma^2) w(r_lm)
                           - S_f S_g / V * I_k ],

with N the atoms in the cell, S_f the sum of the weights of field f over the cell, V its
volume and I_k the shell integral of the spec.  The sum runs over ordered pairs, so a pair of
different field types is counted in both orders, as the molecular descriptor counts it; the
background of such a pair is accordingly 2 S_f S_g / V * I_k.

For a structure periodic in fewer than three directions the pair sum is the same (the
neighbour list respects the periodicity flags) and the background is the one of a uniform
spread over the periodic directions alone,
`(1/M) sum_{l,m} w_lf w_mg J_k(h_lm)`, with M the area or repeat length of the cell, h_lm
the separation of the pair across the non-periodic directions and J_k the ring or axis
integral of `_background_profile`; it reduces to the form above for a crystal.  With
height channels in the spec (`z_max > 0`) a layer gets `slab_autocorrelation`.

### `slab_autocorrelation(crystal: CrystalFields, spec: PeriodicSpec, pairs: Optional[Sequence[Tuple[int, int]]]=None) -> np.ndarray`

The descriptor of a layer resolved by height: (n_pairs * n_rho * n_z * n_z,) flattened.

For a pair of field types (f, g), an in-plane distance channel rho_k and two height
channels z_a, z_b,

    Phi_fg(rho_k, z_a, z_b) = (1/N) [ sum_l sum_{m: rho_lm < r_max} w_lf w_mg
                                          exp(-(rho_lm - rho_k)^2 / 2 sigma^2) w(rho_lm)
                                          Z_a(z_l) Z_b(z_m)
                                    - (R_k / A) sum_l w_lf Z_a(z_l) sum_m w_mg Z_b(z_m) ],

with rho_lm the in-plane distance of atom l to atom or image m, z_l the height of atom l
above the mean plane of the layer, Z_a a Gaussian of width `height_sigma` about the
height channel z_a, R_k the ring integral of the spec and A the area of the cell.  The
background is exact: a uniform spread over the plane puts 2 pi rho d rho / A pairs at every
in-plane distance whatever the heights.  A pair of different field types is summed in both
orders, f at z_a with g at z_b and g at z_a with f at z_b, as the surface kernel of
`invariant.py` sums it; the l = m terms are kept where `include_self` is set; a
disordered site enters with its second moments as in `periodic_autocorrelation`.
Unless the spec is `oriented` the result is the mean over the two faces of the layer,
which is the same descriptor with both height axes reversed.  It is invariant to the
choice of cell, origin and in-plane rotation, to the vacuum and to a shift of the layer
along the normal, and it carries strictly more than the distance descriptor: the distance
is a function of the in-plane distance and the two heights.

### `periodic_feature_matrix(crystals: Sequence[CrystalFields], spec: PeriodicSpec=PeriodicSpec(), n_fields: Optional[int]=None, progress=None) -> np.ndarray`

Periodic autocorrelation features of a set of structures, one row each.

### `periodic_block_sizes(spec: PeriodicSpec, n_fields: int) -> List[int]`

### `periodic_channel_labels(fieldset: FieldSet, spec: PeriodicSpec) -> List[str]`

One label per feature: which pair of field types, at which distance; for a layer with
height channels, at which in-plane distance and which two heights; for the reciprocal-space
channels of plan item M6, at which reciprocal length; for the three-body channels of plan
item M10, at which two radii from a centre and how far apart.

### `image_translations(cell: np.ndarray, r_max: float, pbc: Sequence[bool]=(True, True, True)) -> np.ndarray`

Every lattice translation that can bring an atom of the cell within `r_max` of the cell.

The bound is the perpendicular width of the cell along each axis: a translation further than
`r_max` plus the cell's own diameter cannot reach it.  The translations are generated, not
searched, and `cell_images` then drops the ones whose atoms stay too far away.  No
translation is made along a direction that is not periodic (`pbc`), so a layer is never
paired with its own copy across the vacuum.

### `cell_image_indices(crystal, r_max: float) -> Tuple[np.ndarray, np.ndarray]`

(coordinates, source index) of the atoms of one cell and of every image within `r_max`.

The first rows are the cell itself in its own order, and every later row is an image of the
atom whose index stands beside it.  Everything an image atom carries besides its position --
the field weights, the radius, the element -- is therefore read from the cell through that
index: `cell_images` takes the weights and radii that way, and a viewer takes the
element and its colour.

"Within `r_max`" is measured from the box the cell spans, not from the atoms of this
structure, so the same expansion serves every structure of the series: an image that is
needed by a neighbour's atom is kept even where this structure has no atom near it.

### `cell_images(crystal, r_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`

(coordinates, weights, radii) of the atoms of one cell and of every image within `r_max`.

The expansion itself is `cell_image_indices`; the weights and radii of an image are
those of the atom it is an image of.

### `overlap_cutoff(specs: Sequence[FieldSpec], radii_max: float=2.0) -> float`

The radius beyond which no field of `specs` contributes to an overlap.

It is the cutoff rule of plan item B2, read from `cmfqsar.kernels`: the distance at
which a Gaussian pair overlap has fallen to 1e-6 of its peak, for the widest Gaussian any
field in the set puts on an atom.  Using it as the image radius makes the periodic sum as
exact as the open-boundary kernel with the same cutoff, and no more expensive than it has
to be.

### `expanded_cell_table(table: AtomTable, r_max: float) -> AtomTable`

`table` with the periodic images of every cell added, as one table of the same molecules.

The molecule boundaries are kept, so the kernel of a structure against this table is summed
per structure exactly as against the cells themselves.  The expansion is cached on the table,
because a grid search asks for it once per attenuation and it does not depend on one.

### `periodic_overlap_kernels(P: AtomTable, Q: AtomTable, specs: Sequence[FieldSpec], r_max: Optional[float]=None, cutoff=None, engine=None, symmetrise: bool=True, progress=None, cancel=None, symmetry=None) -> np.ndarray`

Per-field periodic overlap kernels of two tables of crystals: shape (F, nP, nQ).

The atoms of `P` stay in their cell and those of `Q` are expanded over the periodic images
within `r_max` (by default `overlap_cutoff`, the radius at which the widest field of
the set has fallen to 1e-6 of a peak overlap).  Everything else is
`cmfqsar.kernels.kernel_matrices`, so `cutoff` and `engine` mean what they mean there.

The two sets must stand on one parent lattice: the kernel compares the fields point by point
inside the cell, and two different lattices put the same site at different places, which is
the frame problem of a molecular series in another guise (`lattice_spread` reports how
far from one lattice a set is).

**Why the result is symmetrised.**  The integral runs over the cell of the row and the images
of the column, so with two different cells the two orders integrate over different domains
and do not agree; the average of the two is the symmetric form, and it is what a kernel has
to be.  On one lattice the two orders are equal and the average changes nothing, so the size
of the correction is a measure of how far the set is from one lattice
(`overlap_asymmetry` reports it).  `symmetrise=False` returns the one-sided sum and
costs half as much for a block of two different tables.

**The host symmetry** (plan item M14).  `symmetry` is the pair `(rotations, translations)`
of `host_symmetry`: the operations of the space group of the host lattice, in
fractional coordinates.  With it the kernel is the average of `K(P, g Q)` over the
operations `g`, which is the kernel of the orbit-averaged fields and so a kernel in its
own right.  It is needed for a series on one host, because two arrangements of the same
defects that differ by a symmetry operation of the host -- the same vacancy pair two sites
along -- have the same property but, in a fixed frame, fields that do not overlap at all.
The average makes them identical, and leaves every arrangement that is not equivalent as
distinct as the fields make it.  Without `symmetry` the kernel is the fixed-frame one.

### `overlap_asymmetry(table: AtomTable, specs: Sequence[FieldSpec], r_max: Optional[float]=None, **kw) -> float`

The largest relative disagreement between the two orders of the periodic overlap.

Zero on one lattice.  It grows with the spread of the cells, because the integral of the
row's cell and the integral of the column's cell are then not the same integral, and it says
how much of the kernel is an artefact of that spread rather than of the structures.

### `periodic_self_overlaps(table: AtomTable, specs: Sequence[FieldSpec], r_max: Optional[float]=None, symmetry=None) -> np.ndarray`

`K_f(i, i)` of every crystal of `table` under the periodic overlap: shape (F, n).

This is what the cosine normalisation divides by, and it is where the cell size goes: a
supercell of the same structure has a proportionally larger self-overlap, so the normalised
kernel does not know which cell was chosen.  `symmetry` is as in
`periodic_overlap_kernels`: the self-overlap is then that of the orbit-averaged field.

### `lattice_spread(table: AtomTable) -> Dict[str, float]`

How far a set of crystals is from standing on one lattice.

Returns the mean cell volume and the largest relative deviation from the mean of the volume
and of each lattice vector's length.  The periodic overlap kernel is a point-by-point
comparison inside the cell, so a set whose lattice constants differ by more than a few per
cent is being compared in a frame that does not quite exist -- the crystal form of a bad
alignment, and a number a benchmark should state rather than assume.

### `host_symmetry(atoms, symprec: float=0.001) -> Tuple[np.ndarray, np.ndarray]`

The symmetry operations of a host cell: `(rotations, translations)` in fractional
coordinates, shapes (G, 3, 3) and (G, 3), from spglib.

For a supercell they include the translations by the vectors of the primitive cell, so the
group of an 8 x 8 cell of a monolayer with twelve point operations has 768 members.  The
identity is always among them.

### `symmetry_expanded_table(table: AtomTable, rotations: np.ndarray, translations: np.ndarray) -> AtomTable`

`table` with every structure replaced by the mean of its images under the operations.

Every atom is copied once per operation at 1/G of its weight, so the field of the new
structure is the orbit average of the old one, and the overlap of a structure with it is
the average of the overlaps with the G transformed copies.  The structure boundaries are
kept, and the expansion is cached on the table, because a grid search asks for it once per
attenuation and it does not depend on one.

### `nearest_image_cutoff(table: AtomTable, r: float, spread: float=0.25) -> float`

The largest cutoff at which the nearest image of a pair is the only one within reach.

The compiled path of `symmetry_averaged_kernels` takes every atom pair at its
nearest-image displacement instead of summing over the images, which is exact when no
second image lies within the cutoff: the cutoff must then stay under half the cell's width
along every axis the atoms span.  An axis the atoms do not span -- the vacuum direction of
a slab, where their fractional spread is below `spread` -- puts no limit, because the
nearest image along it is the atom itself.  Returns `r` capped accordingly.

### `symmetry_averaged_kernels(P: AtomTable, Q: AtomTable, specs: Sequence[FieldSpec], symmetry, gammas: Sequence[float]=(), mixing: Optional[np.ndarray]=None, r_max: Optional[float]=None, engine=None, self_p: Optional[np.ndarray]=None, self_q: Optional[np.ndarray]=None, progress=None) -> Tuple[np.ndarray, np.ndarray]`

The periodic overlap of two tables of difference fields averaged over the host
symmetry, in both forms at once: `(K_lin, K_layer)`.

`K_lin`, shape (F, nP, nQ), is `mean_g K_f(P_i, g Q_j)`: the linear average, the kernel
of the orbit-averaged fields, which `periodic_overlap_kernels` with `symmetry` also
gives.  It is additive over the defects -- a vacancy pair is two single vacancies wherever
they sit -- so it carries the counts of each kind of defect and nothing else.

`K_layer`, shape (len(gammas), nP, nQ), is the Gaussian layer averaged instead:
`mean_g exp(-gamma (K(P_i, P_i) + K(Q_j, Q_j) - 2 sum_f mixing_f K_f(P_i, g Q_j)))`.
This is the kernel that sees the *arrangement* of a defect series: the layer is not
additive, so the pair and triple structure survives the average, and the average is still
a kernel, the inner product of orbit-averaged features in the layer's own feature space.

`mixing` are the per-field coefficients of the joint kernel (the field heights divided by
the field scales, as the model forms them); None weights the fields equally after scaling
each by its mean self-overlap over `P`.  `gammas` are in absolute units of the joint
kernel's squared distance; `cmfqsar.model.median_gamma_scale` gives the
median-distance unit.  `self_p`, `self_q` are the joint self-kernels, the same under every
operation, computed here when not given.  The structures of `P` and `Q` must share one
cell, and `r_max` is the image radius (by default `overlap_cutoff`).

Two implementations agree to the truncation of the image sum: the compiled loop of
`cmfqsar.fastkernel` (every atom pair at its nearest-image displacement, the cutoff
capped by `nearest_image_cutoff`), which is the one for a set of thousands, and the
NumPy reference, one kernel evaluation per operation on the image-expanded table, which
`engine="numpy"` selects.

### `symmetry_layer_kernel(P: AtomTable, Q: AtomTable, specs: Sequence[FieldSpec], symmetry, gammas: Sequence[float], **kw) -> np.ndarray`

`K_layer` of `symmetry_averaged_kernels`: shape (len(gammas), nP, nQ).

### `match_sites(atoms, host, tol: float=0.1) -> np.ndarray`

For every atom of `atoms`, the index of the site of `host` it sits on, or -1.

Distances are taken in the cell of `host` with the minimum-image rule, so an atom written
on the other side of the cell still finds its site.  Two atoms on one site is an error,
because a difference field would then count that site twice.

### `defect_fields(atoms, host: CrystalFields, fieldset: FieldSet, name: str='', tol: float=0.1) -> CrystalFields`

The field of `atoms` minus the field of its `host`, as a structure of the sites that
changed: a vacancy is the host's atom with its weights negated, a substitution is that and
the new atom with its own weights, an added atom is the new atom alone.

The overlap kernel is bilinear, so the kernel of two differences, plus terms that depend on
one structure at a time, is the kernel of the two whole cells; a model with a bias on the
differences is therefore the same model as one on the cells, at a cost that depends on the
number of defects and not on the size of the supercell.  The charge of an atom that
replaces another is the host's charge for the site it occupies, so that the electrostatic
difference is local, as the structural one is; an atom on no site of the host takes the
oxidation state of its element in the host's composition, or zero when none is assigned.

`host` is the `CrystalFields` of the pristine cell on the same lattice
(`crystal_fields`); `atoms` must stand on that lattice within `tol` Angstrom.

### `defect_table(structures: Sequence, host, fieldset: FieldSet, names: Optional[Sequence[str]]=None, tol: float=0.1) -> AtomTable`

An `fields.AtomTable` of difference fields (`defect_fields`) of a series on
one host.  `host` is an ASE `Atoms` of the pristine cell or its `CrystalFields`.

### `is_difference_table(table: AtomTable) -> bool`

True when every structure of `table` is a difference from a host.

### `defect_count(table: AtomTable) -> np.ndarray`

The number of sites that changed in every structure of a difference table: a vacancy
or an added atom counts one, a substitution one (its two rows share a site).

`defect_fields` records the number; for a record without it the rows are counted,
where a `+` row at the coordinate of a `-` row is a substitution and shares its site.

### `inorganic_fieldset(elements: Sequence[str]=(), physchem: Sequence[str]=INORGANIC_FIELDS, charge_model: str='oxidation', other: bool=True, alpha: float=1.0) -> FieldSet`

A field set for solids: the physico-chemical fields named, one indicator field per element
listed (`cif:<symbol>`) and, with `other`, one more for every element not listed.

`charge_model` says where the electrostatic weights come from: `"oxidation"` guesses
oxidation states from the composition (pymatgen), falling back to the electronegativity proxy
when no assignment exists; `"electronegativity"` uses that proxy throughout: the deviation
of each atom's Pauling electronegativity from the mean over the cell, with the sign of a
charge (an electronegative atom is negative); `"file"` reads `oxi_states`, `charges` or
`bader_charges` (what `cmfqsar.vasp` writes) from the structure's per-atom arrays, or
its initial charges, and refuses when there are none; `"dft"` reads the same arrays, taking
`bader_charges` first, and falls back to `"oxidation"` when the structure carries none,
which is what a series of computed cells wants (plan item BT1).

### `elements_in(structures: Iterable, min_structures: int=5) -> List[str]`

The element symbols present in at least `min_structures` of the structures, sorted.

### `electronegativity(symbol: str) -> float`

Pauling electronegativity (pymatgen's table); NaN where none is defined (noble gases).

### `oxidation_states(symbols: Sequence[str], composition: Optional[Dict[str, float]]=None) -> Optional[Dict[str, float]]`

A charge-balanced assignment of common oxidation states for the composition, or None.

pymatgen's `oxi_state_guesses` on the reduced composition; the first guess is taken,
which is the one with the most common states.  Elemental solids get zero.  `composition`
gives the amounts of the elements directly, which a disordered structure needs, since its
composition is fractional and not a count of its symbols.

### `crystal_charges(atoms, model: str='oxidation') -> Tuple[np.ndarray, str]`

Per-atom charges for the electrostatic field of a solid, and where they came from.

### `site_species(atoms) -> Optional[List[Dict[str, float]]]`

The species sharing every site with their occupancies, or None for an ordered structure.

Two conventions are read: `atoms.info["site_species"]`, one dictionary per site, which
`read_disordered` writes, and ASE's own CIF convention, `atoms.info["occupancy"]`
keyed by the `spacegroup_kinds` of the atoms.  A structure in which every site holds one
species at full occupancy is ordered and gives None, whichever convention it came with.

### `composition_of(atoms) -> Dict[str, float]`

The amounts of the elements in the cell, occupancies included.

### `read_disordered(path: str, occupancy_tolerance: float=1.05, site_tolerance: float=0.001)`

A CIF with partial occupancies as an ASE `Atoms` carrying its site species (M2).

pymatgen reads the occupancies; every site gets its majority species as the symbol, its
total occupancy in `arrays["occupancy"]` and the species that share it with their
occupancies in `info["site_species"]`, which `crystal_fields` turns into
occupancy-weighted field weights.  `site_tolerance` is kept small so that split positions a
fraction of an angstrom apart stay separate sites instead of being merged into one with an
occupancy above one; `occupancy_tolerance` is what a site may sum to before the file is
refused.  An ordered CIF comes back as an ordinary structure.

### `crystal_fields(atoms, fieldset: FieldSet, name: str='') -> CrystalFields`

Evaluate the field weights of one periodic structure (an ASE `Atoms`).

A structure with partial occupancies (`site_species`) gets the occupancy-weighted mean
of the weights of the species sharing each site, and the second moments the self term of the
periodic autocorrelation needs; see the module docstring.

### `crystal_table(structures: Sequence, fieldset: FieldSet, names: Optional[Sequence[str]]=None) -> AtomTable`

An `fields.AtomTable` of periodic structures, which the model takes like any other.

### `read_structures(path: str, index: str=':') -> list`

Every structure of an ASE-readable file (CIF, POSCAR, extended XYZ, ASE JSON, ...).

A CIF whose sites carry partial occupancies is read through `read_disordered`, so that
the occupancies reach the fields; every other file goes through ASE as it is.

### `structure_files(folder: str) -> List[str]`

The structure files of a folder, sorted by name.

A file is taken when its name ends in one of `STRUCTURE_SUFFIXES`, or when it is called
POSCAR or CONTCAR, which VASP writes without an ending.  Nothing is read here, so a folder
that also holds notes, CSV files and pictures costs nothing.

### `pbc_flags(letters: str) -> Tuple[bool, bool, bool]`

Which lattice directions are periodic, from letters: `abc`, `ab`, `a`, or `xyz`.

`abc` is a crystal, `ab` a layer with the vacuum along the third vector, `a` a chain.

### `read_crystal_set(path: str, pbc: Optional[str]=None, index: str=':') -> Tuple[list, List[str]]`

A set of periodic structures with a name for each: one file, or a folder of files.

A folder is read file by file (`structure_files`) and every structure is named after
its file, which is how a set of CIFs downloaded from a database arrives.  A single file may
hold one structure or many; a structure that carries a `name` info key keeps it, and the
rest are named `<formula>_<index>`.

`pbc` sets which lattice directions are periodic for every structure
(`pbc_flags`); without it the files decide.  A structure that is not periodic in all
three directions is gathered about the middle of its cell (`centre_layer`), because a
layer or a chain a file leaves split across the cell boundary has no mean plane.

### `crystal_targets(structures: Sequence, names: Sequence[str], key: Optional[str]=None, csv: Optional[str]=None, column: Optional[str]=None) -> np.ndarray`

The target value of every structure, from an info key of the files or from a CSV.

`key` reads `info[key]` of each structure, which is where an extended-XYZ or ASE JSON
file keeps a per-structure property; `csv` reads a table whose first column holds the
names and matches it on them, taking the second column or the one `column` names.  A
structure with no value gets NaN, and the caller decides what to do with it.

### `structures_from_matbench(table: dict, target: Optional[str]=None)`

The structures and target of a Matbench task file (`{"index", "columns", "data"}`,
structures as pymatgen dictionaries), as ASE `Atoms` and a float array.  Partial
occupancies are not handled; the Matbench tasks have none.

### `channel_weights(model) -> np.ndarray`

The weight of every channel in a fitted linear model on an invariant kernel.

The kernel is the dot product of the scaled, unit-normalised channel vectors, so the
prediction is `b + sum_c w_c phi_c(x)` with `phi` the scaled and normalised channels of
`x` and `w = sum_j a_j phi_j` over the training structures.  That weight vector is to the
autocorrelation kernels what the co-field is to the overlap kernel: which pair of field types
at which distance raises the property, and which lowers it.  Defined for the linear layer
only; a Gaussian or arc-cosine layer has no such vector.

### `top_channels(model, labels: Sequence[str], n: int=10) -> List[Tuple[str, float, float]]`

The `n` channels that move the prediction most, as (label, weight, spread).

`spread` is the standard deviation of the scaled, normalised channel over the training
structures: a weight only matters where the channel varies, so the ranking is by
`|weight| * spread` and both are reported.

## cmfqsar.pharmacophore

Source: `cmfqsar/pharmacophore.py`

A pharmacophore read out of a fitted co-field.

A pharmacophore is a few typed points with tolerance spheres; a co-field is a continuous
function saying, at every point in space, how much field of one type raises or lowers the
predicted activity.  The two are the same statement at different resolutions, and the fields of
this program carry exactly the classical feature types -- hydrogen-bond donor and acceptor,
hydrophobic, steric, electrostatic -- so the discretisation is a matter of finding the lobes:

* a connected region where the co-field of field *f* exceeds a fraction of its largest value is
  one feature; its centre is the field-weighted centroid of the region, and its tolerance radius
  the radius of the sphere of the same volume, so a broad lobe gives a loose feature and a tight
  one a strict feature;
* the sign says what the feature asks for.  Positive electrostatic co-field means a positive
  partial charge there raises the activity, negative means a negative one does; positive steric
  co-field means bulk is wanted, and **negative steric co-field is an exclusion volume** -- the
  model's own statement that atoms there lower the activity;
* each feature carries the integral of the co-field over its region, which is its contribution to
  the prediction for a unit of field density, and is what the features are ranked by.

With a protein loaded, the pocket supplies exclusion volumes of the harder kind: positions
actually occupied by the receptor.  Those come from `pocket_exclusions` and are marked as
such, since a model fitted on ligands alone cannot know them.

The result is an ordinary pharmacophore hypothesis, derived from a fitted model rather than
guessed, which can be used to align molecules of unrelated scaffolds, to filter a library before
the field model scores it, or simply to read what the model has learnt.  What it is not is a
replacement for the co-field: discretising discards what the continuous description was for.

### `class Feature  (dataclass)`

One pharmacophore feature: where, how big, what it asks for, and how much it is worth.

Attributes:

- `kind: str`
- `centre: np.ndarray`
- `radius: float`
- `field: str`
- `sign: int`
- `peak: float` = `0.0`
- `importance: float` = `0.0`
- `meaning: str` = `''`
- `favourable: bool` = `True`

#### `Feature.describe() -> str`

### `features_from_volume(volume: np.ndarray, grid: Grid, field: str, sign: int, level: float=0.5, min_radius: float=0.6) -> List[Feature]`

The lobes of one co-field, of one sign, as features.

### `extract(model, grid: Optional[Grid]=None, level: float=0.5, max_features: int=12, min_radius: float=0.6, fields: Optional[Sequence[str]]=None, spacing: float=0.6, margin: float=3.0, progress=None, min_z: float=0.0, sigma: Optional[float]=None) -> List[Feature]`

Read a pharmacophore out of the co-fields of a fitted model.

`level` is the fraction of each co-field's largest value that bounds a lobe: 0.5 gives the
half-height regions, a larger value tighter and fewer features.  Features are ranked by the
integral of the co-field over them and the strongest `max_features` are kept, exclusion
volumes counted separately so that they never displace the favourable ones.

With `min_z` above zero the co-field is first masked where it lies less than that many
standard deviations from zero (`cmfqsar.cofields.cofield_with_error`), so that a lobe
the activities' noise could have produced never becomes a feature.  Two is the usual value.
A feature that survives is still a feature of this training set, but it is at least not a
feature of its measurement error.

### `pocket_exclusions(protein, grid: Grid, max_spheres: int=18, min_separation: float=2.5, radius: float=1.6) -> List[Feature]`

Exclusion volumes from the receptor itself: where the protein already is.

Heavy atoms inside the grid box are thinned to one sphere every `min_separation` angstrom,
nearest the centre of the box first, so that a few spheres mark the walls instead of
a copy of the pocket.

### `colour_of(feature: 'Feature')`

### `summary(features: Sequence[Feature]) -> str`

### `write_csv(path: str, features: Sequence[Feature]) -> None`

### `write_pdb(path: str, features: Sequence[Feature]) -> None`

Features as pseudo-atoms, so that any viewer can draw them.

The residue name is the feature type, the occupancy its weight and the B-factor its tolerance
radius; the element is chosen per type (N donor, O acceptor, C hydrophobic, Fe exclusion) only
so that a viewer colours them apart.

### `exportable(features: Sequence[Feature]) -> Tuple[List[Feature], List[Feature]]`

(what a query can say, what it cannot) -- so that the caller can report the remainder.

### `write_pharmit(path: str, features: Sequence[Feature], enable_top: int=4) -> List[Feature]`

Write a Pharmit session (JSON).  Returns the features that could not be expressed.

Pharmit (pharmit.csb.pitt.edu) reads this file as a query over its own screening libraries.
The hydrogen-bond features are written without a direction, because a co-field has none: it
says that a donor *here* raises the activity and nothing about where its hydrogen points.

### `write_pml(path: str, features: Sequence[Feature], name: str='CMF co-field', enable_top: int=4) -> List[Feature]`

Write a LigandScout-style pharmacophore (PML, XML).  Returns what it could not express.

Hydrogen-bond features are written as points rather than as the vectors that format usually
carries, for the same reason: the co-field gives a place, not a direction.

### `write_query(path: str, features: Sequence[Feature], enable_top: int=4, name: str='CMF co-field') -> List[Feature]`

Write the query in the format the file extension names; returns what was left out.

## cmfqsar.pharmtypes

Source: `cmfqsar/pharmtypes.py`

Six pharmacophore features as indicator fields (plan item D2).

An indicator field puts a Gaussian of weight one on every atom of one *kind* and nothing anywhere
else, so its co-field is a map of where that kind of atom helps.  The program has had them keyed
by Sybyl atom type since the 2015 paper (`cif:C.ar`, `cif:N.am`, ...), and they work -- on sets
above about forty molecules.  Below that they do not, and the reason is arithmetic rather than
chemistry: a Sybyl vocabulary has twenty-odd types in a congeneric series, each type is rare, and
a field whose weights are zero on most molecules contributes a kernel column of mostly zeros.

The vocabulary here is six features instead of twenty-odd:

============  ==========================================================================
`donor`       an N-H, O-H or S-H that can donate a hydrogen bond
`acceptor`    an acceptor lone pair
`aromatic`    an aromatic ring, every atom of it
`hydrophobe`  a hydrophobic atom or a lumped hydrophobic group, such as a ring face
`cation`      an ionisable basic group, protonated at physiological pH
`anion`       an ionisable acidic group
============  ==========================================================================

Each is common enough to appear in most molecules of a series, so each field has something to say
about every molecule; and the vocabulary is the one a medicinal chemist already uses, so the
co-field of `pharm:donor` *is* a statement about where a donor is wanted, with no translation
step.  `cmfqsar.pharmacophore` knows the six names, so a pharmacophore extracted from such a
model comes out with the right feature types rather than with a field name.

Where the definitions come from
-------------------------------
RDKit's `BaseFeatures.fdef`, the Gobbi-Poppinger definitions that ship with it, rather than
SMARTS written here: they are the standard set, they are what anyone comparing this with another
program will have, and a definition this program invented would have to be defended.  The eight
families of that file map onto the six above, with `Hydrophobe` and `LumpedHydrophobe` joined
and `ZnBinder` dropped.

An atom belongs to a feature or it does not -- the weight is one or zero, never two, even where
the file gives an atom to both `Hydrophobe` and `LumpedHydrophobe`.  Where a family names a
whole group, every atom of the group gets the weight: an aromatic ring is six atoms of `aromatic`,
a carboxylate is three atoms of `anion`.  That is RDKit's grouping, and for a field it is the
right one, since the character of a carboxylate is not on one of its atoms.

### `definitions_used() -> str`

Which definitions the fields are being built from, for a log line or a model file.

### `feature_atoms(mol) -> Dict[str, np.ndarray]`

{feature: boolean array over the atoms} for one molecule.

An atom that several features of one family name is still weight one: these are indicators,
not counts, and an atom that RDKit lists as both a hydrophobe and part of a lumped
hydrophobic group is one hydrophobe.

### `pharm_weights(mol, feature: str) -> np.ndarray`

The weight column of one pharmacophore indicator field: one on its atoms, zero elsewhere.

### `features_in(mols: Sequence, min_molecules: int=5) -> List[str]`

Which of the six occur in at least `min_molecules` molecules of a set.

The same rule `cmfqsar.fields.cif_types_in` applies to Sybyl types, and the reason the
item exists: a feature carried by two molecules of thirty is a kernel column of zeros.

### `coverage(mols: Sequence) -> Dict[str, Dict[str, float]]`

Per feature: the share of molecules that carry it and the mean atoms per molecule.

This is the measurement the item is an argument about, so it is worth being able to print:
a Sybyl vocabulary has many types each carried by few molecules, and these six are meant to
be carried by most.

## cmfqsar.planealign

Source: `cmfqsar/planealign.py`

Alignment onto a plane: the frame of a molecule on a surface, without the surface.

A molecule adsorbed on an electrode, a corrosion inhibitor on steel, a modifier on a copper
cathode, lives in a frame that no reference molecule defines: the normal of the surface is z,
the top atomic layer is z = 0, and the molecule's height, tilt and face are the pose.  The
surface autocorrelation kernel (`cmfqsar.invariant`) reads such poses without any
in-plane registration, and the Fe(110) study found that the in-plane position over the lattice
carries nothing for the adsorption energy while the height carries everything.  What the surface
kernel cannot give is a co-field, because it has no fixed frame to draw one in.  The overlap
kernel can, if the molecules are in the surface frame; and so far the only way into that frame
has been to relax every molecule on a slab with a machine-learned potential.

This module is the cheap way in.  It poses a molecule on z = 0 by rule, the same rules the UMA
adsorption script uses for its *starting* poses:

* `flat`: the plane of the two longest principal axes is laid parallel to the surface, the
  longest axis along x, and the flattest face turned down (the out-of-plane atoms point away
  from the surface, which is where a relaxation leaves a substituent that cannot bond);
  the lowest heavy atom is put at `height`;
* `upright`: the most basic nitrogen (by a rule: aliphatic amine > pyridine-type > imine >
  aniline > pyrrole-type N-H; amide, nitro and quaternary N never; a thioether S or an ether O
  when there is no nitrogen) points its lone pair at the surface and sits at `anchor_height`,
  with the smallest tilt that leaves every other atom above it.

Three further pieces turn a rule into an alignment:

* `surface_field_align` refines a pose by the field similarity to a reference pose, as
  `cmfqsar.align.field_align` does, but with the degrees of freedom a surface leaves:
  translation in the plane and rotation about the normal (`dof="plane"`), plus tilt and
  height (`dof="plane+tilt"`).  The reference is one molecule whose pose is trusted -- a
  relaxed pose, a crystal pose, or the rule applied to the scaffold.
* `plane_align` can try several starting poses (flat on either face, upright) and keep
  the one most similar in field to the reference (`pose="auto"`), which is the alignment
  criterion of the 2013 paper applied to a choice the rules cannot make.
* `in_plane_rmsd` judges a pose against a known one modulo the motions the surface does
  not fix (rotation about z, translation in the plane), so a molecule placed over a different
  lattice site is not counted as misplaced; `height_rmsd` judges the heights alone.

Everything here is rigid: the conformation is never touched.  The measurement that goes with
the module is `benchmarks/run_fe110.py`, which asks whether rule-posed molecules reproduce
the UMA-relaxed poses, and what the overlap kernel in the rule frame recovers of the adsorption
energy against the surface kernel and against the relaxed frame.

### `class PlanePose  (dataclass)`

Attributes:

- `name: str`
- `label: str`
- `anchor: Optional[int]`
- `similarity: float` = `float('nan')`
- `candidates: List[Tuple[str, float]]` = `field(default_factory=list)`

#### `PlanePose.describe() -> str`

### `class PoseGeometry  (dataclass)`

What a pose on a surface is, in two numbers, and what they were measured on.

A molecule in the surface frame has four free motions and the kernel sees only two of them:
how high it sits and how far it is tilted.  (Where it sits in the plane and how it is turned
about the normal are what the surface kernel is invariant to.)  Those two numbers are the
pose, so the application states them beside the picture.

* `height` is the lowest heavy atom above the surface plane, in angstrom.  This is the
  height the Fe(110) reference poses report and the one the rule poses place.
* `tilt` is the angle, in degrees, between the molecule's own flat plane and the surface:
  0 degrees is a molecule lying flat, 90 degrees one standing on edge.  The plane is the one
  spanned by the two longest principal axes of the heavy atoms, so the normal is the
  shortest axis.
* `thickness` is how far the heavy atoms spread along that shortest axis.  A flat molecule
  has a thickness of a few tenths of an angstrom and its tilt means what it says; a bulky
  one has no plane, `flat` is then false, and the tilt is only the direction of its
  thinnest axis.

Attributes:

- `height: float`
- `centre_height: float`
- `tilt: float`
- `thickness: float`
- `flat: bool`
- `n_heavy: int`

#### `PoseGeometry.describe() -> str`

### `anchor_atom(mol: Chem.Mol) -> Optional[int]`

The atom that points at the surface in the upright pose: the most basic nitrogen, by a
simple rule (aliphatic amine > pyridine-type aromatic N > imine > aniline > pyrrole-type
NH; amide, nitro and quaternary N never), else a thiol/thioether S, else a hydroxyl/ether O.
`None` when the molecule has nothing of the kind.  The rule is the one of the UMA
adsorption script, with one repair: an oxygen on a nitrogen (nitro, N-oxide) is not an
anchor, which the script's version let through.

### `lone_pair_direction(mol: Chem.Mol, coords: np.ndarray, anchor: int) -> np.ndarray`

Where the anchor atom's lone pair points: away from the mean of its bonded neighbours
(away from the centroid when it has none).

### `principal_axes(coords: np.ndarray) -> np.ndarray`

Rows: the principal axes of the point cloud by decreasing extent, a right-handed frame.

### `rotation_to_z(direction: np.ndarray, sign: float=-1.0) -> np.ndarray`

The rotation matrix that takes `direction` to `sign * z`.

### `axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray`

Rodrigues' rotation matrix about a unit axis.

### `heavy_mask(mol: Chem.Mol) -> np.ndarray`

### `orient_flat(coords: np.ndarray, heavy: Optional[np.ndarray]=None, face: str='down') -> np.ndarray`

The molecule rotated about its centroid so that the plane of its two longest principal
axes (over the heavy atoms when `heavy` is given) is the xy plane and the longest axis lies
along x.  `face="down"` turns the flattest face to the surface: the atoms off the plane
end up at positive z.  `face="up"` is the other face; `face="asis"` leaves the sign to
the right-handed frame.

### `orient_upright(mol: Chem.Mol, coords: np.ndarray, anchor: int) -> np.ndarray`

The molecule rotated about its centroid so that the anchor atom's lone pair points at
the surface (-z), tilted by the least that puts the anchor lowest, and turned about z so
that its longest in-plane extent lies along x; the anchor over x = y = 0.

### `place_at_height(coords: np.ndarray, heavy: np.ndarray, height: float, anchor: Optional[int]=None) -> np.ndarray`

Shift along z so that the lowest heavy atom -- or the anchor atom -- is at `height`, and
in the plane so that the heavy-atom centroid is over x = y = 0 (the anchor stays over the
origin when there is one).

### `rule_pose(mol: Chem.Mol, label: str, height: float=2.2, anchor_height: float=2.0) -> Tuple[np.ndarray, Optional[int]]`

The coordinates of `mol` in the pose `label` (`flat`, `flat-up` or `upright`), and
the anchor atom used (`None` for the flat poses).  Raises `ValueError` for an upright
pose when the molecule has no anchor.

### `surface_field_align(mol: Chem.Mol, ref: Chem.Mol, fieldset: FieldSet, dof: str='plane', restarts: int=4, seed: int=0, max_iter: int=300, max_tilt: float=30.0, max_dz: float=1.0, _reference: Optional[_Reference]=None) -> Tuple[float, float]`

Refine the pose of `mol` on the surface by the normalised field similarity to `ref`,
moving it only as a surface allows: in the plane and about the normal (`dof="plane"`),
or also tilting it by up to `max_tilt` degrees and lifting it by up to `max_dz` A
(`dof="plane+tilt"`).  Returns (similarity before, similarity after); `mol` is moved in
place.  The restarts are rotations about z spread over the circle, since the in-plane
orientation is the one thing a rule leaves open.

### `plane_align(mol: Chem.Mol, pose: str='flat', height: float=2.2, anchor_height: float=2.0, ref: Optional[Chem.Mol]=None, fieldset: Optional[FieldSet]=None, refine: Optional[str]=None, restarts: int=4, seed: int=0, name: str='', _reference: Optional[_Reference]=None) -> PlanePose`

Pose `mol` on the surface z = 0, in place.

`pose` is `flat`, `flat-up` (the other face down), `upright` or `auto`.  `auto` builds the candidates -- flat with
either face down and, when the molecule has an anchor atom, upright -- and keeps the one
with the highest field similarity to `ref` (which then needs `fieldset`); without a
reference it falls back to `flat`.  `refine` is `None`, `"plane"` or
`"plane+tilt"`: with a reference, each candidate is refined by
`surface_field_align` with those degrees of freedom before the choice is made.

### `plane_align_dataset(mols: Sequence[Chem.Mol], names: Sequence[str], pose: str='flat', height: float=2.2, anchor_height: float=2.0, ref_index: Optional[int]=None, fieldset: Optional[FieldSet]=None, refine: Optional[str]=None, restarts: int=4, seed: int=0, ref_as_is: bool=True, progress=None) -> List[PlanePose]`

Pose every molecule of a set on the surface, in place.  With `ref_index`, the molecule at
that index is the reference the others are compared with: it is posed by the rule first
unless `ref_as_is` is true and its coordinates are already a trusted pose.

### `in_plane_rmsd(coords: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray]=None) -> float`

RMSD between `coords` and `target` after the best rotation about z and translation in
the plane (Kabsch in two dimensions; z is compared as it stands).  This is the pose error a
surface sees: the lattice site a molecule sits over is not part of it.

### `height_rmsd(coords: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray]=None) -> float`

RMSD of the heights alone.

### `rigid_rmsd(coords: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray]=None) -> float`

RMSD after the best rigid superposition in three dimensions: the floor a rigid pose of
this conformation can reach, whatever the frame.

### `pose_geometry(coords: np.ndarray, symbols: Optional[Sequence[str]]=None, plane_z: float=0.0, flat_tolerance: float=0.8) -> PoseGeometry`

The height and the tilt of a posed molecule, measured in the surface frame.

`coords` are the coordinates of the molecule, `symbols` its element symbols (hydrogens are
left out when they are given, because a height is quoted on the heavy atoms), and `plane_z`
the height of the surface plane, which is 0 in the frame this program uses.  See
`PoseGeometry` for what the numbers are.

## cmfqsar.pocketalign

Source: `cmfqsar/pocketalign.py`

Aligning a molecule inside the pocket (plan item C3).

Two criteria for a pose exist in this program and neither is enough on its own.

`cmfqsar.varalign` places a molecule where the *model* is most confident about it -- where
its fields most resemble the training set's, as the 2019 paper proposes.  It knows nothing about
the receptor, so nothing stops it putting a molecule where no molecule could go, and on a series
whose training poses are themselves only a superposition it can only reproduce that superposition.

`cmfqsar.docking` places a molecule where the *pocket* prefers it -- complementarity of the
fields with the receptor, less a clash penalty.  It knows nothing about the series, so nothing
makes the pose it finds comparable with the poses the model was fitted on, and a shallow site
leaves it under-determined.

The item puts them together in the obvious way: the complementarity is a **prior** over poses,
the confidence is the **likelihood** of the pose given the model, and the pose to take is the one
that maximises the product.  In the units everything here is already in,

    minimise   sigma^2(pose)  -  w * [ complementarity(pose) - clash(pose) ]

over the six rigid-body parameters, with the displacement guard of the confidence aligner kept so
that neither criterion may move a molecule out of the site it started in.  `w` is the only new
number: at zero this is the confidence alignment exactly, and as it grows the pocket dominates.

The real caveats are the ones both halves already carry.  The variance term is not a measure of
the applicability domain once a pose has been chosen to minimise it; the pocket term is not a
docking score, since there is no desolvation, no entropy and no torsional search; and the
internal geometry is never touched, so what is being chosen is a placement and not a
conformation.

### `class PocketAlignResult  (dataclass)`

What the joint objective did to one molecule.

Attributes:

- `name: str`
- `coords: np.ndarray`
- `variance_before: float`
- `variance_after: float`
- `pocket_before: float`
- `pocket_after: float`
- `clash_before: float`
- `clash_after: float`
- `objective_before: float`
- `objective_after: float`
- `rmsd: float`
- `n_eval: int` = `0`
- `limited: bool` = `False`

#### `PocketAlignResult.describe() -> str`

### `class PocketAligner(model, pocket: MoleculeFields, pocket_weight: float=POCKET_WEIGHT, max_rmsd: Optional[float]=2.0, penalty: float=50.0, fieldset: Optional[FieldSet]=None, **scorer_kw)`

Rigid-body alignment by the model's confidence and the pocket's complementarity together.

The variance half comes from `cmfqsar.varalign.VarianceAligner` and the pocket half
from `cmfqsar.docking.PocketScorer`, so this class adds no new physics: it is the
combination, the guard and the search.

#### `PocketAligner.align(mol: MoleculeFields, restarts: int=2, seed: int=0, max_iter: int=300, apply: bool=True, any_orientation: bool=False) -> PocketAlignResult`

Find the pose that the model and the pocket agree on.

`any_orientation` has every restart draw a uniformly random orientation instead of turning
the arriving pose by at most about 63 degrees; see `VarianceAligner.align`.

### `pocket_align(mols: Sequence[MoleculeFields], model, pocket: MoleculeFields, pocket_weight: float=POCKET_WEIGHT, max_rmsd: Optional[float]=2.0, restarts: int=2, seed: int=0, any_orientation: bool=False, fieldset: Optional[FieldSet]=None, progress: Optional[Callable[[int, int], None]]=None, cancel: Optional[Callable[[], bool]]=None, **scorer_kw) -> List[PocketAlignResult]`

Align every molecule in place by the model's confidence and the pocket together.

### `summarise(results: Sequence[PocketAlignResult]) -> dict`

What a run of the joint alignment did, in a single line of numbers.

## cmfqsar.poses

Source: `cmfqsar/poses.py`

A protocol that can judge an alignment, because q2 cannot.

Every measurement in this program says the alignment decides more than the kernel, the learner or
the fields: an automatic frame costs about 0.30 in q2 where a curated one succeeds, and on series
whose frame is wrong the fields fall below atom-type counts.  Yet the program has no way to tell
whether an alignment method is any good, and the obvious measure is actively misleading.

**Why q2 cannot judge a pose.**  Realigning changes the geometry that q2 is computed on.  Moving a
molecule until the model is confident about it raises the model's agreement with itself, which is
what q2 measures; the README records q2 and external R2pred moving in opposite directions under
exactly this operation.  A criterion that rewards a method for making the data look like the model
cannot be used to choose between methods.

**What can.**  A crystal pose is an answer obtained without the model.  The FEP benchmark sets
under `examples/openff`, `examples/schindler` and `examples/ross` carry one for every ligand,
co-crystallised in the same frame, so a posing method can be asked the only two questions that
matter:

1. **Did it find the pose?**  The RMSD of the recovered pose to the crystal pose, over heavy atoms,
   symmetry-aware, and **without superposition** -- `rdMolAlign.CalcRMS`, never `GetBestRMS`.
   The frame is the thing being judged, so re-superposing before measuring would report 0.0 for a
   molecule placed in the wrong site: on a molecule displaced by 1 A the two functions return 1.0
   and 0.0 respectively.
2. **Did the prediction improve?**  The external R2pred on held-out molecules posed **without their
   activity**, which is the operational case: a new compound arrives with neither a pose nor a
   measurement.  Its pose comes from the method under test, and its activity is then predicted by a
   model trained on crystal-posed molecules.

**Both, always, and the control first.**  Every run begins by destroying the poses
(`cmfqsar.align.unalign`), so each method is asked to recover a frame rather than to leave
one alone, and every number is reported between two fixed references: the **ceiling**, the crystal
poses themselves, and the **floor**, the scrambled poses left as they are.  A method is worth
something only to the extent that it moves from the floor towards the ceiling, and
`PoseScore.recovery` is that fraction.  Without the control a method that silently does
nothing scores exactly what the starting poses scored, which on a curated set looks like success.

The protocol is agnostic about the method: `POSERS` holds the ones in the program (MCS
superposition on a reference, field-similarity refinement, and the posterior-variance search of
the 2019 paper), and `judge` takes any callable that re-poses molecules in place.

### `class PoseScore  (dataclass)`

What one posing method achieved on one set: where the poses went, and what they predict.

Attributes:

- `method: str` = `''`
- `n: int` = `0`
- `rmsd: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`
- `r2_pred: float` = `float('nan')`
- `rmse_pred: float` = `float('nan')`
- `predicted: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`
- `observed: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`
- `seconds: float` = `0.0`
- `floor_rmsd: float` = `float('nan')`
- `conformer_limit: float` = `0.0`

#### `PoseScore.median_rmsd` -> float  (property)

#### `PoseScore.within(cutoff: float) -> float`

Fraction of molecules whose pose is within `cutoff` angstrom of the crystal pose.

#### `PoseScore.recovery` -> float  (property)

How much of the scrambling was undone: 1 is the crystal pose, 0 is no movement.

Negative means the method left the molecules further from the crystal pose than the
scrambled poses it was given, which happens and should be reported.

#### `PoseScore.row() -> dict`

### `pose_rmsd(mol: Chem.Mol, ref: Chem.Mol, heavy_only: bool=True) -> float`

Symmetry-aware RMSD between two poses of one molecule, **without superposition**.

This is the deviation in the frame the two structures are already in, which is what judging a
pose requires.  `GetBestRMS` would superpose them first and report how similar the
conformations are, a different and much easier question: a molecule translated by 1 A scores
1.0 here and 0.0 there.

Hydrogens are excluded by default, since a crystal structure rarely places them.

### `pose_keep(mols: Sequence[Chem.Mol], ctx: dict) -> None`

Leave the poses alone: the floor, and the score any method that does nothing will get.

### `pose_mcs(mols: Sequence[Chem.Mol], ctx: dict) -> None`

Superimpose each molecule on a reference training molecule through their common substructure.

### `pose_field(mols: Sequence[Chem.Mol], ctx: dict) -> None`

MCS first, then refine on the CMF field similarity with the reference.

### `pose_variance(mols: Sequence[Chem.Mol], ctx: dict) -> None`

Minimise the model's posterior variance: the proposal of the 2019 paper.

The activity of the molecule being posed is not used -- only the model's confidence about it,
which is what makes this legitimate for a molecule that has not been measured.

### `pose_mcs_then_variance(mols: Sequence[Chem.Mol], ctx: dict) -> None`

The two in series: a substructure frame, then the confidence search from there.

### `pose_pocket(mols: Sequence[Chem.Mol], ctx: dict) -> None`

Place every molecule where the *pocket* prefers it: complementarity less clash.

The receptor alone, with no model: this is the prior half of item C3's objective, and it is
here on its own so that the joint pose can be read against it.

### `pose_pocket_variance(mols: Sequence[Chem.Mol], ctx: dict) -> None`

The pocket as the prior and the model's confidence as the likelihood, together (C3).

### `pose_mcs_then_pocket_variance(mols: Sequence[Chem.Mol], ctx: dict) -> None`

A substructure frame first, then the pocket and the confidence together.

### `pose_flexible(mols: Sequence[Chem.Mol], ctx: dict) -> None`

Confidence over rigid motion *and* the rotatable bonds (`cmfqsar.flexalign`).

### `pose_mcs_then_flexible(mols: Sequence[Chem.Mol], ctx: dict) -> None`

A substructure frame, then the flexible confidence search from there.

### `pose_fieldgraph(mols: Sequence[Chem.Mol], ctx: dict) -> None`

Match each molecule's own field extrema to the reference's (`cmfqsar.fieldgraph`).

The only poser here that needs neither a common substructure nor a fitted model: what is
superimposed is what the two molecules present, so a molecule sharing no scaffold with the
reference can still be placed, and a molecule sharing nothing at all is left where it was.

### `pose_fieldgraph_then_field(mols: Sequence[Chem.Mol], ctx: dict) -> None`

Field points for the frame, then the kernel refinement that needs one to start from.

### `pose_fieldgraph_then_variance(mols: Sequence[Chem.Mol], ctx: dict) -> None`

Field points for the frame, then the model's confidence from there.

### `regenerate_conformation(mol: Chem.Mol, seed: int=0) -> Chem.Mol`

The same molecule with a conformation generated from scratch, crystal geometry discarded.

Without this the protocol gives a held-out molecule its crystal *conformation* and only
scrambles its pose, which is the one case a flexible method cannot help with: every torsion
is already correct, so turning one can only do harm.  A molecule that arrives without a
crystal structure arrives without its conformation too, and that is what a flexible
alignment is for.

### `conformer_limit(mol: Chem.Mol, ref: Chem.Mol) -> float`

The RMSD the best possible rigid placement of this conformation would still leave.

No rigid method can beat it, so it is the ceiling for a molecule whose conformation is wrong,
and the number against which a flexible method has to be read.

### `judge(mols: Sequence[Chem.Mol], names: Sequence[str], y: Sequence[float], train: Sequence[int], test: Sequence[int], fieldset: FieldSet, methods: Sequence[str]=('scrambled', 'mcs', 'field', 'variance'), lam: float=0.1, seed: int=0, max_translation: float=2.0, restarts: int=20, max_rotation_deg: float=180.0, max_rmsd: Optional[float]=None, posers: Optional[Dict[str, Callable]]=None, reconform: bool=False, conformers: int=0, pocket: Optional[MoleculeFields]=None, pocket_weight: float=1.0, progress: Optional[Callable[[str], None]]=None) -> Dict[str, PoseScore]`

Judge posing methods against the crystal poses of a set.

`mols` arrive in their crystal poses.  A model is fitted on the training molecules as they
are; the test molecules then have their poses destroyed, and each method is asked to recover
them from that starting point, without ever seeing a test activity.  Returns a score per
method, plus `"crystal"` -- the ceiling, the test molecules in the poses they came in.

The destruction gives every test molecule a uniformly random orientation, so a method that
searches orientations must be allowed to draw any of them: `max_rotation_deg` is 180 here,
where the docking's own default is the 30 degrees a series already in the site needs.
`restarts` is 20 for the same reason.  Both halves are needed: a random orientation is only
useful if enough of them are drawn, and more draws of a bounded turn do not help at all.

### `global_superpose(probe: Sequence[Chem.Mol], ref: Sequence[Chem.Mol]) -> float`

Apply the single rigid motion that best maps the whole `probe` set onto `ref`.

A training frame is only defined up to a global rigid motion: the CMF kernel depends on where
the molecules are *relative to each other*, so rotating the entire set changes no kernel entry
and no prediction.  Judging a recovered frame therefore means removing that freedom first --
one transform for all molecules, never one per molecule, which would measure conformations
instead of the frame.  Returns the RMSD of the concatenated coordinates after the fit.

### `frame_rmsds(probe: Sequence[Chem.Mol], ref: Sequence[Chem.Mol]) -> np.ndarray`

Per-molecule RMSD to the reference frame, after one global superposition of the set.

### `judge_training_frame(mols: Sequence[Chem.Mol], names: Sequence[str], y: Sequence[float], train: Sequence[int], test: Sequence[int], fieldset: FieldSet, methods: Sequence[str]=('scrambled', 'mcs', 'coalign', 'mcs+coalign'), lam: float=0.1, seed: int=0, max_translation: float=2.0, coalign_kwargs: Optional[dict]=None, progress: Optional[Callable[[str], None]]=None) -> Dict[str, PoseScore]`

Judge a method that re-aligns the **training set** itself (item C4).

The training poses are destroyed and recovered; the held-out molecules keep their crystal
poses throughout, so what is measured is the frame and nothing else.  Two numbers per method:
the per-molecule RMSD of the recovered frame to the crystal one, after the global
superposition that the kernel cannot see, and the external R2pred of the held-out molecules
posed onto the recovered frame by substructure alignment -- the operational route, since in a
real application there is no crystal frame to align them to.

Two questions, and they are not the same.  `coalign` starts from the scrambled frame and
asks whether the iteration can *build* one; it cannot, and the reason is structural -- a model
fitted on scrambled poses is meaningless, so the confidence it offers is flat, which is what
C1 found for the rigid search and what the field-graph module was written for.
`mcs+coalign` starts from the substructure frame and asks whether the iteration can
*improve* a usable one, which is the question C4 is actually about.

### `table(scores: Dict[str, PoseScore], title: str='') -> str`

The protocol's answer as a table: where the poses went and what they predict.

## cmfqsar.probes

Source: `cmfqsar/probes.py`

GRID-like probe fields, written as sums of Gaussians so that CMF can keep its algebra.

Goodford's GRID (J. Med. Chem. 28 (1985) 849) describes a molecule by the energy a small probe --
a water, a methyl, a carbonyl oxygen -- would feel at every point around it.  That is a field in
exactly the sense this program means, and it says something the program's own fields do not: not
"there is hydrophobic character here" but "a probe placed here would be bound".  The obstacle is
arithmetic.  Every kernel in CMF is an overlap integral evaluated in closed form over atom pairs,
which works because a field is a sum of atom-centred Gaussians; a Lennard-Jones term is not one,
has no closed-form overlap, and diverges at the atom, so the integral of its square does not even
exist.

The solution is to fit rather than to integrate numerically, and the fitting has to respect
what atom-centred Gaussians can and cannot do.  Three things were learned by measuring, and all
three shape what is implemented here:

* **The repulsive wall cannot be carried.**  Fitting a curve that is +30 kcal/mol at 2.5 A and
  -0.13 at 3.5 A with a few Gaussians produces coefficients of +/- 1e7 that cancel and a
  worst-case error larger than the well it is meant to represent.  The wall is a truncation
  artefact in every grid implementation anyway, and this program already carries it: the steric
  field *is* a Gaussian van der Waals envelope.  So a probe field carries the **attractive** part
  and the repulsion is left where it is already represented.
* **A well at a radius is a shell, and a shell is not a sum of centred Gaussians.**  Atom-centred
  Gaussians are maximal at the atom; representing a peak at 3.5 A needs differences of broad
  Gaussians, which is that same ill-conditioning again.  Each term is therefore *capped* at its
  own optimum -- the Lennard-Jones part below the contact distance, the hydrogen-bond part below
  its preferred separation -- which makes every term monotone in r, and those a sum of Gaussians
  represents naturally.
* **The r^-6 tail is not Gaussian either**, and it is also negligible: beyond 6 A the dispersion
  is below any threshold that matters, so the energy is tapered to zero between 5 and 7 A, which
  is what GRID's own cut-off does.

What comes out is an attraction envelope for a given probe, reproduced to within roughly 10 to 30
per cent in the volume-weighted norm the kernel actually sees -- 15 on the median
combination of probe and atom, 23 on a realistic worst case, and up to 30 where a dispersion
well and an electrostatic repulsion nearly cancel and the whole curve is below 0.1 kcal/mol -- `fit_report` prints the
number for every element and a test holds it to a stated bound.  It is a probe-specific weighting
of dispersion, electrostatics and hydrogen bonding with the right distance dependence; it is not
a reproduction of GRID's numbers, and the directional part of GRID's hydrogen-bond term is not
represented at all.  The place for that, if it is ever wanted, is Gaussians on lone-pair
extension points rather than on the heteroatom, which this framework can carry, since they are
merely more centres with their own weights.

Ion probes, and what they are and are not
-----------------------------------------
Plan item EC7 asks for a cation probe: the field of an ionophore or an electrolyte solvent read as
the energy a lithium, sodium, potassium or magnesium ion would feel in it.  Four such probes are
defined below.  What distinguishes them deserves to be said plainly, because it is less than the
four names suggest.

An ion carries no hydrogen bond and its dispersion well is negligible beside its charge, so for a
singly charged ion `attraction` reduces to the electrostatic term, which is the molecule's
electrostatic potential under this module's distance-dependent screening.  The three +1 probes
therefore differ from one another in **one** quantity, the distance at which that potential is
sampled: the contact distance of the probe with each atom, which for the lithium probe is the
coordination distance measured on the relaxed complexes of `examples/ether/`.  That is a real
difference -- a potassium ion cannot reach the inside of a small cavity that a lithium ion fits --
and it is the whole of the difference.  The magnesium probe adds its charge of +2, which
multiplies the field by two; every kernel in this program is normalised by the self-overlap, so a
constant factor on a field is invisible to it, and the magnesium probe is in consequence the
lithium probe at a slightly shorter distance.  A model that needs more than this needs
polarisation, which a fixed probe charge cannot carry.

### `class Probe  (dataclass)`

A small molecule or group whose interaction energy with the ligand becomes a field.

Attributes:

- `name: str`
- `label: str`
- `radius: float`
- `well: float`
- `charge: float` = `0.0`
- `hbond: float` = `0.0`

### `class GaussianFit  (dataclass)`

The fitted expansion of one (probe, atom kind) attraction curve.

Attributes:

- `coeffs: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`
- `widths: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`
- `rel_error: float` = `float('nan')`
- `depth: float` = `float('nan')`

#### `GaussianFit.describe() -> str`

### `probe_names() -> List[str]`

### `probe(name: str) -> Probe`

### `attraction(r: np.ndarray, p: Probe, symbol: str, charge: float=0.0, hbond: float=0.0) -> np.ndarray`

What `p` would gain by sitting at distance `r` from one atom, as a field can carry it.

Every term is capped at its own optimum so that the result falls monotonically with distance,
and the whole is tapered to zero by 7 A.  Positive values -- the repulsive wall -- are not
represented; see the module docstring for why, and where they are already represented.

### `basis_widths(n_terms: int=4) -> np.ndarray`

The widths, shared by every atom and every probe.

Sharing them is what keeps the kernel affordable: the overlap of two probe fields is a double
sum over atom pairs *and* over pairs of terms, and with common widths each pair of terms is
one pass over the distance matrix under the usual Gaussian-product rule, exactly like an
ordinary field.  Fitting the widths per atom as well is the source of error the module docstring
describes.

### `fit(p: Probe, symbol: str, charge: float=0.0, hbond: float=0.0, n_terms: int=4) -> GaussianFit`

The expansion of one atom's contribution to a probe field, memoised.

`charge` and `hbond` are rounded into the cache key: a fit per distinct partial charge would
be a fit per atom, which is slow and useless, since the electrostatic term is linear in the
charge and the rounding moves the energy far less than the capping does.

### `coefficients(symbols: Sequence[str], charges: Sequence[float], hbond: Sequence[float], p: Probe, n_terms: int=4) -> np.ndarray`

(n_atoms, n_terms) expansion coefficients of one molecule's probe field.

### `fit_report(name: str='water', symbols: Sequence[str]=('C', 'N', 'O', 'S', 'Cl', 'H'), n_terms: int=4, charge: float=0.0, hbond: float=0.0) -> str`

How faithfully the expansion reproduces the attraction, element by element.

## cmfqsar.project

Source: `cmfqsar/project.py`

Projects (plan item G2): a session that can be saved and reopened, the poses before every
alignment step so that a step can be undone, and two models compared on one set of molecules.

A project file holds the data set with its current coordinates and properties, the pose history
(one snapshot of every conformer of every molecule, taken before each step that moved them), the
settings of the application's panels, the fitted model if there is one with the indices it was
fitted and tested on, the hyper-parameter search table, the validation report and the log.  It is
a joblib dump of a plain dictionary, like a model file, and the molecules travel as RDKit binary
blobs with their properties, so nothing is lost between sessions.

Nothing here needs Qt.  The application calls `snapshot_poses` before a step that moves
molecules, `PoseHistory` keeps the snapshots in order, `save_project` and
`load_project` move a `Project` to and from disk, and `compare_models` scores
two models on the same molecules.  The widgets' values are captured and restored by the
application itself (`cmfqsar.gui.panels.capture_settings`); here they are a dictionary.

### `class PoseHistory(steps: Optional[List[dict]]=None, limit: int=50)`

The snapshots taken before each step that moved the molecules, oldest first.

`remember` is called before a step, `undo` puts the molecules back to the latest
snapshot and drops it.  The history is bounded so that a long session of trial alignments on
a large set does not grow without limit; the oldest snapshots go first.

#### `PoseHistory.remember(mols: Sequence[Chem.Mol], label: str) -> dict`

#### `PoseHistory.last_label` -> Optional[str]  (property)

#### `PoseHistory.undo(mols: Sequence[Chem.Mol]) -> Tuple[str, np.ndarray]`

Restore the latest snapshot; returns its label and the displacement it undid.

#### `PoseHistory.labels() -> List[str]`

### `class Project  (dataclass)`

Everything a session holds that is worth reopening.

Attributes:

- `dataset: Dataset`
- `settings: Dict[str, object]` = `field(default_factory=dict)`
- `history: List[dict]` = `field(default_factory=list)`
- `model_state: Optional[dict]` = `None`
- `train_indices: List[int]` = `field(default_factory=list)`
- `test_indices: List[int]` = `field(default_factory=list)`
- `decoy_indices: List[int]` = `field(default_factory=list)`
- `search_rows: List[dict]` = `field(default_factory=list)`
- `validation: Optional[dict]` = `None`
- `log: str` = `''`
- `protein_path: str` = `''`
- `notes: str` = `''`
- `protein: Optional[object]` = `None`
- `target: Optional[np.ndarray]` = `None`
- `saved: str` = `''`
- `program_version: str` = `''`

#### `Project.model` -> Optional[CMFModel]  (property)

### `class ComparisonRow  (dataclass)`

Attributes:

- `index: int`
- `name: str`
- `subset: str`
- `y: float`
- `pred_a: float`
- `pred_b: float`

#### `ComparisonRow.difference` -> float  (property)

### `class Comparison  (dataclass)`

Two models' predictions for one set of molecules, and what they agree on.

Attributes:

- `label_a: str`
- `label_b: str`
- `rows: List[ComparisonRow]`
- `summary: Dict[str, dict]`
- `agreement: dict`

#### `Comparison.lines() -> List[str]`

The comparison as text, for the log.

#### `Comparison.to_csv(path: str) -> None`

### `snapshot_poses(mols: Sequence[Chem.Mol], label: str) -> dict`

Every conformer of every molecule, as it is now, under a label that says what is about
to move them ("Align to reference", "Unalign", ...).

### `restore_poses(mols: Sequence[Chem.Mol], snapshot: dict) -> None`

Put the molecules back where the snapshot has them.  The conformers are replaced, so a
snapshot taken before conformers were generated removes them again.

### `pose_displacement(mols: Sequence[Chem.Mol], snapshot: dict) -> np.ndarray`

How far each molecule's first conformer now stands from the snapshot, as the root mean
square displacement of its atoms in angstrom (nan where the shapes do not match).

### `dataset_state(ds: Dataset) -> dict`

### `dataset_from_state(st: dict) -> Dataset`

### `project_state(project: Project) -> dict`

### `project_from_state(st: dict) -> Project`

### `save_project(path: str, project: Project) -> None`

Write the project; the file is a compressed joblib dump of `project_state`.

### `load_project(path: str) -> Project`

Read a project file; refuses one written by a newer version of the program.

### `is_project_file(path: str) -> bool`

Whether a file on disk is a project rather than a model, judged by its content.

### `predict_dataset(model: CMFModel, ds: Dataset, progress=None) -> np.ndarray`

One prediction per molecule of the data set, with the fields the model was fitted with.

### `compare_models(model_a: CMFModel, model_b: CMFModel, ds: Dataset, labels: Tuple[str, str]=('A', 'B'), progress=None, pred_a: Optional[np.ndarray]=None, pred_b: Optional[np.ndarray]=None) -> Comparison`

Score two models on the same molecules.

Both models see the molecules as they are now, in one frame: a model fitted on another
alignment of the same series is compared on *this* alignment, which is the comparison the
panel is for (did the realignment help?) and also its caveat.  Predictions already in hand
can be passed in to save the fields.

## cmfqsar.protein

Source: `cmfqsar/protein.py`

Protein structures for display behind the molecules, their fields and the co-fields.

The free-energy benchmark collections give their ligands in the coordinate frame of a reference
crystal structure, and every record carries the identifier of that structure, so nothing has to be
superimposed: fetching the entry puts the protein exactly around the training set, and a co-field
computed from those molecules already lies in the same frame.

This module reads a structure (PDB or mmCIF), selects the residues around a set of points (the
binding site), and derives bonds from interatomic distances, which is all the viewer needs.  It is
deliberately independent of RDKit: crystal structures carry altLoc flags, missing elements and
non-standard residues that a chemistry toolkit would refuse to sanitise, and none of that matters
for drawing.

`contact_distance` is the frame check.  Ligands posed in the site sit a few ångström from the
protein; a distance of tens of ångström means the molecules were realigned after the structure was
loaded, or that the structure does not belong to this series.

### `class Protein  (dataclass)`

Atoms of a structure, with everything needed to draw and to select a site.

Attributes:

- `name: str` = `''`
- `coords: np.ndarray` = `field(default_factory=lambda: np.zeros((0, 3)))`
- `elements: List[str]` = `field(default_factory=list)`
- `atom_names: List[str]` = `field(default_factory=list)`
- `resnames: List[str]` = `field(default_factory=list)`
- `resids: np.ndarray` = `field(default_factory=lambda: np.zeros(0, dtype=int))`
- `icodes: List[str]` = `field(default_factory=list)`
- `chains: List[str]` = `field(default_factory=list)`
- `hetatm: np.ndarray` = `field(default_factory=lambda: np.zeros(0, bool))`
- `bonds: np.ndarray` = `field(default_factory=lambda: np.zeros((0, 2), dtype=int))`
- `path: str` = `''`

#### `Protein.n_atoms` -> int  (property)

#### `Protein.n_residues` -> int  (property)

#### `Protein.residue_keys() -> List[Tuple[str, int, str]]`

#### `Protein.is_water() -> np.ndarray`

#### `Protein.residue_label(i: int) -> str`

#### `Protein.subset(idx: Sequence[int], with_bonds: bool=True) -> 'Protein'`

### `read_pdb(path: str, keep_waters: bool=False, keep_hydrogens: bool=True, first_model_only: bool=True, with_bonds: bool=True) -> Protein`

Read a PDB file.  Alternate locations other than the first are dropped.

### `read_mmcif(path: str, keep_waters: bool=False, keep_hydrogens: bool=True, first_model_only: bool=True, with_bonds: bool=True) -> Protein`

Read the atom_site loop of an mmCIF file, which is what large entries are served as.

### `monomer_units(mol, max_ring: int=MAX_MONOMER_RING) -> List[int]`

Split a molecule into its monomer units, one number per atom (plan item SC3).

A host that comes as a molecule file has no residues, and the profiles that make a
complementarity score readable are per residue.  A macrocyclic host does have repeating
units -- a glucose of a cyclodextrin, a glycoluril of a cucurbituril, a phenol of a
calixarene -- and they are its small ring systems: rings that share an atom belong to the
same unit, and a ring larger than `max_ring` is the macrocycle that joins the units rather
than a unit itself, so it is left out of the grouping.

Every atom that is in no ring is given to the unit it is bonded to, following the bonds
outwards, so a hydroxyl goes with its own sugar.  An atom bridging two units -- the
glycosidic oxygen -- goes to the one reached first, which makes the split deterministic.  A
molecule with no rings at all comes back as one unit.

### `read_molecule_host(path: str, add_hydrogens: bool=True, resname: str=MONOMER_RESNAME, max_ring: int=MAX_MONOMER_RING, name: str='', keep_hydrogens: bool=True, keep_waters: bool=False, with_bonds: bool=True) -> Protein`

A host given as a molecule file, with its monomer units as the residues (plan item SC3).

`complement` was written for proteins, and a cyclodextrin, cucurbituril or calixarene comes
as an SDF or a MOL file with no residues at all.  This reads one and names its repeating
units, so that everything the pocket machinery does per residue -- the interaction profile,
the contacts, the residue table -- is done per unit instead.

`keep_hydrogens`, `keep_waters` and `with_bonds` are the arguments the PDB reader takes, so
that `read_structure` can pass whatever it was given: hydrogens are dropped when asked,
a molecule file has no waters to drop, and the bonds come from the file rather than from the
distances.

### `host_from_molecule(mol, add_hydrogens: bool=True, resname: str=MONOMER_RESNAME, max_ring: int=MAX_MONOMER_RING, name: str='', path: str='') -> Protein`

The same from a molecule already in hand, which is what a script or a test has.

### `cavity_frame(protein: Protein, max_ring: int=MAX_MONOMER_RING) -> Tuple[np.ndarray, np.ndarray]`

(centre, axis) of a macrocyclic host: where a guest goes and which way the cavity points.

The centre is the centroid of the monomer units' own centroids, which for a ring of units is
the middle of the ring whether or not the units are the same size; the axis is the direction
in which the units vary least, that is the smallest principal axis of those centroids.

### `read_structure(path: str, **kw) -> Protein`

Read a structure, choosing the parser by extension.

### `write_pdb(protein: Protein, path: str, remarks: Sequence[str]=()) -> None`

Write a structure as PDB, with free-text REMARK lines for provenance.

Used to store a trimmed binding site next to a data set, so that the pocket stays with the
molecules and needs no network access.

### `find_pocket(dataset_path: str, code: Optional[str]=None) -> Optional[str]`

Locate a stored binding site that belongs to a data set.

Searched in order: a structure beside the file, named after it; a `pockets` folder beside
it; and a `pockets` folder further up the tree, which is where
`examples/make_pocket_examples.py` writes them, by file name and then by PDB identifier.

### `cache_dir() -> str`

### `fetch_pdb(code: str, dest_dir: Optional[str]=None, timeout: float=30.0, overwrite: bool=False) -> str`

Download an entry from RCSB and return the local path; cached between runs.

The PDB format is tried first and mmCIF second, because very large entries are not
distributed in the PDB format at all.

### `covalent_bonds(coords: np.ndarray, elements: Sequence[str], tolerance: float=0.45, max_bond: float=2.2) -> np.ndarray`

Bonds from interatomic distances: `|r_ij| < R_i + R_j + tolerance`.

### `residue_anchors(protein: Protein) -> Tuple[np.ndarray, List[str]]`

One labelling point and text per residue: the alpha carbon, or the centroid without one.

Residues are returned in the order they appear, so a label list reads along the chain.

### `contact_distance(protein: Protein, points: np.ndarray, polymer_only: bool=True) -> float`

Smallest distance between any of `points` and the structure.

This is the frame check: ligands posed in the site give a few ångström, while a value of
tens of ångström means the molecules and the structure are not in the same frame.  By
default only polymer atoms count, because a co-crystallised ligand often *is* one of the
molecules of the data set and would give a distance of zero.

### `binding_site(protein: Protein, points: np.ndarray, cutoff: float=6.0, whole_residues: bool=True, keep_waters: bool=False, drop_ligand: bool=True, ligand_overlap: float=2.0) -> Protein`

The part of the structure around `points`, as whole residues by default.

`drop_ligand` removes the co-crystallised ligand, recognised as a non-water HETATM residue of
more than one atom that *overlaps* the points, meaning it comes within `ligand_overlap` of
them.  That is what the crystallographic ligand does when the data set contains it or a close
analogue.  Ions and cofactors that line the site without occupying the same space are kept.

## cmfqsar.protonation

Source: `cmfqsar/protonation.py`

Protonation ensembles: one molecule as the weighted mean of its ionisation states (item D4).

A series measured in acid is usually modelled in one protonation state, chosen by hand or by a
rule.  That choice is a guess wherever a pKa falls near the pH of the experiment: an azole with a
pKa of 1 in 1 M HCl is about half protonated, and neither the neutral nor the cation is the
molecule that was measured.

The mean-map kernel removes the need to choose.  A CMF field is a sum of atom-centred Gaussians
and the kernel is bilinear in the fields, so the mean field of an ensemble is itself a field whose
atoms are the atoms of every member with their weights multiplied by that member's weight
(`cmfqsar.fields.combine_members`).  The kernel of two ensembles is then the weighted mean
of the kernels of their member pairs, which is positive definite, and everything downstream --
co-fields, atom contributions, the decomposition rules -- keeps working because the model is still
linear in this mean field.  Nothing in that argument requires the members to be conformations of
one molecule, which is what this module uses: here they are different molecules.

What the weights are
--------------------
Each ionisable site is treated as independent and given the Henderson-Hasselbalch fraction at its
pKa:

    protonated fraction   = 1 / (1 + 10 ** (pH - pKa))
    deprotonated fraction = 1 / (1 + 10 ** (pKa - pH))

and a state's weight is the product over the sites.  Independent sites are an approximation: a
second deprotonation next to a first one is harder than its own pKa says, and this ignores that.
The pKa values and the patterns that find the sites are Dimorphite-DL's, which are regression
values with a published standard deviation, not measurements of these molecules.  Both limits are
real and are reported with any measurement made with this module.

Geometry
--------
The states are built on the geometry of the molecule as it is given, which matters because a CMF
series is aligned: the heavy atoms keep their coordinates, and only hydrogens are added or
removed, with their positions from RDKit.  So an aligned series stays aligned and the ensemble
costs no further superposition.

Usage
-----
    from cmfqsar.protonation import protonation_fields, protonation_states

    ens = protonation_states(mol, ph=0.0)          # states and weights
    mf = protonation_fields(mol, fieldset, ph=0.0)  # one field for the ensemble

### `class ProtonationUnavailable  — bases: RuntimeError`

Dimorphite-DL is not installed, so the states cannot be enumerated.

### `class Site  (dataclass)`

One ionisable site: where it is, what it is, and the pKa used for it.

Attributes:

- `name: str`
- `atom: int`
- `pka: float`
- `stdev: float`

#### `Site.fraction(ph: float) -> Tuple[float, float]`

(deprotonated, protonated) fraction at this pH, from the Henderson-Hasselbalch law.

### `class State  (dataclass)`

One protonation state of the molecule: a molecule of its own, with a weight.

Attributes:

- `mol: Chem.Mol`
- `weight: float`
- `label: str`
- `charge: int`
- `atom_map: np.ndarray` = `_field(default_factory=lambda: np.zeros(0, int))`
- `heavy_trace: np.ndarray` = `_field(default_factory=lambda: np.zeros(0, int))`

#### `State.smiles` -> str  (property)

### `class Ensemble  (dataclass)`

The states of one molecule at one pH, heaviest first.

Attributes:

- `states: List[State]`
- `ph: float`
- `sites: List[Site]`
- `dropped: float` = `0.0`
- `note: str` = `''`

#### `Ensemble.weights` -> np.ndarray  (property)

#### `Ensemble.is_mixture` -> bool  (property)

More than one state carries real weight.

#### `Ensemble.main` -> State  (property)

#### `Ensemble.mean_charge` -> float  (property)

#### `Ensemble.describe() -> str`

### `available() -> bool`

Whether the states can be enumerated at all.

### `protonation_states(mol: Chem.Mol, ph: float=0.0, min_population: float=MIN_POPULATION, max_states: int=MAX_STATES, max_sites: int=MAX_SITES) -> Ensemble`

Enumerate the protonation states of one molecule and weigh them by their pKa values.

The molecule is used as it is given: its heavy-atom geometry is kept, so an aligned series
stays aligned.  Returns an `Ensemble` with the states heaviest first; a molecule with
no ionisable site, or one the enumeration cannot trace back to its own atoms, comes back as a
single state that is the molecule itself, so a caller never has to special-case it.

### `protonation_fields(mol: Chem.Mol, fieldset: FieldSet, ph: float=0.0, name: str='', min_population: float=MIN_POPULATION, max_states: int=MAX_STATES, report: Optional[Dict]=None) -> MoleculeFields`

The field of one molecule as the pKa-weighted mean over its protonation states.

With one state above the threshold this is exactly `build_fields` on that state, so a
series where every molecule is unambiguous gives the same numbers as the single-state model
with that state chosen.  `report`, if given, is filled with what the ensemble was.

### `ensemble_table(mols: Sequence[Chem.Mol], names: Sequence[str], ph: float=0.0, min_population: float=MIN_POPULATION, max_states: int=MAX_STATES) -> List[Ensemble]`

The ensembles of a whole series, in order.

### `mixture_share(ensembles: Sequence[Ensemble]) -> float`

What share of a series is a real mixture at this pH, which is what the ensemble can change.

## cmfqsar.prototypes

Source: `cmfqsar/prototypes.py`

Inverse design on prototypes: candidates, an oracle and a loop (plan item M11).

A co-pattern says which pair of field types at which distance a crystal should gain. Acting on
that needs three things this module provides.

**A candidate generator.** A *prototype* is one cell whose sites are grouped into classes, with a
list of the elements each class may take. Substituting them enumerates candidate structures.
Two modes cover what the materials items ask for:

* *composition*: every site of a class takes the same element, which is the ABO3 case -- one
  prototype, many compositions;
* *arrangement*: a fixed number of the sites of a class take a substituent and the others keep
  what the prototype has, which is the point-defect case -- one composition, many arrangements.
  The element name `"vac"` removes the atom instead of replacing it, so a vacancy series is an
  arrangement series.

The number of candidates is a product and grows quickly, so nothing is built until it is asked
for: `Prototype.n_candidates` counts them, `candidate_at` builds one from its index, and
`candidates` either walks them all or draws a sample without replacement.

**A test for duplicates.** Two assignments that a symmetry operation of the prototype maps onto
each other are one structure. `site_permutations` reads those operations from spglib and turns
them into permutations of the substitutable sites; `unique` keeps one assignment per orbit. Where
spglib finds no symmetry, `unique_by_channels` compares the descriptor channels instead, which is
also the honest test: two cells with the same channels are the same structure for the model,
whatever their coordinates say.

**An oracle and a loop.** The oracle is what computes the property of a proposed structure.
`LookupOracle` reads it from a table, which is how the loop is measured on a published set;
`CalculatorOracle` runs any ASE calculator, with or without a relaxation, which is how a real
loop runs -- the same interface takes a UMA model or a VASP calculator. `design_loop` then
proposes, computes and refits: at every step it fits the kernel model on what the oracle has
answered so far, chooses the ridge by the evidence, scores every candidate by one of the rules of
`cmfqsar/active.py` and asks the oracle about the best one. The loop never looks at a value the
oracle has not been asked for, and never asks twice about the same candidate.

Minimising is the usual materials goal (a formation energy), and the rules of `active.py` all
maximise, so `design_loop(maximise=False)` negates the target inside. With that, `"ucb"` is the
lower confidence bound of plan item E6: a candidate is proposed only when the model is both
optimistic and sure.

### `class SiteClass  (dataclass)`

One group of sites of a prototype and the elements they may take.

`count` is what separates the two modes. `None` means every site of the class takes the
same element, so the class contributes one candidate per element. An integer *k* means
exactly *k* of the sites take one of `elements` and the rest keep what the prototype has, so
the class contributes one candidate per element and per choice of *k* sites.

Attributes:

- `name: str`
- `sites: Tuple[int, ...]`
- `elements: Tuple[str, ...]`
- `count: Optional[int]` = `None`

#### `SiteClass.n_options` -> int  (property)

How many assignments this class alone allows.

#### `SiteClass.option_at(index: int) -> Tuple[str, Tuple[int, ...]]`

The `index`-th assignment: the element, and the sites that take it.

### `class Candidate  (dataclass)`

One proposed structure: where it came from, what it is called, and its cell.

Attributes:

- `index: int`
- `label: str`
- `assignment: Tuple[Tuple[str, Tuple[int, ...]], ...]`
- `atoms: Any`

#### `Candidate.formula` -> str  (property)

The composition, where the structure can give one, and the label otherwise.

A candidate the generator built holds an ASE cell; a candidate taken from a published set
may hold the field record the descriptor was computed from, which has no formula.

### `class Prototype  (dataclass)`

One cell, its site classes, and everything the generator needs.

Attributes:

- `atoms: Any`
- `classes: Tuple[SiteClass, ...]`
- `name: str` = `''`

#### `Prototype.by_symbol(atoms, choices: Mapping[str, Sequence[str]], counts: Optional[Mapping[str, Optional[int]]]=None, name: str='') -> 'Prototype'`  (static method)

A prototype whose classes are the elements of `atoms`: `{"Sr": ["Ca", "Ba"], ...}`.

`counts` gives the class of an element an arrangement count, as `SiteClass.count`.

#### `Prototype.n_candidates` -> int  (property)

How many candidates the classes allow, counted and not built.

#### `Prototype.sites` -> Tuple[int, ...]  (property)

Every substitutable site, in order.

#### `Prototype.assignment_at(index: int) -> Tuple[Tuple[str, Tuple[int, ...]], ...]`

The `index`-th assignment of elements to sites, as one entry per class.

#### `Prototype.candidate_at(index: int) -> Candidate`

Build the `index`-th candidate.

#### `Prototype.site_codes(index: int) -> Tuple[int, ...]`

The assignment as one number per substitutable site, for the duplicate test.

### `class LookupOracle(values: Mapping[int, float] | Sequence[float], name: str='lookup')`

The property of a candidate, read from a table rather than computed.

This is what a retrospective measurement uses: the pool is a published set, the values are
its published labels, and the loop may not see a value it has not asked for. `calls` counts
the questions, which is the budget the gate of plan item M11 is stated in.

### `class CalculatorOracle(make: Callable[[], Any], relax: bool=True, fmax: float=0.05, steps: int=60, per_atom: bool=True, name: str='calculator')`

The energy of a candidate from an ASE calculator, with an optional relaxation.

`make` is called with no arguments and returns a fresh calculator, so that a calculator that
keeps state cannot carry it from one candidate to the next. The same interface takes a
machine-learned potential or a VASP calculator; what changes is the cost of a call and not
the loop.

### `as_candidates(structures: Sequence[Any], labels: Optional[Sequence[str]]=None) -> List[Candidate]`

A pool of candidates from structures that are already built.

This is what a retrospective measurement needs: the pool is a published set, every candidate
keeps its row as its index, and the loop cannot tell it from a pool the generator proposed.

### `cubic_perovskite(a: float=3.9, a_site: str='Sr', b_site: str='Ti', anion: str='O')`

The five-atom cubic ABO3 cell, which is the prototype the measurement of M11 uses.

A at the corner, B at the body centre and the three anions at the face centres. ASE does not
build this structure, and writing it out keeps the site order fixed, which the site classes
of a prototype depend on.

### `candidates(proto: Prototype, limit: Optional[int]=None, seed: int=0, indices: Optional[Sequence[int]]=None) -> List[Candidate]`

Every candidate of a prototype, or a sample of `limit` of them without replacement.

A sample is drawn over the index range, so a pool of a few thousand can be taken from an
enumeration of billions without building the ones left out.

### `site_permutations(proto: Prototype, symprec: float=0.001, tol: float=0.001) -> np.ndarray`

How the symmetry operations of the prototype permute its substitutable sites.

Shape (G, S): row *g* holds, for every substitutable site, the site that operation *g* sends
it to. Operations that send a substitutable site onto a site of another class, or onto no
site at all, are left out, because they do not map the set of candidates onto itself. The
identity is always the first row.

### `unique(proto: Prototype, pool: Sequence[Candidate], perms: Optional[np.ndarray]=None) -> Tuple[List[Candidate], int]`

One candidate per symmetry orbit, and how many were dropped as duplicates.

### `channel_keys(crystals: Sequence[Any], spec, n_fields: int, decimals: int=8) -> List[Tuple]`

One hashable key per structure from its descriptor channels, rounded to `decimals`.

Two structures with the same key are the same structure for a model built on that
descriptor, which is the test that matters: a two-point descriptor gives one key to a
homometric pair, and the three-body channels of plan item M10 give them two.

### `unique_by_channels(crystals: Sequence[Any], spec, n_fields: int, decimals: int=8) -> Tuple[List[int], int]`

The rows to keep so that no two structures share their channels, and how many were
dropped.

### `feature_matrix(crystals: Sequence[Any], spec, n_fields: int, dtype=float) -> np.ndarray`

The feature matrix of a pool under whichever periodic descriptor `spec` names.

### `block_sizes(spec, n_fields: int) -> List[int]`

The channel block widths of whichever descriptor `spec` names.

### `pool_kernel(F: np.ndarray, blocks: Sequence[int]) -> np.ndarray`

The cosine-normalised linear kernel of a pool, each channel block on its own scale.

### `design_loop(pool: Sequence[Candidate], oracle: Callable[[Candidate], float], *, start: Sequence[int], budget: int, kernel: Optional[np.ndarray]=None, spec=None, n_fields: Optional[int]=None, rule: str='ucb', beta: float=1.0, xi: float=0.0, lams: Sequence[float]=(0.001, 0.01, 0.1, 1.0), maximise: bool=True, fit_bias: bool=True, seed: int=0, log: Optional[Callable[[str], None]]=None) -> dict`

Propose, compute and refit, until the budget of oracle calls is spent.

`start` are positions in `pool` whose values the oracle is asked for before the loop begins;
they are the calculations a design study starts from. `budget` is how many further calls the
loop may make. The kernel over the pool is given or built from `spec`, and nothing about the
pool changes while the loop runs, so a candidate the loop never proposes costs nothing beyond
its descriptor.

With `maximise` false the target is negated inside, so that every rule still prefers a larger
score and `"ucb"` reads as the lower confidence bound.

Returns the record: one entry per step with what was proposed and what the oracle answered,
the best value found, and which candidate holds it.

### `first_better(record: dict, threshold: float) -> Optional[int]`

The step at which the loop first found a candidate past `threshold`, or None.

"Past" means above for a loop that maximises and below for one that minimises, so the same
call reads a formation energy and a figure of merit.

### `summarise(record: dict) -> str`

One line for a log: what the loop spent and what it found.

## cmfqsar.pu

Source: `cmfqsar/pu.py`

Learning from positives and unlabelled molecules (part of plan item E4).

A screening library is not a labelled set.  The compounds that gave a negative result in one assay
at one concentration are *unlabelled* rather than inactive: some are genuinely inert, some were
mis-measured, some would have worked at another concentration, and a few are actives nobody has
recognised yet.  Treating that mixture as a negative class teaches a classifier to reject the very
molecules a screen is looking for, and the error is worst exactly where it matters, at the top of
the ranking.
The remedy used here is **bagging PU** (Mordelet & Vert, *Pattern Recognit. Lett.* **37** (2014)
201), which is the method that needs no estimate of the class prior and no assumption about how
the positives were sampled:

1. draw a small random sample of the unlabelled set and call it negative;
2. fit a classifier on the positives against that sample;
3. score the unlabelled molecules that were *not* in the sample -- out of bag, so the score of a
   molecule is never taken from a model that was told it was negative;
4. repeat, and average each molecule's out-of-bag scores.

An unlabelled molecule that looks like the positives is called negative in only a few bags, and
in the bags where it is left out it scores high; one that is genuinely unlike them is called
negative often and scores low wherever it appears.  The average is the quantity to rank by.

Everything here works on a precomputed kernel, so the CMF kernel enters unchanged and the
positives keep whatever pose they were given.  `bagging_pu` is the general form;
`pu_scores` is the convenience that fits the bags and returns one score per molecule.

### `class PUResult  (dataclass)`

Out-of-bag scores for every molecule, and how they were arrived at.

Attributes:

- `scores: np.ndarray`
- `bags: int`
- `bag_size: int`
- `counts: np.ndarray`
- `positives: np.ndarray`
- `unlabelled: np.ndarray`
- `note: str` = `''`

#### `PUResult.covered` -> float  (property)

The share of the unlabelled molecules that got at least one out-of-bag score.

#### `PUResult.describe() -> str`

### `bagging_pu(K: np.ndarray, positives: Sequence[int], unlabelled: Sequence[int], bags: int=30, bag_size: Optional[int]=None, lam: float=0.1, seed: int=0, also: Optional[Sequence[int]]=None) -> PUResult`

Bagged positive-unlabelled scores on a precomputed kernel.

`also` names molecules that are in neither set -- a held-out test fold, say -- which are
scored by every bag, since no bag ever sees them. Their scores are therefore comparable with
the out-of-bag scores of the unlabelled set, which is what makes a ranking over both of them
fair.

### `pu_scores(K: np.ndarray, positives: Sequence[int], unlabelled: Sequence[int], test: Optional[Sequence[int]]=None, bags: int=30, bag_size: Optional[int]=None, lam: float=0.1, seed: int=0) -> np.ndarray`

The ranking a PU model gives, as one number per molecule (nan where never scored).

### `naive_scores(K: np.ndarray, positives: Sequence[int], unlabelled: Sequence[int], test: Optional[Sequence[int]]=None, lam: float=0.1) -> np.ndarray`

The ranking you get by calling every unlabelled molecule a negative, once.

The control bagging PU has to beat: the same learner and the same kernel, with the whole
unlabelled set used as the negative class and no bagging. Out-of-sample scores are available
only for `test`, so this is the ranking of the held-out molecules, not of the training ones.

## cmfqsar.ranking

Source: `cmfqsar/ranking.py`

Learning to order molecules instead of to reproduce their numbers.

A regression model is asked for the activity and judged on the squared error, which assumes the
numbers mean what they say.  Often they do not.  Potencies pooled from several papers carry
offsets no descriptor can explain, a series measured in two assays carries two scales, and an
IC50 read off the end of a dose-response curve carries a bound rather than a value.  In all three
the *order* survives what the values do not, and a model fitted to the values spends its capacity
on the part that is noise.

The ranking support-vector machine (Herbrich et al. 2000; Joachims, KDD 2002) fits the order.
For every pair of molecules whose activities differ it asks that the more active one score higher
by a margin, and it is penalised for every pair where that fails:

    minimise  |w|^2 / 2 + C sum_p xi_p
    subject to  <w, X_i - X_j> >= 1 - xi_p  for every pair p = (i, j) with y_i > y_j

which is an ordinary support-vector classification on the *difference* of two molecules' fields.
The kernel of two such differences is a sum of four ordinary kernel entries,

    Kpair((i,j), (k,l)) = K(i,k) - K(i,l) - K(j,k) + K(j,l),

so the whole thing is built from the kernel matrix the program already has, with no new fields
and no new parameterisation.

**The fitted model is still a dual expansion over molecules.**  The weight vector is
w = sum_p a_p (X_{i_p} - X_{j_p}), so collecting the terms gives one coefficient per *molecule*:
the score of a new molecule is K(x, train) @ coef exactly as for any other learner here, with no
bias term, and everything that reads a linear CMF model -- co-fields, atom contributions, Shapley
values, the pharmacophore -- keeps working unchanged.  What changes is only what the numbers
mean: a co-field of a ranking model says where more field *promotes* a molecule, not what it adds
to the activity.

What is given up is the scale.  A ranking score has no units and cannot be compared with an
activity; only its order is meaningful.  The program therefore reports Spearman's rho, Kendall's
tau and the fraction of pairs ordered correctly, and does not report r2 or RMSE, which would be
arithmetic on a quantity that has none.

### `class RankFit  (dataclass)`

What a ranking fit produces: one coefficient per training molecule.

Attributes:

- `coef: np.ndarray`
- `pairs: np.ndarray`
- `alpha: np.ndarray`
- `n_pairs: int` = `0`
- `n_support: int` = `0`

#### `RankFit.score(K_test_train: np.ndarray) -> np.ndarray`

The ranking score of new molecules; only its order means anything.

### `make_pairs(y: Sequence[float], gap: float=0.0, groups: Optional[Sequence]=None, max_pairs: Optional[int]=None, seed: int=0) -> np.ndarray`

Ordered pairs (i, j) with y_i > y_j, as an (m, 2) integer array.

`gap` drops the pairs whose activities differ by less than it.  This is not only a way to save
time: a pair separated by less than the experimental error is a matter of chance the model
would be asked to predict correctly, and asking it is how a ranking model learns noise.  Half a
log unit is a defensible default for pooled literature potencies, zero for one careful assay.

`groups` restricts the pairs to within a group -- the same assay, the same paper, the same
series -- which is what makes the method worth having on pooled data: an offset between two
papers cancels inside each of them and is never compared across.

`max_pairs` subsamples when the quadratic count is too large; the subsample is drawn once with
`seed` so that a fit is reproducible.  It matters more than it seems: the solver works on the
kernel *between pairs*, which is m by m for m pairs, so m = 4000 already costs 128 MB and a
set of 300 molecules would otherwise form 45,000 pairs and ask for 16 GB.  Subsampling the
constraints is the standard remedy for the pairwise formulation, and the order of a set is
heavily over-determined: a few thousand comparisons fix it.

### `pair_kernel(K: np.ndarray, pairs: np.ndarray, signs: Optional[np.ndarray]=None, other: Optional[np.ndarray]=None, other_signs: Optional[np.ndarray]=None) -> np.ndarray`

The kernel between difference vectors: K(i,k) - K(i,l) - K(j,k) + K(j,l).

With `signs` the rows are flipped -- a pair used as a negative example is the same difference
taken the other way round -- which is how the one-class problem of "every difference is
positive" is turned into the balanced binary problem a standard solver expects.

### `fit_rank(K: np.ndarray, y: Sequence[float], C: float=1.0, gap: float=0.0, groups: Optional[Sequence]=None, max_pairs: Optional[int]=4000, seed: int=0) -> RankFit`

Fit a ranking SVM on a precomputed kernel and return molecule-level coefficients.

The pairs are turned into a balanced binary problem by flipping half of them -- a difference
taken the other way round with the opposite label is the same constraint -- so that an
ordinary two-class solver can be used.  The intercept it fits is discarded: a constant added
to every score changes no order, and the pairs are symmetric by construction.

### `concordant_pairs(y_true: Sequence[float], score: Sequence[float], gap: float=0.0, groups: Optional[Sequence]=None) -> float`

The fraction of comparable pairs the score puts in the correct order; ties count a half.

### `rank_scores(y_true: Sequence[float], score: Sequence[float], gap: float=0.0, groups: Optional[Sequence]=None) -> Dict[str, float]`

Spearman's rho, Kendall's tau and the share of pairs ordered correctly.

## cmfqsar.reciprocal

Source: `cmfqsar/reciprocal.py`

Reciprocal-space channels of a crystal: the typed structure factors (plan item M6).

The periodic autocorrelation of `periodic.py` is a sum over a neighbour list, so it stops at
the cutoff radius `r_max`.  Order that extends further -- a superstructure with a long repeat,
the long-range part of the electrostatic interaction, a modulation -- leaves no mark on it.  This
module adds the channels that do carry such order, and it pays nothing for the distance: they are
sums over the atoms of **one cell** with no neighbour list at all.

The structure factor
--------------------
For a field f with weight `w_lf` on atom l at position `r_l`, and a vector **G** of the
reciprocal lattice (the lattice of vectors with `G . T = 2 pi n` for every lattice translation
T), the *typed structure factor* is the sum over the atoms of one cell

    S_f(G) = sum_l w_lf exp(i G . r_l).

It is the Fourier transform of the field of the whole infinite crystal, evaluated at G, and it
is exactly what a diffraction experiment measures: `|S_f(G)|^2` is the intensity of the Bragg
reflection at G when field f is the scattering power.  A single value of S_f(G) already holds
information about the whole structure, however far the order extends, which is why this
descriptor has no cutoff radius.

The channels
------------
For a pair of field types (f, g) and a channel at the length `q_k`,

    Psi_fg(q_k) = (1/N) { (2 pi)^3 / (V I_k) sum_{G != 0} A_fg(G) F(|G|) B_k(|G|)
                          + (e_fg - c_fg) <F>_k },

with

* `A_fg(G) = |S_f(G)|^2` for one field type and `2 Re[S_f(G) conj(S_g(G))]` for two, which
  counts a pair of different types in both orders exactly as the real-space descriptor does;
* `c_fg` the part of `A_fg` that comes from an atom with itself, `sum_l w_lf w_lg` times
  the same factor, and `e_fg` the same sum of the *second moments* of a disordered site.  The
  two are equal for an ordered structure, so the term vanishes there; where they differ it is
  the correction that partial occupancy asks for (plan item M2).  Neither is summed over the
  reciprocal lattice: a quantity that does not depend on G contributes `<F>_k`, the mean of
  the form factor over the channel, and taking it from that integral rather than from the grid
  is what makes the descriptor independent of the choice of cell *exactly* (see
  `ReciprocalSpec.form_average`).  `include_self=False` subtracts the self scattering
  altogether, which leaves the distinct scattering of Faber and Ziman;
* `F(q) = exp(-a^2 q^2)` is the *form factor*: the two fields are Gaussians of width `a`
  (`form_sigma`), so their product in reciprocal space carries this factor.  It falls with q,
  which makes `a` a **resolution**: a wide field sees only the first few reflections, a narrow
  one sees fine detail.  This is the reciprocal-space reading of the width that the real-space
  descriptor uses along the distance axis;
* `B_k(q) = exp(-(q - q_k)^2 / 2 sigma^2) w(q)` is the channel, a Gaussian about `q_k` with
  the same tapered window as the real-space cutoff, and `I_k` its shell integral
  `INT 4 pi q^2 B_k(q) dq`;
* `V` is the volume of the cell and `N` the number of atoms in it (the total occupancy for a
  disordered cell).

Dividing by `V I_k / (2 pi)^3` -- the number of reciprocal-lattice points a channel is
expected to contain -- makes the channel the *spherical average of the intensity*, which is the
powder diffraction profile of that pair of field types.  It also makes the descriptor exactly
independent of the choice of cell: a supercell has as many times more reciprocal-lattice points
as it has cells, and the new ones carry no intensity, so numerator and denominator grow by the
same factor.  Nothing here depends on the origin or the orientation either, so the descriptor
has the same invariances as the real-space one and adds no new assumption.

Two further things follow from the definition and are worth stating, because the real-space
module needs both and has them only as approximations:

* **The uniform background is the G = 0 term.**  `A_fg(0) = (sum_l w_lf)(sum_l w_lg)`, the
  forward scattering, and that is the quantity the real-space descriptor subtracts as the
  uniform-density term.  Here it is one term of the sum, left out by `include_zero=False`, and
  no integral has to be approximated to remove it.
* **The two descriptions are one object.**  Summing the field of the crystal in real space and
  summing it over the reciprocal lattice give the same number (Poisson summation, the identity
  an Ewald sum is built on): for any width `u`,

      sum_{l, m, T} w_lf w_mg exp(-|r_l - r_m - T|^2 / 2 u^2)
          = (2 pi u^2)^{3/2} / V  sum_G A_fg(G) exp(-u^2 |G|^2 / 2).

  `parseval_residual` computes both sides and their difference.  The left side is this
  program's real-space pair sum and the right side is this module's sum over G, so the identity
  is a test of the two pieces of code against each other, and it is the statement that the
  reciprocal channels are the same descriptor read at a different resolution rather than a new
  one.

What this adds and what it does not
-----------------------------------
A reciprocal channel at small `q` describes the structure on a length scale `2 pi / q`, so
the low-q channels carry exactly what a real-space cutoff cannot reach.  A cell with few atoms
has few reciprocal-lattice points below `q_max`, and its profile is a short series of sharp
peaks whose positions are set by the lattice metric; a large cell gives a dense profile.  The
descriptor still cannot tell a structure from its mirror image, because the intensity of a
reflection is the same for both: that deficiency is shared with every two-point descriptor and
is the subject of plan item M10.

Only structures periodic in all three directions are accepted.  A layer or a chain has a
reciprocal lattice of lower dimension, and the intensity then depends on the component of G
across the non-periodic directions as well, which these channels do not resolve.

### `class ReciprocalSpec  (dataclass)`

The grid of reciprocal-space channels: where they sit, how wide they are, and the
resolution the form factor sets.

The layout matches `cmfqsar.periodic.PeriodicSpec`, so the block and label helpers of
`invariant.py` accept this spec unchanged: channels at 0, `step`, 2 `step`, ... up to
`q_max`, one block per pair of field types.  `chirality` exists only for those helpers
and is always False.

Attributes:

- `q_max: float` = `8.0`
- `step: float` = `0.25`
- `sigma: float` = `0.25`
- `form_sigma: float` = `0.3`
- `include_self: bool` = `True`
- `include_zero: bool` = `False`
- `taper: float` = `1.0`
- `chirality: bool` = `False`

#### `ReciprocalSpec.d_max` -> float  (property)

The spec's own extent, for the helpers that ask every spec for one.

#### `ReciprocalSpec.has_heights` -> bool  (property)

#### `ReciprocalSpec.centres` -> np.ndarray  (property)

#### `ReciprocalSpec.n_bins` -> int  (property)

#### `ReciprocalSpec.window(q: np.ndarray) -> np.ndarray`

1 below q_max, a half-cosine over the last `taper`, 0 beyond.

#### `ReciprocalSpec.form_factor(q: np.ndarray) -> np.ndarray`

`exp(-a^2 q^2)`: the product of the two Gaussian fields in reciprocal space.

#### `ReciprocalSpec.shell_integral(n_points: int=4000) -> np.ndarray`

`INT 4 pi q^2 exp(-(q - q_k)^2 / 2 sigma^2) w(q) dq` for every channel.

The number of reciprocal-lattice points a channel is expected to hold, per unit density
of those points.  It depends on the grid, the width and the window only, so it is
computed once per spec.

#### `ReciprocalSpec.form_average(n_points: int=4000) -> np.ndarray`

The mean of the form factor over every channel: the shell integral with `F` in it,
divided by the shell integral without.

It is what a quantity that does not depend on the direction or the length of G
contributes to a channel, and the self-scattering term is such a quantity.  Computing
that contribution from this integral rather than by summing it over the
reciprocal-lattice points is what makes the descriptor exactly independent of the choice
of cell: a small cell has few points per channel, so the sum would estimate a constant
badly and differently for a supercell of the same structure.

### `reciprocal_cell(cell: np.ndarray) -> np.ndarray`

The reciprocal lattice vectors as rows: `b_i . a_j = 2 pi delta_ij`.

### `count_reciprocal_vectors(cell: np.ndarray, q_max: float) -> float`

About how many reciprocal-lattice vectors lie inside `q_max`.

The reciprocal lattice has `V / (2 pi)^3` points per unit volume, so the count is that
density times the volume of the sphere.  It grows with the cell and with the cube of the
cutoff, which is what makes a large cell at a high cutoff expensive, and this is the cheap
way to find out before paying.

### `reciprocal_vectors(cell: np.ndarray, q_max: float, include_zero: bool=False, max_vectors: int=4000000) -> Tuple[np.ndarray, np.ndarray]`

The reciprocal-lattice vectors with `|G| <= q_max`, and their lengths.

A vector `G = sum_i h_i b_i` has `h_i = G . a_i / 2 pi`, so `|h_i| <= q_max |a_i| / 2 pi`
bounds the search exactly, whatever the shape of the cell.  The search runs one plane of
indices at a time and keeps only what is inside the sphere: holding the whole box of indices
would cost the volume of the box, which for a large cell at a high cutoff is many times the
sphere and can be gigabytes.  `max_vectors` refuses a request whose result would be larger
than that, rather than letting it exhaust the machine.

### `structure_factors(crystal, G: np.ndarray, weights: Optional[np.ndarray]=None) -> np.ndarray`

`S_f(G)` for every vector of `G` and every field: a complex `(len(G), n_fields)`.

`weights` overrides the field weights of the structure, which is what the Parseval test
and the per-field sums need.

### `reciprocal_autocorrelation(crystal, spec: ReciprocalSpec=ReciprocalSpec(), pairs: Optional[Sequence[Tuple[int, int]]]=None) -> np.ndarray`

Reciprocal-space channels of one structure, flattened: `(n_pairs * n_bins,)`.

The formula is the one in the module docstring.  The sum runs over the reciprocal-lattice
vectors of the cell out to `q_max` and over the atoms of one cell; there is no neighbour
list and no cutoff in real space.

### `reciprocal_feature_matrix(crystals: Sequence, spec: ReciprocalSpec=ReciprocalSpec(), n_fields: Optional[int]=None, progress=None) -> np.ndarray`

Reciprocal-space channels of a set of structures, one row each.

### `reciprocal_block_sizes(spec: ReciprocalSpec, n_fields: int) -> List[int]`

### `reciprocal_channel_labels(fieldset: FieldSet, spec: ReciprocalSpec) -> List[str]`

One label per feature: which pair of field types, at which reciprocal length.

### `joint_features(crystals: Sequence, real_spec, spec: ReciprocalSpec=ReciprocalSpec(), n_fields: Optional[int]=None, progress=None) -> Tuple[np.ndarray, List[int]]`

The real-space and reciprocal-space channels side by side, and the block widths.

The two descriptions of the same structure in one feature matrix: the pair blocks of the
periodic autocorrelation first, then the pair blocks of this module.  Each block keeps its
own scale (`cmfqsar.invariant.channel_scale` takes the widths), so neither description
dominates the kernel by its units.  Mixing the two as separate *kernels* is the other way,
and the one the benchmark uses when it wants to measure how much weight each deserves.

### `joint_labels(fieldset: FieldSet, real_spec, spec: ReciprocalSpec) -> List[str]`

The labels of `joint_features`, in the same order.

### `gaussian_lattice_sum(crystal, width: float, pairs: Optional[Sequence[Tuple[int, int]]]=None, r_cut: float=8.0) -> np.ndarray`

`sum_{l, m, T} w_lf w_mg exp(-|r_l - r_m - T|^2 / 2 u^2)` for every pair of field types.

The real-space side of the identity in the module docstring: a sum over the neighbour list of
the cell, with the atom-with-itself term at distance zero added, out to `r_cut` widths.

### `gaussian_reciprocal_sum(crystal, width: float, pairs: Optional[Sequence[Tuple[int, int]]]=None, q_cut: float=8.0) -> np.ndarray`

`(2 pi u^2)^{3/2} / V sum_G A_fg(G) exp(-u^2 |G|^2 / 2)` for every pair of field types.

The reciprocal side of the same identity, over the reciprocal lattice out to `q_cut / u`.

### `parseval_residual(crystal, width: float=0.5, pairs: Optional[Sequence[Tuple[int, int]]]=None, cut: float=8.0) -> dict`

The two sides of the identity of the module docstring, and how far apart they are.

Returns the real-space sum, the reciprocal-space sum and the largest relative difference
over the pairs of field types.  The two are the same quantity computed by two independent
pieces of code -- a neighbour list over the images, and a sum over the reciprocal lattice --
so a small residual is evidence that both are right, and it is the test the theory section
of the publication plan asks for.

### `channel_spread(model) -> np.ndarray`

The standard deviation of every scaled, normalised channel over the training structures.

A weight only matters where the channel varies, so this is the other half of reading a
fitted model; `cmfqsar.periodic.top_channels` is where the program defines it.

### `reciprocal_pattern(model, n_fields: Optional[int]=None, spec: Optional[ReciprocalSpec]=None, effect: bool=False) -> np.ndarray`

The reciprocal channels of a fitted model read back: `(n_pairs, n_bins)`.

The co-pattern of the real-space descriptor says which pair of field types at which distance
raises the property.  This is the same reading in reciprocal space: which pair of field types
at which reciprocal length, that is, at which spacing of planes in the structure.  A positive
value at `q` means that more intensity there raises the property.

`effect=True` multiplies each weight by the spread of its channel over the training
structures, which is how much that channel actually moves the prediction and is how the
program ranks channels elsewhere.  The weights themselves change sign from one channel to
the next, because neighbouring channels of a Gaussian grid are strongly correlated and a
ridge splits a single feature among them; the product with the spread is the quantity to
read.

### `pattern_table(model, fieldset: FieldSet, spec: Optional[ReciprocalSpec]=None, n: int=10) -> List[Tuple[str, float, float]]`

The `n` reciprocal channels that move the prediction most.

(label, weight times the spread of the channel, the reciprocal length q), ranked by the
size of the first, which is the ranking `cmfqsar.periodic.top_channels` uses.

### `spacings(spec: ReciprocalSpec) -> np.ndarray`

The plane spacing `2 pi / q` of every channel, in angstrom; infinity for the first.

What a channel means as a length: a channel at `q` describes the structure on the scale of
`2 pi / q`, which is how the reciprocal channels are compared with the cutoff radius of the
real-space descriptor.

## cmfqsar.report

Source: `cmfqsar/report.py`

One HTML file that says what a model is, how well it works and what it says (plan item F4).

This is written for the person who will never start the application: a colleague, a reviewer, a
supervisor who wants to see the model that produced a number.  It is one self-contained file --
every picture is a base64 PNG inside it, so it survives being emailed, attached to a report or put
on a shared drive -- and it draws only on what the model file already carries, so a report costs no
refitting and cannot disagree with the model it describes.

Seven sections, in the order a sceptical reader asks for them:

1. **What this model is** -- fields, attenuation, learner, kernel, how the charges were made.
2. **How well it fits and predicts** -- the statistics with the observed-against-predicted plots
   beside them, which is the only fair way to show a q².
3. **Whether to believe that** -- the validation that was actually run: y-randomisation, the
   bootstrap interval, the applicability domain, the conformal level, the error predictors.  A
   check that was not run is reported as not run, never silently omitted.
4. **What the model rests on** -- the mean absolute contribution of each field over the training
   molecules, which is what distinguishes a model that rests on its geometry from one that rests
   on one field.
5. **The co-fields**, projected onto the three coordinate planes from fixed viewpoints: where the
   model needs charge, bulk, a donor.  A projection rather than an isosurface, because a report has
   no camera and a reader has no mouse.
6. **The extremes of the series** -- the most and least active training molecules, each with its
   exact field-level Shapley decomposition and its largest atom contributions.
7. **The pharmacophore** read out of the co-fields, and the provenance of the model file.

Everything degrades rather than fails.  A nonlinear kernel has no co-fields, no contributions and
no pharmacophore; the report says so in place of each and keeps the rest.  A model with no test set
reports no R²pred.  A classifier reports its own statistics.

### `class ReportSpec  (dataclass)`

What the report draws, and how finely.

Attributes:

- `spacing: float` = `0.6`
- `margin: float` = `3.0`
- `level: float` = `0.5`
- `max_features: int` = `12`
- `top_atoms: int` = `8`
- `cofield_z: float` = `2.0`
- `cofields: bool` = `True`
- `pharmacophore: bool` = `True`
- `dpi: int` = `110`

### `scatter_png(panels: Sequence[Dict], spec: ReportSpec) -> bytes`

Observed against predicted, one panel per set of predictions, on a shared square scale.

### `cofield_png(model, f: int, grid, volume: np.ndarray, spec: ReportSpec) -> bytes`

One field's co-field from the three fixed viewpoints, with the training atoms over it.

Each panel is the signed extremum along the viewing axis -- the strongest thing the reader
would see looking down that axis, with its sign kept -- which is the projection that does not
cancel a positive lobe against a negative one behind it.

### `contributions_png(names: Sequence[str], values: Sequence[float], spec: ReportSpec) -> bytes`

The mean absolute contribution of each field, as a bar per field.

### `build_html(model, extra: Optional[Dict]=None, title: Optional[str]=None, spec: Optional[ReportSpec]=None, source: str='', progress=None) -> str`

The whole report as one self-contained HTML document.

### `write_report(path: str, model, extra: Optional[Dict]=None, title: Optional[str]=None, spec: Optional[ReportSpec]=None, source: str='', progress=None) -> str`

Write the report to `path` and return the path.

## cmfqsar.restriction

Source: `cmfqsar/restriction.py`

Scoring a conformational restriction (plan item MC9).

Rigidifying a molecule -- tying two parts of it together into a ring so that it can no longer
rotate freely -- is one of the few moves in medicinal chemistry whose effect is meant to be purely
conformational.  The fourth-dimension verdict of this repository is that an ensemble of
conformations does not predict better than one geometry (plan item D9), and plan item D10 found
that the variance of the field across an ensemble adds nothing as a block of the kernel either.
The variance is still a map of **where a molecule places field only part of the time**, and this
module asks the one question that map is suited to: when a chemist closes a ring, how much of the
field that matters for activity stops moving?

Two numbers describe a restriction
----------------------------------
The co-field :math:`C_f(r)` of a fitted model says how much the field of type *f* at point *r*
matters for the activity.  Its **square** is therefore a weight: positive everywhere, large where
the model reads the field and near zero where it does not.  With that weight, one molecule gets
two amounts:

* **relevant** :math:`\int C_f(r)^2 \bar{X}_f(r)^2 dr`, how much activity-relevant field the
  molecule places, averaged over its conformations;
* **intermittent** :math:`\int C_f(r)^2 V_f(r) dr`, how much of that field is there only part of
  the time, with :math:`V_f` the variance field of `cmfqsar.fluctuation`.

Both are summed over the field types.  A pair of a flexible molecule **A** and its restricted
analogue **B** is then scored by two shares:

* **kept** = relevant(B) / relevant(A) -- did the restriction keep the field that matters, or did
  closing the ring move it away?
* **removed** = 1 - intermittent(B) / intermittent(A) -- how much of the intermittent part the
  restriction took out.

A restriction that works should keep the first and raise the second.  The gate of plan item MC9
is that **removed** tracks the measured potency gain over a set of such pairs;
`benchmarks/run_restriction.py` measures it.

Both members of a pair must come from the same conformer generator with the same settings: the
comparison is between two ensembles, not between two molecules, and a different number of
conformations changes the variance by itself.

### `class Pair  (dataclass)`

A flexible molecule and the analogue that ties part of it into a ring.

Attributes:

- `flexible: int`
- `restricted: int`
- `common: float`
- `rotatable_removed: int`
- `gain: float`
- `names: Tuple[str, str]` = `('', '')`

### `class Amounts  (dataclass)`

Per molecule: the activity-relevant field it places, and how much of it is intermittent.

Attributes:

- `relevant: np.ndarray`
- `intermittent: np.ndarray`
- `per_field: np.ndarray` = `_field(default_factory=lambda: np.zeros((0, 0, 2)))`
- `fields: List[str]` = `_field(default_factory=list)`
- `grid: Optional[Grid]` = `None`

#### `Amounts.intermittent_share` -> np.ndarray  (property)

How much of the activity-relevant field of each molecule is there only part of time.

### `class Score  (dataclass)`

What one restriction did, as two shares.

Attributes:

- `kept: float`
- `removed: float`
- `gain: float`
- `names: Tuple[str, str]` = `('', '')`

### `matched_pairs(mols: Sequence, y: Sequence[float], names: Optional[Sequence[str]]=None, mcs_share: float=MCS_SHARE, max_extra: int=MAX_EXTRA_ATOMS, max_composition: int=MAX_COMPOSITION, timeout: int=2) -> List[Pair]`

Every pair in a set where the second molecule is the first with one more ring.

The rule is deliberately mechanical, so that the pairs are not chosen by eye and the
measurement cannot be steered.  The restricted member has exactly one ring more, at least one
rotatable bond fewer, at most `max_extra` heavy atoms more, an atom composition differing by
at most `max_composition` atoms counted element by element, and a maximum common substructure
covering at least `mcs_share` of the smaller of the two.  The last condition is the one that
decides, and a ring closure that also changes a substituent fails it; the ones before it are
cheap, and keep the common-substructure search off most of the pairs in a set.

A fingerprint is **not** used here, although it is the usual way to ask whether two molecules
are alike.  Closing a ring changes the environment of every atom near the new bond, so
butylphenol and 2-methylchroman -- the same eleven heavy atoms, one bond more -- are only
0.15 alike as Morgan fingerprints, and a threshold high enough to be useful would throw away
exactly the pairs this item is about.  The atom composition has no such blind spot.

### `training_fields(model, grid: Grid) -> List[np.ndarray]`

One (training molecules, grid points) array per enabled field: the co-field basis.

Every co-field of this model, with any coefficients, is a combination of these rows, so the
grid work is done once and the leave-one-pair-out co-fields below cost a matrix product each.

### `cofield_from(basis: Sequence[np.ndarray], model, rows: Optional[Sequence[int]]=None, coef: Optional[np.ndarray]=None) -> List[np.ndarray]`

The co-field of every field on the grid points, with its scales included.

`rows` and `coef` refit nothing; they choose which training molecules the co-field is built
from and with what dual coefficients.  That is how a co-field that has not seen a pair is
made: solve the ridge without the two members, and pass the coefficients here with their rows.

### `cofield_on_grid(model, grid: Grid, rows: Optional[Sequence[int]]=None, coef: Optional[np.ndarray]=None) -> List[np.ndarray]`

The co-field on the grid, evaluating the basis on the spot.

### `grid_volumes(means: Sequence[MoleculeFields], variances: Sequence[MoleculeFields], specs, vspecs, grid: Grid, rows: Optional[Sequence[int]]=None, progress=None) -> np.ndarray`

(molecules, fields, 2, grid points): the mean field and the variance field of each.

The two slices of the third axis are the mean field and the variance field.  They are held as
32-bit numbers because they are only ever summed against a weight, and a set of a few hundred
molecules would otherwise need gigabytes.

### `amounts_from(cofields: Sequence[np.ndarray], volumes: np.ndarray, grid: Grid, fields: Optional[Sequence[str]]=None) -> Amounts`

The two amounts of every molecule, from grid volumes that are already built.

### `weighted_amounts(cofields: Sequence[np.ndarray], means: Sequence[MoleculeFields], variances: Sequence[MoleculeFields], specs, vspecs, grid: Grid, progress=None) -> Amounts`

The two amounts of every molecule, from its mean field and its variance field.

`cofields` holds one array per field over the grid points, as `cofield_on_grid` returns them;
`specs` and `vspecs` are the field descriptions of the mean fields and of the variance fields
(the variance of a Gaussian field is a Gaussian field of twice the width, which is why the
second set is needed; see `cmfqsar.fluctuation.variance_fieldset`).

### `grid_for(means: Sequence[MoleculeFields], margin: float=DEFAULT_MARGIN, spacing: float=DEFAULT_SPACING) -> Grid`

One grid wide enough for every molecule of the set.

### `score_pair(amounts: Amounts, pair: Pair) -> Score`

The two shares of one pair.

### `score_pairs(amounts: Amounts, pairs: Sequence[Pair]) -> List[Score]`

### `correlation(x: Sequence[float], y: Sequence[float], runs: int=2000, seed: int=0) -> Tuple[float, float, float]`

Pearson correlation of two columns with a bootstrap interval over the pairs.

## cmfqsar.scattering

Source: `cmfqsar/scattering.py`

Measured scattering curves as samples: the total, untyped descriptor (plan item M12).

A diffraction experiment on a powder, a glass or a nanoparticle does not give atomic
coordinates.  It gives one curve: the total structure function `F(Q)`, or its sine transform
the reduced pair distribution function `G(r)`.  This module connects that curve to the
descriptors of this program, so that a model can be fitted where no structure is known.

The curve is one channel block of the descriptor
------------------------------------------------
The periodic autocorrelation of `periodic.py` is a *typed* pair-distance distribution: one
block per pair of field types.  An experiment cannot separate the types, because every atom
scatters into the same detector.  What it measures is the single block of one field whose weight
is the atom's **scattering length** -- the neutron scattering length `b`, or for X-rays the
atomic number `Z`.  Written out, with `N` atoms in the cell, number density `rho_0` and
`<b>` the mean scattering length,

    G(r) = Phi_bb(r) / ( sigma sqrt(2 pi) * r * <b>^2 ),

where `Phi_bb` is `cmfqsar.periodic.periodic_autocorrelation` of that one field, with
the uniform background kept and the self term dropped.  The background term of the descriptor is
exactly the `- 4 pi r rho_0` of the standard definition, and the factor
`sigma sqrt(2 pi)` is the integral of the channel's Gaussian.  The identity is exact as the
channel narrows: the error is of order `(sigma / r)^2`, and `pair_distribution_residual`
measures it against a direct sum over the neighbour list.

The same holds in reciprocal space.  The Faber-Ziman total structure factor is the one-field
channel of `reciprocal.py` divided by `<b>^2`,

    S(Q) = Psi_bb(Q) / <b>^2,        F(Q) = Q [ S(Q) - 1 ],

with the form factor switched off, and `G` and `F` are a sine transform apart,
`G(r) = (2/pi) INT F(Q) sin(Qr) dQ`.  So the two things an experiment can report and the two
descriptions this program already had are one object seen four ways.

What this buys, and what it costs
---------------------------------
It buys a sample where there is no structure: a measured curve is a feature row like any other,
so a model fitted on computed curves can predict for a measured one, and curves of amorphous or
nanocrystalline materials become usable.  It costs the typing.  Whether that cost is affordable
is a question about the data and not about the code, and `benchmarks/run_scattering.py` measures
it: the total curve against the typed descriptor on the same structures and the same folds.

A measured curve is also not a computed one.  Four things separate them, and
`measured_like` applies all four to a computed curve so that a model can be tested against
them before any experiment is trusted:

* **a finite range in Q.**  The transform runs to the largest Q the instrument reached, which
  puts ripples of period `2 pi / Q_max` around every peak;
* **counting noise** on `F(Q)`;
* **an error in the number density**, which adds a straight line `-4 pi r delta rho` to
  `G(r)`, because `rho_0` is what the baseline is drawn from;
* **a limited range in r**: the low-r region is dominated by termination ripples and the high-r
  region by the instrument's resolution.

Reading a measured file
-----------------------
`read_curve` reads the two-column text that PDFgetX3, diffpy and the other common programs
write: comment lines starting with `#`, then `r` and `G` (or `Q` and `F`, or `Q` and
`S`), with any further columns ignored.  Values the header happens to carry -- `qmax`,
`rho0`, `composition` -- are returned beside the curve when they are there.

### `class CurveSpec  (dataclass)`

The grids a curve is reported on: `r` for `G(r)` and `Q` for `F(Q)`.

`sigma` is the width of the channel the pair distribution is computed with, which stands
for the resolution of the instrument in real space.  The identity with the descriptor is
exact as `sigma` falls, so a narrow channel is the honest setting and a wide one is the
statement that the measurement is blurred.

Attributes:

- `r_min: float` = `1.0`
- `r_max: float` = `20.0`
- `r_step: float` = `0.05`
- `sigma: float` = `0.05`
- `q_min: float` = `0.5`
- `q_max: float` = `25.0`
- `q_step: float` = `0.05`
- `margin: float` = `6.0`

#### `CurveSpec.r` -> np.ndarray  (property)

#### `CurveSpec.q` -> np.ndarray  (property)

#### `CurveSpec.periodic_spec()`

The descriptor spec whose one-field channels are this curve.

#### `CurveSpec.reciprocal_spec()`

The reciprocal spec whose one-field channels are this structure factor.

The form factor is switched off (`form_sigma` 0): a scattering experiment carries its
own, and putting a second one in would damp the curve twice.

### `scattering_lengths(symbols: Sequence[str], source: str='xray') -> np.ndarray`

The scattering length of every atom: the atomic number, or the neutron value.

For X-rays this is the Q -> 0 limit of the form factor.  A real X-ray form factor falls with
Q, which is why X-ray pair distribution work divides by the average form factor; taking the
limit instead is the usual approximation and is stated here rather than hidden.

### `scattering_record(crystal, source: str='xray')`

The same structure carrying one field: how each of its atoms scatters.

Everything else -- the coordinates, the cell, the periodicity flags -- is shared with the
structure it came from, so this costs nothing but the weights.

### `pair_distribution(crystal, spec: CurveSpec=CurveSpec(), source: str='xray') -> Tuple[np.ndarray, np.ndarray]`

`(r, G(r))`: the reduced pair distribution function of one structure.

`G(r) = 4 pi r rho_0 [g(r) - 1]` in the usual convention, computed as the one-field
periodic autocorrelation divided by `r`, by the mean scattering length squared and by the
integral of the channel's Gaussian (see the module docstring).

### `structure_function(crystal, spec: CurveSpec=CurveSpec(), source: str='xray', route: str='transform') -> Tuple[np.ndarray, np.ndarray]`

`(Q, F(Q))` with `F(Q) = Q [S(Q) - 1]`: the reduced total structure function.

`S(Q)` is the Faber-Ziman total: the *distinct* scattering, with the term an atom gives
with itself taken out, so that `S` approaches 1 at large Q and `F` approaches zero.
Leaving that term in would add `(<b^2>/<b>^2 - 1) Q`, a straight line growing with Q, to
every curve of a structure with more than one element.

Two routes, because they are the same quantity and not the same computation:

* `"transform"` (the default) is the sine transform of the computed `G(r)`, which is the
  operational definition and the one that matches the `G` this module returns, truncation
  at `r_max` included -- exactly as a measured `F` matches the measured `G` it was
  made from;
* `"reciprocal"` sums over the reciprocal lattice (`reciprocal.py`).  That is what the
  structure factor *is*, and for an infinite crystal it is a set of sharp reflections, which
  a channel grid of finite width samples rather than resolves.  It is the right route for a
  cell whose reflections are dense, and the wrong one for a small cell with a few sharp ones.

### `pair_distribution_residual(crystal, spec: CurveSpec=CurveSpec(), source: str='xray') -> dict`

How far the descriptor route is from the definition of `G(r)`.

The definition is a sum over the neighbour list, smeared by a normalised Gaussian; the
descriptor route is the one-field autocorrelation.  They are the same in the limit of a
narrow channel, so the residual says how narrow is narrow enough.

### `sine_transform(q: np.ndarray, f: np.ndarray, r: np.ndarray) -> np.ndarray`

`G(r) = (2/pi) INT F(Q) sin(Qr) dQ`, by the trapezium rule on the given grid.

### `inverse_sine_transform(r: np.ndarray, g: np.ndarray, q: np.ndarray) -> np.ndarray`

`F(Q) = INT G(r) sin(Qr) dr`, the other direction of the same transform.

### `measured_like(spec: CurveSpec, r: np.ndarray, g: np.ndarray, q_max: Optional[float]=None, noise: float=0.0, density_error: float=0.0, density: float=0.0, seed: int=0) -> np.ndarray`

A computed `G(r)` put through what a real measurement does to it.

`q_max` transforms the curve to `F(Q)`, cuts it there and transforms back, which is what
an instrument of that range would have reported, ripples and all.  `noise` adds Gaussian
noise to `F(Q)` with that standard deviation, in units of the root mean square of `F`.
`density_error` is the relative error in the number density used to draw the baseline, and
adds `-4 pi r rho_0 * density_error` to the curve; it needs `density`, the number of
atoms per cubic angstrom of the structure.  Each is optional and they compose in that order.

### `read_curve(path: str) -> dict`

A measured curve from the two-column text the common programs write.

Returns `{"x", "y", "path", "header"}` and whatever of `HEADER_KEYS` the comments
carried.  Lines beginning with `#` are comments, a line of two or more numbers is data, and
columns after the second (the uncertainties) are ignored.  The file says in its own units
what `x` is: `r` in angstrom for a `.gr`, `Q` in inverse angstrom for a `.fq` or
`.sq`; this reader does not guess, it only reads.

### `resample(curve: dict, grid: np.ndarray, fill: float=0.0) -> np.ndarray`

A measured curve put on the grid a model uses, with the range it does not cover filled.

A measurement that stops at 20 angstrom cannot say what happens at 30, and the fill value is
what the model is told instead; 0 is the right one for `G(r)`, which oscillates about zero.

### `computed_curves(crystals: Sequence, spec: CurveSpec=CurveSpec(), source: str='xray', progress=None) -> np.ndarray`

`G(r)` of every structure of a set, one row each, on the spec's grid.

### `curve_matrix(curves: Sequence[dict], grid: np.ndarray, fill: float=0.0) -> np.ndarray`

Measured curves on one grid, one row each: the feature matrix of a set of measurements.

### `curve_kernel(train: np.ndarray, test: Optional[np.ndarray]=None, normalize: bool=True) -> np.ndarray`

The cosine-normalised linear kernel of a set of curves, computed or measured.

It is `cmfqsar.invariant.autocorrelation_kernel` with one channel block, which is what
a total curve is.  Because the grid is the only thing the two sides share, a measured curve
and a computed one can be compared with it directly.

### `curve_labels(spec: CurveSpec) -> List[str]`

One label per column of a curve matrix: the distance it reports.

## cmfqsar.screen

Source: `cmfqsar/screen.py`

Screening a library without holding it in memory (plan item B3).

Predicting with a kernel method needs only the rows `K(X, X_train)` of the kernel: one row per
new molecule, as wide as the training set.  Nothing about a library of 100 000 molecules has to
be in memory at once, and nothing about it has to be kept after its score has been written.  The
rest of this program loads a data set, builds every molecule's fields and keeps them, which is
right for a training set of 200 molecules and impossible for a library.

This module reads the library in blocks and never holds more than one:

1. read `chunk` molecules from the file (a forward supplier, so the file is not indexed);
2. build their fields with the model's own field set, and pose them if asked;
3. compute the (chunk, training) kernel rows and score them;
4. write those rows out, keep the best `top` of them, and discard everything else.

The peak memory is therefore one block of fields plus one `chunk x n_train` kernel, whatever the
length of the file.  With the default block of 500 molecules against a 200-molecule training set
that is a 0.8 MB kernel.

What the score is
-----------------
For a regression model it is the prediction, or -- with `beta` above zero -- the lower confidence
bound `mean - beta * sd`, which is what `cmfqsar.model.CMFModel.rank_score` computes and
what a library should normally be ranked by: a library is mostly molecules unlike the training
set, a kernel method answers the training mean for those, and a point estimate cannot tell that
answer from a real one.  For a one-class or classification model it is the decision value.

What this does not do
---------------------
It does not generate poses.  A field kernel compares molecules in one frame, so the molecules in
the file must already be in the training set's frame, or `pose` must be set, which places each
one by maximising its field similarity to the training molecules
(`cmfqsar.decoys.align_decoys`) -- the same placement the background sets of the benchmarks
are given, and an optimistic one, since a real library has to choose a pose without knowing the
actives.  The measured cost is about half a second per molecule, which is the dominant cost of a
screening run and the reason posing is off by default.

### `class ScreenSpec  (dataclass)`

How a library is read and scored.

Attributes:

- `chunk: int` = `DEFAULT_CHUNK`
- `top: int` = `100`
- `beta: float` = `0.0`
- `larger_is_better: bool` = `True`
- `pose: bool` = `False`
- `restarts: int` = `4`
- `conformers: int` = `0`
- `cutoff: Optional[object]` = `None`
- `engine: Optional[str]` = `'auto'`
- `seed: int` = `0`

### `class ScreenReport  (dataclass)`

What a screening run did, and the best of what it found.

Attributes:

- `n_read: int` = `0`
- `n_scored: int` = `0`
- `engine: str` = `''`
- `n_failed: int` = `0`
- `blocks: int` = `0`
- `seconds: float` = `0.0`
- `kernel_seconds: float` = `0.0`
- `pose_seconds: float` = `0.0`
- `peak_kernel_mb: float` = `0.0`
- `out_path: Optional[str]` = `None`
- `best: List[Tuple[float, str, float, float]]` = `field(default_factory=list)`
- `failures: List[str]` = `field(default_factory=list)`

#### `ScreenReport.per_second` -> float  (property)

#### `ScreenReport.describe() -> str`

### `stream_molecules(path: str, chunk: int=DEFAULT_CHUNK, add_hs: bool=True) -> Iterator[List[Tuple[object, str]]]`

Blocks of (molecule, name) from an SD file or a SMILES file, without indexing it.

`Chem.ForwardSDMolSupplier` reads the file as a stream, so a 100 000-molecule SD file costs
one block of memory rather than all of it.  A SMILES file is read line by line and embedded,
which is slower per molecule and is there for libraries that arrive as SMILES.

### `screen(model, path: str, spec: Optional[ScreenSpec]=None, out: Optional[str]=None, progress: Optional[Callable[[int, float], None]]=None, cancel: Optional[Callable[[], bool]]=None) -> ScreenReport`

Score every molecule of `path` against `model`, one block at a time.

`out` is a CSV written as the run goes, so that a run interrupted after an hour has the scores
of everything it had read.  The report keeps the best `spec.top` molecules and nothing else.

## cmfqsar.shapley

Source: `cmfqsar/shapley.py`

Shapley decomposition of a CMF prediction.

The decomposition rules of the papers are exact but they need the model to be *linear* in the
molecular field: rule 3 splits a prediction into atoms because the field is a sum of atom-centred
Gaussians and the kernel is an overlap integral, hence additive over those atoms.  A nonlinear
kernel layer destroys that additivity, and with it the co-field and the atom contributions.

The Shapley value restores an additive attribution for any kernel.  Its usual obstacle in machine
learning is that "feature absent" has no definition, so a background sample and an imputation rule
have to be invented.  Continuous molecular fields do not have that problem: the field of a subset
of atoms is again a field, and because the kernel is bilinear its value for any subset is a sum of
entries of one precomputed array.  Every coalition value here is therefore the model's own
prediction for a genuine field, with no surrogate model and no background set:

    v(S) = sum_j a_j kappa( K(X_S, X_j), K(X_S, X_S), K(X_j, X_j) ) + b

with K(X_S, X_j) = sum_{l in S} blk[l, j] and K(X_S, X_S) = sum_{l, m in S} B[l, m].

Two games are provided.  In the *atom* game the players are atoms, or groups of atoms such as the
substituents of a common scaffold; the scaffold can be held fixed, so that the attribution explains
the difference from the reference compound rather than from an empty region of space.  In the
*field* game the players are the field types, which replaces the per-field overlap integral that
only a linear model can report.

For the linear kernel without normalisation the atom game is additive, so the Shapley value of an
atom equals its contribution under rule 3 exactly (`cmfqsar.model.CMFModel.atom_contributions`)
and all interaction indices vanish; the tests assert this.  What a nonlinear model needs beyond the
values themselves is the second-order Shapley interaction index (Grabisch & Roubens 1999), which
measures the part of the prediction that two groups produce only together.

References: Shapley (1953); Grabisch & Roubens (1999); Lundberg & Lee (2017) for the use of these
values as model explanations.

### `class AtomGame(model, mol: MoleculeFields, groups: Optional[Sequence[Sequence[int]]]=None, core: Sequence[int]=(), labels: Optional[Sequence[str]]=None)  — bases: _Game`

Players are atoms, or groups of atoms; the field of a coalition is the field of its atoms.

`core` holds atoms that belong to every coalition (the common scaffold), so the empty
coalition is the reference compound rather than empty space.

#### `AtomGame.atoms_of(masks: np.ndarray) -> np.ndarray`

Atom membership (m, N) of the coalitions in `masks` (m, P), including the core.

#### `AtomGame.value(masks: np.ndarray) -> np.ndarray`

### `class AtomFieldGame(model, mol: MoleculeFields, groups: Optional[Sequence[Sequence[int]]]=None, core: Sequence[int]=(), labels: Optional[Sequence[str]]=None)  — bases: _Game`

Players are (atom or group, field type) pairs: what each atom contributes to each field.

This is the game behind a *Shapley field*.  Its values can be mapped back onto the atoms of one
field type and smeared over space with the Gaussians of that field, giving a map per field
whose integral is the attributed part of the prediction, so that a nonlinear model gets one
picture per field type where a linear one has a co-field.

#### `AtomFieldGame.value(masks: np.ndarray) -> np.ndarray`

### `class FieldGame(model, mol: MoleculeFields)  — bases: _Game`

Players are the field types: what each field contributes to the prediction.

For a linear model this is the integral of the overlap of co-field and molecular field
(Eq. 7); for a nonlinear one no such integral exists and the Shapley value takes its place.

#### `FieldGame.value(masks: np.ndarray) -> np.ndarray`

### `class ShapleyResult  (dataclass)`

Shapley decomposition of one prediction.

Attributes:

- `kind: str`
- `labels: List[str]`
- `members: List[List[int]]`
- `values: np.ndarray`
- `baseline: float`
- `prediction: float`
- `exact: bool`
- `n_evaluations: int`
- `stderr: Optional[np.ndarray]` = `None`
- `interactions: Optional[np.ndarray]` = `None`
- `interactions_exact: bool` = `False`
- `player_fields: Optional[List[int]]` = `None`

#### `ShapleyResult.efficiency_error` -> float  (property)

How far the values are from adding up to prediction - baseline (zero if exact).

#### `ShapleyResult.atom_values(n_atoms: int) -> np.ndarray`

Spread the values of the players over the atoms they contain (atom game only).

#### `ShapleyResult.field_atom_values(n_atoms: int, n_fields: int) -> np.ndarray`

Values mapped back onto (field, atom), ready to be smeared into a Shapley field.

#### `ShapleyResult.table() -> List[Tuple[str, float, float]]`

#### `ShapleyResult.top_interactions(k: int=10) -> List[Tuple[str, str, float]]`

### `shapley(game: _Game, samples: int=400, seed: Optional[int]=0, exact_limit: int=14, interactions: bool=False, interaction_samples: int=64, interaction_limit: int=24, progress: Optional[ProgressFn]=None, cancel: Optional[CancelFn]=None) -> ShapleyResult`

Shapley values of a coalition game, exactly for few players and by sampling otherwise.

### `atom_shapley(model, mol: MoleculeFields, groups: Optional[Sequence[Sequence[int]]]=None, core: Sequence[int]=(), labels: Optional[Sequence[str]]=None, samples: int=400, seed: Optional[int]=0, exact_limit: int=14, interactions: bool=False, interaction_samples: int=64, progress: Optional[ProgressFn]=None, cancel: Optional[CancelFn]=None) -> ShapleyResult`

Decompose a prediction over atoms, or over groups of atoms.

`core` atoms are present in every coalition, so the baseline is the prediction for that
fragment alone; with the maximum common substructure as the core the values explain the
difference between this compound and the scaffold.

### `atom_field_shapley(model, mol: MoleculeFields, groups: Optional[Sequence[Sequence[int]]]=None, core: Sequence[int]=(), labels: Optional[Sequence[str]]=None, samples: int=400, seed: Optional[int]=0, exact_limit: int=14, interactions: bool=False, interaction_samples: int=64, progress: Optional[ProgressFn]=None, cancel: Optional[CancelFn]=None) -> ShapleyResult`

Decompose a prediction over (atom, field type) pairs: the basis of a Shapley field.

### `field_shapley(model, mol: MoleculeFields, interactions: bool=False, samples: int=400, seed: Optional[int]=0, exact_limit: int=14, interaction_samples: int=64, progress: Optional[ProgressFn]=None, cancel: Optional[CancelFn]=None) -> ShapleyResult`

Decompose a prediction over the field types.

### `common_core(mol, references: Sequence, timeout: int=10, ring_matches_ring_only: bool=True) -> List[int]`

Atoms of `mol` that belong to its maximum common substructure with `references`.

Hydrogens attached to a core atom are part of the core.  Returns atom indices in the order
of `mol`, which is the order of the fields built from it.

### `substituent_groups(mol, core: Sequence[int]) -> Tuple[List[List[int]], List[str]]`

Group the atoms outside the core into connected substituents, with readable labels.

### `scaffold_decomposition(model, mol: MoleculeFields, rdmol, references: Sequence, **kwargs) -> ShapleyResult`

Shapley values of the substituents of a congeneric series over their common scaffold.

## cmfqsar.shells

Source: `cmfqsar/shells.py`

Solvation shells cut from a bulk electrolyte trajectory (plan item BT5).

Item BT4 builds one cation complex and optimises it: a single geometry, the lowest the builder
could find. A real electrolyte is not that. At a given moment one lithium ion sits in a shell of
four carbonate molecules, the next in a shell of three and an anion, and a moment later they have
exchanged. The quantities an electrolyte is judged by -- conductivity, the transference number,
the diffusion coefficient -- are averages over that distribution, so a descriptor that is one
geometry may be describing the wrong object.

This module takes the other route. It reads a bulk simulation through `cmfqsar.trajectory`,
cuts out every cation with the molecules that touch it, and hands the collection to the ensemble
builder of plan item D4 (`cmfqsar.fields.combine_members`) as members with their own atoms.
The result is one field per electrolyte, the mean field of its solvation shells, which the model
then uses exactly as it uses the field of a molecule. Nothing in the kernel changes: a CMF field
is a sum of atom-centred Gaussians and the kernel is bilinear in the field, so the mean field of
a distribution of shells is itself a field.

What the three steps are, and why each is done this way.

**The cut.** A molecule belongs to the shell when one of its donor atoms -- an oxygen, nitrogen,
fluorine, sulfur or chlorine -- lies within the coordination distance of the ion. The distance is
the one item BT4 measured on relaxed geometries, `ION_OXYGEN_DISTANCE` times
`DEFAULT_SHELL_FACTOR`, so a shell here and a coordination number there mean the same thing.
An anion that comes that close is kept like any other molecule, because a contact ion pair is a
real state of the electrolyte and leaving it out would bias the ensemble towards free ions.

**The periodic boundary.** A simulation is written with every atom wrapped into the cell, so a
molecule that straddles a face arrives in two pieces a box length apart, and a shell beside a
cation at the edge of the cell arrives on the other side of the box. Both faults are silent: they
produce a geometry, not an error. Item D7 met them already and `cmfqsar.trajectory.make_whole`
and `cmfqsar.trajectory.reimage` are the repairs; this module calls them for every molecule
of every shell before anything is measured.

**The thinning.** A two-nanosecond box with fifteen cations and two hundred frames gives three
thousand shells, and the kernel costs the square of the atoms kept, so the ensemble has to be
reduced to a few dozen members. The reduction is done in two stages. Shells are grouped by
composition first -- which species, how many of each -- because the composition is what a model
of an electrolyte can act on, and the groups are then thinned to representatives in proportion to
their population. Inside a group the thinning uses a measure that does not depend on how the
molecules happen to be numbered: the sorted distances from the ion to its donors and the sorted
distances between donors. A field is a sum over atoms and so is already blind to that numbering,
which is why no superposition is attempted here.

**The error of the mean.** An ensemble read from a simulation carries sampling error, and
unconverged sampling error looks like chemistry. Item D8 requires that it be reported, so
`block_error` splits the frames into blocks, builds the mean field of each block on its own,
and returns the standard error of the mean over the blocks as a share of the field's own size. A
value near zero says the frames were enough; a large one says the difference between two
electrolytes may be sampling noise.

### `class ShellError  — bases: RuntimeError`

A shell could not be cut, or could not be turned into a molecule.

### `class Shell  (dataclass)`

One cation with the molecules that touch it, in one frame of one trajectory.

`atoms` indexes the source trajectory and starts with the ion. `coords` holds the same atoms
in the same order, repaired across the periodic boundary and moved to the image beside the
ion. `species` names the residue of each molecule, in the order the molecules appear after
the ion, and `sizes` says how many atoms each of them has.

Attributes:

- `frame: int`
- `ion_atom: int`
- `ion: str`
- `atoms: np.ndarray`
- `coords: np.ndarray`
- `species: Tuple[str, ...]` = `()`
- `sizes: Tuple[int, ...]` = `()`
- `donor_distances: Tuple[float, ...]` = `()`
- `note: str` = `''`

#### `Shell.n_molecules` -> int  (property)

#### `Shell.n_atoms` -> int  (property)

#### `Shell.composition` -> Tuple[Tuple[str, int], ...]  (property)

Which species are in the shell and how many of each, in a fixed order.

#### `Shell.coordination` -> int  (property)

Donor atoms inside the cutoff, which is the coordination number of item BT4.

#### `Shell.molecule_slice(i: int) -> slice`

Where molecule `i` sits in `atoms` and `coords`; the ion is atom 0.

#### `Shell.fingerprint() -> np.ndarray`

A description of the shape that does not depend on how the atoms are numbered.

The sorted ion-to-donor distances followed by the sorted donor-to-donor distances. Two
shells of the same composition whose fingerprints are close are the same arrangement,
whatever order their molecules were written in. A field is a sum over atoms and is blind
to that order in the same way, which is why this and not a superposition is what the
thinning uses.

### `class ShellEnsemble  (dataclass)`

The mean field of a set of shells, with what it was built from.

Attributes:

- `fields: MoleculeFields`
- `members: List[Chem.Mol]` = `_field(default_factory=list)`
- `chosen: List[Shell]` = `_field(default_factory=list)`
- `weights: np.ndarray` = `_field(default_factory=lambda: np.zeros(0))`
- `composition: List[Tuple[Tuple[Tuple[str, int], ...], float]]` = `_field(default_factory=list)`
- `mean_coordination: float` = `float('nan')`
- `n_shells: int` = `0`
- `note: str` = `''`

### `ion_atoms(traj: Trajectory, ion: str='Li') -> np.ndarray`

Indices of every atom of element `ion` in the trajectory.

### `residue_groups(traj: Trajectory) -> List[np.ndarray]`

The atoms of each residue, in the order the residues first appear.

A molecule in a bulk simulation is one residue, which is how every packer and every topology
writer produces it; the groups are what the shell cut moves and repairs as single pieces.

### `shell_cutoff(ion: str='Li', factor: float=DEFAULT_SHELL_FACTOR) -> float`

The shell cutoff in angstrom: the coordination distance of item BT4 times `factor`.

### `shell_at(traj: Trajectory, frame: int, ion_atom: int, groups: Optional[Sequence[np.ndarray]]=None, *, ion: str='Li', cutoff: Optional[float]=None, donor_elements: Sequence[str]=DONOR_ELEMENTS, max_molecules: int=0) -> Shell`

The shell of one ion in one frame, repaired across the periodic boundary.

`max_molecules` above zero keeps only the molecules whose nearest donor is closest to the
ion, which bounds the cost when a cutoff happens to catch an extra molecule.

### `extract_shells(traj: Trajectory, ion: str='Li', *, cutoff: Optional[float]=None, stride: int=1, donor_elements: Sequence[str]=DONOR_ELEMENTS, max_molecules: int=0, min_molecules: int=1, progress: Optional[Callable[[int, int], None]]=None) -> List[Shell]`

Every shell of every ion in every frame taken, as a flat list.

`stride` takes every n-th frame. A shell with fewer than `min_molecules` molecules is
dropped, which removes the few frames in which an ion sits in a gap of the liquid.

### `shell_molecule(shell: Shell, templates: Dict[str, Chem.Mol]) -> Chem.Mol`

One shell as an RDKit molecule, with the ion as atom 0.

A trajectory topology carries bonds but no bond orders, so the species are supplied as
templates: one molecule per residue name, with its hydrogens explicit and its atoms in the
order the simulation writes them. That is the order a packed box has, because the box was
packed from those same files. The template is checked element by element, so a mismatch is an
error here and not a wrong bond order downstream.

### `check_templates(traj: Trajectory, templates: Dict[str, Chem.Mol]) -> None`

Raise unless every residue of the trajectory matches its template element by element.

Called once per trajectory rather than once per shell, because the check is about the
topology and not about a frame.

### `composition_table(shells: Sequence[Shell]) -> List[Tuple[Tuple[Tuple[str, int], ...], float]]`

Each composition and the share of shells that have it, most common first.

### `cluster_shells(shells: Sequence[Shell], *, max_members: int=DEFAULT_MAX_MEMBERS, radius: float=DEFAULT_CLUSTER_RADIUS) -> Tuple[List[int], np.ndarray]`

Representative shells and their weights.

Shells are grouped by composition, the groups are given representatives in proportion to how
many shells they hold, and inside a group the representatives are leaders of a clustering on
`Shell.fingerprint`. The weights are populations: the share of all shells that the
representative stands for. They are never Boltzmann factors of a frame energy, for the reason
item D7 gives -- a trajectory has already sampled the distribution once, and weighting it
again squares it.

The first representative returned is the one from the composition with the most distinct
species, so that the atom map of the ensemble has as few gaps as possible.

### `atom_maps(reps: Sequence[Shell], templates: Dict[str, Chem.Mol]) -> List[np.ndarray]`

For each representative, the atom of the first representative each atom corresponds to.

The ion maps to the ion. An atom of a solvent or anion molecule maps to the same atom of the
*first* molecule of its own species in the first representative, so a per-atom reading of the
model is per atom of a species, summed over however many molecules of that species the shell
holds. Where the first representative has no molecule of that species the atoms map to -1,
which is how `cmfqsar.fields.combine_members` marks an atom with no counterpart.

### `shell_ensemble(shells: Sequence[Shell], templates: Dict[str, Chem.Mol], fieldset: FieldSet, *, max_members: int=DEFAULT_MAX_MEMBERS, radius: float=DEFAULT_CLUSTER_RADIUS, name: str='', frame: bool=True) -> ShellEnsemble`

The mean field of a collection of shells, as one `MoleculeFields`.

With `frame` set, every member is put in the cation frame of item BT4 first: the origin on
the ion and the axes on the donor polyhedron. The aligned kernel needs that; the
alignment-free kernel does not care, and the frame does not change what it reports.

### `member_properties(shell: Shell, weight: float) -> Dict[str, str]`

What has to be written beside a member so that the ensemble can be rebuilt.

A trajectory is too large to keep in a repository and the members are not conformations of
one molecule, so the usual multi-conformer file cannot hold them: each member has its own
atoms. These three properties are what `ensemble_from_molecules` needs instead.

### `ensemble_from_molecules(mols: Sequence[Chem.Mol], fieldset: FieldSet, name: str='') -> MoleculeFields`

The mean field of members read back from a file, in the order the file has them.

Each molecule carries the three properties of `member_properties`. The first molecule
is the one every other member's atoms are mapped onto, which is why the writer puts the
member with the most species first.

### `convergence(shells: Sequence[Shell], templates: Dict[str, Chem.Mol], fieldset: FieldSet, *, n_blocks: int=5, sample: int=200, max_members: int=DEFAULT_MAX_MEMBERS, radius: float=DEFAULT_CLUSTER_RADIUS) -> Dict[str, float]`

How well the mean field is sampled, and what the thinning to members costs.

Two numbers, both as a share of the field's own size, and both exact rather than measured on
a thinned ensemble:

`block_error`
    The frames are split into `n_blocks` consecutive blocks, the same number of shells is
    taken from each, and the number returned is the standard error of the mean over the
    blocks of the distance from a block's mean field to the mean field of all of them. Item
    D8 requires this beside any accuracy taken from a simulation: an unconverged trajectory
    gives a mean field whose sampling error would otherwise read as chemistry.

`thinning_error`
    The distance from the thinned ensemble of `max_members` representatives to the mean field
    of every sampled shell. This is the price of the thinning, which the kernel cost makes
    necessary, and it has to be separated from the sampling error above: thinning more
    finely cannot fix an unconverged trajectory, and a longer trajectory cannot fix a
    thinning that is too coarse.

`sample` is the total number of shells the overlap matrix is built on. The cost is quadratic
in it, so it is a setting rather than a constant.

### `block_error(shells: Sequence[Shell], templates: Dict[str, Chem.Mol], fieldset: FieldSet, *, n_blocks: int=5, sample: int=200, max_members: int=DEFAULT_MAX_MEMBERS, radius: float=DEFAULT_CLUSTER_RADIUS) -> float`

The block-averaged error of the mean field alone; see `convergence`.

## cmfqsar.sites

Source: `cmfqsar/sites.py`

Site-resolved models of a periodic structure (plan item M5).

The periodic autocorrelation of `cmfqsar.periodic` is a sum over ordered pairs of atoms of
one cell, so it splits exactly into one contribution per **site**: the term of a pair belongs to
the atom that stands first in it.  That one observation gives two things the whole-cell descriptor
cannot.

**A sample that is a site.**  The row of the split that belongs to site *l* is the typed radial
profile of the cell as seen *from that site*: for every pair of field types (f, g) and every
distance channel d_k,

    s_l(f, g, d_k) = w_lf * sum_m w_mg G_k(r_lm) + w_lg * sum_m w_mf G_k(r_lm)   (f != g)

with the sum over the atoms of the cell and their periodic images inside the cutoff, the self
term where the spec keeps it, and the uniform background of the site removed.  The pair term is
local: only the neighbours of site *l* inside `r_max` enter it.  The uniform background is not, because it is
the mean field of the whole cell, so two cells of different composition give different
backgrounds at the same site; a profile that depends on nothing beyond the cutoff is taken with
`background=False`.  The profile needs no alignment,
because it is built from distances alone, and it is the periodic counterpart of the atom-centred
environment of `cmfqsar.localenv` (plan item LP2), with two differences: the images of the
cell are summed in, so nothing has to be cut out of the solid, and the profile is one-centred
(pairs from the site to its neighbours) instead of all-pairs inside an envelope.

A site record is an ordinary structure as far as the rest of the program is concerned:
`site_fields` returns a `cmfqsar.periodic.CrystalFields` whose `centre` is the
site, and `cmfqsar.periodic.periodic_autocorrelation` of such a record is that site's row.
So `cmfqsar.model.CMFModel` with `kernel_source="periodic"` fits a set of sites exactly
as it fits a set of cells, and the targets of that shape are the ones
`cmfqsar.atomtargets` already holds (plan item LP1), down to the folds, which hold out whole
**structures**: two sites of one cell, one in the training set and one in the test set, would let
the model see the cell it is being tested on.

**A local reading of a whole-cell model.**  For a cell-level model on the same kernel the split is
exact in the other direction: the cell descriptor is the mean of its site rows, the linear layer
is linear, so the prediction minus the bias is the sum of one number per site
(`site_contributions`).  That is the local co-pattern the autocorrelation kernel could not
give: not only which pair of field types at which distance raises the property, but *which site*
of the cell carries it.  It is the same decomposition rule as the atom contributions of the
overlap kernel (rule 3 of the plan), and it is exact to machine precision rather than attributed.

**The reference under rule 1.**  The simplest thing that could predict a per-site property is a
count: which element stands at the site, and how many atoms of each element lie in each distance
shell around it.  `shell_count_features` builds exactly that from the same cells, the same
sites and the same images, so that the only difference between the two numbers is the descriptor.

**What is not here.**  The height channels of a layer (plan item M1, `z_max > 0`) have no site
split in this version: `site_autocorrelation` refuses a spec that carries them, because the
descriptor of a site would then need the two height axes of the pair and the averaging over the
two faces, which is a second code path and not yet measured.  A site model of a slab therefore
uses the plain distance profile, which does not tell a neighbour above the surface from one below
it at the same distance.

### `site_autocorrelation(crystal: CrystalFields, spec: PeriodicSpec=SITE_SPEC, pairs: Optional[Sequence[Tuple[int, int]]]=None) -> np.ndarray`

The periodic autocorrelation of one cell split over its sites: `(n_atoms, width)`.

Row *l* is the profile of site *l* described above, in the layout of
`cmfqsar.periodic.periodic_autocorrelation`, and the rows are **not** divided by the
number of atoms, so that a site profile does not depend on how large a cell the site was
written in.  The mean of the rows is the descriptor of the cell:
`site_autocorrelation(c, spec).mean(0) == periodic_autocorrelation(c, spec)` for an ordered
cell, and the sum over the rows divided by the expected atom count for a disordered one.

### `site_width(spec: PeriodicSpec=SITE_SPEC, n_fields: int=1) -> int`

How many channels one site profile carries.

### `site_fields(crystal: CrystalFields, site: int, name: str='') -> CrystalFields`

The cell as one sample centred on `site`: a record whose `centre` is that site.

The arrays are shared with `crystal`, not copied; a site record is a view of the cell with a
centre, and `cmfqsar.periodic.periodic_autocorrelation` of it is that site's profile.

### `site_structures(crystals: Sequence[CrystalFields], targets: AtomTargets, names: Optional[Sequence[str]]=None) -> List[CrystalFields]`

One site record per measured sample of `targets`, in the order of its rows.

### `site_table(crystals: Sequence[CrystalFields], targets: AtomTargets) -> AtomTable`

An `cmfqsar.fields.AtomTable` whose structures are the measured sites, which
`cmfqsar.model.CMFModel` with `kernel_source="periodic"` takes like any other.

### `site_names(crystals: Sequence[CrystalFields], targets: AtomTargets) -> List[str]`

`name#index element` of every sample: what a table of predictions is read by.

### `site_features(crystals: Sequence[CrystalFields], targets: AtomTargets, spec: PeriodicSpec=SITE_SPEC, progress=None) -> np.ndarray`

The site profiles of every measured sample, one row each.

The split of one cell gives every one of its sites at once, so each cell is featurised once
however many of its sites were measured.

### `site_kernel(train: np.ndarray, test: Optional[np.ndarray]=None, n_bins: int=1, blocks: Optional[Sequence[int]]=None) -> np.ndarray`

The cosine-normalised linear kernel of the site profiles: the same kernel the whole-cell
autocorrelation uses, so that only the descriptor differs between the two.

### `site_targets(values: Sequence[Sequence[float]], name: str='') -> AtomTargets`

The target index of a per-site property: one value per site of every cell, not a number
where the site was not measured.

### `one_site_targets(sites: Sequence[int], values: Sequence[float], n_sites: Sequence[int], name: str='') -> AtomTargets`

The target index of a set where one site of every cell is measured: the site of each cell,
its value, and how many sites that cell has.

### `elements_of(crystals: Sequence[CrystalFields], min_sites: int=1) -> List[str]`

The elements that appear in the set, in alphabetical order.

### `shell_counts(crystal: CrystalFields, site: int, elements: Sequence[str], shells: Sequence[float]=SHELLS) -> np.ndarray`

`(1 + n_shells, n_elements)` flattened: the element of the site itself, then how many
atoms of every element lie in each distance shell around it, images included.

### `shell_count_features(crystals: Sequence[CrystalFields], targets: AtomTargets, elements: Optional[Sequence[str]]=None, shells: Sequence[float]=SHELLS) -> np.ndarray`

The count reference of every measured sample, one row each.

### `count_labels(elements: Sequence[str], shells: Sequence[float]=SHELLS) -> List[str]`

One label per column of the count reference.

### `cell_norm(crystal: CrystalFields) -> float`

What the cell descriptor divides its site rows by: the number of atoms, or the expected
number on a disordered cell (plan item M2).

### `site_contributions(model, crystal: CrystalFields, spec: Optional[PeriodicSpec]=None) -> np.ndarray`

One number per site of `crystal`: what that site adds to the prediction of a fitted
whole-cell model on the periodic autocorrelation.

The sum of the numbers is the prediction minus the bias, exactly.  The kernel is the dot
product of the scaled, unit-normalised channels, the cell's channels are the mean of its
site profiles, and a linear layer is linear, so the decomposition needs nothing attributed:
it is rule 3 of the plan for the autocorrelation kernel.  Defined for the linear layer only.

### `site_contribution_table(model, crystal: CrystalFields, n: int=10, spec: Optional[PeriodicSpec]=None) -> List[Tuple[str, float]]`

The `n` sites that move the prediction most, as (`index element`, contribution).

### `site_top_channels(model, crystal: CrystalFields, site: int, labels: Sequence[str], n: int=10, spec: Optional[PeriodicSpec]=None) -> List[Tuple[str, float]]`

The `n` channels of one site's profile that move the prediction most, as (label, amount).

The amounts sum to the contribution of that site, so this says which pair of field types at
which distance *from this site* raises the property and which lowers it.

## cmfqsar.slab

Source: `cmfqsar/slab.py`

A metal slab as the receptor (plan item EC6).

`cmfqsar.protein` reads a protein and `cmfqsar.complementarity` scores a molecule
against it, residue by residue.  An electrode is the same problem with a different receptor: a
slab, periodic in the two directions of the surface, with no residues and no side chains.  This
module supplies the three pieces that item EC6 asks for.

**The receptor.**  `slab_receptor` turns a slab into a
`cmfqsar.fields.MoleculeFields`, which is what everything downstream already takes: the
atoms of its top layers, together with their periodic images out to a cutoff, so that a molecule
near the edge of the cell sees the surface continue rather than end.  A slab is read from a POSCAR
or CIF with `read_slab`, or built with `bcc110_slab`.

**What a clean metal surface actually offers, which is less than a protein does.**  The five
physicochemical fields of this program describe an organic molecule.  On a one-element metal slab
three of them are identically zero and the reason is not a gap in the parameters:

* the **electrostatic** field is zero because every atom of a one-element cell has the same
  electronegativity, so every charge model this program has gives it zero;
* the **hydrophobic** field and both **hydrogen-bond** fields are zero because the published
  increments are fitted to organic fragments and a metal carries none (plan item EC4 made that
  explicit rather than leaving a default in place);
* the **steric** field is real, from the van der Waals parameters item EC4 added.

Of the four complementary pairs of `cmfqsar.complementarity`, therefore, *all four* are
identically zero against a clean metal slab, and what is left is burial — the steric overlap.  So
a complementarity score of a molecule against a metal surface, built from the fields the program
has, is a shape score and nothing more.  That is stated here rather than discovered later.

**What a metal offers instead: its image charge.**  A conductor responds to a molecule's own
charges, and that response is the physics the fields miss.  `image_charge_energy` computes
it in closed form: the interaction of the charge distribution with its mirror image through the
surface plane, `-(1/2) sum_ij q_i q_j / |r_i - r_j*|`, which for one charge at height h is the
textbook `-q^2 / 4h`.  It needs no parameters beyond the plane, and it is the term that makes a
polar molecule prefer a metal.

One property of it is worth stating because it decides what a pose search can do: the image
energy of a charge distribution above an infinite plane is **invariant to translation in the plane
and to rotation about the normal**.  It can set a molecule's height and its tilt and it cannot say
where on the lattice the molecule sits.  Only the shape term can do that.

**The pose.**  `dock_on_surface` optimises the four degrees of freedom a surface leaves —
height, tilt, rotation about the normal, translation in the plane — instead of the six of a free
rigid body, with the tilt bounded so that a molecule that arrives lying flat is not turned over.
Item EC2 found that the face a planar molecule presents is a mirror no rule and no reference
decides; here it is a choice the caller makes with `both_faces`.

**The sites.**  `surface_sites` lists the top, bridge and hollow sites of the top layer and
`site_profile` reports how much of a molecule sits over each kind, which is the surface's
answer to a residue profile.

### `class SurfaceDock  (dataclass)`

What happened to one pose on the surface.

Attributes:

- `score_before: float`
- `score_after: float`
- `burial_before: float`
- `burial_after: float`
- `image_before: float`
- `image_after: float`
- `clash_after: float`
- `height_before: float`
- `height_after: float`
- `moved: float`
- `tilt_deg: float`
- `turn_deg: float`
- `flipped: bool` = `False`
- `starts_used: int` = `0`

### `class SurfaceScorer(mol: MoleculeFields, slab: MoleculeFields, fieldset: FieldSet, image_weight: float=0.0, clash_weight: float=1.0, clash_tolerance: float=0.4, plane_z: float=0.0, cutoff: float=12.0)`

Scores a pose against a slab: burial, the image charge, and a clash penalty.

`image_weight` is how many units of burial one electron volt of image attraction is worth.
It defaults to zero, so that the pose a search finds is decided by shape alone: the image term
is invariant to translation in the plane and to rotation about the normal, so it cannot place
a molecule laterally, and letting it set the height while the shape sets the position mixes
two things that are better measured apart.  The term is always reported.

#### `SurfaceScorer.burial(coords: np.ndarray) -> float`

The steric overlap of the molecule with the surface, as a cosine in [0, 1].

#### `SurfaceScorer.image(coords: np.ndarray) -> float`

#### `SurfaceScorer.clash(coords: np.ndarray) -> float`

#### `SurfaceScorer.score(coords: np.ndarray) -> float`

### `class SurfaceScore  (dataclass)`

What a molecule's placement on a surface is worth, term by term.

Attributes:

- `name: str`
- `burial: float` = `float('nan')`
- `image_eV: float` = `float('nan')`
- `clash: float` = `0.0`
- `height: float` = `float('nan')`
- `sites: Dict[str, float]` = `_field(default_factory=dict)`

#### `SurfaceScore.describe() -> str`

### `bcc110_slab(element: str='Fe', nx: int=6, ny: int=6, layers: int=4, a: float=2.78, vacuum: float=15.0)`

A body-centred cubic (110) slab, in the frame this program uses for a surface.

The defaults are the slab the Fe(110) reference poses of `examples/corrosion_fe110` were
computed on: `a = 2.78` angstrom, the minimum of the UMA oc20 head for bcc iron rather than
the experimental 2.87, a 6 x 6 orthogonal surface cell and four layers.  The slab is shifted
so that its **top layer sits at z = 0**, which is the frame those poses are given in.

Needs ASE.

### `read_slab(path: str, top_at_zero: bool=True)`

A slab read from any file ASE understands (POSCAR, CIF, extended XYZ).

With `top_at_zero` the structure is shifted along z so that its highest atom is at z = 0,
which is the surface frame of this program: the normal along z and the top layer at the
origin.

### `layer_heights(atoms, tolerance: float=0.5) -> List[float]`

The distinct layer heights of a slab, highest first.

Two atoms are in the same layer when their heights differ by less than `tolerance`, which is
far below a layer spacing (1.97 angstrom for bcc Fe(110) at a = 2.78) and far above the
numerical spread of a flat layer.

### `top_layer_indices(atoms, top_layers: int=2, tolerance: float=0.5) -> np.ndarray`

The atoms of the top `top_layers` layers of a slab.

### `slab_images(atoms, top_layers: int=2, cutoff: float=10.0, tolerance: float=0.5) -> Tuple[np.ndarray, List[str]]`

The top layers of a slab repeated over the surface, as plain coordinates and symbols.

Returns `(coords, symbols)`.  Only the two surface directions are repeated, far enough
that every point of the primary surface cell has the surface continuing for at least
`cutoff` angstrom around it; a slab is not periodic along its normal, and its images along
the normal are vacuum.

This is the expansion `slab_receptor` turns into fields and the one the application
draws, so the picture shows exactly the atoms the receptor is made of.

### `slab_receptor(atoms, fieldset: FieldSet, top_layers: int=2, cutoff: float=10.0, tolerance: float=0.5, name: str='slab') -> MoleculeFields`

The top layers of a slab, with their periodic images, as fields.

`cutoff` is how far beyond the primary surface cell the images are carried, in angstrom: a
molecule anywhere in the cell then sees the surface continue for at least that far in every
direction.  Only the two surface directions are repeated; a slab is not periodic along its
normal and its images along z are vacuum.

The weights are not perceived from a molecule, because there is no molecule: every field of
the field set is zero except the steric one, which carries the van der Waals well depth of
the element, and an element indicator field `cif:<symbol>`, which carries one.  The
docstring of this module says why the others are zero.

### `image_charge_energy(coords: np.ndarray, charges: np.ndarray, plane_z: float=0.0, min_height: float=MIN_IMAGE_HEIGHT) -> float`

The interaction of a charge distribution with its image in a conducting plane, in eV.

`E = -(1/2) sum_i sum_j q_i q_j / |r_i - r_j*|` with `r*` the mirror of `r` through the
plane.  The one-half is the linear-response factor of an induced charge, and the diagonal term
is a charge with its own image: for a single charge at height h it gives the textbook
`-q^2 / 4h`.  The result is negative (attractive) for any distribution, which is why a
conductor attracts a neutral polar molecule as well as an ion.

Heights are measured from `plane_z`, and a charge closer to it than `min_height` is held at
that height: the point-charge expression diverges at contact, which is an artefact.

### `surface_transform(coords: np.ndarray, p: Sequence[float], centre: Optional[np.ndarray]=None) -> np.ndarray`

Move a molecule by the four freedoms a surface leaves.

`p = (dx, dy, dz, psi, tilt, azimuth)`: a turn of `psi` about the surface normal, a tilt of
`tilt` about the in-plane axis at angle `azimuth`, and a translation.  The tilt and the turn
are applied about `centre`, which defaults to the centroid, so that a bounded tilt moves the
molecule's own orientation and not its position.

### `dock_on_surface(mol: MoleculeFields, slab: MoleculeFields, fieldset: FieldSet, starts: int=8, seed: int=0, max_iter: int=300, max_shift: float=3.0, max_tilt_deg: float=25.0, min_height: float=1.2, max_height: float=6.0, image_weight: float=0.0, clash_weight: float=1.0, plane_z: float=0.0, both_faces: bool=False) -> Tuple[np.ndarray, SurfaceDock]`

Place one molecule on the surface, using the four freedoms a surface leaves.

The search is over translation in the plane, height, rotation about the normal and a bounded
tilt.  Rotation about the normal is the one freedom with no good starting guess -- a flat
molecule may sit at any angle to the lattice -- so the starts are spread uniformly over it.
The height is bounded between `min_height` and `max_height`, measured as the lowest heavy
atom above the plane, which keeps the optimiser from either burying the molecule in the metal
or letting it drift into the vacuum where every term is flat.

`both_faces` tries the molecule the other way up as well and keeps the better score.  Item
EC2 measured that the face is a mirror no rule decides, so whether to search it is the
caller's decision and is reported in the result.

### `surface_sites(atoms, tolerance: float=0.5, height: float=0.0)`

The top, bridge and hollow sites of the top layer, as labels and positions.

A top site is above a surface atom, a bridge site above the midpoint of two neighbours, and a
hollow site above the centre of a triangle of them.  The neighbours and the triangles come
from a Delaunay triangulation of the top layer in the plane, so the three kinds are found for
any surface and not only for the one lattice this module can build.

### `site_profile(coords: np.ndarray, radii: np.ndarray, labels: Sequence[str], points: np.ndarray, alpha: float=1.0) -> Dict[str, float]`

How much of a molecule sits over each kind of surface site.

The molecule's steric field is evaluated at every site position and the values are summed by
kind and divided by the number of sites of that kind, so the three numbers are comparable
whatever the lattice offers.  This is the surface's answer to a residue profile: it says
whether a molecule sits over the metal atoms or between them.

### `surface_score(mol: MoleculeFields, slab: MoleculeFields, fieldset: FieldSet, plane_z: float=0.0, sites=None, alpha: float=1.0) -> SurfaceScore`

Score one placed molecule against the surface, term by term.

`sites` is the pair that `surface_sites` returns; when it is given, the score carries
the site profile as well.

## cmfqsar.solvation

Source: `cmfqsar/solvation.py`

A cation with its first solvation shell as one molecule, built, optimised and framed.

Plan item BT4.  For every property of an electrolyte -- the energy a lithium ion gains from a
solvent, what it costs to strip one molecule off again, how many molecules fit around the ion --
the object that carries the property is not the solvent molecule.  It is the complex: the ion with
the molecules that touch it.  Nothing in this program built one, and this module does.

Three steps, and each is separate so that each can be checked on its own.

**Placing.**  A donor atom is an oxygen or a nitrogen with a lone pair pointing somewhere, and
`donor_direction` says where: away from the bisector of an ether oxygen's two bonds, along
the C=O axis of a carbonyl, along the lone pair of a nitrile.  `place_shell` puts the ion at
the origin, chooses as many directions as there are molecules from the polyhedron of that
coordination number (linear, trigonal, tetrahedral), and for each molecule rotates it so that its
donor's lone pair points back at the ion and translates it so the donor sits at the coordination
distance.  Each molecule is then spun about its own ion-donor axis to the angle that leaves the
most room, which is a starting geometry and not a result: the optimiser decides the rest.

**Optimising.**  GFN2-xTB through `cmfqsar.xtb`, which caches per molecule, per charge, per
method and per starting geometry, so an interrupted run resumes.  ORCA through
`cmfqsar.orca` is the other engine and is far slower on a complex of fifty atoms.  The
energies this module returns are total energies in eV; a binding energy is a difference of three
of them and a desolvation energy a difference of three others, both written out in the functions
that compute them, because a sign error in either is invisible in the number.

**Framing.**  This is what the complex gives that the free solvent cannot.  A series of free
solvent molecules has no common frame -- a glyme has no scaffold to superimpose -- but a series of
complexes does: the ion is one atom in every one of them, and the polyhedron of its donors fixes
the orientation.  `cation_frame` puts the origin on the ion, the axes along the principal
axes of the donor set, and resolves the four proper sign choices by the third moments of the
donors.  With that frame the overlap kernel applies, and with the overlap kernel there is a
co-field: a map in space of where binding is gained, which for a solvation complex is a statement
about the donor oxygens.

**What the module does not do.**  It does not search conformations.  A glyme wraps a lithium ion
with four or five of its own oxygens, and `wrap_ion` offers the intramolecular case -- the
ion placed at the centroid of the donors that already lie closest together, then optimised -- but
the fold it finds is the fold the starting conformer was near. `benchmarks/run_solvation.py`
measures how far that gets against 1,050 complexes relaxed with density functional theory, which
is the honest way to state the limit of a builder.

### `class SolvationError  — bases: ValueError`

A complex that cannot be built or framed, with the reason in the message.

### `class Complex  (dataclass)`

A cation with its shell: the molecule, the ion's atom index and what went into it.

Attributes:

- `mol: Chem.Mol`
- `ion_index: int`
- `ion: str`
- `solvents: Tuple[str, ...]` = `()`
- `donors: Tuple[int, ...]` = `()`
- `charge: int` = `0`
- `energy_ev: float` = `float('nan')`
- `optimised: bool` = `False`
- `note: str` = `''`

#### `Complex.n_solvents` -> int  (property)

#### `Complex.coordination(cutoff: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray]`

### `class Engine  (dataclass)`

How the complexes are optimised: GFN2-xTB by default, ORCA on request.

Attributes:

- `program: str` = `'xtb'`
- `workdir: str` = `'solvation_work'`
- `opt: bool` = `True`
- `solvent: str` = `''`
- `accuracy: float` = `1.0`
- `timeout: Optional[float]` = `1800.0`
- `orca_keywords: str` = `'B3LYP D3BJ def2-SVP'`
- `energies: Dict[str, float]` = `field(default_factory=dict)`

#### `Engine.level() -> str`

### `donor_atoms(mol: Chem.Mol) -> List[Tuple[int, str]]`

`[(atom index, kind)]` of every atom that can point a lone pair at a cation.

An atom matched by more than one pattern keeps the first kind it matched, and the list is in
the order of the atom indices so that it does not depend on the order of the patterns.

### `donor_direction(mol: Chem.Mol, index: int, coords: Optional[np.ndarray]=None) -> np.ndarray`

The unit vector along which a cation should approach the donor `index`.

It points away from the atoms the donor is bonded to: the reverse of the sum of the bond
directions, which is the bisector for two bonds and the bond axis itself for one.  A donor
with no bonds at all -- a bare fluoride -- has no direction, and any direction will do, so
`+z` is returned.

### `best_donor(mol: Chem.Mol, charges: Optional[Sequence[float]]=None) -> int`

The donor a cation would choose: the most negative one, or the first carbonyl or ether.

`charges` are partial charges, one per atom, when the caller has them (from xTB, say).
Without them the order of `DONOR_SMARTS` decides, which puts a carbonyl oxygen before
an ether oxygen, as their basicity does.

### `place_shell(solvents: Sequence[Chem.Mol], ion: str='Li', distance: Optional[float]=None, seed: int=0, spins: int=12, anion: Optional[Chem.Mol]=None) -> Complex`

Put `solvents` (and an `anion`, for a contact ion pair) around one `ion`.

The ion goes at the origin, the molecules on the directions of the polyhedron for their
number, each turned so that its donor's lone pair points at the ion and each spun about the
ion-donor axis to the angle that leaves the largest smallest interatomic distance.  The result
is a starting geometry for an optimiser.

### `wrap_ion(solvent: Chem.Mol, ion: str='Li', n_donors: int=0, seed: int=0) -> Complex`

Put the ion inside one molecule: the intramolecular case of a glyme or a crown ether.

The ion is placed at the centroid of the `n_donors` donor atoms that lie closest together in
the conformer given, or of every donor when `n_donors` is 0, and then pushed out to the
coordination distance from the nearest of them if it landed too close.  No conformational
search is done: the fold this gives is the fold of the conformer it was handed.

### `ion_index(mol: Chem.Mol, ion: str='Li') -> int`

Where the cation is. Raises when there is not exactly one.

### `coordination(mol: Chem.Mol, index: int, ion: str='Li', cutoff: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray]`

`(atom indices, distances)` of the donor atoms inside `cutoff` of the ion.

The default cutoff is 1.3 times the coordination distance of the ion, which for lithium and
oxygen is 2.6 A -- the value the first shell of the relaxed complexes of `examples/ether/`
is counted with.

### `cation_frame(mol: Chem.Mol, index: Optional[int]=None, ion: str='Li', cutoff: Optional[float]=None) -> np.ndarray`

The coordinates of the complex in the frame its own coordination polyhedron defines.

The origin is the ion.  The axes are the principal axes of the donor atoms inside the cutoff,
ordered by their moment, smallest first, which makes the longest extent of the polyhedron the
x axis.  Four sign choices leave those axes and the handedness alone, and the one with the
largest third moments of the donors, taken in the order x, y, z, is chosen; a polyhedron
symmetric enough for all three to vanish is one whose fields are the same in either frame.
A single donor fixes only one direction: the x axis is put along it and the other two are any
perpendicular pair, which is stated in the return of `frame_quality`.

### `frame_quality(mol: Chem.Mol, index: Optional[int]=None, ion: str='Li', cutoff: Optional[float]=None) -> Dict[str, float]`

How well the polyhedron fixes the frame: the number of donors and the moment ratios.

Two equal moments leave a rotation free, and the ratios say how near that is.  A caller that
needs one frame per complex should refuse a complex whose two smallest moments agree to
within a per cent or so, or accept that its orientation about that axis is arbitrary.

### `with_frame(mol: Chem.Mol, index: Optional[int]=None, ion: str='Li', cutoff: Optional[float]=None) -> Chem.Mol`

A copy of the complex whose conformer is the one `cation_frame` gives.

### `optimise(mol: Chem.Mol, name: str, engine: Optional[Engine]=None, index: int=1) -> Tuple[Chem.Mol, float]`

Optimise one molecule or complex and return it with its total energy in eV.

The call is cached by `cmfqsar.xtb` on the molecule, its charge, the method and the
starting geometry, so running this again costs nothing and an interrupted series resumes.

### `ion_energy(ion: str='Li', engine: Optional[Engine]=None) -> float`

The total energy of the bare cation in eV, which every binding energy needs.

### `binding_energy(complex_ev: float, ion_ev: float, solvent_evs: Sequence[float]) -> float`

`E(complex) - E(ion) - sum E(solvent)`, in eV: negative when the complex is bound.

The solvent energies are those of the free molecules at their own optimised geometries, so
this is the binding energy with the deformation of the molecules included, which is how the
reference set of `examples/ether/` defines it as well.

### `best_wrapped_complex(solvent: Chem.Mol, name: str, ion: str='Li', n_donors: int=0, conformers: int=5, engine: Optional[Engine]=None, index: int=1) -> Tuple[Complex, float, List[float]]`

Wrap the ion in several conformers, optimise each, and keep the one of lowest energy.

`wrap_ion` does no conformational search, so the fold it finds is the fold of the
conformer it was handed, and for a chain that wraps a cation with four of its own oxygens
that is usually not enough. This is the smallest honest remedy: a few starting conformers,
each optimised, the lowest kept. Returns the complex, its energy in eV and every energy
tried, so that the spread over the starts can be reported rather than hidden.

### `desolvation_energy(complex_n_ev: float, complex_n_minus_one_ev: float, solvent_ev: float) -> float`

What it costs to take one molecule out of the shell, in eV, positive when it costs energy.

`E(n-1 complex) + E(free solvent) - E(n complex)`: the two products against the one
reactant, both at their own optimised geometries.

## cmfqsar.solvent

Source: `cmfqsar/solvent.py`

Solvent as a field: where a ligand's water sits, as a CMF field (plan item D9).

Every field this program has built so far is centred on the atoms of the molecule. The 2015
indicator-field formalism does not require that -- it accepts any continuous density -- and a
water occupancy map from a simulation is one. This module turns that map into a field of the
same class as every other, so a sixth kernel block carries it and a co-field over it says where
displacing a water raises the predicted activity and where keeping one does.

The chain, and the decision in each step.

1. **The map is an occupancy, not a count.** For every point of a grid around the ligand, the
   share of frames in which some water oxygen lies within `PROBE` of it. That is a number
   in [0, 1] with a plain meaning -- "a water sits here nine tenths of the time" -- and it needs
   no normalisation against bulk density to be read. Bulk water gives about 0.13 at a 1 angstrom
   probe, which is why `MIN_OCCUPANCY` defaults well above it.
2. **Every frame is put in the common frame first**, by the same superposition the trajectory
   reader uses: the binding-site backbone onto the one structure every run was built from. Maps
   of two ligands are then in one frame and their hydration sites are comparable, which is the
   whole point of making a field out of them.
3. **The water is re-imaged before anything else.** A periodic box writes every atom wrapped, so
   the water beside the site may be recorded on the far side of the cell; the image nearest the
   site is the one that is physically there. This is the same correction the trajectory reader
   makes for the ligand, and skipping it does not give a worse map, it gives an empty one.
4. **Hydration sites are the peaks of the map**, taken greedily: the highest point above
   `MIN_OCCUPANCY`, then everything within `MIN_DISTANCE` of it excluded, and again.
   A site is therefore a place a water occupies, not a cluster of water molecules, and two sites
   are never closer than one water across.
5. **The coefficients are fitted, not read off the peaks.** With the centres fixed at the sites
   and the width taken from the field set, the occupancy map is fitted by a non-negative least
   squares, so the field reproduces the map rather than the heights of its maxima. Non-negative
   because an occupancy cannot be negative, and a fitted field with negative water in it would
   be read as a place water avoids, which is a different statement.

What comes out of `water_field` is a `cmfqsar.fields.MoleculeFields` whose atoms
are hydration sites. It is used exactly as the variance field of `cmfqsar/fluctuation.py` is:
built per molecule, kept in a table of its own, turned into a kernel of its own, and mixed with
the kernel of the ordinary fields under one coefficient. It cannot share a table with the
molecule's own fields, because a table has one set of atoms and these atoms are not the
molecule's.

**What this does not claim.** That the water field helps. `benchmarks/run_solvent.py` measures
it against the five physico-chemical fields under rule 1, with the interval of rule 4, and
against two controls: sites placed at random in the same box, and the occupancy map of one
ligand used for every ligand. The first says whether the positions matter and the second whether
anything beyond the shared shape of the site does.

### `class SolventError  — bases: RuntimeError`

### `class Occupancy  (dataclass)`

A water occupancy map of one ligand, in the common frame.

Attributes:

- `points: np.ndarray`
- `values: np.ndarray`
- `frames: int`
- `name: str` = `''`
- `spacing: float` = `SPACING`
- `probe: float` = `PROBE`

#### `Occupancy.bulk` -> float  (property)

What a point in bulk water would score, for comparison with a peak.

The expected number of oxygens in the probe sphere at the density of liquid water,
0.0334 per cubic angstrom, capped at one because an occupancy cannot exceed it.

#### `Occupancy.describe() -> str`

### `water_oxygens(traj: Trajectory) -> np.ndarray`

The oxygen of every water molecule of the topology.

### `occupancy(topology: str, trajectory: str, *, reference: str, ligand: Optional[str]=None, spacing: float=SPACING, map_radius: float=MAP_RADIUS, probe: float=PROBE, stride: int=1, site_radius: float=8.0, engine: str='auto', name: str='') -> Occupancy`

The water occupancy map of one simulation, in the frame of `reference`.

`reference` is the structure every run of the series was built from; without it two ligands'
maps would be in two frames and could not be compared, which is the same requirement the
trajectory reader has.

### `hydration_sites(occ: Occupancy, min_occupancy: float=MIN_OCCUPANCY, min_distance: float=MIN_DISTANCE, max_sites: int=MAX_SITES) -> Tuple[np.ndarray, np.ndarray]`

(centres, occupancies) of the peaks of the map, strongest first.

Greedy: the highest point above `min_occupancy` becomes a site, every point within
`min_distance` of it is removed, and the next highest is taken. Two sites are therefore never
closer than one water across, and a site is a place a water occupies rather than a cluster of
molecules.

### `fit_coefficients(occ: Occupancy, centres: np.ndarray, alpha: float, max_points: int=20000, seed: int=0) -> np.ndarray`

Non-negative coefficients of Gaussians at `centres` that reproduce the occupancy map.

The Gaussians are the field's own, `exp(-alpha/2 |r - c|^2)`, so what comes out is a field
of the ordinary kind rather than a new sort of object. Non-negative because an occupancy
cannot be negative and a negative coefficient would be read as a place water avoids, which
the map does not say. The grid is subsampled when it is large, since a least squares over
sixty thousand points and forty columns says nothing a sample of twenty thousand does not.

### `solvent_fieldset(alpha: float=0.5, name: str='solvent') -> FieldSet`

The one-field set a water field is evaluated with.

A water field cannot share a table with the molecule's own fields, because a table carries
one set of atoms and a hydration site is not an atom of the molecule. It is its own block,
mixed in with a coefficient, exactly as the variance field of `cmfqsar/fluctuation.py` is.

### `water_field(centres: np.ndarray, coefficients: np.ndarray, name: str='', radius: float=WATER_RADIUS) -> MoleculeFields`

The hydration sites of one ligand as a field, one Gaussian per site.

### `coefficients_at(occ: 'Occupancy', centres: np.ndarray, alphas: Sequence[float]) -> dict`

{attenuation: coefficients} for one set of sites.

A coefficient is fitted against Gaussians of the field's own width, so it belongs to one
attenuation and a model at another needs its own. Storing them for every attenuation a
measurement will use keeps the map out of the comparison: the map is needed to fit, and the
fit is all a model needs.

### `field_for(topology: str, trajectory: str, *, reference: str, alpha: float=0.5, ligand: Optional[str]=None, name: str='', **kwargs) -> Tuple[MoleculeFields, Occupancy, np.ndarray]`

The whole chain for one ligand: (water field, its map, the site occupancies).

### `site_contributions(model, fields: Sequence[MoleculeFields], coefficients: Sequence[float]) -> List[np.ndarray]`

What the model asks of the water at each hydration site, per molecule.

The co-field of a linear kernel model is `sum_j a_j X_j`, so the weight the model puts on
one molecule's site is that molecule's dual coefficient times the site's own coefficient. A
positive number means the model wants **more** water there -- keeping that water raises the
prediction -- and a negative one means it wants less, which is where displacing a water is
worth trying.

## cmfqsar.spectral

Source: `cmfqsar/spectral.py`

Clustering a set by its fields rather than by its scaffolds.

Every grouped validation in this program splits the molecules by their Bemis-Murcko scaffold:
`chemotype_labels` groups them, `grouped_cv_predict` holds one group out at a time, and the
benchmarks report that figure as the reliable one.  The reasoning is sound -- random folds put a
molecule's own scaffold-mates in the training part and overstate the quality of a model on a
mixed series -- but it rests on
an assumption nobody checks: that the scaffold is what makes two molecules look alike *to the
model*.  The model does not see scaffolds.  It sees fields, and its notion of similarity is the
kernel.

Spectral clustering (Ng, Jordan & Weiss 2002) partitions that similarity directly.  The kernel is
a similarity matrix already, so there is nothing to build: form the normalised Laplacian, take the
eigenvectors of its smallest eigenvalues, normalise their rows to the unit sphere and run k-means
there.  Clusters that are not linearly separable in field space are separable in those
coordinates, which is the purpose of the construction and the reason plain k-means on the
fields would not do.

Two uses, and the second is the one that helps most:

* **as a grouping** -- hold out a cluster of the model's own similarity rather than a cluster of
  the chemist's, which is a harder and fairer test whenever the two disagree;
* **as a check on the usual grouping** -- the adjusted Rand index between the two partitions says
  how far the scaffold split and the field split agree.  Where it is high the scaffold grouping
  is doing what it claims.  Where it is low, one of the two is measuring something else, and
  which one matters: a leave-one-scaffold-out figure is only pessimistic if the scaffolds are
  what the model responds to.

How many clusters is decided by the **eigengap** when it is not given: the number of eigenvalues
of the normalised Laplacian below the largest jump, which is the standard heuristic and is
reported along with the gap so that a weak one can be seen for what it is.

### `class SpectralFit  (dataclass)`

The partition, the spectrum it came from, and how certain the number of clusters is.

Attributes:

- `labels: np.ndarray`
- `k: int`
- `eigenvalues: np.ndarray`
- `gap: float` = `float('nan')`
- `chosen_by_gap: bool` = `False`

#### `SpectralFit.sizes` -> Dict[int, int]  (property)

#### `SpectralFit.describe() -> str`

### `knn_laplacian(K: np.ndarray, neighbours: int=10, floor: float=0.0) -> np.ndarray`

The normalised Laplacian of the graph in which every molecule keeps its closest neighbours.

The same construction `spectral_clusters` uses, with two changes that are what a
smoothness penalty needs (plan item E10, manifold regularisation):

* the affinity is the **cosine** of the kernel, `K(i,j) / sqrt(K(i,i) K(j,j))`, so that
  "close" means similar in field space and not simply large.  A kernel of unnormalised
  fields grows with the size of a molecule, and without this the neighbours of every
  molecule would be the biggest molecules of the set;
* every row keeps only its `neighbours` largest affinities and the rest are set to zero, so
  the penalty is **local**.  A pair survives if either of the two molecules counts the other
  among its neighbours, which is the usual symmetric form of the k-nearest-neighbour graph.
  Without the truncation every molecule is a neighbour of every other and the penalty asks
  the prediction to be flat over the whole set instead of smooth along it.

`neighbours <= 0` or a number at or above n - 1 keeps the full graph.

### `eigengap(values: Sequence[float], k_max: int=10) -> Tuple[int, float]`

(number of clusters, the size of the gap) from the spectrum of the Laplacian.

### `spectral_clusters(K: np.ndarray, k: Optional[int]=None, k_max: int=10, seed: int=0, floor: float=0.0) -> SpectralFit`

Cluster the molecules of a kernel matrix by their field similarity.

With `k` given it is used; otherwise the eigengap chooses it, over at most `k_max` clusters.

### `adjusted_rand(a: Sequence, b: Sequence) -> float`

Agreement of two partitions, corrected for chance: 1 identical, 0 as good as random.

## cmfqsar.stereo

Source: `cmfqsar/stereo.py`

Handedness as a property of a centre, not of a shape.

C5 of the development plan added signed three-point channels to the field autocorrelation: the
triple product of three displacement vectors, summed over every triple of atoms in the molecule
and resolved over radial shells.  Measured, it does exactly what it claims about one molecule --
the channels of a molecule and its mirror image are exact negatives -- and nothing at all about
two molecules, which is what a model needs.  On twenty pairs the cosine of that block was -0.016
between molecules of the *same* handedness and +0.016 between molecules of opposite handedness,
a difference of 0.03 against a spread of 0.55: handedness was not a direction in that space, so
a kernel method had nothing to transfer, and an activity that was pure handedness could not be
learnt at any setting.

The failure is in the summation, not in the triple product.  A sum over every triple of atoms is
a property of the whole shape, and two molecules of the same configuration have no reason to
share one.  The configuration itself is local: it is the order in which three substituents stand
around a centre, which is what a chemist reads when assigning R or S, and what this module
measures.

For every atom with at least three neighbours, and for every field used as a priority:

1. order the neighbours by their weight in that field, largest first;
2. take the unit vectors from the centre to the first three;
3. the signed volume of that ordered frame is +1 or -1 in character and changes sign under
   reflection, is unchanged by any rotation or translation, and is the same number for two
   molecules whose substituents stand in the same order around a comparable centre.

Point 3 is the whole difference from C5, and the tests assert it directly.

Using the field weights as the priority rather than the CIP rules keeps the descriptor a
property of the fields -- the same quantities the rest of the program is built from -- and not of
a toolkit's implementation of a convention.  The cost is that two substituents of nearly equal
weight have no reliable order, and an unreliable order is an unreliable sign; such centres are
dropped rather than guessed at, and `stereo_matrix` reports how many survived.

### `class StereoSpec  (dataclass)`

How a centre is read.

`min_separation` is the margin, in units of the field's own spread over the molecule, that
two neighbours' weights must differ by before their order is believed.  It matters more than
it seems: at zero every atom with three neighbours contributes a sign, including the ones
whose substituents are the same and whose order is therefore noise, and the handedness of two
molecules separates by 0.18 against a spread of 0.53.  At 0.1 to 0.3 the tied centres drop
out and the separation is 0.86 against 0.70.  Above 0.6 so few centres survive that it falls
away again, so this is a real parameter and not a formality.

Attributes:

- `min_neighbours: int` = `3`
- `min_separation: float` = `0.2`
- `include_hydrogens: bool` = `True`

### `stereo_centres(mol, spec: StereoSpec=StereoSpec()) -> List[Tuple[int, np.ndarray]]`

(atom, its neighbours) for every atom that has enough of them to define a frame.

### `stereo_channels(mol, spec: StereoSpec=StereoSpec(), n_fields: Optional[int]=None) -> np.ndarray`

One number per (priority field, weighting field): the molecule's signed centres.

The priority field decides the order of the substituents around a centre; the weighting
field says how much that centre counts, through the centre's own weight in it.  A molecule
with no usable centre returns zeros, which is the right answer: it has no handedness that
these fields can see.

### `channel_labels(fieldset, spec: StereoSpec=StereoSpec()) -> List[str]`

### `stereo_matrix(mols: Sequence, spec: StereoSpec=StereoSpec(), n_fields: Optional[int]=None) -> Tuple[np.ndarray, dict]`

The channels of a set of molecules, and how many centres they were read from.

## cmfqsar.surrogate

Source: `cmfqsar/surrogate.py`

The model as the fast level of a hierarchy: a standard deviation in the units of the activity,
its calibration on the folds, and the rule that sends a structure back to the fine level (plan
item MS1).

A multiscale scheme uses a fitted CMF model between an electronic-structure method and a coarser
model: the fine level labels a few hundred structures, the model answers for the thousands the
coarse model needs.  For that the model has to say, per structure, whether its answer can be
used.  The posterior variance of `CMFModel.predict_with_variance` is the right quantity,
but it is not usable as it stands, for three reasons this module removes.

* **It is in the units of the kernel.**  The Gaussian process that matches the ridge fit has the
  prior covariance `s K` and the noise `s lambda`, where `s` is the one amplitude the
  evidence profiles out in closed form (`quad / dof`, the mean squared whitened residual of the
  training set).  The predictive variance of a new *measurement* is therefore `s (v + lambda)`,
  with `v` the posterior variance in kernel units, and its square root is in the units of
  `y`.  The constant mean that the ridge fits as its bias is marginalised under a flat prior,
  which adds `(1 - k^T A^{-1} 1)^2 / (1^T A^{-1} 1)` to `v` (Rasmussen and Williams 2006,
  eq. 2.42 with one basis function); that term is what makes a structure far from every training
  one uncertain about the *level* of the activity and not only about its shape.

* **Nobody has checked that it means what it says.**  On the folds, the standardised residual
  `z = (y - mean) / sd` should have a root mean square of one, and the interval of two standard
  deviations should hold about 95 % of the activities.  `Calibration` reports both and the
  factor that would make the root mean square one.  A factor well above one says the model is
  surer than it should be, which is the usual finding when the kernel misses a part of the
  chemistry; the factor is then the honest correction, and it is reported, not applied.

* **A number is not a decision.**  A tolerance in the units of the activity turns it into one:
  `defer` marks every structure whose standard deviation exceeds the tolerance, and
  `Handoff` reports what the rule did, with the error on each side when the activities
  are known.  `tolerance_for_budget` goes the other way, from the share of structures the
  fine level can afford to the tolerance that sends it that many.

The arithmetic is one Cholesky factorisation of the training kernel per fit and one triangular
solve per structure, the same cost as the variance already paid.

### `class Surrogate  (dataclass)`

The Gaussian-process reading of a ridge fit, with its scale in the units of the activity.

`scale` is `s`: the prior covariance is `s K` and the noise is `s lambda`, so a
standard deviation this object returns is in the units of `y`.  The factor of the training
kernel is kept, so that every later block of structures costs one triangular solve.

Attributes:

- `lam: float`
- `fit_bias: bool`
- `scale: float`
- `dof: int`
- `n_train: int`
- `L: np.ndarray`
- `alpha: np.ndarray`
- `bias: float`
- `u: Optional[np.ndarray]`
- `denom: float`

#### `Surrogate.fit(K: np.ndarray, y: Sequence[float], lam: float, fit_bias: bool=True) -> 'Surrogate'`  (class method)

Fit on the training kernel `K` and the activities `y` with the ridge `lam`.

The dual coefficients and the bias are those of `cmfqsar.model.krr_fit`: the
bordered system it solves is the Gaussian process with the constant mean marginalised,
so the predictive mean here is the model's own prediction.

#### `Surrogate.predict(K_test_train: np.ndarray, k_test_self: Sequence[float], with_noise: bool=True) -> Tuple[np.ndarray, np.ndarray]`

(prediction, standard deviation) of new structures, both in the units of the activity.

`K_test_train` holds the kernels of the new structures against the training ones and
`k_test_self` their self-kernels.  With `with_noise` the deviation is that of a new
*measurement*, which includes the noise `s lambda`; without it, that of the underlying
value, which is what two predictions of the same structure share.

#### `Surrogate.noise_sd` -> float  (property)

The standard deviation of the noise the fit attributes to one measurement.

### `class Calibration  (dataclass)`

Does the standard deviation mean what it says?  Measured on activities the fit did not see.

`z_rms` is the root mean square of `(y - mean) / sd`: one for a calibrated deviation,
above one when the model is surer than it should be.  `coverage_1sd` and `coverage_2sd`
are the shares of the activities inside one and two standard deviations of the prediction,
68 % and 95 % for a calibrated Gaussian.  `spearman` is the rank correlation between the
deviation and the absolute error, which says whether the deviation at least *orders* the
structures by their error when its size is off.

Attributes:

- `n: int`
- `rmse: float`
- `mean_sd: float`
- `z_rms: float`
- `coverage_1sd: float`
- `coverage_2sd: float`
- `spearman: float`

#### `Calibration.of(y: Sequence[float], mean: Sequence[float], sd: Sequence[float]) -> 'Calibration'`  (class method)

#### `Calibration.factor` -> float  (property)

What to multiply the standard deviation by to make the root mean square of z one.

#### `Calibration.describe() -> str`

### `class Handoff  (dataclass)`

What the tolerance rule did to one set of structures.

`n_deferred` structures had a standard deviation above the tolerance and go to the fine
level; the rest are answered by the model.  When the activities are known the root mean
square error of each side is measured, so the rule can be judged: the answered side should
be at or under the tolerance, and the deferred side worse.

Attributes:

- `tolerance: float`
- `n: int`
- `n_deferred: int`
- `mean_sd_answered: float`
- `mean_sd_deferred: float`
- `rmse_answered: float`
- `rmse_deferred: float`

#### `Handoff.of(sd: Sequence[float], tolerance: float, y: Optional[Sequence[float]]=None, mean: Optional[Sequence[float]]=None) -> 'Handoff'`  (class method)

#### `Handoff.n_answered` -> int  (property)

#### `Handoff.fraction_answered` -> float  (property)

#### `Handoff.describe() -> str`

### `cross_validate(K: np.ndarray, y: Sequence[float], lam: float, folds: int=10, seed: Optional[int]=0, fit_bias: bool=True, parts: Optional[Sequence[Sequence[int]]]=None) -> Tuple[np.ndarray, np.ndarray]`

Out-of-fold (prediction, standard deviation) on a training kernel, in the units of `y`.

The folds are the ones the model's own cross-validation uses, so the predictions agree with
the model's `y_cv` and the deviations are the ones a structure would have been given had
it been new.  The scale is fitted on the training part of each fold.  `parts` replaces the
random folds by given held-out index sets, for folds that hold out whole structures.

### `defer(sd: Sequence[float], tolerance: float) -> np.ndarray`

True for every structure the fast level must pass to the fine one: `sd > tolerance`.

### `tolerance_for_budget(sd: Sequence[float], fraction: float) -> float`

The tolerance that sends at most this share of the structures to the fine level.

`fraction` is the share the fine level can afford, between 0 and 1.  The answer is the
quantile of the deviations at `1 - fraction`; `defer` at that tolerance marks about
`fraction` of the structures.

## cmfqsar.tautomers

Source: `cmfqsar/tautomers.py`

Tautomer ensembles: the other half of plan item D4.

An azole can be written with its proton on either nitrogen, and the two forms have different
fields.  The choice is usually made by whatever wrote the file.  The ensemble built here averages
over the forms instead, with the same mathematics as the protonation ensemble in
`cmfqsar.protonation`: the members are different molecules, and the mean field of the
ensemble is itself a field (`cmfqsar.fields.combine_members`).

**The weights are the weak part, and they are weaker than the protonation ones.** A protonation
state has a population the pKa and the pH give it.  A tautomer has one only if the free energies
are known, and nothing in this program computes them: RDKit enumerates the forms and scores them
with a heuristic that prefers, for instance, an aromatic ring and a carbonyl over their
alternatives, but that score is not an energy and cannot be turned into a population.  So the
weights offered here are

  `uniform`   every enumerated form counts the same
  `score`     only the forms RDKit scores highest are kept, counted the same
  `given`     weights from a molecule property, for energies computed elsewhere

and none of them is a measurement.  `score` is the mildest assumption and the default: it keeps
the forms a chemist would draw and drops the ones the enumerator reaches only formally.  Energies
from a semiempirical calculation (plan item D3) would replace the guess with a Boltzmann weight.

The geometry is the parent's: a tautomer has the same heavy atoms in the same order, so the heavy
atoms keep their coordinates and only the hydrogens move.  An aligned series stays aligned.

### `class TautomersUnavailable  — bases: RuntimeError`

This RDKit build has no tautomer enumerator.

### `available() -> bool`

Whether the forms can be enumerated at all.

### `tautomer_states(mol: Chem.Mol, weighting: str='score', max_forms: int=MAX_FORMS, energy_prop: str=ENERGY_PROP, kt: float=KT) -> Ensemble`

Enumerate the tautomers of one molecule, on its own geometry, and weigh them.

Returns an `cmfqsar.protonation.Ensemble` so that the two kinds of ensemble are read
the same way.  A molecule with one tautomer, or one whose forms cannot be built on this
geometry, comes back as a single state that is the molecule itself.

### `tautomer_fields(mol: Chem.Mol, fieldset: FieldSet, weighting: str='score', max_forms: int=MAX_FORMS, name: str='', report: Optional[Dict]=None) -> MoleculeFields`

The field of one molecule as the mean over its tautomers.

### `forms_table(mols: Sequence[Chem.Mol], weighting: str='score', max_forms: int=MAX_FORMS) -> List[Ensemble]`

The tautomer ensembles of a whole series, in order.

### `ambiguous_share(ensembles: Sequence[Ensemble]) -> float`

What share of a series has more than one form, which is what the ensemble can change.

## cmfqsar.threebody

Source: `cmfqsar/threebody.py`

Three-body channels and handedness for a periodic cell (plan item M10).

The periodic autocorrelation of `cmfqsar.periodic` is a **two-point** descriptor: it counts
how much field weight sits at each distance from how much other field weight.  Two structures
with the same multiset of interatomic distances therefore give exactly the same channels, however
differently their atoms are arranged.  Two such structures are called **homometric**, and they are
not a curiosity: a defect series on one host lattice is full of arrangements that share their
distance histogram, which is why `benchmarks/run_2dmd.py` finds the autocorrelation near the
accuracy of guessing on a set where the arrangement of the defects is the whole of the signal.
A mirror image is the other blind spot: reflecting a structure leaves every distance unchanged, so
no two-point descriptor can tell a left-handed crystal from a right-handed one.

This module closes both, in the terms of this program.

**The three-body block.**  Every atom of the cell is taken as a centre in turn.  Around a centre,
two neighbours and the centre form a triangle, and a triangle is fixed by its three sides: the two
distances from the centre and the distance between the neighbours.  Writing `r_a` for the
distance of neighbour *a* from the centre and `d_ab` for the distance between two neighbours:

    Phi_fg(k, k', m) = (1/N) sum over centres l of
                       sum over neighbours a, b of l of
                       w_af w_bg R_k(r_a) R_k'(r_b) D_m(d_ab),

with `R_k` a Gaussian window on a grid of radii from the centre, `D_m` one on a grid of
distances between neighbours, and `N` the atoms in the cell.  The neighbourhood of a centre holds
the centre itself (at radius zero, so its own field types enter the channels) and every atom of
every periodic image within the cutoff.  The channel layout is the one of
`cmfqsar.triangles`, which builds the same block around one atom of a molecule (plan item C5):
one block per pair of field types, and inside a block one column per pair of radial windows and
distance window.  Dividing by `N` makes a supercell give exactly the same channels as the cell it
repeats.

**The handedness block.**  Three distances fix a triangle only up to its mirror image, so the block
above is still blind to a reflection.  The signed quantity that is not is the triple product of
three vectors.  Around a centre:

    X_fgh(s) = (1/N) sum over centres l of det[ A_f(l, s), A_g(l, s), A_h(l, s) ],
    A_f(l, s) = sum over neighbours a of l of w_af G_s(r_a) (r_a - r_l),

with `G_s` a Gaussian window on a grid of radii, which keeps the handedness resolved by distance
from the centre rather than reduced to one number.  The development plan expected this block to
cost a sum over triples of atoms, because the factorisation that `cmfqsar.invariant.chirality`
uses for a molecule needs an origin, and a periodic structure has no centroid to use.  With the
centre atom as that origin the factorisation survives in full: the determinant is linear in each of
its three arguments, so the sum over triples of neighbours **is** the determinant of three summed
vectors, and the block costs one pass over the neighbour list, not a cubic sum.  Under a proper
rotation all three vectors turn together and the determinant does not change; under a reflection it
changes sign; a centrosymmetric neighbourhood gives zero, as an achiral environment must.

**When the handedness block is empty.**  A determinant of three vectors is zero whenever they lie
in a plane, and this is not a rare case.  A field weight that depends only on the species of the
atom makes every `A_f` a combination of one vector per species, so a structure of two species
gives vectors in a plane at most and the whole block is identically zero; three species are the
minimum, and their vector sums must not be coplanar.  A site with a rotation axis through it also
gives zero, because a sum that is unchanged by the rotation can only lie along the axis.  The
block is therefore a sharp test for handedness and not a general-purpose descriptor: it is off by
default, and a structure that gives zero in every channel is telling the truth about itself.

**What it costs.**  One pass over the pairs of neighbours of every atom.  Measured by
`benchmarks/run_threebody.py --probe`: a cell of four atoms with a 4 angstrom reach and a
0.5 angstrom grid takes about 0.05 seconds and gives 11,322 channels, and a defect supercell of
191 atoms with a 0.75 angstrom grid takes about 0.26 seconds and gives 14,112.  The number of
channels grows as the square of (fields times radii), so the radial grid is coarse and the cutoff
short; the kernel works in the dual and does not mind the width.

**What is not covered.**  A disordered cell is refused: the mean of a product of three weights on
one site is not the product of their means, and this module tabulates no third moments.  No uniform
background is subtracted, unlike the two-point descriptor, so these channels carry how dense the
structure is as well as how it is arranged; a cosine-normalised kernel removes most of that, and
the two-point channels are there to carry the density properly.  The cutoff is sharp in the sense
that an atom beyond `r_max + reach * sigma` is left out, where it contributes less than
`exp(-reach^2 / 2)` of a window.

### `class ThreeBodySpec  (dataclass)`

The grids the three-body channels of a cell are evaluated on.

`r_max` is how far from a centre the radial grid reaches, in angstrom, and is the cutoff of the
descriptor; the grid of distances between two neighbours reaches twice as far, which is the
longest third side a triangle inside the cutoff can have.  `step` is the spacing of both grids
and `sigma` the width of the Gaussian window on every grid point.  `include_self` keeps the
terms in which the two neighbours are the same atom, which carry the radial density itself.
With `chirality` on, the handedness block is added, resolved over `chirality_shells` radial
windows of width `chirality_sigma` about the centre.

Attributes:

- `r_max: float` = `4.0`
- `step: float` = `0.5`
- `sigma: float` = `0.35`
- `include_self: bool` = `True`
- `reach: float` = `DEFAULT_REACH`
- `chirality: bool` = `False`
- `chirality_shells: int` = `4`
- `chirality_sigma: float` = `0.5`

#### `ThreeBodySpec.triangle` -> TriangleSpec  (property)

The channel layout of `cmfqsar.triangles`, which this block shares.

#### `ThreeBodySpec.radii` -> np.ndarray  (property)

#### `ThreeBodySpec.distances` -> np.ndarray  (property)

#### `ThreeBodySpec.n_radii` -> int  (property)

#### `ThreeBodySpec.n_distances` -> int  (property)

#### `ThreeBodySpec.cutoff` -> float  (property)

Beyond this distance from a centre an atom is not in the sum.

#### `ThreeBodySpec.n_bins` -> int  (property)

The width of the innermost grid, which is the grid of distances between neighbours.

The helpers that scale the channels and build the kernel
(`cmfqsar.invariant.channel_scale`, `cmfqsar.invariant.autocorrelation_kernel`)
ask a spec for one block width to fall back on.  The blocks of this descriptor are not all
of one width, so they are always passed explicitly and this value is only the fallback.

#### `ThreeBodySpec.n_shells` -> int  (property)

Radial windows the handedness block is resolved over; one for a single number.

#### `ThreeBodySpec.shell_centres` -> np.ndarray  (property)

Where the windows of the handedness block sit, measured from the centre atom.

#### `ThreeBodySpec.radial_pairs(same: bool) -> List[Tuple[int, int]]`

#### `ThreeBodySpec.block_sizes(n_fields: int) -> List[int]`

Channels per block: the pair blocks first, then the handedness blocks.

#### `ThreeBodySpec.n_channels(n_fields: int) -> int`

### `neighbourhoods(crystal, cutoff: float) -> List[Tuple[np.ndarray, np.ndarray]]`

For every atom of the cell, the vectors to its neighbours and their field weights.

The neighbourhood of an atom holds the atom itself, at the vector zero, and every atom of
every periodic image whose distance is below `cutoff`.  The vectors are measured from the
centre, so they do not change when the structure is moved, and they turn with it when it is
turned.  The periodicity flags of the cell are respected: a layer has neighbours in two
directions and a chain in one.

### `threebody_autocorrelation(crystal, spec: ThreeBodySpec=ThreeBodySpec(), pairs: Optional[Sequence[Tuple[int, int]]]=None) -> np.ndarray`

The three-body channels of one cell, flattened: `(n_pair_blocks * channels,)`.

The sum runs over the atoms of the cell as centres and over ordered pairs of their
neighbours, so a pair of different field types is counted both ways round, as the two-point
descriptor counts it.  For a site record (plan item M5) only that site is a centre, and the
channels are the three-body profile of that one site.

### `periodic_chirality(crystal, spec: ThreeBodySpec=ThreeBodySpec(), triples: Optional[Sequence[Tuple[int, int, int]]]=None) -> np.ndarray`

The handedness channels of one cell, flattened: `(n_triples * n_shells,)`.

Each channel is the signed volume spanned by three field-weighted vectors, summed over the
atoms of the cell as centres and divided by their number.  A structure and its mirror image
give the same number with the opposite sign, and a centrosymmetric structure gives zero.

### `threebody_features(crystal, spec: ThreeBodySpec=ThreeBodySpec(), pairs: Optional[Sequence[Tuple[int, int]]]=None, triples: Optional[Sequence[Tuple[int, int, int]]]=None) -> np.ndarray`

The channels of one cell: the three-body block, and the handedness block after it.

### `threebody_feature_matrix(crystals: Sequence, spec: ThreeBodySpec=ThreeBodySpec(), n_fields: Optional[int]=None, progress=None, dtype=float) -> np.ndarray`

Three-body features of a list of cells, one row each.

### `threebody_block_sizes(spec: ThreeBodySpec, n_fields: int) -> List[int]`

How many columns each channel block takes, in the order the features are laid out.

### `threebody_channel_labels(fieldset: FieldSet, spec: ThreeBodySpec) -> List[str]`

One label per channel: which two field types, at which two radii and how far apart.

### `channel_spread(features: np.ndarray, blocks: Optional[Sequence[int]]=None) -> np.ndarray`

The standard deviation of every channel over a set of structures.

A ridge weight alone says nothing: a channel that hardly varies can carry a large weight and
move no prediction.  What a reading needs is the weight times the spread, which is the
convention `cmfqsar.periodic.top_channels` uses.

### `pattern_table(weights: Sequence[float], features: np.ndarray, labels: Sequence[str], top: int=20) -> List[Dict[str, float]]`

The channels that move the prediction most, largest first.

Each row carries the label of the channel, the weight the model gives it, the spread of the
channel over the structures and their product, which is how much of the prediction that
channel accounts for.

## cmfqsar.topomer

Source: `cmfqsar/topomer.py`

Topomer fields: a frame that comes from the topology, not from a superposition.

Everything else here needs the molecules to share a frame, and gets it by superimposing them --
which works for a congeneric series and silently stops meaning anything when the substituents are
unrelated, as the chemotype measurements show.  Cramer's topomer idea (topomer CoMFA, 2003)
removes the search altogether: cut the molecule at the bond that attaches a substituent, and
build that fragment into a pose that is a *deterministic function of its topology* -- standard
geometry, every rotatable torsion at a fixed value, the attachment atom at the origin and its
open valence along a fixed axis, the remaining roll fixed by a canonical rule.  The same R-group
then always ends in the same place, whatever molecule it came from and however its atoms were
numbered, so the fields of different substituents at one attachment point can be compared with no
alignment at all.

The pose is a convention, not a bioactive conformation, and the method is clear about that: it
compares comparable things at one attachment point, and says nothing about how the ligand really
sits.  What it gains is that the comparison is reproducible and instant.

Two things make this fit CMF better than it fits CoMFA.  There is no grid: a fragment's field is
its atoms and weights, and a fragment-fragment kernel is an analytic number.  And the fields are
additive, so a molecule's kernel with another decomposes into scaffold and substituent terms that
can be tabulated once per fragment and reused across a library -- which is the topomer-search
technique, obtained here without approximation.

The pose rule, in order:

1. the fragment is renumbered canonically, so two copies of the same R-group that were cut out of
   different molecules have the same atom order and therefore the same everything after it;
2. one conformation is embedded with a fixed seed and every rotatable torsion is then set to
   `torsion` degrees (180 by default, Cramer's anti convention), which replaces whatever the
   embedding happened to produce;
3. the attachment atom goes to the origin and the bond to its open valence along -x, so the
   fragment body extends towards +x;
4. the roll about x is fixed by the principal axis of the heavy atoms in the yz-plane, with the
   sign chosen by the lowest-ranked off-axis atom, and left alone when the fragment is too small
   or too symmetric for that to mean anything.

### `class TopomerSpec  (dataclass)`

How a fragment is built into its topomer pose.

Attributes:

- `torsion: float` = `180.0`
- `seed: int` = `12648430`
- `optimise: bool` = `False`
- `roll: bool` = `True`
- `spacing: float` = `100.0`

### `canonical(frag: Chem.Mol) -> Chem.Mol`

The same fragment renumbered the same way, whatever molecule it was cut out of.

### `topomer_pose(frag: Chem.Mol, spec: TopomerSpec=TopomerSpec()) -> Chem.Mol`

Build one fragment into its topomer pose.  The result depends only on its topology.

### `framework_core(mols: Sequence[Chem.Mol], min_atoms: int=3) -> Optional[Chem.Mol]`

The scaffold to cut a topomer series from: the framework of its largest chemotype.

`common_core` takes the maximum common substructure of everything handed to it, which is
the right answer for a congeneric series and the wrong one for anything else.  A data set that
mixes chemotypes has an MCS that is some small fragment -- `CCO` on the GPB set -- fused into
every ring system, so it cuts nothing off and the topomer route loses every molecule.  The
measurements in `benchmarks/run_topomer.py` are what made this the default.

What a topomer model actually needs is the scaffold of *one* series, so this takes the
Bemis-Murcko framework shared by the largest group of molecules.  Combine it with
`series_indices`, which collects every molecule that framework can be cut off cleanly --
including molecules whose own framework is larger, because an extra ring on a substituent is
part of the R-group and not of the scaffold.

### `series_indices(mols: Sequence[Chem.Mol], core: Chem.Mol) -> List[int]`

Which molecules of a set this core can be cut off cleanly, i.e. which form the series.

### `choose_core(mols: Sequence[Chem.Mol], mode: str='framework', timeout: int=20) -> Tuple[Optional[Chem.Mol], str]`

(core, how it was found) for `mode` in {"framework", "mcs"}, with a fallback either way.

### `common_core(mols: Sequence[Chem.Mol], timeout: int=20, threshold: float=1.0) -> Optional[Chem.Mol]`

The maximum common substructure of a series, which is what the R-groups are attached to.

### `substituents(mol: Chem.Mol, core: Chem.Mol) -> List[Chem.Mol]`

The R-groups of one molecule, each carrying its attachment point.

### `largest_substituent(mol: Chem.Mol, core: Chem.Mol) -> Optional[Chem.Mol]`

The R-group with the most heavy atoms: the one a single-cut topomer model uses.

### `cuttable(mol: Chem.Mol, core: Chem.Mol) -> bool`

True when the core can be cut off this molecule at acyclic single bonds alone.

A core fused to the rest of the molecule -- a benzene ring picked out of a bicyclic system,
say -- cannot be removed without opening a ring, and whatever is left is not a substituent.
The topomer construction needs cuts of the kind a chemist would call an R-group.

### `cuttable_fraction(mols: Sequence[Chem.Mol], core: Chem.Mol) -> float`

How much of a series the core can be cut out of cleanly.

### `attachment_label(frag: Chem.Mol) -> int`

Which core atom this substituent is attached to, as `substituents` labels it.

### `series_labels(mols: Sequence[Chem.Mol], core: Chem.Mol) -> List[int]`

The attachment points the series uses, in a fixed order.

### `topomer_molecule(mol: Chem.Mol, core: Chem.Mol, labels: Sequence[int], spec: TopomerSpec=TopomerSpec(), cache: Optional[Dict[str, Chem.Mol]]=None) -> Optional[Chem.Mol]`

Every substituent of `mol` in its own topomer pose, each at its own attachment site.

The result is one molecule for the field machinery to work on, but it is not a molecule any
chemist would draw: it is the series' substituents, each built by the rule and placed in a
region of space reserved for its attachment point.  A position a molecule does not substitute
simply contributes nothing there.

### `topomer_assembly(frags: Sequence[Chem.Mol], labels: Sequence[int], spec: TopomerSpec=TopomerSpec(), cache: Optional[Dict[str, Chem.Mol]]=None) -> Optional[Chem.Mol]`

The same construction from fragments that were not cut off anything.

`topomer_molecule` cuts a molecule up and calls this; a design loop builds its
candidates by replacing one fragment of that list and calls it directly, so that a molecule
nobody has made is posed by exactly the rule that posed the ones that were.

### `topomer_fields(mols: Sequence[Chem.Mol], names: Sequence[str], fieldset: FieldSet, core: Optional[Chem.Mol]=None, spec: TopomerSpec=TopomerSpec(), labels: Optional[Sequence[int]]=None, progress=None) -> Tuple[List[Optional[MoleculeFields]], List[str]]`

Topomer fields of a series: every substituent posed by the rule, at its own site.

Returns (fields, log).  A molecule whose substituents cannot be built gets `None`, so the
caller can drop it and say which ones were lost.  Identical substituents at the same attachment
point give identical fields, whatever molecule they came from, which is what makes the
comparison free of any alignment.

### `unique_substituents(mols: Sequence[Chem.Mol], core: Optional[Chem.Mol]=None) -> List[str]`

The distinct R-groups of a series, as SMILES -- the library a topomer model sees.

## cmfqsar.trajectory

Source: `cmfqsar/trajectory.py`

A molecular-dynamics trajectory as a conformational ensemble (plan item D7).

The ensemble machinery of this program takes a molecule carrying several conformations and a
weight for each, and builds the mean field over them (the mean-map kernel of
`cmfqsar/fields.py`).  Where those conformations come from is of no concern to the kernel.  So far
they have come from the conformer generator, which samples the molecule on its own, in no
surroundings.  A trajectory of the ligand inside its pocket samples it where it actually sits,
with the protein holding it, and this module turns such a trajectory into exactly the object the
ensemble machinery already takes: one RDKit molecule, a few dozen conformations, one weight each.

The chain is five steps, and each is a decision worth stating.

1. **Read.** `mdtraj` or `MDAnalysis`, whichever is installed, both optional dependencies in
   the way ORCA and xtb are external programs.  XTC, DCD, NetCDF, TRR and multi-model PDB all
   work, because the readers handle them; this module does not parse any format itself.
2. **Strip.** Water, the common ion residues and anything else the caller names are dropped.  A
   trajectory is mostly solvent, and solvent is not part of the ligand's field.
3. **Superimpose.** When the topology holds a receptor, every frame is superimposed on the
   **backbone of the binding site** -- the residues near the ligand in the first frame -- so that
   what is left in the ligand's coordinates is its motion *relative to the pocket*.  That is the
   quantity a field model of a series needs, because it is what differs between two ligands in
   the same site.  Superimposing on the ligand's own atoms instead would remove exactly that and
   leave only its internal flexibility; it is what happens when there is no receptor, and the
   result says which was used.
4. **Subsample.** The kernel of an ensemble of k conformations costs k^2 atom-pair integrals, so
   a hundred thousand frames is not an option and the thinning has to happen here rather than be
   left to the caller.  Frames are clustered by heavy-atom distance with a leader algorithm: the
   first frame starts a cluster, every later frame joins the first cluster whose representative
   is within the cutoff, and starts a new one otherwise.  If that gives more clusters than asked
   for, the cutoff is raised and the pass repeated.  The representative of a cluster is the frame
   that opened it.
5. **Weight.** The weight of a conformation is its **cluster population** divided by the number
   of frames, and nothing else.  It is never a Boltzmann factor of the frame energy.  An
   equilibrated trajectory has already sampled the Boltzmann distribution -- that is what running
   it was for -- so weighting the frames by exp(-E/kT) a second time would square the
   distribution and sharpen the ensemble onto its deepest basin.  The weights are written to the
   molecule property `conformer_weights`, which is what `conformer_weighting="given"` reads.

The force-field charges of the topology, where the format carries them (PSF, PRMTOP, a TPR), are
written as the per-atom property `q_mm`, so that `prop:q_mm` is a field like any other and a
model can use the charges the simulation itself was run with.

**This module is the reader, and the question it was built for is not settled by it.** Plan item
D8 is the measurement: short simulations on a set whose ligands have crystal poses, and the three
comparisons that say whether the trajectory mean field beats the single crystal pose.
`benchmarks/run_trajectory.py` measures what can be measured without those simulations -- that
the two readers agree, what the thinning costs, and how fast the mean field settles as the number
of representatives grows -- and says plainly that it answers the reader's own questions and not
D8's.

### `class TrajectoryError  — bases: RuntimeError`

### `class Trajectory  (dataclass)`

One trajectory as plain arrays, whichever reader produced it. Coordinates in angstrom.

Attributes:

- `coords: np.ndarray`
- `elements: List[str]`
- `atom_names: List[str]`
- `residues: List[str]`
- `resids: np.ndarray`
- `bonds: List[Tuple[int, int]]` = `field(default_factory=list)`
- `charges: Optional[np.ndarray]` = `None`
- `box: Optional[np.ndarray]` = `None`
- `engine: str` = `''`

#### `Trajectory.n_frames` -> int  (property)

#### `Trajectory.n_atoms` -> int  (property)

#### `Trajectory.subset(idx: Sequence[int]) -> 'Trajectory'`

### `class Ensemble  (dataclass)`

What `read_trajectory` returns.

Attributes:

- `mol: Chem.Mol`
- `weights: np.ndarray`
- `populations: np.ndarray`
- `frames: np.ndarray`
- `n_frames: int`
- `stripped: int`
- `fitted_on: str`
- `fit_atoms: int`
- `cutoff: float`
- `engine: str`

#### `Ensemble.describe() -> str`

### `available_engines() -> List[str]`

Which of the two readers can be imported here.

### `load(topology: str, trajectory: Optional[str]=None, stride: int=1, engine: str='auto') -> Trajectory`

Read a topology and (optionally) a trajectory into plain arrays.

`engine` is `mdtraj`, `mdanalysis` or `auto`.  Under `auto` MDAnalysis is preferred
when the topology is one of the formats that carry force-field charges, because those charges
are the point of reading them; otherwise mdtraj is, being the faster reader.

### `strip_solvent(traj: Trajectory, extra: Sequence[str]=()) -> Tuple[Trajectory, int]`

Drop water, counter-ions and any residue named in `extra`; returns (trajectory, dropped).

### `ligand_indices(traj: Trajectory, resname: Optional[str]=None) -> np.ndarray`

The atoms of the ligand.

With `resname` given, the atoms of that residue.  Without it, the largest residue that is
neither protein nor nucleic nor solvent, which is what a ligand is in almost every prepared
system; if there is none, every atom, which is the case of a trajectory of the ligand alone.

### `site_backbone(traj: Trajectory, ligand: Sequence[int], radius: float=DEFAULT_SITE_RADIUS, frame: int=0) -> np.ndarray`

Backbone atoms of the residues with an atom within `radius` of the ligand in `frame`.

Empty when the topology holds no protein, which is how the caller learns that there is no
receptor to superimpose on.

### `make_whole(coords: np.ndarray, group: Sequence[int], box: Optional[np.ndarray]) -> np.ndarray`

Undo the wrapping inside `group`: every atom moved to the image nearest the first one.

A trajectory is usually written with every atom wrapped into the cell, which splits any
molecule that straddles a face: half of it appears a box length away from the other half.
For a molecule small enough that no two of its atoms are further apart than half the cell --
a ligand, always -- taking each atom's image nearest the first atom puts it back together.

### `reimage(coords: np.ndarray, group: Sequence[int], anchor: Sequence[int], box: Optional[np.ndarray]) -> np.ndarray`

Move `group`, as one piece, to the periodic image nearest `anchor`.

The ligand of a simulation wanders across the cell and is written wrapped, so in some frames
it appears a box length from the protein it is bound to. Nothing physical has happened: the
image beside the protein is the same molecule. Without this step those frames look like a
ligand sitting fifty angstroms away, which is how a periodic-boundary artefact turns into a
false conclusion about a simulation.

### `superimpose(coords: np.ndarray, fit_atoms: Sequence[int], reference: int=0, target: Optional[np.ndarray]=None) -> np.ndarray`

Every frame moved so that `fit_atoms` lie on the target.

The target is frame `reference` of this trajectory by default, and `target` replaces it with
positions from somewhere else -- the one structure a whole series was built from, so that
several trajectories come back in one frame rather than each in its own.

The whole system is moved, not only the fitted atoms, so what the ligand's coordinates then
show is its motion relative to whatever was fitted.

### `rmsd_to(coords: np.ndarray, frame: np.ndarray) -> np.ndarray`

Root-mean-square distance of every frame from `frame`, where they stand, with no fit.

### `leader_cluster(coords: np.ndarray, cutoff: float=DEFAULT_CUTOFF, max_frames: int=DEFAULT_MAX_FRAMES, grow: float=1.5, max_passes: int=40) -> Tuple[np.ndarray, np.ndarray, float]`

Thin the frames; returns (representatives, population of each, the radius that was used).

The leader algorithm in one pass over the frames in their own order: a frame joins the first
representative within `cutoff` and becomes a representative itself otherwise.  The order is
the trajectory's, so the result is deterministic and does not depend on a random seed.  When
a pass produces more representatives than `max_frames`, the cutoff is multiplied by `grow`
and the pass repeated, so the number asked for is an upper bound that is always met.  The
radius returned is the one the successful pass used, and every frame is within it of its own
representative.

### `to_molecule(traj: Trajectory, frames: Sequence[int], weights: Sequence[float], template: Optional[Chem.Mol]=None, name: str='', charge_prop: str=CHARGE_PROP) -> Chem.Mol`

One molecule carrying the chosen frames as conformations, with their weights.

`frames` are indices into the trajectory and `weights` the matching populations, normalised
here so that they sum to one.  The weights go to the molecule property `conformer_weights`,
which `conformer_weighting="given"` reads.

### `reference_positions(reference: str, traj: 'Trajectory', fit: Sequence[int]) -> np.ndarray`

The positions the fitted atoms have in `reference`, checked atom by atom.

The trajectory lists the protein first, because that is the order a system is built in, so a
protein atom's index in the trajectory is its index in the reference as well.  The elements
are compared rather than trusted: a mismatch means the two do not describe the same protein
and a superposition on it would be meaningless.

### `read_trajectory(topology: str, trajectory: Optional[str]=None, *, ligand: Optional[str]=None, template: Optional[Chem.Mol]=None, max_frames: int=DEFAULT_MAX_FRAMES, cutoff: float=DEFAULT_CUTOFF, stride: int=1, site_radius: float=DEFAULT_SITE_RADIUS, extra_solvent: Sequence[str]=(), engine: str='auto', name: str='', heavy_only: bool=True, reference: Optional[str]=None) -> Ensemble`

A trajectory as a weighted conformational ensemble of its ligand (plan item D7).

`topology` may be a PDB, PSF, PRMTOP or anything the reader understands; `trajectory` is the
frames (XTC, DCD, NetCDF, TRR), or None when the topology file holds them itself, as a
multi-model PDB does.  `ligand` names the residue; without it the largest residue that is
neither protein nor solvent is taken.  `template` is the ligand as the data set holds it and
is the reliable way to get the bond orders right; without one they are worked out from the
topology's bonds, which can fail and then says so.

`heavy_only` measures the distances between frames on the heavy atoms alone, which is what a
conformational difference means; the hydrogens still travel with the frames that are kept.

`reference` is the structure every frame is superimposed on, and it matters whenever several
trajectories of one series are read.  Without it each trajectory is superimposed on its own
first frame, which is that run's equilibrated protein: two runs equilibrate to protein
conformations that differ by a few tenths of an angstrom, so their ligands end up in frames
that differ by the same rotation and the series loses the common alignment a field kernel
depends on.  Pass the one structure every run was built from -- the prepared receptor -- and
every ligand comes back in that structure's frame.  The file must list the protein atoms
first and in the same order as the trajectory does, which is the case when the systems were
built from it; the elements are checked and a mismatch is refused.

## cmfqsar.transfer

Source: `cmfqsar/transfer.py`

Comparing two models by their co-fields (plan item F3).

A co-field is the model as a function of space: where more of a field raises the predicted
activity and where it lowers it.  Two models that were fitted on molecules sharing a frame
therefore have two functions on the same space, and the overlap integral of those two functions
is a similarity between the *models* -- not between their molecules, and not between their
predictions, but between what they ask the chemistry for.

It needs no grid.  With

    C_f^A(r) = sum_j a_j s_fj X_fj(r),      C_f^B(r) = sum_k b_k s_fk X_fk(r)

the overlap is a sum of overlap integrals of atom-centred Gaussians, which is the field kernel
between the two training sets:

    <C_f^A, C_f^B> = sum_jk a_j b_k s_fj s_fk K_f^raw(j, k) = a^T K_f(A, B) b

one matrix product on a kernel block the program computes for a prediction anyway.  The cosine

    cos_f = <C_f^A, C_f^B> / (||C_f^A|| ||C_f^B||)

is the number to read, because it is invariant to the scale of either co-field -- to the units of
either activity, to the mixing coefficients, to the field scaling -- and so compares the *shapes*
of two demands rather than their sizes.

What it is for, in the order the question usually arises:

* **Two targets.**  A positive cosine says the two targets require the same thing at the same place,
  which is why selectivity is hard; a negative one says they require opposite things, which is where
  selectivity comes from.  The multi-task selectivity co-field is the difference of the two
  co-fields, and this is its inner product -- the same algebra, read as an angle.
* **Two series on one target.**  Agreement means the SAR transfers; disagreement means two
  chemotypes are being fitted by two different explanations and a model pooled over both will be an
  average of them.
* **One series across rounds.**  A model refitted after new compounds should move its co-field
  only where the new compounds taught it something; the cosine with the previous round says how
  much was learned and the per-field cosines say in which field.

**What it cannot do.**  Two co-fields are functions on one space only if their molecules were
posed in one frame: comparing models fitted in different frames compares two different coordinate
systems and means nothing, which is why `incompatible` checks what it can (the same fields,
the same attenuations, overlapping boxes) and the caller is told rather than silently given a
number.  And a cosine is not a prediction: that two models require the same thing does not mean
either is right.

### `class FieldOverlap  (dataclass)`

One field's share of the comparison.

Attributes:

- `field: str`
- `label: str`
- `cosine: float`
- `overlap: float`
- `norm_a: float`
- `norm_b: float`

#### `FieldOverlap.describe() -> str`

### `class ModelOverlap  (dataclass)`

How much two models ask the same thing of the chemistry.

Attributes:

- `name_a: str`
- `name_b: str`
- `cosine: float`
- `fields: List[FieldOverlap]` = `_field(default_factory=list)`
- `notes: List[str]` = `_field(default_factory=list)`

#### `ModelOverlap.by_field() -> Dict[str, float]`

#### `ModelOverlap.describe() -> str`

### `incompatible(a, b) -> Optional[str]`

Why these two models cannot be compared by their co-fields, or None if they can.

The checks are the ones that can be made from the model files: both linear, both on the
overlap kernel, the same fields with the same attenuations, and training boxes that actually
overlap.  What cannot be checked is whether the two alignments mean the same thing, and that
is the assumption the whole comparison rests on.

### `cofield_overlap(model_a, model_b, name_a: str='A', name_b: str='B', include_scale: bool=True, strict: bool=True) -> ModelOverlap`

The co-field cosine of two models, field by field and over all fields together.

`include_scale` weights the fields by each model's own mixing coefficients and field
scaling when the overall cosine is formed, which is how the models themselves weigh them;
the per-field cosines do not depend on it. With `strict` off, a reason the two should not be
compared becomes a note on the result instead of an exception, which is what a matrix over
many models needs.

### `overlap_matrix(models: Sequence, names: Sequence[str], include_scale: bool=True) -> Tuple[np.ndarray, List[ModelOverlap]]`

The full table: every model against every other, by the cosine of their co-fields.

The diagonal is one by construction, and the matrix is symmetric, so only the upper triangle
is computed. A pair that cannot be compared leaves nan and a note.

### `describe_matrix(M: np.ndarray, names: Sequence[str], width: int=7) -> str`

The matrix as a text table, for the command line.

## cmfqsar.transformfield

Source: `cmfqsar/transformfield.py`

A matched pair as a field placed in space (plan item MC6).

A matched molecular pair is usually a row in a table: a transformation, written as two group
names, and the activity change it produced.  In a series whose molecules are in one frame the
same pair is something else as well -- **a difference of two fields**, a local object sitting
where the two groups differ.  Plan item MC6 asks whether that object carries the effect of the
transformation with it: can a change measured in one series be predicted from changes measured in
others?

Plan item D5 is both the warning and the reference.  Increments fitted on one series did not
transfer to another, and the vectors fitted on two sets agreed only to +0.89 at best.  An
increment is a number attached to an atom type; the field difference of this module is the same
transformation with its geometry kept, so it can tell a methyl that points into a pocket from one
that points at the solvent.  Whether that is enough is a measurement, not an expectation.

What one pair gives
-------------------
* Both members are in the frame of the series, so their fields are comparable with each other
  without further work.  What the two molecules do **not** share with another series is a frame,
  so the difference field is placed in a frame of its own: the origin at the atom of the core
  that carries the group, the x axis along the bond to the group, and the plane fixed by one more
  neighbour of that atom.  That frame is local and it is the same one for both members, so
  nothing about the pair is changed by it.
* The difference field is the field of the second group minus the field of the first, each
  restricted to the atoms of the group itself (the mask of `cmfqsar.bioisostere`).  It is an
  ordinary `cmfqsar.fields.MoleculeFields` with signed weights, so every kernel and every
  display of the program applies to it.
* The overlap kernel of two fields is bilinear, so the kernel of two difference fields is
  signed: reversing a pair reverses its kernel row.  A ridge with no intercept therefore predicts
  exactly the opposite change for the reverse of a pair, which is the one property such a model
  must have.

`benchmarks/run_transformfield.py` holds out whole series and asks whether the model beats
predicting no change at all, which is the gate of the item.

### `class Transformation  (dataclass)`

One matched pair as a difference field in the frame of its attachment point.

Attributes:

- `series: str`
- `key: Tuple[str, str]`
- `field: MoleculeFields`
- `delta: float`
- `names: Tuple[str, str]` = `('', '')`

### `attachment(mol, group_atoms: Sequence[int]) -> Optional[Tuple[int, int]]`

(the core atom carrying the group, the group atom bonded to it), or None.

`mol` is the RDKit molecule the group was cut out of.  A single cut leaves exactly one bond
between the two pieces; anything else is not an attachment point and is refused.

### `attachment_frame(mol, fields: MoleculeFields, core_atom: int, group_atom: int) -> Optional[Tuple[np.ndarray, np.ndarray]]`

(origin, rotation) of the attachment point, or None where no frame exists.

The origin is the core atom, the x axis points at the group, and the plane is fixed by one
more neighbour of the core atom.  That neighbour is chosen by its atom type and then by its
distance to the group atom, rounded, so that the two members of a pair -- which share the
core and are superimposed -- choose the same one.

### `posed(fields: MoleculeFields, atoms: Sequence[int], origin: np.ndarray, R: np.ndarray, name: str='') -> MoleculeFields`

The group's own atoms, moved into the frame of its attachment point.

### `pair_field(mols, fields: Sequence[MoleculeFields], pair: MatchedPair, series: str='') -> Optional[Transformation]`

The difference field of one matched pair, posed, or None where no frame exists.

The field is the second group minus the first, so its sign goes with `pair.delta`.

### `reversed_of(t: Transformation) -> Transformation`

The same pair read the other way round: the field and the change both change sign.

### `both_ways(items: Sequence[Transformation]) -> List[Transformation]`

Every pair in both directions, which is what makes a model of them antisymmetric.

### `average_field(items: Sequence[Transformation], name: str='') -> MoleculeFields`

The mean difference field of several pairs of the same transformation.

A field is a sum of Gaussians, so the mean of several is again a field: its atoms are the
atoms of all of them with the weights divided by how many there are.  That is the object plan
item MC6 calls the transformation field, and it can be drawn like any other.

### `by_transformation(items: Sequence[Transformation]) -> Dict[Tuple[str, str], List[Transformation]]`

The pairs grouped by the transformation they are, in the direction they were read.

### `shared(items: Sequence[Transformation], least: int=2) -> Dict[Tuple[str, str], List[str]]`

The transformations seen in at least `least` different series, and in which.

### `increment_table(items: Sequence[Transformation]) -> Dict[Tuple[str, str], float]`

The mean measured change of every transformation, which is what D5 fitted and transferred.

This is the reference the field has to beat: the transformation as a number, with no geometry
in it at all.

### `increment_predict(table: Dict[Tuple[str, str], float], items: Sequence[Transformation]) -> Tuple[np.ndarray, np.ndarray]`

(the predicted change, whether the transformation was seen) for every pair.

## cmfqsar.triangles

Source: `cmfqsar/triangles.py`

The three-distance autocorrelation of an environment: the angle at the centre (plan item C5).

The rotation-invariant local kernel of `cmfqsar.localenv` describes an atom's surroundings by
the two-point autocorrelation of `cmfqsar.invariant`: for every pair of field types, how much
field weight sits at each distance from how much other field weight.  On an environment that is a
histogram of the distances between neighbours of the centre, and it leaves something open.  Two
neighbours and the centre form a triangle; the two-point descriptor resolves the side between the
neighbours and not the two sides that meet at the centre, so it does not know the angle at the
centre.  `benchmarks/run_soap_gate.py` measured what that costs: the SOAP descriptor (Smooth
Overlap of Atomic Positions, Bartok, Kondor and Csanyi 2013), which resolves the triangle, lowers
the error of a carbon-shift model by a third, and a SOAP restricted to its two-body part does not.

This module resolves the triangle in the terms of this program.  For an environment whose first
atom is the centre, with `r_a` the distance of atom *a* from the centre and `d_ab` the distance
between atoms *a* and *b*, the descriptor is

    Phi_fg(k, k', m) = sum over atoms a, b of
                       w_af w_bg R_k(r_a) R_k'(r_b) D_m(d_ab),

with `R_k` a Gaussian window on a grid of radii from the centre and `D_m` one on a grid of
distances.  A triangle is fixed by its three sides, so this carries the same information as the
SOAP power spectrum, which resolves the two radii and the angle between them; the basis differs
and nothing else.  The channels of the two-point descriptor are the sum of these over the two
radii.

**Channels.**  The sum is symmetric under swapping (a, f, k) with (b, g, k'), so one channel is
kept for each unordered pair of (field type, radial window) labels and the two orderings are
added, as the two-point descriptor adds a pair both ways round.  The channels come in blocks, one
per pair of field types, in the order of `cmfqsar.invariant.field_pairs`; a block of one
field type with itself holds the radial pairs `k <= k'` and a block of two different types holds
all of them.  Before the cosine-normalised linear kernel is taken the blocks are scaled, by
default per field (`block_scales`): the two-point kernel's scaling per pair of fields gives
a block of two rare channels the weight of a block of two common ones, and with element channels
that costs accuracy.  `triangle_kernel` can raise the kernel to an integer power, which is
what SOAP does with its exponent zeta; a power above one has no finite feature map, so it is for
the dual form only.

**The channels are the fields.**  SOAP has one density per element; this descriptor has one
weight per field, so the channels are the charges, the steric bulk, the hydrophobicity and the
hydrogen-bond strengths of the atoms around the centre, and a channel pair says which two
properties, at which two radii and which distance apart, the model reads.  An element indicator
is itself a field, so SOAP's channels are one setting of this descriptor and the two can be mixed.

**Cost.**  One pass over the atom pairs of the environment, with a window product per pair and
grid point: for forty atoms, nine radii and seventeen distances it is a few milliseconds.  The
number of channels grows as the square of (fields times radii), which is why the radial grid is
coarse and the cutoff short; the cross-validation of `cmfqsar.localenv` works in the dual
and does not mind.

**What is not resolved.**  Three distances fix a triangle up to its mirror image, so an
environment and its reflection give the same channels.  That is the blind spot plan item C5
measured on whole molecules, and it is unchanged here.

### `class TriangleSpec  (dataclass)`

The two grids the three-distance autocorrelation is evaluated on.

`r_max` is how far from the centre the radial grid reaches, in angstrom, and is the cutoff of
the descriptor; the distance grid reaches twice as far, which is the longest side a triangle
inside the cutoff can have.  `step` is the spacing of both grids and `sigma` the width of the
Gaussian window on every grid point.  Atoms further than `r_max + reach * sigma` from the
centre contribute less than `exp(-reach^2 / 2)` of a window and are left out of the pair
sum.

Attributes:

- `r_max: float` = `4.0`
- `step: float` = `0.5`
- `sigma: float` = `0.35`
- `include_self: bool` = `True`
- `reach: float` = `4.0`

#### `TriangleSpec.radii` -> np.ndarray  (property)

#### `TriangleSpec.distances` -> np.ndarray  (property)

#### `TriangleSpec.n_radii` -> int  (property)

#### `TriangleSpec.n_distances` -> int  (property)

#### `TriangleSpec.cutoff` -> float  (property)

Beyond this distance from the centre an atom is not in the sum.

#### `TriangleSpec.radial_pairs(same: bool) -> List[Tuple[int, int]]`

The (k, k') radial windows of one block: `k <= k'` for a field type with itself.

#### `TriangleSpec.block_sizes(n_fields: int) -> List[int]`

Channels per block, one block per pair of field types.

#### `TriangleSpec.n_channels(n_fields: int) -> int`

### `triangle_features(env: MoleculeFields, spec: TriangleSpec=TriangleSpec(), pairs: Optional[Sequence[Tuple[int, int]]]=None) -> np.ndarray`

The three-distance autocorrelation of one environment, flattened.

The first atom of `env` is the centre, as `cmfqsar.localenv.environment` builds it;
radii are measured from it.  The result depends on the arrangement of the atoms about the
centre and on nothing else: not on where the environment sits or how it is turned.

### `triangle_feature_matrix(envs: Sequence[MoleculeFields], spec: TriangleSpec=TriangleSpec(), n_fields: Optional[int]=None, progress=None) -> np.ndarray`

Three-distance features of a list of environments, one row each.

### `block_scales(rms: Sequence[float], n_fields: int, blocks: Sequence[int], scaling: str='field') -> np.ndarray`

One scale per channel from the root mean square of every block.

`"pair"` divides every block by its own root mean square, as
`cmfqsar.invariant.channel_scale` does for the two-point kernel.  `"field"` divides
the block of fields f and g by the geometric mean of the root mean squares of the blocks of f
with itself and of g with itself, which is the same as giving every field unit scale and
scaling nothing else.  `"none"` leaves the channels as they are.

The difference matters when the channels differ greatly in how much they carry.  With one
indicator per element the block of two rare elements holds a few pairs and the block of two
carbons holds most of the molecule; scaled per pair the rare block counts as much as the
carbon block and the kernel is mostly noise, which on the aromatic carbons of
`benchmarks/run_triangles.py` costs 0.4 ppm.  Per field, every element is given the same
weight as an atom and the pairs keep their proportions; that is the default.  For the
physico-chemical fields, whose units differ, per-field and per-pair scaling are close and
no scaling is worse.

### `triangle_scale(train: np.ndarray, spec: TriangleSpec, n_fields: int, scaling: str='field') -> np.ndarray`

The channel scales a training set gives, to apply to its test rows as well.

### `triangle_kernel(train: np.ndarray, test: Optional[np.ndarray]=None, *, spec: TriangleSpec=TriangleSpec(), n_fields: int, scale: Optional[np.ndarray]=None, scaling: str='field', power: int=1, normalize: bool=True) -> np.ndarray`

Cosine-normalised linear kernel on the three-distance features, to an integer power.

The blocks are scaled by `scaling` (see `block_scales`) unless `scale` is given, and
the normalisation is of the joint kernel.  `power` above one is the exponent SOAP calls zeta:
it sharpens the kernel, keeps it positive definite, and leaves no finite feature map, so the
primal route of `cmfqsar.locallarge` needs `power=1`.

### `triangle_kernel_matrix(envs: Sequence[MoleculeFields], spec: TriangleSpec=TriangleSpec(), n_fields: Optional[int]=None, power: int=1, chunk: int=500, scaling: str='field', progress=None) -> np.ndarray`

The kernel of a set of environments among themselves, in two passes over the set.

The same number as `triangle_kernel(triangle_feature_matrix(envs))`, to single precision,
without ever holding the features in double precision: the first pass collects the block
scales, the second builds the scaled and normalised rows in single precision, chunk by chunk,
and the kernel is their product.  With many fields and a long cutoff the feature matrix is
the largest thing in the computation, and this halves it.

### `triangle_labels(fieldset: FieldSet, spec: TriangleSpec=TriangleSpec()) -> List[str]`

One label per channel: the two field types, their two radii and the distance between.

### `with_weights(env: MoleculeFields, weights: np.ndarray, name: str='') -> MoleculeFields`

The same environment with other field weights, for channels that are not the fields.

The weights of an environment already carry the envelope factor of each atom; weights given
here are taken as they are, so a caller that wants the envelope multiplies it in
(`envelope_factors`).

### `envelope_factors(env: MoleculeFields, rho: float) -> np.ndarray`

The envelope factor `exp(-rho r^2)` of every atom of an environment, from its radius.

### `element_weights(env: MoleculeFields, elements: Sequence[str], rho: Optional[float]=None, other: bool=True) -> np.ndarray`

One indicator weight per element, which makes the descriptor SOAP's channels.

`elements` are the symbols that get a channel of their own; with `other`, every other symbol
shares one more.  With `rho` the envelope factor is multiplied in, as the field weights of an
environment carry it.

## cmfqsar.validation

Source: `cmfqsar/validation.py`

The three questions a 3D-QSAR model has to answer besides "how good is q²".

**Is it chance?**  A kernel built from molecular fields can fit anything if it is flexible enough,
and a congeneric series whose kernel values all exceed 0.97 is exactly the setting in which a
model can look respectable while carrying no information.  `y_scramble` shuffles the activities
and refits, many times, and reports where the real statistic falls in the distribution of
scrambled ones.  This is the check the literature calls y-randomisation, and the program had
everything needed for it except the loop.

**Does the prediction mean anything for *this* molecule?**  A kernel method always returns a
number, and for a molecule unlike anything in the training set that number is the training mean
presented as a prediction.  `domain` measures how close a molecule actually is to the training
set, in the only geometry the model knows -- the field kernel itself -- and compares that with how
close the training molecules are to each other.

**How certain is the statistic?**  A difference of 0.02 in q² between two settings is quoted all
over this repository's benchmarks, and it is worth knowing when such a difference is noise.
`bootstrap_ci` resamples the predictions, `bootstrap_refit` resamples the training set and
refits, and both return a percentile interval.

Nothing here needs the fields themselves, only kernels that have already been computed, so all of
it is cheap compared with the fit it validates.

### `class ScrambleResult  (dataclass)`

Where the real statistic falls among the ones obtained from shuffled activities.

Attributes:

- `statistic: str` = `'q2'`
- `observed: float` = `float('nan')`
- `scrambled: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`
- `runs: int` = `0`
- `reselected: bool` = `False`

#### `ScrambleResult.mean` -> float  (property)

#### `ScrambleResult.sd` -> float  (property)

#### `ScrambleResult.p_value` -> float  (property)

Fraction of shuffles that did as well as the real activities, with the usual +1.

The +1 in numerator and denominator is the standard correction for a permutation test:
with 20 shuffles the smallest reportable p is 1/21, never 0, because 20 shuffles cannot
demonstrate more than that.

#### `ScrambleResult.z_score` -> float  (property)

How many standard deviations above the scrambled mean the real statistic lies.

#### `ScrambleResult.caution() -> Optional[str]`

What the result warrants saying, which is not always what the p-value suggests.

A permutation test cannot report a p below 1/(runs+1), so a few shuffles produce
a large p even for a model that no shuffle came close to.  Saying "not distinguishable
from chance" there would be a false alarm; the correct message is that too few shuffles
were run to say.

#### `ScrambleResult.describe() -> str`

### `class DomainResult  (dataclass)`

How close each scored molecule is to the training set, in the model's own geometry.

Attributes:

- `similarity: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`
- `mean_top: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`
- `distance: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`
- `threshold: float` = `float('nan')`
- `k: int` = `3`

#### `DomainResult.inside` -> np.ndarray  (property)

#### `DomainResult.describe() -> str`

### `class BootstrapResult  (dataclass)`

A percentile confidence interval for one statistic.

Attributes:

- `statistic: str` = `'q2'`
- `value: float` = `float('nan')`
- `lo: float` = `float('nan')`
- `hi: float` = `float('nan')`
- `level: float` = `0.95`
- `runs: int` = `0`
- `refit: bool` = `False`
- `samples: np.ndarray` = `field(default_factory=lambda: np.zeros(0))`

#### `BootstrapResult.describe() -> str`

### `y_scramble(score: Callable[[np.ndarray], float], y: Sequence[float], runs: int=20, seed: int=0, statistic: str='q2', reselected: bool=False, progress: Optional[ProgressFn]=None, cancel: Optional[CancelFn]=None) -> ScrambleResult`

Score the real activities, then `runs` shuffles of them, with the same procedure.

`score` is whatever the caller counts as the model's figure of merit: a closure that takes an
activity vector and returns a number, larger being better.  Pass one that only refits at fixed
hyper-parameters and the test asks whether the *fit* can be achieved by chance; pass one that
repeats the grid search and it asks the stricter question, whether the whole procedure --
selection included -- can be.  `reselected` only records which of the two was done, for the
report.

### `scramble_kernel(K: np.ndarray, y: Sequence[float], method: str='krr', params: Optional[Dict[str, float]]=None, folds: int=10, cv_seed: Optional[int]=0, runs: int=20, seed: int=0, progress: Optional[ProgressFn]=None, cancel: Optional[CancelFn]=None) -> ScrambleResult`

y-randomisation of a cross-validated model on a precomputed kernel.

A ranking model is scrambled on Spearman's rho, since its score has no units and the q2 of
one would be a statistic about nothing -- shuffled or not.

### `domain_threshold(K_train: np.ndarray, k: int=3, quantile: float=0.05) -> float`

The similarity below which a molecule is outside the domain, taken from the training set.

Every training molecule is compared with the others, its mean similarity to its `k` nearest
neighbours is taken, and the `quantile` point of that distribution becomes the threshold.  The
domain is then defined by the training set's own spread rather than by a number chosen
arbitrarily: a series of close analogues gets a tight domain, a diverse set a loose one.

### `domain(K_test_train: np.ndarray, k_test_self: Sequence[float], K_train: np.ndarray, k: int=3, quantile: float=0.05, threshold: Optional[float]=None) -> DomainResult`

Applicability domain of a set of molecules against the training set of a fitted model.

Everything is computed from kernels that the prediction needed anyway: the cross kernel
between the scored molecules and the training set, the scored molecules' self-kernels, and the
training kernel matrix.  Distances are the field-space distances the kernel implies,
`|X - X_j|^2 = K(x,x) + K(j,j) - 2K(x,j)`, which is the same quantity the Gaussian kernel
layer is built on.

### `bootstrap_ci(y: Sequence[float], y_pred: Sequence[float], statistic: str='q2', runs: int=1000, seed: int=0, level: float=0.95, y_ref_mean: Optional[float]=None) -> BootstrapResult`

Percentile interval for q2, R2pred or RMSE, by resampling the (observed, predicted) pairs.

This is the uncertainty of the *statistic* given these predictions -- how much of the number
is the particular molecules it was computed over -- and it is what tells you whether two
settings that differ by 0.02 in q2 differ at all.  It does not refit anything, so it does not
capture the variability of the fit itself; `bootstrap_refit` does that.

### `bootstrap_refit(K: np.ndarray, y: Sequence[float], method: str='krr', params: Optional[Dict[str, float]]=None, runs: int=100, seed: int=0, level: float=0.95, fit_bias: bool=True, progress: Optional[ProgressFn]=None, cancel: Optional[CancelFn]=None) -> BootstrapResult`

Cramer's bootstrap: resample the training set with replacement, refit, score what was left.

Each resample fits on the molecules drawn and predicts the ones that were not (about a third
of the set), so the statistic is measured out of bag rather than on the molecules the model
just saw.  The kernel is never recomputed -- a resample is an index into the matrix -- so the
cost is one linear solve per run.

### `scramble_selection(kernels: Sequence, y: Sequence[float], method: str='krr', param_grid: Sequence[Dict[str, float]]=({'lam': 1.0},), folds: int=10, cv_seed: Optional[int]=0, runs: int=20, seed: int=0, progress: Optional[ProgressFn]=None, cancel: Optional[CancelFn]=None) -> ScrambleResult`

y-randomisation with the model selection repeated inside every shuffle.

`kernels` is a sequence of precomputed joint kernels -- one per attenuation, say -- and
`param_grid` the regularisations tried with each.  Every shuffle is scored by the *best* q2
over that grid, exactly as the real activities were, so the test asks whether the whole
procedure can reach the reported number by chance rather than whether a single fit can.  That
is the stricter and fairer question; it costs one cross-validation per grid point per
shuffle, which is affordable only because the kernels do not depend on the activities and are
therefore computed once.

## cmfqsar.varalign

Source: `cmfqsar/varalign.py`

Alignment by prediction confidence: the pose of a molecule is chosen so that the
posterior variance of its predicted activity is as small as possible.

This implements the idea put forward in the Conclusions of Baskin & Zhokhova,
Future Med. Chem. (2019): kernel probabilistic regression provides the posterior
variance of every prediction, that variance depends sharply on the position and
orientation of the molecule, and molecules can therefore be aligned by maximising
the confidence of their prediction, i.e. by moving them into the applicability
domain of the 3D-QSAR model.

    sigma^2(x) = K(x, x) - k(x)^T (K + lambda I)^-1 k(x)

K(x, x) depends only on the interatomic distances inside the molecule, so it is
invariant under rigid motion.  For a rigid-body search, minimising the variance is
therefore exactly maximising the quadratic form k(x)^T (K + lambda I)^-1 k(x):
similarity to the whole training set, with redundant training molecules down-weighted
by the inverse kernel matrix.  No activity value of the aligned molecule is used, so
the procedure is legitimate for prospective prediction.

Two limitations are built into the method and are worth remembering.

1. It aligns molecules against a *fixed* training alignment.  Applied to the training
   set itself the criterion is degenerate: superimposing all molecules maximises mutual
   similarity and drives every variance to zero, while destroying the structure-activity
   signal.  Use a chemically motivated alignment (maximum common substructure) for the
   training set and this method for new molecules.
2. Because the pose is chosen to make the model confident, the posterior variance of an
   aligned molecule is no longer an independent measure of the applicability domain.

For a search over conformations (rather than rigid motion) use normalised kernels:
K(x, x) is then constant, otherwise the variance could be lowered by changing the
conformation rather than by improving the fit to the training set.

### `class VarianceAlignResult  (dataclass)`

Outcome of `VarianceAligner.align` for one molecule.

Attributes:

- `name: str`
- `coords: np.ndarray`
- `variance_before: float`
- `variance_after: float`
- `prediction_before: float`
- `prediction_after: float`
- `rmsd: float`
- `n_eval: int` = `0`
- `limited: bool` = `False`

#### `VarianceAlignResult.improved` -> bool  (property)

### `class VarianceAligner(model, max_rmsd: Optional[float]=2.0, penalty: float=50.0)`

Rigid-body alignment of molecules by minimising the posterior variance of a CMF model.

The model must be fitted; its training poses define the frame that everything is
aligned to.  Analytic gradients are used (L-BFGS) with optional random restarts.

Parameters
----------
model : fitted `cmfqsar.model.CMFModel`
max_rmsd : maximum displacement (A) from the starting pose.  Beyond it a quadratic
    penalty is applied, which keeps a molecule near its chemically motivated
    (e.g. maximum-common-substructure) pose.  `None` disables the guard.
penalty : weight of that penalty.

#### `VarianceAligner.variance(mol: MoleculeFields, coords: Optional[np.ndarray]=None) -> float`

Posterior variance of the prediction for `mol` in the given pose.

#### `VarianceAligner.predict(mol: MoleculeFields, coords: Optional[np.ndarray]=None) -> float`

#### `VarianceAligner.objective(mol: MoleculeFields, X0: np.ndarray, c: np.ndarray, sp: np.ndarray, kself: float, p: np.ndarray)`

Variance and its analytic gradient for the rigid-body parameters
`p = (rotation vector, translation)` applied about the centroid `c`.

#### `VarianceAligner.align(mol: MoleculeFields, restarts: int=2, seed: int=0, max_iter: int=200, apply: bool=True, any_orientation: bool=False) -> VarianceAlignResult`

Find the rigid-body pose of `mol` that minimises the posterior variance.

A restart turns the arriving pose by at most about 63 degrees, which is what a molecule
already placed in the site needs.  With `any_orientation` every restart draws a uniformly
random orientation instead: a pose whose orientation is unknown cannot be found by
perturbing the wrong one, however many restarts are spent on it.

### `class CVAlignRound  (dataclass)`

Diagnostics of one round of `cv_align_training_set`.

Attributes:

- `round: int`
- `moved_mean: float`
- `moved_max: float`
- `drift: float`
- `similarity: float`

#### `CVAlignRound.describe() -> str`

### `variance_align(mols: Sequence[MoleculeFields], model, max_rmsd: Optional[float]=2.0, restarts: int=2, seed: int=0, any_orientation: bool=False, progress: Optional[Callable[[int, int], None]]=None, cancel: Optional[Callable[[], bool]]=None) -> List[VarianceAlignResult]`

Align every molecule of `mols` in place by minimising the model's posterior variance.

### `ensemble_similarity(mols: Sequence[MoleculeFields], specs) -> float`

Mean off-diagonal normalised kernel of a set of molecules.

It measures how alike the aligned molecules have become.  A value drifting towards 1
means the ensemble is contracting and the kernel matrix is losing its ability to
discriminate between compounds.

### `cv_align_training_set(mols: Sequence[MoleculeFields], y: np.ndarray, model_template, folds: int=5, rounds: int=1, max_rmsd: Optional[float]=1.0, restarts: int=0, seed: int=0, progress: Optional[Callable[[str, float], None]]=None, cancel: Optional[Callable[[], bool]]=None)`

Align a *training* set with the posterior-variance criterion, cross-validated.

In every fold the held-out molecules are aligned against a model fitted to the remaining
ones, so no molecule is ever aligned against a model that contains it.  Repeating over the
folds gives a self-consistent alignment of the whole set.  The molecules are modified in place.

Measured behaviour (thrombin, glycogen phosphorylase and two automatically aligned sets):
the iteration converges and the contraction of the ensemble is mild, but the predictive
gain is not reliable - one set improved, one degraded, two were neutral - and rounds after
the first consistently made the external prediction worse.  Hence the defaults of a single
round and a displacement guard.

Two warnings that the caller should pass on to the user:

* q^2 of the realigned set is **not** a valid criterion here.  Realigning the training set
  changes the geometry q^2 is computed on, and in the tests q^2 and the external R^2
  repeatedly moved in opposite directions.  Judge the result on held-out data only.
* the hyper-parameters were selected on the previous alignment and should be refitted.

Returns
-------
list of `CVAlignRound`

## cmfqsar.vasp

Source: `cmfqsar/vasp.py`

Per-atom arrays from a VASP run, read into an ASE structure (plan item M3).

Every materials study so far had to script the same step by hand: take the Bader charges of a
run, subtract the valence electrons, and put the result on the atoms as an array the periodic
field builder can read.  This module does that once.  It reads the three files a run leaves
behind and returns an ASE `Atoms` whose `arrays` carry what the fields need:

* `CONTCAR` (or `POSCAR`): the structure, with the species in the order VASP wrote them.
* `ACF.dat` of the Bader program (Henkelman group): the electrons each atom holds, in the
  column `CHARGE`.  A *net* charge is the valence electrons of the pseudopotential minus that
  number, so the Bader file alone is not enough.
* `OUTCAR`: the valence electrons per species (`ZVAL`), the ions per species, and, for a
  spin-polarised run, the magnetic moment per atom (the last `magnetization (x)` block).

The result carries `bader_charges` (net Bader charges, also set as the atoms' initial charges,
both of which `cmfqsar.periodic.crystal_charges` reads with the charge model `"file"`),
`bader_electrons` (the raw column), and `magnetic_moment` when the run was spin-polarised.
Any of them can then be a field as `prop:<name>`, and the electrostatic field of `cmfqsar.periodic.inorganic_fieldset` with `charge_model="file"`
takes the net Bader charges in place of the oxidation-state guess.

A file that is missing is reported by name; a file whose atom count does not match the structure
is an error naming both counts, because the alternative is to put a charge on the wrong atom.

### `read_acf(path: str) -> np.ndarray`

The `CHARGE` column of a Bader `ACF.dat`: electrons per atom, in atom order.

### `read_outcar_species(path: str) -> Dict[str, Sequence]`

What the OUTCAR says about the species: symbols, ions per type and valence electrons.

Returns `{"symbols": [...], "counts": [...], "zval": [...]}` with one entry per species in
the order VASP uses.  The symbols come from the `VRHFIN` lines of the pseudopotentials, the
counts from `ions per type` and the valence electrons from the `ZVAL` line that lists
them all at once (the per-pseudopotential `ZVAL` lines are used when that line is absent).

### `read_outcar_magmoms(path: str) -> Optional[np.ndarray]`

The per-atom magnetic moments of the last `magnetization (x)` block, or None.

### `read_outcar_energy(path: str) -> Optional[float]`

The last `energy  without entropy` of an OUTCAR, in eV, or None.

### `net_bader_charges(electrons: np.ndarray, symbols: Sequence[str], zval: Dict[str, float]) -> np.ndarray`

Valence electrons of each atom's species minus the electrons Bader gave it.

### `structure_with_arrays(run_dir: str, structure: str='CONTCAR', zval: Optional[Dict[str, float]]=None, require_bader: bool=True)`

An ASE `Atoms` of a VASP run with its per-atom arrays attached.

`run_dir` holds the run; `structure` names the file the geometry is read from (`CONTCAR`
by default, `POSCAR` for the input geometry).  `zval` overrides the valence electrons per
species, which are otherwise read from the OUTCAR.  With `require_bader` a run without an
`ACF.dat` is an error; without it the structure is returned with whatever arrays exist.

The arrays set are `bader_charges` (net Bader charges, also the initial charges of the
atoms, so that they survive a write to extended XYZ), `bader_electrons`, and
`magnetic_moment` when the OUTCAR holds a magnetisation block; `atoms.info["vasp_energy"]` holds the final
energy.

### `write_structure(atoms, path: str) -> None`

Write the structure with its arrays as extended XYZ, which keeps every per-atom array.

The `bader_charges` array and the initial charges both come back from such a file, and
`cmfqsar.periodic.crystal_charges` reads either under the charge model `"file"`.

### `describe_arrays(atoms) -> str`

One line per per-atom array: its name, mean and range.

## cmfqsar.xtb

Source: `cmfqsar/xtb.py`

A cheap quantum stage: per-atom properties from GFN2-xTB (plan item D3).

`cmfqsar/orca.py` brings quantum-chemical per-atom quantities into the program as `prop:`
fields, and it works, but a density-functional single point takes two to three minutes per
molecule on this machine.  That is affordable for one series and not for a collection.  GFN2-xTB
is a semi-empirical tight-binding method: it solves a parameterised one-electron problem instead
of the Kohn-Sham equations, so a molecule of forty atoms takes a fraction of a second, including
the two ionic states a Fukui index needs.  The question the plan asks is not whether the cheap
method is as good as the expensive one -- it is not -- but whether it is good enough to say, over
a whole collection of series, which series a quantum field helps at all.

What it computes, in **one** call of the program per molecule (`--gfn 2 --vfukui`):

=====================  ===============================================================
`q_xtb`              the GFN2 partial charge of the atom
`fukui_plus`         f+, the charge the atom gains when an electron is added
`fukui_minus`        f-, the charge it loses when an electron is removed
`fukui_zero`         f0, the mean of the two
`dual`               f+ - f-, positive where the atom is electrophilic
`alpha_xtb`          the atomic polarisability alpha(0), in atomic units
=====================  ===============================================================

and per molecule the total energy, the frontier orbital energies, their gap and the dipole.

**The Fukui convention is the same as ORCA's**, which is what makes the two engines
interchangeable in a `prop:` field name.  `cmfqsar/orca.py` forms `f+ = q(N) - q(N+1)` from
Hirshfeld charges; xtb prints f+ directly, and `tests/test_xtb.py` checks on water that the
printed value is the same difference of the program's own charges to three decimals.  What
differs is the population analysis behind the charges -- Hirshfeld there, a Mulliken-like
tight-binding population here -- so `q_xtb` is **not** `q_hirshfeld` and carries a name of its
own.  A model fitted on one must be predicted with the same one.

Three things are built in rather than left to the caller, as in `cmfqsar/orca.py`.  Every job is
**cached** by a key over the molecule, its charge, the method and the geometry, so an interrupted
run resumes.  A molecule that fails is **reported and skipped**, never fatal to the set.  And the
geometry is **not optimised** unless asked: a superimposed series optimised molecule by molecule
is no longer superimposed.

What this module does not provide is the ionisation potential and the electron affinity, and so
not the hardness, the softness or the electrophilicity index that `cmfqsar/orca.py` derives from
them.  The energies xtb prints for the two ionic states of a Fukui run are electronic energies of
the self-consistent charge step alone, not total energies, so a difference against the neutral's
total energy would not be an ionisation potential.  xtb computes those properly with `--vipea`,
which needs a different Hamiltonian (GFN1 with the ipea parameters); mixing a GFN1 descriptor with
GFN2 fields would describe one molecule with two methods, so it is left out.

The command-line front end is `examples/xtb_atom_properties.py`.

### `class XtbError  — bases: RuntimeError`

### `class XtbResult  (dataclass)`

What one xtb call reports; the Fukui entries are None without `--vfukui`.

Attributes:

- `symbols: List[str]`
- `charges: np.ndarray`
- `polarisability: np.ndarray`
- `energy: float` = `float('nan')`
- `homo: float` = `float('nan')`
- `lumo: float` = `float('nan')`
- `gap: float` = `float('nan')`
- `dipole: float` = `float('nan')`
- `fukui_plus: Optional[np.ndarray]` = `None`
- `fukui_minus: Optional[np.ndarray]` = `None`
- `fukui_zero: Optional[np.ndarray]` = `None`
- `version: str` = `''`

#### `XtbResult.n_atoms` -> int  (property)

### `class Config  (dataclass)`

Attributes:

- `xtb: str` = `''`
- `gfn: int` = `DEFAULT_METHOD`
- `opt: bool` = `False`
- `fukui: bool` = `True`
- `solvent: str` = `''`
- `accuracy: float` = `1.0`
- `timeout: Optional[float]` = `600.0`
- `workdir: str` = `'xtb_work'`
- `cancel: Optional[object]` = `None`

#### `Config.level() -> str`

One string naming the method, for the cache key and the provenance of the file.

### `class MoleculeOutcome  (dataclass)`

Attributes:

- `index: int`
- `name: str`
- `mol: Optional[Chem.Mol]` = `None`
- `error: str` = `''`
- `seconds: float` = `0.0`
- `reused: bool` = `False`

### `find_xtb(explicit: Optional[str]=None) -> str`

The xtb executable: the one given, then `XTB_EXE`, then the path, then the default.

### `xtb_version(exe: str) -> str`

The version string the program reports, for the provenance of a result file.

### `parse_output(text: str) -> XtbResult`

Read the quantities this module uses out of an xtb run.

The partial charges are taken from the per-atom table of the output rather than from the
`charges` file, so that a result can be parsed from the text alone; `run_molecule` prefers
the file, which carries more digits.

### `derive_properties(res: XtbResult, charge: int=0, mult: int=1) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]`

(per-atom, per-molecule) properties from one result, named as `cmfqsar/orca.py` names them.

`dual` is f+ - f-, the dual descriptor: positive where an atom takes an electron more
readily than it gives one.  The Fukui entries are absent when the run did not compute them.

### `command(cfg: Config, xyz: str, charge: int, mult: int) -> List[str]`

The argument list for one molecule.

### `run_job(jobdir: str, cfg: Config, charge: int, mult: int) -> XtbResult`

Run xtb in `jobdir` on `geom.xyz`, or reuse the output already there.

### `run_molecule(mol: Chem.Mol, name: str, index: int, cfg: Config) -> MoleculeOutcome`

The whole chain for one molecule: geometry, the xtb call, parsing, the annotated copy.

### `process_molecules(mols: Sequence[Chem.Mol], names: Sequence[str], cfg: Config, jobs: int=1, log=print) -> List[MoleculeOutcome]`

Run every molecule and return the outcomes, in the order of the input.

### `process_dataset(path: str, out: str, cfg: Config, limit: Optional[int]=None, jobs: int=1, log=print) -> List[MoleculeOutcome]`

Run every molecule of `path` and write the annotated SDF `out`; returns the outcomes.

## cmfqsar.gui

Source: `cmfqsar/gui/__init__.py`

PySide6 desktop application for CMF 3D-QSAR.

## cmfqsar.gui.app

Source: `cmfqsar/gui/app.py`

Application entry point.

### `main(argv=None)`

## cmfqsar.gui.copattern_panel

Source: `cmfqsar/gui/copattern_panel.py`

The co-pattern chart: weight against distance for one pair of field types (plan item NT9).

The invariant kernels have no co-field, so there is no map to look at.  What they have is a
weight per channel, and a channel is one pair of field types at one distance.  A ranked list of
twenty channels out of several thousand is a table; the same numbers for one pair of field
types, drawn against distance, are a shape a reader can take in at once.
`cmfqsar.copattern` computes the curve and this widget draws it.

The chart is used in both windows, because both have kernels without a co-field: the molecular
window (the autocorrelation kernel of a set with no common frame, and the surface kernel of an
adsorbate) and the crystal window (the periodic autocorrelation of a cell).

### `class CoPatternChart(parent=None)  — bases: QWidget`

A chart of channel weight against distance, with the pair of field types to choose from.

Call `set_model` after a fit and `clear` when there is nothing to show.  The
widget keeps the model, so changing the pair or the quantity on the y axis costs nothing.

#### `CoPatternChart.clear(message: str='') -> None`

Empty the chart and say why it is empty.

#### `CoPatternChart.set_model(model, fieldset=None) -> bool`

Show the co-pattern of `model`.  Returns False, and says why, when it has none.

#### `CoPatternChart.current_pair() -> int`

#### `CoPatternChart.current_value() -> str`

## cmfqsar.gui.crystal_window

Source: `cmfqsar/gui/crystal_window.py`

A periodic structure as the sample, in the application (plan item M13).

Everything the program can do with crystals -- the periodic field autocorrelation, the periodic
overlap kernel with its co-field inside the cell, the split of a prediction over the sites --
lived on the command line and in the library only.  This window is that side of the program on
screen: open a folder of CIF files or one structure file, give the structures their measured
values, fit a model, and read it on the cell itself.

What is drawn, and why it is not the molecular view
---------------------------------------------------
A crystal is not a molecule with a box around it.  It has no bond table, it has no single
"molecule" to centre on, and half of what the kernel sees lies outside the cell, in the periodic
images.  So the view has three parts the molecular view does not:

* **the cell**, as the twelve edges of its box, with the three lattice vectors marked;
* **the images**, every copy of the cell's atoms that comes within the kernel's own cutoff, drawn
  as dots so that thousands of them cost nothing.  They are what makes the periodic kernel
  periodic, and seeing them is how one checks that the cutoff reaches as far as it should;
* **the reading**: the co-field inside the cell where the kernel has one, and one sphere per site
  coloured by what that site adds to the prediction.

Which reading applies depends on the kernel, and the window says so rather than offering both:

* the **periodic overlap kernel** compares two cells point by point, so it has a co-field --
  C(r) inside the cell, positive where more field raises the property -- and the prediction
  splits over the sites by decomposition rule 3;
* the **periodic field autocorrelation** is a distribution of distances and can never say
  *where*, so it has no co-field.  Its reading is the co-pattern over the channels (which pair
  of field types at which distance) and the per-site split of plan item M5, which is exact.

### `class ColorScale(parent=None)  — bases: QWidget`

The colour scale of the site contributions, with the value at each end written on it.

A colouring with no scale beside it says only which site is red; with the scale it says how
much, in the units of the activity, which is what makes it a reading rather than a picture.

#### `ColorScale.set_range(vmax: float, unit: str='') -> None`

Show ±`vmax`; a scale of zero means nothing is drawn and the labels say so.

### `class CrystalWindow(parent=None, background: bool=True)  — bases: QMainWindow`

A data set of periodic structures: open it, fit a model, read it on the cell.

#### `CrystalWindow.log(text: str) -> None`

#### `CrystalWindow.error(text: str) -> None`

#### `CrystalWindow.run_task(fn, on_done, label='Working')`

#### `CrystalWindow.cancel_worker()`

#### `CrystalWindow.open_folder()`

#### `CrystalWindow.open_file()`

#### `CrystalWindow.load(path: str, target_file: Optional[str]=None, target_key: Optional[str]=None, target_column: Optional[str]=None)`

Read a folder of structure files, or one file, and show what came back.

#### `CrystalWindow.import_targets(path: str='', column: Optional[str]=None)`

Read the measured values from a CSV whose first column holds the structure names.

A table with several numeric columns -- which is what a prepared data set usually is --
asks which column is the property; a plain `name,value` file asks nothing.

#### `CrystalWindow.populate() -> None`

#### `CrystalWindow.current_index() -> int`

#### `CrystalWindow.build_model()`

#### `CrystalWindow.update_reading(*_)`

#### `CrystalWindow.compute_cofield()`

Evaluate the co-field of the fitted model inside the selected cell.

#### `CrystalWindow.center_current()`

#### `CrystalWindow.update_display(*_)`

#### `CrystalWindow.save_model_file()`

#### `CrystalWindow.export_predictions()`

#### `CrystalWindow.save_screenshot()`

### `cell_grid(crystal, spacing: float=0.3, margin: float=3.0, max_points: int=MAX_GRID_POINTS) -> Grid`

A rectangular grid over the box one cell spans, cut to the atoms where there is vacuum.

The grid is axis-aligned, as every grid in this program is, so for a cell that is not
rectangular it covers a little more than the cell.  That costs nothing to look at: the
co-field of a periodic model is itself periodic, so the extra corners carry the field of the
neighbouring cell and not noise.

Where the structure is *not* periodic -- the vacuum above a layer, which is most of its cell
-- the grid is cut to the atoms plus `margin`, because there is no field out there and the
points would only be spent.  The spacing is then coarsened until the grid fits in
`max_points`, so a large supercell is drawn more coarsely rather than not at all.

### `site_contributions_of(model: CMFModel, crystal) -> Tuple[np.ndarray, str]`

One number per site of `crystal`, and one line saying what the numbers are.

The two periodic kernels split a prediction in two different ways, and both are exact:
the overlap kernel over the atoms of the cell (decomposition rule 3), the autocorrelation
over the sites of the cell (plan item M5).  The sum of the numbers is the prediction less
the bias either way.

### `reading_rows(model: CMFModel, fieldset: FieldSet, n: int=20) -> Tuple[List, List]`

The table under the view: the co-pattern of the model, or the sites of one cell.

Returns `(column titles, rows)`.  For the autocorrelation kernel the co-pattern over the
channels is the model's own reading and does not depend on which structure is selected; the
per-site split does, so both are offered and the window shows whichever the user asks for.

### `site_rows(values: Sequence[float], crystal, n: int=20) -> Tuple[List, List]`

The sites of one cell that move its prediction most, largest magnitude first.

## cmfqsar.gui.main_window

Source: `cmfqsar/gui/main_window.py`

Main window of the CMF 3D-QSAR desktop application.

### `class AlignDialog(names: List[str], current: int, parent=None)  — bases: QDialog`

### `class VarianceAlignDialog(n_test: int, n_sel: int, parent=None)  — bases: QDialog`

Options for aligning molecules by the posterior variance of their prediction.

#### `VarianceAlignDialog.max_rmsd()`

#### `VarianceAlignDialog.any_orientation() -> bool`

#### `VarianceAlignDialog.flexible() -> bool`

#### `VarianceAlignDialog.conformers() -> int`

#### `VarianceAlignDialog.energy_slack() -> float`

### `class TrainAlignDialog(n_train: int, parent=None)  — bases: QDialog`

Options for the self-consistent (cross-validated) alignment of the training set.

#### `TrainAlignDialog.max_rmsd()`

### `class GTMDialog(n_mol: int, parent=None)  — bases: QDialog`

Options for the chemical space map.

### `class ChemotypeDialog(n_mol: int, parent=None)  — bases: QDialog`

Options for grouping a heterogeneous series by scaffold.

### `class PlaceChemotypesDialog(names, has_protein: bool, parent=None)  — bases: QDialog`

Options for moving each chemotype, as a rigid body, into a common frame.

#### `PlaceChemotypesDialog.mode() -> str`

### `class ChemotypeValidationDialog(n_groups: int, parent=None)  — bases: QDialog`

Options for leave-one-chemotype-out validation and the alignment-free baseline.

### `class TopomerDialog(n_mol: int, parent=None)  — bases: QDialog`

Options for replacing the molecules by their substituents, posed by rule.

#### `TopomerDialog.core()`

(mol, description) of the core to cut from, or (None, 'mcs').

### `class FieldDifferenceDialog(names, current: int, parent=None)  — bases: QDialog`

Choose the molecule a candidate is meant to imitate.

### `class PharmacophoreDialog(n_fields: int, has_protein: bool, parent=None)  — bases: QDialog`

Options for discretising the co-fields into typed features.

### `class DecoyAlignDialog(n_decoys: int, n_train: int, parent=None)  — bases: QDialog`

Options for giving every background molecule its best pose against the training set.

### `class DockDialog(n_mol: int, n_conf: int, pocket_name: str, parent=None)  — bases: QDialog`

Options for placing molecules, or their conformations, inside the pocket.

### `class ConformerDialog(n_mol: int, n_have: int, parent=None, names=None)  — bases: QDialog`

Options for generating a conformational ensemble: the fourth dimension.

#### `ConformerDialog.alignment() -> str`

The conformer placement chosen, as `cmfqsar.conformers.CONFORMER_ALIGNMENTS`.

#### `ConformerDialog.reference_index()`

### `class UnalignDialog(n_all: int, n_train: int, n_test: int, n_sel: int, parent=None)  — bases: QDialog`

Options for destroying the alignment: give every molecule a random pose.

#### `UnalignDialog.max_rotation()`

### `class OrcaDialog(exe, groups, charges, mults, parent=None, workdir='orca_work', has_mpi=False)  — bases: QDialog`

Options for a quantum-chemical run: what ORCA computes, and for which molecules.

The two kinds of result go to the two places the model keeps them: a per-atom quantity (a
charge, a Fukui index) becomes the weights of a field, and a per-molecule quantity (the gap,
the hardness) becomes a descriptor beside the fields.

#### `OrcaDialog.indices() -> List[int]`

#### `OrcaDialog.level() -> str`

#### `OrcaDialog.workdir() -> str`

#### `OrcaDialog.timeout()`

### `class SelectivityDialog(props, current, parent=None)  — bases: QDialog`

Refit the current model as several endpoints, so that selectivity becomes a co-field.

A separation problem is two properties, and what matters is their difference.  Because a CMF
model is linear in the molecular field, the difference of two such models is itself a model
linear in the field -- so the difference has a co-field of its own, and it can be read, drawn
and exported exactly like any other.

Any number of endpoints can be fitted together.  With more than two, one shared correlation
assumes that every pair is related to the same degree, which is rarely true, so the covariance
can instead be a free-form low-rank matrix selected by the marginal likelihood.

#### `SelectivityDialog.endpoints()`

Every endpoint to fit, with the named pair first and second.

#### `SelectivityDialog.covariance()`

#### `SelectivityDialog.rhos()`

#### `SelectivityDialog.lams()`

### `class CompareModelsDialog(has_current: bool, parent=None)  — bases: QDialog`

Which two models to score on the loaded molecules (plan item G2).

#### `CompareModelsDialog.side(k: int)`

(source index, file path, label) of model A (0) or B (1).

### `class ComparisonWindow(cmp, parent=None)  — bases: QDialog`

The two models' predictions side by side, with the summary and a CSV export.

### `class SlabDialog(parent=None)  — bases: QDialog`

Build a body-centred cubic (110) slab to put behind the molecules (plan item EC10).

The defaults are the slab the Fe(110) reference poses of `examples/corrosion_fe110` were
computed on, so a user who opens that series and accepts the defaults sees the surface those
adsorption energies belong to.

#### `SlabDialog.values() -> dict`

### `class PlanePoseDialog(names: List[str], parent=None)  — bases: QDialog`

Pose the molecules in the surface frame, by rule (plan items EC2 and EC10).

#### `PlanePoseDialog.values() -> dict`

### `class MainWindow(background: bool=True)  — bases: QMainWindow`

#### `MainWindow.toggle_results_span(full_width: bool)`

Choose whether the bottom (Results) dock spans the whole width or only the central area.

Qt dock separators move only perpendicular to themselves, so this switch decides which
borders exist: with full width, the border between the side panels and Results moves
vertically; otherwise the side panels run the full height.

#### `MainWindow.restore_layout()`

#### `MainWindow.log(text: str)`

#### `MainWindow.error(text: str)`

Report a failure: a dialog in the application, the log alone in a script.

A modal dialog waits for a person, so a script or a test driving this window
(`background=False`) must never raise one.

#### `MainWindow.run_task(fn, on_done, label='Working')`

#### `MainWindow.cancel_worker()`

#### `MainWindow.schedule_display()`

#### `MainWindow.open_dataset(kind: str='train')`

#### `MainWindow.load_dataset_file(path: str, kind: str='train', activity: Optional[str]=None, on_loaded=None)`

#### `MainWindow.open_crystal_window()`

Open the window for periodic structures (plan item M13).

A crystal is not a molecule: it has a lattice instead of a pose, no bond table, and its
kernel sums over the images of the cell.  So the periodic side of the program has a
window of its own rather than a mode of this one, and the two do not share a data set.

#### `MainWindow.activity_changed(ix)`

#### `MainWindow.import_activities()`

#### `MainWindow.random_split()`

#### `MainWindow.set_subset(which: str)`

#### `MainWindow.refresh_fieldset()`

#### `MainWindow.build_model()`

#### `MainWindow.run_nested_cv()`

#### `MainWindow.predict_test()`

#### `MainWindow.chemotypes_dialog()`

Step 1: group the molecules by their Bemis-Murcko framework.

#### `MainWindow.align_within_chemotypes()`

Step 2: superimpose the molecules of each chemotype on its largest member.

#### `MainWindow.place_chemotypes_dialog()`

Step 3: move each chemotype, as a rigid body, into one frame.

#### `MainWindow.chemotype_validation_dialog()`

Step 4: hold out a whole chemotype, and compare with a model that ignores the poses.

#### `MainWindow.generate_conformers_dialog()`

#### `MainWindow.dock_dialog()`

Place every conformation of every molecule in the loaded pocket.

#### `MainWindow.load_protein_file()`

#### `MainWindow.fetch_protein()`

#### `MainWindow.clear_protein()`

#### `MainWindow.load_surface_file(path: str='')`

Read a slab from a structure file and put it behind the molecules.

#### `MainWindow.build_slab_dialog()`

#### `MainWindow.clear_surface()`

#### `MainWindow.plane_pose_dialog()`

#### `MainWindow.pose_on_plane(pose: str='flat', height: float=2.2, anchor_height: float=2.0, ref_index: Optional[int]=None, refine: Optional[str]=None)`

Pose every molecule on the plane z = 0, by rule, and report the poses.

This is the frame the surface kernel and the overlap kernel of an adsorbate need: the
normal along z, the top atomic layer at z = 0.  The molecules are moved rigidly, so the
step joins the pose history and can be undone like any other alignment.

#### `MainWindow.compute_complementarity()`

Score every molecule against the pocket, and the co-fields against what it offers.

#### `MainWindow.compute_shapley()`

Decompose the prediction for the selected molecule by Shapley values.

Unlike the co-field and the atom contributions this works for any kernel, so it is the
interpretation available when a nonlinear layer is used.

#### `MainWindow.compute_cofields()`

#### `MainWindow.cofield_noise_toggled(on: bool)`

Compute the error bar of every co-field, once, when the user asks to hide the noise.

It costs about what the co-fields themselves cost, so it is not computed until it is
wanted; unticking the box keeps what was computed, so the two pictures can be compared
by clicking back and forth.

#### `MainWindow.update_display()`

#### `MainWindow.center_current()`

#### `MainWindow.variance_align_dialog()`

Align molecules by minimising the posterior variance of the current model.

#### `MainWindow.clear_pharmacophore() -> None`

Take the features off the view and forget them: they belong to one fitted model.

#### `MainWindow.topomer_dialog()`

Replace the molecules by their substituents, each posed by the topomer rule.

#### `MainWindow.field_difference_dialog()`

Compare the selected molecule's fields with those of a reference molecule.

#### `MainWindow.pharmacophore_dialog()`

Discretise the co-fields of the fitted model into typed features with tolerances.

#### `MainWindow.gtm_map_dialog()`

Chemical space map and activity landscape from the CMF kernels (kernel PCA + GTM).

#### `MainWindow.align_decoys_dialog()`

Give every background molecule its best pose with respect to the training set.

#### `MainWindow.orca_dialog()`

Compute quantum-chemical properties with ORCA: per-atom into fields, per-molecule into descriptors.

#### `MainWindow.save_annotated_sdf()`

Write the molecules with every property they carry, atom property lists included.

#### `MainWindow.selectivity_dialog()`

Refit the current model as two endpoints and offer their difference as a co-field.

#### `MainWindow.endpoint_changed()`

Point the co-field machinery at one endpoint, or at the difference of the two.

#### `MainWindow.unalign_dialog()`

Give the chosen molecules a random pose (control experiment for alignment methods).

#### `MainWindow.train_align_dialog()`

Self-consistent alignment of the training set by the posterior-variance criterion.

#### `MainWindow.align_dataset_dialog()`

#### `MainWindow.undo_alignment()`

Put the molecules back where they were before the last step that moved them.

#### `MainWindow.save_project_file(save_as: bool=False)`

Write the session: molecules and poses, pose history, settings, model, results.

#### `MainWindow.open_project_file()`

#### `MainWindow.load_project_file(path: str, on_loaded=None)`

#### `MainWindow.compare_models_dialog()`

Two models, one set of molecules: the predictions side by side (plan item G2).

#### `MainWindow.save_model_file()`

#### `MainWindow.load_model_file()`

#### `MainWindow.export_predictions()`

#### `MainWindow.export_sdf()`

#### `MainWindow.export_cube()`

#### `MainWindow.export_shapley_cube()`

Write the Shapley field of the selected molecule, so it can be opened with the protein.

#### `MainWindow.save_screenshot(path: Optional[str]=None)`

#### `MainWindow.show_about()`

#### `MainWindow.closeEvent(ev)`

#### `MainWindow.restore_saved_layout()`

Restore the window geometry and dock arrangement of the previous session, if any.

### `molfield_wireframe(choice: str, cofield_drawn: bool) -> bool`

Whether the molecular field is drawn as a mesh rather than a solid surface.

The two maps share a colour scale -- positive and negative in the field's own two colours --
so when both are on screen only the drawing style tells them apart, and the mesh has to be
the molecular field: it is one connected envelope around the whole molecule, and drawn solid
it would hide the molecule and the co-field lobes alike.  Alone on screen it has nothing to
hide and nothing to be confused with, so it is drawn solid.  That is `auto`; a reader who
wants the opposite picture -- lobes as a cage around a solid field -- names the style.

## cmfqsar.gui.panels

Source: `cmfqsar/gui/panels.py`

Dock panels of the CMF application: data set table, model settings, results, viewer controls.

### `class DatasetPanel(parent=None)  — bases: QWidget`

#### `DatasetPanel.row_to_index(row: int) -> int`

#### `DatasetPanel.index_to_row(index: int) -> int`

#### `DatasetPanel.populate(names, activity, subset) -> None`

#### `DatasetPanel.set_note(text: str) -> None`

A line beside the counts saying what these molecules now are.

#### `DatasetPanel.update_subsets(subset) -> None`

#### `DatasetPanel.update_activity(activity) -> None`

#### `DatasetPanel.set_predictions(col: int, values: Dict[int, float]) -> None`

#### `DatasetPanel.clear_predictions() -> None`

#### `DatasetPanel.set_group_labels(labels, names=None) -> None`

Show which chemotype each molecule belongs to, in a column added for the purpose.

The column appears only once a grouping exists, and `None` takes it away again; the
row colours keep meaning the subset, so the group is carried by its own cell.

#### `DatasetPanel.shown_indices() -> List[int]`

#### `DatasetPanel.selected_indices() -> List[int]`

#### `DatasetPanel.current_index() -> int`

#### `DatasetPanel.set_all_shown(on: bool) -> None`

#### `DatasetPanel.show_selected_only() -> None`

#### `DatasetPanel.set_shown(indices, on=True)`

#### `DatasetPanel.select_index(i: int) -> None`

### `class ModelPanel(parent=None)  — bases: QWidget`

#### `ModelPanel.conformer_mode() -> str`

The field-set mode; the learned variant is an ensemble whose weights are fitted.

#### `ModelPanel.learns_weights() -> bool`

#### `ModelPanel.mil_beta() -> float`

#### `ModelPanel.mil_selection() -> str`

#### `ModelPanel.conformer_weighting() -> str`

#### `ModelPanel.set_conformer_info(text: str) -> None`

#### `ModelPanel.selected_physchem() -> List[str]`

#### `ModelPanel.kernel_source() -> str`

#### `ModelPanel.ac_sigma() -> float`

#### `ModelPanel.ac_chirality() -> bool`

#### `ModelPanel.task_key() -> str`

#### `ModelPanel.kernel_layer() -> str`

#### `ModelPanel.mkl_mode()`

False, "align" or "alignf" - the multiple-kernel-learning variant chosen.

#### `ModelPanel.gammas() -> List[float]`

#### `ModelPanel.set_atom_properties(props) -> None`

Offer one field per per-atom property the data set carries (none: hide the row).

#### `ModelPanel.atom_properties() -> List[str]`

#### `ModelPanel.set_descriptor_properties(props) -> None`

Offer every numeric molecule property as a non-field descriptor (none: hide the row).

#### `ModelPanel.probes() -> List[str]`

#### `ModelPanel.probe_terms() -> int`

#### `ModelPanel.descriptors() -> List[str]`

Both kinds: the properties the file carries and the ones the program computes.

#### `ModelPanel.descriptor_h() -> float`

#### `ModelPanel.set_class_properties(props)`

#### `ModelPanel.method_key() -> str`

#### `ModelPanel.oneclass_params() -> dict`

The extra parameters of the one-class descriptions, by method.

#### `ModelPanel.alphas() -> List[float]`

#### `ModelPanel.lams() -> List[float]`

#### `ModelPanel.Cs() -> List[float]`

#### `ModelPanel.epsilons() -> List[float]`

#### `ModelPanel.nus() -> List[float]`

#### `ModelPanel.set_busy(busy: bool) -> None`

#### `ModelPanel.fill_field_table(fieldset: FieldSet, scale: Optional[np.ndarray]=None) -> None`

Populate the field table from a FieldSet, keeping user edits for same-named fields.

#### `ModelPanel.show_fitted_values(fieldset: FieldSet, scale: Optional[np.ndarray]) -> None`

#### `ModelPanel.table_values() -> Dict[str, dict]`

### `class ResultsPanel(parent=None)  — bases: QWidget`

#### `ResultsPanel.complement_statistic() -> str`

#### `ResultsPanel.set_complementarity(header: str, rows, series) -> None`

Fill the complementarity tab: per-molecule terms and series-level numbers.

#### `ResultsPanel.set_residue_profile(header: str, columns, rows) -> None`

Fill the by-residue table: one row per residue, one column per interaction type.

`rows` is (residue, [values], conserved fraction or None); a value that is exactly zero
is left blank, because a residue that does not touch this molecule has no interaction
rather than a weak one.

#### `ResultsPanel.set_surface_poses(header: str, columns, rows) -> None`

Fill the Surface poses tab: one row per molecule, already formatted as text.

#### `ResultsPanel.set_chemotypes(header: str, rows, per_group) -> None`

Fill the Chemotypes tab: the three q² values, then every held-out chemotype.

#### `ResultsPanel.shapley_players() -> str`

#### `ResultsPanel.set_shapley(res, header: str) -> None`

Fill the Shapley tab from a `cmfqsar.shapley.ShapleyResult`.

#### `ResultsPanel.set_points(which: str, x, y, names, indices)`

#### `ResultsPanel.clear_points(which=None)`

#### `ResultsPanel.set_classification_mode(on: bool, activity_name: str='activity', threshold: Optional[float]=None, have_activity: bool=True) -> None`

Switch the plots between the regression and the classification layout.

For a classifier the scatter plot only means something when the classes come from a
threshold on a continuous activity: it then shows how the decision value tracks that
activity, and the two dashed lines cut it into the true and false positives and negatives.
Without such an activity the tab is hidden, because the ranking (Scores) and the ROC curve
are what a classifier is judged by.

#### `ResultsPanel.set_class_points(x, y, classes, is_test, names, indices) -> None`

Scatter of decision value against activity, coloured by true class.

Training molecules are drawn as circles and test molecules as squares.

#### `ResultsPanel.set_roc(curves) -> None`

curves: list of (label, true classes +-1, decision values).

The tab is hidden when none of them can be drawn, so an empty plot is never shown.

#### `ResultsPanel.set_scores(names, scores, classes=None, subsets=None, boundary: float=0.0) -> None`

Ranked decision values: the output of a classification or one-class model.

`classes` is +1/-1/NaN where known, `subsets` the train/test membership.

#### `ResultsPanel.set_enrichment(y_true=None, decision=None, label: str='screening') -> None`

Accumulation curve: actives found against the fraction of the library screened.

#### `ResultsPanel.set_map(image=None, coords=None, values=None, names=None, indices=None, title: str='', cmap: str='viridis') -> None`

Draw a GTM map: `image` is the landscape on the node grid, `coords` the molecules.

#### `ResultsPanel.set_scatter(coords, values=None, names=None, indices=None, title: str='', cmap: str='viridis', xlabel: str='', ylabel: str='') -> None`

Molecules as points with nothing behind them: the kernel-PCA score plot.

The landscape of a GTM fills the square it is drawn on, so every molecule is seen against
a background.  A score plot has no such background and must not imitate one -- what
it shows is the spacing of the points themselves, which is why the axes carry the share
of the kernel variance each component explains and the aspect ratio stays locked.

#### `ResultsPanel.set_search_table(rows)`

#### `ResultsPanel.set_contributions(names, coef, contrib, header: str='') -> None`

Decomposition rule 2: the contribution of every training molecule.

#### `ResultsPanel.set_atom_contributions(symbols, types, contrib) -> None`

Decomposition rule 3: the contribution of every atom of the selected molecule.

#### `ResultsPanel.clear_contributions(message: str='') -> None`

#### `ResultsPanel.append_log(text: str)`

### `class ViewerControls(parent=None)  — bases: QWidget`

#### `ViewerControls.label_kind() -> str`

#### `ViewerControls.style_key() -> str`

#### `ViewerControls.level_fraction() -> float`

#### `ViewerControls.opacity() -> float`

#### `ViewerControls.shapfield_style() -> str`

#### `ViewerControls.protein_style() -> str`

#### `ViewerControls.set_protein_label(text: str) -> None`

#### `ViewerControls.slab_style() -> str`

#### `ViewerControls.slab_radius() -> float`

#### `ViewerControls.arrow_count() -> int`

#### `ViewerControls.set_slab_label(text: str) -> None`

#### `ViewerControls.set_fields(names_labels)`

#### `ViewerControls.current_field() -> Optional[str]`

#### `ViewerControls.set_endpoints(items) -> None`

Offer one entry per endpoint of a two-endpoint model, plus their difference.

#### `ViewerControls.current_endpoint()`

### `capture_settings(panel: QWidget) -> Dict[str, object]`

Every value widget the panel holds as an attribute, keyed by the attribute's name.

Check boxes and radio buttons give a bool, spin boxes a number, combo boxes their current
text and line edits their text.  The keys are the attribute names, so a panel that gains a
widget is captured without a change here, and a project saved before the widget existed
simply leaves it at its default.

### `apply_settings(panel: QWidget, values: Dict[str, object]) -> List[str]`

Put captured values back; returns the keys that no longer match a widget.

### `apply_field_table(panel: 'ModelPanel', values: Dict[str, dict]) -> None`

Set the enabled flag, alpha and h of the fields the table lists, by field name.

## cmfqsar.gui.surface_view

Source: `cmfqsar/gui/surface_view.py`

A surface behind the molecules, in the application (plan item EC10).

The electrochemistry items built three things the window could not show:

* **the slab** (item EC6).  `cmfqsar.slab` turns the top layers of a metal slab, with their
  periodic images, into the same object a protein pocket becomes, and
  `cmfqsar.complementarity` scores a molecule against it.  The viewer already draws a
  pocket behind a ligand; a slab is drawn the same way, with one difference that matters: a slab
  has no edge, so what is drawn is the expansion around the molecules and not one cell.
* **the surface frame** (item EC2).  `cmfqsar.planealign` poses a molecule on the plane
  `z = 0` with the surface normal as z.  The frame is a convention, and a convention that is
  not drawn is one the reader has to take on trust, so the axes are drawn and the two numbers
  the frame is made of -- the height and the tilt -- are stated.
* **the vector co-field** (item EC5).  The co-field of the electric-field kind is a direction at
  a place rather than a value, so it is drawn as arrows; that part lives in
  `cmfqsar.gui.viewer3d.Viewer3D.add_arrows` and in the main window.

This module is the first two: reading a surface, choosing which of its atoms to draw, and saying
in one line what is on screen.  Everything here is a plain function or a small dataclass, so the
tests and `benchmarks/run_surface_view.py` drive it without a window.

### `class Surface  (dataclass)`

A slab in the surface frame: the normal along z, the top atomic layer at z = 0.

Attributes:

- `name: str`
- `atoms: object`
- `source: str` = `''`
- `layers: List[float]` = `None`

#### `Surface.n_atoms` -> int  (property)

#### `Surface.elements` -> List[str]  (property)

#### `Surface.cell2d` -> np.ndarray  (property)

The two surface lattice vectors, as a 2 x 3 array.

#### `Surface.spacing` -> float  (property)

The distance between the first two layers, in angstrom; NaN for a single layer.

#### `Surface.describe() -> str`

### `load_surface(path: str, top_at_zero: bool=True) -> Surface`

Read a slab from any file ASE understands (POSCAR, CIF, extended XYZ) into the frame.

With `top_at_zero` the structure is shifted along z so that its highest atom sits at z = 0,
which is the frame the reference poses of `examples/corrosion_fe110` are given in and the
frame `cmfqsar.planealign` poses a molecule in.

### `build_bcc110(element: str='Fe', nx: int=6, ny: int=6, layers: int=4, a: float=2.78, vacuum: float=15.0) -> Surface`

A body-centred cubic (110) slab, built rather than read; the Fe(110) defaults are FE110.

ASE needs an even number of repeats along the second surface direction for an orthogonal
cell, so `ny` is rounded up to the next even number and the name says what was built.

### `drawn_atoms(surface: Surface, points: np.ndarray, radius: float=8.0, top_layers: int=2, max_atoms: int=20000) -> Tuple[np.ndarray, List[str], np.ndarray]`

The surface atoms worth drawing around `points`, as `(coords, symbols, radii)`.

The top `top_layers` layers are repeated over the surface lattice and then cut to the atoms
whose **in-plane** distance to the nearest of `points` is at most `radius`.  The cut is in
the plane and not in space, so the surface stays a surface under a molecule that sits high
above it instead of thinning out to nothing.

`radius` is the only control a user needs: a small radius shows the patch under the molecule,
a large one shows that the surface has no edge.  Beyond `max_atoms` the drawing would be slow
with no gain, so the radius is reduced until the count fits and the caller is told by the
number of atoms it gets back.

### `site_points(surface: Surface, points: np.ndarray, radius: float=8.0, height: float=0.0) -> Tuple[List[str], np.ndarray]`

The top, bridge and hollow sites of the top layer near `points`.

`cmfqsar.slab.surface_sites` finds them in the primary surface cell; a molecule is
rarely in that cell, so the sites are translated over the surface lattice to the molecules
and then cut to `radius` in the plane, exactly as `drawn_atoms` cuts the atoms.

### `pose_line(coords: np.ndarray, symbols: Sequence[str], name: str='', plane_z: float=0.0) -> str`

One line for the panel: the height and the tilt of a molecule in the surface frame.

### `pose_table(coords: Sequence[np.ndarray], symbols: Sequence[Sequence[str]], names: Sequence[str], plane_z: float=0.0) -> Tuple[List[str], List[List[str]]]`

The pose of every molecule of a set, as the column titles and one row each.

## cmfqsar.gui.viewer3d

Source: `cmfqsar/gui/viewer3d.py`

OpenGL 3D viewer (pyqtgraph.opengl) for molecules, co-field isosurfaces,
molecular-field meshes and atom-contribution spheres.

### `class Viewer3D(parent=None)  — bases: QWidget`

Widget wrapping a GLViewWidget with named groups of items.

#### `Viewer3D.clear_group(name: str) -> None`

#### `Viewer3D.clear_all() -> None`

#### `Viewer3D.has_group(name: str) -> bool`

#### `Viewer3D.add_molecule(mol: MoleculeFields, group: str, style: str='ballstick', carbon_color=None, alpha: float=1.0) -> None`

#### `Viewer3D.add_structure(coords: np.ndarray, bonds: np.ndarray, symbols: Sequence[str], group: str, style: str='lines', carbon_color=(0.55, 0.6, 0.68), alpha: float=1.0, width: float=1.6, radius: float=0.1) -> None`

Draw an arbitrary bonded structure, such as a protein binding site.

Unlike `add_molecule` this takes plain arrays, because a crystal structure is not a
sanitised molecule.  'lines' is a wire frame and is fast enough for thousands of atoms;
'sticks' builds cylinders and should be kept for small sites.

#### `Viewer3D.add_cell(cell: np.ndarray, group: str, origin=(0.0, 0.0, 0.0), color=(0.7, 0.74, 0.82), alpha: float=0.85, width: float=1.6, axes: bool=True) -> None`

The unit cell as the twelve edges of its box.

With `axes`, the three edges that leave the origin are drawn in red, green and blue for
the first, second and third lattice vector, which is the usual convention and says at a
glance which way the cell is turned.  The box is the frame everything periodic is drawn
in: the atoms inside it are one cell, and the images outside it are the same cell
translated.

#### `Viewer3D.add_atoms(coords: np.ndarray, symbols: Sequence[str], group: str, radii: Optional[Sequence[float]]=None, scale: float=ATOM_RADIUS_SCALE, alpha: float=1.0) -> None`

One sphere per atom, coloured by element and sized by its covalent radius.

A crystal has no bond table, so this draws the atoms alone.  `radii` are the covalent
radii a structure carries; without them every atom gets the same size.

#### `Viewer3D.add_points(coords: np.ndarray, symbols: Sequence[str], group: str, size: float=6.0, alpha: float=0.45) -> None`

One dot per atom, coloured by element: the cheap drawing for the periodic images.

A cell with its images out to the kernel's cutoff is thousands of atoms, which is too
many for spheres and exactly what a scatter item is for.  The images are drawn
translucent so that the cell itself stays the thing one looks at.

#### `Viewer3D.add_scalar_surface(verts: np.ndarray, faces: np.ndarray, values: np.ndarray, group: str, alpha: float=0.9, vmax: Optional[float]=None, negative=(0.95, 0.25, 0.2), positive=(0.25, 0.85, 0.35), neutral=(0.85, 0.85, 0.85)) -> None`

A surface coloured per vertex by a signed value, red through grey to green.

Used for the complementarity of a molecule with the pocket, where the sign carries the
meaning: green where the two fields complement each other, red where they conflict.

#### `Viewer3D.add_labels(positions: np.ndarray, texts: Sequence[str], group: str, color=(0.95, 0.95, 0.95), alpha: float=0.95, size: int=9, offset: float=0.0) -> None`

Draw short texts at the given points: atom codes, residue names and the like.

One item per label, so this is meant for the tens of labels of a molecule or a binding
site rather than for a whole structure.

#### `Viewer3D.add_spheres(centers: np.ndarray, radii: Sequence[float], colors, group: str, alpha: float=0.55) -> None`

#### `Viewer3D.add_arrows(points: np.ndarray, vectors: np.ndarray, group: str, scale: float=1.0, color=(0.95, 0.85, 0.2), alpha: float=0.95, shaft_radius: float=0.07, head_radius: float=0.2, head_fraction: float=0.35, min_length: float=0.15) -> None`

One arrow per point: a cylinder with a cone on its end, pointing along the vector.

This is how a field that is a **direction** is drawn.  The co-field of a scalar field is
a value at a place and is drawn as an isosurface; the co-field of the electric-field kind
(plan item EC5) is a vector at a place, and only an arrow can show it.  `scale` is
angstrom per unit of the vector, so the drawing's length scale is a number the caller
states in the legend; an arrow shorter than `min_length` angstrom after scaling is left
out, because a dot in the picture carries no direction.

#### `Viewer3D.add_polyhedra(verts: np.ndarray, faces: np.ndarray, group: str, color=(0.45, 0.65, 0.95), alpha: float=0.35, wireframe: bool=False) -> None`

Coordination polyhedra as one translucent mesh: the inorganic reading of a structure.

A metal atom gets no bonds here (a distance rule would give it twelve), so the shape of
its first coordination shell is what says how the structure is built -- a chain of
corner-sharing octahedra, a layer of edge-sharing tetrahedra.  Every polyhedron of a cell
is drawn as one mesh item, so a cell of hundreds costs one.

#### `Viewer3D.add_frame(origin, group: str, length: float=4.0, labels: Sequence[str]=('x', 'y', 'z'), alpha: float=0.95) -> None`

The three axes of a frame as arrows at `origin`, red, green and blue, with labels.

A surface frame is a convention -- the normal is z and the top atomic layer is z = 0 --
and a convention that is not drawn is a convention a reader has to take on trust.  The
colours are the same as the lattice vectors of a cell carry, so the two views agree.

#### `Viewer3D.add_isosurface(verts: np.ndarray, faces: np.ndarray, color, group: str, alpha: float=0.45, wireframe: bool=False, normals: Optional[np.ndarray]=None) -> None`

#### `Viewer3D.set_axes(on: bool, size: float=5.0) -> None`

#### `Viewer3D.set_dark(dark: bool) -> None`

#### `Viewer3D.center_on(coords: np.ndarray, distance: Optional[float]=None) -> None`

#### `Viewer3D.reset_view() -> None`

#### `Viewer3D.screenshot(path: str) -> None`

### `cofield_colors(field_name: str)`

### `color_name(c) -> str`

The name of one of the fixed co-field colours, for a legend that has to be read.

A legend saying "positive = rgb(0.9,0.2,0.2)" tells the reader nothing; the colours of the
co-fields are fixed, so each one has a name.

### `element_color(symbol: str, default=(0.6, 0.6, 0.6)) -> Tuple[float, float, float]`

The colour of an element: the table above where it has one, else the Jmol colour.

The organic elements keep the colours the molecular views have always used, so that a
molecule looks the same beside a crystal.  Everything else -- the metals, the alkali ions,
the rare earths an inorganic structure is made of -- takes the standard Jmol colour from
ASE's table, which covers every element.  An element neither table knows is grey.

### `signed_spheres(values: Sequence[float], vmax: Optional[float]=None, min_radius: float=0.12, max_radius: float=0.85, positive=(0.95, 0.2, 0.2), negative=(0.2, 0.4, 0.95))`

Radii and colours for one sphere per signed value: size by size, colour by sign.

Returns `(radii, colours, vmax)`.  The radius grows as the square root of the magnitude,
so that the *area* a sphere shows on screen is about proportional to the value; `vmax` is the
value the largest sphere stands for and is returned so that a legend can state it.  This is
the convention the atom contributions of a molecule are drawn with, used here for the
per-site contributions of a cell.

### `molecule_mesh(mol: MoleculeFields, style: str='ballstick', carbon_color=None, alpha: float=1.0)`

Build (vertices, faces, colours) of a ball-and-stick / stick / spacefill molecule.

## cmfqsar.gui.workers

Source: `cmfqsar/gui/workers.py`

Background execution of long computations with progress reporting and cancellation.

### `class TaskWorker(fn: Callable, parent: Optional[QObject]=None)  — bases: QThread`

Run `fn(progress, cancel)` in a thread.

`progress(message: str, fraction: float)` may be called from the task to
report progress; `cancel()` returns True when cancellation was requested.

#### `TaskWorker.cancel() -> None`

#### `TaskWorker.is_cancelled() -> bool`

#### `TaskWorker.run() -> None`
