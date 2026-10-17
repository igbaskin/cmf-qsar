# CMF 3D-QSAR — desktop application for the Continuous Molecular Fields approach

A Python/Qt desktop application (plus a library and a command-line tool) that implements the
**Continuous Molecular Fields (CMF)** approach to 3D-QSAR. The approach is described in the following
publications; the three the program follows most closely are the 2013, 2015 and 2019 papers, which the
text below refers to by year.

**The method**

* Zhokhova N.I., Baskin I.I., Bakhronov D.K., Palyulin V.A., Zefirov N.S. *Method of continuous molecular
  fields in the search for quantitative structure–activity relationships.* Dokl. Chem. **429** (2009) 273–276.
  [doi:10.1134/S0012500809110056](https://doi.org/10.1134/S0012500809110056)
* Baskin I.I., Zhokhova N.I. *The continuous molecular fields approach to building 3D-QSAR models.*
  J. Comput.-Aided Mol. Des. **27** (2013) 427–442.
  [doi:10.1007/s10822-013-9656-4](https://doi.org/10.1007/s10822-013-9656-4)
* Baskin I.I., Zhokhova N.I. *Continuous molecular fields approach applied to structure–activity modeling.*
  In: Gorb L., Kuz'min V., Muratov E. (eds) *Application of Computational Techniques in Pharmacy and
  Medicine.* Challenges and Advances in Computational Chemistry and Physics, vol. 17. Springer, Dordrecht
  (2014) 433–459. [doi:10.1007/978-94-017-9257-8_13](https://doi.org/10.1007/978-94-017-9257-8_13)
  ([arXiv:1311.1495](https://arxiv.org/abs/1311.1495))
* Sitnikov G.V., Zhokhova N.I., Ustynyuk Y.A., Varnek A., Baskin I.I. *Continuous indicator fields: a novel
  universal type of molecular fields.* J. Comput.-Aided Mol. Des. **29** (2015) 233–247.
  [doi:10.1007/s10822-014-9818-z](https://doi.org/10.1007/s10822-014-9818-z)
* Baskin I.I., Zhokhova N.I. *Descriptors based on continuous indicator fields for 3D-QSAR studies.*
  Proceedings of MOL2NET 2016, International Conference on Multidisciplinary Sciences, 2nd edition (2017).
  [doi:10.3390/mol2net-02-03854](https://doi.org/10.3390/mol2net-02-03854)
* Baskin I.I., Zhokhova N.I. *Continuous molecular fields and the concept of molecular co-fields in
  structure–activity studies.* Future Med. Chem. **11** (2019) 2701–2713.
  [doi:10.4155/fmc-2018-0360](https://doi.org/10.4155/fmc-2018-0360)

**One-class classification and virtual screening with CMF**

* Karpov P.V., Baskin I.I., Zhokhova N.I., Zefirov N.S. *Method of continuous molecular fields in the
  one-class classification task.* Dokl. Chem. **440** (2011) 263–265.
  [doi:10.1134/S0012500811100016](https://doi.org/10.1134/S0012500811100016)
* Karpov P.V., Baskin I.I., Zhokhova N.I., Nawrozkij M.B., Zefirov A.N., Yablokov A.S., Novakov I.A.,
  Zefirov N.S. *One-class approach: models for virtual screening of non-nucleoside HIV-1 reverse
  transcriptase inhibitors based on the concept of continuous molecular fields.* Russ. Chem. Bull. **60**
  (2011) 2418–2424. [doi:10.1007/s11172-011-0372-8](https://doi.org/10.1007/s11172-011-0372-8)

![screenshot](docs/screenshot.png)

*The HIF-2α series of the Merck free-energy benchmark, 38 ligands: a kernel ridge model with q² = 0.62,
its electrostatic co-field drawn as ± isosurfaces, and the binding site of the reference crystal
structure, which the data set brings with it and which needs no superposition.*

## What it does

| Stage | Implementation |
|---|---|
| Molecular fields | One conformation per molecule by default, or a **conformational ensemble** whose mean field is the fourth dimension of 4D-QSAR (see below). Every field is a sum of atom-centred Gaussians `w exp(-α/2 |r-r_l|²)` (Eq. 4 of the 2013 paper). Five physico-chemical fields (electrostatic, steric, hydrophobic, H-bond acceptor, H-bond donor) and **continuous indicator fields** (one per Sybyl atom type, 2015 paper). The steric field uses radius-scaled widths `α/R_vdW²` (Eq. 10–11). |
| Kernels | Overlap integrals of the fields computed **analytically** for all atom pairs (Eq. 5 / Eq. 11), summed per field type (Eq. 2) and mixed with coefficients `h_f` (Eq. 1). Optional cosine normalisation. Vectorised NumPy implementation; a 300-molecule set takes a few seconds per α value. An optional **nonlinear kernel layer** (Gaussian or arc-cosine) and **multiple kernel learning** for the mixing coefficients are described below. |
| Learning | **Regression**: kernel ridge regression with bias term (Eq. 12), ε-SVR / ν-SVR (precomputed kernel) and a Gaussian-process view of KRR giving predictive variances. **Binary classification**: C-SVC / ν-SVC, or a **Gaussian-process classifier** fitted by the Laplace approximation, which adds calibrated probabilities, a posterior variance per molecule and model selection by marginal likelihood (see below). **One-class classification** (novelty detection) for similarity-based virtual screening: the 1-SVM, or the centroid, k-nearest-neighbour, Gaussian-process and kernel-PCA descriptions of the active class (see below). |
| Validation | k-fold cross-validation, external (nested) CV with repeats, and, for a series of several scaffolds, **leave-one-chemotype-out** folds that never put a molecule's own scaffold-mates in the training part, with an alignment-free fragment model as the baseline that the aligned model has to beat (see below). |
| Model selection | Grid search over α (shared) and λ (or C/ε/ν) minimising RMSE in k-fold cross-validation (shuffled or interleaved folds), optional continuous Brent minimisation over λ as in the 2013 scripts; optional coordinate-descent optimisation of the mixing coefficients `h_f` or their choice by **centred kernel alignment** (multiple kernel learning), and a grid over the bandwidth of the Gaussian kernel layer; **external (nested) cross-validation** with repeats (q²ex, RMSEcvex), kernels computed once per α and sliced for the outer folds. |
| Statistics | Regression: r², RMSE (fit); q², RMSEcv (internal CV); R²pred, RMSEpred (test set, both variance conventions); q²ex, RMSEcvex (external CV). Classification: accuracy, balanced accuracy, sensitivity, specificity, Matthews correlation, AUC and ROC curves, for the fit, the cross-validation and the test set. |
| Interpretation | **Molecular co-fields** `C_f(r) = Σ_j a_j X_fj(r)` (Eq. 9) rendered as ± isosurfaces; molecular field `X_f(r)` of any molecule in the same colours, with or without a model (co-fields are drawn as solid surfaces and molecular fields as a mesh by default; both styles are switchable); overlap `C·X` whose integral is the field's contribution to the prediction (Eq. 7); the three decomposition rules of the 2019 paper: training-molecule contributions `a_j K(X,X_j)` and **atom contributions** (Eq. 14 of the 2015 paper) shown as red/blue spheres; **Shapley values** over atoms, substituents or field types, which replace the co-field when a nonlinear kernel layer is used and reproduce the atom contributions exactly when it is not, with second-order interaction indices. |
| Tools | **Protein binding sites** fetched from the PDB or read from a local file and drawn behind the molecules and their fields, with atom codes on the molecule and residue names on the alpha carbons; **field complementarity** of the molecules and of the co-fields with the pocket, as numbers and as a coloured surface (see below); MCS-based alignment to a reference molecule, optional pose refinement by maximising the CMF kernel similarity (the alignment criterion proposed in the 2013 paper), **alignment by prediction confidence** (see below), 3D generation from SMILES, model save/load, CSV/SDF/Gaussian-cube export, screenshots. |

### Conformational ensembles: the fourth dimension

A 3D-QSAR model represents a molecule by one conformation, and the answer depends on which one was
picked. 4D-QSAR (Hopfinger et al., J. Am. Chem. Soc. 119 (1997) 10509) replaces the choice with an
ensemble and averages over it. *Model -> Conformations* switches between the two (CLI:
`--conformations single|ensemble`); **single is the default and is untouched**, which the tests
assert by comparing its fields with those built before conformations existed.

In the CMF framework the averaging needs no new mathematics. A field is a sum of atom-centred
Gaussians and the kernel is bilinear in the fields, so the mean field of an ensemble

```
X_f(r) = sum_k w_k X_f^(k)(r)
```

is itself a field, whose atoms are the atoms of every conformation with their weights multiplied by
that conformation's weight. The kernel of two ensembles then comes out as

```
K(A, B) = sum_k sum_l w_k^A w_l^B K(X_A^(k), X_B^(l))
```

which is the mean-map kernel of the two conformational distributions and is positive definite by
construction. A test checks that identity against the pairwise kernels of the conformations. The
model stays linear in that mean field, so **co-fields, the decomposition rules, the Shapley machinery
and the complementarity scores all keep working**; contributions arrive per atom per conformation and
fold back onto the atoms by summation.

*Tools -> Generate conformations* builds the ensembles with ETKDG and MMFF94s, keeping the pose a
molecule already has as its first conformation, so an alignment made earlier survives. A
multi-conformer file can be read instead, with `--ensemble-by name` grouping the records of each
compound. Weights are uniform or Boltzmann from the stored energies, with kT and a cap on the number
of conformations kept; the viewer draws the ensemble as superimposed conformations.

Where the *other* conformations go is a choice of its own, and not a small one. The kernel compares
molecules with each other, so what matters is not where a conformation sits relative to its own
molecule but where it sits relative to the rest of the series; a conformation superimposed on its own
molecule alone is still arbitrarily oriented with respect to every other molecule in the set.
*Place the conformations* in the dialog (CLI: `--conformer-align`) offers four answers, all of them
leaving the molecule's own first conformation exactly where it was:

| `--conformer-align` | where each generated conformation is put |
|---|---|
| `self` | on the molecule's own pose, over all heavy atoms - all that `rdMolAlign.AlignMolConformers` can do, and the default, so nothing changes silently |
| `core` | on the molecule's own pose, over the atoms of the maximum common substructure with a reference molecule only, so the scaffold stays where the published alignment put it and only the flexible part moves |
| `reference` | on a reference molecule of the series, through that same common substructure, which puts every conformation of the series in one frame |
| `field` | as `reference`, then refined by maximising the normalised CMF kernel with the reference - the alignment criterion of the 2013 paper, applied to each conformation |

The reference is the largest molecule of the set unless `--conformer-align-ref` names another one;
its common substructure with the others is the most likely to be the scaffold of the series. The
field set needs no new setting for this: where a conformation is placed is a property of the
geometry, not of the field.

The cost is quadratic in the number of conformations. With five conformations per molecule, the
default settings of `cmf_cli.py build` (RDKit parameterisation, field kernels scaled to unit mean
diagonal, the α grid 0.2–3 and the λ grid 10⁻⁴–10, shuffled 10-fold CV) and the published
train/test split carried by the SD files:

| set | representation | q²cv | R²pred |
|---|---|---|---|
| steroids, rigid | single conformation | 0.810 | **-0.304** |
| steroids, rigid | ensemble, on its own pose (`self`) | 0.831 | -0.961 |
| steroids, rigid | ensemble, own pose, common scaffold (`core`) | 0.839 | -0.937 |
| steroids, rigid | ensemble, on the reference (`reference`) | 0.779 | -0.441 |
| steroids, rigid | ensemble, reference + field (`field`) | **0.883** | -2.293 |
| ACE, flexible | single conformation | **0.741** | **0.604** |
| ACE, flexible | ensemble, on its own pose (`self`) | 0.697 | 0.564 |
| ACE, flexible | ensemble, own pose, common scaffold (`core`) | 0.681 | 0.525 |
| ACE, flexible | ensemble, on the reference (`reference`) | 0.671 | 0.411 |
| ACE, flexible | ensemble, reference + field (`field`) | 0.697 | 0.385 |
| THR, flexible | single conformation | **0.696** | **0.373** |
| THR, flexible | ensemble, on its own pose (`self`) | 0.501 | 0.100 |
| THR, flexible | ensemble, own pose, common scaffold (`core`) | 0.514 | 0.106 |
| THR, flexible | ensemble, on the reference (`reference`) | 0.492 | 0.126 |
| THR, flexible | ensemble, reference + field (`field`) | 0.489 | 0.195 |

Read the table and not the hope: **none of the three new criteria recovers the loss on thrombin**.
Anchoring every conformation on the molecule's own published scaffold lifts q²cv from 0.501 to
0.514 and forcing the whole series into one frame gives 0.492, against 0.696 for the
single conformation; R²pred stays near 0.1 where the single-conformation model reaches 0.373.
On ACE every ensemble is worse than the single conformation as well, and on the steroids R²pred is
negative for every representation, the single-conformation model included, so that column says
nothing there; the higher q²cv of the ensembles on that set is backed by no test-set gain at all.

That is an answer, just not the expected one. Adding conformations to thrombin hurts however they
are placed, which points at the conformations rather than at their orientation: the series carries
poses its authors aligned as bound-like, and ETKDG conformations of an isolated molecule are not
those poses, so averaging them into the field dilutes exactly the arrangement the model lives on.
The ensemble size says the same thing - on thrombin, with the placement varied and everything else
the default:

| conformations per molecule | `self` q²cv | `core` q²cv |
|---|---|---|
| 1 (single) | 0.696 | 0.696 |
| 2 | 0.672 | 0.728 |
| 3 | 0.570 | 0.732 |
| 5 | 0.501 | 0.514 |

The decay is monotone in the number of generated conformations, and `core` only delays it: holding
the scaffold on the published pose keeps two or three conformations harmless (q²cv 0.728 and 0.732,
above the single-conformation 0.696, though R²pred does not follow at 0.150 and -0.221) and then
collapses at five. So the alignment of the ensemble was worth fixing and is now there to use, but it
is not what was wrong on thrombin.

#### Keeping only the conformations near the aligned pose

If gas-phase geometries are the problem, the remedy is to use only those close to the pose the
series was aligned to. `--conformer-window` (the *Keep within* field of the Tools dialog) keeps the
conformations standing within a given heavy-atom RMSD of the pose the molecule arrived in, measured
where they are rather than after a best fit, so the ensemble covers the local flexibility around
that pose instead of the whole conformational surface. The arrival pose is always kept, and a wider
pool is embedded because most of it is discarded.

It does not work either, and the reason is instructive. On thrombin, five conformations requested:

| window | conformations kept | q²cv | R²pred |
|---|---|---|---|
| single conformation | 1.0 | **0.696** | 0.373 |
| 1.0 Å | 1.0 | 0.696 | 0.373 |
| 1.5 Å | 1.0 | 0.696 | 0.382 |
| 2.0 Å | 1.1 | 0.686 | **0.385** |
| 2.5 Å | 1.9 | 0.590 | 0.313 |
| 3.0 Å | 3.7 | 0.600 | 0.275 |
| no window | 5.0 | 0.501 | 0.100 |

ACE behaves the same way: 0.741 for the single conformation, 0.721 at a 1.5 Å window keeping 1.8
conformations, 0.693 at 2 Å keeping 2.9, and 0.692 for the full ensemble. The quality is monotone in
how far the ensemble strays from the arrival pose, and the window recovers the baseline only by
emptying the ensemble: generated conformations of these ligands stand about 3 Å from the aligned
pose even after being superimposed on it, so any window tight enough to be *local* keeps nothing but
the pose itself. There is no middle ground in which the ensemble both differs from one conformation
and predicts better.

Averaging over generated conformations therefore does not help an aligned 3D-QSAR series, and three
placements and six windows did not find a case where it does. What remained was to stop averaging.

#### Letting the model choose the conformation

`cmfqsar/mil.py` (CLI: `cmf_cli.py mil`; in the application, the third entry of
*Model -> Conformations*) treats a molecule as a *bag* of conformations whose label
belongs to the bag, and gives each conformation a weight driven by the model's own score for it,

```
w_ik = exp(beta f(x_ik)) / sum_l exp(beta f(x_il))
```

with two limits that bracket everything: `beta = 0` is the ensemble mode exactly, and a large beta
puts all the weight on the single highest-scoring conformation, which is the witness assumption of
mi-SVM and of multi-instance regression. In between the weighting stays soft, which keeps the bag
kernel a weighted mean of conformation kernels, `W K W^T`, hence positive definite, and keeps the
model linear in the mean field, so co-fields and the decomposition rules survive. Fitting
alternates, as multi-instance methods do: fit, score every conformation, re-weight, repeat; the
conformation kernel is computed once and the rest is matrix algebra. The weights are fitted to the
activities, so they are relearned inside every cross-validation fold, and the folds split molecules,
never conformations.

On synthetic bags that obey the assumption, where the label is the best score among five instances,
the method does what it should: it recovers the hidden instance in 98 % of bags and lifts q²cv from
0.398 to 0.999, where averaging is helpless. That is the test that proves the machinery, and it is
in the suite.

On real series it does not pay, and how it fails is the interesting part. Five conformations per
molecule, the kernels cosine-normalised, α and λ chosen by cross-validation:

| set | representation | q²cv | R²pred | conformations that count |
|---|---|---|---|---|
| THR | single conformation | 0.652 | **0.545** | 1.00 |
| THR | ensemble, `beta = 0` | 0.290 | 0.309 | 5.00 |
| THR | selection, `beta = 3` | **0.656** | 0.547 | 1.09 |
| ACE | single conformation | 0.656 | **0.358** | 1.00 |
| ACE | ensemble, `beta = 0` | **0.717** | 0.284 | 5.00 |
| ACE | selection, chosen by CV | 0.717 | 0.284 | 5.00 |

On thrombin the selection repairs exactly the damage that averaging did, and it does so by choosing
the pose the molecules arrived in for 97 % of them, 100 % under hard selection. In other words the
model, given five conformations and free choice, re-selects the alignment the authors published and
matches the single-conformation result to three decimals. On ACE cross-validation prefers no
selection at all, and the gain it reports does not survive on the test set.

The alternation is a local search, and that shows: started from equal weights instead of from the
arrival pose, the same fit lands at q²cv -0.29 and keeps the published pose for 2 % to 24 % of
molecules. Annealing beta upwards from a soft start, which `anneal_weights` does, lifts that to
0.23 but still falls far short, and it *degrades* a run that started at the published pose. So the
method cannot find a bioactive conformation among gas-phase ones; it can only confirm that the pose
already there is the best of what is offered, which is a useful diagnostic and not a better model.

#### Placing the conformations in the pocket

The remaining idea was to stop generating ensembles in the gas phase and let the receptor place
them. `cmfqsar/docking.py` (CLI: `cmf_cli.py mil --dock`; in the application, *Tools -> Dock into the
pocket*) is a small rigid-body docking written
against what the program already has: the pocket fields, the exact overlap integrals, a soft clash
term, a burial term and a tether. A pose is scored by

```
complementarity with the pocket + burial - clash - tether
```

and the six rigid-body parameters of each conformation are optimised from where it stands, with
bounded restarts. The internal geometry is never touched, so the conformations stay as generated and
only their placement changes. It is not a docking program: there is no desolvation, no entropy and
no torsional search, and the scoring function is not fitted to binding data.

It does what it is asked to. On the thrombin set of the Open Force Field collection the clash
dropped from 7.8 to zero and the score from -15.1 to +0.54, the poses moving 3.0 Å; on the HIF-2α
set of the Merck collection, where the ligands were already well placed, the poses moved 1.0 Å and
the small residual clash vanished. Both took about ten minutes for a hundred to two hundred poses.

It does not help the model, which is the fourth answer of the same kind:

| set | representation | q²cv |
|---|---|---|
| HIF-2α, 38 ligands | single conformation | **0.623** |
| HIF-2α, 38 ligands | gas ensemble | 0.610 |
| HIF-2α, 38 ligands | docked ensemble | 0.582 |
| HIF-2α, 38 ligands | selection on the gas ensemble | 0.528 |
| HIF-2α, 38 ligands | selection on the docked ensemble | 0.537 |

The thrombin set of that collection says nothing either way: with 22 ligands no representation
models it at all, the single conformation included, at q²cv -0.06 and worse.

Two things temper this. These benchmark ligands arrive in bound-like poses already, so the
experiment really asks whether *re-docking a good pose* helps, and it does not; a series with no
trustworthy alignment is a different question. And a proper docking program with a flexible-ligand
search would place poses better than this one does.

That second point suggests where this code is actually worth using. Placing a molecule in a site by
optimising a pocket score is an *alignment* method, and for a series whose alignment is a guess from
a common substructure it may well beat that guess. The ensemble question is closed; docking as an
alignment criterion is open and the machinery for it is now here.

Both are in the application as well as on the command line. *Model -> Conformations* has a third
entry, "ensemble, weights learned", with the hardness beside it and a box for choosing exactly one
conformation; the fit learns the weights, writes them onto the molecules and then proceeds as an
ordinary ensemble model, so co-fields, the decomposition rules and the Shapley machinery stay
available and report the weighting the model was actually fitted with. *Tools -> Dock into the
pocket* places every conformation of every molecule in a loaded structure, reports what the score
and the clash did, and discards the current model, since the molecules it was fitted on have moved.

**The verdict on the fourth dimension.** Averaging, placement, windowing, selection and
receptor-placed ensembles all fail to beat one well-chosen conformation on an aligned series. Five
routes, one answer. The mode stays because it is exact and costs nothing when unused, and because
the pieces it needed, conformer generation, multi-instance weighting and pocket docking, are useful
on their own. Treat it as something to test against the single-conformation model on each series,
with the number of conformations, the window and beta as parameters of their own, rather than as an
improvement to switch on; which is why the default remains a single conformation.

The earlier version of this table reported different numbers for the single-conformation rows, with
a protocol that was not recorded and could not be reproduced; every row above was measured again
with the settings stated, so the rows are comparable with each other.

Boltzmann weighting was worse than uniform on all three sets, because the conformations are generated
from one input pose and their energies then concentrate the weight on a few of them; uniform weighting
is the safer default and is what the application starts with.

One last trap: a seed of exactly zero makes RDKit return the same geometry for every conformation,
which would silently collapse an ensemble; the generator maps it to a fixed non-zero seed and a test
guards that.

### Heterogeneous series: several chemotypes, no one common substructure

Everything above rests on the molecules sharing a frame, and a congeneric series supplies one:
superimpose on the common substructure and the fields become comparable. A series of several
scaffolds has no such substructure, and the usual recipe quietly stops meaning anything — one
reference molecule, one maximum common substructure, and a random cross-validation that puts a
molecule's own scaffold-mates in the training part, which is the most optimistic split available.

`cmfqsar/chemotypes.py` is the workflow for that case, in the order the problem imposes.

1. **Cluster by scaffold.** `chemotype_labels` groups the molecules by their Bemis-Murcko
   framework, either as drawn or made generic so that elements and bond orders stop mattering.
   Groups below a minimum size are pooled into one "other" group, so that a handful of singletons
   does not turn the validation into leave-one-molecule-out.
2. **Align within each cluster**, where a common substructure does exist and does the job well:
   `align_within_groups` superimposes every molecule on the largest member of its own group.
3. **Place the clusters relative to one another**, the step a congeneric series never needs.
   `place_groups` moves each cluster as a rigid body, so step 2 survives intact: either by
   maximising the CMF kernel between its reference and the reference of a chosen cluster (their
   centroids are brought together first, since the kernel is flat where two molecules do not
   overlap and there is nothing to climb), or by docking that reference into the binding site,
   which needs a structure but no reference cluster at all.
4. **Judge it by leave-one-chemotype-out.** `grouped_folds` never splits a group, and
   `grouped_cv_predict` holds out a whole chemotype at a time; `max_folds` bundles the groups
   when there are many small ones.
5. **Compare with a baseline that needs no alignment.** `count_kernel` is the continuous
   indicator fields at vanishing attenuation. Writing the overlap integral of two such fields
   out,

   ```
   K_t(A,B) = (π/α)^{3/2} Σ_{l∈A} Σ_{m∈B} exp(−α |r_l − r_m|² / 4)
            = (π/α)^{3/2} [ n_t(A) n_t(B) − (α/4) Σ_lm d²_lm + O(α²) ],
   ```

   so as α tends to zero every exponential tends to one, the kernel tends to the dot product of
   the atom-type count vectors, and nothing of the geometry survives. By default the baseline is
   that limit, written down from the counts themselves: each field scaled to unit mean diagonal,
   which in the limit is a division by the mean of n_t², and the cosine of what is left. It is
   exactly pose-free rather than nearly so — it never reads a coordinate — and it costs one pass
   over the atoms instead of a kernel over every pair of them, which is 10 to 34 times faster on
   the sets in `examples/`. `alignment_free_q2` then runs that kernel through the same grouped
   folds, with its own λ.

   An explicit attenuation takes the Gaussian route instead, which is the thing the limit is a
   limit of. Two details decide whether that route arrives, and both are measured in the tests:

   - **The attenuation has to be far smaller than a modelling one.** The first correction is
     linear in α and carries the separation of the two molecules' centroids, so the pose
     dependence dies only as fast as α does. A rigid motion of every molecule still moves the
     kernel by 0.70 at α = 0.02 and by 0.12 at 2·10⁻³; at `BASELINE_ALPHA` = 2·10⁻⁴ it is 0.013,
     and the two routes agree to 4·10⁻⁵ by α = 2·10⁻⁶.
   - **The cosine normalisation has to be applied to the joint kernel, not field by field.** A
     field normalised on its own tends to `n_t(A) n_t(B) / (n_t(A) n_t(B))` = 1, so the counts
     cancel against themselves and what is left is the indicator of "both molecules contain type
     *t*" — the baseline would lose the very descriptors it is built from, and the kernel would
     collapse towards a matrix of ones.

   If the aligned model cannot beat this baseline, the alignment is not paying for itself, and on
   a heterogeneous set that is the first thing worth knowing.

`evaluate` runs steps 4 and 5 together and returns the three numbers with a verdict.

**What it shows.** Five sets with more than one scaffold, molecules in their published poses (no
realignment), all five physico-chemical fields, KRR with α over 0.3/0.5/1.0 and λ over 0.01/0.1/1,
groups of at least three molecules; the baseline gets the same λ grid on the same folds:

| set | n | chemotypes | random folds | leave-one-chemotype-out | alignment-free |
|---|---:|---:|---:|---:|---:|
| sutherland/COX2 | 282 | 14 | +0.537 | **+0.425** | +0.174 |
| sutherland/DHFR | 361 | 31 | +0.623 | **+0.463** | +0.159 |
| sutherland/BZR  | 147 | 12 | +0.388 | +0.088 | **+0.144** |
| sutherland/AChE | 111 |  7 | +0.510 | **+0.275** | −0.060 |
| mittal/ACHE     | 111 |  7 | +0.168 | −0.304 | **−0.058** |

Random folds flatter every one of them, by 0.11 to 0.47 in q²; on ACHE they turn a model that
cannot predict a new scaffold at all into a modestly positive number. That gap is the reason the
grouped folds are here. Across scaffolds the aligned model still earns its keep on COX2, DHFR and
Sutherland's AChE, and does not on BZR or on Mittal's ACHE, where a fragment-count model that
ignores the geometry entirely does as well or better. Which it is cannot be guessed from the
random-fold number, and the per-chemotype table says where the failure is: on DHFR one group of
five molecules is predicted 2.07 log units too low whatever is done with it.

How strict the baseline is matters, and shows how little attenuation it takes to smuggle the
geometry back in. The same column taken by the Gaussian route at α = 2·10⁻⁴ instead of at the
limit reads +0.282 on COX2 and +0.250 on DHFR, against +0.174 and +0.159 here — the residual pose
dependence at that attenuation is worth around 0.1 in q² on the two large sets. It does not
change any verdict, but a baseline is only as honest as its limit.

Realigning first did not rescue the sets that fail. Under the same protocol, Mittal's ACHE goes
from −0.304 in its published poses to −0.392 after step 2 and −0.390 after step 3, and BZR from
+0.088 to +0.097 and then +0.050: the same answer the conformational ensembles gave, from the
other direction. A placement chosen to maximise field overlap is not a bioactive-pose hypothesis,
and the model can tell. What the workflow is for is the question in step 4, not the realignment in
steps 2 and 3, which are there for a set that arrives with no usable poses at all.

The hyper-parameters are chosen on the same grouped folds the model is reported on, which is
optimistic in the usual way of any selection on the reported statistic; what matters here is the
comparison between representations, which share the protocol. The Murcko framework is one grouping
among several reasonable ones — a different clustering gives different folds, and the workflow
takes any labelling, not only the one it computes.

```bash
python cmf_cli.py hetero examples/sutherland/COX2.sdf --charges file --min-group 3 --out cox2_groups.csv
```

`--align-within` and `--place field|dock` add steps 2 and 3, `--generic-scaffolds` merges
chemotypes that differ only in their heteroatoms, `--max-folds` bundles the groups, and
`--baseline-alpha` takes the baseline by the Gaussian route at that attenuation instead of at its
limit. In the application the same choice is the "Baseline α" box of step 4, which reads "the
limit (atom counts)" at zero.

In the application the same five steps are a submenu, *Tools -> Heterogeneous sets (several
chemotypes)*, in the order they have to be taken. Step 1 adds a "Chemotype" column to the data-set
table, coloured by group, with the scaffold as its tooltip; steps 2 and 3 move the molecules and
discard a model fitted on where they were; step 4 fills a "Chemotypes" tab with the three q²
values, the verdict and every chemotype's RMSE and bias while it was the one held out.

### Property fields against indicator fields

Three of the five physico-chemical fields are not read off the molecule but fitted: the
hydrophobic field and the two hydrogen-bond fields are one-atom fragment increments obtained by
linear regression of experimental logP and of Abraham's hydrogen-bond acidity and basicity on
counts of atom types, over large heterogeneous sets, keyed by Sybyl type and the number of
attached hydrogens ([increments.py](cmfqsar/increments.py), the `tripos` parameterisation; the
`rdkit` one uses Wildman–Crippen's logP and molar-refractivity contributions, fitted the same
way). The regression coefficients become the atomic weights, and the field is the usual sum of
Gaussians on them. The continuous indicator fields are the same atom types left unweighted, one
field per type.

So a property field is a weighted sum of indicator fields, `Φ_prop(r) = Σ_t c_t Φ_t(r)`, which
suggests it can add nothing. That is true of the fields and false of the kernels, and the model
sees only the kernels:

```
K_prop(A,B) = ⟨Σ_t c_t X_t(A), Σ_u c_u X_u(B)⟩ = Σ_{t,u} c_t c_u ⟨X_t(A), X_u(B)⟩
```

The indicator representation carries only the diagonal terms `t = u`, each as a separate kernel
with its own mixing coefficient. The property field carries the whole cross-type overlap matrix —
the part that says an atom of one type sits where an atom of another type sits in the other
molecule — with coefficients fixed by the external regression. Neither hypothesis space contains
the other, so the comparison is a question for measurement.

Leave-one-chemotype-out q² on the five sets of the section above, identical α and λ grids
throughout, molecules in their published poses:

| representation | fields | COX2 | DHFR | BZR | AChE | ACHE |
|---|---:|---:|---:|---:|---:|---:|
| property fields (Wildman–Crippen) | 5 | **+0.425** | +0.463 | +0.088 | **+0.275** | −0.304 |
| property fields (2013 increments) | 5 | +0.406 | +0.473 | **+0.138** | +0.074 | −0.280 |
| indicator fields | 9–16 | +0.326 | +0.454 | +0.098 | +0.196 | **−0.242** |
| property + indicator | 14–21 | +0.351 | **+0.518** | +0.137 | +0.267 | −0.302 |
| electrostatic and steric only | 2 | +0.305 | +0.401 | +0.103 | +0.180 | −0.244 |
| alignment-free counts (the floor) | — | +0.174 | +0.159 | +0.144 | −0.060 | −0.058 |

**Fitted weights earn their place.** Five property fields beat the 9 to 16 indicator fields on
three sets and draw on a fourth: +0.10 on COX2, +0.08 on AChE, +0.01 on DHFR, −0.01 on BZR,
−0.06 on Mittal's ACHE. The three increment-derived fields are worth +0.05 to +0.12 over
electrostatic and steric alone on every set. A coefficient fitted on thousands of outside
molecules is information the hundred molecules of a QSAR set do not have to pay for.

**The gain is the weighting, not the capacity.** Putting both representations in one model helps
on DHFR (+0.055 over the best single one) and is neutral or slightly worse on the other four,
which is what the kernel argument predicts: the cross-type overlaps the property field carries
are most of what the extra fields would have supplied, and twenty fields on a hundred molecules
is more freedom than the folds can pay for.

**Which increments, matters.** The two parameterisations differ by 0.20 on AChE (+0.275 against
+0.074) and by 0.05 on BZR, in opposite directions. Atomic contributions from a counts regression
are identified only up to the null space of the count matrix — types that always occur together
can trade coefficients with no change in the property fit — and that freedom, invisible in the
property model, moves weight from one place in space to another once the contributions become a
field. Two fits of equal quality give different 3D models. Regularise the fit, pool the rare
types, and treat the result as one defensible decomposition rather than the true one.

**No representation rescues a bad frame.** On BZR and Mittal's ACHE, where the aligned model
already lost to the alignment-free floor, every row is still below it. Representation is not the
binding constraint there; the alignment is.

One aside. The random-fold numbers would have ranked these differently — on AChE electrostatic
and steric alone look best of all under random folds (+0.567) and second worst under grouped ones
(+0.180). Random folds mislead in the choice of a representation, not only in the reporting of
one.

```bash
python cmf_cli.py hetero examples/sutherland/COX2.sdf --charges file                  # property fields
python cmf_cli.py hetero examples/sutherland/COX2.sdf --charges file --param tripos   # 2013 increments
python cmf_cli.py hetero examples/sutherland/COX2.sdf --charges file --fields "" --cif
python cmf_cli.py hetero examples/sutherland/COX2.sdf --charges file --cif            # both
```

### Nonlinear kernel layers and multiple kernel learning

The CMF kernel is an inner product of molecular fields, and the model is linear in the field. That is
what makes the co-field interpretation possible: the dual coefficients turn into a field of regression
coefficients, and a prediction is its overlap integral with the molecule's own field. Two ways of making
the model more expressive are available in the *Model* panel, as *Nonlinear kernel* just below the
method and as *Mixing coeff. h* below the grids (CLI: `--kernel-layer`, `--mkl`), and they differ
exactly in whether that interpretation survives.

**A nonlinear layer on top of the kernel** (`cmfqsar/model.py`, `CMFModel.apply_layer`) composes a
second kernel with the CMF kernel, which is what a deep kernel is:

* `gauss` — a radial basis function of the distance between the fields,
  `exp(-γ d²)` with `d² = K(i,i) + K(j,j) - 2 K(i,j)`, the distance being taken in the Hilbert space of
  the fields. The bandwidth is expressed in units of the inverse median squared distance of the training
  molecules, so `γ = 1` is the median heuristic whatever the size of the fields, and the `γ` grid is
  searched together with α and λ.
* `arccos1` / `arccos2` — the degree-1 arc-cosine kernel of Cho & Saul (2009), the kernel of an
  infinitely wide neural network with one or two hidden layers, applied once or stacked twice. Its
  self-kernel is the base one, so the same diagonals serve every layer.

On the eight Sutherland sets (five physico-chemical fields, cosine-normalised kernels, α, λ and γ by
10-fold cross-validation on the training set, the published test set used once):

| Set | q²cv linear | gauss | arccos1 | arccos2 | R²pred linear | gauss | arccos1 | arccos2 |
|---|---|---|---|---|---|---|---|---|
| BZR | 0.430 | **0.530** | 0.518 | 0.535 | 0.032 | 0.084 | 0.067 | **0.096** |
| COX2 | 0.620 | 0.641 | **0.646** | 0.645 | 0.301 | 0.274 | **0.322** | 0.313 |
| DHFR | 0.669 | **0.679** | 0.675 | 0.678 | 0.533 | **0.632** | 0.569 | 0.581 |
| THR | 0.674 | **0.680** | 0.674 | 0.675 | 0.428 | 0.540 | 0.446 | **0.558** |
| THER | 0.447 | **0.526** | 0.485 | 0.518 | 0.283 | **0.349** | 0.244 | 0.315 |
| GPB | 0.603 | **0.625** | 0.607 | 0.616 | **0.558** | 0.487 | 0.498 | 0.510 |
| ACE | **0.699** | 0.679 | 0.680 | 0.677 | 0.464 | 0.518 | 0.532 | **0.546** |
| AChE | 0.516 | **0.589** | 0.579 | 0.585 | 0.407 | **0.760** | 0.644 | 0.719 |
| mean | 0.582 | **0.619** | 0.608 | 0.616 | 0.376 | **0.456** | 0.415 | 0.455 |

The Gaussian layer raises q²cv on seven of the eight sets and R²pred on six, by 0.08 on average; the
largest gain is on acetylcholinesterase (R²pred 0.41 -> 0.76), the only loss on test is glycogen
phosphorylase. A second arc-cosine layer is worth almost nothing over the first, as is usual for these
kernels: depth beyond one layer buys little when the base kernel is already rich.

The price is interpretation. With a layer active the model is no longer linear in the molecular field,
so there is no co-field and no decomposition into atoms; the application refuses to compute them and
says why, rather than drawing quantities that no longer mean what they claim. What does survive is
decomposition rule 2, the contributions `a_j K(X, X_j)` of the individual training molecules, which is
defined for any kernel and is still shown in the Contributions tab. The recommended use is therefore to
keep the linear model for interpretation and to report the nonlinear one beside it when prediction is
the goal.

**Multiple kernel learning** keeps the kernel linear. The joint kernel is a weighted sum of the per-field
kernels, `K = Σ_f h_f c_f K_f`, so choosing `h_f` *is* the multiple-kernel-learning problem, and the
co-fields of the result are as valid as ever. The weights are set by centred kernel alignment with the
ideal kernel `y yᵀ` (`align_mixing`), either per field (Cristianini et al. 2001) or jointly by
`min_{h ≥ 0} ‖Σ_f h_f K_f^c - y^c y^cᵀ‖_F` (Cortes et al. 2012), the joint solution being shrunk towards
equal weights because the field kernels of a congeneric series are strongly correlated and the
unshrunk solution is nearly sparse. The weights are re-derived inside every cross-validation fold, so the
reported q²cv does not use the left-out activities, and they are rescaled to keep the mean diagonal of
the joint kernel unchanged so that the λ grid stays comparable.

What it is worth depends on how different the fields are, and on the shrinkage. With the five
physico-chemical fields alone it buys nothing: their kernels are strongly correlated on a congeneric
series, so data-driven weights mostly add variance. Over the eight Sutherland sets, with those five
fields:

| mixing | q²cv | R²pred |
|---|---|---|
| fixed at 1 | 0.582 | 0.376 |
| align | 0.591 | 0.317 |
| alignf, default shrinkage | 0.596 | 0.311 |
| alignf, shrinkage off | 0.571 | 0.301 |

Cross-validation barely moves, and the fall in test-set prediction comes almost entirely from one set
(AChE, 0.407 -> -0.02), which is too thin a basis for a general claim; a benchmark over 86 data sets of
the other collections (`docs/benchmark.md`) finds the per-field alignment neutral in external validation
at the default shrinkage (+0.004 q²ex). Turning the shrinkage off is the one setting that does real harm
(-0.069 q²ex there): the unshrunk solution puts about three quarters of the weight on a single field and
zeroes two of the five, which on thrombin collapses the model (q²cv 0.674 -> 0.536, R²pred 0.428 ->
0.053). With the continuous indicator fields added, which really do differ in relevance, it pays:

| Set | fields | q²cv none | align | alignf | R²pred none | align | alignf |
|---|---|---|---|---|---|---|---|
| BZR | 18 | 0.420 | **0.452** | 0.451 | 0.039 | **0.040** | 0.000 |
| COX2 | 17 | 0.630 | **0.642** | 0.635 | 0.263 | 0.280 | **0.292** |
| DHFR | 19 | 0.635 | 0.650 | **0.651** | 0.583 | **0.593** | 0.588 |
| THR | 15 | 0.671 | 0.711 | **0.728** | 0.617 | 0.637 | **0.660** |
| THER | 15 | 0.546 | **0.549** | 0.496 | 0.400 | 0.426 | **0.440** |
| GPB | 12 | **0.647** | 0.607 | 0.574 | 0.544 | 0.547 | **0.566** |
| ACE | 17 | 0.723 | 0.721 | **0.724** | **0.635** | 0.606 | 0.562 |
| AChE | 14 | 0.532 | **0.537** | 0.529 | 0.577 | 0.601 | **0.625** |
| mean | | 0.601 | **0.609** | 0.599 | 0.457 | 0.466 | **0.467** |

The average gain is small (about 0.01 in both statistics) and not uniform: the joint solution helps
thrombin clearly (q²cv 0.671 -> 0.728, R²pred 0.617 -> 0.660) and hurts glycogen phosphorylase. The
weights themselves are informative, and that is arguably the better reason to use them: on thrombin the
H-bond donor field receives four times the weight of the others, which agrees with what the co-field of
that series shows.

### Shapley decomposition: interpretation that survives a nonlinear kernel

The decomposition rules of the papers are exact, but rule 3 splits a prediction into atoms only
because the model is linear in the field. A kernel layer destroys that, and with it the co-field and
the atom contributions. The Shapley value restores an additive attribution for any kernel, and it is
available in the *Shapley* tab of the results panel and as `cmf_cli.py explain`.

Its usual obstacle elsewhere in machine learning does not arise here. Shapley attribution needs a
meaning for "feature absent", which for ordinary descriptors has none, so a background sample and an
imputation rule have to be invented. The field of a subset of atoms is a genuine molecular field, and
because the kernel is bilinear the kernel of any subset against any training molecule is a sum of rows
of one array that the application already computes. Every coalition value is therefore the model's own
prediction for a real field, with no surrogate model and no background set:

```
v(S) = sum_j a_j kappa( K(X_S, X_j), K(X_S, X_S), K(X_j, X_j) ) + b
```

Three sets of players are offered. **Atoms** give one value per atom, which the viewer draws as red and
blue spheres exactly where the atom contributions of a linear model would be. **Substituents** hold the
maximum common substructure of the series fixed, so the baseline is the scaffold rather than empty
space and the values explain the difference from the reference compound; there are then few enough
players to enumerate every coalition. **Field types** say what each field contributes, which is what the
overlap integral of co-field and molecular field reports for a linear model and nothing reports for a
nonlinear one.

Two properties make this an extension of the published rules rather than a competitor. For the linear
kernel without normalisation the game is additive, so the Shapley value of an atom is exactly its
contribution under rule 3 and every interaction index is zero; both are asserted in the tests to about
1e-16. The field game is additive for any linear model, normalised or not, because each field is
normalised on its own, so there the Shapley values reproduce the published per-field contributions
exactly. And because a nonlinear layer is precisely what makes the game non-additive, the second-order
Shapley interaction indices of Grabisch and Roubens measure what the linear model cannot express,
namely the part of a prediction that two substituents produce only together.

| Case | Players | Cost |
|---|---|---|
| phenol, 13 atoms, exact | 13 | 8194 coalitions, well under a second |
| thrombin ligand, 75 atoms, sampled | 75 | 15202 coalitions, about a second, standard error 0.003 in pKi |
| substituents over a scaffold, exact | 2 to 10 | tens to thousands of coalitions |

#### Shapley fields

Taking the players to be (atom, field type) pairs gives one value per atom per field, and those values
can be put back where they came from: each is given the shape of its atom's own Gaussian, with the
prefactor that makes the Gaussian integrate to one,

```
Phi_f(r) = sum_l phi_lf (b_l / 2 pi)^(3/2) exp(-b_l/2 |r - r_l|^2)
```

so that the integral of the map over space is the sum of its values, which is the part of the
prediction attributed to that field. It is the same relation that holds between partial charges and
the charge density they generate, and it is what makes the picture quantitative rather than
decorative. The result is one map per field type, drawn with the same isosurfaces, colours and iso
level as a co-field, which is what a model with a nonlinear kernel otherwise cannot have. Switch the
players to *atoms x field types* in the Shapley tab, then tick **Shapley field Φ(r)** above the 3D
view; *File -> Export current Shapley field* writes a Gaussian cube, so the map opens beside the
protein in any viewer. On the command line it is `--players atomfields`, with `--cube` to write one
cube per field.

For a linear kernel without normalisation these values are exactly the per-field atom contributions of
rule 3, to about 1e-14, so the map is a strict generalisation of what the papers already decompose.
A CDK2 ligand of 49 atoms gives 245 players, which 120 permutations settle in about a second with a
standard error of 0.005 on the pAffinity scale.

The difference from a co-field is not the width of the Gaussians, which only smooths the picture, but
the scope: a co-field belongs to the model and is the same for every molecule, while a Shapley field
belongs to the model *and one molecule*. With a nonlinear kernel no global object exists to be
approximated, and averaging the maps over a series, or over the actives minus the inactives, is the way
back to a picture of the series as a whole.

Three caveats are worth stating. The baseline decides the answer, and the empty field is an
extrapolation no training molecule resembles, which is why the scaffold baseline exists. Cosine
normalisation couples the atoms through the norm of the whole molecule, so even a linear model is not
additive under it and the Shapley values then redistribute what rule 3 assigns; both still sum to the
prediction. And the result is a per-molecule explanation. A nonlinear model has no global co-field, and
no attribution method invents one.

### Protein binding sites behind the fields

The free-energy benchmark collections (`examples/openff`, `examples/schindler`) give their ligands in
the coordinate frame of a reference crystal structure, and every record carries the identifier of that
structure, so nothing has to be superimposed: fetching the entry puts the protein exactly around the
training set. Co-fields, molecular fields and Shapley maps are computed on a grid built from the same
coordinates, so they land in the site as well.

`examples/pockets/` holds the sites of the benchmark sets, already trimmed and already in the frame of
their ligands, so **opening a free-energy set brings its protein with it**: `examples/openff/cdk2.sdf`
finds `examples/pockets/openff/cdk2.pdb` and the pocket appears around the molecules without a single
further click. Each file holds the residues within 8 Å of any ligand of the set, taken whole, without
waters and without the co-crystallised ligand, and records its provenance in `REMARK 400` lines. They
are small, 23 to 61 kB each and 2.9 MB for all 73, and `examples/make_pocket_examples.py` rebuilds or
extends them.

*File -> Fetch protein from the PDB* downloads an entry from RCSB and caches it under `~/.cmfqsar/pdb`;
the dialog is pre-filled with the code the data set names. *File -> Load protein structure* reads a
local PDB or mmCIF file instead. The **protein site** controls above the 3D view then draw the residues
within a chosen distance of the molecules on screen, as a wire frame or as sticks, with hydrogens and
crystallographic waters optional and the residue names available in the legend. Residues come in whole,
ions and cofactors are kept, and the co-crystallised ligand is removed so that it does not sit on top of
the series. The reader is deliberately independent of RDKit, because crystal structures carry alternate
locations, missing elements and non-standard residues that a chemistry toolkit would refuse to sanitise,
and none of that matters for drawing.

Loading reports the closest approach between the molecules and the polymer, which is the frame check:

| data set | structure | closest approach | site at 5 Å |
|---|---|---|---|
| openff/cdk2 | 1H1Q | 1.64 Å | 25 residues, 200 atoms |
| schindler/eg5 | 3L9H | 0.26 Å | 63 residues, 492 atoms |

A few ångström means the two are in the same frame. Tens of ångström means they are not, which happens
when the structure belongs to a different series, or when the molecules were realigned after loading:
the alignment tools, the confidence alignment and *Unalign* all leave the crystal frame, and the
Sutherland and Mittal sets were aligned by their authors in arbitrary frames with no protein at all.

The same check decides what `examples/pockets/` contains. Of the 98 sets of the three free-energy
collections, 73 have ligands sitting in the frame of the entry they name, within 2.4 Å of the protein
in every case; the other 25 are off by 8 to 58 Å and no pocket is stored for them. Those sets are not
broken, their poses were simply prepared in some other frame, and the application will say so if a
structure is loaded for them by hand.

Both sides can be labelled. **Labels** above the 3D view writes a code at every atom of the selected
molecule, either element and number as the contribution tables number them, or the Sybyl atom type that
the indicator fields use, or the partial charge that enters the electrostatic field; the numbering is
that of the whole molecule, so it still matches the tables when hydrogens are hidden. **Residue labels**
in the protein row writes the name and number of each residue of the site at its alpha carbon, or at the
centroid of a residue that has none, such as an ion or a cofactor, and drops the chain letter when the
site has only one chain.

Overlaying a co-field on the site is a real test rather than a decoration. The model never sees the
protein, so its favourable lobes should fall into open subpockets and its unfavourable ones against the
walls; where they do not, the alignment or the model is suspect.

### Complementarity with the pocket, as numbers

A pocket is a set of atoms, so the parameterisation that builds the fields of a ligand builds the
fields of the protein, and the overlap of two Gaussian fields has the same closed form on either
side. The *Complementarity* tab of the results panel (CLI: `cmf_cli.py complement`) turns that into
three answers, in `cmfqsar/complementarity.py`.

**How complementary is a molecule to the pocket.** Overlap integrals between the ligand fields and
the pocket fields, exact and without a grid, each divided by the two norms so that it is a cosine in
[-1, 1] and does not grow with molecular size. Complementarity is not similarity: the electrostatic
term carries a minus sign, because a positive ligand region should meet a negative protein region,
the hydrophobic term carries a plus, and the hydrogen-bond terms pair the donors of one side with
the acceptors of the other. Steric overlap is reported separately as burial, since bulk against the
protein is contact and clash at once, and the number of clashing heavy-atom pairs is reported beside
it, because a complementarity score says nothing about whether the molecule fits at all.

**Where on the molecule it holds.** The same quantities on a smooth van der Waals surface of the
molecule give a value per vertex, which the viewer draws as a green-to-red surface, and a
correlation per field over the surface. For electrostatics the comparison uses potentials rather
than the Gaussian charge densities, because the potential is what the other side feels; that number
is the electrostatic complementarity score of the literature (Bauer & Mackey, J. Comput. Aided Mol.
Des. 2009, and J. Med. Chem. 62 (2019) 3036), reported so that positive means complementary.

**Whether the model agrees with the protein.** A co-field says where more field raises activity; the
pocket says what it offers. Their agreement over the space the series occupies, weighted by that
occupancy because a co-field is unconstrained where no molecule reaches, is a test of a model that
never saw the structure. The default statistic is the magnitude-weighted agreement of sign, which is
how a co-field is read; a cosine, which also demands matching shapes, and a correlation, which only
asks whether the variations track, are available too. The steric row compares the co-field with the
free space around the protein rather than with the protein's own steric field, which is positive
everywhere and would merely report the sign of the co-field.

Two practical findings from building this. Ionising the side chains matters more than anything else:
a PDB file carries no charges, so a reader leaves aspartate protonated and lysine neutral, and with
that left alone the thrombin series gives an electrostatic surface score that correlates **-0.31**
with activity, while setting the standard charges at pH 7 turns the same number into **+0.56**, as a
P1 group meeting Asp189 should. And agreement is a per-field, per-target diagnosis rather than a
score to rank models by: over six targets with 37 to 44 ligands each, the mean agreement ranged from
-0.18 to +0.20 with no clear relation to q²cv, while the hydrophobic agreement reached +0.69 on all
three HIF-2α sets, whose cavity is a buried hydrophobic one.

What this is not is a binding energy. Desolvation, entropy, bridging waters and side-chain motion are
all absent, charges on a truncated pocket are approximate at the cut ends, and the structure is one
rigid snapshot.

### Chemical space and activity landscapes (kernel GTM)

*Tools -> Chemical space map (GTM)* (CLI: `cmf_cli.py map`) builds a two-dimensional map of a data set
from its CMF kernels, in `cmfqsar/gtm.py`. Molecular fields are not vectors, so Generative Topographic
Mapping (Bishop, Svensen & Williams 1998) cannot be applied to them directly; only the kernel is
available. The CMF kernel is an inner product in the Hilbert space of the fields, so kernel PCA on it is
functional PCA and its leading components reproduce the feature-space distances exactly. Running the
ordinary expectation-maximisation GTM in those coordinates is equivalent to a kernelised GTM (cf. Olier,
Vellido & Giraldo, ESANN 2010) and keeps the out-of-sample projection a single kernel row, which a map
needs in order to place new molecules.

The molecules drawn over the landscape can be coloured by their value, or shown as white markers with a
black outline when the landscape itself is the message, or hidden altogether; their size is adjustable.
The control sits above the map and redraws without refitting.

From the responsibilities follow the **activity landscape** (responsibility-weighted mean activity per
node), the **class landscape** (weighted fraction of actives) and the **data density**, which is the
applicability domain; nodes without support are left blank. A property can also be *predicted* from a
landscape, so the map is a model and not only a picture: on the thrombin training set the landscape
reproduces the activities with R² 0.92.

Two points specific to fields. Cosine-normalised kernels are used, otherwise the leading components
follow molecular size. And a CMF kernel between molecules that are not aligned in a common frame is
meaningless, so a map spanning several series would partly show alignment quality — unless the
alignment-free option is used: as the attenuation factor tends to zero a continuous indicator field tends
to the count of atoms of that type (Sitnikov et al. 2015), so the kernel becomes a fragment-descriptor
kernel. With α = 0.02 and indicator fields only, a map of 120 molecules drawn from three different
targets separates them cleanly (centroid separation 1.56 against a within-target spread of 0.60), with no
alignment at all. At α = 0.02 the kernel is not yet strictly pose-free — the dependence dies only
linearly in α, see the heterogeneous-series section — but the separation is: rotating and translating
every molecule at random leaves it at 1.48, and with the strictly pose-free kernel of
`chemotypes.count_kernel` it is 1.40 before and after, to the last digit.

### Classification and virtual screening

The continuous molecular fields enter only through the kernel, so any kernel method can be used with
them; the 2013 paper lists regression, classification, clustering, ranking, dimensionality reduction and
novelty detection, and states that fields of model coefficients exist for classification too, where "in
the case of SVM models they describe the angular orientation of the hyperplane separating active from
inactive compounds in the infinite-dimensional feature space". Both classification modes are available in
the Task selector of the Model panel (CLI: `--task classification` / `--task oneclass`):

* **Binary classification** (C-SVC, ν-SVC, or the Gaussian-process classifier described below).
  Classes come either from a threshold on the activity
  (`--threshold`, with `--active-below` when a smaller value means more active) or from an SD property
  (`--class-property`). The hyper-parameters are selected by cross-validation, minimising 1 − AUC.
* **One-class classification**, fitted to the actives alone, which is the similarity-based
  virtual screening of the 2019 paper: the 1-SVM, or one of the four other descriptions of the
  active class described just below, which beat it on every set tested. No cross-validated objective exists without negatives, so α (the
  level of generalisation, Fig. 5 of that paper) and ν (the expected fraction of outliers) are set
  directly rather than searched, and the program says so.

The **Scatter plot** tab means something for a classifier only when the classes come from a threshold on
a continuous activity. It then shows the decision value against that activity, coloured by true class
(circles for training, squares for test), with the threshold and the decision boundary drawn as dashed
lines, so the four quadrants are the true and false positives and negatives. When the classes come from a
property instead, or no activity values exist, the tab is hidden: a classifier is judged by its ranking
and its ROC curve, not by a regression-style scatter.

Both modes report their scores: the **Scores** tab holds the ranked decision values of every molecule
with a histogram of their distribution (actives against inactives where the classes are known, inside
against outside the boundary otherwise) and the decision boundary marked, and the **ROC** tab the curves
with their AUC. For a one-class model the statistics also give the number of support vectors and the
range of the decision values, so a screening run without any class labels still reports its ranking.
`--predictions` writes the decision values as CSV and `--json` the whole result.

All three decomposition rules carry over unchanged, because the kernel is still a sum over atom pairs:
the co-field is a weighted sum of training-set fields, now signed by class (`a_j = α_j y_j`), so its
positive zones mark field values that push a compound toward the active class; a prediction decomposes
into contributions of training molecules, and support vector machines make this sparse, so a
classification is explained by a few reference compounds; and atom contributions are computed exactly as
for regression. This is covered by a test asserting that atom and training-molecule contributions sum to
the decision value.

**Evaluating a screening model needs a background set.** Every screening metric requires negatives, so
without decoys the program reports only the ranking and the distribution of the scores, never a
performance figure. Load a background file with *Add decoys…* in the data-set panel (CLI: `--decoys FILE`):
those molecules are presumed inactive, are never fitted or cross-validated on, survive a random split,
and are used only to compute the AUC, the enrichment factors in the top 1, 5 and 10 % and the
accumulation curve in the **Enrichment** tab. They need not be verified inactives; random library
molecules or property-matched sets (DUD-E, DEKOIS, MUV) are the usual choice. The resulting numbers are a
property of the background as much as of the model: decoys unlike the actives in size, charge or shape
make the task trivial, and in a field method a decoy may also score low merely because it was aligned
badly, so the evaluation measures the alignment and the model together.

**Decoys must be posed, and the program poses them automatically.** A background molecule has to be
placed in the frame of the training actives before its fields can be compared with theirs, and a
molecule posed badly scores low for the wrong reason, which flatters the model. *Tools -> Align decoys
to the training set* (CLI: `--align-decoys`) therefore gives every decoy its **best** pose: it maximises
a smooth maximum of the cosine-normalised CMF similarity between the decoy and the training molecules,
over the six rigid-body parameters, with analytic gradients and several random starting orientations
(`cmfqsar/decoys.py`). The evaluation is then conservative, since whatever enrichment survives is not an
artefact of sloppy placement. On the thrombin example the similarity of 25 unrelated ligands rises from
0.31 to 0.49 on average, and the enrichment is unchanged.

`examples/decoys/` holds ready-made background sets built this way by `examples/make_decoy_sets.py`:
for each target the decoys are ligands of the *other* example series, presumed inactive against it and
already aligned into its frame, tagged `subset = decoy` so they load as a background set directly.

**`--matched` builds the harder kind, and it turns out hardly to be harder.** The DUD-E recipe is to
pick, for every active, molecules that match its molecular weight, logP, donors, acceptors, rotatable
bonds and net charge while staying topologically unlike it (here: ECFP4 Tanimoto below 0.35 to every
active), drawn from the 3805 unique molecules of the other example series. Those files are shipped too,
as `*_decoys_matched.sdf`. Measured against them, with the five one-class descriptions swept over their
parameters on THR, GPB, ACE and AChE, the mean AUC falls only from 0.986–1.000 to 0.978–0.998, and on
AChE every method still scores exactly 1.000 on both backgrounds.

The reason is worth stating, because it limits what any decoy set can do here. Matching six scalar
properties does not match a three-dimensional field: the actives of a congeneric series resemble one
another far more than anything from another series does, whatever its molecular weight. So for a
congeneric series, decoy-based screening metrics sit at the ceiling and cannot tell two methods apart
— the discriminating negatives are the *weaker molecules of the same series* (AUC 0.6–0.8, see the
one-class section above), or genuinely measured inactives such as LIT-PCBA. The decoy sets here are
useful for checking that a screening model works at all, not for comparing models.

A one-class model scores how typical a molecule is of the training actives in field space. That is what
it is for, and it is not the same as activity. Fitted to the 31 thrombin actives (α = 0.5, ν = 0.2) and
used to rank the 13 test actives against 30 COX-2 molecules placed in the thrombin frame by maximising
their field similarity to the reference ligand, it gives **AUC 1.000** and the maximal possible
enrichment in the top 10 %. The same model ranking those actives against the closely related *inactives*
of the thrombin series itself gives only AUC 0.69, because molecules of the same scaffold look equally
typical. Use binary classification for that case, and one-class models for screening against a
structurally different background.

**One-class models need normalised kernels**, and the program switches them on for this task. On raw
kernels the decision value scales with the self-kernel, so the ranking follows molecular size rather than
similarity to the target class: the same thrombin model scores AUC 0.34 on raw kernels against 0.69 on
normalised ones.

#### Four other descriptions of the actives

A one-class SVM encloses the actives in a ball in the field Hilbert space. That is one geometry
among several, and the others are cheap here because the kernel is already computed, so
[oneclass.py](cmfqsar/oneclass.py) adds four more, all with the same meaning of ν (the fraction
of the training actives left outside) so that they can be compared without re-tuning:

| key | description |
|---|---|
| `ocmean` | distance to the centroid of the actives — with a normalised kernel this is one minus the mean field similarity, the similarity ensemble of the screening literature, with nothing fitted |
| `ocknn` | distance to the k-th nearest active, the classical similarity search |
| `ocgp` | a Gaussian process regressed on a constant target of one (Kemmler et al. 2013); `mean_over_sd` divides by the posterior standard deviation, so a molecule the model knows nothing about is punished as well as one that is unlike the actives |
| `ockpca` | distance from the principal subspace of the actives (Hoffmann 2007) — a congeneric series lies on something much flatter than a ball |

*SVDD*, the usual alternative, is deliberately absent: for a kernel with a constant diagonal the
minimum enclosing hypersphere and the ν-SVM solve the same problem, and a one-class model here
always normalises its kernels, because an unnormalised one-class score ranks by molecular size.

`ocmean` and `ocgp` are linear in the kernel row, `score(x) = Σ_j a_j K(x, x_j) + b`, so the
co-field and the atom contributions carry over unchanged; `ocknn` and `ockpca` are a minimum and
a quadratic form, and the program says so rather than drawing a co-field that does not exist.

**Measured.** The actives of each set (activity above the median) are the training class, scored
by five-fold cross-validation so that no active is ever scored by a model that saw it; α = 0.5,
ν = 0.1, k = 5, five components, nothing tuned for any method. AUC against the inactives of the
same set, which are already in the same frame:

| set | 1-SVM | centroid | k-NN | GP | GP/sd | kernel PCA |
|---|---:|---:|---:|---:|---:|---:|
| COX2 | 0.776 | **0.800** | 0.726 | 0.685 | 0.772 | 0.756 |
| DHFR | 0.685 | 0.785 | 0.789 | 0.748 | **0.833** | 0.799 |
| BZR | 0.569 | 0.557 | **0.717** | 0.652 | 0.634 | 0.641 |
| AChE | 0.505 | 0.522 | 0.609 | **0.740** | 0.647 | 0.605 |
| THR | 0.651 | 0.751 | 0.792 | 0.775 | 0.735 | **0.794** |
| mean | 0.637 | 0.683 | **0.727** | 0.720 | 0.724 | 0.719 |

The ball comes last. It is beaten on every single set, by 0.02 on COX2 and by 0.15 to 0.24 on
DHFR, BZR, AChE and THR, and the enrichment in the top 1 % says the same more bluntly: the
molecules the 1-SVM ranks highest are inactives on four of the five sets (EF 0.00), while k-NN
and kernel PCA reach the ceiling that the roughly equal class sizes allow (EF ≈ 2) on all five.

Nothing above was tuned, so it is a comparison of defaults. Sweeping each description over the
attenuation, over ν and over its own parameter — 15 to 75 settings each — separates two
questions. At its best setting the 1-SVM reaches 0.727 and is fourth of five, ahead of the
parameter-free centroid (0.710) and behind kernel PCA (0.781), the Gaussian process (0.767) and
k-NN (0.824). Taken over all its settings it is last: median AUC 0.672, against 0.685, 0.695,
0.713 and 0.730. So the ball is not hopeless, it is fragile — and since a one-class model has no
cross-validated criterion to tune against, the median is the number a screening run should
expect, not the best. k-NN is first on both counts, and on four of the five sets.

**And a warning the program already gives, now with a number.** The same five methods were also
scored against 150 molecules drawn from the other four targets and posed into the frame by the
decoy aligner. Every method scored AUC between 0.965 and 1.000 there, and the enrichment factors
were identical across methods to the last digit. A background that unlike the actives measures
nothing: it separates on size and shape before any field model is involved. The inactives of the
same series are the harder and more honest negatives, and the decoy numbers in that experiment
say more about the background than about the model — which is exactly what the section above
warns of.

**Against fingerprints.** The Tanimoto coefficient of binary fingerprints is itself a positive
semi-definite kernel, so the same five descriptions run on it unchanged and only the meaning of
"similar" changes: an overlap integral of molecular fields in a common frame, or a count of
shared substructures with no geometry at all. Under the same sweep and the same protocol, mean
AUC of the best description (k-NN, on every one of the six similarities):

| similarity | k-NN, best setting | k-NN, median over settings |
|---|---:|---:|
| CMF fields, α = 1 | **0.807** | **0.730** |
| CMF fields, α = 0.5 | 0.798 | 0.728 |
| ECFP4, 2048 bits | 0.802 | 0.718 |
| ECFP6, 2048 bits | 0.791 | 0.717 |
| MACCS, 166 keys | 0.764 | 0.654 |

On average the three-dimensional fields buy 0.005 over ECFP4 at their best setting and 0.012 at
the median — nothing. The averages hide a real disagreement, though: per target the gap is large
and changes sign. Fields win on COX2 (0.826 against 0.713) and THR (0.816 against 0.763),
fingerprints win on AChE (0.836 against 0.738) and BZR (0.790 against 0.732), and DHFR is a draw.
The two representations see different things, and neither dominates; what the mean says is only
that no one of them is a safe default. MACCS is clearly behind both, so the measurement does
discriminate between representations when there is something to discriminate.

Two further things fall out of that run. k-NN is the best description on *all six* similarities,
so its advantage is a property of the geometry rather than of the fields. And the ball recovers
on fingerprints — the 1-SVM is last on every field kernel (median 0.656 to 0.676) but competitive
on ECFP6, where its median of 0.724 is the best of the five. Its weakness above is specific to
the kernel it was asked to enclose.

This is a screening task judged on same-series inactives, not a QSAR regression, so it says
nothing about what the fields are for — co-fields, atom contributions and complementarity with a
pocket have no fingerprint counterpart. It does say that for *ranking by similarity to a set of
actives*, the alignment and the overlap integrals are not buying what one might hope.

```bash
python cmf_cli.py build examples/sutherland/THR.sdf --charges file --task oneclass \
       --threshold 6.7 --method ockpca --alphas 0.5 --nus 0.1 --oc-components 5
```

In the application the five descriptions are the method list of the one-class task, with the
neighbours or the subspace dimension beside them; the one-class Gaussian process takes its ridge
from the first value of the λ grid.

### A Gaussian-process classifier, for the posterior rather than the accuracy

A support-vector classifier returns a decision value and nothing else: a probability has to be
obtained by fitting a sigmoid to it afterwards, and there is no posterior at all. The third
classification method, `gpc` ([gpc.py](cmfqsar/gpc.py)), is a Gaussian process with a Bernoulli
likelihood, fitted at the posterior mode by the Laplace approximation (Rasmussen and Williams,
chapter 3: the Newton iteration of Algorithm 3.1, the predictions of 3.2 and the marginal
likelihood of 3.3). It works from the same precomputed kernels, and it was added for three
things the hinge loss cannot give.

**Calibrated probabilities.** `p(active)` comes from the likelihood, averaged over the latent
posterior — exactly for the probit link, by MacKay's approximation for the logistic one — so a
molecule the model knows nothing about is reported near one half instead of confidently. The
cross-validated probabilities are close to honest; predicted against observed, in bins, on COX2
(Brier 0.128) and DHFR (0.137):

| predicted | 0.0–0.2 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–1.0 |
|---|---:|---:|---:|---:|---:|
| COX2 observed | 0.05 | 0.19 | 0.45 | 0.77 | 0.91 |
| DHFR observed | 0.03 | 0.18 | 0.47 | 0.80 | 0.93 |

The deviation that remains is the known conservatism of the Laplace approximation at the ends,
and it is in the safe direction: the model is slightly less sure than it turns out to be.

**A posterior variance per molecule.** The latent variance returns to the prior for a molecule
unlike anything in the training set, which is what makes "active" and "no idea" different
answers, and it is what the confidence-based alignment of [varalign.py](cmfqsar/varalign.py)
needs in order to work on a classification set at all.

**The marginal likelihood.** It scores a kernel without holding anything out, so the attenuation
and the amplitude can be chosen from the training molecules alone — one fit per candidate instead
of ten. Measured against the usual cross-validated search on the six sets, it picks a different
pair on five of them and loses nothing in ranking: mean AUC 0.789 by evidence against 0.788 by
cross-validation. By Matthews correlation it is 0.033 behind on average, which is what one should
expect, since the cross-validated search is choosing on exactly that statistic and the evidence
has never seen it.

Accuracy itself is a draw, which is why this method is not the default. Best cross-validated MCC
over the same α grid, ten folds:

| set | GP classifier | C-SVC | Brier (GPC) |
|---|---:|---:|---:|
| THR (cut 6.7) | +0.389 | **+0.411** | 0.201 |
| COX2 | **+0.704** | +0.646 | 0.128 |
| DHFR | +0.662 | **+0.673** | 0.137 |
| BZR | +0.485 | **+0.497** | 0.173 |
| AChE | +0.534 | **+0.576** | 0.195 |
| mittal/ACHE | **+0.243** | +0.207 | 0.242 |

The mean difference is +0.001 in MCC and −0.009 in AUC: the two are the same classifier as far as
these sets can tell. That was the expected answer — a plain ridge on ±1 labels, which is what
`method="gp"` already was, reaches within 0.02 MCC of the SVMs once its decision threshold is
chosen on the training folds rather than left at zero, so most of the apparent gap between the
squared loss and the hinge loss was never about the loss. What the Bernoulli likelihood adds is
the principled way of placing that boundary, and the three things above.

The latent mean stays a kernel expansion, `f(x) = Σ_j a_j K(x, x_j)`, so unlike a nonlinear
kernel layer this classifier leaves the interpretation intact: the co-field, the training-molecule
contributions and the atom contributions are computed exactly as for kernel ridge regression. In
the application the method sits in the Model panel's classification list, takes its amplitude from
the C grid, prints its evidence and Brier score in the Statistics tab, and fills the Scores tab
with probabilities rather than decision values.

```bash
python cmf_cli.py build examples/sutherland/AChE.sdf --charges file --task classification \
       --threshold 6.84 --method gpc --Cs 1,10,100
```

### Alignment by prediction confidence

The Conclusions of the 2019 paper propose aligning molecules by maximising the confidence of their
prediction, that is by minimising the posterior variance `sigma^2(x) = K(x,x) - k(x)^T (K + lambda I)^-1 k(x)`,
which moves a molecule into the applicability domain of the model. This is implemented in
`cmfqsar/varalign.py` (GUI: *Tools → Align by prediction confidence*; CLI: `cmf_cli.py predict
--align-variance`). `K(x,x)` depends only on the interatomic distances inside the molecule, so it is
invariant under rigid motion; minimising the variance is therefore exactly maximising the quadratic form
`k(x)^T (K + lambda I)^-1 k(x)`, i.e. similarity to the whole training set with redundant training
molecules down-weighted by the inverse kernel matrix. The search is a six-parameter rigid-body
optimisation with analytic gradients (L-BFGS, random restarts) and an optional guard on the displacement
from the input pose. No activity value of the aligned molecule is used, so the procedure is valid for
prospective prediction.

Two properties of the criterion are built in and reported by the program:

* It aligns molecules against a **fixed** training alignment. Applied to the training set itself the
  optimum is degenerate, because superimposing all molecules maximises mutual similarity and drives
  every variance to zero while destroying the structure-activity signal. The tool therefore refuses to
  move training molecules.
* Because the pose is chosen to make the model confident, the posterior variance of an aligned molecule
  is no longer an independent measure of its applicability domain.

On the thrombin set (59 training, 29 test molecules) every test molecule was rotated by 40° and translated,
giving 3.35 Å mean deviation from the published alignment. Minimising the variance returned 28 of the 29
to within 1 Å (0.27 Å on average) and restored the prediction error from 1.49 to 0.84 log units, against
0.83 for the published poses.

#### Unalign: the control experiment

*Tools → Unalign (randomise poses)* (CLI: `cmf_cli.py unalign in.sdf out.sdf`) gives every chosen
molecule a random pose: a rotation about its own centroid and a translation, either bounded (a maximum
angle and displacement) or a uniformly random orientation that destroys the alignment completely. The
transformation is rigid, so the internal geometry, and with it the self-kernel of every molecule, is
unchanged; only the mutual arrangement is scrambled. This is the control for any alignment method:
randomise, realign, and measure how much of the original arrangement comes back. The GUI remembers the
poses from before the randomisation and a subsequent alignment reports its recovery against them.

#### Aligning the training set as well

The criterion can also be applied to the training set, cross-validated so that in every fold the held-out
molecules are aligned against a model fitted to the remaining ones and no molecule is ever aligned against
a model that contains it (GUI: *Tools → Align training set by prediction confidence*; CLI:
`cmf_cli.py build --align-train-variance`). The iteration converges: steps shrink by about a factor of two
per round and the contraction of the ensemble is mild.

**The predictive gain is not reliable, and q² must not be used to judge it.** Realigning the training set
changes the geometry q² is computed on, and in the tests below q² and the external R²pred repeatedly moved
in *opposite* directions. Judge the result on a test set that was not realigned.

| Set, starting alignment | q² before → after | R²pred before → after |
|---|---|---|
| Thrombin, published | 0.674 → 0.706 | 0.428 → 0.192 |
| Glycogen phosphorylase, published | 0.540 → 0.493 | 0.545 → 0.554 |
| Glycogen phosphorylase (Mittal), automatic | 0.130 → 0.082 | 0.269 → 0.010 |
| Catechol-O-methyltransferase (Mittal), automatic | 0.468 → 0.385 | −0.213 → 0.103 |

Three rounds each, no displacement limit. One set improved, one degraded badly, two were neutral, and
rounds after the first consistently made the external prediction worse. The defaults are therefore a
single round with a 1 Å displacement guard, and the hyper-parameters are refitted on the new geometry.

## Installation

Python ≥ 3.10 with the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

(RDKit, PySide6, pyqtgraph + PyOpenGL for the 3D view, scikit-image for marching cubes, scikit-learn for SVR.)

## Running the application

```bash
python cmf_app.py
```

or directly with a data set:

```bash
python cmf_app.py examples/benzamides_synthetic.sdf --activity pActivity --test-fraction 0.25
```

### Typical workflow

1. **Data set → Open training set…** — SDF/MOL2 with 3D coordinates of *aligned* molecules (hydrogens are
   added with coordinates if absent; SMILES/CSV files are embedded in 3D with ETKDG+MMFF). Choose the
   SD property holding the activity. Use *Random split* or the *→ test* button to define a test set, or
   *Add test set…* to load a separate file.
2. If the molecules are not aligned: **Tools → Align data set to reference…** (maximum common substructure
   superposition, optional refinement by field similarity).
3. **Model panel** — choose the fields (tick *Continuous indicator fields* to add one indicator field per
   atom type occurring in at least *n* training molecules), the learner, the α and λ grids, and the number of
   CV folds. *Normalise kernels* uses cosine-normalised field kernels; *Scale fields* rescales every field
   kernel to unit mean diagonal before mixing (untick to use the raw-kernel convention of the papers).
4. **Build model** — runs the grid search, the final fit, the internal cross-validation and the test-set
   prediction. Results appear in the *Statistics*, *Scatter plot* and *Hyper-parameter search* tabs and in
   the data-set table. **External CV** runs the nested cross-validation procedure.
5. **Compute co-field** — evaluates the co-fields of all field types on a grid around the training set.
   Select the field type, move the *iso level* slider (percent of the maximum absolute value), and tick
   *molecular field*, *overlap* or *atom contributions* to analyse the molecule currently selected in the
   data-set table. The *Contributions* tab lists the training molecules contributing most to its prediction.
6. **File → Save model…** stores everything needed for later predictions (`.cmf`, joblib); **Export …**
   writes predictions (CSV), the aligned data set (SDF) or the current co-field (Gaussian cube, readable
   by PyMOL/VMD/Jmol).

Co-field colours: electrostatic red(+)/blue(−); steric green(+)/yellow(−); hydrophobic orange(+)/light
blue(−); H-bond acceptor magenta(+)/dark red(−); H-bond donor cyan(+)/purple(−); CIF red(+)/blue(−)
(preferable / undesirable positions of the atom type).

## Command line

```bash
python cmf_cli.py build examples/benzamides_synthetic.sdf --activity pActivity --cif \
       --alphas 0.3,0.5,0.8,1.2 --lams 1e-3,1e-2,0.1,1 --test-fraction 0.25 --nested --save model.cmf
python cmf_cli.py predict model.cmf new_molecules.sdf --out predictions.csv
python cmf_cli.py predict model.cmf new_molecules.sdf --align-variance 2.0 --aligned-out aligned.sdf
python cmf_cli.py build aligned_set.sdf --align-train-variance 1.0 --test-fraction 0.25
python cmf_cli.py unalign aligned.sdf scrambled.sdf --max-rotation 40 --max-translation 2
python cmf_cli.py map examples/sutherland/THR.sdf --charges file --out coords.csv --landscape-out land.csv
python cmf_cli.py build examples/sutherland/THR.sdf --charges file --task classification --threshold 6.7
python cmf_cli.py build examples/sutherland/THR.sdf --charges file --task oneclass --threshold 6.7 \
       --alphas 0.5 --nus 0.2 --decoys background_library.sdf
python cmf_cli.py build examples/sutherland/THR.sdf --charges file --kernel-layer gauss \
       --gammas 0.25,0.5,1,2,4
python cmf_cli.py build examples/sutherland/THR.sdf --charges file --cif --mkl alignf
python cmf_cli.py explain model.cmf examples/sutherland/GPB.sdf --players groups --interactions
python cmf_cli.py complement examples/openff/thrombin.sdf --model thrombin.cmf --out scores.csv
python cmf_cli.py build examples/sutherland/ACE.sdf --charges file --conformations ensemble        --conformers 5 --prune-rmsd 1.0
python cmf_cli.py mil examples/sutherland/THR.sdf --charges file --conformers 5 --hard
python cmf_cli.py mil examples/schindler/hif2a-jmc2023.sdf --conformers 5 --dock
python cmf_cli.py hetero examples/sutherland/COX2.sdf --charges file --min-group 3 --out groups.csv
python cmf_cli.py hetero examples/mittal/ACHE.sdf --align-within --place field --out-sdf placed.sdf
```

`python cmf_cli.py build -h` lists all options (method, kernel normalisation, mixing-coefficient
optimisation, nested-CV repeats, …).  Literature data sets with a separate activity table carrying
the published train/test split (third column 1/2) and MOL2 partial charges are handled like this
(the settings reproduce the protocol of the 2013 paper, see `docs/benchmark.md`):

```bash
python cmf_cli.py build ace-all.mol2 --activity-file ace-data.txt --charges file --param tripos \
       --raw-scale --cv-no-shuffle --lam-search brent --alphas 0.01,0.05,0.1,0.25,0.5,0.75,1,2,3,5,10 \
       --lams 0.01,20 --nested --nested-on all --json ace.json
```

`--param` selects the parameterisation of the hydrophobic and hydrogen-bond fields (`rdkit`,
`tripos`, `tripos-legacy`, see below), `--no-bias` drops the bias term of the KRR system,
`--cv-no-shuffle` uses interleaved folds in file order, `--lam-search brent` minimises RMSEcv over a
continuous λ inside the `--lams` range, `--nested-on all` runs the external cross-validation on the
training *and* test molecules.

The continuous-indicator-field models of the 2015 paper (one field per Sybyl atom type, no
physico-chemical fields, α and λ optimised together by the Nelder–Mead simplex of the original R
scripts on kernels interpolated between the α grid points, external 5-fold CV repeated 100 times)
are reproduced with

```bash
python cmf_cli.py build ligands-all.mol2 --activity-file thrombin-activity.txt --fields , --cif --cif-min 5 \
       --raw-scale --cv-no-shuffle --alpha-search nelder-mead --alphas 0.01,0.05,0.1,0.25,0.5,0.75,1,2,3,5,10 \
       --nested --nested-on all --repeats 100 --json thr-cif.json
```

`--cif-types H` restricts the indicator fields to an explicit list of atom types, `--nm-start`
sets the starting point of the simplex search; with `--repeats` > 1 the statistics of the prediction
averaged over the reshuffles (what the papers report) are printed and written to the JSON file as
`q2_ex_aggr` / `rmse_ex_aggr`.

## Library use

```python
from cmfqsar import FieldSet, CMFModel, load_dataset
from cmfqsar.fields import AtomTable, build_fields, cif_types_in
from cmfqsar.model import grid_search

ds = load_dataset("aligned.sdf", activity_prop="pIC50")
fs = FieldSet.create(cif_types=cif_types_in(ds.mols, "sybyl", 5))
mf = [build_fields(m, fs, n) for m, n in zip(ds.mols, ds.names)]
train = AtomTable.from_molecules(mf)
model = CMFModel(fieldset=fs, method="krr")
grid_search(train, ds.activity, model, alphas=[0.3, 0.5, 1.0], lams=[1e-3, 1e-2, 0.1, 1])
model.fit(train, ds.activity, cv_folds=10, reuse_kernels=True)
print(model.stats.summary())
```

Decomposing one prediction, which works for the nonlinear kernels as well:

```python
from cmfqsar.shapley import atom_shapley, field_shapley, scaffold_decomposition

res = atom_shapley(model, mf[0])                     # one value per atom
res = field_shapley(model, mf[0], interactions=True)  # one value per field type
res = scaffold_decomposition(model, mf[0], ds.mols[0], ds.mols[1:20])   # per substituent
print(res.prediction, res.baseline, res.table(), res.top_interactions(5))
```

## Example data

`examples/benzamides_synthetic.sdf` — 109 aligned substituted N-phenylbenzamides with a *synthetic*
activity generated by `examples/make_example_dataset.py`: the activity is the overlap of the molecular
fields with a hidden pseudo-receptor co-field (steric, electrostatic, hydrophobic and H-bond acceptor
zones) plus noise (σ = 0.25). A CMF model therefore has to recover those zones; with the default settings
one obtains q² ≈ 0.87 and R²pred ≈ 0.84 (noise ceiling ≈ 0.94).

`examples/steroids_cbg.sdf` — the classic CoMFA benchmark: the 31 steroids of Cramer, Patterson & Bunce
(J. Am. Chem. Soc. 110 (1988) 5959) with their affinities to corticosteroid-binding globulin
(`CBG_pK` = log 1/K). The 3D structures are the corrected set distributed by the Gasteiger group
(Wagener, Sadowski & Gasteiger, J. Am. Chem. Soc. 117 (1995) 7769; TORVS web service, University of
Erlangen), re-superimposed here on the 17-atom steroid skeleton of testosterone (the Erlangen file has the
5-ene and estrogen members in other frames). The SD tag `subset` holds the customary split, compounds 1–21
training and 22–31 test (Coats, Perspect. Drug Discov. Des. 12 (1998) 199). With leave-one-out
cross-validation on the 21 training steroids the five physico-chemical fields give q² ≈ 0.79
(RMSEcv 0.53), CIFs alone q² ≈ 0.78; the ten test compounds are predicted with RMSE 0.7–0.8, the
well-known outlier being compound 31 (2α-methyl-9α-fluorocortisol). Note that R²pred computed with the
test-set variance is negative for this set because the ten test affinities span only 1.9 log units.

`examples/sutherland/*.sdf` — the eight literature sets of the 2013 paper (Sutherland, O'Brien & Weaver,
J. Med. Chem. 47 (2004) 5541, supplementary material): ACE (114), AChE (111), BZR (147), COX-2 (282),
DHFR (361), GPB (66), THER (76) and THR (88) compounds, aligned MMFF94s-minimised conformations exactly as
distributed, with the activity (`pIC50` or `pKi`), the published split (`subset`), and the original Sybyl
atom types and partial charges (scaled MNDO ESP-fit; Gasteiger-Marsili for THER) stored as per-atom
property lists (`atom.prop.sybyl_type`, `atom.dprop.partial_charge`). Choose *Partial charges: file* to use
the distributed charges; the Sybyl types are used automatically for continuous indicator fields. The
files were produced from the MOL2 distribution by `examples/make_sutherland_examples.py`; the comparison
with the published CMF/CoMFA/CoMSIA statistics is in `docs/benchmark.md`.

`examples/openff/*.sdf` — 21 congeneric series (513 ligands) from the Open Force Field
*protein-ligand-benchmark* collection (MIT licence, see
`examples/openff/LICENSE_openff_protein-ligand-benchmark.txt`), which curates the free-energy benchmark
series of Wang et al. (J. Am. Chem. Soc. 137 (2015) 2695: BACE, CDK2, JNK1, MCL1, p38, PTP1B, thrombin,
Tyk2), Schindler et al. (J. Chem. Inf. Model. 60 (2020) 5457: CDK8, c-Met, Eg5, HIF-2α, PFKFB3, SHP-2,
SYK, TNKS2), two Janssen BACE series, a Janssen PDE2 series, galectin and PDE10. Every ligand is posed in
the binding site of the target's crystal structure, so each series is an aligned congeneric set. The
activity tag is `pAffinity` (−log10 of Ki, Kd or IC50 in mol/L; `affinity_type` says which), with the
original value, unit, error and the DOI of the measurement kept as tags. No train/test split is defined;
use *Random split*. Produced by `examples/make_fep_examples.py` from the repository's `data` tree (the
current release for 15 targets, release 0.2.1 for BACE, JNK1, galectin and PDE10, whose files are stored
with git-lfs). For the Merck series the complete originals are in `examples/schindler` and should be
preferred; the copies here are the curated subsets (c-Met, for instance, has 5 of 24 ligands).

`examples/ross/<group>/*.sdf` — the Schrodinger public binding free energy benchmark (Ross et al.,
Commun. Chem. 6 (2023) 222; MIT licence, see `examples/ross/LICENSE_schrodinger_benchmark.txt`), the
largest public collection of congeneric series with prepared poses: **67 series, 1254 ligands** in 13
groups, kept in one subfolder per group — the FEP+ R-group set of Wang et al. (`jacs_set`), the Merck set
(`merck`, larger maps than the originals: PFKFB3 87 instead of 40 ligands, SYK and HIF-2α 51), Janssen
BACE series, GPCR ligands, macrocycles and Bayer macrocycles, scaffold- and core-hopping series,
fragments, buried-water cases, charge-change series, MCS-docking cases and OPLS stress tests. The activity
tag is `pAffinity`, converted from the experimental binding free energy (`exp_dG_kcal`) by
pAffinity = −ΔG / 2.303RT at 298.15 K; `protein`, `pdb`, `group`, `series` and `doi` identify the case, and
`affinity_scale` warns for the two OPLS series that store *relative* free energies (a constant offset,
harmless for regression but meaningless on an absolute scale). 25 series with fewer than five ligands
(scaffold-hopping pairs and similar) are not converted. `examples/ross/benchmark_metadata.csv` is the
upstream table with the source DOI and the dynamic range of every case. Produced by
`examples/make_ross_examples.py` from the repository's `fep_benchmark_inputs/structure_inputs` folder.
This collection overlaps `examples/openff` and `examples/schindler` for the JACS and Merck targets; the
versions here generally have more ligands per series.

`examples/pockets/<collection>/<target>.pdb` — the binding site of every free-energy set whose ligands
are in the frame of the structure they name: **73 pockets** of the 98 sets of `openff`, `schindler` and
`ross`, 2.9 MB in total. Each holds the residues within 8 Å of any ligand of its set, taken whole,
without waters and without the co-crystallised ligand, in the coordinates of the source entry, which is
recorded with the cutoff and the measured closest approach in its `REMARK 400` lines. Opening a data set
loads its pocket automatically, so the protein appears around the molecules, their fields and the
co-fields of a model with nothing to align. Structures come from RCSB and are public domain (CC0).
Produced by `examples/make_pocket_examples.py`, which skips any set whose ligands do not sit in the
frame of the entry it names, which is why 25 of the sets have no pocket here.

`examples/schindler/*.sdf` — the complete Merck KGaA FEP benchmark of Schindler et al. (J. Chem. Inf.
Model. 60 (2020) 5457; MIT licence, see `examples/schindler/LICENSE_MCompChem_fep-benchmark.txt`): the
eight series CDK8 (33), c-Met (24), Eg5 (28), HIF-2α (42), PFKFB3 (40), SHP-2 (26), SYK (44) and TNKS2
(27), 264 ligands, plus the Eg5 alternative-loop variant and the HIF-2α cycloalkyl[c]thiophene series of
2023 (38). Unlike the Open Force Field subsets in `examples/openff`, these are the full series as released,
with the LigPrep poses in the binding site, the measured IC50 (ITC Kd for the 2023 HIF-2α set) converted to
`pAffinity`, and the experimental ΔG and the FEP+ prediction (20 ns) kept as tags where the repository
provides them. Produced by `examples/make_schindler_examples.py`.

`examples/mittal/*.sdf` — 38 of the 40 benchmark sets of Mittal, McKinnon & Sorich (J. Chem. Inf. Model.
49 (2009) 1810), compiled for comparing QSAR methods in lead optimisation (ryanodine receptor, vacuolar
ATPase, topoisomerase II, angiotensin II receptor, PDE-IV, cannabinoid receptor, GSK-3β, HIV protease and
reverse transcriptase, estrogen receptor, and many more; 16 to 397 molecules per set). Two sets are
omitted because their SMILES in the Supporting Information are corrupted (MT: cerium placeholder atoms in
all 56 structures; PLA2: dummy atoms), and a handful of unparsable SMILES in other sets are skipped.
The Supporting Information gives only SMILES and activities, so these files were built by
`examples/make_mittal_examples.py`: RDKit ETKDGv3/MMFF94 conformers superimposed on a reference molecule
of each set by maximum common substructure. The tags `align_mcs_atoms` and `align_rmsd` record the size of
the common substructure and the RMSD of the fit for every molecule, so that outliers of the automatic
alignment can be excluded; `description` summarises the biological activity (Table 1 of the paper).
These conformations and alignments are automatic, not those of the original 3D-QSAR studies, so the sets
serve for method comparison on equal footing rather than for reproducing literature statistics. Nine of
them (ACE, ACHE, BZR, COX2, DHFR, GPB, THERM, THR, STEROIDS) duplicate sets for which original aligned
coordinates are available in `examples/sutherland` and `examples/steroids_cbg.sdf`.

A per-molecule SD tag named `subset`, `set` or `split` (values train/test or 1/2) is honoured by the loader
in both the GUI and the CLI. Other literature data sets
(Sutherland's ACE/AChE/BZR/COX-2/DHFR sets, the thrombin set, …) can be loaded as SDF or MOL2 files;
MOL2 Sybyl atom types and partial charges are used when present (choose *Partial charges: file*).
MOL2 blocks that RDKit's reader rejects (unusual valences) are parsed by a tolerant fallback reader.
`docs/benchmark.md` compares the program with Table 2 of the 2013 paper on the eight literature sets
(`benchmarks/run_sutherland.py` reruns the comparison when the data sets are available locally).

## Field parameterisation

Two parameterisations of the physico-chemical fields are available (*Increments* in the model
panel, `--param` on the command line):

| Field | `rdkit` (default) | `tripos` (original CMF scripts) |
|---|---|---|
| electrostatic | partial charge: Gasteiger (default), MMFF94, or charges read from the file | same |
| steric | Tripos vdW well depth `E_vdw`, width `α/R_vdw²`, parameters by element | parameters by Sybyl atom type (Tripos 5.2 table) |
| hydrophobic | Wildman–Crippen atomic logP contribution | in-house one-atom fragment increments keyed by Sybyl type and number of attached H (regression on logP) |
| H-bond acceptor / donor | approximate Abraham-type increments; donor weight on the polar H | in-house increments (regression on Abraham's B and A constants), donor weight on the heavy atom |
| CIF | 1 for atoms of the given type | same |

`tripos` reproduces the parameterisation used for the models of the 2013 paper (`cmfqsar/increments.py`
is a transcription of the original `cmf-params.R`).  `tripos-legacy` additionally reproduces a bug of
those scripts, which only counted an attached hydrogen when it was the first atom of the bond record
(practically never in Sybyl MOL2 files), so that all atoms received the "no hydrogen" increments; it
exists only to reproduce the published numbers exactly (see `docs/benchmark.md`).
Atom-typing scheme for CIFs can be Sybyl, element, or element+aromatic; Sybyl types from MOL2 files
(`_TriposAtomType`) are used when present.

## Tests

```bash
python -m pytest tests
```

The tests verify the analytic kernels against numerical integration, the symmetry/positive
semi-definiteness of kernel matrices, the KRR bias system (with and without bias), the decomposition
rules, that the overlap integral of co-field and molecular field on a grid reproduces the kernel
prediction (primal = dual form), the Tripos increments, the tolerant MOL2 reader, activity tables,
the Brent λ search and that the precomputed-kernel nested CV equals the direct one. For the conformational ensembles they check that the single mode is bit-for-bit what it was, that the ensemble kernel equals the weighted mean of the kernels of the conformation pairs, that one conformation in ensemble mode reproduces the single mode, that the weights behave and fall back to uniform when energies are missing, that contributions still add up to the prediction and fold back onto the atoms, and that a zero seed does not collapse the ensemble. For the multi-instance weighting they check the weighting rules and their limits, that selection recovers a hidden witness where averaging cannot, that beta = 0 is exactly the ensemble kernel, that hard selection is one-hot and settles, that a new molecule is predicted without using its activity, that the folds split molecules and that shuffled activities destroy the fit, and that annealing sharpens the weights. For the pocket docking they check that the score prefers complementary charges and punishes clashes, that burial is what holds a pose in a cavity and the tether what keeps it from wandering, that docking removes a clash, that the internal geometry is preserved to the last decimal because the search is rigid, and that a molecule which cannot be placed is handed back untouched. They also check that weights given on a molecule are read and normalised, that nonsense falls back to uniform, and that the tether can be widened for a molecule that does not start in the site. For the kernel
layers they check positive semi-definiteness, that the linear setting leaves the kernel untouched, that
the self-kernel used for new molecules agrees with the training diagonal, that co-fields, atom
contributions and posterior variances are refused while the training-molecule decomposition still adds
up to the prediction, and that the alignment weights concentrate on the informative field and are
re-derived inside every cross-validation fold. For the protein display they check the PDB and mmCIF
readers on a synthetic structure with alternate locations, waters, an ion, a ligand and a second
model, the distance-based bonds, the selection of whole residues around one probe or many, the
removal of the co-crystallised ligand, the label anchors on the alpha carbons, and the contact
distance that reports a mismatched frame. For complementarity they check the overlap integral against
numerical integration on a grid, that a cation facing an anion scores positive and facing a cation
negative, that the normalised term does not change when a field is simply scaled up, that clashes are
counted apart from the score, that the surface sits on the van der Waals radius and carries the right
sign on the facing side, that the side chains are ionised and the donors get their hydrogens, and that
a model trained so that positive charge raises activity agrees with an anionic site and disagrees with
a cationic one. For the Shapley decomposition they check that it
reproduces the published atom contributions, the per-field contributions and the per-atom per-field
contributions exactly when the kernel is linear, that all interaction indices then vanish, that the
values stay efficient for the nonlinear kernels and under a fixed scaffold, that sampling agrees with
exhaustive enumeration, that pooling atoms into one player sums their contributions, and that a Shapley
field integrates over space to the value it was built from. For the heterogeneous series they check that the
framework strips the side chains and that the generic one merges benzene with pyridine, that small
groups are pooled, that no fold ever splits a chemotype and that bundling keeps whole chemotypes
together, that an activity which is a property of the scaffold is predicted almost perfectly by random
folds and not at all when the chemotype is held out, that the grouped predictions are exactly a fit
without the held-out group, that a cluster is moved as one rigid body and the reference cluster not at
For the one-class descriptions they check that nu really is the fraction of the training molecules
left outside for every one of them, that molecules far from the training cloud land outside and are
ranked below it, that the score falls monotonically along a ray leaving the cloud, that the subspace
description is stricter than the centroid one about molecules beside a flat set, that the two linear
ones reproduce their own kernel expansion while the other two refuse to pretend they have one, and
that the enrichment factor counts what it says. For the Gaussian-process classifier they check that the Newton iteration reaches a point where f = K grad
and that nothing near it scores higher on the objective, that the marginal likelihood equals its textbook
definition computed the slow way, that the latent mean is the same kernel expansion the rest of the
program assumes, that the probabilities are bounded and pulled towards one half by the variance, that the
variance returns to the prior far from the training molecules, that the evidence prefers an informative
kernel over a noise one and agrees there with cross-validation, and that one class alone is refused.
all, that the alignment-free kernel keeps the atom counts rather than cancelling them and gives 1 for a
molecule whose geometry has been destroyed but whose atoms have not, that the attenuated kernel
converges on the limit the default writes down directly, and that the limit does not move at all when
every molecule is rotated and translated.

## Project layout

```
cmf_app.py              GUI launcher
cmf_cli.py              command-line tool
cmfqsar/atomtypes.py    Sybyl atom typing, vdW parameters
cmfqsar/increments.py   hydrophobic / H-bond increments of the original CMF scripts
cmfqsar/fields.py       field parameterisation, FieldSet / MoleculeFields / AtomTable
cmfqsar/kernels.py      analytic field kernels (vectorised)
cmfqsar/model.py        KRR/SVR/GP, cross-validation, grid search, nested CV, statistics
cmfqsar/cofields.py     co-fields and molecular fields on grids, isosurfaces, cube export
cmfqsar/dataio.py       SDF/MOL2/SMILES reading, model persistence, exports
cmfqsar/align.py        3D embedding, MCS alignment, field-similarity refinement, unalign
cmfqsar/varalign.py     alignment by prediction confidence (GP posterior variance)
cmfqsar/decoys.py       posing a background set for virtual-screening evaluation
cmfqsar/gtm.py          chemical space maps and activity landscapes (kernel PCA + GTM)
cmfqsar/shapley.py      Shapley decomposition of a prediction (atoms, substituents, field types)
cmfqsar/conformers.py   conformational ensembles: generation, weighting, grouping (4D)
cmfqsar/mil.py          multi-instance learning: the model chooses the conformation
cmfqsar/docking.py      rigid-body placement of conformations in the pocket
cmfqsar/chemotypes.py   heterogeneous series: scaffold clusters, placement, grouped folds
cmfqsar/gpc.py          Gaussian-process classification (Laplace), probabilities and evidence
cmfqsar/oneclass.py     one-class descriptions: centroid, k-NN, Gaussian process, kernel PCA
cmfqsar/protein.py      PDB/mmCIF reading and writing, binding-site selection, frame check
cmfqsar/complementarity.py  pocket fields, complementarity scores, co-field agreement
cmfqsar/gui/            PySide6 application (main window, panels, OpenGL viewer, workers)
examples/               synthetic benchmark and its generator
benchmarks/             literature benchmark runner (Sutherland data sets) and its results
docs/benchmark.md       comparison with Table 2 of the 2013 paper
tests/                  unit tests
```
