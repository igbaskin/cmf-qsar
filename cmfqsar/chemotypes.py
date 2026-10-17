"""Heterogeneous series: several chemotypes, no single common substructure.

Everything in this program rests on the molecules sharing a frame, and for a congeneric series
superposition on the common substructure supplies one: the alignment experiment rebuilt a published
alignment from scrambled poses to within 0.06 Å for the median molecule.  A series of several
scaffolds has no such substructure to superimpose on, and the usual recipe quietly becomes
meaningless: one reference molecule, one maximum common substructure, one co-field, one random
cross-validation that puts a molecule's own scaffold-mates in the training part.

This module is the workflow for that case, in the order the problem imposes:

1. **Cluster by scaffold.**  :func:`chemotype_labels` groups the molecules by their Bemis-Murcko
   framework, either as drawn or made generic so that ring atoms and bond orders stop mattering.
2. **Align within each cluster**, where a common substructure does exist and does the job well:
   :func:`align_within_groups`.
3. **Place the clusters relative to one another**, which is the part a congeneric series never
   needs: :func:`place_groups` moves each cluster rigidly, as a whole, either by maximising the
   field similarity of its reference with a chosen cluster or by docking that reference into the
   binding site.  Moving the cluster rigidly is what keeps step 2 intact.
4. **Judge the result by leave-one-chemotype-out**, :func:`grouped_folds` and
   :func:`grouped_cv_predict`.  Random folds flatter a multi-scaffold set, because the molecules
   that look most like the one being predicted are its own scaffold-mates and they sit in the
   training part; holding out a whole chemotype asks the question that matters.
5. **Compare against a baseline that needs no alignment at all**, :func:`alignment_free_q2` over
   :func:`count_kernel`: with a small attenuation factor a continuous indicator field tends to a
   count of atoms of its type, so the kernel becomes a fragment-descriptor kernel and the poses
   stop mattering.  By default the baseline is that limit, written down from the counts
   themselves, so no attenuation has to be chosen and the poses are gone exactly rather than
   nearly.  Two details decide whether an attenuated kernel reaches the same place, and both are
   measured in the tests: the attenuation has to be far smaller than a modelling one, since the
   pose dependence dies only linearly in it (at alpha = 0.02 a rigid motion of every molecule
   still moves the kernel by 0.7, at 2e-4 by 0.013), and the cosine normalisation has to be
   applied to the joint kernel rather than field by field, or the counts cancel.  If the aligned
   model cannot beat that baseline, the alignment is not paying for itself, and on a heterogeneous
   set that is the first thing worth knowing.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .align import field_align, mcs_align, set_coords
from .fields import AtomTable, FieldSet, build_fields, cif_types_in
from .model import CMFModel, _Learner, r2_score, rmse

ACYCLIC = "(acyclic)"


# --------------------------------------------------------------------------- #
# 1. Clustering by scaffold
# --------------------------------------------------------------------------- #

def scaffold_smiles(mol: Chem.Mol, generic: bool = False) -> str:
    """Bemis-Murcko framework of a molecule as canonical SMILES; "" has no rings."""
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:                            # noqa: BLE001 - odd valences, keep the molecule
        return ACYCLIC
    if core is None or core.GetNumAtoms() == 0:
        return ACYCLIC
    if generic:
        try:
            core = MurckoScaffold.MakeScaffoldGeneric(core)
        except Exception:                        # noqa: BLE001
            pass
    try:
        return Chem.MolToSmiles(core)
    except Exception:                            # noqa: BLE001
        return ACYCLIC


def chemotype_labels(mols: Sequence[Chem.Mol], generic: bool = False,
                     min_size: int = 1) -> Tuple[np.ndarray, List[str]]:
    """Group index per molecule and the scaffold of every group, largest group first.

    Groups smaller than `min_size` are merged into one "other" group, so that a handful of
    singletons does not turn leave-one-chemotype-out into leave-one-molecule-out.
    """
    keys = [scaffold_smiles(m, generic=generic) if m is not None else ACYCLIC for m in mols]
    counts: Dict[str, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    big = [k for k, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if c >= max(min_size, 1)]
    order = {k: i for i, k in enumerate(big)}
    other = len(big)
    labels = np.array([order.get(k, other) for k in keys], dtype=int)
    names = list(big)
    if (labels == other).any():
        names.append("other (small groups)")
    return labels, names


def group_members(labels: np.ndarray) -> List[np.ndarray]:
    return [np.flatnonzero(labels == g) for g in range(int(labels.max()) + 1)] if len(labels) else []


def summarise_groups(labels: np.ndarray, names: Sequence[str]) -> str:
    members = group_members(labels)
    parts = [f"{len(m)}x {names[g] if g < len(names) else g}" for g, m in enumerate(members) if len(m)]
    return f"{len(members)} chemotypes: " + ", ".join(parts)


# --------------------------------------------------------------------------- #
# 2 and 3. Aligning inside a cluster, then placing the clusters
# --------------------------------------------------------------------------- #

def group_reference(mols: Sequence[Chem.Mol], members: Sequence[int]) -> int:
    """The molecule of a group that others are superimposed on: the largest one."""
    return int(max(members, key=lambda i: mols[i].GetNumHeavyAtoms()))


def align_within_groups(mols: Sequence[Chem.Mol], labels: np.ndarray, progress=None,
                        cancel=None) -> List[str]:
    """Superimpose the molecules of every group on that group's reference. In place."""
    log: List[str] = []
    members = group_members(labels)
    total = max(len(mols), 1)
    done = 0
    for g, idx in enumerate(members):
        if len(idx) == 0:
            continue
        ref = group_reference(mols, idx)
        for i in idx:
            done += 1
            if cancel is not None and cancel():
                return log
            if progress is not None:
                progress(f"Aligning chemotype {g + 1}/{len(members)}", done / total)
            if i == ref:
                log.append(f"group {g}: {i} is the reference")
                continue
            try:
                rmsd, n = mcs_align(mols[i], mols[ref])
                log.append(f"group {g}: molecule {i} on {ref}, {n} atoms, RMSD {rmsd:.2f}")
            except Exception as exc:             # noqa: BLE001 - reported, pose left alone
                log.append(f"group {g}: molecule {i} FAILED: {exc}")
    return log


def _kabsch(before: np.ndarray, after: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The rigid transform that takes `before` onto `after` (both (N, 3), same order)."""
    cb, ca = before.mean(axis=0), after.mean(axis=0)
    H = (before - cb).T @ (after - ca)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, ca - R @ cb


def apply_transform(mol: Chem.Mol, R: np.ndarray, t: np.ndarray) -> None:
    coords = np.array(mol.GetConformer().GetPositions())
    set_coords(mol, coords @ R.T + t)


def place_groups(mols: Sequence[Chem.Mol], labels: np.ndarray, fieldset: FieldSet,
                 mode: str = "field", reference_group: int = 0, pocket=None,
                 restarts: int = 2, seed: int = 0, tether_radius: float = 8.0,
                 progress=None, cancel=None) -> List[str]:
    """Move every cluster, as a rigid body, into a common frame.

    ``field`` superimposes each cluster's reference on the reference of `reference_group` by
    maximising the CMF kernel between them, which needs no common substructure.  ``dock`` places
    each cluster's reference in the binding site instead, which needs a pocket but no reference
    cluster at all.  Either way the transform found for the reference is applied to every molecule
    of its cluster, so the within-cluster alignment of step 2 survives untouched.
    """
    log: List[str] = []
    members = group_members(labels)
    if mode == "dock" and pocket is None:
        raise ValueError("docking needs the fields of a pocket")
    anchor = None
    if mode == "field":
        if reference_group >= len(members) or len(members[reference_group]) == 0:
            raise ValueError("the reference chemotype is empty")
        anchor = group_reference(mols, members[reference_group])
    for g, idx in enumerate(members):
        if len(idx) == 0 or (cancel is not None and cancel()):
            continue
        if progress is not None:
            progress(f"Placing chemotype {g + 1}/{len(members)}", g / max(len(members), 1))
        ref = group_reference(mols, idx)
        if mode == "field" and g == reference_group:
            log.append(f"group {g}: the reference chemotype, left where it is")
            continue
        before = np.array(mols[ref].GetConformer().GetPositions())
        try:
            if mode == "field":
                # the kernel is flat where the two molecules do not overlap at all, and Powell
                # has nothing to climb; put their centroids together first
                here = np.array(mols[ref].GetConformer().GetPositions())
                there = np.array(mols[anchor].GetConformer().GetPositions())
                set_coords(mols[ref], here + (there.mean(axis=0) - here.mean(axis=0)))
                s0, s1 = field_align(mols[ref], mols[anchor], fieldset, restarts=restarts,
                                     seed=seed)
                log.append(f"group {g}: reference {ref} placed by field similarity {s0:.3f} -> {s1:.3f}")
            else:
                from .docking import dock_coords
                single = fieldset.copy()
                single.conformers = "single"
                mf = build_fields(mols[ref], single, "")
                coords, res = dock_coords(mf, pocket, single, restarts=restarts, seed=seed,
                                          tether_radius=tether_radius)
                set_coords(mols[ref], coords)
                log.append(f"group {g}: reference {ref} docked, score {res.score_before:+.2f} -> "
                           f"{res.score_after:+.2f}, clash {res.clash_before:.1f} -> "
                           f"{res.clash_after:.1f}, moved {res.moved:.1f} A")
        except Exception as exc:                 # noqa: BLE001 - the cluster stays where it was
            log.append(f"group {g}: placement FAILED: {exc}")
            continue
        after = np.array(mols[ref].GetConformer().GetPositions())
        R, t = _kabsch(before, after)
        for i in idx:
            if i != ref:
                apply_transform(mols[i], R, t)
    return log


# --------------------------------------------------------------------------- #
# 4. Leave one chemotype out
# --------------------------------------------------------------------------- #

def grouped_folds(labels: np.ndarray, max_folds: int = 0) -> List[np.ndarray]:
    """Folds that never split a chemotype: one fold per group, or groups bundled into `max_folds`.

    This is the honest cross-validation for a heterogeneous series.  With random folds a molecule
    is predicted from its own scaffold-mates, which is a question nobody is asking.
    """
    members = [m for m in group_members(labels) if len(m)]
    if not members:
        return []
    if max_folds and 0 < max_folds < len(members):
        members = sorted(members, key=len, reverse=True)
        bundles: List[List[np.ndarray]] = [[] for _ in range(max_folds)]
        sizes = [0] * max_folds
        for m in members:                        # greedy: the biggest group to the lightest fold
            k = int(np.argmin(sizes))
            bundles[k].append(m)
            sizes[k] += len(m)
        return [np.concatenate(b) for b in bundles if b]
    return members


def grouped_cv_predict(K: np.ndarray, y: np.ndarray, labels: np.ndarray, method: str = "krr",
                       params: Optional[Dict[str, float]] = None, fit_bias: bool = True,
                       max_folds: int = 0) -> np.ndarray:
    """Cross-validated predictions with a whole chemotype held out at a time."""
    y = np.asarray(y, float)
    params = dict(params or {"lam": 1.0})
    out = np.full(len(y), np.nan)
    for te in grouped_folds(labels, max_folds=max_folds):
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(tr) < 2:
            continue
        learner = _Learner(method, params, fit_bias).fit(K[np.ix_(tr, tr)], y[tr])
        out[te] = learner.predict(K[np.ix_(te, tr)])
    return out


@dataclass
class HeteroResult:
    """What the workflow found."""
    labels: np.ndarray
    names: List[str]
    q2_random: float = float("nan")
    q2_grouped: float = float("nan")
    q2_baseline: float = float("nan")
    q2_invariant: float = float("nan")      # geometry kept, frame not assumed
    rmse_grouped: float = float("nan")
    alpha: float = float("nan")
    lam: float = float("nan")
    per_group: List[dict] = _field(default_factory=list)
    log: List[str] = _field(default_factory=list)

    def verdict(self) -> str:
        if not np.isfinite(self.q2_grouped) or not np.isfinite(self.q2_baseline):
            return "not enough chemotypes to judge"
        if self.q2_grouped <= 0:
            return ("the aligned model does not predict a held-out chemotype at all; "
                    "the alignment is not carrying across scaffolds")
        if self.q2_grouped < max(self.q2_baseline,
                                 self.q2_invariant if np.isfinite(self.q2_invariant) else -9e9):
            which = ("fragment-count" if self.q2_baseline >= self.q2_invariant
                     or not np.isfinite(self.q2_invariant) else "geometry-aware alignment-free")
            return (f"the {which} model is better: the alignment is not paying for itself on "
                    f"this series")
        return "the aligned model beats the alignment-free baseline across chemotypes"


# --------------------------------------------------------------------------- #
# 5. The baseline that needs no alignment
# --------------------------------------------------------------------------- #

#: attenuation to use when the alignment-free baseline is taken by the Gaussian route rather
#: than at its limit.  Small enough that the poses are gone: a rigid motion of every molecule
#: changes the kernel by under 0.02, and the kernel is within 0.01 of the count-vector cosine.
#: The default route is the limit itself, which needs no attenuation; see :func:`count_kernel`.
BASELINE_ALPHA = 2e-4


def count_kernel(mols: Sequence[Chem.Mol], names: Sequence[str], alpha: Optional[float] = None,
                 cif_min: int = 3, scheme: str = "sybyl") -> np.ndarray:
    """The kernel of the alignment-free baseline: indicator fields at vanishing attenuation.

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
    """
    types = cif_types_in(mols, scheme, cif_min) if len(mols) else []
    if not types:
        return np.zeros((0, 0))
    fs = FieldSet.create(physchem=[], cif_types=types, alpha=alpha or 1.0, cif_scheme=scheme)
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    if alpha is None:
        N = np.array([m.weights.sum(axis=0) for m in mf])            # counts per type
        w = fs.mixing_vector() / np.maximum((N ** 2).mean(axis=0), 1e-300)
        K = (N * w) @ N.T
    else:
        model = CMFModel(fieldset=fs, method="krr", normalize=False)  # keep the counts
        K = model.joint_kernel(model.prepare_training_kernels(AtomTable.from_molecules(mf)))
    d = np.sqrt(np.clip(np.diag(K), 1e-300, None))
    return K / np.outer(d, d)


def invariant_q2(molfields: Sequence, y: np.ndarray, labels: np.ndarray, lam=1.0,
                 max_folds: int = 0, d_max: float = 12.0, step: float = 0.75,
                 sigma: float = 0.75) -> Tuple[float, float]:
    """Leave-one-chemotype-out q2 of a model that keeps the geometry but needs no alignment.

    The kernel is the field autocorrelation of :mod:`cmfqsar.invariant`: invariant to rotation
    and translation, so no superposition is involved, but still a function of the conformation.
    It sits between the aligned model and the count kernel, and says what the geometry is worth
    once the frame is no longer assumed.
    """
    from .invariant import AutocorrelationSpec, autocorrelation_kernel, feature_matrix
    spec = AutocorrelationSpec(d_max=d_max, step=step, sigma=sigma)
    X = feature_matrix(molfields, spec)
    if X.size == 0:
        return float("nan"), float("nan")
    K = autocorrelation_kernel(X, n_bins=spec.n_bins)
    yy = np.asarray(y, float)
    best = (float("nan"), float("nan"))
    for one in np.atleast_1d(lam).ravel():
        pred = grouped_cv_predict(K, yy, np.asarray(labels), params={"lam": float(one)},
                                  max_folds=max_folds)
        good = np.isfinite(pred)
        if good.sum() < 3:
            continue
        q2 = r2_score(yy[good], pred[good])
        if not np.isfinite(best[0]) or q2 > best[0]:
            best = (q2, rmse(yy[good], pred[good]))
    return best


def alignment_free_q2(mols: Sequence[Chem.Mol], names: Sequence[str], y: np.ndarray,
                      labels: np.ndarray, alpha: Optional[float] = None, lam=1.0,
                      cif_min: int = 3, max_folds: int = 0,
                      scheme: str = "sybyl") -> Tuple[float, float]:
    """Leave-one-chemotype-out q2 of a model that ignores the poses entirely.

    `alpha` is the attenuation of the indicator fields, or `None` for the limit they tend to,
    which is the default and what makes the baseline alignment-free.  `lam` may be one value or a
    grid, which is then searched on the same folds the aligned model is tuned on, so that the two
    representations are compared under one protocol.  Returns (q2, rmse).
    """
    ok = [i for i in range(len(mols)) if mols[i] is not None]
    K = count_kernel([mols[i] for i in ok], [names[i] for i in ok], alpha=alpha, cif_min=cif_min,
                     scheme=scheme)
    if K.size == 0:
        return float("nan"), float("nan")
    yy = np.asarray(y, float)[ok]
    lab = np.asarray(labels)[ok]
    best = (float("nan"), float("nan"))
    for one in (np.atleast_1d(lam).ravel()):
        pred = grouped_cv_predict(K, yy, lab, params={"lam": float(one)}, max_folds=max_folds)
        good = np.isfinite(pred)
        if good.sum() < 3:
            continue
        q2 = r2_score(yy[good], pred[good])
        if not np.isfinite(best[0]) or q2 > best[0]:
            best = (q2, rmse(yy[good], pred[good]))
    return best


# --------------------------------------------------------------------------- #
# The workflow as one call
# --------------------------------------------------------------------------- #

def evaluate(mols: Sequence[Chem.Mol], names: Sequence[str], y: np.ndarray, labels: np.ndarray,
             group_names: Sequence[str], fieldset: FieldSet, alphas: Sequence[float] = (0.3, 0.5, 1.0),
             lams: Sequence[float] = (1e-2, 0.1, 1.0), max_folds: int = 0, normalize: bool = True,
             baseline_alpha: Optional[float] = None, progress=None, cancel=None) -> HeteroResult:
    """Build the aligned model, judge it chemotype by chemotype, and compare with the baseline.

    The hyper-parameters are chosen on the same grouped folds the model is reported on, which is
    optimistic in the usual way of any selection on the reported statistic; what matters here is
    the comparison between representations, which share the protocol.
    """
    y = np.asarray(y, float)
    ok = [i for i in range(len(mols)) if mols[i] is not None and np.isfinite(y[i])]
    yy, lab = y[ok], np.asarray(labels)[ok]
    res = HeteroResult(labels=np.asarray(labels), names=list(group_names))
    if len(set(lab.tolist())) < 2 or len(ok) < 6:
        res.log.append("fewer than two chemotypes with activities: nothing to hold out")
        return res
    best = None
    for k, alpha in enumerate(alphas):
        if cancel is not None and cancel():
            return res
        if progress is not None:
            progress(f"Kernels for alpha = {alpha:g}", 0.1 + 0.7 * k / max(len(alphas), 1))
        fs = fieldset.copy()
        fs.set_alpha(float(alpha))
        mf = [build_fields(mols[i], fs, names[i]) for i in ok]
        model = CMFModel(fieldset=fs, method="krr", normalize=normalize)
        K = model.joint_kernel(model.prepare_training_kernels(AtomTable.from_molecules(mf)))
        for lam in lams:
            pred = grouped_cv_predict(K, yy, lab, params={"lam": float(lam)}, max_folds=max_folds)
            good = np.isfinite(pred)
            if good.sum() < 3:
                continue
            q2 = r2_score(yy[good], pred[good])
            if best is None or q2 > best[0]:
                best = (q2, rmse(yy[good], pred[good]), float(alpha), float(lam), K, pred)
    if best is None:
        res.log.append("no fold could be evaluated")
        return res
    res.q2_grouped, res.rmse_grouped, res.alpha, res.lam, K, pred = best
    from .model import cv_predict as random_cv
    res.q2_random = r2_score(yy, random_cv(K, yy, "krr", {"lam": res.lam}, folds=10, seed=0))
    for g in sorted(set(lab.tolist())):
        m = lab == g
        good = m & np.isfinite(pred)
        if good.sum() >= 1:
            res.per_group.append({
                "group": int(g),
                "name": group_names[g] if g < len(group_names) else str(g),
                "n": int(m.sum()),
                "rmse": float(rmse(yy[good], pred[good])),
                "bias": float(np.mean(pred[good] - yy[good]))})
    if progress is not None:
        progress("Alignment-free baseline", 0.9)
    res.q2_baseline, _ = alignment_free_q2([mols[i] for i in ok], [names[i] for i in ok], yy, lab,
                                           alpha=baseline_alpha, lam=list(lams),
                                           max_folds=max_folds)
    if progress is not None:
        progress("Alignment-free but geometry-aware", 0.95)
    fs = fieldset.copy()
    fs.set_alpha(float(res.alpha) if np.isfinite(res.alpha) else 1.0)
    res.q2_invariant, _ = invariant_q2([build_fields(mols[i], fs, names[i]) for i in ok], yy, lab,
                                       lam=list(lams), max_folds=max_folds)
    return res
