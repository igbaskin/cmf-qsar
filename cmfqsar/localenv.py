"""Atom-centred environments and the invariant local kernel (plan item LP2).

Everything else in this program takes a molecule as the sample.  A carbon shift, an acidity or a
site reactivity belongs to an *atom in a molecule*, so the sample has to be the atom and its
surroundings.  :mod:`cmfqsar.atomtargets` (plan item LP1) holds the targets of that shape; this
module builds the samples that go with them.

**The environment.**  One molecule gives as many samples as it has measured atoms.  The sample for
atom *a* is the molecule's own fields multiplied by a Gaussian envelope about *a*,

    g(r) = f(r) * exp(-rho |r - r_a|^2),

so an atom close to the centre keeps its weight and one far away loses it.  A CMF field is a sum
of atom-centred Gaussians, and this envelope keeps that form: :func:`environment` returns a
:class:`~cmfqsar.fields.MoleculeFields` whose atoms are the atoms of the molecule, whose
coordinates are measured from the centre, and whose weights are damped by the envelope.  Every
tool that works on a molecule's fields therefore works on an environment -- the kernels, the
regression, the variance, the atom decomposition -- because an environment *is* a field.

`rho` is a hyper-parameter like `alpha`, and it says how far the model is allowed to look: the
envelope falls to 1/e at ``1/sqrt(rho)``, which :func:`envelope_width` reports.  A shielding is a
short-range property and wants a narrow envelope; an acidity is felt further away and wants a
wider one.

**The kernel.**  The first kernel to measure is the rotation-invariant field autocorrelation of
:mod:`cmfqsar.invariant`, taken on the environment instead of on the whole molecule, because it
needs no alignment: two environments of different molecules need not be superposed for their
autocorrelations to be compared.  The autocorrelation is a double sum over atom pairs, so on an
environment it becomes

    sum over atoms a, b of  w_a w_b exp(-rho d_a0^2) exp(-rho d_b0^2) * (distance bin of d_ab),

with ``d_a0`` the distance of an atom from the centre.  It is a local descriptor built from the
same fields as the global one, and the prediction still splits over the atoms exactly
(:meth:`cmfqsar.model.CMFModel.atom_contributions`, plan item SC6).

**The reference.**  Under rule 1 of the plan every item is measured against the simplest thing
that could work, which here is an atom-centred count: the type of the centre itself and the counts
of atom types in distance shells around it, which is what a per-atom increment scheme is.
:func:`count_features` builds it from the same molecules, the same atom types and the same
samples, and :func:`cross_validate` runs both through one cross-validation, so the only difference
between the two numbers is the descriptor.

**The folds.**  :meth:`cmfqsar.atomtargets.AtomTargets.folds` holds out whole molecules, and this
module uses nothing else.  Two carbons of one molecule, one in the training set and one in the
test set, would let the model see the molecule it is being tested on.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .atomtargets import AtomTargets
from .fields import MoleculeFields
from .invariant import AutocorrelationSpec, autocorrelation_kernel, feature_matrix
from .model import krr_fit, r2_score, rmse

#: default envelope width parameter, in A^-2: the envelope falls to 1/e at 1.4 A.  This is a
#: starting point and no more -- rho decides how far the model looks, so it has to be tuned for
#: the property, as `benchmarks/run_local.py` does.
DEFAULT_RHO = 0.5
#: an atom whose envelope factor is below this is left out of the environment
ENVELOPE_FLOOR = 1e-4
#: the distance grid for the autocorrelation of an environment: shorter than the molecular one,
#: because the envelope has already removed everything beyond a few angstrom
LOCAL_SPEC = AutocorrelationSpec(d_max=8.0, step=0.5, sigma=0.5)
#: shell edges of the count reference, in A: a bonded neighbour, a 1-3 neighbour, and three
#: wider shells
SHELL_EDGES = (1.8, 2.8, 3.8, 5.0, 6.5, 8.5)
#: ridge values :func:`choose_lam` searches
LAMS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def envelope_width(rho: float = DEFAULT_RHO) -> float:
    """The distance at which the envelope falls to 1/e, in angstrom."""
    r = float(rho)
    if r <= 0:
        raise ValueError("the envelope width parameter rho has to be positive")
    return 1.0 / np.sqrt(r)


def envelope_radius(rho: float = DEFAULT_RHO, floor: float = ENVELOPE_FLOOR) -> float:
    """The distance beyond which an atom is left out of the environment, in angstrom."""
    r = float(rho)
    f = min(max(float(floor), 1e-300), 1.0)
    if r <= 0:
        raise ValueError("the envelope width parameter rho has to be positive")
    if f <= 0 or f >= 1:
        return float("inf")
    return float(np.sqrt(-np.log(f) / r))


def envelope(coords: np.ndarray, centre: Sequence[float], rho: float = DEFAULT_RHO) -> np.ndarray:
    """The envelope factor exp(-rho d^2) of every atom, with d measured from `centre`."""
    c = np.asarray(centre, float).reshape(1, 3)
    d2 = np.sum((np.asarray(coords, float) - c) ** 2, axis=1)
    return np.exp(-float(rho) * d2)


def environment(mol: MoleculeFields, atom: int, rho: float = DEFAULT_RHO,
                floor: float = ENVELOPE_FLOOR, name: str = "") -> MoleculeFields:
    """The fields of `mol` seen from `atom`: the same fields under a Gaussian envelope.

    The result is an ordinary :class:`~cmfqsar.fields.MoleculeFields`, with the centre at the
    origin and every weight multiplied by ``exp(-rho d^2)``, so the whole program applies to it.
    Atoms whose factor is below `floor` are left out, which costs nothing measurable and saves the
    cost of carrying them: at the default floor and the default `rho` that is everything beyond
    4.3 A, which :func:`envelope_radius` reports.  :func:`kept_share` says what share of the
    enveloped weight the floor dropped.

    The centre is the first atom of the result, so that ``e.coords[0]`` is the origin and
    ``e.atom_types[0]`` is the type of the atom the sample belongs to.  Which atom of the parent
    molecule every other atom came from is kept in ``member_atom_map``, which is otherwise used
    for the members of an ensemble and is inert here, so that a per-atom quantity computed on an
    environment can be put back on the molecule it came from.

    An ensemble field may be passed, and `atom` then indexes its concatenated atom list.  The
    result is a plain field rather than an ensemble, because the members' weights are already
    multiplied into the atoms they belong to; the atom decomposition of such an environment is
    therefore per atom of a member, which is what a difference field (:func:`difference_field`)
    means.
    """
    i = int(atom)
    n = mol.n_atoms
    if not 0 <= i < n:
        raise ValueError(f"{mol.name or 'a molecule'} has {n} atoms, so atom {i} is not one of "
                         f"them")
    g = envelope(mol.coords, mol.coords[i], rho)
    keep = np.where(g >= float(floor))[0]
    if len(keep) == 0:                       # cannot happen: the centre's own factor is 1
        keep = np.array([i], int)
    keep = np.concatenate([[i], keep[keep != i]])      # the centre first
    coords = np.asarray(mol.coords, float)[keep] - np.asarray(mol.coords, float)[i]
    weights = np.asarray(mol.weights, float)[keep] * g[keep][:, None]
    radii = np.asarray(mol.radii, float)[keep]
    charges = (np.asarray(mol.charges, float)[keep] if mol.charges is not None
               and len(mol.charges) == n else np.zeros(len(keep)))
    back = -np.ones(n, int)
    back[keep] = np.arange(len(keep))
    bonds = (np.asarray(mol.bonds, int).reshape(-1, 2) if mol.bonds is not None
             else np.zeros((0, 2), int))
    if len(bonds):
        ok = (back[bonds[:, 0]] >= 0) & (back[bonds[:, 1]] >= 0)
        bonds = np.stack([back[bonds[ok, 0]], back[bonds[ok, 1]]], axis=1)
    return MoleculeFields(
        name=name or f"{mol.name or 'molecule'}:{i}",
        coords=coords, weights=weights, radii=radii,
        symbols=[mol.symbols[k] for k in keep],
        atom_types=[mol.atom_types[k] for k in keep],
        bonds=bonds, charges=charges, member_atom_map=keep.astype(int),
        charge_source=mol.charge_source)


def kept_share(mol: MoleculeFields, atom: int, rho: float = DEFAULT_RHO,
               floor: float = ENVELOPE_FLOOR) -> float:
    """The share of the enveloped field weight that the floor keeps, in [0, 1]."""
    g = envelope(mol.coords, mol.coords[int(atom)], rho)
    w = np.abs(np.asarray(mol.weights, float)).sum(axis=1) * g
    total = float(w.sum())
    if total <= 0:
        return 1.0
    return float(w[g >= float(floor)].sum() / total)


def environments(mols: Sequence[MoleculeFields], targets: AtomTargets,
                 rho: float = DEFAULT_RHO, floor: float = ENVELOPE_FLOOR,
                 progress=None) -> List[MoleculeFields]:
    """One environment per measured atom, in the row order of `targets`."""
    out: List[MoleculeFields] = []
    for r in range(len(targets)):
        k, i = int(targets.molecule[r]), int(targets.atom[r])
        out.append(environment(mols[k], i, rho, floor))
        if progress is not None and r % 200 == 0:
            progress(f"Environments {r + 1}/{len(targets)}", r / max(len(targets), 1))
    return out


def difference_field(a: MoleculeFields, b: MoleculeFields, name: str = "") -> MoleculeFields:
    """The field of `a` minus the field of `b`, as one field (plan item LP2).

    A kernel on CMF fields is bilinear, so a difference of two fields is again a field: its atoms
    are the atoms of both molecules, with the second molecule's weights negated.  That is the
    signed counterpart of :func:`cmfqsar.fields.combine_members`, which averages members and so
    needs non-negative weights.

    The acidity of a site is the free energy of a change -- the acid loses a proton -- and the
    difference field of the acid and its anion is that change as a field, which is why it is the
    sample for a pKa.  The two molecules must be in one frame for the difference to mean anything;
    the protonation states built by :mod:`cmfqsar.protonation` are, because they are built on the
    geometry of the parent.
    """
    if a.weights.shape[1] != b.weights.shape[1]:
        raise ValueError(f"{a.weights.shape[1]} fields against {b.weights.shape[1]}: a difference "
                         f"field needs both molecules built from one field set")
    na = a.n_atoms
    bonds_a = np.asarray(a.bonds, int).reshape(-1, 2)
    bonds_b = np.asarray(b.bonds, int).reshape(-1, 2) + na

    def _charges(m: MoleculeFields) -> np.ndarray:
        q = np.asarray(m.charges, float).reshape(-1) if m.charges is not None else np.zeros(0)
        return q if len(q) == m.n_atoms else np.zeros(m.n_atoms)

    return MoleculeFields(
        name=name or f"{a.name or 'a'}-{b.name or 'b'}",
        coords=np.concatenate([np.asarray(a.coords, float), np.asarray(b.coords, float)]),
        weights=np.concatenate([np.asarray(a.weights, float), -np.asarray(b.weights, float)]),
        radii=np.concatenate([np.asarray(a.radii, float), np.asarray(b.radii, float)]),
        symbols=list(a.symbols) + list(b.symbols),
        atom_types=list(a.atom_types) + list(b.atom_types),
        bonds=np.concatenate([bonds_a, bonds_b]) if len(bonds_a) or len(bonds_b)
        else np.zeros((0, 2), int),
        charges=np.concatenate([_charges(a), _charges(b)]),
        charge_source=a.charge_source)


# --------------------------------------------------------------------------- #
# The invariant local kernel
# --------------------------------------------------------------------------- #


def local_features(envs: Sequence[MoleculeFields], spec: AutocorrelationSpec = LOCAL_SPEC,
                   progress=None) -> np.ndarray:
    """Autocorrelation features of a list of environments, one row each."""
    return feature_matrix(list(envs), spec, progress=progress)


def local_kernel(train: np.ndarray, test: Optional[np.ndarray] = None,
                 scale: Optional[np.ndarray] = None, spec: AutocorrelationSpec = LOCAL_SPEC,
                 normalize: bool = True) -> np.ndarray:
    """Cosine-normalised linear kernel on the environment autocorrelations."""
    return autocorrelation_kernel(train, test, scale=scale, n_bins=spec.n_bins,
                                  normalize=normalize)


# --------------------------------------------------------------------------- #
# The reference under rule 1: an atom-centred count
# --------------------------------------------------------------------------- #


def types_of(mols: Sequence[MoleculeFields], min_atoms: int = 20) -> List[str]:
    """The atom types worth a column of their own: those on at least `min_atoms` atoms."""
    counts: Dict[str, int] = {}
    for m in mols:
        for t in m.atom_types:
            counts[t] = counts.get(t, 0) + 1
    return sorted(t for t, c in counts.items() if c >= int(min_atoms))


def type_index(types: Sequence[str]) -> Dict[str, int]:
    """Where each type sits in a count row; everything else shares the last column."""
    return {t: i for i, t in enumerate(types)}


def shell_counts(mol: MoleculeFields, atom: int, types: Sequence[str],
                 edges: Sequence[float] = SHELL_EDGES) -> np.ndarray:
    """What is around one atom: its own type, and the atom types in distance shells about it.

    This is the reference the local kernel has to beat (rule 1 of the plan): an atom-centred
    increment scheme, written as counts.  The row is the one-hot type of the centre followed by
    one count per (type, shell), with a last column in each block for every type too rare to have
    one of its own.
    """
    idx = type_index(types)
    n_t = len(types) + 1
    shells = list(edges)
    i = int(atom)
    out = np.zeros(n_t * (1 + len(shells)))
    out[idx.get(mol.atom_types[i], len(types))] = 1.0
    d = np.linalg.norm(np.asarray(mol.coords, float) - np.asarray(mol.coords, float)[i], axis=1)
    for k in range(mol.n_atoms):
        if k == i:
            continue
        s = int(np.searchsorted(shells, d[k], side="left"))
        if s >= len(shells):
            continue
        out[n_t * (1 + s) + idx.get(mol.atom_types[k], len(types))] += 1.0
    return out


def count_features(mols: Sequence[MoleculeFields], targets: AtomTargets,
                   types: Sequence[str], edges: Sequence[float] = SHELL_EDGES) -> np.ndarray:
    """Count rows of the reference, one per measured atom, in the row order of `targets`."""
    n_t = len(types) + 1
    out = np.zeros((len(targets), n_t * (1 + len(list(edges)))))
    for r in range(len(targets)):
        out[r] = shell_counts(mols[int(targets.molecule[r])], int(targets.atom[r]), types, edges)
    return out


def count_kernel(train: np.ndarray, test: Optional[np.ndarray] = None,
                 normalize: bool = True) -> np.ndarray:
    """Cosine-normalised linear kernel on the count rows.

    Every column is a count, so there is nothing to rescale between them: unlike the field
    channels, which carry charges in one block and ones in another, the blocks here are already
    in the same units.
    """
    A = np.asarray(train, float)
    return autocorrelation_kernel(A, test, scale=np.ones(A.shape[1]), normalize=normalize)


# --------------------------------------------------------------------------- #
# Cross-validation over atoms, with the molecules held out whole
# --------------------------------------------------------------------------- #


@dataclass
class LocalResult:
    """What one descriptor scored on the per-atom target, out of fold."""
    name: str
    predicted: np.ndarray
    q2: float
    error: float
    mae: float
    lam: float
    n_samples: int
    n_molecules: int
    folds: int
    note: str = ""
    #: the fold each sample was predicted in, for a paired comparison
    molecule: np.ndarray = _field(default_factory=lambda: np.zeros(0, int))

    def summary(self) -> str:
        return (f"{self.name}: q2 {self.q2:+.3f}, error {self.error:.3f}, "
                f"mean absolute error {self.mae:.3f} over {self.n_samples} atoms in "
                f"{self.n_molecules} molecules, {self.folds} folds, ridge {self.lam:g}"
                + (f" ({self.note})" if self.note else ""))


def cross_validate(K: np.ndarray, targets: AtomTargets, lam: float = 1e-3, folds: int = 10,
                   seed: Optional[int] = 0, name: str = "", note: str = "") -> LocalResult:
    """Out-of-fold predictions of a kernel on a per-atom target, holding out whole molecules.

    The folds come from :meth:`cmfqsar.atomtargets.AtomTargets.folds`, so every atom of a molecule
    is predicted by a model that never saw that molecule.
    """
    y = np.asarray(targets.y, float)
    parts = targets.folds(folds, seed)
    if not parts:
        raise ValueError("there is nothing to cross-validate: no atom carries the target")
    pred = np.full(len(y), np.nan)
    for te in parts:
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(tr) < 2:
            continue
        a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], float(lam))
        pred[te] = K[np.ix_(te, tr)] @ a + b
    ok = np.isfinite(pred)
    return LocalResult(name=name or "kernel", predicted=pred,
                       q2=r2_score(y[ok], pred[ok]), error=rmse(y[ok], pred[ok]),
                       mae=float(np.mean(np.abs(y[ok] - pred[ok]))), lam=float(lam),
                       n_samples=int(ok.sum()), n_molecules=targets.n_measured_molecules,
                       folds=len(parts), note=note, molecule=np.asarray(targets.molecule, int))


def _ridge_path(Kt: np.ndarray, y: np.ndarray, lams: Sequence[float]):
    """Dual coefficients and bias for every ridge at the cost of one eigendecomposition.

    The bordered system of :func:`cmfqsar.model.krr_fit` splits: with ``x`` and ``z`` the
    solutions of ``(K + lam I) x = y`` and ``(K + lam I) z = 1``, the bias is
    ``b = sum(x) / sum(z)`` and the coefficients are ``a = x - b z``.  Both solutions come from
    one eigendecomposition of the training kernel, so a search over ridge values costs no more
    than a single fit.  The result is the same as solving each system separately; it is only the
    arithmetic that is shared.
    """
    w, V = np.linalg.eigh(np.asarray(Kt, float))
    Vy = V.T @ np.asarray(y, float)
    V1 = V.T @ np.ones(len(y))
    out = []
    for lam in lams:
        d = w + float(lam)
        d = np.where(np.abs(d) < 1e-300, 1e-300, d)
        x = V @ (Vy / d)
        z = V @ (V1 / d)
        s = float(z.sum())
        b = float(x.sum() / s) if abs(s) > 1e-300 else 0.0
        out.append((x - b * z, b))
    return out


def cross_validate_lams(K: np.ndarray, targets: AtomTargets, lams: Sequence[float] = LAMS,
                        folds: int = 10, seed: Optional[int] = 0,
                        name: str = "") -> List[LocalResult]:
    """One result per ridge value, from one pass over the folds."""
    y = np.asarray(targets.y, float)
    parts = targets.folds(folds, seed)
    if not parts:
        raise ValueError("there is nothing to cross-validate: no atom carries the target")
    lams = list(lams)
    pred = [np.full(len(y), np.nan) for _ in lams]
    for te in parts:
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(tr) < 2:
            continue
        for j, (a, b) in enumerate(_ridge_path(K[np.ix_(tr, tr)], y[tr], lams)):
            pred[j][te] = K[np.ix_(te, tr)] @ a + b
    out = []
    for j, lam in enumerate(lams):
        ok = np.isfinite(pred[j])
        out.append(LocalResult(
            name=name or "kernel", predicted=pred[j], q2=r2_score(y[ok], pred[j][ok]),
            error=rmse(y[ok], pred[j][ok]), mae=float(np.mean(np.abs(y[ok] - pred[j][ok]))),
            lam=float(lam), n_samples=int(ok.sum()), n_molecules=targets.n_measured_molecules,
            folds=len(parts), molecule=np.asarray(targets.molecule, int)))
    return out


def choose_lam(K: np.ndarray, targets: AtomTargets, lams: Sequence[float] = LAMS,
               folds: int = 10, seed: Optional[int] = 0, name: str = "") -> LocalResult:
    """The ridge that cross-validates best, and its result.

    Both the local kernel and the count reference are given this same search on the same folds, so
    the choice favours neither.  The q2 it reports is the one the search maximised and is
    therefore optimistic by the amount any such search is; what is being compared is a difference
    between two descriptors treated alike, not an absolute figure.
    """
    results = cross_validate_lams(K, targets, lams, folds, seed, name=name)
    return max(results, key=lambda r: r.q2)


def paired_interval(a: LocalResult, b: LocalResult, targets: AtomTargets, runs: int = 2000,
                    seed: int = 0, statistic: str = "q2") -> Tuple[float, float, float]:
    """(difference, low, high) for `a` against `b`, bootstrapped over **molecules**.

    Resampling atoms would overstate the precision, because the atoms of one molecule are not
    independent: they share a geometry, a charge model and whatever the molecule's chemistry does
    to all of them.  So a bootstrap sample is a sample of molecules, with every measured atom of a
    drawn molecule going in with it.  Rule 4 of the plan then applies as it does to molecules: a
    difference inside the interval is not a result.
    """
    y = np.asarray(targets.y, float)
    mol = np.asarray(targets.molecule, int)
    which = np.unique(mol)
    rows = {int(k): np.where(mol == int(k))[0] for k in which}

    def stat(idx: np.ndarray, pred: np.ndarray) -> float:
        if statistic == "q2":
            return r2_score(y[idx], pred[idx])
        return -rmse(y[idx], pred[idx])

    point = stat(np.arange(len(y)), a.predicted) - stat(np.arange(len(y)), b.predicted)
    if len(which) < 4:
        return (point, float("nan"), float("nan"))
    rng = np.random.RandomState(int(seed))
    out = []
    for _ in range(int(runs)):
        drawn = which[rng.randint(0, len(which), len(which))]
        idx = np.concatenate([rows[int(k)] for k in drawn])
        if np.std(y[idx]) <= 0:
            continue
        out.append(stat(idx, a.predicted) - stat(idx, b.predicted))
    if not out:
        return (point, float("nan"), float("nan"))
    return (point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))
