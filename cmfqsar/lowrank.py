"""Low-rank kernels for large training sets (plan item B6).

The exact model holds one n by n matrix per field and factorises the n by n joint kernel.  On a
32 GB machine that stops at ten to twenty thousand molecules, and memory is the limit rather
than time.  Plan item B6 assessed the ways out and measured one of them; this module implements
it.

**The Nystrom approximation.**  Choose m landmarks Z -- a grid of Gaussian centres in the common
frame, or a subset of the training molecules -- and replace every field kernel by

    K(X, X) ~ K(X, Z) K(Z, Z)^+ K(Z, X).

That is the Gram matrix of the **features** ``Phi = K(X, Z) W`` with ``W = V diag(w)^(-1/2)``
from the eigendecomposition of K(Z, Z), so the model can be fitted in the space of the m columns
instead of the space of the n molecules.  Nothing new has to be computed: ``K(X, Z)`` is what
:func:`cmfqsar.kernels.kernel_matrices` already produces and what :mod:`cmfqsar.screen` already
streams a block at a time.

What it costs and what it saves::

    exact      n*n per field, n^3 to factorise      five fields of 10 000 molecules: 4 GB
    low rank   n*m per field, m^3 to factorise      the same with 300 landmarks: 0.12 GB

and the n by m features need never be held at once: :func:`gram` accumulates ``Phi^T Phi`` and
``Phi^T y`` over blocks of molecules, so the memory of a fit is m by m whatever n is.

**What stays exact.**  The self-kernel K(x, x) of a molecule is one molecule's own integral, so
it costs nothing and is never approximated here.  That matters twice: the cosine normalisation
divides by the exact diagonal, and the posterior variance keeps the exact ``k(x, x)``, which is
what stops it collapsing far from the landmarks.  Writing the variance out,

    var(x) = k(x,x) - phi^T phi + lam * phi^T (Phi^T Phi + lam I)^-1 phi,

which is the exact Gaussian-process variance with K replaced by its Nystrom approximation, and
is the deterministic-training-conditional form of the literature.

**What this module is not.**  The approximation is only as good as the landmarks, and plan item
B6 measured where that fails: a grid of Gaussian centres spans the field while its spacing is not
much larger than the field's own width, so at an attenuation of 2 or more a Sutherland-sized
frame would need tens of thousands of centres.  Landmark molecules have no width problem.
:func:`gaussian_centres` and :func:`landmark_molecules` build the two kinds and
`benchmarks/run_lowrank.py` measures what each costs in accuracy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as _field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .fields import AtomTable, FieldSet
from .kernels import kernel_matrices, normalize_kernels, self_kernels

#: eigenvalues of K(Z, Z) below this share of the largest are dropped from the feature map
RCOND = 1e-12
#: the radius a one-atom landmark is given, in angstroms; the steric field needs one
CENTRE_RADIUS = 1.7
#: molecules per block when the features are streamed
BLOCK = 256


# --------------------------------------------------------------------------- #
# The landmarks
# --------------------------------------------------------------------------- #


def gaussian_centres(table: AtomTable, spacing: float, n_fields: int,
                     radius: float = CENTRE_RADIUS) -> AtomTable:
    """A cubic grid of one-atom landmarks of unit weight covering the atoms of `table`.

    A centre further than one spacing from every atom carries nothing and is left out.  The rank
    such a basis needs depends on the frame and the width of the field, not on how many
    molecules there are, which is the property that makes the approximation worth having.
    """
    from scipy.spatial import cKDTree

    coords = np.asarray(table.coords, float)
    lo = coords.min(axis=0) - 0.5 * float(spacing)
    hi = coords.max(axis=0) + 0.5 * float(spacing)
    axes = [np.arange(lo[k], hi[k] + 1e-9, float(spacing)) for k in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    d, _ = cKDTree(coords).query(pts, k=1)
    pts = pts[d <= float(spacing)]
    m = len(pts)
    return AtomTable(coords=pts, weights=np.ones((m, int(n_fields))),
                     radii=np.full(m, float(radius)), offsets=np.arange(m + 1),
                     names=[f"centre{i}" for i in range(m)], molecules=[])


def landmark_molecules(table: AtomTable, n: int, seed: Optional[int] = 0
                       ) -> Tuple[AtomTable, np.ndarray]:
    """`n` of the molecules themselves as landmarks, drawn at random without replacement.

    Molecules have no width problem: whatever the attenuation, a molecule's own field is in the
    span of the molecules.  The price is that the landmarks are tied to the training set.
    """
    total = table.n_mol
    k = int(min(max(int(n), 1), total))
    rng = np.random.RandomState(seed if seed is not None else 0)
    rows = np.sort(rng.choice(total, k, replace=False))
    return table.subset(rows), rows


def landmark_atoms(table: AtomTable, n_fields: int, molecules: int = 0,
                   min_distance: float = 0.75, seed: Optional[int] = 0,
                   max_landmarks: int = 4000) -> AtomTable:
    """The atoms of a subset of molecules, each one a landmark where it stands.

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
    """
    source = table
    if molecules:
        source = landmark_molecules(table, int(molecules), seed=seed)[0]
    coords = np.asarray(source.coords, float)
    radii = np.asarray(source.radii, float)
    order = (np.arange(len(coords)) if seed is None
             else np.random.RandomState(seed).permutation(len(coords)))
    keep: List[int] = []
    for i in order:
        if len(keep) >= int(max_landmarks):
            break
        if keep and float(min_distance) > 0:
            d = np.linalg.norm(coords[keep] - coords[i], axis=1)
            if float(d.min()) < float(min_distance):
                continue
        keep.append(int(i))
    keep = sorted(keep)
    pts = coords[keep]
    m = len(pts)
    return AtomTable(coords=pts, weights=np.ones((m, int(n_fields))), radii=radii[keep],
                     offsets=np.arange(m + 1), names=[f"atom{i}" for i in range(m)],
                     molecules=[])


# --------------------------------------------------------------------------- #
# The feature map
# --------------------------------------------------------------------------- #


def whitener(K_zz: np.ndarray, rcond: float = RCOND) -> np.ndarray:
    """``W`` with ``W W^T = K(Z, Z)^+``, from a symmetric eigendecomposition.

    The pseudo-inverse is what keeps a landmark set with near-duplicate members from blowing the
    approximation up: an eigenvalue below `rcond` of the largest is dropped rather than divided
    by.
    """
    A = 0.5 * (np.asarray(K_zz, float) + np.asarray(K_zz, float).T)
    w, V = np.linalg.eigh(A)
    keep = w > max(float(w.max()), 0.0) * float(rcond)
    if not keep.any():
        return np.zeros((A.shape[0], 0))
    return V[:, keep] / np.sqrt(w[keep])


@dataclass
class LowRank:
    """The landmarks and the whitening that turn a molecule into a row of features.

    `transform` gives the (molecules, columns) matrix whose inner products are the approximate
    joint kernel, which is all the rest of the model needs.
    """
    landmarks: AtomTable
    specs: Sequence[object]
    whiteners: List[np.ndarray]
    weights: np.ndarray                       # h_f c_f, one per field
    normalize: bool = True
    self_landmarks: Optional[np.ndarray] = None
    kind: str = "landmarks"

    @property
    def rank(self) -> int:
        return int(sum(w.shape[1] for w in self.whiteners))

    def transform(self, table: AtomTable, block: int = BLOCK, progress=None) -> np.ndarray:
        """The feature rows of every molecule of `table`, a block of molecules at a time."""
        out = np.zeros((table.n_mol, self.rank))
        for lo, hi, part in _blocks(table, block):
            out[lo:hi] = self._block(part)
            if progress is not None:
                progress(f"Features {hi}/{table.n_mol}", hi / max(table.n_mol, 1))
        return out

    def _block(self, part: AtomTable) -> np.ndarray:
        K = np.asarray(kernel_matrices(part, self.landmarks, self.specs, symmetric=False), float)
        if self.normalize:
            K = normalize_kernels(K, self._self(part), self.self_landmarks)
        cols = []
        for f, W in enumerate(self.whiteners):
            cols.append(np.sqrt(max(float(self.weights[f]), 0.0)) * (K[f] @ W))
        return np.hstack(cols) if cols else np.zeros((part.n_mol, 0))

    def _self(self, part: AtomTable) -> np.ndarray:
        return np.asarray(self_kernels(part, self.specs), float)

    def joint_self(self, table: AtomTable, block: int = BLOCK) -> np.ndarray:
        """The exact joint self-kernel of every molecule, ``sum_f h_f c_f k_f(x, x)``.

        This is what the variance keeps exact (plan item MS6): with it the deviation of a
        molecule the landmarks do not span stays large instead of falling to zero. Under the
        cosine normalisation a self-kernel is one, except for a field that is identically zero
        for that molecule (no donor, no acceptor, no charge), which the normalisation leaves at
        zero -- the same convention as :meth:`cmfqsar.model.CMFModel.joint_self`.
        """
        out = np.zeros(table.n_mol)
        for lo, hi, part in _blocks(table, block):
            d = self._self(part)
            if self.normalize:
                d = np.where(d > 1e-300, 1.0, 0.0)
            out[lo:hi] = (d.T * np.asarray(self.weights, float)[:len(self.specs)]).sum(axis=1)
        return out


def _blocks(table: AtomTable, block: int):
    n = table.n_mol
    step = max(int(block), 1)
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        yield lo, hi, table.subset(np.arange(lo, hi))


def build(landmarks: AtomTable, fieldset: FieldSet, scale: Optional[Sequence[float]] = None,
          normalize: bool = True, rcond: float = RCOND, kind: str = "landmarks") -> LowRank:
    """The feature map of one landmark set, ready to transform any molecule in the same frame."""
    specs = [s for s in fieldset.specs if s.enabled]
    K_zz = np.asarray(kernel_matrices(landmarks, landmarks, specs, symmetric=True), float)
    # the self-kernel of a landmark is the diagonal of the block that was just built, which is
    # also the only way to have it for a table of one-atom centres: those carry no molecules
    d_z = np.stack([np.diag(K_zz[f]).copy() for f in range(K_zz.shape[0])])
    if normalize:
        K_zz = normalize_kernels(K_zz, d_z, d_z)
    h = np.asarray(fieldset.mixing_vector(), float)
    s = np.ones(len(specs)) if scale is None else np.asarray(scale, float)
    return LowRank(landmarks=landmarks, specs=specs,
                   whiteners=[whitener(K_zz[f], rcond) for f in range(len(specs))],
                   weights=h[:len(specs)] * s[:len(specs)], normalize=normalize,
                   self_landmarks=d_z, kind=kind)


# --------------------------------------------------------------------------- #
# The fit, in the space of the landmarks
# --------------------------------------------------------------------------- #


@dataclass
class Gram:
    """What a fit needs from the whole training set, in m by m memory."""
    PtP: np.ndarray
    Pty: np.ndarray
    Pt1: np.ndarray
    n: int
    sum_y: float
    yty: float


def gram(low: LowRank, table: AtomTable, y: Sequence[float], block: int = BLOCK,
         progress=None) -> Gram:
    """Accumulate ``Phi^T Phi``, ``Phi^T y`` and ``Phi^T 1`` over blocks of molecules.

    The features of the whole set are never held at once, which is the point of the item: the
    memory of a fit is the rank squared, whatever the number of molecules.
    """
    y = np.asarray(y, float).ravel()
    m = low.rank
    PtP = np.zeros((m, m))
    Pty = np.zeros(m)
    Pt1 = np.zeros(m)
    total = 0.0
    yty = 0.0
    for lo, hi, part in _blocks(table, block):
        Phi = low._block(part)
        PtP += Phi.T @ Phi
        Pty += Phi.T @ y[lo:hi]
        Pt1 += Phi.sum(axis=0)
        total += float(y[lo:hi].sum())
        yty += float(y[lo:hi] @ y[lo:hi])
        if progress is not None:
            progress(f"Gram {hi}/{table.n_mol}", hi / max(table.n_mol, 1))
    return Gram(PtP=PtP, Pty=Pty, Pt1=Pt1, n=table.n_mol, sum_y=total, yty=yty)


def fit(g: Gram, lam: float, fit_bias: bool = True) -> Tuple[np.ndarray, float]:
    """(the weights over the features, the bias) of the ridge with this Gram.

    The dual ridge ``(K + lam I) a = y`` with ``K = Phi Phi^T`` is the primal ridge
    ``(Phi^T Phi + lam I) w = Phi^T y`` with ``w = Phi^T a``: the same model, m by m instead of
    n by n.  The bias is the constant the bordered system of the exact solver fits, which here
    is one more row and column.
    """
    m = len(g.Pty)
    A = g.PtP + float(lam) * np.eye(m)
    if not fit_bias:
        return _solve(A, g.Pty), 0.0
    # [ Phi^T Phi + lam I   Phi^T 1 ] [ w ]   [ Phi^T y ]
    # [ 1^T Phi             n       ] [ b ] = [ sum y   ]
    B = np.zeros((m + 1, m + 1))
    B[:m, :m] = A
    B[:m, m] = g.Pt1
    B[m, :m] = g.Pt1
    B[m, m] = float(g.n)
    rhs = np.concatenate([g.Pty, [g.sum_y]])
    sol = _solve(B, rhs)
    return sol[:m], float(sol[m])


def _solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def predict(Phi: np.ndarray, w: np.ndarray, b: float = 0.0) -> np.ndarray:
    return np.asarray(Phi, float) @ np.asarray(w, float) + float(b)


def variance(g: Gram, Phi: np.ndarray, lam: float,
             k_self: Optional[Sequence[float]] = None) -> np.ndarray:
    """The posterior variance of every row of `Phi`, with the exact self-kernel kept.

    ``var = k(x,x) - phi^T phi + lam phi^T (Phi^T Phi + lam I)^-1 phi``.  The first two terms
    are what the landmarks do not span, and they are why this does not fall to zero far from
    them; with `k_self` left out they are dropped and what is left is the plain subset-of-
    regressors variance, which does.
    """
    P = np.asarray(Phi, float)
    A = g.PtP + float(lam) * np.eye(P.shape[1])
    try:
        L = np.linalg.cholesky(A)
        v = np.linalg.solve(L, P.T)
        quad = float(lam) * np.sum(v * v, axis=0)
    except np.linalg.LinAlgError:
        quad = float(lam) * np.einsum("ij,jk,ik->i", P, np.linalg.pinv(A), P)
    if k_self is None:
        return np.maximum(quad, 0.0)
    gap = np.asarray(k_self, float) - np.sum(P * P, axis=1)
    return np.maximum(gap + quad, 0.0)


def log_evidence(g: Gram, lam: float, fit_bias: bool = True, profile: bool = True) -> float:
    """The log marginal likelihood through the Woodbury identity.

    The same quantity :func:`cmfqsar.evidence.log_evidence` computes from the n by n kernel,
    with ``(Phi Phi^T + lam I)^-1 = (I - Phi (Phi^T Phi + lam I)^-1 Phi^T) / lam`` and
    ``log|Phi Phi^T + lam I| = log|Phi^T Phi + lam I| + (n - m) log lam``, so it costs m^3 and
    not n^3.  The conventions -- the constant mean marginalised, the amplitude profiled out --
    are the exact function's, so the two values may be compared.
    """
    n, m = int(g.n), len(g.Pty)
    if n < 3:
        return float("nan")
    lam = float(lam)
    A = g.PtP + lam * np.eye(m)
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return float("-inf")
    log_det = (2.0 * float(np.sum(np.log(np.diag(L)))) - m * math.log(lam)
               + n * math.log(lam))

    if fit_bias:
        # 1^T A^-1 1, 1^T A^-1 y and y^T A^-1 y, each through the same identity
        denom = _cross(L, g.Pt1, g.Pt1, float(n), lam)
        if denom <= 0:
            return float("-inf")
        one_y = _cross(L, g.Pt1, g.Pty, float(g.sum_y), lam)
        y_y = _cross(L, g.Pty, g.Pty, float(g.yty), lam)
        mean = one_y / denom
        quad = y_y - 2.0 * mean * one_y + mean * mean * denom
        dof = n - 1
        extra = -0.5 * math.log(denom)
    else:
        quad = _cross(L, g.Pty, g.Pty, float(g.yty), lam)
        dof = n
        extra = 0.0
    if not np.isfinite(quad) or quad <= 0:
        return float("-inf")
    if profile:
        return float(-0.5 * dof * (math.log(quad / dof) + 1.0)
                     - 0.5 * log_det + extra - 0.5 * dof * math.log(2.0 * math.pi))
    return float(-0.5 * quad - 0.5 * log_det + extra - 0.5 * dof * math.log(2.0 * math.pi))


def _cross(L: np.ndarray, Pta: np.ndarray, Ptb: np.ndarray, ab: float, lam: float) -> float:
    """a^T (Phi Phi^T + lam I)^-1 b from Phi^T a, Phi^T b and a^T b."""
    va = np.linalg.solve(L, np.asarray(Pta, float))
    vb = np.linalg.solve(L, np.asarray(Ptb, float))
    return float((ab - float(va @ vb)) / lam)


# --------------------------------------------------------------------------- #
# What it costs
# --------------------------------------------------------------------------- #


def memory(n: int, m: int, n_fields: int, exact: bool = False) -> float:
    """Bytes the kernels of a fit need, for the exact form and for this one."""
    if exact:
        return 8.0 * n_fields * n * n + 8.0 * n * n
    return 8.0 * (m * m + m)


def cross_validate(low: LowRank, table: AtomTable, y: Sequence[float], lam: float,
                   folds: int = 10, seed: Optional[int] = 0, fit_bias: bool = True,
                   block: int = BLOCK, with_sd: bool = False):
    """(out-of-fold predictions, the fold of every molecule), the features built once.

    With ``with_sd`` the result is (predictions, standard deviations, folds): each fold's
    training part is read as a :class:`cmfqsar.surrogate.LowRankSurrogate`, so the deviations
    are in the units of ``y`` and are the ones a molecule would have been given had it been new
    (plan item MS6). The folds are those of :func:`cmfqsar.model._kfold_indices`, the same ones
    the exact path uses for the same seed, so the two paths are compared like with like.
    """
    from .model import _kfold_indices

    y = np.asarray(y, float).ravel()
    Phi = low.transform(table, block=block)
    k_self = low.joint_self(table, block=block) if with_sd else None
    folds_of = _kfold_indices(len(y), folds, seed)
    pred = np.zeros(len(y))
    sd = np.full(len(y), np.nan)
    which = np.zeros(len(y), int)
    for k, test in enumerate(folds_of):
        train = np.setdiff1d(np.arange(len(y)), test)
        if not len(test) or not len(train):
            continue
        part = Phi[train]
        g = Gram(PtP=part.T @ part, Pty=part.T @ y[train], Pt1=part.sum(axis=0),
                 n=len(train), sum_y=float(y[train].sum()), yty=float(y[train] @ y[train]))
        if with_sd:
            from .surrogate import LowRankSurrogate
            s = LowRankSurrogate.fit(g, lam, fit_bias)
            pred[test], sd[test] = s.predict(Phi[test], k_self[test])
        else:
            w, b = fit(g, lam, fit_bias)
            pred[test] = predict(Phi[test], w, b)
        which[test] = k
    return (pred, sd, which) if with_sd else (pred, which)


# --------------------------------------------------------------------------- #
# The interpretation, indexed by the landmarks
# --------------------------------------------------------------------------- #


def landmark_coefficients(low: LowRank, w: Sequence[float]) -> List[np.ndarray]:
    """The weight the fitted model puts on every landmark, one array per field.

    The features are ``Phi_f = sqrt(h_f c_f) K(X, Z_f) W_f``, so a fit with weights `w` is the
    same model as a dual fit whose coefficients sit on the landmarks:
    ``sqrt(h_f c_f) W_f w_f``.  Everything the program reads off a co-field -- the map itself,
    the atom contributions, the pharmacophore -- keeps its form with the index set changed from
    training molecules to landmarks, which is what makes the approximation usable and not only
    fast.
    """
    w = np.asarray(w, float).ravel()
    out, start = [], 0
    for f, W in enumerate(low.whiteners):
        stop = start + W.shape[1]
        out.append(np.sqrt(max(float(low.weights[f]), 0.0)) * (W @ w[start:stop]))
        start = stop
    return out


def cofield_gaussians(low: LowRank, w: Sequence[float], f: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centres, coefficients, widths) of the co-field of field `f`, over the landmarks.

    The same three arrays :meth:`cmfqsar.model.CMFModel.cofield_coefficients` returns for an
    exact model, so :func:`cmfqsar.cofields.evaluate_gaussians` draws one as readily as the
    other.  A landmark that is a molecule contributes its own atoms; a landmark that is a
    Gaussian centre contributes itself.
    """
    a = landmark_coefficients(low, w)[int(f)]
    Z = low.landmarks
    spec = low.specs[int(f)]
    coords = np.asarray(Z.coords, float)
    widths = (np.full(len(coords), float(spec.alpha)) if spec.kind == "gauss"
              else float(spec.alpha) / np.asarray(Z.radii, float) ** 2)
    weights = np.asarray(Z.weights, float)[:, int(f)].copy()
    offsets = np.asarray(Z.offsets, int)
    scale = np.ones(len(coords))
    if low.normalize and low.self_landmarks is not None:
        d = np.asarray(low.self_landmarks, float)[int(f)]
        per = 1.0 / np.sqrt(np.maximum(d, 1e-300))
        for j in range(len(offsets) - 1):
            scale[offsets[j]:offsets[j + 1]] = per[j]
    coeffs = np.zeros(len(coords))
    for j in range(len(offsets) - 1):
        coeffs[offsets[j]:offsets[j + 1]] = a[j]
    return coords, coeffs * weights * scale, widths
